"""
motion_correct.py
==================

Motion-corrected target templates: instead of one filter per time segment
(`drift_estimate.py`'s approach) or one filter fit to the whole smeared
recording, build one *sharp* template per segment by spatially registering
every one of the unit's spikes -- not just the ones inside that segment --
to a chosen reference position before averaging.

Why this is worth trying alongside segmentation, not instead of it
--------------------------------------------------------------------
Segmentation (`drift_estimate.segment_from_trajectory`, driving
`calibrate_drift_aware.py`'s 'segmented' mode) fixes the smeared-template
problem by cutting the smear out: each segment only ever averages spikes
seen from close to one position, so each piece is sharp. That works, but it
starves every segment of spikes -- a unit that would give a whole-session
fit N spikes gives a K-segment fit roughly N/K each, which is exactly why
`calibrate_drift_aware.py` needs `--min-spikes-per-segment` at all: a
segment with too few spikes produces a *noisier* template than a mildly
smeared one, so segmenting too finely can lose more than it gains (see that
script's tuning history in this branch's commit log).

Registration keeps every spike. For a chosen reference position, it shifts
each *bin's* mean waveform (not each raw spike -- see below) along the
probe's depth axis by the distance between where the unit was in that bin
and the reference, before folding it into a weighted average. The result is
a template that is sharp at the reference position while still drawing on
the unit's whole spike count, not just whichever segment the reference
falls in. This module changes ONLY how a segment's *target* template is
built; channel selection, the noise covariance, and the LCMV solve are
untouched (see `calibrate_drift_aware.py`'s 'registered' mode, which still
segments for channel reselection -- registration and segmentation are
complementary, not alternatives, for the reason in "What this does not fix"
below).

Per-bin, not per-spike
-----------------------
Registering every individual spike's snippet would need a spike-level
position estimate, and `drift_estimate.unit_trajectory` only estimates
position at bin resolution (~150 spikes) to begin with -- a spike-level
estimate would be estimator noise, not signal. Reusing the bins
`unit_trajectory(..., return_waveforms=True)` already computes means this
module shares the one tuned binning pass instead of a second one, and its
per-bin mean waveform is exactly what `generate_filter.mean_waveform`
already computes for `fit_lcmv` -- no new waveform-averaging code.

Nearest channel, not interpolation
------------------------------------
`remap_indices` reassigns each output channel to the *nearest* group
channel at the shifted depth, rather than interpolating between two
bracketing channels. Interpolation would invent values between physical
contacts, which is its own artifact; at this group's ~15um pitch (see
`drift_estimate.py`'s docstring on `shank1only.json`'s two +49 double-steps
-- distances here always come from `chan_y`, never channel id) nearest-
channel is already fine relative to a unit's multi-channel footprint.

What this does not fix
-----------------------
1. Channel SELECTION is still done at the reference position. If the unit's
   drift carries it off the reference position's best channels before the
   filter is redeployed, no amount of template cleanliness helps -- the
   filter is still reading the wrong five wires. This is exactly why
   `calibrate_drift_aware.py`'s 'registered' mode keeps segmentation's
   channel-reselection machinery and only swaps out how each segment's
   target template is built.
2. Amplitude. Neuropixels signal amplitude falls off with true distance
   from the probe, not just the depth coordinate this module shifts along,
   and `sim_truth.npz`'s `drift_amp_scale` field shows the synthetic
   generator models that fall-off explicitly. A bin registered in from far
   away carries its own, uncorrected amplitude scale into the average --
   this module has no second estimate of that scale to correct it with.
   `decay_um` bounds the damage by down-weighting distant bins rather than
   fixing the scale; see `registered_template`'s docstring.
"""

import numpy as np


def remap_indices(chan_y, shift_um):
    """For each output channel i, the index of the group channel whose
    depth is closest to chan_y[i] + shift_um.

    Convention: a bin recorded while the unit sat at depth p contributes,
    at its physical channel with depth Y, the value shape(Y - p) (signal as
    a function of distance from the unit). To synthesize what a channel at
    depth Y would read if the unit instead sat at depth `ref`, i.e.
    shape(Y - ref), we need the *source* channel at depth
    Y' = Y + (p - ref) -- same distance from the unit's true position p as
    Y is from ref. Calling this with shift_um = p - ref (bin depth minus
    reference depth) gives exactly that Y' for every output channel at once.

    O(n^2) distance table; n is one channel group (<=96 here), called once
    per bin per segment -- not a hot loop, not worth a KD-tree.
    """
    chan_y = np.asarray(chan_y, dtype=np.float64)
    target = chan_y + shift_um
    diffs = np.abs(target[:, None] - chan_y[None, :])
    return np.argmin(diffs, axis=1)


def registered_template(bin_y, bin_waveforms, bin_n, ref_y, chan_y, decay_um=30.0):
    """Weighted average of per-bin mean waveforms, each first remapped to
    look as if recorded with the unit sitting at `ref_y`.

    `bin_y`, `bin_waveforms`, `bin_n`: parallel sequences from
    `drift_estimate.unit_trajectory(..., return_waveforms=True)` -- each
    bin's centroid depth (um), mean waveform (template_length x
    n_group_channels, physical channel order), and spike count. Not
    restricted to any one segment: registration is the point at which
    spikes from OUTSIDE the segment being fit get to contribute.

    Weight is `bin_n * exp(-|bin_y - ref_y| / decay_um)`: more spikes lowers
    template variance and is weighted up; a bin far from `ref_y` is
    downweighted rather than excluded outright, since it still carries real
    shape information at an amplitude scale this module does not correct
    (see module docstring, "What this does not fix" #2). `decay_um` trades
    that bias against variance and should sit on the same order as
    `drift_estimate`'s `tol_um` -- both describe the depth scale over which
    the unit's footprint is still considered "the same".

    Returns (template_length, n_group_channels). Raises ValueError if every
    bin has zero spikes (should not happen -- unit_trajectory only returns
    bins it could estimate a position for).
    """
    n_bins = len(bin_waveforms)
    if n_bins == 0:
        raise ValueError("no bins to register")

    acc = None
    wsum = 0.0
    for i in range(n_bins):
        if bin_n[i] <= 0:
            continue
        shift = float(bin_y[i] - ref_y)
        idx = remap_indices(chan_y, shift)
        registered = bin_waveforms[i][:, idx]
        w = float(bin_n[i]) * np.exp(-abs(shift) / max(decay_um, 1e-9))
        acc = registered * w if acc is None else acc + registered * w
        wsum += w

    if acc is None or wsum <= 0:
        raise ValueError("no usable bins (all zero weight)")
    return acc / wsum


# --------------------------------------------------------------------- #
# Sanity check against a synthetic session's known drift
# --------------------------------------------------------------------- #

def _main():
    """Compares a registered template's cross-correlation against the
    ground-truth *instantaneous* template at a chosen reference time to a
    plain whole-session mean waveform's -- the thing registration is meant
    to beat. Needs sim_truth.npz's exact trajectory, so this is a validation
    tool, not something a real session can run (same caveat as
    drift_estimate.py's __main__)."""
    import argparse
    import generate_filter as gf
    import threshold_sweep_real as tsr
    import drift_estimate as de

    ap = argparse.ArgumentParser(
        description="Sanity-check registered_template() against a plain "
                    "whole-session mean waveform, scored by peak-channel "
                    "SNR at a chosen reference time.")
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--meta-path")
    ap.add_argument("--channel-map-json", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--min-spikes", type=int, default=600)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--decay-um", type=float, default=30.0)
    ap.add_argument("--ref-frac", type=float, default=1.0,
                    help="Reference time as a fraction of the recording "
                         "(1.0 = end, i.e. the deployment-relevant position).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.filter = True
    args.car = True
    args.causal_highpass = True
    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(args, rng, dtype=np.float32)

    positions = gf.load_channel_positions_json(args.channel_map_json)
    chan_y = de.channel_y_for_group(np_ch, positions)

    truth = np.load(args.truth, allow_pickle=True)
    t_uid = list(truth["unit_ids"])
    t_grid = truth["t_grid_s"]
    t_pos = truth["drift_position_um"]
    t_kind = truth["drift_kind"]
    ref_time_s = args.ref_frac * data.shape[0] / fs

    ids, counts = np.unique(spike_cl, return_counts=True)
    order = np.argsort(counts)[::-1]
    cand = [int(ids[i]) for i in order if counts[i] >= args.min_spikes
            and str(t_kind[t_uid.index(int(ids[i]))]) != "none"]
    if args.max_units:
        cand = cand[:args.max_units]

    print(f"{'unit':>5} {'kind':>5} {'true drift':>10} {'plain snr':>10} "
          f"{'reg snr':>10} {'improve':>8}")
    improves = []
    for uid in cand:
        st = spike_t[spike_cl == uid]
        t_c, y_raw, wfs, ns = de.unit_trajectory(
            data, st, chan_y, args.template_length, args.template_offset, fs,
            rng=rng, return_waveforms=True)
        if len(wfs) < 2:
            continue
        i = t_uid.index(uid)
        ref_y = float(np.interp(ref_time_s, t_grid, t_pos[i]))

        reg_wf = registered_template(y_raw, wfs, ns, ref_y, chan_y, args.decay_um)
        plain_wf, _ = gf.mean_waveform(data, st, args.template_length,
                                       args.template_offset, 4000, rng)

        def snr(wf):
            amp = np.ptp(wf, axis=0)
            peak = amp.max()
            floor = np.median(amp)
            return float(peak / max(floor, 1e-9))

        s_plain, s_reg = snr(plain_wf), snr(reg_wf)
        improves.append(s_reg / max(s_plain, 1e-9))
        print(f"{uid:>5} {str(t_kind[i]):>5} {float(np.ptp(t_pos[i])):10.1f} "
              f"{s_plain:10.2f} {s_reg:10.2f} {s_reg / max(s_plain, 1e-9):8.2f}x")

    if improves:
        print(f"\nmedian improvement ratio: {np.median(improves):.2f}x "
              f"(n={len(improves)})")


if __name__ == "__main__":
    _main()
