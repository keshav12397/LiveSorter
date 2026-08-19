"""
make_sim_session.py
====================

Builds a full synthetic SpikeGLX + Kilosort session with exactly known
ground truth, including known **spatial drift**, so every stage of this
project can be scored against a right answer instead of against another
estimate.

Why this exists rather than the older make_synthetic_test.py
------------------------------------------------------------
`make_synthetic_test.py` makes a 12-channel, 3-unit, 120 s toy for
exercising `generate_filter.py`'s math offline. It is deliberately tiny and
cannot be replayed by SpikeGLX. This script instead produces a real
`.imec0.ap.bin` + `.meta` pair that a SpikeGLX simulation source can stream,
so the *live* pipeline -- fetch loop, preprocessing, GPU detection, sample
accounting -- runs against data whose every spike time we already know.

That matters more here than it sounds. This project has already lost a long
debugging session to a live run that looked like a broken detector and was
actually a misaligned ground truth (see `live_tracking_bug_report.md`), and
the recovery depended on `stream_alignment.py` *estimating* the stream->file
mapping from the data. An estimate that good is still an estimate. With a
session generated here, the drift trajectories, spike times, and syllable
onsets are all exact, so a disagreement between online and offline results
can be attributed to the pipeline rather than argued about.

What it writes (all under --out-dir)
------------------------------------
    sim_g0_t0.imec0.ap.bin   int16[nSamples, 385], 384 AP + 1 SY, time-major
    sim_g0_t0.imec0.ap.meta  SpikeGLX-style metadata for the above
    sim_ks/                  Kilosort-format ground truth, the same files
                             generate_filter.py / calibrate_all_units.py read
                             from a real sort
    sim_truth.npz            everything the .npy files cannot express:
                             per-unit drift trajectories, amplitudes, rates,
                             syllable onsets/codes
    sim_syllables.csv        onset_sample,offset_sample,code -- ground truth
                             for the syllable/decision path
    README.txt               how to point SpikeGLX at this, and the caveats

Syllable codes ride on the IMEC SY channel, not NI
--------------------------------------------------
The obvious home for syllable codes is the NI digital word, lines 5/6/7,
which is where the real rig puts them and what `NiFetchThread` decodes. But
the SpikeGLX instance this was written for can only replay a simulated
**IMEC** file; its NI stream stays on a `Fake_40kHz` source whose digital
word is all zeros. So there is no way to get synthetic syllables into the NI
stream at all.

Putting them in the IMEC SY channel's spare bits instead is not just a
workaround, it is the better test article, for a reason specific to this
project: codes carried in the same file as the spikes are *rigidly* aligned
to them. Any separate NI-side ground truth would have to be re-aligned to
the IMEC stream through exactly the phase-and-slip estimation that caused
the original misdiagnosis, which would put the thing under test back inside
the measuring instrument.

The SY word has room. Per the SpikeGLX manual, "Status bit #6 is the sync
waveform, the other bits are error flags" -- so bit 6 carries the sync square
wave exactly as a real probe does, and bits 0-2 (error flags, always 0 on a
healthy probe and never read by this codebase) carry the 3-bit code. On real
hardware nothing changes: those bits stay 0 and the NI path is unaffected.

Preprocessing realism
---------------------
Units are placed only within the --channel-map-json group (the same 96
channels the real filters are fit against), so an existing config works
unchanged. Channels outside that group carry noise only; since CAR is
computed across the group alone, their content cannot affect any result, and
they are filled with cheaper noise to keep generation time reasonable. That
is a deliberate shortcut, stated here so nobody later reads structure into
the out-of-group channels that was never put there.

Example
-------
    python make_sim_session.py \\
        --out-dir D:/sim_session \\
        --channel-map-json D:/test_newsorter/rawData/shank1only.json \\
        --duration-s 1800 \\
        --n-units 160 --n-drifting 48
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

# --- Fixed acquisition geometry -------------------------------------------
# Matched to the real recording this project works with (NP2013, see
# rawData/*.imec0.ap.meta): 384 AP channels, 0 LF, 1 SY.
N_AP_CHANS = 384
N_SY_CHANS = 1
N_TOTAL_CHANS = N_AP_CHANS + N_SY_CHANS

IMEC_SYNC_BIT = 6          # SpikeGLX/IMEC firmware convention, not a choice
SYLLABLE_BITS = (0, 1, 2)  # SY error-flag bits, unused on a healthy probe

TEMPLATE_LENGTH = 61
TEMPLATE_OFFSET = 20       # samples from waveform start to its negative peak


# ==========================================================================
# Unit population
# ==========================================================================

def draw_amplitude(rng):
    """Per-unit peak amplitude in ADC counts.

    CALIBRATED AGAINST THE REAL RECORDING, not chosen for plausibility. What
    matters is not any single unit's amplitude but the statistics of the
    *summed* multi-unit signal after highpass+CAR, because that sum is the
    background every filter has to reject. Measured on 900k samples of the
    real recording, and on this generator's output:

                        noise (MAD)   |x| p99.9   SNR    max|x|
        real                 6.37         34.7     5.4     1934
        this generator, before:
                            13.68        177.6    12.0      483

    An earlier version drew 18..320, which is roughly 6x too loud. Real data
    is a quiet background with rare large spikes; that version produced a
    dense carpet of large deflections in which 0.1% of ALL samples exceeded
    177, so every unit's filter crossed threshold continuously on other
    units' activity. The visible symptom was baseline precision of ~3% and
    median f1 0.05 across 160 units, against 0.43 on the real session at the
    same unit density (22.7 vs 23.1 units/100 um) and the same population
    rate (1625 vs 1765 Hz). Density and rate were both ruled out by direct
    measurement before amplitude was found; the scale of the summed signal
    was the whole difference.

    The 6..90 body reproduces the real MAD/p99.9/SNR closely (6.90 / 39.2 /
    5.7). The rare loud tail exists because the real recording's max|x| is
    ~300x its noise MAD -- a handful of exceptional units that a body-only
    distribution cannot produce, and that anchor the top of the achievable
    f1 range.
    """
    if rng.random() < 0.05:
        return float(np.exp(rng.uniform(np.log(90.0), np.log(400.0))))
    return float(np.exp(rng.uniform(np.log(6.0), np.log(90.0))))


def draw_rate(rng):
    """Per-unit firing rate, matched to the real session's distribution
    rather than drawn log-uniformly.

    The real 163-cluster session on this probe has per-unit rate percentiles
    p10 0.25, p25 0.55, p50 1.42, p75 6.87, p90 21.0, max 174 Hz -- a
    population overwhelmingly made of very sparse units with a small, very
    busy tail. A log-uniform draw over the same min/max instead puts the
    median at ~4.4 Hz and caps the tail, which sounds like a minor difference
    and is not: it means the total population rate arrives as ~160 units all
    moderately active and spatially interleaved, instead of a few loud units
    plus a lot of quiet ones. Every unit then collides with every other, and
    the collisions are with spatially overlapping neighbours rather than with
    a handful of identifiable loud units that the interferer nulling can
    actually be pointed at.

    Lognormal reproduces the shape closely; the parameters below were fit to
    the percentiles above. The tail is drawn separately because a lognormal
    wide enough to reach 174 Hz would also push the median far too high.
    """
    if rng.random() < 0.08:
        # The busy tail: ~8% of units carry most of the population rate.
        return float(np.exp(rng.uniform(np.log(25.0), np.log(175.0))))
    return float(np.clip(rng.lognormal(mean=np.log(1.5), sigma=1.60), 0.2, 25.0))


def build_units(rng, n_units, n_drifting, group_yc, duration_s):
    """Draw a population of units with position, amplitude, rate, and (for
    a subset) a drift trajectory.

    Amplitudes and rates are drawn log-uniformly rather than uniformly on
    purpose: the whole difficulty this project keeps running into is that
    real populations are dominated by many sparse, low-amplitude units and a
    few loud busy ones, and a uniform draw produces a population where
    nearly everything is easy to detect. The reported per-unit F1 spread on
    the real 157-unit session is wide for exactly this reason, and a test
    set that does not reproduce that spread would not exercise the cases
    that actually fail.
    """
    y_lo, y_hi = float(group_yc.min()), float(group_yc.max())

    units = []
    for i in range(n_units):
        units.append({
            "unit_id": i,
            # Keep units off the very edges of the group so a drifting unit
            # has somewhere to drift to without falling off the array.
            "y0": rng.uniform(y_lo + 30.0, y_hi - 30.0),
            "_duration_s": float(duration_s),
            # ADC counts -- see draw_amplitude() for why the scale matters.
            "amplitude": draw_amplitude(rng),
            # Hz -- see draw_rate() for why this is not a plain log-uniform.
            "rate_hz": draw_rate(rng),
            # Spatial extent of the unit's footprint, in microns -- measured
            # off this project's real Kilosort templates, not guessed.
            #
            # THIS IS THE PARAMETER THAT DECIDES WHETHER THE SESSION IS
            # SEPARABLE AT ALL, and it was the last of three defects found
            # here. Fitting a Gaussian to the spatial decay of each real
            # template about its peak channel gives sigma p10 12, p50 14,
            # p90 22 um -- i.e. 4 channels above 30% of peak (p10 2, p90 8).
            # An earlier version used 18..42 um, giving 14 channels above
            # 30%: 2.2x the sigma and 3.5x the channel coverage.
            #
            # At this probe's real unit density (~23 units per 100 um) that
            # is the difference between a population where a 5-channel filter
            # captures one unit almost completely and its neighbours are
            # largely disjoint, and one where every unit's footprint overlaps
            # every neighbour's and no 5-channel filter can isolate anything.
            # Neither unit density, population rate, waveform diversity, nor
            # amplitude scale explained the resulting precision collapse --
            # all four were measured against the real recording and matched.
            # This did not.
            "sigma_um": float(np.exp(rng.uniform(np.log(10.0), np.log(26.0)))),
            # Waveform shape. Six parameters, spanning enough shape space that
            # units are separable from each other -- see make_temporal_waveform,
            # where getting this too narrow was a real defect with real
            # consequences.
            "width_samples": float(rng.uniform(1.8, 5.5)),
            "rebound_frac": float(rng.uniform(0.10, 1.30)),
            "rebound_delay": float(rng.uniform(4.0, 20.0)),
            "rebound_width": float(rng.uniform(2.5, 11.0)),
            "prepeak_frac": float(rng.uniform(0.0, 0.35)),
            # ~15% of units record positive-going at their peak channel, as a
            # minority of real ones do.
            "polarity": 1.0 if rng.random() > 0.15 else -1.0,
        })

    # --- drift ------------------------------------------------------------
    # Three shapes, because they fail differently and a drift-aware filter
    # that handles one may not handle the others:
    #   ramp  -- monotonic settling of the probe, the common case
    #   osc   -- slow reversible excursion (breathing/pulsation)
    #   jump  -- an abrupt step partway through, the hard case: a filter
    #            adapting on a slow timescale will lag it badly
    drifting = rng.choice(n_units, size=min(n_drifting, n_units), replace=False)
    kinds = ["ramp", "osc", "jump"]
    for u in units:
        u["drift_kind"] = "none"
        u["drift_um"] = 0.0
        u["drift_period_s"] = 0.0
        u["drift_jump_s"] = 0.0
        u["amp_drift_frac"] = 0.0

    for k, idx in enumerate(drifting):
        u = units[int(idx)]
        u["drift_kind"] = kinds[k % len(kinds)]
        # Up to ~45 um, i.e. ~3 rows of a 15 um-pitch probe -- enough to move
        # a unit's peak channel by several channels over the session.
        u["drift_um"] = float(rng.uniform(12.0, 45.0)) * rng.choice([-1.0, 1.0])
        u["drift_period_s"] = float(rng.uniform(0.35, 0.9)) * duration_s
        u["drift_jump_s"] = float(rng.uniform(0.3, 0.7)) * duration_s
        # Amplitude co-varies with position, as it does physically when a
        # unit moves relative to the electrode.
        u["amp_drift_frac"] = float(rng.uniform(0.10, 0.35))

    return units


def unit_offset_um(u, t_s):
    """Position offset (microns) of unit `u` at time(s) `t_s`. Vectorized."""
    t_s = np.asarray(t_s, dtype=np.float64)
    kind = u["drift_kind"]
    if kind == "none":
        return np.zeros_like(t_s)
    if kind == "ramp":
        return u["drift_um"] * (t_s / u["_duration_s"])
    if kind == "osc":
        return u["drift_um"] * np.sin(2.0 * np.pi * t_s / u["drift_period_s"])
    if kind == "jump":
        # Gentle ramp, then a step. `np.where` rather than a hard boolean so
        # this stays vectorized over an array of spike times.
        ramp = 0.3 * u["drift_um"] * (t_s / u["_duration_s"])
        return ramp + np.where(t_s >= u["drift_jump_s"], 0.7 * u["drift_um"], 0.0)
    raise ValueError("unknown drift kind " + kind)


def unit_amp_scale(u, t_s):
    """Amplitude scale factor at time(s) `t_s`, co-varying with drift."""
    if u["drift_kind"] == "none":
        return np.ones_like(np.asarray(t_s, dtype=np.float64))
    off = unit_offset_um(u, t_s)
    denom = abs(u["drift_um"]) if u["drift_um"] != 0 else 1.0
    return 1.0 + u["amp_drift_frac"] * (off / denom)


# ==========================================================================
# Spike trains
# ==========================================================================

def draw_spike_times(rng, rate_hz, duration_s, fs, refractory_s=0.0015):
    """Poisson spike train with an absolute refractory period.

    Refractory enforcement matters for more than realism: the detector's
    windowed non-max suppression uses minSeparationSamples = templateLength/2
    (~1 ms at 30 kHz), so a train with sub-millisecond ISIs would contain
    ground-truth spikes the detector is *designed* to merge, and would score
    as recall loss that is not one.
    """
    n_expected = int(rate_hz * duration_s * 1.3) + 16
    isis = rng.exponential(1.0 / rate_hz, size=n_expected)
    isis = np.maximum(isis, refractory_s)
    t = np.cumsum(isis)
    t = t[t < duration_s]
    return np.sort((t * fs).astype(np.int64))


def make_temporal_waveform(width_samples, rebound_frac, rebound_delay,
                            rebound_width, prepeak_frac, polarity):
    """Extracellular spike shape, parameterized widely enough that different
    units are actually distinguishable from one another.

    THIS IS THE POINT OF THE PARAMETERIZATION, not incidental realism.
    An earlier version varied only `width_samples` and `rebound_frac`, which
    produced templates cross-correlating at median 0.978 (min 0.806) across
    the whole parameter range -- i.e. every unit looked like every other one.
    An LCMV filter fit to unit A then responds almost as strongly to unit B,
    and since the fit nulls only ~5 explicitly chosen interferers, the other
    ~150 units leak straight through. On a 160-unit session that drove
    baseline precision to ~3% and median f1 to 0.05, which reads as a broken
    detector and is entirely a property of the test data.

    Real Kilosort peak-channel templates from this project's own recording
    cross-correlate at median 0.821 with a minimum of 0.000. The five
    parameters here (main-phase width, rebound size, rebound delay, rebound
    width, a small pre-peak hyperpolarization, and polarity) span enough
    shape space to land in that range -- see the self-check at the bottom of
    this module, which asserts it rather than trusting it.

    Polarity is included because a minority of real units genuinely record
    positive-going at their peak channel, and a population that is uniformly
    negative-going is easier to separate than a real one.
    """
    x = np.arange(TEMPLATE_LENGTH, dtype=np.float64)

    main = -np.exp(-0.5 * ((x - TEMPLATE_OFFSET) / width_samples) ** 2)
    rebound = rebound_frac * np.exp(
        -0.5 * ((x - (TEMPLATE_OFFSET + rebound_delay)) / rebound_width) ** 2)
    # Small depolarizing shoulder before the trough -- present in real
    # extracellular waveforms and a cheap source of shape asymmetry, which is
    # what a symmetric-Gaussian-only model lacks.
    prepeak = prepeak_frac * np.exp(
        -0.5 * ((x - (TEMPLATE_OFFSET - 1.8 * width_samples)) / (0.9 * width_samples)) ** 2)

    wf = polarity * (main + rebound + prepeak)
    return (wf / np.max(np.abs(wf))).astype(np.float32)


# ==========================================================================
# Syllables
# ==========================================================================

def build_syllables(rng, duration_s, fs):
    """Syllable bouts: a code from 1..7 held high for 80-320 ms, separated by
    1.5-6 s gaps. Code 0 is 'no syllable' and is never emitted, matching how
    NiFetchThread treats an all-zero code."""
    events = []
    t = rng.uniform(1.0, 4.0)
    while t < duration_s - 1.0:
        dur = rng.uniform(0.08, 0.32)
        if t + dur >= duration_s - 1.0:
            break
        events.append((int(t * fs), int((t + dur) * fs), int(rng.integers(1, 8))))
        t += dur + rng.uniform(1.5, 6.0)
    return events


def build_sy_channel(n_samples, start_sample, fs, sync_period_s, syllables):
    """SY channel values for one block: sync square wave on bit 6, 3-bit
    syllable code on bits 0-2."""
    idx = np.arange(start_sample, start_sample + n_samples, dtype=np.int64)

    period = int(round(sync_period_s * fs))
    sync_high = (idx % period) < (period // 2)

    sy = np.zeros(n_samples, dtype=np.int64)
    sy |= (sync_high.astype(np.int64) << IMEC_SYNC_BIT)

    blk_lo, blk_hi = idx[0], idx[-1] + 1
    for on, off, code in syllables:
        if off <= blk_lo or on >= blk_hi:
            continue
        lo = max(on, blk_lo) - blk_lo
        hi = min(off, blk_hi) - blk_lo
        for b, bit in enumerate(SYLLABLE_BITS):
            if (code >> b) & 1:
                sy[lo:hi] |= (1 << bit)

    return sy.astype(np.int16)


# ==========================================================================
# Main generation
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--channel-map-json", required=True,
                    help="Same JSON the real filters use (e.g. shank1only.json). "
                         "Units are placed only within this channel group, so an "
                         "existing carChannelMapJson config works unchanged.")
    ap.add_argument("--duration-s", type=float, default=1800.0)
    ap.add_argument("--sample-rate", type=float, default=30000.0)
    ap.add_argument("--n-units", type=int, default=160)
    ap.add_argument("--n-drifting", type=int, default=48)
    ap.add_argument("--noise-sigma", type=float, default=6.5,
                    help="Per-channel white noise sigma in ADC counts. The "
                         "default is calibrated so the post-CAR noise MAD "
                         "matches the real recording's 6.37 -- see draw_amplitude().")
    ap.add_argument("--corr-noise-sigma", type=float, default=4.0,
                    help="Amplitude of a shared noise component added across "
                         "the whole group. This is what CAR exists to remove; "
                         "with it at zero, CAR would be untested.")
    ap.add_argument("--sync-period-s", type=float, default=1.0,
                    help="Matches the real meta's syncSourcePeriod=1.")
    ap.add_argument("--block-s", type=float, default=10.0,
                    help="Generation block size. Bounds peak RAM.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="sim_g0_t0")
    args = ap.parse_args()

    fs = args.sample_rate
    n_samples = int(round(args.duration_s * fs))
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    ks_dir = os.path.join(args.out_dir, "sim_ks")
    os.makedirs(ks_dir, exist_ok=True)

    # --- channel group ----------------------------------------------------
    with open(args.channel_map_json) as f:
        cmap = json.load(f)
    group_chans = np.asarray(cmap["chanMap"], dtype=np.int64)
    group_xc = np.asarray(cmap["xc"], dtype=np.float64)
    group_yc = np.asarray(cmap["yc"], dtype=np.float64)
    n_group = len(group_chans)
    if group_chans.max() >= N_AP_CHANS:
        sys.exit(f"channel map references channel {group_chans.max()} but only "
                 f"{N_AP_CHANS} AP channels exist")

    print(f"Channel group: {n_group} channels, ids {group_chans.min()}..{group_chans.max()}, "
          f"y {group_yc.min():.0f}..{group_yc.max():.0f} um")

    # --- population -------------------------------------------------------
    units = build_units(rng, args.n_units, args.n_drifting, group_yc, args.duration_s)

    # --- spike trains -----------------------------------------------------
    print("Drawing spike trains...")
    all_times, all_clusters = [], []
    for u in units:
        t = draw_spike_times(rng, u["rate_hz"], args.duration_s, fs)
        # Drop spikes too close to either edge to be fully stamped, so ground
        # truth never contains a spike whose waveform is truncated in the file.
        t = t[(t >= TEMPLATE_OFFSET + 1) & (t < n_samples - (TEMPLATE_LENGTH - TEMPLATE_OFFSET) - 1)]
        u["spike_times"] = t
        u["n_spikes"] = int(t.size)
        all_times.append(t)
        all_clusters.append(np.full(t.size, u["unit_id"], dtype=np.int64))

    spike_times = np.concatenate(all_times)
    spike_clusters = np.concatenate(all_clusters)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]
    print(f"  {spike_times.size} spikes across {len(units)} units "
          f"({spike_times.size / args.duration_s:.0f} Hz population rate)")

    # --- syllables --------------------------------------------------------
    syllables = build_syllables(rng, args.duration_s, fs)
    print(f"  {len(syllables)} syllable events")

    # --- precompute per-unit temporal waveforms and channel neighbourhoods --
    for u in units:
        u["_wave"] = make_temporal_waveform(
            u["width_samples"], u["rebound_frac"], u["rebound_delay"],
            u["rebound_width"], u["prepeak_frac"], u["polarity"])

    # Assert the population is actually separable before spending several
    # minutes writing tens of GB of it. A population whose templates all look
    # alike produces a session where the detector's precision collapses for
    # reasons that have nothing to do with the detector -- which happened, and
    # cost a full generate-calibrate-diagnose cycle to find. Checking it here
    # is seconds; discovering it downstream is not.
    Wn = np.array([u["_wave"] / np.linalg.norm(u["_wave"]) for u in units])
    C = np.abs(Wn @ Wn.T)
    iu = np.triu_indices(len(units), 1)
    med_corr = float(np.median(C[iu]))
    print(f"  waveform cross-correlation: median {med_corr:.3f}, "
          f"max {C[iu].max():.3f}  (real Kilosort templates: median 0.821)")
    if med_corr > 0.90:
        sys.exit(
            f"Refusing to generate: unit waveforms cross-correlate at median "
            f"{med_corr:.3f}, so units are barely distinguishable from each "
            f"other and every filter will respond to every unit. Widen the "
            f"shape parameter ranges in build_units().")

    # --- write the .bin ---------------------------------------------------
    bin_path = os.path.join(args.out_dir, f"{args.run_name}.imec0.ap.bin")
    block = int(round(args.block_s * fs))
    n_blocks = (n_samples + block - 1) // block
    total_bytes = n_samples * N_TOTAL_CHANS * 2
    print(f"Writing {bin_path} ({total_bytes / 1e9:.1f} GB, {n_blocks} blocks)...")

    t_start = time.time()
    # Spikes are indexed by block so each block only touches its own.
    spike_block = spike_times // block

    with open(bin_path, "wb", buffering=1024 * 1024) as fout:
        for b in range(n_blocks):
            lo = b * block
            hi = min(lo + block, n_samples)
            nb = hi - lo

            # Group channels get real Gaussian noise plus a shared component
            # (the thing CAR removes).
            grp = rng.normal(0.0, args.noise_sigma, size=(nb, n_group)).astype(np.float32)
            if args.corr_noise_sigma > 0:
                shared = rng.normal(0.0, args.corr_noise_sigma, size=(nb, 1)).astype(np.float32)
                grp += shared

            # Stamp this block's spikes, with a margin so a spike starting
            # just before the block still contributes its tail.
            sel = np.flatnonzero((spike_times >= lo - TEMPLATE_LENGTH) & (spike_times < hi))
            for si in sel:
                st = int(spike_times[si])
                u = units[int(spike_clusters[si])]
                t_s = st / fs

                y = u["y0"] + float(unit_offset_um(u, np.array([t_s]))[0])
                amp = u["amplitude"] * float(unit_amp_scale(u, np.array([t_s]))[0])

                d = group_yc - y
                spatial = np.exp(-0.5 * (d / u["sigma_um"]) ** 2)
                near = np.flatnonzero(spatial > 0.02)
                if near.size == 0:
                    continue

                w0 = st - TEMPLATE_OFFSET - lo          # block-relative start
                s0 = max(0, -w0)                         # clip into template
                s1 = min(TEMPLATE_LENGTH, nb - w0)
                if s1 <= s0:
                    continue

                grp[w0 + s0: w0 + s1, near] += np.outer(
                    u["_wave"][s0:s1] * amp, spatial[near]).astype(np.float32)

            # Assemble the full 385-channel frame.
            out = np.empty((nb, N_TOTAL_CHANS), dtype=np.int16)
            # Out-of-group AP channels: cheap integer noise. Nothing reads
            # them (CAR is over the group only) -- see the module docstring.
            out[:, :N_AP_CHANS] = rng.integers(
                -3 * args.noise_sigma, 3 * args.noise_sigma + 1,
                size=(nb, N_AP_CHANS), dtype=np.int16)
            out[:, group_chans] = np.clip(np.rint(grp), -32768, 32767).astype(np.int16)
            out[:, N_AP_CHANS] = build_sy_channel(nb, lo, fs, args.sync_period_s, syllables)

            fout.write(out.tobytes())

            if b % 20 == 0 or b == n_blocks - 1:
                done = (b + 1) / n_blocks
                el = time.time() - t_start
                eta = el / done - el if done > 0 else 0
                print(f"  block {b + 1}/{n_blocks} ({100 * done:.0f}%)  "
                      f"elapsed {el:.0f}s  eta {eta:.0f}s")

    actual = os.path.getsize(bin_path)
    assert actual == total_bytes, f"wrote {actual} bytes, expected {total_bytes}"
    print(f"  wrote {actual} bytes")

    # --- .meta ------------------------------------------------------------
    meta_path = os.path.join(args.out_dir, f"{args.run_name}.imec0.ap.meta")
    write_meta(meta_path, bin_path, args, n_samples, fs, group_chans)
    print(f"Wrote {meta_path}")

    # --- Kilosort-format ground truth -------------------------------------
    write_kilosort(ks_dir, bin_path, spike_times, spike_clusters, units,
                   group_chans, group_xc, group_yc, fs)
    print(f"Wrote {ks_dir}")

    # --- full truth, including drift --------------------------------------
    truth_path = os.path.join(args.out_dir, "sim_truth.npz")
    write_truth(truth_path, units, spike_times, spike_clusters, syllables,
                args, fs, group_chans, group_yc)
    print(f"Wrote {truth_path}")

    syl_path = os.path.join(args.out_dir, "sim_syllables.csv")
    with open(syl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["onset_sample", "offset_sample", "code"])
        w.writerows(syllables)
    print(f"Wrote {syl_path}")

    write_readme(args, bin_path, n_samples, fs, units, syllables)
    print("Done.")


def write_meta(path, bin_path, args, n_samples, fs, group_chans):
    """SpikeGLX-style metadata.

    Only the keys SglxMetaReader.h and the FilterGen loaders actually read
    are guaranteed meaningful; the probe-identity keys are plausible values
    copied from the real NP2013 recording so SpikeGLX recognizes the file
    shape. Anything genuinely inapplicable to a synthetic file is omitted
    rather than filled with a fake value.
    """
    n_bytes = n_samples * N_TOTAL_CHANS * 2
    lines = [
        ("acqApLfSy", f"{N_AP_CHANS},0,{N_SY_CHANS}"),
        ("appVersion", "20251218"),
        ("fileCreateTime", time.strftime("%Y-%m-%dT%H:%M:%S")),
        ("fileName", bin_path.replace("\\", "/")),
        ("fileSHA1", "0"),
        ("fileSizeBytes", str(n_bytes)),
        ("fileTimeSecs", f"{n_samples / fs:.10f}"),
        ("firstSample", "0"),
        ("gateMode", "Immediate"),
        ("imAiRangeMax", "0.62"),
        ("imAiRangeMin", "-0.62"),
        ("imDatPrb_pn", "NP2013"),
        ("imDatPrb_type", "2013"),
        ("imMaxInt", "2048"),
        ("imSampRate", f"{fs:.10f}"),
        ("nSavedChans", str(N_TOTAL_CHANS)),
        ("snsApLfSy", f"{N_AP_CHANS},0,{N_SY_CHANS}"),
        ("snsSaveChanSubset", f"0:{N_TOTAL_CHANS - 1}"),
        ("syncImInputSlot", "6"),
        ("syncSourceIdx", "3"),
        ("syncSourcePeriod", str(args.sync_period_s)),
        ("typeThis", "imec"),
        # Not a SpikeGLX key. Recorded so the SY bit layout this file uses is
        # discoverable from the file itself rather than only from this
        # script's docstring -- see "Syllable codes ride on the SY channel".
        ("simSyllableBits", ",".join(str(b) for b in SYLLABLE_BITS)),
        ("simSyncBit", str(IMEC_SYNC_BIT)),
        ("simGroupChans", f"{group_chans.min()}:{group_chans.max()}"),
    ]
    with open(path, "w", newline="\r\n") as f:
        for k, v in lines:
            f.write(f"{k}={v}\n")


def write_kilosort(ks_dir, bin_path, spike_times, spike_clusters, units,
                   group_chans, group_xc, group_yc, fs):
    """The subset of Kilosort output this project's loaders actually read.

    generate_filter.load_kilosort() reads spike_times.npy and (note)
    spike_templates.npy, not spike_clusters.npy, plus a label TSV. Both
    cluster arrays are written and are identical here, since a synthetic
    session has no merge/split history to distinguish them.
    """
    np.save(os.path.join(ks_dir, "spike_times.npy"), spike_times.astype(np.int64))
    np.save(os.path.join(ks_dir, "spike_templates.npy"), spike_clusters.astype(np.int32))
    np.save(os.path.join(ks_dir, "spike_clusters.npy"), spike_clusters.astype(np.int32))
    np.save(os.path.join(ks_dir, "channel_map.npy"), group_chans.astype(np.int32))
    np.save(os.path.join(ks_dir, "channel_positions.npy"),
            np.column_stack([group_xc, group_yc]).astype(np.float64))
    np.save(os.path.join(ks_dir, "amplitudes.npy"),
            np.ones(spike_times.size, dtype=np.float32))

    with open(os.path.join(ks_dir, "params.py"), "w") as f:
        f.write(f"n_channels_dat = {N_TOTAL_CHANS}\n")
        f.write("offset = 0\n")
        f.write(f"sample_rate = {fs}\n")
        f.write("dtype = 'int16'\n")
        f.write("hp_filtered = False\n")
        f.write(f"dat_path = '{bin_path.replace(chr(92), '/')}'\n")

    def tsv(name, header, values):
        with open(os.path.join(ks_dir, name), "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["cluster_id", header])
            for u, v in zip(units, values):
                w.writerow([u["unit_id"], v])

    # Every unit is a real, well-isolated unit by construction -- there is no
    # noise cluster in a synthetic session, so labelling any of them "mua"
    # would be inventing a distinction the data does not contain.
    tsv("cluster_KSLabel.tsv", "KSLabel", ["good"] * len(units))
    tsv("cluster_group.tsv", "group", ["good"] * len(units))
    tsv("cluster_Amplitude.tsv", "Amplitude", [f"{u['amplitude']:.2f}" for u in units])
    tsv("cluster_ContamPct.tsv", "ContamPct", ["0.0"] * len(units))


def write_truth(path, units, spike_times, spike_clusters, syllables, args, fs,
                group_chans, group_yc):
    """Everything the Kilosort files cannot express -- above all the drift
    trajectories, which are the ground truth for drift-aware filtering.

    Trajectories are stored sampled on a regular 1 s grid rather than as
    parameters, so a consumer scores against positions without having to
    re-implement (and possibly mis-implement) unit_offset_um().
    """
    t_grid = np.arange(0.0, args.duration_s, 1.0)

    pos_um = np.zeros((len(units), t_grid.size), dtype=np.float32)
    amp_scale = np.zeros((len(units), t_grid.size), dtype=np.float32)
    peak_chan = np.zeros((len(units), t_grid.size), dtype=np.int32)
    peak_gidx = np.zeros((len(units), t_grid.size), dtype=np.int32)

    for i, u in enumerate(units):
        y = u["y0"] + unit_offset_um(u, t_grid)
        pos_um[i] = y
        amp_scale[i] = unit_amp_scale(u, t_grid)
        # Peak channel at each time = the group channel physically closest to
        # the unit's centre. This is the quantity a drift-aware filter has to
        # track, so it is precomputed here rather than left as an exercise.
        gi = np.argmin(np.abs(group_yc[None, :] - y[:, None]), axis=1)
        peak_gidx[i] = gi
        peak_chan[i] = group_chans[gi]

    # BOTH are stored, and the distinction is not pedantic. A real channel map
    # stitches together non-contiguous banks: shank1only.json's chanMap steps
    # by +1 for most of its length and then by +49 twice (at group indices 37
    # and 85). So two channels 15 um apart can have SpikeGLX ids 49 apart, and
    # a unit drifting across one of those seams appears to "jump" ~50 channels
    # while physically moving one row.
    #
    # drift_peak_channel is the SpikeGLX id (what channels.bin and the filter
    # files use). drift_peak_group_index is the position within the
    # depth-ordered channel group, in which adjacent means adjacent. Anything
    # reasoning about drift as a *distance* wants the group index or
    # drift_position_um -- never the raw id.

    np.savez_compressed(
        path,
        t_grid_s=t_grid,
        unit_ids=np.array([u["unit_id"] for u in units], dtype=np.int32),
        y0_um=np.array([u["y0"] for u in units], dtype=np.float32),
        amplitude=np.array([u["amplitude"] for u in units], dtype=np.float32),
        rate_hz=np.array([u["rate_hz"] for u in units], dtype=np.float32),
        sigma_um=np.array([u["sigma_um"] for u in units], dtype=np.float32),
        n_spikes=np.array([u["n_spikes"] for u in units], dtype=np.int64),
        drift_kind=np.array([u["drift_kind"] for u in units]),
        drift_um=np.array([u["drift_um"] for u in units], dtype=np.float32),
        drift_position_um=pos_um,
        drift_amp_scale=amp_scale,
        drift_peak_channel=peak_chan,
        drift_peak_group_index=peak_gidx,
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        syllable_onset=np.array([s[0] for s in syllables], dtype=np.int64),
        syllable_offset=np.array([s[1] for s in syllables], dtype=np.int64),
        syllable_code=np.array([s[2] for s in syllables], dtype=np.int32),
        group_chans=group_chans,
        group_yc=group_yc,
        sample_rate=np.float64(fs),
        duration_s=np.float64(args.duration_s),
        sync_period_s=np.float64(args.sync_period_s),
        imec_sync_bit=np.int32(IMEC_SYNC_BIT),
        syllable_bits=np.array(SYLLABLE_BITS, dtype=np.int32),
    )


def write_readme(args, bin_path, n_samples, fs, units, syllables):
    n_drift = sum(1 for u in units if u["drift_kind"] != "none")
    path = os.path.join(args.out_dir, "README.txt")
    with open(path, "w") as f:
        f.write(f"""Synthetic SpikeGLX session -- generated by FilterGen/make_sim_session.py
=======================================================================

  duration      {args.duration_s:.0f} s ({n_samples} samples @ {fs:g} Hz)
  channels      {N_AP_CHANS} AP + {N_SY_CHANS} SY = {N_TOTAL_CHANS}
  units         {len(units)}, of which {n_drift} drift
  syllables     {len(syllables)} events, codes 1-7
  seed          {args.seed}

Replaying this in SpikeGLX
--------------------------
Point the IMEC simulation source at:

    {bin_path}

The channel count and sample rate in the .meta match what the real rig
serves, so an existing ClosedLoop config needs no changes beyond paths.

READ THIS BEFORE COMPARING DETECTIONS TO GROUND TRUTH
-----------------------------------------------------
A replayed stream's sample counter is NOT phase-locked to this file's read
position, and the offset between them drifts within a single pass. Comparing
a live detection's sample_index directly against sim_ks/spike_times.npy will
produce numbers that look like a broken detector and are actually a broken
comparison -- this project has already made that exact mistake once, at the
cost of a long debugging session. See live_tracking_bug_report.md, and go
through FilterGen/stream_alignment.py, which estimates the mapping and
refuses to report metrics when its correlation peak is not significant.

Scoring the OFFLINE path (calibrate_all_units.py against the .bin directly)
has no such problem: there is no stream, so file position is file position.

Syllable codes are on the IMEC SY channel, not NI
-------------------------------------------------
SY bit 6      sync square wave, {args.sync_period_s:g} s period (real IMEC convention)
SY bits 0,1,2 3-bit syllable code, held for the syllable's duration

The NI stream cannot be simulated on this setup, and codes carried in the
same file as the spikes are rigidly aligned to them, which the NI stream
would not be. On real hardware these bits are error flags that stay 0, and
the NI path is unaffected.

What is NOT modelled
--------------------
- Channels outside the --channel-map-json group carry uniform integer noise,
  not realistic neural background. CAR is computed across the group only, so
  nothing reads them; do not read structure into them.
- No electrode drift artifacts beyond unit position/amplitude: no changing
  noise statistics, no channel dropout, no movement artifacts.
- Waveforms are analytic biphasic shapes, not real extracellular templates.
- Every unit is perfectly isolated with zero contamination.
""")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
