"""
drift_estimate.py
==================

Per-unit spatial drift: estimate where a unit sat on the probe as a function
of time, and cut the recording into time segments inside which it did not
move enough to matter.

Why this exists
---------------
`fit_lcmv` builds its template from a mean waveform over a unit's spikes. If
the unit moved during the recording, that mean is an average over several
different spatial positions -- a *smeared* template that matches the unit at
no single moment. Two things then go wrong, and the second is much worse
than the first:

1. The template shape is a blur, so the matched filter is mismatched.
2. `select_channels` picks channels for the blur, and `lcmv_filter` is asked
   to hold unity gain on a template that now overlaps a neighbour's template
   far more than the true instantaneous one does. LCMV nulls that neighbour
   exactly, so the null cuts into the target too. The constraint set becomes
   near-collinear, the solve responds by growing the weights, and the
   filter's white-noise gain blows up. Unity gain on the target is still
   satisfied -- the *noise floor* is what moves, up past the target's own
   response. That is why the worst drifting units in the synthetic session
   fail with ~1% precision rather than with a modest, graceful loss.

Everything here is an *estimator*, run against exactly the same preprocessed
recording `calibrate_all_units.py` already builds. It deliberately does not
use `sim_truth.npz`: a real session has no ground truth, and an estimator
that needs one is not deployable. `__main__` scores this estimator against
`sim_truth.npz`'s `drift_position_um`, which is what that file is for.

Metric distances
----------------
Positions here are always microns (from the channel map's `yc`), never
SpikeGLX channel ids. `shank1only.json`'s `chanMap` steps by +1 for most of
its length and by +49 twice, so two channels 15 um apart can have ids 49
apart. Anything reasoning about drift as a distance must use `yc` or the
index within the depth-ordered group.
"""

import numpy as np

import generate_filter as gf


def channel_y_for_group(np_ch, channel_positions):
    """Depth (um) of each channel in `np_ch`, in np_ch's own order.

    `channel_positions` is generate_filter.load_channel_positions_json()'s
    dict of raw channel id -> (x, y). Falls back to the index within the
    group scaled by a nominal 15 um pitch when a channel is missing from the
    map, so a partial map degrades instead of raising -- but note the caller
    is then measuring index distance, not true depth.
    """
    y = np.empty(len(np_ch), dtype=np.float64)
    for i, ch in enumerate(np_ch):
        pos = channel_positions.get(int(ch)) if channel_positions else None
        y[i] = float(pos[1]) if pos is not None else 15.0 * i
    return y


def _position_from_waveform(wf, chan_y, n_top):
    """Amplitude-weighted centroid depth of a mean waveform.

    Peak-to-peak per channel, minus the median across channels (a noise
    floor: with few spikes averaged, every channel has a positive ptp purely
    from noise, and leaving it in drags the centroid toward the middle of
    the group regardless of where the unit is), then the energy-weighted
    centroid over the `n_top` largest channels only. Restricting to the top
    channels matters for the same reason: a unit's real footprint is a
    handful of channels, and letting 96 near-zero channels vote adds bias
    proportional to how far the unit is from the group's centre.
    """
    amp = np.ptp(np.asarray(wf, dtype=np.float64), axis=0)
    amp = np.maximum(amp - np.median(amp), 0.0)
    if not np.any(amp > 0):
        return float(np.nan)
    top = np.argsort(amp)[::-1][:n_top]
    w = amp[top] ** 2
    return float(np.sum(w * chan_y[top]) / np.sum(w))


def unit_trajectory(data, spike_times, chan_y, template_length, template_offset,
                    fs, spikes_per_bin=150, max_bins=60, max_bin_spikes=400,
                    n_top=8, rng=None, smooth=3, return_waveforms=False):
    """Estimate one unit's depth over time.

    Returns (t_center_s, y_um), both 1-D and ascending in time. With
    `return_waveforms=True`, returns (t_center_s, y_um, waveforms, n_spikes)
    instead, where `waveforms[i]` is bin i's own mean waveform
    (template_length x n_group_channels, physical channel order, the exact
    array `generate_filter.mean_waveform` returns) and `n_spikes[i]` is that
    bin's spike count -- both consumed by `motion_correct.registered_template`
    to build a single motion-corrected template without a second binning
    pass. `y_um` there is the *unsmoothed* per-bin position (smoothing is a
    segmentation aid, see below; registration wants the bin's own estimate,
    not its neighbours').

    Bins hold an equal *number of spikes*, not an equal amount of time. A
    unit's position can only be estimated where it fired, and a fixed time
    bin gives a 40 Hz unit 40x the averaging a 1 Hz unit gets. Equal-count
    bins put the same estimator variance in every bin and automatically
    place more of them where the unit was active.
    """
    rng = rng or np.random.default_rng(0)
    st = np.asarray(spike_times)
    st = st[(st > template_offset) &
            (st < data.shape[0] - template_length + template_offset)]
    if st.size == 0:
        empty = (np.zeros(0), np.zeros(0))
        return empty + ([], []) if return_waveforms else empty
    st = np.sort(st)

    n_bins = int(np.clip(st.size // spikes_per_bin, 1, max_bins))
    edges = np.linspace(0, st.size, n_bins + 1).astype(int)

    t_c, y, wfs, ns = [], [], [], []
    for b in range(n_bins):
        bin_st = st[edges[b]:edges[b + 1]]
        if bin_st.size == 0:
            continue
        wf, _ = gf.mean_waveform(data, bin_st, template_length, template_offset,
                                 max_spikes=max_bin_spikes, rng=rng)
        pos = _position_from_waveform(wf, chan_y, n_top)
        if np.isnan(pos):
            continue
        t_c.append(float(np.median(bin_st)) / fs)
        y.append(pos)
        wfs.append(wf)
        ns.append(int(bin_st.size))

    t_c = np.asarray(t_c)
    y_raw = np.asarray(y)
    if return_waveforms:
        return t_c, y_raw, wfs, ns

    return t_c, smooth_trajectory(y_raw, smooth)



def pooled_com_motion(data, spike_times, spike_clusters, chan_y,
                      template_length, template_offset, fs, unit_ids,
                      bin_s=20.0, spikes_per_bin=150, n_top=8,
                      n_windows=1, window_overlap=0.5, min_units=5,
                      rng=None):
    """Common-mode probe motion by pooling per-unit centroid trajectories.

    Returns `(t_center_s, motion, win_centers_um)` with `motion` shaped
    (n_windows, n_time) -- the same contract `dredge_lite.estimate_motion`
    plus `window_centers_um` return, so the two are interchangeable at the
    call site.

    Why this exists alongside dredge_lite
    -------------------------------------
    dredge_lite registers an amplitude RASTER by cross-correlation. This
    tracks each unit's amplitude-weighted centroid depth (the same
    `_position_from_waveform` used everywhere else here), converts each to a
    DISPLACEMENT by subtracting that unit's own mean, and takes the median
    across units at each time. Measured on the same sessions:

                              simulator (truth known)      Lav69 (vs KS4 dshift)
        dredge_lite raster    rmse 1.03 um                 corr 0.845  rmse 1.00
        pooled centroid       rmse 0.10 um, slope 0.998    corr 0.962  rmse 0.85

    plus ~11 s of localisation against a raster build and an O(n_time^2)
    correlation sweep. Three structural reasons it wins, not just three
    better numbers:

      - A centroid is CONTINUOUS in depth. The raster path quantises to a
        depth bin, which is what let the comb bug (see dredge_lite's
        build_raster) return identically zero on a real recording. This
        estimator cannot fail that way; there is no grid to align to.
      - Sub-pitch motion needs no interpolation trick. A unit that moves
        2 um moves its centroid 2 um.
      - Each unit's displacement is its own measurement, so pooling is a
        plain robust average rather than a graph solve.

    Monopolar triangulation was tried in place of the centroid and is not
    worth it: on Lav69 it scored slope 0.585 against the centroid's 0.578
    and corr 0.963 against 0.962, for 31 s of solving against 0.1 s. The
    residual amplitude disagreement with KS4 is NOT a centroid artifact --
    it survives changing the localiser, and the centroid recovers simulator
    truth at slope 0.998, so it is more likely dshift's scale than ours.
    That is an inference, not a measurement: Lav69 has no ground truth.

    MEDIAN, not mean, across units. A minority of units have their centroid
    dominated by one channel and barely move; on the simulator both agree
    (slope 0.998 vs 0.984) but on real data the mean is visibly noisier
    (corr 0.873 vs 0.962). The median is what survives contact with a real
    population.

    With `n_windows > 1` the depth axis is cut into overlapping windows and
    units are pooled within each by their own mean depth -- the nonrigid
    case. Windows overlap for the reason dredge_lite states: a motion field
    with discontinuities is not a thing tissue does, and each window needs
    enough units to average. A window holding fewer than `min_units` falls
    back to the all-unit estimate rather than reporting its own noise.
    """
    rng = rng or np.random.default_rng(0)
    n_samples = data.shape[0]
    n_time = max(int(np.ceil(n_samples / (bin_s * fs))), 2)
    grid = (np.arange(n_time) + 0.5) * (n_samples / n_time) / fs

    disp, depth = [], []
    for uid in unit_ids:
        st = spike_times[spike_clusters == uid]
        if st.size < 4 * spikes_per_bin:
            continue
        t_c, y = unit_trajectory(data, st, chan_y, template_length,
                                 template_offset, fs,
                                 spikes_per_bin=spikes_per_bin,
                                 n_top=n_top, rng=rng, smooth=1)
        ok = np.isfinite(y)
        if ok.sum() < 4:
            continue
        # left/right leave NaN outside a unit's own span on purpose: a unit
        # that stopped firing must abstain there, not extrapolate flat and
        # drag the median toward zero.
        yi = np.interp(grid, t_c[ok], y[ok], left=np.nan, right=np.nan)
        if np.all(np.isnan(yi)):
            continue
        disp.append(yi - np.nanmean(yi))
        depth.append(float(np.nanmean(y[ok])))

    if not disp:
        return grid, np.zeros((max(n_windows, 1), n_time)), np.array([np.mean(chan_y)])

    Y = np.asarray(disp)
    depth = np.asarray(depth)

    def _pool(mask):
        if mask.sum() < min_units:
            mask = np.ones(Y.shape[0], dtype=bool)
        with np.errstate(invalid="ignore"):
            m = np.nanmedian(Y[mask], axis=0)
        # A time bin nobody covered is filled from its neighbours rather than
        # left NaN: downstream consumers evaluate the trajectory at arbitrary
        # times and a NaN there silently poisons a filter.
        bad = ~np.isfinite(m)
        if bad.all():
            return np.zeros(n_time)
        if bad.any():
            m[bad] = np.interp(grid[bad], grid[~bad], m[~bad])
        return m - m.mean()

    if n_windows <= 1:
        return grid, _pool(np.ones(Y.shape[0], dtype=bool))[None, :], \
               np.array([float(np.mean(chan_y))])

    y_lo, y_hi = float(np.min(chan_y)), float(np.max(chan_y))
    width = (y_hi - y_lo) / (1.0 + (n_windows - 1) * (1.0 - window_overlap))
    out = np.zeros((n_windows, n_time))
    centers = np.zeros(n_windows)
    for w in range(n_windows):
        lo = y_lo + w * width * (1.0 - window_overlap)
        hi = lo + width
        centers[w] = 0.5 * (lo + hi)
        out[w] = _pool((depth >= lo) & (depth <= hi))
    return grid, out, centers

def smooth_trajectory(y_um, smooth=3):
    """Running median of a per-bin trajectory, `unit_trajectory`'s own
    smoothing step factored out so `calibrate_drift_aware.py`'s 'registered'
    mode can compute the *same* segment boundaries `segment_from_trajectory`
    would get from a normal `unit_trajectory` call, while separately using
    the unsmoothed per-bin waveforms (`return_waveforms=True`) for
    registration -- one binning pass, two consumers, instead of a second
    smoothing implementation.

    A running median, not a mean: a 'jump' trajectory is a step function,
    and a mean smears the step across the window -- which is precisely the
    boundary the segmenter has to find.
    """
    y_um = np.asarray(y_um)
    if smooth <= 1 or y_um.size < smooth:
        return y_um
    half = smooth // 2
    pad = np.pad(y_um, half, mode="edge")
    return np.array([np.median(pad[i:i + smooth]) for i in range(y_um.size)])


def segment_from_trajectory(t_center_s, y_um, spike_times, fs, n_samples,
                            tol_um=12.0, min_spikes_per_segment=250,
                            max_segments=8):
    """Cut [0, n_samples) into segments inside which the unit is ~stationary.

    Returns a list of (lo_sample, hi_sample), contiguous and covering the
    whole recording.

    `tol_um` is the depth span a single filter is asked to cover. Below
    roughly the probe row pitch there is nothing to gain -- the channel set
    cannot change and the template barely does -- and every extra segment
    costs spikes from the template average, so the default sits just under
    one 15 um row.

    A segment that would hold fewer than `min_spikes_per_segment` spikes is
    merged forward. Too few spikes gives a noisy template, and a noisy
    template is a worse filter than a mildly smeared one; this floor is what
    stops the method from hurting sparse units.
    """
    lo_hi = [(0, int(n_samples))]
    if t_center_s.size < 2:
        return lo_hi
    if float(np.ptp(y_um)) <= tol_um:
        # Not measurably drifting: leave it exactly as calibrate_all_units.py
        # would have fit it. This branch is what keeps static units untouched.
        return lo_hi

    # Greedy left-to-right: extend the current segment while its depth span
    # stays inside tol_um. A step in y closes the segment on the bin right
    # after the step, which is where the boundary belongs.
    bounds = [0]
    start = 0
    for i in range(1, t_center_s.size):
        if float(np.ptp(y_um[start:i + 1])) > tol_um:
            bounds.append(i)
            start = i
    bounds.append(t_center_s.size)

    # Bin index boundaries -> sample boundaries, at the midpoint in time
    # between the last bin of one segment and the first bin of the next.
    edges = [0]
    for b in bounds[1:-1]:
        t_mid = 0.5 * (t_center_s[b - 1] + t_center_s[b])
        edges.append(int(round(t_mid * fs)))
    edges.append(int(n_samples))
    edges = sorted(set(e for e in edges if 0 <= e <= n_samples))

    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    segs = _merge_sparse(segs, spike_times, min_spikes_per_segment)
    segs = _cap_segments(segs, spike_times, max_segments)
    return segs


def _seg_counts(segs, spike_times):
    st = np.sort(np.asarray(spike_times))
    return [int(np.searchsorted(st, hi) - np.searchsorted(st, lo)) for lo, hi in segs]


def _merge_sparse(segs, spike_times, min_spikes):
    """Merge any segment below the spike floor into a neighbour, repeatedly.

    Merges into whichever neighbour is itself smaller in spike count, so the
    result stays as evenly populated as the trajectory allows rather than
    growing one giant segment by absorption.
    """
    while len(segs) > 1:
        counts = _seg_counts(segs, spike_times)
        i = int(np.argmin(counts))
        if counts[i] >= min_spikes:
            break
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = len(segs) - 2
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        a, b = min(i, j), max(i, j)
        segs = segs[:a] + [(segs[a][0], segs[b][1])] + segs[b + 1:]
    return segs


def _cap_segments(segs, spike_times, max_segments):
    """Merge the adjacent pair with the fewest combined spikes until the
    segment count fits. The cap exists because each segment is a full
    interferer-pick + LCMV fit + convolution pass, so calibration cost is
    linear in it."""
    while len(segs) > max_segments:
        counts = _seg_counts(segs, spike_times)
        pair = int(np.argmin([counts[i] + counts[i + 1] for i in range(len(segs) - 1)]))
        segs = segs[:pair] + [(segs[pair][0], segs[pair + 1][1])] + segs[pair + 2:]
    return segs


# --------------------------------------------------------------------- #
# Validation against a synthetic session's known drift
# --------------------------------------------------------------------- #

def _main():
    import argparse
    import threshold_sweep_real as tsr

    ap = argparse.ArgumentParser(
        description="Score unit_trajectory() against make_sim_session.py's "
                    "known drift_position_um.")
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--meta-path")
    ap.add_argument("--channel-map-json", required=True)
    ap.add_argument("--truth", required=True, help="sim_truth.npz")
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--spikes-per-bin", type=int, default=150)
    ap.add_argument("--min-spikes", type=int, default=60)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--tol-um", type=float, default=12.0)
    ap.add_argument("--min-spikes-per-segment", type=int, default=250)
    ap.add_argument("--max-segments", type=int, default=8)
    ap.add_argument("--scratch-dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.filter = True
    args.car = True
    args.causal_highpass = True
    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(args, rng, dtype=np.float32)

    positions = gf.load_channel_positions_json(args.channel_map_json)
    chan_y = channel_y_for_group(np_ch, positions)

    truth = np.load(args.truth, allow_pickle=True)
    t_uid = list(truth["unit_ids"])
    t_grid = truth["t_grid_s"]
    t_pos = truth["drift_position_um"]
    t_kind = truth["drift_kind"]

    ids, counts = np.unique(spike_cl, return_counts=True)
    order = np.argsort(counts)[::-1]
    cand = [int(ids[i]) for i in order if counts[i] >= args.min_spikes]
    if args.max_units:
        cand = cand[:args.max_units]

    print(f"{'unit':>5} {'kind':>5} {'true span':>9} {'est span':>9} "
          f"{'bias':>7} {'resid':>7} {'segs':>4}")
    rows = []
    for uid in cand:
        st = spike_t[spike_cl == uid]
        t_c, y = unit_trajectory(data, st, chan_y, args.template_length,
                                 args.template_offset, fs,
                                 spikes_per_bin=args.spikes_per_bin, rng=rng)
        if t_c.size == 0:
            continue
        i = t_uid.index(uid)
        y_true = np.interp(t_c, t_grid, t_pos[i])
        bias = float(np.mean(y - y_true))
        # Residual after removing the constant bias: the centroid estimator
        # is biased by the unit's own footprint width and by the group's
        # finite extent, but a constant offset does not affect segmentation,
        # which only ever looks at differences.
        resid = float(np.std(y - y_true - bias))
        segs = segment_from_trajectory(t_c, y, st, fs, data.shape[0],
                                       args.tol_um, args.min_spikes_per_segment,
                                       args.max_segments)
        print(f"{uid:>5} {str(t_kind[i]):>5} {np.ptp(y_true):9.1f} {np.ptp(y):9.1f} "
              f"{bias:7.1f} {resid:7.2f} {len(segs):4d}")
        rows.append((str(t_kind[i]), float(np.ptp(y_true)), float(np.ptp(y)),
                     resid, len(segs)))

    print()
    for kind in ("none", "ramp", "osc", "jump"):
        sel = [r for r in rows if r[0] == kind]
        if not sel:
            continue
        print(f"{kind:>5}: n={len(sel):2d}  median resid {np.median([r[3] for r in sel]):5.2f} um  "
              f"median segs {np.median([r[4] for r in sel]):.1f}  "
              f"median |span err| {np.median([abs(r[2] - r[1]) for r in sel]):5.1f} um")


if __name__ == "__main__":
    _main()
