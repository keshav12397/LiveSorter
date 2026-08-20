"""
dredge_lite.py
==============

Probe-wide motion estimation by decentralized registration -- a small,
self-contained version of the method in DREDge (Windolf et al.;
https://github.com/evarol/dredge), specialised to the situation this
codebase is already in.

Why this exists, and why it is not just `drift_estimate.py` again
-----------------------------------------------------------------
`drift_estimate.unit_trajectory` tracks each unit *independently*: bin that
unit's spikes, take each bin's mean waveform, follow its amplitude centroid.
That estimator has one irreducible problem. A unit firing at 1.5 Hz gives a
60 s bin about 90 spikes, and the centroid of a 90-spike average moves by
several microns on noise alone. The drift we care about is ~30 um over a
session. The measurement noise is a large fraction of the signal, per unit,
and no amount of smoothing one unit's own trajectory fixes that -- smoothing
trades the noise for exactly the lag that makes an abrupt step unfindable.

The way out is not a better per-unit estimator. It is to stop estimating per
unit at all. **Drift is common-mode**: the tissue slides along the shank and
carries the whole population with it. So 160 units are not 160 separate
problems, they are 160 noisy votes on ONE trajectory, and pooling them beats
any single vote by roughly sqrt(N).

That is the entire idea behind DREDge, and it is worth being precise about
what "decentralized" buys, because it is not obvious:

  - The naive pooled estimate picks a reference time bin and measures every
    other bin against it. That makes the whole trajectory hostage to one
    bin, and it degrades badly when the probe has moved so far that the
    reference no longer overlaps the current state.
  - The decentralized estimate instead measures *every pair* of time bins
    against each other, giving a displacement D_ij and a confidence w_ij,
    and then solves for the single trajectory p that best explains all of
    them at once. No bin is privileged. Pairs that could not be compared
    reliably simply get a low weight instead of corrupting the result.

The solve is a weighted least-squares over a graph, which is a graph
Laplacian system (see `solve_decentralized`), and with uniform weights it
collapses to the pleasing identity p_i = mean_j D_ij.

What is simplified relative to real DREDge
------------------------------------------
DREDge's AP mode starts from raw data and must *localize* each spike --
estimate a position for a single detected event -- before it can build a
raster. We are downstream of a sorter, so we already know which spikes
belong to which unit, and we can average a unit's spikes within a time bin
before measuring anything. The raster here is therefore built from per-unit
per-bin mean-waveform amplitude profiles rather than from single-spike
localizations: same raster, much better SNR per entry, no localizer to
tune. This is the one place where being downstream of a sorter is a real
advantage rather than an inconvenience.

Also omitted: DREDge's online/streaming solver, its robust (L1/Huber)
reweighting, and its GPU paths. `--min-corr` is the only robustness
mechanism here and it is a blunt one.

What this cannot do
-------------------
It recovers the *common-mode* trajectory and nothing else. A unit that
genuinely moves relative to its neighbours is invisible to it -- that motion
is not in the raster's shared structure. `make_sim_session.py` models both
terms separately (`probe_offset_um` and `unit_residual_offset_um`) so this
distinction can be scored rather than assumed.

Metric distances
----------------
As everywhere in this codebase, positions are microns from the channel map's
`yc`, never SpikeGLX channel ids -- `shank1only.json`'s chanMap steps by +49
twice, so two channels 15 um apart can have ids 49 apart. See
`drift_estimate.py`'s header.
"""

import numpy as np

import generate_filter as gf


# --------------------------------------------------------------------- #
# 1. The raster
# --------------------------------------------------------------------- #

def build_raster(data, spike_times, spike_clusters, chan_y, fs,
                 template_length, template_offset, unit_ids=None,
                 bin_s=20.0, depth_bin_um=5.0, min_spikes_per_bin=8,
                 max_spikes_per_bin=400, rng=None):
    """Amplitude as a function of (depth, time): the input every step below
    consumes.

    Returns `(raster, depth_grid_um, t_center_s)` where `raster` has shape
    (n_depth, n_time).

    Time bins are FIXED-DURATION here, unlike `drift_estimate.unit_trajectory`
    which uses equal-spike bins. That difference is forced and not a
    preference: a raster column has to be a snapshot of the whole population
    at one moment, so every unit's bin edges must line up. Equal-spike
    binning gives each unit its own edges, which is the right choice when
    each unit is estimated alone and the wrong one here.

    Each (unit, bin) contributes its mean waveform's per-channel
    peak-to-peak profile, scaled by how many spikes went into it. Keeping the
    full profile rather than collapsing it to a centroid is the point: the
    profile is a ~30 um bump whose *shape* is what cross-correlation
    registers, and it survives being summed with other units' bumps in a way
    a centroid does not.

    `min_spikes_per_bin` drops (unit, bin) cells too sparse to mean
    meaningfully. Their mean waveform is mostly noise, whose ptp profile is
    flat-ish and positive on every channel -- so admitting them does not add
    a weak signal, it adds a DC pedestal across the whole depth axis that
    dilutes every real bump.
    """
    rng = rng or np.random.default_rng(0)
    spike_times = np.asarray(spike_times)
    spike_clusters = np.asarray(spike_clusters)

    n_samples = data.shape[0]
    lo_ok = template_offset
    hi_ok = n_samples - template_length + template_offset

    if unit_ids is None:
        unit_ids = np.unique(spike_clusters)

    bin_samples = int(round(bin_s * fs))
    n_time = max(int(np.ceil(n_samples / bin_samples)), 1)
    t_center = (np.arange(n_time) + 0.5) * bin_samples / fs

    y_lo, y_hi = float(np.min(chan_y)), float(np.max(chan_y))
    depth_grid = np.arange(y_lo, y_hi + depth_bin_um, depth_bin_um)
    raster = np.zeros((depth_grid.size, n_time), dtype=np.float64)

    # The amplitude profile is INTERPOLATED across depth, not scattered onto
    # the bins the channels happen to occupy. This is the difference between
    # an estimator that can see sub-pitch drift and one that cannot.
    #
    # What the obvious version does wrong. Spreading each channel's
    # amplitude linearly between the two grid points bracketing its depth
    # looks like it handles the general case, and it does -- unless the
    # channel pitch is a multiple of the bin size, which it usually is
    # (15 um rows, 5 um bins). Then every channel lands exactly on a grid
    # point, the interpolation weight is 0 for all of them, and the raster
    # becomes a COMB: energy at every third bin, nothing between.
    #
    # A comb cannot be cross-correlated at sub-pitch shifts. Sliding it by
    # one bin moves every tooth into a gap, so correlation does not decay
    # smoothly away from the true shift, it collapses -- measured on Lav69:
    # +0.99 at zero shift, -0.24 at +-5 um, -0.24 at +-10 um. The peak is a
    # delta, argmax pins to 0 for every pair, D comes back all-zero, and the
    # trajectory is flat. Only shifts that are exact multiples of the
    # channel pitch can ever align.
    #
    # This survived its own unit tests and a full simulator validation
    # because the simulated drift was 15 um -- exactly one channel row,
    # exactly three bins -- so the comb realigned perfectly and the reported
    # 1.05 um error was measured at the one displacement the representation
    # can express. Real drift of 6.5 um is not a multiple of the pitch and
    # the estimator returned identically zero.
    #
    # So: average the channels sharing a row (the columns are at the same
    # depth and differ only in x), then linearly interpolate that
    # row-resolved profile onto the fine grid. The result is the continuous
    # function the amplitudes are samples of, and shifting it by any amount
    # changes it smoothly.
    chan_y = np.asarray(chan_y, dtype=np.float64)
    y_rows = np.unique(chan_y)
    row_of = np.searchsorted(y_rows, chan_y)
    row_n = np.bincount(row_of, minlength=y_rows.size).astype(np.float64)
    row_n[row_n == 0] = 1.0

    for uid in unit_ids:
        st = spike_times[spike_clusters == uid]
        st = st[(st > lo_ok) & (st < hi_ok)]
        if st.size == 0:
            continue
        which = np.minimum((st // bin_samples).astype(int), n_time - 1)
        for b in np.unique(which):
            bin_st = st[which == b]
            if bin_st.size < min_spikes_per_bin:
                continue
            wf, _ = gf.mean_waveform(data, bin_st, template_length,
                                     template_offset,
                                     max_spikes=max_spikes_per_bin, rng=rng)
            amp = np.ptp(np.asarray(wf, dtype=np.float64), axis=0)
            # Same noise-floor subtraction as _position_from_waveform, for
            # the same reason: with few spikes averaged every channel has a
            # positive ptp from noise alone.
            amp = np.maximum(amp - np.median(amp), 0.0)
            if not np.any(amp > 0):
                continue
            amp = amp * float(bin_st.size)
            row_amp = np.bincount(row_of, weights=amp,
                                  minlength=y_rows.size) / row_n
            raster[:, int(b)] += np.interp(depth_grid, y_rows, row_amp)

    return raster, depth_grid, t_center


# --------------------------------------------------------------------- #
# 2. Pairwise displacements
# --------------------------------------------------------------------- #

def pairwise_displacements(raster, depth_bin_um, max_disp_um=100.0,
                           max_lag_bins=0, min_corr=0.0):
    """Cross-correlate every admissible pair of time bins.

    Returns `(D, W)`, both (n_time, n_time). `D[i, j]` is the depth shift in
    microns that best takes column i onto column j, and `W[i, j]` is the
    normalized correlation at that shift, used as the pair's weight. `D` is
    antisymmetric and `W` symmetric by construction -- computed on the upper
    triangle and mirrored, not computed twice, so the two can never disagree.

    Correlations are true Pearson coefficients in [-1, 1], so a bin whose
    total amplitude happens to be large does not get a louder vote than a bin
    whose spatial structure is clearer.

    Each shift is scored on the OVERLAPPING depth range only, with the mean
    and norm recomputed over that range. Normalizing the full columns once
    up front and then zeroing whatever a shift pushes off the end is the
    obvious implementation and it is wrong: it leaves the denominator sized
    for the full column while the numerator has lost rows, so every large
    shift is penalised in proportion to its size and the estimate is
    systematically shrunk toward zero. The bias is invisible on a long probe
    with small drift and clearly visible on a narrow nonrigid window, where
    a +-60 um search discards a fifth of a 330 um window's rows.

    `max_lag_bins` bands the problem: pairs further apart in time than this
    are not compared at all. Two reasons, and the second matters more. The
    obvious one is cost, which is quadratic in the number of time bins. The
    real one is that when the probe has drifted far, distant bins genuinely
    do not overlap, and their correlation peak is then matching noise -- a
    confidently wrong D_ij, which is much worse than a missing one. 0 means
    no banding.

    `min_corr` zeroes pairs whose best correlation is below it. A zero weight
    removes the pair from the solve entirely, which is the intended effect:
    an unmeasurable pair should abstain, not vote quietly.
    """
    raster = np.asarray(raster, dtype=np.float64)
    n_depth, n_time = raster.shape
    max_shift = int(round(max_disp_um / depth_bin_um))
    max_shift = int(np.clip(max_shift, 1, max(n_depth - 1, 1)))

    X = raster

    D = np.zeros((n_time, n_time), dtype=np.float64)
    W = np.zeros((n_time, n_time), dtype=np.float64)

    shifts = np.arange(-max_shift, max_shift + 1)
    for i in range(n_time):
        hi = n_time if max_lag_bins <= 0 else min(n_time, i + max_lag_bins + 1)
        if hi <= i + 1:
            continue
        block = X[:, i + 1:hi]                       # (n_depth, k)
        cc = np.full((shifts.size, block.shape[1]), -np.inf, dtype=np.float64)
        for k, s in enumerate(shifts):
            # Rows of `block` that a reference shifted by `s` actually covers.
            lo_b, hi_b = max(0, s), n_depth + min(0, s)
            if hi_b - lo_b < 4:
                continue
            ref = X[lo_b - s:hi_b - s, i]
            blk = block[lo_b:hi_b, :]
            r = ref - ref.mean()
            b = blk - blk.mean(axis=0, keepdims=True)
            rn = np.linalg.norm(r)
            bn = np.linalg.norm(b, axis=0)
            denom = rn * bn
            good = denom > 0
            cc[k, good] = (r @ b[:, good]) / denom[good]
        best = np.argmax(cc, axis=0)
        cols = np.arange(block.shape[1])
        peak = cc[best, cols]

        # Sub-bin refinement, and it is not a polish -- without it this
        # function cannot see drift smaller than one depth bin AT ALL.
        #
        # The shift search is over integer bins, so `shifts[best]` is
        # quantised to depth_bin_um. On a real Lav69 recording the true
        # drift is ~6.5 um against a 5 um bin, every pair's argmax landed on
        # lag 0, D came back identically zero, and the solve dutifully
        # returned a flat trajectory. Nothing looked broken: the
        # correlations were excellent (mean 0.97) precisely BECAUSE the
        # columns nearly align at zero shift. A simulator run at 15 um (3
        # bins) cleared the floor and hid this completely.
        #
        # Fit a parabola through the correlation at (best-1, best, best+1)
        # and take its vertex. Standard sub-sample peak location; the three
        # values are already computed, so this is arithmetic on a (k,)
        # vector, not another correlation pass.
        #
        # Guards, each for a real case rather than defensiveness:
        #   - interior only. At an endpoint there is no bracketing triple
        #     and the vertex is unconstrained.
        #   - finite neighbours. cc is pre-filled with -inf for shifts whose
        #     overlap fell below 4 rows.
        #   - denom < 0 keeps only genuine maxima (concave-down); a
        #     near-zero denominator is a flat ridge whose vertex flies off.
        #   - clip to +-0.5 bin. A vertex further than half a bin from the
        #     argmax contradicts the argmax, so trust the integer answer.
        frac = np.zeros(block.shape[1], dtype=np.float64)
        interior = (best > 0) & (best < shifts.size - 1)
        if np.any(interior):
            ii = np.where(interior)[0]
            c0 = cc[best[ii], ii]
            cm = cc[best[ii] - 1, ii]
            cp = cc[best[ii] + 1, ii]
            ok = np.isfinite(cm) & np.isfinite(cp) & np.isfinite(c0)
            denom = cm - 2.0 * c0 + cp
            ok &= denom < -1e-12
            if np.any(ok):
                f = np.zeros_like(denom)
                f[ok] = 0.5 * (cm[ok] - cp[ok]) / denom[ok]
                frac[ii] = np.clip(f, -0.5, 0.5)

        disp = (shifts[best] + frac) * depth_bin_um

        peak = np.where(peak >= min_corr, np.maximum(peak, 0.0), 0.0)
        disp = np.where(peak > 0.0, disp, 0.0)
        j = np.arange(i + 1, hi)
        D[i, j] = disp
        D[j, i] = -disp
        W[i, j] = peak
        W[j, i] = peak

    return D, W


# --------------------------------------------------------------------- #
# 3. The decentralized solve
# --------------------------------------------------------------------- #

def solve_decentralized(D, W, ridge=1e-6):
    """Least-squares trajectory `p` from pairwise displacements.

    Minimises  sum_{i<j} W_ij * (p_j - p_i - D_ij)^2.

    Setting the gradient to zero gives the normal equations `L p = b` with
    `L = diag(W.sum(1)) - W` -- the weighted graph Laplacian -- and
    `b_k = -sum_j W_kj * D_kj`. `L` is singular: adding a constant to every
    p_i changes no difference and so changes no residual, which is the
    correct statement that only *relative* motion is observable here. The
    ridge picks one solution out of that family and the mean-centring below
    picks the natural one.

    That leading minus on `b` is not a typo and is easy to get backwards --
    `D` is antisymmetric, so both sums in the gradient collapse to the same
    `+D_kj` form and the term lands on the opposite side from where it looks
    like it should. With uniform weights and a fully connected graph the
    solution is p_k = mean_j D_jk: the mean displacement taking every other
    bin *onto* bin k, not off it. `test_dredge_lite.py` asserts exactly that
    identity, which is what caught the sign being wrong the first time.
    """
    W = np.asarray(W, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    n = W.shape[0]
    L = np.diag(W.sum(axis=1)) - W
    b = -np.sum(W * D, axis=1)
    p = np.linalg.solve(L + ridge * np.eye(n), b)
    return p - p.mean()


# --------------------------------------------------------------------- #
# 4. Rigid and nonrigid estimation
# --------------------------------------------------------------------- #

def estimate_motion(raster, depth_grid_um, t_center_s, n_windows=1,
                    window_overlap=0.5, max_disp_um=100.0,
                    max_lag_bins=0, min_corr=0.0):
    """Motion trajectory from a raster. Returns `(t_center_s, motion)`.

    `motion` has shape (n_windows, n_time). With `n_windows == 1` this is the
    rigid estimate: one trajectory for the whole probe. With more, the depth
    axis is cut into overlapping windows and each is solved independently --
    the nonrigid case, where tissue near one end of the probe moves
    differently from the other end.

    Windows OVERLAP on purpose. Disjoint windows give a motion field with
    discontinuities at the boundaries, which is not a thing tissue does, and
    each window would also see only its own share of units -- the pooling
    that makes this method work at all needs the units.

    Use `motion_at` to evaluate the result at a particular depth.
    """
    depth_bin_um = float(depth_grid_um[1] - depth_grid_um[0])
    n_depth = raster.shape[0]

    if n_windows <= 1:
        D, W = pairwise_displacements(raster, depth_bin_um, max_disp_um,
                                      max_lag_bins, min_corr)
        return t_center_s, solve_decentralized(D, W)[None, :]

    # Window centres span the depth axis; each window is wide enough that
    # consecutive ones overlap by `window_overlap`.
    width = n_depth / (1.0 + (n_windows - 1) * (1.0 - window_overlap))
    out = np.zeros((n_windows, t_center_s.size), dtype=np.float64)
    for w in range(n_windows):
        lo = int(round(w * width * (1.0 - window_overlap)))
        hi = int(round(lo + width))
        sub = raster[max(lo, 0):min(hi, n_depth), :]
        if sub.shape[0] < 4:
            continue
        D, W = pairwise_displacements(sub, depth_bin_um, max_disp_um,
                                      max_lag_bins, min_corr)
        out[w] = solve_decentralized(D, W)
    return t_center_s, out


def window_centers_um(depth_grid_um, n_windows, window_overlap=0.5):
    """Depth of each window's centre, matching estimate_motion's geometry."""
    n_depth = depth_grid_um.size
    if n_windows <= 1:
        return np.array([0.5 * (depth_grid_um[0] + depth_grid_um[-1])])
    width = n_depth / (1.0 + (n_windows - 1) * (1.0 - window_overlap))
    c = []
    for w in range(n_windows):
        lo = w * width * (1.0 - window_overlap)
        idx = np.clip(lo + 0.5 * width, 0, n_depth - 1)
        c.append(float(np.interp(idx, np.arange(n_depth), depth_grid_um)))
    return np.asarray(c)


def motion_at(t_query_s, y_query_um, t_center_s, motion, win_centers_um):
    """Evaluate an `estimate_motion` result at arbitrary (time, depth).

    Linear in both axes, and constant (edge-clamped) outside the estimated
    range: extrapolating a motion field past the units that constrained it
    invents drift that nothing measured.
    """
    motion = np.atleast_2d(motion)
    t_query_s = np.asarray(t_query_s, dtype=np.float64)
    if motion.shape[0] == 1:
        return np.interp(t_query_s, t_center_s, motion[0])
    per_win = np.stack([np.interp(t_query_s, t_center_s, motion[w])
                        for w in range(motion.shape[0])], axis=0)
    y = np.clip(float(y_query_um), float(np.min(win_centers_um)),
                float(np.max(win_centers_um)))
    return np.array([np.interp(y, win_centers_um, per_win[:, k])
                     for k in range(t_query_s.size)])


# --------------------------------------------------------------------- #
# 5. Head-to-head against the per-unit estimator
# --------------------------------------------------------------------- #

def _main():
    """Score this estimator against make_sim_session.py's known probe motion,
    and against drift_estimate.unit_trajectory on the same data.

    The comparison is the point. Either estimator alone produces a number
    that looks fine in isolation; what decides whether pooling was worth
    implementing is which one has less error on the same recording, and by
    how much relative to the drift being measured.

    Two truths are reported because they answer different questions:
      probe  -- the common-mode motion. What this method estimates, and the
                only thing it *can* estimate.
      total  -- probe plus each unit's private residual. What a per-unit
                tracker is trying to estimate, and what a drift-aware filter
                would ideally follow.
    Scoring a pooled estimator against `total` and calling the gap a failure
    would be charging it for a quantity outside its model, so both are
    printed rather than one being picked here.
    """
    import argparse
    import threshold_sweep_real as tsr
    import drift_estimate as de

    ap = argparse.ArgumentParser(
        description="dredge_lite vs per-unit tracking, against known truth.")
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--meta-path")
    ap.add_argument("--channel-map-json", required=True)
    ap.add_argument("--truth", required=True, help="sim_truth.npz")
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--bin-s", type=float, default=20.0)
    ap.add_argument("--depth-bin-um", type=float, default=5.0)
    ap.add_argument("--max-disp-um", type=float, default=80.0)
    ap.add_argument("--max-lag-bins", type=int, default=0)
    ap.add_argument("--min-corr", type=float, default=0.1)
    ap.add_argument("--n-windows", type=int, default=1)
    ap.add_argument("--min-spikes", type=int, default=60)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--scratch-dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.filter = True
    args.car = True
    args.causal_highpass = True
    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(
        args, rng, dtype=np.float32)

    positions = gf.load_channel_positions_json(args.channel_map_json)
    chan_y = de.channel_y_for_group(np_ch, positions)

    truth = np.load(args.truth, allow_pickle=True)
    t_uid = list(truth["unit_ids"])
    t_grid = truth["t_grid_s"]
    t_total = truth["drift_position_um"]
    t_probe = (truth["probe_offset_um"] if "probe_offset_um" in truth
               else np.zeros_like(t_total))
    t_y0 = truth["y0_um"]

    ids, counts = np.unique(spike_cl, return_counts=True)
    order = np.argsort(counts)[::-1]
    cand = [int(ids[i]) for i in order if counts[i] >= args.min_spikes]
    if args.max_units:
        cand = cand[:args.max_units]

    print(f"building raster over {len(cand)} units, {args.bin_s:.0f} s bins...")
    raster, depth_grid, t_c = build_raster(
        data, spike_t, spike_cl, chan_y, fs, args.template_length,
        args.template_offset, unit_ids=cand, bin_s=args.bin_s,
        depth_bin_um=args.depth_bin_um, rng=rng)
    print(f"  raster {raster.shape[0]} depths x {raster.shape[1]} time bins, "
          f"{100.0 * np.mean(raster > 0):.0f}% occupied")

    _, motion = estimate_motion(raster, depth_grid, t_c,
                                n_windows=args.n_windows,
                                max_disp_um=args.max_disp_um,
                                max_lag_bins=args.max_lag_bins,
                                min_corr=args.min_corr)
    wc = window_centers_um(depth_grid, args.n_windows)
    print(f"  estimated motion span {np.ptp(motion):.1f} um across "
          f"{args.n_windows} window(s)")

    def rms_centered(a, b):
        e = (a - np.mean(a)) - (b - np.mean(b))
        return float(np.sqrt(np.mean(e ** 2)))

    print(f"\n{'unit':>5} {'probe span':>10} {'dredge':>8} {'per-unit':>9} "
          f"{'vs total':>9}")
    rows = []
    for uid in cand:
        i = t_uid.index(uid)
        st = spike_t[spike_cl == uid]

        t_u, y_u = de.unit_trajectory(data, st, chan_y, args.template_length,
                                      args.template_offset, fs, rng=rng)
        if t_u.size < 2:
            continue

        probe_at_u = np.interp(t_u, t_grid, t_probe[i])
        total_at_u = np.interp(t_u, t_grid, t_total[i])
        dredge_at_u = motion_at(t_u, float(t_y0[i]), t_c, motion, wc)

        e_dredge = rms_centered(dredge_at_u, probe_at_u)
        e_unit = rms_centered(y_u, probe_at_u)
        e_dredge_total = rms_centered(dredge_at_u, total_at_u)
        span = float(np.ptp(probe_at_u))
        print(f"{uid:>5} {span:10.1f} {e_dredge:8.2f} {e_unit:9.2f} "
              f"{e_dredge_total:9.2f}")
        rows.append((span, e_dredge, e_unit, e_dredge_total))

    if not rows:
        print("no units scored")
        return
    a = np.asarray(rows)
    print(f"\nn={len(rows)}  median true probe span {np.median(a[:, 0]):.1f} um")
    print(f"  vs probe motion : dredge_lite {np.median(a[:, 1]):5.2f} um   "
          f"per-unit {np.median(a[:, 2]):5.2f} um")
    print(f"  vs total motion : dredge_lite {np.median(a[:, 3]):5.2f} um")
    better = 100.0 * np.mean(a[:, 1] < a[:, 2])
    print(f"  dredge_lite beats per-unit tracking on {better:.0f}% of units")


if __name__ == "__main__":
    _main()
