"""
test_dredge_lite.py -- correctness of the decentralized registration solve.

Runs against synthetic rasters built here, not against a recording. The
question these answer is narrow and worth keeping separate from any
data-dependent one: given a raster that provably contains a known
trajectory, does the estimator return that trajectory? A failure here is a
bug in the algebra; a failure on real data with these passing is a
statement about the data.

    python test_dredge_lite.py
"""

import numpy as np

import dredge_lite as dl


def synth_raster(n_depth, n_time, motion_um, depth_bin_um, n_units=40,
                 noise=0.0, seed=0):
    """A raster of `n_units` Gaussian amplitude bumps rigidly displaced by
    `motion_um[t]` at time t. This is exactly what build_raster produces from
    a drifting population, with the sorter and the recording taken out."""
    rng = np.random.default_rng(seed)
    y = np.arange(n_depth) * depth_bin_um
    centers = rng.uniform(y[0] + 40.0, y[-1] - 40.0, size=n_units)
    amps = np.exp(rng.uniform(np.log(1.0), np.log(20.0), size=n_units))
    sig = rng.uniform(12.0, 26.0, size=n_units)

    R = np.zeros((n_depth, n_time))
    for t in range(n_time):
        for c, a, s in zip(centers, amps, sig):
            R[:, t] += a * np.exp(-0.5 * ((y - (c + motion_um[t])) / s) ** 2)
    if noise > 0:
        R += rng.normal(0.0, noise * R.mean(), R.shape)
        R = np.maximum(R, 0.0)
    return R, y


def _score(name, p_est, p_true, tol_um):
    # Only differences are observable, so both are mean-centred before
    # comparing -- see solve_decentralized's note on the constant nullspace.
    e = (p_est - p_est.mean()) - (p_true - p_true.mean())
    rms = float(np.sqrt(np.mean(e ** 2)))
    ok = rms <= tol_um
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: rms {rms:.2f} um "
          f"(tol {tol_um:.1f})")
    return ok


def test_uniform_weights_closed_form():
    """With uniform weights and full connectivity the solve must reduce to
    p_i = mean_j D_ij. Checking that identity catches sign errors and
    Laplacian mistakes that a noisy end-to-end test would absorb."""
    rng = np.random.default_rng(3)
    n = 12
    p = rng.normal(size=n) * 10.0
    D = p[None, :] - p[:, None]          # exact, antisymmetric
    W = np.ones((n, n)) - np.eye(n)
    est = dl.solve_decentralized(D, W)
    closed = D.mean(axis=1)
    closed = closed - closed.mean()
    # est solves for p with D_ij = p_j - p_i, so est should equal -closed:
    # b_k = sum_j W_kj D_kj = sum_j (p_j - p_k) = -n*(p_k - mean(p)).
    # Tolerance is set by solve_decentralized's ridge, not by the algebra:
    # the identity is exact, the regularised solve is not.
    ok = _score("uniform weights == closed form", est, -closed, 1e-3)
    ok &= _score("uniform weights recovers p", est, p, 1e-3)
    return ok


def test_rigid_ramp():
    n_time = 40
    motion = np.linspace(0.0, 30.0, n_time)     # the user's 30 um / 2 channels
    R, y = synth_raster(160, n_time, motion, 5.0)
    _, est = dl.estimate_motion(R, y, np.arange(n_time), max_disp_um=60.0)
    return _score("rigid 30 um ramp, noiseless", est[0], motion, 3.0)


def test_rigid_ramp_noisy():
    n_time = 40
    motion = np.linspace(0.0, 30.0, n_time)
    R, y = synth_raster(160, n_time, motion, 5.0, noise=0.30, seed=7)
    _, est = dl.estimate_motion(R, y, np.arange(n_time), max_disp_um=60.0,
                                min_corr=0.1)
    return _score("rigid 30 um ramp, 30% noise", est[0], motion, 4.0)


def test_step():
    """A step is the case a smoothed per-unit tracker cannot follow. The
    decentralized solve has no temporal smoothing at all, so it should place
    the step where it is."""
    n_time = 40
    motion = np.where(np.arange(n_time) >= 20, 25.0, 0.0)
    R, y = synth_raster(160, n_time, motion, 5.0, noise=0.20, seed=11)
    _, est = dl.estimate_motion(R, y, np.arange(n_time), max_disp_um=60.0,
                                min_corr=0.1)
    return _score("25 um step", est[0], motion, 4.0)


def test_no_motion_returns_no_motion():
    """The failure mode that would quietly ruin everything downstream: an
    estimator that hallucinates drift on a static recording, whose filters
    then get re-fit for motion that never happened."""
    n_time = 30
    motion = np.zeros(n_time)
    R, y = synth_raster(160, n_time, motion, 5.0, noise=0.25, seed=5)
    _, est = dl.estimate_motion(R, y, np.arange(n_time), max_disp_um=60.0,
                                min_corr=0.1)
    return _score("static session stays static", est[0], motion, 2.0)


def test_banding_matches_full():
    """Banding is an efficiency measure; on slow drift it must not change the
    answer much, or the default max_lag_bins is silently a modelling choice."""
    n_time = 40
    motion = np.linspace(0.0, 30.0, n_time)
    R, y = synth_raster(160, n_time, motion, 5.0, noise=0.20, seed=13)
    _, full = dl.estimate_motion(R, y, np.arange(n_time), max_disp_um=60.0)
    _, band = dl.estimate_motion(R, y, np.arange(n_time), max_disp_um=60.0,
                                 max_lag_bins=10)
    return _score("banded == full", band[0], full[0], 2.0)


def test_nonrigid():
    """Two depth halves moving oppositely. A rigid fit must average them to
    ~nothing; the windowed fit must separate them. Both halves are checked,
    because a windowed estimator that merely reports *some* depth dependence
    is not the same as one that gets the sign right at each end."""
    n_depth, n_time = 200, 36
    depth_bin = 5.0
    y = np.arange(n_depth) * depth_bin
    top = np.linspace(0.0, 25.0, n_time)
    bot = np.linspace(0.0, -25.0, n_time)

    rng = np.random.default_rng(21)
    R = np.zeros((n_depth, n_time))
    centers, weights = [], []
    for c in rng.uniform(y[0] + 60, y[-1] - 60, size=60):
        a = float(np.exp(rng.uniform(np.log(1.0), np.log(20.0))))
        s = float(rng.uniform(12.0, 26.0))
        centers.append(c)
        weights.append(a)
        # Linear blend between the two ends, so the field is smooth in depth
        # rather than a discontinuity the windows could never represent.
        w = (c - y[0]) / (y[-1] - y[0])
        m = w * bot + (1.0 - w) * top
        for t in range(n_time):
            R[:, t] += a * np.exp(-0.5 * ((y - (c + m[t])) / s) ** 2)

    t_c = np.arange(n_time)
    _, rigid = dl.estimate_motion(R, y, t_c, n_windows=1, max_disp_um=60.0)
    ok = _score("nonrigid: rigid fit averages to ~0", rigid[0],
                np.zeros(n_time), 6.0)

    n_win = 5
    _, nr = dl.estimate_motion(R, y, t_c, n_windows=n_win, max_disp_um=60.0)
    wc = dl.window_centers_um(y, n_win)

    # Evaluate at the WINDOW CENTRES, not at the probe tips. The outermost
    # centres sit ~165 um inside each end, and motion_at edge-clamps beyond
    # them by design, so querying a tip asks the estimator for motion it
    # deliberately refuses to extrapolate and then scores it on the answer.
    # Getting this wrong makes a correct estimator look ~30% off at the deep
    # end purely from the depth gap.
    #
    # The truth a window can be held to is not the motion at its geometric
    # centre either. A window spans ~330 um, across which the true field here
    # varies by ~17 um, and one window fits ONE trajectory to all of it. What
    # it converges to is the AMPLITUDE-WEIGHTED mean of the field over its
    # own units, since a loud unit dominates the cross-correlation. Scoring
    # against the geometric centre instead charges the estimator for the
    # windows' unit distribution being uneven, which is not an error.
    centers = np.asarray(centers)
    weights = np.asarray(weights)
    half = 0.5 * (y[-1] - y[0]) / (1.0 + (n_win - 1) * 0.5)
    for w, c in enumerate(wc):
        inside = np.abs(centers - c) <= half
        wt = weights[inside]
        frac = (centers[inside] - y[0]) / (y[-1] - y[0])
        f_bar = float(np.sum(wt * frac) / np.sum(wt))
        truth = f_bar * bot + (1.0 - f_bar) * top
        est = dl.motion_at(t_c, c, t_c, nr, wc)
        ok &= _score(f"nonrigid: window {w} at {c:.0f} um", est, truth, 4.0)
    return ok


def main():
    tests = [test_uniform_weights_closed_form, test_rigid_ramp,
             test_rigid_ramp_noisy, test_step, test_no_motion_returns_no_motion,
             test_banding_matches_full, test_nonrigid]
    ok = True
    for t in tests:
        print(t.__name__)
        ok &= bool(t())
    print("\n" + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


# --------------------------------------------------------------------- #
# build_raster: the representation, not the algebra
# --------------------------------------------------------------------- #

def synth_recording(motion_um, fs=3000.0, duration_s=300.0, n_rows=24,
                    row_pitch_um=15.0, y0_um=435.0, n_units=10, seed=0):
    """A tiny fake recording: two channel columns per row, Gaussian spatial
    footprints, one biphasic waveform, spikes at known times, and a rigid
    drift of `motion_um(t)`.

    Deliberately goes through build_raster rather than constructing a raster
    directly, because every other test here builds its raster by hand and
    that is exactly the gap the comb bug lived in. fs is low so the array
    stays small; build_raster only uses fs to size time bins.
    """
    rng = np.random.default_rng(seed)
    n_chan = n_rows * 2
    chan_y = np.repeat(y0_um + row_pitch_um * np.arange(n_rows), 2)
    n_samp = int(duration_s * fs)
    data = rng.normal(0.0, 0.5, size=(n_samp, n_chan)).astype(np.float32)

    tt = np.arange(61) - 20
    wave = -np.exp(-0.5 * (tt / 3.0) ** 2) + 0.4 * np.exp(-0.5 * ((tt - 8) / 5.0) ** 2)

    centers = rng.uniform(chan_y.min() + 60, chan_y.max() - 60, size=n_units)
    amps = rng.uniform(8.0, 40.0, size=n_units)
    st, cl = [], []
    for u in range(n_units):
        times = np.sort(rng.integers(40, n_samp - 80, size=900))
        for t in times:
            off = np.interp(t / fs, np.arange(len(motion_um)) * duration_s / len(motion_um),
                            motion_um)
            prof = amps[u] * np.exp(-0.5 * ((chan_y - (centers[u] + off)) / 22.0) ** 2)
            data[t - 20:t + 41, :] += wave[:, None] * prof[None, :]
        st.append(times)
        cl.append(np.full(times.size, u))
    return data, np.concatenate(st), np.concatenate(cl), chan_y, fs


def test_build_raster_sees_sub_pitch_drift():
    """Drift SMALLER than the channel pitch must still be recovered.

    This is the test the comb bug needed and did not have. build_raster puts
    amplitude on a depth grid finer than the channel pitch; if it scatters
    each channel onto its own bin instead of interpolating between rows, the
    raster is a comb with energy every `row_pitch/depth_bin` bins. A comb
    cross-correlates to a delta -- collapsing rather than decaying away from
    the true shift -- so argmax pins to zero and the estimate is flat.

    The drift here is 8 um against a 15 um pitch and a 5 um bin, so it is
    a multiple of NEITHER. That matters: the simulator's 15 um drift was
    exactly one row and three bins, the comb realigned perfectly, and this
    whole failure mode stayed invisible through a full validation.
    """
    n_time = 30
    true = 8.0 * np.linspace(0.0, 1.0, n_time)
    true = true - true.mean()
    data, st, cl, chan_y, fs = synth_recording(true)

    raster, depth_grid, t_center = dl.build_raster(
        data, st, cl, chan_y, fs, 61, 20, bin_s=10.0, depth_bin_um=5.0,
        min_spikes_per_bin=4)

    # The representation itself: no comb. With 15 um rows on a 5 um grid a
    # comb puts amplitude on every third bin and exactly zero on the other
    # two, so ~2/3 of bins inside the active depth range are zero.
    #
    # Scored on the ACTIVE range only -- between the first and last nonzero
    # bin. Zeros outside it are not a comb, they are depths no unit reaches,
    # and build_raster's noise-floor subtraction zeroes channels far from
    # every unit on purpose. Counting those would make the threshold depend
    # on how the fake units happened to be scattered.
    col = raster[:, 0]
    nz = np.flatnonzero(col)
    assert nz.size > 8, "no signal in the raster at all"
    active = col[nz[0]:nz[-1] + 1]
    zero_frac = float(np.mean(active == 0.0))
    assert zero_frac < 0.20, (
        f"raster looks like a comb: {zero_frac:.1%} of bins in the active "
        f"depth range are exactly 0 (a comb gives ~67%)")

    _t, motion = dl.estimate_motion(raster, depth_grid, t_center,
                                    max_disp_um=40.0, min_corr=0.1)
    est = np.asarray(motion)[0]
    est = est - est.mean()

    err = float(np.sqrt(np.mean((est - true) ** 2)))
    zero_err = float(np.sqrt(np.mean(true ** 2)))
    corr = float(np.corrcoef(est, true)[0, 1])

    # Asserted on SHAPE and non-flatness, not on rmse, and the reason is a
    # real limitation rather than a loose test.
    #
    # The comb failure is total: it returns a flat trajectory, so ptp ~ 0
    # and the correlation is undefined. Either assertion below catches it
    # with enormous margin.
    #
    # rmse is deliberately not asserted tightly because this estimator
    # OVERSHOOTS the drift amplitude on sparse, noisy populations -- here
    # ~12.7 um recovered for 8.0 um true, and a 6-unit subset of a real
    # session read 40.3 um against ~15 um truth. On Lav69's full 163-unit
    # population the gain was accurate (5.94 vs KS4's 6.50), so the bias
    # shrinks as units are pooled, which is consistent with it coming from
    # noise in each pair's correlation peak biasing the argmax outward.
    # Pinning rmse here would encode this synthetic setup's particular
    # amount of overshoot as if it were correct. See DRIFT_AWARE_RESULTS.md;
    # the practical consequence is that a drift SPAN from few units must not
    # be trusted to size anything.
    assert np.ptp(est) > 0.4 * np.ptp(true), (
        f"estimate is flat -- the comb failure: ptp {np.ptp(est):.2f} "
        f"vs true {np.ptp(true):.2f}")
    assert corr > 0.7, f"trajectory shape is wrong: corr {corr:+.3f}"
    assert err < zero_err, f"rmse {err:.2f} um, worse than estimating zero ({zero_err:.2f})"
    print(f"  sub-pitch drift: true ptp {np.ptp(true):.2f}  est ptp {np.ptp(est):.2f}  "
          f"corr {corr:+.3f}  rmse {err:.2f} um (zero-estimate {zero_err:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())
