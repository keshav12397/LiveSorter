"""
test_drift_estimate.py -- pooled motion estimation from per-unit centroids.

Runs against a small synthetic recording built here rather than a real one.
The question is narrow on purpose: given a recording that provably contains a
known rigid drift, does `pooled_com_motion` return that drift? A failure here
is a bug in the estimator; a failure on real data with these passing is a
statement about the data.

These replace test_dredge_lite.py, which tested the raster-registration
estimator that `drift_estimate.pooled_com_motion` displaced. The sub-pitch
case below is carried over deliberately -- it is the case that caught the
raster method's comb bug, and it is exactly the case a channel-grid-based
estimator is most likely to fail, so it keeps earning its place even though
the current estimator has no grid to fail on.

    python -m pytest test_drift_estimate.py -v
"""

import numpy as np

import drift_estimate as de


def synth_recording(motion_um, fs=3000.0, duration_s=300.0, n_rows=24,
                    row_pitch_um=15.0, y0_um=435.0, n_units=12,
                    spikes_per_unit=1200, noise=0.5, seed=0):
    """A tiny fake recording: two channel columns per row, Gaussian spatial
    footprints, one biphasic waveform, and a rigid drift of `motion_um(t)`
    applied to every unit.

    `fs` is low so the array stays small; nothing under test uses it for
    anything but converting sample indices to seconds.
    """
    rng = np.random.default_rng(seed)
    n_chan = n_rows * 2
    chan_y = np.repeat(y0_um + row_pitch_um * np.arange(n_rows), 2)
    n_samp = int(duration_s * fs)
    data = rng.normal(0.0, noise, size=(n_samp, n_chan)).astype(np.float32)

    tt = np.arange(61) - 20
    wave = -np.exp(-0.5 * (tt / 3.0) ** 2) + 0.4 * np.exp(-0.5 * ((tt - 8) / 5.0) ** 2)

    mt = np.arange(len(motion_um)) * duration_s / len(motion_um)
    centers = rng.uniform(chan_y.min() + 60, chan_y.max() - 60, size=n_units)
    amps = rng.uniform(10.0, 45.0, size=n_units)

    st, cl = [], []
    for u in range(n_units):
        times = np.sort(rng.integers(40, n_samp - 80, size=spikes_per_unit))
        offs = np.interp(times / fs, mt, motion_um)
        for t, off in zip(times, offs):
            prof = amps[u] * np.exp(-0.5 * ((chan_y - (centers[u] + off)) / 22.0) ** 2)
            data[t - 20:t + 41, :] += wave[:, None] * prof[None, :]
        st.append(times)
        cl.append(np.full(times.size, u))
    return data, np.concatenate(st), np.concatenate(cl), chan_y, fs


def _run(motion_um, **kw):
    data, st, cl, chan_y, fs = synth_recording(motion_um, **kw)
    units = np.unique(cl)
    t_c, motion, win = de.pooled_com_motion(
        data, st, cl, chan_y, 61, 20, fs, units,
        bin_s=10.0, spikes_per_bin=60)
    est = np.asarray(motion)[0]
    return t_c, est - est.mean(), win


def _truth_on(t_c, motion_um, duration_s=300.0):
    mt = np.arange(len(motion_um)) * duration_s / len(motion_um)
    tr = np.interp(t_c, mt, motion_um)
    return tr - tr.mean()


def _score(name, est, true):
    err = float(np.sqrt(np.mean((est - true) ** 2)))
    zero = float(np.sqrt(np.mean(true ** 2)))
    corr = float(np.corrcoef(est, true)[0, 1]) if np.ptp(true) > 0 else float("nan")
    slope = float(np.dot(est, true) / np.dot(true, true)) if np.dot(true, true) > 0 else 0.0
    print(f"  {name}: ptp {np.ptp(est):.2f} (true {np.ptp(true):.2f})  "
          f"corr {corr:+.3f}  slope {slope:+.3f}  rmse {err:.2f} (zero {zero:.2f})")
    return err, zero, corr, slope


def test_rigid_ramp():
    """A steady ramp, several channel rows wide."""
    true_um = 40.0 * np.linspace(0.0, 1.0, 30)
    t_c, est, _ = _run(true_um)
    true = _truth_on(t_c, true_um)
    err, zero, corr, slope = _score("ramp", est, true)
    assert corr > 0.95
    assert abs(slope - 1.0) < 0.25, f"amplitude is off: slope {slope:.3f}"
    assert err < 0.3 * zero


def test_sub_pitch_drift():
    """Drift SMALLER than the channel pitch must still be recovered.

    This is the case that broke the raster estimator this replaced: it binned
    amplitude onto a depth grid, channels landed exactly on grid points
    (15 um rows, 5 um bins), and the raster became a comb whose
    cross-correlation collapsed instead of decaying -- so it returned
    identically zero on a real recording drifting 6.5 um.

    A centroid has no grid, so it should not care. 8 um against a 15 um pitch
    is a multiple of neither the pitch nor the old 5 um bin, which is what
    makes it the discriminating case.
    """
    true_um = 8.0 * np.linspace(0.0, 1.0, 30)
    t_c, est, _ = _run(true_um)
    true = _truth_on(t_c, true_um)
    err, zero, corr, slope = _score("sub-pitch", est, true)
    assert np.ptp(est) > 0.4 * np.ptp(true), \
        f"estimate is flat -- the comb failure mode: ptp {np.ptp(est):.3f}"
    assert corr > 0.9
    assert err < 0.5 * zero


def test_step():
    """An abrupt jump, the shape a probe settling makes."""
    true_um = np.where(np.arange(30) < 15, 0.0, 25.0)
    t_c, est, _ = _run(true_um)
    true = _truth_on(t_c, true_um)
    err, zero, corr, slope = _score("step", est, true)
    assert corr > 0.9
    assert err < 0.4 * zero


def test_static_stays_static():
    """No motion must produce no motion.

    The failure this guards is the expensive one: an estimator that reports
    drift on a stationary probe makes a drift-aware calibrator segment a
    stationary population on measurement noise. That has already happened
    once here with a per-unit tracker, which read up to 71 um on a session
    with none.
    """
    true_um = np.zeros(30)
    t_c, est, _ = _run(true_um)
    print(f"  static: ptp {np.ptp(est):.3f} um, std {est.std():.3f}")
    assert np.ptp(est) < 3.0, f"invented {np.ptp(est):.2f} um of drift"


def test_motion_at_rigid_is_depth_independent():
    """With one window the field is rigid, so depth must not matter."""
    t_c = np.linspace(0.0, 100.0, 11)
    motion = np.linspace(0.0, 10.0, 11)[None, :]
    win = np.array([500.0])
    a = de.motion_at(t_c, 400.0, t_c, motion, win)
    b = de.motion_at(t_c, 900.0, t_c, motion, win)
    assert np.allclose(a, b)
    assert np.allclose(a, motion[0])


def test_motion_at_clamps_outside_windows():
    """Past the outermost window the field is held, not extrapolated.

    Extrapolating a motion field beyond the units that constrained it
    invents drift nothing measured.
    """
    t_c = np.array([0.0, 10.0])
    motion = np.array([[0.0, 4.0], [0.0, 12.0]])
    win = np.array([500.0, 900.0])
    lo = de.motion_at(t_c, 100.0, t_c, motion, win)
    hi = de.motion_at(t_c, 2000.0, t_c, motion, win)
    assert np.allclose(lo, motion[0])
    assert np.allclose(hi, motion[1])


def test_nonrigid_windows_separate():
    """Two depth windows must be able to disagree.

    Only the plumbing is checked here -- that per-window pooling routes units
    to windows by their own depth and returns one trajectory per window with
    matching centres. Whether a genuine depth-dependent motion field is
    recovered needs a synthetic recording with one, which this fixture's
    rigid drift is not.
    """
    true_um = 30.0 * np.linspace(0.0, 1.0, 30)
    data, st, cl, chan_y, fs = synth_recording(true_um)
    t_c, motion, win = de.pooled_com_motion(
        data, st, cl, chan_y, 61, 20, fs, np.unique(cl),
        bin_s=10.0, spikes_per_bin=60, n_windows=3, window_overlap=0.5)
    assert motion.shape == (3, t_c.size)
    assert win.size == 3
    assert np.all(np.diff(win) > 0), "window centres must ascend with depth"
    assert np.all(np.isfinite(motion)), "a window returned NaN"
    for w in range(3):
        assert np.ptp(motion[w]) > 0.4 * np.ptp(true_um), \
            f"window {w} is flat: ptp {np.ptp(motion[w]):.2f}"


if __name__ == "__main__":
    for fn in (test_rigid_ramp, test_sub_pitch_drift, test_step,
               test_static_stays_static, test_motion_at_rigid_is_depth_independent,
               test_motion_at_clamps_outside_windows, test_nonrigid_windows_separate):
        print(fn.__name__)
        fn()
    print("all passed")
