"""
test_banded_refit.py -- the refit path produces the same filter a full fit
would, and fails loudly when it cannot.

The claim under test is equivalence, not approximation: refitting inside a
band must give bit-comparable results to selecting and solving over the whole
group, whenever the selection the full fit would make lies inside the band.
If that holds, "refit without rescanning" is a pure cost saving with no
accuracy cost, which is the entire premise.

    python -m pytest test_banded_refit.py -v
"""

import numpy as np
import pytest

import banded_refit as br
import generate_filter as gf

L = 61
OFFSET = 20
NCH = 5


def _rig(n_samp=60000, n_rows=24, pitch=15.0, y0=435.0, seed=0):
    """A small probe with two columns per row and spatially correlated noise."""
    rng = np.random.default_rng(seed)
    n_ch = n_rows * 2
    chan_y = np.repeat(y0 + pitch * np.arange(n_rows), 2)
    # Two columns per row, at different x -- as a real probe has them.
    # Without this the two channels sharing a row are EXACTLY degenerate,
    # select_channels ties, and the tie breaks by index order, which differs
    # between band-relative and group-relative index spaces. That made an
    # earlier version of this test fail on a fixture artifact rather than on
    # anything the band does. See refit_in_band's note on ties.
    chan_x = np.tile(np.array([27.0, 59.0]), n_rows)
    white = rng.normal(0.0, 1.0, (n_samp, n_ch))
    k = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    data = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, white.T).T
    data = data + white
    return data, chan_y, chan_x, rng


def _templates(chan_y, chan_x, center_um, rng, n_int=2, spread=22.0,
               x_center=43.0):
    """A target waveform plus interferers, as full-group arrays."""
    tt = np.arange(L) - OFFSET
    shape = -np.exp(-0.5 * (tt / 3.0) ** 2) + 0.4 * np.exp(-0.5 * ((tt - 8) / 5.0) ** 2)
    def wf(c, amp, xc):
        d2 = (chan_y - c) ** 2 + (chan_x - xc) ** 2
        return shape[:, None] * (amp * np.exp(-0.5 * d2 / spread ** 2))[None, :]
    target = wf(center_um, 30.0, x_center)
    ints = [wf(center_um + off, 18.0, x_center + xo)
            for off, xo in zip(rng.uniform(-40, 40, size=n_int),
                               rng.uniform(-20, 20, size=n_int))]
    return target, ints


def test_banded_refit_matches_full_group_fit():
    """Same selection, same filter -- band vs whole group."""
    data, chan_y, chan_x, rng = _rig()
    center = float(np.median(chan_y))
    target, ints = _templates(chan_y, chan_x, center, rng)
    spikes = np.sort(rng.integers(300, data.shape[0] - 300, size=200))

    band = br.band_for_depth(chan_y, center, half_width_um=60.0)
    cov_band = br.scan_band(data, band, spikes, L, OFFSET)
    f_band, sel_global, _ = br.refit_in_band(cov_band, band, target, ints,
                                             NCH, L)

    # the reference: select over the whole group, scan those channels, solve
    sel_full = np.asarray(gf.select_channels(target, ints, NCH), dtype=int)
    assert set(sel_full.tolist()) <= set(band.tolist()), \
        "fixture invalid: the full-group selection is not inside the band"

    R_full = gf.noise_covariance_vectorized(data[:, sel_full], spikes, L,
                                            OFFSET, data.shape[0])
    s_flat = target[:, sel_full].T.ravel()
    int_flats = [w[:, sel_full].T.ravel() for w in ints]
    f_full = gf.lcmv_filter(s_flat, int_flats, R_full, ridge=1e-3)
    f_full = f_full.reshape(sel_full.size, L).T

    assert np.array_equal(np.sort(sel_global), np.sort(sel_full)), \
        f"different channels: band {sorted(sel_global)} vs full {sorted(sel_full)}"
    # order can differ (selection is by score within different index spaces),
    # so compare column-wise after matching channels
    order_b = np.argsort(sel_global)
    order_f = np.argsort(sel_full)
    assert np.allclose(f_band[:, order_b], f_full[:, order_f], atol=1e-10), \
        f"max diff {np.max(np.abs(f_band[:, order_b] - f_full[:, order_f])):.3e}"


def test_refit_after_drift_uses_the_same_scan():
    """One scan, several positions -- the actual online case.

    The unit moves; each new position selects different channels; every refit
    reads the same banded covariance and never touches the recording.
    """
    data, chan_y, chan_x, rng = _rig()
    center = float(np.median(chan_y))
    spikes = np.sort(rng.integers(300, data.shape[0] - 300, size=200))
    band = br.band_for_depth(chan_y, center, half_width_um=75.0)
    cov_band = br.scan_band(data, band, spikes, L, OFFSET)

    seen = []
    for shift in (-30.0, 0.0, 30.0):
        target, ints = _templates(chan_y, chan_x, center + shift, rng)
        f, sel_global, _ = br.refit_in_band(cov_band, band, target, ints, NCH, L)
        assert f.shape == (L, NCH)
        assert np.all(np.isfinite(f))
        seen.append(tuple(sorted(sel_global.tolist())))

    assert len(set(seen)) > 1, \
        "channel selection never changed -- fixture does not exercise drift"


def test_unity_gain_and_nulls_hold():
    """The LCMV constraints must survive going through the band.

    A malformed R -- wrong block transpose, mismatched channel order -- can
    still be symmetric and positive-definite and still solve. The constraints
    are what actually detects it.
    """
    data, chan_y, chan_x, rng = _rig()
    center = float(np.median(chan_y))
    target, ints = _templates(chan_y, chan_x, center, rng)
    spikes = np.sort(rng.integers(300, data.shape[0] - 300, size=200))

    band = br.band_for_depth(chan_y, center, half_width_um=60.0)
    cov_band = br.scan_band(data, band, spikes, L, OFFSET)
    f, sel_global, sel_in_band = br.refit_in_band(cov_band, band, target, ints,
                                                  NCH, L)

    f_flat = f.T.ravel()
    s_flat = target[:, band][:, sel_in_band].T.ravel()
    assert abs(float(np.dot(f_flat, s_flat)) - 1.0) < 1e-6, "unity gain lost"
    for i, w in enumerate(ints):
        wi = w[:, band][:, sel_in_band].T.ravel()
        assert abs(float(np.dot(f_flat, wi))) < 1e-6, f"interferer {i} not nulled"


def test_band_centres_on_the_unit_near_an_edge():
    """A unit near the probe end still gets min_channels, still centred.

    Widening blindly would push the band toward whichever side has more
    channels, quietly biasing which interferers are visible to selection.
    """
    _, chan_y, _, _ = _rig()
    top = float(np.max(chan_y))
    band = br.band_for_depth(chan_y, top, half_width_um=5.0, min_channels=12)
    assert band.size >= 12
    # every channel in the band is among the 12 nearest to `top`
    nearest = set(np.argsort(np.abs(chan_y - top), kind="stable")[:band.size].tolist())
    assert set(band.tolist()) == nearest


def test_selection_escaping_the_band_raises():
    """Too narrow a band must fail, not silently return a lesser filter."""
    data, chan_y, chan_x, rng = _rig()
    center = float(np.median(chan_y))
    target, ints = _templates(chan_y, chan_x, center, rng)
    spikes = np.sort(rng.integers(300, data.shape[0] - 300, size=200))

    band = br.band_for_depth(chan_y, center, half_width_um=1.0, min_channels=3)
    cov_band = br.scan_band(data, band, spikes, L, OFFSET)
    with pytest.raises(ValueError, match="band holds only"):
        br.refit_in_band(cov_band, band, target, ints, NCH, L)
