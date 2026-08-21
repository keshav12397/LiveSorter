"""
test_noise_cov_banded.py -- the correctness claim the banded refit rests on.

Background, because this design was arrived at by first getting it wrong.

A refit at a drifted position needs R for a channel selection that was not
known when the recording was scanned. `noise_covariance_from_lags` can
assemble R for any subset of whatever channels were scanned, so the obvious
move is to scan ONCE over the whole probe and share that array across every
unit. That does not work, and it was already known not to work before this
file existed:

    ClosedLoop/NoiseCovariance.h:11-14 -- "the original 'compute once for
    the whole probe' idea doesn't work (an all-clusters mask leaves 0 usable
    spike-free segments on a real, busy recording -- checked empirically
    before writing this)"

    generate_filter.py, in the single-target fit -- "masking out every
    single spike from every cluster ... can blanket the entire recording and
    leave no spike-free segments at all, even though most of those spikes
    are on channels far from our local channel subset"

The exclusion set has to stay PER UNIT -- its own target plus interferers,
about 6 clusters -- because that is what leaves any spike-free time at all,
and because "not noise" should mean "could contaminate THESE channels", not
"any spike anywhere on the probe".

So cheapness has to come from somewhere other than sharing. It comes from
scanning a BAND: for one unit, with that unit's own exclusion mask, scan a
neighbourhood of channels wider than its 5-channel selection. Drift moves the
selection by a few rows, so any post-drift selection inside the band can then
be assembled without touching the data again.

What these tests pin is that banding costs nothing in accuracy: R assembled
from a banded scan is EXACTLY the R a direct scan of those channels would
produce, for the same exclusion set. Not approximately -- exactly, because R
is block-Toeplitz and restricting channels only drops blocks (the same
argument test_noise_cov_subset.py makes for arbitrary subsets).

    python -m pytest test_noise_cov_banded.py -v
"""

import numpy as np
import pytest

import generate_filter as gf

L = 61
OFFSET = 20


def _fixture(n_samp=80000, n_ch=24, seed=0):
    """A recording with spatially structured noise and two 'units'.

    Noise is correlated across neighbouring channels, so the covariance has
    real off-diagonal structure at every lag. White noise would let an
    indexing bug pass.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, (n_samp, n_ch))
    # smear across channels so nearby ones covary
    k = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    data = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, white.T).T
    data = np.cumsum(data, axis=0) * 0.01 + white

    local_spikes = np.sort(rng.integers(300, n_samp - 300, size=250))
    far_spikes = np.sort(rng.integers(300, n_samp - 300, size=2000))
    return data, local_spikes, far_spikes


def test_banded_equals_direct_scan():
    """R from a banded scan == R from scanning only the selected channels.

    This is the whole basis for refitting without rescanning: the band is
    scanned once, and every selection inside it is assemble-only.
    """
    data, local, _ = _fixture()
    band = np.arange(6, 22)                 # 16-channel neighbourhood
    sel_in_band = np.array([3, 4, 5, 6, 7])  # indices WITHIN the band
    sel_global = band[sel_in_band]

    cov_band = gf.noise_cov_by_lag(data[:, band], local, L, OFFSET, data.shape[0])
    R_from_band = gf.noise_covariance_from_lags(cov_band, sel_in_band, L)

    R_direct = gf.noise_covariance_vectorized(data[:, sel_global], local, L,
                                              OFFSET, data.shape[0])

    assert R_from_band.shape == (L * 5, L * 5)
    assert np.allclose(R_from_band, R_direct, rtol=0, atol=1e-12), \
        f"max abs diff {np.max(np.abs(R_from_band - R_direct)):.3e}"


def test_band_covers_a_drifted_selection():
    """A selection that MOVED still assembles from the same banded scan.

    The point of the band is that the post-drift channels were not known when
    the scan ran. Two different selections, both inside the band, must each
    equal their own direct scan -- from ONE scan of the band.
    """
    data, local, _ = _fixture()
    band = np.arange(4, 20)
    cov_band = gf.noise_cov_by_lag(data[:, band], local, L, OFFSET, data.shape[0])

    for shift in (0, 2, 4):     # the unit drifts upward across rows
        sel_in_band = np.array([2, 3, 4, 5, 6]) + shift
        sel_global = band[sel_in_band]
        R_band = gf.noise_covariance_from_lags(cov_band, sel_in_band, L)
        R_direct = gf.noise_covariance_vectorized(data[:, sel_global], local, L,
                                                  OFFSET, data.shape[0])
        assert np.allclose(R_band, R_direct, rtol=0, atol=1e-12), \
            f"shift {shift}: max abs diff {np.max(np.abs(R_band - R_direct)):.3e}"


def test_exclusion_set_changes_R():
    """Per-unit and probe-wide exclusion give DIFFERENT R. Not interchangeable.

    Guards against the tempting shortcut being adopted quietly on the grounds
    that "it's all noise anyway". Excluding every cluster's spikes rather than
    just the local ones is a different estimate, and on a real recording it is
    also usually an impossible one -- see this module's docstring.
    """
    data, local, far = _fixture()
    sel = np.arange(8, 13)
    both = np.sort(np.concatenate([local, far]))

    R_local = gf.noise_covariance_vectorized(data[:, sel], local, L, OFFSET,
                                             data.shape[0])
    R_all = gf.noise_covariance_vectorized(data[:, sel], both, L, OFFSET,
                                           data.shape[0])
    assert not np.allclose(R_local, R_all), \
        "probe-wide and local exclusion produced identical R -- fixture too weak"


def test_probe_wide_exclusion_starves_on_dense_data():
    """The documented failure, reproduced small.

    At realistic density a probe-wide mask leaves too little spike-free time
    to estimate from, and the failure is silent rather than loud: it does not
    raise as long as ONE gap survives. generate_filter.noise_cov_by_lag warns
    in that regime; this pins that it does.
    """
    data, local, _ = _fixture()
    n = data.shape[0]
    # spikes on a near-regular grid, tighter than the blanking width, so the
    # union of masked intervals covers nearly everything
    dense = np.arange(300, n - 300, 130)

    with pytest.warns(RuntimeWarning, match="only"):
        gf.noise_cov_by_lag(data[:, 8:13], dense, L, OFFSET, n)


def test_narrow_band_is_not_silently_wrong():
    """A selection ESCAPING the band must not be assemblable.

    The band is an assumption -- that drift keeps the selection inside it --
    and an out-of-range index has to fail loudly rather than wrap or clamp
    into a plausible-looking wrong matrix.
    """
    data, local, _ = _fixture()
    band = np.arange(6, 12)         # only 6 channels
    cov_band = gf.noise_cov_by_lag(data[:, band], local, L, OFFSET, data.shape[0])

    with pytest.raises(IndexError):
        gf.noise_covariance_from_lags(cov_band, np.array([4, 5, 6, 7, 8]), L)
