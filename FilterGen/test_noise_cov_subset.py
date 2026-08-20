"""
test_noise_cov_subset.py -- the property online refitting rests on.

`noise_cov_by_lag` scans the recording once over EVERY channel;
`noise_covariance_from_lags` assembles R for whatever subset a filter ended
up using. Refitting a drifted unit onto different channels is then an
assembly plus a 305x305 solve (~2.4 ms measured) instead of another pass
over the recording (~286 s for 400k samples x 96 channels).

That only holds if the subset assembled from the full-group array is the
SAME matrix a scan restricted to those channels would produce. It is --
exactly, not approximately, because R is block-Toeplitz and block (i, k)
reads only channel pair (i, k)'s covariance at each lag, so restricting the
channel set just drops blocks. These tests pin that, since the whole refit
design silently becomes wrong if it ever stops being true.

    python -m pytest test_noise_cov_subset.py -v
"""

import numpy as np
import pytest

import generate_filter as gf

L = 61
OFFSET = 20


def _fixture(n_samp=60000, n_ch=12, n_spikes=400, seed=0):
    rng = np.random.default_rng(seed)
    # Correlated across channels and coloured in time, so the covariance has
    # real off-diagonal and off-lag structure. White noise would pass these
    # tests while hiding an indexing bug in the cross terms.
    white = rng.normal(0.0, 1.0, (n_samp, n_ch))
    mix = rng.normal(0.0, 1.0, (n_ch, n_ch))
    data = np.cumsum(white @ mix, axis=0) * 0.01 + white
    spikes = np.sort(rng.integers(200, n_samp - 200, size=n_spikes))
    return data, spikes


@pytest.mark.parametrize("sel", [
    [0, 1, 2, 3, 4],          # contiguous, the usual case
    [11, 7, 2, 9, 0],         # unsorted -- selection is by amplitude, not order
    [5],                      # single channel
    [0, 11],                  # the two ends
])
def test_subset_matches_direct_scan(sel):
    """R assembled from the full-group lags == R from scanning only `sel`."""
    data, spikes = _fixture()
    sel = np.array(sel)

    cov_full = gf.noise_cov_by_lag(data, spikes, L, OFFSET, data.shape[0])
    R_subset = gf.noise_covariance_from_lags(cov_full, sel, L)
    R_direct = gf.noise_covariance_vectorized(data[:, sel], spikes, L, OFFSET,
                                              data.shape[0])

    assert R_subset.shape == (L * sel.size, L * sel.size)
    assert np.allclose(R_subset, R_direct, rtol=0, atol=1e-12), \
        f"max abs diff {np.max(np.abs(R_subset - R_direct)):.3e}"


def test_channel_order_is_respected():
    """Reordering `sel` must permute R's blocks, not scramble them.

    Channel selection comes back in amplitude order, not ascending channel
    order, so an implementation that quietly sorted `sel` would pair each
    filter tap with the wrong wire -- and would still produce a
    symmetric, positive-definite, entirely plausible-looking matrix.
    """
    data, spikes = _fixture()
    cov_full = gf.noise_cov_by_lag(data, spikes, L, OFFSET, data.shape[0])

    a = np.array([2, 7, 5])
    b = a[::-1]
    Ra = gf.noise_covariance_from_lags(cov_full, a, L)
    Rb = gf.noise_covariance_from_lags(cov_full, b, L)

    perm = np.concatenate([np.arange(L) + i * L for i in (2, 1, 0)])
    assert np.allclose(Rb, Ra[np.ix_(perm, perm)])
    assert not np.allclose(Ra, Rb), "fixture too symmetric to detect a scramble"


def test_vectorized_wrapper_unchanged():
    """The original entry point still returns what it always did.

    noise_covariance_vectorized is now the two halves composed. Callers
    across three calibration scripts use it, and the split must be invisible
    to them.
    """
    data, spikes = _fixture(n_ch=6)
    R = gf.noise_covariance_vectorized(data, spikes, L, OFFSET, data.shape[0])
    cov = gf.noise_cov_by_lag(data, spikes, L, OFFSET, data.shape[0])
    R2 = gf.noise_covariance_from_lags(cov, np.arange(6), L)
    assert np.allclose(R, R2)
    assert np.allclose(R, R.T), "R must be symmetric"


def test_r_is_usable_by_lcmv():
    """An assembled subset R must actually solve, and honour the constraints.

    Guards the failure where R comes out subtly malformed -- wrong block
    transpose, say -- and lcmv_filter returns weights that satisfy nothing.
    Unity gain on target and near-zero on each interferer is the contract.
    """
    data, spikes = _fixture()
    rng = np.random.default_rng(1)
    cov = gf.noise_cov_by_lag(data, spikes, L, OFFSET, data.shape[0])
    sel = np.array([3, 8, 1, 6, 0])
    R = gf.noise_covariance_from_lags(cov, sel, L)

    dim = L * sel.size
    s_flat = rng.normal(0, 1, dim)
    interferers = [rng.normal(0, 1, dim) for _ in range(3)]
    f = gf.lcmv_filter(s_flat, interferers, R, ridge=1e-3)

    assert abs(float(np.dot(f, s_flat)) - 1.0) < 1e-6, "unity-gain constraint violated"
    for i, s_int in enumerate(interferers):
        assert abs(float(np.dot(f, s_int))) < 1e-6, f"interferer {i} not nulled"
