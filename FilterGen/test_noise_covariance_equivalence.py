"""
test_noise_covariance_equivalence.py
=====================================

Regression check for generate_filter.noise_covariance_vectorized() against
the original noise_covariance() -- run this after touching either function.
Not wired into a test runner (this repo has no pytest suite); run directly:

    python test_noise_covariance_equivalence.py

Exits nonzero if either case fails to match within tolerance.
"""

import sys
import time

import numpy as np

import generate_filter as gf


def main():
    rng = np.random.default_rng(0)

    # Many-short-segments case: same shape class as real calibration inputs
    # (n_ch=5, template_length=61).
    n_samples = 20000
    n_ch = 5
    template_length = 61
    template_offset = 20
    data = rng.standard_normal((n_samples, n_ch)).astype(np.float32)
    spike_times = np.sort(rng.choice(np.arange(200, n_samples - 200), size=40, replace=False))

    t0 = time.time()
    C_old = gf.noise_covariance(data, spike_times, template_length, template_offset, n_samples)
    t_old = time.time() - t0
    t0 = time.time()
    C_new = gf.noise_covariance_vectorized(data, spike_times, template_length, template_offset, n_samples)
    t_new = time.time() - t0
    ok1 = np.allclose(C_old, C_new, rtol=1e-4, atol=1e-6)
    print(f"many-short-segments: old={t_old*1000:.1f}ms new={t_new*1000:.1f}ms allclose={ok1}")

    # Long-spike-free-segment case -- the one noise_covariance_vectorized's
    # speedup actually targets (see its docstring).
    spike_times2 = np.array([300, 350, 19700, 19750])
    t0 = time.time()
    C_old2 = gf.noise_covariance(data, spike_times2, template_length, template_offset, n_samples)
    t_old2 = time.time() - t0
    t0 = time.time()
    C_new2 = gf.noise_covariance_vectorized(data, spike_times2, template_length, template_offset, n_samples)
    t_new2 = time.time() - t0
    ok2 = np.allclose(C_old2, C_new2, rtol=1e-4, atol=1e-6)
    print(f"long-segment case:   old={t_old2*1000:.1f}ms new={t_new2*1000:.1f}ms allclose={ok2}")

    if not (ok1 and ok2):
        print("MISMATCH -- noise_covariance_vectorized diverged from noise_covariance", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
