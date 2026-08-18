"""
gen_nms_fixture.py
====================

Numerical fixture for ClosedLoop/test_convolution_engine_nms_equivalence.cpp:
a synthetic matched-filter-like D signal (smooth autocorrelated noise plus a
handful of sharp true "spikes", same construction used to first catch the
windowed-NMS-vs-scipy discrepancy) and scipy.signal.find_peaks(distance=...)'s
actual peak list, to validate ConvolutionEngine's fixed candidate-vs-
candidate NMS logic against the real algorithm it's supposed to match.
"""
import argparse
import struct

import numpy as np
from scipy.signal import find_peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--min-sep", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    white = rng.standard_normal(args.n)
    kernel = np.exp(-np.linspace(0, 3, 15))
    D = np.convolve(white, kernel, mode="same")
    for t in rng.choice(args.n, 100, replace=False):
        if 100 < t < args.n - 100:
            D[t - 5:t + 5] += rng.uniform(3, 8)

    idx_expect, _ = find_peaks(D, distance=args.min_sep)

    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<ii", args.n, args.min_sep))
        fh.write(D.astype("<f8").tobytes())
        fh.write(struct.pack("<i", len(idx_expect)))
        fh.write(idx_expect.astype("<i8").tobytes())

    print(f"Wrote fixture to {args.out} (n={args.n}, min_sep={args.min_sep}, "
          f"n_peaks_expect={len(idx_expect)})")


if __name__ == "__main__":
    main()
