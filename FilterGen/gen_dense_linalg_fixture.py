"""
gen_dense_linalg_fixture.py
============================

Generates a small numerical fixture (random SPD matrix + RHS vector/matrix,
solved via numpy) for ClosedLoop/test_dense_linalg_equivalence.cpp to check
CholeskySolver (DenseLinAlg.h) against -- the same "port validated against
Python" pattern as the rest of this session's equivalence tests, just for
the new C++ calibration path's linear-algebra core instead of a signal-
processing function.

Binary layout (all float64, C order), written to --out:
    int32 n, int32 m
    A        (n, n)
    b        (n,)
    B        (n, m)
    x_expect (n,)      -- np.linalg.solve(A, b)
    X_expect (n, m)    -- np.linalg.solve(A, B)
"""
import argparse
import struct
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    M = rng.standard_normal((args.n, args.n))
    A = M @ M.T + args.n * np.eye(args.n)  # guaranteed SPD, well-conditioned
    b = rng.standard_normal(args.n)
    B = rng.standard_normal((args.n, args.m))

    x_expect = np.linalg.solve(A, b)
    X_expect = np.linalg.solve(A, B)

    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<ii", args.n, args.m))
        fh.write(A.astype("<f8").tobytes())
        fh.write(b.astype("<f8").tobytes())
        fh.write(B.astype("<f8").tobytes())
        fh.write(x_expect.astype("<f8").tobytes())
        fh.write(X_expect.astype("<f8").tobytes())

    print(f"Wrote fixture to {args.out} (n={args.n}, m={args.m})")


if __name__ == "__main__":
    main()
