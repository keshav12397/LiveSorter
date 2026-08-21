"""
gen_lcmv_fixture.py
=====================

Numerical fixture for ClosedLoop/test_lcmv_equivalence.cpp: calls the real,
already-validated generate_filter.lcmv_filter() on a synthetic-but-realistic
R/target/interferer setup, dumps everything the C++ port needs to reproduce
the same call, and the expected output -- validates lcmvFilter()'s assembly
logic (building C, ridge-regularizing R, solving for weights) against
Python bit for bit, decoupled from mean_waveform/select_channels (which get
their own validation via a real end-to-end run) and from the noise-
covariance estimate itself (NoiseCovariance.cpp -- R here is
a synthetic stand-in with realistic SPD structure).

Binary layout (float64, C order), written to --out:
    int32 dim, int32 nInterferers
    R            (dim, dim)
    sFlat        (dim,)
    interfererFlats  (nInterferers, dim)
    ridge        (1,)
    fFlat_expect (dim,)          -- generate_filter.lcmv_filter(...)
    gains_expect (1+nInterferers,)  -- dot(f_flat, sFlat), dot(f_flat, each interferer)
"""
import argparse
import struct
import sys

import numpy as np

sys.path.insert(0, "C:/Users/kesha/OneDrive/Desktop/LiveSorter/FilterGen")
import generate_filter as gf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--n-interferers", type=int, default=2)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dim = args.template_length * args.n_channels
    rng = np.random.default_rng(args.seed)

    # Realistic SPD structure: R should look like a noise covariance (well-
    # conditioned, not adversarial) -- same construction as
    # gen_dense_linalg_fixture.py's fixture.
    M = rng.standard_normal((dim, dim))
    R = M @ M.T + dim * np.eye(dim)

    s_flat = rng.standard_normal(dim)
    interferer_flats = [rng.standard_normal(dim) for _ in range(args.n_interferers)]

    f_flat = gf.lcmv_filter(s_flat, interferer_flats, R, ridge=args.ridge)

    gains = [float(np.dot(f_flat, s_flat))]
    for it in interferer_flats:
        gains.append(float(np.dot(f_flat, it)))

    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<ii", dim, args.n_interferers))
        fh.write(R.astype("<f8").tobytes())
        fh.write(s_flat.astype("<f8").tobytes())
        for it in interferer_flats:
            fh.write(it.astype("<f8").tobytes())
        fh.write(struct.pack("<d", args.ridge))
        fh.write(f_flat.astype("<f8").tobytes())
        fh.write(np.array(gains, dtype="<f8").tobytes())

    print(f"Wrote fixture to {args.out} (dim={dim}, nInterferers={args.n_interferers})")
    print(f"gains (target should be ~1.0, interferers ~0.0): {gains}")


if __name__ == "__main__":
    main()
