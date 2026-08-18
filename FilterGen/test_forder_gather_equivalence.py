"""
test_forder_gather_equivalence.py
==================================

Regression check for load_and_prepare's order='F' option
(threshold_sweep_real.py) -- confirms a Fortran-order scratch memmap reads
back byte-for-byte identical data to the default 'C'-order one (only the
on-disk layout differs, never the values), and demonstrates the per-unit
channel-subset gather speedup that's the whole reason calibrate_all_units.py
requests order='F' (each unit gathers a different small channel subset out
of the same shared recording -- see load_and_prepare's docstring/comments).

This uses a smaller array than the real ~20M-sample-per-half recording (so
it runs in seconds, not minutes) -- the *speedup ratio* from avoiding
touching every other channel's interleaved bytes doesn't depend on the
array being that large, only the *magnitude* of the win does (much bigger
on the real multi-GB file, ~12.5x measured there -- see the all_units
branch commit history for that benchmark).

Run directly:

    python test_forder_gather_equivalence.py
"""

import os
import sys
import tempfile
import time

import numpy as np


def main():
    n_samples = 2_000_000
    n_channels = 96
    n_sel = 5
    dtype = np.float32
    rng = np.random.default_rng(0)

    scratch = tempfile.mkdtemp(prefix="forder_test_")
    path_c = os.path.join(scratch, "c_order.f32")
    path_f = os.path.join(scratch, "f_order.f32")

    source = rng.standard_normal((n_samples, n_channels)).astype(dtype)

    for path, order in [(path_c, "C"), (path_f, "F")]:
        m = np.memmap(path, dtype=dtype, mode="w+", shape=(n_samples, n_channels), order=order)
        m[:] = source
        m.flush()
        del m  # release the file handle (Windows won't allow delete-while-open)

    sel = np.array([3, 43, 47, 70, 90])

    m_c = np.memmap(path_c, dtype=dtype, mode="r", shape=(n_samples, n_channels), order="C")
    m_f = np.memmap(path_f, dtype=dtype, mode="r", shape=(n_samples, n_channels), order="F")

    t0 = time.time()
    sub_c = np.asarray(m_c[:, sel])
    t_c = time.time() - t0

    t0 = time.time()
    sub_f = np.asarray(m_f[:, sel])
    t_f = time.time() - t0

    equal = np.array_equal(sub_c, sub_f)
    print(f"C-order gather: {t_c*1000:.2f}ms   F-order gather: {t_f*1000:.2f}ms   "
          f"speedup: {t_c/max(t_f,1e-9):.1f}x")
    print(f"data identical: {equal}")

    # Windows won't let a file be deleted while a memmap still holds it
    # open (unlike POSIX) -- drop every reference and force a GC pass first.
    del m_c, m_f, sub_c, sub_f, source
    import gc
    gc.collect()

    os.remove(path_c)
    os.remove(path_f)
    os.rmdir(scratch)

    if not equal:
        print("MISMATCH -- F-order and C-order returned different data", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
