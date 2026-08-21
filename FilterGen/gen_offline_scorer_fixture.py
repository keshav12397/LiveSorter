"""
gen_offline_scorer_fixture.py
===============================

Numerical fixture for ClosedLoop/test_offline_scorer_equivalence.cpp:
takes ONE unit's already-fit channels+filter (from an existing
calibrate_all_units.py packed output dir), computes its held-out test-split
D-values + ALL candidate peaks (gf.filter_output + find_all_peaks, no
threshold gating) via the real, already-validated Python functions, and
dumps everything the C++ offline GPU scorer needs to reproduce the same
peak list: the filter taps/channels, the CAR channel group (raw ids, for
translating channels into CAR-group indices the same way
ImecFetchThreadGPU::fetchLoop() does), and the expected peak list.

Isolates OfflineScorer.cpp (streaming raw .bin -> Preprocessor ->
GpuConvolutionEngine with an all -infinity threshold) from the fit step
(already validated separately: LcmvFit, NoiseCovariance) -- this fixture
supplies an already-fit real filter directly.
"""
import argparse
import struct
import sys

import numpy as np

sys.path.insert(0, "C:/Users/kesha/OneDrive/Desktop/LiveSorter/FilterGen")
import generate_filter as gf
import threshold_sweep_real as tsr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--channel-map-json", required=True)
    ap.add_argument("--filters-dir", required=True,
                     help="An existing calibrate_all_units.py output dir (unit_ids.bin/channels.bin/filters.bin)")
    ap.add_argument("--unit-id", type=int, required=True)
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--min-sep", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    unit_ids = np.fromfile(f"{args.filters_dir}/unit_ids.bin", dtype="<i4")
    channels = np.fromfile(f"{args.filters_dir}/channels.bin", dtype="<i4").reshape(-1, args.n_channels)
    filters = np.fromfile(f"{args.filters_dir}/filters.bin", dtype="<f4").reshape(
        -1, args.template_length, args.n_channels)

    idx = int(np.where(unit_ids == args.unit_id)[0][0])
    sel_channels_raw = channels[idx]  # raw SpikeGLX ids
    f = filters[idx].astype(np.float32)  # (template_length, n_channels)

    rng = np.random.default_rng(args.seed)

    class _Args:
        pass
    load_args = _Args()
    load_args.ks_dir = args.ks_dir
    load_args.bin_path = args.bin_path
    load_args.meta_path = None
    load_args.channel_map_json = args.channel_map_json
    load_args.filter = True
    load_args.car = True
    load_args.fc = args.fc
    load_args.causal_highpass = True

    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(load_args, rng, dtype=np.float32)

    split_t = int(args.train_frac * data.shape[0])
    data_test = data[split_t:]

    sel = np.array([int(np.where(np_ch == c)[0][0]) for c in sel_channels_raw])  # indices into CAR group

    data_test_sel = data_test[:, sel]
    D_test = gf.filter_output(data_test_sel, f)
    peak_idx, peak_scores = tsr.find_all_peaks(D_test, args.min_sep)

    print(f"unit {args.unit_id}: sel(raw)={sel_channels_raw.tolist()} sel(idx)={sel.tolist()} "
          f"n_peaks={len(peak_idx)}")

    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<iiii", args.n_channels, args.template_length,
                              int(split_t), int(data.shape[0] - split_t)))
        fh.write(np.asarray(sel, dtype="<i4").tobytes())
        fh.write(f.astype("<f4").tobytes())
        fh.write(struct.pack("<i", len(peak_idx)))
        fh.write(np.asarray(peak_idx, dtype="<i8").tobytes())
        fh.write(np.asarray(peak_scores, dtype="<f8").tobytes())

    print(f"Wrote fixture to {args.out}")


if __name__ == "__main__":
    main()
