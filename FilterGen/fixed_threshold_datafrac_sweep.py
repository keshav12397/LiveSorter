"""
fixed_threshold_datafrac_sweep.py
==================================

How much calibration data does the LCMV filter actually need? Fits on just
the first 5%, 10%, and 25% of the recording (by time), evaluates each on
its own held-out remainder, all at one FIXED threshold (rather than
re-sweeping per split) -- so the comparison isolates the effect of
training-data amount, not a re-optimized threshold hiding it.

Loads the (large) raw recording only once and reuses it across all three
fits/evaluations.
"""

import argparse
import os
import json

import numpy as np
from scipy.signal import find_peaks

import generate_filter as gf
import threshold_sweep_real as tsr  # reuse load_and_prepare / fit_lcmv / match_within_window


def evaluate_at_threshold(data, spike_t, spike_cl, np_ch, fs, target, interferer_ids,
                           n_channels, template_length, template_offset, ridge, max_spikes,
                           rng, train_frac, threshold):
    split_t = int(train_frac * data.shape[0])
    data_train, data_test = data[:split_t], data[split_t:]

    print(f"\n{'=' * 60}\ntrain_frac={train_frac}  "
          f"(train: {split_t/fs:.1f}s, test: {(data.shape[0]-split_t)/fs:.1f}s)")

    f, sel, sel_channels, _, _ = tsr.fit_lcmv(
        data_train, spike_t[spike_t < split_t], spike_cl[spike_t < split_t], np_ch,
        target, interferer_ids, n_channels, template_length, template_offset,
        ridge, max_spikes, rng)
    print(f"Selected channels: {sel_channels.tolist()}")

    target_all = spike_t[spike_cl == target]
    target_train = target_all[target_all < split_t]
    target_test = target_all[target_all >= split_t]
    interferer_test = {
        cid: (spike_t[spike_cl == cid])[spike_t[spike_cl == cid] >= split_t]
        for cid in interferer_ids
    }
    print(f"Target: {len(target_train)} train spikes, {len(target_test)} test spikes")

    data_test_sel = data_test[:, sel]
    D_test = gf.filter_output(data_test_sel, f)

    window = template_length // 4
    min_sep = template_length // 2
    detections, _ = find_peaks(D_test, height=threshold, distance=min_sep)
    print(f"{len(detections)} detections at threshold={threshold}")

    target_test_rel = np.sort(target_test - split_t)
    n_target_test = len(target_test_rel)

    hit_mask = tsr.match_within_window(target_test_rel, detections, window)
    hits = int(hit_mask.sum())
    misses = n_target_test - hits

    is_target_match = tsr.match_within_window(detections, target_test_rel, window)
    is_fp = ~is_target_match
    total_fp = int(is_fp.sum())

    fp_by_interferer = {}
    for cid, times in interferer_test.items():
        times_rel = np.sort(times - split_t)
        match = tsr.match_within_window(detections, times_rel, window)
        fp_by_interferer[cid] = int((match & is_fp).sum())

    test_duration_s = (data.shape[0] - split_t) / fs
    recall = hits / max(n_target_test, 1)
    precision = hits / max(hits + total_fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    fp_rate_hz = total_fp / test_duration_s

    result = dict(train_frac=train_frac, threshold=threshold,
                  train_s=split_t / fs, test_s=test_duration_s,
                  n_target_train=len(target_train), n_target_test=n_target_test,
                  hits=hits, misses=misses, recall=recall, precision=precision, f1=f1,
                  total_fp=total_fp, fp_rate_hz=fp_rate_hz,
                  fp_by_interferer=fp_by_interferer,
                  selected_channels=sel_channels.tolist())

    print(f"recall={recall:.2%}  precision={precision:.2%}  f1={f1:.3f}  "
          f"FP_rate={fp_rate_hz:.3f}/s  (total_fp={total_fp} over {test_duration_s:.1f}s)")
    for cid, fp in fp_by_interferer.items():
        print(f"  false positives coinciding with interferer {cid}: {fp}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path")
    ap.add_argument("--meta-path")
    ap.add_argument("--channel-map-json")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--interferers", type=int, nargs="+", required=True)
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--train-fracs", type=float, nargs="+", default=[0.05, 0.10, 0.25])
    ap.add_argument("--threshold", type=float, default=0.75057)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-spikes", type=int, default=2000)
    ap.add_argument("--filter", action="store_true", default=True)
    ap.add_argument("--no-filter", dest="filter", action="store_false")
    ap.add_argument("--car", action="store_true", default=True)
    ap.add_argument("--no-car", dest="car", action="store_false")
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data (once, reused across all train fractions)...")
    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(args, rng)

    results = []
    for frac in args.train_fracs:
        r = evaluate_at_threshold(
            data, spike_t, spike_cl, np_ch, fs, args.target, args.interferers,
            args.n_channels, args.template_length, args.template_offset,
            args.ridge, args.max_spikes, rng, frac, args.threshold)
        results.append(r)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY (fixed threshold={args.threshold})")
    print(f"{'=' * 60}")
    print(f"{'train_frac':>10} {'train_s':>8} {'test_s':>8} {'n_train':>8} "
          f"{'n_test':>7} {'recall':>8} {'precision':>10} {'f1':>6} {'fp/s':>10}")
    for r in results:
        print(f"{r['train_frac']:>10.2f} {r['train_s']:>8.1f} {r['test_s']:>8.1f} "
              f"{r['n_target_train']:>8d} {r['n_target_test']:>7d} "
              f"{r['recall']:>8.2%} {r['precision']:>10.2%} {r['f1']:>6.3f} {r['fp_rate_hz']:>10.4f}")

    out_json = os.path.join(args.out_dir, f"datafrac_sweep_{args.target}.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()
