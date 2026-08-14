"""
threshold_sweep_real.py
========================

Fit the LCMV filter on the FIRST train_frac of a real recording (by time,
no leakage), evaluate on the held-out remainder, and sweep the detection
threshold to build false-positive / false-negative curves -- so you can
pick an operating point instead of trusting a single fixed formula.

Uses matplotlib's non-interactive Agg backend and saves a PNG (does not
call plt.show(), which would block forever in a headless/background run).
"""

import argparse
import os

import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_filter as gf


def load_and_prepare(args, rng):
    spike_t, spike_cl, labels = gf.load_kilosort(args.ks_dir)
    params = gf.load_params_py(args.ks_dir)

    bin_path = args.bin_path or params.get("dat_path")
    bin_path = os.path.join(args.ks_dir, bin_path) if not os.path.isabs(bin_path) else bin_path
    meta_path = args.meta_path or (os.path.splitext(bin_path)[0] + ".meta")
    meta = gf.load_sglx_meta(meta_path)

    fs = float(meta["imSampRate"])
    n_saved_chans = int(meta["nSavedChans"])
    np_ch = np.array(gf.parse_chan_subset(meta["snsSaveChanSubset"]))
    n_sync = gf.find_sync_channel_count(meta)

    keep_mask = np.ones(len(np_ch), dtype=bool)
    if n_sync:
        keep_mask[-n_sync:] = False
    if args.channel_map_json:
        shank_chans = gf.load_channel_map_json(args.channel_map_json)
        keep_mask &= np.isin(np_ch, shank_chans)
    keep_idx = np.where(keep_mask)[0]
    np_ch = np_ch[keep_idx]

    raw = gf.memmap_raw(bin_path, n_saved_chans)
    last_spike = spike_t.max()
    t_max_samples = int(min(raw.shape[0], last_spike + fs))
    print(f"Loading {t_max_samples} samples x {len(keep_idx)} channels...")
    data = np.asarray(raw[:t_max_samples, keep_idx], dtype=np.float64)
    print(f"Loaded ({data.nbytes / 1e9:.1f} GB)")

    if args.filter:
        data = gf.highpass(data, args.fc, fs)
    if args.car:
        data = gf.common_average_reference(data)

    return spike_t, spike_cl, data, np_ch, fs


def fit_lcmv(data, spike_t, spike_cl, np_ch, target, interferer_ids,
             n_channels, template_length, template_offset, ridge, max_spikes, rng):
    target_spikes = spike_t[spike_cl == target]
    interferer_times = [spike_t[spike_cl == cid] for cid in interferer_ids]

    target_wf, _ = gf.mean_waveform(data, target_spikes, template_length,
                                     template_offset, max_spikes, rng)
    interferer_wfs = [gf.mean_waveform(data, t, template_length, template_offset,
                                        max_spikes, rng)[0] for t in interferer_times]

    sel = gf.select_channels(target_wf, interferer_wfs, n_channels)
    sel_channels = np_ch[sel]
    data_sel = data[:, sel]
    s = target_wf[:, sel]
    interferer_wfs_sel = [wf[:, sel] for wf in interferer_wfs]

    local_spike_times = np.sort(np.concatenate([target_spikes] + interferer_times))
    R = gf.noise_covariance(data_sel, local_spike_times, template_length,
                             template_offset, data_sel.shape[0])

    s_flat = s.T.ravel()
    interferer_flats = [wf.T.ravel() for wf in interferer_wfs_sel]
    f_flat = gf.lcmv_filter(s_flat, interferer_flats, R, ridge=ridge)
    f = f_flat.reshape(len(sel), template_length).T

    print(f"Constraint check -- gain on target: {np.dot(f_flat, s_flat):.4f} (want 1.0)")
    for cid, s_int in zip(interferer_ids, interferer_flats):
        print(f"  gain on interferer {cid}: {np.dot(f_flat, s_int):.4e} (want ~0)")

    return f, sel, sel_channels, target_spikes, interferer_times


def find_all_peaks(D, min_sep, min_height=None):
    """All local maxima in D, at least min_sep samples apart. Uses scipy's
    C-implemented find_peaks (with `distance`) instead of a hand-rolled
    Python non-max-suppression loop -- for a real ~10-20M-sample test split,
    a plain Python loop over the (likely tens-of-millions) raw local-maxima
    candidates would be far too slow. min_height lets the caller discard
    obviously-irrelevant low-amplitude noise peaks up front to shrink the
    candidate set before the sweep, without affecting the sweep's own
    (higher) threshold range.
    """
    idx, _ = find_peaks(D, distance=min_sep, height=min_height)
    return idx, D[idx]


def match_within_window(query, reference_sorted, window):
    """For each point in `query`, is there a point in `reference_sorted`
    (already sorted ascending) within +/- window? Fully vectorized via
    searchsorted -- O(log n) per query point, no Python-level loop over
    potentially tens of thousands of spike times per threshold."""
    if len(reference_sorted) == 0 or len(query) == 0:
        return np.zeros(len(query), dtype=bool)
    idx = np.searchsorted(reference_sorted, query)
    idx_right = np.clip(idx, 0, len(reference_sorted) - 1)
    idx_left = np.clip(idx - 1, 0, len(reference_sorted) - 1)
    dist_right = np.abs(reference_sorted[idx_right] - query)
    dist_left = np.abs(reference_sorted[idx_left] - query)
    return (dist_right <= window) | (dist_left <= window)


def sweep_thresholds(peak_idx, peak_scores, target_test, interferer_test, window,
                      n_thresholds=60):
    lo, hi = np.percentile(peak_scores, [50, 99.99])
    thresholds = np.linspace(lo, hi, n_thresholds)

    target_test = np.sort(target_test)
    interferer_test = {cid: np.sort(t) for cid, t in interferer_test.items()}
    n_target = len(target_test)

    results = []
    for th in thresholds:
        detections = peak_idx[peak_scores > th]  # already sorted (peak_idx is ascending)

        # hits: true target spikes with >=1 detection nearby (query direction
        # reversed vs. false-positive check below -- same helper, either way)
        hit_mask = match_within_window(target_test, detections, window)
        hits = int(hit_mask.sum())
        misses = n_target - hits

        # false positives: detections not matching any target spike
        is_target_match = match_within_window(detections, target_test, window)
        is_fp = ~is_target_match
        total_fp = int(is_fp.sum())

        fp_by_interferer = {}
        for cid, times in interferer_test.items():
            match = match_within_window(detections, times, window)
            fp_by_interferer[cid] = int((match & is_fp).sum())

        results.append(dict(threshold=th, hits=hits, misses=misses,
                             total_fp=total_fp, fp_by_interferer=fp_by_interferer,
                             n_detections=len(detections)))
    return results


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
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-spikes", type=int, default=2000)
    ap.add_argument("--filter", action="store_true", default=True)
    ap.add_argument("--no-filter", dest="filter", action="store_false")
    ap.add_argument("--car", action="store_true", default=True)
    ap.add_argument("--no-car", dest="car", action="store_false")
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--n-thresholds", type=int, default=60)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    spike_t, spike_cl, data, np_ch, fs = load_and_prepare(args, rng)

    split_t = int(args.train_frac * data.shape[0])
    print(f"\nTrain: [0, {split_t}) ({split_t/fs:.1f}s)   "
          f"Test: [{split_t}, {data.shape[0]}) ({(data.shape[0]-split_t)/fs:.1f}s)\n")

    data_train, data_test = data[:split_t], data[split_t:]

    def split(times):
        return times[times < split_t], times[times >= split_t]

    print("=" * 60)
    print("Fitting LCMV on TRAIN split only...")
    f, sel, sel_channels, target_spikes, interferer_times = fit_lcmv(
        data_train, spike_t[spike_t < split_t], spike_cl[spike_t < split_t], np_ch,
        args.target, args.interferers, args.n_channels,
        args.template_length, args.template_offset, args.ridge, args.max_spikes, rng)
    print(f"Selected channels: {sel_channels.tolist()}\n")

    target_train, target_test = split(spike_t[spike_cl == args.target])
    interferer_test = {cid: split(spike_t[spike_cl == cid])[1] for cid in args.interferers}
    print(f"Target: {len(target_train)} train spikes, {len(target_test)} test spikes")
    for cid in args.interferers:
        print(f"Interferer {cid}: {len(interferer_test[cid])} test spikes")
    print()

    data_test_sel = data_test[:, sel]
    print("Computing filter output on held-out test split...")
    D_test = gf.filter_output(data_test_sel, f)

    window = args.template_length // 4
    min_sep = args.template_length // 2
    print("Finding all candidate detections (local maxima) in test output...")
    peak_idx, peak_scores = find_all_peaks(D_test, min_sep)
    print(f"Found {len(peak_idx)} candidate peaks in test split.\n")

    print(f"Sweeping {args.n_thresholds} threshold values...")
    target_test_rel = target_test - split_t
    interferer_test_rel = {cid: t - split_t for cid, t in interferer_test.items()}
    results = sweep_thresholds(peak_idx, peak_scores, target_test_rel,
                                interferer_test_rel, window, args.n_thresholds)

    # ---- Summarize + pick optimal operating points -------------------------
    thresholds = np.array([r["threshold"] for r in results])
    hits = np.array([r["hits"] for r in results])
    misses = np.array([r["misses"] for r in results])
    total_fp = np.array([r["total_fp"] for r in results])
    n_target_test = len(target_test)
    test_duration_s = (data.shape[0] - split_t) / fs

    recall = hits / max(n_target_test, 1)
    precision = hits / np.maximum(hits + total_fp, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    fp_rate_hz = total_fp / test_duration_s

    best_f1_idx = np.argmax(f1)
    print(f"\nBest F1 operating point: threshold={thresholds[best_f1_idx]:.4f}  "
          f"recall={recall[best_f1_idx]:.2%}  precision={precision[best_f1_idx]:.2%}  "
          f"F1={f1[best_f1_idx]:.3f}  FP_rate={fp_rate_hz[best_f1_idx]:.2f}/s")

    # highest-recall threshold that keeps FP rate under 1/s, if any
    under_1hz = np.where(fp_rate_hz <= 1.0)[0]
    if under_1hz.size:
        idx = under_1hz[np.argmax(recall[under_1hz])]
        print(f"Best recall with FP rate <= 1/s: threshold={thresholds[idx]:.4f}  "
              f"recall={recall[idx]:.2%}  FP_rate={fp_rate_hz[idx]:.2f}/s")
    else:
        print("No threshold in the sweep achieves FP rate <= 1/s "
              "(lowest achieved: {:.2f}/s at threshold={:.4f})".format(
                  fp_rate_hz.min(), thresholds[np.argmin(fp_rate_hz)]))

    # ---- Save curve data -----------------------------------------------------
    np.savez(os.path.join(args.out_dir, f"threshold_sweep_{args.target}.npz"),
             thresholds=thresholds, hits=hits, misses=misses, total_fp=total_fp,
             recall=recall, precision=precision, f1=f1, fp_rate_hz=fp_rate_hz,
             n_target_test=n_target_test, test_duration_s=test_duration_s)

    # ---- Plot ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(thresholds, recall, label="recall (hit rate)", color="tab:green")
    axes[0, 0].plot(thresholds, 1 - recall, label="miss rate (FN)", color="tab:red")
    axes[0, 0].axvline(thresholds[best_f1_idx], color="k", ls="--", lw=1, label="best F1")
    axes[0, 0].set_xlabel("threshold")
    axes[0, 0].set_ylabel("fraction of target test spikes")
    axes[0, 0].set_title("Recall / Miss rate vs threshold")
    axes[0, 0].legend()

    axes[0, 1].plot(thresholds, fp_rate_hz, color="tab:orange")
    axes[0, 1].axvline(thresholds[best_f1_idx], color="k", ls="--", lw=1)
    axes[0, 1].set_xlabel("threshold")
    axes[0, 1].set_ylabel("false positives / sec")
    axes[0, 1].set_title("False positive rate vs threshold")
    axes[0, 1].set_yscale("log")

    axes[1, 0].plot(recall, precision, marker=".")
    axes[1, 0].scatter([recall[best_f1_idx]], [precision[best_f1_idx]], color="k", zorder=5, label="best F1")
    axes[1, 0].set_xlabel("recall")
    axes[1, 0].set_ylabel("precision")
    axes[1, 0].set_title("Precision-Recall curve (held-out test split)")
    axes[1, 0].legend()

    axes[1, 1].plot(thresholds, f1, color="tab:purple")
    axes[1, 1].axvline(thresholds[best_f1_idx], color="k", ls="--", lw=1)
    axes[1, 1].set_xlabel("threshold")
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].set_title("F1 vs threshold")

    fig.suptitle(f"Cluster {args.target}: LCMV threshold sweep, "
                 f"held-out test split ({test_duration_s:.1f}s, {n_target_test} true spikes)")
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, f"threshold_sweep_{args.target}.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved plot to {out_png}")
    print(f"Saved curve data to {os.path.join(args.out_dir, f'threshold_sweep_{args.target}.npz')}")


if __name__ == "__main__":
    main()
