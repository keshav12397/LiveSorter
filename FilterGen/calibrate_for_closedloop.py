"""
calibrate_for_closedloop.py
============================

Single entry point ClosedLoop/main.cpp shells out to instead of running its
own C++ reimplementation of calibration. Given a training .bin + its
Kilosort output, this:

  1. Fits the LCMV filter on the first `--train-frac` of the recording
     (channel selection, LCMV weights) -- always using the causal RBJ-biquad
     high-pass (matching ClosedLoop/ButterworthHighpass.h bit-for-bit), since
     that's what the live C++ pipeline will actually run.
  2. Sweeps detection thresholds on the held-out remainder against Kilosort
     ground truth, picks the best-F1 operating point.
  3. Saves *that exact* train-split-fitted filter (not a separately refit
     one on the full recording) together with the swept threshold to
     channels_<id>.bin / filter_<id>.bin / threshold_<id>.bin.

Deploying the same filter object whose threshold was actually validated --
rather than fitting on train, validating, then refitting on all the data for
"better" final weights -- is deliberate: refitting after the fact reintroduces
exactly the kind of filter/threshold mismatch this session spent a long time
chasing down as a live-recall killer. A few percent less training data is a
trivial cost next to that.

This reuses generate_filter.py's building blocks and threshold_sweep_real.py's
fit_lcmv/sweep_thresholds so there is exactly one implementation of "how the
filter is fit" and "how a threshold is scored" -- this script just sequences
them and writes the final artifact, instead of ClosedLoop/Calibration.cpp
maintaining an independent (and, this session found out the hard way,
drift-prone) C++ port of the same math.

Exit code 0 + the three .bin files written to --out-dir means success;
nonzero exit code / exception means ClosedLoop.exe should abort startup.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless -- this runs as a subprocess of ClosedLoop.exe
import matplotlib.pyplot as plt

import generate_filter as gf
import threshold_sweep_real as tsr


# --------------------------------------------------------------------- #
# Plots -- always saved on a successful run, next to the .bin files, so
# there's no separate step to remember to visually sanity-check a
# calibration. Two files:
#   template_filter_<id>.png   -- per-channel template waveform + filter
#                                  taps used to fit it, plus an example
#                                  segment of the held-out convolved trace
#                                  with detections and true spikes marked.
#   threshold_sweep_<id>.png   -- recall/precision/F1/FP-rate vs threshold,
#                                  same layout threshold_sweep_real.py's
#                                  standalone sweeps already produce.
# --------------------------------------------------------------------- #

def plot_template_and_detections(out_dir, target_id, f, sel_channels, template_wf,
                                  template_length, D_test, peak_idx, peak_scores,
                                  best_threshold, target_test_rel, fs, window_s=0.15):
    """f, template_wf: (template_length, n_channels_selected) -- filter taps
    and the target's own mean waveform on the SAME selected channels, so
    they're directly comparable column-for-column. Two separate columns
    (not overlaid / dual-axis) since amplitude and filter-weight are
    different units -- see dataviz guidance on never dual-axis a chart."""
    n_ch = len(sel_channels)
    t_ms = (np.arange(template_length) - template_length // 2) / fs * 1000.0

    fig = plt.figure(figsize=(11, 2.1 * n_ch + 3))
    gs = fig.add_gridspec(n_ch + 1, 2, height_ratios=[1] * n_ch + [1.4])

    for i, ch in enumerate(sel_channels):
        ax_wf = fig.add_subplot(gs[i, 0])
        ax_wf.plot(t_ms, template_wf[:, i], color="tab:blue", lw=1.5)
        ax_wf.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax_wf.set_ylabel(f"ch {ch}\n(uV-ish)", fontsize=8)
        if i == 0:
            ax_wf.set_title("Target mean waveform (train split)")
        if i == n_ch - 1:
            ax_wf.set_xlabel("ms from spike center")

        ax_f = fig.add_subplot(gs[i, 1])
        ax_f.plot(t_ms, f[:, i], color="tab:purple", lw=1.5)
        ax_f.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax_f.set_ylabel("weight", fontsize=8)
        if i == 0:
            ax_f.set_title("LCMV filter taps (this channel)")
        if i == n_ch - 1:
            ax_f.set_xlabel("ms from spike center")

    # Example segment of the held-out convolved trace, centered on the
    # first true target spike in the test split (falls back to t=0 if the
    # test split has no target spikes at all).
    ax_trace = fig.add_subplot(gs[n_ch, :])
    center = int(target_test_rel[0]) if len(target_test_rel) else 0
    half_w = int(window_s * fs)
    lo, hi = max(0, center - half_w), min(len(D_test), center + half_w)
    t_trace_ms = (np.arange(lo, hi) - center) / fs * 1000.0

    ax_trace.plot(t_trace_ms, D_test[lo:hi], color="tab:blue", lw=1, label="convolved trace")
    ax_trace.axhline(best_threshold, color="k", ls="--", lw=1, label=f"threshold={best_threshold:.3f}")

    in_window = (peak_idx >= lo) & (peak_idx < hi) & (peak_scores > best_threshold)
    ax_trace.scatter((peak_idx[in_window] - center) / fs * 1000.0, peak_scores[in_window],
                      color="tab:red", zorder=5, s=30, label="detected spike")

    true_in_window = target_test_rel[(target_test_rel >= lo) & (target_test_rel < hi)]
    for k, tt in enumerate(true_in_window):
        ax_trace.axvline((tt - center) / fs * 1000.0, color="tab:green", lw=1, alpha=0.6,
                          label="true spike (Kilosort)" if k == 0 else None)

    ax_trace.set_xlabel("ms")
    ax_trace.set_ylabel("matched-filter score")
    ax_trace.set_title("Example detection window (held-out test split)")
    ax_trace.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Cluster {target_id}: template / filter / example detections")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"template_filter_{target_id}.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_png}")


def plot_threshold_sweep(out_dir, target_id, thresholds, recall, precision, f1,
                          fp_rate_hz, best_idx, test_duration_s, n_target_test):
    """Same layout as threshold_sweep_real.py's standalone sweep plot."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(thresholds, recall, label="recall (hit rate)", color="tab:green")
    axes[0, 0].plot(thresholds, 1 - recall, label="miss rate (FN)", color="tab:red")
    axes[0, 0].axvline(thresholds[best_idx], color="k", ls="--", lw=1, label="best F1")
    axes[0, 0].set_xlabel("threshold")
    axes[0, 0].set_ylabel("fraction of target test spikes")
    axes[0, 0].set_title("Recall / Miss rate vs threshold")
    axes[0, 0].legend()

    axes[0, 1].plot(thresholds, fp_rate_hz, color="tab:orange")
    axes[0, 1].axvline(thresholds[best_idx], color="k", ls="--", lw=1)
    axes[0, 1].set_xlabel("threshold")
    axes[0, 1].set_ylabel("false positives / sec")
    axes[0, 1].set_title("False positive rate vs threshold")
    axes[0, 1].set_yscale("log")

    axes[1, 0].plot(recall, precision, marker=".")
    axes[1, 0].scatter([recall[best_idx]], [precision[best_idx]], color="k", zorder=5, label="best F1")
    axes[1, 0].set_xlabel("recall")
    axes[1, 0].set_ylabel("precision")
    axes[1, 0].set_title("Precision-Recall curve (held-out test split)")
    axes[1, 0].legend()

    axes[1, 1].plot(thresholds, f1, color="tab:purple")
    axes[1, 1].axvline(thresholds[best_idx], color="k", ls="--", lw=1)
    axes[1, 1].set_xlabel("threshold")
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].set_title("F1 vs threshold")

    fig.suptitle(f"Cluster {target_id}: LCMV threshold sweep, "
                 f"held-out test split ({test_duration_s:.1f}s, {n_target_test} true spikes)")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"threshold_sweep_{target_id}.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--meta-path")
    ap.add_argument("--channel-map-json")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--interferers", type=int, nargs="+", default=None,
                     help="Explicit Kilosort cluster ids to null out. Omit "
                          "and pass --auto-interferers instead to pick them "
                          "automatically (the N most active other clusters).")
    ap.add_argument("--auto-interferers", type=int, default=0,
                     help="Auto-pick this many interferers by raw spike "
                          "count (excluding the target and anything labeled "
                          "'noise' in cluster_group.tsv/cluster_KSLabel.tsv). "
                          "Ignored if --interferers is given.")
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-spikes", type=int, default=2000)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--n-thresholds", type=int, default=60)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Reuse threshold_sweep_real's loader -- always causal-highpass, since
    # this filter is only ever meant to run against ClosedLoop's live pipeline.
    class _Args:
        pass
    load_args = _Args()
    load_args.ks_dir = args.ks_dir
    load_args.bin_path = args.bin_path
    load_args.meta_path = args.meta_path
    load_args.channel_map_json = args.channel_map_json
    load_args.filter = True
    load_args.car = True
    load_args.fc = args.fc
    load_args.causal_highpass = True

    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(load_args, rng)

    if args.interferers:
        interferer_ids = args.interferers
    elif args.auto_interferers:
        # Cheap re-load (just two .npy arrays + a .tsv, not the raw data)
        # purely to get cluster_group/KSLabel labels -- load_and_prepare
        # above already loaded spike_t/spike_cl but discards labels.
        _, _, labels = gf.load_kilosort(args.ks_dir)
        positions = gf.load_channel_positions_json(args.channel_map_json) \
            if args.channel_map_json else {}
        interferer_ids = gf.auto_pick_interferers_spatial(
            data, spike_t, spike_cl, np_ch, args.target, labels,
            args.auto_interferers, args.template_length, args.template_offset,
            channel_positions=positions, rng=rng)
        print(f"Auto-picked interferers (top {args.auto_interferers} most "
              f"active non-noise clusters nearest the target's own peak "
              f"channel): {interferer_ids}")
    else:
        raise ValueError("Must pass either --interferers or --auto-interferers")
    args.interferers = interferer_ids

    split_t = int(args.train_frac * data.shape[0])
    print(f"Train: [0, {split_t}) ({split_t/fs:.1f}s)   "
          f"Test: [{split_t}, {data.shape[0]}) ({(data.shape[0]-split_t)/fs:.1f}s)")

    data_train, data_test = data[:split_t], data[split_t:]

    def split(times):
        return times[times < split_t], times[times >= split_t]

    print("Fitting LCMV on TRAIN split (causal high-pass)...")
    f, sel, sel_channels, target_spikes_train, interferer_times_train = tsr.fit_lcmv(
        data_train, spike_t[spike_t < split_t], spike_cl[spike_t < split_t], np_ch,
        args.target, args.interferers, args.n_channels,
        args.template_length, args.template_offset, args.ridge, args.max_spikes, rng)
    print(f"Selected channels: {sel_channels.tolist()}")

    # Recomputed independently of fit_lcmv's internal (already-consumed)
    # subsample -- just for the diagnostic plot below, so it doesn't need
    # fit_lcmv to expose its internal waveform. A fresh subsample from the
    # same spikes/rng is statistically equivalent for visualization purposes.
    target_wf_sel, _ = gf.mean_waveform(data_train[:, sel], target_spikes_train,
                                         args.template_length, args.template_offset,
                                         args.max_spikes, rng)

    target_train, target_test = split(spike_t[spike_cl == args.target])
    interferer_test = {cid: split(spike_t[spike_cl == cid])[1] for cid in args.interferers}
    print(f"Target: {len(target_train)} train spikes, {len(target_test)} test spikes")

    data_test_sel = data_test[:, sel]
    print("Scoring held-out test split against this exact train-fitted filter...")
    D_test = gf.filter_output(data_test_sel, f)

    window = args.template_length // 4
    min_sep = args.template_length // 2
    peak_idx, peak_scores = tsr.find_all_peaks(D_test, min_sep)
    print(f"Found {len(peak_idx)} candidate peaks in test split.")

    target_test_rel = target_test - split_t
    interferer_test_rel = {cid: t - split_t for cid, t in interferer_test.items()}
    results = tsr.sweep_thresholds(peak_idx, peak_scores, target_test_rel,
                                    interferer_test_rel, window, args.n_thresholds)

    thresholds = np.array([r["threshold"] for r in results])
    hits = np.array([r["hits"] for r in results])
    total_fp = np.array([r["total_fp"] for r in results])
    n_target_test = max(len(target_test), 1)
    test_duration_s = (data.shape[0] - split_t) / fs

    recall = hits / n_target_test
    precision = hits / np.maximum(hits + total_fp, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    fp_rate_hz = total_fp / test_duration_s

    best_idx = int(np.argmax(f1))
    best_threshold = float(thresholds[best_idx])
    print(f"Best F1 operating point: threshold={best_threshold:.6f}  "
          f"recall={recall[best_idx]:.2%}  precision={precision[best_idx]:.2%}  "
          f"F1={f1[best_idx]:.3f}  FP_rate={fp_rate_hz[best_idx]:.4f}/s")

    # ---- Save the TRAIN-split-fitted filter + swept threshold -----------
    # Deliberately the same `f`/`sel_channels` object the sweep above just
    # validated -- see module docstring for why this must not be refit.
    gf.save_filter(args.out_dir, args.target, sel_channels, f, best_threshold)
    print(f"Saved channels_{args.target}.bin / filter_{args.target}.bin / "
          f"threshold_{args.target}.bin to {args.out_dir}")

    # ---- Diagnostic plots -- always generated on success ------------------
    plot_threshold_sweep(args.out_dir, args.target, thresholds, recall, precision, f1,
                          fp_rate_hz, best_idx, test_duration_s, n_target_test)
    plot_template_and_detections(args.out_dir, args.target, f, sel_channels, target_wf_sel,
                                  args.template_length, D_test, peak_idx, peak_scores,
                                  best_threshold, target_test_rel, fs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
