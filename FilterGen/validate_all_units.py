"""
validate_all_units.py
======================

Validates ClosedLoopAllUnits.exe's (the all-units GPU detection pipeline)
output against Kilosort ground truth, and summarizes its per-chunk GPU
latency log.

Based on ksfxns/ncafstuff/compareonlinedetects.py's approach -- same
matchSpikeTrains (greedy, one-to-one, sorted-array) matching algorithm,
ported verbatim below rather than imported (that module lives in a separate
project/repo, and pulling one function is cleaner than adding a cross-repo
dependency), and the same precision/recall/F1-against-ground-truth framing
that produced the compare_output plots this validation is meant to
reproduce at scale.

The one real simplification vs. that script: compareonlinedetects.py has to
DISCOVER which Kilosort unit an online method's own (unrelated, opaque)
target-unit id corresponds to, by searching kilosort units near the
method's peak channel and scoring which one's spike train actually lines up
in time (findCandidateKsUnits / identifyGroundTruthUnit). Here that search
isn't needed: ImecFetchThreadGPU's detections CSV already carries the true
Kilosort cluster id for every detection (GpuFilterBank::hostUnitIds), since
calibrate_all_units.py fit each unit's filter directly against that same
cluster id in the first place. So this script goes straight from
detections CSV unit_id -> ground truth spike train, for every unit at once,
instead of one spatial-candidate-search per target.

Also not applicable here (skipped, not reimplemented): the reference
script's syllable-aligned raster/PSTH and offset-drift plots are about
comparing methods running on DIFFERENT clocks/trial structure (OSS vs LCMV,
each with its own online target numbering) against behavioral trial
windows. ImecFetchThreadGPU's detections are already on the exact same
sample clock as ground truth by construction (same live IMEC stream), so
there's no cross-clock offset to look for, and this validation isn't
scoped to any particular behavioral alignment -- it's asking "does the raw
detector agree with Kilosort," full stop.

Usage
-----
    python validate_all_units.py \\
        --ks-dir D:/test_newsorter/ks_out \\
        --detections-csv D:/test_newsorter/spikeTimes_all_units.csv \\
        --summary-csv D:/test_newsorter/filters_all_units/summary.csv \\
        --latency-csv D:/test_newsorter/latency_all_units.csv \\
        --out-dir D:/test_newsorter/validate_all_units_out
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_filter as gf


def match_spike_trains(a, b, tol):
    """Greedy nearest-neighbor one-to-one match between two sorted spike-time
    arrays, walking both in time order -- ported verbatim from
    ksfxns/ncafstuff/compare.py's matchSpikeTrains (see module docstring for
    why ported rather than imported). Two spikes match if |a[i]-b[j]| <= tol;
    each spike is used in at most one match.

    Returns (matchedA_idx, matchedB_idx, aOnly_idx, bOnly_idx) -- index
    arrays into a and b respectively.
    """
    i, j = 0, 0
    matchedA, matchedB, onlyA, onlyB = [], [], [], []
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tol:
            matchedA.append(i)
            matchedB.append(j)
            i += 1
            j += 1
        elif d < 0:
            onlyA.append(i)
            i += 1
        else:
            onlyB.append(j)
            j += 1
    onlyA.extend(range(i, len(a)))
    onlyB.extend(range(j, len(b)))
    return (np.array(matchedA, dtype=int), np.array(matchedB, dtype=int),
            np.array(onlyA, dtype=int), np.array(onlyB, dtype=int))


def compute_unit_metrics(gt_st, det_st, tol):
    """tp/fn/fp/precision/recall/f1 for one unit, given already-sorted,
    already-window-restricted ground-truth and detection spike times."""
    _, _, gt_only, det_only = match_spike_trains(gt_st, det_st, tol)
    tp = len(gt_st) - len(gt_only)
    fn = len(gt_only)
    fp = len(det_only)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0 else np.nan)
    return tp, fn, fp, precision, recall, f1


def plot_f1_distribution(metrics_df):
    """Histogram of per-unit F1 -- the many-units analog of the reference
    script's single-unit precision/recall/F1 bar chart (a bar chart doesn't
    scale to 100 units, so this is a distribution instead)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = metrics_df["f1"].dropna().values
    ax.hist(vals, bins=np.linspace(0, 1, 21), color="steelblue", edgecolor="white")
    ax.axvline(np.median(vals), color="k", ls="--", lw=1,
               label=f"median={np.median(vals):.3f}")
    ax.set_xlabel("F1 (vs Kilosort ground truth)")
    ax.set_ylabel("Units")
    ax.set_title(f"Per-unit F1 distribution (n={len(vals)} units)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_precision_recall_scatter(metrics_df):
    """Precision vs recall per unit, point size ~ ground-truth firing rate
    (more spikes = a more reliably-scored point) -- the many-units analog of
    the reference script's candidate-score bar chart."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = 15 + 40 * (metrics_df["n_gt_spikes"] / max(metrics_df["n_gt_spikes"].max(), 1))
    sc = ax.scatter(metrics_df["recall"], metrics_df["precision"],
                     s=sizes, c=metrics_df["f1"], cmap="viridis", vmin=0, vmax=1,
                     edgecolor="k", linewidth=0.3, alpha=0.85)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision vs recall, {len(metrics_df)} units\n"
                 f"(point size ~ ground-truth spike count, color = F1)")
    fig.colorbar(sc, ax=ax, label="F1")
    fig.tight_layout()
    return fig


def plot_latency(latency_df, fetch_chunk_ms):
    """Latency histogram + the running per-chunk trace, with fetchChunkMs
    marked -- the real-time budget this pipeline needs to stay under, in
    steady state, to keep up with the live fetch cadence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(latency_df["latency_ms"], bins=50, color="tab:purple", edgecolor="white")
    axes[0].axvline(fetch_chunk_ms, color="k", ls="--", lw=1, label=f"fetchChunkMs={fetch_chunk_ms}")
    axes[0].set_xlabel("GPU pipeline latency (ms/chunk)")
    axes[0].set_ylabel("Chunks")
    axes[0].set_title("Latency distribution")
    axes[0].legend()

    axes[1].plot(latency_df["chunk_index"], latency_df["latency_ms"], lw=0.6, color="tab:purple")
    axes[1].axhline(fetch_chunk_ms, color="k", ls="--", lw=1)
    axes[1].set_xlabel("Chunk index")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency over the run")

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--detections-csv", required=True,
                     help="ImecFetchThreadGPU's spikeTimesPath output "
                          "(unit_id,sample_index,score)")
    ap.add_argument("--summary-csv",
                     help="calibrate_all_units.py's summary.csv -- optional, "
                          "only used to also report each unit's OFFLINE "
                          "(held-out-split) predicted metrics alongside the "
                          "live ones for comparison")
    ap.add_argument("--latency-csv",
                     help="ImecFetchThreadGPU's latencyLogPath output "
                          "(chunk_index,n_samples,latency_ms) -- optional")
    ap.add_argument("--fetch-chunk-ms", type=float, default=5.0,
                     help="The run's fetchChunkMs, for the latency plot's "
                          "reference line (does not need to match exactly, "
                          "just for the plot)")
    ap.add_argument("--tol", type=int, default=100,
                     help="Match tolerance in samples (default matches the "
                          "reference script)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Kilosort ground truth...")
    spike_t, spike_cl, _ = gf.load_kilosort(args.ks_dir)

    print(f"Loading detections from {args.detections_csv}...")
    det_df = pd.read_csv(args.detections_csv)
    if det_df.empty:
        sys.exit("Detections CSV is empty -- nothing to validate.")

    # Shared active window: the run only ever fetched/detected within the
    # sample-index range it actually saw. Ground-truth spikes outside that
    # range were never given a chance to be detected and would otherwise
    # inflate false negatives for units that happen to fire heavily outside
    # it -- same rationale as the reference script's
    # findCommonActiveTrials/restrictToActiveWindow, simplified to one
    # shared window since this is one method (not two on independently
    # varying active ranges).
    lo, hi = int(det_df["sample_index"].min()), int(det_df["sample_index"].max())
    print(f"Active window: samples [{lo}, {hi}] ({(hi - lo) / 1000.0:.1f}k samples)")

    unit_ids = sorted(det_df["unit_id"].unique())
    print(f"{len(unit_ids)} units present in detections CSV")

    rows = []
    for uid in unit_ids:
        gt_st = np.sort(spike_t[spike_cl == uid])
        gt_st = gt_st[(gt_st >= lo) & (gt_st <= hi)]
        det_st = np.sort(det_df.loc[det_df["unit_id"] == uid, "sample_index"].values)
        det_st = det_st[(det_st >= lo) & (det_st <= hi)]

        tp, fn, fp, precision, recall, f1 = compute_unit_metrics(gt_st, det_st, args.tol)
        rows.append({
            "unit_id": uid, "n_gt_spikes": len(gt_st), "n_detected": len(det_st),
            "tp": tp, "fn": fn, "fp": fp,
            "precision": precision, "recall": recall, "f1": f1,
        })

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    if args.summary_csv and os.path.isfile(args.summary_csv):
        offline_df = pd.read_csv(args.summary_csv)
        offline_df = offline_df[offline_df["status"] == "ok"][
            ["unit_id", "threshold", "recall", "precision", "f1"]
        ].rename(columns={"threshold": "offline_threshold", "recall": "offline_recall",
                           "precision": "offline_precision", "f1": "offline_f1"})
        metrics_df = metrics_df.merge(offline_df, on="unit_id", how="left")

    metrics_path = os.path.join(args.out_dir, "validation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")

    # Micro-averaged (pooled tp/fn/fp across every unit) aggregate, printed
    # to stdout -- the single headline number for "how well is the whole
    # 100-unit pipeline doing," as opposed to the per-unit macro breakdown
    # already in metrics_df.
    total_tp = metrics_df["tp"].sum()
    total_fn = metrics_df["fn"].sum()
    total_fp = metrics_df["fp"].sum()
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else np.nan
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else np.nan
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else np.nan)
    median_f1 = metrics_df["f1"].median()
    print(f"\nAggregate over {len(metrics_df)} units:")
    print(f"  micro-avg (pooled)  precision={micro_precision:.3f}  recall={micro_recall:.3f}  f1={micro_f1:.3f}")
    print(f"  macro-avg (median)  f1={median_f1:.3f}")

    plot_f1_distribution(metrics_df).savefig(
        os.path.join(args.out_dir, "f1_distribution.png"), dpi=120)
    plot_precision_recall_scatter(metrics_df).savefig(
        os.path.join(args.out_dir, "precision_recall_scatter.png"), dpi=120)
    plt.close("all")
    print(f"Saved f1_distribution.png / precision_recall_scatter.png to {args.out_dir}")

    if args.latency_csv and os.path.isfile(args.latency_csv):
        latency_df = pd.read_csv(args.latency_csv)
        if not latency_df.empty:
            lat = latency_df["latency_ms"]
            print(f"\nGPU pipeline latency over {len(lat)} chunks:")
            print(f"  mean={lat.mean():.3f}ms  median={lat.median():.3f}ms  "
                  f"p95={lat.quantile(0.95):.3f}ms  p99={lat.quantile(0.99):.3f}ms  "
                  f"max={lat.max():.3f}ms")
            over_budget = (lat > args.fetch_chunk_ms).sum()
            print(f"  {over_budget}/{len(lat)} chunks ({100*over_budget/len(lat):.2f}%) "
                  f"exceeded fetchChunkMs={args.fetch_chunk_ms}ms")

            plot_latency(latency_df, args.fetch_chunk_ms).savefig(
                os.path.join(args.out_dir, "latency.png"), dpi=120)
            plt.close("all")

            latency_summary_path = os.path.join(args.out_dir, "latency_summary.csv")
            pd.DataFrame([{
                "n_chunks": len(lat), "mean_ms": lat.mean(), "median_ms": lat.median(),
                "p95_ms": lat.quantile(0.95), "p99_ms": lat.quantile(0.99), "max_ms": lat.max(),
                "fetch_chunk_ms": args.fetch_chunk_ms,
                "n_over_budget": int(over_budget), "frac_over_budget": over_budget / len(lat),
            }]).to_csv(latency_summary_path, index=False)
            print(f"Saved latency.png / {latency_summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
