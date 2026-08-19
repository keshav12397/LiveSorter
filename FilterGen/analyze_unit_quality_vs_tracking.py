"""
analyze_unit_quality_vs_tracking.py
====================================

Answers one question: does a unit's OFFLINE quality (Kilosort's own Amplitude
/ ContamPct / KSLabel, and how many spikes it fired) predict how well
ClosedLoopAllUnits.exe tracks it live, against Kilosort ground truth?

Reuses validate_all_units.py's matching (match_spike_trains,
compute_unit_metrics) and wrap-samples handling verbatim (imported, not
re-derived -- see that module's own docstring for why: a live/simulated
SpikeGLX session's sample counter keeps incrementing across every loop
through the replayed file rather than resetting to 0, and this project has
already gotten that wrong once before). This script only adds the
quality-metric merge and the plots that ask the quality-vs-tracking
question -- validate_all_units.py's own f1_distribution/precision_recall
plots and latency analysis aren't reproduced here, run that script too if
you want those.

Usage
-----
    python analyze_unit_quality_vs_tracking.py \\
        --ks-dir D:/test_newsorter/ks_out \\
        --detections-csv D:/test_newsorter/live_all_units_run/spikeTimes.csv \\
        --summary-csv D:/test_newsorter/filters_all_units_cpp_full/summary.csv \\
        --wrap-samples 39598326 \\
        --out-dir D:/test_newsorter/live_all_units_run/quality_analysis
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_filter as gf
from validate_all_units import match_spike_trains, compute_unit_metrics

# -- palette (dataviz skill's validated default instance) -------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CAT_GOOD = "#2a78d6"   # categorical slot 1 (blue) -- KSLabel "good"
CAT_MUA = "#eb6834"    # categorical slot 2 (orange) -- KSLabel "mua"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.family": "sans-serif", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def cat_color(label):
    return CAT_GOOD if str(label).lower() == "good" else CAT_MUA


def load_quality_metrics(ks_dir):
    """cluster_id -> {amplitude, contam_pct, ks_label, n_spikes_total}."""
    amp = pd.read_csv(os.path.join(ks_dir, "cluster_Amplitude.tsv"), sep="\t")
    contam = pd.read_csv(os.path.join(ks_dir, "cluster_ContamPct.tsv"), sep="\t")
    label_path = os.path.join(ks_dir, "cluster_group.tsv")
    if not os.path.isfile(label_path):
        label_path = os.path.join(ks_dir, "cluster_KSLabel.tsv")
    labels = pd.read_csv(label_path, sep="\t")
    label_col = "group" if "group" in labels.columns else "KSLabel"

    df = amp.merge(contam, on="cluster_id", how="outer")
    df = df.merge(labels[["cluster_id", label_col]], on="cluster_id", how="left")
    df = df.rename(columns={"Amplitude": "amplitude", "ContamPct": "contam_pct",
                             label_col: "ks_label"})
    return df


def scatter_panel(ax, x, y, colors, sizes, xlabel, ylabel, xlog=False):
    if xlog:
        ax.set_xscale("log")
    ax.scatter(x, y, s=sizes, c=colors, edgecolor=INK_PRIMARY, linewidth=0.3, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)


def corr_stats(df, xcol, ycol):
    sub = df[[xcol, ycol]].dropna()
    if len(sub) < 3:
        return np.nan, np.nan, len(sub)
    pearson = sub[xcol].corr(sub[ycol], method="pearson")
    spearman = sub[xcol].corr(sub[ycol], method="spearman")
    return pearson, spearman, len(sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--detections-csv", required=True)
    ap.add_argument("--summary-csv",
                     help="calibrate_all_units.py's summary.csv -- optional, adds "
                          "offline (held-out-split) metrics + n_channels/threshold "
                          "to the merge and an online-vs-offline F1 panel.")
    ap.add_argument("--tol", type=int, default=100)
    ap.add_argument("--wrap-samples", type=int, default=0,
                     help="Recording's true total sample count -- see "
                          "validate_all_units.py's --wrap-samples for why this "
                          "matters for a looping simulated/replayed session. When "
                          "the captured run spans MULTIPLE loop passes (common for "
                          "a >1-file-length live run), a plain modulo merges "
                          "detections from every pass onto the same ground-truth "
                          "spikes -- a unit detected reliably in every pass then "
                          "gets N-1 spurious 'extra' detections credited as false "
                          "positives per spike (the greedy one-to-one matcher only "
                          "accepts the first), which biases precision/recall AGAINST "
                          "exactly the well-tracked units this analysis cares about. "
                          "So when multiple passes are present, this script instead "
                          "auto-selects ONE fully-covered pass (see --loop-pass) and "
                          "analyzes only that, uncontaminated.")
    ap.add_argument("--loop-pass", type=int, default=-1,
                     help="With --wrap-samples set and multiple loop passes present, "
                          "force analysis to this pass index (0-based) instead of "
                          "auto-selecting the fullest-covered one. -1 (default) = auto.")
    ap.add_argument("--min-gt-spikes", type=int, default=20,
                     help="Drop units with fewer than this many ground-truth "
                          "spikes in the active window from the correlation/plots "
                          "-- too few to estimate recall/precision meaningfully.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Kilosort ground truth + quality metrics...")
    spike_t, spike_cl, _ = gf.load_kilosort(args.ks_dir)
    quality_df = load_quality_metrics(args.ks_dir)

    print(f"Loading detections from {args.detections_csv}...")
    det_df = pd.read_csv(args.detections_csv)
    if det_df.empty:
        sys.exit("Detections CSV is empty -- nothing to analyze.")

    if args.wrap_samples:
        raw_lo, raw_hi = int(det_df["sample_index"].min()), int(det_df["sample_index"].max())
        first_pass, last_pass = raw_lo // args.wrap_samples, raw_hi // args.wrap_samples

        if first_pass == last_pass:
            # Single pass -- plain modulo is safe (matches validate_all_units.py's
            # own handling), nothing to disambiguate.
            det_df["sample_index"] = det_df["sample_index"] % args.wrap_samples
            print(f"Wrapped sample_index mod {args.wrap_samples} "
                  f"(raw range was [{raw_lo}, {raw_hi}], single loop pass)")
        else:
            n_passes = last_pass - first_pass + 1
            print(f"Raw sample_index spans {n_passes} loop passes "
                  f"({first_pass} through {last_pass}) -- selecting ONE "
                  f"uncontaminated pass instead of merging them (see --wrap-samples "
                  f"help for why).")

            if args.loop_pass >= 0:
                chosen = args.loop_pass
            else:
                # Auto: the pass whose OWN local coverage is widest -- a pass
                # only partially observed (the run started or ended mid-pass)
                # covers a narrower slice of the file and would undercount that
                # pass's ground-truth spikes outside the observed slice.
                coverage = {}
                for p in range(first_pass, last_pass + 1):
                    lo, hi = p * args.wrap_samples, (p + 1) * args.wrap_samples
                    mask = (det_df["sample_index"] >= lo) & (det_df["sample_index"] < hi)
                    if mask.any():
                        local = det_df.loc[mask, "sample_index"] - lo
                        coverage[p] = (local.max() - local.min(), int(mask.sum()))
                chosen = max(coverage, key=lambda p: coverage[p][0])
                print(f"  pass coverage (span_samples, n_rows): "
                      + ", ".join(f"{p}={v}" for p, v in coverage.items()))

            lo, hi = chosen * args.wrap_samples, (chosen + 1) * args.wrap_samples
            det_df = det_df[(det_df["sample_index"] >= lo) & (det_df["sample_index"] < hi)].copy()
            det_df["sample_index"] = det_df["sample_index"] - lo
            print(f"Using loop pass {chosen} only: {len(det_df)} detection rows, "
                  f"local sample range [{det_df['sample_index'].min()}, "
                  f"{det_df['sample_index'].max()}]")

    lo, hi = int(det_df["sample_index"].min()), int(det_df["sample_index"].max())
    print(f"Active window: samples [{lo}, {hi}] ({(hi - lo) / 30000.0:.1f}s @ 30kHz)")

    unit_ids = sorted(det_df["unit_id"].unique())
    print(f"{len(unit_ids)} units present in detections CSV")

    rows = []
    for uid in unit_ids:
        gt_st = np.sort(spike_t[spike_cl == uid])
        gt_st_window = gt_st[(gt_st >= lo) & (gt_st <= hi)]
        det_st = np.sort(det_df.loc[det_df["unit_id"] == uid, "sample_index"].values)
        det_st = det_st[(det_st >= lo) & (det_st <= hi)]

        tp, fn, fp, precision, recall, f1 = compute_unit_metrics(gt_st_window, det_st, args.tol)
        rows.append({
            "unit_id": uid, "n_gt_spikes": len(gt_st_window), "n_detected": len(det_st),
            "n_gt_spikes_total_session": len(gt_st),
            "tp": tp, "fn": fn, "fp": fp,
            "precision": precision, "recall": recall, "f1": f1,
        })
    metrics_df = pd.DataFrame(rows)

    merged = metrics_df.merge(quality_df, left_on="unit_id", right_on="cluster_id", how="left")

    if args.summary_csv and os.path.isfile(args.summary_csv):
        offline_df = pd.read_csv(args.summary_csv)
        offline_df = offline_df[offline_df["status"] == "ok"][
            ["unit_id", "n_channels", "threshold", "recall", "precision", "f1"]
        ].rename(columns={"threshold": "offline_threshold", "recall": "offline_recall",
                           "precision": "offline_precision", "f1": "offline_f1"})
        merged = merged.merge(offline_df, on="unit_id", how="left")

    out_csv = os.path.join(args.out_dir, "quality_vs_tracking.csv")
    merged.sort_values("f1", ascending=False).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    scored = merged[merged["n_gt_spikes"] >= args.min_gt_spikes].copy()
    print(f"\n{len(scored)}/{len(merged)} units have >= {args.min_gt_spikes} "
          f"ground-truth spikes in the active window -- used for correlations/plots below.")
    if scored.empty:
        sys.exit("No units with enough ground-truth spikes in the active window "
                  "-- check --wrap-samples / that the live stream is really the "
                  "same recording Kilosort was run on.")

    # ---- Correlation summary -------------------------------------------------
    corr_rows = []
    for metric in ("recall", "precision", "f1"):
        for quality in ("amplitude", "contam_pct", "n_gt_spikes_total_session"):
            pear, spear, n = corr_stats(scored, quality, metric)
            corr_rows.append({"tracking_metric": metric, "quality_metric": quality,
                               "pearson_r": pear, "spearman_rho": spear, "n": n})
    corr_df = pd.DataFrame(corr_rows)
    corr_path = os.path.join(args.out_dir, "correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"\nCorrelations (tracking metric vs quality metric):")
    print(corr_df.to_string(index=False))
    print(f"Wrote {corr_path}")

    # ---- Diagnostic plots -----------------------------------------------------
    colors = scored["ks_label"].apply(cat_color).values
    sizes = 18 + 50 * (scored["n_gt_spikes"] / max(scored["n_gt_spikes"].max(), 1))
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CAT_GOOD,
               markeredgecolor=INK_PRIMARY, markeredgewidth=0.3, markersize=8, label="good"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CAT_MUA,
               markeredgecolor=INK_PRIMARY, markeredgewidth=0.3, markersize=8, label="mua"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    scatter_panel(axes[0, 0], scored["amplitude"], scored["recall"], colors, sizes,
                  "Kilosort template amplitude", "Online recall", xlog=True)
    scatter_panel(axes[0, 1], scored["contam_pct"], scored["recall"], colors, sizes,
                  "Kilosort ContamPct (lower = more isolated)", "Online recall")
    scatter_panel(axes[1, 0], scored["amplitude"], scored["f1"], colors, sizes,
                  "Kilosort template amplitude", "Online F1", xlog=True)
    scatter_panel(axes[1, 1], scored["contam_pct"], scored["f1"], colors, sizes,
                  "Kilosort ContamPct (lower = more isolated)", "Online F1")
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Unit quality vs. live tracking accuracy ({len(scored)} units, "
                 f"{(hi - lo) / 30000.0 / 60.0:.1f} min live window)", y=1.06, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "quality_vs_tracking.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Online vs offline F1 (does the offline held-out sweep predict live?) -
    if "offline_f1" in scored.columns and scored["offline_f1"].notna().any():
        fig, ax = plt.subplots(figsize=(6.2, 6))
        sub = scored.dropna(subset=["offline_f1"])
        ax.plot([0, 1], [0, 1], color=BASELINE, lw=1, ls="--", zorder=1)
        ax.scatter(sub["offline_f1"], sub["f1"], s=sizes[:len(sub)],
                   c=sub["ks_label"].apply(cat_color), edgecolor=INK_PRIMARY,
                   linewidth=0.3, alpha=0.85, zorder=2)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Offline F1 (held-out test split)")
        ax.set_ylabel("Online F1 (live, this run)")
        ax.set_title(f"Does the offline calibration sweep predict live tracking?\n(n={len(sub)} units)")
        ax.legend(handles=legend_handles, loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "online_vs_offline_f1.png"), dpi=140)
        plt.close(fig)

    # ---- Firing-rate context (does a unit's own activity level explain it?) ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(scored["n_gt_spikes_total_session"], scored["recall"],
                     s=sizes, c=scored["f1"], cmap=SEQ_BLUE, vmin=0, vmax=1,
                     edgecolor=INK_PRIMARY, linewidth=0.3, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Total spikes fired this session (Kilosort ground truth)")
    ax.set_ylabel("Online recall")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Recall vs. how active the unit is (color = F1, n={len(scored)})")
    fig.colorbar(sc, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "recall_vs_activity.png"), dpi=140)
    plt.close(fig)

    print(f"\nSaved quality_vs_tracking.png, recall_vs_activity.png"
          + (", online_vs_offline_f1.png" if "offline_f1" in scored.columns else "")
          + f" to {args.out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
