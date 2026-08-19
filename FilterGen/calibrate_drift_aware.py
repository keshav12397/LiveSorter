"""
calibrate_drift_aware.py
=========================

Drift-aware batch calibration: like `calibrate_all_units.py`, but each unit
gets one filter *per time segment inside which it did not move*, plus a
schedule saying when to swap them in.

The problem
-----------
`calibrate_all_units.py` fits one filter per unit from a mean waveform over
the whole recording. A unit that drifted contributes spikes from several
positions to that one average, so the template is a blur that matches the
unit at no single moment -- and the damage is not a graceful loss. See
`drift_estimate.py`'s docstring for the mechanism; the short version is that
a smeared target template overlaps a neighbour's template much more than the
true instantaneous one does, LCMV nulls the neighbour exactly, the null cuts
into the target, and the solve compensates by growing the weights until the
filter's own noise floor swamps its unity-gain response. Measured on
`D:/sim_validate`, the two worst units in the whole session were both
`drift_kind='jump'` and both failed this way (f1 0.020 and 0.032, precision
under 2%), while every static unit of comparable amplitude was fine.

No new math
-----------
Every fit here goes through `threshold_sweep_real.fit_lcmv`,
`generate_filter.auto_pick_interferers_spatial`,
`threshold_sweep_real.find_all_peaks` and
`threshold_sweep_real.sweep_thresholds` -- the same functions
`calibrate_all_units.py` calls, not copies of them. The only change made to
any of them is `fit_lcmv`'s optional `template_spike_t`/`template_spike_cl`,
which this file needs to build a template from train-cell spikes only while
still excluding held-out spikes from the noise covariance. This repo has
twice silently lost recall to a second, drifted reimplementation of this
math; there is no second implementation here.

Evaluation protocol (and why it differs from calibrate_all_units.py's)
----------------------------------------------------------------------
`calibrate_all_units.py` trains on the first half of the recording and tests
on the second. For a *drifting* unit that split is itself a confound: the
unit is in a different place in the test half than in the train half, so the
measured drop mixes "the template is smeared" with "the template is stale".
It also makes a time-segmented filter impossible to score at all -- a
segment inside the train half has no held-out data.

So this script interleaves instead. The recording is cut into `--cell-s`
cells; the first `--train-frac` of every cell is train, the rest is test.
Both `--mode global` (one filter per unit, exactly the baseline's fit, just
under this protocol) and `--mode segmented` see the *same* train samples and
are scored on the *same* test samples, so the difference between them is
attributable to segmentation and nothing else. `--mode both` runs both from
one preprocessing pass, which is the only sane way to do this on a 30-minute
recording.

Threshold selection is identical for both modes and gives both exactly one
free parameter: a single best-F1 threshold per unit, swept over the pooled
peaks from all of that unit's segments. A per-segment threshold would give
the segmented mode N free parameters against global's 1, and the comparison
would no longer mean anything. It also is not needed: the LCMV unity-gain
constraint pins each segment's response to its own target at 1.0, so segment
scores already live on a common axis.

Honest caveats
--------------
- Segment *boundaries* are chosen from a trajectory estimated over the whole
  recording, test cells included. The filters never see a held-out spike,
  but the cut points are informed by data the filters are scored on. It is a
  weak leak (a handful of boundary times per unit, from coarse amplitude
  centroids) and it is stated rather than hidden.
- The schedule's times are seconds from the start of the calibration
  recording. Replaying that recording, they are exact. Applying them to a
  genuinely new live run assumes the drift repeats, which it does not --
  see the "What this does not fix" section of the final report.

Output (in --out-dir, per mode)
-------------------------------
    unit_ids.bin / channels.bin / filters.bin / thresholds.bin
        Exactly the format GpuFilterBank::load() already reads, holding each
        unit's *first* segment -- so an unmodified ClosedLoopAllUnits.exe
        loads this directory and runs correctly from t=0.
    drift_schedule.bin / drift_schedule.json
        The remaining segments as timed swap events, for
        GpuFilterBank::updateFilters(). Absent or empty when no unit drifts.
    summary.csv
        One row per attempted unit, with the baseline's columns plus
        n_segments, drift_span_um and the segment boundary times.

Example
-------
    python calibrate_drift_aware.py \\
        --ks-dir D:/sim_validate/sim_ks \\
        --bin-path D:/sim_validate/sim_g0_t0.imec0.ap.bin \\
        --channel-map-json D:/test_newsorter/rawData/shank1only.json \\
        --auto-interferers 5 --min-spikes 60 --max-units 12 --workers 4 \\
        --mode both --out-dir D:/drift_work/validate
"""

import argparse
import contextlib
import csv
import io
import json
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import drift_estimate as de
import generate_filter as gf
import threshold_sweep_real as tsr

# Same reason as calibrate_all_units.py: N worker processes each fanning
# BLAS out across every core oversubscribes the machine badly enough to make
# the pool slower than serial. Must be set before numpy loads in the child,
# which (for spawned children) means setting it in the parent before the
# pool is created.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_worker = {}

SUMMARY_FIELDS = ["unit_id", "n_channels", "sel_channels", "threshold", "recall",
                  "precision", "f1", "fp_rate_hz", "n_train_spikes",
                  "n_test_spikes", "n_segments", "drift_span_um",
                  "segment_bounds_s", "noise_std", "detection_lag", "status"]


def train_mask(sample_idx, cell_samples, train_samples):
    """True for samples in the train part of their interleave cell.

    A single modulo, so it is cheap enough to call on every peak index of a
    30-minute recording. The grid is defined on *absolute* sample index, so
    it is identical for every unit and every segment -- a segment boundary
    can fall anywhere inside a cell without shifting the split.
    """
    return (np.asarray(sample_idx) % cell_samples) < train_samples


def _init_worker(tmp_path, dtype, shape, np_ch, chan_y, positions, labels,
                 spike_t, spike_cl, unit_args):
    # One read-only memmap per worker onto the scratch file the parent has
    # already fully written -- no copy, no re-preprocessing.
    _worker.update(
        data=np.memmap(tmp_path, dtype=dtype, mode="r", shape=shape, order="C"),
        np_ch=np_ch, chan_y=chan_y, positions=positions, labels=labels,
        spike_t=spike_t, spike_cl=spike_cl, args=unit_args)


def _fit_one_segment(data, lo, hi, spike_t, spike_cl, tr_mask, target_id, a,
                     np_ch, positions, labels, rng):
    """Fit one filter on [lo, hi) and score its held-out cells.

    Returns (sel_channels, f, peak_idx_abs, peak_scores, noise_std). Peak
    indices come back in absolute recording samples so segments can be
    pooled into one threshold sweep.
    """
    in_seg = (spike_t >= lo) & (spike_t < hi)
    seg_t = spike_t[in_seg] - lo
    seg_cl = spike_cl[in_seg]
    seg_tr = tr_mask[in_seg]
    data_seg = data[lo:hi]

    # Re-pick interferers per segment, from this segment's spikes only. A
    # drifting unit's nearest neighbours are not the same neighbours it had
    # an hour earlier -- re-picking is half the point of segmenting, and
    # costs nothing extra here since the pick is per-fit anyway.
    interferer_ids = gf.auto_pick_interferers_spatial(
        data_seg, seg_t[seg_tr], seg_cl[seg_tr], np_ch, target_id, labels,
        a["auto_interferers"], a["template_length"], a["template_offset"],
        channel_positions=positions, rng=rng)
    if not interferer_ids:
        raise ValueError("no interferers found nearby")

    extras = {}
    f, sel, sel_channels, _, _ = tsr.fit_lcmv(
        data_seg, seg_t, seg_cl, np_ch, target_id, interferer_ids,
        a["n_channels"], a["template_length"], a["template_offset"],
        a["ridge"], a["max_spikes"], rng, extras=extras,
        template_spike_t=seg_t[seg_tr], template_spike_cl=seg_cl[seg_tr])

    if len(sel_channels) != a["n_channels"]:
        raise ValueError(f"only {len(sel_channels)} channels available, "
                         f"need {a['n_channels']}")
    f = np.asarray(f, dtype=np.float32)

    D = gf.filter_output(data_seg[:, sel], f)
    peak_idx, peak_scores = tsr.find_all_peaks(D, a["template_length"] // 2)
    # Per segment, not per unit: each segment has its own filter and its own
    # template, so it has its own detection lag (see
    # generate_filter.detection_lag). Correcting it here is what lets peaks
    # from different segments be pooled into one threshold sweep against one
    # ground-truth train -- an uncorrected pool would be a mixture of
    # differently-shifted trains and would score as noise.
    lag = gf.detection_lag(f, extras["target_template_sel"], a["template_offset"])
    peak_idx = peak_idx - lag

    # Robust noise scale of the score trace. With LCMV's unity gain the
    # target's own response is ~1.0 by construction, so 1/noise_std is the
    # filter's output SNR and is directly comparable across units and
    # segments -- it is the number that collapses when a smeared template
    # makes the constraint set near-collinear.
    noise_std = float(1.4826 * np.median(np.abs(D - np.median(D))))

    keep = ~train_mask(peak_idx + lo, a["cell_samples"], a["train_samples"])
    return (np.asarray(sel_channels, dtype=np.int32), f, peak_idx[keep] + lo,
            peak_scores[keep], noise_std, interferer_ids, lag)


def _fit_one_unit(target_id, spike_count, mode):
    """Runs in a worker process. `mode` is 'global' (force a single segment,
    i.e. the baseline fit under this protocol) or 'segmented'."""
    data = _worker["data"]
    np_ch, chan_y = _worker["np_ch"], _worker["chan_y"]
    positions, labels = _worker["positions"], _worker["labels"]
    spike_t, spike_cl = _worker["spike_t"], _worker["spike_cl"]
    a = _worker["args"]
    n_samples = data.shape[0]

    row = {k: "" for k in SUMMARY_FIELDS}
    row["unit_id"] = target_id
    row["status"] = "failed"
    buf = io.StringIO()
    packed = None

    # Seeded per unit, not shared across units, so a unit's result does not
    # depend on the order the pool happens to complete units in. Also seeded
    # per mode-independently (same seed for both modes) so global and
    # segmented differ only in segmentation, never in RNG draw.
    rng = np.random.default_rng(a["seed"] + target_id)
    tr_mask = train_mask(spike_t, a["cell_samples"], a["train_samples"])

    try:
        target_all = spike_t[spike_cl == target_id]
        if mode == "segmented":
            t_c, y = de.unit_trajectory(
                data, target_all, chan_y, a["template_length"],
                a["template_offset"], a["fs"],
                spikes_per_bin=a["spikes_per_bin"], rng=rng)
            drift_span = float(np.ptp(y)) if y.size else 0.0
            segs = de.segment_from_trajectory(
                t_c, y, target_all, a["fs"], n_samples, a["tol_um"],
                a["min_spikes_per_segment"], a["max_segments"])
        else:
            drift_span = float("nan")
            segs = [(0, n_samples)]

        all_peaks, all_scores, noise_stds, lags = [], [], [], []
        seg_packed, seg_interferers = [], []
        for lo, hi in segs:
            # stdout inside the fit is fit_lcmv's per-interferer constraint
            # check -- useful for one unit, unreadable for hundreds x
            # segments. Captured, not removed: it is still in the log below.
            with contextlib.redirect_stdout(buf):
                sel_channels, f, pidx, pscore, nstd, ints, lag = _fit_one_segment(
                    data, lo, hi, spike_t, spike_cl, tr_mask, target_id, a,
                    np_ch, positions, labels, rng)
            seg_packed.append((lo, hi, sel_channels, f))
            seg_interferers.extend(ints)
            all_peaks.append(pidx)
            all_scores.append(pscore)
            noise_stds.append(nstd)
            lags.append(lag)

        # Segments are in time order and each contributed only its own
        # absolute indices, so the concatenation is already ascending --
        # which sweep_thresholds relies on.
        peak_idx = np.concatenate(all_peaks)
        peak_scores = np.concatenate(all_scores)
        if peak_idx.size == 0:
            raise ValueError("no candidate peaks in held-out cells")

        target_test = target_all[~train_mask(target_all, a["cell_samples"],
                                             a["train_samples"])]
        if target_test.size == 0:
            raise ValueError("no held-out test spikes")
        interferer_test = {}
        for cid in sorted(set(seg_interferers)):
            t = spike_t[spike_cl == cid]
            interferer_test[cid] = t[~train_mask(t, a["cell_samples"],
                                                 a["train_samples"])]

        window = a["template_length"] // 4
        results = tsr.sweep_thresholds(peak_idx, peak_scores, target_test,
                                       interferer_test, window, a["n_thresholds"])
        thresholds = np.array([r["threshold"] for r in results])
        hits = np.array([r["hits"] for r in results])
        total_fp = np.array([r["total_fp"] for r in results])

        test_duration_s = (1.0 - a["train_frac"]) * n_samples / a["fs"]
        recall = hits / max(target_test.size, 1)
        precision = hits / np.maximum(hits + total_fp, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        best = int(np.argmax(f1))

        packed = (segs, seg_packed, np.float32(thresholds[best]))
        row.update({
            "n_channels": a["n_channels"],
            "sel_channels": " | ".join(" ".join(str(c) for c in sc)
                                       for _, _, sc, _ in seg_packed),
            "threshold": float(thresholds[best]),
            "recall": recall[best], "precision": precision[best], "f1": f1[best],
            "fp_rate_hz": total_fp[best] / test_duration_s,
            "n_train_spikes": int(target_all.size - target_test.size),
            "n_test_spikes": int(target_test.size),
            "n_segments": len(segs),
            "drift_span_um": drift_span,
            "segment_bounds_s": " ".join(f"{lo / a['fs']:.1f}" for lo, _ in segs),
            "noise_std": float(np.mean(noise_stds)),
            "detection_lag": " ".join(str(l) for l in lags),
            "status": "ok",
        })
        print(f"unit {target_id} [{mode}] ({spike_count} spikes, {len(segs)} seg, "
              f"span {drift_span:.1f}um): threshold={thresholds[best]:.4f}  "
              f"recall={recall[best]:.2%}  precision={precision[best]:.2%}  "
              f"f1={f1[best]:.3f}  noise_std={np.mean(noise_stds):.3f}  "
              f"lag={lags}", file=buf)

    except Exception as e:
        row["status"] = f"failed: {e}"
        print(f"unit {target_id} [{mode}]: SKIPPED: {e}", file=buf)

    return target_id, row, packed, buf.getvalue()


# --------------------------------------------------------------------- #
# Output packing
# --------------------------------------------------------------------- #

def write_outputs(out_dir, candidates, results, template_length, n_channels, fs):
    """Write the four flat files GpuFilterBank::load() already reads (each
    unit's first segment) plus the schedule of later segments.

    Splitting it this way is deliberate: an unmodified ClosedLoopAllUnits.exe
    pointed at this directory loads the t=0 state and runs, ignoring the
    schedule. Nothing about the existing format changes.
    """
    os.makedirs(out_dir, exist_ok=True)
    ids, chans, filts, thrs, events = [], [], [], [], []
    rows = []
    for uid in candidates:
        row, packed = results[uid]
        rows.append(row)
        if packed is None:
            continue
        segs, seg_packed, threshold = packed
        unit_index = len(ids)
        ids.append(uid)
        chans.append(seg_packed[0][2])
        filts.append(seg_packed[0][3])
        thrs.append(threshold)
        for lo, _, sel_channels, f in seg_packed[1:]:
            events.append((lo / fs, unit_index, uid, sel_channels, f, float(threshold)))

    if ids:
        np.asarray(ids, dtype=np.int32).tofile(os.path.join(out_dir, "unit_ids.bin"))
        np.stack(chans).astype(np.int32).tofile(os.path.join(out_dir, "channels.bin"))
        np.stack(filts).astype(np.float32).tofile(os.path.join(out_dir, "filters.bin"))
        np.asarray(thrs, dtype=np.float32).tofile(os.path.join(out_dir, "thresholds.bin"))

    # Events sorted by time: DriftSchedule walks this list forward once and
    # never searches, so the online side stays O(events applied).
    events.sort(key=lambda e: e[0])
    write_schedule(out_dir, events, template_length, n_channels)

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(ids), len(events)


def write_schedule(out_dir, events, template_length, n_channels):
    """drift_schedule.bin: a tiny header then one fixed-stride record per
    swap event. Fixed stride for the same reason the filter bank itself has
    one -- the reader indexes rather than parses. drift_schedule.json carries
    the same events in readable form for offline inspection; the C++ side
    reads only the .bin.
    """
    path = os.path.join(out_dir, "drift_schedule.bin")
    with open(path, "wb") as fh:
        # magic, version, nEvents, templateLength, nChannelsPerUnit
        fh.write(struct.pack("<5i", 0x44524654, 1, len(events),
                             template_length, n_channels))
        for t_s, unit_index, _uid, sel_channels, f, threshold in events:
            fh.write(struct.pack("<fii", float(t_s), int(unit_index), 0))
            fh.write(np.asarray(sel_channels, dtype=np.int32).tobytes())
            fh.write(np.asarray(f, dtype=np.float32).tobytes())
            fh.write(struct.pack("<f", float(threshold)))

    with open(os.path.join(out_dir, "drift_schedule.json"), "w") as fh:
        json.dump({
            "version": 1,
            "template_length": template_length,
            "n_channels_per_unit": n_channels,
            "n_events": len(events),
            "events": [{"t_s": round(float(t), 3), "unit_index": int(ui),
                        "unit_id": int(uid),
                        "channels": [int(c) for c in sc],
                        "threshold": float(th)}
                       for t, ui, uid, sc, _f, th in events],
        }, fh, indent=1)


# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--meta-path")
    ap.add_argument("--channel-map-json")
    ap.add_argument("--mode", choices=["global", "segmented", "both"], default="both",
                    help="'global' is the baseline fit under this script's "
                         "interleaved protocol (one filter per unit); "
                         "'segmented' is drift-aware. 'both' runs each from "
                         "one preprocessing pass, which is the only "
                         "controlled comparison -- same train samples, same "
                         "test samples, same RNG seeds.")
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="Fraction of EACH interleave cell used for fitting.")
    ap.add_argument("--cell-s", type=float, default=5.0,
                    help="Interleave cell length in seconds. Must be short "
                         "against the drift timescale (so train and test see "
                         "the unit in the same place) and long against the "
                         "template (so cell edges are negligible).")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-spikes", type=int, default=2000)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--n-thresholds", type=int, default=60)
    ap.add_argument("--auto-interferers", type=int, default=5)
    ap.add_argument("--min-spikes", type=int, default=200)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tol-um", type=float, default=12.0,
                    help="Depth span one filter is asked to cover. A unit "
                         "whose estimated trajectory spans less than this "
                         "gets exactly one segment, i.e. the baseline fit.")
    ap.add_argument("--min-spikes-per-segment", type=int, default=250)
    ap.add_argument("--max-segments", type=int, default=8)
    ap.add_argument("--spikes-per-bin", type=int, default=150,
                    help="Spikes per trajectory-estimation bin. See "
                         "drift_estimate.unit_trajectory.")
    ap.add_argument("--scratch-dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    args.filter = True
    args.car = True
    args.causal_highpass = True
    print("Loading + preprocessing (float32, shared across all units)...")
    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(
        args, rng, dtype=np.float32)

    _, _, labels = gf.load_kilosort(args.ks_dir)
    positions = gf.load_channel_positions_json(args.channel_map_json) \
        if args.channel_map_json else {}
    chan_y = de.channel_y_for_group(np_ch, positions)

    cell_samples = max(1, int(round(args.cell_s * fs)))
    train_samples = int(round(args.train_frac * cell_samples))
    n_cells = data.shape[0] / cell_samples
    print(f"Interleaved protocol: {n_cells:.0f} cells of {args.cell_s:g}s, "
          f"first {args.train_frac:.0%} of each cell trains, rest is held out.")

    unit_ids, counts = np.unique(spike_cl, return_counts=True)
    order = np.argsort(counts)[::-1]
    candidates = []
    for idx in order:
        uid, cnt = int(unit_ids[idx]), int(counts[idx])
        if labels.get(uid, "").lower() == "noise" or cnt < args.min_spikes:
            continue
        candidates.append(uid)
        if args.max_units and len(candidates) >= args.max_units:
            break
    print(f"{len(candidates)} candidate units of {len(unit_ids)} clusters.")

    # Release the parent's read-write handle before workers open their own
    # read-only ones -- on Windows a lingering w+ memmap makes concurrent
    # opens onto the same file unreliable.
    data_path, data_dtype, data_shape = data.filename, data.dtype, data.shape
    del data
    import gc
    gc.collect()

    unit_args = dict(auto_interferers=args.auto_interferers,
                     template_length=args.template_length,
                     template_offset=args.template_offset,
                     n_channels=args.n_channels, ridge=args.ridge,
                     max_spikes=args.max_spikes, n_thresholds=args.n_thresholds,
                     fs=fs, seed=args.seed, cell_samples=cell_samples,
                     train_samples=train_samples, train_frac=args.train_frac,
                     tol_um=args.tol_um, spikes_per_bin=args.spikes_per_bin,
                     min_spikes_per_segment=args.min_spikes_per_segment,
                     max_segments=args.max_segments)
    init_args = (data_path, data_dtype, data_shape, np_ch, chan_y, positions,
                 labels, spike_t, spike_cl, unit_args)

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    modes = ["global", "segmented"] if args.mode == "both" else [args.mode]

    for mode in modes:
        out_dir = os.path.join(args.out_dir, mode) if len(modes) > 1 else args.out_dir
        print(f"\n=== mode={mode} -> {out_dir} "
              f"({len(candidates)} units, {n_workers} workers) ===")
        t0 = time.time()
        results = {}
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=init_args) as pool:
            futs = {pool.submit(_fit_one_unit, uid,
                                int(counts[unit_ids == uid][0]), mode): uid
                    for uid in candidates}
            for i, fut in enumerate(as_completed(futs)):
                uid, row, packed, log = fut.result()
                results[uid] = (row, packed)
                print(f"[{i + 1}/{len(candidates)}] {log.splitlines()[-1]}"
                      if log.strip() else f"[{i + 1}/{len(candidates)}]")
        n_ok, n_events = write_outputs(out_dir, candidates, results,
                                       args.template_length, args.n_channels, fs)
        f1s = [float(r["f1"]) for r, _ in results.values() if r["status"] == "ok"]
        print(f"{n_ok}/{len(candidates)} fit ok, {n_events} schedule events, "
              f"mean f1 {np.mean(f1s) if f1s else float('nan'):.3f}, "
              f"{time.time() - t0:.0f}s -> {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
