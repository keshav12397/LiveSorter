"""
calibrate_all_units.py
=======================

Batch version of calibrate_for_closedloop.py: fits an LCMV filter and sweeps
a detection threshold for EVERY Kilosort unit in a session (not just one
hand-picked target), and packs the results into the flat, fixed-stride binary
format ClosedLoop's GPU pipeline (GpuFilterBank) loads directly -- no
per-unit channels_/filter_/threshold_<id>.bin files, since 500 units x 3
tiny files each is an awkward on-disk shape for a single batched GPU upload.

This does NOT reimplement any fit/sweep math -- it's a loop around exactly
the same building blocks calibrate_for_closedloop.py uses:
generate_filter.auto_pick_interferers_spatial, threshold_sweep_real.fit_lcmv,
threshold_sweep_real.sweep_thresholds. See those modules' docstrings/comments
for why (this session has twice silently lost detection recall/precision to
a second, drifted reimplementation of this math).

Precision: the fit itself (noise covariance estimate + LCMV linear solve)
runs at whatever precision numpy naturally computes it at (effectively
float64, via some already-present internal np.zeros()/np.eye() calls in
generate_filter.py) -- deliberately left alone, since that's a one-shot
per-unit linear solve where stability matters far more than matching the
GPU's runtime precision, and this session's history includes real regressions
from touching validated fit code. What DOES need to match the online GPU
kernel is the artifact actually deployed and the threshold picked for it: the
saved filter taps are cast to float32, and the held-out threshold sweep scores
the SAME float32-cast filter against float32-cast data (load_and_prepare's
scratch memmap is allocated float32 here, and generate_filter.filter_output
now preserves input dtype instead of hardcoding float64) -- so the chosen
threshold sits at the actual float32 score distribution the GPU will produce,
not a higher-precision approximation of it. Exact bit-for-bit parity with the
GPU's own reduction order isn't attempted (unrealistic and unnecessary); this
gets the operating point onto the right distribution instead.

Output (in --out-dir), only for units that fit successfully, all in one
shared order across the four files:
    unit_ids.bin    int32[nUnits]
    channels.bin    int32[nUnits * n_channels]   row-major
    filters.bin     float32[nUnits * template_length * n_channels]  row-major per unit
    thresholds.bin  float32[nUnits]
    summary.csv     one row per attempted unit (including failures), with
                     n_channels, sel_channels, threshold, recall, precision,
                     f1, fp_rate_hz, n_train_spikes, n_test_spikes, status

Example
-------
    python calibrate_all_units.py \\
        --ks-dir D:/test_newsorter/ks_out \\
        --bin-path D:/test_newsorter/rawData/Lav69_d1.0_g0_tcat.imec0.ap.bin \\
        --channel-map-json D:/test_newsorter/rawData/shank1only.json \\
        --auto-interferers 5 \\
        --min-spikes 200 \\
        --out-dir D:/test_newsorter/filters_all_units
"""

import argparse
import csv
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import generate_filter as gf
import threshold_sweep_real as tsr

# Per-unit work (channel-subset gather from the shared preprocessed recording,
# full-length matched-filter convolution, peak-finding, threshold sweep) only
# ever touches that one unit's own small channel slice and reads from a
# scratch file that's already fully built by the time any unit fitting
# starts -- so units are embarrassingly parallel across a process pool.
# Pinning each worker's BLAS thread count to 1 matters here: numpy's BLAS
# calls otherwise each try to fan out across all cores on their own, and
# N worker processes doing that simultaneously massively oversubscribes a
# 6-core/12-thread machine, which would make the pool *slower* than serial.
# This has to be set before numpy/BLAS initializes in each child process --
# spawned children re-run `import numpy` fresh, and they inherit the parent's
# environment at spawn time, so setting this in the parent before the pool is
# created (not inside the worker function, which runs after numpy is already
# loaded) is what actually makes it take effect.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_worker = {}


def _init_worker(tmp_path, dtype, shape, order, np_ch, positions, labels,
                  spike_t, spike_cl, split_t, unit_args):
    # Each worker opens its own read-only memmap onto the SAME scratch file
    # the main process already streamed the preprocessed recording into --
    # no copying, no re-preprocessing, just independent read handles onto
    # one shared on-disk array.
    _worker["data"] = np.memmap(tmp_path, dtype=dtype, mode="r", shape=shape, order=order)
    _worker["np_ch"] = np_ch
    _worker["positions"] = positions
    _worker["labels"] = labels
    _worker["spike_t"] = spike_t
    _worker["spike_cl"] = spike_cl
    _worker["split_t"] = split_t
    _worker["args"] = unit_args


def _fit_one_unit(target_id, spike_count):
    """Runs in a worker process. Same fit+sweep body calibrate_all_units.py
    always ran, just relocated so a process pool can call it per unit --
    no algorithm changes."""
    data = _worker["data"]
    np_ch = _worker["np_ch"]
    positions = _worker["positions"]
    labels = _worker["labels"]
    spike_t = _worker["spike_t"]
    spike_cl = _worker["spike_cl"]
    split_t = _worker["split_t"]
    a = _worker["args"]

    data_train, data_test = data[:split_t], data[split_t:]

    def split(times):
        return times[times < split_t], times[times >= split_t]

    spike_t_train = spike_t[spike_t < split_t]
    spike_cl_train = spike_cl[spike_t < split_t]

    buf = io.StringIO()
    row = {"unit_id": target_id, "n_channels": "", "sel_channels": "",
           "threshold": "", "recall": "", "precision": "", "f1": "",
           "fp_rate_hz": "", "n_train_spikes": "", "n_test_spikes": "",
           "status": "failed"}
    packed = None
    # One RNG per unit (seeded off unit id, not shared across units the way
    # the old serial version's single advancing rng was) -- makes each
    # unit's result independent of what order the pool happens to complete
    # units in, which the old sequential-rng-state version implicitly
    # depended on. Reused across both calls below (not re-seeded per call)
    # to match the original single-stream behavior within a unit.
    unit_rng = np.random.default_rng(a["seed"] + target_id)
    try:
        interferer_ids = gf.auto_pick_interferers_spatial(
            data, spike_t, spike_cl, np_ch, target_id, labels,
            a["auto_interferers"], a["template_length"], a["template_offset"],
            channel_positions=positions, rng=unit_rng)
        if not interferer_ids:
            raise ValueError("no interferers found nearby")

        f, sel, sel_channels, _, _ = tsr.fit_lcmv(
            data_train, spike_t_train, spike_cl_train, np_ch,
            target_id, interferer_ids, a["n_channels"],
            a["template_length"], a["template_offset"],
            a["ridge"], a["max_spikes"], unit_rng)

        if len(sel_channels) != a["n_channels"]:
            raise ValueError(f"only {len(sel_channels)} channels available, "
                              f"need {a['n_channels']}")

        f = np.asarray(f, dtype=np.float32)

        target_train, target_test = split(spike_t[spike_cl == target_id])
        interferer_test = {cid: split(spike_t[spike_cl == cid])[1]
                            for cid in interferer_ids}
        if len(target_test) == 0:
            raise ValueError("no held-out test spikes")

        data_test_sel = data_test[:, sel]
        D_test = gf.filter_output(data_test_sel, f)
        window = a["template_length"] // 4
        min_sep = a["template_length"] // 2
        peak_idx, peak_scores = tsr.find_all_peaks(D_test, min_sep)

        target_test_rel = target_test - split_t
        interferer_test_rel = {cid: t - split_t for cid, t in interferer_test.items()}
        results = tsr.sweep_thresholds(peak_idx, peak_scores, target_test_rel,
                                        interferer_test_rel, window, a["n_thresholds"])

        thresholds = np.array([r["threshold"] for r in results])
        hits = np.array([r["hits"] for r in results])
        total_fp = np.array([r["total_fp"] for r in results])
        n_target_test = max(len(target_test), 1)
        test_duration_s = (data.shape[0] - split_t) / a["fs"]

        recall = hits / n_target_test
        precision = hits / np.maximum(hits + total_fp, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        fp_rate_hz = total_fp / test_duration_s

        best_idx = int(np.argmax(f1))
        best_threshold = float(thresholds[best_idx])

        packed = (np.asarray(sel_channels, dtype=np.int32), f, np.float32(best_threshold))

        row.update({
            "n_channels": len(sel_channels),
            "sel_channels": " ".join(str(c) for c in sel_channels),
            "threshold": best_threshold,
            "recall": recall[best_idx], "precision": precision[best_idx],
            "f1": f1[best_idx], "fp_rate_hz": fp_rate_hz[best_idx],
            "n_train_spikes": len(target_train), "n_test_spikes": len(target_test),
            "status": "ok",
        })
        print(f"unit {target_id} ({spike_count} spikes): "
              f"threshold={best_threshold:.4f}  recall={recall[best_idx]:.2%}  "
              f"precision={precision[best_idx]:.2%}  f1={f1[best_idx]:.3f}", file=buf)

    except Exception as e:
        row["status"] = f"failed: {e}"
        print(f"unit {target_id} ({spike_count} spikes): SKIPPED: {e}", file=buf)

    return target_id, row, packed, buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--meta-path")
    ap.add_argument("--channel-map-json")
    ap.add_argument("--n-channels", type=int, default=5,
                     help="Fixed per-unit channel count -- kept the same for "
                          "every unit so the packed output has a uniform "
                          "stride the GPU loader can index without a "
                          "variable-length (CSR-style) lookup.")
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-spikes", type=int, default=2000)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--n-thresholds", type=int, default=60)
    ap.add_argument("--auto-interferers", type=int, default=5,
                     help="Per-unit spatial auto-pick count (each unit "
                          "nulls out its own N nearest active neighbors) -- "
                          "see generate_filter.auto_pick_interferers_spatial.")
    ap.add_argument("--min-spikes", type=int, default=200,
                     help="Skip units with fewer than this many total spikes "
                          "(too little data to fit/validate a filter).")
    ap.add_argument("--max-units", type=int, default=0,
                     help="Cap the candidate list to this many units (highest "
                          "spike count first), for a bounded/reproducible "
                          "partial-probe run. 0 (default) = no cap, all "
                          "qualifying units.")
    ap.add_argument("--workers", type=int, default=0,
                     help="Number of unit-fitting worker processes to run in "
                          "parallel (each unit's fit+sweep only touches its "
                          "own small channel slice of the shared preprocessed "
                          "recording, so units are independent). 0 (default) "
                          "= os.cpu_count(). Pass 1 to force the old serial "
                          "behavior (e.g. for debugging a single unit's "
                          "traceback without pool noise).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

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

    print("Loading + preprocessing (float32, shared across all units)...")
    # NOT using order='F' here despite each unit gathering a different small
    # channel subset (which is exactly what 'F' speeds up -- see
    # load_and_prepare's docstring): measured great gather speedups (12-14x)
    # in isolation, but real-world use surfaced a worse, RAM-independence-
    # breaking problem on the WRITE side. This function builds the array
    # incrementally (~2M-sample chunks, required by the causal filter's
    # cross-chunk state), and writing each chunk into an 'F'-order memmap
    # scatters it across 96 far-apart file offsets (one per channel) instead
    # of one contiguous block -- 20+ scattered chunk writes turned out far
    # slower in practice than the bulk single-assignment write my benchmark
    # tested. The "obvious" fix (build the chunked result in a RAM buffer,
    # then do one fast bulk write to an 'F'-order file at the end) was
    # rejected: it requires holding the whole recording in RAM regardless of
    # scratch-file order, which doesn't generalize to smaller machines --
    # see the all_units branch commit history for the fuller story. Left as
    # default 'C' order; the per-unit gather cost is real but the
    # noise_covariance_vectorized fix (see generate_filter.py) is the
    # verified, RAM-size-independent win that's actually in effect here.
    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(
        load_args, rng, dtype=np.float32)

    _, _, labels = gf.load_kilosort(args.ks_dir)
    positions = gf.load_channel_positions_json(args.channel_map_json) \
        if args.channel_map_json else {}

    split_t = int(args.train_frac * data.shape[0])
    print(f"Train: [0, {split_t}) ({split_t/fs:.1f}s)   "
          f"Test: [{split_t}, {data.shape[0]}) ({(data.shape[0]-split_t)/fs:.1f}s)")

    unit_ids, counts = np.unique(spike_cl, return_counts=True)
    order = np.argsort(counts)[::-1]  # highest spike count first, for --max-units to pick a stable/reproducible set
    candidates = []
    for idx in order:
        uid, cnt = int(unit_ids[idx]), int(counts[idx])
        if labels.get(uid, "").lower() == "noise":
            continue
        if cnt < args.min_spikes:
            continue
        candidates.append(uid)
        if args.max_units and len(candidates) >= args.max_units:
            break
    print(f"{len(candidates)} candidate units (of {len(unit_ids)} total clusters, "
          f"excluding noise-labeled and <{args.min_spikes}-spike clusters"
          + (f", capped to top {args.max_units} by spike count)" if args.max_units else ")"))

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    print(f"Fitting {len(candidates)} units across {n_workers} worker process(es)...")

    # Drop the main process's read-write handle onto the scratch file before
    # workers each open their own read-only handle onto it. load_and_prepare
    # already flushed all writes; nothing further is written to this file --
    # but on Windows an open memmap handle needs to be explicitly released
    # (del + gc) before other opens onto the same file
    # are guaranteed to behave, rather than assuming a lingering w+ handle is
    # harmless to read past.
    data_path, data_dtype, data_shape = data.filename, data.dtype, data.shape
    del data
    import gc
    gc.collect()
    data = np.memmap(data_path, dtype=data_dtype, mode="r", shape=data_shape, order="C")

    unit_args = dict(auto_interferers=args.auto_interferers,
                      template_length=args.template_length,
                      template_offset=args.template_offset,
                      n_channels=args.n_channels, ridge=args.ridge,
                      max_spikes=args.max_spikes, n_thresholds=args.n_thresholds,
                      fs=fs, seed=args.seed)
    # np.memmap.filename recovers the scratch file load_and_prepare streamed
    # into -- no need to plumb the path out of that function separately.
    # calibrate_all_units.py never requests order='F' from it (see the
    # load call above), so 'C' is always the layout here.
    init_args = (data.filename, data.dtype, data.shape, "C", np_ch, positions,
                 labels, spike_t, spike_cl, split_t, unit_args)

    results_by_id = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                              initargs=init_args) as pool:
        futures = {pool.submit(_fit_one_unit, uid, int(counts[unit_ids == uid][0])): uid
                   for uid in candidates}
        for i, fut in enumerate(as_completed(futures)):
            target_id, row, packed, log_text = fut.result()
            results_by_id[target_id] = (row, packed)
            print(f"[{i+1}/{len(candidates)}] {log_text}", end="")

    packed_ids, packed_channels, packed_filters, packed_thresholds = [], [], [], []
    summary_rows = []
    for target_id in candidates:
        row, packed = results_by_id[target_id]
        summary_rows.append(row)
        if packed is not None:
            sel_channels, f, best_threshold = packed
            packed_ids.append(target_id)
            packed_channels.append(sel_channels)
            packed_filters.append(f)  # (template_length, n_channels), float32
            packed_thresholds.append(best_threshold)

    n_ok = len(packed_ids)
    print(f"\n{n_ok}/{len(candidates)} units fit successfully.")

    if n_ok > 0:
        unit_ids_arr = np.asarray(packed_ids, dtype=np.int32)
        channels_arr = np.stack(packed_channels).astype(np.int32)      # (nUnits, n_channels)
        filters_arr = np.stack(packed_filters).astype(np.float32)      # (nUnits, L, n_channels)
        thresholds_arr = np.asarray(packed_thresholds, dtype=np.float32)

        unit_ids_arr.tofile(os.path.join(args.out_dir, "unit_ids.bin"))
        channels_arr.tofile(os.path.join(args.out_dir, "channels.bin"))
        filters_arr.tofile(os.path.join(args.out_dir, "filters.bin"))
        thresholds_arr.tofile(os.path.join(args.out_dir, "thresholds.bin"))
        print(f"Wrote unit_ids.bin ({unit_ids_arr.size}), "
              f"channels.bin ({channels_arr.shape}), "
              f"filters.bin ({filters_arr.shape}), "
              f"thresholds.bin ({thresholds_arr.size}) to {args.out_dir}")
    else:
        print("No units succeeded -- not writing packed .bin files.", file=sys.stderr)

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()) if summary_rows else
                                 ["unit_id", "status"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")

    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
