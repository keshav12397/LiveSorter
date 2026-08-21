"""
refit_exclusion_ablation.py
=============================

Part A of the refit-f1 investigation: does caching noise_cov_by_lag across
units change what R means in a way that costs or buys detection f1?

fit_lcmv's noise covariance R is built from `local_spike_times` -- the
target unit's own spikes plus its chosen interferers' spikes, and NOTHING
else. That exclusion set is per-unit. A single cached `cov_by_lag`
(generate_filter.noise_cov_by_lag) has exactly one exclusion set, so sharing
it across units necessarily redefines R for every unit but the one it was
built for. This script measures what that redefinition costs, on the SAME
units, SAME channel selection, SAME templates -- the only thing that varies
across the three fits below is which spikes were excluded when the noise
covariance was estimated.

Three variants, one fit_lcmv call each (threshold_sweep_real.fit_lcmv's
`cov_by_lag` parameter -- see its docstring):

    peruser      cov_by_lag=None: fit_lcmv's existing behaviour, a rescan
                 that excludes only this unit's own + interferers' spikes.
                 THE BASELINE, current production.
    shared_all   cov_by_lag built ONCE per session, excluding every sorted
                 spike from every cluster.
    shared_none  cov_by_lag built ONCE per session, excluding nothing.

Channel selection and templates are held fixed across a unit's three fits by
seeding `rng` identically before each fit_lcmv call and passing the same
`interferer_ids` (chosen once, before the three variants run) -- so `sel`
(hence R's shape and channel content) cannot differ between variants for
reasons other than R's own source. What differs is purely which R the LCMV
solve sees.

Protocol: same chronological train/test split calibrate_drift_aware.py uses
(train = first --train-frac of the WHOLE recording, test = the untouched
remainder), same scoring path (threshold_sweep_real.find_all_peaks +
sweep_thresholds, best-F1 threshold per unit per variant, one free parameter
each). No new scorer.

Output: <out-dir>/summary.csv, one row per (unit, variant).

Example
-------
    python refit_exclusion_ablation.py \\
        --ks-dir D:/sim_probe_drift/sim_ks \\
        --bin-path D:/sim_probe_drift/sim_g0_t0.imec0.ap.bin \\
        --channel-map-json D:/test_newsorter/rawData/shank1only.json \\
        --max-units 10 --workers 4 --out-dir D:/refit_work/exclusion_smoke
"""

import argparse
import contextlib
import csv
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import generate_filter as gf
import threshold_sweep_real as tsr

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_worker = {}

# peruser_capped is the control that makes the comparison mean anything.
#
# `peruser` rescans the WHOLE train half (it only needs 5 channels, so it can
# afford to). The shared variants are built over all 96 channels and are
# therefore capped at --cov-t-max seconds, ~30x less data. Without a third
# arm holding data constant, "shared is worse" would be unattributable
# between the exclusion set and the sample count.
#
# peruser_capped keeps per-unit exclusion and takes the SAME window as the
# shared arms. So:
#     peruser        vs peruser_capped  ->  what the shorter window costs
#     peruser_capped vs shared_all      ->  what the shared exclusion costs
# and only the second of those is the question this script exists to answer.
VARIANTS = ["peruser", "peruser_capped", "shared_all", "shared_none"]

SUMMARY_FIELDS = ["unit_id", "variant", "n_channels", "sel_channels",
                   "threshold", "recall", "precision", "f1", "fp_rate_hz",
                   "n_train_spikes", "n_test_spikes", "detection_lag",
                   "interferer_ids", "status"]


def _init_worker(tmp_path, dtype, shape, np_ch, positions, labels,
                  spike_t, spike_cl, unit_args, cov_by_lag_all, cov_by_lag_none):
    _worker.update(
        data=np.memmap(tmp_path, dtype=dtype, mode="r", shape=shape, order="C"),
        np_ch=np_ch, positions=positions, labels=labels,
        spike_t=spike_t, spike_cl=spike_cl, args=unit_args,
        cov_by_lag_all=cov_by_lag_all, cov_by_lag_none=cov_by_lag_none)


def _fit_and_score(data_train, data_test, spike_t_train, spike_cl_train,
                    np_ch, target_id, interferer_ids, a, rng, cov_by_lag,
                    split_t, n_samples, spike_t_all, spike_cl_all,
                    cov_max_samples=None):
    """One fit_lcmv call (R source = cov_by_lag, or a per-unit rescan if
    None) plus scoring against the untouched test half. Mirrors
    calibrate_drift_aware.py's _fit_filter_on_window/_evaluate_on_test, with
    the R source parameterized -- that pair doesn't take a cov_by_lag
    argument, so it isn't reused verbatim, but every actual fitting/scoring
    primitive it calls (fit_lcmv, find_all_peaks, detection_lag,
    sweep_thresholds) is the same one, not a reimplementation.
    """
    extras = {}
    f, sel, sel_channels, _, _ = tsr.fit_lcmv(
        data_train, spike_t_train, spike_cl_train, np_ch, target_id,
        interferer_ids, a["n_channels"], a["template_length"],
        a["template_offset"], a["ridge"], a["max_spikes"], rng,
        extras=extras, cov_by_lag=cov_by_lag,
        cov_max_samples=cov_max_samples)
    if len(sel_channels) != a["n_channels"]:
        raise ValueError(f"only {len(sel_channels)} channels available, "
                          f"need {a['n_channels']}")
    f = np.asarray(f, dtype=np.float32)
    lag = gf.detection_lag(f, extras["target_template_sel"], a["template_offset"])

    data_test_sel = data_test[:, sel]
    D_test = gf.filter_output(data_test_sel, f)
    peak_idx, peak_scores = tsr.find_all_peaks(D_test, a["template_length"] // 2)
    peak_idx = peak_idx - lag

    target_test = spike_t_all[(spike_cl_all == target_id) & (spike_t_all >= split_t)] - split_t
    if target_test.size == 0:
        raise ValueError("no held-out test spikes")
    interferer_test = {cid: spike_t_all[(spike_cl_all == cid) & (spike_t_all >= split_t)] - split_t
                        for cid in interferer_ids}

    window = a["template_length"] // 4
    test_duration_s = (n_samples - split_t) / a["fs"]
    results = tsr.sweep_thresholds(peak_idx, peak_scores, target_test,
                                    interferer_test, window, a["n_thresholds"])
    thresholds = np.array([r["threshold"] for r in results])
    hits = np.array([r["hits"] for r in results])
    total_fp = np.array([r["total_fp"] for r in results])

    recall = hits / max(target_test.size, 1)
    precision = hits / np.maximum(hits + total_fp, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(np.argmax(f1))
    return dict(sel_channels=sel_channels, threshold=float(thresholds[best]),
                recall=float(recall[best]), precision=float(precision[best]),
                f1=float(f1[best]), fp_rate_hz=float(total_fp[best] / test_duration_s),
                n_test_spikes=int(target_test.size), lag=lag)


def _fit_one_unit(target_id, spike_count):
    """Runs in a worker process: fits + scores all three exclusion variants
    for one unit, on identical channel selection / templates (same seeded
    rng, same interferer_ids for each variant)."""
    data = _worker["data"]
    np_ch, positions, labels = _worker["np_ch"], _worker["positions"], _worker["labels"]
    spike_t, spike_cl = _worker["spike_t"], _worker["spike_cl"]
    a = _worker["args"]
    cov_by_lag_all, cov_by_lag_none = _worker["cov_by_lag_all"], _worker["cov_by_lag_none"]
    n_samples = data.shape[0]
    split_t = a["split_t"]

    rows = []
    buf = io.StringIO()

    rng_pick = np.random.default_rng(a["seed"] + target_id)
    data_train, data_test = data[:split_t], data[split_t:]
    spike_t_train = spike_t[spike_t < split_t]
    spike_cl_train = spike_cl[spike_t < split_t]

    try:
        target_train = spike_t_train[spike_cl_train == target_id]
        if target_train.size == 0:
            raise ValueError("no train spikes")

        with contextlib.redirect_stdout(buf):
            interferer_ids = gf.auto_pick_interferers_spatial(
                data_train, spike_t_train, spike_cl_train, np_ch, target_id,
                labels, a["auto_interferers"], a["template_length"],
                a["template_offset"], channel_positions=positions, rng=rng_pick)
        if not interferer_ids:
            raise ValueError("no interferers found nearby")

        for variant in VARIANTS:
            row = {k: "" for k in SUMMARY_FIELDS}
            row["unit_id"], row["variant"] = target_id, variant
            row["status"] = "failed"
            try:
                cov_by_lag = {"peruser": None, "peruser_capped": None,
                              "shared_all": cov_by_lag_all,
                              "shared_none": cov_by_lag_none}[variant]
                cov_cap = a["cov_max_samples"] if variant == "peruser_capped" else None
                # Same seed every variant -- mean_waveform's rng.choice
                # subsample (target + interferers) is then bit-identical
                # across variants, so `sel` cannot differ for any reason
                # other than the R source itself.
                rng_fit = np.random.default_rng(a["seed"] + target_id + 1000)
                with contextlib.redirect_stdout(buf):
                    scored = _fit_and_score(
                        data_train, data_test, spike_t_train, spike_cl_train,
                        np_ch, target_id, interferer_ids, a, rng_fit,
                        cov_by_lag, split_t, n_samples, spike_t, spike_cl,
                        cov_max_samples=cov_cap)
                row.update({
                    "n_channels": a["n_channels"],
                    "sel_channels": " ".join(str(c) for c in scored["sel_channels"]),
                    "threshold": scored["threshold"], "recall": scored["recall"],
                    "precision": scored["precision"], "f1": scored["f1"],
                    "fp_rate_hz": scored["fp_rate_hz"],
                    "n_train_spikes": int(target_train.size),
                    "n_test_spikes": scored["n_test_spikes"],
                    "detection_lag": scored["lag"],
                    "interferer_ids": " ".join(str(c) for c in interferer_ids),
                    "status": "ok",
                })
                print(f"unit {target_id} [{variant}]: f1={scored['f1']:.3f} "
                      f"recall={scored['recall']:.2%} precision={scored['precision']:.2%}",
                      file=buf)
            except Exception as e:
                row["status"] = f"failed: {e}"
                print(f"unit {target_id} [{variant}]: SKIPPED: {e}", file=buf)
            rows.append(row)

    except Exception as e:
        for variant in VARIANTS:
            row = {k: "" for k in SUMMARY_FIELDS}
            row["unit_id"], row["variant"], row["status"] = target_id, variant, f"failed: {e}"
            rows.append(row)
        print(f"unit {target_id}: SKIPPED: {e}", file=buf)

    return target_id, rows, buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--meta-path")
    ap.add_argument("--channel-map-json")
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-spikes", type=int, default=2000)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--n-thresholds", type=int, default=60)
    ap.add_argument("--auto-interferers", type=int, default=5)
    ap.add_argument("--min-spikes", type=int, default=200)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cov-t-max", type=float, default=60.0,
                     help="Starting window (seconds of the train half) for "
                          "the SHARED cov_by_lag arrays (shared_all, "
                          "shared_none) and for peruser_capped's rescan. "
                          "Matches generate_filter.py's own --t-max default "
                          "for noise-covariance estimation. Doubled "
                          "automatically until shared_all finds enough "
                          "spike-free gaps (with every cluster's spikes "
                          "excluded, a short window can have NONE at all --"
                          "measured on D:/sim_probe_drift, 60s wasn't "
                          "enough with 160 units excluded, 120s was) -- the "
                          "window that succeeds is then reused for "
                          "shared_none and peruser_capped too, so all three "
                          "capped variants see the same amount of data and "
                          "only their exclusion set differs. Necessary at "
                          "all because noise_cov_by_lag's cost is "
                          "O(nlags * n_ch^2 * segment_length) over the FULL "
                          "channel group (not just a handful of selected "
                          "channels, unlike the uncapped per-unit baseline) "
                          "-- shared_none with nothing excluded makes the "
                          "whole window ONE segment, and uncapped that did "
                          "not finish in practical time on a 900s window.")
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
    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(args, rng, dtype=np.float32)

    _, _, labels = gf.load_kilosort(args.ks_dir)
    positions = gf.load_channel_positions_json(args.channel_map_json) \
        if args.channel_map_json else {}

    split_t = int(round(args.train_frac * data.shape[0]))
    print(f"Chronological protocol: train [0, {split_t}) ({split_t / fs:.1f}s), "
          f"test [{split_t}, {data.shape[0]}) ({(data.shape[0] - split_t) / fs:.1f}s).")

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

    L, off = args.template_length, args.template_offset
    data_train = data[:split_t]
    spike_t_train_all = spike_t[spike_t < split_t]

    # ONE window length for shared_all, shared_none AND peruser_capped, or
    # the "shared exclusion costs X" comparison is confounded by "shared
    # also saw a different amount of data than peruser_capped" (this is
    # exactly the confound peruser_capped exists to remove -- see the
    # VARIANTS comment above). Start at --cov-t-max and grow it: with every
    # cluster's spikes excluded (shared_all), a short window can have NO
    # gap >= 2*template_length at all -- measured on this session, a flat
    # 60s cap raises noise_cov_by_lag's "No sufficiently long spike-free
    # segments" ValueError, since 160 units' spikes blanket a short window
    # almost solid. Doubling until shared_all succeeds finds the shortest
    # window that actually has usable gaps, instead of guessing one.
    cand_s = args.cov_t_max
    cov_by_lag_all = None
    while cov_by_lag_all is None:
        cand_samples = min(data_train.shape[0], int(cand_s * fs))
        print(f"Building shared cov_by_lag over the FULL channel group "
              f"(first {cand_samples / fs:.1f}s of train), "
              f"excluding ALL sorted spikes (variant 'shared_all')...")
        t0 = time.time()
        try:
            cov_by_lag_all = gf.noise_cov_by_lag(
                data_train, np.sort(spike_t_train_all), L, off, cand_samples)
        except ValueError as e:
            if cand_samples >= data_train.shape[0]:
                raise
            print(f"  {e} -- doubling window and retrying")
            cand_s *= 2
            continue
        print(f"  done ({time.time() - t0:.1f}s), shape {cov_by_lag_all.shape}, "
              f"{cov_by_lag_all.nbytes / 1e6:.1f} MB")
    cov_max_samples = cand_samples

    print(f"Building shared cov_by_lag over the FULL channel group "
          f"(first {cov_max_samples / fs:.1f}s of train, same window as "
          f"shared_all), excluding NOTHING (variant 'shared_none')...")
    t0 = time.time()
    cov_by_lag_none = gf.noise_cov_by_lag(
        data_train, np.array([], dtype=np.int64), L, off, cov_max_samples)
    print(f"  done ({time.time() - t0:.1f}s)")

    data_path, data_dtype, data_shape = data.filename, data.dtype, data.shape
    del data, data_train
    import gc
    gc.collect()

    unit_args = dict(auto_interferers=args.auto_interferers,
                      template_length=args.template_length,
                      template_offset=args.template_offset,
                      n_channels=args.n_channels, ridge=args.ridge,
                      max_spikes=args.max_spikes, n_thresholds=args.n_thresholds,
                      fs=fs, seed=args.seed, split_t=split_t,
                      cov_max_samples=cov_max_samples)
    init_args = (data_path, data_dtype, data_shape, np_ch, positions, labels,
                 spike_t, spike_cl, unit_args, cov_by_lag_all, cov_by_lag_none)

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    print(f"\n=== exclusion ablation: {len(candidates)} units x {len(VARIANTS)} "
          f"variants, {n_workers} workers ===")
    t0 = time.time()
    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                              initargs=init_args) as pool:
        futs = {pool.submit(_fit_one_unit, uid,
                             int(counts[unit_ids == uid][0])): uid
                for uid in candidates}
        for i, fut in enumerate(as_completed(futs)):
            uid, rows, log = fut.result()
            all_rows.extend(rows)
            last = log.splitlines()[-1] if log.strip() else ""
            print(f"[{i + 1}/{len(candidates)}] unit {uid}: {last}")

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    ok = [r for r in all_rows if r["status"] == "ok"]
    for variant in VARIANTS:
        f1s = [float(r["f1"]) for r in ok if r["variant"] == variant]
        print(f"{variant}: {len(f1s)}/{len(candidates)} ok, "
              f"mean f1 {np.mean(f1s) if f1s else float('nan'):.4f}")
    print(f"{time.time() - t0:.0f}s -> {args.out_dir}/summary.csv")


if __name__ == "__main__":
    main()
