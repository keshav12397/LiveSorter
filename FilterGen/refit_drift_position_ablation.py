"""
refit_drift_position_ablation.py
===================================

Part B of the refit-f1 investigation: does the CHEAP refit path
(threshold_sweep_real.fit_lcmv's `cov_by_lag` parameter -- new channel
selection + a registered template + R assembled from a cached per-lag
covariance, no rescan) recover detection f1 at a drifted position, and what
does skipping the rescan cost relative to a full from-scratch refit that DOES
rescan?

Per-unit protocol
------------------
The train half of the recording [0, split_t) is cut in two: position A =
[0, mid) and position B = [mid, split_t). Both are strictly before the test
half [split_t, end), which is untouched by every variant below and is the
only thing ever scored.

    1. fit_A        Ordinary fit_lcmv on window A: own channel selection,
                     own template (this unit's own spikes in A), own R (a
                     full rescan of window A, cov_by_lag=None) -- exactly
                     today's production single-fit behaviour. This filter,
                     UNCHANGED, is variant "never_refit" once scored on test.

                     Alongside the fit, one extra scan builds cov_by_lag_A =
                     generate_filter.noise_cov_by_lag over window A's FULL
                     channel group (not just fit_A's selected channels),
                     excluding this unit's + its interferers' spikes -- the
                     one-time cost a real online system would pay once, at
                     fit time, to make every later refit cheap.

    2. cheap_refit_B    New interferers (auto-picked from window B), new
                        channel selection, a motion-corrected target template
                        (motion_correct.registered_template, built from the
                        unit's WHOLE trajectory across A+B, referenced to
                        window B's own median position) -- but R is
                        cov_by_lag_A assembled onto the new channel selection
                        (fit_lcmv(..., cov_by_lag=cov_by_lag_A)). No pass over
                        window B's data for R. This is the feature under
                        test.

    3. full_refit_B     Identical interferers, channel selection and
                        registered template as cheap_refit_B (same seeded rng
                        before each fit_lcmv call, so `sel` cannot differ for
                        any reason but R's source) -- but R is a genuine
                        rescan of window B's own data (cov_by_lag=None). The
                        gold-standard refit; the gap to cheap_refit_B is
                        exactly the price of skipping the rescan.

All three are scored identically against the untouched test half via
threshold_sweep_real.find_all_peaks + sweep_thresholds, one best-F1 threshold
each. No new scorer.

Session choice matters here in a way it doesn't for Part A: the simulator
(D:/sim_probe_drift, ~30 um total drift) is where B can plausibly be a
different place than A. The real session (D:/catgt_Lav69..., ~6.5 um of
drift) is a negative control -- cheap_refit_B and full_refit_B should track
never_refit closely there, not beat it, since there's barely anything to
refit for.

Example
-------
    python refit_drift_position_ablation.py \\
        --ks-dir D:/sim_probe_drift/sim_ks \\
        --bin-path D:/sim_probe_drift/sim_g0_t0.imec0.ap.bin \\
        --channel-map-json D:/test_newsorter/rawData/shank1only.json \\
        --max-units 10 --workers 4 --out-dir D:/refit_work/position_smoke_sim
"""

import argparse
import contextlib
import csv
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import drift_estimate as de
import generate_filter as gf
import motion_correct as mc
import threshold_sweep_real as tsr

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_worker = {}

VARIANTS = ["never_refit", "cheap_refit_B", "full_refit_B"]

SUMMARY_FIELDS = ["unit_id", "variant", "n_channels", "sel_channels",
                   "threshold", "recall", "precision", "f1", "fp_rate_hz",
                   "n_test_spikes", "detection_lag", "interferer_ids",
                   "drift_span_um", "status"]


def _init_worker(tmp_path, dtype, shape, np_ch, chan_y, positions, labels,
                  spike_t, spike_cl, unit_args):
    _worker.update(
        data=np.memmap(tmp_path, dtype=dtype, mode="r", shape=shape, order="C"),
        np_ch=np_ch, chan_y=chan_y, positions=positions, labels=labels,
        spike_t=spike_t, spike_cl=spike_cl, args=unit_args)


def _score_on_test(data_test, spike_t, spike_cl, target_id, interferer_ids,
                    sel, f, lag, split_t, n_samples, a):
    """Exactly calibrate_drift_aware._evaluate_on_test -- not imported
    because that module's is private, but a direct call to the same
    threshold_sweep_real primitives it wraps (find_all_peaks,
    sweep_thresholds), no new scoring math."""
    data_test_sel = data_test[:, sel]
    D_test = gf.filter_output(data_test_sel, f)
    peak_idx, peak_scores = tsr.find_all_peaks(D_test, a["template_length"] // 2)
    peak_idx = peak_idx - lag

    target_test = spike_t[(spike_cl == target_id) & (spike_t >= split_t)] - split_t
    if target_test.size == 0:
        raise ValueError("no held-out test spikes")
    interferer_test = {cid: spike_t[(spike_cl == cid) & (spike_t >= split_t)] - split_t
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
    return dict(threshold=float(thresholds[best]), recall=float(recall[best]),
                precision=float(precision[best]), f1=float(f1[best]),
                fp_rate_hz=float(total_fp[best] / test_duration_s),
                n_test_spikes=int(target_test.size))


def _fit_one_unit(target_id, spike_count):
    data = _worker["data"]
    np_ch, chan_y = _worker["np_ch"], _worker["chan_y"]
    positions, labels = _worker["positions"], _worker["labels"]
    spike_t, spike_cl = _worker["spike_t"], _worker["spike_cl"]
    a = _worker["args"]
    n_samples = data.shape[0]
    mid, split_t = a["mid"], a["split_t"]
    L, off = a["template_length"], a["template_offset"]

    rows = {v: {k: "" for k in SUMMARY_FIELDS} for v in VARIANTS}
    for v in rows:
        rows[v]["unit_id"], rows[v]["variant"], rows[v]["status"] = target_id, v, "failed"
    buf = io.StringIO()

    try:
        data_A, data_B, data_test = data[:mid], data[mid:split_t], data[split_t:]
        spike_t_A = spike_t[spike_t < mid]
        spike_cl_A = spike_cl[spike_t < mid]
        in_B = (spike_t >= mid) & (spike_t < split_t)
        spike_t_B = spike_t[in_B] - mid
        spike_cl_B = spike_cl[in_B]

        target_A = spike_t_A[spike_cl_A == target_id]
        if target_A.size == 0:
            raise ValueError("no spikes in window A")
        target_B_local = spike_t_B[spike_cl_B == target_id]
        if target_B_local.size == 0:
            raise ValueError("no spikes in window B")

        rng_pick = np.random.default_rng(a["seed"] + target_id)

        # ---- fit_A: ordinary single-window fit, own rescan --------------
        with contextlib.redirect_stdout(buf):
            interferers_A = gf.auto_pick_interferers_spatial(
                data_A, spike_t_A, spike_cl_A, np_ch, target_id, labels,
                a["auto_interferers"], L, off, channel_positions=positions,
                rng=rng_pick)
        if not interferers_A:
            raise ValueError("no interferers found near A")

        extras_A = {}
        rng_A = np.random.default_rng(a["seed"] + target_id + 1000)
        with contextlib.redirect_stdout(buf):
            f_A, sel_A, sel_ch_A, _, _ = tsr.fit_lcmv(
                data_A, spike_t_A, spike_cl_A, np_ch, target_id, interferers_A,
                a["n_channels"], L, off, a["ridge"], a["max_spikes"], rng_A,
                extras=extras_A)
        if len(sel_ch_A) != a["n_channels"]:
            raise ValueError(f"fit_A: only {len(sel_ch_A)} channels available")
        f_A = np.asarray(f_A, dtype=np.float32)
        lag_A = gf.detection_lag(f_A, extras_A["target_template_sel"], off)

        # One-time cache: full-group cov_by_lag over window A, excluding
        # this unit's own + its interferers' spikes in A -- what a real
        # online system would build once, at fit time, to make later
        # refits at drifted positions cheap.
        local_spikes_A = np.sort(np.concatenate(
            [target_A] + [spike_t_A[spike_cl_A == cid] for cid in interferers_A]))
        cov_by_lag_A = gf.noise_cov_by_lag(data_A, local_spikes_A,
                                            L, off, data_A.shape[0])

        scored = _score_on_test(data_test, spike_t, spike_cl, target_id,
                                 interferers_A, sel_A, f_A, lag_A, split_t,
                                 n_samples, a)
        rows["never_refit"].update({
            "n_channels": a["n_channels"],
            "sel_channels": " ".join(str(c) for c in sel_ch_A),
            "threshold": scored["threshold"], "recall": scored["recall"],
            "precision": scored["precision"], "f1": scored["f1"],
            "fp_rate_hz": scored["fp_rate_hz"],
            "n_test_spikes": scored["n_test_spikes"], "detection_lag": lag_A,
            "interferer_ids": " ".join(str(c) for c in interferers_A),
            "drift_span_um": "", "status": "ok",
        })

        # ---- trajectory across A+B, for registration at B ----------------
        target_train = spike_t[(spike_t < split_t) & (spike_cl == target_id)]
        with contextlib.redirect_stdout(buf):
            t_c, y_raw, wfs, ns = de.unit_trajectory(
                data[:split_t], target_train, chan_y, L, off, a["fs"],
                spikes_per_bin=a["spikes_per_bin"], rng=rng_pick,
                return_waveforms=True)
        if len(wfs) < 2:
            raise ValueError("not enough trajectory bins to register")
        drift_span = float(np.ptp(y_raw))

        in_bin_B = (t_c >= mid / a["fs"]) & (t_c < split_t / a["fs"])
        if np.any(in_bin_B):
            ref_y_B = float(np.median(y_raw[in_bin_B]))
        else:
            mid_t = 0.5 * (mid + split_t) / a["fs"]
            ref_y_B = float(y_raw[int(np.argmin(np.abs(t_c - mid_t)))])
        target_override_B = mc.registered_template(
            y_raw, wfs, ns, ref_y_B, chan_y, a["decay_um"])

        # ---- interferers + channel selection + template: shared by BOTH
        # refit variants (same seeded rng before each fit_lcmv call), so the
        # only thing that can differ between cheap_refit_B and full_refit_B
        # is which R the LCMV solve sees.
        with contextlib.redirect_stdout(buf):
            interferers_B = gf.auto_pick_interferers_spatial(
                data_B, spike_t_B, spike_cl_B, np_ch, target_id, labels,
                a["auto_interferers"], L, off, channel_positions=positions,
                rng=rng_pick)
        if not interferers_B:
            raise ValueError("no interferers found near B")

        for variant, cov_by_lag in (("cheap_refit_B", cov_by_lag_A),
                                     ("full_refit_B", None)):
            try:
                extras_B = {}
                rng_B = np.random.default_rng(a["seed"] + target_id + 2000)
                with contextlib.redirect_stdout(buf):
                    f_B, sel_B, sel_ch_B, _, _ = tsr.fit_lcmv(
                        data_B, spike_t_B, spike_cl_B, np_ch, target_id,
                        interferers_B, a["n_channels"], L, off, a["ridge"],
                        a["max_spikes"], rng_B, extras=extras_B,
                        target_waveform_override=target_override_B,
                        cov_by_lag=cov_by_lag)
                if len(sel_ch_B) != a["n_channels"]:
                    raise ValueError(f"only {len(sel_ch_B)} channels available")
                f_B = np.asarray(f_B, dtype=np.float32)
                lag_B = gf.detection_lag(f_B, extras_B["target_template_sel"], off)

                scored_B = _score_on_test(data_test, spike_t, spike_cl, target_id,
                                           interferers_B, sel_B, f_B, lag_B,
                                           split_t, n_samples, a)
                rows[variant].update({
                    "n_channels": a["n_channels"],
                    "sel_channels": " ".join(str(c) for c in sel_ch_B),
                    "threshold": scored_B["threshold"], "recall": scored_B["recall"],
                    "precision": scored_B["precision"], "f1": scored_B["f1"],
                    "fp_rate_hz": scored_B["fp_rate_hz"],
                    "n_test_spikes": scored_B["n_test_spikes"],
                    "detection_lag": lag_B,
                    "interferer_ids": " ".join(str(c) for c in interferers_B),
                    "drift_span_um": drift_span, "status": "ok",
                })
            except Exception as e:
                rows[variant]["status"] = f"failed: {e}"
                print(f"unit {target_id} [{variant}]: SKIPPED: {e}", file=buf)

        print(f"unit {target_id}: never_refit f1={rows['never_refit']['f1']:.3f}  "
              f"cheap_refit_B f1={rows['cheap_refit_B'].get('f1', float('nan')):.3f}  "
              f"full_refit_B f1={rows['full_refit_B'].get('f1', float('nan')):.3f}  "
              f"span={drift_span:.1f}um", file=buf)

    except Exception as e:
        for v in rows:
            if rows[v]["status"] != "ok":
                rows[v]["status"] = f"failed: {e}"
        print(f"unit {target_id}: SKIPPED: {e}", file=buf)

    return target_id, list(rows.values()), buf.getvalue()


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
    ap.add_argument("--train-frac", type=float, default=0.5,
                     help="Fraction of the WHOLE recording used for A+B "
                          "(split in half again for A/B); the untouched "
                          "remainder is test.")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-spikes", type=int, default=2000)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--n-thresholds", type=int, default=60)
    ap.add_argument("--auto-interferers", type=int, default=5)
    ap.add_argument("--min-spikes", type=int, default=200)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--spikes-per-bin", type=int, default=150)
    ap.add_argument("--decay-um", type=float, default=30.0)
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
    chan_y = de.channel_y_for_group(np_ch, positions)

    split_t = int(round(args.train_frac * data.shape[0]))
    mid = split_t // 2
    print(f"Position A: [0, {mid}) ({mid / fs:.1f}s)   "
          f"Position B: [{mid}, {split_t}) ({(split_t - mid) / fs:.1f}s)   "
          f"Test: [{split_t}, {data.shape[0]}) ({(data.shape[0] - split_t) / fs:.1f}s)")

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

    data_path, data_dtype, data_shape = data.filename, data.dtype, data.shape
    del data
    import gc
    gc.collect()

    unit_args = dict(auto_interferers=args.auto_interferers,
                      template_length=args.template_length,
                      template_offset=args.template_offset,
                      n_channels=args.n_channels, ridge=args.ridge,
                      max_spikes=args.max_spikes, n_thresholds=args.n_thresholds,
                      fs=fs, seed=args.seed, mid=mid, split_t=split_t,
                      spikes_per_bin=args.spikes_per_bin, decay_um=args.decay_um)
    init_args = (data_path, data_dtype, data_shape, np_ch, chan_y, positions,
                 labels, spike_t, spike_cl, unit_args)

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    print(f"\n=== drift-position refit ablation: {len(candidates)} units, "
          f"{n_workers} workers ===")
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
