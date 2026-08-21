"""
refit_drift_position_ablation.py
===================================

Part B of the refit-f1 investigation: does refitting at a drifted position
actually recover detection f1, and how much of the achievable recovery does
the CHEAP path (banded_refit.refit_in_band -- new selection + a
motion-corrected template + R assembled from a covariance cached once, no
rescan) give up relative to a genuine from-scratch refit?

Three arms, per unit, on the simulator's own chronological split (train =
first --train-frac of the recording, test = the untouched remainder):

    never_refit     Ordinary fit_lcmv on the WHOLE train half: own channel
                     selection, own template, own R (a full rescan of the
                     whole train half, cov_by_lag=None). Exactly today's
                     production single-fit behaviour -- the baseline. Scored
                     on the untouched test half.

    cheap_refit_B   banded_refit.refit_in_band at the position the unit is
                    predicted to occupy at the train/test boundary ("the
                    drifted position"). The band's covariance
                    (banded_refit.scan_band) is scanned ONCE, over the WHOLE
                    train half, under this unit's own exclusion mask (target
                    + interferers) -- the one-time cost a real online system
                    pays at fit time so every later refit is an
                    assemble-and-solve, no further data access. The target
                    template is motion_correct.registered_template, built
                    from the unit's own per-bin trajectory
                    (drift_estimate.unit_trajectory) and registered to the
                    drifted position using drift_estimate.pooled_com_motion
                    as the motion estimate (the same estimator
                    calibrate_drift_aware.py's 'registered' mode uses).
                    Interferer templates come from window B = the LATE
                    quarter-to-half of the train half, closest in time to
                    the boundary -- a small, cheap waveform average, not a
                    covariance rescan.

    full_refit_B    Identical interferer set, identical registered target
                    template, but R is a genuine rescan of window B's own
                    real data (cov_by_lag=None) -- the actual physically
                    drifted position, since window B sits at the end of the
                    train half. The gold-standard refit at (approximately)
                    the same position cheap_refit_B targets; the gap between
                    the two is exactly the price of skipping the rescan.
                    Channel selection can differ slightly from cheap_refit_B
                    (full-group select_channels vs band-restricted --
                    banded_refit.py's docstring flags this as the one way
                    the two are not bit-identical), but both use the same
                    template and the same interferer waveforms.

Why window B can stand in for "the drifted position" without touching test
data: the simulator drifts coherently and continuously through the whole
recording, so the tail of the train half is *already* physically at (or very
near) wherever the unit will be at the train/test boundary. Rescanning window
B's real data is therefore a genuine full refit at close to the deployment
position, with zero test-half access. cheap_refit_B's band is centered on a
position PREDICTED from window A alone (the unit's own position there, plus
the pooled-motion displacement between window A's time and the boundary) --
so the caching step itself never looks at B either. Only full_refit_B touches
B's raw samples for R; cheap_refit_B touches B only for cheap per-spike
waveform averages (interferer templates), the same kind of access a live
system pays continuously regardless of refit policy.

All three arms are scored identically against the untouched test half via
threshold_sweep_real.find_all_peaks + sweep_thresholds, one best-F1 threshold
each -- the same primitives calibrate_drift_aware.py and
refit_exclusion_ablation.py use. No new scorer.

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

import banded_refit as br
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
                   "drift_span_um", "shift_um", "band_size", "status"]


def _init_worker(tmp_path, dtype, shape, np_ch, chan_y, positions, labels,
                  spike_t, spike_cl, unit_args, t_c_pooled, y_pooled):
    _worker.update(
        data=np.memmap(tmp_path, dtype=dtype, mode="r", shape=shape, order="C"),
        np_ch=np_ch, chan_y=chan_y, positions=positions, labels=labels,
        spike_t=spike_t, spike_cl=spike_cl, args=unit_args,
        t_c_pooled=t_c_pooled, y_pooled=y_pooled)


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
    t_c_pooled, y_pooled = _worker["t_c_pooled"], _worker["y_pooled"]
    n_samples = data.shape[0]
    mid, split_t = a["mid"], a["split_t"]
    L, off = a["template_length"], a["template_offset"]
    fs = a["fs"]

    rows = {v: {k: "" for k in SUMMARY_FIELDS} for v in VARIANTS}
    for v in rows:
        rows[v]["unit_id"], rows[v]["variant"], rows[v]["status"] = target_id, v, "failed"
    buf = io.StringIO()

    try:
        data_train, data_test = data[:split_t], data[split_t:]
        data_A, data_B = data[:mid], data[mid:split_t]
        spike_t_train = spike_t[spike_t < split_t]
        spike_cl_train = spike_cl[spike_t < split_t]
        in_B = (spike_t >= mid) & (spike_t < split_t)
        spike_t_B = spike_t[in_B] - mid
        spike_cl_B = spike_cl[in_B]

        target_train = spike_t_train[spike_cl_train == target_id]
        if target_train.size == 0:
            raise ValueError("no train spikes")
        target_A = target_train[target_train < mid]
        if target_A.size == 0:
            raise ValueError("no spikes in window A")
        target_B_local = spike_t_B[spike_cl_B == target_id]
        if target_B_local.size == 0:
            raise ValueError("no spikes in window B")

        rng_pick = np.random.default_rng(a["seed"] + target_id)

        # ---- interferers: ONE set, fixed across all three arms ----------
        with contextlib.redirect_stdout(buf):
            interferer_ids = gf.auto_pick_interferers_spatial(
                data_train, spike_t_train, spike_cl_train, np_ch, target_id,
                labels, a["auto_interferers"], L, off,
                channel_positions=positions, rng=rng_pick)
        if not interferer_ids:
            raise ValueError("no interferers found near target")

        # ---- never_refit: ordinary single fit on the WHOLE train half ---
        extras_nr = {}
        rng_nr = np.random.default_rng(a["seed"] + target_id + 1000)
        with contextlib.redirect_stdout(buf):
            f_nr, sel_nr, sel_ch_nr, _, _ = tsr.fit_lcmv(
                data_train, spike_t_train, spike_cl_train, np_ch, target_id,
                interferer_ids, a["n_channels"], L, off, a["ridge"],
                a["max_spikes"], rng_nr, extras=extras_nr)
        if len(sel_ch_nr) != a["n_channels"]:
            raise ValueError(f"never_refit: only {len(sel_ch_nr)} channels available")
        f_nr = np.asarray(f_nr, dtype=np.float32)
        lag_nr = gf.detection_lag(f_nr, extras_nr["target_template_sel"], off)
        scored_nr = _score_on_test(data_test, spike_t, spike_cl, target_id,
                                    interferer_ids, sel_nr, f_nr, lag_nr,
                                    split_t, n_samples, a)
        rows["never_refit"].update({
            "n_channels": a["n_channels"],
            "sel_channels": " ".join(str(c) for c in sel_ch_nr),
            "threshold": scored_nr["threshold"], "recall": scored_nr["recall"],
            "precision": scored_nr["precision"], "f1": scored_nr["f1"],
            "fp_rate_hz": scored_nr["fp_rate_hz"],
            "n_test_spikes": scored_nr["n_test_spikes"], "detection_lag": lag_nr,
            "interferer_ids": " ".join(str(c) for c in interferer_ids),
            "drift_span_um": "", "shift_um": "", "band_size": "", "status": "ok",
        })

        # ---- predicted drifted position, from window A + pooled motion --
        # y0_abs: this unit's own absolute depth, estimated from window A
        # alone (early train), so it has an unambiguous reference time.
        wf_A, _ = gf.mean_waveform(data_A, target_A, L, off, a["max_spikes"], rng_pick)
        y0_abs = de._position_from_waveform(wf_A, chan_y, n_top=8)
        if not np.isfinite(y0_abs):
            raise ValueError("could not localise window-A position")
        t0_s = float(np.mean(target_A)) / fs
        ref_time_s = split_t / fs
        # Pooled motion is a zero-mean-per-unit DISPLACEMENT trend, not an
        # absolute depth (drift_estimate.pooled_com_motion docstring); only
        # its DELTA between two times is meaningful, and that delta is a
        # real physical shift in microns regardless of the arbitrary
        # per-unit recentering that built the pooled curve.
        shift_um = float(np.interp(ref_time_s, t_c_pooled, y_pooled) -
                          np.interp(t0_s, t_c_pooled, y_pooled))
        predicted_y = y0_abs + shift_um

        # ---- one-time band covariance cache, over the WHOLE train half --
        local_spikes_train = np.sort(np.concatenate(
            [target_train] + [spike_t_train[spike_cl_train == cid]
                               for cid in interferer_ids]))
        band = br.band_for_depth(chan_y, predicted_y, half_width_um=90.0,
                                  min_channels=max(3 * a["n_channels"], 12))
        cov_band = br.scan_band(data_train, band, local_spikes_train, L, off)

        # ---- motion-corrected target template, registered to the same
        # predicted position (pooled-motion delta, same convention
        # calibrate_drift_aware.py's 'registered' mode uses) ---------------
        with contextlib.redirect_stdout(buf):
            t_c_u, y_raw_u, wfs_u, ns_u = de.unit_trajectory(
                data_train, target_train, chan_y, L, off, fs,
                spikes_per_bin=a["spikes_per_bin"], rng=rng_pick,
                return_waveforms=True)
        if len(wfs_u) < 2:
            raise ValueError("not enough trajectory bins to register")
        drift_span = float(np.ptp(y_raw_u))
        bin_y_pooled = np.interp(t_c_u, t_c_pooled, y_pooled)
        ref_y_template = float(np.interp(ref_time_s, t_c_pooled, y_pooled))
        target_override = mc.registered_template(
            bin_y_pooled, wfs_u, ns_u, ref_y_template, chan_y, a["decay_um"])

        # ---- interferer templates: a cheap, local waveform average from
        # window B (the late train quarter, closest to the boundary), the
        # same recent-data access a live system pays continuously -------
        rng_intB = np.random.default_rng(a["seed"] + target_id + 2000)
        interferer_wfs_B = []
        for cid in interferer_ids:
            st = spike_t_B[spike_cl_B == cid]
            if st.size == 0:
                st = spike_t_train[spike_cl_train == cid] - mid  # fallback: whole train, shifted
                wf, _ = gf.mean_waveform(data_train, spike_t_train[spike_cl_train == cid],
                                          L, off, a["max_spikes"], rng_intB)
            else:
                wf, _ = gf.mean_waveform(data_B, st, L, off, a["max_spikes"], rng_intB)
            interferer_wfs_B.append(wf)

        # ---- cheap_refit_B: assemble-and-solve, no further data access --
        try:
            f_cb, sel_g_cb, sel_ib_cb = br.refit_in_band(
                cov_band, band, target_override, interferer_wfs_B,
                a["n_channels"], L, ridge=a["ridge"])
            f_cb = np.asarray(f_cb, dtype=np.float32)
            tgt_sel_cb = target_override[:, sel_g_cb]
            lag_cb = gf.detection_lag(f_cb, tgt_sel_cb, off)
            scored_cb = _score_on_test(data_test, spike_t, spike_cl, target_id,
                                        interferer_ids, sel_g_cb, f_cb, lag_cb,
                                        split_t, n_samples, a)
            rows["cheap_refit_B"].update({
                "n_channels": a["n_channels"],
                "sel_channels": " ".join(str(c) for c in np_ch[sel_g_cb]),
                "threshold": scored_cb["threshold"], "recall": scored_cb["recall"],
                "precision": scored_cb["precision"], "f1": scored_cb["f1"],
                "fp_rate_hz": scored_cb["fp_rate_hz"],
                "n_test_spikes": scored_cb["n_test_spikes"], "detection_lag": lag_cb,
                "interferer_ids": " ".join(str(c) for c in interferer_ids),
                "drift_span_um": drift_span, "shift_um": shift_um,
                "band_size": band.size, "status": "ok",
            })
        except Exception as e:
            rows["cheap_refit_B"]["status"] = f"failed: {e}"
            print(f"unit {target_id} [cheap_refit_B]: SKIPPED: {e}", file=buf)

        # ---- full_refit_B: genuine rescan of window B's real data -------
        try:
            extras_fb = {}
            rng_fb = np.random.default_rng(a["seed"] + target_id + 2000)
            with contextlib.redirect_stdout(buf):
                f_fb, sel_fb, sel_ch_fb, _, _ = tsr.fit_lcmv(
                    data_B, spike_t_B, spike_cl_B, np_ch, target_id,
                    interferer_ids, a["n_channels"], L, off, a["ridge"],
                    a["max_spikes"], rng_fb, extras=extras_fb,
                    target_waveform_override=target_override, cov_by_lag=None)
            if len(sel_ch_fb) != a["n_channels"]:
                raise ValueError(f"only {len(sel_ch_fb)} channels available")
            f_fb = np.asarray(f_fb, dtype=np.float32)
            lag_fb = gf.detection_lag(f_fb, extras_fb["target_template_sel"], off)
            scored_fb = _score_on_test(data_test, spike_t, spike_cl, target_id,
                                        interferer_ids, sel_fb, f_fb, lag_fb,
                                        split_t, n_samples, a)
            rows["full_refit_B"].update({
                "n_channels": a["n_channels"],
                "sel_channels": " ".join(str(c) for c in sel_ch_fb),
                "threshold": scored_fb["threshold"], "recall": scored_fb["recall"],
                "precision": scored_fb["precision"], "f1": scored_fb["f1"],
                "fp_rate_hz": scored_fb["fp_rate_hz"],
                "n_test_spikes": scored_fb["n_test_spikes"], "detection_lag": lag_fb,
                "interferer_ids": " ".join(str(c) for c in interferer_ids),
                "drift_span_um": drift_span, "shift_um": shift_um,
                "band_size": "", "status": "ok",
            })
        except Exception as e:
            rows["full_refit_B"]["status"] = f"failed: {e}"
            print(f"unit {target_id} [full_refit_B]: SKIPPED: {e}", file=buf)

        print(f"unit {target_id}: never_refit f1={rows['never_refit']['f1']:.3f}  "
              f"cheap_refit_B f1={rows['cheap_refit_B'].get('f1', float('nan')):.3f}  "
              f"full_refit_B f1={rows['full_refit_B'].get('f1', float('nan')):.3f}  "
              f"span={drift_span:.1f}um shift={shift_um:.1f}um", file=buf)

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
                     help="Fraction of the WHOLE recording used as the train "
                          "half; the untouched remainder is test. Window B "
                          "(full_refit_B's real data, and the source of "
                          "cheap_refit_B's interferer templates) is the "
                          "second half of the train half.")
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
    ap.add_argument("--motion-bin-s", type=float, default=20.0)
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
    print(f"Window A: [0, {mid}) ({mid / fs:.1f}s)   "
          f"Window B: [{mid}, {split_t}) ({(split_t - mid) / fs:.1f}s)   "
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

    # Pooled, common-mode motion over the train half -- drift_estimate's own
    # estimator, computed once and shared read-only across every worker.
    # Not re-derived: see drift_estimate.pooled_com_motion's docstring for
    # why this beat the raster-registration alternative it replaced.
    print("Building pooled trajectory (drift_estimate.pooled_com_motion) "
          "from the train half...")
    t_c_pooled, motion_pooled, _win_centers = de.pooled_com_motion(
        data[:split_t], spike_t[spike_t < split_t], spike_cl[spike_t < split_t],
        chan_y, args.template_length, args.template_offset, fs, candidates,
        bin_s=args.motion_bin_s, spikes_per_bin=args.spikes_per_bin,
        n_windows=1)
    y_pooled = motion_pooled[0]
    print(f"pooled trajectory: {t_c_pooled.size} bins, span "
          f"{float(np.ptp(y_pooled)):.1f} um")

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
                 labels, spike_t, spike_cl, unit_args, t_c_pooled, y_pooled)

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    print(f"\n=== drift-position refit ablation: {len(candidates)} units, "
          f"{n_workers} workers ===")
    t0 = time.time()
    all_rows = []
    # Written incrementally, one unit at a time, not just once at the end --
    # a multi-hour run across 160 units must not lose every result to a
    # crash or timeout near the finish. csv_fh stays open for the whole
    # pool block and is flushed after every completed unit.
    csv_path = os.path.join(args.out_dir, "summary.csv")
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    csv_fh.flush()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                              initargs=init_args) as pool:
        futs = {pool.submit(_fit_one_unit, uid,
                             int(counts[unit_ids == uid][0])): uid
                for uid in candidates}
        for i, fut in enumerate(as_completed(futs)):
            uid, rows, log = fut.result()
            all_rows.extend(rows)
            writer.writerows(rows)
            csv_fh.flush()
            os.fsync(csv_fh.fileno())
            last = log.splitlines()[-1] if log.strip() else ""
            print(f"[{i + 1}/{len(candidates)}] unit {uid}: {last}")
    csv_fh.close()

    ok = [r for r in all_rows if r["status"] == "ok"]
    for variant in VARIANTS:
        f1s = [float(r["f1"]) for r in ok if r["variant"] == variant]
        print(f"{variant}: {len(f1s)}/{len(candidates)} ok, "
              f"mean f1 {np.mean(f1s) if f1s else float('nan'):.4f}")
    print(f"{time.time() - t0:.0f}s -> {args.out_dir}/summary.csv")


if __name__ == "__main__":
    main()
