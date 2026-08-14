"""
Fair LCMV-vs-NN comparison: both fit on the same TRAIN portion of the
synthetic recording (by time, no leakage), both evaluated only on the
held-out TEST portion.
"""

import os
import numpy as np
import torch

import generate_filter as gf
import nn_filter as nnf

KS_DIR = "synthetic/kilosort"
TARGET = 0
INTERFERERS = [1, 2]
N_CHANNELS = 5
TEMPLATE_LENGTH = 61
TEMPLATE_OFFSET = 20
TRAIN_FRAC = 0.7
SEED = 0


def load_common():
    rng = np.random.default_rng(SEED)
    spike_t, spike_cl, labels = gf.load_kilosort(KS_DIR)
    params = gf.load_params_py(KS_DIR)
    bin_path = params["dat_path"]
    meta_path = os.path.splitext(bin_path)[0] + ".meta"
    meta = gf.load_sglx_meta(meta_path)

    fs = float(meta["imSampRate"])
    n_saved_chans = int(meta["nSavedChans"])
    np_ch = np.array(gf.parse_chan_subset(meta["snsSaveChanSubset"]))
    n_sync = gf.find_sync_channel_count(meta)

    raw = gf.memmap_raw(bin_path, n_saved_chans)
    last_spike = spike_t.max()
    t_max_samples = int(min(raw.shape[0], last_spike + fs))
    data = np.asarray(raw[:t_max_samples, :], dtype=np.float64)
    if n_sync:
        data = data[:, :-n_sync]
        np_ch = np_ch[:-n_sync]

    return rng, spike_t, spike_cl, data, np_ch, fs


def evaluate(name, D, threshold, target_test, interferer_test_dict, fs, offset=0, window=15):
    above = D > threshold
    peaks = []
    i, n = 0, len(D)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            peaks.append(i + np.argmax(D[i:j]))
            i = j
        else:
            i += 1
    peaks = np.array(peaks) + offset  # shift back to absolute sample index
    target_test_abs = target_test

    hits = sum(1 for t in target_test_abs if np.any(np.abs(peaks - t) <= window))
    misses = len(target_test_abs) - hits

    total_fp = sum(1 for p in peaks if not np.any(np.abs(target_test_abs - p) <= window))
    fp_by_interferer = {}
    for cid, times in interferer_test_dict.items():
        fp_by_interferer[cid] = sum(
            1 for p in peaks
            if np.any(np.abs(times - p) <= window)
            and not np.any(np.abs(target_test_abs - p) <= window)
        )

    print(f"--- {name} ---")
    print(f"  test-split true target spikes: {len(target_test_abs)}  hits: {hits}  misses: {misses}")
    print(f"  total false positives (test split): {total_fp}")
    for cid, fp in fp_by_interferer.items():
        print(f"  false positives coinciding with interferer {cid}: {fp}")
    print()
    return dict(hits=hits, misses=misses, total_fp=total_fp, fp_by_interferer=fp_by_interferer)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    rng, spike_t, spike_cl, data, np_ch, fs = load_common()

    target_spikes = spike_t[spike_cl == TARGET]
    interferer_times_all = [spike_t[spike_cl == cid] for cid in INTERFERERS]

    target_wf, _ = gf.mean_waveform(data, target_spikes, TEMPLATE_LENGTH, TEMPLATE_OFFSET, 2000, rng)
    interferer_wfs = [gf.mean_waveform(data, t, TEMPLATE_LENGTH, TEMPLATE_OFFSET, 2000, rng)[0]
                       for t in interferer_times_all]

    sel = gf.select_channels(target_wf, interferer_wfs, N_CHANNELS)
    sel_channels = np_ch[sel]
    print(f"Selected channels (shared by both models): {sel_channels.tolist()}\n")
    data_sel = data[:, sel]

    split_t = int(TRAIN_FRAC * data_sel.shape[0])
    print(f"Train: [0, {split_t}) ({split_t/fs:.1f}s)   "
          f"Test: [{split_t}, {data_sel.shape[0]}) ({(data_sel.shape[0]-split_t)/fs:.1f}s)\n")

    def split(times):
        return times[times < split_t], times[times >= split_t]

    target_train, target_test = split(target_spikes)
    interferer_train = [split(t)[0] for t in interferer_times_all]
    interferer_test = {cid: split(t)[1] for cid, t in zip(INTERFERERS, interferer_times_all)}

    data_train = data_sel[:split_t]
    data_test = data_sel[split_t:]
    s_train = target_wf[:, sel]  # NOTE: recomputed below from train-only spikes for a fair fit
    print(f"Target: {len(target_train)} train spikes, {len(target_test)} test spikes")
    for cid, t in zip(INTERFERERS, interferer_train):
        print(f"Interferer {cid}: {len(t)} train spikes, {len(interferer_test[cid])} test spikes")
    print()

    # ===================================================================
    # LCMV, fit on TRAIN split only
    # ===================================================================
    print("=" * 60)
    print("Fitting LCMV on train split...")
    target_wf_train, _ = gf.mean_waveform(data_train, target_train, TEMPLATE_LENGTH, TEMPLATE_OFFSET, 2000, rng)
    interferer_wfs_train = [gf.mean_waveform(data_train, t, TEMPLATE_LENGTH, TEMPLATE_OFFSET, 2000, rng)[0]
                             for t in interferer_train]

    s = target_wf_train  # already sliced to sel channels (data_train is pre-sliced)
    interferer_wfs_sel = interferer_wfs_train

    R = gf.noise_covariance(data_train, spike_t[spike_t < split_t], TEMPLATE_LENGTH,
                             TEMPLATE_OFFSET, data_train.shape[0])

    s_flat = s.T.ravel()
    interferer_flats = [wf.T.ravel() for wf in interferer_wfs_sel]
    f_flat = gf.lcmv_filter(s_flat, interferer_flats, R, ridge=1e-3)
    f = f_flat.reshape(len(sel), TEMPLATE_LENGTH).T

    print(f"Constraint check -- gain on target: {np.dot(f_flat, s_flat):.4f} "
          f"(want 1.0)")
    for i, s_int in enumerate(interferer_flats):
        print(f"  gain on interferer {INTERFERERS[i]}: {np.dot(f_flat, s_int):.4e} (want ~0)")

    D_train_lcmv = gf.filter_output(data_train, f)
    threshold_lcmv = gf.compute_threshold(
        D_train_lcmv, spike_t[spike_t < split_t], TEMPLATE_LENGTH, TEMPLATE_OFFSET, k=5.0)
    print(f"LCMV threshold (from train split): {threshold_lcmv:.4f}\n")

    D_test_lcmv = gf.filter_output(data_test, f)

    # ===================================================================
    # NN (linear model, hard-negative mined), fit on TRAIN split only
    # ===================================================================
    print("=" * 60)
    print("Training NN (linear model) on train split...")

    pos_snips, _ = nnf.extract_snippets(data_train, target_train, TEMPLATE_LENGTH, TEMPLATE_OFFSET)
    neg_parts = []
    interferer_snips_train = []
    for cid, t in zip(INTERFERERS, interferer_train):
        snips, _ = nnf.extract_snippets(data_train, t, TEMPLATE_LENGTH, TEMPLATE_OFFSET)
        interferer_snips_train.append(snips)
        neg_parts.append(snips)
    noise_snips = nnf.sample_noise_snippets(
        data_train, spike_t[spike_t < split_t], TEMPLATE_LENGTH, TEMPLATE_OFFSET, n=5000, rng=rng)
    neg_parts.append(noise_snips)

    aug_plain = nnf.augment_positive(pos_snips, rng, 1000, noise_std=float(np.std(data_train)) * 0.3)
    all_interferer_snips = np.concatenate(interferer_snips_train, axis=0)
    aug_overlap = nnf.overlay_augment(pos_snips, all_interferer_snips, rng, 1000)
    pos_all = np.concatenate([pos_snips, aug_plain, aug_overlap], axis=0)
    neg_all = np.concatenate(neg_parts, axis=0)
    print(f"Training set: {pos_all.shape[0]} positive ({pos_snips.shape[0]} real), "
          f"{neg_all.shape[0]} negative\n")

    model = nnf.LinearFilter(len(sel), TEMPLATE_LENGTH).to(device)
    nnf.train_model(model, pos_all, neg_all, device, epochs=300)

    for round_i in range(3):
        hard_negs = nnf.hard_negative_mine(
            model, data_train, INTERFERERS, interferer_train,
            TEMPLATE_LENGTH, TEMPLATE_OFFSET, device)
        if hard_negs.shape[0] == 0:
            print(f"Hard-neg round {round_i}: none found, stopping.\n")
            break
        print(f"Hard-neg round {round_i}: mined {hard_negs.shape[0]} false positives, retraining...")
        neg_all = np.concatenate([neg_all, hard_negs], axis=0)
        nnf.train_model(model, pos_all, neg_all, device, epochs=150, verbose=False)
    print()

    D_train_nn = nnf.score_trace(model, data_train, TEMPLATE_LENGTH, TEMPLATE_OFFSET, device)
    threshold_nn = gf.compute_threshold(
        D_train_nn, spike_t[spike_t < split_t], TEMPLATE_LENGTH, TEMPLATE_OFFSET, k=5.0)
    print(f"NN threshold (from train split): {threshold_nn:.4f}\n")

    D_test_nn = nnf.score_trace(model, data_test, TEMPLATE_LENGTH, TEMPLATE_OFFSET, device)

    # ===================================================================
    # Evaluate both, HELD-OUT test split only
    # ===================================================================
    print("=" * 60)
    print("HELD-OUT TEST SPLIT RESULTS")
    print("=" * 60)
    # target_test / interferer_test are absolute sample indices; peaks found
    # in D_test are relative to the test slice, so evaluate() shifts them
    # by `offset` (=split_t) back to absolute before comparing -- pass
    # target_test/interferer_test as absolute too, not pre-shifted.
    evaluate("LCMV (fit on train, evaluated on held-out test)",
             D_test_lcmv, threshold_lcmv, target_test, interferer_test, fs, offset=split_t)
    evaluate("NN linear model (fit on train, evaluated on held-out test)",
             D_test_nn, threshold_nn, target_test, interferer_test, fs, offset=split_t)


if __name__ == "__main__":
    main()
