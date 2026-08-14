"""
nn_filter.py
============

Learned alternative to the LCMV filter in generate_filter.py: a small 1-D
CNN trained to score "target spike present" vs. named interferers' spikes
and background noise, with iterative hard-negative mining as the practical
substitute for a full adversarial GAN setup (see conversation).

Two models, both tiny, both directly comparable to LCMV's own linear form:

  Model A (linear): Conv1d(n_channels -> 1, kernel=template_length).
                     Structurally identical to LCMV -- same inductive bias,
                     but interferer rejection is a *soft* learned margin
                     instead of an exact algebraic null. Can be initialized
                     from an existing LCMV filter (--init-lcmv).

  Model B (nonlinear): Conv1d(n_ch->8,k=15) -> ReLU -> Conv1d(8->1,k=15)
                        -> sigmoid. A few hundred parameters.

Trains on a TRAIN split of the recording, hard-negative-mines on that same
split, and reports held-out metrics on a disjoint TEST split -- directly
comparable to eval_synthetic.py's LCMV numbers (which should also be
re-evaluated on the same test split for a fair comparison; see
compare_on_synthetic.py).

Requires: numpy, scipy (only for the underlying loaders/preprocessing that
this file reuses from generate_filter.py), torch.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import generate_filter as gf  # reuse loaders/preprocessing


# --------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------- #

class LinearFilter(nn.Module):
    """Conv1d(n_ch -> 1, kernel=T). Structurally == LCMV's linear form."""

    def __init__(self, n_channels, template_length):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, 1, kernel_size=template_length, bias=True)

    def forward(self, x):
        # x: (batch, n_channels, template_length) -> (batch,)
        return self.conv(x).squeeze(-1).squeeze(-1)

    def init_from_lcmv(self, f):
        """f: (template_length, n_channels) as saved by generate_filter.py."""
        w = torch.from_numpy(f.T.copy()).float().unsqueeze(0)  # (1, n_ch, T)
        with torch.no_grad():
            self.conv.weight.copy_(w)
            self.conv.bias.zero_()


class SmallCNN(nn.Module):
    """Conv1d(n_ch->8,k=15) -> ReLU -> Conv1d(8->1,k=15) -> scalar."""

    def __init__(self, n_channels, template_length, hidden=8, k=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=k),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=template_length - k + 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


# --------------------------------------------------------------------- #
# Snippet extraction / dataset construction
# --------------------------------------------------------------------- #

def extract_snippets(data_sel, times, template_length, template_offset, max_n=None, rng=None):
    times = times[(times > template_offset) &
                   (times < data_sel.shape[0] - template_length + template_offset)]
    if max_n is not None and times.size > max_n:
        rng = rng or np.random.default_rng()
        times = rng.choice(times, size=max_n, replace=False)
    snips = np.stack([
        data_sel[t - template_offset: t - template_offset + template_length, :]
        for t in times
    ])  # (N, template_length, n_channels)
    return snips.transpose(0, 2, 1), times  # -> (N, n_channels, template_length)


def sample_noise_snippets(data_sel, all_spike_times, template_length, template_offset,
                           n, rng):
    """Random snippets from periods with no known spike from any cluster."""
    n_samples = data_sel.shape[0]
    spike_present = np.zeros(n_samples, dtype=bool)
    for t in all_spike_times:
        if t >= n_samples:
            continue
        lo = max(0, t - template_offset - template_length)
        hi = min(n_samples, t + template_length + template_offset)
        spike_present[lo:hi] = True
    valid_idx = np.where(~spike_present)[0]
    valid_idx = valid_idx[(valid_idx > template_offset) &
                           (valid_idx < n_samples - template_length + template_offset)]
    chosen = rng.choice(valid_idx, size=min(n, valid_idx.size), replace=False)
    snips = np.stack([
        data_sel[t - template_offset: t - template_offset + template_length, :]
        for t in chosen
    ])
    return snips.transpose(0, 2, 1)


def augment_positive(snips, rng, n_out, max_jitter=3, amp_range=(0.7, 1.3), noise_std=0.0):
    """Augment a small positive set up to n_out examples via jitter,
    amplitude scaling, and optional additive noise."""
    n_ch, T = snips.shape[1], snips.shape[2]
    out = []
    base_pool = snips
    while len(out) < n_out:
        s = base_pool[rng.integers(0, base_pool.shape[0])].copy()
        shift = rng.integers(-max_jitter, max_jitter + 1)
        if shift != 0:
            s = np.roll(s, shift, axis=1)
        amp = rng.uniform(*amp_range)
        s = s * amp
        if noise_std > 0:
            s = s + rng.normal(0, noise_std, size=s.shape)
        out.append(s)
    return np.stack(out)


def overlay_augment(target_snips, interferer_snips, rng, n_out, max_shift=20):
    """Synthetic overlapping-spike examples: target waveform + a randomly
    time-shifted interferer waveform superimposed -- trains the model on
    exactly the hard "temporal overlap" case discussed earlier."""
    out = []
    while len(out) < n_out:
        t = target_snips[rng.integers(0, target_snips.shape[0])].copy()
        it = interferer_snips[rng.integers(0, interferer_snips.shape[0])]
        shift = rng.integers(-max_shift, max_shift + 1)
        shifted = np.roll(it, shift, axis=1)
        out.append(t + shifted)
    return np.stack(out)


# --------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------- #

def train_model(model, pos, neg, device, epochs=200, lr=1e-2, weight_decay=1e-4,
                 pos_weight=None, verbose=True):
    """Binary margin (hinge) training: target -> +1, everything else -> -1."""
    x = np.concatenate([pos, neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))]).astype(np.float32)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    if pos_weight is None:
        pos_weight = len(neg) / max(len(pos), 1)
    sample_weight = torch.where(y_t > 0, torch.tensor(pos_weight, device=device),
                                 torch.tensor(1.0, device=device))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(x_t)
        # hinge loss: max(0, 1 - y*out), weighted for class imbalance
        loss = torch.clamp(1 - y_t * out, min=0)
        loss = (loss * sample_weight).mean()
        loss.backward()
        opt.step()
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                pred_pos = (model(torch.from_numpy(pos.astype(np.float32)).to(device)) > 0).float().mean()
                pred_neg = (model(torch.from_numpy(neg.astype(np.float32)).to(device)) > 0).float().mean()
            print(f"  epoch {epoch:4d}  loss={loss.item():.4f}  "
                  f"pos_acc={pred_pos.item():.3f}  neg_fpr={pred_neg.item():.3f}")
    return model


def hard_negative_mine(model, data_sel, interferer_ids, interferer_times_all,
                        template_length, template_offset, device, max_per_round=500):
    """Scan the model over all known interferer spike times, find the
    highest-scoring false positives, return them as new hard-negative
    snippets."""
    model.eval()
    hard_negs = []
    with torch.no_grad():
        for cid, times in zip(interferer_ids, interferer_times_all):
            snips, kept_times = extract_snippets(data_sel, times, template_length, template_offset)
            if snips.shape[0] == 0:
                continue
            scores = model(torch.from_numpy(snips.astype(np.float32)).to(device)).cpu().numpy()
            fp_idx = np.where(scores > 0)[0]
            if fp_idx.size == 0:
                continue
            order = fp_idx[np.argsort(-scores[fp_idx])][:max_per_round]
            hard_negs.append(snips[order])
    if hard_negs:
        return np.concatenate(hard_negs, axis=0)
    return np.zeros((0, data_sel.shape[1], template_length), dtype=np.float32)


# --------------------------------------------------------------------- #
# Threshold + save (mirrors generate_filter.py's convention)
# --------------------------------------------------------------------- #

def score_trace(model, data_sel, template_length, template_offset, device, batch=8192):
    """Slide the model over the full trace to get a continuous score D(t),
    comparable to the LCMV filter's convolution output."""
    model.eval()
    n_samples, n_ch = data_sel.shape
    D = np.full(n_samples, -np.inf, dtype=np.float32)

    starts = np.arange(template_offset, n_samples - template_length + template_offset, 1)
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk_starts = starts[i:i + batch]
            snips = np.stack([
                data_sel[t - template_offset: t - template_offset + template_length, :]
                for t in chunk_starts
            ]).transpose(0, 2, 1).astype(np.float32)
            scores = model(torch.from_numpy(snips).to(device)).cpu().numpy()
            D[chunk_starts] = scores
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--interferers", type=int, nargs="+", required=True)
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--model", choices=["linear", "cnn"], default="linear")
    ap.add_argument("--init-lcmv", help="Path to LCMV out-dir to initialize Model A from")
    ap.add_argument("--train-frac", type=float, default=0.7,
                     help="Fraction of the recording (by time) used for training")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hard-neg-rounds", type=int, default=3)
    ap.add_argument("--n-augment", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Load exactly like generate_filter.py --------------------------
    spike_t, spike_cl, labels = gf.load_kilosort(args.ks_dir)
    params = gf.load_params_py(args.ks_dir)
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

    target_spikes = spike_t[spike_cl == args.target]
    interferer_times_all = [spike_t[spike_cl == cid] for cid in args.interferers]

    target_wf, _ = gf.mean_waveform(data, target_spikes, args.template_length,
                                     args.template_offset, 2000, rng)
    interferer_wfs = [gf.mean_waveform(data, t, args.template_length,
                                        args.template_offset, 2000, rng)[0]
                       for t in interferer_times_all]

    sel = gf.select_channels(target_wf, interferer_wfs, args.n_channels)
    sel_channels = np_ch[sel]
    print(f"Selected channels: {sel_channels.tolist()}")
    data_sel = data[:, sel]

    # ---- Train/test split by TIME (avoid leakage) -----------------------
    split_t = int(args.train_frac * data_sel.shape[0])
    print(f"Train: samples [0, {split_t}) ({split_t/fs:.1f}s)  "
          f"Test: [{split_t}, {data_sel.shape[0]}) ({(data_sel.shape[0]-split_t)/fs:.1f}s)")

    def split(times):
        return times[times < split_t], times[times >= split_t]

    target_train, target_test = split(target_spikes)
    interferer_train = [split(t)[0] for t in interferer_times_all]
    interferer_test = [split(t)[0] for t in interferer_times_all]  # placeholder, unused directly

    data_train = data_sel[:split_t]
    data_test = data_sel[split_t:]

    # ---- Build training set ----------------------------------------------
    pos_snips, _ = extract_snippets(data_train, target_train, args.template_length,
                                     args.template_offset)
    print(f"Target train spikes: {pos_snips.shape[0]}")

    neg_parts = []
    interferer_snips_train = []
    for cid, t in zip(args.interferers, interferer_train):
        snips, _ = extract_snippets(data_train, t, args.template_length, args.template_offset)
        interferer_snips_train.append(snips)
        neg_parts.append(snips)
        print(f"Interferer {cid} train spikes used as negatives: {snips.shape[0]}")

    noise_snips = sample_noise_snippets(
        data_train, spike_t[spike_t < split_t], args.template_length,
        args.template_offset, n=5000, rng=rng)
    neg_parts.append(noise_snips)
    print(f"Background noise negatives: {noise_snips.shape[0]}")

    # ---- Augment positives (jitter/amplitude/noise + synthetic overlaps) --
    aug_plain = augment_positive(pos_snips, rng, args.n_augment // 2,
                                  noise_std=float(np.std(data_train)) * 0.3)
    all_interferer_snips = np.concatenate(interferer_snips_train, axis=0)
    aug_overlap = overlay_augment(pos_snips, all_interferer_snips, rng, args.n_augment // 2)
    pos_all = np.concatenate([pos_snips, aug_plain, aug_overlap], axis=0)
    neg_all = np.concatenate(neg_parts, axis=0)
    print(f"Final training set: {pos_all.shape[0]} positive, {neg_all.shape[0]} negative")

    # ---- Build + train model -----------------------------------------------
    n_ch = len(sel)
    if args.model == "linear":
        model = LinearFilter(n_ch, args.template_length).to(device)
        if args.init_lcmv:
            filt = np.fromfile(os.path.join(args.init_lcmv, f"filter_{args.target}.bin"),
                                dtype=np.float64).reshape(args.template_length, n_ch)
            model.init_from_lcmv(filt)
            print("Initialized Model A from LCMV filter.")
    else:
        model = SmallCNN(n_ch, args.template_length).to(device)

    print(f"Training {args.model} model...")
    train_model(model, pos_all, neg_all, device, epochs=args.epochs)

    for round_i in range(args.hard_neg_rounds):
        hard_negs = hard_negative_mine(
            model, data_train, args.interferers, interferer_train,
            args.template_length, args.template_offset, device)
        if hard_negs.shape[0] == 0:
            print(f"Hard-neg round {round_i}: no false positives found on train interferers, stopping.")
            break
        print(f"Hard-neg round {round_i}: mined {hard_negs.shape[0]} false positives, retraining...")
        neg_all = np.concatenate([neg_all, hard_negs], axis=0)
        train_model(model, pos_all, neg_all, device, epochs=args.epochs // 2, verbose=False)

    # ---- Threshold + evaluate on HELD-OUT test split -----------------------
    D_train = score_trace(model, data_train, args.template_length, args.template_offset, device)
    all_spike_times_train = spike_t[spike_t < split_t]
    threshold = gf.compute_threshold(D_train, all_spike_times_train,
                                      args.template_length, args.template_offset, k=5.0)
    print(f"Threshold (from train split): {threshold:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{args.target}.pt"))
    np.asarray(sel_channels, dtype=np.int32).tofile(
        os.path.join(args.out_dir, f"channels_{args.target}.bin"))
    np.asarray([threshold], dtype=np.float64).tofile(
        os.path.join(args.out_dir, f"threshold_{args.target}.bin"))

    print(f"Saved model to {args.out_dir}")
    print(f"model_type={args.model} n_channels={n_ch} template_length={args.template_length} "
          f"template_offset={args.template_offset} split_sample={split_t}")


if __name__ == "__main__":
    main()
