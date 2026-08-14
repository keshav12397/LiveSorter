"""
generate_filter.py
===================

Python replacement for matlab_BOTM/generate_filters*.m, specialized for the
single-target case: a sparsely-firing, low-amplitude, poorly-isolated unit
sitting among a small number of much-more-active neighbor units.

Instead of plain BOTM (matched filter whitened only against generic
spike-free background noise), this builds a Linearly-Constrained Minimum
Variance (LCMV) filter:

    f = R^-1 C (C^T R^-1 C)^-1 g

where C = [s | s_1 | s_2 | ... | s_m] stacks the target template `s` and the
templates of the m dominant interferer units (flattened over channels x
taps), and g = [1, 0, ..., 0]^T. This filter is constrained to have unity
gain on the target waveform and *exactly zero* gain on each listed
interferer's waveform, while minimizing residual output variance under the
local noise covariance R. Unlike plain BOTM, it explicitly rejects known
strong interferers rather than relying on generic background statistics to
happen to suppress them.

Output format is unchanged from the MATLAB scripts, so the same C++/GPU
filter loader can be used without modification:

    channels_<id>.bin    int32,   one entry per selected channel (SpikeGLX
                                   channel indices, i.e. np_ch values)
    filter_<id>.bin      float64, shape (template_length, n_channels),
                                   row-major (C order)
    threshold_<id>.bin   float64, single scalar

Requires: numpy, scipy, matplotlib (matplotlib only for --plot).

Example
-------
    python generate_filter.py \\
        --ks-dir /path/to/kilosort/output \\
        --target 42 \\
        --interferers 7 19 \\
        --n-channels 5 \\
        --out-dir /path/to/output \\
        --plot

    # or let the script auto-pick interferers by firing rate:
    python generate_filter.py \\
        --ks-dir /path/to/kilosort/output \\
        --target 42 \\
        --auto-interferers 3 \\
        --n-channels 5 \\
        --out-dir /path/to/output
"""

import argparse
import os
import re
import sys

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from scipy.linalg import toeplitz


# --------------------------------------------------------------------- #
# Kilosort / SpikeGLX loading
# --------------------------------------------------------------------- #

def load_kilosort(ks_dir):
    """Load spike_times.npy / spike_clusters.npy, plus optional labels."""
    spike_t = np.load(os.path.join(ks_dir, "spike_times.npy")).ravel().astype(np.int64)
    spike_cl = np.load(os.path.join(ks_dir, "spike_clusters.npy")).ravel().astype(np.int64)

    keep = spike_t >= 0
    spike_t, spike_cl = spike_t[keep], spike_cl[keep]

    labels = {}
    for fname in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        fpath = os.path.join(ks_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath) as fh:
                next(fh)  # header
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        labels[int(parts[0])] = parts[1]
            break

    return spike_t, spike_cl, labels


def load_params_py(ks_dir):
    """Minimal parser for Kilosort's params.py (dat_path, n_channels_dat, etc.)."""
    fpath = os.path.join(ks_dir, "params.py")
    params = {}
    if os.path.isfile(fpath):
        with open(fpath) as fh:
            src = fh.read()
        ns = {}
        exec(compile(src, fpath, "exec"), {}, ns)
        params.update(ns)
    return params


def load_sglx_meta(meta_path):
    """Parse a SpikeGLX .meta file into a dict of strings."""
    meta = {}
    with open(meta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.lstrip("~")
            meta[key] = val
    return meta


def parse_chan_subset(spec):
    """Parse SpikeGLX snsSaveChanSubset / snsChanMap-style '0:5,7,10:12' spec
    into a flat list of ints. Falls back to comma-separated ints."""
    chans = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            a, b = part.split(":")
            chans.extend(range(int(a), int(b) + 1))
        elif part:
            chans.append(int(part))
    return chans


def load_channel_map_json(path):
    """Load a Kilosort-style probe/channel-map JSON (e.g. a per-shank
    subset like 'shank1only.json') and return its 'chanMap' array -- raw
    SpikeGLX channel indices to restrict processing to."""
    import json
    with open(path) as fh:
        j = json.load(fh)
    return np.array(j["chanMap"], dtype=int)


def find_sync_channel_count(meta):
    # snsApLfSy = "<ap>,<lf>,<sy>"
    parts = meta.get("snsApLfSy", "0,0,0").split(",")
    return int(parts[-1])


def memmap_raw(bin_path, n_chan, dtype=np.int16):
    """Memory-map a SpikeGLX/Kilosort binary as (nSamples, nChan).

    SpikeGLX .bin files are channel-interleaved per timepoint, i.e. the
    on-disk order is [ch0_t0, ch1_t0, ..., chN_t0, ch0_t1, ...] -- this is
    exactly row-major (C order) with shape (nSamples, nChan), no transpose
    needed (unlike MATLAB's column-major memmapfile).
    """
    n_bytes = os.path.getsize(bin_path)
    itemsize = np.dtype(dtype).itemsize
    n_samples = n_bytes // (itemsize * n_chan)
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=(n_samples, n_chan))


# --------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------- #

def common_average_reference(data):
    """In-place-safe CAR: subtract the per-sample median across channels."""
    ref = np.median(data, axis=1, keepdims=True)
    return data - ref


def highpass(data, fc, fs, order=2):
    """Zero-phase high-pass, matching butter(2, ...)+filtfilt in MATLAB."""
    b, a = butter(order, 2 * fc / fs, btype="high")
    return filtfilt(b, a, data, axis=0)


def highpass_causal_biquad(data, fc, fs):
    """Causal high-pass, coefficient-for-coefficient identical to
    ClosedLoop/ButterworthHighpass.h (RBJ Audio-EQ-Cookbook 2nd-order
    Butterworth biquad, Q=1/sqrt(2)).

    Use this (via --causal-highpass) instead of `highpass()`'s zero-phase
    filtfilt when the filter/threshold will be deployed against the C++
    live pipeline, which can only filter causally. `filtfilt` and this
    causal biquad have the same magnitude response shape but different
    phase -- calibrating against filtfilt-preprocessed data while running
    causal-biquad-preprocessed data live silently degrades the matched
    filter's correlation with real spike waveforms (found the hard way:
    C++ live recall capped around 45-63% vs. this function's ~97% when the
    mismatch was still present). Training the filter on the exact same
    causal filter closes that gap by construction, instead of trying to
    make the live filter impossibly match filtfilt's zero-phase response.
    """
    Q = 1.0 / np.sqrt(2.0)
    w0 = 2.0 * np.pi * fc / fs
    cosw0 = np.cos(w0)
    sinw0 = np.sin(w0)
    alpha = sinw0 / (2.0 * Q)
    a0 = 1.0 + alpha
    b0 = ((1.0 + cosw0) / 2.0) / a0
    b1 = (-(1.0 + cosw0)) / a0
    b2 = ((1.0 + cosw0) / 2.0) / a0
    a1 = (-2.0 * cosw0) / a0
    a2 = (1.0 - alpha) / a0
    b = np.array([b0, b1, b2])
    a = np.array([1.0, a1, a2])
    # lfilter (not filtfilt) -- causal, zero initial state, exactly matching
    # ButterworthHighpass::processSample's per-sample recursion starting
    # from x1_=x2_=y1_=y2_=0.
    return lfilter(b, a, data, axis=0)


# --------------------------------------------------------------------- #
# Template / channel selection
# --------------------------------------------------------------------- #

def mean_waveform(data, spike_times, template_length, template_offset,
                   max_spikes=2000, rng=None):
    """Mean waveform (template_length, n_channels) for one cluster's spikes.

    Subsamples to `max_spikes` when there are more (cheap approximation --
    a few hundred spikes is already enough to converge for a stable
    template; this matters mainly for high-firing interferer units).
    """
    rng = rng or np.random.default_rng()

    valid = spike_times[
        (spike_times > template_offset) &
        (spike_times < data.shape[0] - template_length + template_offset)
    ]
    if valid.size == 0:
        raise ValueError("No valid spikes (all too close to recording edges).")

    if valid.size > max_spikes:
        valid = rng.choice(valid, size=max_spikes, replace=False)

    acc = np.zeros((template_length, data.shape[1]), dtype=np.float64)
    for t in valid:
        acc += data[t - template_offset: t - template_offset + template_length, :]
    return acc / valid.size, valid.size


def select_channels(target_wf, interferer_wfs, n_channels, eps=1e-9):
    """Pick channels maximizing target energy relative to interferer energy,
    not just raw target amplitude -- a channel where a big neighbor also
    dominates is a bad channel even if the target is large there.
    """
    target_energy = np.sum(target_wf ** 2, axis=0)

    if interferer_wfs:
        interferer_energy = np.max(
            [np.sum(wf ** 2, axis=0) for wf in interferer_wfs], axis=0
        )
    else:
        interferer_energy = np.zeros_like(target_energy)

    score = target_energy / (interferer_energy + eps)

    # Require some minimum absolute target energy so we don't pick a
    # channel that "wins" the ratio only because both are near zero.
    floor = 0.05 * target_energy.max()
    score = np.where(target_energy >= floor, score, -np.inf)

    order = np.argsort(score)[::-1]
    return order[:n_channels]


def auto_pick_interferers(spike_t, spike_cl, target_id, labels, n_interferers):
    """Pick the n most active *other* clusters as interferers (excluding
    anything labeled 'noise'), by raw spike count."""
    ids, counts = np.unique(spike_cl, return_counts=True)
    order = np.argsort(counts)[::-1]

    picked = []
    for idx in order:
        cid = int(ids[idx])
        if cid == target_id:
            continue
        if labels.get(cid, "").lower() == "noise":
            continue
        picked.append(cid)
        if len(picked) == n_interferers:
            break
    return picked


# --------------------------------------------------------------------- #
# Noise covariance (restricted to the selected channel subset)
# --------------------------------------------------------------------- #

def _biased_xcov(x, y, maxlag):
    """Biased cross-covariance of 1-D signals x, y for lags -maxlag..maxlag,
    matching MATLAB's xcov(x, y, maxlag, 'biased')."""
    x = x - x.mean()
    y = y - y.mean()
    full = np.correlate(x, y, mode="full") / len(x)
    mid = len(x) - 1
    return full[mid - maxlag: mid + maxlag + 1]


def noise_covariance(data_sel, all_spike_times, template_length,
                      template_offset, max_samples):
    """Noise covariance over (template_length * n_sel_channels), estimated
    only from periods with no spikes from *any* cluster, restricted to the
    already-selected channel subset (keeps this small & well-conditioned,
    unlike inverting a full-probe-width covariance matrix)."""
    n_samples, n_ch = data_sel.shape
    n_samples = min(n_samples, max_samples)
    data_sel = data_sel[:n_samples]

    spike_present = np.zeros(n_samples, dtype=bool)
    for t in all_spike_times:
        if t >= n_samples + template_length:
            continue
        lo = max(0, t - template_offset)
        hi = min(n_samples, t + template_length - 1 + template_offset)
        spike_present[lo:hi] = True

    # Gaps of "no spike anywhere" long enough to hold two templates.
    d = np.diff(spike_present.astype(np.int8))
    starts = list(np.where(d == -1)[0] + 1)
    ends = list(np.where(d == 1)[0] + 1)
    if not spike_present[0]:
        starts.insert(0, 0)
    if not spike_present[-1]:
        ends.append(n_samples)
    starts, ends = np.array(starts), np.array(ends)

    lengths = ends - starts
    keep = lengths >= 2 * template_length
    starts, ends, lengths = starts[keep], ends[keep], lengths[keep]
    if starts.size == 0:
        raise ValueError("No sufficiently long spike-free segments found "
                          "for noise covariance estimation.")
    total_len = lengths.sum()

    dim = template_length * n_ch
    C = np.zeros((dim, dim))

    for i in range(n_ch):
        for k in range(i + 1):
            cov_combined = np.zeros(2 * template_length - 1)
            for s, e, L in zip(starts, ends, lengths):
                cov_combined += L * _biased_xcov(
                    data_sel[s:e, i], data_sel[s:e, k], template_length - 1
                )
            cov_combined /= total_len

            col = cov_combined[template_length - 1:]
            row = cov_combined[:template_length][::-1]
            T = toeplitz(row, col)

            C[i * template_length:(i + 1) * template_length,
              k * template_length:(k + 1) * template_length] = T
            if i != k:
                C[k * template_length:(k + 1) * template_length,
                  i * template_length:(i + 1) * template_length] = T.T

    return C


# --------------------------------------------------------------------- #
# LCMV filter
# --------------------------------------------------------------------- #

def lcmv_filter(s_flat, interferer_flats, R, ridge=1e-3):
    """Solve f = R^-1 C (C^T R^-1 C)^-1 g for unity gain on the target and
    zero gain on each interferer. `ridge` regularizes R for stability when
    the noise-covariance estimate is based on limited data."""
    dim = s_flat.shape[0]
    R_reg = R + ridge * (np.trace(R) / dim) * np.eye(dim)

    C = np.column_stack([s_flat] + list(interferer_flats)) if interferer_flats else s_flat[:, None]
    g = np.zeros(C.shape[1])
    g[0] = 1.0

    Rinv_C = np.linalg.solve(R_reg, C)
    M = C.T @ Rinv_C
    weights = np.linalg.solve(M, g)
    f_flat = Rinv_C @ weights
    return f_flat


def filter_output(data_sel, f):
    """Apply the filter to (time, channels) data, returning the summed
    per-channel correlation trace D(t). Matches the convention used by the
    original MATLAB script: conv(data, fliplr(filter), 'same') per channel,
    summed across channels."""
    D = np.zeros(data_sel.shape[0])
    for ch in range(data_sel.shape[1]):
        D += np.convolve(data_sel[:, ch], f[::-1, ch], mode="same")
    return D


def compute_threshold(D, all_spike_times, template_length, template_offset, k=5.0):
    """Empirically-calibrated threshold, appropriate for an LCMV filter.

    NOTE: the classical BOTM log-likelihood-ratio threshold
    (0.5*f.s - log(prior odds)) assumes an *unconstrained* matched filter,
    where f.s is a large SNR-like energy term. Our LCMV filter is
    explicitly constrained to unity gain (f.s == 1 by construction), which
    collapses that term to a constant and lets the sparse-firing log-prior
    term dominate completely -- producing a threshold far outside the
    filter's actual output range and suppressing every detection. (Caught
    by testing this script against a synthetic dataset -- see
    FilterGen/eval_synthetic.py.)

    Instead, calibrate directly from the filter's own output: estimate the
    background noise level of D(t) during periods with no known spike from
    any cluster (robust median + MAD, in case some spikes are missing from
    the calibration window), and set the threshold `k` scaled-MADs above
    it. For a session that runs a long time, replace this fixed threshold
    with a running version of the same computation (e.g. over the last few
    hundred ms) so it tracks drifting background/interference levels.
    """
    n_samples = D.shape[0]
    spike_present = np.zeros(n_samples, dtype=bool)
    for t in all_spike_times:
        if t >= n_samples:
            continue
        lo = max(0, t - template_offset - template_length // 2)
        hi = min(n_samples, t + template_length + template_offset)
        spike_present[lo:hi] = True

    noise = D[~spike_present]
    if noise.size == 0:
        noise = D  # fallback: shouldn't happen with sane inputs

    med = np.median(noise)
    mad = np.median(np.abs(noise - med)) * 1.4826  # -> Gaussian-equivalent std
    return med + k * mad


# --------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------- #

def save_filter(out_dir, cluster_id, channels, filt, threshold):
    os.makedirs(out_dir, exist_ok=True)

    np.asarray(channels, dtype=np.int32).tofile(
        os.path.join(out_dir, f"channels_{cluster_id}.bin"))
    np.asarray(filt, dtype=np.float64).tofile(
        os.path.join(out_dir, f"filter_{cluster_id}.bin"))
    np.asarray([threshold], dtype=np.float64).tofile(
        os.path.join(out_dir, f"threshold_{cluster_id}.bin"))


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks-dir", required=True, help="Kilosort output directory")
    ap.add_argument("--bin-path", help="Override raw .bin path (default: params.py dat_path)")
    ap.add_argument("--meta-path", help="Override .meta path (default: <bin>.meta)")
    ap.add_argument("--channel-map-json",
                     help="Restrict to a channel subset given by a Kilosort-style "
                          "probe/channel-map JSON's 'chanMap' field (e.g. a "
                          "per-shank subset). Use when Kilosort was only run on "
                          "part of the probe.")
    ap.add_argument("--target", type=int, required=True, help="Target cluster id")
    ap.add_argument("--interferers", type=int, nargs="*", default=None,
                     help="Explicit interferer cluster ids to null out")
    ap.add_argument("--auto-interferers", type=int, default=0,
                     help="If --interferers not given, auto-pick this many "
                          "by highest spike count")
    ap.add_argument("--n-channels", type=int, default=5,
                     help="Number of channels to keep in the filter")
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--t-max", type=float, default=60.0,
                     help="Max seconds of data to use for noise covariance")
    ap.add_argument("--car", action="store_true", default=True)
    ap.add_argument("--no-car", dest="car", action="store_false")
    ap.add_argument("--filter", action="store_true", default=True,
                     help="Apply high-pass filter before everything else")
    ap.add_argument("--no-filter", dest="filter", action="store_false")
    ap.add_argument("--fc", type=float, default=300.0, help="High-pass cutoff (Hz)")
    ap.add_argument("--causal-highpass", action="store_true",
                     help="Use the causal RBJ biquad (matching "
                          "ClosedLoop/ButterworthHighpass.h exactly) instead "
                          "of zero-phase filtfilt. Use this when the filter "
                          "will be deployed against the C++ live pipeline, "
                          "which can only filter causally -- see "
                          "highpass_causal_biquad()'s docstring.")
    ap.add_argument("--ridge", type=float, default=1e-3,
                     help="Regularization strength for the R inverse")
    ap.add_argument("--threshold-k", type=float, default=5.0,
                     help="Scaled-MADs above background noise for the "
                          "detection threshold (higher = fewer false positives)")
    ap.add_argument("--max-spikes", type=int, default=2000,
                     help="Cap spikes averaged per waveform (speed)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- Load Kilosort + SpikeGLX metadata -----------------------------
    spike_t, spike_cl, labels = load_kilosort(args.ks_dir)
    params = load_params_py(args.ks_dir)

    bin_path = args.bin_path or params.get("dat_path")
    if bin_path is None:
        sys.exit("Could not determine raw .bin path; pass --bin-path.")
    bin_path = os.path.join(args.ks_dir, bin_path) if not os.path.isabs(bin_path) else bin_path

    meta_path = args.meta_path or (os.path.splitext(bin_path)[0] + ".meta")
    meta = load_sglx_meta(meta_path)

    fs = float(meta["imSampRate"])
    n_saved_chans = int(meta["nSavedChans"])
    np_ch = np.array(parse_chan_subset(meta["snsSaveChanSubset"]))
    n_sync = find_sync_channel_count(meta)

    print(f"Sample rate: {fs} Hz, saved channels: {n_saved_chans}, sync chans: {n_sync}")

    # ---- Determine the channel subset BEFORE loading any data -----------
    # Restricting columns up front (rather than after casting the full
    # channel count to float64) matters a lot for real recordings: a
    # full-probe 385-channel, 20+ minute recording would otherwise need
    # 100+ GB just for this one dense array.
    keep_mask = np.ones(len(np_ch), dtype=bool)
    if n_sync:
        keep_mask[-n_sync:] = False

    if args.channel_map_json:
        shank_chans = load_channel_map_json(args.channel_map_json)
        keep_mask &= np.isin(np_ch, shank_chans)
        print(f"Restricting to {keep_mask.sum()} channels from {args.channel_map_json}")

    bad_ch_path = os.path.join(os.path.dirname(bin_path), "bad_channels.tsv")
    if os.path.isfile(bad_ch_path):
        bad_ch = np.loadtxt(bad_ch_path, dtype=int, ndmin=1)
        n_before = keep_mask.sum()
        keep_mask &= ~np.isin(np_ch, bad_ch)
        print(f"Dropped {n_before - keep_mask.sum()} bad channel(s)")

    keep_idx = np.where(keep_mask)[0]
    np_ch = np_ch[keep_idx]

    # ---- Load raw data, already restricted to the channel subset --------
    raw = memmap_raw(bin_path, n_saved_chans)

    last_spike = spike_t.max()
    t_max_samples = int(min(raw.shape[0], last_spike + fs))  # last spike + 1s
    data = np.asarray(raw[:t_max_samples, keep_idx], dtype=np.float64)
    print(f"Loaded data: {data.shape[0]} samples x {data.shape[1]} channels "
          f"({data.nbytes / 1e9:.1f} GB)")

    if args.filter:
        if args.causal_highpass:
            print("Using causal RBJ biquad highpass (matches C++ live pipeline exactly)")
            data = highpass_causal_biquad(data, args.fc, fs)
        else:
            data = highpass(data, args.fc, fs)
    if args.car:
        data = common_average_reference(data)

    # ---- Pick interferers ------------------------------------------------
    if args.interferers:
        interferer_ids = args.interferers
    elif args.auto_interferers:
        interferer_ids = auto_pick_interferers(
            spike_t, spike_cl, args.target, labels, args.auto_interferers)
        print(f"Auto-picked interferers by spike count: {interferer_ids}")
    else:
        interferer_ids = []

    # ---- Mean waveforms --------------------------------------------------
    target_spikes = spike_t[spike_cl == args.target]
    if target_spikes.size == 0:
        sys.exit(f"Target cluster {args.target} has no spikes in spike_clusters.npy.")

    target_wf_full, n_target_used = mean_waveform(
        data, target_spikes, args.template_length, args.template_offset,
        args.max_spikes, rng)
    print(f"Target cluster {args.target}: {target_spikes.size} spikes total, "
          f"{n_target_used} used for template.")

    interferer_wfs_full = []
    interferer_times_all = []
    for cid in interferer_ids:
        sp = spike_t[spike_cl == cid]
        if sp.size == 0:
            print(f"  (interferer {cid} has no spikes, skipping)")
            continue
        interferer_times_all.append(sp)
        wf, n_used = mean_waveform(
            data, sp, args.template_length, args.template_offset,
            args.max_spikes, rng)
        interferer_wfs_full.append(wf)
        print(f"Interferer cluster {cid}: {sp.size} spikes total, {n_used} used.")

    # ---- Channel selection (target-vs-interferer energy ratio) ----------
    sel = select_channels(target_wf_full, interferer_wfs_full, args.n_channels)
    sel_channels = np_ch[sel]
    print(f"Selected channels (SpikeGLX indices): {sel_channels.tolist()}")

    data_sel = data[:, sel]
    s = target_wf_full[:, sel]
    interferer_wfs_sel = [wf[:, sel] for wf in interferer_wfs_full]

    # ---- Noise covariance, restricted to selected channels ---------------
    # "Not noise" should mean "could contaminate these specific channels",
    # not "any spike anywhere on the probe" -- with many clusters and a
    # busy recording, masking out every single spike from every cluster
    # (regardless of where on the probe it fires) can blanket the entire
    # recording and leave no spike-free segments at all, even though most
    # of those spikes are on channels far from our local channel subset
    # and never actually appear in data_sel. Restrict to target + the
    # named interferers, which are the only units we've established are
    # spatially relevant to these channels.
    all_spike_times = np.concatenate([target_spikes] + interferer_times_all) \
        if interferer_times_all else target_spikes
    all_spike_times.sort()
    R = noise_covariance(data_sel, all_spike_times, args.template_length,
                          args.template_offset, int(args.t_max * fs))

    # ---- LCMV filter -------------------------------------------------------
    s_flat = s.T.ravel()  # (n_channels, template_length) flattened -> matches R's channel-major blocks
    interferer_flats = [wf.T.ravel() for wf in interferer_wfs_sel]

    f_flat = lcmv_filter(s_flat, interferer_flats, R, ridge=args.ridge)
    f = f_flat.reshape(len(sel), args.template_length).T  # back to (template_length, n_channels)

    # Sanity: constraints should hold (up to ridge regularization slack)
    print(f"Constraint check -- gain on target: {np.dot(f_flat, s_flat):.4f} (want 1.0)")
    for i, s_int in enumerate(interferer_flats):
        print(f"  gain on interferer {interferer_ids[i]}: {np.dot(f_flat, s_int):.4e} (want ~0)")

    D = filter_output(data_sel, f)
    threshold = compute_threshold(
        D, all_spike_times, args.template_length, args.template_offset,
        k=args.threshold_k)

    # Calibration-window diagnostics -- how many true target spikes actually
    # cross this threshold nearby, and how many false positives elsewhere.
    window = args.template_length // 4
    hit_mask = np.zeros(target_spikes.shape, dtype=bool)
    for i, t in enumerate(target_spikes):
        if t < D.shape[0]:
            lo, hi = max(0, t - window), min(D.shape[0], t + window)
            hit_mask[i] = np.any(D[lo:hi] > threshold)
    print(f"Calibration-window check: {hit_mask.sum()}/{len(target_spikes)} "
          f"target spikes cross threshold nearby "
          f"(threshold={threshold:.4f}, k={args.threshold_k})")

    save_filter(args.out_dir, args.target, sel_channels, f, threshold)
    print(f"Saved filter for cluster {args.target} to {args.out_dir} "
          f"(threshold={threshold:.4f})")

    # ---- Optional validation plot -----------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt

        gt = np.zeros_like(D)
        gt[target_spikes[(target_spikes >= 0) & (target_spikes < len(gt))]] = D.max()

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

        axes[0].plot(s)
        axes[0].set_title(f"Target cluster {args.target} mean waveform (selected channels)")

        t = np.arange(len(D)) / fs
        axes[1].plot(t, D, lw=0.5, label="filter output")
        axes[1].axhline(threshold, color="r", ls="--", label="threshold")
        axes[1].scatter(target_spikes / fs, np.full(target_spikes.shape, D.max()),
                         marker="|", color="g", label="ground-truth spikes")
        axes[1].set_title("LCMV filter output vs. threshold")
        axes[1].legend(loc="upper right")
        axes[1].set_xlabel("time (s)")

        axes[2].plot(f)
        axes[2].set_title("LCMV filter taps (per selected channel)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
