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
    spike_cl = np.load(os.path.join(ks_dir, "spike_templates.npy")).ravel().astype(np.int64)

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


def load_channel_positions_json(path):
    """Same file as load_channel_map_json(), but returns {raw_channel_id:
    (x, y)} from the 'xc'/'yc' arrays (indexed in parallel with 'chanMap') --
    used for auto_pick_interferers_spatial()'s physical-distance ranking.
    Returns {} if the file has no 'xc'/'yc' fields."""
    import json
    with open(path) as fh:
        j = json.load(fh)
    if "xc" not in j or "yc" not in j:
        return {}
    chan_map = j["chanMap"]
    return {int(c): (float(x), float(y)) for c, x, y in zip(chan_map, j["xc"], j["yc"])}


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
    """CAR: subtract the per-sample median across channels, in place.

    np.median still makes one internal copy of `data` to partition (it can't
    scramble the caller's array since the subtraction below needs the
    original values), but subtracting in place avoids allocating a second
    full-size array on top of that -- meaningful on recordings large enough
    to be RAM-constrained in the first place.
    """
    ref = np.median(data, axis=1, keepdims=True)
    data -= ref
    return data


def highpass(data, fc, fs, order=2):
    """Zero-phase high-pass, matching butter(2, ...)+filtfilt in MATLAB."""
    b, a = butter(order, 2 * fc / fs, btype="high")
    return filtfilt(b, a, data, axis=0)


def causal_biquad_coeffs(fc, fs):
    """(b, a) for the RBJ Audio-EQ-Cookbook 2nd-order Butterworth highpass
    biquad, coefficient-for-coefficient identical to
    ClosedLoop/ButterworthHighpass.h (Q=1/sqrt(2)). Split out from
    highpass_causal_biquad() so a chunked/streaming caller can drive
    scipy.signal.lfilter itself, carrying the 2-sample filter state (zi/zf)
    across chunk boundaries -- that reproduces filtering the whole array in
    one lfilter() call exactly (no overlap padding needed: an IIR filter's
    only cross-chunk dependency is that small state vector).
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
    return b, a


def highpass_causal_biquad(data, fc, fs):
    """Causal high-pass -- see causal_biquad_coeffs() for the coefficients.

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
    b, a = causal_biquad_coeffs(fc, fs)
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
    anything labeled 'noise'), by raw spike count.

    Not spatially aware -- see auto_pick_interferers_spatial() for a version
    that only considers clusters actually near the target's own channels,
    which matters a lot for a sparse/poorly-isolated target: the busiest
    clusters probe-wide are often nowhere near it and don't interfere with
    it at all, while a much less active but spatially adjacent cluster can
    dominate its channels.
    """
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


def auto_pick_interferers_spatial(data, spike_t, spike_cl, np_ch, target_id, labels,
                                   n_interferers, template_length=61, template_offset=20,
                                   pool_size=None, peak_check_spikes=200,
                                   channel_positions=None, rng=None):
    """Spatially-aware version of auto_pick_interferers(): only picks
    interferers whose own peak channel is physically close to the target's,
    instead of just the busiest clusters anywhere on the probe.

    Two-stage to stay fast: first narrows to a `pool_size` (default
    max(3*n_interferers, 30)) shortlist by raw activity (cheap, like
    auto_pick_interferers()), then -- only for that shortlist -- computes a
    quick mean waveform (small `peak_check_spikes` subsample, not the full
    `max_spikes` used for the final fit) over `data`/`np_ch` (already
    restricted to the target's own channel group, e.g. one shank) to find
    each candidate's peak channel, and ranks the shortlist by distance from
    the target's own peak channel. `channel_positions`, if given, is a dict
    mapping raw channel id -> (x, y) (see generate_filter's
    --channel-map-json handling) for true physical distance; without it,
    falls back to index-distance within `np_ch` (fine when np_ch is a
    single, contiguous shank, as it usually is here).
    """
    rng = rng or np.random.default_rng()
    if pool_size is None:
        pool_size = max(3 * n_interferers, 30)

    ids, counts = np.unique(spike_cl, return_counts=True)
    order = np.argsort(counts)[::-1]

    pool = []
    for idx in order:
        cid = int(ids[idx])
        if cid == target_id:
            continue
        if labels.get(cid, "").lower() == "noise":
            continue
        pool.append(cid)
        if len(pool) == pool_size:
            break

    def peak_channel_and_pos(spike_times):
        wf, _ = mean_waveform(data, spike_times, template_length, template_offset,
                               max_spikes=peak_check_spikes, rng=rng)
        peak_local = int(np.argmax(np.ptp(wf, axis=0)))  # index within np_ch
        raw_ch = int(np_ch[peak_local])
        if channel_positions is not None and raw_ch in channel_positions:
            return channel_positions[raw_ch]
        return (float(peak_local), 0.0)  # fallback: index-distance only

    target_pos = peak_channel_and_pos(spike_t[spike_cl == target_id])

    scored = []
    for cid in pool:
        try:
            pos = peak_channel_and_pos(spike_t[spike_cl == cid])
        except ValueError:
            continue  # too few/edge-clipped spikes to form a waveform -- skip
        dist = np.hypot(pos[0] - target_pos[0], pos[1] - target_pos[1])
        scored.append((dist, cid))

    scored.sort(key=lambda x: x[0])
    picked = [cid for _, cid in scored[:n_interferers]]
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


def noise_cov_by_lag(data_sel, all_spike_times, template_length,
                     template_offset, max_samples):
    """Per-lag noise cross-covariance, `(2*template_length-1, n_ch, n_ch)`,
    indexed by lag d = -(template_length-1) .. +(template_length-1).

    This is the part of the noise estimate that costs a pass over the
    recording. `noise_covariance_from_lags` turns it into the space-time
    matrix R that `lcmv_filter` consumes, and
    `noise_covariance_vectorized` is the two composed -- the original
    entry point, byte-for-byte unchanged in behaviour.

    Splitting them is what makes REFITTING a filter cheap. R is
    block-Toeplitz, and each block depends only on one channel pair's
    covariance at each lag, so R for ANY subset of channels is an index
    into this array plus an assembly whose cost depends on the subset size,
    not on the recording length. A unit that has drifted onto different
    channels can therefore be refit without rescanning the data -- see
    `noise_covariance_from_lags`.

    MEMORY is trivial; TIME is not. The result is (2L-1) * n_ch^2 floats --
    121 * 96 * 96 * 8 B = 8.9 MB for this rig, nothing to keep resident. But
    the einsum below computes every channel PAIR at every lag, so the scan
    is O(n_samples * (2L-1) * n_ch^2), quadratic in channel count:

        n_ch     cost            a 1800 s session at 30 kHz
          96     784 us/sample   11.8 HOURS
          16     15.8 us/sample  14 minutes
           5     2.3 us/sample   2 minutes

    So do NOT call this over the full 96-channel group and treat the result
    as a cheap one-off. (An earlier version of this docstring effectively
    did: it quoted 286 s, measured on a 400k-sample fixture ~135x smaller
    than a real session, and the per-refit figure below was correct while
    the setup cost was off by two orders of magnitude.)

    Pass a BAND instead -- `data[:, band]` for a slice of channels around the
    unit, with `sel` then indexed relative to the band. Drift moves a unit's
    channel set by a few rows, so a ~16-channel neighbourhood covers any
    plausible excursion at 1/49th the cost. The band has to be chosen wide
    enough to contain every channel the unit might select after drifting; a
    selection that escapes the band cannot be assembled from it and must
    fall back to a rescan.

    Everything below is the original vectorized implementation, unchanged.
    It computes every
    channel-pair's cross-covariance at every lag for a segment in ONE
    np.einsum call instead of looping over each of the n_ch*(n_ch+1)/2
    channel pairs and calling np.correlate(mode='full') separately per pair.

    That per-pair np.correlate is the real cost: it computes a FULL
    correlation (cost O(segment_length^2)) even though only the narrow band
    of lags -( template_length-1)..+(template_length-1) is ever used. This
    version instead builds sliding windows only as wide as that lag band
    (via np.lib.stride_tricks.sliding_window_view, no data copy) and gets
    every pair/lag via one einsum, which is O(segment_length * lag_band) --
    linear in segment length instead of quadratic. Measured effect on a
    synthetic worst case (one ~19k-sample spike-free segment): ~3.7x faster
    -- real, but more modest than the segment-length complexity alone
    suggests, since np.correlate's direct-method C implementation has a much
    better constant factor than a naive flop count assumes (see
    test_noise_covariance_equivalence.py for the numerical-equivalence +
    timing check this was validated against). Note
    this alone did NOT explain the ~70x real-world slowdown seen during a
    100-unit calibration run that prompted this change -- that was
    independently diagnosed as external CPU contention from concurrent
    processes, not this function. Kept anyway as a legitimate,
    verified-correct efficiency improvement.

    See _biased_xcov/noise_covariance's docstrings for the "biased" MATLAB
    xcov convention and the L-weighting algebra this mirrors -- working
    through that algebra shows the original's `cov_combined += L *
    _biased_xcov(...)` (which itself divides by L internally) then a final
    `/= total_len` reduces to: sum the RAW (un-normalized) cross-correlation
    across every segment, divide once by total_len at the end. No
    per-segment length weighting is actually needed in the loop below --
    it's already cancelled out algebraically.
    """
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
    total_len = int(lengths.sum())

    # How much data actually went in, and a warning when it is not enough.
    #
    # The failure this catches is silent, not loud. Excluding EVERY sorted
    # unit's spikes from a dense recording leaves almost no spike-free time:
    # measured on a 160-unit session, the 900 s train half came out 99.5%
    # spike-present, with 13 usable gaps totalling 1869 samples -- 0.06 s.
    # That does not raise above, because 13 segments is not 0. It returns a
    # (2L-1, n_ch, n_ch) array estimated from 1869 samples and looks exactly
    # like a real answer.
    #
    # The threshold is set against what R actually costs to estimate. The
    # assembled matrix is (template_length * n_selected)^2, and a covariance
    # wants an order of magnitude more samples than its dimension. This
    # function does not know n_selected -- that is chosen per unit, after
    # this runs -- so the floor assumes the 5-channel selection this rig
    # uses: 10 * 5 * template_length. Below that, R is being estimated from
    # fewer than ~10 samples per dimension and the ridge is doing the work.
    #
    # The 1869-sample case above sits at 6 samples per dimension, so a
    # looser floor would have let exactly the case this exists for pass.
    #
    # Note that intervals OVERLAP heavily at these rates, so estimating
    # coverage by summing (spike count x blanking width) is badly wrong:
    # that arithmetic gave 5.2x oversubscribed where the true coverage was
    # 99.5%. Measure the union, or just read total_len here.
    if total_len < 50 * template_length:
        import warnings
        warnings.warn(
            "noise covariance estimated from only {} samples in {} segments "
            "({:.3f} s at 30 kHz). The exclusion set may be leaving almost no "
            "spike-free data -- excluding every sorted unit in a dense "
            "recording typically does. The returned array is shaped like a "
            "valid estimate but is not one.".format(
                total_len, starts.size, total_len / 30000.0 ),
            RuntimeWarning, stacklevel=2 )

    maxlag = template_length - 1
    nlags = 2 * maxlag + 1
    acc = np.zeros((nlags, n_ch, n_ch), dtype=np.float64)  # float64 accumulator regardless of data dtype -- see fit_lcmv's precision-boundary note

    for s, e, L in zip(starts, ends, lengths):
        seg = data_sel[s:e, :].astype(np.float64)
        seg = seg - seg.mean(axis=0, keepdims=True)  # per-segment, per-channel demean, matching _biased_xcov

        padded = np.zeros((L + 2 * maxlag, n_ch), dtype=np.float64)
        padded[maxlag:maxlag + L] = seg
        # (nlags, n_ch, L) view, no copy -- windows[o, k, :] == padded[o:o+L, k]
        windows = np.lib.stride_tricks.sliding_window_view(padded, L, axis=0)

        # cross[o, i, k] = sum_t seg[t, i] * padded[o+t, k]
        #                = sum_t seg[t, i] * windows[o, k, t]
        cross = np.einsum('ti,okt->oik', seg, windows)

        # Offset o and lag d relate by o = maxlag - d (derived from matching
        # np.correlate's own indexing convention -- see class/module comment
        # and the equivalence test this was validated against), so reversing
        # the offset axis (o: 0..2*maxlag ascending) gives lag d: -maxlag..
        # +maxlag ascending, same order _biased_xcov returns.
        acc += cross[::-1]

    # (nlags, n_ch, n_ch), nlags axis indexed by lag d = -maxlag..maxlag
    return acc / total_len


def noise_covariance_from_lags(cov_by_lag, sel, template_length):
    """Assemble the space-time noise covariance R for the channels in `sel`
    from a per-lag covariance produced by `noise_cov_by_lag`.

    `cov_by_lag` is (2*template_length-1, n_ch, n_ch) over the FULL channel
    group; `sel` indexes into that group. Returns
    (template_length*len(sel), template_length*len(sel)), the matrix
    `lcmv_filter` takes.

    R is block-Toeplitz because the noise is treated as wide-sense
    stationary: block (i, k) is the covariance between channel i's
    template_length-sample window and channel k's, which depends only on the
    lag between the two samples, not on absolute time. So the whole block is
    determined by one channel pair's covariance across lags -- which is why
    the expensive scan can be done once over every channel and reused for
    any subset afterwards.

    No recomputation and no data access: this is pure index-and-assemble,
    O(len(sel)^2 * template_length^2). For 5 channels and L=61 that is 15
    upper-triangle blocks of 61x61.
    """
    sel = np.asarray(sel, dtype=int)
    n_ch = sel.size
    dim = template_length * n_ch
    C = np.zeros((dim, dim))

    for i in range(n_ch):
        for k in range(i + 1):
            # length 2*template_length-1, the shape _biased_xcov returned
            cov_combined = cov_by_lag[:, sel[i], sel[k]]

            col = cov_combined[template_length - 1:]
            row = cov_combined[:template_length][::-1]
            T = toeplitz(row, col)
            C[i * template_length:(i + 1) * template_length,
              k * template_length:(k + 1) * template_length] = T
            if i != k:
                C[k * template_length:(k + 1) * template_length,
                  i * template_length:(i + 1) * template_length] = T.T

    return C


def noise_covariance_vectorized(data_sel, all_spike_times, template_length,
                                 template_offset, max_samples):
    """Drop-in replacement for noise_covariance() -- identical inputs and
    output, verified numerically equivalent in FilterGen tests.

    Now a composition of `noise_cov_by_lag` (the pass over the recording)
    and `noise_covariance_from_lags` (the assembly). Callers that will refit
    the same channels' neighbours later should call those two directly and
    keep the per-lag array; callers doing a single fit can keep using this.
    """
    cov_by_lag = noise_cov_by_lag(data_sel, all_spike_times, template_length,
                                  template_offset, max_samples)
    return noise_covariance_from_lags(cov_by_lag, np.arange(data_sel.shape[1]),
                                      template_length)


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
    # dtype follows the input (not hardcoded float64) so that scoring done in
    # float32 (e.g. calibrate_all_units.py, to match the online float32 GPU
    # kernel's precision) actually produces float32 D-values -- otherwise a
    # threshold picked against silently-upcast float64 scores would be
    # calibrated for a distribution the GPU path never actually produces.
    D = np.zeros(data_sel.shape[0], dtype=data_sel.dtype)
    for ch in range(data_sel.shape[1]):
        D += np.convolve(data_sel[:, ch], f[::-1, ch], mode="same")
    return D


def detection_lag(f, target_template_sel, template_offset):
    """Samples between a spike's Kilosort time and the peak filter_output()
    produces for it. Positive means the detector fires late.

    Derived from filter_output()'s own convention rather than measured, so
    it costs nothing and cannot disagree with it. With L = template_length,
    `np.convolve(x, f[::-1], mode='same')` gives

        D[n] = sum_m f[m] * x[n - L//2 + m]

    and mean_waveform() defines the template by x[t - template_offset + k]
    = s[k] for a spike at t. Substituting,

        D[t + d] = sum_m f[m] * s[m + d - (L//2 - template_offset)]

    so the response peaks at d = (L//2 - template_offset) + argmax_l
    <f, s shifted by l>. For a plain matched filter (f proportional to s)
    that argmax is 0 and the whole lag is the fixed L//2 - template_offset
    the README describes. LCMV's f is s whitened by R^-1 and constrained to
    null its neighbours, and that filter is *not* generally aligned with s:
    on D:/sim_validate this term alone ranges over -24..+8 samples across
    twelve units, so the total lag ranges -14..+18.

    That matters because validation matches detections to ground truth
    within +/- template_length//4 (15 samples at L=61). A unit whose lag
    exceeds that window scores near zero recall at *every* threshold while
    detecting perfectly well -- and since best-F1 threshold selection is
    driven by that same scoring, an uncorrected sweep also picks the wrong
    threshold to deploy. See this branch's commit log for the two units this
    was found on.

    Verified against measurement in that same session: predicted lag equals
    the median empirical detection-minus-truth offset exactly for 10 of 12
    units and within 3 samples for the two whose lag sits past the matching
    window (where the empirical median saturates, since a neighbouring peak
    becomes the nearest one).
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(target_template_sel, dtype=np.float64)
    L = f.shape[0]
    lags = np.arange(-(L - 1), L)
    corr = np.empty(lags.size)
    for i, l in enumerate(lags):
        if l >= 0:
            corr[i] = np.sum(f[:L - l] * s[l:])
        else:
            corr[i] = np.sum(f[-l:] * s[:L + l])
    return int(lags[int(np.argmax(corr))]) + (L // 2 - template_offset)


def compute_threshold(D, all_spike_times, template_length, template_offset, k=5.0):
    """Empirically-calibrated threshold, appropriate for an LCMV filter.

    NOTE: the classical BOTM log-likelihood-ratio threshold
    (0.5*f.s - log(prior odds)) assumes an *unconstrained* matched filter,
    where f.s is a large SNR-like energy term. Our LCMV filter is
    explicitly constrained to unity gain (f.s == 1 by construction), which
    collapses that term to a constant and lets the sparse-firing log-prior
    term dominate completely -- producing a threshold far outside the
    filter's actual output range and suppressing every detection. (Caught
    by testing this script against a synthetic dataset.)

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
