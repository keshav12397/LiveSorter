"""Compare plain matched-filter vs LCMV-nulled filter detection performance
on the synthetic dataset."""

import os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "synthetic")
FS = 30000.0


def load_filter(out_dir, cluster_id):
    channels = np.fromfile(os.path.join(out_dir, f"channels_{cluster_id}.bin"), dtype=np.int32)
    n_ch = channels.size
    filt = np.fromfile(os.path.join(out_dir, f"filter_{cluster_id}.bin"), dtype=np.float64)
    filt = filt.reshape(-1, n_ch)  # (template_length, n_channels)
    threshold = np.fromfile(os.path.join(out_dir, f"threshold_{cluster_id}.bin"), dtype=np.float64)[0]
    return channels, filt, threshold


def evaluate(name, out_dir):
    channels, filt, threshold = load_filter(out_dir, 0)

    n_saved = 13  # from synthetic .meta
    raw = np.memmap(os.path.join(OUT, "synthetic.bin"), dtype=np.int16, mode="r")
    raw = raw.reshape(-1, n_saved)
    data_sel = raw[:, channels].astype(np.float64)

    D = np.zeros(data_sel.shape[0])
    for ch in range(data_sel.shape[1]):
        D += np.convolve(data_sel[:, ch], filt[::-1, ch], mode="same")

    target_times = np.load(os.path.join(OUT, "kilosort", "spike_times.npy"))
    clusters = np.load(os.path.join(OUT, "kilosort", "spike_clusters.npy"))
    target_times = target_times[clusters == 0]
    interf1_times = np.load(os.path.join(OUT, "kilosort", "spike_times.npy"))[clusters == 1]

    # peak-find: local maxima above threshold, at least 15 samples apart (0.5ms)
    above = D > threshold
    peaks = []
    i = 0
    n = len(D)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            peaks.append(i + np.argmax(D[i:j]))
            i = j
        else:
            i += 1
    peaks = np.array(peaks)

    # hit = a detected peak within 15 samples (0.5ms) of a true target spike
    window = 15
    hits = 0
    for t in target_times:
        if np.any(np.abs(peaks - t) <= window):
            hits += 1
    miss = len(target_times) - hits

    # false positives near a big interferer-1 spike (within 15 samples)
    fp_near_interf1 = 0
    for p in peaks:
        if np.any(np.abs(interf1_times - p) <= window):
            # only count as FP if it's not actually also near a real target spike
            if not np.any(np.abs(target_times - p) <= window):
                fp_near_interf1 += 1

    total_fp = 0
    for p in peaks:
        if not np.any(np.abs(target_times - p) <= window):
            total_fp += 1

    print(f"--- {name} ---")
    print(f"  channels used: {channels.tolist()}")
    print(f"  threshold: {threshold:.3f}")
    print(f"  total peaks found: {len(peaks)}")
    print(f"  true target spikes: {len(target_times)}  hits: {hits}  misses: {miss}")
    print(f"  total false positives: {total_fp}")
    print(f"  false positives coinciding with interferer-1 spikes: {fp_near_interf1}")
    print()


if __name__ == "__main__":
    evaluate("PLAIN matched filter (no nulling)", os.path.join(OUT, "out_plain"))
    evaluate("LCMV filter (nulls interferers 1 & 2)", os.path.join(OUT, "out_lcmv"))
