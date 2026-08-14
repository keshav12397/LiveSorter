"""
Build a small synthetic Kilosort + SpikeGLX dataset to validate
generate_filter.py end-to-end without needing real recordings.

Scenario mirrors the described use case:
- 12 channels
- target unit: cluster 0, LOW amplitude, fires rarely (sparse), spans
  channels 2-6
- interferer unit: cluster 1, HIGH amplitude, fires often, spans channels
  3-7 (overlaps target's channels -- the whole point of LCMV nulling)
- interferer unit: cluster 2, HIGH amplitude, fires often, different
  channels (8-11), to make sure channel selection avoids it naturally
- Gaussian background noise on all channels, plus correlated noise via a
  shared random signal added to a few adjacent channels (so R isn't just
  diagonal -- exercises the covariance/whitening code path for real)
"""

import os
import shutil

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "synthetic")
KS_DIR = os.path.join(OUT, "kilosort")
FS = 30000.0
N_CHAN = 12
DURATION_S = 120.0
N_SAMPLES = int(FS * DURATION_S)
TEMPLATE_LENGTH = 61
TEMPLATE_OFFSET = 20

rng = np.random.default_rng(42)


def gauss_bump(length, center, width, amp):
    x = np.arange(length)
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def make_waveform(n_chan, chan_lo, chan_hi, amp, polarity=-1):
    """A simple biphasic spike-like waveform spread over a channel range,
    amplitude tapering away from the central channel."""
    wf = np.zeros((TEMPLATE_LENGTH, n_chan))
    center_ch = (chan_lo + chan_hi) // 2
    for ch in range(chan_lo, chan_hi + 1):
        atten = 1.0 - 0.25 * abs(ch - center_ch)
        atten = max(atten, 0.15)
        main = polarity * gauss_bump(TEMPLATE_LENGTH, TEMPLATE_OFFSET, 3, amp * atten)
        rebound = -polarity * gauss_bump(TEMPLATE_LENGTH, TEMPLATE_OFFSET + 10, 5, 0.3 * amp * atten)
        wf[:, ch] = main + rebound
    return wf


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(KS_DIR)

    # ---- background noise -------------------------------------------------
    data = rng.normal(0, 15.0, size=(N_SAMPLES, N_CHAN))

    # shared low-frequency correlated noise across channels 2-9 (like real
    # local-field / crosstalk correlation), gives R off-diagonal structure
    shared = rng.normal(0, 1.0, size=N_SAMPLES)
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 500 / (FS / 2), btype="low")
    shared = filtfilt(b, a, shared) * 8.0
    for ch in range(2, 10):
        data[:, ch] += shared * (1.0 - 0.08 * abs(ch - 5))

    # ---- templates ----------------------------------------------------------
    target_wf = make_waveform(N_CHAN, 2, 6, amp=40.0)     # LOW amplitude
    interf1_wf = make_waveform(N_CHAN, 3, 7, amp=400.0)   # HIGH amplitude, overlapping channels
    interf2_wf = make_waveform(N_CHAN, 8, 11, amp=350.0)  # HIGH amplitude, separate channels

    # ---- spike trains ---------------------------------------------------------
    # target: sparse, ~40 spikes over 120s
    min_gap = int(0.01 * FS)
    target_times = np.sort(rng.choice(
        np.arange(TEMPLATE_OFFSET + min_gap, N_SAMPLES - TEMPLATE_LENGTH - min_gap, min_gap),
        size=40, replace=False))

    # interferer 1: frequent, ~4000 spikes
    interf1_times = np.sort(rng.choice(
        np.arange(TEMPLATE_OFFSET + min_gap, N_SAMPLES - TEMPLATE_LENGTH - min_gap, min_gap),
        size=4000, replace=False))

    # interferer 2: frequent, ~3000 spikes
    interf2_times = np.sort(rng.choice(
        np.arange(TEMPLATE_OFFSET + min_gap, N_SAMPLES - TEMPLATE_LENGTH - min_gap, min_gap),
        size=3000, replace=False))

    def stamp(times, wf):
        for t in times:
            lo = t - TEMPLATE_OFFSET
            data[lo:lo + TEMPLATE_LENGTH, :] += wf

    stamp(target_times, target_wf)
    stamp(interf1_times, interf1_wf)
    stamp(interf2_times, interf2_wf)

    # ---- write raw binary (int16, channel-interleaved per sample) -----------
    scale = 2.34  # arbitrary int16-per-uV-ish scale, doesn't matter for this test
    raw_i16 = np.clip(data * scale, -32768, 32767).astype(np.int16)
    bin_path = os.path.join(OUT, "synthetic.bin")
    raw_i16.tofile(bin_path)

    # ---- write .meta ----------------------------------------------------------
    meta_path = os.path.join(OUT, "synthetic.meta")
    with open(meta_path, "w") as fh:
        fh.write(f"imSampRate={FS}\n")
        fh.write(f"nSavedChans={N_CHAN}\n")
        fh.write(f"snsSaveChanSubset=0:{N_CHAN - 1}\n")
        fh.write("snsApLfSy=" + str(N_CHAN - 1) + ",0,1\n")  # last chan = sync
        fh.write(f"fileSizeBytes={os.path.getsize(bin_path)}\n")

    # Re-write with an actual sync channel present (all zero) so the
    # snsApLfSy=...,1 claim is honest, and n_saved_chans includes it.
    raw_i16_with_sync = np.concatenate(
        [raw_i16, np.zeros((N_SAMPLES, 1), dtype=np.int16)], axis=1)
    raw_i16_with_sync.tofile(bin_path)
    with open(meta_path, "w") as fh:
        n_with_sync = N_CHAN + 1
        fh.write(f"imSampRate={FS}\n")
        fh.write(f"nSavedChans={n_with_sync}\n")
        fh.write(f"snsSaveChanSubset=0:{n_with_sync - 1}\n")
        fh.write(f"snsApLfSy={N_CHAN},0,1\n")
        fh.write(f"fileSizeBytes={os.path.getsize(bin_path)}\n")

    # ---- write Kilosort-style outputs -----------------------------------------
    all_times = np.concatenate([target_times, interf1_times, interf2_times])
    all_clusters = np.concatenate([
        np.zeros_like(target_times),
        np.ones_like(interf1_times),
        2 * np.ones_like(interf2_times),
    ])
    order = np.argsort(all_times)
    all_times, all_clusters = all_times[order], all_clusters[order]

    np.save(os.path.join(KS_DIR, "spike_times.npy"), all_times.astype(np.int64))
    np.save(os.path.join(KS_DIR, "spike_clusters.npy"), all_clusters.astype(np.int64))

    with open(os.path.join(KS_DIR, "cluster_group.tsv"), "w") as fh:
        fh.write("cluster_id\tgroup\n")
        fh.write("0\tgood\n1\tgood\n2\tgood\n")

    with open(os.path.join(KS_DIR, "params.py"), "w") as fh:
        fh.write(f'dat_path = r"{bin_path}"\n')
        fh.write(f"n_channels_dat = {n_with_sync}\n")
        fh.write(f"sample_rate = {FS}\n")

    print(f"Synthetic dataset written to {OUT}")
    print(f"  target cluster 0: {len(target_times)} spikes, amp=40 (LOW), channels 2-6")
    print(f"  interferer cluster 1: {len(interf1_times)} spikes, amp=400 (HIGH), channels 3-7 (overlaps target)")
    print(f"  interferer cluster 2: {len(interf2_times)} spikes, amp=350 (HIGH), channels 8-11 (separate)")
    print(f"  ks_dir = {KS_DIR}")


if __name__ == "__main__":
    main()
