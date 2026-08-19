"""
stream_alignment.py
====================

Maps a live/replayed SpikeGLX stream's absolute sample counter onto the
position in the .bin file Kilosort was run against, so live detections can
be compared to `spike_times.npy` ground truth.

Why this module exists
----------------------
Both validation scripts here (validate_all_units.py,
analyze_unit_quality_vs_tracking.py) used to assume

    file_position = live_sample_index % wrap_samples

That assumption is wrong twice over for a SpikeGLX instance replaying a
recording in a loop, and getting it wrong does not look like a broken
alignment -- it looks like a broken *detector*, which is exactly how it was
misdiagnosed once already (see this repo's live_tracking_bug_report.md):

  1. PHASE. The stream's sample counter and the replayed file's own read
     position are not phase-locked. The counter starts wherever the SpikeGLX
     instance happened to already be when the run connected, and the replay
     loop restarts on its own schedule, so the true mapping is
     `(live_index + D) % wrap_samples` for an unknown constant D -- not
     D == 0. On the run this module was written against, D was ~7.79e6
     samples (~260 s). With D assumed 0, every detection lands on unrelated
     ground truth.

  2. SLIP. D is not even constant within one pass: the observed replay
     source stepped D by ~-149 samples every ~30 s (its stream counter
     advancing slightly further than its file read position). Over a
     3-minute run that is ~750 samples of accumulated slip -- far outside the
     ~100 sample match tolerance -- and over a 52-minute run it is enormous,
     which is why longer runs looked no better than short ones.

Why a wrong alignment reads as "sparse units are broken, dense units are
fine": with the trains unrelated, a "match" within +/-tol samples is pure
coincidence, and coincidence rate scales with the unit's firing rate. A
174 Hz unit gets ~90% of its detections "matched" by chance at ANY lag; a
0.5 Hz unit gets ~0%. That produces a strong recall-vs-spike-count
correlation and no correlation with amplitude or isolation -- precisely the
pattern that looked like a pipeline bug. Any future analysis that sees that
signature should re-check alignment FIRST (align_detections() reports its
own peak significance for exactly this reason).

What this module does instead
-----------------------------
Estimates the mapping from the data, never assumes it:

  * `estimate_global_offset()` -- per-unit circular cross-correlation of
    detections against that unit's own ground truth, binned, over the whole
    file length and summed across units; gives D to within one bin, then a
    sample-resolution refinement. Summing per-unit correlations (rather than
    correlating one pooled train against another) is what makes the peak
    stand out: see that function's docstring.
  * `track_offset()` -- re-estimates D independently in successive blocks of
    the run (seeded from, and searched near, the running estimate) so the
    slip above is followed instead of averaged over.
  * `align_detections()` -- ties both together and returns per-detection file
    positions, plus a loop-pass label so a multi-pass run can be reduced to
    one uncontaminated pass (the same reason
    analyze_unit_quality_vs_tracking.py already had a --loop-pass selector:
    a plain modulo merges every pass onto the same ground-truth spikes, so a
    reliably-detected spike collects N-1 spurious false positives).

Everything here is pure numpy -- no scipy dependency.
"""

import numpy as np


# Correlation-peak z-score below which a global-offset estimate is not
# trusted. The peak is compared against the mean/std of the correlation at
# every other lag, so this is "how many noise-sigma above the rest of the
# search space" -- a genuine alignment on real data scores in the tens (the
# run this was developed against: ~25), pure noise scores ~4-5 simply from
# taking a max over ~1e6 lags.
MIN_PEAK_Z = 8.0


def _binned_counts(samples, n_bins, bin_samples):
    idx = np.asarray(samples, dtype=np.int64) // bin_samples
    idx = idx[(idx >= 0) & (idx < n_bins)]
    return np.bincount(idx, minlength=n_bins).astype(np.float64)


def estimate_global_offset(det_index, det_unit, gt_times, gt_unit, wrap_samples,
                           bin_samples=30, refine_halfwidth=None,
                           max_units=40, min_unit_detections=50):
    """Estimate the constant offset D with `file_pos = (det_index + D) % wrap`.

    det_index, det_unit : live detections' stream sample indices (any pass,
                          unwrapped) and their Kilosort cluster ids.
    gt_times,  gt_unit  : spike_times.npy / spike_templates.npy.
    wrap_samples        : the recording's true total sample count.

    Correlates each unit's detections against THAT unit's own ground truth
    and sums the per-unit normalized correlations. Pooling every unit's
    spikes into one train before correlating does not work: the cross terms
    (unit A's detections against unit B's spikes) contribute a large,
    structured background -- on the run this was developed against it left
    the true peak at only z=4, indistinguishable from noise, whereas the
    per-unit sum puts it at z=25.

    Returns (offset, peak_z); treat peak_z < MIN_PEAK_Z as "no alignment
    found" rather than trusting the offset.
    """
    det_index = np.asarray(det_index, dtype=np.int64)
    det_unit = np.asarray(det_unit)
    gt_times = np.asarray(gt_times, dtype=np.int64)
    gt_unit = np.asarray(gt_unit)
    if det_index.size == 0 or gt_times.size == 0:
        return 0, 0.0

    n_bins = int(wrap_samples // bin_samples) + 1

    # Busiest units first -- they carry the most correlation per FFT pair,
    # and the cap keeps this to a bounded number of ~1e6-point transforms.
    units, counts = np.unique(det_unit, return_counts=True)
    units = units[np.argsort(counts)[::-1]]

    acc = None
    n_used = 0
    for u in units:
        det_u = det_index[det_unit == u]
        if len(det_u) < min_unit_detections:
            continue
        gt_u = gt_times[gt_unit == u]
        if len(gt_u) < min_unit_detections:
            continue
        a = _binned_counts(det_u % wrap_samples, n_bins, bin_samples)
        b = _binned_counts(gt_u % wrap_samples, n_bins, bin_samples)
        a -= a.mean()
        b -= b.mean()
        denom = a.std() * b.std() * n_bins
        if denom <= 0:
            continue
        # corr[k] = sum_n a[n] * b[n - k], so a peak at k means a(t) matches
        # b(t - k): the detections sit k bins LATER than the ground truth
        # they correspond to, and the offset that carries detections onto
        # ground truth is therefore -k bins.
        corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=n_bins) / denom
        acc = corr if acc is None else acc + corr
        n_used += 1
        if n_used >= max_units:
            break

    if acc is None:
        return 0, 0.0

    k = int(np.argmax(acc))
    peak_z = float((acc[k] - acc.mean()) / (acc.std() + 1e-30))

    coarse = -k * bin_samples
    if refine_halfwidth is None:
        refine_halfwidth = 2 * bin_samples
    offset = refine_offset(det_index, gt_times, wrap_samples, coarse, refine_halfwidth)
    return offset, peak_z


def _gt_hit_mask(gt_times, wrap_samples, tol):
    """Boolean lookup of length wrap_samples: True within `tol` of a spike.

    A dense mask makes the per-lag scoring below a single fancy-index instead
    of a searchsorted per lag, which is what keeps the block-by-block
    tracking affordable (thousands of candidate lags x hundreds of blocks).
    """
    wrap_samples = int(wrap_samples)
    mask = np.zeros(wrap_samples, dtype=bool)
    gt = np.asarray(gt_times, dtype=np.int64) % wrap_samples
    for d in range(-tol, tol + 1):
        mask[(gt + d) % wrap_samples] = True
    return mask


def _score_offsets(det_index, hit_mask, wrap_samples, offsets, chunk=4096):
    """How many detections land on a ground-truth spike, per candidate offset.

    Scored for every candidate offset at once against a chunk of detections
    (one (chunk x n_offsets) gather instead of one pass per offset) -- the
    block tracker below calls this thousands of times on a long run, and the
    per-offset loop it replaces dominated the whole analysis's runtime.
    """
    det = np.asarray(det_index, dtype=np.int64) % int(wrap_samples)
    offsets = np.asarray(offsets, dtype=np.int64)
    scores = np.zeros(len(offsets), dtype=np.int64)
    for s in range(0, len(det), chunk):
        block = det[s:s + chunk, None] + offsets[None, :]
        np.mod(block, int(wrap_samples), out=block)
        scores += hit_mask[block].sum(axis=0)
    return scores


def refine_offset(det_index, gt_times, wrap_samples, center, halfwidth, tol=2):
    """Sample-resolution search for the offset maximizing exact-ish hits."""
    hit_mask = _gt_hit_mask(gt_times, wrap_samples, tol)
    offsets = np.arange(center - halfwidth, center + halfwidth + 1, dtype=np.int64)
    scores = _score_offsets(det_index, hit_mask, wrap_samples, offsets)
    return int(offsets[int(np.argmax(scores))])


MIN_BLOCK_SAMPLES = 3000
MAX_BLOCK_SAMPLES = 300000
TARGET_BLOCK_DETECTIONS = 500


def choose_block_samples(det_index):
    """Block length for track_offset(): ~TARGET_BLOCK_DETECTIONS per block.

    The slip this tracks is a STEP function (the source jumps ~150 samples at
    once, every ~30s), so a block straddling a step is half-misaligned and
    loses those detections -- shorter blocks localize each step and measurably
    improve every unit's F1 (on the development run, unit 74: 0.81 at 100k
    samples/block, 0.99 at 3k). The floor exists because a block also needs
    enough detections to locate the offset at all; below that, blocks start
    locking onto chance peaks and F1 falls again.
    """
    det_index = np.asarray(det_index, dtype=np.int64)
    span = max(1, int(det_index.max()) - int(det_index.min()) + 1)
    if len(det_index) == 0:
        return MAX_BLOCK_SAMPLES
    per_detection = span / float(len(det_index))
    return int(np.clip(TARGET_BLOCK_DETECTIONS * per_detection,
                       MIN_BLOCK_SAMPLES, MAX_BLOCK_SAMPLES))


def track_offset(det_index, gt_times, wrap_samples, initial_offset,
                 block_samples=None, track_halfwidth=400, tol=2,
                 min_block_detections=100):
    """Follow the source's slow slip: re-estimate the offset per block.

    Returns (block_edges, block_offsets) -- `block_edges` has one more entry
    than `block_offsets`, so `block_offsets[searchsorted(block_edges, x,
    "right") - 1]` is the offset in force at stream index x.

    block_samples=None (default) sizes blocks from the detection density via
    choose_block_samples(); see there for why block length matters so much.

    Each block searches only +/-track_halfwidth around the previous block's
    result: the slip is slow and continuous, and a wide free search per block
    would let a low-detection block lock onto a chance peak somewhere else.
    Blocks with too few detections to estimate anything reuse the running
    value rather than guessing.
    """
    det_index = np.asarray(det_index, dtype=np.int64)
    if block_samples is None:
        block_samples = choose_block_samples(det_index)
    lo, hi = int(det_index.min()), int(det_index.max())
    n_blocks = max(1, int(np.ceil((hi - lo + 1) / float(block_samples))))
    edges = lo + np.arange(n_blocks + 1, dtype=np.int64) * int(block_samples)
    edges[-1] = hi + 1

    hit_mask = _gt_hit_mask(gt_times, wrap_samples, tol)
    offsets = np.zeros(n_blocks, dtype=np.int64)
    current = int(initial_offset)
    det_sorted = np.sort(det_index)

    for i in range(n_blocks):
        s, e = np.searchsorted(det_sorted, [edges[i], edges[i + 1]])
        block = det_sorted[s:e]
        if len(block) >= min_block_detections:
            cands = np.arange(current - track_halfwidth,
                              current + track_halfwidth + 1, dtype=np.int64)
            scores = _score_offsets(block, hit_mask, wrap_samples, cands)
            # Require the winner to beat the median candidate by a real
            # margin -- otherwise this block carries no alignment information
            # and the running value is the better estimate.
            if scores.max() > 2 * np.median(scores) + 10:
                current = int(cands[int(np.argmax(scores))])
        offsets[i] = current

    return edges, offsets


def align_detections(det_index, det_unit, gt_times, gt_unit, wrap_samples,
                     bin_samples=30, block_samples=None,
                     track_halfwidth=400, track=True, verbose=True):
    """Map live stream indices to file positions. The entry point callers want.

    Returns a dict with:
      file_index   -- int64[n], each detection's position in the file
      pass_index   -- int64[n], which loop pass through the file it came from
                      (0-based, in run order), for reducing a multi-pass run
                      to one pass
      global_offset, peak_z, block_edges, block_offsets -- diagnostics
      ok           -- False if no trustworthy alignment was found, in which
                      case file_index is the naive `det_index % wrap` and the
                      caller should say so loudly rather than report metrics
                      as if they meant something
    """
    det_index = np.asarray(det_index, dtype=np.int64)
    gt_times = np.asarray(gt_times, dtype=np.int64)
    wrap_samples = int(wrap_samples)

    offset, peak_z = estimate_global_offset(det_index, det_unit, gt_times, gt_unit,
                                            wrap_samples, bin_samples=bin_samples)
    ok = peak_z >= MIN_PEAK_Z
    if verbose:
        verdict = "accepted" if ok else "REJECTED (below z=%.1f)" % MIN_PEAK_Z
        print("Stream->file alignment: global offset %+d samples "
              "(%+.2fs @30kHz), correlation peak z=%.1f -- %s"
              % (offset, offset / 30000.0, peak_z, verdict))
    if not ok:
        return {
            "file_index": det_index % wrap_samples,
            "pass_index": np.zeros(len(det_index), dtype=np.int64),
            "global_offset": 0, "peak_z": peak_z,
            "block_edges": None, "block_offsets": None, "ok": False,
        }

    if block_samples is None:
        block_samples = choose_block_samples(det_index)
    if track:
        edges, block_offsets = track_offset(
            det_index, gt_times, wrap_samples, offset,
            block_samples=block_samples, track_halfwidth=track_halfwidth)
        slot = np.clip(np.searchsorted(edges, det_index, side="right") - 1,
                       0, len(block_offsets) - 1)
        per_det_offset = block_offsets[slot]
        if verbose:
            drift = int(block_offsets.max() - block_offsets.min())
            print("  tracked over %d blocks of %d samples: offset drifted %d "
                  "samples across the run (range %+d to %+d)"
                  % (len(block_offsets), block_samples, drift,
                     block_offsets.min(), block_offsets.max()))
    else:
        edges, block_offsets = None, None
        per_det_offset = np.full(len(det_index), offset, dtype=np.int64)

    absolute = det_index + per_det_offset
    file_index = absolute % wrap_samples

    # Loop passes, labelled in run order. Taken from the ABSOLUTE (unwrapped)
    # position so a run that starts mid-pass still gets contiguous labels,
    # and renumbered from 0 for readability.
    pass_raw = absolute // wrap_samples
    pass_index = pass_raw - pass_raw.min()

    return {
        "file_index": file_index, "pass_index": pass_index,
        "global_offset": offset, "peak_z": peak_z,
        "block_edges": edges, "block_offsets": block_offsets, "ok": True,
    }


def select_widest_pass(file_index, pass_index, forced_pass=-1, verbose=True):
    """Reduce a multi-pass run to one pass. Returns a boolean mask.

    Auto-selects the pass covering the widest slice of the file: a pass the
    run only partially observed would undercount its own ground truth outside
    the observed slice, biasing recall down.
    """
    passes = np.unique(pass_index)
    if len(passes) == 1:
        return np.ones(len(file_index), dtype=bool)

    coverage = {}
    for p in passes:
        m = pass_index == p
        coverage[int(p)] = (int(file_index[m].max() - file_index[m].min()), int(m.sum()))
    chosen = forced_pass if forced_pass >= 0 else max(coverage, key=lambda p: coverage[p][0])
    if verbose:
        print("Run spans %d loop passes (span_samples, n_rows): %s"
              % (len(passes), ", ".join("%d=%s" % (p, v)
                                        for p, v in sorted(coverage.items()))))
        print("  analyzing pass %d only (see stream_alignment.py's docstring "
              "for why passes are not merged)" % chosen)
    return pass_index == chosen
