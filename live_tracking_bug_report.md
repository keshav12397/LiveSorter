# RESOLVED: "live GPU detection fails for sparse/low-amplitude units"

**Status: not a detector bug.** The live all-units GPU pipeline tracks every
unit about as well as calibration predicts. The failure was in the validation
tooling, which mapped live detections onto the wrong part of the replayed
recording. Fixed in `fd1440e` (alignment) with a separate, unrelated
correctness fix in `1cf0985`.

This file replaces the original report, which was never committed. Everything
the original ruled out (interferer selection, channel translation, file
truncation, the packed calibration files, chunk-size-dependent kernel
behavior) it ruled out correctly -- none of those were the problem either. The
one hypothesis it did not test was its own ground-truth alignment.

## What was reported

Live detection tracked high-firing/high-SNR units well but low-firing units at
near-zero recall, despite excellent offline-predicted performance for the same
calibrated filter. Concretely, unit 74: offline held-out f1 = 0.981, live
recall = 0.0026 over a 52-minute run, 0.000 over a 3-minute run. Detection
*counts* were plausible; detection *times* looked unrelated to ground truth.
Live recall correlated strongly with a unit's total spike count (r = 0.77-0.92)
but weakly and with the wrong sign against amplitude (r ~ -0.2) and Kilosort's
`ContamPct` (r ~ -0.3).

## What was actually wrong

Both `FilterGen/validate_all_units.py` and
`FilterGen/analyze_unit_quality_vs_tracking.py` mapped a detection's live
`sample_index` onto a position in the recording with a plain
`% wrap_samples`. That assumes the SpikeGLX stream counter is phase-locked to
the replay's own file read position. It is not, in two separate ways:

1. **Phase.** There is a constant unknown offset between the two. On the
   3-minute run that produced the report it was **7,793,477 samples (~260 s)**.
   Every detection was therefore compared against a completely unrelated
   stretch of ground truth.

2. **Slip.** That offset is not even constant within one pass. The replay
   source stepped it by **-149 samples roughly every 30 s** (its stream counter
   advancing slightly further than its file read position), ~750 samples of
   accumulated slip over 3 minutes. This is why the 52-minute run looked no
   better than the 3-minute one, and why re-running against a "fresh"
   simulation source reproduced the same failure: the problem was never in the
   source's data, and a longer run only accumulates more slip.

### Why that looked exactly like "sparse units are broken"

With the two trains unrelated, a match within the ±100-sample tolerance is
pure coincidence, and coincidence rate scales with the unit's firing rate:

- Unit 99 (~174 Hz): its detection-vs-ground-truth hit count as a function of
  lag is **flat** -- median 30,090 hits across all lags versus 30,096 at lag 0.
  Its apparent 0.86-0.90 "recall" was entirely chance. Median nearest-neighbour
  offset was 47 samples, i.e. about a quarter of its mean ISI, exactly what
  unrelated trains give.
- Unit 74 (~0.5 Hz): ~0 hits at any lag.

Scanning all 157 units for excess hits at lag 0 over the all-lag baseline:
**zero units** exceeded 20% excess recall, and only 2 exceeded 5%. *No* unit
was tracking. The reported "high-firing units track excellently" contrast was
an artefact of the tolerance, not a real difference between units -- which is
also why activity level dominated the correlations and amplitude/isolation did
not.

## The fix

`FilterGen/stream_alignment.py` estimates the stream→file mapping from the data
instead of assuming it. Read its docstring before touching any of this.

- Global offset: per-unit binned circular cross-correlation of each unit's
  detections against that unit's own ground truth, summed across units.
  (Correlating one pooled train against another leaves the true peak at z=4,
  buried; the per-unit sum puts it at z=25.)
- Slip: the offset is re-estimated per block, seeded from and searched near the
  running estimate. Blocks are sized from detection density -- the slip is a
  step function, so a block straddling a step is half-misaligned. Unit 74's f1
  goes 0.81 → 0.99 as block length drops from 100k to ~5k samples.
- A correlation peak that never clears z=8 is a hard error, not a silent
  fall-back to metrics that mean nothing.

Both validation scripts now go through it, and both gained `--no-track-drift`
and (for `validate_all_units.py`) `--loop-pass`.

## Result: the same 3-minute live run, re-scored

| unit | online f1 before | online f1 after | offline-predicted f1 |
|---|---|---|---|
| 74  | 0.005 | **0.986** | 0.981 |
| 50  | 0.002 | **0.914** | 0.893 |
| 90  | 0.002 | **0.994** | 0.993 |
| 141 | -     | **0.990** | 0.984 |
| 99  | 0.86 (chance) | **0.991** | 0.991 |

Across all 157 units:

- online-vs-offline f1 correlation: **0.974**
- median online f1 **0.430** vs offline-predicted 0.382 (online is now
  slightly *better* than the held-out offline prediction, as expected -- the
  live window is a different, longer stretch of data)
- f1 vs Kilosort amplitude: r = -0.2 → **+0.61**
- f1 vs `ContamPct`: r = -0.3 → **-0.34** (right sign now)
- f1 vs total spike count: r = 0.77-0.92 → **+0.29**

The correlation structure is now the physically sensible one: tracking quality
follows amplitude and isolation, not raw event count.

## Two side findings

1. **`minSeparationSamples` (commit `61cc21a`) was a no-op.** Both
   `ConvolutionEngine` and `GpuConvolutionEngine` constructors already map a
   non-positive `minSeparationSamples` to `templateLength / 2`, so the omission
   that commit "fixed" never changed behavior -- which is why re-running after
   it changed nothing. Passing it explicitly is still clearer; the misleading
   comments both fetch threads carried have been corrected.

2. **A real, unrelated UB bug in `GpuConvolutionEngine::processChunk`** (fixed
   in `1cf0985`). The overlap-save carry-forward copied the last `L-1` rows of
   `d_combined_` to the front of the same buffer with one `cudaMemcpyAsync`.
   Source and destination overlap whenever `nSamples < L-1`, and overlapping
   `cudaMemcpy` is undefined -- worse here, since the copy runs as a parallel
   kernel. The live loop hits this constantly: it has no pacing, so on this run
   **31% of its 84,111 chunks were under `L-1` = 60 samples**, 1,059 of them a
   single sample. Nothing tested that regime (the NMS equivalence tests stream
   2000-sample chunks, `OfflineScorer.cu` 2000 or 150). Now staged through a
   dedicated scratch buffer, and `ClosedLoop/test_gpu_chunking_equivalence.cu`
   pins the regime. Honest scope: that test passes against both the old and new
   code on this machine (RTX 4500 Ada, CUDA 11.3) -- that driver tolerates the
   overlap -- so this was latent, not the cause of anything observed.

## For next time

If live tracking metrics ever again show *recall correlating with firing rate
and not with amplitude or isolation*, check the ground-truth alignment before
anything else. That signature is what chance-level matching looks like, and it
is much cheaper to rule out than a detection-pipeline bug. `align_detections()`
reports its own correlation-peak z-score for exactly this purpose.
