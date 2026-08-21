# What was removed on `all_units_cleaned`, and why

Companion to `README.md`. This branch was cut from `all_units_cpu` after the
drift-aware work merged, and stripped to what the system actually needs.
This file records everything taken out and the reasoning, so the reasoning
survives even though the text left the code.

Nothing here was deleted because it was merely old. Each removal is either
**actively false** (it describes behaviour the code does not have) or
**dangling** (it names a file, phase, or implementation that does not
exist). Content that was superseded but still true was moved, not dropped.

---

## 1. References to the CUDA implementation

The pipeline ran on GPU until commit `0c42c48` removed CUDA entirely. The
`.cu` files went; the comments describing them did not. Removed in three
categories.

### 1a. Actively false — comments asserting GPU behaviour

| where | said | reality |
|---|---|---|
| `mainAllUnits.cpp:2` | "all-Kilosort-units, **GPU-accelerated** spike detection" | CPU, AVX2 |
| `mainCalibrateAllUnits.cpp:2` | "C++/**GPU** batch calibration driver" | CPU |
| `mainCalibrateAllUnits.cpp:10,14` | phases named `NoiseCovariance.cu`, `OfflineScorer.cu` | those files do not exist |
| `mainOfflineReplay.cpp:29` | "a single batched **GPU** pass over the whole file" | a single batched pass |
| `NoiseCovariance.h:29` | "the full train split must fit in **GPU memory**" | ordinary RAM |
| `NoiseCovariance.h:49` | "`dData`: **device** float32[...]" | a host pointer |
| `test_offline_scorer_equivalence.cpp` | compares "the **GPU** output" | compares C++ against Python |

**The worst one was in `config.example.txt`**, the file users copy from:

> `# Optional per-chunk GPU timing log ... IGNORE CHUNK 0 -- it reads ~200ms`
> `# against ~0.2ms for every chunk after it, because the first launch of`
> `# each kernel loads its module and reserves its local memory.`

This is operational advice about a warm-up cost that cannot occur. There is
no module to load; chunk 0 is an ordinary chunk. Anyone following it would
have discarded a real measurement. Replaced with a statement that every row
means the same thing.

### 1b. Comparative history — "the GPU version did X, we do Y because Z"

About a dozen of these, in `ImecFetchThreadCpu`, `MultiConvolutionEngine`,
`NoiseCovariance`, `OfflineScorer`, `MultiFilterBank`, `FastMatchedFilter`,
and `ThresholdSweep`.

These were not false, and the *reason* in each was worth keeping. What was
removed is the requirement that the reader know a deleted CUDA
implementation in order to parse the sentence. Each was rewritten to state
the reason directly. For example:

- *before*: "Ordinary heap memory: the pinned allocation the GPU version
  used existed only to speed up the host-to-device leg, and there is no such
  leg now."
- *after*: "Sized once from maxChunkSamples and reused, so the fetch loop
  itself never allocates."

The surviving fact is the one that matters — the fetch loop does not
allocate. Why some other implementation allocated differently is not
load-bearing.

**One GPU mention was deliberately kept**, in `mainAllUnits.cpp`: a pointer
to `FastMatchedFilter.h`'s arithmetic showing why a GPU was never needed
here. That is a live claim about the current design, not history.

### 1c. Dangling names

`GpuFilterBank` → `MultiFilterBank`, `GpuPreprocessor` → `Preprocessor`,
`OfflinePreprocessor.cu` → `OfflinePreprocessor`,
`test_gpu_chunking_equivalence.cu` → a description of the regime it tested.
These appeared in Python docstrings too (`calibrate_all_units.py`,
`calibrate_drift_aware.py`, `validate_all_units.py`), where they named the
C++ struct that reads the files those scripts write — so the name was wrong
in the one place a reader would go looking.

---

## 2. "The branch plan's Phase N"

`mainCalibrateAllUnits.cpp`, `OfflinePreprocessor.h`, and `NoiseCovariance.h`
referred to phases 1–5 of a planning document that is not in this repository
and never was. The numbering carried no information to anyone reading the
code.

Section markers were kept as *structure* — they usefully divide a long
`main()` — but renamed to say what each section does:
`// ---- Phase 2: batched noise covariance` → `// ---- Batched noise
covariance`.

---

## 3. `dredge_lite.py` — deleted earlier, recorded here

Deleted in `43517bb`, before this branch. Recorded because the reason is
not obvious from the deletion and is worth not rediscovering.

It was a DREDge-style decentralized raster registration estimator. It was
replaced by `drift_estimate.pooled_com_motion` (amplitude-weighted centroid,
per-unit mean subtracted, median pooled) on measured grounds:

```
                    corr vs Kilosort dshift (real)    rmse (simulated)
pooled centroid            0.962                          0.10 um
raster registration        0.845                          1.03 um
```

The centroid is also far cheaper, which is what makes live tracking viable.

**The bug worth remembering**: `dredge_lite` returned identically zero on
real data. Its raster was a comb — 15 µm channel rows scattered onto a 5 µm
depth grid leave two of every three bins empty, so cross-correlation
collapsed to a delta at zero shift. Every existing test missed it because
the simulated drift was 15 µm — exactly one channel row, exactly three bins
— so the comb realigned with itself perfectly. A test at 8 µm on a 15 µm
pitch fails on the pre-fix code at 65.5% empty bins.

The lesson generalises past this file: a test whose synthetic parameters are
commensurate with the structure under test can validate a completely broken
implementation.

Monopolar triangulation was also implemented and rejected — slope 0.585 vs
0.578 for the centroid, at roughly 300× the cost.

---

## 4. Probe-wide shared noise covariance — never shipped, documented so it
   stays unshipped

Not a removal from the code but a removal from the design space, recorded
because it was re-proposed and re-derived after already having been settled.

The idea: scan every channel once, share one covariance array across all
units. `noise_covariance_from_lags` can assemble R for any subset, so it
looks free.

It does not work, and the repo said so before it was raised again —
`NoiseCovariance.h` and `generate_filter.py`'s single-target fit both carry
the note. Re-measured on the 160-unit session with every sorted spike
excluded:

```
99.53% of the train half is spike-present
13 usable gaps, 1869 samples total  (vs 27,000,000 for per-unit exclusion)
```

Usable data accrues at ~2.1 samples per second of recording. The failure is
**silent**: `noise_cov_by_lag` raises only when zero gaps survive, and 13 is
not zero, so it returns a correctly-shaped `(121, 96, 96)` array estimated
from 0.06 s of data.

The real reason is about relevance, not density: "not noise" should mean
*could contaminate these channels*, not *any spike anywhere on the probe*.

`generate_filter.noise_cov_by_lag` now warns below `50 * templateLength`.
The design that does work is per-unit exclusion with cheapness coming from
channels instead — `banded_refit.py`.

Also recorded because it produced a wrong intermediate answer: **coverage
cannot be estimated as (spike count × blanking width)**. That arithmetic
reported 5.2× oversubscribed where true union coverage is 99.5%. The
intervals overlap heavily at 1570 spikes/s. Measure the union.

---

## 5. The four test_config files

Removed: `test_config.txt`, `test_config_fixed_threshold.txt`,
`test_config_offline_replay.txt`, `test_config_all_units_live_verify.txt`.
Replaced by one runnable template, `test_config_template.txt`.

They were per-machine scratch configs with hard-coded `D:/` paths from
specific past sessions, not templates anyone could copy. Two
(`test_config.txt`, `test_config_fixed_threshold.txt`) were single-target
configs carrying `targetId=74`; the other two were all-units. None of them
were referenced by any code, task, or test.

`config.example.txt` remains the exhaustive key-by-key reference.
`test_config_template.txt` is the minimum runnable all-units set.

### Findings rescued from their comments

`test_config_all_units_live_verify.txt` recorded a real verification run in
its header, and two things in it are worth keeping.

**The live path holds up with a viewer attached.** Two 30 s runs against the
real `ImecFetchThreadCpu`/`sglx_fetch` path:

    run 1 (no viewer)     span 900175 samples, processed 900175, 0 dropped,
                          0 fetch errors, max backlog 62 samples (2.07 ms)
    run 2 (viewer live)   span 900340, processed 900340, 0 dropped,
                          0 fetch errors, max backlog 306 samples (10.2 ms)
                          viewer socket: 98011 records published, 0 dropped

Backlog rises with a viewer attached but stays nowhere near the ~8 s server
ring, and nothing drops. This is also the record that the viewer's live leg
DOES work when launched as `SpikeViewer.exe --live 127.0.0.1:4143` -- worth
knowing, because a viewer started without `--live` opens in CSV mode and
never connects, which looks identical to a broken socket.

**An unresolved DecisionThread throughput ceiling.** In run 2, 215,719
detections in 30 s (~7.2k/s across 160 units) drove `spike_count` to 0-1
despite `spikeCountThreshold=3`. The same ceiling had been seen in
offline-replay findings, and this run confirmed it is real on the live path
rather than an offline-tool artifact. **Still open** -- it is a property of
the decision stage, not of the queue split measured later, and nothing in
the drift or amplitude work touched it.

---

## 6. The C++ calibration path

Removed entirely. It survives on `all_units_cpu` if it is ever needed.

    CalibrateAllUnits.exe        mainCalibrateAllUnits.cpp
    its exclusive dependencies   KilosortReader, LcmvFit, NoiseCovariance,
                                 OfflinePreprocessor, ThresholdSweep,
                                 ScratchMemmap.h, DenseLinAlg.h
    single-target C++ backend    Calibration.cpp/.h, and main.cpp's
                                 calibrationBackend=cpp branch
    all-units C++ backend        mainAllUnits.cpp's calibrationBackend=cpp
    its equivalence tests        test_lcmv_equivalence,
                                 test_noise_covariance_equivalence,
                                 test_dense_linalg_equivalence
    their fixture generators     gen_lcmv_fixture.py,
                                 gen_noise_covariance_fixture.py,
                                 gen_dense_linalg_fixture.py
    the backend comparison       compare_calibration_backends.py
    the build task               "build calibrate all units"

**`OfflineScorer` and `NpyReader` were NOT removed** — they look like part of
this cluster but `OfflineReplay.exe` links them, so
`test_offline_scorer_equivalence.cpp` and `gen_offline_scorer_fixture.py`
stay too. The deletion set was derived by walking the include graph from
every surviving entry point rather than by eye, precisely because that
distinction is not obvious from the filenames.

`FilterGen/test_noise_covariance_equivalence.py` also stays. Despite the
name it compares Python against Python (`noise_covariance_vectorized` vs the
original `noise_covariance`) and has nothing to do with the C++ port.

### Why, given it was a working, tested binary

It was measured against the Python on 4 units of the simulated session,
immediately after being wired in as `calibrationBackend=cpp`:

    channel selection identical on   1 of 4 units
    unit 51 (channels DID match)     tap corr 0.758, rel error 0.66
    f1                               mean -0.013, max |delta| 0.026

    backend   runtime    peak private   peak working set
    python     394.4 s       3.82 GB          61.03 GB
    cpp        387.1 s       3.99 GB           9.56 GB

The two backends select the same units and then diverge: different channels
on three of four, and on the one unit whose channel sets agree the taps
correlate at 0.758 — a different filter, not float noise.

All six equivalence tests passed throughout. They pin the LCMV solve, the
noise covariance and the scorer; the divergence lives in what they do not
cover — candidate selection, interferer picking, RNG draws. **Component
equivalence is not pipeline equivalence**, and that gap is the entire
lesson here.

And it bought nothing to offset that: runtime within noise, private memory
near-identical, both dominated by the same ~20.7 GB preprocessing pass. (The
working-set gap is *not* a saving — it is shared pages of the memory-mapped
scratch file. Reading a working set as memory used produced a wrong
conclusion earlier in this project.)

So the C++ path was a second implementation of shared math that measurably
disagreed with the reference, cost the same to run, and had to be kept in
sync forever. `Calibration.cpp` had already done exactly this once before:
its peer-selection bug capped recall near 50% and cost a long debugging
session that ended in the port, not the algorithm.

The one thing lost is the ability to calibrate on a machine with no
Python/numpy/scipy. Nothing in this project needs that today, and
`all_units_cpu` still has it.

---

## 7. README narrative moved here

The previous README was 852 lines, of which roughly half was investigation
history: sections titled "Why the earlier measurement said the opposite",
"The earlier (superseded) measurement", and "What this replaced, and why the
comparison is recorded".

That material is real and was not discarded — it lives in
`FilterGen/DRIFT_AWARE_RESULTS.md` and `FilterGen/REFIT_RESULTS.md`, which
are the right place for it. The README now describes the system as it is and
points at those documents for how it got there.

Two specific corrections that were embedded in the old narrative and are
worth carrying forward:

- **Coherent drift costs 0.22 f1** at one channel of displacement. An
  earlier measurement said drift cost nothing; that null was an artifact
  (`ab7fb82`).
- **The 285.9 s "one-time scan" figure was wrong by ~100×.** It was measured
  on a 400k-sample fixture and reported as if it were the full recording.
  Real cost is 11.8 h at 96 channels, 14 min at 16, 2 min at 5 (`17f3950`).

---

## What was explicitly kept

Decisions made deliberately, so they are not revisited as oversights:

- **The single-target pipeline** (`ClosedLoop.exe`, `main.cpp`,
  `FilterBank`, `Calibration`, `ImecFetchThread`). Superseded by the
  all-units path — running it with one unit does the same job — but kept as
  a working reference implementation of the original >0.9 f1 single-neuron
  result.
- **`DriftSchedule.cpp/h`**. Nothing includes it and it is in no build task,
  which is statically indistinguishable from dead code. It is not dead: it
  is pending work, sequenced behind the wire-protocol change.
- **`live_tracking_bug_report.md`**, the superseded sections of
  `DRIFT_AWARE_RESULTS.md`, `make_synthetic_test.py`, and
  `stream_alignment.py` — all kept by explicit decision.
- **`analyze_unit_quality_vs_tracking.py`**. It analyses what predicts
  tracking quality, but was run against simulated data, where amplitude is a
  parameter the simulator chose rather than a property of a neuron. Its
  correlations (r=0.64 for amplitude) describe the generator, not biology.
  Kept, flagged here instead of deleted.
- **The equivalence tests and benchmarks**, none of which are in
  `tasks.json`. They are standalone `main()`s built ad hoc. The README now
  documents how to build and run them, which is what was actually missing.
- **`bench_multi_convolution.cpp`**. Briefly deleted during this pass on the
  grounds that nothing includes it and it is in no build task. That was
  wrong twice over: `profile_detection_path.cpp` cites it by name, and it
  produced the unit-scaling numbers behind `ae88d75` (1000 units, 0.73x ->
  2.89x realtime). Restored. "In no build task" is not evidence of death in
  this repo — no test or benchmark is.

## What is still unfinished

- Closing the drift loop. `DriftSchedule` IS now wired into
  `ImecFetchThreadCpu` and verified live (4 of 4 swaps applied), but it
  replays a precomputed plan on a timer. The live tracker's measured motion
  does not feed back into which filters are used.
- The drift work is validated on a simulator with imposed coherent drift.
  The real comparison — `D:/catgt_Lav69_d1.0_g0` against Kilosort's own
  `dshift` — has not been run.
- The viewer's own drift plot. Amplitude extraction is verified live
  (116,223 spikes -> 581,115 records, exactly 5 per spike, 0 out of span),
  and `DriftPool` matches the Python reference exactly on a fixture, but
  nobody has watched the trace render from live records end to end.
- A **DecisionThread throughput ceiling**, recorded above: at ~7.2k
  detections/s the spike count saturates at 0-1 regardless of
  `spikeCountThreshold`. Confirmed on both the offline and live paths.
  Untouched by the queue, drift, and amplitude work.
