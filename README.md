# LCMV LiveSorter

Closed-loop spike detection on a live SpikeGLX session. Matched filters
detect spikes in the IMEC stream, syllable codes are decoded from the NI
stream, and a digital line is raised when the spike count in a window after
a syllable crosses a threshold. Budget: **under 10 ms from detection to
decision**.

Filters are LCMV (linearly-constrained minimum-variance): unity gain on the
target unit's template, exact nulls on nearby interferers, minimum output
variance against the space-time noise covariance. They are designed offline
in Python (`FilterGen/`) from Kilosort output, and loaded by the live C++
app (`ClosedLoop/`).

---

## The binaries

| exe | source | what it does |
|---|---|---|
| `ClosedLoopAllUnits.exe` | `mainAllUnits.cpp` | **the production path.** Detects every qualifying Kilosort unit at once. |
| `ClosedLoop.exe` | `main.cpp` | Single-target path. The original pipeline; kept as a working reference. |
| `OfflineReplay.exe` | `mainOfflineReplay.cpp` | Replays a `.bin` through detection + decision. **Not** a test of the live path. |
| `SpikeViewer.exe` | `Viewer/main.cpp` | Live raster, decisions, and drift trace over a localhost socket. Needs `--live host:port`; without it it opens in CSV mode and never connects. |

`OfflineReplay` has no `sglx_fetch`, no ring buffer, no sample accounting,
and no wall-clock pressure on a fetch loop. Those four things are exactly
what a live run tests and it cannot.

---

## Calibration

**You do not need a pre-built filter bank.** Point a config at a Kilosort
directory and the `.bin` it was sorted from, and `ClosedLoopAllUnits` fits
the bank itself before connecting:

```
filterDir=D:/out/filters          # an OUTPUT: where the fitted bank is written
trainingKsDir=D:/session/ks_out
trainingBinPath=D:/session/rec_g0_t0.imec0.ap.bin
pythonExe=C:/path/to/python.exe
```

`trainingBinPath` must be the **exact** file Kilosort was run on.
Calibration matches ground-truth spike times to raw-file offsets by sample
index, so any mismatch silently produces meaningless thresholds rather than
an error.

Set `skipCalibration=true` to reuse whatever is already in `filterDir` —
which is what you want for a second run against the same session, and what
the drift-aware path needs, since `calibrate_drift_aware.py` also writes
`drift_schedule.bin` and `calibrate_all_units.py` does not. Calibration is
skipped by default only when no `trainingKsDir` is given.

The fit itself is `FilterGen/calibrate_all_units.py`, shelled out to via
`RunProcess.h`. **FilterGen is the only implementation** of "fit a filter"
and "score a threshold" in this branch. A C++ port of it existed and was
removed; `REMOVED.md` records what it cost and what measuring it showed.

Tuning keys — `autoInterferers`, `calibrationTrainFrac`,
`calibrationMaxUnits`, `calibrationWorkers`, `calibrationMinSpikes`,
`calibrationSeed` — all map to that script's own flags; see
`config.example.txt`.

---

## Live architecture

Four threads, each owning exactly one SpikeGLX handle.

```
ImecFetchThreadCpu  --SpikeQueue-->  DecisionThread  --> digital out
        |                             (hot, <10 ms)
        |
        +-----------AnalysisFeed---->  AnalysisThread --> EventPublisher --> SpikeViewer
                                       (slow, >500 ms budget)
```

### Two queues, deliberately

The split exists because drift estimation and plotting need bulk sample
data, and putting that work anywhere near the detection path would spend the
latency budget on it.

| | `SpikeQueue.h` | `AnalysisFeed.h` |
|---|---|---|
| carries | spike times only | preprocessed sample chunks |
| budget | < 10 ms | > 500 ms |
| on overflow | drop-oldest | skip the chunk |
| sharing | no mutex, no state in common | |

`SpikeQueue` is a bounded preallocated ring; the fetch thread pushes a whole
chunk's detections in one batched call, so the mutex is taken once per chunk
rather than once per spike. `drain()` is capped at 4096 elements — an
uncapped drain held the mutex through a 65536-element copy and measured
worse than the deque it replaced.

`AnalysisFeed` hands over the *buffer*, not a copy: `acquire()` takes a free
buffer from a pool, the fetch thread fills it, `publish()` hands back a
pointer. When the pool is dry `acquire()` returns null and the chunk is
simply skipped — that IS the drop policy. There is no allocation to fail and
no waiting. Falling behind costs the analysis side resolution, never the
fetch side time.

**Consumers must `release()` every chunk they `take()`**, including on an
exception path. A leaked buffer is gone from the pool permanently and the
feed silently degrades to dropping everything. `AnalysisThread` uses a scope
guard; `nInFlight()` should return to 0 when the analysis thread is idle.

### Measured

`bench_hotpath.cpp`, PACED regime (10 chunks/s, 150 events/chunk):

```
detection -> decision   p50 30.5us   p99 112.9us   p99.9 179.7us
AnalysisFeed copy cost  mean 1.9us/chunk  (0.04% of a 5 ms chunk budget)
```

Well inside the 10 ms budget. Note this measures the queue handoff interval,
not convolution — `profile_detection_path.cpp` covers the detection compute.

Read PACED numbers only. `bench_queues.cpp`'s FLOOD regime exists solely to
compare overflow policies under overload; reporting its p99.9 as production
latency would be reporting a saturated queue.

### The drop-oldest trade

Under overload `SpikeQueue` drops the oldest entries, and a dropped spike is
a decision that never happens. Blocking instead is a one-line change.
Measured under FLOOD, blocking never drops but moves the cost onto the fetch
thread (push tail 1.4 ms vs 58 µs) and pushes delivered-event latency to
2.5 ms p99.9. A blocked fetch thread also risks a `sglx_fetch` "Too late"
resync against the server's ~8 s ring, which loses a whole chunk rather than
some spikes. Drop-oldest is the default for that reason. Both counters
(`nDropped_`, `maxDepth_`) print at shutdown, so a drop is never silent.

---

## Conventions that must not drift

### Preprocessing must match training

Both calibration and the live fetch thread run input through `Preprocessor`
— high-pass, then common-average reference, in that order — over the **full
CAR channel group**, then subset to the filter's own channels. CAR computed
over a different channel set than training used produces a different signal,
and the threshold calibrated against training becomes meaningless.
`carChannelMapJson` defines the group and must match what
`calibrate_all_units.py` was given.

### Channel indices

`MultiFilterBank.channels` holds **raw SpikeGLX ids** as loaded from disk.
Anything that reads sample data by channel needs them translated to
positions within the CAR group first.

`MultiFilterBank::translateChannelsToCarGroup()` is the single
implementation. Call it once, immediately after `load()` — it is not
idempotent, and calling it twice translates twice.

Two consumers need a translated bank and each translates its own copy: the
fetch thread (which takes the bank by value) and the analysis thread. **This
failure is silent**: a raw probe id is usually still a plausible index, and
one that is out of range looks identical to a genuinely bad channel, which
callers correctly skip. Handing the analysis thread an untranslated bank
produced zero amplitude records on a live run with no error at all.

### Detected-spike sample time

`ConvolutionEngine` computes the same centered correlation
`generate_filter.filter_output()` does (`np.convolve(..., mode='same')`), so
a detection's sample index is directly comparable to Kilosort's
`spike_times.npy` with no correction. The cost is a fixed reporting delay of
`(templateLength-1)/2 + minSeparationSamples` samples — about 2 ms at 30 kHz
— served from already-fetched samples.

Peak selection is **windowed non-max suppression**: a peak at `n` is
accepted only if it is tallest within `[n-minSep, n+minSep]`, matching
`scipy.signal.find_peaks(distance=...)`. Greedy left-to-right selection lets
a small side-lobe just before a real spike claim the exclusion zone and
discard it.

### Cross-stream alignment

Both fetch threads timestamp events as "seconds since MY most recent sync
edge". Comparing across streams is valid while both refer to the same
physical edge — true as long as `windowEndMs` stays well under the 1–3 s
pulse period. Within one stream, compare `sampleIndex`, which is exact and
never wraps.

### Digital line conventions (verify against your wiring)

- **NI DW**: bit *N* = physical line *N*. Sync = line 0; syllable code =
  lines 5/6/7. Set via `niSyncBit` / `niSyllableLines`.
- **IMEC SY**: the sync waveform is **bit 6**, not bit 0 — a fixed
  Neuropixels firmware convention, unrelated to NI numbering. Set via
  `imecSyncBit`, default 6.

### Concurrency rules (do not violate)

`SDK/API`'s `Comm::m_clientMap` is unlocked, process-wide static state.

- Create and connect **every handle sequentially from `main()`**, before any
  thread starts.
- **One handle, one owning thread, forever.** Never call an `sglx_*`
  function on a handle from a thread that does not own it.
- Close and destroy handles sequentially at shutdown, after joining every
  thread.

---

## Drift

A unit that moves relative to the probe drifts off the channels its filter
reads. Three pieces address this; all are measured, and the measurements
live in `FilterGen/DRIFT_AWARE_RESULTS.md` and
`FilterGen/REFIT_RESULTS.md`.

### Estimating motion — `drift_estimate.pooled_com_motion`

Amplitude-weighted centroid per unit per time bin, each unit's own mean
subtracted, then a NaN-safe median across units. Rigid/common-mode by
construction.

Chosen over raster registration (DREDge-style) and monopolar triangulation
after direct comparison: correlation 0.962 vs 0.845 against Kilosort's own
`dshift` on real data, at a fraction of the cost. Triangulation cost 300×
more for no gain.

### Refitting — `banded_refit.py`

R, the space-time noise covariance, is block-Toeplitz: block (i,k) depends
only on the lag between samples. So R for **any** subset of scanned channels
is an index-and-assemble, not a rescan. Scan a *band* around each unit once,
under that unit's own exclusion mask, and every post-drift channel selection
inside that band solves in **2.4 ms** with no data access.

**The exclusion set must stay per unit.** Sharing one covariance across the
probe requires masking every sorted spike, which leaves 99.5% of the
recording spike-present — 1869 usable samples out of 27,000,000. It fails
silently, returning a correctly-shaped array estimated from 0.06 s of data.
`noise_cov_by_lag` now warns below `50 * templateLength`.

Measured, n=134 units on a coherently drifting simulated session:

```
refit vs never refit        mean +0.057 f1    111 better / 20 worse
in-band refit vs rescan     mean -0.000 f1     60 / 60
```

The second line is the point: the cheap refit is indistinguishable from a
full rescan. Broken out by baseline detectability, the marginal band
(f1 0.10–0.30) improves 26/26, mean +0.128.

Caveat: this is a simulator with imposed coherent drift. The pooled shift is
11.05 µm with SD 0.085 µm across all units, so every unit gets the same
correction — which is why the benefit does *not* correlate with per-unit
drift span here, and why that particular claim is untested rather than
refuted.

### Applying a schedule live

`calibrate_drift_aware.py` writes `drift_schedule.bin` alongside the filter
files: one filter-swap event per drifted segment. `ClosedLoopAllUnits` loads
it automatically when present (an absent file is the normal case and loads
as an empty schedule) and applies each swap as its time arrives.

Two things about that are easy to get wrong:

- **Swaps go through `MultiConvolutionEngine::updateUnit()`, never
  `MultiFilterBank::updateFilters()` alone.** The engine widens taps to
  double at construction, so writing the bank alone leaves the copy that
  actually scores untouched — a swap that silently does nothing.
- **Schedule time is stream time**, samples since the first fetch, not wall
  clock. A fetch stall must not slide the schedule relative to the data.

The schedule's `t_s` is "seconds from the calibration recording's start",
mapped onto "seconds since this run's first fetch". That is an **open-loop
plan replayed on a timer**, correct only if the live session begins at the
same point in the drift trajectory the calibration recording did. Closing
the loop — the live tracker feeding measured motion back — is not built.

The shutdown line distinguishes applied, rejected, and never-came-due. A
rejected swap names a channel outside the CAR group and says so.

### Live drift tracking

`AnalysisThread` extracts per-channel peak amplitudes from `AnalysisFeed`
chunks and publishes them as `kWireAmpChannel` records. `Viewer/DriftPool`
reproduces `pooled_com_motion`'s pooling step exactly — verified against a
Python fixture at 0 µm error.

One documented divergence: the viewer bins by 20 s wall clock, the Python
reference by equal spike count. A live tracker cannot know a bin's future
spike count, so the two will not be bit-identical on real data even though
the pooling is.

---

## The wire protocol (v2)

`ClosedLoop/LiveWire.h` is the format, included verbatim by both ends. A
viewer that re-derives the layout from a spec is a second implementation;
this repo has paid for one of those twice.

Localhost TCP. A 32-byte `SessionHeader`, then the preamble, then an
unbounded run of fixed-size 32-byte records. No framing headers — the reader
resynchronises by byte count alone.

**Preamble** (once, after the header): `int32 unitIds[nUnits]`, then
`int32 nChannels[nUnits]`, then a flat unit-major array of
`ChannelGeom{float xUm, yUm}`. Geometry is static per unit, so it is sent
once and never per spike.

**Records**: `kWireSpike`, `kWireSyllable`, `kWireTrial`, `kWireLine`, and
`kWireAmpChannel` (unit id, the spike's sample index as join key, peak
amplitude, channel index into that unit's geometry).

A v1 viewer refuses a v2 stream with an explicit version-mismatch message
rather than misparsing it. Every size is statically asserted at compile time
on both sides.

---

## Building

**`SglxApi.dll` must sit next to the built `.exe`.** The copy in the repo
root is there for that reason. It is byte-identical to `SDK/API/SglxApi.dll`,
so it looks like a redundant duplicate and has been deleted as one at least
once — deleting it makes every build still link fine and then fail at launch
with `0xC0000135` and no message.

MSVC 14.16 (VS2017), `cl.exe`, from PowerShell or a Developer Command
Prompt. `.vscode/tasks.json` has a task per binary. MSYS bash mangles `/I`;
do not build from it.

**Build the all-units binary with `/arch:AVX2`.** `FastMatchedFilter`'s hot
loop is AVX2 intrinsics behind `#if defined(__AVX2__)`; without the flag it
silently falls back to a scalar path that is several times slower and
produces identical results — so nothing fails, it just runs slow.

```
cl.exe /EHsc /std:c++17 /O2 /arch:AVX2 /Fe:ClosedLoopAllUnits.exe /Fo:build\ ^
    ClosedLoop\mainAllUnits.cpp ClosedLoop\Config.cpp ClosedLoop\SglxMetaReader.cpp ^
    ClosedLoop\MultiFilterBank.cpp ClosedLoop\MultiConvolutionEngine.cpp ^
    ClosedLoop\ConvolutionEngine.cpp ClosedLoop\ImecFetchThreadCpu.cpp ^
    ClosedLoop\NiFetchThread.cpp ClosedLoop\DecisionThread.cpp ^
    ClosedLoop\AnalysisThread.cpp ClosedLoop\AmplitudeExtractor.cpp ^
    ClosedLoop\EventPublisher.cpp SDK\API\SglxCppClient.cpp ^
    /I SDK\API /I ClosedLoop /link /LIBPATH:SDK\API libSglxApi.a
```

### Tests

The C++ equivalence tests are not in `tasks.json` — each is a standalone
`main()` built ad hoc. They check the C++ against a Python fixture, which is
the point: two implementations of the same math is how this project has
twice lost recall silently.

```
cl.exe /EHsc /std:c++17 /O2 /Fe:TestLiveWireRoundtrip.exe /Fo:build\ ^
    ClosedLoop\test_livewire_roundtrip.cpp ClosedLoop\EventPublisher.cpp ^
    ClosedLoop\AmplitudeExtractor.cpp ClosedLoop\MultiFilterBank.cpp ^
    Viewer\DriftPool.cpp Viewer\DriftTracker.cpp Viewer\LiveWireClient.cpp ^
    /I ClosedLoop /I Viewer /I SDK\API /link ws2_32.lib
```

**`TestLiveWireRoundtrip.exe` needs a fixture argument** or it skips its
most important check and says so:

```
python FilterGen\gen_drift_fixture.py --out drift_fixture.bin
TestLiveWireRoundtrip.exe drift_fixture.bin
```

Python tests: `python -m pytest FilterGen -v`.

---

## Running

```
ClosedLoopAllUnits.exe path\to\config.txt
```

### Configs

| file | what it is |
|---|---|
| `ClosedLoop/config.example.txt` | the exhaustive reference — every key the pipeline understands, documented inline, including the single-target `ClosedLoop.exe` keys |
| `ClosedLoop/test_config_template.txt` | a runnable template — the minimum set for a live all-units run. Copy it and repoint the paths |

Both are templates. Neither is a config for a real session, and every path
in them is an example.

Two keys are worth knowing before a real experiment:

- **`syllableSource`** — `ni` is production. `imecSy` is a TEST path that
  decodes codes from the IMEC SY word for rigs whose NI stream cannot be
  simulated. It creates no NI handle and no NI thread. The binary announces
  loudly when it is on.
- **`decisionUnitIds`** — which units may drive the decision. "Any of 157
  units fired" is not a decision about anything, so with a bank this size
  the driving units must be named.

### Read the shutdown summary

Sample accounting, queue depth, drop counts, and `AnalysisFeed.nInFlight()`
all print at shutdown. A nonzero in-flight count means a consumer leaked a
pool buffer. Nonzero drops on the hot queue mean decisions were lost.

---

## Where to change things

| change | file |
|---|---|
| fetch cadence / chunk size | `fetchChunkMs` in config |
| digital line assignments | `niSyncBit`, `niSyllableLines`, `imecSyncBit`, `doLine` |
| decision logic | `DecisionThread.cpp`'s `evaluateSyllable()` |
| detection algorithm | `ConvolutionEngine.cpp` — used identically by calibration and live |
| what the analysis thread does | `AnalysisThread::runLoop()` |
| wire format | `LiveWire.h` — both ends recompile or neither does |

## Layout

```
ClosedLoop/   live C++ app, offline calibration, benchmarks, equivalence tests
FilterGen/    Python filter design, drift estimation, calibration, studies
Viewer/       ImGui/ImPlot live viewer
SDK/          SpikeGLX C API
third_party/  imgui, implot (vendored, no package manager)
```

`REMOVED.md` records what was taken out of this branch and why.
