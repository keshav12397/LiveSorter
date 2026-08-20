# ClosedLoop LiveSorter

Real-time, sub-10ms closed-loop app: detects spikes from a single target
neuron in the live IMEC stream, detects syllable-code events in the live NI
stream, and raises an NI digital line (read by LabVIEW to trigger white
noise) when the spike count in a configurable window after a syllable
exceeds a threshold.

This is the live counterpart to `FilterGen/` (which designs the filter
offline in Python from Kilosort output). `ClosedLoop/` loads that filter and
runs it against a live SpikeGLX session.

## Two phases, one binary

1. **Calibration**: before connecting live, (re)fits the LCMV filter on a
   training `.bin` recording and picks a validated detection threshold
   against that recording's Kilosort ground truth. Set `skipCalibration=true`
   in the config to skip this and just use whatever's already saved in
   `channels_<id>.bin`/`filter_<id>.bin`/`threshold_<id>.bin`.

   **Hard requirement**: `trainingBinPath` must be the *exact* file
   Kilosort was run on (`trainingKsDir`) -- calibration matches ground-truth
   spike times to raw-file byte offsets by sample index, so any mismatch
   between the two silently produces a meaningless threshold.

   **Two backends, `calibrationBackend` in config:**
   - **`python`** (default): `main.cpp` shells out to
     `FilterGen/calibrate_for_closedloop.py`, which fits the LCMV filter on
     the first `calibrationTrainFrac` of the recording, sweeps the threshold
     against the held-out remainder, and writes the *train-split-fitted*
     filter (not one refit on all the data afterward) + swept threshold to
     `filterDir`. `main.cpp` then reloads `FilterBank` from those files. This
     is the single implementation of "fit the filter"/"score a threshold" --
     the same code `FilterGen/generate_filter.py` and
     `FilterGen/threshold_sweep_real.py` use for offline design work.
     Every successful run also saves two diagnostic PNGs next to the `.bin`
     files, so there's no separate step to eyeball a calibration:
     - `template_filter_<id>.png` -- per-channel target mean waveform next
       to the LCMV filter taps fit to it, plus an example held-out-test-split
       detection window with the threshold, detected spikes, and Kilosort
       ground truth all marked, so a channel-selection/filter-shape/threshold
       problem is visible at a glance instead of only showing up as a
       recall/precision number.
     - `threshold_sweep_<id>.png` -- recall/miss-rate, FP-rate, precision-
       recall, and F1, all vs. threshold, with the chosen best-F1 point
       marked (same layout `threshold_sweep_real.py`'s standalone sweeps
       already produce).
   - **`cpp`**: `Calibration.cpp`'s own from-scratch port of the same
     fit+sweep math, kept as a fallback. **Prefer `python`** -- the `cpp`
     path is a second implementation of the same logic that can (and did,
     this session) silently drift out of sync with Python's, costing a long
     debugging session to track down a ~50%-recall-cap bug that turned out
     to be in the C++ port's peer-selection, not in the underlying algorithm.
     Kept around mainly so calibration still works on a machine with no
     Python/numpy/scipy available.

   **Interferers** (nearby units the filter must null out, `python` backend
   only): either list them explicitly (`interferers=67,65,82,...`, Kilosort
   cluster ids) or set `autoInterferers=N` to have the script pick them
   automatically via `generate_filter.py`'s
   `auto_pick_interferers_spatial()`. `interferers`, if non-empty, always
   wins. This is spatially aware, not just activity-ranked: it shortlists
   the ~30 most active non-target/non-noise clusters (cheap), computes each
   shortlisted candidate's own peak channel from a quick mean waveform over
   the already-preprocessed data, and picks the N *physically closest* to
   the target's own peak channel (using the channel-map JSON's `xc`/`yc` if
   given, else index-distance within the channel group) -- not just the N
   busiest clusters anywhere on the probe, which for a sparse target can
   easily be spatially irrelevant. Validated against target 74's real data:
   auto-pick independently reproduced the exact same 5 clusters
   (`[67,65,72,82,57]`) that had been manually curated by inspection earlier
   in this project, just in a different (distance-sorted) order.

2. **Live** (`ImecFetchThread`, `NiFetchThread`, `DecisionThread`): three
   threads, each with its own SpikeGLX connection handle (never share a
   handle across threads -- see "Concurrency rules" below):
   - `ImecFetchThread` continuously fetches the **full CAR channel group**
     (see "Preprocessing must match training" below) + the IMEC SY (sync)
     channel in `fetchChunkMs` chunks, runs highpass+CAR across that full
     group, subsets down to the filter's own channels, runs the shared
     `ConvolutionEngine`, and reports spikes (as "seconds since the SY
     channel's most recent sync-pulse edge") to `DecisionThread` and to
     `spikeTimesPath`.
   - `NiFetchThread` continuously fetches the NI digital-word (DW) channel,
     decodes a debounced 3-bit syllable code from lines 5/6/7, tracks the
     sync pulse on line 0, and reports syllable events (also as "seconds
     since that stream's most recent sync edge") to `DecisionThread`, plus
     writes every emitted event's code and NI-stream sample index to
     `syllableTimesPath` (one `code,sampleIndex` line per event -- same
     idea as `ImecFetchThread`'s `spikeTimesPath`, just CSV since there are
     two fields instead of one). Leave `syllableTimesPath` blank to skip
     writing this file (events still reach `DecisionThread` either way).
   - `DecisionThread` owns a third, dedicated handle purely for issuing
     `sglx_ni_DO_set` calls, counts recent spikes in the configured window
     after each syllable event, and raises/lowers the digital line.

## Preprocessing must match training (CAR channel group especially)

Both `Calibration` and `ImecFetchThread` run the filter's input through
`Preprocessor` (high-pass, then common-average-reference, same order as
`FilterGen/generate_filter.py`) before convolving. **CAR must be computed
across the same full channel group the filter was trained with** (e.g. all
96 shank channels via `carChannelMapJson`, matching generate_filter.py's
`--channel-map-json`) -- **not** just the filter's own 5 channels. Computing
CAR over too few channels doesn't reject the shared noise the filter's
statistics (channel selection, LCMV weights, threshold) were calibrated
against.

This was found the expensive way: an earlier version fetched only the
filter's 5 channels live, applied no CAR/highpass at all, and produced over
a million spurious "detections" in about a minute of live data (the
detector was saturating at the minimum re-trigger separation, not finding
real sparse spikes). Both `Calibration` and `ImecFetchThread` now fetch/read
the full CAR group and only subset down to the filter's channels *after*
preprocessing -- see either file's channel-index bookkeeping
(`filterIndexWithinCarGroup`) for exactly how.

The high-pass filter itself is a necessary approximation: `generate_filter.py`
by default uses `scipy.signal.filtfilt` (zero-phase, needs samples from both
before *and* after each point -- impossible live), while `ButterworthHighpass`
uses a standard causal biquad (same cutoff/order, RBJ Audio-EQ-Cookbook
formula, Q=1/sqrt(2)) instead. Same frequency response shape, different phase.

**This phase difference is not just cosmetic -- it measurably degrades a
filter trained on `filtfilt`-preprocessed data.** With CAR+highpass both
correctly implemented as above, live/calibration recall still capped around
45-63% (vs. Python's ~97%) regardless of threshold. The fix is to **train the
filter itself on the same causal biquad the live pipeline uses**, not to try
to make the live pipeline retroactively match `filtfilt` (impossible in true
real time). `generate_filter.py` and `threshold_sweep_real.py` both take a
`--causal-highpass` flag that swaps in `highpass_causal_biquad()` -- a
coefficient-for-coefficient port of `ButterworthHighpass.h`, applied causally
via `scipy.signal.lfilter` instead of `filtfilt` -- so the LCMV weights,
channel selection, and threshold are all calibrated against exactly the
signal statistics the live C++ code will actually see.
`calibrate_for_closedloop.py` (the default `calibrationBackend=python` path,
see above) always uses this causal biquad unconditionally -- there's no flag
to turn it off there, since a filter trained any other way is only valid for
offline/Python analysis, never for deployment against ClosedLoop. Confirmed
to close the gap: causal-biquad-trained filter for target 74 recovered to
recall=97.21%/precision=99.21% in Python's own held-out sweep, and
recall=98.1%/precision=99.1% in the full C++ calibration pipeline against
the same data.

## Cross-stream time alignment

Per the SpikeGLX manual, the shared sync pulse is a square wave with a
1-3 second period, and *"the time coordinate of any event can be referenced
to the nearest pulser edge... sub-millisecond accuracy"*. Both fetch threads
independently timestamp their own events as "seconds since MY most recent
edge" -- comparing those numbers directly across streams is valid as long as
both threads' "most recent edge" refers to the same physical pulse edge,
which holds as long as the analysis window (`windowEndMs`) stays well under
the pulse period. This is safe for the spike-count windows this app is
designed for (tens to a few hundred ms), but is a known simplification: a
monotonic edge-index/counter passed alongside each event would make this
robust even at a window comparable to the pulse period, and isn't currently
implemented since it isn't needed here.

## Digital line / bit conventions (IMPORTANT -- verify against your wiring)

- **NI DW channel**: bit *N* = physical line *N* (confirmed by the SpikeGLX
  manual: "lowest numbered lines... in the lowest order bits"). Sync = line
  0 = bit 0; syllable code = lines 5/6/7 = bits 5/6/7. Configurable via
  `niSyncBit` / `niSyllableLines`.
- **IMEC SY channel**: the sync waveform is **bit 6**, not bit 0 (manual:
  *"Status bit #6 is the sync waveform, the other bits are error flags"*) --
  this is a fixed IMEC/Neuropixels firmware convention, unrelated to the NI
  bit numbering above. Configurable via `imecSyncBit`, defaulting to 6.

## Detected-spike sample-time convention

`ConvolutionEngine` computes exactly the same centered correlation
FilterGen/generate_filter.py's `filter_output()` does (`np.convolve(...,
mode='same')`), so a detection's reported sample index is directly
comparable to Kilosort's own `spike_times.npy` -- no correction needed.
This costs a small, fixed reporting delay: a detection at `n` can't be
finalized until `rightMargin` future samples (`(templateLength-1)/2`) have
arrived to compute D at `n`, *plus* `minSeparationSamples` more so the
windowed peak decision (see below) has enough future score values to compare
against -- ~2ms total at 30kHz for a 61-tap filter's defaults, trivial
against the 10ms budget. That's handled by the existing history/score
buffers, reusing already-fetched samples instead of re-requesting
overlapping data from the server. (An earlier version of this file used a
causal-only, no-lookahead formula plus a hand-derived index correction
instead -- computed that correction wrong, and it silently cratered
Calibration's recall/precision against ground truth. Matching Python's
formula exactly instead of re-deriving an equivalent one removes that whole
class of bug.)

**Peak selection is proper windowed non-max suppression, not greedy
left-to-right.** A detection at `n` is accepted only if its score is the
tallest within `[n-minSeparationSamples, n+minSeparationSamples]` --
matching `scipy.signal.find_peaks(distance=...)`'s actual semantics. An
earlier version instead accepted the first strict local maximum that was
simply "far enough past the *last accepted* peak" -- which let a small
spurious side-lobe a few samples *before* a real spike's true, taller peak
claim the exclusion zone first and silently discard the real detection. This
was diagnosed from calibration recall capping around 45-63% *even at the
loosest possible threshold* (accepting virtually every candidate score) --
a pattern only explainable by genuine spikes never reaching the candidate
list at all, not by a scoring/thresholding problem. Fixing this (plus the
causal-highpass retrain above) brought calibration to
recall=98.1%/precision=99.1%/F1=0.986 against the real training set,
matching Python's own held-out benchmark.

## Concurrency rules (do not violate)

`SDK/API`'s `Comm::m_clientMap`/`m_nextHandle` are unlocked, process-wide
static state (see `SDK/CPP/TestDualHandle.cpp` for the pattern this app
follows). This means:

- **Create and connect every handle sequentially**, from `main()`, before
  any thread starts fetching. Never call `sglx_createHandle`/`sglx_connect`
  concurrently.
- **One handle, one owning thread, always.** `ImecFetchThread` only ever
  touches `hIM`, `NiFetchThread` only `hNI`, `DecisionThread` only `hDO`.
  Never call any `sglx_*` function on a handle from a thread that doesn't
  own it.
- Close/destroy all handles sequentially again at shutdown, after joining
  every thread.

## All-units detection

A second, separate executable, `ClosedLoopAllUnits.exe` (`mainAllUnits.cpp`),
detects **every** Kilosort unit in a session at once, rather than one
hand-picked target. Several source comments point at this section; this is
it.

### There is no GPU any more, and the arithmetic says there never needed to be

This pipeline was CUDA. It is not, and the reason is worth recording because
the instinct to reach for a GPU here is strong and wrong.

**We never convolve 384 channels.** LCMV selects a handful per unit --
`filters.bin` is `(nUnits, 61, 5)` -- so the kernel's inner loop runs over 5
or 6 channels, not the probe. 384 appears only in preprocessing, which is
~130 MFLOP/s of highpass and median CAR. The whole 157-unit bank is:

    157 units x 5 channels x 61 taps = 47,885 MAC/sample
                          at 30 kHz  =  2.87 GFLOP/s

Measured GPU cost was 3.20 us/sample, i.e. ~30 GFLOP/s achieved against
~19 TFLOP/s of sm_80 peak: **0.15% utilisation**. That is not a compute
workload, it is overhead with some arithmetic attached. `nmsDecideKernel` was
the proof -- 157 blocks of ONE thread each, with ~106 KB of `MAX_CAND` local
arrays living in device DRAM, which is the single most GPU-hostile shape a
kernel can have.

Do not reintroduce a GPU without re-measuring first. The crossover is real
but distant: the bank has to grow by roughly 30x in `nUnits * nChannels`
before the FLOPs alone justify one.

### What it costs on CPU

`bench_multi_convolution.exe` measures realtime factor (seconds of neural
data per second of wall clock) rather than GFLOP/s, because on this workload
the two disagree and only one decides whether a session keeps up:

              units  chans   1 thread   12 threads
                157      5      2.76x       12.65x
                500      6      0.65x        3.60x
               1000      6      0.35x        2.89x
               2000      6      0.21x        1.48x

Preprocessing (384 channels, highpass + median CAR, single-threaded) is a
separate ~3.3x, and scales with channel count rather than unit count.

`profile_detection_path.exe` says where the time goes, and is worth running
before optimising anything here -- three successive guesses at the bottleneck
were wrong before it was written. What it found: `std::fmaf` is not lowered
to a `vfmadd` by this compiler, so a 6-output scalar tail cost more than the
whole vectorised body; and MSVC sizes `std::deque` blocks in ELEMENTS (2 for
`double`, 1 for a 16-byte pair), so the decision path was doing a malloc
every couple of samples per unit. Fixing those took 1000 units from 0.73x to
2.89x.

It is deliberately a separate binary and not a mode switch inside
`ClosedLoop.exe`. The single-target path is the validated production one and
stays untouched by work on this.

**It is detection-only.** No `NiFetchThread`, no `DecisionThread`, no digital
output, no calibration subprocess. Detections go to `spikeTimesPath` as
`unit_id,sample_index,score` rows, where `unit_id` is the real Kilosort
cluster id (via `MultiFilterBank::hostUnitIds`), not an internal array index.

### Pipeline

1. `FilterGen/calibrate_all_units.py`, run offline ahead of time, fits an
   LCMV filter and sweeps a threshold for every qualifying unit and packs
   them into four flat, fixed-stride files (`unit_ids.bin`, `channels.bin`,
   `filters.bin`, `thresholds.bin`) plus `summary.csv`. It reuses
   `generate_filter.py`/`threshold_sweep_real.py`'s functions directly --
   this repo has twice lost recall to a second, drifted reimplementation of
   that math. There is also a C++ equivalent, `CalibrateAllUnits.exe`
   (`mainCalibrateAllUnits.cpp`), for the same job without Python.
2. `MultiFilterBank::load()` reads all four arrays once at startup. Every
   unit shares one `nChannelsPerUnit` and one `templateLength`; that fixed
   stride is what lets every consumer index without a per-unit lookup.
3. `ImecFetchThreadCpu` owns one handle (same one-handle-one-thread rule as
   everything else here), fetches the full CAR group + SY channel, and hands
   each chunk to `Preprocessor` (highpass + CAR) then
   `MultiConvolutionEngine`, which scores every unit across a worker pool.

### MultiConvolutionEngine does not contain the algorithm

It owns **one `ConvolutionEngine` per unit** and calls it. `ConvolutionEngine`
records that this repo has twice silently lost recall to a
re-derived-and-wrong version of its matched-filter/NMS math, so the safe
option was taken: not "port it faithfully" but "call the original". What the
class adds is the channel gather, the threshold comparison, and scheduling.

The convolution itself is split out into `FastMatchedFilter` (samples -> D);
`ConvolutionEngine::processPrecomputedD` still does D -> peaks, so the
decision rule stays shared and untouched. `processChunk()` calls that same
method, so there is exactly one copy of it.

**`FastMatchedFilter` accumulates in float32, and that is a correctness
choice before it is a speed one.** `calibrate_all_units.py` picks each
threshold by scoring float32 filters against float32 data, and
`generate_filter.filter_output()` says explicitly that the dtype must follow
the input so a threshold is not "calibrated for a distribution the [online]
path never actually produces". `ConvolutionEngine` is float64 because it
serves the SINGLE-TARGET path, whose Python side writes float64 filters --
a separate, internally consistent convention. Reusing it wholesale silently
moved the all-units path off the precision its thresholds were chosen in.

The float64 path is kept behind a constructor flag as the reference the
float32 one is tested against. On 22,609 peaks they agree on every peak,
with max relative score difference 1.47e-06 -- the predicted
sqrt(L*nCh)*epsilon bound.

**Determinism.** `processChunk()` sorts by `(sampleIndex, unitIndex)`, so
output does not depend on worker count. Without that, a test would pass or
fail by machine core count and no run would be reproducible from its config.

### Sample accounting -- read the shutdown summary

The fetch loop tracks every sample the server produced and prints a summary
at shutdown asserting `span == processed + dropped`. Check it. A run that
lost data says so there and nowhere else.

Two ways samples go missing, both counted:

- **Gap.** `sglx_fetch` returns `headCt`, the index of the first sample it
  actually gives you, which is not necessarily what you asked for. If your
  request has aged out of the ring, the server silently starts you later.
- **Resync.** If the request is too old to serve at all, `sglx_fetch` fails
  ("Too late") and the loop must jump forward into a live part of the ring.

**The server ring is only ~8.0 s deep** on both streams (SpikeGLX
v20251218, measured). Falling that far behind is unrecoverable by
construction, so watch `max backlog` in the summary long before it gets
near. In practice there is enormous headroom: a 157-unit run against the
live server processed 30.05 s of stream in a 30 s window with zero drops and
a max backlog of 76 samples (2.5 ms).

`latencyLogPath` gives per-chunk detection-pipeline timing. The GPU build's
"ignore chunk 0, it reads ~200 ms" caveat is gone with the GPU -- that was
each kernel's first launch loading its module and reserving local memory.
There is no module to load, so chunk 0 is an ordinary chunk and every row of
that log now means the same thing.

### Validating a run

`FilterGen/validate_all_units.py` scores detections against Kilosort ground
truth and summarizes the latency log.

**It must go through `FilterGen/stream_alignment.py`, and you should read
that module's docstring before touching any of this.** A replayed stream's
sample counter is not phase-locked to the replayed file's read position, and
the offset slips *within* a single pass. Assuming otherwise once produced a
result that looked exactly like a broken detector for sparse units and was
entirely a broken comparison -- see `live_tracking_bug_report.md`. If live
metrics ever again show recall correlating with firing rate but not with
amplitude or isolation, that is the signature of chance-level matching:
check alignment first, it is much cheaper to rule out than a pipeline bug.

### Testing without the rig

`FilterGen/make_sim_session.py` generates a replayable synthetic session
(384 ch + SY, ~30 min, ~160 units) with exactly known spike times, syllable
onsets, and **drift trajectories**. Point SpikeGLX's IMEC simulation source
at its `.bin` to exercise the live path, or feed the `.bin` and its
`sim_ks/` directly to `calibrate_all_units.py` for the offline path -- where,
having no stream, there is no alignment problem at all.

Syllable codes ride on the IMEC SY channel's bits 0-2 rather than the NI
digital word, because the NI stream cannot be simulated on this setup. See
that script's docstring for why that is also the better test article.

## Live viewer (SpikeViewer.exe)

`ClosedLoopAllUnits.exe` is no longer detection-only. It runs the same three
threads `ClosedLoop.exe` does -- `ImecFetchThreadCpu` + `NiFetchThread` +
`DecisionThread` -- and publishes every event over a local TCP socket to
`SpikeViewer.exe`, a Dear ImGui + ImPlot viewer (vendored in
`third_party/`, no package manager, no network fetch at build time).

It keeps writing every CSV it wrote before. The socket is for watching a run
happen; the CSVs are what a finished run is read from.

### The wire

One shared header, `ClosedLoop/LiveWire.h`, defines a 32-byte
`SessionHeader` (magic/version/nUnits/sample rates/which syllable source)
followed by the unit-id table, then a stream of 32-byte `WireRecord`s of
four types: spike, syllable, trial, and DO-line transition. Both ends
include that one header, so neither re-derives the format.

**Publishing never blocks the fetch loop.** `EventPublisher` buffers into a
ring and hands off to its own sender thread; if the viewer stalls, the ring
drops oldest-first and *counts* the drops, and the fetch loop is unaffected.
With no viewer attached, records are discarded rather than buffered -- a
viewer joining mid-run should see the run from the moment it joined, not a
burst of stale history. Measured at 0.141 us/record with a deliberately
wedged reader.

### Running it

```
SpikeViewer.exe --live [host:port]                  attach to a running session
SpikeViewer.exe --csv <spikeTimes> [syllables] [decisions]    replay a finished run
SpikeViewer.exe --unit <kilosort_id> --rate <Hz> --quit-after <seconds>
```
With no arguments it starts idle and you connect or open files from the
Session panel; the switches exist so a run can be launched from a script.

Views: a scrolling per-unit raster, rasters and PSTHs aligned to the digital
trigger over a configurable window, syllable events with their codes on the
same axis, and the DO line's state both live and as a per-trial attribute so
trials can be split into triggered and not-triggered.

### syllableSource -- READ THIS BEFORE A REAL EXPERIMENT

`syllableSource=ni` (the default) is the production path: syllable codes
come from the NI digital word's lines 5/6/7, exactly as before.

`syllableSource=imecSy` is a **test-only** path that decodes codes from the
IMEC SY word's bits 0-2 instead. It exists because a SpikeGLX simulation
source can replay a synthetic IMEC file but cannot simulate an NI stream, so
on a test rig the NI digital word is all zeros and the production path
cannot be exercised at all. `FilterGen/make_sim_session.py` puts codes
there; on real hardware those bits are error flags that stay 0, so nothing
changes. Both binaries print a loud banner at startup saying which is
active.

**`syllableSource=ni` has never been exercised end to end on this codebase**
-- no rig with a live NI digital word has been available. Everything
downstream of the decoder (windowing, counting, the DO line, the viewer) is
shared between both paths and is tested, but the NI decode itself is not.
Confirm it on real hardware before trusting a closed-loop session.

### decisionUnitIds

`DecisionThread` used to count *any* spike, which is correct for one
hand-picked target and meaningless across 157 units. `decisionUnitIds` names
which Kilosort cluster ids drive the decision. Empty means "count
everything", which preserves `ClosedLoop.exe`'s behaviour exactly; the
all-units binary **requires** the key rather than defaulting it, since a
default would be a silent choice of which neuron controls the stimulus.

## Drift-aware filters

Neurons move relative to the probe over a recording. A unit's peak channel
migrates and its peak amplitude changes, so a filter fit from a
whole-session mean waveform is fit to a *smeared* template and
underperforms. `FilterGen/calibrate_drift_aware.py` is the drift-aware
counterpart to `calibrate_all_units.py`.

It does not reimplement any fit or sweep math -- it calls
`threshold_sweep_real.fit_lcmv` and `sweep_thresholds` like every other
driver here, for the reason those modules state about themselves.

### RESULT: coherent drift costs 0.22 f1 at one channel of displacement

Measured on a matched pair of 1800 s / 160-unit sessions generated from the
same seed -- identical units, spikes and amplitudes, differing ONLY in
probe-wide motion (30 um ramp, 0.3 nonrigid gain) -- and calibrated with a
CHRONOLOGICAL half/half split, which is the calibrate-once-then-deploy
geometry that actually motivates drift-aware filters.

Restricted to the 61 units detectable at all in the static arm (the
population is bimodal; averaging in units that fail in BOTH arms only
dilutes the effect):

                    static    drift     delta
        recall      0.7929    0.5955   -0.1974
        precision   0.5678    0.3588   -0.2090
        f1          0.6207    0.3963   -0.2244

85% of those units get worse. Note that train->test displacement here is
only ~15 um -- ONE channel -- because a half/half split on a linear ramp
separates the two halves' mean positions by half the total span, not the
full 30 um.

**Recall and precision fall together, by almost exactly the same amount.**
That matters for what a fix has to do: a purely mis-set threshold trades one
against the other, so this is not a thresholding problem. The filter itself
is mismatched -- the template is an average over positions the unit no
longer occupies, and `select_channels` picked for that average.

There is no dose-response visible within this session (correlation between
displacement and f1 change is -0.04) simply because displacement barely
varies across units: the nonrigid gain spreads it only over 13-17 um. This
session tests one dose, and tests it well.

### Why the earlier measurement said the opposite

The subsection below reports a drift penalty of 0.022 and concludes the
drift modes do not help. Both of its premises were wrong, and the difference
between 0.022 and 0.224 is the whole story:

1. **The drift was independent per unit.** Each drifting unit got its own
   direction, magnitude, shape and timing, so units 30 um apart moved
   opposite ways. Real drift is common-mode. Beyond making pooled estimators
   inapplicable, this also let a filter's interferers stay put while its
   target moved, which is a much gentler problem than the whole
   neighbourhood sliding together.
2. **The train/test split was interleaved.** `calibrate_drift_aware.py`
   trains on the first part of every cell, so the control's training data
   spans the entire trajectory. That measures interpolation. Nothing there
   ever had to extrapolate to a position it had not seen.

The detection-lag finding below is unaffected by either point -- it is a bug
in how detections were matched to ground truth, not a claim about drift --
and it stands.

### The earlier (superseded) measurement

The measurement in this subsection was made on a session whose drift model
was wrong in a way that limits what the numbers can mean, and it is being
re-run. Read it with both of these in mind:

1. **The drift was independent per unit.** `make_sim_session.py` gave each
   drifting unit its own direction, magnitude, shape and timing, so units
   30 um apart moved opposite ways. Real drift is common-mode -- the tissue
   slides and carries the population with it. A session built the old way
   cannot evaluate any pooled estimator at all (see "Pooled motion
   estimation" below), and it also understates how *correlated* a real
   drift penalty is across units.
2. **The train/test split was interleaved.** `calibrate_drift_aware.py`
   cuts the session into cells and trains on the first part of every cell,
   so the control's training data spans the whole trajectory. That measures
   interpolation. The case that motivates drift-aware filters is
   extrapolation: calibrate once, then run for hours while the unit walks
   away. `calibrate_all_units.py`'s chronological split is the right
   geometry for that question.

The detection-lag finding below is unaffected by either point -- it is a
bug in how detections were matched to ground truth, not a claim about
drift -- and it stands.

### RESULT (on the model described in the scope warning above)

Measured on a 30 minute / 160 unit synthetic session with known drift, all
modes run under one protocol so nothing but the mode differs, amplitude
>= 20, n=94 paired units:

                        calib_all*   global  segmented  registered
    static  (n=67)           0.520    0.529      0.530       0.531
    drifting(n=27)           0.377    0.508      0.501       0.484
      ramp  (n=11)           0.467    0.587      0.578       0.569
      osc   (n= 5)           0.578    0.597      0.542       0.539
      jump  (n=11)           0.194    0.388      0.405       0.374

    drift penalty            0.144    0.022      0.030       0.047
    (static - drifting)

    * calibrate_all_units.py, BEFORE the detection-lag fix below.

Against the matched `global` control, segmentation is **-0.007** on drifting
units and registration **-0.024**. Neither helps. What closed the gap was
the detection-lag fix alone: **+0.131** on drifting units, collapsing the
drift penalty from 0.144 to 0.022.

That is not a coincidence of this dataset. A drifting unit's template
changes shape as it moves, so the LCMV filter's alignment with it varies,
so the lag varies -- and a varying lag crosses the +/-15 sample matching
window far more often than a fixed one. The bug therefore punished drifting
units specifically, which looks exactly like a drift penalty and is not one.

**So: use the detection-lag fix.** The verdict on the drift modes
themselves is now OPEN, not negative: they were only ever measured against
independent per-unit drift under an interpolation split, and on coherent
drift under a chronological split there is a 0.22 f1 penalty for them to go
after. They should be re-measured on the paired sessions above, with
`dredge_lite`'s pooled trajectory rather than the per-unit one that reports
up to 71 um of motion on a session with none. The remaining 0.022 penalty is close to the noise between
subgroups. The segmented/registered code is kept because a negative result
is only meaningful alongside the code that produced it, and because faster
drift or sparser units than this session contains might still favour it --
but it should not be turned on without re-measuring against `--mode global`
on the actual data.

The one subgroup segmentation helps is `jump` (0.388 -> 0.405), the abrupt
step. It hurts `osc` (0.597 -> 0.542), where a reversible excursion returns
to where it started so the whole-session template is already near the
time-average position and segmenting mostly adds fit noise.

### Two modes

- **`--mode segmented`** cuts the session into segments per unit (from an
  estimated drift trajectory) and fits each segment's filter from that
  segment's own spikes. Sharp templates, but each segment gets roughly
  1/K of the unit's spikes -- which is why `--min-spikes-per-segment`
  exists. Segmenting too finely produces a noisier template than a mildly
  smeared one.
- **`--mode registered`** keeps every spike instead: it registers the
  unit's whole trajectory onto each segment's position before averaging
  (`motion_correct.py`), so the template is sharp at that position while
  still drawing on the full spike count.
- **`--mode global`** is the control -- one filter per unit, the ordinary
  fit, run under this script's own train/test protocol so a comparison
  isolates segmentation from everything else. **Use this, not
  `calibrate_all_units.py`'s output, as the baseline for any drift
  comparison** (see "Comparing against a baseline" below).

`--mode both` / `--mode all` run several in one pass over the data.

Neither drift mode beat plain `--mode global` on the data tested -- see
the result above before using either.

### The trajectory, and channel-id distance

Drift is estimated per unit by binning its spikes, taking each bin's mean
waveform, and tracking the waveform's centre of mass along the probe's
depth axis. `--smooth` applies a running *median*, not a mean: an abrupt
drift step is a step function, and a mean smears it across the window --
which is exactly the boundary the segmenter has to find.

**Channel-id distance is not depth distance.** A real channel map stitches
together non-contiguous banks: `shank1only.json`'s `chanMap` steps by +1
for most of its length and by +49 twice. Two channels 15 um apart can have
SpikeGLX ids 49 apart, so a unit crossing a seam appears to jump ~50
channels while physically moving one row. Anything treating drift as a
distance must use microns or a depth-ordered index, never the raw id.

### Detection lag -- a real bug this work uncovered

The LCMV filter's response to its own target **does not peak at the spike
time**. `generate_filter.detection_lag()` derives the offset from
`filter_output()`'s own convention rather than measuring it. For a plain
matched filter the lag is the fixed `L//2 - template_offset` the
"Detected-spike sample-time convention" section describes, but LCMV's
filter is the template whitened by R^-1 and constrained to null its
neighbours, and that is not generally aligned with the template. Measured
across twelve units the extra term alone ranged -24..+8 samples, for a
total lag of -14..+18.

That matters because validation matches detections to ground truth within
+/- `template_length//4` (15 samples at L=61). A unit whose lag exceeds the
window scores near-zero recall at *every* threshold while detecting
perfectly -- and worse, the best-F1 search then picks a threshold off that
meaningless curve. `calibrate_all_units.py` now subtracts the lag, putting
detections back on Kilosort's own sample convention, and reports it as a
`detection_lag` column in `summary.csv`.

This affects the ordinary single-target and all-units calibration paths,
not just drift work. **Filter banks calibrated before this fix may carry
thresholds chosen from a corrupted sweep for high-lag units.** Recalibrate.

### Comparing against a baseline

Use `--mode global` from this same script. Comparing a drift mode against
`calibrate_all_units.py`'s output instead confounds segmentation with the
detection-lag fix above, which lands in both -- doing exactly that once
inflated segmentation's apparent benefit several-fold.

Two further cautions, both learned the expensive way:

- **Do not evaluate a segmentation method on a short recording.** On a
  300 s session, segments are starved of spikes and segmentation looks
  nearly worthless; the effect is much larger over 30 minutes. Duration is
  a confound, not a detail.
- **Do not band results by `n_train_spikes` across sessions of different
  length.** The same spike count means a different firing rate at a
  different duration.

### Deploying a drift schedule online

`calibrate_drift_aware.py` writes `drift_schedule.bin` alongside the usual
packed filter files: a time-ordered list of (time, unit, new
channels/filter/threshold) events. `ClosedLoop/DriftSchedule.h` loads it and
hands back events as their time arrives, and
`GpuFilterBank::updateFilters()` overwrites one unit's slice in place --
no reallocation, since the bank's dimensions are fixed at `load()`.

**This is not wired into `ImecFetchThreadCpu` yet.** `DriftSchedule.h`
documents the call-site addition, including why the channel-id translation
must reuse that file's existing raw-id -> CAR-group mapping rather than
duplicating it.

## Pooled motion estimation (dredge_lite.py)

`FilterGen/dredge_lite.py` estimates probe-wide motion by decentralized
registration -- a small version of DREDge (Windolf et al.,
https://github.com/evarol/dredge), specialised to being downstream of a
sorter.

### Why pooling, rather than a better per-unit tracker

`drift_estimate.unit_trajectory` follows each unit's own amplitude centroid.
A unit firing at 1.5 Hz puts ~90 spikes in a 60 s bin, and the centroid of a
90-spike average wanders several microns on noise alone -- against a drift
signal of ~30 um over a session. Smoothing does not fix that; it trades the
noise for the lag that makes an abrupt step unfindable.

Pooling does fix it, because **drift is common-mode**. The tissue slides
along the shank and takes the whole population with it, so N units are not
N problems but N noisy votes on one trajectory, worth about sqrt(N).

"Decentralized" specifically means no reference bin is privileged: every
*pair* of time bins is cross-correlated to give a displacement D_ij and a
confidence w_ij, and one trajectory is then solved for that best explains
all pairs at once. A pair that could not be compared reliably gets a low
weight instead of corrupting the answer, which is what a
measure-everything-against-bin-0 scheme cannot do once the probe has moved
far enough that bin 0 no longer overlaps the present.

### What being downstream of a sorter buys

DREDge's AP mode must *localize* each individual spike before it can build
its raster. We already know which spikes belong to which unit, so a unit's
spikes can be averaged within a time bin *before* anything is measured. The
raster here is built from per-unit per-bin mean-waveform amplitude profiles
instead: the same raster, far better SNR per entry, and no localizer to
tune. The full per-channel profile is kept rather than collapsed to a
centroid, because the profile's ~30 um *shape* is what cross-correlation
registers.

Omitted relative to real DREDge: the online/streaming solver, robust
(L1/Huber) reweighting, and its GPU paths. `--min-corr` is the only
robustness mechanism and it is blunt.

### What it cannot do

It recovers the common-mode trajectory and nothing else. A unit genuinely
moving relative to its neighbours is not in the raster's shared structure.
`sim_truth.npz` therefore stores `probe_offset_um` separately from
`drift_position_um`, so a pooled estimator is scored on the quantity it
models rather than charged for one it does not.

### Simulating coherent drift

`make_sim_session.py --probe-drift-um 30` gives every unit the same ramp --
the ~2-channel settling of an ordinary day-long recording. `--probe-jumps`
adds abrupt probe-wide steps on top (kept separate from the ramp because
slow settling and steps break different things: a slow tracker follows the
ramp fine and lags the step badly), and `--probe-nonrigid-gain` makes the
motion depth-dependent, which is the first-order term of any smooth
nonrigid field. `--n-drifting` still exists and now means per-unit motion
*relative to the tissue*, layered on top.

With no `--probe-drift-um` the generator is bit-for-bit what it was, so
older sessions remain reproducible.

**A geometry note that is easy to get wrong.** A 30 um ramp does not give a
30 um train/test displacement under a half/half chronological split. The
train half's mean position and the test half's mean position differ by half
the total span -- 15 um, one channel. Ask for twice the drift you want to
test, or compare against a late window rather than the test mean.

### Two bugs the tests caught

`test_dredge_lite.py` runs against synthetic rasters with known
trajectories, so a failure there is a bug in the algebra rather than a
statement about data. Both of these produced plausible-looking wrong
trajectories rather than obvious breakage:

- **The sign of `b` in the normal equations.** `D` is antisymmetric, so both
  gradient sums collapse to the same `+D_kj` form and the term lands on the
  opposite side from where it looks like it should; the solve returned
  `-p`. The uniform-weight closed form `p_k = mean_j D_jk` is asserted
  directly for exactly this reason.
- **Correlation normalization under shift.** Normalizing each column once
  and then zeroing whatever a shift pushes off the probe end leaves the
  denominator sized for the full column while the numerator has lost rows,
  penalising large shifts in proportion to their size and shrinking every
  displacement toward zero. Scoring each shift on its overlapping range
  only improved every test at once (rigid 30 um ramp 0.76 -> 0.04 um;
  narrowest nonrigid window 4.36 -> 0.59 um). The bias is invisible on a
  long probe with small drift and obvious on a narrow nonrigid window.

### RESULT: pooling is 3x more accurate, and the static control explains
### why the drift modes never helped

1800 s, 160 units, 30 um coherent ramp plus 0.3 nonrigid gain, against a
matched static arm generated from the same seed (identical spikes, units and
amplitudes -- only the probe motion differs). Error is rms against the known
trajectory, in microns:

                              dredge_lite   per-unit tracker
    drift arm  median              1.05           3.03
               90th pct            1.41          21.01
               max                 1.64          56.10
    static arm median              0.00           2.68
               90th pct            0.00          22.51
               max                 0.00          71.34

    estimated motion span: 29.8 um (true 30.0) on the drift arm,
                            0.0 um (true  0.0) on the static arm

**The static arm is the important column.** On a recording with exactly zero
drift, per-unit tracking reports up to 71 um of apparent motion, and 36 of
160 units -- 22% -- exceed the 12 um `tol_um` at which
`segment_from_trajectory` decides a unit is drifting and cuts it into
segments.

That is very likely the whole story behind "the drift modes do not help".
`--mode segmented` was splitting a fifth of the *stationary* population on
pure measurement noise, starving each of those units' segments of spikes for
no reason at all, and the harm there cancelled whatever the genuinely
drifting units gained. The problem was never that drift does not matter. It
was that the trajectory being segmented on was noise-dominated, so the
method spent most of its effort on units that were not moving.

`dredge_lite` reports 0.000 um on the static arm -- not "small", identically
zero, because every pair's correlation peaks at zero shift -- so it has no
false-positive rate to trade away.

### Rigid or nonrigid?

Use rigid unless the nonrigidity is large. On this session the true
depth-dependent spread is 8.1 um out of 34.0 um of total motion, and the
4-window nonrigid fit is slightly *worse* than the rigid one (median 1.12 vs
1.05 um) while over-reporting the motion span (51.9 vs 34.0 um true): each
window sees a quarter of the units, and that costs more than the depth
dependence it recovers. Nonrigid earns its keep when motion actually differs
across the probe by more than the per-window noise, which is a thing to
measure, not assume.

### Running it

`python dredge_lite.py --ks-dir ... --bin-path ... --channel-map-json ...
--truth sim_truth.npz` scores this estimator and
`drift_estimate.unit_trajectory` on the same recording, against both truths.
`--n-windows 1` is rigid; more is nonrigid.

## Building

**`SglxApi.dll` must sit next to the built `.exe`.** The copy in the repo
root is there for exactly that reason -- both build tasks emit their exe
into the repo root, and Windows resolves the DLL from the executable's own
directory. It is byte-identical to `SDK/API/SglxApi.dll`, so it looks like a
redundant duplicate and has been deleted as one at least once; deleting it
makes every build still link fine and then fail at launch with exit code
0xC0000135 (STATUS_DLL_NOT_FOUND) and no message.


`.vscode/tasks.json` has a `"build closed loop"` task, mirroring the
existing `"build demo"` task's `cl.exe` invocation. Run via VS Code's
"Tasks: Run Task", or from a Developer Command Prompt:

Every binary now builds with `cl.exe`; nothing uses `nvcc`. `.vscode/tasks.json`
has a task per executable ("build closed loop", "build closed loop all units",
"build calibrate all units", "build offline replay", "build spike viewer").

Build the all-units binary with `/arch:AVX2` -- `FastMatchedFilter`'s hot loop
is AVX2 intrinsics behind `#if defined(__AVX2__)`, and without the flag it
silently falls back to the scalar path, which is several times slower but
produces identical results.

```
cl.exe /EHsc /std:c++17 /Zi /Fe:ClosedLoop.exe ^
    ClosedLoop\main.cpp ClosedLoop\Config.cpp ClosedLoop\SglxMetaReader.cpp ^
    ClosedLoop\FilterBank.cpp ClosedLoop\ConvolutionEngine.cpp ^
    ClosedLoop\Calibration.cpp ClosedLoop\ImecFetchThread.cpp ^
    ClosedLoop\NiFetchThread.cpp ClosedLoop\DecisionThread.cpp ^
    ClosedLoop\EventPublisher.cpp ^
    SDK\API\SglxCppClient.cpp ^
    /I SDK\API /I ClosedLoop ^
    /link /LIBPATH:SDK\API libSglxApi.a
```

## Running

```
ClosedLoop.exe path\to\config.txt
```

See `config.example.txt` for every setting, with inline comments.

## Modularity -- what to touch for common changes

- **Fetch cadence / chunk size**: `fetchChunkMs` in config, used by both
  fetch threads.
- **Digital line/bit assignments**: `niSyncBit`, `niSyllableLines`,
  `imecSyncBit`, `doLine` in config -- no code changes needed.
- **Decision logic** (window, count threshold, what "counts" as a trigger):
  entirely in `DecisionThread.cpp`'s `evaluateSyllable()` -- swap in
  different logic here without touching either fetch thread.
- **Fetching mechanism itself** (e.g. switch to `sglx_fetchLatest`, change
  channel subsets): `ImecFetchThread.cpp` / `NiFetchThread.cpp` each have a
  single `fetchLoop()` method containing the entire fetch cycle.
- **Convolution/detection algorithm**: `ConvolutionEngine.cpp` is used
  identically by calibration and live fetch -- change it once, both phases
  pick it up.
