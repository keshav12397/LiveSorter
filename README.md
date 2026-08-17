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

## Building

`.vscode/tasks.json` has a `"build closed loop"` task, mirroring the
existing `"build demo"` task's `cl.exe` invocation. Run via VS Code's
"Tasks: Run Task", or from a Developer Command Prompt:

```
cl.exe /EHsc /std:c++17 /Zi /Fe:ClosedLoop.exe ^
    ClosedLoop\main.cpp ClosedLoop\Config.cpp ClosedLoop\SglxMetaReader.cpp ^
    ClosedLoop\FilterBank.cpp ClosedLoop\ConvolutionEngine.cpp ^
    ClosedLoop\Calibration.cpp ClosedLoop\ImecFetchThread.cpp ^
    ClosedLoop\NiFetchThread.cpp ClosedLoop\DecisionThread.cpp ^
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
