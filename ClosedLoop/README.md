# ClosedLoop

Real-time, sub-10ms closed-loop app: detects spikes from a single target
neuron in the live IMEC stream, detects syllable-code events in the live NI
stream, and raises an NI digital line (read by LabVIEW to trigger white
noise) when the spike count in a configurable window after a syllable
exceeds a threshold.

This is the live counterpart to `FilterGen/` (which designs the filter
offline in Python from Kilosort output). `ClosedLoop/` loads that filter and
runs it against a live SpikeGLX session.

## Two phases, one binary

1. **Calibration** (`Calibration.h/.cpp`): before connecting live, streams a
   training `.bin` recording through the *same* `ConvolutionEngine` that
   will run live, in the same `fetchChunkMs`-sized chunks, and sweeps
   detection thresholds against that recording's Kilosort ground truth to
   pick a validated operating threshold (best-F1 by default). Set
   `skipCalibration=true` in the config to skip this and just use the
   threshold already saved in `threshold_<id>.bin`.

   **Hard requirement**: `trainingBinPath` must be the *exact* file
   Kilosort was run on (`trainingKsDir`) -- calibration matches ground-truth
   spike times to raw-file byte offsets by sample index, so any mismatch
   between the two silently produces a meaningless threshold.

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
     since that stream's most recent sync edge") to `DecisionThread`.
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
uses `scipy.signal.filtfilt` (zero-phase, needs samples from both before
*and* after each point -- impossible live), so `ButterworthHighpass` uses a
standard causal biquad (same cutoff/order, RBJ Audio-EQ-Cookbook formula)
instead. Same frequency response shape, different phase -- acceptable for
matching the *scale* of noise the filter was calibrated against, not meant
to be sample-for-sample identical to Python's offline result.

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
This costs a small, fixed reporting delay (`(templateLength-1)/2` samples,
~1ms at 30kHz for a 61-tap filter -- trivial against the 10ms budget): a
detection can't be finalized until that many *future* samples have arrived
in a later chunk. That's handled by the existing history buffer, reusing
already-fetched samples instead of re-requesting overlapping data from the
server. (An earlier version of this file used a causal-only, no-lookahead
formula plus a hand-derived index correction instead -- computed that
correction wrong, and it silently cratered Calibration's recall/precision
against ground truth. Matching Python's formula exactly instead of
re-deriving an equivalent one removes that whole class of bug.)

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
