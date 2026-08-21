#ifndef CLOSEDLOOP_AMPLITUDEEXTRACTOR_H
#define CLOSEDLOOP_AMPLITUDEEXTRACTOR_H

#include <vector>

#include "LiveWire.h"
#include "AnalysisFeed.h"
#include "MultiFilterBank.h"

// Slow-path amplitude extraction for the version-2 wire protocol (see
// LiveWire.h's kWireAmpChannel). This is intentionally a pure function
// with no thread, no queue, and no socket of its own: it is meant to be
// called from whatever eventually drains AnalysisFeed (a future analysis
// thread), NOT from the fetch loop -- amplitude extraction is exactly the
// per-channel work AnalysisFeed exists to keep off ImecFetchThreadCpu's
// <10 ms budget.
//
// NOT WIRED IN YET. ImecFetchThreadCpu.cpp and DecisionThread.cpp are
// owned by a separate change in flight right now; connecting AnalysisFeed's
// consumer side (and therefore this function's call site) is left for
// after that lands, to avoid two changes editing the same fetch loop at
// once. This function is complete and unit-testable on its own in the
// meantime -- see test_livewire_roundtrip.cpp's "amplitude extraction from
// a synthetic chunk" case.
//
// What it computes
// -----------------
// For every SpikeEvent in chunk.spikes, and every channel in that spike's
// unit's channel list (filterBank.unitChannels(u), the SAME CAR-group
// indices MultiConvolutionEngine reads chunk.samples with -- see
// ImecFetchThreadCpu.cpp's translation step), the peak absolute sample
// value over the spike's template window:
//
//   [spikeSample - templateOffset, spikeSample - templateOffset + templateLength)
//
// the same window convention LcmvFit.h's meanWaveform() and
// generate_filter.py's mean_waveform() use. This is PEAK amplitude of one
// spike's own waveform, not peak-to-peak of a many-spike mean -- a single
// spike doesn't have the second one to reference. drift_estimate.py's
// _position_from_waveform works from a mean waveform's ptp because it
// operates on an already-binned average; here that averaging happens on
// the GUI side, over many of these per-spike peak-amplitude records (see
// Viewer/DriftTracker.h), which is the intended division of labour: this
// function stays a single pass per (spike, channel) with no state.
//
// A spike whose window does not fit entirely inside `chunk` (too close to
// either edge) is silently skipped -- there is no cross-chunk stitching
// here, matching AnalysisFeed's whole "falling behind costs resolution,
// never correctness" stance. At ~3000 samples/chunk and a ~61-sample
// window this discards a small fraction of spikes near chunk boundaries,
// never all of them, and never anything on the detection path (SpikeQueue
// still gets every spike; only AnalysisFeed's copy is windowed here).
std::vector<livewire::WireRecord> extractAmplitudeRecords(
    const AnalysisFeed::Chunk &chunk,
    const MultiFilterBank &filterBank,
    int templateOffset );

#endif // CLOSEDLOOP_AMPLITUDEEXTRACTOR_H
