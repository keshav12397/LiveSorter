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
// WHY THIS TAKES A SAMPLE SPAN AND NOT JUST A CHUNK
// -------------------------------------------------
// A detection's sampleIndex refers to a sample EARLIER than the chunk it
// was reported in. ConvolutionEngine uses the centered-correlation
// convention and then needs minSeparationSamples of lookahead before a peak
// can be declared, so a spike surfaces roughly (templateLength-1)/2 +
// minSeparationSamples samples after the sample it names -- about 60 at
// L=61, and its 61-sample window starts templateOffset before that again.
//
// With 5 ms chunks (150 samples at 30 kHz) that window is essentially NEVER
// inside the chunk the spike arrived with. Measured on a live run before
// this was fixed: 49,649 spikes, 0 amplitude records, every one skipped by
// the relStart < 0 test.
//
// So the caller passes a span that covers recent history, not one chunk.
// AnalysisThread keeps a rolling buffer for exactly this. A spike whose
// window still falls outside the span is skipped rather than stitched --
// that is the correct behaviour at a genuine buffer edge, and it now
// discards a small fraction rather than all of them.
// `samples` is [nSamples * nChannels], channel-minor per sample, covering
// absolute indices [firstSampleIndex, firstSampleIndex + nSamples).
// `spikes` need not lie in that span; those that do not are skipped.
std::vector<livewire::WireRecord> extractAmplitudeRecords(
    const float *samples, int nSamples, int nChannels,
    long long firstSampleIndex,
    const std::vector<SpikeEvent> &spikes,
    const MultiFilterBank &filterBank,
    int templateOffset );

// Convenience overload over one chunk's own samples. Correct only when the
// spikes' windows lie inside the chunk, which for LIVE detections they do
// not -- see the note above. Kept for tests and offline callers that place
// spikes inside the span themselves.
std::vector<livewire::WireRecord> extractAmplitudeRecords(
    const AnalysisFeed::Chunk &chunk,
    const MultiFilterBank &filterBank,
    int templateOffset );

#endif // CLOSEDLOOP_AMPLITUDEEXTRACTOR_H
