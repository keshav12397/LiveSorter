#ifndef CLOSEDLOOP_MULTICONVOLUTIONENGINE_H
#define CLOSEDLOOP_MULTICONVOLUTIONENGINE_H

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>

#include "MultiFilterBank.h"
#include "ConvolutionEngine.h"
#include "FastMatchedFilter.h"

// One accepted detection. unitIndex is a position into MultiFilterBank's
// arrays (0..nUnits-1) -- NOT the Kilosort cluster id; translate via
// MultiFilterBank::hostUnitIds[unitIndex]. Same sampleIndex convention as
// ConvolutionEngine.h's PeakEvent (the "same"-mode centered formula,
// directly comparable to spike_times.npy).
//
// Field-for-field identical to the GpuPeakEvent it replaces, so consumers
// (the fetch thread, the offline scorer, the CSV writer) did not change.
struct MultiPeakEvent {
    int       unitIndex;
    long long sampleIndex;
    float     score;
};

// Runs the matched filter + windowed non-max suppression for every unit in a
// MultiFilterBank, across a pool of worker threads. The CPU replacement for
// GpuConvolutionEngine.
//
// Why this is a thin shell and not an algorithm
// ---------------------------------------------
// It owns ONE ConvolutionEngine PER UNIT and calls it, unmodified. None of
// the matched-filter or NMS math is reimplemented here, and that is the
// single most important design decision in this file. ConvolutionEngine.cpp
// records that this codebase has twice silently lost detection
// recall/precision to a re-derived-and-wrong version of that math -- once
// from a hand-derived causal index shift with a miscomputed correction
// constant. GpuConvolutionEngine was itself careful to be a port rather than
// a re-derivation for exactly that reason. Porting back to the CPU makes the
// safe option available again: not "port it faithfully" but "call the
// original".
//
// The consequence worth stating plainly: this engine's output is bit-exact
// with the single-unit CPU path by construction, because it *is* the
// single-unit CPU path, run N times. There is no equivalence to maintain.
//
// What this class actually adds is therefore only three things:
//   1. the channel gather -- slicing each unit's nChannelsPerUnit columns
//      out of the full CAR-group chunk (what the GPU kernel did into shared
//      memory),
//   2. the threshold comparison, and
//   3. thread-pool scheduling across units.
//
// Why a persistent pool
// ---------------------
// At the live default (fetchChunkMs=5) processChunk() is called 200 times a
// second. Spawning threads per chunk would cost more than the work: the
// whole bank at 157 units x 5 channels x 61 taps is ~2.9 GFLOP/s, a fraction
// of one core. Workers are therefore created once and parked on a condition
// variable between chunks.
//
// Units are claimed from a shared atomic counter rather than being dealt out
// in fixed stripes. Per-unit cost is not uniform -- the NMS decision chain
// does more work for a unit that is actually firing -- so a static split
// leaves the last worker holding the noisy units while the rest idle.
//
// Determinism
// -----------
// processChunk()'s returned vector is sorted by (sampleIndex, unitIndex), so
// the output does NOT depend on how many workers ran or the order they
// finished in. This matters more than it looks: without it, an equivalence
// test would pass or fail depending on machine core count, and a run would
// not be reproducible from its config alone.
class MultiConvolutionEngine {
public:
    // nChannelsGroup: size of the full CAR/preprocessed channel group this
    // engine reads from (e.g. 96) -- NOT nChannelsPerUnit.
    // bank.channels must already be translated to indices WITHIN that group
    // (see MultiFilterBank.h) before this is constructed.
    // minSeparationSamples: forwarded to every per-unit ConvolutionEngine; 0
    // means that class's own templateLength/2 default.
    // nThreads: 0 means hardware_concurrency(), clamped to at least 1 and to
    // at most nUnits (more workers than units cannot help).
    //
    // fastPath: which convolution computes D.
    //   true  (default) FastMatchedFilter -- float32, vectorised over output
    //         samples. This is the PRODUCTION path, and float32 is the point
    //         of it as much as the speed: calibrate_all_units.py picks
    //         thresholds by scoring float32 filters against float32 data, so
    //         this is the precision those thresholds were chosen in.
    //   false ConvolutionEngine's own float64 convolution. Kept as the exact
    //         reference the fast path is tested against, and as the answer to
    //         "is this difference the fast path or the data?" -- a question
    //         worth being able to settle in one flag rather than one rebuild.
    // Peak DECISIONS are identical either way: both feed the same
    // ConvolutionEngine, only the D values differ, and only in float32
    // rounding.
    MultiConvolutionEngine( const MultiFilterBank &bank, int nChannelsGroup,
                            long long minSeparationSamples = 0, int nThreads = 0,
                            bool fastPath = true );
    ~MultiConvolutionEngine();

    MultiConvolutionEngine( const MultiConvolutionEngine & ) = delete;
    MultiConvolutionEngine &operator=( const MultiConvolutionEngine & ) = delete;

    // data: nSamples * nChannelsGroup PREPROCESSED doubles, time-major /
    // channel-minor (data[t*nChannelsGroup+ch]) -- exactly what
    // Preprocessor::processChunk() returns.
    // streamSampleOffset: absolute stream sample index of data[0].
    //
    // Returns only detections at or above their unit's threshold, sorted by
    // (sampleIndex, unitIndex).
    std::vector<MultiPeakEvent> processChunk( const double *data, size_t nSamples,
                                              long long streamSampleOffset );

    // Forces a final decision round over whatever every unit currently has
    // buffered -- same rationale and same warning as
    // ConvolutionEngine::flush(): call ONCE at the true end of a finite
    // stream (offline scoring), never on a live session, and never on an
    // engine you intend to keep feeding.
    std::vector<MultiPeakEvent> flush();

    // Applies a drift-schedule swap to one unit: updates the bank AND
    // rebuilds that unit's ConvolutionEngine taps, which the bank alone
    // cannot do because this class widens taps to double at construction.
    // Routing every swap through here is what stops the two copies drifting
    // apart -- see MultiFilterBank::updateFilters()'s note.
    //
    // This DOES reset that one unit's history, unlike the GPU path's
    // in-place cudaMemcpy over the taps. ConvolutionEngine has no
    // setTaps(), and adding one means editing the file this port exists to
    // leave untouched, so the engine is reconstructed instead. The cost is
    // bounded and one-off per event: that unit loses templateLength-1
    // samples of context, ~2 ms at 30 kHz. It is also arguably correct --
    // a swap changes which channels the unit reads, so the buffered history
    // is the wrong data to carry across anyway.
    void updateUnit( int unitIndex, const std::vector<int32_t> &channels,
                     const std::vector<float> &filter, float threshold );

    int nUnits() const { return nUnits_; }
    int nThreads() const { return static_cast<int>( workers_.size() ); }

    // The unit ids and thresholds this engine is currently using, for
    // logging and for the fetch thread's CSV.
    const MultiFilterBank &bank() const { return bank_; }

private:
    // What a worker does to one unit for the current chunk. Split out so
    // processChunk() and flush() share it.
    void runUnit( int u );
    void workerLoop( int workerIndex );
    void dispatch();          // wake workers, wait for all units to be claimed and done

    MultiFilterBank bank_;
    int             nUnits_;
    int             nChannelsGroup_;

    // Retained because updateUnit() reconstructs a unit's ConvolutionEngine
    // and must hand it the SAME separation the others were built with.
    // Passing 0 there instead would silently fall back to the class's
    // templateLength/2 default for exactly the units a drift schedule
    // touches -- the same defect a previous commit had to fix in both live
    // fetch threads' engine construction.
    long long       minSeparationSamples_;

    std::vector<ConvolutionEngine> engines_;

    // Parallel to engines_, and populated only when fastPath_ is set. Each
    // owns its unit's channel-major float32 sample history; engines_[u] then
    // owns only the D-buffer and the decisions.
    bool                           fastPath_;
    std::vector<FastMatchedFilter> fast_;
    std::vector<std::vector<double>> dScratch_;   // per-unit, reused

    // The chunk transposed to channel-major float32 once per processChunk(),
    // shared read-only by every worker: groupT_[ch * jobSamples_ + t]. See
    // FastMatchedFilter::computeD() for why this is not left as a per-unit
    // gather -- briefly, a strided gather out of a 384-channel time-major
    // buffer wastes ~7/8 of every cache line it touches, and doing it per
    // unit repeats that N times over the same data.
    std::vector<float> groupT_;

    // Per-unit scratch for the channel gather, allocated once. Each is
    // nChannelsPerUnit * maxSamplesSeen_ doubles and grows on demand rather
    // than being sized from a declared maximum -- unlike the GPU version,
    // which had to preallocate every device buffer at construction from
    // maxChunkSamples, there is no penalty here for a chunk that turns out
    // larger than expected, and so no cap to get wrong.
    std::vector<std::vector<double>> gather_;

    // Per-unit output, so workers never contend on a shared vector. Merged
    // and sorted by processChunk().
    std::vector<std::vector<MultiPeakEvent>> out_;

    // ---- current job -------------------------------------------------
    const double *jobData_ = nullptr;
    size_t        jobSamples_ = 0;
    long long     jobOffset_ = 0;
    bool          jobIsFlush_ = false;

    std::vector<std::thread> workers_;
    std::mutex               mtx_;
    std::condition_variable  cvStart_, cvDone_;
    std::atomic<int>         nextUnit_{0};
    int                      nDone_ = 0;
    unsigned                 generation_ = 0;
    bool                     stop_ = false;
};

#endif // CLOSEDLOOP_MULTICONVOLUTIONENGINE_H
