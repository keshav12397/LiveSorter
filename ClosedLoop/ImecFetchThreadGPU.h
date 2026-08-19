#ifndef CLOSEDLOOP_IMECFETCHTHREADGPU_H
#define CLOSEDLOOP_IMECFETCHTHREADGPU_H

#include <string>
#include <thread>
#include <atomic>

#include "GpuFilterBank.h"

// All-units, detection-only counterpart to ImecFetchThread -- see
// README.md's "All-units GPU detection" section. Owns exactly one SpikeGLX
// handle (hIM), same concurrency rule as every other *FetchThread. Fetches
// the FULL CAR channel group + IMEC SY channel in small chunks (identical
// fetch loop structure to ImecFetchThread.cpp), but instead of subsetting
// down to one target's 5 channels and running one CPU ConvolutionEngine, it
// hands the whole group to GpuPreprocessor+GpuConvolutionEngine, which score
// every unit in the given GpuFilterBank simultaneously on the GPU.
//
// Detection-only: no DecisionThread/DO
// wiring here, and unlike ImecFetchThread this does NOT push into a
// ThreadSafeQueue<SpikeEvent> -- there is currently no consumer for
// per-unit spike events in this mode, and queuing into an un-drained queue
// for an entire session would just leak memory. Detections go to
// spikeTimesPath only, as `unit_id,sample_index,score` CSV rows -- unit_id
// is the real Kilosort cluster id (translated via
// GpuFilterBank::hostUnitIds), not the internal 0..nUnits-1 array index.
class ImecFetchThreadGPU {
public:
    // carChannelMapJsonPath: same file calibrate_all_units.py's
    // --channel-map-json was run with -- CAR must be computed over the
    // exact channel group the filters were fit against (see Preprocessor.h).
    // latencyLogPath: optional (pass "" to disable). CSV of
    // chunk_index,n_samples,latency_ms -- latency_ms is wall-clock time for
    // just engine.processChunk() (preprocess kernels + matched-filter kernel
    // + NMS/threshold kernel + the D2H copy of accepted detections, i.e. the
    // GPU pipeline's own per-chunk cost, NOT time spent waiting on
    // sglx_fetch/the network -- see FilterGen/validate_all_units.py, which
    // summarizes this against fetchChunkMs to check the pipeline is keeping
    // up with real time).
    //
    // IGNORE CHUNK 0 when reading this log. It reliably measures ~200 ms
    // against ~0.2 ms for every subsequent chunk, because the first launch
    // of each kernel is what actually loads its module and reserves its
    // local-memory backing store; the CUDA context itself is already up by
    // then (the engine's constructor cudaMallocs). Confirmed not to be PTX
    // JIT: embedding a native sm_80 cubin alongside the PTX leaves it
    // unchanged. It is also harmless -- the server's ring absorbs it
    // completely, with measured max backlog of 76 samples (2.5 ms) over a
    // 30 s run. Treat any OTHER chunk over a millisecond as the real signal.
    ImecFetchThreadGPU( void *hSglx, const GpuFilterBank &filterBank,
                         const std::string &carChannelMapJsonPath,
                         bool applyHighpass, double highpassCutoffHz,
                         int imecSyncBit, int fetchChunkMs,
                         const std::string &spikeTimesPath,
                         const std::string &latencyLogPath = "" );

    void start();
    void stop();  // signals the loop to exit; does not join
    void join();

private:
    void fetchLoop();

    void            *hSglx_;
    const GpuFilterBank &filterBank_;
    std::string      carChannelMapJsonPath_;
    bool             applyHighpass_;
    double           highpassCutoffHz_;
    int              imecSyncBit_;
    int              fetchChunkMs_;
    std::string      spikeTimesPath_;
    std::string      latencyLogPath_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_IMECFETCHTHREADGPU_H
