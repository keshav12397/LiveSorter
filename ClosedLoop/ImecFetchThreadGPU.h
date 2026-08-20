#ifndef CLOSEDLOOP_IMECFETCHTHREADGPU_H
#define CLOSEDLOOP_IMECFETCHTHREADGPU_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "GpuFilterBank.h"
#include "ThreadSafeQueue.h"
#include "Events.h"

class EventPublisher;

// All-units counterpart to ImecFetchThread -- see README.md's "All-units GPU
// detection" section. Owns exactly one SpikeGLX handle (hIM), same
// concurrency rule as every other *FetchThread. Fetches the FULL CAR channel
// group + IMEC SY channel in small chunks (identical fetch loop structure to
// ImecFetchThread.cpp), but instead of subsetting down to one target's 5
// channels and running one CPU ConvolutionEngine, it hands the whole group to
// GpuPreprocessor+GpuConvolutionEngine, which score every unit in the given
// GpuFilterBank simultaneously on the GPU.
//
// Detections always go to spikeTimesPath as `unit_id,sample_index,score` CSV
// rows -- unit_id is the real Kilosort cluster id (translated via
// GpuFilterBank::hostUnitIds), not the internal 0..nUnits-1 array index.
//
// It ALSO pushes each detection into a ThreadSafeQueue<SpikeEvent> when one
// is supplied. It deliberately did not, until this class gained a consumer:
// with no DecisionThread draining it, an entire session's per-unit events
// would have accumulated in the queue as a pure leak. So the queue pointer is
// optional and nullptr keeps the old detection-only behaviour exactly --
// which is also what makes it safe to run this binary without a decision
// stage at all.
class ImecFetchThreadGPU {
public:
    // Where syllable events come from when this thread is the one producing
    // them. Only used when `enabled` is true (config `syllableSource=imecSy`);
    // the production path leaves this off and lets NiFetchThread decode the
    // NI digital word as it always has.
    //
    // TEST PATH, and the reason it exists: the SpikeGLX instance available
    // for testing can only replay a simulated IMEC file, and its NI stream is
    // a Fake_40kHz source whose digital word is all zeros -- there is no way
    // to get synthetic syllable codes into the NI stream at all. So
    // FilterGen/make_sim_session.py writes them into the IMEC SY word's bits
    // 0-2 (error flags, always 0 on healthy hardware, never read by this
    // codebase; bit 6 remains the real sync waveform). Read that script's
    // docstring section "Syllable codes ride on the SY channel, not NI".
    //
    // A useful side effect rather than the motivation: codes decoded here
    // share the spikes' own sample clock, so a syllable-to-spike interval is
    // an exact sample difference and the cross-stream alignment question
    // (README.md) does not arise for the thing under test.
    struct SyllableFromSy {
        bool        enabled;
        int         startBit;         // lowest of the contiguous code bits (0)
        int         width;            // number of code bits (3)
        int         debounceSamples;  // same meaning as NiFetchThread's
        ThreadSafeQueue<SyllableEvent> *queue;
        std::string syllableTimesPath;  // "" to skip; same `code,sampleIndex` CSV
                                         // NiFetchThread writes, so the viewer's
                                         // offline reader needs no second parser

        SyllableFromSy()
            :   enabled( false ), startBit( 0 ), width( 3 ), debounceSamples( 10 ),
                queue( 0 )
        {}
    };

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
    //
    // spikeQueue / publisher: both optional (nullptr). Neither can block the
    // fetch loop -- the queue is a bounded-cost mutex push, and
    // EventPublisher::publish() is a memcpy into a fixed ring that drops
    // rather than waits (see EventPublisher.h for why that matters here).
    ImecFetchThreadGPU( void *hSglx, const GpuFilterBank &filterBank,
                         const std::string &carChannelMapJsonPath,
                         bool applyHighpass, double highpassCutoffHz,
                         int imecSyncBit, int fetchChunkMs,
                         const std::string &spikeTimesPath,
                         const std::string &latencyLogPath = "",
                         ThreadSafeQueue<SpikeEvent> *spikeQueue = 0,
                         EventPublisher *publisher = 0,
                         const SyllableFromSy &syllableFromSy = SyllableFromSy() );

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

    ThreadSafeQueue<SpikeEvent> *spikeQueue_;
    EventPublisher              *publisher_;
    SyllableFromSy               syllableFromSy_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_IMECFETCHTHREADGPU_H
