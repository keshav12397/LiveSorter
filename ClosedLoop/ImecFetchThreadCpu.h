#ifndef CLOSEDLOOP_IMECFETCHTHREADCPU_H
#define CLOSEDLOOP_IMECFETCHTHREADCPU_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "MultiFilterBank.h"
#include "ThreadSafeQueue.h"
#include "SpikeQueue.h"
#include "AnalysisFeed.h"
#include "Events.h"

class EventPublisher;

// All-units counterpart to ImecFetchThread -- see README.md's "All-units
// detection" section. Owns exactly one SpikeGLX handle (hIM), same
// concurrency rule as every other *FetchThread. Fetches the FULL CAR channel
// group + IMEC SY channel in small chunks (identical fetch loop structure to
// ImecFetchThread.cpp), but instead of subsetting down to one target's 5
// channels and running one ConvolutionEngine, it hands the whole group to
// Preprocessor + MultiConvolutionEngine, which scores every unit in the
// given MultiFilterBank across a pool of worker threads.
//
// Detections always go to spikeTimesPath as `unit_id,sample_index,score` CSV
// rows -- unit_id is the real Kilosort cluster id (translated via
// MultiFilterBank::hostUnitIds), not the internal 0..nUnits-1 array index.
//
// It ALSO pushes each detection into a ThreadSafeQueue<SpikeEvent> when one
// is supplied. It deliberately did not, until this class gained a consumer:
// with no DecisionThread draining it, an entire session's per-unit events
// would have accumulated in the queue as a pure leak. So the queue pointer is
// optional and nullptr keeps the old detection-only behaviour exactly --
// which is also what makes it safe to run this binary without a decision
// stage at all.
class ImecFetchThreadCpu {
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
    // preprocessing plus engine.processChunk(), i.e. the detection
    // pipeline's own per-chunk cost, NOT time spent waiting on
    // sglx_fetch/the network -- see FilterGen/validate_all_units.py, which
    // summarizes this against fetchChunkMs to check the pipeline is keeping
    // up with real time.
    //
    // Chunk 0 is an ordinary chunk: every row of this log means the same
    // thing, with no warm-up row to discard.
    //
    // spikeQueue / publisher: both optional (nullptr). Neither can block the
    // fetch loop -- the queue is a bounded-cost mutex push, and
    // EventPublisher::publish() is a memcpy into a fixed ring that drops
    // rather than waits (see EventPublisher.h for why that matters here).
    //
    // spikeQueue is the HOT queue (SpikeQueue.h): detections only, pushed
    // batched once per chunk. analysisFeed is the SLOW queue
    // (AnalysisFeed.h), optional and independent of spikeQueue -- a chunk is
    // handed to it whether or not it carried any detections, since drift
    // estimation needs the sample data regardless. Neither queue shares a
    // mutex with the other, and a stalled analysis consumer cannot slow this
    // thread: acquire() either returns a buffer or nullptr immediately, and
    // nullptr just means this chunk is skipped on the slow side.
    ImecFetchThreadCpu( void *hSglx, const MultiFilterBank &filterBank,
                         const std::string &carChannelMapJsonPath,
                         bool applyHighpass, double highpassCutoffHz,
                         int imecSyncBit, int fetchChunkMs,
                         const std::string &spikeTimesPath,
                         const std::string &latencyLogPath = "",
                         SpikeQueue *spikeQueue = 0,
                         EventPublisher *publisher = 0,
                         const SyllableFromSy &syllableFromSy = SyllableFromSy(),
                         AnalysisFeed *analysisFeed = 0 );

    void start();
    void stop();  // signals the loop to exit; does not join
    void join();

private:
    void fetchLoop();

    void            *hSglx_;
    // BY VALUE, not by reference. fetchLoop() translates every unit's raw
    // SpikeGLX channel ids into positions within the CAR group and stores
    // the result back into the bank. Owning a copy keeps that mutation
    // local: taking a reference here would rewrite the caller's bank as a
    // side effect of starting the thread.
    MultiFilterBank  filterBank_;
    std::string      carChannelMapJsonPath_;
    bool             applyHighpass_;
    double           highpassCutoffHz_;
    int              imecSyncBit_;
    int              fetchChunkMs_;
    std::string      spikeTimesPath_;
    std::string      latencyLogPath_;

    SpikeQueue     *spikeQueue_;
    EventPublisher *publisher_;
    SyllableFromSy  syllableFromSy_;
    AnalysisFeed   *analysisFeed_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_IMECFETCHTHREADCPU_H
