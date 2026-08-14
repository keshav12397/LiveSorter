#ifndef CLOSEDLOOP_IMECFETCHTHREAD_H
#define CLOSEDLOOP_IMECFETCHTHREAD_H

#include <string>
#include <thread>
#include <atomic>
#include <fstream>

#include "FilterBank.h"
#include "ConvolutionEngine.h"
#include "SyncEdgeTracker.h"
#include "ThreadSafeQueue.h"
#include "Events.h"

// Owns exactly one SpikeGLX handle (hIM) -- per the concurrency rules in
// README.md, no other thread may touch this handle. Continuously fetches
// the FULL CAR channel group (see Preprocessor.h -- must match the group
// the filter was trained with, not just its own 5 channels) + the IMEC SY
// channel in small chunks (via sglx_fetch, following the exact
// continuous-polling pattern in SDK/CPP/DemoRemoteAPI.cpp's
// latency_test()), runs highpass+CAR across that full group, subsets down
// to the filter's channels, runs ConvolutionEngine, and reports
// threshold-crossing peaks as SpikeEvents (timestamped relative to the SY
// channel's most recent sync edge) to DecisionThread, plus writes every
// detection's raw sample index to spikeTimesPath.
//
// To change the fetching mechanism itself (e.g. switch to
// sglx_fetchLatest, change what's included in the channel subset): this
// class's fetchLoop() is the only place that matters.
class ImecFetchThread {
public:
    // carChannelMapJsonPath: same file passed to FilterGen's
    // --channel-map-json when this filter was trained (e.g.
    // shank1only.json). Pass "" to disable CAR (fetches just the filter's
    // own channels, matching --no-car).
    ImecFetchThread( void *hSglx, FilterBank &filterBank,
                      const std::string &carChannelMapJsonPath,
                      bool applyHighpass, double highpassCutoffHz,
                      int imecSyncBit, int fetchChunkMs,
                      ThreadSafeQueue<SpikeEvent> &spikeQueue,
                      const std::string &spikeTimesPath );

    void start();
    void stop();  // signals the loop to exit; does not join
    void join();

private:
    void fetchLoop();

    void   *hSglx_;
    FilterBank &filterBank_;   // threshold read each iteration -- Calibration may have just set it
    std::string carChannelMapJsonPath_;
    bool    applyHighpass_;
    double  highpassCutoffHz_;
    int     imecSyncBit_;
    int     fetchChunkMs_;
    ThreadSafeQueue<SpikeEvent> &spikeQueue_;
    std::string spikeTimesPath_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_IMECFETCHTHREAD_H
