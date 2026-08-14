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
// the target's filter channels + the IMEC SY channel in small chunks (via
// sglx_fetch, following the exact continuous-polling pattern in
// SDK/CPP/DemoRemoteAPI.cpp's latency_test()), runs them through
// ConvolutionEngine, and reports threshold-crossing peaks as SpikeEvents
// (timestamped relative to the SY channel's most recent sync edge) to
// DecisionThread, plus writes every detection's raw sample index to
// spikeTimesPath.
//
// To change the fetching mechanism itself (e.g. switch to
// sglx_fetchLatest, change what's included in the channel subset): this
// class's fetchLoop() is the only place that matters.
class ImecFetchThread {
public:
    ImecFetchThread( void *hSglx, FilterBank &filterBank, int imecSyncBit,
                      int fetchChunkMs, ThreadSafeQueue<SpikeEvent> &spikeQueue,
                      const std::string &spikeTimesPath );

    void start();
    void stop();  // signals the loop to exit; does not join
    void join();

private:
    void fetchLoop();

    void   *hSglx_;
    FilterBank &filterBank_;   // threshold read each iteration -- Calibration may have just set it
    int     imecSyncBit_;
    int     fetchChunkMs_;
    ThreadSafeQueue<SpikeEvent> &spikeQueue_;
    std::string spikeTimesPath_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_IMECFETCHTHREAD_H
