#ifndef CLOSEDLOOP_ANALYSISTHREAD_H
#define CLOSEDLOOP_ANALYSISTHREAD_H

#include <atomic>
#include <thread>
#include <string>

#include "AnalysisFeed.h"

// Consumer of the SLOW queue (AnalysisFeed.h). This is a STUB: it takes
// chunks and counts them, nothing more. Drift correction and the plotting
// feed a chunk needs to reach are owned elsewhere (a parallel effort on
// this same queue split); this class exists so the AnalysisFeed side of the
// wiring has a real consumer to prove the pool-handoff contract end to end
// -- acquire()/publish() on the fetch thread, take()/release() here -- ahead
// of that payload landing.
//
// The one rule that matters for anyone replacing the stub body: release()
// MUST run for every chunk taken, including on an exception path, or the
// pool drains permanently and AnalysisFeed silently degrades to dropping
// every chunk (see AnalysisFeed.h's ownership contract and nInFlight()).
// runLoop() below does this with a scope guard rather than a bare call at
// the end of the loop body for exactly that reason.
class AnalysisThread {
public:
    explicit AnalysisThread( AnalysisFeed &feed );

    void start();
    void stop();
    void join();

    long long nChunksSeen()  const { return nChunksSeen_.load(); }
    long long nSamplesSeen() const { return nSamplesSeen_.load(); }
    long long nSpikesSeen()  const { return nSpikesSeen_.load(); }

    std::string summary() const;

private:
    void runLoop();

    AnalysisFeed &feed_;

    std::atomic<long long> nChunksSeen_;
    std::atomic<long long> nSamplesSeen_;
    std::atomic<long long> nSpikesSeen_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_ANALYSISTHREAD_H
