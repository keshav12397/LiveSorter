#ifndef CLOSEDLOOP_ANALYSISTHREAD_H
#define CLOSEDLOOP_ANALYSISTHREAD_H

#include <atomic>
#include <thread>
#include <string>
#include <vector>

#include "LiveWire.h"

#include "AnalysisFeed.h"

class MultiFilterBank;
class EventPublisher;

// Consumer of the SLOW queue (AnalysisFeed.h). It drains chunks and, when
// given a filter bank and a publisher, turns each chunk's spikes into
// version-2 kWireAmpChannel records (AmplitudeExtractor.h) and publishes
// them for the viewer's drift tracker.
//
// This is the ONLY place amplitude extraction runs. It is per-spike,
// per-channel work -- precisely what AnalysisFeed exists to keep off
// ImecFetchThreadCpu's <10 ms budget, whose entire obligation to this path
// is acquire(), fill, publish(). If this thread falls behind, acquire()
// starts returning null and chunks are skipped: the drift trace loses
// resolution and the detection path is untouched.
//
// Both references are optional. Constructed feed-only, the body degrades to
// counting chunks, which is what the offline/replay paths and the
// queue-contract tests want.
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
    AnalysisThread( AnalysisFeed &feed, const MultiFilterBank &filterBank,
                     EventPublisher *publisher, int templateOffset );

    void start();
    void stop();
    void join();

    long long nChunksSeen()  const { return nChunksSeen_.load(); }
    long long nSamplesSeen() const { return nSamplesSeen_.load(); }
    long long nSpikesSeen()  const { return nSpikesSeen_.load(); }
    long long nAmpRecords()  const { return nAmpRecords_.load(); }

    std::string summary() const;

private:
    void runLoop();

    AnalysisFeed &feed_;
    const MultiFilterBank *filterBank_;
    EventPublisher        *publisher_;
    int                    templateOffset_;

    std::atomic<long long> nChunksSeen_;
    std::atomic<long long> nSamplesSeen_;
    std::atomic<long long> nSpikesSeen_;
    std::atomic<long long> nAmpRecords_;

    // Reused across chunks so the steady state does not allocate.
    std::vector<livewire::WireRecord> amps_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_ANALYSISTHREAD_H
