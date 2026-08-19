#ifndef CLOSEDLOOP_DECISIONTHREAD_H
#define CLOSEDLOOP_DECISIONTHREAD_H

#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <atomic>
#include <fstream>
#include <chrono>

#include "ThreadSafeQueue.h"
#include "Events.h"

class EventPublisher;

// Owns exactly one SpikeGLX handle (hDO), dedicated solely to
// sglx_ni_DO_set calls -- see README.md's concurrency rules for why this
// needs its own handle rather than sharing hIM or hNI.
//
// This is the module to change for any different decision policy: all of
// it lives in evaluateSpike()/evaluateSyllable() below. The counting
// itself is incremental (each incoming spike immediately checks every
// still-open pending syllable window and raises the line the instant a
// count crosses threshold), not "wait for the window to close then count"
// -- that's what keeps the observable trigger latency close to the last
// qualifying spike's own arrival, not the full window duration.
class DecisionThread {
public:
    // decisionUnitIds: which Kilosort cluster ids may contribute to the
    // count. EMPTY (the default) means "every spike counts", which is the
    // behaviour this class always had and what the single-target
    // ClosedLoop.exe still gets -- there, SpikeEvent::unitId is -1 for every
    // event and there is only one unit anyway. The all-units pipeline pushes
    // 157 units' spikes into this same queue, where counting all of them is
    // meaningless, so it names the driving unit(s) explicitly.
    //
    // decisionCsvPath: optional machine-readable sibling of decisionLogPath,
    // one row per CLOSED trial (`syllable_sample_index,code,triggered,
    // spike_count`). decisionLogPath stays exactly as it was -- it is a
    // human-readable operator log, and parsing it in the viewer would make
    // its prose a wire format.
    //
    // publisher: optional live viewer feed (see EventPublisher.h); nullptr
    // means this class behaves as it did before the viewer existed.
    DecisionThread( void *hSglx, ThreadSafeQueue<SpikeEvent> &spikeQueue,
                     ThreadSafeQueue<SyllableEvent> &syllableQueue,
                     double windowStartS, double windowEndS, int spikeCountThreshold,
                     const std::string &doLine, int doPulseMs,
                     const std::string &decisionLogPath,
                     const std::vector<int> &decisionUnitIds = std::vector<int>(),
                     const std::string &decisionCsvPath = "",
                     EventPublisher *publisher = nullptr );

    void start();
    void stop();
    void join();

private:
    void runLoop();

    // The swappable decision policy -- see class comment.
    void onSpikeEvent( const SpikeEvent &spk );
    void onSyllableEvent( const SyllableEvent &syl );
    void pruneExpired();   // wall-clock-based: drop pending syllables whose window is long past
    void raiseLine();
    void lowerLineIfDue();

    struct PendingSyllable {
        SyllableEvent event;
        int  count;
        bool triggered;
        std::chrono::steady_clock::time_point receivedAt;
    };

    void *hSglx_;
    ThreadSafeQueue<SpikeEvent>    &spikeQueue_;
    ThreadSafeQueue<SyllableEvent> &syllableQueue_;

    double windowStartS_, windowEndS_;
    int    spikeCountThreshold_;
    std::string doLine_;
    int    doPulseMs_;

    // Sorted; empty means "count every unit" (see the constructor comment).
    std::vector<int> decisionUnitIds_;
    bool             countsAllUnits_;
    EventPublisher  *publisher_;

    std::deque<PendingSyllable> pending_;

    bool lineHigh_;
    std::chrono::steady_clock::time_point lineHighUntil_;

    std::ofstream decisionLog_;
    std::ofstream decisionCsv_;

    // Emitted once per trial when its window closes, to both decisionCsv_
    // and the viewer -- the single place a trial's final triggered/count
    // outcome is known.
    void closeTrial( const PendingSyllable &p );

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_DECISIONTHREAD_H
