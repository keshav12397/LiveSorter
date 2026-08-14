#ifndef CLOSEDLOOP_DECISIONTHREAD_H
#define CLOSEDLOOP_DECISIONTHREAD_H

#include <string>
#include <deque>
#include <thread>
#include <atomic>
#include <fstream>
#include <chrono>

#include "ThreadSafeQueue.h"
#include "Events.h"

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
    DecisionThread( void *hSglx, ThreadSafeQueue<SpikeEvent> &spikeQueue,
                     ThreadSafeQueue<SyllableEvent> &syllableQueue,
                     double windowStartS, double windowEndS, int spikeCountThreshold,
                     const std::string &doLine, int doPulseMs,
                     const std::string &decisionLogPath );

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

    std::deque<PendingSyllable> pending_;

    bool lineHigh_;
    std::chrono::steady_clock::time_point lineHighUntil_;

    std::ofstream decisionLog_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_DECISIONTHREAD_H
