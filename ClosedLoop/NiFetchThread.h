#ifndef CLOSEDLOOP_NIFETCHTHREAD_H
#define CLOSEDLOOP_NIFETCHTHREAD_H

#include <vector>
#include <thread>
#include <atomic>

#include "SyncEdgeTracker.h"
#include "ThreadSafeQueue.h"
#include "Events.h"

// Owns exactly one SpikeGLX handle (hNI) -- see README.md's concurrency
// rules. Continuously fetches the NI digital-word (DW) channel, decodes a
// debounced 3-bit syllable code from `syllableLines` (assumed contiguous
// ascending bits, e.g. {5,6,7} -- see decode logic in the .cpp), tracks the
// sync pulse on `syncBit`, and reports one SyllableEvent per debounced
// code *onset* (not continuously while a code is held -- see
// fetchLoop()'s debounce/re-arm logic in the .cpp for exactly how a
// repeated syllable after a return to code 0 is treated as a fresh event).
class NiFetchThread {
public:
    NiFetchThread( void *hSglx, int niSyncBit, const std::vector<int> &syllableLines,
                   int debounceSamples, int fetchChunkMs,
                   ThreadSafeQueue<SyllableEvent> &syllableQueue );

    void start();
    void stop();
    void join();

private:
    void fetchLoop();

    void *hSglx_;
    int   syncBit_;
    std::vector<int> syllableLines_;
    int   debounceSamples_;
    int   fetchChunkMs_;
    ThreadSafeQueue<SyllableEvent> &syllableQueue_;

    std::atomic<bool> stopFlag_;
    std::thread        thread_;
};

#endif // CLOSEDLOOP_NIFETCHTHREAD_H
