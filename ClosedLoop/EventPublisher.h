#ifndef CLOSEDLOOP_EVENTPUBLISHER_H
#define CLOSEDLOOP_EVENTPUBLISHER_H

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

#include "LiveWire.h"

// Publishes live events to SpikeViewer over a loopback TCP socket, without
// ever letting the viewer's speed reach the fetch loop.
//
// The constraint that shapes this whole class
// -------------------------------------------
// The producers are ImecFetchThreadGPU's fetch loop, NiFetchThread's fetch
// loop, and DecisionThread. All three are on the sample-critical path: the
// server's ring is only ~8 s deep (see StreamAccountant.h), and per-chunk
// blocking work has already had to be removed from the fetch loop once (the
// per-chunk ofstream flushes). So publish() must cost a bounded, tiny amount
// of time with no syscall, no allocation, and no possibility of waiting on a
// socket, an accept(), or a slow reader.
//
// Therefore: publish() copies into a fixed-capacity ring under a mutex held
// for the length of a memcpy, and returns. A separate thread owns the
// listening socket, the accept, and every send(), on a non-blocking socket.
// If the ring fills because nobody is draining it, the OLDEST records are
// overwritten and counted -- dropping viewer data is always correct here,
// because the CSV files are the record of a run and this socket is only a
// live view of it.
//
// Nothing in this class touches SpikeGLX, so the one-handle-one-thread rule
// in README.md does not come into it; the publisher thread is not a fetch
// thread and owns no handle.
class EventPublisher {
public:
    // ringCapacity is in records (32 B each). The default holds ~2 s of the
    // worst per-second record rate a 157-unit run has produced, which is far
    // longer than a viewer redraw ever takes; it exists to absorb a stalled
    // reader, not to buffer a run.
    EventPublisher( int port, const std::vector<int> &unitIds,
                     double imecSampleRateHz, double niSampleRateHz,
                     bool syllableSourceIsImecSy,
                     size_t ringCapacity = 65536 );
    ~EventPublisher();

    // Starts the listen/accept/send thread. Safe to never call: an
    // unstarted publisher still accepts publish() calls and simply drops
    // them, so callers do not need a null check on every event.
    bool start();
    void stop();   // signals and joins

    void publish( const livewire::WireRecord &rec );

    // Batched form, for the fetch loop's per-chunk detection vector: one
    // lock acquisition for the whole chunk instead of one per spike.
    void publish( const livewire::WireRecord *recs, size_t n );

    bool hasClient() const { return hasClient_.load(); }

    // For the shutdown summary, in the same spirit as StreamAccountant:
    // a run that silently threw away viewer data should say so somewhere.
    long long nPublished() const { return nPublished_.load(); }
    long long nDropped() const { return nDropped_.load(); }

    std::string summary() const;

private:
    void serverLoop();
    void closeClient();

    int   port_;
    std::vector<int> unitIds_;
    livewire::SessionHeader header_;

    // Fixed-capacity ring. head_ is the next write slot; tail_ the next
    // unsent slot. Both only move under ringMutex_.
    std::vector<livewire::WireRecord> ring_;
    size_t head_, tail_, count_;
    mutable std::mutex ringMutex_;

    std::atomic<bool>      hasClient_;
    std::atomic<long long> nPublished_, nDropped_;

    // uintptr_t rather than SOCKET so this header stays free of
    // winsock2.h -- it is included by ImecFetchThreadGPU.cu, and pulling
    // the Windows socket headers into an nvcc translation unit invites the
    // windows.h/min-max and winsock1-vs-2 include-order problems this
    // project has no reason to take on.
    unsigned long long listenSock_, clientSock_;

    std::atomic<bool> stopFlag_;
    std::thread       thread_;
    bool              started_;
};

#endif // CLOSEDLOOP_EVENTPUBLISHER_H
