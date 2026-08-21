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
// The producers are ImecFetchThreadCpu's fetch loop, NiFetchThread's fetch
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

    // Sets the version-2 channel-geometry preamble (see LiveWire.h): one
    // entry per unit in `unitChannelGeom`, in the same order as the
    // `unitIds` passed to the constructor, each entry the channel positions
    // for that unit in the order a kWireAmpChannel record's `c` field will
    // index into. Optional -- an EventPublisher that never calls this still
    // sends a valid (empty, nChannels=0 for every unit) preamble, so a run
    // with no drift-tracking data configured still speaks valid v2. Must be
    // called before start() (or at least before the first accept()); it is
    // not synchronised against a concurrent serverLoop().
    void setChannelGeometry( const std::vector<std::vector<livewire::ChannelGeom> > &unitChannelGeom,
                              int templateLength );

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

    // Version-2 preamble, flattened for a single send() each: one count per
    // unit, then all channels unit-major (see LiveWire.h). Both empty
    // (nChannels_ all 0) until setChannelGeometry() is called, which is a
    // valid v2 stream -- just one that carries no drift geometry.
    std::vector<int32_t>              nChannels_;
    std::vector<livewire::ChannelGeom> channelGeom_;

    // Fixed-capacity ring. head_ is the next write slot; tail_ the next
    // unsent slot. Both only move under ringMutex_.
    std::vector<livewire::WireRecord> ring_;
    size_t head_, tail_, count_;
    mutable std::mutex ringMutex_;

    std::atomic<bool>      hasClient_;
    std::atomic<long long> nPublished_, nDropped_;

    // uintptr_t rather than SOCKET so this header stays free of winsock2.h.
    // Keeps <windows.h> out of this header's includers, which is worth
    // doing on its own: it drags in min/max macros that break std::min.
    // winsock2.h drags in windows.h, whose min/max macros break any header
    // using std::min/std::max after it, and it must precede any windows.h
    // that would otherwise pull in winsock 1 -- an include-order constraint
    // that would propagate to every file including this one.
    unsigned long long listenSock_, clientSock_;

    std::atomic<bool> stopFlag_;
    std::thread       thread_;
    bool              started_;
};

#endif // CLOSEDLOOP_EVENTPUBLISHER_H
