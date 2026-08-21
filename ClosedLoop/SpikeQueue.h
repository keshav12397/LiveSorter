#ifndef CLOSEDLOOP_SPIKEQUEUE_H
#define CLOSEDLOOP_SPIKEQUEUE_H

#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>
#include <cstddef>

#include "Events.h"

// The HOT queue: detections from the fetch thread to DecisionThread, and
// nothing else.
//
// The split this is half of
// --------------------------
// Two queues cross the fetch/analysis boundary, with deliberately different
// guarantees:
//
//   SpikeQueue (this file)   spike times only, target latency < 10 ms.
//                            Bounded, preallocated, never allocates while
//                            running. A dropped record is a missed decision.
//
//   AnalysisFeed.h           preprocessed chunks for drift estimation and
//                            plotting, budget > 500 ms. Allowed to be slow,
//                            allowed to drop freely -- a centroid built from
//                            140 spikes instead of 150 is fine.
//
// They share no mutex. That is the whole point: backpressure from a slow
// viewer or a slow drift estimator must not be able to reach the path that
// decides whether to fire the DO line. Anything that needs more than a spike
// time belongs in the other queue, and adding a field here is a decision to
// spend hot-path budget.
//
// Why not ThreadSafeQueue<SpikeEvent>
// ------------------------------------
// ThreadSafeQueue wraps std::queue, whose default container is std::deque,
// and MSVC sizes deque blocks in ELEMENTS:
//
//     _DEQUESIZ = sizeof(T) <= 1 ? 16 : <= 2 ? 8 : <= 4 ? 4 : <= 8 ? 2 : 1
//
// SpikeEvent is 24 bytes, so that is ONE element per block -- a malloc on
// every push and a free on every pop, while holding the lock, on the
// sample-critical path. The all-units pipeline pushes ~1500 detections/s.
// This is the same trap RingDeque.h documents (there it cost 12.2 million
// allocations for two seconds of data and 72% of the detection path's
// runtime); the container was the problem, not the algorithm. See
// bench_queues.cpp for the measurement on this queue.
//
// It also grew without bound. ThreadSafeQueue's own comment records the
// consequence: a consumer that falls behind makes the queue lengthen
// forever and the decisions run on ever-staler events. An unbounded queue
// does not actually avoid dropping data, it just converts dropping into
// unbounded latency, which for a closed loop is worse -- a decision made on
// a spike from thirty seconds ago is not a late decision, it is a wrong one.
//
// Overflow policy: DROP OLDEST, and count it.
// -------------------------------------------
// If the consumer stalls long enough to fill the ring, the oldest events are
// overwritten. That is the correct trade for this system for the reason
// above -- the freshest spikes are the ones a pending decision window cares
// about -- but it IS data loss, so nDropped() is exposed and belongs in the
// shutdown summary the way StreamAccountant and EventPublisher report
// theirs. A run that silently discarded detections must say so somewhere.
//
// The default capacity is sized so this cannot happen in normal operation:
// 65536 events at the ~1500/s an all-units run produces is a ~44 s stall.
// Reaching it means something is badly wrong, and the count is how you find
// out rather than a routine occurrence to be tuned around.
class SpikeQueue {
public:
    explicit SpikeQueue( size_t capacity = 65536 )
        : buf_( capacity ), head_( 0 ), tail_( 0 ), count_( 0 ),
          nPushed_( 0 ), nDropped_( 0 ), maxDepth_( 0 ) {}

    // Never allocates, never blocks on anything but the mutex, and holds
    // that mutex for a 24-byte copy and two index updates.
    void push( const SpikeEvent &e )
    {
        {
            std::lock_guard<std::mutex> lock( mutex_ );
            pushLocked( e );
        }
        cv_.notify_one();
    }

    // Batched form for the fetch loop's per-chunk detection vector: one lock
    // acquisition for the whole chunk rather than one per spike. At 157
    // units a chunk can carry dozens of detections, so this is the form the
    // fetch thread should use.
    void push( const SpikeEvent *e, size_t n )
    {
        if( n == 0 )
            return;
        {
            std::lock_guard<std::mutex> lock( mutex_ );
            for( size_t i = 0; i < n; ++i )
                pushLocked( e[i] );
        }
        cv_.notify_one();
    }

    // Moves every queued event into `out` (appending) under ONE lock, and
    // returns how many. Draining rather than popping one at a time is not
    // cosmetic: DecisionThread's loop previously took at most one event per
    // pass and blocked up to 1 ms on whichever queue was empty, capping it
    // near 1000 events/s -- below the ~1500/s the all-units pipeline
    // produces, so the backlog grew forever.
    //
    // `out` is the caller's buffer and is expected to be reused across
    // calls, so the drain itself does not allocate in the steady state.
    //
    // At most kMaxDrain events per call, and the consumer is expected to
    // loop until this returns 0. That cap is not arbitrary tuning -- it
    // bounds how long the producer can be blocked. An uncapped drain holds
    // the mutex for `count_` copies, and with a 65536-deep ring that is a
    // 65536-element memcpy the fetch thread has to wait behind. Measured
    // before the cap existed: under overload the producer's worst chunk hit
    // 5616 us, WORSE than the unbounded std::deque it replaced (1522 us),
    // which would have made this class a regression in exactly the tail the
    // whole split exists to protect.
    static const size_t kMaxDrain = 4096;

    size_t drain( std::vector<SpikeEvent> &out )
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return drainLocked( out );
    }

    // Blocks up to `timeoutMs` for at least one event, then drains. Returns
    // the number drained, 0 on timeout -- which lets the consumer check a
    // stop flag periodically instead of blocking forever.
    size_t waitDrain( std::vector<SpikeEvent> &out, int timeoutMs )
    {
        std::unique_lock<std::mutex> lock( mutex_ );
        if( !cv_.wait_for( lock, std::chrono::milliseconds( timeoutMs ),
                           [this]{ return count_ > 0; } ) ) {
            return 0;
        }
        return drainLocked( out );
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return count_;
    }

    size_t capacity() const { return buf_.size(); }

    long long nPushed()  const { return nPushed_.load(); }
    long long nDropped() const { return nDropped_.load(); }

    // High-water mark, for the shutdown summary: a run that never exceeded a
    // handful of queued events was never near the drop threshold, and one
    // that sat at capacity was dropping continuously.
    size_t maxDepth() const { return maxDepth_.load(); }

    std::string summary() const
    {
        std::string s = "SpikeQueue: " + std::to_string( nPushed() ) + " pushed, "
                      + std::to_string( nDropped() ) + " DROPPED, peak depth "
                      + std::to_string( maxDepth() ) + "/"
                      + std::to_string( capacity() );
        if( nDropped() > 0 )
            s += "  <-- detections were lost; decisions ran on incomplete counts";
        return s;
    }

private:
    // Caller holds mutex_. Returns how many were moved, capped at kMaxDrain.
    size_t drainLocked( std::vector<SpikeEvent> &out )
    {
        size_t n = count_ < kMaxDrain ? count_ : kMaxDrain;
        for( size_t i = 0; i < n; ++i ) {
            out.push_back( buf_[tail_] );
            tail_ = ( tail_ + 1 ) % buf_.size();
        }
        count_ -= n;
        return n;
    }

    // Caller holds mutex_.
    void pushLocked( const SpikeEvent &e )
    {
        if( count_ == buf_.size() ) {
            // Full: overwrite the oldest. tail_ advances so the ring stays
            // exactly `capacity` deep rather than wrapping onto itself.
            tail_ = ( tail_ + 1 ) % buf_.size();
            --count_;
            nDropped_.fetch_add( 1 );
        }
        buf_[head_] = e;
        head_ = ( head_ + 1 ) % buf_.size();
        ++count_;
        nPushed_.fetch_add( 1 );
        if( count_ > maxDepth_.load() )
            maxDepth_.store( count_ );
    }

    std::vector<SpikeEvent> buf_;    // preallocated once, never resized
    size_t head_, tail_, count_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    std::atomic<long long> nPushed_, nDropped_;
    std::atomic<size_t>    maxDepth_;
};

#endif // CLOSEDLOOP_SPIKEQUEUE_H
