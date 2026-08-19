#ifndef CLOSEDLOOP_THREADSAFEQUEUE_H
#define CLOSEDLOOP_THREADSAFEQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

// Small generic mutex+condvar queue -- the only thing that crosses thread
// boundaries in this app (ImecFetchThread -> DecisionThread's SpikeEvent
// queue, NiFetchThread -> DecisionThread's SyllableEvent queue). Each
// producer thread only pushes; DecisionThread is the only consumer of
// either queue, so there's no multi-consumer contention to worry about.
template<typename T>
class ThreadSafeQueue {
public:
    void push( const T &item )
    {
        {
            std::lock_guard<std::mutex> lock( mutex_ );
            queue_.push( item );
        }
        cv_.notify_one();
    }

    // Blocks until an item is available or `timeoutMs` elapses. Returns
    // true and fills `out` if an item was popped, false on timeout (lets
    // DecisionThread periodically check a shared stop-flag instead of
    // blocking forever).
    bool waitPop( T &out, int timeoutMs )
    {
        std::unique_lock<std::mutex> lock( mutex_ );

        if( !cv_.wait_for( lock, std::chrono::milliseconds( timeoutMs ),
                           [this]{ return !queue_.empty(); } ) ) {
            return false;
        }

        out = queue_.front();
        queue_.pop();
        return true;
    }

    // Non-blocking pop. Returns false immediately when the queue is empty.
    //
    // Exists so a consumer can DRAIN a queue instead of taking one item per
    // iteration, which is not a cosmetic distinction here: DecisionThread's
    // loop used to take at most one event per pass and block up to 1 ms on
    // whichever queue was empty, capping it near 1000 events/s. A single
    // target unit fires at ~1 Hz so nothing ever showed, but the all-units
    // pipeline pushes every unit's detections into the same queue at
    // ~1500/s, which outruns that cap permanently -- the queue then grows
    // without bound and the decisions run on ever-staler events.
    bool tryPop( T &out )
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        if( queue_.empty() )
            return false;
        out = queue_.front();
        queue_.pop();
        return true;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return queue_.empty();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return queue_.size();
    }

private:
    mutable std::mutex     mutex_;
    std::condition_variable cv_;
    std::queue<T>           queue_;
};

#endif // CLOSEDLOOP_THREADSAFEQUEUE_H
