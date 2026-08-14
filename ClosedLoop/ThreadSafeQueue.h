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

    bool empty() const
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return queue_.empty();
    }

private:
    mutable std::mutex     mutex_;
    std::condition_variable cv_;
    std::queue<T>           queue_;
};

#endif // CLOSEDLOOP_THREADSAFEQUEUE_H
