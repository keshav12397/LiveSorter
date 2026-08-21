#ifndef CLOSEDLOOP_ANALYSISFEED_H
#define CLOSEDLOOP_ANALYSISFEED_H

#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>
#include <cstddef>

#include "Events.h"

// The SLOW queue: preprocessed sample data for drift estimation and
// plotting. The other half of the split SpikeQueue.h describes.
//
//   SpikeQueue     spike times only, target < 10 ms, must not drop.
//   AnalysisFeed   bulk samples, budget > 500 ms, drops freely.
//
// Different guarantees, no shared mutex, no shared state. A slow drift
// estimator or a stalled viewer cannot reach the decision path through this
// class, because the only thing the fetch thread ever does here is take a
// free buffer, fill it, and hand back a pointer.
//
// Why a buffer POOL and not a queue of copies
// -------------------------------------------
// Drift estimation needs sample data, not spike times: a centroid is an
// amplitude-weighted average over channels, so the analysis side has to see
// waveforms. Three ways to get them there, and only one is acceptable on a
// thread with a 10 ms budget:
//
//   - Per-spike snippets. 61 samples x 96 channels x 4 B = 23 KB per spike,
//     at ~1500 spikes/s = 35 MB/s of copying, and the same sample gets
//     copied once per overlapping spike. Rejected.
//   - Copy each preprocessed chunk into the queue. ~1.15 MB per chunk
//     (3000 samples x 96 ch x float), ~100 us of memcpy per chunk on the
//     fetch thread. Bounded and probably affordable, but it is pure waste:
//     the data already exists in a buffer the fetch thread owns.
//   - Hand over the buffer itself and take a fresh one from a pool. No copy
//     at all; the fetch thread's per-chunk cost is a pointer swap under a
//     mutex. This is what is implemented.
//
// The pool is what makes dropping automatic and bounded. `acquire()` returns
// null when every buffer is checked out, which means the analysis thread is
// behind; the fetch thread then simply does not publish that chunk and
// carries on. There is no allocation to fail, no growth, and no waiting --
// falling behind costs the analysis side resolution, never the fetch side
// time.
//
// Sizing: `nBuffers` buffers of `samplesPerBuffer * nChannels` floats each.
// The default 8 x (3000 x 96 x 4 B) is ~9 MB and tolerates the analysis
// thread being ~0.8 s behind at 10 chunks/s before anything is dropped,
// which is the >500 ms budget with room to spare.
//
// Ownership contract
// ------------------
// `acquire()` -> fetch thread fills -> `publish()` hands it to the queue ->
// analysis thread `take()`s it -> analysis thread MUST `release()` it.
// A buffer that is never released is leaked out of the pool permanently and
// the feed silently degrades to dropping everything, so release() belongs in
// a scope guard on the consumer side. `nInFlight()` exists to catch exactly
// that: it should return to 0 when the analysis thread is idle.
class AnalysisFeed {
public:
    struct Chunk {
        float    *samples;      // [nSamples * nChannels], channel-major per sample
        int       nSamples;
        int       nChannels;
        long long firstSampleIndex;   // absolute index of samples[0], IMEC clock
        double    timeRelSyncS;       // of the first sample, same convention as SpikeEvent
        int       poolIndex;          // for release(); consumers must not change it

        // Detections that fell inside this chunk, copied by value. Small
        // (a few dozen at 157 units) and it saves the analysis side from
        // having to re-derive which spikes belong to which samples.
        std::vector<SpikeEvent> spikes;
    };

    AnalysisFeed( int samplesPerBuffer, int nChannels, int nBuffers = 8 )
        : samplesPerBuffer_( samplesPerBuffer ), nChannels_( nChannels ),
          storage_( static_cast<size_t>( samplesPerBuffer ) * nChannels * nBuffers ),
          free_( nBuffers ), nPublished_( 0 ), nDroppedNoBuffer_( 0 )
    {
        for( int i = 0; i < nBuffers; ++i )
            free_[i] = i;
    }

    // Returns a buffer to fill, or nullptr if the analysis thread is behind.
    // nullptr is NORMAL and means "skip this chunk" -- never a reason to
    // wait, allocate, or retry.
    float *acquire( int &poolIndexOut )
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        if( free_.empty() ) {
            nDroppedNoBuffer_.fetch_add( 1 );
            poolIndexOut = -1;
            return nullptr;
        }
        int idx = free_.back();
        free_.pop_back();
        poolIndexOut = idx;
        return bufferAt( idx );
    }

    // Hands a filled buffer to the consumer. Takes the Chunk by value and
    // moves it, so the caller's `spikes` vector is not copied again.
    void publish( Chunk c )
    {
        {
            std::lock_guard<std::mutex> lock( mutex_ );
            queue_.push_back( std::move( c ) );
        }
        nPublished_.fetch_add( 1 );
        cv_.notify_one();
    }

    // Blocks up to timeoutMs. Returns false on timeout so the consumer can
    // check a stop flag.
    bool take( Chunk &out, int timeoutMs )
    {
        std::unique_lock<std::mutex> lock( mutex_ );
        if( !cv_.wait_for( lock, std::chrono::milliseconds( timeoutMs ),
                           [this]{ return !queue_.empty(); } ) ) {
            return false;
        }
        out = std::move( queue_.front() );
        queue_.erase( queue_.begin() );
        return true;
    }

    // MUST be called for every chunk taken, or the pool drains permanently.
    void release( int poolIndex )
    {
        if( poolIndex < 0 )
            return;
        std::lock_guard<std::mutex> lock( mutex_ );
        free_.push_back( poolIndex );
    }

    // The buffer size every chunk is sized against. A consumer that
    // needs to retain history across chunks must size from THIS, not
    // from the first chunk it happens to see -- the first chunk of a
    // run is routinely a short partial one.
    int samplesPerBuffer() const { return samplesPerBuffer_; }

    int nInFlight() const
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return static_cast<int>( nBuffers() - free_.size() );
    }

    long long nPublished() const { return nPublished_.load(); }
    long long nDroppedNoBuffer() const { return nDroppedNoBuffer_.load(); }

    std::string summary() const
    {
        std::string s = "AnalysisFeed: " + std::to_string( nPublished() )
                      + " chunks published, "
                      + std::to_string( nDroppedNoBuffer() ) + " skipped (pool dry)";
        if( nDroppedNoBuffer() > 0 )
            s += "  <-- analysis thread fell behind; drift/plot resolution reduced,"
                 " detection unaffected";
        return s;
    }

private:
    size_t nBuffers() const
    { return storage_.size() / ( static_cast<size_t>( samplesPerBuffer_ ) * nChannels_ ); }

    float *bufferAt( int i )
    { return storage_.data() + static_cast<size_t>( i ) * samplesPerBuffer_ * nChannels_; }

    int samplesPerBuffer_, nChannels_;
    std::vector<float> storage_;      // one allocation, at construction
    std::vector<int>   free_;         // indices of buffers not checked out
    std::vector<Chunk> queue_;        // filled, waiting for the consumer

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<long long> nPublished_, nDroppedNoBuffer_;
};

#endif // CLOSEDLOOP_ANALYSISFEED_H
