// bench_hotpath.cpp -- did wiring the two-queue split into the live path
// cost the hot path anything, and what does drop-oldest vs. blocking cost
// under overload.
//
// bench_queues.cpp already measured push-call cost in isolation. That is
// not the claim in play here: the claim is "sub 10 ms end-to-end,
// detection to decision", which is a property of the whole handoff --
// push AND the consumer noticing it -- not of push() alone. So this file
// times from the moment a spike is available to be pushed to the moment
// the consumer has it in hand, using the actual BEFORE and AFTER wiring
// shapes:
//
//   BEFORE  ImecFetchThreadCpu pushed ThreadSafeQueue<SpikeEvent> once per
//           spike; DecisionThread drained it with tryPop() in a loop.
//   AFTER   ImecFetchThreadCpu batches a chunk's detections and calls
//           SpikeQueue::push(ptr, n) once; DecisionThread drains with
//           waitDrain()+drain() in a loop (this file's DecisionThread.cpp
//           change).
//
// Three things are measured, each addressing one part of the task:
//
//   1. End-to-end PACED latency, BEFORE vs AFTER -- the sub-10ms claim.
//   2. AnalysisFeed's cost to the FETCH thread: the double->float copy into
//      the pool buffer that publish() requires.
//   3. FLOOD: SpikeQueue's drop-oldest overflow policy vs. a blocking
//      variant, to make the open design question's tradeoff a number
//      instead of an assertion.
//
// Build (PowerShell -- MSYS bash mangles /I):
//   cl.exe /EHsc /std:c++17 /O2 /arch:AVX2 /Fe:bench_hotpath.exe ClosedLoop\bench_hotpath.cpp
//
// Run: bench_hotpath.exe

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <condition_variable>

#include "Events.h"
#include "SpikeQueue.h"
#include "ThreadSafeQueue.h"

using Clock = std::chrono::steady_clock;

// ---- shared stats plumbing, same shape as bench_queues.cpp ---------------
struct Stats { double p50, p99, p999, max, mean; long long n; };

static Stats summarize( std::vector<double> &v )
{
    Stats s{};
    if( v.empty() ) return s;
    std::sort( v.begin(), v.end() );
    auto at = [&]( double q ) { return v[std::min( v.size() - 1,
                     static_cast<size_t>( q * v.size() ) )]; };
    s.p50 = at( 0.50 ); s.p99 = at( 0.99 ); s.p999 = at( 0.999 );
    s.max = v.back(); s.n = static_cast<long long>( v.size() );
    double sum = 0; for( double x : v ) sum += x;
    s.mean = sum / v.size();
    return s;
}

static void report( const char *name, const Stats &s )
{
    std::printf( "  %-32s n=%-7lld mean %8.3f  p50 %8.3f  p99 %8.3f  p99.9 %9.3f  max %10.3f us\n",
                 name, s.n, s.mean, s.p50, s.p99, s.p999, s.max );
}

static const int kEventsPerChunk = 150;   // ~157 units, 10 chunks/s -> ~1500/s
static const int kPacedChunks    = 200;   // 20 s at 10 chunks/s, matches bench_queues.cpp
static const int kFloodChunks    = 2000;

static void paceTo( Clock::time_point start, int chunkIdx )
{
    auto due = start + std::chrono::milliseconds( 100 * chunkIdx );
    std::this_thread::sleep_until( due );
}

// ===========================================================================
// 1. END-TO-END LATENCY: BEFORE (ThreadSafeQueue, per-spike push, tryPop
//    drain) vs AFTER (SpikeQueue, batched push, waitDrain/drain loop).
//
// Latency is measured from immediately before the event enters the queue
// (which is what "detected" means on the fetch thread) to the moment the
// consumer has it in a local buffer ready to run the decision rule on it --
// i.e. exactly the interval the <10ms claim is about. Each event carries a
// unique id in sampleIndex; pushTimestamps[id] records when it went in.
// ===========================================================================

static void runBefore_ThreadSafeQueue( int nChunks, bool paced, std::vector<double> &outLatUs )
{
    ThreadSafeQueue<SpikeEvent> q;
    std::atomic<bool> stop{ false };
    std::atomic<long long> nSeen{ 0 };
    const long long total = static_cast<long long>( nChunks ) * kEventsPerChunk;
    std::vector<Clock::time_point> pushT( static_cast<size_t>( total ) );
    std::mutex latMutex;
    outLatUs.clear();
    outLatUs.reserve( static_cast<size_t>( total ) );

    std::thread consumer( [&]{
        SpikeEvent e;
        while( !stop.load() ) {
            bool any = false;
            // Mirrors the OLD DecisionThread.cpp: drain completely with
            // tryPop() before considering a wait.
            while( q.tryPop( e ) ) {
                any = true;
                double us = std::chrono::duration<double, std::micro>(
                    Clock::now() - pushT[static_cast<size_t>( e.sampleIndex )] ).count();
                { std::lock_guard<std::mutex> lk( latMutex ); outLatUs.push_back( us ); }
                nSeen.fetch_add( 1 );
            }
            if( !any ) {
                // Mirrors the old waitPop(spk, 1) fallback -- one item,
                // 1 ms timeout.
                if( q.waitPop( e, 1 ) ) {
                    double us = std::chrono::duration<double, std::micro>(
                        Clock::now() - pushT[static_cast<size_t>( e.sampleIndex )] ).count();
                    { std::lock_guard<std::mutex> lk( latMutex ); outLatUs.push_back( us ); }
                    nSeen.fetch_add( 1 );
                }
            }
        }
        while( q.tryPop( e ) ) {
            double us = std::chrono::duration<double, std::micro>(
                Clock::now() - pushT[static_cast<size_t>( e.sampleIndex )] ).count();
            outLatUs.push_back( us );
            nSeen.fetch_add( 1 );
        }
    } );

    auto start = Clock::now();
    long long id = 0;
    for( int c = 0; c < nChunks; ++c ) {
        if( paced ) paceTo( start, c );
        // OLD ImecFetchThreadCpu: one push() call PER SPIKE.
        for( int i = 0; i < kEventsPerChunk; ++i ) {
            SpikeEvent ev;
            ev.unitId = i % 157;
            ev.sampleIndex = id;
            ev.timeRelSyncS = 0.001 * i;
            pushT[static_cast<size_t>( id )] = Clock::now();
            q.push( ev );
            ++id;
        }
    }
    while( nSeen.load() < total )
        std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
    stop.store( true );
    consumer.join();
}

static void runAfter_SpikeQueue( int nChunks, bool paced, std::vector<double> &outLatUs )
{
    SpikeQueue q;
    std::atomic<bool> stop{ false };
    std::atomic<long long> nSeen{ 0 };
    const long long total = static_cast<long long>( nChunks ) * kEventsPerChunk;
    std::vector<Clock::time_point> pushT( static_cast<size_t>( total ) );
    std::mutex latMutex;
    outLatUs.clear();
    outLatUs.reserve( static_cast<size_t>( total ) );

    std::thread consumer( [&]{
        std::vector<SpikeEvent> out;
        out.reserve( 8192 );
        while( !stop.load() ) {
            out.clear();
            // Mirrors the NEW DecisionThread.cpp: waitDrain then drain
            // until empty, THEN process everything gathered.
            if( q.waitDrain( out, 1 ) )
                while( q.drain( out ) ) {}
            for( size_t i = 0; i < out.size(); ++i ) {
                double us = std::chrono::duration<double, std::micro>(
                    Clock::now() - pushT[static_cast<size_t>( out[i].sampleIndex )] ).count();
                std::lock_guard<std::mutex> lk( latMutex );
                outLatUs.push_back( us );
            }
            nSeen.fetch_add( static_cast<long long>( out.size() ) );
        }
        out.clear();
        while( q.drain( out ) ) {}
        for( size_t i = 0; i < out.size(); ++i ) {
            double us = std::chrono::duration<double, std::micro>(
                Clock::now() - pushT[static_cast<size_t>( out[i].sampleIndex )] ).count();
            outLatUs.push_back( us );
        }
        nSeen.fetch_add( static_cast<long long>( out.size() ) );
    } );

    auto start = Clock::now();
    long long id = 0;
    std::vector<SpikeEvent> batch( kEventsPerChunk );
    for( int c = 0; c < nChunks; ++c ) {
        if( paced ) paceTo( start, c );
        // NEW ImecFetchThreadCpu: build the chunk's batch, ONE push() call.
        auto now = Clock::now();
        for( int i = 0; i < kEventsPerChunk; ++i ) {
            batch[i].unitId = i % 157;
            batch[i].sampleIndex = id;
            batch[i].timeRelSyncS = 0.001 * i;
            pushT[static_cast<size_t>( id )] = now;
            ++id;
        }
        q.push( batch.data(), batch.size() );
    }
    while( nSeen.load() < total )
        std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
    stop.store( true );
    consumer.join();
}

// ===========================================================================
// 2. AnalysisFeed's cost to the FETCH thread: the double->float conversion
//    copy into the pool buffer that publish() requires (see
//    ImecFetchThreadCpu.cpp's analysisFeed_ block). acquire()/publish()
//    themselves are already measured in bench_queues.cpp; this is the part
//    that file does not cover, because bench_queues.cpp's AnalysisFeed case
//    hands over an already-float buffer with no conversion.
// ===========================================================================

static void benchAnalysisFeedCopyCost()
{
    // fetchChunkMs=5, 30 kHz -> 150 samples/chunk; 96-channel CAR group, the
    // same defaults bench_queues.cpp and mainAllUnits.cpp's all-units config
    // use.
    const int nSamp = 150, nCh = 96;
    const size_t n = static_cast<size_t>( nSamp ) * nCh;

    std::vector<double> src( n );
    for( size_t i = 0; i < n; ++i ) src[i] = static_cast<double>( i ) * 0.001;
    std::vector<float> dst( n );

    const int kReps = 20000;
    std::vector<double> lat; lat.reserve( kReps );
    for( int r = 0; r < kReps; ++r ) {
        auto t0 = Clock::now();
        for( size_t i = 0; i < n; ++i )
            dst[i] = static_cast<float>( src[i] );
        lat.push_back( std::chrono::duration<double, std::micro>( Clock::now() - t0 ).count() );
    }
    Stats s = summarize( lat );
    std::printf( "\n=== 2. AnalysisFeed copy cost (double->float, %d samples x %d ch = %zu values) ===\n",
                 nSamp, nCh, n );
    report( "double->float copy", s );
    std::printf( "      at fetchChunkMs=5 (5000 us budget/chunk), this copy is %.2f%% of the budget"
                 " on its own -- %s\n",
                 100.0 * s.mean / 5000.0,
                 s.mean < 200.0 ? "negligible" : "NOT negligible, see report" );
}

// ===========================================================================
// 3. Blocking variant of SpikeQueue's ring, for the drop-oldest-vs-block
//    comparison the design question asks for. Same ring, same batched
//    push/drain shape as SpikeQueue.h, EXCEPT push() blocks (via a
//    "not full" condition variable) instead of overwriting the oldest
//    record when the ring is full. This is the "one-line change" the task
//    describes -- it is implemented here as its own class, deliberately,
//    rather than by editing SpikeQueue.h, so the shipped default stays
//    drop-oldest unless this measurement says otherwise.
// ===========================================================================

class SpikeQueueBlocking {
public:
    explicit SpikeQueueBlocking( size_t capacity = 65536 )
        : buf_( capacity ), head_( 0 ), tail_( 0 ), count_( 0 ), nPushed_( 0 ) {}

    void push( const SpikeEvent *e, size_t n )
    {
        std::unique_lock<std::mutex> lock( mutex_ );
        for( size_t i = 0; i < n; ++i ) {
            cvNotFull_.wait( lock, [this]{ return count_ < buf_.size(); } );
            buf_[head_] = e[i];
            head_ = ( head_ + 1 ) % buf_.size();
            ++count_;
            ++nPushed_;
        }
        lock.unlock();
        cv_.notify_one();
    }

    static const size_t kMaxDrain = 4096;

    size_t drain( std::vector<SpikeEvent> &out )
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return drainLocked( out );
    }

    size_t waitDrain( std::vector<SpikeEvent> &out, int timeoutMs )
    {
        std::unique_lock<std::mutex> lock( mutex_ );
        if( !cv_.wait_for( lock, std::chrono::milliseconds( timeoutMs ),
                           [this]{ return count_ > 0; } ) )
            return 0;
        return drainLocked( out );
    }

    long long nPushed() const { std::lock_guard<std::mutex> l( mutex_ ); return nPushed_; }
    size_t    size()     const { std::lock_guard<std::mutex> l( mutex_ ); return count_; }

private:
    size_t drainLocked( std::vector<SpikeEvent> &out )
    {
        size_t n = count_ < kMaxDrain ? count_ : kMaxDrain;
        for( size_t i = 0; i < n; ++i ) {
            out.push_back( buf_[tail_] );
            tail_ = ( tail_ + 1 ) % buf_.size();
        }
        count_ -= n;
        cvNotFull_.notify_all();
        return n;
    }

    std::vector<SpikeEvent> buf_;
    size_t head_, tail_, count_;
    mutable std::mutex mutex_;
    std::condition_variable cv_, cvNotFull_;
    long long nPushed_;
};

// Runs the FLOOD regime (unpaced, kFloodChunks) against both overflow
// policies and reports: (a) push-CALL latency -- what the policy costs the
// FETCH thread, which is the thing the whole split exists to protect; and
// (b) end-to-end staleness -- what the policy costs the DATA, i.e. how old
// an event is by the time the decision rule sees it.
static void benchOverflowPolicy()
{
    std::printf( "\n=== 3. Overflow policy under FLOOD (%d chunks, unpaced, %d events/chunk) ===\n",
                 kFloodChunks, kEventsPerChunk );
    std::printf( "      FLOOD is overload, not production -- see bench_queues.cpp's header for why\n"
                 "      only PACED numbers should be read as describing a healthy run. This section\n"
                 "      exists purely to compare the two overflow policies under the condition where\n"
                 "      they actually differ.\n\n" );

    const long long total = static_cast<long long>( kFloodChunks ) * kEventsPerChunk;

    // ---- drop-oldest (SpikeQueue as shipped) ----
    {
        SpikeQueue q( 4096 );   // smaller capacity than default so FLOOD actually saturates it
        std::atomic<bool> stop{ false };
        std::atomic<long long> nSeen{ 0 };
        std::vector<Clock::time_point> pushT( static_cast<size_t>( total ) );
        std::vector<double> endToEndUs; endToEndUs.reserve( static_cast<size_t>( total ) );
        std::mutex latMutex;

        std::thread consumer( [&]{
            std::vector<SpikeEvent> out; out.reserve( 8192 );
            while( !stop.load() ) {
                out.clear();
                if( q.waitDrain( out, 1 ) ) while( q.drain( out ) ) {}
                for( size_t i = 0; i < out.size(); ++i ) {
                    double us = std::chrono::duration<double, std::micro>(
                        Clock::now() - pushT[static_cast<size_t>( out[i].sampleIndex )] ).count();
                    std::lock_guard<std::mutex> lk( latMutex );
                    endToEndUs.push_back( us );
                }
                nSeen.fetch_add( static_cast<long long>( out.size() ) );
            }
        } );

        std::vector<double> pushLat; pushLat.reserve( kFloodChunks );
        std::vector<SpikeEvent> batch( kEventsPerChunk );
        long long id = 0;
        for( int c = 0; c < kFloodChunks; ++c ) {
            auto now = Clock::now();
            for( int i = 0; i < kEventsPerChunk; ++i ) {
                batch[i].unitId = i % 157; batch[i].sampleIndex = id; batch[i].timeRelSyncS = 0;
                pushT[static_cast<size_t>( id )] = now;
                ++id;
            }
            auto t0 = Clock::now();
            q.push( batch.data(), batch.size() );
            pushLat.push_back( std::chrono::duration<double, std::micro>( Clock::now() - t0 ).count() );
        }
        // Let the consumer drain whatever survived, then stop.
        std::this_thread::sleep_for( std::chrono::milliseconds( 200 ) );
        stop.store( true );
        consumer.join();

        Stats ps = summarize( pushLat );
        std::printf( "  drop-oldest (SpikeQueue, cap=4096):\n" );
        report( "  push-call latency", ps );
        std::printf( "    pushed=%lld dropped=%lld (%.1f%%) maxDepth=%zu\n",
                     q.nPushed(), q.nDropped(),
                     100.0 * static_cast<double>( q.nDropped() ) / static_cast<double>( q.nPushed() ),
                     q.maxDepth() );
        std::printf( "    delivered %zu of %lld events; of those, end-to-end latency:\n",
                     endToEndUs.size(), total );
        Stats es = summarize( endToEndUs );
        report( "  end-to-end (delivered)", es );
    }

    // ---- blocking ----
    {
        SpikeQueueBlocking q( 4096 );
        std::atomic<bool> stop{ false };
        std::atomic<long long> nSeen{ 0 };
        std::vector<Clock::time_point> pushT( static_cast<size_t>( total ) );
        std::vector<double> endToEndUs; endToEndUs.reserve( static_cast<size_t>( total ) );
        std::mutex latMutex;

        std::thread consumer( [&]{
            std::vector<SpikeEvent> out; out.reserve( 8192 );
            while( !stop.load() || q.size() > 0 ) {
                out.clear();
                if( q.waitDrain( out, 1 ) ) while( q.drain( out ) ) {}
                for( size_t i = 0; i < out.size(); ++i ) {
                    double us = std::chrono::duration<double, std::micro>(
                        Clock::now() - pushT[static_cast<size_t>( out[i].sampleIndex )] ).count();
                    std::lock_guard<std::mutex> lk( latMutex );
                    endToEndUs.push_back( us );
                }
                nSeen.fetch_add( static_cast<long long>( out.size() ) );
            }
        } );

        std::vector<double> pushLat; pushLat.reserve( kFloodChunks );
        std::vector<SpikeEvent> batch( kEventsPerChunk );
        long long id = 0;
        for( int c = 0; c < kFloodChunks; ++c ) {
            auto now = Clock::now();
            for( int i = 0; i < kEventsPerChunk; ++i ) {
                batch[i].unitId = i % 157; batch[i].sampleIndex = id; batch[i].timeRelSyncS = 0;
                pushT[static_cast<size_t>( id )] = now;
                ++id;
            }
            auto t0 = Clock::now();
            q.push( batch.data(), batch.size() );   // may BLOCK here -- that is the point
            pushLat.push_back( std::chrono::duration<double, std::micro>( Clock::now() - t0 ).count() );
        }
        stop.store( true );   // consumer keeps draining until q.size()==0, see loop condition
        consumer.join();

        Stats ps = summarize( pushLat );
        std::printf( "\n  blocking (never drops, cap=4096):\n" );
        report( "  push-call latency", ps );
        std::printf( "    pushed=%lld dropped=0 (blocks instead)\n", q.nPushed() );
        std::printf( "    delivered %zu of %lld events (all of them, eventually); end-to-end latency:\n",
                     endToEndUs.size(), total );
        Stats es = summarize( endToEndUs );
        report( "  end-to-end (delivered)", es );
    }
}

int main()
{
    std::printf( "bench_hotpath -- did the queue-split wiring cost the hot path anything\n" );
    std::printf( "budget: sub-10 ms detection -> decision, PACED (10 chunks/s, %d events/chunk)\n",
                 kEventsPerChunk );

    std::printf( "\n=== 1. End-to-end latency, PACED (%d chunks, 20 s) ===\n", kPacedChunks );
    {
        std::vector<double> before, after;
        runBefore_ThreadSafeQueue( kPacedChunks, /*paced=*/true, before );
        runAfter_SpikeQueue( kPacedChunks, /*paced=*/true, after );
        report( "BEFORE (ThreadSafeQueue)", summarize( before ) );
        report( "AFTER  (SpikeQueue)",      summarize( after ) );

        Stats sb = summarize( before ), sa = summarize( after );
        std::printf( "      10 ms budget: BEFORE p99.9=%.3f ms (%s)  AFTER p99.9=%.3f ms (%s)\n",
                     sb.p999 / 1000.0, sb.p999 < 10000.0 ? "OK" : "OVER",
                     sa.p999 / 1000.0, sa.p999 < 10000.0 ? "OK" : "OVER" );
    }

    benchAnalysisFeedCopyCost();
    benchOverflowPolicy();

    return 0;
}
