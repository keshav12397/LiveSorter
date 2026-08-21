// bench_queues.cpp -- what the hot/slow queue split actually costs.
//
// Two claims are made in SpikeQueue.h and AnalysisFeed.h, and neither is
// worth anything unmeasured. This project has three times now had a
// confident guess about a bottleneck overturned by a profile, so:
//
//   1. ThreadSafeQueue<SpikeEvent> allocates on every push, because MSVC
//      sizes std::deque blocks in ELEMENTS and SpikeEvent is 24 bytes.
//   2. SpikeQueue never allocates after construction, and its push latency
//      tail is what a <10 ms decision budget needs.
//
// The number that matters is not mean push cost -- it is the TAIL. A mean
// hides exactly the stall a closed loop cares about, since one 12 ms
// allocation pause in a hundred thousand fast pushes still misses a
// decision window. Reported as p50/p99/p99.9/max.
//
// Build (PowerShell -- MSYS bash mangles /I):
//   cl.exe /EHsc /std:c++17 /O2 /Fe:bench_queues.exe ClosedLoop\bench_queues.cpp
//
// Run: bench_queues.exe

#include <cstdio>
#include <cstdlib>
#include <new>
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

#include "Events.h"
#include "SpikeQueue.h"
#include "AnalysisFeed.h"
#include "ThreadSafeQueue.h"

// ------------------------------------------------------------------ //
// Global allocation counter. Same technique profile_detection_path.cpp
// uses: replacing operator new is the only way to see allocations a
// container makes on your behalf, which is precisely the cost being
// hunted here -- nothing in the calling code looks like an allocation.
// ------------------------------------------------------------------ //
static std::atomic<long long> g_allocs{ 0 };
static std::atomic<bool>      g_counting{ false };

void *operator new( size_t n )
{
    if( g_counting.load( std::memory_order_relaxed ) )
        g_allocs.fetch_add( 1, std::memory_order_relaxed );
    void *p = std::malloc( n ? n : 1 );
    if( !p ) throw std::bad_alloc();
    return p;
}
void operator delete( void *p ) noexcept { std::free( p ); }
void operator delete( void *p, size_t ) noexcept { std::free( p ); }

using Clock = std::chrono::steady_clock;
static double usSince( Clock::time_point t0 )
{
    return std::chrono::duration<double, std::micro>( Clock::now() - t0 ).count();
}

struct Stats {
    double p50, p99, p999, max, mean;
};

static Stats summarize( std::vector<double> &v )
{
    std::sort( v.begin(), v.end() );
    Stats s;
    auto at = [&]( double q ) { return v[std::min( v.size() - 1,
                     static_cast<size_t>( q * v.size() ) )]; };
    s.p50 = at( 0.50 ); s.p99 = at( 0.99 ); s.p999 = at( 0.999 );
    s.max = v.back();
    double sum = 0; for( double x : v ) sum += x;
    s.mean = sum / v.size();
    return s;
}

static void report( const char *name, Stats s, long long allocs, long long n )
{
    std::printf( "  %-26s mean %7.3f  p50 %7.3f  p99 %7.3f  p99.9 %7.3f  max %8.3f us"
                 "   allocs %lld (%.2f/push)\n",
                 name, s.mean, s.p50, s.p99, s.p999, s.max, allocs,
                 static_cast<double>( allocs ) / n );
}

// A realistic detection burst: at 157 units and ~1500 detections/s with
// ~10 chunks/s, a chunk carries roughly 150 events.
static const int kEventsPerChunk = 150;

// Two regimes, and reporting only one of them would be misleading.
//
// PACED is production: 10 chunks/s, ~1500 events/s. These are the latency
// numbers that describe a healthy run, and the only ones a <10 ms budget
// should be judged against.
//
// FLOOD is overload: push as fast as the producer can, ~30x production
// rate. Nobody runs here, but it is where the two designs differ in KIND
// rather than degree. An unbounded queue answers overload with unbounded
// latency -- the decisions keep running, on ever-staler spikes, which for a
// closed loop is not a slow decision but a wrong one. The bounded ring
// answers it by dropping and SAYING SO. The flood numbers exist to show
// that, not to characterise normal operation.
//
// An earlier version of this file measured only the flood and reported its
// p99.9 as if it described production. It did not: the queue was pinned at
// capacity for the whole run.
enum Regime { kPaced, kFlood };

static const int kPacedChunks = 200;    // 20 s at 10 chunks/s
static const int kFloodChunks = 2000;

static void pace( Regime r, Clock::time_point start, int chunkIdx )
{
    if( r != kPaced )
        return;
    // Chunk n is due at start + n*100ms. Sleeping to an ABSOLUTE deadline
    // rather than for a fixed interval keeps the rate honest when a chunk
    // takes a while -- otherwise every slow chunk permanently delays the
    // schedule and the measured rate drifts below the one being claimed.
    auto due = start + std::chrono::milliseconds( 100 * chunkIdx );
    std::this_thread::sleep_until( due );
}

template<typename PushFn>
static void runCase( const char *name, Regime regime, int nChunks,
                     PushFn pushChunk, std::function<std::string()> summary )
{
    std::vector<double> lat;
    lat.reserve( nChunks );
    auto start = Clock::now();
    g_allocs.store( 0 );
    g_counting.store( true );
    for( int c = 0; c < nChunks; ++c ) {
        pace( regime, start, c );
        auto t0 = Clock::now();
        pushChunk();
        lat.push_back( usSince( t0 ) );
    }
    g_counting.store( false );
    long long a = g_allocs.load();
    report( name, summarize( lat ), a,
            static_cast<long long>( nChunks ) * kEventsPerChunk );
    if( summary )
        std::printf( "      %s\n", summary().c_str() );
}

int main()
{
    std::printf( "bench_queues -- hot/slow queue split\n\n" );
    std::printf( "sizeof(SpikeEvent) = %zu bytes  ->  MSVC _DEQUESIZ = %d element(s)/block\n",
                 sizeof( SpikeEvent ),
                 sizeof( SpikeEvent ) <= 1 ? 16 : sizeof( SpikeEvent ) <= 2 ? 8 :
                 sizeof( SpikeEvent ) <= 4 ? 4  : sizeof( SpikeEvent ) <= 8 ? 2 : 1 );
    std::printf( "latency is per CHUNK of %d events; budget is 10 ms\n", kEventsPerChunk );

    std::vector<SpikeEvent> batch( kEventsPerChunk );
    for( int i = 0; i < kEventsPerChunk; ++i ) {
        batch[i].timeRelSyncS = 0.001 * i;
        batch[i].sampleIndex  = 30000LL * i;
        batch[i].unitId       = i % 157;
    }

    for( int r = 0; r < 2; ++r ) {
        Regime regime = r == 0 ? kPaced : kFlood;
        int nChunks   = r == 0 ? kPacedChunks : kFloodChunks;
        std::printf( "\n=== %s (%d chunks%s) ===\n",
                     r == 0 ? "PACED -- production, 10 chunks/s"
                            : "FLOOD -- overload, unpaced",
                     nChunks, r == 0 ? ", 20 s" : "" );

        {   // the existing queue
            ThreadSafeQueue<SpikeEvent> q;
            std::atomic<bool> stop{ false };
            std::thread consumer( [&]{
                SpikeEvent e;
                while( !stop.load() ) {
                    if( !q.tryPop( e ) )
                        std::this_thread::sleep_for( std::chrono::microseconds( 50 ) );
                }
                while( q.tryPop( e ) ) {}
            } );
            runCase( "ThreadSafeQueue (deque)", regime, nChunks,
                     [&]{ for( int i = 0; i < kEventsPerChunk; ++i ) q.push( batch[i] ); },
                     nullptr );
            stop.store( true ); consumer.join();
        }

        {   // hot queue, batched -- the form the fetch loop should use
            SpikeQueue q;
            std::atomic<bool> stop{ false };
            std::thread consumer( [&]{
                std::vector<SpikeEvent> out; out.reserve( 8192 );
                while( !stop.load() ) {
                    out.clear();
                    // drain() is capped at kMaxDrain, so the consumer loops
                    // until empty rather than assuming one call suffices.
                    if( q.waitDrain( out, 2 ) )
                        while( q.drain( out ) ) out.clear();
                }
                out.clear(); while( q.drain( out ) ) out.clear();
            } );
            runCase( "SpikeQueue (batched)", regime, nChunks,
                     [&]{ q.push( batch.data(), batch.size() ); },
                     [&]{ return q.summary(); } );
            stop.store( true ); consumer.join();
        }

        {   // the slow feed's cost to the FETCH thread
            const int nSamp = 3000, nCh = 96;
            AnalysisFeed feed( nSamp, nCh, 8 );
            std::atomic<bool> stop{ false };
            std::thread consumer( [&]{
                AnalysisFeed::Chunk c;
                while( !stop.load() ) {
                    if( feed.take( c, 2 ) ) {
                        // A deliberately slow consumer: 50 ms per chunk, well
                        // inside the >500 ms budget and 500x the fetch
                        // thread's own handoff cost.
                        std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
                        feed.release( c.poolIndex );
                    }
                }
                while( feed.take( c, 0 ) ) feed.release( c.poolIndex );
            } );
            runCase( "AnalysisFeed (pool handoff)", regime, nChunks,
                     [&]{
                         int idx = -1;
                         float *buf = feed.acquire( idx );
                         if( buf ) {
                             AnalysisFeed::Chunk ch;
                             ch.samples = buf; ch.nSamples = nSamp; ch.nChannels = nCh;
                             ch.firstSampleIndex = 0; ch.timeRelSyncS = 0.0;
                             ch.poolIndex = idx;
                             feed.publish( std::move( ch ) );
                         }
                     },
                     [&]{ return feed.summary(); } );
            stop.store( true ); consumer.join();
            std::printf( "      in flight at end: %d (must be 0 -- else a buffer leaked)\n",
                         feed.nInFlight() );
        }
    }

    std::printf( "\nCopying a chunk instead of handing over the pool buffer would be"
                 " %.2f MB per chunk.\n", 3000 * 96 * 4.0 / 1e6 );
    return 0;
}
