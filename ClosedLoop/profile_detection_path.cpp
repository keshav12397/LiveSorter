// Where does the per-sample cost of the CPU detection path actually go?
//
// bench_multi_convolution.exe answers "does it keep up"; this answers "and
// what is it spending the time on", which turned out to be a different
// question. After vectorising the convolution the engine still ran at ~7% of
// fp32 peak, and three successive guesses at the reason (auto-vectorisation,
// the scalar tail, cache-line amplification in the gather) were right, right,
// and wrong respectively. So this measures instead.
//
// The path is split into the three phases it really has, timed separately on
// one thread so nothing is hidden behind parallelism:
//
//   transpose  once per chunk   time-major float64 -> channel-major float32
//   computeD   once per unit    FastMatchedFilter: gather + matched filter
//   decide     once per unit    ConvolutionEngine::processPrecomputedD, i.e.
//                               the windowed NMS, its std::deque D-buffer and
//                               whatever it allocates
//
// It also counts heap allocations, because "decideUpTo allocates several
// vectors per call" is a specific claim that should be checked rather than
// asserted -- a global operator new counter turns it into a number.
//
// Run:
//   profile_detection_path.exe [units] [chunkSamples]

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <random>
#include <vector>

#include "MultiFilterBank.h"
#include "FastMatchedFilter.h"
#include "ConvolutionEngine.h"

// ---- allocation counter -------------------------------------------------
// Deliberately global and deliberately crude. Counting is enough; this is not
// trying to be a heap profiler, only to settle whether the decision phase
// allocates per call and how much.
static std::atomic<long long> g_allocs( 0 );
static std::atomic<bool>      g_counting( false );

void *operator new( size_t n )
{
    if( g_counting.load( std::memory_order_relaxed ) )
        g_allocs.fetch_add( 1, std::memory_order_relaxed );
    void *p = std::malloc( n ? n : 1 );
    if( !p ) throw std::bad_alloc();
    return p;
}
void *operator new[]( size_t n ) { return operator new( n ); }
void operator delete( void *p ) noexcept { std::free( p ); }
void operator delete[]( void *p ) noexcept { std::free( p ); }
void operator delete( void *p, size_t ) noexcept { std::free( p ); }
void operator delete[]( void *p, size_t ) noexcept { std::free( p ); }

namespace {

const int    kTemplateLen   = 61;
const int    kChansPerUnit  = 6;
const int    kChannelsGroup = 384;
const double kSampleRate    = 30000.0;
const double kSecondsOfData = 2.0;

struct Timer {
    std::chrono::steady_clock::time_point t0;
    void start() { t0 = std::chrono::steady_clock::now(); }
    double stop() const
    {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0 ).count();
    }
};

} // namespace


int main( int argc, char **argv )
{
    const int    nUnits       = ( argc > 1 ) ? std::atoi( argv[1] ) : 1000;
    const size_t chunkSamples = ( argc > 2 ) ? static_cast<size_t>( std::atoi( argv[2] ) ) : 150;
    const size_t nSamples     = static_cast<size_t>( kSecondsOfData * kSampleRate );

    std::mt19937 rng( 7 );
    std::normal_distribution<double> gauss( 0.0, 1.0 );
    std::uniform_int_distribution<int> chanPick( 0, kChannelsGroup - 1 );

    std::vector<int>     unitIds( nUnits );
    std::vector<int32_t> channels( static_cast<size_t>( nUnits ) * kChansPerUnit );
    std::vector<float>   filters( static_cast<size_t>( nUnits ) * kTemplateLen * kChansPerUnit );
    std::vector<float>   thresholds( nUnits );
    for( int u = 0; u < nUnits; ++u ) {
        unitIds[u] = u;
        for( int c = 0; c < kChansPerUnit; ++c )
            channels[static_cast<size_t>( u ) * kChansPerUnit + c] = chanPick( rng );
        for( int i = 0; i < kTemplateLen * kChansPerUnit; ++i )
            filters[static_cast<size_t>( u ) * kTemplateLen * kChansPerUnit + i] =
                static_cast<float>( gauss( rng ) * 0.1 );
        thresholds[u] = 6.0f;
    }
    MultiFilterBank bank = MultiFilterBank::fromHostArrays(
        unitIds, channels, filters, thresholds, kChansPerUnit, kTemplateLen );

    std::vector<double> data( nSamples * kChannelsGroup );
    for( size_t i = 0; i < data.size(); ++i )
        data[i] = gauss( rng );

    std::vector<ConvolutionEngine> engines;
    std::vector<FastMatchedFilter> fast;
    std::vector<std::vector<double>> dScratch( nUnits );
    engines.reserve( nUnits );
    fast.reserve( nUnits );
    for( int u = 0; u < nUnits; ++u ) {
        const float *f = bank.unitFilter( u );
        std::vector<double> taps( static_cast<size_t>( kTemplateLen ) * kChansPerUnit );
        for( size_t i = 0; i < taps.size(); ++i ) taps[i] = f[i];
        engines.emplace_back( kTemplateLen, kChansPerUnit, taps, kTemplateLen / 2 );
        fast.emplace_back( kTemplateLen, kChansPerUnit, f,
                           engines[u].leftMargin(), engines[u].rightMargin() );
    }

    std::vector<float> groupT( static_cast<size_t>( kChannelsGroup ) * chunkSamples );

    double tTranspose = 0.0, tComputeD = 0.0, tDecide = 0.0;
    long long nChunks = 0, nDetect = 0, nDValues = 0;
    Timer tm;

    // One untimed pass so every buffer reaches its final size first --
    // otherwise first-call allocation is charged to steady state.
    {
        long long firstD = 0;
        for( size_t t = 0; t < chunkSamples; ++t )
            for( int ch = 0; ch < kChannelsGroup; ++ch )
                groupT[static_cast<size_t>( ch ) * chunkSamples + t] =
                    static_cast<float>( data[t * kChannelsGroup + ch] );
        for( int u = 0; u < nUnits; ++u ) {
            size_t nD = fast[u].computeD( groupT.data(), chunkSamples, chunkSamples,
                                          bank.unitChannels( u ), 0, dScratch[u], &firstD );
            if( nD ) engines[u].processPrecomputedD( dScratch[u].data(), nD, firstD );
        }
    }

    g_allocs.store( 0 );
    g_counting.store( true );

    for( size_t off = 0; off + chunkSamples <= nSamples; off += chunkSamples ) {
        const double *src = data.data() + off * kChannelsGroup;

        tm.start();
        for( size_t t = 0; t < chunkSamples; ++t ) {
            const double *row = src + t * kChannelsGroup;
            for( int ch = 0; ch < kChannelsGroup; ++ch )
                groupT[static_cast<size_t>( ch ) * chunkSamples + t] =
                    static_cast<float>( row[ch] );
        }
        tTranspose += tm.stop();

        std::vector<size_t>    nDs( nUnits );
        std::vector<long long> firstDs( nUnits );

        tm.start();
        for( int u = 0; u < nUnits; ++u )
            nDs[u] = fast[u].computeD( groupT.data(), chunkSamples, chunkSamples,
                                       bank.unitChannels( u ),
                                       static_cast<long long>( off ),
                                       dScratch[u], &firstDs[u] );
        tComputeD += tm.stop();

        tm.start();
        for( int u = 0; u < nUnits; ++u ) {
            if( !nDs[u] ) continue;
            std::vector<PeakEvent> pk = engines[u].processPrecomputedD(
                dScratch[u].data(), nDs[u], firstDs[u] );
            nDetect += static_cast<long long>( pk.size() );
            nDValues += static_cast<long long>( nDs[u] );
        }
        tDecide += tm.stop();
        ++nChunks;
    }

    g_counting.store( false );
    const long long allocs = g_allocs.load();

    const double total = tTranspose + tComputeD + tDecide;
    const long long unitChunks = nChunks * nUnits;

    std::printf( "%d units x %d ch x %d taps, chunk %zu, %.1f s of data, 1 thread\n",
                 nUnits, kChansPerUnit, kTemplateLen, chunkSamples, kSecondsOfData );
    std::printf( "%lld chunks, %lld unit-chunks, %lld D values, %lld detections\n\n",
                 nChunks, unitChunks, nDValues, nDetect );

    std::printf( "  %-12s %9s %7s %14s\n", "phase", "wall(s)", "share", "ns/unit-chunk" );
    struct Row { const char *n; double s; };
    Row rows[3] = { { "transpose", tTranspose }, { "computeD", tComputeD },
                    { "decide", tDecide } };
    for( int i = 0; i < 3; ++i )
        std::printf( "  %-12s %9.3f %6.1f%% %14.0f\n", rows[i].n, rows[i].s,
                     100.0 * rows[i].s / total,
                     1e9 * rows[i].s / static_cast<double>( unitChunks ) );
    std::printf( "  %-12s %9.3f %6.1f%%   realtime x%.2f\n\n", "TOTAL", total, 100.0,
                 kSecondsOfData / total );

    std::printf( "heap allocations: %lld total, %.1f per unit-chunk\n",
                 allocs, static_cast<double>( allocs ) / static_cast<double>( unitChunks ) );
    std::printf( "  (the decide phase is the suspect -- processPrecomputedD returns a\n"
                 "   vector and decideUpTo builds several more per call)\n" );
    return 0;
}
