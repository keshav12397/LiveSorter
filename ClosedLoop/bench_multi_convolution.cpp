// How does the CPU detection path scale with unit count?
//
// The question this answers is whether ClosedLoopAllUnits.exe keeps up with
// real time at a given bank size, on this machine, with the engine as
// actually written -- not with an idealised FLOP count. Those differ a lot
// here, because the arithmetic is trivially small (a 1000-unit x 6-channel
// x 61-tap bank is 22 GFLOP/s at 30 kHz, well inside one core's theoretical
// reach) while ConvolutionEngine's inner loop is shaped for clarity rather
// than for SIMD: it accumulates across a unit's few channels, which is a
// short contiguous run ending in a horizontal add, instead of across output
// samples, which would fill vector registers. So the headline number to read
// is not GFLOP/s, it is the REALTIME FACTOR.
//
// Realtime factor = (seconds of neural data) / (seconds of wall time). 1.0 is
// exactly keeping up and is not good enough; the fetch loop, preprocessing,
// the decision thread, the viewer, and SpikeGLX itself all want the same
// cores.
//
// Preprocessing is measured separately rather than folded in. It runs over
// the whole CAR group and so scales with channel count, not unit count --
// mixing the two would hide which one a slowdown came from.
//
// Run:
//   bench_multi_convolution.exe [chunkSamples] [maxThreads]

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>

#include "MultiFilterBank.h"
#include "MultiConvolutionEngine.h"
#include "Preprocessor.h"

namespace {

const int    kTemplateLen  = 61;
const double kSampleRate   = 30000.0;
const double kSecondsOfData = 2.0;

MultiFilterBank makeBank( int nUnits, int nChansPerUnit, int nChannelsGroup,
                          std::mt19937 &rng )
{
    std::normal_distribution<double> gauss( 0.0, 1.0 );
    std::uniform_int_distribution<int> chanPick( 0, nChannelsGroup - 1 );

    std::vector<int>     unitIds( nUnits );
    std::vector<int32_t> channels( static_cast<size_t>( nUnits ) * nChansPerUnit );
    std::vector<float>   filters( static_cast<size_t>( nUnits ) * kTemplateLen * nChansPerUnit );
    std::vector<float>   thresholds( nUnits );

    for( int u = 0; u < nUnits; ++u ) {
        unitIds[u] = u;
        for( int c = 0; c < nChansPerUnit; ++c )
            channels[static_cast<size_t>( u ) * nChansPerUnit + c] = chanPick( rng );
        for( int i = 0; i < kTemplateLen * nChansPerUnit; ++i )
            filters[static_cast<size_t>( u ) * kTemplateLen * nChansPerUnit + i] =
                static_cast<float>( gauss( rng ) * 0.1 );
        // A realistic live threshold: high enough that detections are rare,
        // so the NMS decision chain does the amount of work it does in a real
        // session. Setting it low would benchmark a pathological case.
        thresholds[u] = 6.0f;
    }
    return MultiFilterBank::fromHostArrays( unitIds, channels, filters, thresholds,
                                            nChansPerUnit, kTemplateLen );
}

double benchEngine( int nUnits, int nChansPerUnit, int nChannelsGroup,
                    const std::vector<double> &data, size_t nSamples,
                    size_t chunkSamples, int nThreads, std::mt19937 &rng,
                    long long *detectionsOut )
{
    MultiFilterBank bank = makeBank( nUnits, nChansPerUnit, nChannelsGroup, rng );
    MultiConvolutionEngine eng( bank, nChannelsGroup, kTemplateLen / 2, nThreads );

    // One untimed chunk first: the very first call is where every per-unit
    // gather buffer is allocated to its final size, and charging that to the
    // measurement would report a startup cost as a steady-state one.
    eng.processChunk( data.data(), chunkSamples, 0 );

    long long nDet = 0;
    auto t0 = std::chrono::steady_clock::now();
    for( size_t off = 0; off + chunkSamples <= nSamples; off += chunkSamples ) {
        std::vector<MultiPeakEvent> d = eng.processChunk(
            data.data() + off * nChannelsGroup, chunkSamples,
            static_cast<long long>( off ) );
        nDet += static_cast<long long>( d.size() );
    }
    auto t1 = std::chrono::steady_clock::now();
    *detectionsOut = nDet;
    return std::chrono::duration<double>( t1 - t0 ).count();
}

} // namespace


int main( int argc, char **argv )
{
    const size_t chunkSamples = ( argc > 1 ) ? static_cast<size_t>( std::atoi( argv[1] ) ) : 150;
    const int    maxThreads   = ( argc > 2 ) ? std::atoi( argv[2] ) : 0;

    const int nChannelsGroup = 384;   // full probe: what a 1000-unit bank implies
    const size_t nSamples = static_cast<size_t>( kSecondsOfData * kSampleRate );

    std::mt19937 rng( 7 );
    std::normal_distribution<double> gauss( 0.0, 1.0 );

    std::printf( "chunk %zu samples (%.2f ms), CAR group %d ch, %.1f s of data, "
                 "hw_concurrency %u\n\n",
                 chunkSamples, 1000.0 * chunkSamples / kSampleRate, nChannelsGroup,
                 kSecondsOfData, std::thread::hardware_concurrency() );

    std::vector<double> data( nSamples * nChannelsGroup );
    for( size_t i = 0; i < data.size(); ++i )
        data[i] = gauss( rng );

    // Preprocessing, measured on its own: it scales with the CAR group, not
    // with unit count, so it is a fixed tax the table below does not include.
    {
        std::vector<short> raw( nSamples * nChannelsGroup );
        for( size_t i = 0; i < raw.size(); ++i )
            raw[i] = static_cast<short>( gauss( rng ) * 50.0 );
        Preprocessor pre( nChannelsGroup, 300.0, kSampleRate, true, true );
        auto t0 = std::chrono::steady_clock::now();
        for( size_t off = 0; off + chunkSamples <= nSamples; off += chunkSamples )
            pre.processChunk( raw.data() + off * nChannelsGroup, chunkSamples );
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>( t1 - t0 ).count();
        std::printf( "preprocessing (highpass + median CAR, %d ch, single-threaded):\n"
                     "  %.3f s wall for %.1f s of data  ->  realtime x%.2f\n\n",
                     nChannelsGroup, s, kSecondsOfData, kSecondsOfData / s );
    }

    struct Case { int units; int chans; };
    std::vector<Case> cases = {
        { 157, 5 },    // what the GPU build actually ran
        { 250, 6 },
        { 500, 6 },
        { 1000, 6 },   // the question
        { 2000, 6 },
    };

    int hw = static_cast<int>( std::thread::hardware_concurrency() );
    if( hw <= 0 ) hw = 1;
    std::vector<int> threadCounts = { 1, maxThreads > 0 ? maxThreads : hw };

    for( size_t ti = 0; ti < threadCounts.size(); ++ti ) {
        int nt = threadCounts[ti];
        if( ti > 0 && nt == threadCounts[0] ) continue;
        std::printf( "matched filter + NMS, %d thread(s):\n", nt );
        std::printf( "  %6s %6s %10s %10s %9s %10s\n",
                     "units", "chans", "GMAC/s", "wall(s)", "realtime", "det/s" );
        for( size_t i = 0; i < cases.size(); ++i ) {
            long long nDet = 0;
            double s = benchEngine( cases[i].units, cases[i].chans, nChannelsGroup,
                                    data, nSamples, chunkSamples, nt, rng, &nDet );
            double macPerSample = static_cast<double>( cases[i].units ) *
                                  cases[i].chans * kTemplateLen;
            double gmacs = macPerSample * kSampleRate / 1e9;
            std::printf( "  %6d %6d %10.2f %10.3f %9.2f %10.0f\n",
                         cases[i].units, cases[i].chans, gmacs, s,
                         kSecondsOfData / s, nDet / s );
        }
        std::printf( "\n" );
    }

    std::printf( "realtime must exceed 1.0 with margin -- the fetch loop,\n"
                 "preprocessing, the decision thread and SpikeGLX want cores too.\n" );
    return 0;
}
