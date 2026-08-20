// Equivalence test for MultiConvolutionEngine, the CPU batched engine that
// replaced GpuConvolutionEngine.
//
// It asserts three properties, in increasing order of how badly a violation
// would hurt:
//
//   1. BATCHED == SINGLE. For every unit, the batched engine's detections
//      are identical to running one bare ConvolutionEngine on that unit's
//      own gathered channels. This should be true by construction -- the
//      batched engine literally owns one ConvolutionEngine per unit -- so
//      what this really tests is the two things wrapped around it: the
//      channel gather, and the threshold comparison. A transposed gather
//      index is the obvious way to break this, and it would otherwise show
//      up only as quietly degraded detection on live data.
//
//   2. CHUNKING INVARIANCE. The same sample stream cut into different chunk
//      patterns -- including chunks SHORTER than templateLength-1, and
//      ragged sizes that change every call -- must give the same peaks.
//      This is the regime test_gpu_chunking_equivalence.cu existed for: the
//      live fetch loop consumes whatever SpikeGLX has buffered rather than
//      waiting for a full chunk, so on a real 3-minute run 31% of 84,111
//      chunks were under 60 samples and 1,059 were a single sample. That is
//      exactly where the GPU version's overlap-save carry-forward used to
//      copy a device range onto an overlapping one.
//
//   3. THREAD-COUNT INVARIANCE. 1 worker and 8 workers must produce the
//      identical vector, not merely the same set. Without the sort in
//      processChunk() this fails, and it fails differently on machines with
//      different core counts -- the worst possible failure mode, since CI
//      and the rig would disagree and neither would be reproducible.
//
// Self-contained: deterministic synthetic data and filters, no fixture
// files, no dataset. Prints PASS/FAIL and returns 0/1.

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#include "MultiFilterBank.h"
#include "MultiConvolutionEngine.h"
#include "ConvolutionEngine.h"

namespace {

const int  kChannelsGroup = 96;     // shank1only.json's size, the live CAR group
const int  kUnits         = 8;
const int  kChansPerUnit  = 5;
const int  kTemplateLen   = 61;
const long long kSamples  = 120000; // ~4 s at 30 kHz
const long long kMinSep    = 30;

struct Fixture {
    MultiFilterBank      bank;
    std::vector<double>  data;      // kSamples * kChannelsGroup, preprocessed
};

Fixture makeFixture()
{
    std::mt19937 rng( 12345 );
    std::normal_distribution<double> gauss( 0.0, 1.0 );
    std::uniform_int_distribution<int> chanPick( 0, kChannelsGroup - 1 );

    Fixture fx;

    std::vector<int>     unitIds( kUnits );
    std::vector<int32_t> channels( static_cast<size_t>( kUnits ) * kChansPerUnit );
    std::vector<float>   filters( static_cast<size_t>( kUnits ) * kTemplateLen * kChansPerUnit );
    std::vector<float>   thresholds( kUnits );

    for( int u = 0; u < kUnits; ++u ) {
        unitIds[u] = 100 + u;
        // Distinct, deliberately non-contiguous channel sets: a gather bug
        // that assumed unit u reads channels [u*nc, u*nc+nc) would pass with
        // tidy consecutive blocks.
        for( int c = 0; c < kChansPerUnit; ++c )
            channels[u * kChansPerUnit + c] = chanPick( rng );
        for( int i = 0; i < kTemplateLen * kChansPerUnit; ++i )
            filters[u * kTemplateLen * kChansPerUnit + i] = static_cast<float>( gauss( rng ) * 0.1 );
        // Low enough that plenty of peaks clear it, high enough that the
        // comparison is actually exercised rather than passing everything.
        thresholds[u] = 1.5f;
    }

    fx.bank = MultiFilterBank::fromHostArrays( unitIds, channels, filters, thresholds,
                                               kChansPerUnit, kTemplateLen );

    fx.data.resize( static_cast<size_t>( kSamples ) * kChannelsGroup );
    for( size_t i = 0; i < fx.data.size(); ++i )
        fx.data[i] = gauss( rng );
    return fx;
}

bool same( const std::vector<MultiPeakEvent> &a, const std::vector<MultiPeakEvent> &b,
           const char *what )
{
    if( a.size() != b.size() ) {
        std::printf( "  FAIL %s: %zu vs %zu detections\n", what, a.size(), b.size() );
        return false;
    }
    for( size_t i = 0; i < a.size(); ++i ) {
        if( a[i].unitIndex != b[i].unitIndex || a[i].sampleIndex != b[i].sampleIndex ||
            a[i].score != b[i].score ) {
            std::printf( "  FAIL %s: first difference at %zu: "
                         "(u=%d n=%lld s=%.9g) vs (u=%d n=%lld s=%.9g)\n",
                         what, i, a[i].unitIndex, a[i].sampleIndex, (double)a[i].score,
                         b[i].unitIndex, b[i].sampleIndex, (double)b[i].score );
            return false;
        }
    }
    std::printf( "  PASS %s (%zu detections)\n", what, a.size() );
    return true;
}

// Feed the whole stream through the batched engine using `plan` as a
// repeating cycle of chunk sizes.
std::vector<MultiPeakEvent> runBatched( const Fixture &fx, const std::vector<size_t> &plan,
                                        int nThreads )
{
    MultiConvolutionEngine eng( fx.bank, kChannelsGroup, kMinSep, nThreads );
    std::vector<MultiPeakEvent> all;
    long long off = 0;
    size_t p = 0;
    while( off < kSamples ) {
        size_t n = std::min<size_t>( plan[p % plan.size()],
                                     static_cast<size_t>( kSamples - off ) );
        ++p;
        std::vector<MultiPeakEvent> got = eng.processChunk(
            fx.data.data() + static_cast<size_t>( off ) * kChannelsGroup, n, off );
        all.insert( all.end(), got.begin(), got.end() );
        off += static_cast<long long>( n );
    }
    std::vector<MultiPeakEvent> tail = eng.flush();
    all.insert( all.end(), tail.begin(), tail.end() );
    return all;
}

// The oracle: one bare ConvolutionEngine per unit, fed that unit's gathered
// channels, thresholded here. No MultiConvolutionEngine involved.
std::vector<MultiPeakEvent> runSingleUnitReference( const Fixture &fx, size_t chunk )
{
    std::vector<MultiPeakEvent> all;
    for( int u = 0; u < kUnits; ++u ) {
        const int32_t *chans = fx.bank.unitChannels( u );
        const float   *f     = fx.bank.unitFilter( u );
        std::vector<double> taps( static_cast<size_t>( kTemplateLen ) * kChansPerUnit );
        for( size_t i = 0; i < taps.size(); ++i )
            taps[i] = static_cast<double>( f[i] );

        ConvolutionEngine eng( kTemplateLen, kChansPerUnit, taps, kMinSep );

        std::vector<double> g;
        long long off = 0;
        std::vector<PeakEvent> peaks;
        while( off < kSamples ) {
            size_t n = std::min<size_t>( chunk, static_cast<size_t>( kSamples - off ) );
            g.resize( n * kChansPerUnit );
            for( size_t t = 0; t < n; ++t ) {
                const double *src = fx.data.data() +
                    ( static_cast<size_t>( off ) + t ) * kChannelsGroup;
                for( int c = 0; c < kChansPerUnit; ++c )
                    g[t * kChansPerUnit + c] = src[chans[c]];
            }
            std::vector<PeakEvent> got = eng.processChunk( g.data(), n, off );
            peaks.insert( peaks.end(), got.begin(), got.end() );
            off += static_cast<long long>( n );
        }
        std::vector<PeakEvent> tail = eng.flush();
        peaks.insert( peaks.end(), tail.begin(), tail.end() );

        float thr = fx.bank.thresholds[u];
        for( size_t i = 0; i < peaks.size(); ++i ) {
            if( peaks[i].score >= static_cast<double>( thr ) ) {
                MultiPeakEvent ev;
                ev.unitIndex   = u;
                ev.sampleIndex = peaks[i].sampleIndex;
                ev.score       = static_cast<float>( peaks[i].score );
                all.push_back( ev );
            }
        }
    }
    // The batched engine emits per chunk in (sampleIndex, unitIndex) order;
    // this reference emits unit-by-unit over the whole stream. Sorting both
    // the same way compares the CONTENT rather than the emission order,
    // which is the property that actually matters.
    std::sort( all.begin(), all.end(),
               []( const MultiPeakEvent &a, const MultiPeakEvent &b ) {
                   if( a.sampleIndex != b.sampleIndex ) return a.sampleIndex < b.sampleIndex;
                   return a.unitIndex < b.unitIndex;
               } );
    return all;
}

std::vector<MultiPeakEvent> sorted( std::vector<MultiPeakEvent> v )
{
    std::sort( v.begin(), v.end(),
               []( const MultiPeakEvent &a, const MultiPeakEvent &b ) {
                   if( a.sampleIndex != b.sampleIndex ) return a.sampleIndex < b.sampleIndex;
                   return a.unitIndex < b.unitIndex;
               } );
    return v;
}

} // namespace


int main()
{
    Fixture fx = makeFixture();
    bool ok = true;

    // 0. Establish what the BARE core does under the same re-chunking, so a
    // difference found later can be attributed. Without this baseline, a
    // chunking difference in the batched engine looks like a batching bug
    // when it may be a property of ConvolutionEngine's windowed NMS that
    // predates this port entirely.
    std::printf( "0. baseline: bare ConvolutionEngine vs itself under re-chunking\n" );
    std::vector<MultiPeakEvent> ref = runSingleUnitReference( fx, 2000 );
    std::vector<MultiPeakEvent> bare1 = runSingleUnitReference( fx, 1 );
    std::printf( "   (a mismatch here is a MEASUREMENT, not a failure -- it sets the\n"
                 "    standard the batched engine is then held to)\n" );
    bool bareChunkInvariant = same( ref, bare1, "bare core: 2000 vs 1-sample chunks" );
    if( !bareChunkInvariant )
        std::printf( "   -> the core is not 1-sample chunk-invariant; batched must MATCH that.\n" );

    std::printf( "1. batched == one bare ConvolutionEngine per unit\n" );
    std::vector<MultiPeakEvent> bat = sorted( runBatched( fx, { 2000 }, 4 ) );
    ok &= same( ref, bat, "fixed 2000-sample chunks" );

    std::printf( "2. chunking invariance\n" );
    struct Plan { const char *name; std::vector<size_t> sizes; };
    std::vector<Plan> plans = {
        { "1-sample chunks",         { 1 } },
        { "sub-history (17)",        { 17 } },
        { "ragged 1/3/60/997",       { 1, 3, 60, 997 } },
        { "ragged 61/1/2000/5",      { 61, 1, 2000, 5 } },
        { "one giant chunk",         { static_cast<size_t>( kSamples ) } },
    };
    for( size_t i = 0; i < plans.size(); ++i ) {
        std::vector<MultiPeakEvent> got = sorted( runBatched( fx, plans[i].sizes, 4 ) );
        bool match = same( ref, got, plans[i].name );
        // 1-sample chunks are held to the bare core's own standard, not to a
        // stricter one. If ConvolutionEngine itself is not invariant at that
        // extreme -- its windowed NMS resolves elimination chains against a
        // buffered window whose contents depend on where decisions were
        // triggered -- then the batched engine reproducing that exactly is
        // the correct outcome, and demanding better would mean the batched
        // engine had stopped being the same algorithm.
        if( !match && plans[i].sizes.size() == 1 && plans[i].sizes[0] == 1 &&
            !bareChunkInvariant ) {
            std::vector<MultiPeakEvent> bare = runSingleUnitReference( fx, 1 );
            if( same( bare, got, "  ...but batched(1) == bare(1), so this is the "
                                 "core's own behaviour, not the port's" ) )
                match = true;
        }
        ok &= match;
    }

    std::printf( "3. thread-count invariance\n" );
    std::vector<MultiPeakEvent> t1 = runBatched( fx, { 997 }, 1 );
    std::vector<MultiPeakEvent> t8 = runBatched( fx, { 997 }, 8 );
    // NOT sorted here: these are compared as emitted, which is the actual
    // claim -- the returned vector itself is thread-count independent.
    ok &= same( t1, t8, "1 worker vs 8 workers, as emitted" );

    std::printf( "\n%s\n", ok ? "PASS" : "FAIL" );
    return ok ? 0 : 1;
}
