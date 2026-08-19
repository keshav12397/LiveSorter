// Equivalence test: GpuPreprocessor + GpuConvolutionEngine must produce the
// SAME peak list no matter how the identical sample stream is cut into
// chunks -- in particular for chunks SHORTER than the filter's history
// (templateLength-1), and for ragged chunk sizes that change every call.
//
// Why this regime specifically: every other GPU test/validation in this repo
// streams fixed chunks of 2000 samples (the NMS equivalence tests,
// OfflineScorer.cu's default) or 150 (test_chunksize_diagnostic.cu). The
// live pipeline does neither. ImecFetchThreadGPU.cu's fetch loop has no
// pacing -- it takes whatever SpikeGLX has buffered and immediately asks
// again -- so its chunks are ragged and often tiny: on a real 3-minute
// all-units run, 31% of 84,111 chunks were under 60 samples and 1,059 of
// them were a single sample. That is exactly where GpuConvolutionEngine's
// overlap-save carry-forward used to copy a device range onto an OVERLAPPING
// device range (undefined behavior, and a parallel copy kernel at that), and
// no existing test covered it.
//
// Honest scope: this test PASSES against both the pre-fix and post-fix
// carry-forward on the machine it was written on (RTX 4500 Ada, CUDA 11.3) --
// that driver's device-to-device copy happens to tolerate those overlaps. It
// is here to pin the regime, not to reproduce that particular UB, which by
// definition is not guaranteed to reproduce anywhere.
//
// Self-contained: deterministic synthetic data + filters, no fixture files,
// no dataset. Prints PASS/FAIL and returns 0/1.
//
// Run:
//   test_gpu_chunking_equivalence.exe

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>

#include <cuda_runtime.h>

#include "CudaUtil.h"
#include "GpuFilterBank.h"
#include "GpuPreprocessor.h"
#include "GpuConvolutionEngine.h"

namespace {

const int  kChannelsGroup = 96;    // shank1only.json's size, the live CAR group
const int  kUnits         = 8;
const int  kChansPerUnit  = 5;
const int  kTemplateLen   = 61;
const long long kSamples  = 400000;   // ~13s at 30kHz -- long enough for many tail trims
const double kSampleRate  = 30000.0;

// Every run constructs its engine with the SAME maxChunkSamples, so every
// run gets the same dTailCap_ / decision-window size and the only thing
// varying is how processChunk() calls are cut. Sizing the engine from each
// plan's own largest chunk instead would compare two engines with different
// buffered windows, and the windowed-NMS algorithm is only chunking-
// invariant for a FIXED window (a wider one resolves longer elimination
// chains -- see ConvolutionEngine.cpp's note that no finite margin handles
// unbounded-depth chains).
const int  kEngineMaxChunk = 2000;

// Deterministic synthetic recording: correlated background noise plus
// spike-like biphasic transients on random channels, as int16. The point is
// only that the matched-filter output has plenty of genuine local maxima
// with a wide range of heights, so the NMS decision chain actually has work
// to do and any chunking difference has something to disagree about.
std::vector<short> makeRaw()
{
    std::mt19937 rng( 12345u );
    std::normal_distribution<double> noise( 0.0, 60.0 );
    std::vector<short> raw( static_cast<size_t>( kSamples ) * kChannelsGroup );

    for( size_t i = 0; i < raw.size(); ++i )
        raw[i] = static_cast<short>( noise( rng ) );

    std::uniform_int_distribution<int> chPick( 0, kChannelsGroup - 1 );
    std::uniform_int_distribution<long long> tPick( 100, kSamples - 200 );
    std::uniform_real_distribution<double> ampPick( 120.0, 900.0 );

    for( int s = 0; s < 6000; ++s ) {
        long long t0 = tPick( rng );
        int chCenter = chPick( rng );
        double amp = ampPick( rng );
        for( int dc = -2; dc <= 2; ++dc ) {
            int ch = chCenter + dc;
            if( ch < 0 || ch >= kChannelsGroup )
                continue;
            double chGain = amp / (1.0 + std::abs( dc ));
            for( int k = 0; k < 40; ++k ) {
                double phase = k / 40.0;
                double shape = -std::sin( 2.0 * 3.14159265358979 * phase )
                                * std::exp( -3.0 * phase );
                size_t idx = static_cast<size_t>( t0 + k ) * kChannelsGroup + ch;
                double v = raw[idx] + chGain * shape;
                if( v > 32000.0 )  v = 32000.0;
                if( v < -32000.0 ) v = -32000.0;
                raw[idx] = static_cast<short>( v );
            }
        }
    }
    return raw;
}

// Deterministic filter bank. Thresholds are -infinity so EVERY windowed-NMS
// accepted peak is reported (same trick OfflineScorer.cu uses) -- that makes
// this test maximally sensitive: a chunking difference that only nudged a
// near-threshold peak would still show up as a peak-list difference.
GpuFilterBank makeFilterBank()
{
    std::mt19937 rng( 777u );
    std::uniform_int_distribution<int> chPick( 0, kChannelsGroup - 1 );
    std::normal_distribution<double> tapDist( 0.0, 0.02 );

    std::vector<int> unitIds( kUnits );
    std::vector<int32_t> channels( static_cast<size_t>( kUnits ) * kChansPerUnit );
    std::vector<float> filters(
        static_cast<size_t>( kUnits ) * kTemplateLen * kChansPerUnit );
    std::vector<float> thresholds( kUnits, -std::numeric_limits<float>::infinity() );

    for( int u = 0; u < kUnits; ++u ) {
        unitIds[u] = 100 + u;
        int base = chPick( rng );
        for( int c = 0; c < kChansPerUnit; ++c )
            channels[u * kChansPerUnit + c] = ( base + c ) % kChannelsGroup;
        for( int k = 0; k < kTemplateLen; ++k ) {
            // A smooth envelope over the taps keeps D from being white noise,
            // so peaks are spaced like real matched-filter output rather than
            // one candidate every other sample.
            double env = std::exp( -0.004 * ( k - 30.0 ) * ( k - 30.0 ) );
            for( int c = 0; c < kChansPerUnit; ++c ) {
                size_t idx = ( static_cast<size_t>( u ) * kTemplateLen + k ) * kChansPerUnit + c;
                filters[idx] = static_cast<float>( tapDist( rng ) * env );
            }
        }
    }

    return GpuFilterBank::fromHostArrays( unitIds, channels, filters, thresholds,
                                           kChansPerUnit, kTemplateLen );
}

struct Peak {
    int       unit;
    long long sampleIndex;
    float     score;
};

// Streams `raw` through a FRESH preprocessor+engine, cut according to
// `chunkPlan` (cycled), and returns every reported peak including flush()'s.
std::vector<Peak> runStream( const short *raw, const GpuFilterBank &fb,
                              const std::vector<int> &chunkPlan, int maxChunk )
{

    GpuPreprocessor preprocessor( kChannelsGroup, 300.0, kSampleRate,
                                   /*applyHighpass=*/true, /*applyCar=*/true );
    GpuConvolutionEngine engine( fb, kChannelsGroup, maxChunk,
                                  kTemplateLen / 2, /*detectionCapacity=*/60000 );

    short *d_raw = nullptr;
    CUDA_CHECK( cudaMalloc( &d_raw,
        static_cast<size_t>( maxChunk ) * kChannelsGroup * sizeof(short) ) );

    std::vector<Peak> out;
    long long pos = 0;
    size_t planIdx = 0;

    while( pos < kSamples ) {
        long long n = chunkPlan[planIdx % chunkPlan.size()];
        ++planIdx;
        if( pos + n > kSamples )
            n = kSamples - pos;

        CUDA_CHECK( cudaMemcpy( d_raw, raw + static_cast<size_t>( pos ) * kChannelsGroup,
            static_cast<size_t>( n ) * kChannelsGroup * sizeof(short), cudaMemcpyHostToDevice ) );

        std::vector<GpuPeakEvent> peaks = engine.processChunk(
            preprocessor, d_raw, static_cast<size_t>( n ), pos, /*stream=*/nullptr );
        for( size_t i = 0; i < peaks.size(); ++i ) {
            Peak p;
            p.unit = peaks[i].unitIndex;
            p.sampleIndex = peaks[i].sampleIndex;
            p.score = peaks[i].score;
            out.push_back( p );
        }
        pos += n;
    }

    std::vector<GpuPeakEvent> flushed = engine.flush( /*stream=*/nullptr );
    for( size_t i = 0; i < flushed.size(); ++i ) {
        Peak p;
        p.unit = flushed[i].unitIndex;
        p.sampleIndex = flushed[i].sampleIndex;
        p.score = flushed[i].score;
        out.push_back( p );
    }

    cudaFree( d_raw );

    std::sort( out.begin(), out.end(), []( const Peak &a, const Peak &b ) {
        if( a.unit != b.unit ) return a.unit < b.unit;
        return a.sampleIndex < b.sampleIndex;
    } );
    return out;
}

// Fraction of peaks allowed to appear in one run's list but not the other's.
// NOT slack for a buggy history buffer -- see comparePeaks() for exactly
// what it covers and what stays exact.
const double kMaxMembershipDiff = 0.001;

// Compares two peak lists by (unit, sampleIndex) membership plus score.
//
// The SCORE check is exact-ish (1e-4 relative) and applies to every peak
// both runs found: a D value is one thread's serial dot product over one
// fixed window of the shared history+chunk buffer, so it cannot legitimately
// depend on chunking at all. This is the assertion that catches a corrupted
// carry-forward -- a bad history buffer perturbs D everywhere, not rarely.
//
// MEMBERSHIP is allowed a small mismatch budget. Whether a candidate
// survives windowed NMS can depend on an elimination chain reaching further
// back than the buffered window, and how much is buffered at each decision
// round IS a function of chunk cadence (chunk=2000 decides once over ~2150
// buffered samples; chunk=1 decides every sample over ~152). This is the
// documented finite-margin limitation ConvolutionEngine.cpp's derivation
// comment already calls out -- no finite margin resolves unbounded-depth
// chains -- not a chunking bug, and it is measured here (a couple of peaks
// in ~79,000) rather than assumed small.
bool comparePeaks( const char *label, const std::vector<Peak> &ref,
                    const std::vector<Peak> &got )
{
    size_t i = 0, j = 0, common = 0, refOnly = 0, gotOnly = 0;
    bool scoresOk = true;

    while( i < ref.size() && j < got.size() ) {
        bool less = ref[i].unit < got[j].unit ||
                    ( ref[i].unit == got[j].unit && ref[i].sampleIndex < got[j].sampleIndex );
        bool greater = got[j].unit < ref[i].unit ||
                       ( got[j].unit == ref[i].unit && got[j].sampleIndex < ref[i].sampleIndex );
        if( less )          { ++refOnly; ++i; }
        else if( greater )  { ++gotOnly; ++j; }
        else {
            double denom = std::max( 1e-6, std::fabs( (double)ref[i].score ) );
            double relErr = std::fabs( (double)got[j].score - (double)ref[i].score ) / denom;
            if( relErr > 1e-4 && scoresOk ) {
                std::printf( "  %-28s FAIL: unit %d @ %lld scored %.9g, reference "
                              "%.9g (rel err %.3g) -- D values must not depend on "
                              "chunking\n", label, got[j].unit, got[j].sampleIndex,
                              got[j].score, ref[i].score, relErr );
                scoresOk = false;
            }
            ++common; ++i; ++j;
        }
    }
    refOnly += ref.size() - i;
    gotOnly += got.size() - j;

    size_t diff = refOnly + gotOnly;
    double diffFrac = ref.empty() ? 0.0 : diff / (double)ref.size();
    bool membershipOk = diffFrac <= kMaxMembershipDiff;

    std::printf( "  %-28s %s  common=%zu  ref-only=%zu  run-only=%zu (%.4f%%)\n",
                  label, ( scoresOk && membershipOk ) ? "ok  " : "FAIL",
                  common, refOnly, gotOnly, 100.0 * diffFrac );
    if( !membershipOk )
        std::printf( "      membership difference exceeds the %.3f%% budget\n",
                      100.0 * kMaxMembershipDiff );
    return scoresOk && membershipOk;
}

} // namespace


int main()
{
    std::printf( "GpuConvolutionEngine chunking equivalence\n" );
    std::printf( "  %lld samples, %d channels, %d units, templateLength=%d\n",
                  kSamples, kChannelsGroup, kUnits, kTemplateLen );

    std::vector<short> raw = makeRaw();
    GpuFilterBank fb = makeFilterBank();

    // Reference: the chunk size every previous GPU test used.
    std::vector<int> plan2000( 1, 2000 );
    std::vector<Peak> ref = runStream( raw.data(), fb, plan2000, kEngineMaxChunk );
    std::printf( "  reference (chunk=2000)       %zu peaks\n", ref.size() );
    if( ref.empty() ) {
        std::printf( "FAIL: reference run produced no peaks -- the synthetic "
                      "fixture is not exercising anything\n" );
        return 1;
    }

    bool allOk = true;

    // 150: what fetchChunkMs=5 nominally asks for. Already >= templateLength-1,
    // so this one passed even before the overlapping-copy fix -- kept as the
    // control.
    std::vector<int> plan150( 1, 150 );
    allOk &= comparePeaks( "chunk=150", ref, runStream( raw.data(), fb, plan150, kEngineMaxChunk ) );

    // 30: shorter than templateLength-1 = 60, so the carry-forward's source
    // and destination ranges overlap on EVERY chunk.
    std::vector<int> plan30( 1, 30 );
    allOk &= comparePeaks( "chunk=30 (< L-1)", ref, runStream( raw.data(), fb, plan30, kEngineMaxChunk ) );

    // 1: the degenerate case the live loop really does hit.
    std::vector<int> plan1( 1, 1 );
    allOk &= comparePeaks( "chunk=1", ref, runStream( raw.data(), fb, plan1, kEngineMaxChunk ) );

    // Ragged, straddling the L-1 boundary in both directions -- the actual
    // live shape (see this file's header for the measured distribution).
    int raggedVals[] = { 1, 150, 7, 61, 59, 60, 3, 120, 45, 2, 90, 33, 150, 1, 75 };
    std::vector<int> planRagged( raggedVals, raggedVals + 15 );
    allOk &= comparePeaks( "ragged 1..150", ref, runStream( raw.data(), fb, planRagged, kEngineMaxChunk ) );

    fb.release();

    std::printf( "%s\n", allOk ? "PASS" : "FAIL" );
    return allOk ? 0 : 1;
}
