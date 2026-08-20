#ifndef CLOSEDLOOP_FASTMATCHEDFILTER_H
#define CLOSEDLOOP_FASTMATCHEDFILTER_H

#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>

#if defined( __AVX2__ )
#include <immintrin.h>
#endif

// One unit's matched filter: preprocessed samples in, D (filter output)
// values out. NO peak decisions -- those stay in ConvolutionEngine, which is
// where the windowed non-max suppression derivation lives and where this
// codebase has twice lost recall to a rewrite. See
// ConvolutionEngine::processPrecomputedD().
//
// Two things differ from ConvolutionEngine's own convolution, and both are
// deliberate.
//
// 1. It accumulates in FLOAT32, not double
// -----------------------------------------
// Not a shortcut -- a correction. calibrate_all_units.py picks each unit's
// threshold by scoring float32-cast filters against float32-cast data, and
// generate_filter.filter_output() carries an explicit comment saying the
// dtype must follow the input so that "a threshold picked against
// silently-upcast float64 scores" is not "calibrated for a distribution the
// [online] path never actually produces". The GPU kernel this replaced was
// float32 throughout. ConvolutionEngine is float64 because it serves the
// SINGLE-TARGET path, whose Python side writes float64 filters and
// thresholds -- a separate, internally consistent convention.
//
// So float32 here restores the match between the all-units online path and
// the calibration that chose its thresholds. Accumulating 61*6 = 366
// products in float32 carries roughly sqrt(366) * 6e-8 ~ 1e-6 relative
// error, which is far below any threshold-crossing decision and is in any
// case the exact error the threshold was selected under.
//
// 2. It is laid out CHANNEL-MAJOR, and vectorises over output samples
// -------------------------------------------------------------------
// ConvolutionEngine's inner loop sums across a unit's 5-6 contiguous
// channels and ends in a horizontal add -- clear to read, and the wrong
// shape for SIMD: short partial vectors and a reduction every iteration.
// Measured, it reached ~1.8 GFLOP/s per core against 20-48 GFLOP/s of AVX2
// double-precision peak, about 4-9%, which is what made 1000 units miss
// real time by 12x rather than by any shortage of arithmetic.
//
// Here the sample buffer is channel-major (`chan[c][t]` contiguous in t), so
// for a fixed (tap, channel) pair the multiplier is a scalar broadcast and
// the samples are a contiguous run of consecutive OUTPUT positions. Whole
// vectors, no reductions. The hot loop is AVX2 intrinsics with a scalar
// fallback that produces bit-identical results -- see the comment at the
// intrinsics block for why hand-written rather than a portable loop nest,
// and why the two paths must round the same way.
//
// The layout transform is free: it happens during the channel gather that
// had to run anyway to slice this unit's channels out of the full CAR group.
class FastMatchedFilter {
public:
    // taps: row-major [k*nChannels + c], exactly the layout
    // MultiFilterBank::unitFilter() returns and ConvolutionEngine takes.
    // leftMargin/rightMargin must come from the ConvolutionEngine this feeds
    // (its leftMargin()/rightMargin()), NOT be re-derived -- the centered
    // "same"-mode convention is the one thing in this pipeline that has
    // already been silently mis-derived once.
    FastMatchedFilter( int templateLength, int nChannels, const float *taps,
                       int leftMargin, int rightMargin )
        :   L_( templateLength ), nCh_( nChannels ),
            leftMargin_( leftMargin ), rightMargin_( rightMargin ),
            histLen_( 0 )
    {
        taps_.assign( taps, taps + static_cast<size_t>( L_ ) * nCh_ );
        keep_ = ( L_ > 0 ) ? static_cast<size_t>( L_ - 1 ) : 0;
        chan_.assign( static_cast<size_t>( nCh_ ), std::vector<float>() );
        hist_.assign( static_cast<size_t>( nCh_ ), std::vector<float>( keep_, 0.0f ) );
    }

    void reset()
    {
        histLen_ = 0;
        for( int c = 0; c < nCh_; ++c )
            std::fill( hist_[c].begin(), hist_[c].end(), 0.0f );
    }

    // Copies this unit's channels out of an ALREADY-TRANSPOSED, already
    // float32 chunk, prepends the retained history, and computes every valid
    // centered D value.
    //
    // `groupT` is channel-major: groupT[ch * rowStride + t]. That the caller
    // transposes, rather than this function gathering from the time-major
    // buffer itself, is the single largest performance decision in this file
    // and is worth stating plainly. Gathering here meant reading 6 channels
    // out of 384 with a ~3 KB stride: every 8-byte sample pulled a 64-byte
    // cache line and discarded the rest, ~8x read amplification -- and every
    // one of N units repeated it over the same buffer. Transposing once per
    // chunk turns each unit's gather into a contiguous row copy and makes the
    // amplification a per-chunk cost instead of a per-unit-per-chunk one.
    //
    // `dOut` receives nD values, D[i] centered on absolute sample index
    // (*firstDAbsIndex + i). Returns nD, which is 0 when history+chunk is
    // still shorter than the template.
    //
    // Index arithmetic is reproduced from ConvolutionEngine::processChunk()
    // rather than reinvented: combined = history ++ chunk; valid centered
    // positions run from leftMargin_ to total-1-rightMargin_; and the window
    // start for output idx reduces to exactly idx (the two margins cancel).
    size_t computeD( const float *groupT, size_t nSamples, size_t rowStride,
                     const int *channels, long long streamSampleOffset,
                     std::vector<double> &dOut, long long *firstDAbsIndex )
    {
        const size_t total = histLen_ + nSamples;

        for( int c = 0; c < nCh_; ++c ) {
            std::vector<float> &v = chan_[c];
            v.resize( total );
            if( histLen_ > 0 )
                std::copy( hist_[c].begin(), hist_[c].begin() + histLen_, v.begin() );
            const float *src = groupT + static_cast<size_t>( channels[c] ) * rowStride;
            std::copy( src, src + nSamples, v.begin() + histLen_ );
        }

        const long long firstCombinedAbs =
            streamSampleOffset - static_cast<long long>( histLen_ );

        size_t nD = 0;
        if( total >= static_cast<size_t>( L_ ) ) {
            const size_t firstValid = static_cast<size_t>( leftMargin_ );
            const size_t lastValid  = total - 1 - static_cast<size_t>( rightMargin_ );
            nD = lastValid - firstValid + 1;
            dOut.resize( nD );

            const int L = L_;
            size_t idx = 0;

#if defined( __AVX2__ )
            // FOUR accumulators covering 32 DIFFERENT output positions, not a
            // split of one sum. That distinction is the whole reason this is
            // written with intrinsics rather than as a loop nest:
            //
            //  - Splitting one output's sum across several accumulators would
            //    reorder the addition, and then the vector path and the scalar
            //    tail would round differently. Since which outputs land in the
            //    tail depends on how the stream was chunked, D itself would
            //    become chunking-dependent -- breaking the invariance the
            //    tests check. Here each lane accumulates over (channel, tap)
            //    in exactly the scalar order. Order alone was not enough: the
            //    tail also has to FUSE its multiply-add, see std::fmaf there.
            //  - Four independent chains are still needed, because an FMA has
            //    ~4-5 cycles of latency and one accumulator would stall on
            //    itself. Four also balances the two load ports against the two
            //    FMA ports at 4 loads per 4 FMAs.
            //
            // MSVC 2017's auto-vectoriser refuses the equivalent loop nest
            // outright (/Qvec-report:2 gives reason 1303 on the inner loop),
            // so writing this portably and hoping is not an option -- it was
            // tried, and left 5x on the table.
            for( ; idx + 32 <= nD; idx += 32 ) {
                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps();
                for( int c = 0; c < nCh_; ++c ) {
                    const float *s = chan_[c].data() + idx;
                    const float *f = taps_.data() + c;   // stride nCh_ over k
                    for( int k = 0; k < L; ++k ) {
                        const __m256 w = _mm256_set1_ps( f[static_cast<size_t>( k ) * nCh_] );
                        const float *sk = s + k;
                        a0 = _mm256_fmadd_ps( w, _mm256_loadu_ps( sk ),      a0 );
                        a1 = _mm256_fmadd_ps( w, _mm256_loadu_ps( sk + 8 ),  a1 );
                        a2 = _mm256_fmadd_ps( w, _mm256_loadu_ps( sk + 16 ), a2 );
                        a3 = _mm256_fmadd_ps( w, _mm256_loadu_ps( sk + 24 ), a3 );
                    }
                }
                float tmp[32];
                _mm256_storeu_ps( tmp,      a0 );
                _mm256_storeu_ps( tmp + 8,  a1 );
                _mm256_storeu_ps( tmp + 16, a2 );
                _mm256_storeu_ps( tmp + 24, a3 );
                for( int j = 0; j < 32; ++j )
                    dOut[idx + j] = static_cast<double>( tmp[j] );
            }

            // An 8-wide stage between the 32-wide blocks and the scalar tail.
            // This is not a micro-optimisation, it is most of the win: at the
            // live chunk size nD is 150, so the 32-wide loop covers 128 and
            // would leave 22 outputs to a scalar path roughly 8x slower --
            // about as much time as the entire vector body. Mopping up to a
            // remainder of at most 7 is what makes the vector width actually
            // show up in the measurement.
            for( ; idx + 8 <= nD; idx += 8 ) {
                __m256 a0 = _mm256_setzero_ps();
                for( int c = 0; c < nCh_; ++c ) {
                    const float *s = chan_[c].data() + idx;
                    const float *f = taps_.data() + c;
                    for( int k = 0; k < L; ++k ) {
                        const __m256 w = _mm256_set1_ps( f[static_cast<size_t>( k ) * nCh_] );
                        a0 = _mm256_fmadd_ps( w, _mm256_loadu_ps( s + k ), a0 );
                    }
                }
                float tmp[8];
                _mm256_storeu_ps( tmp, a0 );
                for( int j = 0; j < 8; ++j )
                    dOut[idx + j] = static_cast<double>( tmp[j] );
            }

            // Final remainder as ONE OVERLAPPING 8-wide block ending at nD,
            // recomputing up to 7 outputs that were already written. The
            // recomputation is exactly free of consequence -- same inputs,
            // same order, same fused ops, so it stores the identical bits --
            // and it keeps the scalar path out of the steady state entirely.
            //
            // That matters much more than 7 outputs suggests. The scalar tail
            // has to use std::fmaf to match the vector path's single rounding,
            // and profiling showed fmaf is not lowered to a vfmadd here: it is
            // a real call, and 6 leftover outputs x 366 taps of it cost more
            // than the entire vectorised body. Guessing would never have found
            // that; it looked like six samples of arithmetic.
            if( idx < nD && nD >= 8 ) {
                const size_t start = nD - 8;
                __m256 a0 = _mm256_setzero_ps();
                for( int c = 0; c < nCh_; ++c ) {
                    const float *s = chan_[c].data() + start;
                    const float *f = taps_.data() + c;
                    for( int k = 0; k < L; ++k ) {
                        const __m256 w = _mm256_set1_ps( f[static_cast<size_t>( k ) * nCh_] );
                        a0 = _mm256_fmadd_ps( w, _mm256_loadu_ps( s + k ), a0 );
                    }
                }
                float tmp[8];
                _mm256_storeu_ps( tmp, a0 );
                for( int j = 0; j < 8; ++j )
                    dOut[start + j] = static_cast<double>( tmp[j] );
                idx = nD;
            }
#endif
            // Scalar tail -- and the entire loop on a target without AVX2.
            //
            // std::fmaf, NOT `acc += w * s[k]`. Matching the accumulation
            // ORDER is necessary but was not sufficient: _mm256_fmadd_ps
            // rounds the multiply and the add together, once, while a
            // separate multiply and add round twice. That is a ~1e-7
            // relative difference, and because which outputs fall into this
            // tail depends on where the stream happened to be cut, it made D
            // itself chunking-dependent -- two runs over identical samples
            // disagreeing in the last digits, and occasionally on either side
            // of a threshold. The equivalence test caught it; reasoning about
            // accumulation order alone did not.
            for( ; idx < nD; ++idx ) {
                float acc = 0.0f;
                for( int c = 0; c < nCh_; ++c ) {
                    const float *s = chan_[c].data() + idx;
                    const float *f = taps_.data() + c;
                    for( int k = 0; k < L; ++k )
                        acc = std::fmaf( f[static_cast<size_t>( k ) * nCh_], s[k], acc );
                }
                dOut[idx] = static_cast<double>( acc );
            }
            *firstDAbsIndex = firstCombinedAbs + static_cast<long long>( firstValid );
        }

        const size_t keep = std::min( total, keep_ );
        for( int c = 0; c < nCh_; ++c ) {
            const std::vector<float> &v = chan_[c];
            std::copy( v.end() - static_cast<long>( keep ), v.end(), hist_[c].begin() );
        }
        histLen_ = keep;
        return nD;
    }

private:
    int    L_, nCh_, leftMargin_, rightMargin_;
    size_t keep_, histLen_;
    std::vector<float>              taps_;
    std::vector<std::vector<float>> chan_;   // channel-major working buffer
    std::vector<std::vector<float>> hist_;   // channel-major retained tail
};

#endif // CLOSEDLOOP_FASTMATCHEDFILTER_H
