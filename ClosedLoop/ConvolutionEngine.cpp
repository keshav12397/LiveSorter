#include "ConvolutionEngine.h"

#include <algorithm>
#include <climits>
#include <cstdio>

// -------------------------------------------------------------------------
// This computes exactly the same value FilterGen/generate_filter.py's
// filter_output() does:
//     D_python[n] = sum_ch np.convolve(data[:,ch], f[::-1,ch], mode='same')[n]
//
// Working through numpy's 'same'-mode indexing algebra (full convolution,
// then take the center N-length slice starting at offset (L-1)//2) gives a
// direct formula with no reversal or reindexing tricks needed:
//
//     D[n] = sum_ch sum_{k=0}^{L-1} data[ch][n - leftMargin + k] * taps[k][ch]
//
// where leftMargin = (L-1) - (L-1)//2 and rightMargin = (L-1)//2 -- i.e. the
// L-sample window is centered on n (extending leftMargin samples before it,
// rightMargin samples after), read directly against taps[0..L-1] exactly as
// stored in filter_<id>.bin. For odd L (e.g. 61) leftMargin==rightMargin,
// a genuinely symmetric window.
//
// This is deliberately NOT reformulated into a causal-only (backward-looking)
// version: an earlier version of this file did that, plus a hand-derived
// index shift to translate back to Kilosort's convention, computed that
// shift wrong, and silently cratered Calibration's recall/precision against
// ground truth. Matching Python's formula exactly removes the whole class of
// bug: PeakEvent::sampleIndex is directly comparable to spike_times.npy with
// no correction, at the cost of a small fixed reporting delay -- see below.
//
// Peak decisions match scipy.signal.find_peaks(distance=minSep)'s ACTUAL
// two-stage algorithm: candidates are strict local maxima, then a GREEDY
// suppression pass processes candidates from tallest to shortest, and each
// surviving (not-yet-suppressed) candidate suppresses every other candidate
// within `distance` of it (scipy's _select_by_peak_distance). This is a
// genuinely global/order-dependent algorithm, not a purely local one: a
// candidate that would suppress a neighbor can itself get suppressed first
// by something taller further away, in which case the neighbor survives --
// see selectByDistance()'s and this file's git history for a worked
// 3-candidate example (A=10@0, B=9@25, C=8@50, distance=30) where naively
// checking "is D[n] the tallest local max within [n-distance,n+distance]"
// wrongly rejects C (B looks taller and close enough) even though the true
// greedy algorithm has A eliminate B *before* B gets a chance to eliminate
// C, so C survives. Two earlier versions of this file got this wrong in
// different ways:
//   1. A greedy "far enough from the last *accepted* peak" left-to-right
//      rule let a small spurious side-lobe claim the exclusion zone before
//      a real spike's taller peak, capping recall ~50% even at the loosest
//      threshold.
//   2. A "windowed" version (n accepted iff D[n] is the tallest RAW value,
//      not just tallest among local-maxima candidates, within its window)
//      fixed (1) but still silently discarded ~35% of real candidate peaks
//      relative to scipy on realistic matched-filter output, because it
//      compared against non-candidate samples scipy never considers a
//      competitor, AND because (being purely local) it has exactly the
//      "wrongly rejects C" bug above.
// This version instead buffers a window of candidates and runs the TRUE
// greedy algorithm (selectByDistance(), a direct port of scipy's own
// _select_by_peak_distance) over it, only finalizing/emitting a decision
// for a candidate once `finalizeMarginSamples_` future samples exist beyond
// it -- generous enough that a future candidate revising an already-
// resolved-but-not-yet-finalized neighbor's fate (the A/B/C scenario above)
// is captured before finalization, without requiring unbounded lookahead.
// Validated to match scipy.signal.find_peaks(distance=...) exactly on a
// realistic synthetic fixture, streamed in irregular chunks -- see
// test_convolution_engine_nms_equivalence.cpp.
//
// Reporting delay: deciding candidates up to index `cutoff` needs D-values
// up to `cutoff + minSep + finalizeMargin` (to correctly resolve the widest
// possible chain within the buffered window), and the newest of those D
// values itself needs rightMargin raw samples past that -- so total delay
// is (rightMargin + minSep + finalizeMargin) samples. At 30kHz with a
// 61-tap filter, default minSep=30, finalizeMargin=4*minSep=120, that's
// ~180 samples (~6ms) -- still comfortably inside a 10ms budget, just
// larger than the previous (incorrect) version's ~2ms delay.
// -------------------------------------------------------------------------

namespace {

// Direct port of scipy.signal._peak_finding_utils._select_by_peak_distance:
// positions must be ascending. Returns, per candidate, whether it survives
// greedy tallest-first distance suppression (ties broken by numpy argsort's
// stable-sort order, i.e. earlier array position wins the ascending sort
// and so is processed LAST/first-eliminated among equal heights -- measure-
// zero for real floating-point matched-filter output, not specially tested).
std::vector<bool> selectByDistance( const std::vector<long long> &positions,
                                     const std::vector<double> &heights,
                                     long long distance )
{
    size_t n = positions.size();
    std::vector<bool> keep( n, true );
    std::vector<size_t> order( n );
    for( size_t i = 0; i < n; ++i )
        order[i] = i;
    std::stable_sort( order.begin(), order.end(),
                       [&]( size_t a, size_t b ) { return heights[a] < heights[b]; } );

    for( size_t ii = n; ii-- > 0; ) {
        size_t j = order[ii];
        if( !keep[j] )
            continue;
        long long k = static_cast<long long>( j ) - 1;
        while( k >= 0 && positions[j] - positions[static_cast<size_t>( k )] < distance ) {
            keep[static_cast<size_t>( k )] = false;
            --k;
        }
        k = static_cast<long long>( j ) + 1;
        while( k < static_cast<long long>( n ) &&
               positions[static_cast<size_t>( k )] - positions[j] < distance ) {
            keep[static_cast<size_t>( k )] = false;
            ++k;
        }
    }
    return keep;
}

} // namespace


ConvolutionEngine::ConvolutionEngine( int templateLength, int nChannels,
                                       const std::vector<double> &taps,
                                       long long minSeparationSamples )
    :   templateLength_( templateLength ),
        nChannels_( nChannels ),
        taps_( taps ),
        leftMargin_( (templateLength - 1) - (templateLength - 1) / 2 ),
        rightMargin_( (templateLength - 1) / 2 ),
        minSeparationSamples_( minSeparationSamples > 0
                                    ? minSeparationSamples
                                    : templateLength / 2 ),
        finalizeMarginSamples_( 4 * minSeparationSamples_ )
{
    reset();
}


void ConvolutionEngine::reset()
{
    history_.clear();
    historyCount_ = 0;
    dBuffer_.clear();
    dBufferStartAbsIndex_ = 0;
    recentKeptPeaks_.clear();
    // Sentinel far enough negative that the very first real peak is always
    // eligible (no prior "last decided" index to collide with), while
    // staying safely away from LLONG_MIN so `n - lastDecidedAbsIndex_`-style
    // arithmetic can't overflow.
    lastDecidedAbsIndex_ = LLONG_MIN / 2;
}


void ConvolutionEngine::decideUpTo( long long cutoff, std::vector<PeakEvent> &peaksOut )
{
    if( dBuffer_.empty() || cutoff <= lastDecidedAbsIndex_ )
        return;

    const long long newestBuffered =
        dBufferStartAbsIndex_ + static_cast<long long>( dBuffer_.size() ) - 1;
    cutoff = std::min( cutoff, newestBuffered ); // can't decide past what's actually buffered

    auto at = [&]( long long idx ) -> double {
        return dBuffer_[static_cast<size_t>( idx - dBufferStartAbsIndex_ )];
    };

    // hi: use the FULL buffered window, not just cutoff+minSep -- whether a
    // candidate near `cutoff` survives can depend on a multi-hop
    // elimination chain (candidate X's only threat gets eliminated by an
    // even taller candidate Y further out, so X survives after all -- see
    // this file's git history for a worked example spanning ~50 samples
    // with minSep=30). Bounding `hi` to just cutoff+minSep missed exactly
    // this case. Using the whole buffer (which already extends
    // finalizeMarginSamples_ beyond cutoff for processChunk()'s normal
    // calls, precisely for this reason) resolves chains up to that
    // margin's depth -- unbounded-depth chains can't be handled by ANY
    // finite margin (a fundamental property of this class of algorithm,
    // not a gap in this implementation), an acceptable, documented limit
    // for realistic matched-filter output.
    const long long hi = newestBuffered - 1;

    std::vector<long long> candPos;
    std::vector<double> candVal;

    // Already-finalized kept peaks first (ascending, all positions <=
    // lastDecidedAbsIndex_) -- fixed anchors, NOT re-derived from dBuffer_
    // (see class header / derivation comment for why that silently breaks
    // once dBuffer_ has been trimmed).
    for( size_t i = 0; i < recentKeptPeaks_.size(); ++i ) {
        candPos.push_back( recentKeptPeaks_[i].first );
        candVal.push_back( recentKeptPeaks_[i].second );
    }

    // Freshly scanned local maxima in the newly-decidable range. Clamped to
    // dBufferStartAbsIndex_+1 -- lastDecidedAbsIndex_ starts at a huge
    // negative sentinel (see reset()) before the first decision round ever
    // runs.
    const long long scanFrom = std::max( lastDecidedAbsIndex_ + 1, dBufferStartAbsIndex_ + 1 );
    for( long long p = scanFrom; p <= hi; ++p ) {
        if( p - 1 < dBufferStartAbsIndex_ )
            continue; // boundary -- can't confirm, matches scipy's own edge behavior
        if( at( p - 1 ) < at( p ) && at( p ) > at( p + 1 ) ) {
            candPos.push_back( p );
            candVal.push_back( at( p ) );
        }
    }

    std::vector<bool> keep = selectByDistance( candPos, candVal, minSeparationSamples_ );

    for( size_t i = 0; i < candPos.size(); ++i ) {
        if( candPos[i] > lastDecidedAbsIndex_ && candPos[i] <= cutoff && keep[i] ) {
            PeakEvent pe;
            pe.sampleIndex = candPos[i]; // already Kilosort-convention-aligned, no shift needed
            pe.score       = candVal[i];
            peaksOut.push_back( pe );
            recentKeptPeaks_.push_back( std::make_pair( candPos[i], candVal[i] ) );
        }
    }

    lastDecidedAbsIndex_ = cutoff;

    // Nothing before (lastDecidedAbsIndex_ - minSeparationSamples_) can
    // ever be referenced by a future candidate window again -- trim it
    // (both the raw buffer and the persisted-anchor list).
    const long long trimBefore = lastDecidedAbsIndex_ - minSeparationSamples_;
    while( !dBuffer_.empty() && dBufferStartAbsIndex_ < trimBefore ) {
        dBuffer_.pop_front();
        ++dBufferStartAbsIndex_;
    }
    while( !recentKeptPeaks_.empty() && recentKeptPeaks_.front().first < trimBefore )
        recentKeptPeaks_.pop_front();
}


std::vector<PeakEvent> ConvolutionEngine::flush()
{
    std::vector<PeakEvent> peaks;
    if( !dBuffer_.empty() ) {
        const long long newestBuffered =
            dBufferStartAbsIndex_ + static_cast<long long>( dBuffer_.size() ) - 1;
        decideUpTo( newestBuffered, peaks );
    }
    return peaks;
}


std::vector<PeakEvent> ConvolutionEngine::processPrecomputedD(
    const double *Dvals, size_t nD, long long firstDAbsIndex )
{
    std::vector<PeakEvent> peaks;

    // Append newly computed D-values into the persistent score buffer,
    // skipping indices already appended by a previous call -- a chunk's
    // recomputation range overlaps the previous chunk's tail (same raw
    // history, so identical D values) because of leftMargin_/rightMargin_
    // overlap-save.
    for( size_t idx = 0; idx < nD; ++idx ) {

        const long long absIndex = firstDAbsIndex + static_cast<long long>( idx );

        if( dBuffer_.empty() ) {
            dBufferStartAbsIndex_ = absIndex;
            dBuffer_.push_back( Dvals[idx] );
        }
        else if( absIndex > dBufferStartAbsIndex_ + static_cast<long long>( dBuffer_.size() ) - 1 ) {
            dBuffer_.push_back( Dvals[idx] );
        }
        // else: absIndex already covered by a previous call -- skip.
    }

    // Finalize decisions up to (newestBuffered - finalizeMarginSamples_) --
    // see decideUpTo()'s derivation comment for why this needs the true
    // greedy selectByDistance() over a buffered window, not a per-index
    // local check. flush() (called once at the true end of a finite
    // stream, e.g. Calibration's training-file replay) uses the same
    // helper with cutoff = newestBuffered directly, since there's no more
    // future data to wait for.
    if( !dBuffer_.empty() ) {
        const long long newestBuffered =
            dBufferStartAbsIndex_ + static_cast<long long>( dBuffer_.size() ) - 1;
        decideUpTo( newestBuffered - finalizeMarginSamples_, peaks );
    }

    return peaks;
}


std::vector<PeakEvent> ConvolutionEngine::processChunk(
    const double *data, size_t nSamples, long long streamSampleOffset )
{
    std::vector<PeakEvent> peaks;
    if( nSamples == 0 )
        return peaks;

    const size_t L   = static_cast<size_t>( templateLength_ );
    const size_t nCh = static_cast<size_t>( nChannels_ );
    const size_t histLen = historyCount_;
    const size_t total   = histLen + nSamples;

    // combined = [history_ (histLen samples)] + [new chunk (nSamples samples)],
    // time-major/channel-minor, same layout as the preprocessed data.
    std::vector<double> combined( total * nCh );
    if( histLen > 0 )
        std::copy( history_.begin(), history_.begin() + histLen * nCh, combined.begin() );
    std::copy( data, data + nSamples * nCh, combined.begin() + histLen * nCh );

    // Absolute stream-sample index that combined[0] corresponds to.
    const long long firstCombinedAbsIndex = streamSampleOffset - static_cast<long long>( histLen );

    if( total >= L ) {

        // Valid centered-window positions n (combined-index) need
        // leftMargin_ samples before and rightMargin_ samples after.
        const size_t firstValid = static_cast<size_t>( leftMargin_ );
        const size_t lastValid  = total - 1 - static_cast<size_t>( rightMargin_ );
        const size_t nD         = lastValid - firstValid + 1;

        std::vector<double> Dvals( nD );

        for( size_t idx = 0; idx < nD; ++idx ) {

            const size_t n = firstValid + idx;             // combined-index this D value is centered on
            const size_t windowStart = n - leftMargin_;     // first (oldest) sample in the window
            double sum = 0.0;

            for( size_t k = 0; k < L; ++k ) {

                const double *samplePtr = &combined[(windowStart + k) * nCh];
                const double *tapPtr    = &taps_[k * nCh];

                for( size_t ch = 0; ch < nCh; ++ch )
                    sum += samplePtr[ch] * tapPtr[ch];
            }

            Dvals[idx] = sum;
        }

        // Buffering and deciding are delegated, NOT repeated here -- see
        // processPrecomputedD()'s header comment. One copy of that logic.
        peaks = processPrecomputedD(
            Dvals.data(), nD,
            firstCombinedAbsIndex + static_cast<long long>( firstValid ) );
    }

    // Keep the tail of combined (last L-1 samples) as history for next call
    // -- covers both leftMargin_ (for the next chunk's early indices) and
    // rightMargin_ (samples already seen but whose centered D value hasn't
    // been finalized yet).
    const size_t keep = std::min( total, L > 0 ? L - 1 : size_t(0) );
    history_.assign( combined.end() - static_cast<long>( keep * nCh ), combined.end() );
    historyCount_ = keep;

    return peaks;
}
