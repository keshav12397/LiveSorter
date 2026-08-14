#include "ConvolutionEngine.h"

#include <algorithm>
#include <climits>

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
// Peak decisions use proper windowed non-max suppression (accept n only if
// D[n] is the tallest value in [n-minSep, n+minSep]), matching
// scipy.signal.find_peaks(distance=minSep)'s behavior -- NOT a greedy
// "far enough from the last *accepted* peak" rule. The greedy version let a
// small spurious side-lobe a few samples before a real spike's true, taller
// peak claim the exclusion zone first and silently discard the real
// detection; recall capped around 50% even at the loosest possible
// threshold before this fix, which is only explainable by genuine spikes
// never reaching the candidate list, not a scoring problem.
//
// Deciding n this way needs D-values up to n+minSep, and D[n+minSep] itself
// needs rightMargin raw samples past that -- so the total reporting delay is
// (rightMargin + minSep) samples, held in dBuffer_/history_ respectively
// (functionally identical to fetching overlapping windows from the server,
// just reusing already-fetched samples locally instead). At 30kHz with a
// 61-tap filter and the default minSep=templateLength/2, that's ~60 samples
// (~2ms) -- trivial against a 10ms budget.
// -------------------------------------------------------------------------


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
                                    : templateLength / 2 )
{
    reset();
}


void ConvolutionEngine::reset()
{
    history_.clear();
    historyCount_ = 0;
    dBuffer_.clear();
    dBufferStartAbsIndex_ = 0;
    // Sentinel far enough negative that the very first real peak is always
    // eligible (no prior "last decided" index to collide with), while
    // staying safely away from LLONG_MIN so `n - lastDecidedAbsIndex_`-style
    // arithmetic can't overflow.
    lastDecidedAbsIndex_ = LLONG_MIN / 2;
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

        // Append newly computed D-values into the persistent score buffer,
        // skipping indices already appended by a previous call -- this
        // chunk's recomputation range overlaps the previous chunk's tail
        // (same raw history, so identical D values) because of leftMargin_/
        // rightMargin_ overlap-save.
        for( size_t idx = 0; idx < nD; ++idx ) {

            const long long absIndex = firstCombinedAbsIndex + static_cast<long long>( firstValid + idx );

            if( dBuffer_.empty() ) {
                dBufferStartAbsIndex_ = absIndex;
                dBuffer_.push_back( Dvals[idx] );
            }
            else if( absIndex > dBufferStartAbsIndex_ + static_cast<long long>( dBuffer_.size() ) - 1 ) {
                dBuffer_.push_back( Dvals[idx] );
            }
            // else: absIndex already covered by a previous call -- skip.
        }
    }

    // Decide every buffered index n whose full [n-minSep, n+minSep] window
    // of D-values is now available (i.e. enough *future* score values have
    // arrived), accepting n only if it's the tallest value in that window --
    // true non-max suppression, matching scipy.signal.find_peaks(distance=..).
    //
    // An earlier version accepted the first strict local maximum (immediate
    // neighbors only) that was simply >= minSeparationSamples_ samples past
    // the *last accepted* peak. That greedy left-to-right rule lets a small
    // spurious side-lobe a few samples *before* a real spike's true (taller)
    // peak claim the exclusion zone first, silently discarding the real
    // detection. It was found via calibration recall capping around 50% even
    // at the loosest possible threshold (accepting virtually every candidate
    // score) -- only explainable by genuine spikes never reaching the
    // candidate list at all, not by a scoring/thresholding problem.
    if( !dBuffer_.empty() ) {

        const long long newestBuffered =
            dBufferStartAbsIndex_ + static_cast<long long>( dBuffer_.size() ) - 1;

        long long n = lastDecidedAbsIndex_ + 1;
        if( n < dBufferStartAbsIndex_ )
            n = dBufferStartAbsIndex_;

        for( ; n + minSeparationSamples_ <= newestBuffered; ++n ) {

            const long long winLo = std::max( dBufferStartAbsIndex_, n - minSeparationSamples_ );
            const long long winHi = n + minSeparationSamples_; // <= newestBuffered by loop condition
            const double    dn    = dBuffer_[static_cast<size_t>( n - dBufferStartAbsIndex_ )];

            bool isPeak = true;
            for( long long m = winLo; m <= winHi && isPeak; ++m ) {
                if( m == n )
                    continue;
                if( dBuffer_[static_cast<size_t>( m - dBufferStartAbsIndex_ )] >= dn )
                    isPeak = false; // a taller (or exactly tied) neighbor wins
            }

            if( isPeak ) {
                PeakEvent pe;
                pe.sampleIndex = n; // already Kilosort-convention-aligned, no shift needed
                pe.score       = dn;
                peaks.push_back( pe );
            }

            lastDecidedAbsIndex_ = n;
        }

        // Nothing before (lastDecidedAbsIndex_ - minSeparationSamples_) can
        // ever be referenced by a future window again -- trim it.
        const long long trimBefore = lastDecidedAbsIndex_ - minSeparationSamples_;
        while( !dBuffer_.empty() && dBufferStartAbsIndex_ < trimBefore ) {
            dBuffer_.pop_front();
            ++dBufferStartAbsIndex_;
        }
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
