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
// no correction, at the cost of a small fixed reporting delay (rightMargin
// samples -- ~1ms at 30kHz for a 61-tap filter -- since a detection at n
// can't be finalized until n+rightMargin has arrived in a later chunk).
// That delay is absorbed by the existing history/overlap buffering below;
// it's functionally identical to fetching overlapping windows from the
// server, just reusing already-fetched samples locally instead.
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
    // Sentinels far enough negative that the very first real peak is
    // always eligible (no prior "last accepted"/"last decided" to collide
    // with), while staying safely away from LLONG_MIN so `absIndex -
    // lastAcceptedPeakIndex_` can't overflow.
    lastDecidedAbsIndex_    = LLONG_MIN / 2;
    lastAcceptedPeakIndex_  = LLONG_MIN / 2;
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

        // Local maxima with minimum separation. Only interior points of
        // Dvals can be verified as local maxima (need both neighbors); the
        // very last point is deliberately left un-decided -- it will be
        // re-examined next call once its true right-neighbor sample is
        // available, via lastDecidedAbsIndex_.
        for( size_t idx = 1; idx + 1 < nD; ++idx ) {

            const long long absIndex = firstCombinedAbsIndex + static_cast<long long>( firstValid + idx );

            if( absIndex <= lastDecidedAbsIndex_ )
                continue; // already decided in a previous call

            if( Dvals[idx] > Dvals[idx - 1] && Dvals[idx] > Dvals[idx + 1] ) {

                if( absIndex - lastAcceptedPeakIndex_ >= minSeparationSamples_ ) {
                    PeakEvent pe;
                    pe.sampleIndex = absIndex; // already Kilosort-convention-aligned, no shift needed
                    pe.score       = Dvals[idx];
                    peaks.push_back( pe );
                    lastAcceptedPeakIndex_ = absIndex;
                }
            }

            lastDecidedAbsIndex_ = absIndex;
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
