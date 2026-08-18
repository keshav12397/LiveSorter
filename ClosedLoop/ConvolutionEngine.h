#ifndef CLOSEDLOOP_CONVOLUTIONENGINE_H
#define CLOSEDLOOP_CONVOLUTIONENGINE_H

#include <vector>
#include <deque>
#include <utility>
#include <cstdint>

// One detected local maximum of the matched-filter output.
//
// sampleIndex uses exactly the same centering convention as
// FilterGen/generate_filter.py's filter_output() (np.convolve(..., f[::-1],
// mode='same')) -- i.e. it's directly comparable to Kilosort's own
// spike_times.npy with no extra correction needed. See the derivation
// comment in ConvolutionEngine.cpp for why this "same"-mode formula, not a
// separately-derived causal-index shift, is used: an earlier version of
// this file used a causal (no-lookahead) formula plus a hand-derived
// correction constant, computed that constant wrong, and it silently
// cratered calibration's recall/precision against ground truth. Matching
// Python's formula exactly (at the cost of a small, fixed ~(templateLength/2)
// -sample reporting delay -- trivial against a 10ms budget) removes that
// whole class of bug rather than requiring the derivation to be re-verified
// by hand.
struct PeakEvent {
    long long sampleIndex;
    double    score;
};

// The matched-filter convolution + peak detection core, shared verbatim by
// Calibration (sweeping many thresholds over a training file) and the live
// ImecFetchThread (checking each peak against one fixed threshold). This is
// the single place that "IS the filter" -- change the algorithm here once,
// both phases pick it up identically.
//
// Feed it successive chunks via processChunk(); it maintains its own
// history buffer internally and holds back reporting a detection until
// enough *future* samples (from a later chunk) have arrived to compute its
// centered value -- functionally equivalent to fetching overlapping
// windows from the server, but reusing already-fetched samples locally
// instead of re-requesting them.
class ConvolutionEngine {
public:
    // taps: exactly the array loaded from filter_<id>.bin (FilterBank::taps),
    // row-major [t*nChannels+ch], t = 0..templateLength-1.
    // minSeparationSamples: minimum gap enforced between accepted peaks;
    // defaults to templateLength/2 if 0 is passed (a spike can't
    // meaningfully re-trigger within half its own template width).
    ConvolutionEngine( int templateLength, int nChannels,
                        const std::vector<double> &taps,
                        long long minSeparationSamples = 0 );

    // Clears all internal state (history buffer, last-decided/accepted
    // indices) -- call before reusing one engine instance for a second,
    // unrelated stream (Calibration and live fetch each just construct
    // their own instance instead, so this mainly exists for completeness/
    // testing).
    void reset();

    // data: nSamples * nChannels PREPROCESSED (highpass+CAR'd, see
    // Preprocessor) double values, time-major/channel-minor
    // (data[t*nChannels+ch]). streamSampleOffset = absolute sample index of
    // the stream that data[0] corresponds to.
    std::vector<PeakEvent> processChunk( const double *data, size_t nSamples,
                                          long long streamSampleOffset );

    // Forces a final decision round over whatever is currently buffered,
    // treating the buffer's end as if no more data will ever arrive --
    // call this ONCE at the true end of a finite stream (e.g. Calibration's
    // training-file replay, or any other offline/batch use) to get the
    // last ~finalizeMarginSamples_ worth of candidates that processChunk()
    // alone would otherwise hold back forever waiting for future context
    // that will never come. Never call this on a genuinely live/continuous
    // stream (there IS no "end" to flush at) or on an engine you intend to
    // keep feeding afterward -- flushing commits to decisions made with
    // less lookahead than normal, which is fine at a true stream end but
    // would otherwise reduce accuracy for no reason.
    std::vector<PeakEvent> flush();

private:
    // Runs one decision round: finalizes every not-yet-decided candidate up
    // to and including `cutoff` (must be <= the newest buffered index),
    // appending accepted peaks to `peaksOut`. Shared by processChunk() and
    // flush() -- see processChunk()'s big derivation comment in the .cpp
    // for the algorithm itself.
    void decideUpTo( long long cutoff, std::vector<PeakEvent> &peaksOut );


    int                 templateLength_;
    int                 nChannels_;
    std::vector<double> taps_;              // as loaded, row-major [t*nChannels_+ch]
    int                 leftMargin_;        // samples of history needed before a reported index n
    int                 rightMargin_;       // samples of "future" needed after n (the reporting delay)
    long long           minSeparationSamples_;
    long long           finalizeMarginSamples_;  // see processChunk()'s decision-block comment

    // History retains leftMargin_ + rightMargin_ samples (== templateLength_-1)
    // so a fresh chunk can immediately compute D for indices near its start.
    std::vector<double> history_;
    size_t              historyCount_;

    // Trailing buffer of already-computed D (matched-filter output) values,
    // used for proper windowed non-max suppression -- see the derivation
    // comment in ConvolutionEngine.cpp. dBuffer_[i] is the D value at
    // absolute sample index (dBufferStartAbsIndex_ + i); only meaningful
    // while dBuffer_ is non-empty.
    std::deque<double> dBuffer_;
    long long          dBufferStartAbsIndex_;

    long long           lastDecidedAbsIndex_;   // last sample index already accept/reject-decided

    // Already-finalized (emitted) kept peaks close enough to the decision
    // frontier to still matter as competitors for not-yet-decided
    // candidates -- see processChunk()'s derivation comment for why these
    // must be reused as FIXED anchors rather than re-derived from dBuffer_
    // each round (re-deriving after dBuffer_ has been trimmed silently
    // loses the context that originally justified keeping them, causing
    // occasional incorrect decisions for nearby new candidates). Ascending
    // by position; trimmed the same way dBuffer_ is.
    std::deque<std::pair<long long, double> > recentKeptPeaks_;
};

#endif // CLOSEDLOOP_CONVOLUTIONENGINE_H
