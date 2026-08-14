#ifndef CLOSEDLOOP_CONVOLUTIONENGINE_H
#define CLOSEDLOOP_CONVOLUTIONENGINE_H

#include <vector>
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

    // data: nSamples * nChannels int16 values, time-major/channel-minor
    // (data[t*nChannels+ch]) -- the same layout SpikeGLX's fetch API and
    // .bin files use. streamSampleOffset = absolute sample index of the
    // stream that data[0] corresponds to.
    std::vector<PeakEvent> processChunk( const short *data, size_t nSamples,
                                          long long streamSampleOffset );

private:
    int                 templateLength_;
    int                 nChannels_;
    std::vector<double> taps_;              // as loaded, row-major [t*nChannels_+ch]
    int                 leftMargin_;        // samples of history needed before a reported index n
    int                 rightMargin_;       // samples of "future" needed after n (the reporting delay)
    long long           minSeparationSamples_;

    // History retains leftMargin_ + rightMargin_ samples (== templateLength_-1)
    // so a fresh chunk can immediately compute D for indices near its start.
    std::vector<short>  history_;
    size_t              historyCount_;

    long long           lastDecidedAbsIndex_;   // last sample index already accept/reject-decided
    long long           lastAcceptedPeakIndex_;  // last sample index accepted as a peak
};

#endif // CLOSEDLOOP_CONVOLUTIONENGINE_H
