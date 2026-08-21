#ifndef CLOSEDLOOP_NOISECOVARIANCE_H
#define CLOSEDLOOP_NOISECOVARIANCE_H

#include <vector>

// Batched port of generate_filter.noise_covariance_vectorized(), one job per
// unit. Each unit keeps its OWN small spike-free-segment mask (its own
// target + interferers, ~6 clusters, matching Python) and its OWN selected
// channels; the units run concurrently rather than as hundreds of
// sequential Python calls.
//
// THE EXCLUSION SET MUST STAY PER UNIT. The tempting simplification --
// compute one covariance for the whole probe and share it -- does not work:
// an all-clusters mask leaves 0 usable spike-free segments on a real, busy
// recording (checked empirically before this was written, and re-measured
// since at 99.5% of the train half spike-present). "Not noise" has to mean
// "could contaminate THESE channels", not "any spike anywhere on the
// probe". See FilterGen/REFIT_RESULTS.md for the measurement and
// FilterGen/banded_refit.py for the design that does work.
//
// Segment-finding uses interval merging (O(nSpikes log nSpikes)) rather
// than scanning a dense per-sample boolean mask over the whole recording
// (Python's approach, O(recording length) -- fine there, since Python calls
// it once per unit anyway). Interval-merge finding the same maximal gaps as
// a dense-mask diff is a different algorithm for the same result, so it is
// validated directly against Python's segment boundaries in the equivalence
// test rather than assumed.
//
// Known limitation: the full train split is held in memory at once (no
// chunked/streaming variant). Fine at ~96 channels on a single shank;
// a larger recording would need chunking with cross-chunk segment
// carry-over, the same pattern MultiConvolutionEngine uses for its NMS tail
// state.

struct SpikeFreeSegment {
    int start; // inclusive, sample index within the train split
    int end;   // exclusive
};

// Finds spike-free segments (length >= 2*templateLength) for one unit's
// combined target+interferer spike times, within [0, nSamples). Mirrors
// noise_covariance_vectorized's segment-finding (same templateOffset/
// templateLength margin around each spike, same minimum-length cutoff), via
// interval merging instead of a dense-mask scan.
std::vector<SpikeFreeSegment> findSpikeFreeSegments(
    const std::vector<long long> &spikeTimes, long long nSamples,
    int templateLength, int templateOffset );

// Runs every unit at once. dData: float32[nSamples * nChannelsGroup],
// the full preprocessed train split
// (row-major, time-major/channel-minor -- same layout OfflinePreprocessor
// writes). unitChannels: host int[nUnits * N], each unit's N selected
// channels as indices into the nChannelsGroup dimension (NOT raw SpikeGLX
// ids). segments/segmentOffsets/segmentCounts: per-unit segment lists,
// flattened (segmentOffsets[u]/segmentCounts[u] index into `segments`).
//
// Returns, per unit, its (templateLength*N, templateLength*N) row-major
// noise covariance matrix R -- same Toeplitz-block assembly
// noise_covariance_vectorized's tail does, run on the host from the small
// per-unit (nlags, N, N) accumulator this kernel produces.
std::vector<std::vector<double> > computeNoiseCovarianceBatched(
    const float *dData, long long nSamples, int nChannelsGroup,
    int templateLength, int N,
    const std::vector<int> &unitChannels,
    const std::vector<SpikeFreeSegment> &segments,
    const std::vector<int> &segmentOffsets,
    const std::vector<int> &segmentCounts,
    int nUnits );

#endif // CLOSEDLOOP_NOISECOVARIANCE_H
