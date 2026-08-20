#ifndef CLOSEDLOOP_NOISECOVARIANCE_H
#define CLOSEDLOOP_NOISECOVARIANCE_H

#include <vector>

// GPU-batched port of generate_filter.noise_covariance_vectorized(), one
// block per unit (same batching pattern as MultiConvolutionEngine's
// matchedFilterKernel) -- each unit keeps its OWN small spike-free-segment
// mask (its own target+interferers, ~6 clusters, matching Python) and its
// OWN selected channels, but all units' blocks run concurrently instead of
// 500 sequential Python calls. See the branch plan's Phase 2 for why the
// original "compute once for the whole probe" idea doesn't work (an
// all-clusters mask leaves 0 usable spike-free segments on a real, busy
// recording -- checked empirically before writing this).
//
// Segment-finding itself runs on the HOST (not the GPU): given a small spike
// list (a single unit's target+interferers), the spike-free segments are
// found via interval merging (O(nSpikes log nSpikes)), NOT by scanning a
// dense per-sample boolean mask over the whole recording (Python's
// approach, O(recording length) -- fine there since Python only calls this
// once per unit anyway, but avoiding it here means the GPU kernel never
// needs an O(recording length) per-unit CPU pass to prepare its input).
// Interval-merge finding the same maximal gaps as a dense-mask diff is a
// different algorithm for the same result -- validated directly against
// Python's segment boundaries as part of this phase's equivalence test, not
// just assumed equivalent.
//
// Known limitation (documented, not yet hit in practice): the full train
// split must fit in GPU memory at once (no chunked/streaming variant yet).
// Fine for the current ~22min test recording (~96 channels, single shank);
// would need chunking with cross-chunk segment carry-over (same pattern
// MultiConvolutionEngine already uses for its NMS tail state) if a future
// recording's single-shank preprocessed data doesn't fit in ~24GB.

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

// Runs the batched GPU kernel for every unit at once. dData: device
// float32[nSamples * nChannelsGroup], the full preprocessed train split
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
