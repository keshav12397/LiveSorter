#ifndef CLOSEDLOOP_OFFLINESCORER_H
#define CLOSEDLOOP_OFFLINESCORER_H

#include <vector>
#include <string>
#include <cstdint>

// One unit's full candidate-peak list (every windowed-NMS-accepted local
// maximum, not threshold-gated) over the scored range.
struct UnitPeaks {
    std::vector<long long> sampleIdx; // relative to testStartSample (0-based)
    std::vector<double>    score;
};

// Streams [testStartSample, testStartSample + testNumSamples) of the raw
// SpikeGLX .bin file through Preprocessor + MultiConvolutionEngine (both
// reused completely unmodified from the live detection pipeline -- see the
// FilterGen), using a MultiFilterBank built with an all
// -infinity threshold array (MultiFilterBank::fromHostArrays) so every
// windowed-NMS-accepted peak gets reported instead of only above-threshold
// detections -- this is what makes a held-out threshold SWEEP possible
// afterward by the caller, without a second NMS kernel to validate
// separately.
//
// unitChannelsInCarGroup: [nUnits * N], already translated to indices
// within the CAR/preprocessed channel group (0..carChannelIds.size()-1) --
// same convention ImecFetchThreadCpu::fetchLoop() produces for the live path.
std::vector<UnitPeaks> scoreAllUnitsOffline(
    const std::string &binPath, const std::string &metaPath,
    const std::vector<int> &carChannelIds,
    bool applyHighpass, double highpassCutoffHz,
    const std::vector<int> &unitIds,
    const std::vector<int32_t> &unitChannelsInCarGroup,
    const std::vector<float> &unitFilters,
    int templateLength, int N,
    long long testStartSample, long long testNumSamples,
    int chunkSamples, long long minSeparationSamples, int detectionCapacity );

#endif // CLOSEDLOOP_OFFLINESCORER_H
