#ifndef CLOSEDLOOP_OFFLINEPREPROCESSOR_H
#define CLOSEDLOOP_OFFLINEPREPROCESSOR_H

#include <string>
#include <vector>

// Streams a raw SpikeGLX .bin file through GpuPreprocessor (causal highpass
// + CAR, byte-identical code path to the live detection pipeline -- reusing
// it here removes the causal/acausal drift risk by construction, see the
// branch plan) and writes the result to a float32 scratch file on disk,
// shape (nSamples, carChannelIds.size()) row-major/C order -- same
// on-disk convention as FilterGen/threshold_sweep_real.py's
// load_and_prepare(..., order='C') scratch memmap, so downstream code
// (NoiseCovariance, LcmvFit's mean-waveform gather) can treat it identically
// to that file.
//
// carChannelIds: raw SpikeGLX channel ids to keep (e.g. one shank, from
// --channel-map-json) -- CAR is computed over this whole group, matching
// Preprocessor.h/GpuPreprocessor's existing requirement that CAR see the
// full trained channel group, not just one unit's 5 channels.
//
// Returns the number of samples actually written (bounded by
// maxSamples if given, e.g. last-spike-time + 1s, matching
// threshold_sweep_real.load_and_prepare's t_max_samples).
long long streamPreprocessToScratch(
    const std::string &binPath, const std::string &metaPath,
    const std::vector<int> &carChannelIds,
    bool applyHighpass, double highpassCutoffHz,
    long long maxSamples, int chunkSamples,
    const std::string &outScratchPath );

#endif // CLOSEDLOOP_OFFLINEPREPROCESSOR_H
