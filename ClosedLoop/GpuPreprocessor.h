#ifndef CLOSEDLOOP_GPUPREPROCESSOR_H
#define CLOSEDLOOP_GPUPREPROCESSOR_H

#include <cstddef>

// Device-resident, batched equivalent of Preprocessor.h's two-stage
// pipeline (highpass each channel, THEN common-average-reference across the
// whole group) -- same order, same per-channel persistent IIR state, same
// median-based CAR -- run once over the FULL CAR channel group (e.g. 384
// channels) instead of duplicated per unit, since every unit's matched
// filter reads from this one shared preprocessed buffer.
//
// Plain header (no CUDA types) so it's includable from ordinary .cpp
// translation units; only GpuPreprocessor.cu needs cuda_runtime.h.
class GpuPreprocessor {
public:
    GpuPreprocessor( int nChannels, double highpassCutoffHz, double sampleRateHz,
                      bool applyHighpass, bool applyCar );
    ~GpuPreprocessor();

    GpuPreprocessor( const GpuPreprocessor & ) = delete;
    GpuPreprocessor &operator=( const GpuPreprocessor & ) = delete;

    // d_raw: device int16[nSamples * nChannels_], time-major/channel-minor
    // (same layout Preprocessor::processChunk takes as `short*`, just
    // already resident on-device).
    // d_out: device float32[nSamples * nChannels_], caller-owned -- written
    // directly (not returned/allocated here) so GpuConvolutionEngine can
    // pass a pointer into the middle of its own persistent "combined"
    // history+chunk buffer and skip an extra device-to-device copy.
    // Runs on `stream` so callers can pipeline this with other async work.
    void processChunk( const short *d_raw, size_t nSamples, float *d_out, void *stream );

    int nChannels() const { return nChannels_; }

private:
    int    nChannels_;
    bool   applyHighpass_;
    bool   applyCar_;
    float  b0_, b1_, b2_, a1_, a2_;   // biquad coefficients (same for every channel)
    float *d_biquadState_;           // device float32[nChannels_ * 4]: x1,x2,y1,y2 per channel, persists across calls
};

#endif // CLOSEDLOOP_GPUPREPROCESSOR_H
