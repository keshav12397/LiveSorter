#ifndef CLOSEDLOOP_GPUCONVOLUTIONENGINE_H
#define CLOSEDLOOP_GPUCONVOLUTIONENGINE_H

#include <vector>
#include <cstdint>

#include "GpuFilterBank.h"
#include "GpuPreprocessor.h"

// One accepted detection, as reported back to the host. unitIndex is a
// position into GpuFilterBank's arrays (0..nUnits-1) -- NOT the Kilosort
// cluster id; translate via GpuFilterBank::hostUnitIds[unitIndex]. Same
// sampleIndex convention as ConvolutionEngine.h's PeakEvent (the "same"-mode
// centered formula, directly comparable to spike_times.npy).
struct GpuPeakEvent {
    int       unitIndex;
    long long sampleIndex;
    float     score;
};

// Batched, GPU-resident equivalent of ConvolutionEngine.cpp -- computes the
// exact same centered-window matched-filter formula and windowed
// non-max-suppression decision rule (see that file's derivation comment;
// this is a port, not a re-derivation, for the same reason ConvolutionEngine
// itself insists on that: this codebase has twice silently lost detection
// recall/precision to a re-derived-and-wrong version of this math), for
// every unit in a GpuFilterBank at once, in one kernel launch per fetch
// chunk instead of one CPU object per unit.
//
// One known, deliberate simplification vs. ConvolutionEngine.cpp: the
// device-side channel-group history buffer is always a fixed
// (templateLength-1) samples, zero-initialized at construction, rather than
// exactly replicating ConvolutionEngine's variable-length ramp-up from zero
// history. This only affects the first (templateLength-1) samples
// (~2ms at 30kHz) of a session, before real history has accumulated -- a
// bounded, one-time startup transient, never recurring after that. Every
// chunk after the first behaves identically to the CPU algorithm.
class GpuConvolutionEngine {
public:
    // nChannelsGroup: size of the full CAR/preprocessed channel group this
    // engine reads from (e.g. 384) -- NOT nChannelsPerUnit.
    // filterBank.d_channels must already be translated to indices WITHIN
    // that group (see GpuFilterBank.h's d_channels comment) before this is
    // constructed.
    // maxChunkSamples: the largest nSamples processChunk() will ever be
    // called with (from fetchChunkMs * sampleRate) -- sizes every
    // persistent device buffer once at construction.
    // detectionCapacity: max detections buffered per processChunk() call
    // (default 4096, sized for live use where only rare above-threshold
    // events are expected). The offline batch calibration path (see
    // NoiseCovariance.h / the branch plan's Phase 4) uploads an all
    // -infinity threshold array via GpuFilterBank::fromHostArrays() so
    // EVERY windowed-NMS-accepted peak gets reported -- much higher volume,
    // needs a much larger capacity (and/or a smaller chunk size) to avoid
    // silently dropping detections past the cap.
    GpuConvolutionEngine( const GpuFilterBank &filterBank, int nChannelsGroup,
                           int maxChunkSamples, long long minSeparationSamples = 0,
                           int detectionCapacity = 4096 );
    ~GpuConvolutionEngine();

    GpuConvolutionEngine( const GpuConvolutionEngine & ) = delete;
    GpuConvolutionEngine &operator=( const GpuConvolutionEngine & ) = delete;

    // Runs preprocessor.processChunk() directly into this engine's history
    // buffer (avoiding an extra device-to-device copy -- see .cu), then the
    // batched matched-filter + NMS kernels, then copies back only the (small,
    // expected-sparse) list of accepted detections. d_raw: device
    // int16[nSamples * nChannelsGroup]. streamSampleOffset: absolute stream
    // sample index of d_raw's first sample. stream: cudaStream_t, passed as
    // void* so this header stays includable from plain .cpp files.
    std::vector<GpuPeakEvent> processChunk( GpuPreprocessor &preprocessor,
                                             const short *d_raw, size_t nSamples,
                                             long long streamSampleOffset, void *stream );

    // Forces a final decision round over whatever is currently buffered --
    // same rationale as ConvolutionEngine::flush() (see its header comment):
    // call ONCE at the true end of a finite stream (offline calibration
    // scoring; never a genuinely live/continuous session) to get the last
    // ~finalizeMarginSamples_ worth of candidates that processChunk() alone
    // holds back forever waiting for future context that isn't coming.
    std::vector<GpuPeakEvent> flush( void *stream );

private:
    int nUnits_;
    int nChannelsPerUnit_;
    int templateLength_;
    int nChannelsGroup_;
    int maxChunkSamples_;
    long long minSeparationSamples_;
    long long finalizeMarginSamples_; // see nmsDecideKernel's derivation comment in the .cu
    int leftMargin_, rightMargin_;

    const GpuFilterBank &filterBank_; // not owned

    float     *d_combined_;           // [(templateLength_-1+maxChunkSamples_) * nChannelsGroup_], history in the first (templateLength_-1) rows
    float     *d_historyScratch_;     // [(templateLength_-1) * nChannelsGroup_], staging for the overlap-save carry-forward -- see the .cu
    float     *d_dNew_;                // [nUnits_ * maxChunkSamples_] scratch, this chunk's fresh D-values

    int        dTailCap_;
    float     *d_dTail_;               // [nUnits_ * dTailCap_] persistent per-unit raw-D score tail
    int       *d_dTailCount_;          // [nUnits_] persistent
    long long *d_dTailStartAbsIndex_;  // [nUnits_] persistent
    long long *d_lastDecidedAbsIndex_; // [nUnits_] persistent

    // Already-finalized kept peaks, persisted across kernel launches -- GPU
    // analog of ConvolutionEngine's recentKeptPeaks_ (see its derivation
    // comment for why these must be fixed anchors, not re-derived from
    // d_dTail_ after trimming).
    int        keptCap_;
    long long *d_keptPos_;   // [nUnits_ * keptCap_]
    float     *d_keptVal_;   // [nUnits_ * keptCap_]
    int       *d_keptCount_; // [nUnits_]

    int             detectionCapacity_;
    int            *d_detectionCount_;  // [1]
    GpuPeakEvent   *d_detections_;      // [detectionCapacity_]
    std::vector<GpuPeakEvent> hostDetectionsBuf_; // reused host-side pinned-free scratch for the D2H copy
};

#endif // CLOSEDLOOP_GPUCONVOLUTIONENGINE_H
