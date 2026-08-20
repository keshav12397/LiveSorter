#ifndef CLOSEDLOOP_GPUFILTERBANK_H
#define CLOSEDLOOP_GPUFILTERBANK_H

#include <string>
#include <vector>
#include <cstdint>

// Device-resident multi-unit filter bank, loaded once at startup from the
// flat packed files FilterGen/calibrate_all_units.py writes:
//   unit_ids.bin    int32[nUnits]
//   channels.bin    int32[nUnits * nChannelsPerUnit]                  row-major
//   filters.bin     float32[nUnits * templateLength * nChannelsPerUnit]  row-major per unit
//   thresholds.bin  float32[nUnits]
//
// Fixed nChannelsPerUnit/templateLength across every unit (see
// calibrate_all_units.py's --n-channels/--template-length) -- this is what
// lets GpuConvolutionEngine's kernels index with a flat stride instead of a
// variable-length (CSR-style) per-unit lookup.
//
// Plain header (no CUDA types) so it's includable from ordinary .cpp
// translation units too -- only GpuFilterBank.cu needs cuda_runtime.h.
struct GpuFilterBank {
    int nUnits;
    int nChannelsPerUnit;
    int templateLength;

    // device, int32[nUnits * nChannelsPerUnit]. As loaded, these are raw
    // SpikeGLX channel indices (same convention as channels.bin / the
    // existing single-target FilterBank.channels) -- NOT yet indices into
    // the preprocessed CAR-group buffer GpuConvolutionEngine actually reads
    // from. ImecFetchThreadGPU::fetchLoop() translates raw id -> CAR-group
    // position once at startup and re-uploads the translated array over
    // this same pointer before entering the fetch loop proper (mirrors
    // ImecFetchThread.cpp's filterIndexWithinCarGroup translation).
    int   *d_channels;
    float *d_filters;     // device, float32[nUnits * templateLength * nChannelsPerUnit], row-major [unit][t][c]
    float *d_thresholds;  // device, float32[nUnits]

    std::vector<int> hostUnitIds; // kept host-side for logging (spikeTimesPath CSV) without a D2H round-trip

    GpuFilterBank();

    // nChannelsPerUnit/templateLength must match what calibrate_all_units.py
    // was run with -- there's no per-file self-description of these (the
    // packed files are flat arrays, not self-describing), so a mismatch
    // here would silently misinterpret the byte layout. Throws
    // std::runtime_error if any file is missing or its size is inconsistent
    // with the given dimensions.
    static GpuFilterBank load( const std::string &dir, int nChannelsPerUnit, int templateLength );

    // Builds a GpuFilterBank directly from in-memory host arrays instead of
    // reading the packed files -- for the batch calibration path
    // (mainCalibrateAllUnits.cu, see OfflineScorer.h), where the filter
    // bank is being CONSTRUCTED, not loaded from a previous run's output.
    // channels: int[nUnits * nChannelsPerUnit], already translated to
    // indices within the CAR/preprocessed channel group (same convention
    // ImecFetchThreadGPU::fetchLoop() produces for the live path -- caller's
    // responsibility here, this does no translation). thresholds: pass all
    // -infinity to make every windowed-NMS-accepted peak get reported
    // unconditionally (GpuConvolutionEngine's existing nmsDecideKernel
    // reused completely unmodified -- see NoiseCovariance.h's sibling
    // offline scoring driver for why this is preferred over a second kernel
    // variant).
    static GpuFilterBank fromHostArrays( const std::vector<int> &unitIds,
                                          const std::vector<int32_t> &channels,
                                          const std::vector<float> &filters,
                                          const std::vector<float> &thresholds,
                                          int nChannelsPerUnit, int templateLength );

    // Overwrites ONE unit's channels/filter/threshold in place, for a drift
    // schedule swap event (see DriftSchedule.h). `channels` must already be
    // translated to indices within the CAR-group buffer, same convention as
    // the rest of d_channels after ImecFetchThreadGPU::fetchLoop()'s startup
    // translation -- this function does no translation itself, matching
    // fromHostArrays()'s existing "caller's responsibility" convention.
    // unitIndex is an index into this bank's own arrays (0..nUnits-1), not a
    // Kilosort cluster id -- DriftSchedule::Event::unitIndex is already in
    // that form. Three small cudaMemcpys, no allocation: nUnits/
    // nChannelsPerUnit/templateLength are fixed at load() time and a swap
    // event's arrays are exactly one unit's slice of the existing layout.
    void updateFilters( int unitIndex, const std::vector<int32_t> &channels,
                         const std::vector<float> &filter, float threshold );

    // cudaFree everything. Safe to call more than once (idempotent) --
    // matters because both an explicit release() and the destructor may
    // run.
    void release();

    ~GpuFilterBank();

    // Move-only: this struct owns device memory, copying it would produce
    // two owners racing to free the same pointers.
    GpuFilterBank( GpuFilterBank &&other ) noexcept;
    GpuFilterBank &operator=( GpuFilterBank &&other ) noexcept;
    GpuFilterBank( const GpuFilterBank & ) = delete;
    GpuFilterBank &operator=( const GpuFilterBank & ) = delete;
};

#endif // CLOSEDLOOP_GPUFILTERBANK_H
