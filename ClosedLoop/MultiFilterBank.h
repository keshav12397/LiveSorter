#ifndef CLOSEDLOOP_MULTIFILTERBANK_H
#define CLOSEDLOOP_MULTIFILTERBANK_H

#include <string>
#include <vector>
#include <cstdint>

// Host-resident multi-unit filter bank, loaded once at startup from the same
// flat packed files FilterGen/calibrate_all_units.py writes:
//   unit_ids.bin    int32[nUnits]
//   channels.bin    int32[nUnits * nChannelsPerUnit]                     row-major
//   filters.bin     float32[nUnits * templateLength * nChannelsPerUnit]  row-major per unit
//   thresholds.bin  float32[nUnits]
//
// This is the CPU replacement for GpuFilterBank, and it deliberately keeps
// that struct's exact field names, semantics and call contract so the port
// is a substitution rather than a redesign -- the on-disk format, the
// "channels are raw SpikeGLX ids until the fetch thread translates them"
// convention, and updateFilters()'s meaning are all unchanged.
//
// Fixed nChannelsPerUnit/templateLength across every unit (see
// calibrate_all_units.py's --n-channels/--template-length), which is what
// lets every consumer index with a flat stride instead of a variable-length
// per-unit lookup.
//
// One real difference from the GPU version, and it is a simplification
// rather than a compromise: there is no device memory, so this type is
// ordinarily copyable, needs no release(), and cannot leak. The GPU version
// had to be move-only to stop two owners racing to cudaFree the same
// pointers; that whole class of failure is gone.
//
// filters are stored as float32, which is both the packed file layout and
// the precision calibrate_all_units.py picks thresholds in -- see
// FastMatchedFilter.h. MultiConvolutionEngine's production path convolves
// in that same float32; its float64 reference path widens the taps once at
// construction, because ConvolutionEngine works in double.
struct MultiFilterBank {
    int nUnits = 0;
    int nChannelsPerUnit = 0;
    int templateLength = 0;

    // int32[nUnits * nChannelsPerUnit]. As loaded these are raw SpikeGLX
    // channel indices (same convention as channels.bin and the existing
    // single-target FilterBank::channels) -- NOT yet indices into the
    // preprocessed CAR-group buffer the convolution actually reads from.
    // ImecFetchThreadCpu::fetchLoop() translates raw id -> CAR-group
    // position once at startup, into its own copy of this bank.
    std::vector<int32_t> channels;
    std::vector<float>   filters;     // [nUnits * templateLength * nChannelsPerUnit], [unit][t][c]
    std::vector<float>   thresholds;  // [nUnits]
    std::vector<int>     hostUnitIds; // Kilosort cluster ids, for logging

    // nChannelsPerUnit/templateLength must match what calibrate_all_units.py
    // was run with -- the packed files are flat arrays with no
    // self-description, so a mismatch here silently misinterprets the byte
    // layout rather than failing. Throws std::runtime_error if any file is
    // missing or its size is inconsistent with the given dimensions, which
    // is the only defence available against that.
    static MultiFilterBank load( const std::string &dir, int nChannelsPerUnit,
                                 int templateLength );

    // Builds a bank directly from in-memory arrays instead of the packed
    // files -- for the batch calibration path, where the bank is being
    // CONSTRUCTED rather than loaded. `channels` must already be translated
    // to indices within the CAR/preprocessed group (caller's
    // responsibility, matching the GPU version's convention).
    //
    // Pass all -infinity thresholds to make every NMS-accepted peak get
    // reported unconditionally, which is what offline scoring wants. On the
    // GPU that trick existed to avoid writing a second kernel variant; here
    // it is simply how the threshold comparison is expressed, since
    // ConvolutionEngine reports candidates and the caller thresholds them.
    static MultiFilterBank fromHostArrays( const std::vector<int> &unitIds,
                                           const std::vector<int32_t> &channels,
                                           const std::vector<float> &filters,
                                           const std::vector<float> &thresholds,
                                           int nChannelsPerUnit, int templateLength );

    // Overwrites ONE unit's channels/filter/threshold in place, for a drift
    // schedule swap event (see DriftSchedule.h). `channels` must already be
    // translated to CAR-group indices -- this does no translation itself,
    // matching fromHostArrays()'s convention. `unitIndex` indexes this
    // bank's own arrays (0..nUnits-1), not a Kilosort cluster id;
    // DriftSchedule::Event::unitIndex is already in that form.
    //
    // Nothing is reallocated: nUnits/nChannelsPerUnit/templateLength are
    // fixed at load() and a swap event is exactly one unit's slice of the
    // existing layout. Callers that hold a derived copy of the taps -- above
    // all MultiConvolutionEngine, which widens them to double -- must be
    // told separately; see MultiConvolutionEngine::updateUnit().
    void updateFilters( int unitIndex, const std::vector<int32_t> &channels,
                        const std::vector<float> &filter, float threshold );

    // Pointer to unit `u`'s taps, [templateLength * nChannelsPerUnit].
    const float *unitFilter( int u ) const
    { return filters.data() + static_cast<size_t>( u ) * templateLength * nChannelsPerUnit; }

    // Pointer to unit `u`'s channel list, [nChannelsPerUnit].
    const int32_t *unitChannels( int u ) const
    { return channels.data() + static_cast<size_t>( u ) * nChannelsPerUnit; }
};

#endif // CLOSEDLOOP_MULTIFILTERBANK_H
