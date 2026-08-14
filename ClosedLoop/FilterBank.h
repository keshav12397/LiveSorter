#ifndef CLOSEDLOOP_FILTERBANK_H
#define CLOSEDLOOP_FILTERBANK_H

#include <string>
#include <vector>
#include <cstdint>

// Loads a single target neuron's LCMV filter, exactly as written by
// FilterGen/generate_filter.py's save_filter():
//
//   channels_<id>.bin  raw int32[n_channels]     -- SpikeGLX channel indices
//   filter_<id>.bin    raw float64[T][n_channels] -- row-major, index = t*n_channels+ch
//   threshold_<id>.bin raw float64[1]
//
// `threshold` is intentionally mutable after loading: Calibration may
// overwrite it with a freshly re-derived value before the live threads
// start (see Calibration.h). Everything else is fixed once loaded.
struct FilterBank {
    int                 targetId;
    int                 templateLength;
    std::vector<int>    channels;    // size n_channels, raw SpikeGLX channel indices
    std::vector<double> taps;        // size templateLength * n_channels, row-major [t*n_channels+ch]
    double              threshold;

    int nChannels() const { return static_cast<int>( channels.size() ); }

    // Throws std::runtime_error if any of the three files are missing/malformed,
    // or if file sizes are inconsistent with templateLength/channel count.
    static FilterBank load( const std::string &filterDir, int targetId, int templateLength );

    // Overwrites just the threshold file on disk (e.g. after calibration
    // picks a new value you want to persist for next time). Does not
    // touch channels_*.bin / filter_*.bin.
    void saveThreshold( const std::string &filterDir ) const;
};

#endif // CLOSEDLOOP_FILTERBANK_H
