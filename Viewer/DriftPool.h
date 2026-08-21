#ifndef VIEWER_DRIFTPOOL_H
#define VIEWER_DRIFTPOOL_H

#include <vector>

// Port of FilterGen/drift_estimate.py's pooled_com_motion()'s inner
// `_pool` closure -- the median-across-units pooling step, taken in
// isolation from the per-unit centroid extraction that produces Y (see
// DriftTracker.h for that half, which has no Python analogue: it works
// from streamed per-spike amplitudes rather than pooled_com_motion's batch
// mean-waveform binning).
//
// Y[u][t] is unit u's own per-bin displacement (already y - mean(y) for
// that unit, i.e. pooled_com_motion's `disp`), NaN where the unit had no
// estimate for bin t. grid[t] is the bin's time in seconds, ascending.
// Returns the pooled trace, one value per bin:
//
//   1. median across units at each bin, ignoring NaN (numpy nanmedian
//      semantics: even count of finite values averages the two middle
//      ones)
//   2. a bin no unit covered is filled by linear interpolation from the
//      nearest covered bins on either side, edge-clamped beyond the first/
//      last covered bin (np.interp's default, NOT extrapolated)
//   3. the whole trace is re-centred to zero mean
//
// If every bin is uncovered (Y empty or all-NaN), returns a zero vector of
// the same length as `grid` -- pooled_com_motion's own fallback.
//
// test_livewire_roundtrip.cpp's drift-pool fixture checks this function
// against pooled_com_motion's actual return value (not a reimplementation
// of it) for a synthetic multi-unit recording.
std::vector<double> pooledMedianMotion( const std::vector<std::vector<double> > &Y,
                                         const std::vector<double> &grid );

#endif // VIEWER_DRIFTPOOL_H
