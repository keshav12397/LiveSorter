#ifndef CLOSEDLOOP_THRESHOLDSWEEP_H
#define CLOSEDLOOP_THRESHOLDSWEEP_H

#include <vector>

// Same threshold-sweep logic Calibration.cpp already runs for the single-
// target CPU path (percentile-based threshold range, hits/misses/fp counts
// via sorted binary search, best-F1 selection) -- lifted into a standalone,
// per-unit-callable module for the batch GPU calibration path (see the
// branch plan's Phase 4) instead of duplicated inline, and instead of
// touching Calibration.cpp itself (which stays exactly as validated for the
// single-target path).
//
// Mirrors FilterGen/threshold_sweep_real.py's sweep_thresholds() (total_fp
// only -- this doesn't track per-interferer FP breakdown, since the batch
// calibration script doesn't consume that either).

double percentile( std::vector<double> sorted, double p );

// True if `query` lies within +/- window of any value in `sortedArr`
// (ascending, already sorted).
bool anyWithinWindow( long long query, const std::vector<long long> &sortedArr, long long window );

struct ThresholdSweepResult {
    double bestThreshold;
    double recall, precision, f1, fpRateHz;
    long long hits, totalFalsePositives;
};

// peakSampleIdx/peakScores: every candidate peak (ALL windowed-NMS-accepted
// local maxima, not threshold-gated -- see MultiFilterBank::fromHostArrays's
// -infinity-threshold trick), same convention as
// threshold_sweep_real.find_all_peaks's output. targetSpikesSorted: the
// unit's own held-out test-split spike times (ascending). window: hit/FP
// matching tolerance (template_length/4, matching Python).
// durationSeconds: test-split duration, for the FP rate.
ThresholdSweepResult sweepThresholds(
    const std::vector<long long> &peakSampleIdx, const std::vector<double> &peakScores,
    const std::vector<long long> &targetSpikesSorted, long long window,
    int nThresholds, double durationSeconds );

#endif // CLOSEDLOOP_THRESHOLDSWEEP_H
