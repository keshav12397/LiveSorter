#ifndef CLOSEDLOOP_CALIBRATION_H
#define CLOSEDLOOP_CALIBRATION_H

#include <string>
#include <vector>
#include "FilterBank.h"

// Phase A: streams a training .bin recording through the exact same
// ConvolutionEngine used live, in fetchChunkMs-sized chunks, and sweeps
// detection thresholds against that recording's own Kilosort ground truth
// to pick a validated operating threshold -- the C++ equivalent of
// FilterGen/threshold_sweep_real.py's sweep, run once at startup instead of
// as a separate offline Python pass.
//
// Hard precondition (see README.md): trainingBinPath must be the exact
// file Kilosort (trainingKsDir) was run on.
class Calibration {
public:
    struct CurvePoint {
        double     threshold;
        long long  hits, misses, totalFalsePositives;
        double     recall, precision, f1, fpRateHz;
    };

    struct Result {
        double                    bestThreshold;
        CurvePoint                bestPoint;
        std::vector<CurvePoint>   curve;   // full sweep, for the log file / inspection
    };

    // filterBank supplies the channels + filter taps to convolve with;
    // its own `threshold` field is ignored (that's exactly what this
    // computes). criterion is "best_f1" or "best_recall_under_fp_rate".
    static Result run( const std::string &trainingBinPath,
                        const std::string &trainingKsDir,
                        int targetId,
                        const FilterBank &filterBank,
                        int fetchChunkMs,
                        const std::string &criterion,
                        double maxFalsePositiveRateHz,
                        const std::string &calibrationLogPath );
};

#endif // CLOSEDLOOP_CALIBRATION_H
