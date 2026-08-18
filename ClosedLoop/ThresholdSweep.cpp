#include "ThresholdSweep.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>


double percentile( std::vector<double> sorted, double p )
{
    if( sorted.empty() )
        return 0.0;
    std::sort( sorted.begin(), sorted.end() );
    size_t idx = static_cast<size_t>( p / 100.0 * ( sorted.size() - 1 ) + 0.5 );
    idx = std::min( idx, sorted.size() - 1 );
    return sorted[idx];
}


bool anyWithinWindow( long long query, const std::vector<long long> &sortedArr, long long window )
{
    if( sortedArr.empty() )
        return false;

    std::vector<long long>::const_iterator it =
        std::lower_bound( sortedArr.begin(), sortedArr.end(), query );

    if( it != sortedArr.end() && std::llabs( *it - query ) <= window )
        return true;
    if( it != sortedArr.begin() && std::llabs( *( it - 1 ) - query ) <= window )
        return true;

    return false;
}


ThresholdSweepResult sweepThresholds(
    const std::vector<long long> &peakSampleIdx, const std::vector<double> &peakScores,
    const std::vector<long long> &targetSpikesSorted, long long window,
    int nThresholds, double durationSeconds )
{
    double lo = percentile( peakScores, 50.0 );
    double hi = percentile( peakScores, 99.99 );

    long long nTarget = static_cast<long long>( targetSpikesSorted.size() );

    ThresholdSweepResult best;
    best.bestThreshold = lo;
    best.recall = best.precision = best.f1 = best.fpRateHz = 0.0;
    best.hits = best.totalFalsePositives = 0;
    double bestF1 = -1.0;

    for( int i = 0; i < nThresholds; ++i ) {

        double threshold = ( nThresholds > 1 ) ? lo + ( hi - lo ) * i / ( nThresholds - 1 ) : lo;

        std::vector<long long> detections;
        for( size_t p = 0; p < peakSampleIdx.size(); ++p )
            if( peakScores[p] > threshold )
                detections.push_back( peakSampleIdx[p] );
        std::sort( detections.begin(), detections.end() ); // peakSampleIdx isn't guaranteed pre-sorted here (multi-chunk accumulation)

        long long hits = 0;
        for( long long t = 0; t < nTarget; ++t )
            if( anyWithinWindow( targetSpikesSorted[t], detections, window ) )
                ++hits;

        long long totalFp = 0;
        for( size_t d = 0; d < detections.size(); ++d )
            if( !anyWithinWindow( detections[d], targetSpikesSorted, window ) )
                ++totalFp;

        double recall = static_cast<double>( hits ) / std::max<long long>( nTarget, 1 );
        double precision = static_cast<double>( hits ) / std::max<long long>( hits + totalFp, 1 );
        double f1 = ( precision + recall > 0 ) ? 2 * precision * recall / ( precision + recall ) : 0.0;
        double fpRateHz = totalFp / durationSeconds;

        if( f1 > bestF1 ) {
            bestF1 = f1;
            best.bestThreshold = threshold;
            best.recall = recall;
            best.precision = precision;
            best.f1 = f1;
            best.fpRateHz = fpRateHz;
            best.hits = hits;
            best.totalFalsePositives = totalFp;
        }
    }

    return best;
}
