#include "ThresholdSweep.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <utility>


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

    std::vector<double> thresholds( nThresholds );
    for( int i = 0; i < nThresholds; ++i )
        thresholds[i] = ( nThresholds > 1 ) ? lo + ( hi - lo ) * i / ( nThresholds - 1 ) : lo;

    // Incremental sweep, threshold DESCENDING (highest first): the active
    // detection set {d : score[d] > threshold} only GROWS as threshold
    // falls, so each peak needs to be classified (hit-contributing or FP)
    // exactly once across the whole sweep, not once per threshold. The
    // original version recomputed hits/totalFp from scratch for every one
    // of the nThresholds (60) steps -- O(nThresholds x nPeaks log nTarget)
    // -- which, now that per-unit prep and the per-threshold re-sort were
    // already fixed (see git history), was left as THE dominant per-unit
    // cost in the batch calibration driver (measured ~1.2s/unit x ~150-500
    // units). This produces bit-identical (hits, totalFp) pairs per
    // threshold to the original brute-force computation -- same
    // window-matching definition (anyWithinWindow), just computed via an
    // incrementally-growing active set instead of rebuilding it from
    // scratch each time -- verified against the original loop's semantics:
    // - FP status of a detection depends only on the FIXED target-spike
    //   list, never on which OTHER detections are active, so it can be
    //   decided once, at the moment a detection is first activated.
    // - A target spike's hit status is monotonic (never un-hits as more
    //   detections activate), so "first activated detection landing in a
    //   target spike's window" is exactly when that target spike should
    //   flip to hit -- summing those flips as they occur reproduces the
    //   same hits count a from-scratch scan would find at that threshold.
    std::vector<std::pair<double, long long> > byScoreDesc( peakSampleIdx.size() );
    for( size_t p = 0; p < peakSampleIdx.size(); ++p )
        byScoreDesc[p] = std::make_pair( peakScores[p], peakSampleIdx[p] );
    std::sort( byScoreDesc.begin(), byScoreDesc.end(),
               []( const std::pair<double, long long> &a, const std::pair<double, long long> &b ) {
                   return a.first > b.first;
               } );

    std::vector<char> targetHit( static_cast<size_t>( nTarget ), 0 );
    long long hits = 0, totalFp = 0;
    size_t ptr = 0; // next not-yet-activated index into byScoreDesc

    std::vector<long long> hitsAt( nThresholds ), fpAt( nThresholds );

    for( int i = nThresholds - 1; i >= 0; --i ) {

        double threshold = thresholds[i];
        while( ptr < byScoreDesc.size() && byScoreDesc[ptr].first > threshold ) {

            long long d = byScoreDesc[ptr].second;

            if( !anyWithinWindow( d, targetSpikesSorted, window ) ) {
                ++totalFp;
            }
            else {
                // Mark every not-yet-hit target spike within [d-window, d+window]
                // -- small range in practice (window < the NMS minimum
                // separation between detections, so this rarely spans more
                // than one target spike), same window definition
                // anyWithinWindow already uses.
                std::vector<long long>::const_iterator loIt =
                    std::lower_bound( targetSpikesSorted.begin(), targetSpikesSorted.end(), d - window );
                std::vector<long long>::const_iterator hiIt =
                    std::upper_bound( targetSpikesSorted.begin(), targetSpikesSorted.end(), d + window );
                for( std::vector<long long>::const_iterator it = loIt; it != hiIt; ++it ) {
                    size_t idx = static_cast<size_t>( it - targetSpikesSorted.begin() );
                    if( !targetHit[idx] ) { targetHit[idx] = 1; ++hits; }
                }
            }

            ++ptr;
        }

        hitsAt[i] = hits;
        fpAt[i] = totalFp;
    }

    // Final threshold selection: same ascending-i, strict-> tie-break as
    // the original loop (lowest threshold wins on an exact f1 tie).
    ThresholdSweepResult best;
    best.bestThreshold = lo;
    best.recall = best.precision = best.f1 = best.fpRateHz = 0.0;
    best.hits = best.totalFalsePositives = 0;
    double bestF1 = -1.0;

    for( int i = 0; i < nThresholds; ++i ) {

        double recall = static_cast<double>( hitsAt[i] ) / std::max<long long>( nTarget, 1 );
        double precision = static_cast<double>( hitsAt[i] ) / std::max<long long>( hitsAt[i] + fpAt[i], 1 );
        double f1 = ( precision + recall > 0 ) ? 2 * precision * recall / ( precision + recall ) : 0.0;
        double fpRateHz = fpAt[i] / durationSeconds;

        if( f1 > bestF1 ) {
            bestF1 = f1;
            best.bestThreshold = thresholds[i];
            best.recall = recall;
            best.precision = precision;
            best.f1 = f1;
            best.fpRateHz = fpRateHz;
            best.hits = hitsAt[i];
            best.totalFalsePositives = fpAt[i];
        }
    }

    return best;
}
