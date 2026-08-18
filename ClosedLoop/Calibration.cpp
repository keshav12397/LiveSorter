#include "Calibration.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <map>
#include <cmath>
#include <cstdlib>
#include <string>

#include "SglxMetaReader.h"
#include "ConvolutionEngine.h"
#include "NpyReader.h"
#include "Preprocessor.h"


namespace {

// True if `query` lies within +/- window of any value in `sortedArr`
// (ascending). O(log n) via binary search -- the C++ equivalent of
// FilterGen/threshold_sweep_real.py's match_within_window/np.searchsorted.
bool anyWithinWindow( long long query, const std::vector<long long> &sortedArr, long long window )
{
    if( sortedArr.empty() )
        return false;

    std::vector<long long>::const_iterator it =
        std::lower_bound( sortedArr.begin(), sortedArr.end(), query );

    if( it != sortedArr.end() && std::llabs( *it - query ) <= window )
        return true;
    if( it != sortedArr.begin() && std::llabs( *(it - 1) - query ) <= window )
        return true;

    return false;
}


double percentile( std::vector<double> sorted, double p )
{
    if( sorted.empty() )
        return 0.0;
    std::sort( sorted.begin(), sorted.end() );
    size_t idx = static_cast<size_t>( p / 100.0 * (sorted.size() - 1) + 0.5 );
    idx = std::min( idx, sorted.size() - 1 );
    return sorted[idx];
}

} // namespace


Calibration::Result Calibration::run(
    const std::string &trainingBinPath,
    const std::string &trainingKsDir,
    int targetId,
    const FilterBank &filterBank,
    const std::string &carChannelMapJsonPath,
    bool applyHighpass,
    double highpassCutoffHz,
    int fetchChunkMs,
    const std::string &criterion,
    double maxFalsePositiveRateHz,
    const std::string &calibrationLogPath )
{
    // ---- Load .meta, resolve channel positions --------------------------
    std::string metaPath = trainingBinPath.substr( 0, trainingBinPath.find_last_of( '.' ) ) + ".meta";
    SglxMeta meta = SglxMeta::load( metaPath );

    std::map<int, int> channelPosition; // channel id -> column position in each raw row
    for( size_t i = 0; i < meta.savedChannelIds.size(); ++i )
        channelPosition[meta.savedChannelIds[i]] = static_cast<int>( i );

    // CAR channel group: the full shank (or whatever group was used to
    // train this filter), NOT just filterBank's 5 channels -- see
    // Preprocessor.h for why that distinction matters. Falls back to just
    // the filter's own channels if no map is given (CAR effectively
    // disabled/degenerate in that case).
    bool applyCar = !carChannelMapJsonPath.empty();
    std::vector<int> carChannelIds = applyCar
        ? loadChanMapJson( carChannelMapJsonPath )
        : filterBank.channels;

    std::vector<int> carColumns( carChannelIds.size() ); // position within each raw row
    for( size_t i = 0; i < carChannelIds.size(); ++i ) {
        std::map<int, int>::const_iterator it = channelPosition.find( carChannelIds[i] );
        if( it == channelPosition.end() ) {
            throw std::runtime_error( "Calibration: CAR channel " +
                std::to_string( carChannelIds[i] ) + " not found in training .meta's snsSaveChanSubset" );
        }
        carColumns[i] = it->second;
    }

    // Where each of the filter's channels lands within the CAR group's
    // compact buffer, so we can extract them *after* CAR is applied.
    std::vector<int> filterIndexWithinCarGroup( filterBank.channels.size() );
    for( size_t j = 0; j < filterBank.channels.size(); ++j ) {
        std::vector<int>::const_iterator it =
            std::find( carChannelIds.begin(), carChannelIds.end(), filterBank.channels[j] );
        if( it == carChannelIds.end() ) {
            throw std::runtime_error( "Calibration: filter channel " +
                std::to_string( filterBank.channels[j] ) + " is not part of the CAR channel group" );
        }
        filterIndexWithinCarGroup[j] = static_cast<int>( it - carChannelIds.begin() );
    }

    // ---- Load ground truth ------------------------------------------------
    std::vector<long long> spikeTimes    = readNpy1D( trainingKsDir + "/spike_times.npy" );
    std::vector<long long> spikeClusters = readNpy1D( trainingKsDir + "/spike_clusters.npy" );

    if( spikeTimes.size() != spikeClusters.size() )
        throw std::runtime_error( "Calibration: spike_times.npy / spike_clusters.npy size mismatch" );

    std::vector<long long> targetSpikes;
    for( size_t i = 0; i < spikeTimes.size(); ++i )
        if( spikeClusters[i] == targetId && spikeTimes[i] >= 0 )
            targetSpikes.push_back( spikeTimes[i] );
    std::sort( targetSpikes.begin(), targetSpikes.end() );

    if( targetSpikes.empty() )
        throw std::runtime_error( "Calibration: no spikes for target " + std::to_string( targetId ) +
                                   " in " + trainingKsDir );

    // ---- Stream the training .bin through the SAME ConvolutionEngine ------
    std::ifstream fh( trainingBinPath.c_str(), std::ios::binary );
    if( !fh.is_open() )
        throw std::runtime_error( "Calibration: could not open '" + trainingBinPath + "'" );

    const int nCh    = filterBank.nChannels();
    const int nCarCh = static_cast<int>( carChannelIds.size() );
    const long long chunkSamples = static_cast<long long>(
        fetchChunkMs / 1000.0 * meta.sampleRateHz + 0.5 );

    ConvolutionEngine engine( filterBank.templateLength, nCh, filterBank.taps );
    Preprocessor preprocessor( nCarCh, highpassCutoffHz, meta.sampleRateHz, applyHighpass, applyCar );

    std::vector<short> rawChunk( static_cast<size_t>( chunkSamples ) * meta.nSavedChans );
    std::vector<short> carChunk( static_cast<size_t>( chunkSamples ) * nCarCh );
    std::vector<double> compactChunk( static_cast<size_t>( chunkSamples ) * nCh );

    std::vector<PeakEvent> allPeaks;
    long long streamSampleOffset = 0;

    for( ;; ) {

        fh.read( reinterpret_cast<char*>( &rawChunk[0] ),
                 rawChunk.size() * sizeof(short) );
        std::streamsize gotShorts = fh.gcount() / sizeof(short);
        long long gotSamples = gotShorts / meta.nSavedChans;

        if( gotSamples <= 0 )
            break;

        // Extract the full CAR group first...
        for( long long s = 0; s < gotSamples; ++s ) {
            const short *srcRow = &rawChunk[static_cast<size_t>( s ) * meta.nSavedChans];
            short       *dstRow = &carChunk[static_cast<size_t>( s ) * nCarCh];
            for( int c = 0; c < nCarCh; ++c )
                dstRow[c] = srcRow[carColumns[c]];
        }

        // ...run highpass+CAR across that FULL group...
        std::vector<double> preprocessed =
            preprocessor.processChunk( &carChunk[0], static_cast<size_t>( gotSamples ) );

        // ...and only THEN subset down to the filter's own channels.
        compactChunk.resize( static_cast<size_t>( gotSamples ) * nCh );
        for( long long s = 0; s < gotSamples; ++s ) {
            const double *srcRow = &preprocessed[static_cast<size_t>( s ) * nCarCh];
            double       *dstRow = &compactChunk[static_cast<size_t>( s ) * nCh];
            for( int c = 0; c < nCh; ++c )
                dstRow[c] = srcRow[filterIndexWithinCarGroup[c]];
        }

        std::vector<PeakEvent> peaks = engine.processChunk(
            &compactChunk[0], static_cast<size_t>( gotSamples ), streamSampleOffset );
        allPeaks.insert( allPeaks.end(), peaks.begin(), peaks.end() );

        streamSampleOffset += gotSamples;

        if( gotSamples < chunkSamples )
            break; // short read -- reached EOF
    }

    // True end of this finite training-file stream -- flush the tail
    // engine.processChunk() alone holds back forever waiting for future
    // context that isn't coming (see ConvolutionEngine::flush()'s
    // comment). Without this, the last ~finalizeMarginSamples_ worth of
    // the file would never contribute any peaks to the sweep below.
    std::vector<PeakEvent> flushedPeaks = engine.flush();
    allPeaks.insert( allPeaks.end(), flushedPeaks.begin(), flushedPeaks.end() );

    if( allPeaks.empty() )
        throw std::runtime_error( "Calibration: convolution produced zero candidate peaks -- "
                                   "check the filter/training-file channel mapping" );

    double durationSeconds = static_cast<double>( streamSampleOffset ) / meta.sampleRateHz;
    long long window = filterBank.templateLength / 4;

    // ---- Sweep thresholds ---------------------------------------------------
    std::vector<double> scores;
    scores.reserve( allPeaks.size() );
    for( size_t i = 0; i < allPeaks.size(); ++i )
        scores.push_back( allPeaks[i].score );

    double lo = percentile( scores, 50.0 );
    double hi = percentile( scores, 99.99 );

    const int nThresholds = 60;
    Result result;
    result.curve.reserve( nThresholds );

    for( int i = 0; i < nThresholds; ++i ) {

        double threshold = lo + (hi - lo) * i / (nThresholds - 1);

        std::vector<long long> detections;
        for( size_t p = 0; p < allPeaks.size(); ++p )
            if( allPeaks[p].score > threshold )
                detections.push_back( allPeaks[p].sampleIndex );

        long long hits = 0;
        for( size_t t = 0; t < targetSpikes.size(); ++t )
            if( anyWithinWindow( targetSpikes[t], detections, window ) )
                ++hits;
        long long misses = static_cast<long long>( targetSpikes.size() ) - hits;

        long long totalFp = 0;
        for( size_t d = 0; d < detections.size(); ++d )
            if( !anyWithinWindow( detections[d], targetSpikes, window ) )
                ++totalFp;

        CurvePoint pt;
        pt.threshold           = threshold;
        pt.hits                = hits;
        pt.misses              = misses;
        pt.totalFalsePositives = totalFp;
        pt.recall              = static_cast<double>( hits ) / std::max<long long>( targetSpikes.size(), 1 );
        pt.precision           = static_cast<double>( hits ) / std::max<long long>( hits + totalFp, 1 );
        pt.f1 = (pt.precision + pt.recall > 0)
              ? 2 * pt.precision * pt.recall / (pt.precision + pt.recall)
              : 0.0;
        pt.fpRateHz = totalFp / durationSeconds;

        result.curve.push_back( pt );
    }

    // ---- Pick operating point -----------------------------------------------
    size_t bestIdx = 0;
    if( criterion == "best_recall_under_fp_rate" ) {
        double bestRecall = -1.0;
        bool found = false;
        for( size_t i = 0; i < result.curve.size(); ++i ) {
            if( result.curve[i].fpRateHz <= maxFalsePositiveRateHz && result.curve[i].recall > bestRecall ) {
                bestRecall = result.curve[i].recall;
                bestIdx = i;
                found = true;
            }
        }
        if( !found ) {
            // Fall back to the lowest achievable FP rate in the sweep.
            double bestFp = 1e300;
            for( size_t i = 0; i < result.curve.size(); ++i )
                if( result.curve[i].fpRateHz < bestFp ) { bestFp = result.curve[i].fpRateHz; bestIdx = i; }
        }
    }
    else { // "best_f1" (default)
        double bestF1 = -1.0;
        for( size_t i = 0; i < result.curve.size(); ++i )
            if( result.curve[i].f1 > bestF1 ) { bestF1 = result.curve[i].f1; bestIdx = i; }
    }

    result.bestThreshold = result.curve[bestIdx].threshold;
    result.bestPoint     = result.curve[bestIdx];

    // ---- Write calibration log ------------------------------------------------
    if( !calibrationLogPath.empty() ) {
        std::ofstream log( calibrationLogPath.c_str() );
        if( log.is_open() ) {
            log << "threshold,hits,misses,total_fp,recall,precision,f1,fp_rate_hz\n";
            for( size_t i = 0; i < result.curve.size(); ++i ) {
                const CurvePoint &pt = result.curve[i];
                log << pt.threshold << "," << pt.hits << "," << pt.misses << ","
                    << pt.totalFalsePositives << "," << pt.recall << "," << pt.precision
                    << "," << pt.f1 << "," << pt.fpRateHz << "\n";
            }
        }
    }

    return result;
}
