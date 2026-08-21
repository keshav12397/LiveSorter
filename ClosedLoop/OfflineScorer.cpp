#include "OfflineScorer.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <string>

#include "SglxMetaReader.h"
#include "MultiFilterBank.h"
#include "Preprocessor.h"
#include "MultiConvolutionEngine.h"


std::vector<UnitPeaks> scoreAllUnitsOffline(
    const std::string &binPath, const std::string &metaPath,
    const std::vector<int> &carChannelIds,
    bool applyHighpass, double highpassCutoffHz,
    const std::vector<int> &unitIds,
    const std::vector<int32_t> &unitChannelsInCarGroup,
    const std::vector<float> &unitFilters,
    int templateLength, int N,
    long long testStartSample, long long testNumSamples,
    int chunkSamples, long long minSeparationSamples, int detectionCapacity )
{
    SglxMeta meta = SglxMeta::load( metaPath );
    int nUnits = static_cast<int>( unitIds.size() );

    std::vector<int> channelPosition( meta.savedChannelIds.empty() ? 0 :
        *std::max_element( meta.savedChannelIds.begin(), meta.savedChannelIds.end() ) + 1, -1 );
    for( size_t i = 0; i < meta.savedChannelIds.size(); ++i )
        channelPosition[meta.savedChannelIds[i]] = static_cast<int>( i );

    const int nCarCh = static_cast<int>( carChannelIds.size() );
    std::vector<int> carColumns( nCarCh );
    for( int c = 0; c < nCarCh; ++c ) {
        int id = carChannelIds[c];
        if( id < 0 || id >= static_cast<int>( channelPosition.size() ) || channelPosition[id] < 0 )
            throw std::runtime_error( "scoreAllUnitsOffline: CAR channel " +
                std::to_string( id ) + " not found in '" + metaPath + "'s snsSaveChanSubset" );
        carColumns[c] = channelPosition[id];
    }

    // All -infinity thresholds -- every windowed-NMS-accepted peak is
    // reported unconditionally (see this file's header comment).
    std::vector<float> thresholds( nUnits, -std::numeric_limits<float>::infinity() );
    MultiFilterBank filterBank = MultiFilterBank::fromHostArrays(
        unitIds, unitChannelsInCarGroup, unitFilters, thresholds, N, templateLength );

    Preprocessor preprocessor( nCarCh, highpassCutoffHz, meta.sampleRateHz, applyHighpass, /*applyCar=*/true );
    // There is no detectionCapacity. An offline scoring pass with
    // all--infinity thresholds reports EVERY NMS-accepted peak, which a
    // fixed-size buffer could silently overflow. A std::vector has no
    // such cap, so that failure mode does not exist here.
    (void)detectionCapacity;
    MultiConvolutionEngine engine( filterBank, nCarCh, minSeparationSamples );

    std::ifstream inFh( binPath.c_str(), std::ios::binary );
    if( !inFh.is_open() )
        throw std::runtime_error( "scoreAllUnitsOffline: could not open '" + binPath + "'" );

    std::streamoff startByte = static_cast<std::streamoff>( testStartSample ) *
                                meta.nSavedChans * sizeof(short);
    inFh.seekg( startByte, std::ios::beg );

    std::vector<short> rawBuf( static_cast<size_t>( chunkSamples ) * meta.nSavedChans );
    std::vector<short> carChunkHost( static_cast<size_t>( chunkSamples ) * nCarCh );

    std::vector<UnitPeaks> result( nUnits );
    long long streamOffset = 0;

    for( ;; ) {

        if( testNumSamples > 0 && streamOffset >= testNumSamples )
            break;

        long long samplesToRead = chunkSamples;
        if( testNumSamples > 0 )
            samplesToRead = std::min<long long>( samplesToRead, testNumSamples - streamOffset );

        inFh.read( reinterpret_cast<char*>( rawBuf.data() ),
                   static_cast<std::streamsize>( samplesToRead ) * meta.nSavedChans * sizeof(short) );
        std::streamsize gotShorts = inFh.gcount() / sizeof(short);
        long long gotSamples = gotShorts / meta.nSavedChans;
        if( gotSamples <= 0 )
            break;

        for( long long s = 0; s < gotSamples; ++s ) {
            const short *srcRow = &rawBuf[static_cast<size_t>( s ) * meta.nSavedChans];
            short       *dstRow = &carChunkHost[static_cast<size_t>( s ) * nCarCh];
            for( int c = 0; c < nCarCh; ++c )
                dstRow[c] = srcRow[carColumns[c]];
        }

        std::vector<double> pre = preprocessor.processChunk(
            carChunkHost.data(), static_cast<size_t>( gotSamples ) );
        std::vector<MultiPeakEvent> peaks = engine.processChunk(
            pre.data(), static_cast<size_t>( gotSamples ), streamOffset );

        for( size_t p = 0; p < peaks.size(); ++p ) {
            int u = peaks[p].unitIndex;
            result[u].sampleIdx.push_back( peaks[p].sampleIndex );
            result[u].score.push_back( peaks[p].score );
        }

        streamOffset += gotSamples;

        if( gotSamples < samplesToRead )
            break; // short read -- reached EOF
    }

    // True end of this finite stream -- flush the tail engine.processChunk()
    // alone holds back forever waiting for future context that isn't coming
    // (see ConvolutionEngine::flush()'s comment).
    std::vector<MultiPeakEvent> flushed = engine.flush();
    for( size_t p = 0; p < flushed.size(); ++p ) {
        int u = flushed[p].unitIndex;
        result[u].sampleIdx.push_back( flushed[p].sampleIndex );
        result[u].score.push_back( flushed[p].score );
    }

    // No release(): MultiFilterBank owns nothing that needs freeing.
    return result;
}
