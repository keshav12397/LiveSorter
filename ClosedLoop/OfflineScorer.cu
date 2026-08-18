#include "OfflineScorer.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <string>
#include <cuda_runtime.h>

#include "CudaUtil.h"
#include "SglxMetaReader.h"
#include "GpuFilterBank.h"
#include "GpuPreprocessor.h"
#include "GpuConvolutionEngine.h"


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
    GpuFilterBank filterBank = GpuFilterBank::fromHostArrays(
        unitIds, unitChannelsInCarGroup, unitFilters, thresholds, N, templateLength );

    GpuPreprocessor preprocessor( nCarCh, highpassCutoffHz, meta.sampleRateHz, applyHighpass, /*applyCar=*/true );
    GpuConvolutionEngine engine( filterBank, nCarCh, chunkSamples, minSeparationSamples, detectionCapacity );

    std::ifstream inFh( binPath.c_str(), std::ios::binary );
    if( !inFh.is_open() )
        throw std::runtime_error( "scoreAllUnitsOffline: could not open '" + binPath + "'" );

    std::streamoff startByte = static_cast<std::streamoff>( testStartSample ) *
                                meta.nSavedChans * sizeof(short);
    inFh.seekg( startByte, std::ios::beg );

    std::vector<short> rawBuf( static_cast<size_t>( chunkSamples ) * meta.nSavedChans );
    std::vector<short> carChunkHost( static_cast<size_t>( chunkSamples ) * nCarCh );

    short *d_raw = nullptr;
    CUDA_CHECK( cudaMalloc( &d_raw, static_cast<size_t>( chunkSamples ) * nCarCh * sizeof(short) ) );

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

        CUDA_CHECK( cudaMemcpy( d_raw, carChunkHost.data(),
            static_cast<size_t>( gotSamples ) * nCarCh * sizeof(short), cudaMemcpyHostToDevice ) );

        std::vector<GpuPeakEvent> peaks = engine.processChunk(
            preprocessor, d_raw, static_cast<size_t>( gotSamples ), streamOffset, /*stream=*/nullptr );

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
    // (see GpuConvolutionEngine::flush()'s comment).
    std::vector<GpuPeakEvent> flushed = engine.flush( /*stream=*/nullptr );
    for( size_t p = 0; p < flushed.size(); ++p ) {
        int u = flushed[p].unitIndex;
        result[u].sampleIdx.push_back( flushed[p].sampleIndex );
        result[u].score.push_back( flushed[p].score );
    }

    cudaFree( d_raw );
    filterBank.release();

    return result;
}
