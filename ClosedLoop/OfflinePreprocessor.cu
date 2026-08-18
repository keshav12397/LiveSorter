#include "OfflinePreprocessor.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

#include "CudaUtil.h"
#include "SglxMetaReader.h"
#include "GpuPreprocessor.h"


long long streamPreprocessToScratch(
    const std::string &binPath, const std::string &metaPath,
    const std::vector<int> &carChannelIds,
    bool applyHighpass, double highpassCutoffHz,
    long long maxSamples, int chunkSamples,
    const std::string &outScratchPath )
{
    SglxMeta meta = SglxMeta::load( metaPath );

    // Position of each CAR channel within a raw saved-channel row -- same
    // lookup Calibration.cpp already does for the single-target CPU path.
    std::vector<int> channelPosition( meta.savedChannelIds.empty() ? 0 :
        *std::max_element( meta.savedChannelIds.begin(), meta.savedChannelIds.end() ) + 1, -1 );
    for( size_t i = 0; i < meta.savedChannelIds.size(); ++i )
        channelPosition[meta.savedChannelIds[i]] = static_cast<int>( i );

    const int nCarCh = static_cast<int>( carChannelIds.size() );
    std::vector<int> carColumns( nCarCh );
    for( int c = 0; c < nCarCh; ++c ) {
        int id = carChannelIds[c];
        if( id < 0 || id >= static_cast<int>( channelPosition.size() ) || channelPosition[id] < 0 )
            throw std::runtime_error( "streamPreprocessToScratch: CAR channel " +
                std::to_string( id ) + " not found in '" + metaPath + "'s snsSaveChanSubset" );
        carColumns[c] = channelPosition[id];
    }

    std::ifstream inFh( binPath.c_str(), std::ios::binary );
    if( !inFh.is_open() )
        throw std::runtime_error( "streamPreprocessToScratch: could not open '" + binPath + "'" );

    std::ofstream outFh( outScratchPath.c_str(), std::ios::binary );
    if( !outFh.is_open() )
        throw std::runtime_error( "streamPreprocessToScratch: could not create '" + outScratchPath + "'" );

    GpuPreprocessor preprocessor( nCarCh, highpassCutoffHz, meta.sampleRateHz, applyHighpass, /*applyCar=*/true );

    std::vector<short>  rawRow( meta.nSavedChans );
    std::vector<short>  carChunkHost( static_cast<size_t>( chunkSamples ) * nCarCh );
    std::vector<float>  outChunkHost( static_cast<size_t>( chunkSamples ) * nCarCh );

    short *d_raw = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK( cudaMalloc( &d_raw, static_cast<size_t>( chunkSamples ) * nCarCh * sizeof(short) ) );
    CUDA_CHECK( cudaMalloc( &d_out, static_cast<size_t>( chunkSamples ) * nCarCh * sizeof(float) ) );

    long long totalWritten = 0;
    std::vector<short> rawBuf( static_cast<size_t>( chunkSamples ) * meta.nSavedChans );

    for( ;; ) {

        if( maxSamples > 0 && totalWritten >= maxSamples )
            break;

        long long samplesToRead = chunkSamples;
        if( maxSamples > 0 )
            samplesToRead = std::min<long long>( samplesToRead, maxSamples - totalWritten );

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

        preprocessor.processChunk( d_raw, static_cast<size_t>( gotSamples ), d_out, /*stream=*/nullptr );

        CUDA_CHECK( cudaMemcpy( outChunkHost.data(), d_out,
            static_cast<size_t>( gotSamples ) * nCarCh * sizeof(float), cudaMemcpyDeviceToHost ) );

        outFh.write( reinterpret_cast<const char*>( outChunkHost.data() ),
                     static_cast<std::streamsize>( gotSamples ) * nCarCh * sizeof(float) );

        totalWritten += gotSamples;

        if( gotSamples < samplesToRead )
            break; // short read -- reached EOF
    }

    cudaFree( d_raw );
    cudaFree( d_out );

    return totalWritten;
}
