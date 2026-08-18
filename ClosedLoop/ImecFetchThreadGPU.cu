#include "ImecFetchThreadGPU.h"
#include "CudaUtil.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

#include "SglxCppClient.h"
#include "DigitalWordUtils.h"
#include "SglxMetaReader.h"
#include "SyncEdgeTracker.h"
#include "GpuPreprocessor.h"
#include "GpuConvolutionEngine.h"

namespace {
    const int kImecJs = 2;
    const int kImecIp = 0;
}


ImecFetchThreadGPU::ImecFetchThreadGPU( void *hSglx, const GpuFilterBank &filterBank,
                                         const std::string &carChannelMapJsonPath,
                                         bool applyHighpass, double highpassCutoffHz,
                                         int imecSyncBit, int fetchChunkMs,
                                         const std::string &spikeTimesPath,
                                         const std::string &latencyLogPath )
    :   hSglx_( hSglx ), filterBank_( filterBank ),
        carChannelMapJsonPath_( carChannelMapJsonPath ),
        applyHighpass_( applyHighpass ), highpassCutoffHz_( highpassCutoffHz ),
        imecSyncBit_( imecSyncBit ), fetchChunkMs_( fetchChunkMs ),
        spikeTimesPath_( spikeTimesPath ), latencyLogPath_( latencyLogPath ),
        stopFlag_( false )
{}


void ImecFetchThreadGPU::start()
{
    thread_ = std::thread( &ImecFetchThreadGPU::fetchLoop, this );
}


void ImecFetchThreadGPU::stop()
{
    stopFlag_.store( true );
}


void ImecFetchThreadGPU::join()
{
    if( thread_.joinable() )
        thread_.join();
}


void ImecFetchThreadGPU::fetchLoop()
{
    std::ofstream spikeTimesFile( spikeTimesPath_.c_str(), std::ios::app );
    if( !spikeTimesFile.is_open() )
        std::cerr << "ImecFetchThreadGPU: could not open '" << spikeTimesPath_ << "' for writing\n";
    spikeTimesFile << "unit_id,sample_index,score\n";

    std::ofstream latencyLogFile;
    if( !latencyLogPath_.empty() ) {
        latencyLogFile.open( latencyLogPath_.c_str(), std::ios::app );
        if( !latencyLogFile.is_open() )
            std::cerr << "ImecFetchThreadGPU: could not open '" << latencyLogPath_ << "' for writing\n";
        else
            latencyLogFile << "chunk_index,n_samples,latency_ms\n";
    }
    long long chunkIndex = 0;

    double sampleRate = sglx_getStreamSampleRate( hSglx_, kImecJs, kImecIp );
    if( sampleRate <= 0 ) {
        std::cerr << "ImecFetchThreadGPU: sglx_getStreamSampleRate failed: "
                  << sglx_getError( hSglx_ ) << "\n";
        return;
    }

    cppClient_sglx_get_ints acqChans;
    if( !sglx_getStreamAcqChans( acqChans, hSglx_, kImecJs, kImecIp ) || acqChans.vint.size() < 3 ) {
        std::cerr << "ImecFetchThreadGPU: sglx_getStreamAcqChans failed: "
                  << sglx_getError( hSglx_ ) << "\n";
        return;
    }
    int syChannelIndex = acqChans.vint[0] + acqChans.vint[1];

    bool applyCar = !carChannelMapJsonPath_.empty();
    std::vector<int> carChannelIds = applyCar
        ? loadChanMapJson( carChannelMapJsonPath_ )
        : std::vector<int>(); // GPU pipeline requires a CAR group -- see below
    if( carChannelIds.empty() ) {
        std::cerr << "ImecFetchThreadGPU: carChannelMapJson is required for the "
                     "all-units GPU pipeline (unlike the single-target CPU path, "
                     "there's no per-unit fallback channel set to use instead)\n";
        return;
    }
    const int nCarChans = static_cast<int>( carChannelIds.size() );

    // Translate the filter bank's raw SpikeGLX channel indices to positions
    // within this CAR group -- same requirement/logic as
    // ImecFetchThread.cpp's filterIndexWithinCarGroup, just applied to every
    // unit's channels at once and re-uploaded to replace
    // filterBank_.d_channels in place (see GpuFilterBank.h's comment on that
    // field).
    {
        int nUnits = filterBank_.nUnits;
        int N = filterBank_.nChannelsPerUnit;
        std::vector<int32_t> rawChannels( static_cast<size_t>( nUnits ) * N );
        CUDA_CHECK( cudaMemcpy( rawChannels.data(), filterBank_.d_channels,
                                 rawChannels.size() * sizeof(int32_t), cudaMemcpyDeviceToHost ) );

        std::vector<int32_t> translated( rawChannels.size() );
        for( size_t i = 0; i < rawChannels.size(); ++i ) {
            std::vector<int>::const_iterator it =
                std::find( carChannelIds.begin(), carChannelIds.end(), rawChannels[i] );
            if( it == carChannelIds.end() ) {
                std::cerr << "ImecFetchThreadGPU: unit at index " << (i / N)
                          << " uses channel " << rawChannels[i]
                          << " which is not part of the CAR channel group\n";
                return;
            }
            translated[i] = static_cast<int32_t>( it - carChannelIds.begin() );
        }
        CUDA_CHECK( cudaMemcpy( filterBank_.d_channels, translated.data(),
                                 translated.size() * sizeof(int32_t), cudaMemcpyHostToDevice ) );
    }

    std::vector<int> channelSubset( carChannelIds.begin(), carChannelIds.end() );
    channelSubset.push_back( syChannelIndex );
    const int nFetchChans = static_cast<int>( channelSubset.size() );

    const long long chunkSamples = static_cast<long long>( fetchChunkMs_ / 1000.0 * sampleRate + 0.5 );
    const int maxChunkSamples = static_cast<int>( chunkSamples ) + 1; // small slack for rounding

    GpuPreprocessor preprocessor( nCarChans, highpassCutoffHz_, sampleRate, applyHighpass_, applyCar );
    GpuConvolutionEngine engine( filterBank_, nCarChans, maxChunkSamples );
    SyncEdgeTracker syncTracker( sampleRate );

    // Pinned host staging buffer for the CAR-group-only (SY stripped)
    // portion of each fetch -- this is the fastest the SDK allows (see
    // README.md's "fetch directly to GPU" note): sglx_fetch can only fill a
    // plain std::vector<short>, so this pinned buffer is filled by an
    // ordinary host-side copy, and ONLY the H2D leg after that uses
    // cudaMemcpyAsync from pinned memory.
    short *hostCarChunk = nullptr;
    CUDA_CHECK( cudaHostAlloc( reinterpret_cast<void**>( &hostCarChunk ),
                                static_cast<size_t>( maxChunkSamples ) * nCarChans * sizeof(short),
                                cudaHostAllocDefault ) );

    short *d_rawChunk = nullptr;
    CUDA_CHECK( cudaMalloc( &d_rawChunk, static_cast<size_t>( maxChunkSamples ) * nCarChans * sizeof(short) ) );

    t_ull fromCt = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );

    while( !stopFlag_.load() ) {

        cppClient_sglx_fetch io;
        io.channel_subset = &channelSubset[0];
        io.n_cs           = nFetchChans;
        io.max_samps      = static_cast<int>( chunkSamples );
        io.downsample     = 1;
        io.js             = kImecJs;
        io.ip             = kImecIp;

        t_ull headCt = sglx_fetch( io, hSglx_, fromCt );

        if( headCt == 0 ) {
            std::cerr << "ImecFetchThreadGPU: sglx_fetch error: " << sglx_getError( hSglx_ ) << "\n";
            std::this_thread::sleep_for( std::chrono::milliseconds( fetchChunkMs_ ) );
            continue;
        }

        long long tpts = static_cast<long long>( io.data.size() ) / nFetchChans;
        if( tpts <= 0 ) {
            fromCt = headCt;
            continue;
        }
        if( tpts > maxChunkSamples ) {
            // Server returned more than fetchChunkMs's worth (e.g. after
            // this thread stalled) -- process only what fits this engine's
            // preallocated buffers, remaining data is picked up next
            // iteration since fromCt only advances by what was consumed.
            tpts = maxChunkSamples;
        }

        for( long long t = 0; t < tpts; ++t ) {
            const short *src = &io.data[static_cast<size_t>( t ) * nFetchChans];
            short       *dst = &hostCarChunk[static_cast<size_t>( t ) * nCarChans];
            for( int c = 0; c < nCarChans; ++c )
                dst[c] = src[c];

            short syValue = src[nFetchChans - 1];
            int   syBit   = extractBit( syValue, imecSyncBit_ );
            syncTracker.update( syBit, static_cast<long long>( headCt ) + t );
        }

        CUDA_CHECK( cudaMemcpy( d_rawChunk, hostCarChunk,
                                 static_cast<size_t>( tpts ) * nCarChans * sizeof(short),
                                 cudaMemcpyHostToDevice ) );

        auto t0 = std::chrono::steady_clock::now();
        std::vector<GpuPeakEvent> detections = engine.processChunk(
            preprocessor, d_rawChunk, static_cast<size_t>( tpts ),
            static_cast<long long>( headCt ), nullptr );
        auto t1 = std::chrono::steady_clock::now();

        if( latencyLogFile.is_open() ) {
            double latencyMs = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
            latencyLogFile << chunkIndex << "," << tpts << "," << latencyMs << "\n";
            latencyLogFile.flush();
        }
        ++chunkIndex;

        if( spikeTimesFile.is_open() && !detections.empty() ) {
            for( size_t d = 0; d < detections.size(); ++d ) {
                int unitId = filterBank_.hostUnitIds[detections[d].unitIndex];
                spikeTimesFile << unitId << "," << detections[d].sampleIndex
                                << "," << detections[d].score << "\n";
            }
            spikeTimesFile.flush();
        }

        fromCt = headCt + static_cast<t_ull>( tpts );
    }

    cudaFree( d_rawChunk );
    cudaFreeHost( hostCarChunk );
}
