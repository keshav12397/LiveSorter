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
#include "StreamAccountant.h"

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
    // minSeparationSamples: templateLength/2, same convention every other
    // caller uses (Calibration.cpp's single-target CPU path, OfflineScorer.cu's
    // batch calibration scoring, every NMS equivalence test). Passed
    // explicitly for legibility, not to fix anything: GpuConvolutionEngine's
    // constructor already maps a non-positive value to templateLength/2, so
    // the earlier omission here was a no-op, not the "NO minimum separation
    // enforced" bug an earlier version of this comment claimed. That claim
    // came from misreading a live run in which NO unit tracked ground truth
    // -- the run's detections were being compared against a misaligned
    // ground truth (FilterGen/stream_alignment.py's docstring has the whole
    // story). Realigned, that same run's per-unit F1 matches calibration's
    // offline prediction at r=0.97.
    const long long minSep = filterBank_.templateLength / 2;
    GpuConvolutionEngine engine( filterBank_, nCarChans, maxChunkSamples, minSep );
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

    // Ring-buffer depth, in samples. sglx_fetch cannot serve a start index
    // older than this: the server answers "Too late" and those samples are
    // permanently gone. Measured at ~8.0 s on both streams of SpikeGLX
    // v20251218 (binary-searched how far back FETCH still succeeds). The
    // 0.75 factor keeps the resync target comfortably inside whatever the
    // real depth is on a given build, instead of aiming at the very edge
    // where it would immediately age out again.
    const long long ringDepthSamples     = static_cast<long long>( 8.0 * sampleRate );
    const long long resyncBackoffSamples = static_cast<long long>( 0.75 * ringDepthSamples );

    StreamAccountant acct;
    // Backlog costs an extra round-trip to ask for, so sample it
    // periodically rather than every chunk -- this loop runs at several
    // hundred Hz (see the latency logs referenced in the commit history).
    const long long backlogPollEveryNChunks = 200;

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
            // A failed fetch is NOT retryable from the same fromCt. The
            // dominant cause is "Too late": fromCt has aged out of the
            // server's ~8 s ring, and every future attempt at that same
            // index fails identically while the stream keeps advancing.
            // An earlier version of this loop slept and continue'd without
            // touching fromCt, so the first time this thread fell 8 s
            // behind it wedged permanently -- stderr forever, zero
            // detections for the rest of the session, and nothing in the
            // output saying so. Jump forward into a live part of the ring
            // instead, and count what was skipped as lost rather than
            // letting the record imply the stream was contiguous.
            std::cerr << "ImecFetchThreadGPU: sglx_fetch error: " << sglx_getError( hSglx_ ) << "\n";

            t_ull serverHead = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );
            if( serverHead == 0 ) {
                // Cannot even ask where the stream is -- server down or run
                // stopped. Back off and retry; nothing to record, since we
                // have no idea where we would be resyncing TO.
                std::this_thread::sleep_for( std::chrono::milliseconds( fetchChunkMs_ ) );
                continue;
            }

            t_ull resumeAt = ( static_cast<long long>( serverHead ) > resyncBackoffSamples )
                             ? serverHead - static_cast<t_ull>( resyncBackoffSamples )
                             : 0;

            if( resumeAt > fromCt ) {
                std::cerr << "ImecFetchThreadGPU: resyncing " << fromCt << " -> " << resumeAt
                          << " (" << (resumeAt - fromCt) << " samples lost)\n";
                acct.noteResync( static_cast<long long>( resumeAt ) );
                fromCt = resumeAt;
            }
            else {
                // Failed with fromCt still inside the ring -- a transient
                // (network hiccup, run paused). Leave fromCt alone so no
                // data is skipped, and just retry.
                ++acct.nFetchErrors;
            }

            std::this_thread::sleep_for( std::chrono::milliseconds( fetchChunkMs_ ) );
            continue;
        }

        long long tpts = static_cast<long long>( io.data.size() ) / nFetchChans;
        if( tpts <= 0 ) {
            // No new samples yet. Sleep a fraction of a chunk period rather
            // than spinning: with no pause at all this loop fires a fresh
            // TCP round-trip immediately, which on past runs drove it to
            // several hundred fetches/second for a median of 60 samples
            // each -- CPU and server time spent asking for data that
            // physically is not there yet. A full chunk period would be too
            // coarse (it would add up to fetchChunkMs of avoidable latency
            // to a spike arriving just after we looked), so wait a quarter.
            fromCt = headCt;
            std::this_thread::sleep_for(
                std::chrono::microseconds( std::max( 250, fetchChunkMs_ * 250 ) ) );
            continue;
        }
        if( tpts > maxChunkSamples ) {
            // Server returned more than fetchChunkMs's worth (e.g. after
            // this thread stalled) -- process only what fits this engine's
            // preallocated buffers, remaining data is picked up next
            // iteration since fromCt only advances by what was consumed.
            tpts = maxChunkSamples;
        }

        // Recorded BEFORE processing and with the post-clamp tpts, so the
        // accounting reflects what actually reaches the pipeline. A nonzero
        // gap means the server could not serve our requested start and
        // silently began us later; this is the only place that fact is
        // observable at all.
        long long gap = acct.noteFetch( static_cast<long long>( headCt ), tpts );
        if( gap > 0 ) {
            std::cerr << "ImecFetchThreadGPU: stream gap of " << gap
                      << " samples before headCt=" << headCt << "\n";
        }

        if( chunkIndex % backlogPollEveryNChunks == 0 ) {
            t_ull serverHead = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );
            if( serverHead != 0 )
                acct.noteBacklog( static_cast<long long>( serverHead ) );
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

        // Neither file is flushed per chunk any more. At the several-
        // hundred-chunks-per-second this loop really runs at, a flush per
        // chunk is a syscall in the sample-critical path (and on this
        // machine a OneDrive-synced write). Both are flushed once at
        // shutdown below, and ofstream's own buffering bounds what is at
        // risk meanwhile -- mainAllUnits.cpp's Ctrl+C path goes through
        // stop()/join(), so it reaches that shutdown flush normally.
        if( latencyLogFile.is_open() ) {
            double latencyMs = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
            latencyLogFile << chunkIndex << "," << tpts << "," << latencyMs << "\n";
        }
        ++chunkIndex;

        if( spikeTimesFile.is_open() && !detections.empty() ) {
            for( size_t d = 0; d < detections.size(); ++d ) {
                int unitId = filterBank_.hostUnitIds[detections[d].unitIndex];
                spikeTimesFile << unitId << "," << detections[d].sampleIndex
                                << "," << detections[d].score << "\n";
            }
        }

        fromCt = headCt + static_cast<t_ull>( tpts );
    }

    // Final backlog reading, so a run that ended while behind says so.
    {
        t_ull serverHead = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );
        if( serverHead != 0 )
            acct.noteBacklog( static_cast<long long>( serverHead ) );
    }

    spikeTimesFile.flush();
    if( latencyLogFile.is_open() )
        latencyLogFile.flush();

    std::cout << "\n" << acct.summary( "IMEC", sampleRate );
    if( !acct.balanced() || acct.nDropped > 0 ) {
        std::cerr << "ImecFetchThreadGPU: WARNING -- this run did not process every "
                     "sample the server produced (see the summary above).\n";
    }

    cudaFree( d_rawChunk );
    cudaFreeHost( hostCarChunk );
}
