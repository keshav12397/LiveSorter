#include "ImecFetchThread.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdexcept>

#include "SglxCppClient.h"
#include "DigitalWordUtils.h"
#include "SglxMetaReader.h"
#include "Preprocessor.h"

namespace {
    const int   kImecJs = 2;   // IMEC AP stream selector, per SglxApi.h
    const int   kImecIp = 0;   // first (only) imec probe
}


ImecFetchThread::ImecFetchThread( void *hSglx, FilterBank &filterBank,
                                   const std::string &carChannelMapJsonPath,
                                   bool applyHighpass, double highpassCutoffHz,
                                   int imecSyncBit, int fetchChunkMs,
                                   SpikeQueue &spikeQueue,
                                   const std::string &spikeTimesPath )
    :   hSglx_( hSglx ), filterBank_( filterBank ),
        carChannelMapJsonPath_( carChannelMapJsonPath ),
        applyHighpass_( applyHighpass ), highpassCutoffHz_( highpassCutoffHz ),
        imecSyncBit_( imecSyncBit ), fetchChunkMs_( fetchChunkMs ), spikeQueue_( spikeQueue ),
        spikeTimesPath_( spikeTimesPath ), stopFlag_( false )
{}


void ImecFetchThread::start()
{
    thread_ = std::thread( &ImecFetchThread::fetchLoop, this );
}


void ImecFetchThread::stop()
{
    stopFlag_.store( true );
}


void ImecFetchThread::join()
{
    if( thread_.joinable() )
        thread_.join();
}


void ImecFetchThread::fetchLoop()
{
    std::ofstream spikeTimesFile( spikeTimesPath_.c_str(), std::ios::app );
    if( !spikeTimesFile.is_open() )
        std::cerr << "ImecFetchThread: could not open '" << spikeTimesPath_ << "' for writing\n";

    double sampleRate = sglx_getStreamSampleRate( hSglx_, kImecJs, kImecIp );
    if( sampleRate <= 0 ) {
        std::cerr << "ImecFetchThread: sglx_getStreamSampleRate failed: "
                  << sglx_getError( hSglx_ ) << "\n";
        return;
    }

    // Resolve the SY channel's index dynamically -- never hardcode it.
    // {AP,LF,SY} ordering per SglxApi.h; SY starts right after AP+LF.
    cppClient_sglx_get_ints acqChans;
    if( !sglx_getStreamAcqChans( acqChans, hSglx_, kImecJs, kImecIp ) || acqChans.vint.size() < 3 ) {
        std::cerr << "ImecFetchThread: sglx_getStreamAcqChans failed: "
                  << sglx_getError( hSglx_ ) << "\n";
        return;
    }
    int syChannelIndex = acqChans.vint[0] + acqChans.vint[1]; // nAP + nLF

    // CAR channel group: the FULL group the filter was trained with (see
    // Preprocessor.h) -- NOT just the filter's own 5 channels. Falls back
    // to just the filter's channels (CAR disabled) if no map is given.
    bool applyCar = !carChannelMapJsonPath_.empty();
    std::vector<int> carChannelIds = applyCar
        ? loadChanMapJson( carChannelMapJsonPath_ )
        : filterBank_.channels;

    std::vector<int> filterIndexWithinCarGroup( filterBank_.channels.size() );
    for( size_t j = 0; j < filterBank_.channels.size(); ++j ) {
        std::vector<int>::const_iterator it =
            std::find( carChannelIds.begin(), carChannelIds.end(), filterBank_.channels[j] );
        if( it == carChannelIds.end() ) {
            std::cerr << "ImecFetchThread: filter channel " << filterBank_.channels[j]
                      << " is not part of the CAR channel group\n";
            return;
        }
        filterIndexWithinCarGroup[j] = static_cast<int>( it - carChannelIds.begin() );
    }

    // Fetch subset = the full CAR group, plus the SY channel appended last.
    std::vector<int> channelSubset( carChannelIds.begin(), carChannelIds.end() );
    channelSubset.push_back( syChannelIndex );
    const int nFetchChans  = static_cast<int>( channelSubset.size() );
    const int nCarChans    = static_cast<int>( carChannelIds.size() );
    const int nFilterChans = filterBank_.nChannels();

    const long long chunkSamples = static_cast<long long>( fetchChunkMs_ / 1000.0 * sampleRate + 0.5 );

    // minSeparationSamples: templateLength/2, same convention Calibration.cpp's
    // own held-out threshold sweep already uses to pick this filter's
    // threshold in the first place. Passed explicitly for legibility, not to
    // fix anything: ConvolutionEngine's constructor already maps a
    // non-positive value to templateLength/2, so the earlier omission here
    // was a no-op rather than the "no minimum separation enforced" bug an
    // earlier version of this comment claimed -- see ImecFetchThreadCpu.cpp's
    // matching comment and FilterGen/stream_alignment.py's docstring for
    // where that misreading came from.
    const long long minSep = filterBank_.templateLength / 2;
    ConvolutionEngine engine( filterBank_.templateLength, nFilterChans, filterBank_.taps, minSep );
    Preprocessor       preprocessor( nCarChans, highpassCutoffHz_, sampleRate, applyHighpass_, applyCar );
    SyncEdgeTracker    syncTracker( sampleRate );

    t_ull fromCt = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );

    std::vector<short> carChunk;   // reused buffer for the full CAR-group subset, de-interleaved
    std::vector<double> compact;   // reused buffer for the filter-channel subset, post highpass+CAR

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
            std::cerr << "ImecFetchThread: sglx_fetch error: " << sglx_getError( hSglx_ ) << "\n";
            std::this_thread::sleep_for( std::chrono::milliseconds( fetchChunkMs_ ) );
            continue; // retry from the same fromCt
        }

        long long tpts = static_cast<long long>( io.data.size() ) / nFetchChans;
        if( tpts <= 0 ) {
            fromCt = headCt;
            continue;
        }

        // Extract the full CAR group + track the SY sync bit per sample.
        carChunk.assign( static_cast<size_t>( tpts ) * nCarChans, 0 );

        for( long long t = 0; t < tpts; ++t ) {

            const short *src = &io.data[static_cast<size_t>( t ) * nFetchChans];
            short       *dst = &carChunk[static_cast<size_t>( t ) * nCarChans];

            for( int c = 0; c < nCarChans; ++c )
                dst[c] = src[c];

            short syValue = src[nFetchChans - 1]; // SY appended last
            int   syBit   = extractBit( syValue, imecSyncBit_ );
            syncTracker.update( syBit, static_cast<long long>( headCt ) + t );
        }

        // Highpass+CAR across the FULL group, then subset to the filter's channels.
        std::vector<double> preprocessed = preprocessor.processChunk( &carChunk[0], static_cast<size_t>( tpts ) );

        compact.assign( static_cast<size_t>( tpts ) * nFilterChans, 0.0 );
        for( long long t = 0; t < tpts; ++t ) {
            const double *srcRow = &preprocessed[static_cast<size_t>( t ) * nCarChans];
            double       *dstRow = &compact[static_cast<size_t>( t ) * nFilterChans];
            for( int c = 0; c < nFilterChans; ++c )
                dstRow[c] = srcRow[filterIndexWithinCarGroup[c]];
        }

        std::vector<PeakEvent> peaks = engine.processChunk(
            &compact[0], static_cast<size_t>( tpts ), static_cast<long long>( headCt ) );

        for( size_t p = 0; p < peaks.size(); ++p ) {

            if( peaks[p].score <= filterBank_.threshold )
                continue;

            if( spikeTimesFile.is_open() ) {
                spikeTimesFile << peaks[p].sampleIndex << "\n";
                spikeTimesFile.flush();
            }

            if( syncTracker.hasEdge() ) {
                SpikeEvent ev;
                ev.sampleIndex  = peaks[p].sampleIndex;
                ev.timeRelSyncS = syncTracker.secondsSinceLastEdge( peaks[p].sampleIndex );
                ev.unitId       = -1; // single-target pipeline -- see Events.h
                spikeQueue_.push( ev );
            }
            // else: no sync edge seen yet this session -- still logged to
            // spikeTimes.txt above, just not usable for windowed counting
            // until the first pulse edge arrives.
        }

        fromCt = headCt + static_cast<t_ull>( tpts );
    }
}
