#include "NiFetchThread.h"

#include <iostream>
#include <chrono>
#include <algorithm>

#include "SglxCppClient.h"
#include "DigitalWordUtils.h"

namespace {
    const int kNiJs = 0;  // NI stream selector, per SglxApi.h
    const int kNiIp = 0;
}


NiFetchThread::NiFetchThread( void *hSglx, int niSyncBit, const std::vector<int> &syllableLines,
                               int debounceSamples, int fetchChunkMs,
                               ThreadSafeQueue<SyllableEvent> &syllableQueue )
    :   hSglx_( hSglx ), syncBit_( niSyncBit ), syllableLines_( syllableLines ),
        debounceSamples_( debounceSamples ), fetchChunkMs_( fetchChunkMs ),
        syllableQueue_( syllableQueue ), stopFlag_( false )
{}


void NiFetchThread::start() { thread_ = std::thread( &NiFetchThread::fetchLoop, this ); }
void NiFetchThread::stop()  { stopFlag_.store( true ); }
void NiFetchThread::join()  { if( thread_.joinable() ) thread_.join(); }


void NiFetchThread::fetchLoop()
{
    double sampleRate = sglx_getStreamSampleRate( hSglx_, kNiJs, kNiIp );
    if( sampleRate <= 0 ) {
        std::cerr << "NiFetchThread: sglx_getStreamSampleRate failed: "
                  << sglx_getError( hSglx_ ) << "\n";
        return;
    }

    // Resolve the DW (digital word) channel's index dynamically.
    // {MN,MA,XA,DW} ordering per SglxApi.h; DW starts right after MN+MA+XA.
    cppClient_sglx_get_ints acqChans;
    if( !sglx_getStreamAcqChans( acqChans, hSglx_, kNiJs, kNiIp ) || acqChans.vint.size() < 4 ) {
        std::cerr << "NiFetchThread: sglx_getStreamAcqChans failed: "
                  << sglx_getError( hSglx_ ) << "\n";
        return;
    }
    int dwChannelIndex = acqChans.vint[0] + acqChans.vint[1] + acqChans.vint[2]; // nMN+nMA+nXA

    std::vector<int> channelSubset( 1, dwChannelIndex );

    // syllableLines_ is assumed contiguous ascending (e.g. {5,6,7}) --
    // extractField() reads them as one multi-bit field starting at the
    // lowest line number.
    int fieldStartBit = *std::min_element( syllableLines_.begin(), syllableLines_.end() );
    int fieldWidth     = static_cast<int>( syllableLines_.size() );

    const long long chunkSamples = static_cast<long long>( fetchChunkMs_ / 1000.0 * sampleRate + 0.5 );

    SyncEdgeTracker syncTracker( sampleRate );

    // Debounce state.
    int       candidateCode      = 0;
    long long candidateCount     = 0;
    long long candidateStartIdx  = 0;
    int       lastEmittedCode    = 0;

    t_ull fromCt = sglx_getStreamSampleCount( hSglx_, kNiJs, kNiIp );

    while( !stopFlag_.load() ) {

        cppClient_sglx_fetch io;
        io.channel_subset = &channelSubset[0];
        io.n_cs           = 1;
        io.max_samps      = static_cast<int>( chunkSamples );
        io.downsample     = 1;
        io.js             = kNiJs;
        io.ip             = kNiIp;

        t_ull headCt = sglx_fetch( io, hSglx_, fromCt );

        if( headCt == 0 ) {
            std::cerr << "NiFetchThread: sglx_fetch error: " << sglx_getError( hSglx_ ) << "\n";
            std::this_thread::sleep_for( std::chrono::milliseconds( fetchChunkMs_ ) );
            continue;
        }

        long long tpts = static_cast<long long>( io.data.size() );  // n_cs == 1
        if( tpts <= 0 ) {
            fromCt = headCt;
            continue;
        }

        for( long long t = 0; t < tpts; ++t ) {

            long long idx = static_cast<long long>( headCt ) + t;
            short     w   = io.data[static_cast<size_t>( t )];

            syncTracker.update( extractBit( w, syncBit_ ), idx );

            int code = extractField( w, fieldStartBit, fieldWidth );

            if( code == candidateCode ) {
                ++candidateCount;
            }
            else {
                candidateCode     = code;
                candidateCount    = 1;
                candidateStartIdx = idx;
            }

            if( candidateCount == debounceSamples_ && candidateCode != lastEmittedCode ) {

                if( candidateCode != 0 && syncTracker.hasEdge() ) {
                    SyllableEvent ev;
                    ev.code         = candidateCode;
                    ev.sampleIndex  = candidateStartIdx;
                    ev.timeRelSyncS = syncTracker.secondsSinceLastEdge( candidateStartIdx );
                    syllableQueue_.push( ev );
                }
                // Re-arm regardless of whether we emitted (code==0 "rest"
                // stabilizing also updates lastEmittedCode, so a repeated
                // nonzero code after a rest period is treated as new).
                lastEmittedCode = candidateCode;
            }
        }

        fromCt = headCt + static_cast<t_ull>( tpts );
    }
}
