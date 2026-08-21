// =================================
// OfflineReplay: runs the all-units loop against a .bin on disk instead of
// against a live SpikeGLX stream, and produces the same four outputs a live
// run does (spikeTimes / syllableTimes / decisions CSV+log) plus the same
// live viewer socket.
//
// Why this exists
// ---------------
// The rig this project runs on cannot always be driven from here. SpikeGLX
// refuses sglx_startRun unless a human has validated the run parameters in
// its Configure dialog, and its NI stream is a Fake_40kHz source whose
// digital word is all zeros, so syllable codes cannot reach it at all. That
// leaves the syllable -> decision -> trial path -- and therefore the viewer's
// trigger-aligned rasters and PSTHs, which are the whole point of the viewer
// -- with no way to be exercised against real, coherent data.
//
// FilterGen/make_sim_session.py already writes a session whose spikes AND
// syllable codes are both exactly known and both carried in the same file on
// the same clock. This driver reads that file. What comes out is a complete,
// self-consistent run of the real pipeline: real Preprocessor, real
// MultiConvolutionEngine (via OfflineScorer, which is those two unmodified),
// the real SyllableDecoder, and the real DecisionThread.
//
// What it is NOT
// --------------
// It is not the live path and must never be mistaken for a test of it. There
// is no sglx_fetch, no ring buffer, no sample accounting, and no wall-clock
// pressure on a fetch loop -- all four of those are exactly what a live run
// tests and this cannot. Detection here is a single batched pass over the
// whole file, not a chunked real-time one.
//
// Pacing. DecisionThread's pruneExpired() closes a trial on WALL-CLOCK time,
// not on stream time, so the event feed below is paced at real time by
// default. replaySpeed>1 makes the run finish sooner and makes the decision
// windows wrong in proportion; it is there for regenerating a viewer test
// file quickly, not for producing decisions to believe.
//
// DecisionThread still needs its own SpikeGLX handle for sglx_ni_DO_set. If
// the server is reachable it connects normally and the DO calls simply fail
// on a rig with no such line, which is logged and harmless -- the trial
// bookkeeping, which is what this driver is for, is unaffected.
// =================================

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>


#include "SglxCppClient.h"

#include "Config.h"
#include "DigitalWordUtils.h"
#include "Events.h"
#include "EventPublisher.h"
#include "MultiFilterBank.h"
#include "OfflineScorer.h"
#include "SglxMetaReader.h"
#include "SyllableDecoder.h"
#include "SyncEdgeTracker.h"
#include "ThreadSafeQueue.h"
#include "SpikeQueue.h"
#include "DecisionThread.h"

namespace {

// One thing to feed into the loop, on the IMEC clock. Spikes and syllables
// are merged into one time-ordered list before feeding, because
// DecisionThread's incremental counting assumes both arrive roughly in
// stream order -- feeding all the spikes and then all the syllables would
// leave every window empty.
struct FeedEvent {
    long long sampleIndex;
    double    timeRelSyncS;
    int       unitId;   // -1 for a syllable
    int       code;     // 0 for a spike
    float     score;
};

} // namespace


int main( int argc, char **argv )
{
    std::cout << std::unitbuf;

    if( argc < 2 ) {
        std::cerr << "Usage: " << argv[0] << " <config.txt>\n";
        return 1;
    }

    try {
        Config cfg = Config::load( argv[1] );

        std::string binPath  = cfg.requireString( "replayBinPath" );
        std::string metaPath = cfg.getString( "replayMetaPath", "" );
        if( metaPath.empty() ) {
            // SpikeGLX's own convention: the .meta sits beside the .bin with
            // the same stem.
            size_t dot = binPath.find_last_of( '.' );
            metaPath = ( dot == std::string::npos ) ? binPath + ".meta"
                                                     : binPath.substr( 0, dot ) + ".meta";
        }

        SglxMeta meta = SglxMeta::load( metaPath );
        const double sampleRate = meta.sampleRateHz;
        const int    nChansInFile = meta.nSavedChans;
        const int    syChanInFile = nChansInFile - meta.nSyncChans;

        std::cout << "Replaying " << binPath << "\n"
                   << "  " << nChansInFile << " channels @ " << sampleRate << " Hz, SY at index "
                   << syChanInFile << "\n";

        std::vector<int> carChannelIds = loadChanMapJson( cfg.requireString( "carChannelMapJson" ) );
        if( carChannelIds.empty() )
            throw std::runtime_error( "carChannelMapJson produced an empty channel group" );

        int nChannelsPerUnit = cfg.getInt( "filterNChannels", 5 );
        int templateLength   = cfg.getInt( "templateLength", 61 );

        // ---- filter bank ------------------------------------------------------
        // Loaded through MultiFilterBank so the packed-file reader is not
        // written a second time. An implementation that copied all three
        // arrays back off the device; they are already host vectors here.
        std::vector<int32_t> rawChannels;
        std::vector<float>   filters, thresholds;
        std::vector<int>     unitIds;
        int                  nUnits = 0;
        {
            MultiFilterBank bank = MultiFilterBank::load( cfg.requireString( "filterDir" ),
                                                          nChannelsPerUnit, templateLength );
            nUnits      = bank.nUnits;
            unitIds     = bank.hostUnitIds;
            rawChannels = bank.channels;
            filters     = bank.filters;
            thresholds  = bank.thresholds;
        }
        std::cout << "  " << nUnits << " units\n";

        // Raw SpikeGLX channel id -> position within the CAR group. Same
        // translation ImecFetchThreadCpu::fetchLoop() does, and required by
        // scoreAllUnitsOffline's stated convention.
        std::vector<int32_t> chansInCarGroup( rawChannels.size() );
        for( size_t i = 0; i < rawChannels.size(); ++i ) {
            std::vector<int>::const_iterator it =
                std::find( carChannelIds.begin(), carChannelIds.end(), rawChannels[i] );
            if( it == carChannelIds.end() ) {
                throw std::runtime_error( "unit at index " + std::to_string( i / nChannelsPerUnit )
                    + " uses channel " + std::to_string( rawChannels[i] )
                    + " which is not in the CAR channel group" );
            }
            chansInCarGroup[i] = (int32_t)( it - carChannelIds.begin() );
        }

        // ---- how much of the file to replay -----------------------------------
        long long fileSamples = 0;
        {
            std::ifstream fh( binPath.c_str(), std::ios::binary | std::ios::ate );
            if( !fh.is_open() )
                throw std::runtime_error( "could not open " + binPath );
            fileSamples = (long long)( fh.tellg() / ( sizeof(short) * nChansInFile ) );
        }
        long long nSamples = fileSamples;
        double    limitS   = cfg.getDouble( "replayDurationSec", 0.0 );
        if( limitS > 0 )
            nSamples = std::min( nSamples, (long long)( limitS * sampleRate ) );

        std::cout << "  replaying " << nSamples << " samples ("
                   << ( nSamples / sampleRate ) << " s of " << ( fileSamples / sampleRate ) << " s)\n";

        // ---- SY channel: sync edges and syllable codes -------------------------
        // Read first, in one pass, because the spike timestamps below need the
        // sync-edge positions and there is no reason to touch the file twice.
        std::vector<long long> syncEdges;
        std::vector<FeedEvent> syllableEvents;
        {
            SyncEdgeTracker syncTracker( sampleRate );

            std::vector<int> codeBits = cfg.getIntList( "imecSyllableBits" );
            if( codeBits.empty() ) {
                codeBits.push_back( 0 ); codeBits.push_back( 1 ); codeBits.push_back( 2 );
            }
            SyllableDecoder decoder( codeBits[0], (int)codeBits.size(),
                                      cfg.getInt( "niDebounceSamples", 10 ) );

            const int syncBit = cfg.getInt( "imecSyncBit", 6 );

            std::FILE *fh = std::fopen( binPath.c_str(), "rb" );
            if( !fh )
                throw std::runtime_error( "could not open " + binPath );

            const long long chunk = 65536;
            std::vector<short> buf( (size_t)chunk * nChansInFile );
            long long done = 0;
            int lastBit = 0;
            bool haveBit = false;

            while( done < nSamples ) {
                long long want = std::min( chunk, nSamples - done );
                size_t got = std::fread( &buf[0], sizeof(short), (size_t)( want * nChansInFile ), fh );
                long long n = (long long)( got / nChansInFile );
                if( n <= 0 )
                    break;

                for( long long t = 0; t < n; ++t ) {
                    short     sy  = buf[(size_t)t * nChansInFile + syChanInFile];
                    long long idx = done + t;

                    int bit = extractBit( sy, syncBit );
                    if( haveBit && lastBit == 0 && bit == 1 )
                        syncEdges.push_back( idx );
                    lastBit = bit;
                    haveBit = true;
                    syncTracker.update( bit, idx );

                    int       code     = 0;
                    long long onsetIdx = 0;
                    if( decoder.update( sy, idx, code, onsetIdx ) && syncTracker.hasEdge() ) {
                        FeedEvent e;
                        e.sampleIndex  = onsetIdx;
                        e.timeRelSyncS = syncTracker.secondsSinceLastEdge( onsetIdx );
                        e.unitId       = -1;
                        e.code         = code;
                        e.score        = 0.0f;
                        syllableEvents.push_back( e );
                    }
                }
                done += n;
            }
            std::fclose( fh );
        }
        std::cout << "  " << syncEdges.size() << " sync edges, "
                   << syllableEvents.size() << " syllable onsets decoded from the SY word\n";

        // ---- detection ----------------------------------------------------------
        const int       scoreChunkSamples = cfg.getInt( "scoreChunkSamples", 2000 );
        const long long minSep            = templateLength / 2;
        const int detectionCapacity = std::max( 20000,
            nUnits * ( scoreChunkSamples / (int)std::max( 1LL, minSep ) + 1 ) * 2 );

        std::cout << "Scoring all units in one pass...\n";
        auto tScore0 = std::chrono::steady_clock::now();
        std::vector<UnitPeaks> peaks = scoreAllUnitsOffline(
            binPath, metaPath, carChannelIds,
            cfg.getBool( "applyHighpass", true ), cfg.getDouble( "highpassCutoffHz", 300.0 ),
            unitIds, chansInCarGroup, filters, templateLength, nChannelsPerUnit,
            0, nSamples, scoreChunkSamples, minSep, detectionCapacity );
        std::cout << "  scored in "
                   << std::chrono::duration<double>( std::chrono::steady_clock::now() - tScore0 ).count()
                   << " s\n";

        // scoreAllUnitsOffline reports EVERY NMS-accepted peak, threshold-free,
        // so that a threshold sweep is possible against it. The live pipeline
        // applies the per-unit threshold in the kernel; applying it here is
        // the same comparison, just on the host.
        std::vector<FeedEvent> feed;
        for( int u = 0; u < nUnits; ++u ) {
            for( size_t i = 0; i < peaks[u].sampleIdx.size(); ++i ) {
                if( peaks[u].score[i] < thresholds[u] )
                    continue;
                FeedEvent e;
                e.sampleIndex = peaks[u].sampleIdx[i];
                e.unitId      = unitIds[u];
                e.code        = 0;
                e.score       = (float)peaks[u].score[i];
                e.timeRelSyncS = 0.0;
                feed.push_back( e );
            }
        }
        std::cout << "  " << feed.size() << " detections above threshold\n";

        // Seconds since the most recent sync edge, computed the same way
        // SyncEdgeTracker computes it live -- from the nearest preceding edge.
        for( size_t i = 0; i < feed.size(); ++i ) {
            std::vector<long long>::const_iterator it =
                std::upper_bound( syncEdges.begin(), syncEdges.end(), feed[i].sampleIndex );
            if( it == syncEdges.begin() )
                continue;   // before the first edge: leave 0, same as the live loop does
            feed[i].timeRelSyncS = (double)( feed[i].sampleIndex - *( it - 1 ) ) / sampleRate;
        }

        feed.insert( feed.end(), syllableEvents.begin(), syllableEvents.end() );
        std::stable_sort( feed.begin(), feed.end(),
            []( const FeedEvent &a, const FeedEvent &b ) { return a.sampleIndex < b.sampleIndex; } );

        // ---- outputs ------------------------------------------------------------
        std::ofstream spikeCsv( cfg.requireString( "spikeTimesPath" ).c_str(), std::ios::trunc );
        spikeCsv << "unit_id,sample_index,score\n";

        std::ofstream syllableCsv;
        if( !cfg.getString( "syllableTimesPath", "" ).empty() )
            syllableCsv.open( cfg.getString( "syllableTimesPath", "" ).c_str(), std::ios::trunc );

        // ---- the live half: publisher + DecisionThread ---------------------------
        EventPublisher publisher( cfg.getInt( "viewerPort", livewire::kDefaultPort ),
                                   unitIds, sampleRate, 0.0, /*imecSy=*/true );
        EventPublisher *pub = 0;
        if( cfg.getBool( "viewerEnabled", true ) && publisher.start() )
            pub = &publisher;

        std::string host = cfg.getString( "sglxHost", "localhost" );
        int         port = cfg.getInt( "sglxPort", 4142 );

        void *hDO = sglx_createHandle_std();
        if( !sglx_connect( hDO, host.c_str(), port ) ) {
            // Not fatal here, unlike in the live binaries: this driver's job
            // is the trial bookkeeping and the viewer feed, and both work
            // without a reachable DO line. Said out loud so a run whose
            // digital output silently did nothing is not mistaken for one
            // that worked.
            std::cout << "NOTE: no SpikeGLX at " << host << ":" << port
                       << " -- the decision line will not be driven. Trial counting,\n"
                          "      the CSVs, and the viewer feed are unaffected.\n";
        }

        std::vector<int> decisionUnitIds = cfg.getIntList( "decisionUnitIds" );
        if( decisionUnitIds.empty() )
            throw std::runtime_error( "Config: 'decisionUnitIds' is required (see mainAllUnits.cpp)" );

        SpikeQueue                     spikeQueue;
        ThreadSafeQueue<SyllableEvent> syllableQueue;

        DecisionThread decisionThread(
            hDO, spikeQueue, syllableQueue,
            cfg.getDouble( "windowStartMs", 0.0 ) / 1000.0,
            cfg.getDouble( "windowEndMs", 100.0 ) / 1000.0,
            cfg.getInt( "spikeCountThreshold", 3 ),
            cfg.requireString( "doLine" ),
            cfg.getInt( "doPulseMs", 50 ),
            cfg.getString( "decisionLogPath", "" ),
            decisionUnitIds,
            cfg.getString( "decisionCsvPath", "" ),
            pub );
        decisionThread.start();

        // ---- feed ----------------------------------------------------------------
        const double speed = std::max( 0.001, cfg.getDouble( "replaySpeed", 1.0 ) );
        std::cout << "Feeding " << feed.size() << " events at " << speed << "x real time";
        if( speed != 1.0 ) {
            std::cout << "\n  (decision windows are wall-clock, so anything other than 1.0x\n"
                          "   distorts them -- see this file's header)";
        }
        std::cout << "\n";

        const long long firstSample = feed.empty() ? 0 : feed[0].sampleIndex;
        auto wallStart = std::chrono::steady_clock::now();

        std::vector<livewire::WireRecord> wireBatch;
        wireBatch.reserve( 256 );

        for( size_t i = 0; i < feed.size(); ++i ) {

            // Pace to the event's own stream time. Sleeping per event would
            // be thousands of tiny sleeps a second; sleeping only when we are
            // actually ahead keeps the loop cheap and the pacing honest.
            double streamS = (double)( feed[i].sampleIndex - firstSample ) / sampleRate / speed;
            double wallS   = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - wallStart ).count();
            if( streamS - wallS > 0.001 ) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds( (long long)( ( streamS - wallS ) * 1e6 ) ) );
            }

            if( feed[i].unitId >= 0 ) {
                SpikeEvent ev;
                ev.timeRelSyncS = feed[i].timeRelSyncS;
                ev.sampleIndex  = feed[i].sampleIndex;
                ev.unitId       = feed[i].unitId;
                spikeQueue.push( ev );

                spikeCsv << ev.unitId << "," << ev.sampleIndex << "," << feed[i].score << "\n";

                if( pub ) {
                    wireBatch.push_back( livewire::makeRecord(
                        livewire::kWireSpike, livewire::kStreamImec,
                        ev.unitId, ev.sampleIndex, feed[i].score, (float)ev.timeRelSyncS ) );
                }
            }
            else {
                SyllableEvent ev;
                ev.code         = feed[i].code;
                ev.sampleIndex  = feed[i].sampleIndex;
                ev.timeRelSyncS = feed[i].timeRelSyncS;
                syllableQueue.push( ev );

                if( syllableCsv.is_open() )
                    syllableCsv << ev.code << "," << ev.sampleIndex << "\n";

                if( pub ) {
                    wireBatch.push_back( livewire::makeRecord(
                        livewire::kWireSyllable, livewire::kStreamImec,
                        ev.code, ev.sampleIndex, 0.0f, (float)ev.timeRelSyncS ) );
                }
            }

            if( pub && wireBatch.size() >= 64 ) {
                publisher.publish( &wireBatch[0], wireBatch.size() );
                wireBatch.clear();
            }
        }
        if( pub && !wireBatch.empty() )
            publisher.publish( &wireBatch[0], wireBatch.size() );

        // Let the last trials' windows close before tearing DecisionThread
        // down, or the final few would never get a decision row.
        std::this_thread::sleep_for( std::chrono::milliseconds(
            (long long)( cfg.getDouble( "windowEndMs", 100.0 ) ) + 500 ) );

        decisionThread.stop();
        decisionThread.join();

        spikeCsv.flush();
        if( syllableCsv.is_open() )
            syllableCsv.flush();

        publisher.stop();
        std::cout << publisher.summary();

        sglx_close( hDO );
        sglx_destroyHandle( hDO );

        std::cout << "Done.\n";
        return 0;
    }
    catch( const std::exception &e ) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
