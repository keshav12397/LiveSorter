// =================================
// ClosedLoopAllUnits: all-Kilosort-units, GPU-accelerated spike detection
// driving the same syllable-triggered digital output ClosedLoop.exe drives.
// See README.md's "All-units detection" section.
//
// Deliberately a SEPARATE executable from ClosedLoop.exe (main.cpp), not a
// mode switch inside it -- the single-target path is the validated
// production pipeline and stays untouched by work here.
//
// It was detection-only for its first phase (MultiFilterBank + one
// ImecFetchThreadCpu, spikes to a CSV and nowhere else). It now runs the
// full loop: ImecFetchThreadCpu + NiFetchThread + DecisionThread, the latter
// two reused as-is from main.cpp, plus an EventPublisher feeding
// SpikeViewer.exe. Still no Python calibration subprocess -- filters come
// from FilterGen/calibrate_all_units.py, run offline ahead of time.
//
// Two things here that main.cpp does not have, both config-driven:
//
//   syllableSource=ni|imecSy  -- where syllable events come from. `ni` is
//   production and is exactly main.cpp's path (NiFetchThread on its own
//   handle, decoding the NI digital word). `imecSy` is a TEST path for a rig
//   whose NI stream cannot be simulated; it decodes the codes out of the
//   IMEC SY word ImecFetchThreadCpu already fetches, and creates no NI
//   handle and no NI thread at all.
//
//   decisionUnitIds -- which units may drive the decision. Counting "any of
//   157 units fired" is not a decision about anything, so with a filter bank
//   this size the driving unit(s) have to be named.
// =================================

#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <chrono>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

#include "SglxCppClient.h"

#include "Config.h"
#include "Events.h"
#include "ThreadSafeQueue.h"
#include "SpikeQueue.h"
#include "AnalysisFeed.h"
#include "AnalysisThread.h"
#include "SglxMetaReader.h"
#include "MultiFilterBank.h"
#include "ImecFetchThreadCpu.h"
#include "NiFetchThread.h"
#include "DecisionThread.h"
#include "EventPublisher.h"

namespace {

std::atomic<bool> g_stopRequested( false );

#ifdef _WIN32
BOOL WINAPI consoleCtrlHandler( DWORD ctrlType )
{
    if( ctrlType == CTRL_C_EVENT || ctrlType == CTRL_CLOSE_EVENT || ctrlType == CTRL_BREAK_EVENT ) {
        g_stopRequested.store( true );
        return TRUE;
    }
    return FALSE;
}
#endif

// Same rationale as main.cpp's exeDir()/resolveRelativeToExe() -- relative
// config paths must resolve against this exe's own directory, not whatever
// directory the user happened to launch it from. Duplicated here rather
// than shared with main.cpp since these two executables are deliberately
// independent (see file header).
std::string exeDir()
{
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA( NULL, buf, MAX_PATH );
    if( n == 0 || n == MAX_PATH )
        return ".";
    std::string path( buf, n );
    size_t slash = path.find_last_of( "\\/" );
    return (slash == std::string::npos) ? "." : path.substr( 0, slash );
#else
    return ".";
#endif
}

std::string resolveRelativeToExe( const std::string &path )
{
    if( path.empty() )
        return path;
    bool isAbsolute =
        (path.size() >= 2 && path[1] == ':') ||
        path[0] == '/' || path[0] == '\\';
    if( isAbsolute )
        return path;
    return exeDir() + "\\" + path;
}

// Contiguous ascending bit list -> (startBit, width), the pair
// extractField()/SyllableDecoder want. NiFetchThread makes the same
// assumption of niSyllableLines; it is checked here rather than assumed
// because the imecSy list is hand-written in a config rather than dictated
// by physical wiring.
void bitsToField( const std::vector<int> &bits, const char *keyName,
                   int &startBit, int &width )
{
    if( bits.empty() )
        throw std::runtime_error( std::string( "Config: '" ) + keyName + "' must list the code bits" );

    startBit = bits[0];
    width    = static_cast<int>( bits.size() );

    for( size_t i = 1; i < bits.size(); ++i ) {
        if( bits[i] != bits[i - 1] + 1 ) {
            throw std::runtime_error(
                std::string( "Config: '" ) + keyName + "' must be contiguous and ascending "
                "(it is read as one multi-bit field starting at the lowest bit)" );
        }
    }
}

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

        std::string filterDir      = resolveRelativeToExe( cfg.requireString( "filterDir" ) );
        int         nChannelsPerUnit = cfg.getInt( "filterNChannels", 5 );
        int         templateLength   = cfg.getInt( "templateLength", 61 );

        std::cout << "Loading multi-unit filter bank from " << filterDir << "...\n";
        MultiFilterBank filterBank = MultiFilterBank::load( filterDir, nChannelsPerUnit, templateLength );
        std::cout << "Loaded " << filterBank.nUnits << " units ("
                   << nChannelsPerUnit << " channels/unit, templateLength=" << templateLength << ")\n";

        std::string carChannelMapJson = resolveRelativeToExe( cfg.requireString( "carChannelMapJson" ) );
        bool        applyHighpass     = cfg.getBool( "applyHighpass", true );
        double      highpassCutoffHz  = cfg.getDouble( "highpassCutoffHz", 300.0 );

        // ---- Syllable source ------------------------------------------------
        // Defaults to the production NI path, so a config that says nothing
        // about this gets the real pipeline. The test path must be asked for
        // by name, and says so loudly when it is.
        std::string syllableSource = cfg.getString( "syllableSource", "ni" );
        if( syllableSource != "ni" && syllableSource != "imecSy" ) {
            throw std::runtime_error( "Config: syllableSource must be 'ni' (production) "
                                       "or 'imecSy' (test path), got '" + syllableSource + "'" );
        }
        const bool useImecSy = ( syllableSource == "imecSy" );

        if( useImecSy ) {
            std::cout << "\n*** syllableSource=imecSy -- TEST PATH ***\n"
                          "    Syllable codes are decoded from the IMEC SY word's bits "
                       << cfg.getString( "imecSyllableBits", "0,1,2" ) << ",\n"
                          "    NOT from the NI digital word. No NI handle and no NiFetchThread\n"
                          "    exist in this run. This is for a rig whose NI stream cannot be\n"
                          "    simulated (see FilterGen/make_sim_session.py); it is NOT the\n"
                          "    production configuration. Set syllableSource=ni for that.\n\n";
        }

        // ---- Connect to live SpikeGLX ---------------------------------------
        // Concurrency rule (README.md): create + connect every handle
        // sequentially, from this thread, before any thread starts. One
        // handle, one owning thread, forever. hNI is only created when there
        // is an NI thread to own it.
        std::string host = cfg.getString( "sglxHost", "localhost" );
        int         port = cfg.getInt( "sglxPort", 4142 );

        void *hIM = sglx_createHandle_std();
        void *hNI = useImecSy ? 0 : sglx_createHandle_std();
        void *hDO = sglx_createHandle_std();

        if( !sglx_connect( hIM, host.c_str(), port ) )
            throw std::runtime_error( std::string( "Could not connect hIM: " ) + sglx_getError( hIM ) );
        if( hNI && !sglx_connect( hNI, host.c_str(), port ) )
            throw std::runtime_error( std::string( "Could not connect hNI: " ) + sglx_getError( hNI ) );
        if( !sglx_connect( hDO, host.c_str(), port ) )
            throw std::runtime_error( std::string( "Could not connect hDO: " ) + sglx_getError( hDO ) );

        std::cout << "Connected " << ( hNI ? 3 : 2 ) << " handles to " << host << ":" << port
                   << " (version " << sglx_getVersion( hIM ) << ")\n";

        // Sample rates for the viewer's session header (js=2/ip=0 is IMEC,
        // js=0/ip=0 is NI, per SglxApi.h). Read here, from main, in the same
        // sequential window the connects above run in -- once a thread owns
        // a handle nobody else may call anything on it.
        double imecRate = sglx_getStreamSampleRate( hIM, 2, 0 );
        double niRate   = hNI ? sglx_getStreamSampleRate( hNI, 0, 0 ) : 0.0;

        // ---- Live viewer feed ------------------------------------------------
        EventPublisher publisher(
            cfg.getInt( "viewerPort", livewire::kDefaultPort ),
            filterBank.hostUnitIds, imecRate, niRate, useImecSy );

        EventPublisher *pub = 0;
        if( cfg.getBool( "viewerEnabled", true ) ) {
            if( publisher.start() )
                pub = &publisher;
            // start() already explained itself on stderr if it failed. A
            // viewer socket that cannot bind must never abort a recording
            // session, so this is deliberately not fatal.
        }

        // ---- Threads ----------------------------------------------------------
        // spikeQueue is the HOT queue (SpikeQueue.h): detections only, target
        // < 10 ms, bounded/preallocated, drop-oldest on overflow (see
        // SpikeQueue.h for the reasoning and bench_queues.cpp/
        // bench_hotpath.cpp for the measurement backing that choice).
        SpikeQueue                     spikeQueue;
        ThreadSafeQueue<SyllableEvent> syllableQueue;

        // analysisFeed is the SLOW queue (AnalysisFeed.h): bulk preprocessed
        // samples for drift estimation and plotting, budget > 500 ms, drops
        // freely and shares no mutex with spikeQueue above. Sized the same
        // way ImecFetchThreadCpu::fetchLoop sizes its own per-chunk buffer
        // (fetchChunkMs of samples at the IMEC rate), so every chunk the
        // fetch thread produces fits in one pool buffer whole.
        int fetchChunkMs = cfg.getInt( "fetchChunkMs", 5 );
        int maxChunkSamplesForFeed = static_cast<int>(
            fetchChunkMs / 1000.0 * imecRate + 0.5 ) + 1; // matches the fetch loop's slack
        int nCarChansForFeed = static_cast<int>( loadChanMapJson( carChannelMapJson ).size() );
        AnalysisFeed analysisFeed( maxChunkSamplesForFeed, nCarChansForFeed,
                                    cfg.getInt( "analysisFeedBuffers", 8 ) );
        AnalysisThread analysisThread( analysisFeed );

        ImecFetchThreadCpu::SyllableFromSy syFromSy;
        if( useImecSy ) {
            std::vector<int> bits = cfg.getIntList( "imecSyllableBits" );
            if( bits.empty() ) {
                // make_sim_session.py's SYLLABLE_BITS.
                bits.push_back( 0 ); bits.push_back( 1 ); bits.push_back( 2 );
            }
            bitsToField( bits, "imecSyllableBits", syFromSy.startBit, syFromSy.width );

            syFromSy.enabled           = true;
            // Same debounce key as the NI path: it is the same state machine
            // holding for the same physical duration, just read off a
            // different word -- but note the two streams' sample rates
            // differ, so the same count is a different wall-clock hold.
            syFromSy.debounceSamples   = cfg.getInt( "niDebounceSamples", 10 );
            syFromSy.queue             = &syllableQueue;
            syFromSy.syllableTimesPath = cfg.getString( "syllableTimesPath", "" );
        }

        ImecFetchThreadCpu imecThread(
            hIM, filterBank,
            carChannelMapJson, applyHighpass, highpassCutoffHz,
            cfg.getInt( "imecSyncBit", 6 ),
            fetchChunkMs,
            cfg.requireString( "spikeTimesPath" ),
            cfg.getString( "latencyLogPath", "" ),
            &spikeQueue, pub, syFromSy, &analysisFeed );

        NiFetchThread niThread(
            hNI,
            cfg.getInt( "niSyncBit", 0 ),
            cfg.getIntList( "niSyllableLines" ),
            cfg.getInt( "niDebounceSamples", 10 ),
            fetchChunkMs,
            syllableQueue,
            useImecSy ? std::string( "" ) : cfg.getString( "syllableTimesPath", "" ),
            pub );

        // With 157 units feeding one queue, "any spike counts" is
        // meaningless -- see DecisionThread's constructor comment. Required
        // rather than defaulted, because a default here would be a silent
        // choice of which neuron controls the animal's stimulus.
        std::vector<int> decisionUnitIds = cfg.getIntList( "decisionUnitIds" );
        if( decisionUnitIds.empty() ) {
            throw std::runtime_error(
                "Config: 'decisionUnitIds' is required by ClosedLoopAllUnits -- list the "
                "Kilosort cluster id(s) whose spikes drive the decision. Every unit in the "
                "bank reports into one queue here, so leaving it unset would silently mean "
                "'trigger when any of the units fires'." );
        }
        std::cout << "Decision driven by unit(s):";
        for( size_t i = 0; i < decisionUnitIds.size(); ++i )
            std::cout << " " << decisionUnitIds[i];
        std::cout << "\n";

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

#ifdef _WIN32
        SetConsoleCtrlHandler( consoleCtrlHandler, TRUE );
#endif

        // 0 (default) = run until Ctrl+C. A finite duration lets this run
        // unattended (e.g. from a script/background task) for a fixed-length
        // live session without needing an external process to deliver Ctrl+C
        // at the right moment -- stop()/join() below are the same graceful
        // shutdown path either way, so the last chunks and the
        // sample-accounting summary are not lost to an abrupt kill.
        int runDurationSec = cfg.getInt( "runDurationSec", 0 );

        std::cout << "Starting all-units detection. ";
        if( runDurationSec > 0 )
            std::cout << "Will stop automatically after " << runDurationSec << "s.\n";
        else
            std::cout << "Press Ctrl+C to stop.\n";

        // Started before imecThread so the SLOW queue's consumer is ready
        // the moment the first chunk is published -- otherwise acquire()
        // would be fine (no consumer needed for that) but the pool would
        // fill and start dropping before anything ever took a chunk.
        analysisThread.start();
        imecThread.start();
        if( hNI )
            niThread.start();
        decisionThread.start();

        auto runStart = std::chrono::steady_clock::now();
        while( !g_stopRequested.load() ) {
            if( runDurationSec > 0 ) {
                double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - runStart ).count();
                if( elapsed >= runDurationSec ) {
                    std::cout << "Reached runDurationSec (" << runDurationSec << "s), stopping...\n";
                    break;
                }
            }
            std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
        }

        std::cout << "Stopping...\n";
        imecThread.stop();
        if( hNI )
            niThread.stop();
        decisionThread.stop();

        imecThread.join();
        if( hNI )
            niThread.join();
        decisionThread.join();

        // Only after imecThread has joined (no more chunks can be
        // published) so analysisThread gets a last chance to drain whatever
        // is still queued before it stops.
        analysisThread.stop();
        analysisThread.join();
        std::cout << analysisThread.summary() << "\n";
        std::cout << "AnalysisFeed in flight at shutdown: " << analysisFeed.nInFlight()
                   << " (should be 0 -- nonzero means a buffer leaked)\n";

        publisher.stop();
        std::cout << publisher.summary();

        // Close/destroy sequentially, same rule as connect, and only after
        // every thread has been joined.
        sglx_close( hIM );
        sglx_destroyHandle( hIM );
        if( hNI ) {
            sglx_close( hNI );
            sglx_destroyHandle( hNI );
        }
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
