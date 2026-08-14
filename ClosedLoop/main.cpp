// =================================
// ClosedLoop: real-time spike-triggered digital output.
// See README.md for the full architecture description.
// =================================

#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

#include "SglxCppClient.h"

#include "Config.h"
#include "FilterBank.h"
#include "Calibration.h"
#include "ThreadSafeQueue.h"
#include "Events.h"
#include "ImecFetchThread.h"
#include "NiFetchThread.h"
#include "DecisionThread.h"

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

} // namespace


int main( int argc, char **argv )
{
    if( argc < 2 ) {
        std::cerr << "Usage: " << argv[0] << " <config.txt>\n";
        return 1;
    }

    try {
        Config cfg = Config::load( argv[1] );

        std::string filterDir      = cfg.requireString( "filterDir" );
        int         targetId       = cfg.requireInt( "targetId" );
        int         templateLength = cfg.getInt( "templateLength", 61 );

        FilterBank filterBank = FilterBank::load( filterDir, targetId, templateLength );
        std::cout << "Loaded filter for target " << targetId << ": "
                  << filterBank.nChannels() << " channels, "
                  << "initial threshold=" << filterBank.threshold << "\n";

        // ---- Phase A: calibration -----------------------------------------
        if( !cfg.getBool( "skipCalibration", false ) ) {

            std::cout << "Running calibration against training data...\n";

            Calibration::Result calib = Calibration::run(
                cfg.requireString( "trainingBinPath" ),
                cfg.requireString( "trainingKsDir" ),
                targetId,
                filterBank,
                cfg.getInt( "fetchChunkMs", 5 ),
                cfg.getString( "calibrationCriterion", "best_f1" ),
                cfg.getDouble( "maxFalsePositiveRateHz", 1.0 ),
                cfg.getString( "calibrationLogPath", "" ) );

            filterBank.threshold = calib.bestThreshold;

            std::cout << "Calibration done: threshold=" << calib.bestThreshold
                      << "  recall="    << calib.bestPoint.recall
                      << "  precision=" << calib.bestPoint.precision
                      << "  f1="        << calib.bestPoint.f1
                      << "  fpRateHz="  << calib.bestPoint.fpRateHz << "\n";
        }
        else {
            std::cout << "skipCalibration=true -- using threshold from threshold_"
                      << targetId << ".bin as-is: " << filterBank.threshold << "\n";
        }

        // ---- Connect to live SpikeGLX -----------------------------------------
        // Concurrency rule (README.md): create + connect every handle
        // sequentially, from this thread, before any fetch/decision thread
        // starts. Never share a handle across threads afterward.
        std::string host = cfg.getString( "sglxHost", "localhost" );
        int         port = cfg.getInt( "sglxPort", 4142 );

        void *hIM = sglx_createHandle_std();
        void *hNI = sglx_createHandle_std();
        void *hDO = sglx_createHandle_std();

        if( !sglx_connect( hIM, host.c_str(), port ) )
            throw std::runtime_error( std::string( "Could not connect hIM: " ) + sglx_getError( hIM ) );
        if( !sglx_connect( hNI, host.c_str(), port ) )
            throw std::runtime_error( std::string( "Could not connect hNI: " ) + sglx_getError( hNI ) );
        if( !sglx_connect( hDO, host.c_str(), port ) )
            throw std::runtime_error( std::string( "Could not connect hDO: " ) + sglx_getError( hDO ) );

        std::cout << "Connected 3 handles to " << host << ":" << port
                  << " (version " << sglx_getVersion( hIM ) << ")\n";

        // ---- Spawn threads -------------------------------------------------------
        ThreadSafeQueue<SpikeEvent>    spikeQueue;
        ThreadSafeQueue<SyllableEvent> syllableQueue;

        ImecFetchThread imecThread(
            hIM, filterBank,
            cfg.getInt( "imecSyncBit", 6 ),
            cfg.getInt( "fetchChunkMs", 5 ),
            spikeQueue,
            cfg.requireString( "spikeTimesPath" ) );

        NiFetchThread niThread(
            hNI,
            cfg.getInt( "niSyncBit", 0 ),
            cfg.getIntList( "niSyllableLines" ),
            cfg.getInt( "niDebounceSamples", 10 ),
            cfg.getInt( "fetchChunkMs", 5 ),
            syllableQueue );

        DecisionThread decisionThread(
            hDO, spikeQueue, syllableQueue,
            cfg.getDouble( "windowStartMs", 0.0 ) / 1000.0,
            cfg.getDouble( "windowEndMs", 100.0 ) / 1000.0,
            cfg.getInt( "spikeCountThreshold", 3 ),
            cfg.requireString( "doLine" ),
            cfg.getInt( "doPulseMs", 50 ),
            cfg.getString( "decisionLogPath", "" ) );

#ifdef _WIN32
        SetConsoleCtrlHandler( consoleCtrlHandler, TRUE );
#endif

        std::cout << "Starting fetch/decision threads. Press Ctrl+C to stop.\n";
        imecThread.start();
        niThread.start();
        decisionThread.start();

        while( !g_stopRequested.load() )
            std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );

        std::cout << "Stopping...\n";
        imecThread.stop();
        niThread.stop();
        decisionThread.stop();

        imecThread.join();
        niThread.join();
        decisionThread.join();

        // Close/destroy sequentially, same rule as connect.
        sglx_close( hIM );
        sglx_destroyHandle( hIM );
        sglx_close( hNI );
        sglx_destroyHandle( hNI );
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
