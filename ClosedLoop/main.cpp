// =================================
// ClosedLoop: real-time spike-triggered digital output.
// See README.md for the full architecture description.
// =================================

#include <iostream>
#include <sstream>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

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

// Shells out to FilterGen/calibrate_for_closedloop.py, which fits the LCMV
// filter on a training split, sweeps the detection threshold against the
// held-out remainder's Kilosort ground truth, and writes
// channels_<id>.bin/filter_<id>.bin/threshold_<id>.bin -- see README's
// "Calibration via Python" section for why this replaced a from-scratch C++
// port of the same fit+sweep math (Calibration.cpp remains as an opt-in
// fallback via calibrationBackend=cpp). Throws std::runtime_error if the
// script exits nonzero, so main()'s catch block aborts startup cleanly.
void runPythonCalibration( const Config &cfg, int targetId, const std::string &filterDir )
{
    std::string      pythonExe    = cfg.getString( "pythonExe", "python" );
    std::string      script       = cfg.getString( "calibrationScript",
                                         "FilterGen/calibrate_for_closedloop.py" );
    std::vector<int> interferers  = cfg.getIntList( "interferers" );
    int              autoInterferers = cfg.getInt( "autoInterferers", 0 );

    if( interferers.empty() && autoInterferers <= 0 )
        throw std::runtime_error(
            "runPythonCalibration: config needs either 'interferers' "
            "(comma-separated Kilosort cluster ids of nearby units to null "
            "out) or 'autoInterferers' (a count to auto-pick by activity) "
            "when skipCalibration=false and calibrationBackend=python" );

    std::ostringstream cmd;
    cmd << "\"" << pythonExe << "\""
        << " \"" << script << "\""
        << " --ks-dir \""   << cfg.requireString( "trainingKsDir" )  << "\""
        << " --bin-path \"" << cfg.requireString( "trainingBinPath" ) << "\""
        << " --target " << targetId;

    // Explicit list takes priority; otherwise let the script auto-pick the
    // N most active other (non-noise) clusters -- see
    // generate_filter.py's auto_pick_interferers().
    if( !interferers.empty() ) {
        cmd << " --interferers";
        for( size_t i = 0; i < interferers.size(); ++i )
            cmd << " " << interferers[i];
    }
    else {
        cmd << " --auto-interferers " << autoInterferers;
    }

    std::string carJson = cfg.getString( "carChannelMapJson", "" );
    if( !carJson.empty() )
        cmd << " --channel-map-json \"" << carJson << "\"";

    cmd << " --n-channels "      << cfg.getInt( "filterNChannels", 5 )
        << " --template-length " << cfg.getInt( "templateLength", 61 )
        << " --template-offset " << cfg.getInt( "templateOffset", 20 )
        << " --train-frac "      << cfg.getDouble( "calibrationTrainFrac", 0.5 )
        << " --ridge "           << cfg.getDouble( "calibrationRidge", 1e-3 )
        << " --max-spikes "      << cfg.getInt( "calibrationMaxSpikes", 2000 )
        << " --fc "              << cfg.getDouble( "highpassCutoffHz", 300.0 )
        << " --out-dir \""       << filterDir << "\""
        << " --seed "            << cfg.getInt( "calibrationSeed", 0 );

    std::cout << "Running: " << cmd.str() << "\n";

    int rc = std::system( cmd.str().c_str() );
    if( rc != 0 )
        throw std::runtime_error(
            "FilterGen/calibrate_for_closedloop.py exited with code " + std::to_string( rc ) );
}

} // namespace


int main( int argc, char **argv )
{
    // std::cout is fully buffered (not line-buffered) when redirected to a
    // file/pipe rather than a live console -- without this, status output
    // (calibration results, connection status, etc.) can sit invisible in
    // the buffer for a long time even while the app is actively running
    // and fetching. unitbuf flushes after every insertion.
    std::cout << std::unitbuf;

    if( argc < 2 ) {
        std::cerr << "Usage: " << argv[0] << " <config.txt>\n";
        return 1;
    }

    try {
        Config cfg = Config::load( argv[1] );

        std::string filterDir      = cfg.requireString( "filterDir" );
        int         targetId       = cfg.requireInt( "targetId" );
        int         templateLength = cfg.getInt( "templateLength", 61 );

        // CAR must be computed across the SAME channel group the filter was
        // trained with (see README.md / Preprocessor.h) -- pass the same
        // channel-map JSON given to FilterGen's --channel-map-json.
        std::string carChannelMapJson = cfg.getString( "carChannelMapJson", "" );
        bool        applyHighpass     = cfg.getBool( "applyHighpass", true );
        double      highpassCutoffHz  = cfg.getDouble( "highpassCutoffHz", 300.0 );

        FilterBank filterBank = FilterBank::load( filterDir, targetId, templateLength );
        std::cout << "Loaded filter for target " << targetId << ": "
                  << filterBank.nChannels() << " channels, "
                  << "initial threshold=" << filterBank.threshold << "\n";

        // ---- Phase A: calibration -----------------------------------------
        // Default backend shells out to Python (see runPythonCalibration()
        // above / README's "Calibration via Python" section); set
        // calibrationBackend=cpp to use Calibration.cpp's from-scratch C++
        // port instead (kept as a fallback, not the default -- it's a
        // second implementation of the same fit+sweep math that can drift
        // out of sync with Python, which is exactly what caused this
        // session's earlier ~50%-recall-cap bug hunt).
        std::string calibrationBackend = cfg.getString( "calibrationBackend", "python" );

        if( !cfg.getBool( "skipCalibration", false ) ) {

            if( calibrationBackend == "python" ) {

                std::cout << "Running Python calibration (FilterGen/calibrate_for_closedloop.py)...\n";
                runPythonCalibration( cfg, targetId, filterDir );

                // Python just overwrote channels_<id>.bin/filter_<id>.bin/
                // threshold_<id>.bin on disk -- reload so the in-memory
                // FilterBank reflects the freshly fitted filter, not the
                // (possibly stale) one loaded above.
                filterBank = FilterBank::load( filterDir, targetId, templateLength );

                std::cout << "Calibration done: threshold=" << filterBank.threshold
                          << " (" << filterBank.nChannels() << " channels)\n";
            }
            else {

                std::cout << "Running C++ calibration against training data...\n";

                Calibration::Result calib = Calibration::run(
                    cfg.requireString( "trainingBinPath" ),
                    cfg.requireString( "trainingKsDir" ),
                    targetId,
                    filterBank,
                    carChannelMapJson,
                    applyHighpass,
                    highpassCutoffHz,
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
            carChannelMapJson, applyHighpass, highpassCutoffHz,
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
            syllableQueue,
            cfg.getString( "syllableTimesPath", "" ) );

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
