// =================================
// ClosedLoop: real-time spike-triggered digital output.
// See README.md for the full architecture description.
// =================================

#include <iostream>
#include <sstream>
#include <vector>
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
#include "RunProcess.h"
#include "FilterBank.h"
#include "ThreadSafeQueue.h"
#include "SpikeQueue.h"
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

// Directory containing ClosedLoop.exe itself (not the process's current
// working directory, which depends on how/where the user launched it from --
// e.g. a plain `ClosedLoop.exe C:\...\config.txt` from an arbitrary PowerShell
// cwd, which is exactly what bit us: a relative calibrationScript path
// resolved against that cwd instead of the repo, and std::system() silently
// couldn't find it). Falls back to "." if GetModuleFileName fails for any
// reason (non-Windows build, or the WinAPI call itself erroring).
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

// Relative paths (config-file entries and defaults alike) are resolved
// against exeDir(), not the process's cwd -- see exeDir()'s comment. Leaves
// absolute paths (drive letter or leading slash) untouched.
std::string resolveRelativeToExe( const std::string &path )
{
    if( path.empty() )
        return path;
    bool isAbsolute =
        (path.size() >= 2 && path[1] == ':') ||   // "C:\..." / "C:/..."
        path[0] == '/' || path[0] == '\\';        // "/..." / "\..."
    if( isAbsolute )
        return path;
    return exeDir() + "\\" + path;
}

// Runs `commandLine` (a fully-quoted argv-style string, e.g. `"exe" "arg
// with spaces" --flag value`) and waits for it to exit, returning its exit
// code (or -1 if it couldn't even be launched).
//
// Deliberately NOT std::system(): on Windows, system() runs the command via
// `cmd.exe /c "<commandLine>"`, wrapping it in one more layer of quoting --
// and when <commandLine> itself starts with a quoted path *and* contains
// further quoted arguments (exactly our case: a quoted python.exe path
// followed by a quoted script path and quoted option values), cmd.exe's
// quote-stripping heuristic can misparse the result and fail with "The
// system cannot find the path specified" despite the exact same string
// running fine when typed directly at a shell. CreateProcess sidesteps
// cmd.exe entirely and uses Windows' own (more predictable) argv-splitting
// rules for lpCommandLine, which is also the standard recommended fix for
// this well-known system() quoting gotcha.

// Shells out to FilterGen/calibrate_for_closedloop.py, which fits the LCMV
// filter on a training split, sweeps the detection threshold against the
// held-out remainder's Kilosort ground truth, and writes
// channels_<id>.bin/filter_<id>.bin/threshold_<id>.bin -- see README's
// "Calibration" section for why this replaced a from-scratch C++ port of
// the same fit+sweep math. Throws std::runtime_error if the
// script exits nonzero, so main()'s catch block aborts startup cleanly.
void runPythonCalibration( const Config &cfg, int targetId, const std::string &filterDir )
{
    std::string      pythonExe    = cfg.getString( "pythonExe", "python" );
    std::string      script       = resolveRelativeToExe( cfg.getString( "calibrationScript",
                                         "FilterGen/calibrate_for_closedloop.py" ) );
    std::vector<int> interferers  = cfg.getIntList( "interferers" );
    int              autoInterferers = cfg.getInt( "autoInterferers", 0 );

    if( interferers.empty() && autoInterferers <= 0 )
        throw std::runtime_error(
            "runPythonCalibration: config needs either 'interferers' "
            "(comma-separated Kilosort cluster ids of nearby units to null "
            "out) or 'autoInterferers' (a count to auto-pick by activity) "
            "when skipCalibration=false" );

    std::ostringstream cmd;
    cmd << "\"" << pythonExe << "\""
        // -u: unbuffered stdout/stderr. Without this, Python fully buffers
        // (not line-buffers) output when it isn't attached to a real
        // console -- exactly our case here, piped through
        // CreateProcess/inherited handles -- so progress prints sit
        // invisible for minutes at a time instead of appearing as they
        // happen. Same class of bug as main()'s own `std::cout <<
        // std::unitbuf` fix, just on the Python side of this subprocess call.
        << " -u"
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

    int rc = runProcessAndWait( cmd.str() );
    if( rc != 0 )
        throw std::runtime_error(
            "'" + script + "' exited with code " + std::to_string( rc ) );
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

        // ---- Phase A: calibration -----------------------------------------
        // FilterGen/calibrate_for_closedloop.py is the ONLY implementation of
        // the fit+sweep math. A from-scratch C++ port of it used to live in
        // Calibration.cpp as an opt-in backend; it was removed because a
        // second implementation of shared math drifts, and this one did --
        // its peer-selection bug capped recall near 50% and cost a long
        // debugging session before the cause was found in the port rather
        // than the algorithm.
        bool skipCalibration = cfg.getBool( "skipCalibration", false );

        // Calibration regenerates channels_/filter_/threshold_<id>.bin from
        // scratch and does not read the existing ones, so filterDir need not
        // already hold a filter for this targetId. Only skipCalibration
        // (which uses the files as-is) requires the pre-run load below.
        FilterBank filterBank;

        if( skipCalibration ) {
            filterBank = FilterBank::load( filterDir, targetId, templateLength );
            std::cout << "Loaded filter for target " << targetId << ": "
                      << filterBank.nChannels() << " channels, "
                      << "initial threshold=" << filterBank.threshold << "\n";
        }

        if( !skipCalibration ) {
            std::cout << "Running calibration (FilterGen/calibrate_for_closedloop.py)...\n";
            runPythonCalibration( cfg, targetId, filterDir );

            // The script just wrote channels_<id>.bin/filter_<id>.bin/
            // threshold_<id>.bin (from scratch, if this is the first run) --
            // load so the in-memory FilterBank reflects the fresh fit.
            filterBank = FilterBank::load( filterDir, targetId, templateLength );

            std::cout << "Calibration done: threshold=" << filterBank.threshold
                       << " (" << filterBank.nChannels() << " channels)\n";
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
        SpikeQueue                     spikeQueue;
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

        std::cout << spikeQueue.summary() << "\n";

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
