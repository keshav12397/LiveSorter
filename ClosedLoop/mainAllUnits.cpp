// =================================
// ClosedLoopAllUnits: detection-only, all-Kilosort-units, GPU-accelerated
// spike detection. See README.md's "All-units GPU detection" section.
//
// Deliberately a SEPARATE executable from ClosedLoop.exe (main.cpp), not a
// mode switch inside it -- this keeps the validated single-target
// production pipeline completely untouched while this one is built out.
// This phase is detection-only: no DecisionThread / digital-output wiring,
// no Python calibration subprocess call (filters come from
// FilterGen/calibrate_all_units.py, run offline ahead of time) -- just
// GpuFilterBank + one ImecFetchThreadGPU, running until Ctrl+C.
// =================================

#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

#include "SglxCppClient.h"

#include "Config.h"
#include "GpuFilterBank.h"
#include "ImecFetchThreadGPU.h"

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
        GpuFilterBank filterBank = GpuFilterBank::load( filterDir, nChannelsPerUnit, templateLength );
        std::cout << "Loaded " << filterBank.nUnits << " units ("
                   << nChannelsPerUnit << " channels/unit, templateLength=" << templateLength << ")\n";

        std::string carChannelMapJson = resolveRelativeToExe( cfg.requireString( "carChannelMapJson" ) );
        bool        applyHighpass     = cfg.getBool( "applyHighpass", true );
        double      highpassCutoffHz  = cfg.getDouble( "highpassCutoffHz", 300.0 );

        std::string host = cfg.getString( "sglxHost", "localhost" );
        int         port = cfg.getInt( "sglxPort", 4142 );

        void *hIM = sglx_createHandle_std();
        if( !sglx_connect( hIM, host.c_str(), port ) )
            throw std::runtime_error( std::string( "Could not connect hIM: " ) + sglx_getError( hIM ) );

        std::cout << "Connected to " << host << ":" << port
                   << " (version " << sglx_getVersion( hIM ) << ")\n";

        ImecFetchThreadGPU imecThread(
            hIM, filterBank,
            carChannelMapJson, applyHighpass, highpassCutoffHz,
            cfg.getInt( "imecSyncBit", 6 ),
            cfg.getInt( "fetchChunkMs", 5 ),
            cfg.requireString( "spikeTimesPath" ),
            cfg.getString( "latencyLogPath", "" ) );

#ifdef _WIN32
        SetConsoleCtrlHandler( consoleCtrlHandler, TRUE );
#endif

        std::cout << "Starting all-units GPU detection. Press Ctrl+C to stop.\n";
        imecThread.start();

        while( !g_stopRequested.load() )
            std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );

        std::cout << "Stopping...\n";
        imecThread.stop();
        imecThread.join();

        sglx_close( hIM );
        sglx_destroyHandle( hIM );

        std::cout << "Done.\n";
        return 0;
    }
    catch( const std::exception &e ) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
