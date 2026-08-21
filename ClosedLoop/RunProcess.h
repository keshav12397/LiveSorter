#ifndef CLOSEDLOOP_RUNPROCESS_H
#define CLOSEDLOOP_RUNPROCESS_H

#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdlib>
#endif

// Run a command line, wait for it, return its exit code (-1 if it could not
// be launched at all).
//
// Deliberately NOT std::system() on Windows. system() runs the command via
// `cmd.exe /c "<commandLine>"`, wrapping it in one more layer of quoting --
// and when <commandLine> itself starts with a quoted path AND contains
// further quoted arguments (exactly the calibration case: a quoted
// python.exe followed by a quoted script path and quoted option values),
// cmd.exe's quote-stripping heuristic can misparse it and fail with "The
// system cannot find the path specified", despite the identical string
// working when typed at a shell. CreateProcess sidesteps cmd.exe and uses
// Windows' own, more predictable, argv-splitting for lpCommandLine.
//
// This lives in a header shared by both calibration callers (main.cpp's
// single-target path and mainAllUnits.cpp's) rather than being copied into
// each. The gotcha above cost real debugging time once; a second
// hand-written copy is how it comes back.
inline int runProcessAndWait( const std::string &commandLine )
{
#ifdef _WIN32
    STARTUPINFOA si;
    ZeroMemory( &si, sizeof(si) );
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi;
    ZeroMemory( &pi, sizeof(pi) );

    // CreateProcessA requires a writable buffer for lpCommandLine; it may
    // modify the buffer in place.
    std::vector<char> buf( commandLine.begin(), commandLine.end() );
    buf.push_back( '\0' );

    BOOL ok = CreateProcessA(
        NULL, &buf[0], NULL, NULL, /*bInheritHandles=*/TRUE,
        0, NULL, NULL, &si, &pi );
    if( !ok )
        return -1;

    WaitForSingleObject( pi.hProcess, INFINITE );
    DWORD exitCode = 1;
    GetExitCodeProcess( pi.hProcess, &exitCode );
    CloseHandle( pi.hProcess );
    CloseHandle( pi.hThread );
    return static_cast<int>( exitCode );
#else
    return std::system( commandLine.c_str() );
#endif
}

#endif // CLOSEDLOOP_RUNPROCESS_H
