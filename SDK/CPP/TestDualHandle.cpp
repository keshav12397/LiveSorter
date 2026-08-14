// =================================
// Test: two simultaneous handles/connections to SpikeGLX,
// one fetching the NI stream, one fetching the IMEC stream,
// each running in its own thread.
// =================================

#include "SglxCppClient.h"
#include <stdio.h>
#include <thread>
#include <chrono>
#include <vector>

using namespace std;

static const char*  addr = "129.236.161.4";
static int          port = 4142;


// Fetch loop for one stream, run on its own thread with its own handle.
//

static void fetchLoop( void *hSglx, const char *tag, int js, int ip, int nIters )
{
    double  srate = sglx_getStreamSampleRate( hSglx, js, ip );

    if( !srate ) {
        printf( "[%s] no such stream (js=%d,ip=%d): %s\n", tag, js, ip, sglx_getError( hSglx ) );
        return;
    }

    printf( "[%s] stream (js=%d,ip=%d) sample rate = %.3f\n", tag, js, ip, srate );

    for( int i = 0; i < nIters; ++i ) {

        cppClient_sglx_fetch    io;

        io.chans.push_back( -1 );      // all acquired channels
        io.channel_subset = &io.chans[0];
        io.n_cs         = 1;
        io.max_samps    = int( srate );    // ~1 second
        io.downsample   = 1;
        io.js           = js;
        io.ip           = ip;

        t_ull   headCt = sglx_fetchLatest( io, hSglx );

        if( !headCt ) {
            printf( "[%s] fetch error: %s\n", tag, sglx_getError( hSglx ) );
            return;
        }

        printf( "[%s] iter %d: headCt=%llu samples=%zu\n",
                tag, i, headCt, io.data.size() );

        this_thread::sleep_for( chrono::milliseconds( 300 ) );
    }
}

void testDualHandle()
{
    printf( "\nTesting two simultaneous handles (NI + IMEC)...\n\n" );

    // Create + connect BOTH handles sequentially, from this thread,
    // before any worker thread touches either one.

    void    *hNI = sglx_createHandle_std();
    void    *hIM = sglx_createHandle_std();

    bool    okNI = sglx_connect( hNI, addr, port );
    printf( "NI  handle connect: %s  version <%s>\n",
            okNI ? "OK" : "FAIL",
            okNI ? sglx_getVersion( hNI ) : sglx_getError( hNI ) );

    bool    okIM = sglx_connect( hIM, addr, port );
    printf( "IM  handle connect: %s  version <%s>\n",
            okIM ? "OK" : "FAIL",
            okIM ? sglx_getVersion( hIM ) : sglx_getError( hIM ) );

    if( okNI && okIM ) {

        printf( "\nBoth handles connected simultaneously -- starting fetch threads...\n\n" );

        thread  tNI( fetchLoop, hNI, "NI", 0, 0, 5 );
        thread  tIM( fetchLoop, hIM, "IM", 2, 0, 5 );

        tNI.join();
        tIM.join();
    }

    // Close/destroy sequentially again, back on this thread.

    sglx_close( hNI );
    sglx_destroyHandle( hNI );
    sglx_close( hIM );
    sglx_destroyHandle( hIM );

    printf( "\nDone.\n" );
}


