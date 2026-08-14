// =================================
// Demo entry point for DemoRemoteAPI
// =================================

#include "DemoRemoteAPI.h"
#include "TestDualHandle.h"
#include <stdlib.h>

int main()
{
    //justConnect();
     testDualHandle();    // confirmed: SpikeGLX accepts 2 simultaneous handles/threads

    system( "pause" );
    return 0;
}
