// Reads the fixture FilterGen/gen_nms_fixture.py generates (a synthetic D
// signal + scipy.signal.find_peaks(distance=...)'s real peak list), feeds
// the SAME D values through ConvolutionEngine using a 1-tap identity filter
// (templateLength=1 makes leftMargin=rightMargin=0, so D[n] == data[n]
// exactly -- lets this test the NMS decision logic in isolation from the
// convolution math), streamed across several chunks to also exercise the
// cross-chunk decision/history state machine, and checks the peak list
// matches scipy exactly (not just approximately -- this is the whole point
// of the fix). Run:
//
//   python FilterGen/gen_nms_fixture.py --out nms_fixture.bin
//   test_convolution_engine_nms_equivalence.exe nms_fixture.bin

#include <fstream>
#include <vector>
#include <cstdio>
#include <set>
#include <algorithm>

#include "ConvolutionEngine.h"

int main( int argc, char **argv )
{
    if( argc < 2 ) {
        std::fprintf( stderr, "usage: %s <fixture.bin>\n", argv[0] );
        return 2;
    }

    std::ifstream fh( argv[1], std::ios::binary );
    if( !fh.is_open() ) {
        std::fprintf( stderr, "could not open '%s'\n", argv[1] );
        return 2;
    }

    int n = 0, minSep = 0;
    fh.read( reinterpret_cast<char*>( &n ), sizeof(int) );
    fh.read( reinterpret_cast<char*>( &minSep ), sizeof(int) );

    std::vector<double> D( n );
    fh.read( reinterpret_cast<char*>( D.data() ), n * sizeof(double) );

    int nExpect = 0;
    fh.read( reinterpret_cast<char*>( &nExpect ), sizeof(int) );
    std::vector<long long> idxExpect( nExpect );
    fh.read( reinterpret_cast<char*>( idxExpect.data() ), nExpect * sizeof(long long) );

    std::printf( "n=%d minSep=%d nExpect=%d\n", n, minSep, nExpect );

    std::vector<double> taps( 1, 1.0 ); // 1-tap identity: D[n] == data[n]
    ConvolutionEngine engine( /*templateLength=*/1, /*nChannels=*/1, taps, minSep );

    std::vector<PeakEvent> allPeaks;
    const int chunkSize = 7919; // deliberately not a divisor of n, to exercise ragged chunk boundaries
    long long offset = 0;
    while( offset < n ) {
        int len = static_cast<int>( std::min<long long>( chunkSize, n - offset ) );
        std::vector<PeakEvent> peaks = engine.processChunk( &D[offset], len, offset );
        allPeaks.insert( allPeaks.end(), peaks.begin(), peaks.end() );
        offset += len;
    }
    // True end of this finite stream -- flush the tail processChunk() alone
    // would hold back forever waiting for future context that isn't coming.
    std::vector<PeakEvent> flushed = engine.flush();
    allPeaks.insert( allPeaks.end(), flushed.begin(), flushed.end() );

    std::vector<long long> gotIdx;
    for( size_t i = 0; i < allPeaks.size(); ++i )
        gotIdx.push_back( allPeaks[i].sampleIndex );
    std::sort( gotIdx.begin(), gotIdx.end() );

    std::set<long long> gotSet( gotIdx.begin(), gotIdx.end() );
    std::set<long long> expectSet( idxExpect.begin(), idxExpect.end() );

    int matched = 0;
    for( long long v : expectSet )
        if( gotSet.count( v ) )
            ++matched;

    std::printf( "ConvolutionEngine produced %zu peaks (scipy expected %d)\n", gotIdx.size(), nExpect );
    std::printf( "exact-index matches: %d/%d (%.4f%%)\n", matched, nExpect,
                 nExpect > 0 ? 100.0 * matched / nExpect : 100.0 );

    if( argc > 2 ) {
        std::ofstream out( argv[2], std::ios::binary );
        int cnt = static_cast<int>( gotIdx.size() );
        out.write( reinterpret_cast<const char*>( &cnt ), sizeof(int) );
        out.write( reinterpret_cast<const char*>( gotIdx.data() ), gotIdx.size() * sizeof(long long) );
    }

    if( matched == nExpect && gotIdx.size() == static_cast<size_t>( nExpect ) ) {
        std::printf( "PASS (exact match)\n" );
        return 0;
    }
    std::printf( "FAIL\n" );
    return 1;
}
