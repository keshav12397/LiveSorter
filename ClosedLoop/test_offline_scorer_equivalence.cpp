// Reads the fixture FilterGen/gen_offline_scorer_fixture.py generates (one
// real unit's already-fit filter, scored via the real
// generate_filter.filter_output + threshold_sweep_real.find_all_peaks over
// the held-out test split), runs the same range through
// scoreAllUnitsOffline() (GPU, all -infinity threshold), and checks the
// resulting peak list matches within a small index/score tolerance. Run:
//
//   python FilterGen/gen_offline_scorer_fixture.py --out fixture.bin ...
//   test_offline_scorer_equivalence.exe fixture.bin <binPath> <metaPath> <channelMapJson>

#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <utility>

#include "OfflineScorer.h"
#include "SglxMetaReader.h"

int main( int argc, char **argv )
{
    if( argc < 5 ) {
        std::fprintf( stderr, "usage: %s <fixture.bin> <binPath> <metaPath> <channelMapJson>\n", argv[0] );
        return 2;
    }

    std::ifstream fh( argv[1], std::ios::binary );
    if( !fh.is_open() ) {
        std::fprintf( stderr, "could not open '%s'\n", argv[1] );
        return 2;
    }

    int N = 0, templateLength = 0;
    long long testStart = 0, testNum = 0;
    fh.read( reinterpret_cast<char*>( &N ), sizeof(int) );
    fh.read( reinterpret_cast<char*>( &templateLength ), sizeof(int) );
    int testStart32 = 0, testNum32 = 0;
    fh.read( reinterpret_cast<char*>( &testStart32 ), sizeof(int) );
    fh.read( reinterpret_cast<char*>( &testNum32 ), sizeof(int) );
    testStart = testStart32;
    testNum = testNum32;

    std::vector<int32_t> selChannels( N );
    fh.read( reinterpret_cast<char*>( selChannels.data() ), N * sizeof(int32_t) );

    std::vector<float> filt( static_cast<size_t>( templateLength ) * N );
    fh.read( reinterpret_cast<char*>( filt.data() ), filt.size() * sizeof(float) );

    int nPeaksExpect = 0;
    fh.read( reinterpret_cast<char*>( &nPeaksExpect ), sizeof(int) );
    std::vector<long long> peakIdxExpect( nPeaksExpect );
    std::vector<double> peakScoreExpect( nPeaksExpect );
    fh.read( reinterpret_cast<char*>( peakIdxExpect.data() ), nPeaksExpect * sizeof(long long) );
    fh.read( reinterpret_cast<char*>( peakScoreExpect.data() ), nPeaksExpect * sizeof(double) );

    std::printf( "Fixture: N=%d templateLength=%d testStart=%lld testNum=%lld nPeaksExpect=%d\n",
                 N, templateLength, testStart, testNum, nPeaksExpect );
    std::fflush( stdout );

    std::vector<int> carChannelIds = loadChanMapJson( argv[4] );
    std::printf( "carChannelIds: %zu\n", carChannelIds.size() ); std::fflush( stdout );

    std::vector<int> unitIds( 1, 0 );

    int minSep = templateLength / 2;
    // matchedFilterKernel needs (chunkSamples+L-1)*N*sizeof(float) bytes of
    // PER-BLOCK shared memory (see MultiConvolutionEngine's constructor-time
    // check) -- must stay small (this is why the live pipeline only ever
    // uses ~5ms-sized fetch chunks, not an arbitrarily large one).
    int chunkSamples = 2000;
    int detectionCapacity = 5000;

    std::printf( "calling scoreAllUnitsOffline...\n" ); std::fflush( stdout );
    std::vector<UnitPeaks> result = scoreAllUnitsOffline(
        argv[2], argv[3], carChannelIds, /*applyHighpass=*/true, /*highpassCutoffHz=*/300.0,
        unitIds, selChannels, filt, templateLength, N,
        testStart, testNum, chunkSamples, minSep, detectionCapacity );
    std::printf( "scoreAllUnitsOffline returned\n" ); std::fflush( stdout );

    const UnitPeaks &got = result[0];
    std::printf( "engine produced %zu peaks\n", got.sampleIdx.size() );

    // Build a sorted (sampleIdx -> score) map for the GPU output, match each
    // expected Python peak to the nearest GPU peak within a small window.
    std::vector<std::pair<long long, double> > gotSorted;
    for( size_t i = 0; i < got.sampleIdx.size(); ++i )
        gotSorted.push_back( std::make_pair( got.sampleIdx[i], got.score[i] ) );
    std::sort( gotSorted.begin(), gotSorted.end() );

    const long long tolIdx = 2;
    const double tolScoreRel = 0.05; // float32 GPU vs float32 CPU, different reduction order
    int matched = 0;
    double maxScoreErr = 0.0;

    for( int i = 0; i < nPeaksExpect; ++i ) {
        long long want = peakIdxExpect[i];
        auto it = std::lower_bound( gotSorted.begin(), gotSorted.end(),
                                     std::make_pair( want - tolIdx, -1e300 ) );
        bool found = false;
        for( ; it != gotSorted.end() && it->first <= want + tolIdx; ++it ) {
            double err = std::fabs( it->second - peakScoreExpect[i] ) /
                         std::max( 1e-9, std::fabs( peakScoreExpect[i] ) );
            if( err < tolScoreRel ) {
                found = true;
                maxScoreErr = std::max( maxScoreErr, err );
                break;
            }
        }
        if( found )
            ++matched;
    }

    double matchFrac = nPeaksExpect > 0 ? static_cast<double>( matched ) / nPeaksExpect : 1.0;
    std::printf( "matched %d/%d expected peaks (%.4f%%), max matched score rel err = %.4e\n",
                 matched, nPeaksExpect, 100.0 * matchFrac, maxScoreErr );
    std::printf( "peak count ratio (engine/expected) = %.4f\n",
                 nPeaksExpect > 0 ? static_cast<double>( got.sampleIdx.size() ) / nPeaksExpect : 0.0 );

    if( matchFrac > 0.99 ) {
        std::printf( "PASS\n" );
        return 0;
    }
    std::printf( "FAIL (match fraction threshold 0.99)\n" );
    return 1;
}
