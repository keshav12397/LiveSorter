// One-off diagnostic (not a validated equivalence test -- no fixture, no
// PASS/FAIL exit code): isolates one already-calibrated unit's filter and
// scores it through scoreAllUnitsOffline() at TWO different chunk sizes --
// 2000 (what every earlier equivalence test/OfflineScorer validation used)
// and ~150 (what ImecFetchThreadGPU.cu actually uses live, from
// fetchChunkMs=5 at ~30kHz) -- to check whether GpuConvolutionEngine's
// windowed-NMS/history-buffer logic behaves differently at the much smaller
// chunk size the live pipeline runs at, which was never explicitly
// exercised by this session's validation (all of it used chunkSamples=2000).
//
// Run:
//   test_chunksize_diagnostic.exe <binPath> <metaPath> <channelMapJson>
//       <unitId> <ch0> <ch1> <ch2> <ch3> <ch4> <filtersBinPath> <unitIndex>
//       <templateLength> <nChannels> <testStartSample> <testNumSamples>
//       <threshold> <windowSamples> <durationSeconds>

#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "OfflineScorer.h"
#include "SglxMetaReader.h"
#include "ThresholdSweep.h"

int main( int argc, char **argv )
{
    if( argc < 17 ) {
        std::fprintf( stderr, "usage: %s <binPath> <metaPath> <channelMapJson> <unitId> "
            "<ch0..ch4> <filtersBin> <unitIndex> <templateLength> <nChannels> "
            "<testStart> <testNum> <threshold> <window> <durationSec>\n", argv[0] );
        return 2;
    }

    std::string binPath = argv[1], metaPath = argv[2], chanMapJson = argv[3];
    int unitId = std::atoi( argv[4] );
    std::vector<int32_t> selChannels;
    for( int i = 0; i < 5; ++i )
        selChannels.push_back( std::atoi( argv[5 + i] ) );
    std::string filtersBinPath = argv[10];
    int unitIndex = std::atoi( argv[11] );
    int templateLength = std::atoi( argv[12] );
    int nChannels = std::atoi( argv[13] );
    long long testStart = std::atoll( argv[14] );
    long long testNum = std::atoll( argv[15] );
    double knownThreshold = std::atof( argv[16] );
    long long window = ( argc > 17 ) ? std::atoll( argv[17] ) : templateLength / 4;
    double durationSeconds = ( argc > 18 ) ? std::atof( argv[18] ) : testNum / 30000.0;

    std::vector<int> carChannelIds = loadChanMapJson( chanMapJson );

    // sel_channels.bin/filters.bin store RAW ids / CAR-group-relative filter
    // taps for a *sub*channel selection already made -- scoreAllUnitsOffline
    // wants indices INTO the CAR group, so translate raw -> CAR-group index
    // the same way ImecFetchThreadGPU.cu / mainCalibrateAllUnits.cu do.
    std::vector<int32_t> selCarGroupIdx( 5 );
    for( int c = 0; c < 5; ++c ) {
        std::vector<int>::const_iterator it =
            std::find( carChannelIds.begin(), carChannelIds.end(), selChannels[c] );
        if( it == carChannelIds.end() ) {
            std::fprintf( stderr, "channel %d not in CAR group\n", selChannels[c] );
            return 1;
        }
        selCarGroupIdx[c] = static_cast<int32_t>( it - carChannelIds.begin() );
    }

    std::ifstream fh( filtersBinPath.c_str(), std::ios::binary );
    long long filterFloatsPerUnit = static_cast<long long>( templateLength ) * nChannels;
    fh.seekg( static_cast<std::streamoff>( unitIndex ) * filterFloatsPerUnit * sizeof(float) );
    std::vector<float> filt( filterFloatsPerUnit );
    fh.read( reinterpret_cast<char*>( filt.data() ), filt.size() * sizeof(float) );

    std::vector<int> unitIds( 1, unitId );

    for( int trial = 0; trial < 2; ++trial ) {
        int chunkSamples = ( trial == 0 ) ? 2000 : 150;
        int minSep = templateLength / 2;
        int detectionCapacity = 20000;

        std::printf( "\n=== chunkSamples=%d ===\n", chunkSamples );
        std::vector<UnitPeaks> result = scoreAllUnitsOffline(
            binPath, metaPath, carChannelIds, /*applyHighpass=*/true, /*highpassCutoffHz=*/300.0,
            unitIds, selCarGroupIdx, filt, templateLength, nChannels,
            testStart, testNum, chunkSamples, minSep, detectionCapacity );

        const UnitPeaks &peaks = result[0];
        std::printf( "n candidate peaks (all -inf threshold): %zu\n", peaks.sampleIdx.size() );

        // Score at the KNOWN calibrated threshold directly (no sweep --
        // this is the exact operating point summary.csv already picked).
        long long hits = 0, fp = 0;
        std::vector<long long> detections;
        for( size_t i = 0; i < peaks.sampleIdx.size(); ++i )
            if( peaks.score[i] > knownThreshold )
                detections.push_back( peaks.sampleIdx[i] );
        std::sort( detections.begin(), detections.end() );
        std::printf( "n detections at threshold=%.6f: %zu\n", knownThreshold, detections.size() );

        char outPath[256];
        std::snprintf( outPath, sizeof(outPath), "chunksize_diag_unit%d_chunk%d.csv", unitId, chunkSamples );
        std::ofstream out( outPath );
        out << "sample_index_rel_test_start\n";
        for( size_t i = 0; i < detections.size(); ++i )
            out << detections[i] << "\n";
        std::printf( "wrote %s\n", outPath );
    }

    return 0;
}
