// Reads the fixture FilterGen/gen_noise_covariance_fixture.py generates (a
// real preprocessed recording + one real target/interferer noise-covariance
// estimate from generate_filter.noise_covariance_vectorized), runs the same
// computation through the batched GPU kernel (NoiseCovariance.cu), and
// checks the resulting R matrix matches within tolerance. Run:
//
//   python FilterGen/gen_noise_covariance_fixture.py --out-meta ... --out-scratch ...
//   test_noise_covariance_equivalence.exe <meta> <scratch>

#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>

#include "CudaUtil.h"
#include "ScratchMemmap.h"
#include "NoiseCovariance.h"

int main( int argc, char **argv )
{
    if( argc < 3 ) {
        std::fprintf( stderr, "usage: %s <fixture_meta.bin> <fixture_scratch.f32>\n", argv[0] );
        return 2;
    }

    std::ifstream fh( argv[1], std::ios::binary );
    if( !fh.is_open() ) {
        std::fprintf( stderr, "could not open '%s'\n", argv[1] );
        return 2;
    }

    long long nSamplesTrain = 0;
    int nChannelsGroup = 0, N = 0, templateLength = 0, templateOffset = 0;
    fh.read( reinterpret_cast<char*>( &nSamplesTrain ), sizeof(long long) );
    fh.read( reinterpret_cast<char*>( &nChannelsGroup ), sizeof(int) );
    fh.read( reinterpret_cast<char*>( &N ), sizeof(int) );
    fh.read( reinterpret_cast<char*>( &templateLength ), sizeof(int) );
    fh.read( reinterpret_cast<char*>( &templateOffset ), sizeof(int) );

    int nSpikes = 0;
    fh.read( reinterpret_cast<char*>( &nSpikes ), sizeof(int) );
    std::vector<long long> spikeTimes( nSpikes );
    fh.read( reinterpret_cast<char*>( spikeTimes.data() ), nSpikes * sizeof(long long) );

    std::vector<int> selChannels( N );
    fh.read( reinterpret_cast<char*>( selChannels.data() ), N * sizeof(int) );

    int dim = 0;
    fh.read( reinterpret_cast<char*>( &dim ), sizeof(int) );
    std::vector<double> Rexpect( static_cast<size_t>( dim ) * dim );
    fh.read( reinterpret_cast<char*>( Rexpect.data() ), Rexpect.size() * sizeof(double) );

    std::printf( "Fixture: nSamplesTrain=%lld nChannelsGroup=%d N=%d templateLength=%d "
                 "nSpikes=%d dim=%d\n", nSamplesTrain, nChannelsGroup, N, templateLength, nSpikes, dim );

    ScratchMemmap scratch( argv[2], nSamplesTrain, nChannelsGroup );

    size_t nFloats = static_cast<size_t>( nSamplesTrain ) * nChannelsGroup;
    float *d_data = nullptr;
    CUDA_CHECK( cudaMalloc( &d_data, nFloats * sizeof(float) ) );
    CUDA_CHECK( cudaMemcpy( d_data, scratch.row( 0 ), nFloats * sizeof(float), cudaMemcpyHostToDevice ) );

    std::vector<SpikeFreeSegment> segments = findSpikeFreeSegments(
        spikeTimes, nSamplesTrain, templateLength, templateOffset );
    std::printf( "Found %zu spike-free segments\n", segments.size() );

    std::vector<int> segmentOffsets( 1, 0 );
    std::vector<int> segmentCounts( 1, static_cast<int>( segments.size() ) );

    std::vector<std::vector<double> > result = computeNoiseCovarianceBatched(
        d_data, nSamplesTrain, nChannelsGroup, templateLength, N,
        selChannels, segments, segmentOffsets, segmentCounts, /*nUnits=*/1 );

    cudaFree( d_data );

    const std::vector<double> &R = result[0];
    double maxAbsErr = 0.0, maxAbsVal = 0.0;
    for( size_t i = 0; i < R.size(); ++i ) {
        maxAbsErr = std::max( maxAbsErr, std::fabs( R[i] - Rexpect[i] ) );
        maxAbsVal = std::max( maxAbsVal, std::fabs( Rexpect[i] ) );
    }
    double relErr = maxAbsVal > 0 ? maxAbsErr / maxAbsVal : maxAbsErr;

    std::printf( "max |R - R_expect|        = %.6e\n", maxAbsErr );
    std::printf( "max |R_expect|             = %.6e\n", maxAbsVal );
    std::printf( "max relative error         = %.6e\n", relErr );

    const double tol = 1e-4;
    if( relErr < tol ) {
        std::printf( "PASS\n" );
        return 0;
    }
    std::printf( "FAIL (relative tolerance %.1e)\n", tol );
    return 1;
}
