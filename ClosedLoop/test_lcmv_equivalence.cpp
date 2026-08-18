// Reads the fixture FilterGen/gen_lcmv_fixture.py generates (a synthetic
// R/target/interferer setup solved via the real generate_filter.lcmv_filter),
// calls lcmvFilter() with the same inputs, and checks the result matches
// within a tight float64 tolerance. Run:
//
//   python FilterGen/gen_lcmv_fixture.py --out lcmv_fixture.bin
//   test_lcmv_equivalence.exe lcmv_fixture.bin

#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

#include "LcmvFit.h"

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

    int dim = 0, nInterferers = 0;
    fh.read( reinterpret_cast<char*>( &dim ), sizeof(int) );
    fh.read( reinterpret_cast<char*>( &nInterferers ), sizeof(int) );

    std::vector<double> R( static_cast<size_t>( dim ) * dim );
    fh.read( reinterpret_cast<char*>( R.data() ), R.size() * sizeof(double) );

    std::vector<double> sFlat( dim );
    fh.read( reinterpret_cast<char*>( sFlat.data() ), sFlat.size() * sizeof(double) );

    std::vector<std::vector<double> > interfererFlats( nInterferers, std::vector<double>( dim ) );
    for( int j = 0; j < nInterferers; ++j )
        fh.read( reinterpret_cast<char*>( interfererFlats[j].data() ), dim * sizeof(double) );

    double ridge = 0.0;
    fh.read( reinterpret_cast<char*>( &ridge ), sizeof(double) );

    std::vector<double> fFlatExpect( dim );
    fh.read( reinterpret_cast<char*>( fFlatExpect.data() ), fFlatExpect.size() * sizeof(double) );

    std::vector<double> gainsExpect( 1 + nInterferers );
    fh.read( reinterpret_cast<char*>( gainsExpect.data() ), gainsExpect.size() * sizeof(double) );

    std::vector<double> fFlat = lcmvFilter( sFlat, interfererFlats, R, ridge );

    double maxErrF = 0.0;
    for( int i = 0; i < dim; ++i )
        maxErrF = std::max( maxErrF, std::fabs( fFlat[i] - fFlatExpect[i] ) );

    auto dot = [&]( const std::vector<double> &a, const std::vector<double> &b ) {
        double s = 0.0;
        for( size_t i = 0; i < a.size(); ++i ) s += a[i] * b[i];
        return s;
    };

    std::vector<double> gains;
    gains.push_back( dot( fFlat, sFlat ) );
    for( int j = 0; j < nInterferers; ++j )
        gains.push_back( dot( fFlat, interfererFlats[j] ) );

    double maxErrGain = 0.0;
    for( size_t i = 0; i < gains.size(); ++i )
        maxErrGain = std::max( maxErrGain, std::fabs( gains[i] - gainsExpect[i] ) );

    std::printf( "max |f_flat - expect|   = %.3e\n", maxErrF );
    std::printf( "max |gain - expect|     = %.3e\n", maxErrGain );
    std::printf( "gains: target=%.6f", gains[0] );
    for( int j = 0; j < nInterferers; ++j )
        std::printf( " interferer[%d]=%.3e", j, gains[j + 1] );
    std::printf( "\n" );

    const double tol = 1e-6;
    if( maxErrF < tol && maxErrGain < tol ) {
        std::printf( "PASS\n" );
        return 0;
    }
    std::printf( "FAIL (tolerance %.1e)\n", tol );
    return 1;
}
