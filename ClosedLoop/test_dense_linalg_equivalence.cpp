// Reads the fixture FilterGen/gen_dense_linalg_fixture.py generates (a
// random SPD matrix + RHS solved via numpy), solves the same systems with
// CholeskySolver, and checks the results match within a tight float64
// tolerance. Run:
//
//   python FilterGen/gen_dense_linalg_fixture.py --out fixture.bin
//   test_dense_linalg_equivalence.exe fixture.bin
//
// Exits 0 and prints PASS on success, nonzero and PASS/FAIL detail otherwise.

#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "DenseLinAlg.h"

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

    int n = 0, m = 0;
    fh.read( reinterpret_cast<char*>( &n ), sizeof(int) );
    fh.read( reinterpret_cast<char*>( &m ), sizeof(int) );

    std::vector<double> A( static_cast<size_t>( n ) * n );
    std::vector<double> b( n );
    std::vector<double> B( static_cast<size_t>( n ) * m );
    std::vector<double> xExpect( n );
    std::vector<double> XExpect( static_cast<size_t>( n ) * m );

    fh.read( reinterpret_cast<char*>( A.data() ), A.size() * sizeof(double) );
    fh.read( reinterpret_cast<char*>( b.data() ), b.size() * sizeof(double) );
    fh.read( reinterpret_cast<char*>( B.data() ), B.size() * sizeof(double) );
    fh.read( reinterpret_cast<char*>( xExpect.data() ), xExpect.size() * sizeof(double) );
    fh.read( reinterpret_cast<char*>( XExpect.data() ), XExpect.size() * sizeof(double) );

    CholeskySolver solver( A, n );
    std::vector<double> x = solver.solve( b );
    std::vector<double> X = solver.solveMatrix( B, m );

    const double tol = 1e-8;
    double maxErrX = 0.0, maxErrBigX = 0.0;

    for( int i = 0; i < n; ++i )
        maxErrX = std::max( maxErrX, std::fabs( x[i] - xExpect[i] ) );
    for( size_t i = 0; i < X.size(); ++i )
        maxErrBigX = std::max( maxErrBigX, std::fabs( X[i] - XExpect[i] ) );

    std::printf( "max |x - x_expect|  = %.3e\n", maxErrX );
    std::printf( "max |X - X_expect|  = %.3e\n", maxErrBigX );

    if( maxErrX < tol && maxErrBigX < tol ) {
        std::printf( "PASS\n" );
        return 0;
    }
    std::printf( "FAIL (tolerance %.1e)\n", tol );
    return 1;
}
