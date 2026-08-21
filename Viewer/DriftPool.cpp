#include "DriftPool.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>


namespace {

// numpy nanmedian over one column (across units), at a fixed time index.
// NaN in, NaN out if nothing is finite -- callers handle that via `bad`.
double nanmedianColumn( const std::vector<std::vector<double> > &Y, size_t t )
{
    std::vector<double> vals;
    vals.reserve( Y.size() );
    for( size_t u = 0; u < Y.size(); ++u ) {
        double v = Y[u][t];
        if( std::isfinite( v ) )
            vals.push_back( v );
    }
    if( vals.empty() )
        return std::numeric_limits<double>::quiet_NaN();

    std::sort( vals.begin(), vals.end() );
    size_t n = vals.size();
    if( n % 2 == 1 )
        return vals[n / 2];
    return 0.5 * ( vals[n / 2 - 1] + vals[n / 2] );
}

// np.interp(x, xp, fp): xp strictly ascending, edge-clamped outside range.
double interpClamped( double x, const std::vector<double> &xp, const std::vector<double> &fp )
{
    if( xp.empty() )
        return 0.0;
    if( x <= xp.front() )
        return fp.front();
    if( x >= xp.back() )
        return fp.back();

    // xp is small (one entry per time bin actually covered, typically tens
    // to low hundreds) -- linear scan matches np.interp's own behaviour
    // closely enough that a binary search would only be optimising a cost
    // this function's caller (a >500 ms budget) does not need to care about.
    for( size_t i = 1; i < xp.size(); ++i ) {
        if( x <= xp[i] ) {
            double t = ( x - xp[i - 1] ) / ( xp[i] - xp[i - 1] );
            return fp[i - 1] + t * ( fp[i] - fp[i - 1] );
        }
    }
    return fp.back();
}

} // namespace


std::vector<double> pooledMedianMotion( const std::vector<std::vector<double> > &Y,
                                         const std::vector<double> &grid )
{
    const size_t nTime = grid.size();

    if( Y.empty() )
        return std::vector<double>( nTime, 0.0 );

    std::vector<double> m( nTime );
    for( size_t t = 0; t < nTime; ++t )
        m[t] = nanmedianColumn( Y, t );

    std::vector<bool> bad( nTime );
    bool allBad = true, anyBad = false;
    for( size_t t = 0; t < nTime; ++t ) {
        bad[t] = !std::isfinite( m[t] );
        allBad = allBad && bad[t];
        anyBad = anyBad || bad[t];
    }

    if( allBad )
        return std::vector<double>( nTime, 0.0 );

    if( anyBad ) {
        std::vector<double> goodX, goodY;
        goodX.reserve( nTime );
        goodY.reserve( nTime );
        for( size_t t = 0; t < nTime; ++t ) {
            if( !bad[t] ) {
                goodX.push_back( grid[t] );
                goodY.push_back( m[t] );
            }
        }
        for( size_t t = 0; t < nTime; ++t ) {
            if( bad[t] )
                m[t] = interpClamped( grid[t], goodX, goodY );
        }
    }

    double mean = std::accumulate( m.begin(), m.end(), 0.0 ) / static_cast<double>( nTime );
    for( size_t t = 0; t < nTime; ++t )
        m[t] -= mean;

    return m;
}
