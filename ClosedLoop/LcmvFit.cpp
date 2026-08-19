#include "LcmvFit.h"

#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <utility>

#include "DenseLinAlg.h"


std::vector<double> meanWaveform( const ScratchMemmap &data,
                                   const std::vector<long long> &spikeTimes,
                                   int templateLength, int templateOffset,
                                   int maxSpikes, unsigned int seed,
                                   long long &nUsedOut )
{
    const int nCh = data.nChannels();

    std::vector<long long> valid;
    valid.reserve( spikeTimes.size() );
    for( size_t i = 0; i < spikeTimes.size(); ++i ) {
        long long t = spikeTimes[i];
        if( t > templateOffset && t < data.nSamples() - templateLength + templateOffset )
            valid.push_back( t );
    }
    if( valid.empty() )
        throw std::runtime_error( "meanWaveform: no valid spikes (all too close to recording edges)" );

    if( static_cast<int>( valid.size() ) > maxSpikes ) {
        std::mt19937 rng( seed );
        std::shuffle( valid.begin(), valid.end(), rng );
        valid.resize( maxSpikes );
    }

    std::vector<double> acc( static_cast<size_t>( templateLength ) * nCh, 0.0 );
    for( size_t i = 0; i < valid.size(); ++i ) {
        long long start = valid[i] - templateOffset;
        for( int t = 0; t < templateLength; ++t ) {
            const float *row = data.row( start + t );
            double *accRow = &acc[static_cast<size_t>( t ) * nCh];
            for( int c = 0; c < nCh; ++c )
                accRow[c] += row[c];
        }
    }

    double n = static_cast<double>( valid.size() );
    for( size_t i = 0; i < acc.size(); ++i )
        acc[i] /= n;

    nUsedOut = static_cast<long long>( valid.size() );
    return acc;
}


std::vector<int> selectChannels( const std::vector<double> &targetWf,
                                  const std::vector<std::vector<double> > &interfererWfs,
                                  int nChannelsAll, int templateLength,
                                  int nChannelsPick )
{
    const double eps = 1e-9;

    std::vector<double> targetEnergy( nChannelsAll, 0.0 );
    for( int t = 0; t < templateLength; ++t )
        for( int c = 0; c < nChannelsAll; ++c ) {
            double v = targetWf[static_cast<size_t>( t ) * nChannelsAll + c];
            targetEnergy[c] += v * v;
        }

    std::vector<double> interfererEnergy( nChannelsAll, 0.0 );
    if( !interfererWfs.empty() ) {
        for( size_t k = 0; k < interfererWfs.size(); ++k ) {
            std::vector<double> e( nChannelsAll, 0.0 );
            for( int t = 0; t < templateLength; ++t )
                for( int c = 0; c < nChannelsAll; ++c ) {
                    double v = interfererWfs[k][static_cast<size_t>( t ) * nChannelsAll + c];
                    e[c] += v * v;
                }
            for( int c = 0; c < nChannelsAll; ++c )
                interfererEnergy[c] = std::max( interfererEnergy[c], e[c] );
        }
    }

    double maxTargetEnergy = *std::max_element( targetEnergy.begin(), targetEnergy.end() );
    double floor = 0.05 * maxTargetEnergy;

    std::vector<double> score( nChannelsAll );
    for( int c = 0; c < nChannelsAll; ++c )
        score[c] = ( targetEnergy[c] >= floor )
                 ? targetEnergy[c] / ( interfererEnergy[c] + eps )
                 : -std::numeric_limits<double>::infinity();

    std::vector<int> order( nChannelsAll );
    for( int c = 0; c < nChannelsAll; ++c )
        order[c] = c;
    std::sort( order.begin(), order.end(),
               [&]( int a, int b ) { return score[a] > score[b]; } );

    order.resize( std::min<size_t>( nChannelsPick, order.size() ) );
    return order;
}


namespace {

double ptp( const std::vector<double> &wf, int templateLength, int nChannelsAll, int c )
{
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for( int t = 0; t < templateLength; ++t ) {
        double v = wf[static_cast<size_t>( t ) * nChannelsAll + c];
        lo = std::min( lo, v );
        hi = std::max( hi, v );
    }
    return hi - lo;
}

std::pair<double, double> peakChannelAndPos(
    const ScratchMemmap &data, const std::vector<long long> &spikeTimes,
    int templateLength, int templateOffset, int peakCheckSpikes, unsigned int seed,
    const std::vector<int> &npCh,
    const std::map<int, std::pair<double, double> > &channelPositions )
{
    long long nUsed = 0;
    std::vector<double> wf = meanWaveform( data, spikeTimes, templateLength, templateOffset,
                                            peakCheckSpikes, seed, nUsed );

    int nCh = data.nChannels();
    int peakLocal = 0;
    double bestPtp = -1.0;
    for( int c = 0; c < nCh; ++c ) {
        double p = ptp( wf, templateLength, nCh, c );
        if( p > bestPtp ) { bestPtp = p; peakLocal = c; }
    }

    int rawCh = npCh[peakLocal];
    std::map<int, std::pair<double, double> >::const_iterator it = channelPositions.find( rawCh );
    if( it != channelPositions.end() )
        return it->second;
    return std::make_pair( static_cast<double>( peakLocal ), 0.0 ); // index-distance fallback
}

} // namespace


std::vector<long long> autoPickInterferersSpatial(
    const ScratchMemmap &data,
    const std::map<long long, std::vector<long long> > &spikesByCluster,
    const std::map<long long, std::string> &labels,
    const std::vector<int> &npCh,
    long long targetId, int nInterferers, int templateLength, int templateOffset,
    const std::map<int, std::pair<double, double> > &channelPositions,
    unsigned int seed )
{
    const int peakCheckSpikes = 200;
    int poolSize = std::max( 3 * nInterferers, 30 );

    // Rank all non-target, non-noise clusters by spike count descending --
    // mirrors auto_pick_interferers_spatial()'s two-stage shortlist. Sizes
    // come straight from the already-grouped map, no re-scan of the flat
    // spike arrays.
    std::vector<std::pair<long long, long long> > byCount; // (id, count)
    byCount.reserve( spikesByCluster.size() );
    for( std::map<long long, std::vector<long long> >::const_iterator it = spikesByCluster.begin();
         it != spikesByCluster.end(); ++it )
        byCount.push_back( std::make_pair( it->first, static_cast<long long>( it->second.size() ) ) );
    std::sort( byCount.begin(), byCount.end(),
               []( const std::pair<long long, long long> &a, const std::pair<long long, long long> &b ) {
                   return a.second > b.second;
               } );

    std::vector<long long> pool;
    for( size_t i = 0; i < byCount.size() && static_cast<int>( pool.size() ) < poolSize; ++i ) {
        long long cid = byCount[i].first;
        if( cid == targetId )
            continue;
        std::map<long long, std::string>::const_iterator labIt = labels.find( cid );
        if( labIt != labels.end() ) {
            std::string lab = labIt->second;
            std::transform( lab.begin(), lab.end(), lab.begin(), ::tolower );
            if( lab == "noise" )
                continue;
        }
        pool.push_back( cid );
    }

    auto spikesFor = [&]( long long cid ) -> const std::vector<long long> & {
        static const std::vector<long long> empty;
        std::map<long long, std::vector<long long> >::const_iterator it = spikesByCluster.find( cid );
        return it != spikesByCluster.end() ? it->second : empty;
    };

    std::pair<double, double> targetPos = peakChannelAndPos(
        data, spikesFor( targetId ), templateLength, templateOffset, peakCheckSpikes, seed,
        npCh, channelPositions );

    std::vector<std::pair<double, long long> > scored; // (distance, cid)
    for( size_t i = 0; i < pool.size(); ++i ) {
        long long cid = pool[i];
        const std::vector<long long> &sp = spikesFor( cid );
        if( sp.empty() )
            continue;
        std::pair<double, double> pos;
        try {
            pos = peakChannelAndPos( data, sp, templateLength, templateOffset, peakCheckSpikes,
                                      seed, npCh, channelPositions );
        }
        catch( const std::runtime_error & ) {
            continue; // too few/edge-clipped spikes to form a waveform -- skip, matches Python
        }
        double dx = pos.first - targetPos.first;
        double dy = pos.second - targetPos.second;
        scored.push_back( std::make_pair( std::sqrt( dx * dx + dy * dy ), cid ) );
    }

    std::sort( scored.begin(), scored.end(),
               []( const std::pair<double, long long> &a, const std::pair<double, long long> &b ) {
                   return a.first < b.first;
               } );

    std::vector<long long> picked;
    for( size_t i = 0; i < scored.size() && static_cast<int>( picked.size() ) < nInterferers; ++i )
        picked.push_back( scored[i].second );

    return picked;
}


std::vector<double> lcmvFilter( const std::vector<double> &sFlat,
                                 const std::vector<std::vector<double> > &interfererFlats,
                                 const std::vector<double> &R, double ridge )
{
    int dim = static_cast<int>( sFlat.size() );
    int m = 1 + static_cast<int>( interfererFlats.size() );

    double trace = 0.0;
    for( int i = 0; i < dim; ++i )
        trace += R[static_cast<size_t>( i ) * dim + i];

    std::vector<double> Rreg = R;
    double diag = ridge * ( trace / dim );
    for( int i = 0; i < dim; ++i )
        Rreg[static_cast<size_t>( i ) * dim + i] += diag;

    // C, row-major (dim x m): column 0 = target, columns 1..m-1 = interferers.
    std::vector<double> C( static_cast<size_t>( dim ) * m );
    for( int i = 0; i < dim; ++i )
        C[static_cast<size_t>( i ) * m + 0] = sFlat[i];
    for( int j = 0; j < static_cast<int>( interfererFlats.size() ); ++j )
        for( int i = 0; i < dim; ++i )
            C[static_cast<size_t>( i ) * m + ( j + 1 )] = interfererFlats[j][i];

    CholeskySolver Rsolver( Rreg, dim );
    std::vector<double> RinvC = Rsolver.solveMatrix( C, m ); // (dim x m), row-major

    // M = C^T * RinvC, (m x m).
    std::vector<double> M( static_cast<size_t>( m ) * m, 0.0 );
    for( int a = 0; a < m; ++a )
        for( int b = 0; b < m; ++b ) {
            double sum = 0.0;
            for( int i = 0; i < dim; ++i )
                sum += C[static_cast<size_t>( i ) * m + a] * RinvC[static_cast<size_t>( i ) * m + b];
            M[static_cast<size_t>( a ) * m + b] = sum;
        }

    std::vector<double> g( m, 0.0 );
    g[0] = 1.0;

    CholeskySolver Msolver( M, m );
    std::vector<double> weights = Msolver.solve( g );

    std::vector<double> fFlat( dim, 0.0 );
    for( int i = 0; i < dim; ++i ) {
        double sum = 0.0;
        for( int j = 0; j < m; ++j )
            sum += RinvC[static_cast<size_t>( i ) * m + j] * weights[j];
        fFlat[i] = sum;
    }

    return fFlat;
}
