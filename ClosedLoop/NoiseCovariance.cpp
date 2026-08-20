#include "NoiseCovariance.h"

#include <atomic>
#include <thread>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>


// Local accumulator array size cap -- this project always fits N=5
// (calibrate_all_units.py's --n-channels, fixed per unit by design, see its
// module docstring), 8 leaves headroom without risking local-memory
// spilling from an oversized per-thread array.


std::vector<SpikeFreeSegment> findSpikeFreeSegments(
    const std::vector<long long> &spikeTimes, long long nSamples,
    int templateLength, int templateOffset )
{
    std::vector<std::pair<long long, long long> > intervals; // [start, end)
    intervals.reserve( spikeTimes.size() );

    for( size_t i = 0; i < spikeTimes.size(); ++i ) {
        long long t = spikeTimes[i];
        if( t >= nSamples + templateLength )
            continue; // matches Python's noise_covariance_vectorized skip condition
        long long lo = std::max<long long>( 0, t - templateOffset );
        long long hi = std::min<long long>( nSamples, t + templateLength - 1 + templateOffset );
        if( hi > lo )
            intervals.push_back( std::make_pair( lo, hi ) );
    }

    std::sort( intervals.begin(), intervals.end() );

    // Merge overlapping/touching intervals -- the union of these is exactly
    // what Python's dense spike_present boolean array marks True, just
    // built via interval merging (O(n log n)) instead of a per-sample scan
    // (O(nSamples)).
    std::vector<std::pair<long long, long long> > merged;
    for( size_t i = 0; i < intervals.size(); ++i ) {
        if( !merged.empty() && intervals[i].first <= merged.back().second )
            merged.back().second = std::max( merged.back().second, intervals[i].second );
        else
            merged.push_back( intervals[i] );
    }

    // Complement within [0, nSamples) -- the spike-free segments.
    std::vector<SpikeFreeSegment> segments;
    long long cursor = 0;
    for( size_t i = 0; i < merged.size(); ++i ) {
        if( merged[i].first > cursor ) {
            long long segStart = cursor, segEnd = merged[i].first;
            if( segEnd - segStart >= 2LL * templateLength )
                segments.push_back( SpikeFreeSegment{ static_cast<int>( segStart ), static_cast<int>( segEnd ) } );
        }
        cursor = std::max( cursor, merged[i].second );
    }
    if( cursor < nSamples && nSamples - cursor >= 2LL * templateLength )
        segments.push_back( SpikeFreeSegment{ static_cast<int>( cursor ), static_cast<int>( nSamples ) } );

    return segments;
}


namespace {

// One task per (unit, lag) pair; nlags = 2*(templateLength-1)+1, e.g. 121 for
// templateLength=61, so even a single unit has plenty of independent work to
// spread across cores.
//
// Ports noise_covariance_vectorized's per-segment windowed cross-covariance
// accumulation verbatim (see that function's docstring for the o<->lag
// index-convention derivation this mirrors exactly, not re-derived here):
// for lag d, cross[i,k] = sum_t seg[t,i] * (seg[t-d,k] if 0<=t-d<L else 0).
//
// The GPU version computed each segment's per-channel means on thread 0 and
// broadcast them through shared memory, since the mean does not depend on
// lag. Here they are computed ONCE for all lags before the parallel loop --
// same saving, and no synchronisation at all.
void noiseCovarianceForLag(
    const float *data, long long dataStride /* nChannelsGroup */,
    int templateLength, int N,
    const int *chans,
    const SpikeFreeSegment *segments, int segN,
    const double *segMeans /* [segN * N] */,
    int lag, double *out /* [N * N] */ )
{
    std::vector<double> localAcc( static_cast<size_t>( N ) * N, 0.0 );

    for( int s = 0; s < segN; ++s ) {

        const SpikeFreeSegment &seg = segments[s];
        const int L = seg.end - seg.start;
        const double *mean = segMeans + static_cast<size_t>( s ) * N;

        // t ranges only over positions whose lagged partner is also inside the
        // segment. The GPU version tested `tk < 0 || tk >= L` inside the loop
        // and skipped; clamping the bounds instead is the same set of terms
        // (the skipped ones contribute the zero-padding Python's padded window
        // also contributes) without the per-sample branch.
        const int tLo = ( lag > 0 ) ? lag : 0;
        const int tHi = ( lag > 0 ) ? L : L + lag;

        for( int t = tLo; t < tHi; ++t ) {
            const int tk = t - lag;
            const float *rowT  = data + static_cast<size_t>( seg.start + t ) * dataStride;
            const float *rowTk = data + static_cast<size_t>( seg.start + tk ) * dataStride;
            for( int i = 0; i < N; ++i ) {
                const double vi = rowT[chans[i]] - mean[i];
                double *accRow = &localAcc[static_cast<size_t>( i ) * N];
                for( int k = 0; k < N; ++k )
                    accRow[k] += vi * ( rowTk[chans[k]] - mean[k] );
            }
        }
    }

    for( int i = 0; i < N * N; ++i )
        out[i] = localAcc[i];
}

// Toeplitz-block assembly -- exact port of noise_covariance_vectorized's
// tail (cov_by_lag -> per-channel-pair Toeplitz blocks -> full R). covByLag:
// [nlags * N * N], nlags axis ascending lag -maxlag..+maxlag (same
// convention accOut/threadIdx.x already produces, no reindexing needed).
std::vector<double> assembleR( const std::vector<double> &covByLag, int templateLength, int N,
                                long long totalSegLen )
{
    int dim = templateLength * N;
    std::vector<double> C( static_cast<size_t>( dim ) * dim, 0.0 );

    auto covAt = [&]( int lagIdx, int i, int k ) {
        return covByLag[( static_cast<size_t>( lagIdx ) * N + i ) * N + k] / static_cast<double>( totalSegLen );
    };

    for( int i = 0; i < N; ++i ) {
        for( int k = 0; k <= i; ++k ) {

            // cov_combined[lagIdx], lagIdx 0..nlags-1 <-> lag -maxlag..+maxlag.
            // row = cov_combined[:templateLength][::-1] (lags 0..-maxlag, reversed -> -maxlag..0 becomes 0..-maxlag order... )
            // col = cov_combined[templateLength-1:]      (lags 0..+maxlag)
            // Toeplitz(row, col): T[r][c] = row[r-c] if r>=c else col[c-r]
            // (scipy.linalg.toeplitz convention: first column = row(arg1), first row = col(arg2))
            for( int r = 0; r < templateLength; ++r ) {
                for( int c = 0; c < templateLength; ++c ) {
                    double val;
                    if( r >= c ) {
                        // row[r-c], row[j] = cov_combined[templateLength-1-j] (row is cov_combined[:templateLength] reversed)
                        int j = r - c;
                        val = covAt( templateLength - 1 - j, i, k );
                    }
                    else {
                        // col[c-r], col[j] = cov_combined[templateLength-1+j]
                        int j = c - r;
                        val = covAt( templateLength - 1 + j, i, k );
                    }
                    C[static_cast<size_t>( ( i * templateLength + r ) ) * dim + ( k * templateLength + c )] = val;
                    if( i != k )
                        C[static_cast<size_t>( ( k * templateLength + c ) ) * dim + ( i * templateLength + r )] = val;
                }
            }
        }
    }

    return C;
}

} // namespace


std::vector<std::vector<double> > computeNoiseCovarianceBatched(
    const float *dData, long long nSamples, int nChannelsGroup,
    int templateLength, int N,
    const std::vector<int> &unitChannels,
    const std::vector<SpikeFreeSegment> &segments,
    const std::vector<int> &segmentOffsets,
    const std::vector<int> &segmentCounts,
    int nUnits )
{
    (void)nSamples;

    const int maxlag = templateLength - 1;
    const int nlags  = 2 * maxlag + 1;

    // Per-segment per-channel means, hoisted out of the lag loop -- see
    // noiseCovarianceForLag's note.
    std::vector<std::vector<double> > segMeans( nUnits );
    for( int u = 0; u < nUnits; ++u ) {
        const int segN = segmentCounts[u];
        const int *chans = unitChannels.data() + static_cast<size_t>( u ) * N;
        segMeans[u].assign( static_cast<size_t>( segN ) * N, 0.0 );
        for( int s = 0; s < segN; ++s ) {
            const SpikeFreeSegment &seg = segments[segmentOffsets[u] + s];
            const int L = seg.end - seg.start;
            for( int c = 0; c < N; ++c ) {
                double sum = 0.0;
                const int ch = chans[c];
                for( int t = 0; t < L; ++t )
                    sum += dData[static_cast<size_t>( seg.start + t ) * nChannelsGroup + ch];
                segMeans[u][static_cast<size_t>( s ) * N + c] = sum / L;
            }
        }
    }

    std::vector<double> hostAcc(
        static_cast<size_t>( nUnits ) * nlags * N * N, 0.0 );

    // Flat (unit, lag) task list claimed from one atomic counter. Units have
    // very different segment counts, so a static split would leave one worker
    // holding every long unit.
    const long long nTasks = static_cast<long long>( nUnits ) * nlags;
    std::atomic<long long> next( 0 );

    int nThreads = static_cast<int>( std::thread::hardware_concurrency() );
    if( nThreads <= 0 ) nThreads = 1;
    if( static_cast<long long>( nThreads ) > nTasks )
        nThreads = static_cast<int>( nTasks );
    if( nThreads < 1 ) nThreads = 1;

    auto worker = [&]() {
        for( ;; ) {
            long long t = next.fetch_add( 1 );
            if( t >= nTasks )
                return;
            const int u       = static_cast<int>( t / nlags );
            const int lagIdx  = static_cast<int>( t % nlags );
            const int lag     = lagIdx - maxlag;
            noiseCovarianceForLag(
                dData, nChannelsGroup, templateLength, N,
                unitChannels.data() + static_cast<size_t>( u ) * N,
                segments.data() + segmentOffsets[u], segmentCounts[u],
                segMeans[u].data(), lag,
                hostAcc.data() +
                    ( static_cast<size_t>( u ) * nlags + lagIdx ) * N * N );
        }
    };

    std::vector<std::thread> pool;
    for( int i = 1; i < nThreads; ++i )
        pool.emplace_back( worker );
    worker();
    for( size_t i = 0; i < pool.size(); ++i )
        pool[i].join();

    std::vector<std::vector<double> > result( nUnits );
    for( int u = 0; u < nUnits; ++u ) {

        long long totalSegLen = 0;
        for( int s = 0; s < segmentCounts[u]; ++s ) {
            const SpikeFreeSegment &seg = segments[segmentOffsets[u] + s];
            totalSegLen += ( seg.end - seg.start );
        }
        if( totalSegLen == 0 )
            throw std::runtime_error( "computeNoiseCovarianceBatched: unit index " +
                std::to_string( u ) + " has no spike-free segments" );

        std::vector<double> covByLag(
            hostAcc.begin() + static_cast<long long>( u ) * nlags * N * N,
            hostAcc.begin() + static_cast<long long>( u + 1 ) * nlags * N * N );

        result[u] = assembleR( covByLag, templateLength, N, totalSegLen );
    }

    return result;
}
