#include "NoiseCovariance.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime.h>

#include "CudaUtil.h"

// Local accumulator array size cap -- this project always fits N=5
// (calibrate_all_units.py's --n-channels, fixed per unit by design, see its
// module docstring), 8 leaves headroom without risking local-memory
// spilling from an oversized per-thread array.
#define MAX_N 8


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

// One block per unit, one thread per lag (nlags = 2*(templateLength-1)+1,
// e.g. 121 for templateLength=61 -- comfortably under a 128/256 blockDim).
// Ports noise_covariance_vectorized's per-segment windowed cross-covariance
// accumulation verbatim (see that function's docstring for the o<->lag
// index-convention derivation this mirrors exactly, not re-derived here):
// for lag d, cross[i,k] = sum_t seg[t,i] * (seg[t-d,k] if 0<=t-d<L else 0).
__global__ void noiseCovarianceKernel(
    const float *data, long long dataStride /* nChannelsGroup */,
    int templateLength, int N,
    const int *unitChannels, // [nUnits * N]
    const SpikeFreeSegment *segments, const int *segOffset, const int *segCount,
    int nUnits, double *accOut /* [nUnits * nlags * N * N] */ )
{
    extern __shared__ double sMean[]; // [N]

    int unit = blockIdx.x;
    if( unit >= nUnits )
        return;

    int maxlag = templateLength - 1;
    int nlags = 2 * maxlag + 1;
    if( threadIdx.x >= nlags )
        return;
    int lag = threadIdx.x - maxlag;

    const int *chans = unitChannels + unit * N;
    int segOff = segOffset[unit];
    int segN   = segCount[unit];

    double localAcc[MAX_N * MAX_N];
    for( int i = 0; i < N * N; ++i )
        localAcc[i] = 0.0;

    for( int s = 0; s < segN; ++s ) {

        SpikeFreeSegment seg = segments[segOff + s];
        int L = seg.end - seg.start;

        // Per-channel mean over this segment, computed once (thread 0) and
        // broadcast via shared memory -- shared across all nlags threads
        // for this segment rather than each thread redundantly recomputing
        // it (mean doesn't depend on lag).
        if( threadIdx.x == 0 ) {
            for( int c = 0; c < N; ++c ) {
                double sum = 0.0;
                int ch = chans[c];
                for( int t = 0; t < L; ++t )
                    sum += data[static_cast<size_t>( seg.start + t ) * dataStride + ch];
                sMean[c] = sum / L;
            }
        }
        __syncthreads();

        for( int t = 0; t < L; ++t ) {
            int tk = t - lag;
            if( tk < 0 || tk >= L )
                continue; // zero-padded contribution, matches Python's padded window

            for( int i = 0; i < N; ++i ) {
                double vi = data[static_cast<size_t>( seg.start + t ) * dataStride + chans[i]] - sMean[i];
                for( int k = 0; k < N; ++k ) {
                    double vk = data[static_cast<size_t>( seg.start + tk ) * dataStride + chans[k]] - sMean[k];
                    localAcc[i * N + k] += vi * vk;
                }
            }
        }
        __syncthreads(); // all threads must finish this segment's work before thread 0 overwrites sMean for the next
    }

    double *out = accOut + ( static_cast<size_t>( unit ) * nlags + threadIdx.x ) * N * N;
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
    if( N > MAX_N )
        throw std::runtime_error( "computeNoiseCovarianceBatched: N exceeds MAX_N -- "
                                   "raise MAX_N in NoiseCovariance.cu" );

    int maxlag = templateLength - 1;
    int nlags = 2 * maxlag + 1;

    int *d_unitChannels = nullptr;
    SpikeFreeSegment *d_segments = nullptr;
    int *d_segOffset = nullptr;
    int *d_segCount = nullptr;
    double *d_acc = nullptr;

    CUDA_CHECK( cudaMalloc( &d_unitChannels, unitChannels.size() * sizeof(int) ) );
    CUDA_CHECK( cudaMemcpy( d_unitChannels, unitChannels.data(),
                             unitChannels.size() * sizeof(int), cudaMemcpyHostToDevice ) );

    CUDA_CHECK( cudaMalloc( &d_segments, std::max<size_t>( segments.size(), 1 ) * sizeof(SpikeFreeSegment) ) );
    if( !segments.empty() )
        CUDA_CHECK( cudaMemcpy( d_segments, segments.data(),
                                 segments.size() * sizeof(SpikeFreeSegment), cudaMemcpyHostToDevice ) );

    CUDA_CHECK( cudaMalloc( &d_segOffset, segmentOffsets.size() * sizeof(int) ) );
    CUDA_CHECK( cudaMemcpy( d_segOffset, segmentOffsets.data(),
                             segmentOffsets.size() * sizeof(int), cudaMemcpyHostToDevice ) );

    CUDA_CHECK( cudaMalloc( &d_segCount, segmentCounts.size() * sizeof(int) ) );
    CUDA_CHECK( cudaMemcpy( d_segCount, segmentCounts.data(),
                             segmentCounts.size() * sizeof(int), cudaMemcpyHostToDevice ) );

    size_t accCount = static_cast<size_t>( nUnits ) * nlags * N * N;
    CUDA_CHECK( cudaMalloc( &d_acc, accCount * sizeof(double) ) );

    int blockDim = nlags; // one thread per lag; nlags is small (e.g. 121), well under 1024
    size_t shmemBytes = static_cast<size_t>( N ) * sizeof(double);
    noiseCovarianceKernel<<<nUnits, blockDim, shmemBytes>>>(
        dData, nChannelsGroup, templateLength, N,
        d_unitChannels, d_segments, d_segOffset, d_segCount,
        nUnits, d_acc );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    std::vector<double> hostAcc( accCount );
    CUDA_CHECK( cudaMemcpy( hostAcc.data(), d_acc, accCount * sizeof(double), cudaMemcpyDeviceToHost ) );

    cudaFree( d_unitChannels );
    cudaFree( d_segments );
    cudaFree( d_segOffset );
    cudaFree( d_segCount );
    cudaFree( d_acc );

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
