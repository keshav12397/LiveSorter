#include "GpuConvolutionEngine.h"
#include "CudaUtil.h"

#include <climits>
#include <cuda_runtime.h>

namespace {

// Batched version of ConvolutionEngine.cpp's D[n] = sum_ch sum_k
// data[ch][n-leftMargin+k] * taps[k][ch] -- one block per unit. Each block
// first cooperatively gathers its unit's nChannelsPerUnit columns (via
// filterBank's translated channel indices) out of the full-group `combined`
// buffer into shared memory (small: (nSamples+L-1) x N floats), then every
// thread computes D for a strided subset of the chunk's nSamples output
// positions purely from shared memory.
//
// windowStart for output position idx works out to exactly idx (the
// leftMargin_ in "n = leftMargin+idx" and "windowStart = n-leftMargin"
// cancel identically) -- see ConvolutionEngine.cpp's derivation comment for
// why this centered-window formula is used verbatim rather than a
// causal-only reformulation.
__global__ void matchedFilterKernel(
    const float *combined, int nChannelsGroup, int nSamples, int L,
    const int *unitChannels, const float *filters, int N, int nUnits,
    float *dOut )
{
    extern __shared__ float strip[]; // (nSamples + L - 1) * N floats

    int unit = blockIdx.x;
    if( unit >= nUnits )
        return;

    const int   *chans = unitChannels + unit * N;
    const float *filt  = filters + static_cast<size_t>( unit ) * L * N;

    int stripRows = nSamples + L - 1;
    for( int idx = threadIdx.x; idx < stripRows * N; idx += blockDim.x ) {
        int row = idx / N;
        int c   = idx % N;
        int ch  = chans[c];
        strip[idx] = combined[static_cast<size_t>( row ) * nChannelsGroup + ch];
    }
    __syncthreads();

    for( int t = threadIdx.x; t < nSamples; t += blockDim.x ) {
        float sum = 0.0f;
        for( int k = 0; k < L; ++k ) {
            const float *stripRow = strip + static_cast<size_t>( t + k ) * N;
            const float *filtRow  = filt  + static_cast<size_t>( k ) * N;
            for( int c = 0; c < N; ++c )
                sum += stripRow[c] * filtRow[c];
        }
        dOut[static_cast<size_t>( unit ) * nSamples + t] = sum;
    }
}

// Batched version of ConvolutionEngine.cpp's windowed non-max suppression +
// threshold decision loop + dBuffer_ trim -- one block (single thread; this
// per-unit decision chain is inherently sequential via lastDecidedAbsIndex,
// and is short, ~nSamples iterations, so parallelizing it isn't worth the
// complexity here given the enormous compute margin at this problem size)
// per unit, operating on a persistent per-unit `tail` array standing in for
// the CPU's std::deque<double> dBuffer_ (same append / decide / trim
// structure, just a bounds-checked flat array instead of a deque).
__global__ void nmsThresholdKernel(
    const float *dNew, int nSamples, long long chunkAbsBase,
    int nUnits, long long minSep,
    float *dTail, int *dTailCount, long long *dTailStartAbsIndex,
    long long *lastDecidedAbsIndex, int dTailCap,
    const float *thresholds,
    int *detectionCount, GpuPeakEvent *detections, int detectionCapacity )
{
    if( threadIdx.x != 0 )
        return;
    int unit = blockIdx.x;
    if( unit >= nUnits )
        return;

    float *tail = dTail + static_cast<size_t>( unit ) * dTailCap;
    int       tailCount = dTailCount[unit];
    long long tailStart = dTailStartAbsIndex[unit];
    long long lastDecided = lastDecidedAbsIndex[unit];

    if( tailCount == 0 )
        tailStart = chunkAbsBase;

    for( int i = 0; i < nSamples; ++i ) {
        long long absIdx = chunkAbsBase + i;
        long long pos = absIdx - tailStart;
        if( pos >= 0 && pos < dTailCap ) {
            if( static_cast<int>( pos + 1 ) > tailCount )
                tailCount = static_cast<int>( pos + 1 );
            tail[pos] = dNew[static_cast<size_t>( unit ) * nSamples + i];
        }
        // else: dTailCap is sized generously at construction (maxChunkSamples
        // + 4*minSep) so this should never trigger -- if it somehow does,
        // the sample is dropped rather than corrupting adjacent memory.
    }

    long long newestBuffered = tailStart + tailCount - 1;

    long long n = lastDecided + 1;
    if( n < tailStart )
        n = tailStart;

    for( ; n + minSep <= newestBuffered; ++n ) {
        long long winLo = (tailStart > n - minSep) ? tailStart : (n - minSep);
        long long winHi = n + minSep;
        float dn = tail[n - tailStart];

        bool isPeak = true;
        for( long long m = winLo; m <= winHi && isPeak; ++m ) {
            if( m == n )
                continue;
            if( tail[m - tailStart] >= dn )
                isPeak = false;
        }

        if( isPeak && dn > thresholds[unit] ) {
            int slot = atomicAdd( detectionCount, 1 );
            if( slot < detectionCapacity ) {
                detections[slot].unitIndex   = unit;
                detections[slot].sampleIndex = n;
                detections[slot].score       = dn;
            }
        }
        lastDecided = n;
    }

    long long trimBefore = lastDecided - minSep;
    if( trimBefore > tailStart ) {
        long long shift = trimBefore - tailStart;
        if( shift >= tailCount ) {
            tailCount = 0;
        }
        else {
            for( long long i = 0; i < tailCount - shift; ++i )
                tail[i] = tail[i + shift];
            tailCount = static_cast<int>( tailCount - shift );
        }
        tailStart = trimBefore;
    }

    dTailCount[unit]          = tailCount;
    dTailStartAbsIndex[unit]  = tailStart;
    lastDecidedAbsIndex[unit] = lastDecided;
}

} // namespace


GpuConvolutionEngine::GpuConvolutionEngine( const GpuFilterBank &filterBank, int nChannelsGroup,
                                             int maxChunkSamples, long long minSeparationSamples )
    :   nUnits_( filterBank.nUnits ), nChannelsPerUnit_( filterBank.nChannelsPerUnit ),
        templateLength_( filterBank.templateLength ), nChannelsGroup_( nChannelsGroup ),
        maxChunkSamples_( maxChunkSamples ), filterBank_( filterBank ),
        d_combined_( nullptr ), d_dNew_( nullptr ),
        d_dTail_( nullptr ), d_dTailCount_( nullptr ),
        d_dTailStartAbsIndex_( nullptr ), d_lastDecidedAbsIndex_( nullptr ),
        d_detectionCount_( nullptr ), d_detections_( nullptr )
{
    leftMargin_  = (templateLength_ - 1) - (templateLength_ - 1) / 2;
    rightMargin_ = (templateLength_ - 1) / 2;
    minSeparationSamples_ = minSeparationSamples > 0 ? minSeparationSamples : templateLength_ / 2;

    // Shared-memory budget check: matchedFilterKernel needs
    // (maxChunkSamples + L - 1) * nChannelsPerUnit floats of shared memory
    // per block -- verify it fits before launching anything, rather than
    // failing opaquely inside the kernel launch.
    int device;
    CUDA_CHECK( cudaGetDevice( &device ) );
    cudaDeviceProp prop;
    CUDA_CHECK( cudaGetDeviceProperties( &prop, device ) );
    size_t neededShmem = static_cast<size_t>( maxChunkSamples_ + templateLength_ - 1 )
                        * nChannelsPerUnit_ * sizeof(float);
    if( neededShmem > prop.sharedMemPerBlock )
        throw std::runtime_error(
            "GpuConvolutionEngine: matchedFilterKernel's per-block shared "
            "memory requirement exceeds this GPU's limit -- reduce "
            "fetchChunkMs or n_channels" );

    size_t combinedCount = static_cast<size_t>( templateLength_ - 1 + maxChunkSamples_ ) * nChannelsGroup_;
    CUDA_CHECK( cudaMalloc( &d_combined_, combinedCount * sizeof(float) ) );
    CUDA_CHECK( cudaMemset( d_combined_, 0, combinedCount * sizeof(float) ) ); // zero history at startup -- see class comment

    CUDA_CHECK( cudaMalloc( &d_dNew_, static_cast<size_t>( nUnits_ ) * maxChunkSamples_ * sizeof(float) ) );

    dTailCap_ = maxChunkSamples_ + 4 * static_cast<int>( minSeparationSamples_ ) + 16;
    CUDA_CHECK( cudaMalloc( &d_dTail_, static_cast<size_t>( nUnits_ ) * dTailCap_ * sizeof(float) ) );
    CUDA_CHECK( cudaMalloc( &d_dTailCount_, static_cast<size_t>( nUnits_ ) * sizeof(int) ) );
    CUDA_CHECK( cudaMalloc( &d_dTailStartAbsIndex_, static_cast<size_t>( nUnits_ ) * sizeof(long long) ) );
    CUDA_CHECK( cudaMalloc( &d_lastDecidedAbsIndex_, static_cast<size_t>( nUnits_ ) * sizeof(long long) ) );
    CUDA_CHECK( cudaMemset( d_dTailCount_, 0, static_cast<size_t>( nUnits_ ) * sizeof(int) ) );

    // Sentinel far enough negative that the very first real peak is always
    // eligible -- same rationale as ConvolutionEngine::reset()'s
    // lastDecidedAbsIndex_ initialization.
    std::vector<long long> sentinel( nUnits_, LLONG_MIN / 2 );
    CUDA_CHECK( cudaMemcpy( d_lastDecidedAbsIndex_, sentinel.data(),
                             sentinel.size() * sizeof(long long), cudaMemcpyHostToDevice ) );

    detectionCapacity_ = 4096;
    CUDA_CHECK( cudaMalloc( &d_detectionCount_, sizeof(int) ) );
    CUDA_CHECK( cudaMalloc( &d_detections_, static_cast<size_t>( detectionCapacity_ ) * sizeof(GpuPeakEvent) ) );
    hostDetectionsBuf_.resize( detectionCapacity_ );
}


GpuConvolutionEngine::~GpuConvolutionEngine()
{
    if( d_combined_ )              cudaFree( d_combined_ );
    if( d_dNew_ )                  cudaFree( d_dNew_ );
    if( d_dTail_ )                 cudaFree( d_dTail_ );
    if( d_dTailCount_ )            cudaFree( d_dTailCount_ );
    if( d_dTailStartAbsIndex_ )    cudaFree( d_dTailStartAbsIndex_ );
    if( d_lastDecidedAbsIndex_ )   cudaFree( d_lastDecidedAbsIndex_ );
    if( d_detectionCount_ )        cudaFree( d_detectionCount_ );
    if( d_detections_ )            cudaFree( d_detections_ );
}


std::vector<GpuPeakEvent> GpuConvolutionEngine::processChunk(
    GpuPreprocessor &preprocessor, const short *d_raw, size_t nSamples,
    long long streamSampleOffset, void *streamVoid )
{
    cudaStream_t stream = static_cast<cudaStream_t>( streamVoid );

    if( nSamples == 0 )
        return {};
    if( static_cast<int>( nSamples ) > maxChunkSamples_ )
        throw std::runtime_error( "GpuConvolutionEngine::processChunk: nSamples exceeds maxChunkSamples "
                                   "this engine was constructed for" );

    // Preprocess directly into the tail of d_combined_ (right after the
    // retained history rows) -- no extra device-to-device copy.
    size_t historyOffsetFloats = static_cast<size_t>( templateLength_ - 1 ) * nChannelsGroup_;
    preprocessor.processChunk( d_raw, nSamples, d_combined_ + historyOffsetFloats, stream );

    int L = templateLength_;
    int N = nChannelsPerUnit_;
    int stripRows = static_cast<int>( nSamples ) + L - 1;
    size_t shmemBytes = static_cast<size_t>( stripRows ) * N * sizeof(float);
    int blockDim = 256;

    matchedFilterKernel<<<nUnits_, blockDim, shmemBytes, stream>>>(
        d_combined_, nChannelsGroup_, static_cast<int>( nSamples ), L,
        filterBank_.d_channels, filterBank_.d_filters, N, nUnits_, d_dNew_ );

    CUDA_CHECK( cudaMemsetAsync( d_detectionCount_, 0, sizeof(int), stream ) );

    // chunkAbsBase: see class .h comment / ConvolutionEngine.cpp's
    // derivation -- absIndex(idx) = streamSampleOffset - rightMargin_ + idx.
    long long chunkAbsBase = streamSampleOffset - rightMargin_;

    nmsThresholdKernel<<<nUnits_, 32, 0, stream>>>(
        d_dNew_, static_cast<int>( nSamples ), chunkAbsBase, nUnits_, minSeparationSamples_,
        d_dTail_, d_dTailCount_, d_dTailStartAbsIndex_, d_lastDecidedAbsIndex_, dTailCap_,
        filterBank_.d_thresholds,
        d_detectionCount_, d_detections_, detectionCapacity_ );

    // Carry the last (L-1) rows of this chunk's combined data forward as
    // history for the next call -- same overlap-save pattern as
    // ConvolutionEngine::processChunk's history_.assign(...) tail-keep.
    size_t totalRows = static_cast<size_t>( templateLength_ - 1 ) + nSamples;
    size_t keepRows  = static_cast<size_t>( templateLength_ - 1 );
    if( totalRows > keepRows ) {
        const float *tailSrc = d_combined_ + (totalRows - keepRows) * nChannelsGroup_;
        CUDA_CHECK( cudaMemcpyAsync( d_combined_, tailSrc, keepRows * nChannelsGroup_ * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream ) );
    }

    int detectionCount = 0;
    CUDA_CHECK( cudaMemcpyAsync( &detectionCount, d_detectionCount_, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream ) );
    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    if( detectionCount <= 0 )
        return {};
    if( detectionCount > detectionCapacity_ ) {
        // Overflowed the fixed detection buffer this chunk (should be very
        // rare -- would mean an implausibly large number of simultaneous
        // threshold crossings across all units). Clamp to what's actually
        // valid in the buffer rather than reading past it.
        detectionCount = detectionCapacity_;
    }

    CUDA_CHECK( cudaMemcpy( hostDetectionsBuf_.data(), d_detections_,
                             static_cast<size_t>( detectionCount ) * sizeof(GpuPeakEvent),
                             cudaMemcpyDeviceToHost ) );

    return std::vector<GpuPeakEvent>( hostDetectionsBuf_.begin(),
                                       hostDetectionsBuf_.begin() + detectionCount );
}
