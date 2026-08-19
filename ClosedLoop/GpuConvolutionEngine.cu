#include "GpuConvolutionEngine.h"
#include "CudaUtil.h"

#include <climits>
#include <cuda_runtime.h>

// Local candidate-array cap for nmsDecideKernel's single-threaded-per-block
// decision routine -- see the kernel's derivation comment. Must cover the
// worst case (every other buffered sample a local maximum) over the widest
// window the kernel ever examines (maxChunkSamples + finalizeMargin), so
// this bounds how large a chunk this engine can be constructed for; checked
// at construction time (throws rather than silently corrupting/overflowing
// a fixed-size local array).
#define MAX_CAND 4096

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

// Batched, GPU port of ConvolutionEngine::decideUpTo() -- see that
// function's derivation comment in ConvolutionEngine.cpp for the full
// story (candidates = strict local maxima, greedy tallest-first distance
// suppression among candidates only, matching scipy.signal.find_peaks
// exactly; a purely local "tallest in my window" check is provably wrong
// because a candidate's only threat can itself be eliminated by something
// even taller further out, needing the FULL buffered window -- not just
// cutoff+minSep -- to resolve correctly). One block per unit, single
// thread (this per-unit decision chain is inherently sequential, and is
// short given the enormous compute margin at this problem size, same
// rationale the original version of this kernel already used).
//
// Local scratch arrays are capped at MAX_CAND -- see that macro's comment.
// Runs with nSamples=0 and finalFlush=true for flush() (offline/finite-
// stream use only): decides everything currently buffered instead of
// holding the last finalizeMargin_ samples back forever waiting for future
// context that will never come in a live/continuous session.
__global__ void nmsDecideKernel(
    const float *dNew, int nSamples, long long chunkAbsBase,
    int nUnits, long long minSep, long long finalizeMargin, bool finalFlush,
    float *dTail, int *dTailCount, long long *dTailStartAbsIndex,
    long long *lastDecidedAbsIndex, int dTailCap,
    long long *keptPos, float *keptVal, int *keptCount, int keptCap,
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

    if( tailCount == 0 && nSamples > 0 )
        tailStart = chunkAbsBase;

    for( int i = 0; i < nSamples; ++i ) {
        long long absIdx = chunkAbsBase + i;
        long long pos = absIdx - tailStart;
        if( pos >= 0 && pos < dTailCap ) {
            if( static_cast<int>( pos + 1 ) > tailCount )
                tailCount = static_cast<int>( pos + 1 );
            tail[pos] = dNew[static_cast<size_t>( unit ) * nSamples + i];
        }
        // else: dTailCap is sized generously at construction so this should
        // never trigger -- if it somehow does, the sample is dropped rather
        // than corrupting adjacent memory.
    }

    if( tailCount == 0 ) {
        dTailCount[unit] = tailCount;
        dTailStartAbsIndex[unit] = tailStart;
        lastDecidedAbsIndex[unit] = lastDecided;
        return;
    }

    long long newestBuffered = tailStart + tailCount - 1;
    long long cutoff = finalFlush ? newestBuffered : ( newestBuffered - finalizeMargin );
    if( cutoff > newestBuffered )
        cutoff = newestBuffered;

    if( cutoff > lastDecided ) {

        long long *kPos = keptPos + static_cast<size_t>( unit ) * keptCap;
        float     *kVal = keptVal + static_cast<size_t>( unit ) * keptCap;
        int kCount = keptCount[unit];

        long long candPos[MAX_CAND];
        float     candVal[MAX_CAND];
        int nCand = 0;

        for( int i = 0; i < kCount && nCand < MAX_CAND; ++i ) {
            candPos[nCand] = kPos[i];
            candVal[nCand] = kVal[i];
            ++nCand;
        }

        long long hi = newestBuffered - 1;
        long long scanFrom = lastDecided + 1;
        if( scanFrom < tailStart + 1 )
            scanFrom = tailStart + 1;

        for( long long p = scanFrom; p <= hi && nCand < MAX_CAND; ++p ) {
            float a = tail[p - 1 - tailStart];
            float b = tail[p - tailStart];
            float c = tail[p + 1 - tailStart];
            if( a < b && b > c ) {
                candPos[nCand] = p;
                candVal[nCand] = b;
                ++nCand;
            }
        }

        // selectByDistance: repeatedly take the tallest not-yet-processed
        // surviving candidate and eliminate its still-alive neighbors
        // within `minSep` -- equivalent to scipy's tallest-first sweep
        // (see ConvolutionEngine.cpp's selectByDistance() for the same
        // algorithm expressed via an explicit sort instead).
        bool keep[MAX_CAND];
        bool processed[MAX_CAND];
        for( int i = 0; i < nCand; ++i ) { keep[i] = true; processed[i] = false; }

        for( int step = 0; step < nCand; ++step ) {
            int best = -1;
            float bestVal = -3.0e38f;
            for( int i = 0; i < nCand; ++i )
                if( !processed[i] && keep[i] && candVal[i] > bestVal ) { bestVal = candVal[i]; best = i; }
            if( best < 0 )
                break;
            processed[best] = true;
            for( int k = best - 1; k >= 0; --k ) {
                if( candPos[best] - candPos[k] < minSep ) keep[k] = false; else break;
            }
            for( int k = best + 1; k < nCand; ++k ) {
                if( candPos[k] - candPos[best] < minSep ) keep[k] = false; else break;
            }
        }

        long long newKeptPos[MAX_CAND];
        float     newKeptVal[MAX_CAND];
        int newKeptCount = 0;

        for( int i = 0; i < nCand; ++i ) {
            if( candPos[i] > lastDecided && candPos[i] <= cutoff && keep[i] ) {
                if( candVal[i] > thresholds[unit] ) {
                    int slot = atomicAdd( detectionCount, 1 );
                    if( slot < detectionCapacity ) {
                        detections[slot].unitIndex   = unit;
                        detections[slot].sampleIndex = candPos[i];
                        detections[slot].score       = candVal[i];
                    }
                }
                if( newKeptCount < MAX_CAND ) {
                    newKeptPos[newKeptCount] = candPos[i];
                    newKeptVal[newKeptCount] = candVal[i];
                    ++newKeptCount;
                }
            }
        }

        lastDecided = cutoff;
        long long trimBefore = lastDecided - minSep;

        // Rebuild the persisted kept-anchor list: still-relevant old
        // anchors (ascending) followed by newly kept ones (ascending) --
        // concatenation preserves ascending order since old anchors are
        // all <= the previous lastDecided < any new one.
        int mergedCount = 0;
        for( int i = 0; i < kCount && mergedCount < keptCap; ++i )
            if( kPos[i] >= trimBefore ) { kPos[mergedCount] = kPos[i]; kVal[mergedCount] = kVal[i]; ++mergedCount; }
        // (kPos/kVal are being overwritten in place while also being read
        // from -- safe because mergedCount <= i always here, so a write at
        // index mergedCount never clobbers an not-yet-read source index i.)
        for( int i = 0; i < newKeptCount && mergedCount < keptCap; ++i )
            if( newKeptPos[i] >= trimBefore ) { kPos[mergedCount] = newKeptPos[i]; kVal[mergedCount] = newKeptVal[i]; ++mergedCount; }
        keptCount[unit] = mergedCount;

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
    }

    dTailCount[unit]          = tailCount;
    dTailStartAbsIndex[unit]  = tailStart;
    lastDecidedAbsIndex[unit] = lastDecided;
}

} // namespace


GpuConvolutionEngine::GpuConvolutionEngine( const GpuFilterBank &filterBank, int nChannelsGroup,
                                             int maxChunkSamples, long long minSeparationSamples,
                                             int detectionCapacity )
    :   nUnits_( filterBank.nUnits ), nChannelsPerUnit_( filterBank.nChannelsPerUnit ),
        templateLength_( filterBank.templateLength ), nChannelsGroup_( nChannelsGroup ),
        maxChunkSamples_( maxChunkSamples ), filterBank_( filterBank ),
        d_combined_( nullptr ), d_historyScratch_( nullptr ), d_dNew_( nullptr ),
        d_dTail_( nullptr ), d_dTailCount_( nullptr ),
        d_dTailStartAbsIndex_( nullptr ), d_lastDecidedAbsIndex_( nullptr ),
        d_keptPos_( nullptr ), d_keptVal_( nullptr ), d_keptCount_( nullptr ),
        d_detectionCount_( nullptr ), d_detections_( nullptr )
{
    leftMargin_  = (templateLength_ - 1) - (templateLength_ - 1) / 2;
    rightMargin_ = (templateLength_ - 1) / 2;
    minSeparationSamples_ = minSeparationSamples > 0 ? minSeparationSamples : templateLength_ / 2;
    finalizeMarginSamples_ = 4 * minSeparationSamples_; // same choice as ConvolutionEngine.cpp, see its derivation comment

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

    CUDA_CHECK( cudaMalloc( &d_historyScratch_,
                             static_cast<size_t>( templateLength_ - 1 ) * nChannelsGroup_ * sizeof(float) ) );

    CUDA_CHECK( cudaMalloc( &d_dNew_, static_cast<size_t>( nUnits_ ) * maxChunkSamples_ * sizeof(float) ) );

    dTailCap_ = maxChunkSamples_ + 2 * static_cast<int>( finalizeMarginSamples_ ) + 16;
    if( dTailCap_ > MAX_CAND )
        throw std::runtime_error(
            "GpuConvolutionEngine: maxChunkSamples too large for nmsDecideKernel's "
            "fixed-size local candidate arrays (MAX_CAND=" + std::to_string( MAX_CAND ) +
            ") -- reduce fetchChunkMs/chunkSamples" );
    CUDA_CHECK( cudaMalloc( &d_dTail_, static_cast<size_t>( nUnits_ ) * dTailCap_ * sizeof(float) ) );
    CUDA_CHECK( cudaMalloc( &d_dTailCount_, static_cast<size_t>( nUnits_ ) * sizeof(int) ) );
    CUDA_CHECK( cudaMalloc( &d_dTailStartAbsIndex_, static_cast<size_t>( nUnits_ ) * sizeof(long long) ) );
    CUDA_CHECK( cudaMalloc( &d_lastDecidedAbsIndex_, static_cast<size_t>( nUnits_ ) * sizeof(long long) ) );
    CUDA_CHECK( cudaMemset( d_dTailCount_, 0, static_cast<size_t>( nUnits_ ) * sizeof(int) ) );

    keptCap_ = dTailCap_; // generous -- kept peaks are far sparser than raw samples
    CUDA_CHECK( cudaMalloc( &d_keptPos_, static_cast<size_t>( nUnits_ ) * keptCap_ * sizeof(long long) ) );
    CUDA_CHECK( cudaMalloc( &d_keptVal_, static_cast<size_t>( nUnits_ ) * keptCap_ * sizeof(float) ) );
    CUDA_CHECK( cudaMalloc( &d_keptCount_, static_cast<size_t>( nUnits_ ) * sizeof(int) ) );
    CUDA_CHECK( cudaMemset( d_keptCount_, 0, static_cast<size_t>( nUnits_ ) * sizeof(int) ) );

    // Sentinel far enough negative that the very first real peak is always
    // eligible -- same rationale as ConvolutionEngine::reset()'s
    // lastDecidedAbsIndex_ initialization.
    std::vector<long long> sentinel( nUnits_, LLONG_MIN / 2 );
    CUDA_CHECK( cudaMemcpy( d_lastDecidedAbsIndex_, sentinel.data(),
                             sentinel.size() * sizeof(long long), cudaMemcpyHostToDevice ) );

    detectionCapacity_ = detectionCapacity;
    CUDA_CHECK( cudaMalloc( &d_detectionCount_, sizeof(int) ) );
    CUDA_CHECK( cudaMalloc( &d_detections_, static_cast<size_t>( detectionCapacity_ ) * sizeof(GpuPeakEvent) ) );
    hostDetectionsBuf_.resize( detectionCapacity_ );
}


GpuConvolutionEngine::~GpuConvolutionEngine()
{
    if( d_combined_ )              cudaFree( d_combined_ );
    if( d_historyScratch_ )        cudaFree( d_historyScratch_ );
    if( d_dNew_ )                  cudaFree( d_dNew_ );
    if( d_dTail_ )                 cudaFree( d_dTail_ );
    if( d_dTailCount_ )            cudaFree( d_dTailCount_ );
    if( d_dTailStartAbsIndex_ )    cudaFree( d_dTailStartAbsIndex_ );
    if( d_lastDecidedAbsIndex_ )   cudaFree( d_lastDecidedAbsIndex_ );
    if( d_keptPos_ )               cudaFree( d_keptPos_ );
    if( d_keptVal_ )               cudaFree( d_keptVal_ );
    if( d_keptCount_ )             cudaFree( d_keptCount_ );
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

    nmsDecideKernel<<<nUnits_, 32, 0, stream>>>(
        d_dNew_, static_cast<int>( nSamples ), chunkAbsBase, nUnits_, minSeparationSamples_,
        finalizeMarginSamples_, /*finalFlush=*/false,
        d_dTail_, d_dTailCount_, d_dTailStartAbsIndex_, d_lastDecidedAbsIndex_, dTailCap_,
        d_keptPos_, d_keptVal_, d_keptCount_, keptCap_,
        filterBank_.d_thresholds,
        d_detectionCount_, d_detections_, detectionCapacity_ );

    // Carry the last (L-1) rows of this chunk's combined data forward as
    // history for the next call -- same overlap-save pattern as
    // ConvolutionEngine::processChunk's history_.assign(...) tail-keep.
    //
    // Staged through d_historyScratch_ rather than copied within
    // d_combined_ directly: source and destination OVERLAP whenever
    // nSamples < (L-1), and cudaMemcpy on overlapping device ranges is
    // undefined behavior exactly as memcpy's is (worse here -- the copy runs
    // as a parallel kernel, so a block can write a destination element that
    // another block has not read as its source yet, silently splicing
    // time-shifted signal into the history). Live fetches hit this
    // constantly: at fetchChunkMs=5 the loop consumes whatever SpikeGLX has
    // ready with no pacing, and on a real 3-minute run 31% of 84,111 chunks
    // came back with fewer than the L-1=60 samples needed to make the ranges
    // disjoint (1,059 of them were a single sample). No test exercised it:
    // the NMS equivalence tests run at chunkSamples=2000, OfflineScorer.cu
    // at 2000 or 150, all >= 60.
    size_t totalRows = static_cast<size_t>( templateLength_ - 1 ) + nSamples;
    size_t keepRows  = static_cast<size_t>( templateLength_ - 1 );
    if( totalRows > keepRows ) {
        const float *tailSrc = d_combined_ + (totalRows - keepRows) * nChannelsGroup_;
        size_t keepBytes = keepRows * nChannelsGroup_ * sizeof(float);
        CUDA_CHECK( cudaMemcpyAsync( d_historyScratch_, tailSrc, keepBytes,
                                      cudaMemcpyDeviceToDevice, stream ) );
        CUDA_CHECK( cudaMemcpyAsync( d_combined_, d_historyScratch_, keepBytes,
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


std::vector<GpuPeakEvent> GpuConvolutionEngine::flush( void *streamVoid )
{
    cudaStream_t stream = static_cast<cudaStream_t>( streamVoid );

    CUDA_CHECK( cudaMemsetAsync( d_detectionCount_, 0, sizeof(int), stream ) );

    nmsDecideKernel<<<nUnits_, 32, 0, stream>>>(
        d_dNew_, /*nSamples=*/0, /*chunkAbsBase=*/0, nUnits_, minSeparationSamples_,
        finalizeMarginSamples_, /*finalFlush=*/true,
        d_dTail_, d_dTailCount_, d_dTailStartAbsIndex_, d_lastDecidedAbsIndex_, dTailCap_,
        d_keptPos_, d_keptVal_, d_keptCount_, keptCap_,
        filterBank_.d_thresholds,
        d_detectionCount_, d_detections_, detectionCapacity_ );

    int detectionCount = 0;
    CUDA_CHECK( cudaMemcpyAsync( &detectionCount, d_detectionCount_, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream ) );
    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    if( detectionCount <= 0 )
        return {};
    if( detectionCount > detectionCapacity_ )
        detectionCount = detectionCapacity_;

    CUDA_CHECK( cudaMemcpy( hostDetectionsBuf_.data(), d_detections_,
                             static_cast<size_t>( detectionCount ) * sizeof(GpuPeakEvent),
                             cudaMemcpyDeviceToHost ) );

    return std::vector<GpuPeakEvent>( hostDetectionsBuf_.begin(),
                                       hostDetectionsBuf_.begin() + detectionCount );
}
