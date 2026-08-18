#include "GpuPreprocessor.h"
#include "CudaUtil.h"

#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// One thread per channel (channels are independent), each sequentially
// recursing the same direct-form-I biquad ButterworthHighpass::processSample
// implements -- ported verbatim (see ButterworthHighpass.h), just in
// float32 and batched across channels instead of one C++ object per
// channel. Coalesced: at a fixed t, consecutive threads (consecutive ch)
// read/write consecutive addresses (raw/out are time-major/channel-minor).
__global__ void highpassKernel( const short *raw, float *out, int nSamples, int nChannels,
                                 float b0, float b1, float b2, float a1, float a2,
                                 float *state /* [nChannels*4]: x1,x2,y1,y2 */ )
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if( ch >= nChannels )
        return;

    float x1 = state[ch * 4 + 0];
    float x2 = state[ch * 4 + 1];
    float y1 = state[ch * 4 + 2];
    float y2 = state[ch * 4 + 3];

    for( int t = 0; t < nSamples; ++t ) {
        float x = static_cast<float>( raw[t * nChannels + ch] );
        float y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x;
        y2 = y1; y1 = y;
        out[t * nChannels + ch] = y;
    }

    state[ch * 4 + 0] = x1;
    state[ch * 4 + 1] = x2;
    state[ch * 4 + 2] = y1;
    state[ch * 4 + 3] = y2;
}

// Raw (non-highpassed) pass-through, when applyHighpass is false -- still
// needs int16->float32 conversion into the shared `out` buffer.
__global__ void castKernel( const short *raw, float *out, int nSamples, int nChannels )
{
    long long idx = static_cast<long long>( blockIdx.x ) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>( nSamples ) * nChannels;
    if( idx >= total )
        return;
    out[idx] = static_cast<float>( raw[idx] );
}

// One block per timestep, one thread per (padded) channel slot. Block-wide
// bitonic sort of that timestep's nChannels values (padded to the next
// power of 2 with +INF so padding never lands in the real, low half the
// median reads from), then every thread subtracts the median from ITS OWN
// original (pre-sort) value -- matches Preprocessor.h's medianInPlace()
// semantics exactly, including the even-count average-of-two-middle-values
// case (nth_element-equivalent, not e.g. a mean).
__global__ void carKernel( float *data, int nChannels, int paddedSize )
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float original = ( tid < nChannels ) ? data[blockIdx.x * nChannels + tid] : CUDART_INF_F;
    sdata[tid] = original;
    __syncthreads();

    for( int size = 2; size <= paddedSize; size <<= 1 ) {
        for( int stride = size >> 1; stride > 0; stride >>= 1 ) {
            int pos = tid ^ stride;
            if( pos > tid ) {
                bool ascending = ( (tid & size) == 0 );
                float a = sdata[tid];
                float b = sdata[pos];
                if( (a > b) == ascending ) {
                    sdata[tid] = b;
                    sdata[pos] = a;
                }
            }
            __syncthreads();
        }
    }

    float median;
    if( nChannels % 2 == 1 )
        median = sdata[nChannels / 2];
    else
        median = 0.5f * ( sdata[nChannels / 2 - 1] + sdata[nChannels / 2] );

    if( tid < nChannels )
        data[blockIdx.x * nChannels + tid] = original - median;
}

int nextPow2( int n )
{
    int p = 1;
    while( p < n )
        p <<= 1;
    return p;
}

} // namespace


GpuPreprocessor::GpuPreprocessor( int nChannels, double highpassCutoffHz, double sampleRateHz,
                                   bool applyHighpass, bool applyCar )
    :   nChannels_( nChannels ), applyHighpass_( applyHighpass ), applyCar_( applyCar ),
        d_biquadState_( nullptr )
{
    // Same RBJ Butterworth biquad coefficient derivation as
    // ButterworthHighpass.h's constructor -- computed once in double on the
    // host (cheap, one-shot), then stored as float32 for the kernel, so the
    // *coefficients* keep double-precision derivation accuracy even though
    // the per-sample recursion runs in float32.
    const double Q = 0.70710678118654752;
    const double w0 = 2.0 * M_PI * highpassCutoffHz / sampleRateHz;
    const double cosw0 = std::cos( w0 );
    const double alpha = std::sin( w0 ) / (2.0 * Q);
    double a0 = 1.0 + alpha;

    b0_ = static_cast<float>( ( (1.0 + cosw0) / 2.0 ) / a0 );
    b1_ = static_cast<float>( ( -(1.0 + cosw0) )      / a0 );
    b2_ = static_cast<float>( ( (1.0 + cosw0) / 2.0 ) / a0 );
    a1_ = static_cast<float>( ( -2.0 * cosw0 )        / a0 );
    a2_ = static_cast<float>( ( 1.0 - alpha )         / a0 );

    if( applyHighpass_ ) {
        CUDA_CHECK( cudaMalloc( &d_biquadState_, static_cast<size_t>( nChannels_ ) * 4 * sizeof(float) ) );
        CUDA_CHECK( cudaMemset( d_biquadState_, 0, static_cast<size_t>( nChannels_ ) * 4 * sizeof(float) ) );
    }
}


GpuPreprocessor::~GpuPreprocessor()
{
    if( d_biquadState_ )
        cudaFree( d_biquadState_ );
}


void GpuPreprocessor::processChunk( const short *d_raw, size_t nSamples, float *d_out, void *streamVoid )
{
    cudaStream_t stream = static_cast<cudaStream_t>( streamVoid );

    if( applyHighpass_ ) {
        int blockDim = 256;
        int gridDim  = (nChannels_ + blockDim - 1) / blockDim;
        highpassKernel<<<gridDim, blockDim, 0, stream>>>(
            d_raw, d_out, static_cast<int>( nSamples ), nChannels_,
            b0_, b1_, b2_, a1_, a2_, d_biquadState_ );
    }
    else {
        long long total = static_cast<long long>( nSamples ) * nChannels_;
        int blockDim = 256;
        long long gridDim = (total + blockDim - 1) / blockDim;
        castKernel<<<static_cast<unsigned int>( gridDim ), blockDim, 0, stream>>>(
            d_raw, d_out, static_cast<int>( nSamples ), nChannels_ );
    }

    if( applyCar_ ) {
        int paddedSize = nextPow2( nChannels_ );
        size_t shmemBytes = static_cast<size_t>( paddedSize ) * sizeof(float);
        carKernel<<<static_cast<unsigned int>( nSamples ), paddedSize, shmemBytes, stream>>>(
            d_out, nChannels_, paddedSize );
    }
}
