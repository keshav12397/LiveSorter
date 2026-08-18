#ifndef CLOSEDLOOP_CUDAUTIL_H
#define CLOSEDLOOP_CUDAUTIL_H

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

// Every CUDA Runtime call in this project's GPU pipeline is wrapped in this
// (not left unchecked) -- a silently-ignored cudaMalloc/cudaMemcpy failure
// would otherwise show up as a much more confusing downstream symptom
// (garbage detections, or a crash on first kernel launch far from the real
// cause). Throws std::runtime_error, consistent with the rest of this
// codebase's error handling (FilterBank, Config, etc.).
#define CUDA_CHECK( expr )                                                   \
    do {                                                                     \
        cudaError_t _cudaErr = (expr);                                       \
        if( _cudaErr != cudaSuccess ) {                                      \
            std::ostringstream _oss;                                        \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                 << cudaGetErrorString( _cudaErr );                          \
            throw std::runtime_error( _oss.str() );                         \
        }                                                                    \
    } while( 0 )

#endif // CLOSEDLOOP_CUDAUTIL_H
