#ifndef CLOSEDLOOP_SCRATCHMEMMAP_H
#define CLOSEDLOOP_SCRATCHMEMMAP_H

#include <string>
#include <stdexcept>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX // windows.h's max()/min() macros clobber std::max/std::min and numeric_limits<T>::max() -- must be defined before the include
#endif
#include <windows.h>
#endif

// Read-only memory-mapped view of the float32 scratch file
// OfflinePreprocessor.cu writes -- (nSamples, nChannels) row-major/C order.
// Backed by the OS page cache (Windows CreateFileMapping/MapViewOfFile), not
// a RAM-resident copy -- matters for the same reason
// FilterGen/threshold_sweep_real.py's scratch memmap does this instead of
// loading the whole preprocessed recording into RAM: total size is bounded
// by disk space, not RAM, on a machine that might have far less RAM than
// the recording is large.
class ScratchMemmap {
public:
    ScratchMemmap( const std::string &path, long long nSamples, int nChannels )
        : nSamples_( nSamples ), nChannels_( nChannels ),
          hFile_( INVALID_HANDLE_VALUE ), hMap_( nullptr ), data_( nullptr )
    {
#ifdef _WIN32
        hFile_ = CreateFileA( path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr );
        if( hFile_ == INVALID_HANDLE_VALUE )
            throw std::runtime_error( "ScratchMemmap: could not open '" + path + "'" );

        hMap_ = CreateFileMappingA( hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr );
        if( hMap_ == nullptr ) {
            CloseHandle( hFile_ );
            throw std::runtime_error( "ScratchMemmap: CreateFileMapping failed for '" + path + "'" );
        }

        data_ = static_cast<const float*>( MapViewOfFile( hMap_, FILE_MAP_READ, 0, 0, 0 ) );
        if( data_ == nullptr ) {
            CloseHandle( hMap_ );
            CloseHandle( hFile_ );
            throw std::runtime_error( "ScratchMemmap: MapViewOfFile failed for '" + path + "'" );
        }
#else
        throw std::runtime_error( "ScratchMemmap: only implemented for Windows in this project" );
#endif
    }

    ~ScratchMemmap()
    {
#ifdef _WIN32
        if( data_ )   UnmapViewOfFile( data_ );
        if( hMap_ )   CloseHandle( hMap_ );
        if( hFile_ != INVALID_HANDLE_VALUE ) CloseHandle( hFile_ );
#endif
    }

    ScratchMemmap( const ScratchMemmap & ) = delete;
    ScratchMemmap &operator=( const ScratchMemmap & ) = delete;

    long long nSamples()  const { return nSamples_; }
    int       nChannels() const { return nChannels_; }

    // Row-major element access: sample s (0..nSamples-1), channel c.
    float at( long long s, int c ) const
    {
        return data_[static_cast<size_t>( s ) * nChannels_ + c];
    }

    const float *row( long long s ) const
    {
        return data_ + static_cast<size_t>( s ) * nChannels_;
    }

private:
    long long nSamples_;
    int nChannels_;
#ifdef _WIN32
    HANDLE hFile_;
    HANDLE hMap_;
#else
    void *hFile_;
    void *hMap_;
#endif
    const float *data_;
};

#endif // CLOSEDLOOP_SCRATCHMEMMAP_H
