#ifndef CLOSEDLOOP_NPYREADER_H
#define CLOSEDLOOP_NPYREADER_H

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>

// Minimal, single-header reader for simple 1-D .npy arrays -- just enough
// to load Kilosort's spike_times.npy / spike_clusters.npy for Calibration's
// ground-truth matching. Deliberately NOT a general NPY parser: assumes
// little-endian, C-contiguous ('fortran_order': False), 1-D, and one of
// the common Kilosort integer dtypes (<i8, <i4, <u8, <u4). Throws
// std::runtime_error on anything else, rather than silently misreading.
//
// NPY format reminder: 6-byte magic "\x93NUMPY", 2 version bytes, a
// header-length field (2 bytes for v1.0, 4 bytes for v2.0+), then an ASCII
// dict-literal header (padded so the whole preamble is a multiple of 64
// bytes) ending in '\n', then raw data.
inline std::vector<long long> readNpy1D( const std::string &path )
{
    std::ifstream fh( path.c_str(), std::ios::binary );
    if( !fh.is_open() )
        throw std::runtime_error( "readNpy1D: could not open '" + path + "'" );

    char magic[6];
    fh.read( magic, 6 );
    if( std::memcmp( magic, "\x93NUMPY", 6 ) != 0 )
        throw std::runtime_error( "readNpy1D: '" + path + "' is not a .npy file" );

    unsigned char verMajor, verMinor;
    fh.read( reinterpret_cast<char*>( &verMajor ), 1 );
    fh.read( reinterpret_cast<char*>( &verMinor ), 1 );

    unsigned int headerLen = 0;
    if( verMajor == 1 ) {
        unsigned short hlen16;
        fh.read( reinterpret_cast<char*>( &hlen16 ), 2 );
        headerLen = hlen16;
    }
    else {
        unsigned int hlen32;
        fh.read( reinterpret_cast<char*>( &hlen32 ), 4 );
        headerLen = hlen32;
    }

    std::vector<char> headerBuf( headerLen );
    fh.read( &headerBuf[0], headerLen );
    std::string header( headerBuf.begin(), headerBuf.end() );

    if( header.find( "'fortran_order': False" ) == std::string::npos &&
        header.find( "\"fortran_order\": False" ) == std::string::npos ) {
        throw std::runtime_error( "readNpy1D: '" + path + "' is fortran-ordered; not supported" );
    }

    // Extract dtype string, e.g. "'descr': '<i8'"
    size_t descrPos = header.find( "descr" );
    if( descrPos == std::string::npos )
        throw std::runtime_error( "readNpy1D: no 'descr' field in '" + path + "'" );
    size_t q1 = header.find( '\'', descrPos + 5 );
    q1 = header.find( '\'', q1 + 1 ); // skip the "descr" key's own quotes
    size_t q2 = header.find( '\'', q1 + 1 );
    std::string dtype = header.substr( q1 + 1, q2 - q1 - 1 );

    // Extract shape tuple, e.g. "'shape': (9320648,)" -- take first int only
    // (this reader only supports 1-D arrays).
    size_t shapePos = header.find( "shape" );
    size_t parenOpen = header.find( '(', shapePos );
    size_t firstDigit = header.find_first_of( "0123456789", parenOpen );
    size_t firstNonDigit = header.find_first_not_of( "0123456789", firstDigit );
    long long n = std::atoll( header.substr( firstDigit, firstNonDigit - firstDigit ).c_str() );

    std::vector<long long> out;
    out.reserve( static_cast<size_t>( n ) );

    if( dtype == "<i8" || dtype == "<u8" ) {
        std::vector<int64_t> buf( static_cast<size_t>( n ) );
        if( n > 0 )
            fh.read( reinterpret_cast<char*>( &buf[0] ), n * sizeof(int64_t) );
        out.assign( buf.begin(), buf.end() );
    }
    else if( dtype == "<i4" || dtype == "<u4" ) {
        std::vector<int32_t> buf( static_cast<size_t>( n ) );
        if( n > 0 )
            fh.read( reinterpret_cast<char*>( &buf[0] ), n * sizeof(int32_t) );
        out.assign( buf.begin(), buf.end() );
    }
    else {
        throw std::runtime_error( "readNpy1D: unsupported dtype '" + dtype +
                                   "' in '" + path + "' (expected <i8/<u8/<i4/<u4)" );
    }

    return out;
}

#endif // CLOSEDLOOP_NPYREADER_H
