#include "GpuFilterBank.h"
#include "CudaUtil.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

// Same "read whole file, verify exact expected element count" pattern as
// FilterBank.cpp's readBinFile -- kept independent (not shared) since this
// one is deliberately host-buffer-then-upload rather than FilterBank's
// keep-in-a-std::vector, and duplicating ~15 lines is cheaper than forcing a
// shared template across a CUDA/non-CUDA translation-unit boundary.
template<typename T>
std::vector<T> readBinFileExact( const std::string &path, long long expectedCount )
{
    std::ifstream fh( path.c_str(), std::ios::binary | std::ios::ate );
    if( !fh.is_open() )
        throw std::runtime_error( "GpuFilterBank: could not open '" + path + "'" );

    std::streamsize bytes = fh.tellg();
    fh.seekg( 0, std::ios::beg );

    long long count = bytes / static_cast<long long>( sizeof(T) );
    if( bytes % static_cast<long long>( sizeof(T) ) != 0 || count != expectedCount ) {
        std::ostringstream oss;
        oss << "GpuFilterBank: '" << path << "' has " << bytes
            << " bytes (" << count << " elements), expected exactly "
            << expectedCount << " elements ("
            << (expectedCount * static_cast<long long>( sizeof(T) )) << " bytes) -- "
            << "does this filterDir match the --n-channels/--template-length "
            << "calibrate_all_units.py was run with?";
        throw std::runtime_error( oss.str() );
    }

    std::vector<T> data( static_cast<size_t>( count ) );
    if( count > 0 )
        fh.read( reinterpret_cast<char*>( &data[0] ), bytes );
    return data;
}

// unit_ids.bin's element count IS nUnits -- read it first, un-sized, to
// learn nUnits, then every other file is checked against that.
std::vector<int32_t> readUnitIds( const std::string &path )
{
    std::ifstream fh( path.c_str(), std::ios::binary | std::ios::ate );
    if( !fh.is_open() )
        throw std::runtime_error( "GpuFilterBank: could not open '" + path + "'" );
    std::streamsize bytes = fh.tellg();
    fh.seekg( 0, std::ios::beg );
    if( bytes % static_cast<std::streamsize>( sizeof(int32_t) ) != 0 )
        throw std::runtime_error( "GpuFilterBank: '" + path + "' size is not a multiple of 4" );
    std::vector<int32_t> data( static_cast<size_t>( bytes / static_cast<std::streamsize>( sizeof(int32_t) ) ) );
    if( !data.empty() )
        fh.read( reinterpret_cast<char*>( &data[0] ), bytes );
    return data;
}

} // namespace


GpuFilterBank::GpuFilterBank()
    :   nUnits( 0 ), nChannelsPerUnit( 0 ), templateLength( 0 ),
        d_channels( nullptr ), d_filters( nullptr ), d_thresholds( nullptr )
{}


GpuFilterBank GpuFilterBank::load( const std::string &dir, int nChannelsPerUnit, int templateLength )
{
    GpuFilterBank fb;
    fb.nChannelsPerUnit = nChannelsPerUnit;
    fb.templateLength   = templateLength;

    std::vector<int32_t> unitIds = readUnitIds( dir + "/unit_ids.bin" );
    fb.nUnits = static_cast<int>( unitIds.size() );
    if( fb.nUnits == 0 )
        throw std::runtime_error( "GpuFilterBank: '" + dir + "/unit_ids.bin' is empty" );
    fb.hostUnitIds.assign( unitIds.begin(), unitIds.end() );

    long long nUnitsL = fb.nUnits;
    std::vector<int32_t> channels = readBinFileExact<int32_t>(
        dir + "/channels.bin", nUnitsL * nChannelsPerUnit );
    std::vector<float> filters = readBinFileExact<float>(
        dir + "/filters.bin", nUnitsL * templateLength * nChannelsPerUnit );
    std::vector<float> thresholds = readBinFileExact<float>(
        dir + "/thresholds.bin", nUnitsL );

    CUDA_CHECK( cudaMalloc( &fb.d_channels, channels.size() * sizeof(int32_t) ) );
    CUDA_CHECK( cudaMalloc( &fb.d_filters, filters.size() * sizeof(float) ) );
    CUDA_CHECK( cudaMalloc( &fb.d_thresholds, thresholds.size() * sizeof(float) ) );

    CUDA_CHECK( cudaMemcpy( fb.d_channels, channels.data(),
                             channels.size() * sizeof(int32_t), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( fb.d_filters, filters.data(),
                             filters.size() * sizeof(float), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( fb.d_thresholds, thresholds.data(),
                             thresholds.size() * sizeof(float), cudaMemcpyHostToDevice ) );

    return fb;
}


void GpuFilterBank::release()
{
    if( d_channels )   { cudaFree( d_channels );   d_channels   = nullptr; }
    if( d_filters )    { cudaFree( d_filters );    d_filters    = nullptr; }
    if( d_thresholds ) { cudaFree( d_thresholds ); d_thresholds = nullptr; }
}


GpuFilterBank::~GpuFilterBank()
{
    release();
}


GpuFilterBank::GpuFilterBank( GpuFilterBank &&other ) noexcept
    :   nUnits( other.nUnits ), nChannelsPerUnit( other.nChannelsPerUnit ),
        templateLength( other.templateLength ),
        d_channels( other.d_channels ), d_filters( other.d_filters ),
        d_thresholds( other.d_thresholds ), hostUnitIds( std::move( other.hostUnitIds ) )
{
    other.d_channels = nullptr;
    other.d_filters = nullptr;
    other.d_thresholds = nullptr;
}


GpuFilterBank &GpuFilterBank::operator=( GpuFilterBank &&other ) noexcept
{
    if( this != &other ) {
        release();
        nUnits = other.nUnits;
        nChannelsPerUnit = other.nChannelsPerUnit;
        templateLength = other.templateLength;
        d_channels = other.d_channels;
        d_filters = other.d_filters;
        d_thresholds = other.d_thresholds;
        hostUnitIds = std::move( other.hostUnitIds );
        other.d_channels = nullptr;
        other.d_filters = nullptr;
        other.d_thresholds = nullptr;
    }
    return *this;
}
