#include "MultiFilterBank.h"

#include <algorithm>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

// Same "read whole file, verify exact expected element count" pattern
// FilterBank.cpp uses. Kept local for the same
// reason they kept theirs local: it is ~15 lines, and the alternative is a
// shared template header that all three must agree on forever.
template<typename T>
std::vector<T> readBinFileExact( const std::string &path, long long expectedCount )
{
    std::ifstream fh( path.c_str(), std::ios::binary | std::ios::ate );
    if( !fh.is_open() )
        throw std::runtime_error( "MultiFilterBank: could not open '" + path + "'" );

    std::streamsize bytes = fh.tellg();
    fh.seekg( 0, std::ios::beg );

    long long count = bytes / static_cast<long long>( sizeof(T) );
    if( bytes % static_cast<long long>( sizeof(T) ) != 0 || count != expectedCount ) {
        std::ostringstream oss;
        oss << "MultiFilterBank: '" << path << "' has " << bytes
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
// learn nUnits, then check every other file against that.
std::vector<int32_t> readUnitIds( const std::string &path )
{
    std::ifstream fh( path.c_str(), std::ios::binary | std::ios::ate );
    if( !fh.is_open() )
        throw std::runtime_error( "MultiFilterBank: could not open '" + path + "'" );
    std::streamsize bytes = fh.tellg();
    fh.seekg( 0, std::ios::beg );
    if( bytes % static_cast<std::streamsize>( sizeof(int32_t) ) != 0 )
        throw std::runtime_error( "MultiFilterBank: '" + path + "' size is not a multiple of 4" );
    std::vector<int32_t> data(
        static_cast<size_t>( bytes / static_cast<std::streamsize>( sizeof(int32_t) ) ) );
    if( !data.empty() )
        fh.read( reinterpret_cast<char*>( &data[0] ), bytes );
    return data;
}

} // namespace


MultiFilterBank MultiFilterBank::load( const std::string &dir, int nChannelsPerUnit,
                                       int templateLength )
{
    MultiFilterBank fb;
    fb.nChannelsPerUnit = nChannelsPerUnit;
    fb.templateLength   = templateLength;

    std::vector<int32_t> unitIds = readUnitIds( dir + "/unit_ids.bin" );
    fb.nUnits = static_cast<int>( unitIds.size() );
    if( fb.nUnits == 0 )
        throw std::runtime_error( "MultiFilterBank: '" + dir + "/unit_ids.bin' is empty" );
    fb.hostUnitIds.assign( unitIds.begin(), unitIds.end() );

    const long long n  = fb.nUnits;
    const long long nc = static_cast<long long>( nChannelsPerUnit );
    const long long L  = static_cast<long long>( templateLength );

    fb.channels   = readBinFileExact<int32_t>( dir + "/channels.bin",   n * nc );
    fb.filters    = readBinFileExact<float>(   dir + "/filters.bin",    n * L * nc );
    fb.thresholds = readBinFileExact<float>(   dir + "/thresholds.bin", n );

    return fb;
}


MultiFilterBank MultiFilterBank::fromHostArrays( const std::vector<int> &unitIds,
                                                 const std::vector<int32_t> &channels,
                                                 const std::vector<float> &filters,
                                                 const std::vector<float> &thresholds,
                                                 int nChannelsPerUnit, int templateLength )
{
    MultiFilterBank fb;
    fb.nUnits           = static_cast<int>( unitIds.size() );
    fb.nChannelsPerUnit = nChannelsPerUnit;
    fb.templateLength   = templateLength;

    const size_t n = unitIds.size();
    if( channels.size() != n * static_cast<size_t>( nChannelsPerUnit ) ||
        filters.size()  != n * static_cast<size_t>( templateLength ) *
                               static_cast<size_t>( nChannelsPerUnit ) ||
        thresholds.size() != n ) {
        throw std::runtime_error(
            "MultiFilterBank::fromHostArrays: array sizes are inconsistent with "
            "nUnits/nChannelsPerUnit/templateLength" );
    }

    fb.hostUnitIds = unitIds;
    fb.channels    = channels;
    fb.filters     = filters;
    fb.thresholds  = thresholds;
    return fb;
}


void MultiFilterBank::updateFilters( int unitIndex, const std::vector<int32_t> &newChannels,
                                     const std::vector<float> &filter, float threshold )
{
    if( unitIndex < 0 || unitIndex >= nUnits )
        throw std::runtime_error( "MultiFilterBank::updateFilters: unitIndex out of range" );

    if( newChannels.size() != static_cast<size_t>( nChannelsPerUnit ) ||
        filter.size() != static_cast<size_t>( templateLength ) *
                         static_cast<size_t>( nChannelsPerUnit ) ) {
        throw std::runtime_error(
            "MultiFilterBank::updateFilters: array size mismatch for this bank's "
            "nChannelsPerUnit/templateLength -- a drift swap event must be exactly "
            "one unit's slice of the existing layout, since nothing is reallocated" );
    }

    std::copy( newChannels.begin(), newChannels.end(),
               channels.begin() + static_cast<size_t>( unitIndex ) * nChannelsPerUnit );
    std::copy( filter.begin(), filter.end(),
               filters.begin() + static_cast<size_t>( unitIndex ) * templateLength * nChannelsPerUnit );
    thresholds[static_cast<size_t>( unitIndex )] = threshold;
}


int MultiFilterBank::carGroupIndexOf( const std::vector<int> &carChannelIds,
                                       int32_t rawId )
{
    std::vector<int>::const_iterator it =
        std::find( carChannelIds.begin(), carChannelIds.end(), rawId );
    if( it == carChannelIds.end() )
        return -1;
    return static_cast<int>( it - carChannelIds.begin() );
}


bool MultiFilterBank::translateChannelsToCarGroup(
    const std::vector<int> &carChannelIds, int *badUnit, int32_t *badChannel )
{
    const int N = nChannelsPerUnit;
    for( size_t i = 0; i < channels.size(); ++i ) {
        int idx = carGroupIndexOf( carChannelIds, channels[i] );
        if( idx < 0 ) {
            if( badUnit )    *badUnit = static_cast<int>( i / ( N > 0 ? N : 1 ) );
            if( badChannel ) *badChannel = channels[i];
            return false;
        }
        channels[i] = static_cast<int32_t>( idx );
    }
    return true;
}
