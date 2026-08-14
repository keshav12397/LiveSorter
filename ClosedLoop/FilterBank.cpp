#include "FilterBank.h"

#include <fstream>
#include <sstream>
#include <stdexcept>


namespace {

// Reads an entire binary file into a vector<T>, verifying the file size is
// an exact multiple of sizeof(T) (and, if expectedCount >= 0, that the
// count matches exactly -- catches a stale/mismatched filter directory
// early and loudly instead of silently reading garbage).
template<typename T>
std::vector<T> readBinFile( const std::string &path, long long expectedCount = -1 )
{
    std::ifstream fh( path.c_str(), std::ios::binary | std::ios::ate );
    if( !fh.is_open() )
        throw std::runtime_error( "FilterBank: could not open '" + path + "'" );

    std::streamsize bytes = fh.tellg();
    fh.seekg( 0, std::ios::beg );

    if( bytes % sizeof(T) != 0 ) {
        std::ostringstream oss;
        oss << "FilterBank: '" << path << "' size (" << bytes
            << " bytes) is not a multiple of " << sizeof(T);
        throw std::runtime_error( oss.str() );
    }

    long long count = bytes / static_cast<long long>( sizeof(T) );

    if( expectedCount >= 0 && count != expectedCount ) {
        std::ostringstream oss;
        oss << "FilterBank: '" << path << "' has " << count
            << " elements, expected " << expectedCount;
        throw std::runtime_error( oss.str() );
    }

    std::vector<T> data( static_cast<size_t>( count ) );
    if( count > 0 )
        fh.read( reinterpret_cast<char*>( &data[0] ), bytes );

    return data;
}

} // namespace


FilterBank FilterBank::load( const std::string &filterDir, int targetId, int templateLength )
{
    FilterBank fb;
    fb.targetId       = targetId;
    fb.templateLength = templateLength;

    std::ostringstream idStr;
    idStr << targetId;

    std::string channelsPath  = filterDir + "/channels_"  + idStr.str() + ".bin";
    std::string filterPath    = filterDir + "/filter_"    + idStr.str() + ".bin";
    std::string thresholdPath = filterDir + "/threshold_" + idStr.str() + ".bin";

    std::vector<int32_t> rawChannels = readBinFile<int32_t>( channelsPath );
    fb.channels.assign( rawChannels.begin(), rawChannels.end() );

    long long expectedTaps = static_cast<long long>( templateLength ) * fb.nChannels();
    fb.taps = readBinFile<double>( filterPath, expectedTaps );

    std::vector<double> th = readBinFile<double>( thresholdPath, 1 );
    fb.threshold = th[0];

    return fb;
}


void FilterBank::saveThreshold( const std::string &filterDir ) const
{
    std::ostringstream idStr;
    idStr << targetId;
    std::string path = filterDir + "/threshold_" + idStr.str() + ".bin";

    std::ofstream fh( path.c_str(), std::ios::binary | std::ios::trunc );
    if( !fh.is_open() )
        throw std::runtime_error( "FilterBank::saveThreshold: could not open '" + path + "' for writing" );

    fh.write( reinterpret_cast<const char*>( &threshold ), sizeof(threshold) );
}
