#include "SglxMetaReader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>


std::map<std::string, std::string> parseMetaFile( const std::string &metaPath )
{
    std::ifstream fh( metaPath.c_str() );
    if( !fh.is_open() )
        throw std::runtime_error( "parseMetaFile: could not open '" + metaPath + "'" );

    std::map<std::string, std::string> meta;
    std::string line;

    while( std::getline( fh, line ) ) {

        if( !line.empty() && line[line.size() - 1] == '\r' )
            line.erase( line.size() - 1 );

        size_t eq = line.find( '=' );
        if( eq == std::string::npos )
            continue;

        std::string key = line.substr( 0, eq );
        std::string val = line.substr( eq + 1 );

        if( !key.empty() && key[0] == '~' )
            key = key.substr( 1 );

        meta[key] = val;
    }

    return meta;
}


std::vector<int> parseChanSubset( const std::string &spec )
{
    std::vector<int>   out;
    std::stringstream  ss( spec );
    std::string        part;

    while( std::getline( ss, part, ',' ) ) {

        size_t colon = part.find( ':' );

        if( colon != std::string::npos ) {
            int a = std::atoi( part.substr( 0, colon ).c_str() );
            int b = std::atoi( part.substr( colon + 1 ).c_str() );
            for( int i = a; i <= b; ++i )
                out.push_back( i );
        }
        else if( !part.empty() )
            out.push_back( std::atoi( part.c_str() ) );
    }

    return out;
}


std::vector<int> loadChanMapJson( const std::string &path )
{
    std::ifstream fh( path.c_str() );
    if( !fh.is_open() )
        throw std::runtime_error( "loadChanMapJson: could not open '" + path + "'" );

    std::stringstream buf;
    buf << fh.rdbuf();
    std::string content = buf.str();

    size_t key = content.find( "\"chanMap\"" );
    if( key == std::string::npos )
        throw std::runtime_error( "loadChanMapJson: no 'chanMap' field in '" + path + "'" );

    size_t open = content.find( '[', key );
    size_t close = content.find( ']', open );
    if( open == std::string::npos || close == std::string::npos )
        throw std::runtime_error( "loadChanMapJson: malformed 'chanMap' array in '" + path + "'" );

    std::string arr = content.substr( open + 1, close - open - 1 );
    std::vector<int> out;
    std::stringstream ss( arr );
    std::string tok;
    while( std::getline( ss, tok, ',' ) ) {
        if( !tok.empty() )
            out.push_back( std::atoi( tok.c_str() ) );
    }

    return out;
}


namespace {

// Shared by loadChanMapPositions(): extracts the bracketed numeric array
// following a top-level "<key>" field, e.g. "\"xc\": [1.0, 2.5, ...]".
// Returns false (rather than throwing) if the field isn't present -- xc/yc
// are optional, matching generate_filter.load_channel_positions_json()'s
// "returns {} if missing" behavior.
bool extractNumericArray( const std::string &content, const std::string &key,
                           std::vector<double> &out )
{
    std::string quoted = "\"" + key + "\"";
    size_t pos = content.find( quoted );
    if( pos == std::string::npos )
        return false;

    size_t open = content.find( '[', pos );
    size_t close = content.find( ']', open );
    if( open == std::string::npos || close == std::string::npos )
        throw std::runtime_error( "loadChanMapPositions: malformed '" + key + "' array" );

    std::string arr = content.substr( open + 1, close - open - 1 );
    std::stringstream ss( arr );
    std::string tok;
    out.clear();
    while( std::getline( ss, tok, ',' ) ) {
        if( !tok.empty() )
            out.push_back( std::atof( tok.c_str() ) );
    }
    return true;
}

} // namespace


std::map<int, std::pair<double, double> > loadChanMapPositions( const std::string &path )
{
    std::ifstream fh( path.c_str() );
    if( !fh.is_open() )
        throw std::runtime_error( "loadChanMapPositions: could not open '" + path + "'" );

    std::stringstream buf;
    buf << fh.rdbuf();
    std::string content = buf.str();

    std::vector<double> chanMap, xc, yc;
    if( !extractNumericArray( content, "chanMap", chanMap ) )
        throw std::runtime_error( "loadChanMapPositions: no 'chanMap' field in '" + path + "'" );

    std::map<int, std::pair<double, double> > out;
    if( !extractNumericArray( content, "xc", xc ) || !extractNumericArray( content, "yc", yc ) )
        return out; // no positions available -- caller falls back to index-distance

    size_t n = std::min( chanMap.size(), std::min( xc.size(), yc.size() ) );
    for( size_t i = 0; i < n; ++i )
        out[static_cast<int>( chanMap[i] )] = std::make_pair( xc[i], yc[i] );

    return out;
}


SglxMeta SglxMeta::load( const std::string &metaPath )
{
    std::map<std::string, std::string> meta = parseMetaFile( metaPath );

    SglxMeta m;
    m.sampleRateHz  = std::atof( meta["imSampRate"].c_str() );
    m.nSavedChans   = std::atoi( meta["nSavedChans"].c_str() );
    m.savedChannelIds = parseChanSubset( meta["snsSaveChanSubset"] );

    // snsApLfSy = "<ap>,<lf>,<sy>" -- last field is the sync-channel count.
    std::string apLfSy = meta["snsApLfSy"];
    size_t      lastComma = apLfSy.find_last_of( ',' );
    m.nSyncChans = (lastComma != std::string::npos)
                 ? std::atoi( apLfSy.substr( lastComma + 1 ).c_str() )
                 : 0;

    return m;
}
