#include "SglxMetaReader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>


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
