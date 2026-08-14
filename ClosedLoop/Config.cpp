#include "Config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace {

std::string trim( const std::string &s )
{
    size_t a = s.find_first_not_of( " \t\r\n" );
    if( a == std::string::npos )
        return "";
    size_t b = s.find_last_not_of( " \t\r\n" );
    return s.substr( a, b - a + 1 );
}

} // namespace


Config Config::load( const std::string &path )
{
    std::ifstream fh( path.c_str() );
    if( !fh.is_open() )
        throw std::runtime_error( "Config::load: could not open '" + path + "'" );

    Config cfg;
    std::string line;

    while( std::getline( fh, line ) ) {

        std::string t = trim( line );

        if( t.empty() || t[0] == '#' )
            continue;

        size_t eq = t.find( '=' );
        if( eq == std::string::npos )
            continue; // silently skip malformed lines

        std::string key = trim( t.substr( 0, eq ) );
        std::string val = trim( t.substr( eq + 1 ) );

        cfg.values_[key] = val;
    }

    return cfg;
}


std::string Config::getString( const std::string &key, const std::string &def ) const
{
    std::map<std::string, std::string>::const_iterator it = values_.find( key );
    return (it != values_.end()) ? it->second : def;
}


int Config::getInt( const std::string &key, int def ) const
{
    std::map<std::string, std::string>::const_iterator it = values_.find( key );
    if( it == values_.end() )
        return def;
    return std::atoi( it->second.c_str() );
}


double Config::getDouble( const std::string &key, double def ) const
{
    std::map<std::string, std::string>::const_iterator it = values_.find( key );
    if( it == values_.end() )
        return def;
    return std::atof( it->second.c_str() );
}


bool Config::getBool( const std::string &key, bool def ) const
{
    std::map<std::string, std::string>::const_iterator it = values_.find( key );
    if( it == values_.end() )
        return def;

    std::string v = it->second;
    std::transform( v.begin(), v.end(), v.begin(), ::tolower );
    return (v == "true" || v == "1" || v == "yes");
}


std::vector<int> Config::getIntList( const std::string &key ) const
{
    std::vector<int>    out;
    std::string         v = getString( key );
    std::stringstream   ss( v );
    std::string         tok;

    while( std::getline( ss, tok, ',' ) ) {
        tok = trim( tok );
        if( !tok.empty() )
            out.push_back( std::atoi( tok.c_str() ) );
    }

    return out;
}


std::string Config::requireString( const std::string &key ) const
{
    std::map<std::string, std::string>::const_iterator it = values_.find( key );
    if( it == values_.end() )
        throw std::runtime_error( "Config: missing required key '" + key + "'" );
    return it->second;
}


int Config::requireInt( const std::string &key ) const
{
    return std::atoi( requireString( key ).c_str() );
}
