#ifndef CLOSEDLOOP_CONFIG_H
#define CLOSEDLOOP_CONFIG_H

#include <string>
#include <vector>
#include <map>

// Minimal key=value config file reader.
//
// Deliberately the same trivial format/parsing style as SpikeGLX .meta
// files (see SglxMetaReader) -- one shared mental model, no JSON library
// dependency. Lines starting with '#' (after optional leading whitespace)
// are comments; blank lines are ignored.
//
// This is the single place to touch if you want a new tunable parameter --
// add a getter here, read it in main.cpp/whichever module needs it, and
// document it in config.example.txt.
class Config {
public:
    // Load and parse a config file. Throws std::runtime_error on I/O failure.
    static Config load( const std::string &path );

    std::string getString( const std::string &key, const std::string &def = "" ) const;
    int         getInt( const std::string &key, int def = 0 ) const;
    double      getDouble( const std::string &key, double def = 0.0 ) const;
    bool        getBool( const std::string &key, bool def = false ) const;

    // Parses a comma-separated list of ints, e.g. "5,6,7" -> {5,6,7}.
    std::vector<int> getIntList( const std::string &key ) const;

    // Throws std::runtime_error if the key is missing -- use for settings
    // that have no sane default (paths, target id, etc.).
    std::string requireString( const std::string &key ) const;
    int         requireInt( const std::string &key ) const;

private:
    std::map<std::string, std::string> values_;
};

#endif // CLOSEDLOOP_CONFIG_H
