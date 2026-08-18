#include "KilosortReader.h"

#include <fstream>
#include <sstream>

#include "NpyReader.h"


namespace {

// Mirrors load_kilosort()'s label-loading loop: tries cluster_group.tsv
// first, falls back to cluster_KSLabel.tsv, stops after whichever one is
// actually found (not both -- matches Python's for/break behavior).
std::map<long long, std::string> loadLabels( const std::string &ksDir )
{
    std::map<long long, std::string> labels;
    const char *candidates[] = { "cluster_group.tsv", "cluster_KSLabel.tsv" };

    for( int i = 0; i < 2; ++i ) {

        std::string path = ksDir + "/" + candidates[i];
        std::ifstream fh( path.c_str() );
        if( !fh.is_open() )
            continue;

        std::string line;
        std::getline( fh, line ); // header

        while( std::getline( fh, line ) ) {

            if( !line.empty() && line[line.size() - 1] == '\r' )
                line.erase( line.size() - 1 );
            if( line.empty() )
                continue;

            size_t tab = line.find( '\t' );
            if( tab == std::string::npos )
                continue;

            long long id = std::atoll( line.substr( 0, tab ).c_str() );
            std::string label = line.substr( tab + 1 );
            labels[id] = label;
        }

        break; // matches Python: first file found wins, no merging
    }

    return labels;
}

} // namespace


KilosortData KilosortData::load( const std::string &ksDir )
{
    std::vector<long long> rawTimes    = readNpy1D( ksDir + "/spike_times.npy" );
    std::vector<long long> rawClusters = readNpy1D( ksDir + "/spike_templates.npy" );

    if( rawTimes.size() != rawClusters.size() )
        throw std::runtime_error( "KilosortData::load: spike_times.npy / "
                                   "spike_templates.npy size mismatch in '" + ksDir + "'" );

    KilosortData out;
    out.spikeTimes.reserve( rawTimes.size() );
    out.spikeClusters.reserve( rawTimes.size() );

    // Matches Python's `keep = spike_t >= 0` filter.
    for( size_t i = 0; i < rawTimes.size(); ++i ) {
        if( rawTimes[i] >= 0 ) {
            out.spikeTimes.push_back( rawTimes[i] );
            out.spikeClusters.push_back( rawClusters[i] );
        }
    }

    out.labels = loadLabels( ksDir );
    return out;
}
