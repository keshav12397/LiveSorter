#ifndef CLOSEDLOOP_SGLXMETAREADER_H
#define CLOSEDLOOP_SGLXMETAREADER_H

#include <string>
#include <vector>
#include <map>

// Parses a SpikeGLX .meta file and the handful of fields Calibration needs
// to read a raw .bin file directly (i.e. NOT via the live socket API --
// this is only used for the offline training-file calibration pass).
//
// Ported from FilterGen/generate_filter.py's load_sglx_meta /
// parse_chan_subset / find_sync_channel_count -- keep these in sync if the
// Python side changes its parsing.
struct SglxMeta {
    double              sampleRateHz;
    int                 nSavedChans;
    std::vector<int>    savedChannelIds;  // parsed snsSaveChanSubset, e.g. "0:384" -> [0..384]
    int                 nSyncChans;       // from snsApLfSy's last field

    static SglxMeta load( const std::string &metaPath );
};

// Parses a MATLAB-range-style channel spec, e.g. "0,1,4:8,10" -> {0,1,4,5,6,7,8,10}.
std::vector<int> parseChanSubset( const std::string &spec );

// Raw key=value line parser shared by SglxMeta -- exposed in case callers
// need a field this struct doesn't surface.
std::map<std::string, std::string> parseMetaFile( const std::string &metaPath );

#endif // CLOSEDLOOP_SGLXMETAREADER_H
