#ifndef CLOSEDLOOP_SGLXMETAREADER_H
#define CLOSEDLOOP_SGLXMETAREADER_H

#include <string>
#include <vector>
#include <map>
#include <utility>

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

// Minimal loader for a Kilosort-style probe/channel-map JSON's "chanMap"
// field (e.g. FilterGen's shank1only.json) -- deliberately NOT a general
// JSON parser, just enough to pull out one integer array, matching
// FilterGen/generate_filter.py's load_channel_map_json(). This is the CAR
// channel group: per the README/this project's own live testing, CAR must
// be computed across the SAME full channel group used when the filter was
// trained (all 96 shank channels here), not just the filter's own 5
// channels -- computing CAR over too few channels doesn't reject the
// shared noise the filter's statistics were calibrated against, and
// caused a real, large false-positive blowup found via live testing.
std::vector<int> loadChanMapJson( const std::string &path );

// Same file as loadChanMapJson(), but returns {raw_channel_id: (x, y)} from
// the "xc"/"yc" arrays (indexed in parallel with "chanMap") -- mirrors
// generate_filter.load_channel_positions_json(), used for
// auto_pick_interferers_spatial()'s physical-distance ranking. Returns an
// empty map if the file has no "xc"/"yc" fields (caller falls back to
// index-distance, same as the Python side).
std::map<int, std::pair<double, double> > loadChanMapPositions( const std::string &path );

#endif // CLOSEDLOOP_SGLXMETAREADER_H
