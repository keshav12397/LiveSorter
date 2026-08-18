#ifndef CLOSEDLOOP_KILOSORTREADER_H
#define CLOSEDLOOP_KILOSORTREADER_H

#include <string>
#include <vector>
#include <map>

// Loads Kilosort output the same way FilterGen/generate_filter.py's
// load_kilosort() does, for the batch C++ calibration path.
//
// IMPORTANT: reads spike_templates.npy for cluster ids, NOT
// spike_clusters.npy. Calibration.cpp's existing single-target path reads
// spike_clusters.npy -- a different file with (potentially) different ids
// from manual curation. load_kilosort() (the function every Python fit path
// actually calls) reads spike_templates.npy into what it calls spike_cl, so
// the batch C++ path must match that convention, not Calibration.cpp's, or
// units would be selected/labeled against a different clustering than the
// one Python fits against.
struct KilosortData {
    std::vector<long long> spikeTimes;     // samples, ascending order NOT guaranteed (matches spike_times.npy's own order)
    std::vector<long long> spikeClusters;  // parallel to spikeTimes, from spike_templates.npy
    std::map<long long, std::string> labels; // cluster id -> label string ("good"/"mua"/"noise"/...), from cluster_group.tsv or cluster_KSLabel.tsv

    static KilosortData load( const std::string &ksDir );
};

#endif // CLOSEDLOOP_KILOSORTREADER_H
