#ifndef CLOSEDLOOP_LCMVFIT_H
#define CLOSEDLOOP_LCMVFIT_H

#include <vector>
#include <map>
#include <utility>
#include <string>

#include "ScratchMemmap.h"
#include "KilosortReader.h"

// C++ port of FilterGen/generate_filter.py's fit-side functions (NOT the
// noise-covariance estimation -- see NoiseCovariance.h/.cu for that, GPU-
// batched across units). Each function here mirrors one Python function of
// the same name/purpose; see that module's docstrings for the underlying
// rationale, not re-derived here.

// Mean waveform over data's full channel group (templateLength x
// nChannelsAll, row-major [t*nChannelsAll+c]) for one cluster's spikes,
// subsampled to maxSpikes when there are more. Mirrors
// generate_filter.mean_waveform(). Throws std::runtime_error if no spike is
// far enough from the recording edges to form a full window (same
// precondition as the Python version).
std::vector<double> meanWaveform( const ScratchMemmap &data,
                                   const std::vector<long long> &spikeTimes,
                                   int templateLength, int templateOffset,
                                   int maxSpikes, unsigned int seed,
                                   long long &nUsedOut );

// Picks nChannelsPick channel indices (into the 0..nChannelsAll-1 range
// targetWf/interfererWfs are laid out in) maximizing target energy relative
// to the worst interferer's energy on that channel -- mirrors
// generate_filter.select_channels().
std::vector<int> selectChannels( const std::vector<double> &targetWf,
                                  const std::vector<std::vector<double> > &interfererWfs,
                                  int nChannelsAll, int templateLength,
                                  int nChannelsPick );

// Spatially-aware interferer auto-pick -- mirrors
// generate_filter.auto_pick_interferers_spatial(): narrows to a
// by-activity shortlist first, then ranks that shortlist by physical
// distance (channelPositions, raw channel id -> (x,y)) from the target's
// own peak channel. npCh: the full channel group's raw SpikeGLX ids, in the
// same order as data's columns (index i of npCh <-> data column i).
std::vector<long long> autoPickInterferersSpatial(
    const ScratchMemmap &data, const KilosortData &ks, const std::vector<int> &npCh,
    long long targetId, int nInterferers, int templateLength, int templateOffset,
    const std::map<int, std::pair<double, double> > &channelPositions,
    unsigned int seed );

// LCMV solve: f = R^-1 * C * (C^T R^-1 C)^-1 * g, unity gain on the target,
// zero gain on each interferer -- mirrors generate_filter.lcmv_filter().
// sFlat/interfererFlats: flattened (nChannelsSel * templateLength) target/
// interferer templates. R: that same dim, square, row-major, symmetric
// positive-(semi)definite noise covariance (ridge-regularized internally,
// same as Python's R + ridge*(trace(R)/dim)*I).
std::vector<double> lcmvFilter( const std::vector<double> &sFlat,
                                 const std::vector<std::vector<double> > &interfererFlats,
                                 const std::vector<double> &R, double ridge );

#endif // CLOSEDLOOP_LCMVFIT_H
