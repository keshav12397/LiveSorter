#ifndef VIEWER_DRIFTTRACKER_H
#define VIEWER_DRIFTTRACKER_H

#include <vector>
#include <deque>
#include <unordered_map>
#include <cstdint>

#include "LiveWire.h"
#include "DriftPool.h"

// Live, streaming counterpart to FilterGen/drift_estimate.py's
// pooled_com_motion(), built from kWireAmpChannel records rather than a
// batch recording. See DriftPool.h for the part that IS numerically ported
// (the median-across-units pooling); this class is the part that has no
// direct Python analogue, because it has fundamentally different inputs --
// the >500 ms budget bought by AmplitudeExtractor.h means the GUI only
// ever sees one peak amplitude per (spike, channel), never a raw waveform,
// so there is no mean_waveform/select_channels stage to port. What it does
// instead:
//
//   1. Group kWireAmpChannel records back into one spike's reading, keyed
//      by (unit id, sample index) -- see the wire format's own note on
//      that join key. A spike whose group never completes (a dropped
//      record) is discarded; the pending set is capacity-bounded so a
//      permanently incomplete group cannot leak memory over a live run.
//   2. Per completed spike, an amplitude-weighted centroid over that
//      unit's channels: subtract the per-channel median (noise floor,
//      same reasoning as _position_from_waveform), keep only the top
//      `nTop` channels, weight by amplitude^2. Faithful to
//      _position_from_waveform's math, applied per spike instead of per
//      averaged-waveform.
//   3. Bin those centroids into fixed `binS`-second time bins per unit
//      (mean of the spikes landing in each bin) -- this is the one
//      deliberate structural difference from pooled_com_motion, which
//      bins by equal SPIKE COUNT via unit_trajectory. A live tracker has
//      no way to know in advance how many spikes a bin will get, so it
//      bins by wall-clock time instead, at the same 20 s default
//      pooled_com_motion uses for its output grid.
//   4. Per-unit mean subtraction, gap-fill within each unit's own active
//      span, and hand the resulting Y matrix to pooledMedianMotion() --
//      from here on it IS the ported math.
class DriftTracker {
public:
    DriftTracker();

    // unitChannelGeom must be sized nUnits, in unitIds' order, each entry
    // that unit's channels in kWireAmpChannel's `c`-field order -- exactly
    // LiveWireClient::unitChannelGeom(). Safe to call again (e.g. on
    // reconnect); resets all accumulated state.
    void configure( const std::vector<int> &unitIds,
                     const std::vector<std::vector<livewire::ChannelGeom> > &unitChannelGeom,
                     double binS = 20.0, int nTop = 8, int minBinsToInclude = 4 );

    void setSampleRate( double imecHz ) { imecHz_ = imecHz; }

    void reset();

    // Feed one kWireAmpChannel record's fields.
    void onAmpChannel( int unitId, long long sampleIndex, int channelIndex, float amplitude );

    struct Trace {
        std::vector<double> tCenterS;
        std::vector<double> motionUm;
        int                 nUnitsPooled = 0;   // units that made it into Y (diagnostic only)
    };

    // Recomputes the pooled trace from everything accumulated so far. Cheap
    // enough to call once per GUI frame at this bin count (tens to low
    // hundreds of bins x tens to low hundreds of units), but callers doing
    // a lot of polling can cache and only recompute when nAmpRecordsSeen()
    // has advanced.
    Trace trace() const;

    long long nAmpRecordsSeen() const { return nAmpRecordsSeen_; }
    long long nSpikesCompleted() const { return nSpikesCompleted_; }
    long long nSpikesDropped() const { return nSpikesDropped_; }   // evicted incomplete

    // Diagnostic/test accessor: unit `unitIndex`'s raw (pre-pool,
    // pre-mean-subtraction) mean centroid in bin `bin`, or NaN if that
    // unit has no completed spike in that bin. trace() only exposes the
    // fully pooled result, which for a single-bin case is trivially 0 (a
    // lone bin's own mean subtracted from itself) -- this is what
    // test_livewire_roundtrip.cpp's grouping test checks instead, to
    // actually verify the per-spike centroid math rather than the
    // pooling math DriftPool.h already covers on its own fixture.
    double unitBinMean( int unitIndex, int bin ) const;

private:
    struct Pending {
        int                unitIndex;
        long long          sampleIndex;
        std::vector<float> amp;
        std::vector<bool>  got;
        int                nGot;
    };

    int unitIndexOf( int unitId ) const;
    void finishSpike( Pending &p );

    std::vector<int>                                   unitIds_;
    std::vector<std::vector<livewire::ChannelGeom> >    unitChannelGeom_;
    std::unordered_map<int, int>                        unitIdToIndex_;

    double imecHz_;
    double binS_;
    int    nTop_;
    int    minBinsToInclude_;

    long long firstSampleIndex_;   // origin for binning; set from the first completed spike

    // Pending spike groups, keyed by a combination of unit index and
    // sample index. Bounded by kMaxPending via pendingOrder_ (FIFO
    // eviction) so a stream that never completes a group cannot grow this
    // without limit.
    std::unordered_map<uint64_t, Pending> pending_;
    std::deque<uint64_t>                  pendingOrder_;

    // Per unit, per bin: running sum/count of that unit's completed-spike
    // centroids landing in the bin. Grows as bins arrive; index 0 is
    // always bin 0 (time [0, binS_)), so unitBinSum_[u].size() is simply
    // "how many bins this unit has ever needed", not how many it has data
    // in (see unitBinCount_ for that).
    std::vector<std::vector<double> > unitBinSum_;
    std::vector<std::vector<int> >    unitBinCount_;

    long long nAmpRecordsSeen_;
    long long nSpikesCompleted_;
    long long nSpikesDropped_;
};

#endif // VIEWER_DRIFTTRACKER_H
