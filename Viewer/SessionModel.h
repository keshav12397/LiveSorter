#ifndef VIEWER_SESSIONMODEL_H
#define VIEWER_SESSIONMODEL_H

#include <string>
#include <vector>
#include <map>

#include "LiveWire.h"

// Everything SpikeViewer knows about a session, and the only place events are
// interpreted. The UI reads from here and computes nothing itself.
//
// One clock for everything
// ------------------------
// All times in this model are IMEC sample indices. That is not an arbitrary
// choice of unit -- it is the only clock that spikes are natively on, and
// converting everything to it once, at ingest, means the rest of the viewer
// never has to ask which stream a number came from.
//
// Syllables need converting only when they came from the NI stream. That
// conversion uses exactly the assumption DecisionThread itself runs on and
// README.md's "Cross-stream time alignment" section describes: each stream
// timestamps its events as seconds since ITS most recent sync edge, and those
// are comparable while the window of interest is well under the sync period.
// Concretely, every IMEC spike record lets us re-derive where the IMEC sync
// edge sits on the IMEC clock (sampleIndex - timeRelSyncS * rate), and an NI
// syllable is then placed at that edge plus its own timeRelSyncS. Marked
// `clockExact = false` when it happens, and the UI says so, because it is an
// estimate riding on an assumption -- this project has already lost a long
// debugging session to an unexamined stream-to-stream mapping (see
// live_tracking_bug_report.md), and an alignment that quietly presents itself
// as exact is how that happens.
//
// With syllableSource=imecSy there is no conversion at all: the codes are on
// the spikes' own clock, `clockExact` is true, and the whole question is
// absent. That is the point of that test path.
class SessionModel {
public:
    struct Spike {
        long long sampleIndex;   // IMEC clock
        float     score;
    };

    struct Syllable {
        long long sampleIndex;   // IMEC clock (converted if it came from NI)
        int       code;
        bool      clockExact;    // false = placed via the sync-edge estimate above

        // Filled in when the trial's window closes and a Trial record
        // arrives. -1 until then: a trial in flight is genuinely neither
        // triggered nor not-triggered yet, and collapsing that into "not"
        // would silently bias every triggered-vs-not comparison against the
        // most recent trials.
        int triggered;   // -1 unknown, 0 no, 1 yes
        int spikeCount;
    };

    SessionModel();

    // ---- lifetime ---------------------------------------------------------
    void reset();
    void setUnitIds( const std::vector<int> &unitIds );
    void setSampleRates( double imecHz, double niHz );

    // Live ingest. Records are applied in arrival order; the model tolerates
    // gaps (the publisher drops oldest under overload) without corrupting.
    void applyRecord( const livewire::WireRecord &rec );

    // Offline ingest. Returns false and fills `error` if a file could not be
    // read; a missing optional file (syllables/decisions) is not an error.
    //
    // syllableSampleOffset shifts every syllable sample index, for the one
    // case the CSVs cannot resolve on their own: a run recorded with
    // syllableSource=ni writes NI-clock sample indices, and the offset
    // between the two clocks is not recorded anywhere in the CSVs. With
    // syllableSource=imecSy (which is what these files usually come from)
    // the correct offset is 0 and the alignment is exact.
    bool loadCsv( const std::string &spikeCsv, const std::string &syllableCsv,
                   const std::string &decisionCsv, double imecHz,
                   bool syllablesOnImecClock, long long syllableSampleOffset,
                   std::string &error );

    // ---- history bound ----------------------------------------------------
    // Live sessions run for hours; the viewer keeps a rolling window and
    // discards the rest. Trials whose spikes have aged out are discarded too,
    // so a PSTH never silently loses trials' contents while keeping their
    // trial count.
    void setHistorySeconds( double s ) { historyS_ = s; }
    double historySeconds() const { return historyS_; }
    void trimHistory();

    // ---- queries ----------------------------------------------------------
    const std::vector<int> &unitIds() const { return unitIds_; }
    int nUnits() const { return static_cast<int>( unitIds_.size() ); }
    double imecRate() const { return imecRate_; }
    double niRate() const { return niRate_; }

    // Spikes for one unit, ascending by sampleIndex.
    const std::vector<Spike> &unitSpikes( int unitIndex ) const { return perUnit_[unitIndex]; }

    const std::vector<Syllable> &syllables() const { return syllables_; }

    long long latestSample() const { return latestSample_; }
    long long firstSample() const { return firstSample_; }
    double seconds( long long sampleIndex ) const;

    bool lineHigh() const { return lineHigh_; }
    long long nLineRaises() const { return nLineRaises_; }

    long long nSpikes() const { return nSpikes_; }
    bool anySyllableEstimated() const { return anySyllableEstimated_; }

    // Firing rate of one unit over the last `windowS` seconds of stream.
    double firingRateHz( int unitIndex, double windowS ) const;

    // ---- trigger-aligned analysis ----------------------------------------
    // One trial's spike times for one unit, as seconds relative to the
    // trigger, restricted to [winStartS, winEndS].
    struct TrialSpikes {
        int                 syllableIndex;
        int                 code;
        int                 triggered;    // -1/0/1, as in Syllable
        std::vector<double> relS;
    };

    // Trials are returned newest-last, matching the order they occurred, and
    // filtered by code (codeMask bit c set = include code c) and by trigger
    // outcome.
    void collectTrials( int unitIndex, double winStartS, double winEndS,
                         unsigned codeMask, bool includeTriggered,
                         bool includeUntriggered,
                         std::vector<TrialSpikes> &out ) const;

    // PSTH in spikes/s/trial over the same window and filters. `counts` gets
    // nBins entries; `binCentersS` the matching x values. nTrials reports how
    // many trials contributed, because a rate computed from three trials and
    // one from three hundred should not look alike in the UI.
    void computePsth( int unitIndex, double winStartS, double winEndS,
                       int nBins, unsigned codeMask,
                       bool includeTriggered, bool includeUntriggered,
                       std::vector<double> &binCentersS,
                       std::vector<double> &rateHz, int &nTrials ) const;

private:
    int unitIndexOf( int unitId );

    std::vector<int>                unitIds_;
    std::map<int, int>              unitIdToIndex_;
    std::vector<std::vector<Spike> > perUnit_;
    std::vector<Syllable>           syllables_;

    double    imecRate_, niRate_;
    long long firstSample_, latestSample_;
    long long nSpikes_;
    double    historyS_;

    // Latest estimate of where the IMEC sync edge sits on the IMEC clock,
    // re-derived from every spike record. -1 until the first one.
    long long imecSyncEdgeSample_;
    bool      anySyllableEstimated_;

    bool      lineHigh_;
    long long nLineRaises_;
};

#endif // VIEWER_SESSIONMODEL_H
