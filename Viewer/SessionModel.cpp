#include "SessionModel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

// Lower bound on a unit's ascending spike vector by sample index.
size_t lowerBoundBySample( const std::vector<SessionModel::Spike> &v, long long sample )
{
    size_t lo = 0, hi = v.size();
    while( lo < hi ) {
        size_t mid = ( lo + hi ) / 2;
        if( v[mid].sampleIndex < sample )
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

} // namespace


SessionModel::SessionModel()
{
    reset();
}


void SessionModel::reset()
{
    unitIds_.clear();
    unitIdToIndex_.clear();
    perUnit_.clear();
    syllables_.clear();

    imecRate_   = 30000.0;
    niRate_     = 0.0;
    firstSample_  = -1;
    latestSample_ = -1;
    nSpikes_    = 0;
    historyS_   = 120.0;

    imecSyncEdgeSample_   = -1;
    anySyllableEstimated_ = false;

    lineHigh_    = false;
    nLineRaises_ = 0;
}


void SessionModel::setUnitIds( const std::vector<int> &unitIds )
{
    unitIds_ = unitIds;
    unitIdToIndex_.clear();
    for( size_t i = 0; i < unitIds_.size(); ++i )
        unitIdToIndex_[unitIds_[i]] = static_cast<int>( i );
    perUnit_.assign( unitIds_.size(), std::vector<Spike>() );
}


void SessionModel::setSampleRates( double imecHz, double niHz )
{
    if( imecHz > 0 )
        imecRate_ = imecHz;
    niRate_ = niHz;
}


// A unit id that was not in the session header should never happen, but if it
// does, growing the table beats dropping the spike: an unexpected id is
// information, and silently discarding it would look like a dead unit.
int SessionModel::unitIndexOf( int unitId )
{
    std::map<int, int>::const_iterator it = unitIdToIndex_.find( unitId );
    if( it != unitIdToIndex_.end() )
        return it->second;

    int idx = static_cast<int>( unitIds_.size() );
    unitIds_.push_back( unitId );
    unitIdToIndex_[unitId] = idx;
    perUnit_.push_back( std::vector<Spike>() );
    return idx;
}


double SessionModel::seconds( long long sampleIndex ) const
{
    if( firstSample_ < 0 )
        return 0.0;
    return static_cast<double>( sampleIndex - firstSample_ ) / imecRate_;
}


void SessionModel::applyRecord( const livewire::WireRecord &rec )
{
    switch( rec.type ) {

    case livewire::kWireSpike: {
        int idx = unitIndexOf( rec.a );

        Spike s;
        s.sampleIndex = rec.sampleIndex;
        s.score       = rec.b;
        perUnit_[idx].push_back( s );
        ++nSpikes_;

        if( firstSample_ < 0 )
            firstSample_ = rec.sampleIndex;
        if( rec.sampleIndex > latestSample_ )
            latestSample_ = rec.sampleIndex;

        // Re-derive where the IMEC sync edge is, on the IMEC clock. Only
        // useful for placing NI syllables (see the header); costs one
        // multiply per spike and keeps the estimate as fresh as the stream.
        if( rec.timeRelSyncS > 0.0f ) {
            imecSyncEdgeSample_ = rec.sampleIndex
                - static_cast<long long>( static_cast<double>( rec.timeRelSyncS ) * imecRate_ + 0.5 );
        }
        break;
    }

    case livewire::kWireSyllable: {
        Syllable sy;
        sy.code       = rec.a;
        sy.triggered  = -1;
        sy.spikeCount = 0;

        if( rec.stream == livewire::kStreamImec ) {
            // syllableSource=imecSy: same clock as the spikes, nothing to do.
            sy.sampleIndex = rec.sampleIndex;
            sy.clockExact  = true;
        }
        else {
            // NI clock. Place it relative to the IMEC sync edge; see header.
            if( imecSyncEdgeSample_ < 0 )
                return;   // no spike yet, so no edge estimate -- drop rather than guess
            sy.sampleIndex = imecSyncEdgeSample_
                + static_cast<long long>( static_cast<double>( rec.timeRelSyncS ) * imecRate_ + 0.5 );
            sy.clockExact  = false;
            anySyllableEstimated_ = true;
        }

        syllables_.push_back( sy );
        break;
    }

    case livewire::kWireTrial: {
        // A Trial record carries its syllable's own sampleIndex, in that
        // syllable's original stream. Two cases, and they need different
        // joins:
        //
        //   imecSy -- the stored index IS that index, so join on equality.
        //   Searched newest-first because a trial closes shortly after its
        //   syllable.
        //
        //   NI -- ingest re-placed the syllable onto the IMEC clock, so the
        //   stored index no longer equals what the Trial record carries. Join
        //   on the OLDEST still-open trial instead: DecisionThread closes
        //   trials strictly in arrival order (pruneExpired pops from the
        //   front of a FIFO), so that is the one closing, unambiguously.
        for( size_t i = syllables_.size(); i-- > 0; ) {
            if( !syllables_[i].clockExact )
                break;   // fall through to the NI join below
            if( syllables_[i].sampleIndex == rec.sampleIndex ) {
                syllables_[i].triggered  = rec.c ? 1 : 0;
                syllables_[i].spikeCount = static_cast<int>( rec.b );
                return;
            }
        }
        for( size_t i = 0; i < syllables_.size(); ++i ) {
            if( !syllables_[i].clockExact && syllables_[i].triggered < 0 ) {
                syllables_[i].triggered  = rec.c ? 1 : 0;
                syllables_[i].spikeCount = static_cast<int>( rec.b );
                return;
            }
        }
        break;
    }

    case livewire::kWireLine:
        lineHigh_ = ( rec.a != 0 );
        if( lineHigh_ )
            ++nLineRaises_;
        break;

    default:
        break;   // an unknown type from a newer publisher is skipped, not fatal
    }
}


void SessionModel::trimHistory()
{
    if( latestSample_ < 0 || historyS_ <= 0 )
        return;

    long long cutoff = latestSample_ - static_cast<long long>( historyS_ * imecRate_ );
    if( cutoff <= firstSample_ )
        return;

    for( size_t u = 0; u < perUnit_.size(); ++u ) {
        std::vector<Spike> &v = perUnit_[u];
        size_t keepFrom = lowerBoundBySample( v, cutoff );
        if( keepFrom > 0 )
            v.erase( v.begin(), v.begin() + static_cast<long>( keepFrom ) );
    }

    // A trial whose window now extends into discarded spikes would contribute
    // an empty row to every raster and drag the PSTH down, so it goes with
    // them. Half a second of margin covers the widest window the UI offers.
    long long sylCutoff = cutoff + static_cast<long long>( 0.5 * imecRate_ );
    size_t drop = 0;
    while( drop < syllables_.size() && syllables_[drop].sampleIndex < sylCutoff )
        ++drop;
    if( drop > 0 )
        syllables_.erase( syllables_.begin(), syllables_.begin() + static_cast<long>( drop ) );

    firstSample_ = cutoff;
}


double SessionModel::firingRateHz( int unitIndex, double windowS ) const
{
    if( unitIndex < 0 || unitIndex >= static_cast<int>( perUnit_.size() )
        || latestSample_ < 0 || windowS <= 0 ) {
        return 0.0;
    }

    long long from = latestSample_ - static_cast<long long>( windowS * imecRate_ );
    const std::vector<Spike> &v = perUnit_[unitIndex];
    size_t i = lowerBoundBySample( v, from );
    return static_cast<double>( v.size() - i ) / windowS;
}


bool SessionModel::loadCsv( const std::string &spikeCsv, const std::string &syllableCsv,
                             const std::string &decisionCsv, double imecHz,
                             bool syllablesOnImecClock, long long syllableSampleOffset,
                             std::string &error )
{
    reset();
    imecRate_ = ( imecHz > 0 ) ? imecHz : 30000.0;

    // ---- spikes: unit_id,sample_index,score (header line present) --------
    {
        std::ifstream fh( spikeCsv.c_str() );
        if( !fh.is_open() ) {
            error = "could not open " + spikeCsv;
            return false;
        }

        std::string line;
        std::getline( fh, line );   // header

        while( std::getline( fh, line ) ) {
            if( line.empty() )
                continue;
            // A run appends to these files, so a restarted session repeats
            // the header mid-file. Skipping non-numeric first characters
            // handles that without a special case.
            if( !( ( line[0] >= '0' && line[0] <= '9' ) || line[0] == '-' ) )
                continue;

            std::istringstream ss( line );
            std::string a, b, c;
            std::getline( ss, a, ',' );
            std::getline( ss, b, ',' );
            std::getline( ss, c, ',' );

            int idx = unitIndexOf( std::atoi( a.c_str() ) );
            Spike s;
            s.sampleIndex = std::atoll( b.c_str() );
            s.score       = static_cast<float>( std::atof( c.c_str() ) );
            perUnit_[idx].push_back( s );
            ++nSpikes_;

            if( firstSample_ < 0 || s.sampleIndex < firstSample_ )
                firstSample_ = s.sampleIndex;
            if( s.sampleIndex > latestSample_ )
                latestSample_ = s.sampleIndex;
        }
    }

    // The CSV is written in detection order, which is per-chunk and therefore
    // per-unit-interleaved but still ascending within a unit. Sorted anyway:
    // a file concatenated from two runs, or one where a resync jumped the
    // stream backwards, would otherwise break every binary search silently.
    for( size_t u = 0; u < perUnit_.size(); ++u ) {
        std::vector<Spike> &v = perUnit_[u];
        bool sorted = true;
        for( size_t i = 1; i < v.size() && sorted; ++i )
            sorted = ( v[i - 1].sampleIndex <= v[i].sampleIndex );
        if( !sorted ) {
            std::stable_sort( v.begin(), v.end(),
                []( const Spike &x, const Spike &y ) { return x.sampleIndex < y.sampleIndex; } );
        }
    }

    // ---- syllables: code,sampleIndex (no header) -------------------------
    if( !syllableCsv.empty() ) {
        std::ifstream fh( syllableCsv.c_str() );
        std::string line;
        while( std::getline( fh, line ) ) {
            if( line.empty() || !( ( line[0] >= '0' && line[0] <= '9' ) || line[0] == '-' ) )
                continue;
            std::istringstream ss( line );
            std::string a, b;
            std::getline( ss, a, ',' );
            std::getline( ss, b, ',' );

            Syllable sy;
            sy.code        = std::atoi( a.c_str() );
            sy.sampleIndex = std::atoll( b.c_str() ) + syllableSampleOffset;
            sy.clockExact  = syllablesOnImecClock && ( syllableSampleOffset == 0 );
            sy.triggered   = -1;
            sy.spikeCount  = 0;
            if( !sy.clockExact )
                anySyllableEstimated_ = true;
            syllables_.push_back( sy );
        }
        std::stable_sort( syllables_.begin(), syllables_.end(),
            []( const Syllable &x, const Syllable &y ) { return x.sampleIndex < y.sampleIndex; } );
    }

    // ---- trials: syllable_sample_index,code,triggered,spike_count --------
    if( !decisionCsv.empty() ) {
        std::ifstream fh( decisionCsv.c_str() );
        std::string line;
        std::getline( fh, line );   // header

        while( std::getline( fh, line ) ) {
            if( line.empty() || !( ( line[0] >= '0' && line[0] <= '9' ) || line[0] == '-' ) )
                continue;
            std::istringstream ss( line );
            std::string a, b, c, d;
            std::getline( ss, a, ',' );
            std::getline( ss, b, ',' );
            std::getline( ss, c, ',' );
            std::getline( ss, d, ',' );

            long long sylSample = std::atoll( a.c_str() ) + syllableSampleOffset;
            for( size_t i = 0; i < syllables_.size(); ++i ) {
                if( syllables_[i].sampleIndex == sylSample ) {
                    syllables_[i].triggered  = ( std::atoi( c.c_str() ) != 0 ) ? 1 : 0;
                    syllables_[i].spikeCount = std::atoi( d.c_str() );
                    if( syllables_[i].triggered == 1 )
                        ++nLineRaises_;
                    break;
                }
            }
        }
    }

    // Offline, the whole file is the history; trimming it would throw away
    // the run the user just opened.
    historyS_ = 0.0;
    return true;
}


void SessionModel::collectTrials( int unitIndex, double winStartS, double winEndS,
                                   unsigned codeMask, bool includeTriggered,
                                   bool includeUntriggered,
                                   std::vector<TrialSpikes> &out ) const
{
    out.clear();
    if( unitIndex < 0 || unitIndex >= static_cast<int>( perUnit_.size() ) )
        return;

    const std::vector<Spike> &v = perUnit_[unitIndex];
    const long long preSamples  = static_cast<long long>( winStartS * imecRate_ );
    const long long postSamples = static_cast<long long>( winEndS * imecRate_ );

    for( size_t s = 0; s < syllables_.size(); ++s ) {

        const Syllable &sy = syllables_[s];

        if( sy.code >= 0 && sy.code < 32 && !( codeMask & ( 1u << sy.code ) ) )
            continue;
        // A trial still in flight (triggered == -1) is shown only when both
        // outcome filters are on, so neither "triggered" nor "not triggered"
        // ever silently includes trials whose outcome is not yet known.
        if( sy.triggered == 1 && !includeTriggered )
            continue;
        if( sy.triggered == 0 && !includeUntriggered )
            continue;
        if( sy.triggered < 0 && !( includeTriggered && includeUntriggered ) )
            continue;

        TrialSpikes t;
        t.syllableIndex = static_cast<int>( s );
        t.code          = sy.code;
        t.triggered     = sy.triggered;

        size_t i = lowerBoundBySample( v, sy.sampleIndex + preSamples );
        for( ; i < v.size() && v[i].sampleIndex <= sy.sampleIndex + postSamples; ++i ) {
            t.relS.push_back( static_cast<double>( v[i].sampleIndex - sy.sampleIndex ) / imecRate_ );
        }

        out.push_back( t );
    }
}


void SessionModel::computePsth( int unitIndex, double winStartS, double winEndS,
                                 int nBins, unsigned codeMask,
                                 bool includeTriggered, bool includeUntriggered,
                                 std::vector<double> &binCentersS,
                                 std::vector<double> &rateHz, int &nTrials ) const
{
    binCentersS.assign( nBins > 0 ? nBins : 0, 0.0 );
    rateHz.assign( nBins > 0 ? nBins : 0, 0.0 );
    nTrials = 0;

    if( nBins <= 0 || winEndS <= winStartS )
        return;

    const double binW = ( winEndS - winStartS ) / nBins;
    for( int b = 0; b < nBins; ++b )
        binCentersS[b] = winStartS + ( b + 0.5 ) * binW;

    std::vector<TrialSpikes> trials;
    collectTrials( unitIndex, winStartS, winEndS, codeMask,
                    includeTriggered, includeUntriggered, trials );
    nTrials = static_cast<int>( trials.size() );
    if( nTrials == 0 )
        return;

    for( size_t t = 0; t < trials.size(); ++t ) {
        for( size_t i = 0; i < trials[t].relS.size(); ++i ) {
            int b = static_cast<int>( ( trials[t].relS[i] - winStartS ) / binW );
            if( b >= 0 && b < nBins )
                rateHz[b] += 1.0;
        }
    }

    // spikes/s/trial, so the y axis is a firing rate and does not change
    // meaning when the bin width or the trial count does.
    const double denom = binW * nTrials;
    for( int b = 0; b < nBins; ++b )
        rateHz[b] /= denom;
}
