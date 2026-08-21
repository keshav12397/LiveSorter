#include "DriftTracker.h"

#include <cmath>
#include <algorithm>
#include <limits>


namespace {
const size_t kMaxPending = 4096;   // bounded incomplete-spike-group cache
}


DriftTracker::DriftTracker()
    :   imecHz_( 30000.0 ), binS_( 20.0 ), nTop_( 8 ), minBinsToInclude_( 4 ),
        firstSampleIndex_( -1 ),
        nAmpRecordsSeen_( 0 ), nSpikesCompleted_( 0 ), nSpikesDropped_( 0 )
{}


void DriftTracker::configure( const std::vector<int> &unitIds,
                               const std::vector<std::vector<livewire::ChannelGeom> > &unitChannelGeom,
                               double binS, int nTop, int minBinsToInclude )
{
    unitIds_          = unitIds;
    unitChannelGeom_  = unitChannelGeom;
    binS_             = binS;
    nTop_             = nTop;
    minBinsToInclude_ = minBinsToInclude;

    unitIdToIndex_.clear();
    for( size_t i = 0; i < unitIds_.size(); ++i )
        unitIdToIndex_[unitIds_[i]] = static_cast<int>( i );

    if( unitChannelGeom_.size() < unitIds_.size() )
        unitChannelGeom_.resize( unitIds_.size() );

    reset();
}


void DriftTracker::reset()
{
    pending_.clear();
    pendingOrder_.clear();
    unitBinSum_.assign( unitIds_.size(), std::vector<double>() );
    unitBinCount_.assign( unitIds_.size(), std::vector<int>() );
    firstSampleIndex_  = -1;
    nAmpRecordsSeen_   = 0;
    nSpikesCompleted_  = 0;
    nSpikesDropped_    = 0;
}


int DriftTracker::unitIndexOf( int unitId ) const
{
    std::unordered_map<int, int>::const_iterator it = unitIdToIndex_.find( unitId );
    return ( it == unitIdToIndex_.end() ) ? -1 : it->second;
}


void DriftTracker::onAmpChannel( int unitId, long long sampleIndex, int channelIndex, float amplitude )
{
    ++nAmpRecordsSeen_;

    int u = unitIndexOf( unitId );
    if( u < 0 )
        return;
    const std::vector<livewire::ChannelGeom> &geom = unitChannelGeom_[u];
    if( geom.empty() || channelIndex < 0 || static_cast<size_t>( channelIndex ) >= geom.size() )
        return;

    uint64_t key = ( static_cast<uint64_t>( static_cast<uint32_t>( u ) ) << 40 )
                 ^ static_cast<uint64_t>( sampleIndex );

    std::unordered_map<uint64_t, Pending>::iterator it = pending_.find( key );
    if( it == pending_.end() ) {
        if( pending_.size() >= kMaxPending && !pendingOrder_.empty() ) {
            // Evict the oldest incomplete group. It never got all of its
            // channel records (ring drop or a genuinely stalled reader);
            // counting it here, not silently -- see nSpikesDropped().
            uint64_t oldKey = pendingOrder_.front();
            pendingOrder_.pop_front();
            if( pending_.erase( oldKey ) > 0 )
                ++nSpikesDropped_;
        }
        Pending p;
        p.unitIndex   = u;
        p.sampleIndex = sampleIndex;
        p.amp.assign( geom.size(), 0.0f );
        p.got.assign( geom.size(), false );
        p.nGot = 0;
        it = pending_.insert( std::make_pair( key, p ) ).first;
        pendingOrder_.push_back( key );
    }

    Pending &p = it->second;
    if( !p.got[channelIndex] ) {
        p.got[channelIndex] = true;
        ++p.nGot;
    }
    p.amp[channelIndex] = amplitude;

    if( p.nGot == static_cast<int>( geom.size() ) ) {
        finishSpike( p );
        pending_.erase( it );
        // pendingOrder_ still holds `key`; it is a harmless no-op removal
        // attempt the next time it reaches the front of eviction (erase()
        // on a key already gone is simply not counted as a drop, since
        // `pending_.erase` above already ran). Left in place rather than
        // scanned out of a deque, which is the whole reason FIFO order was
        // chosen over a structure that supports O(1) arbitrary removal.
    }
}


void DriftTracker::finishSpike( Pending &p )
{
    ++nSpikesCompleted_;

    const std::vector<livewire::ChannelGeom> &geom = unitChannelGeom_[p.unitIndex];
    size_t n = p.amp.size();

    // Noise-floor subtraction: median across this spike's own channels,
    // same as _position_from_waveform's `amp - median(amp)`.
    std::vector<float> sorted( p.amp );
    std::sort( sorted.begin(), sorted.end() );
    float median = ( n % 2 == 1 ) ? sorted[n / 2]
                                   : 0.5f * ( sorted[n / 2 - 1] + sorted[n / 2] );

    std::vector<double> amp( n );
    bool anyPositive = false;
    for( size_t c = 0; c < n; ++c ) {
        double v = static_cast<double>( p.amp[c] ) - static_cast<double>( median );
        amp[c] = ( v > 0.0 ) ? v : 0.0;
        anyPositive = anyPositive || ( amp[c] > 0.0 );
    }
    if( !anyPositive )
        return;   // NaN in the reference -- nothing to place this spike at

    // Top nTop_ channels by amplitude (n <= nTop_ just uses all of them).
    std::vector<size_t> order( n );
    for( size_t c = 0; c < n; ++c ) order[c] = c;
    std::sort( order.begin(), order.end(),
               [&]( size_t a, size_t b ) { return amp[a] > amp[b]; } );
    size_t nUse = std::min( static_cast<size_t>( std::max( nTop_, 1 ) ), n );

    double wSum = 0.0, wySum = 0.0;
    for( size_t k = 0; k < nUse; ++k ) {
        size_t c = order[k];
        double w = amp[c] * amp[c];
        wSum  += w;
        wySum += w * static_cast<double>( geom[c].yUm );
    }
    if( wSum <= 0.0 )
        return;
    double centroidY = wySum / wSum;

    if( firstSampleIndex_ < 0 )
        firstSampleIndex_ = p.sampleIndex;

    double tS = static_cast<double>( p.sampleIndex - firstSampleIndex_ ) / imecHz_;
    if( tS < 0.0 )
        tS = 0.0;   // a spike that arrived out of order relative to the origin
    int bin = static_cast<int>( tS / binS_ );

    std::vector<double> &sums   = unitBinSum_[p.unitIndex];
    std::vector<int>    &counts = unitBinCount_[p.unitIndex];
    if( static_cast<int>( sums.size() ) <= bin ) {
        sums.resize( bin + 1, 0.0 );
        counts.resize( bin + 1, 0 );
    }
    sums[bin]   += centroidY;
    counts[bin] += 1;
}


double DriftTracker::unitBinMean( int unitIndex, int bin ) const
{
    if( unitIndex < 0 || static_cast<size_t>( unitIndex ) >= unitBinSum_.size() )
        return std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> &sums   = unitBinSum_[unitIndex];
    const std::vector<int>    &counts = unitBinCount_[unitIndex];
    if( bin < 0 || static_cast<size_t>( bin ) >= sums.size() || counts[bin] == 0 )
        return std::numeric_limits<double>::quiet_NaN();
    return sums[bin] / counts[bin];
}


DriftTracker::Trace DriftTracker::trace() const
{
    Trace out;

    size_t nBins = 0;
    for( size_t u = 0; u < unitBinSum_.size(); ++u )
        nBins = std::max( nBins, unitBinSum_[u].size() );

    if( nBins == 0 )
        return out;

    out.tCenterS.resize( nBins );
    for( size_t t = 0; t < nBins; ++t )
        out.tCenterS[t] = ( static_cast<double>( t ) + 0.5 ) * binS_;

    // Per-unit mean-centred displacement, NaN outside that unit's own
    // active span, gap-filled by linear interpolation WITHIN the span --
    // pooled_com_motion's unit_trajectory + np.interp(..., left=nan,
    // right=nan) does the same thing, just over equal-spike-count bins
    // instead of these fixed-time ones.
    std::vector<std::vector<double> > Y;
    Y.reserve( unitBinSum_.size() );

    for( size_t u = 0; u < unitBinSum_.size(); ++u ) {
        const std::vector<double> &sums   = unitBinSum_[u];
        const std::vector<int>    &counts = unitBinCount_[u];

        std::vector<double> y( nBins, std::numeric_limits<double>::quiet_NaN() );
        int lo = -1, hi = -1, nCovered = 0;
        for( size_t t = 0; t < sums.size(); ++t ) {
            if( counts[t] > 0 ) {
                y[t] = sums[t] / counts[t];
                if( lo < 0 ) lo = static_cast<int>( t );
                hi = static_cast<int>( t );
                ++nCovered;
            }
        }
        if( nCovered < minBinsToInclude_ )
            continue;

        // Gap-fill inside [lo, hi] from the covered bins on either side.
        std::vector<double> knownX, knownY;
        for( int t = lo; t <= hi; ++t ) {
            if( !std::isnan( y[t] ) ) {
                knownX.push_back( out.tCenterS[t] );
                knownY.push_back( y[t] );
            }
        }
        for( int t = lo; t <= hi; ++t ) {
            if( std::isnan( y[t] ) ) {
                // Linear interpolation between the surrounding known
                // points; knownX always brackets t here because lo/hi are
                // themselves covered bins.
                double x = out.tCenterS[t];
                size_t k = 1;
                while( k < knownX.size() && knownX[k] < x ) ++k;
                double x0 = knownX[k - 1], x1 = knownX[k];
                double y0 = knownY[k - 1], y1 = knownY[k];
                double frac = ( x1 > x0 ) ? ( x - x0 ) / ( x1 - x0 ) : 0.0;
                y[t] = y0 + frac * ( y1 - y0 );
            }
        }

        double mean = 0.0;
        for( int t = lo; t <= hi; ++t )
            mean += y[t];
        mean /= static_cast<double>( hi - lo + 1 );

        std::vector<double> disp( nBins, std::numeric_limits<double>::quiet_NaN() );
        for( int t = lo; t <= hi; ++t )
            disp[t] = y[t] - mean;

        Y.push_back( disp );
    }

    out.motionUm    = pooledMedianMotion( Y, out.tCenterS );
    out.nUnitsPooled = static_cast<int>( Y.size() );
    return out;
}
