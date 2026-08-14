#ifndef CLOSEDLOOP_SYNCEDGETRACKER_H
#define CLOSEDLOOP_SYNCEDGETRACKER_H

#include <stdexcept>

// Tracks rising edges of a single digital bit across successive samples
// (fed one at a time, possibly across many processChunk() calls / fetch
// cycles), and reports "seconds since the most recent rising edge" -- the
// cross-stream time-alignment technique described in the SpikeGLX manual
// ("the time coordinate of any event can be referenced to the nearest
// pulser edge... sub-millisecond accuracy") and in README.md.
//
// One instance per tracked bit -- ImecFetchThread owns one for the SY
// channel's sync bit, NiFetchThread owns one for the NI DW channel's sync
// bit. Each stream's tracker is independent; see README.md's "Cross-stream
// time alignment" section for the assumption this relies on (analysis
// window << sync pulse period).
class SyncEdgeTracker {
public:
    explicit SyncEdgeTracker( double sampleRateHz )
        :   sampleRateHz_( sampleRateHz ),
            lastBitValue_( 0 ),
            haveLastBit_( false ),
            haveEdge_( false ),
            lastEdgeSampleIndex_( 0 )
    {}

    // Feed one new sample's bit value, at absolute stream sample index
    // `sampleIndex`. Samples must be fed in increasing sampleIndex order
    // (true for a single continuously-polled fetch stream).
    void update( int bitValue, long long sampleIndex )
    {
        if( haveLastBit_ && lastBitValue_ == 0 && bitValue != 0 ) {
            haveEdge_ = true;
            lastEdgeSampleIndex_ = sampleIndex;
        }

        lastBitValue_ = bitValue;
        haveLastBit_  = true;
    }

    bool hasEdge() const { return haveEdge_; }

    long long lastEdgeSampleIndex() const { return lastEdgeSampleIndex_; }

    // Seconds from the most recent rising edge to `sampleIndex` (positive
    // if sampleIndex is after the edge, as it always will be for a live
    // causal detection). Throws if no edge has been seen yet -- callers
    // should check hasEdge() first and simply hold events until the first
    // edge arrives (this is expected at stream startup).
    double secondsSinceLastEdge( long long sampleIndex ) const
    {
        if( !haveEdge_ )
            throw std::runtime_error( "SyncEdgeTracker::secondsSinceLastEdge: no edge seen yet" );

        return static_cast<double>( sampleIndex - lastEdgeSampleIndex_ ) / sampleRateHz_;
    }

private:
    double     sampleRateHz_;
    int        lastBitValue_;
    bool       haveLastBit_;
    bool       haveEdge_;
    long long  lastEdgeSampleIndex_;
};

#endif // CLOSEDLOOP_SYNCEDGETRACKER_H
