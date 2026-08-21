#include "AnalysisThread.h"

#include "AmplitudeExtractor.h"
#include "EventPublisher.h"
#include "MultiFilterBank.h"

#include <cstring>

AnalysisThread::AnalysisThread( AnalysisFeed &feed )
    :   feed_( feed ),
        filterBank_( 0 ), publisher_( 0 ), templateOffset_( 0 ),
        nChunksSeen_( 0 ), nSamplesSeen_( 0 ), nSpikesSeen_( 0 ),
        nAmpRecords_( 0 ),
        histFirstSample_( -1 ), histSamples_( 0 ),
        histCapacity_( 0 ), histChannels_( 0 ),
        nOutOfSpan_( 0 ),
        stopFlag_( false )
{}


AnalysisThread::AnalysisThread( AnalysisFeed &feed,
                                 const MultiFilterBank &filterBank,
                                 EventPublisher *publisher,
                                 int templateOffset )
    :   feed_( feed ),
        filterBank_( &filterBank ), publisher_( publisher ),
        templateOffset_( templateOffset ),
        nChunksSeen_( 0 ), nSamplesSeen_( 0 ), nSpikesSeen_( 0 ),
        nAmpRecords_( 0 ),
        histFirstSample_( -1 ), histSamples_( 0 ),
        histCapacity_( 0 ), histChannels_( 0 ),
        nOutOfSpan_( 0 ),
        stopFlag_( false )
{}


void AnalysisThread::start() { thread_ = std::thread( &AnalysisThread::runLoop, this ); }
void AnalysisThread::stop()  { stopFlag_.store( true ); }
void AnalysisThread::join()  { if( thread_.joinable() ) thread_.join(); }


void AnalysisThread::runLoop()
{
    AnalysisFeed::Chunk c;

    while( !stopFlag_.load() ) {

        if( !feed_.take( c, 100 ) )
            continue;

        // Scope guard: release() runs whether the stub body below returns
        // normally or an exception unwinds through it. A chunk taken and
        // never released leaks a buffer out of the pool permanently -- see
        // AnalysisFeed.h and this class's header comment.
        struct ReleaseGuard {
            AnalysisFeed &f;
            int idx;
            ~ReleaseGuard() { f.release( idx ); }
        } guard{ feed_, c.poolIndex };

        nChunksSeen_.fetch_add( 1 );
        nSamplesSeen_.fetch_add( c.nSamples );
        nSpikesSeen_.fetch_add( static_cast<long long>( c.spikes.size() ) );

        // Amplitude extraction for the viewer's drift tracker. Runs HERE,
        // never on the fetch thread -- see this class's header comment.
        if( filterBank_ && publisher_ ) {
            appendToHistory( c );

            if( !c.spikes.empty() && histSamples_ > 0 ) {
                amps_ = extractAmplitudeRecords(
                    &hist_[0], histSamples_, histChannels_, histFirstSample_,
                    c.spikes, *filterBank_, templateOffset_ );
                if( !amps_.empty() ) {
                    publisher_->publish( &amps_[0], amps_.size() );
                    nAmpRecords_.fetch_add( static_cast<long long>( amps_.size() ) );
                }
                // One record per (spike, channel), so a spike that produced
                // none fell outside the history span. Counted rather than
                // silent: a persistent nonzero here means the history is too
                // short for this rig's detection lag.
                long long got = static_cast<long long>( amps_.size() )
                                 / ( filterBank_->nChannelsPerUnit > 0
                                      ? filterBank_->nChannelsPerUnit : 1 );
                long long missed = static_cast<long long>( c.spikes.size() ) - got;
                if( missed > 0 )
                    nOutOfSpan_.fetch_add( missed );
            }
        }
    }
}


void AnalysisThread::appendToHistory( const AnalysisFeed::Chunk &c )
{
    if( c.nSamples <= 0 || c.nChannels <= 0 || !c.samples )
        return;

    // Sized on the first chunk from the real detection lag plus a whole
    // template, with slack for a chunk that arrives larger than nominal.
    // templateLength/2 + minSeparationSamples (= templateLength/2) is the
    // reporting lag ConvolutionEngine imposes, so templateLength * 2 covers
    // lag + window; the extra chunks are margin for a fetch hiccup.
    // Sized from the feed's NOMINAL buffer size, never from c.nSamples.
    // The first chunk of a run is routinely a short partial one -- 44
    // samples against a nominal 150 on the first live run -- and sizing off
    // it produced a 298-sample history where 722 was needed, leaving every
    // spike's window ~20 samples short of the buffer start. Every spike was
    // then counted out-of-span, which looks exactly like the lag bug this
    // history was added to fix.
    const int nominal = feed_.samplesPerBuffer() > 0
                         ? feed_.samplesPerBuffer() : c.nSamples;
    if( histCapacity_ == 0 || c.nChannels != histChannels_ ||
        c.nSamples > histCapacity_ - 2 * ( filterBank_ ? filterBank_->templateLength : 61 ) ) {
        const int tLen = filterBank_ ? filterBank_->templateLength : 61;
        histChannels_  = c.nChannels;
        // Must span: the detection reporting lag ((tLen-1)/2 lookahead plus
        // minSeparationSamples = tLen/2), the templateOffset before the
        // spike, the tLen window itself, and a whole arriving chunk. 4x the
        // nominal chunk plus 4x tLen covers all of it with margin for a
        // chunk that arrives larger than nominal.
        histCapacity_  = tLen * 4 + ( nominal > c.nSamples ? nominal : c.nSamples ) * 4;
        hist_.assign( static_cast<size_t>( histCapacity_ ) * histChannels_, 0.0f );
        histSamples_     = 0;
        histFirstSample_ = -1;
    }

    const size_t rowFloats = static_cast<size_t>( histChannels_ );

    // A gap (dropped samples, or a resync) makes the retained history
    // non-contiguous with the arriving chunk, and a window spanning the seam
    // would read samples that are not adjacent in time. Start over.
    if( histFirstSample_ >= 0 &&
        c.firstSampleIndex != histFirstSample_ + histSamples_ ) {
        histSamples_     = 0;
        histFirstSample_ = -1;
    }

    int keep = histCapacity_ - c.nSamples;
    if( keep < 0 ) {
        // A chunk larger than the whole buffer: keep only its tail.
        const int take = histCapacity_;
        const int skip = c.nSamples - take;
        std::memcpy( &hist_[0], c.samples + static_cast<size_t>( skip ) * rowFloats,
                     static_cast<size_t>( take ) * rowFloats * sizeof( float ) );
        histSamples_     = take;
        histFirstSample_ = c.firstSampleIndex + skip;
        return;
    }

    if( histSamples_ > keep ) {
        const int drop = histSamples_ - keep;
        std::memmove( &hist_[0],
                      &hist_[static_cast<size_t>( drop ) * rowFloats],
                      static_cast<size_t>( keep ) * rowFloats * sizeof( float ) );
        histFirstSample_ += drop;
        histSamples_ = keep;
    }

    std::memcpy( &hist_[static_cast<size_t>( histSamples_ ) * rowFloats],
                 c.samples,
                 static_cast<size_t>( c.nSamples ) * rowFloats * sizeof( float ) );
    if( histFirstSample_ < 0 )
        histFirstSample_ = c.firstSampleIndex;
    histSamples_ += c.nSamples;
}


std::string AnalysisThread::summary() const
{
    return "AnalysisThread: " + std::to_string( nChunksSeen() ) + " chunks seen, "
         + std::to_string( nSamplesSeen() ) + " samples, "
         + std::to_string( nSpikesSeen() ) + " spikes, "
         + std::to_string( nAmpRecords() ) + " amplitude records published, "
         + std::to_string( nSpikesOutOfSpan() ) + " spikes outside the history span"
         + ( filterBank_ && publisher_
              ? ""
              : "  (counting only -- no filter bank/publisher given)" );
}
