#include "AnalysisThread.h"

#include "AmplitudeExtractor.h"
#include "EventPublisher.h"
#include "MultiFilterBank.h"

AnalysisThread::AnalysisThread( AnalysisFeed &feed )
    :   feed_( feed ),
        filterBank_( 0 ), publisher_( 0 ), templateOffset_( 0 ),
        nChunksSeen_( 0 ), nSamplesSeen_( 0 ), nSpikesSeen_( 0 ),
        nAmpRecords_( 0 ),
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
        if( filterBank_ && publisher_ && !c.spikes.empty() ) {
            amps_ = extractAmplitudeRecords( c, *filterBank_, templateOffset_ );
            if( !amps_.empty() ) {
                publisher_->publish( &amps_[0], amps_.size() );
                nAmpRecords_.fetch_add( static_cast<long long>( amps_.size() ) );
            }
        }
    }
}


std::string AnalysisThread::summary() const
{
    return "AnalysisThread: " + std::to_string( nChunksSeen() ) + " chunks seen, "
         + std::to_string( nSamplesSeen() ) + " samples, "
         + std::to_string( nSpikesSeen() ) + " spikes, "
         + std::to_string( nAmpRecords() ) + " amplitude records published"
         + ( filterBank_ && publisher_
              ? ""
              : "  (counting only -- no filter bank/publisher given)" );
}
