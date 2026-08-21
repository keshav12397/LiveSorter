#include "AnalysisThread.h"

AnalysisThread::AnalysisThread( AnalysisFeed &feed )
    :   feed_( feed ),
        nChunksSeen_( 0 ), nSamplesSeen_( 0 ), nSpikesSeen_( 0 ),
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

        // ---- stub body -----------------------------------------------
        // Real drift estimation / plotting payload lands here later (owned
        // by a parallel effort on this queue split). For now: count what
        // arrived, so the wiring can be verified end to end.
        nChunksSeen_.fetch_add( 1 );
        nSamplesSeen_.fetch_add( c.nSamples );
        nSpikesSeen_.fetch_add( static_cast<long long>( c.spikes.size() ) );
        // ----------------------------------------------------------------
    }
}


std::string AnalysisThread::summary() const
{
    return "AnalysisThread: " + std::to_string( nChunksSeen() ) + " chunks seen, "
         + std::to_string( nSamplesSeen() ) + " samples, "
         + std::to_string( nSpikesSeen() ) + " spikes"
         + "  (stub consumer -- drift/plot payload not yet wired)";
}
