#include "DecisionThread.h"

#include <iostream>
#include <algorithm>

#include "SglxCppClient.h"
#include "EventPublisher.h"

namespace {
    // sglx_ni_DO_set's `bits` parameter maps bits to the lines named in its
    // `lines` string argument (see SglxApi.h) -- since doLine_ always names
    // exactly one line, bit 0 set/clear is sufficient regardless of which
    // physical line it is.
    const unsigned int kLineHighBits = 0x1;
    const unsigned int kLineLowBits  = 0x0;
}


DecisionThread::DecisionThread( void *hSglx, ThreadSafeQueue<SpikeEvent> &spikeQueue,
                                 ThreadSafeQueue<SyllableEvent> &syllableQueue,
                                 double windowStartS, double windowEndS, int spikeCountThreshold,
                                 const std::string &doLine, int doPulseMs,
                                 const std::string &decisionLogPath,
                                 const std::vector<int> &decisionUnitIds,
                                 const std::string &decisionCsvPath,
                                 EventPublisher *publisher )
    :   hSglx_( hSglx ), spikeQueue_( spikeQueue ), syllableQueue_( syllableQueue ),
        windowStartS_( windowStartS ), windowEndS_( windowEndS ),
        spikeCountThreshold_( spikeCountThreshold ), doLine_( doLine ), doPulseMs_( doPulseMs ),
        decisionUnitIds_( decisionUnitIds ), countsAllUnits_( decisionUnitIds.empty() ),
        publisher_( publisher ),
        lineHigh_( false ), stopFlag_( false )
{
    // Sorted once here so onSpikeEvent can binary-search instead of doing a
    // linear scan per spike -- that call runs on every one of 157 units.
    std::sort( decisionUnitIds_.begin(), decisionUnitIds_.end() );

    if( !decisionLogPath.empty() )
        decisionLog_.open( decisionLogPath.c_str(), std::ios::app );

    if( !decisionCsvPath.empty() ) {
        decisionCsv_.open( decisionCsvPath.c_str(), std::ios::app );
        if( decisionCsv_.is_open() )
            decisionCsv_ << "syllable_sample_index,code,triggered,spike_count\n";
    }
}


void DecisionThread::start() { thread_ = std::thread( &DecisionThread::runLoop, this ); }
void DecisionThread::stop()  { stopFlag_.store( true ); }
void DecisionThread::join()  { if( thread_.joinable() ) thread_.join(); }


void DecisionThread::runLoop()
{
    // Ensure the line starts low.
    sglx_ni_DO_set( hSglx_, doLine_, kLineLowBits );

    while( !stopFlag_.load() ) {

        SpikeEvent spk;
        if( spikeQueue_.waitPop( spk, 1 ) )
            onSpikeEvent( spk );

        SyllableEvent syl;
        if( syllableQueue_.waitPop( syl, 1 ) )
            onSyllableEvent( syl );

        pruneExpired();
        lowerLineIfDue();
    }

    sglx_ni_DO_set( hSglx_, doLine_, kLineLowBits );
}


void DecisionThread::onSyllableEvent( const SyllableEvent &syl )
{
    PendingSyllable p;
    p.event      = syl;
    p.count      = 0;
    p.triggered  = false;
    p.receivedAt = std::chrono::steady_clock::now();
    pending_.push_back( p );
}


void DecisionThread::onSpikeEvent( const SpikeEvent &spk )
{
    // Unit gating. With one target there is nothing to gate (and unitId is
    // -1), so an empty list keeps the original "any spike counts" rule; with
    // a whole session's units on the queue, a decision driven by "any of 157
    // units fired" is not a decision about anything.
    if( !countsAllUnits_
        && !std::binary_search( decisionUnitIds_.begin(), decisionUnitIds_.end(), spk.unitId ) ) {
        return;
    }

    for( size_t i = 0; i < pending_.size(); ++i ) {

        PendingSyllable &p = pending_[i];
        if( p.triggered )
            continue;

        double dt = spk.timeRelSyncS - p.event.timeRelSyncS;
        if( dt < windowStartS_ || dt > windowEndS_ )
            continue;

        ++p.count;

        if( p.count >= spikeCountThreshold_ ) {
            raiseLine();
            p.triggered = true;

            if( decisionLog_.is_open() ) {
                decisionLog_ << "TRIGGER syllable_code=" << p.event.code
                             << " syllable_t=" << p.event.timeRelSyncS
                             << " count=" << p.count
                             << " triggering_spike_t=" << spk.timeRelSyncS << "\n";
                decisionLog_.flush();
            }
        }
    }
}


void DecisionThread::pruneExpired()
{
    // Wall-clock based (not sample-relative-time based): a syllable whose
    // window closed a while ago in real time gets logged and dropped,
    // independent of whether new spikes have arrived to drive the
    // sample-relative-time comparisons in onSpikeEvent().
    std::chrono::milliseconds windowEndMs(
        static_cast<long long>( windowEndS_ * 1000.0 ) + 50 ); // small margin for fetch/queue latency

    while( !pending_.empty() ) {

        PendingSyllable &p = pending_.front();
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

        if( now - p.receivedAt < windowEndMs )
            break; // front (oldest) hasn't expired yet, and pending_ is FIFO by arrival order

        if( !p.triggered && decisionLog_.is_open() ) {
            decisionLog_ << "NO_TRIGGER syllable_code=" << p.event.code
                         << " syllable_t=" << p.event.timeRelSyncS
                         << " final_count=" << p.count << "\n";
            decisionLog_.flush();
        }

        closeTrial( p );
        pending_.pop_front();
    }
}


// A trial is only ever closed by pruneExpired(), which is FIFO by arrival,
// so these rows and records come out in trial order.
void DecisionThread::closeTrial( const PendingSyllable &p )
{
    if( decisionCsv_.is_open() ) {
        decisionCsv_ << p.event.sampleIndex << "," << p.event.code << ","
                     << ( p.triggered ? 1 : 0 ) << "," << p.count << "\n";
        decisionCsv_.flush();
    }

    if( publisher_ ) {
        publisher_->publish( livewire::makeRecord(
            livewire::kWireTrial, livewire::kStreamNi,
            p.event.code, p.event.sampleIndex,
            static_cast<float>( p.count ),
            static_cast<float>( p.event.timeRelSyncS ),
            p.triggered ? 1 : 0 ) );
    }
}


void DecisionThread::raiseLine()
{
    if( !sglx_ni_DO_set( hSglx_, doLine_, kLineHighBits ) )
        std::cerr << "DecisionThread: sglx_ni_DO_set (raise) failed: " << sglx_getError( hSglx_ ) << "\n";

    lineHigh_ = true;
    lineHighUntil_ = std::chrono::steady_clock::now() + std::chrono::milliseconds( doPulseMs_ );

    if( publisher_ ) {
        publisher_->publish( livewire::makeRecord(
            livewire::kWireLine, livewire::kStreamNone, 1, 0, 0.0f, 0.0f ) );
    }
}


void DecisionThread::lowerLineIfDue()
{
    if( lineHigh_ && std::chrono::steady_clock::now() >= lineHighUntil_ ) {
        if( !sglx_ni_DO_set( hSglx_, doLine_, kLineLowBits ) )
            std::cerr << "DecisionThread: sglx_ni_DO_set (lower) failed: " << sglx_getError( hSglx_ ) << "\n";
        lineHigh_ = false;

        if( publisher_ ) {
            publisher_->publish( livewire::makeRecord(
                livewire::kWireLine, livewire::kStreamNone, 0, 0, 0.0f, 0.0f ) );
        }
    }
}
