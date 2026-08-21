#include "ImecFetchThreadCpu.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdexcept>


#include "SglxCppClient.h"
#include "DigitalWordUtils.h"
#include "SglxMetaReader.h"
#include "SyncEdgeTracker.h"
#include "Preprocessor.h"
#include "MultiConvolutionEngine.h"
#include "StreamAccountant.h"
#include "SyllableDecoder.h"
#include "EventPublisher.h"

namespace {
    const int kImecJs = 2;
    const int kImecIp = 0;
}


ImecFetchThreadCpu::ImecFetchThreadCpu( void *hSglx, const MultiFilterBank &filterBank,
                                         const std::string &carChannelMapJsonPath,
                                         bool applyHighpass, double highpassCutoffHz,
                                         int imecSyncBit, int fetchChunkMs,
                                         const std::string &spikeTimesPath,
                                         const std::string &latencyLogPath,
                                         SpikeQueue *spikeQueue,
                                         EventPublisher *publisher,
                                         const SyllableFromSy &syllableFromSy,
                                         AnalysisFeed *analysisFeed,
                                         DriftSchedule *driftSchedule )
    :   hSglx_( hSglx ), filterBank_( filterBank ),
        carChannelMapJsonPath_( carChannelMapJsonPath ),
        applyHighpass_( applyHighpass ), highpassCutoffHz_( highpassCutoffHz ),
        imecSyncBit_( imecSyncBit ), fetchChunkMs_( fetchChunkMs ),
        spikeTimesPath_( spikeTimesPath ), latencyLogPath_( latencyLogPath ),
        driftSchedule_( driftSchedule ), nSwapsApplied_( 0 ),
        spikeQueue_( spikeQueue ), publisher_( publisher ),
        syllableFromSy_( syllableFromSy ),
        analysisFeed_( analysisFeed ),
        stopFlag_( false )
{}


void ImecFetchThreadCpu::start()
{
    thread_ = std::thread( &ImecFetchThreadCpu::fetchLoop, this );
}


void ImecFetchThreadCpu::stop()
{
    stopFlag_.store( true );
}


void ImecFetchThreadCpu::join()
{
    if( thread_.joinable() )
        thread_.join();
}


void ImecFetchThreadCpu::fetchLoop()
{
    std::ofstream spikeTimesFile( spikeTimesPath_.c_str(), std::ios::app );
    if( !spikeTimesFile.is_open() )
        std::cerr << "ImecFetchThreadCpu: could not open '" << spikeTimesPath_ << "' for writing\n";
    spikeTimesFile << "unit_id,sample_index,score\n";

    std::ofstream latencyLogFile;
    if( !latencyLogPath_.empty() ) {
        latencyLogFile.open( latencyLogPath_.c_str(), std::ios::app );
        if( !latencyLogFile.is_open() )
            std::cerr << "ImecFetchThreadCpu: could not open '" << latencyLogPath_ << "' for writing\n";
        else
            latencyLogFile << "chunk_index,n_samples,latency_ms\n";
    }
    // Same `code,sampleIndex` format NiFetchThread writes, deliberately:
    // the viewer's offline reader then has one syllable-CSV parser rather
    // than one per source. Only opened on the imecSy test path -- on the
    // production path NiFetchThread owns this file.
    std::ofstream syllableTimesFile;
    if( syllableFromSy_.enabled && !syllableFromSy_.syllableTimesPath.empty() ) {
        syllableTimesFile.open( syllableFromSy_.syllableTimesPath.c_str(), std::ios::app );
        if( !syllableTimesFile.is_open() ) {
            std::cerr << "ImecFetchThreadCpu: could not open '"
                      << syllableFromSy_.syllableTimesPath << "' for writing\n";
        }
    }

    long long chunkIndex = 0;

    double sampleRate = sglx_getStreamSampleRate( hSglx_, kImecJs, kImecIp );
    if( sampleRate <= 0 ) {
        std::cerr << "ImecFetchThreadCpu: sglx_getStreamSampleRate failed: "
                  << sglx_getError( hSglx_ ) << "\n";
        return;
    }

    cppClient_sglx_get_ints acqChans;
    if( !sglx_getStreamAcqChans( acqChans, hSglx_, kImecJs, kImecIp ) || acqChans.vint.size() < 3 ) {
        std::cerr << "ImecFetchThreadCpu: sglx_getStreamAcqChans failed: "
                  << sglx_getError( hSglx_ ) << "\n";
        return;
    }
    int syChannelIndex = acqChans.vint[0] + acqChans.vint[1];

    bool applyCar = !carChannelMapJsonPath_.empty();
    std::vector<int> carChannelIds = applyCar
        ? loadChanMapJson( carChannelMapJsonPath_ )
        : std::vector<int>(); // all-units path requires a CAR group -- see below
    if( carChannelIds.empty() ) {
        std::cerr << "ImecFetchThreadCpu: carChannelMapJson is required for the "
                     "all-units pipeline (unlike the single-target path, "
                     "there's no per-unit fallback channel set to use instead)\n";
        return;
    }
    const int nCarChans = static_cast<int>( carChannelIds.size() );

    // Translate the filter bank's raw SpikeGLX channel indices to positions
    // within this CAR group -- same requirement/logic as
    // ImecFetchThread.cpp's filterIndexWithinCarGroup, applied to every
    // unit's channels at once. filterBank_ is this thread's own copy (see
    // the header), so this rewrites it in place with no round-trip.
    //
    // This MUST run before the engine is constructed. MultiConvolutionEngine
    // validates that every channel index is a position within the CAR group,
    // and an untranslated raw SpikeGLX id is usually still a plausible small
    // integer -- so getting the order wrong would silently filter the wrong
    // channels rather than fail.
    //
    // A drift swap arrives with raw SpikeGLX ids too, so the SAME mapping has
    // to run on it. It is a lambda rather than a repeated loop because a
    // second, independently-written copy of this translation is exactly the
    // failure this codebase has already paid for twice -- and here it would
    // be silent, since an untranslated id is usually still a valid index.
    {
        int     badUnit = -1;
        int32_t badChan = -1;
        if( !filterBank_.translateChannelsToCarGroup( carChannelIds, &badUnit, &badChan ) ) {
            std::cerr << "ImecFetchThreadCpu: unit at index " << badUnit
                      << " uses channel " << badChan
                      << " which is not part of the CAR channel group\n";
            return;
        }
    }

    std::vector<int> channelSubset( carChannelIds.begin(), carChannelIds.end() );
    channelSubset.push_back( syChannelIndex );
    const int nFetchChans = static_cast<int>( channelSubset.size() );

    const long long chunkSamples = static_cast<long long>( fetchChunkMs_ / 1000.0 * sampleRate + 0.5 );
    const int maxChunkSamples = static_cast<int>( chunkSamples ) + 1; // small slack for rounding

    Preprocessor preprocessor( nCarChans, highpassCutoffHz_, sampleRate, applyHighpass_, applyCar );
    // minSeparationSamples: templateLength/2, same convention every other
    // caller uses (OfflineScorer.cpp's
    // batch calibration scoring, every NMS equivalence test). Passed
    // explicitly for legibility, not to fix anything: ConvolutionEngine's
    // constructor already maps a non-positive value to templateLength/2, so
    // the earlier omission here was a no-op, not the "NO minimum separation
    // enforced" bug an earlier version of this comment claimed. That claim
    // came from misreading a live run in which NO unit tracked ground truth
    // -- the run's detections were being compared against a misaligned
    // ground truth (FilterGen/stream_alignment.py's docstring has the whole
    // story). Realigned, that same run's per-unit F1 matches calibration's
    // offline prediction at r=0.97.
    const long long minSep = filterBank_.templateLength / 2;
    // nThreads 0 = hardware_concurrency. Worth revisiting on a rig where
    // SpikeGLX is the CPU-hungry neighbour: the whole bank is a fraction of
    // one core's worth of arithmetic, so capping this costs little detection
    // headroom and can hand the acquisition side its cores back.
    MultiConvolutionEngine engine( filterBank_, nCarChans, minSep, 0 );
    SyncEdgeTracker syncTracker( sampleRate );

    // The SAME debounce state machine NiFetchThread runs, not a second one
    // written to match it (SyllableDecoder.h was extracted from that loop
    // for exactly this). Constructed unconditionally because it is ~40 bytes
    // and costs nothing unused; it is only fed when enabled.
    SyllableDecoder syllableDecoder( syllableFromSy_.startBit, syllableFromSy_.width,
                                      syllableFromSy_.debounceSamples );

    // Staging buffer for the CAR-group-only (SY stripped) portion of each
    // fetch. Sized once from maxChunkSamples and reused, so the fetch loop
    // itself never allocates.
    std::vector<short> hostCarChunkStorage(
        static_cast<size_t>( maxChunkSamples ) * nCarChans );
    short *hostCarChunk = hostCarChunkStorage.data();

    // Ring-buffer depth, in samples. sglx_fetch cannot serve a start index
    // older than this: the server answers "Too late" and those samples are
    // permanently gone. Measured at ~8.0 s on both streams of SpikeGLX
    // v20251218 (binary-searched how far back FETCH still succeeds). The
    // 0.75 factor keeps the resync target comfortably inside whatever the
    // real depth is on a given build, instead of aiming at the very edge
    // where it would immediately age out again.
    const long long ringDepthSamples     = static_cast<long long>( 8.0 * sampleRate );
    const long long resyncBackoffSamples = static_cast<long long>( 0.75 * ringDepthSamples );

    // Reused across chunks so the fan-out below never allocates inside the
    // fetch loop. Reserve one chunk's worst case: every unit firing once.
    std::vector<livewire::WireRecord> wireBatch;
    wireBatch.reserve( static_cast<size_t>( filterBank_.nUnits ) );

    // Reused across chunks for the HOT queue's batched push -- one lock
    // acquisition per chunk on SpikeQueue rather than one per spike (see
    // SpikeQueue.h). Also handed by value into AnalysisFeed::Chunk::spikes
    // when the SLOW side is wired, so the analysis thread does not have to
    // re-derive which detections belong to which chunk.
    std::vector<SpikeEvent> spikeBatch;
    spikeBatch.reserve( static_cast<size_t>( filterBank_.nUnits ) );

    StreamAccountant acct;
    // Backlog costs an extra round-trip to ask for, so sample it
    // periodically rather than every chunk -- this loop runs at several
    // hundred Hz (see the latency logs referenced in the commit history).
    const long long backlogPollEveryNChunks = 200;

    t_ull fromCt = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );

    while( !stopFlag_.load() ) {

        cppClient_sglx_fetch io;
        io.channel_subset = &channelSubset[0];
        io.n_cs           = nFetchChans;
        io.max_samps      = static_cast<int>( chunkSamples );
        io.downsample     = 1;
        io.js             = kImecJs;
        io.ip             = kImecIp;

        t_ull headCt = sglx_fetch( io, hSglx_, fromCt );

        if( headCt == 0 ) {
            // A failed fetch is NOT retryable from the same fromCt. The
            // dominant cause is "Too late": fromCt has aged out of the
            // server's ~8 s ring, and every future attempt at that same
            // index fails identically while the stream keeps advancing.
            // An earlier version of this loop slept and continue'd without
            // touching fromCt, so the first time this thread fell 8 s
            // behind it wedged permanently -- stderr forever, zero
            // detections for the rest of the session, and nothing in the
            // output saying so. Jump forward into a live part of the ring
            // instead, and count what was skipped as lost rather than
            // letting the record imply the stream was contiguous.
            std::cerr << "ImecFetchThreadCpu: sglx_fetch error: " << sglx_getError( hSglx_ ) << "\n";

            t_ull serverHead = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );
            if( serverHead == 0 ) {
                // Cannot even ask where the stream is -- server down or run
                // stopped. Back off and retry; nothing to record, since we
                // have no idea where we would be resyncing TO.
                std::this_thread::sleep_for( std::chrono::milliseconds( fetchChunkMs_ ) );
                continue;
            }

            t_ull resumeAt = ( static_cast<long long>( serverHead ) > resyncBackoffSamples )
                             ? serverHead - static_cast<t_ull>( resyncBackoffSamples )
                             : 0;

            if( resumeAt > fromCt ) {
                std::cerr << "ImecFetchThreadCpu: resyncing " << fromCt << " -> " << resumeAt
                          << " (" << (resumeAt - fromCt) << " samples lost)\n";
                acct.noteResync( static_cast<long long>( resumeAt ) );
                fromCt = resumeAt;
            }
            else {
                // Failed with fromCt still inside the ring -- a transient
                // (network hiccup, run paused). Leave fromCt alone so no
                // data is skipped, and just retry.
                ++acct.nFetchErrors;
            }

            std::this_thread::sleep_for( std::chrono::milliseconds( fetchChunkMs_ ) );
            continue;
        }

        long long tpts = static_cast<long long>( io.data.size() ) / nFetchChans;
        if( tpts <= 0 ) {
            // No new samples yet. Sleep a fraction of a chunk period rather
            // than spinning: with no pause at all this loop fires a fresh
            // TCP round-trip immediately, which on past runs drove it to
            // several hundred fetches/second for a median of 60 samples
            // each -- CPU and server time spent asking for data that
            // physically is not there yet. A full chunk period would be too
            // coarse (it would add up to fetchChunkMs of avoidable latency
            // to a spike arriving just after we looked), so wait a quarter.
            fromCt = headCt;
            std::this_thread::sleep_for(
                std::chrono::microseconds( std::max( 250, fetchChunkMs_ * 250 ) ) );
            continue;
        }
        if( tpts > maxChunkSamples ) {
            // Server returned more than fetchChunkMs's worth (e.g. after
            // this thread stalled) -- process only what fits this engine's
            // preallocated buffers, remaining data is picked up next
            // iteration since fromCt only advances by what was consumed.
            tpts = maxChunkSamples;
        }

        // Recorded BEFORE processing and with the post-clamp tpts, so the
        // accounting reflects what actually reaches the pipeline. A nonzero
        // gap means the server could not serve our requested start and
        // silently began us later; this is the only place that fact is
        // observable at all.
        long long gap = acct.noteFetch( static_cast<long long>( headCt ), tpts );

        // ---- Drift-schedule filter swaps --------------------------------
        // Time is taken from the STREAM clock (samples since the first
        // successful fetch), not wall clock. A fetch stall or a resync must
        // not slide the schedule relative to the data it was computed for.
        //
        // ASSUMPTION, and it is a real one: the schedule's t_s is "seconds
        // from the calibration recording's start", and this maps it onto
        // "seconds since this run's first fetch". That is only right if the
        // live session begins at the same point in the drift trajectory the
        // calibration recording did. It is an open-loop plan replayed on a
        // timer -- the closed-loop version is the live tracker feeding
        // measured motion back, which is not wired yet.
        if( driftSchedule_ && !driftSchedule_->events.empty()
            && acct.firstSampleIndex >= 0 ) {

            double nowS = static_cast<double>(
                static_cast<long long>( headCt ) - acct.firstSampleIndex ) / sampleRate;

            while( const DriftSchedule::Event *ev = driftSchedule_->next( nowS ) ) {
                if( ev->unitIndex < 0 || ev->unitIndex >= filterBank_.nUnits ) {
                    std::cerr << "ImecFetchThreadCpu: drift event at t=" << ev->t_s
                              << "s names unit index " << ev->unitIndex
                              << ", out of range for a " << filterBank_.nUnits
                              << "-unit bank -- skipped\n";
                    continue;
                }

                std::vector<int32_t> translated( ev->channels.size() );
                bool ok = true;
                for( size_t k = 0; k < ev->channels.size(); ++k ) {
                    int idx = MultiFilterBank::carGroupIndexOf( carChannelIds,
                                                                 ev->channels[k] );
                    if( idx < 0 ) {
                        std::cerr << "ImecFetchThreadCpu: drift event at t=" << ev->t_s
                                  << "s for unit index " << ev->unitIndex
                                  << " uses channel " << ev->channels[k]
                                  << ", not part of the CAR channel group -- skipped\n";
                        ok = false;
                        break;
                    }
                    translated[k] = static_cast<int32_t>( idx );
                }
                if( !ok )
                    continue;

                // Through the ENGINE, not the bank. MultiConvolutionEngine
                // widens the taps to double at construction, so writing the
                // bank alone would leave the copy that actually scores
                // untouched -- a swap that silently does nothing.
                engine.updateUnit( ev->unitIndex, translated, ev->filter,
                                    ev->threshold );
                ++nSwapsApplied_;
            }
        }
        if( gap > 0 ) {
            std::cerr << "ImecFetchThreadCpu: stream gap of " << gap
                      << " samples before headCt=" << headCt << "\n";
        }

        if( chunkIndex % backlogPollEveryNChunks == 0 ) {
            t_ull serverHead = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );
            if( serverHead != 0 )
                acct.noteBacklog( static_cast<long long>( serverHead ) );
        }

        for( long long t = 0; t < tpts; ++t ) {
            const short *src = &io.data[static_cast<size_t>( t ) * nFetchChans];
            short       *dst = &hostCarChunk[static_cast<size_t>( t ) * nCarChans];
            for( int c = 0; c < nCarChans; ++c )
                dst[c] = src[c];

            short     syValue = src[nFetchChans - 1];
            long long syIdx   = static_cast<long long>( headCt ) + t;
            int       syBit   = extractBit( syValue, imecSyncBit_ );
            syncTracker.update( syBit, syIdx );

            // Cost when disabled is one predictable branch per sample, which
            // does not register against the per-sample channel copy directly
            // above it; when enabled it adds a shift/mask and a few integer
            // compares. Neither is a syscall, an allocation, or a lock, so
            // this stays clear of the rule that removed per-chunk flushes
            // from this loop.
            if( syllableFromSy_.enabled ) {
                int       code     = 0;
                long long onsetIdx = 0;
                if( syllableDecoder.update( syValue, syIdx, code, onsetIdx )
                    && syncTracker.hasEdge() ) {

                    SyllableEvent ev;
                    ev.code         = code;
                    ev.sampleIndex  = onsetIdx;
                    ev.timeRelSyncS = syncTracker.secondsSinceLastEdge( onsetIdx );

                    if( syllableFromSy_.queue )
                        syllableFromSy_.queue->push( ev );

                    if( syllableTimesFile.is_open() )
                        syllableTimesFile << ev.code << "," << ev.sampleIndex << "\n";

                    if( publisher_ ) {
                        // kStreamImec, not kStreamNi: these codes are on the
                        // IMEC clock, and mislabelling that would put the
                        // viewer back into a cross-stream alignment guess
                        // for events that need none.
                        publisher_->publish( livewire::makeRecord(
                            livewire::kWireSyllable, livewire::kStreamImec,
                            ev.code, ev.sampleIndex, 0.0f,
                            static_cast<float>( ev.timeRelSyncS ) ) );
                    }
                }
            }
        }

        auto t0 = std::chrono::steady_clock::now();
        // Preprocess (highpass + CAR over the FULL group), then score every
        // unit.
        std::vector<double> preprocessed =
            preprocessor.processChunk( hostCarChunk, static_cast<size_t>( tpts ) );
        std::vector<MultiPeakEvent> detections = engine.processChunk(
            preprocessed.data(), static_cast<size_t>( tpts ),
            static_cast<long long>( headCt ) );
        auto t1 = std::chrono::steady_clock::now();

        // Neither file is flushed per chunk any more. At the several-
        // hundred-chunks-per-second this loop really runs at, a flush per
        // chunk is a syscall in the sample-critical path (and on this
        // machine a OneDrive-synced write). Both are flushed once at
        // shutdown below, and ofstream's own buffering bounds what is at
        // risk meanwhile -- mainAllUnits.cpp's Ctrl+C path goes through
        // stop()/join(), so it reaches that shutdown flush normally.
        if( latencyLogFile.is_open() ) {
            double latencyMs = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
            latencyLogFile << chunkIndex << "," << tpts << "," << latencyMs << "\n";
        }
        ++chunkIndex;

        if( !detections.empty() ) {

            // Two reusable buffers, filled per chunk and each handed off in
            // one batched call -- the publisher's mutex and the HOT queue's
            // mutex are each taken once per chunk rather than once per spike.
            wireBatch.clear();
            spikeBatch.clear();

            for( size_t d = 0; d < detections.size(); ++d ) {

                int unitId = filterBank_.hostUnitIds[detections[d].unitIndex];

                if( spikeTimesFile.is_open() ) {
                    spikeTimesFile << unitId << "," << detections[d].sampleIndex
                                    << "," << detections[d].score << "\n";
                }

                // timeRelSyncS is what DecisionThread compares against a
                // syllable's own; leave it 0 until the first sync edge and
                // let DecisionThread see a spike it cannot place rather than
                // throwing a startup exception out of the fetch loop.
                double relS = syncTracker.hasEdge()
                              ? syncTracker.secondsSinceLastEdge( detections[d].sampleIndex )
                              : 0.0;

                if( spikeQueue_ ) {
                    SpikeEvent ev;
                    ev.timeRelSyncS = relS;
                    ev.sampleIndex  = detections[d].sampleIndex;
                    ev.unitId       = unitId;   // real Kilosort cluster id -- see Events.h
                    spikeBatch.push_back( ev );
                }

                if( publisher_ ) {
                    wireBatch.push_back( livewire::makeRecord(
                        livewire::kWireSpike, livewire::kStreamImec,
                        unitId, detections[d].sampleIndex,
                        detections[d].score, static_cast<float>( relS ) ) );
                }
            }

            // HOT path: one lock, one batched push. This is the only place
            // spikeQueue_ is touched -- see SpikeQueue.h for why per-spike
            // pushes here would have been the same std::deque-allocation
            // trap RingDeque.h documents.
            if( spikeQueue_ && !spikeBatch.empty() )
                spikeQueue_->push( spikeBatch.data(), spikeBatch.size() );

            if( publisher_ && !wireBatch.empty() )
                publisher_->publish( &wireBatch[0], wireBatch.size() );
        }
        else {
            spikeBatch.clear();
        }

        // SLOW path: hand the preprocessed chunk to the analysis feed,
        // independent of whether it carried any detections -- drift
        // estimation needs the bulk samples regardless. acquire() returning
        // nullptr means the analysis thread is behind; that chunk is simply
        // skipped here (see AnalysisFeed.h's ownership contract), never
        // waited or retried, so this can never add latency to the hot path
        // above it.
        if( analysisFeed_ ) {
            int poolIndex = -1;
            float *dst = analysisFeed_->acquire( poolIndex );
            if( dst ) {
                const size_t n = static_cast<size_t>( tpts ) * nCarChans;
                const double *src = preprocessed.data();
                for( size_t i = 0; i < n; ++i )
                    dst[i] = static_cast<float>( src[i] );

                AnalysisFeed::Chunk c;
                c.samples          = dst;
                c.nSamples         = static_cast<int>( tpts );
                c.nChannels        = nCarChans;
                c.firstSampleIndex = static_cast<long long>( headCt );
                c.timeRelSyncS     = syncTracker.hasEdge()
                                      ? syncTracker.secondsSinceLastEdge(
                                            static_cast<long long>( headCt ) )
                                      : 0.0;
                c.poolIndex        = poolIndex;
                c.spikes           = spikeBatch;   // copy -- spikeBatch is reused next chunk

                analysisFeed_->publish( std::move( c ) );
            }
        }

        fromCt = headCt + static_cast<t_ull>( tpts );
    }

    // Final backlog reading, so a run that ended while behind says so.
    {
        t_ull serverHead = sglx_getStreamSampleCount( hSglx_, kImecJs, kImecIp );
        if( serverHead != 0 )
            acct.noteBacklog( static_cast<long long>( serverHead ) );
    }

    spikeTimesFile.flush();
    if( latencyLogFile.is_open() )
        latencyLogFile.flush();
    if( syllableTimesFile.is_open() )
        syllableTimesFile.flush();

    if( driftSchedule_ && !driftSchedule_->events.empty() ) {
        // cursor is how far next() walked, i.e. how many events came DUE.
        // Anything due but not applied was rejected above and already said
        // why. Reporting the shortfall as "not yet due" would have hidden a
        // rejected swap behind a benign-sounding message -- which it did,
        // on the first live run.
        const long long nDue     = static_cast<long long>( driftSchedule_->cursor );
        const long long nTotal   = static_cast<long long>( driftSchedule_->events.size() );
        const long long nSkipped = nDue - nSwapsApplied_;
        std::cout << "\nDriftSchedule: " << nSwapsApplied_ << " of " << nTotal
                   << " swaps applied";
        if( nSkipped > 0 )
            std::cout << ", " << nSkipped << " REJECTED (see errors above)";
        if( nDue < nTotal )
            std::cout << ", " << ( nTotal - nDue ) << " never came due";
        std::cout << "\n";
    }
    std::cout << "\n" << acct.summary( "IMEC", sampleRate );
    if( !acct.balanced() || acct.nDropped > 0 ) {
        std::cerr << "ImecFetchThreadCpu: WARNING -- this run did not process every "
                     "sample the server produced (see the summary above).\n";
    }

    // Both queues' shutdown summaries, so a run that silently dropped
    // detections or fell behind on the slow side says so rather than
    // leaving it to be inferred. spikeQueue_/analysisFeed_ are owned by
    // main and outlive this thread, so it is safe to read their counters
    // here after the loop has stopped producing into them.
    if( spikeQueue_ )
        std::cout << spikeQueue_->summary() << "\n";
    if( analysisFeed_ )
        std::cout << analysisFeed_->summary() << "\n";
}
