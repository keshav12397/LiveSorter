#include "MultiConvolutionEngine.h"

#include <algorithm>
#include <stdexcept>

MultiConvolutionEngine::MultiConvolutionEngine( const MultiFilterBank &bank,
                                                int nChannelsGroup,
                                                long long minSeparationSamples,
                                                int nThreads )
    :   bank_( bank ), nUnits_( bank.nUnits ), nChannelsGroup_( nChannelsGroup ),
        minSeparationSamples_( minSeparationSamples )
{
    if( nUnits_ <= 0 )
        throw std::runtime_error( "MultiConvolutionEngine: filter bank has no units" );
    if( nChannelsGroup <= 0 )
        throw std::runtime_error( "MultiConvolutionEngine: nChannelsGroup must be > 0" );

    const int nc = bank_.nChannelsPerUnit;
    const int L  = bank_.templateLength;

    // Every channel index must already be a position within the CAR group --
    // checked here rather than trusted, because an untranslated raw SpikeGLX
    // id is usually still a plausible-looking small integer, so the failure
    // mode is silently filtering the wrong channels rather than crashing.
    for( size_t i = 0; i < bank_.channels.size(); ++i ) {
        int ch = bank_.channels[i];
        if( ch < 0 || ch >= nChannelsGroup_ ) {
            throw std::runtime_error(
                "MultiConvolutionEngine: channel index out of range for the CAR group -- "
                "were bank.channels translated from raw SpikeGLX ids to group positions "
                "before construction? (see MultiFilterBank.h)" );
        }
    }

    engines_.reserve( static_cast<size_t>( nUnits_ ) );
    for( int u = 0; u < nUnits_; ++u ) {
        // ConvolutionEngine works in double; the packed bank stores float.
        // Widening once here, rather than per chunk, is why updateUnit()
        // has to exist: after this the bank's floats are no longer the
        // taps actually being convolved.
        const float *f = bank_.unitFilter( u );
        std::vector<double> taps( static_cast<size_t>( L ) * nc );
        for( size_t i = 0; i < taps.size(); ++i )
            taps[i] = static_cast<double>( f[i] );
        engines_.emplace_back( L, nc, taps, minSeparationSamples );
    }

    gather_.resize( static_cast<size_t>( nUnits_ ) );
    out_.resize( static_cast<size_t>( nUnits_ ) );

    int hw = static_cast<int>( std::thread::hardware_concurrency() );
    if( nThreads <= 0 )
        nThreads = hw > 0 ? hw : 1;
    nThreads = std::max( 1, std::min( nThreads, nUnits_ ) );

    // nThreads-1 background workers: the thread that calls processChunk()
    // runs the same claim loop rather than blocking, so a request for N
    // threads uses N cores instead of N+1 with one of them idle.
    for( int w = 1; w < nThreads; ++w )
        workers_.emplace_back( [this, w]() { workerLoop( w ); } );
}


MultiConvolutionEngine::~MultiConvolutionEngine()
{
    {
        std::lock_guard<std::mutex> lk( mtx_ );
        stop_ = true;
        ++generation_;
    }
    cvStart_.notify_all();
    for( size_t i = 0; i < workers_.size(); ++i ) {
        if( workers_[i].joinable() )
            workers_[i].join();
    }
}


void MultiConvolutionEngine::runUnit( int u )
{
    out_[u].clear();

    std::vector<PeakEvent> peaks;
    if( jobIsFlush_ ) {
        peaks = engines_[u].flush();
    }
    else {
        const int nc = bank_.nChannelsPerUnit;
        const int32_t *chans = bank_.unitChannels( u );

        // The channel gather: slice this unit's nc columns out of the full
        // CAR-group chunk into a contiguous [t*nc+c] buffer, which is the
        // layout ConvolutionEngine expects. This is the CPU counterpart of
        // the GPU kernel's staging into shared memory. Doing it inside the
        // worker (rather than once up front for all units) keeps each
        // unit's small buffer hot in that core's own cache.
        std::vector<double> &g = gather_[u];
        g.resize( jobSamples_ * static_cast<size_t>( nc ) );
        for( size_t t = 0; t < jobSamples_; ++t ) {
            const double *src = jobData_ + t * static_cast<size_t>( nChannelsGroup_ );
            double *dst = &g[t * static_cast<size_t>( nc )];
            for( int c = 0; c < nc; ++c )
                dst[c] = src[chans[c]];
        }

        peaks = engines_[u].processChunk( g.data(), jobSamples_, jobOffset_ );
    }

    const float thr = bank_.thresholds[static_cast<size_t>( u )];
    for( size_t i = 0; i < peaks.size(); ++i ) {
        // >= not >, matching the GPU nmsDecideKernel's comparison, so a
        // threshold sweep's chosen value keeps the same meaning. An
        // all--infinity threshold array therefore reports every
        // NMS-accepted peak, which is what offline scoring relies on.
        if( peaks[i].score >= static_cast<double>( thr ) ) {
            MultiPeakEvent ev;
            ev.unitIndex   = u;
            ev.sampleIndex = peaks[i].sampleIndex;
            ev.score       = static_cast<float>( peaks[i].score );
            out_[u].push_back( ev );
        }
    }
}


void MultiConvolutionEngine::workerLoop( int )
{
    unsigned seen = 0;
    for( ;; ) {
        {
            std::unique_lock<std::mutex> lk( mtx_ );
            cvStart_.wait( lk, [this, &seen]() { return stop_ || generation_ != seen; } );
            if( stop_ )
                return;
            seen = generation_;
        }

        for( ;; ) {
            int u = nextUnit_.fetch_add( 1 );
            if( u >= nUnits_ )
                break;
            runUnit( u );
        }

        {
            std::lock_guard<std::mutex> lk( mtx_ );
            ++nDone_;
        }
        cvDone_.notify_one();
    }
}


void MultiConvolutionEngine::dispatch()
{
    const int nWorkers = static_cast<int>( workers_.size() );
    {
        std::lock_guard<std::mutex> lk( mtx_ );
        nextUnit_.store( 0 );
        nDone_ = 0;
        ++generation_;
    }
    cvStart_.notify_all();

    // The calling thread is a worker too.
    for( ;; ) {
        int u = nextUnit_.fetch_add( 1 );
        if( u >= nUnits_ )
            break;
        runUnit( u );
    }

    // Workers report once, on leaving the claim loop, rather than once per
    // unit: at 157 units and 200 chunks a second, a per-unit lock would be
    // ~31k acquisitions a second of pure contention for no information.
    if( nWorkers > 0 ) {
        std::unique_lock<std::mutex> lk( mtx_ );
        cvDone_.wait( lk, [this, nWorkers]() { return nDone_ >= nWorkers; } );
    }
}


std::vector<MultiPeakEvent> MultiConvolutionEngine::processChunk(
    const double *data, size_t nSamples, long long streamSampleOffset )
{
    std::vector<MultiPeakEvent> merged;
    if( nSamples == 0 )
        return merged;

    jobData_    = data;
    jobSamples_ = nSamples;
    jobOffset_  = streamSampleOffset;
    jobIsFlush_ = false;
    dispatch();

    size_t total = 0;
    for( int u = 0; u < nUnits_; ++u )
        total += out_[u].size();
    merged.reserve( total );
    for( int u = 0; u < nUnits_; ++u )
        merged.insert( merged.end(), out_[u].begin(), out_[u].end() );

    // Sorted so the result cannot depend on worker count or completion
    // order -- see the header's Determinism note.
    std::sort( merged.begin(), merged.end(),
               []( const MultiPeakEvent &a, const MultiPeakEvent &b ) {
                   if( a.sampleIndex != b.sampleIndex )
                       return a.sampleIndex < b.sampleIndex;
                   return a.unitIndex < b.unitIndex;
               } );
    return merged;
}


std::vector<MultiPeakEvent> MultiConvolutionEngine::flush()
{
    jobData_    = nullptr;
    jobSamples_ = 0;
    jobOffset_  = 0;
    jobIsFlush_ = true;
    dispatch();

    std::vector<MultiPeakEvent> merged;
    size_t total = 0;
    for( int u = 0; u < nUnits_; ++u )
        total += out_[u].size();
    merged.reserve( total );
    for( int u = 0; u < nUnits_; ++u )
        merged.insert( merged.end(), out_[u].begin(), out_[u].end() );

    std::sort( merged.begin(), merged.end(),
               []( const MultiPeakEvent &a, const MultiPeakEvent &b ) {
                   if( a.sampleIndex != b.sampleIndex )
                       return a.sampleIndex < b.sampleIndex;
                   return a.unitIndex < b.unitIndex;
               } );
    return merged;
}


void MultiConvolutionEngine::updateUnit( int unitIndex, const std::vector<int32_t> &channels,
                                         const std::vector<float> &filter, float threshold )
{
    if( unitIndex < 0 || unitIndex >= nUnits_ )
        throw std::runtime_error( "MultiConvolutionEngine::updateUnit: unitIndex out of range" );
    for( size_t i = 0; i < channels.size(); ++i ) {
        if( channels[i] < 0 || channels[i] >= nChannelsGroup_ )
            throw std::runtime_error(
                "MultiConvolutionEngine::updateUnit: channel index out of range for the "
                "CAR group -- drift schedule channels must be translated the same way "
                "the bank's were at startup" );
    }

    bank_.updateFilters( unitIndex, channels, filter, threshold );

    // Rebuild the double-precision taps this unit is actually convolved
    // with. ConvolutionEngine has no setTaps(), and adding one would mean
    // touching the file this port exists to leave alone -- so the engine is
    // reconstructed in place instead. That does reset its history, which is
    // the one behavioural difference from the GPU path's in-place
    // cudaMemcpy, and it is bounded: the unit loses its (templateLength-1)
    // samples of context, ~2 ms at 30 kHz, once per swap event. A swap
    // changes which channels the unit reads anyway, so the old history was
    // the wrong data to carry across regardless.
    const int nc = bank_.nChannelsPerUnit;
    const int L  = bank_.templateLength;
    std::vector<double> taps( static_cast<size_t>( L ) * nc );
    for( size_t i = 0; i < taps.size(); ++i )
        taps[i] = static_cast<double>( filter[i] );
    engines_[unitIndex] = ConvolutionEngine( L, nc, taps, minSeparationSamples_ );
}
