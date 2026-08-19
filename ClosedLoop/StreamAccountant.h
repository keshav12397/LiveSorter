#ifndef CLOSEDLOOP_STREAMACCOUNTANT_H
#define CLOSEDLOOP_STREAMACCOUNTANT_H

#include <string>
#include <sstream>
#include <algorithm>

// Per-stream bookkeeping that answers one question at the end of a run:
// "was every sample the server produced either processed by us, or
// explicitly counted as lost?"
//
// Why this exists
// ---------------
// A fetch loop that silently loses samples looks exactly like a broken
// detector, and this project has already spent one long debugging session
// on a symptom (live recall near zero) whose cause was a bookkeeping
// mistake rather than a signal-processing one -- see
// live_tracking_bug_report.md. Anything that can silently drop samples
// gets counted here instead.
//
// Two distinct ways samples go missing, tracked separately because they
// have different causes and different fixes:
//
//   GAP. sglx_fetch returns headCt = the index of the first sample it
//   actually gives us. That is NOT necessarily the `fromCt` we asked for:
//   if the requested start has already aged out of the server's ring
//   buffer, the server starts us later and the samples in between are gone
//   for good. Every such (headCt - fromCt) span is counted as dropped.
//
//   STALL/RESYNC. If the request is so far behind that the server refuses
//   it outright ("Too late"), sglx_fetch fails and the caller must jump
//   `fromCt` forward to something still in the ring. That jump is also a
//   loss, and is counted the same way. (Before this existed, the fetch
//   loop kept retrying the same doomed `fromCt` forever, producing zero
//   detections and an endless stderr spew -- a wedge, not a recovery.)
//
// The ring is not deep: measured ~8.0 s on both the IMEC and NI streams of
// the SpikeGLX build this was written against (v20251218). Falling that far
// behind is unrecoverable by construction, so `maxBacklogSamples` is worth
// watching well before it gets there.
struct StreamAccountant {

    long long firstSampleIndex;   // headCt of the first successful fetch, -1 until then
    long long nextExpectedIndex;  // where a perfectly contiguous stream would resume
    long long nProcessed;         // samples actually handed to the pipeline
    long long nDropped;           // samples the server produced that we never saw
    long long nGaps;              // number of distinct discontinuities
    long long largestGap;
    long long maxBacklogSamples;  // worst observed (server head - our position)
    long long nFetchErrors;

    StreamAccountant()
        :   firstSampleIndex( -1 ), nextExpectedIndex( -1 ), nProcessed( 0 ),
            nDropped( 0 ), nGaps( 0 ), largestGap( 0 ), maxBacklogSamples( 0 ),
            nFetchErrors( 0 )
    {}

    // Call once per successful fetch, BEFORE processing, with the headCt the
    // server returned and the sample count it actually delivered. Returns the
    // size of the gap this fetch revealed (0 when contiguous), so the caller
    // can log the individual event if it wants to.
    long long noteFetch( long long headCt, long long nSamples )
    {
        long long gap = 0;

        if( firstSampleIndex < 0 )
            firstSampleIndex = headCt;
        else if( headCt > nextExpectedIndex ) {
            gap = headCt - nextExpectedIndex;
            nDropped += gap;
            ++nGaps;
            largestGap = std::max( largestGap, gap );
        }
        // headCt < nextExpectedIndex would mean the server handed back data
        // we already consumed. Not a loss, so nothing to count -- the caller
        // is responsible for not double-processing it.

        nProcessed += nSamples;
        nextExpectedIndex = headCt + nSamples;
        return gap;
    }

    // Call when a fetch fails and the caller jumps `fromCt` forward to
    // `resumeAt` to get unstuck. Everything between where we were and where
    // we resume is lost.
    void noteResync( long long resumeAt )
    {
        ++nFetchErrors;
        if( nextExpectedIndex >= 0 && resumeAt > nextExpectedIndex ) {
            long long gap = resumeAt - nextExpectedIndex;
            nDropped += gap;
            ++nGaps;
            largestGap = std::max( largestGap, gap );
        }
        nextExpectedIndex = resumeAt;
    }

    // Call whenever the server's current head is known (it costs an extra
    // round-trip, so callers sample this periodically rather than per chunk).
    void noteBacklog( long long serverHeadIndex )
    {
        if( nextExpectedIndex < 0 )
            return;
        long long backlog = serverHeadIndex - nextExpectedIndex;
        if( backlog > maxBacklogSamples )
            maxBacklogSamples = backlog;
    }

    // Total samples the server produced across the run's span. Every one of
    // these must be either processed or dropped -- that identity is what
    // makes this accounting rather than a set of loose counters, and
    // summary() asserts it.
    long long spanSamples() const
    {
        if( firstSampleIndex < 0 )
            return 0;
        return nextExpectedIndex - firstSampleIndex;
    }

    bool balanced() const { return spanSamples() == nProcessed + nDropped; }

    std::string summary( const std::string &streamName, double sampleRateHz ) const
    {
        std::ostringstream ss;
        ss << streamName << " sample accounting:\n";

        if( firstSampleIndex < 0 ) {
            ss << "  no samples were ever fetched\n";
            return ss.str();
        }

        long long span = spanSamples();
        ss << "  stream span   : [" << firstSampleIndex << ", " << nextExpectedIndex
           << ")  = " << span << " samples";
        if( sampleRateHz > 0 )
            ss << " (" << (span / sampleRateHz) << " s)";
        ss << "\n";
        ss << "  processed     : " << nProcessed << "\n";
        ss << "  dropped       : " << nDropped
           << "  in " << nGaps << " gap(s), largest " << largestGap << "\n";
        ss << "  fetch errors  : " << nFetchErrors << "\n";
        ss << "  max backlog   : " << maxBacklogSamples << " samples";
        if( sampleRateHz > 0 )
            ss << " (" << (maxBacklogSamples / sampleRateHz) << " s)";
        ss << "\n";

        if( balanced() && nDropped == 0 )
            ss << "  RESULT: every sample accounted for, none dropped.\n";
        else if( balanced() )
            ss << "  RESULT: every sample accounted for, but " << nDropped
               << " were LOST (see gaps above).\n";
        else
            ss << "  RESULT: BOOKKEEPING ERROR -- processed+dropped ("
               << (nProcessed + nDropped) << ") != span (" << span
               << "). This is a bug in the fetch loop, not a data problem.\n";

        return ss.str();
    }
};

#endif // CLOSEDLOOP_STREAMACCOUNTANT_H
