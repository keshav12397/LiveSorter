#ifndef CLOSEDLOOP_EVENTS_H
#define CLOSEDLOOP_EVENTS_H

// The two event types that cross thread boundaries via ThreadSafeQueue.
// Both are computed entirely within their producing thread (which owns
// both the raw detection and the relevant SyncEdgeTracker), so no shared
// state is needed just to fill these in -- see README.md.

struct SpikeEvent {
    double     timeRelSyncS;  // seconds since the IMEC stream's most recent SY sync edge
    long long  sampleIndex;   // absolute IMEC stream sample index (for spikeTimes.txt)
    int        unitId;        // Kilosort cluster id. -1 for the single-target pipeline
                               // (ImecFetchThread/DecisionThread never read this field);
                               // a real cluster id when produced by the all-units
                               // pipeline (ImecFetchThreadCpu) -- see MultiFilterBank::hostUnitIds.
};

struct SyllableEvent {
    int        code;          // 1-7 (debounce-confirmed 3-bit code from NI lines 5/6/7)
    double     timeRelSyncS;  // seconds since the NI stream's most recent sync edge
    long long  sampleIndex;   // absolute NI stream sample index (debounce-confirmed onset)
};

#endif // CLOSEDLOOP_EVENTS_H
