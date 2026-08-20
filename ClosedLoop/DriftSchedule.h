#ifndef CLOSEDLOOP_DRIFTSCHEDULE_H
#define CLOSEDLOOP_DRIFTSCHEDULE_H

#include <string>
#include <vector>
#include <cstdint>

// Loads drift_schedule.bin, written by FilterGen/calibrate_drift_aware.py's
// write_schedule() -- one filter-swap event per later segment of every
// drifting unit that got more than one segment. Units that never drift, or
// units run through calibrate_all_units.py (no drift awareness at all),
// simply have an empty or absent schedule.
//
// This is intentionally NOT wired into ImecFetchThreadGPU.cu here. That file
// is being actively extended by another agent (NI/decision/socket code) and
// this project's own README explains why a from-scratch rewrite of a
// validated fetch loop is exactly the kind of change that has cost real
// debugging time before. What's here is the minimal, self-contained piece:
// load the schedule, and hand back events whose time has come. Wiring it in
// is a ~5-line addition at the top of ImecFetchThreadGPU::fetchLoop()'s per-
// chunk loop:
//
//     double nowS = <elapsed seconds since the fetch loop's t=0, same
//                     origin the calibration recording used>;
//     while( const DriftSchedule::Event *ev = schedule.next( nowS ) ) {
//         std::vector<int32_t> translated = translateToCarGroup( ev->channels, ... );
//         filterBank.updateFilters( ev->unitIndex, translated, ev->filter, ev->threshold );
//     }
//
// left for whoever owns that file next, since the channel-translation step
// (raw SpikeGLX id -> CAR-group buffer index) has to reuse whatever helper
// that file already has for the startup translation GpuFilterBank.h
// documents (see its d_channels comment) -- duplicating that mapping here
// would be exactly the kind of second implementation this codebase has
// already been burned by twice (see generate_filter.py's module docstring).
struct DriftSchedule {
    struct Event {
        double t_s;                        // seconds from calibration recording start
        int32_t unitIndex;                 // index into GpuFilterBank's unit arrays (NOT Kilosort cluster id)
        std::vector<int32_t> channels;      // raw SpikeGLX channel ids, length nChannelsPerUnit -- caller must translate
        std::vector<float> filter;          // length templateLength * nChannelsPerUnit, row-major [t][c]
        float threshold;
    };

    int version;
    int templateLength;
    int nChannelsPerUnit;
    std::vector<Event> events;  // ascending by t_s, exactly as written

    // Index of the next not-yet-applied event, i.e. how far `next()` has
    // walked. Exposed so a caller can persist/restore progress if it ever
    // needs to (not currently used by anything in this repo).
    size_t cursor;

    DriftSchedule() : version( 0 ), templateLength( 0 ), nChannelsPerUnit( 0 ), cursor( 0 ) {}

    // Reads drift_schedule.bin. An absent file is not an error -- it means
    // "no drift-aware schedule for this filter set" -- and yields an empty
    // schedule (events.empty()), so a caller can unconditionally construct
    // one and only act on it if non-empty. Throws std::runtime_error only on
    // a malformed (present but corrupt/truncated) file.
    static DriftSchedule load( const std::string &dir );

    // Returns a pointer to the next event whose t_s <= nowS, advancing the
    // internal cursor past it, or nullptr if there is none yet (or none
    // left). Call in a `while` loop at each chunk boundary -- more than one
    // event can become due in a single chunk if the fetch loop fell behind
    // or a unit has closely-spaced segments. Events are walked forward only,
    // once, in time order (matches how calibrate_drift_aware.py wrote them,
    // and matches this project's other one-pass, no-seeking event streams,
    // e.g. NiFetchThread's syllable events) -- never re-applies a past event.
    const Event *next( double nowS );
};

#endif // CLOSEDLOOP_DRIFTSCHEDULE_H
