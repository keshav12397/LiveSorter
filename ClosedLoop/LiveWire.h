#ifndef CLOSEDLOOP_LIVEWIRE_H
#define CLOSEDLOOP_LIVEWIRE_H

#include <cstdint>
#include <cstring>

// The ClosedLoop -> SpikeViewer wire format, declared exactly once.
//
// Both ends include THIS header. That is the whole point of the file: a
// viewer that re-derives the layout from a written spec is a second
// implementation of the same thing, and this repo has twice paid for one of
// those (Calibration.cpp's fit/sweep port, and the two validation scripts'
// duplicated `% wrap_samples` alignment -- see live_tracking_bug_report.md).
// If a field changes here, both sides change together or neither compiles.
//
// Transport is a plain TCP stream on localhost: a fixed-size SessionHeader,
// then `nUnits` int32 Kilosort cluster ids, then an unbounded run of
// fixed-size WireRecords. Everything is little-endian host order, which is
// exact rather than lazy: producer and consumer are both x86-64 processes on
// the same machine by construction (the publisher binds to loopback only).
//
// Fixed-size records with no framing header, on purpose. The reader can
// resynchronise by byte count alone and there is no length field to get
// wrong; the cost is that every record carries a few fields it does not use,
// which at these rates is nothing (32 B/record, and a 157-unit run produces
// on the order of 10^4 records/s worst case = ~300 kB/s over loopback).

namespace livewire {

const uint16_t kProtocolVersion = 1;
const uint16_t kDefaultPort     = 4143;   // 4142 is SpikeGLX's own; +1 is ours

// What a record describes.
enum WireType : uint8_t {
    kWireSpike    = 1,  // one detected spike, one unit
    kWireSyllable = 2,  // one debounce-confirmed syllable-code onset
    kWireTrial    = 3,  // one closed syllable trial + whether it fired
    kWireLine     = 4   // the DO line changed state (live indicator)
};

// Which stream's clock `sampleIndex` is counted in. Records from different
// streams are NOT directly comparable by sampleIndex -- see the note on
// timeRelSyncS below and README.md's "Cross-stream time alignment".
enum WireStream : uint8_t {
    kStreamImec = 0,
    kStreamNi   = 1,
    kStreamNone = 2   // kWireLine: a host-side event, on no acquisition clock
};

#pragma pack(push, 1)

// Sent once, immediately after accept, before any record. Followed on the
// wire by int32 unitIds[nUnits] -- the real Kilosort cluster ids, in the
// filter bank's own order, so the viewer can lay out a raster row per unit
// before a single spike arrives (and so a unit that never fires still gets
// a row instead of silently not existing).
struct SessionHeader {
    char     magic[4];          // 'L','C','M','V'
    uint16_t version;           // kProtocolVersion
    uint16_t nUnits;
    double   imecSampleRateHz;  // 0 if unknown at connect time
    double   niSampleRateHz;    // 0 when syllableSource=imecSy (no NI thread)
    uint8_t  syllableSourceIsImecSy;  // 1 = SY-channel test path, 0 = production NI
    uint8_t  reserved[7];       // keeps the header 8-byte aligned and leaves
                                 // room to add a field without a version bump
                                 // reading as garbage on an older viewer
};

// 32 bytes. The `a`/`b`/`c` fields are deliberately generic and documented
// per type here rather than being a union of named structs: a union would
// need every arm to be exactly the same size anyway, and the reader has to
// switch on `type` either way.
struct WireRecord {
    uint8_t  type;          // WireType
    uint8_t  stream;        // WireStream
    uint16_t reserved;
    int32_t  a;             // Spike: Kilosort unit id
                             // Syllable: code 1-7
                             // Trial: code 1-7
                             // Line: 1 = went high, 0 = went low
    int64_t  sampleIndex;   // absolute sample index in `stream`'s clock.
                             // Trial: the *syllable's* index, so a trial row
                             // joins to its Syllable record by equality.
                             // Line: 0 (no acquisition clock -- see kStreamNone)
    float    b;             // Spike: detector score
                             // Trial: spikes counted in the window
                             // else: 0
    float    timeRelSyncS;  // seconds since THIS stream's most recent sync
                             // edge -- the same quantity DecisionThread
                             // compares across streams, and subject to the
                             // same caveat (valid while the window of
                             // interest is well under the sync period; see
                             // README.md). Records from one stream should be
                             // compared by sampleIndex instead, which is
                             // exact and never wraps.
    int32_t  c;             // Trial: 1 if the DO line was raised for this
                             // trial, 0 if the window closed without firing
                             // else: 0
    uint32_t reserved2;
};

#pragma pack(pop)

// A stream of records is only interpretable if these hold; both ends static
// assert rather than trusting that two compilers agreed about packing.
static_assert( sizeof(SessionHeader) == 32, "SessionHeader must be 32 bytes" );
static_assert( sizeof(WireRecord) == 32, "WireRecord must be 32 bytes" );

inline void fillMagic( SessionHeader &h )
{
    h.magic[0] = 'L'; h.magic[1] = 'C'; h.magic[2] = 'M'; h.magic[3] = 'V';
}

inline bool magicOk( const SessionHeader &h )
{
    return h.magic[0] == 'L' && h.magic[1] == 'C'
        && h.magic[2] == 'M' && h.magic[3] == 'V';
}

inline WireRecord makeRecord( WireType type, WireStream stream, int32_t a,
                               long long sampleIndex, float b, float timeRelSyncS,
                               int32_t c = 0 )
{
    WireRecord r;
    std::memset( &r, 0, sizeof(r) );
    r.type         = static_cast<uint8_t>( type );
    r.stream       = static_cast<uint8_t>( stream );
    r.a            = a;
    r.sampleIndex  = static_cast<int64_t>( sampleIndex );
    r.b            = b;
    r.timeRelSyncS = timeRelSyncS;
    r.c            = c;
    return r;
}

} // namespace livewire

#endif // CLOSEDLOOP_LIVEWIRE_H
