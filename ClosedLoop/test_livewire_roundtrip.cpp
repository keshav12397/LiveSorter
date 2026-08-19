// Round-trips EventPublisher -> LiveWireClient over a real loopback socket,
// with both halves in one process, and checks the three properties the fetch
// loop actually depends on:
//
//   1. Fidelity. Every record a reader receives is byte-identical to the one
//      published, and they arrive in order. This is what makes the shared
//      LiveWire.h header worth having; a packing or endian mistake shows up
//      here rather than as a raster full of nonsense.
//
//   2. publish() never blocks. Timed with the reader deliberately stalled,
//      so the socket's send buffer is full and the publisher thread is
//      parked in its WSAEWOULDBLOCK wait. If publish() were on the socket's
//      side of that, this is where the fetch loop would lose samples --
//      which is the whole reason EventPublisher is built the way it is.
//
//   3. Overflow drops the OLDEST and says so. A viewer that cannot keep up
//      must cost viewer data and nothing else, and the loss must be counted
//      rather than silent (same principle as StreamAccountant).
//
// Build (from the repo root, with the MSVC env set -- see README):
//   cl.exe /EHsc /std:c++17 /Fe:TestLiveWireRoundtrip.exe ^
//       ClosedLoop\test_livewire_roundtrip.cpp ClosedLoop\EventPublisher.cpp ^
//       Viewer\LiveWireClient.cpp /I ClosedLoop /I Viewer

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <numeric>

#include "EventPublisher.h"
#include "LiveWireClient.h"

namespace {

const int kTestPort = 4199;   // not the default, so this never fights a real run

int g_failures = 0;

void check( bool cond, const std::string &what )
{
    std::cout << ( cond ? "  ok   " : "  FAIL " ) << what << "\n";
    if( !cond )
        ++g_failures;
}

// Distinct, reproducible content per record, so an ordering or aliasing bug
// cannot pass by accident.
livewire::WireRecord makeTestRecord( int i )
{
    return livewire::makeRecord(
        livewire::kWireSpike, livewire::kStreamImec,
        /*a*/ 1000 + i,
        /*sampleIndex*/ 500000000LL + i * 7LL,
        /*b*/ static_cast<float>( i ) * 0.25f,
        /*timeRelSyncS*/ static_cast<float>( i ) * 1e-4f );
}

// Waits for a condition rather than sleeping a guessed interval, so this
// test does not become flaky on a loaded machine.
template<typename F>
bool waitUntil( F pred, int timeoutMs )
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds( timeoutMs );
    while( std::chrono::steady_clock::now() < deadline ) {
        if( pred() )
            return true;
        std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
    }
    return pred();
}

} // namespace


int main()
{
    std::vector<int> unitIds;
    for( int i = 0; i < 157; ++i )
        unitIds.push_back( i * 3 + 1 );   // not 0..n-1, so an index/id mix-up shows

    // -------------------------------------------------------------------
    // 1. Fidelity and ordering.
    // -------------------------------------------------------------------
    std::cout << "session header + record fidelity\n";
    {
        EventPublisher pub( kTestPort, unitIds, 30000.0, 25000.0, /*imecSy=*/true, 65536 );
        if( !pub.start() ) {
            std::cerr << "publisher failed to start -- is port " << kTestPort << " in use?\n";
            return 2;
        }

        LiveWireClient cli;
        if( !cli.connect( "127.0.0.1", kTestPort ) ) {
            std::cerr << "client failed to connect: " << cli.lastError() << "\n";
            return 2;
        }

        check( cli.header().nUnits == 157, "header nUnits" );
        check( cli.header().imecSampleRateHz == 30000.0, "header imec rate" );
        check( cli.header().niSampleRateHz == 25000.0, "header ni rate" );
        check( cli.header().syllableSourceIsImecSy == 1, "header syllableSource flag" );
        check( cli.unitIds() == unitIds, "unit id table survives the wire" );

        // publish() drops while no client is attached (by design), and the
        // publisher thread only notices the accept on its next pass.
        check( waitUntil( [&]{ return pub.hasClient(); }, 2000 ), "publisher sees the client" );

        const int kN = 20000;
        for( int i = 0; i < kN; ++i )
            pub.publish( makeTestRecord( i ) );

        std::vector<livewire::WireRecord> got;
        waitUntil( [&]{ cli.poll( got ); return static_cast<int>( got.size() ) >= kN; }, 10000 );

        check( static_cast<int>( got.size() ) == kN, "all records arrived" );

        bool identical = ( static_cast<int>( got.size() ) == kN );
        for( int i = 0; identical && i < kN; ++i ) {
            livewire::WireRecord expect = makeTestRecord( i );
            identical = ( std::memcmp( &expect, &got[i], sizeof(expect) ) == 0 );
            if( !identical )
                std::cout << "    first mismatch at record " << i << "\n";
        }
        check( identical, "every record byte-identical and in order" );
        check( pub.nDropped() == 0, "nothing dropped when the reader keeps up" );
    }

    // -------------------------------------------------------------------
    // 2. publish() does not block on a stalled reader.
    // -------------------------------------------------------------------
    std::cout << "publish() latency with the reader stalled\n";
    {
        // Small ring on purpose: this forces the overflow path to be the
        // thing being timed, not an unused branch.
        EventPublisher pub( kTestPort, unitIds, 30000.0, 0.0, false, 4096 );
        if( !pub.start() )
            return 2;

        LiveWireClient cli;
        if( !cli.connect( "127.0.0.1", kTestPort ) )
            return 2;
        waitUntil( [&]{ return pub.hasClient(); }, 2000 );

        // The client never polls below, so the socket's send buffer fills
        // and the publisher thread parks in its WSAEWOULDBLOCK wait.
        const int kN = 200000;
        std::vector<livewire::WireRecord> batch( 64 );

        std::vector<double> batchUs;
        batchUs.reserve( kN / 64 + 1 );

        auto t0 = std::chrono::steady_clock::now();
        for( int i = 0; i < kN; i += 64 ) {
            for( int j = 0; j < 64; ++j )
                batch[j] = makeTestRecord( i + j );

            auto b0 = std::chrono::steady_clock::now();
            pub.publish( &batch[0], batch.size() );
            batchUs.push_back( std::chrono::duration<double, std::micro>(
                                    std::chrono::steady_clock::now() - b0 ).count() );
        }
        double totalMs = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0 ).count();

        std::vector<double> sorted( batchUs );
        std::sort( sorted.begin(), sorted.end() );
        double medianUs = sorted[sorted.size() / 2];
        double p999Us   = sorted[( sorted.size() * 999 ) / 1000];
        double worstBatchUs = sorted.back();

        // Reported as a distribution, not just a worst case: the worst is an
        // OS scheduling outlier and quoting it alone would misrepresent what
        // the fetch loop actually pays per chunk.
        std::cout << "    " << kN << " records in " << totalMs << " ms ("
                   << ( totalMs * 1000.0 / kN ) << " us/record)\n"
                   << "    per 64-record publish(): median " << medianUs
                   << " us, p99.9 " << p999Us << " us, worst " << worstBatchUs << " us\n";

        // A whole fetch chunk is 5 ms and carries at most a few hundred
        // records. Any single publish() call taking longer than a
        // millisecond would mean the socket had reached back into the fetch
        // loop; the real measured figure is orders of magnitude under this.
        check( worstBatchUs < 1000.0, "no single publish() call approaches a fetch chunk" );

        // -------------------------------------------------------------------
        // 3. Overflow drops, and is counted.
        // -------------------------------------------------------------------
        std::cout << "overflow accounting\n";
        check( pub.nDropped() > 0, "a stalled reader causes counted drops" );
        check( pub.nPublished() == kN, "published count is what the producer offered" );

        // The reader, once it resumes, must still see a valid record stream
        // (drop-oldest keeps record alignment; a partial-record drop would
        // desynchronise it permanently).
        std::vector<livewire::WireRecord> got;
        waitUntil( [&]{ cli.poll( got ); return got.size() > 1000; }, 5000 );
        check( !got.empty(), "reader resumes and receives records" );

        // Drop-oldest means `a` may JUMP -- that is the drop, not a bug. What
        // must hold is that it never goes backwards and that every record
        // still reconstructs exactly, which is what byte misalignment would
        // destroy. A reader that had slipped by a few bytes would read
        // fields out of neighbouring records and fail both.
        bool aligned = true;
        for( size_t i = 0; i < got.size(); ++i ) {
            livewire::WireRecord expect = makeTestRecord( got[i].a - 1000 );
            if( std::memcmp( &expect, &got[i], sizeof(expect) ) != 0
                || ( i > 0 && got[i].a <= got[i - 1].a ) ) {
                aligned = false;
                std::cout << "    alignment lost at " << i << ": a=" << got[i].a << "\n";
                break;
            }
        }
        check( aligned, "record alignment and content survive a drop burst" );

        // And the drop must be a clean excision of whole records, not a
        // gradual leak -- if it were, the surviving stream would show many
        // small gaps rather than the few large ones a ring wrap produces.
        int nGaps = 0;
        for( size_t i = 1; i < got.size(); ++i ) {
            if( got[i].a != got[i - 1].a + 1 )
                ++nGaps;
        }
        std::cout << "    " << got.size() << " records received across " << nGaps
                   << " gap(s)\n";

        std::cout << "    " << pub.summary();
    }

    std::cout << ( g_failures == 0 ? "PASS\n" : "FAIL\n" );
    return g_failures == 0 ? 0 : 1;
}
