// Scores SyllableDecoder against a make_sim_session.py session's own
// ground truth, by streaming the SY channel out of the .bin exactly the way
// ImecFetchThreadGPU streams it out of the live server.
//
// Why this test rather than a live run
// ------------------------------------
// The syllableSource=imecSy path exists because the NI stream on this rig
// cannot be simulated. That same fact means the path cannot be checked
// end-to-end against a live server either -- the codes only exist in the
// synthetic file. Reading the file directly is therefore the ONLY place the
// decode can be scored against a right answer, and it is a good place: the
// decoder consumes one short and one sample index at a time and has no other
// input, so feeding it from a file and feeding it from sglx_fetch are the
// same thing to it.
//
// What is NOT covered: the fetch loop's chunking, the sync tracker, the
// queue hand-off, DecisionThread. Those are shared with the already-exercised
// production path.
//
// The chunk loop below deliberately uses an awkward, non-round chunk size.
// Live chunks are whatever the server felt like returning -- a past run had
// 31% of its chunks under 60 samples and 1,059 of them a single sample -- and
// a decoder with per-chunk state would pass at 30000 and fail there.
//
// Usage:
//   test_syllable_decoder_sy.exe <session_dir>
// e.g. D:/sim_validate
//
// Build (from the repo root, with the MSVC env set -- see README):
//   cl.exe /EHsc /std:c++17 /Fe:TestSyllableDecoderSy.exe ^
//       ClosedLoop\test_syllable_decoder_sy.cpp /I ClosedLoop

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "SyllableDecoder.h"

namespace {

// make_sim_session.py's fixed geometry, also stated in each session's
// README.txt: 384 AP + 1 SY, int16, time-major.
const int  kNChans      = 385;
const int  kSyChanIndex = 384;
const int  kSyncBit     = 6;    // real IMEC firmware convention, not a choice
const int  kCodeStartBit = 0;   // make_sim_session.py's SYLLABLE_BITS
const int  kCodeWidth    = 3;

// Same default the config carries, and what NiFetchThread has always used.
const int  kDebounceSamples = 10;

struct TruthEvent {
    long long onset;
    long long offset;
    int       code;
};

std::vector<TruthEvent> loadTruth( const std::string &path )
{
    std::vector<TruthEvent> out;
    std::ifstream fh( path.c_str() );
    if( !fh.is_open() ) {
        std::cerr << "could not open " << path << "\n";
        std::exit( 2 );
    }

    std::string line;
    std::getline( fh, line );   // header: onset_sample,offset_sample,code

    while( std::getline( fh, line ) ) {
        if( line.empty() )
            continue;
        std::istringstream ss( line );
        std::string a, b, c;
        std::getline( ss, a, ',' );
        std::getline( ss, b, ',' );
        std::getline( ss, c, ',' );
        TruthEvent e;
        e.onset  = std::atoll( a.c_str() );
        e.offset = std::atoll( b.c_str() );
        e.code   = std::atoi( c.c_str() );
        out.push_back( e );
    }
    return out;
}

} // namespace


int main( int argc, char **argv )
{
    if( argc < 2 ) {
        std::cerr << "Usage: " << argv[0] << " <session_dir>\n";
        return 1;
    }

    std::string dir      = argv[1];
    std::string binPath  = dir + "/sim_g0_t0.imec0.ap.bin";
    std::string csvPath  = dir + "/sim_syllables.csv";

    std::vector<TruthEvent> truth = loadTruth( csvPath );
    std::cout << "ground truth: " << truth.size() << " syllable events\n";

    std::FILE *fh = std::fopen( binPath.c_str(), "rb" );
    if( !fh ) {
        std::cerr << "could not open " << binPath << "\n";
        return 2;
    }

    SyllableDecoder decoder( kCodeStartBit, kCodeWidth, kDebounceSamples );

    struct Detected { long long onset; int code; };
    std::vector<Detected> detected;

    // Non-round on purpose -- see the file header. 1499 samples is 49.97 ms,
    // and shares no factor with the debounce length or with any syllable
    // duration, so code transitions land at every possible chunk phase.
    const long long kChunkSamples = 1499;

    std::vector<short> buf( static_cast<size_t>( kChunkSamples ) * kNChans );
    long long sampleIndex = 0;
    long long nSyncEdges  = 0;
    int       lastSyncBit = 0;
    bool      haveSyncBit = false;

    while( true ) {
        size_t got = std::fread( &buf[0], sizeof(short), buf.size(), fh );
        long long n = static_cast<long long>( got / kNChans );
        if( n <= 0 )
            break;

        for( long long t = 0; t < n; ++t ) {

            short sy = buf[static_cast<size_t>( t ) * kNChans + kSyChanIndex];

            // Counted only as a sanity check that this really is the SY
            // channel and that bit 6 carries what the README says it does --
            // a wrong channel index would decode plausible-looking garbage
            // from AP noise otherwise.
            int syncBit = ( static_cast<unsigned short>( sy ) >> kSyncBit ) & 1;
            if( haveSyncBit && lastSyncBit == 0 && syncBit == 1 )
                ++nSyncEdges;
            lastSyncBit = syncBit;
            haveSyncBit = true;

            int       code     = 0;
            long long onsetIdx = 0;
            if( decoder.update( sy, sampleIndex, code, onsetIdx ) ) {
                Detected d;
                d.onset = onsetIdx;
                d.code  = code;
                detected.push_back( d );
            }
            ++sampleIndex;
        }
    }
    std::fclose( fh );

    std::cout << "read " << sampleIndex << " samples ("
               << ( sampleIndex / 30000.0 ) << " s)\n";
    std::cout << "sync rising edges on bit " << kSyncBit << ": " << nSyncEdges
               << "  (expect ~1 per second)\n";
    std::cout << "decoded " << detected.size() << " syllable onsets\n";

    // Matching. The decoder reports the FIRST sample of the held run, so a
    // correct onset is exactly the truth onset -- not "within a tolerance".
    // The only expected offset is zero, which is worth asserting rather than
    // hiding under a window: a systematic +debounceSamples error is exactly
    // the kind of thing a generous tolerance would let through, and this
    // project has already been burned by a hand-derived index correction
    // that was silently wrong (see README's detected-spike time convention).
    size_t    nMatched     = 0;
    size_t    nWrongCode   = 0;
    long long worstOffset  = 0;

    size_t di = 0;
    for( size_t ti = 0; ti < truth.size(); ++ti ) {

        // Truth and detections are both in increasing onset order, so this
        // walks once rather than rescanning.
        while( di < detected.size() && detected[di].onset < truth[ti].onset - 64 )
            ++di;

        if( di < detected.size() && detected[di].onset <= truth[ti].onset + 64 ) {
            long long off = detected[di].onset - truth[ti].onset;
            if( std::abs( off ) > std::abs( worstOffset ) )
                worstOffset = off;
            if( detected[di].code != truth[ti].code ) {
                ++nWrongCode;
                std::cout << "  code mismatch at " << truth[ti].onset
                           << ": truth=" << truth[ti].code
                           << " decoded=" << detected[di].code << "\n";
            }
            else {
                ++nMatched;
            }
            ++di;
        }
        else {
            std::cout << "  MISSED truth onset " << truth[ti].onset
                       << " code " << truth[ti].code << "\n";
        }
    }

    long long nExtra = static_cast<long long>( detected.size() )
                       - static_cast<long long>( nMatched ) - static_cast<long long>( nWrongCode );

    std::cout << "matched          : " << nMatched << " / " << truth.size() << "\n";
    std::cout << "wrong code       : " << nWrongCode << "\n";
    std::cout << "unmatched extras : " << nExtra << "\n";
    std::cout << "worst onset error: " << worstOffset << " samples\n";

    bool pass = ( nMatched == truth.size() ) && ( nWrongCode == 0 )
                && ( nExtra == 0 ) && ( worstOffset == 0 ) && ( nSyncEdges > 0 );

    std::cout << ( pass ? "PASS\n" : "FAIL\n" );
    return pass ? 0 : 1;
}
