#ifndef CLOSEDLOOP_SYLLABLEDECODER_H
#define CLOSEDLOOP_SYLLABLEDECODER_H

#include "DigitalWordUtils.h"

// The debounced 3-bit-syllable-code state machine, lifted verbatim out of
// NiFetchThread::fetchLoop() so the IMEC-SY test path can run the SAME
// decoder rather than a second one written to match.
//
// This is a pure extraction, not a rewrite: the field read, the
// candidate/count/startIdx bookkeeping, the `count == debounceSamples &&
// code != lastEmitted` emit condition, and the emit-only-if-nonzero /
// re-arm-regardless rule are all the original lines in the original order.
// NiFetchThread now calls this instead of carrying its own copy, so there is
// exactly one implementation and the production NI path cannot drift away
// from the path used to test it. (README.md and this repo's history are
// emphatic about the cost of a second implementation of shared logic --
// Calibration.cpp's fit/sweep port and the two validation scripts'
// duplicated alignment both cost long debugging sessions.)
//
// Callers differ only in where the digital word comes from and what its
// sample index means:
//   - NiFetchThread   : NI DW channel, lines 5/6/7 (bits 5/6/7), NI clock.
//   - ImecFetchThreadCpu (syllableSource=imecSy): IMEC SY channel bits 0-2,
//     IMEC clock. Test path only -- see README.md and
//     FilterGen/make_sim_session.py.
class SyllableDecoder {
public:
    // fieldStartBit/fieldWidth read as one multi-bit field starting at the
    // lowest line number (extractField's convention), which is why
    // NiFetchThread requires syllableLines to be contiguous ascending.
    SyllableDecoder( int fieldStartBit, int fieldWidth, int debounceSamples )
        :   fieldStartBit_( fieldStartBit ), fieldWidth_( fieldWidth ),
            debounceSamples_( debounceSamples ),
            candidateCode_( 0 ), candidateCount_( 0 ), candidateStartIdx_( 0 ),
            lastEmittedCode_( 0 )
    {}

    // Feed one digital-word sample at absolute stream index `sampleIndex`.
    // Returns true and fills outCode/outOnsetIndex exactly when the original
    // loop would have built a SyllableEvent -- i.e. a nonzero code has just
    // been held for debounceSamples consecutive samples and differs from the
    // last stabilized code. outOnsetIndex is the FIRST sample of the held
    // run, not the sample that confirmed it.
    bool update( short word, long long sampleIndex, int &outCode, long long &outOnsetIndex )
    {
        int code = extractField( word, fieldStartBit_, fieldWidth_ );

        if( code == candidateCode_ ) {
            ++candidateCount_;
        }
        else {
            candidateCode_     = code;
            candidateCount_    = 1;
            candidateStartIdx_ = sampleIndex;
        }

        if( candidateCount_ == debounceSamples_ && candidateCode_ != lastEmittedCode_ ) {

            bool emit = ( candidateCode_ != 0 );
            int       code_       = candidateCode_;
            long long onsetIndex_ = candidateStartIdx_;

            // Re-arm regardless of whether we emitted (code==0 "rest"
            // stabilizing also updates lastEmittedCode, so a repeated
            // nonzero code after a rest period is treated as new).
            lastEmittedCode_ = candidateCode_;

            if( emit ) {
                outCode        = code_;
                outOnsetIndex  = onsetIndex_;
                return true;
            }
        }

        return false;
    }

private:
    int       fieldStartBit_;
    int       fieldWidth_;
    int       debounceSamples_;

    int       candidateCode_;
    long long candidateCount_;
    long long candidateStartIdx_;
    int       lastEmittedCode_;
};

#endif // CLOSEDLOOP_SYLLABLEDECODER_H
