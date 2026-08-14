#ifndef CLOSEDLOOP_DIGITALWORDUTILS_H
#define CLOSEDLOOP_DIGITALWORDUTILS_H

// Tiny bit-extraction helpers for SpikeGLX digital-word channels.
//
// Convention (SpikeGLX manual): "the lowest numbered lines... in the lowest
// order bits" -- i.e. bit N of a digital-word sample = physical line N.
// This holds for NI DW channels. It does NOT hold for the IMEC SY channel,
// whose bits are a fixed firmware status word (bit 6 = sync waveform, other
// bits = error flags) -- see README.md. Both bit numbers are still passed
// in as config values here, never hardcoded, specifically because the SY
// convention was wrong once already in this project's own history.

// Extract a single bit (0 or 1) from a raw digital-word sample.
inline int extractBit( short word, int bitIndex )
{
    return (static_cast<unsigned short>( word ) >> bitIndex) & 0x1;
}

// Extract a multi-bit field starting at bitIndex, `nBits` wide, e.g.
// extractField(word, 5, 3) reads bits 5,6,7 as a 3-bit value (0-7) -- used
// for the NI syllable code on lines 5/6/7.
inline int extractField( short word, int bitIndex, int nBits )
{
    unsigned short mask = static_cast<unsigned short>( (1 << nBits) - 1 );
    return (static_cast<unsigned short>( word ) >> bitIndex) & mask;
}

#endif // CLOSEDLOOP_DIGITALWORDUTILS_H
