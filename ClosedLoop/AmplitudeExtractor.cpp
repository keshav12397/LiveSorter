#include "AmplitudeExtractor.h"

#include <cmath>
#include <algorithm>
#include <unordered_map>


std::vector<livewire::WireRecord> extractAmplitudeRecords(
    const AnalysisFeed::Chunk &chunk,
    const MultiFilterBank &filterBank,
    int templateOffset )
{
    std::vector<livewire::WireRecord> out;
    if( chunk.spikes.empty() || chunk.nSamples <= 0 || chunk.nChannels <= 0 )
        return out;

    out.reserve( chunk.spikes.size() * static_cast<size_t>( filterBank.nChannelsPerUnit ) );

    // Kilosort cluster id -> filter-bank unit index. Chunks carry "a few
    // dozen" spikes (AnalysisFeed.h) against up to ~157 units, so this map
    // is built once per chunk rather than once per spike.
    std::unordered_map<int, int> unitIndexOf;
    unitIndexOf.reserve( filterBank.hostUnitIds.size() * 2 );
    for( size_t i = 0; i < filterBank.hostUnitIds.size(); ++i )
        unitIndexOf[filterBank.hostUnitIds[i]] = static_cast<int>( i );

    const int nCh    = chunk.nChannels;
    const int nSamp  = chunk.nSamples;
    const int nChPer = filterBank.nChannelsPerUnit;
    const int tLen   = filterBank.templateLength;

    for( size_t si = 0; si < chunk.spikes.size(); ++si ) {
        const SpikeEvent &spk = chunk.spikes[si];

        std::unordered_map<int, int>::const_iterator it = unitIndexOf.find( spk.unitId );
        if( it == unitIndexOf.end() )
            continue;   // unit not in this bank (e.g. single-target -1 id) -- nothing to extract
        int u = it->second;

        // Window in chunk-relative sample coordinates.
        long long winStartAbs = spk.sampleIndex - templateOffset;
        long long relStart64  = winStartAbs - chunk.firstSampleIndex;
        if( relStart64 < 0 )
            continue;   // window starts before this chunk -- skip, don't stitch
        int relStart = static_cast<int>( relStart64 );
        if( relStart + tLen > nSamp )
            continue;   // window runs past this chunk's end -- skip, don't stitch

        const int32_t *chans = filterBank.unitChannels( u );

        for( int c = 0; c < nChPer; ++c ) {
            int32_t ch = chans[c];
            if( ch < 0 || ch >= nCh )
                continue;   // defensive: an untranslated/out-of-range channel id must not read OOB

            float peak = 0.0f;
            for( int t = 0; t < tLen; ++t ) {
                float v = chunk.samples[ static_cast<size_t>( relStart + t ) * nCh + ch ];
                float av = std::fabs( v );
                if( av > peak )
                    peak = av;
            }

            out.push_back( livewire::makeRecord(
                livewire::kWireAmpChannel, livewire::kStreamImec,
                /*a*/ spk.unitId,
                /*sampleIndex*/ spk.sampleIndex,
                /*b*/ peak,
                /*timeRelSyncS*/ static_cast<float>( spk.timeRelSyncS ),
                /*c*/ c ) );
        }
    }

    return out;
}
