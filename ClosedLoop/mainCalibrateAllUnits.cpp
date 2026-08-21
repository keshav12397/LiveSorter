// =================================
// mainCalibrateAllUnits: C++ batch calibration driver. Ties together, for
// every qualifying Kilosort unit in one run:
//   KilosortReader/SglxMetaReader  -- ground truth + meta
//   OfflinePreprocessor            -- .bin -> scratch (once)
//   LcmvFit.cpp                    -- mean waveform, channel selection,
//                                     interferer pick, then the LCMV solve
//   NoiseCovariance.cpp            -- per-unit space-time covariance
//   OfflineScorer.cpp              -- threshold-sweep scoring, ONE
//                                     streaming pass for every unit
//   ThresholdSweep.cpp             -- per-unit best-F1 pick
//
// Mirrors FilterGen/calibrate_all_units.py's candidate selection, fit
// sequence, and output format exactly (see that file's docstring) -- this
// is a port of its control flow, not a re-derivation; every numerical piece
// it calls into was already validated against Python independently (see
// each header's equivalence-test comment).
//
// Output (in --out-dir / outDir config key), same as calibrate_all_units.py:
//   unit_ids.bin, channels.bin, filters.bin, thresholds.bin, summary.csv
// =================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <chrono>


#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "Config.h"
#include "SglxMetaReader.h"
#include "KilosortReader.h"
#include "OfflinePreprocessor.h"
#include "ScratchMemmap.h"
#include "LcmvFit.h"
#include "NoiseCovariance.h"
#include "OfflineScorer.h"
#include "ThresholdSweep.h"

namespace {

struct UnitCandidate {
    long long id;
    long long spikeCount;
};

// Same rationale/logic as main.cpp's exeDir()/resolveRelativeToExe() and
// mainAllUnits.cpp's own copy -- duplicated deliberately (see that file's
// header comment), each of this project's executables is independent.
std::string exeDir()
{
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA( NULL, buf, MAX_PATH );
    if( n == 0 || n == MAX_PATH )
        return ".";
    std::string path( buf, n );
    size_t slash = path.find_last_of( "\\/" );
    return (slash == std::string::npos) ? "." : path.substr( 0, slash );
#else
    return ".";
#endif
}

std::string resolveRelativeToExe( const std::string &path )
{
    if( path.empty() )
        return path;
    bool isAbsolute =
        (path.size() >= 2 && path[1] == ':') ||
        path[0] == '/' || path[0] == '\\';
    if( isAbsolute )
        return path;
    return exeDir() + "\\" + path;
}

std::string deriveMetaPath( const std::string &binPath )
{
    size_t dot = binPath.find_last_of( '.' );
    return (dot == std::string::npos) ? binPath + ".meta" : binPath.substr( 0, dot ) + ".meta";
}

// One candidate's fit result, once its cheap prep + LCMV solve succeed --
// everything scoreAllUnitsOffline / ThresholdSweep need next.
struct FitResult {
    long long unitId;
    std::vector<int> selCarGroupIdx;   // N indices into the CAR group (0..nCarCh-1) -- NOT raw ids
    std::vector<float> filterTFxC;     // [templateLength * N], row-major [t][c] -- matches filters.bin's layout
    std::vector<long long> targetTestSpikesRel; // held-out spikes, relative to splitT, sorted
};

} // namespace


int main( int argc, char **argv )
{
    std::cout << std::unitbuf;

    if( argc < 2 ) {
        std::cerr << "Usage: " << argv[0] << " <config.txt>\n";
        return 1;
    }

    try {
        auto t0 = std::chrono::steady_clock::now();
        auto tStage = t0;
        auto lap = [&]( const char *label ) {
            auto now = std::chrono::steady_clock::now();
            std::cout << "  [" << label << ": "
                       << std::chrono::duration<double>( now - tStage ).count() << "s]\n";
            tStage = now;
        };

        Config cfg = Config::load( argv[1] );

        std::string ksDir          = resolveRelativeToExe( cfg.requireString( "ksDir" ) );
        std::string binPath        = resolveRelativeToExe( cfg.requireString( "binPath" ) );
        std::string metaPath       = cfg.getString( "metaPath", "" );
        metaPath = metaPath.empty() ? deriveMetaPath( binPath ) : resolveRelativeToExe( metaPath );
        std::string channelMapJson = resolveRelativeToExe( cfg.requireString( "channelMapJson" ) );
        std::string outDir         = resolveRelativeToExe( cfg.requireString( "outDir" ) );
        std::string scratchPath    = cfg.getString( "scratchPath", "" );
        scratchPath = scratchPath.empty() ? (outDir + "/scratch_preprocessed.f32") : resolveRelativeToExe( scratchPath );

#ifdef _WIN32
        CreateDirectoryA( outDir.c_str(), nullptr ); // ok if it already exists
#endif

        int    nChannels       = cfg.getInt( "nChannels", 5 );
        int    templateLength  = cfg.getInt( "templateLength", 61 );
        int    templateOffset  = cfg.getInt( "templateOffset", 20 );
        double trainFrac       = cfg.getDouble( "trainFrac", 0.5 );
        double ridge           = cfg.getDouble( "ridge", 1e-3 );
        int    maxSpikes       = cfg.getInt( "maxSpikes", 2000 );
        bool   applyHighpass   = cfg.getBool( "applyHighpass", true );
        double highpassCutoffHz = cfg.getDouble( "highpassCutoffHz", 300.0 );
        int    nThresholds     = cfg.getInt( "nThresholds", 60 );
        int    autoInterferers = cfg.getInt( "autoInterferers", 5 );
        long long minSpikes    = cfg.getInt( "minSpikes", 200 );
        int    maxUnits        = cfg.getInt( "maxUnits", 0 );
        unsigned int seed      = static_cast<unsigned int>( cfg.getInt( "seed", 0 ) );
        int    preprocessChunkSamples = cfg.getInt( "preprocessChunkSamples", 2000000 );
        int    scoreChunkSamples      = cfg.getInt( "scoreChunkSamples", 2000 );
        int    detectionCapacityCfg   = cfg.getInt( "detectionCapacity", 0 );

        // ---- Ground truth + meta ---------------------------------------------
        std::cout << "Loading Kilosort ground truth from " << ksDir << "...\n";
        KilosortData ks = KilosortData::load( ksDir );
        std::cout << "  " << ks.spikeTimes.size() << " spikes\n";

        std::vector<int> carChannelIds = loadChanMapJson( channelMapJson );
        std::map<int, std::pair<double, double> > positions = loadChanMapPositions( channelMapJson );
        SglxMeta meta = SglxMeta::load( metaPath );
        const int nCarCh = static_cast<int>( carChannelIds.size() );

        // ---- Preprocess the FULL recording once, shared across every unit --
        std::cout << "Preprocessing (causal highpass + CAR) -> " << scratchPath << "...\n";
        long long nSamplesTotal = streamPreprocessToScratch(
            binPath, metaPath, carChannelIds, applyHighpass, highpassCutoffHz,
            /*maxSamples=*/0, preprocessChunkSamples, scratchPath );
        std::cout << "  " << nSamplesTotal << " samples ("
                   << (nSamplesTotal / meta.sampleRateHz) << "s)\n";
        lap( "preprocess" );

        long long splitT = static_cast<long long>( trainFrac * nSamplesTotal );
        double testDurationSeconds = (nSamplesTotal - splitT) / meta.sampleRateHz;
        std::cout << "Train: [0, " << splitT << ")   Test: [" << splitT << ", " << nSamplesTotal << ")\n";

        // Two views onto the same scratch file: `fullView` for
        // auto_pick_interferers_spatial (which Python runs against the FULL
        // recording, not just the train split -- it's only picking a peak
        // channel, not fitting anything), `trainView` (bounded to splitT) for
        // mean_waveform/select_channels inside fit_lcmv, matching Python's
        // data_train slice exactly (see FilterGen/threshold_sweep_real.py's
        // fit_lcmv -- it's called with data_train, not data).
        ScratchMemmap fullView( scratchPath, nSamplesTotal, nCarCh );
        ScratchMemmap trainView( scratchPath, splitT, nCarCh );

        // ---- Candidate unit list --------------------------------------------
        // Same policy as calibrate_all_units.py: highest spike count first
        // (stable/reproducible --max-units cap), excluding noise-labeled and
        // <minSpikes clusters.
        std::map<long long, long long> counts;
        for( size_t i = 0; i < ks.spikeClusters.size(); ++i )
            ++counts[ks.spikeClusters[i]];

        std::vector<UnitCandidate> byCount;
        byCount.reserve( counts.size() );
        for( std::map<long long, long long>::const_iterator it = counts.begin(); it != counts.end(); ++it )
            byCount.push_back( UnitCandidate{ it->first, it->second } );
        std::sort( byCount.begin(), byCount.end(),
                   []( const UnitCandidate &a, const UnitCandidate &b ) { return a.spikeCount > b.spikeCount; } );

        std::vector<UnitCandidate> candidates;
        for( size_t i = 0; i < byCount.size(); ++i ) {
            std::map<long long, std::string>::const_iterator labIt = ks.labels.find( byCount[i].id );
            if( labIt != ks.labels.end() ) {
                std::string lab = labIt->second;
                std::transform( lab.begin(), lab.end(), lab.begin(), ::tolower );
                if( lab == "noise" )
                    continue;
            }
            if( byCount[i].spikeCount < minSpikes )
                continue;
            candidates.push_back( byCount[i] );
            if( maxUnits > 0 && static_cast<int>( candidates.size() ) >= maxUnits )
                break;
        }
        std::cout << candidates.size() << " candidate units (of " << counts.size()
                   << " total clusters, excluding noise-labeled and <" << minSpikes << "-spike clusters)\n";

        // Per-cluster spike time lookup (built once, reused per candidate and
        // per interferer -- same O(nSpikes) cost calibrate_all_units.py's own
        // spike_t[spike_cl==id] boolean-mask gather pays per call, just
        // grouped up front instead of re-scanning per lookup).
        std::map<long long, std::vector<long long> > spikesByCluster;
        for( size_t i = 0; i < ks.spikeClusters.size(); ++i )
            spikesByCluster[ks.spikeClusters[i]].push_back( ks.spikeTimes[i] );
        for( std::map<long long, std::vector<long long> >::iterator it = spikesByCluster.begin();
             it != spikesByCluster.end(); ++it )
            std::sort( it->second.begin(), it->second.end() );

        // ---- Per-candidate cheap prep (CPU): interferer pick, mean
        // waveform, channel selection -- must run BEFORE the batched
        // covariance kernel, since that kernel needs each unit's already-
        // selected channels + target/interferer spike mask as input (see
        // NoiseCovariance.h's note on per-unit exclusion).
        struct PrepResult {
            bool ok;
            std::string failReason;
            long long unitId;
            std::vector<long long> interfererIds;
            std::vector<int> sel; // N indices into the CAR group
            std::vector<double> sFlat;                          // [N*templateLength], channel-major [c][t]
            std::vector<std::vector<double> > interfererFlats;  // each [N*templateLength], channel-major [c][t]
            long long nTrainSpikes;
        };

        std::vector<PrepResult> prep( candidates.size() );

        std::cout << "Per-unit prep (interferer pick, mean waveform, channel selection)...\n";
        for( size_t u = 0; u < candidates.size(); ++u ) {

            PrepResult &pr = prep[u];
            pr.ok = false;
            pr.unitId = candidates[u].id;
            unsigned int unitSeed = seed + static_cast<unsigned int>( candidates[u].id );

            try {
                pr.interfererIds = autoPickInterferersSpatial(
                    fullView, spikesByCluster, ks.labels, carChannelIds, pr.unitId, autoInterferers,
                    templateLength, templateOffset, positions, unitSeed );
                if( pr.interfererIds.empty() )
                    throw std::runtime_error( "no interferers found nearby" );

                const std::vector<long long> &targetSpikesAll = spikesByCluster[pr.unitId];
                std::vector<long long> targetTrain;
                for( size_t i = 0; i < targetSpikesAll.size(); ++i )
                    if( targetSpikesAll[i] < splitT )
                        targetTrain.push_back( targetSpikesAll[i] );

                long long nUsed = 0;
                std::vector<double> targetWf = meanWaveform(
                    trainView, targetTrain, templateLength, templateOffset, maxSpikes, unitSeed, nUsed );
                // Total train-split spike count, matching Python's summary.csv
                // n_train_spikes (len(target_train)) -- NOT meanWaveform's nUsed,
                // which is the maxSpikes-subsampled count actually averaged.
                pr.nTrainSpikes = static_cast<long long>( targetTrain.size() );

                std::vector<std::vector<double> > interfererWfs;
                for( size_t k = 0; k < pr.interfererIds.size(); ++k ) {
                    const std::vector<long long> &spAll = spikesByCluster[pr.interfererIds[k]];
                    std::vector<long long> spTrain;
                    for( size_t i = 0; i < spAll.size(); ++i )
                        if( spAll[i] < splitT )
                            spTrain.push_back( spAll[i] );
                    long long nUsedInt = 0;
                    interfererWfs.push_back( meanWaveform(
                        trainView, spTrain, templateLength, templateOffset, maxSpikes, unitSeed, nUsedInt ) );
                }

                pr.sel = selectChannels( targetWf, interfererWfs, nCarCh, templateLength, nChannels );
                if( static_cast<int>( pr.sel.size() ) != nChannels )
                    throw std::runtime_error( "only " + std::to_string( pr.sel.size() ) +
                                               " channels available, need " + std::to_string( nChannels ) );

                // Gather down to the selected channels, channel-major [c][t]
                // flat order -- matches Python's s.T.ravel() (s = target_wf[:, sel],
                // shape (templateLength, N); s.T is (N, templateLength); ravel()
                // is then channel-major) and NoiseCovariance's assembleR dim
                // convention (both must agree, or the LCMV solve mixes up
                // which entries of R correspond to which entries of s_flat).
                pr.sFlat.resize( static_cast<size_t>( nChannels ) * templateLength );
                for( int c = 0; c < nChannels; ++c )
                    for( int t = 0; t < templateLength; ++t )
                        pr.sFlat[static_cast<size_t>( c ) * templateLength + t] =
                            targetWf[static_cast<size_t>( t ) * nCarCh + pr.sel[c]];

                pr.interfererFlats.resize( interfererWfs.size() );
                for( size_t k = 0; k < interfererWfs.size(); ++k ) {
                    pr.interfererFlats[k].resize( static_cast<size_t>( nChannels ) * templateLength );
                    for( int c = 0; c < nChannels; ++c )
                        for( int t = 0; t < templateLength; ++t )
                            pr.interfererFlats[k][static_cast<size_t>( c ) * templateLength + t] =
                                interfererWfs[k][static_cast<size_t>( t ) * nCarCh + pr.sel[c]];
                }

                pr.ok = true;
            }
            catch( const std::exception &e ) {
                pr.failReason = e.what();
            }
        }

        int nPrepOk = 0;
        for( size_t u = 0; u < prep.size(); ++u )
            if( prep[u].ok )
                ++nPrepOk;
        std::cout << "  " << nPrepOk << "/" << candidates.size() << " units prepped successfully.\n";
        lap( "per-unit prep" );

        // ---- Batched noise covariance across every prepped unit
        // (see NoiseCovariance.h). Reads the scratch memmap in place -- the
        // The whole train split is memory-mapped rather than copied,
        // which for a 30-minute session was several GB and put a hard ceiling
        // on session length that no longer exists.

        std::vector<int> unitChannelsFlat;
        std::vector<SpikeFreeSegment> segmentsFlat;
        std::vector<int> segmentOffsets, segmentCounts;
        std::vector<size_t> prepIdxForCovUnit; // maps covariance-batch index -> prep[] index

        for( size_t u = 0; u < prep.size(); ++u ) {
            if( !prep[u].ok )
                continue;

            for( int c = 0; c < nChannels; ++c )
                unitChannelsFlat.push_back( prep[u].sel[c] );

            std::vector<long long> localSpikeTimes;
            localSpikeTimes.insert( localSpikeTimes.end(),
                spikesByCluster[prep[u].unitId].begin(), spikesByCluster[prep[u].unitId].end() );
            for( size_t k = 0; k < prep[u].interfererIds.size(); ++k )
                localSpikeTimes.insert( localSpikeTimes.end(),
                    spikesByCluster[prep[u].interfererIds[k]].begin(),
                    spikesByCluster[prep[u].interfererIds[k]].end() );
            std::sort( localSpikeTimes.begin(), localSpikeTimes.end() );

            std::vector<SpikeFreeSegment> segs = findSpikeFreeSegments(
                localSpikeTimes, splitT, templateLength, templateOffset );

            segmentOffsets.push_back( static_cast<int>( segmentsFlat.size() ) );
            segmentCounts.push_back( static_cast<int>( segs.size() ) );
            segmentsFlat.insert( segmentsFlat.end(), segs.begin(), segs.end() );

            prepIdxForCovUnit.push_back( u );
        }

        int nCovUnits = static_cast<int>( prepIdxForCovUnit.size() );
        std::cout << "Running batched noise covariance for " << nCovUnits << " units...\n";
        std::vector<std::vector<double> > Rmats = computeNoiseCovarianceBatched(
            trainView.row( 0 ), splitT, nCarCh, templateLength, nChannels,
            unitChannelsFlat, segmentsFlat, segmentOffsets, segmentCounts, nCovUnits );
        lap( "batched noise covariance" );

        // ---- LCMV solve: per unit -------------------------------------------
        std::cout << "Solving LCMV filter for each unit...\n";
        std::vector<FitResult> fits;
        std::vector<std::string> fitFailReason( prep.size() );

        for( int i = 0; i < nCovUnits; ++i ) {
            size_t u = prepIdxForCovUnit[i];
            PrepResult &pr = prep[u];

            try {
                std::vector<double> fFlat = lcmvFilter( pr.sFlat, pr.interfererFlats, Rmats[i], ridge );

                FitResult fr;
                fr.unitId = pr.unitId;
                fr.selCarGroupIdx = pr.sel;
                fr.filterTFxC.resize( static_cast<size_t>( templateLength ) * nChannels );
                for( int c = 0; c < nChannels; ++c )
                    for( int t = 0; t < templateLength; ++t )
                        fr.filterTFxC[static_cast<size_t>( t ) * nChannels + c] =
                            static_cast<float>( fFlat[static_cast<size_t>( c ) * templateLength + t] );

                const std::vector<long long> &targetSpikesAll = spikesByCluster[pr.unitId];
                for( size_t i2 = 0; i2 < targetSpikesAll.size(); ++i2 )
                    if( targetSpikesAll[i2] >= splitT )
                        fr.targetTestSpikesRel.push_back( targetSpikesAll[i2] - splitT );
                std::sort( fr.targetTestSpikesRel.begin(), fr.targetTestSpikesRel.end() );

                if( fr.targetTestSpikesRel.empty() )
                    throw std::runtime_error( "no held-out test spikes" );

                fits.push_back( fr );
            }
            catch( const std::exception &e ) {
                fitFailReason[u] = e.what();
                prep[u].ok = false; // demote -- excluded from scoring/packed output below
            }
        }
        std::cout << "  " << fits.size() << "/" << candidates.size() << " units fit successfully.\n";
        lap( "LCMV solve" );

        // ---- Threshold-sweep scoring, ONE streaming
        // pass over the held-out test split for every fit unit at once.
        std::vector<UnitPeaks> allPeaks;
        if( !fits.empty() ) {
            std::vector<int> unitIdsInt;
            std::vector<int32_t> unitChannelsInCarGroup;
            std::vector<float> unitFilters;
            for( size_t i = 0; i < fits.size(); ++i ) {
                unitIdsInt.push_back( static_cast<int>( fits[i].unitId ) );
                for( int c = 0; c < nChannels; ++c )
                    unitChannelsInCarGroup.push_back( fits[i].selCarGroupIdx[c] );
                unitFilters.insert( unitFilters.end(), fits[i].filterTFxC.begin(), fits[i].filterTFxC.end() );
            }

            int minSep = templateLength / 2;
            int detectionCapacity = detectionCapacityCfg > 0 ? detectionCapacityCfg :
                std::max( 20000, static_cast<int>( fits.size() ) * ( scoreChunkSamples / std::max( 1, minSep ) + 1 ) * 2 );

            std::cout << "Scoring " << fits.size() << " units over the held-out test split "
                       << "(one streaming pass, detectionCapacity=" << detectionCapacity << ")...\n";
            allPeaks = scoreAllUnitsOffline(
                binPath, metaPath, carChannelIds, applyHighpass, highpassCutoffHz,
                unitIdsInt, unitChannelsInCarGroup, unitFilters, templateLength, nChannels,
                splitT, nSamplesTotal - splitT, scoreChunkSamples, minSep, detectionCapacity );
        }
        lap( "batched offline scoring" );

        // ---- Threshold sweep per unit, pick best F1 ---------------------------
        struct SummaryRow {
            long long unitId;
            int nChannelsOut;
            std::vector<int> selChannelsRaw;
            double threshold, recall, precision, f1, fpRateHz;
            long long nTrainSpikes, nTestSpikes;
            std::string status;
            bool packed;
        };
        std::vector<SummaryRow> summaryRows;

        std::vector<long long> packedIds;
        std::vector<int32_t> packedChannels; // raw SpikeGLX ids
        std::vector<float> packedFilters;
        std::vector<float> packedThresholds;

        std::cout << "Sweeping thresholds...\n";
        for( size_t i = 0; i < fits.size(); ++i ) {
            const FitResult &fr = fits[i];

            SummaryRow row;
            row.unitId = fr.unitId;
            row.nChannelsOut = nChannels;
            row.selChannelsRaw.resize( nChannels );
            for( int c = 0; c < nChannels; ++c )
                row.selChannelsRaw[c] = carChannelIds[fr.selCarGroupIdx[c]];
            row.nTestSpikes = static_cast<long long>( fr.targetTestSpikesRel.size() );
            row.packed = false;
            row.status = "failed";

            for( size_t u = 0; u < prep.size(); ++u )
                if( prep[u].ok && prep[u].unitId == fr.unitId ) { row.nTrainSpikes = prep[u].nTrainSpikes; break; }

            const UnitPeaks &peaks = allPeaks[i];
            ThresholdSweepResult sweep = sweepThresholds(
                peaks.sampleIdx, peaks.score, fr.targetTestSpikesRel,
                templateLength / 4, nThresholds, testDurationSeconds );

            row.threshold = sweep.bestThreshold;
            row.recall = sweep.recall;
            row.precision = sweep.precision;
            row.f1 = sweep.f1;
            row.fpRateHz = sweep.fpRateHz;
            row.status = "ok";
            row.packed = true;

            packedIds.push_back( fr.unitId );
            for( int c = 0; c < nChannels; ++c )
                packedChannels.push_back( row.selChannelsRaw[c] );
            packedFilters.insert( packedFilters.end(), fr.filterTFxC.begin(), fr.filterTFxC.end() );
            packedThresholds.push_back( static_cast<float>( sweep.bestThreshold ) );

            std::cout << "unit " << fr.unitId << ": threshold=" << sweep.bestThreshold
                       << "  recall=" << sweep.recall << "  precision=" << sweep.precision
                       << "  f1=" << sweep.f1 << "\n";

            summaryRows.push_back( row );
        }
        lap( "threshold sweep" );

        // Failed units (prep or fit stage) go into the summary too, matching
        // calibrate_all_units.py's summary.csv (one row per attempted unit).
        for( size_t u = 0; u < candidates.size(); ++u ) {
            bool alreadyListed = false;
            for( size_t i = 0; i < summaryRows.size(); ++i )
                if( summaryRows[i].unitId == candidates[u].id ) { alreadyListed = true; break; }
            if( alreadyListed )
                continue;

            SummaryRow row;
            row.unitId = candidates[u].id;
            row.nChannelsOut = 0;
            row.threshold = row.recall = row.precision = row.f1 = row.fpRateHz = 0.0;
            row.nTrainSpikes = row.nTestSpikes = 0;
            row.packed = false;
            row.status = "failed: " + (!prep[u].failReason.empty() ? prep[u].failReason :
                                        (!fitFailReason[u].empty() ? fitFailReason[u] : "unknown"));
            summaryRows.push_back( row );
        }

        // ---- Write outputs ------------------------------------------------------
        std::cout << packedIds.size() << "/" << candidates.size() << " units fit successfully.\n";

        if( !packedIds.empty() ) {
            std::vector<int32_t> unitIdsOut;
            unitIdsOut.reserve( packedIds.size() );
            for( size_t i = 0; i < packedIds.size(); ++i )
                unitIdsOut.push_back( static_cast<int32_t>( packedIds[i] ) );
            std::ofstream( outDir + "/unit_ids.bin", std::ios::binary )
                .write( reinterpret_cast<const char*>( unitIdsOut.data() ), unitIdsOut.size() * sizeof(int32_t) );
            std::ofstream( outDir + "/channels.bin", std::ios::binary )
                .write( reinterpret_cast<const char*>( packedChannels.data() ), packedChannels.size() * sizeof(int32_t) );
            std::ofstream( outDir + "/filters.bin", std::ios::binary )
                .write( reinterpret_cast<const char*>( packedFilters.data() ), packedFilters.size() * sizeof(float) );
            std::ofstream( outDir + "/thresholds.bin", std::ios::binary )
                .write( reinterpret_cast<const char*>( packedThresholds.data() ), packedThresholds.size() * sizeof(float) );
            std::cout << "Wrote unit_ids.bin (" << unitIdsOut.size() << "), channels.bin, filters.bin, "
                       << "thresholds.bin to " << outDir << "\n";
        }
        else {
            std::cerr << "No units succeeded -- not writing packed .bin files.\n";
        }

        std::ofstream summary( outDir + "/summary.csv" );
        summary << "unit_id,n_channels,sel_channels,threshold,recall,precision,f1,fp_rate_hz,"
                   "n_train_spikes,n_test_spikes,status\n";
        for( size_t i = 0; i < summaryRows.size(); ++i ) {
            const SummaryRow &row = summaryRows[i];
            summary << row.unitId << ",";
            if( row.packed ) {
                summary << row.nChannelsOut << ",\"";
                for( size_t c = 0; c < row.selChannelsRaw.size(); ++c )
                    summary << (c ? " " : "") << row.selChannelsRaw[c];
                summary << "\"," << row.threshold << "," << row.recall << "," << row.precision << ","
                        << row.f1 << "," << row.fpRateHz << "," << row.nTrainSpikes << "," << row.nTestSpikes;
            }
            else {
                summary << ",,,,,,,,";
            }
            summary << "," << row.status << "\n";
        }
        std::cout << "Wrote " << outDir << "/summary.csv\n";

        auto t1 = std::chrono::steady_clock::now();
        double elapsedSec = std::chrono::duration<double>( t1 - t0 ).count();
        std::cout << "Total wall time: " << elapsedSec << "s\n";

        return packedIds.empty() ? 1 : 0;
    }
    catch( const std::exception &e ) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
