// =================================
// SpikeViewer: live/offline viewer for the ClosedLoop pipeline.
//
// Two sources, one model. Live, it reads ClosedLoopAllUnits' loopback socket
// (Viewer/LiveWireClient.h, sharing ClosedLoop/LiveWire.h with the publisher
// so neither side describes the format in its own words). Offline, it opens
// the same run's CSVs -- which the pipeline keeps writing regardless, so a
// finished run is always replayable and a viewer that was not running never
// costs anything.
//
// Everything is interpreted in SessionModel and nothing here; the UI below
// only asks it questions. In particular the cross-stream time question lives
// entirely in that file, and its answer is stated in its header rather than
// re-derived per plot.
//
// Win32 + Direct3D 11 backend: both halves ship with the Windows SDK this
// repo already builds against, so this executable adds nothing to install.
// The window/device boilerplate below follows Dear ImGui's own
// examples/example_win32_directx11 -- it is transcription, not design.
// =================================

// windows.h arrives via the Win32 backend below and defines min/max as
// macros, which then break every std::min/std::max in this file.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx11.h"
#include "implot.h"

#include <d3d11.h>
#include <tchar.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "LiveWireClient.h"
#include "SessionModel.h"

// ---------------------------------------------------------------------------
// D3D11 / Win32 plumbing (transcribed from imgui's example backend)
// ---------------------------------------------------------------------------
static ID3D11Device            *g_pd3dDevice        = nullptr;
static ID3D11DeviceContext     *g_pd3dDeviceContext = nullptr;
static IDXGISwapChain          *g_pSwapChain        = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView  *g_mainRenderTargetView = nullptr;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler( HWND hWnd, UINT msg,
                                                               WPARAM wParam, LPARAM lParam );

static void CreateRenderTarget()
{
    ID3D11Texture2D *pBackBuffer = nullptr;
    g_pSwapChain->GetBuffer( 0, IID_PPV_ARGS( &pBackBuffer ) );
    if( pBackBuffer ) {
        g_pd3dDevice->CreateRenderTargetView( pBackBuffer, nullptr, &g_mainRenderTargetView );
        pBackBuffer->Release();
    }
}

static void CleanupRenderTarget()
{
    if( g_mainRenderTargetView ) {
        g_mainRenderTargetView->Release();
        g_mainRenderTargetView = nullptr;
    }
}

static bool CreateDeviceD3D( HWND hWnd )
{
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof(sd) );
    sd.BufferCount        = 2;
    sd.BufferDesc.Width   = 0;
    sd.BufferDesc.Height  = 0;
    sd.BufferDesc.Format  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator   = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags              = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow       = hWnd;
    sd.SampleDesc.Count   = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed           = TRUE;
    sd.SwapEffect         = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };

    HRESULT res = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, featureLevelArray, 2,
        D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext );
    if( res == DXGI_ERROR_UNSUPPORTED ) {
        res = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, featureLevelArray, 2,
            D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext );
    }
    if( res != S_OK )
        return false;

    CreateRenderTarget();
    return true;
}

static void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if( g_pSwapChain )        { g_pSwapChain->Release();        g_pSwapChain = nullptr; }
    if( g_pd3dDeviceContext ) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if( g_pd3dDevice )        { g_pd3dDevice->Release();        g_pd3dDevice = nullptr; }
}

static LRESULT WINAPI WndProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam )
{
    if( ImGui_ImplWin32_WndProcHandler( hWnd, msg, wParam, lParam ) )
        return true;

    switch( msg ) {
    case WM_SIZE:
        if( wParam == SIZE_MINIMIZED )
            return 0;
        g_ResizeWidth  = (UINT)LOWORD( lParam );
        g_ResizeHeight = (UINT)HIWORD( lParam );
        return 0;
    case WM_SYSCOMMAND:
        if( ( wParam & 0xfff0 ) == SC_KEYMENU )   // disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage( 0 );
        return 0;
    }
    return ::DefWindowProcW( hWnd, msg, wParam, lParam );
}


// ---------------------------------------------------------------------------
// Viewer state
// ---------------------------------------------------------------------------
namespace {

struct ViewerState {
    SessionModel   model;
    LiveWireClient client;

    bool   live;
    char   host[64];
    int    port;
    char   status[512];

    // Offline paths.
    char   spikeCsv[512];
    char   syllableCsv[512];
    char   decisionCsv[512];
    float  csvImecRate;
    bool   csvSyllablesOnImecClock;
    int    csvSyllableOffset;

    // Live raster.
    float  rasterSpanS;      // seconds of stream visible
    bool   paused;
    double pausedAtS;        // stream time the pause froze at (scrubbable)
    float  historyS;
    int    maxRasterPoints;

    // Unit filtering.
    std::vector<char> unitVisible;   // char, not bool -- vector<bool> has no data()
    char   unitFilter[64];
    float  minRateHz;

    // Trigger-aligned view.
    int    alignUnit;        // index into model.unitIds()
    float  winStartMs, winEndMs;
    int    psthBins;
    bool   showTriggered, showUntriggered;
    bool   splitByOutcome;
    unsigned codeMask;

    ViewerState()
        :   live( false ), port( livewire::kDefaultPort ),
            csvImecRate( 30000.0f ), csvSyllablesOnImecClock( true ), csvSyllableOffset( 0 ),
            rasterSpanS( 10.0f ), paused( false ), pausedAtS( 0.0 ),
            historyS( 120.0f ), maxRasterPoints( 200000 ),
            minRateHz( 0.0f ), alignUnit( 0 ),
            winStartMs( -200.0f ), winEndMs( 500.0f ), psthBins( 70 ),
            showTriggered( true ), showUntriggered( true ), splitByOutcome( true ),
            codeMask( 0xFFFFFFFFu )
    {
        std::snprintf( host, sizeof(host), "127.0.0.1" );
        std::snprintf( status, sizeof(status), "Not connected." );
        spikeCsv[0] = syllableCsv[0] = decisionCsv[0] = 0;
        unitFilter[0] = 0;
    }
};

ViewerState g_st;

// Defined below, next to main(): the Session panel's buttons and the command
// line call the same two functions, so the two entry points cannot diverge.
bool connectLive();
bool openCsvs();

// Reused across frames so the plotting below never allocates per frame at
// several hundred thousand points.
std::vector<double> g_rx, g_ry;
std::vector<double> g_sx, g_sy;
std::vector<livewire::WireRecord> g_pollBuf;

const ImVec4 kTriggeredCol   = ImVec4( 1.00f, 0.42f, 0.28f, 1.0f );
const ImVec4 kUntriggeredCol = ImVec4( 0.38f, 0.68f, 1.00f, 1.0f );
const ImVec4 kPendingCol     = ImVec4( 0.65f, 0.65f, 0.65f, 1.0f );

void syncUnitVisibility()
{
    size_t n = g_st.model.unitIds().size();
    if( g_st.unitVisible.size() != n )
        g_st.unitVisible.resize( n, 1 );
}

bool unitPassesFilter( int unitIndex )
{
    if( !g_st.unitVisible.empty() && !g_st.unitVisible[unitIndex] )
        return false;

    if( g_st.minRateHz > 0.0f
        && g_st.model.firingRateHz( unitIndex, 10.0 ) < g_st.minRateHz ) {
        return false;
    }

    if( g_st.unitFilter[0] ) {
        char buf[32];
        std::snprintf( buf, sizeof(buf), "%d", g_st.model.unitIds()[unitIndex] );
        if( !std::strstr( buf, g_st.unitFilter ) )
            return false;
    }
    return true;
}


// ---------------------------------------------------------------------------
void drawSessionPanel()
{
    ImGui::Begin( "Session" );

    ImGui::TextUnformatted( "Live source (ClosedLoopAllUnits viewer socket)" );
    ImGui::InputText( "host", g_st.host, sizeof(g_st.host) );
    ImGui::InputInt( "port", &g_st.port );

    if( !g_st.client.connected() ) {
        if( ImGui::Button( "Connect" ) )
            connectLive();
    }
    else {
        if( ImGui::Button( "Disconnect" ) ) {
            g_st.client.disconnect();
            g_st.live = false;
        }
        ImGui::SameLine();
        ImGui::Text( "%lld records received", g_st.client.nReceived() );
    }

    ImGui::Separator();
    ImGui::TextUnformatted( "Offline source (this run's CSVs)" );
    ImGui::InputText( "spikeTimesPath",   g_st.spikeCsv,    sizeof(g_st.spikeCsv) );
    ImGui::InputText( "syllableTimesPath", g_st.syllableCsv, sizeof(g_st.syllableCsv) );
    ImGui::InputText( "decisionCsvPath",  g_st.decisionCsv, sizeof(g_st.decisionCsv) );
    ImGui::InputFloat( "IMEC sample rate", &g_st.csvImecRate );

    ImGui::Checkbox( "syllable indices are on the IMEC clock", &g_st.csvSyllablesOnImecClock );
    if( ImGui::IsItemHovered() ) {
        ImGui::SetTooltip(
            "True for a run recorded with syllableSource=imecSy, where the codes ride on the\n"
            "spikes' own clock and the alignment is exact.\n\n"
            "For syllableSource=ni the CSV holds NI-stream sample indices and the offset to\n"
            "the IMEC clock is recorded nowhere -- use the nudge below, and note that every\n"
            "trigger-aligned plot is then only as good as that number." );
    }
    ImGui::InputInt( "syllable index nudge (samples)", &g_st.csvSyllableOffset );

    if( ImGui::Button( "Open CSVs" ) )
        openCsvs();

    ImGui::Separator();
    ImGui::TextWrapped( "%s", g_st.status );

    if( g_st.model.anySyllableEstimated() ) {
        ImGui::PushStyleColor( ImGuiCol_Text, ImVec4( 1.0f, 0.75f, 0.2f, 1.0f ) );
        ImGui::TextWrapped(
            "Syllable times are on a DIFFERENT stream clock and have been placed onto the "
            "IMEC clock via the shared sync pulse. That is the same assumption DecisionThread "
            "runs on and it is good to well under a millisecond, but it is an estimate. "
            "syllableSource=imecSy removes it entirely." );
        ImGui::PopStyleColor();
    }

    ImGui::Separator();

    // ---- the decision line, live -----------------------------------------
    bool  high = g_st.model.lineHigh();
    ImVec4 col = high ? kTriggeredCol : ImVec4( 0.25f, 0.28f, 0.32f, 1.0f );
    ImGui::PushStyleColor( ImGuiCol_Button, col );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, col );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, col );
    ImGui::Button( high ? "DO LINE HIGH" : "DO line low", ImVec2( -1, 40 ) );
    ImGui::PopStyleColor( 3 );

    int nTrig = 0, nUntrig = 0, nPending = 0;
    const std::vector<SessionModel::Syllable> &sy = g_st.model.syllables();
    for( size_t i = 0; i < sy.size(); ++i ) {
        if( sy[i].triggered == 1 )      ++nTrig;
        else if( sy[i].triggered == 0 ) ++nUntrig;
        else                            ++nPending;
    }
    ImGui::Text( "trials: %d triggered / %d not / %d in flight", nTrig, nUntrig, nPending );
    ImGui::Text( "line raised %lld time(s) this session", g_st.model.nLineRaises() );
    ImGui::Text( "%lld spikes held, %d units", g_st.model.nSpikes(), g_st.model.nUnits() );

    ImGui::End();
}


// ---------------------------------------------------------------------------
void drawRaster()
{
    ImGui::Begin( "Spike raster" );

    ImGui::SetNextItemWidth( 140 );
    ImGui::SliderFloat( "span (s)", &g_st.rasterSpanS, 0.5f, 120.0f, "%.1f" );
    ImGui::SameLine();
    ImGui::Checkbox( "pause", &g_st.paused );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 140 );
    if( ImGui::SliderFloat( "history (s)", &g_st.historyS, 10.0f, 600.0f, "%.0f" ) )
        g_st.model.setHistorySeconds( g_st.live ? g_st.historyS : 0.0f );

    const double nowS = g_st.model.seconds( g_st.model.latestSample() );
    if( !g_st.paused )
        g_st.pausedAtS = nowS;

    if( g_st.paused ) {
        ImGui::SetNextItemWidth( -100 );
        float scrub = (float)g_st.pausedAtS;
        float lo    = (float)g_st.model.seconds( g_st.model.firstSample() ) + g_st.rasterSpanS;
        if( ImGui::SliderFloat( "scrub (s)", &scrub, lo, (float)nowS, "%.2f" ) )
            g_st.pausedAtS = scrub;
    }

    const double tEnd   = g_st.paused ? g_st.pausedAtS : nowS;
    const double tStart = tEnd - g_st.rasterSpanS;

    // Build the visible point set. Units are laid out on the y axis in filter
    // order, not unit-id order, so hiding units closes the gaps instead of
    // leaving blank rows.
    g_rx.clear();
    g_ry.clear();

    std::vector<int> rowUnit;
    const long long fromSample = g_st.model.firstSample()
        + (long long)( tStart * g_st.model.imecRate() );
    const long long toSample   = g_st.model.firstSample()
        + (long long)( tEnd * g_st.model.imecRate() );

    bool truncated = false;
    for( int u = 0; u < g_st.model.nUnits(); ++u ) {
        if( !unitPassesFilter( u ) )
            continue;

        double row = (double)rowUnit.size();
        rowUnit.push_back( u );

        const std::vector<SessionModel::Spike> &v = g_st.model.unitSpikes( u );
        // Linear scan from a binary search would be ideal, but the model's
        // vectors are ascending, so std::lower_bound does it directly.
        size_t i = (size_t)( std::lower_bound( v.begin(), v.end(), fromSample,
            []( const SessionModel::Spike &s, long long x ) { return s.sampleIndex < x; } ) - v.begin() );

        for( ; i < v.size() && v[i].sampleIndex <= toSample; ++i ) {
            if( (int)g_rx.size() >= g_st.maxRasterPoints ) {
                truncated = true;
                break;
            }
            g_rx.push_back( g_st.model.seconds( v[i].sampleIndex ) );
            g_ry.push_back( row );
        }
        if( truncated )
            break;
    }

    if( truncated ) {
        ImGui::SameLine();
        ImGui::TextColored( ImVec4( 1, 0.75f, 0.2f, 1 ),
            "showing first %d points -- narrow the span or filter units", g_st.maxRasterPoints );
    }

    if( ImPlot::BeginPlot( "##raster", ImVec2( -1, -1 ) ) ) {

        ImPlot::SetupAxes( "stream time (s)", "unit",
                            ImPlotAxisFlags_None, ImPlotAxisFlags_None );
        ImPlot::SetupAxisLimits( ImAxis_X1, tStart, tEnd, ImPlotCond_Always );
        ImPlot::SetupAxisLimits( ImAxis_Y1, -1.0, (double)rowUnit.size(), ImPlotCond_Always );

        // Syllable onsets, coloured by what the decision did with them. This
        // is the "was the line set high" question answered per trial and on
        // the same axis as the spikes, which is the only place the two can be
        // compared by eye.
        const std::vector<SessionModel::Syllable> &sy = g_st.model.syllables();
        for( size_t s = 0; s < sy.size(); ++s ) {
            double t = g_st.model.seconds( sy[s].sampleIndex );
            if( t < tStart || t > tEnd )
                continue;

            ImVec4 c = ( sy[s].triggered == 1 ) ? kTriggeredCol
                       : ( sy[s].triggered == 0 ) ? kUntriggeredCol : kPendingCol;

            double xs[1] = { t };
            ImPlot::SetNextLineStyle( c, 2.0f );
            char label[32];
            std::snprintf( label, sizeof(label), "##syl%d", (int)s );
            ImPlot::PlotInfLines( label, xs, 1 );

            char txt[16];
            std::snprintf( txt, sizeof(txt), "%d", sy[s].code );
            ImPlot::Annotation( t, (double)rowUnit.size(), c, ImVec2( 0, -4 ), false, "%s", txt );
        }

        if( !g_rx.empty() ) {
            ImPlot::SetNextMarkerStyle( ImPlotMarker_Square, 1.4f,
                                         ImVec4( 0.85f, 0.90f, 1.0f, 0.85f ), 0.0f );
            ImPlot::PlotScatter( "spikes", &g_rx[0], &g_ry[0], (int)g_rx.size() );
        }

        ImPlot::EndPlot();
    }

    ImGui::End();
}


// ---------------------------------------------------------------------------
void drawUnitsPanel()
{
    ImGui::Begin( "Units" );

    syncUnitVisibility();

    ImGui::InputText( "id contains", g_st.unitFilter, sizeof(g_st.unitFilter) );
    ImGui::SetNextItemWidth( 140 );
    ImGui::SliderFloat( "min rate (Hz)", &g_st.minRateHz, 0.0f, 50.0f, "%.1f" );

    if( ImGui::Button( "all" ) )
        std::fill( g_st.unitVisible.begin(), g_st.unitVisible.end(), (char)1 );
    ImGui::SameLine();
    if( ImGui::Button( "none" ) )
        std::fill( g_st.unitVisible.begin(), g_st.unitVisible.end(), (char)0 );

    ImGui::Separator();

    if( ImGui::BeginTable( "units", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY ) ) {

        ImGui::TableSetupColumn( "show", ImGuiTableColumnFlags_WidthFixed, 40 );
        ImGui::TableSetupColumn( "unit" );
        ImGui::TableSetupColumn( "rate (Hz, 10 s)" );
        ImGui::TableSetupColumn( "spikes held" );
        ImGui::TableSetupScrollFreeze( 0, 1 );
        ImGui::TableHeadersRow();

        for( int u = 0; u < g_st.model.nUnits(); ++u ) {

            ImGui::TableNextRow();
            ImGui::PushID( u );

            ImGui::TableSetColumnIndex( 0 );
            bool vis = g_st.unitVisible[u] != 0;
            if( ImGui::Checkbox( "##v", &vis ) )
                g_st.unitVisible[u] = vis ? 1 : 0;

            ImGui::TableSetColumnIndex( 1 );
            if( ImGui::Selectable( std::to_string( g_st.model.unitIds()[u] ).c_str(),
                                    g_st.alignUnit == u, ImGuiSelectableFlags_SpanAllColumns ) ) {
                g_st.alignUnit = u;
            }

            ImGui::TableSetColumnIndex( 2 );
            ImGui::Text( "%.2f", g_st.model.firingRateHz( u, 10.0 ) );

            ImGui::TableSetColumnIndex( 3 );
            ImGui::Text( "%d", (int)g_st.model.unitSpikes( u ).size() );

            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    ImGui::End();
}


// ---------------------------------------------------------------------------
void drawTriggerAligned()
{
    ImGui::Begin( "Trigger-aligned" );

    if( g_st.model.nUnits() == 0 ) {
        ImGui::TextUnformatted( "No session loaded." );
        ImGui::End();
        return;
    }

    g_st.alignUnit = std::max( 0, std::min( g_st.alignUnit, g_st.model.nUnits() - 1 ) );

    ImGui::Text( "unit %d", g_st.model.unitIds()[g_st.alignUnit] );
    ImGui::SameLine();
    ImGui::TextDisabled( "(select in the Units panel)" );

    ImGui::SetNextItemWidth( 120 );
    ImGui::InputFloat( "window start (ms)", &g_st.winStartMs );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 120 );
    ImGui::InputFloat( "window end (ms)", &g_st.winEndMs );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 120 );
    ImGui::SliderInt( "PSTH bins", &g_st.psthBins, 5, 400 );

    ImGui::Checkbox( "triggered", &g_st.showTriggered );
    ImGui::SameLine();
    ImGui::Checkbox( "not triggered", &g_st.showUntriggered );
    ImGui::SameLine();
    ImGui::Checkbox( "split PSTH by outcome", &g_st.splitByOutcome );
    if( ImGui::IsItemHovered() ) {
        ImGui::SetTooltip(
            "Overlays the triggered and not-triggered PSTHs separately.\n"
            "The triggered set is, by construction, the trials in which this unit fired\n"
            "at least spikeCountThreshold times in the decision window -- so a difference\n"
            "inside that window is the selection, not a finding. Look outside it." );
    }

    // Syllable code filter.
    ImGui::TextUnformatted( "codes:" );
    for( int c = 1; c <= 7; ++c ) {
        ImGui::SameLine();
        bool on = ( g_st.codeMask & ( 1u << c ) ) != 0;
        char lbl[8];
        std::snprintf( lbl, sizeof(lbl), "%d", c );
        if( ImGui::Checkbox( lbl, &on ) ) {
            if( on ) g_st.codeMask |= ( 1u << c );
            else     g_st.codeMask &= ~( 1u << c );
        }
    }

    const double w0 = g_st.winStartMs / 1000.0;
    const double w1 = g_st.winEndMs / 1000.0;
    if( w1 <= w0 ) {
        ImGui::TextColored( ImVec4( 1, 0.4f, 0.3f, 1 ), "window end must be after window start" );
        ImGui::End();
        return;
    }

    std::vector<SessionModel::TrialSpikes> trials;
    g_st.model.collectTrials( g_st.alignUnit, w0, w1, g_st.codeMask,
                               g_st.showTriggered, g_st.showUntriggered, trials );

    ImGui::Text( "%d trial(s)", (int)trials.size() );

    float avail = ImGui::GetContentRegionAvail().y;

    // ---- per-trial raster ------------------------------------------------
    if( ImPlot::BeginPlot( "trials", ImVec2( -1, avail * 0.55f ) ) ) {

        ImPlot::SetupAxes( "time from syllable onset (s)", "trial" );
        ImPlot::SetupAxisLimits( ImAxis_X1, w0, w1, ImPlotCond_Always );
        ImPlot::SetupAxisLimits( ImAxis_Y1, -1.0, (double)trials.size(), ImPlotCond_Always );

        double zero[1] = { 0.0 };
        ImPlot::SetNextLineStyle( ImVec4( 1, 1, 1, 0.5f ), 1.5f );
        ImPlot::PlotInfLines( "##onset", zero, 1 );

        // Two passes so the two outcomes get one legend entry each rather
        // than one per trial.
        for( int pass = 0; pass < 2; ++pass ) {
            g_sx.clear();
            g_sy.clear();
            for( size_t t = 0; t < trials.size(); ++t ) {
                bool isTrig = ( trials[t].triggered == 1 );
                if( ( pass == 0 ) != isTrig )
                    continue;
                for( size_t i = 0; i < trials[t].relS.size(); ++i ) {
                    g_sx.push_back( trials[t].relS[i] );
                    g_sy.push_back( (double)t );
                }
            }
            if( g_sx.empty() )
                continue;
            ImPlot::SetNextMarkerStyle( ImPlotMarker_Square, 2.0f,
                pass == 0 ? kTriggeredCol : kUntriggeredCol, 0.0f );
            ImPlot::PlotScatter( pass == 0 ? "triggered" : "not triggered",
                                  &g_sx[0], &g_sy[0], (int)g_sx.size() );
        }

        ImPlot::EndPlot();
    }

    // ---- PSTH --------------------------------------------------------------
    if( ImPlot::BeginPlot( "PSTH", ImVec2( -1, -1 ) ) ) {

        ImPlot::SetupAxes( "time from syllable onset (s)", "rate (spikes/s/trial)" );
        ImPlot::SetupAxisLimits( ImAxis_X1, w0, w1, ImPlotCond_Always );

        double zero[1] = { 0.0 };
        ImPlot::SetNextLineStyle( ImVec4( 1, 1, 1, 0.5f ), 1.5f );
        ImPlot::PlotInfLines( "##onset", zero, 1 );

        std::vector<double> centers, rate;
        int nTrials = 0;

        if( g_st.splitByOutcome ) {
            const char *names[2] = { "triggered", "not triggered" };
            for( int pass = 0; pass < 2; ++pass ) {
                g_st.model.computePsth( g_st.alignUnit, w0, w1, g_st.psthBins, g_st.codeMask,
                                         pass == 0, pass == 1, centers, rate, nTrials );
                if( nTrials == 0 || centers.empty() )
                    continue;
                char lbl[64];
                std::snprintf( lbl, sizeof(lbl), "%s (n=%d)", names[pass], nTrials );
                ImPlot::SetNextLineStyle( pass == 0 ? kTriggeredCol : kUntriggeredCol, 2.0f );
                ImPlot::PlotLine( lbl, &centers[0], &rate[0], (int)centers.size() );
            }
        }
        else {
            g_st.model.computePsth( g_st.alignUnit, w0, w1, g_st.psthBins, g_st.codeMask,
                                     g_st.showTriggered, g_st.showUntriggered,
                                     centers, rate, nTrials );
            if( nTrials > 0 && !centers.empty() ) {
                char lbl[64];
                std::snprintf( lbl, sizeof(lbl), "all (n=%d)", nTrials );
                ImPlot::SetNextLineStyle( ImVec4( 0.85f, 0.90f, 1.0f, 1.0f ), 2.0f );
                ImPlot::PlotLine( lbl, &centers[0], &rate[0], (int)centers.size() );
            }
        }

        ImPlot::EndPlot();
    }

    ImGui::End();
}

// Connect / open, factored out of the buttons so the command line can do
// exactly what a click does rather than a parallel version of it.
bool connectLive()
{
    if( !g_st.client.connect( g_st.host, g_st.port ) ) {
        std::snprintf( g_st.status, sizeof(g_st.status), "%s", g_st.client.lastError().c_str() );
        return false;
    }
    g_st.model.reset();
    g_st.model.setUnitIds( g_st.client.unitIds() );
    g_st.model.setSampleRates( g_st.client.header().imecSampleRateHz,
                                g_st.client.header().niSampleRateHz );
    g_st.model.setHistorySeconds( g_st.historyS );
    g_st.live = true;
    syncUnitVisibility();
    std::snprintf( g_st.status, sizeof(g_st.status),
        "Connected. %d units, IMEC %.0f Hz, syllable source = %s.",
        (int)g_st.client.unitIds().size(),
        g_st.client.header().imecSampleRateHz,
        g_st.client.header().syllableSourceIsImecSy ? "IMEC SY (TEST PATH)" : "NI (production)" );
    return true;
}

bool openCsvs()
{
    std::string err;
    if( !g_st.model.loadCsv( g_st.spikeCsv, g_st.syllableCsv, g_st.decisionCsv,
                              g_st.csvImecRate, g_st.csvSyllablesOnImecClock,
                              g_st.csvSyllableOffset, err ) ) {
        std::snprintf( g_st.status, sizeof(g_st.status), "%s", err.c_str() );
        return false;
    }
    g_st.live = false;
    g_st.client.disconnect();
    syncUnitVisibility();
    std::snprintf( g_st.status, sizeof(g_st.status),
        "Loaded %lld spikes, %d units, %d syllables from CSV.",
        g_st.model.nSpikes(), g_st.model.nUnits(), (int)g_st.model.syllables().size() );
    return true;
}

void usage( const char *exe )
{
    std::printf(
        "Usage: %s [--live [host:port]] [--csv <spikeTimes> [syllableTimes] [decisions]]\n"
        "          [--rate <Hz>] [--unit <kilosort_id>] [--quit-after <seconds>]\n"
        "\n"
        "With no arguments the viewer starts idle and you connect or open files from\n"
        "the Session panel. The switches do exactly what those controls do; they exist\n"
        "so a run can be launched from a script or a build task.\n"
        "--quit-after closes the window on a timer, for unattended screenshots.\n", exe );
}

} // namespace


// ---------------------------------------------------------------------------
int main( int argc, char **argv )
{
    bool   wantLive = false, wantCsv = false;
    double quitAfterS = 0.0;
    int    wantUnitId = -1;

    for( int i = 1; i < argc; ++i ) {
        std::string a = argv[i];
        if( a == "--live" ) {
            wantLive = true;
            if( i + 1 < argc && argv[i + 1][0] != '-' ) {
                std::string hp = argv[++i];
                size_t colon = hp.find( ':' );
                if( colon != std::string::npos ) {
                    std::snprintf( g_st.host, sizeof(g_st.host), "%s", hp.substr( 0, colon ).c_str() );
                    g_st.port = std::atoi( hp.substr( colon + 1 ).c_str() );
                }
                else {
                    std::snprintf( g_st.host, sizeof(g_st.host), "%s", hp.c_str() );
                }
            }
        }
        else if( a == "--csv" ) {
            wantCsv = true;
            if( i + 1 < argc && argv[i + 1][0] != '-' )
                std::snprintf( g_st.spikeCsv, sizeof(g_st.spikeCsv), "%s", argv[++i] );
            if( i + 1 < argc && argv[i + 1][0] != '-' )
                std::snprintf( g_st.syllableCsv, sizeof(g_st.syllableCsv), "%s", argv[++i] );
            if( i + 1 < argc && argv[i + 1][0] != '-' )
                std::snprintf( g_st.decisionCsv, sizeof(g_st.decisionCsv), "%s", argv[++i] );
        }
        else if( a == "--rate" && i + 1 < argc ) {
            g_st.csvImecRate = (float)std::atof( argv[++i] );
        }
        else if( a == "--unit" && i + 1 < argc ) {
            wantUnitId = std::atoi( argv[++i] );
        }
        else if( a == "--quit-after" && i + 1 < argc ) {
            quitAfterS = std::atof( argv[++i] );
        }
        else {
            usage( argv[0] );
            return ( a == "--help" || a == "-h" ) ? 0 : 1;
        }
    }

    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
                        GetModuleHandle( nullptr ), nullptr, nullptr, nullptr, nullptr,
                        L"SpikeViewer", nullptr };
    ::RegisterClassExW( &wc );
    HWND hwnd = ::CreateWindowW( wc.lpszClassName, L"SpikeViewer -- ClosedLoop live view",
                                  WS_OVERLAPPEDWINDOW, 60, 60, 1600, 950,
                                  nullptr, nullptr, wc.hInstance, nullptr );

    if( !CreateDeviceD3D( hwnd ) ) {
        CleanupDeviceD3D();
        ::UnregisterClassW( wc.lpszClassName, wc.hInstance );
        return 1;
    }

    ::ShowWindow( hwnd, SW_SHOWDEFAULT );
    ::UpdateWindow( hwnd );

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init( hwnd );
    ImGui_ImplDX11_Init( g_pd3dDevice, g_pd3dDeviceContext );

    if( wantCsv )
        openCsvs();
    if( wantLive )
        connectLive();
    if( wantUnitId >= 0 ) {
        for( int u = 0; u < g_st.model.nUnits(); ++u ) {
            if( g_st.model.unitIds()[u] == wantUnitId ) {
                g_st.alignUnit = u;
                break;
            }
        }
    }

    const ImVec4 clear_color = ImVec4( 0.09f, 0.10f, 0.12f, 1.00f );
    bool firstFrame = true;
    bool done = false;
    auto startedAt = std::chrono::steady_clock::now();

    while( !done ) {

        MSG msg;
        while( ::PeekMessage( &msg, nullptr, 0U, 0U, PM_REMOVE ) ) {
            ::TranslateMessage( &msg );
            ::DispatchMessage( &msg );
            if( msg.message == WM_QUIT )
                done = true;
        }
        if( done )
            break;

        if( quitAfterS > 0.0
            && std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - startedAt ).count() >= quitAfterS ) {
            break;
        }

        if( g_SwapChainOccluded && g_pSwapChain->Present( 0, DXGI_PRESENT_TEST ) == DXGI_STATUS_OCCLUDED ) {
            ::Sleep( 10 );
            continue;
        }
        g_SwapChainOccluded = false;

        if( g_ResizeWidth != 0 && g_ResizeHeight != 0 ) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers( 0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0 );
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // ---- drain the socket, once per frame ----------------------------
        // Non-blocking by contract (LiveWireClient::poll), so a publisher
        // that stops sending costs nothing and a frame is never held up
        // waiting for one.
        if( g_st.client.connected() ) {
            g_pollBuf.clear();
            bool alive = g_st.client.poll( g_pollBuf );
            for( size_t i = 0; i < g_pollBuf.size(); ++i )
                g_st.model.applyRecord( g_pollBuf[i] );
            if( !g_pollBuf.empty() ) {
                syncUnitVisibility();
                g_st.model.trimHistory();
            }
            if( !alive ) {
                g_st.live = false;
                std::snprintf( g_st.status, sizeof(g_st.status),
                    "Publisher closed the connection. %lld records received; the CSVs hold the full run.",
                    g_st.client.nReceived() );
            }
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // A fixed first-run layout, so the four panels are usable without
        // dragging them into place. Only "FirstUseEver", so a rearranged
        // layout survives in imgui.ini.
        if( firstFrame ) {
            ImGui::SetNextWindowPos( ImVec2( 10, 10 ), ImGuiCond_FirstUseEver );
            ImGui::SetNextWindowSize( ImVec2( 430, 560 ), ImGuiCond_FirstUseEver );
        }
        drawSessionPanel();

        if( firstFrame ) {
            ImGui::SetNextWindowPos( ImVec2( 10, 580 ), ImGuiCond_FirstUseEver );
            ImGui::SetNextWindowSize( ImVec2( 430, 350 ), ImGuiCond_FirstUseEver );
        }
        drawUnitsPanel();

        if( firstFrame ) {
            ImGui::SetNextWindowPos( ImVec2( 450, 10 ), ImGuiCond_FirstUseEver );
            ImGui::SetNextWindowSize( ImVec2( 1130, 460 ), ImGuiCond_FirstUseEver );
        }
        drawRaster();

        if( firstFrame ) {
            ImGui::SetNextWindowPos( ImVec2( 450, 480 ), ImGuiCond_FirstUseEver );
            ImGui::SetNextWindowSize( ImVec2( 1130, 450 ), ImGuiCond_FirstUseEver );
        }
        drawTriggerAligned();

        firstFrame = false;

        ImGui::Render();
        const float clear_with_alpha[4] = { clear_color.x, clear_color.y, clear_color.z, clear_color.w };
        g_pd3dDeviceContext->OMSetRenderTargets( 1, &g_mainRenderTargetView, nullptr );
        g_pd3dDeviceContext->ClearRenderTargetView( g_mainRenderTargetView, clear_with_alpha );
        ImGui_ImplDX11_RenderDrawData( ImGui::GetDrawData() );

        HRESULT hr = g_pSwapChain->Present( 1, 0 );   // vsync
        g_SwapChainOccluded = ( hr == DXGI_STATUS_OCCLUDED );
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow( hwnd );
    ::UnregisterClassW( wc.lpszClassName, wc.hInstance );
    return 0;
}
