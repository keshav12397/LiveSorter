#include "EventPublisher.h"

#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET sock_t;
static const sock_t kInvalidSock = INVALID_SOCKET;
#else
#error "EventPublisher is Windows-only, like the rest of the live path (SglxApi)."
#endif

namespace {

// Winsock needs a process-wide WSAStartup before any socket call and a
// matching WSACleanup after the last one. Doing it in the constructor of a
// function-local static gets both, exactly once, without any main() having
// to remember -- and without an ordering dependency on SglxApi.dll, which
// does its own WSAStartup (the refcount makes that safe).
struct WinsockInit {
    bool ok;
    WinsockInit() {
        WSADATA wsa;
        ok = ( WSAStartup( MAKEWORD( 2, 2 ), &wsa ) == 0 );
    }
    ~WinsockInit() { if( ok ) WSACleanup(); }
};

bool ensureWinsock()
{
    static WinsockInit init;
    return init.ok;
}

sock_t toSock( unsigned long long v ) { return static_cast<sock_t>( v ); }

} // namespace


EventPublisher::EventPublisher( int port, const std::vector<int> &unitIds,
                                 double imecSampleRateHz, double niSampleRateHz,
                                 bool syllableSourceIsImecSy,
                                 size_t ringCapacity )
    :   port_( port ), unitIds_( unitIds ),
        ring_( ringCapacity ), head_( 0 ), tail_( 0 ), count_( 0 ),
        hasClient_( false ), nPublished_( 0 ), nDropped_( 0 ),
        listenSock_( static_cast<unsigned long long>( kInvalidSock ) ),
        clientSock_( static_cast<unsigned long long>( kInvalidSock ) ),
        stopFlag_( false ), started_( false )
{
    std::memset( &header_, 0, sizeof(header_) );
    livewire::fillMagic( header_ );
    header_.version          = livewire::kProtocolVersion;
    header_.nUnits           = static_cast<uint16_t>( unitIds_.size() );
    header_.imecSampleRateHz = imecSampleRateHz;
    header_.niSampleRateHz   = niSampleRateHz;
    header_.syllableSourceIsImecSy = syllableSourceIsImecSy ? 1 : 0;
    header_.templateLength   = 0;

    // Valid empty v2 preamble until/unless setChannelGeometry() is called.
    nChannels_.assign( unitIds_.size(), 0 );
}


void EventPublisher::setChannelGeometry(
    const std::vector<std::vector<livewire::ChannelGeom> > &unitChannelGeom,
    int templateLength )
{
    nChannels_.assign( unitIds_.size(), 0 );
    channelGeom_.clear();

    // Defensive rather than fatal: a mismatched geometry vector (caller
    // bug) degrades to "no geometry for the units past the mismatch"
    // instead of crashing the whole recording session, matching this
    // file's existing stance that the viewer socket must never be able to
    // take the run down.
    size_t n = std::min( unitChannelGeom.size(), unitIds_.size() );
    for( size_t i = 0; i < n; ++i ) {
        nChannels_[i] = static_cast<int32_t>( unitChannelGeom[i].size() );
        channelGeom_.insert( channelGeom_.end(),
                              unitChannelGeom[i].begin(), unitChannelGeom[i].end() );
    }

    header_.templateLength = static_cast<uint8_t>(
        std::min( templateLength, 255 ) );
}


EventPublisher::~EventPublisher()
{
    stop();
}


bool EventPublisher::start()
{
    if( !ensureWinsock() ) {
        std::cerr << "EventPublisher: WSAStartup failed, viewer socket disabled\n";
        return false;
    }

    sock_t ls = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
    if( ls == kInvalidSock ) {
        std::cerr << "EventPublisher: socket() failed (" << WSAGetLastError() << ")\n";
        return false;
    }

    // A viewer that was connected when the previous run exited leaves the
    // port in TIME_WAIT; without this, restarting the pipeline within a
    // couple of minutes fails to bind for no reason the operator can see.
    BOOL yes = TRUE;
    setsockopt( ls, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>( &yes ), sizeof(yes) );

    sockaddr_in addr;
    std::memset( &addr, 0, sizeof(addr) );
    addr.sin_family = AF_INET;
    addr.sin_port   = htons( static_cast<u_short>( port_ ) );
    // Loopback only, never INADDR_ANY. This socket carries no
    // authentication of any kind and there is no reason for a rig machine on
    // a lab network to be answering it.
    addr.sin_addr.s_addr = htonl( INADDR_LOOPBACK );

    if( bind( ls, reinterpret_cast<sockaddr*>( &addr ), sizeof(addr) ) != 0 ) {
        std::cerr << "EventPublisher: bind(127.0.0.1:" << port_ << ") failed ("
                  << WSAGetLastError() << "), viewer socket disabled\n";
        closesocket( ls );
        return false;
    }
    if( listen( ls, 1 ) != 0 ) {
        std::cerr << "EventPublisher: listen() failed (" << WSAGetLastError() << ")\n";
        closesocket( ls );
        return false;
    }

    u_long nonblocking = 1;
    ioctlsocket( ls, FIONBIO, &nonblocking );

    listenSock_ = static_cast<unsigned long long>( ls );
    started_    = true;
    thread_     = std::thread( &EventPublisher::serverLoop, this );

    std::cout << "EventPublisher: listening on 127.0.0.1:" << port_ << "\n";
    return true;
}


void EventPublisher::stop()
{
    if( !started_ )
        return;
    stopFlag_.store( true );
    if( thread_.joinable() )
        thread_.join();

    closeClient();
    if( toSock( listenSock_ ) != kInvalidSock ) {
        closesocket( toSock( listenSock_ ) );
        listenSock_ = static_cast<unsigned long long>( kInvalidSock );
    }
    started_ = false;
}


void EventPublisher::publish( const livewire::WireRecord &rec )
{
    publish( &rec, 1 );
}


void EventPublisher::publish( const livewire::WireRecord *recs, size_t n )
{
    if( !started_ || n == 0 )
        return;

    // No client attached: throw the records away here rather than filling
    // the ring with a whole run of history that the eventual viewer would
    // then receive as a burst of stale events on connect. A viewer joining
    // mid-run sees the run from the moment it joined, which is what a live
    // view means; the CSVs are what a finished run is read from.
    if( !hasClient_.load() )
        return;

    std::lock_guard<std::mutex> lock( ringMutex_ );

    const size_t cap = ring_.size();
    for( size_t i = 0; i < n; ++i ) {
        ring_[head_] = recs[i];
        head_ = ( head_ + 1 ) % cap;
        if( count_ == cap ) {
            // Full: this write overwrote the oldest unsent record, so the
            // tail has to move with the head. Drop-oldest rather than
            // drop-newest so the viewer stays current when it falls behind,
            // which is the useful failure mode for a live view.
            tail_ = head_;
            nDropped_.fetch_add( 1 );
        }
        else {
            ++count_;
        }
    }
    nPublished_.fetch_add( static_cast<long long>( n ) );
}


void EventPublisher::closeClient()
{
    if( toSock( clientSock_ ) != kInvalidSock ) {
        closesocket( toSock( clientSock_ ) );
        clientSock_ = static_cast<unsigned long long>( kInvalidSock );
    }
    hasClient_.store( false );
}


void EventPublisher::serverLoop()
{
    // Scratch the sender drains into, so the ring mutex is never held
    // across a send(). Sized to bound one send batch; more than this just
    // takes another pass around the loop.
    std::vector<livewire::WireRecord> batch( 4096 );

    while( !stopFlag_.load() ) {

        if( !hasClient_.load() ) {
            sock_t c = accept( toSock( listenSock_ ), NULL, NULL );
            if( c == kInvalidSock ) {
                std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
                continue;
            }

            // Nagle would coalesce these small records into ~40 ms bundles,
            // which is exactly the latency a live view exists to avoid.
            BOOL yes = TRUE;
            setsockopt( c, IPPROTO_TCP, TCP_NODELAY,
                        reinterpret_cast<const char*>( &yes ), sizeof(yes) );

            // The header is the one thing that must arrive intact, so send
            // it while the socket is still blocking, before switching to
            // non-blocking. It is 32 bytes plus a few hundred unit ids into
            // an empty send buffer, so it cannot realistically block -- and
            // if it somehow did, blocking here costs nothing: no producer
            // ever waits on this thread.
            bool headerOk = ( send( c, reinterpret_cast<const char*>( &header_ ),
                                     sizeof(header_), 0 ) == sizeof(header_) );
            if( headerOk && !unitIds_.empty() ) {
                std::vector<int32_t> ids( unitIds_.begin(), unitIds_.end() );
                int bytes = static_cast<int>( ids.size() * sizeof(int32_t) );
                headerOk = ( send( c, reinterpret_cast<const char*>( &ids[0] ), bytes, 0 ) == bytes );
            }
            // Version-2 channel-geometry preamble: nChannels[nUnits] then
            // the flat unit-major ChannelGeom array (see LiveWire.h). Sent
            // unconditionally at v2 -- even an EventPublisher that never
            // called setChannelGeometry() has a valid all-zero nChannels_
            // from the constructor, so this is always a well-formed
            // preamble for a v2-speaking client to read.
            if( headerOk && !nChannels_.empty() ) {
                int bytes = static_cast<int>( nChannels_.size() * sizeof(int32_t) );
                headerOk = ( send( c, reinterpret_cast<const char*>( &nChannels_[0] ), bytes, 0 ) == bytes );
            }
            if( headerOk && !channelGeom_.empty() ) {
                int bytes = static_cast<int>( channelGeom_.size() * sizeof(livewire::ChannelGeom) );
                headerOk = ( send( c, reinterpret_cast<const char*>( &channelGeom_[0] ), bytes, 0 ) == bytes );
            }
            if( !headerOk ) {
                closesocket( c );
                continue;
            }

            u_long nonblocking = 1;
            ioctlsocket( c, FIONBIO, &nonblocking );

            // Start this client from NOW, not from whatever is still in the
            // ring -- see publish()'s comment about stale bursts.
            {
                std::lock_guard<std::mutex> lock( ringMutex_ );
                tail_  = head_;
                count_ = 0;
            }

            clientSock_ = static_cast<unsigned long long>( c );
            hasClient_.store( true );
            std::cout << "EventPublisher: viewer connected\n";
            continue;
        }

        size_t n = 0;
        {
            std::lock_guard<std::mutex> lock( ringMutex_ );
            const size_t cap = ring_.size();
            n = ( count_ < batch.size() ) ? count_ : batch.size();
            for( size_t i = 0; i < n; ++i ) {
                batch[i] = ring_[tail_];
                tail_ = ( tail_ + 1 ) % cap;
            }
            count_ -= n;
        }

        if( n == 0 ) {
            std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
            continue;
        }

        const char *p    = reinterpret_cast<const char*>( &batch[0] );
        int         left = static_cast<int>( n * sizeof(livewire::WireRecord) );
        bool        lost = false;

        while( left > 0 && !stopFlag_.load() ) {
            int sent = send( toSock( clientSock_ ), p, left, 0 );
            if( sent > 0 ) {
                p    += sent;
                left -= sent;
                continue;
            }
            if( sent < 0 && WSAGetLastError() == WSAEWOULDBLOCK ) {
                // Reader is behind. Wait here rather than in publish() --
                // this thread is the one that is allowed to be slow. The
                // ring absorbs new events meanwhile, and starts dropping
                // the oldest if the reader never recovers.
                std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
                continue;
            }
            lost = true;
            break;
        }

        if( lost ) {
            std::cout << "EventPublisher: viewer disconnected\n";
            closeClient();
        }
    }
}


std::string EventPublisher::summary() const
{
    std::ostringstream ss;
    ss << "viewer socket: " << nPublished_.load() << " records published, "
       << nDropped_.load() << " dropped";
    if( nDropped_.load() > 0 )
        ss << " (viewer could not keep up -- the CSV files are still complete)";
    ss << "\n";
    return ss.str();
}
