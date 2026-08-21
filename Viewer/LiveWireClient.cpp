#include "LiveWireClient.h"

#include <cstring>
#include <sstream>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

namespace {

// Same one-shot pattern EventPublisher.cpp uses; both processes need it and
// neither should make main() remember.
struct WinsockInit {
    bool ok;
    WinsockInit() { WSADATA w; ok = ( WSAStartup( MAKEWORD( 2, 2 ), &w ) == 0 ); }
    ~WinsockInit() { if( ok ) WSACleanup(); }
};

bool ensureWinsock()
{
    static WinsockInit init;
    return init.ok;
}

const unsigned long long kNoSock = static_cast<unsigned long long>( INVALID_SOCKET );

} // namespace


LiveWireClient::LiveWireClient()
    :   sock_( kNoSock ), connected_( false ), nReceived_( 0 )
{
    std::memset( &header_, 0, sizeof(header_) );
}


LiveWireClient::~LiveWireClient()
{
    disconnect();
}


bool LiveWireClient::readExactBlocking( char *dst, int nBytes )
{
    int got = 0;
    while( got < nBytes ) {
        int n = recv( static_cast<SOCKET>( sock_ ), dst + got, nBytes - got, 0 );
        if( n <= 0 )
            return false;
        got += n;
    }
    return true;
}


bool LiveWireClient::connect( const std::string &host, int port )
{
    disconnect();
    lastError_.clear();

    if( !ensureWinsock() ) {
        lastError_ = "WSAStartup failed";
        return false;
    }

    SOCKET s = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
    if( s == INVALID_SOCKET ) {
        lastError_ = "socket() failed";
        return false;
    }

    sockaddr_in addr;
    std::memset( &addr, 0, sizeof(addr) );
    addr.sin_family = AF_INET;
    addr.sin_port   = htons( static_cast<u_short>( port ) );
    if( inet_pton( AF_INET, host.c_str(), &addr.sin_addr ) != 1 ) {
        lastError_ = "could not parse host '" + host + "' as an IPv4 address";
        closesocket( s );
        return false;
    }

    if( ::connect( s, reinterpret_cast<sockaddr*>( &addr ), sizeof(addr) ) != 0 ) {
        std::ostringstream ss;
        ss << "connect to " << host << ":" << port << " failed (" << WSAGetLastError()
           << ") -- is ClosedLoopAllUnits running with viewerEnabled=true?";
        lastError_ = ss.str();
        closesocket( s );
        return false;
    }

    sock_ = static_cast<unsigned long long>( s );

    // The session header is read while the socket is still blocking. It is
    // the one thing that must arrive whole before anything can be
    // interpreted, and there is no frame to skip to if it does not.
    if( !readExactBlocking( reinterpret_cast<char*>( &header_ ), sizeof(header_) ) ) {
        lastError_ = "peer closed before sending the session header";
        disconnect();
        return false;
    }
    if( !livewire::magicOk( header_ ) ) {
        lastError_ = "bad magic -- whatever is on that port is not ClosedLoop";
        disconnect();
        return false;
    }
    if( header_.version != livewire::kProtocolVersion ) {
        std::ostringstream ss;
        ss << "protocol version mismatch: publisher speaks v" << header_.version
           << ", this viewer speaks v" << livewire::kProtocolVersion
           << " -- rebuild both from the same tree";
        lastError_ = ss.str();
        disconnect();
        return false;
    }

    unitIds_.assign( header_.nUnits, 0 );
    if( header_.nUnits > 0 ) {
        std::vector<int32_t> raw( header_.nUnits );
        int bytes = static_cast<int>( raw.size() * sizeof(int32_t) );
        if( !readExactBlocking( reinterpret_cast<char*>( &raw[0] ), bytes ) ) {
            lastError_ = "peer closed partway through the unit-id table";
            disconnect();
            return false;
        }
        for( size_t i = 0; i < raw.size(); ++i )
            unitIds_[i] = raw[i];
    }

    // Version-2 channel-geometry preamble (see LiveWire.h). The version
    // check above already guarantees this stream is exactly
    // kProtocolVersion, so if that constant is 2 this is always present --
    // no separate per-field version gate is needed, and a mismatch here is
    // treated the same as any other truncated preamble.
    unitChannelGeom_.assign( header_.nUnits, std::vector<livewire::ChannelGeom>() );
    if( header_.nUnits > 0 ) {
        std::vector<int32_t> nChannels( header_.nUnits );
        int bytes = static_cast<int>( nChannels.size() * sizeof(int32_t) );
        if( !readExactBlocking( reinterpret_cast<char*>( &nChannels[0] ), bytes ) ) {
            lastError_ = "peer closed partway through the channel-count preamble";
            disconnect();
            return false;
        }

        long long total = 0;
        for( size_t i = 0; i < nChannels.size(); ++i ) {
            if( nChannels[i] < 0 ) {
                lastError_ = "channel-count preamble has a negative count -- stream is corrupt";
                disconnect();
                return false;
            }
            total += nChannels[i];
        }

        if( total > 0 ) {
            std::vector<livewire::ChannelGeom> flat( static_cast<size_t>( total ) );
            int geomBytes = static_cast<int>( flat.size() * sizeof(livewire::ChannelGeom) );
            if( !readExactBlocking( reinterpret_cast<char*>( &flat[0] ), geomBytes ) ) {
                lastError_ = "peer closed partway through the channel-geometry preamble";
                disconnect();
                return false;
            }
            size_t pos = 0;
            for( size_t i = 0; i < nChannels.size(); ++i ) {
                size_t n = static_cast<size_t>( nChannels[i] );
                unitChannelGeom_[i].assign( flat.begin() + pos, flat.begin() + pos + n );
                pos += n;
            }
        }
    }

    u_long nonblocking = 1;
    ioctlsocket( s, FIONBIO, &nonblocking );

    remainder_.clear();
    nReceived_ = 0;
    connected_ = true;
    return true;
}


void LiveWireClient::disconnect()
{
    if( sock_ != kNoSock ) {
        closesocket( static_cast<SOCKET>( sock_ ) );
        sock_ = kNoSock;
    }
    connected_ = false;
    unitChannelGeom_.clear();
}


bool LiveWireClient::poll( std::vector<livewire::WireRecord> &out )
{
    if( !connected_ )
        return false;

    // 64 KiB per pass = 2048 records. A full ring's worth arrives over
    // several passes rather than one enormous read, which keeps the frame
    // this runs on bounded.
    char  buf[65536];
    bool  alive = true;

    while( true ) {
        int n = recv( static_cast<SOCKET>( sock_ ), buf, sizeof(buf), 0 );

        if( n > 0 ) {
            remainder_.insert( remainder_.end(), buf, buf + n );

            const size_t recSize = sizeof(livewire::WireRecord);
            size_t whole = remainder_.size() / recSize;
            if( whole > 0 ) {
                const livewire::WireRecord *recs =
                    reinterpret_cast<const livewire::WireRecord*>( &remainder_[0] );
                out.insert( out.end(), recs, recs + whole );
                nReceived_ += static_cast<long long>( whole );
                remainder_.erase( remainder_.begin(),
                                   remainder_.begin() + static_cast<long>( whole * recSize ) );
            }

            if( n < static_cast<int>( sizeof(buf) ) )
                break;   // drained the socket buffer this pass
            continue;
        }

        if( n == 0 ) {
            alive = false;   // orderly close by the publisher
            break;
        }

        if( WSAGetLastError() == WSAEWOULDBLOCK )
            break;           // nothing more right now, which is the normal case

        alive = false;
        lastError_ = "recv failed";
        break;
    }

    if( !alive )
        disconnect();

    return alive;
}
