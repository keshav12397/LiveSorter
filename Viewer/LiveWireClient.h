#ifndef VIEWER_LIVEWIRECLIENT_H
#define VIEWER_LIVEWIRECLIENT_H

#include <string>
#include <vector>

#include "LiveWire.h"

// The reading half of the ClosedLoop -> SpikeViewer socket. Shares
// LiveWire.h with the writing half, so neither side describes the layout in
// its own words (see that header).
//
// Non-blocking by construction: poll() drains whatever has arrived and
// returns immediately, because it is called once per rendered frame and a
// GUI that blocks on a socket is a GUI that stops repainting. Partial
// records are kept in an internal remainder buffer and completed on a later
// poll -- TCP is a byte stream and a 32-byte record can and does straddle a
// read.
class LiveWireClient {
public:
    LiveWireClient();
    ~LiveWireClient();

    // Attempts one connect. Returns false and fills lastError() on failure;
    // the caller decides whether to retry (SpikeViewer offers a button
    // rather than reconnecting in a hidden loop, so a rig that is simply not
    // running yet does not look like a broken viewer).
    bool connect( const std::string &host, int port );
    void disconnect();

    bool connected() const { return connected_; }
    const std::string &lastError() const { return lastError_; }

    // Valid once connect() has succeeded.
    const livewire::SessionHeader &header() const { return header_; }
    const std::vector<int> &unitIds() const { return unitIds_; }

    // Appends everything currently readable to `out`. Returns false if the
    // peer closed or errored, in which case connected() is now false and
    // whatever was already read stays in `out`.
    bool poll( std::vector<livewire::WireRecord> &out );

    long long nReceived() const { return nReceived_; }

private:
    bool readExactBlocking( char *dst, int nBytes );

    unsigned long long sock_;
    bool               connected_;
    std::string        lastError_;

    livewire::SessionHeader header_;
    std::vector<int>        unitIds_;

    // Bytes of a record that arrived without the rest of it.
    std::vector<char> remainder_;
    long long         nReceived_;
};

#endif // VIEWER_LIVEWIRECLIENT_H
