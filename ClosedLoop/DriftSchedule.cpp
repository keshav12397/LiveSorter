#include "DriftSchedule.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
const int32_t kMagic = 0x44524654; // 'DRFT', matches calibrate_drift_aware.py's write_schedule()
}

DriftSchedule DriftSchedule::load( const std::string &dir )
{
    DriftSchedule sched;
    std::string path = dir + "/drift_schedule.bin";

    std::ifstream fh( path.c_str(), std::ios::binary );
    if( !fh.is_open() )
        return sched; // absent file == no schedule, not an error (see header)

    int32_t header[5];
    fh.read( reinterpret_cast<char*>( header ), sizeof(header) );
    if( !fh )
        throw std::runtime_error( "DriftSchedule: '" + path + "' too short for header" );

    if( header[0] != kMagic ) {
        std::ostringstream oss;
        oss << "DriftSchedule: '" << path << "' bad magic 0x" << std::hex << header[0]
            << ", expected 0x" << kMagic;
        throw std::runtime_error( oss.str() );
    }

    sched.version         = header[1];
    int32_t nEvents       = header[2];
    sched.templateLength  = header[3];
    sched.nChannelsPerUnit = header[4];

    if( nEvents < 0 || sched.templateLength <= 0 || sched.nChannelsPerUnit <= 0 )
        throw std::runtime_error( "DriftSchedule: '" + path + "' has invalid header dimensions" );

    sched.events.reserve( static_cast<size_t>( nEvents ) );
    for( int32_t i = 0; i < nEvents; ++i ) {
        // Matches write_schedule()'s per-event layout exactly:
        //   struct.pack("<fii", t_s, unit_index, 0)
        //   int32[nChannelsPerUnit] channels
        //   float32[templateLength * nChannelsPerUnit] filter
        //   struct.pack("<f", threshold)
        float t_s;
        int32_t unitIndex, reserved;
        fh.read( reinterpret_cast<char*>( &t_s ), sizeof(t_s) );
        fh.read( reinterpret_cast<char*>( &unitIndex ), sizeof(unitIndex) );
        fh.read( reinterpret_cast<char*>( &reserved ), sizeof(reserved) );

        Event ev;
        ev.t_s = static_cast<double>( t_s );
        ev.unitIndex = unitIndex;

        ev.channels.resize( static_cast<size_t>( sched.nChannelsPerUnit ) );
        fh.read( reinterpret_cast<char*>( ev.channels.data() ),
                 ev.channels.size() * sizeof(int32_t) );

        ev.filter.resize( static_cast<size_t>( sched.templateLength ) *
                           static_cast<size_t>( sched.nChannelsPerUnit ) );
        fh.read( reinterpret_cast<char*>( ev.filter.data() ),
                 ev.filter.size() * sizeof(float) );

        fh.read( reinterpret_cast<char*>( &ev.threshold ), sizeof(ev.threshold) );

        if( !fh )
            throw std::runtime_error( "DriftSchedule: '" + path + "' truncated at event " +
                                       std::to_string( i ) );
        sched.events.push_back( std::move( ev ) );
    }

    return sched;
}

const DriftSchedule::Event *DriftSchedule::next( double nowS )
{
    if( cursor >= events.size() )
        return nullptr;
    if( events[cursor].t_s > nowS )
        return nullptr;
    return &events[cursor++];
}
