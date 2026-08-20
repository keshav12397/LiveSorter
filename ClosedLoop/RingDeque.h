#ifndef CLOSEDLOOP_RINGDEQUE_H
#define CLOSEDLOOP_RINGDEQUE_H

#include <vector>
#include <cstddef>

// A push_back / pop_front / index-from-front queue backed by one circular
// std::vector. Drop-in for the subset of std::deque that ConvolutionEngine
// uses -- empty, size, clear, push_back, pop_front, front, operator[].
//
// Why this exists rather than just using std::deque
// --------------------------------------------------
// MSVC's std::deque sizes its blocks in ELEMENTS, not bytes:
//
//     _DEQUESIZ = sizeof(T) <= 1 ? 16 : <= 2 ? 8 : <= 4 ? 4 : <= 8 ? 2 : 1
//
// So a deque<double> allocates a fresh block every 2 elements, and a deque of
// 16-byte pairs allocates one PER ELEMENT. ConvolutionEngine pushes one D
// value per sample and pops them again as the window slides, which on that
// implementation is a malloc and a free per couple of samples, per unit.
//
// Profiling the all-units path put a number on it: 30.5 heap allocations per
// unit-chunk, 12.2 million for two seconds of data at 1000 units, with the
// decision phase at 72% of total runtime. The algorithm was never the
// problem -- the container was. This is a portability trap rather than a
// standard-library flaw, since libstdc++ blocks by BYTES (512) and would
// hardly allocate at all; code that profiles clean on Linux can be
// allocation-bound here.
//
// Capacity doubles and never shrinks, so a steady-state stream stops
// allocating entirely once the window reaches its working size.
template<typename T>
class RingDeque {
public:
    RingDeque() : head_( 0 ), size_( 0 ) {}

    bool   empty() const { return size_ == 0; }
    size_t size()  const { return size_; }

    void clear() { head_ = 0; size_ = 0; }

    // Index from the FRONT, matching std::deque's operator[].
    T       &operator[]( size_t i )       { return buf_[wrap( head_ + i )]; }
    const T &operator[]( size_t i ) const { return buf_[wrap( head_ + i )]; }

    T       &front()       { return buf_[head_]; }
    const T &front() const { return buf_[head_]; }

    void push_back( const T &v )
    {
        if( size_ == buf_.size() )
            grow();
        buf_[wrap( head_ + size_ )] = v;
        ++size_;
    }

    void pop_front()
    {
        if( size_ == 0 )
            return;
        head_ = wrap( head_ + 1 );
        --size_;
    }

private:
    size_t wrap( size_t i ) const
    {
        const size_t n = buf_.size();
        return ( i >= n ) ? i - n : i;   // i is always < 2n at call sites
    }

    void grow()
    {
        // Linearise on grow: copy out in logical order so head_ returns to 0.
        const size_t n = buf_.size();
        std::vector<T> next( n ? n * 2 : 16 );
        for( size_t i = 0; i < size_; ++i )
            next[i] = buf_[wrap( head_ + i )];
        buf_.swap( next );
        head_ = 0;
    }

    std::vector<T> buf_;
    size_t         head_;
    size_t         size_;
};

#endif // CLOSEDLOOP_RINGDEQUE_H
