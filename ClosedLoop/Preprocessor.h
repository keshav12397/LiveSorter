#ifndef CLOSEDLOOP_PREPROCESSOR_H
#define CLOSEDLOOP_PREPROCESSOR_H

#include <vector>
#include <algorithm>
#include "ButterworthHighpass.h"

// Applies the same preprocessing, in the same order, as
// FilterGen/generate_filter.py's main(): high-pass each channel first (if
// enabled), THEN common-average-reference across the whole channel group
// (median, matching Python's np.median -- see common_average_reference()).
//
// IMPORTANT: this must run across the FULL CAR channel group (e.g. all 96
// shank channels), not just the filter's own 5 channels -- CAR computed
// over too few channels doesn't reject the shared noise the filter's
// statistics (channel selection, LCMV weights, threshold) were calibrated
// against. This exact mismatch caused live testing to blow up to over a
// million spurious "detections" in about a minute. Both Calibration and
// ImecFetchThread fetch/read the full CAR group, run it through one shared
// Preprocessor instance, and only subset down to the filter's 5 channels
// *after* this class returns -- see each caller's channel-index bookkeeping.
class Preprocessor {
public:
    Preprocessor( int nChannels, double highpassCutoffHz, double sampleRateHz,
                  bool applyHighpass, bool applyCar )
        :   nChannels_( nChannels ), applyHighpass_( applyHighpass ), applyCar_( applyCar )
    {
        if( applyHighpass_ ) {
            filters_.reserve( nChannels );
            for( int c = 0; c < nChannels; ++c )
                filters_.push_back( ButterworthHighpass( highpassCutoffHz, sampleRateHz ) );
        }
    }

    void reset()
    {
        for( size_t c = 0; c < filters_.size(); ++c )
            filters_[c].reset();
    }

    // data: nSamples * nChannels_ raw int16, time-major/channel-minor.
    // Returns the same shape as doubles, high-pass filtered (if enabled)
    // then CAR'd (if enabled).
    std::vector<double> processChunk( const short *data, size_t nSamples )
    {
        std::vector<double> out( nSamples * nChannels_ );

        for( int c = 0; c < nChannels_; ++c ) {
            if( applyHighpass_ ) {
                for( size_t s = 0; s < nSamples; ++s )
                    out[s * nChannels_ + c] =
                        filters_[c].processSample( static_cast<double>( data[s * nChannels_ + c] ) );
            }
            else {
                for( size_t s = 0; s < nSamples; ++s )
                    out[s * nChannels_ + c] = static_cast<double>( data[s * nChannels_ + c] );
            }
        }

        if( applyCar_ ) {
            std::vector<double> row( nChannels_ );
            for( size_t s = 0; s < nSamples; ++s ) {

                double *rowPtr = &out[s * nChannels_];
                for( int c = 0; c < nChannels_; ++c )
                    row[c] = rowPtr[c];

                double median = medianInPlace( row );
                for( int c = 0; c < nChannels_; ++c )
                    rowPtr[c] -= median;
            }
        }

        return out;
    }

private:
    static double medianInPlace( std::vector<double> &v )
    {
        size_t n = v.size();
        size_t mid = n / 2;
        std::nth_element( v.begin(), v.begin() + mid, v.end() );
        double hi = v[mid];
        if( n % 2 == 1 )
            return hi;
        std::nth_element( v.begin(), v.begin() + mid - 1, v.begin() + mid );
        double lo = v[mid - 1];
        return (lo + hi) / 2.0;
    }

    int  nChannels_;
    bool applyHighpass_;
    bool applyCar_;
    std::vector<ButterworthHighpass> filters_;
};

#endif // CLOSEDLOOP_PREPROCESSOR_H
