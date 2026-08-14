#ifndef CLOSEDLOOP_BUTTERWORTHHIGHPASS_H
#define CLOSEDLOOP_BUTTERWORTHHIGHPASS_H

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Causal 2nd-order (12dB/oct) Butterworth high-pass biquad, one instance
// per channel (each channel needs its own filter *state*, even though all
// instances share the same coefficients for a given cutoff/sample rate).
//
// FilterGen/generate_filter.py's highpass() uses scipy's butter(2, Wn,
// 'high') + filtfilt -- zero-phase (forward AND backward), which is
// fundamentally non-causal and impossible to replicate exactly in a live
// low-latency system (there is no "backward through not-yet-fetched
// future data"). This uses the standard RBJ Audio-EQ-Cookbook biquad
// formula with Q = 1/sqrt(2) (maximally flat / Butterworth response) as
// the causal equivalent -- same cutoff and rolloff shape, applied forward
// only, with an inherent phase shift filtfilt wouldn't have. That's an
// accepted, deliberate approximation (real-time DSP always uses causal
// filters), not a bug -- it corrects the same failure mode discovered
// live (huge low-frequency-driven correlation values without any
// high-pass at all), just not sample-for-sample identical to Python's
// offline zero-phase result.
class ButterworthHighpass {
public:
    ButterworthHighpass( double cutoffHz, double sampleRateHz )
    {
        const double Q = 0.70710678118654752; // 1/sqrt(2) -- Butterworth (maximally flat) Q
        const double w0 = 2.0 * M_PI * cutoffHz / sampleRateHz;
        const double cosw0 = std::cos( w0 );
        const double alpha = std::sin( w0 ) / (2.0 * Q);

        double a0 =  1.0 + alpha;
        b0_ = ( (1.0 + cosw0) / 2.0 ) / a0;
        b1_ = ( -(1.0 + cosw0) )      / a0;
        b2_ = ( (1.0 + cosw0) / 2.0 ) / a0;
        a1_ = ( -2.0 * cosw0 )        / a0;
        a2_ = ( 1.0 - alpha )         / a0;

        reset();
    }

    void reset()
    {
        x1_ = x2_ = y1_ = y2_ = 0.0;
    }

    // Direct-form-I biquad, one new sample in, one filtered sample out.
    double processSample( double x )
    {
        double y = b0_ * x + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;

        x2_ = x1_; x1_ = x;
        y2_ = y1_; y1_ = y;

        return y;
    }

private:
    double b0_, b1_, b2_, a1_, a2_;
    double x1_, x2_, y1_, y2_;
};

#endif // CLOSEDLOOP_BUTTERWORTHHIGHPASS_H
