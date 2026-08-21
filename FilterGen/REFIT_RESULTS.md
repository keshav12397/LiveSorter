# Refitting a drifted unit: what it costs, and what does not work

Companion to `DRIFT_AWARE_RESULTS.md`. Session: `D:/sim_probe_drift`
(simulated, coherent probe drift, 160 units). Protocol:
`refit_exclusion_ablation.py --split-mode chronological`, train = first 900 s,
test = last 900 s, scoring via `calibrate_drift_aware.py`'s own primitives.

## The question

A unit that drifts moves off the channels its filter reads. Refitting it needs
a new channel selection and a new LCMV solve, and the solve needs R -- the
space-time noise covariance for whatever channels get selected. Computing R is
a pass over the recording. Doing that per swap, per unit, online, is not
possible, so the question is what can be precomputed.

## What does not work: one covariance for the whole probe

`generate_filter.noise_covariance_from_lags` can assemble R for any subset of
whatever channels were scanned, so the obvious move is to scan every channel
once and share that array across every unit. **It does not work, and this repo
already said so** before the idea was raised again here:

- `ClosedLoop/NoiseCovariance.h:11-14` -- "the original 'compute once for the
  whole probe' idea doesn't work (an all-clusters mask leaves 0 usable
  spike-free segments on a real, busy recording -- checked empirically before
  writing this)"
- `generate_filter.py`, single-target fit -- "masking out every single spike
  from every cluster ... can blanket the entire recording and leave no
  spike-free segments at all, even though most of those spikes are on channels
  far from our local channel subset"

The second gives the better reason, and it is about **relevance, not density**:
"not noise" should mean *could contaminate these channels*, not *any spike
anywhere on the probe*. A probe-wide mask discards time on account of spikes
that never appear in `data_sel` at all.

Re-measured here, on the 900 s train half with every sorted spike excluded:

    99.53% of samples are spike-present
    13 gaps >= 122 samples, totalling 1869 samples (0.06 s)
    against 27,000,000 samples for the per-unit exclusion

Usable data accrues at ~2.1 samples per second of recording, so reaching even
a bare sufficiency floor (~3050 samples, 10 per dimension of a 5-channel R)
would need ~1450 s of train half. It does not exist. This is structural, not a
tuning problem.

**The failure is silent, which is what makes it dangerous.**
`noise_cov_by_lag` raises only when ZERO usable gaps survive. Thirteen gaps is
not zero, so at 900 s it returns a correctly-shaped `(121, 96, 96)` array
estimated from 0.06 s of data and looks exactly like a real answer. Three
successive attempts failed differently: capped at 60 s it raised; uncapped it
returned garbage; and "grow the window until it stops raising" stops at the
first window that produces a shaped array, which is not the same as a usable
one. `generate_filter.noise_cov_by_lag` now warns below 50*template_length.

Also worth recording, because it produced a wrong intermediate answer:
coverage cannot be estimated as (spike count x blanking width). That
arithmetic gives 5.2x oversubscribed where the true union coverage is 99.5% --
the intervals overlap heavily at 1570 spikes/s. Measure the union.

`shared_none` (exclude nothing) is computable but was dropped too, for a
different reason: it puts every spike's energy into R, and R is what LCMV
minimises output variance against, so spike energy in R is precisely what the
estimate exists to remove. Not a design anyone would ship.

## What does work: a band per unit

Exclusion stays per unit -- its own target plus interferers, ~6 clusters, the
set `fit_lcmv` already uses and the only one that leaves usable gaps.
Cheapness comes from CHANNELS instead: scan a band around the unit, wider than
the 5 channels it will select. Every post-drift selection inside the band is
then assemble-and-solve with no data access. `banded_refit.py`;
`test_banded_refit.py` and `test_noise_cov_banded.py` pin that this is exact
rather than approximate.

    n_ch                        cost            900 s train half
      5  (a single fit today)   2.3 us/sample    ~62 s per unit
     16  (band, refit-capable)  15.8 us/sample   ~427 s per unit

~7x the current per-unit cost offline. A refit itself is **2.4 ms** (assemble
+ 305x305 solve).

## How much data the covariance actually needs

Measured directly, since the rolling-window online design depends on it:
per-unit exclusion with R built from the whole 900 s train half (`peruser`)
against the same exclusion with R capped to the first 60 s
(`peruser_capped`) -- 15x less data, everything else identical.

    unit     peruser    capped     delta      rec_d   prec_d
     29        0.083     0.078    -0.006     -0.019   -0.003
     35        0.813     0.786    -0.026     -0.059   +0.030
     56        0.878     0.817    -0.062     +0.024   -0.114
     80        0.087     0.096    +0.009     +0.014   +0.007
    112        0.471     0.489    +0.018     -0.058   +0.049
    122        0.514     0.536    +0.022     +0.013   +0.025

    mean f1  0.4743 -> 0.4669     mean delta -0.0075, median +0.0017
                                  worse on 3/6

**A short covariance window is close to free.** The sign splits evenly, the
median is slightly positive, and the mean is carried by one unit (56, -0.062,
driven by precision). 60 s is ~1.8M samples per channel, far above the
sufficiency floor, so this is not a data-starvation regime.

That is what a rolling-window online refit needs: R can be maintained over
recent data rather than the whole session, at negligible cost in f1.

**n = 6.** The direction is clear and the spread (-0.062 to +0.022) is not
wide, but this is a smoke-test population, not an estimate with an error bar
on it. Worth repeating at 160 units before it is load-bearing.

## Still open

Whether refitting at a drifted position recovers f1 -- the question part B of
this investigation was for. The mechanism is now built and tested
(`banded_refit.py`) and its cost is known; what has not been measured is the
f1 difference between never refitting, refitting cheaply in-band, and a full
from-scratch fit at the new position.
