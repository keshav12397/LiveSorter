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

## Part B: does refitting at the drifted position recover f1?

Yes, and the cheap in-band refit gets all of it.

`refit_drift_position_ablation.py`, same session and same chronological
split, three arms per unit with ONE interferer set fixed across all three
(so an arm-to-arm difference cannot be a different interferer set):

    never_refit     the train-half filter applied to the test half unchanged
                    -- the status quo, and the baseline
    cheap_refit_B   banded_refit.refit_in_band at the predicted drifted
                    position: motion-corrected template, re-selected
                    channels, R assembled from the band. No data access.
    full_refit_B    a genuine rescan of window B (the late train quarter) at
                    the same predicted position

160 units attempted, **n = 134 complete**. The 26 skips are all the same
condition -- "not enough trajectory bins to register", i.e. fewer than two
150-spike bins, so under ~300 train spikes. That is a real limit of the
method and not a harness failure: a unit that never fires enough to build a
trajectory cannot have its motion estimated, and there is nothing to
register it to.

### The result

    cheap_refit_B - never_refit    mean +0.0566  median +0.0151   111 up / 20 down
    full_refit_B  - never_refit    mean +0.0569  median +0.0149   109 up / 20 down
    cheap_refit_B - full_refit_B   mean -0.0002  median +0.0000    60 up / 60 down

The third line is the one the online design rests on. **A 2.4 ms
assemble-and-solve is indistinguishable from a full rescan** -- an exact
60/60 split and a mean of -0.0002. Banding costs nothing in f1, which is
what `test_noise_cov_banded.py` proves analytically and this measures
empirically.

### Split by baseline detectability

Detectability here is bimodal, so a pooled mean describes the mixture rather
than the method. Broken out by `never_refit` f1:

    band                        n     never -> cheap    mean d   improved
    usable    (f1 >= 0.30)     29     0.649 -> 0.742    +0.092    23/29
    marginal  (0.10-0.30)      26     0.179 -> 0.307    +0.128    26/26
    undetected(f1 < 0.10)      79     0.024 -> 0.044    +0.020    62/79

The marginal band is where this matters most: 26 of 26 improve, and the mean
moves from unusable toward usable. The 20 regressions are nearly all in the
undetected band (only 4 have baseline f1 >= 0.30) and the worst is -0.012 --
noise on units that were not trackable either way.

### The prediction that failed

The benefit was expected to scale with how far a unit drifted. It does not:

    corr(drift_span_um, delta)   all 134 units   -0.187
                                 usable only     +0.145

Both are approximately nothing, and the reason is visible in the data. The
drift in this session is COHERENT: the pooled shift is 11.05 um with an SD
of 0.085 um across all 134 units. Every unit receives essentially the same
correction, so there is no variation in applied shift for a benefit to
correlate with. Per-unit `drift_span_um` (mean 33, SD 34, max 188) is
dominated by trajectory-estimation noise on low-count units, not by real
differential motion.

So the absence of a correlation here is not evidence against the mechanism;
it is evidence that this dataset cannot test that particular claim. A
session with genuinely non-rigid drift could.

### What this does and does not license

It licenses the online design: scan a band once per unit under its own
exclusion mask, and refit from it whenever the motion estimate says the unit
moved. Part A priced the covariance window (60 s is ~free); Part B prices
the refit itself (in-band == full rescan). Both halves are now measured.

It does not license claims about real recordings. This is the 160-unit
simulator with imposed coherent drift, and the same caveat
`DRIFT_AWARE_RESULTS.md` carries applies unchanged: the amplitudes,
waveforms, and drift are all drawn by the simulator. The real test is
`D:/catgt_Lav69_d1.0_g0` against Kilosort's own `dshift`, which has not
been run.
