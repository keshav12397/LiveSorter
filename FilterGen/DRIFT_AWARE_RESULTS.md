# Drift-aware filter calibration: what 160 units say

Session: `D:/sim_probe_drift` (simulated, coherent probe drift, ~15 um true
span over 1800 s). Protocol: `calibrate_drift_aware.py --mode all
--split chronological --workers 6`, train = first 900 s, test = last 900 s.
Output: `D:/sim_probe_drift/filters_da_full`, log `calib_da_full.log`.

## Result

                        all 160 units          detectable 59 (global f1>=0.10)
    global                 0.1682                        0.4224
    segmented              0.1682  (+0.0000)             0.4224  (+0.0000)
    registered             0.2013  (+0.0331)             0.4990  (+0.0766)

`registered` is better on 109/160 units overall and 47/59 detectable ones.

The population mean is dominated by units that are undetectable at any
drift (median global f1 is 0.028), which is why the detectable subset is
reported separately -- a mean over mostly-zero units hides the effect.

## segmented is not merely close to global, it is IDENTICAL

Bit-identical f1 on all 160 units, and the reason is that the pooled
trajectory estimated a 14.8 um span, which at `tol_um=12` yields ONE
segment. Segmentation with a correct drift estimate did not fire at all on
this session.

That retro-explains the 6-unit smoke test, where `segmented` appeared to
beat `global` by +0.112. There the pooled estimate read **40.3 um** against
a ~15 um truth -- a small-N artifact, six units is not enough to average
down the per-unit localisation noise -- and that inflated span cut the train
half into 4 segments, so the deployed filter was simply the last 140 s of
training data. The apparent gain was recency, exactly as suspected.

Two things follow, and the second is the useful one:

1. `--mode recency` (added in 393ea2f) is no longer needed to settle this
   particular question, since segmentation provably did nothing here. Keep
   it: any future session whose true drift exceeds `tol_um` will segment
   for real, and the control is what will say whether that helped.
2. **A drift-span estimate is only trustworthy pooled over many units.**
   The estimator was validated at 1.05 um error on the full session with
   the full population, and it holds up here (14.8 vs ~15). It does not
   hold up on 6 units. Do not size segments from a small unit subset.

## What registration actually does

`registered` also runs a single segment -- it does not resegment. It uses
the trajectory to register the training waveforms to a common depth before
fitting, so the template is built from spikes aligned to one position
instead of smeared across the drift range. That is the half of the idea
that pays here, and it pays *because* it does not need the drift to exceed
a segmentation threshold to be worth applying. At 15 um of drift there is
no segment boundary to find, but there is still 15 um of smear to remove.

## The units it hurts

12 of 59 detectable units get worse, one substantially:

    unit 142   0.682 -> 0.493   -0.188
    unit 105   0.149 -> 0.069   -0.079
    unit 110   0.164 -> 0.131   -0.033

Not yet explained. Candidates worth checking before deploying registration
unconditionally: units whose residual (per-unit, non-probe) motion is large
compared to the coherent component, since registering by the POOLED
trajectory then moves them the wrong way; and units near a probe edge where
registration shifts support off the array.

## Protocol validity

This script's `global` control reproduces `calibrate_all_units.py` on the
same units to within 0.001 (0.4787 vs 0.4796 on the 6-unit set). The
comparison is therefore against the real production baseline, not against a
reimplementation of it -- which the earlier drift measurement never
established, and which is why that one's null result could not be trusted.
