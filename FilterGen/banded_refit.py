"""
banded_refit.py -- refit one unit's LCMV filter at a new position without
touching the recording again.

The problem
-----------
A unit that drifts moves off the channels its filter reads. Fixing that needs
a new channel selection and a new solve, and the solve needs R, the space-time
noise covariance for whatever channels get selected. Computing R is a pass
over the recording; doing that per swap, per unit, online, is not possible.

What makes it possible
----------------------
R is block-Toeplitz: block (i, k) is the covariance between channel i's
`template_length`-sample window and channel k's, and depends only on the lag
between the two samples. So the block reads exactly one channel pair's
covariance across lags, and R for ANY subset of scanned channels is an index
plus an assembly -- `generate_filter.noise_covariance_from_lags`. Measured:
2.4 ms for a 5-channel selection, against ~286 s to rescan 400k samples.
`test_noise_cov_subset.py` proves the assembly is exact.

Why a BAND and not the whole probe
----------------------------------
The obvious move is to scan every channel once and share that array across
every unit. It does not work, and this repo said so before this module
existed:

    ClosedLoop/NoiseCovariance.h:11-14 -- "the original 'compute once for
    the whole probe' idea doesn't work (an all-clusters mask leaves 0 usable
    spike-free segments on a real, busy recording -- checked empirically
    before writing this)"

    generate_filter.py's single-target fit -- "masking out every single
    spike from every cluster ... can blanket the entire recording and leave
    no spike-free segments at all, even though most of those spikes are on
    channels far from our local channel subset"

The second gives the real reason, and it is about relevance rather than
density: "not noise" should mean *could contaminate these channels*, not
*any spike anywhere on the probe*. A probe-wide mask throws away time on
account of spikes that never appear in `data_sel`. Measured on the 160-unit
simulator, a probe-wide mask leaves 99.5% of the train half spike-present --
1869 usable samples out of 27,000,000.

So the exclusion set stays per unit (its own target plus interferers, ~6
clusters), and cheapness comes from channels instead: scan a BAND around the
unit, wider than the 5 channels it will select, under that unit's own mask.
Every post-drift selection inside the band is then assemble-only.

    n_ch                       cost           900 s train half
      5  (a single fit today)  2.3 us/sample   ~62 s per unit
     16  (band, refit-capable) 15.8 us/sample  ~427 s per unit

About 7x the current per-unit cost offline. Online it is cheaper still: a
live refit only needs a rolling recent window, not the whole recording.

What this module does NOT do
----------------------------
- It does not estimate drift. See `drift_estimate.pooled_com_motion`.
- It does not build the registered template. See
  `motion_correct.registered_template`; pass its output in as `target_wf`.
- It does not re-derive channel selection or the LCMV solve. It calls
  `generate_filter.select_channels` and `generate_filter.lcmv_filter`, which
  are the same functions every calibration path uses. This repo has twice
  silently lost recall to a second, drifted implementation of that math.
"""

import numpy as np

import generate_filter as gf


def band_for_depth(chan_y, center_um, half_width_um=60.0, min_channels=12):
    """Indices of the channels within `half_width_um` of `center_um`.

    Returns indices into `chan_y` (i.e. into the preprocessed channel group),
    ascending. Widens symmetrically until at least `min_channels` are
    included, so a unit near a probe edge still gets a usable band instead of
    a truncated one.

    `half_width_um` has to cover the drift the unit will actually undergo
    plus the spatial extent of its own footprint, because the post-drift
    SELECTION must land inside the band -- outside it, there is no covariance
    to assemble from and the caller has to rescan. 60 um is four rows at this
    rig's 15 um pitch and comfortably covers the ~30 um excursions measured
    here; it is not a bound on what drift can do.
    """
    chan_y = np.asarray(chan_y, dtype=np.float64)
    d = np.abs(chan_y - float(center_um))
    band = np.flatnonzero(d <= half_width_um)
    if band.size < min_channels:
        # Take the nearest min_channels instead of widening blindly, so the
        # band stays centred on the unit rather than drifting toward
        # whichever side has more channels.
        band = np.argsort(d, kind="stable")[:min(min_channels, chan_y.size)]
        band = np.sort(band)
    return band


def scan_band(data, band, exclusion_spike_times, template_length,
              template_offset, max_samples=None):
    """Per-lag noise covariance over `band` only, under this unit's own mask.

    `exclusion_spike_times` must be the unit's OWN target + interferer spike
    times -- the same set `threshold_sweep_real.fit_lcmv` uses, and NOT every
    sorted spike. See the module docstring for why the probe-wide set is both
    wrong in principle and usually impossible in practice.

    Returns `(2*template_length-1, len(band), len(band))`, indexed within the
    band. Keep it alongside `band`: the pair is what every later refit reads,
    and an index into one without the other is meaningless.
    """
    band = np.asarray(band, dtype=int)
    sub = data[:, band]
    n = sub.shape[0] if max_samples is None else min(int(max_samples), sub.shape[0])
    return gf.noise_cov_by_lag(sub, exclusion_spike_times, template_length,
                               template_offset, n)


def refit_in_band(cov_band, band, target_wf, interferer_wfs, n_channels,
                  template_length, ridge=1e-3):
    """Select channels and solve, using only `cov_band`. No data access.

    `target_wf` and `interferer_wfs` are full-group waveforms
    (template_length, n_group_channels) -- the same shape
    `generate_filter.mean_waveform` returns, so a motion-corrected template
    from `motion_correct.registered_template` drops straight in.

    Returns `(f, sel_global, sel_in_band)`:
        f            (template_length, n_channels) filter taps
        sel_global   channel indices into the full group, for the caller's
                     bank -- these still need translating to raw SpikeGLX ids
                     the same way every other path does
        sel_in_band  the same channels indexed within `band`

    Raises ValueError if the selection escapes the band. That is a real
    condition and not a defensive check: it means the unit drifted further
    than the band was built to cover, so no covariance exists for the
    channels it now wants, and the honest response is to rescan a wider band
    rather than assemble R from whatever happens to be in range.

    One caveat on exactness. `select_channels` scores every channel and takes
    the top `n_channels`, breaking ties by index order -- so its result is
    not invariant to the index space it is handed, and a band-relative call
    can resolve an exact tie to a different channel than a group-relative one
    would. Scores are identical for any channel present in both, so this only
    bites on an exact tie, which needs two channels with bit-identical target
    AND interferer energy. Real probes avoid that because the columns sharing
    a row sit at different x and therefore different distances from the unit;
    a synthetic fixture that builds waveforms from depth alone hits it every
    time, which is how it was found. Worth knowing before treating band and
    full-group selection as interchangeable in a proof.
    """
    band = np.asarray(band, dtype=int)

    # Selection is restricted to the band by construction: scoring channels
    # the band does not contain could pick one, and there would be no
    # covariance for it. Restricting the SCORE rather than clipping the
    # RESULT matters -- clipping would silently return a different filter
    # from the one the scores asked for.
    tgt_band = target_wf[:, band]
    int_band = [wf[:, band] for wf in interferer_wfs]

    sel_in_band = gf.select_channels(tgt_band, int_band, n_channels)
    sel_in_band = np.asarray(sel_in_band, dtype=int)
    if sel_in_band.size < n_channels:
        raise ValueError(
            "band holds only {} usable channels, need {} -- widen "
            "half_width_um".format(sel_in_band.size, n_channels))
    if sel_in_band.max() >= band.size or sel_in_band.min() < 0:
        raise ValueError("selection escaped the band")

    R = gf.noise_covariance_from_lags(cov_band, sel_in_band, template_length)

    s = tgt_band[:, sel_in_band]
    ints = [wf[:, sel_in_band] for wf in int_band]
    s_flat = s.T.ravel()
    int_flats = [wf.T.ravel() for wf in ints]
    f_flat = gf.lcmv_filter(s_flat, int_flats, R, ridge=ridge)

    f = f_flat.reshape(sel_in_band.size, template_length).T
    return f, band[sel_in_band], sel_in_band
