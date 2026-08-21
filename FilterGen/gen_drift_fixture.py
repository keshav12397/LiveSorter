"""
gen_drift_fixture.py
=====================

Numerical fixture for ClosedLoop/test_livewire_roundtrip.cpp's drift-pool
check: calls the real, already-validated drift_estimate.pooled_com_motion()
on a synthetic-but-realistic multi-unit recording with injected spatial
drift, and dumps everything the C++ port (Viewer/DriftPool.h's
pooledMedianMotion()) needs to reproduce its POOLING step -- same "port
validated against Python" pattern as gen_lcmv_fixture.py.

Why this only tests pooling, not the whole pipeline
-----------------------------------------------------
pooled_com_motion() has two stages: (1) per-unit centroid trajectories from
mean_waveform-averaged, equal-spike-count bins (unit_trajectory), and (2) a
median-across-units pool with per-unit mean subtraction, NaN-safe median,
and gap-fill (the `_pool` closure). The live viewer's DriftTracker.h does
NOT reimplement stage (1): the wire protocol never carries raw waveforms
(see LiveWire.h's design note -- the fetch loop's <10 ms budget rules that
out), so DriftTracker.h builds its per-unit trajectories from streamed
per-spike peak amplitudes and fixed time bins instead, which is a
structurally different computation with no single Python function to check
it against bit-for-bit. What IS a direct, checkable port is stage (2), so
that is what this fixture isolates: it builds Y (each unit's own
mean-subtracted per-bin displacement, NaN outside that unit's covered
bins) and grid by calling the real unit_trajectory() per unit -- the exact
lines pooled_com_motion() itself runs before handing off to `_pool` -- and
separately captures pooled_com_motion()'s own actual return value as the
expected pooled trace. Note the rng is never actually consumed by either
path here (see below), so there is no hidden dependency on call order
between the two.

No randomness actually enters the comparison
-----------------------------------------------
unit_trajectory()'s only randomness is mean_waveform()'s rng.choice(),
which only fires when a bin has MORE than max_bin_spikes (400) spikes.
This fixture keeps every unit's spikes-per-bin well under that (~150-200),
so mean_waveform never subsamples and rng.choice() is never called --
which means computing Y via a fresh per-unit unit_trajectory() call here is
bit-identical to what happened inside pooled_com_motion()'s own internal
loop, without needing to reproduce its shared-rng call order.

Binary layout (float64 unless noted, C order), written to --out:
    int32 nUnits, int32 nTime
    Y            (nUnits, nTime)   -- NaN where a unit has no bin estimate
    grid         (nTime,)          -- bin center times, seconds
    motion_expect (nTime,)         -- pooled_com_motion(...)[1][0], the
                                       actual rigid (n_windows=1) trace
"""
import argparse
import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_estimate as de


def build_synthetic_recording(rng, n_units, n_channels, n_samples,
                               template_length, template_offset,
                               spikes_per_unit, fs, chan_y):
    """A noisy recording with `n_units` synthetic spikes, each unit's
    spatial footprint a Gaussian bump over channels whose CENTER drifts
    linearly over the recording -- enough to give pooled_com_motion()
    something nontrivial (and non-identical across units) to pool, without
    needing the full make_sim_session.py simulator.
    """
    data = rng.normal( 0.0, 2.0, size=(n_samples, n_channels) ).astype(np.float64)

    t_idx = np.arange(template_length)
    # A single negative deflection back to baseline -- ptp of the injected
    # part alone equals its peak amplitude, same as a real spike waveform.
    temporal = -np.exp(-0.5 * ((t_idx - template_offset) / 2.5) ** 2)

    spike_times_list = []
    spike_clusters_list = []

    anchors = np.linspace(3.0, n_channels - 4.0, n_units)
    drift_channels = rng.uniform(-2.5, 2.5, size=n_units)
    amps = rng.uniform(30.0, 50.0, size=n_units)

    margin = template_length + template_offset + 4
    # Each unit is only active over a random sub-window of the recording
    # (not the whole thing) -- this is what makes Y genuinely contain NaN
    # gaps to test: a unit's own trajectory is NaN outside its own span
    # (unit_trajectory's np.interp(..., left=nan, right=nan)), and a
    # fixture where every unit spans the whole recording would never
    # exercise that path.
    active_frac = rng.uniform(0.35, 1.0, size=n_units)
    active_start = rng.uniform(0.0, 1.0 - active_frac)
    for u in range(n_units):
        lo = margin + int(active_start[u] * (n_samples - 2 * margin))
        hi = margin + int((active_start[u] + active_frac[u]) * (n_samples - 2 * margin))
        hi = max(hi, lo + 1)
        st = np.sort(rng.integers(lo, hi, size=spikes_per_unit))
        spike_times_list.append(st)
        spike_clusters_list.append(np.full(spikes_per_unit, u, dtype=np.int64))

        frac = st.astype(np.float64) / n_samples
        centers = anchors[u] + drift_channels[u] * frac
        for k, s in enumerate(st):
            channels = np.arange(n_channels)
            spatial = np.exp(-0.5 * ((channels - centers[k]) / 1.3) ** 2)
            bump = amps[u] * np.outer(temporal, spatial)
            lo = s - template_offset
            data[lo:lo + template_length, :] += bump

    spike_times = np.concatenate(spike_times_list)
    spike_clusters = np.concatenate(spike_clusters_list)
    return data, spike_times, spike_clusters


def replicate_pre_pool(data, spike_times, spike_clusters, chan_y,
                        template_length, template_offset, fs, unit_ids,
                        bin_s, spikes_per_bin, n_top):
    """The exact lines pooled_com_motion() runs before `_pool` -- see the
    module docstring for why duplicating this little glue (not the
    underlying math) is legitimate here.
    """
    n_samples = data.shape[0]
    n_time = max(int(np.ceil(n_samples / (bin_s * fs))), 2)
    grid = (np.arange(n_time) + 0.5) * (n_samples / n_time) / fs

    disp = []
    for uid in unit_ids:
        st = spike_times[spike_clusters == uid]
        if st.size < 4 * spikes_per_bin:
            continue
        t_c, y = de.unit_trajectory(data, st, chan_y, template_length, template_offset,
                                    fs, spikes_per_bin=spikes_per_bin, n_top=n_top,
                                    rng=None, smooth=1)
        ok = np.isfinite(y)
        if ok.sum() < 4:
            continue
        yi = np.interp(grid, t_c[ok], y[ok], left=np.nan, right=np.nan)
        if np.all(np.isnan(yi)):
            continue
        disp.append(yi - np.nanmean(yi))

    Y = np.asarray(disp) if disp else np.zeros((0, n_time))
    return Y, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-units", type=int, default=6)
    ap.add_argument("--n-channels", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=1_000_000)
    ap.add_argument("--spikes-per-unit", type=int, default=800)
    ap.add_argument("--template-length", type=int, default=21)
    ap.add_argument("--template-offset", type=int, default=10)
    ap.add_argument("--fs", type=float, default=30000.0)
    ap.add_argument("--bin-s", type=float, default=20.0)
    ap.add_argument("--spikes-per-bin", type=int, default=150)
    ap.add_argument("--n-top", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    chan_y = np.arange(args.n_channels, dtype=np.float64) * 15.0

    data, spike_times, spike_clusters = build_synthetic_recording(
        rng, args.n_units, args.n_channels, args.n_samples,
        args.template_length, args.template_offset, args.spikes_per_unit,
        args.fs, chan_y)

    unit_ids = list(range(args.n_units))

    t_center_s, motion, win_centers_um = de.pooled_com_motion(
        data, spike_times, spike_clusters, chan_y,
        args.template_length, args.template_offset, args.fs, unit_ids,
        bin_s=args.bin_s, spikes_per_bin=args.spikes_per_bin, n_top=args.n_top,
        n_windows=1, rng=np.random.default_rng(args.seed))
    motion_expect = motion[0]

    Y, grid = replicate_pre_pool(
        data, spike_times, spike_clusters, chan_y,
        args.template_length, args.template_offset, args.fs, unit_ids,
        args.bin_s, args.spikes_per_bin, args.n_top)

    assert grid.shape == t_center_s.shape
    assert np.allclose(grid, t_center_s), "fixture's own grid must match pooled_com_motion's"

    nUnits, nTime = Y.shape
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<ii", nUnits, nTime))
        fh.write(Y.astype("<f8").tobytes())
        fh.write(grid.astype("<f8").tobytes())
        fh.write(motion_expect.astype("<f8").tobytes())

    nFiniteBins = int(np.isfinite(Y).any(axis=0).sum()) if nUnits else 0
    print(f"Wrote fixture to {args.out} (nUnits={nUnits}, nTime={nTime}, "
          f"{nFiniteBins} bins covered by >=1 unit)")
    print(f"motion_expect range: [{motion_expect.min():.3f}, {motion_expect.max():.3f}] um")


if __name__ == "__main__":
    main()
