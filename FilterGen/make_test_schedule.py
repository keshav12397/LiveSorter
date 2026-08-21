"""Write a drift_schedule.bin against the LIVE filter bank, to exercise the
swap path end to end on a real stream.

This is a wiring test, not a science claim. The swapped-in filters are the
bank's own taps for a channel set shifted by one probe row, so a swap is
observable (channel list changes, threshold changes) without inventing
filter values that mean nothing.

Layout must match FilterGen/calibrate_drift_aware.py's write_schedule() and
ClosedLoop/DriftSchedule.cpp's reader:
    header: int32 magic 'DRFT', version, nEvents, templateLength, nChans
    per event: float32 t_s, int32 unitIndex, int32 reserved,
               int32[nChans] channels, float32[L*nChans] filter,
               float32 threshold
"""
import argparse
import struct
import numpy as np

MAGIC = 0x44524654


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="dir with unit_ids/channels/filters/thresholds.bin")
    ap.add_argument("--out", required=True, help="dir to write drift_schedule.bin into")
    ap.add_argument("--times", default="3,6,9", help="comma-separated t_s for the swaps")
    ap.add_argument("--row-shift", type=int, default=2,
                    help="channel offset per swap (2 = one row on a 2-column probe)")
    a = ap.parse_args()

    unit_ids = np.fromfile(a.bank + "/unit_ids.bin", dtype=np.int32)
    channels = np.fromfile(a.bank + "/channels.bin", dtype=np.int32)
    filters = np.fromfile(a.bank + "/filters.bin", dtype=np.float32)
    thresholds = np.fromfile(a.bank + "/thresholds.bin", dtype=np.float32)

    n_units = unit_ids.size
    n_ch = channels.size // n_units
    L = filters.size // (n_units * n_ch)
    channels = channels.reshape(n_units, n_ch)
    filters = filters.reshape(n_units, L * n_ch)

    # Which channels the CAR group actually contains bounds the shift: a swap
    # naming a channel outside the group is a legitimate error the fetch
    # thread must reject, but it is not what this test is for.
    lo, hi = int(channels.min()), int(channels.max())

    times = [float(t) for t in a.times.split(",")]
    events = []
    for i, t in enumerate(times):
        # Rotate through units so successive swaps hit different ones.
        u = (i * 7) % n_units
        shift = a.row_shift * (i + 1)
        newch = channels[u] + shift
        if newch.max() > hi or newch.min() < lo:
            newch = channels[u] - shift
        if newch.max() > hi or newch.min() < lo:
            print("unit %d: cannot shift %+d within [%d,%d], skipping"
                  % (u, shift, lo, hi))
            continue
        events.append(dict(
            t_s=t, unit_index=u,
            channels=newch.astype(np.int32),
            # Same taps, scaled slightly so the swap is distinguishable in
            # any downstream comparison without being a different filter in
            # kind.
            filt=(filters[u] * 1.01).astype(np.float32),
            threshold=float(thresholds[u]) * 1.05,
        ))

    path = a.out + "/drift_schedule.bin"
    with open(path, "wb") as fh:
        fh.write(struct.pack("<iiiii", MAGIC, 1, len(events), L, n_ch))
        for ev in events:
            fh.write(struct.pack("<fii", ev["t_s"], ev["unit_index"], 0))
            fh.write(ev["channels"].tobytes())
            fh.write(ev["filt"].tobytes())
            fh.write(struct.pack("<f", ev["threshold"]))

    print("wrote %s: %d events, L=%d nCh=%d" % (path, len(events), L, n_ch))
    for ev in events:
        print("  t=%.1fs unit_index=%-3d ks_id=%-4d channels %s -> %s  thr %.4f -> %.4f"
              % (ev["t_s"], ev["unit_index"], unit_ids[ev["unit_index"]],
                 channels[ev["unit_index"]].tolist(), ev["channels"].tolist(),
                 thresholds[ev["unit_index"]], ev["threshold"]))


if __name__ == "__main__":
    main()
