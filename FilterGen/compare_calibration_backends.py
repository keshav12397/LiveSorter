"""Do the python and cpp calibration backends produce the same filter bank?

Component-level equivalence tests already pin the LCMV solve, the noise
covariance, and the scorer against Python. What none of them cover is the
whole pipeline: candidate selection, interferer picking, RNG draws, and the
threshold sweep's tie-breaking all sit outside those tests.
"""
import io
import csv
import numpy as np

A = "D:/scratch/cal_py2"
B = "D:/scratch/cal_cpp"


def load(d):
    u = np.fromfile(d + "/unit_ids.bin", dtype=np.int32)
    ch = np.fromfile(d + "/channels.bin", dtype=np.int32)
    f = np.fromfile(d + "/filters.bin", dtype=np.float32)
    th = np.fromfile(d + "/thresholds.bin", dtype=np.float32)
    n = u.size
    nch = ch.size // n
    L = f.size // (n * nch)
    return dict(u=u, ch=ch.reshape(n, nch), f=f.reshape(n, L, nch), th=th,
                n=n, nch=nch, L=L)


def summary(d):
    rows = {}
    with io.open(d + "/summary.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["unit_id"])] = r
    return rows


a, b = load(A), load(B)
print("python : %d units, %d ch/unit, L=%d" % (a["n"], a["nch"], a["L"]))
print("cpp    : %d units, %d ch/unit, L=%d" % (b["n"], b["nch"], b["L"]))

print("\nunit ids")
print("  python:", a["u"].tolist())
print("  cpp   :", b["u"].tolist())
same_units = set(a["u"].tolist()) == set(b["u"].tolist())
print("  same set:", same_units)
if not same_units:
    print("  -> the two backends did not even select the same units;"
          " everything below is only over the overlap")

common = [x for x in a["u"].tolist() if x in set(b["u"].tolist())]
ia = {int(v): i for i, v in enumerate(a["u"])}
ib = {int(v): i for i, v in enumerate(b["u"])}

sa, sb = summary(A), summary(B)

print("\n%-6s %-22s %-22s %9s %9s" % ("unit", "channels py", "channels cpp",
                                       "thr py", "thr cpp"))
n_same_ch = 0
for uid in common:
    ca = a["ch"][ia[uid]].tolist()
    cb = b["ch"][ib[uid]].tolist()
    if sorted(ca) == sorted(cb):
        n_same_ch += 1
    print("%-6d %-22s %-22s %9.4f %9.4f"
          % (uid, ca, cb, a["th"][ia[uid]], b["th"][ib[uid]]))

print("\nchannel selection identical on %d/%d common units" % (n_same_ch, len(common)))

# Filter taps, only where the channel set matches (otherwise not comparable).
print("\nfilter taps, where channels match:")
for uid in common:
    ca = a["ch"][ia[uid]].tolist()
    cb = b["ch"][ib[uid]].tolist()
    if sorted(ca) != sorted(cb):
        print("  unit %-5d SKIP (different channels)" % uid)
        continue
    oa = np.argsort(ca)
    ob = np.argsort(cb)
    fa = a["f"][ia[uid]][:, oa]
    fb = b["f"][ib[uid]][:, ob]
    denom = max(float(np.abs(fa).max()), 1e-30)
    print("  unit %-5d max|dpy-cpp| = %.3e   rel = %.3e   corr = %.6f"
          % (uid, np.abs(fa - fb).max(), np.abs(fa - fb).max() / denom,
             np.corrcoef(fa.ravel(), fb.ravel())[0, 1]))

# What actually matters: does the bank detect the same?
print("\nf1 from each backend's own summary.csv:")
print("%-6s %9s %9s %9s" % ("unit", "f1 py", "f1 cpp", "delta"))
d = []
for uid in common:
    if uid in sa and uid in sb and sa[uid].get("f1") and sb[uid].get("f1"):
        fa1, fb1 = float(sa[uid]["f1"]), float(sb[uid]["f1"])
        d.append(fb1 - fa1)
        print("%-6d %9.4f %9.4f %+9.4f" % (uid, fa1, fb1, fb1 - fa1))
if d:
    d = np.array(d)
    print("mean delta %+.4f   max |delta| %.4f" % (d.mean(), np.abs(d).max()))
