"""Prereg §2.5 family R analysis (written before data). Usage: python analysis/gc.py RESULTS_ROOT  (gc_warp, gc_uniform)."""
import glob, json, sys
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "src/results"
def load(d): return [json.loads(l)["result"] for f in glob.glob(f"{root}/{d}/shard_*.jsonl") for l in open(f)]
def eff(R, c): return np.array([np.mean([r[f"j{j}|base"]["dep"] - r[f"j{j}|{c}"]["dep"] for j in (2, 3)]) for r in R])
rng = np.random.default_rng(20261113)
def boot(x):
    n = len(x); B = np.concatenate([(rng.multinomial(n, np.full(n, 1/n), size=1000) @ x) / n for _ in range(20)]); return B
W, U = load("gc_warp"), load("gc_uniform")
x1 = eff(W, "late_neither") - eff(W, "late_nofb") - eff(W, "late_nostate"); b1 = boot(x1); p1 = float((b1 <= 0).mean())
a = eff(U, "late_nofb"); ba = boot(a); pa = float((ba >= 0.10).mean())
c = eff(U, "late_neither"); bc = boot(c); pc = float((bc <= 0).mean()); p2 = max(pa, pc)
x3 = eff(U, "mid_nofb") - eff(U, "late_nofb"); b3 = boot(x3); p3 = float((b3 <= 0).mean())
res = {"R1": (x1.mean(), b1, p1), "R2": (a.mean(), ba, p2), "R3": (x3.mean(), b3, p3)}
print(f"n warp {len(W)}, uniform {len(U)}")
print(f"R1 redundancy index (warp late) {x1.mean():+.3f} [{np.percentile(b1,2.5):+.3f},{np.percentile(b1,97.5):+.3f}] p={p1:.3g}")
print(f"R2 uniform late nofb {a.mean():+.3f} (upper 95% {np.percentile(ba,95):+.3f} vs 0.10, p={pa:.3g}); neither {c.mean():+.3f} [{np.percentile(bc,2.5):+.3f},{np.percentile(bc,97.5):+.3f}] p={pc:.3g} → p={p2:.3g}")
print(f"R3 uniform nofb mid − late {x3.mean():+.3f} [{np.percentile(b3,2.5):+.3f},{np.percentile(b3,97.5):+.3f}] p={p3:.3g}")
order = sorted(res, key=lambda k: res[k][2]); alive = True
for i, k in enumerate(order):
    ok = alive and res[k][2] <= 0.05 / (3 - i); alive = alive and ok
    print(f"  {k}: p={res[k][2]:.3g} → {'SUPPORTED' if ok else 'not supported'} (Holm)")
