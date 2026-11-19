"""Prereg §2.8 family E2 (ELF schedule manipulation; written before data).
Usage: python analysis/elf_sched.py LATE_judged.json FRONT_judged.json
Per sample (target j = 64): Möbius N main = v(0,1,0) - v(0,0,0); Möbius F main = v(0,0,1) - v(0,0,0); T.
E2-1: N main (latejump) - N main (frontload) > 0.  E2-2: F main (frontload) - F main (latejump) > 0.
Unpaired across schedules (different trajectories); bootstrap over samples within each schedule (20,000).
Holm over 2, alpha 0.05. Required: T one-sided 95% LB > 0 in both schedules."""
import itertools, json, sys
import numpy as np
def load(p):
    J = json.load(open(p)); cells = next(iter(J.values())); n = len(next(iter(cells.values())))
    rows = []
    for i in range(n):
        v = {k: cells[f"O{k[0]}N{k[1]}F{k[2]}"][i] for k in itertools.product((0, 1), repeat=3)}
        if any(x is None for x in v.values()): continue
        b = v[(0, 0, 0)]; rows.append([v[(1, 1, 1)] - b, v[(0, 1, 0)] - b, v[(0, 0, 1)] - b])
    return np.array(rows)
L, Fr = load(sys.argv[1]), load(sys.argv[2]); rng = np.random.default_rng(20261113)
def bm(x, B=20000): return x[rng.integers(0, len(x), (B, len(x)))].mean(1)
bL, bF = {c: bm(L[:, c]) for c in range(3)}, {c: bm(Fr[:, c]) for c in range(3)}
ok = np.percentile(bL[0], 5) > 0 and np.percentile(bF[0], 5) > 0
for nm, X, b in (("latejump", L, bL), ("frontload", Fr, bF)):
    print(f"{nm}: n={len(X)} T {X[:,0].mean():+.3f} [{np.percentile(b[0],2.5):+.3f},{np.percentile(b[0],97.5):+.3f}]  N main {X[:,1].mean():+.3f}  F main {X[:,2].mean():+.3f}")
d1 = bL[1] - bF[1]; d2 = bF[2] - bL[2]
p = {"E2-1": float((d1 <= 0).mean()) if ok else 1.0, "E2-2": float((d2 <= 0).mean()) if ok else 1.0}
print(f"E2-1 N(late) - N(front) = {L[:,1].mean()-Fr[:,1].mean():+.3f} [{np.percentile(d1,2.5):+.3f},{np.percentile(d1,97.5):+.3f}] p={p['E2-1']:.3g}")
print(f"E2-2 F(front) - F(late) = {Fr[:,2].mean()-L[:,2].mean():+.3f} [{np.percentile(d2,2.5):+.3f},{np.percentile(d2,97.5):+.3f}] p={p['E2-2']:.3g}")
alive = True
for i, k in enumerate(sorted(p, key=p.get)):
    s = alive and p[k] <= 0.05 / (2 - i); alive = alive and s; print(f"  {k}: {'SUPPORTED' if s else 'not supported'} (Holm)")
