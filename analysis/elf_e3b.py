"""Prereg §2.11 (E3b; written before its data). Usage: python analysis/elf_e3b.py <dir with own_ and swap_judged_fixedctx.json>
Paired by sample (complete cases in both variants, target j = 64). Quantity: phi = 1 - mean(F_swap) / mean(F_own),
the fraction of the F route's absolute Shapley effect removed when neighbours' final latents are fixed at decode.
Percentile bootstrap over samples (20,000, seed 20261113). Decision: 2.5th pct > 0.5 → SUPPORTED; 97.5th pct < 0.25 → REFUTED;
else INCONCLUSIVE. Required: mean F_own > 0 with its 2.5th pct > 0, else not estimable. Descriptive: same ratio for T and N."""
import json, itertools, math, sys
import numpy as np
keys = list(itertools.product((0, 1), repeat=3))
def per(path):
    J = json.load(open(path))["64"]; out = {}
    for i in range(len(J["O0N0F0"])):
        v = {k: J[f"O{k[0]}N{k[1]}F{k[2]}"][i] for k in keys}
        if any(x is None for x in v.values()): continue
        b = v[(0, 0, 0)]; val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
        sh = {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S))) for k in range(3)
                     for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}
        out[i] = [v[(1, 1, 1)] - b, sh["N"], sh["F"]]
    return out
d = sys.argv[1]; o, s = per(f"{d}/own_judged_fixedctx.json"), per(f"{d}/swap_judged_fixedctx.json")
c = sorted(set(o) & set(s)); A = np.array([o[i] for i in c]); B = np.array([s[i] for i in c]); m = len(c)
idx = np.random.default_rng(20261113).integers(0, m, (20000, m))
bF = A[idx, 2].mean(1); est = A[:, 2].mean() > 0 and np.percentile(bF, 2.5) > 0
print(f"paired n {m}; F_own {A[:,2].mean():+.3f} → {'estimable' if est else 'NOT ESTIMABLE'}")
for nm, col in (("T", 0), ("N", 1), ("F", 2)):
    r = 1 - B[:, col].mean() / A[:, col].mean(); br = 1 - B[idx, col].mean(1) / A[idx, col].mean(1)
    lo, hi = np.percentile(br, [2.5, 97.5]); print(f"{nm}: own {A[:,col].mean():+.3f} swap {B[:,col].mean():+.3f} phi {r:.3f} [{lo:.3f}, {hi:.3f}]")
    if nm == "F": dec = "NOT ESTIMABLE" if not est else "SUPPORTED" if lo > 0.5 else "REFUTED" if hi < 0.25 else "INCONCLUSIVE"
print("E3b:", dec)
