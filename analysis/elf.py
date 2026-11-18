"""Prereg §2.7 family E (ELF; written before data). Usage: python analysis/elf.py results/elf_decomp/cells_judged.json
Unit = sample (cluster); per sample, outcome = mean over targets j in {32, 64, 96} of the Mistral span log-prob
(complete-case per (sample, j): all 8 cells scored). E-1: mN - 0.2T > 0. E-2: Shapley(N) - 0.5T > 0. Holm over 2,
alpha 0.05, one-sided bootstrap over samples (20,000). Required: T one-sided 95% LB > 0, else 'not estimable'.
Descriptive: Shapley O and F shares."""
import itertools, json, math, sys
import numpy as np
J = json.load(open(sys.argv[1])); keys = list(itertools.product((0, 1), repeat=3))
n = len(next(iter(next(iter(J.values())).values())))
U = np.full((n, 5), np.nan); per = {i: [] for i in range(n)}
for j, cells in J.items():
    for i in range(n):
        v = {k: cells[f"O{k[0]}N{k[1]}F{k[2]}"][i] for k in keys}
        if any(x is None for x in v.values()): continue
        b = v[(0, 0, 0)]; T = v[(1, 1, 1)] - b; mN = v[(0, 1, 0)] - b
        val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
        sh = {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(Sx) | {p}) - val(set(Sx))) for k in range(3)
                  for Sx in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}
        per[i].append([T, mN, sh["N"], sh["O"], sh["F"]])
A = np.array([np.mean(per[i], 0) for i in range(n) if per[i]]); m = len(A); rng = np.random.default_rng(20261113)
def boot(x): return np.concatenate([(rng.multinomial(m, np.full(m, 1 / m), size=1000) @ x) / m for _ in range(20)])
bT = boot(A[:, 0]); est = A[:, 0].mean() > 0 and np.percentile(bT, 5) > 0
print(f"n samples {m}; T = {A[:,0].mean():+.3f} [{np.percentile(bT,2.5):+.3f},{np.percentile(bT,97.5):+.3f}] → {'estimable' if est else 'NOT ESTIMABLE'}")
print(f"Shapley shares: N {A[:,2].sum()/A[:,0].sum():.3f}, O {A[:,3].sum()/A[:,0].sum():.3f}, F {A[:,4].sum()/A[:,0].sum():.3f}; Möbius N {A[:,1].sum()/A[:,0].sum():.3f}")
res = {}
for k, x in (("E-1", A[:, 1] - 0.2 * A[:, 0]), ("E-2", A[:, 2] - 0.5 * A[:, 0])):
    b = boot(x); res[k] = float((b <= 0).mean()) if est else 1.0
    print(f"{k}: {x.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}] p={res[k]:.3g}")
order = sorted(res, key=res.get); alive = True
for i, k in enumerate(order):
    ok = alive and res[k] <= 0.05 / (2 - i); alive = alive and ok
    print(f"  {k}: {'SUPPORTED' if ok else 'not supported'} (Holm)")
