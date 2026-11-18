"""Prereg §2.6 family S7 (7B mechanism, written before data). Usage: python analysis/s7b.py RESULTS_ROOT
Reads <root>/conf7b_h2n (7B H2-N cells, own-AR coh/dep) and <root>/conf7b_judge (Mistral spans per cell).
S7-1: contrast mN - 0.2 T > 0 on coh_{2,3} (own-AR).   S7-2: Shapley N - 0.5 T > 0 (N carries >= half), coh own-AR.
S7-3: Shapley O > 0 on coh_{2,3} (own feedback contributes at 7B). Holm over 3, alpha 0.05, one-sided cluster
bootstrap by document (20,000). Co-requirement for S7-2: Mistral Shapley N share point estimate >= 0.5."""
import glob, itertools, json, math, sys
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "src/results"
def load(d):
    R = {}
    for f in glob.glob(f"{root}/{d}/shard_*.jsonl"):
        for l in open(f):
            r = json.loads(l); R[r["task_id"]] = r["result"]
    return R
H, J = load("conf7b_h2n"), load("conf7b_judge")
def decomp(v):
    b = v[(0, 0, 0)]; T = v[(1, 1, 1)] - b; mN = v[(0, 1, 0)] - b
    val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
    sh = {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(Sx) | {p}) - val(set(Sx))) for k in range(3)
              for Sx in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}
    return T, mN, sh
rows, jrows = [], []
for tid, r in H.items():
    a = np.zeros(4); ok = True
    for j in (2, 3):
        v = {k: r["cells"][f"j{j}_O{k[0]}N{k[1]}F{k[2]}"]["coh"] for k in itertools.product((0, 1), repeat=3)}
        if not all(np.isfinite(list(v.values()))): ok = False; break
        T, mN, sh = decomp(v); a += np.array([T, mN, sh["N"], sh["O"]]) / 2
    if ok: rows.append(a)
    if tid in J:
        b2 = np.zeros(2)
        for j in (2, 3):
            v = {k: J[tid][f"j{j}_O{k[0]}N{k[1]}F{k[2]}"][j] for k in itertools.product((0, 1), repeat=3)}
            T, mN, sh = decomp(v); b2 += np.array([T, sh["N"]]) / 2
        jrows.append(b2)
A = np.array(rows); n = len(A); rng = np.random.default_rng(20261113)
def boot(x):
    return np.concatenate([(rng.multinomial(n, np.full(n, 1 / n), size=1000) @ x) / n for _ in range(20)])
tests = {"S7-1": A[:, 1] - 0.2 * A[:, 0], "S7-2": A[:, 2] - 0.5 * A[:, 0], "S7-3": A[:, 3]}
res = {}
print(f"n = {n} docs; T = {A[:,0].mean():+.3f}; Shapley N share {A[:,2].sum()/A[:,0].sum():.3f}, O share {A[:,3].sum()/A[:,0].sum():.3f}")
for k, x in tests.items():
    b = boot(x); res[k] = float((b <= 0).mean())
    print(f"{k}: {x.mean():+.3f} [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}] p={res[k]:.3g}")
Jr = np.array(jrows); jn = Jr[:, 1].sum() / Jr[:, 0].sum() if len(Jr) else float("nan")
print(f"co-requirement (S7-2): Mistral Shapley N share = {jn:.3f} (n = {len(Jr)}) → {'met' if jn >= 0.5 else 'NOT met'}")
order = sorted(res, key=res.get); alive = True
for i, k in enumerate(order):
    ok = alive and res[k] <= 0.05 / (3 - i); alive = alive and ok
    verdict = "SUPPORTED" if ok else "not supported"
    if k == "S7-2" and ok and not jn >= 0.5: verdict = "JUDGE-DEPENDENT"
    print(f"  {k}: {verdict} (Holm)")
