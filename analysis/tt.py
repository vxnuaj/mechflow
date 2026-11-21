"""Prereg §2.18 (family TT: text-type generality, math = GSM8K and code = MBPP; written before the confirmation data).
Usage: python analysis/tt.py RESULTS_DIR   (holds tt_{math,code}_confirm and tt_{math,code}_confirm_fx; conf_h2n schema)
Per type (500 fresh docs, 0.5B, MR), on coh (dep is too noisy at this n: exploration T for dep was not significant):
  TT-1  actual-context coh: contrast N − 0.2T > 0 (as H2-N / S7). Cluster bootstrap over docs (confirm.summarize).
  TT-2  fixed-context coh (target-mediated, as R-FC): the neighbour routes carry ≥ 80%: (N + F) − 0.8·T > 0; and O ≤ 0.2·T
        (signed point estimate; own feedback contributes at most 20%, and a negative O (own feedback hurting) is consistent with "not selves").
Holm over the 2 types × 2 predictions at α = 0.05 (one-sided p from the bootstrap). A type is SUPPORTED iff both hold.
Gates: identity on/off true on ≥ 99% of docs; T(coh) lower bound > 0 (else NOT ESTIMABLE for that type)."""
import itertools, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirm as C
C.ROOT = sys.argv[1]; rng = np.random.default_rng(20261121); K = list(itertools.product((0, 1), repeat=3))


def units(name, metric):
    R = C.load(name); rows = []; ident = [0, 0]
    for tid, r in R.items():
        ident[0] += bool(r["identity_on"]); ident[1] += bool(r["identity_off"]); acc = np.zeros(5); k = 0
        for j in (2, 3):
            v = {x: r["cells"][f"j{j}_O{x[0]}N{x[1]}F{x[2]}"][metric] for x in K}
            if not all(map(math.isfinite, v.values())): break
            b = v[(0, 0, 0)]; val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
            sh = {p: sum(math.factorial(q) * math.factorial(2 - q) / 6 * (val(set(S) | {p}) - val(set(S))) for q in range(3)
                         for S in itertools.combinations([o for o in "ONF" if o != p], q)) for p in "ONF"}
            T = v[(1, 1, 1)] - b; acc += [T, sh["O"], sh["N"], sh["F"], (v[(0, 1, 0)] - b) - 0.2 * T]; k += 1
        else:
            rows.append(np.append(acc, k))
    A = np.array(rows); return A, len(R), ident


ps = {}; out = {}
for ty in ("math", "code"):
    A, n, ident = units(f"tt_{ty}_confirm", "coh"); F_, nf, _ = units(f"tt_{ty}_confirm_fx", "coh")
    Tr = C.summarize(A[:, 0], A[:, 5], 0.0, rng, bca=False); t1 = C.summarize(A[:, 4], A[:, 5], 0.0, rng, bca=False)
    t2 = C.summarize(F_[:, 2] + F_[:, 3] - 0.8 * F_[:, 0], F_[:, 5], 0.0, rng, bca=False)
    o_share = F_[:, 1].sum() / F_[:, 0].sum()
    est = Tr["lb1s"] > 0 and min(ident) >= 0.99 * n
    print(f"[{ty}] docs {n}; identity {ident}; T(coh) {C.fmt(Tr)} → {'estimable' if est else 'NOT ESTIMABLE'}")
    print(f"[{ty}] TT-1 contrast N − 0.2T (actual ctx) {C.fmt(t1)}")
    print(f"[{ty}] TT-2 (N + F) − 0.8T (fixed ctx) {C.fmt(t2)}; |O| share {abs(o_share):.3f} (≤ 0.2 required)")
    sh = lambda X: " ".join(f"{k} {X[:, i].sum() / X[:, 0].sum():+.2f}" for k, i in (("O", 1), ("N", 2), ("F", 3)))
    print(f"[{ty}] shares actual ctx: {sh(A)} | fixed ctx: {sh(F_)}")
    out[ty] = dict(est=est, oshare=o_share <= 0.2)
    ps[f"{ty}-TT1"] = t1["p"] if est else 1.0; ps[f"{ty}-TT2"] = t2["p"] if est else 1.0
order = sorted(ps, key=ps.get); alive = True; ok = {}
for i, k in enumerate(order):
    h = alive and ps[k] <= 0.05 / (len(order) - i); alive = alive and h; ok[k] = h
    print(f"  {k}: p = {ps[k]:.4g} → {'holds' if h else 'not supported'} (Holm)")
for ty in ("math", "code"):
    dec = "NOT ESTIMABLE" if not out[ty]["est"] else "SUPPORTED" if (ok[f"{ty}-TT1"] and ok[f"{ty}-TT2"] and out[ty]["oshare"]) else "NOT SUPPORTED"
    print(f"TT DECISION [{ty}]: {dec}")
