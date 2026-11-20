"""Prereg §2.12 (toy access × form factorial; written before its data).
Usage: python analysis/toy_access.py <dir with {access}_{form}_s{seed}_decomp.json>

Unit: sample (prompt × draw); per sample, outcome = mean over targets j of coh_j. Per (access, form, t_end, seed):
the 8 cell means → absolute Shapley values O, N, F and the total T = cell111 − cell000.
Pooled over seeds: bootstrap over samples within each seed (10,000 resamples, seed 20261119), averaging the seeds.

Conditions: C = (access, t_end) ∈ {read, zero} × {1.0, 0.5}; form ∈ {exp, cont}.
Feasibility: a (C, form) cell is IDENTIFIABLE iff the pooled T 2.5th percentile > 0 and T > 0 in all 3 seeds.
Contrast D = N − F (absolute), pooled with a 95% CI; per-seed signs.

Predictions of the account (H-clean + access), for identifiable cells:
  P1  access=read, both t_end:          D > 0 (N dominates)
  P2  access=zero, t_end=0.5 (noisy):   D < 0 (F dominates: commitment at the SC-free decode)
  P3  access=zero, t_end=1.0 (clean):   D > 0 (commitment during the flow, where SC is readable)
  P4  form irrelevant:                  within each C, the two forms give the same sign of D
  P5  |O| ≤ 0.15·T in every identifiable cell (pooled point estimate)
A prediction "holds" in a cell when the pooled 95% CI excludes 0 in the predicted direction AND all 3 seeds agree
in sign. It "fails" when the pooled CI excludes 0 in the opposite direction. Otherwise it is undetermined.

Decision:
  SUPPORTED: P1–P3 hold in every identifiable cell (at least one identifiable cell for each of P1, P2, P3),
             P4 holds (no C where the forms' pooled CIs exclude 0 in opposite directions), and P5 holds.
  REFUTED:   any of P1–P3 fails in an identifiable cell, OR P4 fails.
  Otherwise INCONCLUSIVE, and NOT IDENTIFIABLE if no cell is identifiable for P2 (the critical prediction).
Rival readings: "readout access alone decides" ⇔ P3 fails; "form decides" ⇔ P4 fails.
"""
import glob, itertools, json, math, os, sys
import numpy as np

KEYS = list(itertools.product((0, 1), repeat=3))


def shapley(c):
    b = c[(0, 0, 0)]; val = lambda S: c[tuple(int(p in S) for p in "ONF")] - b
    return {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S)))
                   for k in range(3) for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}


def per_sample(cells):
    """cells: {'O0N0F0': [[coh per sample] per j], ...} → array [n, 8] of means over j."""
    return np.stack([np.mean(np.array(cells[f"O{o}N{n}F{f}"]), 0) for o, n, f in KEYS], 1)


def stats(X):
    c = {k: X[:, i].mean() for i, k in enumerate(KEYS)}; sh = shapley(c)
    return {"T": c[(1, 1, 1)] - c[(0, 0, 0)], **sh}


def main(d):
    runs = {}
    for f in sorted(glob.glob(f"{d}/*_decomp.json")):
        r = json.load(open(f)); m = r["meta"]
        for te in ("1.0", "0.5"):
            runs.setdefault((m["access"], m["form"], te), {})[m["seed"]] = per_sample(r[te]["cells"])
    rng = np.random.default_rng(20261119); R = 10000; out = {}
    print(f"{'access':6} {'form':5} {'t_end':5} | {'T':>7} {'O':>7} {'N':>7} {'F':>7} | D=N-F [95% CI]  seeds(sign D)  ident")
    for key in sorted(runs):
        seeds = runs[key]; pts = {s: stats(X) for s, X in seeds.items()}
        boots = {"T": np.zeros(R), "D": np.zeros(R), "O": np.zeros(R)}
        for s, X in seeds.items():
            idx = rng.integers(0, len(X), (R, len(X)))
            M = X[idx].mean(1)                                   # [R, 8]
            c = {k: M[:, i] for i, k in enumerate(KEYS)}; sh = shapley(c)
            boots["T"] += (c[(1, 1, 1)] - c[(0, 0, 0)]) / len(seeds); boots["D"] += (sh["N"] - sh["F"]) / len(seeds); boots["O"] += sh["O"] / len(seeds)
        P = {k: np.mean([pts[s][k] for s in pts]) for k in ("T", "O", "N", "F")}; D = P["N"] - P["F"]
        lo, hi = np.percentile(boots["D"], [2.5, 97.5]); tlo = np.percentile(boots["T"], 2.5)
        signs = [int(np.sign(pts[s]["N"] - pts[s]["F"])) for s in sorted(pts)]
        ident = tlo > 0 and all(pts[s]["T"] > 0 for s in pts) and len(pts) == 3
        out[key] = dict(P, D=D, lo=lo, hi=hi, signs=signs, ident=ident)
        print(f"{key[0]:6} {key[1]:5} {key[2]:5} | {P['T']:+.3f} {P['O']:+.3f} {P['N']:+.3f} {P['F']:+.3f} | {D:+.3f} [{lo:+.3f},{hi:+.3f}]  {signs}  {ident}")

    def verdict(key, want):           # want: +1 (D > 0) or −1
        r = out.get(key)
        if r is None or not r["ident"]: return "n/a"
        if want > 0 and r["lo"] > 0 and all(s > 0 for s in r["signs"]): return "holds"
        if want < 0 and r["hi"] < 0 and all(s < 0 for s in r["signs"]): return "holds"
        if (want > 0 and r["hi"] < 0) or (want < 0 and r["lo"] > 0): return "FAILS"
        return "undetermined"
    preds = {}
    for form in ("exp", "cont"):
        preds[f"P1 read/1.0/{form}"] = verdict(("read", form, "1.0"), +1)
        preds[f"P1 read/0.5/{form}"] = verdict(("read", form, "0.5"), +1)
        preds[f"P2 zero/0.5/{form}"] = verdict(("zero", form, "0.5"), -1)
        preds[f"P3 zero/1.0/{form}"] = verdict(("zero", form, "1.0"), +1)
    p4_fail = []
    for acc, te in itertools.product(("read", "zero"), ("1.0", "0.5")):
        a, b = out.get((acc, "exp", te)), out.get((acc, "cont", te))
        if a and b and a["ident"] and b["ident"] and ((a["lo"] > 0 and b["hi"] < 0) or (a["hi"] < 0 and b["lo"] > 0)): p4_fail.append(f"{acc}/{te}")
    p5_fail = [f"{k}" for k, r in out.items() if r["ident"] and abs(r["O"]) > 0.15 * r["T"]]
    for k, v in preds.items(): print(f"  {k}: {v}")
    print(f"  P4 (form irrelevant): {'FAILS at ' + ', '.join(p4_fail) if p4_fail else 'holds'}")
    print(f"  P5 (|O| ≤ 0.15 T): {'fails at ' + ', '.join(p5_fail) if p5_fail else 'holds'}")
    fam = lambda tag: [v for k, v in preds.items() if k.startswith(tag)]
    if all(v == "n/a" for v in fam("P2")): dec = "NOT IDENTIFIABLE (no identifiable cell for P2)"
    elif any(v == "FAILS" for v in preds.values()) or p4_fail: dec = "REFUTED"
    elif all(any(v == "holds" for v in fam(t)) and all(v in ("holds", "n/a") for v in fam(t)) for t in ("P1", "P2", "P3")) and not p5_fail: dec = "SUPPORTED"
    else: dec = "INCONCLUSIVE"
    rivals = []
    if not p4_fail and any(v == "FAILS" for k, v in preds.items() if k.startswith("P3")): rivals.append("readout-access-alone rival favoured (P3 failed)")
    if p4_fail: rivals.append("form rival favoured (P4 failed)")
    print("TOY ACCESS DECISION:", dec, ("; " + "; ".join(rivals)) if rivals else "")


if __name__ == "__main__":
    main(sys.argv[1])
