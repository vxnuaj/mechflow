"""Prereg §2.12 v2 (toy access × estimator-recipe factorial; revised after the GPT red-team, before any decomposition data).
Usage: python analysis/toy_access.py <dir with {access}_{form}_s{seed}_decomp.json>

PRIMARY OUTCOME (red-team #1): target-mediated coherence. For target j in sample b,
  coh_fixed_j = mean over neighbours i ≠ j of 1[mode(y_j) = ref_mode_i ≠ −1],
with ref_mode_i the neighbour's mode in the O1N1F1 (= SC-on) run with the same noise. The neighbours' actual tokens
never enter it, so F can only score by changing y_j. Per sample: mean over j = 0..3.
SECONDARY:
  (a) joint coherence (against the neighbours' actual tokens in each cell);
  (b) excess over independence: the prompt-level coh_fixed minus Σ_m P(y_j in m)·P(ref_i in m), both estimated over the
      prompt's draws (red-team #7);
  (c) diversity: the mean over prompts of |P(mode a) − 0.5| at position 0 in the reference run (0 = balanced, 0.5 = collapse).

Per seed: the 8 cell means → absolute Shapley O, N, F; T = c111 − c000; D = N − F. Full 8-cell surfaces are printed.
INFERENCE (red-team #9): the seed is the unit (10 seeds, each with its own task instance). Seed-level percentile
bootstrap (20,000, seed 20261119) for every condition-level quantity; recipes are paired by seed index for contrasts.

CONDITIONS: (train access, inference access, t_end, recipe). Read-trained models are sampled with decode SC read and
zeroed (zeroed is in-distribution: SC is dropped 50% in decode training). Zero-trained models are sampled zeroed only.
IDENTIFIABLE: seed-level T 2.5th percentile > 0.
MANIPULATION CHECK M (red-team #2), per (train, inference = zero, recipe): Δcad = commit_at_decode(t_end 0.5) −
commit_at_decode(t_end 1.0), where commit_at_decode = the fraction of SC-on positions whose decoded token differs from
the token implied by the last flow estimate. M holds iff the seed-level 2.5th pct of Δcad > 0 and mean Δcad ≥ 0.05.
If M fails, P2/P3 in that family are NOT TESTABLE (reported, not counted as failures).

MANDATORY PREDICTIONS (12 cells; each holds iff the seed-level 95% CI of D excludes 0 in the predicted direction):
  P1   (read, read, 1.0 and 0.5, both recipes)  D > 0   [4]
  P2   (zero, zero, 0.5, both recipes)          D < 0   [2]
  P2w  (read, zero, 0.5, both recipes)          D < 0   [2]  within-model access manipulation
  P3   (zero, zero, 1.0, both recipes)          D > 0   [2]
  P3w  (read, zero, 1.0, both recipes)          D > 0   [2]
A cell FAILS iff it is identifiable (and, for P2/P3, testable) and the CI excludes 0 in the opposite direction.
P4 (recipe does not flip the route), per (train, inference, t_end): FAILS iff the two recipes' D CIs exclude 0 in opposite
directions. Δ = D_exp − D_cont (paired by seed) is reported with its CI. The claim is narrowed to "no sign flip"
(red-team #5, #11).
P5 (subsidiary "own ≈ 0"): per identifiable cell, the seed-level CI of |O| − 0.15·T: holds if the upper bound < 0, fails if
the lower bound > 0.

DECISION (red-team #3: every mandatory cell is required; a refutation is never hidden):
  REFUTED            if any mandatory cell FAILS or P4 FAILS (listed);
  SUPPORTED          else if all 12 mandatory cells hold;
  NOT IDENTIFIABLE   else if every P2 and P2w cell is non-identifiable or not testable;
  INCONCLUSIVE       otherwise (missing / undetermined cells listed).
P5 is reported separately and bears only on the "own ≈ 0" sub-claim.
Rival readings:
- "inference readout access alone" predicts P3/P3w fail;
- "training adaptation" (the route set by how the decoder was trained) predicts P2w/P3w follow train access, i.e. P2w fails;
- "recipe decides" predicts P4 fails.
"""
import glob, itertools, json, math, sys
import numpy as np

KEYS = list(itertools.product((0, 1), repeat=3)); L = 4; R = 20000


def shapley(c):
    b = c[(0, 0, 0)]; val = lambda S: c[tuple(int(p in S) for p in "ONF")] - b
    return {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S)))
                   for k in range(3) for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}


def cell_outcomes(cfg):
    ref = np.array(cfg["ref_modes"])                              # [B, L]
    prim, joint = {}, {}
    for o, n, f in KEYS:
        key = f"O{o}N{n}F{f}"; jm = np.array(cfg["cells"][key])  # [L(j), B]
        nb = np.array(cfg["nb_actual"][key])                     # [L(j), B, L]
        pj, jj = [], []
        for j in range(L):
            others = [i for i in range(L) if i != j]
            pj.append(np.mean([(jm[j] == ref[:, i]) & (jm[j] >= 0) for i in others], 0))
            jj.append(np.mean([(jm[j] == nb[j][:, i]) & (jm[j] >= 0) for i in others], 0))
        prim[(o, n, f)] = np.mean(pj, 0); joint[(o, n, f)] = np.mean(jj, 0)
    return prim, joint


def excess(cfg):
    ref = np.array(cfg["ref_modes"]); pr = np.array(cfg["prompt"]); out = {}
    for o, n, f in KEYS:
        jm = np.array(cfg["cells"][f"O{o}N{n}F{f}"]); vals = []
        for p in np.unique(pr):
            sel = pr == p; acc = []
            for j in range(L):
                for i in range(L):
                    if i == j: continue
                    obs = np.mean((jm[j][sel] == ref[sel, i]) & (jm[j][sel] >= 0))
                    base = sum(np.mean(jm[j][sel] == m) * np.mean(ref[sel, i] == m) for m in (0, 1))
                    acc.append(obs - base)
            vals.append(np.mean(acc))
        out[(o, n, f)] = float(np.mean(vals))
    return out


def diversity(cfg):
    ref = np.array(cfg["ref_modes"]); pr = np.array(cfg["prompt"])
    return float(np.mean([abs(np.mean(ref[pr == p, 0] == 0) - 0.5) for p in np.unique(pr)]))


def summarize(c):
    sh = shapley(c); T = c[(1, 1, 1)] - c[(0, 0, 0)]
    return {"T": T, "O": sh["O"], "N": sh["N"], "F": sh["F"], "D": sh["N"] - sh["F"], "Oex": abs(sh["O"]) - 0.15 * T}


def seed_boot(vals, rng):
    v = np.asarray(vals, float); idx = rng.integers(0, len(v), (R, len(v)))
    return v.mean(), np.percentile(v[idx].mean(1), [2.5, 97.5])


def load(d):
    per = {}                                                     # (train, inf, t_end, recipe) -> {seed: stats}
    for f in sorted(glob.glob(f"{d}/*_decomp.json")):
        r = json.load(open(f)); m = r["meta"]
        for ck, cfg in r.items():
            if ck == "meta": continue
            t_end, ia = ck.split("|"); prim, joint = cell_outcomes(cfg)
            st = summarize({k: v.mean() for k, v in prim.items()})
            st["joint"] = summarize({k: v.mean() for k, v in joint.items()}); st["excess"] = summarize(excess(cfg))
            st["cad"] = cfg["commit_at_decode"]; st["div"] = diversity(cfg); st["cells"] = {k: float(v.mean()) for k, v in prim.items()}
            per.setdefault((m["access"], ia, t_end, m["form"]), {})[m["seed"]] = st
    return per


def main(d):
    per = load(d); rng = np.random.default_rng(20261119); res = {}
    print(f"{'train':5} {'inf':4} {'t_end':5} {'recipe':6} n | {'T':>7} {'O':>7} {'N':>7} {'F':>7} | D [95% CI]            | joint D  excess D  cad   div")
    for key in sorted(per):
        S = per[key]; seeds = sorted(S)
        q = {x: seed_boot([S[s][x] for s in seeds], rng) for x in ("T", "O", "N", "F", "D", "Oex", "cad")}
        jD = np.mean([S[s]["joint"]["D"] for s in seeds]); eD = np.mean([S[s]["excess"]["D"] for s in seeds]); dv = np.mean([S[s]["div"] for s in seeds])
        res[key] = {"q": q, "ident": q["T"][1][0] > 0, "seeds": seeds, "S": S}
        print(f"{key[0]:5} {key[1]:4} {key[2]:5} {key[3]:6} {len(seeds):2d}| {q['T'][0]:+.3f} {q['O'][0]:+.3f} {q['N'][0]:+.3f} {q['F'][0]:+.3f} | "
              f"{q['D'][0]:+.3f} [{q['D'][1][0]:+.3f},{q['D'][1][1]:+.3f}] | {jD:+.3f}  {eD:+.3f}  {q['cad'][0]:.3f} {dv:.3f}")
        cm = {k: np.mean([S[s]["cells"][k] for s in seeds]) for k in KEYS}
        print("      8 cells (O,N,F): " + "  ".join(f"{''.join(map(str, k))}={cm[k]:.3f}" for k in KEYS))
    M = {}
    for tr, rc in itertools.product(("read", "zero"), ("exp", "cont")):
        a, b = per.get((tr, "zero", "0.5", rc)), per.get((tr, "zero", "1.0", rc))
        if not a or not b: M[(tr, rc)] = False; continue
        sd = sorted(set(a) & set(b)); mean, ci = seed_boot([a[s]["cad"] - b[s]["cad"] for s in sd], rng)
        M[(tr, rc)] = ci[0] > 0 and mean >= 0.05
        print(f"  M ({tr}-trained, inference zero, {rc}): Δcad {mean:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] → {'holds' if M[(tr, rc)] else 'FAILS (P2/P3 not testable)'}")

    def cell(key, want, needs_M):
        r = res.get(key)
        if r is None or not r["ident"]: return "n/a (not identifiable)"
        if needs_M and not M.get((key[0], key[3]), False): return "n/a (not testable: M failed)"
        lo, hi = r["q"]["D"][1]
        if (want > 0 and lo > 0) or (want < 0 and hi < 0): return "holds"
        if (want > 0 and hi < 0) or (want < 0 and lo > 0): return "FAILS"
        return "undetermined"
    mand = {}
    for rc in ("exp", "cont"):
        for te in ("1.0", "0.5"): mand[f"P1  read/read/{te}/{rc}"] = cell(("read", "read", te, rc), +1, False)
        mand[f"P2  zero/zero/0.5/{rc}"] = cell(("zero", "zero", "0.5", rc), -1, True)
        mand[f"P2w read/zero/0.5/{rc}"] = cell(("read", "zero", "0.5", rc), -1, True)
        mand[f"P3  zero/zero/1.0/{rc}"] = cell(("zero", "zero", "1.0", rc), +1, True)
        mand[f"P3w read/zero/1.0/{rc}"] = cell(("read", "zero", "1.0", rc), +1, True)
    for k, v in mand.items(): print(f"  {k}: {v}")
    p4 = []
    for tr, ia, te in sorted({(k[0], k[1], k[2]) for k in res}):
        a, b = res.get((tr, ia, te, "exp")), res.get((tr, ia, te, "cont"))
        if not a or not b: continue
        sd = sorted(set(a["S"]) & set(b["S"])); m_, ci = seed_boot([a["S"][s]["D"] - b["S"][s]["D"] for s in sd], rng)
        flip = a["ident"] and b["ident"] and ((a["q"]["D"][1][0] > 0 and b["q"]["D"][1][1] < 0) or (a["q"]["D"][1][1] < 0 and b["q"]["D"][1][0] > 0))
        if flip: p4.append(f"{tr}/{ia}/{te}")
        print(f"  P4 {tr}/{ia}/{te}: Δ(D_exp − D_cont) {m_:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {'SIGN FLIP' if flip else 'no flip'}")
    p5 = {k: ("holds" if r["q"]["Oex"][1][1] < 0 else "fails" if r["q"]["Oex"][1][0] > 0 else "undetermined") for k, r in res.items() if r["ident"]}
    p5s = "fails" if any(v == "fails" for v in p5.values()) else "holds" if p5 and all(v == "holds" for v in p5.values()) else "undetermined"
    fails = [k for k, v in mand.items() if v == "FAILS"]
    if fails or p4: dec = "REFUTED (" + ", ".join(fails + [f"P4 {x}" for x in p4]) + ")"
    elif all(v == "holds" for v in mand.values()): dec = "SUPPORTED"
    elif all(v.startswith("n/a") for k, v in mand.items() if k.startswith("P2")): dec = "NOT IDENTIFIABLE"
    else: dec = "INCONCLUSIVE (" + ", ".join(f"{k}: {v}" for k, v in mand.items() if v != "holds") + ")"
    riv = []
    if any(mand[k] == "FAILS" for k in mand if k.startswith("P3")): riv.append("inference-readout-alone rival favoured (P3/P3w failed)")
    if any(mand[k] == "FAILS" for k in mand if k.startswith("P2w")) and not any(mand[k] == "FAILS" for k in mand if k.startswith("P2 ")): riv.append("training-adaptation rival favoured (P2w failed, P2 not)")
    if p4: riv.append("recipe rival favoured (P4 sign flip)")
    print(f"  P5 (own ≈ 0, subsidiary): {p5s}")
    print("TOY ACCESS DECISION:", dec, ("; " + "; ".join(riv)) if riv else "")


if __name__ == "__main__":
    main(sys.argv[1])
