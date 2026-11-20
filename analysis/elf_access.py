"""Prereg §2.13 v2 (family E4: ELF decode-access test; revised after the GPT red-team, before any data).
Usage: python analysis/elf_access.py <root>
Layout:
  <root>/<arm>_s<seed>/<n_steps>_<access>/{cells_judged_fixedctx.json, meta.json}
  <root>/<arm>_s<seed>/heldout_ce.json
  arm readable: configs 32_read, 32_zero, 24_read, 24_zero, 24_shuf; arm zeroed: 32_zero, 24_zero; seeds 0–7 in each arm.
Outcome: the Mistral span log-prob of token j = 64 against ONE common reference context per sample (the base ELF-B SC-on
tokens for the same noise; red-team C4). Per sample and config: 8 cells → signed Shapley O, N, F in nats; T = c111 − c000.
COMMON SAMPLE SET (M6/M7): per seed, the samples complete in all readable-arm configs. Missingness > 5% in any seed →
NOT TESTABLE.
Per-sample contrasts (readable arm, paired within sample):
  Δ24 = N(24, read) − N(24, zero);  Δ32 = N(32, read) − N(32, zero);  Δ24s = N(24, shuf) − N(24, zero)
  Q = Δ32 − 0.5·Δ24  (M11);  R = Δ24s − 0.5·Δ24;  I = Δ24 − Δ32 (descriptive).
Inference: a two-level bootstrap (resample seeds, then samples within each seed; 20,000; seed 20261120). Each statistic is
the mean over seeds of the per-seed sample means. Per-seed signs are reported.
SESOI = 0.1·|T(readable, 24, zero)| (pooled point estimate). The account predicts a non-trivial effect, since the
feedback carries far more decodable information than the state at n = 24.
Mandatory predictions:
  M1  Δ24 ≥ SESOI: holds iff the CI lower bound > 0, point ≥ SESOI and ≥ 6/8 seeds have Δ24 > 0;
      FAILS iff the CI upper bound < SESOI (M10).
  M2  Q < 0 (at full length the readable feedback is redundant with the state, so the increase is less than half of the
      truncated one): holds iff the CI upper bound < 0 and ≥ 6/8 seeds have Q < 0; FAILS iff the CI lower bound > 0.
  M3  R < 0 (shuffled feedback, with matched marginal statistics, raises N by less than half the real feedback; M14):
      holds iff the CI upper bound < 0 and ≥ 6/8 seeds have R < 0; FAILS iff the CI lower bound > 0.
Validity gates, every seed (else NOT TESTABLE):
  V1 identities: token agreement ≥ 0.99 and decode-logit max-diff ≤ 2.0 (bf16) in every config (M3);
  V2 regime (C1/M4/M9): in the readable arm at 24_zero, lp_fb − lp_state ≥ 0.5 nats both globally and in the |i − j| ≤ 16
     window; at 32_zero, |lp_fb − lp_state| ≤ 0.1;
  V3 uptake (C5/M12): at 24_read, P(token changes when decode SC → zero) ≥ 0.10;
  V4 fluency (M8): the held-out decoder CE with zero SC: mean(readable) ≤ mean(zeroed) + 0.05 nats/token, and no
     readable seed > mean(zeroed) + 0.10;
  V5 missingness ≤ 5%.
Decision:
  NOT TESTABLE  if any gate fails (listed);
  REFUTED       else if any of M1–M3 FAILS;
  SUPPORTED     else if M1, M2 and M3 all hold;
  INCONCLUSIVE  otherwise.
Rival labels: "readout access regardless of information" if M2 fails; "any new decode input" if M3 fails.
Reported: all configs' T, O, N, F and the full 8-cell means; S1 = N(readable, 24, read) vs N(zeroed arm, 24, zero) (between
checkpoints, descriptive); held-out ce_sc and ce_shuf (descriptive uptake on the training distribution).
"""
import glob, itertools, json, math, os, re, sys
import numpy as np

KEYS = list(itertools.product((0, 1), repeat=3)); R_ = 20000
READ_CFGS = ["32_read", "32_zero", "24_read", "24_zero", "24_shuf"]; ZERO_CFGS = ["32_zero", "24_zero"]


def shap(v):
    b = v[(0, 0, 0)]; val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
    sh = {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S)))
                 for k in range(3) for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}
    return {"T": v[(1, 1, 1)] - b, **sh}


def load(root):
    D = {}
    for d in glob.glob(os.path.join(root, "*_s*")):
        m = re.fullmatch(r"(readable|zeroed)_s(\d+)", os.path.basename(d)); assert m, f"unexpected dir {d}"
        arm, seed = m.group(1), int(m.group(2))
        cfgs = READ_CFGS if arm == "readable" else ZERO_CFGS
        rec = {"ce": json.load(open(os.path.join(d, "heldout_ce.json"))), "cfg": {}}
        for c in cfgs:
            p = os.path.join(d, c); J = json.load(open(os.path.join(p, "cells_judged_fixedctx.json")))["64"]
            rec["cfg"][c] = {"J": J, "meta": json.load(open(os.path.join(p, "meta.json")))}
        D[(arm, seed)] = rec
    return D


def per_seed_tables(rec, cfgs):
    n = len(rec["cfg"][cfgs[0]]["J"]["O0N0F0"]); assert all(len(rec["cfg"][c]["J"]["O0N0F0"]) == n for c in cfgs)
    ok = [i for i in range(n) if all(all(rec["cfg"][c]["J"][f"O{k[0]}N{k[1]}F{k[2]}"][i] is not None for k in KEYS) for c in cfgs)]
    tab = {c: [shap({k: rec["cfg"][c]["J"][f"O{k[0]}N{k[1]}F{k[2]}"][i] for k in KEYS}) for i in ok] for c in cfgs}
    return tab, 1 - len(ok) / n


def boot(per_seed_vecs, rng):
    """per_seed_vecs: {seed: per-sample array}. Returns the point estimate, the 95% CI and per-seed means."""
    seeds = sorted(per_seed_vecs); means = {s: float(np.mean(per_seed_vecs[s])) for s in seeds}
    pt = float(np.mean(list(means.values()))); bs = np.empty(R_)
    for r in range(R_):
        ss = rng.choice(seeds, len(seeds)); acc = 0.0
        for s in ss:
            v = per_seed_vecs[s]; acc += v[rng.integers(0, len(v), len(v))].mean()
        bs[r] = acc / len(ss)
    return pt, np.percentile(bs, [2.5, 97.5]), means


def main(root):
    D = load(root); rng = np.random.default_rng(20261120)
    rs = sorted(s for a, s in D if a == "readable"); zs = sorted(s for a, s in D if a == "zeroed")
    assert rs == list(range(8)) and zs == list(range(8)), f"seed grid incomplete: readable {rs}, zeroed {zs}"
    T = {}; miss = {}; vecs = {k: {} for k in ("d24", "d32", "d24s", "Q", "R", "I")}
    for s in rs:
        tab, mfrac = per_seed_tables(D[("readable", s)], READ_CFGS); miss[s] = mfrac
        N = {c: np.array([r["N"] for r in tab[c]]) for c in READ_CFGS}; T[s] = np.mean([r["T"] for r in tab["24_zero"]])
        d24 = N["24_read"] - N["24_zero"]; d32 = N["32_read"] - N["32_zero"]; d24s = N["24_shuf"] - N["24_zero"]
        for k, v in (("d24", d24), ("d32", d32), ("d24s", d24s), ("Q", d32 - 0.5 * d24), ("R", d24s - 0.5 * d24), ("I", d24 - d32)): vecs[k][s] = v
        for c in READ_CFGS:
            r = tab[c]; print(f"readable s{s} {c:8} n={len(r)} T {np.mean([x['T'] for x in r]):+.3f} O {np.mean([x['O'] for x in r]):+.3f} N {np.mean([x['N'] for x in r]):+.3f} F {np.mean([x['F'] for x in r]):+.3f}")
    zN24 = []
    for s in zs:
        tab, _ = per_seed_tables(D[("zeroed", s)], ZERO_CFGS); zN24.append(np.mean([r["N"] for r in tab["24_zero"]]))
    sesoi = 0.1 * abs(float(np.mean(list(T.values()))))
    q = {k: boot(v, rng) for k, v in vecs.items()}
    for k in ("d24", "d32", "d24s", "Q", "R", "I"):
        pt, ci, means = q[k]; print(f"{k:5}: {pt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  per seed: " + " ".join(f"{means[s]:+.2f}" for s in rs))
    print(f"SESOI = {sesoi:.3f};  S1 (descriptive, between checkpoints): N(zeroed arm, 24, zero) mean {np.mean(zN24):+.3f}")
    v1 = all(m["meta"]["identity_min"] >= 0.99 and m["meta"].get("logit_maxdiff", 0) <= 2.0 for rec in D.values() for m in rec["cfg"].values())
    v2s = {}
    for s in rs:
        m24 = D[("readable", s)]["cfg"]["24_zero"]["meta"]; m32 = D[("readable", s)]["cfg"]["32_zero"]["meta"]
        v2s[s] = bool(m24["lp_fb"] - m24["lp_state"] >= 0.5 and m24["lp_fb_win"] - m24["lp_state_win"] >= 0.5 and abs(m32["lp_fb"] - m32["lp_state"]) <= 0.1)
    v2 = all(v2s.values())
    v3s = {s: bool(D[("readable", s)]["cfg"]["24_read"]["meta"].get("uptake", 0) >= 0.10) for s in rs}; v3 = all(v3s.values())
    cz = [D[("zeroed", s)]["ce"]["ce_zero"] for s in zs]; cr = [D[("readable", s)]["ce"]["ce_zero"] for s in rs]
    v4 = bool(np.mean(cr) <= np.mean(cz) + 0.05 and max(cr) <= np.mean(cz) + 0.10)
    v5 = all(m <= 0.05 for m in miss.values())
    print(f"V1 identities {v1} | V2 regime {v2} {v2s} | V3 uptake {v3} {v3s} | V4 fluency {v4} (readable {np.mean(cr):.4f}, zeroed {np.mean(cz):.4f}) | V5 missingness {v5} {miss}")
    print(f"descriptive held-out (readable): ce_zero {np.mean(cr):.4f} ce_sc {np.mean([D[('readable', s)]['ce']['ce_sc'] for s in rs]):.4f} ce_shuf {np.mean([D[('readable', s)]['ce']['ce_shuf'] for s in rs]):.4f}")
    nseed = lambda k, pred: sum(1 for s in rs if pred(q[k][2][s]))
    pt, ci, _ = q["d24"]
    m1 = "holds" if (ci[0] > 0 and pt >= sesoi and nseed("d24", lambda x: x > 0) >= 6) else "FAILS" if ci[1] < sesoi else "undetermined"
    pt, ci, _ = q["Q"]
    m2 = "holds" if (ci[1] < 0 and nseed("Q", lambda x: x < 0) >= 6) else "FAILS" if ci[0] > 0 else "undetermined"
    pt, ci, _ = q["R"]
    m3 = "holds" if (ci[1] < 0 and nseed("R", lambda x: x < 0) >= 6) else "FAILS" if ci[0] > 0 else "undetermined"
    print(f"M1 (Δ24 ≥ SESOI): {m1}   M2 (Q < 0): {m2}   M3 (R < 0, shuffled control): {m3}")
    bad = [n for n, ok in (("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4), ("V5", v5)) if not ok]
    if bad: dec = "NOT TESTABLE (" + ", ".join(bad) + ")"
    elif "FAILS" in (m1, m2, m3): dec = "REFUTED (" + ", ".join(n for n, v in (("M1", m1), ("M2", m2), ("M3", m3)) if v == "FAILS") + ")"
    elif (m1, m2, m3) == ("holds",) * 3: dec = "SUPPORTED"
    else: dec = "INCONCLUSIVE"
    riv = []
    if m2 == "FAILS": riv.append("pattern of 'readout access regardless of information'")
    if m3 == "FAILS": riv.append("pattern of 'any new decode input'")
    print("E4 DECISION:", dec, ("; " + "; ".join(riv)) if riv else "")


if __name__ == "__main__":
    main(sys.argv[1])
