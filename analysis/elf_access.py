"""Prereg §2.13 (family E4: ELF decode-access test; written before its data).
Usage: python analysis/elf_access.py <root>
  <root>/<arm>_s<seed>/<n_steps>_<access>/cells_judged_fixedctx.json   ({"64": {cell: [span log-prob or None]}})
  <root>/<arm>_s<seed>/<n_steps>_<access>/meta.json                    (dis, identity_min, ...)
  arm ∈ {readable, zeroed}; seed ∈ {0, 1, 2}; readable arm: n_steps {32, 24} × access {read, zero}; zeroed arm: {32, 24} × zero.

Outcome: the Mistral span log-prob of target token j = 64 against the fixed O1N1F1 left context of the same config
(target-mediated, as in E3/E3b). Per sample (complete cases): 8 cell values → absolute Shapley O, N, F; T = c111 − c000.
Δ_n = N(readable arm, access read, n) − N(readable arm, access zero, n), paired by sample (the same noise and model).

Predictions of the account (H-clean + access):
  M1 (mandatory)  Δ_24 > 0: in the truncated regime (states uninformative, feedback informative) making the decode-pass
                  feedback readable increases direct neighbour reading. Holds iff the CI lower bound > 0. FAILS iff the CI
                  upper bound < SESOI = 0.1·|T(readable, zero, 24)| (a clear null or a reversal).
  M2 (mandatory)  I = Δ_24 − Δ_32 > 0: the increase is larger when the states are uninformative. At full length
                  x̂ ≈ final latent (calibration: decode(state) = decode(feedback) for 100%), so readable feedback is
                  redundant there. M2 holds iff the CI lower bound of I > 0 AND the CI upper bound of Δ_32 < 0.5·Δ_24 (point).
                  M2 FAILS iff the CI upper bound of I < 0.5·Δ_24 (the rival's pattern: Δ_32 is at least about half of
                  Δ_24), or the CI lower bound of Δ_32 > 0.5·Δ_24 (a clear full-length increase, contradicting
                  redundancy), or the CI of I lies below 0.
  S1 (supportive) N(readable, read, 24) − N(zeroed arm, zero, 24) > 0 (between matched fine-tunes).
Rival "readout access alone decides": Δ_24 ≈ Δ_32 > 0, so M2 fails or is undetermined while Δ_32 > 0.

Inference: a two-level bootstrap (20,000; seed 20261120). Resample fine-tune seeds, then samples within each seed; each
quantity is the mean over the resampled seeds. Holds iff the 95% CI excludes 0 in the predicted direction; FAILS iff the
CI excludes 0 in the opposite direction; otherwise undetermined.

Validity / manipulation checks (all must pass, else NOT TESTABLE):
  V1 identities: identity_min ≥ 0.99 in every config;
  V2 regime: in the readable arm, dis(24) ≥ 0.2 and dis(32) ≤ 0.05 (seed means);
  V3 identifiability: T(readable, read, 24) and T(readable, zero, 24) have CI lower bounds > 0;
  V4 fluency: mean c111 at (readable, read, 32) ≥ mean c111 at (zeroed, zero, 32) − 0.2·|mean c111 at (zeroed, zero, 32)|
     (the readable fine-tune has not degraded the model by more than 20%).
Decision:
  NOT TESTABLE  if any of V1–V4 fails;
  REFUTED       else if M1 or M2 FAILS;
  SUPPORTED     else if M1 and M2 both hold;
  INCONCLUSIVE  otherwise.
S1 and the full tables (T, O, N, F per config; Δ_32) are reported.
"""
import glob, itertools, json, math, os, sys
import numpy as np

KEYS = list(itertools.product((0, 1), repeat=3)); R = 20000


def shap_rows(J):
    """J: {cell: [v]} → array [n_complete, 5] of (T, O, N, F, c111) plus sample indices."""
    n = len(J["O0N0F0"]); rows, idx = [], []
    for i in range(n):
        v = {k: J[f"O{k[0]}N{k[1]}F{k[2]}"][i] for k in KEYS}
        if any(x is None for x in v.values()): continue
        b = v[(0, 0, 0)]; val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
        sh = {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S)))
                     for k in range(3) for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}
        rows.append([v[(1, 1, 1)] - b, sh["O"], sh["N"], sh["F"], v[(1, 1, 1)]]); idx.append(i)
    return np.array(rows), np.array(idx)


def load(root):
    D = {}
    for d in sorted(glob.glob(f"{root}/*_s*/*_*")):
        arm_seed, cfg = d.split("/")[-2], d.split("/")[-1]; arm, seed = arm_seed.rsplit("_s", 1); n, acc = cfg.split("_")
        J = json.load(open(f"{d}/cells_judged_fixedctx.json"))["64"]; meta = json.load(open(f"{d}/meta.json"))
        X, idx = shap_rows(J); D[(arm, int(seed), int(n), acc)] = {"X": X, "idx": idx, "meta": meta}
    return D


def two_level(fn, seeds, rng):
    """fn(seed, sample_index_array) → scalar for one seed. Returns the point estimate and 95% CI."""
    pt = np.mean([fn(s, None) for s in seeds]); bs = np.empty(R)
    for r in range(R):
        ss = rng.choice(seeds, len(seeds)); bs[r] = np.mean([fn(s, rng) for s in ss])
    return pt, np.percentile(bs, [2.5, 97.5])


def main(root):
    D = load(root); rng = np.random.default_rng(20261120)
    seeds = sorted({k[1] for k in D if k[0] == "readable"}); zseeds = sorted({k[1] for k in D if k[0] == "zeroed"})
    print("config                         n   T        O        N        F       c111   dis   ident")
    for k in sorted(D):
        X = D[k]["X"]; m = D[k]["meta"]
        print(f"{k[0]:8} s{k[1]} n{k[2]:2d} {k[3]:4}   {len(X):4d} {X[:,0].mean():+.3f}  {X[:,1].mean():+.3f}  {X[:,2].mean():+.3f}  {X[:,3].mean():+.3f}  {X[:,4].mean():+.3f}  {m['dis']:.3f} {m['identity_min']:.4f}")

    def paired(n):
        def f(s, r):
            A, B = D[("readable", s, n, "read")], D[("readable", s, n, "zero")]
            common = np.intersect1d(A["idx"], B["idx"]); ia = np.searchsorted(A["idx"], common); ib = np.searchsorted(B["idx"], common)
            d = A["X"][ia, 2] - B["X"][ib, 2]
            if r is not None: d = d[r.integers(0, len(d), len(d))]
            return d.mean()
        return f
    def diff_of(fa, fb): return lambda s, r: fa(s, r) - fb(s, r)
    def col(key_fn, c):
        def f(s, r):
            X = D[key_fn(s)]["X"][:, c]
            if r is not None: X = X[r.integers(0, len(X), len(X))]
            return X.mean()
        return f
    d24 = two_level(paired(24), seeds, rng); d32 = two_level(paired(32), seeds, rng)
    I = two_level(diff_of(paired(24), paired(32)), seeds, rng)
    T24r = two_level(col(lambda s: ("readable", s, 24, "read"), 0), seeds, rng)
    T24z = two_level(col(lambda s: ("readable", s, 24, "zero"), 0), seeds, rng)
    n_read24 = np.mean([D[("readable", s, 24, "read")]["X"][:, 2].mean() for s in seeds])
    n_zarm24 = np.mean([D[("zeroed", s, 24, "zero")]["X"][:, 2].mean() for s in zseeds]) if zseeds else float("nan")
    print(f"Δ_24 = {d24[0]:+.3f} [{d24[1][0]:+.3f}, {d24[1][1]:+.3f}]   Δ_32 = {d32[0]:+.3f} [{d32[1][0]:+.3f}, {d32[1][1]:+.3f}]   I = {I[0]:+.3f} [{I[1][0]:+.3f}, {I[1][1]:+.3f}]")
    print(f"S1 (descriptive): N(readable, read, 24) {n_read24:+.3f} vs N(zeroed arm, zero, 24) {n_zarm24:+.3f}")
    v1 = all(v["meta"]["identity_min"] >= 0.99 for v in D.values())
    dis24 = np.mean([D[("readable", s, 24, "read")]["meta"]["dis"] for s in seeds]); dis32 = np.mean([D[("readable", s, 32, "read")]["meta"]["dis"] for s in seeds])
    v2 = dis24 >= 0.2 and dis32 <= 0.05
    v3 = T24r[1][0] > 0 and T24z[1][0] > 0
    c_r = np.mean([D[("readable", s, 32, "read")]["X"][:, 4].mean() for s in seeds]); c_z = np.mean([D[("zeroed", s, 32, "zero")]["X"][:, 4].mean() for s in zseeds])
    v4 = c_r >= c_z - 0.2 * abs(c_z)
    print(f"V1 identities {v1} | V2 regime dis24 {dis24:.3f} dis32 {dis32:.3f} → {v2} | V3 T24 read {T24r[0]:+.3f} [{T24r[1][0]:+.3f}], zero {T24z[0]:+.3f} [{T24z[1][0]:+.3f}] → {v3} | V4 fluency c111 readable {c_r:+.3f} vs zeroed {c_z:+.3f} → {v4}")
    st = lambda q, want: "holds" if (q[1][0] > 0 if want > 0 else q[1][1] < 0) else "FAILS" if (q[1][1] < 0 if want > 0 else q[1][0] > 0) else "undetermined"
    sesoi = 0.1 * abs(T24z[0])                                   # smallest effect of interest: 10% of the truncated SC effect
    m1 = "holds" if d24[1][0] > 0 else "FAILS" if (d24[1][1] < 0 or d24[1][1] < sesoi) else "undetermined"
    half = 0.5 * max(d24[0], 0.0)
    if I[1][0] > 0 and d32[1][1] < half: m2 = "holds"
    elif I[1][1] < 0 or (d24[0] > 0 and I[1][1] < half) or (d24[0] > 0 and d32[1][0] > half): m2 = "FAILS"
    else: m2 = "undetermined"
    print(f"SESOI for M1 = {sesoi:.3f}; M2 threshold 0.5·Δ_24 = {half:.3f}")
    print(f"M1 (Δ_24 > 0): {m1}   M2 (I > 0): {m2}")
    if not (v1 and v2 and v3 and v4): dec = "NOT TESTABLE (" + ", ".join(n for n, ok in (("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4)) if not ok) + ")"
    elif "FAILS" in (m1, m2): dec = "REFUTED"
    elif m1 == "holds" and m2 == "holds": dec = "SUPPORTED"
    else: dec = "INCONCLUSIVE"
    riv = " ; pattern matches the rival 'readout access alone' (Δ_32 > 0, I not > 0)" if (d32[1][0] > 0 and m2 != "holds") else ""
    print("E4 DECISION:", dec + riv)


if __name__ == "__main__":
    main(sys.argv[1])
