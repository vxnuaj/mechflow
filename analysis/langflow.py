"""Prereg §2.15 (family G: LangFlow generality test; written before its data).
Usage: python analysis/langflow.py <dir with cells_judged_fixedctx.json and meta.json>
Outcome: the Mistral span log-prob of target token j = L/2 against a common reference context (the SC-on tokens for the same
noise). Per sample (complete cases): 8 cells → signed Shapley O, N, F in nats; T = c111 − c000. Bootstrap over samples
(20,000; seed 20261120).
Regime (measured before this prereg; src/results/langflow_calib.json): at 50% committed (step 48), the state → final-token
log-prob is −10.7 vs feedback −2.5 (acc 0.07 vs 0.53). Commitment happens while states are uninformative and the
feedback is readable, so the account predicts N-dominance.
Gates (else NOT TESTABLE): identity_min ≥ 0.99 (fp32); T 95% CI lower bound > 0; missingness ≤ 5%.
Mandatory predictions:
  G1  D = N − F > 0:      holds iff the CI lower bound > 0; FAILS iff the CI upper bound < 0.
  G2  N − 0.5·T > 0:      (N carries at least half the SC benefit) holds iff the CI lower bound > 0; FAILS iff the CI upper bound < 0.
Subsidiary: G3 |mean O| − 0.15·mean T < 0 (computed on bootstrap means) ("not selves"): holds iff the CI upper bound < 0; fails iff the CI lower bound > 0.
Decision: NOT TESTABLE / REFUTED (G1 or G2 FAILS) / SUPPORTED (G1 and G2 hold) / INCONCLUSIVE.
Rival "block locality (N needs small blocks)" predicts N small: G2 fails."""
import itertools, json, math, os, sys
import numpy as np
KEYS = list(itertools.product((0, 1), repeat=3))
d = sys.argv[1]; J = json.load(open(os.path.join(d, "cells_judged_fixedctx.json"))); J = J[next(iter(J))]; meta = json.load(open(os.path.join(d, "meta.json")))
n = len(J["O0N0F0"]); rows = []
for i in range(n):
    v = {k: J[f"O{k[0]}N{k[1]}F{k[2]}"][i] for k in KEYS}
    if any(x is None for x in v.values()): continue
    b = v[(0, 0, 0)]; val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
    sh = {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S))) for k in range(3)
                 for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}
    T = v[(1, 1, 1)] - b; rows.append([T, sh["O"], sh["N"], sh["F"], sh["N"] - sh["F"], sh["N"] - 0.5 * T])
A = np.array(rows); m = len(A); miss = 1 - m / n
idx = np.random.default_rng(20261120).integers(0, m, (20000, m)); B = A[idx].mean(1)
B = np.concatenate([B, (np.abs(B[:, 1]) - 0.15 * B[:, 0])[:, None]], 1)           # G3 on means: |mean O| − 0.15·mean T
A = np.concatenate([A, np.full((m, 1), np.nan)], 1)
ci = lambda c: np.percentile(B[:, c], [2.5, 97.5])
names = ["T", "O", "N", "F", "D=N−F", "N−0.5T", "|O|−0.15T"]
pt = list(A[:, :6].mean(0)) + [abs(A[:, 1].mean()) - 0.15 * A[:, 0].mean()]
for c, nm in enumerate(names): print(f"{nm:10} {pt[c]:+.3f} [{ci(c)[0]:+.3f}, {ci(c)[1]:+.3f}]")
print(f"Shapley shares: O {A[:,1].sum()/A[:,0].sum():.3f} N {A[:,2].sum()/A[:,0].sum():.3f} F {A[:,3].sum()/A[:,0].sum():.3f}; n {m}/{n}; identity {meta['identity_min']:.4f}")
gates = {"identity": meta["identity_min"] >= 0.99, "T>0": ci(0)[0] > 0, "missing<=5%": miss <= 0.05}
st = lambda c: "holds" if ci(c)[0] > 0 else "FAILS" if ci(c)[1] < 0 else "undetermined"
g1, g2 = st(4), st(5); g3 = "holds" if ci(6)[1] < 0 else "fails" if ci(6)[0] > 0 else "undetermined"
print(f"gates {gates} | G1 {g1}  G2 {g2}  G3 (subsidiary) {g3}")
if not all(gates.values()): dec = "NOT TESTABLE (" + ", ".join(k for k, ok in gates.items() if not ok) + ")"
elif "FAILS" in (g1, g2): dec = "REFUTED (" + ", ".join(k for k, v in (("G1", g1), ("G2", g2)) if v == "FAILS") + ")"
elif g1 == g2 == "holds": dec = "SUPPORTED"
else: dec = "INCONCLUSIVE"
print("G DECISION:", dec, "; pattern of the block-locality rival" if g2 == "FAILS" else "")
