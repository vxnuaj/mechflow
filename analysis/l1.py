"""L1 analysis (prereg §2.4): theta = (reduction_H* − mean reduction over random size/layer-matched sets) / N,
reduction = dep_{2,3}(base) − dep_{2,3}(cut), N = dep_{2,3}(base) − dep_{2,3}(all). Ratio of sums; cluster bootstrap by
document (20,000). Decision: one-sided 95% (here α = 0.025, Holm over {L1, L2} with L2 not yet run) lower bound > 0.30."""
import glob, json, sys
import numpy as np

R = [json.loads(l)["result"] for f in glob.glob((sys.argv[1] if len(sys.argv) > 1 else "src/results/l1") + "/shard_*.jsonl") for l in open(f)]
def red(r, c, m="dep"): return np.mean([r[f"j{j}|base"][m] - r[f"j{j}|{c}"][m] for j in (2, 3)])
for m in ("dep", "coh"):
    Nn = np.array([red(r, "all", m) for r in R]); Hs = np.array([red(r, "Hstar", m) for r in R])
    Rn = np.array([np.mean([red(r, s, m) for s in r["random_sets"]]) for r in R])
    num = Hs - Rn
    est = num.sum() / Nn.sum()
    rng = np.random.default_rng(20261113); n = len(R); B = []
    for _ in range(20):
        c = rng.multinomial(n, np.full(n, 1 / n), size=1000).astype(float)
        B.append((c @ num) / (c @ Nn))
    B = np.concatenate(B)
    print(f"[{m}] n = {n}: share H* = {Hs.sum()/Nn.sum():.3f}, share random = {Rn.sum()/Nn.sum():.3f}, "
          f"theta = {est:.3f} [95% {np.percentile(B,2.5):.3f}, {np.percentile(B,97.5):.3f}], one-sided LB (α=0.025) {np.percentile(B,2.5):.3f}"
          + ("" if m != "dep" else f"  → {'L1 SUPPORTED' if np.percentile(B,2.5) > 0.30 else ('L1 REJECTED' if np.percentile(B,97.5) < 0.30 else 'L1 inconclusive')} (bar 0.30)"))
