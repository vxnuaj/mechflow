"""E1 analysis (exploratory): where (layers) and when (denoising steps) is neighbour feedback read?
Effect of a cut = mean over (doc, j in {2,3}) of [measure(base) − measure(cut)]; share = effect / effect(all).
Cluster bootstrap by document (4,000 resamples)."""
import glob, json, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "src/results/e1_map"
R = [json.loads(l)["result"] for f in glob.glob(path + "/shard_*.jsonl") for l in open(f)]
conds = sorted({k.split("|", 1)[1] for k in R[0] if k.startswith("j2|")})
rng = np.random.default_rng(0)
for meas in ("dep", "coh"):
    eff = {c: np.array([np.mean([r[f"j{j}|base"][meas] - r[f"j{j}|{c}"][meas] for j in (2, 3)]) for r in R]) for c in conds}
    n = len(R); idx = rng.integers(0, n, (4000, n))
    tot = eff["all"][idx].mean(1)
    print(f"\n== {meas} (n = {n} docs): effect of cutting neighbour-feedback reads (base − cut), share of the full cut ==")
    order = ["all"] + [c for c in conds if c.startswith("L") and "-" in c] + ["L0", "L1"] + \
            [c for c in conds if c.startswith("s") and "|" not in c] + sorted(c for c in conds if "|" in c)
    for c in order:
        e = eff[c]; b = e[idx].mean(1); sh = b / tot
        print(f"  {c:16s} {e.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]   share {e.mean()/eff['all'].mean():5.2f} [{np.percentile(sh,2.5):.2f},{np.percentile(sh,97.5):.2f}]")
