"""L2 analysis (prereg §2.4): theta = non-valid rate (hybrid + other) under H* minus the mean over the random sets,
paired per prompt; cluster bootstrap by pair (20,000). Decision: one-sided lower bound (alpha = 0.025, as logged for
L1) of theta > 0.03."""
import glob, json, sys
import numpy as np
R = [json.loads(l) for f in glob.glob((sys.argv[1] if len(sys.argv) > 1 else "src/results/l2") + "/shard_*.jsonl") for l in open(f)]
bank = {json.loads(l)["id"]: json.loads(l) for l in open("data/bank.jsonl")}
by = {}
for r in R:
    c = r["result"]["conds"]; nv = lambda n: float(c[n]["label"] not in ("mode_a", "mode_b"))
    d = nv("Hstar") - np.mean([nv(s) for s in r["result"]["random_sets"]])
    p = bank[r["task_id"]]["pair_id"]; by.setdefault(p, []).append((d, nv("Hstar"), np.mean([nv(s) for s in r["result"]["random_sets"]]), nv("base")))
P = list(by); num = np.array([sum(x[0] for x in by[p]) for p in P]); den = np.array([len(by[p]) for p in P], float)
est = num.sum() / den.sum(); rng = np.random.default_rng(20261113); n = len(P)
B = np.concatenate([((c := rng.multinomial(n, np.full(n, 1/n), size=1000).astype(float)) @ num) / (c @ den) for _ in range(20)])
lb = np.percentile(B, 2.5)
allx = [x for p in P for x in by[p]]
print(f"prompts {len(allx)}, pairs {n}: non-valid base {np.mean([x[3] for x in allx]):.3f}, H* {np.mean([x[1] for x in allx]):.3f}, random {np.mean([x[2] for x in allx]):.3f}")
print(f"theta (H* − random) = {est*100:+.2f} pp [95% {np.percentile(B,2.5)*100:+.2f}, {np.percentile(B,97.5)*100:+.2f}]; one-sided LB (α=0.025) {lb*100:+.2f} pp vs 3 pp → "
      + ("L2 SUPPORTED" if lb > 0.03 else ("L2 REJECTED" if np.percentile(B, 97.5) < 0.03 else "L2 inconclusive")))
