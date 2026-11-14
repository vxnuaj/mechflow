"""Analyze 02_block_tf.py output: per-config accuracy vs gold and vs AR, paired bootstrap CIs.

Metrics per config (over blocks):
  tok_acc_gold   mean per-token match with gold tokens
  em_gold        whole-block exact match with gold
  tok_acc_ar     mean per-token match with the checkpoint's own AR greedy block
  em_ar          whole-block match with AR
  indep_gap      em_gold - prod_j(per-position accuracy)  (>0: correctness is positively coupled across positions)
Paired comparisons (SC - noSC) at matched settings with 95% bootstrap CIs over blocks.
"""
import json
import random
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/block_tf.jsonl"
recs = [json.loads(l) for l in open(path)]
names = list(recs[0]["out"])
B = 4


def per_block(name, ref):
    tok, em = [], []
    for r in recs:
        tgt = r["target"] if ref == "gold" else r["out"]["AR"]
        out = r["out"][name]
        m = [int(j < len(out) and j < len(tgt) and out[j] == tgt[j]) for j in range(B)]
        tok.append(m)
        em.append(int(all(m)))
    return np.array(tok), np.array(em)


def boot(diff, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    s = diff[idx].mean(1)
    return diff.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


print(f"n_blocks={len(recs)}")
print(f"{'config':24s} {'tok_gold':>8s} {'em_gold':>8s} {'tok_AR':>8s} {'em_AR':>8s} {'indep_gap':>9s}  per-pos acc(gold)")
cache = {}
for name in names:
    tg, eg = per_block(name, "gold")
    ta, ea = per_block(name, "ar")
    cache[name] = (tg, eg, ta, ea)
    pp = tg.mean(0)
    print(f"{name:24s} {tg.mean():8.3f} {eg.mean():8.3f} {ta.mean():8.3f} {ea.mean():8.3f} {eg.mean() - pp.prod():9.3f}  "
          + " ".join(f"{x:.2f}" for x in pp))

print("\nPaired SC - noSC (95% bootstrap CI over blocks):")
for base in sorted({n.rsplit('_', 1)[0] for n in names if n.endswith('_SC') or n.endswith('_noSC')}):
    a, b = f"{base}_SC", f"{base}_noSC"
    if a not in cache or b not in cache:
        a, b = (f"paper_{base}_SC", f"{base}_noSC") if f"paper_{base}_SC" in cache else (a, b)
    if a not in cache or b not in cache:
        continue
    for lab, k in [("tok_gold", 0), ("em_gold", 1), ("tok_AR", 2), ("em_AR", 3)]:
        x, y = cache[a][k], cache[b][k]
        d = (x.mean(1) - y.mean(1)) if x.ndim == 2 else (x - y)
        m, lo, hi = boot(d.astype(float))
        print(f"  {base:18s} {lab:9s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
