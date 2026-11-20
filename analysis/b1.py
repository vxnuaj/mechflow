"""Prereg §2.16 (B1 human validation; written before any ratings). Usage (on CARC, where the key lives):
  python analysis/b1.py ratings/key.json ratings/sheet_<rater1>.md [ratings/sheet_<rater2>.md ...]   (N raters)
Agreement = the fraction of non-tie human answers matching the LM preference (higher coh_2 + coh_3). Wilson 95% interval.
Corroborated iff the point is ≥ 0.70 and the Wilson LB > 0.50; undermined iff the Wilson UB < 0.50; else inconclusive.
Reported per rater and pooled, plus the SC-on preference rate, tie rate and Cohen's κ between raters."""
import json, math, re, sys
key = {k["item"]: k for k in json.load(open(sys.argv[1]))}
def answers(p):
    t = open(p).read(); out = {}
    for m in re.finditer(r"^## (\d+)\b.*?^Answer:[ \t]*([ABTabt])?", t, re.S | re.M):
        if m.group(2): out[int(m.group(1))] = m.group(2).upper()
    return out
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h
def score(ans, label):
    agree = n = on = ties = 0
    for i, a in ans.items():
        k = key[i]
        if a == "T": ties += 1; continue
        human_on = (a == "A") == k["A_is_sc_on"]; lm_on = k["lm_on"] > k["lm_off"]
        n += 1; agree += human_on == lm_on; on += human_on
    p, lo, hi = wilson(agree, n)
    v = "CORROBORATED" if (p >= 0.70 and lo > 0.50) else "UNDERMINED" if hi < 0.50 else "INCONCLUSIVE"
    print(f"{label}: answered {len(ans)}, ties {ties}; human–LM agreement {agree}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}] → {v}; human prefers SC-on {on}/{n}")
    return v
R = [answers(p) for p in sys.argv[2:]]
for p, a in zip(sys.argv[2:], R): score(a, p)
if len(R) >= 2:   # pooled over all raters (each rater-item answer counts once) + pairwise Cohen's κ
    agree = n = 0
    for a in R:
        for i, x in a.items():
            if x == "T": continue
            n += 1; agree += ((x == "A") == key[i]["A_is_sc_on"]) == (key[i]["lm_on"] > key[i]["lm_off"])
    p, lo, hi = wilson(agree, n)
    print(f"pooled ({len(R)} raters): {agree}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}] → " + ("CORROBORATED" if (p >= 0.70 and lo > 0.50) else "UNDERMINED" if hi < 0.50 else "INCONCLUSIVE"))
    for x in range(len(R)):
        for y in range(x + 1, len(R)):
            both = [i for i in key if i in R[x] and i in R[y]]
            if not both: continue
            po = sum(R[x][i] == R[y][i] for i in both) / len(both)
            pe = sum((sum(R[x][i] == c for i in both) / len(both)) * (sum(R[y][i] == c for i in both) / len(both)) for c in "ABT")
            print(f"κ({sys.argv[2 + x]}, {sys.argv[2 + y]}): {len(both)} items, raw {po:.3f}, κ {(po - pe) / (1 - pe) if pe < 1 else float('nan'):.3f}")
