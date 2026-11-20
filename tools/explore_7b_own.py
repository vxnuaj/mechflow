"""Phase 14.5 (EXPLORATORY, h2n_7b_300 only): where does own feedback (O) help at 7B?
Per (doc, target j ∈ {2, 3}): signed Shapley O, N, F on coh; T. The token class of the SC-on target token:
  cont  = continues a word (no leading space/newline, alphanumeric start, previous token alphanumeric-ending);
  start = begins a word (leading space or newline, or the first block position after punctuation);
  punct = punctuation/whitespace only.
Report mean O, N, T and the O share by class and by j."""
import glob, json, itertools, math, os, sys, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("lafa-hf/lafa-7B", revision="434c100d0fb2cce29ac72d76432da4b8a81eb0b7", trust_remote_code=True)
K = list(itertools.product((0, 1), repeat=3)); agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
def cls(prev, cur):
    s = tok.decode([cur]); p = tok.decode([prev]) if prev is not None else ""
    if not s.strip() or all(not ch.isalnum() for ch in s.strip()): return "punct"
    if s[0] in " \n\t": return "start"
    return "cont" if (p and p[-1].isalnum()) else "start"
for f in glob.glob("src/results/h2n_7b_300/shard_*.jsonl"):
    for l in open(f):
        c = json.loads(l)["result"]["cells"]
        for j in (2, 3):
            v = {k: c[f"j{j}_O{k[0]}N{k[1]}F{k[2]}"]["coh"] for k in K}
            if not all(map(math.isfinite, v.values())): continue
            b = v[(0, 0, 0)]; val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
            sh = {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S))) for k in range(3)
                         for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}
            y = c[f"j{j}_O1N1F1"]["y"]; k_ = cls(y[j - 1], y[j])
            for key in (k_, f"j{j}", "all"):
                a = agg[key]; a[0] += sh["O"]; a[1] += sh["N"]; a[2] += sh["F"]; a[3] += v[(1, 1, 1)] - b; a[4] += 1
print(f"{'class':6} {'n':>4} {'O':>7} {'N':>7} {'F':>7} {'T':>7} {'O/T':>6} {'N/T':>6}")
for k in ("all", "j2", "j3", "start", "cont", "punct"):
    if k not in agg: continue
    O, N, F, T, n = agg[k]; print(f"{k:6} {n:4d} {O/n:+.3f} {N/n:+.3f} {F/n:+.3f} {T/n:+.3f} {O/T if T else float('nan'):+.2f} {N/T if T else float('nan'):+.2f}")
