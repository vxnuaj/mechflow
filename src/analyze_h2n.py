"""2x2x2 (O x N x F) decomposition of SC's effect at target positions 2-3 (phase4 §2.6; prereg H2-N)."""
import glob, itertools, json, math, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/h2n_pilot"
R = [json.loads(l) for f in glob.glob(path + "/shard_*.jsonl") for l in open(f)]
print("blocks:", len(R), "| identity on/off all true:", all(r['result']['identity_on'] for r in R), all(r['result']['identity_off'] for r in R))
for m in ("dep", "coh"):
    rows = []
    for r in R:
        c = r['result']['cells']
        for j in (2, 3):
            v = {k: c[f"j{j}_O{k[0]}N{k[1]}F{k[2]}"][m] for k in itertools.product((0, 1), repeat=3)}
            b = v[(0, 0, 0)]
            mO, mN, mF = v[(1, 0, 0)] - b, v[(0, 1, 0)] - b, v[(0, 0, 1)] - b
            mON = v[(1, 1, 0)] - v[(1, 0, 0)] - v[(0, 1, 0)] + b
            mOF = v[(1, 0, 1)] - v[(1, 0, 0)] - v[(0, 0, 1)] + b
            mNF = v[(0, 1, 1)] - v[(0, 1, 0)] - v[(0, 0, 1)] + b
            T = v[(1, 1, 1)] - b
            mONF = T - (mO + mN + mF + mON + mOF + mNF)
            val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
            sh = {}
            for p in "ONF":
                others = [q for q in "ONF" if q != p]
                sh[p] = sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S)))
                            for k in range(3) for S in itertools.combinations(others, k))
            rows.append(dict(block=r['task_id'], j=j, T=T, mO=mO, mN=mN, mF=mF, mON=mON, mOF=mOF, mNF=mNF, mONF=mONF,
                             shO=sh['O'], shN=sh['N'], shF=sh['F']))
    A = {k: np.array([x[k] for x in rows]) for k in rows[0] if k not in ("block", "j")}
    blocks = sorted({x['block'] for x in rows})
    idx = {b: [i for i, x in enumerate(rows) if x['block'] == b] for b in blocks}
    rng = np.random.default_rng(0)

    def boot(fn):
        est = fn(np.arange(len(rows))); vals = []
        for _ in range(4000):
            s = rng.choice(len(blocks), len(blocks)); ii = np.concatenate([idx[blocks[k]] for k in s]); vals.append(fn(ii))
        return est, np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    print(f"\n== outcome: {m}  (positions 2-3 pooled; n={len(rows)} target-blocks)")
    e, lo, hi = boot(lambda ii: A['T'][ii].mean()); print(f"  total SC effect T (full - all-off):            {e:+.3f} [{lo:+.3f},{hi:+.3f}]")
    for k, lab in (("mO", "O: target's own feedback (main)"), ("mN", "N: reads neighbours' feedback (main)"),
                   ("mF", "F: neighbours' own feedback/state route (main)")):
        e, lo, hi = boot(lambda ii, k=k: A[k][ii].mean()); print(f"  {lab:46s} {e:+.3f} [{lo:+.3f},{hi:+.3f}]")
    for k in ("mON", "mOF", "mNF", "mONF"):
        e, lo, hi = boot(lambda ii, k=k: A[k][ii].mean()); print(f"  interaction {k:34s} {e:+.3f} [{lo:+.3f},{hi:+.3f}]")
    for k, lab in (("mN", "N share (Mobius main / T)"), ("shN", "N share (Shapley / T)"), ("shO", "O share (Shapley / T)"),
                   ("shF", "F share (Shapley / T)")):
        e, lo, hi = boot(lambda ii, k=k: A[k][ii].sum() / A['T'][ii].sum()); print(f"  {lab:46s} {e:.3f} [{lo:.3f},{hi:.3f}]")
    e, lo, hi = boot(lambda ii: (A['mN'][ii] - 0.2 * A['T'][ii]).mean()); print(f"  pre-reg contrast N - 0.20*T (>0 supports M2): {e:+.3f} [{lo:+.3f},{hi:+.3f}]")
