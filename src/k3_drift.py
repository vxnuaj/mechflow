"""Phase 6 / prereg §5 K3: instrument validation for single-position 'off' (phase4 §2.7).

Data: exploration-only GSM8K teacher-forced prefixes from the smoke test (excluded from
confirmation). Regime MR (g=1, noise 4.0, K=32), conditional branch.

(i)  Mixed-context OOD: residual-stream states (selected layers/steps) of ALL block positions under
     single-position off at i (candidates a/b/c, i cycles over blocks) vs reference distributions
     (full-on, block-off). Mahalanobis (PCA 95% var + Ledoit-Wolf) to the nearer reference;
     pass if 95th pct <= 1.5 x the references' own held-out 95th pct.
(ii) Cross-candidate agreement: single-step delta_D at targets j in {2,3} from sources i in {0,1}
     (off applied at one step s only), on log p_j(clean final token); sign and within factor 2.
(iii) Efficacy for b, c: applied at all positions approximates block-off (token agreement >= 0.9).
Usage: python k3_drift.py N out.json
"""
import json
import random
import re
import sys

import numpy as np
import torch
from datasets import load_dataset
from sklearn.covariance import LedoitWolf

from load import load_model, chat_prompt_ids
from lafa_instr import Sampler, BlockCfg, sc_apply, run_meta

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/k3_drift.json"
LAYERS = [3, 7, 11, 15, 19, 23]
STEPS = [4, 8, 12, 16, 20, 24]
CFG = BlockCfg(32, 1.0, 4.0, True, seed=7)
EXCL = set(json.load(open("../smoke/results/EXCLUDE_gsm8k_test_indices.json")))


def clean_solution(ans):
    ans = re.sub(r"<<[^>]*>>", "", ans)
    body, _, final = ans.partition("####")
    return body.strip() + f"\nThe final answer is {final.strip()}."


class Capture:
    def __init__(self, model):
        self.on = False
        self.buf = {}
        self.step = None
        layers = model.backbone.model.layers
        for li in LAYERS:
            layers[li].register_forward_hook(self._mk(li))

    def _mk(self, li):
        def hook(mod, inp, out):
            if self.on and self.step in STEPS:
                h = out[0] if isinstance(out, tuple) else out
                self.buf[(li, self.step)] = h[0, -4:].float().cpu().numpy()
        return hook


def main():
    tok, model = load_model()
    S = Sampler(model)
    cap = Capture(model)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = sorted(EXCL)  # exploration-only items already used by the smoke test
    rng = random.Random(11)
    rng.shuffle(idx)
    idx = idx[:N]
    states = {k: [] for k in ["on", "off", "a", "b", "c"]}   # list of (layer, step, pos, vec, block_i)
    agree = {"b": [], "c": []}
    dD = {c: [] for c in "abc"}

    def run(cache, L, sc_fn=None, record_states=None, track_u=False, ctx=None):
        cap.buf = {}
        cap.on = record_states is not None
        ctx = ctx if ctx is not None else {}

        def sc_wrap(s, br, z, e, c):
            cap.step = s if br == "c" else None
            return sc_fn(s, br, z, e, c) if sc_fn else (model._apply_self_cond(z, e) if e is not None else z)
        res = S.run_block(cache, L, CFG, sc_fn=sc_wrap, ctx=ctx, track_uncond=track_u)
        cap.on = False
        if record_states is not None:
            for (li, st), v in cap.buf.items():
                for p in range(4):
                    record_states.append((li, st, p, v[p]))
        return res

    for n, gi in enumerate(idx):
        q, a = ds[gi]["question"], ds[gi]["answer"]
        p_ids = chat_prompt_ids(tok, q)
        r_ids = tok(clean_solution(a), add_special_tokens=False).input_ids
        if len(r_ids) < 6:
            continue
        c = random.Random(gi).randint(0, len(r_ids) - 4)
        cache, L = S.prefill(p_ids + r_ids[:c])
        i_off = n % 4
        on = run(cache, L, record_states=states["on"], track_u=True)
        off = run(cache, L, sc_fn=lambda s, br, z, e, cx: z, record_states=states["off"])
        # candidate (a)
        run(cache, L, sc_fn=lambda s, br, z, e, cx: sc_apply(model, z, e, [i_off], "a"), record_states=states["a"])

        # candidate (b): bootstrap q at i_off
        def sc_b(s, br, z, e, cx):
            boot = cx["bootstrap_esc"]["c"]
            return sc_apply(model, z, e, [i_off], "b", repl=boot) if e is not None else z
        run(cache, L, sc_fn=sc_b, record_states=states["b"])

        # candidate (c): prefix-free branch's fed-back prediction at i_off (needs track_uncond)
        def sc_c(s, br, z, e, cx):
            if br == "u":
                return model._apply_self_cond(z, e) if e is not None else z
            ue = cx.get("esc", {}).get("u", cx["bootstrap_esc"]["u"])
            return sc_apply(model, z, e, [i_off], "c", repl=ue) if e is not None else z
        run(cache, L, sc_fn=sc_c, record_states=states["c"], track_u=True)

        # (iii) efficacy: b and c at ALL positions vs block-off tokens
        allp = [0, 1, 2, 3]
        rb = run(cache, L, sc_fn=lambda s, br, z, e, cx: sc_apply(model, z, e, allp, "b", repl=cx["bootstrap_esc"]["c"]) if e is not None else z)

        def sc_c_all(s, br, z, e, cx):
            if br == "u":
                return model._apply_self_cond(z, e) if e is not None else z
            ue = cx.get("esc", {}).get("u", cx["bootstrap_esc"]["u"])
            return sc_apply(model, z, e, allp, "c", repl=ue) if e is not None else z
        rc = run(cache, L, sc_fn=sc_c_all, track_u=True)
        agree["b"].append(np.mean([x == y for x, y in zip(rb.tokens, off.tokens)]))
        agree["c"].append(np.mean([x == y for x, y in zip(rc.tokens, off.tokens)]))

        # (ii) single-step delta_D: off at source i in {0,1} at step s only; read j in {2,3} at step s
        y = on.tokens
        for s_star in (8, 16):
            base_ctx = {"keep_probs": True}
            run(cache, L, track_u=True, ctx=base_ctx)
            pb = base_ctx["probs_hist"][s_star]["c"]
            for i in (0, 1):
                for cand in "abc":
                    def sc_one(s, br, z, e, cx, i=i, cand=cand):
                        if e is None:
                            return z
                        if br == "u" or s != s_star:
                            return model._apply_self_cond(z, e)
                        repl = None
                        if cand == "b":
                            repl = cx["bootstrap_esc"]["c"]
                        if cand == "c":
                            repl = cx.get("esc", {}).get("u", cx["bootstrap_esc"]["u"])
                        return sc_apply(model, z, e, [i], cand, repl=repl)
                    cx = {"keep_probs": True}
                    run(cache, L, sc_fn=sc_one, track_u=True, ctx=cx)
                    pi = cx["probs_hist"][s_star]["c"]
                    for j in (2, 3):
                        dD[cand].append(float(torch.log(pb[j, y[j]].clamp_min(1e-12)) - torch.log(pi[j, y[j]].clamp_min(1e-12))))
        print(n, flush=True)

    # (i) Mahalanobis
    def arr(lst, li, st, p):
        return np.array([v for (l, s, q, v) in lst if l == li and s == st and q == p])

    res_i = {c: [] for c in "abc"}
    for li in LAYERS:
        for st in STEPS:
            for p in range(4):
                ref = {}
                for nm in ("on", "off"):
                    X = arr(states[nm], li, st, p)
                    if len(X) < 20:
                        continue
                    half = len(X) // 2
                    Xf, Xh = X[:half], X[half:]
                    mu = Xf.mean(0)
                    U, Sg, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
                    var = np.cumsum(Sg ** 2) / np.sum(Sg ** 2)
                    k = int(np.searchsorted(var, 0.95) + 1)
                    P = Vt[:k].T
                    lw = LedoitWolf().fit((Xf - mu) @ P)
                    prec = lw.precision_

                    def md(Y, mu=mu, P=P, prec=prec):
                        Z = (Y - mu) @ P
                        return np.sqrt(np.einsum("ij,jk,ik->i", Z, prec, Z))
                    ref[nm] = (md, np.percentile(md(Xh), 95))
                if len(ref) < 2:
                    continue
                thr = 1.5 * max(ref["on"][1], ref["off"][1])
                for cand in "abc":
                    Y = arr(states[cand], li, st, p)
                    if len(Y) == 0:
                        continue
                    d = np.minimum(ref["on"][0](Y), ref["off"][0](Y))
                    res_i[cand].append(float(np.percentile(d, 95) / thr))
    summary = {
        "n_blocks": len(agree["b"]),
        "i_ood_ratio_max": {c: (max(v) if v else None) for c, v in res_i.items()},
        "i_ood_ratio_median": {c: (float(np.median(v)) if v else None) for c, v in res_i.items()},
        "i_pass": {c: (bool(v) and max(v) <= 1.0) for c, v in res_i.items()},
        "iii_efficacy_agreement": {c: float(np.mean(v)) for c, v in agree.items()},
        "ii_deltaD_mean": {c: float(np.mean(v)) for c, v in dD.items()},
        "ii_deltaD_se": {c: float(np.std(v, ddof=1) / np.sqrt(len(v))) for c, v in dD.items()},
    }
    json.dump({"meta": run_meta(CFG), "summary": summary, "ood_ratios": res_i}, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
