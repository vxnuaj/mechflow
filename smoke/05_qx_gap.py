"""Phase 4 diagnostic: does the fed-back prediction q_{k-1} carry mode information that the
state x_k does not (the "q-x information gap")? Exploratory; uses the smoke bimodal pair prompts.

For each pair prompt (modes A, B) and each distinguishing block position j, record at every
Euler step k (SC on, conditional branch):
  s_q(k) = cos(qE_j, E[a_j]) - cos(qE_j, E[b_j])     (feedback fed into step k)
  s_x(k) = cos(x_k,j, E[a_j]) - cos(x_k,j, E[b_j])   (state at step k)
and the final token at j. Decodability = P(sign(s) predicts the final mode | final in {A,B}).
Configs: g in {1,8} x noise in {0.1, 4.0}, K=32.
Usage: python 05_qx_gap.py [out.json]
"""
import importlib.util
import json
import sys

import numpy as np
import torch

from common import load, device, chat_ids

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/qx_gap.json"
_argv, sys.argv = sys.argv, [sys.argv[0]]
spec = importlib.util.spec_from_file_location("hyb", "03_hybrid.py")
hyb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hyb)
sys.argv = _argv

CONFIGS = {f"g{g}_n{n}": dict(gen_mode="flow", num_steps=32, cfg_scale=float(g), noise_scale_override=n)
           for g in (1, 8) for n in (0.1, 4.0)}


def answer_tokens(tok, prompt_text, ans):
    """Tokenize the answer in context: chat prompt + answer, take the answer span."""
    base = tok.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False,
                                   add_generation_prompt=True)
    p = tok(base, add_special_tokens=False).input_ids
    pa = tok(base + ans, add_special_tokens=False).input_ids
    assert pa[:len(p)] == p
    return pa[len(p):]


def main():
    tok, model = load(torch.bfloat16)
    dev = device()
    E = model.embed_tokens.weight.detach().float()
    En = E / E.norm(dim=-1, keepdim=True)
    rec_calls = []
    orig = model._apply_self_cond

    def hook(z, e_prev):
        rec_calls.append((z.detach().float()[0].clone(), e_prev.detach().float()[0].clone()))
        return orig(z, e_prev)

    model._apply_self_cond = hook
    pairs = [(cat, text) for cat, kind, text in hyb.prompts() if kind == "pair"]
    results = {c: [] for c in CONFIGS}
    for cat, text in pairs:
        # recover the two options from the template text
        names = [e for e in hyb.CATS[cat] if e in text]
        if len(names) != 2:
            continue
        a_name, b_name = names
        a = answer_tokens(tok, text, a_name)
        b = answer_tokens(tok, text, b_name)
        D = [j for j in range(min(4, len(a), len(b))) if a[j] != b[j]]
        if not D:
            continue
        inputs = chat_ids(tok, text, dev)
        for cname, kw in CONFIGS.items():
            rec_calls.clear()
            out = model.generate(**inputs, max_new_tokens=4, **kw)
            final = out[0, inputs.input_ids.shape[1]:].tolist()
            g = kw["cfg_scale"]
            calls = rec_calls[::2] if g > 1 else rec_calls  # conditional-branch calls only
            for j in D:
                if j >= len(final):
                    continue
                lab = "A" if final[j] == a[j] else ("B" if final[j] == b[j] else "none")
                sq, sx = [], []
                for (z, e) in calls:
                    zn = z[j] / z[j].norm()
                    en = e[j] / e[j].norm()
                    sq.append((en @ En[a[j]] - en @ En[b[j]]).item())
                    sx.append((zn @ En[a[j]] - zn @ En[b[j]]).item())
                results[cname].append({"prompt": text, "j": j, "final": lab, "sq": sq, "sx": sx})
        print(text[:60], flush=True)
    model._apply_self_cond = orig
    json.dump(results, open(OUT, "w"))

    # summary: decodability by step (sign predicts final mode), among final in {A,B}
    print("\nDecodability of the final mode by step (q vs x), j in distinguishing positions:")
    for cname, rows in results.items():
        rows = [r for r in rows if r["final"] in ("A", "B")]
        if not rows:
            continue
        K = min(len(r["sq"]) for r in rows)
        y = np.array([1 if r["final"] == "A" else -1 for r in rows])
        accq = [np.mean(np.sign([r["sq"][k] for r in rows]) == y) for k in range(K)]
        accx = [np.mean(np.sign([r["sx"][k] for r in rows]) == y) for k in range(K)]
        pick = [0, 2, 4, 8, 12, 16, 20, 24, 28, K - 1]
        print(f"{cname} (n={len(rows)}, steps={K})")
        print("   k  : " + " ".join(f"{k:5d}" for k in pick))
        print("   q  : " + " ".join(f"{accq[k]:5.2f}" for k in pick))
        print("   x  : " + " ".join(f"{accx[k]:5.2f}" for k in pick))


if __name__ == "__main__":
    main()
