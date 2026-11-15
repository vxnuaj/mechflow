"""P1 pilot job (exploration): hybrid rate, flow vs. AR, SC on/off (prereg §2.1 P1; phase4 §2.3).

One task = one bank pair prompt. Conditions (paper regime for flow):
  AR      : gen_mode="ar", 8 new tokens
  FLOW_SC : 32 steps, g=8, noise 0.1, SC on, 2 blocks (8 tokens)
  FLOW_NO : same, SC off
Token-aligned classification of the answer span (generated ids up to the first newline/period/EOS):
  mode_a / mode_b : span equals that mode's in-context token sequence
  hybrid          : span = non-empty prefix of one mode + non-empty suffix of the other (token level)
  other           : anything else (garble / other-valid / invalid are separated later with judges)
"""
import torch

import load


def _span(tok, ids):
    out = []
    for t in ids:
        s = tok.decode([t])
        if t == tok.eos_token_id or "\n" in s or s.strip() in (".", "!", "?") or "<|im_end|>" in s:
            break
        out.append(t)
    return out


def classify(span, a, b):
    if span == a:
        return "mode_a"
    if span == b:
        return "mode_b"
    for x, y in ((a, b), (b, a)):
        for i in range(1, len(x)):
            for j in range(1, len(y)):
                if span == x[:i] + y[j:] and span not in (a, b):
                    return "hybrid"
    return "other"


def p1_task(task, ctx):
    tok, model = ctx["tok"], ctx["model"]
    ids = load.chat_prompt_ids(tok, task["prompt"])
    inp = torch.tensor([ids], device=model.embed_tokens.weight.device)
    out = {}
    for name, kw in {"AR": dict(gen_mode="ar"),
                     "FLOW_SC": dict(gen_mode="flow", num_steps=32, cfg_scale=8.0, noise_scale_override=0.1, self_cond=True),
                     "FLOW_NO": dict(gen_mode="flow", num_steps=32, cfg_scale=8.0, noise_scale_override=0.1, self_cond=False)}.items():
        o = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=8, **kw)
        gen = o[0, len(ids):].tolist()
        sp = _span(tok, gen)
        out[name] = {"label": classify(sp, task["tok_a"], task["tok_b"]), "text": tok.decode(gen, skip_special_tokens=True)}
    return {"labels": {k: v["label"] for k, v in out.items()}, "texts": {k: v["text"] for k, v in out.items()},
            "hyb_flow_minus_ar": int(out["FLOW_SC"]["label"] == "hybrid") - int(out["AR"]["label"] == "hybrid"),
            "hyb_sc": int(out["FLOW_SC"]["label"] == "hybrid"), "hyb_nosc": int(out["FLOW_NO"]["label"] == "hybrid")}
