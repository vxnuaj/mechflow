"""Generic neighbour-feedback cut job (exploratory; Phase 9.5/10 localization screens).
Cuts are read from the JSON file in env CUTS: {name: {"layers": {layer: [heads]}, "steps": [..] or null}}.
For each block and target j in {2, 3} (MR, 0.5B): run the bf16-matched uncut baseline (reroute=True), then each cut.
Records per cut: y, dep_j, coh_j, and the smooth screen metric drop_j = log p_base(y_j) − log p_cut(y_j)
(the final-step distribution at j, evaluated at the baseline's own output token)."""
import json, os, random

import torch
import torch.nn.functional as F

import load
from lafa_instr import BlockCfg, run_block_headcut
from jobs_h2n import ar_logprob, sample_prefixes

CFG = BlockCfg(32, 1.0, 4.0, True, seed=1, grid_mode=os.environ.get("GRID", "warp"))
S_SAMPLES = 16
_CUTS = None


def cuts():
    global _CUTS
    if _CUTS is None:
        raw = json.load(open(os.environ["CUTS"]))
        _CUTS = {k: ({int(l): h for l, h in v["layers"].items()}, set(v["steps"]) if v.get("steps") else None)
                 for k, v in raw.items()}
    return _CUTS


def cut_task(task, ctx):
    tok, model, S = ctx["tok"], ctx["model"], ctx["sampler"]
    p_ids = load.chat_prompt_ids(tok, task["user"])
    r_ids = tok(task["response"], add_special_tokens=False).input_ids
    rng = random.Random(task["task_id"])
    c = 4 * rng.randint(0, max(0, (len(r_ids) - 8) // 4))
    prefix = p_ids + r_ids[:c]
    cache, L = S.prefill(prefix)
    gen = torch.Generator().manual_seed(task["task_id"])
    out = {"cut": c}
    for j in (2, 3):
        samp = sample_prefixes(model, prefix, j, S_SAMPLES, gen)

        def score(y):
            coh = float(ar_logprob(model, [prefix + y[:j + 1]], L + j)[0])
            alt = ar_logprob(model, [s + [y[j]] for s in samp], L + j)
            return coh, coh - float(torch.logsumexp(alt, 0) - torch.log(torch.tensor(float(S_SAMPLES))))

        yb, lb = run_block_headcut(S, cache, L, CFG, j, {}, reroute=True)
        lpb = F.log_softmax(lb[j].float(), -1)
        coh, dep = score(yb)
        out[f"j{j}|base"] = {"y": yb, "coh": coh, "dep": dep, "drop": 0.0}
        for name, (layers, steps) in cuts().items():
            y, lg = run_block_headcut(S, cache, L, CFG, j, layers, reroute=True, steps=steps)
            lp = F.log_softmax(lg[j].float(), -1)
            coh, dep = score(y)
            out[f"j{j}|{name}"] = {"y": y, "coh": coh, "dep": dep, "drop": float(lpb[yb[j]] - lp[yb[j]])}
    return out
