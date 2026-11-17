"""E1 (exploratory, exploration split): step × layer map of the neighbour-feedback (N) effect. insights.md §2.
For each natural block and target j in {2, 3}: cut j's reads of the neighbours' SC-carrying K/V (all heads in the
given layers, at the given denoising steps) and measure dep_j and coh_j, versus the bf16-matched uncut baseline.
MR regime (g = 1, noise 4.0, K = 32), 0.5B. All conditions use the same B + A' code path (reroute=True)."""
import random

import torch

import load
from lafa_instr import BlockCfg, run_block_headcut
from jobs_h2n import ar_logprob, sample_prefixes

CFG = BlockCfg(32, 1.0, 4.0, True, seed=1)
S_SAMPLES = 16
NL, H = 24, 14
BANDS = {f"L{a}-{a+3}": list(range(a, a + 4)) for a in range(0, 24, 4)}
SINGLE = {"L0": [0], "L1": [1]}
WINDOWS = {f"s{a}-{a+7}": set(range(a, a + 8)) for a in range(0, 32, 8)}
COARSE = {"L0-7": list(range(0, 8)), "L8-15": list(range(8, 16)), "L16-23": list(range(16, 24))}


def conditions():
    allL = list(range(NL))
    C = {"base": (None, None), "all": (allL, None)}
    for k, v in {**BANDS, **SINGLE}.items():
        C[k] = (v, None)
    for k, w in WINDOWS.items():
        C[k] = (allL, w)
        for kb, b in COARSE.items():
            C[f"{k}|{kb}"] = (b, w)
    return C


def e1_task(task, ctx):
    tok, model, S = ctx["tok"], ctx["model"], ctx["sampler"]
    p_ids = load.chat_prompt_ids(tok, task["user"])
    r_ids = tok(task["response"], add_special_tokens=False).input_ids
    rng = random.Random(task["task_id"])
    cut = 4 * rng.randint(0, max(0, (len(r_ids) - 8) // 4))
    prefix = p_ids + r_ids[:cut]
    cache, L = S.prefill(prefix)
    gen = torch.Generator().manual_seed(task["task_id"])
    out = {"cut": cut}
    for j in (2, 3):
        samp = sample_prefixes(model, prefix, j, S_SAMPLES, gen)
        for name, (layers, steps) in conditions().items():
            cutd = {} if layers is None else {l: list(range(H)) for l in layers}
            y, _ = run_block_headcut(S, cache, L, CFG, j, cutd, reroute=True, steps=steps)
            coh = float(ar_logprob(model, [prefix + y[:j + 1]], L + j)[0])
            alt = ar_logprob(model, [s + [y[j]] for s in samp], L + j)
            dep = coh - float(torch.logsumexp(alt, 0) - torch.log(torch.tensor(float(S_SAMPLES))))
            out[f"j{j}|{name}"] = {"y": y, "coh": coh, "dep": dep}
    return out
