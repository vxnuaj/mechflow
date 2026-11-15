"""H2-N pilot job (exploration, MR regime: g=1, noise 4.0, K=32): 2x2x2 O x N x F decomposition.

Task = one natural-text block: prefix = chat(user) + response[:cut] (cut block-aligned, random).
For each target j in {2,3} and each cell (O,N,F) in {0,1}^3:
  pass A: SC at neighbours (all p != j) + at j if O;   pass B: SC at j only if O
  take[j] = A if N else B;  take[p != j] = A if F else B
Outcome on the produced block y: own-AR coh_j and dep_j (sampled-predecessor PMI, S=16).
Identity checks per task: cell (1,1,1) == normal SC run; cell (0,0,0) == SC-off run (tokens).
"""
import itertools
import random

import torch
import torch.nn.functional as F

import load
from lafa_instr import BlockCfg, run_block_two_pass

CFG = BlockCfg(32, 1.0, 4.0, True, seed=1)
S_SAMPLES = 16


@torch.no_grad()
def ar_logprob(model, seqs, pos):
    """log p_AR(seq[pos] | seq[:pos]) for a batch of equal-length sequences."""
    x = torch.tensor(seqs, device=model.embed_tokens.weight.device)
    lg = model(input_ids=x).logits.float()
    lp = torch.log_softmax(lg[:, pos - 1], -1)
    return lp.gather(1, x[:, pos:pos + 1])[:, 0].cpu()


@torch.no_grad()
def sample_prefixes(model, prefix, n_tokens, S, gen):
    seqs = [list(prefix) for _ in range(S)]
    for _ in range(n_tokens):
        x = torch.tensor(seqs, device=model.embed_tokens.weight.device)
        p = torch.softmax(model(input_ids=x).logits[:, -1].float(), -1)
        nxt = torch.multinomial(p.cpu(), 1, generator=gen)[:, 0].tolist()
        for s, t in zip(seqs, nxt):
            s.append(t)
    return seqs


def h2n_task(task, ctx):
    tok, model, S = ctx["tok"], ctx["model"], ctx["sampler"]
    p_ids = load.chat_prompt_ids(tok, task["user"])
    r_ids = tok(task["response"], add_special_tokens=False).input_ids
    rng = random.Random(task["task_id"])
    cut = 4 * rng.randint(0, max(0, (len(r_ids) - 8) // 4))
    prefix = p_ids + r_ids[:cut]
    cache, L = S.prefill(prefix)
    allp = {0, 1, 2, 3}
    out = {"cut": cut, "cells": {}}
    # identity checks
    y_on, _ = run_block_two_pass(S, cache, L, CFG, fbA=allp, fbB=set(), take=["A"] * 4)
    ref_on = S.run_block(cache, L, CFG).tokens
    y_off, _ = run_block_two_pass(S, cache, L, CFG, fbA=allp, fbB=set(), take=["B"] * 4)
    ref_off = S.run_block(cache, L, BlockCfg(32, 1.0, 4.0, False, seed=CFG.seed)).tokens
    out["identity_on"] = y_on == ref_on
    out["identity_off"] = y_off == ref_off
    gen = torch.Generator().manual_seed(task["task_id"])
    for j in (2, 3):
        nb = allp - {j}
        samp = sample_prefixes(model, prefix, j, S_SAMPLES, gen)      # y' ~ p_AR, positions < j
        for O, N, Fc in itertools.product((0, 1), repeat=3):
            fbA = set(nb) | ({j} if O else set())
            fbB = {j} if O else set()
            take = ["A" if Fc else "B"] * 4
            take[j] = "A" if N else "B"
            y, _ = run_block_two_pass(S, cache, L, CFG, fbA=fbA, fbB=fbB, take=take)
            coh = float(ar_logprob(model, [prefix + y[:j + 1]], L + j)[0])
            alt = ar_logprob(model, [s + [y[j]] for s in samp], L + j)
            marg = float(torch.logsumexp(alt, 0) - torch.log(torch.tensor(float(S_SAMPLES))))
            out["cells"][f"j{j}_O{O}N{N}F{Fc}"] = {"y": y, "coh": coh, "dep": coh - marg}
    return out
