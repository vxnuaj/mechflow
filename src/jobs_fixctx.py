"""Robustness R-FC (prereg §2.14): re-score the H2-N confirmation cells with a FIXED left context (red-team C4/#1, applied
to Lafa). For target j ∈ {2, 3} and each cell, coh_fx = log p_AR(y_j^cell | prompt, prefix, y_<j^{O1N1F1}). dep_fx =
coh_fx − log mean_s p_AR(y_j^cell | prefix, y'_<j,s), with the same S = 16 predecessor samples as the original (generator
seeded with task_id). No new generation. The output has the same schema as conf_h2n (cells: {key: {y, coh, dep}} +
identity flags) so analysis/confirm.py's h2n_units runs on it unchanged."""
import glob, json, os
import torch
import load
from jobs_h2n import ar_logprob, sample_prefixes, S_SAMPLES

SRC = os.environ.get("FIXCTX_SRC", "conf_h2n"); _R = None


def _res():
    global _R
    if _R is None:
        _R = {}
        for f in glob.glob(f"results/{SRC}/shard_*.jsonl"):
            for l in open(f):
                r = json.loads(l); _R[r["task_id"]] = r["result"]
    return _R


def fixctx_task(task, ctx):
    tok, model = ctx["tok"], ctx["model"]; res = _res()[task["task_id"]]
    p_ids = load.chat_prompt_ids(tok, task["user"]); r_ids = tok(task["response"], add_special_tokens=False).input_ids
    prefix = p_ids + r_ids[:res["cut"]]; L = len(prefix)
    gen = torch.Generator().manual_seed(task["task_id"])
    out = {"cut": res["cut"], "identity_on": res["identity_on"], "identity_off": res["identity_off"], "cells": {}}
    for j in (2, 3):
        samp = sample_prefixes(model, prefix, j, S_SAMPLES, gen)
        ref = res["cells"][f"j{j}_O1N1F1"]["y"]
        for key, cell in res["cells"].items():
            if not key.startswith(f"j{j}_"): continue
            yj = cell["y"][j]
            coh = float(ar_logprob(model, [prefix + ref[:j] + [yj]], L + j)[0])
            alt = ar_logprob(model, [s + [yj] for s in samp], L + j)
            marg = float(torch.logsumexp(alt, 0) - torch.log(torch.tensor(float(S_SAMPLES))))
            out["cells"][key] = {"y": cell["y"], "coh": coh, "dep": coh - marg}
    return out
