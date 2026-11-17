"""Batched version of jobs_cut.cut_task (same outputs; conditions run BATCH at a time in one batch).
Validated against jobs_cut on exploration docs (tokens identical; logits differ only by bf16 batch rounding)."""
import os, random

import torch
import torch.nn.functional as F

import load
from lafa_instr import prefill_batched, run_block_headcut_batched
from jobs_cut import CFG, S_SAMPLES, cuts
from jobs_h2n import ar_logprob, sample_prefixes

BATCH = int(os.environ.get("BATCH", "8"))


def cut_task(task, ctx):
    tok, model, S = ctx["tok"], ctx["model"], ctx["sampler"]
    p_ids = load.chat_prompt_ids(tok, task["user"])
    r_ids = tok(task["response"], add_special_tokens=False).input_ids
    rng = random.Random(task["task_id"])
    c = 4 * rng.randint(0, max(0, (len(r_ids) - 8) // 4))
    prefix = p_ids + r_ids[:c]
    gen = torch.Generator().manual_seed(task["task_id"])
    names = ["base"] + list(cuts())
    conds = [({}, None)] + list(cuts().values())
    out = {"cut": c}
    for j in (2, 3):
        samp = sample_prefixes(model, prefix, j, S_SAMPLES, gen)
        res = []
        for a in range(0, len(conds), BATCH):
            chunk = conds[a:a + BATCH]
            cB, L = prefill_batched(S, prefix, len(chunk))
            res += run_block_headcut_batched(S, cB, L, CFG, j, chunk)
            del cB
        yb, lb = res[0]
        lpb = F.log_softmax(lb[j].float(), -1)
        for name, (y, lg) in zip(names, res):
            coh = float(ar_logprob(model, [prefix + y[:j + 1]], L + j)[0])
            alt = ar_logprob(model, [s + [y[j]] for s in samp], L + j)
            dep = coh - float(torch.logsumexp(alt, 0) - torch.log(torch.tensor(float(S_SAMPLES))))
            drop = 0.0 if name == "base" else float(lpb[yb[j]] - F.log_softmax(lg[j].float(), -1)[yb[j]])
            out[f"j{j}|{name}"] = {"y": y, "coh": coh, "dep": dep, "drop": drop}
    return out
