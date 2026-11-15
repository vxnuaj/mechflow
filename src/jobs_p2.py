"""P2 pilot job (exploration): does SC improve in-block coherence/dependence at PR, under the own-AR
judge AND an external judge (Mistral-7B-Instruct-v0.3)? (prereg §2.1 P2 + co-requirement)

Task = natural block (UltraChat exploration). Generate one block at PR (K=32, g=8, noise 0.1) with
SC on and SC off. For each output y:
  own-AR: coh_j = log p_lafaAR(y_j | c, y_<j); dep_j (sampled-predecessor PMI, S=16)   j = 0..3
  external: Mistral log-prob of each Lafa token's character span given the detokenized context
"""
import random

import torch

import load
from lafa_instr import BlockCfg
from jobs_h2n import ar_logprob, sample_prefixes

JUDGE = "mistralai/Mistral-7B-Instruct-v0.3"
S_SAMPLES = 16


def _judge(ctx):
    if "judge" not in ctx:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        jt = AutoTokenizer.from_pretrained(JUDGE)
        jm = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16).to(load.device()).eval()
        ctx["judge"] = (jt, jm)
    return ctx["judge"]


@torch.no_grad()
def judge_spans(ctx, user, resp_prefix, piece_texts):
    """Mistral log-prob summed per Lafa piece (character spans), conditioned on the chat context."""
    jt, jm = _judge(ctx)
    head = jt.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    ctx_text = head + resp_prefix
    full = ctx_text + "".join(piece_texts)
    enc = jt(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    x = torch.tensor([ids], device=jm.device)
    lp = torch.log_softmax(jm(input_ids=x).logits[0].float(), -1)
    bounds, start = [], len(ctx_text)
    for p in piece_texts:
        bounds.append((start, start + len(p))); start += len(p)
    out = [0.0] * len(piece_texts)
    for t in range(1, len(ids)):
        s = offs[t][0]
        for k, (a, b) in enumerate(bounds):
            if a <= s < b:
                out[k] += float(lp[t - 1, ids[t]])
    return out


def p2_task(task, ctx):
    tok, model, S = ctx["tok"], ctx["model"], ctx["sampler"]
    p_ids = load.chat_prompt_ids(tok, task["user"])
    r_ids = tok(task["response"], add_special_tokens=False).input_ids
    rng = random.Random(task["task_id"] + 777)
    cut = 4 * rng.randint(0, max(0, (len(r_ids) - 8) // 4))
    prefix = p_ids + r_ids[:cut]
    cache, L = S.prefill(prefix)
    gen = torch.Generator().manual_seed(task["task_id"])
    samp = {j: sample_prefixes(model, prefix, j, S_SAMPLES, gen) for j in range(1, 4)}
    out = {"cut": cut}
    resp_prefix = tok.decode(r_ids[:cut])
    for name, sc in (("SC", True), ("NO", False)):
        y = S.run_block(cache, L, BlockCfg(32, 8.0, 0.1, sc, seed=42)).tokens
        coh = [float(ar_logprob(model, [prefix + y[:j + 1]], L + j)[0]) for j in range(4)]
        dep = [None]
        for j in range(1, 4):
            alt = ar_logprob(model, [s + [y[j]] for s in samp[j]], L + j)
            marg = float(torch.logsumexp(alt, 0) - torch.log(torch.tensor(float(S_SAMPLES))))
            dep.append(coh[j] - marg)
        pieces = [tok.decode(y[:j + 1])[len(tok.decode(y[:j])):] for j in range(4)]
        ext = judge_spans(ctx, task["user"], resp_prefix, pieces)
        out[name] = {"y": y, "coh": coh, "dep": dep, "ext": ext}
    return out
