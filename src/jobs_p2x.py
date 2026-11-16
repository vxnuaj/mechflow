"""S3 / S4 confirmation job (prereg §2.2): the SC effect on dep_j (j = 2, 3) in two extra regimes, on the
same documents, cut points and sampled predecessors as the P2 job (same task_id → same cut and samples):
  G1N01 : g = 1, noise 0.1   (with P2's PR arm: S3 = SC × g interaction at noise 0.1)
  G1N4  : g = 1, noise 4.0   (S4: SC effect at noise 0.1 minus at noise 4.0, both g = 1)
"""
import random

import torch

import load
from lafa_instr import BlockCfg
from jobs_h2n import ar_logprob, sample_prefixes

S_SAMPLES = 16


def p2x_task(task, ctx):
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
    for reg, (g, nz) in (("G1N01", (1.0, 0.1)), ("G1N4", (1.0, 4.0))):
        for name, sc in (("SC", True), ("NO", False)):
            y = S.run_block(cache, L, BlockCfg(32, g, nz, sc, seed=42)).tokens
            coh = [float(ar_logprob(model, [prefix + y[:j + 1]], L + j)[0]) for j in range(4)]
            dep = [None]
            for j in range(1, 4):
                alt = ar_logprob(model, [s + [y[j]] for s in samp[j]], L + j)
                dep.append(coh[j] - float(torch.logsumexp(alt, 0) - torch.log(torch.tensor(float(S_SAMPLES)))))
            out[f"{reg}_{name}"] = {"y": y, "coh": coh, "dep": dep}
    return out
