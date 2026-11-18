"""L2 (prereg §2.4, behavioural): circuit-validation bimodal prompts, paper regime (K = 32, g = 8, noise 0.1).
Per prompt, conditions in one batch: base (no cut, same code path), H*, and 4 of the 20 layer-matched random sets
(rotated by task_id, as in L1). Cut = at those heads every block query reads other block positions' K/V from a
no-feedback pass (`run_block_cutall_cfg_batched`). Two blocks (8 tokens), as P1; outputs labelled with the
audited text classifier (`classify_text`)."""
import os

import load
from classify_text import classify_text
from jobs_cut import cuts
from lafa_instr import BlockCfg, prefill_rows, run_block_cutall_cfg_batched

CFG = BlockCfg(32, 8.0, 0.1, True, seed=42)


def l2_task(task, ctx):
    tok, S = ctx["tok"], ctx["sampler"]
    C = cuts()
    names = ["base", "Hstar"] + [f"R{(4 * task['task_id'] + r) % 20:02d}" for r in range(4)]
    conds = [{} if n == "base" else C[n][0] for n in names]
    ids = load.chat_prompt_ids(tok, task["prompt"])
    rows = [list(ids) for _ in names]
    for _blk in range(2):
        cache, L = prefill_rows(S, rows)
        out = run_block_cutall_cfg_batched(S, cache, L, CFG, conds)
        rows = [r + y for r, (y, _) in zip(rows, out)]
        del cache
    res = {}
    for n, r in zip(names, rows):
        text = tok.decode(r[len(ids):], skip_special_tokens=True)
        res[n] = {"text": text, "label": classify_text(tok, text, task["mode_a"], task["mode_b"], task["tok_a"], task["tok_b"])}
    return {"conds": res, "random_sets": names[2:]}
