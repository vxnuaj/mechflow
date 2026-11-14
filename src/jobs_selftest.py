"""Tiny job used to validate runner.py end-to-end: one paper-regime block per task."""
from lafa_instr import BlockCfg
import load


def one_block(task, ctx):
    S, tok = ctx["sampler"], ctx["tok"]
    cache, L = S.prefill(load.chat_prompt_ids(tok, task["prompt"]))
    r = S.run_block(cache, L, BlockCfg(32, 8.0, 0.1, True, seed=task.get("seed", 42)), record=True)
    return {"tokens": r.tokens, "n_fwd": r.n_fwd, "rec_steps": len(r.rec["z"])}
