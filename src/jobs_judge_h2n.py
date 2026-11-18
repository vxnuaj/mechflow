"""G-b (exploratory re-analysis): score the confirmation H2-N cell outputs with the external judge
(Mistral-7B-Instruct-v0.3, same span scoring as P2). No new Lafa generation: reads results/conf_h2n.
Per doc, target j in {2,3}, cell (O,N,F): ext_j = Mistral log-prob of the character span of token j given the
chat context, the response prefix and the cell's tokens 0..j-1."""
import glob, json

import load
from jobs_p2 import judge_spans

_R = None


def _results():
    global _R
    if _R is None:
        _R = {}
        for f in glob.glob("results/conf_h2n/shard_*.jsonl"):
            for l in open(f):
                r = json.loads(l); _R[r["task_id"]] = r["result"]
    return _R


def judge_task(task, ctx):
    tok = ctx["tok"]
    res = _results()[task["task_id"]]
    r_ids = tok(task["response"], add_special_tokens=False).input_ids
    resp_prefix = tok.decode(r_ids[:res["cut"]])
    out = {}
    for key, cell in res["cells"].items():
        y = cell["y"]
        pieces = [tok.decode(y[:k + 1])[len(tok.decode(y[:k])):] for k in range(4)]
        out[key] = judge_spans(ctx, task["user"], resp_prefix, pieces)
    return out
