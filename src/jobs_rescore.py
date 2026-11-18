"""Re-score existing H2-N cell outputs (no new generation): dep/coh with S predecessor samples (env S_RESCORE,
default 64) under the model in LAFA_REPO/LAFA_REV; reads results/<RESCORE_SRC> (default h2n_7b_300)."""
import glob, json, os, random
import torch
import load
from jobs_h2n import ar_logprob, sample_prefixes

S_N = int(os.environ.get("S_RESCORE", "64")); SRC = os.environ.get("RESCORE_SRC", "h2n_7b_300")
_R = None


def _res():
    global _R
    if _R is None:
        _R = {}
        for f in glob.glob(f"results/{SRC}/shard_*.jsonl"):
            for l in open(f):
                r = json.loads(l); _R[r["task_id"]] = r["result"]
    return _R


def rescore_task(task, ctx):
    tok, model = ctx["tok"], ctx["model"]
    res = _res()[task["task_id"]]
    p_ids = load.chat_prompt_ids(tok, task["user"])
    r_ids = tok(task["response"], add_special_tokens=False).input_ids
    prefix = p_ids + r_ids[:res["cut"]]; L = len(prefix)
    gen = torch.Generator().manual_seed(task["task_id"] + 64)
    out = {}
    for j in (2, 3):
        samp = []
        for _ in range(S_N // 16):
            samp += sample_prefixes(model, prefix, j, 16, gen)
        for key, cell in res["cells"].items():
            if not key.startswith(f"j{j}_"): continue
            y = cell["y"]
            coh = float(ar_logprob(model, [prefix + y[:j + 1]], L + j)[0])
            alt = torch.cat([ar_logprob(model, [s + [y[j]] for s in samp[i:i + 16]], L + j) for i in range(0, len(samp), 16)])
            out[key] = {"coh": coh, "dep": coh - float(torch.logsumexp(alt, 0) - torch.log(torch.tensor(float(len(samp)))))}
    return out
