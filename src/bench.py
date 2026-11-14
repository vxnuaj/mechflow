"""Phase 6 throughput benchmark: forward passes/sec for single-block runs.
Usage: LAFA_REPO=... LAFA_REV=... python bench.py [dtype] [out.json]
"""
import json
import os
import sys
import time

import torch

import load
from lafa_instr import Sampler, BlockCfg, run_meta

dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[1] if len(sys.argv) > 1 else "bf16"]
out = sys.argv[2] if len(sys.argv) > 2 else "results/bench.json"
if dtype == torch.float32 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
tok, model = load.load_model(dtype)
if dtype == torch.float32:
    model = model.float()
S = Sampler(model)
prompts = ["Write one sentence about the ocean.", "What is 17 + 25? Answer briefly.",
           "Choose either New York or San Francisco. Answer with just the name."]
res = {}
for name, cfg in {"PR_g8_K32": BlockCfg(32, 8.0, 0.1, True), "MR_g1_K32": BlockCfg(32, 1.0, 4.0, True)}.items():
    caches = [S.prefill(load.chat_prompt_ids(tok, p)) for p in prompts]
    S.run_block(*caches[0], cfg)  # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0, nf = time.time(), 0
    for _ in range(3):
        for c, L in caches:
            nf += S.run_block(c, L, cfg).n_fwd
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    res[name] = {"passes": nf, "sec": dt, "ms_per_pass": 1000 * dt / nf}
res["max_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
meta = run_meta(extra={"repo": load.REPO, "rev": load.REVISION, "dtype": str(dtype),
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
json.dump({"meta": meta, "res": res}, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
