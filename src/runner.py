"""Sharded multi-worker runner for latency-bound single-block experiments.

Usage (inside a SLURM job):
  python runner.py --tasks tasks.jsonl --job module:function --out results/run_name \
                   --workers-per-gpu 6
Each task line is a JSON dict. The job function signature is fn(task: dict, ctx: dict) -> dict,
where ctx holds {"tok", "model", "sampler"} loaded once per worker. Output: one JSONL per worker
(shard), each line = {"task_id", "result", "meta"}; the run's meta header goes to meta.json.
Sharding: global shard = array_task_id * workers_per_gpu + worker_idx (SLURM_ARRAY_TASK_ID,
SLURM_ARRAY_TASK_COUNT). Resumable: tasks already present in a shard's output are skipped.
"""
import argparse
import importlib
import json
import os
import sys
import time

import torch.multiprocessing as mp


def worker(widx, args, n_shards_total, shard_base):
    import torch
    import load
    from lafa_instr import Sampler, run_meta
    shard = shard_base + widx
    out_path = os.path.join(args.out, f"shard_{shard:04d}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["task_id"])
            except Exception:
                pass
    tasks = [json.loads(l) for l in open(args.tasks)]
    mine = [t for i, t in enumerate(tasks) if i % n_shards_total == shard and t["task_id"] not in done]
    if not mine:
        return
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if dtype == torch.float32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tok, model = load.load_model(dtype)
    if dtype == torch.float32:
        model = model.float()
    ctx = {"tok": tok, "model": model, "sampler": Sampler(model)}
    mod, fn = args.job.split(":")
    job = getattr(importlib.import_module(mod), fn)
    meta = run_meta(extra={"job": args.job, "shard": shard, "dtype": args.dtype,
                           "repo": load.REPO, "rev": load.REVISION})
    with open(out_path, "a") as f:
        for t in mine:
            t0 = time.time()
            r = job(t, ctx)
            f.write(json.dumps({"task_id": t["task_id"], "result": r, "sec": time.time() - t0,
                                "code_sha256": meta["code_sha256"]}) + "\n")
            f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arr = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    narr = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    w = args.workers_per_gpu
    n_total = narr * w
    if arr == 0:
        from lafa_instr import run_meta
        json.dump(run_meta(extra=vars(args)), open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    mp.set_start_method("spawn", force=True)
    procs = [mp.Process(target=worker, args=(i, args, n_total, arr * w)) for i in range(w)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
