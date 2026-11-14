"""Phase 3.5 sanity check: end-to-end GSM8K slice, SC on vs off at paper settings.

Prompt format approximates lm-eval gsm8k_cot zero-shot with chat template, so absolute
numbers are NOT comparable to the paper; only the SC on/off gap direction matters.
Uses GSM8K test indices disjoint from 02_block_tf.py's sample is NOT guaranteed; all
indices used here are logged and must be excluded from later confirmation splits.

Usage: python 01_gsm8k_e2e.py [N] [out.jsonl]
"""
import json
import random
import re
import sys
import time

import torch
from datasets import load_dataset

from common import load, device, chat_ids, PAPER

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/gsm8k_e2e.jsonl"
CONFIGS = {"SC": dict(PAPER), "noSC": dict(PAPER, self_cond=False)}


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def main():
    tok, model = load(torch.bfloat16)
    dev = device()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(777).sample(range(len(ds)), N)
    t0 = time.time()
    with open(OUT, "w") as f:
        for n, i in enumerate(idx):
            q = ds[i]["question"]
            gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
            inputs = chat_ids(tok, q + "\nLet's think step by step, then give the final answer as a number.", dev)
            rec = {"gsm8k_idx": i, "gold": gold, "out": {}}
            for name, kw in CONFIGS.items():
                o = model.generate(**inputs, max_new_tokens=512, **kw)
                text = tok.decode(o[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred = last_number(text)
                rec["out"][name] = {"text": text, "pred": pred, "correct": pred is not None and float(pred) == float(gold)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(n, f"{time.time() - t0:.0f}s", {k: v["correct"] for k, v in rec["out"].items()}, flush=True)


if __name__ == "__main__":
    main()
