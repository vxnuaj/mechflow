"""Score each block from 02_block_tf.py by the checkpoint's own AR log-probability given the prefix.

Question: when SC on/off produce different blocks, is the no-SC block a *less coherent*
continuation (lower AR log-prob; hybrid-like), or an equally good alternative?
Usage: python 04_coherence.py [in.jsonl] [out.jsonl]
"""
import importlib.util
import json
import sys

import torch
from datasets import load_dataset

from common import load, device

IN = sys.argv[1] if len(sys.argv) > 1 else "results/block_tf.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/block_tf_coherence.jsonl"
_argv, sys.argv = sys.argv, [sys.argv[0]]  # 02_block_tf parses argv at import time
spec = importlib.util.spec_from_file_location("btf", "02_block_tf.py")
btf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(btf)
sys.argv = _argv


@torch.no_grad()
def block_logprob(model, prefix, block, dev):
    ids = torch.tensor([prefix + block], device=dev)
    logits = model(input_ids=ids).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    L = len(prefix)
    per = [lp[L - 1 + j, block[j]].item() for j in range(len(block))]
    return per


def main():
    tok, model = load(torch.bfloat16)
    dev = device()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    with open(IN) as fi, open(OUT, "w") as fo:
        for n, line in enumerate(fi):
            r = json.loads(line)
            q, a = ds[r["gsm8k_idx"]]["question"], ds[r["gsm8k_idx"]]["answer"]
            prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            r_ids = tok(btf.clean_solution(a), add_special_tokens=False).input_ids
            prefix = p_ids + r_ids[:r["cut"]]
            assert r_ids[r["cut"]:r["cut"] + 4] == r["target"], "prefix reconstruction mismatch"
            scores = {"gold": block_logprob(model, prefix, r["target"], dev)}
            for name, blk in r["out"].items():
                if len(blk) == 4:
                    scores[name] = block_logprob(model, prefix, blk, dev)
            fo.write(json.dumps({"gsm8k_idx": r["gsm8k_idx"], "lp": scores}) + "\n")
            if n % 50 == 0:
                print(n, flush=True)


if __name__ == "__main__":
    main()
