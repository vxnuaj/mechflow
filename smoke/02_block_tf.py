"""Phase 3.5 smoke test: teacher-forced single-block denoising on GSM8K gold solutions.

For each example: prefix = chat prompt + gold solution truncated at a random cut c.
Generate exactly one 4-token flow block under several decoding configs and score it
against the gold tokens resp[c:c+4]. Exploratory only; the GSM8K indices used are saved
and must be excluded from any later confirmation split.

Usage: python 02_block_tf.py [N] [out.jsonl]
"""
import json
import random
import re
import sys
import time

import torch
from datasets import load_dataset

from common import load, device

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/block_tf.jsonl"
B = 4
rng = random.Random(1234)

CONFIGS = {
    # name: generate kwargs (flow configs all use the trained 'off' = self_cond=False)
    "paper_K32_g8_n0.1_SC": dict(gen_mode="flow", num_steps=32, cfg_scale=8.0, noise_scale_override=0.1),
    "K32_g8_n0.1_noSC": dict(gen_mode="flow", num_steps=32, cfg_scale=8.0, noise_scale_override=0.1, self_cond=False),
    "K32_g1_n0.1_SC": dict(gen_mode="flow", num_steps=32, cfg_scale=1.0, noise_scale_override=0.1),
    "K32_g1_n0.1_noSC": dict(gen_mode="flow", num_steps=32, cfg_scale=1.0, noise_scale_override=0.1, self_cond=False),
    "K32_g8_n4.0_SC": dict(gen_mode="flow", num_steps=32, cfg_scale=8.0, noise_scale_override=4.0),
    "K32_g8_n4.0_noSC": dict(gen_mode="flow", num_steps=32, cfg_scale=8.0, noise_scale_override=4.0, self_cond=False),
    "K4_g8_n0.1_SC": dict(gen_mode="flow", num_steps=4, cfg_scale=8.0, noise_scale_override=0.1),
    "K4_g8_n0.1_noSC": dict(gen_mode="flow", num_steps=4, cfg_scale=8.0, noise_scale_override=0.1, self_cond=False),
    "AR": dict(gen_mode="ar"),
}


def clean_solution(ans: str) -> str:
    ans = re.sub(r"<<[^>]*>>", "", ans)
    body, _, final = ans.partition("####")
    return body.strip() + f"\nThe final answer is {final.strip()}."


def main():
    tok, model = load(torch.bfloat16)
    dev = device()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = rng.sample(range(len(ds)), N)
    import os
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    t0 = time.time()
    with open(OUT, "w") as f:
        for n, i in enumerate(idx):
            q, a = ds[i]["question"], ds[i]["answer"]
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            r_ids = tok(clean_solution(a), add_special_tokens=False).input_ids
            if len(r_ids) < B + 1:
                continue
            c = rng.randint(0, len(r_ids) - B)
            prefix = torch.tensor([p_ids + r_ids[:c]], device=dev)
            target = r_ids[c:c + B]
            rec = {"gsm8k_idx": i, "cut": c, "target": target, "out": {}}
            for name, kw in CONFIGS.items():
                o = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                   max_new_tokens=B, **kw)
                gen = o[0, prefix.shape[1]:].tolist()[:B]
                rec["out"][name] = gen
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{N}  {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
