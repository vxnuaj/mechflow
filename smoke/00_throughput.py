"""Step 1: does Lafa-0.5B run here, and how fast? Prints outputs and timing for a few configs."""
import sys
import torch
from common import load, chat_ids, gen, PAPER, device

dtype = torch.float32 if "--fp32" in sys.argv else torch.bfloat16
tok, model = load(dtype)
dev = device()
print("device", dev, "dtype", dtype)

inputs = chat_ids(tok, "What is 17 + 25? Answer briefly.", dev)
configs = [
    ("ar", dict(gen_mode="ar")),
    ("flow K4 g1 (HF default)", dict(gen_mode="flow", num_steps=4, cfg_scale=1.0, noise_scale_override=0.1)),
    ("flow K32 g8 SC on (paper)", dict(PAPER)),
    ("flow K32 g8 SC off", dict(PAPER, self_cond=False)),
]
for name, kw in configs:
    gen(model, tok, inputs, max_new_tokens=8, **kw)  # warm-up
    text, dt, st = gen(model, tok, inputs, max_new_tokens=32, **kw)
    print(f"\n== {name}: {dt:.2f}s for 32 tokens | stats={ {k: st[k] for k in list(st)[:6]} }\n{text!r}")
