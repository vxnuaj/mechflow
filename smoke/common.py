"""Shared loading utilities for the Phase 3.5 smoke test (exploratory only)."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "lafa-hf/lafa-0.5B"
REVISION = "18deca3522f7884898a58d0fb2d2afb3fc91f87e"  # snapshot downloaded 2026-11-13

# Paper settings (Table 3/4): 32 Euler steps, prefix guidance g=8, inference noise 0.1.
PAPER = dict(gen_mode="flow", num_steps=32, cfg_scale=8.0, noise_scale_override=0.1)


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REVISION, trust_remote_code=True, dtype=dtype
    ).to(device()).eval()
    return tok, model


def chat_ids(tok, user_text, dev):
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )
    return tok(text, add_special_tokens=False, return_tensors="pt").to(dev)


def gen(model, tok, inputs, max_new_tokens=64, **kw):
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, **kw)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    dt = time.time() - t0
    text = tok.decode(out.sequences[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text, dt, out.stats[0]
