"""Model loading for confirmatory code (pinned snapshot)."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.environ.get("LAFA_REPO", "lafa-hf/lafa-0.5B")
REVISION = os.environ.get("LAFA_REV", "18deca3522f7884898a58d0fb2d2afb3fc91f87e")


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, trust_remote_code=True,
                                                 dtype=dtype).to(device()).eval()
    return tok, model


def chat_prompt_ids(tok, user_text):
    text = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids
