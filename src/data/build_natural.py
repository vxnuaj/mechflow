"""Natural-text documents for block-level measures (prereg §4.2, amended §11: UltraChat only).

Source: HuggingFaceH4/ultrachat_200k, split test_sft; document = conversation; text = first user
turn (prompt) + first assistant turn (response). Keep responses with >= 24 tokens.
Split by document (seed 20261113): 20% exploration, 80% confirmation, of which 10% reserved as
circuit-validation. Output: data/natural_docs.jsonl + data/natural_manifest.json (hashes).
Usage: python -m data.build_natural   (from src/)
"""
import hashlib
import json
import os
import random

from datasets import load_dataset

SEED = 20261113
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def main():
    from load import REPO, REVISION
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    rng = random.Random(SEED)
    recs = []
    for ex in ds:
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        resp = msgs[1]["content"].strip()
        if len(tok(resp, add_special_tokens=False).input_ids) < 24:
            continue
        u = rng.random()
        split = "exploration" if u < 0.20 else ("circuit_validation" if u < 0.28 else "confirmation")
        recs.append({"doc_id": ex["prompt_id"], "split": split, "user": msgs[0]["content"], "response": resp})
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "natural_docs.jsonl")
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    ids = {s: sorted(r["doc_id"] for r in recs if r["split"] == s) for s in ("exploration", "confirmation", "circuit_validation")}
    man = {"source": "HuggingFaceH4/ultrachat_200k:test_sft", "seed": SEED,
           "docs_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
           "split_ids_sha256": hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
           "counts": {s: len(v) for s, v in ids.items()}}
    json.dump(man, open(os.path.join(OUT, "natural_manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
