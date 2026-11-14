"""Build the bimodal prompt bank and its splits (prereg §4.1). Deterministic (seed 20261113).

Output: data/bank.jsonl (one prompt per line) + data/bank_manifest.json (counts, hashes).
Each record: id, split (exploration | confirmation | circuit_validation), category, pair_id,
template_kind (open | pair), template_id, order, prompt, mode_a, mode_b (pair only),
tok_a, tok_b (in-context answer-token ids), D (distinguishing positions within the first block).
Pair filter: keep only pairs whose in-context tokenizations differ within the first block (D non-empty).
Usage: python -m data.build_bank  (run from src/)
"""
import hashlib
import itertools
import json
import os
import random

from data.entities import (CATEGORIES, EXPLORATION, CONFIRMATION, OPEN_TEMPLATES_EXPL,
                           PAIR_TEMPLATES_EXPL, PAIR_TEMPLATES_CONF)

SEED = 20261113
PAIRS_PER_CAT = 80
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def main():
    from load import REPO, REVISION
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)

    def ans_ids(prompt, ans):
        base = chat(prompt)
        p = tok(base, add_special_tokens=False).input_ids
        pa = tok(base + ans, add_special_tokens=False).input_ids
        assert pa[:len(p)] == p
        return pa[len(p):]

    rng = random.Random(SEED)
    recs, drops = [], {}
    for cat, (noun, ents) in CATEGORIES.items():
        assert all(len(e.split()) == 2 for e in ents), cat
        assert len(set(ents)) == len(ents), cat
        is_expl = cat in EXPLORATION
        pairs = list(itertools.combinations(ents, 2))
        rng.shuffle(pairs)
        templates = PAIR_TEMPLATES_EXPL if is_expl else PAIR_TEMPLATES_CONF
        kept = 0
        drops[cat] = 0
        for a, b in pairs:
            if kept >= PAIRS_PER_CAT:
                break
            # tokenization filter on the first template, both orders
            probe = templates[0].format(a=a, b=b)
            ta, tb = ans_ids(probe, a), ans_ids(probe, b)
            D = [j for j in range(min(4, len(ta), len(tb))) if ta[j] != tb[j]]
            if not D:
                drops[cat] += 1
                continue
            pair_id = f"{cat}:{kept:03d}"
            kept += 1
            split = "exploration" if is_expl else ("circuit_validation" if rng.random() < 0.15 else "confirmation")
            for ti, tmpl in enumerate(templates):
                for order, (x, y) in enumerate([(a, b), (b, a)]):
                    prompt = tmpl.format(a=x, b=y)
                    tx, ty = ans_ids(prompt, a), ans_ids(prompt, b)
                    Dp = [j for j in range(min(4, len(tx), len(ty))) if tx[j] != ty[j]]
                    recs.append({"split": split, "category": cat, "pair_id": pair_id,
                                 "template_kind": "pair", "template_id": ti, "order": order,
                                 "prompt": prompt, "mode_a": a, "mode_b": b,
                                 "tok_a": tx, "tok_b": ty, "D": Dp})
        if is_expl:
            for ti, tmpl in enumerate(OPEN_TEMPLATES_EXPL):
                recs.append({"split": "exploration", "category": cat, "pair_id": None,
                             "template_kind": "open", "template_id": ti, "order": 0,
                             "prompt": tmpl.format(cat=noun), "mode_a": None, "mode_b": None,
                             "tok_a": None, "tok_b": None, "D": None})
    for i, r in enumerate(recs):
        r["id"] = i
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "bank.jsonl")
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    split_ids = {s: sorted({r["pair_id"] for r in recs if r["split"] == s and r["pair_id"]}) for s in
                 ("exploration", "confirmation", "circuit_validation")}
    hs = hashlib.sha256(json.dumps(split_ids, sort_keys=True).encode()).hexdigest()
    counts = {s: {"prompts": sum(r["split"] == s for r in recs), "pairs": len(v)} for s, v in split_ids.items()}
    man = {"seed": SEED, "bank_sha256": h, "split_membership_sha256": hs, "counts": counts,
           "tokenization_drops_per_category": drops,
           "per_category_pairs": {c: len({r["pair_id"] for r in recs if r["category"] == c and r["pair_id"]})
                                  for c in CATEGORIES}}
    json.dump(man, open(os.path.join(OUT_DIR, "bank_manifest.json"), "w"), indent=1)
    json.dump(split_ids, open(os.path.join(OUT_DIR, "bank_splits.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
