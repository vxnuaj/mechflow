"""Phase 3.5 smoke test: do cross-mode hybrids ("New Francisco") occur, and does SC change their rate?

Prompts ask for a two-word entity whose words recombine into invalid hybrids. The first
generated block (4 tokens) contains the entity. Each output is scored at word level:
  valid  = (w1, w2) is a real entity of the category
  hybrid = w1 is a first word and w2 a second word of *different* entities in the category
  other  = anything else
Exploratory only.

Usage: python 03_hybrid.py [out.jsonl]
"""
import itertools
import json
import re
import sys
import time

import torch

from common import load, device, chat_ids

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/hybrid.jsonl"

CATS = {
    "US city": ["New York", "Los Angeles", "San Francisco", "San Diego", "Las Vegas", "New Orleans",
                "San Antonio", "San Jose", "Santa Fe", "Kansas City"],
    "country": ["New Zealand", "South Africa", "South Korea", "North Korea", "Saudi Arabia",
                "Costa Rica", "Sri Lanka", "El Salvador", "United States", "United Kingdom"],
    "scientist": ["Albert Einstein", "Isaac Newton", "Marie Curie", "Charles Darwin", "Niels Bohr",
                  "Alan Turing", "Ada Lovelace", "Galileo Galilei"],
}
OPEN_TEMPLATES = [
    "Name one {cat} with a two-word name. Reply with only the name.",
    "Give me a random example of a {cat} whose name has two words. Just the name, nothing else.",
    "Quick: a {cat} with a two-word name? Answer with the name only.",
]
PAIR_TEMPLATES = [
    "Pick one of these at random: {a} or {b}. Reply with only your choice.",
    "Choose either {a} or {b}. Answer with just the name.",
]
PAIRS = {
    "US city": [("New York", "San Francisco"), ("Los Angeles", "San Diego"), ("Las Vegas", "New Orleans"),
                ("San Antonio", "Kansas City")],
    "country": [("New Zealand", "South Africa"), ("South Korea", "Saudi Arabia"), ("Costa Rica", "Sri Lanka"),
                ("United States", "El Salvador")],
    "scientist": [("Albert Einstein", "Isaac Newton"), ("Marie Curie", "Charles Darwin"),
                  ("Alan Turing", "Ada Lovelace")],
}

CONFIGS = {}
for K, sc, g in itertools.product([1, 4, 32], [True, False], [1.0, 8.0]):
    CONFIGS[f"K{K}_g{int(g)}_n0.1_{'SC' if sc else 'noSC'}"] = (
        dict(gen_mode="flow", num_steps=K, cfg_scale=g, noise_scale_override=0.1, self_cond=sc), [42])
for sc in [True, False]:
    CONFIGS[f"K32_g8_n4.0_{'SC' if sc else 'noSC'}"] = (
        dict(gen_mode="flow", num_steps=32, cfg_scale=8.0, noise_scale_override=4.0, self_cond=sc), [1, 2, 3, 4])
CONFIGS["AR"] = (dict(gen_mode="ar"), [42])


def words(text):
    return [re.sub(r"[^\w]", "", w) for w in text.strip().split()][:2]


def classify(text, cat):
    ents = [tuple(e.split()) for e in CATS[cat]]
    w = tuple(words(text))
    if len(w) < 2:
        return "other"
    if w in ents:
        return "valid"
    firsts = {e[0] for e in ents}
    seconds = {e[1] for e in ents}
    if w[0] in firsts and w[1] in seconds:
        return "hybrid"
    return "other"


def prompts():
    for cat in CATS:
        for t in OPEN_TEMPLATES:
            yield cat, "open", t.format(cat=cat)
        for a, b in PAIRS[cat]:
            for x, y in [(a, b), (b, a)]:
                for t in PAIR_TEMPLATES:
                    yield cat, "pair", t.format(a=x, b=y)


def main():
    tok, model = load(torch.bfloat16)
    dev = device()
    t0 = time.time()
    with open(OUT, "w") as f:
        for n, (cat, kind, text) in enumerate(prompts()):
            inputs = chat_ids(tok, text, dev)
            rec = {"cat": cat, "kind": kind, "prompt": text, "out": {}}
            for name, (kw, seeds) in CONFIGS.items():
                outs = []
                for s in seeds:
                    o = model.generate(**inputs, max_new_tokens=4, seed=s, **kw)
                    g = tok.decode(o[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    outs.append({"text": g, "label": classify(g, cat)})
                rec["out"][name] = outs
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{n} {time.time() - t0:.0f}s {text[:50]!r} -> paper:{rec['out']['K32_g8_n0.1_SC'][0]['text']!r} "
                  f"K1noSC:{rec['out']['K1_g8_n0.1_noSC'][0]['text']!r}", flush=True)


if __name__ == "__main__":
    main()
