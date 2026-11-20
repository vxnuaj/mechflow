"""Phase 14.3 tasks: H2-N on math (GSM8K train: question → worked answer) and code (MBPP full train+validation+test:
text → code). Disjoint random splits (seed 20261121): exploration 100 + confirmation 500 per type.
Task ids: math explore 700000+, code explore 710000+, math confirm 720000+, code confirm 730000+.
Writes src/tasks_tt_{math,code}_{explore,confirm}.jsonl (fields: task_id, doc_id, user, response)."""
import hashlib, json, random
from datasets import load_dataset
rng = random.Random(20261121)
gsm = [(r["question"], r["answer"].split("####")[0].strip()) for r in load_dataset("openai/gsm8k", "main", split="train")]
mb = []
for sp in ("train", "validation", "test"):
    mb += [(r["text"], r["code"]) for r in load_dataset("google-research-datasets/mbpp", "full", split=sp)]
for name, rows, b_ex, b_cf in (("math", gsm, 700000, 720000), ("code", mb, 710000, 730000)):
    rows = [r for r in rows if len(r[1]) > 80]; rng.shuffle(rows)
    for split, part, base in (("explore", rows[:100], b_ex), ("confirm", rows[100:600], b_cf)):
        with open(f"src/tasks_tt_{name}_{split}.jsonl", "w") as f:
            for i, (u, a) in enumerate(part):
                f.write(json.dumps({"task_id": base + i, "doc_id": hashlib.sha256((u + a).encode()).hexdigest(), "user": u, "response": a}) + "\n")
        print(name, split, len(part))
