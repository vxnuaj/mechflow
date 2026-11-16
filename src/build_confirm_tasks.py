"""Build the confirmatory task files (prereg §4, §8, §11). Confirmation split only; circuit-validation untouched.
Natural docs are shuffled once with seed 20261113; each test takes a prefix of the same order.
  P2 (+ judge)  : 7,100 docs  (required n, pilot SD 2.74, §8 rule)
  S3/S4 arms    : first 2,000 of those docs (same task_ids → same cuts and samples as P2)
  H2-N (MR)     : first 2,000 docs
  S12 (H2-N PR) : first 750 docs (80% power at half the pilot effect)
  P1 (0.5B)     : all 9,632 confirmation prompts (602 pairs)
  S1 (7B P1)    : all prompts of 150 random confirmation pairs
"""
import json, random, hashlib

D = "../data/"
docs = [json.loads(l) for l in open(D + "natural_docs.jsonl")]
conf = sorted([d for d in docs if d["split"] == "confirmation"], key=lambda d: d["doc_id"])
random.Random(20261113).shuffle(conf)
nat = [{"task_id": 100000 + i, "doc_id": d["doc_id"], "user": d["user"], "response": d["response"]} for i, d in enumerate(conf)]
bank = [json.loads(l) for l in open(D + "bank.jsonl")]
cb = [b for b in bank if b["split"] == "confirmation"]
keep = ("prompt", "tok_a", "tok_b", "mode_a", "mode_b", "pair_id", "category", "template_id", "order")
p1 = [dict(task_id=b["id"], **{k: b[k] for k in keep}) for b in cb]
pairs = sorted({b["pair_id"] for b in cb}); s1p = set(random.Random(20261114).sample(pairs, 150))
files = {"tasks_conf_p2.jsonl": nat[:7100], "tasks_conf_p2x.jsonl": nat[:2000], "tasks_conf_h2n.jsonl": nat[:2000],
         "tasks_conf_s12.jsonl": nat[:750], "tasks_conf_p1.jsonl": p1, "tasks_conf_s1_7b.jsonl": [t for t in p1 if t["pair_id"] in s1p]}
for f, rows in files.items():
    txt = "".join(json.dumps(r) + "\n" for r in rows)
    open(f, "w").write(txt)
    print(f, len(rows), hashlib.sha256(txt.encode()).hexdigest()[:16])
