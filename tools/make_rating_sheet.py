"""Build the B1 human-rating sheet (prereg §2.16). Run on CARC (mechflow env; Lafa tokenizer from the HF cache).
Writes ratings/sheet.md (for raters) and ratings/key.json (hidden: side of SC-on, LM scores)."""
import hashlib, json, glob, os, random, sys
sys.path.insert(0, "src")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
import load
tok = AutoTokenizer.from_pretrained(load.REPO, revision=load.REVISION, trust_remote_code=True)
tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open("src/tasks_conf_h2n.jsonl")}
res = {}
for f in glob.glob("src/results/conf_h2n/shard_*.jsonl"):
    for l in open(f): r = json.loads(l); res[r["task_id"]] = r["result"]
order = sorted(res, key=lambda t: hashlib.sha256(str(t).encode()).hexdigest())
rng = random.Random(20261121); items, key = [], []
for t in order:
    r = res[t]; c = r["cells"]; on, off = c["j2_O1N1F1"]["y"], c["j2_O0N0F0"]["y"]
    if on == off: continue
    task = tasks[t]; p = load.chat_prompt_ids(tok, task["user"]); rid = tok(task["response"], add_special_tokens=False).input_ids
    prefix = p + rid[:r["cut"]]; ctx = tok.decode(prefix[-60:], skip_special_tokens=True)
    lm_on = c["j2_O1N1F1"]["coh"] + c["j3_O1N1F1"]["coh"]; lm_off = c["j2_O0N0F0"]["coh"] + c["j3_O0N0F0"]["coh"]
    a_is_on = rng.random() < 0.5
    A, B = (on, off) if a_is_on else (off, on)
    items.append((ctx, tok.decode(A), tok.decode(B)))
    key.append({"item": len(items), "task_id": t, "A_is_sc_on": a_is_on, "lm_on": lm_on, "lm_off": lm_off})
    if len(items) == 100: break
os.makedirs("ratings", exist_ok=True)
with open("ratings/sheet.md", "w") as f:
    f.write("# Coherence ratings (blind)\n\nFor each item, read the context, then the two continuations. Which continuation fits the context more naturally and coherently? Write **A**, **B**, or **T** (tie / can't tell) after `Answer:`. Don't overthink; ~15 s per item. Continuations are 4 tokens, so they may start or end mid-word.\n\n")
    for i, (ctx, a, b) in enumerate(items, 1):
        f.write(f"## {i}\n\n**Context:** …{ctx}\n\n- **A:** `{a}`\n- **B:** `{b}`\n\nAnswer: \n\n")
json.dump(key, open("ratings/key.json", "w"), indent=0); print("items", len(items))
