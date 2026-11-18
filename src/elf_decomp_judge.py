"""Score ELF 2x2x2 outputs (mechflow env): for each target j, cell and sample, Mistral log-prob of the decoded text
span of token j given the decoded tokens 0..j-1 (plain text, no chat template). T5 tokenizer for decoding."""
import json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
src = sys.argv[1]; R = json.load(open(src))
t5 = AutoTokenizer.from_pretrained(os.environ.get("T5_PATH", "t5-small"))
jp = os.environ["JUDGE_PATH"]; jt = AutoTokenizer.from_pretrained(jp)
jm = AutoModelForCausalLM.from_pretrained(jp, dtype=torch.bfloat16).to("cuda").eval()
@torch.no_grad()
def span_lp(prefix, piece):
    full = prefix + piece
    enc = jt(full, return_offsets_mapping=True, add_special_tokens=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if not piece.strip(): return None
    x = torch.tensor([ids], device="cuda"); lp = torch.log_softmax(jm(input_ids=x).logits[0].float(), -1)
    a, b = len(prefix), len(full); s = 0.0; hit = False
    for t in range(1, len(ids)):
        if a <= offs[t][0] < b: s += float(lp[t - 1, ids[t]]); hit = True
    return s if hit else None
out = {}
for j, cells in R.items():
    j_ = int(j); out[j] = {}
    for cell, rows in cells.items():
        vals = []
        for ids in rows:
            pre = t5.decode(ids[:j_], skip_special_tokens=True); full = t5.decode(ids[:j_ + 1], skip_special_tokens=True)
            vals.append(span_lp(pre, full[len(pre):]))
        out[j][cell] = vals
        print(j, cell, sum(v for v in vals if v is not None) / max(1, sum(v is not None for v in vals)), flush=True)
json.dump(out, open(src.replace(".json", "_judged.json"), "w"))
