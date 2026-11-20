"""Exploratory sensitivity analysis for family E (insights §20.1): score each cell's token at target j against a
FIXED left context, the (O1N1F1) cell's decoded tokens 0..j-1 for the same sample. Only j's own choice varies
across cells. Mistral-7B-Instruct-v0.3 span log-prob, as in elf_decomp_judge.py.
E4 v2: if REF_FILE is set ({"tokens": [[ids] per sample]}), that common reference context is used for every cell and
config (the base ELF-B SC-on tokens for the same noise), so contexts do not differ between access conditions."""
import json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
DEV = os.environ.get("JUDGE_DEVICE", "mps")
R = json.load(open(sys.argv[1]))
t5 = AutoTokenizer.from_pretrained(os.environ.get("T5_PATH", "t5-small"))
jp = os.environ["JUDGE_PATH"]; jt = AutoTokenizer.from_pretrained(jp)
jm = AutoModelForCausalLM.from_pretrained(jp, dtype=getattr(torch, os.environ.get("JUDGE_DTYPE", "bfloat16"))).to(DEV).eval()
if DEV == "cpu": torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
@torch.no_grad()
def span_lp(prefix, piece):
    if not piece.strip(): return None
    full = prefix + piece; enc = jt(full, return_offsets_mapping=True, add_special_tokens=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    lp = torch.log_softmax(jm(input_ids=torch.tensor([ids], device=DEV)).logits[0].float(), -1)
    a, b = len(prefix), len(full); s, hit = 0.0, False
    for t in range(1, len(ids)):
        if a <= offs[t][0] < b: s += float(lp[t - 1, ids[t]]); hit = True
    return s if hit else None
REF = json.load(open(os.environ["REF_FILE"]))["tokens"] if os.environ.get("REF_FILE") else None   # E4 v2 (red-team C4)
out = {}
for j, cells in R.items():
    j_ = int(j); ref = REF if REF is not None else cells["O1N1F1"]; out[j] = {}
    for cell, rows in cells.items():
        vals = []
        for i, ids in enumerate(rows):
            ctx = ref[i][:j_] + [ids[j_]]
            pre = t5.decode(ref[i][:j_], skip_special_tokens=True); full = t5.decode(ctx, skip_special_tokens=True)
            vals.append(span_lp(pre, full[len(pre):]))
        out[j][cell] = vals
        print(j, cell, sum(v for v in vals if v is not None) / max(1, sum(v is not None for v in vals)), flush=True)
json.dump(out, open(sys.argv[1].replace(".json", "_judged_fixedctx.json"), "w"))
