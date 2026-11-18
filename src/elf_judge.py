"""Score ELF step-0 texts with Mistral-7B-Instruct-v0.3 (mechflow env). usage: python elf_judge.py results/elf_sc_check_texts.json"""
import json, math, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
texts = json.load(open(sys.argv[1]))
jp = snapshot_download("mistralai/Mistral-7B-Instruct-v0.3", revision="c170c708c41dac9275d15a8fff4eca08d52bab71")
jt = AutoTokenizer.from_pretrained(jp); jm = AutoModelForCausalLM.from_pretrained(jp, dtype=torch.bfloat16).to("cuda").eval()
res = {}
for sc, ts in texts.items():
    lps, reps, d2 = [], [], []
    for txt in ts:
        ids = jt(txt, return_tensors="pt").input_ids[:, :256].to("cuda")
        if ids.shape[1] < 3: continue
        with torch.no_grad():
            lp = torch.log_softmax(jm(ids).logits[0, :-1].float(), -1).gather(1, ids[0, 1:, None])[:, 0]
        lps.append(float(lp.mean()))
        w = txt.split(); reps.append(sum(w[i] == w[i - 1] for i in range(1, len(w))) / max(1, len(w) - 1))
        bg = list(zip(w, w[1:])); d2.append(len(set(bg)) / max(1, len(bg)))
    m = lambda x: sum(x) / max(1, len(x)); sd = lambda x: (sum((v - m(x)) ** 2 for v in x) / max(1, len(x) - 1)) ** .5
    res[sc] = {"n": len(lps), "judge_lp_mean": m(lps), "judge_lp_se": sd(lps) / math.sqrt(max(1, len(lps))),
               "immediate_word_repeat": m(reps), "distinct2": m(d2), "lps": lps}
    print(sc, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[sc].items() if k != "lps"}, flush=True)
json.dump(res, open(sys.argv[1].replace("_texts.json", "_judged.json"), "w"), indent=1)
