"""Run E4 steering over exploration bimodal prompts (see insights §15). Resumable JSONL output."""
import json, os, sys
import load
from lafa_instr import Sampler, BlockCfg
from steer import run_logged, run_transplant
OUT = "results/steer/steer.jsonl"; os.makedirs("results/steer", exist_ok=True)
done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
tok, m = load.load_model(); S = Sampler(m)
bank = [json.loads(l) for l in open("../data/bank.jsonl") if json.loads(l)["split"] == "exploration"]
bank = [b for b in bank if b.get("tok_a") and b.get("tok_b") and len(b["tok_a"]) >= 2 and len(b["tok_b"]) >= 2]
W = {"all": None, "late": set(range(24, 32)), "early": set(range(0, 16))}
n = len(done)
with open(OUT, "a") as f:
    for b in bank:
        if n >= 200: break
        if b["id"] in done: continue
        ids = load.chat_prompt_ids(tok, b["prompt"]); cache, L = S.prefill(ids)
        runs = {}
        for s in range(1, 13):
            y, log = run_logged(S, cache, L, BlockCfg(32, 1.0, 4.0, True, seed=s))
            lab = "A" if y[:2] == b["tok_a"][:2] else ("B" if y[:2] == b["tok_b"][:2] else "-")
            runs[s] = (y, log, lab)
        rec = None
        for X, Y in (("A", "B"), ("B", "A")):
            xs = [s for s in runs if runs[s][2] == X]; ys = [s for s in runs if runs[s][2] == Y]
            if len(xs) >= 2 and ys:
                rec = (X, Y, xs[0], ys[0], xs[1]); break
        if rec is None:
            continue
        X, Y, r, dy, dx = rec
        tx = b["tok_a"] if X == "A" else b["tok_b"]; ty = b["tok_b"] if X == "A" else b["tok_a"]
        cfg = BlockCfg(32, 1.0, 4.0, True, seed=r)
        res = {"id": b["id"], "pair_id": b["pair_id"], "X": X, "rec": r, "y_donor": dy, "x_donor": dx,
               "tx": tx[:2], "ty": ty[:2], "base": run_transplant(S, cache, L, cfg, 1, None)[0]}
        for wn, w in W.items():
            res[f"Y_{wn}"] = run_transplant(S, cache, L, cfg, 1, runs[dy][1], steps=w)[0]
            res[f"X_{wn}"] = run_transplant(S, cache, L, cfg, 1, runs[dx][1], steps=w)[0]
        f.write(json.dumps(res) + "\n"); f.flush(); n += 1
        print(n, b["mode_a"], "/", b["mode_b"], "base", repr(tok.decode(res["base"][:2])), "Y_all", repr(tok.decode(res["Y_all"][:2])), "X_all", repr(tok.decode(res["X_all"][:2])), flush=True)
