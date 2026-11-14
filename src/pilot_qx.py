"""Phase 7.3 pilot (exploration only): q–x window for K12 (prereg §5; phase4 §2.5, §5.3).

For every exploration pair prompt, run the MR regime (g=1, noise 4.0, K=32, SC on, seed=1) with
record=True. At each distinguishing position j (first block) whose final token is mode A or B,
store the state x_k^j and the fed-back expected embedding (q_{k-1}E)^j for every step k.
Then fit a held-out logistic probe (split BY PAIR) per step for x and for q:
window = steps with acc_q >= 0.85 and acc_x <= 0.65.
Usage: python pilot_qx.py [out_prefix]
"""
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import load
from lafa_instr import Sampler, BlockCfg, run_meta

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_qx"
CFG = BlockCfg(32, 1.0, 4.0, True, seed=1)
BANK = os.path.join(os.path.dirname(__file__), "..", "data", "bank.jsonl")


def collect():
    tok, model = load.load_model()
    S = Sampler(model)
    recs = [json.loads(l) for l in open(BANK)]
    recs = [r for r in recs if r["split"] == "exploration" and r["template_kind"] == "pair"]
    X, Q, Y, G, meta = [], [], [], [], []
    for n, r in enumerate(recs):
        ids = load.chat_prompt_ids(tok, r["prompt"])
        cache, L = S.prefill(ids)
        res = S.run_block(cache, L, CFG, record=True)
        for j in r["D"]:
            if j >= len(res.tokens):
                continue
            yj = 1 if res.tokens[j] == r["tok_a"][j] else (0 if res.tokens[j] == r["tok_b"][j] else None)
            if yj is None:
                continue
            X.append(np.stack([z[j].numpy() for z in res.rec["z"]]).astype(np.float16))
            Q.append(np.stack([e[j].numpy() for e in res.rec["esc_c"]]).astype(np.float16))
            Y.append(yj)
            G.append(r["pair_id"])
            meta.append({"id": r["id"], "j": j})
        if n % 100 == 0:
            print(n, len(recs), len(Y), flush=True)
    np.savez_compressed(OUT + "_states.npz", X=np.array(X), Q=np.array(Q), Y=np.array(Y), G=np.array(G))
    json.dump({"meta": run_meta(CFG), "n_instances": len(Y), "items": meta}, open(OUT + "_meta.json", "w"))


def probe():
    """Pair-relative linear probe. A raw-embedding probe can't generalize across pairs, because
    'mode A' is a different token in every pair. Features per instance and step:
    [cos(v,E_a) - cos(v,E_b), <v, E_a - E_b>/|E_a - E_b|, |v|], standardized; logistic regression
    trained on 75% of pairs, tested on held-out pairs."""
    D = np.load(OUT + "_states.npz", allow_pickle=True)
    X, Q, Y, G = D["X"].astype(np.float32), D["Q"].astype(np.float32), D["Y"], D["G"]
    items = json.load(open(OUT + "_meta.json"))["items"]
    bank = {}
    for line in open(BANK):
        r = json.loads(line)
        bank[r["id"]] = r
    tok, model = load.load_model()
    E = model.embed_tokens.weight.detach().float().cpu().numpy()
    ta = np.array([bank[it["id"]]["tok_a"][it["j"]] for it in items])
    tb = np.array([bank[it["id"]]["tok_b"][it["j"]] for it in items])
    Ea, Eb = E[ta], E[tb]
    Ean = Ea / np.linalg.norm(Ea, axis=1, keepdims=True)
    Ebn = Eb / np.linalg.norm(Eb, axis=1, keepdims=True)
    dvec = (Ea - Eb) / np.linalg.norm(Ea - Eb, axis=1, keepdims=True)

    def feats(V):
        nv = np.linalg.norm(V, axis=1, keepdims=True) + 1e-6
        Vn = V / nv
        return np.stack([(Vn * Ean).sum(1) - (Vn * Ebn).sum(1), (V * dvec).sum(1), nv[:, 0]], 1)

    pairs = sorted(set(G))
    rng = random.Random(7)
    rng.shuffle(pairs)
    test = set(pairs[: len(pairs) // 4])
    te = np.array([g in test for g in G])
    K = X.shape[1]
    res = {"acc_x": [], "acc_q": [], "acc_x_signonly": [], "acc_q_signonly": [],
           "n_train": int((~te).sum()), "n_test": int(te.sum()),
           "base_rate": float(max(Y[te].mean(), 1 - Y[te].mean()))}
    for k in range(K):
        for name, A in (("x", X), ("q", Q)):
            F = feats(A[:, k, :])
            mu, sd = F[~te].mean(0), F[~te].std(0) + 1e-6
            clf = LogisticRegression(C=1.0, max_iter=2000).fit((F[~te] - mu) / sd, Y[~te])
            res["acc_" + name].append(float(clf.score((F[te] - mu) / sd, Y[te])))
            res["acc_" + name + "_signonly"].append(float(np.mean((F[te, 0] > 0) == (Y[te] == 1))))
    win = [k for k in range(K) if res["acc_q"][k] >= 0.85 and res["acc_x"][k] <= 0.65]
    res["window_steps"] = win
    res["K12_triggered"] = len(win) < 3
    json.dump(res, open(OUT + "_probe.json", "w"), indent=1)
    print("base rate %.2f, n_test %d (held-out pairs)" % (res["base_rate"], res["n_test"]))
    print("k     :", " ".join(f"{k:4d}" for k in range(0, K, 2)))
    for nm in ("acc_q", "acc_x", "acc_q_signonly", "acc_x_signonly"):
        print(f"{nm:14s}:", " ".join(f"{res[nm][k]:.2f}" for k in range(0, K, 2)))
    print("window steps:", win, "| K12 triggered:", res["K12_triggered"])


if __name__ == "__main__":
    if not os.path.exists(OUT + "_states.npz"):
        collect()
    probe()
