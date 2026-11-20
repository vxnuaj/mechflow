"""Prereg §2.17 (Phase 14.2 message-content probe; written before the full data).
Usage: python analysis/probe.py <root>   (root/probe_warp/h/*.npz, root/probe_uniform/h/*.npz; Lafa E loaded via load.py)

Question: does the neighbours' fed-back prediction put information about *what the neighbours will finally write* into
the target position's state, specifically in the window where neighbour reading matters causally (E1, E1c)?

Data: per doc and target j ∈ {2, 3}, the residual state of j at layers {3, 8, 15, 19}, every step k, in pass A (j reads the
neighbours with feedback; the trajectory) and pass B (the same states, the neighbours without feedback = N-cut).
Split: tasks_h2n_expl_300 docs, sorted by SHA-256(task_id): first 200 = probe training, last 100 = test (by task, both grids).
Probe: per layer, a ridge map W: h → E[y_i] (embedding of neighbour i's FINAL token), fit on pass-A states from the
training docs pooled over steps, targets and neighbours (λ = 10, standardised h; fixed a priori). The same W is applied to
pass A and pass B test states, so the readout is identical.
Metric: top-10 accuracy of the true neighbour token among C = 2048 candidates (the true token + 2047 fixed random
vocabulary tokens, seed 20261121) by cosine with W·h; averaged over the 3 neighbours, 2 targets and 4 layers (no layer
selection). gap(k) = acc_A(k) − acc_B(k): what the neighbours' feedback adds to j's knowledge of their final tokens.
Bootstrap over test docs (10,000; seed 20261121).
Predictions:
  Q1 (warp)    G_late − G_early > 0, where G_late = mean gap over steps 24–31 and G_early = over steps 0–15; and G_late > 0.
  Q2 (uniform) G_mid − G_late > 0, where G_mid = mean gap over steps 8–15 (the content gap moves earlier with the schedule, as
               the causal N effect does in E1c).
Each holds iff the 95% CI lower bound > 0 (for Q1, both parts); FAILS iff the CI upper bound < 0 (either part).
Decision: SUPPORTED iff Q1 and Q2 hold; REFUTED iff Q1 or Q2 FAILS; else INCONCLUSIVE.
Validity (else NOT TESTABLE): identity true on every doc; probe test accuracy acc_A at the last step ≥ 0.2 (the probe
reads final-token content at all)."""
import glob, hashlib, json, os, sys
import numpy as np


def split(ids):
    o = sorted(ids, key=lambda t: hashlib.sha256(str(t).encode()).hexdigest()); return set(o[:200]), set(o[200:])


def load_grid(d):
    D = {}
    for f in glob.glob(f"{d}/h/*.npz"):
        z = np.load(f); D[int(os.path.basename(f)[:-4])] = (z["hA"].astype(np.float32), z["hB"].astype(np.float32), z["y"])
    return D


def fit_and_score(D, train, test, E, cand):
    """Returns accA, accB [n_test_docs, K]."""
    nL = next(iter(D.values()))[0].shape[2]; K = next(iter(D.values()))[0].shape[1]
    accA = np.zeros((len(test), K)); accB = np.zeros((len(test), K)); tl = sorted(test)
    for l in range(nL):
        X, Y = [], []
        for t in train:
            hA, _, y = D[t]
            for ji, j in enumerate((2, 3)):
                for i in range(4):
                    if i == j: continue
                    X.append(hA[ji, :, l]); Y.append(np.repeat(E[y[i]][None], K, 0))
        X = np.concatenate(X); Y = np.concatenate(Y); mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
        W = np.linalg.solve(Xs.T @ Xs + 10.0 * np.eye(Xs.shape[1]), Xs.T @ Y)
        for di, t in enumerate(tl):
            hA, hB, y = D[t]
            for ji, j in enumerate((2, 3)):
                for i in range(4):
                    if i == j: continue
                    C = np.concatenate([[y[i]], cand[cand != y[i]][:len(cand) - 1]]); Ec = E[C]; Ec = Ec / np.linalg.norm(Ec, axis=1, keepdims=True)
                    for H, acc in ((hA, accA), (hB, accB)):
                        P = ((H[ji, :, l] - mu) / sd) @ W; P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
                        S = P @ Ec.T; rank = (S > S[:, :1]).sum(1)
                        acc[di] += (rank < 10) / (2 * 3 * nL)
    return accA, accB


def decide(gw, gu, rng):
    """gw, gu: gap arrays [n_docs, K] for warp and uniform."""
    n = gw.shape[0]; idx = rng.integers(0, n, (10000, n))
    late = lambda g: g[:, 24:32].mean(1); early = lambda g: g[:, 0:16].mean(1); mid = lambda g: g[:, 8:16].mean(1)
    q1a = late(gw) - early(gw); q1b = late(gw); q2 = mid(gu) - late(gu)
    ci = lambda v: np.percentile(v[idx].mean(1), [2.5, 97.5])
    r = {k: (v.mean(), ci(v)) for k, v in (("Q1 late−early (warp)", q1a), ("Q1 late>0 (warp)", q1b), ("Q2 mid−late (uniform)", q2))}
    for k, (m, c) in r.items(): print(f"{k:24} {m:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]")
    q1 = "holds" if (r["Q1 late−early (warp)"][1][0] > 0 and r["Q1 late>0 (warp)"][1][0] > 0) else "FAILS" if (r["Q1 late−early (warp)"][1][1] < 0 or r["Q1 late>0 (warp)"][1][1] < 0) else "undetermined"
    q2 = "holds" if r["Q2 mid−late (uniform)"][1][0] > 0 else "FAILS" if r["Q2 mid−late (uniform)"][1][1] < 0 else "undetermined"
    return q1, q2


def main(root):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import load, torch
    tok, model = load.load_model(dtype=torch.float32); E = model.embed_tokens.weight.detach().float().numpy() if hasattr(model, "embed_tokens") else model.get_input_embeddings().weight.detach().float().numpy()
    cand = np.random.default_rng(20261121).choice(E.shape[0], 2048, replace=False)
    ident = all(json.loads(l)["result"]["identity"] for g in ("warp", "uniform") for f in glob.glob(f"{root}/probe_{g}/shard_*.jsonl") for l in open(f))
    gaps = {}; last_acc = {}
    for g in ("warp", "uniform"):
        D = load_grid(f"{root}/probe_{g}"); tr, te = split(D.keys()); aA, aB = fit_and_score(D, tr, te, E, cand)
        gaps[g] = aA - aB; last_acc[g] = aA[:, -1].mean()
        print(f"[{g}] test docs {len(te)}; acc_A by step: " + " ".join(f"{x:.2f}" for x in aA.mean(0)))
        print(f"[{g}] gap A−B by step:  " + " ".join(f"{x:+.3f}" for x in gaps[g].mean(0)))
    valid = ident and min(last_acc.values()) >= 0.2
    q1, q2 = decide(gaps["warp"], gaps["uniform"], np.random.default_rng(20261121))
    print(f"validity: identity {ident}, last-step acc_A {last_acc} → {'ok' if valid else 'NOT TESTABLE'}")
    dec = "NOT TESTABLE" if not valid else "REFUTED" if "FAILS" in (q1, q2) else "SUPPORTED" if q1 == q2 == "holds" else "INCONCLUSIVE"
    print(f"Q1 {q1}  Q2 {q2}\nPROBE DECISION: {dec}")


if __name__ == "__main__":
    main(sys.argv[1])
