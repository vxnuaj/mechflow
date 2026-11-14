"""K3 null-calibrated Mahalanobis analysis (prereg §12, 2026-11-13 clarification).

Reads the states npz from `k3_drift.py --states-only`. For each (layer, step, position) cell:
  - fit reference distributions on half of "on" and half of "off" states (PCA 95% var + Ledoit-Wolf),
  - threshold = 1.5 x max(95th pct of each reference's held-out distances),
  - score conditions: on2 (in-distribution null: full-on, different noise seed), a, b, c.
Per condition: pooled ratio = 95th pct over all cells of (distance / cell threshold);
               exceed frac  = fraction of cells whose 95th-pct ratio > 1.
Rule: pass(i) iff pooled <= 1.0 and exceed_frac <= null_exceed_frac + 0.05.
"""
import json
import sys

import numpy as np
from sklearn.covariance import LedoitWolf

npz = sys.argv[1] if len(sys.argv) > 1 else "results/k3_calib_states.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "results/k3_calib.json"
D = np.load(npz)
data = {k: D[k] for k in D.files}
cells = sorted({(int(r[0]), int(r[1]), int(r[2])) for r in data["on"]})


def get(k, c):
    A = data[k]
    m = (A[:, 0] == c[0]) & (A[:, 1] == c[1]) & (A[:, 2] == c[2])
    return A[m, 3:]


per_cell = {k: [] for k in ("on2", "a", "b", "c")}
pooled = {k: [] for k in per_cell}
for c in cells:
    refs = {}
    for nm in ("on", "off"):
        X = get(nm, c)
        if len(X) < 20:
            break
        half = len(X) // 2
        Xf, Xh = X[:half], X[half:]
        mu = Xf.mean(0)
        _, Sg, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(Sg ** 2) / np.sum(Sg ** 2), 0.95) + 1)
        P = Vt[:k].T
        prec = LedoitWolf().fit((Xf - mu) @ P).precision_

        def md(Y, mu=mu, P=P, prec=prec):
            Z = (Y - mu) @ P
            return np.sqrt(np.einsum("ij,jk,ik->i", Z, prec, Z))
        refs[nm] = (md, np.percentile(md(Xh), 95))
    if len(refs) < 2:
        continue
    thr = 1.5 * max(refs["on"][1], refs["off"][1])
    for k in per_cell:
        Y = get(k, c)
        if len(Y) == 0:
            continue
        d = np.minimum(refs["on"][0](Y), refs["off"][0](Y)) / thr
        per_cell[k].append(float(np.percentile(d, 95)))
        pooled[k].extend(d.tolist())

res = {}
for k in per_cell:
    v = np.array(per_cell[k])
    res[k] = {"n_cells": len(v), "exceed_frac": float(np.mean(v > 1)) if len(v) else None,
              "cell_ratio_median": float(np.median(v)) if len(v) else None,
              "cell_ratio_max": float(v.max()) if len(v) else None,
              "pooled_ratio": float(np.percentile(pooled[k], 95)) if pooled[k] else None}
null = res["on2"]["exceed_frac"]
for k in ("a", "b", "c"):
    res[k]["pass_i"] = bool(res[k]["pooled_ratio"] <= 1.0 and res[k]["exceed_frac"] <= null + 0.05)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
