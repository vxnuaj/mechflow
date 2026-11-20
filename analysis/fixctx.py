"""Prereg §2.14 (R-FC, robustness of H2-N to a fixed left context; written before its data).
Usage: python analysis/fixctx.py RESULTS_DIR   (holds conf_h2n_fx in the conf_h2n schema)
Uses analysis/confirm.py's frozen h2n_units / summarize (cluster bootstrap over documents, B as in confirm.py).
R-FC holds iff, on dep_23 under the fixed context, the contrast N − 0.2T has a one-sided 95% lower bound > 0 AND the
Shapley N share ≥ 0.5. Otherwise "NOT ROBUST" (reported as a limitation of the H2-N headline). coh is descriptive."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirm as C
C.ROOT = sys.argv[1]
rng = np.random.default_rng(20261113)
H = C.h2n_units("conf_h2n_fx"); r = C.summarize(H["c"], H["k"], 0.0, rng, bca=False); shN = H["shN"].sum() / H["T"].sum()
Hc = C.h2n_units("conf_h2n_fx", "coh"); rc = C.summarize(Hc["c"], Hc["k"], 0.0, rng, bca=False); shNc = Hc["shN"].sum() / Hc["T"].sum()
print(f"fixed-context dep_23: contrast N − 0.2T = {C.fmt(r)}; Shapley N share = {shN:.3f}  [docs {H['n']}, excluded {H['excl']}]")
print(f"fixed-context coh_23 (descriptive): contrast = {C.fmt(rc)}; Shapley N share = {shNc:.3f}")
ok = r["lb1s"] > 0 and shN >= 0.5
print("R-FC:", "ROBUST" if ok else "NOT ROBUST")
