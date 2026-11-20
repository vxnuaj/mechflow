# Prompt: red-team a pre-registered experiment before it runs

You are a skeptical reviewer for an interpretability study. You have no other context; everything you need is below. **Find flaws** in the design and decision rule *before the data exist*. Rank problems by severity (critical / major / minor), quote the line, and give a fix that can be made before running. Do not praise. Do not invent facts.

## Background
- **Self-conditioning (SC):** a diffusion/flow language model denoises all positions in parallel over steps. Each step, every position also receives its own previous clean-estimate x̂ as an extra input; training drops SC 50% of the time.
- **ELF** (Embedded Language Flows, arXiv 2605.10938) is a full-sequence flow over T5 encoder latents. Tokens are read out by a final "decode" pass of the same network, which **receives zero SC by design** (paper Alg. 5).
- **Causal decomposition (2×2×2):** for target position j, three routes are switched independently, with each route's pass recomputed:
  - **O:** j's own input carries its feedback;
  - **N:** j's attention reads the neighbours' keys/values from the pass where they carry feedback;
  - **F:** the neighbours' own trajectories and outputs evolve with feedback.

  We report absolute Shapley values of the outcome gain. The outcome is the log-prob of token j under Mistral-7B with a fixed left context (target-mediated).
- **Established findings:**
  - In Lafa (a blockwise flow LM), N dominates. Under a schedule where states become clean, the N effect is exactly 0 late in decoding even though SC is readable at readout, so "readout access alone" is already refuted there.
  - In ELF, F dominates. A pre-registered test showed that freezing the neighbours' final latents at the decode pass removes 89% [81, 97]% of F's effect, so ELF's tokens commit at the decode pass from the neighbours' final latents.
  - Exploratory: at full trajectory length, decode(final latent) and decode(last estimate x̂) agree 100%.
  - A toy-model test failed because its unmeasured auxiliary (states uninformative at commitment) was false: the toy committed at step 0. The lesson: measure the regime before testing.
- **Account under test:** a position reads its neighbours' feedback as a substitute for their states while those states are uninformative, and only where the feedback is readable in the computation where the token commits.

## Calibration (regime diagnostic, separate noise seed; verbatim JSON)
```json
{"rows": [{"n": 8, "t_n": 0.19778668880462646, "dis": 0.960205078125, "agree_full_state": 0.0361328125, "agree_full_fb": 0.3480224609375}, {"n": 12, "t_n": 0.26125094294548035, "dis": 0.949951171875, "agree_full_state": 0.0517578125, "agree_full_fb": 0.505859375}, {"n": 16, "t_n": 0.291354238986969, "dis": 0.92919921875, "agree_full_state": 0.0699462890625, "agree_full_fb": 0.7010498046875}, {"n": 20, "t_n": 0.31790032982826233, "dis": 0.9083251953125, "agree_full_state": 0.0926513671875, "agree_full_fb": 0.7978515625}, {"n": 24, "t_n": 0.35040491819381714, "dis": 0.8671875, "agree_full_state": 0.133056640625, "agree_full_fb": 0.870361328125}, {"n": 28, "t_n": 0.6340638995170593, "dis": 0.1302490234375, "agree_full_state": 0.85791015625, "agree_full_fb": 0.9388427734375}, {"n": 32, "t_n": 1.0, "dis": 0.0, "agree_full_state": 1.0, "agree_full_fb": 1.0}], "pick": {"n": 24, "t_n": 0.35040491819381714, "dis": 0.8671875, "agree_full_state": 0.133056640625, "agree_full_fb": 0.870361328125}}
```

## The pre-registration (verbatim)
### 2.13 Family E4: ELF decode-access test (added 2026-11-20, before any of its data; basis: E3b, insights §26, the family T lesson)
**Account under test (H-clean + access):** a position reads its neighbours' fed-back predictions as a substitute for their states while those states are uninformative, and only where the feedback is readable in the computation where the token commits.

**Rival:** readout access alone decides the route.

**Design:**
- **Matched fine-tunes of ELF-B** (`src/elf_ft_decode.py`; hcfrl ELF code patched in memory, read-only): arm `readable` (decoder rows receive a stop-gradient same-state SC estimate with p = 0.5) vs arm `zeroed` (unchanged ELF training).
- Both arms: init ELF-B (step 95085); OWT-t5 shards 0–1; 3,000 steps × batch 64 × L 1024; lr 2e-5 constant after 100 warmup; an identical extra no-grad forward; fine-tune seeds {0, 1, 2}.
- **Decomposition** (`src/elf_access_decomp.py`): routed O/N/F over the trajectory and the decode pass; target j = 64; n = 256; noise seed 2 (unused in E/E3/E3b/calibration); 32 logit-normal steps; truncation n_steps ∈ {32, 24}.
  - n_steps = 24 was chosen by a pre-set rule on separate noise (seed 7): `src/results/elf_trunc_calib.json`, t = 0.35, state→token 13% vs feedback→token 87%.
  - Access at inference: {read, zero} for `readable`; {zero} for `zeroed`.
- **Outcome:** the Mistral span log-prob of token j against the fixed O1N1F1 left context (`src/elf_judge_fixedctx.py`).

**Analysis and decision:** `analysis/elf_access.py` (frozen with this entry; its docstring is the rule).
- **M1:** Δ_24 > 0.
- **M2:** I = Δ_24 − Δ_32 > 0 with Δ_32 < 0.5·Δ_24.
- Validity V1–V4: identities, regime, identifiability, fluency.
- **Rule check before freezing:** 8 synthetic scenarios returned the intended verdicts: account → SUPPORTED; rival → REFUTED; clear null → REFUTED; reversal → REFUTED; no SC effect, regime failure and fluency failure → NOT TESTABLE; partial rival → INCONCLUSIVE.

**Platform:** Prime GPU pod (the user's decision, 2026-11-20). Smoke tests on CARC CPU passed (identities 1.0000).

## Frozen analysis (verbatim)
```python
"""Prereg §2.13 (family E4: ELF decode-access test; written before its data).
Usage: python analysis/elf_access.py <root>
  <root>/<arm>_s<seed>/<n_steps>_<access>/cells_judged_fixedctx.json   ({"64": {cell: [span log-prob or None]}})
  <root>/<arm>_s<seed>/<n_steps>_<access>/meta.json                    (dis, identity_min, ...)
  arm ∈ {readable, zeroed}; seed ∈ {0, 1, 2}; readable arm: n_steps {32, 24} × access {read, zero}; zeroed arm: {32, 24} × zero.

Outcome: the Mistral span log-prob of target token j = 64 against the fixed O1N1F1 left context of the same config
(target-mediated, as in E3/E3b). Per sample (complete cases): 8 cell values → absolute Shapley O, N, F; T = c111 − c000.
Δ_n = N(readable arm, access read, n) − N(readable arm, access zero, n), paired by sample (the same noise and model).

Predictions of the account (H-clean + access):
  M1 (mandatory)  Δ_24 > 0: in the truncated regime (states uninformative, feedback informative) making the decode-pass
                  feedback readable increases direct neighbour reading. Holds iff the CI lower bound > 0. FAILS iff the CI
                  upper bound < SESOI = 0.1·|T(readable, zero, 24)| (a clear null or a reversal).
  M2 (mandatory)  I = Δ_24 − Δ_32 > 0: the increase is larger when the states are uninformative. At full length
                  x̂ ≈ final latent (calibration: decode(state) = decode(feedback) for 100%), so readable feedback is
                  redundant there. M2 holds iff the CI lower bound of I > 0 AND the CI upper bound of Δ_32 < 0.5·Δ_24 (point).
                  M2 FAILS iff the CI upper bound of I < 0.5·Δ_24 (the rival's pattern: Δ_32 is at least about half of
                  Δ_24), or the CI lower bound of Δ_32 > 0.5·Δ_24 (a clear full-length increase, contradicting
                  redundancy), or the CI of I lies below 0.
  S1 (supportive) N(readable, read, 24) − N(zeroed arm, zero, 24) > 0 (between matched fine-tunes).
Rival "readout access alone decides": Δ_24 ≈ Δ_32 > 0, so M2 fails or is undetermined while Δ_32 > 0.

Inference: a two-level bootstrap (20,000; seed 20261120). Resample fine-tune seeds, then samples within each seed; each
quantity is the mean over the resampled seeds. Holds iff the 95% CI excludes 0 in the predicted direction; FAILS iff the
CI excludes 0 in the opposite direction; otherwise undetermined.

Validity / manipulation checks (all must pass, else NOT TESTABLE):
  V1 identities: identity_min ≥ 0.99 in every config;
  V2 regime: in the readable arm, dis(24) ≥ 0.2 and dis(32) ≤ 0.05 (seed means);
  V3 identifiability: T(readable, read, 24) and T(readable, zero, 24) have CI lower bounds > 0;
  V4 fluency: mean c111 at (readable, read, 32) ≥ mean c111 at (zeroed, zero, 32) − 0.2·|mean c111 at (zeroed, zero, 32)|
     (the readable fine-tune has not degraded the model by more than 20%).
Decision:
  NOT TESTABLE  if any of V1–V4 fails;
  REFUTED       else if M1 or M2 FAILS;
  SUPPORTED     else if M1 and M2 both hold;
  INCONCLUSIVE  otherwise.
S1 and the full tables (T, O, N, F per config; Δ_32) are reported.
"""
import glob, itertools, json, math, os, sys
import numpy as np

KEYS = list(itertools.product((0, 1), repeat=3)); R = 20000


def shap_rows(J):
    """J: {cell: [v]} → array [n_complete, 5] of (T, O, N, F, c111) plus sample indices."""
    n = len(J["O0N0F0"]); rows, idx = [], []
    for i in range(n):
        v = {k: J[f"O{k[0]}N{k[1]}F{k[2]}"][i] for k in KEYS}
        if any(x is None for x in v.values()): continue
        b = v[(0, 0, 0)]; val = lambda S: v[tuple(int(p in S) for p in "ONF")] - b
        sh = {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S)))
                     for k in range(3) for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}
        rows.append([v[(1, 1, 1)] - b, sh["O"], sh["N"], sh["F"], v[(1, 1, 1)]]); idx.append(i)
    return np.array(rows), np.array(idx)


def load(root):
    D = {}
    for d in sorted(glob.glob(f"{root}/*_s*/*_*")):
        arm_seed, cfg = d.split("/")[-2], d.split("/")[-1]; arm, seed = arm_seed.rsplit("_s", 1); n, acc = cfg.split("_")
        J = json.load(open(f"{d}/cells_judged_fixedctx.json"))["64"]; meta = json.load(open(f"{d}/meta.json"))
        X, idx = shap_rows(J); D[(arm, int(seed), int(n), acc)] = {"X": X, "idx": idx, "meta": meta}
    return D


def two_level(fn, seeds, rng):
    """fn(seed, sample_index_array) → scalar for one seed. Returns the point estimate and 95% CI."""
    pt = np.mean([fn(s, None) for s in seeds]); bs = np.empty(R)
    for r in range(R):
        ss = rng.choice(seeds, len(seeds)); bs[r] = np.mean([fn(s, rng) for s in ss])
    return pt, np.percentile(bs, [2.5, 97.5])


def main(root):
    D = load(root); rng = np.random.default_rng(20261120)
    seeds = sorted({k[1] for k in D if k[0] == "readable"}); zseeds = sorted({k[1] for k in D if k[0] == "zeroed"})
    print("config                         n   T        O        N        F       c111   dis   ident")
    for k in sorted(D):
        X = D[k]["X"]; m = D[k]["meta"]
        print(f"{k[0]:8} s{k[1]} n{k[2]:2d} {k[3]:4}   {len(X):4d} {X[:,0].mean():+.3f}  {X[:,1].mean():+.3f}  {X[:,2].mean():+.3f}  {X[:,3].mean():+.3f}  {X[:,4].mean():+.3f}  {m['dis']:.3f} {m['identity_min']:.4f}")

    def paired(n):
        def f(s, r):
            A, B = D[("readable", s, n, "read")], D[("readable", s, n, "zero")]
            common = np.intersect1d(A["idx"], B["idx"]); ia = np.searchsorted(A["idx"], common); ib = np.searchsorted(B["idx"], common)
            d = A["X"][ia, 2] - B["X"][ib, 2]
            if r is not None: d = d[r.integers(0, len(d), len(d))]
            return d.mean()
        return f
    def diff_of(fa, fb): return lambda s, r: fa(s, r) - fb(s, r)
    def col(key_fn, c):
        def f(s, r):
            X = D[key_fn(s)]["X"][:, c]
            if r is not None: X = X[r.integers(0, len(X), len(X))]
            return X.mean()
        return f
    d24 = two_level(paired(24), seeds, rng); d32 = two_level(paired(32), seeds, rng)
    I = two_level(diff_of(paired(24), paired(32)), seeds, rng)
    T24r = two_level(col(lambda s: ("readable", s, 24, "read"), 0), seeds, rng)
    T24z = two_level(col(lambda s: ("readable", s, 24, "zero"), 0), seeds, rng)
    n_read24 = np.mean([D[("readable", s, 24, "read")]["X"][:, 2].mean() for s in seeds])
    n_zarm24 = np.mean([D[("zeroed", s, 24, "zero")]["X"][:, 2].mean() for s in zseeds]) if zseeds else float("nan")
    print(f"Δ_24 = {d24[0]:+.3f} [{d24[1][0]:+.3f}, {d24[1][1]:+.3f}]   Δ_32 = {d32[0]:+.3f} [{d32[1][0]:+.3f}, {d32[1][1]:+.3f}]   I = {I[0]:+.3f} [{I[1][0]:+.3f}, {I[1][1]:+.3f}]")
    print(f"S1 (descriptive): N(readable, read, 24) {n_read24:+.3f} vs N(zeroed arm, zero, 24) {n_zarm24:+.3f}")
    v1 = all(v["meta"]["identity_min"] >= 0.99 for v in D.values())
    dis24 = np.mean([D[("readable", s, 24, "read")]["meta"]["dis"] for s in seeds]); dis32 = np.mean([D[("readable", s, 32, "read")]["meta"]["dis"] for s in seeds])
    v2 = dis24 >= 0.2 and dis32 <= 0.05
    v3 = T24r[1][0] > 0 and T24z[1][0] > 0
    c_r = np.mean([D[("readable", s, 32, "read")]["X"][:, 4].mean() for s in seeds]); c_z = np.mean([D[("zeroed", s, 32, "zero")]["X"][:, 4].mean() for s in zseeds])
    v4 = c_r >= c_z - 0.2 * abs(c_z)
    print(f"V1 identities {v1} | V2 regime dis24 {dis24:.3f} dis32 {dis32:.3f} → {v2} | V3 T24 read {T24r[0]:+.3f} [{T24r[1][0]:+.3f}], zero {T24z[0]:+.3f} [{T24z[1][0]:+.3f}] → {v3} | V4 fluency c111 readable {c_r:+.3f} vs zeroed {c_z:+.3f} → {v4}")
    st = lambda q, want: "holds" if (q[1][0] > 0 if want > 0 else q[1][1] < 0) else "FAILS" if (q[1][1] < 0 if want > 0 else q[1][0] > 0) else "undetermined"
    sesoi = 0.1 * abs(T24z[0])                                   # smallest effect of interest: 10% of the truncated SC effect
    m1 = "holds" if d24[1][0] > 0 else "FAILS" if (d24[1][1] < 0 or d24[1][1] < sesoi) else "undetermined"
    half = 0.5 * max(d24[0], 0.0)
    if I[1][0] > 0 and d32[1][1] < half: m2 = "holds"
    elif I[1][1] < 0 or (d24[0] > 0 and I[1][1] < half) or (d24[0] > 0 and d32[1][0] > half): m2 = "FAILS"
    else: m2 = "undetermined"
    print(f"SESOI for M1 = {sesoi:.3f}; M2 threshold 0.5·Δ_24 = {half:.3f}")
    print(f"M1 (Δ_24 > 0): {m1}   M2 (I > 0): {m2}")
    if not (v1 and v2 and v3 and v4): dec = "NOT TESTABLE (" + ", ".join(n for n, ok in (("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4)) if not ok) + ")"
    elif "FAILS" in (m1, m2): dec = "REFUTED"
    elif m1 == "holds" and m2 == "holds": dec = "SUPPORTED"
    else: dec = "INCONCLUSIVE"
    riv = " ; pattern matches the rival 'readout access alone' (Δ_32 > 0, I not > 0)" if (d32[1][0] > 0 and m2 != "holds") else ""
    print("E4 DECISION:", dec + riv)


if __name__ == "__main__":
    main(sys.argv[1])

```

## Fine-tune runner (verbatim)
```python
"""Phase 13 step 3: matched ELF-B decoder-access fine-tunes (hcfrl's vendored ELF code is used read-only, patched in memory).

ARM readable: decoder-mode rows receive a stop-gradient self-conditioning input with probability self_cond_prob (0.5),
              the same as denoiser rows. The estimate comes from a no-SC pass on the decoder's own corrupted input
              (same-state SC, standard practice; ELF Alg. 3), at t = the row's mean corruption level.
ARM zeroed:   unchanged ELF training (decoder rows get zero SC). The control.
Both arms: identical data order, seed, steps, lr and init (ELF-B weights), and an identical extra no-grad forward
(computed and discarded in `zeroed`), so compute and RNG streams match.

Usage: python elf_ft_decode.py --hcfrl <root> --init <ELF-B snapshot dir> --data <dataset dir> --arm readable|zeroed
       --out <dir> [--steps 3000 --batch 64 --max_length 1024 --lr 2e-5]
"""
import argparse, os, re, sys, types

ap = argparse.ArgumentParser()
ap.add_argument("--hcfrl", required=True); ap.add_argument("--init", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--arm", choices=["readable", "zeroed"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max_length", type=int, default=1024); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()

ELF = os.path.join(a.hcfrl, "hcfrl", "elf"); sys.path.insert(0, ELF); os.chdir(ELF)
import train_step as TS                                   # noqa: E402
import train as TR                                        # noqa: E402

ORIG = "        sc_half = x_pred_cond * (1.0 - decoder_mask_B11)\n"
PATCH = '''        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            _t_dec_est = decoder_lambda_t.mean(dim=(1, 2)).to(dtype)
            _out_dec = model(torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1), _t_dec_est,
                             deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale)
        _, _x_dec = net_out_to_v_x(_out_dec, decoder_z, _t_dec_est, t_eps)
        _x_dec = restore_cond(_x_dec, x0, cond_seq_mask) * use_self_cond_mask.to(dtype)
        if getattr(config, "decoder_sc_readable", False):
            sc_half = x_pred_cond * (1.0 - decoder_mask_B11) + _x_dec * decoder_mask_B11
        else:
            sc_half = x_pred_cond * (1.0 - decoder_mask_B11)
'''
src = open(TS.__file__).read()
assert src.count(ORIG) == 1, "train_step.py changed; the patch anchor was not found exactly once"
mod = types.ModuleType("train_step_ft"); mod.__file__ = TS.__file__
exec(compile(src.replace(ORIG, PATCH), TS.__file__ + " [mechflow decoder-access patch]", "exec"), mod.__dict__)
TR.train_step = mod.train_step                            # train.py does `from train_step import train_step`

_y = open(os.path.join(ELF, "configs", "training_configs", "train_owt_ELF-B.yml")).read().replace(
    "src/configs/sampling_configs/", os.path.join(ELF, "configs", "sampling_configs") + "/")
_tmp = f"/tmp/elf_ft_cfg_{os.getpid()}.yml"; open(_tmp, "w").write(_y)
cfg = TR.load_config_from_yaml(_tmp)
over = dict(data_path=a.data, eval_data_path=None, init_from=a.init, resume=None, output_dir=a.out,
            max_length=a.max_length, global_batch_size=a.batch, batch_size=None, grad_accum_steps=1, lr=a.lr,
            lr_schedule="constant", warmup_steps=100, epochs=1, seed=a.seed, decoder_sc_readable=(a.arm == "readable"),
            sampling_configs_path=None, sampling_configs=[], online_eval=False, eval_freq=10 ** 9, save_freq=10 ** 9,
            use_wandb=False, num_workers=2)
for k, v in over.items(): setattr(cfg, k, v)
# exactly `steps` optimizer steps: 1 epoch over the first steps × batch examples (the same subset and order in both arms)
_orig_load = TR.load_dataset
def _load_subset(config, *args, **kw):
    tr, ev = _orig_load(config, *args, **kw)
    return tr.select(range(min(len(tr), a.steps * a.batch))), ev
TR.load_dataset = _load_subset
print(f"[elf_ft] arm={a.arm} steps={a.steps} batch={a.batch} L={a.max_length} lr={a.lr} seed={a.seed} decoder_sc_readable={cfg.decoder_sc_readable}", flush=True)
TR.run_training(cfg, force_cpu=a.cpu)

```

## Decomposition driver (verbatim)
```python
"""Phase 13 step 3: O/N/F decomposition for a (fine-tuned) ELF-B checkpoint, with truncation and decode-access settings.
For each config (n_steps ∈ --n_steps, inference access ∈ --access), target j = --j:
  - plain SC-on and SC-off runs (identity references) and the manipulation check
    dis = P(decode(z) ≠ decode(x̂)) (both with zero SC), as in elf_trunc_calib.py;
  - 8 routed cells (trajectory routing as in family E; decode pass routed, with SC readable iff access = read);
  - identities: O1N1F1 = SC-on (decode SC as configured) and O0N0F0 = SC-off, ≥ 99% token agreement (bf16 on GPU).
Writes <out>/<n>_<access>/cells.json in the {j: {cell: [ids]}} format read by elf_judge_fixedctx.py, plus meta.json."""
import argparse, itertools, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_instr import load_elf, run_cell_x, run_plain_x, decode_cell, decode_plain_x
ap = argparse.ArgumentParser()
ap.add_argument("--hcfrl", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_steps", default="32"); ap.add_argument("--access", default="zero"); ap.add_argument("--j", type=int, default=64)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--L", type=int, default=128)
ap.add_argument("--seed", type=int, default=2); ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device)
if dev.type == "cpu": torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
model, cfg, tok = load_elf(a.hcfrl, a.ckpt, a.L, dev)
from utils.sampling_utils import get_sampling_steps
torch.manual_seed(1); ts = get_sampling_steps(32, time_schedule="logit_normal").to(dev)
z0_all = torch.randn((a.n, a.L, model.text_encoder_dim), generator=torch.Generator().manual_seed(a.seed)) * cfg.denoiser_noise_scale
cells = [(o, n, f) for o in (0, 1) for n in (0, 1) for f in (0, 1)]
for nst, acc in itertools.product([int(x) for x in a.n_steps.split(",")], a.access.split(",")):
    rd = acc == "read"; tsn = ts[:nst + 1]; out = {f"O{o}N{n}F{f}": [] for o, n, f in cells}; ident = []; dis = []; t0 = time.time()
    for b in range(0, a.n, a.bs):
        z0 = z0_all[b:b + a.bs].to(dev)
        zon, xon = run_plain_x(model, cfg, z0, tsn, True); zoff, xoff = run_plain_x(model, cfg, z0, tsn, False)
        yon = decode_plain_x(model, zon, xon, rd); yoff = decode_plain_x(model, zoff, xoff, False)
        dis.append((decode_plain_x(model, zon, xon, False) != decode_plain_x(model, xon, torch.zeros_like(xon), False)).float().mean().item())
        for o, n, f in cells:
            z, xp = run_cell_x(model, cfg, z0, tsn, a.j, o, n, f)
            y = decode_cell(model, z, xp, a.j, o, n, f, rd)
            if (o, n, f) == (1, 1, 1): ident.append((y == yon).float().mean().item())
            if (o, n, f) == (0, 0, 0): ident.append((y == yoff).float().mean().item())
            out[f"O{o}N{n}F{f}"] += y.tolist()
        print(f"n_steps {nst} access {acc} batch {b}: {time.time()-t0:.0f}s, identity min {min(ident):.4f}", flush=True)
    assert min(ident) >= 0.99, f"identity agreement {min(ident)}"
    d = f"{a.out}/{nst}_{acc}"; os.makedirs(d, exist_ok=True)
    json.dump({str(a.j): out}, open(f"{d}/cells.json", "w"))
    json.dump({"ckpt": a.ckpt, "n_steps": nst, "t_end": tsn[-1].item(), "access": acc, "j": a.j, "n": a.n, "seed": a.seed,
               "identity_min": min(ident), "dis": sum(dis) / len(dis)}, open(f"{d}/meta.json", "w"))

```

## Instrumentation (verbatim; routing patch, trajectories, routed decode)
```python
"""2 x 2 x 2 (O x N x F) decomposition of self-conditioning in ELF (gap G-e). Uses the ELF implementation in the user's
hcfrl project READ-ONLY (imported via sys.path at runtime; nothing is vendored here).

ELF self-conditioning: per position, input = [z_i, x_prev_i]; SC off = zeros in place of x_prev (as in training).
Three streams per denoising step, run as one batch of 3B rows through a patched Attention:
  S: every position gets zero SC input;  F: every position gets its x_prev;  O: target j gets x_prev_j iff O = 1.
  S attends S; F attends F; O-row queries read key q from O if q is the query itself, a prefix token or the target,
  and otherwise from F (N = 1) or S (N = 0).
Trajectory: neighbours take (v, x) from F (F = 1) or S (F = 0); target j takes (v, x) from O.
Identities: (1,1,1) == SC-on sampling; (0,0,0) == SC-off sampling (checked by `check_identities`).
"""
import itertools, os, sys

import torch
import torch.nn.functional as F_

ROUTE = None          # {"P": prefix length, "j": target seq index, "n": 0/1}
_ORIG = None


def load_elf(hcfrl_root, ckpt_dir, L, device):
    sys.path.insert(0, hcfrl_root)
    import hcfrl.elf_path  # noqa: F401
    from transformers import AutoTokenizer
    from configs.config import load_config_from_yaml, apply_config_overrides
    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder
    from utils.checkpoint_utils import load_checkpoint
    from utils.train_utils import TrainState, get_optimizer
    y = open(os.path.join(ckpt_dir, "config.yml")).read().replace(
        "src/configs/sampling_configs/uncond_sampling_configs.yml",
        os.path.join(hcfrl_root, "hcfrl/elf/configs/sampling_configs/uncond_sampling_configs.yml"))
    tmp = f"/tmp/elf_cfg_{os.getpid()}.yml"; open(tmp, "w").write(y)
    cfg = apply_config_overrides(load_config_from_yaml(tmp), [f"max_length={L}", "use_wandb=false"])
    tok = AutoTokenizer.from_pretrained(cfg.encoder_model_name)
    enc_cfg, _ = get_encoder(cfg.encoder_model_name, torch.float32)
    model = ELF_models[cfg.model](
        text_encoder_dim=enc_cfg.d_model, max_length=cfg.max_length, attn_drop=0.0, proj_drop=0.0,
        num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens, bottleneck_dim=cfg.bottleneck_dim)
    state = TrainState(model=model, optimizer=get_optimizer(model, cfg, lr=1e-4), lr_scheduler=None,
                       ema_params1=TrainState.init_ema(model), step=0, epoch=0,
                       dropout_generator=torch.Generator(device="cpu").manual_seed(0))
    state, _ = load_checkpoint(ckpt_dir, state)
    model = state.model
    if state.ema_params1:
        model.load_state_dict(state.ema_params1)
    model = model.to(device).eval()
    _install()
    return model, cfg, tok


def _install():
    global _ORIG
    from modules.layers import Attention, scaled_dot_product_attention
    if _ORIG is not None:
        return
    _ORIG = Attention.forward

    def forward(self, x, rope_fn, attention_mask=None, deterministic=True):
        if ROUTE is None:
            return _ORIG(self, x, rope_fn, attention_mask=attention_mask, deterministic=deterministic)
        assert attention_mask is None
        B3, N, C = x.shape; B = B3 // 3; hd = self.dim // self.num_heads
        qkv = self.qkv(x).reshape(B3, N, 3, self.num_heads, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope_fn is not None:
            q, k = rope_fn(q), rope_fn(k)
        (qs, qf, qo), (ks, kf, ko), (vs, vf, vo) = q.split(B), k.split(B), v.split(B)
        os_ = F_.scaled_dot_product_attention(qs, ks, vs)
        of_ = F_.scaled_dot_product_attention(qf, kf, vf)
        P, j, n = N - ROUTE["L"], ROUTE["j"], ROUTE["n"]
        src = torch.full((N, N), 1 if n else 0, dtype=torch.long, device=x.device)
        src[:, :P] = 2; src[:, P + j] = 2; src[torch.arange(N), torch.arange(N)] = 2
        allow = torch.cat([src == 0, src == 1, src == 2], dim=1)                       # (N, 3N)
        oo = F_.scaled_dot_product_attention(qo, torch.cat([ks, kf, ko], 2), torch.cat([vs, vf, vo], 2), attn_mask=allow)
        out = torch.cat([os_, of_, oo], 0).permute(0, 2, 1, 3).reshape(B3, N, C)
        return self.proj(out)
    Attention.forward = forward


def _vx(out, z, t, t_eps):
    from utils.sampling_utils import net_out_to_v_x
    return net_out_to_v_x(out, z, t, t_eps)


@torch.no_grad()
def run_cell(model, cfg, z0, t_steps, j, O, N, F, w=1.0, autocast=True):
    """One trajectory for cell (O, N, F) at target sequence index j. Returns final z."""
    global ROUTE
    B, L, Cd = z0.shape
    dev = z0.device
    z = z0.clone(); xprev = torch.zeros_like(z)
    wt = torch.full((3 * B,), float(w), device=dev)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and dev.type == "cuda")
    with ctx:
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item()
            zero = torch.zeros_like(z)
            o_in = zero.clone()
            if O:
                o_in[:, j] = xprev[:, j]
            inp = torch.cat([torch.cat([z, zero], -1), torch.cat([z, xprev], -1), torch.cat([z, o_in], -1)], 0)
            tb = torch.full((3 * B,), t, device=dev)
            ROUTE = {"L": L, "j": j, "n": N}
            try:
                out = model(inp, tb, deterministic=True, self_cond_cfg_scale=wt)
            finally:
                ROUTE = None
            v, x = _vx(out, torch.cat([z, z, z], 0), tb, cfg.t_eps)
            vS, vF, vO = v.split(B); xS, xF, xO = x.split(B)
            vv = (vF if F else vS).clone(); xx = (xF if F else xS).clone()
            vv[:, j] = vO[:, j]; xx[:, j] = xO[:, j]
            z = z + (tn - t) * vv
            xprev = xx
    return z


@torch.no_grad()
def run_plain(model, cfg, z0, t_steps, sc_on, w=1.0, autocast=True):
    """Reference ELF sampling loop (no routing): SC on = previous x-prediction, off = zeros."""
    B = z0.shape[0]; dev = z0.device; z = z0.clone(); xp = torch.zeros_like(z)
    wt = torch.full((B,), float(w), device=dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and dev.type == "cuda"):
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item()
            tb = torch.full((B,), t, device=dev)
            out = model(torch.cat([z, xp], -1), tb, deterministic=True, self_cond_cfg_scale=wt)
            v, x = _vx(out, z, tb, cfg.t_eps)
            z = z + (tn - t) * v
            xp = x if sc_on else torch.zeros_like(x)
    return z


def decode(model, cfg, z, t_final, w=1.0):
    from utils.generation_utils import _dlm_decode_batch
    return _dlm_decode_batch(z, model, t_final, cfg, w)


# ---- Phase 13 step 3 (decoder access): truncated trajectories + routed decode pass with optional readable SC ----
@torch.no_grad()
def run_cell_x(model, cfg, z0, t_steps, j, O, N, F, w=1.0, autocast=True):
    """As run_cell, but returns (z, xprev) after len(t_steps) - 1 steps (pass a prefix of t_steps to truncate)."""
    global ROUTE
    B, L, Cd = z0.shape; dev = z0.device
    z = z0.clone(); xprev = torch.zeros_like(z); wt = torch.full((3 * B,), float(w), device=dev)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and dev.type == "cuda")
    with ctx:
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item(); zero = torch.zeros_like(z); o_in = zero.clone()
            if O: o_in[:, j] = xprev[:, j]
            inp = torch.cat([torch.cat([z, zero], -1), torch.cat([z, xprev], -1), torch.cat([z, o_in], -1)], 0)
            tb = torch.full((3 * B,), t, device=dev); ROUTE = {"L": L, "j": j, "n": N}
            try: out = model(inp, tb, deterministic=True, self_cond_cfg_scale=wt)
            finally: ROUTE = None
            v, x = _vx(out, torch.cat([z, z, z], 0), tb, cfg.t_eps)
            vS, vF, vO = v.split(B); xS, xF, xO = x.split(B)
            vv = (vF if F else vS).clone(); xx = (xF if F else xS).clone(); vv[:, j] = vO[:, j]; xx[:, j] = xO[:, j]
            z = z + (tn - t) * vv; xprev = xx
    return z, xprev


@torch.no_grad()
def run_plain_x(model, cfg, z0, t_steps, sc_on, w=1.0, autocast=True):
    B = z0.shape[0]; dev = z0.device; z = z0.clone(); xp = torch.zeros_like(z); wt = torch.full((B,), float(w), device=dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and dev.type == "cuda"):
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item(); tb = torch.full((B,), t, device=dev)
            out = model(torch.cat([z, xp], -1), tb, deterministic=True, self_cond_cfg_scale=wt)
            v, x = _vx(out, z, tb, cfg.t_eps); z = z + (tn - t) * v; xp = x if sc_on else torch.zeros_like(x)
    return z, xp


def _dec_logits(model, inp, w):
    B = inp.shape[0]; dev = inp.device
    tf = torch.ones((B,), device=dev, dtype=inp.dtype); sc = torch.full((B,), float(w), device=dev, dtype=inp.dtype)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        _, lg = model(inp, tf, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=torch.ones((B,), device=dev, dtype=inp.dtype))
    return lg


@torch.no_grad()
def decode_plain_x(model, z, xp, readable, w=1.0):
    """Decode pass; SC input = xp if readable else zeros (ELF default)."""
    return _dec_logits(model, torch.cat([z, xp if readable else torch.zeros_like(z)], -1), w).argmax(-1)


@torch.no_grad()
def decode_cell(model, z, xprev, j, O, N, F, readable, w=1.0):
    """Routed decode pass. S: zero SC; F: SC = xprev if readable; O: target j gets xprev_j iff O and readable.
    Tokens: neighbours from F (F = 1) or S; target from O. If not readable, S = F = O inputs (SC is zero everywhere)."""
    global ROUTE
    B, L, _ = z.shape; zero = torch.zeros_like(z); fb = xprev if readable else zero; o_in = zero.clone()
    if O and readable: o_in[:, j] = xprev[:, j]
    inp = torch.cat([torch.cat([z, zero], -1), torch.cat([z, fb], -1), torch.cat([z, o_in], -1)], 0)
    ROUTE = {"L": L, "j": j, "n": N}
    try: lg = _dec_logits(model, inp, w)
    finally: ROUTE = None
    yS, yF, yO = lg.argmax(-1).split(B); y = (yF if F else yS).clone(); y[:, j] = yO[:, j]
    return y

```

## Your tasks
1. **Construct validity.**
   - Does the `readable` fine-tune create a decoder that can actually use SC at decode, or could 3,000 steps from a decoder trained with zeros leave it ignoring the channel?
   - Is the training-time SC estimate (a no-SC pass on the decoder's own corrupted input at t = mean λ) a reasonable match for inference-time x̂ (the last flow-step estimate)?
   - Does truncation at n = 24 (t = 0.35) put the decoder out of distribution? Its training corruption is a per-token λ ~ sigmoid(N(p_mean, p_std)).
2. **Routing at the decode pass.** Check `decode_cell` and `run_cell_x`: are the O/N/F semantics correct? Do the identities hold for the right reason?
3. **Is Δ_32 ≈ 0 really what the account predicts?** Or does redundancy make N unidentifiable at 32 (so M2 is trivially satisfied)? Could M2 pass for an uninteresting reason?
4. **Decision rule.** Find any scenario where the rule gives a misleading verdict. Consider:
   - SESOI and threshold choices;
   - the two-level bootstrap with 3 seeds;
   - complete-case filtering;
   - the fluency check;
   - V2's thresholds.
5. **Confounds and rivals.** What else could produce M1 and M2 without the account being true (e.g. fine-tuning changes unrelated to access, the judge's sensitivity to context)? What control would rule it out?
6. **Statistics and power.** Are n = 256 and 3 fine-tune seeds enough? Is pairing by sample valid across access conditions within a model?
7. **Concrete fixes,** each implementable before data.

**Output:** a ranked problem list, then "Would you trust a SUPPORTED verdict? A REFUTED verdict? Why?"
