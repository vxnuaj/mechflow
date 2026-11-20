# TODO

## mechflow — pre-experiment checklist

Proposal: `mechflow/mechflow.md`. Started 2026-11-13. ICML 2027 deadline ≈ late Jan 2027.

Rule: no confirmatory experiment runs until Phases 1–6 are done. Phase 7 is exploratory only, and its data is never reused for confirmation.

---

### Phase 1 — Pin down the claim ✅ (2026-11-13) → `mechflow/phase1.md`
Writing these down first keeps the reading from quietly reshaping the question.

- [x] Write the core claim in one sentence, in a form a reviewer could disagree with. (§1: claim, sub-claims C1–C4, scope, operationalization)
- [x] Write the *strongest skeptical alternative* in one sentence. (§2.1: R0/R6 + R1, "noisy-state coordination + generic leakage"; rival table R0–R9 in §2)
- [x] List what evidence would convince *you* the claim is false. (§3: kill criteria K1–K10 in phase order, interval-vs-margin rule, success criteria, exhaustive outcome lattice)
- [x] List what the paper's headline figure would be if everything works. (§4: 4-panel Figure 1 sketch + headline sentence)
- [x] (added) Assumptions the claim rests on, each mapped to the check that tests it. (§5: A1–A8)
- [x] Independent adversarial review: 3 rounds (v1 → v3.1), reviewer signed off with no blockers. All issues logged in `phase1.md` §8.
- [ ] *(optional)* **You:** skim `phase1.md` §1 and §3.4; push back if you'd frame the claim differently.

### Phase 2 — Ground truth on the artifact ✅ (2026-11-13) → `mechflow/phase2.md`
Everything downstream depends on these answers.

**Lafa paper (read in full, not the abstract):**
- [x] Exact self-conditioning ablation numbers. **32.3 → 38.1 (+5.8 pp)**, Table 3. GPT was right; the "~4" summary was wrong.
- [x] Train-time or inference-time ablation? **Inference-time** (same model, self-conditioning off at decode).
- [x] Training recipe. Stop-grad pass on the same I_t; p_sc = 0.5; when dropped, the input is plain x; W₁, W₂ zero-initialized.
- [x] Sampler details. Euler (x-prediction) on a learned **asynchronous** per-position grid; readout = per-position argmax at t_{K−1}; "prefix guidance" = CFG against a prefix-free pass, with separate feedback per branch.
- [x] Settings behind the reported numbers. 0.5B, B = 4, 32 steps, **g = 8**, inference noise **0.1** (training 4.0).

**Official code (`lafa-gh/large-flow-lms`):**
- [x] "Off" input. **Plain x_t** (no W₁, no W₂). q = 0 would give x + xW₁ᵀ, which was never trained.
- [x] q at the first step. A bootstrap pass on the raw noise at t = 0 (per branch under CFG).
- [x] Block boundaries. Absolute in training; **prompt end at inference**, so alignment is controllable (A3 ✅).
- [x] Hook points. `_apply_self_cond` (FM:1185); eager HF attention; `DiTAdaLN.patch_layer` wrap point.
- [x] **K1 faithfulness.** Per-query K/V selection is feasible: append off-pass K/V to the `DynamicCache`, use a dense additive mask, then trim. **Drop granularity = whole batch per optimizer step** → mixed per-position masks are out of distribution (A7 ⚠️; K3 is now critical).
- [x] Determinism. Deterministic given the seed; the only randomness is the initial noise (batched eval shares it across rows).
- [x] Checkpoints. Apache-2.0; 0.5B 2.9 GB, 7B 30.9 GB, 8B 33.3 GB (EMA + raw).

- [x] **Gate.** K1 not triggered. §0 of `mechflow.md` updated. Experiments affected → Phase 4 revision list (`phase2.md` §3–4): scope/origination, C3 donor source, primary regime g = 8, lead–lag no longer counts as evidence.

### Phase 2.5 — Widen the hypothesis set ✅ (2026-11-13)
After Phase 2, our cross-position claim is one of **four competing explanations** for the +5.8 pp self-conditioning gain. Phase 3 has to check novelty for all of them, so list them first. The full rewrite waits for Phase 4.

- [x] Add the four mechanisms to `mechflow.md` (new §1.1):
  - **M1: per-position stabilization** (the skeptic). Predicts: the gain survives at g = 1; per-token CE improves; the hybrid rate barely changes.
  - **M2: cross-position coordination** (our original claim, `phase1.md`). Predicts: the gain survives at g = 1; the hybrid rate drops with feedback on, concentrated on coupled blocks.
  - **M3: guidance anchoring** (new). The feedback is the *unguided* prediction while the step follows the 8× guided one. Predicts: the gain mostly **disappears at g = 1**.
  - **M4: tie-breaking at low noise** (new). Sampling starts at 0.1·σ, so something has to break ties between answers. Predicts: the gain is larger at noise 0.1 than at 4.0; without feedback, positions stall between answers.
- [x] New framing (to be written up properly in Phase 4): *decompose the self-conditioning gain causally across M1–M4, then dissect whichever mechanism wins.* M2 stays the headline candidate.

### Phase 3 — Literature review ✅ (2026-11-13) → `mechflow/phase3.md`
Goal: decide whether the idea is novel and what's already known, not to collect citations.

- [x] Verify every reference in the proposal (existence, authors, what it actually claims). Fix [3], [4], [5], [7], [8], [9], [10].
- [x] **Novelty threat 1:** mode mixing / factorization error in parallel decoding (masked and discrete diffusion, copula-style fixes, remasking). Has anyone shown continuous flows or self-conditioning fix it?
- [x] **Novelty threat 2:** concurrent work on Lafa (papers citing it; arXiv from Oct–Nov 2026).
- [x] **Read first:** the papers Lafa itself cites as closest prior work: [47, 57] (noise-to-token maps capturing joint dependencies) and [62, 63] (independent sampling limits few-step quality). These are the M2 threat.
- [x] **M3 novelty:** Li & Xu, untied self-conditioning (2608.22244). It feeds the raw prediction back and uses the corrected one for the step, which is close to what Lafa does with respect to guidance. Also search for self-conditioning × classifier-free-guidance interactions.
- [x] **M4 novelty:** symmetry breaking / tie-breaking in low-temperature or deterministic diffusion sampling; reduced-noise sampling in continuous diffusion LMs.
- [x] Commitment and symmetry breaking in diffusion (Raya & Ambrogioni; Biroli & Mézard; Sclocchi et al.). Can we borrow a principled definition of "commitment time"?
- [x] Self-conditioning origins and recipes (Analog Bits; SED) and continuous diffusion LMs (Diffusion-LM, CDCD, Plaid, LangFlow).
- [x] Fixed-point self-conditioning (2607.00714): read in full. Does "can be removed or compressed" apply to Lafa's setting?
- [x] Mech interp of diffusion LMs: has anyone already patched self-conditioning inputs?
- [x] For each paper, record one line: *relevant / threat / irrelevant*, plus its implication for H1–H4.
- [x] **Gate:** for each of M1–M4, record *novel / partly known / known*. If M2 is already shown, the decomposition framing still stands, but the headline moves to whichever mechanism is still open.
- [x] **Outcome:**
  - M2 mechanism **open** (phenomenon known: FMLM, CoCommit);
  - M3 demoted to moderator;
  - new rival **M5** (Shabalin et al. 2604.02028);
  - commitment-time definition and χ(t) adopted.
  
  Not yet read (before submission): full D3PM, DFM, ReMDM and TESS 2; the tails of the Lafa/Yoo/Shabalin/Davis HTML.
- [ ] **Ongoing:** check arXiv weekly for the Kim/Boffi/Lee/Yoo group and for papers citing Lafa (scoop risk).
- [ ] **Before submission:** read the full texts of D3PM, DFM, ReMDM and TESS 2, and the unread tails of the Lafa / Yoo (2607.00714) / Shabalin (2604.02028) / Davis (2610.13834) HTML versions. Update `phase3.md` §6.

### Phase 3.5 — Smoke test ✅ (2026-11-13) → `mechflow/smoke_memo.md`
Cheap, and decisive about which mechanism to bet on. Prompts used here are **never** reused for confirmation.

- [x] **Decide where to run it:** Mac for the smoke test (see throughput); CARC later for full benchmarks.
- [x] Set up Lafa-0.5B in an isolated venv (done: `mechflow/smoke/.venv`, py3.12, torch 2.10, transformers 5.3.0, + pillow/matplotlib, which the remote code requires; checkpoint snapshot `18deca35`, EMA only): `pip install -r requirements.txt -c constraints-tested.txt`; checkpoint `lafa-hf/lafa-0.5B`, 2.9 GB. Pass `num_steps`, `cfg_scale` and `noise_scale_override` explicitly; the HF defaults ≠ the paper's settings.
- [x] Measure throughput (sec per block-step) → decide Mac vs. CARC for the rest. **Done 2026-11-13: Mac M4 Pro (MPS, bf16) ≈ 13 ms per forward pass. Paper settings (32 steps, g = 8) ≈ 66 passes per block ≈ 0.9 s per block. End-to-end GSM8K ≈ 1 min per problem per config → Mac is OK for slices; use CARC for full benchmarks.** Setup in `mechflow/smoke/` (`.venv`, `common.py`, `00_throughput.py`).
- [x] **Block-level teacher-forced test** — *done; results in `smoke_memo.md` §1. SC barely changes accuracy vs gold, but pulls blocks toward the model's own AR mode (+8 pp block EM vs AR, CI excludes 0); the effect is ≈ equal at g = 1 and g = 8, so M3 is not supported* — (`02_block_tf.py`; cheap and higher-powered). The prefix is the chat prompt plus a gold GSM8K solution truncated at a random point; generate one 4-token block; score against the gold tokens (per-token accuracy, block exact match, cross-position correctness correlation). Configs: SC {on, off} × g {1, 8} × noise {0.1, 4.0} at 32 steps, plus 4 steps. ~300 paired blocks per config. This carries the M1/M3/M4 comparisons below; the end-to-end slice is only the sanity check.
- [x] Reproduce one sanity number (*done: SC 0.38 vs noSC 0.34, +0.04 [−0.10, +0.18], n = 50; direction matches the paper, underpowered*): a small GSM8K slice (`01_gsm8k_e2e.py`, n ≈ 50, runs in the background) at the paper settings (32 steps, g = 8, noise 0.1), self-conditioning on vs. off. It should show a gap in the direction of 38.1 vs. 32.3. *Caveat:* the prompt format approximates lm-eval `gsm8k_cot`, so absolute numbers aren't comparable to the paper.
- [x] Full vs. no self-conditioning at **g = 1 vs. g = 8** → separates M3. *SC effect ≈ equal at g = 1 and g = 8 → M3 dropped.*
- [x] Full vs. none at **noise 0.1 vs. 4.0** → separates M4. *The SC benefit is larger at 4.0 → M4 contradicted; dropped.*
- [x] Hybrid rate, full vs. none, at **1 / 4 / 32 steps** on ~50 hand-made ambiguous prompts (1 step = positive control) → existence check for M2. *Script written: `03_hybrid.py`* (≈ 60 prompts with two-word entities in 3 categories, open + 'pick one of A or B'; word-level valid/hybrid/other; K {1, 4, 32} × SC × g {1, 8} at noise 0.1, plus noise 4.0 × 4 seeds). *Done: hybrids ≈ 30% on bimodal prompts at the paper setting vs. 0% for AR; without SC, many sub-word garbles ("Albertaac"); SC climbs garble → hybrid → valid and never makes outputs worse.*
- [x] Per-token CE vs. joint consistency, full vs. none → separates M1/M5 from M2. *Covered by the coherence check: the gain sits at in-block positions 2–3. Consistent with M2, but doesn't yet separate it from M1/M5 (later positions are also the slowest).*
- [x] **Coherence check** (`04_coherence.py`; *done: in differing blocks SC is +3.6 nats more coherent [+2.4, +4.9], almost all at positions 2–3*): AR log-prob of the SC vs. no-SC blocks. Are no-SC blocks less coherent (hybrid-like), or valid alternatives? Needed because gold-match can't separate "worse" from "different".
- [x] Write a half-page memo `mechflow/smoke_memo.md`: which of M1–M5 the evidence favours. **This is the input to Phase 4.**

### Phase 4 — Sharpen hypotheses and definitions ✅ (2026-11-13) → `mechflow/phase4.md` (v4.3; supersedes `phase1.md` as the working spec)
- [x] **Apply the Phase 3.5 verdict** (`smoke_memo.md` §5):
  - [x] narrow to **M2 vs. R0 vs. M1/M5**. M3/M4 are "not supported (underpowered)" and kept as measured secondary interactions (phase4 §0.1);
  - [x] primary metrics: own-AR coherence + **external judge (co-primary)** + **in-block dependence dep_j**; token-level hybrid classifier with alignment and precedence; gold-match demoted (phase4 §2.2–2.3);
  - [x] position-specific predictions recast as discriminating magnitudes (phase4 §3.1);
  - [x] regime decision: **mechanism at noise 4.0 / g = 1** (q–x window), **headline at the paper regime** (phase4 §4);
  - [x] novelty check for P1 ("flow LMs mode-mix where their own AR head doesn't"): **novel** (phase4 §9);
  - [x] exclusion list saved: `smoke/results/EXCLUDE_gsm8k_test_indices.json` (434) + smoke prompts (phase4 §7).
- [x] **q–x information-gap diagnostic** (new; `smoke/05_qx_gap.py`): at noise 0.1 the state catches up with the feedback by step ~2; at noise 4.0 the feedback leads by ~15–20 steps → mechanism regime moved; new kill criterion K12 (phase4 §0.2, §5).
- [x] Apply the Phase 3 design changes:
  - rival M5 via a stale-own test + accumulation null;
  - cloning τ_i (per-position re-noising, ε sweep, R ≥ 32) + margin proxy;
  - within-block χ;
  - fixed-point gap (descriptive only);
  - α/cutoff baselines.
  
  M3 sweep → `leads/01`; M4 null → `leads/05`. **B = 8/16 checkpoints are not released → block-size test dropped.**
- [x] Reframe around M2 vs. R0 vs. M1/M5, with per-mechanism claims and predictions (phase4 §1, §3.1) and the lattice → paper mapping (§5.5).
- [x] Revise H1–H4 → H1–H5 (phase4 §3; proposal §4 marked superseded).
- [x] Apply the Phase 2 revision list:
  - origination back in scope (secondary);
  - minimal-pair donors;
  - both CFG branches;
  - matched schedule → x-neutralization intervention;
  - power redone per estimand (prompts as the unit).
- [x] Single-position "off" candidates (a)/(b)/(c), plus the CFG rule. Selection by **efficacy + own-fidelity**, not neighbour drift (phase4 §2.1, §2.7).
- [x] Operational definitions: off; coherence / dep_j; hybrid classes; mode margin; cloning commitment; χ; state decodability (phase4 §2).
- [x] Confound tables per experiment (phase4 §4), plus a positive control (§2.8) and a multiplicity plan (§5.4).
- [x] Adversarial review round 1 → v4.1 (phase4 §10).
- [x] Review rounds 2–3 → v4.3. All blockers resolved; the last one (N share zero by construction) was fixed with the reviewer's proposed shadow-pass decoupling, which the reviewer stated makes the spec ready for Phase 5.

### Phase 5 — Pre-register ✅ (2026-11-13) → `mechflow/prereg.md`
Write to `mechflow/prereg.md` and don't edit it after confirmatory runs start (departures → prereg §12).
- [x] **Decide the paper framing:** phenomenon-first (adopted per recommendation; PI may override until the first confirmatory run; prereg §0).
- [x] **Scope trim:** primary family = P1, P2, C1′, H2-N. M5, χ and cloning commitment are optional/exploratory (prereg §0, §2.3).
- [x] Predictions for each hypothesis, with direction and rough size (prereg §2; prior on shares stated).
- [x] Kill criteria and go/no-go thresholds: gates PC → K3 → K12 → K11 → G1 → Phase 8 go/no-go (prereg §5); outcome → framing map (§7).
- [x] Primary vs. secondary metrics: dep_j, hybrid rate and μ_j primary; coherence judges as co-requirements; benchmarks secondary; gold-match not primary (prereg §3).
- [x] Data splits: exploration (smoke categories + 3 templates) / confirmation (7 held-out categories × held-out templates) / circuit-validation (15% of confirmation pairs); natural text split by document; GSM8K test excluded from block-level work (prereg §4).
- [x] Statistical plan: cluster bootstrap; interval-vs-margin decisions; Holm (primary); BH (secondary); TOST (S3); one Pocock-bounded extension; exclusion rules; BH for head sweeps (prereg §6).
- [x] Power + compute (prereg §8; [PILOT] items fixed by rule in §11).
- [x] Independent methodological review of prereg.md: round 1 (v1.0 → v1.1: margins, clustering, gate substitutions, extension removed); round 2 (v1.1 → v1.2: lattice holes, S6 fallback, external anchoring); round 3 (v1.3: last lattice cell). **Reviewer confirmed freeze-ready → v1.3 FROZEN.** Hashes in `prereg_hashes.log`.
- [x] External hash anchoring is required before confirmation runs → moved to **Phase 8** (it's a user action due before the first confirmatory run, not a Phase 5 deliverable).

### Phase 5.5 — Pre-run verification ✅ (2026-11-13) → prereg v1.4
Found 2026-11-13 after the freeze; fixes are allowed because no confirmation data exists yet (log as prereg v1.4 in §12 and re-hash).
- [x] **External judge isn't independent.** Lafa-0.5B/7B were initialized from **Qwen2.5-Instruct**, so a Qwen2.5-7B-Instruct judge shares pretraining lineage and would agree with Lafa's own notion of "typical". → *Done: switched to **Mistral-7B-Instruct-v0.3** (Apache-2.0, ungated; Llama and Gemma are gated).* Replace it with a different family (candidates: Llama-3.1-8B-Instruct or Gemma-2-9B-it; check the license and HF access on CARC). Amend prereg §2.1 (P2 co-requirement), §9 and `phase4.md` §2.2.
- [x] **Natural-text contamination.** Lafa's SFT data (Dolci-Instruct-SFT) includes OpenMathInstruct-2 (built on GSM8K/MATH *train* problems) and persona-GSM, so **GSM8K-train blocks may be near-training data** (inflated coherence). UltraChat isn't listed in the Dolci card. → Make **UltraChat-200k `test_sft` the primary** natural-text source. Keep GSM8K-train only if a contamination check passes: own-AR coherence and AR-agreement on GSM8K-train blocks are not higher than on a held-out math source (e.g. SVAMP) by more than 0.25 nats/block on the exploration split. Amend prereg §4.2.
- [x] **Quick scoop check before committing CARC compute** (*done 2026-11-13: nothing new on mode mixing/coherence in Lafa*): arXiv since 2026-11-13 for papers citing Lafa or from the Kim/Boffi/Lee/Yoo group, on self-conditioning / mode mixing / coherence. (Then weekly; the next is due ~2026-11-20.)
- [x] Apply the amendments → prereg v1.4, hash in `prereg_hashes.log`.
- [ ] Weekly scoop check — next due ~2026-11-20.

### Phase 6 — Build and validate infrastructure ✅ (2026-11-14)
A hook bug silently invalidates every downstream result, so test the instrument before using it.

- [x] **CARC readiness** — *done 2026-11-14. Instead of multi-block batching, parallelism is multi-worker sharding (runner verified: 24/24 tasks, 2 GPUs × 4 workers, no duplicates); the benchmark confirms batch-1 passes are latency-bound, so workers are the right lever. Account via gitignored `src/slurm/carc.env`.* (*access verified 2026-11-13: Claude can run commands via the `ssh carc` alias, using your key + a persistent ControlMaster session; account `<slurm-account>`; `gpu` partition (A100 / A40 / L40S); work in `/scratch1/<carc-user>` (9.8 TiB quota); avoid `/project2` (92% full) and keep `/home1` light; `module load conda` needs `bash -l`*) (before any full-benchmark or 7B run): batch the block-level scripts (many blocks per forward pass; batch-1 runs are latency-bound, so a GPU alone gives only ~2–3×). *Done so far: env at `/scratch1/<carc-user>/mechflow/env` (uv, py3.12, torch 2.10+cu128, transformers 5.3.0, lm-eval 0.4.11); models in `hf_home` (0.5B `18deca35`, 7B `434c100d`, Mistral-7B judge); `src/runner.py` (multi-worker per GPU, sharded, resumable) + `src/slurm/run_sharded.sbatch` (array jobs); runner self-test job 14229410 ✓.*
- [x] Reproduce plain Lafa-0.5B generation with the official code (smoke `00_throughput.py`; instrumented sampler `src/lafa_instr.py` mirrors `_euler_denoise_block`).
- [x] Reproduce at least one reported number: **official harness, GSM8K (`gsm8k_cot`, flexible-extract), 100 problems, paper settings (32 steps, g = 8, noise 0.1): 30.0% ± 4.6 vs. paper 30.4%** ✓. Run locally (MPS, `results_local/repro_gsm8k_K32_g8_mps/`, ~45 s/problem); strict-match is 2% (formatting). CARC copies cancelled.
- [x] Determinism: same seed → bitwise identical (T2: max diff 0.0, MPS).
- [x] **Identity-hook test** (T1: instrumented sampler vs. `_euler_denoise_block`, 6 prompts × 6 configs: **max logit diff 0.0**, same tokens and same #forwards).
- [x] **K/V-selection identity test** (T4): tokens identical; logits within **5e-5 in fp32** (CPU). In bf16, step 0 is exact, but re-routing adds ~1e-2 prob rounding noise from step 1, compounding to ≤ 2.6 logits by step 32. → **Rule: every arm of a comparison runs through the same re-routed code path (identity re-route in controls); precision-critical runs in fp32 with TF32 off.**
- [x] **A7 / K3 instrument validation** (`src/k3_drift.py`, `src/k3_calibrate.py`; 120 exploration blocks; MR regime). **K3 NOT triggered → candidate (a) plain-x selected.**
  - (i) Null-calibrated Mahalanobis (rule pre-registered in prereg §12 before calibration, pushed as `ced33a2`): the null exceeds in 6.3% of cells (pooled 0.85). (a): 5.6%, pooled 0.82 ✓; (b) 6.9%, 0.84 ✓; (c) 3.5%, 0.81 ✓.
  - (iii) Efficacy: (b) 0.86 ✗, (c) 0.75 ✗ (both < 0.9). (a) is trivially 1.
  - (ii) Cross-candidate agreement not applicable (only (a) passes at full efficacy).
  - Single-step δ^D (log-prob of the clean token, targets 2–3 from sources 0–1): (a) 0.060 ± 0.008, (b) 0.029 ± 0.005, (c) 0.152 ± 0.011 → **effect sizes depend on the off definition; report as sensitivity.**
- [x] **Off-hook test** (T3: hook returning plain x == `self_cond=False`, g ∈ {1, 8}: max logit diff 0.0).
- [x] Single-position knockout changes only the intended position's input (T5: all rows ok).
- [x] **Positive control** (prereg §5 PC; T6): a planted one-hot feedback signal at position 1 changes position 3's prediction (KL > 0); cutting all other positions' reads of position 1 removes it (injected/cut KL ratio ≥ 4e8); position 1's own effect is kept. *The x-erasure arm of PC is deferred to Phase 7 (needs LEACE maps fit on exploration).*
- [x] Trajectory logging (`record=True`): a full trajectory record is ~0.75 MB/block (PR) / 0.52 MB (MR) in fp16 (~1.4 / 1.0 MB in fp32) → ~30–45 GB for 7,350 blocks × 8 cells; fits scratch. **Policy:** full states only for exploration probing/LEACE subsets; confirmatory runs store scalars + window-step states.
- [x] Throughput and memory benchmark (*ran on A40 via the debug partition, 2026-11-14; A100 never got scheduled*): batch-1 single process, 0.5B bf16 ≈ 24 ms/pass (1.5 GB), 0.5B fp32 ≈ 21.5 ms (3.1 GB), 7B bf16 ≈ 41–43 ms (15.5 GB). **Latency-bound (slower than the Mac's 13 ms) → throughput comes from multiple workers per GPU** (`runner.py`: ~8 workers for 0.5B, 2–3 for 7B on 48 GB).
  **GPU-hour budget (A40, with workers):**
  - 2 × 2 × 2 decomposition (7,350 blocks, ~3.8M passes): ~3 GPU-h.
  - Exp F phenomenon (~2,600 prompts × ~20 conditions × ≤ 2 blocks): ~6 GPU-h.
  - C1′ single-step and donor swaps: < 2 GPU-h.
  - 7B (K11 + S1 + C1′ subset, ~1,000 prompts): ~3 GPU-h.
  - End task (GSM8K + HumanEval + IFEval, 4 configs, ~46 s/problem per worker): ~13 GPU-h.
  - **Total ≈ 30 A40 GPU-h** (+ ~50% contingency ≈ 45 GPU-h). Feasible; the bottleneck is queue wait, not compute.
- [x] Experiment tracking: `run_meta()` (code SHA-256 of `src/*.py`, host, versions, model snapshot, config) is saved with every result and runner shard.

### Phase 7 — Exploratory pilot (exploration split only; produces the prereg §11 [PILOT] values)
Purpose: confirm the instruments and phenomena on exploration data, and freeze every pilot-derived value **before** any confirmation data exists. Each §11 value is hashed and pushed when it's set.

**7.1 Data (exploration and confirmation built now; confirmation is NOT run until Phase 8 go)**
- [x] Two-answer prompt bank (*done: `src/data/entities.py`, `src/data/build_bank.py` → `data/bank.jsonl`; exploration 240 pairs, confirmation 602 (9,632 prompts), circuit-validation 118; 0 tokenization drops; hashed in §11*): 12 categories × ≥ 40 two-word entities → ≥ 80 pairs per category; exploration = 3 smoke categories × 6 templates (5 smoke + 1 new); confirmation = 9 categories × 8 new templates; circuit-validation = 15% of confirmation pairs; in-context tokenization filter. Hash the bank and split membership (§11).
- [x] Natural text (*done: `src/data/build_natural.py` → UltraChat only; exploration 4,536 / confirmation 16,258 / circuit-validation 1,777 docs; hashed in §11*): UltraChat-200k `test_sft` (primary) + GSM8K-train + SVAMP; document-level 20/80 split; hash the document lists (§11).
- [x] *(not executable as specified: SVAMP has no solution text → GSM8K-train conservatively excluded; recorded in §11)* GSM8K-train contamination check (exploration): own-AR coherence vs. SVAMP ≤ +0.25 nats/block → keep or drop GSM8K-train (§11).

**7.2 Phenomenon and measures (exploration)**
- [x] Token-aligned hybrid classifier + **audit of 150 labels** (2026-11-15; stratified 45 hybrid / 55 other / 50 mode; `src/results/classifier_audit_sample.json`):
  - **148/150 = 98.7% agreement; hybrids 45/45** → PASS (≥ 90% / ≥ 85%).
  - The 2 errors were "valid answer + extra word" → fixed in `classify_text.py` (logged in prereg §12).
  - Note: ~27% of "other" are messy cross-answer mixes outside the strict prefix + suffix definition (e.g. "Janeeyall", "Tyels Brahe") → the pre-registered hybrid rate is a conservative undercount; report descriptively.
  - [ ] **YOU: spot-check ~20 rows** of the audit file (optional).
- [x] P1 point estimate on **non-smoke** exploration prompts (0.5B, PR, 1,436 prompts; text-level token-boundary classifier `src/classify_text.py`, **not yet audited**):
  - **Hybrid: flow(SC) − AR = 7.5 pp [5.5, 9.7] → below the 10 pp go/no-go bar.**
  - Non-valid (hybrid + garble/truncation): 20.3 pp [16.9, 24.0].
  - SC effect: hybrid −11.3 pp [−14.2, −8.3]; valid +31 pp [26, 35].
  - Lower than the smoke test's ~30% because the smoke pairs were hand-picked to recombine; countries 2.5%, cities 8.3%, scientists 10.4%.
  - **→ Flag for the Phase 8 decision**; the audit may move the hybrid/other boundary, but the definition must not be tuned to clear the bar.
- [x] dep_j / coherence (own-AR + Mistral judge) on natural blocks, SC on vs. off (PR) → P2 pilot (300 exploration blocks; `src/jobs_p2.py`):
  - **dep_{2,3} +0.50 [0.20, 0.82]** (one-sided LB 0.24 > margin 0.10 ✓).
  - Own-AR coh_{2,3} +0.75 [0.46, 1.04] ✓; **Mistral coh_{2,3} +1.44 [1.01, 1.90]** (LB 1.07 > 0 ✓); position 0 ≈ 0 (later-position pattern ✓).
  - SC changes 62% of blocks; conditional on change, Mistral +2.32.
- [ ] Natural-text check: incoherence-event rate, flow vs. AR (S2 pilot).

**7.3 Mechanism pilot values (exploration, MR)**
- [x] Linear probes for "which answer" on x and on q per step (held-out pairs; pair-relative features; `src/pilot_qx.py`) → **q–x window = [] → K12 TRIGGERED** (q leads x by 0.1–0.2 at every step, but never q ≥ 0.85 with x ≤ 0.65). **Primary family → {P1, P2, H2-N}**; C1′ = not separable (§11, §12).
- [ ] LEACE erasers per (position, step bucket); verify the probe on erased x ≤ 0.55; hash the maps (§11). Then the PC erasure arm.
- [x] ~~margin_C1 and the C1 floor~~: moot, since C1′ left the primary family via K12 (it may still run as exploratory).
- [x] b_S3 = b_S4 = max(50% × 0.50, 0.05) = **0.25 nats** (§11).
- [ ] Pilot SDs (cluster-level) → required n for P1 / P2 / C1′; H2-N simulated power; expand the bank if needed (§11).
  - **H2-N pilot (300/300 blocks, 0.5B, MR; identity checks all pass) — STRONGLY SUPPORTS M2 (coordination):**
    - dep: total T = +0.95 [0.50, 1.40]; **N main = +0.70 [0.40, 1.02]**; O ≈ 0; F ≈ 0; **N share 0.74 (Möbius) / 1.02 (Shapley)**; N − 0.20·T = +0.51 [0.25, 0.78].
    - coh: T = +1.71; **N = +1.19 [0.88, 1.52]**; N share 0.70 / 0.93; contrast +0.85 [0.56, 1.14].
    - (The 196-block interim gave the same picture.)
    - → SC's coherence benefit is carried almost entirely by positions **reading neighbours' fed-back predictions**; own feedback and the state route ≈ 0 (contrary to the stated prior).
  - **H2-N 7B replication (100/100 blocks, MR; the 99-block numbers below are unchanged to 2 d.p. except where noted in `src/analyze_h2n.py` output):**
    - **coh:** T = +2.56 [1.63, 3.53]; **N = +1.42 [0.77, 2.09]**; N share 0.55 (Möbius) / 0.80 (Shapley); **O share 0.19 [0.02, 0.32] (small but real at 7B)**; F ≈ 0; **N×F synergy +1.45 [0.49, 2.47]**; contrast +0.91 [0.36, 1.47] ✓.
    - **dep:** the total effect is not resolved at 7B (T = +0.49 [−0.53, 1.46]); N +0.64 [0.00, 1.35]; contrast +0.55 [−0.04, 1.16] (borderline).
    - → Coordination replicates in direction and dominance at 7B, with a modest own-feedback share and an N×F synergy (iterative mutual agreement).
    - Still needed: confirmation split; G1 relevance; Mistral judge (P2 pilot queued: `src/jobs_p2.py`). (`src/analyze_h2n.py`)

**7.4 Gates**
- [x] **K11 (scale) — NOT TRIGGERED:** 7B on exploration (CARC). **Hybrid part DONE (400 prompts, same prompts as 0.5B):**
  - 7B flow(SC) hybrids − AR = **+0.3 pp [0.0, 0.8]** (0.5B: +7.7); non-valid +1.8 pp (0.5B: +22.5).
  - **Without SC, 7B is 38% garbled and 1.3% hybrid; SC raises valid answers by +37.5 pp [32.5, 42.6]** (0.5B: +31).
  - → First K11 conjunct met (h_flow ≤ 5%). The coherence conjunct is still to run.
  - → **P1 is a small-model effect; the robust, scale-general finding is that SC is necessary for coherent parallel multi-token output.**
  - **Coherence conjunct: 7B SC gain on coh_{2,3} = +2.38 (own-AR) / +2.37 (Mistral) vs. 0.5B +0.57 / +1.23 → 4.2× / 1.9× larger at 7B → K11 NOT TRIGGERED** (both conjuncts were required).
- [x] **G1 (regime relevance) — PASSED (2026-11-15):**
  - Coherence half: MR/PR ratio 1.54 (own-AR) / 1.01 (Mistral).
  - End-task half (official harness, MR, reserved 25% slices, paired per problem; 43 chunked units on Mac + CARC debug): GSM8K +4.2 pp [0.6, 7.9], IFEval +9.6 [0.7, 18.5], HumanEval +2.4 [−4.9, 12.2]; **mean +5.4 pp [1.3, 9.7]** (LB > 1 pp ✓). Absolute MR accuracy is low (GSM8K 8.8% vs. PR ~30%), so frame it carefully. (`src/results/g1_final.txt`)
- [x] Record all gate outcomes and the primary-family substitution (§5.2) in §11; hash and push. (PC ✓, K3 clear, K12 triggered → {P1, P2, H2-N}, K11 not triggered, G1 passed.)

- [ ] **Interp additions (exploratory; `interp_methods.md` §3 Tier 1):**
  - [ ] T1a: information-flow routes + AtP* head × step screen (fp32);
  - [ ] T1b: cross-position Jacobian (q-path vs. x-path) over steps;
  - [ ] T1c: interpretable-bottleneck test (top-1 feedback);
  - [ ] T1d: contrastive "which answer" q-direction + dose–response steering.
- [x] *(covered in `phase3.md` §8)* Add to the literature ledger (`phase3.md` §6): DiffusionGemma transparency (2606.20560, "token smearing" — must differentiate), DiffusionGemma commitment (2606.14620), CANDI (2510.22510), DifFRACT (2606.15796), PEAP (2502.04577), MIB (2504.13151).

- [x] *(answered by the Phase 3.5 smoke test)* SC gap at g = 1 vs. g = 8 (≈ equal); hybrid rate at 1 step (positive control: hybrids present); seed diversity at noise 0.1 (seeds barely matter → minimal-pair donors); feedback-off vs. full on natural blocks (coherence drop at positions 2–3).

### Phase 7.5 — Strengthening additions (decided 2026-11-15)
- [x] **Pre-register head-level localization as confirmatory** (*done 2026-11-15: prereg §2.4 L1/L2 + §12; pushed*) (prereg §12 amendment + push, BEFORE any confirmation data): per-head q-path vs. x-path path patching → named "feedback-reader" heads; minimal circuit with MIB CPR/CMD faithfulness; behavioural prediction (knocking out the heads lowers SC's coherence gain / raises garbles). Circuit-validation split. (`interp_methods.md` §3 Tier 2)
- [ ] **Second model family:** *Feasibility (2026-11-15): official ELF code is JAX (github.com/lillian039/ELF); torch checkpoints exist on HF (`*-torch`), but the official torch code isn't released → needs an unofficial port (Ugness/elf-torch, sslnon/elf-s2t) validated against JAX; ELF is unconditional, uses SC as a CFG token, and works in a contextual T5 latent space. Estimate 1–2 days → do AFTER Lafa confirmation + localization.* ELF (`embedded-language-flows/ELF-B/M/L-owt`, torch versions; whole-sequence flow with self-conditioning). Feasibility check (load + one 2×2×2 block), then replicate H2-N (+ scale via B/M/L). Backups: LangFlow (`Continuous-Rivals-Discrete/langflow-owt`), self-conditioned FMLM (`wonjKim/self-conditioned-fmlm`).
- [x] **Guidance-on (paper-regime) 2 × 2 × 2 (2026-11-16):** `run_block_two_pass_cfg` (exact identities). Exploration pilot, 286 blocks, PR:
  - **dep: N share 0.90, contrast +0.39 [0.16, 0.65]; coh: N share 0.70, contrast +0.44 [0.24, 0.65]**; O ≈ 0; F ≈ 0; N × F synergy +0.44 / +0.50.
  - → **The coordination mechanism holds at the deployed settings.**
  - Pre-registered as secondary S12 (prereg §2.2, §12).
- [x] **Localization tool built (2026-11-15):** `PerLayerMasks` (per-layer, per-head attention masks via forward pre-hooks) + `run_block_headcut` (head-resolved N cut for target j). Tests: no cut = normal decoding (exact, 0.0); all-heads cut = the 2 × 2 × 2 cell (O1, N0, F1) (identical tokens; ≤ 0.4 logit bf16 re-routing noise) → the baseline uses `reroute=True` (same code path).
- [ ] Localization discovery (exploration, after Phase 8): layer-level sweep (24 layers) → head-level within the top layers → k, H* frozen in §11 → test L1/L2 on circuit-validation.
- [ ] **Mechanism → better sampler (upside):** strengthen only the cross-position feedback reads (or feed fresher neighbour predictions) and test for fewer garbles at 4–8 steps / higher speculative acceptance.
- [x] **Targeted novelty check for the NEW headline** (*done 2026-11-15 → `phase3.md` §8: decomposition + neighbour-reading result NOVEL; scale trend partly known (SCMDM); cross-position intuition anticipated by CANDI (untested). To do: read 2610.19628 App. D.5.2; re-check papers citing Lafa*) ("SC's benefit comes mainly from positions reading each other's fed-back predictions; grows with scale"):
  - full read of CANDI (2510.22510; SC as anchors for conditional dependencies, the closest idea);
  - DiffusionGemma transparency (2606.20560, "token smearing");
  - anything since 2026-11-13 from the Lafa group or citing Lafa;
  - prior claims about SC benefit vs. model scale;
  - framing related work (parallel-decoding consensus/commitment).
  
  Update the `phase3.md` ledger + differentiation statement. Do BEFORE confirmation and writing.
- [ ] Start drafting the paper in parallel (intro, related work, method) while the GPUs run.
- [ ] Brief the co-author with `README.md` + the pilot numbers.

### Phase 8 — Decision gate ✅ (2026-11-16) → `mechflow/phase8_memo.md`
- [x] Write a one-page pilot memo (`mechflow/phase8_memo.md`, 2026-11-15): recommendation **GO**, reframed around the coordination mechanism.
- [ ] Update `mechflow/mechflow.md` (framing, experiments, primary step-count regime). *(moved to the writing phase)*
- [x] **Decision: GO** (user, 2026-11-16). Guidance-on 2 × 2 × 2 added as S12 (pilot: coordination holds at deployed settings). Classifier audit passed (the user's spot-check is optional).
- [x] **Localization decision:** committed. L1/L2 were pre-registered in prereg §2.4 before any confirmation data; the per-head tool is built. Runs in Phase 10.
- [x] Prereg hash anchored externally: each §11 entry was pushed to the public GitHub repo before any confirmation data (final freeze sha256 `90ea6df0…`, 2026-11-16).
- [x] Freeze: §11 complete (n, code hash, snapshots, BH family = {S1, S3, S4, S11, S12}); scope deviations logged in §12 before data.

### Phase 9 — Confirmatory runs ✅ (2026-11-17) → `mechflow/confirmation_results.md`
- [x] P1 (0.5B, PR; Mac MPS): 9,632 prompts / 602 pairs: **+6.97 pp [6.09, 7.95] ✓**.
- [x] P2 (0.5B, PR, + Mistral): 7,100 docs: **+0.540 [0.477, 0.604] ✓**, co-requirement met.
- [x] S3 / S4 arms: 2,000 docs: S3 equivalent ✓, S4 non-inferior ✓.
- [x] H2-N (0.5B, MR): 2,000 docs: **contrast +0.515 [0.409, 0.624] ✓**, Shapley N share 1.14.
- [x] S12 (PR, guidance on): 750 docs: **+0.440 ✓**, N share 0.73.
- [x] S1 (7B P1): 2,400 prompts / 150 pairs: +0.25 pp ✗ (mixing ≈ 0 at 7B).
- [x] Analysis exactly as pre-registered: **primary 3 / 3 supported (Holm); secondary 4 / 5 (BH)**; §7 row = phenomenon + feedback-channel mechanism.

### Phase 9.5 — Insight experiments (exploratory; exploration split) → `mechflow/insights.md`
- [x] Zero-GPU checks on the confirmation 2 × 2 × 2 outputs (2026-11-17):
  - N is not subword stitching (word-start +0.71 ≈ continuation +0.66);
  - **N prevents within-block duplication** (2.9% → 1.1–1.3%) → fixes both NAT multimodality symptoms;
  - layer-0 KV heads are tuned to the SC channel (1.3–1.5× gain).
- [ ] E1 step × layer map of N (critical window vs. depth shortcut).
- [ ] Symmetry-breaking coupling: cross-position A−B Jacobian; forward–backward joint switching with vs. without N.
- [ ] E4 message steering / answer-ID vector (binding IDs).
- [ ] E2 source-resolved N (left vs. right neighbours) + reader-head offset classification + base-Qwen comparison.
- [ ] E7 toy factor-graph flow model vs. exact BP marginals.
- [ ] Verify all citations from the 2026-11-17 interp review (abstract-only) before use.

### Phase 9.6 — Closing the hard-to-vary gaps (decided 2026-11-18)
- [x] *(2026-11-18: unresolved by rule → post-hoc two-channel account → confirmed as family R)* **G-a (critical): causal substitution control.** Under the `uniform` grid at late steps, cut neighbour feedback **and** corrupt the neighbours' states (replace the neighbour x with its no-SC-pass / noise-matched x).
  - Substitution predicts: feedback-only cut ≈ 0, joint cut > 0.
  - "Nothing matters late" predicts: both ≈ 0.
  - Commit the predictions before data.
- [x] *(2026-11-18: Mistral N share 0.80 Möbius / 1.04 Shapley; O −0.09; insights §16)* **G-b: external-judge decomposition.** Score the 2,000 confirmation H2-N cell outputs with Mistral (no new generation); report the N share under the external judge.
- [x] *(2026-11-18: family R R1–R3 all SUPPORTED; prereg §12, insights §17)* **G-c: pre-registered held-out replication of E1c** (both grids) on ~1,000 unused confirmation docs, with the decision rule from insights §7, pre-registered before running.
- [ ] **G-d: 7B mechanism.**
  - Exploratory H2-N (227 / 300 interim): coh N share 0.73, O 0.23; dep unresolved.
  - Running: S = 64 rescoring + Mistral judge (auto after 300 blocks); E1c grids at 7B.
  - **S7 confirmation: ALL SUPPORTED (2026-11-19)** — N 0.86 (Mistral 0.76), O 0.07 (insights §22).
- [x] *(toy done 2026-11-18: neighbour-only SC ≥ full, own-only ≈ none; insights §13)*
- [ ] *(ELF: step 0 v2 shows SC +1.46 nats/token; 2 × 2 × 2 routing hook built and identity-checked; **family E pre-registered** (prereg §2.7); generating on the Mac, judging on CARC, auto-analysis)* **G-e: generality.** Toy factor-graph model (E7) + **ELF** second model (cached torch port: ELF-B/M-owt-torch on CARC); run H2-N there.
- [ ] **ELF build plan (recon 2026-11-18):**
  - A PyTorch ELF exists in the user's `hcfrl` project on CARC (`hcfrl/elf/modules/model.py`, `utils/sampling_utils.py`); use it read-only. Checkpoint: ELF-B-owt-torch (T5-small latents, 1,024-length full-sequence denoising, `self_cond_proj([z, x_prev])`, uncond SC = zeros).
  - **Build:**
    - (1) a per-query K/V routing hook on ELF attention: two passes per step, with and without neighbour x_prev;
    - (2) a 2 × 2 (O × N) decomposition at sampled target positions;
    - (3) outcome = coherence of decoded tokens scored by an external LM (Mistral) at the target positions, plus SC-on/off validity.
  - ~1 day.
- [ ] Weekly scoop check (due 2026-11-20); final literature pass before writing.

- [x] **E4 steering (2026-11-18):** neighbour-feedback transplant flips j to the donor mode in 39.5% vs 0.5% (control); late window only; 77/79 flips are hybrids (insights §15.1).
- [x] **G-b external judge:** Mistral N share 0.80 / 1.04; O −0.09 (insights §16).

### Phase 13 — Generalization programme (2026-11-19; merged Claude search + GPT review → `generalization_search.md`; every external fact verified → `verification_log.md`)
**Rules:**
- Decide on absolute route effects (with total T), never on shares alone (lesson from E3).
- Before freezing any decision rule, write out what it returns if the hypothesis is true and if it is false. Freeze it only if the two outcomes differ.

**Account under test (H-clean + access):** a position reads its neighbours' fed-back predictions as a substitute for their states while those states are uninformative, and only where the feedback is readable in the computation where the token commits.
- Lafa: commits at noisy steps with SC readable → N.
- ELF: commits at the SC-free decode pass → F.
- Already shown: the strong "readout access alone decides" version is refuted by E1c (N = 0 in late steps under `uniform` despite SC at readout); E1b: the final step alone carries 0.05.

**Run order (stop and rethink on failure):**
1. [x] **E3b: SUPPORTED (2026-11-19)**, φ_F 0.891 [0.805, 0.971] (CARC CPU job 14391603): ELF decode-pass replication, fresh seed; φ_F lower bound > 0.5.
   - **Fail** → the ELF explanation is wrong; rethink before step 3.
2. [ ] **Toy factorial** (primary test of access; no external model varies it): readout SC on/off × feedback form (expected embedding vs continuous latent), 3 seeds each, with matched data, architecture and budget.
   - [x] 2a. Write `toy/toy_access.py`:
     - ELF-style decode pass (SC readable vs zeroed);
     - expected-embedding vs regression-head feedback;
     - prompt as a causal prefix;
     - three-stream routed sampler for O/N/F over the whole trajectory and the decode pass;
     - built-in identity checks (O1N1F1 = SC-on sampling, O0N0F0 = SC-off).
   - [x] 2b. Smoke test on CARC CPU (tiny steps): identity checks pass, and the cost per seed is measured. Result: pass (step-0 stream diff < 1e-4); ~18 min to train and ~7 min to decompose per model (8 CPUs).
   - [x] 2c. Pre-register as prereg §2.12. Decision rule written out for "account true", "readout-alone rival" and "form rival" scenarios; analysis script frozen; hash and push.
   - [x] 2d-i. GPT red-team prompt written → `prompts/gpt_redteam_toy.md`.
   - [x] 2d-ii. GPT red-team received and verified against the code (2026-11-19). Decision-rule bugs reproduced.
     - **Adopted:**
       - #1 fixed-reference outcome;
       - #2 measured commitment manipulation check plus a shared step grid;
       - #3 the rule is rewritten (all mandatory cells required; refutations always reported);
       - #5 FORM renamed to "estimator recipe";
       - #7 excess-over-independence and diversity as secondary outcomes;
       - #9 10 seeds with seed-specific tasks and seed-level inference;
       - #11 CI-based P4/P5;
       - #12 RNG matched;
       - plus an inference-time access factor.
     - **Not adopted, with reasons (in prereg §12):** #4 (substitution is already tested in Lafa family R; the toy scope is access), #6 (the hybrid world is the intended estimand, as in Lafa), #10 (same-state SC training is standard practice in Lafa and ELF).
   - [x] 2d-iii. Implement T v2 (code + analysis):
     - synthetic rule tests incl. partial-identifiability cases;
     - log the deviation in prereg §12;
     - hash and push;
     - discard the v1 trainings (never decomposed).
   - [x] 2e-iii. Submitted 40 train+decompose jobs (CARC CPU 14396920–59) plus the dependent frozen analysis (job 14396960 → `toy/runs_T2/T_final.txt`). v2 smoke test passed.
   - [x] 2e-i. 12 trainings done 2026-11-19 (13–23 min each; final losses: denoise 0.03–0.09, decode 0.04–0.10). Design check (weights only, no outcomes): embedding norms 11.8–12.2 vs noise norm 11.3, so no embedding inflation and the noise is not trivially defeated.
   - [~] 2e-ii. Superseded by v2 (the v1 checkpoints were never decomposed, so no data was seen).
   - [x] 2f. **Outcome: REFUTED (P1)**; manipulation check failed everywhere, so the access factor was inert (prereg §12; `toy/results_T2/T_final.txt`).
   - [x] 2g. **Done:** the toy commits at step 0–1 from the noise itself (step-0 estimate matches the final mode 78–85%; 99% committed by t = 0.25). P1's auxiliary was false, so the account was untested (insights §25). Diagnosed the failed auxiliary (exploratory, existing checkpoints): per step, the decodability of each position's final mode from its state vs from its feedback, in SC-on sampling. Does the toy commit while neighbour states are already informative? (CARC CPU)
   - [ ] 2h. **v3 (decided from 2g): low-SNR toy.**
     - Add a noise-scale σ (Lafa MR uses 4.0).
     - **Calibration** on separate seeds (100–101), using regime diagnostics only and no O/N/F data. Choose the smallest σ ∈ {2, 4, 8} such that SC-on commitment reaches 0.9 only at t ≥ 0.3 and neighbour state accuracy is < 0.8 when 50% of positions have committed. Choose t_end such that zero-access commit-at-decode ≥ 0.2.
     - Pre-register v3 with **state-decodability manipulation checks (M0)** plus M; red-team; then 10 test seeds (0–9).
     - [x] Code: σ and configurable t_end added to `toy/toy_access.py`; calibration script `toy/toy_calib.py`.
     - [x] Calibration done: **the rule is not met at any σ** (commitment at step 0 at every σ; the decode pass coordinates by itself, so the SC effect is small). v3 not built (insights §25).
     - Previously: decide on a v3 only from 2g. If the states are informative before commitment, design a low-SNR toy (noise scale as in Lafa MR) with a *state-decodability* manipulation check (red-team #4, now adopted). Pre-register, red-team and run. If the states were uninformative, the account's P1 is refuted outright → revise the account before steps 3–5.
   - [x] 2i. Resolved 2026-11-20: the toy line for access is closed (the auxiliary failed; access untested, not refuted). Steps 3–5 are unpaused; the access test moves to step 3 (ELF).
   - **Prediction:** the route follows access, not form (SC-readable → N; SC-zeroed readout → F), on absolute effects.
   - **Fail** (the route follows form) → the core account is refuted; reinterpret steps 3–5 before running them.
3. [ ] **ELF decode-access manipulation (now the primary access test):** a brief fine-tune that lets the decode pass read SC (vs a matched fine-tune that keeps it zeroed).
   - **Prediction (revised 2026-11-20, lesson from family T):** depends on how informative the neighbour states are at ELF's decode pass.
     - Account (H-clean + access): N rises **only if** neighbour states are uninformative at decode. If the final latents are already informative, N does **not** rise.
     - Rival ("readout access decides"): N rises regardless.
     - The test discriminates only once state informativeness at decode is measured **first**.
   - [x] 3a-i. From existing E3 tokens: j's token at decode is driven mostly by the neighbours' final latents (23% → 77% SC-on-token when only the neighbours are swapped), so x̂ ≈ the final latent at full t_end, and a full-t_end access test would be near-tautological (insights §26).
   - [ ] 3a-ii. (Exploratory, CPU) Measure neighbour-state informativeness at ELF-B's decode pass: linear-probe / nearest-token decodability of each position's final token from its final latent vs from its feedback (x̂_prev), on 256 samples. Also commit-at-decode (the decoded token vs the token implied by the last flow estimate).
   - [x] 3a-iii. **Pick: n_steps = 24 (t = 0.35)**: state → full-token 13%, feedback → full-token 87%, disagreement 0.87 (`src/results/elf_trunc_calib.json`). Calibrate a truncated t_end for ELF-B (regime diagnostics only; noise seed 7, unused elsewhere): `src/elf_trunc_calib.py` (CARC CPU).
     - **Pre-set rule (revised before running; replaces the swap-agreement wording):** truncated t_end = the largest t_n with P(decode(state) ≠ decode(feedback)) ≥ 0.2.
   - **GPU platform decided (user, 2026-11-20): Prime.** Start the pod only once all GPU jobs (step 3 fine-tune + decomposition, step 4 LangFlow, step 5 Plaid) are coded, smoke-tested on CPU and pre-registered; run them in one session with verified-termination timers.
   - [x] 3b. Data: 2 of 75 shards of `embedded-language-flows/openwebtext-t5` on CARC (`/scratch1/<user>/data/owt_t5`; state.json restricted to them, original kept as state_all75.json).
     - Fine-tune runner `src/elf_ft_decode.py`: patches hcfrl's train_step in memory, so hcfrl itself is untouched. Smoke test passed for both arms; init_from loaded ELF-B (step 95085).
     - Decomposition `src/elf_access_decomp.py` plus routed decode in `elf_instr.py`: smoke test passed, identities 1.0000 at n_steps 32/16 × access read/zero.
   - Original note: OWT data is NOT on CARC (only game24/countdown in hcfrl/data). Download an OWT shard for the fine-tune. Check whether ELF's OWT training data and `hcfrl/elf/train_step.py` support a short decoder-branch fine-tune (SC readable at decode w.p. 0.5 vs a zeroed control; same data, steps and seed).
   - [x] 3c-i. Write `analysis/elf_access.py` (frozen rule) and test it on synthetic scenarios (account true; readout-access-alone; training adaptation; partial identifiability; failed manipulation check).
   - [x] 3c-ii-a. Prereg §2.13 and the GPT red-team prompt written (`prompts/gpt_redteam_elf_access.md`). The rule was checked on 8 synthetic scenarios; M1/M2 failure criteria were added before freezing.
   - [x] 3c-ii-b. GPT red-team received (2026-11-20) and verified against the code.
     - **Adopted:**
       - C4 (confirmed: the judge context came from each config's own O1N1F1; now a common reference = base ELF-B SC-on tokens for the same noise);
       - C5/M12 uptake gate (decode-feedback zeroing changes ≥ 10% of tokens at n = 24, per seed) + M14 shuffled-feedback control (new mandatory M3);
       - C1 information measure (decoder log-prob of the eventual token, per seed, global and |i − j| ≤ 16);
       - M5/M15 8 fine-tune seeds per arm (MDE ≈ 0.20 nats from E3b variance vs SESOI 0.41);
       - M6 joint per-sample contrast; M7 explicit missingness (common sample set; > 5% missing → NOT TESTABLE); M10 minimum effect explicit in M1; M11 Q = Δ32 − 0.5·Δ24 tested directly; M13 positive-T gate dropped;
       - M3 continuous identity checks (logit max-diff); M9 per-seed regime; M4 neighbour-window regime;
       - M8 fluency → native decoder CE on held-out OWT;
       - minor: grid/seed asserts, explicit path parsing, boundary-case tests.
     - **Rejected:**
       - C2: the truncated state is in distribution. The decoder is trained on λ·x + (1 − λ)·5·e with λ ~ sigmoid(N(0.8, 0.8)); the flow state at t = 0.35 has noise/signal 3.7 ≈ λ 0.58 ≈ the 27th percentile.
     - **Partly adopted:**
       - C3: the account *is* information-dependent use; claim narrowed; the rivals are "access regardless of information" and "any new input" (shuffled control);
       - M1: same-state SC training is standard; limitation noted; uptake measured at inference;
       - M2: report all 8 cells + interactions.
   - [x] 3c-ii-c. E4 v2 implemented and frozen: smoke test passes in all 6 configs (token identity 1.0000; decode-logit max-diff ≤ 0.014 after fixing the check's stream selection); 10 synthetic scenarios give the intended verdicts. Implement E4 v2:
     - `elf_judge_fixedctx.py --ref`; decomposition adds access=shuf, regime log-probs, uptake rate, logit identity diffs and base-model reference tokens;
     - `src/elf_heldout_ce.py` (fluency/uptake on held-out OWT);
     - analysis v2 + synthetic/boundary tests;
     - prereg amendment; hash; push.
   - [x] 3c-ii-d. Second red-team pass skipped (2026-11-20; the user delegated): all valid points adopted, diminishing returns.
   - [ ] 3c-iii. Code and smoke-test steps 4 (LangFlow) and 5 (Plaid) decompositions on CPU so that one Prime session runs everything.
     - LangFlow: attention is the module-level `regular_attention_multi_headed`, so the same three-stream patch applies. SC = softmax(logits) @ E via `_embed_tokens`; Euler-EDM over a Gumbel-proposal gamma schedule (128 steps); the final forward uses SC.
     - Plaid: custom repo (`lib/models.py`); check the flash-attn dependency.
     - [x] `src/langflow_instr.py` (routed attention patch + plain/routed samplers) and `src/langflow_calib.py` (regime diagnostics, seed 7, + identity smoke) written; einops added to the CARC env and requirements.lock.
     - [x] `src/langflow_decomp.py` written (reuses elf_judge_fixedctx.py via T5_PATH = GPT-2 tokenizer + REF_FILE).
     - **Plaid decision (2026-11-20):** it needs apex + flash-attn v1 + mup + fused-MLP kernels, so a pure-PyTorch port with weight remapping is required. It is validated by reproducing Plaid's reported likelihood before any decomposition. Deferred to a second Prime session; the first session runs E4 + LangFlow.
     - LangFlow calibration attempt 1 failed: mechflow env transformers 5.3.0 is incompatible with LangFlow's remote code (`all_tied_weights_keys`). Rerun with the hcfrl env (transformers 4.44.2, the same as the pod pins): job 14403590. **All LangFlow jobs use the hcfrl env.**
     - [x] LangFlow calibration done: N-predicting regime (state 7% vs feedback 53% at 50% committed). The routed identity failed in bf16 (0.92), so everything now runs fp32 (1.000).
     - [x] Family G pre-registered (prereg §2.15; `analysis/langflow.py`).
     - [~] LangFlow decomposition + judge + analysis submitted on CARC GPU.
     - Earlier: LangFlow calibration running (CARC CPU) → then pre-register family G (LangFlow) with predictions conditioned on the measured regime (lesson from family T).
   - [x] 3c-iv-a. Pod files written: `pod/requirements_e4.txt` (pinned to the hcfrl env), `pod/setup_e4.sh` (revision-pinned downloads; all repos ungated and SHAs verified), `pod/run_e4.sh` (flock single driver; stages A–D; LangFlow calibration on the pod; LangFlow decomposition gated on ~/LF_GO).
   - [x] 3c-iv-b0. Prime API lists **zero GPU resources of any type** (2026-11-20 ~03:30), even after upgrading the CLI 0.5.38 → 0.22.1. Either an outage or the account balance (the CLI cannot show it; the user can check the web UI).
     - **Fallback submitted:** the full E4 chain on CARC GPUs (`tools/carc_e4.sh`; A40/A100/L40S; 16 fine-tunes → 16 decompositions + CE → 56 judge jobs → frozen analysis job 14400976 → `/scratch1/<user>/mechflow/e4run/E4_final.txt`). 91 jobs pending.
     - If Prime returns first: run `pod/run_e4.sh` there and cancel the CARC chain (never both).
     - 2026-11-21 11:30: Prime API still `{}` 8 h after the top-up, and no CARC gpu job started overnight. **Ask the user** to check the Prime web dashboard (is any GPU listed? any account notice/verification?).
     - **User topped up (2026-11-20): use Prime.** The Prime API itself still returns `{}` for availability (raw REST call), so the problem is on Prime's side, not the CLI or the balance. A laptop watcher polls every 15 min (≤ 24 h) and exits on availability. Then: launch `pod/run_e4.sh`, verify it is running, and only then cancel the CARC chain (never both).
     - Earlier note: **Prime balance $3.94 (user, 2026-11-20)**: likely why availability is empty; insufficient for E4 (~$5–10). Decision: stay on CARC unless the user tops up. LangFlow calibration moved to the CARC GPU queue.
   - [ ] 3c-iv-b. Launch when GPUs are available (no A6000 at 2026-11-20 ~03:00).
     - From the laptop: create the pod; rsync the repo at the pinned commit + hcfrl_min (hcfrl/elf_path.py + hcfrl/elf) via a CARC → laptop → pod stream; run setup + driver under nohup.
     - **Kill timer under `caffeinate -i`** (the last timer failed while the Mac slept); verify termination.
   - [ ] 3c-iv. Prime session: fine-tunes (readable, zeroed; 3,000 steps each) → decompositions (n_steps {32, 24} × access {read, zero} for readable; zero for zeroed; n = 256, seed 2) → Mistral fixed-context judge, plus LangFlow and Plaid. Verified-termination timer.
   - [ ] 3c. Pre-register the design: fine-tune readable vs zeroed × sampling t_end {1.0, truncated} × inference access, interaction prediction (insights §26). Predictions from 3a; rule checked on synthetic scenarios incl. partial identifiability; fixed-context outcome; fluency validity ≤ 20% drop; within-model readable-vs-zeroed-at-inference contrast). GPT red-team. GPU run (~6 GPU-h; Prime or CARC, the user's choice).
   - **Validity:** decode fluency drop ≤ 20% vs the matched control.
   - ~6 GPU-h.
3.9 [ ] **Prep for steps 4–5 (not data; can run now):**
   - [x] Downloaded LangFlow (HF cache) and Plaid 1B weights (`/scratch1/<user>/models/plaid/plaid1b_weights`) to CARC scratch.
   - [ ] Write the LangFlow decomposition (`src/langflow_decomp.py`, three-stream routing as in ELF) after family T's verdict (stopping rule).
4. [ ] **LangFlow 130M** (`Continuous-Rivals-Discrete/langflow-owt`; verified: expected-embedding SC, SC at readout, full sequence, L = 1024).
   - **Prediction:** N-dominant, O ≈ 0; timing tracks when neighbour states become decodable (measured on a separate calibration split).
   - **Block-locality rival:** N small.
   - ~2–3 GPU-h.
5. [ ] **Plaid 1B** (verified release): the same predictions as step 4. ~4 GPU-h.
6. [ ] Stretch: **DiffusionGemma 26B-A4B** (discrete; adaptive-acceptance confound; expensive).
- [~] C1 (Lafa SC-off final readout): largely answered by E1b/E1c; demoted.

**Process for each experiment:**
1. Draft the pre-registration (prediction, decision rule with both outcome scenarios, feasibility criterion "no identifiable mechanism" separate from "refuted").
2. Run a 16-sample microbenchmark to measure real cost.
3. **GPT red-team:** the user sends the draft rules to GPT with a "find flaws" prompt (Claude writes it).
4. Fold in valid criticism, hash and push, then launch.

**Robustness item from the red-team (Lafa):**
- [x] R-FC pre-registered (prereg §2.14). 2026-11-21: nothing started on the CARC gpu partition in 8 h (priority 4834), so moved to the debug partition via `tools/debug_sched.py tools/units_fixctx.json` (8 resumable units, ≤ 5 submitted). Toggling N changes y_<j in 10.6% of pairs; the analysis reproduces H2-N exactly on the original data. Re-score job submitted on CARC GPU → `src/results/conf_h2n_fx`, then `python analysis/fixctx.py src/results`.
- [ ] (original item) The Lafa N claim uses coh_j against the *actual* context y_<j, which N can shift slightly via j's state. Rescore the H2-N confirmation cells with a fixed context (cell O1N1F1's y_<j), as already done for ELF. First check whether per-cell tokens were saved; exploratory robustness.

**Threat checks in every run:**
- sham swaps;
- the identity cells reproduce normal sampling;
- absolute effects + interactions reported;
- commitment horizon from a separate calibration split.

### Phase 12 — Hard-to-vary programme (decided 2026-11-19; Deutsch / Popper: explanations whose every part makes a risky prediction; no patching after failures) → `assumptions.md`
- [x] Background-assumptions register (`assumptions.md`): instrument / measurement / statistical assumptions with status; scope claims NOT assumed.
- [x] **ELF decoder hypothesis (E3, §2.10):** REFUTED under its share rule; exploratory: 82% of F's absolute effect arises at decode (insights §23).
- [ ] **E3b (§2.11):** absolute measure, fresh seed 1, CARC CPU job 14391603 (~4 h).
- [ ] **Own feedback at 7B:** formulate a predictive account (when does own feedback help?) and test it on existing 7B cells first (which tokens / steps), then on fresh data.
- [ ] **Message content (D6):** linear probe of what j's state encodes about its neighbour's final token, with vs. without neighbour feedback (exploration docs).
- [ ] **Measure validity (B1):** human judgements on ~100 stratified output pairs vs. both LM judges (needs the user's labels).
- [ ] **Text-type generality (D4):** H2-N decomposition on code or math text (0.5B, exploration-style, then a pre-registered confirmation).

### Phase 10 — Localization ✅ (2026-11-18) → `insights.md` §6–§14
- [x] E1 step × layer map → late steps, mid layers (L8–11).
- [x] E1b single layers (L8 ≈ 20%) → head screen (126 heads).
- [x] Joint top-k on held-out exploration docs → **k = 26**. H\* frozen and published before circuit-validation.
- [x] **L1** (share, 1,777 docs): θ = 0.377 [0.298, 0.454] → **inconclusive** under the stricter logged rule (would pass under the original Holm, LB 0.311).
- [x] **L2** (behaviour, 1,888 prompts): **+28.9 pp [24.6, 33.3] → SUPPORTED.** Cutting H\* nearly triples cross-answer and garbled outputs.

### Phase 11 — Extensions and paper
- [ ] ELF second model (unofficial port validated against JAX).
- [ ] Better-sampler upside (exploratory).
- [ ] Read 2610.19628 App. D.5.2; re-check papers citing Lafa.
- [ ] ~~Draft the paper~~ *(not Claude's task: the user said 2026-11-18 not to work on the paper; a rough draft exists in `paper/` for reference; the writing is done by the user and co-author)*
