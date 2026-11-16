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

### Phase 9 — Confirmatory runs (confirmation split; circuit-validation untouched)
- [ ] **P1** (0.5B, PR): 9,632 prompts / 602 pairs → `results/conf_p1` (also gives S11).
- [ ] **P2** (0.5B, PR, + Mistral judge): 7,100 documents → `results/conf_p2`.
- [ ] **S3 / S4** arms (g = 1 at noise 0.1 and 4.0): first 2,000 P2 documents → `results/conf_p2x`.
- [ ] **H2-N** (0.5B, MR, 2 × 2 × 2): 2,000 documents → `results/conf_h2n`.
- [ ] **S12** (0.5B, PR, 2 × 2 × 2 with CFG): 750 documents → `results/conf_s12`.
- [ ] **S1** (7B P1): 2,400 prompts / 150 pairs → `results/conf_s1_7b`.
- [ ] Analysis exactly as pre-registered:
  - primary: Holm over {P1, P2, H2-N}, cluster bootstrap with 20,000 resamples;
  - secondary: BH q = 0.10 over the 5 tests;
  - report the §7 interpretation; commit and push.

### Phase 10 — Localization (circuit-validation only after k, H* are frozen)
- [ ] Layer sweep → head screen on exploration/confirmation docs → freeze k and H\* in §11 and push → L1/L2 test on circuit-validation.

### Phase 11 — Extensions and paper
- [ ] ELF second model (unofficial port validated against JAX).
- [ ] Better-sampler upside (exploratory).
- [ ] Read 2610.19628 App. D.5.2; re-check papers citing Lafa.
- [ ] Draft the paper (`academic-writing` + `icml` guidance); brief the co-author.
