# mechflow — Pre-Registration (Phase 5)

**Version:** 1.4 (2026-11-13) — **FROZEN** (analysis rules; external anchor pending, see header)
**Status:** analysis rules FROZEN. Values marked **[PILOT]** are computed by the stated rule on the **exploration split** in Phase 7. They are recorded in §11 with a SHA-256 entry in `prereg_hashes.log` **before any confirmation data is generated**.
**External anchoring (required):** the local hash log is not tamper-evident. Before any confirmation data is generated, the PI posts the current prereg.md SHA-256, and each §11 entry's hash, to an external timestamped record: an OSF registration (preferred), a public gist, or an email to a co-author or advisor. The external URL or message is recorded in `prereg_hashes.log`. **No confirmation run may start until this anchor exists for the then-current version.**
**Basis:** `phase4.md` v4.3. Where this file sets a value that differs from `phase4.md`, this file governs, and the difference is listed in §13.
**Change policy:**
- Before confirmation data exists, edits are allowed but must be logged in §12 and re-hashed.
- After the first confirmation run, no edits. Departures go in §12 with date, reason and expected impact.

---

## 0. Decisions Adopted in Phase 5

| Decision | Adopted | Override |
|---|---|---|
| **Paper framing** | Phenomenon first: (1) flow LMs mode-mix where their own AR head does not; (2) SC repairs later-position coherence; (3) causal decomposition of which channel carries the repair. The M2 mechanism is the upside, not the floor. | By the PI, **only before confirmation data is generated**; logged in §12 and re-hashed. |
| **Primary family** | P1, P2, C1′, H2-N (§2), subject to the gate substitutions in §5.2. | Same. |
| **Optional** (first cut if Phase 6 slips) | M5 test; within-block χ; cloning commitment τ_i. Exploratory only. | Same. |

---

## 1. Model, Settings, Units

- **Model:**
  - **Primary:** `lafa-hf/lafa-0.5B`, snapshot `18deca3522f7884898a58d0fb2d2afb3fc91f87e`, EMA weights, bf16, bundled remote code.
  - **Replication:** `lafa-hf/lafa-7B` (snapshot pinned in §11).
- **Regimes** (`phase4.md` §4):
  - **Paper (PR):** K = 32, g = 8, noise 0.1, interventions in both CFG branches. **Early window = steps 0–1.**
  - **Mechanism (MR):** K = 32, **g = 1**, noise 4.0.
- **AR reference decoding:** `gen_mode="ar"`, greedy (the only mode the HF interface supports).
- **Clusters (the bootstrap unit):**
  - **entity pair** for bimodal tasks (all templates and orders of one pair form one cluster);
  - **source document** for natural text.
  
  Prompts, prefixes, seeds and positions are nested within clusters.
- **Seeds:**
  - PR: one fixed seed per prompt (seed 42);
  - MR: 4 seeds per prompt (1–4).

---

## 2. Hypotheses, Estimands, Predictions

All primary tests are **one-sided** against a margin m: H0: θ ≤ m vs. H1: θ > m.

### 2.1 Primary family

**P1 — Mode mixing.**
- **θ** = h_flow − h_AR: the paired difference in hybrid rate (`phase4.md` §2.3 classifier) on confirmation bimodal prompts, 0.5B, PR.
- **m = 5 pp.**
- **Prediction:** support. Expected 15–30 pp (smoke 30 vs. 0).

**P2 — SC repairs in-block dependence.**
- **θ** = the paired SC-on − SC-off difference in **dep_j** (`phase4.md` §2.2: sampled-predecessor PMI, S = 16, temperature 1), averaged over j ∈ {2, 3}, on confirmation natural-text blocks, 0.5B, PR.
- **m = 0.10 nats** (fixed now; ≈ 20% of the smoke per-position coherence effect at positions 2–3, ~0.5 nats).
- **Co-requirement** (conjunctive, not a separate test):
  - the external-judge (Mistral-7B-Instruct-v0.3) SC effect on coh_j averaged over j ∈ {2, 3} has a one-sided 95% lower bound > 0;
  - **and** the own-AR SC effect on coh_{2,3} has a positive point estimate.
  
  If P2's test passes but the co-requirement fails, P2 = **"judge-dependent"** (§7).
- **Prediction:** support; |θ| at positions 2–3 > |θ| at positions 0–1 (descriptive).

**C1′ — Feedback effect beyond the state.**
- **θ** = the mean single-step direct effect δ^D_{i→j} on μ_j (`phase4.md` §2.4) **with x_i LEACE-erased** (`phase4.md` §2.6), where:
  - δ^D = μ_j(q_i on) − μ_j(q_i off), with "off" = the Phase 6 selected candidate;
  - sources i ∈ {0, 1}, targets j ∈ {2, 3}, at the q–x window steps [PILOT] in MR;
  - pooled with **equal weight per (prompt, i, j, step) instance**;
  - on confirmation coupled bimodal prompts, 0.5B.
- **m = margin_C1 [PILOT]:** 20% of the exploration estimate of the single-step **block-level** SC effect on μ_j (all positions' feedback off vs. on, x not erased), pooled with the **same instance weights**.
  - **Floor rule:** if that exploration estimate is ≤ 0.05 nats, C1′ is removed from the primary family before confirmation (§5.2).
- **Prediction (M2):** support. R0 predicts θ < m.

**H2-N — Feedback-coordination share.**
- From the trajectory 2 × 2 × 2 (`phase4.md` §2.6: O × N × F, shadow-pass decoupled), on dep_j averaged over j ∈ {2, 3}, confirmation natural-text blocks, 0.5B, MR:
  - T = the full − all-off contrast;
  - N = the Möbius main effect of N.
- **Decision on the linear contrast θ = N − 0.20·T**, with m = 0, i.e. N-share > 0.20.
- **Also required:**
  - T's one-sided 95% lower bound > 0; otherwise H2-N is **"not estimable"**;
  - the Shapley share of N (point estimate) ≥ 0.20.
- **Prediction:** support under M2.
- **Stated prior:** O share largest (~0.4–0.6); N ~0.1–0.3; F ~0.1–0.3.

### 2.2 Secondary family (Benjamini–Hochberg, q = 0.10; one p-value each, as listed)

| ID | Test (one-sided unless stated) | p-value used |
|---|---|---|
| S1 | P1 at 7B: h_flow − h_AR > 5 pp (confirmation subset) | Bootstrap p |
| S2 | Natural text: incoherence-event rate, flow − AR > 0 (PR) | Bootstrap p |
| S3 | **Equivalence:** SC × g interaction on Δdep_{2,3} (g = 8 vs. g = 1, noise 0.1) within ±b_S3 [PILOT: max(50% of the exploration SC main effect on dep_{2,3} at PR, **0.05 nats**); frozen before confirmation] | TOST p = max of the two one-sided p-values |
| S4 | **Non-inferiority:** SC effect on dep_{2,3} at noise 0.1 minus at noise 4.0 (g = 1) < +b_S4 [PILOT: same rule as b_S3, including the 0.05-nat floor] | Bootstrap p |
| S5 | C2: δ^D (erased), coupled − entropy-matched independent blocks > 0 (MR) | Bootstrap p |
| S6 | C3: flip-toward-donor rate, minimal-pair − entropy-matched non-mode donor > 3 pp (MR, at the window steps; **if K12 triggered or the window is empty, at fixed steps {2, 6, 10, 14}**) | Bootstrap p |
| S8 | C4: share of SC's Δdep_{2,3} lost when the cross-position feedback channel is cut everywhere (PR) ≥ 10% (as the contrast lost − 0.10·T > 0) | Bootstrap p |
| S9 | PR early window: δ^D (erased) at steps 0–1 > 0 | Bootstrap p |
| S10a | End task, PR: SC gain on the mean of GSM8K / HumanEval / IFEval > 0 (on the 75% not reserved for G1) | Bootstrap p over problems |
| S10b | End task, MR: same, > 1 pp (on the 75% of each benchmark not reserved for G1) | Bootstrap p over problems |
| S11 | Hybrid ladder: (upward − downward) paired class transitions > 0 | Bootstrap p |
| S12 | **H2-N in the paper regime** (PR: K = 32, g = 8, noise 0.1; 2 × 2 × 2 with CFG, both branches; `run_block_two_pass_cfg`): linear contrast N − 0.20·T on dep_{2,3} > 0, 0.5B, confirmation natural-text documents | Bootstrap p |

- **Descriptive only (no test):** S7 (the F and O shares, with CIs).
- **Under K3:** S5, S8 and S9 rely on knockouts or channel cuts. They are marked invalid and removed from the BH family.
- **The BH family** is the remaining listed tests, minus S6 if it is promoted to primary. The family size is recorded in §11.

### 2.3 Exploratory (no inferential claims)
- M5 (stale vs. degraded-fresh);
- within-block χ;
- cloning τ_i and the margin proxy;
- fixed-point gap;
- α-scaling and cutoff baselines;
- per-category breakdowns;
- localization (circuit-validation split only).

---

### 2.4 Localization family (added 2026-11-15 by §12 amendment; before any confirmation data)

**Procedure.**
- **Discovery** (exploration split, 0.5B, MR regime):
  - Run per-head path patching of the feedback channel. For head (l, h) at target j ∈ {2, 3}, replace j's reads of the neighbours' K/V **from the shadow pass** (neighbours with feedback) by their K/V **from the off pass**, at that head only. This is the head-resolved N factor of the 2 × 2 × 2.
  - Rank heads by their effect on dep_{2,3}.
  - **k** = the smallest number of top heads whose joint patch reproduces ≥ 50% of the full N effect on exploration, capped at 10% of all heads (≤ 33 of 336). This defines **H\*** (the "feedback-reader heads").
  - k and H\* are frozen and hashed in §11 before any circuit-validation data is run.
- **Test** (circuit-validation split: 118 pairs, 1,777 documents; 0.5B, MR).

**Estimands and decisions** (Holm over {L1, L2}, familywise α = 0.05; separate from the primary family):
- **L1 (necessity and specificity).** θ = (reduction of the N effect on dep_{2,3} when H\*'s feedback-channel reads are cut) − (the mean reduction for 20 random size-matched head sets, matched on layer distribution). Decide on the share contrast: (reduction_H\* − reduction_random) / N > **0.30**. Cluster bootstrap by document.
- **L2 (behavioural).** On circuit-validation bimodal prompts (PR regime), cutting H\*'s feedback-channel reads raises the non-valid answer rate (hybrid + garble) by **> 3 pp** more than random size-matched head sets. Paired, cluster bootstrap by pair.

**Reported secondary (no test):**
- MIB CPR/CMD faithfulness curves under zero / mean / resample ablation;
- the 7B replication of H\*'s role (at the layer-normalized positions);
- base-Qwen2.5-0.5B comparison: do the same heads attend across positions before flow training?

**Outcome mapping.** L1 ✓ (± L2) → "a sparse set of heads reads neighbours' self-conditioning feedback"; mech-interp headline. L1 ✗ → "the coordination is distributed"; reported as such.


### 2.5 Mechanism family R (added 2026-11-18, before any of its data; exploratory basis: insights.md §7, §11)
**Account under test (two redundant channels):** j obtains its neighbours' identity through their states and their fed-back predictions; either channel can suffice. The feedback's causal effect is what the state channel cannot supply.

**Data:** 1,000 confirmation documents never used before (`src/tasks_gc.jsonl`; shuffled confirmation positions ≥ 7,100, excluding every document in any `tasks_conf_*` file). 0.5B, MR (g = 1, noise 4.0, K = 32), j ∈ {2, 3}, outcome dep_{2,3}.

**Manipulation:** `run_block_headcut_batched`, all layers. In window w, j reads the neighbours from a pass with: nofb (state, no feedback), nostate (feedback, noise state) or neither. Windows: late = steps 24–31, mid = steps 8–15. Grids: `warp` (default) and `uniform`.

**Tests** (Holm over 3, α = 0.05; cluster bootstrap by document, 20,000; one-sided):
- **R1** (redundancy under the default sampler): `warp`, late. Redundancy index (neither − nofb − nostate) > 0.
- **R2** (feedback dispensable once states are informative): `uniform`, late. nofb < 0.10 nats (upper one-sided 95% bound) **and** neither > 0.
  - R2 is supported only if both parts hold; p = max of the two.
- **R3** (the schedule moves the feedback effect): nofb effect at `uniform` mid minus nofb at `uniform` late > 0.

**Interpretation:**
- R1 ∧ R2 ∧ R3 → the two-channel account is confirmed.
- R2 failing on its first part → feedback matters even with informative states (the account is wrong).
- R1 failing → no redundancy (the channels are independent).


### 2.6 Family S7: the mechanism at 7B (added 2026-11-18, before any of its data; exploratory basis: insights.md §19)
**Data:** 500 confirmation documents never used before (`src/tasks_conf7b.jsonl`; shuffled confirmation order, excluding every document in any `tasks_conf_*` file and in `tasks_gc.jsonl`). lafa-7B @ 434c100d…, MR (g = 1, noise 4.0, K = 32), the 2 × 2 × 2 of §2.1 H2-N (`jobs_h2n:h2n_task`), j ∈ {2, 3}.

**Primary outcome:** coh_{2,3} under the 7B's own AR head. dep is too noisy at 7B in exploration and is reported descriptively.

**Tests** (Holm over 3, α = 0.05; one-sided cluster bootstrap by document, 20,000; `analysis/s7b.py`):
- **S7-1:** mN − 0.2·T > 0.
- **S7-2:** Shapley(N) − 0.5·T > 0 (neighbour reading carries at least half). Co-requirement: Mistral Shapley N share ≥ 0.5 (point estimate), else "judge-dependent".
- **S7-3:** Shapley(O) > 0 (a position's own feedback contributes at 7B).

**Interpretation:**
- S7-1 ∧ S7-2 → neighbours dominate at 7B.
- S7-3 → own feedback emerges with scale (it is ≈ 0 or negative at 0.5B, §11 / insights §16).


### 2.7 Family E: the mechanism in a second architecture, ELF (added 2026-11-18, before any of its data; basis: insights.md §18)
**Model:** ELF-B (OpenWebText; full-sequence continuous flow over T5-small latents; SC input [z, x_prev], off = zeros as in training), used through the authors' PyTorch implementation (read-only).

**Generation:** unconditional, L = 128, 32 logit-normal steps (generator seed 1), SC guidance token w = 1, n = 256 samples (noise seed 0), identical noise per sample across cells.

**Intervention:** `src/elf_instr.py` three-stream routing. The identity cells reproduce SC-on / SC-off sampling to float precision (verified on CPU). Targets j ∈ {32, 64, 96}, run separately.

**Outcome:** Mistral-7B-Instruct-v0.3 log-prob of the decoded text span of token j given the decoded tokens 0..j−1 (`src/elf_decomp_judge.py`). Unit: the sample (mean over the targets with complete cells).

**Tests** (`analysis/elf.py`; Holm over 2, α = 0.05; one-sided bootstrap over samples, 20,000; "not estimable" unless T's one-sided 95% LB > 0):
- **E-1:** mN − 0.2·T > 0.
- **E-2:** Shapley(N) − 0.5·T > 0.

O and F shares are descriptive.


### 2.8 Family E2: schedule manipulation in ELF (added 2026-11-18 after family E, before any of its data)
**Prediction of the confirmed two-channel account (§2.5) for ELF:** when states stay uninformative longer, neighbour information must travel through the fed-back predictions, so the direct feedback-reading route (N) grows. When states become informative early, the state route (F) dominates.

**Design:** as family E, but target j = 64 only, n = 256, under two schedules of 32 steps:
- **latejump:** t = linspace(0, 0.2, 32), then a jump to 1;
- **frontload:** t_k = 1 − (1 − k/32)³.

Outcome: Mistral span log-prob of token j. Mac MPS (as family E).

**Tests** (`analysis/elf_sched.py`; Holm over 2, α = 0.05; bootstrap within schedule, 20,000; both T one-sided LB > 0 required):
- **E2-1:** Möbius N main (latejump) − (frontload) > 0.
- **E2-2:** Möbius F main (frontload) − (latejump) > 0.


### 2.9 L1-R: replication of the localization share test (added 2026-11-19, before any of its data)
**Motivation:** L1 was inconclusive (θ = 0.377 [0.298, 0.454]; one-sided LB 0.298 vs 0.30 at α = 0.025).

**Design:** identical to L1 (§2.4, §12 2026-11-17 entry).
- **Same** frozen H\* and the **same** 20 layer-matched random sets (`src/cuts_l1.json`, sha256 prefix b8ae9e6f3bdccb16), the same rotation (4 of 20 per document), the same job (`jobs_l1:l1_task`), the same estimand θ = (reduction_H\* − mean reduction_random) / N on dep_{2,3}, and the same analysis (`analysis/l1.py`).
- **Data:** 3,000 confirmation documents never used before (`src/tasks_l1r.jsonl`; shuffled confirmation order, excluding every document in any earlier task file). 0.5B, MR.

**Decision:** unchanged, a one-sided lower bound at **α = 0.025** > 0.30 → supported. Upper bound < 0.30 → rejected. Otherwise inconclusive. No data extension.


### 2.10 Family E3: ELF decoder hypothesis (added 2026-11-19, before any of its data; basis: insights §20–21)
**Hypothesis:** ELF's state-route dominance (F ≈ 0.7, even when states stay noisy) arises at its final decoding pass, which attends across all positions' final latents. It does not arise during denoising.

**Test:** same setup as family E (ELF-B, L = 128, 32 logit-normal steps, w = 1, n = 256, noise seed 0), target j = 64. For each 2 × 2 × 2 cell, decode token j with (own) the cell's final latents and (swap) the O1N1F1 cell's final latents everywhere except position j (`src/elf_decode_swap.py`). Outcome: Mistral span log-prob of token j against the fixed O1N1F1 left context (`src/elf_judge_fixedctx.py`). Analysis: `analysis/elf.py` (Shapley shares).

**Decision:**
- F Shapley share (swap) < 0.25 **and** N the largest share → the decoder hypothesis is **supported**.
- F share (swap) ≥ 0.5 → **refuted**.
- Otherwise: inconclusive.

Platform: CARC CPU (main partition).


### 2.11 Family E3b: ELF decoder hypothesis, absolute measure (added 2026-11-19 18:20, after E3's data; basis: E3 outcome, §12)
**Why:** E3's pre-registered criterion used the F *share* within the swap variant. Shares are normalised by that variant's own total, which itself shrank 83%, so the criterion did not measure what the hypothesis claims (where F's effect arises). E3 stays REFUTED under its own rule. This is a new test on fresh data, not a re-analysis.

**Hypothesis:** most of the F route's absolute effect on token j arises at ELF's cross-position decode pass (which receives zeroed SC).

**Test:** identical to E3, with **fresh noise seed 1** (E3 used seed 0), target j = 64, n = 256, the same judge and the same fixed context.

**Analysis:** `analysis/elf_e3b.py` (frozen with this entry). φ = 1 − mean F_swap / mean F_own, paired by sample.

**Decision:**
- 2.5th pct > 0.5 → SUPPORTED;
- 97.5th pct < 0.25 → REFUTED;
- otherwise INCONCLUSIVE.

**Exploratory value on E3 (seed 0) data:** φ_F = 0.819 [0.735, 0.897].

## 3. Metrics
- **Primary:** dep_j (positions 2–3); the hybrid rate; μ_j.
- **Co-requirement:** external-judge and own-AR coh_j.
- **Secondary:** benchmark accuracy; incoherence events; agreement with AR greedy; class transitions.
- **Not primary:** per-token accuracy vs. gold.

---

## 4. Data and Splits (assignment fixed now; RNG seed 20261113)

### 4.1 Bimodal prompt bank
- **Categories (12):**
  - *exploration:* two-word US cities, countries, scientists (the smoke categories);
  - *confirmation:* authors, athletes, two-word companies, dishes, two-word films, musicians, landmarks, fictional characters, painters.
- **Pairs:** ≥ 80 per category (≈ 960 total; confirmation ≈ 720).
- **Templates:**
  - **Exploration templates** = the 5 smoke templates (`smoke/03_hybrid.py`: 3 open + 2 pair) + 1 new pair template.
  - **Confirmation templates** = 8 new "choose A or B" templates.
  - **Off-diagonal prompts** (an exploration category with a confirmation template, or the reverse) are **not generated**.
- **Orders:** both orders for pair templates. Open templates have one form (no order).
- **Tokenization filter:** as in `phase4.md` §2.3; drop rates are reported.
- **Circuit-validation:** 15% of confirmation pairs (by pair), reserved.
- **Expected confirmation size:**
  - ≈ 612 pairs × 8 templates × 2 orders ≈ 9,800 prompts before the tokenization filter;
  - **the effective n is the number of pairs (≈ 600).**
  - If Phase 7 power (§8) shows < 80% with the available pairs, add **confirmation categories** (never templates) before confirmation, and log it.

### 4.2 Natural-text blocks
- **Sources:**
  - **Primary:** **UltraChat-200k `test_sft`** split, using the first assistant turn (chat). It is not listed among Lafa's SFT sources (Dolci-Instruct-SFT).
  - **Conditional:** GSM8K **train** solutions (math). Dolci includes OpenMathInstruct-2 (built on GSM8K/MATH train problems) and persona-GSM, so contamination is possible. They are used **only if** a contamination check on the exploration split passes: own-AR coherence per block on GSM8K-train is not higher than on a held-out math source (SVAMP) by more than **0.25 nats/block**. Otherwise they are excluded (decision recorded in §11).
- **IFEval is not used for blocks** (it is an end task only).
- **Cut points:** uniformly random, block-aligned.
- **Split by document:**
  - 20% exploration, 80% confirmation (no overlap);
  - circuit-validation = 10% of the confirmation documents, reserved.

### 4.3 Exclusions
- **All 1,319 GSM8K test items** are excluded from block-level and bimodal work. They are used only as the end task (S10).
- The 434 smoke-used indices are listed in `smoke/results/EXCLUDE_gsm8k_test_indices.json` for the record.
- All smoke prompts are in exploration and are **excluded from the go/no-go check** (§5).

---

## 5. Gates

### 5.1 Gates (evaluated in order; outcomes recorded in §11)

| Gate | Phase | Rule | Consequence |
|---|---|---|---|
| **PC** | 6 | The planted one-hot signal in q_i is detected at j by single-step δ^D, by the per-query cut, and after x-erasure. Detection = effect > 10× the 95th percentile of the random-perturbation floor. | No mechanism results until it passes. |
| **K3** | 6 | `phase4.md` §2.7: no candidate passes the mixed-context Mahalanobis check, or cross-candidate agreement fails | §5.2 substitution |
| **K12** | 7 | Held-out linear probe, **MR (g = 1) only:** < 3 steps with acc_q ≥ 0.85 and acc_x ≤ 0.65 | §5.2 substitution |
| **C1 floor** | 7 | The exploration block-level single-step SC effect on μ_j ≤ 0.05 nats | C1′ removed (§5.2) |
| **K11** | 7 | **7B, exploration split:** h_flow ≤ 5% **and** the SC effect on coh_{2,3} (own-AR **and** external, both point estimates) is < ½ of the 0.5B value | The paper is reframed as 0.5B-specific; S1 reported |
| **G1** | 7 + CARC | **MR, exploration:** the SC effect on coh_{2,3} under **both** judges ≥ ½ of PR's (point estimates) **and** the end-task SC gain (mean of the 3 benchmarks), on a **25% random subset of each benchmark reserved for G1** (seed 20261113; excluded from S10a/b), has a 95% lower bound > 1 pp | If failed: MR mechanism claims are qualified "training-noise regime only"; the headline rests on P1, P2 and S9 |
| **Go/no-go** | 8 | Go if PC passes; K3 clear or the substitution is viable; and the exploration P1 point estimate on **non-smoke** exploration prompts is ≥ 10 pp | Otherwise reframe (§7) or stop |

**Candidate selection rule (Phase 6):** among single-position off candidates that pass `phase4.md` §2.7, prefer **(a) plain-x, then (c) prefix-free, then (b) bootstrap**.

### 5.2 Primary-family substitutions (Holm denominator = the family's size)

| Gate outcome | Primary family |
|---|---|
| All clear | {P1, P2, C1′, H2-N} (Holm over 4) |
| K12 or C1 floor triggered | {P1, P2, H2-N} (3) |
| K3 triggered (knockouts invalid; affects C1′ and H2-N) | {P1, P2, **S6 promoted** (C3 donor swap; m = 3 pp)} (3) |
| K3 and (K12 or C1 floor) | {P1, P2, S6} (3) |

The substitution is decided and logged **before** confirmation data is generated.

---

## 6. Statistical Plan

- **Bootstrap:**
  - cluster bootstrap (pairs or documents), **20,000** resamples;
  - **BCa** intervals for the primary family, percentile intervals for the rest;
  - paired across conditions.
- **p-values:** one-sided bootstrap p = fraction of resamples with θ* ≤ m, corrected by the BCa adjustment for primaries.
- **Primary decisions:**
  - Holm step-down on the one-sided p-values at familywise α = 0.05 → "**supported**".
  - "**Rejected**" if the one-sided 95% upper bound < m (per estimand, unadjusted).
  - Otherwise "**inconclusive**". **There is no data extension.** Inconclusive is reported as inconclusive.
- **Secondary:** BH at q = 0.10 over the secondary p-values remaining after §2.2's adjustments (11 if no gate triggers). S3 uses the TOST p.
- **Ratios:** decided via the linear contrasts given in §2 (no pre-test on the denominator, beyond H2-N's stated "not estimable" condition, which is applied before Holm and counted as a failure to support).
- **Missing or failed generations:**
  - **complete-case**: a unit is analysed only if valid under **all** paired conditions of that estimand;
  - exclusion rates are reported per condition;
  - **sensitivity analysis:** treat exclusions as failures (worst case for the hypothesis).
- **No outlier removal.**
- **Two-sided 95% CIs** are reported for all estimands, descriptively.
- **Localization (if pursued):** BH over heads, q = 0.05, on the circuit-validation split only.

---

## 7. Outcome → Interpretation → Framing (frozen; gates as in §5)

Legend: ✓ supported, ✗ rejected, ? inconclusive, n/e not estimable or removed by a gate, JD judge-dependent (P2).

| P1 | P2 | C1′ | H2-N | Interpretation | Framing |
|---|---|---|---|---|---|
| ✓ | ✓ | ✓ | ✓ | Mixing persists; SC repairs coherence partly by passing commitments across positions | **Full paper**, phenomenon-led |
| ✓ | ✓ | ✓ | ✗/?/n/e | Feedback carries mode content beyond the state; its share of the repair is small, unresolved or not estimable | Phenomenon + a minor/undetermined mechanism |
| ✓ | ✓ | ✗/?/n/e | ✓ | The feedback channel carries a substantial share; beyond-state content isn't shown (redundant with the states, unresolved, or not separable) | Phenomenon + a "feedback channel carries the repair" mechanism, with the beyond-state question open |
| ✓ | ✓ | ✗/?/n/e | ✗/?/n/e | No supported feedback-channel mechanism. If both are ✗, the repair is own-position/state-mediated; otherwise the mechanism is unresolved. If K3 triggered, S6 (donor swaps) is reported as the mechanism evidence | Phenomenon + decomposition (analysis paper), with the mechanism stated as rejected / unresolved / swap-only accordingly |
| ✓ | JD | any | any | The SC dependence gain depends on the judge | Phenomenon paper; SC's effect reported as judge-dependent |
| ✓ | ✗/? | any | any | Mixing is real; SC doesn't (demonstrably) repair dependence | Phenomenon paper |
| ✗/? | ✓ | any | any | No reliable mixing at confirmation scale | Narrower SC-coherence paper |
| ✗/? | ✗/?/JD | any | any | Neither the phenomenon nor the repair is established | Stop, or short negative report |

- Mechanism columns are interpreted only when P2 is ✓ or JD.
- S3/S4 (M3/M4) are reported in every row.

---

## 8. Power and Compute

- **Primary tests** run at Holm level, with one-sided α down to 0.0125 for the first test (z ≈ 2.24).
- **Required n** at 80% power, assuming a true effect = 2 × margin:
  - n = ((2.24 + 0.84) · SD / margin)² **clusters**;
  - the SDs are of **cluster-level mean** differences [PILOT].

| Estimand | Provisional (smoke-based) | Rule |
|---|---|---|
| P1 | Pair-level SD of the hybrid difference ≤ 0.5 → n ≤ (3.08 · 0.5 / 0.05)² ≈ 950 pairs worst case. With a pilot SD ~0.3, ≈ 340 pairs | Phase 7 SD; if > 600 pairs are needed, add confirmation categories |
| P2 | Document-level SD unknown. With a block SD ~2.5 nats and ~4 blocks per document, a document-level SD ~1.5 → (3.08 · 1.5 / 0.10)² ≈ 2,100 documents (~8,500 blocks) | Phase 7 SD |
| C1′ | Unknown | Phase 7 SD; capped by the available coupled instances |
| H2-N | By simulation from the pilot's joint (N, T) distribution | Phase 7 |

**Compute:** all confirmatory runs on CARC (A100), batched. The 7B work is restricted to K11 (exploration), S1 and a C1′ replication subset. Phase 6 benchmarks replace the estimates.

---

## 9. Software and Procedures
- Confirmatory code lives in `mechflow/src/` (Phase 6). It is hashed and recorded in §11 before the first confirmation run.
- A determinism check runs in Phase 6.
- Each result file records the model snapshot, code hash, config, seeds, split and timestamp.
- External judge: Mistral-7B-Instruct-v0.3, snapshot pinned in §11.

## 10. What Would Change Our Mind
- **Against M2:** C1′ rejected with PC passed and K3 clear, **or** H2-N rejected with the Shapley share agreeing.
- **Against the phenomenon framing:** P1 rejected at 0.5B, or K11 triggered.
- **For the absorption account at deployed settings:** S9 not supported while C1′ is supported.

## 11. Frozen-Values Log (each entry hashed in `prereg_hashes.log` before confirmation data exists)

| Item | Value | Date |
|---|---|---|
| Selected off candidate; PC; K3 | **(a) plain-x**. PC passed (planted-signal detection and cut; injected/cut KL ratio ≥ 4e8; the erasure arm is deferred to Phase 7 when LEACE exists). K3 **not triggered** (null-calibrated rule, §12): (a) pooled 0.82, exceed 5.6% vs. null 6.3%; (b)/(c) fail efficacy (0.86/0.75) | 2026-11-13 |
| q–x window steps (MR, g = 1); K12 | **Window = [] (0 steps) → K12 TRIGGERED.** Held-out pair-relative linear probe (3,036 instances; 771 test from held-out pairs; base rate 0.54): acc_q rises 0.73 → 0.99, acc_x 0.57 → 0.93. q leads x by ~0.1–0.2 at every step, but acc_q first reaches ≥ 0.85 at step 14, when acc_x = 0.71 > 0.65. → Per §5.2, the **primary family = {P1, P2, H2-N}** (Holm over 3); C1′ reported as 'not separable in Lafa' | 2026-11-14 |
| margin_C1 and the C1 floor outcome | Not applicable: C1′ removed by K12 (2026-11-14) | 2026-11-16 |
| b_S3, b_S4 | **0.25 nats** each (= max(50% × exploration PR SC effect on dep_{2,3} = 0.50, 0.05)). P2 pilot (exploration, 300 blocks): dep_{2,3} +0.50 [0.20, 0.82]; Mistral coh_{2,3} +1.44 [1.01, 1.90] | 2026-11-15 |
| Pilot SDs; required n; H2-N simulated power; bank expansion (if any) | **H2-N pilot (exploration, 196 blocks):** linear contrast N − 0.20·T on dep_{2,3} = +0.48 [0.13, 0.84] (n = 196 clusters; SE ≈ 0.18) → at ≥ 2,000 confirmation documents the expected SE ≈ 0.06, so power ≫ 80% if the effect holds. P1 pair-level pilot: 7.5 pp [5.5, 9.7] (0.5B). Bank expansion: not needed for H2-N; P1 is now secondary in practice (small-model effect, K11) | 2026-11-14 |
| LEACE maps (hash); probe accuracies | Not built: only needed for C1′ / S5 / S9, all removed (§12, 2026-11-16) | 2026-11-16 |
| Classifier audit agreement | **148 / 150 = 98.7%; hybrids 45 / 45** (stratified 45 hybrid / 55 other / 50 mode; `src/results/classifier_audit_sample.json`) ≥ 90% / 85% ✓ | 2026-11-15 |
| Prompt-bank hash; split-membership hash; natural-text document lists (hash); GSM8K-train contamination-check outcome | bank sha256 `030d096f1c224068…`, split membership `72b775ac07058809…` (exploration 240 pairs; confirmation 602; circuit-validation 118; 0 tokenization drops). Natural text: UltraChat-200k test_sft, docs `aac4f15f0c050983…`, split ids `5fa96eb55e910c84…` (exploration 4,536; confirmation 16,258; circuit-validation 1,777). **GSM8K-train excluded:** the pre-registered check (vs. SVAMP) can't be run as specified (SVAMP has no solution text) → conservative exclusion; natural text = UltraChat only | 2026-11-14 |
| 7B snapshot; Qwen judge snapshot | lafa-hf/lafa-7B @ `434c100d0fb2cce29ac72d76432da4b8a81eb0b7`; external judge **Mistral-7B-Instruct-v0.3** @ `c170c708c41dac9275d15a8fff4eca08d52bab71` (replaces Qwen, §12) | 2026-11-16 |
| K11; G1 outcomes; primary-family substitution (§5.2) | Substitution: **K12 → {P1, P2, H2-N}** (2026-11-14). **K11 NOT TRIGGERED (2026-11-15):** first conjunct met (7B hybrid rate 0.3% ≤ 5%), but the second fails. The 7B SC effect on coh_{2,3} is **4.2× (own-AR: +2.38 vs. +0.57) and 1.9× (Mistral: +2.37 vs. +1.23) the 0.5B value** on the same 100 exploration blocks, not < ½. → No reframe to 0.5B-specific; S1 reported (7B hybrids ≈ 0). **G1 PASSED (2026-11-15):** (i) coherence: MR/PR ratio of the SC gain at positions 2–3 = 1.54 (own-AR), 1.01 (Mistral) ≥ 0.5; (ii) end task (MR, reserved 25% slices, paired): GSM8K +4.2 pp [0.6, 7.9] (n = 330), IFEval +9.6 [0.7, 18.5] (n = 135), HumanEval +2.4 [−4.9, 12.2] (n = 41); **mean +5.4 pp [1.3, 9.7], LB > 1 pp** ✓ → MR mechanism claims are not regime-qualified | complete |
| Confirmatory code hash | sha256 over `src/*.py` = `83822b819778bd320857c869c0cafae02298684102e15616e132c47a6bedba3d`. Task files `src/tasks_conf_*.jsonl` (built by `build_confirm_tasks.py`), combined sha256 prefix `cbece0122778e560`. **Confirmatory n:** P2 7,100 documents (pilot SD 2.74 → §8 rule); S3/S4 arms on the first 2,000; H2-N 2,000 documents (resampled power ≈ 1 at the pilot effect, and > 0.8 at half of it); S12 750 (> 0.8 at half the pilot effect); P1 all 9,632 confirmation prompts (602 pairs); S1 (7B) all prompts of 150 random confirmation pairs (2,400). **BH secondary family = {S1, S3, S4, S11, S12} (size 5)** | 2026-11-16 |
| Localization: k, H* (head list hash) | **k = 26** (smallest top-k with joint dep_{2,3} share ≥ 0.50 on 200 held-out exploration docs: 0.507). **H\*** = L19h3, L9h13, L8h8, L8h7, L5h0, L10h2, L3h7, L12h8, L19h11, L3h1, L15h5, L15h2, L18h13, L10h11, L19h9, L8h5, L10h3, L15h3, L18h11, L18h1, L18h5, L19h1, L8h4, L8h6, L5h8, L12h6. `src/cuts_l1.json` sha256 prefix `b8ae9e6f3bdccb16` (H\*, 20 layer-matched random sets, seed 20261117); circuit-validation tasks `src/tasks_cv_l1.jsonl` (1,777 docs) prefix `2a3bd5163225cc64` | 2026-11-17 |
| Phase 8 decision | **GO** (user decision, 2026-11-16), reframed around the coordination mechanism (`phase8_memo.md`). Primary = {P1, P2, H2-N}, Holm over 3 | 2026-11-16 |

## 12. Deviations / Change Log
- **2026-11-19: E3 outcome (ELF decoder test, seed 0, n = 190 complete): REFUTED under its pre-registered rule.**
  - F Shapley share in the swap variant = 0.843 (≥ 0.5). Own-variant shares reproduce family E: N 0.18, O 0.01, F 0.82.
  - **Exploratory (not pre-registered), paired:** fixing neighbours' final latents at decode removes 83% of the total SC effect on token j (T 4.51 → 0.77; 0.829 [0.737, 0.913]) and 82% of F's absolute effect (0.819 [0.735, 0.897]).
  - The share criterion was mis-specified for the hypothesis, so a new test on fresh data is pre-registered as §2.11 (E3b). E3's verdict is not changed.
- **2026-11-19: S7 outcome (7B, 500 untouched confirmation docs; frozen `analysis/s7b.py`): S7-1, S7-2, S7-3 all SUPPORTED (Holm).**
  - T(coh) = +2.167; Shapley N share 0.856, O share 0.066.
  - S7-1 +1.012 [0.765, 1.261]; S7-2 +0.772 [0.601, 0.949], Mistral co-requirement N share 0.757 (met); S7-3 +0.143 [−0.024, 0.311], p = 0.0455 (supported at the third Holm step, α = 0.05).
  - Operations: shards 1–3 on Prime 4× A6000 (MPS); shard 0 on CARC. Shard 1 with AR_CHUNK = 4 (logged). A duplicated pod driver (after a laptop crash) computed some tasks twice; deduplicated by task_id (identical seeds).
- **2026-11-19: L1-R outcome (3,000 untouched docs; frozen `analysis/l1.py`): SUPPORTED.**
  - dep: H\* share 0.495 vs random 0.042; θ = 0.453 [0.391, 0.519]; one-sided LB (α = 0.025) 0.391 > 0.30.
  - coh: θ = 0.424 [0.385, 0.463].
  - **This resolves the inconclusive L1.**
  - Duplicate check (2026-11-19, after outcome): every duplicated task row is identical to its copy except the wall-clock field `sec` (L1-R 1,955; S7 gen 210; S7 judge 23), so deduplication cannot have changed any result.
  - Operations: Prime 4× A6000; shard 3 split into two halves; 1,957 duplicate rows from a duplicated driver were dropped by task_id.
- **2026-11-19 (S7, before shard 1's remaining data): memory-only chunked AR scoring for S7 shard 1.**
  - One long confirmation document in S7 shard 1 runs out of memory when the 16 predecessor samples are scored in one 7B batch (11 GB attention; this also stalled the shard on CARC).
  - Shard 1 (docs after its first 2) is therefore run with `AR_CHUNK=4` (scoring in 4 chunks of 4).
  - Verified on the pod (7B, bf16): sampled predecessors identical; max |Δ log-prob| 0.11 nats per sample (batch-size rounding).
  - All other S7 shards: unchunked, as before. Platform for S7 shards 1–3 (and the S7 judge): Prime Intellect 4× A6000 (MPS); shard 0 ran on CARC A40.
- **2026-11-18: family E2 outcome (ELF schedules; frozen `analysis/elf_sched.py`): NOT SUPPORTED.**
  - frontload T = +0.204 [−0.334, 0.744] fails the estimability requirement, so p := 1 for both tests.
  - **Descriptive:**
    - latejump: T +1.339 [0.674, 1.997], N main +0.063, F main +0.765;
    - frontload: N main −0.537, F main +0.209;
    - N(late) − N(front) +0.600 [−0.001, 1.224];
    - F(front) − F(late) −0.556 [−1.379, 0.254] (opposite to the prediction).
  - **The two-channel account's prediction for ELF's channel choice fails.** ELF's SC benefit runs through the neighbour-state route even when states stay noisy.
- **2026-11-18: family E outcome (ELF-B; 246 samples with complete cells; frozen `analysis/elf.py`; generated and judged on the Mac): E-1 and E-2 NOT SUPPORTED.**
  - T = +1.896 [1.490, 2.314] (estimable).
  - Shapley shares: **N 0.347, O −0.040, F 0.694**; Möbius N 0.001.
  - E-1 contrast −0.378 [−0.641, −0.118]; E-2 −0.290 [−0.502, −0.087].
  - **Reading:** in ELF, SC's benefit runs mainly through the neighbours' own feedback improving their states (F), with neighbour-feedback reading contributing only jointly with F. A position's own feedback is ≈ 0, as in Lafa 0.5B.
- **2026-11-18 (before any family E data): ELF schedule seeding.** `get_sampling_steps` in this ELF version takes no generator; the schedule is fixed by `torch.manual_seed(1)` immediately before the call (equivalent determinism), and is printed in the run log.
- **2026-11-18 (before any family E data): ELF platform.** Because CARC was congested (fair-share), family E runs on the user's Mac. Generation uses CPU/MPS in a local env matching ELF's pinned requirements (transformers 4.44, torch 2.x). The Mistral judge runs on MPS (mechflow env). The design, n, seeds and analysis are unchanged.
- **2026-11-18: family R outcome (§2.5; 1,000 untouched confirmation docs; frozen `analysis/gc.py`): R1, R2, R3 all SUPPORTED (Holm).**
  - **R1:** redundancy index (`warp` late) +0.553 [0.315, 0.791].
  - **R2:** `uniform` late nofb +0.000 (upper 95% bound 0.004 < 0.10); neither +0.206 [0.141, 0.276].
  - **R3:** `uniform` nofb mid − late +0.539 [0.414, 0.671].
  - **The two-redundant-channels account is confirmed.**
  - Operations: the `uniform` half ran on a Prime A100 (MPS); the `warp` half on CARC, with 2 shards partly on the pod (synced partial progress resumed on CARC; no duplicates).
- **2026-11-18: L2 outcome (frozen analysis; 1,888 circuit-validation prompts, 118 pairs; Apple MPS).** Non-valid rate: base 0.176, random layer-matched sets 0.200, **H\* 0.488**. **θ = +28.9 pp [24.6, 33.3]; one-sided LB (α = 0.025) 24.6 pp > 3 pp → L2 SUPPORTED.**
  - Under H\*: hybrids 0.062 → 0.116, other 0.113 → 0.372. Examples: "Chinua Achebe" → "Chin Gaiman", "Franz Kafka" → "Fran King".
  - **L1 restated transparently:** with L2 supported at α = 0.025, the original §2.4 Holm step-down would test L1 at α = 0.05, where its one-sided LB is 0.311 > 0.30. **Per the stricter rule logged before data (L1 at α = 0.025 regardless), L1 remains INCONCLUSIVE.** Both are reported; L1 is not claimed.
- **2026-11-18 (before any L2 data): L2 implementation details.**
  - **Data:** all 1,888 circuit-validation bimodal prompts (118 pairs; `src/tasks_cv_l2.jsonl`).
  - **Generation:** paper regime (K = 32, g = 8, noise 0.1, seed 42), two blocks (8 tokens) as in P1.
  - **Intervention:** at the heads of a set, every block query reads the other block positions' K/V from a no-feedback pass (`run_block_cutall_cfg_batched`; the batched no-cut baseline matches the reference sampler on tokens in 5/6 spot checks, with the differences due to bf16 batch rounding).
  - **Conditions per prompt:** base, H\*, and 4 of the 20 L1 random sets (rotated by task_id).
  - **Labels:** the audited text classifier; non-valid = hybrid + other.
  - **Decision:** θ = mean non-valid(H\*) − mean non-valid(random), cluster bootstrap by pair, one-sided lower bound at α = 0.025 > 3 pp (`analysis/l2.py`).
  - **Platform:** the whole of L2 runs on Apple MPS (the user's Mac; bf16; pinned commit). All conditions of each prompt share one batch, so the paired contrast is unaffected by platform.
- **2026-11-18: L1 outcome (frozen analysis).** dep_{2,3}: θ = 0.377 [0.298, 0.454], one-sided LB 0.298 < 0.30 → **inconclusive**. coh: θ = 0.427 [0.379, 0.475]. H\* share 0.44 vs random 0.06. Operations: one worker ran out of memory on long documents; its shard was resumed (the 8 docs computed twice keep their first copy). L2 not run (tool pending).
- **2026-11-17 (before any circuit-validation data): localization procedure details.**
  - **Discovery restricted to the 9 layers** with the largest single-layer effect in E1b (L8, L19, L9, L15, L3, L5, L10, L12, L18; 126 heads).
  - **Heads ranked by the smooth metric** (drop in the final-step log-prob of the baseline token at j) on 100 exploration docs, because per-head dep_{2,3} is too noisy.
  - **k selected on a disjoint set of 200 exploration docs**, using the pre-registered dep_{2,3} criterion. Grid: k ∈ {1, 2, 3, 5, 8, 10, 15, 20, 25, 26, …, 33}.
  - **Random sets:** layer-matched (the same number of heads per layer as H\*), drawn without replacement from the non-H\* heads of those layers.
  - **L1 design:** all 1,777 circuit-validation docs. Each doc runs base, the full cut, H\*, and 4 of the 20 random sets (rotated: sets (4·task_id + r) mod 20, r = 0…3). The random-set mean is estimated across docs.
  - Batched head-cut implementation (validated: identical tokens to the unbatched one), run on a Prime Intellect A100 (pinned commit, locked env).
  - **α:** L2 needs a guidance-on head-cut tool that does not yet exist. So L1 is decided at one-sided α = 0.025 (Holm's first step over {L1, L2}), conservatively, whether or not L2 is run later.
  - Analysis: `analysis/l1.py` (hashed with this entry).
- **2026-11-16 (during confirmation; no outcome inspected): P1 (0.5B) runs on Apple MPS, not CARC.** CARC was congested, so the full P1 set (9,632 prompts; same code, task file, 8-shard layout, bf16) runs on the user's Mac (MPS), as G1 did. All three conditions of each prompt run on the same device, so the paired contrast is unaffected. Bit-level outputs can differ from CUDA; the device is recorded.
- **2026-11-16 (during confirmation; no outcome inspected): memory-only fix to AR scoring.**
  - **The problem:** `ar_logprob` / `sample_prefixes` (`jobs_h2n.py`, also used by P2 and S3/S4) materialized full [16 × length × 152k] logits. Confirmation documents reach 2,855 tokens (exploration max 882), so workers ran out of GPU memory and their shards stalled.
  - **The fix:** the same forward path (embed → t = 1 → backbone), with `lm_head` applied only at the needed position.
  - **Verified on a CARC A40 (bf16):** sampled predecessor sequences identical 12/12; max |Δ log-prob| = 0.0 (bitwise equal).
  - Results generated before the fix are kept, since the computation is identical. New code sha256 = `ab8345e9f5384a60469b522b79f9bc311a0cba60f2fa56f07d40fc4f3a99e428` (the frozen hash `83822b81…` covers the earlier results).
  - No analysis rule, n or estimand changed.
- **2026-11-16 (before any confirmation data): secondary-family scope and provenance fix.**
  - **Removed from the BH family:**
    - **S5, S9** (need the C1′ machinery and LEACE, which are moot after K12);
    - **S6** (the C3 donor-flip tooling was never built; C1′ is not separable);
    - **S8** (largely redundant with S12, the N main effect at PR);
    - **S2** (no natural-text incoherence-event classifier was built or audited);
    - **S10a / S10b** (end task on the 75% slices would cost ~3× the G1 run per regime; G1 already gives the held-out MR end-task result, and Lafa reports the PR SC ablation).
  - Final BH family = {S1, S3, S4, S11, S12}.
  - **Provenance:** `run_meta` hard-coded the 0.5B model name, so the K11 / 7B exploration result headers say 0.5B. Those runs did use lafa-7B (6× per-task time; 348-tensor checkpoint; distinct outputs). Fixed: the header now reads LAFA_REPO / LAFA_REV and the regime env vars.
- **2026-11-16 (before any confirmation data): S12 added** to the secondary family: the H2-N contrast in the paper regime (guidance on). Exploration pilot (286 blocks): dep N share 0.90 (Möbius), contrast +0.39 [0.16, 0.65]; coh N share 0.70, contrast +0.44 [0.24, 0.65]. The BH family size becomes 12 (if no gate removes tests).
- **2026-11-15 (before confirmation data): classifier audit + one implementation fix.** Audit of 150 stratified labels against the §2.3 definition: 98.7% agreement, hybrids 100%. Fix: a complete valid answer followed by extra words (e.g. "Gregor Mendel would") is labelled as that mode (it was 'other'). The hybrid definition (prefix of one mode + suffix of the other, at token boundaries) is **unchanged**; messier multi-way mixes remain 'other' (a conservative undercount, reported descriptively).
- **2026-11-15 (before any confirmation data): localization family added (§2.4).** Head-level discovery on exploration; test on the circuit-validation split; L1 (necessity/specificity vs. random size-matched heads, share contrast > 0.30) and L2 (behavioural, > 3 pp); Holm over {L1, L2}. Motivation: the exploration H2-N pilot (N share 0.7–0.9 at 0.5B) makes component-level localization a planned, confirmatory question rather than a post-hoc one.
- **2026-11-14 (Phase 7, exploration only): K12 probe implementation.**
  - **First implementation:** a single logistic probe on raw x/q vectors predicting "mode A vs. B" across pairs. It was **invalid**: "A" is a different token in every pair, so no single direction generalizes across pairs. Diagnostic: acc_q was only 0.64 at the final step, where q *is* the readout. It was discarded without being used for any decision.
  - **Replacement (used for K12):** a pair-relative linear probe. Features per instance are cos(v, E_a) − cos(v, E_b), the projection on the normalized (E_a − E_b), and |v|. Standardized logistic regression, trained on 75% of pairs and tested on 25% held-out pairs. This implements the pre-registered "held-out linear probe" faithfully.
  - **Robustness of the K12 conclusion:** any richer x-probe can only raise acc_x, which makes the window (acc_x ≤ 0.65) harder to meet, so the trigger is conservative in the hypothesis-neutral direction.
- **2026-11-13 (Phase 6, before any confirmation data): clarification of the K3 Mahalanobis rule (`phase4.md` §2.7(i)), fixed BEFORE seeing the calibration result.**
  - **Why:** the rule did not say whether the threshold applies per (layer, step, position) cell or pooled. The first exploration run (120 blocks) gave candidate (a) a median cell ratio of 0.76, a max of 1.10, and 6% of 144 cells > 1. With ~60 reference samples per cell, the per-cell maximum is noise-dominated.
  - **Rule adopted:** candidate (i)-passes if **both**
    1. its **pooled** ratio (the 95th percentile of nearer-reference distances over all cells, each normalised by its cell threshold) is ≤ 1.0; **and**
    2. its fraction of exceeding cells is ≤ that of an **in-distribution null** + 0.05. The null is held-out fully-on runs with a different noise seed, scored with the identical procedure.
  - Computed on exploration blocks only; the outcome is recorded in §11.
- **2026-11-13 v1.3 → v1.4** (Phase 5.5, before any data generation):
  - **external judge changed from Qwen2.5-7B-Instruct to Mistral-7B-Instruct-v0.3** (Apache-2.0, ungated). Lafa was initialized from Qwen2.5-Instruct, so a Qwen judge isn't independent;
  - **UltraChat is the primary natural-text source**; GSM8K-train is conditional on a contamination check, because Dolci-Instruct-SFT includes OpenMathInstruct-2 and persona-GSM;
  - scoop check (2026-11-13): no new work on mode mixing/coherence in Lafa found.
- **2026-11-13 v1.2 → v1.3** (final review): the last lattice cell (P1 ✗/? with P2 JD) is mapped. **The reviewer confirmed freeze-ready**, with the PI's external anchor as a precondition for confirmation runs.
- **2026-11-13 v1.1 → v1.2** (before any confirmation data; review round 2):
  - lattice rows cover n/e and mixed ✗/? combinations;
  - S6 fallback steps when K12 triggers;
  - external hash anchoring required before confirmation;
  - b_S3/b_S4 floor of 0.05 nats;
  - S5/S8/S9 invalid under K3;
  - BH family size rule;
  - G1 uses a reserved 25% benchmark slice, separate from S10.
- **2026-11-13 v1.0 → v1.1** (before any confirmation data; methodological review). Changes:
  - margins fixed (P1 5 pp, P2 0.10 nats) and listed vs. `phase4.md` in §13;
  - clustering by pair and by document;
  - power at Holm-level z per cluster;
  - gate substitutions (§5.2) and a complete lattice (§7);
  - **sequential extension removed**;
  - one-sided Holm on bootstrap p-values (BCa, 20k resamples);
  - H2-N decided on the linear contrast;
  - P2 co-requirement specified;
  - S3/S4 bounds frozen from exploration; BH family enumerated;
  - complete-case exclusions;
  - G1/K11 judges and splits named;
  - all GSM8K test items excluded from block work; UltraChat named; IFEval end-task only;
  - template / off-diagonal rules;
  - 12 categories × ≥ 80 pairs;
  - go/no-go on non-smoke prompts;
  - overrides allowed only before confirmation data;
  - hash log.

## 13. Differences from `phase4.md` (this file governs)
- **P1 margin:** 5 pp. `phase4.md` §5.2 lists 3 pp for hybrid-rate contrasts; that 3 pp now applies to the secondary S6 only.
- **P2 margin:** fixed at 0.10 nats instead of "25% of the pilot dep effect", because that rule would have derived the margin from the target effect.
- **H2-N** decided on the linear contrast N − 0.20·T (equivalent to the share > 0.20 when T > 0).
- The sequential extension is not used.
