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
| margin_C1 and the C1 floor outcome | — | — |
| b_S3, b_S4 | — | — |
| Pilot SDs; required n; H2-N simulated power; bank expansion (if any) | — | — |
| LEACE maps (hash); probe accuracies | — | — |
| Classifier audit agreement | — | — |
| Prompt-bank hash; split-membership hash; natural-text document lists (hash); GSM8K-train contamination-check outcome | bank sha256 `030d096f1c224068…`, split membership `72b775ac07058809…` (exploration 240 pairs; confirmation 602; circuit-validation 118; 0 tokenization drops). Natural text: UltraChat-200k test_sft, docs `aac4f15f0c050983…`, split ids `5fa96eb55e910c84…` (exploration 4,536; confirmation 16,258; circuit-validation 1,777). **GSM8K-train excluded:** the pre-registered check (vs. SVAMP) can't be run as specified (SVAMP has no solution text) → conservative exclusion; natural text = UltraChat only | 2026-11-14 |
| 7B snapshot; Qwen judge snapshot | — | — |
| K11; G1 outcomes; primary-family substitution (§5.2) | Substitution: **K12 → {P1, P2, H2-N}** (2026-11-14). K11, G1: pending | partial |
| Confirmatory code hash | — | — |
| Phase 8 decision | — | — |

## 12. Deviations / Change Log
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
