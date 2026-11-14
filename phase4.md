# mechflow — Phase 4: Consolidated Hypotheses, Definitions and Design (v4.3)

Date: 2026-11-13. **This is now the working spec.** It supersedes `phase1.md` where they conflict. `phase1.md` stays as the historical record, and its machinery is reused by reference (§ numbers given).

Inputs:
- `phase1.md`, `phase2.md`, `phase3.md`;
- `smoke_memo.md` (Phase 3.5);
- the q–x diagnostic (§0.2; `smoke/05_qx_gap.py`);
- one adversarial review round (§10).

---

## 0. What Changed Since Phase 1, and Why

### 0.1 Summary

| Change | Source |
|---|---|
| The project is a **decomposition of self-conditioning's benefit**, focused on **M2 (cross-position coordination via feedback) vs. M1/M5 (own-position stabilization) vs. R0 (coordination via states)**. | Phases 2.5–3.5 |
| **M3 (guidance anchoring) and M4 (low-noise tie-breaking): "not supported (underpowered)"**, not "dropped". Smoke CIs overlap, which is not equivalence. Both stay as **measured secondary interactions** (g and noise factors in Exp F), with equivalence tests at scale. | smoke §1–3; review #7 |
| **Phenomenon (P1):** ~30% word-level hybrids on bimodal prompts at paper settings vs. 0% for the same checkpoint's AR head. **Novelty check: no published measurement of hybrid rates in a real continuous flow LM, or of flow vs. own-AR coherence** (only toys / illustrations: FMLM Fig. 3, CoCommit, ParallelBench, Zhang et al. 2602.00286). | smoke §2; novelty §9 |
| SC's measured benefit is **within-block coherence at later positions** under the own-AR judge. *Caveat (review #4):* own-AR coherence also rewards **typicality**, so validity needs a co-primary external judge and an in-block dependence measure (§2.2). | smoke §3 |
| **New, from the q–x diagnostic: at the paper's noise (0.1), the state x_k carries the same mode information as the fed-back q within ~2 steps.** At training noise (4.0), q leads x by ~15–20 steps. → **The M2-vs-R0 mechanism test runs primarily at noise 4.0** (§0.2, §4). | `smoke/05_qx_gap.py` |
| Asynchronous time: lead–lag isn't evidence. **Matching on τ = matching on (position, step)** (the warp is content-independent) → vary decodability by **intervening on x_i**, not by stratifying. | phase2; review #5 |
| Minimal-pair donors; drift-test redesign (review #3); classifier alignment (review #8); per-estimand margins and power (review #2); M5 test redesign (review #6). | below |
| Only B = 4 checkpoints are released → the block-size test is dropped. | HF `lafa-hf`, 2026-11-13 |

### 0.2 The q–x information-gap diagnostic (2026-11-13; exploratory, smoke pair prompts, n ≈ 86 position-instances per config)

**Setup**
- Decodability = P(sign of the cosine score toward mode A vs. B predicts the final mode), for the fed-back q_{k−1} (as qE) vs. the state x_k.
- Taken at the distinguishing positions, SC on, conditional branch, K = 32.

| Config | Signal | k = 0 | 2 | 4 | 8 | 12 | 16 | 20 | 24 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|
| g1, n0.1 | q | .77 | .83 | .86 | .91 | .91 | .91 | .93 | .94 | 1.0 |
| | x | .51 | .79 | .84 | .86 | .91 | .92 | .91 | .92 | .95 |
| **g1, n4.0** | q | .78 | .86 | .89 | .90 | .92 | .95 | .95 | .98 | 1.0 |
| | x | .54 | **.57** | **.59** | **.59** | **.63** | **.67** | .75 | .85 | .98 |
| g8, n0.1 | q | .79 | .88 | .93 | .96 | .99 | 1.0 | 1.0 | 1.0 | 1.0 |
| | x | .47 | .88 | .88 | .91 | .95 | 1.0 | 1.0 | 1.0 | 1.0 |
| g8, n4.0 | q | .78 | .87 | .89 | .90 | .94 | .99 | .99 | .98 | 1.0 |
| | x | .52 | .63 | .71 | .85 | .89 | .90 | .94 | .99 | 1.0 |

**Reading**
- At noise 0.1, the "q confident, x not decodable" window is essentially **steps 0–1**. Consistent with the code: with x₀ ≈ 0, x_k is a recency-weighted average of past qE.
- At **noise 4.0 (the training noise scale), the window spans ~steps 2–20 at g = 1**. This is where M2 (neighbours read q) and R0 (neighbours read x) make different predictions. It also fits the smoke finding that SC's coherence benefit is larger at noise 4.0 (+6.4 vs. +3.6 nats).
- *Caveats:* a cosine-sign decoder (a linear probe may decode x better); bimodal prompts only; n is small. **Re-run with a linear probe and more prompts in Phase 7.**

---

## 1. Claims

### 1.1 The question
SC makes Lafa's parallel blocks more internally coherent. **Through which channel?**
- **M2:** neighbours read each other's fed-back predictions within a step.
- **R0:** neighbours read each other's *states*, and SC only makes those states better.
- **M1/M5:** each position's own feedback stabilizes itself.

These channels are **not exclusive**. The output is a set of magnitudes per channel, tested against margins.

### 1.2 Phenomenon claims (P), descriptive; confirm at scale (Exp F)

| ID | Claim | Status |
|---|---|---|
| **P1** | At published settings, flow decoding produces cross-mode hybrids on bimodal prompts at a substantial rate. The same checkpoint's AR decoding does not. | Smoke 30% vs. 0% (0.5B, n = 53). Confirm at n ≈ 2,600 prompts (§6), 7B, natural text. |
| **P2** | SC improves in-block coherence, concentrated at later positions, under **both** the own-AR judge and an external judge, and in the **in-block dependence** measure (§2.2). | Smoke: own-AR only (+1.04 nats/block; positions 2–3 ≈ 92% of the gain). |
| **P3** | The SC effect is not substantially moderated by g, and is not larger at low noise. | Smoke: point estimates only. Equivalence tests at scale. |

### 1.3 Mechanism claims

**M2: feedback-channel coordination (headline candidate).** Within a step, position j's prediction depends on neighbour i's fed-back q_i **beyond what i's state x_i carries**, and this dependence carries mode content. Sub-claims, using `phase1.md` §1.2 C1–C4 with amendments:
- **C1′ (direct, beyond the state):** the single-step direct effect δ^D_{i→j} (§2.6) on μ_j is > δ_μ, **in the regime where q_i is decodable and x_i is not** (noise 4.0, steps ~2–20; §0.2).
  - *Additionally:* **intervene on x_i** (replace x_i with a mode-neutral state at matched norm) while keeping q_i. M2 predicts the q_i → j effect persists. R0 predicts j loses the mode information.
- **C2 (specificity):** the direct effect is larger on coupled than on **independent** blocks matched on per-position entropy.
- **C3 (content):** minimal-pair donor q_i moves j toward the donor's mode, beyond entropy-matched non-mode donors and the random floor.
- **C4 (significance):** cutting the feedback channel across positions (per-query K/V, own feedback kept) costs ≥ δ_C4 of SC's coherence and end-task gain.

**R0: state-channel coordination.** Cross-position agreement flows through x. SC helps because better own-predictions make better states.
- Predictions: δ^D ≈ 0 beyond the state; the x_i-neutralization intervention removes j's mode information even with q_i intact.

**M1: own-position, within-step.** j's own single-step feedback removal (neighbours' streams frozen to clean, §2.6) accounts for most of the SC effect at j.

**M5: own-trajectory closeness** (Shabalin et al.). Own feedback helps by keeping successive predictions close.
- **Test (secondary/exploratory; §2.6):**
  - **stale** own feedback q_{k−2}^j vs. **degraded-fresh** q_{k−1}^j, matched in mode decodability and entropy (equal informativeness, different temporal closeness);
  - **M5 predicts stale < degraded-fresh by more than δ; M1 predicts them ≈ equal.**
- At the H2 level, M1 and M5 are pooled as the O share of the 2 × 2 × 2.

### 1.4 Scope
- Lafa B = 4. 0.5B primary for mechanism; 7B for P1–P2 and the C1′ headline.
- Teacher-forced prefixes for C1′–C3; free generation for P1, C4 and the end task.
- **Origination is back in scope as a secondary question** (review #7): does SC help *break* the symmetry? The noise-4.0 regime, where x is uninformative for longer, is the natural test.

---

## 2. Operational Definitions

### 2.1 "Feedback off"
- **Block-level off (in-distribution):** `self_cond=False`, so the input is plain x at all positions.
- **Single-position off at position i**, as candidates:
  - **(a) plain-x_i:** no W₁/W₂ term at i.
  - **(b) bootstrap-q_i:** q_0^i. *Caveat:* it carries prefix-driven mode information.
  - **(c) prefix-free q_i:** the prediction from a prefix-masked pass at the same step.
    - *Under CFG (g > 1):* the unconditional branch is itself prefix-free, so (c) is identity there. **In the unconditional branch, use (a) for both.**
    - *At g = 1:* one extra prefix-masked pass per step.
- **Implementation:** wrap `_apply_self_cond(z, e_prev)` (FM ≈1185; hook validated in `smoke/05_qx_gap.py`).

### 2.2 Outcomes: coherence, validity, dependence
- **Own-AR coherence:** coh_j(y) = log p_AR(y_j | c, y_{<j}), using the same checkpoint's AR head (`smoke/04_coherence.py`). Measures *the model's own* consistency. **Also rewards typicality.**
- **External coherence (co-primary):** the same quantity under Mistral-7B-Instruct-v0.3 (changed from Qwen2.5-7B in prereg v1.4: Lafa is Qwen-initialized), computed on the **detokenized** block and re-tokenized, summing over tokens that cover each Lafa token's character span.
  - **Rule:** a claim about coherence requires the same sign in both judges. Sign disagreement → reported as judge-dependent.
- **In-block dependence (primary for M2): PMI against sampled predecessors.**
  - **dep_j(y) = coh_j(y) − log[(1/S) Σ_{s=1..S} p_AR(y_j | c, y′^{(s)}_{<j})]**, with y′^{(s)} ~ p_AR(· | c) sampled (temperature 1) for positions < j, and S = 16.
  - dep_j > 0 means y_j fits *its own* predecessors better than predecessors drawn from the model's distribution. It is not biased toward or against AR-greedy blocks; v4.1 compared against greedy predecessors and was biased against SC.
  - Implementation: S extra AR passes per block, batched.
- **Change-conditional reporting:** always report both (i) the rate at which a condition changes the output, and (ii) the outcome conditional on change.

### 2.3 Hybrid classifier (bimodal tasks)
1. Generate **two blocks** (8 tokens), so the full answer span is scored and block-boundary truncation ("Ada Lov") is avoided.
2. Tokenize modes A, B **in context** (prompt + answer → answer span) and extract the output's answer span (up to the first newline, period or EOS).
3. **Align** the output span to each mode with token-level edit distance.
4. Classes, in this **precedence order:**
   1. **mode-A/B:** exact match to a mode span.
   2. **hybrid:** the output equals a concatenation of a non-empty prefix of one mode's span and a non-empty suffix of the other's, possibly split *within* a word at a token boundary. ("San York", "Albert"+"aac".)
   3. **other-valid:** a well-formed answer that isn't A or B. External-judge coherence must be within 3 nats of the better mode's, **and** it must pass a category check (an entity list or an external judge yes/no).
   4. **garble:** contains a non-mode token at a distinguishing position, with external coherence < log 0.01 at that token.
   5. **invalid:** everything else.
5. **Validation before freezing:** a manual audit of ≥ 150 labels, stratified by class, with agreement ≥ 90% overall and ≥ 85% on hybrids. Otherwise revise and re-audit.

### 2.4 Mode margin
μ_j = log q_j(a_j) − log q_j(b_j) at a distinguishing position j, signed toward the full run's final mode A. Runs that end in a hybrid or garble have no A; they are excluded and their rate reported (`phase1.md` §1.4).

### 2.5 Commitment and susceptibility
- **Cloning commitment:**
  - At step k, re-noise **along the forward process at each position's own time**: x_k^{i,(r)} = x_k^i + (1 − τ^i(t_k)) · ε · σ_emb · ξ, with ε ∈ {0.1, 1, 4} swept (0.1 = inference scale, 4 = training scale).
  - **R ≥ 32 clones.**
  - φ_i(k) = fraction of clones keeping the final token_i.
  - τ_i = the first step after which φ_i ≥ 0.775 at **all** evaluated later steps. Clone at every 2nd step; report the grid.
- **Margin proxy:** as in v4 (argmax stable and log-margin ≥ log 4); validated against cloning by rank correlation.
- **Susceptibility (within block, mode flips only):**
  - For each block, over clones, f_i^r = 1[clone r's final token_i flips to the *other mode's* token at i].
  - **χ_b(k) = Σ_{i,j∈D} Cov_r(f_i, f_j) / Σ_{i∈D} Var_r(f_i)**, computed **per block**, then averaged over blocks with Var > 0.
  - **χ > 1:** joint flips.
- **State decodability:** linear probe (preferred) or cosine sign for mode A vs. B from x_k^i, fit on the exploration split.

### 2.6 Interventions
- **Single-step direct effect δ^D_{S→j}:** `phase1.md` §1.5. With x_k fixed, this equals μ_j(P) − μ_j(P_S^off), where P_S^off uses the chosen single-position off at S.
- **x-neutralization (C1′):** remove mode information from x_k^i with **LEACE-style linear concept erasure**:
  - fit on the exploration split, per (position, step bucket), against the mode-A/B label;
  - **verify** that the held-out linear probe on the erased x is at chance (≤ 0.55);
  - keep q_i, and compare j's μ_j with q_i intact vs. q_i also off.
  
  **Controls:** (i) random-subspace erasure of equal rank (floor); (ii) the OOD check of §2.7(i) on the erased states. Erasure creates a q/x conflict never seen in training (R2), so report the R2 diagnostics alongside.
- **Own-only removal (M1):** j's own feedback off at step k, with **neighbours' streams frozen to the clean pass** (removes the j → neighbours → j loop).
- **M5 vs. M1 (secondary/exploratory):** two arms for j's own feedback:
  - **(S) stale:** q_{k−2}^j;
  - **(N) degraded-fresh:** q_{k−1}^j transformed with **two knobs**, temperature T and a uniform-mixing weight λ, solved jointly so that both its mode decodability and its entropy match q_{k−2}^j's (Phase 6 detail).
  
  Equal informativeness, different temporal closeness. **M5 predicts S < N; M1 predicts S ≈ N.** Closeness is never used as a mediator (§2.9).
- **Trajectory 2 × 2 × 2 (the share decomposition for H2)**, per target j and over the whole trajectory:
  - **O:** j's own feedback on/off;
  - **N:** what j *reads* from the neighbours. At every step, two passes are run for the neighbours: a **shadow pass** in which each neighbour receives its **own current prediction** as feedback, and an **off pass** without feedback.
    - **N-on:** j reads the neighbours' K/V from the shadow pass.
    - **N-off:** j reads them from the off pass.
    - This holds **in every cell, including F-off cells**, where the "current prediction" is whatever the neighbour actually predicted in that cell.
  - **F:** what drives the neighbours' *own evolution*. Their outputs (hence their state updates and next-step predictions) come from the feedback-on pass (F-on) or the off pass (F-off). This is the **R0 / state route**: neighbours' SC improves their states, which j then reads.
  - N and F are thereby decoupled, as in `phase1.md` §1.5 (F × R). N's main effect is well-defined from the all-off baseline and is not forced to zero (fixes v4.2).
  
  All-off = block-level off and all-on = full. Möbius terms (3 main effects + 3 pairwise + 1 three-way) **sum exactly** to the total full − block-off effect on j.
  - **M2 ↔ N share; R0 ↔ F share; M1/M5 ↔ O share.**
  - Report all interaction terms; the **denominator** is the 2 × 2 × 2's own full − all-off contrast.
  - **Robustness (pre-registered, secondary):** also report each factor's **Shapley value** over the 8 cells. Claims require the Möbius main-effect share and the Shapley share to agree in sign, with both ≥ δ_share for a "support" call.
  - This replaces the v4.1 O × N 2 × 2 (which left out the F route) and subsumes the `phase1.md` F × R design.
- **Donors (C3):** minimal-pair steered prompts P^A / P^B, length-matched; donor q_i from P^B at the same k and position; entropy-matched non-mode donor; random floor.

### 2.7 Instrument validation (tests the actual risk: the mixed on/off context)
For each single-position off candidate (a)/(b)/(c):
- **(i) Mixed-context OOD check.**
  - At every layer, compute the residual-stream state of position i **and of each neighbour** under single-position off.
  - Measure its Mahalanobis distance to two reference distributions: the full-on run and the block-off run (fit on the exploration split, per layer and position).
  - **Requirement:** the 95th percentile of the distance to the nearer reference is ≤ the 95th percentile of the reference's own in-distribution distances × 1.5.
- **(ii) Cross-candidate agreement** (as `phase1.md` K3): the estimated single-step δ^D at positions 2–3 must agree in sign, and in magnitude within a factor of 2, **across candidates that pass (i) and have equal efficacy (iii)**. Candidate (b) keeps prefix-driven mode information and may legitimately show a weaker effect, so it is **reported separately** and excluded from the agreement rule.
- **Covariance estimation for (i):** Ledoit–Wolf shrinkage, with ≥ 5·d samples per reference where feasible (d = 896 at 0.5B), or a PCA to the top components explaining 95% of the variance.
- **(iii) Efficacy for (b) and (c):** applied at all positions, they should approximate block-level off (token agreement ≥ 0.9). For (a) this holds by construction (plain x everywhere = `self_cond=False`), so it isn't a test.
- **Never select on the neighbour effect.**
- **K3 triggers** if no candidate passes (i), or if (ii) fails → knockout instruments are invalid → swap-only designs (C3, x-neutralization).

### 2.8 Positive control (new)
- Inject a synthetic, strong signal into q_i: a one-hot distribution on a rare token z placed at neighbour i, at a mid step.
- Verify that the instruments (δ^D, x-neutralization, per-query K/V cut) detect a change in j's distribution and attribute it to the q-channel.
- If they can't detect a known path, results from them aren't interpretable.

### 2.9 Fixed-point gap (descriptive only)
g_k^i = ‖q_k^i − q_{k−1}^i‖₁, reported per position. It is **not** used for matching or as a mediator, because it is post-treatment (review #6).

---

## 3. Hypotheses (replaces proposal §4)

- **H1 (phenomenon):** P1–P3. Exp F.
- **H2 (decomposition):** in the noise-4.0 mechanism regime, SC's effect on positions 2–3 decomposes, in the 2 × 2 × 2, into own (O), neighbour-feedback channel (N), neighbour-state route (F) and interaction shares that sum to the total. **M2 predicts the N share ≥ δ_share**.
- **H3 (beyond-the-state):** C1′. The neighbour q-effect persists under x-neutralization.
- **H4 (dynamics):** SC raises within-block χ at intermediate k, and SC-off delays τ_j at positions 2–3.
- **H5 (generalization):** H1–H3 at 7B; bimodal and natural text.

### 3.1 Discriminating predictions (magnitudes, not "≈ 0"; the architecture guarantees nonzero leakage)

| Test | M2 | R0 | M1/M5 |
|---|---|---|---|
| Neighbour direct effect δ^D on μ_j, **with x_i neutralized** | ≥ δ_μ | < δ_μ (the information was in x) | < δ_μ |
| j's mode information after x_i neutralization, q_i on vs. off | Large difference | Small | Small |
| N share in the 2 × 2 × 2 (positions 2–3) | ≥ δ_share | < δ_share | < δ_share |
| F share (state route) | Any | ≥ δ_share | < δ_share |
| O share | Any | Any | ≥ δ_share and the largest |
| Stale (S) vs. degraded-fresh (N) own feedback, matched informativeness | — | — | M5: S < N by > δ; M1: S ≈ N |
| Within-block χ, SC on − off, intermediate k | > δ_χ | May also be > 0 (states coordinate). **Not discriminating alone**; used only together with the rows above | ≈ 0 or < δ_χ |
| dep_j (in-block dependence, sampled-predecessor PMI), SC on − off | > δ_dep | > 0 possible | Smaller than its effect on coh_j. **Not discriminating alone**; read together with the share rows |

**Asymmetry (earlier → later)** is expected under every account (schedule) and is **not** used as evidence.

---

## 4. Experiments

**Regimes**
- **Mechanism regime (primary for C1′–C3, H2–H4): noise 4.0, g = 1, K = 32.** It has the widest q–x gap (§0.2), training-matched noise, and is 2× cheaper.
- **Paper regime (primary for P1–P3, C4 end task): noise 0.1, g = 8, K = 32.** Interventions apply to both CFG branches.
- **Cross-regime replication:** C1′ and C3 are replicated at g = 8 / noise 4.0, and at the paper regime **restricted to its q–x window (steps 0–1)**. In the paper regime the feedback can only act through the earliest steps (early symmetry breaking), and that is tested directly. Outside the window, a null is expected and reported as the **absorption finding**.
- **Regime-relevance gate G1** (before the mechanism results are headlined): at noise 4.0 / g = 1, SC must give
  - (i) a coherence gain ≥ ½ of the paper-regime gain. *Smoke evidence covers only g = 8 at noise 4.0 (+6.4 vs. +3.6 nats) and g = 1 at noise 0.1 (+3.8); **noise 4.0 / g = 1 is unmeasured**, so it is measured in Phase 7;*
  - **and** (ii) an end-task gain whose **95% lower bound > 1 pp** (CARC). If G1 fails, mechanism claims are reported as "in the training-noise regime" only, and the headline rests on P1–P3 plus the paper-regime early-window test.

| Exp | Purpose | Key design | Main confounds → controls |
|---|---|---|---|
| **F** | P1–P3 at scale | ≈ 2,600 confirmation bimodal prompts (≥ 10 categories, ≥ 4 templates, both orders, programmatic; §6) + ≥ 2,000 natural blocks. AR vs. flow SC on/off × g {1, 8} × noise {0.1, 4.0} × K {1, 4, 32}. 0.5B + 7B. | Template → held-out templates. Tokenization → in-context + alignment. Classifier → audit. Typicality → external judge + dep_j. Small model → 7B. |
| **A** | C1′, M1 | Single-step δ^D (neighbour subsets), x-neutralization, own-only (frozen neighbours), at 8 steps in the q–x window | Perturbation size → subset factorial. OOD off → §2.7. Norm/entropy → random floor. Schedule → per-(position, step) reporting. State info → x-neutralization. Leakage → C2 contrast. |
| **B** | H2 shares; M5 (secondary) | Trajectory O × N × F 2 × 2 × 2 (8 cells); stale vs. degraded-fresh own arms | Post-treatment mediators → not used. Loops → frozen streams. Shares sum exactly (Möbius). |
| **C** | C3 | Minimal-pair donors at 4 steps in the q–x window | Sharpening → entropy-matched donors. Chaos → random floor. State conflict → donor applied with and without x-erasure (LEACE + random-subspace floor). |
| **D** | C4 | Per-query K/V cut of the feedback channel everywhere; own feedback kept; coherence (teacher-forced) and end task (CARC) | Patch machinery → identity patch. |
| **E** | H4 | Cloning (R ≥ 32, ε sweep) → τ_i, within-block χ, SC on vs. off | χ degeneracy → R ≥ 32, per-block, mode flips only. |
| **PC** | Instrument validity | Positive control (§2.8) and instrument validation (§2.7) | — |

**Baselines:** α-scaling and cutoff of the feedback (2607.00588).

**Moved to `leads/`:** the full g × feedback-variant sweep (01); the noise-scale curve (05).

---

## 5. Decision Criteria

### 5.1 Machinery reused
The interval-vs-margin rule, the support/kill/inconclusive convention and the outcome lattice (`phase1.md` §3.1–3.4), applied **per estimand**, with these margins.

### 5.2 Margins (substantive minima)
*Prov.* values are finalized in Phase 5 from the rules below. They are computed on the **exploration split** as fractions of *total* effects, never from the target effect's own pilot estimate.

| Estimand | Margin rule | Prov. value |
|---|---|---|
| Single-step δ^D on μ_j (C1′) | δ_μ = 20% of the single-step **block-level** SC effect on μ_j at the same (position, step) | from pilot |
| Trajectory N share (H2) | δ_share = 0.20 of SC's total effect on coh_j/dep_j at positions 2–3 | 0.20 |
| dep_j change | δ_dep = 25% of the smoke-equivalent total SC effect on dep_j | from pilot |
| Hybrid rate | 3 pp | 3 pp |
| Within-block χ | 0.1 | 0.1 |
| C4 | 10% of the SC gain | 10% |

### 5.3 New kill criteria
- **K11 (scale):** at 7B, bimodal hybrid rate ≤ 5% **and** the SC coherence gain at positions 2–3 is < ½ of the 0.5B value → reframe.
- **K12 (q–x window):** define the window as the steps where held-out linear-probe accuracy is **≥ 0.85 for q and ≤ 0.65 for x**. If, at scale, this window is < 3 steps at noise 4.0 for both g values, then M2 and R0 aren't separable in Lafa. **Report as a finding**, and restrict to the D/M and own-vs-neighbour shares.

### 5.4 Multiplicity
- **Primary family** (Holm-corrected): C1′ (x-neutralized δ^D at positions 2–3), H2 N share, P1, P2.
- Everything else is secondary or exploratory and labelled as such.

### 5.5 Lattice → paper mapping
- **C1′ + C2 + C3 ✓:** *"Self-conditioning is how parallel positions agree"* (M2).
- **R0 wins:** *"Flow LMs still mode-mix; self-conditioning improves coherence through better states, not by passing commitments"* (an analysis paper).
- **M1/M5 wins:** the same, with the own-stabilization mechanism.
- **All outcomes report:** P1 (novel measurement), the P2/P3 decomposition, and the q–x absorption finding at inference noise.

---

## 6. Power (per estimand; assume a true effect of 2 × margin, 80% power for "lower bound > margin")

For a mean-difference estimand: n ≈ ((1.96 + 0.84) · SD / margin)², then × the design effect for clustering by prompt (assume 1.5; re-estimate in the pilot).

| Estimand | Basis | n (units) |
|---|---|---|
| Trajectory N effect on coh_j at positions 2–3 (absolute, nats) | Smoke per-position Δcoh SD ≈ 2.5 at positions 2–3 (71% zeros → **re-estimate in the pilot**); margin ≈ 0.2 × 0.5 = 0.10 nats | 4,900 × 1.5 ≈ **7,350 target-blocks** (natural text; cheap) |
| N **share** (a ratio of pooled sums) | Variance by delta method / cluster bootstrap. **Power by simulation** from the pilot's joint distribution of (N effect, total effect) | Pilot simulation |
| C1′ single-step δ^D on μ_j | Unknown SD | Pilot |
| Hybrid rate (paired), margin 3 pp, true 6 pp | n ≈ 7.84 · p_disc / 0.03², with p_disc ≈ 0.30 (smoke) | **≈ 2,600 prompts** (× design effect if templates cluster) |
| Within-block χ | — | Pilot |

*Feasibility:*
- 7,350 blocks × 8 cells (2 × 2 × 2) × 32 steps × ~2 passes ≈ 3.8M single-block passes. That is ~14 h on the Mac, or ~1–2 h on CARC batched → **CARC**.
- Bimodal prompts are **generated programmatically**: ≥ 10 categories × ≥ 40 pairs × 2 orders × ≥ 4 templates ≈ 3,200, which covers 2,600 + the exploration split.

## 7. Data Hygiene
- **Excluded from all confirmation splits:**
  - the GSM8K test indices in `smoke/results/EXCLUDE_gsm8k_test_indices.json` (434);
  - all smoke bimodal prompts, templates and pairs (`smoke/03_hybrid.py`);
  - the q–x diagnostic prompts (the same set).
- **Splits:**
  - **exploration** (Phase 7; sets the margins, SDs, probe and classifier);
  - **confirmation** (held-out templates, categories and sources);
  - **circuit-validation** (if localization is pursued).

---

## 8. Open Items for Phases 5–6
- [Phase 5] Freeze the margin rules and values, thresholds (§2.2–2.5), the primary family, and the lattice mapping. Pre-register §3.1.
- [Phase 6] Instrument validation (§2.7), positive control (§2.8), per-query K/V identity test, CARC batching / SLURM.
- [Phase 7] Linear-probe q–x window at scale (K12); classifier audit; pilot SDs; Exp F including 7B (K11).
- [CARC] End-task SC gap at noise 4.0 / g = 1 vs. the paper regime: is the mechanism regime one where SC helps end tasks at all?

---

## 9. Novelty of the Phenomenon Claim P1 (targeted search, 2026-11-13)
- **(a)** Hybrid rates measured in a real continuous / flow LM: **novel**. Only toys and illustrations exist: FMLM 2602.16813 Fig. 3; CoCommit 2607.04469 (hypothetical, discrete); Zhang et al. 2602.00286 (Block-HMM); a Nov 2026 survey calls its example "an illustration, not an experimental result".
- **(b)** Flow vs. own-AR coherence comparison: **novel**. Lafa only reports accuracy and throughput; its self-speculation *hides* the problem by forcing the output to match AR.
- **(c)** SC reducing mode mixing / later-position coherence: **novel**. Prior SC work reports aggregate gains only.
- **(d)** Discrete analogues: **partly known** (ParallelBench: task validity, with "New City" as an anecdote).
- **(e)** AR-conditional in-block scoring as a metric: **novel**. CoDiLA uses AR as a decoder, not as a judge.
- **Safe wording:** "To our knowledge, the first measurement of cross-mode hybrids in a real continuous flow LM: under Lafa's published decoding settings, ~X% of samples on two-answer prompts are hybrids vs. 0% for the same checkpoint's AR head. Factorization-style errors previously shown only in toys or for discrete samplers persist despite joint transport." Tie the claim to these settings; don't say "provably".
- **Cite and differentiate:** FMLM, Lafa, CoCommit (+ ParallelBench).
- *Caveat:* ELF, LLCDM, aFMLM and Posterior Refinement were checked by abstract only. CoBiT, BitLM, DSL-LLaDA, Plaid, CDCD and TESS were not checked → re-check before submission.

---

## 10. Review Log
- **2026-11-13 v4 → v4.1:** one adversarial review. Blockers fixed:
  - (1) q–x gap: measured (§0.2) → mechanism regime moved to noise 4.0; new K12;
  - (2) margins per estimand, power with an assumed true effect of 2 × margin and a design effect;
  - (3) circular drift test replaced (§2.7); CFG rule for candidate (c);
  - (4) external judge co-primary plus the in-block dependence measure dep_j;
  - (5) predictions recast as magnitudes; χ within-block / mode flips / R ≥ 32; x-intervention instead of τ-stratification;
  - (6) M5 via stale-own feedback plus an accumulation null; frozen neighbours for own-only; O × N 2 × 2.
  
  Also fixed: M3/M4 wording, origination back in scope, classifier alignment and precedence, R ≥ 32, per-position re-noising with an ε sweep, a multiplicity plan, the positive control, and "~95%" corrected to 92%.
- **2026-11-13 v4.1 → v4.2:** review round 2. Fixed:
  - §2.7 now tests the mixed-context OOD risk (Mahalanobis on residual streams) plus cross-candidate agreement; the efficacy test is noted as trivial for (a);
  - dep_j → PMI against AR-sampled predecessors (not greedy);
  - O × N 2 × 2 → **O × N × F 2 × 2 × 2**, whose Möbius shares sum exactly and include the R0 state route (F);
  - M5 → stale vs. degraded-fresh own feedback at matched informativeness (secondary);
  - x-neutralization → LEACE erasure + probe verification + random-subspace floor + OOD check;
  - hybrid n → ≈ 2,600; share power by simulation; 7,350 arithmetic;
  - K12 thresholds (q ≥ 0.85, x ≤ 0.65);
  - regime-relevance gate G1 + paper-regime early-window test.
- **2026-11-13 v4.2 → v4.3:** review round 3. All round-2 blockers were confirmed resolved except the 2 × 2 × 2.
  - **Fixed:** N and F are decoupled with a shadow pass (N reads the neighbours' current predictions in every cell), so N's main effect isn't zero by construction. Shapley values are added as pre-registered robustness.
  - **Minors:** two-knob degraded-fresh matching; cross-candidate agreement restricted to equal-efficacy candidates, with (b) reported separately; G1 citation corrected (noise 4.0 / g = 1 unmeasured) and its end-task condition put against a margin; Ledoit–Wolf covariance.
  - **Status:** the reviewer stated that either fix to the N share makes the spec ready for Phase 5. The fix applied is the reviewer's first option.
- **2026-11-13 (Phase 5 note):** `prereg.md` v1.1 governs the final margins: P1 hybrid margin 5 pp (3 pp kept for secondary hybrid contrasts); P2 fixed at 0.10 nats; H2-N via the linear contrast. See prereg §13.
