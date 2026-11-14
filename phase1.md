# mechflow — Phase 1: The Claim, Its Rivals, and How It Dies

> **Superseded as the working spec by `phase4.md` (2026-11-13).** Kept as the historical record. Its machinery (§1.5 interventions, §3 statistics and lattice) is reused by reference from `phase4.md`.

Date: 2026-11-13 (v3.1, after three rounds of adversarial review; reviewer signed off on Phase 2 readiness; see §8). Written **before** the Phase 2 code/paper read and the Phase 3 literature review, so neither quietly reshapes the question. Anything Phase 2–3 overturns is revised in Phase 4 and logged in §8, not silently edited.

Companion files: proposal `mechflow/mechflow.md`, checklist `mechflow/TODO.md`.

---

## 1. The Core Claim

> **In Lafa, a position's self-conditioning feedback is read *directly*, within the same denoising step, by the other positions of its flow block, and it carries which joint continuation that position currently favours. When a block's continuation has several valid but token-coupled modes, this direct cross-position read makes partner positions commit to the *same* mode. Removing it (while leaving every position's own feedback and state intact) delays commitment and produces more cross-mode hybrids.**

Short form: *self-conditioning is how parallel positions agree.*

### 1.1 The structural fact the design must respect

In x̃_k = x_k + x_kW₁ᵀ + (q_{k-1}E)W₂ᵀ, the feedback q_{k-1,i} enters the network **only at position i**. Neighbour i's feedback can therefore reach target j in exactly two ways:

- **Direct (D):** within the same step, q_i changes i's residual stream, and j reads it through attention (K/V at i).
- **Mediated (M):** q_i changes i's prediction, hence x_i and q_i at later steps, and j reads *those* through attention later. (M runs through the noisy-state channel. It is the skeptic's route, R0.)

**The claim is about D.** Any whole-trajectory "turn neighbour feedback off" intervention mixes D and M, so it can't tell the claim from R0. Every primary test below isolates D.

### 1.2 Sub-claims

| ID | Sub-claim | Primary measurement |
|---|---|---|
| **C1: Direct existence** | With x_k fixed, letting the other positions see neighbour i *as if q_i were off* (while i itself keeps q_i) moves j's mode margin away from the mode it commits to. | Signed single-step direct effect δ^D_{i→j} on μ_j (§1.4–1.5), plus its trajectory-level consequence for τ_j and hybrid rate. |
| **C2: Specificity (coupling contrast)** | The direct effect is larger on **coupled**-multimodal blocks than on multimodal blocks whose positions are **independent**, matched on per-position entropy. | Interaction: (coupled − independent) direct effect. |
| **C3: Mode content** | Before j commits, making j read neighbour i's feedback **from a matched trajectory that ended in the competing mode** moves j toward that mode, more than entropy- and norm-matched non-mode donors at the same step do. | Flip-toward-donor rate vs. matched-donor and random-perturbation floors. |
| **C4: Significance** | Removing only the direct cross-position path for the whole trajectory lowers end-task quality on natural prompts by a non-trivial share of the full-vs-none self-conditioning gap. | Task metric under path-removed vs. full vs. none trajectories. **Needs no block classification.** |

C1–C3 together are the mechanism claim; C4 is what makes it matter. **A compact circuit (H4) is not part of the claim.** Without one the paper is weaker, but the claim is not refuted.

### 1.3 Scope (what is *not* claimed)

- **Model:** Lafa checkpoints (0.5B primary; 7B/8B replication). Nothing about flow LMs in general.
- **Unit of analysis:** a single target block with a **teacher-forced, identical prefix** across all conditions (C1–C3). Prefixes never diverge between arms. C4 is the exception: it is deliberately a full-generation, task-level measure, so later prefixes may differ.
- **Sampler:** deterministic, at the step count Phase 2 shows was used for the reported results, plus 4/8/16/32-step sweeps. An effect that exists only at very few steps narrows the claim (K5).
- **Propagation, not origination.** The claim is that neighbour feedback *spreads and locks in* a commitment, not that it makes the choice. With deterministic sampling the initial noise plausibly picks the mode. This premise is tested, not assumed: a probe on x_T / early x_k must predict the final mode above chance (A8). If it can't, the scope is revisited.
  - Operationally, propagation means **lead–lag**. The position that commits first (i) predicts its partners' later commitment, and a q_i donor swap flips j **before** j commits but not after.

### 1.4 Measurement definitions (finalized in Phase 4; fixed here in kind)

- **Modes of a block:** the set of valid joint continuations M = {m₁, m₂, …} for the teacher-forced prefix. In constructed tasks this set is known by design. On natural text it comes from an external reference (§1.6).
- **Mode pair (fixed per block):** the two leading modes A, B are fixed once per block from the reference, never re-chosen per step. A is defined as the mode the **full** trajectory ends in (for that seed). If the full run ends in a hybrid or other-valid output, A is undefined and that (block, seed) is excluded from μ-based statistics, though kept for hybrid-rate statistics. This exclusion selects committed cases, so its rate is reported per condition. Blocks whose modes' tokenizations misalign at the positions of interest are dropped, and the drop rate is reported.
- **Mode margin μ_j:** for a position j where A and B differ, μ_j = log q_j(token_j^A) − log q_j(token_j^B). This is signed so that positive means toward the full run's mode. Effects are **signed**: δ = μ_j(full) − μ_j(intervened), so a positive value means j is less committed to, or flipped away from, A. It replaces cross-entropy as the primary single-step outcome, because CE against a fixed reference treats a switch to the other valid mode as an error.
- **Commitment time τ_j:** the first step after which sign(μ_j) never changes and |μ_j| > θ for the rest of the trajectory.
- **Output classes** (token-level, scored only at mode-distinguishing positions, so shared tokens like ", the" never count):
  - **mode**: all distinguishing positions match one mode;
  - **hybrid**: distinguishing positions take tokens from ≥ 2 different modes;
  - **other-valid**: matches no mode, but the reference scores it comparably to the modes;
  - **invalid**: everything else.
  
  Only *hybrid* counts as mode mixing; the other classes are reported separately.
- **Hybrid attribution (R8):** compare each position's final token with its token in the full run for the same seed.
  - **Source defection:** the intervened position i changed.
  - **Target failure:** a non-intervened position j changed.
  - **Both:** report both.
  
  This works for 1–1 splits such as "New York", where a majority rule is undefined.
- **Block types:**
  - **coupled**: modes exist, and cross-mode recombinations are invalid or low-probability;
  - **independent**: per-position alternatives freely combine, matched on per-position entropy;
  - **unimodal**: reported but **not** used as a control (see R1).

### 1.5 Interventions and statistics

**Single-step direct effect (primary; fixes the leaky v2 spec).** At step k, with the state x_k and all q fixed, for a source set S of positions:

1. **Clean pass P:** full feedback everywhere.
2. **Off pass P_S^off:** q_s set to the "off" value for every s ∈ S, **jointly** (one pass per S, not one per neighbour).
3. **Patched pass P_S^D:**
   - every position s ∈ S has its **whole residual stream frozen** to its values from P, at every layer;
   - every position h ∉ S, when attending to any s ∈ S, reads s's K/V from P_S^off. h's own stream is computed live.

The other positions read S "as if its feedback were off", while S itself keeps its feedback and its output is unchanged. q_S can no longer reach any non-S position through any route, including relays through third positions h, because every non-S reader sees off-K/V for S. And S cannot be perturbed by the patch, because its stream is frozen.

**Simplification.** With x_k fixed, every non-S position computes the same values in P_S^D as in P_S^off (by induction over layers), so the single-step estimand reduces to μ_j(P) − μ_j(P_S^off). The frozen-patched pass P_S^D is needed only to keep S's *own* output clean in rollouts.

**Estimand:** δ^D_{S→j} = μ_j(P) − μ_j(P_S^D) = μ_j(P) − μ_j(P_S^off) for j ∉ S. This is the **total within-step effect of S's feedback on j**, through all within-step routes. It is defined by this intervention; we make no claim that it is a "pure path".

**Neighbour factorial and Möbius terms.**
- For each target j, S ranges over all nonempty subsets of j's 3 neighbours, and the empty set is the clean pass.
- Define the presence game v_j(T) = μ_j when the neighbours in T are visible with feedback and the rest are seen off.
- Report its Möbius (Harsanyi) terms: main effects per neighbour, and pairwise and three-way **neighbour-redundancy terms**.
- **Predicted sign:** when ≥ 2 neighbours carry the same mode, their signals are redundant, so the pairwise terms are **≤ 0** (sub-additive: removing one does little, removing all does a lot). When only one neighbour distinguishes the mode ("New York"), its main effect carries everything and the interaction terms ≈ 0.
- **Naming:** "neighbour-redundancy terms" (Möbius, within a block) are distinct from the "**coupling contrast**" (C2: coupled minus entropy-matched independent blocks).

**Trajectory-level decomposition, per target j (2 × 2; fixes v2's unidentified D/M).** Two factors are crossed:
- **F (neighbour outputs):** each neighbour's own prediction, which drives its state update and its next-step q, comes from the pass with its feedback **on** or **off**.
- **R (j's reads):** j reads neighbours' K/V from a pass **with** their current feedback, or from an **off** pass.

The factors must be decoupled this way: with "feedback off" alone, F = off would automatically remove what j reads, and the design would collapse.

| | R = on | R = off |
|---|---|---|
| **F = on** | full | D cut (neighbours evolve normally) |
| **F = off** | M cut (neighbours evolve as without feedback; j still reads their current feedback) | both cut |

D, M and D × M are the factorial main effects and the interaction of this design, on τ_j and the hybrid rate. They are **design-defined estimands**: D includes the downstream consequences of cutting D, and that is stated as such.
- **What neighbours read in mixed cells:** neighbours always read j's **live** stream in that cell, never a clean copy of j. The cell is then one consistent counterfactual world for j.
- **Pure state-mediated effect:** in the F = off, R = on cell, j still reads the neighbours' *altered* feedback directly, so the M main effect is not purely state-mediated. The **pure state-mediated effect** is reported separately as the F effect at R = off.

**C4 (all positions at once).** Every position reads every other position's K/V from the off pass and keeps its own feedback in its own stream. This needs **per-query K/V selection** (live K/V for itself, off-pass K/V for everyone else), implemented through a custom attention path. It is the main implementation risk, and Phase 2's K1 faithfulness check targets it first. C4 = (full − D-cut-all) / (full − none), computed as a ratio of pooled sums. It is interpreted as the *share of the self-conditioning gain lost when the cross-position direct path is cut throughout*, not as "the share mediated by D".

**Floors.**
- Matched random perturbations of q_S (same norm and entropy, applied through the same patching) give the floor for every intervention (R9).
- For C3, donors are matched on entropy **at the same step k** (R7).

**Aggregation.**
- Use ratios of pooled sums, never means of per-example ratios.
- Use a bootstrap **clustered by prompt and seed**.
- Use **several seeds per prompt**, because with deterministic sampling the mode depends on the initial noise.

### 1.6 Block classification on natural text

- **C2/C3 primary evidence uses constructed tasks with known modes.** No classifier is needed there.
- **Natural-text classification** (secondary, used for C2 generalization and the pilot) uses an **external** AR reference (e.g., Qwen2.5-7B), with the Lafa checkpoint's own AR mode as a second reference. It is pre-registered:
  - mode-mass threshold;
  - recombination log-prob gap;
  - a disagreement rule: analyse only blocks where both references agree, and report the agreement rate. **Also report every result using the external reference alone**, because agreement-filtering on the checkpoint's own AR can bias the sample.
- **Validate the classifier before use.** Measure its precision and recall on the constructed tasks with known modes, and the checkpoint's AR perplexity against base Qwen.
- **C4 does not depend on this classifier.**

---

## 2. Rival Explanations

Each rival makes a **different prediction** from the claim on at least one pre-specified measurement.

| ID | Rival | Separating prediction |
|---|---|---|
| **R0: Noisy-state coordination** | Positions coordinate only by attending to each other's states. Feedback helps each position with its own token, and neighbour influence runs entirely through the **mediated** path (i's later states). | **Direct** neighbour effect ≈ 0 (within equivalence bound δ); all neighbour influence is mediated. (Kills C1.) |
| **R1: Generic leakage** | Perturbing i's feedback disturbs i's representation, and the disturbance leaks into j through attention, scaling with uncertainty rather than content. | Direct effect is equal on **coupled vs. entropy-matched independent** blocks. Mode donors ≈ matched donors. (Kills C2, C3.) |
| **R2: State conflict** | Swap effects arise because the swapped q conflicts with an x_i that already encodes the other mode, an out-of-distribution input the model reacts to arbitrarily. | Coherent flips occur **only** when x_i (decoded) disagrees with the donor q, and not when x_i is still uncommitted. |
| **R3: Off/mixed-mask artifact** | The "off" value, or mixing on and off across positions, is out of distribution, so knockouts measure damage. | On unimodal blocks, mixed-mask passes show calibration and CE drift comparable to the claimed effects. Effects vanish under an alternative in-distribution "off". |
| **R4: Integration artifact** | The effect exists only at coarse Euler steps, where generation is broken anyway. | Effect → 0 by 16–32 steps, and present only where task quality is far below baseline. |
| **R5: Rarity** | The mechanism is real but natural blocks are rarely coupled, so it explains little of the benchmark gain. | C1–C3 hold on constructed tasks; the C4 share ≈ 0. |
| **R6: Mediated-only** (R0's strongest form) | Neighbour feedback matters at trajectory level only because it improves the neighbour's later states. | In the 2 × 2 design: the D main effect ≈ 0, while the M main effect is large. |
| **R7: Generic sharpening** | Neighbour feedback lowers the neighbours' entropy, and j commits to *whatever* is sharp, so there's no mode content. | Entropy-matched non-mode donors at the same k move j as much as mode donors do. |
| **R8: Defector** | Hybrids under knockout come from the *perturbed* neighbour drifting off-mode, not from j failing to coordinate. | Hybrids are attributed (§1.4) mostly to the intervened position, not to targets. |
| **R9: Chaotic sensitivity** | In a deterministic ODE, any small perturbation flips the mode. | Matched random perturbations produce flip and hybrid rates comparable to the targeted interventions. |

### 2.1 The strongest skeptical alternative (one sentence)

> **R0/R6 + R1:** *"Parallel positions coordinate through attention over each other's states; self-conditioning sharpens each position's own token, and whatever neighbour influence you find either flows through the neighbours' improved later states or is uncertainty-proportional leakage, not a signal about which mode to choose."*

Each core sub-claim must beat this rival on its own terms:
- **C1 beats R0/R6:** single-step direct effect > δ (M cannot act within a step), and the D main effect > δ in the 2 × 2 design.
- **C2 beats R1:** coupled > entropy-matched independent.
- **C3 beats R1/R7/R9:** mode donors > entropy-matched donors > random floor.

---

## 3. Decision Criteria

### 3.1 Statistical convention (applies to every criterion)

Each confirmatory criterion compares a 95% cluster-bootstrap interval on the **confirmation split** against a **margin δ**:

- **Kill** if the upper bound < δ (an interval-vs-margin equivalence rule: the effect is smaller than anything we care about).
- **Support** if the lower bound > δ.
- **Inconclusive** otherwise. This is a power problem, never a kill: add data within budget, or report it as inconclusive.

**How δ is set (fixes v2's regression-to-the-mean trap and phase inversion).** δ is **not** derived from the pilot's observed effect size.
- **Phase 5 pre-registers the rule:** δ = max(δ_sub, δ_floor).
  - δ_sub is a substantive minimum fixed in advance. *Prov.:* signed Δμ_j ≥ 0.5 nats; hybrid rate +2 pp; Δτ_j ≥ 2 steps (at 32 steps); C4 ≥ 10% of the gap.
  - δ_floor is the 95th percentile of the **sampling distribution of the pooled statistic** under matched random perturbations (R9 floor), obtained by bootstrap over examples. It is **not** the per-example distribution, which would kill real effects.
  - **Preferred form:** test the paired difference (targeted − random) against δ_sub directly. δ_floor is then only a sanity bound.
  - For contrasts (C2, C3), the same applies to the contrast statistic.
- **Phase 7 computes δ_floor** on the exploration split.
- **δ is frozen before the confirmation split is touched.**

### 3.2 Kill criteria, in phase order

| # | Triggers if… | Phase | If triggered |
|---|---|---|---|
| **K1: Untestable** | No usable in-distribution "off" exists (no training-time drop, unusable first-step q, no defensible alternative), **or** K/V path patching can't be implemented faithfully. | 2 | Reframe to swap-only (C3) designs; narrow the claim. |
| **K2: Not novel** | Prior or concurrent work already shows self-conditioning (or equivalent feedback) carrying mode information across parallel positions. | 3 | Reframe (e.g., to D-vs-M decomposition or localization, if only the behaviour is known) or kill. |
| **K3: Artifact instrument** | Mixed-mask or path-patched passes on **unimodal** blocks show CE or calibration drift ≥ the pilot's effect size, **or** effects disappear under a second justified "off". | 6 | Knockout designs invalid; fall back to swap designs (themselves checked against R2). |
| **K4: Absent** | On the exploration split, the **conditional** hybrid-rate difference (full vs. none) among coupled blocks is < 2 pp *(prov.)* **or** coupled blocks are < 2% *(prov.)* of natural blocks at every step count. | 7 | Rarity → constructed-task-only paper (weaker) or kill. |
| **K5: Regime** | Pilot effects appear only at ≤ 4 *(prov.)* steps, where task quality is far below the 32-step baseline. | 7 | Narrow to a few-step phenomenon; consider a sampler contribution instead. |
| **K6: R0/R6 wins** | On coupled blocks, both the pooled signed single-step direct effect δ^D (summed over each target's neighbours) **and** the 2 × 2 D main effect on τ_j and the hybrid rate have upper bound < δ. | Confirmatory | **Kill** core claim (report D-vs-M decomposition as analysis). |
| **K7: R1 wins** | (Coupled − entropy-matched independent) direct effect: upper bound < δ. | Confirmatory | **Kill** C2 → claim collapses to "neighbours matter". |
| **K8: No content** | (Mode-donor − entropy-matched donor) flip rate before j commits: upper bound < δ, **or** mode-donor flip rate not above the random floor (R9). | Confirmatory | **Kill** C3. |
| **K9: Conflict-only** | Coherent flips occur in the uncommitted-x_i regime at rate < δ while present in the conflict regime (R2). | Confirmatory | C3 reflects out-of-distribution reaction; narrow/kill. *(Note: x-only swaps reproducing effects is expected since q = f(x) in training and is **not** a kill.)* |
| **K10: Insignificant** | C4: share of the full-vs-none task-quality gap lost under trajectory-level D-removal has upper bound < δ_C4 *(prov. 10% of gap)*. | Confirmatory | Mechanism real but minor → reframe as mechanism-only paper (weaker) or narrow venue. |

### 3.3 Success criteria (mirror of the kill criteria)

The core claim is **supported** if C1, C2 and C3 all reach "support" (lower bound > δ) on the confirmation split, with K3 cleared. It is **significant** if, in addition, C4's lower bound > δ_C4. Anything else is reported as the exact pattern observed, using the outcome map below.

### 3.4 Outcome map (exhaustive; fixes v2's holes)

**Precedence rules, applied in order:**

1. **Instrument gates.**
   - K1 or K3 triggered → knockout instruments are invalid. The C1 column becomes "N/A", and the mechanism is judged on C3 (swap) alone. C2 is then measured on the swap instrument. See the N/A rows below.
   - K9 triggered → C3 counts as ✗.
1b. **Single C1 verdict:**
   - **✓** if the single-step direct effect is ✓, whatever the 2 × 2 D effect shows;
   - **✗** only if both are ✗ (matching K6).
   
   If the single-step effect is ✓ but the 2 × 2 D effect is ✗, mark C1 as **✓ (compensated)**: the direct read exists, but the trajectory routes around its removal. Rows with C1 ✓ then carry the qualifier "compensated", and C4 is expected to be weak.
2. **Inconclusive columns.** An inconclusive column supports no conclusion either way. Claims that depend on it are reported as open, and the decision uses only the decided columns. If a decision hinges on an inconclusive column, extend data within budget before deciding.
3. **C2 is measured on whichever instrument (knockout C1, swap C3, or both) shows an effect.** If neither does, C2 is uninterpretable ("–").

**Lattice over the decided columns (C1 knockout, C2 coupling contrast, C3 mode-content swap):**

| # | C1 | C2 | C3 | Interpretation | Action |
|---|---|---|---|---|---|
| 1 | ✗ | – | ✗ | R0/R6: coordination via states | Publish the D/M decomposition as a negative/analysis result, or kill. |
| 2 | ✓ | ✗ | ✗ | R1: generic leakage | Kill as main contribution. |
| 3 | ✓ | ✓ | ✗ | Coupling-specific sensitivity without mode content (R7 sharpening) | Reframe around sharpening; test the entropy-only account. |
| 4 | ✗ | ✗ | ✓ | j reads neighbours' token identity, but not coupling-specifically, and removing feedback is compensated | Weak; reframe as "feedback carries token identity" or kill. |
| 5 | ✗ | ✓ | ✓ | Mode content is read coupling-specifically, but knockout shows no effect (j falls back on states) | **Narrower "content" claim**, not the full claim (§3.3 requires C1). Test the compensation account directly (does M take over in the 2 × 2?). |
| 6 | ✓ | ✗ | ✓ | Content-carrying reads, but general rather than coupling-specific | Reframe: general cross-position feedback reads; the coordination framing is dropped. |
| 8 | ✓ | ✓ | ✓ | Full mechanism claim | Proceed to localization (H4) and replication. |
| N/A-a | N/A | – | ✗ | Instruments invalid and no swap effect | Kill, or reframe as an instrument/methods note. |
| N/A-b | N/A | ✗ | ✓ | Swap-only: content read, not coupling-specific | As row 6, with a weaker evidence base. |
| N/A-c | N/A | ✓ | ✓ | Swap-only: coupling-specific mode content | Narrower "content" claim (as row 5); localization proceeds on the swap instrument. |

(Row 7 from v3 is deleted: under rule 3, C2 is "–" whenever neither instrument shows an effect, so C1✗/C2✓/C3✗ cannot occur.)

**C4 modifier (applies to every row):**
- C4 ✓ → "significant".
- C4 ✗ → "real but minor" (mechanism paper, weaker framing).
- C4 ✓ in row 1 (step-level effects absent, task-level effect present) → small effects may compound over steps. Run the multi-step 2 × 2 decomposition before concluding anything.

**Does not kill the claim:** no compact circuit (H4); larger effects at 7B/8B than 0.5B; own-feedback effects exceeding neighbour effects (expected; the claim needs a non-trivial direct neighbour effect, not a dominant one); x-only swaps reproducing effects.

---

## 4. The Headline Figure (what success looks like)

```
(a) Setup                                 (b) Commitment on coupled blocks (teacher-forced prefix)
 prefix: "...flew to"                      fraction of blocks committed to a consistent mode
 modes:  [ New | York | , | the ]            1.0 |                ____ full
         [ San | Fran.| , | the ]                |           ___/ ___ D-cut (direct path cut,   
 hybrid: [ New | Fran.| , | the ]                |        __/ __/          sources' streams frozen)
 paths:  q_i ─D─▶ j (same step)                  |      _/ _/    hybrid rate ↑, τ delayed
         q_i ─M─▶ x_i(t+1) ─▶ j                  |    _/_/
                                             0.0 |__//
                                                  0      denoising time t       1

(c) Donor swap of q_i, by j's commitment status    (d) Direct effect δ^D, coupled vs independent
 flip-toward-donor rate                              coupled:      main ▇▇▇  redundancy ≤0 ▁
  ^  ● mode donor (before j commits)                  independent:  main ▏    (coupling contrast)
  |  ○ entropy-matched donor                         + 2×2 D / M / D×M on τ_j and hybrid rate
  |  · random floor                                  + C4: % of SC gain lost, D cut everywhere
  |  ●●●                     ●● (after j commits)
  +---------------------------------------------->
     before τ_j        |        after τ_j
```

Predictions, each derived from a sub-claim:
- **(b) [C1]:** removing the direct path delays commitment and raises the hybrid rate on coupled blocks. *No prediction is made about how own-feedback removal differs in shape.* Own removal may also cause defection, and hybrids are attributed with the defector rule.
- **(c) [C3 + propagation]:** the swap effect is larger **before** j commits than after, at matched step k. *It is not predicted to peak early,* since early q is near uniform and donors barely differ.
- **(d) [C2, C4]:**
  - the **coupling contrast** (coupled minus independent direct effect) is positive;
  - the **neighbour-redundancy** terms are ≤ 0 where ≥ 2 neighbours carry the mode;
  - the D main effect is non-trivial in the 2 × 2 design;
  - C4 is reported as a % of the gain.

**Headline sentence if it works:** *"Self-conditioning feedback is how parallel positions agree: within each step, positions read their neighbours' feedback for which mode to take; cutting only this direct path raises the hybrid rate on coupled continuations by X pp and costs Y% of self-conditioning's gain."*

If panel (b) can't be produced (no D-cut vs. full gap on coupled blocks), the paper doesn't exist in this form. **This is the single most important early result.**

---

## 5. Assumptions

| ID | Assumption | Checked by | If false |
|---|---|---|---|
| A1 | An in-distribution "off" feedback value exists (training-time drop input, or first-step q). | Phase 2, code | K1 → swap-only. |
| A2 | Sampling is deterministic given the seed. | Phase 2 + Phase 6 test | Paired seeds, many repeats; effects become distributional. |
| A3 | Block boundaries can be aligned so a coupled span sits in one block; prefixes can be teacher-forced. | Phase 2, code | Restrict to naturally aligned spans; report coverage. |
| A4 | The external AR reference classifies blocks accurately (precision/recall on known-mode tasks). | Phase 7 | Use constructed tasks only for C2/C3. |
| A5 | 0.5B produces enough coupled blocks to measure. | Phase 7 | Move primary results to 7B (cost: 8–16× passes; budget it). |
| A6 | Feedback enters only via (qE)W₂ᵀ at the input. | Phase 2, code | Extend interventions to all paths. |
| A7 | Mixed per-position masks / path patching are tolerable: training-time drop granularity is known (whole-block vs. per-position), and unimodal-block calibration drift is small. | Phase 2 (granularity) + Phase 6 (drift) | K3. |
| A8 | The initial noise / early x_k predicts the final mode above chance (premise of "propagation, not origination"). | Phase 7 probe | Revisit the scope in §1.3. |

---

## 6. Compute Sanity (feeds the Phase 5 power analysis; fixes v2's ~20× undercount)

A unit is a forward pass over one 4-token block with a cached, teacher-forced prefix. Counts are per example, per seed, at 32 steps, using 4 intervention steps k (mean ≈ 16 remaining steps).

| Component | Passes | ≈ Count |
|---|---|---|
| Base trajectory | 32 | 32 |
| Single-step direct effects | P_S^off is **target-independent** and sufficient (see the simplification in §1.5): 14 S (nonempty, never all four positions) × 4 k | 56 |
| Rollouts for τ_j / hybrid after each single-step intervention (S, k) | 14 S × 4 k × 16 steps × 2 (clean pass P gives S's own outputs; off pass P_S^off gives the non-S outputs) | 1,800 |
| 2 × 2 D/M trajectories | 4 targets × 3 non-full cells × 32 steps × 2 (on/off) | 770 |
| C3 donor trajectories | 2 modes × 32 | 64 |
| C3 swap rollouts | 4 sources × 4 k × 16 steps × (1 mode + 1 matched + 5 random) | 1,800 |
| C3 matched-seed search | must find seeds whose full run ends in the *competing* mode: about 1/p(B) draws × 32 passes each (e.g., p(B) = 0.3 → ≈ 3 draws ≈ 100 passes; rare modes cost much more) | ≈ 100+ |
| **Total per example-seed** | | **≈ 4.6k (+ the donor search)** |

- **Seeds:** ×4 → ≈ 19k passes per example. Blocks where the competing mode has p(B) < 0.1 under the sampler are capped or excluded from C3, and the exclusion rate is reported.
- **Step-count sweeps (4/8/16):** a reduced design (single-step + 2 × 2 only) adds ≈ +15%.
- **C4:** about 3 conditions × full generation per prompt, priced separately and running on benchmark prompts.
- **Order of magnitude:** these are tiny batched passes (4 query tokens). If one pass is about 2–5 ms amortized at 0.5B, that's ≈ 1–2 GPU-min per example, or ≈ 15–35 GPU-h per 1,000 examples. At 7B, expect ≈ 10–15× that, so replication must use a subset: fewer seeds, 2 values of k, no step-count sweeps. **These are estimates; Phase 6 benchmarks replace them.**
- If coupled blocks are about 5% of natural blocks, natural-text C2 needs about 20× more raw blocks than constructed tasks. That's why constructed tasks carry the primary C2/C3 evidence.

## 7. Decisions Taken in This Phase (and why)

1. **The claim is about the *direct* within-step path.** Feedback enters only at its own position, so whole-trajectory knockouts mix in the noisy-state route (R0). The direct effect is defined by freezing the sources' streams and serving off-K/V to every other reader. At trajectory level, a 2 × 2 design that decouples neighbour *outputs* from j's *reads* gives D, M and D × M as design-defined estimands.
2. **Möbius terms over the neighbour-subset factorial, not Shapley over 16 masks.** Shapley spreads interactions evenly and would hide them. The predicted sign is **redundancy (≤ 0)**, not synergy, and it is kept separate from the coupling contrast (C2). The own position is not a "player", which removes the 3-vs-1 imbalance.
3. **Mode margin and commitment time, not CE.** CE punishes a switch to the other valid mode as an error.
4. **Entropy-matched independent blocks, not unimodal blocks, are the C2 control.** Generic leakage scales with entropy (R1).
5. **Interval-vs-margin rule with δ = max(substantive minimum, random-perturbation floor).** The rule is pre-registered in Phase 5 and the floor is computed in Phase 7, before the confirmation data is seen. δ is never derived from the pilot effect size, which avoids regression to the mean.
6. **C4 is decoupled from any block classifier.** Significance is measured on task quality.
7. **The scope stays at propagation, not origination, but is made testable** (lead–lag + A8 probe).
8. **Circuit (H4) demoted to non-core.**

---

## 8. Change Log

- **2026-11-13 v1:** initial version.
- **2026-11-13 v2:** revised after an independent adversarial review (AC-style). Issues fixed:
  - *(critical)* neighbour knockout confounded with the mediated path, so R0 was not separable → claim restated around the direct path, with K/V path patching;
  - *(critical)* prefix divergence → teacher-forced prefix;
  - *(major)* CE metric mismatch → μ_j, τ_j, hybrid rate;
  - *(major)* mixed-mask out-of-distribution inputs → A7, K3;
  - *(major)* Shapley hides synergy → Möbius terms;
  - *(major)* unimodal control confounded with entropy → entropy-matched independent blocks;
  - *(major)* incoherent kill logic (CI-includes-0, redundant K2 conjunct, non-monotone order) → TOST convention, renumbered by phase;
  - *(major)* no criterion for C4 / no success criteria → K10, §3.3, outcome map §3.4;
  - *(major)* missing rivals → R6–R9 added with floors and defector attribution;
  - *(major)* panel predictions not derived → (b)/(c) predictions restated;
  - *(minor)* undefined thresholds → token-level hybrid definition, pre-registration items;
  - *(minor)* no compute estimate → §6.
  
  - *(self-check)* v2 draft defined an "own-only" arm identical to D-removed, which made the M estimate trivially 0 → replaced with neighbours-off(j).

  Open questions resolved:
  - (a) propagation scope kept and made testable (A8, lead–lag);
  - (b) K3 absolute thresholds replaced by a δ set on the exploration split, with an entropy-matched null;
  - (c) C4 decoupled from the AR reference; the reference is validated and paired with an external model.
- **2026-11-13 v3:** revised after a second review round. Issues fixed:
  - *(N1, critical)* the direct-path spec leaked (source streams read patched positions; relays through third positions) → source streams frozen, off-K/V served to every non-source reader, joint off pass per subset;
  - *(N2)* D/M not identified → 2 × 2 crossing neighbour outputs × j's reads. The reviewer's suggested 2 × 2 collapses unless outputs and reads are decoupled, so it was corrected accordingly;
  - *(N3)* interaction sign → redundancy ≤ 0 predicted; "redundancy terms" named separately from the "coupling contrast";
  - *(N4)* mode pair fixed per block from the reference; misaligned blocks dropped;
  - *(N5)* δ = max(substantive, floor), rule pre-registered in Phase 5, floor computed in Phase 7, never from the pilot effect;
  - *(N6)* exhaustive outcome lattice with precedence rules;
  - *(N7)* compute recounted (≈ 23k passes per example with 4 seeds);
  - partials: signed Δμ; other-valid class restored; hybrid attribution by comparison with the full run (handles 1–1 splits); results also reported with the external reference alone.
- **2026-11-13 v3.1:** third review round. The reviewer **signed off** (no blockers) and confirmed the corrected 2 × 2. Folded-in fixes:
  - single-step δ^D simplifies to μ_j(P) − μ_j(P_S^off);
  - neighbours read j's live stream in mixed 2 × 2 cells, and the pure state-mediated effect (F at R = off) is reported separately;
  - mode A is undefined when the full run doesn't commit → excluded and its rate reported;
  - C4's per-query K/V selection flagged as the main implementation risk (the Phase 2 K1 check targets it);
  - δ_floor defined on the pooled statistic's sampling distribution, with the (targeted − random) paired test preferred;
  - lattice: row 7 deleted, row 5 relabelled as a narrower claim, single-C1-verdict rule ("compensated"), N/A rows added;
  - compute: 14 subsets, 2 passes/step, plus the cost of searching for matched donor seeds.
- **2026-11-13, Phase 2 findings (pending Phase 4 revision; claims not yet edited).** See `phase2.md`.
  - **A1 ✅** at block level.
  - **A7 ⚠️**: training drops are whole-batch, so mixed masks are out of distribution → K3 critical; three single-position "off" candidates to compare.
  - **A8 ⚠️**: inference noise is 0.1·σ, so seeds barely matter and "propagation, not origination" must be revisited; C3 donors move to minimal-pair prompts.
  - **Asynchronous time**: lead–lag is built into the schedule, so it no longer counts as evidence; the direct effect must beat x_i at matched schedule position.
  - **Primary regime** g = 8 / 32 steps / noise 0.1, with interventions applied in both CFG branches.
  - **A2, A3, A6 ✅.** K1 not triggered.

