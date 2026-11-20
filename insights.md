# mechflow — First-Principles Insights and New Experiments (2026-11-17)

Status: exploratory. Nothing here changes the frozen confirmatory analysis. New experiments run on the **exploration** split; re-analysis of existing confirmation outputs is labelled *exploratory*.

---

## 1. Two quick zero-GPU tests on the confirmation 2 × 2 × 2 outputs (exploratory; 1,955 documents)

| Prediction | Result | Verdict |
|---|---|---|
| The neighbour-reading (N) effect concentrates at **word-continuation** tokens (stitching multi-token words) | N effect on dep_j: word-start **+0.71** [0.57, 0.86] (n = 3,049); continuation **+0.66** [0.23, 1.12] (n = 397); punctuation +0.41 | ✗ **Wrong.** Coordination is general sequence-level consistency, not subword stitching |
| Repetition = over-coordination, so neighbour-reading **adds** repetition (cf. Zhang et al. 2607.00588) | Immediate repeat rate y_j = y_{j−1}: N = 0 cells **2.6–2.9%**; N = 1 cells **1.1–1.8%**; full − (N cut) = **−1.4 pp** [−1.9, −0.9] | ✗ **Opposite.** Neighbour-reading **prevents** within-block duplication |

**Interpretation.** These are the two classic symptoms of the **non-autoregressive "multi-modality problem"** (Gu et al. 2018; Mask-Predict, Ghazvininejad et al. 2019): when positions predict independently, they (a) mix modes, giving hybrids, and (b) duplicate the same high-probability token. Neighbour-reading of self-conditioning fixes **both**. So SC is a learned solution to the NAT multi-modality problem, carried by cross-position messages.

This does not contradict Zhang et al.: their repetition is long-range 4-gram self-repetition across blocks in unconditional ELF; ours is within-block duplication. *(To check: does N affect cross-block repetition?)*

---

## 2. First-principles account: why would *neighbours'* feedback matter but not one's own?

**Setup.** At step k, position j sees the noisy states x(k) of all positions plus the SC message q(k−1), the previous step's predicted distribution embedded as qE·W2. Both q_j(k−1) and q_i(k−1) were computed from x(k−1) ≈ x(k).

**A data-processing puzzle.** q(k−1) is a deterministic function of x(k−1), so in pure information terms it adds almost nothing beyond x(k). The benefit therefore has to be **computational** (it saves work the network cannot otherwise do in one pass), not informational. Two candidate accounts make different predictions:

**Account A: depth shortcut ("cross-step skip connection").**
- q_i is the **output of a full forward pass** for position i. Reading it, j gets i's *finished* conclusion at layer 0.
- Without SC, j can only read i's *partial* layer-ℓ state through attention.
- Own feedback q_j is redundant, because j's own forward pass recomputes its own conclusion anyway. Neighbours' conclusions are expensive to recompute *inside* j's stream. Hence O ≈ 0, N large.
- **Predictions:**
  - (A1) the feedback-reader heads sit in **early layers**;
  - (A2) the N effect persists at **all** denoising steps, including late ones;
  - (A3) deeper models need it less: consistent with the **lower N share at 7B** (55–80% vs ≈100%).

**Account B: early information ("symmetry breaking before commitment").**
- Early in denoising, x is mostly noise, but q is already sharp (K12: q leads x by 0.1–0.2 probe accuracy at every step).
- Positions must agree on a mode **before** their states drift apart; each x moves toward its own prediction, so the drift is self-reinforcing.
- **Predictions:**
  - (B1) the N effect concentrates in an **early-to-mid critical window** of steps and vanishes late;
  - (B2) layer location is unconstrained;
  - (B3) the effect is larger for genuinely ambiguous (multi-modal) contexts.

**The decisive experiment:** a **step × layer map of the N effect**. Cut neighbour-feedback reads only in step window s and layer band ℓ. A → early layers, flat over steps. B → early steps, any layer. Both accounts are mechanistic stories in the literature (A ≈ SC as depth/scratchpad, 2610.19628; B ≈ critical windows in diffusion). Telling them apart is a conceptual contribution.

**A third view: message passing.** With SC, parallel denoising becomes an **iterated exchange of beliefs**: each step, every position updates its belief given its neighbours' last beliefs. That is the form of mean-field / loopy belief propagation, unrolled over denoising time. Without SC, positions only see each other's noisy states, a much lossier channel. This framing makes predictions:
- (M1) agreement between positions grows over steps like a consensus process (measurable: cross-seed mutual information between positions' argmax per step);
- (M2) **editing a neighbour's message steers j**: transplanting neighbour feedback from a run that committed to mode B should flip j toward B. This is the vivid causal demo.

**A spectrum that unifies decoders:**
- independent parallel (no SC) → **soft iterative messages (SC)** → hard sequential commitment (AR).
- Question: does SC make decoding **emergently ordered**, e.g. position 0 committing first and the others following, or is it symmetric consensus? Measure the per-position commitment step and the **source-resolved N** (does j rely on left or right neighbours?). If right neighbours matter as much as left ones, that's non-causal coordination AR can't do. If left dominates, SC is "soft autoregression over denoising time".

---

## 3. Candidate experiments (exploration split unless noted; ranked by insight per GPU-hour)

| # | Experiment | Cost | What makes it interesting |
|---|---|---|---|
| E1 | **Step × layer map of N** (windowed neighbour-feedback cuts) | ~2–4 GPU-h | Distinguishes depth-shortcut vs. critical-window accounts; feeds the head search (which layers) |
| E2 | **Source-resolved N:** cut reads from left vs. right neighbours separately | ~1 GPU-h | "Soft autoregression" vs. genuine bidirectional consensus |
| E3 | **What is in the message:** replace neighbour q by (a) top-1 one-hot, (b) entropy-matched vocabulary-permuted q (keeps confidence, destroys content), (c) q from another document | ~1–2 GPU-h | Content message vs. confidence/timing signal |
| E4 | **Message steering:** transplant neighbour feedback from a mode-B run into a mode-A run (bimodal prompts) | ~1 GPU-h | A causal demonstration figure: editing what a neighbour said last step changes what j says |
| E5 | **Weight-space prior for feedback-reader heads:** align W2's column space with each head's W_K / W_V (zero GPU) | ~0 | Candidate heads before patching; does weight geometry predict the causal ranking? |
| E6 | **Dose–response:** scale only neighbour feedback by α ∈ [0, 2]: coherence, hybrids, repetition | ~1 GPU-h | Is more neighbour feedback better? A sampler knob (the better-sampler upside) |
| E7 | **Toy model:** a tiny flow LM on a synthetic bimodal task, trained with own-only vs. neighbour-only SC | ~CPU/Mac | Shows the mechanism is a property of the algorithm, not of Lafa (also answers "one model family" cheaply) |
| E8 | **Consensus dynamics:** per-step cross-seed agreement between positions, SC vs. no-SC | ~1 GPU-h | The message-passing picture (M1) |

### E5 result (zero GPU, 2026-11-17)
- **Measure:** the gain with which each KV head's (RMSNorm-scaled) k_proj / v_proj picks up the directions written by `sc_lin_b` (W2) for typical messages, relative to typical centred token-embedding directions. 0.5B: 24 layers, 2 KV heads per layer.
- **Layer 0 is specifically tuned to the SC channel:** v_proj gains are 1.48 / 1.33 and k_proj 1.31 / 1.33. All other layers sit at ≈ 1.0 (max ≈ 1.15 at L1).
- **Caveat:** the direct-path analysis is exact only at layer 0. Deeper layers see a transformed residual, so ≈ 1.0 there is weak evidence.
- **Reading:** consistent with Account A1 (neighbour feedback read at the **input**). It gives the head search a prior: start the layer sweep with L0–L1.

---

## 4. Literature review (2026-11-17; abstracts only; **verify every citation before use**, as with the earlier review)

The review independently converged on the same framing: **self-conditioning = belief propagation / mean-field message passing unrolled over denoising time**, curing the NAT multimodality problem.

| Source | Principle | Experiment it gives us |
|---|---|---|
| Gu et al. 1711.02281 (NAT) | The "multimodality problem" (hybrids such as "Danke Dank", plus repeated tokens) comes from conditional independence | **Lineage + framing.** Our §1 shows SC's neighbour-reading fixes *both* symptoms |
| Shai et al. 2602.02385 | Transformers prefer **factored** per-position representations even when the factors are dependent (lossy) | Probe the joint-answer subspace vs. per-position subspaces at j; does the joint posterior appear only when N is on? |
| Garnier-Brun, Mézard et al. 2408.15138 | Transformers trained on tree data implement **exact BP** across layers | **Toy model:** a 4-variable factor graph with agreement potentials; does q at step k match BP marginals after k rounds? (= E7, sharpened) |
| Raya & Ambrogioni 2305.19693; Biroli, Mézard et al. 2402.18491 | Generation passes a **symmetry-breaking bifurcation** ("speciation") | Cross-position Jacobian of the A−B projection near commitment: a coupled block (SC) vs. independent symmetry breaking (no SC → hybrids) |
| Li & Chen 2403.01633; Sclocchi et al. 2402.16991, 2410.13770 | Features fix in **critical windows**; forward–backward re-denoising shows correlated chunk switching | (i) windowed N patch (= E1, step axis); (ii) re-noise / re-denoise: P(switch at j \| switch at i) with vs. without N, a behavioural coordination signature with no head labels |
| Feng & Steinhardt 2310.17191 | **Binding IDs**: abstract vectors tag which entity an attribute belongs to | The A−B mean-difference of neighbour qE as an "answer-ID" vector; does it transfer across unrelated pairs? (= E4, sharpened) |
| Catruna & Radoi 2607.15893 | Masked DLMs learn previous-/next-token and bidirectional induction heads | Classify reader heads by offset (j−1, j+1, whole block); do they pre-exist in base Qwen2.5? (pairs with E2) |
| Piotrowski, …, Shai 2502.01954 | Attention does constrained belief updates: QK = positional, OV = content | Split reader heads into QK vs. OV; swap only the neighbour's qE value |
| Blayney et al. 2604.11791 (looped LMs) | Each layer converges to a fixed point; head behaviour stabilizes as the loop converges | Do reader heads stabilize *before* the block commits? (drivers vs. consequences) |
| Subliminal Clocks 2607.01774 | DLMs carry a steerable timestep subspace | Control: hold the time direction fixed when patching across steps |
| Lu 2604.23235 | Direct vs. collateral effects of re-masking in masked DLMs (mostly local) | Masked-diffusion baseline: Lafa's collateral share should be large |

**No 2026 paper found on head-level analysis of a continuous-flow LM or on SC circuits.**

---

## 5. Priority plan (merged)

| Rank | Experiment | Why first |
|---|---|---|
| 1 | **Step × layer map of N** (E1) | Separates depth-shortcut vs. critical-window accounts; directly tells the head search where to look (E5 prior: L0–L1) |
| 2 | **Symmetry-breaking coupling** (Jacobian of A−B projections; forward–backward joint switching) | Turns "San York" into "independent symmetry breaking in a factorized system", with physics theory attached |
| 3 | **Message steering / answer-ID vector** (E4 + binding IDs) | Causal demo figure; tests abstract content vs. token identity |
| 4 | **Head search** (prereg §2.4) + offset classification + QK/OV split + base-Qwen comparison | The circuit result; the "soft autoregression" (left-reading) vs. bidirectional-consensus question |
| 5 | **Toy factor-graph model vs. exact BP** (E7) | Theory anchor; the mechanism as a property of the algorithm; partial answer to "one model family" |

**Framing (candidate headline):** *Self-conditioning turns a factorized parallel decoder into belief propagation unrolled over denoising time. Each position broadcasts its current belief, and neighbours read it through attention (layer-0 KV heads are tuned to this channel). Without it, positions break symmetry independently, producing both symptoms of the NAT multimodality problem: mixed answers and duplicated tokens.*

---

## 6. E1 interim (152 / 300 exploration docs, 2026-11-17): the step × layer map

Measure: cut j's reads of neighbours' SC-carrying K/V. Share = effect / effect of cutting everything (dep_{2,3}; coh similar).

- **When (steps):**
  - last quarter (steps 24–31) **0.90** [0.72, 1.13];
  - steps 16–23: 0.05;
  - steps 8–15: 0.09;
  - steps 0–7: 0.02.
  - **The reading is a late-stage process.** This refutes Account B (early warning) and Account A2 (flat over steps).
- **Where (layers):**
  - distributed;
  - largest at L8–11 (0.25), then L12–15 (0.18), L16–19 (0.16), L0–3 (0.14), L4–7 (0.06), L20–23 (0.03);
  - **L0 alone ≈ 0.01**, so the E5 weight prior was wrong.
  - The bands add up to ≈ 0.8, so the effect is roughly additive across layers.
- **Peak cell:** steps 24–31 × L8–15 = 0.36.
- **Caveat to resolve (E1b, running):** late cuts act directly on the final readout, while earlier cuts may be repaired by later steps. E1b splits the last step from steps 24–30, and cuts each layer singly (100 docs, smooth drop metric).
- **Emerging picture:** not a depth shortcut and not early symmetry breaking. Positions check neighbours' fed-back beliefs during the **final commitment phase**, through mid-layer attention spread across many layers. *If confirmed:* the "distributed" branch of L1 becomes more likely, and the head search should screen mid layers (4–19) at late steps.

---

## 7. E1c: discriminating WHY the neighbour reading is late (predictions committed before data, 2026-11-17)

**Fact:**
- With the default `warp` grid, t rises only 0 → 0.44 over steps 0–30 (each step moves the state 1–9% toward the prediction).
- Step 31 then jumps to t = 1.
- The output is always the final step's argmax prediction.

**Hypotheses:**
- **H-readout:** coordination is needed where the output is read, i.e. at the final prediction(s), whatever the noise level. Earlier cuts are repaired by later re-reading.
- **H-gate:** the network gates neighbour-feedback reading on the noise level t (learned through time conditioning). Reading switches on above some t\*.
- **H-clean:** feedback is read because the states are uninformative. Already strained: at PR (noise 0.1, informative states) neighbour reading still carries 73% (S12).

**Manipulation:** rerun the step-window map (base, all, s0-7, s8-15, s16-23, s24-31, s24-30, s31) under two more grids, on the same 100 exploration docs. In both, the final prediction is at the last step:
- `uniform_range` (t: 0 → 0.2, never high);
- `uniform` (t: 0 → 1; 0.44 is reached at step ≈ 14).

**Predictions:**

| | `uniform_range` (t ≤ 0.2) | `uniform` (t → 1) |
|---|---|---|
| H-readout | Late windows still carry most of the effect (s24-31 ≫ early) | Late windows still carry most of the effect |
| H-gate | Total N effect small (t never reaches the gate) | The effect shifts **earlier**, to wherever t > t\* (≈ steps ≥ 10–14), spread over the steps after that point |
| H-clean | Large effect, late | The effect shrinks late, as the states become informative |

**Decision rule:** whichever hypothesis predicts the observed pattern in **both** grids. If none does, report "unresolved" and say so.

**Measure (fixed before data):** the decision uses **dep_{2,3}** (and coh) of the generated tokens, n = 300 exploration docs per grid.
- The smooth "drop" metric saturates under `uniform`: at the final step, t ≈ 0.97 and the prediction is near-certain whatever happened earlier.
- **Implementation check:** Lafa's reference sampler treats a 1-D grid as the same t for every position. `run_block_headcut` now broadcasts 1-D grids identically.

### 7.1 E1c RESULT (n = 300 docs per grid; Prime A100 under MPS; decision per the pre-committed rule)

| Grid | Where the N effect (dep_{2,3}, share of the full cut) sits | Total |
|---|---|---|
| `uniform_range` (t ≤ 0.2: states never clean) | s24-31 **0.95** [0.82, 1.08], of which s31 0.70; s0-23 ≈ 0 | +1.31 [0.91, 1.73] |
| `uniform` (t → 1: states clean by the late steps) | s8-15 **0.63** [0.44, 0.84]; s16-23 0.16; s0-7 0.10; **s24-31 = 0.00 exactly** | +1.09 [0.73, 1.46] |

(coh: same pattern.)

**Verdict:**
- H-readout ✗ (late = 0 under `uniform`);
- H-gate ✗ (large effect under `uniform_range`, although t never exceeds 0.2);
- **H-clean ✓ in both grids.**

**Mechanism statement:** a position reads its neighbours' fed-back predictions as a **substitute for the neighbours' states while those states are still uninformative**. Once the states become readable, the feedback channel stops mattering (exactly 0 effect).
- The default `warp` schedule keeps states noisy until the final jump, which is why the reading appears late there (E1, E1b).
- Consistent with S12: at low inference noise (states more informative), the N share is lower (0.73) than at MR (≈ 1.0), i.e. partial substitution.

**Refinement to test:**
- Under `uniform`, the effect peaks at steps 8–15 (t ≈ 0.25–0.47) rather than at steps 0–7. So the feedback matters most while positions are **committing** and before the neighbours' states are readable.
- A direct test: per step, measure the linear decodability of each neighbour's final token from its state x vs. from its feedback q, and correlate the per-step N effect with (q-decodability − x-decodability).

## 8. E1b result (n = 100 docs, smooth drop metric)
- **Steps (warp):**
  - final step alone 0.05;
  - steps 24–30 0.28;
  - steps 28–31 0.26;
  - steps 16–23 0.17.
  - So it is not a readout artifact.
- **Single layers:**
  - **L8 0.20** [0.09, 0.32], L19 0.12, L9 0.07, L15 0.05, L3 / L5 / L10 0.04;
  - the sum of single-layer drops is 0.73 of the full cut, so the effect is **mostly additive and concentrated**, not redundant;
  - the dep metric was too noisy for single layers, so use drop for screening.
- **Head screen 1 launched:** single heads in L8, L19, L9, L15, L3, L5, L10, L12, L18 (126 heads), 100 docs, on the pod.

## 9. E1 final (n = 300 docs; CARC)
- Steps 24–31 **0.74** [0.57, 0.94] (coh 0.58); steps 16–23 0.14; steps 8–15 0.10; steps 0–7 0.01.
- Layer bands: L8–11 **0.19** (largest), L0–3 0.13, L4–7 0.10, L12–15 0.09, L16–19 0.06, L20–23 0.02.
- Peak cell: steps 24–31 × L8–15 = 0.26.
- Consistent with E1b (L8 hotspot) and with E1c (late because the default schedule keeps states uninformative until the end).

## 10. Head screen 1 (single-head cuts; 126 heads in L8, L19, L9, L15, L3, L5, L10, L12, L18; 100 docs; smooth drop metric)
- **Top heads (share of the full cut):**
  - L19h3 0.051 [0.009, 0.107];
  - L9h13 0.043;
  - L8h8 0.037;
  - L8h7 0.033;
  - L5h0 0.033;
  - L10h2 0.032;
  - L3h7 0.025;
  - L12h8 0.020;
  - the rest ≤ 0.015.
- **Additive approximation** (sum of single-head shares): top 5 = 0.20, top 10 = 0.30, top 33 = 0.47. Below the prereg's 50% bar at the 33-head cap *if* the effects are additive.
- **Within-layer synergy:** cutting the whole of L8 = 0.196, vs. 0.115 for the sum of its heads, so the joint cuts may exceed the additive sum. L19 and L9 are additive.
- **Next (running):** joint top-k cuts (k = 1…33, ranking from these 100 docs), evaluated on **200 different exploration docs**, so the selection is not evaluated on its own data. k = the smallest k with joint share ≥ 0.50 on dep_{2,3} (the prereg estimand; drop also reported), capped at 33.
- **Prereg note (to log in §12 before circuit-validation):**
  - ranking used the smooth drop metric (dep is too noisy per head);
  - the screen was restricted to the 9 layers selected by E1b;
  - k is selected on a held-out exploration subset.

## 11. G-a: causal substitution control (predictions committed before data, 2026-11-18)
**Question:** is "feedback stops mattering late under `uniform`" substitution (j reads the neighbours' now-informative states instead), or trivial ("nothing about the neighbours matters late")?

**Manipulation** (all layers; only what query j reads changes; the neighbours' own trajectories are untouched). In window w, j reads the neighbours from a pass where they have:
- **nofb:** state intact, feedback removed (= the N cut);
- **nostate:** feedback intact, state replaced by pure noise (initial-noise scale);
- **neither:** noise state, no feedback.

Windows: late = steps 24–31, early = steps 8–15. Grids: `uniform`, `warp`. n = 300 exploration docs per grid; outcome dep_{2,3} (coh also reported).

**Predictions:**

| | `uniform` late (states informative) | `warp` late (states uninformative) | `uniform` early |
|---|---|---|---|
| **Substitution** | nofb ≈ 0; **nostate > 0; neither > 0** | **nofb > 0**; nostate ≈ small (states already uninformative); neither ≥ nofb | nofb > 0 |
| **"Nothing matters late"** | nofb ≈ nostate ≈ neither ≈ 0 | — | — |

**Decision:**
- Substitution is supported if, in `uniform` late, nostate and neither are > 0 (95% CI excludes 0) while nofb ≈ 0, **and** in `warp` late, nofb > 0.
- "Nothing matters late" is supported if all three are ≈ 0 in `uniform` late.
- Otherwise: unresolved.

### 11.1 G-a RESULT (n = 300 per grid; Prime A100) → by the pre-committed rule: **UNRESOLVED**

| dep_{2,3} | nofb | nostate | neither | redundancy index (neither − nofb − nostate) |
|---|---|---|---|---|
| `uniform` late | +0.02 [0.00, 0.06] | +0.01 [−0.00, 0.04] | **+0.15** [0.05, 0.28] | +0.12 |
| `warp` late | **+0.81** [0.54, 1.11] | **+1.31** [0.98, 1.66] | **+3.02** [2.55, 3.51] | +0.90 |
| `uniform` early | +0.65 | +0.56 | +1.42 | +0.21 |

(coh: same pattern.)

**Scored against the predictions:**
- "Nothing matters late" ✗: `uniform` late neither > 0.
- Strict substitution ✗: `uniform` late nostate ≈ 0 (predicted > 0); `warp` late nostate is large (predicted small).

**Post-hoc account (must be confirmed on new data, G-c): two redundant channels.**
- Position j obtains its neighbours' identity through their **states** and through their **fed-back predictions**. Either one alone often suffices (an OR structure); removing both is always super-additive.
- The feedback's causal contribution = what the state channel cannot supply. That is ≈ 0 once states are clean (`uniform` late), and large while states are noisy (`warp` late, which is also why the default schedule's reading is late).
- This explains E1c and G-a jointly, and it predicts a positive redundancy index in every window.

## 12. L1 RESULT (pre-registered, circuit-validation, n = 1,777; frozen analysis `analysis/l1.py`)
- **dep_{2,3} (decision measure):** H\* share 0.438, random layer-matched share 0.061. **θ = 0.377 [0.298, 0.454]**; one-sided LB (α = 0.025) **0.298 vs the bar 0.30** → **L1 INCONCLUSIVE** (missed by 0.002). No data extension, per prereg §6.
- **coh (reported):** H\* 0.463 vs random 0.035; θ = 0.427 [0.379, 0.475].
- **Reading:** H\* is ~7× more important than random head sets of the same size and layer profile and carries ~44–46% of the N effect. The pre-registered margin is not established at the decision measure.
- **Operations note:** 8 docs were computed twice when a crashed shard was resumed; the first copy was kept (dedupe by task_id; `l1_merged`).

## 13. Toy model (gap G-e; `toy/toy_flow.py`; 3 seeds per variant; exploratory)
**Setup:**
- Lafa-style blockwise flow LM (4 layers, d = 128) trained from scratch.
- Task: 64 prompts, each with two valid 4-token answers differing at every position.
- SC routed **exactly** per variant with a three-stream construction (S: no feedback; F: feedback everywhere; O: the variant's output stream).
- Variants are applied at **training and inference**.

| Trained with | valid | hybrid |
|---|---|---|
| no SC | 0.26 | 0.72 |
| **own-only** SC | **0.30** | 0.68 |
| full SC | 0.53 | 0.44 |
| **neighbour-only** SC | **0.72** (0.71–0.75) | **0.25** |

**Reading:**
1. Own feedback is ≈ useless even when trained for.
2. Neighbour feedback carries the whole benefit. The mechanism is a property of the algorithm, not of Lafa.
3. **New:** neighbour-only SC *beats* full SC (non-overlapping seeds), so own feedback may *hurt* coordination. Hypothesis: own feedback makes each position lock onto its independently preferred mode (self-reinforcement), resisting consensus.

**Follow-ups:**
- (a) Measure per-step commitment / mode flips under own vs. neighbour feedback.
- (b) In Lafa, test inference-time "neighbour-only SC" (remove each position's own feedback) as a **sampler improvement**. If it lowers hybrids/garbles, it is the practical upside.
- **Follow-up (b) in Lafa** (zero-GPU re-analysis of the confirmation 2 × 2 × 2 cells, exploratory). Own feedback removed, neighbours kept (O0 N1 F1), minus full (O1 N1 F1):
  - **MR** (2,000 docs): dep **+0.134** [0.028, 0.240], coh **+0.230** [0.118, 0.338]. Removing own feedback *helps*, as in the toy.
  - **PR / guidance on** (750 docs): dep −0.04 [−0.14, 0.07], coh −0.14 [−0.24, −0.04]. No help.
  - **So:** own feedback's harm is regime-dependent (present at high noise / no guidance). Not a free improvement to the deployed sampler; report it as a mechanism finding.

## 14. L2 RESULT (pre-registered; 1,888 circuit-validation prompts): **SUPPORTED**
- Non-valid answers: base 17.6%, random 26-head sets 20.0%, **H\* 48.8%**. θ = +28.9 pp [24.6, 33.3] vs a 3 pp bar.
- Hybrids 6.2% → 11.6%; garbles 11.3% → 37.2%. Examples: "Chinua Achebe" → "Chin Gaiman"; "Franz Kafka" → "Fran King"; "Salman Rushdie" → "Sal Twain".
- **Headline:** cutting the feedback reads of 26 identified heads (~8% of heads) nearly triples cross-answer and garbled outputs at deployed settings. Random layer-matched heads have no effect.
- L1 (mechanistic share) remains inconclusive under our stricter logged rule; it would pass under the original Holm step-down (LB 0.311).

## 15. E4 steering by message transplant (predictions committed before data, 2026-11-18; exploratory; `src/steer.py`)

**Design:**
- Exploration bimodal prompts whose two modes both span ≥ 2 tokens; MR regime, 0.5B.
- For each prompt, run seeds 1–12 and label each run by its first two tokens: committed to mode X, committed to mode Y, or neither.
- **Recipient** = the first X-committed seed. **Y-donor** = a Y-committed seed. **X-donor control** = a different X-committed seed.
- Target j = 1 (the second token of the answer). Neighbour 0 carries the first word.
- **Transplant:** j reads its neighbours through (recipient states, donor feedback), all layers, in window w ∈ {all steps, late 24–31, early 0–15}.

**Outcome:** flip(j) = 1 if the recipient's token at j becomes Y's second token.

**Predictions:**
- (P-a) flip rate with the Y-donor (all steps) ≫ with the X-donor control (≈ 0).
- (P-b) late-window transplant ≥ early-window transplant (E1: the reading is late under `warp`).
- (P-c) Flips should produce **hybrids** (recipient's X first word + Y's second token, e.g. "Key Jose"): the transplant writes the neighbour's *message*, not its state. A causal "San York" generator.

**n:** up to 200 eligible exploration prompts (Mac, MPS).

## 16. G-b RESULT: external-judge decomposition (Mistral-7B, 2,000 confirmation docs; exploratory re-analysis)
- T (coh, full − all-off) = +1.97 [1.74, 2.20].
- **N share:** Möbius 0.80 [0.72, 0.91]; Shapley **1.04** [0.95, 1.14].
- **O share −0.09 [−0.17, −0.03]** (own feedback slightly harmful); F share 0.06 [−0.01, 0.12].
- Contrast N − 0.2T = +1.19 [1.04, 1.35].
- **Closes the circularity gap:** the neighbour-reading result does not depend on scoring with Lafa's own AR head.

## 17. Family R RESULT (pre-registered §2.5; 1,000 untouched confirmation docs): **all supported**
- R1 redundancy (`warp` late) +0.55 [0.32, 0.79]; R2 `uniform` late feedback-only cut 0.000 (≤ 0.004) while removing both +0.21 [0.14, 0.28]; R3 schedule shift +0.54 [0.41, 0.67].
- **Confirmed mechanism statement:** a position learns its neighbours' content through two redundant channels, their states and their fed-back predictions. The feedback is causally necessary exactly when the state channel cannot supply it (noisy states). Lafa's default schedule keeps states noisy until the final jump, so coordination runs through the feedback in the late steps.

### 15.1 E4 RESULT (200 exploration prompts, 62 pairs; Mac MPS): **all three predictions confirmed**
- **(P-a)** Flip to the donor's mode at j: Y-donor **39.5%** vs X-donor control **0.5%**; diff +39.0 pp [32.0, 45.5].
- **(P-b)** Late window (steps 24–31) **37.0%** vs early window (0–15) **0.5%**.
- **(P-c)** **77 / 79 flips are hybrids** (recipient's own first token + donor's second token).
- **Reading:** editing *only* what the neighbours told j last step (their fed-back predictions), with all states intact, causally produces "San York"-type hybrids in ~40% of cases, and only when done during the late commitment window. This is a direct causal demonstration of the message channel; it fits E1 (late), family R (feedback needed while states are uninformative) and L2.

## 18. ELF step 0 (gap G-e)
- **v1 (INVALID as an SC ablation):** compared SC "scale" 0 / 1 / 3 with the previous prediction always supplied.
  - ELF-B trains SC-off as a zero self-conditioning input (50% of rows), with the scale passed as a separate guidance token from a sampled range. Scale 0 is out of distribution (the guidance weight is 1 − 1/w).
  - Kept for the record: judge log-prob/token −4.83 (w = 0), −4.63 (w = 1), −4.24 (w = 3).
- **v2 (correct), RESULT:** SC off = zero input vs. SC on = previous x-prediction. Same noise and schedule; 256 samples; Mistral judge.
  - Judge log-prob/token: off −6.13, **on (w = 1) −4.67**, on (w = 3) −4.22, off (w = 3) −6.14.
  - **SC matters strongly in ELF** (+1.46 nats/token at w = 1). The scale token has no effect without feedback, which validates the corrected design.
  - **Decision: build the ELF decomposition (gap G-e)** and pre-register its predictions before running.

## 19. 7B mechanism (G-d), interim (227 / 300 exploration blocks)
- **coh:** T +2.74 [2.15, 3.35]; Shapley N share **0.73** [0.60, 0.89]; O share **0.23** [0.14, 0.31]; F 0.05; contrast +0.85 [0.44, 1.26].
- **dep:** T +0.44 [−0.22, 1.12] (not resolved, as in the 7B pilot).
- **Reading:** neighbours dominate at 7B, but own feedback contributes (~1/4), unlike 0.5B (≈ 0 or harmful).

**Plan to resolve dep at 7B (predictions committed before re-scoring, 2026-11-18):**
- (i) Re-score the existing 7B cell outputs with **S = 64** predecessor samples (instead of 16); no new generation.
- (ii) Score the same outputs with the external judge (Mistral), as in G-b.
- (iii) If still unresolved: 500 more blocks with the same design.

**Predictions:** dep T > 0 with S = 64; N is the largest Shapley share on both dep and Mistral-coh, with share ≥ 0.5; O share 0.1–0.3 (own feedback emerging at scale).

### 19.1 7B exploration, complete (300 blocks) + E1c uniform at 7B (100 docs)
- **coh:** T +2.73 [2.22, 3.24]; Shapley **N 0.76** [0.65, 0.90], **O 0.19** [0.12, 0.26], F 0.05; contrast N − 0.2T **+0.90** [0.54, 1.27].
- **dep:** T +0.37 [−0.20, 0.93] (unresolved); contrast +0.38 [0.05, 0.71].
- **E1c uniform at 7B** (share of the full N cut):
  - s0-7 **0.62–0.87** (dep / coh / drop);
  - s8-15 0.05–0.18;
  - s16-31 ≈ 0.
  - The 0.5B peak was at s8-15.
  - **The larger model's feedback reading ends earlier.** Under the two-channel account this means 7B states become informative sooner; it is a second-scale confirmation of the timing account (exploratory).
- **E1c `uniform_range` at 7B** (states never clean; 100 docs):
  - all-cut effect dep +1.58 [0.70, 2.47], coh +2.48 [1.79, 3.22];
  - the reading persists and is late-weighted (s24-31 share: dep 0.47, coh 0.30; spread over s8-31 by drop).
  - **With `uniform` (effect only in s0-7), the 7B timing tracks state informativeness exactly as at 0.5B.** The two-channel account holds at both scales (exploratory at 7B).

## 20. Family E RESULT (ELF-B, pre-registered §2.7): E-1, E-2 **not supported**
- T +1.90 [1.49, 2.31]; Shapley **N 0.35, O −0.04, F 0.69**; Möbius N ≈ 0 (N helps only with F; N × F synergy).
- **Cross-architecture picture:**
  - (i) **own feedback ≈ 0** in Lafa 0.5B (≤ 0, two judges) and ELF (−0.04); 0.19 at Lafa 7B;
  - (ii) SC acts by improving what a position learns about its neighbours, **through the feedback channel in Lafa** (N ≈ 0.75–1.0) **and mainly through the state channel in ELF** (F 0.69).
- **Consistent with the confirmed two-channel account (family R):** states and fed-back predictions carry neighbour information redundantly, and which carries SC's benefit depends on the architecture and schedule. ELF: full-sequence denoising, continuous x-prediction feedback (not an expected token embedding), logit-normal schedule.
- **Scope for any claim:** "SC is not self-refinement; it is cross-position communication, via feedback (Lafa) or via feedback-improved states (ELF)."
- **Possible follow-up (exploratory):** ELF E1c-style schedule manipulation. Does N grow when ELF's states are kept uninformative longer (e.g. `uniform_range`-like)? This would test the two-channel account's prediction in ELF directly.

### 20.1 Sensitivity analysis for family E (exploratory; prediction committed before running, 2026-11-18)
- **Concern:** the family E outcome scores token j given the cell's own decoded left context. Route F also improves the 127 neighbours' tokens, so part of F's 0.69 share may be "better context for the judge" rather than "j's own choice improved". In Lafa (4-token blocks, fixed prefix) this was negligible.
- **Analysis:** re-score each cell's token at j against a FIXED context (the O1N1F1 cell's tokens 0..j−1, same sample), so only j's choice varies (`src/elf_judge_fixedctx.py`). Same family E data; no new generation.
- **Prediction (if the confound explains the gap):** the F share falls and the N share rises, with N the largest Shapley share.
- **If the shares stay as in family E:** the state-route reading stands.
- Reported alongside the pre-registered result in either case, never instead of it.

## 21. Family E2 RESULT (ELF schedule manipulation, pre-registered §2.8): **not supported**
- latejump: T +1.34 [0.67, 2.00]; N main +0.06; **F main +0.77**.
- frontload: T +0.20 (not estimable); N −0.54; F +0.21.
- **Reading:**
  - In ELF, SC's benefit flows through the neighbours' own feedback improving the neighbours (F), even when states stay noisy. Direct feedback reading does not take over.
  - The two-channel account (confirmed in Lafa, family R) **does not predict ELF's channel choice**.
  - When ELF's states become informative early, SC hardly matters at all (T ≈ 0), which is consistent with SC mattering only while states are uninformative.
- **Next (queued):** the fixed-context sensitivity analysis (§20.1) asks whether ELF's F benefit reaches token j's own choice or mainly improves the judged context.
- **§20.1 RESULT:** fixed-context re-scoring (249 samples): T +4.26 [3.74, 4.81]; Shapley **N 0.25, O 0.01, F 0.74**; Möbius N −0.01.
  - **Prediction failed.** The context confound does not explain ELF's F dominance; F improves token j's own choice.
  - **Conclusion on ELF (robust across family E, E2 and this analysis):**
    - SC acts mainly through the neighbours' feedback-improved **states** (F);
    - direct reading of neighbours' feedback is secondary (~0.25, via interaction);
    - own feedback ≈ 0.
  - **Cross-architecture invariant:** SC's benefit is not self-refinement of the target (O ≈ 0 in Lafa 0.5B and ELF; 0.19 at Lafa 7B). It reaches each position through its neighbours. **The channel (feedback in Lafa; states in ELF) is architecture-dependent and is not predicted by the schedule account.**

## 22. 2026-11-19 confirmations
- **S7 (7B, pre-registered §2.6): all supported.** Neighbours carry the coherence benefit at 7B (Shapley N 0.86; Mistral 0.76); own feedback contributes a small but positive share (0.07; S7-3 p = 0.046). The exploratory 0.19 own share was an overestimate.
- **L1-R (pre-registered §2.9): supported.** The 26 frozen heads carry 49.5% of the neighbour-reading effect vs 4.2% for layer-matched random sets; θ 0.45 [0.39, 0.52]. Together with L2 (behaviour), **H\* is a confirmed distributed circuit.**

## 23. ELF decoder test (E3, 2026-11-19)
- **Pre-registered rule: refuted.** The F share within the swap variant is 0.84.
- **But the share criterion was the wrong quantity.** Fixing neighbours' final latents at decode removes **83% of SC's whole effect on token j** (T 4.51 → 0.77) and **82% of the F route's absolute effect** [0.74, 0.90] (exploratory). The remaining 17% keeps the same F-dominated split, which is why the share did not move.
- **Reading:** in ELF, neighbours' feedback improves their final latents, and token j reads those latents at the decode pass, where SC is zeroed by design (2605.10938). This is the hypothesis; E3b (§2.11, fresh seed) is its pre-registered test.

## 24. E3b (2026-11-19): ELF decoder hypothesis confirmed (pre-registered, fresh seed)
- Freezing neighbours' final latents at ELF's decode pass removes **89% [81%, 97%]** of the state route's absolute effect (and 89% of SC's whole effect on token j).
- **Confirmed reading:** in ELF, feedback improves the neighbours' final latents, and token j reads them at the SC-free decode pass. That is where j commits.
- This turns the ELF "difference" into a specific, confirmed mechanism, consistent with the access account: the route follows where the feedback is readable at commitment.
