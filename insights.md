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
