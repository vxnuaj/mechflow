
# Research Proposal: Does Self-Conditioning Coordinate Parallel Predictions? Cross-Position Mode Commitment in Continuous-Flow Language Models

**Target venue:** ICML 2027 (deadline ≈ late January 2027, ~10 weeks from writing)
**Status:** Proposed; pilot not yet run
**Date:** November 13, 2026
**Primary model:** Lafa (Large Flow Language Models) [1]
**Area:** Mechanistic interpretability, continuous-flow / diffusion language models

---

## 0. Fact-Checks (Phase 2 resolved 2026-11-13; details in `phase2.md`)

> **Note:** the design in §§4–6 below is the original proposal. It is **superseded by `phase4.md`** (the current working spec, built on `phase1.md`) (claim, rivals, kill criteria and interventions), as amended by `phase2.md` §3. Read those first.

- [x] **Ablation number.** 32.3 → 38.1 (+5.8 pp), Table 3, 0.5B, B = 4, 32 Euler steps, g = 8.
- [x] **Train-time or inference-time?** **Inference-time**: the same model with self-conditioning switched off at decode.
- [x] **Training drop.** p_sc = 0.5, decided **once per optimizer step for the whole batch**. The dropped input is **plain x** (W₁ and W₂ both skipped). Block-level "off" is in distribution; per-position mixing is not (A7).
- [x] **How q is computed in training.** Stop-grad pass on the same noisy I_t (same t). At inference, q is one step stale, and the first q comes from a bootstrap pass at t = 0.
- [x] **Block alignment.** At inference, blocks start at the prompt end, so alignment is controllable.
- [x] **Further facts** that change the design (`phase2.md` §3):
  - asynchronous left-to-right time;
  - inference noise 0.1 vs. 4.0 in training;
  - guidance g = 8 with separate feedback per branch;
  - per-position argmax readout.
- [x] **References verified** (Phase 3; `phase3.md` §1). Phase 3 verdicts: M2 mechanism open (phenomenon known); M3 demoted to moderator; new rival M5 (Shabalin et al.).

## 1. Research Question

In a blockwise flow LM, the B tokens of a block are predicted in parallel. Parallel prediction has a well-known failure mode: when the joint distribution over a block is multimodal, independently sharpened marginals can mix modes. For example, "New York" and "San Francisco" are both plausible, but parallel marginals can produce "New Francisco".

**Question.** Does Lafa's self-conditioning channel let positions within a block read each other's previous commitments and converge on a consistent joint mode? And is this cross-position effect a substantial part of why self-conditioning helps?

The project has two goals. The first is a clean causal decomposition of self-conditioning's benefit into same-position and cross-position parts. The second, conditional on a large cross-position part, is a mechanistic account of how commitment propagates across positions and denoising steps.

---

### 1.1 Competing explanations for the self-conditioning gain (added after Phase 2)

Our cross-position claim is one of four candidate mechanisms for the +5.8 pp gain. The project now **decomposes the gain causally across them and dissects whichever wins**; the full rewrite is in Phase 4 (see `TODO.md`).

| | Mechanism | Distinguishing prediction |
|---|---|---|
| M1 | Per-position stabilization (skeptic) | Gain survives at g = 1; per-token CE improves; hybrid rate unchanged. |
| M2 | Cross-position coordination (original claim; `phase1.md`) | Gain survives at g = 1; hybrid rate drops with feedback, concentrated on coupled blocks. |
| M3 | Guidance anchoring (unguided feedback steadies the 8× guided step) | Gain mostly disappears at g = 1. |
| M4 | Tie-breaking at low noise (sampling starts at 0.1·σ) | Gain larger at noise 0.1 than 4.0; without feedback, positions stall between answers. |
| M5 | Cross-step trajectory stabilization (Shabalin et al. [12]; added after Phase 3) | Acts through each position's *own* feedback over time; the neighbour direct effect ≈ 0; hybrids drop through own-feedback effects. |

Phase 3 status: M1 best-supported in the literature; **M2 mechanism open**; M3 untested but contradicted as the main mechanism (self-conditioning helps without guidance in SED/CDCD/LangFlow) → moderator; M4 partly known; M5 new first-class rival. See `phase3.md`.

## 2. Background

### 2.1 Lafa

Lafa [1] adapts pretrained AR transformers (Qwen2.5, Qwen3) into blockwise continuous-flow denoisers. Attention is bidirectional inside each block and causal over the prefix. Released checkpoints (0.5B, 7B, 8B) use 4-token blocks [2]. The self-conditioned denoiser input is

$$
\tilde{x}_k = x_k + x_k W_1^\top + (q_{k-1}E)W_2^\top
$$

where x_k is the current continuous state, q_{k-1} the previous step's predicted token distributions, and E the embedding matrix.

### 2.2 Why "latent vs. feedback information" is the wrong frame

Self-conditioning is typically trained by computing q with a stop-gradient pass on the *same* noisy state (to verify for Lafa; §0). Under that scheme, q is a deterministic function of x_k at training time, so in distribution it carries no information that x_k lacks. Its value is computational: it hands the next evaluation the output of a full previous forward pass, which effectively adds depth across steps [8].

That pass already mixed information across all positions in the block. So the feedback at position i, q_i^{k-1}, is not "position i's private signal". It is position i's *resolved prediction after having seen its neighbours*. Feeding it back gives position j a chance to condition on the neighbours' current commitments rather than their noisy states. This is the specific cross-position computation we hypothesize.

Consequence for the design: we do not try to measure how much information q holds beyond x_k. We measure how much of self-conditioning's benefit comes from position j seeing *other positions'* feedback, versus its own.

---

## 3. Related Work and Positioning

- **Lafa [1].** Establishes the architecture and that self-conditioning helps on benchmarks. It does not explain why.
- **Fixed-point view of self-conditioning (2607.00714) [10].** Frames self-conditioned flow LMs as solving fixed-point iterations and reports that self-conditioning can be *removed or compressed without degradation*. **This is the most direct threat to our premise.** If it transfers to Lafa, the cross-position benefit may be small. Our same-only/cross-only decomposition (§4, Exp 1) tests this directly, and a clean negative result is still informative.
- **Untied self-conditioning [6].** Attributes few-step degradation to a train/inference mismatch in the self-conditioning input after a solver step. They fix it at inference by feeding the raw (uncorrected) prediction back. Relevant for our 4- and 8-step robustness conditions.
- **One-dimensional self-conditioning attractor [7].** Shows that repetition in continuous diffusion LMs comes from a contractive attractor along a single direction of the feedback loop, which can be removed by projecting it out. This is prior mechanistic work on self-conditioning. Ours differs in targeting a *beneficial, cross-position* function rather than a degenerate one.
- **Self-conditioning and depth [8].** Theoretical motivation that feedback across evaluations adds computational capacity.
- **Design space of continuous diffusion LMs [5].** Gives alternative explanations (train/inference mismatch, token identification from noisy states) that our same-position condition must absorb.
- **AR→diffusion mechanism shift [4]; ICL circuit [3]; DLM-Scope [9].** Methodological precedent. Showing that the flow model's circuits differ from the AR base is *not* a contribution by itself.

**Claimed gap (to keep verifying):** no prior work decomposes self-conditioning's benefit into same-position and cross-position parts in a blockwise flow LM, or connects it to joint-mode consistency in parallel decoding.

---

## 4. Hypotheses

> **Superseded:** see `phase4.md` §3 for the revised H1–H4 and the position-specific predictions.

**H1 (decomposition).** A substantial share of self-conditioning's benefit requires *cross-position* feedback. "Same-only" self-conditioning, where position j receives only its own feedback, recovers significantly less of the full-vs-none gap than "cross-only" self-conditioning.

**H2 (mode commitment).** The cross-position benefit concentrates on inputs whose block-level continuation is multimodal with coupled tokens. It shows up as improved *joint consistency* (fewer mixed-mode blocks), not just per-token accuracy.

**H3 (dynamics).** Commitment propagates. Early in denoising, one position's feedback sharpens first and the others follow, conditioned on it. Swapping in a source position's feedback from the competing mode flips its partners to that mode.

**H4 (mechanism, conditional on H1–H3).** The propagation is carried by a small, identifiable set of attention heads that read neighbours' feedback-derived representations, and it generalizes across task families and to 7B/8B.

H1 is the go/no-go hypothesis. H4 is supporting work, not the core bet.

---

## 5. Experiments

### Setup (all experiments)

Use frozen weights and the official inference code. Defaults: deterministic Euler ODE sampling, 32 steps, guidance scale 1.0, fixed seeds. Robustness conditions: 4 and 8 steps, with and without untied self-conditioning [6]. "Feedback off" always means the in-distribution training-time drop condition from §0, and the alternatives (zero (qE)W₂ᵀ only; zero the whole transform) are reported as sensitivity checks.

### Experiment 1: Same-Only vs. Cross-Only Self-Conditioning (go/no-go, ~1 week)

With B = 4, cross-position feedback can be separated from same-position feedback without circuit analysis:

- **Full:** all positions receive their feedback.
- **None:** no position receives feedback.
- **Same-only:** for each target j, run a forward pass in which only position j receives its feedback term and all others get the "off" input; keep only output j. That is 4 passes per step.
- **Cross-only:** for each target j, run a pass in which position j's feedback is off and all other positions' feedback is on; keep output j.

Run full trajectories under each condition and measure:
- end-task accuracy on GSM8K and HumanEval subsets (to connect to the paper's numbers);
- joint consistency on the multimodal tasks below;
- per-step target cross-entropy.

Caveat: in same-only, other positions' residual streams also change because their own feedback is removed. This is the intended counterfactual ("j sees neighbours without their feedback"), but state it explicitly.

**Decision rule:**
- **Same-only ≈ full:** H1 fails. Write up the decomposition as a short negative or analysis result, or pivot.
- **Cross-only recovers a large share of the full–none gap:** proceed to Experiments 2–4.

Pre-register the threshold for "large share" and the evaluation subsets before running.

### Experiment 2: Joint-Consistency Tasks

Build prompts where the prefix admits **several valid but internally coupled** continuations that fit inside one aligned 4-token block:

- **Multi-token entities:** an ambiguous prompt whose answer is one of several multi-token names ("New York" / "San Francisco"; person names with first and last name). Score: is the block a valid entity or a cross-mode hybrid?
- **Paired structure:** matched brackets or quotes, `key: value` formats, date or number formats where tokens must agree (e.g., `12/31` vs `31.12`).
- **Coupled arbitrary labels:** in-context mappings where the answer is a two-token label drawn from a set whose tokens recombine into invalid labels (adapting the arbitrary-label design of [3]).

For each task, generate ground truth programmatically, balance modes and positions, check tokenization, hold out templates, and drop configurations where the model is near chance or always picks one mode. Start at 0.5B and move to 7B if competence is lacking.

Primary metric: the **mode-mixing rate** under full / none / same-only / cross-only.

### Experiment 3: Commitment-Propagation Interventions

On examples with two valid modes A and B:

1. Run a trajectory that ends in mode A. At step k, replace the feedback at source position i with its value from a matched trajectory that ends in mode B. Hold x_k and all other feedback fixed.
2. Measure whether target positions j ≠ i flip toward B, using the margin m_j = log p(y_j^B) − log p(y_j^A), and whether the final block is a consistent B, a hybrid, or stays A.
3. Sweep k (early, middle, late) to find when commitment becomes locked in.
4. Controls: same-position swaps (i = j); norm- and entropy-matched non-mode donors; random-direction perturbations of matched magnitude; swapping x_{k,i} instead of q_{k-1,i}.

Because the donors are realistic predictions from real trajectories, the intervention stays close to the model's natural input distribution. A real mode-commitment mechanism predicts that cross-position swaps flip partners to a *consistent* B, rather than just adding noise.

### Experiment 4: Localization (only if Experiments 1–3 are positive)

Using the clean (A) and swapped (B-donor) runs from Experiment 3:

- Patch attention-head outputs and the residual stream at target positions to find the components that carry the flip. Use path patching restricted to attention edges from source i to target j.
- Test necessity (ablating the candidate heads blocks the flip and raises the natural mode-mixing rate) and sufficiency (restoring them in the swapped run restores the flip).
- Compare against random head sets with the same intervention budget. Apply multiple-comparison control across heads.
- Discover components on one split; validate on held-out templates.

Report the recovery ratio R_P = (m_patched − m_corrupted)/(m_clean − m_corrupted) only when the denominator is large, and do not read it as a mediation percentage.

### Experiment 5: Replication

Repeat Experiments 1–3 (and Experiment 4 if it is positive) on Lafa-7B (Qwen2.5-Instruct base) and Lafa-8B (Qwen3 base). These are different backbones, so compare functional roles, not head indices.

---

## 6. Alternative Explanations to Rule Out

| Alternative | How it is addressed |
|---|---|
| Benefit is same-position stabilization | Same-only condition (Exp 1) |
| Cross-position sensitivity is just bidirectional attention | Exp 3 requires *consistent* flips, not just sensitivity; matched non-mode donors |
| Information is already in x_k | Swapping x_{k,i} vs q_{k-1,i} (Exp 3); §2.2 framing |
| Norm or entropy artifacts | Norm- and entropy-matched controls |
| Inference-time "off" is out of distribution | Use the training-time drop condition (§0); sensitivity to alternatives |
| Few-step solver mismatch [6] | Robustness with untied self-conditioning |
| Template or position shortcuts | Held-out templates; balanced modes and positions |

Statistics: paired comparisons on identical prompts and seeds; bootstrap CIs over examples; held-out splits for every discovered component.

---

## 7. Pilot and Timeline

| Week | Work | Gate |
|---|---|---|
| 1 | Resolve §0; reproduce Lafa-0.5B generation; implement the feedback hook and same-only / cross-only passes | Can reproduce reported numbers to within noise |
| 2 | Exp 1 on GSM8K/HumanEval subsets; build Exp 2 tasks; check competence | **Go/no-go on H1** |
| 3–4 | Exp 2 mode-mixing rates; Exp 3 propagation sweeps | Consistent cross-position flips beyond controls |
| 5–7 | Exp 4 localization; Exp 5 replication at 7B/8B | Held-out validation passes |
| 8–10 | Writing | — |

Compute: frozen-model inference only. Same-only and cross-only cost about 4× forward passes per step at B = 4, which is cheap at 0.5B. Store activations selectively. Benchmark memory and throughput on A100s before scaling to 7B/8B. No training required.

---

## 8. Risks

- **Self-conditioning is mostly same-position or compressible** (cf. [10]). Exp 1 detects this in about 2 weeks. The decomposition is still reportable as an analysis result.
- **Little mode mixing at B = 4.** The prefix may already determine most blocks. Mitigation: choose prompts with real ambiguity; measure the natural mode-mixing rate under "none" first.
- **No compact circuit.** H4 may fail even if H1–H3 hold. The paper then rests on the behavioural and causal decomposition, which is weaker but still a contribution.
- **Competition and timeline.** Lafa is new and openly released, and other groups will analyze it. Prioritize Exp 1 and Exp 3 results over completeness.

---

## 9. What a Strong Paper Looks Like

1. A clean decomposition: X% of self-conditioning's benefit requires cross-position feedback (Exp 1).
2. Evidence that this benefit is mode commitment: it lowers mode mixing, and swapping one position's feedback flips its partners consistently (Exp 2–3).
3. A characterization of when commitment locks in during denoising.
4. (Bonus) A small, validated set of components that carries it, replicated across backbones.

A collection of attention maps or nonzero source→target effects alone is not enough.

---

## 10. References

[1] Chavan, A., Lee, C., Bamba, U., Yoo, J., Hong, S., Raghunathan, A., & Kim, J. (2026). *Large Flow Language Models.* arXiv:2610.28559. (Read in full, Phase 2.)

[2] Lafa Authors (2026). *Large Flow Language Models — Official Implementation.* https://github.com/lafa-gh/large-flow-lms (Apache-2.0). Checkpoints: https://huggingface.co/lafa-hf/lafa-0.5B, -7B, -8B.

[3] Gao, Andrew, & Gao, William (2026). *A Circuit for In-Context Learning.* arXiv:2611.05344.

[4] Kong, I., Lee, H., & Jo, Y. (2026). *Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models.* EMNLP 2026 (main), pp. 24615–24643. arXiv:2601.14758.

[5] Davis, O., Roos, D., Bizeul, A., Shidani, A., Turrisi, V., Béthune, L., Ablin, P., & Cuturi, M. (2026). *On the Design Space of Continuous Diffusion Language Models.* arXiv:2610.13834. (Self-conditioning addresses the train–inference mismatch; token identifiability is handled by discrete reveals, not self-conditioning.)

[6] Li, B., & Xu, L. (2026). *Improving Few-Step Language Flows with Untied Self-Conditioning.* arXiv:2608.22244.

[7] Zhang, S., Chen, Z., Qiu, H., He, H., Du, L., & Lan, Z. (2026). *A Dominant Self-Conditioning Direction Drives Repetition in Unconditional Continuous Diffusion Language Models.* arXiv:2607.00588 (v2; v1 was titled "Low Perplexity is Repetition…").

[8] Zhang, Z., Gao, L., Gu, M., Chen, X., Luo, M., Lyu, K., & He, T. (2026). *Self-Conditioning Enables Deep Sequential Reasoning in Flow Matching.* arXiv:2610.19628.

[9] Wang, X., Jiang, B., Wan, Y., Yang, B., Kong, L., & Zou, D. (2026). *DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders.* ICML 2026, PMLR 306. arXiv:2602.05859.

[10] Yoo, J., Kim, W., Eijkelboom, F., Lee, C., Boffi, N. M., Hong, S., & Kim, J. (2026). *Self-conditioned Flow Map Language Models via Fixed-point Flows.* arXiv:2607.00714. ("Without degradation" holds at 32 steps; the 8-step student only approaches the teacher.)

Additional key references from Phase 3 (full ledger in `phase3.md` §6):

[11] Lee, C., et al. (2026). *Flow Map Language Models.* arXiv:2602.16813. (Same "new-york / san-diego" hybrid toy.)

[12] Shabalin, et al. (2026). *Why Gaussian Diffusion Models Fail on Discrete Data and How to Prevent It.* arXiv:2604.02028. (The competing density/multi-modality account of self-conditioning → rival M5.)

[13] Yao, L. (2026). *Don't Commit Alone* (CoCommit). arXiv:2607.04469.

[14] Biroli, G., Bonnaire, T., de Bortoli, V., & Mézard, M. (2024). *Dynamical regimes of diffusion models.* Nat. Commun.; arXiv:2402.18491.

[15] Sclocchi, A., Favero, A., Levi, N., & Wyart, M. (2025). ICLR 2025, arXiv:2410.13770; and Sclocchi, Favero & Wyart (2025), PNAS, arXiv:2402.16991.
