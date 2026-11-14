# mechflow — Phase 2: Ground Truth on Lafa

Date: 2026-11-13. Sources:
- the paper, arXiv:2610.28559v1 (read in full: main text, Appendix B, Table 4, Table 5);
- the official repo `lafa-gh/large-flow-lms` (shallow clone of 2026-11-13; read only, never executed).

Code references are relative to the repo root, and **FM** means `src/lafa/flow_model.py`. Items marked ✔︎ were checked directly in code by me. All other code claims come from a careful code read by a subagent with file:line citations.

---

## 1. Answers to the Phase 2 Checklist

### 1.1 Paper

| Question | Answer | Source |
|---|---|---|
| Self-conditioning ablation numbers | **No self-conditioning 32.3 → self-conditioning 38.1** (avg. IFEval / GSM8K / HumanEval), 0.5B, B = 4, 32 Euler steps. GPT's +5.8 pp was **right**; the secondary-source "~4 pp" was wrong. Prefix guidance: 34.3 → 38.1. | Table 3 |
| Train-time or inference-time ablation? | **Inference-time.** Table 3 lists it under "Inference": the same trained model, with self-conditioning switched off at decode time. | Table 3, §3.4 |
| Self-conditioning training recipe | Stop-gradient pass on the **same** noisy I_t (same t). Its prediction x̄₁ = sg(D_t(I_t)E) feeds Ĩ_t = I_t + I_t W₁ᵀ + x̄₁W₂ᵀ, and the loss is on the second pass. Self-conditioning is used with p_sc = 0.5; when dropped, the input is the **plain I_t**. W₁, W₂ ∈ ℝ^{d×d} are zero-initialized. | App. B.2 eq. 24; ✔︎ FM:868–910 |
| Sampler | Euler in x-prediction form on a **learned, per-position warped time grid** (asynchronous: earlier positions are denoised ahead of later ones). The q from step k−1 feeds step k. The first q comes from an extra bootstrap pass on the raw noise at t = 0. | eq. 8, 25; Alg. 3; ✔︎ FM:1060–1090 |
| Readout | **Per-position argmax** of the conditional logits from the last forward pass (at t_{K−1} < 1; the loop breaks before the final Euler update). No remasking, no AR fallback. | eq. 8; FM:1229, 1720 |
| Guidance | "Prefix guidance" = CFG with a **prefix-free** pass (all prefix keys masked): ê = ê_u + g(ê_c − ê_u). It doubles the forward passes. **The conditional and unconditional branches each keep their own feedback state.** The feedback for the next step is the *unguided* conditional prediction. | eq. 27; ✔︎ FM:1196–1227 |
| Settings behind the reported numbers | Table 3 ablations: 0.5B, B = 4, **32 steps**, and (since the full-config rows are equal) **g = 8**. Table 4: inference noise scale **0.1** (vs. **4.0** in training); g = 8.0. | Tables 3–5; docs/evaluation.md |

### 1.2 Code

| Question | Answer | Source |
|---|---|---|
| "Off" input used in training | **Plain x_t**: neither W₁ nor W₂ is applied (`return e_t`). Setting q = 0 instead gives x + xW₁ᵀ, an input **never seen in training**. | ✔︎ FM:877–881 |
| Drop granularity | **Per global optimizer step**: one draw for the entire batch, all ranks, all positions (`manual_seed(0x5C0FF + step)`). Every training input is either "self-conditioning at all positions" or "self-conditioning at none". | ✔︎ FM:861–866 |
| q fed at the first step | Bootstrap pass on the raw noise at time_grid[0] (= 0), with no self-conditioning; softmax @ E. With CFG, a separate bootstrap is run for the unconditional branch. | ✔︎ FM:1067–1090 |
| Block boundaries | Training: **fixed absolute positions** (multiples of B). Inference: blocks **start at the prompt end** (`block_start_pos = len(gen_ids)`). So we choose alignment by choosing the prompt. | FM:646, 1661; attention.py |
| Hook points | Self-conditioning: `_apply_self_cond(z_fwd, e_self_cond)` (✔︎ FM:1185–1189 and the batched path). Attention: stock HF Qwen2/Qwen3 attention, **forced to eager** (no flash-attn, no SDPA, no torch.compile, no packing at inference). Decoder layers are patched by `DiTAdaLN.patch_layer` (layers.py:105–182), a natural wrap point. | modeling_lafa.py:34–44 |
| Other paths for q? | **None.** q enters only at the input (resolves A6). | FM review |
| Determinism | Deterministic given the seed. The only randomness is the initial noise (SDE options are off by default). The batched eval path **shares initial noise across rows**. | FM:1619, 2130 |
| Defaults | HF `generate` defaults: `num_steps=4, cfg_scale=1.0, noise_scale_override=0.1`. Paper eval commands: `num_steps=K, cfg_scale=8, noise_scale_override=0.1`. **The defaults ≠ the paper's settings; always pass them explicitly.** | modeling_lafa.py:157; docs/evaluation.md |
| AR decoding of the same checkpoint | `model.generate(..., gen_mode="ar")` → `generate_ar` (clean stream at t = 1). | FM:1377 |
| Checkpoints | `lafa-hf/lafa-0.5B` (2.9 GB repo), `-7B` (30.9 GB), `-8B` (33.3 GB), all B = 4 and all **Apache-2.0**. HF weights are EMA; `subfolder="raw"` gives non-EMA weights. 0.5B uses per-layer adaLN; 7B/8B share it across layers. | HF API; configs |

---

## 2. Status of the Phase 1 Assumptions

| ID | Assumption | Status | Consequence |
|---|---|---|---|
| **A1** | An in-distribution "off" exists. | ✅ **Holds at block level**: plain x at *all* positions, matching the p_sc = 0.5 drop. | The "none" condition and the full-vs-none gap are in-distribution, and the paper's 32.3 vs. 38.1 is exactly this contrast. |
| **A7** | Mixed per-position "off" is tolerable. | ⚠️ **Not supported by training.** Drops are whole-batch, so a block with some positions on and some off was **never seen**. Drift must be measured. | **K3 is now the critical instrument check.** Candidate "off" values for a single position, to compare in Phase 6 by drift on unimodal blocks: **(i)** plain x_i (training's drop value, mixed context); **(ii)** a stale q_i from the bootstrap pass (a real feedback value, uninformative about the current step); **(iii)** q_i from the prefix-free branch (a real feedback value, carrying no prefix information). Option (ii) or (iii) may be *more* in-distribution than (i), because the input still has the trained form x + xW₁ᵀ + qEW₂ᵀ. |
| **A2** | Deterministic given the seed. | ✅ Holds. | Paired designs are exact. |
| **A8** | The initial noise picks the mode ("propagation, not origination"). | ⚠️ **Doubtful.** Sampling starts at **0.1·σ** (training uses 4·σ), so x₀ ≈ 0 and seeds give little diversity. The mode is chosen mostly by the prefix plus deterministic dynamics. | (a) The **Phase 1 scope needs revisiting** (§3). (b) **C3 donors can't come from re-seeding** (§3). |
| **A3** | Spans can be aligned within a block; prefix teacher-forced. | ✅ Holds. Inference blocks start at the prompt end, and the prompt is prefilled into the KV cache. | Choose prompt length and phrasing to place coupled spans in one block. |
| **A6** | Feedback enters only at the input. | ✅ Holds. | No extra paths to cut. |
| **A4, A5** | AR reference quality; enough coupled blocks at 0.5B. | ⏳ Phase 7. | — |

### Kill criterion K1 (untestable)

**Not triggered.**
- An in-distribution block-level "off" exists.
- Per-query K/V selection looks implementable: eager attention, dense additive masks, and a `DynamicCache` that already supports append-and-trim. The recipe is:
  1. append the off-pass K/V for the block (same `block_pos_ids`, so RoPE is already applied) after the prefix;
  2. run the live block with a dense mask so that query i sees the prefix, the off-pass keys j ∉ {i}, and its own live key;
  3. trim the cache.
- Under CFG, the same surgery must be applied to the unconditional pass.
- **Phase 6 must confirm faithfulness with the identity test:** "alt = same pass" must reproduce the clean outputs bit-for-bit or within tolerance.

---

## 3. Facts That Change the Design (input to Phase 4)

These are recorded here, and the Phase 1 claims will be revised in Phase 4 rather than silently now.

1. **Asynchronous tokenwise time: commitment is left-to-right by construction.**
   - Earlier block positions are denoised ahead of later ones (learned τ^i(t) > τ^j(t) for i < j; paper Fig. 5).
   - So **lead–lag is partly built into the schedule**. "Position 1 commits first, and later positions follow" is expected even under R0, because x₁ itself becomes nearly clean early.
   - *Implication:*
     - the direct-effect matrix is expected to be **asymmetric** (earlier → later);
     - the test of C1 and C3 has to beat the skeptic **at matched schedule position**, i.e. compare the direct effect of q_i against what x_i alone already carries at that step;
     - lead–lag by itself proves nothing and is dropped as evidence for propagation.

2. **Near-zero initial noise (0.1·σ): decoding is close to deterministic given the prefix.**
   - The "origination vs. propagation" framing assumed that the noise picks the mode. With x₀ ≈ 0, symmetry is broken by the prefix and the dynamics, **possibly including self-conditioning itself**. This could make the feedback *more* central than assumed, not less.
   - *Implications:*
     - **(a)** Revisit the scope: origination may be in scope after all.
     - **(b)** C3 donors must come from **minimal-pair prompts** that steer toward mode B (as the original proposal suggested), or from a **training-matched noise condition** (noise_scale = 4.0) used as a secondary regime. They can't come from re-seeding at 0.1.
     - **(c)** "Several seeds per prompt" buys little at 0.1, so variance comes mainly from prompts. Re-do the power analysis accordingly.

3. **The documented self-conditioning benefit is at g = 8 and 32 steps.**
   - Guidance is a strong extrapolation (ê_u + 8(ê_c − ê_u)), and the unconditional branch has its **own** feedback state.
   - *Decision for Phase 4:*
     - **primary regime = the paper's** (g = 8, 32 steps, noise 0.1), because that is where the 5.8 pp effect is documented;
     - every intervention is applied **consistently to both branches**;
     - **g = 1** is a secondary regime, and its own full-vs-none gap is measured in the Phase 7 pilot.
   - The proposal's earlier "guidance 1.0 to avoid confounding" is **superseded**: at g = 1 the benefit we want to explain may not even exist.

4. **The readout is a per-position argmax at t_{K−1} < 1.**
   - Hybrids can arise mechanically from independent argmaxes when the block hasn't fully converged.
   - The paper itself states that one-step drafts "can be inconsistent across positions" (§2.3), which is direct evidence that **mode mixing exists at few steps** (relevant to K4).
   - At 32 steps it is unknown. The hybrid metric must use exactly this readout.

5. **Mixed on/off feedback is out of distribution (A7).**
   - Primary C1 knockout validity now rests on Phase 6's drift test across the three candidate "off" values (§2).
   - If all three drift badly, use swap-only designs (outcome lattice, N/A rows).

6. **Training/inference mismatches that are part of the model's normal behaviour (not ours to fix):**
   - noise scale 4.0 vs. 0.1;
   - absolute vs. prompt-end block alignment;
   - same-t feedback in training vs. one-step-stale feedback at inference;
   - bootstrap q from t = 0.
   
   These are noted only because "in distribution" for our interventions means **matching the inference distribution** (one-step-stale feedback), not the training one.

7. **The 0.5B model is weak** (B = 4, 32 steps: GSM8K 30.4, HumanEval 31.1, IFEval 52.9). Constructed tasks must be easy, and a move to 7B (30.9 GB; per-step cost about 15×) is likely for the primary results (A5).

8. **Novelty pointer for Phase 3.** The Lafa paper itself cites prior work that "learn[s] deterministic noise-to-token maps that capture joint dependencies without independent output sampling" ([47, 57] in the paper), and notes that independent sampling limits few-step quality ([62, 63]). **Read these first in Phase 3. They are the closest prior work to H2.**

---

## 4. Gate Outcome

- **K1: not triggered.** The project is testable.
- **No experiment is invalidated outright**, but four parts of `phase1.md` must be revised in Phase 4:
  - scope (§1.3, origination);
  - C3 donor source;
  - the primary regime (g = 8, both branches);
  - the lead–lag evidence (no longer counts).
- **New top risk:** A7 / K3. Every knockout result depends on finding a single-position "off" that doesn't drift. **Phase 6's first job.**
- **Phase 7 additions:**
  - the full-vs-none gap at g = 1 vs. g = 8;
  - the hybrid rate at 1/4/8/32 steps;
  - seed diversity at noise 0.1 vs. 4.0.
