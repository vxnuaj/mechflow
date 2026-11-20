# Generalization search (2026-11-19, Claude; compare with the GPT runs)

## Sharpened explanation ("decision-locus" account)
SC helps coherence only by carrying neighbours' tentative commitments into the computation that fixes token j. The route that wins is whichever channel is the best available carrier of neighbour commitment *at the steps where j's token is actually decided*:
- **N (direct read of neighbours' feedback)** wins when that feedback is visible there and neighbours' states are not yet informative (Lafa).
- **F (neighbours' feedback-improved states)** wins when the deciding computation has no feedback, or states are already informative (ELF: decided at the SC-zeroed decode pass).

O ≈ 0 everywhere: own feedback carries nothing about neighbours' current commitments.

**Implication:** block vs full sequence, feedback form and scale should matter only through *when* the decision happens and *what each channel carries then*.

## Most discriminating released models (facts marked V = verified)

| Model | Weights | SC form | SC at readout | Structure | Why it discriminates |
|---|---|---|---|---|---|
| **Plaid 1B** (2305.18619) | V: GitHub release, 3 × 1.6 GB | V: softmax(logits) @ embedding (expected embedding, as in Lafa), zero-init linear | V: yes (final call uses x_selfcond, then argmax) | full sequence, continuous | Same feedback form and readout as Lafa, but full-sequence and a different group. Our account predicts **N-dominant**. "Block locality" predicts N small. |
| DiffusionGemma 26B-A4B (2608.00146) | V: `google/diffusiongemma-26B-A4B-it` (MoE, 3.8B active) | soft embedding of logits | yes in canvas; zeroed at commit pass | 256-token blocks, uniform-state discrete | A discrete state channel. Expensive (~10 A100-h), with an adaptive-acceptance confound. |
| TESS-2 (2502.13917) | V: `hamishivi/tess2` 7B | V: "logits_mean", fused into the state | yes | full sequence, simplex | Tests fused injection |
| Nemotron-Labs-Sigma 3B/8B (2610.02665) | V: nvidia collection | V: expected embedding | yes | 32-token blocks, continuous | Near-replication of Lafa's design; low discrimination, adds scale |

**Excluded:**
- No SC: SSD-LM, Diffusion-LM, DiffuSeq, Genie.
- No weights found: CDCD, SED, SCMDM.

## Proposed programme (ordered by information per GPU-hour)
1. **C1, a Lafa SC-off readout pass (~2 A40-h).**
   - Add a final readout pass with SC zeroed. This is in-distribution, since SC is dropped 50% of the time in training.
   - **Prediction:** the N share stays within 0.15 of baseline, because Lafa's decision is made before readout.
   - "SC at readout decides the route" predicts that N collapses.
2. **Plaid 1B 2×2×2 (~4 A40-h at K = 256).**
   - **Prediction:** N ≥ 0.6 and O ≤ 0.1.
   - N ≤ 0.3 would refute our account in favour of block locality.
3. **ELF with self-only decode attention**, briefly fine-tuned (~6 A40-h).
   - This removes the decode-pass route, so **the prediction is that N must rise**.
   - "T5 latents make feedback redundant" predicts N stays ~0.18.

Also proposed: a probe-based quantitative prediction of the N share (decision profile × information in each channel), computed before each run. It is more ambitious; use only if it retrodicts Lafa and ELF first.

## Rival explanations to rule out
- SC at readout;
- contextual latent space (T5);
- block locality;
- state SNR alone (already weakened);
- injection method;
- capacity.

## Merge with GPT review (2026-11-19)
- **Convergence:** both independently propose the same account ("commitment propagation through the channel accessible where the decision is made"). Both rank within-model access manipulations above more model replications.
- **Adopted from GPT:**
  - matched toy factorial (readout SC access × feedback form);
  - absolute effects, never shares alone (we made exactly this error in E3);
  - commitment horizon from a separate calibration split;
  - sham swaps;
  - a feasibility category ("no identifiable mechanism") distinct from refutation.
- **GPT claims corrected by code inspection:**
  - LangFlow's feedback is an **expected token embedding** (softmax → `_embed_tokens`), not a continuous latent prediction. It is injected ELF-style (concat + zero-init projection), and SC **is** passed to the final readout call.
  - Plaid's readout SC is verified present (no audit needed).
- **Priority disagreement:** GPT ranks DiffusionGemma first among external models, but schedules it last in its own programme. We keep it last (cost, MoE, adaptive-acceptance confound). LangFlow replaces Plaid as the first external test (130M, HF, the same predictions).
- **Where the two theories diverge (tested by C1):** "readout access determines the route" vs "the decision is made at late noisy steps before readout". Removing SC only at Lafa's final readout should collapse N under the first and barely change it under the second.
