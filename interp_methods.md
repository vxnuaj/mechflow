# Mechanistic-Interpretability Methods for mechflow (literature review, 2026-11-13)

**Goal of this note:** decide which mech-interp methods to add so the project
- (a) separates the three channel hypotheses more rigorously:
  - **M2**: neighbours read each other's fed-back prediction q;
  - **R0**: neighbours read each other's state x;
  - **M1**: each position stabilizes itself;
- (b) **localizes** the components (heads/MLPs) that implement cross-position reading. That is what makes it a mech-interp paper rather than a causal-analysis paper.

**Sources:** two parallel literature reviews (causal / circuit-discovery methods; diffusion-and-flow interpretability + representation tools). Citations are arXiv IDs. Those marked † were cited from memory by the reviewer and should be spot-checked before they go into the paper.

---

## 1. What we already have (and what it is called in the literature)

| Our tool (`phase4.md`) | Standard name | Status |
|---|---|---|
| Single-step feedback knockout with x fixed | Activation patching / causal mediation, input-node ablation | Built, validated |
| Per-query K/V selection (j reads i's K/V from another pass) | **Path patching** at block granularity (Wang et al. 2211.00593†; Goldowsky-Dill et al. 2304.05969†) | Built, validated |
| LEACE erasure of "which answer" from x | Concept erasure (LEACE 2306.03819†); amnesic probing (2006.00995†) | Specified |
| O × N × F factorial with Möbius/Shapley shares | Causal mediation: natural direct/indirect effects (Vig et al. 2004.12265†), plus interaction terms | Specified |
| Minimal-pair donor swaps of q | Interchange interventions (Geiger et al. 2112.00826†) | Specified |
| Positive control, OOD (Mahalanobis) check, null calibration, bf16 rule | Best-practice validation (Zhang & Nanda 2309.16042; Heimersheim & Nanda 2404.15255†) | Done |

**Gap:** everything is at **channel** level (feedback vs. state vs. own). Nothing is yet at **component** level (which heads read the neighbours' feedback).

---

## 2. Key lessons from the literature (they change how we do things)

1. **Holding x fixed separates M2 from R0 only within one step.** Over a trajectory, x already integrates past feedback. Our 2 × 2 × 2 (N vs. F) is the right fix; single-step results must never be presented as trajectory-level claims.
2. **Attention knockout removes x and q together**, so it can't separate M2 from R0 (Geva et al. 2304.14767†; Ferrando & Voita 2403.00824†). Use it only for coarse necessity. Separation needs **q-path vs. x-path** patches.
3. **Measure attention *outputs* / OV contributions, not attention weights** (ConceptAttention 2502.04320). "Head h attends to i" is not evidence of reading q.
4. **Learned-subspace patching can create illusions** (Makelov et al. 2311.17030†). DAS-style results need null-space controls and held-out pairs.
5. **Faithfulness depends on the ablation type** (Miller et al. 2407.08734; Hanna et al. 2403.17806†). Report zero / mean / resample ablations, plus random and size-matched baselines.
6. **Decodable ≠ used** (Hewitt & Liang control tasks 1909.03368†). Every probe result is paired with an intervention. Re-fit probes and erasers **per denoising step bucket**, because the noise level changes the covariance.
7. **Circuits change across denoising steps** (DifFRACT 2606.15796, image diffusion; DLM-Scope 2602.05859). Report results per step band.
8. **CFG amplifies small patch differences** (scale 8 multiplies cond − uncond). Run localization with CFG off as well. (Our mechanism regime is already g = 1.)
9. **bf16 noise can be the size of the effect** → fp32 for gradient-based screening, fixed seeds, and a reported noise floor. Consistent with our Phase 6 rule.

---

## 3. Recommended additions

### Tier 1 — cheap; add to the Phase 7 exploration pilot (no pre-registration change needed; exploratory)

| # | Method | Concrete experiment | Cost (0.5B) |
|---|---|---|---|
| T1a | **Information-flow routes** (Ferrando & Voita 2403.00824†) **+ AtP\*** screening (Kramár et al. 2403.00745†) | Head × step heatmap of j ← i reads on q-donor minimal pairs; pick the top ~2% of the 43k head sites and the step bands where coordination happens. AtP* in fp32; validate the ranking with real patches on a random subset. | Routes: 1 forward/step. AtP*: ~3 passes/step/pair |
| T1b | **Cross-position Jacobian / coupling spectrum** (dynamical-systems view; Sussillo & Barak 2013†) | ∂q^i_{k+1}/∂q^j_k vs. ∂q^i_{k+1}/∂x^j_k for all (i ≠ j) and steps k. M2 predicts off-diagonal q-coupling peaking near commitment; M1 predicts block-diagonal dominance. Compare the peak with commitment time. | Autograd, small |
| T1c | **Interpretable-bottleneck test** (after "How Transparent is DiffusionGemma?" 2606.20560) | Replace each fed-back q with its top-1 token embedding (or top-k). If agreement and coherence survive, the feedback carries **token identity**; if not, it carries "soft" uncertainty. | One extra trajectory per block |
| T1d | **Contrastive "which answer" direction in q-space** (ACE recipe from 2607.00588) **+ dose–response steering** (ActAdd 2308.10248†, CAA 2312.06681†) | Estimate the A-vs-B direction in q; add it **only to neighbours' q** → does position j flip coherently? (Sufficiency.) Erase it from q-only vs. x-only. (Necessity per channel.) Report norm-matched dose–response. | Cheap |

### Tier 2 — localization; do at Phase 8 **if** C1′ or H2-N survives (on the circuit-validation split; prereg §2.3)

| # | Method | Concrete experiment |
|---|---|---|
| T2a | **Head-level path patching, q-path vs. x-path** | Sender: the q-term of position i. Receivers: K/V of head (l, h) at i, as read only by query j. Everything else frozen, including i's reads of j. Same patch from the x-path. **Output: a per-head M2-vs-R0 score → a named set of "feedback-reader" heads.** Second stage: which heads/MLPs at j consume them. |
| T2b | **EAP-IG, position-resolved (PEAP)** (Syed et al. 2310.10348†; Hanna et al. 2403.17806†; Haklay et al. 2502.04577) | Minimal circuit with q^i as an explicit input node; positions aligned by token **role** (answer-token-1, …), not index. Report **MIB** CPR/CMD (Mueller et al. 2504.13151) under zero / mean / resample ablation, with random and size-matched baselines. |
| T2c | **Mediation (NIE through the identified heads)**, mapped onto our 2 × 2 × 2 | Show that the indirect effect through the feedback-reader heads accounts for the N share. |
| T2d | **DAS / interchange-intervention accuracy** (Geiger et al. 2303.02536†; 2301.04709†) | Encode M2, R0 and M1 as high-level causal models; train DAS at the identified heads' K/V inputs; compare IIA on held-out minimal pairs. Include null-space / illusion controls. |
| T2e | **Causal scrubbing** (Chan et al. 2022, Alignment Forum) as the final validation | Fraction of loss recovered under the M2 vs. R0 vs. M1 resampling rules (resample within matched noise levels). **Behavioural prediction:** knocking out the circuit raises the hybrid rate; patching it in lowers it. |

**Cost of T2a:** ≈ 8k block forwards per step for all heads × ordered pairs (≈ 260k over 32 steps). That is feasible batched on CARC, but triage with T1a first.

### Tier 3 — optional; potential methodological contributions (nothing published, per both reviews)

- **Path patching through a recurrent feedback edge:** treat q_{k−1} as a cross-step edge and path-patch "step-(k−1) heads → q → step-k reads", with x clamped. Existing path patching assumes a feed-forward DAG. **Both reviews flagged this as publishable novelty**, and our per-query K/V machinery already does most of it.
- **One-checkpoint mode diffing:** an AR-mode vs. flow-mode **crosscoder** on the same weights (BatchTopK; latent-scaling checks from 2504.02922). Flow-exclusive latents at block positions are candidate "agreement features". This removes the different-checkpoint confound in Kong et al. (2601.14758).
- **Step-crosscoder:** one latent dictionary shared across denoising steps (one decoder per step bucket), to follow an answer feature from first appearance to commitment.
- **Step × layer tuned lens** (Belrose et al. 2303.08112†): a commitment map, checked against Prophet-style top-2 gaps (2508.19982) and our cloning measure.
- **Attribution graphs unrolled through self-conditioning** (Transformer Circuits 2025; DifFRACT 2606.15796): stitch per-step transcoder graphs via q.

---

## 4. Controls reviewers will expect (add to the analysis plan)

- A noise floor (repeated identical runs) and CIs over prompts and seeds, across multiple prompt families.
- IOI-style **faithfulness / completeness / minimality**, with random and size-matched circuit baselines.
- Robustness to the **ablation type**, to **single-step vs. trajectory** metrics, and to **CFG on vs. off**.
- **Base-model comparison:** do the identified heads exist in base Qwen2.5-0.5B (before adaptation) as general "read the neighbour token" heads, with W₂ repurposing them? Or are they created by flow training?
- An explicit argument for M2 vs. R0 **across the trajectory**, not only within one step (2 × 2 × 2).
- At least one **confirmed out-of-sample behavioural prediction** (e.g. circuit knockout raises the hybrid rate).

---

## 5. Relevant related work surfaced by the reviews (add to `phase3.md` before writing)

- **"How Transparent is DiffusionGemma?"** (Engels, …, Nanda; 2606.20560): reports "token smearing" across positions, an analogue of our hybrids. **Must cite and differentiate** (masked diffusion vs. continuous flow; we test the self-conditioning channel).
- **How DiffusionGemma commits tokens** (2606.14620): a methodological checklist for commitment metrics.
- **CANDI** (2510.22510): reads self-conditioning as pseudo-clean anchors for conditional dependencies, which leans toward M2. Cite.
- **DifFRACT** (2606.15796): timestep-conditioned transcoders + circuit tracing (image diffusion).
- **PEAP** (2502.04577), **MIB** (2504.13151), **Miller et al.** (2407.08734): circuit-discovery standards.

---

## 6. Recommendation
- **Now (Phase 7 pilot):** add T1a–T1d. They are cheap, exploratory, and tell us at Phase 8 whether there is anything to localize, and where (which steps and layers).
- **Phase 8 decision:** if C1′ or H2-N looks real, commit to Tier 2 (T2a → T2b → T2c, with T2d/T2e as validation) on the circuit-validation split. That is what turns the paper into a mech-interp paper.
- **Tier 3:** "path patching through the recurrent feedback edge" is the strongest candidate for a methods contribution, because our infrastructure already supports it.
- **Pre-registration:** Tier 1 is exploratory (allowed). If Tier 2 localization is to be **confirmatory**, its estimands and criteria must be added to `prereg.md` §12 (and pushed) **before** any confirmation data is generated.
