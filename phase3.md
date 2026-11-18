# mechflow — Phase 3: Literature Review and Novelty

Date: 2026-11-13. Four parallel reviews (reference verification; M2; M3 + self-conditioning theory; M4 + commitment theory + mech interp), synthesized here. Each entry is marked **[full]** when the full text (or most of it) was read, **[abs]** when only the abstract was read, or **[sec]** for secondary sources only. Where reviewers disagreed, the source with a verified URL wins.

**Headline:** each of M1–M4 is **partly known**, but the specific gap is **open**:
- **No one has intervened causally on the self-conditioning channel with timestep and position resolution**, in a prefix-conditioned or blockwise flow LM.
- **No one has decomposed the self-conditioning gain** into competing mechanisms.

Several authoritative sources state that why self-conditioning works for language is "still unclear" or "poorly understood": Dieleman's Aug 2026 blog, LangFlow, Yoo et al., and "How to Guide Your Language Flow". The decomposition framing is therefore well-motivated, and M2 survives as the headline candidate with a precise differentiation (§3).

---

## 1. Reference Corrections (applied to `mechflow.md` §10)

| Ref | Status | Fix |
|---|---|---|
| [3] Gao & Gao, 2611.05344 | ✅ exists; authors **Andrew** Gao, **William** Gao | Studies 3–4B models (0.6B–8B scaling); key–value memory over label positions. |
| [4] Kong, Lee & Jo | ✅ EMNLP 2026 main (verified on the ACL Anthology); arXiv **2601.14758** | Add the arXiv ID. |
| [5] Davis et al., 2610.13834 | ✅ exists; **description was wrong** | Self-conditioning addresses the train–inference mismatch. Token identifiability is handled by *discrete reveals*, not by self-conditioning. |
| [6] Li & Xu, 2608.22244 | ✅ | — |
| [7] 2607.00588 | ✅; **retitled in v2** | v2: "A Dominant Self-Conditioning Direction Drives Repetition in Unconditional Continuous Diffusion Language Models." Authors: Shuai Zhang, Zijie Chen, Huachuan Qiu (added in v2), Hongliang He, Lun Du, Zhenzhong Lan. |
| [8] Zhang et al., 2610.19628 | ✅ | — |
| [9] DLM-Scope | ✅ ICML 2026; arXiv **2602.05859** | Add the arXiv ID. |
| [10] Yoo et al., 2607.00714 | ✅; authors Yoo, Kim, Eijkelboom, Lee, Boffi, Hong, Kim | "Without degradation" holds only at **32 steps**; the 8-step student only *approaches* the teacher. |

---

## 2. Per-Mechanism Verdicts

### M1 — Per-position stabilization (the skeptic)
**Status: the best-supported explanation in the literature.**
- Sources: Analog Bits, CDCD (reuse of the previous estimate); LangFlow (prevents drift to high-frequency tokens at high noise); Yoo (contractive fixed-point refinement); Davis (train–inference mismatch); Li & Xu (feedback redundant with the latent); 2607.00588 (contractive attractor).
- **Implication:** if M1 wins in Lafa, the result is **least novel**: a careful negative result on M2, not an ICML headline.

### M2 — Cross-position coordination (our claim)
**Status: the phenomenon is known; the mechanism is open.**

**Known:**
- the hybrid failure itself: FMLM 2602.16813 Fig. 3 uses the *same* "new-york / san-diego" toy; ParallelBench 2510.04767; CoCommit 2607.04469; DCD 2410.01949; CoDD 2603.00045;
- that continuous flows avoid it in principle, through deterministic joint transport (FMLM; Posterior Refinement 2606.24773; Dieleman's blog);
- that self-conditioning improves coherence (Yoo App. D.1; Loopholing 2510.19304) and relates to multi-modality and density (Shabalin et al. 2604.02028).

**Not found anywhere:** the claim that (qE)W₂ᵀ is read **across positions within a step**, carries the favoured joint mode, and reduces hybrids, together with any causal test or attribution of the self-conditioning gain to it.

**Closest work:**

| Paper | Relation to M2 | Read |
|---|---|---|
| **Shabalin et al., "Why Gaussian Diffusion Models Fail on Discrete Data…" (2604.02028)** | **Strongest competing account.** Self-conditioning keeps trajectories out of low-density regions by keeping consecutive predictions close, with sequence-level modes. Hybrids *are* low-density regions, so this must be separated experimentally (→ new rival M5, §4). | [full, ~55%] |
| CoCommit, "Don't Commit Alone" (2607.04469) | Same "New/San" motivation; a learned cross-position coordination fix in *discrete* MDLM that deliberately doesn't pass token values. M2 is exactly the value-carrying channel it omits. | [full] |
| FMLM (2602.16813) | Same toy, credited to deterministic transport; self-conditioning not analysed. | [full] |
| Yoo et al. (2607.00714) | Fixed-point refinement improves coherence. **Prop. 3.1:** under conditional independence, the Bayes-optimal denoiser ignores the self-conditioning input, so self-conditioning only matters where single-pass prediction is imperfect. That fits our §2.2 view (self-conditioning = computation, not information). | [full] |
| Draxler et al., PTP (2512.21323); Gumbel distillation (2603.22216) | Alternative route to joint coherence (deterministic noise→token maps). Cite as alternatives. | [abs] |
| D3PM, DFM | Problem statement only (independent per-token sampling). | not read; cite via Lafa |

### M3 — Guidance anchoring
**Status: the exact claim is untested, and the prior evidence goes against it as the *main* mechanism.**
- No paper measures the self-conditioning gain as a function of the CFG scale, compares guided vs. unguided feedback, or varies the unguided branch's feedback.
- **But** SED, CDCD, LangFlow, Plaid and Davis all report large self-conditioning gains **with no guidance**, so M3's prediction ("the gain mostly vanishes at g = 1") will probably fail.
- **Closest:** Li & Xu, whose untied self-conditioning (solver steps on a corrected prediction, feedback stays raw, justified by mismatch) is structurally the same as Lafa's "step on guided, feed back unguided".
- **Other relevant work:**
  - ELF (2605.10938) is the opposite design: self-conditioning is the guidance condition and the fed-back prediction is guided.
  - Analog Bits proposed "self-guidance" (CFG between with/without self-conditioning).
  - SED zeroes self-conditioning in the unguided branch.
  - "How to Guide Your Language Flow" (2609.19356).
- **Decision:** demote M3 from a candidate headline to a **moderator**. Measure how the gain depends on g, but don't bet on it.

### M4 — Tie-breaking at low initial noise
**Status: partly known; the interaction is open.**
- **Known:**
  - reduced initial noise is a low-temperature, mode-seeking knob (CDCD: σ = 0.5 improves the NLL/entropy trade-off);
  - self-conditioning feedback can create attractors (2607.00588);
  - self-conditioning helps at every denoiser temperature (Davis).
- **Open:**
  - the self-conditioning gain as a function of initial noise;
  - "without feedback, positions stall between modes";
  - any commitment-time analysis of self-conditioning in a prefix-conditioned blockwise flow.
- **Null to beat:** under the ODE, the unstable central fixed point (Raya & Ambrogioni) plus the prefix-induced posterior asymmetry plus g = 8 may break symmetry **without any feedback**.

---

## 3. Safe Differentiation Statement (draft for the paper's intro)

> Prior work attributes coherent parallel generation in continuous flow LMs to deterministic joint transport (Lee et al. 2026), and explains self-conditioning as cross-step fixed-point refinement (Yoo et al.), density or trajectory stabilization (Shabalin et al.), train–inference mismatch correction (Davis et al.; Li & Xu), or per-position token identification (LangFlow). Whether, and through which channel, self-conditioning yields Lafa's +5.8 pp is untested. We decompose this gain causally and identify [the winning mechanism]. In particular, we test a distinct cross-position channel: within a denoising step, positions read their neighbours' fed-back distributions, which encode the favoured joint mode.

**Don't claim:**
- being first to observe hybrids;
- being first to link self-conditioning to coherence or multi-modality;
- being first to intervene on self-conditioning inputs (2607.00588 did α-scaling, cutoff and direction subtraction).

---

## 4. What Phase 3 Changes in the Design (→ Phase 4)

1. **New rival M5 (Shabalin et al.): cross-step trajectory stabilization.** Self-conditioning keeps each trajectory's consecutive predictions close, away from low-density (hybrid) regions. This acts mostly through each position's *own* feedback over time.
   - *Separating test:* our own-vs-neighbour decomposition (single-step direct effect, 2 × 2 F × R).
   - *Plus:* a control that matches cross-step prediction closeness ‖q_k − q_{k−1}‖ between conditions.
   - M5 predicts the neighbour direct effect ≈ 0, with hybrids reduced through own-feedback effects.
2. **A principled commitment-time definition** (replacing the ad-hoc τ_j in `phase1.md` §1.4), adapted from Biroli et al. 2402.18491 (cloning), Sclocchi et al. 2402.16991 (forward–backward, categorical-in-continuous) and Li & Chen 2403.01633 / 2502.00921 (critical windows):
   - **φ_i(t_k; ε) = P_ξ[argmax token_i(ODE_{t_k→1}(x_{t_k} + εξ)) = argmax token_i(ODE_{t_k→1}(x_{t_k}))]**, with ξ ~ N(0, I) and ε matched to 0.1σ;
   - **τ_i = inf{t_k : φ_i(t_s) ≥ 1 − δ ∀ s ≥ k}**, with δ = 0.225 (Biroli's φ(t_S) ≈ 0.775);
   - cheap proxy (Prophet 2508.19982): the top-1 − top-2 margin of q with a fixed argmax.
   
   Report on both the base time t_k and the per-position τ^i(t).
3. **Block-level susceptibility χ(t) = Σ_{ij} C_ij / Σ_i C_ii** (Sclocchi et al. 2410.13770) over the 4 positions. This measures whether positions change *jointly* or independently, and directly operationalizes "coordination" (M2) vs. "stalling" (M4).
4. **M3 design** (cheap, in the smoke test):
   - g ∈ {1, 2, 4, 8} × feedback ∈ {none, unguided q (Lafa), guided q} × unguided-branch feedback ∈ {own q (Lafa), zero (SED)};
   - also measure repetition (2607.00588's failure mode) with and without self-conditioning at g = 8.
5. **M4 null:** compare against a no-feedback run with matched tiny perturbations, to test whether noise amplification alone breaks ties.
6. **Fixed-point gap (Yoo):** ‖D̂(x, q) − q‖ per position over steps, a cheap M1/M5 diagnostic.
7. **Baselines for interventions:** reuse 2607.00588's α-scaling and cutoff of the feedback as standard comparisons.
8. **Yoo Prop. 3.1 → block-size prediction:** self-conditioning's value grows with within-block dependency. Lafa-0.5B was trained at B = 4/8/16 (paper Table 4), but **only B = 4 checkpoints appear to be released.** If B = 8/16 are obtainable, the self-conditioning gain vs. B is a strong M2 test.

---

## 5. Scoop and Competition Risk

- **The Kim / Boffi / Lee / Yoo group is very active on exactly this space:** FMLM (Feb), fixed-point self-conditioning (Jul), Lafa (Oct), Posterior Refinement (Jun), and **Asynchronous Flow Map LMs (2610.42018, Oct 30)**. They call self-conditioning's benefit "poorly understood" and are the most likely group to explain it. **Monitor arXiv weekly** for this group and for anything citing Lafa.
- Only one paper citing Lafa so far: aFMLM. Its abstract has nothing on self-conditioning or coordination [abs].
- Speed matters. The smoke test should run before Phase 4 finishes.

---

## 6. Per-Paper Ledger (one line each)

| Paper | Verdict | Implication |
|---|---|---|
| Lafa 2610.28559 [full] | context | Gives no explanation of self-conditioning; leaves room. |
| FMLM 2602.16813 [full] | phenomenon threat | Cite the hybrid toy; differentiate the channel. |
| Shabalin et al. 2604.02028 [full ~55%] | **main competing account** | Becomes rival M5; needs a separating test. |
| CoCommit 2607.04469 [full] | partial | Same motivation, discrete, no value passing. |
| Yoo et al. 2607.00714 [full] | partial | Fixed-point view; Prop. 3.1 supports self-conditioning-as-computation. |
| Davis et al. 2610.13834 [full ~50%] | partial (M1) | Per-position account; self-conditioning helps at all temperatures. |
| LangFlow 2604.11748 [full] | partial (M1/M4) | Prevents frequency drift at high noise; "open question". |
| Li & Xu 2608.22244 [full ~70%] | partial (M3) | Untied feedback ≈ Lafa's guided/unguided split. |
| ELF 2605.10938 [full ~50%] | partial (M3) | Opposite design (guided feedback). |
| How to Guide Your Language Flow 2609.19356 [full] | partial (M3) | Self-conditioning as guidance; "poorly understood". |
| SED 2211.04236 [full] | counter-evidence to M3 | Big self-conditioning gain without guidance. |
| CDCD 2211.15089 [full, keyword] | partial (M4), counter-evidence to M3 | Reduced noise = mode-seeking; self-conditioning helps at train + sample. |
| Plaid 2305.18619 [full] | minor | Self-conditioning helps the NLL bound. |
| Analog Bits 2208.04202 [full] | origin | Mismatch argument; self-guidance. |
| 2607.00588 [full, partial] | partial (prior causal self-conditioning work) | Attractor; α/cutoff/direction interventions as baselines. |
| 2610.19628 [abs] | supporting M2 | Self-conditioning adds sequential depth. |
| Raya & Ambrogioni 2305.19693 [full] | tool (M4) | Unstable fixed point → symmetry breaking; the null. |
| Biroli et al. 2402.18491 [full] | tool | Speciation time; cloning test. |
| Sclocchi et al. 2402.16991, 2410.13770 [full] | tool | Categorical-in-continuous phase transition; χ(t). |
| Li & Chen 2403.01633 / 2502.00921 [abs/full] | tool | Critical windows (formal definition). |
| Prophet 2508.19982 [sec] | tool | Margin-based commitment proxy. |
| ParallelBench, DCD, CoDD, DAPD, CoDiLA, SetCommit [abs] | partial | Related-work sentence on joint-vs-marginal fixes. |
| PTP 2512.21323, Gumbel distillation 2603.22216 [abs] | partial | Alternative joint-coherence route. |
| Loopholing 2510.19304, LAW 2610.30068 | partial | Soft-state feedback claimed relational; untested. |
| Gao & Gao 2611.05344; Kong et al. 2601.14758; DLM-Scope 2602.05859 [abs] | method precedent | Circuit and SAE methods in (D)LMs; no self-conditioning. |
| aFMLM 2610.42018 [abs] | watch | Concurrent; same group. |
| Dieleman blog (Aug 2026) [full] | motivation | "Still unclear why self-conditioning works." |

**Not yet read** (marked for before submission): the full texts of D3PM, DFM, ReMDM and TESS 2; the tails of the Lafa, Yoo, Shabalin and Davis HTML versions. The CDCD and SED texts were read only partly.

---

## 7. Gate Outcome

| Mechanism | Status | Role going forward |
|---|---|---|
| M1 | partly known (best-supported) | Null / baseline. Winning = least novel. |
| **M2** | phenomenon known, **mechanism open** | **Headline candidate** (differentiate from FMLM, CoCommit, Shabalin). |
| M3 | untested, but evidence against it as the main mechanism | **Demoted to moderator** (g-dependence measured). |
| M4 | partly known; interaction open | Secondary candidate; needs the "noise alone" null. |
| **M5 (new)** | Shabalin et al.'s cross-step density account | **First-class rival** to M2. |

**Gate:** M2 is **not** already shown → no reframe needed beyond the decomposition framing. **Proceed to Phase 4.**

---

## 8. Update (2026-11-15): novelty of the NEW headline (after the Phase 7 pilots)

**Claim checked:** "SC works mainly through positions reading each other's fed-back predictions (N share 0.7–0.9 at 0.5B, 0.55–0.8 at 7B; own feedback and the state route small); the SC repair grows with scale." Full reads unless noted.

| Paper | What it says | Verdict |
|---|---|---|
| **CANDI** (Pynadath, Shi, Zhang; 2510.22510) | App. B.3.2 speculates that SC works as "pseudo-clean anchors" (confident positions help denoise *others*); untested | **Partial: the closest prior statement.** We turn it into a causal measurement |
| **How Transparent is DiffusionGemma?** (Engels…Nanda; 2606.20560) | The SC matrix = softmax·W_E; top-8 truncation matches baseline; token smearing; §7.2 poses activation patching across the diffusion axis and circuit analysis (e.g. a token-smearing circuit) as open. **It does NOT mention self-conditioning or per-/cross-position patching in §7** (verified verbatim 2026-11-18; an earlier summary overstated this) | Partial (framing). We do patching across denoising steps + circuit analysis for an SC channel |
| Yoo et al. (2607.00714) | SC = sequence-level fixed point; mechanism "poorly understood" | Partial (competing explanation) |
| Shabalin et al. (2604.02028) | SC avoids low-density regions (temporal, per trajectory); toy scale | Partial |
| Davis et al. (2610.13834) | SC = mismatch + identifiability (per position); no size ablation | Partial (main competing explanation) |
| Li & Xu (2608.22244) | Solver redundancy / mismatch; no attention analysis | Irrelevant to the mechanism (cite for failure modes) |
| **SCMDM** (Cardei et al.; 2604.26985) | "Benefit of full SC grows as the pretrained backbone becomes more capable" (cross-backbone) | **Partial for the scale finding.** Ours is a controlled 0.5B-vs-7B comparison within one family |
| SC enables deep sequential reasoning (2610.19628) | SC as a scratchpad (depth); attention share to SC overall, no own/other split; **App. D.5.2 not read** | Partial. **Read D.5.2 before submission** |
| Async FMLM (2610.42018) | "Tokens that denoise earlier inform others"; no SC | Partial (coordination framing) |
| Loopholing All the Way (2610.30068), Posterior Refinement (2606.24773) | No SC mechanism | Irrelevant / related |
| CoCommit (2607.04469), DiffusionGemma commitment (2606.14620), ParallelBench (2510.04767), Mean-Field Parallel Decoding (2606.15805), Measured Coupling (2611.08307) | Cross-position coordination / factorization error in parallel decoding | Framing (related work) |

**Bottom line**
- **Novel:** the causal own / neighbour-read / neighbour-state decomposition of SC, and the result that neighbour-reading dominates (no prior split; DiffusionGemma-transparency lists per-position SC patching as open).
- **Partly known:** the scale trend (SCMDM, cross-backbone).
- **Partly anticipated:** the cross-position intuition (CANDI's untested remark).

**Draft differentiation statement (intro):** "Prior explanations of self-conditioning are per-position or temporal: correcting a train/inference mismatch or token identifiability (Davis et al., 2026), implementing fixed-point refinement (Yoo et al., 2026), or avoiding low-density regions (Shabalin et al., 2026); CANDI (Pynadath et al., 2025) speculates, without testing, that previous predictions act as cross-position anchors. We give the first causal decomposition of self-conditioning's benefit into own-feedback, neighbour-reading, and neighbour-state routes, and show in a blockwise flow LM at 0.5B and 7B that neighbour-reading dominates: self-conditioning is primarily a cross-position coordination channel."

**Must-cite:** Davis et al.; Yoo et al.; CANDI (+ SCMDM for scale).

**To do before submission:**
- read 2610.19628 App. D.5.2;
- re-check papers citing Lafa (the Semantic Scholar rate limit blocked it) and the arXiv listings for 14–15 Nov;
- skim TDLM 2610.36940, Simplex Diffusion 2609.35553, CoDD 2603.00045, DAPD 2603.12996, CADD 2510.01329.

### 8.1 Merged with an independent GPT review (2026-11-16; all IDs below verified to exist)

Same verdict: the decomposition and the dominance of neighbour-reading are **apparently novel**; the cross-position intuition is partly known (CANDI).

**Wording constraints that change the paper:**
- Do **not** claim the "first causal intervention on SC": **Zhang et al. 2607.00588** (an SC direction drives repetition; a causal steering intervention, which transfers from ELF-B to ELF-M/L, App. F.2). We claim the first **own vs. neighbour route decomposition**.
- Do **not** claim the first scale study of feedback: **Loopholing (Jo et al. 2510.19304)**, which compares 169M vs. 424M (App. D.9) and latent vs. sampled-token feedback (App. D.11); also **SCMDM 2604.26985**. We claim a **controlled 0.5B → 7B comparison within one family**.
- **State the scale ratio precisely:** the SC coherence gain is 4.2× (own-AR) / 1.9× (Mistral) at 7B vs. 0.5B; 38% of 7B no-SC answers are garbled. Name the metric for every ratio.

**New must-cite:** **He, Welleck & Fried 2602.03769** (Reasoning with Latent Tokens): causal evidence that positions rely on joint reasoning about undecoded tokens. *We differ:* we isolate communication through the **temporally fed-back predicted embeddings**, not within-pass hidden states.

**Must-cite set (final):**
- Davis 2610.13834;
- Yoo 2607.00714;
- CANDI 2510.22510;
- He et al. 2602.03769;
- Zhang 2607.00588;
- SCMDM 2604.26985 / Loopholing 2510.19304 (scale);
- DiffusionGemma-transparency 2606.20560.

**Related work (verified):**
- Alpha-DLM 2609.38066 (marginal fitting → inconsistency; "factorization alone is not the problem");
- DA-DLM 2609.15070;
- Loopholing All the Way 2610.30068;
- TDLM 2610.36940;
- DStill 2610.27379 (token factorization error);
- DLM-Scope 2602.05859 (ICML 2026; the PMLR page is unconfirmed);
- TSF survey (Findings EMNLP 2026.findings-emnlp.1739; search window ends Mar 2026);
- Nemotron-Labs-Sigma 2610.02665;
- Async FMLM 2610.42018 (cites Lafa).

**Do not cite:** PURL 2611.11551, which was **withdrawn** (v2, 10 Nov 2026).

**Novelty statement (merged, for the intro):** "Prior work explains self-conditioning as iterative fixed-point refinement (Yoo et al.), correction of train/inference mismatch and token identifiability (Davis et al.), or avoidance of low-density regions (Shabalin et al.), and causal interventions on self-conditioning have so far targeted failure modes such as repetition (Zhang et al.). CANDI conjectures that previous predictions act as pseudo-clean anchors for other positions, and He et al. show causally that diffusion LMs reason jointly across undecoded positions. We isolate the routes through which fed-back predictions act (a position's own feedback, neighbours' feedback read through attention, and neighbour-state refinement) and find, in Lafa at 0.5B and 7B and in both the mechanism and deployed decoding regimes, that reading neighbours' feedback carries most of the benefit."

### 8.2 Novelty of the new mechanism claims (2026-11-17)

**Substitution:** neighbour feedback stands in for neighbours' uninformative states; the effect follows the schedule and goes to 0 once states become informative. **Localization:** distributed over ~25–33 mid-layer heads, L8 ≈ 20%.

**Both are novel.**

**Closest work:**
- **Li & Xu 2608.22244** (full text): SC is redundant when the solver has folded the previous prediction into the latent, a **feature-space, per-position** train/inference mismatch that grows with step width. No cross-position, attention or per-timestep analysis. **PARTIAL.**
- **Davis et al. 2610.13834 App. F.6:** a theoretical per-noise-level bound (an SC input beats a noisy input); per-position, no ablation. **PARTIAL.**
- **Shabalin 2604.02028:** SC acts in a "critical interval" t/T ∈ [0, 0.4] (low-noise end); observational, trajectory-level. **PARTIAL.**
- **Induction in Both Directions 2607.15893:** causal head localization in small masked DLMs without SC (sparse circuits). **PARTIAL** (framing).
- **DiffusionGemma transparency 2606.20560 §7.2** lists activation patching across denoising steps and circuit analysis as open. **Cite this as the stated gap.**
- Irrelevant: Yoo 2607.00714, Loopholing 2510.19304, TDLM 2610.36940, 2610.22330.
- Analog Bits: abstract only; check for a timestep ablation before citing.

**Differentiation from Li & Xu:**
- They treat SC redundancy as a feature-space train/inference mismatch: a token's previous prediction is already folded into its own latent.
- We show causally that SC's benefit comes from positions reading their neighbours' fed-back predictions through attention, as a stand-in for neighbours' still-uninformative states. So the effect follows the sampler's schedule, vanishes once states become informative, and is carried by distributed mid-layer heads.


### 8.3 Correction (2026-11-18, verified verbatim from arXiv HTML v2)
- DiffusionGemma-transparency §7.2, item 2: "Can we do 'activation patching' ... across the diffusion axis ...?" Item 7: "is there a simple circuit implementing token smearing?" **§7 does not mention self-conditioning or per-/cross-position effects.**
- **Safe wording:** "poses activation patching across denoising steps and circuit analysis of non-autoregressive behaviours (e.g. token smearing) as open problems."
- **Contrast to discuss:** their §7.2 item 4 says "ablating almost all information from intermediate vectors barely damages performance" (the SC channel is highly compressible in DiffusionGemma). In Lafa, removing the cross-position feedback causes large coherence losses and failures. Discuss as a regime / architecture difference (masked vs. continuous flow; block size), not a contradiction.
