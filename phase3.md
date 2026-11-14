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
