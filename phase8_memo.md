# mechflow — Phase 8 Decision Memo (2026-11-15)

**Recommendation: GO, reframed around the self-conditioning coordination mechanism.**

---

## 1. What the exploration pilots showed (all pre-confirmation; exploration split only)

| Question | Result | Verdict |
|---|---|---|
| **P1:** Do flow LMs mix answers where their own AR head doesn't? | 0.5B: +7.5 pp hybrids [5.5, 9.7] (non-valid +20 pp). **7B: +0.3 pp** | Real but small, and **a small-model effect**. Below the 10 pp pilot bar; would likely pass the 5 pp confirmatory margin at 0.5B. |
| **P2:** Does SC repair in-block coherence? | dep_{2,3} **+0.50 [0.20, 0.82]**; own-AR coh +0.75; **Mistral judge +1.44 [1.01, 1.90]**; effect at later positions only | **Strong pass** |
| **K11 (scale)** | 7B hybrids ≈ 0, **but the SC coherence repair is 2–4× larger at 7B**; without SC, 38% of 7B answers are garbled | **Not triggered.** The core effect grows with scale |
| **K12 (feedback vs. state window)** | Feedback leads the state by 0.1–0.2 accuracy at every step, but never "q confident while x uninformative" | **Triggered** → C1′ out; primary family = **{P1, P2, H2-N}** |
| **H2-N:** Which channel carries SC's benefit? (2 × 2 × 2) | **0.5B: neighbour-feedback reads = 70–90%** of the effect (dep and coh); own feedback ≈ 0; state route ≈ 0. **7B (coh): 55–80%**, own feedback ~19%, N × F synergy | **Strong support for coordination (M2)** |
| **G1 (regime relevance)** | Coherence gain MR/PR = 1.0–1.5×; end task MR mean **+5.4 pp [1.3, 9.7]** | **Passed** |
| Instruments | Exact sampler, positive control, K3 null-calibrated pass, paper number reproduced (30.0 vs. 30.4) | Valid |

## 2. What changed
- **Headline:** "flow LMs mix answers" is **demoted** (small-model effect). The new headline is that **self-conditioning is what keeps parallel decoding coherent, it grows with scale, and it works mainly through positions reading each other's fed-back predictions.**
- **Mechanism evidence** now rests on H2-N (C1′ was not separable), and the pilot evidence for it is strong at both scales.
- **Localization** was pre-registered as a confirmatory family (L1/L2, §2.4) before any confirmation data, and the per-head patching tool is built and verified.

## 3. Decisions needed from you
1. **GO / reframe / kill.** Recommendation: **GO**.
2. **Add the guidance-on 2 × 2 × 2 before confirmation?** It answers "the mechanism regime isn't the deployed one". It must be pre-registered (as a secondary test) before any confirmation data. Cost: ~½ day of code + validation. **Recommended.**
3. **Classifier audit spot-check.** P1 stays in the primary family and needs the audited classifier (≥ 90% agreement). I'll do the 150-label first pass; you spot-check ~20 (≈ 10 min).

## 4. Remaining before confirmation runs
- [ ] Classifier audit (P1)
- [ ] (if decision 2 = yes) guidance-on 2 × 2 × 2 + prereg amendment + push
- [ ] Freeze: the final prereg hash pushed (the GitHub commit is the external anchor)
- Optional, not blocking: S2 natural-text pilot, LEACE (only needed for the secondary S9 now), Tier-1 interp screens, literature-ledger additions

## 5. Then (if GO)
1. **Confirmatory runs** (held-out): P1 (~600 pairs), P2 (~2,000 docs), H2-N (~2,000 docs × 8 cells, 0.5B; 7B subset), secondary tests.
2. **Localization:** layer sweep → heads → freeze k, H\* → L1/L2 on circuit-validation.
3. ELF replication → sampler upside → paper.

**Wall-clock estimate:** ~1–2 weeks to a full draft, GPU-queue bound.
