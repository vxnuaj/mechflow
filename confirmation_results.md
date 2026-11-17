# mechflow — Confirmatory Results (2026-11-17)

Analysis: `analysis/confirm.py` (hashed 2026-11-16 before any confirmation data; validated on exploration pilots). Raw output: `src/results/confirm_final.txt`. All confirmation runs are complete, with zero exclusions. Every 2 × 2 × 2 block passed both identity checks (2,000 / 2,000 MR; 750 / 750 PR).

Deviations: all logged in prereg §12 before any outcome was inspected:
- the secondary-family scope;
- the memory-only AR-scoring fix (bitwise-equivalent);
- P1 run on Apple MPS.

## Primary family (Holm, α = 0.05): **all three SUPPORTED**

| Test | Estimate [95% BCa] | One-sided 95% bounds | Margin | p | Decision |
|---|---|---|---|---|---|
| **P1** mode mixing: h_flow − h_AR (0.5B, PR; 602 pairs, 9,632 prompts) | **+6.97 pp** [6.09, 7.95]. Hybrids: flow-SC 7.1%, flow-noSC 14.2%, AR 0.1% | [6.23, 7.79] pp | 5 pp | < 5 × 10⁻⁵ | ✓ |
| **P2** SC repairs in-block dependence: Δdep_{2,3} (0.5B, PR; 7,100 docs) | **+0.540** [0.477, 0.604] | [0.487, 0.594] | 0.10 | < 5 × 10⁻⁵ | ✓ |
| P2 co-requirement: Mistral Δcoh_{2,3} LB > 0 and own-AR Δcoh_{2,3} > 0 | Mistral **+1.34** [1.24, 1.43] (LB +1.26); own-AR +0.97 | — | — | — | met |
| **H2-N** feedback-coordination share: N − 0.20·T on dep_{2,3} (0.5B, MR; 2,000 docs) | **+0.515** [0.409, 0.624]; T = +0.754 [0.591, 0.917]; Shapley N share **1.14** (coh: 0.99) | [0.426, 0.607] | 0 | < 5 × 10⁻⁵ | ✓ (estimable) |

## Secondary family (BH, q = 0.10): 4 / 5 supported

| Test | Estimate [95%] | p | Decision |
|---|---|---|---|
| S11 hybrid ladder: up − down transitions (SC-off → SC-on) | +0.387 [0.367, 0.407] | < 5 × 10⁻⁵ | ✓ |
| S12 H2-N in the paper regime (guidance on): N − 0.20·T | +0.440 [0.302, 0.584]; Shapley N share 0.73 | < 5 × 10⁻⁵ | ✓ |
| S3 SC × guidance interaction within ±0.25 (TOST) | −0.002 [−0.164, +0.162] | 0.0015 | ✓ (equivalent) |
| S4 SC effect at noise 0.1 minus 4.0 < +0.25 | +0.027 [−0.196, +0.250] | 0.025 | ✓ (non-inferior) |
| S1 P1 at 7B: h_flow − h_AR > 5 pp | +0.25 pp [0.08, 0.46] | 1.0 | ✗ (mixing ≈ 0 at 7B) |

## Interpretation (prereg §7, frozen)
Row **P1 ✓, P2 ✓, C1′ n/e (K12), H2-N ✓** maps to: *"The feedback channel carries a substantial share; beyond-state content isn't shown (not separable)"*, i.e. phenomenon plus the "feedback channel carries the repair" mechanism. On the data:
- **Mechanism (headline):** SC's in-block dependence gain is carried almost entirely by position j **reading its neighbours' fed-back predictions** (Shapley share ≈ 1.0 for both dependence and coherence). This also holds with guidance on (0.73).
- **Repair:** SC robustly repairs in-block dependence, and the external judge agrees (Mistral +1.34 nats).
- **Not guidance or noise:** the SC effect is equivalent across guidance strength (S3) and non-inferior across inference noise (S4). So M3/M4 are not the explanation.
- **Phenomenon:** flow mixing beyond AR is real at 0.5B (+7.0 pp) but **≈ 0 at 7B** (S1). Report it as a small-model phenomenon, with SC halving it (14.2% → 7.1%) and moving outputs up the ladder.
