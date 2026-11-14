# mechflow — Phase 3.5 Smoke Test Memo

Date: 2026-11-13. **Exploratory only.**
- **Model and hardware:** Lafa-0.5B (HF snapshot `18deca35`, EMA weights, bf16) on an Apple M4 Pro (MPS).
- **Code:** `mechflow/smoke/`.
- **Data hygiene:** every GSM8K test index used here is logged in `smoke/results/*.jsonl` and must be **excluded** from any later confirmation split.

---

## 1. Block-Level Teacher-Forced Test (`02_block_tf.py`, n = 400 blocks) — DONE

**Setup**
- For each example, the prefix is the chat prompt plus a gold GSM8K solution cut at a random point.
- One 4-token flow block is generated under each configuration and scored against:
  - (a) the gold tokens;
  - (b) the checkpoint's own **AR greedy** block from the same prefix.
- Every "SC off" condition uses the trained off-state (`self_cond=False`, plain x input).

**Results** (paired SC − noSC, 95% bootstrap CI over blocks)

| Setting | Δ tok vs gold | Δ EM vs gold | Δ tok vs AR | Δ EM vs AR |
|---|---|---|---|---|
| K32, g8, noise 0.1 (paper) | +0.005 [−0.011, +0.020] | +0.007 [−0.018, +0.033] | **+0.045 [+0.022, +0.068]** | **+0.083 [+0.045, +0.120]** |
| K32, g1, noise 0.1 | +0.006 [−0.009, +0.021] | +0.030 [+0.003, +0.058] | **+0.035 [+0.015, +0.056]** | **+0.075 [+0.040, +0.115]** |
| K32, g8, noise 4.0 | +0.007 [−0.007, +0.021] | +0.022 [−0.003, +0.048] | **+0.039 [+0.021, +0.058]** | **+0.065 [+0.033, +0.098]** |
| K4, g8, noise 0.1 | +0.013 [−0.001, +0.028] | +0.020 [−0.005, +0.045] | **+0.028 [+0.009, +0.047]** | **+0.048 [+0.015, +0.083]** |

Absolute accuracies vs gold are nearly identical across all configurations, including AR: token accuracy ≈ 0.58–0.60, EM ≈ 0.37–0.41.

**Breakdown at the paper setting**
- SC changes the output in **116 / 400 blocks (29%)**.
- Agreement with AR rises most at the **later** block positions, which denoise more slowly under the asynchronous schedule:

  | Position | 0 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | SC on | 0.93 | 0.86 | 0.79 | 0.73 |
  | SC off | 0.90 | 0.83 | 0.74 | 0.67 |

- In blocks where the two differ, the SC output matches the full AR block 48 times vs. 15 times without SC.
- The most common no-SC patterns are "**matches AR on the first 2–3 positions, then deviates**" (`1100`: 24, `1110`: 20).

**Reading (provisional)**
1. **SC barely changes teacher-forced correctness against gold.** Its effect is to pull the block toward the model's own dominant continuation (AR greedy). One gold solution is only one of many valid continuations, so this metric can't tell "worse" from "different". → **The coherence check (`04_coherence.py`) is pending.** It tests whether no-SC blocks are *less coherent* under the AR head, or just alternative valid continuations.
2. **M3 (guidance anchoring) is not supported as the main mechanism.** The SC effect is about the same at g = 1 as at g = 8 (Δ EM vs AR +0.075 vs. +0.083; CIs overlap heavily). This matches the literature.
3. **M4 (tie-breaking at low noise): weak at most.** The SC effect is slightly smaller at noise 4.0 than at 0.1 (+0.065 vs. +0.083), but the CIs overlap.
4. **The SC effect is larger at 32 steps than at 4 steps** (+0.083 vs. +0.048 Δ EM vs AR).
5. **Suggestive for M2 / M5, not decisive.** No-SC errors concentrate at later positions *after* earlier positions already matched ("prefix match, then deviate"). Two readings fit:
   - **M2:** later positions fail to stay consistent with earlier positions' committed tokens;
   - **M1/M5:** later positions are simply harder, and per-position stabilization helps them more.
   
   Separating these needs the Phase 4–6 interventions. The cross-position correctness coupling (EM − product of per-position agreement) doesn't change with SC (0.219 vs. 0.232 at g = 8).
6. **Teacher-forced single blocks show no gold-accuracy gain.** This is resolved by §3: gold-match is a poor metric because many continuations are valid. What SC improves is *coherence*.

## 2. Hybrid Test (`03_hybrid.py`, 53 prompts) — DONE

**Setup**
- Prompts ask for a two-word entity in three categories (US cities, countries, scientists).
- Two kinds of prompt: 9 "open" prompts, and 44 "pick one of A or B" prompts (A and B are recombinable, e.g. New York / San Francisco).
- Scoring is at word level: **valid** (a real entity), **hybrid** (words from different entities), **other**.
- Decoding is deterministic at noise 0.1; noise 4.0 is run with 4 seeds.

**Rates** (valid / hybrid / other)

| Config | valid | hybrid | other |
|---|---|---|---|
| **AR** | **0.89** | **0.00** | 0.11 |
| K32 g8 n0.1 SC (paper) | 0.53 | 0.28 | 0.19 |
| K32 g8 n0.1 noSC | 0.38 | 0.23 | 0.40 |
| K32 g1 n0.1 SC / noSC | 0.62 / 0.38 | 0.17 / 0.25 | 0.21 / 0.38 |
| K32 g8 n4.0 SC / noSC (4 seeds) | 0.57 / 0.43 | 0.13 / 0.18 | 0.30 / 0.38 |
| K4 g8 SC / noSC | 0.47 / 0.40 | 0.26 / 0.21 | 0.26 / 0.40 |
| K1 g8 SC / noSC | 0.42 / 0.25 | 0.32 / 0.36 | 0.26 / 0.40 |

On the pair prompts alone (n = 44):
- **AR is 100% valid.**
- Paper flow with SC: 61% valid, **30% hybrid**.
- Flow without SC: 45% valid, 27% hybrid, 27% other.

**"Other" without SC is mostly sub-word mode mixing**, which the word-level classifier misses. At the paper setting: "Albertaac" (Albert + Isaac), "Marileo" (Marie + Galileo), "Newleveland", "Newolulu", "Costrist Rica", "Charles Cur", "Ada Lov". With SC, "other" is mostly legitimate off-list answers (Cleveland, Canada, Cuba).

**Paired noSC → SC transitions** (paper setting):
- valid→valid 20; other→other 10; **other→hybrid 9**; hybrid→hybrid 6; **hybrid→valid 6**; other→valid 2;
- **never valid→worse.**

**SC moves outputs up a coherence ladder: garbled sub-word mix → word-level hybrid → valid entity.** It reduces mode mixing but doesn't remove it.

Examples of hybrids at the paper setting with SC: "San York", "Las Orleans", "New City", "Saudi Korea", "Albert Newton", "Charles Curie".

Note: at K = 1, g = 1 and g = 8 give identical outputs. The loop reads out the conditional logits before any guided Euler update, so guidance has no effect at one step.

**Reading**
1. **The M2 phenomenon exists at the paper setting, at a high rate.** On bimodal prompts, Lafa's flow decoding produces cross-mode hybrids (~30% word-level, plus sub-word garbles without SC), while the same checkpoint in AR mode produces **none**. This contradicts the common claim that continuous flows avoid factorization errors in practice (FMLM 2602.16813 shows a toy). → **K4 (phenomenon absent) is not triggered.** This is itself a potential headline finding.
2. SC partially mitigates it: it climbs the coherence ladder and never makes outputs worse. It isn't sufficient.
3. *Caveats:* the prompts are deliberately bimodal and artificial; n = 53 with deterministic outputs, so the CIs are wide (±~12 pp); the word-level classifier undercounts sub-word hybrids.

## 3. Coherence Check (`04_coherence.py`, the 400 blocks from §1) — DONE

Each block is scored by the **checkpoint's own AR log-probability given the prefix**, summed over the 4 tokens.

**Mean block log-prob**

| Config | Mean log-prob |
|---|---|
| AR | −0.74 |
| paper SC | **−1.28** |
| paper noSC | −2.31 |
| g1 SC / noSC | −1.69 / −2.84 |
| n4.0 SC / noSC | −1.56 / −3.16 |
| K4 SC / noSC | −1.85 / −2.76 |
| gold | −4.53 (gold is one valid but less-modal continuation) |

**In blocks where SC changes the output** (paired SC − noSC; 95% CI)

| Setting | n differing | Δ block log-prob | SC more coherent | Per-position Δ (pos 0, 1, 2, 3) |
|---|---|---|---|---|
| K32 g8 n0.1 (paper) | 116 | **+3.58 [+2.38, +4.88]** | 73% | +0.17, +0.13, **+1.48, +1.81** |
| K32 g1 n0.1 | 123 | **+3.76 [+2.42, +5.17]** | 69% | +0.03, +0.69, +1.30, +1.74 |
| K32 g8 n4.0 | 100 | **+6.39 [+4.63, +8.13]** | 75% | +0.01, +1.56, +1.80, +3.02 |
| K4 g8 n0.1 | 110 | **+3.31 [+1.76, +4.83]** | 73% | +0.08, +0.73, +1.23, +1.27 |

**Reading**
1. **Without SC, blocks are substantially less coherent, not just different.** This resolves the §1 puzzle: SC's benefit shows up as coherence, which gold-match can't see.
2. **The gain is concentrated at later block positions.** At positions 2–3 the AR log-prob conditions on the *earlier tokens of the same block*. Position 0 depends only on the prefix and is essentially unchanged.
   - Without SC, later tokens fit their in-block predecessors poorly. That's exactly the cross-position inconsistency M2 is about.
   - **But it doesn't yet distinguish M2 from M1/M5.** Later positions are also the slowest under asynchronous time, so per-position stabilization could produce the same pattern. The Phase 4–6 direct-path interventions are required.
3. **M4 is contradicted.** M4 predicted a larger SC benefit at *low* noise. The coherence benefit is *larger* at noise 4.0 (+6.4 vs. +3.6).
4. **M3 is not supported.** The coherence benefit is the same at g = 1 and g = 8.

## 4. End-to-End GSM8K Sanity Check (`01_gsm8k_e2e.py`, n = 50) — DONE

Paper settings; the prompt only approximates lm-eval `gsm8k_cot`, so absolute numbers aren't comparable to the paper.

| | Accuracy |
|---|---|
| SC on | 0.38 |
| SC off | 0.34 |
| Δ | +0.04 [−0.10, +0.18] |

7 problems were solved only with SC, 5 only without. The direction matches the paper; n = 50 is underpowered (expected). Full benchmarks belong on CARC.

## 5. Verdict on M1–M5 → Input to Phase 4

| Mechanism | Smoke evidence | Status for Phase 4 |
|---|---|---|
| **M1** per-position stabilization | Can't be separated from M2 yet: later positions are both slowest and most improved. | Live rival; the direct-path tests decide. |
| **M2** cross-position coordination | **Phenomenon confirmed and large:** hybrids ~30% on bimodal prompts vs. 0% for AR. SC improves within-block coherence specifically at positions that condition on in-block predecessors (+1.5–1.8 nats), climbs the hybrid ladder, and never hurts. | **Headline candidate, strengthened.** |
| **M3** guidance anchoring | SC effect ≈ equal at g = 1 and g = 8, on every metric. | **Drop as a candidate.** Keep g = 1 as a cheaper regime for mechanism work (2× fewer passes). |
| **M4** low-noise tie-breaking | SC benefit is *larger* at noise 4.0. | **Contradicted; drop.** |
| **M5** cross-step trajectory stabilization (own feedback) | Not tested directly; still predicts the same position pattern as M2. | Live rival; separated by the own-vs-neighbour direct-path design. |

**Implications for Phase 4**
1. The project narrows to **M2 vs. M1/M5**. That is exactly what the Phase 1 direct-path / 2 × 2 machinery was built for.
2. **Primary metrics:**
   - in-block **AR-coherence per position** (log-prob under the checkpoint's own AR head, conditioned on in-block predecessors);
   - a **token-level hybrid classifier** that also catches sub-word mixtures. Gold-match is demoted.
3. **Positions 2–3 are where the effect lives.** Pre-register position-specific predictions.
4. **A new potential headline**, independent of the mechanism: *continuous-flow LMs at their published settings still mode-mix heavily on bimodal choices, and the AR head of the same model does not.* Check novelty against FMLM / CoCommit before leaning on it.
5. g = 1 is a valid and 2× cheaper regime for mechanism experiments, given that M3 is dropped. **Still confirm that the end-to-end SC gap exists at g = 1** (on CARC).
6. Data hygiene: exclude the GSM8K test indices used here (logged in `smoke/results/`) from all confirmation splits.
