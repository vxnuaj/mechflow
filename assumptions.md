# mechflow: background assumptions register

We do not justify claims from axioms. Every result rests on background assumptions we provisionally accept so we can test anything at all. Each is stated so it can be criticized, with its status and the check that would expose it. A failed prediction may implicate any of them (Duhem–Quine), so claims cite the assumptions they depend on.

**Rules:**
- assume what any test needs, and check it where checking is cheap;
- assume only what is independently corroborated;
- never assume the claim under test.

## A. Instrument assumptions

| ID | Assumption | Status | Evidence / check |
|---|---|---|---|
| A1 | The interventions change only what we intend (routing of K/V, feedback inputs) | **Checked** | The (1,1,1) and (0,0,0) cells reproduce SC-on / SC-off sampling token-for-token on every block (Lafa 0.5B, 7B, PR with CFG); ELF to float precision; batched = unbatched (identical tokens) |
| A2 | Removing a channel removes information rather than corrupting the computation (no off-distribution breakage) | **Strongly supported** | Under `uniform`, late in decoding the feedback-only cut has exactly zero effect (family R, R2: 0.000, upper bound 0.004); off-distribution damage would show there too. SC-off is in-distribution (both models train with SC dropped half the time) |
| A3 | The sampler is the one the authors deploy | **Checked** | Reproduces Lafa's GSM8K number (30.0 vs 30.4); the ELF loop matches the authors' implementation |
| A4 | Memory-only code changes do not change results | **Checked** | Lean AR scoring bitwise-equal (A40); chunked 7B scoring: identical samples, ≤ 0.11 nats per-sample log-prob difference (logged) |

## B. Measurement assumptions

| ID | Assumption | Status | Evidence / check |
|---|---|---|---|
| B1 | Log-probability under a language model tracks coherence / consistency | **Assumed; partly corroborated** | Two independent judges (Lafa's own AR head; Mistral-7B) agree on every mechanism result. **Open check:** agreement with human judgements on a stratified sample (~100 pairs) |
| B4 | Route attribution is not confounded by context co-variation | **Checked (R-FC):** under fixed-context scoring O ≈ 0 and N is unchanged in absolute terms, but F rises from ≈ 0 to ≈ N. Actual-context scoring masks F |
| B2 | dep_j (PMI with sampled predecessors) measures in-block dependence | **Assumed** (pre-registered definition) | Agrees in sign and route shares with coh and with the external judge; noisy at 7B (`AR_CHUNK` / S = 64 rescoring planned) |
| B3 | The hybrid classifier labels outputs correctly | **Checked** | 150-label audit: 98.7% agreement, hybrids 100% |

## C. Statistical assumptions

| ID | Assumption | Status |
|---|---|---|
| C1 | Documents (or prompt pairs) are independent clusters; the cluster bootstrap is valid | Standard; clustering by document/pair handles within-unit dependence |
| C2 | Exploration and confirmation splits come from the same distribution | By construction (random split, seed 20261113) |

## D. Scope assumptions (NOT assumed; tested or reported as limits)

| ID | Claim we do NOT assume | Status |
|---|---|---|
| D1 | The mechanism is the same in other architectures | **Tested: partly fails** (ELF uses the state route). Candidate design-level cause: ELF zeroes the SC input at its decode pass (2605.10938 Alg. 5), so feedback can reach the output only through states. E3 is under test |
| D2 | The mechanism is the same at scale | **Tested, confirmed (S7, 7B):** N share 0.86 (Mistral 0.76); own feedback a small positive 0.07 |
| D3 | The toy model's mechanism is the real one | Not assumed: an analogy that shows the algorithmic property |
| D4 | The mechanism holds beyond chat answers and two-answer prompts | **Open:** a decomposition on another text type (code / math) is planned |
| D5 | Attention heads are the natural unit | Not assumed: the effect is distributed (26 heads carry 49.5% vs 4.2% random; L1-R confirmed) |
| D6 | The transmitted content is "which answer" in general text | Shown only for bimodal prompts (steering). **Open:** probe what j's state knows about its neighbour's token with vs. without feedback |
| D7 | The mechanism regime (no guidance, training noise) describes deployment | **Checked:** S12 (N 0.73 with guidance), S3/S4 equivalence, G1 end tasks |
