# Paper outline (working; 2026-11-18)

**Working title:** *Neighbours, Not Selves: How Self-Conditioning Coordinates Parallel Decoding in Flow Language Models*

## Claim stack (evidence → section)

1. **Phenomenon** (§3): without SC, parallel decoding fails in the two classic NAT ways (cross-answer hybrids such as "San York", and duplicated tokens).
   - P1 confirmed: +7.0 pp hybrids vs. AR at 0.5B.
   - 7B: hybrids ≈ 0 with SC, but 32% non-valid without SC.
   - SC halves hybrids and climbs the ladder (S11).
2. **SC repairs coherence** (§4): P2 confirmed (+0.54 dep; external judge +1.34). Not guidance (S3), not noise (S4). Regime-relevant (G1 +5.4 pp end task).
3. **Mechanism: neighbours, not selves** (§5):
   - H2-N confirmed: neighbour-feedback reading carries ≈ 100% (MR) / 73% (guidance on, S12).
   - Own feedback ≈ 0, and can hurt (MR).
   - Toy model trained with neighbour-only SC ≥ full SC; own-only ≈ none, so the mechanism is a property of the algorithm.
4. **When and why** (§6):
   - The schedule manipulation (E1c) moves the effect: feedback matters while the neighbours' states are uninformative.
   - Two redundant channels, states and feedback (G-a; family R confirmation pending).
5. **Where** (§7):
   - Late steps, mid layers (L8 hotspot).
   - Distributed over 26 heads (k selected on held-out exploration docs).
   - **L2: cutting them nearly triples cross-answer/garbled outputs at deployed settings (+28.9 pp vs random)**.
   - L1 share test inconclusive (reported transparently).
6. **Generality** (§8): toy model; 7B; ELF (pending).

## Methods contributions
- Causal 2 × 2 × 2 decomposition of a recurrent feedback channel (shadow-pass decoupling).
- Per-query K/V routing for path patching through the denoising-time feedback edge.
- Schedule manipulation as a falsification tool for mechanistic accounts.
- Full pre-registration with public hash anchoring; deviations logged before data.

## Figures (planned)
1. Hybrid examples + rates (SC on/off, 0.5B / 7B, AR).
2. 2 × 2 × 2 decomposition bars (MR, PR, toy).
3. Step × layer heat map + schedule manipulation (`warp` / `uniform` / `uniform_range`).
4. Two-channel redundancy (G-a / family R).
5. L2 behavioural cut: non-valid rate base vs. random vs. H\*, with examples ("Franz Kafka" → "Fran King").

## Related work anchors
Davis et al.; Yoo et al.; CANDI; He et al. (latent tokens); Zhang et al. (repetition direction); Li & Xu; Shabalin; SCMDM / Loopholing (scale); DiffusionGemma transparency (open problem §7.2); Gu et al. (NAT multimodality); Shai et al. (factored representations); Induction in Both Directions.
