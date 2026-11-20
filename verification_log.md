# Source-of-truth verification log (2026-11-19)

Every external fact the generalization plan relies on, checked by Claude against a primary source: paper full text (arXiv HTML), official code, or the Hugging Face API.
- ✅ = verified at the source.
- ❌ = the claim was wrong.
- ⚠️ = not verified.

## Models
| Fact | Source checked | Status |
|---|---|---|
| ELF zeroes SC at the decode pass | arXiv 2605.10938v3 Alg. 5: `z = self_cond_proj(concat([z, zeros_like(z)])); h = net(z, t=1, …, mode="decode")`. Alg. 4: the decoder is trained on corrupted z = p·x + (1−p)·e, also with zero SC | ✅ |
| ELF authors | arXiv API: Hu, Qiu, Lu, Zhao, … (incl. Kim, Andreas, He) | ✅ |
| Plaid 1B weights; expected-embedding SC; SC at readout | github igul222/plaid `sample.py` (L147, 162: x_selfcond = x_reconst; L181–183: final call with x_selfcond, then argmax), `lib/models.py` L137, L175; release "Plaid 1B Weights" (3 parts, 4.74 GB) | ✅ |
| LangFlow weights; expected-embedding SC; SC at readout | HF `Continuous-Rivals-Discrete/langflow-owt` `model.py`: L540–545 softmax → `_embed_tokens` → x_self_cond; L551–558 final forward with x_self_cond, then argmax. Injection concat + zero-init proj (L331, L357). Config: 12 layers, 768 d, L = 1024, GPT-2 tokenizer | ✅ |
| GPT: "LangFlow feeds back a continuous latent prediction" | same code | ❌ It is an expected token embedding |
| DiffusionGemma weights | HF API `google/diffusiongemma-26B-A4B-it`: 25.8B params, ungated | ✅ |
| DiffusionGemma SC form | transformers `modeling_diffusion_gemma.py` L1260–1269: softmax(logits) @ embed_tokens × embed_scale → gated MLP + add + norm (L805–817) | ✅ Expected embedding (same family as Lafa) |
| DiffusionGemma SC at readout | `generation_diffusion_gemma.py`: output = argmax of the last denoising step's logits (L793), computed with SC (`_denoising_step`); feedback = temperature-processed logits | ✅ |
| DiffusionGemma uniform-state canvas; entropy-bound acceptance; re-noising; 256-token canvas | same files (L344–449, L1556–1561); config canvas_length 256 | ✅ |
| Agent: "SC zeroed on the commit pass" (DiffusionGemma) | not found in the HF code | ⚠️ Unverified; irrelevant to the decomposition |
| Agent: TESS-2 repo `hamishivi/tess2` | HF API | ❌ Does not exist. Correct: `hamishivi/tess2-v0.3` (7.25B, Mistral-based) |
| TESS-2 SC = logits_mean, mixed before weights | HF config.json; `sdlm/models/mixins/modeling_mixin.py` L45–49: input probs and previous-pred probs mixed, then embedded | ✅ Feedback is **fused into the state input** (no separate channel) |
| TESS-2 SC at readout | `sdlm/pipelines/simplex_ddpm.py` L125–205: every step (incl. the last) uses previous_pred; returns the last logits | ✅ |
| Sigma 3B / 8B-instruct weights | HF API: 3.83B / 8.49B, ungated | ✅ |
| Sigma SC = sg(x_θ E) expected embedding; SC at readout | arXiv 2610.02665v2 Alg. 1 L10; Alg. 2 L19 argmax with e_sc. The unconditional CFG branch uses zero SC | ✅ |

## Papers / quotes
| Claim | Status |
|---|---|
| 2610.19628 D.5.2 "declining nearly monotonically from 31.3% to 20.0% … uses it most heavily at early steps, where the noisy state is less informative" (first layer, averaged over heads and instances) | ✅ verbatim |
| 2610.19628 D.5.3 condition swap: "accuracy falls to zero only once k exceeds 35" | ✅ verbatim |
| LOOT 2611.19127: hollow transformers predict "each token from its surrounding context, independently of the input at that position" | ✅ (abstract) |
| TDLM 2610.36940: each unresolved position "attend[s] to its own history of hidden representations" | ✅ (abstract) |
| Gorgun 2512.08486; TimeROME-DLM 2606.12841; Token Commitment 2611.21669; DiffusionGemma TR 2608.00146; TESS 2 2502.13917; Plaid 2305.18619; LangFlow 2604.11748; Sigma 2610.02665 | ✅ titles, authors, dates (arXiv API). Body claims beyond the abstracts not verified |

## Our own data, re-checked for the plan
- **"Lafa's decision is made before the final readout":** E1b (insights §8, n = 100, exploratory): the final step alone carries 0.05 of the N effect under `warp`; steps 24–30 carry 0.28. ✅
- **"Readout access does not by itself determine N":** E1c (insights §7.1, pre-committed rule, n = 300): under `uniform`, SC is present at readout, but cutting N in steps 24–31 has **exactly 0** effect (states clean); the N effect sits at steps 8–15. H-readout was rejected there. ✅
  - **Consequence:** the planned C1 (Lafa SC-off readout) is largely answered already. It is demoted; the strong "readout access determines the route" version is already refuted by existing data.
- **Pod deduplication:** all 2,188 duplicated rows are identical except the timing field. ✅

## Consequences for the plan
1. The working account combines **H-clean** (feedback is read as a substitute for uninformative states, during the commitment window) with **access** (it must be readable in the computation where commitment happens). ELF's commitment happens at an SC-free decode pass, so F. Lafa's happens at noisy late or mid steps with SC readable, so N.
2. **Every external candidate (Plaid, LangFlow, Sigma, DiffusionGemma, TESS-2) has SC present at readout**, and four of the five use expected-embedding feedback. The account predicts **N-dominant, O ≈ 0** for all of them. Their timing should track when states become decodable. So the externals test generality, but none tests the access variable. **Only the toy factorial and an ELF decode-access manipulation test access.**
3. TESS-2 fuses feedback into the state input, so "N" there means reading neighbours' fused inputs. That makes it a weaker test of route separation.
