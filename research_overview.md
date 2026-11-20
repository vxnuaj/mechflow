# Research overview: why self-conditioning works in diffusion language models

## Background

Most language models (GPT, Llama) write one token at a time. **Diffusion and flow language models** work differently. They start a block of positions as random noise and refine all of them together over many steps until they become words. This lets them write several tokens in parallel.

Almost all of these models use a trick called **self-conditioning (SC)**. At every refinement step, each position also receives the model's own previous guess as an extra input. It reliably improves quality, but there has been no causal explanation of *why*. Existing accounts treat it as each position correcting itself over time.

Our main model is **Lafa**, a flow language model that writes text in blocks of 4 tokens over 32 refinement steps (0.5B and 7B versions). We also test **ELF** (a flow model from a different group that works in a T5 latent space) and **LangFlow** (an independently built full-sequence model).

## The core idea

Self-conditioning is a **message channel between positions**, not a self-correction mechanism.

Writing tokens in parallel has an old failure mode. If a question can be answered "San Diego" or "New York", each position guessing on its own can produce a mashup like **"San York"**. With self-conditioning, each position can read what its **neighbours** are about to commit to, so the block settles on one consistent answer. A position's *own* previous guess turns out to be nearly useless.

## How we test it

We use **causal interventions**: we change what the model can see during generation and measure the effect.

- **Route decomposition.** For a target position, we separately switch on or off three routes:
  1. **Own:** the position gets its own feedback.
  2. **Neighbour reading:** the position's attention can read its neighbours' feedback.
  3. **Neighbour states:** the neighbours themselves get feedback, so their states improve.

  We run all 8 on/off combinations and measure how much of SC's benefit each route carries.
- **Measuring coherence.** We score whether each word fits with the words around it, using the model's own scorer and an independent judge model (Mistral-7B).
- **Pre-registration.** Before each confirmatory test, we publicly timestamp the exact prediction and the pass/fail rule. Then we run it on fresh data.

## Main results

**1. Without SC, parallel decoding fails in the classic way, and SC fixes it.**
- Without SC, Lafa produces about **7 percentage points more cross-answer mashups**.
- SC clearly improves coherence under both judges.

**2. Neighbours, not selves.**
- At 0.5B, **a position's own feedback contributes about zero**; the whole benefit comes from its neighbours.
- It reaches the position through **two roughly equal routes**: directly reading the neighbours' fed-back guesses, and the neighbours' own states improving because *they* got feedback. This shows up when each word is scored against a fixed context, so that changes in the neighbours' words can't mask it. A pre-registered robustness check confirmed this.
- At 7B, neighbours carry **86%** (76% by the independent judge); own feedback contributes a small **7%**.
- In a toy model trained from scratch, giving positions **only their neighbours' feedback** works as well as or better than full SC (72% vs 53% valid answers). Giving them only their **own** feedback is no better than nothing.

**3. When and why: feedback substitutes for unreadable neighbours.**
- Positions learn what their neighbours are writing through two channels: the neighbours' current noisy states, and their fed-back guesses.
- **The feedback matters exactly while the states are still too noisy to read.** If we change the noise schedule so the states become clean earlier, the effect moves earlier. Once states are clean, cutting the feedback has **exactly zero effect**.
- This was a pre-registered prediction, confirmed on fresh data.

**4. Where: a specific set of attention heads.**
- **26 attention heads**, fewer than 1 in 12, carry about **50%** of the neighbour-reading effect, against **4%** for random sets of the same size (pre-registered, replicated on 3,000 fresh documents).
- Cutting just their feedback reads nearly **triples** garbled or mashup answers at deployed settings (18% → 49%, vs 20% for random heads).

**5. Steering by editing the messages.**
- Swapping **only the neighbours' fed-back guesses** with those from a run that chose the other answer creates mashups on demand: **40% vs 0.5%**.
- This only works in the late steps, which is when the timing result says the channel is in use.

**6. A different architecture uses a different route, for a reason we could pin down.**
- In ELF, own feedback is also useless, but the benefit flows through the neighbours' **improved states** rather than direct reading of their feedback.
- Our first explanation failed its pre-registered test. The literature then showed that **ELF switches the feedback off at its final word-decoding step by design**.
- We then confirmed, on fresh data, that **89%** of the effect arises at that final step, where the model reads its neighbours' feedback-improved states. So the route depends on whether the feedback is visible at the moment a word is decided.

## What didn't work (we report these too)

- **A toy-model test of the access idea** failed as pre-registered. The diagnostic showed the toy commits to its answer at the very first step, before feedback exists, so it never tested the idea. We reported it as refuted rather than re-tuning it.
- **The head-share test** first missed its pre-registered bar by a hair; it passed decisively on a larger fresh replication.
- **An earlier ELF prediction** failed outright, which is what led to the decode-step explanation above.

## What's running now

The goal is to show the explanation **generalizes**, and that the "which route" pattern is predictable rather than a per-model story.

1. **ELF access test.** We fine-tune ELF so its final step *can* see the feedback, against a matched control. The prediction is that neighbour reading then rises, but only when the states are still noisy at that step. This separates our explanation from simpler rivals. A shuffled-feedback control checks that it's the *information* that matters, not just having an extra input.
2. **LangFlow.** We measured first that LangFlow commits its words while its states are still unreadable and its feedback is informative. So our account predicts that neighbour reading dominates there too. A rival account ("this only works in small blocks like Lafa's") predicts it won't.
3. **A robustness re-check** of the main Lafa result, scoring each word against a fixed context so the neighbours' words can't shift the score.

**Compute:** roughly **30 to 50 GPU-hours on A40/A100-class GPUs**, in many short jobs. All the code is written and tested, but the university cluster's queue is the bottleneck.
