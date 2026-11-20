# Prompt: red-team a pre-registered experiment before it runs

You are a skeptical reviewer for an interpretability study. You have no other context; everything you need is below. Your job is to **find flaws** in the design and decision rules of one pre-registered experiment *before its data exist*. Do not praise; list problems, ranked by severity. Do not invent facts.

## Background (minimum needed)
- **Self-conditioning (SC):** a diffusion/flow language model denoises all positions of a block in parallel over S steps. Each step it also receives its own previous-step prediction as an extra input ("feedback"); training drops SC 50% of the time.
- **Causal decomposition:** for a target position j, a 2×2×2 intervention routes three things independently:
  - **O:** whether j's own input carries its feedback;
  - **N:** whether j's attention reads the neighbours' keys/values from a pass in which the neighbours carry feedback;
  - **F:** whether the neighbours' own trajectories (their states and outputs) evolve with feedback.

  We report Shapley values of the coherence gain.
- **Findings so far in two real models:**
  - Lafa (blockwise flow LM): N dominates and O ≈ 0.
  - ELF (full-sequence flow over T5 latents, whose final decode pass receives zero SC by design): F dominates and O ≈ 0.
  - In Lafa, under a uniform time grid, the N effect is exactly 0 in late steps, where states are already clean, even though SC is readable at readout. So "readout access alone decides the route" is already refuted there.
- **Account under test:** a position reads its neighbours' feedback as a substitute for their states while those states are uninformative, and only where the feedback is readable in the computation where the token commits.

## The pre-registration (verbatim)
### 2.12 Family T: toy access × form factorial (added 2026-11-19, before any of its data; basis: Phase 13, `verification_log.md`)
**Account under test (H-clean + access):** a position reads its neighbours' fed-back predictions as a substitute for their states while those states are uninformative, and only where the feedback is readable in the computation where the token commits.

**Design (`toy/toy_access.py`):**
- **Task:** the two-mode toy (K = 64 prompts, V = 64, 4-token answers differing at every position).
- **Model:** Lafa-like rectified flow on token embeddings; the prompt is a causal prefix; tokens are read out by an ELF-style decode pass.
- **Training factors:**
  - ACCESS: the decode pass receives the last step's SC (`read`) or zero SC (`zero`, as in ELF);
  - FORM: feedback = expected embedding with CE denoise loss (`exp`), or a regression-head estimate with MSE (`cont`);
  - 3 seeds (0, 1, 2) and 12,000 steps each.
- **Sampling factor:** t_end ∈ {1.0, 0.5}, i.e. the state is clean or still noisy at decode. S = 16 Euler steps; 64 prompts × 32 draws.
- **Decomposition:** 2 × 2 × 2 O/N/F over the whole trajectory and the decode pass (three-stream routing), averaged over targets j = 0..3.
  - Outcome: coh_j = the fraction of neighbours whose token comes from the same mode as y_j.
  - Identities checked in every run: O1N1F1 = SC-on and O0N0F0 = SC-off sampling, ≥ 99.9% token agreement.
- **Smoke test** (300 training steps, not data): identities pass; step-0 stream differences < 1e-4.

**Analysis:** `analysis/toy_access.py` (frozen with this entry). Absolute Shapley O, N, F and total T. D = N − F with a 95% bootstrap CI (10,000, within seed, averaged over seeds) and per-seed signs. A cell is identifiable iff the pooled T 2.5th percentile > 0 and T > 0 in all 3 seeds.

**Predictions:**
- **P1** (read, both t_end): D > 0.
- **P2** (zero, t_end 0.5): D < 0.
- **P3** (zero, t_end 1.0): D > 0.
- **P4:** the form does not flip the sign of D.
- **P5:** |O| ≤ 0.15·T.

A prediction holds when the CI excludes 0 in the predicted direction and all 3 seeds agree; it fails when the CI excludes 0 in the opposite direction.

**Decision:**
- **SUPPORTED:** P1–P3 hold in every identifiable cell (with at least one identifiable cell each), P4 holds, and P5 holds.
- **REFUTED:** any of P1–P3 fails, or P4 fails.
- **NOT IDENTIFIABLE:** no identifiable P2 cell.
- **INCONCLUSIVE:** otherwise.

**Rival readings:** "readout access alone decides" ⇔ P3 fails; "form decides" ⇔ P4 fails.

**Rule check before freezing:** on synthetic data the rule returns SUPPORTED under the account, REFUTED (P3) under readout-alone, REFUTED (P4) under form-decides, and NOT IDENTIFIABLE with no effect.

**Scope:** the toy shows whether the access principle *can* govern the route under matched training. It does not show that Lafa or ELF use it (that is Phase 13 steps 3–5).

**Platform:** CARC CPU (main partition). Trainings may start before the GPT red-team; the decompositions (the data) start after it, and any design change is logged here before data.

## Code (verbatim): model, training, routed sampler
```python
"""Toy access × form factorial (prereg §2.12; Phase 13 step 2).

Task (as in toy_flow.py): K prompts, each with exactly two valid 4-token answers differing at every position;
data = (prompt, one mode at 50/50). Independent per-position decoding gives cross-mode hybrids.

Model: Lafa-like rectified flow on token embeddings (x_t = (1 - t) * noise + t * E[y]), bidirectional within the
4-token block, and the prompt as a causal prefix (it attends only to itself, so it is identical in every routing
stream). Tokens are read out by an ELF-style **decode pass**: a forward pass with a learned decode embedding on the
final state. Two factors are fixed at training time:
  ACCESS = 'read' : the decode pass receives the last step's feedback (SC readable where tokens commit)
           'zero' : the decode pass receives zero feedback (as in ELF, 2605.10938 Alg. 4-5)
  FORM   = 'exp'  : feedback = softmax(logits) @ E (expected token embedding; Lafa); denoise loss = CE
           'cont' : feedback = a regression head's clean-embedding estimate (ELF-like); denoise loss = MSE
The flow update always uses the model's clean estimate (= its feedback). SC is dropped with p = 0.5 (whole batch) in
both the denoise and decode branches. The decode branch is trained on z = t * E[y] + (1 - t) * noise, t ~ U(0, 1).

Sampling: Euler on the grid linspace(0, t_end, S + 1), then the decode pass on the final state. t_end = 1 → the state
is clean before decode; t_end = 0.5 → the state is still noisy at decode.

Decomposition (target j; all j ∈ {0..3} averaged): three streams per forward (S: no feedback anywhere; F: feedback
everywhere; O: the target's stream, whose query at j reads its own key from O, the neighbours from F iff N, and the
prompt from S). O: j's own input carries its feedback. F: the neighbours' estimates (hence their next states, their
feedback and their decode tokens) come from stream F, else from S. Applied at every flow step and at the decode pass.
Identities (checked): O1N1F1 = ordinary SC-on sampling; O0N0F0 = ordinary SC-off sampling, token for token.

Outcome: coh_j = the fraction of neighbours i with (y_i, y_j) drawn from the same mode (both equal to m_i, m_j for a
mode m). This is always defined.

Usage:
  python toy_access.py train --access read --form exp --seed 0 --out runs/
  python toy_access.py decomp --ckpt runs/read_exp_s0.pt --out runs/
  python toy_access.py smoke
"""
import argparse, itertools, json, os, random, time

import torch
import torch.nn as nn
import torch.nn.functional as F

L = 4
V_DEF, K_DEF = 64, 64


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h; s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def _heads(s, x):
        B, T, d = x.shape; hd = d // s.h
        q, k, v = s.qkv(s.ln1(x)).split(d, -1)
        f = lambda z: z.view(B, T, s.h, hd).transpose(1, 2)
        return f(q), f(k), f(v)

    def _out(s, x, a):
        B, T, d = x.shape
        x = x + s.o(a.transpose(1, 2).reshape(B, T, d))
        return x + s.mlp(s.ln2(x))

    def single(s, x, mask):
        q, k, v = s._heads(x)
        return s._out(x, F.scaled_dot_product_attention(q, k, v, attn_mask=mask))

    def routed(s, xs, xf, xo, mask, mo):
        """S attends S, F attends F (both with `mask`); O attends the concatenated [S, F, O] keys via `mo`."""
        (qs, ks, vs), (qf, kf, vf), (qo, ko, vo) = s._heads(xs), s._heads(xf), s._heads(xo)
        a_s = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask)
        a_f = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask)
        a_o = F.scaled_dot_product_attention(qo, torch.cat([ks, kf, ko], 2), torch.cat([vs, vf, vo], 2), attn_mask=mo)
        return s._out(xs, a_s), s._out(xf, a_f), s._out(xo, a_o)


class Toy(nn.Module):
    def __init__(s, V, K, form, d=128, h=4, n=4):
        super().__init__()
        s.form = form
        s.E = nn.Embedding(V, d); s.P = nn.Embedding(K, d); s.pos = nn.Parameter(torch.randn(1 + L, d) * 0.02)
        s.t = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d)); s.dec = nn.Parameter(torch.zeros(d))
        s.W1 = nn.Linear(d, d, bias=False); s.W2 = nn.Linear(d, d, bias=False)
        nn.init.zeros_(s.W1.weight); nn.init.zeros_(s.W2.weight)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(n)]); s.lnf = nn.LayerNorm(d)
        s.xhead = nn.Linear(d, d)                                  # used only for form = 'cont'
        T = 1 + L; m = torch.zeros(T, T); m[0, 1:] = float("-inf")  # prompt = causal prefix
        s.register_buffer("mask", m)

    def _base(s, prompt, x, t, decode):
        te = s.t(t[:, None, None].float()) + (s.dec if decode else 0)
        return torch.cat([s.P(prompt)[:, None], x + s.W1(x)], 1) + s.pos + te

    def _heads_out(s, h):
        h = s.lnf(h[:, 1:]); logits = h @ s.E.weight.T
        est = F.softmax(logits, -1) @ s.E.weight if s.form == "exp" else s.xhead(h)
        return logits, est

    def forward(s, prompt, x, t, fb=None, decode=False):
        h = s._base(prompt, x, t, decode)
        if fb is not None: h = torch.cat([h[:, :1], h[:, 1:] + s.W2(fb)], 1)
        for b in s.blocks: h = b.single(h, s.mask)
        return s._heads_out(h)

    def routed(s, prompt, x, t, fb, j, O, N, decode=False):
        """Returns (logits_S, est_S), (logits_F, est_F), (logits_O, est_O). x: [B, L, d] (the state of every
        position as seen by the streams; the caller assembles it); fb: [B, L, d] (each position's own feedback)."""
        T = 1 + L; base = s._base(prompt, x, t, decode)
        xs = base; xf = torch.cat([base[:, :1], base[:, 1:] + s.W2(fb)], 1)
        xo = base.clone()
        if O: xo[:, 1 + j] = xo[:, 1 + j] + s.W2(fb[:, j])
        mo = torch.full((T, 3 * T), float("-inf"), device=x.device)
        for p in range(T):
            for q in range(T):
                if p == 0 and q != 0: continue                    # prompt attends only to itself
                if q == 0: src = 0                                # prompt key from S (identical in all streams)
                elif q == p: src = 2
                else: src = 1 if N else 0
                mo[p, src * T + q] = 0.0
        for b in s.blocks: xs, xf, xo = b.routed(xs, xf, xo, s.mask, mo)
        return s._heads_out(xs), s._heads_out(xf), s._heads_out(xo)


def make_modes(V, K, seed=1234):
    g = random.Random(seed); modes = []
    for _ in range(K):
        a = [g.randrange(V) for _ in range(L)]
        b = [g.choice([v for v in range(V) if v != a[i]]) for i in range(L)]
        modes.append((a, b))
    return modes


def train(access, form, seed, steps, V=V_DEF, K=K_DEF, dev="cpu", log=None):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); rr = random.Random(seed * 100003)
    modes = make_modes(V, K)
    mt = torch.tensor(modes)                                     # [K, 2, L]
    m = Toy(V, K, form).to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.01)
    for it in range(steps):
        k = torch.randint(0, K, (256,), generator=g); w = torch.randint(0, 2, (256,), generator=g)
        y = mt[k, w].to(dev); k = k.to(dev)
        clean = m.E(y)
        # denoise branch
        t = torch.rand(256, generator=g).to(dev); e = torch.randn(clean.shape, generator=g).to(dev)
        x = (1 - t[:, None, None]) * e + t[:, None, None] * clean
        fb = None
        if rr.random() < 0.5:
            with torch.no_grad(): fb = m(k, x, t)[1]
        lg, est = m(k, x, t, fb)
        l_den = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)) if form == "exp" else F.mse_loss(est, clean.detach())
        # decode branch (ELF Alg. 4-like): corrupted state, decode embedding, SC readable or zeroed
        t2 = torch.rand(256, generator=g).to(dev); e2 = torch.randn(clean.shape, generator=g).to(dev)
        z = (1 - t2[:, None, None]) * e2 + t2[:, None, None] * clean
        fb2 = None
        if access == "read" and rr.random() < 0.5:
            with torch.no_grad(): fb2 = m(k, z, t2)[1]
        lg2, _ = m(k, z, torch.ones_like(t2), fb2, decode=True)
        l_dec = F.cross_entropy(lg2.reshape(-1, V), y.reshape(-1))
        loss = l_den + l_dec
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (it % 1000 == 0 or it == steps - 1): print(f"it {it} den {l_den.item():.4f} dec {l_dec.item():.4f}", flush=True)
    return m, modes


@torch.no_grad()
def sample_plain(m, k, sc, access, S, t_end, seed, dev="cpu"):
    g = torch.Generator().manual_seed(seed); B = k.shape[0]; d = m.E.weight.shape[1]
    x = torch.randn(B, L, d, generator=g).to(dev); grid = torch.linspace(0, t_end, S + 1); fb = None
    for i in range(S):
        t0, t1 = grid[i].item(), grid[i + 1].item()
        _, est = m(k, x, torch.full((B,), t0, device=dev), fb if sc else None)
        if sc: fb = est
        x = x + (t1 - t0) / max(1e-6, 1 - t0) * (est - x)
    lg, _ = m(k, x, torch.ones(B, device=dev), fb if (sc and access == "read") else None, decode=True)
    return lg.argmax(-1)


@torch.no_grad()
def sample_cell(m, k, j, O, N, Fc, access, S, t_end, seed, dev="cpu"):
    """Routed sampler for target j and cell (O, N, F). Neighbours follow stream F if Fc else S; j follows stream O."""
    g = torch.Generator().manual_seed(seed); B = k.shape[0]; d = m.E.weight.shape[1]
    x = torch.randn(B, L, d, generator=g).to(dev); grid = torch.linspace(0, t_end, S + 1)
    fb = torch.zeros_like(x); nb = [i for i in range(L) if i != j]
    for i in range(S):
        t0, t1 = grid[i].item(), grid[i + 1].item()
        (_, es), (_, ef), (_, eo) = m.routed(k, x, torch.full((B,), t0, device=dev), fb, j, O, N)
        est = (ef if Fc else es).clone(); est[:, j] = eo[:, j]
        if i == 0:                                               # step 0: no feedback yet, every stream equal
            d1 = (es - ef).abs().max().item(); d2 = (es[:, j] - eo[:, j]).abs().max().item()
            assert d1 < 1e-4 and d2 < 1e-4, f"step-0 streams differ: S-F {d1:.2e}, S-O {d2:.2e} (O={O},N={N},F={Fc},j={j})"
        fb = est
        x = x + (t1 - t0) / max(1e-6, 1 - t0) * (est - x)
    fbd = fb if access == "read" else torch.zeros_like(fb)
    (ls, _), (lf, _), (lo, _) = m.routed(k, x, torch.ones(B, device=dev), fbd, j, O, N, decode=True)
    y = (lf if Fc else ls).argmax(-1); y[:, j] = lo[:, j].argmax(-1)
    return y


def coh(y, k, modes, j):
    mt = torch.tensor(modes)[k.cpu()]                            # [B, 2, L]
    y = y.cpu(); out = torch.zeros(y.shape[0])
    for i in range(L):
        if i == j: continue
        same = ((y[:, j] == mt[:, 0, j]) & (y[:, i] == mt[:, 0, i])) | ((y[:, j] == mt[:, 1, j]) & (y[:, i] == mt[:, 1, i]))
        out += same.float()
    return out / (L - 1)


def decomp(m, modes, access, S, t_ends, reps, seed, dev="cpu"):
    K = len(modes); k = torch.arange(K, device=dev).repeat(reps); res = {}; ident = {}
    for t_end in t_ends:
        on = sample_plain(m, k, True, access, S, t_end, seed, dev); off = sample_plain(m, k, False, access, S, t_end, seed, dev)
        cells = {}
        for j in range(L):
            for O, N, Fc in itertools.product((0, 1), repeat=3):
                y = sample_cell(m, k, j, O, N, Fc, access, S, t_end, seed, dev)
                if (O, N, Fc) in ((1, 1, 1), (0, 0, 0)):          # identities (float-level: allow rare argmax ties)
                    ref = on if (O, N, Fc) == (1, 1, 1) else off; agree = (y == ref).float().mean().item()
                    ident.setdefault(str(t_end), []).append(agree)
                    assert agree >= 0.999, f"identity O{O}N{N}F{Fc} agreement {agree}"
                cells.setdefault(f"O{O}N{N}F{Fc}", []).append(coh(y, k, modes, j).tolist())
        valid = lambda yy: sum(1 for r, kk in zip(yy.tolist(), k.tolist()) if r in (modes[kk][0], modes[kk][1])) / len(k)
        res[str(t_end)] = {"cells": cells, "valid_on": valid(on), "valid_off": valid(off), "identity_agreement_min": min(ident[str(t_end)])}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "decomp", "smoke"])
    ap.add_argument("--access", default="read"); ap.add_argument("--form", default="exp"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--S", type=int, default=16); ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--ckpt"); ap.add_argument("--out", default="runs")
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))); os.makedirs(a.out, exist_ok=True)
    if a.cmd == "train":
        t0 = time.time(); m, modes = train(a.access, a.form, a.seed, a.steps, log=True)
        p = f"{a.out}/{a.access}_{a.form}_s{a.seed}.pt"
        torch.save({"state": m.state_dict(), "access": a.access, "form": a.form, "seed": a.seed, "steps": a.steps}, p)
        print(f"saved {p} ({time.time()-t0:.0f}s)")
    elif a.cmd == "decomp":
        ck = torch.load(a.ckpt); m = Toy(V_DEF, K_DEF, ck["form"]); m.load_state_dict(ck["state"]); m.eval()
        t0 = time.time(); r = decomp(m, make_modes(V_DEF, K_DEF), ck["access"], a.S, [1.0, 0.5], a.reps, 99 + ck["seed"])
        r["meta"] = {k: ck[k] for k in ("access", "form", "seed", "steps")} | {"S": a.S, "reps": a.reps}
        json.dump(r, open(a.ckpt.replace(".pt", "_decomp.json"), "w")); print(f"decomp done ({time.time()-t0:.0f}s)")
    else:
        for acc, form in itertools.product(("read", "zero"), ("exp", "cont")):
            t0 = time.time(); m, modes = train(acc, form, 0, 300); m.eval(); t1 = time.time()
            r = decomp(m, modes, acc, 4, [1.0, 0.5], 2, 7); t2 = time.time()
            print(f"smoke {acc}/{form}: identities OK; train 300 it {t1-t0:.1f}s; decomp(S=4, reps=2) {t2-t1:.1f}s; valid_on {r['1.0']['valid_on']:.2f}", flush=True)
        print("SMOKE PASS")

```

## Frozen analysis (verbatim)
```python
"""Prereg §2.12 (toy access × form factorial; written before its data).
Usage: python analysis/toy_access.py <dir with {access}_{form}_s{seed}_decomp.json>

Unit: sample (prompt × draw); per sample, outcome = mean over targets j of coh_j. Per (access, form, t_end, seed):
the 8 cell means → absolute Shapley values O, N, F and the total T = cell111 − cell000.
Pooled over seeds: bootstrap over samples within each seed (10,000 resamples, seed 20261119), averaging the seeds.

Conditions: C = (access, t_end) ∈ {read, zero} × {1.0, 0.5}; form ∈ {exp, cont}.
Feasibility: a (C, form) cell is IDENTIFIABLE iff the pooled T 2.5th percentile > 0 and T > 0 in all 3 seeds.
Contrast D = N − F (absolute), pooled with a 95% CI; per-seed signs.

Predictions of the account (H-clean + access), for identifiable cells:
  P1  access=read, both t_end:          D > 0 (N dominates)
  P2  access=zero, t_end=0.5 (noisy):   D < 0 (F dominates: commitment at the SC-free decode)
  P3  access=zero, t_end=1.0 (clean):   D > 0 (commitment during the flow, where SC is readable)
  P4  form irrelevant:                  within each C, the two forms give the same sign of D
  P5  |O| ≤ 0.15·T in every identifiable cell (pooled point estimate)
A prediction "holds" in a cell when the pooled 95% CI excludes 0 in the predicted direction AND all 3 seeds agree
in sign. It "fails" when the pooled CI excludes 0 in the opposite direction. Otherwise it is undetermined.

Decision:
  SUPPORTED: P1–P3 hold in every identifiable cell (at least one identifiable cell for each of P1, P2, P3),
             P4 holds (no C where the forms' pooled CIs exclude 0 in opposite directions), and P5 holds.
  REFUTED:   any of P1–P3 fails in an identifiable cell, OR P4 fails.
  Otherwise INCONCLUSIVE, and NOT IDENTIFIABLE if no cell is identifiable for P2 (the critical prediction).
Rival readings: "readout access alone decides" ⇔ P3 fails; "form decides" ⇔ P4 fails.
"""
import glob, itertools, json, math, os, sys
import numpy as np

KEYS = list(itertools.product((0, 1), repeat=3))


def shapley(c):
    b = c[(0, 0, 0)]; val = lambda S: c[tuple(int(p in S) for p in "ONF")] - b
    return {p: sum(math.factorial(k) * math.factorial(2 - k) / 6 * (val(set(S) | {p}) - val(set(S)))
                   for k in range(3) for S in itertools.combinations([q for q in "ONF" if q != p], k)) for p in "ONF"}


def per_sample(cells):
    """cells: {'O0N0F0': [[coh per sample] per j], ...} → array [n, 8] of means over j."""
    return np.stack([np.mean(np.array(cells[f"O{o}N{n}F{f}"]), 0) for o, n, f in KEYS], 1)


def stats(X):
    c = {k: X[:, i].mean() for i, k in enumerate(KEYS)}; sh = shapley(c)
    return {"T": c[(1, 1, 1)] - c[(0, 0, 0)], **sh}


def main(d):
    runs = {}
    for f in sorted(glob.glob(f"{d}/*_decomp.json")):
        r = json.load(open(f)); m = r["meta"]
        for te in ("1.0", "0.5"):
            runs.setdefault((m["access"], m["form"], te), {})[m["seed"]] = per_sample(r[te]["cells"])
    rng = np.random.default_rng(20261119); R = 10000; out = {}
    print(f"{'access':6} {'form':5} {'t_end':5} | {'T':>7} {'O':>7} {'N':>7} {'F':>7} | D=N-F [95% CI]  seeds(sign D)  ident")
    for key in sorted(runs):
        seeds = runs[key]; pts = {s: stats(X) for s, X in seeds.items()}
        boots = {"T": np.zeros(R), "D": np.zeros(R), "O": np.zeros(R)}
        for s, X in seeds.items():
            idx = rng.integers(0, len(X), (R, len(X)))
            M = X[idx].mean(1)                                   # [R, 8]
            c = {k: M[:, i] for i, k in enumerate(KEYS)}; sh = shapley(c)
            boots["T"] += (c[(1, 1, 1)] - c[(0, 0, 0)]) / len(seeds); boots["D"] += (sh["N"] - sh["F"]) / len(seeds); boots["O"] += sh["O"] / len(seeds)
        P = {k: np.mean([pts[s][k] for s in pts]) for k in ("T", "O", "N", "F")}; D = P["N"] - P["F"]
        lo, hi = np.percentile(boots["D"], [2.5, 97.5]); tlo = np.percentile(boots["T"], 2.5)
        signs = [int(np.sign(pts[s]["N"] - pts[s]["F"])) for s in sorted(pts)]
        ident = tlo > 0 and all(pts[s]["T"] > 0 for s in pts) and len(pts) == 3
        out[key] = dict(P, D=D, lo=lo, hi=hi, signs=signs, ident=ident)
        print(f"{key[0]:6} {key[1]:5} {key[2]:5} | {P['T']:+.3f} {P['O']:+.3f} {P['N']:+.3f} {P['F']:+.3f} | {D:+.3f} [{lo:+.3f},{hi:+.3f}]  {signs}  {ident}")

    def verdict(key, want):           # want: +1 (D > 0) or −1
        r = out.get(key)
        if r is None or not r["ident"]: return "n/a"
        if want > 0 and r["lo"] > 0 and all(s > 0 for s in r["signs"]): return "holds"
        if want < 0 and r["hi"] < 0 and all(s < 0 for s in r["signs"]): return "holds"
        if (want > 0 and r["hi"] < 0) or (want < 0 and r["lo"] > 0): return "FAILS"
        return "undetermined"
    preds = {}
    for form in ("exp", "cont"):
        preds[f"P1 read/1.0/{form}"] = verdict(("read", form, "1.0"), +1)
        preds[f"P1 read/0.5/{form}"] = verdict(("read", form, "0.5"), +1)
        preds[f"P2 zero/0.5/{form}"] = verdict(("zero", form, "0.5"), -1)
        preds[f"P3 zero/1.0/{form}"] = verdict(("zero", form, "1.0"), +1)
    p4_fail = []
    for acc, te in itertools.product(("read", "zero"), ("1.0", "0.5")):
        a, b = out.get((acc, "exp", te)), out.get((acc, "cont", te))
        if a and b and a["ident"] and b["ident"] and ((a["lo"] > 0 and b["hi"] < 0) or (a["hi"] < 0 and b["lo"] > 0)): p4_fail.append(f"{acc}/{te}")
    p5_fail = [f"{k}" for k, r in out.items() if r["ident"] and abs(r["O"]) > 0.15 * r["T"]]
    for k, v in preds.items(): print(f"  {k}: {v}")
    print(f"  P4 (form irrelevant): {'FAILS at ' + ', '.join(p4_fail) if p4_fail else 'holds'}")
    print(f"  P5 (|O| ≤ 0.15 T): {'fails at ' + ', '.join(p5_fail) if p5_fail else 'holds'}")
    fam = lambda tag: [v for k, v in preds.items() if k.startswith(tag)]
    if all(v == "n/a" for v in fam("P2")): dec = "NOT IDENTIFIABLE (no identifiable cell for P2)"
    elif any(v == "FAILS" for v in preds.values()) or p4_fail: dec = "REFUTED"
    elif all(any(v == "holds" for v in fam(t)) and all(v in ("holds", "n/a") for v in fam(t)) for t in ("P1", "P2", "P3")) and not p5_fail: dec = "SUPPORTED"
    else: dec = "INCONCLUSIVE"
    rivals = []
    if not p4_fail and any(v == "FAILS" for k, v in preds.items() if k.startswith("P3")): rivals.append("readout-access-alone rival favoured (P3 failed)")
    if p4_fail: rivals.append("form rival favoured (P4 failed)")
    print("TOY ACCESS DECISION:", dec, ("; " + "; ".join(rivals)) if rivals else "")


if __name__ == "__main__":
    main(sys.argv[1])

```

## Your tasks
1. **Construct validity.** Does each factor manipulate what it claims?
   - Does ACCESS = zero really make the decode pass the place where tokens commit when t_end = 0.5?
   - Is t_end = 1.0 really "commitment during the flow"? At t_end = 1 the last Euler step sets x = est exactly; think about it.
   - Does FORM vary only the feedback representation, or also the training objective (CE vs MSE) and hence everything else?
2. **Routing validity.** Check the three-stream routed sampler for errors:
   - O/N/F semantics at the flow steps and the decode pass;
   - how the prompt is handled;
   - whether the feedback used at step i+1 is each position's own estimate from the correct stream;
   - whether the identities really hold.
3. **Outcome measure.** Is coh_j (the fraction of neighbours from the same mode as y_j) appropriate? Can it reward degenerate behaviour (e.g. collapsing to one mode, or "other" tokens)?
4. **Decision rule.** For each scenario below, say what the frozen rule returns and whether that is the right verdict:
   - (a) the account is true;
   - (b) "readout access alone decides";
   - (c) "form decides";
   - (d) SC has no benefit in some cells;
   - (e) N and F are both large and redundant (interaction-heavy);
   - (f) the effects differ across seeds.

   Find any scenario where the rule gives a misleading verdict.
5. **Statistics.** Is a bootstrap over samples within seed, averaged over 3 seeds, adequate? Are 3 seeds enough? Is "all 3 seeds agree in sign" too strict or too lax?
6. **Missing controls / rivals.** What alternative explanation could produce P1–P3 without the account being true? What extra condition would rule it out?
7. **Concrete fixes.** For each serious problem, give a specific change (code or rule) that can be made **before** the data exist.

**Output:** a ranked list of problems (severity: critical / major / minor), each with the evidence (quote a line) and a fix. Then: "Would you trust a SUPPORTED verdict from this design? Why or why not?"
