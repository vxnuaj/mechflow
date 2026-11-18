"""Toy blockwise flow LM with self-conditioning (E7 / gap G-e). Exploratory; seeds fixed; runs on CPU/MPS.

Task: K prompts; each prompt has exactly two valid 4-token answers (modes) over a vocabulary of V tokens, with
all positions differing between the modes. Data = (prompt, one of its two modes, 50/50). Independent per-position
decoding yields cross-mode hybrids; the joint is consistent only if positions coordinate.

Model (Lafa-style): prompt embedding as a prefix token + 4 block positions; bidirectional transformer; time
conditioning; x-prediction via logits (cross-entropy). Rectified flow on token embeddings:
x_t = (1 - t) * noise + t * E[y]. Self-conditioning: x~ = x + W1 x + W2 (softmax(prev) @ E), trained with 2 passes
and SC dropped for the whole batch with probability 0.5 (as in Lafa).

SC variants (training AND inference): 'full' (each position gets its own feedback via the input, which attention
then shares), 'own' (feedback only to the own position, invisible to others' attention: per-position K/V carry no
feedback), 'nb' (only neighbours' feedback is readable: own feedback removed from own query path), 'none'.
Implemented exactly with a two-stream input: stream S (state only) and stream F (state + feedback); attention for
query p reads keys/values of position q from F iff the variant allows p to see q's feedback.
"""
import argparse, json, math, random, time

import torch
import torch.nn as nn
import torch.nn.functional as F

L = 4


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h; s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, xs, xf, xo, src):
        """Three streams [B, T, d]: S (no feedback anywhere), F (feedback everywhere), O (variant output stream).
        S attends S; F attends F; O's query p reads key q from src[p, q] in {0: S, 1: F, 2: O}."""
        B, T, d = xs.shape; hd = d // s.h
        def heads(x):
            q, k, v = s.qkv(s.ln1(x)).split(d, -1)
            f = lambda z: z.view(B, T, s.h, hd).transpose(1, 2)
            return f(q), f(k), f(v)
        (qs, ks, vs), (qf, kf, vf), (qo, ko, vo) = heads(xs), heads(xf), heads(xo)
        a_s = F.scaled_dot_product_attention(qs, ks, vs)
        a_f = F.scaled_dot_product_attention(qf, kf, vf)
        kk = torch.cat([ks, kf, ko], 2); vv = torch.cat([vs, vf, vo], 2)
        m = torch.full((T, 3 * T), float("-inf"), device=xs.device)
        for j in range(3):
            m[:, j * T:(j + 1) * T] = torch.where(src == j, 0.0, float("-inf"))
        a_o = F.scaled_dot_product_attention(qo, kk, vv, attn_mask=m)
        mrg = lambda a: a.transpose(1, 2).reshape(B, T, d)
        out = []
        for x, a_ in ((xs, a_s), (xf, a_f), (xo, a_o)):
            x = x + s.o(mrg(a_)); out.append(x + s.mlp(s.ln2(x)))
        return out


class Toy(nn.Module):
    def __init__(s, V, K, d=128, h=4, n=4):
        super().__init__()
        s.E = nn.Embedding(V, d); s.P = nn.Embedding(K, d); s.pos = nn.Parameter(torch.randn(1 + L, d) * 0.02)
        s.t = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        s.W1 = nn.Linear(d, d, bias=False); s.W2 = nn.Linear(d, d, bias=False)
        nn.init.zeros_(s.W1.weight); nn.init.zeros_(s.W2.weight)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(n)]); s.lnf = nn.LayerNorm(d)

    def forward(s, prompt, x, t, fb=None, variant="full", own_fb=None, nb_fb=None):
        """prompt [B]; x [B, L, d]; t [B]; fb [B, L, d] expected embeddings or None.
        variant sets (own_fb, nb_fb): full (1,1), own (1,0), nb (0,1), none (0,0); explicit flags override
        (used for the O/N decomposition at inference). Returns logits [B, L, V] from stream O."""
        T = 1 + L
        te = s.t(t[:, None, None].float())
        base = torch.cat([s.P(prompt)[:, None], x + s.W1(x)], 1) + s.pos + te
        o_, n_ = {"full": (1, 1), "own": (1, 0), "nb": (0, 1), "none": (0, 0)}[variant]
        if own_fb is not None: o_ = own_fb
        if nb_fb is not None: n_ = nb_fb
        if fb is None: o_, n_ = 0, 0
        xs = base
        xf = base.clone()
        if fb is not None: xf[:, 1:] = xf[:, 1:] + s.W2(fb)
        xo = base.clone()
        if o_: xo[:, 1:] = xo[:, 1:] + s.W2(fb)
        src = torch.zeros(T, T, dtype=torch.long, device=x.device)
        for p in range(T):
            for q in range(T):
                if p == q: src[p, q] = 2
                elif q >= 1 and p >= 1: src[p, q] = 1 if n_ else 0
                else: src[p, q] = 0                    # prompt token never carries feedback
        for b in s.blocks:
            xs, xf, xo = b(xs, xf, xo, src)
        return s.lnf(xo[:, 1:]) @ s.E.weight.T


def make_data(V, K, seed):
    g = random.Random(seed); modes = []
    for _ in range(K):
        a = [g.randrange(V) for _ in range(L)]
        b = [g.choice([v for v in range(V) if v != a[i]]) for i in range(L)]
        modes.append((a, b))
    return modes


def batch(modes, B, dev, g):
    k = torch.randint(0, len(modes), (B,), generator=g)
    which = torch.randint(0, 2, (B,), generator=g)
    y = torch.tensor([modes[i][w] for i, w in zip(k.tolist(), which.tolist())])
    return k.to(dev), y.to(dev)


def train(variant, V, K, steps, dev, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    modes = make_data(V, K, 1234)
    m = Toy(V, K).to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.01)
    sigma = 1.0
    for it in range(steps):
        k, y = batch(modes, 256, dev, g)
        t = torch.rand(y.shape[0], device=dev)
        clean = m.E(y); noise = torch.randn_like(clean) * sigma
        x = (1 - t[:, None, None]) * noise + t[:, None, None] * clean
        use_sc = variant != "none" and random.Random(seed * 100003 + it).random() < 0.5
        fb = None
        if use_sc:
            with torch.no_grad():
                fb = F.softmax(m(k, x, t, None, "none"), -1) @ m.E.weight
        logits = m(k, x, t, fb, variant if use_sc else "none")
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m, modes


@torch.no_grad()
def sample(m, k, variant, steps, dev, seed, sigma=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = k.shape[0]; d = m.E.weight.shape[1]
    x = (torch.randn(B, L, d, generator=g) * sigma).to(dev)
    grid = torch.linspace(0, 1, steps + 1)
    fb = None; lg = None
    for i in range(steps):
        t0, t1 = grid[i].item(), grid[i + 1].item()
        t = torch.full((B,), t0, device=dev)
        lg = m(k, x, t, fb if variant != "none" else None, variant)
        e = F.softmax(lg, -1) @ m.E.weight
        if variant != "none": fb = e
        x = x + (t1 - t0) / max(1e-6, 1 - t0) * (e - x)
    return lg.argmax(-1)


def classify(y, modes, k):
    out = []
    for row, kk in zip(y.tolist(), k.tolist()):
        a, b = modes[kk]
        if row == a or row == b: out.append("valid")
        elif all(row[i] in (a[i], b[i]) for i in range(L)): out.append("hybrid")
        else: out.append("other")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--V", type=int, default=64); ap.add_argument("--K", type=int, default=64); ap.add_argument("--sample_steps", type=int, default=16)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    res = {}
    for variant in ("full", "nb", "own", "none"):
        for seed in range(a.seeds):
            t0 = time.time(); m, modes = train(variant, a.V, a.K, a.steps, dev, seed); m.eval()
            k = torch.arange(a.K, device=dev).repeat(32)
            c = classify(sample(m, k, variant, a.sample_steps, dev, 99 + seed), modes, k)
            c0 = classify(sample(m, k, "none", a.sample_steps, dev, 99 + seed), modes, k) if variant != "none" else c
            r = {x: c.count(x) / len(c) for x in ("valid", "hybrid", "other")}
            r0 = {x: c0.count(x) / len(c0) for x in ("valid", "hybrid", "other")}
            res[f"{variant}/{seed}"] = {"with_sc": r, "sc_off_at_inference": r0}
            print(f"{variant:5s} seed {seed}: SC {r} | SC off {r0} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "results_toy.json"), "w"), indent=1)
