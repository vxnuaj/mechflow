"""Phase 13 step 4: O/N/F instrumentation for LangFlow (arXiv 2604.11748; HF Continuous-Rivals-Discrete/langflow-owt).
LangFlow: a full-sequence continuous diffusion LM on token embeddings with 12 DDiT blocks, d = 768 and the GPT-2 tokenizer.
SC = the expected embedding of softmax(logits), injected as x + self_cond_proj([x, x_sc]) (zeros when off). Euler-EDM
over a Gumbel-proposal gamma schedule; the final forward pass uses SC, then argmax (verified in model.py; verification_log.md).

The routing mirrors family E: three streams per forward (S: zero SC everywhere; F: SC everywhere; O: SC only at the target
j iff O). S attends S and F attends F; O-row queries read key q from O if q == query or q == j, otherwise from F (N = 1) or
S (N = 0). Neighbours take their estimate from F (F = 1) or S; target j from O. Applied at every step and at the final
forward. No prefix (unconditional generation).
Identities: O1N1F1 = SC-on sampling and O0N0F0 = SC-off sampling (checked by the callers)."""
import os, sys
import einops
import torch
import torch.nn.functional as F_

ROUTE = None
_ORIG = None


FP32 = os.environ.get("MF_FP32", "1") == "1"   # bf16 breaks routed/plain identity (0.92 vs 1.000 in fp32; 2026-11-20)


class _TorchNoAutocast:
    """Stands in for `torch` inside LangFlow's module so its hard-coded bf16 autocast is disabled (fp32 everywhere)."""
    def __getattr__(self, k): return getattr(torch, k)
    def autocast(self, *a, **k): return torch.autocast(device_type="cuda", enabled=False)


def load_langflow(path, device, dtype=torch.float32):
    from transformers import AutoModelForMaskedLM
    if FP32: torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    mod = sys.modules[type(model).__module__]
    if FP32: mod.torch = _TorchNoAutocast()
    _install(mod)
    return model


def _install(mod):
    global _ORIG
    if _ORIG is not None: return
    _ORIG = mod.regular_attention_multi_headed

    def routed(q, k, v):                                          # q, k, v: [B3, s, h, d]
        if ROUTE is None: return _ORIG(q, k, v)
        B3, N, H, D = q.shape; B = B3 // 3
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [B3, h, s, d]
        (qs, qf, qo), (ks, kf, ko), (vs, vf, vo) = q.split(B), k.split(B), v.split(B)
        os_ = F_.scaled_dot_product_attention(qs, ks, vs); of_ = F_.scaled_dot_product_attention(qf, kf, vf)
        j, n = ROUTE["j"], ROUTE["n"]
        src = torch.full((N, N), 1 if n else 0, dtype=torch.long, device=q.device)
        src[:, j] = 2; src[torch.arange(N), torch.arange(N)] = 2
        allow = torch.cat([src == 0, src == 1, src == 2], dim=1)
        oo = F_.scaled_dot_product_attention(qo, torch.cat([ks, kf, ko], 2), torch.cat([vs, vf, vo], 2), attn_mask=allow)
        out = torch.cat([os_, of_, oo], 0).transpose(1, 2)
        return einops.rearrange(out, 'b s h d -> b s (h d)')
    mod.regular_attention_multi_headed = routed


def schedule(model, K, device):
    eps = 1e-5; t = torch.linspace(1.0 - eps, eps, K, device=device)
    with torch.no_grad(): return model.proposal(t)


def _fwd(model, z, g, sc):
    return model(noisy_embeds=z, timesteps=g.expand(z.shape[0]), x_self_cond=sc, return_dict=False).float()


@torch.no_grad()
def sample_plain(model, z0, gam, sc_on, n_steps=None, record=False):
    """LangFlow's generate_samples loop (sc_on = config.self_conditioning). n_steps truncates the loop (None = full).
    Returns (tokens, z, x_pred, rec), where rec holds per-step (z, x_pred) if record."""
    z = z0.clone(); sc = None; rec = []; steps = len(gam) - 1 if n_steps is None else n_steps
    for i in range(steps):
        lg = _fwd(model, z, gam[i], sc); xp = model._embed_tokens(F_.softmax(lg, -1))
        if record: rec.append((z.clone(), xp.clone()))
        if sc_on: sc = xp
        z = model._euler_edm_step(z, xp, gam[i], gam[i + 1])
    g_last = gam[steps] if n_steps is not None else gam[-1]
    y = _fwd(model, z, g_last, sc).argmax(-1)
    return y, z, (sc if sc is not None else torch.zeros_like(z)), rec


@torch.no_grad()
def sample_cell(model, z0, gam, j, O, N, F, n_steps=None):
    global ROUTE
    B = z0.shape[0]; z = z0.clone(); xprev = torch.zeros_like(z); steps = len(gam) - 1 if n_steps is None else n_steps
    def routed_fwd(g):
        global ROUTE
        zero = torch.zeros_like(z); o_in = zero.clone()
        if O: o_in[:, j] = xprev[:, j]
        sc = torch.cat([zero, xprev, o_in], 0); ROUTE = {"j": j, "n": N}
        try: lg = _fwd(model, torch.cat([z, z, z], 0), g, sc)
        finally: ROUTE = None
        return lg.split(B)
    for i in range(steps):
        lS, lF, lO = routed_fwd(gam[i])
        xs, xf, xo = (model._embed_tokens(F_.softmax(l, -1)) for l in (lS, lF, lO))
        xx = (xf if F else xs).clone(); xx[:, j] = xo[:, j]
        z = model._euler_edm_step(z, xx, gam[i], gam[i + 1]); xprev = xx
    g_last = gam[steps] if n_steps is not None else gam[-1]
    lS, lF, lO = routed_fwd(g_last)
    y = (lF if F else lS).argmax(-1).clone(); y[:, j] = lO[:, j].argmax(-1)
    return y
