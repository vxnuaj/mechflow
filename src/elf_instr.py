"""2 x 2 x 2 (O x N x F) decomposition of self-conditioning in ELF (gap G-e). Uses the ELF implementation in the user's
hcfrl project READ-ONLY (imported via sys.path at runtime; nothing is vendored here).

ELF self-conditioning: per position, input = [z_i, x_prev_i]; SC off = zeros in place of x_prev (as in training).
Three streams per denoising step, run as one batch of 3B rows through a patched Attention:
  S: every position gets zero SC input;  F: every position gets its x_prev;  O: target j gets x_prev_j iff O = 1.
  S attends S; F attends F; O-row queries read key q from O if q is the query itself, a prefix token or the target,
  and otherwise from F (N = 1) or S (N = 0).
Trajectory: neighbours take (v, x) from F (F = 1) or S (F = 0); target j takes (v, x) from O.
Identities: (1,1,1) == SC-on sampling; (0,0,0) == SC-off sampling (checked by `check_identities`).
"""
import itertools, os, sys

import torch
import torch.nn.functional as F_

ROUTE = None          # {"P": prefix length, "j": target seq index, "n": 0/1}
_ORIG = None


def load_elf(hcfrl_root, ckpt_dir, L, device):
    sys.path.insert(0, hcfrl_root)
    import hcfrl.elf_path  # noqa: F401
    from transformers import AutoTokenizer
    from configs.config import load_config_from_yaml, apply_config_overrides
    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder
    from utils.checkpoint_utils import load_checkpoint
    from utils.train_utils import TrainState, get_optimizer
    y = open(os.path.join(ckpt_dir, "config.yml")).read().replace(
        "src/configs/sampling_configs/uncond_sampling_configs.yml",
        os.path.join(hcfrl_root, "hcfrl/elf/configs/sampling_configs/uncond_sampling_configs.yml"))
    tmp = f"/tmp/elf_cfg_{os.getpid()}.yml"; open(tmp, "w").write(y)
    cfg = apply_config_overrides(load_config_from_yaml(tmp), [f"max_length={L}", "use_wandb=false"])
    tok = AutoTokenizer.from_pretrained(cfg.encoder_model_name)
    enc_cfg, _ = get_encoder(cfg.encoder_model_name, torch.float32)
    model = ELF_models[cfg.model](
        text_encoder_dim=enc_cfg.d_model, max_length=cfg.max_length, attn_drop=0.0, proj_drop=0.0,
        num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens, bottleneck_dim=cfg.bottleneck_dim)
    state = TrainState(model=model, optimizer=get_optimizer(model, cfg, lr=1e-4), lr_scheduler=None,
                       ema_params1=TrainState.init_ema(model), step=0, epoch=0,
                       dropout_generator=torch.Generator(device="cpu").manual_seed(0))
    state, _ = load_checkpoint(ckpt_dir, state)
    model = state.model
    if state.ema_params1:
        model.load_state_dict(state.ema_params1)
    model = model.to(device).eval()
    _install()
    return model, cfg, tok


def _install():
    global _ORIG
    from modules.layers import Attention, scaled_dot_product_attention
    if _ORIG is not None:
        return
    _ORIG = Attention.forward

    def forward(self, x, rope_fn, attention_mask=None, deterministic=True):
        if ROUTE is None:
            return _ORIG(self, x, rope_fn, attention_mask=attention_mask, deterministic=deterministic)
        assert attention_mask is None
        B3, N, C = x.shape; B = B3 // 3; hd = self.dim // self.num_heads
        qkv = self.qkv(x).reshape(B3, N, 3, self.num_heads, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope_fn is not None:
            q, k = rope_fn(q), rope_fn(k)
        (qs, qf, qo), (ks, kf, ko), (vs, vf, vo) = q.split(B), k.split(B), v.split(B)
        os_ = F_.scaled_dot_product_attention(qs, ks, vs)
        of_ = F_.scaled_dot_product_attention(qf, kf, vf)
        P, j, n = N - ROUTE["L"], ROUTE["j"], ROUTE["n"]
        src = torch.full((N, N), 1 if n else 0, dtype=torch.long, device=x.device)
        src[:, :P] = 2; src[:, P + j] = 2; src[torch.arange(N), torch.arange(N)] = 2
        allow = torch.cat([src == 0, src == 1, src == 2], dim=1)                       # (N, 3N)
        oo = F_.scaled_dot_product_attention(qo, torch.cat([ks, kf, ko], 2), torch.cat([vs, vf, vo], 2), attn_mask=allow)
        out = torch.cat([os_, of_, oo], 0).permute(0, 2, 1, 3).reshape(B3, N, C)
        return self.proj(out)
    Attention.forward = forward


def _vx(out, z, t, t_eps):
    from utils.sampling_utils import net_out_to_v_x
    return net_out_to_v_x(out, z, t, t_eps)


@torch.no_grad()
def run_cell(model, cfg, z0, t_steps, j, O, N, F, w=1.0, autocast=True):
    """One trajectory for cell (O, N, F) at target sequence index j. Returns final z."""
    global ROUTE
    B, L, Cd = z0.shape
    dev = z0.device
    z = z0.clone(); xprev = torch.zeros_like(z)
    wt = torch.full((3 * B,), float(w), device=dev)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and dev.type == "cuda")
    with ctx:
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item()
            zero = torch.zeros_like(z)
            o_in = zero.clone()
            if O:
                o_in[:, j] = xprev[:, j]
            inp = torch.cat([torch.cat([z, zero], -1), torch.cat([z, xprev], -1), torch.cat([z, o_in], -1)], 0)
            tb = torch.full((3 * B,), t, device=dev)
            ROUTE = {"L": L, "j": j, "n": N}
            try:
                out = model(inp, tb, deterministic=True, self_cond_cfg_scale=wt)
            finally:
                ROUTE = None
            v, x = _vx(out, torch.cat([z, z, z], 0), tb, cfg.t_eps)
            vS, vF, vO = v.split(B); xS, xF, xO = x.split(B)
            vv = (vF if F else vS).clone(); xx = (xF if F else xS).clone()
            vv[:, j] = vO[:, j]; xx[:, j] = xO[:, j]
            z = z + (tn - t) * vv
            xprev = xx
    return z


@torch.no_grad()
def run_plain(model, cfg, z0, t_steps, sc_on, w=1.0, autocast=True):
    """Reference ELF sampling loop (no routing): SC on = previous x-prediction, off = zeros."""
    B = z0.shape[0]; dev = z0.device; z = z0.clone(); xp = torch.zeros_like(z)
    wt = torch.full((B,), float(w), device=dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and dev.type == "cuda"):
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item()
            tb = torch.full((B,), t, device=dev)
            out = model(torch.cat([z, xp], -1), tb, deterministic=True, self_cond_cfg_scale=wt)
            v, x = _vx(out, z, tb, cfg.t_eps)
            z = z + (tn - t) * v
            xp = x if sc_on else torch.zeros_like(x)
    return z


def decode(model, cfg, z, t_final, w=1.0):
    from utils.generation_utils import _dlm_decode_batch
    return _dlm_decode_batch(z, model, t_final, cfg, w)


# ---- Phase 13 step 3 (decoder access): truncated trajectories + routed decode pass with optional readable SC ----
@torch.no_grad()
def run_cell_x(model, cfg, z0, t_steps, j, O, N, F, w=1.0, autocast=True):
    """As run_cell, but returns (z, xprev) after len(t_steps) - 1 steps (pass a prefix of t_steps to truncate)."""
    global ROUTE
    B, L, Cd = z0.shape; dev = z0.device
    z = z0.clone(); xprev = torch.zeros_like(z); wt = torch.full((3 * B,), float(w), device=dev)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and dev.type == "cuda")
    with ctx:
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item(); zero = torch.zeros_like(z); o_in = zero.clone()
            if O: o_in[:, j] = xprev[:, j]
            inp = torch.cat([torch.cat([z, zero], -1), torch.cat([z, xprev], -1), torch.cat([z, o_in], -1)], 0)
            tb = torch.full((3 * B,), t, device=dev); ROUTE = {"L": L, "j": j, "n": N}
            try: out = model(inp, tb, deterministic=True, self_cond_cfg_scale=wt)
            finally: ROUTE = None
            v, x = _vx(out, torch.cat([z, z, z], 0), tb, cfg.t_eps)
            vS, vF, vO = v.split(B); xS, xF, xO = x.split(B)
            vv = (vF if F else vS).clone(); xx = (xF if F else xS).clone(); vv[:, j] = vO[:, j]; xx[:, j] = xO[:, j]
            z = z + (tn - t) * vv; xprev = xx
    return z, xprev


@torch.no_grad()
def run_plain_x(model, cfg, z0, t_steps, sc_on, w=1.0, autocast=True):
    B = z0.shape[0]; dev = z0.device; z = z0.clone(); xp = torch.zeros_like(z); wt = torch.full((B,), float(w), device=dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and dev.type == "cuda"):
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item(); tb = torch.full((B,), t, device=dev)
            out = model(torch.cat([z, xp], -1), tb, deterministic=True, self_cond_cfg_scale=wt)
            v, x = _vx(out, z, tb, cfg.t_eps); z = z + (tn - t) * v; xp = x if sc_on else torch.zeros_like(x)
    return z, xp


def _dec_logits(model, inp, w):
    B = inp.shape[0]; dev = inp.device
    tf = torch.ones((B,), device=dev, dtype=inp.dtype); sc = torch.full((B,), float(w), device=dev, dtype=inp.dtype)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        _, lg = model(inp, tf, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=torch.ones((B,), device=dev, dtype=inp.dtype))
    return lg


@torch.no_grad()
def decode_plain_x(model, z, xp, readable, w=1.0):
    """Decode pass; SC input = xp if readable else zeros (ELF default)."""
    return _dec_logits(model, torch.cat([z, xp if readable else torch.zeros_like(z)], -1), w).argmax(-1)


@torch.no_grad()
def decode_cell(model, z, xprev, j, O, N, F, readable, w=1.0):
    """Routed decode pass. S: zero SC; F: SC = xprev if readable; O: target j gets xprev_j iff O and readable.
    Tokens: neighbours from F (F = 1) or S; target from O. If not readable, S = F = O inputs (SC is zero everywhere)."""
    global ROUTE
    B, L, _ = z.shape; zero = torch.zeros_like(z); fb = xprev if readable else zero; o_in = zero.clone()
    if O and readable: o_in[:, j] = xprev[:, j]
    inp = torch.cat([torch.cat([z, zero], -1), torch.cat([z, fb], -1), torch.cat([z, o_in], -1)], 0)
    ROUTE = {"L": L, "j": j, "n": N}
    try: lg = _dec_logits(model, inp, w)
    finally: ROUTE = None
    yS, yF, yO = lg.argmax(-1).split(B); y = (yF if F else yS).clone(); y[:, j] = yO[:, j]
    return y
