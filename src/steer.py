"""E4 steering (exploratory): transplant neighbours' fed-back predictions from a donor run into what target j reads.

MR regime (g = 1, noise 4.0, K = 32), 0.5B. Recipient run R and donor run D share the prompt and differ in seed.
At every step (or a step window), j reads its neighbours' K/V from a pass whose neighbour inputs are
apply_self_cond(z_R[nb], esc_D[nb]): the recipient's own neighbour STATES, the donor's neighbour FEEDBACK.
Everything else (j's own input, the neighbours' own trajectories) is the recipient's.
"""
import torch
import torch.nn.functional as F

from lafa_instr import BlockCfg, PerLayerMasks


@torch.no_grad()
def run_logged(S, cache, block_start, cfg):
    """Normal SC run (reference path, MR) returning tokens and the per-step feedback esc[k] used at step k."""
    m, E, bd, dt, dev = S.m, S.E, S.bd, S.dtype, S.dev
    saved = m.noise_scale_mult; m.noise_scale_mult = float(cfg.noise)
    orig = getattr(m.backbone.config, "_attn_implementation", None); m.backbone.config._attn_implementation = "eager"
    try:
        torch.manual_seed(cfg.seed)
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2)
        K = grid.shape[-1] - 1
        pos = torch.arange(block_start, block_start + bd, device=dev).unsqueeze(0)
        cl = cache.get_seq_length(); mask = torch.zeros(1, 1, bd, cl + bd, dtype=dt, device=dev)
        e_t = torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult
        lg0, _, _ = S._fwd(e_t, m._compute_t_bn(grid[:, 0].to(dt)[None, :]), mask, cache, pos)
        esc = (F.softmax(lg0, -1).to(dt) @ E).unsqueeze(0); log = []
        for step in range(K):
            log.append(esc.clone())
            tf_, to_ = grid[:, step], grid[:, step + 1]; z = e_t
            lg, _, _ = S._fwd(m._apply_self_cond(z, esc), m._compute_t_bn(tf_.to(dt)[None, :]), mask, cache, pos)
            e_den = (F.softmax(lg, -1).to(dt) @ E).unsqueeze(0); esc = e_den.detach(); last = lg
            if step == K - 1: break
            tf, to = tf_[None, :, None].float(), to_[None, :, None].float()
            e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
        return last.argmax(-1).tolist(), log
    finally:
        m.noise_scale_mult = saved; m.backbone.config._attn_implementation = orig


@torch.no_grad()
def run_transplant(S, cache, block_start, cfg, j, donor_log, steps=None):
    """Recipient run (seed cfg.seed) where query j reads neighbours' K/V computed from (recipient states, donor
    feedback) at all layers and heads, at the given steps (None = all). donor_log=None gives the matched baseline
    (j reads the recipient's own neighbours through the same extra-pass code path)."""
    m, E, bd, dt, dev = S.m, S.E, S.bd, S.dtype, S.dev
    saved = m.noise_scale_mult; m.noise_scale_mult = float(cfg.noise)
    orig = getattr(m.backbone.config, "_attn_implementation", None); m.backbone.config._attn_implementation = "eager"
    H = m.backbone.config.num_attention_heads; nb = [p for p in range(bd) if p != j]; neg = float("-inf")
    try:
        torch.manual_seed(cfg.seed)
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2)
        K = grid.shape[-1] - 1
        pos = torch.arange(block_start, block_start + bd, device=dev).unsqueeze(0)
        cl = cache.get_seq_length(); base = torch.zeros(1, 1, bd, cl + bd, dtype=dt, device=dev)
        e_t = torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult
        lg0, _, _ = S._fwd(e_t, m._compute_t_bn(grid[:, 0].to(dt)[None, :]), base, cache, pos)
        esc = (F.softmax(lg0, -1).to(dt) @ E).unsqueeze(0)
        default = torch.zeros(1, 1, bd, cl + 2 * bd, dtype=dt, device=dev); default[..., cl:cl + bd] = neg
        mk = default.expand(1, H, bd, cl + 2 * bd).clone()
        for p in nb:
            mk[0, :, j, cl + p] = 0.0; mk[0, :, j, cl + bd + p] = neg
        masks = {li: mk for li in range(len(m.backbone.model.layers))}
        for step in range(K):
            tf_, to_ = grid[:, step], grid[:, step + 1]; z = e_t
            tb = m._compute_t_bn(tf_.to(dt)[None, :])
            full = m._apply_self_cond(z, esc)
            active = steps is None or step in steps
            fb_src = donor_log[step] if (donor_log is not None and active) else esc
            eB = m._apply_self_cond(z, fb_src)                                  # neighbours: own state, chosen feedback
            lgA, _, _ = S._fwd(full, tb, base, cache, pos)
            m._forward_backbone(eB, tb, base, past_key_values=cache, use_cache=False, position_ids=pos)
            with PerLayerMasks(m, masks):
                lgA2, _, _ = m._forward_backbone(full, tb, default, past_key_values=cache, use_cache=False, position_ids=pos)
            S._trim(cache, cache.get_seq_length() - cl)
            lg = lgA.clone(); lg[j] = lgA2[0].float()[j]
            e_den = (F.softmax(lg, -1).to(dt) @ E).unsqueeze(0); esc = e_den.detach(); last = lg
            if step == K - 1: break
            tf, to = tf_[None, :, None].float(), to_[None, :, None].float()
            e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
        return last.argmax(-1).tolist(), last.cpu()
    finally:
        m.noise_scale_mult = saved; m.backbone.config._attn_implementation = orig
