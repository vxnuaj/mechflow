"""Phase 14.2 message-content probe: data collection (exploration docs only; prereg §2.17 is written before analysis).
On one normal SC-on trajectory (MR regime; `PROBE_GRID` ∈ {warp, uniform}), at every step it runs two passes on the same
states:
  pass A: SC at all positions (target j reads its neighbours *with* their feedback); the trajectory follows A;
  pass B: SC only at j (j keeps its own feedback; the neighbours have none), so this is the N-cut on the same states.
It captures the target's residual state at layers PROBE_LAYERS in both passes, for j ∈ {2, 3}, plus the final tokens.
A copy of lafa_instr.run_block_two_pass (frozen code untouched) with capture added; fbA = all, fbB = {j}, take = all A
reproduces normal SC sampling (the identity is checked against S.run_block on every doc).
Saves results/<OUT>/h/<task_id>.npz: hA, hB [2 (j), K, n_layers, d] float16; y [4]; layers; grid."""
import os, random
import numpy as np
import torch
import torch.nn.functional as F
import load
from lafa_instr import BlockCfg, run_block_headcut

LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "3,8,15,19").split(",")]
GRID = os.environ.get("PROBE_GRID", "warp"); OUTDIR = os.environ.get("PROBE_OUT", "results/probe")
CFG = BlockCfg(32, 1.0, 4.0, True, seed=1, grid_mode=GRID)


@torch.no_grad()
def _two_pass_capture(S, cache, L, cfg, j):
    m, E, bd, dt, dev = S.m, S.E, S.bd, S.dtype, S.dev
    saved_noise = m.noise_scale_mult; m.noise_scale_mult = float(cfg.noise)
    orig = getattr(m.backbone.config, "_attn_implementation", None); m.backbone.config._attn_implementation = "eager"
    cap = {}; layers = m.backbone.model.layers; hooks = []
    for li in LAYERS:
        def h(mod, inp, out, li=li):
            if cap.get("on"): cap.setdefault(li, []).append((out[0] if isinstance(out, tuple) else out)[0, j].detach().float().cpu())
        hooks.append(layers[li].register_forward_hook(h))
    try:
        torch.manual_seed(cfg.seed)
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2); K = grid.shape[-1] - 1
        if grid.dim() == 1: grid = grid[None].expand(bd, -1)          # uniform grids: same times for every position (as lafa_instr L367)
        pos_ids = torch.arange(L, L + bd, device=dev).unsqueeze(0); cache_len = cache.get_seq_length()
        base_mask = torch.zeros(1, 1, bd, cache_len + bd, dtype=dt, device=dev)
        e_t = torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult
        tb0 = m._compute_t_bn(grid[:, 0].to(dt)[None, :]); lg0, _, _ = S._fwd(e_t, tb0, base_mask, cache, pos_ids)
        esc = (F.softmax(lg0, -1).to(dt) @ E).unsqueeze(0); last = None
        for step in range(K):
            t_fwd, t_out = grid[:, step], grid[:, step + 1]; z = e_t; tb = m._compute_t_bn(t_fwd.to(dt)[None, :])
            full = m._apply_self_cond(z, esc); outs = {}
            for name, idx in (("A", list(range(bd))), ("B", [j])):
                e_in = z.clone(); e_in[:, idx, :] = full[:, idx, :]
                cap["on"] = True; lg, _, _ = S._fwd(e_in, tb, base_mask, cache, pos_ids); cap["on"] = False
                outs[name] = lg
            lg_mix = outs["A"]; probs = F.softmax(lg_mix, -1).to(dt); e_den = (probs @ E).unsqueeze(0); esc = e_den.detach(); last = lg_mix
            if step == K - 1: break
            tf, to = t_fwd[None, :, None].float(), t_out[None, :, None].float()
            e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
        H = {li: torch.stack(v).view(K, 2, -1) for li, v in cap.items() if li != "on"}            # [K, pass, d]
        hA = torch.stack([H[li][:, 0] for li in LAYERS], 1); hB = torch.stack([H[li][:, 1] for li in LAYERS], 1)
        return last.argmax(-1).tolist(), hA, hB, K
    finally:
        for hk in hooks: hk.remove()
        m.noise_scale_mult = saved_noise; m.backbone.config._attn_implementation = orig


@torch.no_grad()
def probe_task(task, ctx):
    tok, S = ctx["tok"], ctx["sampler"]
    p_ids = load.chat_prompt_ids(tok, task["user"]); r_ids = tok(task["response"], add_special_tokens=False).input_ids
    rng = random.Random(task["task_id"]); cut = 4 * rng.randint(0, max(0, (len(r_ids) - 8) // 4))
    prefix = p_ids + r_ids[:cut]; cache, L = S.prefill(prefix)
    # identity reference: S.run_block mirrors only the warp grid; for other grids use the validated two-pass primitive
    # (the head-cut sampler with no cuts = normal SC sampling; the path E1c used under `uniform`)
    ref = S.run_block(cache, L, CFG).tokens if GRID == "warp" else run_block_headcut(S, cache, L, CFG, 2, {})[0]
    hAs, hBs, ys = [], [], []
    for j in (2, 3):
        y, hA, hB, K = _two_pass_capture(S, cache, L, CFG, j); ys.append(y); hAs.append(hA); hBs.append(hB)
    os.makedirs(f"{OUTDIR}/h", exist_ok=True)
    np.savez_compressed(f"{OUTDIR}/h/{task['task_id']}.npz", hA=torch.stack(hAs).half().numpy(), hB=torch.stack(hBs).half().numpy(),
                        y=np.array(ys[0]), layers=np.array(LAYERS), K=K, grid=GRID)
    return {"cut": cut, "identity": all(y == ref for y in ys), "y": ys[0]}
