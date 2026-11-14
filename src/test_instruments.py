"""Phase 6 instrument validation (runs on MPS/CUDA with Lafa-0.5B).

T1 identity:     our sampler == model._euler_denoise_block (logits, tokens), several configs
T2 determinism:  two runs bitwise identical
T3 off-hook:     sc_fn(z) at all positions == reference with self_cond=False
T4 K/V identity: per-query K/V selection with alt == same input reproduces clean logits (both branches)
T5 knockout:     single-position 'off' changes only the intended position's input row
T6 positive ctl: a planted feedback signal at position i changes j's prediction; cutting j's
                 (and all others') reads of i's K/V removes the effect at j (floor = cut residual)
Writes a JSON report with run_meta.
"""
import json
import sys

import torch
import torch.nn.functional as F

from load import load_model, chat_prompt_ids
from lafa_instr import Sampler, BlockCfg, sc_apply, run_meta

PROMPTS = [
    "What is 17 + 25? Answer briefly.",
    "Choose either New York or San Francisco. Answer with just the name.",
    "Pick one of these at random: Albert Einstein or Isaac Newton. Reply with only your choice.",
    "Write one sentence about the ocean.",
    "Name a primary color.",
    "List two prime numbers.",
]
CFGS = [BlockCfg(32, 8.0, 0.1, True), BlockCfg(32, 1.0, 0.1, True), BlockCfg(32, 8.0, 4.0, True),
        BlockCfg(32, 8.0, 0.1, False), BlockCfg(4, 8.0, 0.1, True), BlockCfg(32, 1.0, 4.0, True)]


@torch.no_grad()
def reference(model, cache, start, cfg):
    m = model
    sn, ss = m.noise_scale_mult, m.enable_self_cond
    oa = getattr(m.backbone.config, "_attn_implementation", None)
    m.noise_scale_mult, m.enable_self_cond = cfg.noise, cfg.self_cond and ss
    m.backbone.config._attn_implementation = "eager"
    try:
        torch.manual_seed(cfg.seed)
        dev = m.embed_tokens.weight.device
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode="warp", t_max=0.2)
        pos = torch.arange(start, start + m.bd_size, device=dev).unsqueeze(0)
        cl = cache.get_seq_length()
        mask = torch.zeros(1, 1, m.bd_size, cl + m.bd_size, dtype=m.embed_tokens.weight.dtype, device=dev)
        lg, nf = m._euler_denoise_block(cache, pos, grid, mask, cfg_scale=cfg.cfg_scale)
        return lg[0].float().cpu(), nf
    finally:
        m.noise_scale_mult, m.enable_self_cond = sn, ss
        m.backbone.config._attn_implementation = oa


def main(out="results/instrument_validation.json"):
    tok, model = load_model()
    S = Sampler(model)
    rep = {"meta": run_meta(), "T1": [], "T2": [], "T3": [], "T4": [], "T5": [], "T6": []}

    for p in PROMPTS:
        ids = chat_prompt_ids(tok, p)
        cache, L = S.prefill(ids)
        # T1 identity
        for cfg in CFGS:
            ref, nf_ref = reference(model, cache, L, cfg)
            ours = S.run_block(cache, L, cfg)
            d = (ref - ours.logits).abs().max().item()
            rep["T1"].append({"prompt": p, "cfg": cfg.__dict__, "max_abs_logit_diff": d,
                              "tokens_equal": ref.argmax(-1).tolist() == ours.tokens,
                              "n_fwd_equal": nf_ref == ours.n_fwd})
        # T2 determinism
        a = S.run_block(cache, L, CFGS[0]); b = S.run_block(cache, L, CFGS[0])
        rep["T2"].append({"prompt": p, "max_abs_diff": (a.logits - b.logits).abs().max().item()})
        # T3 off-hook (block-level off via hook == self_cond=False reference)
        for g in (8.0, 1.0):
            cfg_on = BlockCfg(32, g, 0.1, True)
            ours = S.run_block(cache, L, cfg_on, sc_fn=lambda s, br, z, e, c: z)
            ref, _ = reference(model, cache, L, BlockCfg(32, g, 0.1, False))
            rep["T3"].append({"prompt": p, "g": g, "max_abs_logit_diff": (ref - ours.logits).abs().max().item()})
        # T4 K/V-selection identity (alt == same input, random src), both branches (g=8)
        gen = torch.Generator().manual_seed(0)
        src = torch.rand(4, 4, generator=gen) < 0.5
        src.fill_diagonal_(False)
        clean = S.run_block(cache, L, CFGS[0])
        kv = S.run_block(cache, L, CFGS[0], kv_fn=lambda s, br, e_in, c: (e_in.clone(), src))
        rep["T4"].append({"prompt": p, "max_abs_logit_diff": (clean.logits - kv.logits).abs().max().item(),
                          "tokens_equal": clean.tokens == kv.tokens})
        # T5 single-position knockout touches only its row (candidate a)
        rows_ok = []

        def sc_check(s, br, z, e, c):
            full = model._apply_self_cond(z, e) if e is not None else z
            mixed = sc_apply(model, z, e, off_positions=[1], candidate="a")
            if e is not None:
                others = [0, 2, 3]
                rows_ok.append(bool(torch.equal(full[:, others], mixed[:, others])) and
                               bool(torch.equal(mixed[:, 1], z[:, 1])))
            return mixed
        S.run_block(cache, L, CFGS[1], sc_fn=sc_check)
        rep["T5"].append({"prompt": p, "all_rows_ok": all(rows_ok), "n_checked": len(rows_ok)})

        # T6 positive control at g=1, noise 0.1, inject at step 12, i=1 -> observe j=3 (same step)
        STEP, I, J = 12, 1, 3
        cfg = BlockCfg(32, 1.0, 0.1, True)
        rare = tok.convert_tokens_to_ids("Ġzebra") if "Ġzebra" in tok.get_vocab() else 1000
        E = model.embed_tokens.weight

        def inject(s, br, z, e, c):
            if s == STEP and e is not None:
                e = e.clone(); e[:, I, :] = E[rare]
            return model._apply_self_cond(z, e) if e is not None else z

        def cut(s, br, e_in, c):  # everyone except I reads I's K/V from the clean (non-injected) input
            if s != STEP:
                return None
            e_alt = e_in.clone()
            z, e = c["_z"], c["_e"]
            e_alt = model._apply_self_cond(z, e) if e is not None else z
            srcm = torch.zeros(4, 4, dtype=torch.bool); srcm[:, I] = True; srcm[I, I] = False
            return (e_alt, srcm)

        def inject_and_stash(s, br, z, e, c):
            c["_z"], c["_e"] = z, e
            return inject(s, br, z, e, c)

        def probs_at_step(res_ctx):
            return res_ctx["probs_hist"][STEP]["c"]
        c0 = {"keep_probs": True}; S.run_block(cache, L, cfg, ctx=c0)
        c1 = {"keep_probs": True}; S.run_block(cache, L, cfg, sc_fn=inject, ctx=c1)
        c2 = {"keep_probs": True}; S.run_block(cache, L, cfg, sc_fn=inject_and_stash, kv_fn=cut, ctx=c2)
        p0, p1, p2 = probs_at_step(c0), probs_at_step(c1), probs_at_step(c2)

        def kl(a, b):
            return (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(-1)
        rep["T6"].append({"prompt": p,
                          "KL_j_injected": kl(p1[J], p0[J]).item(), "KL_j_cut": kl(p2[J], p0[J]).item(),
                          "KL_i_injected": kl(p1[I], p0[I]).item(), "KL_i_cut": kl(p2[I], p0[I]).item()})
        print("done", p[:40], flush=True)

    # summary
    summ = {
        "T1_max_logit_diff": max(r["max_abs_logit_diff"] for r in rep["T1"]),
        "T1_all_tokens_equal": all(r["tokens_equal"] for r in rep["T1"]),
        "T1_all_nfwd_equal": all(r["n_fwd_equal"] for r in rep["T1"]),
        "T2_max_diff": max(r["max_abs_diff"] for r in rep["T2"]),
        "T3_max_logit_diff": max(r["max_abs_logit_diff"] for r in rep["T3"]),
        "T4_max_logit_diff": max(r["max_abs_logit_diff"] for r in rep["T4"]),
        "T4_all_tokens_equal": all(r["tokens_equal"] for r in rep["T4"]),
        "T5_all_ok": all(r["all_rows_ok"] for r in rep["T5"]),
        "T6_min_ratio_injected_over_cut": min(r["KL_j_injected"] / max(r["KL_j_cut"], 1e-12) for r in rep["T6"]),
        "T6_i_effect_kept_under_cut": all(r["KL_i_cut"] > 0.5 * r["KL_i_injected"] for r in rep["T6"]),
    }
    rep["summary"] = summ
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
