"""
CPU validation of the Plaid port:
  (a) strict state_dict load of all four modules; config sanity
  unit checks of each replaced kernel against a reference written from the
      original code (apex manual_rms_norm, lib.rotary TorchScript fallback and
      the flash rotary formula, flash varlen attention semantics, FusedMLP
      fallback path)
  tiny real-weight forward at t=0 (fp32 and CPU-bf16-autocast).
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plaid_torch as P  # noqa: E402
from elbo_torch import cross_entropy  # noqa: E402


def ref_manual_rms_norm(input, normalized_shape, weight, eps):
    # verbatim apex.normalization.fused_layer_norm.manual_rms_norm
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    if weight is None:
        return input
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)
    return weight * input


def ref_rotary_torchscript(qkv, cos, sin):
    return (qkv * cos) + (P.rotate_half(qkv) * sin)


def ref_rotary_flash(qkv, cos_full, sin_full):
    # flash_attn apply_rotary_emb_qkv_ (non-interleaved): cos,sin [s, d/2]
    cos = cos_full[0, :, 0, 0, :cos_full.shape[-1] // 2].float()
    sin = sin_full[0, :, 0, 0, :sin_full.shape[-1] // 2].float()
    out = qkv.float().clone()
    for j in (0, 1):
        x = out[:, :, j]
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        out[:, :, j] = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.to(qkv.dtype)


def ref_flash_varlen(qkv_packed, cu_seqlens, causal=False):
    # qkv_packed: [(b s), 3, h, d]; per-segment exact softmax attention
    outs = []
    for a, b in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        q, k, v = qkv_packed[a:b].float().unbind(1)  # s h d
        sc = torch.einsum('qhd,khd->hqk', q, k) / math.sqrt(q.shape[-1])
        if causal:
            sc = sc.masked_fill(torch.ones_like(sc, dtype=torch.bool).triu(1), -1e30)
        outs.append(torch.einsum('hqk,khd->qhd', sc.softmax(-1), v))
    return torch.cat(outs)


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def unit_checks():
    torch.manual_seed(0)
    res = {}
    # RMSNorm (fp32 path == apex manual/fused fp32)
    x = torch.randn(2, 7, 64)
    m = P.RMSNorm(64)
    m.weight.data = torch.randn(64)
    res['rmsnorm_fp32'] = maxdiff(m(x), ref_manual_rms_norm(x, (64,), m.weight, 1e-5))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = m(x)
    ref = ref_manual_rms_norm(x.bfloat16(), (64,), m.weight.bfloat16(), 1e-5)
    res['rmsnorm_bf16_dtype'] = str(y.dtype)
    res['rmsnorm_bf16_vs_apex_manual'] = maxdiff(y, ref)
    # Rotary
    rot = P.Rotary(64)
    qkv = torch.randn(2, 9, 3, 4, 64)
    cos, sin = rot(qkv)
    res['rotary_vs_torchscript_fp32'] = maxdiff(P.apply_rotary_pos_emb(qkv, cos, sin),
                                                ref_rotary_torchscript(qkv, cos, sin))
    res['rotary_vs_flash_formula_fp32'] = maxdiff(P.apply_rotary_pos_emb(qkv, cos, sin),
                                                  ref_rotary_flash(qkv, cos, sin))
    qb = qkv.bfloat16()
    res['rotary_vs_flash_formula_bf16'] = maxdiff(
        P.apply_rotary_pos_emb(qb, cos.bfloat16(), sin.bfloat16()),
        ref_rotary_flash(qb, cos.bfloat16(), sin.bfloat16()))
    # Attention: SDPA (non-causal) vs flash varlen semantics with equal-length cu_seqlens
    b, s, h, d = 3, 11, 4, 64
    qkv = torch.randn(b, s, 3, h, d)
    q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
    out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b * s, h, d)
    cu = torch.arange(0, (b + 1) * s, s)
    res['sdpa_vs_varlen_ref'] = maxdiff(out, ref_flash_varlen(qkv.reshape(b * s, 3, h, d), cu))
    # MLP vs flash fused_mlp_func fallback path
    mlp = P.FusedMLP(64, 256)
    x = torch.randn(2, 5, 64)
    ref = F.linear(F.gelu(F.linear(x, mlp.fc1.weight), approximate='tanh'), mlp.fc2.weight)
    res['mlp_vs_fused_mlp_fallback'] = maxdiff(mlp(x), ref)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights_path', required=True)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))

    print('== unit checks (max abs diff) ==')
    for k, v in unit_checks().items():
        print(f'  {k}: {v}')

    print('== (a) load ==')
    modules, report = P.load_plaid(args.weights_path, device='cpu')
    ok = all(not m and not u for m, u in report.values())
    print('strict load OK (no missing/unexpected keys):', ok)
    model = modules['model']
    print('model keys (first 12):', list(model.state_dict().keys())[:12])
    print('n params model:', sum(p.numel() for p in model.parameters()))
    print('width_mult:', model.output_linear.width_mult(),
          'output_mult:', model.output_linear.output_mult,
          'residual_scale:', model.blocks[0].residual_scale)
    g0, g1 = modules['gamma_bounds']()
    print('gamma_0, gamma_1:', g0.item(), g1.item())
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
    print('rotary inv_freq matches recomputed:',
          torch.allclose(model.rotary_emb.inv_freq, inv))
    t = torch.tensor([0., 0.25, 0.5, 0.75, 1.], dtype=torch.float64)
    print('normalized gamma(t) at t=0,.25,.5,.75,1:',
          modules['noise_schedule'](t).tolist())
    E = modules['embedding_matrix']()
    print('embedding rows norm (min,max):', E.norm(dim=1).min().item(), E.norm(dim=1).max().item())

    print('== tiny real-weight forward, t=0 (reconstruction) ==')
    tok = P.load_tokenizer()
    text = (" The city council voted on Tuesday to approve a new budget that "
            "increases funding for public schools and road repairs, despite "
            "objections from several members who argued that the plan relies "
            "on overly optimistic revenue projections for the next fiscal year.")
    ids = torch.tensor(tok.encode(text).ids)[None]
    print('n tokens:', ids.shape[1], 'roundtrip:', tok.decode(ids[0].tolist()) == text)
    gamma = g0.reshape(1)
    g = torch.Generator().manual_seed(0)
    x_embed = E[ids]
    a = torch.sigmoid(-gamma).sqrt().float()
    s = torch.sigmoid(gamma).sqrt().float()
    z = a * x_embed + s * torch.randn(x_embed.shape, generator=g)
    for amp in (False, True):
        model.amp = amp
        with torch.no_grad():
            _, xsc = model(z, gamma, E, 1., torch.zeros_like(z))
            logits, _ = model(z, gamma, E, 1., xsc)
        ce = cross_entropy(logits, ids).mean().item()
        acc = (logits.argmax(-1) == ids).float().mean().item()
        print(f'  amp(bf16)={amp}: CE={ce:.4f} nats/token, argmax acc={acc:.3f}, '
              f'finite={torch.isfinite(logits).all().item()}')
        if not amp:
            ref_logits = logits
        else:
            print('  max |logits_bf16 - logits_fp32|:', maxdiff(logits, ref_logits))


if __name__ == '__main__':
    main()
