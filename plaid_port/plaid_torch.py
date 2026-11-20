"""
Pure-PyTorch port of Plaid 1B (Gulrajani & Hashimoto, arXiv 2305.18619;
github.com/igul222/plaid, lib/models.py + lib/rotary.py).

No apex, flash-attn, mup or fused-dense dependencies. Module structure and
parameter names are identical to the original, so the released
model.pt / embedding_matrix.pt / noise_schedule.pt / gamma_bounds.pt load with
strict=True and NO key remapping.

Replacements (original -> port):
  apex.normalization.FusedRMSNorm(dim)        -> RMSNorm (param `weight`, eps=1e-5;
        under autocast input+weight are cast to bf16 exactly like apex's
        _cast_if_autocast_enabled, statistics in fp32, output in bf16)
  flash_attn.ops.fused_dense.FusedMLP(...)    -> FusedMLP: fc1 -> GELU(tanh approx,
        FusedMLP default activation='gelu_approx') -> fc2, no biases
  flash_attn_unpadded_qkvpacked_func          -> F.scaled_dot_product_attention
        (non-causal, default scale 1/sqrt(head_dim) like flash; cu_seqlens dropped)
  flash_attn.layers.rotary.apply_rotary_emb_qkv_ -> apply_rotary_pos_emb below
        (non-interleaved / rotate_half, q and k only; computed in fp32 with cos/sin
        pre-rounded to qkv dtype, then rounded once, like the CUDA kernel)
  mup.MuReadout(dim, vocab)                   -> MuReadout(nn.Linear) with
        output_mult=1.0 and width_mult() = dim / 256 (mup.set_base_shapes base
        model has dim 256), i.e. 8.0 for Plaid 1B. The forward only uses these
        two numbers: x *= output_mult / width_mult().
  hard-coded device='cuda' / global default dtype float64 -> explicit devices
        and explicit float64 where the original relied on
        torch.set_default_dtype(torch.float64).

Mixed precision: the original runs the transformer blocks under
torch.cuda.amp.autocast(bfloat16). In that regime the residual stream becomes
bf16 after the first block's torch.addmm (addmm autocasts to bf16) and stays
bf16. The port keeps exactly the same op sequence (residual_linear via
torch.addmm, torch.add with alpha=residual_scale) inside
torch.autocast(device_type, bf16), so this behaviour is reproduced. Set
`model.amp = False` for a full-fp32 forward.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MUP_BASE_DIM = 256  # base_modules = create_modules(256, 4) in sample.py/train.py


def _autocast_dtype(device_type):
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return None


class LayerNorm(nn.Module):
    """Identical to lib.models.LayerNorm (used for output_norm)."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class RMSNorm(nn.Module):
    """Replacement for apex.normalization.FusedRMSNorm(dim) (eps=1e-5, affine)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.eps = eps

    def forward(self, x):
        w = self.weight
        ac = _autocast_dtype(x.device.type)
        if ac is not None:  # apex: _cast_if_autocast_enabled(input, weight)
            x, w = x.to(ac), w.to(ac)
        xf = x.float()
        invvar = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (w.float() * (xf * invvar)).to(x.dtype)


class FusedMLP(nn.Module):
    """Replacement for flash_attn FusedMLP(dim, 4*dim, bias1=False, bias2=False)."""
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))


class MuReadout(nn.Linear):
    """Stand-in for mup.MuReadout: only output_mult and width_mult() are used."""
    def __init__(self, in_features, out_features, output_mult=1.0,
                 base_width=MUP_BASE_DIM):
        super().__init__(in_features, out_features, bias=True)
        self.output_mult = output_mult
        self._width_mult = in_features / base_width

    def width_mult(self):
        return self._width_mult


class Rotary(nn.Module):
    """Identical to lib.rotary.Rotary (inv_freq is a persistent buffer)."""
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if (seq_len != self.seq_len_cached
                or self.cos_cached.device != x.device):
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # identity on v
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    """Same math as lib.rotary (both the flash kernel and the TorchScript
    fallback): q,k rotated GPT-NeoX style, v untouched (cos=1, sin=0).
    cos/sin arrive already cast to qkv dtype (as in the original); compute in
    fp32 and round once, as the flash rotary kernel does."""
    qf = qkv.float()
    out = qf * cos.float() + rotate_half(qf) * sin.float()
    return out.to(qkv.dtype)


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x (verbatim from lib.models)."""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, causal, residual_scale):
        super().__init__()
        self.causal = causal
        self.dim = dim
        self.n_heads = n_heads
        self.residual_scale = residual_scale

        self.rmsnorm1 = RMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.rmsnorm2 = RMSNorm(dim)
        self.mlp = FusedMLP(dim, 4 * dim)

    def forward(self, x, rotary_cos_sin, cu_seqlens=None):
        if cu_seqlens is not None:
            raise NotImplementedError(
                'packed/variable-length sequences (cu_seqlens) are only used '
                'in training; not supported in the port')

        # Self-attention block
        x_skip = x
        x = self.rmsnorm1(x)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d',
                        three=3, h=self.n_heads)
        half_dtype = qkv.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(half_dtype), sin.to(half_dtype))
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # b h s d
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.,
                                           is_causal=self.causal)
        x = rearrange(x, 'b h s d -> b s (h d)')
        x = residual_linear(x, self.attn_out.weight, x_skip, self.residual_scale)

        # Feedforward block
        x_skip = x
        x = self.rmsnorm2(x)
        x = self.mlp(x)
        x = torch.add(x_skip, x, alpha=self.residual_scale)
        return x


class EmbeddingMatrix(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.matrix.data /= self.matrix.data.norm(p=2, dim=1, keepdim=True)

    def forward(self):
        norm = torch.linalg.norm(self.matrix, dim=1, keepdim=True)
        return self.matrix / (norm + 1e-8)


class NoiseSchedule(nn.Module):
    """Monotone learned schedule; returns normalized gamma in [0,1] (float64).
    Call with float64 t (the original ran with default dtype float64)."""
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(1024, 1))
        self.b1 = nn.Parameter(torch.randn(1024))
        self.W2 = nn.Parameter(torch.randn(1, 1024))

    def forward(self, t):
        W1 = F.softplus(self.W1.double())
        W2 = 0.01 * F.softplus(self.W2.double())

        def gamma_tilde(t):
            h = t[:, None] - 0.5
            h = (h @ W1.T) + self.b1[None, :].double()
            h = torch.tanh(h)
            return (h @ W2.T)[:, 0]

        t0 = torch.tensor([0.], device=t.device, dtype=torch.float64)
        t1 = torch.tensor([1.], device=t.device, dtype=torch.float64)
        g0, g1, gt = gamma_tilde(t0), gamma_tilde(t1), gamma_tilde(t)
        return (gt - g0) / (g1 - g0)


class GammaBounds(nn.Module):
    def __init__(self, gamma_0, gamma_1):
        super().__init__()
        self.gamma_0 = nn.Parameter(torch.tensor(float(gamma_0)))
        self.gamma_1 = nn.Parameter(torch.tensor(float(gamma_1)))

    def forward(self):
        return self.gamma_0.clone().double(), self.gamma_1.clone().double()


class DiffusionModel(nn.Module):
    def __init__(self, dim, embed_dim, n_blocks, n_heads, vocab_size):
        super().__init__()
        self.input_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear.weight.data.zero_()
        self.gamma_linear = nn.Linear(64, dim, bias=False)
        self.gamma_linear.weight.data.zero_()

        self.rotary_emb = Rotary(dim // n_heads)

        residual_scale = float(1. / np.sqrt(n_blocks))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, False, residual_scale)
            for _ in range(n_blocks)
        ])

        self.output_norm = LayerNorm(dim)
        self.output_linear = MuReadout(dim, vocab_size)
        self.output_linear.weight.data.zero_()
        self.output_linear.bias.data.zero_()

        self.dim = dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.amp = True  # bf16 autocast inside the blocks, as in the original

    def forward(self, z, gamma, embedding_matrix, bias_scale, x_selfcond,
                selfcond_mask=None, cu_seqlens=None, return_hidden=False):
        dev = z.device
        if selfcond_mask is None:
            selfcond_mask = torch.ones(z.shape[0], device=dev, dtype=torch.float64)

        alpha_squared = torch.sigmoid(-gamma)[:, None, None]
        sigma_squared = torch.sigmoid(gamma)[:, None, None]
        alpha = alpha_squared.sqrt()

        # Rescale input to stdev 1
        z_variance = (alpha_squared / self.embed_dim) + sigma_squared
        x = z / z_variance.sqrt().float()

        x = self.input_linear(x)
        x = x + self.selfcond_linear(x_selfcond * float(np.sqrt(self.embed_dim)))

        gamma_embed = torch.linspace(-5., 5., 64 // 2, device=dev, dtype=torch.float64)
        gamma_embed = gamma_embed.exp()[None, :] * gamma[:, None]
        gamma_embed = torch.cat([gamma_embed.sin(), gamma_embed.cos()], dim=1)
        gamma_embed = self.gamma_linear(gamma_embed.float())[:, None, :]
        x = x + gamma_embed

        rotary_cos_sin = self.rotary_emb(x)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=self.amp):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, cu_seqlens=cu_seqlens)

        x = self.output_norm(x.float())
        hidden = x
        x = x * (self.output_linear.output_mult / self.output_linear.width_mult())

        W = torch.cat([
            self.output_linear.weight.T,
            embedding_matrix.T,
            embedding_matrix.T.detach()
        ], dim=0)
        z_scaled_for_bias = bias_scale * (alpha / sigma_squared).float() * z
        m = selfcond_mask.float()[:, None, None]
        x = torch.cat([x, z_scaled_for_bias * (1 - m), z_scaled_for_bias * m], dim=2)
        logits = torch.addmm(
            self.output_linear.bias.view(1, self.vocab_size),
            x.view(-1, self.dim + 2 * self.embed_dim),
            W.view(self.dim + 2 * self.embed_dim, self.vocab_size)
        ).view(x.shape[0], x.shape[1], self.vocab_size)

        x_reconst = F.softmax(logits, dim=2)
        x_reconst = x_reconst @ torch.cat([embedding_matrix, embedding_matrix.detach()], dim=1)
        x_reconst = torch.lerp(x_reconst[:, :, :self.embed_dim],
                               x_reconst[:, :, self.embed_dim:], m)
        if return_hidden:
            return logits, x_reconst, hidden
        return logits, x_reconst


# Plaid 1B config (README / sample.py defaults)
PLAID1B = dict(dim=2048, n_blocks=24, n_heads=32, embed_dim=16, vocab_size=32768,
               gamma_0=-3., gamma_1=6.)


def create_modules(cfg=PLAID1B):
    return {
        'noise_schedule': NoiseSchedule(),
        'gamma_bounds': GammaBounds(cfg['gamma_0'], cfg['gamma_1']),
        'embedding_matrix': EmbeddingMatrix(cfg['vocab_size'], cfg['embed_dim']),
        'model': DiffusionModel(cfg['dim'], cfg['embed_dim'], cfg['n_blocks'],
                                cfg['n_heads'], cfg['vocab_size']),
    }


def remap_state_dict(name, sd):
    """Key remapping hook. The port uses the original names, so this is the
    identity; it only strips a DDP 'module.' prefix if one is present."""
    return {(k[len('module.'):] if k.startswith('module.') else k): v
            for k, v in sd.items()}


def load_plaid(weights_path, device='cpu', cfg=PLAID1B, strict=True, verbose=True):
    """Build the four Plaid modules on the meta device and load the released
    weights with load_state_dict(strict=True, assign=True). Returns
    (modules_dict, report) where report[name] = (missing, unexpected)."""
    with torch.device('meta'):
        modules = create_modules(cfg)
    report = {}
    for name, module in modules.items():
        sd = torch.load(os.path.join(weights_path, f'{name}.pt'),
                        map_location='cpu', mmap=True, weights_only=True)
        sd = remap_state_dict(name, sd)
        res = module.load_state_dict(sd, strict=strict, assign=True)
        report[name] = (list(res.missing_keys), list(res.unexpected_keys))
        module.float().to(device).eval()
        for p in module.parameters():
            p.requires_grad_(False)
        if verbose:
            n = sum(p.numel() for p in module.parameters())
            print(f'[load] {name}: {len(sd)} tensors, {n:,} params, '
                  f'missing={report[name][0]} unexpected={report[name][1]}',
                  flush=True)
    return modules, report


def load_tokenizer(path=None):
    from tokenizers import Tokenizer
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'owt2_tokenizer.json')
    return Tokenizer.from_file(path)
