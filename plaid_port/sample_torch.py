"""
Faithful port of Plaid's sample.py unconditional sampler (generate_samples
with guidance_tokens=[]), pure PyTorch.

Kept exactly from the original:
  * z_1 ~ N(0, initial_noise_scale^2 I) in float64; x_selfcond = 0 (fp32)
  * loop over t in linspace(1, 0, T) (float64), s = t - 1/T
  * gamma = gamma_0 + (gamma_1 - gamma_0) * noise_schedule(t)
  * model called with z.float(), gamma_t.float(), bias_scale=1, x_selfcond
  * x_selfcond <- x_reconst (self-conditioning carried across steps)
  * score temperature: eps_pred /= score_temp (0.9); x_reconst recomputed
  * ancestral update (VDM App. A.4 eq. 33):
        c = -expm1(gamma_s - gamma_t)
        z *= (1-c) alpha_s/alpha_t ; z += c alpha_s x_reconst
        z += sqrt(c (1-alpha_s^2)) * N(0, I)
    (or DDIM if --ddim), skipped at t == 0
  * final: logits = model(z, gamma_t(=gamma at t=0), x_selfcond); argmax
  * torch.backends.cuda.matmul.allow_tf32 = True (as in sample.py)

Additions: --no_selfcond (x_selfcond stays 0 at every step, incl. the final
call), --fp32 (disable bf16 autocast in the blocks, disable TF32), seed.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plaid_torch  # noqa: E402


@torch.no_grad()
def generate_samples(modules, n_samples, seq_len, steps, selfcond=True,
                     score_temp=0.9, initial_noise_scale=1.0, ddim=False,
                     generator=None, device='cuda', return_traj=False):
    model = modules['model']
    embed_dim = model.embed_dim
    f64 = torch.float64
    embedding_matrix = modules['embedding_matrix']()

    gamma_0, gamma_1 = modules['gamma_bounds']()

    z = torch.randn((n_samples, seq_len, embed_dim), device=device, dtype=f64,
                    generator=generator) * initial_noise_scale
    x_selfcond = torch.zeros_like(z).float()
    ts = torch.linspace(1., 0., steps, dtype=f64, device=device)
    for t in ts:
        t = t[None]
        s = t - 1. / steps
        gamma_s = modules['noise_schedule'](s).double()
        gamma_t = modules['noise_schedule'](t).double()
        gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
        gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
        alpha_squared_s = torch.sigmoid(-gamma_s)
        alpha_squared_t = torch.sigmoid(-gamma_t)
        alpha_s = alpha_squared_s.sqrt()
        alpha_t = alpha_squared_t.sqrt()
        sigma_squared_s = torch.sigmoid(gamma_s)
        sigma_squared_t = torch.sigmoid(gamma_t)
        sigma_s = sigma_squared_s.sqrt()
        sigma_t = sigma_squared_t.sqrt()

        _, x_reconst = model(
            z=z.to(torch.float32, copy=True),
            gamma=gamma_t.float(),
            embedding_matrix=embedding_matrix,
            bias_scale=1.,
            x_selfcond=x_selfcond,
        )
        if selfcond:
            x_selfcond = x_reconst.clone().detach()
        x_reconst = x_reconst.double()
        epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
        epsilon_pred /= score_temp
        x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
        if t > 0:
            if ddim:
                z = (alpha_s * x_reconst) + (sigma_s * epsilon_pred)
            else:
                c = -torch.expm1(gamma_s - gamma_t)
                z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                z += c * (alpha_squared_s.sqrt() * x_reconst.double())
                z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn(
                    z.shape, device=device, dtype=f64, generator=generator)

    logits, _ = model(
        z=z.float(),
        gamma=gamma_t.float(),
        embedding_matrix=embedding_matrix,
        bias_scale=1.,
        x_selfcond=x_selfcond,
    )
    return logits.argmax(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights_path', required=True)
    ap.add_argument('--n_samples', type=int, default=8)
    ap.add_argument('--seq_len', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=4096)  # sampling_timesteps
    ap.add_argument('--score_temp', type=float, default=0.9)
    ap.add_argument('--initial_noise_scale', type=float, default=1.0)
    ap.add_argument('--ddim', action='store_true')
    ap.add_argument('--no_selfcond', action='store_true')
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=0, help='0 = all at once')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = not args.fp32
    torch.backends.cudnn.allow_tf32 = not args.fp32
    modules, _ = plaid_torch.load_plaid(args.weights_path, device=args.device)
    modules['model'].amp = not args.fp32
    tok = plaid_torch.load_tokenizer()
    run(modules, tok, args)


def run(modules, tok, args):
    g = torch.Generator(device=args.device).manual_seed(args.seed)
    bs = args.batch_size or args.n_samples
    t0 = time.time()
    out = []
    for i in range(0, args.n_samples, bs):
        n = min(bs, args.n_samples - i)
        xs = generate_samples(modules, n, args.seq_len, args.steps,
                              selfcond=not args.no_selfcond,
                              score_temp=args.score_temp,
                              initial_noise_scale=args.initial_noise_scale,
                              ddim=args.ddim, generator=g, device=args.device)
        out.extend(xs.tolist())
    dt = time.time() - t0
    texts = [tok.decode(x, skip_special_tokens=False) for x in out]
    tag = f"SC={'off' if args.no_selfcond else 'on'} steps={args.steps} " \
          f"seq_len={args.seq_len} seed={args.seed} fp32={args.fp32}"
    print(f'=== samples [{tag}] ({dt:.1f}s) ===', flush=True)
    for x in texts:
        print(x.replace('\n', '↵'))
        print('---')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'seconds': dt,
                       'tokens': out, 'texts': texts}, f, indent=1)
    return texts


if __name__ == '__main__':
    main()
