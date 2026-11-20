"""
Port of Plaid's held-out NLL bound (train.py: forward(x_eval=...) and
compute_nll), pure PyTorch, plus the zero-shot WikiText protocol of
lib/datasets.py::_eval_dataset.

The repo's eval-mode estimator, per batch of B sequences:
  * selfcond_mask = 1 for all rows (held-out eval always unrolls SC to y2):
    a no-grad forward with x_selfcond=0 gives x_reconst, which is fed as
    x_selfcond to the main forward.
  * reconst_bs ~ B/8 rows get t=0, the rest get low-discrepancy t in (0,1).
  * reconst  = CE(logits(t=0), x).mean over positions          (nats/token)
  * diffusion = -0.5 * SNR'(t) * mean_pos(sum_dim (x_embed - x_reconst)^2),
               SNR'(t) = -exp(-gamma) * gamma'(t), gamma'(t) by autograd
  * prior = KL(N(alpha_1 x_embed, sigma_1^2) || N(0,1)) summed over the 16
            embedding dims, mean over positions
  * nll = reconst/avg_reconst_bs + diffusion/(B - avg_reconst_bs) + prior
i.e. an unbiased single-sample estimate of
     E_x[ reconst(t=0) + E_{t~U(0,1)} diffusion(t) + prior ]   (nats/token)

This script evaluates the same expectation with lower variance: for each
sequence, the reconst term at t=0 (1 noise draw) plus the diffusion term at K
stratified times t_k = (k + u)/K, averaged. Token-weighted aggregation as in
compute_nll (nll * X.numel()).

WikiText: chunks of 1024 non-overlapping tokens of the detokenized test set
(the repo's detokenizer, verbatim); word-level PPL = exp(total_nats / words),
words = len(raw.split(' ')) as in _wikitext_untokenized.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plaid_torch  # noqa: E402


# ---------------------------------------------------------------- loss pieces
def cross_entropy(logits, targets):
    logprobs = F.log_softmax(logits, dim=2)
    return -logprobs.gather(2, targets[:, :, None])[:, :, 0]


def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    """KL(p||q)"""
    return (sigma_q.log() - sigma_p.log()
            + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2) - 0.5)


def gamma_and_prime(modules, t):
    """gamma(t) (unnormalized) and d gamma/dt, both float64, as in train.py."""
    t = t.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        g = modules['noise_schedule'](t)
        g_prime = torch.autograd.grad(g.sum(), [t])[0]
    gamma_0, gamma_1 = modules['gamma_bounds']()
    return (gamma_0 + (gamma_1 - gamma_0) * g.detach(),
            (gamma_1 - gamma_0) * g_prime.detach())


@torch.no_grad()
def model_with_selfcond(modules, z, gamma, embedding_matrix, selfcond=True):
    model = modules['model']
    x_selfcond = torch.zeros_like(z)
    if selfcond:
        _, x_selfcond = model(z, gamma, embedding_matrix, 1., torch.zeros_like(z))
    mask = torch.full([z.shape[0]], 1. if selfcond else 0., device=z.device,
                      dtype=torch.float64)
    return model(z, gamma, embedding_matrix, 1., x_selfcond, selfcond_mask=mask)


@torch.no_grad()
def elbo_terms(modules, x, t, generator=None, selfcond=True):
    """x: [B, L] int64, t: [B] float64 in [0,1]. Returns per-sequence float64
    (reconst CE if t==0 else nan, diffusion term, prior term), nats/token."""
    dev = x.device
    embedding_matrix = modules['embedding_matrix']()
    gamma, gamma_prime = gamma_and_prime(modules, t)
    gamma_0, gamma_1 = modules['gamma_bounds']()
    alpha_squared = torch.sigmoid(-gamma)
    sigma_squared = torch.sigmoid(gamma)
    alpha, sigma = alpha_squared.sqrt(), sigma_squared.sqrt()
    snr_prime = -(-gamma).exp() * gamma_prime
    alpha_1 = torch.sigmoid(-gamma_1).sqrt()
    sigma_1 = torch.sigmoid(gamma_1).sqrt()

    x_embed = embedding_matrix[x]
    z = torch.randn([x.shape[0], x.shape[1], x_embed.shape[-1]],
                    dtype=torch.float32, device=dev, generator=generator)
    z.mul_(sigma[:, None, None])
    z.add_(alpha[:, None, None] * x_embed)

    logits, x_reconst = model_with_selfcond(modules, z, gamma, embedding_matrix,
                                            selfcond)
    reconst = cross_entropy(logits, x).mean(dim=1).double()
    reconst = torch.where(t == 0, reconst, torch.full_like(reconst, float('nan')))
    prior = gaussian_kl(alpha_1 * x_embed, sigma_1,
                        torch.tensor(0., device=dev), torch.tensor(1., device=dev)
                        ).sum(dim=2).mean(dim=1).double()
    diffusion = (x_embed - x_reconst).pow(2).mean(dim=1).double().sum(dim=1)
    diffusion = -0.5 * (snr_prime * diffusion)
    diffusion = torch.where(t == 0, torch.zeros_like(diffusion), diffusion)
    return reconst, diffusion, prior


def eval_sequences(modules, seqs, K, batch_size, device, seed, selfcond=True):
    """seqs: list of 1-D LongTensors (same length within a batch preferred).
    Returns per-sequence dict list."""
    g = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.RandomState(seed)
    rows = []
    # group by length
    by_len = {}
    for i, s in enumerate(seqs):
        by_len.setdefault(len(s), []).append(i)
    results = [None] * len(seqs)
    for L, idxs in by_len.items():
        # work items: (seq index, t) with t=0 once and K stratified t's
        items = []
        for i in idxs:
            u = rng.uniform()
            items.append((i, 0.))
            items += [(i, (k + u) / K) for k in range(K)]
        acc = {i: {'reconst': None, 'diff': [], 'prior': None} for i in idxs}
        for b in range(0, len(items), batch_size):
            chunk = items[b:b + batch_size]
            x = torch.stack([seqs[i] for i, _ in chunk]).to(device)
            t = torch.tensor([tt for _, tt in chunk], dtype=torch.float64, device=device)
            r, d, p = elbo_terms(modules, x, t, generator=g, selfcond=selfcond)
            for j, (i, tt) in enumerate(chunk):
                if tt == 0.:
                    acc[i]['reconst'] = r[j].item()
                    acc[i]['prior'] = p[j].item()
                else:
                    acc[i]['diff'].append(d[j].item())
        for i in idxs:
            a = acc[i]
            dm = float(np.mean(a['diff']))
            results[i] = dict(n_tokens=L, reconst=a['reconst'], prior=a['prior'],
                              diffusion=dm,
                              diffusion_se=float(np.std(a['diff'], ddof=1) / math.sqrt(K)),
                              nll=a['reconst'] + dm + a['prior'])
    return results


def summarize(results):
    n = np.array([r['n_tokens'] for r in results], dtype=np.float64)
    nll = np.array([r['nll'] for r in results])
    tot = float((nll * n).sum())
    mean = tot / n.sum()
    # standard error of the token-weighted mean across sequences
    w = n / n.sum()
    se = float(np.sqrt(((w * (nll - mean)) ** 2).sum() * len(n) / max(len(n) - 1, 1)))
    comp = {k: float((np.array([r[k] for r in results]) * n).sum() / n.sum())
            for k in ['reconst', 'diffusion', 'prior']}
    return dict(n_seqs=len(results), n_tokens=int(n.sum()), total_nats=tot,
                nats_per_token=mean, se=se, bits_per_token=mean / math.log(2),
                components=comp)


# ---------------------------------------------------------------- datasets
def wikitext_detokenizer(string):
    """Verbatim from lib/datasets.py::_wikitext_untokenized (megatron-3d)."""
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def wikitext_sequences(parquet_path, tok, seq_len=1024):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet_path).column('text').to_pylist()
    # HF stores the original wiki.test.tokens line by line, with blank
    # lines (" \n" in the original file) as "". Rebuild the file text.
    raw = ''.join(r if r != '' else ' \n' for r in rows)
    dataset = raw[1:]  # [1:] to drop an initial space (as in the repo)
    total_words = len(dataset.split(' '))
    dataset = wikitext_detokenizer(dataset)
    ids = torch.tensor(tok.encode(dataset).ids, dtype=torch.int64)
    seqs = [ids[i:i + seq_len] for i in range(0, len(ids), seq_len)]
    return seqs, total_words, len(ids)


def owt_sequences(parquet_path, tok, seq_len, n_seqs, from_end=True):
    """OpenWebText (Skylion007, = OWT1) docs, tokenized and packed like
    lib/datasets.py::_to_chunks (EOT id 0 appended after every document)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    rg = pf.num_row_groups - 1 if from_end else 0
    docs = pf.read_row_group(rg, columns=['text']).column('text').to_pylist()
    if from_end:
        docs = docs[::-1]
    buf, out = [], []
    for d in docs:
        buf += tok.encode(d).ids + [0]
        while len(buf) >= seq_len and len(out) < n_seqs:
            out.append(torch.tensor(buf[:seq_len], dtype=torch.int64))
            buf = buf[seq_len:]
        if len(out) >= n_seqs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights_path', required=True)
    ap.add_argument('--data', choices=['wikitext', 'owt'], required=True)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--seq_len', type=int, default=1024)
    ap.add_argument('--n_seqs', type=int, default=32, help='owt only')
    ap.add_argument('--max_seqs', type=int, default=0, help='wikitext: 0 = all')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--no_selfcond', action='store_true')
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
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
    extra = {}
    if args.data == 'wikitext':
        seqs, words, ntok = wikitext_sequences(args.parquet, tok, args.seq_len)
        if args.max_seqs:
            seqs = seqs[:args.max_seqs]
        extra = dict(total_words=words, total_tokens_full=ntok,
                     words_per_token=words / ntok)
    else:
        seqs = owt_sequences(args.parquet, tok, args.seq_len, args.n_seqs)
    t0 = time.time()
    res = eval_sequences(modules, seqs, args.K, args.batch_size, args.device,
                         args.seed, selfcond=not args.no_selfcond)
    summ = summarize(res) | extra
    summ['seconds'] = time.time() - t0
    if args.data == 'wikitext' and not args.max_seqs:
        summ['word_ppl'] = math.exp(summ['total_nats'] / extra['total_words'])
        # +-1 SE band on the word-level PPL
        se_tot = summ['se'] * summ['n_tokens']
        summ['word_ppl_lo'] = math.exp((summ['total_nats'] - se_tot) / extra['total_words'])
        summ['word_ppl_hi'] = math.exp((summ['total_nats'] + se_tot) / extra['total_words'])
    tag = f"{args.data} {os.path.basename(args.parquet)} L={args.seq_len} K={args.K} " \
          f"SC={'off' if args.no_selfcond else 'on'} fp32={args.fp32}"
    print(f'=== ELBO [{tag}] ===')
    print(json.dumps(summ, indent=1), flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'summary': summ, 'per_seq': res}, f, indent=1)
    return summ


if __name__ == '__main__':
    main()
