"""GPU validation driver: loads Plaid 1B once, then runs (b) ELBO evals and
(c) unconditional samples. Writes JSON results to --out_dir."""
import argparse
import json
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elbo_torch  # noqa: E402
import plaid_torch  # noqa: E402
import sample_torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights_path', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--only', default='', help='comma list of run names')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda'
    # as in sample.py / train.py
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    modules, report = plaid_torch.load_plaid(args.weights_path, device=dev)
    tok = plaid_torch.load_tokenizer()
    print(torch.cuda.get_device_name(), flush=True)

    D = args.data_dir
    common = dict(weights_path=args.weights_path, device=dev, seed=0)
    elbo_runs = {
        'wt103_L1024': dict(data='wikitext', parquet=f'{D}/wikitext-103-v1-test.parquet',
                            seq_len=1024, K=8, batch_size=8, max_seqs=0, n_seqs=0),
        'wt2_L1024': dict(data='wikitext', parquet=f'{D}/wikitext-2-v1-test.parquet',
                          seq_len=1024, K=8, batch_size=8, max_seqs=0, n_seqs=0),
        'owt_L256': dict(data='owt', parquet=f'{D}/owt-0000.parquet', seq_len=256,
                         n_seqs=64, K=16, batch_size=32, max_seqs=0),
        'owt_L1024': dict(data='owt', parquet=f'{D}/owt-0000.parquet', seq_len=1024,
                          n_seqs=32, K=8, batch_size=8, max_seqs=0),
    }
    variants = [('', dict(no_selfcond=False, fp32=False))]
    only = set(filter(None, args.only.split(',')))
    summary = {}
    for name, cfg in elbo_runs.items():
        vs = variants
        if name == 'owt_L256':  # extra precision / SC ablations on the cheap run
            vs = variants + [('_fp32', dict(no_selfcond=False, fp32=True)),
                             ('_noSC', dict(no_selfcond=True, fp32=False))]
        for suffix, v in vs:
            run_name = name + suffix
            if only and run_name not in only:
                continue
            a = Namespace(**common, **cfg, **v, out=f'{args.out_dir}/elbo_{run_name}.json')
            torch.backends.cuda.matmul.allow_tf32 = not a.fp32
            modules['model'].amp = not a.fp32
            summary[run_name] = elbo_torch.run(modules, tok, a)
    torch.backends.cuda.matmul.allow_tf32 = True
    modules['model'].amp = True

    for sc in (True, False):
        run_name = f'samples_SC{"on" if sc else "off"}'
        if only and run_name not in only:
            continue
        a = Namespace(**common, n_samples=4, seq_len=256, steps=256, score_temp=0.9,
                      initial_noise_scale=1.0, ddim=False, no_selfcond=not sc,
                      fp32=False, batch_size=0, out=f'{args.out_dir}/{run_name}.json')
        sample_torch.run(modules, tok, a)

    with open(f'{args.out_dir}/elbo_summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
