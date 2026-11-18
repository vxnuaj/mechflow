"""ELF 2x2x2 generation (prereg §2.7). hcfrl venv. Writes results/elf_decomp/cells.json:
{target j: {cell: [[token ids] per sample]}} for n samples (same noise per sample across cells)."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_instr import load_elf, run_cell, decode
ap = argparse.ArgumentParser(); ap.add_argument("--hcfrl", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--L", type=int, default=128); ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="results/elf_decomp/cells.json")
a = ap.parse_args(); dev = torch.device(a.device)
model, cfg, tok = load_elf(a.hcfrl, a.ckpt, a.L, dev)
from utils.sampling_utils import get_sampling_steps
t_steps = get_sampling_steps(a.steps, time_schedule="logit_normal", generator=torch.Generator().manual_seed(1)).to(dev)
z0 = torch.randn((a.n, a.L, model.text_encoder_dim), generator=torch.Generator().manual_seed(0)) * cfg.denoiser_noise_scale
res = {}
for j in (32, 64, 96):
    res[j] = {}
    for O in (0, 1):
        for N in (0, 1):
            for F in (0, 1):
                t0 = time.time(); ids = []
                for b in range(0, a.n, a.bs):
                    z = run_cell(model, cfg, z0[b:b + a.bs].to(dev), t_steps, j, O, N, F, w=1.0)
                    ids += decode(model, cfg, z, t_steps[-1], 1.0).tolist()
                res[j][f"O{O}N{N}F{F}"] = ids
                print(f"j={j} O{O}N{N}F{F}: {time.time()-t0:.0f}s", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"))
