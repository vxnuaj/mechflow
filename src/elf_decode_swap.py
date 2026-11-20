"""ELF decoder test (prereg §2.10). For target j and each 2x2x2 cell, decode token j twice:
  own:  decode(z_cell)                                   (as in family E)
  swap: decode(z_full with position j replaced by z_cell[:, j])   (neighbours' final latents fixed to the O1N1F1 cell)
If F's influence on j arises at ELF's cross-position decoder (reading neighbours' final latents), it vanishes in `swap`.
Writes {j: {cell: [ids]}} for each variant (decoded tokens over the whole sequence; only position j differs in swap)."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_instr import load_elf, run_cell, decode
ap = argparse.ArgumentParser(); ap.add_argument("--hcfrl", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--L", type=int, default=128); ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--device", default="cpu"); ap.add_argument("--j", type=int, default=64)
ap.add_argument("--outdir", default="results/elf_swap"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = torch.device(a.device); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
model, cfg, tok = load_elf(a.hcfrl, a.ckpt, a.L, dev)
from utils.sampling_utils import get_sampling_steps
torch.manual_seed(1); t_steps = get_sampling_steps(a.steps, time_schedule="logit_normal").to(dev)
z0 = torch.randn((a.n, a.L, model.text_encoder_dim), generator=torch.Generator().manual_seed(a.seed)) * cfg.denoiser_noise_scale
cells = [(o, n, f) for o in (0, 1) for n in (0, 1) for f in (0, 1)]
own = {f"O{o}N{n}F{f}": [] for o, n, f in cells}; swap = {k: [] for k in own}
for b in range(0, a.n, a.bs):
    t0 = time.time(); zs = {}
    for o, n, f in cells:
        zs[f"O{o}N{n}F{f}"] = run_cell(model, cfg, z0[b:b + a.bs].to(dev), t_steps, a.j, o, n, f, w=1.0)
    zf = zs["O1N1F1"]
    for k, z in zs.items():
        own[k] += decode(model, cfg, z, t_steps[-1], 1.0).tolist()
        zm = zf.clone(); zm[:, a.j] = z[:, a.j]
        swap[k] += decode(model, cfg, zm, t_steps[-1], 1.0).tolist()
    print(f"batch {b}: {time.time()-t0:.0f}s", flush=True)
os.makedirs(a.outdir, exist_ok=True)
json.dump({str(a.j): own}, open(f"{a.outdir}/own.json", "w")); json.dump({str(a.j): swap}, open(f"{a.outdir}/swap.json", "w"))
