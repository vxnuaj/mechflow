"""Phase 13 step 3: O/N/F decomposition for a (fine-tuned) ELF-B checkpoint, with truncation and decode-access settings.
For each config (n_steps ∈ --n_steps, inference access ∈ --access), target j = --j:
  - plain SC-on and SC-off runs (identity references) and the manipulation check
    dis = P(decode(z) ≠ decode(x̂)) (both with zero SC), as in elf_trunc_calib.py;
  - 8 routed cells (trajectory routing as in family E; decode pass routed, with SC readable iff access = read);
  - identities: O1N1F1 = SC-on (decode SC as configured) and O0N0F0 = SC-off, ≥ 99% token agreement (bf16 on GPU).
Writes <out>/<n>_<access>/cells.json in the {j: {cell: [ids]}} format read by elf_judge_fixedctx.py, plus meta.json."""
import argparse, itertools, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_instr import load_elf, run_cell_x, run_plain_x, decode_cell, decode_plain_x
ap = argparse.ArgumentParser()
ap.add_argument("--hcfrl", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_steps", default="32"); ap.add_argument("--access", default="zero"); ap.add_argument("--j", type=int, default=64)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--L", type=int, default=128)
ap.add_argument("--seed", type=int, default=2); ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device)
if dev.type == "cpu": torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
model, cfg, tok = load_elf(a.hcfrl, a.ckpt, a.L, dev)
from utils.sampling_utils import get_sampling_steps
torch.manual_seed(1); ts = get_sampling_steps(32, time_schedule="logit_normal").to(dev)
z0_all = torch.randn((a.n, a.L, model.text_encoder_dim), generator=torch.Generator().manual_seed(a.seed)) * cfg.denoiser_noise_scale
cells = [(o, n, f) for o in (0, 1) for n in (0, 1) for f in (0, 1)]
for nst, acc in itertools.product([int(x) for x in a.n_steps.split(",")], a.access.split(",")):
    rd = acc == "read"; tsn = ts[:nst + 1]; out = {f"O{o}N{n}F{f}": [] for o, n, f in cells}; ident = []; dis = []; t0 = time.time()
    for b in range(0, a.n, a.bs):
        z0 = z0_all[b:b + a.bs].to(dev)
        zon, xon = run_plain_x(model, cfg, z0, tsn, True); zoff, xoff = run_plain_x(model, cfg, z0, tsn, False)
        yon = decode_plain_x(model, zon, xon, rd); yoff = decode_plain_x(model, zoff, xoff, False)
        dis.append((decode_plain_x(model, zon, xon, False) != decode_plain_x(model, xon, torch.zeros_like(xon), False)).float().mean().item())
        for o, n, f in cells:
            z, xp = run_cell_x(model, cfg, z0, tsn, a.j, o, n, f)
            y = decode_cell(model, z, xp, a.j, o, n, f, rd)
            if (o, n, f) == (1, 1, 1): ident.append((y == yon).float().mean().item())
            if (o, n, f) == (0, 0, 0): ident.append((y == yoff).float().mean().item())
            out[f"O{o}N{n}F{f}"] += y.tolist()
        print(f"n_steps {nst} access {acc} batch {b}: {time.time()-t0:.0f}s, identity min {min(ident):.4f}", flush=True)
    assert min(ident) >= 0.99, f"identity agreement {min(ident)}"
    d = f"{a.out}/{nst}_{acc}"; os.makedirs(d, exist_ok=True)
    json.dump({str(a.j): out}, open(f"{d}/cells.json", "w"))
    json.dump({"ckpt": a.ckpt, "n_steps": nst, "t_end": tsn[-1].item(), "access": acc, "j": a.j, "n": a.n, "seed": a.seed,
               "identity_min": min(ident), "dis": sum(dis) / len(dis)}, open(f"{d}/meta.json", "w"))
