"""Phase 13 step 4: O/N/F decomposition for LangFlow (family G). Target j = L/2; n samples, noise seed --seed.
Writes <out>/cells.json ({j: {cell: [ids]}}, for elf_judge_fixedctx.py with T5_PATH = the GPT-2 tokenizer and REF_FILE =
plain_on.json, i.e. a common reference context = the SC-on tokens for the same noise), plain_on.json and meta.json
(identity token agreement for O1N1F1 / O0N0F0)."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from langflow_instr import load_langflow, schedule, sample_plain, sample_cell
ap = argparse.ArgumentParser(); ap.add_argument("--path", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--L", type=int, default=1024)
ap.add_argument("--K", type=int, default=128); ap.add_argument("--seed", type=int, default=2); ap.add_argument("--device", default="cuda")
ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=0)   # chunking (0 = all); noise is sliced from the full draw
a = ap.parse_args(); dev = torch.device(a.device)
if dev.type == "cpu": torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
m = load_langflow(a.path, dev); D = m.config.hidden_size; j = a.L // 2; gam = schedule(m, a.K, dev)
z0_all = torch.randn(a.n, a.L, D, generator=torch.Generator().manual_seed(a.seed))
if a.count: z0_all = z0_all[a.start:a.start + a.count]
cells = [(o, n, f) for o in (0, 1) for n in (0, 1) for f in (0, 1)]; out = {f"O{o}N{n}F{f}": [] for o, n, f in cells}; plain = []; ident = []; t0 = time.time()
for b in range(0, len(z0_all), a.bs):
    z0 = z0_all[b:b + a.bs].to(dev)
    yon, *_ = sample_plain(m, z0, gam, True); yoff, *_ = sample_plain(m, z0, gam, False); plain += yon.tolist()
    for o, n, f in cells:
        y = sample_cell(m, z0, gam, j, o, n, f)
        if (o, n, f) == (1, 1, 1): ident.append((y == yon).float().mean().item())
        if (o, n, f) == (0, 0, 0): ident.append((y == yoff).float().mean().item())
        out[f"O{o}N{n}F{f}"] += y.tolist()
    print(f"batch {b}: {time.time()-t0:.0f}s identity min {min(ident):.4f}", flush=True)
assert min(ident) >= 0.99, f"identity {min(ident)}"
os.makedirs(a.out, exist_ok=True)
json.dump({str(j): out}, open(f"{a.out}/cells.json", "w")); json.dump({"tokens": plain}, open(f"{a.out}/plain_on.json", "w"))
json.dump({"n": len(z0_all), "start": a.start, "L": a.L, "K": a.K, "j": j, "seed": a.seed, "identity_min": min(ident)}, open(f"{a.out}/meta.json", "w"))
