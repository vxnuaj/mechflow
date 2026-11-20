"""Phase 13 step 3, family E4 v2: O/N/F decomposition for a (fine-tuned) ELF-B checkpoint with truncation and
decode-access settings. Access ∈ {read, zero, shuf}:
  read: the decode pass receives each position's own last estimate x̂;
  zero: zero SC at decode (ELF default);
  shuf: the decode pass receives x̂ from another sample (batch rolled by 1), same marginal statistics (red-team M14).
Per config (n_steps, access), target j:
  - cells.json: {j: {cell: [token ids per sample]}} (8 routed cells);
  - plain_on.json: {"tokens": SC-on tokens with decode as configured} (a base-model run gives the common reference);
  - meta.json:
    - identities: token agreement and continuous max-diffs (final z, decode logits) for O1N1F1 vs SC-on and O0N0F0 vs SC-off;
    - regime (red-team C1/M4/M9): the decoder's mean log-prob of the full-length SC-on token from the truncated state
      vs from the truncated estimate x̂ (zero SC), globally and for |i − j| ≤ 16 (excluding j);
    - uptake (C5/M12): P(decode token changes when the decode SC is switched from the configured input to zero),
      for access read/shuf;
    - dis: P(decode(z) ≠ decode(x̂)) (kept for continuity)."""
import argparse, itertools, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_instr import load_elf, run_cell_x, run_plain_x, decode_plain_x, _dec_logits
import elf_instr as EI
ap = argparse.ArgumentParser()
ap.add_argument("--hcfrl", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_steps", default="32"); ap.add_argument("--access", default="zero"); ap.add_argument("--j", type=int, default=64)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--L", type=int, default=128)
ap.add_argument("--seed", type=int, default=2); ap.add_argument("--device", default="cuda"); ap.add_argument("--plain_only", action="store_true")
a = ap.parse_args(); dev = torch.device(a.device)
if dev.type == "cpu": torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
model, cfg, tok = load_elf(a.hcfrl, a.ckpt, a.L, dev)
from utils.sampling_utils import get_sampling_steps
torch.manual_seed(1); ts = get_sampling_steps(32, time_schedule="logit_normal").to(dev)
z0_all = torch.randn((a.n, a.L, model.text_encoder_dim), generator=torch.Generator().manual_seed(a.seed)) * cfg.denoiser_noise_scale
cells = [(o, n, f) for o in (0, 1) for n in (0, 1) for f in (0, 1)]
win = torch.tensor([i for i in range(a.L) if abs(i - a.j) <= 16 and i != a.j])


@torch.no_grad()
def decode_cell_lg(z, xprev, fbd, j, O, N, F):
    """Routed decode with an explicit decode-SC tensor fbd (x̂, zeros or shuffled x̂). Returns tokens, the neighbour-source
    stream's logits (F if F else S) and O-stream logits. The identity check compares the streams actually used."""
    B, L, _ = z.shape; zero = torch.zeros_like(z); o_in = zero.clone()
    if O: o_in[:, j] = fbd[:, j]
    inp = torch.cat([torch.cat([z, zero], -1), torch.cat([z, fbd], -1), torch.cat([z, o_in], -1)], 0)
    EI.ROUTE = {"L": L, "j": j, "n": N}
    try: lg = _dec_logits(model, inp, 1.0)
    finally: EI.ROUTE = None
    lS, lF, lO = lg.split(B); y = (lF if F else lS).argmax(-1).clone(); y[:, j] = lO[:, j].argmax(-1)
    return y, (lF if F else lS), lO                               # the stream whose logits produce the neighbours' tokens


def fb_for(acc, xp):
    if acc == "read": return xp
    if acc == "shuf": return torch.roll(xp, 1, 0)
    return torch.zeros_like(xp)


for nst, acc in itertools.product([int(x) for x in a.n_steps.split(",")], a.access.split(",")):
    tsn = ts[:nst + 1]; out = {f"O{o}N{n}F{f}": [] for o, n, f in cells}; plain = []
    M = {"tok_ident": [], "z_maxdiff": [], "logit_maxdiff": [], "lp_state": [], "lp_fb": [], "lp_state_win": [], "lp_fb_win": [], "uptake": [], "dis": []}
    t0 = time.time()
    for b in range(0, a.n, a.bs):
        z0 = z0_all[b:b + a.bs].to(dev)
        zon, xon = run_plain_x(model, cfg, z0, tsn, True); zoff, xoff = run_plain_x(model, cfg, z0, tsn, False)
        fbd_on = fb_for(acc, xon)
        lg_on = _dec_logits(model, torch.cat([zon, fbd_on], -1), 1.0); yon = lg_on.argmax(-1)
        lg_off = _dec_logits(model, torch.cat([zoff, torch.zeros_like(zoff)], -1), 1.0); yoff = lg_off.argmax(-1)
        plain += yon.tolist()
        zf, xf = run_plain_x(model, cfg, z0, ts, True); yfull = decode_plain_x(model, zf, xf, False)       # full-length SC-on (default decode)
        ls = torch.log_softmax(_dec_logits(model, torch.cat([zon, torch.zeros_like(zon)], -1), 1.0).float(), -1).gather(-1, yfull[..., None])[..., 0]
        lf = torch.log_softmax(_dec_logits(model, torch.cat([xon, torch.zeros_like(xon)], -1), 1.0).float(), -1).gather(-1, yfull[..., None])[..., 0]
        M["lp_state"].append(ls.mean().item()); M["lp_fb"].append(lf.mean().item())
        M["lp_state_win"].append(ls[:, win].mean().item()); M["lp_fb_win"].append(lf[:, win].mean().item())
        M["dis"].append((decode_plain_x(model, zon, xon, False) != decode_plain_x(model, xon, torch.zeros_like(xon), False)).float().mean().item())
        if acc != "zero": M["uptake"].append((yon != decode_plain_x(model, zon, xon, False)).float().mean().item())
        if a.plain_only: continue
        for o, n, f in cells:
            z, xp = run_cell_x(model, cfg, z0, tsn, a.j, o, n, f)
            y, lN, lO = decode_cell_lg(z, xp, fb_for(acc, xp), a.j, o, n, f)
            if (o, n, f) in ((1, 1, 1), (0, 0, 0)):
                zr, lr, yr = (zon, lg_on, yon) if (o, n, f) == (1, 1, 1) else (zoff, lg_off, yoff)
                M["tok_ident"].append((y == yr).float().mean().item()); M["z_maxdiff"].append((z - zr).abs().max().item())
                M["logit_maxdiff"].append(max((lN.float() - lr.float()).abs().max().item(), (lO[:, a.j].float() - lr[:, a.j].float()).abs().max().item()))
            out[f"O{o}N{n}F{f}"] += y.tolist()
        print(f"n_steps {nst} access {acc} batch {b}: {time.time()-t0:.0f}s, identity min {min(M['tok_ident']):.4f}, logit maxdiff {max(M['logit_maxdiff']):.3g}", flush=True)
    d = f"{a.out}/{nst}_{acc}"; os.makedirs(d, exist_ok=True)
    json.dump({"tokens": plain}, open(f"{d}/plain_on.json", "w"))
    meta = {"ckpt": a.ckpt, "n_steps": nst, "t_end": tsn[-1].item(), "access": acc, "j": a.j, "n": a.n, "seed": a.seed}
    for k, v in M.items():
        if v: meta[k] = (min(v) if k == "tok_ident" else max(v) if k.endswith("maxdiff") else sum(v) / len(v))
    meta["identity_min"] = meta.get("tok_ident", 1.0)
    if not a.plain_only:
        assert meta["identity_min"] >= 0.99, f"identity agreement {meta['identity_min']}"
        json.dump({str(a.j): out}, open(f"{d}/cells.json", "w"))
    json.dump(meta, open(f"{d}/meta.json", "w"))
