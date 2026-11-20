"""Phase 13 step 3a-iii (regime diagnostics only; no O/N/F decomposition). Noise seed 7 (not used in E/E3/E3b, seeds 0/1).
For ELF-B, SC-on sampling truncated after n of 32 logit-normal steps, then the standard decode pass (SC zeroed, t = 1).
Per n it reports:
- t_n;
- dis = P(decode(z_n) ≠ decode(x̂_n)), where x̂_n is the last clean estimate (= the feedback that would be readable at decode);
- agree_full = P(decode(z_n) = the full-trajectory SC-on token);
- the same for decode(x̂_n).
Pre-set rule (TODO 3a-iii, written before running): the truncated t_end = the largest t_n with dis ≥ 0.2, i.e. state and
feedback disagree at decode, so readable feedback could add information."""
import argparse, os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_instr import load_elf, run_plain, decode
ap = argparse.ArgumentParser(); ap.add_argument("--hcfrl", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=64); ap.add_argument("--L", type=int, default=128); ap.add_argument("--out", default="results/elf_trunc_calib.json")
a = ap.parse_args(); dev = torch.device("cpu"); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
model, cfg, tok = load_elf(a.hcfrl, a.ckpt, a.L, dev)
from utils.sampling_utils import get_sampling_steps, net_out_to_v_x
torch.manual_seed(1); ts = get_sampling_steps(32, time_schedule="logit_normal").to(dev)
z0 = torch.randn((a.n, a.L, model.text_encoder_dim), generator=torch.Generator().manual_seed(7)) * cfg.denoiser_noise_scale
full = decode(model, cfg, run_plain(model, cfg, z0, ts, True), ts[-1], 1.0)
res = []
for n in (8, 12, 16, 20, 24, 28, 32):
    z = z0.clone(); xp = torch.zeros_like(z); wt = torch.full((a.n,), 1.0)
    with torch.no_grad():
        for i in range(n):
            t, tn = ts[i].item(), ts[i + 1].item(); tb = torch.full((a.n,), t)
            out = model(torch.cat([z, xp], -1), tb, deterministic=True, self_cond_cfg_scale=wt)
            v, x = net_out_to_v_x(out, z, tb, cfg.t_eps); z = z + (tn - t) * v; xp = x
    yz = decode(model, cfg, z, ts[-1], 1.0); yx = decode(model, cfg, xp, ts[-1], 1.0)
    r = {"n": n, "t_n": ts[n].item(), "dis": (yz != yx).float().mean().item(),
         "agree_full_state": (yz == full).float().mean().item(), "agree_full_fb": (yx == full).float().mean().item()}
    res.append(r); print(r, flush=True)
ok = [r for r in res if r["dis"] >= 0.2]
pick = max(ok, key=lambda r: r["t_n"]) if ok else None
print("PICK (largest t_n with dis >= 0.2):", pick)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump({"rows": res, "pick": pick}, open(a.out, "w"))
