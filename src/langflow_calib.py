"""Phase 13 step 4 (regime diagnostics only, no O/N/F data; noise seed 7) + routing smoke test.
SC-on sampling (K = 128 native steps, L = 1024 native). Every 8 steps, measured against the final SC-on tokens y:
  lp_fb    = mean log p(y) under the step's own logits (what the feedback x̂ = E·softmax(logits) encodes);
  lp_state = mean log p(y) when the current state z is read as clean (forward at gamma_min, no SC), the analogue of
             ELF's decode(state);
  acc_*    = the argmax versions;
  committed = the fraction of positions whose per-step argmax already equals y at all later recorded steps;
  both globally and for the |i − j| ≤ 16 window around j = L/2.
Smoke test: O1N1F1 / O0N0F0 routed cells vs plain sampling at K = 16, n = 2 (token agreement, logit max-diff)."""
import argparse, json, os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from langflow_instr import load_langflow, schedule, sample_plain, sample_cell, _fwd
ap = argparse.ArgumentParser(); ap.add_argument("--path", required=True); ap.add_argument("--n", type=int, default=16)
ap.add_argument("--L", type=int, default=1024); ap.add_argument("--K", type=int, default=128); ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/langflow_calib.json")
a = ap.parse_args(); dev = torch.device(a.device); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
m = load_langflow(a.path, dev); D = m.config.hidden_size; j = a.L // 2
win = torch.tensor([i for i in range(a.L) if abs(i - j) <= 16 and i != j])
# smoke identities
gs = schedule(m, 16, dev); z0s = torch.randn(2, 128, D, generator=torch.Generator().manual_seed(3)).to(dev)
yon, *_ = sample_plain(m, z0s, gs, True); yoff, *_ = sample_plain(m, z0s, gs, False)
y111 = sample_cell(m, z0s, gs, 64, 1, 1, 1); y000 = sample_cell(m, z0s, gs, 64, 0, 0, 0)
smoke = {"ident_111": (y111 == yon).float().mean().item(), "ident_000": (y000 == yoff).float().mean().item()}
print("SMOKE", smoke, flush=True)
# regime
gam = schedule(m, a.K, dev); z0 = torch.randn(a.n, a.L, D, generator=torch.Generator().manual_seed(7)).to(dev)
y, _, _, rec = sample_plain(m, z0, gam, True, record=True)
yoff_full, *_ = sample_plain(m, z0, gam, False)
rows = []; arg_hist = []
for i in range(0, len(rec), 8):
    z, xp = rec[i]
    with torch.no_grad():
        lg_fb = _fwd(m, z, gam[i], rec[i - 1][1] if i > 0 else None)         # the logits that produced x̂_i (SC = x̂_{i−1})
        lg_st = _fwd(m, z, gam[-1], None)                                    # the state read as clean
    lpf = torch.log_softmax(lg_fb, -1).gather(-1, y[..., None])[..., 0]; lps = torch.log_softmax(lg_st, -1).gather(-1, y[..., None])[..., 0]
    af = (lg_fb.argmax(-1) == y).float(); as_ = (lg_st.argmax(-1) == y).float(); arg_hist.append(lg_fb.argmax(-1))
    rows.append({"step": i, "gamma": gam[i].item(), "lp_fb": lpf.mean().item(), "lp_state": lps.mean().item(),
                 "lp_fb_win": lpf[:, win].mean().item(), "lp_state_win": lps[:, win].mean().item(),
                 "acc_fb": af.mean().item(), "acc_state": as_.mean().item()})
stay = torch.ones_like(y, dtype=torch.bool)
for r, am in zip(reversed(rows), reversed(arg_hist)):
    stay = stay & (am == y); r["committed"] = stay.float().mean().item(); r["committed_win"] = stay[:, win].float().mean().item()
for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
mismatch = (y != yoff_full).float().mean().item()
print(f"SC-on vs SC-off token disagreement (full length): {mismatch:.3f}")
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump({"smoke": smoke, "rows": rows, "sc_on_off_disagree": mismatch}, open(a.out, "w"))
