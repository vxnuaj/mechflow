"""Phase 13 step 2g (EXPLORATORY): when do toy states and feedback become informative, relative to commitment?
For SC-on sampling (t_end 1.0, decode SC as trained), at every flow step and position, it measures:
  acc_state = P(the mode implied by state x — nearer of the two mode-token embeddings by dot product — equals the final decoded mode)
  acc_fb    = the same for the feedback (clean estimate)
  commit    = the fraction of positions whose estimate-implied mode already equals the final mode for all remaining steps.
Usage: python toy_diag.py runs_T2"""
import glob, sys, torch
from toy_access import Toy, make_modes, V_DEF, K_DEF, L
torch.set_num_threads(8)
@torch.no_grad()
def run(m, modes, dec_read, S=16, reps=32, seed=99):
    K = len(modes); k = torch.arange(K).repeat(reps); B = len(k); d = m.E.weight.shape[1]
    mt = torch.tensor(modes)[k]                                   # [B, 2, L]
    Ea, Eb = m.E(mt[:, 0]), m.E(mt[:, 1])                         # [B, L, d]
    side = lambda v: ((v * Eb).sum(-1) > (v * Ea).sum(-1)).long()  # 0 = mode a, 1 = mode b
    g = torch.Generator().manual_seed(seed); x = torch.randn(B, L, d, generator=g); grid = torch.linspace(0, 1, S + 1); fb = None
    xs, es = [], []
    for i in range(S):
        t0, t1 = grid[i].item(), grid[i + 1].item()
        _, est = m(k, x, torch.full((B,), t0), fb); fb = est
        xs.append(side(x)); es.append(side(est))
        x = x + (t1 - t0) / max(1e-6, 1 - t0) * (est - x)
    lg, _ = m(k, x, torch.ones(B), fb if dec_read else None, decode=True); y = lg.argmax(-1)
    fin = torch.where(y == mt[:, 1], 1, torch.where(y == mt[:, 0], 0, -1)); ok = fin >= 0
    acc_s = [((s == fin) & ok).float().sum().item() / ok.sum().item() for s in xs]
    acc_e = [((e == fin) & ok).float().sum().item() / ok.sum().item() for e in es]
    stay = torch.ones_like(fin, dtype=torch.bool); com = []
    for e in reversed(es): stay = stay & (e == fin); com.append(((stay) & ok).float().sum().item() / ok.sum().item())
    return acc_s, acc_e, list(reversed(com)), grid[:S].tolist()
agg = {}
for f in sorted(glob.glob(f"{sys.argv[1]}/*.pt")):
    ck = torch.load(f); m = Toy(V_DEF, K_DEF, ck["form"]); m.load_state_dict(ck["state"]); m.eval()
    r = run(m, make_modes(V_DEF, K_DEF, ck["seed"]), ck["access"] == "read")
    agg.setdefault(ck["form"], []).append(r)
for form, rs in agg.items():
    n = len(rs); S = len(rs[0][0]); t = rs[0][3]
    print(f"== recipe {form} ({n} models, SC-on, t_end 1.0)\n step   t     acc_state  acc_feedback  committed")
    for i in range(S):
        print(f" {i:2d}   {t[i]:.3f}   {sum(r[0][i] for r in rs)/n:.3f}      {sum(r[1][i] for r in rs)/n:.3f}        {sum(r[2][i] for r in rs)/n:.3f}")
