"""Family T v3 calibration (regime diagnostics only; no O/N/F decomposition). Calibration seeds 100–101.
Per sigma (averaged over the calibration models), it reports:
- the commitment curve and state accuracy (SC-on, decode SC as trained, full grid);
- t90 = the first t at which committed ≥ 0.9;
- acc_state at the first step where committed ≥ 0.5;
- commit-at-decode for decode SC zeroed, at t_end ∈ {0.25, 0.375, 0.5, 0.75, 1.0}.
Usage: python toy_calib.py runs_cal"""
import glob, sys, torch
from toy_access import Toy, make_modes, V_DEF, K_DEF, L, sample_plain
torch.set_num_threads(8)
@torch.no_grad()
def curve(m, modes, dec_read, sigma, S=16, reps=32, seed=99):
    K = len(modes); k = torch.arange(K).repeat(reps); B = len(k); d = m.E.weight.shape[1]
    mt = torch.tensor(modes)[k]; Ea, Eb = m.E(mt[:, 0]), m.E(mt[:, 1])
    side = lambda v: ((v * Eb).sum(-1) > (v * Ea).sum(-1)).long()
    g = torch.Generator().manual_seed(seed); x = torch.randn(B, L, d, generator=g) * sigma; grid = torch.linspace(0, 1, S + 1); fb = None; xs, es = [], []
    for i in range(S):
        t0, t1 = grid[i].item(), grid[i + 1].item(); _, est = m(k, x, torch.full((B,), t0), fb); fb = est
        xs.append(side(x)); es.append(side(est)); x = x + (t1 - t0) / max(1e-6, 1 - t0) * (est - x)
    lg, _ = m(k, x, torch.ones(B), fb if dec_read else None, decode=True); y = lg.argmax(-1)
    fin = torch.where(y == mt[:, 1], 1, torch.where(y == mt[:, 0], 0, -1)); ok = fin >= 0
    acc = [((s == fin) & ok).float().sum().item() / max(1, ok.sum().item()) for s in xs]
    stay = torch.ones_like(fin, dtype=torch.bool); com = []
    for e in reversed(es): stay = stay & (e == fin); com.append((stay & ok).float().sum().item() / max(1, ok.sum().item()))
    valid = sum(1 for r, kk in zip(y.tolist(), k.tolist()) if r in (modes[kk][0], modes[kk][1])) / B
    return acc, list(reversed(com)), grid[:S].tolist(), valid
res = {}
for f in sorted(glob.glob(f"{sys.argv[1]}/*.pt")):
    ck = torch.load(f); m = Toy(V_DEF, K_DEF, ck["form"]); m.load_state_dict(ck["state"]); m.eval(); sg = ck["sigma"]; modes = make_modes(V_DEF, K_DEF, ck["seed"])
    acc, com, t, val = curve(m, modes, ck["access"] == "read", sg)
    k = torch.arange(K_DEF).repeat(32); cad = {}
    for te in (0.25, 0.375, 0.5, 0.75, 1.0):
        y, pre = sample_plain(m, k, True, False, 16, te, 99, sigma=sg); cad[te] = (y != pre).float().mean().item()
    off_valid = None
    yo, _ = sample_plain(m, k, False, ck["access"] == "read", 16, 1.0, 99, sigma=sg)
    off_valid = sum(1 for r, kk in zip(yo.tolist(), k.tolist()) if r in (modes[kk][0], modes[kk][1])) / len(k)
    res.setdefault(sg, []).append((ck["access"], ck["form"], acc, com, t, cad, val, off_valid))
for sg in sorted(res):
    rs = res[sg]; n = len(rs); t = rs[0][4]
    acc = [sum(r[2][i] for r in rs) / n for i in range(16)]; com = [sum(r[3][i] for r in rs) / n for i in range(16)]
    t90 = next((t[i] for i in range(16) if com[i] >= 0.9), None); i50 = next((i for i in range(16) if com[i] >= 0.5), None)
    print(f"== sigma {sg} ({n} models)  t90 = {t90}  acc_state at 50% committed (step {i50}, t {t[i50] if i50 is not None else None}) = {acc[i50] if i50 is not None else None}")
    print("   committed by step: " + " ".join(f"{c:.2f}" for c in com))
    print("   acc_state by step: " + " ".join(f"{a:.2f}" for a in acc))
    for te in (0.25, 0.375, 0.5, 0.75, 1.0):
        print(f"   commit-at-decode (decode SC zeroed) t_end {te}: " + "  ".join(f"{r[0]}/{r[1]} {r[5][te]:.3f}" for r in rs))
    print("   valid SC-on / SC-off: " + "  ".join(f"{r[0]}/{r[1]} {r[6]:.2f}/{r[7]:.2f}" for r in rs))
