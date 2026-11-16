"""Confirmatory analysis (prereg §2, §6, §7). Written and hashed BEFORE any confirmation data existed.

Usage:  python analysis/confirm.py RESULTS_DIR      (RESULTS_DIR holds conf_p1, conf_p2, conf_p2x, conf_h2n, conf_s12, conf_s1_7b)

Rules implemented (prereg §6):
  * cluster bootstrap, 20,000 resamples (pairs for P1/S1/S11; documents otherwise); seed 20261113
  * primaries: BCa intervals; one-sided p = fraction θ* ≤ m, BCa-adjusted; Holm over {P1, P2, H2-N} at α = 0.05
      "rejected" if the one-sided 95% (BCa) upper bound < m; else "inconclusive"
  * P2 co-requirement: Mistral SC effect on coh_{2,3} one-sided 95% lower bound > 0 AND own-AR coh_{2,3} point > 0,
      else "judge-dependent" (JD) if P2's test passes
  * H2-N: "not estimable" (counts as not supported, p := 1) unless T's one-sided 95% lower bound > 0 and the
      Shapley N share point estimate ≥ 0.20
  * secondaries: percentile intervals; BH q = 0.10 over {S1, S3, S4, S11, S12}
      S1  : 7B, θ = h_flow(SC) − h_AR, m = 0.05
      S3  : TOST of I = Δdep(g8, n0.1) − Δdep(g1, n0.1) within ±0.25; p = max(P*(I* ≤ −0.25), P*(I* ≥ +0.25))
      S4  : D = Δdep(g1, n0.1) − Δdep(g1, n4.0) < +0.25; p = P*(D* ≥ 0.25)
      S11 : ladder rank other = 0 < hybrid = 1 < valid mode = 2; per prompt up = [rank(SC) > rank(NO)],
            down = [rank(SC) < rank(NO)]; θ = mean(up − down), m = 0
      S12 : H2-N contrast in the paper regime, m = 0
  * complete-case: a unit is analysed only if all its paired values are finite; exclusion counts are printed
  * classification of P1/S1/S11 outputs: the audited text-level classifier (src/classify_text.py)
"""
import glob, itertools, json, math, os, sys

import numpy as np
from scipy.stats import norm

B = 20000
SEED = 20261113
ROOT = sys.argv[1] if len(sys.argv) > 1 else "src/results"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def load(name):
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROOT, name, "shard_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            rows[r["task_id"]] = r["result"]          # de-duplicates resubmitted tasks
    return rows


# ---------- cluster bootstrap of a ratio of sums: θ = Σ num / Σ den over clusters ----------
def boot(num, den, rng):
    n = len(num)
    out = np.empty(B)
    for s in range(0, B, 1000):
        c = rng.multinomial(n, np.full(n, 1.0 / n), size=min(1000, B - s)).astype(np.float64)
        out[s:s + len(c)] = (c @ num) / (c @ den)
    return out


def jack(num, den):
    n = len(num); S, D = num.sum(), den.sum()
    return (S - num) / (D - den)


def summarize(num, den, m, rng, bca, lower_tail=True):
    num, den = np.asarray(num, float), np.asarray(den, float)
    est = num.sum() / den.sum()
    th = boot(num, den, rng)
    if bca:
        z0 = norm.ppf(np.clip((th < est).mean(), 1e-6, 1 - 1e-6))
        jk = jack(num, den); d = jk.mean() - jk
        a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5 + 1e-300)
        def q(alpha):
            za = norm.ppf(alpha); u = z0 + (z0 + za) / (1 - a * (z0 + za))
            return np.quantile(th, norm.cdf(u))
        lo95, hi95, lo1s, hi1s = q(0.025), q(0.975), q(0.05), q(0.95)
        p0 = np.clip((th <= m).mean(), 1e-6, 1 - 1e-6)
        w = norm.ppf(p0) - z0
        p = float(norm.cdf(w / (1 + a * w) - z0)) if (th <= m).mean() > 0 else 0.0
    else:
        lo95, hi95, lo1s, hi1s = np.quantile(th, [0.025, 0.975, 0.05, 0.95])
        p = float((th <= m).mean())
    return dict(est=est, ci=(lo95, hi95), lb1s=lo1s, ub1s=hi1s, p=p, th=th, n=len(num))


def fmt(r, scale=1.0, unit=""):
    return f"{r['est']*scale:+.3f}{unit} [{r['ci'][0]*scale:+.3f}, {r['ci'][1]*scale:+.3f}] (n = {r['n']}, p = {r['p']:.2g})"


# ---------- P1-type data ----------
def p1_units(name, tok, ctasks):
    from classify_text import classify_text
    R = load(name)
    rank = {"other": 0, "hybrid": 1, "mode_a": 2, "mode_b": 2}
    by_pair = {}
    for tid, r in R.items():
        t = ctasks[tid]
        lab = {c: classify_text(tok, r["texts"][c], t["mode_a"], t["mode_b"], t["tok_a"], t["tok_b"]) for c in ("AR", "FLOW_SC", "FLOW_NO")}
        d = by_pair.setdefault(t["pair_id"], dict(hyb=0.0, lad=0.0, n=0, hsc=0, hno=0, har=0))
        d["hyb"] += (lab["FLOW_SC"] == "hybrid") - (lab["AR"] == "hybrid")
        d["lad"] += np.sign(rank[lab["FLOW_SC"]] - rank[lab["FLOW_NO"]])
        d["hsc"] += lab["FLOW_SC"] == "hybrid"; d["hno"] += lab["FLOW_NO"] == "hybrid"; d["har"] += lab["AR"] == "hybrid"
        d["n"] += 1
    P = list(by_pair.values())
    return P, len(R)


# ---------- 2x2x2 ----------
def h2n_units(name, metric="dep"):
    R = load(name)
    num_c, num_T, num_shN, n_units, excl, ident = [], [], [], [], 0, [0, 0]
    for tid, r in R.items():
        ident[0] += bool(r["identity_on"]); ident[1] += bool(r["identity_off"])
        c, C, Tt, S, k = r["cells"], 0.0, 0.0, 0.0, 0
        ok = True
        for j in (2, 3):
            v = {x: c[f"j{j}_O{x[0]}N{x[1]}F{x[2]}"][metric] for x in itertools.product((0, 1), repeat=3)}
            if not all(np.isfinite(list(v.values()))):
                ok = False; break
            b = v[(0, 0, 0)]; T = v[(1, 1, 1)] - b; mN = v[(0, 1, 0)] - b
            val = lambda Sset: v[tuple(int(p in Sset) for p in "ONF")] - b
            shN = sum(math.factorial(kk) * math.factorial(2 - kk) / 6 * (val(set(Ss) | {"N"}) - val(set(Ss)))
                      for kk in range(3) for Ss in itertools.combinations("OF", kk))
            C += mN - 0.2 * T; Tt += T; S += shN; k += 1
        if not ok:
            excl += 1; continue
        num_c.append(C); num_T.append(Tt); num_shN.append(S); n_units.append(k)
    return dict(c=np.array(num_c), T=np.array(num_T), shN=np.array(num_shN), k=np.array(n_units, float),
                n=len(R), excl=excl, ident=ident)


def avg23(x):
    return (x[2] + x[3]) / 2


def main():
    rng = np.random.default_rng(SEED)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("lafa-hf/lafa-0.5B", revision="18deca3522f7884898a58d0fb2d2afb3fc91f87e")
    ct = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "..", "data", "bank.jsonl"))}
    res, prim = {}, {}

    # ---- P1 (+ S11) ----
    P, n = p1_units("conf_p1", tok, ct)
    nn = np.array([d["n"] for d in P], float)
    r = summarize([d["hyb"] for d in P], nn, 0.05, rng, bca=True); prim["P1"] = r
    print(f"P1  h_flow − h_AR = {fmt(r, 100, ' pp')}  [prompts {n}, pairs {len(P)}]")
    print(f"    hybrid rates: flow-SC {sum(d['hsc'] for d in P)/nn.sum():.3f}, flow-noSC {sum(d['hno'] for d in P)/nn.sum():.3f}, AR {sum(d['har'] for d in P)/nn.sum():.3f}")
    res["S11"] = summarize([d["lad"] for d in P], nn, 0.0, rng, bca=False)
    print(f"S11 ladder up − down = {fmt(res['S11'])}")

    # ---- P2 ----
    R2 = load("conf_p2")
    U = [(avg23(r["SC"]["dep"]) - avg23(r["NO"]["dep"]), avg23(r["SC"]["ext"]) - avg23(r["NO"]["ext"]),
          avg23(r["SC"]["coh"]) - avg23(r["NO"]["coh"])) for r in R2.values()]
    good = [u for u in U if np.all(np.isfinite(u))]
    G = np.array(good); one = np.ones(len(G))
    r = summarize(G[:, 0], one, 0.10, rng, bca=True); prim["P2"] = r
    ext = summarize(G[:, 1], one, 0.0, rng, bca=True); own = G[:, 2].mean()
    prim["P2"]["coreq"] = bool(ext["lb1s"] > 0 and own > 0)
    print(f"P2  Δdep_23 = {fmt(r)}  [docs {len(R2)}, excluded {len(U) - len(good)}]")
    print(f"    co-req: Mistral Δcoh_23 = {fmt(ext)} one-sided LB {ext['lb1s']:+.3f}; own-AR Δcoh_23 = {own:+.3f} → {'met' if prim['P2']['coreq'] else 'NOT met'}")

    # ---- H2-N ----
    H = h2n_units("conf_h2n")
    r = summarize(H["c"], H["k"], 0.0, rng, bca=True)
    Tr = summarize(H["T"], H["k"], 0.0, rng, bca=True)
    shN = H["shN"].sum() / H["T"].sum()
    r["estimable"] = bool(Tr["lb1s"] > 0 and shN >= 0.20); prim["H2N"] = r
    print(f"H2-N contrast N − 0.2T (dep_23) = {fmt(r)}  [docs {H['n']}, excluded {H['excl']}, identity on/off {H['ident']}]")
    print(f"    T = {fmt(Tr)}; Shapley N share = {shN:.3f} → {'estimable' if r['estimable'] else 'NOT ESTIMABLE'}")
    Hc = h2n_units("conf_h2n", "coh"); rc = summarize(Hc["c"], Hc["k"], 0.0, rng, bca=False)
    print(f"    (descriptive) coh contrast = {fmt(rc)}; Shapley N share (coh) = {Hc['shN'].sum()/Hc['T'].sum():.3f}")

    # ---- Holm over primaries ----
    p_eff = {"P1": prim["P1"]["p"], "P2": prim["P2"]["p"], "H2N": prim["H2N"]["p"] if prim["H2N"]["estimable"] else 1.0}
    order = sorted(p_eff, key=p_eff.get); sup, alive = {}, True
    for i, k in enumerate(order):
        ok = alive and p_eff[k] <= 0.05 / (3 - i)
        sup[k] = ok; alive = alive and ok
    print("\n== PRIMARY (Holm, α = 0.05) ==")
    for k, m in (("P1", 0.05), ("P2", 0.10), ("H2N", 0.0)):
        rr = prim[k]
        if k == "H2N" and not rr["estimable"]:
            v = "n/e (not estimable)"
        elif sup[k]:
            v = "SUPPORTED" if not (k == "P2" and not rr["coreq"]) else "JUDGE-DEPENDENT"
        elif rr["ub1s"] < m:
            v = "REJECTED"
        else:
            v = "inconclusive"
        print(f"  {k:4s} p = {p_eff[k]:.3g}  one-sided 95% bounds [{rr['lb1s']:+.4f}, {rr['ub1s']:+.4f}] vs m = {m} → {v}")

    # ---- secondaries ----
    P7, n7 = p1_units("conf_s1_7b", tok, ct); n7c = np.array([d["n"] for d in P7], float)
    res["S1"] = summarize([d["hyb"] for d in P7], n7c, 0.05, rng, bca=False)
    print(f"\nS1  7B h_flow − h_AR = {fmt(res['S1'], 100, ' pp')}")
    RX = load("conf_p2x")
    I, D = [], []

    for tid, x in RX.items():
        if tid not in R2:
            continue
        pr = avg23(R2[tid]["SC"]["dep"]) - avg23(R2[tid]["NO"]["dep"])
        g1 = avg23(x["G1N01_SC"]["dep"]) - avg23(x["G1N01_NO"]["dep"])
        n4 = avg23(x["G1N4_SC"]["dep"]) - avg23(x["G1N4_NO"]["dep"])
        if np.all(np.isfinite([pr, g1, n4])):
            I.append(pr - g1); D.append(g1 - n4)
    if len(I) < 2:
        print("S3/S4: NO DATA → counted as not supported (p = 1)")
        res["S3"] = res["S4"] = {"p": 1.0}
    else:
        o = np.ones(len(I))
        s3 = summarize(I, o, -0.25, rng, bca=False); s3["p"] = max(float((s3["th"] <= -0.25).mean()), float((s3["th"] >= 0.25).mean()))
        res["S3"] = s3
        s4 = summarize(D, o, 0.25, rng, bca=False); s4["p"] = float((s4["th"] >= 0.25).mean()); res["S4"] = s4
        print(f"S3  SC × g interaction (TOST ±0.25) = {fmt(s3)}")
        print(f"S4  SC effect n0.1 − n4.0 (< +0.25) = {fmt(s4)}")
    H12 = h2n_units("conf_s12"); res["S12"] = summarize(H12["c"], H12["k"], 0.0, rng, bca=False)
    print(f"S12 PR contrast N − 0.2T = {fmt(res['S12'])}  [Shapley N share {H12['shN'].sum()/H12['T'].sum():.3f}; identity {H12['ident']}/{H12['n']}]")
    ks = sorted(["S1", "S3", "S4", "S11", "S12"], key=lambda k: res[k]["p"]); mlen = len(ks); kmax = 0
    for i, k in enumerate(ks, 1):
        if res[k]["p"] <= 0.10 * i / mlen:
            kmax = i
    print("== SECONDARY (BH, q = 0.10) ==")
    for i, k in enumerate(ks, 1):
        print(f"  {k:4s} p = {res[k]['p']:.3g} → {'supported' if i <= kmax else 'not supported'}")


if __name__ == "__main__":
    main()
