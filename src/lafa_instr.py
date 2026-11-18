"""Instrumented single-block sampler for Lafa (Phase 6).

Mirrors FlowModelWrapper._euler_denoise_block (snapshot 18deca35, flow_model.py ~997-1256)
for the default inference path (per-position warp grid, no SDE, euler_alpha=1, no logit_shift,
ode_argmax=False), and adds intervention hooks:

  * sc_fn(step, branch, z, e_prev, ctx) -> e_in
        builds the denoiser input from state z and fed-back expected embedding e_prev.
        Default = model._apply_self_cond(z, e_prev) (or z if SC disabled / e_prev None).
  * kv_fn(step, branch, e_in, ctx) -> None | (e_in_alt, src)
        per-query K/V selection: run an extra "alt" pass on e_in_alt first; then in the live
        pass, query position qi reads key/value of block position kj from the alt pass iff
        src[qi, kj] is True (else from the live pass). Prefix keys are shared.
  * out_fn(step, branch, logits_live, logits_alt, ctx) -> logits used downstream
        e.g. take some positions' outputs from the alt pass ("frozen streams").
  * record: per-step tensors (state, fed-back embedding, probs top-k, chosen token probs).

Branch is "c" (conditional) or "u" (prefix-free CFG branch).
Everything is batch size 1, as in the reference implementation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class BlockCfg:
    num_steps: int = 32
    cfg_scale: float = 8.0
    noise: float = 0.1
    self_cond: bool = True
    seed: int = 42
    grid_mode: str = "warp"


@dataclass
class BlockResult:
    tokens: list
    logits: torch.Tensor                      # [bd, V] final conditional logits (fp32, cpu)
    n_fwd: int
    rec: dict = field(default_factory=dict)   # per-step records


def trim(cache, n):
    """Remove the last n entries from a DynamicCache (mirrors cache.trim_kv_cache)."""
    if n <= 0:
        return
    # transformers 5.x DynamicCache exposes .layers with .keys/.values
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None and layer.keys.numel():
                layer.keys = layer.keys[..., :-n, :]
                layer.values = layer.values[..., :-n, :]
    else:  # older API
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :-n, :]
            cache.value_cache[i] = cache.value_cache[i][..., :-n, :]


class Sampler:
    def __init__(self, model):
        self.m = model
        self.E = model.embed_tokens.weight
        self.bd = model.bd_size
        self.dtype = self.E.dtype
        self.dev = self.E.device
        try:  # use the bundled trim if importable (exact reference behaviour)
            import importlib
            mod = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".cache")
            self._trim = lambda c, n: mod.trim_kv_cache(c, n)
        except Exception:
            self._trim = trim

    # ---------- prefix ----------
    @torch.no_grad()
    def prefill(self, prompt_ids: list[int]):
        ids = torch.tensor([prompt_ids], device=self.dev)
        emb = self.m.embed_tokens(ids)
        t_bn = self.m._compute_t_bn(torch.ones(1, ids.shape[1], device=self.dev, dtype=self.dtype))
        _, cache, _ = self.m._forward_backbone(emb, t_bn, attn_mask=None, use_cache=True)
        return cache, ids.shape[1]

    # ---------- one forward with optional per-query K/V selection ----------
    def _fwd(self, e_in, t_bn, base_mask, cache, pos_ids, alt=None):
        """Returns (logits_live [bd,V] fp32, logits_alt or None, n_fwd)."""
        cache_len = cache.get_seq_length()
        bd = self.bd
        if alt is None:
            lg, _, _ = self.m._forward_backbone(e_in, t_bn, base_mask, past_key_values=cache,
                                                use_cache=False, position_ids=pos_ids)
            self._trim(cache, cache.get_seq_length() - cache_len)
            return lg[0].float(), None, 1
        e_alt, src = alt                       # src: bool [bd(query), bd(key)] -> read alt
        lg_a, _, _ = self.m._forward_backbone(e_alt, t_bn, base_mask, past_key_values=cache,
                                              use_cache=False, position_ids=pos_ids)
        # cache now holds prefix + alt block K/V; build live mask over prefix|alt|live
        pref = base_mask[..., :cache_len]
        neg = float("-inf")
        within = base_mask[..., cache_len:cache_len + bd]   # [1,1,bd,bd] (zeros if bidirectional)
        src = src.to(self.dev)
        alt_cols = torch.where(src[None, None], within, torch.full_like(within, neg))
        live_cols = torch.where(~src[None, None], within, torch.full_like(within, neg))
        mask = torch.cat([pref, alt_cols, live_cols], dim=-1)
        lg, _, _ = self.m._forward_backbone(e_in, t_bn, mask, past_key_values=cache,
                                            use_cache=False, position_ids=pos_ids)
        self._trim(cache, cache.get_seq_length() - cache_len)
        return lg[0].float(), lg_a[0].float(), 2

    # ---------- the block ----------
    @torch.no_grad()
    def run_block(self, cache, block_start: int, cfg: BlockCfg, sc_fn=None, kv_fn=None,
                  out_fn=None, record: bool = False, rec_topk: int = 0, ctx: dict | None = None,
                  init_noise: torch.Tensor | None = None, track_uncond: bool = False):
        """track_uncond: also run the prefix-free branch when cfg_scale == 1 (its predictions and
        fed-back state are tracked for candidate (c), but it is NOT used for guidance)."""
        m, E, bd, dt, dev = self.m, self.E, self.bd, self.dtype, self.dev
        ctx = ctx if ctx is not None else {}
        saved_noise, saved_sc = m.noise_scale_mult, m.enable_self_cond
        m.noise_scale_mult = float(cfg.noise)
        m.enable_self_cond = bool(cfg.self_cond) and saved_sc
        orig_attn = getattr(m.backbone.config, "_attn_implementation", None)
        m.backbone.config._attn_implementation = "eager"
        try:
            torch.manual_seed(cfg.seed)
            grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2)
            per_pos = grid.dim() == 2
            assert per_pos, "only the per-position warp grid is mirrored"
            K = grid.shape[-1] - 1
            pos_ids = torch.arange(block_start, block_start + bd, device=dev).unsqueeze(0)
            cache_len = cache.get_seq_length()
            base_mask = torch.zeros(1, 1, bd, cache_len + bd, dtype=dt, device=dev)
            use_cfg = abs(cfg.cfg_scale - 1.0) > 1e-6
            umask = None
            run_u = use_cfg or track_uncond
            if run_u:
                umask = base_mask.clone()
                umask[..., :cache_len] = float("-inf")
            masks = {"c": base_mask, "u": umask}
            n_fwd = 0

            e_t = (torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult
                   if init_noise is None else init_noise.to(dev, dt))
            sc_on = m.enable_self_cond
            esc = {"c": None, "u": None}
            if sc_on:
                t0 = grid[:, 0].to(dt)[None, :]
                tb0 = m._compute_t_bn(t0)
                for br in (["c", "u"] if run_u else ["c"]):
                    lg, _, nf = self._fwd(e_t, tb0, masks[br], cache, pos_ids)
                    n_fwd += nf
                    esc[br] = (F.softmax(lg, -1).to(dt) @ E).unsqueeze(0).detach()
            ctx["bootstrap_esc"] = {k: (v.clone() if v is not None else None) for k, v in esc.items()}

            rec = {k: [] for k in ("z", "esc_c", "esc_u", "topk_c", "t")} if record else {}
            logits_c = None
            for step in range(K):
                t_fwd = grid[:, step]
                t_out = grid[:, step + 1]
                z = e_t
                tb = m._compute_t_bn(t_fwd.to(dt)[None, :])
                probs = {}
                for br in (["c", "u"] if run_u else ["c"]):
                    if sc_fn is not None:
                        e_in = sc_fn(step, br, z, esc[br], ctx)
                    else:
                        e_in = m._apply_self_cond(z, esc[br]) if (sc_on and esc[br] is not None) else z
                    alt = kv_fn(step, br, e_in, ctx) if kv_fn is not None else None
                    lg, lg_alt, nf = self._fwd(e_in, tb, masks[br], cache, pos_ids, alt=alt)
                    n_fwd += nf
                    if out_fn is not None:
                        lg = out_fn(step, br, lg, lg_alt, ctx)
                    probs[br] = F.softmax(lg, dim=-1).to(dt)
                    if br == "c":
                        logits_c = lg
                if record:
                    rec["z"].append(z[0].float().cpu())
                    rec["esc_c"].append(esc["c"][0].float().cpu() if esc["c"] is not None else None)
                    rec["esc_u"].append(esc["u"][0].float().cpu() if esc["u"] is not None else None)
                    rec["t"].append(t_fwd.float().cpu())
                    if rec_topk:
                        v, i = probs["c"].float().topk(rec_topk, dim=-1)
                        rec["topk_c"].append((v.cpu(), i.cpu()))
                if ctx.get("keep_probs"):
                    ctx.setdefault("probs_hist", []).append({k: v.float().cpu() for k, v in probs.items()})
                e_den = (probs["c"] @ E).unsqueeze(0)
                if use_cfg:
                    e_u = (probs["u"] @ E).unsqueeze(0)
                    e_den = e_u + cfg.cfg_scale * (e_den - e_u)
                if sc_on:
                    esc["c"] = (probs["c"] @ E).unsqueeze(0).detach()
                    if run_u:
                        esc["u"] = (probs["u"] @ E).unsqueeze(0).detach()
                ctx["esc"] = esc
                if step == K - 1:
                    break
                tf = t_fwd[None, :, None].float()
                to = t_out[None, :, None].float()
                e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
            tokens = logits_c.argmax(-1).tolist()
            return BlockResult(tokens=tokens, logits=logits_c.cpu(), n_fwd=n_fwd, rec=rec)
        finally:
            m.noise_scale_mult, m.enable_self_cond = saved_noise, saved_sc
            m.backbone.config._attn_implementation = orig_attn


def sc_apply(model, z, e_prev, off_positions=(), candidate="a", repl=None):
    """Self-conditioning input with single-position 'off' at given positions.

    candidate 'a': plain z at those positions (no W1, no W2 term).
    candidate 'b'/'c': replace e_prev rows at those positions with repl rows (bootstrap q /
                       prefix-free q), keep the trained functional form.
    """
    if e_prev is None:
        return z
    if candidate == "a":
        e_in = model._apply_self_cond(z, e_prev)
        if off_positions:
            idx = list(off_positions)
            e_in = e_in.clone()
            e_in[:, idx, :] = z[:, idx, :]
        return e_in
    e2 = e_prev.clone()
    if off_positions:
        idx = list(off_positions)
        e2[:, idx, :] = repl[:, idx, :]
    return model._apply_self_cond(z, e2)


def run_meta(cfg: BlockCfg | None = None, extra: dict | None = None) -> dict:
    """Experiment-tracking header saved with every result file."""
    import hashlib, json, os, platform, socket, datetime, glob
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(here, "*.py"))):
        h.update(open(f, "rb").read())
    meta = {"code_sha256": h.hexdigest(), "host": socket.gethostname(), "python": platform.python_version(),
            "torch": torch.__version__, "time_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "model": f"{os.environ.get('LAFA_REPO', 'lafa-hf/lafa-0.5B')}@{os.environ.get('LAFA_REV', '18deca3522f7884898a58d0fb2d2afb3fc91f87e')}",
            "h2n_regime": os.environ.get("H2N_REGIME"), "p2_regime": os.environ.get("P2_REGIME")}
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["cuda"] = torch.version.cuda
        meta["driver"] = os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip()
    elif torch.backends.mps.is_available():
        meta["gpu"] = "apple-mps"
    meta["platform"] = os.environ.get("MF_PLATFORM", "carc" if os.path.isdir("/scratch1") else "local")
    meta["git_commit"] = os.popen(f"git -C {here} rev-parse HEAD 2>/dev/null").read().strip() or os.environ.get("MF_COMMIT")
    meta["cuts"] = os.environ.get("CUTS"); meta["grid"] = os.environ.get("GRID")
    if cfg is not None:
        meta["cfg"] = cfg.__dict__
    if extra:
        meta.update(extra)
    return meta


@torch.no_grad()
def run_block_two_pass(S: "Sampler", cache, block_start: int, cfg: BlockCfg, fbA: set, fbB: set, take: list):
    """2x2x2 trajectory primitive (phase4 §2.6; no CFG, i.e. cfg_scale must be 1).

    Every step runs two passes on the same states:
      pass A: self-conditioning applied at positions in fbA (others get plain x = candidate (a));
      pass B: self-conditioning applied at positions in fbB.
    Position p's output (its prediction, hence its state update and its next-step feedback) is
    taken from pass take[p] in {"A","B"}. Returns final tokens and per-position final logits.
    Identities: fbA=all, take=all A  ->  normal SC decoding; fbB=empty, take=all B -> SC off.
    """
    m, E, bd, dt, dev = S.m, S.E, S.bd, S.dtype, S.dev
    assert abs(cfg.cfg_scale - 1.0) < 1e-6
    saved_noise, saved_sc = m.noise_scale_mult, m.enable_self_cond
    m.noise_scale_mult = float(cfg.noise)
    orig_attn = getattr(m.backbone.config, "_attn_implementation", None)
    m.backbone.config._attn_implementation = "eager"
    try:
        torch.manual_seed(cfg.seed)
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2)
        K = grid.shape[-1] - 1
        pos_ids = torch.arange(block_start, block_start + bd, device=dev).unsqueeze(0)
        cache_len = cache.get_seq_length()
        base_mask = torch.zeros(1, 1, bd, cache_len + bd, dtype=dt, device=dev)
        e_t = torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult
        # bootstrap feedback (as in the reference: one pass at t=0 without SC)
        tb0 = m._compute_t_bn(grid[:, 0].to(dt)[None, :])
        lg0, _, _ = S._fwd(e_t, tb0, base_mask, cache, pos_ids)
        esc = (F.softmax(lg0, -1).to(dt) @ E).unsqueeze(0)
        idxA = sorted(fbA); idxB = sorted(fbB)
        last = None
        for step in range(K):
            t_fwd, t_out = grid[:, step], grid[:, step + 1]
            z = e_t
            tb = m._compute_t_bn(t_fwd.to(dt)[None, :])
            full = m._apply_self_cond(z, esc)
            outs = {}
            for name, idx in (("A", idxA), ("B", idxB)):
                e_in = z.clone()
                if idx:
                    e_in[:, idx, :] = full[:, idx, :]
                lg, _, _ = S._fwd(e_in, tb, base_mask, cache, pos_ids)
                outs[name] = lg
            lg_mix = torch.stack([outs[take[p]][p] for p in range(bd)], 0)
            probs = F.softmax(lg_mix, dim=-1).to(dt)
            e_den = (probs @ E).unsqueeze(0)
            esc = e_den.detach()
            last = lg_mix
            if step == K - 1:
                break
            tf, to = t_fwd[None, :, None].float(), t_out[None, :, None].float()
            e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
        return last.argmax(-1).tolist(), last.cpu()
    finally:
        m.noise_scale_mult, m.enable_self_cond = saved_noise, saved_sc
        m.backbone.config._attn_implementation = orig_attn


class PerLayerMasks:
    """Context manager: replace the attention mask passed to specific layers' self-attention.

    masks: {layer_idx: tensor [1, H or 1, q, k]} (additive, 0 = allowed, -inf = blocked).
    Uses forward pre-hooks (with_kwargs) on layer.self_attn, so Lafa's patched layer code is untouched.
    """

    def __init__(self, model, masks):
        self.layers = model.backbone.model.layers
        self.masks = masks
        self.handles = []

    def __enter__(self):
        for li, m in self.masks.items():
            def hook(mod, args, kwargs, m=m):
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = m
                return args, kwargs
            self.handles.append(self.layers[li].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_block_headcut(S: "Sampler", cache, block_start: int, cfg: BlockCfg, j: int, cut: dict, reroute: bool = False,
                      steps=None):
    """Head-resolved N cut for target j (MR, cfg_scale must be 1), all three factors on otherwise.

    Each step: pass B = neighbours without SC (j keeps SC); pass A = everyone with SC (normal);
    pass A' = like A, but at layer l, head h in cut[l], query j reads the neighbours' K/V from pass B.
    Outputs: j from A'; neighbours from A (so neighbours evolve normally). cut = {} reproduces normal SC.
    reroute=True runs the B + A' path even with an empty cut: the bf16-matched baseline for comparisons (Phase 6 rule).
    steps: optional set of step indices where the cut applies; at other steps the same B + A' path runs with
    no cut (so every condition shares one code path). None = all steps.
    """
    m, E, bd, dt, dev = S.m, S.E, S.bd, S.dtype, S.dev
    assert abs(cfg.cfg_scale - 1.0) < 1e-6
    saved_noise = m.noise_scale_mult
    m.noise_scale_mult = float(cfg.noise)
    orig_attn = getattr(m.backbone.config, "_attn_implementation", None)
    m.backbone.config._attn_implementation = "eager"
    H = m.backbone.config.num_attention_heads
    nb = [p for p in range(bd) if p != j]
    try:
        torch.manual_seed(cfg.seed)
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2)
        if grid.dim() == 1:                                   # uniform grids: same times for every position
            grid = grid[None].expand(bd, -1)
        K = grid.shape[-1] - 1
        pos_ids = torch.arange(block_start, block_start + bd, device=dev).unsqueeze(0)
        cl = cache.get_seq_length()
        base_mask = torch.zeros(1, 1, bd, cl + bd, dtype=dt, device=dev)
        e_t = torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult
        tb0 = m._compute_t_bn(grid[:, 0].to(dt)[None, :])
        lg0, _, _ = S._fwd(e_t, tb0, base_mask, cache, pos_ids)
        esc = (F.softmax(lg0, -1).to(dt) @ E).unsqueeze(0)
        neg = float("-inf")
        # masks over [prefix | B-block | A'-block]
        default = torch.zeros(1, 1, bd, cl + 2 * bd, dtype=dt, device=dev)
        default[..., cl:cl + bd] = neg                       # nobody reads B by default
        layer_masks = {}
        for li, heads in cut.items():
            mk = default.expand(1, H, bd, cl + 2 * bd).clone()
            for h in heads:
                for p in nb:
                    mk[0, h, j, cl + p] = 0.0                # j reads neighbour p from B ...
                    mk[0, h, j, cl + bd + p] = neg           # ... instead of from A'
            layer_masks[li] = mk
        last = None
        for step in range(K):
            t_fwd, t_out = grid[:, step], grid[:, step + 1]
            z = e_t
            tb = m._compute_t_bn(t_fwd.to(dt)[None, :])
            full = m._apply_self_cond(z, esc)
            eB = z.clone(); eB[:, [j], :] = full[:, [j], :]
            lgA, _, _ = S._fwd(full, tb, base_mask, cache, pos_ids)          # pass A (normal)
            if cut or reroute:
                lgB, _, _ = m._forward_backbone(eB, tb, base_mask, past_key_values=cache, use_cache=False,
                                                position_ids=pos_ids)        # appends B K/V
                with PerLayerMasks(m, layer_masks if (steps is None or step in steps) else {}):
                    lgA2, _, _ = m._forward_backbone(full, tb, default, past_key_values=cache, use_cache=False,
                                                     position_ids=pos_ids)
                S._trim(cache, cache.get_seq_length() - cl)
                lg_j = lgA2[0].float()[j]
            else:
                lg_j = lgA[j]
            lg = lgA.clone(); lg[j] = lg_j
            probs = F.softmax(lg, dim=-1).to(dt)
            e_den = (probs @ E).unsqueeze(0)
            esc = e_den.detach()
            last = lg
            if step == K - 1:
                break
            tf, to = t_fwd[None, :, None].float(), t_out[None, :, None].float()
            e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
        return last.argmax(-1).tolist(), last.cpu()
    finally:
        m.noise_scale_mult = saved_noise
        m.backbone.config._attn_implementation = orig_attn


@torch.no_grad()
def run_block_two_pass_cfg(S: "Sampler", cache, block_start: int, cfg: BlockCfg, fbA: set, fbB: set, take: list):
    """2x2x2 primitive WITH prefix guidance (paper regime). Mirrors _euler_denoise_block with CFG:
    conditional branch (prefix visible) and prefix-free branch (prefix keys masked) each keep their own
    feedback state; in EACH branch, passes A/B apply SC at fbA/fbB and position p takes its output from
    pass take[p]. Guided prediction e = e_u + g (e_c - e_u); next-step feedback = the (mixed) unguided
    predictions of each branch, as in the reference. Readout = argmax of mixed conditional logits.
    Identities: fbA=all, take=all A -> normal SC decoding with CFG; take=all B with fbB=empty -> SC off."""
    m, E, bd, dt, dev = S.m, S.E, S.bd, S.dtype, S.dev
    g = cfg.cfg_scale
    saved_noise = m.noise_scale_mult
    m.noise_scale_mult = float(cfg.noise)
    orig_attn = getattr(m.backbone.config, "_attn_implementation", None)
    m.backbone.config._attn_implementation = "eager"
    try:
        torch.manual_seed(cfg.seed)
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2)
        K = grid.shape[-1] - 1
        pos_ids = torch.arange(block_start, block_start + bd, device=dev).unsqueeze(0)
        cl = cache.get_seq_length()
        cmask = torch.zeros(1, 1, bd, cl + bd, dtype=dt, device=dev)
        umask = cmask.clone(); umask[..., :cl] = float("-inf")
        masks = {"c": cmask, "u": umask}
        e_t = torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult
        tb0 = m._compute_t_bn(grid[:, 0].to(dt)[None, :])
        esc = {}
        for br in ("c", "u"):
            lg0, _, _ = S._fwd(e_t, tb0, masks[br], cache, pos_ids)
            esc[br] = (F.softmax(lg0, -1).to(dt) @ E).unsqueeze(0)
        idxA, idxB = sorted(fbA), sorted(fbB)
        last = None
        for step in range(K):
            t_fwd, t_out = grid[:, step], grid[:, step + 1]
            z = e_t
            tb = m._compute_t_bn(t_fwd.to(dt)[None, :])
            mixed = {}
            for br in ("c", "u"):
                full = m._apply_self_cond(z, esc[br])
                outs = {}
                for name, idx in (("A", idxA), ("B", idxB)):
                    e_in = z.clone()
                    if idx:
                        e_in[:, idx, :] = full[:, idx, :]
                    lg, _, _ = S._fwd(e_in, tb, masks[br], cache, pos_ids)
                    outs[name] = lg
                mixed[br] = torch.stack([outs[take[p]][p] for p in range(bd)], 0)
            pc = F.softmax(mixed["c"], -1).to(dt); pu = F.softmax(mixed["u"], -1).to(dt)
            ec, eu = (pc @ E).unsqueeze(0), (pu @ E).unsqueeze(0)
            e_den = eu + g * (ec - eu)
            esc = {"c": ec.detach(), "u": eu.detach()}
            last = mixed["c"]
            if step == K - 1:
                break
            tf, to = t_fwd[None, :, None].float(), t_out[None, :, None].float()
            e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
        return last.argmax(-1).tolist(), last.cpu()
    finally:
        m.noise_scale_mult = saved_noise
        m.backbone.config._attn_implementation = orig_attn


@torch.no_grad()
def prefill_batched(S: "Sampler", prompt_ids: list, B: int):
    """Prefix KV cache replicated over a batch of B (one batched forward)."""
    ids = torch.tensor([prompt_ids] * B, device=S.dev)
    emb = S.m.embed_tokens(ids)
    t_bn = S.m._compute_t_bn(torch.ones(B, ids.shape[1], device=S.dev, dtype=S.dtype))
    _, cache, _ = S.m._forward_backbone(emb, t_bn, attn_mask=None, use_cache=True)
    return cache, ids.shape[1]


@torch.no_grad()
def run_block_headcut_batched(S: "Sampler", cache_B, block_start: int, cfg: BlockCfg, j: int, conds: list):
    """Batched version of run_block_headcut(..., reroute=True): B = len(conds) conditions in one batch.
    conds: list of (cut: {layer: [heads]}, steps: set or None[, src]) with src in {'nofb','nostate','neither'}
    (what query j reads for the neighbours at cut layers/steps; default 'nofb' = the N cut). cache_B: prefix cache with batch B
    (prefill_batched). Same initial noise for every condition, as in the unbatched runs (same seed).
    Returns [(tokens, logits [bd, V] cpu)] per condition."""
    m, E, bd, dt, dev = S.m, S.E, S.bd, S.dtype, S.dev
    B = len(conds)
    assert abs(cfg.cfg_scale - 1.0) < 1e-6
    saved_noise = m.noise_scale_mult
    m.noise_scale_mult = float(cfg.noise)
    orig_attn = getattr(m.backbone.config, "_attn_implementation", None)
    m.backbone.config._attn_implementation = "eager"
    H = m.backbone.config.num_attention_heads
    nb = [p for p in range(bd) if p != j]
    neg = float("-inf")
    try:
        torch.manual_seed(cfg.seed)
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2)
        if grid.dim() == 1:
            grid = grid[None].expand(bd, -1)
        K = grid.shape[-1] - 1
        pos_ids = torch.arange(block_start, block_start + bd, device=dev).unsqueeze(0).expand(B, -1)
        cl = cache_B.get_seq_length()
        base_mask = torch.zeros(B, 1, bd, cl + bd, dtype=dt, device=dev)
        e_t = (torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult).expand(B, -1, -1).contiguous()
        tb0 = m._compute_t_bn(grid[:, 0].to(dt)[None, :].expand(B, -1))
        lg0, _, _ = m._forward_backbone(e_t, tb0, base_mask, past_key_values=cache_B, use_cache=False, position_ids=pos_ids)
        S._trim(cache_B, cache_B.get_seq_length() - cl)
        esc = F.softmax(lg0.float(), -1).to(dt) @ E
        default = torch.zeros(B, 1, bd, cl + 2 * bd, dtype=dt, device=dev)
        default[..., cl:cl + bd] = neg
        srcs = [c[2] if len(c) > 2 else "nofb" for c in conds]          # what j reads from the neighbours in pass B
        conds = [(c[0], c[1]) for c in conds]
        gN = torch.Generator(device="cpu").manual_seed(7)
        z_noise = (torch.randn(1, bd, m._d_model, generator=gN) * m._noise_sigma * m.noise_scale_mult).to(dev, dt)
        layers = sorted({l for c, _ in conds for l in c})
        last = None
        for step in range(K):
            masks = {}
            for li in layers:
                mk = default.expand(B, H, bd, cl + 2 * bd).clone()
                for b, (c, steps) in enumerate(conds):
                    if li in c and (steps is None or step in steps):
                        for h in c[li]:
                            for p in nb:
                                mk[b, h, j, cl + p] = 0.0
                                mk[b, h, j, cl + bd + p] = neg
                masks[li] = mk
            t_fwd, t_out = grid[:, step], grid[:, step + 1]
            z = e_t
            tb = m._compute_t_bn(t_fwd.to(dt)[None, :].expand(B, -1))
            full = m._apply_self_cond(z, esc)
            eB = z.clone(); eB[:, j, :] = full[:, j, :]
            # neighbour inputs for pass B per condition: 'nofb' = state, no feedback (default N cut);
            # 'nostate' = feedback kept, state replaced by pure noise; 'neither' = noise state, no feedback
            if any(x != "nofb" for x in srcs):
                zn = z_noise.expand(B, -1, -1)
                fulln = m._apply_self_cond(zn, esc)
                for b, x in enumerate(srcs):
                    if x == "nostate":
                        eB[b, nb] = fulln[b, nb]
                    elif x == "neither":
                        eB[b, nb] = zn[b, nb]
            lgA, _, _ = m._forward_backbone(full, tb, base_mask, past_key_values=cache_B, use_cache=False, position_ids=pos_ids)
            S._trim(cache_B, cache_B.get_seq_length() - cl)
            m._forward_backbone(eB, tb, base_mask, past_key_values=cache_B, use_cache=False, position_ids=pos_ids)
            with PerLayerMasks(m, masks):
                lgA2, _, _ = m._forward_backbone(full, tb, default, past_key_values=cache_B, use_cache=False, position_ids=pos_ids)
            S._trim(cache_B, cache_B.get_seq_length() - cl)
            lg = lgA.float().clone(); lg[:, j] = lgA2[:, j].float()
            e_den = F.softmax(lg, dim=-1).to(dt) @ E
            esc = e_den.detach()
            last = lg
            if step == K - 1:
                break
            tf, to = t_fwd[None, :, None].float(), t_out[None, :, None].float()
            e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
        return [(last[b].argmax(-1).tolist(), last[b].cpu()) for b in range(B)]
    finally:
        m.noise_scale_mult = saved_noise
        m.backbone.config._attn_implementation = orig_attn


@torch.no_grad()
def run_block_cutall_cfg_batched(S: "Sampler", cache_B, block_start: int, cfg: BlockCfg, conds: list):
    """Paper-regime (prefix guidance) block generation for B = len(conds) conditions in one batch, where at the
    heads in cut = {layer: [heads]} EVERY query p reads every other block position q's K/V from a pass in which
    nobody receives feedback (no-SC pass), instead of from the normal SC pass. Own-position reads are unchanged.
    Conditional and prefix-free branches each keep their own feedback (as _euler_denoise_block). conds: list of cut
    dicts ({} = bf16-matched baseline through the same path). Returns [(tokens, cond_logits)] per condition."""
    m, E, bd, dt, dev = S.m, S.E, S.bd, S.dtype, S.dev
    B = len(conds); g = cfg.cfg_scale
    saved_noise = m.noise_scale_mult
    m.noise_scale_mult = float(cfg.noise)
    orig_attn = getattr(m.backbone.config, "_attn_implementation", None)
    m.backbone.config._attn_implementation = "eager"
    H = m.backbone.config.num_attention_heads
    neg = float("-inf")
    try:
        torch.manual_seed(cfg.seed)
        grid = m._get_generation_time_grid(cfg.num_steps, dev, grid_mode=cfg.grid_mode, t_max=0.2)
        if grid.dim() == 1:
            grid = grid[None].expand(bd, -1)
        K = grid.shape[-1] - 1
        pos_ids = torch.arange(block_start, block_start + bd, device=dev).unsqueeze(0).expand(B, -1)
        cl = cache_B.get_seq_length()
        cmask = torch.zeros(B, 1, bd, cl + bd, dtype=dt, device=dev)
        umask = cmask.clone(); umask[..., :cl] = neg
        e_t = (torch.randn(1, bd, m._d_model, device=dev, dtype=dt) * m._noise_sigma * m.noise_scale_mult).expand(B, -1, -1).contiguous()
        tb0 = m._compute_t_bn(grid[:, 0].to(dt)[None, :].expand(B, -1))
        esc = {}
        for br, mk in (("c", cmask), ("u", umask)):
            lg0, _, _ = m._forward_backbone(e_t, tb0, mk, past_key_values=cache_B, use_cache=False, position_ids=pos_ids)
            S._trim(cache_B, cache_B.get_seq_length() - cl)
            esc[br] = F.softmax(lg0.float(), -1).to(dt) @ E
        layers = sorted({l for c in conds for l in c})
        def build(prefix_visible):
            d = torch.zeros(B, 1, bd, cl + 2 * bd, dtype=dt, device=dev)
            if not prefix_visible:
                d[..., :cl] = neg
            d[..., cl:cl + bd] = neg                                    # by default nobody reads the no-SC pass
            out = {}
            for li in layers:
                mk = d.expand(B, H, bd, cl + 2 * bd).clone()
                for b, c in enumerate(conds):
                    for h in c.get(li, []):
                        for p in range(bd):
                            for q in range(bd):
                                if p != q:
                                    mk[b, h, p, cl + q] = 0.0
                                    mk[b, h, p, cl + bd + q] = neg
                out[li] = mk
            return d, out
        dflt = {"c": build(True), "u": build(False)}
        base = {"c": cmask, "u": umask}
        last = None
        for step in range(K):
            t_fwd, t_out = grid[:, step], grid[:, step + 1]
            z = e_t
            tb = m._compute_t_bn(t_fwd.to(dt)[None, :].expand(B, -1))
            lgs = {}
            for br in ("c", "u"):
                full = m._apply_self_cond(z, esc[br])
                if layers:
                    m._forward_backbone(z, tb, base[br], past_key_values=cache_B, use_cache=False, position_ids=pos_ids)
                    d, masks = dflt[br]
                    with PerLayerMasks(m, masks):
                        lg, _, _ = m._forward_backbone(full, tb, d, past_key_values=cache_B, use_cache=False, position_ids=pos_ids)
                else:
                    lg, _, _ = m._forward_backbone(full, tb, base[br], past_key_values=cache_B, use_cache=False, position_ids=pos_ids)
                S._trim(cache_B, cache_B.get_seq_length() - cl)
                lgs[br] = lg.float()
            pc = F.softmax(lgs["c"], -1).to(dt); pu = F.softmax(lgs["u"], -1).to(dt)
            ec, eu = pc @ E, pu @ E
            e_den = eu + g * (ec - eu)
            esc = {"c": ec.detach(), "u": eu.detach()}
            last = lgs["c"]
            if step == K - 1:
                break
            tf, to = t_fwd[None, :, None].float(), t_out[None, :, None].float()
            e_t = (z.float() + (to - tf) * (e_den.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)).to(dt)
        return [(last[b].argmax(-1).tolist(), last[b].cpu()) for b in range(B)]
    finally:
        m.noise_scale_mult = saved_noise
        m.backbone.config._attn_implementation = orig_attn


@torch.no_grad()
def prefill_rows(S: "Sampler", rows: list):
    """Prefix KV cache for a batch of equal-length token rows (one batched forward)."""
    ids = torch.tensor(rows, device=S.dev)
    emb = S.m.embed_tokens(ids)
    t_bn = S.m._compute_t_bn(torch.ones(ids.shape, device=S.dev, dtype=S.dtype))
    _, cache, _ = S.m._forward_backbone(emb, t_bn, attn_mask=None, use_cache=True)
    return cache, ids.shape[1]
