"""Family E4 v2 fluency gate (red-team M8) and training-distribution uptake (descriptive): the native ELF decoder CE on
held-out OWT-t5 rows (the last 2,048 of the 259,660-row shard pair; fine-tunes use the first steps × batch = 192,000).
Corruption as in ELF training: per-token λ ~ sigmoid(N(decoder_p_mean, decoder_p_std)), z = λ·x0 + (1 − λ)·e·decoder_noise_scale,
t = 1, decode mode. CE per non-pad target token with decode SC = zero, the same-state estimate (as in the `readable` arm's
training), or a shuffled estimate (batch rolled by 1). Fixed corruption seed 11. Writes JSON {ce_zero, ce_sc, ce_shuf, n_tokens}."""
import argparse, json, os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_instr import load_elf, _dec_logits, _vx
ap = argparse.ArgumentParser(); ap.add_argument("--hcfrl", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--rows", type=int, default=2048)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--L", type=int, default=128); ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device)
model, cfg, tok = load_elf(a.hcfrl, a.ckpt, a.L, dev)
from datasets import load_from_disk
from modules.t5_encoder import get_encoder
from utils.encoder_utils import encode_text
_, enc = get_encoder(cfg.encoder_model_name, torch.float32); enc = enc.to(dev).eval()
ds = load_from_disk(a.data); N = len(ds); rows = ds.select(range(N - a.rows, N))
g = torch.Generator(device="cpu").manual_seed(11); pad = tok.pad_token_id
tot = {"zero": 0.0, "sc": 0.0, "shuf": 0.0}; ntok = 0
with torch.no_grad():
    for b in range(0, a.rows, a.bs):
        ids = [r[:a.L] + [pad] * max(0, a.L - len(r)) for r in rows[b:b + a.bs]["input_ids"]]
        ids = torch.tensor(ids, device=dev); am = (ids != pad).float()
        x0 = encode_text(input_ids=ids, attention_mask=am, encoder=enc, latent_mean=cfg.latent_mean, latent_std=cfg.latent_std,
                         use_bf16=dev.type == "cuda").float()
        B = ids.shape[0]
        lam = torch.sigmoid(torch.randn((B, a.L, 1), generator=g) * cfg.decoder_p_std + cfg.decoder_p_mean).to(dev)
        e = torch.randn(x0.shape, generator=g).to(dev) * cfg.decoder_noise_scale
        z = lam * x0 + (1 - lam) * e
        tdec = lam.mean(dim=(1, 2))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model(torch.cat([z, torch.zeros_like(z)], -1), tdec, deterministic=True, self_cond_cfg_scale=torch.ones((B,), device=dev))
        _, xh = _vx(out, z, tdec, cfg.t_eps)
        for k, sc in (("zero", torch.zeros_like(z)), ("sc", xh), ("shuf", torch.roll(xh, 1, 0))):
            lg = _dec_logits(model, torch.cat([z, sc], -1), 1.0).float()
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids.reshape(-1), reduction="none").reshape(B, a.L)
            tot[k] += (ce * am).sum().item()
        ntok += am.sum().item()
res = {f"ce_{k}": v / ntok for k, v in tot.items()} | {"n_tokens": ntok, "ckpt": a.ckpt}
print(res); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
