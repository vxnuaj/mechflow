"""ELF step 0 (exploratory; gap G-e): does self-conditioning matter in ELF-B (OWT, unconditional)?
Generates N samples with identical noise and schedule while varying only the self-conditioning CFG scale
(0 = no SC, 1 = SC, 3 = ELF default), then scores the decoded text with Mistral-7B-Instruct-v0.3 (mean token
log-prob) and simple repetition metrics. Uses the user's hcfrl ELF implementation READ-ONLY (sys.path).
usage (CARC, hcfrl venv): python elf_sc_check.py --hcfrl /scratch1/$USER/hcfrl --n 128 --steps 32 --method ode
"""
import argparse, json, math, os, sys, time
from types import SimpleNamespace

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--hcfrl", required=True); ap.add_argument("--n", type=int, default=128)
ap.add_argument("--steps", type=int, default=32); ap.add_argument("--method", default="ode")
ap.add_argument("--L", type=int, default=128); ap.add_argument("--out", default="results/elf_sc_check.json")
ap.add_argument("--judge", default=None)
a = ap.parse_args()
sys.path.insert(0, a.hcfrl)
import hcfrl.elf_path  # noqa: F401  (puts the vendored ELF root on sys.path)
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config import load_config_from_yaml, apply_config_overrides
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import load_checkpoint
from utils.train_utils import TrainState, get_optimizer
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch
from huggingface_hub import snapshot_download

dev = torch.device("cuda")
ck = snapshot_download("embedded-language-flows/ELF-B-owt-torch")
_y = open(os.path.join(ck, "config.yml")).read().replace(
    "src/configs/sampling_configs/uncond_sampling_configs.yml",
    os.path.join(a.hcfrl, "hcfrl/elf/configs/sampling_configs/uncond_sampling_configs.yml"))
_tmp = f"/tmp/elf_cfg_{os.getpid()}.yml"; open(_tmp, "w").write(_y)
cfg = load_config_from_yaml(_tmp)
cfg = apply_config_overrides(cfg, [f"max_length={a.L}", "use_wandb=false"])
tok = AutoTokenizer.from_pretrained(cfg.encoder_model_name)
enc_cfg, _ = get_encoder(cfg.encoder_model_name, torch.float32)
model = ELF_models[cfg.model](
    text_encoder_dim=enc_cfg.d_model, max_length=cfg.max_length, attn_drop=0.0, proj_drop=0.0,
    num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
    vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
    bottleneck_dim=cfg.bottleneck_dim).to(dev)
state = TrainState(model=model, optimizer=get_optimizer(model, cfg, lr=1e-4), lr_scheduler=None,
                   ema_params1=TrainState.init_ema(model), step=0, epoch=0,
                   dropout_generator=torch.Generator(device="cpu").manual_seed(0))
state, step = load_checkpoint("embedded-language-flows/ELF-B-owt-torch", state)
model = state.model.to(dev)
if state.ema_params1: model.load_state_dict(state.ema_params1)
model.eval()
print("ELF loaded, step", step, flush=True)

g = torch.Generator().manual_seed(0)
z0 = torch.randn((a.n, cfg.max_length, model.text_encoder_dim), generator=g) * cfg.denoiser_noise_scale
gs = torch.Generator().manual_seed(1)
t_steps = get_sampling_steps(a.steps, time_schedule="logit_normal", generator=gs).to(dev) if "generator" in get_sampling_steps.__code__.co_varnames else get_sampling_steps(a.steps, time_schedule="logit_normal").to(dev)
samp = SimpleNamespace(sampling_method=a.method, sde_gamma=getattr(cfg, "sde_gamma", 0.0))
from utils.sampling_utils import net_out_to_v_x
texts = {}
# Conditions follow ELF training: "SC off" = zero self-conditioning input (as for the 50% of training rows with
# SC dropped); "SC on" = previous x-prediction. The scale token w is passed as in training (w = 1: plain SC;
# w = 3: ELF default guidance). Same noise z0 and schedule for every condition.
def run(z, sc_on, w):
    B = z.shape[0]; xp = torch.zeros_like(z)
    wt = torch.full((B,), float(w), device=dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(t_steps) - 1):
            t, tn = t_steps[i].item(), t_steps[i + 1].item()
            tb = torch.full((B,), t, device=dev)
            out = model(torch.cat([z, xp], -1), tb, deterministic=True, self_cond_cfg_scale=wt)
            v, x = net_out_to_v_x(out, z, tb, cfg.t_eps)
            z = z + (tn - t) * v
            xp = x if sc_on else torch.zeros_like(x)
    return z
for name, sc_on, w in (("off_w1", False, 1.0), ("on_w1", True, 1.0), ("on_w3", True, 3.0), ("off_w3", False, 3.0)):
    t0 = time.time(); out = []
    for b in range(0, a.n, 32):
        with torch.no_grad():
            z = run(z0[b:b + 32].to(dev), sc_on, w)
            ids = _dlm_decode_batch(z, model, t_steps[-1], cfg, w)
        out += [tok.decode(r, skip_special_tokens=True) for r in ids.tolist()]
    texts[name] = out
    print(f"{name}: generated {len(out)} in {time.time()-t0:.0f}s; e.g. {out[0][:120]!r}", flush=True)
del model; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(texts, open(a.out.replace(".json", "_texts_v2.json"), "w"), indent=1)
if a.judge == "skip":
    sys.exit(0)

jp = a.judge or snapshot_download("mistralai/Mistral-7B-Instruct-v0.3", revision="c170c708c41dac9275d15a8fff4eca08d52bab71")
jt = AutoTokenizer.from_pretrained(jp); jm = AutoModelForCausalLM.from_pretrained(jp, dtype=torch.bfloat16).to(dev).eval()
res = {}
for sc, ts in texts.items():
    lps, reps, d2 = [], [], []
    for txt in ts:
        ids = jt(txt, return_tensors="pt", add_special_tokens=True).input_ids[:, :256].to(dev)
        if ids.shape[1] < 3: continue
        with torch.no_grad():
            lp = torch.log_softmax(jm(ids).logits[0, :-1].float(), -1).gather(1, ids[0, 1:, None])[:, 0]
        lps.append(float(lp.mean()))
        w = txt.split(); reps.append(sum(w[i] == w[i - 1] for i in range(1, len(w))) / max(1, len(w) - 1))
        bg = list(zip(w, w[1:])); d2.append(len(set(bg)) / max(1, len(bg)))
    m = lambda x: sum(x) / max(1, len(x)); sd = lambda x: (sum((v - m(x)) ** 2 for v in x) / max(1, len(x) - 1)) ** .5
    res[sc] = {"n": len(lps), "judge_lp_mean": m(lps), "judge_lp_se": sd(lps) / math.sqrt(max(1, len(lps))),
               "immediate_word_repeat": m(reps), "distinct2": m(d2), "examples": ts[:3]}
    print(sc, {k: v for k, v in res[sc].items() if k != "examples"}, flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
