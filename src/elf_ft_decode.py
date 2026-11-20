"""Phase 13 step 3: matched ELF-B decoder-access fine-tunes (hcfrl's vendored ELF code is used read-only, patched in memory).

ARM readable: decoder-mode rows receive a stop-gradient self-conditioning input with probability self_cond_prob (0.5),
              the same as denoiser rows. The estimate comes from a no-SC pass on the decoder's own corrupted input
              (same-state SC, standard practice; ELF Alg. 3), at t = the row's mean corruption level.
ARM zeroed:   unchanged ELF training (decoder rows get zero SC). The control.
Both arms: identical data order, seed, steps, lr and init (ELF-B weights), and an identical extra no-grad forward
(computed and discarded in `zeroed`), so compute and RNG streams match.

Usage: python elf_ft_decode.py --hcfrl <root> --init <ELF-B snapshot dir> --data <dataset dir> --arm readable|zeroed
       --out <dir> [--steps 3000 --batch 64 --max_length 1024 --lr 2e-5]
"""
import argparse, os, re, sys, types

ap = argparse.ArgumentParser()
ap.add_argument("--hcfrl", required=True); ap.add_argument("--init", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--arm", choices=["readable", "zeroed"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max_length", type=int, default=1024); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()

ELF = os.path.join(a.hcfrl, "hcfrl", "elf"); sys.path.insert(0, ELF); os.chdir(ELF)
import train_step as TS                                   # noqa: E402
import train as TR                                        # noqa: E402

ORIG = "        sc_half = x_pred_cond * (1.0 - decoder_mask_B11)\n"
PATCH = '''        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            _t_dec_est = decoder_lambda_t.mean(dim=(1, 2)).to(dtype)
            _out_dec = model(torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1), _t_dec_est,
                             deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale)
        _, _x_dec = net_out_to_v_x(_out_dec, decoder_z, _t_dec_est, t_eps)
        _x_dec = restore_cond(_x_dec, x0, cond_seq_mask) * use_self_cond_mask.to(dtype)
        if getattr(config, "decoder_sc_readable", False):
            sc_half = x_pred_cond * (1.0 - decoder_mask_B11) + _x_dec * decoder_mask_B11
        else:
            sc_half = x_pred_cond * (1.0 - decoder_mask_B11)
'''
src = open(TS.__file__).read()
assert src.count(ORIG) == 1, "train_step.py changed; the patch anchor was not found exactly once"
mod = types.ModuleType("train_step_ft"); mod.__file__ = TS.__file__
exec(compile(src.replace(ORIG, PATCH), TS.__file__ + " [mechflow decoder-access patch]", "exec"), mod.__dict__)
TR.train_step = mod.train_step                            # train.py does `from train_step import train_step`

_y = open(os.path.join(ELF, "configs", "training_configs", "train_owt_ELF-B.yml")).read().replace(
    "src/configs/sampling_configs/", os.path.join(ELF, "configs", "sampling_configs") + "/")
_tmp = f"/tmp/elf_ft_cfg_{os.getpid()}.yml"; open(_tmp, "w").write(_y)
cfg = TR.load_config_from_yaml(_tmp)
over = dict(data_path=a.data, eval_data_path=None, init_from=a.init, resume=None, output_dir=a.out,
            max_length=a.max_length, global_batch_size=a.batch, batch_size=None, grad_accum_steps=1, lr=a.lr,
            lr_schedule="constant", warmup_steps=100, epochs=1, seed=a.seed, decoder_sc_readable=(a.arm == "readable"),
            sampling_configs_path=None, sampling_configs=[], online_eval=False, eval_freq=10 ** 9, save_freq=float(os.environ.get("FT_SAVE_FREQ", "1e9")),
            use_wandb=False, num_workers=2)
for k, v in over.items(): setattr(cfg, k, v)
# exactly `steps` optimizer steps: 1 epoch over the first steps × batch examples (the same subset and order in both arms)
_orig_load = TR.load_dataset
def _load_subset(config, *args, **kw):
    tr, ev = _orig_load(config, *args, **kw)
    return tr.select(range(min(len(tr), a.steps * a.batch))), ev
TR.load_dataset = _load_subset
# Resumable on short (55-min) partitions: with FT_SAVE_FREQ < 1, train.py saves every that fraction of the epoch and
# auto-resumes from the latest checkpoint in output_dir (its init_from is then ignored). Keep only the newest checkpoint.
import glob as _g, shutil as _sh
_ck = sorted(_g.glob(os.path.join(a.out, "checkpoint_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
for _old in _ck[:-1]:
    (_sh.rmtree if os.path.isdir(_old) else os.remove)(_old)
if _ck: print(f"[elf_ft] resuming from {_ck[-1]}", flush=True)
print(f"[elf_ft] arm={a.arm} steps={a.steps} batch={a.batch} L={a.max_length} lr={a.lr} seed={a.seed} decoder_sc_readable={cfg.decoder_sc_readable}", flush=True)
TR.run_training(cfg, force_cpu=a.cpu)
