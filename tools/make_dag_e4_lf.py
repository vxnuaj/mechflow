"""Build tools/dag_e4_lf.json: family E4 (ELF decode access) + family G (LangFlow) as debug-partition tasks for debug_dag.py."""
import json
U = "/scratch1/$USER"; HC = f"{U}/hcfrl"; PH = f"{HC}/.venv/bin/python"; PM = f"{U}/mechflow/env/bin/python"
ELF = f"$(ls -d {U}/hf_cache/hub/models--embedded-language-flows--ELF-B-owt-torch/snapshots/*/)"
T5 = f"$(ls -d {U}/hf_cache/hub/models--t5-small/snapshots/*/ | head -1)"
MIS = f"$(ls -d {U}/mechflow/hf_home/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/)"
LFP = f"$(ls -d {U}/hf_cache/hub/models--Continuous-Rivals-Discrete--langflow-owt/snapshots/*/)"
G2 = f"{U}/hf_cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
R = f"{U}/mechflow/e4dbg"; L = f"{U}/mechflow/lfdbg"; GPU = "gpu:a40:1"
T = [dict(name="e4_base", cmd=f"{PH} elf_access_decomp.py --hcfrl {HC} --ckpt {ELF} --out {R}/base_ref --n_steps 32 --access zero --plain_only --n 256 --bs 16 --seed 2",
          done=f"{R}/base_ref/32_zero/plain_on.json", deps=[], gres=GPU)]
judges = []; lf_chunks = []
for s in range(8):
    for arm in ("readable", "zeroed"):
        ft = f"e4ft_{arm[0]}{s}"; ck = f"{R}/ft/{arm}_s{s}"
        T.append(dict(name=ft, cmd=f"FT_SAVE_FREQ=0.2 {PH} elf_ft_decode.py --hcfrl {HC} --init {ELF} --data {U}/data/owt_t5 --arm {arm} --seed {s} --out {ck} --steps 3000 --batch 64 --accum 4 --max_length 1024 --lr 2e-5",
                      done=f"{ck}/checkpoint_12000", deps=[], gres=GPU, max_tries=10, mem="64G"))
        T.append(dict(name=f"e4ce_{arm[0]}{s}", cmd=f"{PH} elf_heldout_ce.py --hcfrl {HC} --ckpt {ck} --data {U}/data/owt_t5 --out {R}/e4/{arm}_s{s}/heldout_ce.json --bs 8",
                      done=f"{R}/e4/{arm}_s{s}/heldout_ce.json", deps=[ft], gres=GPU))
        cfgs = [("32", "read"), ("32", "zero"), ("24", "read"), ("24", "zero"), ("24", "shuf")] if arm == "readable" else [("32", "zero"), ("24", "zero")]
        for n, acc in cfgs:
            dn = f"e4d_{arm[0]}{s}_{n}{acc}"; d = f"{R}/e4/{arm}_s{s}/{n}_{acc}"
            T.append(dict(name=dn, cmd=f"{PH} elf_access_decomp.py --hcfrl {HC} --ckpt {ck} --out {R}/e4/{arm}_s{s} --n_steps {n} --access {acc} --n 256 --bs 16 --seed 2",
                          done=f"{d}/meta.json", deps=[ft], gres=GPU))
            jn = f"e4j_{arm[0]}{s}_{n}{acc}"; judges.append(jn)
            T.append(dict(name=jn, cmd=f"T5_PATH={T5} JUDGE_PATH={MIS} JUDGE_DEVICE=cuda REF_FILE={R}/base_ref/32_zero/plain_on.json {PM} elf_judge_fixedctx.py {d}/cells.json",
                          done=f"{d}/cells_judged_fixedctx.json", deps=[dn, "e4_base"], gres=GPU, mem="64G"))
for st in range(0, 256, 16):   # 16-sample chunks: ~500 s per batch of 4 in fp32 at L 1024, so ~35 min per chunk (< 55-min limit)
    cn = f"lf_c{st}"; lf_chunks.append(cn)
    T.append(dict(name=cn, cmd=f"{PH} langflow_decomp.py --path {LFP} --out {L}/chunk_{st} --start {st} --count 16 --bs 4 --seed 2 --device cuda",
                  done=f"{L}/chunk_{st}/meta.json", deps=[], gres=GPU, mem="64G"))
ce = [t["name"] for t in T if t["name"].startswith("e4ce_")]
T.append(dict(name="e4_analysis", cmd=f"cd .. && {PM} analysis/elf_access.py {R}/e4 > {R}/E4_final.txt 2>&1", done=f"{R}/E4_final.txt", deps=judges + ce, gres="", mem="16G"))
T.append(dict(name="lf_merge", cmd=f"{PM} langflow_merge.py {L}", done=f"{L}/merged/meta.json", deps=lf_chunks, gres="", mem="16G"))
T.append(dict(name="lf_judge", cmd=f"T5_PATH={G2} JUDGE_PATH={MIS} JUDGE_DEVICE=cuda REF_FILE={L}/merged/plain_on.json {PM} elf_judge_fixedctx.py {L}/merged/cells.json",
              done=f"{L}/merged/cells_judged_fixedctx.json", deps=["lf_merge"], gres=GPU, mem="64G"))
T.append(dict(name="lf_analysis", cmd=f"cd .. && {PM} analysis/langflow.py {L}/merged > {L}/G_final.txt 2>&1", done=f"{L}/G_final.txt", deps=["lf_judge"], gres="", mem="8G"))
json.dump(T, open("tools/dag_e4_lf.json", "w"), indent=1); print(len(T), "tasks")
