#!/bin/bash
# Family E4 on CARC GPUs (fallback while the Prime API lists no GPUs). Run ON CARC from src/. Slurm dependency chain per
# (arm, seed): A fine-tune → B decomposition + held-out CE → C judge (one job per config); D frozen analysis after all.
set -euo pipefail; cd /scratch1/$USER/mechflow/src; source slurm/carc.env
R=/scratch1/$USER/mechflow/e4run; mkdir -p $R/logs; HC=/scratch1/$USER/hcfrl; PH=$HC/.venv/bin/python; PM=/scratch1/$USER/mechflow/env/bin/python
ELF=$(ls -d /scratch1/$USER/hf_cache/hub/models--embedded-language-flows--ELF-B-owt-torch/snapshots/*/); T5=$(ls -d /scratch1/$USER/hf_cache/hub/models--t5-small/snapshots/*/ | head -1)
MIS=$(ls -d /scratch1/$USER/mechflow/hf_home/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/)
ENV="export HF_HUB_CACHE=/scratch1/$USER/hf_cache/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false"
G="-p gpu --gres=gpu:1 --constraint=a40|a100|l40s --cpus-per-task=8 --mem=64G"
sub() { sbatch --parsable $G --time=$1 --job-name=$2 --output=$R/logs/$2_%j.out ${3:+--dependency=afterok:$3} --wrap "$ENV; $4"; }
BASE=$(sub 03:00:00 e4_base "" "$PH elf_access_decomp.py --hcfrl $HC --ckpt $ELF --out $R/base_ref --n_steps 32 --access zero --plain_only --n 256 --seed 2")
CJ=""
for arm in readable zeroed; do for s in 0 1 2 3 4 5 6 7; do
  A=$(sub 08:00:00 e4A_${arm}_s$s "" "$PH elf_ft_decode.py --hcfrl $HC --init $ELF --data /scratch1/$USER/data/owt_t5 --arm $arm --seed $s --out $R/ft/${arm}_s$s --steps 3000 --batch 64 --max_length 1024 --lr 2e-5")
  acc=$([ $arm = readable ] && echo read,zero,shuf || echo zero)
  B=$(sub 08:00:00 e4B_${arm}_s$s $A "$PH elf_access_decomp.py --hcfrl $HC --ckpt $R/ft/${arm}_s$s --out $R/e4/${arm}_s$s --n_steps 32,24 --access $acc --n 256 --seed 2 && $PH elf_heldout_ce.py --hcfrl $HC --ckpt $R/ft/${arm}_s$s --data /scratch1/$USER/data/owt_t5 --out $R/e4/${arm}_s$s/heldout_ce.json && rm -rf $R/e4/${arm}_s$s/32_shuf")
  cfgs=$([ $arm = readable ] && echo "32_read 32_zero 24_read 24_zero 24_shuf" || echo "32_zero 24_zero")
  for c in $cfgs; do
    C=$(sub 03:00:00 e4C_${arm}_s${s}_$c "$B:$BASE" "T5_PATH=$T5 JUDGE_PATH=$MIS JUDGE_DEVICE=cuda REF_FILE=$R/base_ref/32_zero/plain_on.json $PM elf_judge_fixedctx.py $R/e4/${arm}_s$s/$c/cells.json")
    CJ="$CJ:$C"
  done
done; done
D=$(sbatch --parsable -p main --cpus-per-task=4 --mem=16G --time=02:00:00 --job-name=e4D --output=$R/logs/e4D_%j.out --dependency=afterok$CJ --wrap "cd /scratch1/$USER/mechflow && $PM analysis/elf_access.py $R/e4 > $R/E4_final.txt 2>&1")
echo "base $BASE; analysis $D; $(echo $CJ | tr ':' '\n' | grep -c .) judge jobs"
