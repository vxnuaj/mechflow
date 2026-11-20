#!/bin/bash
# Run ON the pod after setup_e4.sh. A single driver (flock). Stages:
#   A 16 fine-tunes (2 arms × 8 seeds); B decompositions + held-out CE + base reference; C Mistral judge (common ref);
#   D frozen E4 analysis → ~/out/E4_final.txt.
#   LangFlow: calibration runs at once; the decomposition waits for ~/LF_GO (created only after its prereg is frozen).
exec 9>~/run_e4.lock; flock -n 9 || { echo "driver already running"; exit 1; }
set -uo pipefail; source ~/venv/bin/activate; export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
NG=$(nvidia-smi -L | wc -l); SLOTS=${SLOTS:-2}; S=~/mf/src; O=~/out; mkdir -p $O/logs
snap() { ls -d $HF_HOME/hub/models--$1/snapshots/*/ | head -1; }
ELF=$(snap embedded-language-flows--ELF-B-owt-torch); MIS=$(snap mistralai--Mistral-7B-Instruct-v0.3); LF=$(snap Continuous-Rivals-Discrete--langflow-owt)
GPT2=$(snap openai-community--gpt2); T5=$(snap t5-small); H=~/hcfrl_min
run_pool() {  # run commands from a file, SLOTS per GPU, each on a CUDA device round-robin
  local f=$1 i=0; local -a pids=()
  while read -r cmd; do
    [ -z "$cmd" ] && continue
    g=$(( i % NG )); ( CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" ) & pids+=($!); i=$((i+1))
    while [ $(jobs -rp | wc -l) -ge $(( NG * SLOTS )) ]; do sleep 5; done
  done < $f; wait
}
# LangFlow calibration (regime only), in the background on the last GPU
( cd $S && CUDA_VISIBLE_DEVICES=$((NG-1)) python langflow_calib.py --path $LF --device cuda --out $O/langflow_calib.json > $O/logs/lf_calib.log 2>&1 ) &
# A: fine-tunes
: > $O/A.cmds
for arm in readable zeroed; do for s in 0 1 2 3 4 5 6 7; do
  echo "cd $S && python elf_ft_decode.py --hcfrl $H --init $ELF --data ~/data/owt_t5 --arm $arm --seed $s --out $O/ft/${arm}_s$s --steps 3000 --batch 64 --max_length 1024 --lr 2e-5 > $O/logs/ft_${arm}_s$s.log 2>&1" >> $O/A.cmds
done; done
SLOTS=1 run_pool $O/A.cmds; echo "A done $(date)" >> $O/progress.txt
# B: base reference + decompositions + held-out CE
: > $O/B.cmds
echo "cd $S && python elf_access_decomp.py --hcfrl $H --ckpt $ELF --out $O/base_ref --n_steps 32 --access zero --plain_only --n 256 --seed 2 > $O/logs/base_ref.log 2>&1" >> $O/B.cmds
for arm in readable zeroed; do for s in 0 1 2 3 4 5 6 7; do
  acc=$([ $arm = readable ] && echo read,zero,shuf || echo zero)
  echo "cd $S && python elf_access_decomp.py --hcfrl $H --ckpt $O/ft/${arm}_s$s --out $O/e4/${arm}_s$s --n_steps 32,24 --access $acc --n 256 --seed 2 > $O/logs/dec_${arm}_s$s.log 2>&1 && python elf_heldout_ce.py --hcfrl $H --ckpt $O/ft/${arm}_s$s --data ~/data/owt_t5 --out $O/e4/${arm}_s$s/heldout_ce.json > $O/logs/ce_${arm}_s$s.log 2>&1" >> $O/B.cmds
done; done
run_pool $O/B.cmds; echo "B done $(date)" >> $O/progress.txt
# prune configs not in the frozen grid (readable 32_shuf is not analysed)
rm -rf $O/e4/readable_s*/32_shuf
# C: judge (one Mistral per GPU)
: > $O/C.cmds
for c in $O/e4/*/*/cells.json; do
  echo "cd $S && T5_PATH=$T5 JUDGE_PATH=$MIS JUDGE_DEVICE=cuda REF_FILE=$O/base_ref/32_zero/plain_on.json python elf_judge_fixedctx.py $c > ${c%.json}_judge.log 2>&1" >> $O/C.cmds
done
SLOTS=1 run_pool $O/C.cmds; echo "C done $(date)" >> $O/progress.txt
# D: frozen analysis
cd ~/mf && python analysis/elf_access.py $O/e4 > $O/E4_final.txt 2>&1; echo "D done $(date)" >> $O/progress.txt
# LangFlow decomposition: gated on its prereg
while [ ! -f ~/LF_GO ]; do sleep 60; done
cd $S && python langflow_decomp.py --path $LF --out $O/lf --n 256 --seed 2 --device cuda > $O/logs/lf_decomp.log 2>&1 && \
  T5_PATH=$GPT2 JUDGE_PATH=$MIS JUDGE_DEVICE=cuda REF_FILE=$O/lf/plain_on.json python elf_judge_fixedctx.py $O/lf/cells.json > $O/logs/lf_judge.log 2>&1
echo "LF done $(date)" >> $O/progress.txt; echo ALL_DONE >> $O/progress.txt
