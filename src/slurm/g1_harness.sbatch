#!/bin/bash
#SBATCH --job-name=mf_g1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch1/%u/mechflow/logs/%x_%j.out
# G1 (prereg §5.1): end-task SC gain in the mechanism regime (K=32, g=1, noise 4.0) on the reserved
# 25% benchmark slice (data/g1_samples.json). Env: TASK, SC (true|false), FEWSHOT, CHAT (yes|no), EXTRA.
set -euo pipefail
B=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow
cd $B/large-flow-lms
export HF_HOME=$B/hf_home TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
SAMPLES=$(python3 -c "import json;d=json.load(open('$B/src/../data/g1_samples.json'));print(json.dumps({'$TASK':d['$TASK']}))")
$B/env/bin/python -m src.evaluation.harness --model lafa --tasks $TASK --num_fewshot ${FEWSHOT:-0} \
  $([ "${CHAT:-yes}" = yes ] && echo --apply_chat_template) ${EXTRA:-} --samples "$SAMPLES" \
  --gen_kwargs max_gen_toks=${MAXTOK:-512},do_sample=false,temperature=0.0 \
  --output_path $B/results/g1_${TASK}_sc${SC} \
  --model_args "pretrained=lafa-hf/lafa-0.5B,gen_mode=flow,num_steps=32,cfg_scale=1,noise_scale_override=4.0,self_cond=${SC},max_new_tokens=${MAXTOK:-512},max_length=4096"
echo DONE
