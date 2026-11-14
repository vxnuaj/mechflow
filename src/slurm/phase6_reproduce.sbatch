#!/bin/bash
#SBATCH --job-name=mf_repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch1/%u/mechflow/logs/%x_%j.out

# Phase 6: reproduce a reported number with the OFFICIAL harness (paper: Lafa-0.5B, B=4, 32 Euler
# steps, g=8, noise 0.1 -> GSM8K 30.4, Table 5). Subset of 100 problems (limit) => SE ~ 4.6 pp (~6 min/problem on V100; use bf16-native GPUs).
# GSM8K test is the end task only (prereg §4.3), so this does not touch confirmation data.
set -euo pipefail
# Account: export SBATCH_ACCOUNT=<your-slurm-account> (or source slurm/carc.env) before sbatch.
cd ${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/large-flow-lms
export HF_HOME=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/hf_home TOKENIZERS_PARALLELISM=false
PY=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/env/bin/python
$PY -m src.evaluation.harness \
  --model lafa --tasks gsm8k_cot --num_fewshot 0 --apply_chat_template --limit 100 \
  --gen_kwargs max_gen_toks=512,do_sample=false,temperature=0.0 \
  --output_path ${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/results/repro_gsm8k_K32_g8 \
  --model_args "pretrained=lafa-hf/lafa-0.5B,gen_mode=flow,num_steps=32,cfg_scale=8,noise_scale_override=0.1,max_new_tokens=512,max_length=4096"
echo DONE
