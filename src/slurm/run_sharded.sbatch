#!/bin/bash
#SBATCH --job-name=mf_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch1/%u/mechflow/logs/%x_%A_%a.out
# Generic sharded experiment job. Submit as an array, e.g.:
#   sbatch --array=0-7 --export=ALL,TASKS=...,JOB=module:fn,OUT=...,WPG=6,DTYPE=bf16 run_sharded.sbatch
# Resumable: re-submitting skips finished tasks.
set -euo pipefail
# Account: export SBATCH_ACCOUNT=<your-slurm-account> (or source slurm/carc.env) before sbatch.
cd ${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/src
export HF_HOME=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/env/bin/python runner.py --tasks "$TASKS" --job "$JOB" --out "$OUT" \
  --workers-per-gpu "${WPG:-6}" --dtype "${DTYPE:-bf16}"
