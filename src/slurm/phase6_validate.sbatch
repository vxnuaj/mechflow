#!/bin/bash
#SBATCH --job-name=mf_phase6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/%u/mechflow/logs/%x_%j.out

# Phase 6: instrument validation on CUDA (any GPU type; throughput is in phase6_bench_a100.sbatch).
set -euo pipefail
# Account: export SBATCH_ACCOUNT=<your-slurm-account> (or source slurm/carc.env) before sbatch.
cd ${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/src
export HF_HOME=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/env/bin/python
mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== validation suite (0.5B, bf16)"
$PY test_instruments.py results/instrument_validation_cuda.json

echo DONE
