#!/bin/bash
#SBATCH --job-name=mf_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/%u/mechflow/logs/%x_%j.out
# Phase 6 throughput on A100 (the GPU type we budget on).
set -euo pipefail
cd ${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/src
export HF_HOME=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow/env/bin/python
mkdir -p results
nvidia-smi --query-gpu=name --format=csv
$PY bench.py bf16 results/bench_05b_bf16.json
$PY bench.py fp32 results/bench_05b_fp32.json
LAFA_REPO=lafa-hf/lafa-7B LAFA_REV=434c100d0fb2cce29ac72d76432da4b8a81eb0b7 $PY bench.py bf16 results/bench_7b_bf16.json
echo DONE
