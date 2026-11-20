#!/bin/bash
#SBATCH --job-name=plaid_gpu
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch1/%u/mechflow/plaid_port/logs/%x_%j.out
# (b) ELBO on WikiText-103/-2 (L=1024) and OpenWebText (L=256, 1024); (c) samples SC on/off.
set -euo pipefail
P=/scratch1/$USER/mechflow/plaid_port
export TOKENIZERS_PARALLELISM=false
/scratch1/$USER/hcfrl/.venv/bin/python $P/run_gpu_validation.py \
  --weights_path /scratch1/$USER/models/plaid/plaid1b_weights \
  --data_dir /scratch1/$USER/data/plaid_eval --out_dir $P/results ${ONLY:+--only $ONLY}
echo DONE
