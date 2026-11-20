#!/bin/bash
#SBATCH --job-name=plaid_elbo_cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/%u/mechflow/plaid_port/logs/%x_%j.out
# (b, CPU fp32 reference) ELBO on 32 OpenWebText sequences of length 256, K=4.
set -euo pipefail
P=/scratch1/$USER/mechflow/plaid_port
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p $P/results
/scratch1/$USER/hcfrl/.venv/bin/python $P/elbo_torch.py --weights_path /scratch1/$USER/models/plaid/plaid1b_weights \
  --data owt --parquet /scratch1/$USER/data/plaid_eval/owt-0000.parquet --seq_len 256 --n_seqs 32 --K 4 \
  --batch_size 10 --fp32 --device cpu --out $P/results/elbo_owt_L256_cpu_fp32.json
echo DONE
