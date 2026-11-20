#!/bin/bash
#SBATCH --job-name=plaid_cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/%u/mechflow/plaid_port/logs/%x_%j.out
# (a) strict state_dict load + kernel unit checks + tiny t=0 forward, CPU only.
set -euo pipefail
P=/scratch1/$USER/mechflow/plaid_port
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
/scratch1/$USER/hcfrl/.venv/bin/python $P/validate_load.py --weights_path /scratch1/$USER/models/plaid/plaid1b_weights
echo DONE
