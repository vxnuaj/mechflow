#!/bin/bash
#SBATCH --job-name=plaid_sample_cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/%u/mechflow/plaid_port/logs/%x_%j.out
# (c) 4 unconditional samples, 256 steps, seq_len 256, fp32 on CPU. SC=on|off via env SC.
set -euo pipefail
P=/scratch1/$USER/mechflow/plaid_port
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p $P/results
EXTRA=""; [ "${SC:-on}" = "off" ] && EXTRA="--no_selfcond"
/scratch1/$USER/hcfrl/.venv/bin/python $P/sample_torch.py --weights_path /scratch1/$USER/models/plaid/plaid1b_weights \
  --n_samples 4 --seq_len 256 --steps 256 --seed 0 --fp32 --device cpu $EXTRA --out $P/results/samples_cpu_fp32_SC${SC:-on}.json
echo DONE
