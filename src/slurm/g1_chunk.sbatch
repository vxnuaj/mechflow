#!/bin/bash
#SBATCH --job-name=mf_g1c
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch1/%u/mechflow/logs/%x_%j.out
# One G1 unit: official harness, MR regime, one benchmark chunk, SC on|off, per-sample logging.
set -euo pipefail
B=${MF_ROOT_BASE:-/scratch1/$USER}/mechflow
cd $B/large-flow-lms
export HF_HOME=$B/hf_home TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
case $TASK in gsm8k_cot) MAXTOK=512;; humaneval_instruct) MAXTOK=640; EXTRA=--confirm_run_unsafe_code;; ifeval) MAXTOK=1024;; esac
$B/env/bin/python -m src.evaluation.harness --model lafa --tasks $TASK --num_fewshot 0 --apply_chat_template ${EXTRA:-} \
  --samples "$(cat $B/data/g1_chunks/$CHUNK.json)" --log_samples \
  --gen_kwargs max_gen_toks=$MAXTOK,do_sample=false,temperature=0.0 \
  --output_path $B/results/g1c/${CHUNK}_sc${SC} \
  --model_args "pretrained=lafa-hf/lafa-0.5B,gen_mode=flow,num_steps=32,cfg_scale=1,noise_scale_override=4.0,self_cond=${SC},max_new_tokens=$MAXTOK,max_length=4096"
echo DONE
