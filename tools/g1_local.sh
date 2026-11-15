#!/bin/bash
# G1 end-task runs locally (official harness, MPS): MR regime, SC on/off, reserved 25% slices.
set -u
M=/Users/vxuaj/Documents/code/research/icml-attempts/mechflow
cd $M/third_party/large-flow-lms
PY=$M/smoke/.venv/bin/python
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
for spec in "gsm8k_cot 512 yes" "humaneval_instruct 640 yes" "ifeval 1024 yes"; do
  set -- $spec; TASK=$1; MAXTOK=$2
  SAMPLES=$(python3 -c "import json;d=json.load(open('$M/data/g1_samples.json'));print(json.dumps({'$TASK':d['$TASK']}))")
  EXTRA=""; [ "$TASK" = humaneval_instruct ] && EXTRA="--confirm_run_unsafe_code"
  for SC in true false; do
    OUT=$M/results_local/g1_${TASK}_sc${SC}
    [ -d "$OUT" ] && ls $OUT/*/results_*.json >/dev/null 2>&1 && { echo "skip $TASK sc=$SC (done)"; continue; }
    echo "$(date +%H:%M) start $TASK sc=$SC"
    caffeinate -i -s $PY -m src.evaluation.harness --model lafa --tasks $TASK --num_fewshot 0 --apply_chat_template $EXTRA \
      --samples "$SAMPLES" --gen_kwargs max_gen_toks=$MAXTOK,do_sample=false,temperature=0.0 --output_path $OUT \
      --model_args "pretrained=lafa-hf/lafa-0.5B,gen_mode=flow,num_steps=32,cfg_scale=1,noise_scale_override=4.0,self_cond=$SC,max_new_tokens=$MAXTOK,max_length=4096" \
      > $M/results_local/g1_${TASK}_sc${SC}.log 2>&1
    echo "$(date +%H:%M) done $TASK sc=$SC: $(grep -E 'exact_match|pass@1|strict_acc' $M/results_local/g1_${TASK}_sc${SC}.log | head -3 | tr -s ' ' | tr '\n' ' ')"
  done
done
