#!/bin/bash
# G1 scheduler: gsm8k units run locally (Mac, sequential); other units are submitted to CARC debug (<=2 at once).
M=/Users/vxuaj/Documents/code/research/icml-attempts/mechflow
PY=$M/smoke/.venv/bin/python
done_remote(){ ssh -o BatchMode=yes carc "ls /scratch1/\$USER/mechflow/results/g1c/$1/*/results_*.json" >/dev/null 2>&1; }
done_local(){ ls $M/results_local/g1c/$1/*/results_*.json >/dev/null 2>&1; }
local_runner(){
  for u in $(python3 -c "import json;[print(x['chunk']+'_sc'+x['sc']) for x in json.load(open('$M/data/g1_units.json')) if x['task']=='gsm8k_cot']"); do
    done_local $u && continue
    CHUNK=${u%_sc*}; SC=${u##*_sc}
    echo "$(date +%H:%M) local start $u"
    (cd $M/third_party/large-flow-lms && HF_ALLOW_CODE_EVAL=1 caffeinate -i -s $PY -m src.evaluation.harness --model lafa --tasks gsm8k_cot \
      --num_fewshot 0 --apply_chat_template --samples "$(cat $M/data/g1_chunks/$CHUNK.json)" --log_samples \
      --gen_kwargs max_gen_toks=512,do_sample=false,temperature=0.0 --output_path $M/results_local/g1c/$u \
      --model_args "pretrained=lafa-hf/lafa-0.5B,gen_mode=flow,num_steps=32,cfg_scale=1,noise_scale_override=4.0,self_cond=$SC,max_new_tokens=512,max_length=4096" \
      > $M/results_local/g1c_$u.log 2>&1)
    echo "$(date +%H:%M) local done $u"
  done
}
remote_runner(){
  while true; do
    pending=""
    for u in $(python3 -c "import json;[print(x['chunk']+'_sc'+x['sc']) for x in json.load(open('$M/data/g1_units.json')) if x['task']!='gsm8k_cot']"); do
      done_remote $u || pending="$pending $u"
    done
    [ -z "$pending" ] && { echo "$(date +%H:%M) remote units all done"; return; }
    Q=$(ssh -o BatchMode=yes carc "squeue -u \$USER -h -n mf_g1c -o %k" 2>/dev/null)
    n=$(echo "$Q" | grep -c .)
    for u in $pending; do
      [ $n -ge 2 ] && break
      echo "$Q" | grep -qx "$u" && continue
      CHUNK=${u%_sc*}; SC=${u##*_sc}; T=${CHUNK%_*}
      ssh -o BatchMode=yes carc "bash -lc 'cd /scratch1/\$USER/mechflow/src && source slurm/carc.env && sbatch --comment=$u --export=ALL,TASK=$T,CHUNK=$CHUNK,SC=$SC slurm/g1_chunk.sbatch'" >/dev/null 2>&1 && { echo "$(date +%H:%M) submitted $u"; n=$((n+1)); }
    done
    sleep 120
  done
}
mkdir -p $M/results_local/g1c
local_runner & remote_runner; wait
echo "$(date +%H:%M) G1 all units done"
