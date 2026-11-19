#!/bin/bash
# Single lightweight watcher (replaces the per-task schedulers; low memory).
# Keeps S7 (7B confirmation, 4 shards) and L1-R (6 shards) queued on the CARC gpu partition, submits the S7 judge
# when generation completes, and runs both frozen analyses when their data are complete. Checks every 15 min.
cd /Users/vxuaj/Documents/code/research/icml-attempts/mechflow
R7=",LAFA_REPO=lafa-hf/lafa-7B,LAFA_REV=434c100d0fb2cce29ac72d76432da4b8a81eb0b7"
AL=",PYTORCH_ALLOC_CONF=expandable_segments:True"
sub() { # name tasks job out arr count extra time
  ssh -o BatchMode=yes carc "bash -lc 'cd /scratch1/\$USER/mechflow/src && source slurm/carc.env && sbatch --parsable -p gpu --gres=gpu:1 --constraint=\"a40|a100|l40s\" --time=$8 --job-name=$1 --export=ALL,TASKS=$2,JOB=$3,OUT=$4,WPG=1,DTYPE=bf16,SLURM_ARRAY_TASK_ID=$5,SLURM_ARRAY_TASK_COUNT=$6$7 slurm/run_sharded.sbatch'" >/dev/null 2>&1 && echo "$(date +%H:%M) submitted $1"
}
s7_done=0; l1r_done=0; judge_done=0
while true; do
  ST=$(ssh -o BatchMode=yes carc 'cd /scratch1/$USER/mechflow/src && python3 - <<PY
import json, glob, os
def done(out, tasks, arr, count):
    T = [json.loads(l)["task_id"] for l in open(tasks)]
    need = {t for i, t in enumerate(T) if i % count == arr}
    p = f"{out}/shard_{arr:04d}.jsonl"
    d = {json.loads(l)["task_id"] for l in open(p)} if os.path.exists(p) else set()
    return int(need <= d)
print(" ".join(str(done("results/conf7b_h2n", "tasks_conf7b.jsonl", a, 4)) for a in range(4)),
      "|", " ".join(str(done("results/l1r", "tasks_l1r.jsonl", a, 6)) for a in range(6)),
      "|", " ".join(str(done("results/conf7b_judge", "tasks_conf7b.jsonl", a, 2)) for a in range(2)))
PY
squeue -u $USER -h -o %j | tr "\n" " "' 2>/dev/null)
  [ -z "$ST" ] && { sleep 300; continue; }
  read -r a0 a1 a2 a3 _ b0 b1 b2 b3 b4 b5 _ c0 c1 rest <<< "$ST"; Q=" $rest "
  echo "$(date +%H:%M) S7 [$a0$a1$a2$a3] L1R [$b0$b1$b2$b3$b4$b5] judge [$c0$c1]"
  i=0; for d in $a0 $a1 $a2 $a3; do [ "$d" = 0 ] && ! grep -q " mf_s7h_$i " <<< "$Q" && sub mf_s7h_$i tasks_conf7b.jsonl jobs_h2n:h2n_task results/conf7b_h2n $i 4 ",H2N_REGIME=MR$R7$AL" 08:00:00; i=$((i+1)); done
  i=0; for d in $b0 $b1 $b2 $b3 $b4 $b5; do [ "$d" = 0 ] && ! grep -q " mf_l1r_$i " <<< "$Q" && ! grep -q " mf_l1r " <<< "$Q" && sub mf_l1r_$i tasks_l1r.jsonl jobs_l1:l1_task results/l1r $i 6 ",CUTS=cuts_l1.json$AL" 08:00:00; i=$((i+1)); done
  if [ "$a0$a1$a2$a3" = 1111 ]; then
    i=0; for d in $c0 $c1; do [ "$d" = 0 ] && ! grep -q " mf_s7j_$i " <<< "$Q" && sub mf_s7j_$i tasks_conf7b.jsonl jobs_judge_h2n:judge_task results/conf7b_judge $i 2 ",JUDGE_SRC=conf7b_h2n$AL" 04:00:00; i=$((i+1)); done
    if [ "$c0$c1" = 11 ] && [ $s7_done = 0 ]; then
      for d in conf7b_h2n conf7b_judge; do mkdir -p src/results/$d; rsync -az --delete carc:/scratch1/verajuan/mechflow/src/results/$d/ src/results/$d/; done
      python3 analysis/s7b.py src/results > src/results/s7_final.txt 2>&1; echo "S7 ANALYSIS DONE"; cat src/results/s7_final.txt; s7_done=1
    fi
  fi
  if [ "$b0$b1$b2$b3$b4$b5" = 111111 ] && [ $l1r_done = 0 ]; then
    mkdir -p src/results/l1r; rsync -az --delete carc:/scratch1/verajuan/mechflow/src/results/l1r/ src/results/l1r/
    python3 analysis/l1.py src/results/l1r > src/results/l1r_final.txt 2>&1; echo "L1R ANALYSIS DONE"; cat src/results/l1r_final.txt; l1r_done=1
  fi
  [ $s7_done = 1 ] && [ $l1r_done = 1 ] && { echo "ALL DONE"; exit 0; }
  sleep 900
done
