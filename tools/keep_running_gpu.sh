#!/bin/bash
# Like keep_running.sh but on the main gpu partition as a job array (shards = ARRAY * WPG, kept constant).
# usage: keep_running_gpu.sh NAME TASKS JOB OUT TARGET WPG ARRAY TIME [extra exports]
NAME=$1; TASKS=$2; JOB=$3; OUT=$4; TARGET=$5; WPG=$6; ARR=$7; TIME=$8; EXTRA=$9
while true; do
  N=$(ssh -o BatchMode=yes carc "cat /scratch1/\$USER/mechflow/src/$OUT/shard_*.jsonl 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ')
  [ -n "$N" ] && [ "$N" -ge "$TARGET" ] && { echo "$(date +%H:%M) $NAME complete: $N/$TARGET"; exit 0; }
  Q=$(ssh -o BatchMode=yes carc "squeue -u \$USER -h -n $NAME -o %T" 2>/dev/null)
  if [ -z "$Q" ]; then
    ssh -o BatchMode=yes carc "bash -lc 'cd /scratch1/\$USER/mechflow/src && source slurm/carc.env && sbatch --parsable -p gpu --array=0-$((ARR-1)) --gres=gpu:1 --constraint=\"a40|a100|l40s\" --time=$TIME --job-name=$NAME --export=ALL,TASKS=$TASKS,JOB=$JOB,OUT=$OUT,WPG=$WPG,DTYPE=bf16$EXTRA slurm/run_sharded.sbatch'" >/dev/null 2>&1 && echo "$(date +%H:%M) $NAME submitted at $N/$TARGET"
  fi
  sleep 180
done
