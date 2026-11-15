#!/bin/bash
# Keep resubmitting resumable runner jobs (debug A40, 55 min) until each output has its target line count.
# usage: keep_running.sh NAME TASKS JOB OUT TARGET WPG [extra exports]
NAME=$1; TASKS=$2; JOB=$3; OUT=$4; TARGET=$5; WPG=$6; EXTRA=$7
while true; do
  N=$(ssh -o BatchMode=yes carc "cat /scratch1/\$USER/mechflow/src/$OUT/shard_*.jsonl 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ')
  [ -n "$N" ] && [ "$N" -ge "$TARGET" ] && { echo "$(date +%H:%M) $NAME complete: $N/$TARGET"; exit 0; }
  Q=$(ssh -o BatchMode=yes carc "squeue -u \$USER -h -n $NAME -o %T" 2>/dev/null)
  if [ -z "$Q" ]; then
    ssh -o BatchMode=yes carc "bash -lc 'cd /scratch1/\$USER/mechflow/src && source slurm/carc.env && sbatch --parsable -p debug --gres=gpu:a40:1 --time=00:55:00 --job-name=$NAME --export=ALL,TASKS=$TASKS,JOB=$JOB,OUT=$OUT,WPG=$WPG,DTYPE=bf16$EXTRA slurm/run_sharded.sbatch'" >/dev/null 2>&1 && echo "$(date +%H:%M) $NAME resubmitted at $N/$TARGET"
  fi
  sleep 120
done
