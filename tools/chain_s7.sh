#!/bin/bash
# S7 chain: 7B H2-N on 500 untouched docs -> Mistral judge -> frozen analysis (analysis/s7b.py).
cd /Users/vxuaj/Documents/code/research/icml-attempts/mechflow
cnt() { ssh -o BatchMode=yes carc "cat /scratch1/\$USER/mechflow/src/results/$1/shard_*.jsonl 2>/dev/null | wc -l" | tr -d ' '; }
python3 -u tools/debug_sched.py tools/units_s7.json
echo "$(date +%H:%M) S7 generation complete ($(cnt conf7b_h2n))"
python3 -u tools/debug_sched.py tools/units_s7_judge.json
echo "$(date +%H:%M) S7 judge complete ($(cnt conf7b_judge))"
for d in conf7b_h2n conf7b_judge; do mkdir -p src/results/$d; rsync -az --delete carc:/scratch1/$USER/mechflow/src/results/$d/ src/results/$d/; done
python3 analysis/s7b.py src/results | tee src/results/s7_final.txt
echo "S7 DONE"
