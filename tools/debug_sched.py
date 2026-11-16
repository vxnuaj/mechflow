#!/usr/bin/env python3
"""Priority scheduler for the confirmatory runs on the CARC debug partition (A40, 55 min).
Each unit = a fixed set of shards (array element `arr` of `count`, WPG workers), so the shard layout always
matches the original 8-shard assignment and resubmission resumes exactly. Shard selection for a non-array
job is passed via SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT exports (read by runner.py).
Keeps at most MAX_SUB debug jobs submitted; submits units in priority order; exits when all are complete.
"""
import json, shlex, subprocess, sys, time

MAX_SUB = 3
U = [  # name, tasks, job, out, wpg, arr, count, extra exports, skip-if-running (other job name)
    ("mf_dh2n", "tasks_conf_h2n.jsonl", "jobs_h2n:h2n_task", "results/conf_h2n", 8, 0, 1, "H2N_REGIME=MR", None),
    ("mf_dp2_1", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 2, 1, 4, "P2_REGIME=PR", None),
    ("mf_dp2_2", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 2, 2, 4, "P2_REGIME=PR", None),
    ("mf_dp2_3", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 2, 3, 4, "P2_REGIME=PR", None),
    ("mf_dp2_0", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 2, 0, 4, "P2_REGIME=PR", "mf_cp2"),
    ("mf_dp1", "tasks_conf_p1.jsonl", "jobs_p1:p1_task", "results/conf_p1", 8, 0, 1, "", None),
    ("mf_cs12", "tasks_conf_s12.jsonl", "jobs_h2n:h2n_task", "results/conf_s12", 8, 0, 1, "H2N_REGIME=PR", None),
    ("mf_cp2x", "tasks_conf_p2x.jsonl", "jobs_p2x:p2x_task", "results/conf_p2x", 8, 0, 1, "", None),
]

STATUS = r'''
import json, os, sys
U = json.loads(sys.argv[1]); out = {}
os.chdir(os.path.expandvars("/scratch1/$USER/mechflow/src"))
for name, tasks, out_dir, wpg, arr, count in U:
    T = [json.loads(l)["task_id"] for l in open(tasks)]
    n = wpg * count; shards = [arr * wpg + w for w in range(wpg)]
    need = left = 0
    for s in shards:
        exp = {t for i, t in enumerate(T) if i % n == s}
        done = set()
        p = f"{out_dir}/shard_{s:04d}.jsonl"
        if os.path.exists(p):
            for l in open(p):
                try: done.add(json.loads(l)["task_id"])
                except Exception: pass
        need += len(exp); left += len(exp - done)
    out[name] = [need - left, need]
print(json.dumps(out))
'''


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "carc", cmd], capture_output=True, text=True, timeout=300).stdout


def main():
    arg = json.dumps([[u[0], u[1], u[3], u[4], u[5], u[6]] for u in U])
    while True:
        try:
            st = json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", "carc", "python3 - " + shlex.quote(arg)], input=STATUS,
                                           capture_output=True, text=True, timeout=300).stdout)
            q = ssh("squeue -u $USER -h -o %j").split()
        except Exception as e:
            print(time.strftime("%H:%M"), "status error", e, flush=True); time.sleep(120); continue
        todo = [u for u in U if st[u[0]][0] < st[u[0]][1]]
        if not todo:
            print(time.strftime("%H:%M"), "ALL DEBUG UNITS COMPLETE", flush=True); return
        sub = sum(1 for u in U if u[0] in q)
        line = " ".join(f"{u[0][3:]}:{st[u[0]][0]}/{st[u[0]][1]}" for u in U)
        for u in todo:
            name, tasks, job, out, wpg, arr, count, extra, skip = u
            if sub >= MAX_SUB:
                break
            if name in q or (skip and skip in q):
                continue
            ex = f"ALL,TASKS={tasks},JOB={job},OUT={out},WPG={wpg},DTYPE=bf16"
            if count > 1:
                ex += f",SLURM_ARRAY_TASK_ID={arr},SLURM_ARRAY_TASK_COUNT={count}"
            if extra:
                ex += "," + extra
            ssh(f"bash -lc 'cd /scratch1/$USER/mechflow/src && source slurm/carc.env && sbatch --parsable -p debug "
                f"--gres=gpu:a40:1 --time=00:55:00 --job-name={name} --export={ex} slurm/run_sharded.sbatch'")
            sub += 1; line += f" | submit {name}"
        print(time.strftime("%H:%M"), line, flush=True)
        time.sleep(120)


if __name__ == "__main__":
    main()
