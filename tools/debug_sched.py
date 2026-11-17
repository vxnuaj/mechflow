#!/usr/bin/env python3
"""Priority scheduler for the confirmatory runs on the CARC debug partition (A40, 55 min).
Each unit = a fixed set of shards (array element `arr` of `count`, WPG workers), so the shard layout always
matches the original 8-shard assignment and resubmission resumes exactly. Shard selection for a non-array
job is passed via SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT exports (read by runner.py).
Keeps at most MAX_SUB debug jobs submitted; submits units in priority order; exits when all are complete.
"""
import json, shlex, subprocess, sys, time

MAX_SUB = 3
U = [  # name, tasks, job, out, wpg, arr, count, extra exports, skip-if-any-of-these-jobs-queued
    ("mf_h2n_0", "tasks_conf_h2n.jsonl", "jobs_h2n:h2n_task", "results/conf_h2n", 4, 0, 2, "H2N_REGIME=MR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dh2n"]),
    ("mf_h2n_1", "tasks_conf_h2n.jsonl", "jobs_h2n:h2n_task", "results/conf_h2n", 4, 1, 2, "H2N_REGIME=MR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dh2n"]),
    ("mf_p2_0", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 1, 0, 8, "P2_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dp2_1", "mf_dp2_2", "mf_dp2_3", "mf_cp2"]),
    ("mf_p2_1", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 1, 1, 8, "P2_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dp2_1", "mf_dp2_2", "mf_dp2_3", "mf_cp2"]),
    ("mf_p2_2", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 1, 2, 8, "P2_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dp2_1", "mf_dp2_2", "mf_dp2_3", "mf_cp2"]),
    ("mf_p2_3", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 1, 3, 8, "P2_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dp2_1", "mf_dp2_2", "mf_dp2_3", "mf_cp2"]),
    ("mf_p2_4", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 1, 4, 8, "P2_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dp2_1", "mf_dp2_2", "mf_dp2_3", "mf_cp2"]),
    ("mf_p2_5", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 1, 5, 8, "P2_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dp2_1", "mf_dp2_2", "mf_dp2_3", "mf_cp2"]),
    ("mf_p2_6", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 1, 6, 8, "P2_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dp2_1", "mf_dp2_2", "mf_dp2_3", "mf_cp2"]),
    ("mf_p2_7", "tasks_conf_p2.jsonl", "jobs_p2:p2_task", "results/conf_p2", 1, 7, 8, "P2_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_dp2_1", "mf_dp2_2", "mf_dp2_3", "mf_cp2"]),
    ("mf_dp1", "tasks_conf_p1.jsonl", "jobs_p1:p1_task", "results/conf_p1", 8, 0, 1, "PYTORCH_ALLOC_CONF=expandable_segments:True", []),
    ("mf_s12_0", "tasks_conf_s12.jsonl", "jobs_h2n:h2n_task", "results/conf_s12", 4, 0, 2, "H2N_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_cs12"]),
    ("mf_s12_1", "tasks_conf_s12.jsonl", "jobs_h2n:h2n_task", "results/conf_s12", 4, 1, 2, "H2N_REGIME=PR,PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_cs12"]),
    ("mf_p2x_0", "tasks_conf_p2x.jsonl", "jobs_p2x:p2x_task", "results/conf_p2x", 4, 0, 2, "PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_cp2x"]),
    ("mf_p2x_1", "tasks_conf_p2x.jsonl", "jobs_p2x:p2x_task", "results/conf_p2x", 4, 1, 2, "PYTORCH_ALLOC_CONF=expandable_segments:True", ["mf_cp2x"]),
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
            qd = ssh("squeue -u $USER -h -p debug -o %j").split()   # only debug jobs count toward MAX_SUB
        except Exception as e:
            print(time.strftime("%H:%M"), "status error", e, flush=True); time.sleep(120); continue
        todo = [u for u in U if st[u[0]][0] < st[u[0]][1]]
        if not todo:
            print(time.strftime("%H:%M"), "ALL DEBUG UNITS COMPLETE", flush=True); return
        sub = len(qd)
        line = " ".join(f"{u[0][3:]}:{st[u[0]][0]}/{st[u[0]][1]}" for u in U if st[u[0]][0] < st[u[0]][1])
        for u in todo:
            name, tasks, job, out, wpg, arr, count, extra, skip = u
            if sub >= MAX_SUB:
                break
            if name in q or any(x in q for x in skip):
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
