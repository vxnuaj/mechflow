#!/usr/bin/env python3
"""Dependency-aware scheduler for the CARC debug partition (≤ 5 submitted jobs per user, 55-min limit).
Usage: python tools/debug_dag.py <dag.json>
dag.json: [{"name", "cmd" (run on CARC from mechflow/src with env), "done" (remote path that exists when complete),
            "deps": [names], "gres": "gpu:a40:1" | "" (CPU job on main), "mem", "max_tries"}].
Each loop (every 90 s): one ssh call reports the done/queued state; ready tasks (deps done, not done, not queued) are
submitted in list order while fewer than MAX_SUB debug jobs are queued. A task re-runs (resume) until `done` exists, up
to max_tries; tasks over the limit are reported and their dependents blocked. Exits when nothing more can run."""
import json, shlex, subprocess, sys, time

MAX_SUB = 5
D = json.load(open(sys.argv[1])); byname = {t["name"]: t for t in D}
tries = {t["name"]: 0 for t in D}; failed = set()
STATUS = r'''
import json, os, subprocess, sys
D = json.loads(sys.argv[1]); out = {"done": [], "queued": []}
for n, p in D:
    if os.path.exists(os.path.expandvars(p)): out["done"].append(n)
q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j|%P"], stdout=subprocess.PIPE, universal_newlines=True).stdout.split()   # py3.6 on CARC login
out["queued"] = [x.split("|")[0] for x in q]; out["n_debug"] = sum(1 for x in q if x.endswith("|debug"))
print(json.dumps(out))
'''


def ssh(cmd, timeout=120):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "carc", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.returncode


def status():
    arg = json.dumps([[t["name"], t["done"]] for t in D])
    out, rc = ssh(f"python3 -c {shlex.quote(STATUS)} {shlex.quote(arg)}")
    if rc != 0 or not out: return None
    return json.loads(out)


def submit(t):
    env = "export HF_HUB_CACHE=/scratch1/$USER/hf_cache/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false"
    wrap = f"{env}; cd /scratch1/$USER/mechflow/src && {t['cmd']}"
    part = f"-p debug --gres={t['gres']} --time=00:55:00" if t.get("gres") else "-p main --time=02:00:00"
    cmd = (f"cd /scratch1/$USER/mechflow/src && source slurm/carc.env && sbatch --parsable {part} --cpus-per-task=8 "
           f"--mem={t.get('mem', '48G')} --job-name={t['name']} --output=/scratch1/$USER/mechflow/logs/dag_{t['name']}_%j.out "
           f"--wrap {shlex.quote(wrap)}")
    out, rc = ssh(cmd)
    return out if rc == 0 else None


fails = 0
while True:
    try:
        st = status()
    except Exception as e:
        st = None; print(f"{time.strftime('%H:%M')} status error: {e}", flush=True)
    if st is None:
        fails += 1
        if fails in (1, 3) or fails % 20 == 0: print(f"{time.strftime('%H:%M')} WARNING: status check failed ({fails}x)", flush=True)
        time.sleep(90); continue
    fails = 0
    done, queued = set(st["done"]), set(st["queued"]); nsub = st["n_debug"]
    remaining = [t for t in D if t["name"] not in done]
    if not remaining:
        print(f"{time.strftime('%H:%M')} ALL DONE ({len(D)} tasks)", flush=True); break
    progressed = False
    for t in remaining:
        n = t["name"]
        if n in queued: progressed = True; continue
        if n in failed: continue
        if any(d in failed for d in t["deps"]): failed.add(n); print(f"{time.strftime('%H:%M')} BLOCKED {n} (failed dependency)", flush=True); continue
        if not all(d in done for d in t["deps"]): progressed = True; continue
        if tries[n] >= t.get("max_tries", 3):
            failed.add(n); print(f"{time.strftime('%H:%M')} FAILED {n} after {tries[n]} tries", flush=True); continue
        progressed = True
        if t.get("gres") and nsub >= MAX_SUB: continue
        jid = submit(t)
        if jid:
            tries[n] += 1; nsub += 1 if t.get("gres") else 0
            print(f"{time.strftime('%H:%M')} submitted {n} (try {tries[n]}) job {jid}", flush=True)
    if not progressed:
        print(f"{time.strftime('%H:%M')} STOPPED: {len(remaining)} tasks cannot run: " + ", ".join(sorted(failed))[:500], flush=True); break
    time.sleep(90)
