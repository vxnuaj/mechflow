#!/usr/bin/env python3
"""Family R orchestration: pod runs gc_uniform; when done, pod takes over gc_warp shards 2,3 from CARC (shard layout
count=4, WPG=1 preserved; CARC partial shards copied to the pod first so nothing is redone); CARC keeps shards 0,1.
When all data is in: terminate the pod (verified) and run the frozen analysis (analysis/gc.py)."""
import glob, json, os, subprocess, time

P = "c86ebe07431e4ab48b3b7a0d65134d40"; IP = "ubuntu@150.136.39.93"
ROOT = "/Users/vxuaj/Documents/code/research/icml-attempts/mechflow"
CARC_RES = "carc:/scratch1/$USER/mechflow/src/results/gc_warp/"
os.chdir(ROOT)
os.environ["PRIME_DISABLE_VERSION_CHECK"] = "1"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw).stdout


def rs(src, dst, delete=True):
    os.makedirs(dst, exist_ok=True)
    sh(f"rsync -az {'--delete' if delete else ''} -e 'ssh -o BatchMode=yes' {src} {dst}")


def ids(paths):
    return {json.loads(l)["task_id"] for p in paths for f in glob.glob(p) for l in open(f)}


def log(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


phase = "uniform"
while True:
    time.sleep(300)
    rs(f"{IP}:mf/src/results/gc_uniform/", "src/results/gc_uniform/")
    rs(CARC_RES, "src/results/_gc_warp_carc/")
    if phase != "uniform":
        rs(f"{IP}:mf/src/results/gc_warp/", "src/results/_gc_warp_pod/")
    nu = len(ids(["src/results/gc_uniform/shard_*.jsonl"]))
    nw = len(ids(["src/results/_gc_warp_carc/shard_000[01].jsonl", "src/results/_gc_warp_pod/shard_000[23].jsonl",
                  "src/results/_gc_warp_carc/shard_000[23].jsonl"]))
    log("phase", phase, "uniform", nu, "warp", nw)
    if phase == "uniform" and nu >= 1000:
        # stop CARC scheduler, cancel warp shards 2,3 there, restart scheduler with w0,w1 + G-b
        sh("pkill -f units_carc_gcw_gb.json")
        sh("ssh -o BatchMode=yes carc 'scancel -n mf_gc_w2; scancel -n mf_gc_w3'")
        time.sleep(30)
        rs(CARC_RES, "src/results/_gc_warp_carc/")
        U = [u for u in json.load(open("tools/units_carc_gcw_gb.json")) if u[0] not in ("mf_gc_w2", "mf_gc_w3")]
        json.dump(U, open("tools/units_carc_gcw01_gb.json", "w"), indent=1)
        subprocess.Popen(f"python3 -u tools/debug_sched.py tools/units_carc_gcw01_gb.json > "
                         f"/private/tmp/claude-501/-Users-vxuaj-Documents-code-research-icml-attempts/620153de-6aca-43c8-8ed0-adce3423751f/scratchpad/sched_carc.log 2>&1",
                         shell=True)
        # seed the pod with CARC's partial shards 2,3 and launch them there
        sh(f"ssh -o BatchMode=yes {IP} 'mkdir -p ~/mf/src/results/gc_warp'")
        for s in (2, 3):
            f = f"src/results/_gc_warp_carc/shard_000{s}.jsonl"
            if os.path.exists(f):
                sh(f"scp -q {f} {IP}:mf/src/results/gc_warp/")
        sh(f"ssh -o BatchMode=yes {IP} 'cd ~/mf/src && export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false "
           f"MF_PLATFORM=prime-a100-40gb-sxm-mps BATCH=7 CUTS=cuts_gc.json GRID=warp PYTORCH_ALLOC_CONF=expandable_segments:True "
           f"SLURM_ARRAY_TASK_COUNT=4; for s in 2 3; do (SLURM_ARRAY_TASK_ID=$s nohup ../env/bin/python runner.py --tasks tasks_gc.jsonl "
           f"--job jobs_cutb:cut_task --out results/gc_warp --workers-per-gpu 1 > gc_warp_$s.log 2>&1 < /dev/null &); done'")
        phase = "warp_split"; log("uniform done; pod took over warp shards 2,3")
        continue
    if phase == "warp_split":
        pod23 = ids(["src/results/_gc_warp_pod/shard_000[23].jsonl"])
        tasks = [json.loads(l)["task_id"] for l in open("src/tasks_gc.jsonl")]
        need23 = {t for i, t in enumerate(tasks) if i % 4 in (2, 3)}
        if need23 <= pod23 and phase != "pod_off":
            for _ in range(3):
                sh(f"prime pods terminate {P} --yes < /dev/null"); time.sleep(20)
                if P not in sh("prime pods list"):
                    log("POD VERIFIED TERMINATED"); break
            phase = "pod_off"
    if phase == "pod_off":
        need01 = {t for i, t in enumerate([json.loads(l)["task_id"] for l in open("src/tasks_gc.jsonl")]) if i % 4 in (0, 1)}
        if need01 <= ids(["src/results/_gc_warp_carc/shard_000[01].jsonl"]):
            # assemble gc_warp: CARC shards 0,1 + pod shards 2,3
            os.makedirs("src/results/gc_warp", exist_ok=True)
            for s, srcd in ((0, "_gc_warp_carc"), (1, "_gc_warp_carc"), (2, "_gc_warp_pod"), (3, "_gc_warp_pod")):
                sh(f"cp src/results/{srcd}/shard_000{s}.jsonl src/results/gc_warp/")
            log("FAMILY R DATA COMPLETE; running frozen analysis")
            out = sh("python3 analysis/gc.py src/results")
            open("src/results/gc_final.txt", "w").write(out); print(out, flush=True)
            break
