"""mf — progress dashboard for mechflow runs (CARC + local).

Commands:
  mf                     dashboard: CARC jobs, runner experiments (done/total, rate, ETA),
                         local background jobs (progress parsed from logs)
  mf watch [SECONDS]     auto-refreshing dashboard (default 30 s)
  mf curve RUN METRIC [--remote]
                         running-estimate curve: mean ± 95% CI of METRIC (dotted path inside
                         each record's "result", e.g. delta_coh or stats.hyb) vs. tasks completed.
                         RUN = a runner output directory name (under src/results/).
  mf runs [--remote]     list runner output directories
  mf log JOBID           tail a CARC job log

Runner output convention (src/runner.py): <out>/meta.json + shard_*.jsonl lines
{"task_id", "result": {...}, "sec"}; tasks file path is in meta.json ("tasks").
"""
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../mechflow
SRC = os.path.join(ROOT, "src")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "carc"]

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    console = Console()
except ImportError:  # minimal fallback
    console = None


# ------------------------------------------------------------------ collectors
REMOTE_PY = r'''
import glob, json, os, time
base = os.path.expandvars("/scratch1/$USER/mechflow/src")
out = []
for meta in glob.glob(base + "/results/*/meta.json"):
    d = os.path.dirname(meta)
    try:
        m = json.load(open(meta))
    except Exception:
        m = {}
    tasks = m.get("tasks")
    total = None
    if tasks:
        tp = tasks if os.path.isabs(tasks) else os.path.join(base, tasks)
        try:
            total = sum(1 for _ in open(tp))
        except Exception:
            pass
    done, secs, last = 0, [], 0
    for sh in glob.glob(d + "/shard_*.jsonl"):
        last = max(last, os.path.getmtime(sh))
        for line in open(sh):
            done += 1
            try:
                secs.append(json.loads(line).get("sec", 0))
            except Exception:
                pass
    out.append({"run": os.path.basename(d), "done": done, "total": total,
                "mean_sec": (sum(secs) / len(secs)) if secs else None,
                "age_s": time.time() - last if last else None, "job": m.get("job")})
print(json.dumps(out))
'''

REMOTE_SERIES = r'''
import glob, json, os, sys
base = os.path.expandvars("/scratch1/$USER/mechflow/src")
run, key = sys.argv[1], sys.argv[2]
vals = []
for sh in sorted(glob.glob(f"{base}/results/{run}/shard_*.jsonl")):
    for line in open(sh):
        try:
            r = json.loads(line)
            v = r["result"]
            for k in key.split("."):
                v = v[k]
            vals.append((os.path.getmtime(sh), float(v)))
        except Exception:
            pass
print(json.dumps([v for _, v in vals]))
'''


def run_cmd(cmd, inp=None, timeout=25):
    try:
        p = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=timeout)
        return p.stdout
    except Exception:
        return ""


def carc_jobs():
    out = run_cmd(SSH + ['squeue -u $USER -h -o "%i|%j|%T|%M|%l|%b|%R"; echo ===; '
                         'squeue -u $USER --start -h -o "%i|%S"'])
    if not out:
        return None
    jobs_txt, _, start_txt = out.partition("===")
    starts = dict(l.split("|", 1) for l in start_txt.strip().splitlines() if "|" in l)
    jobs = []
    for l in jobs_txt.strip().splitlines():
        f = l.split("|")
        if len(f) >= 7:
            jobs.append({"id": f[0], "name": f[1], "state": f[2], "time": f[3], "limit": f[4],
                         "gres": f[5].replace("gres/gpu:", ""), "reason": f[6],
                         "start": starts.get(f[0], "").replace("T", " ")})
    return jobs


def remote_runs():
    out = run_cmd(SSH + ["python3 -"], inp=REMOTE_PY)
    try:
        return json.loads(out)
    except Exception:
        return None


def local_runs():
    res = []
    for meta in glob.glob(os.path.join(SRC, "results", "*", "meta.json")):
        d = os.path.dirname(meta)
        m = json.load(open(meta))
        tasks = m.get("tasks")
        total = None
        if tasks:
            tp = tasks if os.path.isabs(tasks) else os.path.join(SRC, tasks)
            if os.path.exists(tp):
                total = sum(1 for _ in open(tp))
        done, secs, last = 0, [], 0
        for sh in glob.glob(os.path.join(d, "shard_*.jsonl")):
            last = max(last, os.path.getmtime(sh))
            for line in open(sh):
                done += 1
                try:
                    secs.append(json.loads(line).get("sec", 0))
                except Exception:
                    pass
        res.append({"run": os.path.basename(d), "done": done, "total": total,
                    "mean_sec": sum(secs) / len(secs) if secs else None,
                    "age_s": time.time() - last if last else None, "job": m.get("job")})
    return res


TQDM = re.compile(r"(\d+)/(\d+) \[(\S+)<(\S+),\s*([\d.]+\s*\S+/\S+|[\d.]+s/it|[\d.]+it/s)")
COUNTER = re.compile(r"^(\d+)(?:/(\d+))?\b")


def local_jobs():
    """Running python processes under mechflow + progress from their newest logs."""
    ps = run_cmd(["ps", "-axo", "pid=,etime=,command="])
    procs = [l.strip() for l in ps.splitlines()
             if "python" in l and ("mechflow" in l or "smoke/.venv" in l or "evaluation.harness" in l)
             and "tools/mf.py" not in l and "multiprocessing" not in l]
    logs = []
    for pat in ("results_local/*.log", "src/results/*.log", "smoke/results/*.log"):
        logs += glob.glob(os.path.join(ROOT, pat))
    logs = sorted(logs, key=os.path.getmtime, reverse=True)[:6]
    entries = []
    for lg in logs:
        try:
            tail = open(lg, "rb").read()[-6000:].decode("utf-8", "ignore").replace("\r", "\n")
            tail = "\n".join(l for l in tail.splitlines()
                              if not re.search(r"Loading weights|Fetching \d+ files|Generating (train|test) split|Map:|Downloading", l))
        except Exception:
            continue
        prog = None
        m = list(TQDM.finditer(tail))
        if m:
            g = m[-1].groups()
            prog = {"done": int(g[0]), "total": int(g[1]), "elapsed": g[2], "eta": g[3], "rate": g[4]}
        else:
            lines = [l.strip() for l in tail.splitlines() if l.strip()]
            for l in reversed(lines):
                c = COUNTER.match(l)
                if c:
                    prog = {"done": int(c.group(1)), "total": int(c.group(2)) if c.group(2) else None}
                    break
        entries.append({"log": os.path.relpath(lg, ROOT), "age_s": time.time() - os.path.getmtime(lg),
                        "prog": prog, "last": tail.strip().splitlines()[-1][:90] if tail.strip() else ""})
    return procs, entries


# ------------------------------------------------------------------ rendering
def fmt_age(s):
    if s is None:
        return "-"
    s = int(s)
    return f"{s}s" if s < 90 else (f"{s // 60}m" if s < 5400 else f"{s // 3600}h{(s % 3600) // 60:02d}m")


def bar(done, total, width=22):
    if not total:
        return f"{done} done"
    f = min(1.0, done / total)
    n = int(f * width)
    return f"[{'█' * n}{'░' * (width - n)}] {done}/{total} ({100 * f:.0f}%)"


def eta(done, total, mean_sec, workers=None):
    if not total or not mean_sec or done >= total:
        return "-"
    return fmt_age((total - done) * mean_sec / max(1, workers or 1))


def dashboard():
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    jobs, rruns = carc_jobs(), remote_runs()
    lruns = local_runs()
    procs, logs = local_jobs()
    if console is None:
        print(t0, json.dumps({"carc": jobs, "remote_runs": rruns, "local_runs": lruns, "logs": logs}, indent=1))
        return
    console.rule(f"[bold]mechflow status[/] — {t0}")
    tb = Table(box=box.SIMPLE_HEAVY, title="CARC jobs", title_justify="left")
    for c in ("job", "name", "state", "elapsed / limit", "gpu", "reason / est. start"):
        tb.add_column(c)
    if jobs is None:
        tb.add_row("[red]ssh carc failed[/] (session expired? re-open the ControlMaster)", "", "", "", "", "")
    elif not jobs:
        tb.add_row("[dim]no jobs in queue[/]", "", "", "", "", "")
    for j in jobs or []:
        col = {"RUNNING": "green", "PENDING": "yellow"}.get(j["state"], "white")
        tb.add_row(j["id"], j["name"], f"[{col}]{j['state']}[/]", f"{j['time']} / {j['limit']}", j["gres"],
                   j["start"] if j["state"] == "PENDING" and j["start"] not in ("", "N/A") else j["reason"])
    console.print(tb)

    tr = Table(box=box.SIMPLE_HEAVY, title="Runner experiments (shards)", title_justify="left")
    for c in ("where", "run", "progress", "s/task", "last write", "ETA (1 worker)"):
        tr.add_column(c)
    for where, runs in (("CARC", rruns), ("local", lruns)):
        if runs is None:
            tr.add_row(where, "[red]unavailable[/]", "", "", "", "")
            continue
        for r in sorted(runs, key=lambda r: r.get("age_s") or 1e9):
            tr.add_row(where, r["run"], bar(r["done"], r["total"]),
                       f"{r['mean_sec']:.1f}" if r["mean_sec"] else "-", fmt_age(r["age_s"]),
                       eta(r["done"], r["total"], r["mean_sec"]))
    if not (rruns or lruns):
        tr.add_row("-", "[dim]no runner outputs yet[/]", "", "", "", "")
    console.print(tr)

    tl = Table(box=box.SIMPLE_HEAVY, title=f"Local jobs ({len(procs)} python process(es) running)",
               title_justify="left")
    for c in ("log", "progress", "rate / ETA", "updated", "last line"):
        tl.add_column(c, overflow="fold")
    for e in logs:
        p = e["prog"]
        pr = bar(p["done"], p["total"]) if p else "-"
        re_ = f"{p.get('rate', '')}  ETA {p.get('eta', '-')}" if p and p.get("rate") else "-"
        stale = e["age_s"] > 600
        tl.add_row(e["log"], pr, re_, ("[dim]" if stale else "[green]") + fmt_age(e["age_s"]) + " ago[/]",
                   e["last"])
    console.print(tl)
    console.print("[dim]mf curve RUN METRIC [--remote] → running-estimate curve · mf watch → auto-refresh[/]")


def series(run, key, remote):
    if remote:
        out = run_cmd(SSH + [f"python3 - {run} {key}"], inp=REMOTE_SERIES)
        return json.loads(out or "[]")
    vals = []
    for sh in sorted(glob.glob(os.path.join(SRC, "results", run, "shard_*.jsonl"))):
        for line in open(sh):
            try:
                v = json.loads(line)["result"]
                for k in key.split("."):
                    v = v[k]
                vals.append(float(v))
            except Exception:
                pass
    return vals


def ascii_chart(xs, mu, lo, hi, width=90, height=18):
    """Running mean as '●', 95% CI band as '·'; y-axis labels on the left."""
    k = min(len(lo) - 1, max(1, min(10, len(lo) // 10)))   # ignore the first few, very wide CIs for scaling
    ymin, ymax = min(lo[k:]), max(hi[k:])
    if ymax - ymin < 1e-12:
        ymax, ymin = ymax + 1e-6, ymin - 1e-6
    n = len(xs)
    cols = min(width, n)
    grid = [[" "] * cols for _ in range(height)]

    def row(y):
        return height - 1 - int(round((y - ymin) / (ymax - ymin) * (height - 1)))
    for c in range(cols):
        i = int(c * (n - 1) / max(1, cols - 1))
        clamp = lambda r: max(0, min(height - 1, r))
        for r in range(clamp(row(hi[i])), clamp(row(lo[i])) + 1):
            grid[r][c] = "·"
        grid[clamp(row(mu[i]))][c] = "●"
    lines = []
    for r in range(height):
        y = ymax - r * (ymax - ymin) / (height - 1)
        label = f"{y:>10.4f} │" if r in (0, height // 2, height - 1) else " " * 10 + " │"
        lines.append(label + "".join(grid[r]))
    lines.append(" " * 11 + "└" + "─" * cols)
    lines.append(" " * 12 + f"1{' ' * (cols - len(str(n)) - 1)}{n}   (tasks completed)")
    return "\n".join(lines)


def curve(run, key, remote):
    import shutil
    v = series(run, key, remote)
    if len(v) < 2:
        print(f"not enough values for {run}:{key} (found {len(v)})")
        return
    xs, mu, lo, hi = [], [], [], []
    s = s2 = 0.0
    for n, x in enumerate(v, 1):
        s += x
        s2 += x * x
        m = s / n
        sd = math.sqrt(max(0.0, (s2 / n - m * m) * n / max(1, n - 1)))
        half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
        xs.append(n); mu.append(m); lo.append(m - half); hi.append(m + half)
    w = max(30, min(100, shutil.get_terminal_size((110, 30)).columns - 16))
    print(f"{run} · {key}   n={len(v)}   mean={mu[-1]:.4f}   95% CI [{lo[-1]:.4f}, {hi[-1]:.4f}]")
    print(ascii_chart(xs, mu, lo, hi, width=w))


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "watch":
        sec = int(a[1]) if len(a) > 1 else 30
        try:
            while True:
                (console.clear() if console else os.system("clear"))
                dashboard()
                time.sleep(sec)
        except KeyboardInterrupt:
            pass
    elif cmd == "curve" and len(a) >= 3:
        curve(a[1], a[2], "--remote" in a)
    elif cmd == "runs":
        rs = remote_runs() if "--remote" in a else local_runs()
        for r in rs or []:
            print(r["run"], r["done"], "/", r["total"])
    elif cmd == "log" and len(a) >= 2:
        print(run_cmd(SSH + [f"tail -40 $(ls /scratch1/$USER/mechflow/logs/*_{a[1]}*.out | head -1)"]))
    elif cmd in ("", "status"):
        dashboard()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
