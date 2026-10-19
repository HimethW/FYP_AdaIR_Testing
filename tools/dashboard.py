"""Training monitor for qbits: GPU / system stats, every run's progress, losses, validation, mask health.

Every --interval seconds it samples the GPU (nvidia-smi) and system, reads runs/*/ (args.json,
metrics.jsonl, train.log, checkpoints, eval*.json), appends the GPU sample to <out>/gpu_history.jsonl
(kept 7 days) and writes <out>/data.json plus a self-contained <out>/index.html (data embedded, so
the file also works as an offline snapshot). With --port it serves <out> on 127.0.0.1 only; the page
then re-fetches data.json every interval without reloading.

  source /home/minura/Himeth/env.sh
  setsid nohup python tools/dashboard.py --port 8765 > /home/minura/Himeth/dashboard/dashboard.out 2>&1 &
  # on your laptop (PowerShell):  ssh -N -L 8765:127.0.0.1:8765 minura@192.248.10.67
  # then open http://localhost:8765
  python tools/dashboard.py --once      # just write the snapshot

Everything stays in /home/minura/Himeth (default --out /home/minura/Himeth/dashboard).
"""
import argparse
import glob
import http.server
import json
import os
import pwd
import re
import shutil
import subprocess
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIMETH = os.path.dirname(REPO)
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html")
HISTORY_DAYS = 7


def gpu_sample():
    q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        num = lambda s: float(s) if re.match(r"^-?[0-9.]+$", s) else None  # noqa: E731
        gpus.append(dict(index=int(p[0]), name=p[1], util=num(p[2]), mem_used=num(p[3]), mem_total=num(p[4]),
                         temp=num(p[5]), power=num(p[6]), power_limit=num(p[7])))
    return gpus


def gpu_processes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    procs = []
    for line in out.strip().splitlines():
        pid, mem, name = [x.strip() for x in line.split(",", 2)]
        user = "?"
        try:
            with open(f"/proc/{pid}/status") as f:
                uid = int(next(l for l in f if l.startswith("Uid:")).split()[1])
            user = pwd.getpwuid(uid).pw_name
        except Exception:
            pass
        procs.append(dict(pid=int(pid), mem=float(mem), name=os.path.basename(name), user=user))
    return procs


def system_sample():
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            mem[k] = int(v.split()[0]) / 2 ** 20  # GB
    du = shutil.disk_usage(HIMETH)
    return dict(load=os.getloadavg(), cpus=os.cpu_count(), ram_total=mem["MemTotal"],
                ram_avail=mem["MemAvailable"], disk_total=du.total / 1e9, disk_free=du.free / 1e9)


def training_processes():
    """run name -> pid for live train_fyp.py processes."""
    live = {}
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(f"{d}/cmdline", "rb") as f:
                cmd = f.read().split(b"\0")
        except Exception:
            continue
        cmd = [c.decode(errors="ignore") for c in cmd if c]
        if any(c.endswith("train_fyp.py") for c in cmd) and "--name" in cmd:
            i = cmd.index("--name")
            if i + 1 < len(cmd):
                live[cmd[i + 1]] = int(os.path.basename(d))
    return live


def downsample(points, n=1500):
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)] + [points[-1]]


def read_run(path, live):
    name = os.path.basename(path)
    with open(os.path.join(path, "args.json")) as f:
        args = json.load(f)
    series = {}
    last = {}
    mpath = os.path.join(path, "metrics.jsonl")
    if os.path.exists(mpath):
        with open(mpath) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = m.pop("step")
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        series.setdefault(k, []).append([step, v])
                        last[k] = [step, v]
    log_lines, per_epoch, started = [], None, None
    lpath = os.path.join(path, "train.log")
    if os.path.exists(lpath):
        with open(lpath, errors="ignore") as f:
            lines = f.readlines()
        log_lines = [l.rstrip("\n")[:240] for l in lines[-25:]]
        for l in lines:
            m = re.search(r"train set: (\d+) samples / epoch", l)
            if m:
                per_epoch = int(m.group(1))
            if started is None and l.startswith("["):
                started = l[1:20]
    step = max((v[-1][0] for v in series.values()), default=0)
    done_text = "\n".join(log_lines[-3:])
    mtime = max((os.path.getmtime(p) for p in glob.glob(os.path.join(path, "*")) if os.path.isfile(p)), default=0)
    if name in live:
        status = "running"
    elif "finished" in done_text:
        status = "finished"
    elif "stopped at step" in done_text:
        status = "stopped"
    else:
        status = "not running"
    ckpts = [dict(name=os.path.basename(p), mb=round(os.path.getsize(p) / 2 ** 20), mtime=os.path.getmtime(p))
             for p in sorted(glob.glob(os.path.join(path, "*.pt")))]
    evals = {}
    for p in glob.glob(os.path.join(path, "eval*.json")):
        try:
            with open(p) as f:
                e = json.load(f)
            evals[os.path.basename(p)] = {k: [v["psnr"], v["ssim"], v["n"]] for k, v in e["results"].items()}
        except Exception:
            pass
    samples_per_step = args.get("batch", 1) * args.get("accum", 1)
    return dict(name=name, args=args, status=status, pid=live.get(name), step=step, iters=args.get("iters"),
                per_epoch=per_epoch, samples_per_step=samples_per_step, started=started, updated=mtime,
                last=last, series={k: downsample(v) for k, v in series.items()}, log=log_lines, ckpts=ckpts,
                evals=evals)


def released_evals(runs_dir):
    out = {}
    for p in glob.glob(os.path.join(runs_dir, "eval_released", "*.json")):
        try:
            with open(p) as f:
                e = json.load(f)
            out[os.path.basename(p)] = {k: [v["psnr"], v["ssim"], v["n"]] for k, v in e["results"].items()}
        except Exception:
            pass
    return out


def update(runs_dir, out_dir, history_path):
    now = time.time()
    gpus = gpu_sample()
    with open(history_path, "a") as f:
        f.write(json.dumps({"t": now, "gpus": [{k: g[k] for k in ("util", "mem_used", "temp", "power")} for g in gpus]})
                + "\n")
    hist = []
    cutoff = now - HISTORY_DAYS * 86400
    with open(history_path) as f:
        for line in f:
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                continue
            if h["t"] >= cutoff:
                hist.append(h)
    if hist and hist[0]["t"] < cutoff + 3600 and len(hist) > 50000:  # trim the file now and then
        with open(history_path, "w") as f:
            f.writelines(json.dumps(h) + "\n" for h in hist)
    live = training_processes()
    runs = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "args.json"))):
        try:
            runs.append(read_run(os.path.dirname(p), live))
        except Exception as e:  # a run being written must not break the page
            runs.append(dict(name=os.path.basename(os.path.dirname(p)), error=str(e)))
    data = dict(updated=now, host=os.uname().nodename, gpus=gpus, gpu_procs=gpu_processes(), system=system_sample(),
                gpu_history=[[h["t"]] + [[g["util"], g["mem_used"], g["temp"], g["power"]] for g in h["gpus"]]
                             for h in hist],
                runs=runs, released=released_evals(runs_dir))
    blob = json.dumps(data, separators=(",", ":"))
    tmp = os.path.join(out_dir, "data.json.tmp")
    with open(tmp, "w") as f:
        f.write(blob)
    os.replace(tmp, os.path.join(out_dir, "data.json"))
    with open(TEMPLATE) as f:
        html = f.read().replace("/*__DATA__*/null", blob.replace("</", "<\\/"))
    tmp = os.path.join(out_dir, "index.html.tmp")
    with open(tmp, "w") as f:
        f.write(html)
    os.replace(tmp, os.path.join(out_dir, "index.html"))


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(REPO, "runs"))
    ap.add_argument("--out", default=os.path.join(HIMETH, "dashboard"))
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--port", type=int, default=0, help="serve on 127.0.0.1:<port>; 0 = don't serve")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    history = os.path.join(args.out, "gpu_history.jsonl")
    update(args.runs, args.out, history)
    if args.once:
        print("wrote", os.path.join(args.out, "index.html"))
        return
    if args.port:
        handler = lambda *a, **k: Quiet(*a, directory=args.out, **k)  # noqa: E731
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"serving {args.out} on http://127.0.0.1:{args.port}", flush=True)
    while True:
        time.sleep(args.interval)
        try:
            update(args.runs, args.out, history)
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "update failed:", e, flush=True)


if __name__ == "__main__":
    main()
