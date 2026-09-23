"""GPU lock launcher (house rule). python gpujob.py SPEC.json
SPEC = {"tag": str, "session": str, "record": jobs.jsonl path, "jobs": [{"name", "cmd": [..], "out": path}]}
Acquires SP/phase_k/K1/gpu.lock.d by atomic mkdir (retry every 60 s), writes owner.txt, runs every job as a
fresh subprocess while holding it, releases in finally. A lock older than 3 h whose PID is dead is removed and
the removal is posted to the board. Samples nvidia-smi compute PIDs every 30 s and records any foreign ones."""
import datetime, json, os, re, shutil, subprocess, sys, threading, time
import psutil

SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
LOCK = SP + "/phase_k/K1/gpu.lock.d"
BOARD = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
LAST = SP + "/phase_k/K1/wilson/last_release.txt"  # dispatcher rule: no re-acquire within 180 s of our own release
now_iso = lambda: datetime.datetime.now().isoformat(timespec="seconds")


def gpu_pids():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], timeout=20)
        return [int(x) for x in out.decode(errors="ignore").split() if x.strip().isdigit()]
    except Exception:
        return None


def try_clear_stale():
    try:
        age = time.time() - os.stat(LOCK).st_ctime
        txt = open(os.path.join(LOCK, "owner.txt")).read()
    except OSError:
        return
    m = re.search(r"pid\D{0,3}(\d+)", txt, re.I)
    if age > 3 * 3600 and m and not psutil.pid_exists(int(m.group(1))):
        shutil.rmtree(LOCK, ignore_errors=True)
        with open(BOARD, "a") as f:
            f.write(json.dumps({"t": "finding", "agent": "Wilson", "text": f"removed stale gpu.lock.d: age {age / 3600:.2f} h, owner PID {m.group(1)} dead; owner.txt was: {txt.strip()[:200]}"}) + "\n")


def acquire(tag):
    try:
        wait = 180 - (time.time() - float(open(LAST).read()))
    except (OSError, ValueError):
        wait = 0
    if wait > 0:
        print(f"[LOCK] {now_iso()} cooldown {wait:.0f} s after own release", flush=True)
        time.sleep(wait)
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            try_clear_stale()
            print(f"[LOCK] {now_iso()} held, waiting 60 s", flush=True)
            time.sleep(60)
    with open(os.path.join(LOCK, "owner.txt"), "w") as f:
        f.write(f"seat=Wilson pid={os.getpid()} start={now_iso()} tag={tag}\n")


def release():
    try:
        if f"pid={os.getpid()} " in open(os.path.join(LOCK, "owner.txt")).read():
            shutil.rmtree(LOCK)
            open(LAST, "w").write(str(time.time()))
    except OSError:
        pass


def run_job(job, spec):
    foreign, sightings, stop = set(), [], threading.Event()
    os.makedirs(os.path.dirname(os.path.abspath(job["out"])), exist_ok=True)
    t0, iso0 = time.time(), now_iso()
    with open(job["out"], "a") as out:
        p = subprocess.Popen(job["cmd"], stdout=out, stderr=subprocess.STDOUT)
        mine = {p.pid, os.getpid()}

        def sample():
            while not stop.wait(30):
                pids = gpu_pids() or []
                try:
                    kids = {c.pid for c in psutil.Process(p.pid).children(recursive=True)}
                except psutil.Error:
                    kids = set()
                for fp in set(pids) - mine - kids:
                    foreign.add(fp)
                    try:
                        nm = psutil.Process(fp).name()
                    except psutil.Error:
                        nm = None
                    sightings.append([now_iso(), fp, nm])
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        rc = p.wait()
        stop.set()
    rec = {"session": spec["session"], "tag": spec["tag"], "name": job["name"], "cmd": job["cmd"], "out": job["out"],
           "start": iso0, "end": now_iso(), "t_start": t0, "t_end": time.time(), "wall_s": time.time() - t0, "rc": rc,
           "gpu_pids_before": job["_before"], "foreign_gpu_pids_seen": sorted(foreign), "foreign_sightings": sightings, "gpu_pids_after": gpu_pids()}
    with open(spec["record"], "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[JOB] {job['name']} rc={rc} wall_s={rec['wall_s']:.1f}", flush=True)
    return rc


def main():
    spec = json.load(open(sys.argv[1]))
    acquire(spec["tag"])
    print(f"[LOCK] {now_iso()} acquired by pid {os.getpid()}", flush=True)
    try:
        for job in spec["jobs"]:
            job["_before"] = gpu_pids()
            run_job(job, spec)
    finally:
        release()
        print(f"[LOCK] {now_iso()} released", flush=True)


if __name__ == "__main__":
    main()
