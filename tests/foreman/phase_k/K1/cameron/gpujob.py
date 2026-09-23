"""Run GPU jobs one at a time under the shared lock SP/phase_k/K1/gpu.lock.d (mkdir = atomic acquire).
Usage: python gpujob.py <jobs.json>   (a list of {"name": ..., "argv": [...]}; each runs in a fresh subprocess).
The lock is taken per job and removed in `finally`. Skips a job whose out dir already holds result.json."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
LOCK = HERE.parent / "gpu.lock.d"


def run(job, start):
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            time.sleep(60)
    try:
        hits = new_cmax(start)  # Dispatcher after PASS4: re-check the board after acquiring, before the arm starts
        if hits:
            print(f"{time.strftime('%H:%M:%S')} STOP after acquiring, before {job['name']}: {hits[0][:300]}", flush=True)
            return False
        (LOCK / "owner.txt").write_text(f"Cameron pid={os.getpid()} start={time.strftime('%Y-%m-%d %H:%M:%S')} job={job['name']}\n")
        t0 = time.time()
        with open(HERE / "runs" / f"{job['name']}.stdout", "a") as f:
            rc = subprocess.run([sys.executable, str(HERE / "rdepth.py")] + job["argv"], stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"{time.strftime('%H:%M:%S')} {job['name']} rc={rc} {time.time() - t0:.0f}s", flush=True)
    finally:
        shutil.rmtree(LOCK, ignore_errors=True)
    time.sleep(180)  # fairness rule (Dispatcher after PASS4): wait 3 min after releasing before re-acquiring
    return True


BOARD = Path("C:/Users/seal/Desktop/New folder (32)/house-events.jsonl")


def new_cmax(start_line):
    """Dispatcher rule (after Inspector PASS3): before each far-band arm, stop if Foreman posted a new c_max."""
    lines = BOARD.read_text(encoding="utf-8", errors="replace").splitlines()[start_line:]
    out = []
    for l in lines:
        try:
            e = json.loads(l)
        except ValueError:
            continue
        txt = str(e.get("text", ""))
        if e.get("agent") == "Foreman" and e.get("t") == "finding" and ("c_max" in txt or "c max" in txt.lower() or "reach factor" in txt.lower()):
            out.append(l)
    return out


if __name__ == "__main__":
    (HERE / "runs").mkdir(exist_ok=True)
    start = int(sys.argv[2]) if len(sys.argv) > 2 else len(BOARD.read_text(encoding="utf-8", errors="replace").splitlines())
    for job in json.loads(Path(sys.argv[1]).read_text()):
        hits = new_cmax(start)
        if hits:
            print(f"{time.strftime('%H:%M:%S')} STOP before {job['name']}: Foreman c_max/far-band line on the board: {hits[0][:300]}", flush=True)
            break
        out = Path(job["argv"][job["argv"].index("--out") + 1])
        if (out / "result.json").exists():
            print(f"skip {job['name']} (done)", flush=True)
            continue
        if not run(job, start):
            break
