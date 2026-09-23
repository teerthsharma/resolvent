"""House GPU lock: atomic mkdir of SP/phase_k/K1/gpu.lock.d, owner.txt inside, removed in finally.
A lock older than 3 h whose PID is dead is removed and the removal is posted on the board.
Fairness (Dispatcher, 2026-09-23 ~04:45): after this seat releases the lock it waits 3 min before re-acquiring
(release time kept in LAST, shared by every Chase process)."""
import contextlib, datetime, json, os, shutil, time
import psutil

SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
LOCK = SP + "/phase_k/K1/gpu.lock.d"
BOARD = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
LAST = SP + "/phase_k/K1/chase/.gpu_last_release"
GAP_S = 180


def _stale():
    try:
        age = time.time() - os.path.getmtime(LOCK)
        txt = open(LOCK + "/owner.txt").read()
        pid = int(txt.split("pid=")[1].split()[0])
    except Exception:
        return False, ""
    return age > 3 * 3600 and not psutil.pid_exists(pid), txt.strip()


@contextlib.contextmanager
def gpu_lock(tag):
    try:
        wait = GAP_S - (time.time() - float(open(LAST).read()))
    except Exception:
        wait = 0
    if wait > 0:
        time.sleep(wait)
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            stale, txt = _stale()
            if stale:
                shutil.rmtree(LOCK, ignore_errors=True)
                with open(BOARD, "a") as f:
                    f.write(json.dumps({"t": "finding", "agent": "Chase",
                                        "text": "removed stale gpu.lock.d (> 3 h, pid dead): " + txt}) + "\n")
                continue
            time.sleep(60)
    with open(LOCK + "/owner.txt", "w") as f:
        f.write(f"seat=Chase pid={os.getpid()} start={datetime.datetime.now().isoformat(timespec='seconds')} tag={tag}\n")
    try:
        yield
    finally:
        shutil.rmtree(LOCK, ignore_errors=True)
        with open(LAST, "w") as f:
            f.write(str(time.time()))
