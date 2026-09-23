"""One GPU job under the shared K1 lock SP/phase_k/K1/gpu.lock.d: atomic mkdir, owner.txt, removed in finally; a
fresh subprocess per job; >= 180 s since this lane's last release before re-acquiring. Usage: lockjob.py <name> <script> [args]"""
import os, shutil, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).parent
LOCK = HERE.parent.parent / "K1" / "gpu.lock.d"
REL = HERE / "lock_releases.log"
name, argv = sys.argv[1], sys.argv[2:]
if REL.exists():
    last = float(REL.read_text().split()[-1])
    time.sleep(max(0.0, 180.0 - (time.time() - last)))
while True:
    try:
        os.mkdir(LOCK)
        break
    except FileExistsError:
        time.sleep(60)
try:
    (LOCK / "owner.txt").write_text(f"Cameron pid={os.getpid()} start={time.strftime('%Y-%m-%d %H:%M:%S')} job={name}\n")
    print(f"{time.strftime('%H:%M:%S')} acquired {name}", flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    rc = subprocess.run([sys.executable] + argv, cwd=HERE, env=env).returncode
finally:
    shutil.rmtree(LOCK, ignore_errors=True)
    with open(REL, "a") as f:
        f.write(f"{name} released {time.strftime('%H:%M:%S')} {time.time()}\n")
print(f"{time.strftime('%H:%M:%S')} released {name} rc={rc}", flush=True)
sys.exit(rc)
