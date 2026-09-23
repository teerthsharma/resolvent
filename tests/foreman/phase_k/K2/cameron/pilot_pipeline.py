"""K2.1 pilot pipeline (Cameron): wait for Chase's READY_SP, then pilot_fR_sp_s0 (as registered in test_k2_pilot.py)
under the GPU lock, then its walk (k2_ptr.py walk), then test_k2_pilot.py; board test lines per row family.
No launch if READY_SP names no hook .py in K2/chase that defines ResolventAttention."""
import json, os, re, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).parent
CHASE = HERE.parent / "chase"
READY = CHASE / "READY_SP"
K1C = HERE.parent.parent / "K1" / "cameron"
BOARD = Path("C:/Users/seal/Desktop/New folder (32)/house-events.jsonl")
say = lambda s: print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)

deadline = time.time() + 6 * 3600
while not READY.exists():
    if time.time() > deadline:
        say("timeout: no READY_SP after 6 h; nothing launched"); sys.exit(4)
    time.sleep(60)
time.sleep(10)
txt = READY.read_text(errors="replace")
say("READY_SP:\n" + txt[:3000])
cands = [c for c in dict.fromkeys(re.findall(r"[\w.\-]+\.py", txt)) if (CHASE / c).is_file()
         and "class ResolventAttention" in (CHASE / c).read_text(errors="replace") and "stub" not in c]
if not cands:
    say("ABORT: READY_SP names no non-stub hook .py in K2/chase defining ResolventAttention; nothing launched"); sys.exit(2)
spec = str((CHASE / cands[0]).resolve()).replace("\\", "/") + ":ResolventAttention"
say(f"hook {spec}")
run = HERE / "runs" / "pilot_fR_sp_s0"
argv = [sys.executable, "lockjob.py", "pilot_fR_sp_s0", str(K1C / "rdepth.py"), "--arm", "fR", "--layers", "4", "--hook", spec,
        "--bed", "kpf", "--emb", "frozen", "--par_init", "orth", "--lr", "3e-3", "--steps", "24000", "--probe_every", "2000",
        "--eval_ns", "1024,4096,8192,16384", "--gamma_sched", "0.5:6000,0.9:12000,0.99:18000,0.999", "--seed", "0", "--out", str(run)]
with open(HERE / "runs" / "pilot_fR_sp_s0.stdout", "a") as f:
    rc = subprocess.run(argv, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
say(f"pilot rc={rc}")
if rc:
    sys.exit(rc)
with open(HERE / "k2_pilot_walk.log", "a") as f:
    rc = subprocess.run([sys.executable, "lockjob.py", "pilot_walk", "k2_ptr.py", "walk", str(run)], cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
say(f"walk rc={rc}")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
p = subprocess.run([sys.executable, "test_k2_pilot.py", "0", "runs"], cwd=HERE, env=env, capture_output=True, text=True)
(HERE / "run_test_k2_pilot.log").write_text(f"=== run on the measurement ({time.strftime('%Y-%m-%d %H:%M:%S')}); producer logs runs/pilot_fR_sp_s0.stdout, k2_pilot_walk.log\n"
                                            + p.stdout + p.stderr + f"exit {p.returncode}\n")
say(p.stdout[-3000:])
fam = {}
for line in p.stdout.splitlines():
    m = re.match(r"(PASS|FAIL|UNREAD) (cameron\.k2\.\w+)", line)
    if m:
        fam[m.group(2)] = fam.get(m.group(2), True) and m.group(1) == "PASS"
with open(BOARD, "a", encoding="utf-8") as f:
    for name, ok in fam.items():
        f.write(json.dumps({"t": "test", "agent": "Cameron", "status": "green" if ok else "red", "name": name}) + "\n")
say(f"board: {len(fam)} test lines; test exit {p.returncode}")
