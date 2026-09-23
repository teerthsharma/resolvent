"""Identity-grid ward (Cameron, K2.1; released by the Dispatcher after Inspector K2 PASS2). Runs grid_id_jobs.json in
order (seed-major), each training run and each fR walk as its own lockjob.py hold (fresh subprocess, K1/gpu.lock.d,
180 s cooldown). Every run keeps its launch name; a board line when each run lands. Budget guard: after each landed run,
projected total = landed wall + remaining x 2346 s x (measured / estimated so far) + remaining cooldowns; above 14 h the
ward stops and waits for the Dispatcher. On any rc != 0 it stops. test_k2_grid.py id runs once, only when all 18 runs
(and the 6 fR walks) have landed. State in ward_state.json."""
import json, os, re, subprocess, sys, time
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).parent
K1C = HERE.parent.parent / "K1" / "cameron"
BOARD = Path("C:/Users/seal/Desktop/New folder (32)/house-events.jsonl")
JOBS = json.loads((HERE / "grid_id_jobs.json").read_text())
EST, COOL, LIMIT = 2346.0, 180.0, 14 * 3600.0
STATE = HERE / "ward_state.json"
say = lambda s: print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)


def post(text):
    with open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": "finding", "agent": "Cameron", "text": text}) + "\n")


def state(status, **kw):
    STATE.write_text(json.dumps({"status": status, "pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}, indent=1))


def landed(j):
    d = Path(j["argv"][j["argv"].index("--out") + 1])
    return (d / "result.json").exists() and (not j["fR"] or (d / "walk" / "meta.json").exists()), d


def hold(name, argv, log):
    with open(HERE / "runs" / log, "a") as f:
        return subprocess.run([sys.executable, "lockjob.py", name] + argv, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode


(HERE / "runs").mkdir(exist_ok=True)
state("running")
say(f"ward start pid {os.getpid()}, {len(JOBS)} jobs")
walls = []
for i, j in enumerate(JOBS):
    ok, d = landed(j)
    if ok:
        say(f"skip {j['name']} (landed)")
        walls.append(json.loads((d / "result.json").read_text())["wall_s"])
        continue
    if not (d / "result.json").exists():
        say(f"launch {j['name']}")
        rc = hold(j["name"], [str(K1C / "rdepth.py")] + j["argv"], f"{j['name']}.stdout")
        if rc or not (d / "result.json").exists():
            post(f"grid id run FAILED: {j['name']} rc {rc} (runs/{j['name']}.stdout); ward stopped, no rename, waiting for the Dispatcher")
            state("stopped", reason=f"{j['name']} rc {rc}"); say("stopped"); sys.exit(3)
    if j["fR"]:
        rc = hold(f"{j['name']}_walk", ["k2_ptr.py", "walk", str(d)], f"{j['name']}.walk.log")
        if rc:
            post(f"grid id walk FAILED: {j['name']} rc {rc}; ward stopped, waiting for the Dispatcher")
            state("stopped", reason=f"{j['name']} walk rc {rc}"); say("stopped"); sys.exit(3)
    w = json.loads((d / "result.json").read_text())["wall_s"]
    walls.append(w)
    left = len(JOBS) - (i + 1)
    ratio = sum(walls) / (EST * len(walls))
    proj = sum(walls) + left * EST * ratio + (left + sum(1 for x in JOBS[i + 1:] if x["fR"])) * COOL
    post(f"grid id run landed: {j['name']} ({i + 1}/{len(JOBS)}), rc 0, run wall {w:.0f} s (estimate 2346 s)"
         + (", walk done" if j["fR"] else "") + f"; projected grid total {proj / 3600:.1f} h")
    say(f"landed {j['name']} wall {w:.0f} proj {proj / 3600:.2f} h")
    state("running", landed=i + 1, projected_h=round(proj / 3600, 2))
    if left and proj > LIMIT:
        post(f"grid id budget guard: projected total {proj / 3600:.1f} h > 14 h after {j['name']}; ward stopped before the next run, waiting for the Dispatcher")
        state("stopped", reason="budget guard", projected_h=round(proj / 3600, 2)); say("budget stop"); sys.exit(5)

if not all(landed(j)[0] for j in JOBS):
    state("stopped", reason="not all landed"); sys.exit(4)
TL = HERE / "run_test_k2_grid_id.log"
if TL.exists():
    say("test already run once; not rerun"); state("done"); sys.exit(0)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
p = subprocess.run([sys.executable, "test_k2_grid.py", "id", "runs"], cwd=HERE, env=env, capture_output=True, text=True)
TL.write_text(f"=== run once, all 18 runs landed ({time.strftime('%Y-%m-%d %H:%M:%S')}); ward_grid_id.log\n" + p.stdout + p.stderr + f"exit {p.returncode}\n")
fam = {}
for line in p.stdout.splitlines():
    m = re.match(r"(PASS|FAIL|UNREAD) (cameron\.k2\.\w+)", line)
    if m:
        fam[m.group(2)] = fam.get(m.group(2), True) and m.group(1) == "PASS"
with open(BOARD, "a", encoding="utf-8") as f:
    for name, ok in fam.items():
        f.write(json.dumps({"t": "test", "agent": "Cameron", "status": "green" if ok else "red", "name": name}) + "\n")
last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "no output"
post(f"grid id: test_k2_grid.py id (sha a622cd95) run once after all 18 runs landed: {last}, exit {p.returncode} (run_test_k2_grid_id.log)")
state("done", test_exit=p.returncode)
say("done")
