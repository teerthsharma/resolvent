"""Drives r3_worker.py: 2 runs without determinism flags, 2 with (R4-(3)), plus
2 more with-flags runs held out as the R4-(4) must-fire reproduction pair.
Same recipe throughout: seed=0, split_seed=0 (fixed data order both ways).
"""
import json
import subprocess
import sys

PY = sys.executable
WORKER = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\r3_worker.py"
OUT = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"

# seed=0 fixed steps=400 hidden=256 L=6 heads=8(d_head=32) seq=256 batch=8 eval_every=100 eval_batches=8 split_seed=0
RECIPE = ["0", "400", "256", "6", "8", "256", "8", "100", "8", "0"]

RUNS = [
    ("nondet", "r3_nondet_1.json"),
    ("nondet", "r3_nondet_2.json"),
    ("det", "r3_det_1.json"),
    ("det", "r3_det_2.json"),
    ("det", "r3_mustfire_1.json"),
    ("det", "r3_mustfire_2.json"),
]

results = []
for kind, fname in RUNS:
    flag = "1" if kind == "det" else "0"
    out_path = OUT + "\\" + fname
    args = [PY, WORKER, flag, out_path] + RECIPE
    print("running", kind, fname, flush=True)
    r = subprocess.run(args, capture_output=True, text=True)
    print(r.stdout[-500:], flush=True)
    if r.returncode != 0:
        print("STDERR", r.stderr[-3000:], flush=True)
        raise SystemExit("run failed: " + fname)
    with open(out_path) as fh:
        rec = json.load(fh)
    results.append((kind, fname, rec["final_eval_loss"], rec["eval_losses"]))

summary_path = OUT + "\\r3_sweep_summary.json"
with open(summary_path, "w") as fh:
    json.dump([dict(kind=k, file=f, final_eval_loss=l, eval_losses=e)
              for k, f, l, e in results], fh, indent=2)
print("SUMMARY written to", summary_path)
for k, f, l, e in results:
    print(k, f, "final_eval_loss=", l)
