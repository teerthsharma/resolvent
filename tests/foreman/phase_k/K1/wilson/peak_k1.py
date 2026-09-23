"""bf16 matmul peak, C24 recipe (4096^3, 10 warmup + 50 timed, 5 reps), with temperature and SM clock read before
and after every rep. python peak_k1.py SESSION LABEL OUT.json  (run only under gpujob.py, i.e. holding the lock)."""
import datetime, json, statistics, subprocess, sys, time
import torch

Q = "temperature.gpu,clocks.sm,clocks.max.sm,power.draw"
tele = lambda: subprocess.check_output(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader"]).decode().strip()
session, label, out = sys.argv[1:4]
t0 = time.time()
torch.manual_seed(0)
n = 4096
a, b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16), torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
tf, tel = [], []
for rep in range(5):
    before = tele()
    for _ in range(10): a @ b
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50): a @ b
    torch.cuda.synchronize(); dt = time.time() - t
    tf.append(2 * n ** 3 * 50 / dt / 1e12)
    tel.append({"before": before, "after": tele(), "query": Q})
rec = {"session": session, "label": label, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "n": n,
       "tflops": tf, "median": statistics.median(tf), "drift_last_vs_first": tf[-1] / tf[0] - 1,
       "t_start": t0, "t_end": time.time(), "iso_end": datetime.datetime.now().isoformat(timespec="seconds"), "telemetry": tel}
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("label", "tflops", "median", "drift_last_vs_first")}))
