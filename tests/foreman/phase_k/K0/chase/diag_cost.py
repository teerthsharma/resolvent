# diagnostic (not a bar): is the fs5c forward launch-bound here? fs5 vs fs5c, eager vs graphed, S=4096
import sys, os, time, numpy as np, torch, torch.nn.functional as F
SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
sys.path.insert(0, SP); sys.path.insert(0, ".")
import design4x5; assert design4x5.poll_until_free(timeout_s=600)
import resolvent as R; from test_chase_k0 import _qkv
q, k, v = _qkv(4096); cn = R._cost_n()
R.fused_forward(q, k, v, 0.99)
arms = {"sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        "fs5": lambda: cn.fs5(q, k, v, 256), "fs5c": lambda: R.fs5c(q, k, v, 0.99),
        "fs5g": lambda: cn.graphed(cn.fs5, q, k, v, 256),
        "fs5cg": lambda: cn.graphed(lambda q, k, v, c: R.fs5c(q, k, v, 0.99, c), q, k, v, 256)}
with torch.no_grad():
    for f in arms.values(): f()
    torch.cuda.synchronize()
    ts = {n: [] for n in arms}; cpu = {n: [] for n in arms}
    for _ in range(9):
        for n, f in arms.items():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); a.record(); f(); b.record(); t1 = time.perf_counter(); torch.cuda.synchronize()
            ts[n].append(a.elapsed_time(b)); cpu[n].append((t1 - t0) * 1e3)
print({n: (round(float(np.median(t)), 2), "cpu_launch_ms", round(float(np.median(cpu[n])), 2)) for n, t in ts.items()})
print("loadavg-ish: os.cpu_count", os.cpu_count())
