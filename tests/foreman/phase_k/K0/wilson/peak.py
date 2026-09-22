"""Attained bf16 matmul peak, C24 recipe: 4096^3 bf16, 10 warmup + 50 timed, repeated 5x. Polls the card first."""
import json, sys, time
sys.path.insert(0, "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad")
import design4x5
assert design4x5.poll_until_free(timeout_s=600), "GPU busy"
import torch
torch.manual_seed(0)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
a, b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16), torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
res = []
for rep in range(5):
    for _ in range(10): a @ b
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50): a @ b
    torch.cuda.synchronize(); dt = time.time() - t
    res.append(2 * n ** 3 * 50 / dt / 1e12)
print(json.dumps({"n": n, "tflops": res, "device": torch.cuda.get_device_name(0),
                  "clock_mhz": __import__("subprocess").check_output(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.limit", "--format=csv,noheader"]).decode().strip()}))
