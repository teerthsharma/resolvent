"""Sparsemax variant cost (path densec: W materialized, sort-based sparsemax, dense trsm) fwd+bwd vs one SDPA fwd+bwd,
B1 H8 D64 fp32, S from argv. Fresh process per S. REPORTED, NOT SCORED. Appends to cost_sparse_rows.jsonl."""
import json, sys, os, numpy as np, torch, torch.nn.functional as F
SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design4x5; assert design4x5.poll_until_free(timeout_s=600)
import resolvent as R; from test_chase_k0 import _qkv
S = int(sys.argv[1]); reps = 5
q, k, v = _qkv(S); qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda")).contiguous()
gout = _qkv(S, seed=3)[0]; L = [t.clone().requires_grad_() for t in (qs, k, v)]
def sd():
    for t in L: t.grad = None
    F.scaled_dot_product_attention(*L, is_causal=True).backward(gout)
def sp():
    for t in L: t.grad = None
    R.resolvent(*L, 0.99, kind="sparsemax", path="densec").backward(gout)
arms = dict(sdpa_fb=sd, sparsemax_fb=sp)
for f in arms.values(): f()
torch.cuda.synchronize(); ts = {n: [] for n in arms}
for _ in range(reps):
    for n, f in arms.items():
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); f(); b.record(); torch.cuda.synchronize(); ts[n].append(a.elapsed_time(b))
med = {n: float(np.median(t)) for n, t in ts.items()}
row = dict(S=S, path="densec sparsemax", ms=med, ratio_fwd_bwd=med["sparsemax_fb"] / med["sdpa_fb"],
           peak_mib=torch.cuda.max_memory_allocated() / 2**20, reps=reps)
print(json.dumps(row)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_sparse_rows.jsonl"), "a").write(json.dumps(row) + "\n")
