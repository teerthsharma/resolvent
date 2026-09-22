"""Forward+backward cost of the resolvent layer (path fusedcg, graphed; eager fusedc also timed = fs5c forward + backward_blocked) vs ONE SDPA
forward+backward of the same shape (default dispatch), B1 H8 D64 fp32, tf32 off, SSMax a=1 b=0 pre-applied to q in
both arms (the scale multiply is outside both timed regions). Arms alternate inside one loop; CUDA events; median.
Usage: python cost.py S   (one fresh process per S). Appends a row to cost_rows.jsonl. REPORTED, NOT SCORED."""
import json, sys, os, math
import numpy as np
import torch, torch.nn.functional as F
SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design4x5
assert design4x5.poll_until_free(timeout_s=600), "card not free"
import resolvent as R
from test_chase_k0 import _qkv
torch.backends.cuda.matmul.allow_tf32 = False
S = int(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 9
q, k, v = _qkv(S, seed=0)
qs = R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda")).contiguous()
gout = _qkv(S, seed=3)[0]
leaves = [t.clone().requires_grad_() for t in (qs, k, v)]


def sdpa_fb():
    for t in leaves: t.grad = None
    F.scaled_dot_product_attention(*leaves, is_causal=True).backward(gout)


def res_fb():
    for t in leaves: t.grad = None
    R.resolvent(*leaves, 0.99, path="fusedcg").backward(gout)


def res_fb_eager():
    for t in leaves: t.grad = None
    R.resolvent(*leaves, 0.99, path="fusedc").backward(gout)


def sdpa_f():
    with torch.no_grad(): F.scaled_dot_product_attention(qs, k, v, is_causal=True)


def res_f():
    with torch.no_grad(): R.resolvent(qs, k, v, 0.99, path="fusedcg")


arms = dict(sdpa_fb=sdpa_fb, res_fb=res_fb, res_fb_eager=res_fb_eager, sdpa_f=sdpa_f, res_f=res_f)
for f in arms.values(): f()
torch.cuda.synchronize()
ts = {n: [] for n in arms}
for _ in range(reps):
    for n, f in arms.items():
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); f(); b.record(); torch.cuda.synchronize()
        ts[n].append(a.elapsed_time(b))
med = {n: float(np.median(t)) for n, t in ts.items()}
# forward accuracy at this S vs float64 dense solve, one head at a time (Cameron's reference form)
with torch.no_grad():
    x = R.fs5c(qs, k, v, 0.99)
    errs, mx = 0.0, 0.0
    mask = torch.ones(S, S, dtype=torch.bool, device="cuda").triu(1)
    eye = torch.eye(S, dtype=torch.float64, device="cuda")
    for h in range(8):
        z = (qs[0, h].double() @ k[0, h].double().T) / 8.0
        W = torch.softmax(z.masked_fill(mask, float("-inf")), -1); del z
        ref = torch.linalg.solve_triangular(eye - 0.99 * W, 0.01 * v[0, h].double(), upper=False); del W
        errs = max(errs, float((x[0, h].double() - ref).abs().max())); mx = max(mx, float(ref.abs().max()))
row = dict(S=S, B=1, H=8, D=64, gamma=0.99, dtype="fp32", tf32=False, ssmax=dict(a=1.0, b=0.0), reps=reps,
           path="fusedcg (graphed fs5c + graphed backward_blocked)", ms=med, ratio_fwd_bwd=med["res_fb"] / med["sdpa_fb"], ratio_fwd_bwd_eager=med["res_fb_eager"] / med["sdpa_fb"], ratio_fwd=med["res_f"] / med["sdpa_f"],
           ms_res_bwd_only=med["res_fb"] - med["res_f"], ms_sdpa_bwd_only=med["sdpa_fb"] - med["sdpa_f"],
           fwd_rel_err_vs_f64=errs / mx, peak_mib=torch.cuda.max_memory_allocated() / 2**20,
           raw=ts, torch=torch.__version__, gpu=torch.cuda.get_device_name())
print(json.dumps({kk: vv for kk, vv in row.items() if kk != "raw"}), flush=True)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_rows.jsonl"), "a").write(json.dumps(row) + "\n")
