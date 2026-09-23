# Foreman K1: the ALiBi twin's absolute-position channel on bed_k. Bars: test_poschan.py (stated construction) and
# test_k1c.py harness_channel_coarse (bias rounded to bfloat16 as K0/wilson/train_ladder.py builds it under autocast).
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
K0 = "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K0"
sys.path[:0] = [K0 + "/cameron", K0 + "/wilson"]
import bed_k                      # noqa: E402
from train_ladder import get_slopes  # noqa: E402

HERE = Path(__file__).parent
N, LO = 16384, 4096


def head_out(roots, m, bias_dtype, out_dtype, block=512):
    """One ALiBi head, content logit 0, value 1 on roots: o(t) = ALiBi softmax mass on the roots. Softmax in float32."""
    v = torch.from_numpy(roots.astype(np.float32))
    j = torch.arange(N, dtype=torch.float32)
    out = torch.empty(N, dtype=torch.float32)
    for t0 in range(0, N, block):
        i = torch.arange(t0, t0 + block, dtype=torch.float32)[:, None]
        b = (-m * (i - j[None, :])).to(bias_dtype).float()     # train_ladder: fp32 bias, then .to(q.dtype)
        b = b.masked_fill(j[None, :] > i, float("-inf"))
        out[t0:t0 + block] = torch.softmax(b, dim=1) @ v
    return out.to(out_dtype).double().numpy()


def pair_err(o, m):
    e = -np.log(o[LO:]) / m - np.arange(LO, N)
    return float(e.max() - e.min())


def main():
    slopes = get_slopes(2)
    m = float(slopes[1])
    res = {"slope": m, "ladder_slopes_R0": [float(s) for s in slopes], "bf16_max_pair_err": 0.0, "fp32_max_pair_err": 0.0}
    harness = 0.0
    for k in range(4):
        b = bed_k.make_test(np.random.default_rng([81, k]), N)
        roots = b["parent"] < 0
        res["fp32_max_pair_err"] = max(res["fp32_max_pair_err"], pair_err(head_out(roots, m, torch.float32, torch.float32), m))
        res["bf16_max_pair_err"] = max(res["bf16_max_pair_err"], pair_err(head_out(roots, m, torch.float32, torch.bfloat16), m))
        harness = max(harness, pair_err(head_out(roots, m, torch.bfloat16, torch.bfloat16), m))
    (HERE / "poschan.json").write_text(json.dumps(res, indent=1))
    (HERE / "poschan_harness.json").write_text(json.dumps({"max_pair_err": harness, "slope": m, "beds": 4, "t_range": [LO, N]}, indent=1))
    return res, harness


if __name__ == "__main__":
    print(main())
