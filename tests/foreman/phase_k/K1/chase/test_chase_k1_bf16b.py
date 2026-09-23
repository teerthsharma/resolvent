"""Chase K1 reroute of bf16_leak_fused_read1 (RED 1.39e-5 at S 4096 against its 1e-5 line; the bar could not tell the
leak from fp32's own floor at S 4096). Registered 2026-09-23 after that RED and before the fp32 read of V = 1 on these
inputs was measured; RED first with CHASE_STUB=1.
 bf16_leak_read1_vs_fp32  same inputs as test_chase_k1_ckpt._leak (K0 setup, seed 0, SSMax a = 1, b = 0, g = .999),
                          S in {1024, 4096}, V = 1: the leak's max|x - 1| <= 2 * (the hook's fp32 max|x - 1|) + 1e-6 at
                          both S (the leak adds nothing to the read of 1 beyond the fp32 floor), AND the leak's
                          max|x - 1| < 1e-3 at both S (K0's materialized bf16 W gave a read of 2.53 at S 4096).
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_chase_k1_ckpt as T


@T.bar("bf16_leak_read1_vs_fp32")
def t_leak_vs_fp32():
    T._stub()
    import torch
    import instr as I
    out, ok = {}, True
    for S in (1024, 4096):
        qs, k, v, x1, _ = T._leak(S)
        dl = float((x1 - 1).abs().max())
        df = float((I.hk.read(qs, k, torch.ones_like(v)) - 1).abs().max())
        out[S] = {"leak": dl, "fp32_hook": df}
        ok &= math.isfinite(dl) and dl <= 2 * df + 1e-6 and dl < 1e-3
    return ok, out


if __name__ == "__main__":
    from gpulock import gpu_lock
    if os.environ.get("CHASE_STUB") == "1":
        t_leak_vs_fp32()
    else:
        with gpu_lock("chase.k1 bf16b"):
            t_leak_vs_fp32()
    print("SUMMARY", json.dumps(T.results))
    sys.exit(1 if any(s == "red" for s in T.results.values()) else 0)
