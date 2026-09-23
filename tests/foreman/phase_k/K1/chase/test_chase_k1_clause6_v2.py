"""Chase K1 clause 6 v2 (Dispatcher's amendment). Registered 2026-09-23 ~04:40, before any checkpoint was read; RED first
with CHASE_STUB=1. Uses test_chase_k1_ckpt_v2.measure() (instr2: the resolvent layer's own fp32 qs, k, v captured
under the trainer's bf16 autocast; Cameron checkpoints at S 1024 and 16384 on their own evaluation beds, Wilson
checkpoints at ctx on the first eval draw of 2 sequences).
 clause6_v2        >= 1 finished (f_R) R0 checkpoint, and on every (checkpoint, S): g * max_i rowsum(P) < 1 (P = fp32
                   max-shifted softmax row blocks, c = 256, g = .999) AND read1_dev <= 2 * floor, where
                   read1_dev = max|x - 1| of the hook's read of V = 1 (fs5c, fp32) and
                   floor = max|x_hook - x64| / max|x64| of the hook's fp32 read of the CAPTURED v on the same qs, k,
                   against a float64 blocked forward substitution (row blocks of 1024, W rows materialized in float64,
                   naive pivot 1 - g W_ii as K0's float64 reference).
                   The original line (c6_numerics_R0_v2, read1_dev <= 1e-5) stays registered and is reported beside it;
                   a 16k RED there is recorded as "the line sits below the fp32 floor" with the floor number.
 clause6_v2_power  the ratio can fail on these inputs: on >= 1 (checkpoint, S) the K0 fs5 forward (naive pivot, fp32)
                   gives read1_dev_fs5 > 2 * floor. (If RED, clause6_v2 is vacuous on these checkpoints.)
Also runs the v2 checkpoint bars (c6_numerics_R0_v2 ... rrange_aL_slope_R0_v2) from test_chase_k1_ckpt_v2.py in the same
process, so both clause-6 lines are read from one capture.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_chase_k1_ckpt_v2 as T

FLOOR = {}


def x64_blocked(qs, k, v, g, c=1024):
    import torch, math
    q64, k64, v64 = qs.double(), k.double(), v.double()
    S, D = q64.shape[-2], q64.shape[-1]
    X = torch.empty_like(v64)
    for lo in range(0, S, c):
        hi = min(S, lo + c)
        z = (q64[..., lo:hi, :] @ k64[..., :hi, :].transpose(-1, -2)) / math.sqrt(D)
        i = torch.arange(lo, hi, device=z.device)[:, None]
        z = z.masked_fill(torch.arange(hi, device=z.device)[None, :] > i, float("-inf"))
        W = torch.softmax(z, -1)
        rhs = (1 - g) * v64[..., lo:hi, :]
        if lo:
            rhs = rhs + g * (W[..., :lo] @ X[..., :lo, :])
        A = torch.eye(hi - lo, dtype=torch.float64, device=z.device) - g * W[..., lo:hi]
        X[..., lo:hi, :] = torch.linalg.solve_triangular(A, rhs, upper=False)
    return X


def floors():
    if FLOOR:
        return FLOOR
    import torch
    import instr2 as I
    for ck, cap, rec in T._fr():
        for S, c in cap.items():
            qs, k, v = c["res"]["qs"].float(), c["res"]["k"].float(), c["res"]["v"].float()
            x = I.hk.read(qs, k, v)
            ref = x64_blocked(qs, k, v, I.GAMMA)
            FLOOR[(ck["path"], S)] = float((x.double() - ref).abs().max() / ref.abs().max())
            del ref
            torch.cuda.empty_cache()
    return FLOOR


def _cells():
    fl = floors()
    return [(ck, S, row, fl[(ck["path"], S)]) for ck, _, rec in T._fr() for S, row in rec["S"].items()]


@T.bar("clause6_v2")
def t_c6v2():
    T._stub()
    cells = _cells()
    out, ok = {}, len(cells) >= 1
    for ck, S, row, fl in cells:
        good = row["gamma_max_rowsum"] < 1 and row["finite"] and row["read1_dev"] <= 2 * fl
        ok &= good
        out[f"{ck['lane']}:{os.path.basename(os.path.dirname(ck['path']))}:S{S}"] = {
            "g_max_rowsum": row["gamma_max_rowsum"], "read1_dev": row["read1_dev"], "floor": fl,
            "ratio": row["read1_dev"] / fl if fl > 0 else None, "line_1e-5": row["read1_dev"] <= 1e-5, "ok": good}
    return ok, {"n_ckpt": len(T._fr()), **out}


@T.bar("clause6_v2_power")
def t_c6v2_power():
    T._stub()
    cells = _cells()
    n = sum(row["read1_dev_fs5"] > 2 * fl for _, _, row, fl in cells)
    return n >= 1, {"cells": len(cells), "fs5_over_2floor": n,
                    "fs5_ratio": [row["read1_dev_fs5"] / fl if fl > 0 else None for _, _, row, fl in cells]}


if __name__ == "__main__":
    v2 = [T.t_c6, T.t_c6_power, T.t_c7, T.t_c7_route, T.t_rr_depth, T.t_rr_span, T.t_rr_al]
    mine = [t_c6v2, t_c6v2_power]
    from gpulock import gpu_lock
    if os.environ.get("CHASE_STUB") == "1":
        for t in mine:
            t()
    else:
        with gpu_lock("chase.k1 clause6_v2 + ckpt v2"):
            for t in v2 + mine:
                t()
    print("SUMMARY", json.dumps(T.results))
    sys.exit(1 if any(s == "red" for s in T.results.values()) else 0)
