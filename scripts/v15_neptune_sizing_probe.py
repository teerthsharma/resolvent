"""NEPTUNE's v15 sizing probe: validate the calibrated memory model, then re-fit
the wall-clock law on THIS box.

WHY THIS FILE EXISTS. `ceq/sizing.py` carries CALIBRATED constants
(`C_OPERATOR = 3.9`, `DTYPE_MODES['bf16_autocast'] = (2.2, 3.4)`) fitted on an
LM-shaped workload: `ceq.lm.TinyLM`, B=4 d=256 L=4 H=4, a [B,H,S,S] operator and
a [B,S,V] cross-entropy head. The v15 deciding cells R1/R2 are a DIFFERENT shape
-- `scale/m3_capability.Arm`, ONE layer, ONE head, `d_model = 16`, `s = 64`, a
scalar readout, and FULL-BATCH training where the batch dimension IS `n`. A
constant calibrated on one shape and applied to another is exactly how the old
line came to be 6.63x low. So the model is re-validated here against the
allocator before any R2 verdict is read off it.

L-LEAN. Nothing here trains. The data is `torch.randn`, not the chain corpus;
no NRMSE, no eval, no seed sweep, no verdict. Step cost is a function of TENSOR
SHAPE, not of tensor values, so random data times the real thing; the arm and
the optimizer are the shipped ones so the shapes are not a local invention.

WHAT IS MEASURED
  A  environment: GPU name, total/free VRAM, host RAM.
  B  CUDA peak-allocated vs `sizing.activation_bytes` at n = 256/512/1024, and
     at s = 64/128/256, fp32. The n-sweep validates the extrapolation to
     n = 32768 (both model terms are LINEAR in n, so a ratio measured at n=256
     is the ratio at n=32768); the s-sweep separates the residual term from the
     [n,s,s] operator term and so back-solves C_OPERATOR at this shape.
  C  CPU per-step wall clock at four n, threads pinned, log-log OLS. The
     inherited figure is `secs ~ n^1.338` (endpoint, threads=12, steps=150,
     t*=8; V13_DAG_TASKLIST.md:88-91). L-TIME requires it re-measured here.
  D  the same on CUDA, because the previous loop died of CPU unaffordability
     with a CUDA device idle in the same box.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import sizing                                   # noqa: E402
from scale.m3_capability import Arm, D_MODEL, LR         # noqa: E402

S, GIB = 64, 1024 ** 3
THREADS = 12          # the pin the inherited timing was taken at
WARMUP, TIMED = 2, 12


def arm_cfg(seq: int = S) -> sizing.Config:
    """`sizing.Config` for the BED-M arm.

    `n_layers = 1` and `n_heads = 1`: the arm is one operator, one MLP, no stack.
    `vocab = 1`: the readout is `nn.Linear(d_model, 1)`, so `C_HEAD * n * s * V`
    is 1/32000 of what it models on an LM and is kept only so the call is the
    module's own. `d_ff_mult` is IRRELEVANT here -- it enters `params` and
    `flops_per_token`, never `activation_bytes` -- so the arm's 8x MLP (HIDDEN
    128 against d_model 16) is not expressible in this Config and is one of the
    two known reasons C_RESIDUAL may not transfer. Stated, not hidden.
    """
    return sizing.Config(d_model=D_MODEL, n_layers=1, n_heads=1, d_head=D_MODEL,
                         vocab=1, seq=seq, d_ff_mult=8)


def predicted(n: int, seq: int = S, dtype: str = "fp32") -> dict:
    """The module's own numbers, and the two terms separately.

    `arm="signed"` and NOT `arm="softmax"`. In `ceq/sizing.py` "softmax" means
    the FUSED SDPA path, whose activation memory is O(S) because no [S,S] tensor
    is ever formed. `scale/m3_capability.Arm`'s softmax branch calls
    `ceq.bench._softmax_operator`, which materialises [n,s,s] explicitly and hands
    it to autograd. On the memory axis the BED-M softmax arm is a "signed" arm:
    same operator tensor, different nonlinearity. Calling `arm="softmax"` here
    would drop the whole operator term and under-predict by the exact quantity
    this node exists to price.
    """
    cfg = arm_cfg(seq)
    rb, ob = sizing.DTYPE_MODES[dtype] if dtype in sizing.DTYPE_MODES else (4.0, 4.0)
    resid = sizing.C_RESIDUAL * cfg.n_layers * n * seq * cfg.d_model * rb
    op = sizing.C_OPERATOR * cfg.n_layers * cfg.n_heads * n * seq * seq * ob
    head = sizing.C_HEAD * n * seq * cfg.vocab * 4
    total = sizing.activation_bytes(cfg, batch=n, arm="signed", dtype=dtype)
    assert abs(total - (resid + op + head)) < 1.0, (total, resid + op + head)
    return dict(resid=resid, operator=op, head=head, total=total)


def measure_peak(n: int, seq: int, device: str, kind: str = "softmax") -> dict:
    """Peak CUDA bytes for ONE forward + backward, above the resident baseline.

    Baseline is taken AFTER the model, the inputs and the mask cache exist, so
    what is returned is activation memory and gradient memory only -- the same
    quantity `sizing.activation_bytes` claims to predict. Parameter gradients
    are 2,689 floats at this shape (10.8 KB) and are left inside the number
    rather than subtracted, because they are inside the model's `C_RESIDUAL`
    fit too.
    """
    torch.manual_seed(0)
    model = Arm(kind, seq).to(device)
    x = torch.randn(n, seq, D_MODEL, device=device)
    y = torch.randn(n, device=device)
    model(x[:1])                       # warm the mask cache and cuBLAS workspace
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.nn.functional.mse_loss(model(x), y).backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del model, x, y
    torch.cuda.empty_cache()
    return dict(n=n, seq=seq, measured=float(peak))


def time_steps(n: int, seq: int, device: str, steps: int, kind: str = "softmax") -> float:
    """Median seconds per optimizer step: forward + backward + step.

    Median over the timed steps, not the mean: on a laptop the scheduler
    occasionally donates a 3x outlier and a mean would carry it into the fit.
    """
    torch.manual_seed(0)
    model = Arm(kind, seq).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.randn(n, seq, D_MODEL, device=device)
    y = torch.randn(n, device=device)
    per = []
    for i in range(WARMUP + steps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= WARMUP:
            per.append(dt)
    del model, opt, x, y
    if device == "cuda":
        torch.cuda.empty_cache()
    return float(statistics.median(per))


def ols_loglog(xs, ys) -> dict:
    """Slope, intercept and R^2 of `log y = a + b log x`.

    OLS over ALL points, not the two endpoints. `V13_CLAIM_AUDIT.md:272-278`
    records that the inherited 1.338 is an ENDPOINT slope and that the same
    three points give 1.317 by OLS -- the relation is convex over the measured
    range, so the two disagree by construction. Both are reported below so the
    comparison is like against like.
    """
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((yy - (a + b * xx)) ** 2 for xx, yy in zip(lx, ly))
    ss_tot = sum((yy - my) ** 2 for yy in ly)
    endpoint = (ly[-1] - ly[0]) / (lx[-1] - lx[0])
    return dict(slope=b, intercept=a, r2=1.0 - ss_res / ss_tot,
                endpoint_slope=endpoint,
                resid=[yy - (a + b * xx) for xx, yy in zip(lx, ly)])


def main() -> int:
    out = {}
    print("=" * 78)
    print("A. ENVIRONMENT")
    print("=" * 78)
    cuda = torch.cuda.is_available()
    print("torch %s   cuda_available=%s" % (torch.__version__, cuda))
    if cuda:
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print("gpu            %s  sm_%d%d  MPs=%d" % (p.name, p.major, p.minor,
                                                      p.multi_processor_count))
        print("vram total     %.3f GiB (%d B)" % (total / GIB, total))
        print("vram free      %.3f GiB (%d B)" % (free / GIB, free))
        out["gpu"] = dict(name=p.name, total=total, free=free)
    try:
        import psutil
        vm = psutil.virtual_memory()
        print("host RAM       %.2f GiB total, %.2f GiB available (%.1f%% used)"
              % (vm.total / GIB, vm.available / GIB, vm.percent))
        out["ram"] = dict(total=vm.total, available=vm.available)
    except ImportError:
        print("host RAM       NOT MEASURED (psutil absent)")
    print("cell shape     s=%d d_model=%d layers=1 heads=1, FULL BATCH (batch == n)"
          % (S, D_MODEL))

    print()
    print("=" * 78)
    print("B1. MEMORY MODEL vs ALLOCATOR -- n sweep at s=%d, fp32, CUDA" % S)
    print("=" * 78)
    if not cuda:
        print("NOT MEASURED: no CUDA device.")
    else:
        print("%8s %12s %12s %12s %12s %7s" %
              ("n", "pred_resid", "pred_op", "pred_total", "measured", "m/p"))
        rows = []
        for n in (256, 512, 1024, 2048):
            pr = predicted(n)
            ms = measure_peak(n, S, "cuda")
            r = ms["measured"] / pr["total"]
            rows.append(dict(**pr, **ms, ratio=r))
            print("%8d %12.3f %12.3f %12.3f %12.3f %7.3f"
                  % (n, pr["resid"] / GIB, pr["operator"] / GIB,
                     pr["total"] / GIB, ms["measured"] / GIB, r))
        print("   (GiB; m/p = measured / predicted. Linear in n, so this ratio")
        print("    is the ratio at n = 32768 as well.)")
        out["n_sweep"] = rows

    print()
    print("=" * 78)
    print("B2. TERM SEPARATION -- s sweep at n=256, fp32, CUDA")
    print("=" * 78)
    if not cuda:
        print("NOT MEASURED: no CUDA device.")
    else:
        print("%6s %12s %12s %12s %7s" %
              ("s", "pred_total", "measured", "meas_B/(n*s*s)", "m/p"))
        srows = []
        for seq in (64, 128, 256, 512):
            pr = predicted(256, seq)
            ms = measure_peak(256, seq, "cuda")
            srows.append(dict(**pr, **ms,
                              ratio=ms["measured"] / pr["total"]))
            print("%6d %12.4f %12.4f %12.4f %7.3f"
                  % (seq, pr["total"] / GIB, ms["measured"] / GIB,
                     ms["measured"] / (256 * seq * seq), ms["measured"] / pr["total"]))
        # Back-solve C_RESIDUAL and C_OPERATOR from the two extreme s points:
        #   M(s) = CR * n * s * d * 4 + CO * n * s * s * 4 + CH * n * s * 4
        # Two unknowns, two equations, exact.
        a, b = srows[0], srows[-1]
        n = 256
        def coeffs(r):
            s = r["seq"]
            return (n * s * D_MODEL * 4.0, n * s * s * 4.0,
                    r["measured"] - sizing.C_HEAD * n * s * 1 * 4)
        a1, a2, a3 = coeffs(a)
        b1, b2, b3 = coeffs(b)
        det = a1 * b2 - a2 * b1
        cr = (a3 * b2 - a2 * b3) / det
        co = (a1 * b3 - a3 * b1) / det
        print()
        print("back-solved from s=%d and s=%d (exact, 2 eqns 2 unknowns):"
              % (a["seq"], b["seq"]))
        print("   C_RESIDUAL  module %5.2f   this shape %6.2f" % (sizing.C_RESIDUAL, cr))
        print("   C_OPERATOR  module %5.2f   this shape %6.2f" % (sizing.C_OPERATOR, co))
        out["s_sweep"] = srows
        out["backsolved"] = dict(C_RESIDUAL=cr, C_OPERATOR=co)

    print()
    print("=" * 78)
    print("C. WALL CLOCK -- CPU, threads=%d, median of %d timed steps" % (THREADS, TIMED))
    print("=" * 78)
    torch.set_num_threads(THREADS)
    ns = (2048, 4096, 8192, 16384)
    cpu_secs = []
    print("%8s %14s %16s" % ("n", "s/step", "s/150 steps"))
    for n in ns:
        t = time_steps(n, S, "cpu", TIMED)
        cpu_secs.append(t)
        print("%8d %14.4f %16.1f" % (n, t, t * 150))
    fit = ols_loglog(ns, cpu_secs)
    print()
    print("log-log OLS over %d points: slope %.4f  R^2 %.6f" % (len(ns), fit["slope"], fit["r2"]))
    print("endpoint slope n=%d..%d:      %.4f" % (ns[0], ns[-1], fit["endpoint_slope"]))
    print("inherited (V13_DAG_TASKLIST.md:88-91): endpoint 1.33813, 3-pt OLS 1.31700")
    print("log residuals: %s" % ", ".join("%+.4f" % r for r in fit["resid"]))
    out["cpu"] = dict(n=list(ns), secs_per_step=cpu_secs, **{k: v for k, v in fit.items()})

    print()
    print("=" * 78)
    print("D. WALL CLOCK -- CUDA, median of %d timed steps" % TIMED)
    print("=" * 78)
    if not cuda:
        print("NOT MEASURED: no CUDA device.")
    else:
        gns, gsecs = [], []
        print("%8s %14s %16s %14s" % ("n", "s/step", "s/150 steps", "cpu/gpu"))
        for n, c in zip(ns, cpu_secs):
            pr = predicted(n)["total"]
            free, _ = torch.cuda.mem_get_info()
            if pr * 1.4 > free:                 # predict first, then probe
                print("%8d  SKIPPED: predicted %.2f GiB x1.4 margin > %.2f GiB free"
                      % (n, pr / GIB, free / GIB))
                continue
            t = time_steps(n, S, "cuda", TIMED)
            gns.append(n)
            gsecs.append(t)
            print("%8d %14.4f %16.1f %14.1fx" % (n, t, t * 150, c / t))
        gfit = ols_loglog(gns, gsecs)
        print()
        print("log-log OLS over %d points: slope %.4f  R^2 %.6f"
              % (len(gns), gfit["slope"], gfit["r2"]))
        out["cuda"] = dict(n=gns, secs_per_step=gsecs,
                           **{k: v for k, v in gfit.items()})

    print()
    print("=" * 78)
    print("F. R2 AT FULL SIZE -- n=32768 MEASURED, not extrapolated")
    print("=" * 78)
    #: PREDICT FIRST, THEN PROBE. The brief forbids attempting an allocation
    #: already predicted not to fit, so the guard below is the model deciding
    #: whether the model gets tested -- and it is stated before the attempt.
    pr32 = predicted(32768)
    print("predicted activations at n=32768, s=64, fp32: %.3f GiB "
          "(residual %.3f + operator %.3f + head %.4f)"
          % (pr32["total"] / GIB, pr32["resid"] / GIB, pr32["operator"] / GIB,
             pr32["head"] / GIB))
    print("plus x_train 32768*64*16*4 = %.3f GiB resident"
          % (32768 * S * D_MODEL * 4 / GIB))
    need = pr32["total"] + 32768 * S * D_MODEL * 4
    print("working set the model says R2 needs: %.3f GiB" % (need / GIB))
    if cuda:
        free, _ = torch.cuda.mem_get_info()
        print("free VRAM %.3f GiB -> %s" % (free / GIB,
              "ATTEMPTING" if need < free else "REFUSING (predicted not to fit)"))
        if need < free:
            m32 = measure_peak(32768, S, "cuda")
            t32 = time_steps(32768, S, "cuda", 6)
            print("  CUDA measured peak %.3f GiB   measured/predicted %.3f"
                  % (m32["measured"] / GIB, m32["measured"] / pr32["total"]))
            print("  CUDA %.4f s/step  ->  150 steps x 8 seeds = %.3f h;"
                  "  9600-step ladder x 8 seeds = %.2f h"
                  % (t32, t32 * 150 * 8 / 3600, t32 * 9600 * 8 / 3600))
            out["r2_cuda"] = dict(peak=m32["measured"], secs_per_step=t32,
                                  predicted=pr32["total"])
    try:
        import psutil
        avail = psutil.virtual_memory().available
        print("host RAM available %.3f GiB -> %s" % (avail / GIB,
              "ATTEMPTING" if need * 1.2 < avail else "REFUSING (no 20%% margin)"))
        if need * 1.2 < avail:
            t32c = time_steps(32768, S, "cpu", 6)
            print("  CPU  %.4f s/step  ->  150 steps x 8 seeds = %.3f h;"
                  "  9600-step ladder x 8 seeds = %.2f h"
                  % (t32c, t32c * 150 * 8 / 3600, t32c * 9600 * 8 / 3600))
            out["r2_cpu"] = dict(secs_per_step=t32c)
    except ImportError:
        print("host RAM       NOT MEASURED (psutil absent)")

    print()
    print("=" * 78)
    print("G. LARGEST AFFORDABLE n ON THIS GPU")
    print("=" * 78)
    if cuda and "n_sweep" in out:
        #: bytes per training example, least squares through the origin over the
        #: MEASURED peaks. Through the origin because both model terms are exactly
        #: linear in n and an intercept would be fitting allocator granularity.
        pts = [(r["n"], r["measured"]) for r in out["n_sweep"] if r["n"] >= 512]
        if "r2_cuda" in out:
            pts.append((32768, out["r2_cuda"]["peak"]))
        alpha = sum(n * m for n, m in pts) / sum(n * n for n, _ in pts)
        per = alpha + S * D_MODEL * 4        # peak activations + resident x_train
        free, _ = torch.cuda.mem_get_info()
        print("measured bytes/example: activations %.0f + x_train %d = %.0f"
              % (alpha, S * D_MODEL * 4, per))
        print("eval-set reserve held back: 0.200 GiB (x_eval at n_eval=4096 plus "
              "its no-grad forward)")
        budget0 = free - 0.200 * GIB
        print("%10s %14s %16s" % ("margin", "budget GiB", "largest n"))
        for margin in (0.00, 0.10, 0.20, 0.30):
            b = budget0 * (1 - margin)
            print("%9.0f%% %14.3f %16d" % (margin * 100, b / GIB, int(b // per)))
        out["alpha_bytes_per_example"] = alpha
    else:
        print("NOT MEASURED: no CUDA device.")

    print()
    print("=" * 78)
    print("E. R1 / R2 PROJECTION from the fits above")
    print("=" * 78)
    for name, n, tstar in (("R1", 2048, 2), ("R2", 32768, 8)):
        pr = predicted(n)
        print("%s  BED-M t*=%d n=%d" % (name, tstar, n))
        print("     predicted activations  %.3f GiB "
              "(residual %.3f + operator %.3f), fp32" %
              (pr["total"] / GIB, pr["resid"] / GIB, pr["operator"] / GIB))
        for tag, f in (("cpu", fit), ("cuda", out.get("cuda"))):
            if f is None:
                continue
            sec = math.exp(f["intercept"] + f["slope"] * math.log(n))
            print("     %-4s %.4f s/step -> 150 steps x 8 seeds = %.2f h"
                  % (tag, sec, sec * 150 * 8 / 3600.0))
    print()
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("backsolved",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
