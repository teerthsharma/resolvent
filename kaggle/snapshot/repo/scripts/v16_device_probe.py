"""NEPTUNE's v16 device probe: diagnose R1's wall-clock gap, re-validate the
calibrated memory model against a COMPLEX arm, and price both devices.

D-DEVICE (`CEQ_V16_CONTRACT.md` PART V): "re-certification: 637 CPU-h vs 47
GPU-h is a 13x decision; NEPTUNE prices both with the calibrated model and the
round runs on the certified device only."

WHY THIS FILE EXISTS RATHER THAN A RE-READ OF `scripts/v15_neptune_sizing_probe.py`.
Three things happened after that probe was written and each of them invalidates
a number in it:

  1. `V15_R1.md` section 9 measured ARM PL at `91.63 s` per 150 steps at
     `n = 2048` and softmax at `71.32 s`, against that probe's re-measured
     `12.2 s` for the SAME operator -- a 5.8x gap it declared UNDIAGNOSED and
     forbade repricing from. Section B here is that diagnosis: the factors are
     varied ONE AT A TIME (threads, data source, arm, burst-vs-sustained,
     host contention, device) and the one that reproduces the gap is named.
  2. `ceq/arm_phase.py` did not exist when the memory model was last validated.
     Its operator is COMPLEX (`torch.complex64` at fp32 parameters, 8 B/elt =
     2x fp32), and `C_OPERATOR` for a complex arm is NOT MEASURED anywhere.
     Section C back-solves it from an `s` sweep, per arm.
  3. `scale/r10_capacity_sweep.py --device cuda` ABORTED by design
     (`V15_MERCURY_DEVICE.md` (e)) because `negation_scope.calibrate_bar` was
     structurally CPU-only. That is being lifted WHILE THIS RUNS by the node
     that owns those two files, so section E does not assert their state -- it
     READS it at run time and prints what it found, which is the only form of
     that claim that survives a concurrent edit.

L-LEAN / "nothing trains as a research reading". Every timing and allocation
below runs on `torch.randn` or on a corpus draw used ONLY for its shape and
values-as-bytes; no NRMSE is scored, no seed sweep is pooled, no verdict is
formed, no cell of R1', R2 or any bed is run. Section E calls `calibrate_bar`
for its WALL CLOCK on each device -- its five clause values are deliberately NOT
printed, because publishing them would be performing the re-certification that
D-DEVICE assigns to another node.
Step cost and peak bytes are functions of tensor SHAPE, not of tensor values.

PROVENANCE. Everything printed here is [MEASURED] on this box in this run
unless the line says otherwise. `V16_DEVICE_CERT.md` carries the tags.

ONE PROVENANCE CAVEAT, STATED HERE RATHER THAN DISCOVERED LATER. Section E
reads `scale/negation_scope.py` AS IT STANDS IN THE WORKING TREE, and that file
is `M` in `git status` while this probe runs: a concurrent node is editing it
this iteration. `calibrate_bar` gained a `device=` parameter between this file's
first and second runs, and `r10_capacity_sweep.py`'s abort was deleted between
its second and third. Section E therefore measures a moving target and prints
the state it read; nothing else in this probe depends on either file.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import textwrap
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import sizing                                             # noqa: E402
from ceq import arm_pl                                             # noqa: E402
from ceq.arm_phase import ArmPhase                                 # noqa: E402
from scale.m3_capability import Arm, D_MODEL, LR                   # noqa: E402
from scale import negation_scope as ns                             # noqa: E402

S, D_CHAIN, GIB = 64, 24, 1024 ** 3
WARMUP, TIMED = 2, 12
R1_THREADS = 8            # the pin V15_R1.md ran at
NEP_THREADS = 12          # the pin V15_NEPTUNE_SYSTEMS.md section 4.2 ran at

#: Bytes per element of the OPERATOR tensor, per arm. Read off the live tensor
#: in section C rather than assumed: ArmPhase's operator is complex64 when its
#: parameters are fp32, which is 2x fp32 and is the whole reason this node
#: re-validates the model instead of inheriting it.
ARMS = ("softmax", "arm_pl", "arm_phase")


def build(kind: str, seq: int = S):
    if kind == "arm_pl":
        return arm_pl.ArmPL(seq, d_model=D_MODEL)
    if kind == "arm_phase":
        return ArmPhase(seq, d_model=D_MODEL)
    return Arm(kind, seq)


def corpus(n: int, seq: int = S, t_star: int = 2, seed: int = 0, device=None):
    """R1's actual training tensors. Shape and bytes only -- never scored.

    `make_equilibrium_batch` draws on a CPU generator and only then moves
    (`negation_scope.py:96, 426-427`), so the corpus BYTES are identical on
    both devices by construction. That is what makes a device comparison of
    step cost a comparison of arithmetic and not of two different draws.
    """
    x, y, _, _ = ns.make_equilibrium_batch(n, seq, D_CHAIN, t_star=t_star,
                                           d_model=D_MODEL, seed=seed,
                                           device=device)
    sigma = float(y.std(unbiased=False)) or 1.0
    return x, (y - float(y.mean())) / sigma


def timed_steps(kind: str, n: int, seq: int, device: str, steps: int,
                data: str = "randn", threads: int = R1_THREADS) -> dict:
    """Both statistics, from ONE loop, so they cannot disagree about the run.

    `median` is `V15_NEPTUNE_SYSTEMS.md` section 4.2's statistic (median of the
    timed steps, warm-up discarded). `r1_stat` is `V15_R1.md`'s: total elapsed
    over the whole loop divided by the step count, warm-up INCLUDED, which is
    what `scripts/v15_r1.py:train_one` records as `secs`. Reporting both is the
    only way to tell a statistic difference from a machine difference.
    """
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = build(kind, seq).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    if data == "corpus":
        x, y = corpus(n, seq, device=device)
    else:
        x = torch.randn(n, seq, D_MODEL, device=device)
        y = torch.randn(n, device=device)
    per = []
    if device == "cuda":
        torch.cuda.synchronize()
    t_all = time.perf_counter()
    for i in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        per.append(time.perf_counter() - t0)
    total = time.perf_counter() - t_all
    del model, opt, x, y
    if device == "cuda":
        torch.cuda.empty_cache()
    body = per[WARMUP:] if len(per) > WARMUP else per
    return dict(kind=kind, n=n, device=device, data=data, threads=threads,
                steps=steps, median=float(statistics.median(body)),
                r1_stat=float(total / steps), total=float(total),
                first=float(per[0]), last=float(per[-1]))


# --------------------------------------------------------------------- memory

def predict(n: int, seq: int, c_resid: float, c_op: float,
            rb: float = 4.0, ob: float = 4.0) -> dict:
    """The module's two-term form, with the operator's bytes/element explicit.

    `sizing.activation_bytes` hard-codes `DTYPE_MODES` for the operator term and
    has no complex mode, so the term is written out here with `ob` supplied from
    the LIVE tensor's `element_size()`. The softmax row is asserted equal to
    `sizing.activation_bytes(arm="signed", dtype="fp32")` so the decomposition
    is the module's and not a second copy of it.
    """
    resid = c_resid * n * seq * D_MODEL * rb
    op = c_op * n * seq * seq * ob
    head = sizing.C_HEAD * n * seq * 1 * 4
    return dict(resid=resid, op=op, head=head, total=resid + op + head)


def module_predict(n: int, seq: int) -> float:
    cfg = sizing.Config(d_model=D_MODEL, n_layers=1, n_heads=1, d_head=D_MODEL,
                        vocab=1, seq=seq, d_ff_mult=8)
    return float(sizing.activation_bytes(cfg, batch=n, arm="signed",
                                         dtype="fp32"))


def measure_peak(kind: str, n: int, seq: int) -> dict:
    """Peak CUDA bytes for one forward + backward above the resident baseline,
    plus the operator tensor's dtype and bytes/element, read off the live graph.
    """
    torch.manual_seed(0)
    model = build(kind, seq).to("cuda")
    x = torch.randn(n, seq, D_MODEL, device="cuda")
    y = torch.randn(n, device="cuda")
    model(x[:1])                        # warm mask cache + cuBLAS workspace
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.nn.functional.mse_loss(model(x), y).backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    ob = operator_bytes_per_element(model, x[:1])
    del model, x, y
    torch.cuda.empty_cache()
    return dict(kind=kind, n=n, seq=seq, measured=float(peak), ob=ob)


def operator_bytes_per_element(model, x1) -> tuple:
    """(dtype name, bytes/element) of the [n,s,s] operator this arm forms."""
    with torch.no_grad():
        q, k = model.wq(x1), model.wk(x1)
        # Both gated arms expose `heads(x)`; the softmax arm has none. The head
        # tuple is spread into `_operator` exactly as each arm's own `forward`
        # does, so this reads the shipped operator and not a second copy.
        extra = model.heads(x1) if hasattr(model, "heads") else ()
        a = model._operator(q, k, *extra)
    return (str(a.dtype).replace("torch.", ""), a.element_size())


def backsolve(rows: dict, ob: float) -> dict:
    """C_RESIDUAL and C_OPERATOR from two `s` points at fixed `n`, exactly.

    Two equations, two unknowns, `C_HEAD` held at the module value. This is the
    only axis on which the two terms scale differently (`s` against `s^2`), so
    it is the only one that separates them.
    """
    (s1, m1), (s2, m2) = sorted(rows.items())
    n = m1["n"]
    h1 = sizing.C_HEAD * n * s1 * 4
    h2 = sizing.C_HEAD * n * s2 * 4
    a1, b1, y1 = n * s1 * D_MODEL * 4.0, n * s1 * s1 * ob, m1["measured"] - h1
    a2, b2, y2 = n * s2 * D_MODEL * 4.0, n * s2 * s2 * ob, m2["measured"] - h2
    det = a1 * b2 - a2 * b1
    return dict(C_RESIDUAL=(y1 * b2 - y2 * b1) / det,
                C_OPERATOR=(a1 * y2 - a2 * y1) / det)


# ------------------------------------------------------------------------ fits

def ols_loglog(xs, ys) -> dict:
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((yy - (a + b * xx)) ** 2 for xx, yy in zip(lx, ly))
    ss_tot = sum((yy - my) ** 2 for yy in ly)
    return dict(slope=b, intercept=a, r2=1.0 - ss_res / ss_tot,
                resid=[yy - (a + b * xx) for xx, yy in zip(lx, ly)])


def predict_step(fit: dict, n: int) -> float:
    return math.exp(fit["intercept"] + fit["slope"] * math.log(n))


# ------------------------------------------------------------------ contention

#: argv: seconds, hold_gib. A 2048x2048 fp32 matmul streams 48 MB of operands
#: per iteration, which does not fit this box's L3, so each spinner costs one
#: saturated core AND a share of DRAM bandwidth. The cell being timed is itself
#: bandwidth-bound on a materialised [n,s,s] operator (33 MB at n=2048), so
#: bandwidth is the contention axis that can actually reach it -- an L2-resident
#: spinner competes for cores only and understates the pressure.
SPINNER = textwrap.dedent("""
    import sys, time, torch
    torch.set_num_threads(1)
    hold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    ballast = None
    if hold > 0:
        ballast = torch.ones(int(hold * (1024 ** 3) / 4), dtype=torch.float32)
        ballast.add_(1.0)                      # touch it so it is resident
    a = torch.randn(2048, 2048); b = torch.randn(2048, 2048)
    end = time.time() + float(sys.argv[1])
    while time.time() < end:
        a = (a @ b).clamp_(-3, 3)
""").strip()


def with_contention(k: int, seconds: float = 30.0, hold_gib: float = 0.0) -> dict:
    """Time R1's cell with `k` other torch processes on the box.

    This is the one candidate for R1's gap that cannot be tested by varying an
    argument, because it is not an argument: the campaign dispatches planets and
    moons in parallel, and `V15_R1.md` does not record what else held the box
    while its sixteen cells ran.

    `hold_gib` is the SECOND contention axis. `V13_DAG_TASKLIST.md:105-110`
    records that Windows commits beyond available-physical, so the failure mode
    of a full host is not an exception but page-file traffic that silently
    inflates the wall clock. Held ballast is allocated in the spinner and
    touched, so it is resident and not merely committed.
    """
    procs = []
    for _ in range(k):
        procs.append(subprocess.Popen([sys.executable, "-c", SPINNER,
                                       str(seconds), str(hold_gib)],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
    try:
        if k:
            time.sleep(4.0)                      # let them reach steady state
        r = timed_steps("softmax", 2048, S, "cpu", WARMUP + TIMED,
                        threads=R1_THREADS)
        # The same cell on the GPU under the SAME host pressure. This is the
        # schedulability half of the device decision and it is separate from
        # the speed half: a cpu cell's wall clock is a function of what else
        # holds the box, and nothing else in this campaign touches the GPU.
        g = timed_steps("softmax", 2048, S, "cuda", WARMUP + TIMED,
                        threads=R1_THREADS)
    finally:
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()
    r["contenders"] = k
    r["hold_gib"] = hold_gib
    r["cuda_median"] = g["median"]
    return r


# =========================================================================== A

def section_a() -> dict:
    print("=" * 78)
    print("A. ENVIRONMENT")
    print("=" * 78)
    import psutil
    vm = psutil.virtual_memory()
    env = dict(torch=torch.__version__, cuda=torch.cuda.is_available(),
               cpu_count=os.cpu_count(), default_threads=torch.get_num_threads(),
               ram_total=vm.total, ram_avail=vm.available, ram_pct=vm.percent,
               cpu_busy=psutil.cpu_percent(interval=1.0),
               procs=len(psutil.pids()))
    print("torch %s   cuda_available=%s" % (env["torch"], env["cuda"]))
    if env["cuda"]:
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        env.update(gpu=p.name, sm="sm_%d%d" % (p.major, p.minor),
                   mps=p.multi_processor_count, vram_free=free, vram_total=total)
        print("gpu            %s  %s  MPs=%d" % (p.name, env["sm"], env["mps"]))
        print("vram total     %.3f GiB (%d B)" % (total / GIB, total))
        print("vram free      %.3f GiB (%d B)" % (free / GIB, free))
    print("host RAM       %.2f GiB total, %.2f GiB available (%.1f%% used)"
          % (vm.total / GIB, vm.available / GIB, vm.percent))
    print("cpu            %d logical; torch reads %d threads after imports "
          "(scale/m3_capability.py:64 pins 2 at import; every timing below "
          "sets its own pin)" % (env["cpu_count"], env["default_threads"]))
    print("box at rest    cpu %.1f%% busy, %d processes"
          % (env["cpu_busy"], env["procs"]))
    print("cell shape     s=%d d_model=%d layers=1 heads=1, FULL BATCH (batch==n)"
          % (S, D_MODEL))
    return env


# =========================================================================== B

def section_b() -> dict:
    out = {}
    print()
    print("=" * 78)
    print("B1. R1's GAP -- THREAD COUNT (n=2048, s=64, softmax, randn, cpu)")
    print("=" * 78)
    print("V15_R1.md ran threads=8; V15_NEPTUNE_SYSTEMS.md section 4.2 ran 12.")
    print("%8s %14s %14s %14s" % ("threads", "median s/step", "R1 stat", "x150 s"))
    rows = []
    for th in (2, 6, 8, 12, 20):
        r = timed_steps("softmax", 2048, S, "cpu", WARMUP + TIMED, threads=th)
        rows.append(r)
        print("%8d %14.4f %14.4f %14.1f"
              % (th, r["median"], r["r1_stat"], r["r1_stat"] * 150))
    out["threads"] = rows
    t8 = [r for r in rows if r["threads"] == 8][0]["median"]
    t12 = [r for r in rows if r["threads"] == 12][0]["median"]
    print("threads=8 / threads=12 = %.3f   (a 5.8x gap needs 5.8)" % (t8 / t12))

    print()
    print("=" * 78)
    print("B2. R1's GAP -- DATA SOURCE and ARM, at R1's own pin and step count")
    print("=" * 78)
    print("150 steps, threads=8, cpu. 'R1 stat' is total/steps, the statistic")
    print("scripts/v15_r1.py:train_one records as `secs`, warm-up included.")
    print("%-10s %-7s %11s %11s %11s   %s"
          % ("arm", "data", "median", "R1 stat", "x150 s", "V15_R1.md filed"))
    filed = {("softmax", "corpus"): 71.32, ("arm_pl", "corpus"): 91.63}
    rows = []
    for kind in ARMS:
        for data in ("randn", "corpus"):
            r = timed_steps(kind, 2048, S, "cpu", 150, data=data,
                            threads=R1_THREADS)
            rows.append(r)
            f = filed.get((kind, data))
            print("%-10s %-7s %11.4f %11.4f %11.1f   %s"
                  % (kind, data, r["median"], r["r1_stat"],
                     r["r1_stat"] * 150,
                     ("%.2f s -> reproduced at %.2fx"
                      % (f, r["r1_stat"] * 150 / f)) if f else "-"))
    out["factorial"] = rows

    print()
    print("=" * 78)
    print("B3. R1's GAP -- HOST CONTENTION (n=2048, softmax, randn, threads=8)")
    print("=" * 78)
    print("k = concurrent 1-thread torch processes on 2048x2048 matmuls, each")
    print("holding `hold` GiB of touched ballast. Two axes: cores+bandwidth,")
    print("and host memory. R1's gap needs 5.8x from somewhere.")
    print("The CUDA column is the SAME cell on the GPU under the SAME host")
    print("pressure: the schedulability half of the device decision.")
    print("%4s %6s %11s %10s %9s %11s %9s %8s"
          % ("k", "hold", "cpu s/step", "cpu x150", "vs k=0",
             "cuda s/step", "vs k=0", "RAM free"))
    rows = []
    base = gbase = None
    import psutil
    for k, hold in ((0, 0.0), (4, 0.0), (8, 0.0), (12, 0.0),
                    (4, 0.75), (8, 0.75)):
        r = with_contention(k, hold_gib=hold)
        base = base or r["median"]
        gbase = gbase or r["cuda_median"]
        r["ram"] = psutil.virtual_memory().available
        rows.append(r)
        print("%4d %6.2f %11.4f %10.1f %8.2fx %11.4f %8.2fx %7.2fG"
              % (k, hold, r["median"], r["median"] * 150, r["median"] / base,
                 r["cuda_median"], r["cuda_median"] / gbase, r["ram"] / GIB))
    out["contention"] = rows
    print("cpu spread over the sweep %.2fx;  cuda spread %.2fx"
          % (max(x["median"] for x in rows) / min(x["median"] for x in rows),
             max(x["cuda_median"] for x in rows)
             / min(x["cuda_median"] for x in rows)))

    print()
    print("=" * 78)
    print("B5. R1's GAP -- WHICH CORES THE 8 THREADS LANDED ON")
    print("=" * 78)
    print("This box is an Intel i7-14700HX: 20 physical cores, 28 logical, which")
    print("is 8 P-cores (SMT, logical 0-15) plus 12 E-cores (logical 16-27).")
    print("`torch.set_num_threads(8)` names a COUNT, never a placement, so the")
    print("Windows scheduler decides -- and it demotes work it reads as")
    print("background. A cell whose 8 threads land on E-cores is a different")
    print("machine from one whose 8 land on P-cores, at identical settings.")
    print("Crossed with B3's contention axis, because neither factor alone")
    print("reaches R1's number and the two do not compose multiplicatively.")
    import psutil
    proc = psutil.Process()
    original = proc.cpu_affinity()
    P_ALL, E_ALL = list(range(16)), list(range(16, 28))
    lanes = [("P-cores, SMT siblings", P_ALL, 0),
             ("P-cores, 1 thread/core", [0, 2, 4, 6, 8, 10, 12, 14], 0),
             ("default (no pin)", None, 0),
             ("E-cores only", E_ALL, 0),
             ("default (no pin)", None, 12),
             ("E-cores only", E_ALL, 12)]
    print("%-24s %4s %13s %10s %10s"
          % ("lane", "k", "median s/step", "x150 s", "vs P-core"))
    rows, pbase = [], None
    for label, aff, k in lanes:
        procs = [subprocess.Popen([sys.executable, "-c", SPINNER, "45", "0"],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
                 for _ in range(k)]
        try:
            if k:
                time.sleep(4.0)
            proc.cpu_affinity(original if aff is None else aff)
            r = timed_steps("softmax", 2048, S, "cpu", WARMUP + TIMED,
                            threads=R1_THREADS)
        finally:
            proc.cpu_affinity(original)
            for q in procs:
                q.kill()
            for q in procs:
                q.wait()
        pbase = pbase or r["median"]
        rows.append((label, k, r["median"]))
        print("%-24s %4d %13.4f %10.1f %9.2fx"
              % (label, k, r["median"], r["median"] * 150,
                 r["median"] / pbase))
    out["cores"] = rows
    worst = max(v for _, _, v in rows)
    print()
    print("worst lane / P-core lane: %.2fx.  V15_R1.md filed 71.32 s against"
          % (worst / pbase))
    print("this P-core lane's %.1f s, a factor of %.2f. Neither placement nor"
          % (pbase * 150, 71.32 / (pbase * 150)))
    print("contention reaches that alone; the CROSS does most of it, and the")
    print("cross is super-multiplicative -- see V16_DEVICE_CERT.md section 2.")

    print()
    print("=" * 78)
    print("B4. THE SAME SHAPE ON BOTH DEVICES (n=2048, s=64, 150 steps)")
    print("=" * 78)
    print("CPU column is B2's randn row, not a second run of the same thing.")
    print("%-10s %11s %11s %11s %11s %9s"
          % ("arm", "cpu med", "cpu x150", "cuda med", "cuda x150", "cpu/gpu"))
    cpu_med = {r["kind"]: r["median"] for r in out["factorial"]
               if r["data"] == "randn"}
    devices = {}
    for kind in ARMS:
        g = timed_steps(kind, 2048, S, "cuda", 150, threads=R1_THREADS)
        c = cpu_med[kind]
        devices[kind] = (c, g["median"])
        print("%-10s %11.4f %11.1f %11.4f %11.1f %8.1fx"
              % (kind, c, c * 150, g["median"], g["median"] * 150,
                 c / g["median"]))
    out["devices"] = devices
    return out


# =========================================================================== C

def section_c() -> dict:
    out = {}
    print()
    print("=" * 78)
    print("C1. MEMORY MODEL vs ALLOCATOR -- n sweep at s=64, CUDA, ALL THREE ARMS")
    print("=" * 78)
    print("`pred` is `sizing.activation_bytes(arm='signed', dtype='fp32')`, the")
    print("module's own number, which knows nothing about a complex operator.")
    print("`B/example` is the model-free quantity: both terms are LINEAR in n at")
    print("fixed s, so B/example measured at one n IS B/example at every n, and")
    print("that -- not the model -- is what the R2 extrapolation rests on.")
    print("%-10s %8s %12s %12s %8s %14s"
          % ("arm", "n", "pred GiB", "meas GiB", "m/p", "meas B/example"))
    per_ex = {}
    rows = []
    for kind in ARMS:
        for n in (512, 2048, 8192):
            p = predict(n, S, sizing.C_RESIDUAL, sizing.C_OPERATOR)
            mod = module_predict(n, S)
            assert abs(mod - p["total"]) < 1.0, (mod, p["total"])
            m = measure_peak(kind, n, S)
            rows.append((kind, n, p, m))
            print("%-10s %8d %12.3f %12.3f %8.3f %14.0f"
                  % (kind, n, p["total"] / GIB, m["measured"] / GIB,
                     m["measured"] / p["total"], m["measured"] / n))
            per_ex.setdefault(kind, []).append(m["measured"] / n)
    print()
    for kind in ARMS:
        v = per_ex[kind]
        print("  %-10s B/example over n=512..8192: %.0f .. %.0f  (spread %.2f%%)"
              % (kind, min(v), max(v), 100 * (max(v) - min(v)) / min(v)))
    out["nsweep"] = rows
    out["per_example"] = {k: max(v) for k, v in per_ex.items()}   # pessimistic

    print()
    print("=" * 78)
    print("C2. TERM SEPARATION PER ARM -- s sweep at n=256, CUDA")
    print("=" * 78)
    print("C_OPERATOR is back-solved per arm from s=64 and s=512. The operator's")
    print("bytes/element is READ OFF THE LIVE TENSOR, not assumed: ArmPhase is")
    print("complex64 at fp32 parameters, which is 2x fp32.")
    print("%-10s %-10s %6s %12s %12s %12s %12s"
          % ("arm", "op dtype", "B/elt", "C_RESIDUAL", "C_OPERATOR",
             "meas s=64", "meas s=512"))
    consts = {}
    for kind in ARMS:
        rows = {}
        for seq in (64, 512):
            rows[seq] = measure_peak(kind, 256, seq)
        ob = float(rows[64]["ob"][1])
        bs = backsolve(rows, ob)
        consts[kind] = dict(ob=ob, dtype=rows[64]["ob"][0], **bs)
        print("%-10s %-10s %6.0f %12.2f %12.3f %12.4f %12.4f"
              % (kind, rows[64]["ob"][0], ob, bs["C_RESIDUAL"], bs["C_OPERATOR"],
                 rows[64]["measured"] / GIB, rows[512]["measured"] / GIB))
    print("module values for comparison:  C_RESIDUAL %.2f   C_OPERATOR %.2f"
          % (sizing.C_RESIDUAL, sizing.C_OPERATOR))
    out["consts"] = consts

    print()
    print("=" * 78)
    print("C3. PREDICT-THEN-PROBE AT R2's SIZE (n=32768, s=64) -- PER ARM")
    print("=" * 78)
    print("Predicted with each arm's OWN back-solved constants from C2, against")
    print("free VRAM now. Nothing is attempted that the prediction refuses.")
    free, _ = torch.cuda.mem_get_info()
    xb = S * D_MODEL * 4                       # x_train bytes per example
    print("free VRAM %.3f GiB (total %.3f);  x_train adds %d B/example"
          % (free / GIB, torch.cuda.mem_get_info()[1] / GIB, xb))
    print("%-10s %12s %12s %12s %10s %12s"
          % ("arm", "pred act", "pred wset", "measured", "m/p", "verdict"))
    rows = []
    for kind in ARMS:
        c = consts[kind]
        p = predict(32768, S, c["C_RESIDUAL"], c["C_OPERATOR"], ob=c["ob"])
        wset = p["total"] + 32768 * xb
        free, _ = torch.cuda.mem_get_info()
        if wset > 0.80 * free:
            print("%-10s %12.3f %12.3f %12s %10s %12s"
                  % (kind, p["total"] / GIB, wset / GIB, "-", "-",
                     "REFUSED"))
            rows.append(dict(kind=kind, pred=p["total"], wset=wset,
                             measured=None, fits=False))
            continue
        m = measure_peak(kind, 32768, S)
        rows.append(dict(kind=kind, pred=p["total"], wset=wset,
                         measured=m["measured"], fits=True))
        print("%-10s %12.3f %12.3f %12.3f %10.3f %12s"
              % (kind, p["total"] / GIB, wset / GIB, m["measured"] / GIB,
                 m["measured"] / p["total"], "FITS"))
    out["r2"] = rows

    print()
    print("=" * 78)
    print("C4. LARGEST AFFORDABLE n PER ARM ON THIS GPU")
    print("=" * 78)
    print("From C1's measured B/example (the pessimistic end of its own spread)")
    print("plus x_train, against free VRAM less a 0.200 GiB eval-set reserve")
    print("(x_eval at n_eval=4096 and its no-grad forward).")
    free, _ = torch.cuda.mem_get_info()
    budget0 = free - 0.200 * GIB
    print("%-10s %14s %12s %12s %12s %12s"
          % ("arm", "B/example", "n @ 0%", "n @ 10%", "n @ 20%", "last 2^k"))
    caps = {}
    for kind in ARMS:
        be = out["per_example"][kind] + xb
        ns_ = [int(budget0 * (1 - m) / be) for m in (0.0, 0.10, 0.20)]
        pow2 = 2 ** int(math.floor(math.log2(ns_[2])))
        caps[kind] = dict(bytes_per_example=be, n0=ns_[0], n10=ns_[1],
                          n20=ns_[2], pow2=pow2)
        print("%-10s %14.0f %12d %12d %12d %12d"
              % (kind, be, ns_[0], ns_[1], ns_[2], pow2))
    out["caps"] = caps
    return out


# =========================================================================== D

def section_d() -> dict:
    out = {}
    for device, ns_ in (("cpu", (2048, 4096, 8192, 16384, 32768)),
                        ("cuda", (2048, 4096, 8192, 16384, 32768))):
        print()
        print("=" * 78)
        print("D%s. WALL CLOCK -- %s, softmax, threads=%d, median of timed steps"
              % ("1" if device == "cpu" else "2", device.upper(), R1_THREADS))
        print("=" * 78)
        print("%8s %14s %14s" % ("n", "s/step", "s/150 steps"))
        xs, ys = [], []
        for n in ns_:
            steps = WARMUP + (6 if n >= 16384 else TIMED)
            r = timed_steps("softmax", n, S, device, steps, threads=R1_THREADS)
            xs.append(n)
            ys.append(r["median"])
            print("%8d %14.4f %14.1f" % (n, r["median"], r["median"] * 150))
        fit = ols_loglog(xs, ys)
        print("log-log OLS over %d points: slope %.4f  R^2 %.6f"
              % (len(xs), fit["slope"], fit["r2"]))
        print("log residuals: %s"
              % ", ".join("%+.4f" % v for v in fit["resid"]))
        out[device] = dict(n=xs, s=ys, fit=fit)

    print()
    print("=" * 78)
    print("D3. ARM MULTIPLIER on the certified device (n=2048, CUDA)")
    print("=" * 78)
    print("The wall-clock law above is fitted on softmax. Every other arm is")
    print("priced as softmax x this multiplier, measured at one shape.")
    base = timed_steps("softmax", 2048, S, "cuda", WARMUP + TIMED)["median"]
    mult = {}
    for kind in ARMS:
        r = timed_steps(kind, 2048, S, "cuda", WARMUP + TIMED)
        mult[kind] = r["median"] / base
        print("  %-10s %.5f s/step   x%.3f" % (kind, r["median"], mult[kind]))
    out["mult"] = mult
    return out


# =========================================================================== E

def section_e() -> dict:
    print()
    print("=" * 78)
    print("E. WHAT CUDA CERTIFICATION ACTUALLY REQUIRES")
    print("=" * 78)
    sig = inspect.signature(ns.calibrate_bar)
    takes_device = "device" in sig.parameters
    print("negation_scope.calibrate_bar signature, AS IT STANDS IN THE WORKING")
    print("TREE (see the provenance note below -- this file is `M` in `git")
    print("status` and is owned by another node this iteration):")
    print("   (%s)" % ", ".join(str(p) for p in sig.parameters.values()))
    print("   accepts a `device` argument: %s" % takes_device)
    body = inspect.getsource(ns.calibrate_bar).split('"""')[-1]
    blind = [ln.strip() for ln in body.splitlines()
             if any(t in ln for t in ("batch_fn or make_batch",
                                      "torch.Generator(",
                                      "torch.nn.Sequential(",
                                      "net = net.to("))
             and not ln.strip().startswith("#")]
    print("   the constructions that decide the device, in its body:")
    for ln in blind:
        print("      %s" % ln)
    # Read the CALLER's state instead of asserting it: this file is also being
    # edited by the concurrent node, and a hardcoded claim about it goes stale
    # between two runs of this probe -- it did.
    sweep = (pathlib.Path(__file__).resolve().parents[1]
             / "scale" / "r10_capacity_sweep.py").read_text(encoding="utf-8")
    print("   caller state, read off scale/r10_capacity_sweep.py just now:")
    print("      passes device= into calibrate_bar : %s"
          % ("device=device" in sweep.split("cal = calibrate_bar")[-1][:400]))
    print("      still aborts on --device cuda     : %s"
          % ("ABORT: --device cuda" in sweep))
    print("      journals `device` on every record : %s"
          % ("device=a.device" in sweep))
    print("      refuse_cross_device_pool present  : %s"
          % ("def refuse_cross_device_pool" in sweep))

    print()
    print("   cost of ONE calibrate_bar call at the SHIPPED settings")
    print("   (n=N_EVAL=4096, s=64, d=24, steps=600, M3_TASKS['e3_t2'] hooks):")
    batch_fn, oracle_fn, feature_fn, _fd = ns.M3_TASKS["e3_t2"]
    torch.set_num_threads(R1_THREADS)
    cal = {}
    for device in (None, "cuda"):
        t0 = time.perf_counter()
        err = None
        try:
            ns.calibrate_bar(n=4096, s=S, d=D_CHAIN, steps=600, lr=LR,
                             batch_fn=batch_fn, oracle_fn=oracle_fn,
                             feature_fn=feature_fn,
                             **({} if device is None else {"device": device}))
        except Exception as exc:                              # noqa: BLE001
            err = "%s: %s" % (type(exc).__name__, exc)
        if device == "cuda":
            torch.cuda.synchronize()
        cal[device or "cpu"] = dict(secs=time.perf_counter() - t0, err=err)
        print("      device=%-5s %8.3f s   %s"
              % (device or "None", cal[device or "cpu"]["secs"],
                 err or "RAN"))
    print("      Clause VALUES are deliberately NOT printed: reading them is the")
    print("      re-certification D-DEVICE assigns to another node, and this")
    print("      node prices work rather than performing it.")

    print()
    print("   0-step RED gate cost, 16 untrained seeds, both devices:")
    red = {}
    for device in ("cpu", "cuda"):
        torch.set_num_threads(R1_THREADS)
        t0 = time.perf_counter()
        for seed in range(16):
            torch.manual_seed(seed)
            m = build("softmax", S).to(device)
            xt = torch.randn(2048, S, D_MODEL, device=device)
            xe = torch.randn(4096, S, D_MODEL, device=device)
            with torch.no_grad():
                m.eval()
                m(xt)
                m(xe)
            del m, xt, xe
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        red[device] = time.perf_counter() - t0
        print("      %-5s %.2f s for 16 seeds" % (device, red[device]))
    print()
    print("   TOTAL COMPUTE TO RE-CERTIFY ON DEVICE: %.2f s"
          % (cal["cuda"]["secs"] + red["cuda"]))
    print("   (one calibrate_bar on cuda + 16 untrained cuda seeds for GATE_TOL)")
    return dict(calibrate=cal, red=red, takes_device=takes_device)


# =========================================================================== F

#: The v16 cell census. A CELL is one (arm, bed, t*, n, seed) training run.
#: `N = 8` is contract-fixed (PART IV). Arms-per-node is NOT, so every entry
#: carries the assumption that produced its arm count; those assumptions are
#: restated and argued in V16_DEVICE_CERT.md section 6.
#: `P` = phase-gate arm (complex operator), `S` = a real-valued softmax-class
#: arm. Every arm count is an ASSUMPTION about a node the contract specifies by
#: its reading and not by its arm list; V16_DEVICE_CERT.md section 6 argues each
#: one. `arm_phase` is used to price ANY arm whose operator may be complex --
#: including R3's composed arm, whose dtype is NOT MEASURED -- because that is
#: the pessimistic end and the R11 bias runs the other way.
P, Sx = "arm_phase", "softmax"
PLAN = [
    # (iteration, label, [arms], n)
    ("7",  "R1' BED-M t*=2",          [P, Sx],            2048),
    ("10", "R2 BED-M t*=8",           [P, Sx],            32768),
    ("10", "R-SKY @ R1'",             [Sx],               2048),
    ("10", "R-SKY @ R2",              [Sx],               32768),
    ("12", "MARS param-match control", [Sx],              2048),
    ("17", "R3(a) delay bed",         [Sx, Sx, Sx, P],    32768),
    ("18", "R3(b) power-law bed",     [Sx, Sx, Sx, P],    32768),
    ("20", "R4 skylines, both beds",  [Sx, Sx],           32768),
    ("25", "R5 BED-1",                [P, Sx],            32768),
    ("28", "R6 hidden-cause",         [Sx, Sx, Sx, Sx],   2048),
    ("29", "R7 zeta rung n=512",      [P, Sx],            512),
    ("29", "R7 zeta rung n=1024",     [P, Sx],            1024),
    ("29", "R7 zeta rung n=2048",     [P, Sx],            2048),
]
SEEDS = 8
LADDERS = (150, 9600)


def section_f(fits: dict, mult: dict, caps: dict,
              cont_cpu: float, cont_cuda: float) -> dict:
    print()
    print("=" * 78)
    print("F. THE 40-ITERATION PLAN, PRICED PER ARM ON BOTH DEVICES")
    print("=" * 78)
    print("Fitted laws: cpu  s/step = exp(%.4f) * n^%.4f  (R^2 %.6f)"
          % (fits["cpu"]["fit"]["intercept"], fits["cpu"]["fit"]["slope"],
             fits["cpu"]["fit"]["r2"]))
    print("             cuda s/step = exp(%.4f) * n^%.4f  (R^2 %.6f)"
          % (fits["cuda"]["fit"]["intercept"], fits["cuda"]["fit"]["slope"],
             fits["cuda"]["fit"]["r2"]))
    print("Per-arm multipliers on the softmax law: %s"
          % ", ".join("%s x%.3f" % (k, v) for k, v in mult.items()))
    print("`fit?` is C4's cap at a 20%% margin: n_max %s"
          % ", ".join("%s %d" % (k, v["n20"]) for k, v in caps.items()))
    print()
    totals = {}
    for ladder in LADDERS:
        print("--- %d-step budget ---" % ladder)
        print("%-5s %-26s %6s %8s %10s %10s  %s"
              % ("it", "node", "cells", "n", "cpu h", "cuda h", "fit?"))
        tc = tg = 0.0
        cells_total = 0
        misfits = []
        for it, label, arms, n in PLAN:
            hc = hg = 0.0
            bad_arms = []
            for a in arms:
                cells_total += SEEDS
                hc += SEEDS * ladder * predict_step(fits["cpu"]["fit"], n) \
                    * mult[a] / 3600
                hg += SEEDS * ladder * predict_step(fits["cuda"]["fit"], n) \
                    * mult[a] / 3600
                if n > caps[a]["n20"]:
                    bad_arms.append(a)
            tc += hc
            tg += hg
            if bad_arms:
                misfits.append((it, label, n, sorted(set(bad_arms))))
            print("%-5s %-26s %6d %8d %10.2f %10.3f  %s"
                  % (it, label, len(arms) * SEEDS, n, hc, hg,
                     "OK" if not bad_arms
                     else "NO FIT: " + ",".join(sorted(set(bad_arms)))))
        print("%-5s %-26s %6d %8s %10.2f %10.3f"
              % ("", "TOTAL", cells_total, "", tc, tg))
        # Each device gets ITS OWN measured spread from B3. Applying the cpu
        # factor to the cuda column would manufacture a pessimism the GPU lane
        # was measured not to have, which is the R11 bias pointed backwards.
        print("%-5s %-26s %6s %8s %10.2f %10.3f"
              % ("", "pessimistic (cpu x%.2f, cuda x%.2f)"
                 % (cont_cpu, cont_cuda), "", "",
                 tc * cont_cpu, tg * cont_cuda))
        print()
        totals[ladder] = dict(cpu=tc, cuda=tg, cells=cells_total,
                              cpu_pess=tc * cont_cpu,
                              cuda_pess=tg * cont_cuda,
                              misfits=[(i, l, n, a) for i, l, n, a in misfits])
    if totals[LADDERS[0]]["misfits"]:
        print("CELLS THAT DO NOT FIT, at any step budget (memory is not a")
        print("function of the step count):")
        for it, label, n, arms in totals[LADDERS[0]]["misfits"]:
            for a in arms:
                print("  it.%-4s %-26s n=%d with %s -- cap is n=%d at 20%%,"
                      " %d at 0%%; last power of two %d"
                      % (it, label, n, a, caps[a]["n20"], caps[a]["n0"],
                         caps[a]["pow2"]))
    return totals


def section_g() -> dict:
    """The price of the determinism the device lane is required to declare.

    `V15_MERCURY_DEVICE.md` (b.3) measured `0.921x` -- no slowdown -- and the
    `--device cuda` lane sets the flag unconditionally. L-TIME requires the
    iteration that CITES a timing figure to re-measure it, and this certificate
    cites it, so it is re-measured here rather than carried.

    Runs LAST and restores the flag: `use_deterministic_algorithms` is global
    and would otherwise change every measurement above it.
    """
    print()
    print("=" * 78)
    print("G. THE PRICE OF DETERMINISM ON DEVICE (n=2048, s=64, CUDA)")
    print("=" * 78)
    print("Run in SUBPROCESSES: `CUBLAS_WORKSPACE_CONFIG` must be set before")
    print("the CUDA context exists, so an in-process toggle raises instead of")
    print("measuring -- which is what the first version of this section did.")
    print("Per arm, not per batch, because one arm blocking must not hide the")
    print("others -- and one of them does.")
    src = textwrap.dedent("""
        import json, statistics, sys, time, torch
        sys.path.insert(0, sys.argv[3])
        from ceq import arm_pl
        from ceq.arm_phase import ArmPhase
        from scale.m3_capability import Arm, D_MODEL, LR
        mode, kind = sys.argv[1], sys.argv[2]
        if mode == "strict":
            torch.use_deterministic_algorithms(True)
        elif mode == "warn":
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = mode != "off"
        def make():
            torch.manual_seed(0)
            return (arm_pl.ArmPL(64, d_model=D_MODEL) if kind == "arm_pl" else
                    ArmPhase(64, d_model=D_MODEL) if kind == "arm_phase" else
                    Arm(kind, 64)).to("cuda")
        try:
            m = make()
            o = torch.optim.Adam(m.parameters(), lr=LR)
            torch.manual_seed(1)
            x = torch.randn(2048, 64, D_MODEL, device="cuda")
            y = torch.randn(2048, device="cuda")
            per = []
            for i in range(14):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                o.zero_grad()
                torch.nn.functional.mse_loss(m(x), y).backward()
                o.step()
                torch.cuda.synchronize(); per.append(time.perf_counter() - t0)
            # Run-to-run REPRODUCIBILITY at the same seed, same process: the
            # quantity condition 3 exists to protect. Fresh model each repeat,
            # identical inputs, compare the gradient that Adam would consume.
            grads = []
            for _ in range(4):
                mm = make()
                torch.nn.functional.mse_loss(mm(x), y).backward()
                grads.append(torch.cat([p.grad.reshape(-1)
                                        for p in mm.parameters()]).clone())
            spread = max(float((g - grads[0]).abs().max()) for g in grads[1:])
            print(json.dumps(dict(secs=statistics.median(per[2:]),
                                  grad_spread=spread)))
        except Exception as exc:
            print(json.dumps(dict(error="%s: %s"
                                  % (type(exc).__name__,
                                     str(exc).split(".")[0]))))
    """).strip()
    root = str(pathlib.Path(__file__).resolve().parents[1])
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    out = {}
    for mode in ("off", "strict", "warn"):
        out[mode] = {}
        for kind in ARMS:
            r = subprocess.run([sys.executable, "-c", src, mode, kind, root],
                               capture_output=True, text=True, env=env)
            try:
                out[mode][kind] = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:                                 # noqa: BLE001
                out[mode][kind] = {"error": (r.stderr.strip().splitlines() or
                                             ["no output"])[-1]}
    print()
    print("%-10s %-24s %-24s %s"
          % ("arm", "det OFF s/step", "det STRICT", "det WARN_ONLY s/step"))
    for kind in ARMS:
        cells = []
        for mode in ("off", "strict", "warn"):
            v = out[mode][kind]
            cells.append("%.5f" % v["secs"] if "secs" in v
                         else "BLOCKED: " + v["error"][:14])
        print("%-10s %-24s %-24s %s" % (kind, cells[0], cells[1], cells[2]))
    print()
    print("Ratio warn_only/off, the price of the DECLARATION condition 3 asks")
    print("for: %s" % ", ".join(
        "%s %.3fx" % (k, out["warn"][k]["secs"] / out["off"][k]["secs"])
        for k in ARMS
        if "secs" in out["warn"].get(k, {}) and "secs" in out["off"].get(k, {})))
    print("V15_MERCURY_DEVICE.md (b.3) filed 0.921x, measured on the SOFTMAX")
    print("arm only -- the one arm in this table that strict mode does not")
    print("block. L-TIME requires the citing iteration to re-measure; done.")
    print()
    print("RUN-TO-RUN GRADIENT SPREAD on CUDA at a fixed seed, 4 repeats,")
    print("max |grad - grad_0| over all %d parameters. This is the quantity"
          % 4820)
    print("condition 3 protects: kernel noise must not enter the 8-seed sd.")
    print("M-10's thread-count floor for comparison: 2.345e-03 on eval_nrmse.")
    for kind in ARMS:
        v = out["warn"].get(kind, {})
        if "grad_spread" in v:
            print("   %-10s warn_only: %.3e" % (kind, v["grad_spread"]))
    return out


def section_h() -> dict:
    """Can the round's deciding arm take a gradient step at all?

    NOT a capability question and not a reading: no NRMSE, no seed sweep, no
    verdict, random data throughout. The only quantity is whether the gradient
    is FINITE, which is a precondition of every hour priced in section F. A
    schedule for an arm that cannot complete 150 steps is not a schedule, and
    finding that out here costs a minute against it.7's whole budget.

    WHY IT CAN FAIL. `ceq/arm_phase.py:121,128` take `log(m)` with
    `m = clamp(u, 0, 1)` (`magnitude`). The delta's word is CLOSED -- `0` and
    `1` are attainable VALUES, not limits -- so `m = 0` is reachable BY DESIGN,
    and `log(0) = -inf` with a `1/0` backward. `identity_heads()` starts at
    `m = 1`; the question is whether training stays away from the endpoint.
    """
    print()
    print("=" * 78)
    print("H. CAN THE PHASE ARM TAKE A GRADIENT STEP? (random data, finiteness")
    print("   only -- no NRMSE, no seed sweep, no verdict)")
    print("=" * 78)
    out = {}
    STEPS = 40
    for kind in ARMS:
        for init in ("as-constructed", "identity_heads()"):
            if init.startswith("identity") and kind == "softmax":
                continue
            for device in ("cpu", "cuda"):
                torch.set_num_threads(R1_THREADS)
                torch.manual_seed(0)
                model = build(kind, S)
                if init.startswith("identity"):
                    if hasattr(model, "identity_heads"):
                        model.identity_heads()
                    elif hasattr(model, "zero_heads"):
                        model.zero_heads()
                    else:
                        continue
                model = model.to(device)
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                x = torch.randn(512, S, D_MODEL, device=device)
                y = torch.randn(512, device=device)
                first = None
                zero_m = 0.0
                for i in range(STEPS):
                    opt.zero_grad()
                    torch.nn.functional.mse_loss(model(x), y).backward()
                    g = torch.cat([q.grad.reshape(-1)
                                   for q in model.parameters()])
                    if first is None and not torch.isfinite(g).all():
                        first = i
                        if hasattr(model, "heads"):
                            hs = model.heads(x)
                            from ceq.arm_phase import magnitude as _mag
                            if kind == "arm_phase":
                                zero_m = float((_mag(hs[0]) == 0).float().mean())
                    opt.step()
                out[(kind, init, device)] = first
                print("   %-10s %-18s %-5s first non-finite gradient: %s%s"
                      % (kind, init, device,
                         "step %d of %d" % (first, STEPS) if first is not None
                         else "none in %d steps" % STEPS,
                         ("   (%.4f%% of positions at m=0)" % (100 * zero_m))
                         if zero_m else ""))
                del model, opt, x, y
                if device == "cuda":
                    torch.cuda.empty_cache()
    print()
    print("`m = clamp(u, 0, 1)` and `log(m)` (ceq/arm_phase.py:121,128): the")
    print("CLOSED lower endpoint the X36 delta chose on purpose is a `-inf` in")
    print("the forward and a `1/0` in the backward. Device-independent -- the")
    print("STEP it fires at differs by device, the fact that it fires does not.")
    return {"%s|%s|%s" % k: v for k, v in out.items()}


def section_i() -> dict:
    """The determinism boundary AT THE ARMS' OWN SHAPES, and the TF32 state.

    `V16_R1_DEVICE_READY.md` measured `torch.cumsum` bitwise-reproducible at
    `n = 64` and `n = 10 000` and drifting `6.82e-13` at `n = 1 000 000`, and
    called that "evidence, not a licence". It is evidence about a ONE-DIMENSIONAL
    tensor, and the arms do not use one: `arm_phase.scan_phase` and
    `arm_pl.scan` call `cumsum(dim=-1)` on `[n, s]` with `s = 64` and `n` up to
    32,768 -- a REDUCTION LENGTH of 64 and a LAUNCH of 2.1 million elements.
    Whether the hazard follows the reduction length or the element count is the
    whole question, it decides whether the narrow claim covers R2, and nobody
    has measured it. This section does, at R2's exact shape.

    It also prints the TF32 state, because `V16_R1_DEVICE_READY.md` measured
    `allow_tf32=True` moving the shipped complex64 product by `2.958e-04`
    relative with no dtype in any record changing. A certificate that does not
    assert the flag's value has not certified the arithmetic.
    """
    print()
    print("=" * 78)
    print("I. THE DETERMINISM BOUNDARY AT THE ARMS' SHAPES, AND TF32")
    print("=" * 78)
    out = {}
    print("TF32 state on this box, read live:")
    for name, val in (("torch.backends.cuda.matmul.allow_tf32",
                       torch.backends.cuda.matmul.allow_tf32),
                      ("torch.backends.cudnn.allow_tf32",
                       torch.backends.cudnn.allow_tf32),
                      ("torch.get_float32_matmul_precision()",
                       torch.get_float32_matmul_precision())):
        print("   %-42s %s" % (name, val))
        out[name] = val

    print()
    print("TF32's effect on the SHIPPED arm (not a bare matmul), at")
    print("`identity_heads()`. The identity setting is used BECAUSE the")
    print("as-constructed arm's operator already contains non-finite values --")
    print("section H's `log(clamp(u,0,1))` at m=0 -- so a TF32 delta measured")
    print("there reads `nan` and says nothing about TF32. At identity, m=1 and")
    print("log(1)=0, the operator is finite, and the float32 q@k inside the")
    print("logits is still exactly where TF32 would bite.")
    torch.manual_seed(0)
    m = ArmPhase(S, d_model=D_MODEL).identity_heads().to("cuda")
    x = torch.randn(256, S, D_MODEL, device="cuda")
    refs = {}
    for flag in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = flag
        with torch.no_grad():
            a = m._operator(m.wq(x), m.wk(x), *m.heads(x))
            o = m(x)
        for label, v in (("operator", a), ("forward output", o)):
            if not flag:
                refs[label] = v.clone()
                if label == "operator":
                    print("   allow_tf32=False  reference; operator dtype %s, "
                          "finite=%s" % (v.dtype, bool(torch.isfinite(v).all())))
            else:
                r = refs[label]
                den = float(r.abs().max())
                rel = float((v - r).abs().max()) / den if den else float("nan")
                print("   allow_tf32=True   %-16s max relative change %.6e"
                      % (label, rel))
                out["tf32_rel_" + label.split()[0]] = rel
    torch.backends.cuda.matmul.allow_tf32 = False
    del m, x, refs, a, o
    torch.cuda.empty_cache()

    print()
    print("cumsum(dim=-1) run-to-run spread on CUDA, 20 repeats, same input.")
    print("`reduce` is the reduction length; `batch` is the number of rows.")
    print("The arms run reduce=64 at batch up to 32768. R2's exact shape is")
    print("marked. A NONZERO spread is the must-fire: the length at which the")
    print("narrow claim stops holding.")
    print("%9s %9s %13s %16s  %s"
          % ("reduce", "batch", "elements", "max|rep-first|", "note"))
    grid = [(64, 1, "1-D, the sibling node's short case"),
            (10_000, 1, "1-D, sibling's middle case"),
            (1_000_000, 1, "1-D, sibling's drifting case"),
            (64, 2048, "R1' shape"),
            (64, 32768, "**R2's exact shape**"),
            (64, 262144, "8x R2, to push the launch further"),
            (4096, 512, "same elements as R2, longer reduction")]
    rows = []
    for reduce_n, batch, note in grid:
        torch.manual_seed(0)
        v = torch.randn(batch, reduce_n, dtype=torch.float64, device="cuda")
        first = torch.cumsum(v, dim=-1)
        worst = 0.0
        for _ in range(19):
            worst = max(worst,
                        float((torch.cumsum(v, dim=-1) - first).abs().max()))
        rows.append(dict(reduce=reduce_n, batch=batch, spread=worst))
        print("%9d %9d %13d %16.6e  %s"
              % (reduce_n, batch, reduce_n * batch, worst, note))
        del v, first
        torch.cuda.empty_cache()
    out["cumsum"] = rows

    print()
    print("The same question one level up: the GRADIENT of a scan arm at R2's")
    print("shape, 8 repeats, fresh model each time, under warn_only. This is")
    print("the quantity a verdict is actually built from, not a bare op.")
    print("(arm_pl only -- arm_phase does not fit at n=32768, section C3.)")
    root = str(pathlib.Path(__file__).resolve().parents[1])
    src = textwrap.dedent("""
        import json, sys, torch
        sys.path.insert(0, sys.argv[1])
        from ceq import arm_pl
        from scale.m3_capability import D_MODEL
        torch.use_deterministic_algorithms(True, warn_only=True)
        n = int(sys.argv[2])
        torch.manual_seed(1)
        x = torch.randn(n, 64, D_MODEL, device="cuda")
        y = torch.randn(n, device="cuda")
        ref, worst = None, 0.0
        for _ in range(8):
            torch.manual_seed(0)
            m = arm_pl.ArmPL(64, d_model=D_MODEL).to("cuda")
            torch.nn.functional.mse_loss(m(x), y).backward()
            g = torch.cat([q.grad.reshape(-1) for q in m.parameters()])
            if ref is None:
                ref = g.clone()
            else:
                worst = max(worst, float((g - ref).abs().max()))
            del m, g
            torch.cuda.empty_cache()
        print(json.dumps({"n": n, "spread": worst}))
    """).strip()
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    grads = []
    for n in (2048, 32768):
        r = subprocess.run([sys.executable, "-c", src, root, str(n)],
                           capture_output=True, text=True, env=env)
        try:
            rec = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:                                     # noqa: BLE001
            rec = {"n": n, "error": (r.stderr.strip().splitlines()
                                     or ["no output"])[-1]}
        grads.append(rec)
        print("   arm_pl gradient at n=%-6d %s"
              % (n, ("max|rep-first| over 8 repeats = %.6e" % rec["spread"])
                 if "spread" in rec else "FAILED: " + rec["error"][:60]))
    out["arm_gradient"] = grads
    print()
    print("it11_verdict.by_seed's same-seed disagreement rule is 1e-9; M-10's")
    print("cross-thread floor on eval_nrmse is 2.345e-03. Both are printed")
    print("beside these numbers in V16_DEVICE_CERT.md section 5.3 so the")
    print("comparison is against a stated rule and not against zero.")
    return out


def main() -> int:
    env = section_a()
    b = section_b()
    c = section_c()
    d = section_d()
    e = section_e()
    g = section_g()
    h = section_h()
    i = section_i()
    cont = max(r["median"] for r in b["contention"]) / \
        min(r["median"] for r in b["contention"])
    cont_g = max(r["cuda_median"] for r in b["contention"]) / \
        min(r["cuda_median"] for r in b["contention"])
    f = section_f(d, d["mult"], c["caps"], cont, cont_g)
    print()
    print("=" * 78)
    print("MACHINE-READABLE SUMMARY")
    print("=" * 78)
    print(json.dumps(dict(
        gpu=env.get("gpu"), vram_free=env.get("vram_free"),
        ram_avail=env.get("ram_avail"),
        cpu_law=dict(slope=d["cpu"]["fit"]["slope"], r2=d["cpu"]["fit"]["r2"],
                     intercept=d["cpu"]["fit"]["intercept"]),
        cuda_law=dict(slope=d["cuda"]["fit"]["slope"], r2=d["cuda"]["fit"]["r2"],
                      intercept=d["cuda"]["fit"]["intercept"]),
        constants=c["consts"], caps=c["caps"], arm_multiplier=d["mult"],
        contention_factor=cont, contention_factor_cuda=cont_g,
        calibrate_takes_device=e["takes_device"],
        recert_seconds=e["calibrate"]["cuda"]["secs"] + e["red"]["cuda"],
        determinism={str(k): v for k, v in g.items()},
        finite_gradient=h, determinism_boundary=i,
        totals=f), indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
