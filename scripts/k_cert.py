"""G0.6 K-CERT -- the device certificate the Kaggle round reproduces against.

ONE SCRIPT, ONE JSON CERTIFICATE. It runs unchanged on the certified local box
(RTX 4060 Laptop, 8 GiB) and on a Kaggle P100-16GB / T4-16GB, and writes a
record keyed by `box` so a later Kaggle run APPENDS a second certificate beside
this one rather than overwriting it. The local box DECIDES; Kaggle REPRODUCES.

WHAT IT CERTIFIES, and each half carries its must-fire negative in
`tests/gate0/test_g06_kcert.py`:

  1. THROUGHPUT LAW, REFITTED, WITH R^2. `s/step = exp(log_a) * n^b`, log-log
     OLS over the cell ladder, per arm. A fit with no R^2 is not a law, and an
     R^2 that never rejects anything is not a test -- `refuse_weak_fit` raises
     below `MIN_R2` and the test plants a non-power-law series to see it fire.
  2. MEMORY LAW, REFITTED AGAINST THE CUDA ALLOCATOR. `ceq/sizing.py` carries
     `C_RESIDUAL = 18`, `C_OPERATOR = 3.9` and `DTYPE_MODES["bf16_autocast"] =
     (2.2, 3.4)`, all calibrated on this box in an earlier round. They are
     RE-SOLVED here from `torch.cuda.max_memory_allocated()` at the same shapes
     the module's docstring names (B=4 d=256 L=4 H=4, seq sweep, fp32 and bf16
     autocast). If they reproduce, that is a MEASURED CONFIRMATION and is
     tagged as one; it is not an inheritance. This script does not edit the
     module -- if a constant is wrong the certificate says so and the report
     carries the correct one with its measurement.
  3. BAR RE-CERTIFICATION delta/tol TABLE. `calibrate_bar` on cpu and on cuda,
     four rungs x five clauses, every tolerance quoted from code that predates
     this script (`bar_verdict`'s body, `flipper_tol`'s default). The worst
     `delta/tol` must be < 50%; at or above it the script PRINTS HALT and the
     certificate carries the halt string.
  4. 0-STEP RED GATE AT ALL CELL SHAPES. `train_with_checkpoints`'s gate --
     NaN checked first, then `>= 1.0 - GATE_TOL` on both splits -- evaluated on
     the as-constructed arm at every `(arm, n_train)` in the ladder. A shape
     that does not fit is recorded as unreachable rather than skipped.
  5. DETERMINISM AT THE ARMS' REDUCTION LENGTH (64), IN BOTH REGIMES. The
     workhorse arm is `ceq/arm_smprime.py`, whose hop is a masked CUMPROD.
     `torch.use_deterministic_algorithms(True)` is measured for EXECUTABILITY
     and for bitwise repeatability with the flag ON and with it OFF, and both
     regimes are reported. Neither is picked as "the" answer here: the flag's
     executability for the phase/PL arms (cumsum) is another node's item and
     the contrast is printed beside this arm's reading.

NOTHING TRAINS AS A RESEARCH READING (L-LEAN). Every timing and allocation runs
on `torch.randn` or on an `M3_TASKS` corpus draw used for its SHAPE and BYTES
only. `calibrate_bar` is called for its five clause values because the delta/tol
table IS those values compared across devices -- no arm is scored, no cell of
R1'/R2/any bed is run, no seed is pooled, no verdict is formed. The 0-step gate
reads an UNTRAINED arm by construction: zero gradient steps are taken anywhere
in this file.

L-TIME. No timing figure is inherited. `V16_DEVICE_CERT.md`'s CUDA law
`exp(-11.9670) * n^0.9734` is quoted in the report as the thing this run is
compared AGAINST, never as an input to it.

PROVENANCE. Everything in the certificate is [MEASURED] on the box named in
`box` in this run unless the key says otherwise.
"""
from __future__ import annotations

import os

#: Set BEFORE torch imports and before any cuBLAS handle exists. `main()` in
#: `scale/r10_capacity_sweep.py` sets the same value for its cuda path; the bar
#: comparison below is only a comparison of the shipped paths if this matches.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
import math                                                        # noqa: E402
import pathlib                                                     # noqa: E402
import platform                                                    # noqa: E402
import statistics                                                  # noqa: E402
import subprocess                                                  # noqa: E402
import sys                                                         # noqa: E402
import time                                                        # noqa: E402

import torch                                                       # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceq import lm, sizing                                         # noqa: E402
from ceq import arm_smprime                                        # noqa: E402
from scale.m3_capability import Arm, D_MODEL, LR, bad              # noqa: E402
from scale.negation_scope import (M3_TASKS, calibrate_bar,         # noqa: E402
                                  bar_verdict, nrmse)
from scale.r10_capacity_sweep import GATE_TOL, N_EVAL, S, D        # noqa: E402

SCHEMA = "k_cert/1"

#: A fit below this is not published as a law. 0.99 is not a convention picked
#: here: `V16_DEVICE_CERT.md` section 4.1's two published fits read 0.999704 and
#: 0.999384, so 0.99 sits an order of magnitude below the worst fit this
#: instrument has ever produced and still rejects a series with no structure.
MIN_R2 = 0.99

#: The contract's line. At or above it, G0.6 HALTS.
HALT_FRACTION = 0.50

#: The registered `n` ladder. R1' runs at 2048 (`scripts/v15_r1.py --n-train`
#: default) and R2 at 32768 (`CEQ_V16_CONTRACT.md` PART IV); the three between
#: are the powers of two the wall-clock law is fitted over.
CELL_NS = (2048, 4096, 8192, 16384, 32768)

#: The arms priced. `arm_smprime` is the WORKHORSE (`CEQ_V16_CONTRACT.md`
#: PART IV names it for R1'); `softmax` is the control the law was historically
#: fitted on, kept so this refit is comparable to the one it replaces.
#: `arm_phase` is ABSENT ON PURPOSE -- it never trains, and pricing a chunk
#: table with an arm that will not take a step is the defect this round exists
#: to avoid.
CERT_ARMS = ("softmax", "arm_smprime")

#: The arms' reduction length: `s = 64` is the sequence axis every `e3` row
#: uses, and it is the length of both contractions the operator performs -- the
#: masked cumulative product along a row, and the `a @ x` sum over `j`.
REDUCTION_LENGTH = 64

WARMUP, TIMED = 2, 12

#: A timed window must be at least this long, MEASURED REASON: at `n = 2048`
#: twelve steps of the softmax arm is 0.12 s of work, and two runs of exactly
#: that window on this box read 0.010200 and 0.011836 s/step -- 14% apart, which
#: moved the workhorse arm's fitted exponent from 0.8698 to 1.0034 and its R^2
#: from 0.993309 to 0.999996. A laptop GPU's clock is still ramping inside a
#: window that short. Steps are therefore added until BOTH `TIMED` steps and
#: this many seconds of timed work have been done.
MIN_TIMED_SECONDS = 3.0
#: Hard cap so a pathologically fast shape cannot spin: 2048 steps at the
#: fastest measured rate here is ~20 s.
MAX_TIMED_STEPS = 2048
GIB = 1024 ** 3

#: The bar's tolerances, quoted from code that predates this script. `None`
#: means the clause is ONE-SIDED and has no numeric tolerance to divide by, so
#: `delta` is reported against the reading's own margin to the bar -- the
#: quantity that would have to be crossed for the verdict to move.
BAR_TOL = {
    "predict_the_mean": 1e-6,        # bar_verdict: abs(x - 1.0) > 1e-6 -> FAIL
    "payload_only": None,            # bar_verdict: x >= 1.0; margin is x - 1.0
    "oracle": 1e-6,                  # bar_verdict: x < 1e-6
    "flipper_dependence": 0.05,      # bar_verdict(flipper_tol=0.05) default
    "trained_two_feature": None,     # bar_verdict: x < 1.0; margin is 1.0 - x
}
BAR_TASKS = ("e3_t1", "e3_t2", "e3_t8", "e3_t32")


class NotALaw(Exception):
    """A fit whose R^2 is below `MIN_R2`. Raised rather than printed, because a
    law nobody can rely on must not be reachable by a caller that forgot to
    look at the R^2 column."""


# ============================================================== pure helpers
# Everything in this block is a function of its arguments only, so
# `tests/gate0/test_g06_kcert.py` can plant a fabricated reading in each and
# watch the corresponding half of the certificate fire.

def fit_power_law(xs, ys) -> dict:
    """log-log OLS of `y = exp(log_a) * x^b`, with the R^2 of the log fit.

    R^2 is on the LOG residuals, which is the quantity the fit actually
    minimises; quoting an R^2 from a different space than the fit is how a
    law gets published with a number that flatters it.
    """
    lx = [math.log(float(x)) for x in xs]
    ly = [math.log(float(y)) for y in ys]
    n = len(lx)
    if n < 3:
        raise ValueError("a power law over %d points is not a fit" % n)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    b = sum((lx[i] - mx) * (ly[i] - my) for i in range(n)) / sxx
    log_a = my - b * mx
    ss_res = sum((ly[i] - (log_a + b * lx[i])) ** 2 for i in range(n))
    ss_tot = sum((v - my) ** 2 for v in ly)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return dict(log_a=log_a, b=b, r2=r2, n_points=n,
                x=[float(v) for v in xs], y=[float(v) for v in ys])


def refuse_weak_fit(fit: dict, what: str) -> dict:
    if fit["r2"] < MIN_R2:
        raise NotALaw("%s: R^2 %.6f < %.6f -- not published as a law"
                      % (what, fit["r2"], MIN_R2))
    return fit


def predict_secs(fit: dict, n: int) -> float:
    return math.exp(fit["log_a"]) * float(n) ** fit["b"]


def steps_in_hours(fit: dict, n: int, hours: float) -> int:
    """Steps that fit in `hours` at `n`, from the REFITTED law. The chunk table
    is this function; if it stops depending on `fit` it stops being a table."""
    return int(hours * 3600.0 / predict_secs(fit, n))


def delta_over_tol(cpu: float, cuda: float, tol: float):
    """(delta, delta/tol). `tol` is the clause's own tolerance or, for a
    one-sided clause, the reading's margin to the bar."""
    d = abs(float(cpu) - float(cuda))
    return d, (float("inf") if tol == 0 else d / float(tol))


def halt_on_bar(rows):
    """The 50% line. Returns the HALT string, or None if every row clears it."""
    live = [r for r in rows if r.get("delta_over_tol") is not None]
    if not live:
        return "HALT: no bar row carries a delta/tol -- the table cannot certify anything"
    w = max(live, key=lambda r: r["delta_over_tol"])
    if w["delta_over_tol"] >= HALT_FRACTION:
        return ("HALT: worst delta/tol %.6f >= %.2f at %s/%s -- the bar does not "
                "survive the device boundary and no cell may be read against it"
                % (w["delta_over_tol"], HALT_FRACTION, w["task"], w["clause"]))
    return None


def zero_step_gate(r0t: float, r0e: float) -> bool:
    """`train_with_checkpoints`'s RED gate, NaN checked FIRST because
    `float('nan') >= 1.0` is False and would pass a broken instrument
    silently."""
    return ((not bad(r0t)) and (not bad(r0e))
            and r0t >= 1.0 - GATE_TOL and r0e >= 1.0 - GATE_TOL)


def determinism_verdict(deltas) -> dict:
    m = max(abs(float(d)) for d in deltas) if deltas else float("nan")
    return dict(bitwise=bool(m == 0.0), max_abs=m, repeats=len(deltas))


def ols(design, target) -> dict:
    """Least squares for a small over-determined system, with R^2.

    Normal equations solved by `torch.linalg.lstsq` on float64 -- already a
    dependency, and a 2x2 hand-rolled inverse is the kind of thing that is
    silently wrong on a near-collinear design.
    """
    A = torch.tensor(design, dtype=torch.float64)
    y = torch.tensor(target, dtype=torch.float64).unsqueeze(1)
    coef = torch.linalg.lstsq(A, y).solution.squeeze(1)
    pred = (A @ coef.unsqueeze(1)).squeeze(1)
    ss_res = float(((y.squeeze(1) - pred) ** 2).sum())
    ss_tot = float(((y.squeeze(1) - y.mean()) ** 2).sum())
    return dict(coef=[float(c) for c in coef], n_points=len(target),
                r2=1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot,
                pred=[float(v) for v in pred], y=[float(v) for v in target])


# ================================================================= the device

def build_arm(kind: str, s: int = S):
    """CONSTRUCT, THEN MOVE -- `scripts/v15_r1.py::train_one`'s ordering, so the
    initial weights are drawn from the seeded CPU generator on every device and
    a Kaggle run starts from the same bytes this box did."""
    if kind == "arm_smprime":
        return arm_smprime.ArmSMPrime(s, d_model=D_MODEL)
    return Arm(kind, s)


def _oom(exc) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc)


def probe(what: str, kind: str, n: int, s: int = S) -> dict:
    """Run ONE shape measurement in a CHILD PROCESS and read its JSON back.

    NOT over-engineering: measured here, an exhausted 8 GiB device raises
    `RuntimeError: CUDA error: out of memory` -- the DRIVER's OOM, not the
    caching allocator's `torch.cuda.OutOfMemoryError` -- and that error POISONS
    the context: the very next `torch.cuda.empty_cache()` raises the same thing
    and every later shape in the same process reads NO FIT whether or not it
    fits. An in-process try/except therefore cannot find the fit boundary; it
    can only find the first shape past it. One process per shape is the
    smallest thing that makes "does this cell fit" a measurement.

    A child that dies for any reason is recorded as NOT FITTING with the
    reason, so a crash can never be read as a passing measurement.
    """
    spec = json.dumps(dict(what=what, kind=kind, n=int(n), s=int(s)))
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()),
                        "--probe", spec],
                       capture_output=True, text=True, cwd=str(ROOT))
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    lines = (r.stderr or r.stdout).strip().splitlines()
    named = [l for l in lines if "out of memory" in l or "Error" in l]
    return dict(kind=kind, n=int(n), s=int(s), fits=False, median=None, peak=None,
                why="child exit %d: %s" % (r.returncode,
                                           (named or lines or ["no output"])[-1].strip()[:160]))


def timed_steps(kind: str, n: int, device: str, s: int = S) -> dict:
    """Median seconds per (forward + backward + `Adam.step`), warm-up discarded.

    Median and not mean: a laptop GPU's clock is not stationary over a run and
    one thermal excursion moves a mean. `torch.randn` data -- step cost is a
    function of SHAPE, and no value here is ever scored.
    """
    torch.manual_seed(0)
    total_vram = (torch.cuda.get_device_properties(0).total_memory
                  if device == "cuda" else 0)
    model = build_arm(kind, s).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.randn(n, s, D_MODEL, device=device)
    y = torch.randn(n, device=device)
    per, spilled, peak, reserved = [], False, 0, 0
    try:
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        while True:
            t0 = time.perf_counter()
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
            per.append(time.perf_counter() - t0)
            #: SPILL, DETECTED RATHER THAN INFERRED, and the quantity is
            #: RESERVED and not ALLOCATED. Reserved is what the caching
            #: allocator actually took from the driver; when it exceeds the
            #: card, the driver is backing the reservation with HOST memory
            #: over PCIe and the loop is timing the bus. Measured here: the
            #: workhorse arm at `n = 8192` allocates 3.179 GiB and reserves
            #: 5.289 GiB (0.66 of the card) and runs at 0.377 s/step; at
            #: `n = 16384` it allocates 6.343 GiB -- still under the card --
            #: but reserves 11.721 GiB, 1.47x the card, and runs at 1.172
            #: s/step, 3.1x what the in-VRAM law extrapolates. ALLOCATED alone
            #: calls that shape resident and would have put a PCIe number in
            #: the law.
            if total_vram and reserved > total_vram:
                spilled = True
            timed = sum(per[WARMUP:])
            if (len(per) >= WARMUP + TIMED and timed >= MIN_TIMED_SECONDS) \
                    or len(per) >= MAX_TIMED_STEPS:
                break
            #: A spilled shape is still reported -- with a median over at least
            #: three timed steps -- but it does not buy twelve more paging steps
            #: for a number that cannot enter a fit.
            if spilled and len(per) >= WARMUP + 3:
                break
    except RuntimeError as e:
        if not _oom(e):
            raise
        del model, opt, x, y
        if device == "cuda":
            torch.cuda.empty_cache()
        return dict(kind=kind, n=n, s=s, device=device, fits=False, spilled=None,
                    median=None, peak=None,
                    why="driver OOM: %s" % str(e).splitlines()[0][:120])
    body = per[WARMUP:] or per
    del model, opt, x, y
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(kind=kind, n=n, s=s, device=device, fits=True, spilled=bool(spilled),
                median=float(statistics.median(body)), peak=float(peak),
                reserved=float(reserved), total_vram=float(total_vram),
                first=float(per[0]), last=float(per[-1]),
                spread=float(max(body) - min(body)), steps=len(per),
                timed_seconds=float(sum(body)),
                why=("allocator RESERVED %.3f GiB from a %.3f GiB card (allocated "
                     "%.3f GiB) -- the driver is paging over PCIe and this is a bus "
                     "measurement, not an arm measurement"
                     % (reserved / GIB, total_vram / GIB, peak / GIB)) if spilled else None)


def cell_peak_bytes(kind: str, n: int, device: str, s: int = S) -> dict:
    """Peak allocator bytes above the resident baseline for one fwd+bwd, plus
    the operator tensor's bytes/element read off the LIVE graph (the workhorse
    arm's operator is complex, and a memory model that assumes fp32 for it is
    wrong by 2x)."""
    torch.manual_seed(0)
    model = build_arm(kind, s).to(device)
    x = torch.randn(n, s, D_MODEL, device=device)
    y = torch.randn(n, device=device)
    try:
        model(x[:1])                       # warm cuBLAS workspace + any mask cache
        model.zero_grad(set_to_none=True)
        ob = _operator_bytes(model, x[:1])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        torch.nn.functional.mse_loss(model(x), y).backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        reserved = torch.cuda.max_memory_reserved()
        total_vram = torch.cuda.get_device_properties(0).total_memory
    except RuntimeError as e:
        if not _oom(e):
            raise
        del model, x, y
        torch.cuda.empty_cache()
        return dict(kind=kind, n=n, s=s, fits=False, peak=None, why="CUDA OOM")
    del model, x, y
    torch.cuda.empty_cache()
    return dict(kind=kind, n=n, s=s, fits=True, peak=float(peak),
                reserved=float(reserved), total_vram=float(total_vram),
                resident=bool(reserved <= total_vram),
                operator_dtype=ob[0], operator_bytes_per_element=ob[1])


@torch.no_grad()
def _operator_bytes(model, x1):
    q, k = model.wq(x1), model.wk(x1)
    a = (model._operator(q, k, *model.heads(x1)) if hasattr(model, "heads")
         else model._operator(q, k))
    return str(a.dtype), int(a.element_size())


# ---------------------------------------------------- the sizing recalibration

def tinylm_peak(kind: str, seq: int, *, bs=4, d=256, layers=4, heads=4,
                autocast=False) -> float:
    """`ceq/sizing.py`'s own calibration shape, re-run against the allocator.

    The module docstring's table is B=4 d=256 L=4 H=4 fp32 forward+backward, so
    that is what this reproduces -- not a nearby shape whose agreement would be
    a different claim.
    """
    torch.manual_seed(0)
    m = lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=0).to("cuda")
    x = torch.randint(0, lm.VOCAB, (bs, seq), device="cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = m(x)
    torch.nn.functional.cross_entropy(
        out.reshape(-1, lm.VOCAB).float(), x.reshape(-1)).backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del m, x, out
    torch.cuda.empty_cache()
    return float(peak)


def sizing_refit(seqs=(128, 256, 512, 1024), *, bs=4, d=256, layers=4, heads=4,
                 autocast=False) -> dict:
    """Re-solve `C_RESIDUAL` and `C_OPERATOR` from the allocator.

    Two terms, and they are separated by the SEQ SWEEP and not by assertion:
    the residual term grows as `L*B*S*d` and the operator term as
    `L*H*B*S*S`, so at fixed `B, d, L, H` they are linearly independent in `S`.
    An `n` sweep alone could not separate them -- both are linear in `n` -- and
    a one-point back-solve is not a fit.

    The head term is held at the module's `C_HEAD = 2` (logits + the
    log-softmax `cross_entropy` forms) and subtracted, because it is the one
    term whose count is structural rather than fitted.
    """
    bpe = 2.0 if autocast else 4.0
    design, target, points = [], [], []
    for kind, is_signed in (("softmax", 0.0), ("signed", 1.0)):
        for sq in seqs:
            peak = tinylm_peak(kind, sq, bs=bs, d=d, layers=layers, heads=heads,
                               autocast=autocast)
            head = sizing.C_HEAD * bs * sq * lm.VOCAB * 4.0
            design.append([layers * bs * sq * d, is_signed * layers * heads * bs * sq * sq])
            target.append(peak - head)
            points.append(dict(kind=kind, seq=sq, peak=peak, head=head))
    f = ols(design, target)
    c_resid_bytes, c_op_bytes = f["coef"]
    return dict(points=points, r2=f["r2"], n_points=f["n_points"],
                autocast=bool(autocast), nominal_bytes_per_element=bpe,
                c_residual_x_bytes=c_resid_bytes, c_operator_x_bytes=c_op_bytes,
                c_residual_at_4B=c_resid_bytes / 4.0,
                c_operator_at_4B=c_op_bytes / 4.0,
                c_residual_at_module_count=c_resid_bytes / sizing.C_RESIDUAL,
                c_operator_at_module_count=c_op_bytes / sizing.C_OPERATOR,
                module_c_residual=sizing.C_RESIDUAL,
                module_c_operator=sizing.C_OPERATOR,
                module_dtype_modes=sizing.DTYPE_MODES)


# ------------------------------------------------------------------- the bar

def bar_rows(device_label: str, device) -> dict:
    out = {}
    for task in BAR_TASKS:
        batch_fn, oracle_fn, feature_fn, fd_fn = M3_TASKS[task]
        cal = calibrate_bar(n=N_EVAL, s=S, d=D, steps=600, lr=LR,
                            batch_fn=batch_fn, oracle_fn=oracle_fn,
                            feature_fn=feature_fn, device=device)
        ok, why = bar_verdict(cal, flipper_dependence=None if fd_fn is None else fd_fn(S))
        out[task] = dict(cal={k: float(v) for k, v in cal.items()},
                         ok=bool(ok), why=why, device=device_label,
                         exact_fd=None if fd_fn is None else float(fd_fn(S)))
        print("  [%s] %-7s BAR %s -- %s" % (device_label, task,
                                            "CALIBRATED" if ok else "BROKEN", why),
              flush=True)
    return out


def bar_table(cpu: dict, cuda: dict) -> dict:
    rows = []
    for task in BAR_TASKS:
        for clause, tol in BAR_TOL.items():
            c, g = cpu[task]["cal"][clause], cuda[task]["cal"][clause]
            if tol is None:
                # one-sided: the margin to the bar IS the tolerance the clause
                # is used at. `payload_only >= 1.0` and `trained_two_feature < 1.0`.
                use = abs(c - 1.0)
            else:
                use = tol
            d, dt = delta_over_tol(c, g, use)
            rows.append(dict(task=task, clause=clause, cpu=c, cuda=g, delta=d,
                             tol=use, tol_is_margin=tol is None, delta_over_tol=dt))
    worst = max(rows, key=lambda r: r["delta_over_tol"])
    return dict(rows=rows, worst=worst, halt=halt_on_bar(rows),
                verdict_cpu={t: cpu[t]["ok"] for t in BAR_TASKS},
                verdict_cuda={t: cuda[t]["ok"] for t in BAR_TASKS})


# ------------------------------------------------------------- the 0-step gate

def zero_step_cells(device, seeds=(0, 1, 2), task: str = "e3_t2") -> list:
    """The RED gate at every `(arm, n_train)` in the ladder.

    The eval split is drawn ONCE at `N_EVAL` seed 12345 -- `r10_capacity_sweep`
    and `scripts/v15_r1.py` both pin it there, and re-drawing it per cell would
    make the column a reading of the draw rather than of the arm.
    """
    batch_fn = M3_TASKS[task][0]
    x_ev, y_ev, _, _ = batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345, device=device)
    cells = []
    for n in CELL_NS:
        try:
            x_tr, y_tr, _, _ = batch_fn(n, S, D, d_model=D_MODEL, seed=0, device=device)
        except RuntimeError as e:
            if not _oom(e):
                raise
            for kind in CERT_ARMS:
                cells.append(dict(arm=kind, n_train=n, seed=None, ok=None,
                                  why="corpus draw OOM"))
            continue
        mu = float(y_tr.mean())
        sigma = float(y_tr.std(unbiased=False)) or 1.0
        for kind in CERT_ARMS:
            for seed in seeds:
                torch.manual_seed(seed)
                model = build_arm(kind, S).to(device)
                try:
                    model.eval()
                    with torch.no_grad():
                        r0t = nrmse(model(x_tr) * sigma + mu, y_tr)
                        r0e = nrmse(model(x_ev) * sigma + mu, y_ev)
                except RuntimeError as e:
                    if not _oom(e):
                        raise
                    del model
                    torch.cuda.empty_cache()
                    cells.append(dict(arm=kind, n_train=n, seed=seed, ok=None,
                                      why="forward OOM at this cell shape"))
                    continue
                del model
                ok = zero_step_gate(r0t, r0e)
                cells.append(dict(arm=kind, n_train=n, seed=seed, ok=bool(ok),
                                  nrmse0_train=float(r0t), nrmse0_eval=float(r0e),
                                  gate_tol=GATE_TOL,
                                  margin=float(min(r0t, r0e) - (1.0 - GATE_TOL))))
                print("    %-12s n=%-6d seed=%d  r0t %.9f  r0e %.9f  %s"
                      % (kind, n, seed, r0t, r0e, "OK" if ok else "RED-GATE FIRED"),
                      flush=True)
        del x_tr, y_tr
        if str(device) == "cuda":
            torch.cuda.empty_cache()
    return cells


# ------------------------------------------------------------- determinism@64

def determinism_at_64(device, repeats: int = 8, n: int = 512) -> dict:
    """Bitwise repeatability of the WORKHORSE arm at reduction length 64.

    Three quantities, because they fail for different reasons: the arm's hop
    (a masked CUMPROD along a row of 64), the arm's whole forward (the hop plus
    the `a @ x` contraction over j, also length 64), and the arm's GRADIENT
    (the backward of both). `cumsum` is measured beside them ONLY as the
    contrast the phase/PL arms would hit -- this arm does not use it.
    """
    torch.manual_seed(0)
    model = build_arm("arm_smprime", S).to(device)
    x = torch.randn(n, S, D_MODEL, device=device)
    y = torch.randn(n, device=device)
    m = torch.randn(n, S, device=device)
    th = torch.randn(n, S, device=device)
    out = dict(repeats=repeats, n=n, reduction_length=REDUCTION_LENGTH)

    def repeatability(fn):
        """(verdict | not-executable) for ONE quantity.

        PER QUANTITY and not per regime, because they do not stand or fall
        together: `cumprod`'s FORWARD is executable under the flag on CUDA in
        torch 2.5.1 and its BACKWARD is not -- autograd differentiates a
        cumulative product through `cumsum`, which has no deterministic CUDA
        kernel. Collapsing the three into one boolean would report "the arm is
        blocked" when what is blocked is the gradient step alone, and the
        0-step gate and the bar both run under `no_grad`.
        """
        try:
            ref = fn()
            d = [float((fn() - ref).abs().max()) for _ in range(repeats)]
        except RuntimeError as e:
            return dict(executable=False,
                        error="%s: %s" % (type(e).__name__, str(e).splitlines()[0][:300]))
        return dict(determinism_verdict(d), executable=True, error=None)

    def grad_vec():
        model.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), y).backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters()
                          if p.grad is not None]).clone()

    def fwd():
        with torch.no_grad():
            return model(x)

    out["hop"] = repeatability(lambda: arm_smprime.hop(m, th, g=1.0)[1])
    out["forward"] = repeatability(fwd)
    out["gradient"] = repeatability(grad_vec)
    del model, x, y, m, th
    if str(device) == "cuda":
        torch.cuda.empty_cache()

    parts = [out["hop"], out["forward"], out["gradient"]]
    live = [p for p in parts if p["executable"]]
    out["executable"] = all(p["executable"] for p in parts)
    out["executable_no_grad"] = out["hop"]["executable"] and out["forward"]["executable"]
    out["error"] = next((p["error"] for p in parts if not p["executable"]), None)
    out["bitwise"] = all(p["bitwise"] for p in live) if live else None
    out["max_abs"] = max((p["max_abs"] for p in live), default=None)
    return out


def cumsum_under_flag(device, n: int = 512) -> dict:
    """The contrast the report owes the reader: `cumsum` is what `arm_phase`
    and `arm_pl` reduce with, and whether IT is executable under the flag is a
    different question from whether this arm's `cumprod` is. Measured, not
    asserted, and never used to decide this arm's verdict."""
    try:
        t = torch.randn(n, S, device=device, requires_grad=True)
        c = torch.cumsum(t, dim=-1)
        c.sum().backward()
        return dict(executable=True, error=None,
                    grad_finite=bool(torch.isfinite(t.grad).all()))
    except RuntimeError as e:
        return dict(executable=False,
                    error="%s: %s" % (type(e).__name__, str(e)[:400]))


# ==================================================================== the run

def git(*args) -> str:
    try:
        return subprocess.check_output(["git"] + list(args), cwd=str(ROOT),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:                                    # pragma: no cover
        return "UNAVAILABLE: %s" % e


def box_record(device: str) -> dict:
    r = dict(device=device, torch=torch.__version__, python=platform.python_version(),
             platform=platform.platform(), threads=torch.get_num_threads(),
             cuda_available=bool(torch.cuda.is_available()),
             allow_tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
             allow_tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
             cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
             device_name="cpu")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        r.update(device_name=p.name, capability="%d.%d" % (p.major, p.minor),
                 total_vram_bytes=int(p.total_memory), n_gpus=torch.cuda.device_count())
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "results" / "k_cert_local.json"))
    ap.add_argument("--threads", type=int, default=8,
                    help="pinned AFTER imports; 8 is the pin every e3 row uses")
    ap.add_argument("--only", default="",
                    help="comma list of stages to run (throughput,memory,bar,"
                         "zerostep,determinism); empty runs all")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent children per timed shape; a shape must "
                         "complete in ALL of them to enter the fit")
    ap.add_argument("--probe", default=None,
                    help="INTERNAL: run one shape measurement and print its JSON. "
                         "See `probe()` for why a shape is measured in a child.")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    if a.probe:
        spec = json.loads(a.probe)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        fn = timed_steps if spec["what"] == "time" else cell_peak_bytes
        print(json.dumps(fn(spec["kind"], spec["n"], dev, spec["s"])))
        return 0
    want = set(x.strip() for x in a.only.split(",") if x.strip())

    def run(stage):
        return not want or stage in want

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cert = dict(schema=SCHEMA, when=time.strftime("%Y-%m-%d %H:%M:%S"),
                box=box_record(dev), min_r2=MIN_R2, halt_fraction=HALT_FRACTION,
                cell_ns=list(CELL_NS), arms=list(CERT_ARMS),
                git=dict(head=git("rev-parse", "HEAD"),
                         porcelain=git("status", "--porcelain")))
    print("=== K-CERT  %s  torch %s  device %s  threads %d ==="
          % (cert["box"]["device_name"], torch.__version__, dev, a.threads), flush=True)
    print("git HEAD %s" % cert["git"]["head"], flush=True)
    t_start = time.time()

    # ---------------------------------------------------------- 1. throughput
    if run("throughput"):
        print("\n--- 1. THROUGHPUT LAW, REFITTED (L-TIME: nothing inherited) ---",
              flush=True)
        points, laws, rejected = [], {}, {}
        for kind in CERT_ARMS:
            for n in CELL_NS:
                #: SEVERAL INDEPENDENT CHILDREN PER SHAPE. Two things are being
                #: separated and they are not the same failure:
                #:
                #:   RESIDENCY is physical and is the admission rule. A child
                #:   whose allocator RESERVED more than the card holds timed the
                #:   PCIe bus, and no number from it may enter a law.
                #:
                #:   A CHILD DEATH is a box event. `n = 4096` for the workhorse
                #:   arm -- 1.9 GiB reserved on an 8.0 GiB card, resident by a
                #:   factor of four -- lost one child of three to a hard Windows
                #:   process abort (exit 0xC0000409) in one run and completed
                #:   3/3 in another. Excluding a demonstrably resident shape
                #:   because the driver fell over once would delete a real point
                #:   and bend the exponent. Deaths are RECORDED, and a majority
                #:   of children must still complete.
                reps = [probe("time", kind, n) for _ in range(a.repeats)]
                ok = [r for r in reps if r["fits"]]
                med = [r["median"] for r in ok]
                p = dict(kind=kind, n=n, repeats=a.repeats, n_completed=len(ok),
                         medians=med,
                         spilled=any(bool(r.get("spilled")) for r in ok),
                         peak=max((r["peak"] for r in ok), default=None),
                         reserved=max((r.get("reserved", 0) for r in ok), default=None),
                         child_deaths=[str(r["why"]) for r in reps if not r["fits"]],
                         why="; ".join(sorted({str(r["why"]) for r in reps
                                               if r.get("why")})) or None)
                p["median"] = statistics.median(med) if ok else None
                #: Reported, not gated: an inconsistent point is already what
                #: the R^2 gate downstream exists to catch, and gating twice
                #: would need a second threshold nobody measured.
                p["child_spread_rel"] = ((max(med) - min(med)) / p["median"]
                                         if len(med) > 1 else None)
                p["admitted"] = bool(len(ok) >= 2 and not p["spilled"])
                points.append(p)
                if ok:
                    print("  %-12s n=%-6d %.6f s/step  x150 = %.1f s  resv %.3f GiB  "
                          "%d/%d children%s  %s"
                          % (kind, n, p["median"], 150 * p["median"],
                             p["reserved"] / GIB, len(ok), a.repeats,
                             "" if p["child_spread_rel"] is None
                             else " spread %.2f%%" % (100 * p["child_spread_rel"]),
                             "ADMITTED" if p["admitted"] else "EXCLUDED: " + str(p["why"])),
                          flush=True)
                else:
                    print("  %-12s n=%-6d DOES NOT RUN: %s" % (kind, n, p["why"]),
                          flush=True)
            got = [p for p in points if p["kind"] == kind and p["admitted"]]
            if len(got) >= 3:
                f = fit_power_law([p["n"] for p in got], [p["median"] for p in got])
                f["fitted_over"] = [p["n"] for p in got]
                print("  FIT %-12s s/step = exp(%.4f) * n^%.4f   R^2 %.6f  (%d points: %s)"
                      % (kind, f["log_a"], f["b"], f["r2"], f["n_points"],
                         f["fitted_over"]), flush=True)
                try:
                    laws[kind] = refuse_weak_fit(f, kind)
                except NotALaw as e:
                    #: A weak fit is NOT filed under `laws`. "Print the R^2" is
                    #: only a discipline if the R^2 can keep a number out.
                    rejected[kind] = dict(f, refused=str(e))
                    print("  %s" % e, flush=True)
            else:
                rejected[kind] = dict(n_points=len(got), r2=None,
                                      refused="%d admitted points -- NOT A LAW" % len(got))
                print("  %s: %d admitted points -- NOT A LAW" % (kind, len(got)), flush=True)
        cert["throughput"] = dict(points=points, laws=laws, rejected_laws=rejected,
                                  repeats=a.repeats,
                                  min_timed_seconds=MIN_TIMED_SECONDS,
                                  statistic="median over %d independent children of the "
                                            "median of >=%d timed steps AND >=%.1f s of "
                                            "timed work, %d warm-up steps discarded"
                                            % (a.repeats, TIMED, MIN_TIMED_SECONDS, WARMUP))

    # -------------------------------------------------------------- 2. memory
    if run("memory"):
        print("\n--- 2. MEMORY LAW, REFITTED AGAINST THE CUDA ALLOCATOR ---", flush=True)
        mem = dict(cells=[])
        if dev == "cuda":
            fp32 = sizing_refit(autocast=False)
            bf16 = sizing_refit(autocast=True, seqs=(256, 512, 1024))
            mem["law"] = fp32
            mem["law_bf16_autocast"] = bf16
            print("  fp32  C_RESIDUAL %.3f (module %d)   C_OPERATOR %.3f (module %.2f)"
                  "   R^2 %.6f"
                  % (fp32["c_residual_at_4B"], sizing.C_RESIDUAL,
                     fp32["c_operator_at_4B"], sizing.C_OPERATOR, fp32["r2"]), flush=True)
            print("  bf16  effective bytes/elem: residual %.3f (module %.1f)  operator "
                  "%.3f (module %.1f)   R^2 %.6f"
                  % (bf16["c_residual_at_module_count"],
                     sizing.DTYPE_MODES["bf16_autocast"][0],
                     bf16["c_operator_at_module_count"],
                     sizing.DTYPE_MODES["bf16_autocast"][1], bf16["r2"]), flush=True)
            for kind in CERT_ARMS:
                for n in CELL_NS:
                    c = probe("mem", kind, n)
                    mem["cells"].append(c)
                    print("  %-12s n=%-6d %s" % (
                        kind, n, ("peak %8.3f GiB  reserved %8.3f GiB  %-12s "
                                  "operator %s @ %d B/elem"
                                  % (c["peak"] / GIB, c["reserved"] / GIB,
                                     "RESIDENT" if c["resident"] else "SPILLS",
                                     c["operator_dtype"],
                                     c["operator_bytes_per_element"]))
                        if c["fits"] else "DOES NOT FIT: " + c["why"]), flush=True)
        else:
            mem["law"] = dict(r2=float("nan"), why="no CUDA allocator on this box")
        cert["memory"] = mem

    # ----------------------------------------------------------------- 3. bar
    if run("bar"):
        print("\n--- 3. BAR RE-CERTIFICATION, delta/tol ---", flush=True)
        print("  cpu pass: shipped defaults, deterministic_algorithms %s"
              % torch.are_deterministic_algorithms_enabled(), flush=True)
        cpu = bar_rows("cpu", None)
        cuda = None
        if dev == "cuda":
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            print("  cuda pass: deterministic_algorithms True, "
                  "CUBLAS_WORKSPACE_CONFIG=%s" % os.environ["CUBLAS_WORKSPACE_CONFIG"],
                  flush=True)
            try:
                cuda = bar_rows("cuda", torch.device("cuda"))
            finally:
                torch.use_deterministic_algorithms(False)
        if cuda is None:
            cert["bar"] = dict(rows=[], worst=None,
                               halt="HALT: no cuda pass -- delta/tol is not defined "
                                    "on a single device", cpu=cpu)
        else:
            t = bar_table(cpu, cuda)
            t["cpu"], t["cuda"] = cpu, cuda
            cert["bar"] = t
            for r in sorted(t["rows"], key=lambda r: -r["delta_over_tol"])[:6]:
                print("  %-7s %-20s delta %.3e  tol %.3e  delta/tol %.6e%s"
                      % (r["task"], r["clause"], r["delta"], r["tol"],
                         r["delta_over_tol"], "  (margin)" if r["tol_is_margin"] else ""),
                      flush=True)
            print("  WORST delta/tol %.6e at %s/%s against the %.0f%% line -- %s"
                  % (t["worst"]["delta_over_tol"], t["worst"]["task"],
                     t["worst"]["clause"], HALT_FRACTION * 100,
                     t["halt"] or "CLEARS"), flush=True)
            if t["halt"]:
                print("  " + t["halt"], flush=True)

    # ---------------------------------------------------------- 4. 0-step gate
    if run("zerostep"):
        print("\n--- 4. 0-STEP RED GATE AT ALL CELL SHAPES ---", flush=True)
        cells = zero_step_cells(torch.device(dev) if dev == "cuda" else None)
        ran = [c for c in cells if c["ok"] is not None]
        cert["zero_step"] = dict(
            cells=cells, gate_tol=GATE_TOL, task="e3_t2", n_eval=N_EVAL, s=S, d=D,
            n_reachable=len(ran), n_unreachable=len(cells) - len(ran),
            all_ok=bool(ran) and all(c["ok"] for c in ran),
            worst_margin=min((c["margin"] for c in ran), default=None))
        print("  %d/%d cell shapes reachable; gate holds on %s; worst margin %s"
              % (len(ran), len(cells),
                 "ALL" if cert["zero_step"]["all_ok"] else "NOT ALL",
                 cert["zero_step"]["worst_margin"]), flush=True)

    # --------------------------------------------------------- 5. determinism
    if run("determinism"):
        print("\n--- 5. DETERMINISM AT THE ARMS' REDUCTION LENGTH (%d) ---"
              % REDUCTION_LENGTH, flush=True)
        d = dict(reduction_length=REDUCTION_LENGTH, arm="arm_smprime",
                 device=dev, note="arm_smprime reduces with CUMPROD; cumsum is "
                                  "measured only as the contrast arm_phase/arm_pl hit")
        dv = torch.device(dev)
        def show(tag, r):
            print("  flag %s: hop %s  forward %s  gradient %s  |  bitwise %s  "
                  "max|delta| %s" % (
                      tag,
                      *[("bitwise" if q["bitwise"] else "MOVES %.3e" % q["max_abs"])
                        if q["executable"] else "NOT EXECUTABLE"
                        for q in (r["hop"], r["forward"], r["gradient"])],
                      r["bitwise"], r["max_abs"]), flush=True)

        d["flag_off"] = determinism_at_64(dv)
        d["flag_off"]["cumsum"] = cumsum_under_flag(dv)
        show("OFF", d["flag_off"])
        torch.use_deterministic_algorithms(True)
        try:
            d["flag_on"] = determinism_at_64(dv)
            d["flag_on"]["cumsum"] = cumsum_under_flag(dv)
        finally:
            torch.use_deterministic_algorithms(False)
        show("ON ", d["flag_on"])
        if d["flag_on"]["error"]:
            print("  flag ON blocks: %s" % d["flag_on"]["error"][:200], flush=True)
        print("  cumsum (arm_phase/arm_pl's reduction) under the flag: executable %s%s"
              % (d["flag_on"]["cumsum"]["executable"],
                 "" if d["flag_on"]["cumsum"]["executable"]
                 else " -- " + d["flag_on"]["cumsum"]["error"][:160]), flush=True)
        cert["determinism"] = d

    cert["elapsed_s"] = round(time.time() - t_start, 1)
    cert["git_end"] = dict(head=git("rev-parse", "HEAD"),
                           porcelain=git("status", "--porcelain"))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    print("\nwrote %s  (%.1f s)" % (out, cert["elapsed_s"]), flush=True)

    halt = cert.get("bar", {}).get("halt")
    if halt:
        print(halt, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
