"""TWO QUESTIONS ABOUT `_causal_sgate_operator`, MEASURED.

GEOMETRY. Every number below is taken at `scale/step_profile.py`'s geometry --
`--n 8192 --s 64 --d 24 --arm pivot_signed --seed 0`, `d_model=16`, `window=0`,
`torch.set_num_threads(2)`, CPU only -- because that is the run DONE.md:366
records as `operator build 25.4%` / `backward 57.4%`. Note `d=24` is
`make_batch`'s flipper DISTANCE, not the head width; the head width is
`d_model=16`, so the operator divides by `sqrt(16)`.

TASK A. `_causal_sgate_operator` calls softmax twice over the same logits `w`:
`pp = softmax(w)` and `pm = softmax(-w)`. Over the visible entries of a row,
`softmax(-w)_j` is proportional to `1/exp(w_j)`, hence to `1/pp_j`, so a derived
route exists on paper. It is implemented here and compared to the shipped `pm`
with `torch.equal` FIRST and a maxdiff second, at two geometries: the harness
one (tiny logits) and plain `randn` q/k at the same shapes (O(1) logits).

TASK B. The internal breakdown of one operator call, by TWO methods that do not
share a mechanism: `torch.profiler` self-CPU per aten op, and staged wall clock
on pre-built inputs (median of `--reps` after `--warmup`). Both tables are
printed. Forward-only and forward+backward are both reported because the step
profile puts 57% of the step in backward.

DECLARED SHORTCUTS: one seed. The staged wall clock frees nothing between
stages, so each stage sees a warm allocator. No bootstrap on the medians; min
and max are printed instead so the spread is visible.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench
from scale.m3_capability import Arm, D_MODEL, SGATE_RHO, SGATE_LAM
from scale.negation_scope import make_batch

torch.set_num_threads(2)


# ---------------------------------------------------------------- inputs ----
def build_qk(geom: str, n: int, s: int, d: int):
    """`(q, k)` at the shipped shapes. `harness` = the real arm's projections."""
    if geom == "harness":
        x, _, _, _ = make_batch(n, s, d, d_model=D_MODEL, seed=0)
        torch.manual_seed(0)
        arm = Arm("pivot_signed", s)
        with torch.no_grad():
            return arm.wq(x).contiguous(), arm.wk(x).contiguous()
    torch.manual_seed(0)
    return torch.randn(n, s, D_MODEL), torch.randn(n, s, D_MODEL)


def shipped_parts(q, k, window: int = 0):
    """The exact statements of `_causal_sgate_operator`, kept as separate names."""
    _, nm = bench._causal_mask_pair(q.shape[-2], window, str(q.device))
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    neg = torch.finfo(q.dtype).min
    pp = torch.softmax(w.masked_fill(nm, neg), -1).masked_fill(nm, 0.0)
    pm = torch.softmax((-w).masked_fill(nm, neg), -1).masked_fill(nm, 0.0)
    return nm, w, neg, pp, pm


# ---------------------------------------------------------------- task A ----
def task_a(geom: str, n: int, s: int, d: int) -> None:
    print(f"\n{'=' * 78}\nTASK A -- geometry `{geom}`  n={n} s={s} d={d} "
          f"d_model={D_MODEL} window=0\n{'=' * 78}")
    q, k = build_qk(geom, n, s, d)
    with torch.no_grad():
        nm, w, neg, pp, pm = shipped_parts(q, k)
        vis = ~nm                                  # [s, s], broadcasts over n

        wv = w[:, vis] if vis.any() else w
        print(f"w        min {float(w.min()):+.6e}  max {float(w.max()):+.6e}")
        print(f"w  (vis) min {float(wv.min()):+.6e}  max {float(wv.max()):+.6e}  "
              f"mean|w| {float(wv.abs().mean()):.6e}")
        print(f"pp       min {float(pp.min()):.6e}  max {float(pp.max()):.6e}")
        ppv = pp[:, vis]
        nz = ppv[ppv > 0]
        print(f"pp (vis) min {float(ppv.min()):.6e}  max {float(ppv.max()):.6e}")
        print(f"smallest NONZERO pp anywhere: "
              f"{float(pp[pp > 0].min()):.6e}   (visible-only: "
              f"{float(nz.min()) if nz.numel() else float('nan'):.6e})")

        # -- the naive reciprocal, UNGUARDED, counted before anything is fixed --
        naive = 1.0 / pp
        n_nonfinite = int((~torch.isfinite(naive)).sum())
        print(f"non-finite entries in naive 1/pp: {n_nonfinite} of "
              f"{naive.numel()}  ({100.0 * n_nonfinite / naive.numel():.3f}%)"
              f"   [masked entries alone = {int(nm.sum()) * n}]")
        del naive

        # ---- the derived route, masked entries guarded to 0 before the sum ----
        inv = torch.where(nm, torch.zeros_like(pp), 1.0 / pp)
        den = inv.sum(-1, keepdim=True)
        pm_derived = (inv / den.clamp_min(torch.finfo(pp.dtype).tiny)
                      ).masked_fill(nm, 0.0)
        eq = torch.equal(pm, pm_derived)
        md = float((pm - pm_derived).abs().max())
        print(f"\nROUTE 1  pm_derived = (1/pp)/sum_visible(1/pp)")
        print(f"  torch.equal(pm_shipped, pm_derived) = {eq}")
        print(f"  maxdiff                             = {md:.6e}")
        print(f"  non-finite in pm_derived            = "
              f"{int((~torch.isfinite(pm_derived)).sum())}")
        del inv, den, pm_derived

        # ---- ROUTE 2: reuse the ALREADY-MASKED logits instead of recomputing.
        # `(-wm)` equals `-w` bitwise on visible entries (negation is exact) and
        # the masked entries are overwritten by the masked_fill regardless.
        wm = w.masked_fill(nm, neg)
        pm_r2 = torch.softmax((-wm).masked_fill(nm, neg), -1).masked_fill(nm, 0.0)
        print(f"\nROUTE 2  pm = softmax((-wm).masked_fill(nm,neg))  "
              f"[reuses wm, still one softmax]")
        print(f"  torch.equal(pm_shipped, route2)     = {torch.equal(pm, pm_r2)}")
        print(f"  maxdiff                             = "
              f"{float((pm - pm_r2).abs().max()):.6e}")
        del pm_r2

        # ---- ROUTE 3: drop the POST-softmax masked_fill. exp(neg - max)
        # underflows to exactly 0 on every row that has at least one visible
        # entry; only the fully-masked row (row 0 of a strictly-causal mask)
        # differs, where softmax returns uniform 1/s.
        raw = torch.softmax(wm, -1)
        print(f"\nROUTE 3  pp without the post-softmax masked_fill")
        print(f"  torch.equal(pp_shipped, raw)        = {torch.equal(pp, raw)}")
        print(f"  maxdiff                             = "
              f"{float((pp - raw).abs().max()):.6e}")
        dead = nm.all(-1)                       # fully-masked rows
        print(f"  fully-masked rows per example       = {int(dead.sum())} of {s}")
        raw2 = raw.clone()
        raw2[:, dead, :] = 0.0
        print(f"  torch.equal(pp_shipped, raw with fully-masked rows zeroed) "
              f"= {torch.equal(pp, raw2)}")
        print(f"  maxdiff                             = "
              f"{float((pp - raw2).abs().max()):.6e}")
        del raw, raw2, wm


# ---------------------------------------------------------------- task B ----
def _median(fn, reps: int, warmup: int):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def task_b_wall(q, k, reps: int, warmup: int, window: int = 0) -> None:
    s = q.shape[-2]
    dh = q.shape[-1]
    dev = str(q.device)
    nm, w, neg, pp, pm = shipped_parts(q, k)
    wm = w.masked_fill(nm, neg)
    sp = torch.softmax(wm, -1)
    nw = -w
    nwm = nw.masked_fill(nm, neg)
    sm = torch.softmax(nwm, -1)
    rho, lam = SGATE_RHO, SGATE_LAM

    stages = [
        ("mask_pair lookup",  lambda: bench._causal_mask_pair(s, window, dev)),
        ("q@kT + /sqrt(d)",   lambda: (q @ k.transpose(-2, -1)) / math.sqrt(dh)),
        ("masked_fill#1 w",   lambda: w.masked_fill(nm, neg)),
        ("softmax#1 (pp)",    lambda: torch.softmax(wm, -1)),
        ("masked_fill#2 pp",  lambda: sp.masked_fill(nm, 0.0)),
        ("negation -w",       lambda: -w),
        ("masked_fill#3 -w",  lambda: nw.masked_fill(nm, neg)),
        ("softmax#2 (pm)",    lambda: torch.softmax(nwm, -1)),
        ("masked_fill#4 pm",  lambda: sm.masked_fill(nm, 0.0)),
        ("final arith",       lambda: rho * (pp - lam * pm) / (1.0 + lam)),
    ]
    whole = lambda: bench._causal_sgate_operator(q, k, rho=rho, lam=lam,
                                                 window=window)
    wmed, wmin, wmax = _median(whole, reps, warmup)

    print(f"\n--- METHOD 2: staged wall clock, median of {reps} after "
          f"{warmup} warmups, FORWARD ONLY ---")
    print(f"{'statement':>20} {'median s':>12} {'min s':>12} {'max s':>12} "
          f"{'% of build':>11}")
    tot = 0.0
    soft = mfill = 0.0
    for name, fn in stages:
        m, lo, hi = _median(fn, reps, warmup)
        tot += m
        if name.startswith("softmax"):
            soft += m
        if name.startswith("masked_fill"):
            mfill += m
        print(f"{name:>20} {m:>12.6f} {lo:>12.6f} {hi:>12.6f} "
              f"{100 * m / wmed:>10.1f}%")
    print(f"{'WHOLE operator call':>20} {wmed:>12.6f} {wmin:>12.6f} "
          f"{wmax:>12.6f} {100.0:>10.1f}%")
    print(f"{'sum of statements':>20} {tot:>12.6f} {'':>12} {'':>12} "
          f"{100 * tot / wmed:>10.1f}%")
    print(f"\nHEADLINE (METHOD 2, forward)  two softmaxes = {soft:.6f} s = "
          f"{100 * soft / wmed:.1f}% of operator build")
    print(f"HEADLINE (METHOD 2, forward)  four masked_fills = {mfill:.6f} s = "
          f"{100 * mfill / wmed:.1f}% of operator build")

    # ---------------------------- forward + backward ------------------------
    qg = q.detach().clone().requires_grad_(True)
    kg = k.detach().clone().requires_grad_(True)

    def fwd_bwd():
        a = bench._causal_sgate_operator(qg, kg, rho=rho, lam=lam, window=window)
        a.sum().backward()
        qg.grad = None
        kg.grad = None

    def fwd_only_grad():
        with torch.no_grad():
            bench._causal_sgate_operator(qg, kg, rho=rho, lam=lam, window=window)

    fm, flo, fhi = _median(fwd_only_grad, reps, warmup)
    bm, blo, bhi = _median(fwd_bwd, max(3, reps // 3), 2)
    print(f"\n--- METHOD 2: operator forward vs forward+backward "
          f"(a.sum().backward()) ---")
    print(f"{'variant':>28} {'median s':>12} {'min s':>12} {'max s':>12}")
    print(f"{'forward only (no_grad)':>28} {fm:>12.6f} {flo:>12.6f} {fhi:>12.6f}")
    print(f"{'forward+backward':>28} {bm:>12.6f} {blo:>12.6f} {bhi:>12.6f}")
    print(f"{'backward-implied (fb - f)':>28} {bm - fm:>12.6f}")


def task_b_profiler(q, k, window: int = 0, backward: bool = False) -> None:
    from torch.profiler import profile, ProfilerActivity
    rho, lam = SGATE_RHO, SGATE_LAM
    if backward:
        qq = q.detach().clone().requires_grad_(True)
        kk = k.detach().clone().requires_grad_(True)
    else:
        qq, kk = q, k
    # warm the allocator and the mask cache so the profiled call is steady-state
    for _ in range(2):
        if backward:
            a = bench._causal_sgate_operator(qq, kk, rho=rho, lam=lam,
                                             window=window)
            a.sum().backward()
            qq.grad = None
            kk.grad = None
        else:
            with torch.no_grad():
                bench._causal_sgate_operator(qq, kk, rho=rho, lam=lam,
                                             window=window)
    tag = "FORWARD+BACKWARD" if backward else "FORWARD ONLY"
    print(f"\n--- METHOD 1: torch.profiler, ONE call, self CPU time, {tag} ---")
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        if backward:
            a = bench._causal_sgate_operator(qq, kk, rho=rho, lam=lam,
                                             window=window)
            a.sum().backward()
        else:
            with torch.no_grad():
                bench._causal_sgate_operator(qq, kk, rho=rho, lam=lam,
                                             window=window)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
    ev = prof.key_averages()
    tot = sum(e.self_cpu_time_total for e in ev)
    soft = sum(e.self_cpu_time_total for e in ev if "softmax" in e.key.lower())
    mf = sum(e.self_cpu_time_total for e in ev
             if "masked_fill" in e.key.lower() or "fill_" in e.key.lower())
    print(f"total self CPU        {tot / 1e6:.6f} s")
    print(f"softmax family        {soft / 1e6:.6f} s = {100 * soft / tot:.1f}%")
    print(f"masked_fill family    {mf / 1e6:.6f} s = {100 * mf / tot:.1f}%")


def demo() -> None:
    """One runnable check: the guarded reciprocal route must not be silently
    non-finite, and route 2 must be bitwise, at a size that runs in a blink."""
    q, k = build_qk("randn", 4, 8, 3)
    with torch.no_grad():
        nm, w, neg, pp, pm = shipped_parts(q, k)
        inv = torch.where(nm, torch.zeros_like(pp), 1.0 / pp)
        der = (inv / inv.sum(-1, keepdim=True).clamp_min(
            torch.finfo(pp.dtype).tiny)).masked_fill(nm, 0.0)
        assert torch.isfinite(der).all(), "guarded derivation produced non-finite"
        assert der.shape == pm.shape
        wm = w.masked_fill(nm, neg)
        r2 = torch.softmax((-wm).masked_fill(nm, neg), -1).masked_fill(nm, 0.0)
        assert torch.equal(pm, r2), "route 2 is not bitwise at demo size"
    print("demo OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--task", default="all", choices=["a", "b", "all", "demo"])
    a = ap.parse_args()

    if a.task == "demo":
        demo()
        return 0

    print(f"n={a.n} s={a.s} d={a.d} d_model={D_MODEL} arm=pivot_signed seed=0 "
          f"window=0 rho={SGATE_RHO} lam={SGATE_LAM}")
    print(f"threads={torch.get_num_threads()} torch={torch.__version__} "
          f"device=cpu")

    if a.task in ("a", "all"):
        task_a("harness", a.n, a.s, a.d)
        task_a("randn", a.n, a.s, a.d)

    if a.task in ("b", "all"):
        print(f"\n{'=' * 78}\nTASK B -- step_profile geometry, harness q/k"
              f"\n{'=' * 78}")
        q, k = build_qk("harness", a.n, a.s, a.d)
        task_b_profiler(q, k, backward=False)
        task_b_profiler(q, k, backward=True)
        task_b_wall(q, k, a.reps, a.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
