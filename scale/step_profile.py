"""WHERE DOES ONE FULL-BATCH TRAINING STEP OF `run_arm` ACTUALLY GO?

THE CLAIM UNDER TEST. One full step at n=2048 and at n=8192 was reported to cost
~2.1 s BOTH. Every component of `Arm.forward` is O(n) -- the operator is [n,s,s],
the hop-2 gathers are [n,s,k], the MLP is [n,s,h] -- and the ONLY size-independent
work in the step is `opt.step()`, which walks ~4.7k parameters. Equal cost at 4x
the data is therefore not something the arithmetic can produce, and this script
exists to find out what does.

METHOD, stated because the number is worthless without it:

  * `torch.set_num_threads(2)`, `CUDA_VISIBLE_DEVICES=` empty, CPU only.
  * The staged forward below is NOT a reimplementation. It calls the SAME bound
    submodules (`model.wq`, `model._operator`, `batched_pivot_hop2`, `model.mlp`,
    `model.readout`) in the SAME order as `Arm.forward`, and it is BOUND to it by
    `torch.equal` -- BITWISE, never allclose -- before any timing is credited.
    If the bind fails the script aborts and reports nothing, because a profile of
    a different function is not a profile.
  * Each stage is timed with `time.perf_counter()` around the stage alone, inside
    a step that is otherwise identical to `run_arm`'s. `--repeats` steps are run
    after one warmup step and the MEDIAN is reported.
  * `backward` is ONE row: it is a single call over the whole graph and cannot be
    split by wall clock. `--profile` adds a second pass under
    `torch.autograd.profiler.profile` and reports self-CPU time per aten op,
    which is what attributes the backward across the stages.
  * RSS and available system memory are sampled around the step, because at
    n=8192 the operator alone is [8192,64,64] float32 = 134 MB and the forward
    holds roughly eight such tensors for autograd. If the step is paging, the
    wall clock is measuring the page cache and not the arithmetic, and that is a
    different finding than "the MLP is slow".

DECLARED SHORTCUTS: one seed, median of `--repeats` (default 3) steps after one
warmup. No bootstrap. The stage split is a wall-clock split of the FORWARD only.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.m3_capability import Arm, D_MODEL, LR
from scale.negation_scope import make_batch
from scale.pivot_probe import batched_select_pivots, batched_pivot_hop2

torch.set_num_threads(2)

STAGES = ("proj q/k", "operator build", "hop-2 select+bmm", "a@x + hop2@x",
          "MLP", "readout+loss")


def staged_forward(model, x, times: dict | None = None):
    """`Arm.forward` with a clock between the stages. BOUND bitwise to it."""
    n, s, _ = x.shape

    def tick(t0, name):
        if times is not None:
            times[name] = times.get(name, 0.0) + (time.perf_counter() - t0)

    t = time.perf_counter()
    q, k = model.wq(x), model.wk(x)
    tick(t, "proj q/k")

    t = time.perf_counter()
    a = model._operator(q, k)
    tick(t, "operator build")

    t = time.perf_counter()
    hop2 = batched_pivot_hop2(a, batched_select_pivots(k, model.k_pivots))
    tick(t, "hop-2 select+bmm")

    t = time.perf_counter()
    z = x + a @ x
    z = z + hop2 @ x
    tick(t, "a@x + hop2@x")

    t = time.perf_counter()
    h = model.mlp(z)
    tick(t, "MLP")

    t = time.perf_counter()
    out = model.readout(h).squeeze(-1)
    out = out[:, s - 1]
    tick(t, "readout+loss")
    return out


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20
    except Exception:
        return float("nan")


def avail_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 20
    except Exception:
        return float("nan")


def one_step(model, opt, x, y_std, times: dict | None):
    model.train()
    opt.zero_grad()
    out = staged_forward(model, x, times)
    loss = torch.nn.functional.mse_loss(out, y_std)

    t = time.perf_counter()
    loss.backward()
    bwd = time.perf_counter() - t

    t = time.perf_counter()
    opt.step()
    step = time.perf_counter() - t
    if times is not None:
        times["backward"] = times.get("backward", 0.0) + bwd
        times["optimiser"] = times.get("optimiser", 0.0) + step
    return float(loss)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--arm", default="pivot_signed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()

    print(f"n={a.n} s={a.s} d={a.d} arm={a.arm} seed={a.seed} "
          f"repeats={a.repeats} threads={torch.get_num_threads()} "
          f"torch={torch.__version__}")
    print(f"RSS start {rss_mb():.0f} MB   system available {avail_mb():.0f} MB")

    x, y, _, _ = make_batch(a.n, a.s, a.d, d_model=D_MODEL, seed=a.seed)
    torch.manual_seed(a.seed)
    model = Arm(a.arm, a.s)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mu, sigma = float(y.mean()), float(y.std(unbiased=False)) or 1.0
    y_std = (y - mu) / sigma
    print(f"RSS after data {rss_mb():.0f} MB   system available {avail_mb():.0f} MB")

    # ---- BIND FIRST. A profile of a different function is not a profile. ----
    with torch.no_grad():
        ref = model(x)
        got = staged_forward(model, x)
    eq = torch.equal(ref, got)
    print(f"\nBIND  torch.equal(Arm.forward(x), staged_forward(x)) = {eq}   "
          f"maxdiff={float((ref - got).abs().max()):.6e}   n={a.n}")
    if not eq:
        print("ABORT: staged forward is not bitwise the shipped forward. "
              "Nothing is credited.")
        return 1
    del ref, got

    one_step(model, opt, x, y_std, None)          # warmup, discarded
    print(f"RSS after warmup step {rss_mb():.0f} MB   "
          f"system available {avail_mb():.0f} MB")

    per: dict[str, list[float]] = {}
    walls = []
    for r in range(a.repeats):
        t0 = time.perf_counter()
        times: dict[str, float] = {}
        one_step(model, opt, x, y_std, times)
        walls.append(time.perf_counter() - t0)
        for kk, v in times.items():
            per.setdefault(kk, []).append(v)

    order = list(STAGES) + ["backward", "optimiser"]
    med = {kk: statistics.median(v) for kk, v in per.items()}
    total = statistics.median(walls)
    print(f"\n=== SECONDS PER COMPONENT, median of {a.repeats} steps, n={a.n} ===")
    print(f"{'component':>20} {'median s':>12} {'% of step':>11} {'min s':>11} {'max s':>11}")
    for kk in order:
        v = per.get(kk, [float('nan')])
        print(f"{kk:>20} {med.get(kk, float('nan')):>12.4f} "
              f"{100 * med.get(kk, 0) / total:>10.1f}% {min(v):>11.4f} {max(v):>11.4f}")
    print(f"{'FULL STEP (wall)':>20} {total:>12.4f} {100.0:>10.1f}%"
          f" {min(walls):>11.4f} {max(walls):>11.4f}")
    print(f"{'sum of components':>20} {sum(med.values()):>12.4f}")
    print(f"RSS end {rss_mb():.0f} MB   system available {avail_mb():.0f} MB")

    if a.profile:
        print(f"\n=== torch.autograd.profiler, ONE step, self CPU time, n={a.n} ===")
        with torch.autograd.profiler.profile(record_shapes=False) as prof:
            one_step(model, opt, x, y_std, None)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=28))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
