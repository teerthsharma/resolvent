"""CANDIDATE 3, after candidates 1 and 2 both FAILED the bitwise bind.

WHAT KILLED THE FIRST TWO. `scale/lastrow_bind.py` measured a 1.4x (MLP-slice)
and an 8.0x (last-row) speedup and BOTH changed the number: fwd maxdiff
7.45e-09, grad maxdiff 1.49e-08. The cause is not a reassociation anywhere in the
Python -- it is that both candidates change a GEMM's `m` dimension (n*s -> n for
the MLP, s -> 1+k for the operator), and this BLAS selects a different
micro-kernel at a different `m`, with a different accumulation order inside the
k-loop. **On this platform, any optimisation that changes a matmul's shape
changes the bits.** That rules out the whole "slice earlier / build fewer rows"
family as OPTIMISATIONS; they are new arms.

WHAT IS LEFT IS SHAPE-PRESERVING WORK. Candidate 3 changes no tensor shape and no
arithmetic op -- it removes recomputation only:

  * `_causal_mask` allocates `ones(s,s,bool).tril(-1)` on EVERY operator call.
  * `_causal_sgate_operator` then evaluates `~m` FOUR times per call, and
    `_softmax_operator` twice.

Both are pure functions of `(s, window, device)`. Caching them cannot move a bit,
which is exactly why it is the candidate that can be bound.

HONEST EXPECTATION, stated BEFORE the run so the reading cannot drift: the mask
is [s,s] = 4096 entries against an [n,s,s] operator of 8.4M entries at n=2048, so
this is a small-constant win and the point of measuring it is to find out whether
it is 0.2% or 5%, not to hope for 8x.
"""
from __future__ import annotations

import argparse
import functools
import math
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.m3_capability import Arm, D_MODEL, SGATE_RHO, SGATE_LAM
from scale.negation_scope import make_batch
from scale.pivot_probe import batched_select_pivots, batched_pivot_hop2

torch.set_num_threads(2)


@functools.lru_cache(maxsize=32)
def _mask_pair(s: int, window: int, device: str):
    """(m, ~m). Pure function of the arguments -- caching cannot move a bit."""
    m = torch.ones(s, s, dtype=torch.bool, device=device).tril(-1)
    m = m if window <= 0 else m.triu(-window)
    return m, ~m


def sgate_cached(q, k, rho=SGATE_RHO, lam=SGATE_LAM, window=0):
    m, nm = _mask_pair(q.shape[-2], window, str(q.device))
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    neg = torch.finfo(q.dtype).min
    pp = torch.softmax(w.masked_fill(nm, neg), -1).masked_fill(nm, 0.0)
    pm = torch.softmax((-w).masked_fill(nm, neg), -1).masked_fill(nm, 0.0)
    return rho * (pp - lam * pm) / (1.0 + lam)


def fwd_maskcache(model, x):
    n, s, _ = x.shape
    q, k = model.wq(x), model.wk(x)
    a = sgate_cached(q, k)
    z = x + a @ x
    hop2 = batched_pivot_hop2(a, batched_select_pivots(k, model.k_pivots))
    z = z + hop2 @ x
    return model.readout(model.mlp(z)).squeeze(-1)[:, s - 1]


def grads(model, fn, x, y):
    model.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(fn(x), y).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def timeit(fn, x, reps):
    ts = []
    for _ in range(reps + 1):
        t = time.perf_counter()
        out = fn(x)
        torch.nn.functional.mse_loss(out, torch.zeros_like(out)).backward()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--arm", default="pivot_signed")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    print(f"arm={a.arm} s={a.s} d={a.d} threads={torch.get_num_threads()} "
          f"torch={torch.__version__}")

    print("\n=== BIND: torch.equal against the shipped Arm.forward ===")
    print(f"{'n':>7} {'fwd equal':>10} {'fwd maxdiff':>14} {'grad equal':>11} "
          f"{'grad maxdiff':>14}")
    ok = True
    for n in (1, 8, 64, 512, 2048, 8192):
        x, _, _, _ = make_batch(n, a.s, a.d, d_model=D_MODEL, seed=0)
        torch.manual_seed(0)
        model = Arm(a.arm, a.s)
        yq = torch.zeros(n)
        ref = model(x)
        gref = grads(model, model.__call__, x, yq)
        got = fwd_maskcache(model, x)
        gg = grads(model, lambda xx: fwd_maskcache(model, xx), x, yq)
        fe = torch.equal(ref, got)
        ge = all(torch.equal(p, q) for p, q in zip(gref, gg))
        ok = ok and fe and ge
        print(f"{n:>7} {str(fe):>10} {float((ref-got).abs().max()):>14.6e} "
              f"{str(ge):>11} "
              f"{max(float((p-q).abs().max()) for p, q in zip(gref, gg)):>14.6e}")
    print(f"BIND VERDICT: {'BITWISE at every n tested' if ok else 'FAILED -- NEW ARM, not an optimisation'}")

    print(f"\n=== SPEED, fwd+bwd, median of {a.reps} after 1 warmup ===")
    print(f"{'n':>7} {'shipped s':>11} {'cached s':>11} {'speedup':>9} {'saved s':>10}")
    for n in (2048, 8192):
        x, _, _, _ = make_batch(n, a.s, a.d, d_model=D_MODEL, seed=0)
        torch.manual_seed(0)
        model = Arm(a.arm, a.s)
        t0 = timeit(model.__call__, x, a.reps)
        t1 = timeit(lambda xx: fwd_maskcache(model, xx), x, a.reps)
        print(f"{n:>7} {t0:>11.4f} {t1:>11.4f} {t0/t1:>8.3f}x {t0-t1:>10.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
