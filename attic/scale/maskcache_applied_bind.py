"""POST-APPLICATION bind for the `_causal_mask_pair` memoisation in ceq/bench.py.

`scale/maskcache_bind.py` bound the CANDIDATE -- a private `sgate_cached` against
the then-shipped `Arm.forward`. Once the candidate is applied, that file compares
the cache against itself and proves nothing. This one holds a FROZEN, verbatim
copy of the pre-change operators (git 810bf0c, `ceq/bench.py` lines 124-196) and
binds the shipped path against it.

WHAT IS FROZEN. The four `_ref_*` functions below are byte-for-byte the pre-change
bodies with `_causal_mask` renamed `_ref_mask`. Nothing else. They are copied in
rather than imported from a git revision so the bind still runs after the next
commit; a bind whose reference moves with HEAD is not a bind.

WHAT IS TESTED. Both levels, because they fail differently:
  * OPERATOR level -- the raw [n,s,s] matrix out of each operator, `torch.equal`.
  * ARM level -- `scale.m3_capability.Arm` forward AND every parameter gradient,
    `torch.equal`, over all four arms, with the shipped operator monkey-patched
    to the frozen reference to produce the control. Same module, same
    parameters, same batch; the operator is the only thing that differs.

`torch.equal`, never `allclose`. A faster path that changes a number is a new
arm, not an optimisation.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench
from scale.m3_capability import Arm, D_MODEL, SGATE_RHO, SGATE_LAM, W_WINDOW
from scale.negation_scope import make_batch

torch.set_num_threads(2)


# -- FROZEN pre-change reference, git 810bf0c ceq/bench.py:124-196 ------------

def _ref_mask(s: int, device, window: int = 0) -> torch.Tensor:
    m = torch.ones(s, s, dtype=torch.bool, device=device).tril(-1)
    return m if window <= 0 else m.triu(-window)


def _ref_signed(q, k, rho: float = 0.9, window: int = 0):
    m = _ref_mask(q.shape[-2], q.device, window)
    w = ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])).masked_fill(~m, 0.0)
    return rho * w / w.abs().sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)


def _ref_softmax(q, k, window: int = 0):
    m = _ref_mask(q.shape[-2], q.device, window)
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(w.masked_fill(~m, torch.finfo(q.dtype).min), -1).masked_fill(~m, 0.0)


def _ref_sgate(q, k, rho: float = 1.5, lam: float = 0.10, window: int = 0):
    m = _ref_mask(q.shape[-2], q.device, window)
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    neg = torch.finfo(q.dtype).min
    pp = torch.softmax(w.masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    pm = torch.softmax((-w).masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    return rho * (pp - lam * pm) / (1.0 + lam)


def _ref_signmag(q, k, rho: float = 1.5, window: int = 0):
    m = _ref_mask(q.shape[-2], q.device, window)
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mag = torch.softmax(w.abs().masked_fill(~m, torch.finfo(q.dtype).min), -1)
    return rho * torch.sign(w) * mag.masked_fill(~m, 0.0)


# -- the bind -----------------------------------------------------------------

def grads(model, fn, x, y):
    model.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(fn(x), y).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def operator_level(s: int, d_model: int) -> bool:
    print("\n=== OPERATOR LEVEL: shipped vs frozen pre-change, torch.equal ===")
    print(f"{'op':>16} {'window':>7} {'n':>7} {'equal':>7} {'maxdiff':>14}")
    ok = True
    cases = [("softmax", bench._softmax_operator, _ref_softmax, {}),
             ("sgate", bench._causal_sgate_operator, _ref_sgate,
              dict(rho=SGATE_RHO, lam=SGATE_LAM)),
             ("signed", bench._causal_signed_operator, _ref_signed, {}),
             ("signmag", bench._causal_signmag_operator, _ref_signmag, {})]
    for name, new, ref, kw in cases:
        for window in (0, W_WINDOW):
            for n in (1, 64, 2048):
                g = torch.Generator().manual_seed(0)
                q = torch.randn(n, s, d_model, generator=g)
                k = torch.randn(n, s, d_model, generator=g)
                a, b = new(q, k, window=window, **kw), ref(q, k, window=window, **kw)
                eq = torch.equal(a, b)
                ok = ok and eq
                print(f"{name:>16} {window:>7} {n:>7} {str(eq):>7} "
                      f"{float((a - b).abs().max()):>14.6e}")
    return ok


def arm_level(s: int, d: int, ns) -> bool:
    print("\n=== ARM LEVEL: forward + every parameter gradient, torch.equal ===")
    print(f"{'arm':>16} {'n':>7} {'fwd equal':>10} {'fwd maxdiff':>14} "
          f"{'grad equal':>11} {'grad maxdiff':>14}")
    ok = True
    shipped_sm, shipped_sg = bench._softmax_operator, bench._causal_sgate_operator
    for arm in ("softmax", "pivot_unsigned", "pivot_signed", "windowed_signed"):
        for n in ns:
            x, _, _, _ = make_batch(n, s, d, d_model=D_MODEL, seed=0)
            torch.manual_seed(0)
            model = Arm(arm, s)
            y = torch.zeros(n)
            # control: the frozen pre-change operators, same module, same batch
            bench._softmax_operator, bench._causal_sgate_operator = _ref_softmax, _ref_sgate
            ref = model(x)
            gref = grads(model, model.__call__, x, y)
            # shipped: the memoised path
            bench._softmax_operator, bench._causal_sgate_operator = shipped_sm, shipped_sg
            got = model(x)
            ggot = grads(model, model.__call__, x, y)
            fe = torch.equal(ref, got)
            ge = all(torch.equal(p, q) for p, q in zip(gref, ggot))
            ok = ok and fe and ge
            print(f"{arm:>16} {n:>7} {str(fe):>10} "
                  f"{float((ref - got).abs().max()):>14.6e} {str(ge):>11} "
                  f"{max(float((p - q).abs().max()) for p, q in zip(gref, ggot)):>14.6e}")
            del x, ref, got, gref, ggot, model
    bench._softmax_operator, bench._causal_sgate_operator = shipped_sm, shipped_sg
    return ok


def speed(s: int, d: int, ns, reps: int) -> None:
    """The 1.019x, RE-MEASURED after application. Not part of the verdict.

    A/B inside ONE process so the two paths see the same allocator, the same
    thread pool and the same page cache: only the operator differs.
    """
    import statistics
    import time
    print(f"\n=== SPEED, fwd+bwd, median of {reps} after 1 warmup ===")
    print(f"{'arm':>16} {'n':>7} {'frozen s':>10} {'cached s':>10} {'speedup':>9}")
    shipped_sm, shipped_sg = bench._softmax_operator, bench._causal_sgate_operator

    def clock(model, x):
        ts = []
        for _ in range(reps + 1):
            t = time.perf_counter()
            out = model(x)
            torch.nn.functional.mse_loss(out, torch.zeros_like(out)).backward()
            ts.append(time.perf_counter() - t)
            model.zero_grad(set_to_none=True)
        return statistics.median(ts[1:])

    for arm in ("pivot_signed", "pivot_unsigned"):
        for n in ns:
            x, _, _, _ = make_batch(n, s, d, d_model=D_MODEL, seed=0)
            torch.manual_seed(0)
            model = Arm(arm, s)
            bench._softmax_operator, bench._causal_sgate_operator = _ref_softmax, _ref_sgate
            t0 = clock(model, x)
            bench._softmax_operator, bench._causal_sgate_operator = shipped_sm, shipped_sg
            t1 = clock(model, x)
            print(f"{arm:>16} {n:>7} {t0:>10.4f} {t1:>10.4f} {t0 / t1:>8.3f}x")
            del x, model
    bench._softmax_operator, bench._causal_sgate_operator = shipped_sm, shipped_sg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[1, 8, 64, 512, 2048, 8192])
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    print(f"s={a.s} d={a.d} threads={torch.get_num_threads()} "
          f"torch={torch.__version__} cache={bench._causal_mask_pair.cache_info()}")
    ok = operator_level(a.s, D_MODEL)
    ok = arm_level(a.s, a.d, a.ns) and ok
    if a.speed:
        speed(a.s, a.d, [n for n in a.ns if n >= 2048] or a.ns, a.reps)
    print(f"\ncache after: {bench._causal_mask_pair.cache_info()}")
    print(f"BIND VERDICT: "
          f"{'BITWISE everywhere tested' if ok else 'FAILED -- REVERT, the cache is a new arm'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
