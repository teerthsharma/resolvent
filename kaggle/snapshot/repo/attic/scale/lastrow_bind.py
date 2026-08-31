"""THE PROPOSED OPTIMISATION AND ITS BITWISE BIND, IN ONE FILE.

THE OBSERVATION. `Arm.forward` ends `return out[:, s - 1]`. Every other position
is built, back-propagated through, and thrown away. The MLP and the readout are
POSITION-WISE, so 63 of 64 positions of both are dead work; and because `z`'s only
surviving row is `s-1`, the only rows of the operator `a` that can reach the
output are row `s-1` and the `k` PIVOT rows that `hop2` composes through --
9 rows of 64 for the shipped `k_pivots = 8`.

TWO CANDIDATES, measured separately because they carry different risk:

  MLP-SLICE   z is built in full, then `readout(mlp(z[:, s-1]))`. The arithmetic
              per surviving row is unchanged; only the GEMM's m dimension changes
              from n*s to n.
  LAST-ROW    additionally builds only the rows of `a` that survive. This changes
              the m dimension of the operator's own matmul, and a BLAS may
              dispatch a small-m GEMM to a different micro-kernel with a
              different accumulation order.

NEITHER IS ADOPTED ON AN ARGUMENT. `torch.equal` -- BITWISE, never allclose --
against the shipped `Arm.forward`, forward AND gradient. A faster path that
changes a number is a NEW ARM, not an optimisation, and this repository has
published one of those before. If a candidate fails the bind, the max difference
is REPORTED, not tolerated away.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.m3_capability import Arm, D_MODEL
from scale.negation_scope import make_batch
from scale.pivot_probe import batched_select_pivots, batched_pivot_hop2

torch.set_num_threads(2)


def fwd_mlp_slice(model, x):
    """Candidate 1: full operator, MLP and readout at position s-1 only."""
    n, s, _ = x.shape
    q, k = model.wq(x), model.wk(x)
    a = model._operator(q, k)
    z = x + a @ x
    hop2 = batched_pivot_hop2(a, batched_select_pivots(k, model.k_pivots))
    z = z + hop2 @ x
    return model.readout(model.mlp(z[:, s - 1])).squeeze(-1)


def fwd_mlp_slice_contig(model, x):
    """Candidate 1b: the same slice, made CONTIGUOUS first.

    Candidate 1 failed the bind and the isolation run says why, in two parts:
    a bare `mm` changes bits when `m` drops from 4096 to 64 but NOT when it drops
    from 131072 to 2048, and -- separately -- a pure ELEMENTWISE `gelu` changes
    bits between a contiguous input and a strided one, while
    `gelu(z)[:, -1] == gelu(z[:, -1].contiguous())` is exactly equal at every n
    tested. So the residual difference at large n is not the GEMM at all; it is
    PyTorch taking a different vectorised path over a strided view. One
    `.contiguous()` removes that half.
    """
    n, s, _ = x.shape
    q, k = model.wq(x), model.wk(x)
    a = model._operator(q, k)
    z = x + a @ x
    hop2 = batched_pivot_hop2(a, batched_select_pivots(k, model.k_pivots))
    z = z + hop2 @ x
    return model.readout(model.mlp(z[:, s - 1].contiguous())).squeeze(-1)


def fwd_last_row(model, x):
    """Candidate 2: only the 1 + k rows of `a` that can reach the output.

    `hop2[:, s-1, :] = cols[:, s-1, :] @ rows`, and `cols[:, s-1, :]` is
    `a[:, s-1, pivots]` while `rows` is `a[:, pivots, :]`. Association order is
    preserved exactly -- `(cols_last @ rows) @ x`, never `cols_last @ (rows @ x)`
    -- because reassociating a float matmul is not an optimisation, it is a
    different number.
    """
    n, s, _ = x.shape
    q, k = model.wq(x), model.wk(x)
    piv = batched_select_pivots(k, model.k_pivots)                  # [n,kp]
    kp = piv.shape[-1]
    rows_idx = torch.cat([piv, torch.full((n, 1), s - 1, dtype=piv.dtype)], 1)
    qsub = torch.gather(q, 1, rows_idx[:, :, None].expand(n, kp + 1, q.shape[-1]))
    asub = model._operator_rows(qsub, k, rows_idx)                  # [n,kp+1,s]
    a_last = asub[:, kp:, :]                                        # [n,1,s]
    a_piv = asub[:, :kp, :]                                         # [n,kp,s]
    cols_last = torch.gather(a_last, 2, piv[:, None, :])            # [n,1,kp]
    zl = x[:, s - 1:] + a_last @ x + (cols_last @ a_piv) @ x        # [n,1,d]
    return model.readout(model.mlp(zl)).squeeze(-1).squeeze(-1)


def _operator_rows(self, qsub, k, rows_idx):
    """`self._operator(q, k)[batch, rows_idx, :]`, built from qsub only.

    Bolted on rather than edited into `m3_capability.Arm` because it is a
    CANDIDATE. Nothing in the shipped path calls it until its bind passes.
    """
    import math
    from ceq import bench
    n, s, _ = k.shape
    r = qsub.shape[1]
    # the causal mask restricted to the selected rows: j < i for each row i
    ar = torch.arange(s)
    m = ar[None, None, :] < rows_idx[:, :, None]                    # [n,r,s]
    w = (qsub @ k.transpose(-2, -1)) / math.sqrt(qsub.shape[-1])
    if self.kind in ("softmax", "pivot_unsigned"):
        return torch.softmax(w.masked_fill(~m, torch.finfo(w.dtype).min),
                             -1).masked_fill(~m, 0.0)
    neg = torch.finfo(w.dtype).min
    pp = torch.softmax(w.masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    pm = torch.softmax((-w).masked_fill(~m, neg), -1).masked_fill(~m, 0.0)
    from scale.m3_capability import SGATE_RHO, SGATE_LAM
    return SGATE_RHO * (pp - SGATE_LAM * pm) / (1.0 + SGATE_LAM)


Arm._operator_rows = _operator_rows


def grads(model, fn, x, y):
    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(fn(x), y)
    loss.backward()
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
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--arm", default="pivot_signed")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    print(f"arm={a.arm} s={a.s} d={a.d} threads={torch.get_num_threads()} "
          f"torch={torch.__version__}")

    print("\n=== BIND: torch.equal against the shipped Arm.forward ===")
    print(f"{'n':>7} {'candidate':>18} {'fwd equal':>10} {'fwd maxdiff':>14} "
          f"{'grad equal':>11} {'grad maxdiff':>14}")
    for n in (1, 8, 64, 512, a.n):
        x, y, _, _ = make_batch(n, a.s, a.d, d_model=D_MODEL, seed=0)
        torch.manual_seed(0)
        model = Arm(a.arm, a.s)
        yq = torch.zeros(n)
        ref = model(x)
        gref = grads(model, model.__call__, x, yq)
        for name, raw in (("mlp-slice", fwd_mlp_slice),
                          ("mlp-slice-contig", fwd_mlp_slice_contig),
                          ("last-row", fwd_last_row)):
            fn = (lambda xx, _r=raw: _r(model, xx))
            got = fn(x)
            gg = grads(model, fn, x, yq)
            fe = torch.equal(ref, got)
            fd = float((ref - got).abs().max())
            ge = all(torch.equal(p, q) for p, q in zip(gref, gg))
            gd = max(float((p - q).abs().max()) for p, q in zip(gref, gg))
            print(f"{n:>7} {name:>18} {str(fe):>10} {fd:>14.6e} "
                  f"{str(ge):>11} {gd:>14.6e}")

    print(f"\n=== SPEED, fwd+bwd, median of {a.reps} after 1 warmup ===")
    print(f"{'n':>7} {'shipped s':>11} {'mlp-contig s':>14} {'last-row s':>12} "
          f"{'contig x':>10} {'last x':>8}")
    for n in (2048, 8192):
        x, _, _, _ = make_batch(n, a.s, a.d, d_model=D_MODEL, seed=0)
        torch.manual_seed(0)
        model = Arm(a.arm, a.s)
        t0 = timeit(model.__call__, x, a.reps)
        t1 = timeit(lambda xx: fwd_mlp_slice_contig(model, xx), x, a.reps)
        t2 = timeit(lambda xx: fwd_last_row(model, xx), x, a.reps)
        print(f"{n:>7} {t0:>11.4f} {t1:>13.4f} {t2:>12.4f} "
              f"{t0/t1:>7.2f}x {t0/t2:>7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
