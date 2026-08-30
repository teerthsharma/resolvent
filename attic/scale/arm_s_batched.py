"""ARM S settling, BATCHED over examples, bound against `scale/arm_s.py`.

WHY. `scale/arm_s.py` settles ONE `[s, d]` example per call. The M3 capability
harness (`scale/m3_capability.py`) trains on `[n, s, d]` with n up to 8192, and
`scale/pivot_probe.py:120-137` already measured what a Python loop over examples
costs there: the FORWARD is 4-11x slow, but the BACKWARD is superlinear
(0.0369 -> 0.6588 -> 24.5587 s at n = 128 -> 512 -> 2048) because the loop
builds one autograd subgraph per example. This file carries the batch axis on
tensor ops instead of on the interpreter.

WHAT IS BOUND, AND HOW. Every output is checked with `torch.equal` against the
per-example original over the same inputs -- never `allclose`. The bind is on
the MAP, not on pivot selection: `piv` is an argument on both sides, so both
sides settle the same pivot set and any difference is arithmetic.

WHERE THE BIND HOLDS AND WHERE IT DOES NOT. Measured, not claimed; the
self-check below prints the number.

  * `log_gram`, `log_gate`, `batched_log_alpha_step`, `batched_settle_log` are
    BITWISE identical at every shape and seed tested. The settling map itself --
    the whole of the k-dimensional iteration -- moves no bits.
  * `av` (= A_P @ V) is bitwise at `chunk = 1` and at every shape whose
    per-slice contraction `k * s` is at or below 2048, and diverges by one ULP
    otherwise. It is a `torch.bmm` / `torch.mm` dispatch split, not an algebra
    difference: at `torch.set_num_threads(1)` the two agree exactly, and at 2
    threads `mm` splits the s-contraction across threads while `bmm` does not.
    `scale/ceq/bench.py:133-137` already recorded this failure class -- changing
    a GEMM's shape changes which BLAS micro-kernel runs and therefore the
    accumulation order.
    `av` is NOT on the settling path. `arm_s.arm_s` uses it once, at the end,
    as `log_alpha.exp() @ av`; nothing in the fixed-point iteration reads it.

THE ONE DELIBERATE DIVERGENCE IN CONTROL FLOW. `batched_settle_log` has NO
early exit. The original returns as soon as d_H < tol, which is a per-example
stopping time and therefore a per-example number of kernel launches. A fixed
`max_steps` keeps the batch on one kernel path. The self-check compares against
`arm_s.settle_log(..., tol=0.0, ...)`, where `r < 0.0` is unreachable for a
non-negative residual, so the original also runs exactly `max_steps` steps and
the two are compared step for step.

MEMORY. The `[n, k, k, s]` logsumexp intermediate is never materialised whole:
the whole of `batched_log_pivot_context` runs inside one loop over batch chunks,
so the peak is `[chunk, k, k, s]`. At the default chunk = 64, k = 32, s = 64
that is 64*32*32*64 = 4,194,304 float64 = 33,554,432 bytes.

THREADS ARE PINNED IN THIS FILE, as in `scale/arm_s.py`.
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                       # pinned HERE, not by the launcher

from ceq import bench                                            # noqa: E402
from scale import arm_s                                          # noqa: E402


# ------------------------------------------------------------------ the arm

def batched_log_pivot_context(q: torch.Tensor, kk: torch.Tensor,
                              v: torch.Tensor, piv: torch.Tensor, *,
                              chunk: int = 64):
    """`arm_s.log_pivot_context(q[i], kk[i], v[i], piv[i])` for every i.

    q, kk, v are `[n, s, d]`; piv is `[n, k]`. Returns
    `(log_gram [n,k,k], log_gate [n,k], av [n,k,d])`, float64 exactly where the
    original is float64: the gather off `q` happens in q's own dtype and the
    cast to double lands in the same place as `scale/arm_s.py:155`.

    `chunk` is over the BATCH, not over the pivot rows. The original chunks over
    p because its `[k,k,s]` intermediate is the only thing that grows; here the
    batch axis multiplies every intermediate, so the whole body -- logits, mask,
    log_softmax, the Gram logsumexp, and A_P @ V -- sits inside the batch loop
    and the peak is `[chunk, k, k, s]` rather than `[n, k, k, s]`.

    `chunk` is a memory knob for `log_gram`, `log_gate` and the logit block: all
    three are bitwise invariant in it. It is NOT a free knob for `av`, which is
    a bmm whose batch extent is the chunk -- see the module docstring and the
    THREAD DIAGNOSIS block in `main`.
    """
    n, s, d = q.shape[0], q.shape[1], q.shape[-1]
    k = int(piv.shape[-1])
    _, nm = bench._causal_mask_pair(s, 0, str(q.device))
    last = torch.full((n, 1), s - 1, dtype=piv.dtype, device=piv.device)
    want = torch.cat([piv, last], dim=1)                          # [n, k+1]

    log_gram = torch.empty(n, k, k, dtype=torch.float64)
    log_gate = torch.empty(n, k, dtype=torch.float64)
    av = torch.empty(n, k, d, dtype=torch.float64)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        wc, pc = want[lo:hi], piv[lo:hi]
        m = hi - lo
        qs = torch.gather(q[lo:hi], 1, wc[:, :, None].expand(m, k + 1, d))
        w = (qs.double() @ kk[lo:hi].double().transpose(-2, -1)) / math.sqrt(d)
        w = w.masked_fill(nm[wc], float("-inf"))
        la = torch.log_softmax(w[:, :-1], dim=-1)                 # [m, k, s]
        log_gate[lo:hi] = torch.gather(
            torch.log_softmax(w[:, -1], dim=-1), 1, pc)
        log_gram[lo:hi] = torch.logsumexp(la.unsqueeze(2) + la.unsqueeze(1),
                                          dim=-1)
        av[lo:hi] = la.exp() @ v[lo:hi].double()
    return log_gram, log_gate, av


def batched_log_alpha_step(log_alpha: torch.Tensor, log_gate: torch.Tensor,
                           log_gram: torch.Tensor, beta: float) -> torch.Tensor:
    """`arm_s.log_alpha_step` with the batch axis carried by broadcasting.

    The original's `log_alpha.unsqueeze(0)` puts the alpha index on the COLUMN
    of the `[k,k]` broadcast, so the batched form unsqueezes at 1, not at 0 --
    the batch axis has taken dim 0 over.
    """
    lw = log_gate + beta * torch.logsumexp(
        log_gram + log_alpha.unsqueeze(1), dim=-1)
    return lw - torch.logsumexp(lw, dim=-1, keepdim=True)


def batched_settle_log(log_gate: torch.Tensor, log_gram: torch.Tensor,
                       beta: float, max_steps: int) -> torch.Tensor:
    """`arm_s.settle_log` for a whole batch, with NO early exit. -> log_alpha.

    Fixed `max_steps` on purpose: every example does the same work, so the loop
    is one kernel path and not n stopping times. The residual is not read here
    -- `arm_s.settle_log` computes d_H only to decide when to stop, and it never
    feeds back into log_alpha, so dropping it cannot move a bit.
    """
    log_alpha = log_gate - torch.logsumexp(log_gate, dim=-1, keepdim=True)
    for _ in range(max_steps):
        log_alpha = batched_log_alpha_step(log_alpha, log_gate, log_gram, beta)
    return log_alpha


# ---------------------------------------------------------- the self-check

def stacked_pivots_of(kk: torch.Tensor, k_piv: int):
    """`arm_s.pivots_of` per example, stacked to `[n, k]`. -> (piv, counts).

    NOT `pivot_probe.batched_select_pivots`. That function's own docstring
    (`scale/pivot_probe.py:111-114`) declines the call shape this needs:

        NO `exclude`. `select_pivots`'s `exclude` sets entries to -inf and then
        counts survivors, which is per-example bookkeeping this does not
        replicate; the batched form is not offered for that call shape rather
        than silently ignoring the argument.

    and `arm_s.pivots_of` (`scale/arm_s.py:97-108`) is exactly that call shape:

        s = kk.shape[0]
        piv = select_pivots(kk, min(k_piv, s - 2), exclude=(s - 1,))
        return piv[piv > 0]

    Two divergences from `batched_select_pivots(key, k)`, either one fatal:
      1. `exclude=(s - 1,)` -- the query row is struck from the score before
         topk. `batched_select_pivots` takes no `exclude` at all.
      2. `piv[piv > 0]` -- index 0 is dropped AFTER topk, so the result is
         RAGGED: k or k-1 entries depending on whether row 0's key-norm made the
         top-k. `batched_select_pivots` always returns exactly `min(k, s)`.

    Divergence 2 also blocks a rectangular `[n, k]` stack outright, so this
    truncates every example to the batch-minimum count and reports the counts.
    Truncation drops the LOWEST-scoring pivots (topk returns sorted), and the
    SAME tensor is handed to both sides of the bind, so it cannot flatter the
    comparison -- it only narrows which pivot set is bound.
    """
    per = [arm_s.pivots_of(kk[i], k_piv) for i in range(kk.shape[0])]
    counts = [int(p.numel()) for p in per]
    kmin = min(counts)
    return torch.stack([p[:kmin] for p in per]), counts


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    dif = (a - b).abs().max()
    return float(dif) if torch.isfinite(dif) else float("nan")


# `av` is reported, not asserted: see the module docstring. Everything on the
# settling path IS asserted, so a regression in the map still fails the run.
SETTLING_PATH = ("log_gram", "log_gate", "alpha_step", "settle")


def main() -> int:
    th = torch.get_num_threads()
    beta, steps, chunk = 0.5, 12, 64
    shapes = [(4, 64, 24, 8), (3, 64, 24, 16), (2, 128, 16, 32)]
    print(f"=== arm_s_batched BITWISE BIND vs scale/arm_s.py. threads={th} ===")
    print(f"  beta={beta}  max_steps={steps}  chunk={chunk}  input dtype float32")
    print("  PIVOTS: stacked_pivots_of -- a LOOP over arm_s.pivots_of, NOT")
    print("    pivot_probe.batched_select_pivots. pivots_of passes")
    print("    exclude=(s-1,) and post-filters piv[piv > 0]; batched_select_pivots")
    print("    takes no exclude and never drops index 0, and its own docstring")
    print("    declines that call shape. See stacked_pivots_of.__doc__.")
    print("  BASELINE settle_log is called with tol=0.0 so its `r < tol` early")
    print("    exit is unreachable and it runs the same 12 steps as the batch.")
    print()
    verdicts = []
    for n, s, d, k_piv in shapes:
        for seed in (0, 1):
            g = torch.Generator().manual_seed(seed)
            q = torch.randn(n, s, d, generator=g)
            kk = torch.randn(n, s, d, generator=g)
            v = torch.randn(n, s, d, generator=g)
            piv, counts = stacked_pivots_of(kk, k_piv)
            k = int(piv.shape[-1])

            bG, bg, bav = batched_log_pivot_context(q, kk, v, piv, chunk=chunk)
            ba1 = batched_log_alpha_step(
                bg - torch.logsumexp(bg, dim=-1, keepdim=True), bg, bG, beta)
            bla = batched_settle_log(bg, bG, beta, steps)

            eG, eg, eav = (torch.empty_like(bG), torch.empty_like(bg),
                           torch.empty_like(bav))
            ea1, ela = torch.empty_like(ba1), torch.empty_like(bla)
            for i in range(n):
                G, gt, a = arm_s.log_pivot_context(q[i], kk[i], v[i], piv[i])
                eG[i], eg[i], eav[i] = G, gt, a
                l0 = gt - torch.logsumexp(gt, dim=0)
                ea1[i] = arm_s.log_alpha_step(l0, gt, G, beta)
                ela[i] = arm_s.settle_log(gt, G, beta, tol=0.0,
                                          max_steps=steps)[0]

            named = [("log_gram", bG, eG), ("log_gate", bg, eg),
                     ("av", bav, eav), ("alpha_step", ba1, ea1),
                     ("settle", bla, ela)]
            tag = f"n={n} s={s} d={d} k_piv={k_piv} seed={seed}"
            print(f"  {tag}")
            print(f"    arm_s.pivots_of counts per example = {counts}"
                  f"   -> stacked k = {k}   k*s = {k * s}")
            for nm, b, e in named:
                print(f"    {nm:>11}  shape {tuple(b.shape)!s:>14}  "
                      f"torch.equal = {torch.equal(b, e)!s:>5}  "
                      f"max abs diff = {_maxdiff(b, e):.17e}")
            broke = [(nm, _maxdiff(b, e)) for nm, b, e in named
                     if not torch.equal(b, e)]
            if broke:
                print("    VERDICT: " + "  ".join(
                    f"{nm}: BITWISE: NO, max abs diff = {dv!r}"
                    for nm, dv in broke))
            else:
                print("    VERDICT: BITWISE: YES (all five outputs)")
            print()
            verdicts.append((tag, broke))
            for nm, b, e in named:
                if nm in SETTLING_PATH:
                    assert torch.equal(b, e), (
                        f"{tag} {nm}: BITWISE: NO, max abs diff = "
                        f"{_maxdiff(b, e)!r}")

    # ---- MUST-FIRE. Perturbing one pivot index changes the pivot SET, so an
    # implementation that ignored `piv` would still pass every assert above.
    g = torch.Generator().manual_seed(7)
    q = torch.randn(2, 128, 16, generator=g)
    kk = torch.randn(2, 128, 16, generator=g)
    v = torch.randn(2, 128, 16, generator=g)
    piv, _ = stacked_pivots_of(kk, 32)
    bad = piv.clone()
    bad[0, 0] = (int(bad[0, 0]) % 125) + 1
    fired = not torch.equal(
        batched_log_pivot_context(q, kk, v, piv, chunk=chunk)[0],
        batched_log_pivot_context(q, kk, v, bad, chunk=chunk)[0])
    print(f"  MUST-FIRE control (one pivot index moved): log_gram differs = {fired}")
    assert fired, "control did not fire: log_gram is not reading piv"

    # ---- CHUNK INVARIANCE, per output, at the shape where av breaks.
    c1 = batched_log_pivot_context(q, kk, v, piv, chunk=1)
    cN = batched_log_pivot_context(q, kk, v, piv, chunk=4096)
    print("  chunk=1 vs chunk=4096 at n=2 s=128 k=32:")
    for nm, a, b in zip(("log_gram", "log_gate", "av"), c1, cN):
        print(f"    {nm:>11}  torch.equal = {torch.equal(a, b)!s:>5}  "
              f"max abs diff = {_maxdiff(a, b):.17e}")
    for nm, a, b in zip(("log_gram", "log_gate", "av"), c1, cN):
        if nm != "av":
            assert torch.equal(a, b), f"chunk moved bits in {nm}"

    # ---- THREAD DIAGNOSIS for av. Same slices, same algebra, two dispatches.
    print("  THREAD DIAGNOSIS -- av is A_P @ V; bmm(batch) vs loop of mm:")
    ga = torch.Generator().manual_seed(11)
    A = torch.randn(4, 32, 128, generator=ga, dtype=torch.float64).exp()
    B = torch.randn(4, 128, 16, generator=ga, dtype=torch.float64)
    loop = torch.stack([A[i] @ B[i] for i in range(4)])
    print(f"    threads=2  bmm == loop-of-mm: {torch.equal(A @ B, loop)}  "
          f"max abs diff = {_maxdiff(A @ B, loop):.17e}")
    torch.set_num_threads(1)
    one = torch.stack([A[i] @ B[i] for i in range(4)])
    print(f"    threads=1  bmm == loop-of-mm: {torch.equal(A @ B, one)}  "
          f"max abs diff = {_maxdiff(A @ B, one):.17e}")
    torch.set_num_threads(2)

    print()
    print("  SUMMARY")
    for tag, broke in verdicts:
        line = ("BITWISE: YES" if not broke else "  ".join(
            f"{nm}: BITWISE: NO, max abs diff = {dv!r}" for nm, dv in broke))
        print(f"    {tag:<40} {line}")
    print("  ASSERTED BITWISE (and passed): " + ", ".join(SETTLING_PATH))
    print("  REPORTED, NOT ASSERTED: av -- see SUMMARY above and module docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
