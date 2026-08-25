"""Batched pivot hop-2, without the Python loop over the batch.

`scale/m3_capability.py::Arm.forward` computes the pivot arms' hop-2 term as

    hop2 = torch.stack([pivot_hop2(a[i], select_pivots(k[i], self.k_pivots))
                        for i in range(n)], dim=0)

which is `n` Python iterations, `n` topk launches and `n` matmul launches per
forward. At n_train=8192 that is 8192 iterations per forward, x150 steps,
x(forward+backward).

This module is the same arithmetic with the batch axis carried by the tensor
ops instead of by the interpreter. It is BOUND BITWISE against the loop in
`test_hop2_vec.py` -- not `allclose`. A faster path that changes a number is a
new arm, not an optimisation.

WHERE THIS BELONGS. `batched_pivot_hop2` sits next to `pivot_hop2` in
`scale/pivot_probe.py` once the measurement now running releases that file; it
lives here only because `scale/` is read-only while `m3_capability.py` is being
measured. The test imports it from here; after the diff lands the import moves
to `from scale.pivot_probe import batched_pivot_hop2` and this file is deleted.

SCOPE. `batched_select_pivots` covers the ONE call shape `m3_capability.py`
uses: `select_pivots(key, k)` with no `exclude`. `select_pivots`'s `exclude`
argument sets entries to -inf and then counts the survivors; nothing in the
batched path replicates that, and passing `exclude` here is not supported
rather than silently ignored.
"""
from __future__ import annotations

import torch


def batched_select_pivots(key: torch.Tensor, k: int) -> torch.Tensor:
    """`select_pivots(key[i], k)` for every i, as one topk. key [n,s,d] -> [n,k'].

    k' is `min(k, s)` -- the loop's `min(k, int((score > -inf).sum()))` with no
    exclusions, where the count is exactly `s`. It is NOT `min(k, n*s)`: summing
    the batched score matrix would give that, and would differ from the loop
    whenever k > s.
    """
    score = key.norm(dim=-1)                                   # [n,s]
    return torch.topk(score, min(k, score.shape[-1]), dim=-1).indices


def batched_pivot_hop2(a: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    """`pivot_hop2(a[i], pivots[i])` for every i. a [n,s,s], pivots [n,k] -> [n,s,s].

    `pivot_hop2` is `a[:, P] @ a[P, :]`, i.e. gather k COLUMNS then k ROWS then
    one [s,k]@[k,s] matmul. Batched, the columns are a gather along dim 2 and
    the rows a gather along dim 1, and the n matmuls become one `bmm`.
    """
    n, s, _ = a.shape
    kp = pivots.shape[-1]
    cols = torch.gather(a, 2, pivots[:, None, :].expand(n, s, kp))   # [n,s,k]
    rows = torch.gather(a, 1, pivots[:, :, None].expand(n, kp, s))   # [n,k,s]
    return torch.bmm(cols, rows)
