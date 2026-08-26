"""Exact rectangular assignment for a Monge cost on a line, as an oracle.

The +3/-5 gate that decides this round's shape runs a RECTANGULAR assignment: a
few dozen causal tokens matched into a filler pool of thousands, by key norm. The
existing exact check covers only the square case, so the assignment that gate
depends on has had no independent verification. That is the shape of defect this
project keeps paying for, and it is closed here rather than after the gate reads.

THE STRUCTURE. The cost C[i][j] = |a_i - b_j| on a line is a Monge matrix once
both sides are sorted, and a Monge cost always admits an optimal assignment that
is MONOTONE: if causal token i is served before token i', then the filler chosen
for i comes before the one chosen for i'. Crossing a pair can never help, because
for sorted a_i <= a_i' and b_j <= b_j' the uncrossed sum |a_i - b_j| +
|a_i' - b_j'| never exceeds the crossed one.

That reduces the assignment to choosing an increasing subsequence of the pool,
which is an exact dynamic program:

    f[i][j] = min( f[i][j-1],                    skip pool element j
                   f[i-1][j-1] + |a_i - b_j| )   use it for causal token i

in O(n*m) time and O(m) space. It shares no code and no algorithm with a
Hungarian or auction solver, so agreement between them is a genuine second path
rather than one method run twice.

WHAT THIS IS FOR. It is an ORACLE, not a replacement. The production matcher
should keep using a general solver; this exists so that solver can be checked on
the rectangular instances the gate actually uses, and so a no-scipy environment
still has an exact path for this particular cost.
"""
from __future__ import annotations

import math

import torch

__all__ = ["monge_assign", "monge_cost"]


def _sorted(a: torch.Tensor, b: torch.Tensor):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    if a.numel() > b.numel():
        raise ValueError(f"pool too small: {b.numel()} candidates for "
                         f"{a.numel()} items")
    ia = torch.argsort(a)
    ib = torch.argsort(b)
    return a[ia], b[ib], ia, ib


def monge_cost(a: torch.Tensor, b: torch.Tensor) -> float:
    """Minimum total |a_i - b_j| over injective assignments. Exact."""
    sa, sb, _, _ = _sorted(a, b)
    n, m = sa.numel(), sb.numel()
    if n == 0:
        return 0.0
    prev = [math.inf] * (m + 1)          # f[i-1][*], starting at i = 1
    cur = [0.0] * (m + 1)                # f[0][j] = 0 for every j
    for i in range(1, n + 1):
        prev, cur = cur, prev
        cur[0] = math.inf
        for j in range(1, m + 1):
            take = prev[j - 1] + abs(float(sa[i - 1]) - float(sb[j - 1]))
            cur[j] = take if take < cur[j - 1] else cur[j - 1]
    return cur[m]


def monge_assign(a: torch.Tensor, b: torch.Tensor) -> list[int]:
    """Indices into `b`, one per element of `a`, in the ORIGINAL order of `a`.

    Returns the argmin of the same objective `monge_cost` reports, so the two
    describe one instrument rather than two. A solver whose cost and assignment
    disagree has been a real defect in this project more than once.
    """
    sa, sb, ia, ib = _sorted(a, b)
    n, m = sa.numel(), sb.numel()
    if n == 0:
        return []
    # Full table, so the choice at each cell can be walked back. O(n*m) memory —
    # acceptable at the gate's shape (tens by thousands) and stated rather than
    # silently assumed: at n*m beyond ~1e8 this needs the Hirschberg treatment.
    f = [[math.inf] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        f[0][j] = 0.0
    for i in range(1, n + 1):
        ai = float(sa[i - 1])
        row, prev = f[i], f[i - 1]
        for j in range(1, m + 1):
            take = prev[j - 1] + abs(ai - float(sb[j - 1]))
            row[j] = take if take < row[j - 1] else row[j - 1]

    out = [0] * n
    i, j = n, m
    while i > 0:
        # Walk back: if skipping j was optimal the value is unchanged from j-1.
        if f[i][j] == f[i][j - 1]:
            j -= 1
            continue
        out[int(ia[i - 1])] = int(ib[j - 1])
        i -= 1
        j -= 1
    return out
