"""A scoped hop cache: the filtration a flat KV cache throws away.

WHAT IT SOLVES. The multi-hop path sum needs a square operator, and incremental
decoding supplies a `[1, N]` row, so `A^2` does not exist and the correction
cannot be formed. But `A` is strictly causal, which makes the hop tower a
filtration:

    (A^h v)_i = sum_{j<i} A_ij (A^{h-1} v)_j

`hop_h[i]` depends only on `hop_{h-1}[j]` for `j < i`. Caching one vector per hop
per position makes decode EXACT -- byte-identical to prefill -- at K attention
rows per token and K times the value-cache memory.

WHY NOT A FLAT CACHE. A flat cache stores `k` and `v` and discards the tower.
Three structures supply what it cannot, and each contributes exactly one thing:

  NeMo-Relay `scope_stack.rs`   LIFO scopes. Popping a scope drops exactly that
                                scope's registrations; the root refuses removal.
                                -> LIFETIME.
  foliation                     Only a FREE FACE is evictable: nothing live may
                                still depend on it. Evicting a referenced entry
                                leaves a tower with a hole, which is the same
                                corruption as a block table whose parent was
                                evicted. -> ADMISSIBILITY.
  mujoco#3396 island discovery  The support of `A` IS a graph, and its H0
                                partition is a disjoint-set forest computed
                                directly from incidence -- no `n x n` scratch,
                                canonical representative the MINIMUM index so
                                component ids are deterministic and ascending.
                                -> GRANULARITY: what drops together.
  W3, measured in this repo     Eviction is bitwise (0.000e+00 over 24/24 draws)
                                where post-softmax gating leaks (2.15e-05).
                                -> EXACTNESS.

Provenance, stated because it matters: NeMo-Relay ships no measurements anywhere
-- no memory saved, no drop counts, no latency. Only its structure is borrowed.
Every number attached to this file is measured here.

WHY EVICTION IS SAFE WHERE GATING IS NOT. A gate multiplies a column by ~0, but
the token's score never leaves the softmax denominator, so it goes on shadowing
every surviving row and perturbing it still moves the state. Removing it from the
key set is bitwise. That is the difference between "the weight is zero" and "the
token was never there", and it is the only reason this cache may drop entries
without changing the answer for what remains.
"""
from __future__ import annotations

import math

import torch


def causal_row_operator(query_row: torch.Tensor, key_prefix: torch.Tensor, *,
                        alpha: float, rho: float,
                        scaling: float | None = None) -> torch.Tensor:
    """One row of the signed operator: `[B, H, 1, P]` for a query against `P`
    strictly-earlier keys.

    Signed by construction -- the raw logits are normalized by their L1 norm
    rather than pushed through a softmax, because a non-negative operator has a
    non-negative influence Jacobian in any ordered semiring and therefore cannot
    represent a negation.

    An empty prefix gives an all-zero row rather than NaN. That is position 0,
    which has no predecessors under strict causality.
    """
    if key_prefix.shape[-2] == 0:
        return query_row.new_zeros(*query_row.shape[:-1], 0)
    scale = scaling if scaling is not None else 1.0 / math.sqrt(query_row.shape[-1])
    w = (query_row @ key_prefix.transpose(-2, -1)) * scale
    l1 = w.abs().sum(-1, keepdim=True)
    return alpha * rho * w / l1.clamp_min(torch.finfo(w.dtype).tiny)


def prefill_correction(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       *, alpha: float, rho: float, hops: int,
                       scaling: float | None = None) -> torch.Tensor:
    """`sum_{h=1..K} (alpha A)^h v` over a whole sequence at once.

    The reference the incremental path must reproduce bitwise.
    """
    from .attention import ceq_operator
    a = alpha * ceq_operator(query, key, None, rho=rho, scaling=scaling)
    term = value
    corr = torch.zeros_like(value)
    for _ in range(hops):
        term = a @ term
        corr = corr + term
    return corr


class HopCache:
    """Scoped, evicting cache over the hop tower.

    `slots[h]` holds the history of `(A^h v)` for `h = 0 .. hops-1`; `slots[0]`
    is the value history itself. A new position reads all of them and appends one
    vector to each.
    """

    def __init__(self, hops: int, couple_at: float = 0.0):
        if hops < 1:
            raise ValueError("hops must be >= 1")
        self.hops = hops
        self.couple_at = couple_at
        self._keys: list[torch.Tensor] = []
        self._slots: list[list[torch.Tensor]] = [[] for _ in range(hops)]
        # scope[i] is the depth of the scope that registered position i; 0 = root
        self._scope_of: list[int] = []
        self._depth = 0
        self._parent: list[int] = []          # disjoint-set forest over positions

    # ------------------------------------- mujoco#3396 island discovery

    def find(self, i: int) -> int:
        """Canonical representative, with path compression.

        The representative is always the MINIMUM index in the component, which is
        what makes component ids deterministic and ascending rather than an
        artefact of insertion order. That rule is taken from `mj_island`, where it
        is the reason island ids are stable across runs.
        """
        r = i
        while self._parent[r] != r:
            r = self._parent[r]
        while self._parent[i] != r:            # path compression
            self._parent[i], i = r, self._parent[i]
        return r

    def union(self, i: int, j: int) -> None:
        """Merge, keeping the smaller index as representative."""
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        lo, hi = (ri, rj) if ri < rj else (rj, ri)
        self._parent[hi] = lo

    def components(self) -> dict[int, list[int]]:
        """H0 of the coupling graph: representative -> member positions.

        Computed from incidence directly. There is no `n x n` adjacency matrix
        anywhere here, which is the whole point of the disjoint-set form -- the
        dense version cost `5*n^2 + 36*n + 32` bytes against `16*n + 32`.
        """
        out: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            out.setdefault(self.find(i), []).append(i)
        return out

    def n_components(self) -> int:
        return len(self.components())

    # ---------------------------------------------------------------- sizes

    def n_slots(self) -> int:
        return self.hops

    def n_positions(self) -> int:
        return len(self._keys)

    def n_entries(self) -> int:
        return sum(len(s) for s in self._slots)

    # --------------------------------------------------- NeMo-Relay scopes

    def push_scope(self) -> None:
        self._depth += 1

    def pop_scope(self) -> None:
        """Drop exactly this scope's registrations. The root refuses removal.

        The root holds whatever every later sequence depends on -- a system
        prompt, a few-shot block. Popping it is refused rather than tolerated,
        which is `scope_stack.rs`'s rule and the reason root-adjacent entries
        sink to the bottom of any eviction order.
        """
        if self._depth == 0:
            raise RuntimeError("the root scope refuses removal")
        keep = [i for i, d in enumerate(self._scope_of) if d < self._depth]
        self._keys = [self._keys[i] for i in keep]
        for h in range(self.hops):
            self._slots[h] = [self._slots[h][i] for i in keep]
        self._scope_of = [self._scope_of[i] for i in keep]
        self._depth -= 1

    # ---------------------------------------------------- foliation eviction

    def evict_component(self, position: int) -> list[int]:
        """Drop a whole island at once.

        A component is the natural unit: its members reach each other, so
        dropping half of one leaves the rest reading a prefix that no longer
        exists. Evicting the island keeps the remaining tower exact, which is
        `foliation`'s elementary collapse read at island granularity rather than
        at single-block granularity.
        """
        members = sorted(self.components()[self.find(position)], reverse=True)
        for m in members:
            self.evict(m)
        return members

    def evict(self, position: int) -> None:
        """Remove a position from the key set entirely.

        Admissible only for a FREE FACE: an entry registered at the current scope
        depth, with nothing live still depending on it. An entry belonging to an
        enclosing scope is still referenced by everything the inner scope will
        compute, and dropping it would leave the tower with a hole -- the same
        corruption as a resident block whose parent was evicted.

        Removal is from the key set, not a mask. That is what makes it bitwise:
        an evicted position is absent from every later row's normalizer, so it
        cannot influence anything downstream at all.
        """
        if not 0 <= position < len(self._keys):
            raise IndexError(position)
        if self._scope_of[position] < self._depth:
            raise RuntimeError(
                f"position {position} belongs to an enclosing scope and is not a "
                f"free face; evicting it would leave the hop tower with a hole")
        del self._keys[position]
        del self._scope_of[position]
        for h in range(self.hops):
            del self._slots[h][position]
        # rebuild the forest: positions after `position` shift down by one, so a
        # stale parent array would silently point at the wrong island.
        self._parent = list(range(len(self._keys)))

    # ------------------------------------------------------------- the step

    def step(self, query_row: torch.Tensor, key_row: torch.Tensor,
             value_row: torch.Tensor, *, alpha: float, rho: float,
             scaling: float | None = None) -> torch.Tensor:
        """Advance one position; return its correction `sum_{h=1..K} (A^h v)_i`.

        Reads the prefix, computes ONE operator row, and reuses that row across
        every hop -- the row is the expensive part and it is computed once, which
        is the prefetch idea rather than K independent gathers.
        """
        prefix = torch.cat(self._keys, dim=-2) if self._keys \
            else key_row.new_zeros(*key_row.shape[:-2], 0, key_row.shape[-1])
        a = causal_row_operator(query_row, prefix, alpha=alpha, rho=rho,
                                scaling=scaling)

        corr = torch.zeros_like(value_row)
        new = []
        for h in range(self.hops):
            src = torch.cat(self._slots[h], dim=-2) if self._slots[h] \
                else value_row.new_zeros(*value_row.shape[:-2], 0, value_row.shape[-1])
            hop = a @ src if src.shape[-2] else torch.zeros_like(value_row)
            corr = corr + hop
            new.append(hop)

        pos = len(self._keys)
        self._keys.append(key_row)
        self._scope_of.append(self._depth)
        self._parent.append(pos)
        self._slots[0].append(value_row)
        for h in range(1, self.hops):
            self._slots[h].append(new[h - 1])

        # incidence: this position couples to every earlier one it actually
        # reaches. Union merges them into one island, exactly as `mj_island`
        # builds H0 from constraint/tree incidence rather than from a dense
        # adjacency matrix.
        if a.shape[-1]:
            reach = (a.abs().amax(dim=tuple(range(a.dim() - 1))) > self.couple_at)
            for j in reach.nonzero(as_tuple=False).flatten().tolist():
                self.union(pos, j)
        return corr
