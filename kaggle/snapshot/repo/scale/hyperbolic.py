"""Gromov delta-hyperbolicity for corpus graphs. Contract v10.1 section N2.

EVIDENCE CLASS: DERIVED (the formula and its algebra below), with the numeric
optimality claims tagged RUN where they were verified by exhaustive sweep rather
than proved in this module.

THE OPERATIVE DEFINITION. With the Gromov product

    (y|z)_x = 1/2 * ( d(x,y) + d(x,z) - d(y,z) ),

a metric space `(X, d)` is delta-hyperbolic iff for all quadruples of DISTINCT
points `w, x, y, z` with `x != y`,

    (x|y)_w >= min((x|z)_w, (z|y)_w) - delta,

and this module computes the OPTIMAL constant

    delta_hat = max_{w,x,y,z} [ min((x|z)_w, (z|y)_w) - (x|y)_w ].          (*)

WHY DISTINCT POINTS, AND WHY ONLY x != y IS EXCLUDED. Setting `y = x` gives
`(x|x)_w = 0 < (x|z)_w` whenever `d(w,x) + d(w,z) > d(x,z)`, so under literal
repeats NO metric space of diameter > 0 would read 0 -- trees included. The
definition is therefore quantified over distinct tuples with `x != y`, which is
the reading under which the classical statements hold. The REMAINING overlaps
are provably harmless: if `z in {x, y}` or `z = w` or `x = w` or `y = w`, then
`min((x|z)_w, (z|y)_w)` contains a zero factor while `(x|y)_w >= 0`, so the term
is `<= 0` and can never win the supremum. Only the `x = y` diagonal is excluded,
implemented as a masked diagonal of the violation matrix.

TREES READ EXACTLY ZERO. In a tree, let `p_xy` be the meeting point of the
`w->x` and `w->y` paths. The three meeting points satisfy (after relabelling)
`p_xz = p_yz` lying on the `w -> p_xy` path, whence
`(x|y)_w >= min((x|z)_w, (z|y)_w)` identically -- the classical four-point
lemma -- so the supremum in (*) is exactly the scalar 0.0.

EQUIVALENT FOUR-POINT-SUM FORM (the derivation of what is implemented).
Expand the products with `a = d(w,x)+d(w,y)+d(w,z)` and opposite-pair
distances `p = d(x,y), q = d(x,z), r = d(y,z)`:

    (x|y)_w = a - d(w,z) - p,   (x|z)_w = a - d(w,y) - q,   (z|y)_w = a - d(w,x) - r,

so the violation reorganises into the three PERFECT-MATCHING SUMS of the
quadruple, sigma_1 = d(x,y)+d(z,w), sigma_2 = d(x,z)+d(y,w),
sigma_3 = d(x,w)+d(y,z):

    min((x|z)_w,(z|y)_w) - (x|y)_w  =  sigma_1 - max(sigma_2, sigma_3).

Maximising over all labellings assigns each matching the sigma_1 role in turn,
hence

    delta_hat = max over unordered 4-sets {i,j,k,l} of (sigma_top - sigma_second),

the gap between the two largest of the three matching sums -- the standard
four-point condition. FACTOR CONVENTION: texts that define delta as HALF that
gap report values smaller by a factor of 2; this module implements the
product-form (*) pinned by the contract, under which trees read 0 and cycles
read girth/4.

CYCLES READ n/4 WHEN 4 | n. Attainment: anchor w at a point and place
x, y, z at arc offsets m, m/... concretely for n = 4m place x at m, y at 3m,
z at 2m; the matching sums are 3m, 2m, 2m, giving sigma_top - sigma_second = m
= n/4. Optimality for general n is consistent with the girth/4 folklore value;
for n <= 64 it was VERIFIED exhaustively against a literal brute-force
transcription of (*) during certification (tests/foreman/test_gromov_delta.py),
which is a RUN statement, not a proof. Odd cycles land between floor effects
and are asserted only against sanity envelopes.

COMPLETE GRAPHS READ ZERO. Every off-diagonal distance is 1, every matching sum
is exactly 2, the top-two gap is 0 -- the sharpest bound consistent with
diameter-<=1 geometry.

ALGORITHM AND COST. Exact mode evaluates (*) for EVERY anchor w by the
(max, min)-semiring product: with P_w[u,v] = (u|v)_w,

    B_w[x,y] = max_z min(P_w[x,z], P_w[z,y]),      violation sup_w = max(B_w - P_w)

with the diagonal masked. Each anchor costs O(n^2) memory and O(n^3) work, so
exact mode is O(n^4) time overall -- n <= 1024 runs in minutes on the CUDA
backend this module auto-selects. All quantities are multiples of 1/2 bounded by
n, exactly representable in float32 (<= 2**24 halves), so the fast dtype is
lossless on integral/half-integral inputs; arbitrary real metrics fall back to
float64. SAMPLED MODE maximises over a random subset of anchors and bases and is
therefore a MONOTONE LOWER BOUND on delta_hat -- it can miss the argmax
quadruple but can never invent a larger violation. It carries no concentration
guarantee and its output is labelled as a lower bound wherever it is reported.

DISCONNECTED INPUTS ARE REJECTED. Infinite distances make the products
undefined; callers pass a connected node set (`hop_distances(..., nodes=...)`),
by convention the merged component the corpus cares about. This matches how the
corpus's own lambda_2 is measured (`scale.foreman_lambda2` works on the merged
component), so the two covariates in results/delta_hyperbolicity.* describe the
SAME vertex set.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

try:
    import torch
except ImportError as _exc:  # pragma: no cover - the repo pins torch
    raise ImportError("scale.hyperbolic requires torch (repo requirement)") from _exc

__all__ = ["gromov_delta", "hop_distances"]

_HALF_INT_LIMIT = 2 ** 23          # float32 represents k/2 exactly up to here


# ------------------------------------------------------------------ metrics --
def hop_distances(node_count: int, edges, nodes=None) -> np.ndarray:
    """All-pairs shortest-hop matrix of the (weighted 1) graph, restricted to
    `nodes` (default: every node 0..node_count-1).

    Duplicate edges and self-loops are ignored. Raises ValueError if the induced
    subgraph on `nodes` is disconnected -- `gromov_delta` refuses infinite
    metrics rather than silently measuring a component of them.
    """
    keep = range(node_count) if nodes is None else list(nodes)
    keep_set = set(keep)
    adjacent = {v: set() for v in keep}
    for a, b in edges:
        if a == b or a not in keep_set or b not in keep_set:
            continue
        adjacent[a].add(b)
        adjacent[b].add(a)
    order = list(keep)
    index = {v: i for i, v in enumerate(order)}
    n = len(order)
    dist = np.full((n, n), np.inf, dtype=np.float64)
    for i, src in enumerate(order):
        dist[i, index[src]] = 0.0
        seen = {src}
        queue = deque([(src, 0)])
        while queue:
            u, du = queue.popleft()
            for v in adjacent[u]:
                if v not in seen:
                    seen.add(v)
                    dist[i, index[v]] = du + 1
                    queue.append((v, du + 1))
    if not np.all(np.isfinite(dist)):
        raise ValueError(
            "induced subgraph on `nodes` is disconnected; pass a connected "
            "component (e.g. the merged component) instead")
    return dist


# ------------------------------------------------------------------- core ----
def _prepare(dist_matrix, device):
    d = np.asarray(dist_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {d.shape}")
    if d.shape[0] < 2:
        raise ValueError("need at least 2 points")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance matrix has inf/nan; pass a connected component")
    if float(d.min()) < 0.0:
        raise ValueError("negative distance entry")
    if float(np.abs(np.diag(d)).max()) != 0.0:
        raise ValueError("diagonal must be zero")
    if float(np.abs(d - d.T).max()) > 1e-9 * max(1.0, float(np.abs(d).max())):
        raise ValueError("distance matrix is not symmetric")
    twice_is_int = np.all(2.0 * d == np.round(2.0 * d))
    if device == "cpu":
        return torch.from_numpy(d), torch.float64
    if twice_is_int and 2.0 * float(d.max()) <= _HALF_INT_LIMIT:
        return torch.from_numpy(d), torch.float32
    return torch.from_numpy(d), torch.float64


def _anchor_sweep(D_t, w_list, base_subset, progress=None):
    """max over anchors w in `w_list` of the violation (*); `base_subset` is None
    for exact mode (all indices) or an index tensor restricting x, y, z."""
    dev = D_t.device
    n = D_t.shape[0]
    best = -float("inf")
    t0 = time.time()
    for step, w in enumerate(w_list):
        dw = D_t[w]
        if base_subset is None:
            P = 0.5 * (dw.unsqueeze(1) + dw.unsqueeze(0) - D_t)
        else:
            sub = D_t[base_subset][:, base_subset]
            dws = dw[base_subset]
            P = 0.5 * (dws.unsqueeze(1) + dws.unsqueeze(0) - sub)
        B = torch.full_like(P, -float("inf"))
        m = P.shape[0]
        for z in range(m):
            torch.maximum(B, torch.minimum(P[:, z].unsqueeze(1),
                                           P[z, :].unsqueeze(0)), out=B)
        V = B - P
        V.fill_diagonal_(-float("inf"))
        cand = float(V.max().item())
        if cand > best:
            best = cand
        if progress and (step + 1) % progress == 0:
            dt = time.time() - t0
            print(f"    anchor {step + 1}/{len(w_list)}  best={best:.6f}  "
                  f"({dt:.1f}s)", flush=True)
    return best


def gromov_delta(dist_matrix, sampled=False, rng=None, *, anchors: int = 64,
                 max_side: int = 256, device=None, progress=None) -> float:
    """Optimal delta-hyperbolicity of a finite metric, formula (*) in the
    module docstring.

    Parameters
    ----------
    dist_matrix :
        Square symmetric non-negative matrix, zero diagonal, finite (pass a
        connected component; disconnected input raises).
    sampled :
        False (default): exact sweep over every anchor vertex, O(n^4) time.
        True: monotone LOWER bound -- `anchors` vertices are drawn without
        replacement by `rng` (a fresh `numpy.random.default_rng()` if absent)
        and each anchor's x/y/z domain is a uniform subset of size at most
        `max_side` containing the anchor. Never exceeds the exact value;
        equality holds only if the argmax quadruple survives sampling.
    rng : numpy.random.Generator, optional
        Required for reproducibility in sampled mode; ignored otherwise.
    anchors, max_side :
        Sampled-mode budgets (see above). Ignored in exact mode.
    device :
        "cuda" / "cpu"; default auto-selects CUDA when available. Inputs whose
        doubled entries are integers below 2**24 run in lossless float32 on
        CUDA; everything else (and all of CPU) runs in float64.
    progress : int, optional
        Print an anchor-progress line every `progress` anchors.

    Returns
    -------
    float
        delta_hat >= 0. For fewer than 4 points no distinct quadruple exists
        and the convention delta_hat = 0.0 is returned.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    D_np, dtype = _prepare(dist_matrix, device)
    n = D_np.shape[0]
    if n < 4:
        return 0.0
    D_t = D_np.to(device=device, dtype=dtype)

    if not sampled:
        best = _anchor_sweep(D_t, range(n), None, progress=progress)
        return max(0.0, float(best))

    rng = rng if rng is not None else np.random.default_rng()
    k = int(min(n, max(1, anchors)))
    anchor_idx = rng.choice(n, size=k, replace=False)
    best = -float("inf")
    for w in anchor_idx:
        side = int(min(n, max_side))
        if side >= n:
            subset = None
        else:
            subset = rng.choice(n, size=side, replace=False)
            if not np.any(subset == w):
                subset[0] = w
            subset = np.sort(subset)
        subset_t = None if subset is None else torch.as_tensor(subset, device=device)
        cand = _anchor_sweep(D_t, [int(w)], subset_t)
        best = max(best, cand)
    return max(0.0, float(best))
