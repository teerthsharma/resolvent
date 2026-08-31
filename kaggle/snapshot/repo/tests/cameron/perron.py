"""A nonnegative, Perron-certified settling operator. CPU and CUDA, float64.

This is the kernel the R3/R4/R6 falsifiers run against. It is deliberately one
file and one operator, because all three requirements turn out to be questions
about the same object: a nonnegative matrix A and the certificate

    A >= 0 entrywise, and there exists w > 0 with A w <= rho w

which is `CEQ.Contraction.weighted_contraction` in `lean/CEQ/Contraction.lean`.
Row-stochasticity is the w = 1, rho = 1 corner of that condition and is the
thing being given up. Everything else -- nonnegativity, contraction, a settled
state that exists and is unique -- is kept.

Nothing here is imported from any prior repository. The measured results that
motivated the shape of it are recorded in `house-events.jsonl`.
"""

from __future__ import annotations

import torch

MAD_TO_SIGMA = 1.4826      # median-absolute-deviation -> sigma, for a normal
OUTLIER_MADS = 3.0         # the conventional robust-outlier cut
CONTROL = 0                # channel carrying "this token is a flag / a bound"


def _proj(d: int, dtype, device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(d, d, generator=g, dtype=dtype).to(device) / d ** 0.5


def robust_z(v: torch.Tensor) -> torch.Tensor:
    """Standardize by median and 1.4826*MAD, never mean and sigma.

    Measured reason (BACKLOG.md B1): on distilgpt2 the max per-dimension sigma
    is 22.1 against a median of 0.308 -- 72x, concentrated in ~5 channels. A
    mean/sigma standardization hands those channels the whole scale.
    """
    med = v.median()
    mad = (v - med).abs().median()
    return (v - med) / (MAD_TO_SIGMA * mad + torch.finfo(v.dtype).eps)


def scores(x: torch.Tensor) -> torch.Tensor:
    """Attention logits. The control channel enters as an additive bias so the
    softmax baseline sees EXACTLY the signal the gate below sees. A baseline
    denied the module's information is not a baseline."""
    d = x.shape[-1]
    q = x @ _proj(d, x.dtype, x.device, 101)
    k = x @ _proj(d, x.dtype, x.device, 202)
    return q @ k.transpose(-2, -1) / d ** 0.5 + robust_z(x[:, CONTROL])[None, :]


def transition(x: torch.Tensor) -> torch.Tensor:
    """Row-stochastic P. Shared by every arm, so only the OPERATOR differs."""
    return torch.softmax(scores(x), dim=-1)


def attention(x: torch.Tensor) -> torch.Tensor:
    """arm 1: one softmax hop, values = the tokens themselves."""
    return transition(x) @ x


def salience(x: torch.Tensor, rho: float) -> torch.Tensor:
    """Per-SOURCE gate s_j in [0, rho]. Gating the source, not the reader, is
    what lets a token's contribution be destroyed rather than redistributed."""
    z = robust_z(x[:, CONTROL])
    return rho * torch.sigmoid(z - OUTLIER_MADS)


def perron_operator(x: torch.Tensor, rho: float) -> torch.Tensor:
    """arm 3: A_ij = s_j * P_ij. Nonnegative, row sums free, certified by w = 1
    because (A 1)_i = sum_j P_ij s_j <= max_j s_j <= rho < 1."""
    return transition(x) * salience(x, rho)[None, :]


def neumann_steps(rho: float, tol: float = 1e-13) -> int:
    """K such that the truncation error rho^K/(1-rho) is below tol.

    `CEQ.Occupancy.occupancy_telescope` -- (1-A) * sum_{k<N} A^k = 1 - A^N --
    is the certificate that this is the resolvent and not an approximation of
    unknown quality.
    """
    import math
    return max(1, math.ceil(math.log(tol * (1.0 - rho)) / math.log(rho)))


def settle(a: torch.Tensor, b: torch.Tensor, rho: float = 0.9) -> torch.Tensor:
    """z* = (I - A)^-1 b by truncated Neumann. K matvecs, no dense solve."""
    z = b.clone()
    term = b
    for _ in range(neumann_steps(rho)):
        term = a @ term
        z = z + term
    return z


def certificate(a: torch.Tensor, w: torch.Tensor):
    """Check the Lean hypothesis pointwise. Returns (ok, rho, min w).

    Fails CLOSED: a nonnegative A with any w entry at zero has an undefined
    weighted norm, and that is the measured reducible-A failure mode
    (min(w) = 6.6e-14), so it is refused here rather than certified.
    """
    nonneg = bool((a >= 0).all())
    wmin = float(w.min())
    if not nonneg or wmin <= 0:
        return False, float("inf"), wmin
    rho = float((a @ w / w).max())
    return rho < 1.0, rho, wmin


def linear_probe_r2(z: torch.Tensor, y: torch.Tensor) -> float:
    """R^2 of the best linear reconstruction of y from z. The deadness guard:
    an operator that answers R3 by collapsing every row onto one token wins the
    perturbation ratio and scores near zero here."""
    zz = torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)], 1)
    beta = torch.linalg.lstsq(zz, y).solution
    res = ((zz @ beta - y) ** 2).sum()
    tot = ((y - y.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - res / tot)


def effective_size(z: torch.Tensor) -> float:
    """Participation ratio of the singular-value spectrum: (sum s)^2 / sum s^2.

    R4's falsifier is 'effective size of the settled state across many
    contexts', and 'near full width means no compression story'. This is that
    number; full width is z.shape[1].
    """
    s = torch.linalg.svdvals(z.to(torch.float64))
    return float(s.sum() ** 2 / (s ** 2).sum())


# ---------------------------------------------------------------- max-plus side
#
# In the log domain the same nonnegative matrix is a max-plus matrix and its
# Perron root is a max CYCLE MEAN. That single number is the convergence
# condition of the max-plus star z* = max_a(r_a + gamma A_a z*) and it is also
# the obstruction to a global potential: a set of pairwise readings g admits a
# potential phi with g_e = phi_head - phi_tail if and only if every cycle sums
# to zero. Contraction and agreement are the same statistic read on different
# operators.

def spanning_forest(edges, n_nodes):
    """BFS spanning forest. Returns (tree_edge_indices, adjacency).

    BFS and not insertion-order union-find: the fundamental cycles are as SHORT
    as the graph allows, and cycle length is the noise budget -- a defect that
    has to be read across a 39-edge tree path competes with sigma*sqrt(39).
    """
    nbr = [[] for _ in range(n_nodes)]
    for i, (u, v) in enumerate(edges):
        nbr[u].append((v, i, +1))
        nbr[v].append((u, i, -1))
    seen = [False] * n_nodes
    tree, adj = [], [[] for _ in range(n_nodes)]
    from collections import deque
    for root in range(n_nodes):
        if seen[root]:
            continue
        seen[root], q = True, deque([root])
        while q:
            a = q.popleft()
            for b, i, sgn in nbr[a]:
                if not seen[b]:
                    seen[b] = True
                    tree.append(i)
                    adj[a].append((b, i, sgn))
                    adj[b].append((a, i, -sgn))
                    q.append(b)
    return tree, adj


def cycle_defects(edges, n_nodes, g):
    """Sum of the readings around each fundamental cycle, and its length.

    ponytail: fundamental cycles from one spanning tree, not every cycle. That
    is a lower bound on the true max cycle mean; upgrade to Karp's O(VE)
    algorithm if a planted contradiction is ever found that hides outside the
    fundamental basis.
    """
    tree, adj = spanning_forest(edges, n_nodes)
    tree_set = set(tree)
    psi = [None] * n_nodes
    depth = [0] * n_nodes
    for root in range(n_nodes):
        if psi[root] is not None:
            continue
        psi[root], stack = 0.0, [root]
        while stack:
            a = stack.pop()
            for b, ei, sgn in adj[a]:
                if psi[b] is None:
                    psi[b] = psi[a] + sgn * float(g[ei])
                    depth[b] = depth[a] + 1
                    stack.append(b)
    out = []
    for i, (u, v) in enumerate(edges):
        if i in tree_set or psi[u] is None or psi[v] is None:
            continue
        out.append((float(g[i]) - (psi[v] - psi[u]), 1 + depth[u] + depth[v]))
    return out


def holonomy_z(edges, n_nodes, g) -> float:
    """The module's disagreement statistic: worst cycle defect, standardized.

    max_k |sum_{e in C_k} g_e| / sqrt(|C_k|). The max CYCLE MEAN divides by
    |C_k| because that is what makes it a max-plus eigenvalue; for DETECTION
    that is the wrong denominator, because a planted contradiction does not
    grow with cycle length while accumulated reading noise grows as its square
    root. Same subspace, same L-infinity read, correct normalisation.
    """
    d = cycle_defects(edges, n_nodes, g)
    return max((abs(s) / L ** 0.5 for s, L in d), default=0.0)


def max_cycle_mean(edges, n_nodes, g) -> float:
    """The module's disagreement statistic: worst cycle, per unit length.

    An L-infinity read of the cycle space. The averaging baseline reads the
    SAME subspace in L2 -- that is what 'averaged away' means concretely.
    """
    d = cycle_defects(edges, n_nodes, g)
    return max((abs(s) / L for s, L in d), default=0.0)


def potential_residual(edges, n_nodes, g) -> float:
    """The averaging baseline: fit one global potential by least squares and
    report the RMS residual. It distributes a local contradiction over every
    edge, which is exactly the behaviour R6 says must be beaten."""
    import torch as _t
    b = _t.zeros(len(edges), n_nodes, dtype=_t.float64)
    for i, (u, v) in enumerate(edges):
        b[i, v] += 1.0
        b[i, u] -= 1.0
    gg = g.to(_t.float64).reshape(-1, 1).cpu()
    phi = _t.linalg.lstsq(b, gg).solution
    return float(((b @ phi - gg) ** 2).mean().sqrt())


def linear_probe_r2_heldout(ztr, ytr, zte, yte) -> float:
    """R^2 on a held-out split. A 32->32 fit on a few hundred samples scores
    near 1.0 in-sample whatever it has learned, which would make every R4
    reconstruction claim vacuous."""
    ones = lambda z: torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype,
                                              device=z.device)], 1)
    beta = torch.linalg.lstsq(ones(ztr), ytr).solution
    res = ((ones(zte) @ beta - yte) ** 2).sum()
    tot = ((yte - ytr.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - res / tot)
