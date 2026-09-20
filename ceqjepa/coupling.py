"""Block coupling: is the curriculum building ONE manifold or three?

Each curriculum phase owns a disjoint block of singular directions in the
encoder's embedding space (the low-rank delta_logits factorisation supplies
the frame). Disjoint blocks give forgetting protection for free, but perfect
disjointness is a direct sum -- no forgetting AND no transfer, which is three
models sharing a tensor.

beta_0 of the coupling graph decides which one happened:

    beta_0 == n_blocks   the blocks are separate components; a direct sum
    beta_0 == 1          one connected manifold

and the radius at which it drops from n_blocks to 1 says how STRONGLY they are
coupled, not merely whether. Merging at small epsilon is tight coupling;
merging only at large epsilon is technically connected and practically
separate.

The same number decides whether a rank-constrained completion of Q can couple
the blocks at all: low-rank completion propagates constraints only along
connected observation patterns, so a disconnected graph cannot force any
cross-block agreement no matter how long it trains.

Union-find, no dependencies. Aether-Lang's persistence.rs computes the full
diagram (BettiNumbers3, betti_at) if the whole curve is ever wanted; beta_0
alone does not need it.
"""

import torch

__all__ = ["block_frame", "coupling_graph", "betti_0", "merge_radius", "coupling_report"]


def block_frame(delta_a, n_blocks, block_dim):
    """Orthonormal per-block frames from the low-rank factor.

    delta_a: [d_enc, n*rank] or [d_enc, k] -- the encoder-side factor of the
    per-example modulation. Returns a list of [d_enc, block_dim] column-
    orthonormal frames, one per phase, taken from disjoint singular directions
    so no two phases can occupy the same direction by construction.
    """
    U, S, _ = torch.linalg.svd(delta_a.double(), full_matrices=False)
    need = n_blocks * block_dim
    if U.shape[1] < need:
        raise ValueError(
            "delta_a supplies %d singular directions, curriculum needs %d "
            "(%d blocks x %d dims). Raise --rank or lower block_dim."
            % (U.shape[1], need, n_blocks, block_dim))
    return [U[:, i * block_dim:(i + 1) * block_dim] for i in range(n_blocks)]


def coupling_graph(E, blocks, eps):
    """Edges between embedding samples within eps, labelled by block.

    E: [N, d_enc] encoder outputs. blocks: [N] long, which phase each sample
    came from. Returns (n_nodes, edges) with edges as (i, j) index pairs.
    Only CROSS-BLOCK edges are returned: within-block connectivity is
    guaranteed by construction and would mask the question being asked.
    """
    D = torch.cdist(E.double(), E.double())
    N = E.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    close = D[iu[0], iu[1]] <= eps
    cross = blocks[iu[0]] != blocks[iu[1]]
    keep = close & cross
    return N, list(zip(iu[0][keep].tolist(), iu[1][keep].tolist()))


def betti_0(n_nodes, edges, blocks):
    """Connected components of the BLOCK graph induced by cross-block edges.

    Counts components over blocks, not over samples: the question is whether
    phase 1's directions reach phase 3's, not whether two draws are close.
    """
    labels = sorted(set(blocks.tolist()))
    parent = {b: b for b in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        a, b = find(int(blocks[i])), find(int(blocks[j]))
        if a != b:
            parent[a] = b
    return len({find(b) for b in labels})


def merge_radius(E, blocks, lo=1e-3, hi=None, steps=40):
    """Smallest eps at which beta_0 reaches 1, or None if it never does.

    This is the strength of the coupling. A small radius means the blocks are
    genuinely interleaved; a radius near the diameter means they touch only
    because everything touches at that scale.
    """
    D = torch.cdist(E.double(), E.double())
    if hi is None:
        hi = float(D.max())
    n_blocks = len(set(blocks.tolist()))
    best = None
    for t in range(steps):
        eps = lo * (hi / lo) ** (t / max(steps - 1, 1))
        n, edges = coupling_graph(E, blocks, eps)
        if betti_0(n, edges, blocks) == 1:
            best = eps
            hi = eps
    return best


def coupling_report(E, blocks, eps=None):
    """One line per eval. beta_0 == n_blocks means the curriculum is filing."""
    n_blocks = len(set(blocks.tolist()))
    D = torch.cdist(E.double(), E.double())
    if eps is None:
        # eps is calibrated to WITHIN-block spread, not global spread. Two
        # blocks count as coupled when cross-block distances are comparable to
        # the distances a block already spans internally. The global median is
        # the wrong operating point: on widely separated blocks it exceeds the
        # separation itself and reports coupling that is an artefact of scale
        # (measured: three blocks offset by 40/80/120 read beta_0=2 at the
        # global median while their true structure is three components).
        same = blocks.unsqueeze(0) == blocks.unsqueeze(1)
        within = D[same & (D > 0)]
        eps = float(within.median()) if within.numel() else float(D[D > 0].median())
    n, edges = coupling_graph(E, blocks, eps)
    b0 = betti_0(n, edges, blocks)
    r = merge_radius(E, blocks)
    return {
        "n_blocks": n_blocks,
        "beta_0": b0,
        "unified": b0 == 1,
        "eps": eps,
        "cross_block_edges": len(edges),
        "merge_radius": r,
        "merge_radius_over_diameter": (r / float(D.max())) if r else None,
    }


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)

    # three blocks that genuinely overlap -> should unify
    E1 = torch.randn(90, 16, generator=g) * 0.6
    b1 = torch.arange(90) % 3
    r1 = coupling_report(E1, b1)
    assert r1["beta_0"] == 1, r1
    print("overlapping blocks : beta_0=%d unified=%s merge_r/diam=%.3f"
          % (r1["beta_0"], r1["unified"], r1["merge_radius_over_diameter"]))

    # three blocks pushed far apart -> must NOT unify at the median radius
    E2 = torch.randn(90, 16, generator=g) * 0.05
    off = torch.zeros(90, 16)
    for k in range(3):
        off[torch.arange(90) % 3 == k, k] = 40.0 * (k + 1)
    E2 = E2 + off
    b2 = torch.arange(90) % 3
    r2 = coupling_report(E2, b2)
    assert r2["beta_0"] == 3, r2
    print("separated blocks   : beta_0=%d unified=%s  (the direct-sum failure)"
          % (r2["beta_0"], r2["unified"]))

    # the planted negative: a detector that cannot report 3 is not a detector
    assert r1["beta_0"] != r2["beta_0"], "beta_0 does not separate the two cases"
    print("ALL SELF-CHECKS PASSED")
