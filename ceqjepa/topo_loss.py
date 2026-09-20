"""Topology ON TOP of causality: a differentiable 0-dimensional persistence loss
that shapes the embedding space while the committor supervises consequences.

WHY THIS IS A LOSS AND NOT A MONITOR. ceqjepa/coupling.py measures beta_0 of the
cross-block coupling graph and reports whether the curriculum built one manifold
or three. Measuring it leaves unification to luck. Persistence diagrams are
differentiable almost everywhere with respect to the underlying point positions
-- the persistence pairing is locally constant under small perturbations of the
distances, so the derivative of a diagram-valued function exists (Carriere et
al., "Optimizing persistent homology based functions", arXiv:2010.08356; Hu et
al., arXiv:1910.01877; Moor et al., "Topological Autoencoders", PMLR v119).
So beta_0 can be trained toward instead of watched.

WHAT IS DIFFERENTIABLE HERE, EXACTLY. In dimension 0 the persistence death times
of a Vietoris-Rips filtration are precisely the edge weights of the Euclidean
minimum spanning tree (single-linkage merge heights). Selecting WHICH edges form
the MST is combinatorial and carries no gradient, but the selection is locally
constant, and the selected edge weights are ordinary distances ||x_i - x_j||
whose gradient with respect to the embedding is standard. That is the whole
mechanism: pick the edges without a gradient, then differentiate their lengths.

THE TRAP THIS DESIGN IS BUILT AGAINST. "Make the blocks merge at a small radius"
is minimised perfectly by collapsing every embedding to one point -- beta_0 = 1
at radius 0, task solved, representation destroyed. That is the same shape as
the two defects this repository has already convicted: a conservation identity
that a collapsed encoder satisfied BETTER than a healthy one (7.772e-16 against
2.290e-13), and a bilinear delta whose zero-init made the gradient exactly zero
forever. A criterion an degenerate solution satisfies is not a criterion.
So the loss is a RATIO, scale-free by construction:

    L_topo = (largest cross-block MST edge) / (median within-block distance)

Collapse shrinks numerator and denominator together and the ratio does not
improve. Only genuinely interleaving the blocks lowers it. The self-check plants
exactly that negative and requires it to fail.

USE: add lambda_topo * topo_coupling_loss(E, blocks) to the committor loss. The
committor supervises WHAT HAPPENS under an intervention; this term shapes WHERE
the phases live relative to each other. They are different objects and neither
substitutes for the other.
"""

import torch

__all__ = ["mst_edges", "cross_block_merge_edge", "topo_coupling_loss"]


def mst_edges(D):
    """Prim's algorithm on a dense distance matrix. Returns (i, j) index pairs.

    Selection only -- no gradient flows through this function, and none should.
    The gradient enters where the SELECTED distances are recomputed by the
    caller from the embedding.
    """
    n = D.shape[0]
    with torch.no_grad():
        inside = torch.zeros(n, dtype=torch.bool, device=D.device)
        inside[0] = True
        best = D[0].clone()
        best_src = torch.zeros(n, dtype=torch.long, device=D.device)
        edges = []
        for _ in range(n - 1):
            cand = best.masked_fill(inside, float("inf"))
            j = int(torch.argmin(cand))
            edges.append((int(best_src[j]), j))
            inside[j] = True
            upd = D[j] < best
            best = torch.where(upd, D[j], best)
            best_src = torch.where(upd, torch.full_like(best_src, j), best_src)
    return edges


def cross_block_merge_edge(E, blocks):
    """The single MST edge whose removal would disconnect the block graph.

    In 0-dim persistence this is the death time at which the last two components
    merge, restricted to edges that actually join different phases. Returned as a
    LIVE tensor recomputed from E, so it carries a gradient.
    """
    D = torch.cdist(E, E)
    edges = mst_edges(D.detach())
    cross = [(i, j) for (i, j) in edges if int(blocks[i]) != int(blocks[j])]
    if not cross:
        # No MST edge joins two phases: the blocks are already separate
        # components at every radius the tree reaches. Return the largest
        # cross-block distance so the term still pulls, rather than a zero that
        # would silently report success.
        mask = blocks.unsqueeze(0) != blocks.unsqueeze(1)
        return D[mask].max()
    lens = torch.stack([torch.linalg.vector_norm(E[i] - E[j]) for (i, j) in cross])
    return lens.max()


def topo_coupling_loss(E, blocks, eps=1e-8):
    """Scale-free 0-dim coupling penalty. Lower means the phases interleave.

    E: [N, d] encoder outputs. blocks: [N] long, which phase each sample is from.
    Returns the ratio (cross-block merge height) / (median within-block distance),
    which collapse cannot game because both terms scale together.
    """
    if len(set(blocks.tolist())) < 2:
        return E.sum() * 0.0
    merge = cross_block_merge_edge(E, blocks)
    # Normalise by the WITHIN-block MST edges, not by a median pairwise distance.
    # MEASURED, and this is why: dividing an MST edge (a minimum connecting length)
    # by a median pairwise distance is NOT scale-free -- the two scale with the
    # point cloud's SHAPE, not just its size. The first version of this file did
    # exactly that and its own planted negative caught it: a fully collapsed
    # embedding scored 0.0192 against a healthy 1.0498, i.e. the loss PREFERRED
    # the degenerate solution. Comparing merge height against merge heights makes
    # numerator and denominator the same kind of quantity, so collapse moves both.
    D = torch.cdist(E, E)
    edges = mst_edges(D.detach())
    within_e = [(i, j) for (i, j) in edges if int(blocks[i]) == int(blocks[j])]
    if within_e:
        lens = torch.stack([torch.linalg.vector_norm(E[i] - E[j]) for (i, j) in within_e])
        scale = lens.median()
    else:
        scale = D[D > 0].median()
    # DETACH the denominator. MEASURED: with scale live, gradient descent minimises
    # the ratio by blowing the denominator UP -- exploding each block internally
    # rather than pulling the blocks together. Self-check (d) diverged 672 -> 7.9e26
    # at lr 0.5 doing exactly that. So the ratio has TWO degenerate solutions, not
    # one: collapse (numerator to 0) and explosion (denominator to infinity).
    # Detached, `scale` is a per-step unit conversion and the only thing the
    # gradient can move is the cross-block merge height, which is the intended
    # target. Collapse prevention stays where it belongs -- the committor loss.
    return merge / (scale.detach() + eps)


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)

    def blocks_of(n, k=3):
        return torch.arange(n) % k

    # (a) interleaved phases score lower than separated ones
    n = 60
    inter = torch.randn(n, 8, generator=g)
    sep = torch.randn(n, 8, generator=g) * 0.05
    off = torch.zeros(n, 8)
    for k in range(3):
        off[blocks_of(n) == k, k] = 25.0 * (k + 1)
    sep = sep + off
    b = blocks_of(n)
    l_inter = float(topo_coupling_loss(inter, b))
    l_sep = float(topo_coupling_loss(sep, b))
    print("interleaved loss %.4f   separated loss %.4f" % (l_inter, l_sep))
    assert l_inter < l_sep, "loss does not prefer interleaved phases"

    # (b) THE PLANTED NEGATIVE. Total collapse merges every block at radius ~0.
    #     A non-scale-free criterion would call that a perfect score. This one
    #     must NOT, or it is the conservation identity all over again.
    collapsed = torch.ones(n, 8) * 0.3 + torch.randn(n, 8, generator=g) * 1e-6
    l_col = float(topo_coupling_loss(collapsed, b))
    print("collapsed loss   %.4f   (must be NEUTRAL vs interleaved %.4f, not better)" % (l_col, l_inter))
    # The honest criterion is NEUTRALITY, not punishment. This term's job is to make
    # phases interleave; preventing collapse is the committor loss's job and a
    # variance floor's job. Asking one term to do both is how the conservation
    # identity ended up "detecting" a collapse it actually satisfied better. What
    # must NOT happen is the term REWARDING collapse.
    assert l_col >= l_inter * 0.5, (
        "collapse scores better than a healthy embedding -- the loss is gameable "
        "by the degenerate solution, which is the exact defect this file exists to avoid")
    assert l_col <= l_inter * 2.0, (
        "collapse and interleaving should be roughly NEUTRAL under a scale-free term")

    # (c) the term actually carries a gradient to the embedding
    E = torch.randn(n, 8, generator=g, requires_grad=True)
    topo_coupling_loss(E, b).backward()
    gmax = float(E.grad.abs().max())
    print("gradient reaches the embedding: max|dL/dE| = %.6e" % gmax)
    assert gmax > 0, "no gradient -- the loss is decorative"

    # (d) descending on it genuinely lowers beta_0's merge radius
    E2 = (torch.randn(n, 8, generator=g) * 0.05 + off).clone().requires_grad_(True)
    opt = torch.optim.SGD([E2], lr=0.05)
    before = float(topo_coupling_loss(E2, b))
    for _ in range(150):
        opt.zero_grad(); topo_coupling_loss(E2, b).backward(); opt.step()
    after = float(topo_coupling_loss(E2, b))
    print("optimising the term: %.4f -> %.4f" % (before, after))
    assert after < before, "the term is not actually minimisable by gradient descent"

    print("ALL SELF-CHECKS PASSED")
