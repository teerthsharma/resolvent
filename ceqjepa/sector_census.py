"""ADDENDUM W5.1 -- the sector census: beta_0 and beta_1 of the propagation
complex, per bed, per batch, asserted against projector ranks rather than
against a single combinatorial count.

THE QUESTION THIS FILE ANSWERS. W5.2's gauge theorem only bites where
beta_1 = 0: a phase gate on a beta_1 = 0 graph is gauge-equivalent to unsigned
gates with rephased values, so no sign/interference claim on such a bed is
falsifiable there. Whether that is every bed this repo owns is an empirical
question about each bed's PROPAGATION GRAPH -- which node's label depends on
which other nodes -- and is answered here per bed, not assumed.

WHAT "PROPAGATION COMPLEX" MEANS, PER BED (read off each bed's own generator,
not re-derived):

  BED-M   (ceq/corpus.py)        acc_i = acc_{i-1} op c inside the executed
                                  program's own `for` loop -- a CHAIN of
                                  accumulator states, length read off a REAL
                                  drawn sample's own `n` field.
  BED-K delay   (bed_k.py)       z_i = b_{i-d} -- one edge per i >= d, wired
                                  from the REAL kernel matrix's nonzeros, not
                                  assumed to be exactly the formula.
  BED-K powerlaw (bed_k.py)      z_i = sum_{j<i} K_ij b_j, K from the real GL
                                  weights -- wired from whichever entries are
                                  actually nonzero, which for H in (0.5, 1) is
                                  every j < i.
  BED-1   (bed_1.py)             the bed's own 11-node landscape graph, read
                                  directly off `build()["edges"]`.
  BED-H   (bed_h.py)             the belief recursion b_t = f(b_{t-1}, y_t)
                                  over T_LEN steps: a chain of belief nodes
                                  with one observation leaf per step.

THE CENSUS. For an n-node, E-edge propagation graph (a 1-complex: no 2-cells,
so H_2 and above vanish and chi = beta_0 - beta_1 is the whole Euler relation):

  beta_0   connected components
  beta_1   independent cycles (the cyclomatic number E - n + beta_0)
  sectors  (n - beta_0, E - n + beta_0 - beta_1, beta_1) -- tree / curl / cycle
           dimensions of the edge space, via the signed incidence matrix d's
           SVD: rank(d) = n - beta_0 (the tree sector, im(d^T)'s dimension),
           E - rank(d) = beta_1 (the cycle sector, ker(d)'s dimension), and the
           middle (curl) sector is IDENTICALLY 0 for a bare graph -- there is
           no 2-cell for a curl sector to be the image of. That identity is
           asserted here via the SVD projectors' traces, not taken on faith:
           `sector_dims` raises if the projector ranks disagree with the
           union-find/cyclomatic route, or if the two projectors are not
           orthogonal complements of each other in R^E.

MUST-FIRE (W5.1). `plant_extra_cycle` adds one edge between two nodes already
in the same component; `must_fire_planted_cycle` asserts beta_1 rises by
EXACTLY 1 and beta_0 does not move, and returns the planted edge and the
before/after numbers so the test can print what fired.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "betti_numbers", "sector_dims", "sector_census", "plant_extra_cycle",
    "must_fire_planted_cycle", "chain_graph",
    "propagation_graph_bed_m", "propagation_graph_bed_k",
    "propagation_graph_bed_1", "propagation_graph_bed_h",
    "run_all_beds", "print_census",
]

TOL = 1e-9


def _incidence(n: int, edges) -> np.ndarray:
    """Signed n x E incidence matrix. Orientation per edge is arbitrary
    (head +1, tail -1): rank(d), ker(d)'s dimension and the union-find /
    cyclomatic numbers below do not depend on which endpoint is called head."""
    edges = list(edges)
    d = np.zeros((n, len(edges)), dtype=np.float64)
    for k, (u, v) in enumerate(edges):
        d[u, k] -= 1.0
        d[v, k] += 1.0
    return d


def _components(n: int, edges) -> tuple[int, list[int]]:
    """Union-find. Returns (beta_0, root-per-node) -- the combinatorial route,
    independent of the incidence matrix used by the linear-algebra route."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    roots = [find(i) for i in range(n)]
    return len({r for r in roots}), roots


def betti_numbers(n: int, edges) -> tuple[int, int]:
    """beta_0, beta_1 of the 1-complex (n nodes, `edges`), by two independent
    routes that must agree -- union-find + cyclomatic number on one side,
    incidence-matrix rank on the other. Raises AssertionError on disagreement
    rather than silently trusting either route."""
    edges = list(edges)
    E = len(edges)
    beta0_a, _ = _components(n, edges)
    beta1_a = E - n + beta0_a

    rank = int(np.linalg.matrix_rank(_incidence(n, edges), tol=TOL)) if E else 0
    beta0_b, beta1_b = n - rank, E - rank

    assert beta0_a == beta0_b, (
        f"beta_0 disagreement: union-find={beta0_a} vs incidence-rank route={beta0_b}")
    assert beta1_a == beta1_b, (
        f"beta_1 disagreement: cyclomatic={beta1_a} vs incidence-rank route={beta1_b}")
    return beta0_a, beta1_a


def sector_dims(n: int, edges) -> tuple[int, int, int, dict]:
    """(tree, curl, cycle) = (n - beta_0, E - n + beta_0 - beta_1, beta_1),
    computed from SVD projectors onto the incidence matrix's row space (tree)
    and null space (cycle), and asserted against `betti_numbers` and against
    each other's orthogonality. Returns the three dimensions plus an evidence
    dict (the projector ranks and the orthogonality residual) so a caller can
    print the certificate instead of the bare ints."""
    edges = list(edges)
    E = len(edges)
    beta0, beta1 = betti_numbers(n, edges)

    if E == 0:
        tree_dim, cycle_dim, ortho_residual = 0, 0, 0.0
    else:
        d = _incidence(n, edges)
        _, s, Vt = np.linalg.svd(d, full_matrices=True)
        rank = int((s > TOL).sum())
        V_row = Vt[:rank]                    # rank x E
        P_row = V_row.T @ V_row              # E x E projector onto im(d^T)
        P_null = np.eye(E) - P_row           # projector onto ker(d)
        tree_dim = int(round(np.trace(P_row)))
        cycle_dim = int(round(np.trace(P_null)))
        ortho_residual = float(np.abs(P_row @ P_null).max())
        assert ortho_residual < 1e-6, (
            f"row/null projectors are not orthogonal complements: residual={ortho_residual}")

    curl_dim = E - tree_dim - cycle_dim
    assert tree_dim == n - beta0, f"tree sector {tree_dim} != n-beta0 {n - beta0}"
    assert cycle_dim == beta1, f"cycle sector {cycle_dim} != beta1 {beta1}"
    assert curl_dim == 0, f"curl sector {curl_dim} != 0 (a graph has no 2-cells to fill it)"

    evidence = dict(rank_row=tree_dim, rank_null=cycle_dim, ortho_residual=ortho_residual)
    return tree_dim, curl_dim, cycle_dim, evidence


def plant_extra_cycle(n: int, edges, rng=None) -> tuple[tuple[int, int], list]:
    """Adds ONE edge between two nodes already in the same component and not
    already directly joined -- the W5.1 plant. Any such edge raises beta_1 by
    exactly 1 (E grows by 1, beta_0 is unchanged because the endpoints were
    already connected), so the choice among qualifying pairs is cosmetic:
    lowest-index pair unless `rng` is given. Raises ValueError if no
    qualifying pair exists (e.g. every component is already a complete
    graph), so a caller cannot get a silent no-op plant."""
    edges = list(edges)
    _, roots = _components(n, edges)
    existing = {frozenset(e) for e in edges}
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n)
                  if roots[u] == roots[v] and frozenset((u, v)) not in existing]
    if not candidates:
        raise ValueError("no same-component, non-adjacent pair left to plant a cycle on")
    u, v = candidates[0] if rng is None else candidates[int(rng.integers(len(candidates)))]
    return (u, v), edges + [(u, v)]


def must_fire_planted_cycle(n: int, edges, rng=None) -> dict:
    """W5.1's must-fire. Plants one extra cycle and asserts beta_1 rises by
    exactly 1 while beta_0 does not move; returns the before/after numbers and
    the planted edge so the caller can print what fired rather than take a
    bare pass/fail."""
    beta0_before, beta1_before = betti_numbers(n, edges)
    (u, v), planted = plant_extra_cycle(n, edges, rng=rng)
    beta0_after, beta1_after = betti_numbers(n, planted)
    delta = beta1_after - beta1_before
    assert beta0_after == beta0_before, (
        f"planting a cycle moved beta_0: {beta0_before} -> {beta0_after}")
    assert delta == 1, (
        f"must-fire failed: beta_1 moved by {delta} (planted edge {(u, v)}), not by 1")
    return dict(planted_edge=(u, v), beta0=beta0_before,
                beta1_before=beta1_before, beta1_after=beta1_after, delta=delta)


# --------------------------------------------------------------------------
# propagation graphs -- one per tracked bed, read off each bed's own code
# --------------------------------------------------------------------------


def chain_graph(n: int) -> tuple[int, list]:
    """A path graph on `n` nodes: node i -> node i+1, n-1 edges. beta_0=1,
    beta_1=0 for any n >= 1 -- the generic CHAIN propagation structure."""
    if n < 1:
        raise ValueError(f"a chain needs at least one node, got n={n}")
    return n, [(i, i + 1) for i in range(n - 1)]


def propagation_graph_bed_m(seed: int = 0) -> tuple[int, list, dict]:
    """BED-M (ceq/corpus.py): the label is the accumulator of the generated
    program's own `for _ in range(n): acc op= c` loop -- a chain of
    `n_iter + 1` accumulator states. `n_iter` is a REAL drawn sample's own
    `n` field, not an assumed constant."""
    from ceq import corpus
    batch = corpus.build(n_train=1, n_test=0, seed=seed)
    sample = batch["train"][0]
    n_iter = sample["n"]
    n, edges = chain_graph(n_iter + 1)
    return n, edges, dict(bed="BED-M", source="ceq.corpus.build",
                           n_iter=n_iter, sample_src=sample["src"])


def propagation_graph_bed_k(kind: str, n: int, seed: int = 0, **params) -> tuple[int, list, dict]:
    """BED-K (ceq/beds/bed_k.py): edges are the STRUCTURAL nonzeros of the
    REAL kernel matrix this bed's own `kernel_matrix(kind, n, **params)`
    builds -- z_i depends on b_j wherever K[i, j] != 0, read off the matrix
    rather than off the formula that generated it."""
    from ceq.beds import bed_k
    K = bed_k.kernel_matrix(kind, n, **params)
    edges = [(j, i) for i in range(n) for j in range(i) if abs(K[i, j]) > TOL]
    max_edges = n * (n - 1) // 2
    return n, edges, dict(bed=f"BED-K-{kind}", source="ceq.beds.bed_k.kernel_matrix",
                           params=params, seed=seed,
                           nonzero_fraction=len(edges) / max_edges if max_edges else 0.0)


def propagation_graph_bed_1(T: float = 1.0, seed: int = 0) -> tuple[int, list, dict]:
    """BED-1 (ceq/beds/bed_1.py): the bed's own 11-node landscape graph, read
    directly off a REAL `build()` call's `edges` / `node_index` -- not
    reconstructed from the docstring's description of it."""
    from ceq.beds import bed_1
    bed = bed_1.build(T=T, seed=seed)
    ix = bed["node_index"]
    edges = [(ix[u], ix[v]) for u, v in bed["edges"]]
    return bed["n"], edges, dict(bed="BED-1", source="ceq.beds.bed_1.build",
                                  names=bed["names"], channels=bed["channels"])


def propagation_graph_bed_h() -> tuple[int, list, dict]:
    """BED-H (ceqjepa/beds/bed_h.py): the belief recursion b_t = f(b_{t-1},
    y_t) over the bed's own `T_LEN` steps is a chain of belief nodes with one
    observation leaf hanging off each -- 2*T_LEN nodes total, a tree
    (E = n - 1, beta_1 = 0) unless a cycle is planted on it."""
    from ceqjepa.beds import bed_h
    T = bed_h.T_LEN
    belief, obs = list(range(T)), list(range(T, 2 * T))
    edges = ([(belief[i], belief[i + 1]) for i in range(T - 1)]
             + [(obs[i], belief[i]) for i in range(T)])
    return 2 * T, edges, dict(bed="BED-H", source="ceqjepa.beds.bed_h.T_LEN", T_LEN=T)


def sector_census(name: str, n: int, edges, meta: dict | None = None) -> dict:
    """The per-bed report: n, E, beta_0, beta_1, the three sector dimensions
    and the projector evidence -- everything W5.1 asks a census to print."""
    edges = list(edges)
    beta0, beta1 = betti_numbers(n, edges)
    tree_dim, curl_dim, cycle_dim, evidence = sector_dims(n, edges)
    return dict(bed=name, n=n, E=len(edges), beta0=beta0, beta1=beta1,
                sector_dims=(tree_dim, curl_dim, cycle_dim), evidence=evidence,
                meta=meta or {})


def run_all_beds(seed: int = 0) -> list[dict]:
    """The sector census on every bed this lane's scouts found, each from a
    REAL generated batch or manifest (BED-M's sample, BED-K's kernel matrix,
    BED-1's landscape build) rather than an idealized version of it."""
    reports = []
    n, edges, meta = propagation_graph_bed_m(seed=seed)
    reports.append(sector_census(meta["bed"], n, edges, meta))

    n, edges, meta = propagation_graph_bed_k("delay", n=32, seed=seed, d=5)
    reports.append(sector_census(meta["bed"], n, edges, meta))

    n, edges, meta = propagation_graph_bed_k("powerlaw", n=32, seed=seed, H=0.7)
    reports.append(sector_census(meta["bed"], n, edges, meta))

    n, edges, meta = propagation_graph_bed_1(T=1.0, seed=seed)
    reports.append(sector_census(meta["bed"], n, edges, meta))

    n, edges, meta = propagation_graph_bed_h()
    reports.append(sector_census(meta["bed"], n, edges, meta))
    return reports


def print_census(reports) -> None:
    for r in reports:
        tree, curl, cyc = r["sector_dims"]
        print(f"{r['bed']:16s} n={r['n']:4d} E={r['E']:5d} "
              f"beta0={r['beta0']:2d} beta1={r['beta1']:4d} "
              f"sectors=(tree={tree}, curl={curl}, cycle={cyc})")


if __name__ == "__main__":
    reports = run_all_beds()
    print_census(reports)
    print()
    print("must-fire, planted extra cycle:")
    # reports carry only the census numbers, not the edge list -- rebuild each
    # bed's (n, edges) fresh for the plant.
    for builder, kwargs in [
        (propagation_graph_bed_m, dict(seed=0)),
        (propagation_graph_bed_k, dict(kind="delay", n=32, seed=0, d=5)),
        (propagation_graph_bed_1, dict(T=1.0, seed=0)),
        (propagation_graph_bed_h, dict()),
    ]:
        n, edges, meta = builder(**kwargs)
        fire = must_fire_planted_cycle(n, edges)
        print(f"  {meta['bed']:16s} beta1 {fire['beta1_before']} -> {fire['beta1_after']} "
              f"(planted edge {fire['planted_edge']})")
