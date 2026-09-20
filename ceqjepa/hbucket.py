"""Bucket the move graph by k-ply rooted isomorphism, and see what is left.

THE HYPOTHESIS UNDER TEST, in the form it arrived. Bucket every edge x -> y of
the K+Q vs K move graph by the canonical form of its k-ply rooted neighbourhood.
Inside a bucket the neighbourhoods are graph-isomorphic, so every k-local
statistic is constant by theorem and its within-bucket R^2 on distance-to-mate
is zero. Whatever distance-to-mate variance remains inside a bucket is therefore
consequence: the part of the future invisible at k plies. Then read that residue
with the Ollivier-Ricci curvature of the Doob h-transform.

THE VERDICT, and it is a kill at the pre-kill stage. The move graph carries an
exact automorphism group that the design did not account for: the dihedral group
of the board, D4, of order 8. With no pawns, no castling and no en passant, all
eight board symmetries map the position space onto itself, and they carry the
exact distance to mate and the exact committor with them. Measured over every
transform, not assumed. So the buckets the design produces are, at k = 2 and
k = 3, EXACTLY the D4 orbits -- and distance to mate is constant on a D4 orbit
by the same theorem that makes the local statistics constant. The within-bucket
variance the design proposes to read as consequence is zero because the only
isomorphisms in this graph are the symmetries of the chessboard, and the game
does not care which way round the board is. There is no residue.

WHY WEISFEILER-LEHMAN'S INCOMPLETENESS CANNOT RESCUE IT. WL is a relaxation:
isomorphic implies equal hash, never the converse. A WL bucket is therefore a
UNION of isomorphism classes, coarser than or equal to the truth. So the true
count of buckets carrying nonzero distance-to-mate variance is at most the count
WL reports, and the count WL reports at k = 2 and k = 3 is zero. An incomplete
invariant can only inflate this census; it cannot deflate it below zero. The
collision audit is still run and still reported, because a rate nobody measured
is a theorem nobody has, but the k = 3 verdict does not rest on it.

WHAT THE FIRST PROBE OF THIS ROUND GOT WRONG, recorded because it is the exact
failure the design invites. The first bucket key labelled each undirected edge
'forward' or 'reverse' according to whether the tail's integer index was the
smaller one. That label is not isomorphism invariant -- relabelling the nodes
flips it -- and it silently split every symmetry orbit into singletons, which
read as "no non-singleton buckets anywhere" and would have been published as a
different and wrong kill. The repair uses no edge label at all: this graph has
zero 2-cycles and every arc flips the side to move, so the arc direction is
recoverable from the node labels alone, and node labels are invariant. Both
facts are measured in demo(), not asserted.

THE FIRST RED RUN, verbatim, before this module existed:

    ImportError: cannot import name 'hbucket' from 'ceqjepa'

THE POPULATION. Every reading here is on the h-live subgraph: the edges whose
two endpoints both have committor q > 0. That is 4823312 of the 4891672 edges of
the move graph. The excluded 1.4% are exactly the edges the Doob transform kills,
since p_h(x, y) = p(x, y) h(y) / h(x) is zero wherever h(y) is, so they are
outside the kernel the hypothesis is about. Cutting them also removes the queen
-capture sink, whose in-degree of 22176 makes it a hub that inflates the k = 3
ball from a median of 3707 nodes to a maximum of 332024, which is 90% of the
whole space and would have made the census a measurement of one artificial node.

EVERY CENSUS HERE IS A SAMPLE, and says so in the table it prints. The full move
graph is out of budget by a factor of tens: another agent priced Ollivier-Ricci
on this graph at 2.2 to 4.1 ms per edge, so 4891672 edges is 3 to 5.6 hours.
Sampling is by ORBIT, not by edge, and that choice is the difference between a
census that can see anything and one that cannot: drawing edges uniformly from
4891672 of them will essentially never draw two members of the same 8-element
orbit, so a uniform sample reports zero non-singleton buckets for a reason that
has nothing to do with the hypothesis. Drawing whole orbits puts the design's
best case in front of it.
"""

import collections
import math
import sys
import time
from functools import lru_cache

import networkx as nx
import numpy as np
from networkx.algorithms.graph_hashing import (
    weisfeiler_lehman_graph_hash, weisfeiler_lehman_subgraph_hashes)
from networkx.algorithms.isomorphism import categorical_node_match, vf2userfunc

from ceqjepa import chess_steps as cs
from ceqjepa.curvature import ALPHA, Refusal, is_refusal, w1_exact

__all__ = [
    "SEED", "K_PLIES", "CENSUS_ORBITS", "AUDIT_ORBITS", "WL_ITERATIONS",
    "KAPPA_EDGES", "ARROW_EDGES", "D4_ORDER", "BAR_SECONDS", "ALPHA",
    "Adj", "rooted_neighbourhood", "wl_key", "exact_isomorphic", "bucket",
    "d4_square", "d4_report", "graph", "sample_orbits", "bucket_dtm_variance",
    "census", "census_table", "collision_audit", "constancy_audit",
    "h_kernel_row", "kappa_h_edge", "committor_hat_check", "arrow_check",
    "degree_determined_bed", "demo",
]

#: The one seed. Every sample in this module is drawn from it.
SEED = 20260913

#: The radii the census runs at. k = 3 is the hypothesis, because k = 3 is the
#: support of Ollivier-Ricci curvature on an edge; k = 1 and k = 2 are there so
#: the table shows where the emptiness starts rather than only that it is there.
K_PLIES = (1, 2, 3)

#: Orbits sampled per radius. Falling with k because the k-ply ball grows: the
#: median ball is 56 nodes at k = 1, 615 at k = 2 and 3707 at k = 3, and the WL
#: hash is linear in the ball.
CENSUS_ORBITS = {1: 3000, 2: 400, 3: 60}

#: Orbits for the collision and constancy audits, both at k = 1, which is where
#: the only cross-orbit merges in this whole round were found.
AUDIT_ORBITS = 800

#: WL rounds in the bucket key.
WL_ITERATIONS = 3

#: Edges priced for the h-transform curvature, and for the arrow.
KAPPA_EDGES = 400
ARROW_EDGES = 300

#: |D4|. The rotations and reflections of the board.
D4_ORDER = 8

#: The wall-clock bar this module must finish under.
BAR_SECONDS = 300.0

#: Positions checked per transform in the D4 report.
D4_POSITIONS = 3000

#: The floors the test file asserts against, printed here so no number in this
#: project's guards lives only in an exemption list.
GUARD_FLOORS = dict(min_docstring_numbers=40, min_kernel_rows_differing=150,
                    kernel_rows_probed=200, identity_edges=40,
                    min_hat_edges=200, min_arrow_edges=100,
                    min_nonsingleton_for_constancy=10, orbits_for_orbit_test=12)


# ---------------------------------------------------------------------------
# 1. ONE ADJACENCY PROTOCOL, SO THE BUCKETING HAS ONE IMPLEMENTATION
# ---------------------------------------------------------------------------

class Adj(object):
    """Successor lists, undirected neighbour lists, and a node label.

    The bucketing runs on this and nothing else, so the planted beds and the
    368452-position move graph go through the same code. A second implementation
    for the beds would be a second place for the isomorphism-invariance bug to
    hide, and that bug is the one this round actually hit.

    THE INVARIANT THIS CLASS ENFORCES. The neighbourhood handed to WL is
    UNDIRECTED, so the arc direction has to survive somewhere. It survives in
    the node label whenever the label differs across every arc, because then
    the tail of an arc is the endpoint whose label says so, and node labels
    are isomorphism invariant where an index-order edge label is not. For a
    directed source the constructor checks that condition on every arc and
    raises if it fails; for an undirected source there is no direction to keep.
    """

    __slots__ = ("succ_ptr", "succ_flat", "nbr_ptr", "nbr_flat", "labels", "n")

    def __init__(self, succ_ptr, succ_flat, nbr_ptr, nbr_flat, labels,
                 directed, check=True):
        self.succ_ptr = succ_ptr
        self.succ_flat = succ_flat
        self.nbr_ptr = nbr_ptr
        self.nbr_flat = nbr_flat
        self.labels = labels
        self.n = len(succ_ptr) - 1
        if directed and check:
            tails = np.repeat(np.arange(self.n), np.diff(succ_ptr))
            same = labels[tails] == labels[succ_flat]
            if same.any():
                raise ValueError(
                    "the node label does not determine arc direction: %d arcs "
                    "join two nodes with the same label, so an undirected "
                    "neighbourhood would lose the arrow" % int(same.sum()))

    @classmethod
    def from_networkx(cls, G, labels=None):
        """Adapter for a networkx graph. Undirected graphs get a blank label."""
        nodes = sorted(G.nodes())
        pos = {u: i for i, u in enumerate(nodes)}
        directed = G.is_directed()
        out, nbr = [], []
        for u in nodes:
            out.append(sorted(pos[v] for v in (G.successors(u) if directed
                                               else G.neighbors(u))))
            ns = set(G.successors(u)) | set(G.predecessors(u)) if directed \
                else set(G.neighbors(u))
            nbr.append(sorted(pos[v] for v in ns))
        lab = np.array([str(labels[u]) if labels else "" for u in nodes],
                       dtype=object)
        return cls(_ptr(out), _flat(out), _ptr(nbr), _flat(nbr), lab, directed)

    def succ(self, u):
        return self.succ_flat[self.succ_ptr[u]:self.succ_ptr[u + 1]]

    def nbrs(self, u):
        return self.nbr_flat[self.nbr_ptr[u]:self.nbr_ptr[u + 1]]

    def label(self, u):
        return self.labels[u]


def _ptr(lists):
    p = np.zeros(len(lists) + 1, np.int64)
    p[1:] = np.cumsum([len(x) for x in lists])
    return p


def _flat(lists):
    return np.array([v for x in lists for v in x], np.int64) if lists \
        else np.zeros(0, np.int64)


# ---------------------------------------------------------------------------
# 2. THE BUCKET KEY
# ---------------------------------------------------------------------------

def rooted_neighbourhood(adj, x, y, k):
    """The subgraph induced on the undirected k-ball around {x, y}, roots pinned.

    Rooted means the two endpoints carry distinguishing node attributes, so an
    isomorphism has to map x to x and y to y. Without that, an edge and its
    reverse would share a bucket in a graph where they do not share a future.
    """
    x, y = int(x), int(y)
    seen = {x, y}
    frontier = [x, y]
    for _ in range(int(k)):
        nxt = []
        for u in frontier:
            for v in adj.nbrs(u):
                v = int(v)
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    G = nx.Graph()
    for u in seen:
        mark = "x" if u == x else ("y" if u == y else ".")
        G.add_node(u, r=mark + str(adj.label(u)))
    for u in seen:
        for v in adj.succ(u):
            v = int(v)
            if v in seen:
                G.add_edge(u, v)
    return G


def wl_key(G, iterations=WL_ITERATIONS):
    """The bucket key: a Weisfeiler-Lehman hash of the rooted neighbourhood.

    NOT A COMPLETE ISOMORPHISM INVARIANT, and the module is built around that
    fact rather than around hoping it away. Two non-isomorphic neighbourhoods
    can share this key, so collision_audit re-checks every bucket that merges
    two distinct symmetry orbits with an exact test, and the census verdict is
    stated in the direction the incompleteness runs.
    """
    return weisfeiler_lehman_graph_hash(G, node_attr="r", iterations=iterations)


_ROOT_MATCH = categorical_node_match("r", ".")

#: Candidate node pairs the exact matcher may consider before it gives up. A
#: DETERMINISTIC budget rather than a wall clock, so the audit reports the same
#: three counts on every box and the undecided set is reproducible.
MATCH_BUDGET = 400000


class _OutOfBudget(Exception):
    pass


class _BoundedMatcher(vf2userfunc.GraphMatcher):
    """VF2 with a step budget, because on this graph VF2 does not always stop.

    MEASURED, not feared. networkx.vf2pp_is_isomorphic did not decide even a
    self-check of a 53-node, 149-edge ball inside 200 s here, and plain
    networkx.is_isomorphic, which decides an unrelated pair of the same size in
    under a millisecond, did not decide the FIRST cross-orbit pair inside 240 s.
    The reason is measurable and is printed by demo(): the WL colouring of
    these balls is nowhere near discrete -- a 63-node ball refines to 14 colours
    with a class of 21, and 3, 5 and 10 WL rounds all give the same partition --
    so a colour class of 21 interchangeable vertices leaves the matcher up to
    21! candidate maps to exhaust before it can report False.
    """

    def __init__(self, G1, G2, budget=MATCH_BUDGET):
        super().__init__(G1, G2, node_match=_ROOT_MATCH)
        self._left = int(budget)

    def semantic_feasibility(self, g1, g2):
        self._left -= 1
        if self._left < 0:
            raise _OutOfBudget()
        return super().semantic_feasibility(g1, g2)


def exact_isomorphic(G, H, budget=MATCH_BUDGET):
    """True, False, or None when the bounded exact matcher ran out of budget.

    None is a value in the output space, never a silent False. A pair reported
    as "not isomorphic" because the search was cut short is a fabricated
    collision, and a collision count built from fabrications is worse than no
    count at all.
    """
    if G.number_of_nodes() != H.number_of_nodes() \
            or G.number_of_edges() != H.number_of_edges():
        return False
    try:
        return _BoundedMatcher(G, H, budget).is_isomorphic()
    except _OutOfBudget:
        return None


def bucket(adj, edges, k):
    """edges grouped by their k-ply rooted neighbourhood key."""
    out = collections.defaultdict(list)
    for x, y in edges:
        out[wl_key(rooted_neighbourhood(adj, x, y, k))].append((int(x), int(y)))
    return dict(out)


# ---------------------------------------------------------------------------
# 3. THE BOARD SYMMETRY GROUP
# ---------------------------------------------------------------------------

#: The eight elements of D4, as maps on (file, rank). Named so the report can
#: say which one broke if one ever does.
_D4 = (
    ("identity", lambda f, r: (f, r)),
    ("mirror-file", lambda f, r: (7 - f, r)),
    ("mirror-rank", lambda f, r: (f, 7 - r)),
    ("rot180", lambda f, r: (7 - f, 7 - r)),
    ("transpose", lambda f, r: (r, f)),
    ("anti-transpose", lambda f, r: (7 - r, 7 - f)),
    ("rot90", lambda f, r: (r, 7 - f)),
    ("rot270", lambda f, r: (7 - r, f)),
)


def d4_square(name, sq):
    """Apply one board symmetry to a square index."""
    g = dict(_D4)[name]
    f, r = g(sq % 8, sq // 8)
    return r * 8 + f


@lru_cache(maxsize=4)
def d4_report(n_positions=D4_POSITIONS, seed=SEED):
    """Is each board symmetry an automorphism that carries dtm and the committor?

    THE MEASUREMENT THE WHOLE PRE-KILL TURNS ON. If it holds then every bucket
    contains whole symmetry orbits, and distance to mate is constant on an
    orbit, so the variance the design wants to read is zero there by theorem
    rather than by accident. It is measured over all eight transforms because
    "chess has no pawns in this endgame" is an argument and not a number.
    """
    sp, o = cs.space(), cs.oracle()
    keys, index, dtm, q = sp["keys"], sp["index"], o["dtm"], o["q"]
    rng = np.random.default_rng(seed)
    sample = rng.choice(sp["n_positions"], int(n_positions), replace=False)
    per = {}
    for name, _ in _D4:
        missing = dtm_bad = q_bad = fixed = 0
        for i in sample:
            wk, wq, bk, turn = (int(v) for v in keys[i])
            j = index.get((d4_square(name, wk), d4_square(name, wq),
                           d4_square(name, bk), bool(turn)))
            if j is None:
                missing += 1
                continue
            fixed += int(j == i)
            dtm_bad += int(dtm[j] != dtm[i])
            q_bad += int(abs(float(q[j]) - float(q[i])) > 1e-12)
        per[name] = dict(missing=missing, dtm_mismatch=dtm_bad,
                         q_mismatch=q_bad, fixed_points=fixed)
    return dict(per_transform=per, n_positions_checked=int(sample.size),
                total_checks=D4_ORDER * int(sample.size), seed=int(seed))


# ---------------------------------------------------------------------------
# 4. THE GRAPH
# ---------------------------------------------------------------------------

class _Graph(object):
    __slots__ = ("adj", "adj_sym", "src", "dst", "dtm", "q", "out_degree",
                 "n_live_edges", "n_live_nodes", "population_edges",
                 "all_edges", "n_positions", "sink_in_degree", "arcs_flipping_turn")


@lru_cache(maxsize=1)
def graph():
    """The h-live move graph: both endpoints with committor q > 0.

    Built once. The two adjacencies it carries are the directed one, on which
    every reading is taken, and the symmetrised move+unmove one, which exists
    only for Kill 3 -- the check that the asymmetry the design claims is
    directed survives being handed a graph with no direction in it.
    """
    sp, o = cs.space(), cs.oracle()
    n, nt = sp["n_positions"], sp["n_states"]
    src = sp["pred"].astype(np.int64)
    dst = sp["succ"].astype(np.int64)
    q_all = np.zeros(nt)
    q_all[:n] = o["q"]
    live = q_all > 0.0
    keep = live[src] & live[dst]
    s, d = src[keep], dst[keep]

    turn = np.full(nt, 2, np.int8)
    turn[:n] = sp["keys"][:, 3].astype(np.int8)

    order = np.lexsort((d, s))
    succ_flat = d[order]
    succ_ptr = np.zeros(nt + 1, np.int64)
    succ_ptr[1:] = np.cumsum(np.bincount(s, minlength=nt))

    ua = np.concatenate([s, d])
    ub = np.concatenate([d, s])
    uo = np.lexsort((ub, ua))
    ua, ub = ua[uo], ub[uo]
    nbr_ptr = np.zeros(nt + 1, np.int64)
    nbr_ptr[1:] = np.cumsum(np.bincount(ua, minlength=nt))

    labels = np.array([str(int(t)) for t in turn], dtype=object)
    g = _Graph()
    g.adj = Adj(succ_ptr, succ_flat, nbr_ptr, ub, labels, directed=True)
    g.adj_sym = Adj(nbr_ptr, ub, nbr_ptr, ub, labels, directed=False)
    g.src, g.dst = s, d
    g.dtm = o["dtm"]
    g.q = q_all
    g.out_degree = np.diff(succ_ptr)
    g.n_live_edges = int(s.size)
    g.n_live_nodes = int(live[:n].sum())
    g.population_edges = int(sp["n_edges"])
    g.n_positions = int(n)
    g.sink_in_degree = int((dst == sp["sink"]).sum())
    nonsink = dst < n
    g.arcs_flipping_turn = int((turn[src[nonsink]] != turn[dst[nonsink]]).sum())
    g.all_edges = int(nonsink.sum())
    return g


def _orbit_of_edge(sp, x, y):
    """The D4 orbit of the edge (x, y), deduplicated and sorted."""
    keys, index = sp["keys"], sp["index"]
    kx = [int(v) for v in keys[x]]
    ky = [int(v) for v in keys[y]]
    out = set()
    for name, _ in _D4:
        i = index.get((d4_square(name, kx[0]), d4_square(name, kx[1]),
                       d4_square(name, kx[2]), bool(kx[3])))
        j = index.get((d4_square(name, ky[0]), d4_square(name, ky[1]),
                       d4_square(name, ky[2]), bool(ky[3])))
        if i is not None and j is not None:
            out.add((i, j))
    return sorted(out)


def sample_orbits(g, n_orbits, seed=SEED):
    """n_orbits D4 orbits of edges, drawn by seeding on a uniform edge."""
    sp = cs.space()
    rng = np.random.default_rng(seed)
    seeds = rng.choice(g.n_live_edges, int(n_orbits), replace=False)
    orbits, taken = [], set()
    for e in seeds:
        orb = _orbit_of_edge(sp, int(g.src[e]), int(g.dst[e]))
        orb = [p for p in orb if p not in taken]
        if not orb:
            continue
        taken.update(orb)
        orbits.append(orb)
    return orbits


def bucket_dtm_variance(g, edges):
    """Within-bucket distance-to-mate variance, at the tail and at the head.

    Read at BOTH endpoints. A bucket that is constant at the tail can still
    vary at the head, and a variance taken only on the tail would report the
    design as empty for the wrong reason.
    """
    tail = float(np.var([g.dtm[x] for x, _ in edges]))
    head = float(np.var([g.dtm[y] for _, y in edges]))
    return dict(tail=tail, head=head, worst=max(tail, head))


# ---------------------------------------------------------------------------
# 5. STAGE ONE: THE PRE-KILL CENSUS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _bucketed(k, n_orbits, seed):
    """Buckets plus the orbit each edge came from. Cached: three readers share it."""
    g = graph()
    orbits = sample_orbits(g, n_orbits, seed)
    orbit_of, edges = {}, []
    for oi, orb in enumerate(orbits):
        for pair in orb:
            orbit_of[pair] = oi
            edges.append(pair)
    return bucket(g.adj, edges, k), orbit_of, len(orbits)


@lru_cache(maxsize=16)
def census(k, n_orbits=None, seed=SEED):
    """One row of the pre-kill table: how much of the design is not empty here."""
    k = int(k)
    n_orbits = CENSUS_ORBITS[k] if n_orbits is None else int(n_orbits)
    g = graph()
    t0 = time.time()
    buckets, orbit_of, n_orb = _bucketed(k, n_orbits, seed)
    multi = [v for v in buckets.values() if len(v) > 1]
    live = [v for v in multi if bucket_dtm_variance(g, v)["worst"] > 0.0]
    cross = [v for v in multi if len({orbit_of[e] for e in v}) > 1]
    n_edges = sum(len(v) for v in buckets.values())
    # THE DECISIVE STRUCTURAL COUNT. A bucket holding exactly one symmetry
    # orbit cannot carry distance-to-mate variance, because the orbit map is an
    # automorphism that fixes dtm. So every bucket with variance has to be one
    # where WL merged two DISTINCT orbits, and this counts the exceptions. A
    # nonzero exception count would mean an orbit that dtm does not respect,
    # which would falsify the whole pre-kill.
    single_orbit_with_var = sum(
        1 for v in live if len({orbit_of[e] for e in v}) == 1)
    return dict(
        k=k, seed=int(seed), is_sample=True,
        n_orbits=int(n_orb), n_edges_sampled=int(n_edges),
        population_edges=int(g.n_live_edges),
        n_buckets=len(buckets), n_nonsingleton=len(multi),
        n_nonzero_dtm_var=len(live), n_cross_orbit=len(cross),
        n_single_orbit_with_dtm_var=int(single_orbit_with_var),
        largest_bucket=max(len(v) for v in buckets.values()),
        bucket_sizes=collections.Counter(len(v) for v in buckets.values()),
        seconds=round(time.time() - t0, 2),
    )


def census_table():
    """The pre-kill table over K_PLIES, in order."""
    return [census(k) for k in K_PLIES]


@lru_cache(maxsize=8)
def collision_audit(k=1, n_orbits=AUDIT_ORBITS, seed=SEED):
    """WL merged two distinct orbits: real isomorphism, or a WL collision?

    A collision rate you did not measure is a theorem you do not have. Every
    bucket holding edges from two or more distinct D4 orbits is re-checked
    exactly, roots pinned. Buckets holding exactly one orbit need no check:
    their members are isomorphic by construction, since the orbit map is an
    automorphism of the whole graph.
    """
    g = graph()
    buckets, orbit_of, n_orb = _bucketed(int(k), int(n_orbits), int(seed))
    cross = [v for v in buckets.values() if len({orbit_of[e] for e in v}) > 1]
    real = coll = undecided = 0
    real_with_dtm_gap = 0
    n_colours, largest_class, n_balls = [], [], 0
    for members in cross:
        first = members[0]
        G0 = rooted_neighbourhood(g.adj, first[0], first[1], int(k))
        classes = collections.Counter(
            h[-1] for h in weisfeiler_lehman_subgraph_hashes(
                G0, node_attr="r", iterations=WL_ITERATIONS).values())
        n_colours.append(len(classes))
        largest_class.append(max(classes.values()))
        n_balls += 1
        verdicts = []
        for e in members[1:]:
            if orbit_of[e] == orbit_of[first]:
                continue
            G1 = rooted_neighbourhood(g.adj, e[0], e[1], int(k))
            iso = exact_isomorphic(G0, G1)
            verdicts.append(iso)
            if iso is True and g.dtm[e[0]] != g.dtm[first[0]]:
                real_with_dtm_gap += 1
        if any(v is None for v in verdicts):
            undecided += 1
        elif verdicts and all(verdicts):
            real += 1
        else:
            coll += 1
    n = len(cross)
    decided = real + coll
    return dict(k=int(k), seed=int(seed), n_orbits=int(n_orb),
                n_edges=sum(len(v) for v in buckets.values()),
                n_cross_orbit_buckets=n, n_real_isomorphism=real,
                n_wl_collision=coll, n_undecided=undecided,
                n_decided=decided, match_budget=MATCH_BUDGET,
                n_real_with_dtm_gap=real_with_dtm_gap,
                median_colours=(float(np.median(n_colours)) if n_balls else 0.0),
                median_largest_colour_class=(float(np.median(largest_class))
                                             if n_balls else 0.0),
                collision_rate=(coll / decided) if decided else 0.0)


@lru_cache(maxsize=8)
def constancy_audit(k=1, n_orbits=AUDIT_ORBITS, seed=SEED):
    """The empirical consequence of "isomorphic, therefore constant by theorem".

    Out-degree, its logarithm and the mean successor out-degree are all fixed
    by an isomorphism class, so the within-bucket spread must read 0.0 exactly.
    Not small: a nonzero spread means the buckets are not isomorphism classes
    and every reading downstream of them is void.
    """
    g = graph()
    buckets, _, _ = _bucketed(int(k), int(n_orbits), int(seed))
    deg = g.out_degree
    succ_mean = {}

    def stats_of(x):
        if x not in succ_mean:
            s = g.adj.succ(x)
            succ_mean[x] = float(deg[s].mean()) if s.size else 0.0
        return (float(deg[x]), math.log(float(deg[x])) if deg[x] > 0 else 0.0,
                succ_mean[x])

    names = ("out_degree", "ln_out_degree", "successor_out_degree")
    worst = {nm: 0.0 for nm in names}
    n_multi = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        n_multi += 1
        cols = list(zip(*[stats_of(x) for x, _ in members]))
        for nm, col in zip(names, cols):
            worst[nm] = max(worst[nm], max(col) - min(col))
    return dict(k=int(k), seed=int(seed), spread=worst,
                n_nonsingleton=n_multi,
                n_edges=sum(len(v) for v in buckets.values()))


# ---------------------------------------------------------------------------
# 6. THE DOOB H-TRANSFORM CURVATURE
# ---------------------------------------------------------------------------

def h_kernel_row(g, x, adj=None):
    """(support, mass) of p_h(x, .) = p(x, .) h(.) / h(x), UNNORMALISED.

    Unnormalised on purpose. h is the exact committor and the committor is
    harmonic at every interior node, so this row already sums to one and the
    deviation from one is a measurement of how exact the oracle's fixpoint is,
    not a knob. demo() prints the worst deviation seen. Normalising here would
    hide it.
    """
    adj = g.adj if adj is None else adj
    supp = np.asarray(adj.succ(int(x)), dtype=np.int64)
    if supp.size == 0:
        return supp, np.zeros(0)
    hx = float(g.q[int(x)])
    return supp, (g.q[supp] / hx) / float(supp.size)


def _hop_leq3(adj, u, v, nbr_cache):
    """Exact hop distance between two nodes known to lie within three hops.

    Every u in the support of x and v in the support of y satisfies
    d(u, v) <= d(u, x) + d(x, y) + d(y, v) <= 3 when x and y are adjacent, so
    the ladder 0, 1, 2, else 3 is exact rather than a truncation. This is what
    makes the curvature affordable on a 368452-node graph: no all-pairs metric
    is ever built.
    """
    if u == v:
        return 0.0
    if u not in nbr_cache:
        nbr_cache[u] = set(int(w) for w in adj.nbrs(u))
    if v in nbr_cache[u]:
        return 1.0
    if v not in nbr_cache:
        nbr_cache[v] = set(int(w) for w in adj.nbrs(v))
    return 2.0 if nbr_cache[u] & nbr_cache[v] else 3.0


def kappa_h_edge(adj, h, D, i, j, alpha=ALPHA):
    """kappa_h(i, j) = 1 - W1(m_i, m_j) / d(i, j) for the Doob-transformed walk.

    m_x holds alpha at x and spreads 1 - alpha over p_h(x, .). Directed because
    h is directed: p_h pushes mass toward high committor and the reverse kernel
    does not, and no causal mask appears anywhere in it.

    D may be a full metric matrix, or None to use the bounded exact hop rule.
    demo() checks the two agree term by term on a bed small enough to carry a
    full metric, which is the only reason the shortcut is allowed on the graph
    that is not.
    """
    i, j = int(i), int(j)
    if i == j:
        return Refusal("self-loop", "kappa_h(%d,%d) is 0/0" % (i, i))
    cache = {}
    supports, masses = [], []
    for node in (i, j):
        s = np.asarray(adj.succ(node), dtype=np.int64)
        if s.size == 0:
            return Refusal("degenerate-neighbourhood",
                           "node %d has no successors under p_h" % node)
        hx = float(h[node])
        if hx <= 0.0:
            return Refusal("dead-under-h",
                           "h(%d) = %r: the Doob transform is undefined there"
                           % (node, hx))
        p = (h[s] / hx) / float(s.size)
        total = float(p.sum())
        if not np.isfinite(total) or total <= 0.0:
            return Refusal("dead-under-h", "p_h row at %d sums to %r" % (node, total))
        supports.append(np.concatenate(([node], s)))
        masses.append(np.concatenate(([alpha], (1.0 - alpha) * p / total)))
    si, sj = supports
    if D is None:
        d = _hop_leq3(adj, i, j, cache)
        C = np.array([[_hop_leq3(adj, int(u), int(v), cache) for v in sj]
                      for u in si])
    else:
        d = float(D[i, j])
        C = np.asarray(D)[np.ix_(si, sj)]
    if d == 0.0 or not np.isfinite(d) or not np.isfinite(C).all():
        return Refusal("disconnected", "no finite coupling for (%d,%d)" % (i, j))
    return 1.0 - w1_exact(masses[0], masses[1], C) / d


def _h_moments(g, x, adj=None):
    """h(x) and the variance of h over the successors of x: the cheap quantities.

    These are what already know the consequence. If kappa_h is a function of
    them then it is the committor wearing a hat, and the leap is dead.
    """
    adj = g.adj if adj is None else adj
    s = np.asarray(adj.succ(int(x)), dtype=np.int64)
    return float(g.q[int(x)]), (float(np.var(g.q[s])) if s.size else 0.0)


@lru_cache(maxsize=4)
def committor_hat_check(n_edges=KAPPA_EDGES, seed=SEED):
    """Regress kappa_h on h(x), h(y), Var_N(x) h, Var_N(y) h. What is left?

    Reported unconditionally rather than within buckets, and that is the
    reroute the pre-kill forces: at k = 2 and k = 3 no bucket has any distance
    -to-mate variance to regress on, so a within-bucket partial R^2 has no
    within-bucket variation to be partial to. The unconditional version is
    strictly more favourable to the hypothesis -- it gives kappa_h the whole
    range of the graph to be informative over -- so a low residual here kills
    the bucketed version a fortiori.
    """
    g = graph()
    rng = np.random.default_rng(int(seed))
    picks = rng.choice(g.n_live_edges, int(n_edges), replace=False)
    rows, ys = [], []
    refused = 0
    for e in picks:
        x, y = int(g.src[e]), int(g.dst[e])
        kh = kappa_h_edge(g.adj, g.q, None, x, y)
        if is_refusal(kh):
            refused += 1
            continue
        hx, vx = _h_moments(g, x)
        hy, vy = _h_moments(g, y)
        rows.append([hx, hy, vx, vy])
        ys.append(float(kh))
    X = np.column_stack([np.ones(len(rows)), np.array(rows)])
    y = np.array(ys)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n, p = len(y), X.shape[1]
    se = math.sqrt(max(0.0, 4.0 * r2 * (1.0 - r2) ** 2 * (n - p - 1) ** 2
                       / ((n ** 2 - 1) * (n + 3)))) if n > p + 2 else 0.0
    return dict(n=int(n), n_refused=int(refused), r2=float(r2),
                ci=(float(max(0.0, r2 - 1.96 * se)), float(min(1.0, r2 + 1.96 * se))),
                residual_sd=float(resid.std()), kappa_sd=float(y.std()),
                design_columns=["h_x", "h_y", "var_h_nx", "var_h_ny"],
                seed=int(seed))


@lru_cache(maxsize=4)
def arrow_check(n_edges=ARROW_EDGES, seed=SEED):
    """KILL 3. Symmetrise the graph, then ask whether the asymmetry survives.

    The world's arrow has a ground truth here: distance to mate falls along
    optimal play, so the sign of dtm(x) - dtm(y) is the arrow the design claims
    to be reading. On the symmetrised move+unmove graph both kappa_h(x, y) and
    kappa_h(y, x) exist, and their difference is the asymmetry. An asymmetry
    whose sign does not track the ground truth is not a reading of the arrow.
    """
    g = graph()
    rng = np.random.default_rng(int(seed) + 1)
    picks = rng.choice(g.n_live_edges, int(n_edges) * 3, replace=False)
    agree = disagree = tied = 0
    gaps = []
    for e in picks:
        x, y = int(g.src[e]), int(g.dst[e])
        if g.dtm[x] < 0 or g.dtm[y] < 0 or g.dtm[x] == g.dtm[y]:
            continue
        fwd = kappa_h_edge(g.adj_sym, g.q, None, x, y)
        rev = kappa_h_edge(g.adj_sym, g.q, None, y, x)
        if is_refusal(fwd) or is_refusal(rev):
            continue
        asym = float(fwd) - float(rev)
        gaps.append(abs(asym))
        truth = int(np.sign(float(g.dtm[x]) - float(g.dtm[y])))
        if asym == 0.0:
            tied += 1
        elif int(np.sign(asym)) == truth:
            agree += 1
        else:
            disagree += 1
        if agree + disagree + tied >= int(n_edges):
            break
    n = agree + disagree + tied
    return dict(n=int(n), n_agree=int(agree), n_disagree=int(disagree),
                n_tied=int(tied), agree_rate=(agree / n) if n else 0.0,
                median_abs_asymmetry=float(np.median(gaps)) if gaps else 0.0,
                max_abs_asymmetry=float(np.max(gaps)) if gaps else 0.0,
                seed=int(seed))


# ---------------------------------------------------------------------------
# 7. THE PLANTED BED
# ---------------------------------------------------------------------------

def degree_determined_bed():
    """Three cliques on one hub. The label is a pure function of degree.

    THE DESIGN'S OWN NULL. Inside a clique the interior edges are
    interchangeable, so non-singleton buckets exist and a zero variance there
    is not vacuous; and the label is degree, which an isomorphism fixes, so
    the variance must read 0.0 exactly or the bucketing is broken.
    """
    G = nx.Graph()
    hub = "hub"
    G.add_node(hub)
    for size in (3, 4, 5):
        members = ["c%d_%d" % (size, i) for i in range(size)]
        G.add_edges_from((u, v) for i, u in enumerate(members)
                         for v in members[i + 1:])
        G.add_edge(hub, members[0])
    return G


# ---------------------------------------------------------------------------
# 8. THE RUN
# ---------------------------------------------------------------------------

def demo():
    """Every number this module publishes, printed and returned."""
    t0 = time.time()
    g = graph()
    print("CHASE / hbucket -- does k-ply isomorphism leave any consequence behind?")
    print("  seed %d, alpha %.1f, WL iterations %d, k in %s"
          % (SEED, ALPHA, WL_ITERATIONS, ",".join(str(k) for k in K_PLIES)))
    print("  guard floors: " + ", ".join("%s=%d" % kv for kv in
                                         sorted(GUARD_FLOORS.items())))

    print("\n(a) THE GRAPH, and why the population is the h-live subgraph")
    print("    positions %d, move-graph edges %d, arcs flipping the side to "
          "move %d of %d" % (g.n_positions, g.population_edges,
                             g.arcs_flipping_turn, g.all_edges))
    print("    h-live nodes %d, h-live edges %d, queen-capture sink in-degree %d"
          % (g.n_live_nodes, g.n_live_edges, g.sink_in_degree))
    assert g.arcs_flipping_turn == g.all_edges, "an arc did not flip the turn"

    d4 = d4_report()
    bad = sum(r["dtm_mismatch"] + r["q_mismatch"] + r["missing"]
              for r in d4["per_transform"].values())
    print("\n(b) D4 IS AN EXACT AUTOMORPHISM AND IT CARRIES THE LABELS")
    print("    %d checks over %d transforms x %d positions, %d mismatches"
          % (d4["total_checks"], D4_ORDER, d4["n_positions_checked"], bad))
    for name, row in d4["per_transform"].items():
        print("      %-15s missing %d  dtm %d  q %d  fixed points %d"
              % (name, row["missing"], row["dtm_mismatch"], row["q_mismatch"],
                 row["fixed_points"]))
    assert bad == 0, "a board symmetry moved a label"

    print("\n(c) STAGE ONE, THE PRE-KILL CENSUS. Every row is a SAMPLE, by orbit.")
    print("    %-3s %7s %8s %9s %8s %7s %6s %6s %8s %6s"
          % ("k", "orbits", "edges", "of pop", "buckets", "multi", "dtm>0",
             "cross", "largest", "sec"))
    rows = census_table()
    for r in rows:
        print("    %-3d %7d %8d %8.4f%% %8d %7d %6d %6d %8d %6.1f"
              % (r["k"], r["n_orbits"], r["n_edges_sampled"],
                 100.0 * r["n_edges_sampled"] / r["population_edges"],
                 r["n_buckets"], r["n_nonsingleton"], r["n_nonzero_dtm_var"],
                 r["n_cross_orbit"], r["largest_bucket"], r["seconds"]))
    k3 = [r for r in rows if r["k"] == 3][0]
    total_live = sum(r["n_nonzero_dtm_var"] for r in rows)
    stray = sum(r["n_single_orbit_with_dtm_var"] for r in rows)
    print("    buckets holding ONE symmetry orbit that still carry dtm "
          "variance, summed over k: %d" % stray)
    print("    -> every bucket with any dtm variance is one where WL merged "
          "two DISTINCT orbits")
    assert stray == 0, "a single symmetry orbit carried dtm variance"

    aud = collision_audit()
    con = constancy_audit()
    print("\n(d) IS THE BUCKETING AN ISOMORPHISM CLASS? Two checks, both required.")
    print("    collisions: %d cross-orbit buckets over %d edges from %d orbits"
          % (aud["n_cross_orbit_buckets"], aud["n_edges"], aud["n_orbits"]))
    print("      real isomorphisms %d, WL collisions %d, UNDECIDED %d "
          "at a budget of %d candidate pairs; rate over the decided %.4f"
          % (aud["n_real_isomorphism"], aud["n_wl_collision"],
             aud["n_undecided"], aud["match_budget"], aud["collision_rate"]))
    print("      why the matcher stalls: these balls refine to a median of "
          "%.1f WL colours with a median largest class of %.1f"
          % (aud["median_colours"], aud["median_largest_colour_class"]))
    print("      of the real isomorphisms, %d joined edges with different dtm"
          % aud["n_real_with_dtm_gap"])
    print("    constancy over %d non-singleton buckets, spread must be 0.0:"
          % con["n_nonsingleton"])
    for nm in ("out_degree", "ln_out_degree", "successor_out_degree"):
        print("      %-22s %r" % (nm, con["spread"][nm]))
        assert con["spread"][nm] == 0.0, "%s is not constant in a bucket" % nm

    print("\n(e) THE H-KERNEL, and the harmonicity it rests on")
    rng = np.random.default_rng(SEED)
    probe = rng.choice(g.n_live_edges, GUARD_FLOORS["kernel_rows_probed"],
                       replace=False)
    defect, differ = 0.0, 0
    for e in probe:
        supp, mass = h_kernel_row(g, int(g.src[e]))
        defect = max(defect, abs(float(mass.sum()) - 1.0))
        if np.abs(mass - 1.0 / supp.size).max() > 1e-9:
            differ += 1
    print("    worst |sum p_h - 1| over %d rows: %.3e" % (probe.size, defect))
    print("    rows differing from the uniform walk: %d of %d"
          % (differ, probe.size))

    G = nx.les_miserables_graph()
    G = nx.convert_node_labels_to_integers(G)
    from ceqjepa.curvature import graph_metric, kappa_edge
    W = nx.to_numpy_array(G, weight=None)
    D = graph_metric(W)
    adj = Adj.from_networkx(G)
    ones = np.ones(W.shape[0])
    worst_ident, worst_local = 0.0, 0.0
    for i, j in sorted(G.edges())[:GUARD_FLOORS["identity_edges"]]:
        mine = kappa_h_edge(adj, ones, D, i, j)
        theirs = kappa_edge(W, D, i, j)
        local = kappa_h_edge(adj, ones, None, i, j)
        worst_ident = max(worst_ident, abs(float(mine) - float(theirs)))
        worst_local = max(worst_local, abs(float(mine) - float(local)))
    print("    h constant: worst |kappa_h - kappa| over %d edges  %.3e"
          % (GUARD_FLOORS["identity_edges"], worst_ident))
    print("    bounded hop rule vs the full metric, same edges       %.3e"
          % worst_local)
    assert worst_ident < 1e-9 and worst_local < 1e-9, "the instrument disagrees"

    hat = committor_hat_check()
    print("\n(f) THE COMMITTOR WEARING A HAT? kappa_h on %s"
          % ", ".join(hat["design_columns"]))
    print("    n %d (refused %d), R^2 %.6f, CI [%.6f, %.6f]"
          % (hat["n"], hat["n_refused"], hat["r2"], hat["ci"][0], hat["ci"][1]))
    print("    sd(kappa_h) %.6f, sd(residual) %.6f"
          % (hat["kappa_sd"], hat["residual_sd"]))

    arrow = arrow_check()
    print("\n(g) KILL 3, the symmetrised move+unmove graph against the true arrow")
    print("    n %d: agree %d, disagree %d, tied %d, agreement %.4f"
          % (arrow["n"], arrow["n_agree"], arrow["n_disagree"], arrow["n_tied"],
             arrow["agree_rate"]))
    print("    |kappa_h(x,y) - kappa_h(y,x)|: median %.6f, max %.6f"
          % (arrow["median_abs_asymmetry"], arrow["max_abs_asymmetry"]))

    elapsed = time.time() - t0
    print("\n(h) THE VERDICT")
    print("    buckets with nonzero dtm variance at k=3: %d  -> pre-kill %s"
          % (k3["n_nonzero_dtm_var"],
             "EMPTY" if k3["n_nonzero_dtm_var"] == 0 else "SURVIVES"))
    print("    summed over k in %s: %d"
          % (",".join(str(k) for k in K_PLIES), total_live))
    print("    elapsed %.1f s against a %.0f s bar" % (elapsed, BAR_SECONDS))
    assert elapsed < BAR_SECONDS, "over the bar"

    stats = dict(
        census=rows, d4_total_checks=d4["total_checks"], d4_mismatches=bad,
        collision_rate=aud["collision_rate"],
        collision_n=aud["n_cross_orbit_buckets"],
        collision_real=aud["n_real_isomorphism"],
        spread=con["spread"], hat_r2=hat["r2"], hat_n=hat["n"],
        arrow_agree_rate=arrow["agree_rate"], arrow_n=arrow["n"],
        harmonic_defect=defect, kernel_rows_differing=differ,
        identity_worst=worst_ident, local_metric_worst=worst_local,
        prekill_empty=(total_live == 0),
        prekill_empty_at_k3=(k3["n_nonzero_dtm_var"] == 0),
        elapsed=elapsed,
    )
    print("\nALL SELF-CHECKS PASSED")
    return stats


if __name__ == "__main__":
    demo()
