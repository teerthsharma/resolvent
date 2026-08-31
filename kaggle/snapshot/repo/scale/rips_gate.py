"""The E4 admission gates: does the Rips connectivity label need an iteration?

WHAT THIS DECIDES. `LOOP_PROMPT.md` section 1.7d admits `ceq/rips.py` as an E4
candidate CONDITIONALLY, behind two gates that must both print before any E4 number
is read:

  (a) TRUNCATION. A `k`-hop reading of the label must be bounded away from the label
      on the critical cases and must tighten with `k`. A 1-hop reading that gets the
      label makes E4 a third static task.
  (b) THE DECODER MUST-FIRE. A static decoder on local degree statistics must FAIL on
      the critical cases. It was expected to PASS on `StableSparse_S2Rips_64` and
      `SupercriticalDense_S2Rips_256`, and that passing half IS the control: a decoder
      that fails everywhere fails for the wrong reason. If the decoder also passes at
      criticality, E4 is another adjacency and is struck without appeal.

The component COUNT is banned as a feature and appears nowhere below. Every feature
here is a degree or a ball size, both strictly local.

WHAT THIS MODULE MEASURED. Gate (a) passes. Gate (b) strikes E4, three times over:

  1. `SupercriticalDense_S2Rips_256` cannot serve as the passing half at all. All 256
     of its nodes lie in one component, so every node pair carries the label 1, the
     label's standard deviation is exactly 0 and NRMSE is undefined. The same holds
     for `GroundedStaticRepeated_S2Rips_256`. The prescribed control is vacuous.
  2. The other prescribed passing case, `StableSparse_S2Rips_64`, does not pass
     either: degree-only reads 0.89 there, no better than the mean predictor. Both
     halves of the prescribed control are unusable as written.
  3. The degree-only decoder PASSES AT CRITICALITY on `CriticalLarge_S2Rips_1024`
     under a balanced marginal draw. That is the strike condition, verbatim.

WHY, AND THIS IS THE PART THAT TRANSFERS. `_add_critical_bridge` joins the two
NEAREST distinct components. On `S^2` at these degrees the nearest cross-component
pair is always a speck against the giant: the merged sizes are `3 x 222` and
`1016 x 4`, measured, not argued. So the post-bridge partition is (giant, tiny
fringe) and component membership degenerates into "is my own ball small", which a
radius-2 or radius-3 decoder reads without traversing anything. The 62-hop diameter
gate (a) measures is the GIANT's diameter, and no decoder ever has to cross it: gate
(a) truncated the wrong reading. Reachability from `i` to `j` needs the diameter;
deciding the label does not.

THE REROUTE, AND IT IS ON THE SAME CODE PATH. `ceq.rips.rerouted_corpus()` changes
exactly one thing -- which two components the single bridge edge joins -- and joins
the two LARGEST instead. On `LargestJoin_S2Rips_1024` the merge is `32 x 30`, neither
endpoint's ball saturates, and every static local decoder collapses to the mean
predictor out to radius 5 while the planted control still fires on the same
instances, same decoder, same split. That is the E4 the gates were asking for.
"""
from __future__ import annotations

import math
import random

import numpy as np

from ceq.rips import components, corpus, rerouted_corpus, rips_edges, sample_sphere

#: Pre-registered bars, fixed before any number below was read. A mean predictor
#: scores exactly 1.0, so `PASS_BAR` is "the decoder removed three quarters of the
#: label variance" and `FAIL_BAR` is "the decoder is doing essentially nothing".
PASS_BAR = 0.5
FAIL_BAR = 0.9

#: Rungs the truncation ladder prints at.
LADDER_KS = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def adjacency(node_count: int, edges) -> list[set[int]]:
    """Simple undirected adjacency: self-loops dropped, repeats collapsed.

    The corpus deliberately carries static rows `(v, v)` and duplicated rows
    `(a, b), (b, a)` on two cases, because upstream was testing a disjoint-set
    structure against exactly that noise. Neither changes connectivity, so neither
    may change any reading here.
    """
    adj: list[set[int]] = [set() for _ in range(node_count)]
    for a, b in edges:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def ball(adj, source: int, radius: int) -> set[int]:
    """Every node within `radius` hops of `source`, inclusive of `source`."""
    seen = {source}
    frontier = [source]
    for _ in range(radius):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return seen


def eccentricity_diameter(adj, nodes) -> int:
    """Exact diameter of the component spanned by `nodes`, by BFS from each."""
    best = 0
    for s in nodes:
        dist = {s: 0}
        queue = [s]
        while queue:
            nxt = []
            for u in queue:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            queue = nxt
        best = max(best, max(dist.values()))
    return best


def nrmse(pred: np.ndarray, y: np.ndarray) -> float:
    """RMSE normalised by the std of y, so the mean predictor is exactly 1.0.

    Returns NaN when the label is constant, which is not a failure mode to be
    swallowed: it is the finding that `SupercriticalDense_S2Rips_256` cannot host a
    control at all, and the caller must print it rather than skip it.
    """
    sd = float(y.std())
    if sd == 0.0:
        return float("nan")
    return float(np.sqrt(((pred - y) ** 2).mean()) / sd)


def _ball_size_cache(adj, radius: int):
    cache: dict[int, int] = {}

    def size(node: int) -> int:
        if node not in cache:
            cache[node] = len(ball(adj, node, radius))
        return cache[node]

    return size


def local_features(pairs, graphs, radius: int) -> np.ndarray:
    """Symmetric local features for each `((i, j), which_graph)` instance.

    `radius = 0` is the degree-only decoder that section 1.7d names. Higher radii add
    ball sizes, which are still static and still strictly local -- no iteration to a
    fixed point, no component count, no partition. Pairs are unordered, so every
    feature is built from `min`/`max` and never from the draw order, which would
    otherwise let the decoder read an artefact of the sampler.
    """
    sizers = [[_ball_size_cache(g, k) for k in range(1, radius + 2)] for g in graphs]
    out = []
    for (i, j), which in pairs:
        row = [1.0]
        for sizer in sizers[which]:
            a, b = sizer(i), sizer(j)
            lo, hi = min(a, b), max(a, b)
            row += [lo, hi, lo * hi, hi - lo]
        out.append(row)
    return np.asarray(out, dtype=float)


def fit_eval(x: np.ndarray, y: np.ndarray) -> float:
    """Least squares on the first half, NRMSE on the held-out second half.

    A held-out split is not decoration. The decoder has up to twenty free parameters
    against a binary label, so an in-sample reading would credit memorisation as
    decoding and the strike below would be unearned.
    """
    n_train = len(y) // 2
    xt, yt = x[:n_train], y[:n_train]
    weights = np.linalg.lstsq(xt.T @ xt + 1e-6 * np.eye(x.shape[1]), xt.T @ yt,
                              rcond=None)[0]
    return nrmse(x[n_train:] @ weights, y[n_train:])


def planted_degree_labels(pairs, graphs):
    """Two labels the degree decoder MUST get, on the very instances where it failed.

    Without these the absence is worthless. A decoder that reads NRMSE 1.0 on the
    connectivity label has told us nothing until the same decoder, the same feature
    matrix and the same train/test split are seen to fire on a label that IS a
    function of degree.

    `sum` is the degree sum itself, so a linear decoder that can read these features
    at all must return essentially 0 -- it isolates "the fit works on these exact
    instances" from every question about difficulty. `median` is a BINARY label at
    the median of that sum over the drawn set, so it is balanced by construction and
    matches the connectivity label's own shape; it is the one that shows the decoder
    firing at the same difficulty class it failed at.
    """
    total = np.asarray([len(graphs[w][i]) + len(graphs[w][j]) for (i, j), w in pairs],
                       dtype=float)
    return {"sum": total,
            "median": (total > float(np.median(total))).astype(float)}


def draw_balanced_marginal(case, n_each: int = 1024):
    """Equal numbers of same-component and different-component pairs, drawn.

    Uniform pairs are not usable on this corpus: the same-component base rate is
    0.9922 on `CriticalLarge_S2Rips_1024` and 0.0074 on `StableRepeated_S2Rips_1024`,
    so a constant predictor is already near-perfect and NRMSE stops discriminating.
    Balancing fixes the label's std at exactly 0.5 on every case. Returns `None` when
    one side cannot be drawn at all, which is itself the reading.
    """
    n, lab = case.node_count, case.partition
    rng = random.Random(0x33960000 ^ case.seed)
    pos, neg = [], []
    for _ in range(600_000):
        if len(pos) >= n_each and len(neg) >= n_each:
            break
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        same = lab[i] >= 0 and lab[i] == lab[j]
        bucket = pos if same else neg
        if len(bucket) < n_each:
            bucket.append((i, j))
    if len(pos) < n_each or len(neg) < n_each:
        return None, (len(pos), len(neg))
    pairs = [(p, 0) for p in pos] + [(p, 0) for p in neg]
    y = np.asarray([1.0] * n_each + [0.0] * n_each)
    order = np.random.RandomState(0).permutation(len(y))
    return ([pairs[t] for t in order], y[order]), (len(pos), len(neg))


def draw_do_paired(name, node_count, target_degree, seed, join, n_pairs=1024):
    """The `do()`-paired draw: one pair, both sides of the single bridge edge.

    This is the instance set the bridge actually defines, and section 1.7c already
    reads consequence fidelity over drawn single-token `do()` interventions. Each
    drawn pair appears TWICE -- once against the pre-bridge graph with label 0, once
    against the post-bridge graph with label 1 -- so the label is exactly balanced by
    construction with no sampling knob left free, and the ONLY thing that can carry
    it is the one edge that changed.
    """
    points = sample_sphere(node_count, seed)
    pre_edges = rips_edges(points, target_degree)
    lab = components(node_count, pre_edges)
    from ceq.rips import _add_critical_bridge
    edge = _add_critical_bridge(points, pre_edges, node_count, join=join)
    post_edges = pre_edges + [edge]
    left = [v for v in range(node_count) if lab[v] == lab[edge[0]]]
    right = [v for v in range(node_count) if lab[v] == lab[edge[1]]]
    rng = random.Random(seed)
    drawn = [(rng.choice(left), rng.choice(right)) for _ in range(n_pairs)]
    pairs = [(p, 0) for p in drawn] + [(p, 1) for p in drawn]
    y = np.asarray([0.0] * n_pairs + [1.0] * n_pairs)
    order = np.random.RandomState(0).permutation(len(y))
    graphs = [adjacency(node_count, pre_edges), adjacency(node_count, post_edges)]
    return ([pairs[t] for t in order], y[order]), graphs, edge, (len(left), len(right))


def truncation_ladder(pairs, y, graphs) -> list[tuple[int, float]]:
    """Gate (a): the `k`-hop reachability reading of the label, rung by rung.

    A model with a hop budget of `k` cannot see further than `k` hops from the marked
    node, so predicting "same component iff `j` is inside `i`'s `k`-ball" is exactly
    what that budget buys. Bounded away from the label at small `k` and tightening
    with `k` is the property section 1.7 demands.
    """
    out = []
    for k in LADDER_KS:
        caches = [_ball_reach_cache(g, k) for g in graphs]
        pred = np.asarray([1.0 if j in caches[w](i) else 0.0 for (i, j), w in pairs])
        out.append((k, nrmse(pred, y)))
    return out


def _ball_reach_cache(adj, radius: int):
    cache: dict[int, set[int]] = {}

    def reach(node: int) -> set[int]:
        if node not in cache:
            cache[node] = ball(adj, node, radius)
        return cache[node]

    return reach


def _fmt(v: float) -> str:
    return "   n/a" if v != v else f"{v:6.4f}"


def report() -> str:
    lines = []
    w = lines.append

    w("=" * 78)
    w("GATE (b) PART 1 -- can the prescribed control be drawn at all?")
    w("=" * 78)
    for case in corpus():
        drawn, (n_pos, n_neg) = draw_balanced_marginal(case)
        sizes = sorted((sum(1 for x in case.partition if x == c)
                        for c in set(v for v in case.partition if v >= 0)), reverse=True)
        isolated = sum(1 for x in case.partition if x < 0)
        w(f"{case.name:34s} components={sizes[:4]}{'...' if len(sizes) > 4 else ''} "
          f"isolated={isolated}")
        if drawn is None:
            w(f"{'':34s}   VACUOUS: drew {n_pos} same / {n_neg} different. The label is "
              f"constant, its std is 0, NRMSE is undefined.")
            continue
        pairs, y = drawn
        row = [fit_eval(local_features(pairs, [adjacency(case.node_count, case.edges)],
                                       r), y) for r in (0, 1, 2, 3)]
        w(f"{'':34s}   decoder NRMSE  deg={_fmt(row[0])}  +1hop={_fmt(row[1])}  "
          f"+2hop={_fmt(row[2])}  +3hop={_fmt(row[3])}")

    w("")
    w("=" * 78)
    w("GATE (a) AND GATE (b) ON THE do()-PAIRED DRAW, AND THE REROUTE")
    w("=" * 78)
    specs = [("CriticalBridge_S2Rips_256", 256, 0.75 * math.log(256.0), 0x33960002,
              "nearest"),
             ("CriticalLarge_S2Rips_1024", 1024, math.ceil(math.log(1024.0)),
              0x33960006, "nearest"),
             ("LargestJoin_S2Rips_64  [REROUTE]", 64, 2.0, 0x33960001, "largest"),
             ("LargestJoin_S2Rips_1024 [REROUTE]", 1024, 2.0, 0x33960005, "largest")]
    for name, n, degree, seed, join in specs:
        (pairs, y), graphs, edge, (nl, nr) = draw_do_paired(name, n, degree, seed, join)
        w(f"{name}   bridge={edge}   merged {nl} x {nr}")
        lad = truncation_ladder(pairs, y, graphs)
        w("   gate (a) k-hop reachability: "
          + "  ".join(f"k={k}:{_fmt(v)}" for k, v in lad))
        for radius, tag in ((0, "degree only"), (1, "+1hop"), (2, "+2hop"),
                            (3, "+3hop"), (5, "+5hop")):
            x = local_features(pairs, graphs, radius)
            w(f"   gate (b) {tag:12s} NRMSE={_fmt(fit_eval(x, y))}"
              + ("   <-- SPEC" if radius == 0 else ""))
        x0 = local_features(pairs, graphs, 0)
        planted = planted_degree_labels(pairs, graphs)
        w(f"   PLANTED degree-sum       NRMSE={_fmt(fit_eval(x0, planted['sum']))}"
          f"   PLANTED degree-median NRMSE="
          f"{_fmt(fit_eval(x0, planted['median']))}   (control must fire)")
        w("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
