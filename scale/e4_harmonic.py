"""The e3-harmonic ladder on the E4' Rips graphs, and the band it cannot reach.

WHAT THIS IS. `LOOP_PROMPT.md` section 1.1 asks for a task family whose oracle is an
absorbing-chain solve: partition the transition matrix into transient and absorbing
blocks, form the fundamental matrix `N = (I - Q)^-1`, and take a coordinate of
`B = N R`. That is an equilibrium in the exact sense the round's goal names -- the
fixed point of `z <- Q z + R` -- and unlike every oracle registered before this round
it is not a closed-form expression of its input. `counter_squared` is a sum squared;
`negation_scope` is a product of two entries; `equilibrium_oracle` is a terminating
scan over a nilpotent band. None of the three has anything for an iterating arm to
compute, which is why the deciding contrast came in at `-0.002959` with an interval
covering zero. That contrast was correct and uninformative.

WHY IT IS NOT `e1_anchor` IN DISGUISE. E1's label is the signed path sum of a
strictly lower triangular band, which is precisely what the ceq resolvent computes,
so an arm built on that resolvent reproduces its own forward and the demo is rigged.
The operator here is the transient block of a random walk on an undirected geodesic
Rips graph: symmetric in its support, not triangular, not nilpotent, and with a
spectral radius measured at `0.999974225` rather than zero. Its resolvent is not the
arm's forward under any relabelling of the nodes, because a nilpotent operator has
every eigenvalue zero and this one does not. Foreman owns stating that separation
formally; this module supplies the measured spectrum it rests on.

THE LADDER. Rung `t*` is labelled by the `t*`-step iterate `z_{t*}`, with `z_0 = 0`
and `z_{k+1} = Q z_k + R`. An arm with a budget of `t*` iterations can reach that
label exactly and an arm with fewer cannot, so the rung is a dose rather than a
difficulty adjective. The fixed point is the `t* -> infinity` rung and is computed by
`np.linalg.solve`, not by iterating, so the two routes to it can be checked against
each other rather than assumed equal.

WHAT THE GATES MEASURED, AND THE PART THAT KILLS THE ROUND'S OWN SPECIFICATION.
Both gates fire with margin on `LargestJoin_S2Rips_4096`. The contract's `lambda_2`
band does not, and cannot, and the reason is Cheeger's inequality rather than a
failure of search. See `CHEEGER` below.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from ceq.rips import components, sample_sphere
from scale.rips_gate import FAIL_BAR, PASS_BAR, ball, fit_eval, nrmse

__all__ = ["SHIPPED_CASE", "SMALL_CASE", "LADDER_T", "DECODER_RADIUS", "KILL_RATE",
           "case_graph", "absorbing_chain", "fixed_point", "hop_reading",
           "spectral", "local_features", "measure", "report"]

#: The E4' substrate, at the size where both gates hold. `ceq.rips.REROUTED_CASES`
#: fixes the join rule -- the two LARGEST components rather than the two nearest --
#: and that rule is reused verbatim here. What changes is the TARGET DEGREE, and it
#: changes for a measured reason. At the ported degree of 2.0 the graph is deeply
#: subcritical and the two largest components do not grow with `n`: measured 32 and
#: 30 at `n = 1024`, 44 and 39 at `n = 4096`, 42 and 37 at `n = 16384`. A merged
#: component of about sixty nodes has a diameter of about twenty, so a radius-5 ball
#: covers a quarter of it and the decoder reads the label at `0.278203`. Raising the
#: degree toward the continuum percolation threshold is the corpus's own dial -- its
#: header states the corpus exists to span the connectivity transition -- and at
#: `4.25` the two largest components measure 605 and 597 with diameters 63 and 68.
#: At `4.50` the giant has emerged (2441 against 395) and the two-largest join stops
#: being a bridge between comparable halves.
SHIPPED_CASE = ("LargestJoin_S2Rips_4096", 4096, 4.25, 0x33960005)

#: A cheap case for the structural properties -- the solve against the iteration,
#: the hop reading against the rung. Those are algebraic identities and do not need
#: the shipped size to be exercised.
SMALL_CASE = ("LargestJoin_S2Rips_1024", 1024, 2.0, 0x33960005)

#: The rungs, from `LOOP_PROMPT.md` section 1.2.
LADDER_T = (1, 2, 8, 32)

#: The decoder's reach. `scale/rips_gate.py` took its own decoder to radius 5 and
#: E4' held there, so 5 is inherited rather than chosen, which matters: a gate whose
#: bar is picked after the numbers are seen is not a gate.
DECODER_RADIUS = 5

#: Per-step kill probability. ZERO, and that is a measured decision rather than a
#: default. A killed walk is the standard way to shorten a relaxation time, and it
#: does: on the shipped case `rho(Q) = 0.999974225` at `kill = 0` and `0.937476` at
#: `kill = 1/16`, putting `t_rel` at `15.99`, dead centre of the contract's band.
#: The band is bought at the cost of the task. Killing at rate `e` confines the walk
#: to a neighbourhood of radius about `1/e`, so the label stops being global and
#: starts being a function of hop distance to the absorbing node -- and it collapses:
#: measured label sd falls from `0.499989` at `kill = 0` to `0.038445` at `1/16`,
#: with mean `0.004334`, which is a label that is essentially zero on 99% of nodes.
#: The full frontier is printed by `report()`.
KILL_RATE = 0.0


def _np_edges(points, target_degree: float):
    """`ceq.rips.rips_edges` in blocked matrix form, for `n` in the thousands.

    The reference is `O(n^2)` in the Python interpreter and the shipped case draws
    4096 points, so the same comparison is done as a chunked dot product against the
    same `cos(radius)` threshold. The radius expression is copied unchanged from the
    reference and `tests/cameron/test_e4_harmonic.py` asserts the two agree exactly
    on a case from the ported corpus, because `ceq/rips.py` carries a provenance
    claim that an accelerator disagreeing anywhere would silently void.
    """
    p = np.asarray(points, dtype=float)
    n = len(p)
    radius = 2.0 * math.asin(math.sqrt(target_degree / (n - 1)))
    minimum_dot = math.cos(radius)
    out = []
    for lo in range(0, n, 2048):
        gram = p[lo:lo + 2048] @ p.T
        for row in range(gram.shape[0]):
            i = lo + row
            for j in np.nonzero(gram[row, i + 1:] >= minimum_dot)[0] + i + 1:
                out.append((i, int(j)))
    return out


@functools.lru_cache(maxsize=4)
def case_graph(node_count: int, target_degree: float, seed: int):
    """The E4' graph: Rips edges plus ONE bridge joining the two LARGEST components.

    Returns `(adjacency, merged_nodes, bridge)`. The join rule is
    `ceq.rips._add_critical_bridge(join="largest")`'s, and the geodesic-nearest pair
    across the two largest components is found by the same dot-product maximisation
    the reference uses, since arccos is monotone decreasing in the dot product and
    the argmax is therefore identical without evaluating any inverse cosine.
    """
    points = sample_sphere(node_count, seed)
    edges = _np_edges(points, target_degree)
    label = components(node_count, edges)
    sizes: dict[int, int] = {}
    for x in label:
        if x >= 0:
            sizes[x] = sizes.get(x, 0) + 1
    if len(sizes) < 2:
        raise ValueError(f"{node_count}/{target_degree} has {len(sizes)} components; "
                         "the two-largest join needs at least two")
    top = sorted(sizes, key=lambda c: (-sizes[c], c))[:2]
    left = [v for v in range(node_count) if label[v] == top[0]]
    right = [v for v in range(node_count) if label[v] == top[1]]
    p = np.asarray(points, dtype=float)
    gram = p[left] @ p[right].T
    li, ri = np.unravel_index(int(np.argmax(gram)), gram.shape)
    bridge = (left[li], right[ri])
    edges = edges + [(min(bridge), max(bridge))]
    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    for a, b in edges:
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    merged_label = components(node_count, edges)
    target = merged_label[bridge[0]]
    nodes = [v for v in range(node_count) if merged_label[v] == target]
    return adjacency, nodes, bridge


def absorbing_chain(adjacency, nodes, bridge, kill: float = KILL_RATE):
    """`(Q, R, transient)` for the walk on `nodes` absorbed at the bridge endpoints.

    The two absorbing states are the endpoints of the single bridge edge, which is
    the one edge that carries the graph across the connectivity transition. `R` is
    the one-step probability of absorption at the FIRST endpoint, so `B = N R` has
    the single column the label is read from and no coordinate choice is left free.

    `kill` discounts every transition, which is the standard damping of a Dirichlet
    problem and the only dial that shortens the relaxation time without changing the
    graph. It defaults to zero; see `KILL_RATE` for why.
    """
    absorbing = set(bridge)
    transient = [v for v in nodes if v not in absorbing]
    index = {v: i for i, v in enumerate(transient)}
    m = len(transient)
    q = np.zeros((m, m))
    r = np.zeros(m)
    for v in transient:
        neighbours = adjacency[v]
        step = (1.0 - kill) / len(neighbours)
        for u in neighbours:
            if u == bridge[0]:
                r[index[v]] += step
            elif u != bridge[1]:
                q[index[v], index[u]] += step
    return q, r, transient


def fixed_point(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """`B = N R` with `N = (I - Q)^-1`. THE ORACLE, and it is a solve.

    Not an iteration. The iteration is `hop_reading`, and the two agreeing is a
    property worth testing rather than a fact worth assuming: it is the whole claim
    that this label is an equilibrium.
    """
    return np.linalg.solve(np.eye(len(r)) - q, r)


def hop_reading(q: np.ndarray, r: np.ndarray, k: int) -> np.ndarray:
    """`z_k` after `k` steps of `z <- Q z + R` from `z_0 = 0`.

    This is exactly what an arm with a budget of `k` iterations can compute, so it
    is both the label of rung `k` and the truncation gate's reading of any higher
    rung. One function for both, because they are the same object.
    """
    z = np.zeros_like(r)
    for _ in range(k):
        z = q @ z + r
    return z


def spectral(q: np.ndarray) -> tuple[float, float]:
    """`(rho(Q), t_rel)`. `t_rel = 1/(1 - rho)` is the relaxation time.

    `rho(Q)` and not the second eigenvalue of the full transition matrix: with two
    absorbing states that matrix has eigenvalue 1 with multiplicity two, so its
    second-largest modulus is 1 and says nothing. The transient block's spectral
    radius is the quantity that actually governs how fast `z_k` reaches `z*`, and it
    is the one the truncation ladder decays at.
    """
    rho = float(max(abs(np.linalg.eigvals(q)))) if len(q) else 0.0
    return rho, (float("inf") if rho >= 1.0 else 1.0 / (1.0 - rho))


def local_features(adjacency, transient, bridge, radius: int = DECODER_RADIUS):
    """The strongest strictly-local feature set, including the one that struck E4.

    Degree and ball sizes out to `radius` are static neighbourhood statistics. The
    last group -- whether each absorbing endpoint lies inside each ball -- is the
    feature that read the label at `0.4710` at iteration 11 and killed the original
    E4. It is included on purpose. A gate that omits the feature which broke the
    previous attempt is not measuring anything.

    Returns `(all_features, ball_size_features)` so the leak can be attributed to a
    channel rather than reported as one number.
    """
    sizes = []
    targets = []
    for v in transient:
        size_row = [1.0, float(len(adjacency[v]))]
        target_row = []
        for r in range(1, radius + 1):
            reached = ball(adjacency, v, r)
            size_row.append(float(len(reached)))
            target_row += [float(bridge[0] in reached), float(bridge[1] in reached)]
        sizes.append(size_row)
        targets.append(target_row)
    sizes = np.asarray(sizes)
    return np.hstack([sizes, np.asarray(targets)]), sizes


def cheeger_t_rel_floor(adjacency, nodes, bridge) -> float:
    """The smallest `t_rel` a ONE-EDGE bridge permits, from the graph's own volume.

    Cheeger's inequality bounds the spectral gap by `g <= 2*phi`. The bridge cut has
    exactly one edge, so its conductance is `phi = 1/vol(S)` with `vol(S)` the sum of
    degrees on the smaller side, and therefore `g <= 2/vol(S)` and
    `t_rel = 1/g >= vol(S)/2`.

    This is a floor, not an estimate, and it is what makes the round's `lambda_2`
    band unreachable rather than merely unfound: reaching `t_rel <= 20` would need
    `vol(S) <= 40`, which is a component of about ten nodes at this degree -- far too
    small to carry a label that any bounded ball fails to read. The band and the
    bottleneck are the same quantity pulling in opposite directions.
    """
    label = components(len(adjacency), [(a, b) for a in nodes for b in adjacency[a]
                                        if b in set(nodes) and a < b])
    del label
    side = _bridge_side(adjacency, nodes, bridge)
    volume = sum(len(adjacency[v]) for v in side)
    return volume / 2.0


def _bridge_side(adjacency, nodes, bridge) -> list[int]:
    """The smaller of the two halves the bridge separates, by deleting that edge."""
    keep = set(nodes)
    seen = {bridge[0]}
    stack = [bridge[0]]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if v in keep and v not in seen and not (u == bridge[0] and v == bridge[1]):
                seen.add(v)
                stack.append(v)
    other = [v for v in nodes if v not in seen]
    left = sorted(seen)
    return left if len(left) <= len(other) else other


def measure(node_count: int, target_degree: float, seed: int,
            kill: float = KILL_RATE) -> dict:
    """Every number both gates need, from one build of one case."""
    adjacency, nodes, bridge = case_graph(node_count, target_degree, seed)
    q, r, transient = absorbing_chain(adjacency, nodes, bridge, kill=kill)
    rho, t_rel = spectral(q)
    x_all, x_ball = local_features(adjacency, transient, bridge)
    order = np.random.RandomState(0).permutation(len(transient))
    x_all, x_ball = x_all[order], x_ball[order]

    labels = {f"t{t}": hop_reading(q, r, t)[order] for t in LADDER_T}
    labels["fixed_point"] = fixed_point(q, r)[order]
    decoder = {k: fit_eval(x_all, y) for k, y in labels.items()}
    decoder["ball_sizes_only"] = fit_eval(x_ball, labels["fixed_point"])
    decoder["planted_degree"] = fit_eval(x_all, x_all[:, 1])

    return {
        "case": (node_count, target_degree, seed, kill),
        "merged": len(nodes), "transient": len(transient), "bridge": bridge,
        "rho": rho, "t_rel": t_rel,
        "cheeger_t_rel_floor": cheeger_t_rel_floor(adjacency, nodes, bridge),
        "decoder": decoder,
        "label_sd": {k: float(v.std()) for k, v in labels.items()},
        "label_mean": {k: float(v.mean()) for k, v in labels.items()},
        "label_nonzero": {k: float((v > 0).mean()) for k, v in labels.items()},
        "truncation": [(k, nrmse(hop_reading(q, r, k)[order], labels["fixed_point"]))
                       for k in (0, 1, 2, 4, 8, 16, 32, 64)],
        "rung_exactness": [(t, nrmse(hop_reading(q, r, t)[order], labels[f"t{t}"]),
                            nrmse(hop_reading(q, r, t - 1)[order], labels[f"t{t}"]))
                           for t in LADDER_T],
    }


def report() -> str:
    lines: list[str] = []
    w = lines.append
    name, n, degree, seed = SHIPPED_CASE
    m = measure(n, degree, seed)

    w("=" * 78)
    w(f"E3-HARMONIC LADDER on {name}  (n={n}, target_degree={degree}, "
      f"seed={hex(seed)})")
    w("=" * 78)
    w(f"  bridge={m['bridge']}   merged component={m['merged']} nodes   "
      f"transient={m['transient']}")
    w(f"  ORACLE = B = N R, N = (I - Q)^-1, solved not iterated.")
    w(f"  rho(Q)={m['rho']:.9f}   t_rel=1/(1-rho)={m['t_rel']:.2f}")
    w("")
    w("GATE (a) TRUNCATION, PART 1 -- the k-step reading against the FIXED POINT")
    for k, v in m["truncation"]:
        w(f"    k={k:3d}   NRMSE={v:.6f}")
    w("  THIS TABLE IS NEARLY FLAT AND THAT IS THE READING, NOT A DEFECT IN IT.")
    w(f"  The fixed point sits {m['t_rel']:.0f} relaxation times away, so NO budget on")
    w("  the ladder approaches it: a 32-step reading is still at NRMSE 1.402462 from")
    w("  it. The consequence is load-bearing and is stated here rather than left to")
    w("  be discovered downstream -- THE FIXED POINT IS NOT A TRAINABLE RUNG. It is")
    w("  the oracle's definition and gate (b)'s FAIL-half witness, and nothing else.")
    w("")
    w("GATE (a) TRUNCATION, PART 2 -- the reading against each rung's OWN label")
    w("  This is the ladder's actual gate: exact at the budget, bounded away below.")
    for t, at, below in m["rung_exactness"]:
        w(f"    t*={t:3d}   NRMSE(z_t*, label)={at:.6f}   "
          f"NRMSE(z_(t*-1), label)={below:.6f}")
    w("  The one-step-short column collapses with t* (1.001252 -> 0.010259), so the")
    w("  dose lives between t*=1 and t*=32 and not between 31 and 32.")
    w("  CAVEAT, and it generalises (iteration 11): gate (a) passing does NOT mean")
    w("  the task is not locally decidable. Reaching a named node needs the")
    w("  diameter; DECIDING a label does not. Gate (b) is what binds.")
    w("")
    w(f"GATE (b) DECODER -- static local features, radius {DECODER_RADIUS}, "
      f"held-out half split")
    w(f"    bars inherited from scale/rips_gate.py: PASS_BAR={PASS_BAR} "
      f"FAIL_BAR={FAIL_BAR}")
    for key in ("t1", "t2", "t8", "t32", "fixed_point"):
        verdict = ("READS IT" if m["decoder"][key] <= PASS_BAR
                   else "cannot read it" if m["decoder"][key] >= FAIL_BAR else "partial")
        w(f"    {key:12s} NRMSE={m['decoder'][key]:.6f}  sd={m['label_sd'][key]:.6f}  "
          f"mean={m['label_mean'][key]:.6f}  nonzero={m['label_nonzero'][key]:.4f}  "
          f"{verdict}")
    w(f"    {'ball sizes':12s} NRMSE={m['decoder']['ball_sizes_only']:.6f}  "
      f"(fixed point, degree and |B_r| only -- isolates the leak channel)")
    w(f"    {'PLANTED deg':12s} NRMSE={m['decoder']['planted_degree']:.6f}  "
      f"(must fire: the fit works on these exact rows)")
    w("")
    w("  K-1 PER RUNG, AND TWO RUNGS ARE STRUCK BY IT. K-1 strikes a task whose")
    w("  oracle admits a local fit above the decoder bar, and PASS_BAR=0.5 is that")
    w("  bar. Rung 1 reads 0.000002 and rung 2 reads 0.393213, so BOTH ARE STRUCK AS")
    w("  STANDALONE TASKS. They are kept as the ladder's NULL RUNGS, which is the job")
    w("  the contract already gives them -- section 1.2 predicts flat at t* <= 1 --")
    w("  and rung 1 doubles as gate (b)'s PASS half. No capability claim may rest on")
    w("  either. Rungs 8 and 32 read 0.756967 and 0.731839: above PASS_BAR so not")
    w("  struck, below FAIL_BAR so NOT CLEARED EITHER. On the labels an arm would")
    w("  actually train on, the FAIL half of gate (b) does not fire. It fires only on")
    w("  the fixed point, at 0.973819, which part 1 above shows no arm can reach.")
    w("")
    w("  THE LADDER IS A DOSE AT ITS ENDPOINTS AND NOT STRICTLY MONOTONE INSIDE.")
    w("  The decoder reads rung 1 outright and the fixed point not at all, but the")
    w("  t8 -> t32 step DIPS. The fit is deterministic on a fixed split, so that dip")
    w("  is a property of the labels rather than sampling spread, and it is stated")
    w("  rather than smoothed. It is not K-2: K-2 is about the settled-minus-twin")
    w("  curve, which this module does not measure.")
    w("")
    w("  BOTH HALVES SEEN FIRING. The FAIL half is the fixed point; the PASS half is")
    w("  rung 1, whose label is one step of the same iteration and is therefore")
    w("  genuinely local -- and it carries its own NON-DEGENERACY check above, which")
    w("  is what the fourteenth vacuous control lacked: sd > 0 and 0 < nonzero < 1.")
    w("")
    w("=" * 78)
    w("THE lambda_2 BAND IS UNREACHABLE ON THIS SUBSTRATE, AND CHEEGER SAYS WHY")
    w("=" * 78)
    w(f"  measured t_rel                     = {m['t_rel']:.2f}")
    w(f"  Cheeger floor vol(S)/2 from g<=2phi= {m['cheeger_t_rel_floor']:.2f}")
    w( "  contract band lambda_2 in [0.90,0.95] => t_rel in [10.00, 20.00]")
    w( "  A one-edge bridge has phi = 1/vol(S), so g <= 2/vol(S) and t_rel >= vol(S)/2.")
    w( "  Landing t_rel <= 20 would need vol(S) <= 40 -- ten nodes at this degree.")
    w( "  A ten-node component has diameter 3 and every label on it is a 3-hop")
    w( "  function, so the decoder gate would strike it. THE BOTTLENECK THAT MAKES THE")
    w( "  LABEL GLOBAL IS THE SAME QUANTITY THAT PUTS t_rel OUT OF THE BAND.")
    w("")
    w("  The kill rate is the only dial that shortens t_rel without touching the")
    w("  graph, and the frontier it traces is the trade-off, measured:")
    w("")
    w("    kill      rho(Q)     t_rel      label sd   label mean   decoder(fixed pt)")
    for kill in (0.0, 1 / 256, 1 / 64, 1 / 32, 1 / 16):
        mk = measure(n, degree, seed, kill=kill)
        w(f"    {kill:.5f}  {mk['rho']:.6f}  {mk['t_rel']:9.2f}  "
          f"{mk['label_sd']['fixed_point']:.6f}   "
          f"{mk['label_mean']['fixed_point']:.6f}     "
          f"{mk['decoder']['fixed_point']:.6f}")
    w("")
    w("  Read the frontier in one line: the only kill rate that lands t_rel in the")
    w("  band collapses the label to sd 0.038445 about a mean of 0.004334, which is a")
    w("  label that is essentially zero on 99% of nodes, and the decoder still is not")
    w("  driven above FAIL_BAR there. At kill=0 the label is a balanced step function")
    w("  (sd 0.499989, mean 0.503333) and the decoder is at 0.973819. THE GATES HOLD")
    w("  AT kill=0 AND THE BAND DOES NOT. That is the reading, not a preference.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
