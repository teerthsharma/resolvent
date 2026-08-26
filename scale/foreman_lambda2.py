"""Engineer and MEASURE the relaxation dial of the E4' absorbing chain.

WHAT THIS FILE OWNS AND WHAT IT DOES NOT. It owns one number per graph: the largest
eigenvalue modulus of the chain below one, engineered into `[0.90, 0.95]` and then
measured rather than assumed. It does not build the corpus (`ceq/rips.py`, Cameron),
does not gate it (`scale/rips_gate.py`, Cameron) and does not read any arm
(`scale/m3_quintuple.py`, Chase). It imports the corpus and returns numbers.

WHY THE NUMBER MATTERS. `LOOP_PROMPT.md` 1.2: the `t*` ladder `{1, 2, 8, 32}` is only
a falsifiable prediction if the rate at which the truncation error falls is set by a
spectral quantity measured independently on the graph. With
`t_rel = 1/(1 - lambda_2)` in `10..20`, `t* = 1` and `t* = 2` are far from the fixed
point and `t* = 32` is close to it. An ASSUMED `lambda_2` makes that prediction
vacuous, so every number below is computed from the actual transition matrix.

WHAT `lambda_2` IS HERE, EXACTLY, AND WHY THE OBVIOUS READING IS WRONG. Order the
states transient-then-absorbing; the transition matrix is block upper triangular,

    P = [[Q, R], [0, I]]

so `spec(P) = spec(Q) union {1 repeated |B| times}`. With two absorbing states the
eigenvalue `1` has multiplicity two, and the literal "second largest eigenvalue
modulus of P" is therefore `1` and carries no information. The quantity that governs
`z <- Q z + R` -- the only quantity the `t*` ladder truncates -- is the PERRON ROOT
OF THE TRANSIENT BLOCK, `rho(Q)`. This module defines `lambda_2 := rho(Q)` and says
so at every print.

THE ORACLE IS CONSTANT UNLESS THERE ARE TWO ABSORBING TARGETS, AND THAT IS A
VACUOUS-CONTROL TRAP. With a single absorbing class and no killing, every transient
state is absorbed there with probability exactly `1`, so the label is the constant
function `1`, its standard deviation is `0`, and NRMSE against it is undefined -- the
exact shape of the fourteenth struck control (`SupercriticalDense_S2Rips_256`, one
component, constant label). Two absorbing targets with boundary values `1` and `0`
make the label the solution of a discrete DIRICHLET problem, which is non-constant.
`degenerate_single_target` is the must-fire that shows the constant case is reachable
and is actually rejected.

## THE CONDUCTANCE DIAL DOES NOT CONTROL THE QUANTITY THE LADDER TRUNCATES

`LOOP_PROMPT.md` 1.2 states that "Cheeger's inequality bounds `lambda_2` by the
conductance from both sides, so the engineering target is reachable by construction
rather than by search". MEASURED, ON THESE GRAPHS, THAT IS FALSE, in two separate
ways, and both are recorded rather than worked around.

**First, Cheeger is about a different matrix.** `h^2/2 <= mu_2 <= 2 h` is a statement
about the ERGODIC walk, where `mu_2 = 1 - lambda_2(P)` and `h` is the minimum
conductance over all cuts. The absorbing chain's `rho(Q)` is set by how fast the walk
reaches the absorbing set, not by how fast it crosses the bottleneck, and the two are
numerically different here: on `LargestJoin_S2Rips_1024` at bridge weight `1` the
ergodic `lambda_2` is `0.9964078857` while `rho(Q)` is `0.9984623637`.

**Second, the dial runs the wrong way and cannot reach.** Weighting the bridge edge
was the obvious construction, and `bridge_sweep` measures it: on
`LargestJoin_S2Rips_64` `rho(Q)` runs `0.9789623187` at `w = 1` up to `0.9998912353`
at `w = 4096`, MONOTONE INCREASING, because a heavy bridge traps the walk shuttling
across it instead of letting it reach a target. The band `[0.90, 0.95]` is not
attained anywhere on that sweep, on either graph. The ergodic `lambda_2` does move
into the band on the 64-node case (`0.9474294328` at `w = 16`), which is precisely
the point: the conductance dial controls the number Cheeger is about and not the
number the ladder truncates.

**Also, only one half of Cheeger was ever usable.** The bridge cut gives an UPPER
bound `h <= Phi_bridge`, and an upper bound on `h` composes only with `mu_2 <= 2h`.
The lower half `h^2/2 <= mu_2` needs `h` itself, a minimum over `2^(n-1)` cuts, which
is not computed here. So even for the ergodic chain the construction could only ever
have guaranteed "not too fast", never "not too slow".

## THE REPLACEMENT DIAL, AND IT IS EXACT RATHER THAN BRACKETED

`Q := alpha * P_TT` with `R` unchanged is the PARTIALLY ABSORBING walk: at each step
the walk is killed with probability `1 - alpha` instead of moving to a transient
state. Scaling a matrix scales its spectrum, so

    rho(alpha * P_TT) = alpha * rho(P_TT)      exactly

and `alpha := target / rho(P_TT)` lands `lambda_2` on the target with no bracket, no
bisection and no search. `rho(P_TT)` is still measured per graph, so the graph still
supplies the quantity; what the dial supplies is a scalar. `engineer` reports the
achieved `lambda_2` from a fresh eigensolve rather than from the construction, and
`demo` asserts the two agree.

THE HONEST COST OF THAT DIAL, STATED HERE AND NOT BURIED. The killed walk survives
about `t_rel` steps, so a node farther than roughly `t_rel` hops from both targets
has a label indistinguishable from zero. On `LargestJoin_S2Rips_1024`, whose merged
component has diameter `29` against `t_rel = 13.3333`, `0.5833` of the transient
nodes read below `1e-3` at the targets' widest placement. `dead_fraction` measures
this at every configuration and `target_separation_sweep` maps it, so the corpus
decision is made against numbers. It is Cameron's decision, not this file's.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ceq.rips import make_case, REROUTED_CASES

__all__ = ["TARGET_LO", "TARGET_HI", "TARGET", "LADDER", "Chain",
           "merged_component", "bridge_edge", "lobes", "build_chain", "diameter",
           "lambda_2", "ergodic_lambda_2", "bridge_conductance",
           "absorption_probabilities", "truncated_absorption", "relative_error",
           "DEGENERATE_STD", "dead_fraction", "bridge_sweep", "killing_rate", "engineer",
           "degenerate_single_target", "no_kill_control", "bridge_dial_refuted",
           "target_separation_sweep", "report", "demo"]

#: The band `LOOP_PROMPT.md` 1.2 asks for, and the ladder it must straddle.
TARGET_LO, TARGET_HI = 0.90, 0.95
TARGET = 0.925
LADDER = (1, 2, 8, 32)

#: The bridge weights the refuted dial is measured over.
BRIDGE_WEIGHTS = (1.0, 2.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0)

#: A label below this reads as zero for any predictor with finite precision on a
#: label whose own spread is order 0.1; `dead_fraction` counts them.
DEAD = 1e-3

#: A label whose spread is below this is constant to float64 and cannot be scored.
DEGENERATE_STD = 1e-12


class Chain:
    """The absorbing chain on one E4' graph, with everything measured on it."""

    __slots__ = ("name", "nodes", "w_bridge", "alpha", "absorbing", "transient",
                 "Q", "R", "P", "vol", "bridge", "adjacent")

    def __init__(self, name, nodes, w_bridge, alpha, absorbing, transient, Q, R, P,
                 vol, bridge, adjacent):
        self.name, self.nodes, self.w_bridge, self.alpha = name, nodes, w_bridge, alpha
        self.absorbing, self.transient = absorbing, transient
        self.Q, self.R, self.P, self.vol = Q, R, P, vol
        self.bridge, self.adjacent = bridge, adjacent


def merged_component(case) -> list[int]:
    """The nodes of the component the bridge created, ascending.

    `make_case(..., bridge="largest")` joins the two LARGEST components, so the
    component containing the bridge is the largest one afterwards. Selecting by size
    rather than by walking the bridge keeps this correct if the generator's
    tie-breaking ever changes, and `build_chain` cross-checks that both bridge
    endpoints landed in the set that comes back.
    """
    sizes: dict[int, int] = {}
    for lab in case.partition:
        if lab >= 0:
            sizes[lab] = sizes.get(lab, 0) + 1
    top = sorted(sizes, key=lambda c: (-sizes[c], c))[0]
    return [v for v, lab in enumerate(case.partition) if lab == top]


def bridge_edge(case, spec) -> tuple[int, int]:
    """The single added edge, recovered as a SET DIFFERENCE against the same case
    built with `bridge=False`.

    `ceq.rips._add_critical_bridge` is private and is deliberately not called from
    here. Rebuilding the unbridged case through the public `make_case` and taking the
    difference uses only the exported surface, and it fails loudly if the added edge
    is ever not unique -- which is the failure this would otherwise hide.
    """
    base = make_case(spec[0], spec[1], spec[2], spec[3], False, spec[5], spec[6])
    extra = {tuple(e) for e in case.edges} - {tuple(e) for e in base.edges}
    if len(extra) != 1:
        raise ValueError(f"{spec[0]}: expected exactly one added edge, got {extra}")
    return tuple(extra.pop())


def _adjacent_of(nodes, edges) -> dict[int, set[int]]:
    keep = set(nodes)
    adjacent: dict[int, set[int]] = {v: set() for v in nodes}
    for a, b in edges:
        if a != b and a in keep and b in keep:
            adjacent[a].add(b)
            adjacent[b].add(a)
    return adjacent


def _bfs(adjacent, source, blocked=()) -> dict[int, int]:
    """Hop distances from `source`, skipping the `blocked` edge set."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adjacent[u]:
            if (u, v) in blocked or (v, u) in blocked:
                continue
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def lobes(nodes, edges, bridge) -> tuple[list[int], list[int]]:
    """The two halves the bridge joins, in bridge-endpoint order.

    Removing the bridge from the merged component must leave exactly two pieces --
    that is what "a single edge carries the graph across the transition" means -- and
    this raises rather than guesses if it does not.
    """
    adjacent = _adjacent_of(nodes, edges)
    left = _bfs(adjacent, bridge[0], blocked={bridge})
    right = _bfs(adjacent, bridge[1], blocked={bridge})
    if set(left) & set(right):
        raise ValueError("bridge removal did not disconnect the component")
    if len(left) + len(right) != len(nodes):
        raise ValueError(f"lobes {len(left)}+{len(right)} != {len(nodes)} nodes")
    return sorted(left), sorted(right)


def diameter(chain: Chain) -> int:
    """Hop diameter of the merged component. Printed beside `t_rel` because the two
    together decide whether a killed walk can reach anything."""
    return max(max(_bfs(chain.adjacent, v).values()) for v in chain.nodes)


def _far_node(adjacent, source, within) -> int:
    """The node of `within` at greatest hop distance from `source`; ties to the
    smallest index, so the choice is deterministic and reproducible."""
    dist = _bfs(adjacent, source)
    return max(((dist[v], -v) for v in within if v in dist))[1] * -1


def build_chain(case, spec, w_bridge: float = 1.0, *, alpha: float = 1.0,
                single_target: bool = False, targets=None) -> Chain:
    """The absorbing chain: by default one absorbing target per lobe, at the far end.

    THE TARGETS SIT AS FAR FROM THE BRIDGE AS THE LOBE ALLOWS. Absorption probability
    at a target adjacent to the bridge would be readable from a small ball around the
    query, which is the local-decoder leak that struck E4 as specified. Placing each
    target at its lobe's eccentric node makes the label a Dirichlet solution whose
    level sets run the length of both lobes.

    `alpha < 1` kills the walk with probability `1 - alpha` at every step it would
    otherwise have moved to a transient state; see the module header for why this and
    not the bridge weight is the dial that reaches the band.

    `single_target=True` builds the DEGENERATE chain deliberately -- one absorbing
    state and no killing, so absorption probability is identically `1`. It exists to
    be rejected. `targets` overrides the placement for the separation sweep.
    """
    nodes = merged_component(case)
    bridge = bridge_edge(case, spec)
    if bridge[0] not in nodes or bridge[1] not in nodes:
        raise ValueError(f"bridge {bridge} is not inside the merged component")
    adjacent = _adjacent_of(nodes, case.edges)

    if targets is None:
        left, right = lobes(nodes, case.edges, bridge)
        absorbing = [_far_node(adjacent, bridge[0], left)]
        if not single_target:
            absorbing.append(_far_node(adjacent, bridge[1], right))
    else:
        absorbing = list(targets)

    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n), dtype=np.float64)
    for a, b in case.edges:
        if a != b and a in index and b in index:
            W[index[a], index[b]] = W[index[b], index[a]] = 1.0
    W[index[bridge[0]], index[bridge[1]]] = float(w_bridge)
    W[index[bridge[1]], index[bridge[0]]] = float(w_bridge)

    vol = W.sum(axis=1)
    P = W / vol[:, None]
    b_idx = [index[v] for v in absorbing]
    t_idx = [i for i in range(n) if i not in set(b_idx)]
    Q = float(alpha) * P[np.ix_(t_idx, t_idx)]
    R = P[np.ix_(t_idx, b_idx)]
    return Chain(case.name, nodes, float(w_bridge), float(alpha), absorbing,
                 [nodes[i] for i in t_idx], Q, R, P, vol, bridge, adjacent)


def lambda_2(chain: Chain) -> float:
    """`rho(Q)`: the largest eigenvalue modulus of the chain that is below one.

    See the module header for why this and not the literal second eigenvalue of `P`.
    """
    return float(np.abs(np.linalg.eigvals(chain.Q)).max())


def ergodic_lambda_2(chain: Chain) -> float:
    """Second largest eigenvalue of the ERGODIC walk on the same weighted graph.

    Computed from the symmetric conjugate `D^{-1/2} W D^{-1/2}`, which is similar to
    `P` and has real spectrum, so `eigvalsh` applies and the ordering is exact rather
    than a sort of complex moduli. This is the quantity Cheeger's inequality is about;
    `lambda_2` above is not, and the two are printed side by side so the gap between
    them is visible rather than elided.
    """
    d = np.sqrt(chain.vol)
    W = chain.P * chain.vol[:, None]
    return float(np.sort(np.linalg.eigvalsh(W / d[:, None] / d[None, :]))[-2])


def bridge_conductance(chain: Chain) -> float:
    """`Phi = w / min(vol(lobe_1), vol(lobe_2))` for the bridge cut.

    ONE cut, so this is an upper bound on the Cheeger constant `h` and never a value
    for it. The module header states which half of Cheeger that upper bound may be
    composed with, and why the other half was never available.
    """
    edges = [(chain.nodes[i], chain.nodes[j])
             for i in range(len(chain.nodes)) for j in range(i + 1, len(chain.nodes))
             if chain.P[i, j] > 0.0]
    left, right = lobes(chain.nodes, edges, chain.bridge)
    index = {v: i for i, v in enumerate(chain.nodes)}
    v_left = float(sum(chain.vol[index[v]] for v in left))
    v_right = float(sum(chain.vol[index[v]] for v in right))
    return chain.w_bridge / min(v_left, v_right)


def absorption_probabilities(chain: Chain) -> np.ndarray:
    """`B = (I - Q)^{-1} R`, the fixed point of `z <- Q z + R`.

    Solved rather than inverted: `numpy.linalg.solve` on `(I - Q)` is the same object
    and is the reference the truncated iteration below is measured against.
    """
    return np.linalg.solve(np.eye(chain.Q.shape[0], dtype=np.float64) - chain.Q,
                           chain.R)


def truncated_absorption(chain: Chain, t: int) -> np.ndarray:
    """`z_t = sum_{m < t} Q^m R`, i.e. `t` steps of `z <- Q z + R` from `z_0 = 0`.

    Exactly what a `t*`-truncated arm can compute. `z_0 = 0` makes the `t = 0` reading
    the zero predictor rather than a partially informed one.
    """
    z = np.zeros_like(chain.R)
    for _ in range(t):
        z = chain.Q @ z + chain.R
    return z


def relative_error(chain: Chain, t: int, *, column: int = 0) -> float:
    """`||z_t - B||_2 / ||B||_2` on one boundary column: the truncation error whose
    decay rate the `lambda_2` prediction is about."""
    exact = absorption_probabilities(chain)[:, column]
    approx = truncated_absorption(chain, t)[:, column]
    return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))


def dead_fraction(chain: Chain, *, column: int = 0) -> float:
    """Fraction of transient nodes whose label is below `DEAD`.

    The cost of the killing dial, measured. A killed walk survives about `t_rel`
    steps, so nodes farther than that from both targets carry no signal, and a label
    that is mostly zero is a degenerate label however good its `lambda_2` is.
    """
    label = absorption_probabilities(chain)[:, column]
    return float((label < DEAD).mean())


def bridge_sweep(spec, weights=BRIDGE_WEIGHTS) -> list[tuple[float, float, float]]:
    """`(w, rho(Q), ergodic lambda_2)` over the bridge weights. THE REFUTED DIAL."""
    case = make_case(*spec)
    out = []
    for w in weights:
        chain = build_chain(case, spec, w)
        out.append((float(w), lambda_2(chain), ergodic_lambda_2(chain)))
    return out


def killing_rate(chain_at_one: Chain, target: float) -> float:
    """`alpha = target / rho(P_TT)`. Exact, because scaling scales the spectrum.

    KILLING ONLY EVER SPEEDS ABSORPTION UP. `alpha` is a probability, so `alpha > 1`
    is not a chain; it is the arithmetic saying the unkilled chain is ALREADY faster
    than the target and that this dial cannot slow it down. Raising here rather than
    returning the number keeps a non-chain from being measured and reported as one --
    `target_separation_sweep` catches the raise and records the configuration as
    unattainable instead of dropping it.
    """
    rho = lambda_2(chain_at_one)
    alpha = target / rho
    if alpha > 1.0:
        raise ValueError(f"rho(P_TT) = {rho!r} is already below the target "
                         f"{target!r}; alpha would be {alpha!r} > 1 and killing "
                         f"cannot slow a chain down")
    return alpha


def engineer(spec, target: float = TARGET) -> dict:
    """Everything this module owes Cameron, for one graph, all of it measured."""
    case = make_case(*spec)
    shipped = build_chain(case, spec, 1.0)
    alpha = killing_rate(shipped, target)
    chain = build_chain(case, spec, 1.0, alpha=alpha)

    lam = lambda_2(chain)
    label = absorption_probabilities(chain)[:, 0]
    ladder = [(t, relative_error(chain, t)) for t in LADDER]
    # The decay ratio far out is the ladder's OWN estimate of lambda_2, and it is the
    # independent check on the prediction rather than a restatement of it.
    deep = relative_error(chain, 40) / relative_error(chain, 39)
    return {
        "name": case.name,
        "n_nodes": len(chain.nodes),
        "n_transient": chain.Q.shape[0],
        "diameter": diameter(chain),
        "bridge": chain.bridge,
        "absorbing": chain.absorbing,
        "target_separation": _bfs(chain.adjacent, chain.absorbing[0])[
            chain.absorbing[1]],
        "rho_P_TT": lambda_2(shipped),
        "t_rel_shipped": 1.0 / (1.0 - lambda_2(shipped)),
        "alpha": alpha,
        "lambda_2": lam,
        "lambda_2_construction": alpha * lambda_2(shipped),
        "lambda_2_ergodic": ergodic_lambda_2(shipped),
        "t_rel": 1.0 / (1.0 - lam),
        "in_band": TARGET_LO <= lam <= TARGET_HI,
        "phi_bridge": bridge_conductance(shipped),
        "mu_2": 1.0 - ergodic_lambda_2(shipped),
        "cheeger_upper_holds": (1.0 - ergodic_lambda_2(shipped)
                                <= 2.0 * bridge_conductance(shipped)),
        "label_std": float(label.std()),
        "label_min": float(label.min()),
        "label_max": float(label.max()),
        "label_median": float(np.median(label)),
        "dead_fraction": dead_fraction(chain),
        "ladder": ladder,
        "ladder_predicted": [(t, lam ** t) for t in LADDER],
        "decay_ratio": deep,
    }


def degenerate_single_target(spec) -> dict:
    """MUST-FIRE. One absorbing state and no killing makes the label the constant `1`.

    This is the fifteenth candidate for the vacuous-control list and it is caught here
    rather than after a run: with a single absorbing class every transient state
    absorbs there with probability exactly one, the label's standard deviation is
    exactly zero, and NRMSE against it is undefined. Two targets are not a modelling
    preference, they are what makes the oracle a function at all.
    """
    chain = build_chain(make_case(*spec), spec, 1.0, single_target=True)
    label = absorption_probabilities(chain)[:, 0]
    # CONSTANT TO FLOAT64, not bit-exactly constant: `(I - Q)^{-1} R` is a dense
    # solve, so the measured deviation from the exact `1` is round-off. Both
    # graphs read `max|label - 1|` at `1.1102230246251565e-14` and
    # `1.0658141036401503e-14`, and `label.std()` at `3.1086244689504383e-15` and
    # `4.062193776730142e-15`. Testing `== 0.0` would have made this control
    # UNFIREABLE for a reason that is about LAPACK and not about the task, which
    # is the shape of a vacuous control rather than a working one.
    return {"label_std": float(label.std()),
            "max_abs_dev_from_one": float(np.abs(label - 1.0).max()),
            "fires": float(label.std()) < DEGENERATE_STD}


def no_kill_control(spec) -> dict:
    """MUST-FIRE. The shipped chain, `alpha = 1`, must sit OUTSIDE the band, so the
    band check is shown to reject something and the dial is shown to be needed."""
    lam = lambda_2(build_chain(make_case(*spec), spec, 1.0))
    return {"alpha": 1.0, "lambda_2": lam, "in_band": TARGET_LO <= lam <= TARGET_HI,
            "fires": not (TARGET_LO <= lam <= TARGET_HI)}


def bridge_dial_refuted(spec) -> dict:
    """MUST-FIRE. No bridge weight puts `rho(Q)` in the band, and the sweep is
    monotone in the direction OPPOSITE to the one the construction assumed."""
    sweep = bridge_sweep(spec)
    rhos = [r for _w, r, _e in sweep]
    ergs = [e for _w, _r, e in sweep]
    return {"sweep": sweep,
            "rho_min": min(rhos), "rho_max": max(rhos),
            "any_in_band": any(TARGET_LO <= r <= TARGET_HI for r in rhos),
            "increasing_in_w": rhos == sorted(rhos),
            "ergodic_min": min(ergs),
            "ergodic_reaches_band": any(TARGET_LO <= e <= TARGET_HI for e in ergs),
            "fires": not any(TARGET_LO <= r <= TARGET_HI for r in rhos)}


def target_separation_sweep(spec, separations=(4, 8, 12, 16, 20, 24),
                            target: float = TARGET) -> list[dict]:
    """`dead_fraction` against target separation, at a fixed engineered `lambda_2`.

    The route out of the killing dial's cost, mapped rather than argued: if a closer
    target pair carries more of the graph, the corpus can trade non-locality for
    signal knowing exactly what each hop costs. Cameron owns that trade; this owns the
    numbers it is made against.
    """
    case = make_case(*spec)
    base = build_chain(case, spec, 1.0)
    out = []
    seen: set[int] = set()
    for want in separations:
        best = None
        for u in base.nodes:
            dist = _bfs(base.adjacent, u)
            for v in base.nodes:
                if v == u or v not in dist:
                    continue
                gap = abs(dist[v] - want)
                if best is None or gap < best[0]:
                    best = (gap, u, v, dist[v])
        _gap, u, v, sep = best
        # The graph's diameter caps the separation, so a requested 12, 16, 20 and 24
        # all resolve to the same pair on an 8-diameter graph. Printing that pair four
        # times would read as four measurements of one fact.
        if sep in seen:
            continue
        seen.add(sep)
        probe = build_chain(case, spec, 1.0, targets=[u, v])
        try:
            alpha = killing_rate(probe, target)
        except ValueError as exc:
            out.append({"separation": sep, "targets": (u, v), "alpha": None,
                        "lambda_2": lambda_2(probe), "dead_fraction": None,
                        "label_std": None, "label_median": None,
                        "unattainable": str(exc)})
            continue
        chain = build_chain(case, spec, 1.0, alpha=alpha, targets=[u, v])
        label = absorption_probabilities(chain)[:, 0]
        out.append({"separation": sep, "targets": (u, v), "alpha": alpha,
                    "lambda_2": lambda_2(chain), "dead_fraction": dead_fraction(chain),
                    "label_std": float(label.std()),
                    "label_median": float(np.median(label)),
                    "unattainable": None})
    return out


def _fmt(v: float) -> str:
    return f"{v:.10f}"


def report() -> str:
    lines = [
        "lambda_2 := rho(Q), the Perron root of the transient block.",
        "P = [[Q, R], [0, I]] is block triangular, so spec(P) = spec(Q) u {1,1};",
        "the literal second eigenvalue of P is 1 and carries no information.",
        f"target band [{TARGET_LO}, {TARGET_HI}], midpoint {TARGET}, ladder {LADDER}",
        "",
        "REFUTED DIAL -- bridge weight. Cheeger's object is the ERGODIC lambda_2,",
        "which is not what the t* ladder truncates.",
    ]
    for spec in REROUTED_CASES:
        d = bridge_dial_refuted(spec)
        lines.append(f"  {spec[0]}")
        lines.append("      w        rho(Q)          ergodic lambda_2")
        for w, rho, erg in d["sweep"]:
            lines.append(f"   {w:8.1f}  {_fmt(rho)}   {_fmt(erg)}")
        lines.append(f"   rho(Q) in [{_fmt(d['rho_min'])}, {_fmt(d['rho_max'])}]"
                     f"  any in band {d['any_in_band']}"
                     f"  increasing in w {d['increasing_in_w']}"
                     f"  FIRES {d['fires']}")
        lines.append(f"   ergodic lambda_2 min {_fmt(d['ergodic_min'])}"
                     f"  reaches band {d['ergodic_reaches_band']}")
    lines.append("")
    lines.append("ENGINEERED DIAL -- killing rate. rho(alpha P_TT) = alpha rho(P_TT),")
    lines.append("exact, so alpha = target / rho(P_TT) with no bracket and no search.")
    for spec in REROUTED_CASES:
        r = engineer(spec)
        lines.append("")
        lines.append(f"{r['name']}   nodes={r['n_nodes']} transient={r['n_transient']}"
                     f" diameter={r['diameter']}  bridge={r['bridge']}"
                     f"  absorbing={r['absorbing']} separation={r['target_separation']}")
        lines.append(f"   shipped rho(P_TT) {_fmt(r['rho_P_TT'])}"
                     f"   t_rel {r['t_rel_shipped']:.4f}"
                     f"   ergodic lambda_2 {_fmt(r['lambda_2_ergodic'])}")
        lines.append(f"   Phi_bridge {_fmt(r['phi_bridge'])}  mu_2 {_fmt(r['mu_2'])}"
                     f"  mu_2 <= 2 Phi {r['cheeger_upper_holds']}")
        lines.append(f"   alpha {_fmt(r['alpha'])}"
                     f"   construction says {_fmt(r['lambda_2_construction'])}"
                     f"   MEASURED {_fmt(r['lambda_2'])}")
        lines.append(f"   t_rel {r['t_rel']:.4f}   IN BAND {r['in_band']}")
        lines.append(f"   label  std {_fmt(r['label_std'])}  min {_fmt(r['label_min'])}"
                     f"  median {_fmt(r['label_median'])}  max {_fmt(r['label_max'])}"
                     f"  dead<{DEAD} {_fmt(r['dead_fraction'])}")
        lines.append("   ladder   t   rel err to fixed point   lambda_2**t")
        for (t, e), (_t, p) in zip(r["ladder"], r["ladder_predicted"]):
            lines.append(f"            {t:<3d} {_fmt(e)}            {_fmt(p)}")
        lines.append(f"   decay ratio at t=40/39 {_fmt(r['decay_ratio'])}"
                     f"   vs lambda_2 {_fmt(r['lambda_2'])}"
                     f"   |diff| {_fmt(abs(r['decay_ratio'] - r['lambda_2']))}")
    lines.append("")
    lines.append("TARGET SEPARATION vs DEAD FRACTION, at the engineered lambda_2")
    for spec in REROUTED_CASES:
        lines.append(f"  {spec[0]}")
        for row in target_separation_sweep(spec):
            if row["alpha"] is None:
                lines.append(f"   sep {row['separation']:<3d} targets {row['targets']}"
                             f"  UNATTAINABLE -- rho(P_TT) "
                             f"{_fmt(row['lambda_2'])} is already below the target,"
                             f" killing cannot slow a chain down")
                continue
            lines.append(f"   sep {row['separation']:<3d} targets {row['targets']}"
                         f"  alpha {_fmt(row['alpha'])}"
                         f"  lambda_2 {_fmt(row['lambda_2'])}"
                         f"  dead {_fmt(row['dead_fraction'])}"
                         f"  std {_fmt(row['label_std'])}")
    lines.append("")
    lines.append("MUST-FIRE CONTROLS")
    for spec in REROUTED_CASES:
        d = degenerate_single_target(spec)
        n = no_kill_control(spec)
        b = bridge_dial_refuted(spec)
        lines.append(f"  {spec[0]}")
        lines.append(f"    single absorbing target, alpha=1: label std "
                     f"{d['label_std']:.6e}  max|label - 1| "
                     f"{d['max_abs_dev_from_one']:.6e}"
                     f"  (constant to float64, bar {DEGENERATE_STD:.0e})"
                     f"  FIRES {d['fires']}")
        lines.append(f"    no killing, alpha=1:              lambda_2 "
                     f"{_fmt(n['lambda_2'])}  in band {n['in_band']}"
                     f"  FIRES {n['fires']}")
        lines.append(f"    bridge dial over w<=4096:         band reached "
                     f"{b['any_in_band']}  FIRES {b['fires']}")
    return "\n".join(lines)


def demo() -> None:
    """One runnable check: the band is hit by construction and confirmed by an
    independent eigensolve, the controls fire, and the ladder's own decay rate agrees
    with the eigenvalue it was supposed to be predicted by."""
    for spec in REROUTED_CASES:
        r = engineer(spec)
        assert r["in_band"], (r["name"], r["lambda_2"])
        assert abs(r["lambda_2"] - TARGET) < 1e-12, (r["name"], r["lambda_2"])
        assert abs(r["lambda_2"] - r["lambda_2_construction"]) < 1e-12, r["name"]
        assert 10.0 <= r["t_rel"] <= 20.0, (r["name"], r["t_rel"])
        assert r["cheeger_upper_holds"], (r["name"], r["mu_2"], r["phi_bridge"])

        # PASS-half non-degeneracy: the label must actually vary, and the shallow
        # rungs must actually be far from the fixed point while the deep one is near.
        assert r["label_std"] > 0.05, (r["name"], r["label_std"])
        assert r["label_max"] - r["label_min"] > 0.4, r["name"]
        errs = [e for _t, e in r["ladder"]]
        assert errs[0] > 0.5, errs
        assert errs == sorted(errs, reverse=True), errs
        assert errs[-1] < 0.1, errs
        # The prediction: the ladder's measured decay rate IS lambda_2. The tolerance
        # is loose on purpose and the achieved gap is printed by `report`, because on
        # the 1024 case the near-zero tail of the label keeps sub-dominant modes alive
        # at t = 40 and the ratio has not settled to 1e-3.
        assert abs(r["decay_ratio"] - r["lambda_2"]) < 2e-2, (r["decay_ratio"],
                                                              r["lambda_2"])
        # Controls seen to fire, all three.
        assert degenerate_single_target(spec)["fires"]
        assert no_kill_control(spec)["fires"]
        assert bridge_dial_refuted(spec)["fires"]
    print("foreman_lambda2 demo OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        print(report())
