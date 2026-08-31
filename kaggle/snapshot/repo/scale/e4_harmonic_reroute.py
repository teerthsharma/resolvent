"""The RULE 5 reroute for X17, measured rather than proposed.

WHAT DIED. `scale/e4_harmonic.py` builds the absorbing-chain ladder on the E4'
single-bridge graph and both gates fire there, but the contract's `lambda_2` band is
unreachable on that substrate and Cheeger says why: a one-edge bridge has
`phi = 1/vol(S)`, so `g <= 2/vol(S)` and `t_rel >= vol(S)/2 = 1372.50`, against a
band of 10 to 20. Shortening `t_rel` with a kill rate does reach the band -- 15.99 at
`kill = 1/16` -- but collapses the label to sd 0.038445 about a mean of 0.004334.

WHAT THE FRONTIER ACTUALLY SAID, AND IT IS NOT WHAT IT LOOKED LIKE. In band the
decoder reads 0.732380, which is above `PASS_BAR = 0.5` and therefore NOT a K-1
strike. The blocker in band was never decodability. It was DEGENERACY: absorption at
a single node is rare when the walk dies in about sixteen steps, so almost every node
carries a label of essentially zero.

THE REROUTE, AND WHY IT IS SHARPER. Two changes, each aimed at the measured cause.
First, absorb on two antipodal spherical CAPS rather than two single nodes, so
absorption is common inside the kill horizon and the label stops being a spike.
Second, read the CONDITIONAL label `B_A / (B_A + B_B)` -- which cap wins given that
the walk is absorbed at all -- rather than the raw absorption probability. The second
change is the one that carries it, and the difference is large enough that reporting
only the raw label would have hidden the route entirely: at `kill = 1/16` the raw
label reads 0.340523 and IS struck by K-1, while the conditional label reads 0.838080
and is not.

WHAT IT BUYS AND WHAT IT COSTS, both measured below. It buys the band
(`t_rel ~ 15.62`) and a balanced label (sd 0.468073 about a mean of 0.494608, against
0.038445 about 0.004334). It costs decoder margin: the single-bridge fixed point sits
at 0.973819, clear of `FAIL_BAR = 0.9`, and this lands at 0.838080, between the bars.
Clearing `FAIL_BAR` is one further measured step -- shrinking the caps lengthens the
distance a radius-5 ball has to cover -- and that step is NOT taken here, so this
module reroutes the goal without yet delivering it.
"""
from __future__ import annotations

import time

import numpy as np

from ceq.rips import components, sample_sphere
from scale.e4_harmonic import _np_edges
from scale.rips_gate import FAIL_BAR, PASS_BAR, ball, fit_eval

#: `n`, target degree, seed. The degree is SUPERCRITICAL here, unlike the E4' cases:
#: the reroute drops the bridge, so it needs one connected giant rather than two
#: comparable components. Measured giant 2046 of 2048.
CASE = (2048, 8.0, 0x33960005)

#: Cap half-angle as a dot product against the pole. `0.95` gives caps of 58 and 46
#: nodes, about 5% of the giant between them.
CAP_DOT = 0.95

#: The kill rate that puts `t_rel` in the contract's band. Not a free choice: the
#: band `lambda_2 in [0.90, 0.95]` is `t_rel in [10, 20]`, and 1/16 lands at 15.62.
KILL = 1 / 16


def build(node_count: int, target_degree: float, seed: int, cap_dot: float = CAP_DOT):
    """Giant component of a supercritical Rips graph, plus two antipodal caps."""
    points = np.asarray(sample_sphere(node_count, seed), dtype=float)
    edges = _np_edges(points, target_degree)
    label = components(node_count, edges)
    sizes: dict[int, int] = {}
    for x in label:
        if x >= 0:
            sizes[x] = sizes.get(x, 0) + 1
    giant_label = max(sizes, key=lambda c: sizes[c])
    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    for a, b in edges:
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    giant = [v for v in range(node_count) if label[v] == giant_label]
    height = points @ np.array([0.0, 0.0, 1.0])
    cap_a = {v for v in giant if height[v] >= cap_dot}
    cap_b = {v for v in giant if height[v] <= -cap_dot}
    return adjacency, giant, cap_a, cap_b


def chain(adjacency, giant, cap_a, cap_b, kill: float):
    """`(Q, R_A, R_B, transient)` for the killed walk absorbed on either cap."""
    absorbing = cap_a | cap_b
    transient = [v for v in giant if v not in absorbing]
    index = {v: i for i, v in enumerate(transient)}
    m = len(transient)
    q = np.zeros((m, m))
    ra = np.zeros(m)
    rb = np.zeros(m)
    for v in transient:
        step = (1.0 - kill) / len(adjacency[v])
        for u in adjacency[v]:
            if u in cap_a:
                ra[index[v]] += step
            elif u in cap_b:
                rb[index[v]] += step
            elif u in index:
                q[index[v], index[u]] += step
    return q, ra, rb, transient


def features(adjacency, transient, cap_a, cap_b, radius: int = 5):
    """The same decoder `scale/e4_harmonic.py` uses, with cap membership in place of
    endpoint membership. Returns `(all, ball_sizes_only)`."""
    sizes, targets = [], []
    for v in transient:
        size_row = [1.0, float(len(adjacency[v]))]
        target_row = []
        for r in range(1, radius + 1):
            reached = ball(adjacency, v, r)
            size_row.append(float(len(reached)))
            target_row += [float(bool(reached & cap_a)), float(bool(reached & cap_b))]
        sizes.append(size_row)
        targets.append(target_row)
    sizes = np.asarray(sizes)
    return np.hstack([sizes, np.asarray(targets)]), sizes


def _rho_by_residual(q, r, target, steps: int = 400, tail: int = 50) -> float:
    """`rho(Q)` from the decay of `||z_k - z*||`, avoiding a dense eigensolve.

    The transient block here is 1942 x 1942 and a full eigendecomposition of it ran
    past fifteen minutes without returning. The residual of the fixed-point iteration
    decays at exactly `rho(Q)` asymptotically, so the median of the last `tail`
    successive ratios is the same number at a fraction of the cost. It is reported as
    an estimate and written `~` wherever it is printed, because it is one.
    """
    z = np.zeros_like(r)
    previous = None
    ratios = []
    for _ in range(steps):
        z = q @ z + r
        residual = float(np.linalg.norm(z - target))
        if previous is not None and previous > 0.0:
            ratios.append(residual / previous)
        previous = residual
    return float(np.median(ratios[-tail:]))


def report() -> str:
    lines: list[str] = []
    w = lines.append
    started = time.time()
    n, degree, seed = CASE
    adjacency, giant, cap_a, cap_b = build(n, degree, seed)
    q0, ra0, rb0, transient = chain(adjacency, giant, cap_a, cap_b, KILL)
    x_all, x_ball = features(adjacency, transient, cap_a, cap_b)
    order = np.random.RandomState(0).permutation(len(transient))
    x_all, x_ball = x_all[order], x_ball[order]

    w("=" * 78)
    w("X17 REROUTE -- antipodal caps, killed walk, CONDITIONAL label")
    w("=" * 78)
    w(f"  n={n}  target_degree={degree}  seed={hex(seed)}  cap_dot={CAP_DOT}")
    w(f"  giant={len(giant)}/{n}   capA={len(cap_a)}   capB={len(cap_b)}   "
      f"transient={len(transient)}")
    w(f"  bars inherited unchanged: PASS_BAR={PASS_BAR}  FAIL_BAR={FAIL_BAR}")
    w("")
    for kill in (KILL, 0.0):
        q, ra, rb, _ = chain(adjacency, giant, cap_a, cap_b, kill)
        identity = np.eye(len(ra))
        za = np.linalg.solve(identity - q, ra)
        zb = np.linalg.solve(identity - q, rb)
        ratio = za / np.maximum(za + zb, 1e-300)
        rho = _rho_by_residual(q, ra, za)
        w(f"  kill={kill:.5f}   rho(Q)~{rho:.6f}   t_rel~{1.0 / (1.0 - rho):.2f}"
          + ("   <-- INSIDE the contract band [10, 20]" if 10.0 <= 1.0 / (1.0 - rho)
             <= 20.0 else "   (outside the band)"))
        for name, y in (("raw B_A", za[order]), ("conditional", ratio[order])):
            decoded = fit_eval(x_all, y)
            verdict = ("STRUCK by K-1" if decoded <= PASS_BAR
                       else "clears FAIL_BAR" if decoded >= FAIL_BAR
                       else "between the bars")
            w(f"      {name:12s} sd={y.std():.6f}  mean={y.mean():.6f}  "
              f"decoder={decoded:.6f}  ball-only={fit_eval(x_ball, y):.6f}  {verdict}")
        w(f"      planted degree control decoder={fit_eval(x_all, x_all[:, 1]):.6f}"
          f"   (must fire)")
        w("")
    w("  THE READING. The conditional label buys the band and a balanced spread that")
    w("  the single-bridge in-band configuration could not have: sd 0.468073 about a")
    w("  mean of 0.494608, against 0.038445 about 0.004334. The raw absorption")
    w("  probability at the same kill rate is STRUCK by K-1 at 0.340523, so reporting")
    w("  only the raw label would have hidden the route.")
    w("")
    w("  WHAT IS NOT DELIVERED. The decoder lands at 0.838080, between the bars. The")
    w("  single-bridge fixed point clears FAIL_BAR at 0.973819 and this does not, so")
    w("  the reroute trades decoder margin for the band. Clearing FAIL_BAR is one")
    w("  further measured step -- smaller caps lengthen what a radius-5 ball must")
    w("  cover -- and that step is NOT taken here.")
    w("")
    w(f"  [{time.time() - started:.1f}s, CPU]")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
