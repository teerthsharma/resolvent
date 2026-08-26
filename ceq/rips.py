"""Geodesic Vietoris-Rips graphs on the sphere, spanning a connectivity transition.

PROVENANCE. This is a Python port of `test/benchmark/island_benchmark_test.cc` as it
stood at commit `5d91d878c2b2f04847130c80953a73180c1d93d6` in
`google-deepmind/mujoco`, Apache-2.0, authored by this project's author inside pull
request #3396. The file was added and then removed within that same pull request and
did not reach the merge commit, where it returns 404. Its removal was an upstream
scope decision rather than a defect finding: the merged change carries the simplest
test that establishes correctness, and a benchmark corpus is outside what that review
wanted. The generator is reproduced here with its arithmetic unchanged.

WHY THIS CORPUS AND NOT A FRESH ONE. Its own header states the property that makes it
useful: it "spans the connectivity transition of points sampled on S^2". The corpus is
parameterised by a target degree, and the component count falls from 178 to 1 across
the six cases. That parameter is a difficulty dial with a critical point rather than an
arbitrary knob, which is what an iterate-to-stable task needs and what this project's
existing tasks lack: both registered oracles are closed-form functions of their input,
so an arm that iterates to a fixed point has nothing to compute, and the deciding
contrast came in near zero by construction.

WHAT IS PORTED AND WHAT IS NOT. The point sampler, the Rips edge rule, the bridge
edge, the static and repeated incidence rows and the component labelling are ported.
The disjoint-set implementation under test upstream is not: that algorithm is the
subject there and is irrelevant here, where the graphs themselves are the object.

FLOATING POINT. `SplitMix64` is exact integer arithmetic and ports exactly. The
sampler and the radius rule call `sqrt`, `asin`, `cos` and `sin`, so agreement with a
C++ build depends on both toolchains' libm being correctly rounded at these inputs,
which the C++ standard does not guarantee. The counts are therefore checked against an
independent derivation rather than assumed to transfer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["GraphCase", "splitmix64", "sample_sphere", "rips_edges",
           "components", "make_case", "corpus", "CASES"]

_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_TWO_PI = 6.283185307179586476925286766559


class _State:
    """A mutable 64-bit word, standing in for the C++ `uint64_t&` parameter."""

    __slots__ = ("v",)

    def __init__(self, seed: int) -> None:
        self.v = seed & _MASK


def splitmix64(state: _State) -> int:
    state.v = (state.v + _GOLDEN) & _MASK
    value = state.v
    value = ((value ^ (value >> 30)) * _MIX1) & _MASK
    value = ((value ^ (value >> 27)) * _MIX2) & _MASK
    return value ^ (value >> 31)


def _uniform01(state: _State) -> float:
    """`(x >> 11) * 2**-53`, the 53-bit construction, reproduced exactly."""
    return (splitmix64(state) >> 11) * (2.0 ** -53)


def sample_sphere(count: int, seed: int) -> list[tuple[float, float, float]]:
    """`count` points uniform on the unit sphere, by the Archimedes z-band method.

    Sampling `z` uniformly on `[-1, 1]` and the azimuth uniformly is exactly uniform
    on the sphere, because the projection from the sphere to the enclosing cylinder
    preserves area. Sampling two angles uniformly instead would crowd the poles.
    """
    st = _State(seed)
    pts = []
    for _ in range(count):
        z = 2.0 * _uniform01(st) - 1.0
        angle = _TWO_PI * _uniform01(st)
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        pts.append((radial * math.cos(angle), radial * math.sin(angle), z))
    return pts


def rips_edges(points, target_degree: float) -> list[tuple[int, int]]:
    """Every pair inside a geodesic ball whose radius is set by the target degree.

    The radius solves for an expected degree rather than being chosen directly. A
    spherical cap of geodesic radius `r` covers a fraction `sin^2(r/2)` of the
    sphere, so setting that fraction equal to `p = target_degree/(n-1)` gives
    `r = 2*asin(sqrt(p))`. Comparing dot products against `cos(r)` then decides
    membership without computing any distance.
    """
    n = len(points)
    probability = target_degree / (n - 1)
    radius = 2.0 * math.asin(math.sqrt(probability))
    minimum_dot = math.cos(radius)
    out = []
    for i in range(n):
        xi, yi, zi = points[i]
        for j in range(i + 1, n):
            xj, yj, zj = points[j]
            if xi * xj + yi * yj + zi * zj >= minimum_dot:
                out.append((i, j))
    return out


def components(node_count: int, edges) -> list[int]:
    """Component label per node, with `-1` for a node touched by no edge.

    Labels are assigned in ascending order of each component's smallest node, so the
    labelling is canonical and two runs compare directly without a permutation
    search.
    """
    adj: list[list[int]] = [[] for _ in range(node_count)]
    active = [False] * node_count
    for a, b in edges:
        active[a] = True
        active[b] = True
        if a != b:
            adj[a].append(b)
            adj[b].append(a)
    label = [-1] * node_count
    nxt = 0
    for start in range(node_count):
        if not active[start] or label[start] != -1:
            continue
        stack = [start]
        label[start] = nxt
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if label[v] == -1:
                    label[v] = nxt
                    stack.append(v)
        nxt += 1
    return label


@dataclass(frozen=True)
class GraphCase:
    name: str
    node_count: int
    target_degree: float
    seed: int
    edges: list[tuple[int, int]] = field(repr=False)
    partition: list[int] = field(repr=False)
    n_components: int
    pre_bridge_components: int


def _count(node_count: int, edges) -> int:
    lab = components(node_count, edges)
    return len({x for x in lab if x >= 0})


def _add_critical_bridge(points, edges, node_count):
    """Join the two nearest distinct components with one geodesic edge.

    One edge is what carries a graph across the connectivity transition, which is the
    property this corpus exists to exhibit. Adding several would step over it.
    """
    lab = components(node_count, edges)
    best = None
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if lab[i] < 0 or lab[j] < 0 or lab[i] == lab[j]:
                continue
            xi, yi, zi = points[i]
            xj, yj, zj = points[j]
            dot = max(-1.0, min(1.0, xi * xj + yi * yj + zi * zj))
            d = math.acos(dot)
            if best is None or d < best[0]:
                best = (d, i, j)
    return None if best is None else (best[1], best[2])


def make_case(name: str, node_count: int, target_degree: float, seed: int,
              bridge: bool, static_rows: bool, repeated_rows: bool) -> GraphCase:
    points = sample_sphere(node_count, seed)
    edges = rips_edges(points, target_degree)
    pre = -1
    if bridge:
        pre = _count(node_count, edges)
        e = _add_critical_bridge(points, edges, node_count)
        if e is not None and pre >= 2:
            edges = edges + [e]
    unique = list(edges)
    if static_rows:
        edges = edges + [(node, node) for node in range(0, node_count, 17)]
    if repeated_rows:
        extra = []
        for i in range(0, len(unique), 11):
            a, b = unique[i]
            extra.append((a, b))
            extra.append((b, a))
        edges = edges + extra
    lab = components(node_count, edges)
    return GraphCase(name, node_count, target_degree, seed, edges, lab,
                     len({x for x in lab if x >= 0}), pre)


#: The six cases, with the upstream names, seeds and degree expressions verbatim.
CASES = [
    ("StableSparse_S2Rips_64", 64, 2.0, 0x33960001, False, False, False),
    ("CriticalBridge_S2Rips_256", 256, 0.75 * math.log(256.0), 0x33960002,
     True, False, False),
    ("SupercriticalDense_S2Rips_256", 256, 2.0 * math.ceil(math.log(256.0)),
     0x33960003, False, False, False),
    ("GroundedStaticRepeated_S2Rips_256", 256, 2.0 * math.ceil(math.log(256.0)),
     0x33960004, False, True, True),
    ("StableRepeated_S2Rips_1024", 1024, 2.0, 0x33960005, False, False, True),
    ("CriticalLarge_S2Rips_1024", 1024, math.ceil(math.log(1024.0)), 0x33960006,
     True, False, False),
]


def corpus() -> list[GraphCase]:
    return [make_case(*c) for c in CASES]
