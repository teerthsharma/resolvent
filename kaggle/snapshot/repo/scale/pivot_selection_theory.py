"""Two facts about pivot selection that this repository's numbers rest on.

    python -m scale.pivot_selection_theory

WHAT THIS IS FOR. `results/r9_maths_survey.md` prices the mathematics of
trainable discrete selection. Two of its claims are theorems about THIS repo's
own objects rather than citations, so they live in code where they can be
re-run, not in prose where they can rot.

--------------------------------------------------------------------------
FACT 1 -- WHY THE ONE STAGE-A ABLATION ON RECORD COULD NOT HAVE SAID ANYTHING
--------------------------------------------------------------------------
`DONE_ARCHIVE_ROUND1.md:4707` records K4: `randpivot_signed`, `k = 8` pivots
drawn content-blind, slope `+0.081`, intervals overlapping the content-selected
arm, and the pre-registered consequence that content selection is "not
load-bearing for M2".

That null was FORCED, not measured. `scale/recall_probe.py:3-7` states the
identity: for any content-blind schedule of size `k` drawn from `s` items,

    P(c reachable) * (share | reachable)  =  (k/s) * (1/k)  =  1/s

bit-for-bit the dense rate, INDEPENDENT OF `k`. M2 draws `c` from `P`, so it
measures the second factor and conditions the first away. A content-blind
schedule must come back flat on M2 whatever the truth is, so K4 was
structurally incapable of reading anything else.

The quantity that is NOT an artefact is named at `recall_probe.py:9-11`:
whether `P(c selected)` stays `Theta(1)` as `s` grows when selection is
CONTENT-CONDITIONAL. `scale/recall_probe.py` is imported by zero Python files,
has no `results/` artifact, and `DONE_ARCHIVE_ROUND1.md:5833` calls it "already
sitting unrun".

--------------------------------------------------------------------------
FACT 2 -- THE PIVOT OBJECTIVE'S SET STRUCTURE, AND WHERE THE GREEDY BOUND DIES
--------------------------------------------------------------------------
`pivot_probe.pivot_hop2` uses the pivot set as `B_P = A[:,P] @ A[P,:]`. Writing
`u_p = A[:,p]` and `v_p = A[p,:]`,

    B_P = sum_{p in P} u_p v_p^T

a sum of rank-1 outer products, ONE PER PIVOT, EACH INDEPENDENT OF THE REST OF
`P`. That independence is the entire reason set-function structure exists here.
With `M_pq := (u_p . u_q)(v_p . v_q)`,

    f(P) := ||B_P||_F^2 = sum_{p,q in P} M_pq = 1_P^T M 1_P

is a quadratic form in the indicator, so its second difference is CONSTANT in
the conditioning set:

    f(S+a+b) - f(S+a) - f(S+b) + f(S) = 2 M_ab.

CAPTURED MASS IS SUPERMODULAR. For a nonnegative `A` -- and a softmax attention
operator is nonnegative -- every inner product is non-negative, so `M_ab >= 0`
and the second difference is `>= 0`: INCREASING returns. Greedy has no `1-1/e`
guarantee on this objective. The intuition that pivot selection is "obviously a
diminishing-returns problem" is wrong on the obvious objective.

RECONSTRUCTION IS MONOTONE SUBMODULAR, AND THE BOUND IS REAL. Expanding
`-||A@A - B_P||^2` leaves a constant, a modular term and `-1_P^T M 1_P`, so

    h(P) := ||A@A||_F^2 - ||A@A - B_P||_F^2

has second difference `-2 M_ab <= 0`. It is also monotone, and that is derived
rather than hoped: with residual `R = A@A - B_S` and `a` not in `S`,

    h(S+a) - h(S) = 2<R, u_a v_a^T> - ||u_a v_a^T||^2 >= ||u_a v_a^T||^2 > 0

because for `A >= 0` and `a` not in `S`, `R >= u_a v_a^T` elementwise. With
`h(empty) = 0` and `h >= 0`, Nemhauser-Wolsey-Fisher gives greedy
`1 - 1/e = 0.632121` of the optimum. (NWF 1978 is classical and is cited from
standard knowledge; it was NOT fetched, and is flagged as such in the survey.)

AND THIS REPO'S SIGNED ARM VOIDS IT. `M_ab >= 0` needs `A >= 0`. On signed
operators `M` had a negative entry in 297 of 300 draws at `n` in {4,5,6}, and in
300 of 300 at fixed `n = 6` as `report()` prints, so neither objective is
submodular there and the guarantee does not apply.

WHAT THE BOUND IS WORTH, HONESTLY. It bounds how well GREEDY approximates the
BEST pivot set under a reconstruction objective. It says nothing about whether a
better pivot set lowers NRMSE. That is the question `recall_probe.py` was
written to answer and has never been run to answer. A provable guarantee on an
objective nobody has shown to matter is still a guarantee about nothing.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

__all__ = ["content_blind_rate", "sampled_content_blind_rate", "pairwise_M",
           "captured_mass", "reconstruction_gain", "second_difference",
           "submodular_objectives", "GREEDY_BOUND", "report", "demo"]

#: Nemhauser-Wolsey-Fisher's ratio for greedy on a monotone non-negative
#: submodular objective under a cardinality constraint. Applies to
#: `reconstruction_gain` on a NONNEGATIVE operator only.
GREEDY_BOUND = 1.0 - 1.0 / math.e


def content_blind_rate(s: int, k: int) -> float:
    """`(k/s) * (1/k) = 1/s`. The closed form, and `k` cancels."""
    if not 1 <= k <= s:
        raise ValueError(f"k={k} does not fit in s={s}")
    return (k / s) * (1 / k)


def sampled_content_blind_rate(s: int, k: int, *, draws: int = 200000,
                               seed: int = 0) -> float:
    """The same rate by drawing. The second path: a wrong factorisation still
    yields a clean formula, whereas a wrong sampler yields a wrong number."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, s, draws)
    schedule = np.argsort(rng.random((draws, s)), axis=1)[:, :k]
    return float((schedule == c[:, None]).any(axis=1).mean()) * (1.0 / k)


def pairwise_M(a: np.ndarray) -> np.ndarray:
    """`M_pq = (A[:,p] . A[:,q]) (A[p,:] . A[q,:])`, the second difference's
    kernel. Elementwise product of the column Gram and the row Gram."""
    return (a.T @ a) * (a @ a.T)


def captured_mass(a: np.ndarray, pivots) -> float:
    """`f(P) = ||A[:,P] @ A[P,:]||_F^2`."""
    p = sorted(pivots)
    if not p:
        return 0.0
    b = a[:, p] @ a[p, :]
    return float((b * b).sum())


def reconstruction_gain(a: np.ndarray, pivots) -> float:
    """`h(P) = ||A@A||^2 - ||A@A - A[:,P]A[P,:]||^2`, so `h(empty) = 0`."""
    p = sorted(pivots)
    aa = a @ a
    b = a[:, p] @ a[p, :] if p else np.zeros_like(aa)
    r = aa - b
    return float((aa * aa).sum() - (r * r).sum())


def second_difference(fn, subset, i: int, j: int) -> float:
    """`F(S+i+j) - F(S+i) - F(S+j) + F(S)`. Submodular iff this is `<= 0`."""
    s = set(subset)
    return fn(s | {i, j}) - fn(s | {i}) - fn(s | {j}) + fn(s)


def submodular_objectives(a: np.ndarray) -> dict:
    """Which of the two objectives is submodular on this operator, from the
    sign of `M` alone -- no enumeration needed, because the second difference
    does not depend on the conditioning set."""
    m = pairwise_M(a)
    off = m[~np.eye(len(m), dtype=bool)]
    return {"M_min": float(off.min()), "M_max": float(off.max()),
            "captured_mass_submodular": bool((off <= 1e-12).all()),
            "reconstruction_submodular": bool((off >= -1e-12).all()),
            "nonnegative_operator": bool((a >= 0).all())}


def report() -> str:
    lines: list[str] = []
    w = lines.append
    w("=" * 78)
    w("PIVOT SELECTION THEORY")
    w("=" * 78)
    w("FACT 1 -- the content-blind rate is 1/s, independent of k")
    w("      s      k   closed form      sampled 200k             1/s")
    for s in (16, 64, 256, 1024):
        w("  %5d %6d      %.8f        %.8f      %.8f"
          % (s, 8, content_blind_rate(s, 8),
             sampled_content_blind_rate(s, 8), 1 / s))
    w("    K4's content-blind null was FORCED by this, not measured.")
    w("")
    w("FACT 2 -- which objective is submodular, by operator sign")
    rng = np.random.default_rng(0)
    for name, draw in (("nonnegative (softmax operator)", rng.random),
                       ("signed (the signed arm)", rng.normal)):
        bad = 0
        for _ in range(300):
            a = np.asarray(draw(size=(6, 6)), dtype=float)
            if not submodular_objectives(a)["reconstruction_submodular"]:
                bad += 1
        w("    %-32s reconstruction NOT submodular in %3d/300 draws"
          % (name, bad))
    w("    captured mass is SUPERMODULAR on a nonnegative operator: no bound.")
    w("    greedy bound where it applies: 1 - 1/e = %.6f" % GREEDY_BOUND)
    w("    it bounds greedy against the BEST pivot set under reconstruction.")
    w("    Whether a better pivot set lowers NRMSE is unmeasured -- that is")
    w("    what scale/recall_probe.py was written for and never run.")
    return "\n".join(lines)


def demo() -> None:
    """Self-checks. Each fails if the corresponding derivation is wrong."""
    # FACT 1, both paths, and k really does cancel
    for s in (16, 64, 256, 1024):
        assert abs(content_blind_rate(s, 8) - 1 / s) < 1e-15
        for k in (4, 8, 32):
            if k >= s:
                continue
            assert content_blind_rate(s, k) == content_blind_rate(s, 8)
            got = sampled_content_blind_rate(s, k, draws=40000, seed=1)
            reach = got * k
            tol = 4 * math.sqrt(max(reach * (1 - reach), 1e-12) / 40000) / k
            assert abs(got - 1 / s) < tol + 1e-12, (s, k, got, 1 / s)

    rng = np.random.default_rng(0)

    # FACT 2, closed form against direct evaluation of the four sets
    for _ in range(20):
        a = rng.random((5, 5))
        m = pairwise_M(a)
        for i, j in itertools.combinations(range(5), 2):
            rest = [x for x in range(5) if x not in (i, j)]
            for r in range(len(rest) + 1):
                for sub in itertools.combinations(rest, r):
                    d_f = second_difference(lambda p: captured_mass(a, p),
                                            sub, i, j)
                    d_h = second_difference(lambda p: reconstruction_gain(a, p),
                                            sub, i, j)
                    assert abs(d_f - 2 * m[i, j]) < 1e-8 * max(1.0, abs(d_f))
                    assert abs(d_h + 2 * m[i, j]) < 1e-8 * max(1.0, abs(d_h))
                    assert d_f >= -1e-9        # supermodular on nonnegative A
                    assert d_h <= 1e-9         # submodular on nonnegative A

    # monotonicity of h on a nonnegative operator, against the derived bound
    for _ in range(20):
        a = rng.random((5, 5))
        for i in range(5):
            rest = [x for x in range(5) if x != i]
            for r in range(len(rest) + 1):
                for sub in itertools.combinations(rest, r):
                    gain = (reconstruction_gain(a, set(sub) | {i})
                            - reconstruction_gain(a, sub))
                    ua = np.outer(a[:, i], a[i, :])
                    assert gain >= (ua * ua).sum() - 1e-9, (gain, sub, i)
                    assert gain > 0.0

    # h(empty) = 0 and f(empty) = 0, which the 1-1/e statement needs
    a = rng.random((5, 5))
    assert reconstruction_gain(a, ()) == 0.0
    assert captured_mass(a, ()) == 0.0

    # the signed arm loses it -- the PASS half of this control is non-degenerate
    lost = sum(1 for _ in range(200)
               if not submodular_objectives(
                   rng.normal(size=(6, 6)))["reconstruction_submodular"])
    assert lost > 100, f"signed operators kept submodularity in {200 - lost}/200"
    kept = sum(1 for _ in range(200)
               if submodular_objectives(
                   rng.random((6, 6)))["reconstruction_submodular"])
    assert kept == 200, f"nonnegative operators lost submodularity in {200 - kept}"

    assert abs(GREEDY_BOUND - 0.6321205588285577) < 1e-15


if __name__ == "__main__":
    demo()
    print(report())
    print("\nself-checks pass")
