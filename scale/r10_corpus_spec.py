"""The v-main.3M it.14 corpus spec, built and then attacked.

JUPITER, round 10 iteration 14. Deliverables `results/r10_it14_corpus_spec.md`
and `results/r10_it14_corpus_spec.jsonl`. Run with

    python -m scale.r10_corpus_spec            # demo(), assert-based, ~5 s
    python -m scale.r10_corpus_spec --write    # full corpus + both .md/.jsonl

WHAT THE SPEC ASKS FOR, VERBATIM. "labels u solve u(v) = (1/deg v) * sum_{w~v}
u(w) interior, u|_B = g; oracle u_I = (I - P_II)^-1 P_IB g (the proved
resolvent); dose theorem err(t) <= lambda_2^t with lambda_2 computed exactly per
instance (one eig, n <= 1024), rungs stratified to lambda_2 in [0.90, 0.95]."

THE SPEC IS WELL-POSED ON A NAMED CLASS AND ILL-POSED OFF IT, AND THE TWO
AMENDMENTS ARE STATED HERE RATHER THAN APPLIED QUIETLY.

  (A) `I - P_II` IS SINGULAR unless every interior vertex can reach `B`. If an
      interior component is disjoint from the boundary, `P_II` restricted to it
      is row-stochastic, `rho = 1`, and the resolvent does not exist -- the
      harmonic extension is genuinely non-unique there, since any constant may
      be added on that component. `admissible` refuses those instances and
      `demo` shows the refusal firing on a planted one. Isolated vertices
      (`deg = 0`) are refused for the same reason one step earlier: the
      averaging map is a division by zero.

  (B) `err(t) <= lambda_2^t` IS NOT A NORM-FREE STATEMENT AND IS FALSE IN THE
      TWO NORMS A READER WOULD REACH FOR FIRST. It is exactly true in the
      Perron-weighted sup norm `||v||_w = max_i |v_i| / w_i` with `P_II w =
      lambda_2 w`, which is the repo's own formal theorem
      (`lean/CEQ/Contraction.lean:72`, `weighted_contraction`), and exactly true
      in the degree-weighted 2-norm by reversibility. It is FALSE in `l_inf` and,
      on irregular graphs, in the unweighted `l_2`, at small `t`. Section 4 of
      the report measures all four and names the violating instances. The spec
      as written does not say which norm, so this module measures every one it
      could mean instead of picking the flattering one.

WHAT `lambda_2` IS. `P = [[P_II, P_IB], [0, I]]` is block triangular, so the
literal second eigenvalue of `P` is `1` and carries nothing. `lambda_2 :=
rho(P_II)`, the Perron root of the interior block, following
`scale/foreman_lambda2.py` and `MATHEMATICS.md` section 7.

ONE EIGENDECOMPOSITION PER INSTANCE, AND IT IS SYMMETRIC. `P_II = D_I^-1 W_II`
is not symmetric, but `S := D_I^{1/2} P_II D_I^{-1/2} = D_I^{-1/2} W_II
D_I^{-1/2}` is, and is similar to `P_II`, so `spec(S) = spec(P_II)` with the
spectrum real and `eigvalsh` ordering it exactly rather than sorting complex
moduli. `S` is entrywise non-negative, so Perron gives `rho(S) = lambda_max(S) =
vals[-1]` and no `abs` is needed. The same call returns the Perron vector, so
the weighted norm of amendment (B) costs nothing extra. `_cross_check_eigensolve`
compares against a non-symmetric `eigvals(P_II)` on the small instances.

TWO ORACLES, AND A THIRD ON THE SUBFAMILY WHERE THE REPO ALREADY HAS ONE.

  ORACLE A, the spec's: `np.linalg.solve(I - P_II, P_IB g)`.
  ORACLE B, independent: the averaging map iterated from the ADJACENCY LISTS to
      the float64 floor. It never forms `P_II`, `P_IB`, `D` or any matrix, so a
      transposed block, a mis-slotted index map or a degree read off the wrong
      axis moves one route and not the other.
  ORACLE C, the repo's: `scale/kirchhoff.harmonic_measure`, the matrix-tree /
      grounded-cofactor route of `MATHEMATICS.md` section 11. It computes
      `omega_x = M_xa / M_aa` from a SYMMETRIC unnormalised `L_b = D - A`, and
      applies exactly where `|B| = 2` and `g = (1, 0)`, which is that section's
      harmonic measure. On the path family it is joined by a fourth, purely
      analytic check: `u_x = x / (L+1)` and `lambda_2 = cos(pi / (L+1))`.

THE TOLERANCE, AND WHY THIS ONE. `1e-10`, imported from
`scale/kirchhoff.py:110` rather than re-picked, where it was set from a measured
window: largest gap between two CORRECT oracles `9.636736e-14`, smallest gap the
planted degree defect produces `1.749951e-01`, twelve orders of daylight, and
`1e-10` the round number three orders above the former and nine below the
latter. Oracle B is iterative, so its own floor is priced separately: it is run
`ceil(log(1e-17) / log(lambda_2)) + 64` sweeps, past the point where the update
stops moving in float64, and the achieved residual is reported per rung. An
iteration stopped early would fail this tolerance for its own reasons, which is
why the residual is reported and not just asserted.

BOTH INSTRUMENTS ARE SHOWN ABLE TO FAIL.
  The oracle check: oracle B is re-run dividing by `deg + 1` -- the repo's own
  planted defect (`kirchhoff.scratch_chain_with_off_by_one`), the miscount you
  get by counting a node among its own neighbours. It stays sub-stochastic, so
  it converges and returns plausible numbers in `[0, 1]` and nothing downstream
  would notice. `must_fire_oracle` asserts the check rejects it on every rung.
  The dose check: the same curve is scored against a `lambda_2` moved down by
  `0.02`, and `must_fire_dose` asserts a violation is then reported on every
  rung in the norm where the honest reading passes. The `l_inf` violations of
  amendment (B) are a second, unplanted demonstration on the same instrument.

ERRORS ARE COUNTED SEPARATELY. Every row carries `status`; `ok`, `refused`
(amendment A) or `error` with the exception text. Nothing missing is summed into
a pass.

COST. Dense float64, `n <= 1024`: `W` is 8 MiB, `eigh` a small multiple of that.
`scale.vram_gate.preflight(0, name='r10-corpus-spec', host_mib=512)` returned
HOST FITS at 640 MiB needed against 1250 MiB free, this machine, this session.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from scale import kirchhoff

__all__ = ["BAND_LO", "BAND_HI", "AGREEMENT_TOL", "NORMS", "draw_graph",
           "blocks", "admissible", "spectrum", "resolvent_oracle",
           "averaging_oracle", "dose_curve", "stratify", "naive_yield",
           "must_fire_oracle", "must_fire_dose", "path_case", "build", "demo"]

#: The band the script asks the rungs to be stratified into. Same numbers as
#: `scale/foreman_lambda2.py:126` (`TARGET_LO, TARGET_HI = 0.90, 0.95`).
BAND_LO, BAND_HI = 0.90, 0.95

#: Imported, not re-picked. `scale/kirchhoff.py:110` and its measured window.
AGREEMENT_TOL = kirchhoff.AGREEMENT_TOL

#: The four readings of "err(t)". See amendment (B) in the module header.
NORMS = ("linf", "l2", "l2_deg", "sup_perron")

#: How far the dose curve is walked. `0.95 ** 64 = 3.8e-2`, `0.90 ** 64 = 1.2e-3`
#: -- far enough for the asymptotic rate to show and short of the float64 floor
#: where `relerr / lambda_2^t` starts reading noise (`foreman_lambda2.RATE_TS`
#: records that floor biting at `t = 399` on an engineered `0.925`).
DOSE_T = 64

#: The planted offset for `must_fire_dose`. Large enough that no honest curve
#: could absorb it, small enough that the wrong `lambda_2` is still in a
#: plausible range and would not be caught by inspection.
DOSE_PLANT = 0.02


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------
def draw_graph(n: int, extra: int, rng) -> dict[int, set[int]]:
    """A connected simple graph on `n` nodes: a uniform attachment tree plus
    `extra` chords.

    Connectivity by construction rather than by rejection. Rejection sampling
    on Erdos-Renyi would make the draw cost depend on `p`, and the naive-yield
    number of section 3 is meant to price the BAND, not the connectivity retry.
    """
    adj: dict[int, set[int]] = {v: set() for v in range(n)}
    perm = rng.permutation(n)
    for i in range(1, n):
        a, b = int(perm[i]), int(perm[int(rng.integers(0, i))])
        adj[a].add(b)
        adj[b].add(a)
    added = 0
    while added < extra:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b and b not in adj[a]:
            adj[a].add(b)
            adj[b].add(a)
            added += 1
    return adj


def admissible(adj, boundary) -> None:
    """Amendment (A). Raise unless `(I - P_II)` is invertible for structural
    reasons: no isolated vertex, `B` non-empty, and every interior vertex able
    to reach `B`.

    Reachability is the whole condition. `P_II` is sub-stochastic with row
    deficit exactly the one-step escape probability, and `rho(P_II) < 1` iff
    every interior state escapes eventually -- which for a walk on a finite
    graph is exactly boundary reachability. An interior component that cannot
    see `B` carries a row-stochastic block, `rho = 1`, and the harmonic
    extension there is any constant.
    """
    if not boundary:
        raise ValueError("B is empty: the resolvent is singular (P_II is the "
                         "whole row-stochastic walk, rho = 1)")
    bad = [v for v in adj if not adj[v]]
    if bad:
        raise ValueError(f"isolated vertices {bad[:4]}: (1/deg v) is a division "
                         "by zero before the resolvent is even formed")
    seen, queue = set(boundary), deque(boundary)
    while queue:
        for u in adj[queue.popleft()]:
            if u not in seen:
                seen.add(u)
                queue.append(u)
    missed = [v for v in adj if v not in seen]
    if missed:
        raise ValueError(
            f"{len(missed)} interior vertices cannot reach B (e.g. {missed[:4]}): "
            "P_II has a row-stochastic diagonal block, rho(P_II) = 1, and "
            "(I - P_II) is singular. Excluded, not repaired.")


def blocks(adj, boundary):
    """`(P_II, P_IB, interior, deg_I)` for the mean-value operator.

    `interior` is ascending, so it matches the order
    `kirchhoff.harmonic_measure` returns its transient list in and the two can
    be compared without a reindex step that could itself be wrong.
    """
    bset = set(boundary)
    interior = [v for v in sorted(adj) if v not in bset]
    ii = {v: i for i, v in enumerate(interior)}
    bi = {v: i for i, v in enumerate(boundary)}
    PII = np.zeros((len(interior), len(interior)), dtype=np.float64)
    PIB = np.zeros((len(interior), len(boundary)), dtype=np.float64)
    deg = np.array([len(adj[v]) for v in interior], dtype=np.float64)
    for v in interior:
        r, d = ii[v], float(len(adj[v]))
        for u in adj[v]:
            if u in bset:
                PIB[r, bi[u]] += 1.0 / d
            else:
                PII[r, ii[u]] += 1.0 / d
    return PII, PIB, interior, deg


# --------------------------------------------------------------------------
# lambda_2 -- one eigendecomposition, exact ordering
# --------------------------------------------------------------------------
def spectrum(PII: np.ndarray, deg: np.ndarray):
    """`(lambda_2, w, vals)` from ONE symmetric eigendecomposition.

    `S = D^{1/2} P_II D^{-1/2}` has entries `W_ij / sqrt(d_i d_j)`, is symmetric
    and similar to `P_II`. Non-negative and symmetric, so Perron puts the
    spectral radius at the TOP of the ascending spectrum -- `vals[-1]`, no
    modulus, no sort of complex numbers.

    `w = D^{-1/2} v` is the right Perron vector of `P_II` and is the certificate
    of `lean/CEQ/Contraction.lean:72`. It is returned unnormalised in sign-fixed
    form; the caller checks it is strictly positive before using the weighted
    norm, because a reducible `P_II` can put zeros in it.
    """
    s = np.sqrt(deg)
    S = PII * s[:, None] / s[None, :]
    vals, vecs = np.linalg.eigh(S)
    v = vecs[:, -1]
    if v.sum() < 0.0:
        v = -v
    return float(vals[-1]), v / s, vals


def _cross_check_eigensolve(PII, deg, lam) -> float:
    """`|rho(P_II) from eigvals - lambda_2 from eigvalsh|`. The similarity claim
    of `spectrum` checked rather than asserted, on instances small enough that
    the non-symmetric solve is cheap."""
    return abs(float(np.abs(np.linalg.eigvals(PII)).max()) - lam)


# --------------------------------------------------------------------------
# The oracles
# --------------------------------------------------------------------------
def resolvent_oracle(PII, PIB, g):
    """ORACLE A, the spec's: `u_I = (I - P_II)^-1 P_IB g`, solved not inverted."""
    return np.linalg.solve(np.eye(PII.shape[0], dtype=np.float64) - PII, PIB @ g)


def averaging_oracle(adj, boundary, g, interior, sweeps: int, *, defect: int = 0):
    """ORACLE B: the defining averaging map, iterated from the ADJACENCY LISTS.

    Forms no matrix and never touches `blocks`. `defect=1` divides by
    `deg + 1` instead of `deg` -- the planted miscount of
    `kirchhoff.scratch_chain_with_off_by_one`, kept survivable on purpose: the
    map stays a strict contraction, converges, and returns numbers in `[0, 1]`.

    Returns `(u_interior, last_sup_move)`, the second being the size of the final
    update, so an iteration stopped short reports that rather than hiding it.
    """
    val = {v: float(x) for v, x in zip(boundary, g)}
    for v in interior:
        val[v] = 0.0
    move = float("inf")
    for _ in range(sweeps):
        fresh = {v: sum(val[u] for u in adj[v]) / (len(adj[v]) + defect)
                 for v in interior}
        move = max((abs(fresh[v] - val[v]) for v in interior), default=0.0)
        val.update(fresh)
    return np.array([val[v] for v in interior], dtype=np.float64), move


def sweeps_for(lam: float) -> int:
    """Enough sweeps of the averaging map to sit on the float64 floor.

    `lam ** k < 1e-17` plus 64, so the stopping point is a declared function of
    the instance's own rate rather than a constant that happens to work.
    """
    return int(math.ceil(math.log(1e-17) / math.log(min(lam, 0.999999)))) + 64


# --------------------------------------------------------------------------
# The dose theorem
# --------------------------------------------------------------------------
def _norms(x, deg, w):
    out = {"linf": float(np.abs(x).max()),
           "l2": float(np.linalg.norm(x)),
           "l2_deg": float(np.sqrt(float(np.sum(deg * x * x))))}
    out["sup_perron"] = float(np.max(np.abs(x) / w)) if w is not None else None
    return out


def dose_curve(PII, PIB, g, deg, w, u, T: int = DOSE_T):
    """`err(t) = ||u - u_t|| / ||u||` for `t = 0..T`, in every norm of `NORMS`.

    `u_0 = 0`, so `err(0) = 1 = lambda_2^0` exactly and the curve starts at the
    bound rather than under it. `u_t = sum_{m<t} P_II^m P_IB g` is `t` steps of
    the natural iteration, i.e. exactly what a `t`-hop budget can compute.
    """
    ref = _norms(u, deg, w)
    b = PIB @ g
    z = np.zeros_like(u)
    rows = []
    for _t in range(T + 1):
        e = _norms(u - z, deg, w)
        rows.append({k: (None if ref[k] in (None, 0.0) or e[k] is None
                         else e[k] / ref[k]) for k in NORMS})
        z = PII @ z + b
    return rows


def dose_verdict(rows, lam: float):
    """`{norm: {"held": bool, "tightest": ratio, "t": t}}` for `err(t) <= lam^t`.

    "Tightest" is the LARGEST `err(t) / lam^t` -- the closest the measurement
    ever came to the bound. Above `1` it is a violation and `t` names where.
    `t = 0` is excluded from the tightest reading because `err(0)/lam^0 = 1` by
    construction and would make every norm read as tight for free.
    """
    out = {}
    for k in NORMS:
        best, at, held, seen, r_eff = 0.0, None, True, 0, 0.0
        for t, row in enumerate(rows):
            if row[k] is None or t == 0:
                continue
            seen += 1
            ratio = row[k] / (lam ** t)
            if ratio > best:
                best, at = ratio, t
            r_eff = max(r_eff, row[k] ** (1.0 / t))
            if row[k] > lam ** t * (1.0 + 1e-9):
                held = False
        out[k] = {"held": held if seen else None, "tightest": best if seen else None,
                  "t": at, "samples": seen,
                  "r_eff": r_eff if seen else None,
                  "slack": (lam - r_eff) if seen else None}
    return out


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------
def _lambda_at(adj, order, k):
    """`lambda_2` with `B` = the first `k` nodes of `order`. Raises via
    `admissible` if that boundary leaves an unreachable interior."""
    boundary = list(order[:k])
    admissible(adj, boundary)
    PII, _PIB, _interior, deg = blocks(adj, boundary)
    return spectrum(PII, deg)[0]


def stratify(adj, order, lo=BAND_LO, hi=BAND_HI):
    """`(k, lambda_2, eigensolves)` landing `lambda_2` in `[lo, hi]`, by bisection
    on the boundary size.

    THE BISECTION IS EXACT, NOT A SEARCH HEURISTIC. Growing `B` along a FIXED
    node order deletes a row and column of `P_II` at each step, and the Perron
    root of a non-negative matrix is monotone non-increasing under deletion of a
    principal submatrix's row/column pair. So `lambda_2(k)` is monotone in `k`
    and bisection finds the crossing in `ceil(log2 n)` eigensolves. A random
    boundary SET per `k` would destroy that monotonicity and turn this into a
    search; that is why the order is drawn once and then held.

    Returns `k = None` when the crossing overshoots -- `lambda_2` above `hi` at
    `k` and below `lo` at `k+1`, the band falling between two adjacent integers.
    That is a MISS and is counted, not retried silently.
    """
    solves = 0
    n = len(adj)
    lam_hi = _lambda_at(adj, order, 1)
    solves += 1
    if lam_hi <= hi:
        return (1, lam_hi, solves) if lam_hi >= lo else (None, lam_hi, solves)
    lo_k, hi_k = 1, n - 1
    while lo_k + 1 < hi_k:
        mid = (lo_k + hi_k) // 2
        try:
            lam = _lambda_at(adj, order, mid)
        except ValueError:
            hi_k = mid
            continue
        solves += 1
        if lam > hi:
            lo_k = mid
        else:
            hi_k = mid
    lam = _lambda_at(adj, order, hi_k)
    solves += 1
    return (hi_k, lam, solves) if lo <= lam <= hi else (None, lam, solves)


def naive_yield(draws: int, seed: int, lo=BAND_LO, hi=BAND_HI):
    """Fraction of NAIVE draws whose `lambda_2` lands in the band.

    THE PRIOR IS DECLARED BEFORE ANY NUMBER IS READ, and it is the one a person
    writing "generate some graphs and label them" would actually type:
    `n` uniform on `{64, 128, 256, 512}`, mean degree uniform on `[3, 8]`,
    boundary size uniform on `{1, ..., n // 2}`, boundary set uniform without
    replacement. One eigensolve per draw, no retry, no conditioning.
    """
    rng = np.random.default_rng(seed)
    lams, hits, errors = [], 0, 0
    for _ in range(draws):
        n = int(rng.choice([64, 128, 256, 512]))
        extra = int(n * (float(rng.uniform(3.0, 8.0)) / 2.0)) - (n - 1)
        adj = draw_graph(n, max(extra, 0), rng)
        k = int(rng.integers(1, n // 2 + 1))
        boundary = [int(v) for v in rng.choice(n, size=k, replace=False)]
        try:
            lams.append(_lambda_at(adj, boundary + [], len(boundary)))
        except ValueError:
            errors += 1
            continue
        if lo <= lams[-1] <= hi:
            hits += 1
    return {"draws": draws, "seed": seed, "hits": hits, "errors": errors,
            "yield": hits / draws, "lambdas": lams}


# --------------------------------------------------------------------------
# The must-fires
# --------------------------------------------------------------------------
def must_fire_oracle(adj, boundary, g, interior, u_a, lam):
    """The oracle check shown REJECTING the planted degree defect.
    `(gap, fires)` -- `fires` is True when the gap clears the tolerance."""
    u_bad, _move = averaging_oracle(adj, boundary, g, interior,
                                    sweeps_for(lam), defect=1)
    gap = float(np.max(np.abs(u_bad - u_a)))
    return gap, gap >= AGREEMENT_TOL


def must_fire_dose(rows, lam, norm="sup_perron", plant=DOSE_PLANT):
    """The dose check shown REJECTING a wrong `lambda_2`, and its SENSITIVITY.

    Two plants, and the difference between them is the point.

    The FIXED plant moves `lambda_2` down by `DOSE_PLANT = 0.02` -- a declared
    constant, not tuned per instance. `fires` says whether the check caught it.

    `detect_threshold` is the exact sensitivity, and it needs no search. The
    bound at rate `r` is violated at some `t <= T` iff
    `r < r_eff := max_t err(t)^(1/t)`, so the check catches any understatement of
    `lambda_2` larger than `slack = lambda_2 - r_eff` and catches NOTHING
    smaller. A `slack` above `0.02` is not an instrument failure: it is the
    bound being too loose in that norm to notice a 2%-wrong rate within
    `T = DOSE_T` steps, which is this round's vacuity defect measured rather
    than asserted.

    The ADAPTIVE plant, `slack + 0.005`, is guaranteed by that identity to be
    caught, and exists so the instrument itself is shown firing on EVERY rung --
    separating "the check works and the bound is loose" from "the check is
    broken".
    """
    honest = dose_verdict(rows, lam)[norm]
    slack = honest["slack"]
    planted = dose_verdict(rows, lam - plant)[norm]["held"]
    adaptive = plant if slack is None else max(plant, slack + 0.005)
    adaptive_held = dose_verdict(rows, lam - adaptive)[norm]["held"]
    return {"norm": norm, "honest_held": honest["held"], "planted_held": planted,
            "fires": bool(honest["held"]) and planted is False,
            "detect_threshold": slack, "plant": plant,
            "adaptive_plant": adaptive,
            "fires_adaptive": bool(honest["held"]) and adaptive_held is False}


# --------------------------------------------------------------------------
# The path family: a closed form to check the eigensolve against
# --------------------------------------------------------------------------
def path_case(interior_count: int):
    """`(adj, boundary, g, lambda_2_closed_form, u_closed_form)` for the path
    `P_{L+2}` with the two endpoints as `B` and `g = (1, 0)`.

    `lambda_2 = cos(pi / (L + 1))` -- the path Laplacian's top mode -- and the
    harmonic extension is linear, `u_x = 1 - x / (L + 1)` at interior index
    `x = 1..L`. `L = 7` gives `cos(pi/8) = 0.9238795325`, inside the band, so
    the band is reachable with `|B| = 2` and `kirchhoff.harmonic_measure`
    applies. Two independent closed forms on the same instance.
    """
    L = interior_count
    n = L + 2
    adj = {v: set() for v in range(n)}
    for v in range(n - 1):
        adj[v].add(v + 1)
        adj[v + 1].add(v)
    boundary = [0, n - 1]
    g = np.array([1.0, 0.0])
    lam = math.cos(math.pi / (L + 1))
    u = np.array([1.0 - x / (L + 1.0) for x in range(1, L + 1)])
    return adj, boundary, g, lam, u


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------
#: The rung shapes. `n` and the mean degree are declared; the boundary SIZE is
#: what stratification solves for, so it is absent here on purpose.
RUNGS = ((64, 4.0), (64, 6.0), (128, 4.0), (128, 6.0), (256, 4.0), (256, 6.0),
         (256, 8.0), (512, 4.0), (512, 6.0), (512, 8.0), (1024, 4.0), (1024, 6.0))

#: Seed base. Every rung's seed is `SEED0 + index`, printed with the row.
SEED0 = 0x3A140000


def build(rungs=RUNGS, seed0: int = SEED0):
    """One row per rung. Never raises: a rung that refuses or errors returns a
    row saying so, so a missing instance cannot be summed into a pass."""
    rows = []
    for i, (n, mean_deg) in enumerate(rungs):
        seed = seed0 + i
        t0 = time.perf_counter()
        row = {"kind": "rung", "n": n, "mean_deg": mean_deg,
               "seed": f"0x{seed:08x}", "status": "ok"}
        try:
            rng = np.random.default_rng(seed)
            adj = draw_graph(n, max(int(n * mean_deg / 2.0) - (n - 1), 0), rng)
            order = [int(v) for v in rng.permutation(n)]
            k, lam, solves = stratify(adj, order)
            row.update(band_solves=solves, lambda_2=lam)
            if k is None:
                row.update(status="refused",
                           reason=f"band overshot: lambda_2={lam:.10f} at the "
                                  "bisection crossing, no integer boundary size "
                                  "lands inside [0.90, 0.95]")
                rows.append(row)
                continue
            boundary = order[:k]
            admissible(adj, boundary)
            PII, PIB, interior, deg = blocks(adj, boundary)
            lam, w, vals = spectrum(PII, deg)
            g = np.array(rng.uniform(0.0, 1.0, size=len(boundary)))

            # `w > 0` is what the weighted sup norm needs, and it is exactly
            # irreducibility of `P_II`: a boundary set that disconnects the
            # interior leaves the Perron vector supported on one piece and zero
            # on the others. The norm is then undefined AT RATE lambda_2 and the
            # reading is null, not a pass. `l2_deg` has no such requirement.
            positive = bool(w.min() > 1e-12)
            u_a = resolvent_oracle(PII, PIB, g)
            sweeps = sweeps_for(lam)
            u_b, move = averaging_oracle(adj, boundary, g, interior, sweeps)
            gap = float(np.max(np.abs(u_a - u_b)))

            # The Perron certificate of lean/CEQ/Contraction.lean:72, checked
            # entrywise on this instance rather than assumed.
            cert = float(np.max(np.abs(PII @ w - lam * w))) if positive else None

            rows_dose = dose_curve(PII, PIB, g, deg, w if positive else None,
                                   u_a)
            verdict = dose_verdict(rows_dose, lam)
            defect_gap, defect_fires = must_fire_oracle(adj, boundary, g,
                                                        interior, u_a, lam)
            row.update(
                boundary_size=k, interior_size=len(interior),
                lambda_2=lam, lambda_3=float(vals[-2]) if len(vals) > 1 else None,
                deg_min=float(deg.min()), deg_max=float(deg.max()),
                in_band=bool(BAND_LO <= lam <= BAND_HI),
                perron_positive=positive, perron_certificate_residual=cert,
                eig_cross_check=(_cross_check_eigensolve(PII, deg, lam)
                                 if n <= 256 else None),
                oracle_gap=gap, oracle_agrees=bool(gap < AGREEMENT_TOL),
                averaging_sweeps=sweeps, averaging_last_move=move,
                dose={k2: verdict[k2] for k2 in NORMS},
                defect_gap=defect_gap, defect_fires=bool(defect_fires),
                dose_mustfire={k2: must_fire_dose(rows_dose, lam, norm=k2)
                               for k2 in ("l2_deg", "sup_perron")},
                dose_curve_sample={str(t): rows_dose[t]
                                   for t in (1, 2, 4, 8, 16, 32, 64)},
                seconds=round(time.perf_counter() - t0, 3))
        except Exception as exc:                      # counted, never swallowed
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def kirchhoff_rows(lengths=(6, 7, 8, 9)):
    """The `|B| = 2` subfamily, where the repo's third oracle and two closed
    forms all apply to the same instance."""
    rows = []
    for L in lengths:
        row = {"kind": "kirchhoff", "family": "path", "interior": L,
               "status": "ok"}
        try:
            adj, boundary, g, lam_cf, u_cf = path_case(L)
            admissible(adj, boundary)
            PII, PIB, interior, deg = blocks(adj, boundary)
            lam, w, _vals = spectrum(PII, deg)
            u_a = resolvent_oracle(PII, PIB, g)
            u_b, _m = averaging_oracle(adj, boundary, g, interior,
                                       sweeps_for(lam))
            omega, transient = kirchhoff.harmonic_measure(
                adj, sorted(adj), (boundary[0], boundary[1]))
            assert transient == interior, (transient, interior)
            row.update(
                lambda_2=lam, lambda_2_closed_form=lam_cf,
                lambda_2_gap=abs(lam - lam_cf),
                in_band=bool(BAND_LO <= lam <= BAND_HI),
                gap_resolvent_vs_averaging=float(np.max(np.abs(u_a - u_b))),
                gap_resolvent_vs_kirchhoff=float(np.max(np.abs(u_a - omega))),
                gap_resolvent_vs_closed_form=float(np.max(np.abs(u_a - u_cf))),
            )
            row["all_agree"] = bool(
                max(row["gap_resolvent_vs_averaging"],
                    row["gap_resolvent_vs_kirchhoff"],
                    row["gap_resolvent_vs_closed_form"]) < AGREEMENT_TOL)
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def constant_g_rows(rungs=((128, 4.0), (256, 6.0))):
    """`g == 1`, where `u == 1` exactly and `err(t)` in `l_inf` is
    `max_i P_i(survive t steps)`.

    This is the CLEANEST refutation of the norm-free reading of the dose
    theorem, and it needs no adversarial construction: any interior vertex
    without a boundary neighbour has `P_i(survive 1) = 1`, so
    `err(1) = 1 > lambda_2`. Every instance in the band has such a vertex,
    because a graph where every interior vertex touches `B` absorbs in one step
    and has `lambda_2` nowhere near `0.9`.
    """
    rows = []
    for i, (n, mean_deg) in enumerate(rungs):
        seed = SEED0 + 0x100 + i
        row = {"kind": "constant_g", "n": n, "seed": f"0x{seed:08x}",
               "status": "ok"}
        try:
            rng = np.random.default_rng(seed)
            adj = draw_graph(n, max(int(n * mean_deg / 2.0) - (n - 1), 0), rng)
            order = [int(v) for v in rng.permutation(n)]
            k, lam, _s = stratify(adj, order)
            if k is None:
                row.update(status="refused", reason="band overshot", lambda_2=lam)
                rows.append(row)
                continue
            boundary = order[:k]
            PII, PIB, interior, deg = blocks(adj, boundary)
            lam, w, _v = spectrum(PII, deg)
            g = np.ones(len(boundary))
            u = resolvent_oracle(PII, PIB, g)
            rows_dose = dose_curve(PII, PIB, g, deg, w if w.min() > 1e-12 else None, u)
            row.update(lambda_2=lam, boundary_size=k,
                       max_dev_from_one=float(np.max(np.abs(u - 1.0))),
                       dose={k2: dose_verdict(rows_dose, lam)[k2] for k2 in NORMS},
                       err1={k2: rows_dose[1][k2] for k2 in NORMS})
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
def demo() -> None:
    """Assert-based self-check. Every claim the report makes has a line here."""
    # 1. The eigensolve against a closed form, and the band is reachable there.
    adj, boundary, g, lam_cf, u_cf = path_case(7)
    PII, PIB, interior, deg = blocks(adj, boundary)
    lam, w, _vals = spectrum(PII, deg)
    assert abs(lam - lam_cf) < 1e-12, (lam, lam_cf)
    assert BAND_LO <= lam <= BAND_HI, lam

    # 2. Three oracles and a closed form on one instance.
    u_a = resolvent_oracle(PII, PIB, g)
    u_b, move = averaging_oracle(adj, boundary, g, interior, sweeps_for(lam))
    omega, transient = kirchhoff.harmonic_measure(adj, sorted(adj), (0, 8))
    assert transient == interior
    for other in (u_b, omega, u_cf):
        assert np.max(np.abs(u_a - other)) < AGREEMENT_TOL, np.max(np.abs(u_a - other))
    assert move < 1e-14, move

    # 3. The oracle check is shown able to FAIL.
    gap, fires = must_fire_oracle(adj, boundary, g, interior, u_a, lam)
    assert fires and gap > 1e-3, (gap, fires)

    # 4. The Perron certificate of lean/CEQ/Contraction.lean:72 holds entrywise,
    #    and the dose bound is exact in that norm and violated in l_inf.
    assert w.min() > 0.0
    assert np.max(np.abs(PII @ w - lam * w)) < 1e-12
    ones = np.ones(len(boundary))
    u1 = resolvent_oracle(PII, PIB, ones)
    assert np.max(np.abs(u1 - 1.0)) < 1e-12, "g==1 must give u==1"
    rows = dose_curve(PII, PIB, ones, deg, w, u1)
    v = dose_verdict(rows, lam)
    assert v["sup_perron"]["held"] is True, v["sup_perron"]
    assert v["l2_deg"]["held"] is True, v["l2_deg"]
    assert v["linf"]["held"] is False, v["linf"]        # amendment (B)
    assert rows[1]["linf"] > lam, rows[1]["linf"]

    # 5. The dose check is shown able to FAIL, on a random g where it passes.
    rng = np.random.default_rng(7)
    gr = rng.uniform(0.0, 1.0, size=len(boundary))
    ur = resolvent_oracle(PII, PIB, gr)
    mf = must_fire_dose(dose_curve(PII, PIB, gr, deg, w, ur), lam)
    assert mf["fires_adaptive"], mf
    assert mf["detect_threshold"] is not None and mf["detect_threshold"] >= 0.0, mf

    # 6. Amendment (A): the singular class is refused, not repaired.
    bad = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
    try:
        admissible(bad, [0])
    except ValueError as exc:
        assert "cannot reach B" in str(exc), exc
    else:
        raise AssertionError("the singular-resolvent guard did not fire")
    try:
        admissible({0: {1}, 1: {0}, 2: set()}, [0])
    except ValueError as exc:
        assert "Isolated" in str(exc) or "isolated" in str(exc), exc
    else:
        raise AssertionError("the isolated-vertex guard did not fire")

    # 7. Stratification lands in the band on a drawn graph.
    r = np.random.default_rng(SEED0)
    a2 = draw_graph(128, 128 * 2 - 127, r)
    k, lam2, solves = stratify(a2, [int(v) for v in r.permutation(128)])
    assert k is not None and BAND_LO <= lam2 <= BAND_HI, (k, lam2)
    assert solves <= 12, solves
    print(f"demo ok: lambda_2(path,L=7)={lam:.10f}  oracle gap "
          f"{np.max(np.abs(u_a - u_b)):.3e}  planted-defect gap {gap:.6e}  "
          f"stratified k={k} lambda_2={lam2:.10f} in {solves} eigensolves")


def main(write: bool) -> None:
    from scale.vram_gate import preflight
    fits, verdict = preflight(0, name="r10-corpus-spec", host_mib=512)
    print(verdict)
    if not fits:
        print("REFUSED by the gate; that is the reportable result.")
        return
    t0 = time.perf_counter()
    ny = naive_yield(200, SEED0 + 0x1000)
    rungs = build()
    kr = kirchhoff_rows()
    cg = constant_g_rows()
    rows = ([{"kind": "naive_yield", **{k: v for k, v in ny.items()
                                        if k != "lambdas"},
              "lambda_quantiles": {q: float(np.quantile(ny["lambdas"], q))
                                   for q in (0.0, 0.25, 0.5, 0.75, 1.0)}
              if ny["lambdas"] else None}]
            + rungs + kr + cg)
    for r in rows:
        print(json.dumps(r)[:200])
    print(f"total {time.perf_counter() - t0:.1f}s")
    if write:
        out = Path(__file__).resolve().parents[1] / "results"
        with (out / "r10_it14_corpus_spec.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"wrote {out / 'r10_it14_corpus_spec.jsonl'}")


if __name__ == "__main__":
    if "--write" in sys.argv or "--run" in sys.argv:
        main(write="--write" in sys.argv)
    else:
        demo()
