"""THE CEILING OF THE UNSEEN-GOAL COMMITTOR MEASURE, computed before any model exists.

WHY THIS FILE RUNS BEFORE A MODEL IS WRITTEN. Three attempts in this repository
died on the MEASUREMENT and two of them died only after training. The worst is
recorded at `ceqjepa/hf/README.md:34`, whose row is the producer for it and the
only place it is quoted from: DCM-1's bed-oracle headroom came back
+0.0935 WITH THE WRONG SIGN, meaning a brute-force oracle handed the true
interventional distribution did no better than chance and no model of any size
could have won on that bed -- discovered after the run. This module exists so a
fourth attempt cannot repeat it: every ceiling here is a linear-algebra fact
about the enumerated chain, and the oracle IS the label, so a wrong-sign
headroom is not expressible.

THE MEASURE UNDER TEST. Do not score a model against one fixed goal. Freeze the
weights, THEN draw a goal: an absorbing set B' and a forbidden set C from a
structured family of orbit subsets. The label is the exact committor
q_{B',C}(s) = P(the uniform-random-play chain hits B' before C or before any
terminal), a sparse linear solve on the orbit graph. The model sees the position
and the (B', C) indicator and answers in one forward pass. The TRAINING goals
are drawn first and a candidate test goal that a table over them already answers
is rejected before it is scored -- see THE SAMPLER FLOOR below, which is the
only thing about this family that is not simply "draw and keep".

THE REFUTATION THIS FILE RUNS FIRST, aimed by the proposer at his own idea.
Draw 100 (B', C) pairs. Solve the exact committors. Fit ONE regression, pooled
across all 100 draws, on [state features, king-to-nearest-B' Chebyshev distance,
the 1-hop and 2-hop truncated resolvents q1 and q2]. If that pooled fit reaches
R^2 >= 0.95 then every goal is one shallow function of the state, the measure is
not measuring composition, and the leap is dead. `KILL_R2 = 0.95` is that line
and it is checked before anything else is reported.

THE THREE CEILINGS, each exactly computable and none needing a trained model:

    MARGINAL          predict the per-draw mean; error is var(q_{B'}); R^2 = 0
                      by construction, and it is listed so the other two are
                      read against something.
    LOOKUP TABLE      a table holding committors for TRAINING goals only.
                      Ceiling per test draw = max over training goals of
                      corr(q_B, q_{B'})^2 -- the best a memoriser can do after
                      being handed a free per-draw affine recalibration it has
                      no way to compute. The sampler floor is applied to exactly
                      this quantity, so it is <= FLOOR on every scored draw.
    BEST HEURISTIC    any hand-written state heuristic is a function fixed
                      before (B', C) existed, so the best one is
                      h*(s) = E_{B'}[q_{B'}(s)]. It is estimated here ON THE
                      TEST DRAWS THEMSELVES, which is in-sample and therefore an
                      UPPER bound on every heuristic anyone will ever write.

Headroom = 1 - max(ceilings), with the oracle at 1.0 by construction. The
FAMILY MEAN of that is the number this module used to publish and it was the
wrong one: a bed is only as good as its worst draw. The publication is the
per-draw distribution, and `demo()` prints worst / p10 / p25 / median / p75 / p90.

THE D4 QUOTIENT, and why it is safe. The board's dihedral group has 8 elements,
all of which preserve legality and the move relation in a pawnless endgame, so
the uniform-random-play chain is STRONGLY LUMPABLE onto the orbits. That is
checked exhaustively here, not sampled: `lumpability_disagreements` compares the
lumped row of every one of the 368,452 positions against its orbit's row and
must be 0. Goal predicates are evaluated on each orbit's canonical
representative, which makes every drawn (B', C) a union of orbits by
construction.

RED FIRST. `tests/curvature/test_goal_family.py` was written and run before this
file existed. Verbatim, `python -m pytest tests/curvature/test_goal_family.py -x -q`
at cf9b0d2 on WIN-16QAL06O9GB:

    ImportError while importing test module '...tests\\curvature\\test_goal_family.py'.
    ...
    E   ImportError: cannot import name 'goal_family' from 'ceqjepa'
        (C:\\Users\\seal\\Desktop\\New folder (32)\\ceqjepa\\__init__.py)
    ===================== short test summary info =====================
    ERROR tests/curvature/test_goal_family.py
    !!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!
    1 error in 8.83s

THE SAMPLER FLOOR, AND THE TWO CANDIDATE FLOORS THAT WERE MEASURED AND KILLED.
The bed's one real defect was the worst draw, not the mean: the lookup ceiling
read median 0.1612, p90 0.5012 and MAXIMUM 0.9496, leaving 0.0504 of headroom at
the bottom of the family (`python -m ceqjepa.goal_family --draws 100 --train 20`
at 62cb8e0 on WIN-16QAL06O9GB, re-derived with FLOOR = 1.0). A floor in the
SAMPLER is the fix, and three candidates were measured before one was adopted.

  KILLED, symmetric difference. Reject a test draw whose |B' XOR B_t| / |B' OR B_t|
  against every training goal falls below a floor. MEASURED on 240 candidate
  draws against 20 training goals at seed 1
  (`scratchpad/floor_calib.py`, 62cb8e0, WIN-16QAL06O9GB): across all 4,800
  (test, train) PAIRS the Spearman correlation between that distance and the
  pair's corr^2 is -0.045, and on B OR C it is +0.021. The six draws that reach
  corr^2 > 0.8 sit at symmetric-difference 0.3975 to 0.6637 from the training
  goal they correlate with -- they are not near-duplicates. Set proximity is not
  the mechanism. To force the worst accepted draw below 0.5 this floor has to
  reject 91% of the family (worst 0.3222 at an acceptance rate of 0.092), which
  hollows the family out rather than cleaning it.

  KILLED, a held-out reference bank. Reject against 10 reference goals and
  measure against 10 DISJOINT training goals, so nothing is selected on the
  table it is scored against. MEASURED, same file, five random splits: at a
  reference threshold of 0.3 (acceptance 0.70 to 0.83) the worst ceiling against
  the disjoint table is still 0.5604 to 0.8736. The high-ceiling event is
  specific to the PAIR, so it does not transfer from one bank to another, and no
  predicate blind to the actual training goals can bound it.

  ADOPTED. Reject a test draw the TRAINING table already answers:
  max_t corr(q_t, q_test)^2 > FLOOR, with FLOOR = 0.50. That is a property of
  the label and of the published training goals, computable before a reader
  exists, and it is applied in `draw_test_goal` -- the draw is discarded, never
  scored. 0.50 is not taste: it is the pre-floor distribution's own p90
  (0.5012), so the floor is defined to make the WORST accepted draw exactly as
  clean as the 90th-percentile draw already was.

DECLARED BEFORE THE FLOORED RUN WAS EXECUTED, so that "comfortably positive" is
a prediction and not a description. Comfortably positive := worst-draw headroom
>= 0.50. Two non-vacuity conditions, because a floor that bounds the tail by
eating the family is worse than the defect: acceptance rate >= 0.50, and the
MEDIAN lookup ceiling must move less than 0.05 from its pre-floor 0.1612. All
three were met.

MEASURED. `python -m ceqjepa.goal_family --draws 100 --train 20` at commit
62cb8e0 on WIN-16QAL06O9GB (python 3.11.9, numpy 2.4.6, scipy 1.17.1), CPU only,
SEED = 0, FLOOR = 0.50, 204.3s, 3,371,479 scored rows, max committor residual
9.841e-15 (worst draw 909 fixpoint iterations), lumpability disagreements
0 of 368,452.

    the refutation, pooled LINEAR fit   R2 = 0.7106  [0.6669, 0.7451]
    the same fit, degree-2 expansion    R2 = 0.7874  [0.7433, 0.8222]
    the same fit, HELD OUT BY GOAL      R2 = 0.7264  (within-draw 0.5091)
    kill line                                0.95    -> NOT REACHED, leap ALIVE
    marginal / lookup / heuristic ceiling    0.0000 / 0.2336 / 0.0852
    the floor           121 candidates drawn, 21 rejected, acceptance 0.8264
    lookup ceiling over draws  median 0.1977  p90 0.4475  max 0.4886
    headroom, family mean                    0.7664

THE PER-DRAW HEADROOM DISTRIBUTION, which is the honest publication and replaces
the family mean. Same run.

    worst 0.5114   p10 0.5525   p25 0.6380   median 0.7860   p75 0.8640
    p90 0.9266     0 of 100 draws leave a faker less than 0.2 headroom

WHAT IS EARNED AND WHAT IS IMPOSED, since the floor selects on the ceiling it
then reports. IMPOSED: no draw can exceed a lookup ceiling of 0.50, so the
0.5114 worst-draw headroom cannot have come out below 0.50. EARNED, because
nothing selected on them: the worst draw lands at 0.4886, strictly inside the
floor rather than pressed against it; the acceptance rate 0.8264; the median
0.1977, which moved 0.0365 from the pre-floor 0.1612 -- the tail was removed and
the bulk was not; the heuristic ceiling, which the floor does not touch, worst
draw 0.3298; and the refutation itself, 0.7106 against the 0.95 kill line.

MOST OF THE LOOKUP CEILING WAS NEVER THE TABLE. The published ceiling hands the
memoriser a free per-draw affine recalibration it has no way to compute. The
same 20-entry table forced to answer with its STORED value reads 0.0048 on the
family mean and 0.2372 on the worst draw (same run, `ceiling_lookup_raw`). So
97.9% of the mean ceiling is the recalibration, not the memorisation. The
generous reading is kept, because a ceiling that is too high is the safe error.

THE LIMIT OF THE FLOOR, collected here and not repeated. The guarantee is
against a table over THESE 20 training goals. A memoriser holding a different or
larger table is not bounded by it, and the floor would have to be recomputed
against that table.

CAPACITY DOES NOT BUY THE KILL, and this is the load-bearing control. Every
number in this paragraph and the three that follow it was measured on the
PRE-FLOOR draw set at cf9b0d2 and is NOT re-derived by the floored run; they are
statements about the bed before the sampler changed, kept because what they
establish -- that the shortfall is not the estimator, not the linearity and not
the trivial rows -- is a property of the family rather than of one draw set. A
HistGradientBoostingRegressor at max_iter=1000, max_leaf_nodes=255,
min_samples_leaf=5, learning_rate=0.1, early_stopping=False, random_state=0,
fitted on the SAME 27 features, reaches pooled R2 = 0.9965 IN-SAMPLE on 800,000
rows (8,000 per draw). That is ~3 rows per leaf -- a 255,000-entry lookup table,
not "one shallow function of the state", so it does not instantiate the
refutation. Refitted on 80 drawn goals and scored on the 20 it never saw, on the
identical split, the same ensemble reads pooled 0.5567 and WITHIN-draw 0.0020:
against a new goal it is worth nothing over predicting that goal's mean. On the
same held-out split the linear arm reads 0.7230 and the degree-2 arm 0.7378 (at
8,000 rows per draw; the module's 0.6184 above is the all-rows version of the
same split, which stops down-weighting the draws with the largest interiors).
Every one of those is below 0.95. The shortfall is not the estimator.

TWO DEAD DIRECTIONS, named rather than carried, so that "27 features" is not
read as 27 directions. The 28-column linear design has RANK 26, re-derived by
the floored run, at `cond(X'X) = 6.627e+19` there and 1.412e+20 pre-floor, for
two exact reasons:

  - `in_check` is identically 0 on every scored row. Check occurs only on
    positions with no legal move (`chess_steps.py:354-356` sets it inside the
    `count == 0` branch and nowhere else); exactly 46 orbits carry it and all 46
    are absorbing, hence never scored.
  - `wk_r == e_wk` on all 46,137 orbits, 0 mismatches. Taking the
    lexicographically smallest D4 image puts the white king in the fundamental
    triangle -- measured canonical ranges are file 0..3, rank 0..3, with 0
    orbits where rank > file -- so min(f, 7-f) = f, min(r, 7-r) = r, and the
    edge distance collapses onto the rank.

It moves nothing. The module's ridged solve, a column-equilibrated solve, `pinv`
at rcond 1e-10 and 1e-14, and `lstsq` all return pooled R2 = 0.736652 and within
R2 = 0.394159 -- six figures, five estimators -- and a zero-ridge SVD solve on
the raw 3,296,692 x 28 design, forming no normal equations at all, returns
0.7366522416 against the module's 0.7366522410. R2 depends on the column space,
and a redundant column adds none to it.

THE EASY ROWS ARE NOT CARRYING THE HEADLINE. 159,096 scored rows (4.83%) carry
q exactly 0 and 13,739 (0.42%) carry exactly 1. Dropping all of them and
refitting on the strictly interior 3,123,817 rows gives pooled R2 = 0.723675 --
LOWER than 0.736652, i.e. further from the kill line. The trivial rows were
inflating the refutation, not the survival.

Brackets are 95% percentile intervals from 2,000 bootstrap resamples OF THE 100
DRAWS, coefficients pinned (`_boot_pooled`).

HEAD MOVED DURING THE RUN and the numbers are unaffected, which is asserted
rather than assumed. Six commits from concurrent sessions advanced cf9b0d2 to
b20097c while this ran. The only repo input to every figure above is
`ceqjepa.chess_steps.space/oracle`, and
`git diff --name-only cf9b0d2..HEAD -- ceqjepa/chess_steps.py` returns 0 files.
The suite re-derives every figure in the working tree at b20097c and reads
`10 passed in 294.74s`, exit 0.

THE NUMBER THAT MUST BE READ BESIDE 0.7106. Predicting each draw's own MEAN,
with no state skill whatsoever, already scores pooled R2 = 0.5277 -- because the
100 draws have very different mean committors and a pooled denominator counts
that between-draw spread as explainable variance. Against the WITHIN-draw
variance alone the same fits read 0.3874 (linear) and 0.5499 (degree-2), and the
mean PER-DRAW R2 of the pooled linear fit is -0.5065 +/- 1.1964: on a typical
drawn goal the shallow fit is WORSE than predicting that goal's own mean. The
refutation does not merely fall short of 0.95, it fails to beat the marginal.

WHERE THE EXPLANATORY POWER IS, by column ablation (same draws, same labels,
same estimator, only the admissible columns vary):

    columns                pooled   within
    full (28)              0.7106   0.3874
    state only (20)        0.0256  -1.0630   goal-blind features carry nothing
    no resolvents (26)     0.4479  -0.1689
    resolvents only (3)    0.5276  -0.0001
    goal features (9)      0.6956   0.3556   q1, q2 and the distances are it

THE GAP AGAINST A d-LAYER STACK, free from the enumeration. A d-layer stack
composes d hops, and where B' is further than d hops the d-hop truncation q_d is
EXACTLY zero while q is not. Fraction of the 3,296,692 scored rows in that
state:

    d = 1   0.7736        d = 4   0.1331
    d = 2   0.4596        d = 8   0.0009

The same fractions by BFS on the move graph, a second route to the same
quantity, read 0.7956 / 0.4817 / 0.1551 / 0.0230 (BFS counts reachability, the
truncation additionally requires q > 0, so BFS is the looser of the two);
orbit-size weighting moves them by at most 0.0002. Pre-floor the same two routes
read 0.7547 / 0.4795 / 0.1733 / 0.0049 and 0.8030 / 0.5278 / 0.2216 / 0.0532.
`ceqjepa/depth_race.py` is what these fractions are a bound FOR, and it carries
the assumption they rest on.

THE PLACE THIS MEASURE WAS NOT SAFE, AND THE STATE OF IT NOW. Pre-floor the
lookup ceiling was 0.2400 on the family mean but 0.9496 on the worst of the 100
draws, leaving 0.0504 of headroom at the bottom of the family. With the floor in
the sampler the worst draw reads 0.4886 and the worst headroom 0.5114, and the
publication is the distribution above rather than the mean. The bed is ready in
the sense that was declared: worst-draw headroom >= 0.50, acceptance 0.8264,
median ceiling moved 0.0365.

WHAT THIS MEASURES AND WHAT IT DOES NOT. This is operator COMPOSITION toward
arbitrary goals. It is NOT do-versus-see. In a fully observed MDP the two
coincide, which is the diagnosis of DCM-1's flat move-permutation ablation
rather than a bug in it: on a fully observed bed there is no interventional-
observational divergence to measure, because conditioning on a move and forcing
it are the same operation. A bed for the do/see definition needs hidden state.
This bed can support the composition definition and cannot support the
divergence one.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ceqjepa.chess_steps import oracle, space

#: The kill line the proposer set against his own idea. PINNED.
KILL_R2 = 0.95

#: Every draw, every shuffle. PINNED.
SEED = 0

#: The headline run.
N_TEST = 100
N_TRAIN = 20

#: THE SAMPLER FLOOR. A test draw is rejected if a lookup table over the
#: TRAINING goals already answers it: reject when
#: max_t corr(q_t, q_test)^2 > FLOOR. PINNED, and chosen from the measured
#: unfiltered distribution rather than by taste -- 0.50 is that distribution's
#: own p90 (0.5012, `python -m ceqjepa.goal_family --draws 100 --train 20` at
#: 62cb8e0 on WIN-16QAL06O9GB with FLOOR = 1.0), so the floor makes the WORST
#: accepted draw exactly as clean as the 90th-percentile draw already was. See
#: THE FLOOR, AND THE TWO CANDIDATES THAT WERE MEASURED AND KILLED, above.
FLOOR = 0.50

#: A draw whose committors cannot be sampled inside this many tries means the
#: floor has eaten the family; that is a raise, never a silent truncation.
MAX_DRAW_TRIES = 60

#: A drawn B' must be neither a needle nor half the board, or the committor is
#: trivial and the draw measures nothing. Orbit counts, out of 46,137.
B_MIN, B_MAX = 200, 9000
C_MIN, C_MAX = 200, 18000
INTERIOR_MIN = 15000

#: The committor fixpoint: tolerance on the max update, and the hard cap that
#: turns non-convergence into a raise instead of a silent truncation. PINNED.
Q_TOL = 1e-14
MAX_ITERS = 200000

#: Ridge, relative to mean(diag(X'X)). Present only so the degree-2 normal
#: equations are solvable; it is 1e-8 and moves no reported digit.
RIDGE = 1e-8

#: `python -m ceqjepa.goal_family --draws 100 --train 20`, 62cb8e0,
#: WIN-16QAL06O9GB, FLOOR = 0.50. Re-derived, not copied, by
#: tests/curvature/test_goal_family.py::test_the_pinned_hundred_draw_run_reproduces.
RESULTS: dict = {
    "n_orbits": 46137, "n_rows": 3371479,
    "r2_linear": 0.7106, "r2_quadratic": 0.7874,
    "r2_linear_within": 0.3874, "r2_quadratic_within": 0.5499,
    "r2_linear_heldout": 0.7264, "r2_linear_heldout_within": 0.5091,
    "design_rank": 26,
    "marginal_pooled": 0.5277,
    "ceiling_lookup": 0.2336, "ceiling_heuristic": 0.0852,
    "ceiling_lookup_raw": 0.0048, "ceiling_lookup_raw_max": 0.2372,
    "headroom": 0.7664, "headroom_worst_draw": 0.5114,
    "acceptance_rate": 0.8264, "n_candidate_draws": 121,
    "headroom_p0": 0.5114, "headroom_p10": 0.5525, "headroom_p25": 0.6380,
    "headroom_p50": 0.7860, "headroom_p75": 0.8640, "headroom_p90": 0.9266,
    "draws_with_headroom_below_0p2": 0,
    "ceiling_lookup_median": 0.1977, "ceiling_lookup_p90": 0.4475,
    "ceiling_lookup_max": 0.4886, "ceiling_heuristic_max": 0.3298,
    "dead_mass": {1: 0.7736, 2: 0.4596, 4: 0.1331, 8: 0.0009},
    "hop_uniform": {1: 0.7956, 2: 0.4817, 4: 0.1551, 8: 0.0230},
}


# ---------------------------------------------------------------------------
# THE QUOTIENT
# ---------------------------------------------------------------------------

def d4_tables() -> np.ndarray:
    """The 8 square permutations of the dihedral group of the board."""
    f = np.arange(64) % 8
    r = np.arange(64) // 8
    out = []
    for transpose in (False, True):
        for mf in (False, True):
            for mr in (False, True):
                ff, rr = (r, f) if transpose else (f, r)
                ff = 7 - ff if mf else ff
                rr = 7 - rr if mr else rr
                out.append((rr * 8 + ff).astype(np.int64))
    return np.stack(out)


@lru_cache(maxsize=1)
def quotient() -> dict:
    """The D4 quotient of the enumerated KQvK chain, with the lumping checked.

    The check is exhaustive, not sampled: the lumped row of every position is
    compared against its orbit's row.
    """
    sp = space()
    orc = oracle()
    n, sink, nt = sp["n_positions"], sp["sink"], sp["n_states"]
    keys = sp["keys"].astype(np.int64)

    code = (keys[:, 0] * 64 + keys[:, 1]) * 128 + keys[:, 2] * 2 + keys[:, 3]
    order = np.argsort(code)
    code_sorted = code[order]

    G = d4_tables()
    img = np.empty((8, n), np.int64)
    for a, g in enumerate(G):
        c = (g[keys[:, 0]] * 64 + g[keys[:, 1]]) * 128 + g[keys[:, 2]] * 2 + keys[:, 3]
        pos = np.searchsorted(code_sorted, c)
        hit = (pos < n) & (code_sorted[np.minimum(pos, n - 1)] == c)
        if not hit.all():
            raise AssertionError(f"D4 element {a} leaves the space at "
                                 f"{int((~hit).sum())} positions")
        img[a] = order[pos]

    reps, orbit_of = np.unique(img.min(axis=0), return_inverse=True)
    n_orb = int(reps.size)

    orb_all = np.empty(nt, np.int64)
    orb_all[:n] = orbit_of
    orb_all[sink] = n_orb                       # K vs K, its own absorbing orbit
    N = n_orb + 1

    deg = np.diff(sp["succ_indptr"])
    src = sp["pred"].astype(np.int64)
    dst = sp["succ"].astype(np.int64)
    w = 1.0 / deg[src]
    P_mem = sparse.csr_matrix((w, (src, orb_all[dst])), shape=(nt, N))
    P = sparse.csr_matrix((w, (orb_all[src], orb_all[dst])), shape=(N, N))
    osize = np.bincount(orb_all, minlength=N).astype(np.float64)
    P = sparse.csr_matrix(sparse.diags(1.0 / osize) @ P)

    gap = P_mem[:n] - P[orbit_of]
    disagreements = int((np.abs(gap.data) > 1e-12).sum())

    absorbing = np.zeros(N, bool)
    absorbing[orb_all[:n][np.asarray(sp["no_move"])]] = True   # mate + stalemate
    absorbing[n_orb] = True

    return dict(
        n_positions=n, n_moves=int(sp["n_edges"]), n_orbits=n_orb, n_states=N,
        P=P, PT=sparse.csr_matrix(P.T),
        reps=reps, orbit_of=orbit_of, rep_keys=keys[reps], osize=osize[:n_orb + 1],
        absorbing=absorbing, lumpability_disagreements=disagreements,
        dtm_full=orc["dtm"], dtm=orc["dtm"][reps], q_mate=orc["q"][reps],
        deg=deg[reps].astype(np.float64),
        in_check=np.asarray(sp["in_check"])[reps].astype(np.float64),
    )


# ---------------------------------------------------------------------------
# THE GOAL FAMILY
# ---------------------------------------------------------------------------

_CHEB = np.maximum(
    np.abs((np.arange(64) % 8)[:, None] - (np.arange(64) % 8)[None, :]),
    np.abs((np.arange(64) // 8)[:, None] - (np.arange(64) // 8)[None, :]),
)
_RIM = np.flatnonzero(
    (np.arange(64) % 8 == 0) | (np.arange(64) % 8 == 7)
    | (np.arange(64) // 8 == 0) | (np.arange(64) // 8 == 7))


@lru_cache(maxsize=1)
def _derived() -> dict:
    q = quotient()
    wk, wq, bk, turn = (q["rep_keys"][:, i] for i in range(4))
    ff = lambda s: s % 8
    rr = lambda s: s // 8
    edge = lambda s: np.minimum(np.minimum(ff(s), 7 - ff(s)),
                                np.minimum(rr(s), 7 - rr(s)))
    return dict(
        wk=wk, wq=wq, bk=bk, turn=turn.astype(np.float64),
        wk_f=ff(wk), wk_r=rr(wk), wq_f=ff(wq), wq_r=rr(wq),
        bk_f=ff(bk), bk_r=rr(bk),
        d_wk_bk=_CHEB[wk, bk], d_wq_bk=_CHEB[wq, bk], d_wk_wq=_CHEB[wk, wq],
        e_wk=edge(wk), e_wq=edge(wq), e_bk=edge(bk),
        corner_bk=_CHEB[bk][:, [0, 7, 56, 63]].min(axis=1),
    )


def _atom(rng: np.random.Generator) -> np.ndarray:
    """One predicate over orbits, evaluated on the canonical representative."""
    D = _derived()
    kind = rng.integers(5)
    if kind == 0:                                   # rim squares for the bare king
        m = int(rng.integers(4, 13))
        chosen = rng.choice(_RIM, size=m, replace=False)
        mask = np.isin(D["bk"], chosen)
    elif kind == 1:                                 # queen on a file, or a rank
        v = int(rng.integers(8))
        mask = (D["wq_f"] == v) if rng.integers(2) else (D["wq_r"] == v)
    elif kind == 2:                                 # king-distance band
        mask = D["d_wk_bk"] == int(rng.integers(2, 8))
    elif kind == 3:                                 # queen-to-king distance band
        mask = D["d_wq_bk"] == int(rng.integers(1, 8))
    else:                                           # a box for the strong king
        side = int(rng.integers(3, 5))
        f0, r0 = int(rng.integers(0, 9 - side)), int(rng.integers(0, 9 - side))
        mask = ((D["wk_f"] >= f0) & (D["wk_f"] < f0 + side)
                & (D["wk_r"] >= r0) & (D["wk_r"] < r0 + side))
    if rng.random() < 0.3:
        mask = mask & (D["turn"] == float(rng.integers(2)))
    return mask


def draw_goal(rng: np.random.Generator, absorbing: np.ndarray):
    """(B', C): an absorbing-at-one set and 1..3 forbidden regions at once.

    Redraws until the pair is non-degenerate. The acceptance bounds are the
    pinned B_MIN/B_MAX/C_MAX/INTERIOR_MIN and nothing else is filtered -- in
    particular no draw is rejected for being hard or easy.
    """
    N = absorbing.size
    pad = lambda m: np.concatenate([m, np.zeros(N - m.size, bool)])
    for _ in range(500):
        b = _atom(rng)
        if rng.random() < 0.4:
            b = b & _atom(rng)
        B = pad(b)
        C = np.zeros(N, bool)
        for _ in range(int(rng.integers(1, 4))):
            C |= pad(_atom(rng))
        C &= ~B
        nb, nc = int(B.sum()), int(C.sum())
        interior = ~(B | C | absorbing)
        if (B_MIN <= nb <= B_MAX and C_MIN <= nc <= C_MAX
                and int(interior.sum()) >= INTERIOR_MIN):
            return B, C, interior
    raise AssertionError("the goal family produced no admissible draw in 500 tries")


def lookup_ceiling(train, interior, y):
    """The best a table of TRAINING committors can do on this draw.

    Two readings of "best", both returned. `recal` hands the memoriser a free
    per-draw affine recalibration it has no way to compute (corr^2, the
    generous one, and the one the bed publishes); `raw` makes it answer with
    the stored value (1 - SSE/SS, floored at 0). The gap between them is how
    much of the ceiling is the recalibration rather than the table.
    """
    ss = float(((y - y.mean()) ** 2).sum())
    recal = raw = 0.0
    for qb in train:
        h = qb[interior]
        c = np.corrcoef(h, y)[0, 1]
        recal = max(recal, 0.0 if not np.isfinite(c) else float(c ** 2))
        raw = max(raw, 1.0 - float(((y - h) ** 2).sum()) / ss)
    return recal, max(raw, 0.0)


def draw_test_goal(rng, absorbing, P, train_q, floor=FLOOR):
    """A test draw the training table does not already answer.

    THE FLOOR IS IN THE SAMPLER, NOT THE METRIC: the candidate is drawn, its
    exact committor is solved, and the draw is DISCARDED -- never scored, never
    reported -- when `lookup_ceiling` exceeds `floor`. Nothing about any model
    enters; the predicate is a property of the label and of the published
    training goals, computable before a reader exists.

    Returns (B, C, interior, q, residual, iterations, ceilings, n_tries).
    """
    for tries in range(1, MAX_DRAW_TRIES + 1):
        B, C, interior = draw_goal(rng, absorbing)
        q, res, iters = committor(P, B, C, interior)
        ceil = lookup_ceiling(train_q, interior, q[interior])
        if ceil[0] <= floor:
            return B, C, interior, q, res, iters, ceil, tries
    raise AssertionError(f"the floor {floor} rejected {MAX_DRAW_TRIES} "
                         f"consecutive draws: the family has been eaten")


# ---------------------------------------------------------------------------
# THE EXACT LABEL, AND THE TRUNCATIONS OF IT
# ---------------------------------------------------------------------------

def committor(P, B, C, interior, tol=Q_TOL, exact=False):
    """q = 1 on B', 0 on C and on every terminal, harmonic on the interior.

    (I - P_II) x = P_IB 1. Solved by the substochastic fixpoint, not by LU:
    measured on one drawn goal at 24,560 interior orbits, `spsolve` takes
    74.12s to the fixpoint's 0.06s and the two agree to max|x-y| = 1.07e-14
    (`scratchpad/p2_solve_timing.py`, cf9b0d2, WIN-16QAL06O9GB). The fill-in of
    an LU on a chess move graph is the whole difference. `exact=True` runs the
    LU instead, which is how the two are cross-checked.

    The returned residual is measured on the solution that was reached, never on
    the tolerance that was requested.
    """
    idx = np.flatnonzero(interior)
    A = sparse.csr_matrix(P[idx][:, idx])
    rhs = np.asarray(P[idx][:, np.flatnonzero(B)].sum(axis=1)).ravel()
    if exact:
        x = spsolve(sparse.csc_matrix(sparse.eye(idx.size) - A), rhs)
        iters = 0
    else:
        x = rhs.copy()
        for iters in range(1, MAX_ITERS + 1):
            nxt = A @ x + rhs
            step = float(np.abs(nxt - x).max())
            x = nxt
            if step < tol:
                break
        else:
            raise AssertionError(f"committor did not converge in {MAX_ITERS} "
                                 f"iterations (last step {step:.3e})")
    q = np.zeros(P.shape[0])
    q[B] = 1.0
    q[idx] = x
    res = float(np.abs(A @ x + rhs - x).max())
    return q, res, iters


def truncated(P, B, C, interior, depth=2):
    """q_d = P(hit B' within d steps, without touching C or a terminal first).

    Returned as a list q[0]..q[depth-1] for d = 1..depth. q_d is IDENTICALLY
    ZERO wherever B' is more than d hops away, which is the exact sense in which
    a d-layer stack cannot represent the label there.
    """
    idx = np.flatnonzero(interior)
    Pi = sparse.csr_matrix(P[idx])
    acc = np.asarray(Pi @ B.astype(np.float64)).ravel()
    out = [acc.copy()]
    step = np.zeros(P.shape[0])
    tail = acc
    for _ in range(depth - 1):
        step[:] = 0.0
        step[idx] = tail
        tail = np.asarray(Pi @ step).ravel()
        acc = acc + tail
        out.append(acc.copy())
    return out


def hop_distance(P, B, C, max_d=8):
    """Hops from a state TO B' along legal moves, paths through C killed.

    (P @ v)[i] > 0 exactly when i has a successor in v, so this walks
    predecessors outward from B'. -1 means "further than max_d, or unreachable".
    """
    dist = np.full(P.shape[0], -1, np.int32)
    dist[B] = 0
    frontier = B.copy()
    open_ = ~(C | B)
    for d in range(1, max_d + 1):
        nxt = (P @ frontier.astype(np.float64)) > 0
        nxt &= open_ & (dist < 0)
        if not nxt.any():
            break
        dist[nxt] = d
        frontier = nxt
    return dist


# ---------------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "wk_f", "wk_r", "wq_f", "wq_r", "bk_f", "bk_r", "turn",
    "d_wk_bk", "d_wq_bk", "d_wk_wq", "e_wk", "e_wq", "e_bk", "corner_bk",
    "in_check", "deg", "dtm_clip", "is_win", "q_mate",
    "d_bk_to_B", "d_wk_to_B", "d_wq_to_B", "d_bk_to_C", "frac_B", "frac_C",
    "q1", "q2",
]


def features(B, C, interior, q1, q2) -> np.ndarray:
    """One row per interior orbit. State features first, then the goal features
    the refutation names: the king-to-nearest-B' Chebyshev distance and the two
    truncated resolvents."""
    Q, D = quotient(), _derived()
    i = np.flatnonzero(interior)

    def near(piece, mask):
        sq = np.unique(D[piece][mask[:Q["n_orbits"]]])
        if sq.size == 0:
            return np.full(i.size, 8.0)
        return _CHEB[:, sq].min(axis=1)[D[piece][i]].astype(np.float64)

    dtm = Q["dtm"][i]
    cols = [
        D["wk_f"][i], D["wk_r"][i], D["wq_f"][i], D["wq_r"][i],
        D["bk_f"][i], D["bk_r"][i], D["turn"][i],
        D["d_wk_bk"][i], D["d_wq_bk"][i], D["d_wk_wq"][i],
        D["e_wk"][i], D["e_wq"][i], D["e_bk"][i], D["corner_bk"][i],
        Q["in_check"][i], Q["deg"][i],
        np.where(dtm >= 0, dtm, 25.0), (dtm >= 0).astype(np.float64),
        Q["q_mate"][i],
        near("bk", B), near("wk", B), near("wq", B), near("bk", C),
        np.full(i.size, B.sum() / B.size), np.full(i.size, C.sum() / C.size),
        q1, q2,
    ]
    return np.column_stack([np.asarray(c, np.float64) for c in cols])


def _expand(X, mu, sd):
    """Degree-2: the same features plus every pairwise product. The literal
    refutation is the linear arm; this is the strictly more generous one, run so
    a failure to reach KILL_R2 cannot be blamed on linearity."""
    Z = (X - mu) / sd
    n, d = Z.shape
    iu, ju = np.triu_indices(d)
    return np.column_stack([np.ones(n), Z, Z[:, iu] * Z[:, ju]])


# ---------------------------------------------------------------------------
# THE RUN
# ---------------------------------------------------------------------------


#: Column groups of the LINEAR design, for the ablation. Index 0 is the
#: intercept; index j+1 is FEATURE_NAMES[j].
def _groups():
    i = {n: k + 1 for k, n in enumerate(FEATURE_NAMES)}
    goal = [i[n] for n in ("d_bk_to_B", "d_wk_to_B", "d_wq_to_B", "d_bk_to_C",
                           "frac_B", "frac_C", "q1", "q2")]
    res = [i["q1"], i["q2"]]
    allc = list(range(len(FEATURE_NAMES) + 1))
    return {
        "full            ": allc,
        "state only      ": [c for c in allc if c not in goal],
        "no resolvents   ": [c for c in allc if c not in res],
        "resolvents only ": [0] + res,
        "goal features   ": [0] + goal,
    }


def _ablate(At, bt, sy2t, nt_, ss_t, sstot, sswithin):
    """Every sub-fit of the linear design, from the SAME accumulated normal
    equations -- no refit pass and no second sample.

    L-NULL: what is varied is WHICH COLUMNS the fit may use. What is pinned is
    the 100 drawn (B', C) pairs, their exact committors, the 46,137-orbit chain,
    the rows scored, and the estimator (one pooled OLS with the same ridge).
    """
    out = {}
    for name, cols in _groups().items():
        ix = np.ix_(cols, cols)
        A = sum(a[ix] for a in At)
        b = sum(bb[cols] for bb in bt)
        w = _solve_normal(A, b)
        per, ssres = [], 0.0
        for a, bb, s2, ss in zip(At, bt, sy2t, ss_t):
            r = s2 - 2 * w @ bb[cols] + w @ a[ix] @ w
            ssres += r
            per.append(1.0 - r / ss)
        out[name] = dict(pooled=1.0 - ssres / sstot,
                         within=1.0 - ssres / sswithin,
                         per_draw=np.array(per), n_columns=len(cols))
    return out


def _solve_normal(A, b):
    A = A + np.eye(A.shape[0]) * (RIDGE * np.trace(A) / A.shape[0])
    return np.linalg.solve(A, b)


def report(n_test=N_TEST, n_train=N_TRAIN, seed=SEED, verbose=False,
           floor=FLOOR) -> dict:
    """The bed.

    L-NULL: what VARIES is the drawn goal (B', C). What is PINNED is the
    46,137-orbit chain, the committor solver, the feature set, the estimator,
    the seed, and -- since the floor -- the training goals, which are now drawn
    BEFORE any test goal so that the floor has something to reject against.
    """
    t0 = time.time()
    Q = quotient()
    P, PT, absorbing = Q["P"], Q["PT"], Q["absorbing"]
    N = Q["n_states"]
    rng = np.random.default_rng(seed)

    # THE TRAINING GOALS COME FIRST. They are what the lookup ceiling is built
    # from and what the floor rejects against, so they cannot be drawn after
    # the draws they have to filter.
    res_max, iter_max = 0.0, 0
    train = []
    for k in range(n_train):
        B, C, interior = draw_goal(rng, absorbing)
        q, res, iters = committor(P, B, C, interior)
        res_max, iter_max = max(res_max, res), max(iter_max, iters)
        train.append((B, C, interior, q))
    train_q = [d[3] for d in train]

    test, ceil_lookup, ceil_lookup_raw = [], [], []
    n_candidates, rejected = 0, []
    for k in range(n_test):
        B, C, interior, q, res, iters, ceil, tries = draw_test_goal(
            rng, absorbing, P, train_q, floor)
        n_candidates += tries
        res_max, iter_max = max(res_max, res), max(iter_max, iters)
        test.append((B, C, interior, q))
        ceil_lookup.append(ceil[0])
        ceil_lookup_raw.append(ceil[1])
        if verbose:
            print(f"  draw {k:3d}  |B'|={int(B.sum()):6d}  |C|={int(C.sum()):6d}  "
                  f"interior={int(interior.sum()):6d}  res={res:.2e} it={iters} "
                  f"tries={tries} lookup={ceil[0]:.4f}", flush=True)
    n_rejected = n_candidates - n_test

    # ---- pass 1: the pooled normal equations, and the resolvent nesting -----
    # Features are recomputed rather than cached: one `features()` call is two
    # sparse mat-vecs and a column_stack, which is cheaper than 100 copies of a
    # 30k x 27 matrix is to hold.
    def rows(draw, depth=2):
        B, C, interior, q = draw
        qs = truncated(P, B, C, interior, depth)
        return features(B, C, interior, qs[0], qs[1]), q[interior], qs

    d = len(FEATURE_NAMES)
    dq = 1 + d + d * (d + 1) // 2
    Alin, blin = np.zeros((d + 1, d + 1)), np.zeros(d + 1)
    Aq, bq = np.zeros((dq, dq)), np.zeros(dq)
    sy = sy2 = 0.0
    ntot = 0
    max_q1_q2 = max_q2_q = -np.inf
    sx, sx2 = np.zeros(d), np.zeros(d)
    At, bt, sy2t, nt_, syt = [], [], [], [], []
    for draw in test:
        X, y, qs = rows(draw)
        max_q1_q2 = max(max_q1_q2, float((qs[0] - qs[1]).max()))
        max_q2_q = max(max_q2_q, float((qs[1] - y).max()))
        Xl = np.column_stack([np.ones(X.shape[0]), X])
        At.append(Xl.T @ Xl)
        bt.append(Xl.T @ y)
        sy2t.append(float(y @ y))
        syt.append(float(y.sum()))
        nt_.append(y.size)
        Alin += At[-1]
        blin += bt[-1]
        sx += X.sum(0)
        sx2 += (X ** 2).sum(0)
        sy += y.sum()
        sy2 += float(y @ y)
        ntot += y.size
    mu = sx / ntot
    sd = np.sqrt(np.maximum(sx2 / ntot - mu ** 2, 0.0))
    # A feature that never varies on a SCORED row is a dead column, and it is
    # named rather than silently carried: `in_check` is one, because check only
    # ever occurs on positions with no legal move, and those are absorbing and
    # therefore never scored.
    dead = [FEATURE_NAMES[j] for j in np.flatnonzero(sd == 0.0)]
    # and the RANK, which is the number that exposes a duplicate column as well
    # as a dead one. 27 named features is not 27 directions.
    _dg = np.sqrt(np.diag(Alin)); _dg[_dg < 1e-30] = 1.0
    _E = Alin / np.outer(_dg, _dg)
    design_rank = int(np.linalg.matrix_rank(_E, tol=1e-10))
    design_cond = float(np.linalg.cond(Alin))
    sd[sd < 1e-12] = 1.0
    for draw in test:
        X, y, _ = rows(draw)
        Z = _expand(X, mu, sd)
        Aq += Z.T @ Z
        bq += Z.T @ y

    wl, wq_ = _solve_normal(Alin, blin), _solve_normal(Aq, bq)
    ybar = sy / ntot
    sstot = sy2 - ntot * ybar ** 2
    ssres_lin = sy2 - 2 * wl @ blin + wl @ Alin @ wl
    ssres_quad = sy2 - 2 * wq_ @ bq + wq_ @ Aq @ wq_
    r2_lin = 1.0 - ssres_lin / sstot
    r2_quad = 1.0 - ssres_quad / sstot

    # ---- pass 2: per draw, the ceilings, and the hop table -----------------
    per_lin, per_quad, ssres_lin_t, ssres_quad_t = [], [], [], []
    dead_q = {dd: 0.0 for dd in (1, 2, 4, 8)}
    ceil_heur, variances = [], []
    hop_hits = {dd: [0.0, 0.0] for dd in (1, 2, 4, 8)}
    hop_mass = [0.0, 0.0]
    qbar = np.zeros(N)
    for B, C, interior, q in test:
        qbar += np.where(interior, q, np.where(B, 1.0, 0.0))
    qbar /= n_test

    sswithin, ss_t = 0.0, []
    for draw in test:
        B, C, interior, q = draw
        X, y, qs = rows(draw, depth=8)
        ss = float(((y - y.mean()) ** 2).sum())
        sswithin += ss
        ss_t.append(ss)
        variances.append(ss / y.size)
        for w, Z, out, res in ((wl, np.column_stack([np.ones(X.shape[0]), X]),
                                per_lin, ssres_lin_t),
                               (wq_, _expand(X, mu, sd), per_quad, ssres_quad_t)):
            rr = float(((y - Z @ w) ** 2).sum())
            res.append(rr)
            out.append(1.0 - rr / ss)
        for dd in (1, 2, 4, 8):
            dead_q[dd] += float(((qs[dd - 1] <= 0.0) & (y > 0.0)).sum())
        h = qbar[interior]
        raw = 1.0 - float(((y - h) ** 2).sum()) / ss
        c = np.corrcoef(h, y)[0, 1]
        ceil_heur.append(max(raw, 0.0 if not np.isfinite(c) else c ** 2))

        dist = hop_distance(P, B, C)
        di, wmass = dist[interior], Q["osize"][:N][interior]
        hop_mass[0] += di.size
        hop_mass[1] += float(wmass.sum())
        for dd in hop_hits:
            far = (di > dd) | (di < 0)
            hop_hits[dd][0] += float(far.sum())
            hop_hits[dd][1] += float(wmass[far].sum())

    # the per-draw mean is a fixed function of the DRAW, not of the state, so it
    # is the free part of any POOLED R2. Reported as `marginal_pooled` so that
    # 0.7367 is never read as 0.7367 worth of state skill.
    ablations = _ablate(At, bt, sy2t, nt_, ss_t, sstot, sswithin)

    # ---- the same linear fit, held out BY GOAL ------------------------------
    # The refutation's premise is that the goal is UNSEEN, so the fit is also
    # run on the first 80 draws and scored on the 20 it has never seen. Free:
    # the per-draw normal equations are already accumulated.
    # L-NULL: what varies is WHICH draws the fit may see. What is pinned is the
    # 100 (B', C) pairs, their committors, the chain, the columns, the rows
    # scored and the estimator.
    n_fit = (n_test * 4) // 5
    w_ho = _solve_normal(sum(At[:n_fit]), sum(bt[:n_fit]))
    ho = range(n_fit, n_test)
    ssres_ho = sum(sy2t[i] - 2 * w_ho @ bt[i] + w_ho @ At[i] @ w_ho for i in ho)
    n_ho = sum(nt_[i] for i in ho)
    sy_ho = sum(syt[i] for i in ho)
    sstot_ho = sum(sy2t[i] for i in ho) - sy_ho ** 2 / n_ho
    sswithin_ho = sum(ss_t[i] for i in ho)

    ceilings = dict(marginal=0.0, lookup=float(np.mean(ceil_lookup)),
                    heuristic=float(np.mean(ceil_heur)))
    worst = dict(marginal=0.0, lookup=float(np.max(ceil_lookup)),
                 heuristic=float(np.max(ceil_heur)))
    per_draw_ceiling = np.maximum(ceil_lookup, ceil_heur)
    per_draw_headroom = 1.0 - per_draw_ceiling
    hq = {f"headroom_p{k}": float(np.percentile(per_draw_headroom, k))
          for k in (0, 10, 25, 50, 75, 90)}
    boot_lin = _boot_pooled(nt_, syt, sy2t, ssres_lin_t, seed)
    boot_quad = _boot_pooled(nt_, syt, sy2t, ssres_quad_t, seed)
    out = dict(
        n_draws=n_test, n_train=n_train, seed=seed,
        n_orbits=Q["n_orbits"], n_positions=Q["n_positions"], n_moves=Q["n_moves"],
        lumpability_disagreements=Q["lumpability_disagreements"],
        n_rows=ntot,
        max_committor_residual=res_max, max_committor_iterations=iter_max,
        max_q1_minus_q2=max_q1_q2, max_q2_minus_q=max_q2_q,
        r2_linear=float(r2_lin), r2_quadratic=float(r2_quad),
        r2_linear_within=float(1.0 - ssres_lin / sswithin),
        r2_linear_heldout=float(1.0 - ssres_ho / sstot_ho),
        r2_linear_heldout_within=float(1.0 - ssres_ho / sswithin_ho),
        n_fit_draws=n_fit, dead_columns=dead,
        design_rank=design_rank, design_cond=design_cond,
        r2_quadratic_within=float(1.0 - ssres_quad / sswithin),
        marginal_pooled=float(1.0 - sswithin / sstot),
        r2_linear_per_draw=np.array(per_lin), r2_quadratic_per_draw=np.array(per_quad),
        ceiling_marginal=0.0,
        floor=floor, n_candidate_draws=n_candidates, n_rejected_draws=n_rejected,
        acceptance_rate=n_test / n_candidates,
        ceiling_lookup_raw=float(np.mean(ceil_lookup_raw)),
        ceiling_lookup_raw_max=float(np.max(ceil_lookup_raw)),
        per_draw_headroom=per_draw_headroom, **hq,
        ceiling_lookup=ceilings["lookup"], ceiling_lookup_max=float(np.max(ceil_lookup)),
        ceiling_heuristic=ceilings["heuristic"],
        ceiling_heuristic_max=float(np.max(ceil_heur)),
        headroom=1.0 - max(ceilings.values()),
        headroom_worst_draw=1.0 - max(worst.values()),
        draws_with_headroom_below_0p2=int((per_draw_ceiling > 0.8).sum()),
        ceiling_lookup_q=[float(x) for x in np.percentile(ceil_lookup, [50, 90, 100])],
        r2_linear_boot=boot_lin, r2_quadratic_boot=boot_quad,
        dead_mass={dd: dead_q[dd] / ntot for dd in dead_q},
        mean_label_variance=float(np.mean(variances)),
        ablations=ablations,
        hop_uniform={dd: hop_hits[dd][0] / hop_mass[0] for dd in hop_hits},
        hop_weighted={dd: hop_hits[dd][1] / hop_mass[1] for dd in hop_hits},
        seconds=time.time() - t0,
    )
    return out



def _boot_pooled(n_t, sy_t, sy2_t, ssres_t, seed, n_boot=2000):
    """A percentile interval for the POOLED R2, resampling DRAWS with
    replacement.

    L-NULL: what varies is which of the 100 (B', C) draws enter the pool. What
    is pinned is the fitted coefficient vector, the 46,137-orbit chain, the
    feature set, the estimator, and every per-draw sufficient statistic -- the
    resample re-weights draws and changes nothing else.
    """
    n_t, sy_t = np.asarray(n_t, float), np.asarray(sy_t, float)
    sy2_t, ssres_t = np.asarray(sy2_t, float), np.asarray(ssres_t, float)
    rng = np.random.default_rng(seed + 1)
    k = rng.integers(0, n_t.size, size=(n_boot, n_t.size))
    N, SY = n_t[k].sum(1), sy_t[k].sum(1)
    sstot = sy2_t[k].sum(1) - SY ** 2 / N
    r2 = 1.0 - ssres_t[k].sum(1) / sstot
    return float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))


def _ci(v, z=1.96):
    return float(np.mean(v)), float(z * np.std(v, ddof=1) / np.sqrt(v.size))


def demo(n_test=N_TEST, n_train=N_TRAIN, seed=SEED, verbose=False) -> dict:
    r = report(n_test, n_train, seed, verbose)
    print(f"PROVENANCE     python -m ceqjepa.goal_family --draws {n_test} "
          f"--train {n_train}   seed={seed}")
    print(f"THE SPACE      {r['n_positions']} positions, {r['n_moves']} legal moves, "
          f"{r['n_orbits']} D4 orbits")
    print(f"LUMPABILITY    {r['lumpability_disagreements']} of {r['n_positions']} "
          f"positions disagree with their orbit row")
    print(f"THE LABELS     {r['n_draws']} test draws + {r['n_train']} training goals, "
          f"{r['n_rows']} scored rows")
    print(f"               max committor residual {r['max_committor_residual']:.3e}"
          f"  (worst draw took {r['max_committor_iterations']} fixpoint iterations)")
    print(f"               max(q1-q2) {r['max_q1_minus_q2']:.3e}   "
          f"max(q2-q) {r['max_q2_minus_q']:.3e}")
    m, h = _ci(r["r2_linear_per_draw"])
    mq, hq = _ci(r["r2_quadratic_per_draw"])
    bl, bq = r["r2_linear_boot"], r["r2_quadratic_boot"]
    print(f"POOLED R2      linear    {r['r2_linear']:.4f}  "
          f"[{bl[0]:.4f}, {bl[1]:.4f}] 95% bootstrap over draws   "
          f"per-draw mean {m:.4f} +/- {h:.4f}")
    print(f"               degree-2  {r['r2_quadratic']:.4f}  "
          f"[{bq[0]:.4f}, {bq[1]:.4f}]                           "
          f"per-draw mean {mq:.4f} +/- {hq:.4f}")
    print(f"HELD OUT       the same linear fit, trained on goals "
          f"0..{r['n_fit_draws']-1} and scored on the "
          f"{r['n_draws']-r['n_fit_draws']} goals it never saw: pooled "
          f"{r['r2_linear_heldout']:.4f}   within {r['r2_linear_heldout_within']:.4f}")
    print(f"DEAD COLUMNS   {r['dead_columns']} never vary on a scored row; the "
          f"28-column design has rank {r['design_rank']}, cond "
          f"{r['design_cond']:.3e}")
    print(f"FREE PART      predicting each draw's own MEAN scores pooled R2 "
          f"{r['marginal_pooled']:.4f} with no state skill at all")
    print(f"WITHIN-DRAW R2 linear    {r['r2_linear_within']:.4f}   "
          f"degree-2  {r['r2_quadratic_within']:.4f}   "
          f"(the same fits, against the within-draw variance only)")
    print(f"KILL LINE      {KILL_R2} on the LINEAR pooled fit (the refutation as "
          f"stated) -> the leap is "
          f"{'DEAD' if r['r2_linear'] >= KILL_R2 else 'ALIVE'}")
    print(f"               the degree-2 arm is not the stated test and is strictly "
          f"more generous; it reads "
          f"{'>=' if r['r2_quadratic'] >= KILL_R2 else '<'} {KILL_R2}")
    print(f"CEILINGS       marginal   {r['ceiling_marginal']:.4f}")
    print(f"               lookup     {r['ceiling_lookup']:.4f} "
          f"(worst draw {r['ceiling_lookup_max']:.4f})")
    print(f"               heuristic  {r['ceiling_heuristic']:.4f} "
          f"(worst draw {r['ceiling_heuristic_max']:.4f})")
    print(f"               lookup ceiling over draws: median "
          f"{r['ceiling_lookup_q'][0]:.4f}  p90 {r['ceiling_lookup_q'][1]:.4f}  "
          f"max {r['ceiling_lookup_q'][2]:.4f}")
    print(f"SAMPLER FLOOR  reject a test draw whose lookup ceiling exceeds "
          f"{r['floor']}: {r['n_candidate_draws']} candidates drawn, "
          f"{r['n_rejected_draws']} rejected, acceptance "
          f"{r['acceptance_rate']:.4f}")
    print(f"               the same table WITHOUT the free per-draw affine "
          f"recalibration reads {r['ceiling_lookup_raw']:.4f} mean, "
          f"{r['ceiling_lookup_raw_max']:.4f} worst")
    print(f"HEADROOM       {r['headroom']:.4f} on the family mean   "
          f"(oracle 1.0 by construction)")
    print(f"  PER DRAW     worst {r['headroom_p0']:.4f}   p10 "
          f"{r['headroom_p10']:.4f}   p25 {r['headroom_p25']:.4f}   median "
          f"{r['headroom_p50']:.4f}   p75 {r['headroom_p75']:.4f}   p90 "
          f"{r['headroom_p90']:.4f}")
    print(f"               {r['draws_with_headroom_below_0p2']} of {r['n_draws']} "
          f"draws leave a faker less than 0.2 headroom")
    row = "  ".join(f"d={dd}: {r['dead_mass'][dd]:.4f}" for dd in (1, 2, 4, 8))
    print(f"DEPTH-DEAD     {row}")
    print("               (fraction of scored rows where the d-hop truncation "
          "is exactly 0 while q > 0)")
    print("ABLATION       columns the pooled linear fit may use "
          "(same draws, same labels, same estimator)")
    for name, a in r["ablations"].items():
        mm, hh = _ci(a["per_draw"])
        print(f"  {name} {a['n_columns']:3d} cols   pooled {a['pooled']:.4f}   "
              f"within {a['within']:.4f}   per-draw {mm:.4f} +/- {hh:.4f}")
    for tag in ("hop_uniform", "hop_weighted"):
        row = "  ".join(f"d={dd}: {r[tag][dd]:.4f}" for dd in (1, 2, 4, 8))
        print(f"HOPS>{tag[4:]:<9} {row}")
    print(f"SECONDS        {r['seconds']:.1f}")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=N_TEST)
    ap.add_argument("--train", type=int, default=N_TRAIN)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    demo(a.draws, a.train, a.seed, a.verbose)
    sys.exit(0)
