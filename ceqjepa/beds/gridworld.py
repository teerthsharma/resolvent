"""ceqjepa/beds/gridworld.py -- reach-avoid on a grid, with do(a) = BLOCK ONE CELL.

WHY THIS FILE EXISTS. The live demo (scratchpad/counterfactual-field.html) shows a
cursor moving over a grid while every cell recolours by what would happen if that
cell were blocked. Its numbers are exact, because the page is a JavaScript linear
solver: buildChain() assembles Q and R for a uniform walk on the grid and
solveCommittor() runs Gaussian elimination with partial pivoting, once per candidate
block. Nothing is predicted there. To put a MODEL behind that page you need a bed in
the page's own domain, and this repo did not have one: every trained checkpoint here
is a chess model (X_DIM=769, nA=4 outcomes) and the demo's domain is a grid with two
outcomes. This module is that bed.

THE CHAIN, AND WHAT do(a) IS. Cells are GOAL, HAZARD, WALL or free. The walker steps
to a uniformly random in-bounds non-wall neighbour; GOAL and HAZARD absorb. The label
is the committor

    q_i = P(reach GOAL before HAZARD | start at i),      q_goal + q_hazard = 1

and the intervention do(b) is turning cell b into a wall -- exactly what the page's
cursor does (`if(j===blockIdx) continue` inside its buildChain). Every sample carries
the observational committor at its start cell AND the committor under each candidate
block, so the do-arm has an EXACT target rather than a rollout mean: chess_do.py's
Var[p_hat] = p(1-p)/R noise is simply absent here, because a linear solve is not a
sample. `do_outcome_R = 0` is how that is recorded in the InterventionSample-shaped
record: zero rollouts, one exact solve.

AGREEMENT WITH THE PAGE (measured, node v25.8.1 running the page's own buildChain and
solveCommittor verbatim, driven from scratchpad/js/ref.js):

    demo frame-0 grid, G=14, 169 transient cells   worst |q_py - q_js| = 3.553e-15
      the same grid with cell 45 / 105 / 149 blocked      1.332e-15 / 2.665e-15 / 3.109e-15
    the pinned 5x5 grid below, 21 transient cells        worst = 2.220e-16
    conservation, max |q_goal + q_hazard - 1|            2.665e-15 (py), 9.104e-15 (js)

That is solver tolerance, so the demo and this bed compute the same object. JS_5X5_Q
below is the page's own output for that 5x5 grid, pinned as a literal; demo() asserts
against it, so the agreement is re-checked on every run without needing node.

TWO BUGS IN THE PAGE, FOUND BY THAT COMPARISON, NOT PAPERED OVER.
 1. computeField() reads the WRONG ROW of its own solve. It recomputes the cursor's
    index as the position of `target` in tList WITH the blocked cell removed, but
    buildChain(b) does not remove b from tList -- it returns nT = tList.length rows
    and leaves b's own row in place. So whenever the blocked cell precedes the cursor
    in raster order the field is read one row early. Measured on the demo's frame-0
    grid with the cursor at r7c7: 92 of 168 blocked cells report a wrong delta-q, mean
    absolute error 0.0313, worst 0.0908 -- against a mean TRUE |delta q| of 0.0087 and
    a largest true |delta q| of 0.0552. The error averages 3.6x the effect it is
    reporting and the worst wrong reading exceeds the biggest real one, so field mode
    is showing mostly artefact. The fix is `qb[tIdx[target]*2]`; delete the `ti` loop.
 2. The page's table claims a Sherman-Morrison update verified against a dense
    re-solve, but the shipped computeField() performs a full buildChain +
    solveCommittor per candidate. There is no rank-1 update in the file.
Neither bug is inherited here: this module blocks a cell by writing WALL into a copy
of the grid, so the block IS a wall and there is exactly one chain builder.

WHY NOT operator.committor(). It is the repo's committor and it was the first thing
tried, and it CANNOT be used for a grid. It solves with
torch.linalg.solve_triangular(..., upper=False), which is exact for the causal
(lower-triangular) chain the architecture emits and silently wrong for anything else.
Fed the demo page's frame-0 grid chain, embedded as a row-stochastic P with two
absorbing rows, it returns rows summing to between 0.000000 and 0.490398 instead of 1,
worst coordinate error 9.0365e-01 against the general solve -- no exception, no NaN,
just a wrong q. demo() re-measures the same failure on a grid of its own (rows in
[0.000000, 1.000000], worst 9.6966e-01) as planted negative 4. Ground truth here
therefore comes from kappa_bed.committor(), which is the same repo, the same math, and
a general torch.linalg.solve. This is a note about where solve_triangular applies, not
a defect in operator.py.

THE TWO KNOBS, AND WHAT EACH ONE MOVES.

  `gap` -- the width of the opening in a wall that splits the grid. gap=OPEN is no
  wall at all, the plain uniform walk, and is this bed's control arm the way
  chess_policy's T=UNIFORM is: it is not a large gap, it is the absence of the
  barrier. Narrowing the gap makes which side of the barrier you start on dominate
  the outcome, so the board tells you more about the label. MEASURED, G=9, 4 noise
  walls, N=200 instances x R=20 Bernoulli draws from the exact q, 200-permutation
  label-shuffle null, seed 0 (demo()'s own output):

    gap    I(X;Y)   shuffle null      sigma  H(pi)   sd(q)  outcome balance
    OPEN   0.1073   0.0256 +- 0.0027   40.5  0.6923  0.222  goal 0.480 / haz 0.520
    5      0.1685   0.0257 +- 0.0027   63.2  0.6928  0.269  goal 0.486 / haz 0.514
    3      0.2119   0.0257 +- 0.0026   82.7  0.6930  0.304  goal 0.508 / haz 0.492
    1      0.3986   0.0257 +- 0.0025  157.2  0.6930  0.395  goal 0.507 / haz 0.493

  For scale, chess_policy's uniform control reads 0.1363 nats at 28.3 sigma and its
  recommended T=0.25 reads 0.3084 at 54.9 sigma. The label is balanced at every gap
  (H is within 0.001 of ln 2 = 0.6931), so none of this is a bed that always says the
  same thing.

  WHERE THE gap KNOB STOPS MEANING ANYTHING. gap=1 is the narrowest PASSABLE opening
  and is where the asserted monotonicity ends. Sealing the barrier (gap=0) measures
  I=0.6675 at 264 sigma with sd(q)=0.500 -- a bigger number and a worthless one: with
  no opening q is exactly 0 or 1 and the label is "which side did you start on", not
  anything about the walk. That is the same shape of trap chess_policy records below
  T=0.25, where I recovers only because the entropy it is measured against collapsed.
  Do not read the trend past gap=1.

  `stay` -- the walker's probability of standing still. P' = stay*I + (1-stay)*P, so
  Q' = stay*I + (1-stay)*Q and R' = (1-stay)*R, hence (I-Q') = (1-stay)(I-Q) and

      q' = q  EXACTLY,      kappa' = kappa / (1 - stay)  EXACTLY.

  This is the kappa_bed lesson in its cleanest form. That module's docstring records
  a generator whose kappa could not be controlled at all -- targets {5,20,50,100} all
  measured kappa in [3.48, 5.45] -- until the self-loop was restored. Here the
  self-loop is the knob, and because it is a pure lazification it moves the absorption
  time without moving a single label. MEASURED (G=9, gap=3, 4 noise walls, seed 0,
  MEDIAN over 20 instances, kappa via kappa_bed.kappa_of; demo()'s own output):

    stay      0.00     0.50     0.80      0.90      0.95      0.98
    kappa  109.265  218.530  546.324  1092.649  2185.297  5463.243
    max |q(stay) - q(stay=0)| over all of them: 0.000e+00 .. 4.885e-15

  50x span, monotone, and the ratio to kappa(stay=0) tracks 1/(1-stay) to 4.97e-14.
  The range actually measured over those instances, min to max, is kappa in
  [67.7, 6963.3]. It is not a claim that any particular kappa was targeted -- the knob
  is measured per instance and reported, never assumed.

  KAPPA IS MEASURED HERE, AND COMPARING IT TO 1/TELEPORT WAS A CATEGORY ERROR. An
  earlier version of this docstring raised an alarm that the bed's kappa (median
  109.3 on a bare 9x9) exceeds the 1/TELEPORT = 80 that operator.py holds the MODEL's
  operator to. It also quoted "164.25" for the demo's 14x14 grid -- a number nothing
  in this file ever computed, written as a literal into a print() and into this
  docstring. Both are struck. q is HARMONIC, so a model never has to reproduce the
  chain, only its committor: an operator built by operator.build_operator at
  kappa_model = 1.000000 fits a target field whose true chain reads kappa = 300.8 to
  1.110e-16, the float64 floor. The bed's kappa is not a representation constraint.

  WHAT THE TELEPORT ACTUALLY COSTS, measured on this bed, 1,859 blocked-cell do()
  solves at G=9 gap=3: it attenuates the interventional effect the do-arm is trained
  on by 43.1% (mean |dq| 0.021501 true vs 0.012590 teleported; mean displacement
  0.010243, 47.6% of the bite), with 7 sign flips among the 724 solves whose true
  effect clears 0.01. At c = 0 the same path is exact to 1.887e-15, so that is the
  teleport and not the solver. The quantity worth watching is c * kappa_model -- the
  teleport's share of the committor, computed every step at train.py:643 and never
  multiplied by c -- not the bed's kappa, which q is invariant to.

DOES THE INTERVENTION BITE? Partly, and sparsely. MEASURED, G=9, 60 instances, EVERY
blockable cell blocked in turn against one random start cell per instance (demo()'s
own output; "best" is the largest |dq| in an instance, median over instances):

    gap    n      mean    median     p90      p99     max   |dq|>.01 |dq|>.05  best
    OPEN  4798   0.0164   0.0068   0.0431   0.1249  0.2947   42.0%    7.9%   0.1311
    3     4800   0.0146   0.0018   0.0451   0.1438  0.5921   30.5%    8.8%   0.1227
    1     4799   0.0095   0.0001   0.0214   0.1946  0.5176   16.4%    4.8%   0.0708

READ THAT HONESTLY: the median block does almost nothing, and it does LESS as the gap
narrows -- the bite is not monotone in the knob that maximises information. What
narrowing buys is the tail: the biggest single block goes from 0.2947 to 0.5921,
because at gap=3 the decisive cells are the three gap cells and blocking one of them
moves the committor by half a probability. The counterfactual field this bed feeds is
therefore SPARSE: mostly flat, with a bottleneck that lights up. That is a real
property of reach-avoid on a grid, it is what the demo page will show, and it is a
weaker do-arm signal than a uniformly-sampled candidate suggests -- 69.5% of uniformly
drawn blocks at gap=3 move q by less than 0.01, so a do-arm trained on uniform
candidates is mostly trained on "nothing happened". gap=3 is the recommended setting
because it maximises the fraction of blocks that move q by more than 0.05 (8.8%, vs
7.9% at OPEN and 4.8% at gap=1) while keeping I(X;Y) at 83 sigma over its null.

PLANTED NEGATIVES. demo() runs four checks that must FAIL on deliberately broken input
before the passing ones mean anything, and prints the before/after of each (measured,
demo()'s own output):

  1 labels decoupled from the board  I 0.2119 @ 82.7 sigma -> -0.0015 @ -0.6 sigma
  2 a grid with no HAZARD           q spread 0.9451 -> 4.552e-15, DegenerateBedError
  3 a block on an already-WALL cell  delta -0.000426 -> exactly +0.000000e+00
  4 operator.committor on this chain row sums [1,1] -> [0.000000, 1.000000]

Negative 1 is here in its SECOND form because the first form could not fail. It
permuted q across instances, and read 81.5 sigma: plug-in MI is computed from the
count table and is invariant to row order, so permuting which row holds which q is a
no-op on the statistic. The version that ships draws each outcome from a uniformly
random instance's committor instead of its own, which breaks the board-outcome link
while leaving the marginal alone, and reads -0.6 sigma. This repo has shipped
self-checks that could not fail -- including a corner test that passed an empty
absorbing set and an eval script that printed ALL SELF-CHECKS PASSED while calling a
function that did not exist -- so a check that has never been seen to fire is not
counted here as a check, including one of mine.

INTERFACE. `build_intervention_dataset` returns a list of `GridSample`, whose fields
are chess_do.InterventionSample's field-for-field (game_id / ply_idx / fen / obs_uci /
obs_outcome / candidate_ucis / do_outcome_mean / do_outcome_R) so a ChessDoBed-shaped
consumer needs no new field names; `fen` is a grid string rather than a chess FEN and
`fen_to_vec` here is the one place that turns it into a model input. `GridBed` is the
ChessDoBed-shaped adapter (nA / x_dim / batch / batch_do / subset / group_ids /
row_keys / pad_rate), and demo() proves it by driving it through train.py's own
draw() and heldout_split() without importing anything from this file into train.py.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch

from ceqjepa.beds.kappa_bed import committor as _solve_committor, kappa_of
from ceqjepa.beds.chess_policy import _plugin_mi   # ONE definition of the MI estimator
from ceqjepa.operator import TELEPORT

__all__ = [
    "FREE", "GOAL", "HAZARD", "WALL", "OPEN", "N_OUTCOMES", "OUTCOME_NAMES",
    "G_DEFAULT", "N_WALLS_DEFAULT", "X_DIM", "JS_5X5_Q",
    "DegenerateBedError", "GridSample", "GridSolve", "GridBed",
    "make_grid", "grid_to_fen", "fen_to_grid", "fen_to_vec", "cell_name",
    "solve_grid", "delta_field", "check_informative", "outcome_distribution",
    "build_intervention_dataset", "measure_information", "measure_kappa",
    "measure_bite", "demo",
]

FREE, GOAL, HAZARD, WALL = 0, 1, 2, 3
_CHR = ".GHW"

N_OUTCOMES = 2
OUTCOME_NAMES = ("goal", "hazard")

#: gap=OPEN is the ABSENCE of the barrier, not a wide one -- the control arm, the way
#: chess_policy.UNIFORM is chess.py:109 itself rather than a high temperature.
OPEN = math.inf

G_DEFAULT = 9
N_WALLS_DEFAULT = 4
X_PLANES = 4                                   # is_goal, is_hazard, is_wall, is_start
X_DIM = X_PLANES * G_DEFAULT * G_DEFAULT       # 324 at the default G

#: counterfactual-field.html's OWN solveCommittor output, q(reach goal first), for the
#: 5x5 grid GOAL=(0,4) HAZARD=(4,0) WALL=(2,1),(2,3), in raster order over the 21
#: transient cells. Produced by node v25.8.1 driving the page's verbatim buildChain and
#: solveCommittor. demo() asserts this bed reproduces it.
JS_5X5_Q = (
    0.5323340471092071, 0.5875802997858666, 0.6695931477516053, 0.8034261241970018,
    0.47708779443254745, 0.5608137044967874, 0.617773019271948, 0.7406852248394,
    0.8008565310492501, 0.3381156316916483, 0.4999999999999996, 0.6618843683083508,
    0.1991434689507492, 0.25931477516059925, 0.382226980728051, 0.4391862955032116,
    0.5229122055674514, 0.19657387580299757, 0.33040685224839356, 0.4124197002141323,
    0.4676659528907919,
)


class DegenerateBedError(ValueError):
    """The committor field is constant, so the label carries no information about the
    start cell. A grid with no HAZARD is the canonical case: q == 1 everywhere and a
    model scores perfectly by ignoring its input. Raised, not warned."""


# --------------------------------------------------------------------- geometry

def _neighbours(G, i):
    r, c = divmod(i, G)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < G and 0 <= cc < G:
            out.append(rr * G + cc)
    return out


def cell_name(G, i):
    r, c = divmod(i, G)
    return "r%dc%d" % (r, c)


def make_grid(rng, G=G_DEFAULT, gap=3, n_walls=N_WALLS_DEFAULT, hazard=True):
    """One instance. A wall down column G//2 with an opening `gap` rows wide, a 2x2
    GOAL on the far side, a 2x2 HAZARD on the near side, plus `n_walls` random extra
    walls so no two instances are the same board.

    gap=OPEN removes the barrier entirely (the control arm). gap=0 seals it, which is
    the [LIMIT] case demo() reports: the label becomes 'which side did you start on',
    exactly binary, and stops being about the walk at all.

    `hazard=False` builds the deliberately broken grid the planted negative needs.
    """
    assert G >= 8, "G must be at least 8 for the barrier / goal / hazard layout to fit"
    cells = np.zeros((G, G), np.uint8)
    bar = G // 2
    if gap < G:
        lo = rng.randrange(1, G - int(gap)) if gap >= 1 else 0
        for r in range(G):
            if not (gap >= 1 and lo <= r < lo + int(gap)):
                cells[r, bar] = WALL
    r0 = rng.randrange(0, max(1, bar - 2))
    c0 = rng.randrange(bar + 1, G - 1)
    cells[r0:r0 + 2, c0:c0 + 2] = GOAL
    if hazard:
        r1 = rng.randrange(bar + 2, G - 1)
        c1 = rng.randrange(0, max(1, bar - 1))
        cells[r1:r1 + 2, c1:c1 + 2] = HAZARD
    free = [(r, c) for r, c in zip(*np.where(cells == FREE))]
    for r, c in rng.sample(free, min(n_walls, len(free))):
        cells[r, c] = WALL
    return cells


def grid_to_fen(cells, start):
    """'<G>:<G*G cell chars>:<start cell index>'. The grid analogue of a FEN, and the
    only string form -- fen_to_vec below is the one place it becomes a model input."""
    G = cells.shape[0]
    return "%d:%s:%d" % (G, "".join(_CHR[v] for v in cells.ravel()), int(start))


def fen_to_grid(fen):
    g, body, start = fen.split(":")
    G = int(g)
    cells = np.array([_CHR.index(ch) for ch in body], np.uint8).reshape(G, G)
    return cells, int(start)


def fen_to_vec(fen):
    """[4*G*G] float32: is_goal, is_hazard, is_wall, is_start. The label depends on
    the start cell, so the start cell is IN the input -- a board with no start marker
    does not determine q and the bed would be measuring the wrong thing."""
    cells, start = fen_to_grid(fen)
    flat = cells.ravel()
    n = flat.size
    v = np.zeros(X_PLANES * n, np.float32)
    v[0 * n:1 * n] = (flat == GOAL)
    v[1 * n:2 * n] = (flat == HAZARD)
    v[2 * n:3 * n] = (flat == WALL)
    v[3 * n + start] = 1.0
    return v


# ------------------------------------------------------------------ the chain

def _absorbing_reachable(cells):
    """Free cells that can reach GOAL or HAZARD through free cells. A cell that cannot
    never absorbs, so its row would make I-Q singular; the page returns null there
    (`solveCommittor` refuses on a zero pivot) and this bed drops it from the transient
    set and counts it instead. Dropping is exact: an unreachable pocket appears in no
    other row's equation, so no other committor changes."""
    G = cells.shape[0]
    seen = np.zeros((G, G), bool)
    stack = []
    for r in range(G):
        for c in range(G):
            if cells[r, c] != FREE:
                continue
            for j in _neighbours(G, r * G + c):
                if cells[divmod(j, G)] in (GOAL, HAZARD):
                    seen[r, c] = True
                    stack.append((r, c))
                    break
    while stack:
        r, c = stack.pop()
        for j in _neighbours(G, r * G + c):
            rr, cc = divmod(j, G)
            if cells[rr, cc] == FREE and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return seen


@dataclass
class GridSolve:
    """One solved instance. Every field is measured off the grid that was actually
    built, in the spirit of kappa_bed.BedSpec: nothing here is a requested target."""
    cells: np.ndarray
    stay: float
    Q: np.ndarray
    R: np.ndarray
    tlist: list
    tidx: dict
    q: np.ndarray            # [nT, 2] -- columns are (goal, hazard)
    n_stranded: int          # free cells that can never absorb; excluded, not solved

    @property
    def kappa(self):
        """||(I-Q)^{-1}||_inf = max_i E_i[tau], MEASURED via kappa_bed.kappa_of."""
        return kappa_of(torch.from_numpy(self.Q))

    @property
    def conservation(self):
        return float(np.abs(self.q.sum(1) - 1.0).max())

    def q_at(self, cell):
        return self.q[self.tidx[cell]]


def solve_grid(cells, stay=0.0):
    """Build Q, R for the walk on `cells` and solve q = (I-Q)^{-1} R.

    The chain is counterfactual-field.html's buildChain: uniformly random in-bounds
    non-wall neighbour, GOAL and HAZARD absorbing. `stay` adds a self-loop
    P' = stay*I + (1-stay)*P, which leaves q identical and scales kappa by 1/(1-stay).
    """
    assert 0.0 <= stay < 1.0, "stay must be in [0,1); got %r" % (stay,)
    G = cells.shape[0]
    ok = _absorbing_reachable(cells)
    tlist = [r * G + c for r in range(G) for c in range(G) if ok[r, c]]
    n_stranded = int((cells == FREE).sum()) - len(tlist)
    assert tlist, "no free cell can reach GOAL or HAZARD -- there is nothing to solve"
    tidx = {i: a for a, i in enumerate(tlist)}
    nT = len(tlist)
    Q = np.zeros((nT, nT))
    R = np.zeros((nT, N_OUTCOMES))
    for a, i in enumerate(tlist):
        nb = [j for j in _neighbours(G, i) if cells[divmod(j, G)] != WALL]
        p = (1.0 - stay) / len(nb)
        Q[a, a] += stay
        for j in nb:
            k = cells[divmod(j, G)]
            if k == GOAL:
                R[a, 0] += p
            elif k == HAZARD:
                R[a, 1] += p
            else:
                Q[a, tidx[j]] += p
    q = _solve_committor(torch.from_numpy(Q), torch.from_numpy(R)).numpy()
    return GridSolve(cells, stay, Q, R, tlist, tidx, q, n_stranded)


def check_informative(sol, tol=1e-9):
    """Raise if the committor field is constant. Called on every instance the dataset
    builds, so a bed that teaches nothing dies at construction."""
    spread = float(sol.q[:, 0].max() - sol.q[:, 0].min())
    if spread < tol:
        raise DegenerateBedError(
            "committor field is constant (q_goal spread %.3e < %.3e over %d transient "
            "cells, value %.6f): the label does not depend on the start cell, so a model "
            "scores perfectly without reading the board. A grid with no HAZARD does this."
            % (spread, tol, len(sol.tlist), float(sol.q[0, 0])))
    return spread


def _blocked(cells, b):
    """do(b): the block IS a wall. One chain builder, so a block on an already-walled
    cell is bitwise the same grid and its delta is exactly 0.0 by construction rather
    than by a tolerance."""
    out = cells.copy()
    out[divmod(int(b), cells.shape[0])] = WALL
    return out


def delta_field(cells, start, stay=0.0, base=None):
    """[G*G] of q_goal(start | block b) - q_goal(start), NaN where b cannot be read.

    This is the demo's 'field' mode, computed correctly -- see the module docstring on
    computeField's off-by-one. NaN at b == start and wherever blocking b strands the
    start cell (no path to either absorbing set), which is a refusal, not a zero."""
    G = cells.shape[0]
    if base is None:
        base = solve_grid(cells, stay)
    b0 = float(base.q_at(start)[0])
    out = np.full(G * G, np.nan)
    for b in range(G * G):
        if b == start:
            continue
        sb = solve_grid(_blocked(cells, b), stay)
        if start in sb.tidx:
            out[b] = float(sb.q_at(start)[0]) - b0
    return out


def outcome_distribution(vectors):
    d = np.asarray(vectors).mean(axis=0)
    return {OUTCOME_NAMES[i]: float(d[i]) for i in range(N_OUTCOMES)}


# ---------------------------------------------------------------- the dataset

@dataclass
class GridSample:
    """chess_do.InterventionSample's fields, field for field, so a ChessDoBed-shaped
    consumer needs no new names. What differs is the CONTENT, and it differs in the
    bed's favour:

      obs_outcome     [2] the EXACT committor at the start cell, not a one-hot draw.
                      Sums to 1 by conservation, which is the same invariant the demo
                      page displays as max |sum q - 1|.
      do_outcome_mean [m, 2] the EXACT committor under each block. Not a mean of
                      anything -- see do_outcome_R.
      do_outcome_R    0. Zero rollouts: this is a linear solve, so chess_do.py's
                      Var[p_hat] = p(1-p)/R <= 1/(4R) label noise is not present.
      fen             a grid string (see grid_to_fen), not a chess FEN.
      obs_uci         '-', the null intervention: the observational arm blocks nothing.
      candidate_ucis  'rNcM' cell names. SHORTER THAN m when a candidate would strand
                      the start cell; the adapter pads and masks exactly the way
                      ChessDoBed does for short endgame move lists.
      ply_idx         the start cell index.

    Two fields are ADDED beyond InterventionSample, both because a grid can supply
    exactly what chess cannot: `kappa`, the instance's measured ||(I-Q)^{-1}||_inf, so
    conditioning is journalled per row rather than assumed from the knob; and
    `next_fen`, the same grid with the start cell advanced one step of the walk, which
    is the JEPA next-state arm (ChessDoBed reconstructs its x_nx by pushing obs_uci,
    which has no grid analogue -- there is no single 'move that was played').
    """
    game_id: int
    ply_idx: int
    fen: str
    obs_uci: str
    obs_outcome: np.ndarray
    candidate_ucis: list
    do_outcome_mean: np.ndarray
    do_outcome_R: int
    kappa: float             # MEASURED per instance, kappa_bed.kappa_of; never assumed
    next_fen: str            # the grid after one step of the walk (the JEPA next state)


def build_intervention_dataset(n_instances=300, seed=0, G=G_DEFAULT, gap=3,
                               n_walls=N_WALLS_DEFAULT, m_candidates=8, stay=0.0):
    """`n_instances` grids; from one random start cell of each, pair the observational
    committor with `m_candidates` blocked-cell committors, all exact.

    One random.Random(seed) drives grid layout, start cell and candidate choice in that
    order, so a fixed (seed, G, gap, n_walls, m_candidates, stay) regenerates a
    byte-identical dataset (asserted in demo())."""
    assert m_candidates >= 1
    rng = random.Random(seed)
    out = []
    for gid in range(n_instances):
        cells = make_grid(rng, G, gap, n_walls)
        sol = solve_grid(cells, stay)
        check_informative(sol)
        start = sol.tlist[rng.randrange(len(sol.tlist))]
        pool = [b for b in sol.tlist if b != start]
        cand = rng.sample(pool, min(m_candidates, len(pool)))
        step = rng.choice([j for j in _neighbours(G, start)
                           if cells[divmod(j, G)] != WALL])

        names, tgt = [], []
        for b in cand:
            sb = solve_grid(_blocked(cells, b), stay)
            if start not in sb.tidx:
                continue                      # blocking b strands the start: refuse it
            names.append(cell_name(G, b))
            tgt.append(sb.q_at(start))
        assert names, "every candidate stranded the start cell -- grid is too fragile"

        out.append(GridSample(
            game_id=gid, ply_idx=int(start), fen=grid_to_fen(cells, start),
            obs_uci="-", obs_outcome=sol.q_at(start).astype(np.float32),
            candidate_ucis=names,
            do_outcome_mean=np.stack(tgt).astype(np.float32), do_outcome_R=0,
            kappa=sol.kappa, next_fen=grid_to_fen(cells, step),
        ))
    assert out, "no instances built"
    return out


class GridBed:
    """ChessDoBed-shaped adapter. Same attribute surface, same tensor shapes, same
    padding-with-a-mask rule -- train.py's draw(), bed_groups() and heldout_split()
    take it unchanged, which demo() proves by calling them."""
    nA = N_OUTCOMES

    def __init__(self, samples, m):
        self.m = m
        G = fen_to_grid(samples[0].fen)[0].shape[0]
        self.x_dim = X_PLANES * G * G
        xs, xns, qs, vs, mvs, tgts, msks = [], [], [], [], [], [], []
        for s in samples:
            xs.append(fen_to_vec(s.fen))
            xns.append(fen_to_vec(s.next_fen))
            qs.append(s.obs_outcome)
            vs.append(s.ply_idx)
            k = len(s.candidate_ucis)
            # PAD by repeating candidate 0 and carry the mask, exactly as ChessDoBed
            # does. Not by dropping short rows: candidates are short precisely where
            # the grid is fragile, and filtering those would bias the bed toward open
            # boards without saying so.
            pad = [0] * (m - k)
            order = list(range(min(k, m))) + pad
            mvs.append([[s.ply_idx, self._cell(G, s.candidate_ucis[c])] for c in order])
            tgts.append(s.do_outcome_mean[order])
            msks.append([c < k for c in range(min(k, m))] + [False] * len(pad))
        self.x = torch.from_numpy(np.stack(xs)).float()
        self.x_nx = torch.from_numpy(np.stack(xns)).float()
        self.q_star = torch.from_numpy(np.stack(qs)).float()
        self.v_idx = torch.tensor(vs, dtype=torch.long)
        self.moves = torch.tensor(mvs, dtype=torch.long)          # [N,m,2] (start, block)
        self.do_tgt = torch.from_numpy(np.stack(tgts)).float()    # [N,m,nA]
        self.do_mask = torch.tensor(msks, dtype=torch.bool)       # [N,m]
        self.pad_rate = 1.0 - float(self.do_mask.float().mean())
        self.group_ids = torch.tensor([s.game_id for s in samples], dtype=torch.long)
        self.row_keys = [a.tobytes() for a in xs]
        self.kappa = torch.tensor([s.kappa for s in samples], dtype=torch.float64)

    @staticmethod
    def _cell(G, name):
        r, c = name[1:].split("c")
        return int(r) * G + int(c)

    def batch_do(self, gen, B):
        i = torch.randint(len(self.x), (B,), generator=gen)
        return (self.x[i], self.x_nx[i], self.q_star[i], self.v_idx[i],
                self.moves[i], self.do_tgt[i], self.do_mask[i])

    def batch(self, gen, B):
        return self.batch_do(gen, B)[:4]

    def subset(self, idx):
        s = object.__new__(GridBed)
        s.m, s.x_dim = self.m, self.x_dim
        for k in ('x', 'x_nx', 'q_star', 'v_idx', 'moves', 'do_tgt', 'do_mask',
                  'group_ids', 'kappa'):
            setattr(s, k, getattr(self, k)[idx])
        s.row_keys = [self.row_keys[i] for i in idx.tolist()]
        s.pad_rate = 1.0 - float(s.do_mask.float().mean())
        return s


# ---------------------------------------------------------------- measurement

def measure_information(gap=3, n_instances=200, rollouts=20, G=G_DEFAULT,
                        n_walls=N_WALLS_DEFAULT, seed=0, n_null=200, shuffle=False):
    """I(X;Y) between a grid-plus-start-cell X and a sampled outcome Y, plug-in MINUS
    a label-permutation null -- the same estimator chess_policy uses, imported from it
    rather than rewritten, so the two beds' numbers are comparable.

    Y is drawn Bernoulli(q) `rollouts` times per instance. q itself is exact, so the
    only sampling noise is the one this estimator's null already accounts for. Below
    3 sigma the bed is NOT USABLE.

    `shuffle=True` is the planted negative, which must read at chance. It draws each
    of the N*R outcomes from a UNIFORMLY RANDOM INSTANCE's committor instead of its
    own, so the board stops determining the outcome while the marginal label
    distribution is unchanged. It is deliberately NOT a permutation of q across rows:
    that was tried first and read 81.5 sigma, because permuting which row holds which
    q leaves the count table a row-permutation of itself and plug-in MI is invariant
    to row order. A negative control that cannot fail is the thing this bed exists to
    avoid, so the broken one is recorded here rather than quietly replaced."""
    rng = random.Random(seed)
    qs, spreads = [], []
    while len(qs) < n_instances:
        sol = solve_grid(make_grid(rng, G, gap, n_walls))
        if len(sol.tlist) < 4:
            continue
        spreads.append(check_informative(sol))
        qs.append(float(sol.q[rng.randrange(len(sol.tlist)), 0]))
    qs = np.asarray(qs)

    nrng = np.random.default_rng(seed)
    p = qs[nrng.integers(n_instances, size=(n_instances, rollouts))] if shuffle \
        else np.repeat(qs[:, None], rollouts, axis=1)
    draws = (nrng.random((n_instances, rollouts)) < p).astype(int)
    counts = np.stack([np.bincount(r, minlength=N_OUTCOMES) for r in draws]).astype(float)

    i_plugin = _plugin_mi(counts)
    labels = np.repeat(np.arange(N_OUTCOMES), counts.sum(0).astype(int))
    null = np.empty(n_null)
    for b in range(n_null):
        sh = nrng.permutation(labels).reshape(n_instances, rollouts)
        null[b] = _plugin_mi(
            np.stack([np.bincount(r, minlength=N_OUTCOMES) for r in sh]).astype(float))
    nm, nsd = float(null.mean()), float(null.std(ddof=1))
    info = i_plugin - nm
    py = counts.sum(0) / counts.sum()
    h = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return dict(
        gap=gap, shuffle=shuffle, n_instances=n_instances, rollouts=rollouts,
        i_plugin=i_plugin, null_mean=nm, null_sd=nsd, info=info,
        sigmas=info / nsd if nsd > 0 else float("inf"), h=h,
        ppl_marginal=math.exp(h), ppl_bayes=math.exp(max(h - info, 0.0)),
        ppl_chance=float(N_OUTCOMES), q_sd=float(qs.std()), q_mean=float(qs.mean()),
        mean_spread=float(np.mean(spreads)),
        outcome_freq={OUTCOME_NAMES[k]: round(float(py[k]), 4) for k in range(N_OUTCOMES)},
        usable=bool(info > 3.0 * nsd),
    )


def measure_kappa(stay, n_instances=20, G=G_DEFAULT, gap=3, n_walls=N_WALLS_DEFAULT, seed=0):
    """kappa per instance via kappa_bed.kappa_of, plus the exact drift of q against the
    stay=0 labels -- the claim that this knob moves conditioning and nothing else."""
    rng = random.Random(seed)
    ks, drift = [], 0.0
    for _ in range(n_instances):
        cells = make_grid(rng, G, gap, n_walls)
        s0, ss = solve_grid(cells, 0.0), solve_grid(cells, stay)
        ks.append(ss.kappa)
        drift = max(drift, float(np.abs(ss.q - s0.q).max()))
    ks = np.asarray(ks)
    return dict(stay=stay, n=n_instances, kappa_med=float(np.median(ks)),
                kappa_min=float(ks.min()), kappa_max=float(ks.max()), q_drift=drift,
                over_teleport_bound=float((ks > 1.0 / TELEPORT).mean()))


def measure_bite(gap=3, n_instances=60, G=G_DEFAULT, n_walls=N_WALLS_DEFAULT, seed=1):
    """|q_do - q_obs| over EVERY blockable cell, one random start cell per instance.
    This is the demo's whole field, so it is the honest answer to 'does the cursor do
    anything', not a statistic over hand-picked blocks."""
    rng = random.Random(seed)
    ds, tops = [], []
    for _ in range(n_instances):
        cells = make_grid(rng, G, gap, n_walls)
        sol = solve_grid(cells)
        if len(sol.tlist) < 6:
            continue
        start = sol.tlist[rng.randrange(len(sol.tlist))]
        f = np.abs(delta_field(cells, start, base=sol))
        f = f[~np.isnan(f)]
        ds.append(f)
        tops.append(f.max())
    d = np.concatenate(ds)
    return dict(gap=gap, n=int(d.size), n_instances=len(ds), mean=float(d.mean()),
                median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                p99=float(np.percentile(d, 99)), max=float(d.max()),
                frac_01=float((d > 0.01).mean()), frac_05=float((d > 0.05).mean()),
                top_med=float(np.median(tops)))


# --------------------------------------------------------------------- demo

def _pinned_5x5():
    cells = np.zeros((5, 5), np.uint8)
    cells[0, 4] = GOAL
    cells[4, 0] = HAZARD
    cells[2, 1] = WALL
    cells[2, 3] = WALL
    return cells


def demo():
    """Assert-based self-check plus the measured report. CPU only, no network, no GPU."""
    # (0) determinism -- the interface promise, kept.
    a = build_intervention_dataset(n_instances=12, seed=0, m_candidates=4)
    b = build_intervention_dataset(n_instances=12, seed=0, m_candidates=4)
    assert [s.fen for s in a] == [s.fen for s in b], "seed did not regenerate the dataset"
    assert all(np.allclose(x.do_outcome_mean, y.do_outcome_mean) for x, y in zip(a, b))

    # (a) MUST-FIRE #1: agree with the HTML page's OWN solver, bitwise-pinned.
    sol5 = solve_grid(_pinned_5x5())
    worst_js = float(np.abs(sol5.q[:, 0] - np.asarray(JS_5X5_Q)).max())
    assert sol5.tlist == [i for i in range(25) if i not in (4, 20, 11, 13)]
    assert worst_js < 1e-12, (
        "this bed and counterfactual-field.html compute DIFFERENT chains: worst "
        "|q_py - q_js| = %.3e on the pinned 5x5 grid" % worst_js)
    print("[MEASURED] vs counterfactual-field.html's own buildChain+solveCommittor "
          "(node v25.8.1): worst |q_py - q_js| = %.3e on the pinned 5x5 grid, 21 transient "
          "cells; 3.553e-15 on the page's frame-0 G=14 grid, 169 transient cells "
          "(scratch run, see module docstring)." % worst_js)
    print("[MEASURED] conservation, max |q_goal + q_hazard - 1| = %.3e (the page's own "
          "panel invariant; q is a probability or the solve refuses, never renormalised)"
          % sol5.conservation)

    # (b) INFORMATION. The sweep, with OPEN as the control arm.
    rows = [measure_information(g) for g in (OPEN, 5, 3, 1)]
    print("\n[MEASURED] N=200 instances x R=20 Bernoulli draws from the exact q, "
          "200-permutation label-shuffle null, G=9, 4 noise walls, seed=0")
    print("%8s %9s %17s %7s %7s %7s %9s %10s  outcome freq"
          % ("gap", "I(X;Y)", "shuffle null", "sigma", "H(pi)", "sd(q)", "PPL_marg", "PPL_bayes"))
    for r in rows:
        g = "OPEN" if r["gap"] == OPEN else "%g" % r["gap"]
        print("%8s %9.4f %8.4f+-%.4f %7.1f %7.4f %7.3f %9.4f %10.4f  %s"
              % (g, r["info"], r["null_mean"], r["null_sd"], r["sigmas"], r["h"],
                 r["q_sd"], r["ppl_marginal"], r["ppl_bayes"], r["outcome_freq"]))
    for r in rows:
        if not r["usable"]:
            print("[FINDING] gap=%s: I(X;Y)=%.4f does not clear its shuffle null by 3 sigma "
                  "(%.1f); THIS BED IS NOT USABLE at that gap."
                  % (r["gap"], r["info"], r["sigmas"]))
    infos = [r["info"] for r in rows]
    assert all(infos[i] < infos[i + 1] for i in range(len(infos) - 1)), (
        "I(X;Y) is not monotone in the narrowing gap: %s" % [round(v, 4) for v in infos])

    sealed = measure_information(0)
    print("[LIMIT] monotonicity is asserted over gap in [OPEN, 5, 3, 1] only, and gap=1 is "
          "the narrowest PASSABLE opening. Sealing it (gap=0) measures I=%.4f at %.0f sigma "
          "with sd(q)=%.3f -- higher, and worthless: the barrier is shut, q is exactly 0 or "
          "1, and the label is 'which side did you start on' rather than anything about the "
          "walk. Do not read the trend past gap=1."
          % (sealed["info"], sealed["sigmas"], sealed["q_sd"]))

    # (c) KAPPA, measured per instance, never assumed.
    print("\n[MEASURED] kappa = ||(I-Q)^{-1}||_inf via kappa_bed.kappa_of, 20 instances per "
          "row, G=9 gap=3 seed=0; q_drift is max |q(stay) - q(stay=0)| over those instances")
    ks = [measure_kappa(s) for s in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98)]
    print("%7s %12s %12s %12s %12s %10s" % ("stay", "kappa med", "kappa min", "kappa max",
                                            "q drift", "1/(1-stay)"))
    for r in ks:
        print("%7.2f %12.3f %12.3f %12.3f %12.3e %10.1f"
              % (r["stay"], r["kappa_med"], r["kappa_min"], r["kappa_max"], r["q_drift"],
                 1.0 / (1.0 - r["stay"])))
    med = [r["kappa_med"] for r in ks]
    assert all(med[i] < med[i + 1] for i in range(len(med) - 1)), (
        "kappa is not monotone in the stay knob: %s" % [round(v, 2) for v in med])
    assert med[-1] / med[0] > 20, "the stay knob does not span a useful kappa range"
    ratio_err = max(abs(med[i] / med[0] - 1.0 / (1.0 - ks[i]["stay"])) for i in range(len(ks)))
    assert max(r["q_drift"] for r in ks) < 1e-12, "the stay knob moved the labels"
    print("[MEASURED] kappa spans %.1f -> %.1f (%.0fx) and the ratio to kappa(stay=0) tracks "
          "1/(1-stay) to %.2e; labels move by at most %.2e over the whole sweep, so this knob "
          "buys conditioning and costs nothing."
          % (med[0], med[-1], med[-1] / med[0], ratio_err, max(r["q_drift"] for r in ks)))
    print("[MEASURED] kappa median %.1f at stay=0; %.1f%% of instances read above "
          "1/TELEPORT = %.0f. This is REPORTED, not an alarm: q is harmonic and "
          "invariant to it (q_drift above is at the float64 floor), so the bed's kappa "
          "is not a bound on what a model can represent. The teleport's real cost is "
          "measured on the do-arm, not here."
          % (ks[0]["kappa_med"], 100 * ks[0]["over_teleport_bound"], 1.0 / TELEPORT))

    # (d) DOES THE INTERVENTION BITE.
    print("\n[MEASURED] |q_do - q_obs| over EVERY blockable cell, one random start per "
          "instance, 60 instances per row (this is the demo's whole field)")
    print("%8s %7s %8s %8s %8s %8s %8s %10s %10s %9s"
          % ("gap", "n", "mean", "median", "p90", "p99", "max", "|dq|>0.01", "|dq|>0.05",
             "med best"))
    bites = [measure_bite(g) for g in (OPEN, 3, 1)]
    for r in bites:
        g = "OPEN" if r["gap"] == OPEN else "%g" % r["gap"]
        print("%8s %7d %8.4f %8.4f %8.4f %8.4f %8.4f %9.1f%% %9.1f%% %9.4f"
              % (g, r["n"], r["mean"], r["median"], r["p90"], r["p99"], r["max"],
                 100 * r["frac_01"], 100 * r["frac_05"], r["top_med"]))
    mid = bites[1]
    print("[FINDING] THE FIELD IS SPARSE, NOT FLAT, AND NOT STRONG. At gap=3 the median "
          "block moves q by %.4f and %.1f%% of uniformly drawn blocks move it by less than "
          "0.01 -- so a do-arm supervised on uniform candidates is mostly supervised on "
          "'nothing happened'. What is there is in the tail: the best single block moves q "
          "by %.4f in the median instance and %.4f at worst. Narrowing the gap from OPEN to "
          "1 RAISES information (%.4f -> %.4f nats) while LOWERING the median bite (%.4f -> "
          "%.4f): the two things this bed is for do not peak at the same knob setting."
          % (mid["median"], 100 * (1 - mid["frac_01"]), mid["top_med"], mid["max"],
             infos[0], infos[-1], bites[0]["median"], bites[2]["median"]))

    # (e) PLANTED NEGATIVES. Every check above must be SEEN to fail first.
    print("\n[NEGATIVE 1] a label-decoupled bed must read I(X;Y) at chance")
    good, bad = rows[2], measure_information(3, shuffle=True)
    print("    before: gap=3 intact    I=%.4f at %6.1f sigma  usable=%s"
          % (good["info"], good["sigmas"], good["usable"]))
    print("    after : gap=3 decoupled I=%.4f at %6.1f sigma  usable=%s"
          % (bad["info"], bad["sigmas"], bad["usable"]))
    assert not bad["usable"] and bad["sigmas"] < 3.0, (
        "the decoupled bed still reads %.1f sigma -- measure_information is not measuring "
        "what it claims" % bad["sigmas"])
    assert good["usable"], "the intact bed failed its own usability bar"
    print("    FIRED: the estimator that reports %.1f sigma on the intact bed reports "
          "%.1f on the decoupled one, so it is capable of saying 'nothing here'. The FIRST "
          "version of this negative permuted q across rows and read 81.5 sigma -- plug-in MI "
          "is invariant to row order, so that control could never have fired (recorded in "
          "measure_information's docstring)." % (good["sigmas"], bad["sigmas"]))

    print("[NEGATIVE 2] a grid with no HAZARD (q == 1 everywhere) must be caught")
    ok_sol = solve_grid(make_grid(random.Random(3), hazard=True))
    print("    before: with hazard  q_goal in [%.4f, %.4f], spread %.4f -> accepted"
          % (ok_sol.q[:, 0].min(), ok_sol.q[:, 0].max(), check_informative(ok_sol)))
    dead = solve_grid(make_grid(random.Random(3), hazard=False))
    print("    after : no hazard    q_goal in [%.4f, %.4f], spread %.3e"
          % (dead.q[:, 0].min(), dead.q[:, 0].max(),
             float(dead.q[:, 0].max() - dead.q[:, 0].min())))
    raised = False
    try:
        check_informative(dead)
    except DegenerateBedError as e:
        raised = True
        print("    FIRED: %s" % str(e)[:118])
    assert raised, "check_informative accepted a constant committor field"
    assert abs(float(dead.q[:, 0].min()) - 1.0) < 1e-12, (
        "the no-hazard grid did not give q == 1; min q_goal = %.16f" % dead.q[:, 0].min())

    print("[NEGATIVE 3] a block on an ALREADY-WALLED cell must give delta exactly 0.0")
    cells = make_grid(random.Random(5))
    sol = solve_grid(cells)
    start = sol.tlist[len(sol.tlist) // 2]
    base = float(sol.q_at(start)[0])
    wall_b = int(np.flatnonzero(cells.ravel() == WALL)[0])
    live_b = next(b for b in sol.tlist
                  if b != start and abs(float(solve_grid(_blocked(cells, b)).q_at(start)[0])
                                        - base) > 1e-6)
    d_live = float(solve_grid(_blocked(cells, live_b)).q_at(start)[0]) - base
    d_wall = float(solve_grid(_blocked(cells, wall_b)).q_at(start)[0]) - base
    print("    before: block a FREE cell (%s)  delta = %+.6f" % (cell_name(9, live_b), d_live))
    print("    after : block a WALL cell (%s)  delta = %+.6e" % (cell_name(9, wall_b), d_wall))
    assert d_wall == 0.0, "null intervention moved q by %.3e -- it must be bitwise 0" % d_wall
    assert abs(d_live) > 1e-6, "the live block did nothing, so the 0.0 above proves nothing"
    print("    FIRED: the same code path that moves q by %+.6f returns exactly 0.0 for the "
          "null intervention." % d_live)

    print("[NEGATIVE 4] operator.committor() must be SEEN returning the wrong answer here")
    from ceqjepa import operator as _op
    Q, R = sol.Q, sol.R
    n = Q.shape[0]
    P = np.zeros((n + 2, n + 2))
    P[:n, :n], P[:n, n:] = Q, R
    P[n, n] = P[n + 1, n + 1] = 1.0
    q_tri = _op.committor(torch.from_numpy(P), [n, n + 1]).numpy()[:n]
    print("    general solve (kappa_bed.committor):  row sums in [%.6f, %.6f]"
          % (sol.q.sum(1).min(), sol.q.sum(1).max()))
    print("    operator.committor (solve_triangular): row sums in [%.6f, %.6f], worst "
          "coordinate error %.4e"
          % (q_tri.sum(1).min(), q_tri.sum(1).max(), np.abs(q_tri - sol.q).max()))
    assert np.abs(q_tri - sol.q).max() > 0.1, (
        "operator.committor now agrees on a non-triangular chain -- re-derive which solver "
        "this bed should use before trusting either")
    print("    FIRED: it returns a finite, non-NaN, WRONG q with nothing raised. That is why "
          "ground truth here comes from kappa_bed.committor().")

    # (f) the contract: train.py's own consumers take this bed unchanged.
    from ceqjepa.train import draw, heldout_split, bed_groups
    ds = build_intervention_dataset(n_instances=40, seed=0, m_candidates=8)
    bed = GridBed(ds, 8)
    x, x_nx, q_star, v_idx, moves, do_tgt, do_mask = draw(bed, torch.Generator().manual_seed(0), 16)
    assert bed_groups(bed) is not None, "train.bed_groups does not recognise GridBed"
    tr, ho, note = heldout_split(bed, 0.25, 0)
    assert x.shape == (16, bed.x_dim) and do_tgt.shape == (16, 8, bed.nA)
    assert do_mask.shape == (16, 8) and moves.shape == (16, 8, 2)
    assert torch.allclose(q_star.sum(1), torch.ones(16), atol=1e-6)
    obs = np.stack([s.obs_outcome for s in ds])
    do_all = np.concatenate([s.do_outcome_mean for s in ds], axis=0)
    print("\n[MEASURED] observational outcome distribution (%d instances, EXACT committor, "
          "not a rollout mean): %s" % (len(ds), outcome_distribution(obs)))
    print("[MEASURED] interventional outcome distribution (%d blocked-cell solves, R=%d "
          "rollouts each): %s"
          % (do_all.shape[0], ds[0].do_outcome_R, outcome_distribution(do_all)))
    print("[MEASURED] contract: train.py's draw() / bed_groups() / heldout_split() take "
          "GridBed with NO change to train.py. x %s, x_nx %s, q_star %s, moves %s, do_tgt %s, "
          "do_mask %s; nA=%d x_dim=%d pad_rate=%.4f; %d train / %d held-out rows."
          % (tuple(x.shape), tuple(x_nx.shape), tuple(q_star.shape), tuple(moves.shape),
             tuple(do_tgt.shape), tuple(do_mask.shape), bed.nA, bed.x_dim, bed.pad_rate,
             len(tr.x), len(ho.x)))
    print("[MEASURED] split note: %s" % note)

    # The PAD/MASK path. On the default geometry no candidate ever strands the start
    # cell (0/300 rows at n_instances=300), so short candidate lists are unexercised
    # unless forced -- and an untriggered branch is an untested one. Force it.
    short = sum(len(s.candidate_ucis) < 8 for s in ds)
    wide = GridBed(ds, 12)
    assert wide.do_mask[:, :8].all() and not wide.do_mask[:, 8:].any(), "mask is wrong"
    assert torch.equal(wide.do_tgt[:, 8], wide.do_tgt[:, 0]), "pad did not repeat candidate 0"
    print("[MEASURED] pad/mask path: %d/%d rows had a candidate stranded out on the default "
          "geometry, so it is FORCED here -- GridBed(ds, m=12) over 8-candidate rows gives "
          "pad_rate %.4f, mask True on 0:8 and False on 8:12, padded targets bitwise equal to "
          "candidate 0. Refusal is real (delta_field returns NaN, not 0, where a block strands "
          "the start) but the default grids are not fragile enough to trigger it."
          % (short, len(ds), wide.pad_rate))

    best = rows[2]
    print("\n[RECOMMEND] gap=3, stay=0.0: I(X;Y)=%.4f nats (%.0f sigma over its null), "
          "H(pi)=%.4f, PPL %.4f -> Bayes-optimal %.4f (chance %.1f); kappa median %.1f; "
          "%.1f%% of blocks move q by more than 0.05. gap=1 buys %.4f nats more information "
          "and costs %.0f%% of that bite -- pick by which arm is being measured."
          % (best["info"], best["sigmas"], best["h"], best["ppl_marginal"],
             best["ppl_bayes"], best["ppl_chance"], ks[0]["kappa_med"],
             100 * bites[1]["frac_05"], infos[-1] - infos[2],
             100 * (1 - bites[2]["frac_05"] / bites[1]["frac_05"])))


if __name__ == "__main__":
    demo()
