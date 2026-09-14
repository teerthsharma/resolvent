"""THE NORTH-STAR RACE: one solve against depth-matched masked-attention stacks.

THE CLAIM UNDER TEST, in its first falsifiable form. The committor toward an
unseen goal is ONE triangular solve. A d-layer stack composes d hops. Where B'
is further than d hops away the d-hop truncation is EXACTLY zero while q is not,
and that fraction is read off the enumeration BEFORE any model is trained. So:
a single solve reaches goals a depth-matched stack cannot, and the fraction is
computable in advance.

THE ASSUMPTION, CARRIED AND NOT BURIED. The hop fractions bound a stack that
contributes AT MOST ONE HOP OF THIS OPERATOR PER LAYER. That is not a theorem
about attention in general -- a dense-attention transformer over the 46,138
orbits reaches every state in one layer and the bound says nothing about it.
Here the assumption is made LITERAL rather than assumed: every baseline layer is
masked to the successor sets of P, so one layer is one hop by construction, and
`receptive_field_violation` asserts it on the network rather than in prose. The
arm that BREAKS the assumption on purpose is `attn-2+global`, which gives every
node a global pooled token and therefore unbounded range in two layers. If that
arm -- or any arm -- beats the bound, the assumption was wrong, and that is the
finding.

WHAT THE SOLVE ARM IS, stated so it is not read as a win on points. Handed the
same (B', C) indicator every other arm gets, one solve of (I - P_II)x = P_IB 1
returns the label exactly, so its R2 is 1.0 BY CONSTRUCTION and no arm can beat
it. The falsifiable content is therefore NOT "the solve wins" but "the stacks
lose, by the predicted amount". The pre-registered loss condition for the
one-solve claim is stated in `LOSS_CONDITION` and checked in `race()`: if a
stack at depth <= 2 reaches within-draw R2 >= 0.90 on unseen goals, depth was
never the binding constraint and the strongest remaining argument for the
operator is retired.

L-NULL. What VARIES is the READER: the solve, the exact d-hop truncations, and
the trained d-layer stacks. What is PINNED is the 46,137-orbit chain, the drawn
goals, their exact committors, the train/test split, the three ceilings, the
input columns, the optimiser, the seed, and the parameter budget.

THE RESOLVENTS ARE NOT INPUTS. `q1` and `q2` are hops. Handing them to a
d-layer stack makes it a (d+2)-layer stack with no trace in the depth column, so
`INPUT_NAMES` excludes them for every arm.

RED FIRST. `tests/curvature/test_depth_race.py` was written and run before this
file existed. Verbatim, `python -m pytest tests/curvature/test_depth_race.py -x -q`
at 62cb8e0 on WIN-16QAL06O9GB:

    ImportError while importing test module 'tests/curvature/test_depth_race.py'.
    Hint: make sure your test modules/packages have valid Python names.
    ...
    E   ImportError: cannot import name 'depth_race' from 'ceqjepa'
        (ceqjepa/__init__.py)
    ERROR tests/curvature/test_depth_race.py
    !!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!
    1 error in 2.68s

MEASURED. `python -m ceqjepa.depth_race --draws 100 --train 20 --epochs 25` at
commit 62cb8e0 on WIN-16QAL06O9GB (python 3.11.9, numpy 2.4.6, scipy 1.17.1,
torch 2.14.0+cpu), CPU only, SEED = 0, FLOOR = 0.50, 2385.7s, 100 unseen goals,
20 training goals, 3,371,479 scored rows. WITHIN-DRAW R2 on the unseen goals,
which is the column that matters: a pooled R2 on this bed is 0.5277 free
(`goal_family.RESULTS["marginal_pooled"]`).

    arm             params  width   within   bound   TRAINfit
    solve                2      -   1.0000  0.4792     1.0000   one solve
    trunc-1              2      -  -0.4598  0.4792    -0.4726   exact 1 hop
    trunc-2              2      -  -0.1064  0.6847    -0.0157   exact 2 hops
    trunc-4              2      -   0.2806  0.8534     0.4153   exact 4 hops
    trunc-8              2      -   0.6314  0.9331     0.7453   exact 8 hops
    attn-1            9676     45   0.0454  0.4792     0.6231   trained
    attn-2            9505     32   0.1791  0.6847     0.7428   trained
    attn-4            9684     23   0.3657  0.8534     0.8606   trained
    attn-8            9425     16   0.3826  0.9331     0.8471   trained
    attn-2+global     9505     32   0.1700  0.6847     0.7408   range unbounded

    ceilings   marginal 0.0000   lookup 0.2336   best-heuristic 0.0852
    the same three on the WORST draw: 0 / 0.4886 / 0.3298, headroom 0.5114

THE ONE-SOLVE CLAIM SURVIVES ITS PRE-REGISTERED LOSS CONDITION. The strongest
depth-2 arm reads 0.1791 against a retirement line of 0.90, and every trained
stack at every depth stays below the solve by more than 0.6.

AND THE HALF OF THE CLAIM THAT DID NOT SURVIVE, which is the more useful result.
"The fraction is computable in advance" is FALSE as a prediction and true only as
a bound. `bound` above is that advance quantity done properly -- not the row
fraction but the VARIANCE share carried by rows further than d hops, which is
the most a d-limited arm can lose. It is never violated, at any depth, by any
arm, so the one-hop-per-layer assumption is not refuted. But it over-predicts
every measured arm by a wide margin: 0.43 to 0.55 for the trained stacks and
0.30 to 0.94 for the EXACT truncations, which have no optimiser to blame. Depth
is therefore not what binds these baselines first, and the advance number cannot
be quoted as a forecast of the gap. Read as row fractions the advance quantity
is worse still: 79.56% of rows sit beyond 1 hop but they carry 52.08% of the
variance, and at 8 hops 2.30% of rows carry 6.69%.

THE ARM THAT WAS SUPPOSED TO BREAK IT, AND DID NOT. `attn-2+global` has the same
depth and the same 9,505 parameters as `attn-2` and unbounded range -- the
receptive-field probe moves its output from 5 hops away, where the masked arm
reads exactly 0. It reads 0.1700 against `attn-2`s 0.1791: removing the range
limit bought -0.0091. So on this bed the deficit is NOT mainly the receptive
field. Whatever the trained stacks are missing, reaching further does not supply
it, and the depth story must not be told as a reach story.

CAPACITY IS NOT WHAT THE STACKS ARE SHORT OF, and this is the load-bearing
control, because 9,500 parameters invites "the baseline was too small". The
SAME depth-2 arm at 94,235 parameters -- 9.9x the matched budget, width 106,
every other thing pinned -- reads within-draw 0.1433 on the unseen goals against
the matched arm's 0.1791. Ten times the capacity made it WORSE, and its own
training-goal fit fell too, 0.6785 against 0.7428: at 20 training goals the extra
width buys memorisation of those 20 and nothing that transfers.
`dr.race(n_test=100, n_train=20, epochs=25, seed=0, budget=94235, depths=(2,),
with_global=False)`, 62cb8e0, WIN-16QAL06O9GB, 1193.3s.

RUN THE TWO SUITES AS TWO PROCESSES. `pytest test_goal_family.py
test_depth_race.py` in ONE process reached 25 of 27 tests and then died with no
summary and exit 1 on this box (15.7 GB, 5.7 GB free, other sessions resident):
the module-scoped fixtures of both files plus the autograd graph over 609,873
edges do not fit together. Separately they are green -- 16 passed in 129.62s and
11 passed in 582.56s, both exit 0, 62cb8e0, WIN-16QAL06O9GB.

THE DEPTH ARGUMENT IS RETIRED, and the sweep that retired it is below. The
hole named in the last paragraph of the previous revision -- "the transfer
failure may be a few-goal failure rather than a depth failure" -- was run, and
few-goal transfer is not exonerated: it was most of the deficit. Varying ONLY
the training-goal count at depth 2 and 9,505 parameters, a two-layer stack
passes the eight-layer stack.

`scratchpad/gsweep/point.py N E OUT D`, which is
`dr.race(n_test=100, n_train=N, epochs=E, seed=0, budget=9500, depths=(D,),
with_global=False, pool=80)`, at 4287508 plus the working-tree `pool`/`ckpt`/
`cache` arguments added here -- and identically at 9336cb0, which lands during
this sweep and touches neither this file, `goal_family` nor `chess_steps`
(`git diff --stat 4287508 9336cb0 -- ceqjepa/depth_race.py ceqjepa/goal_family.py
ceqjepa/chess_steps.py` is empty) -- on WIN-16QAL06O9GB (python 3.11.9, numpy 2.4.6,
scipy 1.17.1, torch 2.14.0+cpu), CPU only, SEED = 0, FLOOR = 0.50. All six
points are scored on ONE pinned test set: digest 13d863616a03b028, 3,384,017
rows, asserted equal across every point by `GOAL_SWEEP` and
`test_the_goal_sweep_retires_the_depth_argument`.

    point     arm     depth goals epochs steps  within  TRAINfit  seconds
    n20       attn-2      2    20     25    500  0.0555   0.7428    738.3
    n40       attn-2      2    40     25   1000  0.3633   0.6940    596.1
    n80       attn-2      2    80     25   2000  0.4542   0.6834   1077.2
    s20e100   attn-2      2    20    100   2000  0.3005   0.8914   2415.8
    s40e50    attn-2      2    40     50   2000  0.4858   0.7568   1066.1
    d8n20     attn-8      8    20     25    500  0.3324   0.8471    696.3

THE LINE IT CROSSES. Pre-registered before the first run: depth 2 reaching
0.3826, the published within-draw R2 of the EIGHT-layer stack, retires the
depth argument. It reads 0.4542 at 80 goals and 0.4858 at 40 goals with the
same 2,000 steps. Re-derived rather than copied, the eight-layer stack on THESE
draws reads 0.3324, so the crossing is 0.1218 on the published line's own test
set and 0.1534 on this one. Depth 2 with a better training recipe is a better
reader of unseen goals than depth 8 with the published one, at the same
parameter budget and the same receptive-field wiring.

WHAT THE DEPTH COLUMN WAS ACTUALLY MEASURING. Every arm of the published ladder
saw 500 gradient steps. Hold the goals at 20 and raise the steps to 2,000 and
depth 2 moves 0.0555 -> 0.3005, which is the whole distance to depth 8's 0.3324
bought with no depth at all. Hold the steps at 2,000 and raise the goals 20 ->
40 -> 80 and it moves 0.3005 -> 0.4858 -> 0.4542. So the ladder 0.0454 / 0.1791
/ 0.3657 / 0.3826 was a ladder of UNDERTRAINED readers, and the advance bound's
0.43-to-0.55 over-prediction was measured against an optimiser that had not
finished, not against a depth limit.

THE INTERNAL NULL, which is why this is not a test-set effect. The same three
training sets refit the two-parameter arms in the same runs. Over 20 -> 80
goals the trained stack moves +0.3987 while the exact truncations move +0.0076,
+0.0093, +0.0074 and +0.0043 and the solve stays at 1.0000. Paired over the
identical 100 draws the stack improves on 80 of them (sign test p = 1.12e-9,
paired per-draw mean +0.8407 +/- 0.4036, median +0.2778); the per-draw
distribution is heavy-tailed on small-variance draws, so the aggregate
within-draw R2 is the pre-registered primary and the sign count is the robust
second reading.

THE SIGNATURE THAT SAYS TRANSFER, registered before the run as the thing that
would distinguish transfer from underfitting: the unseen-goal fit RISES while
the training-goal fit FALLS, 0.7428 -> 0.6940 -> 0.6834 at pinned epochs. It
did, monotonically, and the 9.9x-capacity control's falling train fit now reads
as the same mechanism seen from the other side.

WHAT REPLACES IT. The dead thing was "one solve reaches goals a depth-matched
stack cannot, and the gap is computable in advance". The surviving route to the
same goal is not depth and not the advance bound: it is the SLOPE. At 2,000
steps the stack is at 0.4858 against the solve's 1.0000 with 0.5034 of headroom
on the worst draw, and the goal-count increments are +0.3078 then +0.0909 per
doubling -- a ratio of 0.295, which extrapolates to 0.4810 at 160 goals and an
asymptote near 0.4923 if the decay holds, or 0.5451 at 160 if the last slope
holds. That asymptote, measured rather than assumed, is the sharper claim,
because it is about what the stack CANNOT reach with unlimited goals rather
than about how many hops it has. The next point costs more than one run: the
floor is applied against the pool, so a 160-goal point needs pool=160 and
re-draws the test set, which re-runs the whole sweep, about 2 hours on a quiet
box (4,000 steps for the point itself, ~45 minutes).

RESUME IS NOT A NUMBER. `ckpt` exists because this box is shared: with another
session's suite fleet resident, torch segfaults on a failed 78 MB edge-tensor
allocation, and the 2,000-step points died at steps ~450, ~900 and ~200 before
checkpointing was added. The resumed trajectory is the same trajectory -- the
step order is drawn from the seed, the init is seeded, Adam's whole state is
saved -- and the run shows it: the loss at step 200 of n80 reads 0.00832 in the
13:44 process that died and in all five later processes that resumed it.

THE HOLE THIS DOES NOT CLOSE, collected here and nowhere else. The stacks are
graph-masked message passers, not dense transformers; a dense-attention model
over the 46,138 orbits is not depth-matched in hops and is not run here. Twenty
training goals is a small training set and the transfer failure may be a
few-goal failure rather than a depth failure -- the capacity control above
points that way, and separating the two needs a training-goal sweep that is not
run here. The solve arm is handed the exact operator, so its 1.0000 is a
construction and not a win on points; what is measured is the stacks distance
from it. The goal sweep pins its test set by flooring every test draw against
the pool of 80, which makes those draws HARDER than the published run's -- the
identical n20 network, TRAINfit 0.7428 to the digit, reads 0.0555 here against
0.1791 there -- so numbers may be compared within this sweep and with d8n20,
never across to the published table except through d8n20. Nothing here says
where the stack SATURATES: 160 goals was not run, the extrapolation is two
increments of arithmetic and not a measurement, and s40e50 reading 0.4858 above
n80's 0.4542 means the goal-count and step-count axes are not separable at this
resolution. And the one-solve claim's own pre-registered loss condition, a
depth<=2 arm at 0.90, is still not met: 0.4858 is the best reader measured.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from ceqjepa import goal_family as gf

SEED = 0
PARAM_BUDGET = 9500          #: every stack arm is fitted to this, +/- 10%.
DEPTHS = (1, 2, 4, 8)
EPOCHS = 25
LR = 3e-3

#: Pre-registered. A stack this shallow reaching this within-draw R2 on unseen
#: goals retires the one-solve claim.
LOSS_CONDITION = dict(depth=2, within=0.90)

#: `python -m ceqjepa.depth_race --draws 100 --train 20 --epochs 25`, 62cb8e0,
#: WIN-16QAL06O9GB. The two-parameter half is re-derived, not copied, by
#: test_depth_race.py::test_the_exact_arms_reproduce_at_the_headline_size.
RESULTS: dict = {
    "n_rows": 3371479,
    "hop_fraction": {1: 0.7956, 2: 0.4817, 4: 0.1551, 8: 0.0230},
    "far_variance_share": {1: 0.5208, 2: 0.3153, 4: 0.1466, 8: 0.0669},
    "bound_within": {1: 0.4792, 2: 0.6847, 4: 0.8534, 8: 0.9331},
    "trunc_within": {1: -0.4598, 2: -0.1064, 4: 0.2806, 8: 0.6314},
    "attn_within": {1: 0.0454, 2: 0.1791, 4: 0.3657, 8: 0.3826},
    "attn_params": {1: 9676, 2: 9505, 4: 9684, 8: 9425},
    "attn_train_within": {1: 0.6231, 2: 0.7428, 4: 0.8606, 8: 0.8471},
    "global_within": 0.1700, "global_params": 9505,
    "generous_attn2_params": 94235, "generous_attn2_within": 0.1433,
    "generous_attn2_train_within": 0.6785,
    "ceilings": {"marginal": 0.0, "lookup": 0.2336, "heuristic": 0.0852},
    "ceiling_lookup_worst": 0.4886, "ceiling_heuristic_worst": 0.3298,
    "headroom_worst": 0.5114,
}

GOAL_SWEEP: dict = {
    "retirement_line_published": 0.3826,
    "retirement_line_rederived": 0.3324,
    "depth_argument_retired": True,
    "ceilings": {'marginal': 0.0, 'lookup': 0.26357708661014584, 'heuristic': 0.1103769080909132},
    "points": {
        "n20": {'arm': 'attn-2', 'depth': 2, 'goals': 20, 'epochs': 25, 'steps': 500, 'params': 9505, 'width': 32, 'pool': 80, 'n_rows': 3384017, 'test_digest': '13d863616a03b028', 'within': 0.0555, 'train_within': 0.7428, 'pooled': 0.6008, 'bound_within': 0.711, 'trunc_within': {1: -0.672, 2: -0.2643, 4: 0.1855, 8: 0.5979}, 'seconds': 738.3},
        "n40": {'arm': 'attn-2', 'depth': 2, 'goals': 40, 'epochs': 25, 'steps': 1000, 'params': 9505, 'width': 32, 'pool': 80, 'n_rows': 3384017, 'test_digest': '13d863616a03b028', 'within': 0.3633, 'train_within': 0.694, 'pooled': 0.7309, 'bound_within': 0.711, 'trunc_within': {1: -0.7759, 2: -0.3323, 4: 0.1537, 8: 0.5902}, 'seconds': 596.1},
        "n80": {'arm': 'attn-2', 'depth': 2, 'goals': 80, 'epochs': 25, 'steps': 2000, 'params': 9505, 'width': 32, 'pool': 80, 'n_rows': 3384017, 'test_digest': '13d863616a03b028', 'within': 0.4542, 'train_within': 0.6834, 'pooled': 0.7693, 'bound_within': 0.711, 'trunc_within': {1: -0.6644, 2: -0.255, 4: 0.1929, 8: 0.6022}, 'seconds': 1077.2},
        "s20e100": {'arm': 'attn-2', 'depth': 2, 'goals': 20, 'epochs': 100, 'steps': 2000, 'params': 9505, 'width': 32, 'pool': 80, 'n_rows': 3384017, 'test_digest': '13d863616a03b028', 'within': 0.3005, 'train_within': 0.8914, 'pooled': 0.7043, 'bound_within': 0.711, 'trunc_within': {1: -0.672, 2: -0.2643, 4: 0.1855, 8: 0.5979}, 'seconds': 2415.8},
        "s40e50": {'arm': 'attn-2', 'depth': 2, 'goals': 40, 'epochs': 50, 'steps': 2000, 'params': 9505, 'width': 32, 'pool': 80, 'n_rows': 3384017, 'test_digest': '13d863616a03b028', 'within': 0.4858, 'train_within': 0.7568, 'pooled': 0.7827, 'bound_within': 0.711, 'trunc_within': {1: -0.7759, 2: -0.3323, 4: 0.1537, 8: 0.5902}, 'seconds': 1066.1},
        "d8n20": {'arm': 'attn-8', 'depth': 8, 'goals': 20, 'epochs': 25, 'steps': 500, 'params': 9425, 'width': 16, 'pool': 80, 'n_rows': 3384017, 'test_digest': '13d863616a03b028', 'within': 0.3324, 'train_within': 0.8471, 'pooled': 0.7178, 'bound_within': 0.9095, 'trunc_within': {1: -0.672, 2: -0.2643, 4: 0.1855, 8: 0.5979}, 'seconds': 696.3},
    },
}

STATE_NAMES = ["wk_f", "wk_r", "wq_f", "wq_r", "bk_f", "bk_r", "turn",
               "d_wk_bk", "d_wq_bk", "d_wk_wq", "e_wk", "e_wq", "e_bk",
               "corner_bk", "deg", "dtm_clip", "is_win", "q_mate"]
GOAL_NAMES = ["d_bk_to_B", "d_wk_to_B", "d_wq_to_B", "d_bk_to_C",
              "frac_B", "frac_C", "in_B", "in_C", "absorbing"]
INPUT_NAMES = STATE_NAMES + GOAL_NAMES


# ---------------------------------------------------------------------------
# THE INPUT EVERY ARM SEES
# ---------------------------------------------------------------------------

def _pad(v, N):
    out = np.zeros(N, np.float64)
    out[:v.size] = v
    return out


def node_features(B, C) -> np.ndarray:
    """One row per orbit STATE, not per interior state: a message-passing arm
    has to read the nodes it absorbs on."""
    Q, D = gf.quotient(), gf._derived()
    N, n = Q["n_states"], Q["n_orbits"]
    dtm = Q["dtm"]

    def near(piece, mask):
        sq = np.unique(D[piece][mask[:n]])
        if sq.size == 0:
            return np.full(N, 8.0)
        return _pad(gf._CHEB[:, sq].min(axis=1)[D[piece]].astype(np.float64), N)

    cols = [_pad(D[k].astype(np.float64), N) for k in
            ("wk_f", "wk_r", "wq_f", "wq_r", "bk_f", "bk_r", "turn",
             "d_wk_bk", "d_wq_bk", "d_wk_wq", "e_wk", "e_wq", "e_bk",
             "corner_bk")]
    cols += [_pad(Q["deg"], N),
             _pad(np.where(dtm >= 0, dtm, 25.0), N),
             _pad((dtm >= 0).astype(np.float64), N),
             _pad(Q["q_mate"], N)]
    cols += [near("bk", B), near("wk", B), near("wq", B), near("bk", C),
             np.full(N, B.sum() / B.size), np.full(N, C.sum() / C.size),
             B.astype(np.float64), C.astype(np.float64),
             Q["absorbing"].astype(np.float64)]
    X = np.column_stack(cols)
    assert X.shape[1] == len(INPUT_NAMES), (X.shape, len(INPUT_NAMES))
    return X


# ---------------------------------------------------------------------------
# THE BASELINE: ONE HOP OF P PER LAYER, WIRED
# ---------------------------------------------------------------------------

class HopAttention(nn.Module):
    """One masked self-attention layer over the successor sets of P.

    `src` attends over `dst`, so information moves ONE hop backwards along the
    move graph per layer. That is the assumption, made structural.
    """

    def __init__(self, h, use_global=False):
        super().__init__()
        self.v = nn.Linear(h, h, bias=False)
        self.a_src = nn.Parameter(torch.zeros(h))
        self.a_dst = nn.Parameter(torch.zeros(h))
        self.mlp = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h))
        self.norm = nn.LayerNorm(h)
        self.use_global = use_global
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)

    def forward(self, x, src, dst, N):
        v = self.v(x)
        e = torch.nn.functional.leaky_relu(
            (v[src] * self.a_src).sum(1) + (v[dst] * self.a_dst).sum(1), 0.2)
        m = torch.full((N,), -1e30, dtype=e.dtype).scatter_reduce_(
            0, src, e, reduce="amax", include_self=False)
        if self.use_global:
            g = v.mean(0)
            eg = torch.nn.functional.leaky_relu(
                (v * self.a_src).sum(1) + (g * self.a_dst).sum(), 0.2)
            # the 156 absorbing orbits have NO successors, so their row max is
            # the -1e30 sentinel; subtracting it from the global logit overflows
            # to inf and poisons the whole forward pass. The softmax reference
            # has to cover the global term, not just the edges.
            m = torch.maximum(m, eg)
        ex = torch.exp(e - m[src])
        den = torch.zeros(N, dtype=e.dtype).index_add_(0, src, ex)
        if self.use_global:
            exg = torch.exp(eg - m)
            den = den + exg
        msg = torch.zeros_like(v).index_add_(
            0, src, (ex / den[src].clamp_min(1e-30)).unsqueeze(1) * v[dst])
        if self.use_global:
            msg = msg + (exg / den.clamp_min(1e-30)).unsqueeze(1) * g
        return self.norm(x + self.mlp(torch.cat([x, msg], 1)))


class Stack(nn.Module):
    def __init__(self, depth, h, f, use_global=False):
        super().__init__()
        self.depth, self.h = depth, h
        self.enc = nn.Linear(f, h)
        self.layers = nn.ModuleList(
            [HopAttention(h, use_global) for _ in range(depth)])
        self.head = nn.Linear(h, 1)

    def forward(self, X, src, dst):
        x = self.enc(X)
        for L in self.layers:
            x = L(x, src, dst, X.shape[0])
        return torch.sigmoid(self.head(x)).squeeze(1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def width_for(depth, f, budget=PARAM_BUDGET, use_global=False):
    """The width that puts this depth on the parameter budget. Depth is the
    axis of the race; width is the confound, so it is solved for, not chosen."""
    best, best_h = None, None
    for h in range(4, 400):
        p = n_params(Stack(depth, h, f, use_global))
        if best is None or abs(p - budget) < abs(best - budget):
            best, best_h = p, h
        if p > budget * 1.5:
            break
    return best_h, best


# ---------------------------------------------------------------------------
# THE ASSUMPTION, ASSERTED ON THE NETWORK
# ---------------------------------------------------------------------------

def receptive_field_violation(depth, seed=SEED, probe_hop=None, n_probe=64,
                             use_global=False):
    """Max |change in a d-layer arm's output| from perturbing nodes exactly
    `probe_hop` hops downstream. Must be 0.0 at probe_hop = depth + 1 for a
    masked arm, and must be NON-ZERO for the global-token arm at any probe --
    which is the whole point of carrying that arm."""
    torch.manual_seed(seed)
    Q = gf.quotient()
    P, N = Q["P"], Q["n_states"]
    src = torch.from_numpy(np.repeat(np.arange(N), np.diff(P.indptr)))
    dst = torch.from_numpy(P.indices.astype(np.int64))
    f = len(INPUT_NAMES)
    net = Stack(depth, 12, f, use_global)
    rng = np.random.default_rng(seed)
    probe = probe_hop if probe_hop is not None else depth + 1

    deg = np.diff(P.indptr)
    roots = rng.choice(np.flatnonzero(deg > 0), size=n_probe, replace=False)
    reach = np.zeros(N, bool)
    reach[roots] = True
    ring = reach.copy()
    for _ in range(probe):
        nxt = np.zeros(N, bool)
        nxt[P[ring].indices] = True
        ring = nxt & ~reach
        reach |= ring
    if not ring.any():
        raise AssertionError("no node exactly %d hops out" % probe)

    X = torch.from_numpy(rng.standard_normal((N, f))).float()
    with torch.no_grad():
        y0 = net(X, src, dst)
        X2 = X.clone()
        X2[torch.from_numpy(np.flatnonzero(ring))] += 7.0
        y1 = net(X2, src, dst)
    return float((y1 - y0)[torch.from_numpy(roots)].abs().max())


# ---------------------------------------------------------------------------
# THE PINNED GOALS, THE PINNED CEILINGS
# ---------------------------------------------------------------------------

def draws(n_test, n_train, seed, floor, pool=None, cache=None):
    """The SAME draw stream `goal_family.report` uses: training goals first,
    then floored test goals. Pinned.

    `pool` is what makes a training-goal SWEEP possible at all. Without it the
    unseen goals move with `n_train` twice over -- the two draws share one rng
    stream, and `gf.draw_test_goal` rejects a candidate against the training
    table, which gets stricter as the table grows -- so two points of a sweep
    would be scored on two different test sets of two different hardnesses.
    With `pool` set to the LARGEST training count in the sweep, `pool` training
    goals are drawn, every test draw is floored against ALL of them, and the
    first `n_train` are handed back for fitting. Nested training sets, one
    pinned test set, and "unseen" meaning the same thing at every point.
    """
    pool = n_train if pool is None else max(pool, n_train)
    # THE DRAWS ARE A PURE FUNCTION of (n_test, pool, seed, floor), so on a box
    # that kills this process mid-run they are worth 170s of solves exactly
    # once. The cache is keyed on all four and asserted against the digest it
    # was written with; it is off unless a path is passed.
    ck = (None if cache is None else
          "%s.draws.%d_%d_%d_%s.pkl" % (cache, n_test, pool, seed, floor))
    if ck is not None and os.path.exists(ck):
        with open(ck, "rb") as fh:
            train, test, dg = pickle.load(fh)
        assert digest(test) == dg, "the cached draws do not match their digest"
        assert len(train) == pool and len(test) == n_test, (len(train), len(test))
        return train[:n_train], test
    Q = gf.quotient()
    P, absorbing = Q["P"], Q["absorbing"]
    rng = np.random.default_rng(seed)
    train = []
    for _ in range(pool):
        B, C, interior = gf.draw_goal(rng, absorbing)
        q, res, _ = gf.committor(P, B, C, interior)
        assert res < 1e-9, res
        train.append((B, C, interior, q))
    train_q = [d[3] for d in train]
    test = []
    for _ in range(n_test):
        B, C, interior, q, res, _, ceil, _ = gf.draw_test_goal(
            rng, absorbing, P, train_q, floor)
        assert res < 1e-9, res
        test.append((B, C, interior, q, ceil))
    if ck is not None:
        with open(ck + ".tmp", "wb") as fh:
            pickle.dump((train, test, digest(test)), fh, protocol=4)
        os.replace(ck + ".tmp", ck)
    return train[:n_train], test


def digest(items) -> str:
    """SHA-256 over the packed (B, C) masks, so "the same goals" is asserted on
    the goals themselves and not on a seed that was supposed to imply them."""
    h = hashlib.sha256()
    for it in items:
        h.update(np.packbits(it[0]).tobytes())
        h.update(np.packbits(it[1]).tobytes())
    return h.hexdigest()[:16]


def ceilings(train, test):
    """The three the bed already computes, on THESE test draws."""
    Q = gf.quotient()
    N = Q["n_states"]
    qbar = np.zeros(N)
    for B, C, interior, q, _ in test:
        qbar += np.where(interior, q, np.where(B, 1.0, 0.0))
    qbar /= len(test)
    look, heur = [], []
    for B, C, interior, q, ceil in test:
        y = q[interior]
        ss = float(((y - y.mean()) ** 2).sum())
        look.append(ceil[0])
        h = qbar[interior]
        c = np.corrcoef(h, y)[0, 1]
        heur.append(max(1.0 - float(((y - h) ** 2).sum()) / ss,
                        0.0 if not np.isfinite(c) else float(c ** 2)))
    return (dict(marginal=0.0, lookup=float(np.mean(look)),
                 heuristic=float(np.mean(heur))), np.array(look), np.array(heur))


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

class Score:
    """Pooled and within-draw R2, plus the same restricted to the rows the hop
    bound says a depth-`d` arm cannot reach."""

    def __init__(self, depth):
        self.depth = depth
        self.n = self.sy = self.sy2 = self.ss = self.sse = 0.0
        self.fss = self.fsse = self.fshare = 0.0
        self.per = []

    def add(self, y, p, dist):
        ss = float(((y - y.mean()) ** 2).sum())
        sse = float(((y - p) ** 2).sum())
        self.n += y.size
        self.sy += float(y.sum())
        self.sy2 += float(y @ y)
        self.ss += ss
        self.sse += sse
        self.per.append(1.0 - sse / ss)
        far = (dist > self.depth) | (dist < 0)
        # THE BOUND, COMPUTED IN ADVANCE. An arm blind to the rows beyond its
        # own depth can at best answer the draw's mean there, so its within-draw
        # R2 cannot exceed 1 - fshare/ss. This is the number the hop fractions
        # are a proxy for, and it is a VARIANCE share, not a row count.
        self.fshare += float(((y[far] - y.mean()) ** 2).sum())
        if far.sum() > 2:
            yf = y[far]
            self.fss += float(((yf - yf.mean()) ** 2).sum())
            self.fsse += float(((yf - p[far]) ** 2).sum())

    def out(self, **kw):
        sstot = self.sy2 - self.sy ** 2 / self.n
        return dict(pooled=1.0 - self.sse / sstot,
                    within=1.0 - self.sse / self.ss,
                    far_within=(1.0 - self.fsse / self.fss if self.fss
                                else float("nan")),
                    far_variance_share=self.fshare / self.ss,
                    bound_within=1.0 - self.fshare / self.ss,
                    per_draw=np.array(self.per), depth=self.depth, **kw)


def _affine(pairs):
    """Least squares [v, 1] -> y over the training goals. Two parameters, and
    they are the ONLY parameters the solve and truncation arms have."""
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for v, y in pairs:
        Z = np.column_stack([v, np.ones(v.size)])
        A += Z.T @ Z
        b += Z.T @ y
    return np.linalg.solve(A + np.eye(2) * 1e-10 * np.trace(A), b)


# ---------------------------------------------------------------------------
# THE RACE
# ---------------------------------------------------------------------------

def race(n_test=100, n_train=20, epochs=EPOCHS, seed=SEED, floor=gf.FLOOR,
         verbose=False, train_stacks=True, budget=PARAM_BUDGET, depths=DEPTHS,
         with_global=True, pool=None, ckpt=None, cache=None) -> dict:
    """`train_stacks=False` scores only the two-parameter arms -- the solve and
    the exact d-hop truncations. Those are the arms the "computable in advance"
    claim rests on, they involve no optimiser and no randomness beyond the
    pinned draw stream, and they re-derive in minutes instead of an hour."""
    t0 = time.time()
    torch.manual_seed(seed)
    Q = gf.quotient()
    P, N = Q["P"], Q["n_states"]
    f = len(INPUT_NAMES)
    src = torch.from_numpy(np.repeat(np.arange(N), np.diff(P.indptr)))
    dst = torch.from_numpy(P.indices.astype(np.int64))

    train, test = draws(n_test, n_train, seed, floor, pool=pool, cache=cache)
    ceil, look, heur = ceilings(train, test)
    if verbose:
        print("  draws done %.0fs" % (time.time() - t0), flush=True)

    # standardisation is fitted on the TRAINING goals only and then pinned
    Xtr = [node_features(B, C) for B, C, _, _ in train]
    mu = np.mean([X.mean(0) for X in Xtr], 0)
    sd = np.sqrt(np.mean([X.var(0) for X in Xtr], 0))
    sd[sd < 1e-9] = 1.0
    Ttr = [torch.from_numpy((X - mu) / sd).float() for X in Xtr]
    ytr = [torch.from_numpy(q).float() for _, _, _, q in train]
    mtr = [torch.from_numpy(i).bool() for _, _, i, _ in train]

    # ---- the two-parameter arms: one solve, and the exact d-hop truncations -
    tr_trunc = [gf.truncated(P, B, C, i, depth=max(DEPTHS))
                for B, C, i, _ in train]
    affine = {"solve": _affine([(q[i], q[i]) for _, _, i, q in train])}
    for dpt in DEPTHS:
        affine["trunc-%d" % dpt] = _affine(
            [(t[dpt - 1], q[i]) for t, (_, _, i, q) in zip(tr_trunc, train)])

    # ---- the trained depth-matched stacks ----------------------------------
    cfg = ([("attn-%d" % d, d, False) for d in depths]
           + ([("attn-2+global", 2, True)] if with_global else [])
           ) if train_stacks else []
    nets, losses = {}, {}
    for name, dpt, glob in cfg:
        h, npar = width_for(dpt, f, budget=budget, use_global=glob)
        torch.manual_seed(seed)
        net = Stack(dpt, h, f, glob)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        order = np.random.default_rng(seed).permutation(
            np.tile(np.arange(n_train), epochs))
        curve = []
        # RESUME, AND WHY IT CHANGES NO NUMBER. `order` is drawn from the seed
        # and not from the process, the init is seeded, and Adam's whole state
        # is saved, so a resumed run walks the identical trajectory -- the
        # checkpoint is against a SHARED BOX, not against the mathematics. With
        # another agent's suite fleet resident this run segfaults on a failed
        # 78 MB edge-tensor allocation, twice at ~500 steps and once at ~900,
        # and 2,000 steps never land in one piece.
        done = 0
        cpath = None if ckpt is None else "%s.%s.pt" % (ckpt, name)
        if cpath is not None and os.path.exists(cpath):
            st = torch.load(cpath, weights_only=False)
            net.load_state_dict(st["net"])
            opt.load_state_dict(st["opt"])
            curve, done = st["curve"], st["step"]
            assert st["order_hash"] == int(order.sum()), "resumed a different run"
            if verbose:
                print("  %s RESUMED at step %d/%d" % (name, done, order.size),
                      flush=True)
        for step, gi in enumerate(order):
            if step < done:
                continue
            opt.zero_grad()
            p = net(Ttr[gi], src, dst)
            loss = ((p[mtr[gi]] - ytr[gi][mtr[gi]]) ** 2).mean()
            loss.backward()
            opt.step()
            curve.append(float(loss.item()))
            if verbose and step % 100 == 0:
                print("  %s h=%d step %d/%d loss %.5f %.0fs"
                      % (name, h, step, order.size, curve[-1], time.time() - t0),
                      flush=True)
            if cpath is not None and (step % 100 == 99 or step == order.size - 1):
                tmp = cpath + ".tmp"
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(),
                                curve=curve, step=step + 1,
                                order_hash=int(order.sum())), tmp)
                os.replace(tmp, cpath)
        nets[name] = (net, dpt, h, npar, glob)
        losses[name] = [float(np.mean(curve[:20])), float(np.mean(curve[-40:]))]

    # ---- ONE pass over the unseen goals, every arm scored on the same rows --
    names = ["solve"] + ["trunc-%d" % d for d in DEPTHS] + [c[0] for c in cfg]
    depth_of = dict({"solve": 1}, **{"trunc-%d" % d: d for d in DEPTHS})
    depth_of.update({c[0]: c[1] for c in cfg})
    sc = {n: Score(depth_of[n]) for n in names}
    sc_tr = {n: Score(depth_of[n]) for n in names}
    hop_far = {d: 0.0 for d in DEPTHS}
    n_rows = 0

    def score_into(bank, items, with_ceiling):
        nonlocal n_rows
        for item in items:
            B, C, interior, q = item[:4]
            y = q[interior]
            dist = gf.hop_distance(P, B, C)[interior]
            tr = gf.truncated(P, B, C, interior, depth=max(DEPTHS))
            X = torch.from_numpy((node_features(B, C) - mu) / sd).float()
            a = affine["solve"]
            bank["solve"].add(y, a[0] * y + a[1], dist)
            for dpt in DEPTHS:
                a = affine["trunc-%d" % dpt]
                bank["trunc-%d" % dpt].add(y, a[0] * tr[dpt - 1] + a[1], dist)
            with torch.no_grad():
                for nm, (net, _, _, _, _) in nets.items():
                    bank[nm].add(y, net(X, src, dst).numpy()[interior], dist)
            if with_ceiling:
                n_rows += y.size
                for dd in DEPTHS:
                    hop_far[dd] += float(((dist > dd) | (dist < 0)).sum())

    score_into(sc, test, True)
    if verbose:
        print("  test scored %.0fs" % (time.time() - t0), flush=True)
    score_into(sc_tr, train, False)

    arms = {}
    for nm in names:
        extra = {}
        if nm in nets:
            _, dpt, h, npar, glob = nets[nm]
            extra = dict(n_params=npar, width=h, loss_first=losses[nm][0],
                         loss_last=losses[nm][1],
                         note=("global token: breaks one-hop-per-layer" if glob
                               else "%d hops of P" % dpt))
        else:
            extra = dict(n_params=2, note=("one triangular solve" if nm == "solve"
                                           else "exact %s-hop truncation"
                                                % nm.split("-")[1]))
        arms[nm] = sc[nm].out(train_within=sc_tr[nm].out()["within"], **extra)

    hop = {d: hop_far[d] / n_rows for d in DEPTHS}
    shallow = [a for n, a in arms.items()
               if n.startswith("attn") and a["depth"] <= LOSS_CONDITION["depth"]]
    retired = any(a["within"] >= LOSS_CONDITION["within"] for a in shallow)
    return dict(n_test=n_test, n_train=n_train, epochs=epochs, seed=seed,
                floor=floor, n_rows=n_rows, pool=pool,
                test_digest=digest(test), train_digest=digest(train),
                arms=arms, ceilings=ceil, hop_fraction=hop,
                ceiling_lookup_worst=float(look.max()),
                ceiling_heuristic_worst=float(heur.max()),
                headroom_worst=float(1.0 - max(look.max(), heur.max())),
                one_solve_claim_retired=bool(retired),
                train_goal_ids=["train-%d" % i for i in range(n_train)],
                test_goal_ids=["test-%d" % i for i in range(n_test)],
                seconds=time.time() - t0)


def demo(n_test=100, n_train=20, epochs=EPOCHS, seed=SEED, verbose=False,
         budget=PARAM_BUDGET, depths=DEPTHS):
    r = race(n_test, n_train, epochs, seed, verbose=verbose, budget=budget,
             depths=depths)
    print("PROVENANCE     python -m ceqjepa.depth_race --draws %d --train %d "
          "--epochs %d   seed=%d  floor=%s"
          % (n_test, n_train, epochs, seed, r["floor"]))
    print("PINNED         %d unseen goals, %d training goals, %d scored rows; "
          "the reader is the only thing that varies"
          % (r["n_test"], r["n_train"], r["n_rows"]))
    print("INPUTS         %d columns, q1 and q2 EXCLUDED (they are hops)"
          % len(INPUT_NAMES))
    print("CEILINGS       marginal %.4f   lookup %.4f   best-heuristic %.4f"
          % (r["ceilings"]["marginal"], r["ceilings"]["lookup"],
             r["ceilings"]["heuristic"]))
    print("               worst draw: lookup %.4f   heuristic %.4f   "
          "headroom %.4f" % (r["ceiling_lookup_worst"],
                             r["ceiling_heuristic_worst"], r["headroom_worst"]))
    print("HOPS>d         %s   (test mass further than d hops from B')"
          % "  ".join("d=%d: %.4f" % (d, r["hop_fraction"][d]) for d in DEPTHS))
    print("DEPTH TABLE    arm               params  width   pooled    within   "
          "bound   far(>d)  TRAINfit  per-draw mean")
    for name, a in r["arms"].items():
        m, hh = gf._ci(a["per_draw"])
        print("  %-17s %7d  %5s  %8.4f  %7.4f  %7.4f  %7.4f  %7.4f  %6.3f +/- %.3f   %s"
              % (name, a["n_params"], a.get("width", "-"), a["pooled"],
                 a["within"], a["bound_within"], a["far_within"],
                 a["train_within"], m, hh, a["note"]))
    print("LOSS CONDITION a depth<=%d stack reaching within-draw R2 >= %.2f "
          "retires the one-solve claim -> %s"
          % (LOSS_CONDITION["depth"], LOSS_CONDITION["within"],
             "RETIRED" if r["one_solve_claim_retired"] else "SURVIVES"))
    print("SECONDS        %.1f" % r["seconds"])
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--budget", type=int, default=PARAM_BUDGET)
    ap.add_argument("--depths", type=int, nargs="*", default=list(DEPTHS))
    a = ap.parse_args()
    demo(a.draws, a.train, a.epochs, a.seed, a.verbose,
         budget=a.budget, depths=tuple(a.depths))
    sys.exit(0)
