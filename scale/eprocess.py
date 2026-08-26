"""The anytime-valid e-process for the M3 settled-vs-twin contrast.

======================================================================
PART 1: WHY THIS OBJECT, AND WHAT IT REPLACES
======================================================================

The M3 deciding cell has been read three times as a fixed-sample paired
bootstrap over five training seeds. A fixed-sample interval is only valid if it
is read once, at a sample size fixed before the data land. Three rounds have now
slipped their schedule, and a slipped schedule turns a fixed-sample reading into
an optional-stopping problem: the interval that gets reported is the one that
happened to be reachable when the deadline arrived.

An e-process removes that failure mode. For a hypothesis `H0`, an e-process is a
nonnegative supermartingale `E_t` with `E[E_0] <= 1`. Ville's inequality gives

    P( sup_t E_t >= 1/alpha )  <=  alpha           under H0

so the process may be read at ANY time -- mid-run, at a deadline, after an
interruption -- and the error guarantee still holds. A slipped schedule no
longer invalidates the verdict; it just reads the current `E_t`.

`scale/sprt.py` is the Bernoulli sibling of this file. Both are Ville-governed;
the SPRT decides a per-draw rate, this decides a continuous paired difference.

======================================================================
PART 2: THE CONSTRUCTION
======================================================================

For paired per-seed differences of CLIPPED outcomes

    d_i = min(NRMSE_reference(seed_i), C) - min(NRMSE_arm(seed_i), C)

(positive `d_i` means the arm has the lower error, i.e. the arm wins), the
process is a uniform mixture over a fixed grid of `lambda`:

    E_t  =  (1/|Lambda|) * SUM_{lam in Lambda} PROD_{i<=t} ( 1 + lam * d_i / B )

with `Lambda` a grid inside `[0, 1/2]` and `B` an a-priori bound on `|d_i|`.

WHY IT IS AN E-PROCESS, in two lines:

  * NONNEGATIVE. `|d_i| <= B` and `lam <= 1/2` give every factor `>= 1/2 > 0`.
    The bound is enforced, not assumed: `update` RAISES on `|d| > B` rather than
    clamping, because a clamped process keeps printing a guarantee it has
    already voided.
  * SUPERMARTINGALE. `E[1 + lam*d_i/B | F_{i-1}] = 1 + lam*E[d_i|F_{i-1}]/B <= 1`
    under `H0: E[d_i | F_{i-1}] <= 0`, for every `lam >= 0`. Each grid arm is
    therefore a nonnegative supermartingale starting at 1, and a convex
    combination of supermartingales is a supermartingale. `lambda` is fixed
    before the data, so it is trivially predictable.

The mixture, rather than one `lambda`, is what makes the process predictable
without knowing the effect size in advance. Its price is stated in Part 4 and
measured, not asserted.

THE DECISION RULE, immutable once the first real seed lands:

    E_t >= 40   ->  the arm beats the reference, at ANY t
    symmetric process on `-d_i` for the reference direction
    neither crossing by phase end  ->  "undecided at evidence E_t = [value]"

TWO DIRECTIONS, ONE BUDGET. Ville bounds each direction separately at `1/tau`.
BOTH directions are live -- which arm wins is not fixed in advance -- so the
event that actually gets reported is the UNION of the two, whose bound is
`N_DIRECTIONS / tau`. A family-wise `0.05` therefore requires

    2 / tau <= 0.05        i.e.  tau >= 40,   alpha per direction = 0.025

and NOT `tau = 20`, which delivers `2/20 = 0.10`. The alternative route -- keep
`tau = 20` and report only "the settled direction crossed at 0.05", never "some
direction crossed" -- is arithmetically valid but is enforced by prose rather
than by a constant, and the direction is not pre-specified here, so the union is
the operative event. The threshold route is taken. `calibrate` reports the
per-direction and either-direction rates separately so that neither has to be
inferred from a sentence.

======================================================================
PART 3: `B`, AND ITS PROVENANCE
======================================================================

`B` is the width of the CLIPPED outcome range. The clip is part of the
definition of `d_i`, applied to each ARM before differencing.

  * `negation_scope.nrmse` is `RMSE / std(y)`, a ratio of nonnegative
    quantities, so it is bounded below by 0 and is exactly 1.0 for the mean
    predictor (`scale/negation_scope.py:96-101`). It is NOT bounded above.
  * The earlier derivation took `B = 1.0` from the CREDIT bar of
    `M3_QUINTUPLE_PREREGISTERED_READING.md` section 5, last row: "any cell whose
    seed-mean NRMSE is at or above 1.0 ... is credited with nothing, whatever
    its contrast says." That bar governs what a cell EARNS; it does not bound
    what `nrmse` RETURNS, and `read_paired` does not apply it. Scanning every
    `results/**/*.jsonl` finds 59 `eval_nrmse` readings of which 21 are at or
    above `1.0`, the largest `1.0742670875495859` at
    `results/m3_quintuple.jsonl::softmax_k0_s64_d24_st20_ntr256_nev128_b21_sd0`.
    `B = 1.0` was therefore an assumption about which cells would be read.

  * `min(x, C)` lies in `[0, C]` for every `x >= 0`, so the difference of two
    clipped arms lies in `[-C, +C]` with no assumption at all. `B = C = 2.0`.

WHAT THE CLIP COSTS, STATED HERE BECAUSE IT IS NOT FREE.

  1. IT MOVES THE ESTIMAND. The null tested is
     `H0: E[min(NRMSE_ref, C)] <= E[min(NRMSE_arm, C)]`, a hypothesis about the
     clipped outcome. It is NOT sign-preserving on the raw metric: on the
     instance that refuted the post-hoc clip (`d = -100` w.p. `0.01`, `+0.5`
     w.p. `0.99`, `E[d] = -0.505`) the per-arm clip gives `+0.48` and the
     post-hoc clip `+0.475`. Both flip the sign. What the per-arm clip has that
     the post-hoc clip lacks is that it is a fixed monotone transform of each
     OUTCOME, fixed before the data and independent of the pairing, so it
     defines an estimand rather than depending on which differences turn up.

     WHERE THE CLIP BITES. On the READING PATH it does not: `read_paired`
     consumes bucket journals, and the largest `eval_nrmse` in any
     `results/**/*.jsonl` is `1.0742670875495859`, below `C`. Repo-wide the
     metric does exceed `C` -- ten readings in `results/m3_capability.txt`
     reach it, the largest `3.696671` at
     `results/m3_capability.txt:589` (`s=192 d=64 steps=150 n_train=128
     n_eval=256 seed=1`). Every one of those ten is an `n_train=128` cell and
     every one is far above the section-5 credit bar of `1.0`, so the region
     the clip truncates is a region in which no cell earns anything. `C = 2.0`
     is therefore twice the credit bar, not a guess at the metric's range: it
     truncates only where the pre-registration has already zeroed the cell.
  2. IT DOUBLES THE PRICE. The increments enter as `lam * d_i / B`, so doubling
     `B` halves the per-seed log-evidence rate and roughly doubles the seed
     count at a fixed effect. The price table printed by `main()` carries the
     new figures; the pre-clip figures it replaces are at `DONE.md:369-388`.

A tighter `B` would make the process grow faster, and the temptation is to set
`B` from the observed spread (the pre-registration measures a paired per-seed
delta standard deviation of `0.056889`). That would be fitting the instrument to
the data it is about to read. A `B` that a single seed exceeds destroys
nonnegativity and with it Ville's inequality, which is exactly the failure the
null calibration in Part 5 is built to catch; `update` RAISES rather than
clamping, so that failure is loud.

======================================================================
PART 4: THE CEILING -- READ THIS BEFORE PRICING ANY RUN
======================================================================

Because every factor is at most `1 + lam <= 1.5`, the process has a hard ceiling
that no data can beat:

    max attainable E_t  =  (1/|Lambda|) * SUM_lam (1 + lam) ** t

The ceiling does not depend on `B`: the largest legal increment is `d_i = B` and
`1 + lam*B/B = 1 + lam` whatever `B` is. So the kill below survived the change
of `B` for a reason, not by luck.

At the pre-registered FIVE seeds this ceiling is `3.801691`, and the loosest
bound (one arm at `lam = 1/2`) is `1.5 ** 5 = 7.59375`. Both are below the
threshold `40`. THE FIVE-SEED CONTRAST CANNOT BE DECIDED BY THIS PROCESS IN
EITHER DIRECTION, whatever the numbers turn out to be, and it cannot be decided
before `t = MIN_T_MIXTURE = 13` seeds even if every seed lands at the maximal
legal effect. Raising the threshold from `20` to `40` widened that gap: the
minimum horizons moved from `8` and `11` to `10` and `13`.

That is a fact about the design, found before any real seed landed. `main()`
prints the arithmetic for the seed counts a real effect size would need.

======================================================================
PART 5: THE MUST-FIRE, BOTH DIRECTIONS
======================================================================

An instrument that has never been run against a known answer is not an
instrument. Three checks, all in `calibrate`, all reported by `main()`:

  1. NULL. Streams with `E[d_i] = 0` must not cross `20` more often than the
     nominal `alpha = 0.05`. Run at the maximal-variance mean-zero bounded null
     (`d_i = +-B` equiprobable), which is the worst case, and at the realistic
     null drawn at the measured paired spread.
  2. PLANTED. A stream carrying a real effect must cross, or a null reading
     would mean nothing.
  3. PEEK. The same null run through a deliberately BROKEN construction, in
     which `lambda_i` is chosen after seeing `d_i` (`lam_i = sign(d_i)/2`).
     Predictability is the property that makes the supermartingale argument
     work; deleting it must be SEEN to break the null, or the null check has no
     teeth.

THE HORIZON IS PART OF THE CONTROL. A null calibration run over a horizon
shorter than `MIN_T_MIXTURE` cannot cross whatever the construction does, and
would pass while measuring nothing. `calibrate` refuses such a horizon.

NO THREAD DEPENDENCE. numpy elementwise transcendentals and cumulative sums,
which do not dispatch to a threaded BLAS. `torch.set_num_threads(2)` is pinned
here anyway so that this file, not its launcher, is the authority.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import NamedTuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                      # pinned HERE, not by the launcher

NAME = "chase_eprocess"

# ---------------------------------------------------------------------------
# THE PRE-REGISTERED CONSTANTS. Immutable once the first real seed lands.
# ---------------------------------------------------------------------------
NRMSE_FLOOR = 0.0     #: negation_scope.nrmse is a ratio of nonnegative terms
NRMSE_BAR = 1.0       #: M3 pre-registration section 5: the CREDIT bar, not a bound
CLIP_C = 2.0          #: the outcome clip, applied per arm INSIDE the definition
B = CLIP_C - NRMSE_FLOOR                   #: 2.0 -- the a-priori bound on |d_i|

ALPHA_FAMILY = 0.05   #: the budget the report is allowed to quote
N_DIRECTIONS = 2      #: settled-wins and twin-wins are both live, so both count
ALPHA = ALPHA_FAMILY / N_DIRECTIONS        #: 0.025, per direction
THRESHOLD = 1.0 / ALPHA                    #: 40.0, per direction

#: A uniform grid inside (0, 1/2]. `j / 20` so that the endpoint is exactly 0.5.
LAMBDA_GRID = tuple(j / 20.0 for j in range(1, 11))

#: The measured paired per-seed delta spread, M3 pre-registration section 3.
#: Used ONLY to draw realistic calibration streams. It does not enter `B`.
SD_PAIRED = 0.056889

#: The stated resolution floor of the five-seed bootstrap, same section.
EFFECT_FLOOR = 0.05

#: Journal the real seeds land in.
DEFAULT_JOURNAL = pathlib.Path(__file__).resolve().parents[1] / \
    "results" / "m3_quintuple_v2.jsonl"


def clipped_nrmse(x: float) -> float:
    """`min(x, CLIP_C)`, refusing a reading the metric cannot legally produce.

    `negation_scope.nrmse` is a ratio of nonnegative quantities, so a negative
    reading is a broken journal rather than a small effect, and `nan` is what
    its zero-variance branch returns (`scale/negation_scope.py:99-100`). Both
    raise: a clip that silently absorbed them would hide the fault it exists to
    make impossible.
    """
    if not math.isfinite(x) or x < NRMSE_FLOOR:
        raise ValueError(
            f"illegal NRMSE reading {x!r}: negation_scope.nrmse is a ratio of "
            f"nonnegative quantities and returns nan only when std(y) == 0. "
            f"The clip does not absorb this; find the cause.")
    return min(x, CLIP_C)


def paired_difference(nrmse_ref: float, nrmse_arm: float) -> float:
    """`min(ref, C) - min(arm, C)` -- the clip is INSIDE the definition.

    Each arm is clipped before the subtraction. Clipping the DIFFERENCE instead
    would be a transform of the paired quantity rather than of the outcome, and
    would change what is being estimated as a function of which differences
    happened to turn up. Clipping the outcome is a fixed monotone transform of
    each arm, chosen before the data; it still moves the estimand to the clipped
    metric (Part 3, cost 1), and that is pre-registered rather than hidden.
    """
    return clipped_nrmse(nrmse_ref) - clipped_nrmse(nrmse_arm)


def max_attainable(t: int) -> float:
    """The largest value the MIXTURE can reach in `t` steps, at `d_i = B`."""
    return sum((1.0 + lam) ** t for lam in LAMBDA_GRID) / len(LAMBDA_GRID)


def max_attainable_single_arm(t: int) -> float:
    """The looser ceiling that ignores the mixture penalty: `1.5 ** t`."""
    return (1.0 + max(LAMBDA_GRID)) ** t


def _min_t(ceiling) -> int:
    t = 1
    while ceiling(t) < THRESHOLD:
        t += 1
    return t


MIN_T_MIXTURE = _min_t(max_attainable)             #: 13 -- the real instrument
MIN_T_SINGLE_ARM = _min_t(max_attainable_single_arm)   #: 10 -- the loose bound


# ---------------------------------------------------------------------------
# the process
# ---------------------------------------------------------------------------
def _logsumexp(xs) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


class Eprocess:
    """One direction. `update(d)` folds in one paired difference.

    Kept in log space per grid arm. G8: the decision compares a mixture value
    against a constant, and the underflow that would matter is an arm decaying
    to zero, which in log space decays linearly and cannot reach the boundary
    from below by rounding.
    """

    def __init__(self, *, b: float = B, grid=LAMBDA_GRID,
                 threshold: float = THRESHOLD):
        self.b, self.grid, self.threshold = b, tuple(grid), threshold
        self.log_arm = [0.0] * len(self.grid)
        self.t = 0
        self.peak = 1.0
        self.history: list[float] = []

    @property
    def value(self) -> float:
        return math.exp(_logsumexp(self.log_arm) - math.log(len(self.grid)))

    @property
    def crossed(self) -> bool:
        return self.peak >= self.threshold

    def update(self, d: float) -> float:
        if not math.isfinite(d):
            raise ValueError(f"non-finite difference: {d!r}")
        if abs(d) > self.b:
            raise ValueError(
                f"|d| = {abs(d)!r} exceeds the a-priori bound B = {self.b!r}; "
                "nonnegativity and with it Ville's inequality are void. The "
                "bound is not clamped: find the cause.")
        for g, lam in enumerate(self.grid):
            self.log_arm[g] += math.log1p(lam * d / self.b)
        self.t += 1
        v = self.value
        self.peak = max(self.peak, v)
        self.history.append(v)
        return v


class Pair:
    """Both directions of one contrast, run on the same stream."""

    def __init__(self, **kw):
        self.settled = Eprocess(**kw)
        self.twin = Eprocess(**kw)

    def update(self, d: float) -> None:
        self.settled.update(d)
        self.twin.update(-d)

    @property
    def decision(self) -> str | None:
        if self.settled.crossed and not self.twin.crossed:
            return "settled"
        if self.twin.crossed and not self.settled.crossed:
            return "twin"
        if self.settled.crossed and self.twin.crossed:
            return "both"          # impossible for a real stream; reported, not hidden
        return None


# ---------------------------------------------------------------------------
# the calibration
# ---------------------------------------------------------------------------
class Spec(NamedTuple):
    name: str
    kind: str            #: "rademacher" or "gauss"
    mean: float
    sd: float


NULL_RADEMACHER = Spec("null +-B (max-variance H0)", "rademacher", 0.0, B)
NULL_GAUSSIAN = Spec("null N(0, sd_paired)", "gauss", 0.0, SD_PAIRED)
PLANTED_FLOOR = Spec("planted at the resolution floor", "gauss",
                     EFFECT_FLOOR, SD_PAIRED)
PLANTED_LARGE = Spec("planted 10x the floor", "gauss", 10 * EFFECT_FLOOR,
                     SD_PAIRED)


def _draw(spec: Spec, rng, shape) -> np.ndarray:
    if spec.kind == "rademacher":
        return np.where(rng.random(shape) < 0.5, -spec.sd, spec.sd)
    d = rng.normal(spec.mean, spec.sd, shape)
    return np.clip(d, -B, B)          # the a-priori bound, enforced on the draw


def _run_block(d: np.ndarray, b: float, peek: bool) -> np.ndarray:
    """log E_t for every replay and every t. `d` is (R, T); returns (R, T)."""
    if peek:
        # THE BROKEN CONTROL: lambda chosen after seeing d. One arm, no mixture.
        lam = 0.5 * np.sign(d)
        return np.cumsum(np.log1p(lam * d / b), axis=1)
    lam = np.asarray(LAMBDA_GRID)[None, None, :]
    logs = np.cumsum(np.log1p(lam * d[:, :, None] / b), axis=1)
    m = logs.max(axis=2, keepdims=True)
    return (m[:, :, 0] + np.log(np.exp(logs - m).sum(axis=2))
            - math.log(len(LAMBDA_GRID)))


def calibrate(spec: Spec, *, n_rep: int = 10000, horizon: int = 400,
              seed: int = 0, peek: bool = False, block: int = 500) -> dict:
    """Run the process against a stream whose answer is known in advance.

    Returns the empirical rate at which `sup_t E_t` crosses `THRESHOLD`, in each
    direction separately and in either direction, plus the evidence that the
    replays had a real opportunity to cross.
    """
    if horizon < MIN_T_MIXTURE:
        raise ValueError(
            f"horizon {horizon} < MIN_T_MIXTURE {MIN_T_MIXTURE}: the process "
            "cannot cross at this horizon whatever the construction does, so "
            "the calibration would pass while measuring nothing.")
    rng = np.random.default_rng(seed)
    log_thr = math.log(THRESHOLD)
    n_s = n_t = n_either = n_a2 = 0
    max_log = -math.inf
    cross_t: list[int] = []
    done = 0
    while done < n_rep:
        r = min(block, n_rep - done)
        d = _draw(spec, rng, (r, horizon))
        ls = _run_block(d, B, peek)
        lt = _run_block(-d, B, peek)
        hit_s = ls >= log_thr
        hit_t = lt >= log_thr
        any_s, any_t = hit_s.any(axis=1), hit_t.any(axis=1)
        n_s += int(any_s.sum())
        n_t += int(any_t.sum())
        n_either += int((any_s | any_t).sum())
        n_a2 += int((ls >= math.log(2.0)).any(axis=1).sum())
        max_log = max(max_log, float(ls.max()))
        if any_s.any():
            cross_t.extend((hit_s[any_s].argmax(axis=1) + 1).tolist())
        done += r
    return dict(
        spec=spec.name, peek=peek, n_rep=n_rep, horizon=horizon, seed=seed,
        cross_settled=n_s / n_rep, cross_twin=n_t / n_rep,
        cross_either=n_either / n_rep, frac_above_2=n_a2 / n_rep,
        max_peak=math.exp(min(max_log, 700.0)),
        median_cross_t=(sorted(cross_t)[len(cross_t) // 2] if cross_t else None),
    )


def growth_rate(mean: float, sd: float, lam: float = 0.5, *, n: int = 200000,
                seed: int = 7) -> float:
    """E[ log(1 + lam*d/B) ] at one lambda -- the per-seed log-evidence rate."""
    rng = np.random.default_rng(seed)
    d = np.clip(rng.normal(mean, sd, n), -B, B)
    return float(np.log1p(lam * d / B).mean())


def seeds_needed(mean: float, sd: float = SD_PAIRED) -> dict:
    """How many paired seeds this construction needs at a given effect size.

    Two numbers, because the mixture is not free: `best_arm` is the count the
    single best `lambda` would need, and `mixture` adds the worst-case mixture
    penalty `log|Lambda|`, which is what the shipped process actually pays.
    """
    g = growth_rate(mean, sd)
    if g <= 0.0:
        return dict(mean=mean, rate=g, best_arm=None, mixture=None)
    best = math.ceil(math.log(THRESHOLD) / g)
    mix = math.ceil((math.log(THRESHOLD) + math.log(len(LAMBDA_GRID))) / g)
    return dict(mean=mean, rate=g, best_arm=max(best, MIN_T_SINGLE_ARM),
                mixture=max(mix, MIN_T_MIXTURE))


def min_detectable_effect(t: int, sd: float = SD_PAIRED) -> float:
    """The smallest effect whose CONSERVATIVE mixture price is at most `t` seeds.

    The inverse of `seeds_needed`, found by bisection on the same function so
    that the two cannot drift apart. `math.inf` means no legal effect fits the
    budget: not even every seed landing at the maximal difference `d_i = B`
    would carry the process across in `t` steps.

    This answers the question a reroute actually has to answer -- "what would
    the effect have to BE for this budget to buy a decision" -- which is the
    useful direction when the effect has never been measured.
    """
    if seeds_needed(B, 0.0)["mixture"] > t:
        return math.inf
    lo, hi = 0.0, B
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = seeds_needed(mid, sd)["mixture"]
        if r is not None and r <= t:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# the pilot screen -- SIZING ONLY, never a verdict
# ---------------------------------------------------------------------------
#: The pilot is three paired seeds, six units. It exists to decide whether to
#: spend the e-process budget on a task whose effect has never been measured.
PILOT_SEEDS = 3
PILOT_GATE = 0.10


def pilot_rates(effect: float, *, n_seeds: int = PILOT_SEEDS,
                gate: float = PILOT_GATE, sd: float = SD_PAIRED,
                n_rep: int = 200000, seed: int = 17) -> float:
    """P(the pilot says GO) at a given true effect.

    THIS IS NOT A TEST AND ITS OUTPUT IS NOT A VERDICT. It is a sizing
    measurement whose only licensed use is deciding whether to start the
    e-process. It is a fixed-sample paired mean, so it carries none of the
    anytime-validity that makes `Eprocess` readable at a slipped deadline, and
    a "GO" from it may never be reported as evidence that an arm won.

    Why a pilot is the right shape here: "is the effect at least 0.2 NRMSE" is
    a point-estimate question, and at the pre-registration's paired spread the
    three-seed paired mean has standard error `sd / sqrt(3)`, which separates
    `0.2` from the `0.05` resolution floor by more than four standard errors.
    Six units therefore decides a `2 * seeds_needed(0.20)["mixture"]`-unit
    commitment.
    """
    rng = np.random.default_rng(seed)
    dbar = rng.normal(effect, sd / math.sqrt(n_seeds), n_rep)
    return float((dbar >= gate).mean())


# ---------------------------------------------------------------------------
# the live reading -- what makes the process LIVE rather than a plan
# ---------------------------------------------------------------------------
def _parse_key(key: str) -> tuple[str, str, int]:
    """`{cell}_k{k}_{config}_sd{seed}` -> (cell, config-without-k, seed)."""
    head, _, tail = key.rpartition("_sd")
    cell, _, rest = head.partition("_")
    k_field, _, config = rest.partition("_")
    if not k_field.startswith("k"):
        raise ValueError(f"unparsable journal key: {key!r}")
    return cell, config, int(tail)


def read_paired(path, *, ref: str = "twin", arm: str = "settled"):
    """Paired per-seed differences from a bucket journal, in SEED ORDER.

    `d_i = min(NRMSE_ref(seed), C) - min(NRMSE_arm(seed), C)`, positive meaning
    `arm` wins. The clip is applied HERE, on the way in, because this is the
    only place a raw journal reading enters the process.

    Seed order is the fixed pre-registered order, so it does not depend on the
    data -- which is what keeps `lambda` predictable. A seed present for only
    one of the two cells is DROPPED, not imputed. Two cells whose configs
    differ raise, rather than pairing across settings.
    """
    got: dict[int, dict[str, float]] = {}
    cfgs: dict[str, str] = {}
    p = pathlib.Path(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cell, config, seed = _parse_key(row["key"])
        if cell not in (ref, arm):
            continue
        prev = cfgs.setdefault(cell, config)
        if prev != config:
            raise ValueError(f"{cell} appears at two configs: {prev} vs {config}")
        got.setdefault(seed, {})[cell] = float(row["value"]["eval_nrmse"])
    if len(cfgs) == 2 and cfgs[ref] != cfgs[arm]:
        raise ValueError(
            f"config mismatch: {ref} at {cfgs[ref]!r}, {arm} at {cfgs[arm]!r}")
    return [(s, paired_difference(got[s][ref], got[s][arm])) for s in sorted(got)
            if ref in got[s] and arm in got[s]]


def live(path=DEFAULT_JOURNAL, *, ref: str = "twin", arm: str = "settled") -> dict:
    """The current reading. Safe on an empty or partial journal."""
    paired = read_paired(path, ref=ref, arm=arm)
    pair = Pair()
    void = None
    for seed, d in paired:
        try:
            pair.update(d)
        except ValueError as exc:
            void = f"VOID at seed {seed}: {exc}"
            break
    dec = pair.decision
    if void is not None:
        text = void
    elif dec is None:
        text = (f"undecided at evidence E_t = {pair.settled.value!r} "
                f"({arm} direction), {pair.twin.value!r} ({ref} direction), "
                f"t = {pair.settled.t}")
    else:
        text = (f"{arm if dec == 'settled' else ref} beats the other at "
                f"alpha = {ALPHA}, anytime-valid, at t = {pair.settled.t}, "
                f"E_t = {max(pair.settled.peak, pair.twin.peak)!r}")
    return dict(t=pair.settled.t, e_settled=pair.settled.value,
                e_twin=pair.twin.value, peak_settled=pair.settled.peak,
                peak_twin=pair.twin.peak, decision=dec, text=text,
                per_seed=paired, void=void,
                ceiling_here=max_attainable(pair.settled.t),
                can_decide=max_attainable(pair.settled.t) >= THRESHOLD)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def print_construction() -> None:
    print("=== E-PROCESS CONSTRUCTION (fixed before the first real seed) ===")
    print(f"  E_t = mean_lam PROD_i (1 + lam*d_i/B)")
    print(f"  d_i = min(NRMSE_ref, C) - min(NRMSE_arm, C)   "
          f"CLIP INSIDE THE DEFINITION, both arms, before differencing")
    print(f"  C = {CLIP_C!r}   B = C - {NRMSE_FLOOR!r} = {B!r}   "
          f"min(x, C) in [0, C] for every x >= 0, so |d_i| <= B unconditionally")
    print(f"    negation_scope.nrmse >= 0 (ratio of nonnegative terms), "
          f"= 1.0 for the mean predictor, UNBOUNDED ABOVE")
    print(f"    the section-5 bar 1.0 is a CREDIT bar, not a bound: "
          f"21 of 59 readings in results/**/*.jsonl are >= 1.0 "
          f"(max 1.0742670875495859)")
    print(f"    the clip moves the estimand to E[min(NRMSE, C)]; INERT on the "
          f"reading path (max journal reading 1.0742670875495859 < C), ACTIVE "
          f"elsewhere -- 10 readings in results/m3_capability.txt exceed C, "
          f"max 3.696671 at line 589 -- but all 10 are n_train=128 cells far "
          f"above the section-5 credit bar 1.0, so the truncated region earns "
          f"nothing anyway")
    print(f"  Lambda = {LAMBDA_GRID!r}   |Lambda| = {len(LAMBDA_GRID)}")
    print(f"  alpha family-wise = {ALPHA_FAMILY!r} over "
          f"{N_DIRECTIONS} live directions -> alpha = {ALPHA!r} per direction "
          f"-> threshold = {THRESHOLD!r}")
    print(f"    union bound actually reported: "
          f"{N_DIRECTIONS}/{THRESHOLD!r} = {N_DIRECTIONS / THRESHOLD!r}  "
          f"(threshold 20 would have given {N_DIRECTIONS / 20.0!r})")
    print(f"  worst-case factor at |d| = B, lam = 1/2: "
          f"{1.0 + 0.5 * (-B) / B!r}  (> 0, so nonnegativity holds)")


def print_ceiling() -> None:
    print("\n=== THE CEILING: what no data can beat ===")
    print(f"{'t':>4} {'max mixture E_t':>18} {'max single-arm 1.5**t':>23} "
          f"{'>= ' + repr(THRESHOLD) + '?':>8}")
    for t in (5, 8, 10, 11, 12, 13, 20):
        print(f"{t:>4} {max_attainable(t):>18.6f} "
              f"{max_attainable_single_arm(t):>23.6f} "
              f"{'YES' if max_attainable(t) >= THRESHOLD else 'no':>8}")
    print(f"  MIN_T_MIXTURE = {MIN_T_MIXTURE}  "
          f"MIN_T_SINGLE_ARM = {MIN_T_SINGLE_ARM}")
    print(f"  The pre-registered FIVE seeds cannot cross in either direction: "
          f"ceiling {max_attainable(5)!r} < {THRESHOLD!r}.")


def print_must_fire(n_rep: int, horizon: int) -> None:
    print(f"\n=== MUST-FIRE  n_rep={n_rep} horizon={horizon} "
          f"alpha={ALPHA!r} threshold={THRESHOLD!r} ===")
    print(f"{'stream':>34} {'horizon':>8} {'cross settled':>14} "
          f"{'cross twin':>11} {'either':>8} {'frac E>2':>9} {'max peak':>11} "
          f"{'med t':>7} {'expected':>26}")
    # A planted stream needs a horizon LONGER than its own price, or a
    # "did not cross" would be a property of the schedule and DIRECTION 2 would
    # measure nothing. `seeds_needed` supplies the price; the null rows keep the
    # caller's horizon, since a longer horizon only makes a null test harder.
    def planted_horizon(spec: Spec) -> int:
        need = seeds_needed(spec.mean, spec.sd)["mixture"] or MIN_T_MIXTURE
        return max(horizon, 2 * need)

    rows = [
        (NULL_RADEMACHER, False, horizon, f"both <= alpha={ALPHA}"),
        (NULL_GAUSSIAN, False, horizon, f"both <= alpha={ALPHA}"),
        (PLANTED_FLOOR, False, planted_horizon(PLANTED_FLOOR),
         "settled crosses"),
        (PLANTED_LARGE, False, planted_horizon(PLANTED_LARGE),
         "settled crosses"),
        (NULL_RADEMACHER, True, horizon, "BROKEN: must exceed alpha"),
    ]
    verdicts = []
    for i, (spec, peek, hz, want) in enumerate(rows):
        c = calibrate(spec, n_rep=n_rep, horizon=hz, seed=200 + i,
                      peek=peek)
        label = spec.name + ("  [lambda PEEKS: broken]" if peek else "")
        mt = "-" if c["median_cross_t"] is None else str(c["median_cross_t"])
        print(f"{label:>34} {hz:>8} {c['cross_settled']:>14.4f} "
              f"{c['cross_twin']:>11.4f} {c['cross_either']:>8.4f} "
              f"{c['frac_above_2']:>9.4f} {c['max_peak']:>11.4g} {mt:>7} "
              f"{want:>26}")
        verdicts.append((spec, peek, c))
    nulls = [c for s, pk, c in verdicts if s.mean == 0.0 and not pk]
    planted = [c for s, pk, c in verdicts if s.mean > 0.0 and not pk]
    broken = [c for s, pk, c in verdicts if pk]
    ok_null = all(c["cross_settled"] <= ALPHA and c["cross_twin"] <= ALPHA
                  for c in nulls)
    ok_planted = all(c["cross_settled"] > 0.5 for c in planted)
    ok_broken = all(c["cross_settled"] > ALPHA for c in broken)
    worst = max(max(c["cross_settled"], c["cross_twin"]) for c in nulls)
    se = math.sqrt(worst * (1.0 - worst) / n_rep)
    print(f"\n  DIRECTION 1 (null does NOT cross): "
          f"{'PASS' if ok_null else 'FAIL'}  "
          f"-- worst empirical rate {worst:.4f} +- {se:.4f} (1 s.e., "
          f"n_rep={n_rep}) vs nominal alpha {ALPHA!r}")
    print(f"      worst either-direction rate "
          f"{max(c['cross_either'] for c in nulls):.4f} vs the union bound "
          f"{N_DIRECTIONS * ALPHA!r} (= the family-wise budget "
          f"{ALPHA_FAMILY!r})")
    print(f"  DIRECTION 2 (planted effect DOES cross): "
          f"{'PASS' if ok_planted else 'FAIL'}")
    seen = ("SEEN TO FIRE" if ok_broken else
            "DID NOT FIRE -> the null check measures nothing")
    print(f"  CONTROL (predictability deleted -> null breaks): {seen}")
    if not (ok_null and ok_planted and ok_broken):
        print("  THE CONSTRUCTION IS NOT CALIBRATED. Nothing anytime-valid may "
              "be claimed until the cause is found.")


def print_price(n_rep: int = 2000, horizon: int = 1500) -> None:
    """The reprice arithmetic. Two seed counts per effect size, on purpose.

    `seeds, mixture` is the CONSERVATIVE bound: it charges the full worst-case
    mixture penalty `log|Lambda|`, which assumes every grid arm but the best
    contributes nothing. `seeds, measured` is the simulated median crossing
    time, where the other arms do contribute. The bound is what to plan with;
    the median is what to expect.
    """
    print("\n=== THE PRICE: paired seeds needed to CROSS, by effect size ===")
    print(f"{'effect (NRMSE)':>16} {'log-rate/seed':>15} {'seeds, best lam':>17} "
          f"{'seeds, mixture':>16} {'seeds, measured':>17} {'cross rate':>11} "
          f"{'units (2/seed)':>16}")
    for m in (EFFECT_FLOOR, 0.10, 0.20, 0.50, 1.00):
        r = seeds_needed(m)
        mix = r["mixture"]
        sd = 0.0 if m >= B else SD_PAIRED
        c = calibrate(Spec(f"price {m}", "gauss", m, sd), n_rep=n_rep,
                      horizon=horizon, seed=int(1000 * m) + 300)
        med = "-" if c["median_cross_t"] is None else str(c["median_cross_t"])
        print(f"{m:>16.4f} {r['rate']:>15.8f} {r['best_arm']:>17} "
              f"{mix:>16} {med:>17} {c['cross_settled']:>11.4f} "
              f"{2 * mix:>16}")
    print(f"  One unit = one (cell, k, seed) training run; the settled-vs-twin "
          f"contrast is 2 units per seed.")
    print(f"  The pre-registered run is 5 seeds = 10 units for this contrast, "
          f"and 5 seeds cannot cross at any effect size.")
    print(f"  Clipping the outcome moved B from 1.0 to {B!r}, which halves "
          f"lam*d/B; raising the threshold from 20.0 to {THRESHOLD!r} adds a "
          f"further factor log(40*10)/log(20*10) = "
          f"{math.log(THRESHOLD * len(LAMBDA_GRID)) / math.log(20.0 * len(LAMBDA_GRID))!r}"
          f" on the mixture bound.")
    print(f"  Measured against the pre-clip table at DONE.md:369-388, the "
          f"mixture column moved 218->486, 110->244, 56->123, 24->51, 14->27, "
          f"i.e. by 2.229, 2.218, 2.196, 2.125, 1.929.")
    print(f"  Measured column: n_rep={n_rep} horizon={horizon}; a '-' means the "
          f"stream did not cross within the horizon.")


def print_reroute() -> None:
    """What a budget would have to BUY, and the six units that decide it.

    The reroute target `counter_squared` has no measured NRMSE effect anywhere
    in the repo -- it exists only as a Hankel-rank function
    (`ceq/hankel.py:367-370`) registered in a rank instrument
    (`ceq/hankel.py:380-387`), with no runnable M3 task and no `eval_nrmse`.
    So the question is inverted: not "what is the effect" but "what would the
    effect have to be", and then a cheap pilot to find out.
    """
    print("\n=== THE REROUTE: what a budget would have to buy ===")
    print(f"{'paired seeds':>13} {'units (2/seed)':>15} "
          f"{'min detectable effect (NRMSE)':>31}")
    for t in (MIN_T_MIXTURE - 1, MIN_T_MIXTURE, 20, 26, 40, 60, 100, 123):
        mu = min_detectable_effect(t)
        shown = ("unreachable at any legal effect" if mu == math.inf
                 else f"{mu:.4f}")
        print(f"{t:>13} {2 * t:>15} {shown:>31}")
    bound_floor = seeds_needed(B, 0.0)["mixture"]
    print(f"  Two floors, and they differ because one is exact and one is a "
          f"bound. EXACT CEILING: {MIN_T_MIXTURE} seeds = "
          f"{2 * MIN_T_MIXTURE} units, reached only if every seed lands at "
          f"d_i = B = {B!r}. CONSERVATIVE BOUND (what to plan with): "
          f"{bound_floor} seeds = {2 * bound_floor} units at the same effect.")
    print(f"  The pre-registered run is 10 units. NO effect size rescues it.")
    print(f"\n  PILOT SCREEN (SIZING ONLY, NEVER A VERDICT): "
          f"{PILOT_SEEDS} paired seeds = {2 * PILOT_SEEDS} units, "
          f"go iff paired mean >= {PILOT_GATE!r}")
    print(f"{'true effect':>13} {'P(GO)':>10}")
    for m in (0.0, EFFECT_FLOOR, 0.10, 0.20, 0.50):
        print(f"{m:>13.4f} {pilot_rates(m):>10.4f}")
    print(f"  It decides a {2 * seeds_needed(0.20)['mixture']}-unit commitment "
          f"at {2 * PILOT_SEEDS} units. It is a fixed-sample paired mean, so it "
          f"carries no anytime-validity and may not be quoted as a result.")
    print(f"  SENSITIVITY: sd_paired = {SD_PAIRED!r} was measured on "
          f"negation_scope, NOT on the reroute target. If the target's paired "
          f"spread is larger the screen degrades:")
    print(f"{'sd multiple':>13} {'P(GO | 0.20)':>14} "
          f"{'P(GO | ' + repr(EFFECT_FLOOR) + ')':>16}")
    for mult in (1.0, 2.0, 3.0):
        sd = mult * SD_PAIRED
        print(f"{mult:>13.1f} {pilot_rates(0.20, sd=sd):>14.4f} "
              f"{pilot_rates(EFFECT_FLOOR, sd=sd):>16.4f}")


def print_live(path) -> None:
    print(f"\n=== LIVE READING  journal={path} ===")
    try:
        r = live(path)
    except FileNotFoundError:
        print(f"  NOT FOUND: {path}")
        return
    print(f"  paired seeds t = {r['t']}   per-seed d = {r['per_seed']!r}")
    print(f"  E_t settled-direction = {r['e_settled']!r}  "
          f"peak {r['peak_settled']!r}")
    print(f"  E_t twin-direction    = {r['e_twin']!r}  peak {r['peak_twin']!r}")
    print(f"  ceiling at t = {r['t']}: {r['ceiling_here']!r}   "
          f"decidable here: {r['can_decide']}")
    print(f"  {r['text']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=10000)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    a = ap.parse_args()
    print(f"threads={torch.get_num_threads()} numpy={np.__version__}")
    print_construction()
    print_ceiling()
    print_must_fire(a.n_rep, a.horizon)
    print_price()
    print_reroute()
    print_live(a.journal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
