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

For paired per-seed differences

    d_i = NRMSE_reference(seed_i) - NRMSE_arm(seed_i)

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

    E_t >= 20   ->  the arm beats the reference at alpha = 0.05, at ANY t
    symmetric process on `-d_i` for the reference direction
    neither crossing by phase end  ->  "undecided at evidence E_t = [value]"

TWO DIRECTIONS, TWO TESTS. Each direction is a one-sided test at alpha = 0.05,
which is how the rule is pre-registered. The statement "SOME direction crossed"
is a different event with a union bound of 0.10, and `calibrate` reports both
rates separately so that nobody has to infer which one a sentence is using.

======================================================================
PART 3: `B`, AND ITS PROVENANCE
======================================================================

`B` is the width of the CREDITED NRMSE range, not a fresh guess:

  * `negation_scope.nrmse` is `RMSE / std(y)`, a ratio of nonnegative
    quantities, so it is bounded below by 0 and is exactly 1.0 for the mean
    predictor (`scale/negation_scope.py:96-101`).
  * `M3_QUINTUPLE_PREREGISTERED_READING.md` section 5, last row: "any cell whose
    seed-mean NRMSE is at or above 1.0 ... is credited with nothing, whatever
    its contrast says."

A credited per-seed NRMSE therefore lies in `[0, 1)`, and the difference of two
of them satisfies `|d_i| < 1`. Hence `B = 1.0`.

THIS IS THE HONEST BOUND AND IT IS EXPENSIVE. A tighter `B` would make the
process grow faster, and the temptation is to set `B` from the observed spread
(the pre-registration measures a paired per-seed delta standard deviation of
`0.056889`). That would be fitting the instrument to the data it is about to
read: the bound must hold for every `d_i` the run could produce, not for the
ones it did produce. A `B` that a single seed exceeds destroys nonnegativity and
with it Ville's inequality, which is exactly the failure the null calibration in
Part 5 is built to catch.

======================================================================
PART 4: THE CEILING -- READ THIS BEFORE PRICING ANY RUN
======================================================================

Because every factor is at most `1 + lam <= 1.5`, the process has a hard ceiling
that no data can beat:

    max attainable E_t  =  (1/|Lambda|) * SUM_lam (1 + lam) ** t

At the pre-registered FIVE seeds this ceiling is `2.86`, and the loosest bound
(one arm at `lam = 1/2`) is `1.5 ** 5 = 7.59375`. Both are below the threshold
`20`. THE FIVE-SEED CONTRAST CANNOT BE DECIDED BY THIS PROCESS IN EITHER
DIRECTION, whatever the numbers turn out to be, and it cannot be decided before
`t = MIN_T_MIXTURE` seeds even if every seed lands at the maximal legal effect.

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
NRMSE_BAR = 1.0       #: M3 pre-registration section 5: at or above 1.0 -> no credit
B = NRMSE_BAR - NRMSE_FLOOR                #: 1.0 -- the a-priori bound on |d_i|

ALPHA = 0.05
THRESHOLD = 1.0 / ALPHA                    #: 20.0, per direction

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


MIN_T_MIXTURE = _min_t(max_attainable)             #: 11 -- the real instrument
MIN_T_SINGLE_ARM = _min_t(max_attainable_single_arm)   #: 8 -- the loose bound


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

    `d_i = NRMSE_ref(seed) - NRMSE_arm(seed)`, positive meaning `arm` wins.

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
    return [(s, got[s][ref] - got[s][arm]) for s in sorted(got)
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
    print(f"  E_t = mean_lam PROD_i (1 + lam*d_i/B),  d_i = NRMSE_ref - NRMSE_arm")
    print(f"  B = {B!r}   from the credited NRMSE range "
          f"[{NRMSE_FLOOR!r}, {NRMSE_BAR!r})")
    print(f"    negation_scope.nrmse >= 0 (ratio of nonnegative terms), "
          f"= 1.0 for the mean predictor")
    print(f"    M3_QUINTUPLE_PREREGISTERED_READING.md section 5: "
          f"seed-mean NRMSE >= 1.0 is credited with nothing")
    print(f"  Lambda = {LAMBDA_GRID!r}   |Lambda| = {len(LAMBDA_GRID)}")
    print(f"  alpha = {ALPHA!r}   threshold = {THRESHOLD!r} PER DIRECTION "
          f"(either-direction union bound = {2 * ALPHA!r})")
    print(f"  worst-case factor at |d| = B, lam = 1/2: "
          f"{1.0 + 0.5 * (-B) / B!r}  (> 0, so nonnegativity holds)")


def print_ceiling() -> None:
    print("\n=== THE CEILING: what no data can beat ===")
    print(f"{'t':>4} {'max mixture E_t':>18} {'max single-arm 1.5**t':>23} "
          f"{'>= 20?':>8}")
    for t in (5, 8, 10, 11, 12, 20):
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
    print(f"{'stream':>34} {'cross settled':>14} {'cross twin':>11} "
          f"{'either':>8} {'frac E>2':>9} {'max peak':>11} {'med t':>7} "
          f"{'expected':>26}")
    rows = [
        (NULL_RADEMACHER, False, f"both <= alpha={ALPHA}"),
        (NULL_GAUSSIAN, False, f"both <= alpha={ALPHA}"),
        (PLANTED_FLOOR, False, "settled crosses"),
        (PLANTED_LARGE, False, "settled crosses"),
        (NULL_RADEMACHER, True, "BROKEN: must exceed alpha"),
    ]
    verdicts = []
    for i, (spec, peek, want) in enumerate(rows):
        c = calibrate(spec, n_rep=n_rep, horizon=horizon, seed=200 + i,
                      peek=peek)
        label = spec.name + ("  [lambda PEEKS: broken]" if peek else "")
        mt = "-" if c["median_cross_t"] is None else str(c["median_cross_t"])
        print(f"{label:>34} {c['cross_settled']:>14.4f} {c['cross_twin']:>11.4f} "
              f"{c['cross_either']:>8.4f} {c['frac_above_2']:>9.4f} "
              f"{c['max_peak']:>11.4g} {mt:>7} {want:>26}")
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
          f"{2 * ALPHA!r}")
    print(f"  DIRECTION 2 (planted effect DOES cross): "
          f"{'PASS' if ok_planted else 'FAIL'}")
    seen = ("SEEN TO FIRE" if ok_broken else
            "DID NOT FIRE -> the null check measures nothing")
    print(f"  CONTROL (predictability deleted -> null breaks): {seen}")
    if not (ok_null and ok_planted and ok_broken):
        print("  THE CONSTRUCTION IS NOT CALIBRATED. Nothing anytime-valid may "
              "be claimed until the cause is found.")


def print_price(n_rep: int = 2000, horizon: int = 600) -> None:
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
    print(f"  Measured column: n_rep={n_rep} horizon={horizon}; a '-' means the "
          f"stream did not cross within the horizon.")


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
    print_live(a.journal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
