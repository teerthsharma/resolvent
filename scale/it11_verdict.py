"""The it.11 gate, computed over the population the rule names.

THE RULE, pre-registered in the v-main.3M script, iteration 11:

    region learnable iff >=1 cell < 1.0 with N=8 SEED CI excluding 1.0
    (bootstrap, B=1e4)

WHY THIS FILE EXISTS. Every cell in the it.8/9 capacity journals already ships a
`boot_lo`/`boot_hi` pair, and for `t*=8` at n=32768 that pair is
`[0.9641, 0.9814]` -- it excludes 1.0. A reader satisfying it.11 by checking the
shipped interval gets True and stops. The interval is real and correctly
computed, but `r10_capacity_sweep.py:101` builds it with
`bootstrap_ci(pe, y_eval, seed=seed)`: a resample of EVAL EXAMPLES inside ONE
trained model. It answers "is this model below the bar on this eval set". The
rule asks "does a randomly-seeded training run land below the bar". Those are
different populations, and at n=32768 the second quantity did not exist in the
journal at all -- `--seed-cap` restricts seeds beyond the first to smaller rungs,
so every large-rung cell was seed 0.

The proxy is biased in the flattering direction, which is why it survived: the
eval-example CI is 0.0173 wide where the measured seed-to-seed range at n=2048 is
0.0345 -- about half, and tighter means a lone crossing looks better established
than it is. Recorded as instance 16 in R10_MECHANISM.md.

TWO REFUSALS, both deliberate.

`verdict()` refuses to answer below N=8 rather than reporting a CI over the seeds
it happens to have. A gate that quietly weakens its own N is the vacuous-control
shape this campaign catalogues; the caller gets INSUFFICIENT and the seed count,
never a verdict computed on a standard nobody registered.

It also DEDUPLICATES BY SEED before counting. The journals carry bit-identical
duplicate rows -- two processes ran the same config, and at fixed thread count
this harness is deterministic, so seed 0 appears twice reading 0.9723723935 both
times. Counting rows instead of distinct seeds turns N=1 into N=2 at no cost and
in the direction that reaches the threshold sooner. The row count is the surface
proxy; the distinct seed count is the thing the rule means.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import math
import pathlib
import random
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Journal lines `cells()` could not parse. A seed count is an absence claim --
#: "this cell has 6 seeds" means "no seventh row exists" -- and a reader that
#: silently drops rows it cannot parse reports an absence it has not earned
#: (MISTAKES.md V-7, and V-16 for the unreadable case). `skipped()` makes the
#: loss countable so a caller can tell "6 seeds" from "6 seeds and 2 unreadable".
_SKIPPED: list[str] = []

BAR = 1.0
REQUIRED_SEEDS = 8
B = 10_000


@dataclasses.dataclass(frozen=True)
class Verdict:
    t_star: int
    n_train: int
    n_seeds: int
    mean: float | None
    lo: float | None
    hi: float | None
    reading: str

    def __str__(self) -> str:
        if self.mean is None:
            return (f"t*={self.t_star} n={self.n_train}: {self.reading} "
                    f"({self.n_seeds}/{REQUIRED_SEEDS} distinct seeds)")
        return (f"t*={self.t_star} n={self.n_train}: {self.reading} -- "
                f"mean {self.mean:.4f}, seed CI [{self.lo:.4f}, {self.hi:.4f}] "
                f"over N={self.n_seeds}, B={B:,}")


def cells(n_train: int | None = None, steps: int | None = None) -> list[dict]:
    """Every `t:cell` record across the capacity journals, optionally filtered.

    `steps` is a REQUIRED filter for any seed interval. The first draft of
    `verdict()` selected on (t*, n) alone and pooled every step rung, so a
    150-step cell and a 600-step cell sharing a seed read as one seed recorded
    twice with different values. It was accidentally correct at n=32768, where
    only steps=150 was ever run, and wrong at every rung that has more than one
    -- which is where the pooling would have widened a seed interval with an
    effect that is not seed variance at all.
    """
    _SKIPPED.clear()
    out = []
    for f in sorted(glob.glob(str(ROOT / "results" / "r10_it8_capacity_softmax_t*.jsonl"))):
        out.extend(_cells_in(pathlib.Path(f), n_train, steps))
    return out


def _cells_in(path: pathlib.Path, n_train: int | None,
              steps: int | None) -> list[dict]:
    """`t:cell` rows from one journal, recording the file if any line is unreadable.

    Split out of `cells()` so the unreadable-line path is directly testable. A
    must-fire that can only reach this branch through the real `results/`
    directory is a must-fire that never runs, since those journals parse clean.
    """
    out = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if str(path) not in _SKIPPED:
                _SKIPPED.append(str(path))
            continue
        if (r.get("t") == "cell"
                and (n_train is None or r.get("n_train") == n_train)
                and (steps is None or r.get("steps") == steps)):
            out.append(r)
    return out


def skipped() -> list[str]:
    """Journal files that had at least one unparseable line during the last read."""
    return list(_SKIPPED)


def by_seed(rows: list[dict]) -> dict[int, float]:
    """{seed: eval_nrmse} at ONE thread count, one entry per DISTINCT seed.

    A repeated seed is a determinism check, not a replicate. But this harness is
    deterministic only at FIXED thread count: the same (t*, n, steps, seed) run at
    threads=6 and threads=8 differs by up to 2.345e-3 from float reduction order
    (measured across all three cross-thread pairs in the journals: 2.345e-3,
    4.911e-4, 7.311e-6).

    So a seed interval must be built at a fixed thread count. Pooling thread
    counts would put a non-seed variance source inside an interval that claims to
    measure seed variance -- instance 16's defect, a CI over the wrong population,
    committed one field over. This picks the thread count carrying the most
    distinct seeds and reports what it dropped.

    An earlier draft RAISED on any same-seed disagreement, with a message saying
    the rows "differ in something the cell record does not name". The record names
    it: `threads`. That message was written before the thread effect was measured
    and contradicted the field the explanation later relied on. A genuine
    same-seed same-threads disagreement is still an anomaly and still raises.
    """
    by_threads: dict[int, dict[int, float]] = {}
    for r in rows:
        seen = by_threads.setdefault(r["threads"], {})
        s, v = r["seed"], r["eval_nrmse"]
        if s in seen and abs(seen[s] - v) > 1e-9:
            raise ValueError(
                f"seed {s} recorded twice at threads={r['threads']} with different "
                f"values ({seen[s]!r} vs {v!r}). At fixed thread count this harness "
                f"is deterministic, so these two rows differ in something no field "
                f"of the cell record names. Resolve it before computing an interval."
            )
        seen[s] = v
    if not by_threads:
        return {}
    best = max(by_threads, key=lambda t: len(by_threads[t]))
    return by_threads[best]


def thread_split(rows: list[dict]) -> dict[int, int]:
    """{threads: distinct seed count}, so a caller can see what `by_seed` dropped."""
    out: dict[int, set[int]] = {}
    for r in rows:
        out.setdefault(r["threads"], set()).add(r["seed"])
    return {t: len(s) for t, s in sorted(out.items())}


def seed_ci(values: list[float], b: int = B, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the MEAN over seeds. Resamples seeds, not examples."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(b))
    return means[int(0.025 * b)], means[int(0.975 * b)]


def verdict(t_star: int, n_train: int = 32768, steps: int = 150) -> Verdict:
    """it.11 applied at ONE (t*, n, steps) cell.

    INSUFFICIENT below N=8 -- never a verdict on an unregistered N.
    """
    vals = by_seed([c for c in cells(n_train, steps) if c["t_star"] == t_star])
    n = len(vals)
    if n < REQUIRED_SEEDS:
        return Verdict(t_star, n_train, n, None, None, None,
                       f"INSUFFICIENT -- it.11 requires N={REQUIRED_SEEDS} distinct "
                       f"seeds and this cell has {n}. No CI is reported, because a "
                       f"CI over {n} seeds is not the quantity the rule names.")
    xs = list(vals.values())
    lo, hi = seed_ci(xs)
    if hi < BAR:
        reading = "LEARNABLE"
    elif lo > BAR:
        reading = "NOT LEARNABLE"
    else:
        reading = "NO READING -- seed CI straddles the bar"
    return Verdict(t_star, n_train, n, statistics.fmean(xs), lo, hi, reading)


def demo() -> None:
    """Self-check: the gate refuses a short N, and its CI can land on either side."""
    # Must-fire: an N below the registered one is refused, not answered.
    #
    # THIS CHECK EXPIRED ONCE AND WAS CAUGHT. It read
    #     v = verdict(8)
    #     if v.n_seeds < REQUIRED_SEEDS: assert v.mean is None and ...
    # which exercised the refusal only while `t*=8` at n=32768 still had fewer
    # than eight seeds. When the eighth seed landed the branch stopped executing,
    # and tests/loop/test_phase1a_modules_are_bound.py proved the consequence by
    # disabling `if n < REQUIRED_SEEDS:` and watching demo() still exit 0 --
    # reporting a CI "over N=7" and printing "demo OK". A must-fire whose
    # reachability depends on transient data state has an expiry date, and this
    # one had passed it.
    #
    # Bound to a CONSTRUCTED cell instead so it cannot expire: `n_train=-1`
    # matches no journal row, so this verdict is over zero seeds by construction
    # and the refusal must fire on every run, forever, whatever the journals hold.
    empty = verdict(8, n_train=-1)
    assert empty.n_seeds == 0, f"the synthetic short cell found rows: {empty}"
    assert empty.mean is None and "INSUFFICIENT" in empty.reading, (
        "a short seed count must refuse, never report a CI on an unregistered N")

    # Must-fire: a sample clearly below the bar reads LEARNABLE; clearly above,
    # NOT LEARNABLE; straddling, NO READING. Without all three the reading
    # carries no information.
    lo, hi = seed_ci([0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.90, 0.89])
    assert hi < BAR, f"a sample at ~0.90 must clear the bar, got hi={hi}"
    lo, hi = seed_ci([1.10, 1.11, 1.09, 1.12, 1.10, 1.11, 1.10, 1.09])
    assert lo > BAR, f"a sample at ~1.10 must fail the bar, got lo={lo}"
    lo, hi = seed_ci([0.95, 1.05, 0.97, 1.03, 0.99, 1.01, 0.96, 1.04])
    assert lo < BAR < hi, f"a sample straddling 1.0 must read NO READING, got [{lo},{hi}]"

    # Must-fire: duplicate rows for one seed count ONCE.
    dup = [dict(seed=0, eval_nrmse=0.97, threads=12),
           dict(seed=0, eval_nrmse=0.97, threads=12)]
    assert len(by_seed(dup)) == 1, "duplicate rows for one seed inflated the seed count"

    # Must-fire: a genuine disagreement under one seed is raised, not averaged away.
    try:
        by_seed([dict(seed=0, eval_nrmse=0.97, threads=12),
                 dict(seed=0, eval_nrmse=0.98, threads=12)])
    except ValueError:
        pass
    else:
        raise AssertionError("two different values under one seed passed silently")

    # Must-fire: a row with no `threads` field must RAISE, not be silently
    # bucketed. The first version of this demo passed synthetic rows without it
    # and broke the moment by_seed started grouping on it -- a self-check that
    # did not exercise the shape its own module reads.
    try:
        by_seed([dict(seed=0, eval_nrmse=0.97)])
    except KeyError:
        pass
    else:
        raise AssertionError("a cell with no threads field was bucketed anyway")

    # MUST-FIRE, MISTAKES.md V-16: an unreadable row must be COUNTED, not
    # silently dropped. A seed count is an absence claim -- "this cell has 6
    # seeds" asserts no seventh row exists -- so a reader that quietly discards
    # what it cannot parse under-reports N in the direction that reads as "fewer
    # seeds exist" rather than "I could not read some".
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        bad = pathlib.Path(d) / "j.jsonl"
        good_row = json.dumps(dict(t="cell", t_star=2, n_train=1, steps=1,
                                   seed=0, eval_nrmse=0.5, threads=1))
        bad.write_text(good_row + "\n{ this is not json\n", encoding="utf-8")
        _SKIPPED.clear()
        got = _cells_in(bad, None, None)
        assert len(got) == 1, f"the readable row was lost: {got}"
        assert skipped() == [str(bad)], (
            f"an unparseable line was dropped without being counted: "
            f"skipped()={skipped()}. That is an absence the reader did not earn."
        )
    _SKIPPED.clear()

    # MUST-FIRE, found by the INSPECTOR: this demo passed on ZERO journals.
    # The print(verdict(...)) lines below assert nothing, so with no results/
    # directory the gate that decides "the region is learnable" exited 0 saying
    # "demo OK" -- it could not tell its journals from an empty folder. A
    # self-check that cannot distinguish having data from having none is the
    # vacuous-control shape this module was written to refuse.
    assert cells(), (
        "no cells parsed from results/r10_it8_capacity_softmax_t*.jsonl. The "
        "journals are missing or unreadable; this demo must fail rather than "
        "report OK over an empty directory.")
    known = verdict(2, 2048, 150)
    assert known.n_seeds == REQUIRED_SEEDS and known.reading == "LEARNABLE", (
        f"the it.11-compliant reference cell (t*=2, n=2048, steps=150) no longer "
        f"reads as expected: {known}. Either the journals changed or this reader "
        "did; both need a human before any verdict below is believed.")
    assert known.lo is not None and known.hi is not None
    assert abs(known.lo - 0.9462) < 5e-4 and abs(known.hi - 0.9591) < 5e-4, (
        f"the published it.11 interval [0.9462, 0.9591] no longer reproduces: "
        f"[{known.lo:.4f}, {known.hi:.4f}]")

    for t in (2, 8, 32):
        print(" ", verdict(t, 2048, 150))
    for t in (2, 8, 32):
        print(" ", verdict(t))
    print("demo OK: refuses a short N, counts seeds not rows, reads both sides of the bar")


# ---------------------------------------------------------------- TOST / parity
#
# THE THIRD REFUSAL. v-main.6 fixed the parity grammar: equality is established
# by two one-sided tests, never by a difference test that failed to reject.
# "p > 0.05, therefore equal" is the amateur's parity -- it confuses "no evidence
# of a difference" with "evidence of no difference", and at N=8 a difference test
# is underpowered enough that failing to reject is the DEFAULT outcome, not a
# finding. `refuse_difference_test_parity` exists to be called by anything that
# might be handed such an argument, and it raises rather than returning False.
#
# THE MARGIN FLOOR. `Delta_eq = 0.5 * sd(reference seeds)` is the pre-registered
# margin. It is bounded below by this harness's own arithmetic: `by_seed` above
# records a reduction-order drift of 2.345e-3 between thread counts, so a margin
# near that size declares "equivalent" over a window the float accumulation order
# can move a cell across by itself. `delta_eq` refuses below 2x that floor.

THREAD_FLOOR = 2.345e-3   # measured, three cross-thread pairs; see `by_seed`


@dataclasses.dataclass(frozen=True)
class TOST:
    n_arm: int
    n_ref: int
    diff: float
    lo90: float
    hi90: float
    delta_eq: float
    p_lower: float
    p_upper: float
    power: float
    reading: str

    def __str__(self) -> str:
        return (f"diff {self.diff:+.6f}  90% CI [{self.lo90:+.6f}, {self.hi90:+.6f}]  "
                f"Delta_eq +/-{self.delta_eq:.6f}  p_lo {self.p_lower:.4f} "
                f"p_hi {self.p_upper:.4f}  power {self.power:.3f}  {self.reading}")


def delta_eq(reference: list[float]) -> float:
    """Half the reference arm's own seed sd, refused if below the harness floor."""
    if len(reference) < 2:
        raise ValueError(
            f"Delta_eq needs a seed spread and got {len(reference)} value(s). "
            "A margin cannot be derived from a single run.")
    d = 0.5 * statistics.stdev(reference)
    if d < 2 * THREAD_FLOOR:
        raise ValueError(
            f"Delta_eq = {d:.6g} is below 2x the reduction-order floor "
            f"({2 * THREAD_FLOOR:.6g}). A margin this narrow would declare two arms "
            f"equivalent over a window that changing --threads moves a single cell "
            f"across ({THREAD_FLOOR:.6g}, measured). Widen the margin with a stated "
            "reason, or report that this cell cannot support an equivalence claim.")
    return d


def refuse_difference_test_parity(p_value: float, *, claim: str = "") -> None:
    """MUST-FIRE. Raises on any attempt to read a failed difference test as parity."""
    raise ValueError(
        f"REFUSED: a difference test with p={p_value} does not establish equality"
        f"{' -- ' + claim if claim else ''}. Failing to reject H0 is not evidence "
        "for H0; at N=8 it is the expected outcome whether or not the arms differ. "
        "Parity requires TOST: both one-sided tests rejecting at alpha, i.e. the "
        "90% CI of the contrast lying inside (-Delta_eq, +Delta_eq). Call `tost()`.")


def tost(arm: list[float], reference: list[float], *, alpha: float = 0.05,
         margin: float | None = None, sims: int = 4000, seed: int = 0) -> TOST:
    """Two one-sided Welch tests for equivalence of `arm` to `reference`.

    Power is estimated by simulation under a TRUE DIFFERENCE OF ZERO: given these
    two spreads and this N, how often would genuinely equivalent arms be correctly
    declared equivalent. That is the quantity Mars's standing attack needs, because
    the attack is "parity was reached by being unable to see" and its evidence is a
    design that would fail to certify even identical arms.

    Power is NOT evaluated at a true difference sitting on +Delta_eq. At the margin
    boundary a correctly-sized test rejects at about alpha by construction, so that
    number is near zero for any valid design and diagnoses nothing.
    """
    from scipy import stats
    #: Refuse a short N for the same reason `verdict` and `cap_verdict` do. An
    #: equivalence statement computed over fewer seeds than the round registered
    #: is a verdict on a standard nobody set, and TOST is the one place where a
    #: short N flatters: fewer seeds widen the interval, and a wide interval that
    #: happens to sit inside a wide margin still reads PARITY.
    for name, xs in (("arm", arm), ("reference", reference)):
        if len(xs) < REQUIRED_SEEDS:
            raise ValueError(
                f"TOST refuses: the {name} sample has {len(xs)} distinct seeds and "
                f"the round registers N={REQUIRED_SEEDS}. Report the seed count and "
                "the reason the cell is short, never an equivalence verdict over it.")
    d = delta_eq(reference) if margin is None else float(margin)
    diff = statistics.fmean(arm) - statistics.fmean(reference)
    # both one-sided Welch tests
    lo_t = stats.ttest_ind(arm, reference, equal_var=False, alternative="greater")
    hi_t = stats.ttest_ind(arm, reference, equal_var=False, alternative="less")
    sa, sr, na, nr = (statistics.stdev(arm), statistics.stdev(reference),
                      len(arm), len(reference))
    se = math.sqrt(sa * sa / na + sr * sr / nr)
    df = (se ** 4) / ((sa * sa / na) ** 2 / (na - 1) + (sr * sr / nr) ** 2 / (nr - 1))
    tcrit = stats.t.ppf(1 - alpha, df)
    lo90, hi90 = diff - tcrit * se, diff + tcrit * se
    # shift each sample so the contrast sits on the margin, then resample
    rng = random.Random(seed)
    hits = 0
    for _ in range(sims):
        a = [rng.choice(arm) for _ in range(na)]
        r = [rng.choice(reference) for _ in range(nr)]
        dd = statistics.fmean(a) - statistics.fmean(r) - diff
        s = math.sqrt(statistics.stdev(a) ** 2 / na + statistics.stdev(r) ** 2 / nr)
        if s == 0:
            continue
        if dd - tcrit * s > -d and dd + tcrit * s < d:
            hits += 1
    power = hits / sims
    inside = lo90 > -d and hi90 < d
    if inside and power >= 0.8:
        reading = "PARITY"
    elif inside:
        reading = f"PARITY CLAIMED BUT UNDERPOWERED (power {power:.2f} < 0.80)"
    elif lo90 > d or hi90 < -d:
        reading = "NOT EQUIVALENT -- CI disjoint from the margin"
    else:
        reading = "NO VERDICT -- CI wider than the margin; more seeds or a wider margin"
    return TOST(na, nr, diff, lo90, hi90, d, float(lo_t.pvalue), float(hi_t.pvalue),
                power, reading)


# ------------------------------------------------------------------ C-CAP / floor
#
# WHY THIS IS A DIFFERENT SHAPE FROM TOST. C-PAR asks whether two arms agree and
# needs a two-sample margin, which `delta_eq` shows is unreachable at N=8. C-CAP
# asks whether ONE arm's reading lies below a CONSTANT that is closed-form from
# the task -- no margin, no second sample, no equivalence. It is therefore
# decidable at the N this round actually runs, and it is the half of the ladder
# that distinguishes the architecture rather than matching it.


def hop_floor(t_star: int, hops: int = 1) -> float:
    """Best possible NRMSE for a model with a budget of `hops`: sqrt((t*-h)/t*).

    Closed form from `negation_scope.equilibrium_hop_reading`. The floor DROPS as
    the budget grows, so reading below `hop_floor(t, h)` is evidence of strictly
    more than `h` hops -- that is the direction the verdict below exploits.
    """
    return math.sqrt(max(0.0, float(t_star - hops)) / float(t_star))


@dataclasses.dataclass(frozen=True)
class Cap:
    t_star: int
    n_seeds: int
    mean: float | None
    lo: float | None
    hi: float | None
    floor1: float
    proven_hops: int
    h_hat: float | None
    reading: str

    def __str__(self) -> str:
        if self.mean is None:
            return f"t*={self.t_star}: {self.reading}"
        #: h_hat is None for any cell above the bar, where the hop count is
        #: undefined. An earlier draft formatted it unconditionally and raised a
        #: TypeError on exactly those cells -- the branch the demo asserted but
        #: never printed, which is M-12's shape committed one entry after filing it.
        hh = "n/a" if self.h_hat is None else f"{self.h_hat:.3f}"
        return (f"t*={self.t_star} mean {self.mean:.6f} CI [{self.lo:.6f}, {self.hi:.6f}] "
                f"floor_1 {self.floor1:.6f} h_hat {hh} -- {self.reading}")


def cap_verdict(t_star: int, values: list[float]) -> Cap:
    """C-CAP at one cell: is the seed CI entirely below the 1-hop floor.

    Refuses below N=8 for the same reason `verdict` does. `proven_hops` is the
    largest h whose floor the whole interval clears, so the claim it licenses is
    "strictly more than `proven_hops` hops", and 0 means nothing was proven.
    """
    floor1 = hop_floor(t_star, 1)
    n = len(values)
    if n < REQUIRED_SEEDS:
        return Cap(t_star, n, None, None, None, floor1, 0, None,
                   f"INSUFFICIENT -- C-CAP requires N={REQUIRED_SEEDS} distinct seeds "
                   f"and this cell has {n}.")
    mean = statistics.fmean(values)
    lo, hi = seed_ci(values)
    proven = 0
    for h in range(1, t_star + 1):
        if hi < hop_floor(t_star, h):
            proven = h
        else:
            break
    #: h_hat inverts the floor at the observed reading. It is UNDEFINED above
    #: NRMSE 1.0, where it goes negative; a cell that did not clear the bar has no
    #: hop count, and reporting one there would read a failure as a mechanism.
    h_hat = t_star * (1.0 - mean ** 2) if mean < BAR else None
    if mean >= BAR:
        reading = "NO READING -- did not clear the bar; C-CAP not applicable"
    elif proven >= 1:
        reading = (f"PROVEN MULTI-HOP -- CI clears floor_{proven} "
                   f"({hop_floor(t_star, proven):.6f}); strictly more than {proven} hop(s)")
    else:
        reading = "consistent with 1 hop -- CI does not clear floor_1"
    return Cap(t_star, n, mean, lo, hi, floor1, proven, h_hat, reading)


def demo_cap() -> None:
    #: a planted arm sitting well below floor_1 must be PROVEN MULTI-HOP
    strong = [0.50, 0.51, 0.49, 0.52, 0.48, 0.505, 0.495, 0.515]      # t*=2 floor_1 = 0.7071
    r = cap_verdict(2, strong)
    assert r.proven_hops >= 1 and "PROVEN MULTI-HOP" in r.reading, r
    #: the measured softmax cell must NOT be proven -- it reads 0.9524 against 0.7071
    soft = [0.9401595494039541, 0.9451389005879479, 0.9456877509378661,
            0.9605261424068585, 0.9690874072329486, 0.9714324268559540,
            0.9500000000000000, 0.9550000000000000]
    r2 = cap_verdict(2, soft)
    assert r2.proven_hops == 0 and "consistent with 1 hop" in r2.reading, r2
    #: a cell above the bar has no hop count at all, not a negative one
    dead = [1.10 + 0.01 * i for i in range(8)]
    r3 = cap_verdict(8, dead)
    assert r3.h_hat is None and "NO READING" in r3.reading, r3
    assert "h_hat n/a" in str(r3), f"the above-bar branch must render: {r3!r}"
    #: below N=8 it refuses rather than reporting a CI over the seeds it has
    r4 = cap_verdict(2, strong[:3])
    assert "INSUFFICIENT" in r4.reading, r4
    print("  planted multi-hop :", r)
    print("  measured softmax  :", r2)
    print("  above the bar     :", r3)
    print("cap demo OK: proves a real crossing, refuses a short N, gives no hop count "
          "above the bar")


def demo_tost() -> None:
    #: identical arms must read PARITY; the margin must clear the harness floor
    ref = [0.9401595494039541, 0.9451389005879479, 0.9456877509378661,
           0.9605261424068585, 0.9690874072329486, 0.9714324268559540,
           0.9500000000000000, 0.9550000000000000]
    d = delta_eq(ref)
    assert d > 2 * THREAD_FLOOR, d
    #: THE REACHABILITY FACT, asserted so it cannot quietly stop being true.
    #: At Delta_eq = 0.5*sd, the 90% CI half-width for a two-sample contrast is
    #: t(.95, 2N-2)*sqrt(2/N) in units of sd -- 0.8807 at N=8, against a margin of
    #: 0.5. So at N=8 even two BIT-IDENTICAL arms return NO VERDICT: the interval
    #: is wider than the window it must fit inside. Parity first becomes reachable
    #: at N=23. A run that reports PARITY at N=8 has not measured equivalence.
    same = tost([v + 1e-6 for v in ref], ref)
    assert same.reading.startswith("NO VERDICT"), (
        f"identical arms at N=8 should be undecidable at Delta_eq=0.5*sd, got: {same}")
    #: Reachability is not the same as adequacy. The CI first fits inside the
    #: margin at N=23, but TOST power at a true difference of zero only clears
    #: 0.80 at N=70 -- 8.8x the pre-registered N=8. Between those two points a run
    #: can report an interval inside the margin while being unable to certify even
    #: identical arms, which is exactly the underpowered-parity shape Mars attacks.
    big = ref * 9                      # N=72, same spread
    wide = tost([v + 1e-6 for v in big], big, sims=2000)
    assert wide.reading == "PARITY", (
        f"identical arms at N={len(big)} should reach PARITY, got: {wide}")
    mid = tost([v + 1e-6 for v in ref * 3], ref * 3, sims=2000)   # N=24
    assert "UNDERPOWERED" in mid.reading, (
        f"N=24 should be inside-margin but underpowered, got: {mid}")
    #: a shifted arm, far outside the margin, must NOT read parity
    far = tost([v + 0.10 for v in ref], ref)
    assert far.reading.startswith("NOT EQUIVALENT"), far
    #: a margin below the reduction-order floor is refused, not narrowed
    tight = [0.5000000, 0.5000010, 0.5000020, 0.5000030,
             0.5000040, 0.5000050, 0.5000060, 0.5000070]
    try:
        delta_eq(tight)
        raise AssertionError("delta_eq accepted a margin below the thread floor")
    except ValueError as e:
        assert "reduction-order floor" in str(e), e
    #: a short N is refused rather than reported
    try:
        tost(ref[:3], ref)
        raise AssertionError("TOST accepted a 3-seed arm")
    except ValueError as e:
        assert "TOST refuses" in str(e), e
    #: the must-fire: a failed difference test may never be read as parity
    try:
        refuse_difference_test_parity(0.42, claim="arms look the same")
        raise AssertionError("difference-test parity was not refused")
    except ValueError as e:
        assert "REFUSED" in str(e), e
    print("  identical arms, N=8  :", same)
    print("  identical arms, N=24 :", mid)
    print("  identical arms, N=72 :", wide)
    print("  shifted arms   :", far)
    print("tost demo OK: refuses sub-floor margins, refuses difference-test parity, "
          "reports achieved power")


if __name__ == "__main__":
    demo()
    demo_tost()
    demo_cap()
