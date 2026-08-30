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


if __name__ == "__main__":
    demo()
