"""Is the iteration-2 spot-check draw powered against the kind of error this repo ACTUALLY makes?

`scale/spotcheck_draw.py` states its own limit in its docstring: a uniform draw is
powered against errors spread across the sheet, and "if the sheet's errors are
CONCENTRATED ... a uniform 10 can miss the cluster entirely. That probability is
MARS's iteration-2 calculation and it is not answered here."

This file answers it. The cluster hypothesis is not a prior -- it is iteration 1's
MEASURED error. P1' condemned 27 rows; 12 of them were doing real work; all 12 were
`def test_claim_*` files; and on the shipped sheet those same 12 are a single rule
stratum, reason cell "REFUTATION INSTRUMENT: defines test_claim_* functions".

So the measured error mode here is RULE-GENERATED: one classing rule misfires and
every row it touched moves together. The draw samples PATHS uniformly. Rules are not
uniform over paths, so path-uniform sampling does not sample rules uniformly.

Three of these four are RED against the shipped uniform draw. They are the bar the
iteration-3 stratified draw has to clear; `stratified_allocation()` below is that
draw, executable, so iteration 3 needs no further design work.
"""
from __future__ import annotations

import collections
import pathlib
import re
import subprocess
import sys
from math import comb

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale.spotcheck_draw import N_KEEP, SEED, draw, rows  # noqa: E402

MEASURED_CLUSTER = 12          # P1' misfired on 12 rows at once; see MISTAKES.md / AUDIT.md P1
MISS_BAR = 0.10                # a draw may miss the measured cluster at most 1 time in 10


def rule_key(reason: str) -> str:
    """The classing MECHANISM, with row-specific counts and paths normalised out.

    ponytail: regex normalisation, not a parse. It is stable because the reason cells
    are generated from templates; if AUDIT.md stops being template-generated this
    needs a real parse.
    """
    s = re.sub(r"`[^`]*`", "@", reason)
    s = re.sub(r"\d+", "N", s)
    return re.sub(r"\s+", " ", s).strip()


def keep_rows() -> list[tuple[str, str]]:
    """(path, rule_key) for every KEEP row. Reuses spotcheck_draw's own parser."""
    AUDIT = ROOT / "AUDIT.md"
    ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|")
    out = []
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7 or cells[5] != "KEEP":
            continue
        out.append((m.group(1), rule_key(cells[6])))
    return out


def p_hit(c: int, n: int, k: int) -> float:
    """P(a uniform draw of k from n contains >=1 of a cluster of size c). Hypergeometric."""
    if k <= 0:
        return 0.0
    if c >= n or n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)


def stratified_allocation(budget: int = 30) -> list[tuple[int, int]]:
    """THE ITERATION-3 REPLACEMENT DRAW. Returns [(stratum_size, draws), ...].

    Floor of 1 on every rule stratum holding >=2 rows, singletons pooled into one
    stratum, remainder allocated proportional to stratum size. The floor is the whole
    point: it makes P(detect a wholly-wrong rule) = 1 for every rule, which no uniform
    draw buys at any budget below the full sheet.
    """
    sizes = collections.Counter(k for _p, k in keep_rows())
    multi = sorted((v for v in sizes.values() if v >= 2), reverse=True)
    strata = multi + [sum(1 for v in sizes.values() if v == 1)]   # pooled singletons last
    k = [1] * len(strata)
    rem = budget - sum(k)
    if rem < 0:
        raise ValueError(f"budget {budget} below the floor of {len(strata)} strata")
    want = [rem * n / sum(strata) for n in strata]
    extra = [int(w) for w in want]
    while sum(extra) < rem:
        i = max(range(len(strata)), key=lambda j: want[j] - extra[j])
        extra[i] += 1
    return [(n, min(a + b, n)) for n, a, b in zip(strata, k, extra)]


@pytest.fixture(scope="module")
def drawn() -> list[str]:
    keep, _attic, _nk, _na = draw([(p, c, d) for p, c, d in rows()])
    return keep


def test_the_measured_error_cluster_is_one_rule_stratum_of_exactly_twelve():
    """GROUNDING (expected GREEN). The cluster is measured, not assumed."""
    sheet = {p for p, k in keep_rows() if "REFUTATION INSTRUMENT" in k}
    tracked = subprocess.run(
        ["git", "ls-files", "--", "tests/*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    # Scoped to the census population. Files added AFTER AUDIT.md was written are not
    # on the sheet and cannot be classed by it -- measured: tests/saturn/
    # test_census_does_not_attic_refutation_instruments.py is one such, staged during
    # iteration 2. Comparing against the whole tree would test AUDIT.md's freshness,
    # which is a different claim than the one this file makes.
    on_sheet = {p for p, _k in keep_rows()}
    grepped = {
        f for f in tracked
        if f in on_sheet
        and "def test_claim_" in (ROOT / f).read_text(encoding="utf-8", errors="ignore")
    }
    assert len(sheet) == MEASURED_CLUSTER, f"sheet's REFUTATION INSTRUMENT stratum is {len(sheet)}"
    assert sheet == grepped, f"stratum != grep set; sym-diff {sheet ^ grepped}"


def test_the_draw_samples_the_cluster_that_iteration_1_already_misclassed(drawn):
    """RED. The one cluster with a measured failure history gets zero verification."""
    cluster = {p for p, k in keep_rows() if "REFUTATION INSTRUMENT" in k}
    hit = sorted(cluster & set(drawn))
    assert hit, (
        f"the draw takes {len(hit)} of the {len(cluster)} rows that P1' already got wrong once. "
        f"P(a uniform {N_KEEP} of 307 misses all {len(cluster)}) = "
        f"{1 - p_hit(len(cluster), 307, N_KEEP):.4f} -- this is the expected outcome, not bad luck"
    )


def test_every_rule_stratum_holding_more_than_one_row_is_sampled(drawn):
    """RED. A rule the draw never samples is a rule the draw cannot falsify."""
    rows_ = keep_rows()
    sizes = collections.Counter(k for _p, k in rows_)
    by_path = dict(rows_)
    touched = {by_path[p] for p in drawn}
    missed = {k: n for k, n in sizes.items() if n >= 2 and k not in touched}
    assert not missed, (
        f"{len(missed)} of {sum(1 for n in sizes.values() if n >= 2)} multi-row rule strata "
        f"get zero draws, covering {sum(missed.values())} of {len(rows_)} KEEP rows"
    )


def test_the_miss_probability_against_the_measured_cluster_clears_the_bar():
    """RED. The headline number."""
    miss = 1 - p_hit(MEASURED_CLUSTER, 307, N_KEEP)
    assert miss <= MISS_BAR, (
        f"seed={SEED} uniform {N_KEEP}/307 misses a {MEASURED_CLUSTER}-row cluster with "
        f"p={miss:.4f}, bar is {MISS_BAR}. A uniform draw needs k=67 to reach 95% on c=12. "
        f"Replacement: stratified_allocation() -- {stratified_allocation(30)}"
    )


def demo() -> None:
    """Self-check: the allocation closes on its budget and gives every stratum a draw."""
    for b in (30, 40, 60):
        a = stratified_allocation(b)
        assert sum(k for _n, k in a) == b, f"allocation does not close on budget {b}"
        assert all(k >= 1 for _n, k in a), "a stratum got zero draws"
        assert all(k <= n for n, k in a), "drew more rows than a stratum holds"
    a = stratified_allocation(30)
    worst = min(p_hit(MEASURED_CLUSTER, n, k) for n, k in a if n >= MEASURED_CLUSTER)
    print(f"demo OK: {len(a)} strata, budget 30 -> {a}")
    print(f"  P(detect a wholly-wrong rule) = 1.0000 for every stratum")
    print(f"  P(detect c=12 in its own rule stratum) = 1.0000")
    print(f"  P(detect c=12 placed in the WORST stratum) = {worst:.4f}")


if __name__ == "__main__":
    demo()
