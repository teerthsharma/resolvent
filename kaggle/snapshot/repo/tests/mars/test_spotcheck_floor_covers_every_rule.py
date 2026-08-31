"""Does the stratified draw deliver the rule coverage it PRINTS?

`scale/spotcheck_draw2.py:141-142` prints, for the sheet it is about to hand SATURN:

    every rule sampled: True  => P(detect a wholly-wrong rule) = 1.0000 for all 61 rules

and its docstring at :24 makes the same claim for the 46-rule sheet it was designed on.
MARS wrote both. Both are false, and this file is the arithmetic.

The floor runs over STRATA, not over RULES. `build()` pools every singleton rule --
every rule holding exactly one KEEP row -- into ONE stratum, and `allocate()` then
gives that one stratum ONE floor draw plus its proportional share. At f823b02 the pool
is 44 distinct rules over 44 rows drawing 3. So 41 of the 61 rules receive zero draws
and are sampled with probability 3/44 = 0.0682, not 1.0000. `all(n >= 1 for n in draws)`
is True over the 18 strata and the print interpolates `len(by_rule)` = 61 next to it.

This is the axis the stratified draw was ADOPTED on. The uniform draw was rejected in
that same docstring for leaving "39 rules over 151 rows at zero draws". Measured at the
matched revision 06a180c, the stratified draw leaves 28 rules at zero draws. It cut the
unsampled ROW count 151 -> 28, which is real and is the gain to keep. It did not cut the
unsampled RULE count to 0; it cut it 39 -> 28.

TARGET A, tested here and REFUTED. The floor is not a treadmill. Across SATURN's 7
amendments the rule count went 46 -> 61 (+15) and the STRATUM count went 17 -> 18 (+1),
because 14 of the 15 minted rules were singletons and singletons pool for free. The floor
budget is 18 and grows ~1 per 7 amendments; it does not converge on the 314-row sheet.

THE ROUTE, executable below. The advertised guarantee's price is the RULE count, not the
stratum count: stop pooling, and budget 61 gives every rule its floor draw. A singleton
stratum is EXHAUSTED by its one draw, so budget 61 buys P=1.0000 over all 61 rules and
complete verification of the 44 singleton rows. 61/314 = 19.4% of the sheet.
"""
from __future__ import annotations

import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale.spotcheck_draw2 import allocate, build, rows, rule_key, sheet_at  # noqa: E402

PINNED = "f823b02"          # the revision the shipped draw is pinned to
DESIGNED_ON = "06a180c"     # the revision MARS measured when he wrote the 1.0000 claim


def rule_sizes(rev: str) -> collections.Counter:
    """rule_key -> KEEP rows carrying it, at a pinned revision."""
    return collections.Counter(
        rule_key(r) for _p, _c, d, r in rows(sheet_at(rev)) if d == "KEEP"
    )


def test_the_shipped_draw_samples_every_rule_its_own_output_says_it_samples():
    """RED. The headline. 41 of 61 rules get zero draws behind a printed 1.0000."""
    all_rows, by_rule, strata, draws, keep, _attic = build(PINNED)
    by_path = {p: rule_key(r) for p, _c, d, r in all_rows if d == "KEEP"}
    touched = {by_path[p] for p in keep}
    missed = set(by_rule) - touched
    pool = [r for r, v in by_rule.items() if len(v) == 1]
    assert not missed, (
        f"{len(missed)} of {len(by_rule)} classing rules get ZERO draws, over "
        f"{sum(len(by_rule[r]) for r in missed)} of "
        f"{sum(len(v) for v in by_rule.values())} KEEP rows. "
        f"scale/spotcheck_draw2.py:141 prints 'every rule sampled: True => "
        f"P(detect a wholly-wrong rule) = 1.0000 for all {len(by_rule)} rules'; the floor "
        f"runs over {len(strata)} STRATA, and the {len(pool)} singleton rules are pooled "
        f"into one stratum drawing {draws[-1]}, so each is sampled with p="
        f"{draws[-1]}/{len(pool)}={draws[-1]/len(pool):.4f}, not 1.0000"
    )


def test_the_floor_budget_does_not_grow_with_the_rules_the_amendments_mint():
    """GREEN. Target A refuted: 7 amendments minted +15 rules and +1 stratum."""
    seen = {}
    for rev in (DESIGNED_ON, PINNED):
        sizes = rule_sizes(rev)
        multi = sum(1 for n in sizes.values() if n >= 2)
        singles = sum(1 for n in sizes.values() if n == 1)
        seen[rev] = (len(sizes), multi + (1 if singles else 0))
    (r0, s0), (r1, s1) = seen[DESIGNED_ON], seen[PINNED]
    assert (r0, s0) == (46, 17) and (r1, s1) == (61, 18), f"revision drift: {seen}"
    assert s1 - s0 < r1 - r0, (
        f"the floor tracks the rule count: rules +{r1 - r0}, strata +{s1 - s0}"
    )


def test_the_route_prices_the_advertised_guarantee_at_the_rule_count():
    """GREEN. Iteration 4 executes this, no design work left: unpool, budget 61."""
    _all, by_rule, _strata, _draws, _k, _a = build(PINNED)
    unpooled = sorted((sorted(v) for v in by_rule.values()), key=lambda v: (-len(v), v[0]))
    price = len(unpooled)
    assert price == 61, f"price of the guarantee at {PINNED} is {price}"
    draws = allocate(unpooled, price)
    assert all(n >= 1 for n in draws), "unpooled floor still starves a rule"
    exhausted = [s for s, n in zip(unpooled, draws) if n == len(s)]
    assert len(exhausted) == 44, f"{len(exhausted)} strata exhausted, want the 44 singletons"
    try:
        allocate(unpooled, 30)
    except ValueError as e:
        assert "below the floor" in str(e)
    else:
        raise AssertionError("budget 30 must not satisfy a 61-stratum floor")
