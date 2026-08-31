"""The label's variance law is stated twice, incompatibly, and the round rests on it.

THE DEFECT. `scale/negation_scope.py` states the law of the label six times, and
one of them disagrees with the other five:

    :367              "the label is `N(0, t_star + 1)` EXACTLY"
    :419              "so it is `N(0, t*)` exactly. Hence
                       flipper_dependence = 2 E|N(0,1)| / E|N(0, t*)| = 2 / sqrt(t*)."
    :421 :528 :649 :651   `N(0, t*)` again, four more times

MARS filed this at iteration 13 as two docstrings disagreeing. Running the check
measured the split and it is **5 to 1**: `:367` is a single stale sentence, not
half of a genuine ambiguity, which settles the repair direction without needing
the measurements to arbitrate.

They cannot both hold. Every executable line takes `Var(y) = t*`, and so does
every reference line the round's grid is scored against -- the 1-hop ceiling
`sqrt((t*-1)/t*)` and the calibration band `2/sqrt(t*)`. The mechanism: the
operator is nilpotent of index `t*+1`, so the variance was read off the
nilpotency index, but `b[s-1] = 0` -- added to pass the 0-step RED gate -- kills
the `m=0` term and drops the count to `t*`. The `:367` docstring was never
decremented.

MEASURED, SATURN it.13, K=200 draws per block: mean Var(y) = 2.002719 /
7.984942 / 32.069818 at `t*` = 2 / 8 / 32. At `t*=32` the CI covers `t*` and
excludes `t*+1` by 9.8 half-widths. `Var(y) = t*` is settled by measurement, so
the five `t*` sites are right and `:367` is the one that must move.

WHY THIS IS A GUARD AND NOT A TYPO REPORT. No number in the shipped grid moves:
the code already computes `t*`. What is at risk is the REPAIR. The shipped
`t*=2` journal records `flipper_dependence = 1.4012436552018552`:

    |1.4012436552 - 2/sqrt(2)| = 0.0129699   -> passes flipper_tol = 0.05
    |1.4012436552 - 2/sqrt(3)| = 0.2465431   -> 4.93x the tolerance, ABORT

So making the code agree with its own `:367` docstring aborts the `t*=2` block
with "calibration bar failed" -- and `t*=2` at n=2048 is the only it.11-compliant
LEARNABLE verdict in the round, the single cell the region-is-learnable verdict stands on.

BUT THE FAILURE IS LOUD, NOT SILENT, AND THIS FILE FIRST SAID OTHERWISE. MARS
filed the attack claiming
`tests/cameron/test_m3_etasks.py::test_the_chain_flipper_dependence_is_its_closed_form`
would stay green through the repair, and this docstring repeated it. Checked:
that test is parametrized over `t* in [1, 2, 8, 32, 63]`, passes 5/5 as shipped,
and its first assertion is

    assert abs(want - 2.0 / math.sqrt(t_star)) < 1e-12

a HARD-CODED `2/sqrt(t*)` at 1e-12, so any change to the closed form fails it
immediately. Its second assertion compares that form against
`cal["flipper_dependence"]`, which `negation_scope.py:1460` computes as
`moved / scale` -- MEASURED from data, so it does not move when a constant does.
The repair therefore fails twice in that test AND aborts the sweep. It is
self-announcing.

What remains is still worth guarding: the repair is *written down as correct* in
`:367` and in `test_m3_etasks.py` at `:38` and `:278`, so a maintainer has three
documented invitations to attempt it, and will only discover the cost by breaking
the round's central block and reading the failure. This guard makes the
inconsistency visible before that, not after.

WHAT THIS TEST CLAIMS. Not that `t*` is the right law -- the measurements say it
is, and SATURN's it.13 binding puts a CI on it. It claims the law must be stated
ONCE, so that the next person to touch it cannot find written permission for the
change that breaks the round.

THE ROUTE. Two edits, neither of which moves a number: correct the `:367`
docstring to `N(0, t*)` with a sentence saying why (`b[s-1] = 0` kills the `m=0`
term, so the nilpotency index `t*+1` overcounts the drivers by one), and give
`test_the_chain_flipper_dependence_is_its_closed_form` a docstring deriving the
form it actually asserts.
"""
from __future__ import annotations

import math
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scale" / "negation_scope.py"

#: Any prose statement of the label's distribution. Deliberately matches BOTH
#: spellings of the parameter (`t_star`, `t*`) and both laws, because a pattern
#: that could only find the form it expects would report agreement by being
#: unable to see the disagreement -- MISTAKES.md V-7.
VARIANCE_LAW = re.compile(r"N\(0,\s*(t_star|t\*)\s*(\+\s*1)?\s*\)")

#: Words that turn a MENTION of the law into something other than an ASSERTION of
#: it. Added after a planted negative (MISTAKES.md V-15) fired this guard on the
#: exact repair its own THE ROUTE section prescribes: a corrected docstring that
#: names the form it is correcting -- "N(0, t*) exactly, NOT N(0, t_star + 1)" --
#: matched the pattern and was condemned. A rule that condemns fails by firing on
#: the innocent, and the innocent here is the fix.
NOT_AN_ASSERTION = re.compile(
    r"\b(not|never|rather than|instead of|corrected from|was|no longer|"
    r"incorrectly|wrongly|mistakenly)\b", re.I)


def stated_laws() -> list[tuple[int, str, bool]]:
    """(line number, matched text, is_t_star_plus_one) for every ASSERTION found.

    A line carrying a negation or correction marker before the match is a mention,
    not an assertion, and is skipped -- see NOT_AN_ASSERTION.
    """
    out = []
    for i, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines(), 1):
        for m in VARIANCE_LAW.finditer(line):
            before = line[:m.start()]
            if NOT_AN_ASSERTION.search(before):
                continue
            out.append((i, m.group(0), m.group(2) is not None))
    return out


def test_the_pattern_finds_statements_at_all():
    """Must-fire for this file's own reader. A pattern matching nothing would make
    the disagreement check below pass by vacuity, which is the shape this whole
    round catalogues."""
    laws = stated_laws()
    assert len(laws) >= 2, (
        f"the reader found {len(laws)} statements of the label's law in "
        f"{SOURCE.name}; the pattern is broken, not the source"
    )


def test_the_pattern_can_see_both_forms():
    """Must-fire, second half. A pattern blind to one form would report unanimity."""
    assert VARIANCE_LAW.search("the label is `N(0, t_star + 1)` EXACTLY")
    assert VARIANCE_LAW.search("so it is `N(0, t*)` exactly")
    plus_one = VARIANCE_LAW.search("N(0, t_star + 1)")
    plain = VARIANCE_LAW.search("N(0, t*)")
    assert plus_one is not None and plus_one.group(2) is not None
    assert plain is not None and plain.group(2) is None


def test_the_variance_law_is_stated_consistently():
    """THE DEFECT. Every prose statement of the label's law must agree.

    RED while `:367` says `t_star + 1` and `:419` says `t*`. Green once the law
    is stated once, in one form, whichever the measurements support.
    """
    laws = stated_laws()
    plus_one = [(n, t) for n, t, p in laws if p]
    plain = [(n, t) for n, t, p in laws if not p]
    assert not (plus_one and plain), (
        f"{SOURCE.name} states the label's variance law both ways.\n"
        f"  as t*+1 at lines {[n for n, _ in plus_one]}: {[t for _, t in plus_one]}\n"
        f"  as t*   at lines {[n for n, _ in plain]}: {[t for _, t in plain]}\n"
        "Both passages claim exactness and they cannot both hold. Every executable "
        "line takes t*, and so does every reference line the grid is scored "
        "against. This is not cosmetic: making the code match the t*+1 statement "
        "moves flipper_dependence from 2/sqrt(2)=1.4142 to 2/sqrt(3)=1.1547, which "
        "is 0.2465 from the shipped 1.4012436552 against flipper_tol=0.05, and "
        "aborts the t*=2 block -- the only it.11-compliant LEARNABLE verdict in the round."
    )


#: The value the shipped t*=2 journal records, and the tolerance it is held to.
SHIPPED_FLIPPER = 1.4012436552018552
FLIPPER_TOL = 0.05


@pytest.mark.parametrize("t_star", [2])
def test_the_shipped_calibration_only_survives_one_of_the_two_laws(t_star: int):
    """The arithmetic that makes the inconsistency load-bearing rather than cosmetic.

    This test is GREEN and stays green. It exists so the cost of the repair is
    recorded as an executable fact next to the guard, instead of living only in a
    round record nobody reads before editing a docstring.
    """
    under_t = abs(SHIPPED_FLIPPER - 2 / math.sqrt(t_star))
    under_t_plus_1 = abs(SHIPPED_FLIPPER - 2 / math.sqrt(t_star + 1))
    assert under_t < FLIPPER_TOL, (
        f"the shipped flipper_dependence does not even satisfy 2/sqrt(t*): "
        f"gap {under_t:.7f} against tol {FLIPPER_TOL}"
    )
    assert under_t_plus_1 > FLIPPER_TOL, (
        "the two laws are indistinguishable at this tolerance, so the trap "
        "described in this file's docstring does not exist and the guard above "
        "is guarding nothing"
    )


def test_the_guard_does_not_condemn_a_correction(tmp_path):
    """PLANTED NEGATIVE, MISTAKES.md V-15. A rule that CONDEMNS fails by firing on
    the innocent, and rule 5's planted positive cannot catch that: every positive
    it fires on reads as a success.

    The innocent here is the repair this file's own THE ROUTE section asks for. A
    corrected docstring that names the form it corrects mentions `N(0, t*+1)`
    without asserting it. Before NOT_AN_ASSERTION existed, both lines below
    tripped the guard -- it would have condemned its own fix.
    """
    src = tmp_path / "negation_scope.py"
    src.write_text(
        "the label is `N(0, t*)` exactly -- NOT `N(0, t_star + 1)`, which\n"
        "overcounts the drivers by one\n"
        "corrected from `N(0, t_star + 1)`: b[s-1]=0 kills the m=0 term\n",
        encoding="utf-8")
    global SOURCE
    saved, SOURCE = SOURCE, src
    try:
        laws = stated_laws()
    finally:
        SOURCE = saved
    assert [t for _n, t, p in laws if p] == [], (
        f"the guard read a CORRECTION as an assertion of the wrong law: {laws}. "
        "It would condemn the repair its own docstring prescribes."
    )
    assert len(laws) == 1 and not laws[0][2], (
        f"the correction's own assertion of the RIGHT law was lost: {laws}. "
        "Excluding too much makes this guard vacuous (V-7) -- it must still see "
        "the form the corrected text does assert."
    )


def test_the_exclusion_did_not_make_the_guard_vacuous(tmp_path):
    """MUST-FIRE after the V-15 repair: the exclusion must not swallow a real
    assertion of the wrong law.

    THIS CHECK EXPIRED ONCE, THE WAY MISTAKES.md V-16's sibling defect does. It
    read `assert [n for n, _t, p in laws if p] == [367]` -- pinned to the LIVE
    defect at `:367`. That was correct while the defect existed. When `:367` was
    repaired at it.16 the list went empty and the assertion failed, not because
    the exclusion over-reached but because its premise had been fixed. A must-fire
    whose premise is "the bug is still there" cannot survive the bug being fixed,
    and it fails in the direction that looks like a regression.

    Bound to a PLANTED assertion in a temp file instead, so it holds whatever the
    real source says: a bare statement of the wrong law with no correction marker
    must still be seen.
    """
    src = tmp_path / "negation_scope.py"
    src.write_text("\n".join([
        "the label is `N(0, t_star + 1)` EXACTLY, which puts it in closed form",
        "and a second line saying `N(0, t*)` so both forms are present",
        "",
    ]), encoding="utf-8")
    global SOURCE
    saved, SOURCE = SOURCE, src
    try:
        laws = stated_laws()
    finally:
        SOURCE = saved
    plus_one = [(n, t) for n, t, p in laws if p]
    assert plus_one == [(1, "N(0, t_star + 1)")], (
        f"a bare assertion of the wrong law was not seen: {laws}. The "
        "NOT_AN_ASSERTION exclusion has over-reached and the guard can no longer "
        "catch the defect it exists for."
    )
