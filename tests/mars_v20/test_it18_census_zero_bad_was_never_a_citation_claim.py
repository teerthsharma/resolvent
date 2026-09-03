"""MARS it.18 STRIKE 3 -- "re-scoped" is the wrong verb; the claim was withdrawn.

JUPITER's it.15 checker returns `123/123, 0 bad`. The Inspector's spot-check
returns `8 of 20 do not land`. JUPITER reconciled the two as *resolvability* vs
*landing* and RE-SCOPED rather than WITHDREW.

Re-scoping is honest when the original number measured a real, narrower property.
Here it did not. `tests/jupiter/test_v20_r15_it14_theory_table.py::bad_citations`
appends to `bad` on exactly two conditions -- the file is missing, or the line
number exceeds the file's length. That is a FILE-EXISTS check and a LINE-COUNT
bound. It has no access to the cited line's CONTENT, so "0 bad" was never a
statement about citations at any scope. It is `1 <= n <= len(lines)`.

The receipt is in JUPITER's own it.17 file. `WITHDRAWN` there lists 25 pointers
he certifies are WRONG and repaired. Every one of them was inside the 123 and
every one of them scored clean. A checker that passes 25 citations their own
author has withdrawn did not measure a narrower version of citation fitness; it
measured a different quantity that happens to correlate with it.

RED against unmutated code: `bad_citations` is called on the 25 known-wrong
pointers and names none of them.

Run:  python -m pytest tests/mars_v20/test_it18_census_zero_bad_was_never_a_citation_claim.py -q
"""
from __future__ import annotations

from tests.jupiter.test_v20_r15_it14_theory_table import bad_citations
from tests.jupiter.test_v20_r15_it17_citation_landing import (
    MANIFEST, WITHDRAWN, misses,
)


def test_the_withdrawn_pointers_are_a_real_planted_negative():
    """Precondition: these are citations their own author calls wrong."""
    assert len(WITHDRAWN) >= 20, f"only {len(WITHDRAWN)} withdrawn pointers"
    assert all(":" in c for c in WITHDRAWN)


def test_the_it14_checker_names_none_of_the_25_withdrawn_pointers():
    """RED: 'resolvability' passes every citation its author has withdrawn."""
    text = "\n".join(f"`{c}`" for c in WITHDRAWN)
    named = bad_citations(text)
    assert len(named) == len(WITHDRAWN), (
        f"the it.14 checker named {len(named)} of {len(WITHDRAWN)} pointers that "
        f"JUPITER himself withdrew as wrong. Its two failure conditions are 'no "
        f"such file' and 'file has N lines' -- so `0 bad` on 123 citations was a "
        f"line-count result reported under a citation headline. Re-scoping "
        f"presumes the narrower property was measured; it was not. The verb is "
        f"WITHDRAW.\nnamed: {named}"
    )


def test_the_landing_checker_covers_the_census_it_answers():
    """RED: the replacement certifies 29 of 123 and inherits the headline."""
    assert misses(MANIFEST) == [], "the repaired manifest itself does not land"
    assert len(MANIFEST) >= 123, (
        f"the it.17 landing instrument holds {len(MANIFEST)} hand-verified "
        f"citations against a census of 123. The Inspector's own ruling is that "
        f"the honest report is `14/123`, not `0 bad`; after the repair it is "
        f"{len(MANIFEST)}/123. The other {123 - len(MANIFEST)} are still carried "
        f"by the line-count checker under the citation headline."
    )
