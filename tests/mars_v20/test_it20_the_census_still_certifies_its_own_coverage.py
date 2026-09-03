"""MARS it.20 STRIKE 3 -- the landing instrument still asserts nothing about
how much of the census it opens.

The standing RED leaving it.18 was `30 >= 123`. At it.20 the numbers are
`32 >= 129`, measured here rather than quoted: 32 distinct `path:line` citations
are individually opened (the union of it.17's MANIFEST and it.18's REPAIRS), out
of 103 distinct citations / 129 occurrences in V20_R15_THEORY_TABLE.md.

The two numbers live in different files and are never brought into contact.
`POPULATION_AT_IT18 = 129` is compared only against itself
(`len(cites) == POPULATION_AT_IT18`), and it.17's sensitivity calibration builds
`shifted` FROM `MANIFEST` and compares `len(named)` TO `len(MANIFEST)` -- a
legitimate line-sensitivity probe that is arithmetically incapable of failing for
a coverage reason. A census that certifies its own coverage is the mechanism this
office struck at it.9 and it.18; it is still standing.

RED against unmutated code: no shipped file is touched. This node imports the two
instrument modules and the table and states the ratio nothing else states.

Run:  python -m pytest tests/mars_v20/test_it20_the_census_still_certifies_its_own_coverage.py -q
"""
from __future__ import annotations

import pathlib
import re

import pytest

from tests.jupiter import test_v20_r15_it17_citation_landing as IT17
from tests.jupiter import test_v20_r15_it18_citation_landing as IT18

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

#: The floor a landing instrument has to clear to be a census rather than a sample.
COVERAGE_FLOOR = 0.90


#: it.17's manifest is keyed by `path:line`; it.18's REPAIRS is keyed by CLAIM ID
#: (C17, C64, C103) and carries the citation in its values, so the two cannot be
#: unioned as-is. Pulling `path:line` out of the repr keeps the numerator honest
#: without assuming either file's record shape.
CITE_SHAPE = re.compile(r"[\w./\-]+\.[A-Za-z0-9]+:\d+")


def _opened() -> set[str]:
    return set(IT17.MANIFEST) | set(CITE_SHAPE.findall(repr(IT18.REPAIRS)))


def _census() -> list[str]:
    text = TABLE.read_text(encoding="utf-8", errors="replace")
    return [f"{p}:{n}" for p, n in IT18.CITE_RE.findall(text)
            if IT18.EXT_RE.search(p)]


def test_the_two_numbers_are_never_brought_into_contact():
    """The premise. Neither instrument file names a coverage ratio at all."""
    for mod in (IT17, IT18):
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert not re.search(r"coverage|COVERAGE", src), (
            f"premise gone: {pathlib.Path(mod.__file__).name} now names coverage; "
            f"re-derive the strike")


def test_every_opened_citation_is_a_real_member_of_the_census():
    """Calibration: the numerator is not inflated by phantoms."""
    census = set(_census())
    phantom = sorted(c for c in _opened() if c not in census)
    assert not phantom, f"opened citations absent from the table: {phantom}"


def test_the_landing_instrument_opens_most_of_the_census():
    """THE RED."""
    opened, census = _opened(), _census()
    distinct = set(census)
    ratio = len(opened) / len(distinct)
    assert ratio >= COVERAGE_FLOOR, (
        f"{len(opened)} of {len(distinct)} distinct citations "
        f"({len(opened)}/{len(census)} occurrences) are individually opened -- "
        f"{ratio:.1%}. The remaining {len(distinct) - len(opened)} have never been "
        f"read at the line they name by anything. Nothing in the repo asserts this "
        f"ratio: POPULATION_AT_IT18 is compared only against itself, and it.17's "
        f"+10 shift probe is built from MANIFEST and compared to len(MANIFEST). "
        f"The standing `30 >= 123` is now `{len(opened)} >= {len(census)}`.")
