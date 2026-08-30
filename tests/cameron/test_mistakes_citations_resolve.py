"""MISTAKES.md must not rot into the failure classes it documents.

WHY THIS FILE EXISTS. `MISTAKES.md` is 35 entries whose whole value is the
citation on each: an entry whose `file:line` no longer resolves is advice
without evidence, which is P-3 (a stale claim never retracted) and P-6
(line-reference drift) committed by the file that names them. That is not
hypothetical. Both have already happened here:

  * Building the file in R9 iteration 1, four of its citations were wrong on
    first write, and TWO of them drifted because the same commit's edits moved
    the lines being cited.
  * Adding the R9 iteration 3 entries, `mercury-report.md:49` was written
    without its directory and resolved to nothing. This checker is what caught
    it, before the commit.

P-5's own rule is "assert cited symbols exist -- one test that imports every
name a docstring names is cheaper than the audit that finds them missing." This
is that test, for the document rather than for the code.

THE PLANTED NEGATIVE IS THE POINT. A checker that reports "every citation is
fine" is an absence claim, and V-7 and V-13 are both about absence claims from
searches that could not have found anything. So the extractor is run against
deliberately broken input and required to catch each break. Without that half,
a regex that silently matched nothing would print a clean bill of health
forever.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = ROOT / "MISTAKES.md"

#: `path.ext:line` or `path.ext:line-line`, inside backticks.
NUMBERED = re.compile(r"`([\w/\.\-]+\.(?:py|md|jsonl|txt)):(\d+)(?:-(\d+))?`")
#: a bare `path.ext` inside backticks.
BARE = re.compile(r"`([\w/\.\-]+\.(?:py|md|jsonl|txt))`")

#: DELIBERATE non-resolving references, each with the reason it must stay.
#: Anything not on this list has to resolve, so the list is the only place a
#: dead citation can hide and it is three lines long.
ALLOWED_DEAD = {
    # P-6 quotes the STALE form as the example of what not to write. The whole
    # entry is that `arm_s.py:97` is wrong and `arm_s.pivots_of` is right.
    "arm_s.py",
    # P-5 quotes `capability_table.py`'s own TASK_SOURCE string verbatim, and
    # that string names the file without its directory.
    "m3_quintuple.py",
}


@pytest.fixture(scope="module")
def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    return set(out)


def _numbered(text):
    return sorted(set(NUMBERED.findall(text)))


def test_the_document_exists_and_the_extractor_finds_its_citations():
    """The instrument before the reading. A regex that matched nothing would
    make every assertion below vacuously true -- V-7's exact shape."""
    text = DOC.read_text(encoding="utf-8")
    entries = [ln for ln in text.splitlines() if ln.startswith("### ")]
    refs = _numbered(text)
    print(f"\n  {len(entries)} entries, {len(refs)} numbered citations, "
          f"{len(set(BARE.findall(text)))} bare file references")
    assert len(entries) >= 35, len(entries)
    assert len(refs) >= 60, len(refs)


def test_every_numbered_citation_resolves_and_is_in_range(tracked):
    """A `file:line` that points past the end of the file, or at a file that is
    not in the repository, is P-6 and P-3 respectively."""
    text = DOC.read_text(encoding="utf-8")
    bad = []
    for f, lo, hi in _numbered(text):
        if f in ALLOWED_DEAD:
            continue
        q = ROOT / f
        if not q.exists():
            bad.append(f"{f}:{lo} does not exist")
            continue
        if f not in tracked:
            bad.append(f"{f}:{lo} is not tracked -- a reader cannot open it")
        n = len(q.read_text(encoding="utf-8").splitlines())
        if int(hi or lo) > n or int(lo) < 1:
            bad.append(f"{f}:{lo}-{hi} is out of range (file has {n} lines)")
    assert not bad, "\n".join(bad)


def test_every_bare_file_reference_is_a_tracked_file(tracked):
    text = DOC.read_text(encoding="utf-8")
    bad = [f for f in sorted(set(BARE.findall(text)))
           if f not in ALLOWED_DEAD and (not (ROOT / f).exists()
                                         or f not in tracked)]
    assert not bad, bad


def test_every_allowed_dead_reference_is_still_actually_dead():
    """The allowlist is a liability, so it is checked from both ends. If one of
    these starts resolving, the exemption is stale and must be deleted rather
    than left to cover a future real break."""
    for f in ALLOWED_DEAD:
        assert not (ROOT / f).exists(), (
            f"{f} now resolves -- remove it from ALLOWED_DEAD, it is no longer "
            f"an exemption and is now hiding real breaks")


@pytest.mark.parametrize("broken,why", [
    ("`scale/no_such_file.py:12`", "a file that does not exist"),
    ("`MISTAKES.md:999999`", "a line past the end of the file"),
    ("`results/r9_pricing.md:0`", "a line number below 1"),
])
def test_the_checker_catches_a_planted_break(broken, why, tracked):
    """THE PLANTED NEGATIVE. Each break is run through the SAME extractor and
    the SAME predicate the real test uses, and must be caught. A citation
    checker that cannot fail is worth exactly as much as a control that cannot
    fail, which is the subject of the document it is checking."""
    caught = []
    for f, lo, hi in _numbered(broken):
        q = ROOT / f
        if not q.exists() or f not in tracked:
            caught.append(f)
            continue
        n = len(q.read_text(encoding="utf-8").splitlines())
        if int(hi or lo) > n or int(lo) < 1:
            caught.append(f)
    print(f"  planted {broken} ({why}) -> caught={bool(caught)}")
    assert caught, f"the checker MISSED {broken}: {why}"
