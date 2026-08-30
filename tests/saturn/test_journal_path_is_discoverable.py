"""SATURN P0.1 -- a census keyed on literal paths under-reports journal producers.

THE FAILURE THIS EXISTS TO PREVENT, with its cost. Building the round-10 census, a
scan that resolved "which file produced which journal" by matching the literal string
`results/<name>.jsonl` in each module reported NO JOURNAL ENTRY for
`scale/s2_units.py` and `scale/r4b_units.py`-class modules, and the census presumed
them ORPHAN on the brief's own rule ("zero importers AND zero results = orphan").

That was false. `scale/s2_units.py:56` sets `NAME = "s2"` and `:161` calls
`Journal(NAME)`; `scale/bucket.py:45` then builds `self.path = RESULTS / f"{name}.jsonl"`.
The journal is `results/s2.jsonl`, 13 records, and 13 of that module's 15
six-significant-digit readings reproduce from `results/*.jsonl` at abs=5e-7. The
producer was live and the scan could not have found it on any input, because the path
it looks for is never written down.

This is the same class as the defect `scale/journal_scan.py` exists to prevent -- a
search structurally incapable of finding a thing, its silence read as absence
(MISTAKES.md V-7). It is recorded here as a mechanism rather than a note, because the
next census will be written by someone who did not build this one.

RED FIRST. Against the tree as it stands this test FAILS, naming every module whose
journal is reachable only through the constructed path.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

#: the constructed-path site itself, so the test names its own cause
BUCKET_LINE = "self.path = RESULTS / f\"{name}.jsonl\""


def _journal_callers() -> dict[str, str]:
    """scale modules that write a journal through `Journal(NAME)`, mapped to NAME."""
    out: dict[str, str] = {}
    for p in sorted((ROOT / "scale").glob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "Journal(" not in src or p.name == "bucket.py":
            continue
        m = re.search(r'^NAME\s*=\s*"([^"]+)"', src, re.M)
        if m:
            out[f"scale/{p.name}"] = m.group(1)
    return out


def test_the_constructed_path_site_is_where_this_test_says_it_is():
    """The premise, bound. If bucket.py stops constructing the path, this test is moot."""
    src = (ROOT / "scale" / "bucket.py").read_text(encoding="utf-8")
    assert BUCKET_LINE in src, (
        "scale/bucket.py no longer builds the journal path from a NAME. This test's "
        "whole premise was that the path is constructed and therefore invisible to a "
        "literal scan; re-derive it before trusting the assertion below.")


def test_at_least_one_module_journals_through_the_constructed_path():
    """A must-fire arm. A census rule about an empty set is its own vacuous control."""
    callers = _journal_callers()
    assert callers, (
        "no scale module calls Journal(NAME) with a module-level NAME. The finding "
        "this file records would then be unreachable and the test below vacuous.")


@pytest.mark.parametrize("mod,name", sorted(_journal_callers().items()))
def test_the_journal_a_module_writes_is_named_in_that_module(mod: str, name: str):
    """RED. The artifact a module produces must be discoverable by reading the module.

    Not a style preference: the census that reads these files is the instrument that
    decides KEEP or ATTIC, and it cannot cite a path the source never spells.
    """
    artifact = f"results/{name}.jsonl"
    exists = (ROOT / artifact).exists()
    src = (ROOT / mod).read_text(encoding="utf-8", errors="replace")
    assert artifact in src, (
        f"{mod} writes {artifact} (exists on disk: {exists}) through Journal(NAME=\"{name}\") "
        f"and scale/bucket.py:45, but the string {artifact!r} appears nowhere in {mod}. "
        f"A census keyed on literal paths reports NO JOURNAL ENTRY here and, on the rule "
        f"'zero importers AND zero results = orphan', presumes the producer dead. "
        f"Name the artifact in the module -- a comment carrying the literal is enough.")
