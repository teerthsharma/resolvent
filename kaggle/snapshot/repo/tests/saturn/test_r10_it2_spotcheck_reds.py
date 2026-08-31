"""The REDs behind the R10 iteration-2 spot-check findings.

Each test states what AUDIT.md NEEDS to be true for a drawn row's disposition to
be a verdict rather than a presumption, and is run against the sheet and the tree
exactly as they stand. Written to fail first; the ones that go green after the
sheet is amended stand as the regression guard, and the ones that stay red are
the finding.

The draw that selected these rows is `scale/spotcheck_draw.py` (SEED=10002) and
was fixed before this file existed; the protocol that defines pass and dead is
`results/r10_it2_protocol.md`, committed at 06a180c before any row was measured.
Neither is editable from here, which is the point of both.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT = ROOT / "AUDIT.md"
CONFTEST = ROOT / "tests" / "chase" / "conftest.py"

#: A census naming a row is not a reader of it. Fixed in the protocol, section 1,
#: before any count was taken, so a citation total cannot be tuned by this list.
EXCLUDED = ("AUDIT.md", "R10_ITERATION_01.md", "scale/spotcheck_draw.py",
            "results/r10_it2_draw.txt", "results/r10_it2_protocol.md",
            "house-events", ".claude/worktrees", "tests/saturn/test_r10_it2")


def _rows():
    """(path, class, disposition, reason) per classed AUDIT.md row."""
    out = {}
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7:
            continue
        out[m.group(1)] = (cells[4], cells[5], cells[6])
    return out


@pytest.fixture(scope="module")
def rows():
    got = _rows()
    # Calibrate before asserting. A parser that returned {} would make every
    # claim below vacuously true -- MISTAKES.md V-7, the shape this whole
    # iteration exists to catch.
    assert len(got) >= 300, f"sheet parsed to {len(got)} rows; refusing to judge on that"
    return got


@pytest.fixture(scope="module")
def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    return [p for p in out if not any(x in p for x in EXCLUDED)]


# ------------------------------------------------------------------ FINDING A

def test_no_row_claims_no_python_importer_while_importers_exist(rows, tracked):
    """A reason cell reading "no python importer" is an ABSENCE CLAIM, and this
    repo's V-7 entry is about absence claims from searches that could not have
    found anything. Re-measured with a front-door grep instead."""
    pys = [p for p in tracked if p.endswith(".py")]
    texts = {p: (ROOT / p).read_text(encoding="utf-8", errors="replace") for p in pys}
    wrong = {}
    claimed = 0
    for path, (_cls, _dis, reason) in rows.items():
        if "no python importer" not in reason.lower():
            continue
        if not path.startswith("scale/") or not path.endswith(".py"):
            continue
        # An AMENDED cell QUOTES the stale form as the thing it corrected -- the
        # same shape as ALLOWED_DEAD in test_mistakes_citations_resolve.py, where
        # P-6 quotes `arm_s.py:97` precisely to say it is wrong. Skipping the
        # quotation is not weakening the check; asserting on it would forbid the
        # sheet from recording its own corrections.
        if "AMENDED r10-it2" in reason:
            continue
        claimed += 1
        stem = pathlib.Path(path).stem
        pat = re.compile(rf"(?:^|\n)\s*(?:from\s+scale\.{stem}\s+import|"
                         rf"from\s+scale\s+import\s+[^\n]*\b{stem}\b|"
                         rf"import\s+scale\.{stem}\b)")
        hits = [p for p, t in texts.items() if p != path and pat.search(t)]
        if hits:
            wrong[path] = hits
    # Calibrate: if the predicate selected nothing the assertion below is
    # vacuous, which is the exact failure this file was written to catch.
    assert claimed > 0, "no unamended row asserts 'no python importer'; check is vacuous"
    assert not wrong, (
        "AUDIT.md claims 'no python importer' for rows that have front-door importers: "
        + json.dumps({k: v for k, v in wrong.items()}, indent=2))


# ------------------------------------------------------------------ FINDING B

def _ledgered_files():
    return sorted(set(re.findall(r'"(test_\w+\.py)::', CONFTEST.read_text(encoding="utf-8"))))


def test_the_known_red_ledger_calibrates():
    """The instrument before the reading."""
    got = _ledgered_files()
    assert len(got) >= 10, got
    assert "test_hf_shipping.py" in got


@pytest.mark.parametrize("leaf", _ledgered_files())
def test_a_file_the_conftest_reaches_into_every_run_is_not_dead(rows, leaf):
    """`pytest_collection_modifyitems` in tests/chase/conftest.py resolves each
    ledger key against collected node ids on EVERY run and marks the item
    xfail(strict=True). That is a live reader holding a live reference: the file
    cannot be an orphan, and its reds are ledgered findings, not vacuity.

    The census read these as "ran ALL-GREEN in the iteration-1 sweep". A file
    whose every test is a strict xfail reports exit 0 and zero failures, so
    ALL-GREEN was read off an exit status that cannot tell a passing test from a
    recorded finding -- the same surface-feature mistake as P1, P3 and P1'."""
    path = "tests/chase/" + leaf
    cls, dis, _reason = rows.get(path, ("<absent>", "<absent>", ""))
    assert dis == "KEEP", (
        f"{path} is named {CONFTEST.name}'s KNOWN_RED ledger and is marked "
        f"xfail(strict=True) at every collection, yet the sheet classes it "
        f"{cls}/{dis}")


# ------------------------------------------------------------------ FINDING C

def _journal_pool():
    sys.path.insert(0, str(ROOT))
    from scale.journal_scan import walk_numbers
    pool = set()
    for j in sorted((ROOT / "results").glob("*.jsonl")):
        for line in j.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for _p, v in walk_numbers(rec):
                pool.add(v)
    return pool


_NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)")


def _sig(s):
    return len(re.sub(r"[^0-9]", "", s.split("e")[0].split("E")[0]).lstrip("0"))


def _reproducing(doc, pool, tol=5e-7):
    text = (ROOT / doc).read_text(encoding="utf-8", errors="replace")
    reads = sorted({float(s) for s in _NUM.findall(text) if _sig(s) >= 6})
    return [r for r in reads if any(abs(r - v) <= tol for v in pool)], reads


@pytest.fixture(scope="module")
def pool():
    got = _journal_pool()
    assert len(got) > 10_000, f"journal scan found {len(got)} leaves; witness failed"
    return got


def test_the_reading_extractor_reproduces_the_sheets_own_control(pool):
    """Calibration, and it is what makes the next test admissible. AUDIT.md's
    cell for LOOP_PROMPT_ROUND6_ARCHIVE.md reads 8/23. If this extractor does
    not return exactly that, it is not the census's extractor and any
    disagreement it reports is its own."""
    hits, reads = _reproducing("LOOP_PROMPT_ROUND6_ARCHIVE.md", pool)
    assert (len(hits), len(reads)) == (8, 23), (len(hits), len(reads))


def test_every_doc_kept_as_provenance_has_readings_that_reproduce(pool, rows):
    """A SUPERSEDED root doc retained on the ground that it is "the provenance of
    published readings" must have at least one reading that reproduces. If none
    do, that ground is empty and the KEEP rests on nothing.

    Keyed on the sheet's LIVE claim rather than a fixed list, so it guards every
    future row that invokes provenance -- which is what makes it a regression
    guard instead of a note about one document."""
    claiming = [p for p, (_c, dis, r) in rows.items()
                if p.endswith(".md") and "/" not in p and dis == "KEEP"
                and "provenance of published readings" in r
                and "AMENDED r10-it2" not in r]   # an amended cell quotes the ground it withdrew
    assert claiming, "no row invokes the provenance ground; check is vacuous"
    empty = {}
    for doc in claiming:
        hits, reads = _reproducing(doc, pool)
        if not hits:
            empty[doc] = len(reads)
    assert not empty, (
        "rows kept as 'the provenance of published readings' whose readings do not "
        f"reproduce from results/*.jsonl at abs=5e-7: {empty}")


# ------------------------------------------------------------------ AMENDMENT

def test_an_intentional_red_in_tests_chase_is_on_the_ledger():
    """tests/chase is the one directory carrying a KNOWN_RED ledger, so in it an
    intentional red is recorded there. tests/foreman reds are unledgered by
    design because foreman has no ledger; chase does, and an unledgered
    intentional red inside it is invisible to CI as deliberate."""
    conf = CONFTEST.read_text(encoding="utf-8")
    offenders = []
    for f in sorted((ROOT / "tests" / "chase").glob("test_*.py")):
        text = f.read_text(encoding="utf-8", errors="replace")
        if re.search(r"RED ON PURPOSE|RED on purpose", text) and f.name not in conf:
            offenders.append(f.name)
    assert not offenders, (
        "files in tests/chase declare 'RED on purpose' but are absent from the "
        f"KNOWN_RED ledger, so their deliberate reds surface as hard CI failures: {offenders}")
