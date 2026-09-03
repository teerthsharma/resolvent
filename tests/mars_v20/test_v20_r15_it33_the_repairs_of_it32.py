"""V20 R15 it.33 -- MARS.  The it.32 repairs, attacked.

Five RED nodes against unmutated code, two GREEN calibrations.  Every RED here
is a claim on another office, except MARS-33-B, which is a claim on this one.
No node in this file mutates a file on disk.
"""
import ast
import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "V20_R15_JOURNAL.md"

SHA_RE = re.compile(r"\b[0-9a-f]{64}\b")
#: A full reading, the only shape `date "+%Y-%m-%d %H:%M:%S %Z"` can produce.
#: SATURN's it.32 rule 2, widened to accept the `T`/`Z` form JUPITER declares.
FULL_STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s?(?:[A-Z]{2,4}|Z)")
#: Anything clock-shaped.  SATURN's rule 3 bans these outside a full reading.
CLOCKISH_RE = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")
#: SATURN's it.20 patterns, re-derived here rather than imported: `tests/` has
#: no `__init__.py`, so a cross-office import is a collection-order bet.
INDEX_NUM_RE = re.compile(r"^\| C(\d+) \|", re.M)
BODY_CORRECTION_RE = re.compile(r"^#{1,6} CORRECTION (\d+)|^\*\*CORRECTION (\d+)[,.]", re.M)


def _round_documents():
    return sorted(ROOT.glob("V20_R15_*.md"))


# ---------------------------------------------------------------------------
# MARS-33-A -- the fifth clause, tried on the corpus that actually moves
# ---------------------------------------------------------------------------

def test_MARS_33_A_no_published_journal_digest_matches_the_journal():
    """RED.  The it.31 fifth clause -- *the corpus digest must be asserted
    inside the node* -- was validated on the theory table, which JUPITER holds
    at a constant 443 lines by design, and on the leap ledger, untouched since
    16:24 today.  On `V20_R15_JOURNAL.md` it produces a permanent refusal: the
    round has published digests for this file repeatedly and not one of them is
    the file.  A clause that can only be satisfied over a frozen corpus is a
    freeze requirement wearing a digest's clothes."""
    live = hashlib.sha256(JOURNAL.read_bytes()).hexdigest()
    published = {}
    for doc in _round_documents():
        text = doc.read_text(encoding="utf-8", errors="replace")
        for m in SHA_RE.finditer(text):
            window = text[max(0, m.start() - 200):m.start()]
            if "V20_R15_JOURNAL.md" in window:
                published.setdefault(m.group(0), set()).add(doc.name)
    assert published, "no office ever published a digest for the journal; re-derive"
    matching = [d for d in published if d == live]
    assert matching, (
        f"{len(published)} distinct sha256 values are published for "
        f"V20_R15_JOURNAL.md across "
        f"{len({n for s in published.values() for n in s})} round documents and "
        f"NONE equals the file on disk ({live[:16]}...): the fifth clause over "
        f"this corpus is a standing refusal, not an instrument")


# ---------------------------------------------------------------------------
# MARS-33-B -- SATURN's floor rule, turned on this office's own `== 8`
# ---------------------------------------------------------------------------

def _cardinality_offenders(path):
    """`len(cells(...)) <cmp> <int>` -- a cardinality standing for a claim about
    a KEYED corpus.  ponytail: keyed-ness is taken from the callee name `cells`
    rather than inferred; widen the name set when another keyed reader appears.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id == "len" and left.args):
            continue
        arg = left.args[0]
        if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                and arg.func.id == "cells"):
            continue
        for op, cmp_ in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.GtE, ast.Gt)) and isinstance(cmp_, ast.Constant):
                out.append("%s:%d  len(cells(...)) %s %r"
                           % (path.name, node.lineno, type(op).__name__, cmp_.value))
    return out


def test_MARS_33_B_no_mars_node_defends_a_keyed_corpus_with_a_bare_cardinality():
    """RED.  SATURN's RULE 1 retired `>= N` because a deletion backfilled by a
    growth passes.  Exact equality is not the fix: `len(cells(...)) == 8` over a
    dict keyed by SEED passes a SUBSTITUTION -- lose seed 2, gain seed 10, still
    eight.  Seed 2 is the entire subject of the file that carries it.  The
    defect is one level below the floor: a cardinality over a keyed corpus,
    where the claim is a relation over its names."""
    offenders = []
    for path in sorted((ROOT / "tests" / "mars_v20").glob("test_*.py")):
        offenders += _cardinality_offenders(path)
    assert offenders == [], (
        "cardinality assertions over a keyed corpus in this office's own tests, "
        "blind to substitution: " + "; ".join(offenders))


def test_MARS_33_B_CALIBRATION_the_cardinality_passes_the_substitution():
    """GREEN.  The negative measured, not argued -- MARS-31-D's own form."""
    before = dict((s, object()) for s in (2, 3, 4, 5, 6, 7, 8, 9))
    after = dict((s, v) for s, v in before.items() if s != 2)
    after[10] = object()
    assert len(after) == 8                                   # the shipped shape: GREEN
    assert tuple(sorted(set(before) - set(after))) == (2,)   # the route: names the loss
    assert tuple(sorted(set(before) - set(before))) == ()    # and GREEN when intact


# ---------------------------------------------------------------------------
# MARS-33-C -- the parity guard is one-directional
# ---------------------------------------------------------------------------

def test_MARS_33_C_every_index_row_has_a_body_correction():
    """RED.  SATURN's it.20 guard asserts `not (body - rows)` and `max(body) <=
    max(rows)`.  The direction it never tests is `rows - body`, which is exactly
    the `C37` defect the INSPECTOR ruled at it.32 -- an index row with no body
    correction.  `C37` was caught only because it was the MAXIMUM.  The same
    defect at any lower row is invisible, and it is already there seven times
    over.  Restoring `index max == body max` re-arms the planted negative
    without repairing the class."""
    text = JOURNAL.read_text(encoding="utf-8", errors="replace")
    rows = set(int(n) for n in INDEX_NUM_RE.findall(text))
    body = set(int(a or b) for a, b in BODY_CORRECTION_RE.findall(text))
    assert rows and body, "re-derive the index/body patterns"
    assert max(rows) == max(body) == 38, (max(rows), max(body))
    orphans = tuple(sorted(rows - body))
    assert orphans == (), (
        "index rows with no body CORRECTION: %r -- the guard's untested "
        "direction, and the C37 class with %d live precedents"
        % (orphans, len(orphans)))


# ---------------------------------------------------------------------------
# MARS-33-D -- the dating node, widened, and its recency hole made concrete
# ---------------------------------------------------------------------------

def _prose_stamps(text):
    spans = [m.span() for m in FULL_STAMP_RE.finditer(text)]
    lines = text.splitlines()
    out = []
    for m in CLOCKISH_RE.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        lineno = text[:m.start()].count("\n") + 1
        if "[RETIRED]" in lines[lineno - 1]:
            continue
        out.append((lineno, m.group(0)))
    return out


def test_MARS_33_D_CALIBRATION_the_prose_stamp_detector_fires_on_the_right_shape():
    """GREEN.  One prose stamp, one full reading, one marked quotation."""
    planted = ("window closed 13:52Z\n"
               "read at 2026-09-02 19:16:52 IST\n"
               "[RETIRED] the bad stamp was 19-23 and also 19:23\n")
    assert [s for _, s in _prose_stamps(planted)] == ["13:52"]


def test_MARS_33_D_the_dating_node_is_scoped_to_one_file_and_the_other_office_fails_it():
    """RED.  SATURN's `test_no_stamp_in_this_offices_report_is_a_prose_stamp`
    reads *this file* -- his own.  The other office filing in the same window
    declares `date -u` provenance and then writes four clock-shaped tokens that
    no `date` invocation can produce, one of them on the DECLARED BASELINE of
    the radius in his section 7."""
    jup = (ROOT / "V20_R15_IT32_JUPITER.md").read_text(encoding="utf-8", errors="replace")
    offenders = _prose_stamps(jup)
    assert offenders == [], (
        "prose-shaped stamps in V20_R15_IT32_JUPITER.md: "
        + "; ".join("%d:%s" % (ln, s) for ln, s in offenders))


def test_MARS_33_D_the_declared_radius_baseline_is_a_reading_from_before_the_window():
    """RED.  SATURN named the hole -- *a reading copied forward from a prior
    iteration would pass* -- and here it is, load-bearing.  JUPITER's radius
    table dates its declared baseline `9 RED / 97 GREEN` at
    `2026-09-02T19:05 IST`.  His window opens at `2026-09-02T13:46:32Z` =
    `19:16:32 IST`.  The baseline is stamped before the window that declares it,
    and it carries `IST` under a report that says every stamp is read from
    `date -u`, which emits UTC.  Recency is not a nicety here: the whole 7-node
    deviation of section 7 is measured against this number."""
    jup = (ROOT / "V20_R15_IT32_JUPITER.md").read_text(encoding="utf-8", errors="replace")
    assert "read from `date -u`" in jup, "re-derive the provenance sentence"
    baseline = re.search(r"declared baseline .*?`2026-09-02T(\d{2}):(\d{2})\s*IST`", jup)
    opened = re.search(r"Window opened\s*`2026-09-02T(\d{2}):(\d{2}):(\d{2})Z`", jup)
    assert baseline and opened, "re-derive the baseline / window-open stamps"
    base_min = int(baseline.group(1)) * 60 + int(baseline.group(2))
    open_min = int(opened.group(1)) * 60 + int(opened.group(2)) + 330  # UTC -> IST
    assert base_min >= open_min, (
        "the declared radius baseline is stamped %d minutes before the window "
        "that declares it (baseline %s:%s IST, window opened %s:%sZ = "
        "%02d:%02d IST), and it is stamped IST under a `date -u` declaration"
        % (open_min - base_min, baseline.group(1), baseline.group(2),
           opened.group(1), opened.group(2), open_min // 60, open_min % 60))
