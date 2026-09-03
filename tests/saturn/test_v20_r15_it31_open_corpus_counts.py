"""SATURN it.31. Two rules with a node behind each, enforced over `tests/saturn/`.

    python -m pytest tests/saturn/test_v20_r15_it31_open_corpus_counts.py -q

RULE 1 (open-corpus counts). A node may not assert `len(population) == N` for
a literal N >= 2 when the population comes from a corpus the round is still
writing. The three retired shapes are MONOTONE (`>= N`, survives growth),
PARTITIONED (a named closed subset, `set(...) >= {...}`), and STAMPED (a digest
of the subject asserted in the same function, so the node REFUSES when the
subject moves rather than asserting about a subject it did not read).

The detector's discrimination is not asserted, it is MEASURED: at it.29 two
nodes over `V20_R15_THEORY_TABLE.md` counted 112 ints and 64 floats and did NOT
break, while a node over `tests/mercury/` counted 27 files and DID (`28 == 27`,
twenty-five minutes after filing, file digest unchanged). The first two carry
`assert digest == DIGEST`; the third carries nothing. STAMPED is the exemption
this detector grants, and it is exactly the line the observed failure drew.

RULE 2 (liveness). `V-7`, derived independently four times this round and
adopted in one file. Any function that shells out to git must assert the
result non-empty before asserting anything about its contents -- a scan whose
witness is missing must RAISE, not return an empty list that every later
assertion passes over vacuously.

Both rules are enforced by an EMPTY-SET assertion, not a count. An empty
offender list is a fixed point: it cannot drift upward as the round writes, and
any node added after this one that breaks either rule turns it RED. A rule
asserting `len(offenders) == 3` would be the very defect it polices.
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SATURN = ROOT / "tests" / "saturn"
SELF = pathlib.Path(__file__).name

# A corpus the round is still writing: the repo tree itself, git's index, or a
# round document read whole off disk.
OPEN_SCAN = re.compile(
    r"""ls-files|["']grep["']|log["'],\s*["']-S|rglob\(|ROOT\.glob\(|\.read_text\(""")
# The STAMPED exemption: the function pins its subject by digest.
STAMPED = re.compile(r"digest|sha256|hexdigest|_SHA\b|SHA256")
GIT = re.compile(r"""\[\s*["']git["']""")
# A liveness assertion on NAME: bare truthiness, a positive lower bound, or a
# non-empty comparison. `== N` is NOT one: it passes no witness when N could be
# reached by an empty read.
def _liveness(name):
    n = re.escape(name)
    return re.compile(
        rf"assert\s+{n}\s*[,\)\n]|assert\s+len\(\s*{n}\s*\)\s*(>|>=)\s*\d"
        rf"|assert\s+{n}\s*!=\s*(\[\]|\(\)|''|\"\")|assert\s+len\(\s*{n}\s*\)\s*!=\s*0")


def _funcs(path):
    src = path.read_text(encoding="utf-8", errors="replace")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield src, node, ast.get_source_segment(src, node) or ""


def _saturn_files():
    return [p for p in sorted(SATURN.glob("test_*.py")) if p.name != SELF]


def open_corpus_counts():
    """Every `len(x) == N`, N >= 2, over an unstamped open corpus."""
    bad = []
    for path in _saturn_files():
        for src, fn, seg in _funcs(path):
            if not OPEN_SCAN.search(seg) or STAMPED.search(seg):
                continue
            for a in [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]:
                t = a.test
                if not (isinstance(t, ast.Compare) and len(t.ops) == 1
                        and isinstance(t.ops[0], ast.Eq)):
                    continue
                c = t.comparators[0]
                left = t.left
                if not (isinstance(c, ast.Constant) and type(c.value) is int and c.value >= 2):
                    continue
                if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                        and left.func.id in ("len", "sum")):
                    continue
                bad.append(f"{path.name}:{a.lineno}  {(ast.get_source_segment(src, a) or '').strip()[:70]}")
    return bad


def git_calls_without_liveness():
    """Every function that runs git and never asserts the result non-empty."""
    bad = []
    for path in _saturn_files():
        for src, fn, seg in _funcs(path):
            if not GIT.search(seg):
                continue
            names = [tgt.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
                     for tgt in n.targets if isinstance(tgt, ast.Name)]
            if not any(_liveness(nm).search(seg) for nm in names):
                bad.append(f"{path.name}:{fn.lineno}  {fn.name}")
    return bad


# --- the two rules ----------------------------------------------------------

def test_no_saturn_node_pins_a_count_over_a_corpus_the_round_is_writing():
    """RULE 1. Empty-set, so it survives every file this office adds later."""
    bad = open_corpus_counts()
    assert bad == [], (
        "hardcoded count over an open corpus -- retire to monotone (`>= N`), "
        "partitioned (`set(...) >= {names}`), or stamped (assert the subject "
        "digest in the same function):\n  " + "\n  ".join(bad))


def test_every_saturn_git_scan_asserts_its_own_liveness():
    """RULE 2, V-7. A git scan with no witness must raise, not return []."""
    bad = git_calls_without_liveness()
    assert bad == [], (
        "git-backed scan with no non-empty assertion; an empty result passes "
        "every later node vacuously:\n  " + "\n  ".join(bad))


# --- the detectors are calibrated, not trusted ------------------------------

def test_both_detectors_fire_on_a_planted_offender(tmp_path):
    """Neither rule may be satisfied by a detector that reads nothing."""
    global SATURN
    planted = tmp_path / "test_planted.py"
    planted.write_text(
        "import subprocess\n"
        "def test_a():\n"
        "    rows = open('x').read_text()\n"
        "    assert len(rows) == 17\n"
        "def test_b():\n"
        "    out = subprocess.run(['git', 'ls-files'], capture_output=True).stdout.split()\n"
        "    assert len(out) == 9\n", encoding="utf-8")
    keep, SATURN = SATURN, tmp_path
    try:
        counts, live = open_corpus_counts(), git_calls_without_liveness()
    finally:
        SATURN = keep
    assert [c.split()[0] for c in counts] == ["test_planted.py:4", "test_planted.py:7"], counts
    assert [b.split()[0] for b in live] == ["test_planted.py:5"], live


def test_the_stamped_exemption_is_the_line_the_it29_failure_drew(tmp_path):
    """The exemption is granted to a digest-pinned function and to nothing else."""
    global SATURN
    planted = tmp_path / "test_stamped.py"
    planted.write_text(
        "def test_stamped():\n"
        "    digest, ints = screen(open('t').read_text())\n"
        "    assert digest == DIGEST\n"
        "    assert len(ints) == 112\n", encoding="utf-8")
    keep, SATURN = SATURN, tmp_path
    try:
        got = open_corpus_counts()
    finally:
        SATURN = keep
    assert got == [], got
