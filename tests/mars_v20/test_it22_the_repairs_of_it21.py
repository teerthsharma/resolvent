"""MARS it.22 -- the three it.21 repairs, attacked.

Every node here runs against UNMUTATED shipped code. Nothing in this file writes
to a file the round owns; the heading strike grows its input IN MEMORY, which is
the same methodology JUPITER's own planted negative uses.
"""
import ast
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "V20_R15_LEAP_LEDGER.md"
JOURNAL = ROOT / "V20_R15_JOURNAL.md"


# =====================================================================
# STRIKE 1 -- the re-imported pin has no witness outside the test tree
# =====================================================================

def _ledger_binding() -> dict:
    """W-id -> arm name, read out of the round's own ledger.

    `V20_R15_LEAP_LEDGER.md` binds each wing to its arm in prose that predates
    the pin and is written by a different office than the one that wrote the
    pin. This is the second witness the pin claims to be and is not.
    """
    txt = LEDGER.read_text(encoding="utf-8", errors="replace")
    out = {}
    for wing, arm in re.findall(r"(W\d+)\s+`(arm_[a-z_]+)`", txt):
        out.setdefault(wing, set()).add(arm)
    return {w: sorted(a) for w, a in out.items()}


def test_the_wing_pin_is_a_fifth_clause_not_a_second_witness():
    """MARS it.20 STRIKE 1 re-filed against the it.21 repair.

    SATURN re-imported `WING_ARM` on the ruling that the direction it is read in
    is what licenses it: struck at it.14 as a SOURCE, restored as an ASSERTION
    TARGET. His own limit states the cost -- "wing identity now rests on a
    two-entry map that nothing corroborates."

    It is worse than uncorroborated. The V-26 threat model is an editor who
    moves every clause of the manifest together. Against that editor a pin
    living in the same tree, written by the same office, raises the price of the
    move from four edits to five. It does not add a witness; it adds a clause.

    A real witness was in the tree the whole time and is not consulted:
    `V20_R15_LEAP_LEDGER.md` states W1 `arm_smprime` and W3 `arm_pl` in prose,
    outside `tests/`, written by the ledger office rather than by SATURN.
    """
    from tests.saturn.test_v20_r15_it14_saturn import WING_ARM

    ledger = _ledger_binding()
    for wing, arm in sorted(WING_ARM.items()):
        assert ledger.get(wing) == [arm], (
            f"the corroborator is not even available for {wing}: "
            f"the ledger binds it to {ledger.get(wing)}")

    pin_file = ROOT / "tests" / "saturn" / "test_v20_r15_it14_saturn.py"
    src = pin_file.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    node = next(a.value for a in ast.walk(tree)
                if isinstance(a, ast.Assign)
                and any(getattr(t, "id", None) == "WING_ARM" for t in a.targets))

    # The file already has the corroborator open on its desk: it reads
    # V20_R15_LEAP_LEDGER.md for the L-16 recount and for `## RULING J-17d`.
    assert "LEAP_LEDGER" in src, "premise gone: the pin's module no longer reads the ledger"

    literal = (isinstance(node, ast.Dict)
               and all(isinstance(k, ast.Constant) for k in node.keys)
               and all(isinstance(v, ast.Constant) for v in node.values))
    assert not literal, (
        f"WING_ARM is a hand-typed literal at {pin_file.name}:{node.lineno} "
        f"({ {k.value: v.value for k, v in zip(node.keys, node.values)} }). "
        "The only node that checks it -- "
        "test_the_wing_ids_are_pinned_to_their_arms_outside_the_citation_set -- "
        "compares it to `_arm_of`, which derives from the manifest's own "
        "clause-(a) citations, i.e. the very clauses the V-26 move edits. So "
        "the pin is not a second witness; it is a fifth clause. An editor who "
        "moves the four clauses and this one line leaves both guardian nodes "
        "green. Meanwhile this same module already opens "
        "V20_R15_LEAP_LEDGER.md for two other facts, and that file binds "
        + ", ".join(f"{w} -> {a[0]}" for w, a in sorted(ledger.items())
                    if w in WING_ARM)
        + " in prose written by a different office, which no node reads. The "
          "corroborator was on the desk and was not opened.")


# =====================================================================
# STRIKE 2 -- J-21a's heading resolver cannot see a fenced code block
# =====================================================================

def _resolver():
    from tests.jupiter.test_v20_r15_it21_heading_anchor import resolve
    return resolve


def _fence():
    return ["```bash", "# a shell comment inside a fenced block", "echo hi", "```"]


def test_a_shell_comment_in_a_fenced_block_launders_an_ambiguous_anchor():
    """J-21a says uniqueness at RESOLVE and at LAND is the entire mechanism.
    `_level()` counts leading `#` on any line and knows nothing about fenced
    code blocks, so a `# comment` inside a ``` fence is a level-1 heading to the
    resolver. A fence dropped between two occurrences of a `want` truncates the
    section early; the later occurrences fall outside, the count collapses to 1,
    and an anchor the ruling REFUSES as ambiguous comes back as a confident line
    number. That is the failure J-21a says it exists to prevent -- "a resolver
    taking the first match would ... report a line number with full confidence"
    -- reached by a different road, and silently.

    The input is the round's own house style: these reports quote bash with `#`
    comments, and `V20_R15_LEAP_LEDGER.md` is a LIVE file under active write --
    the exact premise J-20b was written on.
    """
    resolve = _resolver()
    src = LEDGER.read_text(encoding="utf-8", errors="replace").splitlines()

    control = resolve("V20_R15_LEAP_LEDGER.md", "## it.7", "F4", src)
    assert isinstance(control, str) and control.startswith("REFUSED"), (
        f"premise gone: 'F4' is no longer ambiguous in that section ({control})")
    n_hits = int(re.search(r"on (\d+) lines", control).group(1))
    assert n_hits > 1, control

    head = next(i for i, l in enumerate(src) if l.startswith("##") and "it.7" in l)
    hits = [i for i in range(head, len(src)) if "F4" in src[i]]
    grown = src[:hits[0] + 1] + _fence() + src[hits[0] + 1:]

    after = resolve("V20_R15_LEAP_LEDGER.md", "## it.7", "F4", grown)

    assert isinstance(after, str) and after.startswith("REFUSED"), (
        f"FAIL-OPEN. On the real file the resolver REFUSES: {control}\n"
        f"   After a 4-line ```bash fence carrying one `# comment` is inserted "
        f"inside the section, it returns line {after} with full confidence.\n"
        f"   The `# comment` is read as a level-1 heading, the section ends "
        f"there, and {n_hits - 1} of the {n_hits} occurrences are scoped out of "
        f"existence. Uniqueness was not established; it was manufactured by "
        f"markdown the instrument cannot parse.")


def test_a_heading_quoted_inside_a_fence_kills_three_live_anchors():
    """The fail-closed half of the same blindness: a heading quoted inside a
    fence counts as a second heading, and the three LIVE anchors on `## it.7`
    die at once."""
    resolve = _resolver()
    src = LEDGER.read_text(encoding="utf-8", errors="replace").splitlines()
    live = {"H9": "F4** | unattempted",
            "H10": "**L-4** | **`Lean #21 [S]`**",
            "H11": "machine-true and domain-empty"}
    for cid, want in live.items():
        assert isinstance(resolve("V20_R15_LEAP_LEDGER.md", "## it.7", want, src), int), \
            f"premise gone: {cid} no longer resolves on the real file"

    grown = src + ["", "```", "## it.7", "```", ""]
    dead = {cid: resolve("V20_R15_LEAP_LEDGER.md", "## it.7", want, grown)
            for cid, want in live.items()}
    assert all(isinstance(v, int) for v in dead.values()), (
        "FAIL-CLOSED. Appending five lines that quote `## it.7` inside a "
        "``` fence -- the house style of every report this round has filed -- "
        "kills all three LIVE anchors at once:\n   "
        + "\n   ".join(f"{c}: {v}" for c, v in sorted(dead.items()))
        + "\n   J-20b retired digit anchors because they expire on the next "
          "append to a LIVE file. J-21a expires on a different append to the "
          "same LIVE file.")


# =====================================================================
# STRIKE 3 -- the index digest and the sentence that publishes it disagree
# =====================================================================

INDEX_ROW = re.compile(r"^\| C\d+ \|.*$", re.M)


def test_the_index_recipe_states_a_row_count_it_no_longer_reads():
    """The digest is CORRECT. Its provenance sentence is not.

    `V20_R15_JOURNAL.md` publishes INDEX-SHA256 over "the **31** `C`-rows".
    Recomputed with the file's own published recipe the digest matches -- over
    32 rows. `C32` was appended and the count in the sentence was not moved, so
    the one number a reader would use to check the row set before trusting the
    digest is off by one.

    This is the deletion channel MARS struck in the pid file, in prose: a recipe
    whose stated population disagrees with the population it reads cannot detect
    a row going missing, because it already disagrees by one.
    """
    txt = JOURNAL.read_text(encoding="utf-8", errors="replace")
    rows = INDEX_ROW.findall(txt)
    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()

    m = re.search(r"INDEX-SHA256 = `([0-9a-f]{64})`\*\* over the \*\*(\d+)\*\* `C`-rows", txt)
    assert m, "the INDEX-SHA256 publication sentence changed shape"
    published, stated = m.group(1), int(m.group(2))

    assert digest == published, f"digest moved: {digest} vs published {published}"
    assert stated == len(rows), (
        f"the index publishes `INDEX-SHA256 = {published[:8]}...` over "
        f'"the **{stated}** `C`-rows". The published recipe, run on the '
        f"unmodified file, reads **{len(rows)}** rows (C1..C{len(rows)}) and "
        f"reproduces that digest exactly. The digest is right and its own count "
        f"is stale by one: `C32` was appended without moving the number. A "
        f"reader who counts 31 and stops has been told the wrong population by "
        f"the sentence whose only job is to state it.")
