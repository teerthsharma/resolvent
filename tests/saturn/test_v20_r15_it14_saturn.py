"""it.14 SATURN -- the V-26 sibling sweep, the theory-table freeze, the ledger arithmetic.

VENUS filed `V-26` at it.13: *a marginal assertion standing in for a joint claim*.
Her detection rule is executable and is the only thing section A does:

    for a claim about a relation between two collections, the falsifying
    mutation is the one that PRESERVES EVERY PER-COLLECTION STATISTIC.

A node that survives that mutation is not asserting the relation. Applied to
this office's own two instruments first -- the it.4 `FREEZE-SHA256` and the
it.12 per-row ledger baseline -- because both certify *these rows are these
rows*, which is the exact shape, and because the office that just lost a ruling
audits its own instruments hardest.

Result, stated up front so the file is not read as a fishing trip:
  * it.4 clause (b) is TWO MARGINALS. It stays green under the mutation. Repaired.
  * it.12's per-row baseline is JOINT. It fires. It clears, and the control is here.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re

from tests.saturn import test_v20_r15_freeze_manifest as freeze
from tests.saturn import test_v20_r15_it12_saturn as it12
from tests.saturn import test_v20_r15_it27_wing_arm_citation as it27

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = ROOT / "V20_R15_LEAP_LEDGER.md"
CONTRACT = ROOT / "CEQ_V20_R15_CONTRACT.md"
JOURNAL = ROOT / "V20_R15_JOURNAL.md"
THEORY = ROOT / "V20_R15_THEORY_TABLE.md"

#: The manifest's wing ids mapped to the arm name a journalled cell carries in
#: its `kind` field. This map is the JOINT the manifest asserts in prose and
#: never asserted in a node until this iteration.
#:
#: it.27: DERIVED, not typed. MARS's it.22 STRIKE 1 -- the hand-typed literal was a
#: fifth manifest clause, not a second witness. The it.26 answer (a six-office
#: consensus) was withdrawn by its own author: all 32 votes descend from the
#: manifest, so a founding mistake at it.1 reads GREEN 32 times. This derivation
#: reads ledger citations that must land inside the source they claim, so a swapped
#: pin fails on CODE rather than on agreement.
WING_ARM = it27.ledger_wing_arm()


# ==========================================================================
# A.  THE V-26 SWEEP
# ==========================================================================

def _lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _kind_at(lines: list[str], lineno: int) -> str | None:
    """The `kind` of the journalled record on a 1-based line, or None."""
    if not 1 <= lineno <= len(lines):
        return None
    try:
        rec = json.loads(lines[lineno - 1])
    except Exception:
        return None
    return rec.get("kind") if isinstance(rec, dict) else None


def _kind_counts(lines: list[str]) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for ln in lines:
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if isinstance(rec, dict) and "kind" in rec:
            c[rec["kind"]] += 1
    return c


def _clause_b_sites() -> dict[str, tuple[pathlib.Path, int, str]]:
    """wing -> (results file, 1-based line, anchor) off the frozen list."""
    out = {}
    for w, clause, cite, anchor in freeze.rows():
        if clause != "b":
            continue
        path_s, _, line_s = cite.rpartition(":")
        out[w] = (ROOT / path_s, int(line_s), anchor)
    return out


def _swap_kinds(lines: list[str], a: int, b: int) -> list[str]:
    """MARGINAL-PRESERVING MUTATION. Exchange the `kind` of two records.

    Every per-collection statistic is invariant: the multiset of kinds is
    unchanged, so every count is unchanged; no numeric field is touched, so
    every anchor still sits on its own line; the manifest file is never opened,
    so `FREEZE-SHA256` cannot move. Only the PAIRING of wing to record moves.
    """
    out = list(lines)
    ra, rb = json.loads(out[a - 1]), json.loads(out[b - 1])
    ra["kind"], rb["kind"] = rb["kind"], ra["kind"]
    out[a - 1], out[b - 1] = json.dumps(ra), json.dumps(rb)
    return out


def test_the_clause_b_pairing_is_true_on_the_unmutated_tree():
    """The joint claim is TRUE today. It has to be, or the strike below would
    be a bug report about the data rather than about the instrument."""
    for w, (path, lineno, _anchor) in sorted(_clause_b_sites().items()):
        got = _kind_at(_lines(path), lineno)
        assert got == WING_ARM[w], (
            f"{w} clause (b) cites {path.name}:{lineno}, which journals "
            f"{got!r}, not {WING_ARM[w]!r}")


def test_the_it4_freeze_is_two_marginals_and_stays_green_under_the_joint_break():
    """THE STRIKE, against this office's own it.4 instrument.

    `FREEZE-SHA256` + clause (b) claim a RELATION between two collections: the
    manifest's wing rows and the journalled records in `results/`. The it.4
    nodes assert two functions of each collection separately --
    `journalled_cells(wing) > 0` over the whole of `results/`, and `anchor is
    on the cited line` over the text of one file. Neither ranges over the pair.
    """
    sites = _clause_b_sites()
    paths = {p for p, _, _ in sites.values()}
    assert len(paths) == 1, f"clause (b) spans {len(paths)} files; widen the mutation"
    path = paths.pop()
    lines = _lines(path)
    (wa, (_pa, la, aa)), (wb, (_pb, lb, ab)) = sorted(sites.items())

    mutated = _swap_kinds(lines, la, lb)

    # marginal 1 -- every per-kind count is bitwise identical, so
    # `journalled_cells(wing) > 0` and the arm_phase zero-calibration hold.
    assert _kind_counts(mutated) == _kind_counts(lines), "the mutation is not marginal-preserving"
    for arm in WING_ARM.values():
        assert _kind_counts(mutated)[arm] > 0
    assert _kind_counts(mutated)["arm_phase"] == 0

    # marginal 2 -- every anchor still sits on its own cited line.
    assert aa in mutated[la - 1], "the mutation moved an anchor; it is not marginal-preserving"
    assert ab in mutated[lb - 1], "the mutation moved an anchor; it is not marginal-preserving"

    # marginal 3 -- the declared digest is over manifest rows, which never moved.
    assert freeze.digest(freeze.rows()) == freeze.SHA_LINE.search(freeze._text()).group(1)

    # the joint -- the only thing that moved, and nothing at it.4 was watching it.
    assert _kind_at(mutated, la) == WING_ARM[wb], "the swap did not land"
    assert _kind_at(mutated, lb) == WING_ARM[wa], "the swap did not land"
    assert _kind_at(mutated, la) != WING_ARM[wa], (
        "V-26: the manifest certifies the wrong wing's record and every it.4 node is green")


def test_the_joint_clause_b_check_is_shipped_in_the_it4_node_file():
    """Every kill ships a replacement route, and the route lives in the
    instrument it repairs -- not only in this iteration's file."""
    src = (ROOT / "tests" / "saturn" / "test_v20_r15_freeze_manifest.py").read_text(
        encoding="utf-8", errors="replace")
    assert "def test_every_clause_b_line_journals_that_wings_own_arm" in src, (
        "the it.4 file carries no joint clause-(b) node; the strike has no repair")


def _swap_row_bodies(text: str, ra: str, rb: str) -> str:
    """MARGINAL-PRESERVING MUTATION for a per-row digest set: exchange the
    BODIES of two rows, keeping their ids in place. The multiset of body texts
    -- and therefore the multiset of digests -- is bitwise unchanged."""
    def row(rid: str) -> tuple[str, str]:
        m = re.search(rf"^\|\s*\*\*{re.escape(rid)}\*\*\s*\|(.*)\|\s*$", text, re.M)
        assert m, f"no row {rid}"
        return m.group(0), m.group(1)

    line_a, body_a = row(ra)
    line_b, body_b = row(rb)
    new_a = line_a.replace(body_a, body_b, 1)
    new_b = line_b.replace(body_b, body_a, 1)
    return text.replace(line_a, "\x00", 1).replace(line_b, new_b, 1).replace("\x00", new_a, 1)


def test_the_it12_ledger_baseline_is_joint_and_fires_on_the_same_class_of_mutation():
    """THE CONTROL, and the office's other instrument CLEARS.

    A checker that compared digest SETS would be blind to a body swap, because
    the multiset of digests is preserved by construction. This one keys each
    digest by the row id parsed out of the row itself, so the pair moves and
    the checker names both rows.
    """
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    mutated = _swap_row_bodies(text, "L-13", "L-14")
    assert mutated != text, "the planted swap did not apply; the test is vacuous"

    # marginal preserved: the multiset of row BODIES is bitwise identical, so a
    # checker comparing a set of body digests would report nothing at all.
    def bodies(t: str) -> list[str]:
        return sorted(" ".join(m.group(2).split())
                      for m in map(it12.ROW_RE.match, t.splitlines()) if m)
    assert bodies(mutated) == bodies(text), "the swap was not marginal-preserving"
    after = it12.ledger_rows(mutated)

    declared = it12.declared_digests()
    moved = [k for k in sorted(after) if it12.row_digest(after[k]) != declared[k]]
    assert moved == ["L-13", "L-14"], (
        f"the per-row baseline is not joint: a body swap moved {moved}")


def test_the_coverage_node_carries_no_vacuous_disjunct():
    """This office's own it.12 file shipped `... or n_table > 0`, which is the
    `or True` class its own census counted in two other suites. Repaired there;
    asserted gone here."""
    src = (ROOT / "tests" / "saturn" / "test_v20_r15_it12_saturn.py").read_text(
        encoding="utf-8", errors="replace")
    assert "or n_table > 0" not in src, (
        "the it.12 coverage node still short-circuits its own assertion")


def test_dropping_a_row_from_the_digest_block_is_caught():
    """PLANTED NEGATIVE for the repaired coverage node: a freeze over a subset
    is not a freeze."""
    block = it12.DIGEST_BLOCK.search(
        LEDGER.read_text(encoding="utf-8", errors="replace")).group(1)
    thinned = "\n".join(l for l in block.splitlines() if not l.startswith("L-9 "))
    assert thinned != block, "the planted deletion did not apply"
    assert set(dict(it12.DIGEST_RE.findall(thinned))) != set(it12.ledger_rows())


# ==========================================================================
# B.  THE THEORY TABLE FREEZE  --  the it.4 shape, carried forward
# ==========================================================================

#: `| Q4/W1 | ... |` -- a per-cell row.
CELL_RE = re.compile(r"^\|\s*\*{0,2}(Q\d+\s*/\s*W\d+)\*{0,2}\s*\|(.*)\|\s*$")
#: `| **Q2 OUTSIDE** | **F4** ... | **F1 + const** |` -- the §1 grade matrix,
#: one row per question and one column per wing. Twelve cells, six lines.
MATRIX_RE = re.compile(r"^\|\s*\*\*(Q\d+)(?!\s*/)[^|*]*\*\*\s*\|([^|]*)\|([^|]*)\|\s*$")
#: The wing ids in matrix column order, read off the matrix header rather than
#: written down -- a hardcoded column order is a second place to be wrong.
HEADER_RE = re.compile(r"^\|\s*\|\s*\*\*(W\d+)[^|]*\|\s*\*\*(W\d+)[^|]*\|\s*$")


CELLS_12 = frozenset(f"Q{q}/W{w}" for q in range(1, 7) for w in (1, 3))


def theory_cells(text: str) -> dict[str, str]:
    """`Qn/Wm` -> whitespace-normalised cell text. Cells only.

    Both shapes the table uses are read: the §1 grade matrix (six rows x two
    wing columns) and any per-cell `| Qn/Wm | ... |` row. Keyed by cell, so the
    digest below is JOINT over (cell, content) -- exchanging two cells' grades
    moves it, which is the `V-26` property this instrument is built to have.
    """
    out: dict[str, str] = {}
    cols: tuple[str, str] | None = None
    for line in text.splitlines():
        h = HEADER_RE.match(line)
        if h:
            cols = (h.group(1), h.group(2))
            continue
        m = MATRIX_RE.match(line)
        if m and cols:
            for wing, cell in zip(cols, (m.group(2), m.group(3))):
                out[f"{m.group(1)}/{wing}"] = " ".join(cell.split())
            continue
        m = CELL_RE.match(line)
        if m:
            out.setdefault(re.sub(r"\s+", "", m.group(1)), " ".join(line.split()))
    return out


def theory_digest(cells: dict[str, str]) -> str:
    """sha256 over the sorted `id|row` pairs -- joint by construction, so a
    swap of two cells' bodies moves it. Rows only; see the report's LIMITS."""
    body = "\n".join(f"{k}|{v}" for k, v in sorted(cells.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_the_theory_table_is_filed_and_carries_twelve_cells():
    """STANDING RED until JUPITER files the table. It is the leap's primary
    input at it.35 and the arena's ticket at it.15; an unfiled primary input is
    a finding, not a skip."""
    assert THEORY.is_file(), (
        f"{THEORY.name} does not exist at HEAD. The twelve-cell table is the "
        f"it.35 leap input and the it.15 arena ticket, and it cannot be frozen "
        f"because it is not filed.")
    cells = theory_cells(THEORY.read_text(encoding="utf-8", errors="replace"))
    # it.31 RULE 1: partitioned. The table is a corpus the round is still
    # writing; `== 12` reads a growing subject. The closed claim is the
    # twelve NAMES, and it survives a thirteenth cell being added.
    missing = sorted(CELLS_12 - set(cells))
    assert not missing, f"the table is missing {missing}; read {sorted(cells)}"


def test_the_theory_digest_instrument_fires_on_a_planted_cell_edit(tmp_path):
    """CALIBRATION, on a table PLANTED ON DISK rather than simulated -- the it.4
    shape, whose planted negative still fires ten iterations on.

    The instrument must (i) move its whole-table digest and (ii) NAME the cell
    that moved. Detection is not the bar; location is.
    """
    planted = tmp_path / "PLANTED_THEORY_TABLE.md"
    rows = [f"| **Q{q} / W{w}** | claim {q}{w} | **F{q % 5}** | verdict {q}{w} |"
            for q in range(1, 7) for w in (1, 3)]
    planted.write_text("| cell | claim | grade | verdict |\n|---|---|---|---|\n"
                       + "\n".join(rows) + "\n", encoding="utf-8")

    before = theory_cells(planted.read_text(encoding="utf-8"))
    assert len(before) == 12, sorted(before)
    d_before = theory_digest(before)

    planted.write_text(planted.read_text(encoding="utf-8").replace(
        "| claim 41 |", "| claim 41 EDITED |", 1), encoding="utf-8")
    after = theory_cells(planted.read_text(encoding="utf-8"))
    assert after != before, "the planted edit did not apply; the calibration is vacuous"

    assert theory_digest(after) != d_before, "the whole-table digest is blind to a cell edit"
    moved = [k for k in sorted(after) if it12.row_digest(after[k]) != it12.row_digest(before[k])]
    assert moved == ["Q4/W1"], moved


def test_the_theory_digest_is_joint_over_cell_and_row(tmp_path):
    """V-26 applied to the instrument this iteration ships, before it is used.
    Exchange two cells' bodies: the multiset of bodies is preserved, so a digest
    over the body SET would be blind. This digest keys by cell id, so it moves.
    """
    planted = tmp_path / "PLANTED_THEORY_TABLE.md"
    rows = [f"| **Q{q} / W{w}** | claim {q}{w} | **F{q % 5}** | verdict {q}{w} |"
            for q in range(1, 7) for w in (1, 3)]
    planted.write_text("\n".join(rows) + "\n", encoding="utf-8")
    before = theory_cells(planted.read_text(encoding="utf-8"))

    swapped = dict(before)
    a, b = "Q1/W1", "Q6/W3"
    swapped[a], swapped[b] = before[b].replace("Q6 / W3", "Q1 / W1"), \
        before[a].replace("Q1 / W1", "Q6 / W3")
    assert sorted(v.replace("Q1 / W1", "X").replace("Q6 / W3", "X") for v in swapped.values()) \
        == sorted(v.replace("Q1 / W1", "X").replace("Q6 / W3", "X") for v in before.values()), \
        "the swap was not body-preserving"
    assert theory_digest(swapped) != theory_digest(before), (
        "the theory digest is blind to a cell swap: it is a set digest, not a joint one")


# ==========================================================================
# C.  THE LEDGER ARITHMETIC THE it.35 GATE READS
# ==========================================================================

GRADE_RE = re.compile(r"\bF[0-4]\b")


def f4_rows_with_a_gate_verdict() -> dict[str, str]:
    """Rows the it.35 gate cannot consume. `CEQ_V20_R15_CONTRACT.md:142-144`
    sorts every **F1/F2/F3** failure; these carry an F4 grade with a
    LEAPABLE/TERMINAL verdict already written."""
    out = {}
    for rid, row in it12.ledger_rows().items():
        if rid == "V-it7":
            grades, verdict = set(GRADE_RE.findall(row)), "LEAPABLE"
        else:
            cells = [c.strip() for c in row.strip("| ").split("|")]
            grades = set(GRADE_RE.findall(" ".join(cells[:3])))
            verdict = it12.verdict_cell(row)
        if "F4" in grades and ("LEAPABLE" in verdict or "TERMINAL" in verdict):
            out[rid] = verdict[:70]
    return out


#: Frozen at it.14. SATURN reported five at it.12; the it.12 Inspector struck
#: that count and named L-8 as the sixth. The detector is run here rather than
#: either number carried, and it returns the Inspector's six.
KNOWN_F4_AT_THE_GATE = {"L-3", "L-4", "L-7", "L-8", "L-13", "L-14"}


def test_the_F4_rows_the_gate_cannot_consume_are_counted_by_a_detector():
    found = f4_rows_with_a_gate_verdict()
    assert set(found) == KNOWN_F4_AT_THE_GATE, (
        f"the F4-at-the-gate set moved. new={sorted(set(found) - KNOWN_F4_AT_THE_GATE)} "
        f"gone={sorted(KNOWN_F4_AT_THE_GATE - set(found))}; details={found}")


def test_Q2_W1_now_has_its_ledger_row_and_the_recount_is_seven():
    """RETIRED AND REPLACED at it.20. The node this replaces asserted Q2/W1 had
    *no* ledger row and said it turns red the moment one is filed. JUPITER filed
    `L-16` at it.19 and wrote the recount into `V20_R15_LEAP_LEDGER.md` -- so the
    red was the repair landing, and leaving it red across an iteration boundary
    would publish a repair as a defect. The recount, bound rather than quoted:
    **seven** F4 objects, **six** unconsumable verdicts, because `L-16`'s verdict
    cell is `NOT-PUT` and carries neither LEAPABLE nor TERMINAL."""
    rows = it12.ledger_rows()
    hit = [rid for rid, row in rows.items() if re.search(r"Q2\s*/\s*W1", row)]
    assert hit == ["L-16"], f"Q2/W1's ledger row moved or multiplied: {hit}"
    assert "F4" in set(GRADE_RE.findall(rows["L-16"])), f"L-16 is not F4: {rows['L-16'][:90]}"
    assert "L-16" not in f4_rows_with_a_gate_verdict(), (
        "L-16's verdict now reads LEAPABLE/TERMINAL: the unconsumable count is "
        "seven, not six, and KNOWN_F4_AT_THE_GATE is stale")


M14_BLOCK = re.compile(r"^M14 CHEEGER STRATIFICATION.*?(?=^M15 )", re.M | re.S)
#: The only legal resolution of a pre-registration: superseded in place, the
#: pre-registered sentence left standing rather than rewritten (C4's class).
SUPERSEDED_RE = re.compile(r"\[SUPERSEDED[^\]]*?M14 IS (F[0-4])", re.S)
JOURNAL_M14_RE = re.compile(r"Annex:?\s*M14 at \*{0,2}(F[0-4])")


def live_M14_grades(contract: str | None = None, journal: str | None = None) -> dict[str, str]:
    """M14's LIVE grade in each file that still carries one.

    **REPAIRED at it.20 (JUPITER's strike).** The retired node
    `test_M14_carries_three_grades_in_three_files` bound the presence of three
    TEXTS, so the ledger's claim that *"a repair in any of the three files turns
    it red"* was false for the only repair shape a pre-registration permits: a
    pre-registered sentence may not be rewritten, superseding in place is the
    sole legal move, and a presence check walks straight past it. This binds the
    grade TOKENS instead -- a superseded grade is not returned, and the grade its
    supersession declares is."""
    contract = CONTRACT.read_text(encoding="utf-8", errors="replace") if contract is None else contract
    journal = JOURNAL.read_text(encoding="utf-8", errors="replace") if journal is None else journal
    out: dict[str, str] = {}

    m = M14_BLOCK.search(contract)
    assert m, "the contract's M14 block moved; re-derive the pattern"
    sup = SUPERSEDED_RE.search(m.group(0))
    if sup:
        out["contract"] = sup.group(1)
    else:
        pre = re.search(r"grade (F[0-4])", m.group(0))
        assert pre, "the contract's M14 block carries no grade at all"
        out["contract"] = pre.group(1)

    l3 = it12.ledger_rows()["L-3"]
    assert "M14" in l3, f"L-3 is no longer M14's row: {l3[:80]}"
    grades = set(GRADE_RE.findall(l3))
    assert len(grades) == 1, f"L-3 carries more than one grade: {sorted(grades)}"
    out["ledger"] = grades.pop()

    j = JOURNAL_M14_RE.search(journal)
    # The journal is append-only, so its F1 line cannot be edited away. It dies
    # by being indexed: a CORRECTIONS INDEX row that quotes it retires it.
    if j and not re.search(r"^\| C\d+ \|.*Annex:?\s*M14 at \*{0,2}F1", journal, re.M):
        out["journal"] = j.group(1)
    return out


def test_M14_carries_exactly_one_live_grade(contract: str | None = None,
                                            journal: str | None = None):
    """The event this node exists to witness is the RESOLUTION, and it fires on
    it: three live grades before it.19, one after. `RULING J-17d`, `C20`, and the
    contract's `[SUPERSEDED it.19]` block are what moved it."""
    live = live_M14_grades(contract=contract, journal=journal)
    assert set(live.values()) == {"F4"}, (
        f"M14's live grades disagree across the files the it.35 gate reads: {live}. "
        f"The one grade and its ground: V20_R15_LEAP_LEDGER.md, '## RULING J-17d'")
    assert "journal" not in live, (
        "the journal's scoreless SCOREBOARD F1 reads as live again: its CORRECTIONS "
        "INDEX row is gone")


def test_the_M14_contradiction_is_flagged_inside_the_ledger():
    """The Inspector approved flagging inside the ledger over repairing another
    office's row. The flag must name both grades, both files, and the owner."""
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"## FLAG — M14 CARRIES THREE GRADES.*?(?=\n## |\Z)", text, re.S)
    assert m, "the ledger carries no M14 grade-contradiction flag"
    for token in ("F3", "F4", "F1", "CEQ_V20_R15_CONTRACT.md", "V20_R15_JOURNAL.md", "JUPITER"):
        assert token in m.group(0), f"the flag does not name {token}"


def test_the_flag_moved_no_row_digest():
    """Appending prose to the ledger must not disturb the it.12 baseline; if it
    did, the flag would be an in-place edit wearing a note's clothes."""
    declared, actual = it12.declared_digests(), it12.ledger_rows()
    assert set(declared) == set(actual)
    assert [k for k in sorted(actual) if it12.row_digest(actual[k]) != declared[k]] == []


def test_the_five_inadmissible_FIELD_rows_are_still_unrepaired():
    assert set(it12.inadmissible_leapable_rows()) == it12.KNOWN_INADMISSIBLE


REPORT = ROOT / "V20_R15_IT14_SATURN.md"
FREEZE_BLOCK = re.compile(r"```theory-freeze\n(.*?)```", re.DOTALL)
THEORY_SHA_RE = re.compile(r"^THEORY-SHA256\s*=\s*([0-9a-f]{64})\s*$", re.M)
THEORY_CELL_RE = re.compile(r"^(Q\d+/W\d+)\s+([0-9a-f]{16})\s*$", re.M)


def declared_theory() -> tuple[str, dict[str, str]]:
    m = FREEZE_BLOCK.search(REPORT.read_text(encoding="utf-8", errors="replace"))
    assert m, "V20_R15_IT14_SATURN.md carries no ```theory-freeze block"
    sha = THEORY_SHA_RE.search(m.group(1))
    assert sha, "the theory-freeze block declares no THEORY-SHA256"
    return sha.group(1), dict(THEORY_CELL_RE.findall(m.group(1)))


def test_the_declared_theory_digest_matches_the_table_at_head():
    """The it.4 shape: a digest over sorted rows, declared in a file this
    office owns, over a table another office owns."""
    cells = theory_cells(THEORY.read_text(encoding="utf-8", errors="replace"))
    sha, per_cell = declared_theory()
    assert set(per_cell) == set(cells), (
        f"cells added/removed since the freeze: added={sorted(set(cells) - set(per_cell))} "
        f"removed={sorted(set(per_cell) - set(cells))}")
    moved = [k for k in sorted(cells) if it12.row_digest(cells[k]) != per_cell[k]]
    assert moved == [], f"cells edited in place since the freeze: {moved}"
    assert theory_digest(cells) == sha, "the whole-table digest does not match the declaration"


def test_the_planted_negative_is_applied_to_the_real_table_on_disk(tmp_path):
    """PLANTED NEGATIVE against the FILED table, copied to disk and edited
    there rather than simulated in memory -- the it.4 discipline. One grade is
    changed; the instrument must move the whole-table digest and name the cell.
    """
    copy = tmp_path / THEORY.name
    src = THEORY.read_text(encoding="utf-8", errors="replace")
    copy.write_text(src, encoding="utf-8")
    before = theory_cells(copy.read_text(encoding="utf-8", errors="replace"))

    copy.write_text(copy.read_text(encoding="utf-8", errors="replace").replace(
        "| **Q3 LEARNABILITY** | **F2** |", "| **Q3 LEARNABILITY** | **F1** |", 1),
        encoding="utf-8")
    after = theory_cells(copy.read_text(encoding="utf-8", errors="replace"))
    assert after != before, "the planted edit did not apply; the test is vacuous"

    assert theory_digest(after) != theory_digest(before)
    moved = [k for k in sorted(after) if it12.row_digest(after[k]) != it12.row_digest(before[k])]
    assert moved == ["Q3/W1"], moved
