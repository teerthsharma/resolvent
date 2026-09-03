"""SATURN it.19 REPAIR 3 -- the theory digest is blind to what the leap reads.

JUPITER's it.17/it.18 finding, twice: he made 29 citation repairs INSIDE the twelve
`### CELL` bodies of V20_R15_THEORY_TABLE.md and `THEORY-SHA256` came back
bit-identical, `9989f0ef...c63057`, both times. `theory_cells()` reads only the
twelve section-1 grade tokens -- its own docstring closes "Rows only" -- so the
freeze covers the grade matrix and nothing the grades are FOR. The it.35 leap and
the it.15 arena read the cell bodies.

Worse, the per-cell keys collide. `it12.row_digest` hashes the row VALUE alone, so
twelve cells yield seven distinct digests: Q4/W1 and Q4/W3 hash the same, Q6/W1 and
Q6/W3 hash the same. A digest that cannot name which of two wings moved cannot
locate anything, and location was the bar the it.4 instrument set for itself.

  RED 1  two different wings' cells are byte-identical in the grade matrix, so the
         digest of record cannot tell them apart. The CELL BODIES differ; the
         instrument is reading the wrong object.

  RED 2  editing a citation inside a `### CELL` body must move the digest of
         record. Against `theory_digest(theory_cells(...))` it does not.

  RED 3  the twelve per-cell digests must be twelve distinct values. The row
         digest yields seven.

RED 1-3 were measured against the SHIPPED instrument and are retained here with the
superseded behaviour pinned as a premise, so a re-adoption of the row digest fires
them rather than passing quietly. RED 4 is the adoption itself.

  RED 4  the digest of record must be DECLARED in the it.19 record and match the
         table at HEAD, per-cell and whole-table.

Run:  python -m pytest tests/saturn/test_v20_r15_it19_theory_digest.py -x -q
"""
from __future__ import annotations

import hashlib
import pathlib
import re

from tests.saturn import test_v20_r15_it12_saturn as it12
from tests.saturn import test_v20_r15_it14_saturn as it14
from tests.saturn import test_v20_r15_it27_subject_provenance as it27

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"
REPORT = ROOT / "V20_R15_IT19_SATURN.md"

#: `### CELL Q4/W1 -- COST LAW` opens a cell body.
CELL_HEAD = re.compile(r"^###\s+CELL\s+(Q\d+\s*/\s*W\d+)\b")
#: The declared block, in the record this office owns.
FREEZE_BLOCK = re.compile(r"```theory-cells-freeze\n(.*?)```", re.DOTALL)
CELLS_SHA_RE = re.compile(r"^THEORY-CELLS-SHA256\s*=\s*([0-9a-f]{64})\s*$", re.M)
PER_CELL_RE = re.compile(r"^(Q\d+/W\d+)\s+([0-9a-f]{16})\s*$", re.M)


def cell_bodies(text: str) -> dict[str, str]:
    """`Qn/Wm` -> the whitespace-normalised body of its `### CELL` section.

    This is what the leap reads. `theory_cells()` reads the grade token, which is
    what the leap is graded ON -- two different objects, and only one was frozen.
    """
    out: dict[str, str] = {}
    key: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = CELL_HEAD.match(line)
        if m or (key is not None and line.startswith("## ")):
            if key is not None:
                out[key] = " ".join("\n".join(buf).split())
            key = re.sub(r"\s+", "", m.group(1)) if m else None
            buf = []
            continue
        if key is not None:
            buf.append(line)
    if key is not None:
        out[key] = " ".join("\n".join(buf).split())
    return out


def cell_digest(key: str, body: str) -> str:
    """KEYED, so two cells with equal content cannot share a digest. `row_digest`
    hashes the value alone and gave 7 distinct digests for 12 cells."""
    return hashlib.sha256(f"{key}|{body}".encode("utf-8")).hexdigest()[:16]


def cells_digest(bodies: dict[str, str]) -> str:
    """sha256 over the sorted keyed per-cell digests -- joint over (cell, body)."""
    body = "\n".join(f"{k}|{cell_digest(k, v)}" for k, v in sorted(bodies.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


CELLS_12 = frozenset(f"Q{q}/W{w}" for q in range(1, 7) for w in (1, 3))


def declared() -> tuple[str, dict[str, str]]:
    assert REPORT.is_file(), f"{REPORT.name} does not exist; the digest is undeclared"
    m = FREEZE_BLOCK.search(REPORT.read_text(encoding="utf-8", errors="replace"))
    assert m, f"{REPORT.name} carries no theory-cells-freeze block"
    sha = CELLS_SHA_RE.search(m.group(1))
    assert sha, "the block declares no THEORY-CELLS-SHA256"
    return sha.group(1), dict(PER_CELL_RE.findall(m.group(1)))


def test_the_table_carries_twelve_cell_bodies():
    bodies = cell_bodies(TABLE.read_text(encoding="utf-8", errors="replace"))
    # it.31 RULE 1: partitioned, for the reason at it.14:283.
    missing = sorted(CELLS_12 - set(bodies))
    assert not missing, f"no body for {missing}; read {sorted(bodies)}"
    assert all(v for v in bodies.values()), "a cell body read empty"


def test_two_wings_cells_are_distinguishable_by_the_digest_of_record():
    """RED 1, and the reason the instrument was swapped rather than patched.

    Q4/W1 and Q4/W3 are BYTE-IDENTICAL in the grade matrix, and so are Q6/W1 and
    Q6/W3, so no digest over the grade rows can tell the two wings apart -- swap
    them and nothing moves. The bodies differ, so the adopted digest can.
    """
    text = TABLE.read_text(encoding="utf-8", errors="replace")
    rows = it14.theory_cells(text)
    bodies = cell_bodies(text)
    for a, b in (("Q4/W1", "Q4/W3"), ("Q6/W1", "Q6/W3")):
        assert rows[a] == rows[b], (
            f"premise gone: {a} and {b} no longer collide in the grade matrix; "
            "re-derive the strike")
        assert bodies[a] != bodies[b], (
            f"{a} and {b} have identical BODIES too. The adopted digest is as blind "
            "as the one it replaces.")
        assert cell_digest(a, bodies[a]) != cell_digest(b, bodies[b])


def test_a_citation_edit_inside_a_cell_body_moves_the_digest(tmp_path):
    """RED 2: JUPITER made 29 such edits and the digest was bit-identical."""
    copy = tmp_path / TABLE.name
    src = TABLE.read_text(encoding="utf-8", errors="replace")
    before = it14.theory_digest(it14.theory_cells(src))

    lines = src.splitlines()
    heads = [i for i, l in enumerate(lines) if CELL_HEAD.match(l)]
    assert heads, "no `### CELL` headings; re-derive the strike"
    lines[heads[0] + 2] = lines[heads[0] + 2] + " EDITED-CITATION"
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mutated = copy.read_text(encoding="utf-8")
    assert mutated != src, "the planted edit did not apply; the test is vacuous"
    assert cell_bodies(mutated) != cell_bodies(src), "the edit missed every cell body"

    # The superseded instrument, pinned as a fact rather than left to memory: this
    # is exactly what JUPITER measured 29 times over, and if anyone re-adopts the
    # row digest this node says so.
    assert it14.theory_digest(it14.theory_cells(mutated)) == before, (
        "premise gone: the it.14 digest now moves on a cell-body edit; re-derive")
    assert cells_digest(cell_bodies(mutated)) != cells_digest(cell_bodies(src)), (
        "a citation edited inside a `### CELL` body moves nothing in the ADOPTED "
        "digest either. The freeze would cover no line of what the leap reads.")


def test_the_twelve_per_cell_digests_are_twelve_distinct_values():
    """RED 3: `row_digest` hashes the value alone -- 7 distinct for 12 cells."""
    text = TABLE.read_text(encoding="utf-8", errors="replace")
    rows = it14.theory_cells(text)
    unkeyed = {k: it12.row_digest(v) for k, v in rows.items()}
    dupes = sorted(k for k in unkeyed
                   if list(unkeyed.values()).count(unkeyed[k]) > 1)
    assert len(set(unkeyed.values())) == 7, (
        f"premise gone: the row digest now yields "
        f"{len(set(unkeyed.values()))} distinct values, not 7; re-derive the strike")
    # JUPITER named Q4 and Q6 by hand; the census is wider -- EIGHT of twelve cells
    # share a digest with another cell, so two thirds of the table is unlocatable.
    assert len(dupes) == 8, dupes
    assert {"Q4/W1", "Q4/W3", "Q6/W1", "Q6/W3"} <= set(dupes), dupes
    keyed = {k: cell_digest(k, v) for k, v in cell_bodies(text).items()}
    assert len(set(keyed.values())) == 12, (
        f"the adopted per-cell digests collide too: "
        f"{sorted(k for k in keyed if list(keyed.values()).count(keyed[k]) > 1)}")


def test_the_keyed_per_cell_digests_cannot_collide():
    """The repair's own side of RED 3, on the bodies."""
    bodies = cell_bodies(TABLE.read_text(encoding="utf-8", errors="replace"))
    keyed = {k: cell_digest(k, v) for k, v in bodies.items()}
    assert len(set(keyed.values())) == len(keyed) == 12, keyed


def test_the_declared_cells_digest_matches_the_table_at_head():
    """RED 4: declared in the record this office owns, over a table JUPITER owns.

    it.27 REPAIR: the subject is a file another office edits mid-iteration, so every
    verdict below carries the subject's digest, mtime and size, and a subject that
    moves ACROSS the read is a REFUSAL rather than a verdict rendered off stale bytes.
    The INSPECTOR's it.25 ruling, conceded at it.26 and unrepaired until here.
    """
    sub = it27.read_subject(TABLE)
    where = it27.provenance(sub)
    assert not sub["torn"], (
        f"REFUSED: the subject moved across the read -- {where}. No comparison to the "
        "freeze is available; this is not the same finding as a moved cell body.")
    bodies = cell_bodies(sub["text"])
    sha, per_cell = declared()
    assert set(per_cell) == set(bodies), (
        f"added={sorted(set(bodies) - set(per_cell))} "
        f"removed={sorted(set(per_cell) - set(bodies))} [{where}]")
    moved = [k for k in sorted(bodies) if cell_digest(k, bodies[k]) != per_cell[k]]
    assert moved == [], f"cell bodies edited since the freeze: {moved} [{where}]"
    assert cells_digest(bodies) == sha, f"the whole-cells digest does not match [{where}]"


def test_the_planted_negative_fires_against_the_declaration(tmp_path):
    """PLANTED NEGATIVE, applied on disk: one citation changed inside one body.
    The instrument must move the whole digest AND name the one cell."""
    copy = tmp_path / TABLE.name
    src = TABLE.read_text(encoding="utf-8", errors="replace")
    before = cell_bodies(src)
    lines = src.splitlines()
    heads = [i for i, l in enumerate(lines) if CELL_HEAD.match(l)]
    target = re.sub(r"\s+", "", CELL_HEAD.match(lines[heads[0]]).group(1))
    lines[heads[0] + 2] = lines[heads[0] + 2] + " EDITED-CITATION"
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = cell_bodies(copy.read_text(encoding="utf-8"))
    assert after != before, "the planted edit did not apply"
    assert cells_digest(after) != cells_digest(before), "the digest is blind to the edit"
    moved = [k for k in sorted(after) if cell_digest(k, after[k]) != cell_digest(k, before[k])]
    assert moved == [target], moved
