"""it.27 SATURN REPAIR 2 -- a verdict must carry the SUBJECT it was rendered against.

The INSPECTOR named this at it.25 and this office confirmed it at it.26 and repaired
nothing:

    `test_the_declared_cells_digest_matches_the_table_at_head` -- whose subject is a
    file JUPITER was editing mid-iteration.

The it.26 reading: cell bodies edited since the freeze were
`['Q1/W1', 'Q2/W3', 'Q3/W1', 'Q6/W1', 'Q6/W3']`, against a table whose mtime was
`12:23:28Z`, DURING that iteration. The node is deterministic against the bytes on
disk. The bytes on disk are not a fixed subject.

The named fix, built here: **record the subject's digest AND its mtime with the
verdict**, and REFUSE rather than report when the subject moves across the read.

This is an instance of the INSPECTOR's ruling that *provenance authenticates that a
reading happened, never that the subject had a value.* A digest-plus-mtime verdict is
one of the few devices this round has that authenticates the SUBJECT: it does not say
the table was correct, it says exactly which bytes the verdict is about, so a later
reader can tell a stale verdict from a moved subject instead of guessing.

  RED 2  the shipped it.19 verdict node renders `moved == []` without recording one
         byte of subject identity. Two runs against two different tables produce two
         indistinguishable verdicts.

Run:  python -m pytest tests/saturn/test_v20_r15_it27_subject_provenance.py -x -q
"""
from __future__ import annotations

import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"
IT19 = ROOT / "tests" / "saturn" / "test_v20_r15_it19_theory_digest.py"


def read_subject(path: pathlib.Path, _on_read=None) -> dict:
    """Read `path` and return the bytes WITH the identity of what was read.

    stat, read, stat. If the second stat disagrees with the first, the subject moved
    across the read and no verdict is available -- `torn` is True and the caller must
    refuse rather than report. `_on_read` is the seam that makes that refusal
    testable: it fires between the read and the second stat, which is exactly the
    window a concurrent editor occupies.

    What this CANNOT see, stated because the round's habit is to state it: an edit
    that lands and completes between two ticks of the filesystem's mtime clock while
    leaving the size unchanged is invisible here. It authenticates WHICH BYTES the
    verdict is about; it does not authenticate that those bytes were ever correct,
    and it does not authenticate who wrote them.
    """
    before = path.stat()
    data = path.read_bytes()
    if _on_read is not None:
        _on_read(path)
    after = path.stat()
    return {
        "path": str(path.relative_to(ROOT)) if ROOT in path.parents else path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_ns": before.st_mtime_ns,
        "size": len(data),
        "torn": (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size),
        "text": data.decode("utf-8", errors="replace"),
    }


def provenance(sub: dict) -> str:
    """The line that goes beside every count this office publishes off a subject."""
    return (f"{sub['path']} sha256={sub['sha256'][:16]} "
            f"mtime_ns={sub['mtime_ns']} size={sub['size']}")


# ==========================================================================
#  RED 2 -- the shipped verdict names no subject
# ==========================================================================

def test_the_head_verdict_records_the_subject_it_was_rendered_against():
    """RED 2, against the shipped it.19 node, unmutated.

    Its verdict is `moved == [...]` and its message carries the cell names and
    nothing else. Rendered at 12:23Z against a table JUPITER was editing, and
    rendered a minute later against a different table, the two verdicts are
    byte-identical. That is a reading whose subject is unrecoverable.
    """
    src = IT19.read_text(encoding="utf-8", errors="replace")
    node = re.search(
        r"def test_the_declared_cells_digest_matches_the_table_at_head\(.*?\n(?=\n\ndef |\Z)",
        src, re.DOTALL)
    assert node, "the it.19 verdict node is gone; re-derive the strike"
    body = node.group(0)
    assert "provenance(" in body or "read_subject(" in body, (
        "the it.19 HEAD verdict renders `moved` against `TABLE.read_text()` and "
        "records no digest, no mtime and no size of the subject. JUPITER is editing "
        "V20_R15_THEORY_TABLE.md this iteration, as he was at it.25 when the "
        "INSPECTOR filed this; the it.26 reading `['Q1/W1', 'Q2/W3', 'Q3/W1', "
        "'Q6/W1', 'Q6/W3']` cannot be re-associated with the bytes it was taken "
        "from, so it can be neither reproduced nor refuted. A verdict without its "
        "subject's identity is a claim that a reading happened, not a reading.")


# ==========================================================================
#  The repaired reader, and what it refuses
# ==========================================================================

def test_the_subject_read_is_recorded_and_untorn_on_the_live_table():
    """The instrument measures a MOVING file this iteration -- that is the test
    case, not the obstacle. On a quiet read it must produce a full identity."""
    sub = read_subject(TABLE)
    assert not sub["torn"], f"the table moved across the read: {provenance(sub)}"
    assert len(sub["sha256"]) == 64 and sub["size"] > 0, sub
    assert re.fullmatch(r"[\w./\\-]+ sha256=[0-9a-f]{16} mtime_ns=\d+ size=\d+",
                        provenance(sub)), provenance(sub)


def test_a_subject_that_moves_across_the_read_is_a_refusal_not_a_verdict(tmp_path):
    """PLANTED NEGATIVE, applied on disk: the editor lands inside the read window.

    The old node would have rendered a confident `moved == [...]` off the first
    half of the file. This one refuses, and the refusal names both mtimes.
    """
    copy = tmp_path / TABLE.name
    copy.write_text("cell body A\n", encoding="utf-8")
    quiet = read_subject(copy)
    assert not quiet["torn"], "the control read was already torn; the test is vacuous"

    def editor(path: pathlib.Path) -> None:
        st = path.stat()
        path.write_text("cell body A -- EDITED MID-READ\n", encoding="utf-8")
        assert path.stat().st_mtime_ns != st.st_mtime_ns or path.stat().st_size != st.st_size, (
            "the planted edit moved neither mtime nor size; the seam is vacuous")

    torn = read_subject(copy, _on_read=editor)
    assert torn["torn"], (
        "an edit landing inside the read window left the subject identity unchanged; "
        "the refusal cannot fire and the verdict would be rendered off stale bytes")
    assert torn["sha256"] == quiet["sha256"], (
        "premise check: the bytes returned are the PRE-edit bytes -- which is exactly "
        "why the read must be refused rather than reported")


def test_the_refusal_is_distinguishable_from_a_moved_cell_body():
    """The whole point of the repair: two different failures, two different verdicts.

    `moved == ['Q6/W1']` means the subject sat still and its content disagrees with
    the freeze. `torn` means no comparison was possible. Before this repair both
    arrived as the same sentence.
    """
    sub = read_subject(TABLE)
    verdict = "REFUSED (subject moved across the read)" if sub["torn"] else "COMPARABLE"
    assert verdict == "COMPARABLE", provenance(sub)
    # and the identity travels with it, so the verdict is re-associable to bytes
    assert TABLE.name in provenance(sub), provenance(sub)
    assert sub["sha256"] == hashlib.sha256(TABLE.read_bytes()).hexdigest(), (
        "the recorded digest is not the digest of the subject on disk")
