"""MARS, V20 R15 it.25 -- three strikes against the it.24 repairs.

Each node is RED against UNMUTATED code at HEAD 207e7b9 and carries a control
so a future GREEN cannot be an artefact of the probe failing everything.

STRIKE A  SATURN it.24 published `tests/mars_v20 31 failed, 59 passed` "at HEAD
          207e7b9". Not one file of tests/mars_v20 is known to git, so a HEAD
          SHA identifies nothing about the tree that produced that count. He
          proved `git diff` is blind to untracked files in the same filing and
          then used a git SHA as the provenance for a count over them.

STRIKE B  JUPITER it.24 withdrew C63 (`...it12_constants.py:13`) and re-issued
          it as C131 (`:13-14`). His EDIT/RE-TAKE/RE-DECLARE procedure re-took
          only tests/jupiter. The withdrawn pointer is opened by a MARS
          instrument, which his edit turned RED. That is the +1 failure between
          his 31 and the 32 this office measures 5/5.

STRIKE C  J-24a exempts `:*` from J-20b because an append cannot break it.
          `region(path,'*')` returns the WHOLE FILE, so a `:*` anchor is a grep,
          and a grep cannot tell a want from a line that says the want was
          removed. This is the CORRECTIONS-INDEX property -- the repair record
          reads as the defect -- imported into the citation instrument.
"""

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tests.jupiter.test_v20_r15_it20_citation_freeze import (  # noqa: E402
    CENSUS,
    REISSUED,
    lands,
    region,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUITE = "tests/mars_v20"
CONTROL_SUITE = "tests/jupiter"


def _tracked(d: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", d], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    return [x for x in out if x.endswith(".py")]


def _present(d: str) -> list[str]:
    return sorted(p.name for p in (ROOT / d).glob("test_*.py"))


# ---------------------------------------------------------------- STRIKE A

def test_the_control_a_git_sha_DOES_bind_a_tracked_suite():
    """Control: the probe is not false for everything. tests/jupiter is tracked."""
    assert _tracked(CONTROL_SUITE), (
        "control gone: tests/jupiter is no longer tracked either, so STRIKE A's "
        "probe now returns the RED answer for every input -- re-derive it")


def test_a_head_sha_is_a_provenance_for_the_suite_it_counts():
    """RED: SATURN's `31 failed, 59 passed` at HEAD 207e7b9 over an untracked suite."""
    tracked, present = _tracked(SUITE), _present(SUITE)
    assert len(tracked) == len(present), (
        f"{len(tracked)} of {len(present)} {SUITE} files are known to git. A count "
        f"published `at HEAD 207e7b9` over this suite names a commit that does not "
        f"contain it: the same blindness SATURN proved for `git diff --stat` on an "
        f"untracked script, one section later, used as provenance instead of as a "
        f"defect. It is why his `39 failed` at it.21 cannot be sourced -- there is "
        f"no history to source it from.")


# ---------------------------------------------------------------- STRIKE B

def _censused_pointers() -> set[str]:
    return {f"{path}:{spec}" for path, spec, _ in CENSUS.values()}


def _it17_manifest() -> set[str]:
    from tests.jupiter import test_v20_r15_it17_citation_landing as IT17
    return set(IT17.MANIFEST)


def test_the_control_most_of_it17s_manifest_is_still_censused():
    """Control: the probe is not RED for everything -- most pointers still match."""
    kept = _it17_manifest() & _censused_pointers()
    assert len(kept) >= 20, (
        f"control gone: only {len(kept)} of it.17's manifest still matches the "
        f"census, so STRIKE B's probe is RED for every input -- re-derive it")


def test_a_withdrawal_does_not_strand_a_pointer_another_office_reads():
    """RED: C63's withdrawn `:13` is still the datum foreign instruments open."""
    stranded = sorted(_it17_manifest() - _censused_pointers())
    assert not stranded, (
        f"it.24 withdrew cids {sorted(REISSUED.values())} from CENSUS and re-issued "
        f"them widened, but it.17's MANIFEST -- which it.24 states was 'not edited' "
        f"-- still carries the OLD pointer, and MANIFEST is what foreign instruments "
        f"read: tests/mars_v20/test_it20_the_census_still_certifies_its_own_coverage.py "
        f"unions it directly and now reports it as a phantom. SUPERSEDED records the "
        f"widening inside the editing office's own assertion; it does not move the "
        f"datum. EDIT/RE-TAKE/RE-DECLARE re-took tests/jupiter only -- `231 passed, "
        f"3 failed` is the editing office grading its own blast radius. This is the "
        f"+1 between JUPITER's `31 failed` and the `32 failed, 58 passed` this office "
        f"measures 5/5 at the same HEAD. "
        f"stranded: {stranded}")


# ---------------------------------------------------------------- STRIKE C

STAR_CID = 55  # ceq/arm_pl.py:* -- 'ARM PL -- one causal softmax head'


def _obituary(text: str, want: str) -> str:
    """The want deleted from every line, and one line recording the deletion."""
    kept = [ln for ln in text.splitlines() if want not in ln]
    return "\n".join(kept + [f'# the want "{want}" was removed at it.25'])


def _deleted(text: str, want: str) -> str:
    """The want deleted and NOTHING said about it."""
    return "\n".join(ln for ln in text.splitlines() if want not in ln)


def test_the_control_a_star_anchor_breaks_on_a_silent_deletion():
    """Control: `:*` is not true for everything -- a silent deletion IS caught."""
    path, spec, want = CENSUS[STAR_CID]
    assert spec == "*", f"C{STAR_CID} is no longer a `:*` pointer -- re-derive"
    text = region(path, "*")
    assert lands(want, text), "premise gone: the want no longer lands unmutated"
    assert not lands(want, _deleted(text, want)), (
        "control gone: `:*` no longer detects a silent deletion, so STRIKE C's "
        "probe is false for everything -- re-derive it")


def test_a_star_pointer_can_tell_a_want_from_its_own_obituary():
    """RED: the repair record reads as the defect, inside the citation instrument."""
    path, _, want = CENSUS[STAR_CID]
    text = region(path, "*")
    assert not lands(want, _obituary(text, want)), (
        f"`{path}:*` still LANDS on a file where the want was deleted and only a "
        f"line recording its removal remains. J-24a exempts `:*` from J-20b "
        f"because an append cannot break it -- and an append is exactly what "
        f"repairs the deletion it cannot see. This is the CORRECTIONS INDEX "
        f"property: 33 rows quote the claims they overturn, so a grep finds every "
        f"corrected claim still present, and two offices read MARS STRIKE 3 as "
        f"live for three iterations after it was repaired at it.23. A file-scoped "
        f"anchor is a grep. The route is to score `:*` against a COUNT of "
        f"occurrences frozen at census, not against presence.")
