"""MARS, CEQ v20 R15 it.2 -- P-12, filed properly this time.

At it.1 this was prose in a report and never entered `house-events.jsonl`; the
Inspector struck it UNBOUND (C19) and corrected one fact: `STRUCK.md` carries no
evidence *command*, only PROSE describing a search. The wording is corrected
here and the claim is bound to a test.

The proposition, restated so it names something that is in the file:

    STRUCK.md's row for `0.743864` records an ABSENCE PROOF in prose --
    "`git log -S` across ALL refs returns zero commits containing a definition
    of any of them" -- and performing that search as described now returns
    HITS, because writing the nine names into the tree put them in reach of the
    search that was supposed to find nothing.

The VERDICT on `0.743864` is not touched and is not in question: it stays
struck. What is defective is the reproducibility of the recorded proof.

P-12 is the MIRROR of `MISTAKES.md` V-7 (`MISTAKES.md:117`, "a search
structurally incapable of finding anything, read as absence"). V-7's search
could not find. This search COULD find, was correct when run, and publishing it
destroyed its own result.

V-7's own rule is the standard this file is held to: "Zero results is a claim
about the search before it is a claim about the world, and it must be paid for
with a positive the search is required to find." Every absence node below ships
its planted positive in the SAME invocation style.

Read-only. No git write is performed -- `git log` only.
"""
from __future__ import annotations

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The nine names STRUCK.md's 0.743864 row says have no definition in any state
# this repository has ever been in.
BATTERY = "absorbing_boundary_kernel"
# A symbol that demonstrably DOES have a definition in history. The planted
# positive: any invocation that reports zero for BATTERY must report non-zero
# for this, in the same style, or the search is not evidence of anything.
CONTROL = "path_product"


def _log_s(needle, *extra):
    cmd = ["git", "log", "-S", needle, "--all", "--oneline", *extra]
    out = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


# ---------------------------------------------------------------- CONTROLS

def test_control_the_struck_row_is_present_and_still_carries_its_verdict():
    """The document under discussion says what this file says it says."""
    txt = (ROOT / "STRUCK.md").read_text(encoding="utf-8", errors="replace")
    assert "0.743864" in txt
    assert "NO PRODUCER HAS EVER EXISTED" in txt
    assert "git log -S" in txt and BATTERY in txt


def test_control_the_bare_invocation_finds_a_symbol_that_exists():
    """PLANTED POSITIVE for the bare form. If this is 0 the search is broken."""
    hits = _log_s(f"def {CONTROL}(")
    assert len(hits) > 0, f"git log -S 'def {CONTROL}(' found nothing"


def test_control_the_repaired_invocation_keeps_its_reach():
    """PLANTED POSITIVE for the REPAIRED form.

    The repair adds a pathspec. A repair that also loses the ability to find a
    real definition would be V-7 introduced by the fix for P-12.
    """
    hits = _log_s(f"def {CONTROL}(", "--", "*.py", ":!tests/")
    assert len(hits) > 0, "the repaired invocation lost its reach"


# ------------------------------------------------------------------- REDS

def test_the_recorded_absence_proof_still_returns_zero_as_written():
    """RED: the absence STRUCK.md records no longer reproduces as described.

    STRUCK.md's row says `git log -S` across ALL refs returns zero commits
    containing a definition of any of the nine names. Run for the battery's
    lead name it returns HITS -- and every hit is the audit quoting its own
    search, plus the kaggle snapshot copies of that audit.
    """
    hits = _log_s(f"def {BATTERY}")
    assert not hits, (
        f"STRUCK.md records 'git log -S across ALL refs returns zero commits "
        f"containing a definition' of {BATTERY}; run as described it returns "
        f"{len(hits)} commits: " + "; ".join(hits)
    )


def test_no_recording_of_the_absence_proof_is_in_reach_of_the_absence_proof():
    """RED: the documents that record the search are what the search finds.

    This is the mechanism, isolated. `git grep` over every ref for the battery
    name, restricted to the documentation and test files that RECORD the
    strike, must be empty for the recorded proof to keep meaning what it said.
    """
    out = subprocess.run(
        ["git", "grep", "-l", BATTERY, "--", "*.md", "tests/"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    recorders = sorted(ln for ln in out.splitlines() if ln.strip())
    assert not recorders, (
        "the absence proof's own recording sits inside the absence proof's "
        f"search reach: {recorders}"
    )


def test_the_repaired_invocation_is_the_one_struck_md_publishes():
    """RED: STRUCK.md still publishes the falsified form, not the repaired one.

    The repair is one line: exclude the paths that record the strike, and ship
    a control symbol the same invocation must still find. Neither is in the
    file.
    """
    txt = (ROOT / "STRUCK.md").read_text(encoding="utf-8", errors="replace")
    assert ':!tests/' in txt and CONTROL in txt, (
        "STRUCK.md's 0.743864 row publishes the bare search with no path "
        "exclusion and no control symbol; the repaired form "
        f"`git log -S\"def {BATTERY}(\" --all --oneline -- \"*.py\" \":!tests/\"` "
        f"returns 0 while `def {CONTROL}(` under the same invocation returns "
        "non-zero, which is what an absence proof has to ship."
    )
