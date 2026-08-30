"""`scale/chase_struck_coverage.py` must actually walk files, and be shown to.

THE DEFECT THIS FILE PINS SHUT. The scanner excluded any path one of whose
components was `.git`, `__pycache__`, `.pytest_cache`, `.claude` or
`.benchmarks`, and it tested those against `p.parts` -- the ABSOLUTE path. Every
agent worktree in this project is checked out under
`<repo>/.claude/worktrees/<name>/`, so `.claude` was a component of the absolute
path of every file in the tree and the filter dropped all of them. 366 candidate
files became 0. The scan printed `SCANNING 0 PATHS THE SHIPPED CHECK DOES NOT
COVER` and `uncovered .md: 0, uncovered .py: 0`, then exited 0 -- which reads as
"no struck constant is asserted in any uncovered path" and is in fact "no path
was looked at". A coverage tool whose answer depends on where the repository
happens to sit on disk is not a coverage tool, and the failure mode was silent
in the direction that passes.

Venus found it because adding three constants to `STRUCK` (9 -> 12) produced no
collateral at all, and treated a convenient result as suspicious rather than as
good news.

WHAT IS ASSERTED HERE. That the target list is non-empty on the real tree; that
the exclusion is relative to the scan root rather than absolute, tested on a
root deliberately buried under a directory named `.claude`; that a struck
constant planted in a file the scanner should reach IS found; that the same text
carrying a strike marker is NOT found, so the passing half is non-degenerate;
and that `main` refuses to exit 0 on an empty scan.
"""
import pathlib

import scale.chase_struck_coverage as C

PLANTED = 1.471448          # M5 tail norm at s=128, struck as fabricated

def test_the_scanner_walks_a_non_empty_target_list():
    """The regression. This is the single assertion that would have caught it."""
    targets = C.collect_targets()
    assert len(targets) > 100, f"scanner reaches only {len(targets)} paths"
    rels = [rel for _p, rel in targets]
    assert any(r.endswith(".md") for r in rels)
    assert any(r.endswith(".py") for r in rels)
    # and it still honours the two exclusions that are real
    assert not any(r in C.COVERED for r in rels)
    assert not any(pathlib.PurePosixPath(r).name in C.HISTORY for r in rels)

def test_the_exclusion_is_relative_to_the_root_not_absolute(tmp_path):
    """THE ROOT CAUSE, pinned. A scan root buried under a directory literally
    named `.claude` must still see its own contents; only `.claude` INSIDE the
    root is a skip."""
    root = tmp_path / ".claude" / "worktrees" / "agent-xyz"
    (root / "scale").mkdir(parents=True)
    (root / "README_local.md").write_text("plain text\n", encoding="utf-8")
    (root / "scale" / "thing.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".claude").mkdir()
    (root / ".claude" / "internal.md").write_text("skip me\n", encoding="utf-8")

    rels = [rel for _p, rel in C.collect_targets(root)]
    assert "README_local.md" in rels
    assert "scale/thing.py" in rels
    assert ".claude/internal.md" not in rels, "inner .claude must still be skipped"

def test_a_planted_struck_constant_in_a_reachable_file_is_found(tmp_path):
    """THE MUST-FIRE. Planted in a file on disk that the scanner must reach by
    its own target selection -- not handed to `scan_text` directly, which would
    prove only that the matcher works and nothing about coverage."""
    root = tmp_path / ".claude" / "worktrees" / "agent-xyz"
    root.mkdir(parents=True)
    (root / "PLANTED_READING.md").write_text(
        "The tail norm at s=128 is {} and the fit holds.\n".format(PLANTED),
        encoding="utf-8")

    targets = C.collect_targets(root)
    assert [rel for _p, rel in targets] == ["PLANTED_READING.md"]
    hits = []
    for path, rel in targets:
        hits += C.scan_text(path.read_text(encoding="utf-8"), rel)
    assert len(hits) == 1, hits
    rel, _line, value, _text = hits[0]
    assert rel == "PLANTED_READING.md"
    assert value == PLANTED

def test_the_same_constant_with_a_strike_marker_is_not_found(tmp_path):
    """The PASS half, non-degenerate. If a marker did not silence the hit, the
    must-fire above would be firing on the presence of digits rather than on the
    absence of a strike."""
    root = tmp_path / ".claude" / "worktrees" / "agent-xyz"
    root.mkdir(parents=True)
    (root / "PLANTED_READING.md").write_text(
        "The {} printed here was struck as fabricated.\n".format(PLANTED),
        encoding="utf-8")
    hits = []
    for path, rel in C.collect_targets(root):
        hits += C.scan_text(path.read_text(encoding="utf-8"), rel)
    assert hits == []

def test_a_reachable_file_with_no_struck_constant_is_silent(tmp_path):
    """The third arm. Without it, a scanner that flagged every file would pass
    both tests above."""
    root = tmp_path / ".claude" / "worktrees" / "agent-xyz"
    root.mkdir(parents=True)
    (root / "PLAIN.md").write_text("Nothing numeric of interest here.\n",
                                   encoding="utf-8")
    hits = []
    for path, rel in C.collect_targets(root):
        hits += C.scan_text(path.read_text(encoding="utf-8"), rel)
    assert hits == []

def test_main_refuses_to_pass_on_an_empty_scan(monkeypatch, capsys):
    """The guard that makes the failure loud from the other direction too: even
    if some future filter empties the target list again, the tool must not
    report success."""
    monkeypatch.setattr(C, "collect_targets", lambda *a, **kw: [])
    assert C.main() == 1
    assert "SCANNED NOTHING" in capsys.readouterr().out

def test_the_shipped_control_still_discriminates():
    """`chase_struck_coverage.control()` is the file's own must-fire. It passed
    throughout the period the scanner walked zero paths, which is exactly why a
    matcher control is not a coverage control."""
    assert C.control() is True

def test_the_absolute_parts_filter_is_degenerate_on_this_checkout():
    """The root cause as a property of the tree and the filter, stated without
    reference to `collect_targets` so that a later refactor cannot make it
    vacuous. Both filters are written out inline here; the absolute one drops
    everything whenever the repository lives under a skipped directory name,
    which every agent worktree in this project does."""
    root = C.ROOT
    candidates = list(root.rglob("*.md")) + list(root.rglob("*.py"))
    assert len(candidates) > 100

    def survives(path, parts):
        return not any(part in C.SKIP_DIRS for part in parts)

    absolute = sum(1 for p in candidates if survives(p, p.parts))
    relative = sum(1 for p in candidates
                   if survives(p, p.relative_to(root).parts))
    assert relative > 100, relative
    if any(part in C.SKIP_DIRS for part in root.parts):
        # this checkout sits inside a skipped directory name, so the old filter
        # is provably degenerate here: the bug reproducing
        assert absolute == 0, absolute
    else:
        assert absolute == relative
