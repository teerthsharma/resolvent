"""Importing a module must not write a file or read the command line.

THE DEMONSTRATED HAZARD, not a style opinion. `scale/replay_census.py` binds
`OUT = pathlib.Path(sys.argv[1])` at line 16 and opens it `"w"` at line 23, both
at module scope with no `__main__` guard. So `import scale.replay_census`
truncates whatever file happens to be at `sys.argv[1]`.

Measured, round 10 iteration 2, against a sacrificial file in a scratch dir:

    before: 37 bytes   ("PRECIOUS CONTENT\\nline two\\nline three\\n")
    after: 251 bytes   ("37 journalled+live units, threads=2 / [ 0] DRIFT ...")

The import destroyed the file's contents and replaced them with census output.
Under pytest, `sys.argv[1]` is normally a test path, so an import of this module
during a test run overwrites a test file with a drift report. The round-1 census
recorded this as "nothing imports it, so it is currently inert". Inert is a
property of today's import graph, not of the module, and the import graph is
edited every iteration.

`scale/p1prime.py` is the second instance and it is one day old. It has no
`__main__` guard at all: a `git ls-files` subprocess at line 17, a scan over 191
files from line 29, and writes to `results/p1prime_rows.json` and
`results/p1prime_front_door.txt` at lines 63 and 65. It was written to rescue
provenance out of a temp directory, and it reintroduced the defect class it was
rescuing from -- the same shape MISTAKES.md records for V-7, where the fix for a
defect carried the defect one level up.

WHY THIS BITES THIS ROUND SPECIFICALLY. Iteration 2's spot-check protocol scores
a `scale/` module as a passing KEEP if it is "importable without side effects".
A nurse executing that check against `scale/p1prime.py` rewrites two committed
files in `results/` as a side effect of measuring whether it has side effects.

SCOPE, AND WHY THE CHECK IS AST AND NOT A GREP. The check walks module-scope
statements only. It deliberately does NOT descend into `FunctionDef`,
`AsyncFunctionDef`, or `ClassDef` bodies -- code inside a function does not run
at import -- and it skips `if __name__ == "__main__":` blocks, which is the
correct place for exactly this code. A first draft of this scan walked function
bodies too and reported 6 offenders instead of 2; the three extra were ordinary
functions that open files when called. A scan that cannot distinguish "runs at
import" from "contains an open() somewhere" is not measuring the hazard.

THE ROUTE for both files: wrap the executable body in `def main():` plus
`if __name__ == "__main__": sys.exit(main())`. `scale/spotcheck_draw.py` in this
same repo is the shape to copy. Neither file loses a capability; both stop firing
on import.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Modes that destroy or extend a file. "r" and "rb" are absent on purpose:
# reading at import is slow and rude but it does not lose data.
DESTRUCTIVE_MODES = ("w", "a", "x")

# Not executed when the module is imported.
NOT_AT_IMPORT = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)


def _is_main_guard(node: ast.stmt) -> bool:
    """True for `if __name__ == "__main__":` -- the correct home for script bodies."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


def _open_mode(call: ast.Call) -> str | None:
    """The mode string of an `open(...)` / `.open(...)` call, positional or keyword."""
    for arg in call.args[1:2]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return None


def import_time_side_effects(path: pathlib.Path) -> list[str]:
    """Every module-scope statement that writes a file or reads argv, with line numbers."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, NOT_AT_IMPORT) or _is_main_guard(node):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in ("open", "write_text", "write_bytes"):
                    mode = _open_mode(sub) if name == "open" else "w"
                    if mode and any(c in mode for c in DESTRUCTIVE_MODES):
                        found.append(f"line {sub.lineno}: {name}(mode={mode!r})")
            if (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "argv"
            ):
                found.append(f"line {sub.lineno}: sys.argv[...] read at import")
    return sorted(set(found))


def tracked_modules() -> list[str]:
    """Every module in the TREE, not every module in the INDEX.

    The first draft used `git ls-files` alone. `scale/p1prime.py` -- which carries
    the exact module-scope `open(mode='w')` this file exists to catch, at line 63 --
    left the index between two runs and the guard silently stopped reporting it:
    "2 failed" became "1 failed" with the defect untouched on disk. The Health
    Inspector struck the count.

    Keying on tracked-ness is a surface proxy for "is part of the repo", and round
    10 has now recorded ten separate rules that failed by keying on a surface proxy
    instead of the thing itself. An untracked file still imports, still runs, and
    still truncates whatever `sys.argv[1]` names -- and NEPTUNE's own iteration-2
    finding was that untracked files here carry live claims. The union is the
    honest population: the index catches files deleted from disk but still staged,
    the walk catches files on disk but not staged.
    """
    indexed = subprocess.run(
        ["git", "ls-files", "--", "scale/*.py", "ceq/*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    walked = [
        str(p.relative_to(ROOT)).replace("\\", "/")
        for d in ("scale", "ceq")
        for p in (ROOT / d).rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    return sorted(set(indexed) | set(walked))


def test_the_scan_can_find_a_planted_offender():
    """Must-fire. A scan that cannot find what it searches for proves nothing by finding none.

    This is the guard MISTAKES.md V-7 exists for: a journal scan once reported a
    false absence because it could not descend to where the values lived, and the
    absence was used to strike a colleague's evidence.
    """
    planted = ast.parse("import sys\nOUT = sys.argv[1]\nopen(OUT, 'w').write('x')\n")
    tmp = ROOT / "results" / "__scan_witness_probe.py"
    try:
        tmp.write_text("import sys\nOUT = sys.argv[1]\nf = open(OUT, 'w')\n", encoding="utf-8")
        hits = import_time_side_effects(tmp)
        assert len(hits) == 2, f"the scan missed a planted offender: {hits}"
    finally:
        tmp.unlink(missing_ok=True)
    assert planted is not None


def test_the_scan_does_not_fire_on_a_guarded_script():
    """Must-not-fire. `spotcheck_draw.py` writes only under a __main__ guard and inside functions."""
    clean = ROOT / "scale" / "spotcheck_draw.py"
    if not clean.exists():
        pytest.skip("scale/spotcheck_draw.py absent; the negative control needs it")
    assert import_time_side_effects(clean) == [], (
        "the scan fires on a correctly-guarded script, so a hit means nothing"
    )


@pytest.mark.parametrize("rel", tracked_modules())
def test_module_has_no_import_time_write_or_argv_read(rel: str):
    """THE DEFECT, one case per module so a partial repair reads as a partial pass."""
    hits = import_time_side_effects(ROOT / rel)
    assert not hits, (
        f"{rel} acts on the filesystem or the command line at IMPORT time: {hits}. "
        "Move the body into `def main():` under `if __name__ == \"__main__\":` "
        "(see scale/spotcheck_draw.py). Measured: importing scale/replay_census.py "
        "truncated a 37-byte file and wrote 251 bytes of census output into it."
    )


def test_no_source_mutation_was_left_planted_in_the_tree():
    """The plant harness's own guard, actually invoked.

    `scale/planted.py::assert_no_stale_plant` exists because round 10 iteration 3
    left a deliberate off-by-one in `ceq/hankel.py` for ~2 minutes: the mutation and
    its revert shared one shell command, the command hit its timeout, and the revert
    died with it while three agents were measuring against the tree.

    The harness reverts in a `finally` and traps SIGTERM/SIGINT, but nothing in
    process survives SIGKILL, so it writes a sentinel before mutating and removes it
    after reverting. That sentinel is only worth having if something LOOKS at it --
    an unread guard is the vacuous-control shape MISTAKES.md catalogues, and until
    this test existed `assert_no_stale_plant` had zero callers anywhere in the repo.

    This is the call site. Any pytest run over tests/loop now refuses to be trusted
    while a mutation is unaccounted for.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from scale.planted import assert_no_stale_plant
    assert_no_stale_plant()
