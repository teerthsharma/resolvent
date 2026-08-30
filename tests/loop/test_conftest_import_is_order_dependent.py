"""Collection succeeds or fails depending on which files share the command line.

THE DEFECT. Five files under `tests/chase/` import symbols with a bare
`from conftest import ...`. `conftest` is not pytest machinery there -- it is an
ordinary module name resolved through `sys.path`. Only 2 of the 26 test
directories carry an `__init__.py`, so pytest's default prepend import mode
inserts each collected file's directory into `sys.path`, and `conftest` binds to
whichever test directory was inserted first.

Collect `tests/chase/test_kernel_contracts.py` on its own, or collect the whole
`tests/chase/` directory, and `conftest` is `tests/chase/conftest.py`, which
defines `run_isolated`. Collect it in the same command as a file from another
test directory and `conftest` can be that other directory's, which does not:

    ImportError: cannot import name 'run_isolated' from 'conftest'
                 (tests/w6/conftest.py)

WHY IT HAS SURVIVED. A full-tree run walks `tests/chase/` before the shadowing
directories, so `pytest` over the whole repo collects 2071 tests with zero
errors and the defect is invisible. It only surfaces on a SUBSET invocation --
which is exactly what a census sweep, a bisect, or a targeted re-run does. Round
10 iteration 1 hit it twice: a sweep over 22 census-condemned files reported
"5 errors during collection", and the same five files collected clean when run
alone. It was recorded as an unexplained name collision. It is not unexplained.

WHAT THIS TEST BINDS. Not "the five files import from conftest" -- that is a
grep, and a grep passes whether or not the import resolves. This runs pytest's
own collector on the failing pair and asserts it collects, so the test is red
while the defect is present and green only when `conftest` stops being a shared
global name. It is deliberately the CHEAPEST reproduction: two files, collect
only, no test bodies executed.

THE ROUTE, for whoever repairs it. Two options, both small:
  (a) give the shared symbols a module of their own -- `tests/chase/_chase_env.py`
      holding `DEVICES`, `HAS_CUDA`, `requires_triton`, `run_isolated`, imported
      by `conftest.py` for fixture use and by the five files directly. One new
      file, six edited import lines, and `conftest` stops being importable by
      name at all.
  (b) add `__init__.py` to the 24 directories that lack one, making the test tree
      a package so `conftest` is never a top-level module. 24 new empty files,
      zero source edits, but it changes rootdir semantics for every test.
(a) is the smaller blast radius and the one this file's failure points at.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The importer and a shadowing file from a different test directory. Both are
# tracked; neither is a fixture of this test. Chosen because tests/w6/conftest.py
# is a bare pytest.fixture shim with none of tests/chase/conftest.py's symbols.
IMPORTER = "tests/chase/test_kernel_contracts.py"
SHADOWER = "tests/w6/test_w6_attention.py"

# Every file that reaches conftest as a plain module rather than through pytest.
BARE_CONFTEST_IMPORTERS = [
    "tests/chase/test_kernel_contracts.py",
    "tests/chase/test_multizoom_cost.py",
    "tests/chase/test_multizoom_kernel.py",
    "tests/chase/test_multizoom_r5.py",
    "tests/chase/test_rollback_flex_attention.py",
]


def collect(*rel_paths: str) -> subprocess.CompletedProcess[str]:
    """Run pytest's collector only. No test body executes, so this is cheap.

    Paths go through `resolve()`. Measured at it.4, immediately after the move:
    `SHADOWER` is `tests/w6/test_w6_attention.py`, an ATTIC row, and handing that
    literal to the subprocess returned "file or directory not found" -- so both
    premise binds failed with returncode 4 and reported the shadowing defect as
    unreproducible. `resolve()` already existed here and already looked under
    `attic/`; it was applied to the files this guard READS and not to the files it
    COLLECTS. Same fix, second call site. norecursedirs skips `attic/` on directory
    recursion only, so an explicitly named path still collects.
    """
    args = [str(resolve(r) or (ROOT / r)) for r in rel_paths]
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args, "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_importer_collects_when_it_is_alone():
    """Premise bind. If this fails the file is broken outright and the pair test proves nothing."""
    r = collect(IMPORTER)
    assert r.returncode == 0, f"{IMPORTER} does not collect even alone:\n{r.stdout[-2000:]}"


def shadower_path() -> str:
    """The shadower at its own path, or where iteration 4's attic move put it.

    `tests/w6/test_w6_attention.py` was one of the 33 rows retired at iteration 4,
    so immediately after the move this premise bind failed -- correctly, and that is
    what a premise bind is for: it reported that its own precondition had moved
    rather than passing on a file that was no longer there.

    The defect it supports is unchanged by the retirement. Any `tests/w6` file whose
    conftest lacks `run_isolated` shadows `tests/chase`'s when both are collected in
    one command, retired or not, because pytest inserts the collected file's
    directory onto `sys.path` either way.
    """
    for candidate in (SHADOWER, f"attic/{SHADOWER}"):
        if (ROOT / candidate).exists():
            return candidate
    return SHADOWER


def test_the_shadowing_file_collects_when_it_is_alone():
    """Premise bind, other half. Neither file is individually broken."""
    r = collect(shadower_path())
    assert r.returncode == 0, f"{shadower_path()} does not collect even alone:\n{r.stdout[-2000:]}"


def test_collecting_the_pair_together_does_not_break_the_import():
    """THE DEFECT. Two files that each collect alone must collect together.

    Red while `conftest` resolves through sys.path; green once the shared symbols
    live in a module whose name is not claimed by 24 other directories.
    """
    r = collect(IMPORTER, shadower_path())
    assert r.returncode == 0, (
        "collecting two individually-collectable files together fails.\n"
        "`conftest` bound to the wrong directory:\n" + r.stdout[-2000:]
    )


def resolve(rel: str) -> pathlib.Path | None:
    """The file at its own path, or where iteration 4's attic move puts it.

    Three of the five files below are ATTIC on the pinned sheet
    (`test_multizoom_cost.py`, `test_multizoom_kernel.py`, `test_multizoom_r5.py`).
    Resolving only `ROOT / rel` makes these cases raise FileNotFoundError the moment
    the move runs -- an error, not a finding -- while the defect they report is
    unchanged and merely relocated. A retired file that still does
    `from conftest import` still breaks a collection that includes it.
    """
    for candidate in (ROOT / rel, ROOT / "attic" / rel):
        if candidate.exists():
            return candidate
    return None


@pytest.mark.parametrize("path", BARE_CONFTEST_IMPORTERS)
def test_no_test_file_imports_conftest_as_a_bare_module(path: str):
    """The mechanism, per file, so a partial repair is visible as a partial pass."""
    found = resolve(path)
    if found is None:
        pytest.fail(
            f"{path} resolves to no file, at its own path or under attic/. "
            "A named offender that exists nowhere is not a repair; say where it went."
        )
    src = found.read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in src.splitlines()
        if line.startswith(("from conftest import", "import conftest"))
    ]
    assert not offenders, (
        f"{path} reaches conftest as a top-level module: {offenders}. "
        "With only 2 of 26 test dirs carrying __init__.py, that name resolves by "
        "collection order."
    )
