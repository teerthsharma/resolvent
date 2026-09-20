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

THE ROUTE, for whoever repairs it. Two options, PRICED AGAINST THE SCAN rather
than against the stale five-file list this once carried:
  (a) give the shared symbols a module of their own -- `tests/chase/_chase_env.py`
      holding `_has_cuda`, `HAS_CUDA`, `DEVICES`, `has_triton`, `HAS_TRITON`,
      `requires_cuda`, `requires_triton` and `run_isolated`, imported by
      `conftest.py` for fixture use and by the offenders directly. The name cannot
      be shadowed because only `tests/chase/` defines it, whereas `conftest` is
      claimed by 26 directories. Measured cost: 1 new module extracted from a
      291-line `conftest.py` whose fixtures are interleaved with these helpers,
      plus 11 import sites across 10 files (7 live, 3 under `attic/`), of which
      6 are function-local.
  (b) add `__init__.py` to the 24 directories that lack one, making the test tree
      a package so `conftest` is never a top-level module. 24 new empty files,
      zero source edits, but it changes rootdir semantics for every test.

(a) remains the smaller blast radius. NOT ATTEMPTED HERE, deliberately: extracting
interleaved helpers out of a conftest is a change whose only real verification is a
full-suite run on a quiet tree, and this guard was repaired while a training job and
two agents held the machine. Attempting it here would have meant either a slow
verification or an unverified edit to the collection machinery every chase test
depends on. The finding is bound and correctly sized; the repair is priced and
deferred, which is a different thing from being forgotten.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The importer and a shadowing file from a different test directory. Both are
# tracked; neither is a fixture of this test.
#
# SHADOWER was tests/w6/test_w6_attention.py. `git show --stat -M 228a048` shows
# that commit MOVED it to attic/tests/w6/test_w6_attention.py (a rename, not the
# delete an earlier report claimed from running --diff-filter=D without -M), and
# c71527a then deleted the attic copy outright (`git show --stat c71527a` shows
# `attic/tests/w6/test_w6_attention.py | 252 -------`). It exists at neither path
# now, so `shadower_path()`'s attic fallback below can never find it again --
# retargeting the constant, not extending the fallback, is the fix. Retargeted to
# tests/w2/test_w2_nonnormal.py: tests/w2/conftest.py is the same shape tests/w6's
# was, a bare pytest.fixture shim with none of tests/chase/conftest.py's symbols,
# and it is the cheapest live candidate measured -- `pytest tests/w2/test_w2_nonnormal.py
# --collect-only -q` read 0.18s against 20 other directories checked the same way,
# all between 0.97s and 2.92s -- cheap collection being the property this guard
# was built to keep (see THE ROUTE above).
IMPORTER = "tests/chase/test_kernel_contracts.py"
SHADOWER = "tests/w2/test_w2_nonnormal.py"

def bare_conftest_importers() -> list[str]:
    """Every tracked test file that reaches `conftest` as a plain module.

    SCANNED, NOT LISTED. This was a hardcoded list of five paths, and by the time
    it was re-read the tree had moved past it in both directions: three of the
    five (`test_multizoom_{cost,kernel,r5}.py`) had been retired to `attic/` at
    iteration 4, and FIVE live offenders were never in the list at all --
    `test_ceq_hub_package.py`, `test_hf_shipping.py`,
    `test_hub_package_hardening.py`, `test_schedule_rebuild.py`,
    `test_stochastic_P.py`. So the guard condemned five files while five more went
    unreported, which is MISTAKES.md V-14: a control that validates against a
    fixed roster instead of the actual reach.

    Function-local imports count. `from conftest import X` inside a test body
    resolves through `sys.path` at call time exactly as a module-level one does,
    so deferring it changes when the collision happens, not whether it can.
    """
    roots = [ROOT / "tests", ROOT / "attic" / "tests"]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for f in sorted(root.rglob("test_*.py")):
            text = f.read_text(encoding="utf-8", errors="replace")
            if re.search(r"^\s*(from conftest import|import conftest)", text, re.M):
                found.append(str(f.relative_to(ROOT)).replace("\\", "/"))
    return found


BARE_CONFTEST_IMPORTERS = bare_conftest_importers()


def test_the_scanner_still_recognizes_a_bare_conftest_import():
    """Canary for the REGEX, not the tree. Read this before trusting a zero.

    BARE_CONFTEST_IMPORTERS is empty right now. That is a real count: the seven
    files it used to find -- `test_ceq_hub_package.py`, `test_hf_shipping.py`,
    `test_hub_package_hardening.py`, `test_kernel_contracts.py`,
    `test_rollback_flex_attention.py`, `test_schedule_rebuild.py`,
    `test_stochastic_P.py`, all still bare-importing `conftest` at HEAD -- were
    rewritten on disk (uncommitted) to import `tests/chase/_chase_env.py`
    instead, a module name no other test directory defines, so it cannot be
    shadowed the way `conftest` was. Grepping the tracked tree for
    `^\\s*(from conftest import|import conftest)` outside `attic/` and this
    file's own docstring confirms zero hits. That is coverage correctly
    reporting a fixed tree, not coverage lost.

    But an empty parametrize list looks IDENTICAL whether the tree is clean or
    the regex itself broke -- pytest reports one `[NOTSET]` skip either way,
    so a typo'd pattern would sit there silently agreeing with a clean tree
    forever. This runs `bare_conftest_importers`'s own pattern against a
    string built to trip it, so a future zero keeps meaning "the tree has no
    offenders" instead of "the scanner stopped looking."
    """
    live_sample = "import torch\n    from conftest import run_isolated\n"
    assert re.search(r"^\s*(from conftest import|import conftest)", live_sample, re.M), (
        "the bare-import pattern no longer matches a known-bad line -- an "
        "empty BARE_CONFTEST_IMPORTERS would silently mean nothing. Fix the "
        "regex in bare_conftest_importers() before trusting a zero from it."
    )


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


def require_fixture(rel_path: str) -> None:
    """Fail as a MISSING FIXTURE, by name, before `collect()` can misreport it as a collision.

    THE FALSE RED THIS GUARDS AGAINST. `collect()` hands pytest a path that does not
    exist and pytest reports "file or directory not found", returncode 4. The two
    assertions that used to call `collect()` directly could not tell that apart from
    a genuine conftest binding collision (also a nonzero returncode) and asserted the
    collision message either way -- so `test_the_shadowing_file_collects_when_it_is_alone`
    and `test_collecting_the_pair_together_does_not_break_the_import` both read
    "`conftest` bound to the wrong directory" while the real stderr said
    `ERROR: file or directory not found: ...\\tests\\w6\\test_w6_attention.py`, rc=4.
    Whoever reads that failure next goes looking for an import-shadowing bug that
    is not there. Checking existence first, and failing here with the absent path
    named, keeps the two defects from being reported as each other.
    """
    if resolve(rel_path) is None:
        pytest.fail(
            f"MISSING FIXTURE, not a binding collision: {rel_path} resolves to no "
            "file, at its own path or under attic/. A subprocess collect against a "
            "path that does not exist returns nonzero for that reason alone -- "
            "restore the file, retarget the constant that names it, or retire the "
            "test that needs it before trusting a red from collect() here."
        )


def test_the_importer_collects_when_it_is_alone():
    """Premise bind. If this fails the file is broken outright and the pair test proves nothing."""
    require_fixture(IMPORTER)
    r = collect(IMPORTER)
    assert r.returncode == 0, f"{IMPORTER} does not collect even alone:\n{r.stdout[-2000:]}"


def shadower_path() -> str:
    """The shadower at its own path, or where iteration 4's attic move put it.

    `tests/w6/test_w6_attention.py` (the SHADOWER before this constant was
    retargeted) was one of the 33 rows retired at iteration 4, so immediately after
    the move this premise bind failed -- correctly, and that is what a premise bind
    is for: it reported that its own precondition had moved rather than passing on
    a file that was no longer there. c71527a then deleted the attic copy outright,
    so for that specific path the fallback below now finds nothing either -- SHADOWER
    was retargeted to a live file instead of teaching this function a third place to
    look for one that is gone from the tree.

    The fallback itself stays: the defect this test supports is unchanged by a future
    retirement of WHATEVER file SHADOWER names. Any test file, in a directory whose
    conftest lacks `run_isolated`, shadows `tests/chase`'s when both are collected in
    one command, retired-to-attic or not, because pytest inserts the collected file's
    directory onto `sys.path` either way.
    """
    for candidate in (SHADOWER, f"attic/{SHADOWER}"):
        if (ROOT / candidate).exists():
            return candidate
    return SHADOWER


def test_the_shadowing_file_collects_when_it_is_alone():
    """Premise bind, other half. Neither file is individually broken."""
    path = shadower_path()
    require_fixture(path)
    r = collect(path)
    assert r.returncode == 0, f"{path} does not collect even alone:\n{r.stdout[-2000:]}"


def test_collecting_the_pair_together_does_not_break_the_import():
    """THE DEFECT. Two files that each collect alone must collect together.

    Red while `conftest` resolves through sys.path; green once the shared symbols
    live in a module whose name is not claimed by 24 other directories. Both
    fixtures are checked to exist before `collect()` runs, so a red here is that
    collision and not the missing-file rc=4 `collect()` cannot tell apart on its
    own -- see `require_fixture`.
    """
    path = shadower_path()
    require_fixture(IMPORTER)
    require_fixture(path)
    r = collect(IMPORTER, path)
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
    # `.lstrip()` before matching. Without it this assertion saw only UNINDENTED
    # imports while `bare_conftest_importers()` scans with `^\s*`, so five of the
    # ten parametrised files were collected and could never fail -- the selector
    # and the check disagreeing about what counts, which makes half the cases
    # vacuous. A function-local `from conftest import X` resolves through
    # `sys.path` at call time exactly as a module-level one does.
    offenders = [
        line.strip()
        for line in src.splitlines()
        if line.lstrip().startswith(("from conftest import", "import conftest"))
    ]
    assert not offenders, (
        f"{path} reaches conftest as a top-level module: {offenders}. "
        "With only 2 of 26 test dirs carrying __init__.py, that name resolves by "
        "collection order."
    )
