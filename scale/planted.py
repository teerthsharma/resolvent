"""Plant a defect, measure that something catches it, and get the tree back.

WHY THIS EXISTS, with its cost. Round 10 iteration 3: HOUSE planted an off-by-one
in `ceq/hankel.py` to measure which test files catch a broken Hankel instrument,
inside a single shell command of the shape

    sed -i '<mutate>' ceq/hankel.py && pytest a b c ; git checkout -- ceq/hankel.py

The command hit its 2-minute timeout during the third pytest run and was killed.
The revert never executed. The plant sat in the working tree for roughly two
minutes while three other agents were measuring against it, and `ceq.hankel` is
imported by `scale/eprocess.py` and `scale/negation_scope.py`, so the blast radius
was not the three test files -- it was anything reaching either module. Nobody's
reading was demonstrably corrupted, but nobody could show it was not, which is the
same problem.

The defect is not carelessness, it is SHAPE: a revert written as the last clause of
the command it protects dies with that command. MARS ran the same kind of mutation
in the same iteration and left the tree clean, because he ran mutate / measure /
revert as three separate commands and the Health Inspector could verify the result.

WHAT THIS GUARANTEES, AND WHAT IT CANNOT. `plant()` reverts in a `finally`, so it
survives an exception and a normal early return. It installs SIGTERM and SIGINT
handlers, so it survives a timeout kill and a Ctrl-C. It CANNOT survive SIGKILL or
a power loss -- nothing in-process can. For that case it writes a sentinel before
mutating and removes it after reverting, so `assert_no_stale_plant()` at the top of
any measurement finds the wreck instead of measuring against it. A death leaves
evidence, which is ADR-001's rule applied to source rather than to journals.

The revert is `git checkout --`, so this only works on tracked files with no
uncommitted changes. That is checked before mutating, not after: planting on top of
someone's unstaged work and then reverting would destroy it.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SENTINEL = ROOT / "results" / ".planted_mutation.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def assert_no_stale_plant() -> None:
    """Refuse to measure while a previous plant is unaccounted for.

    Call this at the top of anything whose reading would be wrong against a
    mutated tree. It is cheap and it is the only part of this module that
    survives SIGKILL.
    """
    if SENTINEL.exists():
        info = json.loads(SENTINEL.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"a planted mutation in {info['path']} was never reverted "
            f"(planted at {info['when']}, reason: {info['reason']}). "
            f"Run: git checkout -- {info['path']} && rm {SENTINEL}"
        )


@contextlib.contextmanager
def plant(rel_path: str, old: str, new: str, *, reason: str):
    """Replace `old` with `new` in a tracked file, then always put it back.

    Raises before mutating if the file is not tracked, is already modified, or
    does not contain `old` exactly once -- a plant that lands in the wrong place,
    or twice, measures something nobody described.
    """
    assert_no_stale_plant()
    path = ROOT / rel_path

    if _git("ls-files", "--", rel_path).strip() != rel_path.replace("\\", "/"):
        raise RuntimeError(f"{rel_path} is not tracked; git checkout cannot restore it")
    if _git("status", "--short", "--", rel_path).strip():
        raise RuntimeError(
            f"{rel_path} has uncommitted changes; planting would destroy them on revert"
        )

    src = path.read_text(encoding="utf-8")
    hits = src.count(old)
    if hits != 1:
        raise RuntimeError(f"the plant site appears {hits} times in {rel_path}, want exactly 1")

    def restore(*_a):
        _git("checkout", "--", rel_path)
        SENTINEL.unlink(missing_ok=True)

    def on_signal(signum, _frame):
        restore()
        signal.signal(signum, signal.SIG_DFL)
        sys.exit(128 + signum)

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, AttributeError, OSError):
            previous[sig] = signal.signal(sig, on_signal)

    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps(
        {"path": rel_path, "reason": reason, "old": old, "new": new,
         "when": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
    try:
        path.write_text(src.replace(old, new), encoding="utf-8")
        yield path
    finally:
        restore()
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError, AttributeError, OSError):
                signal.signal(sig, handler)


def demo() -> None:
    """Self-check: the plant lands, reverts, and reverts even when the body raises.

    Plants into `scale/_plant_probe.txt`, a tracked one-line fixture nothing reads.
    Two earlier drafts planted into THIS file and both were refused by this module's
    own guards -- once for uncommitted changes, once because the plant site appeared
    5 times, since the demo body quotes it. Planting into live code while other work
    is running is the wrong target regardless of whether the guards allow it.
    """
    rel = "scale/_plant_probe.txt"
    site, replacement = "nothing reads", "NOTHING READS"
    before = (ROOT / rel).read_text(encoding="utf-8")

    with plant(rel, site, replacement, reason="self-check"):
        assert replacement in (ROOT / rel).read_text(encoding="utf-8"), "the plant did not land"
        assert SENTINEL.exists(), "no sentinel while planted"
    assert (ROOT / rel).read_text(encoding="utf-8") == before, "revert did not restore bytes"
    assert not SENTINEL.exists(), "sentinel outlived the plant"

    with contextlib.suppress(ZeroDivisionError):
        with plant(rel, site, replacement, reason="self-check"):
            raise ZeroDivisionError
    assert (ROOT / rel).read_text(encoding="utf-8") == before, "revert did not survive an exception"
    assert not SENTINEL.exists(), "sentinel outlived a raising plant"

    SENTINEL.write_text('{"path": "x", "reason": "stale", "when": "t"}', encoding="utf-8")
    try:
        assert_no_stale_plant()
        raise AssertionError("assert_no_stale_plant did not fire on a stale sentinel")
    except RuntimeError:
        pass
    finally:
        SENTINEL.unlink(missing_ok=True)

    print("demo OK: lands, reverts on exit, reverts on exception, "
          "stale sentinel refuses, tree byte-identical")


if __name__ == "__main__":
    demo()
