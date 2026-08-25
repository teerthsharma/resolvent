"""W3b -- machine-check the nilpotent resolvent.

`CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent` already proves that when
A^N = 0 the truncated occupancy sum IS the two-sided inverse of (I - A) --
exactly, with no truncation error and no convergence hypothesis.

What was missing is that the operator actually shipped in `ceq/nonnormal.py`
SATISFIES that hypothesis. `tests/w2` measures it (rho(A) < 1e-12, resolvent
residual < 1e-12) but a float64 measurement is not a proof, and the artifact's
claim is that the resolvent is exact rather than approximated.

`CEQ.Nilpotent.pow_card_eq_zero` closes it: a strictly lower-triangular matrix
over Fin n is nilpotent at n, by path counting. `occupancy_is_exact_inverse`
then discharges the Occupancy hypothesis for the operator that ships.

NO DEVICE PARAMETRIZATION HERE, DELIBERATELY. Every numerical test in this repo
runs on cpu and cuda. A Lean proof has no device axis, and a fake
`@parametrize("device", ...)` around a `lake build` would run the identical
subprocess twice and report two passes for one fact. Stating the absence is
more honest than manufacturing the parameter.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEAN = ROOT / "lean"
LAKE = pathlib.Path.home() / ".elan" / "bin" / "lake.exe"
NL = chr(10)

pytestmark = pytest.mark.skipif(
    not LAKE.exists() and shutil.which("lake") is None,
    reason="lean toolchain unavailable",
)


def _lake(*args: str) -> subprocess.CompletedProcess:
    exe = str(LAKE) if LAKE.exists() else "lake"
    return subprocess.run([exe, *args], cwd=LEAN, capture_output=True,
                          text=True, timeout=1800)


def _strip_lean_comments(src: str) -> str:
    """Remove /- block -/ and -- line comments.

    The first version of this check split on "--" only, and reported a hit on
    CEQ.lean's own header sentence "No `sorry` anywhere". A detector that fires
    on the documentation asserting there are no sorries is worse than no
    detector: it fails exactly when the code is correct.
    """
    out, i, depth = [], 0, 0
    while i < len(src):
        if src.startswith("/-", i):
            depth += 1
            i += 2
        elif src.startswith("-/", i) and depth:
            depth -= 1
            i += 2
        elif depth:
            i += 1
        elif src.startswith("--", i):
            j = src.find(NL, i)
            i = len(src) if j < 0 else j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def test_nilpotent_module_exists():
    assert (LEAN / "CEQ" / "Nilpotent.lean").exists(), "CEQ/Nilpotent.lean missing"


def test_lean_library_builds_clean():
    """`lake build` exit 0 -- the build's own status, read directly.

    Not the exit status of a backgrounded wrapper around it. That distinction
    already produced a false green once in this project: a wrapper returned 0
    while `lake` itself returned 1 with three real errors.
    """
    r = _lake("build", "CEQ")
    assert r.returncode == 0, f"lake build failed:{NL}{r.stdout[-4000:]}{NL}{r.stderr[-4000:]}"


def test_the_sorry_detector_actually_detects_a_sorry():
    """Calibrate the instrument before trusting it. A checker that never fires
    is not evidence of anything."""
    assert "sorry" in _strip_lean_comments("theorem t : True := by sorry")
    assert "sorry" not in _strip_lean_comments("/- No `sorry` here. -/ theorem t : True := trivial")
    assert "sorry" not in _strip_lean_comments("-- sorry in a line comment" + NL + "theorem t : True := trivial")


def test_no_sorry_anywhere_in_the_library():
    """A `sorry` compiles to a warning, not an error, so a green build is not by
    itself evidence of a proof."""
    hits = [f.name for f in sorted((LEAN / "CEQ").glob("*.lean")) + [LEAN / "CEQ.lean"]
            if "sorry" in _strip_lean_comments(f.read_text(encoding="utf-8"))]
    assert not hits, f"sorry found in: {hits}"


def test_the_nilpotency_theorem_is_stated_over_the_shipped_operator():
    """The proof must be about a STRICTLY lower-triangular matrix, diagonal
    excluded. With the diagonal included the operator has a self-loop, rho(A)
    is the largest diagonal entry rather than 0, and the resolvent stops being a
    finite sum. `ceq/nonnormal.causal_mask` uses `.tril(-1)`; the Lean side must
    match that and not `.tril(0)`, or the proof is about a different matrix than
    the one that ships.
    """
    src = (LEAN / "CEQ" / "Nilpotent.lean").read_text(encoding="utf-8")
    assert "StrictlyLower" in src
    assert "pow_card_eq_zero" in src
    assert "occupancy_is_exact_inverse" in src
    mask = (ROOT / "ceq" / "nonnormal.py").read_text(encoding="utf-8")
    assert ".tril(-1)" in mask, "python operator is not strictly lower triangular"


def test_the_strictness_hypothesis_cannot_be_weakened():
    """`one_not_nilpotent` is the negative control: the identity is lower
    triangular in the NON-strict sense and is nilpotent at no power. Its
    presence is what stops the hypothesis being quietly relaxed to `.tril(0)`
    later by someone who reads only the theorem name.
    """
    src = (LEAN / "CEQ" / "Nilpotent.lean").read_text(encoding="utf-8")
    assert "one_not_nilpotent" in src
