"""`CEQ.OracleSeparation` must be compiled, and its hypotheses must hold of the
matrix that actually ships.

WHY THIS FILE EXISTS AND WHY IT IS NOT COVERED BY THE EXISTING GATES. Two of them
miss it, and both misses were found by reading rather than by a failure:

  * `tests/chase/test_lean_refcount_binding.py::test_refcount_is_reachable_from_the_root_module`
    is hardcoded to the string `import CEQ.Refcount`. `lakefile.lean` sets
    `roots := #[`CEQ]`, so `lake build CEQ` compiles only what `CEQ.lean` transitively
    imports; a module the root does not import is dead code behind a green suite. The
    existing test protects exactly one module by name and would not fire for this one.
  * `tests/w3b/test_w3b_lean_nilpotent.py` globs `CEQ/*.lean` for `sorry` and so does
    see this file, but greps rather than compiles it, so a type error would pass.

AND THE PART THAT MATTERS MORE THAN EITHER: a theorem about a different matrix than
the one that ships is worse than no proof. `CEQ/Nilpotent.lean`'s own header says so,
and `tests/w3b/` greps the Python for `.tril(-1)` for that reason. The Lean statement
here is quantified over matrices satisfying `Nonneg`, `SymmSupport` and one strictly
positive entry. Those three are checked below against the transient block that
`scale/foreman_lambda2.py` builds from `ceq/rips.py`, at the engineered `alpha`, on
both E4' graphs -- so the hypotheses are discharged by measurement on the shipped
object rather than assumed to transfer.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEAN = ROOT / "lean"
MODULE = LEAN / "CEQ" / "OracleSeparation.lean"
LAKE = pathlib.Path.home() / ".elan" / "bin" / "lake.exe"
NL = chr(10)

#: Named in `MODEL_CARD.md` and relied on by the round-8 ladder. A rename that keeps
#: the file compiling but drops one of these is exactly the silent break this catches.
LOAD_BEARING = ("CEQ.OracleSeparation.not_isNilpotent",
                "CEQ.OracleSeparation.oracle_ne_resolvent",
                "CEQ.OracleSeparation.truncation_never_exact",
                "CEQ.OracleSeparation.zero_not_a_counterexample",
                "CEQ.OracleSeparation.resolvent_operator_isNilpotent")

STANDARD_AXIOMS = frozenset(("propext", "Quot.sound", "Classical.choice"))


# ------------------------------------------------------------------ reachability

def test_the_module_is_reachable_from_the_root_module():
    root = (LEAN / "CEQ.lean").read_text(encoding="utf-8")
    assert "import CEQ.OracleSeparation" in root, (
        "CEQ.lean does not import CEQ.OracleSeparation, so `lake build CEQ` never "
        "compiles it and the w3b build gate does not cover it")


def test_the_load_bearing_names_are_declared():
    text = MODULE.read_text(encoding="utf-8")
    declared = {"CEQ.OracleSeparation." + line.split()[1].split("{")[0].split("(")[0]
                for line in text.splitlines() if line.startswith("theorem ")}
    missing = [n for n in LOAD_BEARING if n not in declared]
    assert not missing, "cited by name but not declared: {}".format(missing)


def test_no_sorry_in_this_module():
    """A `sorry` compiles to a warning, not an error, so a green build is not by
    itself evidence of a proof. The detector is calibrated on the next line."""
    assert "sorry" in "have h : True := sorry", "the sorry detector cannot see one"
    body = MODULE.read_text(encoding="utf-8")
    body = body[body.index("import Mathlib"):]      # drop the header comment
    assert "sorry" not in body, "sorry found in CEQ/OracleSeparation.lean"


# ------------------------------------------------- the axiom probe, if lean is here

@pytest.mark.skipif(not LAKE.exists() and shutil.which("lake") is None,
                    reason="lean toolchain unavailable")
def test_the_theorems_rest_on_standard_axioms_only(tmp_path):
    """`#print axioms` on each load-bearing name, with a planted `sorry` first so the
    probe is seen to detect `sorryAx` before its silence on the real names is read as
    evidence."""
    lines = ["import CEQ",
             "theorem foreman_calibration_sorry : True := by sorry",
             "#print axioms foreman_calibration_sorry"]
    lines += ["#print axioms " + n for n in LOAD_BEARING]
    scratch = tmp_path / "oracle_axioms.lean"
    scratch.write_text(NL.join(lines) + NL, encoding="utf-8")

    exe = str(LAKE) if LAKE.exists() else "lake"
    r = subprocess.run([exe, "env", "lean", str(scratch)], cwd=LEAN,
                       capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    assert r.returncode == 0, (
        "lean refused the axiom probe -- a theorem was renamed or deleted, or the "
        "library does not build:" + NL + out[-4000:])
    assert "sorryAx" in out, (
        "the probe did not report sorryAx for a theorem proved BY sorry, so its "
        "silence on the real names would mean nothing:" + NL + out[-2000:])
    for name in LOAD_BEARING:
        line = next(ln for ln in out.splitlines() if ln.startswith("'" + name + "'"))
        assert "sorryAx" not in line, line
        used = {tok.strip(" ,[]") for tok in line.split(":", 1)[1].split()}
        assert used <= STANDARD_AXIOMS, "{} uses {}".format(name, used - STANDARD_AXIOMS)


# -------------------------------------- the hypotheses, on the matrix that ships

def _shipped_blocks():
    """`(name, Q)` for both E4' graphs at the engineered killing rate."""
    from ceq.rips import REROUTED_CASES, make_case
    from scale.foreman_lambda2 import TARGET, build_chain, killing_rate

    out = []
    for spec in REROUTED_CASES:
        case = make_case(*spec)
        base = build_chain(case, spec, 1.0)
        chain = build_chain(case, spec, 1.0, alpha=killing_rate(base, TARGET))
        out.append((case.name, chain.Q))
    return out


def test_the_lean_hypotheses_hold_of_the_shipped_transient_block():
    """`Nonneg`, `SymmSupport`, and at least one strictly positive entry.

    These are the three hypotheses of `CEQ.OracleSeparation.not_isNilpotent`, checked
    entry by entry on the actual `Q`. The third is the one that can fail quietly: a
    corpus whose absorbing set swallowed every transient-to-transient edge would leave
    a `Q` that is nilpotent and the theorem would say nothing about it.
    """
    for name, Q in _shipped_blocks():
        assert (Q >= 0.0).all(), name
        support = Q > 0.0
        assert (support == support.T).all(), (
            "{}: support is not symmetric, so the undirected-graph hypothesis "
            "SymmSupport does not hold of the shipped block".format(name))
        assert int(support.sum()) > 0, name
        # And the conclusion, measured: some power stays nonzero. The theorem says
        # every power does; float64 only lets this be checked at a finite one, and the
        # diagonal entry the proof tracks is the thing checked.
        i = int(np.argmax(support.any(axis=1)))
        assert float((Q @ Q)[i, i]) > 0.0, name
        assert float(np.linalg.matrix_power(Q, 64)[i, i]) > 0.0, name


def test_the_hypothesis_check_fires_on_a_matrix_that_is_actually_nilpotent():
    """MUST-FIRE. The arm's own operator -- strictly lower triangular, the `.tril(-1)`
    that `ceq/nonnormal.py` builds -- must FAIL the symmetric-support check.

    Without this the passing half above is worthless: a check that cannot reject
    anything has not accepted anything either. This is the fifteenth control in this
    project written against that failure mode rather than around it.
    """
    rng = np.random.default_rng(0x33960008)
    A = np.tril(rng.random((16, 16)), -1)
    support = A > 0.0
    assert (A >= 0.0).all(), "the planted matrix is non-negative, as Q is"
    assert int(support.sum()) > 0, "the planted matrix is not the zero matrix"
    assert not (support == support.T).all(), (
        "the symmetric-support check did NOT reject a strictly lower-triangular "
        "matrix, so it cannot distinguish the oracle's operator from the arm's")
    assert float(np.abs(np.linalg.matrix_power(A, 16)).max()) == 0.0, (
        "the planted matrix is nilpotent at 16, which is the property the "
        "separation turns on")
