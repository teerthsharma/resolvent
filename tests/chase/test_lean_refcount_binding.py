"""Bind `lean/CEQ/Refcount.lean` -- the only live provenance claim -- to a test.

WHAT WAS ALREADY GATED, AND WHAT WAS NOT. `tests/w3b/test_w3b_lean_nilpotent.py`
runs `lake build CEQ` and greps every `CEQ/*.lean` for `sorry`, so Refcount.lean
was covered for *compiling* and for *sorry-freedom* the moment `CEQ.lean` gained
`import CEQ.Refcount`. Nothing covered its CONTENT.

THE RED, MEASURED BEFORE THIS FILE EXISTED. `free_face_floor_unchanged` and
`shared_plaque_floor_drops` -- the corollary the whole object exists for and the
converse that stops it being vacuous -- were deleted from Refcount.lean, taking
it from 10 theorems to 8. `lake build CEQ` returned **exit 0** in 32.9 s and the
w3b gate was untouched: a file that compiles and has no `sorry` still compiles
and still has no `sorry` when its two load-bearing statements are gone. The
`sorry` grep cannot see a deletion, and the build cannot see one either, because
nothing in the library imports these theorems.

WHY THIS FILE MATTERS MORE THAN THE OTHER FOUR LEAN MODULES. `Contraction`,
`Occupancy`, `Nilpotent` and `OrbitBound` each state a fact about the operator.
Refcount states the only fact that is the AUTHOR'S OWN on both sides: `caustic`
Theorem 1 (Zenodo 10.5281/zenodo.21997746) supplies the floor, `foliation`
supplies the refcount, and the theorem is that they are the same integer. That
is the provenance the release rests on, so it is the one that must fail loudly.

FOUR GATES, ONE LEAN SUBPROCESS. `#print axioms` subsumes them at once: a broken
build makes `lake env lean` exit non-zero, a renamed or deleted theorem makes it
exit non-zero with "unknown identifier", a `sorry` anywhere in a proof puts
`sorryAx` in that theorem's axiom list, and an added axiom shows up by name. The
scratch file carries its OWN deliberate `sorry` AND a real proof with the word
`sorry` in a comment beside it, so both directions of the detector are calibrated
in the same process -- a checker never seen to fire is not evidence, and one that
fires on the documentation asserting there are no sorries is worse than none.

THE SCOPE IS THE WHOLE LIBRARY, not just this module. Refcount is the headline
because it is the provenance claim, but the same gap covered all 27 theorems
across `Contraction`, `Nilpotent`, `Occupancy`, `OrbitBound` and `Refcount`: the
theorem list is DISCOVERED from the sources so a newly added proof is covered the
day it lands, and `LOAD_BEARING` is a separate hand-written list of the names
that documents and module docstrings cite, which is the only way to catch a
deletion.

NO DEVICE PARAMETRISATION, for the reason `tests/w3b` states: a Lean proof has no
device axis and a fake `device` parameter would run the identical subprocess
twice and report two passes for one fact.
"""
from __future__ import annotations

import itertools
import pathlib
import random
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEAN = ROOT / "lean"
REFCOUNT = LEAN / "CEQ" / "Refcount.lean"
LAKE = pathlib.Path.home() / ".elan" / "bin" / "lake.exe"
NL = chr(10)

#: The claims each module's own header prose makes, BY NAME. This is the contract
#: between the prose and the proofs: a rename or a deletion of any of these makes
#: `#print axioms` exit non-zero with "unknown identifier". The Refcount five are
#: the provenance claim; the rest are the statements MODEL_CARD.md and
#: `ceq/attention.py` cite by name in their own docstrings.
LOAD_BEARING = (
    "CEQ.Refcount.floor_add_orbits",
    "CEQ.Refcount.floor_eq_sum_refcount_pred",
    "CEQ.Refcount.image_survivors",
    "CEQ.Refcount.evict_floor_add_refcount_pred",
    "CEQ.Refcount.free_face_floor_unchanged",
    "CEQ.Refcount.shared_plaque_floor_drops",
    "CEQ.Nilpotent.pow_card_eq_zero",
    "CEQ.Nilpotent.occupancy_is_exact_inverse",
    "CEQ.Nilpotent.one_not_nilpotent",
    "CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent",
    "CEQ.OrbitBound.orbit_error_bound",
    "CEQ.OrbitBound.orbit_error_bound_attained",
    "CEQ.Contraction.weighted_contraction",
    "CEQ.Contraction.rowStochastic_perron",
)

#: What a mathlib proof is allowed to rest on. `sorryAx` is the one that must
#: never appear. Some of these theorems are pure `Nat` arithmetic and rest on
#: FEWER than three, so the invariant is a subset, not equality.
STANDARD_AXIOMS = frozenset(("propext", "Quot.sound", "Classical.choice"))

DOI = "10.5281/zenodo.21997746"

_DECL = re.compile(r"(?:private\s+|protected\s+)?(?:theorem|lemma)\s+"
                   r"([A-Za-z_][A-Za-z0-9_'!?]*)")

needs_lean = pytest.mark.skipif(
    not LAKE.exists() and shutil.which("lake") is None,
    reason="lean toolchain unavailable",
)


def declarations():
    """Every top-level `theorem`/`lemma` in the library, fully qualified.

    DISCOVERED, NOT LISTED. A hand-written list covers the theorems that existed
    the day it was written; the next one added is unchecked and nobody notices.
    `LOAD_BEARING` above is the separate, deliberately hand-written list, and it
    exists for the opposite job -- catching a DELETION, which discovery cannot.
    """
    out = []
    for f in sorted((LEAN / "CEQ").glob("*.lean")):
        ns = []
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.match(r"namespace\s+(\S+)", line)
            if m:
                ns.append(m.group(1))
                continue
            m = re.match(r"end\s+(\S+)\s*$", line)
            if m and ns and ns[-1] == m.group(1):
                ns.pop()
                continue
            m = _DECL.match(line)
            if m:
                out.append(".".join(ns + [m.group(1)]))
    return out


def _axiom_report(tmp_path, names):
    """One `lean` subprocess, `#print axioms` for every name, plus BOTH halves of
    the detector calibration. Returns `{name: frozenset(axioms)}`.

    THE CALIBRATION IS IN THE SAME PROCESS ON PURPOSE. A `sorry` compiles to a
    warning, so a green build proves nothing, and a checker never seen to fire
    proves nothing either. `chase_calibration_sorry` is a literal `by sorry` and
    its axiom list MUST contain `sorryAx`. `chase_calibration_prose` carries the
    word sorry in a comment and a real proof, and MUST NOT -- that is the trap a
    previous grep-based detector fell into when it fired on CEQ.lean's own
    sentence "No `sorry` anywhere".
    """
    lines = ["import CEQ",
             "theorem chase_calibration_sorry : True := by sorry",
             "#print axioms chase_calibration_sorry",
             "-- No `sorry` anywhere below this line; the detector must stay silent.",
             "theorem chase_calibration_prose : True := trivial",
             "#print axioms chase_calibration_prose"]
    lines += ["#print axioms " + n for n in names]
    scratch = tmp_path / "ceq_axioms.lean"
    scratch.write_text(NL.join(lines) + NL, encoding="utf-8")

    exe = str(LAKE) if LAKE.exists() else "lake"
    r = subprocess.run([exe, "env", "lean", str(scratch)], cwd=LEAN,
                       capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, (
        "lean refused the axiom probe -- a theorem was renamed or deleted, or "
        "the library does not build:" + NL + out[-4000:])

    got = {}
    for line in out.splitlines():
        m = re.match(r"'(\S+)' depends on axioms: \[(.*)\]", line.strip())
        if m:
            got[m.group(1)] = frozenset(
                a.strip() for a in m.group(2).split(",") if a.strip())
        else:
            m = re.match(r"'(\S+)' does not depend on any axioms", line.strip())
            if m:
                got[m.group(1)] = frozenset()

    assert got.get("chase_calibration_sorry") and "sorryAx" in got["chase_calibration_sorry"], (
        "the calibration theorem is a literal `by sorry` and sorryAx did not "
        "appear, so this check cannot detect a real one:" + NL + out)
    assert got.get("chase_calibration_prose") == frozenset(), (
        "the word `sorry` in a COMMENT made the detector fire; that is the false "
        "positive a grep-based detector already produced once:" + NL + out)
    return got


# ------------------------------------------------------------------ the proofs

@needs_lean
def test_every_theorem_in_the_library_rests_on_no_axiom_beyond_the_classical_three(tmp_path):
    """The single Lean subprocess, over EVERY theorem the library declares.

    This one check subsumes three separate gates: a broken build makes `lake env
    lean` exit non-zero, a renamed or deleted `LOAD_BEARING` name makes it exit
    non-zero with "unknown identifier", and a `sorry` reached from any proof puts
    `sorryAx` in that theorem's axiom list. `tests/w3b` already gates the build
    and greps for `sorry`; neither notices a deletion, and neither notices an
    added axiom.

    Measured cost: 27.0 s for five names, against 30.2 s for a no-op
    `lake build CEQ`. The cost is olean loading, so it is one subprocess for all
    of them rather than one per module.
    """
    names = declarations()
    assert len(names) >= 27, (len(names), names)
    got = _axiom_report(tmp_path, names)

    missing = [n for n in names if n not in got]
    assert not missing, ("no axiom line came back for: {}".format(missing))

    bad = {n: sorted(a) for n, a in got.items()
           if n.startswith("CEQ.") and not a <= STANDARD_AXIOMS}
    assert not bad, (
        "these rest on an axiom outside {} -- sorryAx here means a proof was "
        "abandoned:{}{}".format(sorted(STANDARD_AXIOMS), NL, bad))


@needs_lean
def test_the_named_theorems_the_documents_cite_still_exist(tmp_path):
    """The deletion gate. Discovery in the test above cannot catch a removal --
    it enumerates what is there. THE RED, MEASURED: `free_face_floor_unchanged`
    and `shared_plaque_floor_drops` were deleted from Refcount.lean, taking it
    from 10 theorems to 8, and `lake build CEQ` returned exit 0 in 32.9 s with
    the whole w3b gate untouched."""
    got = _axiom_report(tmp_path, list(LOAD_BEARING))
    for name in LOAD_BEARING:
        assert name in got, (
            "{} is cited by name in a document or a module docstring and lean "
            "did not report axioms for it".format(name))
        assert got[name] <= STANDARD_AXIOMS, (name, sorted(got[name]))


def test_the_load_bearing_names_are_actually_declared_in_the_sources(tmp_path):
    """The cheap half, so a broken toolchain does not silently skip the deletion
    gate entirely. Runs no subprocess."""
    declared = set(declarations())
    missing = [n for n in LOAD_BEARING if n not in declared]
    assert not missing, "cited by name but not declared: {}".format(missing)


def test_refcount_is_reachable_from_the_root_module():
    """`lake build CEQ` only compiles what `CEQ.lean` imports. Drop that one
    import line and the w3b build gate silently stops covering this file while
    still reporting exit 0 -- the file would be dead code with a green suite."""
    root = (LEAN / "CEQ.lean").read_text(encoding="utf-8")
    assert "import CEQ.Refcount" in root, (
        "CEQ.lean does not import CEQ.Refcount, so `lake build CEQ` never "
        "compiles it and the w3b build gate does not cover it")


# ------------------------------------------- the arithmetic, on actual finite data

def _floor(f, plaques):
    """`n - m` on a concrete map, computed the way `caustic` Theorem 1 states."""
    return len(f) - len(plaques)


def _model(f):
    """(plaques, refcount) for a map given as a tuple `f[e] = p`."""
    plaques = set(f)
    return plaques, {p: sum(1 for e in f if e == p) for p in plaques}


def test_the_linking_identity_is_true_of_actual_finite_caches():
    """`n - m = sum over plaques of (refcount - 1)`, checked exhaustively over
    every map from 5 prefixes to 3 plaques (243 of them) and then on random
    larger ones. A Lean theorem that is true only of the empty case would pass
    `#print axioms` and fail here."""
    for f in itertools.product(range(3), repeat=5):
        plaques, rc = _model(f)
        assert _floor(f, plaques) == sum(rc[p] - 1 for p in plaques), f

    rng = random.Random(0)
    for _ in range(200):
        n, k = rng.randint(1, 30), rng.randint(1, 8)
        f = tuple(rng.randrange(k) for _ in range(n))
        plaques, rc = _model(f)
        assert _floor(f, plaques) == sum(rc[p] - 1 for p in plaques), f


def test_evicting_a_free_face_leaves_the_floor_where_it_was_and_a_shared_one_lowers_it():
    """The corollary and its converse, as arithmetic. The converse is what stops
    the criterion carrying zero bits: without it, "free faces are safe" would be
    consistent with everything being safe."""
    rng = random.Random(1)
    saw_free = saw_shared = 0
    for _ in range(400):
        n, k = rng.randint(2, 24), rng.randint(1, 6)
        f = tuple(rng.randrange(k) for _ in range(n))
        plaques, rc = _model(f)
        before = _floor(f, plaques)
        for p in plaques:
            surv = tuple(e for e in f if e != p)
            after = _floor(surv, set(surv))
            assert after + (rc[p] - 1) == before, (f, p)
            if rc[p] == 1:
                assert after == before, (f, p)
                saw_free += 1
            else:
                assert after < before, (f, p)
                saw_shared += 1
    assert saw_free and saw_shared, (saw_free, saw_shared)


def test_the_truncated_subtraction_hazard_the_header_names_is_real():
    """Refcount.lean's header says `omega` refused the first draft because at
    `n = 3, m = 3, k = 3` the subtractive form has left side 0 and right side
    truncating to 2. Reproduced here in Python's own truncated subtraction, so
    the reason the theorem is stated over `survivors` is checkable rather than
    remembered. If this stops reproducing, the header's account of why the
    theorem has the shape it has is wrong."""
    def nat_sub(a, b):
        return max(a - b, 0)

    n, m, k = 3, 3, 3                                  # prefixes, plaques, refcount
    left = nat_sub(n, m)                               # the floor, `n - m`
    right = nat_sub(nat_sub(n, k), nat_sub(m, 1)) + nat_sub(k, 1)   # the draft's sum
    assert (left, right) == (0, 2), (left, right)

    #: and the control: no map from 3 prefixes onto 3 distinct plaques has a
    #: fibre of size 3, which is exactly why `survivors` makes the constraint a
    #: consequence instead of an assumption.
    assert not [f for f in itertools.product(range(3), repeat=3)
                if len(set(f)) == 3 and max(_model(f)[1].values()) == 3]


# --------------------------------------------- the shipped side of the identity

def test_the_free_face_criterion_is_named_on_both_sides_and_the_shipped_one_refuses():
    """The `.tril(-1)` bind, repeated for this pair: the Lean predicate and the
    Python refusal must both be present, or proof and code have drifted.

    THE CORRESPONDENCE IS PARTIAL AND THAT IS STATED, NOT PAPERED OVER.
    `CEQ.Refcount.IsFreeFace` is `refcount f p = 1`, a fibre cardinality.
    `ceq/hopcache.py::HopCache.evict` admits on SCOPE DEPTH -- an entry
    registered by an enclosing scope is refused. The two agree on the shape of
    the rule (only a free face is evictable, and the refusal is an exception
    rather than a silent corruption) and disagree on the predicate: no refcount
    is maintained anywhere in the shipped Python. `grep -rn refcount --include=*.py`
    hits documentation prose only. So these theorems certify the CRITERION, not
    yet this implementation of it, and the day someone maintains a real refcount
    this test must fail and be rewritten to bind the two predicates together.
    """
    src = REFCOUNT.read_text(encoding="utf-8")
    assert "def IsFreeFace" in src and "refcount f p = 1" in src, (
        "the Lean free-face predicate is not `refcount f p = 1` any more")

    hop = (ROOT / "ceq" / "hopcache.py").read_text(encoding="utf-8")
    assert "free face" in hop, "the shipped eviction no longer names the criterion"

    import torch

    from ceq.hopcache import HopCache

    c = HopCache(hops=2)
    q = torch.zeros(1, 1, 1, 4)
    for _ in range(3):
        c.step(q, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4), alpha=0.0, rho=1.0)
    c.push_scope()
    c.step(q, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4), alpha=0.0, rho=1.0)
    with pytest.raises(RuntimeError, match="free face"):
        c.evict(0)                       # enclosing scope: not a free face
    c.evict(3)                           # current scope: admissible
    assert c.n_positions() == 3

    #: the predicate the theorems use is absent from the shipped package, and
    #: this asserts that absence so the docstring above cannot go stale silently.
    py = [p for p in sorted((ROOT / "ceq").glob("*.py"))
          if "refcount" in p.read_text(encoding="utf-8").lower()
          and "def refcount" in p.read_text(encoding="utf-8")]
    assert not py, (
        "a real refcount now ships in {}; rewrite this test to bind it to "
        "CEQ.Refcount.refcount".format([p.name for p in py]))


# ------------------------------------------------------------------- provenance

def test_the_provenance_doi_matches_the_theorem_it_is_borrowed_from():
    """Refcount.lean's floor is `caustic` Theorem 1, which `OrbitBound.lean`
    formalises. Two files citing two different DOIs for one result is the exact
    failure a provenance claim cannot survive."""
    assert DOI in REFCOUNT.read_text(encoding="utf-8")
    assert DOI in (LEAN / "CEQ" / "OrbitBound.lean").read_text(encoding="utf-8")


def test_the_model_card_cites_the_module_and_counts_the_theorems_it_ships():
    """The Inspector's finding was that this file is cited in no document. The
    model card is what reaches the Hub, so it is the document that has to name
    it -- and the theorem count it prints has to be the real one, or the first
    thing a reader checks is already wrong."""
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "CEQ.Refcount" in card, "MODEL_CARD.md does not cite CEQ.Refcount"
    assert DOI in card, "MODEL_CARD.md does not carry the caustic DOI"

    actual = sum(
        sum(1 for line in f.read_text(encoding="utf-8").splitlines()
            if line.startswith(("theorem ", "lemma ")))
        for f in sorted((LEAN / "CEQ").glob("*.lean")))
    assert "**{} theorems**".format(actual) in card, (
        "MODEL_CARD.md's theorem count is not {}; the lean library has {} "
        "top-level theorems across {} modules".format(
            actual, actual, len(list((LEAN / "CEQ").glob("*.lean")))))
