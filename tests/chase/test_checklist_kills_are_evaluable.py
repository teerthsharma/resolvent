"""CHECKLIST kill clauses that cannot fire. RED ON PURPOSE.

Chase established the defect class on M2: the frozen kill's second clause reads
`slope(c NOT in P)`, a quantity that is ZERO BY CONSTRUCTION -- `hop2[i,j] =
sum_{p in P} A[i,p]A[p,j]` contains no term indexed by `c` when `c` is not in
`P` -- and `scale/m2_units.py::_verdict` maps the resulting NaN slope to
`k2 = False`, printed as "control decays as the mechanism requires".

This file audits the REST of the checklist for the same class. Every assertion
below states the SAFE property: that the quantity the kill clause names is
capable of taking a value that fires it, or that some code in this repo
evaluates it at all. A test that fails here is a kill clause that cannot kill.

Nothing is reimplemented. `ceq.eviction` is imported from the module under test;
the corpus in `_context` is the same draw `tests/w3/test_w3_eviction.py` uses,
because a different corpus would measure a different thing.
"""
from __future__ import annotations

import pathlib

import torch

from ceq import eviction as ev

ROOT = pathlib.Path(__file__).resolve().parents[2]

N_TOK, D_FEAT, RHO, KEEP = 96, 32, 0.9, 16

SELF = "tests/chase/test_checklist_kills_are_evaluable.py"


def _sources() -> dict:
    """Every .py and .lean in the repo, excluding vendored Lean packages."""
    out = {}
    for pat in ("**/*.py", "**/*.lean"):
        for f in ROOT.glob(pat):
            s = str(f)
            if ".lake" in s or "__pycache__" in s:
                continue
            out[str(f.relative_to(ROOT)).replace("\\", "/")] = f.read_text(
                encoding="utf-8", errors="replace")
    return out


def _hits(needles, src=None) -> dict:
    got = {}
    for name, text in (src or _sources()).items():
        if name == SELF:
            continue
        found = [n for n in needles if n in text]
        if found:
            got[name] = found
    return got


# --------------------------------------------------------------------- M4

def _context(seed: int):
    """tests/w3/test_w3_eviction.py::context, same generator, same edits."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(N_TOK, D_FEAT, generator=g, dtype=torch.float64)
    x[:, 0] *= 0.25
    causal = int(torch.randint(2, N_TOK, (1,), generator=g))
    x[causal, 0] = 3.0
    return x


def _edit(x, j, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 7919)
    d = torch.randn(x.shape[1], generator=g, dtype=x.dtype)
    y = x.clone()
    y[j] = y[j] + d / d.norm()
    return y


def test_m4_max_change_under_eviction_is_capable_of_exceeding_1e_minus_12():
    """M4 Kill: "max change > 1e-12 in the evict-before-read window".

    The quantity is `(settle_evicted(y) - settle_evicted(x)).abs().max()` under
    the W3 protocol. `settle_evicted` reads `x[keep]` and nothing else
    (ceq/eviction.py:118-121), and the perturbed token comes from
    `lowest_salience_token(..., exclude=keep)` (ceq/eviction.py:71-76), which
    excludes `keep` by construction. So `y[keep] == x[keep]` bitwise and the
    difference is identically 0.0 -- the M2 `not_in_P` defect, on a tensor.

    SAFE PROPERTY: over the W3 draws the perturbed token must sometimes land
    inside `keep`, and the measured change must sometimes be nonzero. Otherwise
    the clause reads a structural constant.
    """
    in_keep = 0
    moved = 0
    worst = 0.0
    for s in range(8):
        x = _context(s)
        keep = ev.keep_indices(x, KEEP, RHO)
        crushed = ev.lowest_salience_token(x, RHO, exclude=keep)
        in_keep += int((keep == crushed).any())
        y = _edit(x, crushed, s)
        d = float((ev.settle_evicted(y, keep, RHO)
                   - ev.settle_evicted(x, keep, RHO)).abs().max())
        worst = max(worst, d)
        moved += int(d > 1e-12)
    assert in_keep > 0, (
        "the perturbed token was outside `keep` in all 8 draws; "
        "lowest_salience_token(exclude=keep) cannot return a kept token")
    assert moved > 0, (
        "max change was exactly 0.0 in {} of 8 draws (worst seen {!r}); the M4 "
        "kill reads a quantity that is zero by construction".format(
            8 - moved, worst))


# --------------------------------------------------------------------- M5

def test_m5_the_lean_truncation_index_hypothesis_is_bound_to_the_shipped_hops():
    """M5 Kill: "any hypothesis of the Lean theorem not satisfied by the tensor
    that actually ships".

    `CEQ.Occupancy.occupancy A N = sum_{k<N} A^k` (lean/CEQ/Occupancy.lean:42)
    and `occupancy_is_exact_inverse` (lean/CEQ/Nilpotent.lean:96-100) is stated
    at `N = n`, the matrix dimension. The shipped forward truncates at
    `DEFAULT_HOPS = 4` (ceq/attention.py:88) and the parity point is `hops = 2`
    (ceq/hf/modeling_ceq.py:329), while `n` is the sequence length. `A^hops` is
    NOT zero at those settings, so that hypothesis is violated by the shipping
    tensor -- and the grep-binding
    `tests/w3b/test_w3b_lean_nilpotent.py:105-118` never looks at it: it scans
    for three identifier strings and for `.tril(-1)` in ceq/nonnormal.py.

    SAFE PROPERTY: the M5 binding must read the truncation depth the module
    actually runs at. `tests/w15:77` and `tests/w6:251` raise `A` to the power
    `n` (the matrix dimension), which is the theorem's own index and not the
    shipped one; no file raises it to the power `hops`.
    """
    a = torch.randn(64, 64, dtype=torch.float64).tril(-1)
    p = torch.linalg.matrix_power(a, 4)
    assert float(p.abs().max()) > 0.0, (
        "A^4 vanished on a 64x64 strictly lower matrix; the demonstration that "
        "the truncation hypothesis is violated at hops = 4 is itself broken")

    w3b = _sources()["tests/w3b/test_w3b_lean_nilpotent.py"]
    reads_the_shipped_depth = ("hops" in w3b) or ("DEFAULT_HOPS" in w3b)
    assert reads_the_shipped_depth, (
        "the M5 grep-binding never mentions the hop count. It checks three "
        "identifier strings and `.tril(-1)`; the theorem is instantiated at "
        "N = n (the sequence length) and the module truncates at "
        "DEFAULT_HOPS = 4, where A^N is demonstrably nonzero. That hypothesis "
        "cannot fire the kill because nothing reads it")


# --------------------------------------------------------------------- M3

def test_m3_kill_quantities_are_measured_by_something():
    """M3 Kill: "pivot arm above NRMSE 1.0, OR CIs overlap softmax at every
    d >= 256, OR the unsigned-pivot ablation matches it".

    `scale/negation_scope.py` builds the corpus, the oracle and the bar, and
    stops there: `main()` runs `calibrate_bar` only (lines 148-175). No arm is
    trained, no softmax failure distance is recorded, and `bootstrap_ci`
    (line 104) has no caller anywhere.

    SAFE PROPERTY: the three quantities must be produced by some code.
    """
    src = _sources()
    callers = [n for n, t in src.items()
               if "bootstrap_ci" in t and n not in (SELF, "scale/negation_scope.py")]
    ablation = [n for n, t in src.items()
                if n != SELF and "unsigned" in t and "nrmse" in t.lower()]
    assert callers, (
        "`scale/negation_scope.bootstrap_ci` is defined and never called; the "
        "'CIs overlap softmax' clause reads a CI nothing computes")
    assert ablation, (
        "no file computes an NRMSE for an unsigned arm; the 'unsigned-pivot "
        "ablation matches it' clause reads a number that does not exist")


# --------------------------------------------------------------------- M6

def test_m6_the_numerical_radius_guard_the_kill_reads_exists():
    """M6 Kill: "guard reintroduces decay in M2, OR training diverges under the
    constraint at every lr in the sweep".

    Both quantities are conditional on the guard `w(A) <= rho` existing in the
    forward. The forward normalizes ROW L1 to rho (ceq/attention.py:189), which
    bounds `||A||_inf`, not the numerical radius. ARSENAL.md:71 records the
    measured `w(A) = 1.499315 > 1` and calls the bound "vacuous as it stands".

    SAFE PROPERTY: some code must compute w(A) and enforce it, or neither half
    of the kill has an input.
    """
    got = _hits(("numerical_radius", "numerical radius", "field_of_values",
                 "num_radius", "w_of_A"))
    assert got, (
        "no module computes a numerical radius; M6's forward guard does not "
        "exist, so 'guard reintroduces decay in M2' and 'training diverges "
        "under the constraint' both read unmeasured quantities")


# --------------------------------------------------------------------- S3

def test_s3_the_derived_hop_coefficients_exist_to_be_compared():
    """S3: "Derived hop coefficients (Carnot eps^h grading) beat learned scalars
    gamma_k at matched params -- else learned scalars ship".

    The learned side exists: `ceq/bench.py:390` runs `sum_k gamma_k A^k h`. The
    derived side appears only in ARSENAL.md and CHECKLIST.md prose, and
    ARSENAL.md:191 already records that the operator is "NOT horizontal", so
    "Carnot structure is VACUOUS" on it.

    SAFE PROPERTY: an eps^h graded coefficient vector must exist in code for the
    comparison this clause decides to be runnable at all.
    """
    got = _hits(("carnot", "Carnot", "CARNOT"))
    assert got, (
        "no Python or Lean file implements the Carnot eps^h grading; S3 "
        "compares a derived coefficient vector that was never written")


# --------------------------------------------------------------------- G4

def test_g4_has_an_evaluator():
    """G4: "M1-M6 all GREEN but S2 shows the capability lives entirely in the
    unsigned ablation -- stop, rewrite the claim, re-enter at M3".

    `scale/s2_probe.py` can now measure `pivot_unsigned`, so the input exists.
    What does not exist is anything that reads it and decides G4: the string
    "G4" occurs in exactly one Python file, as prose inside
    `scale/negation_scope.py`'s docstring (line 40).

    SAFE PROPERTY: some code must compare the unsigned ablation against the
    signed arm and emit a G4 verdict, the way `run_calib.py` emits G2 and
    `tests/loop/test_arms_distinct.py` emits G3.
    """
    src = _sources()
    got = {n: t for n, t in src.items()
           if n.endswith(".py") and n != SELF and "G4" in t}
    evaluators = [n for n, t in got.items() if "def " in t and "s2" in t.lower()]
    assert evaluators, (
        "G4 appears in {} and in no evaluator; G2 has run_calib.py and G3 has "
        "tests/loop/test_arms_distinct.py, G4 has nothing that can fire "
        "it".format(sorted(got)))
