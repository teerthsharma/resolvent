"""G0.10 RULING 5 -- the Q1/Q2 instrument, hashed, and its batch path checked.

`scripts/v15_r1.py` is ADOPTED as the named instrument for Q1/Q2
(`V17K_RULINGS.md` RULING 5, verbatim: "Hash into the identity manifest; one
L-SCOPE check (its batch path == production path); Q1/Q2 cite it by hash.").
Three things are owed and each gets its own block below:

  1. THE HASH. `identity_manifest.instrument_manifest` (new, additive --
     `scale/identity_manifest.py` is not reimplemented, its existing `_sha`
     and `_code_fingerprint` are reused). Proved non-vacuous with a planted
     one-character edit, and proved to cover more than the file's own bytes
     with a planted edit to a helper `scripts/v15_r1.py` calls out to.
  2. THE L-SCOPE CHECK. `scripts/v15_r1.py`'s batch construction, RUN, against
     `scale/r10_capacity_sweep.py`'s -- the production capacity-sweep script
     -- RUN, on the same seed, and diffed by VALUE. Not a name comparison:
     `test_l_scope_a_mismatched_seed_is_caught` plants a divergence and shows
     the comparison can fail before trusting the one that says it does not.
  3. THE CITATION. `scripts/v15_r1.py`'s own `t="cell"` records now carry
     `instrument_hash`, wired at the write site rather than left as a step a
     future run has to remember.

NOTHING TRAINS AS A RESEARCH READING. Every `main()` invocation below runs at
a throwaway shape (`n_train<=64`) and a throwaway `--tag`, journalled to
`tmp_path` and never to `results/`. No NRMSE, no bar reading and no cell here
is a finding -- they exist only so the two files' batch-construction CALLS can
be captured and diffed.

CPU only, per this node's remit.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

from scale import identity_manifest
from scale import negation_scope

V15_PATH = None  # resolved in a fixture below, once scripts.v15_r1 is importable


# ============================================================ 1. THE HASH


def test_instrument_manifest_is_a_pure_read_and_reproducible():
    """Two calls over the same file and the same reach set agree bitwise --
    G2: a fix here may not move a published number, and an instrument that
    changed hash on every call could never be cited by anything downstream."""
    from scripts import v15_r1

    m1 = identity_manifest.instrument_manifest(v15_r1.__file__, reaches=())
    m2 = identity_manifest.instrument_manifest(v15_r1.__file__, reaches=())
    assert m1 == m2
    assert m1["n_reaches"] == 0
    assert len(m1["hash"]) == 64  # sha256 hex


def test_hash_moves_on_a_planted_one_character_edit(tmp_path):
    """RULING 5's own requirement: 'prove it with a planted one-character
    edit and show the digest changes.' The manifest never imports or runs the
    file -- a byte flip that breaks Python syntax is still a valid input."""
    from scripts import v15_r1

    original = identity_manifest.instrument_manifest(v15_r1.__file__, reaches=())

    src = open(v15_r1.__file__, "rb").read()
    assert src.count(b"BIND_BAR = 1e-6") == 1, "the constant this test flips moved"
    mutated_src = src.replace(b"BIND_BAR = 1e-6", b"BIND_BAR = 1e-7")
    mutated_path = tmp_path / "v15_r1_mutated.py"
    mutated_path.write_bytes(mutated_src)

    mutated = identity_manifest.instrument_manifest(str(mutated_path), reaches=())

    assert mutated["file"] != original["file"]
    assert mutated["hash"] != original["hash"]
    # NON-DEGENERACY, the other direction: an UNCHANGED copy must NOT move.
    unchanged_path = tmp_path / "v15_r1_copy.py"
    unchanged_path.write_bytes(src)
    unchanged = identity_manifest.instrument_manifest(str(unchanged_path), reaches=())
    assert unchanged["file"] == original["file"]
    assert unchanged["hash"] == original["hash"]


def _load_helper_module(name: str, source: str, tmp_path) -> types.ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_file_bytes_alone_are_defeated_by_a_helper_edit_in_another_module(tmp_path):
    """THE REASON `reach` EXISTS, DEMONSTRATED. `scripts/v15_r1.py` scores a
    cell through callables that live in `scale/negation_scope.py`,
    `ceq/arm_pl.py` and friends -- files this test's own repo edit rules
    forbid touching. So the demonstration is built from two throwaway helper
    modules standing in for 'a module the instrument calls out to', edited by
    exactly one character, with `scripts/v15_r1.py`'s OWN path held fixed and
    unedited on both sides.

    A hash over the instrument's bytes ALONE (`reaches=()`) is identical for
    both -- the file did not move. `reach` is what catches it.
    """
    from scripts import v15_r1

    helper_v1 = _load_helper_module(
        "g14_helper_v1", "def batch_draw(n):\n    return n + 1\n", tmp_path)
    helper_v2 = _load_helper_module(
        "g14_helper_v2", "def batch_draw(n):\n    return n + 2\n", tmp_path)

    file_only_a = identity_manifest.instrument_manifest(v15_r1.__file__, reaches=())
    file_only_b = identity_manifest.instrument_manifest(v15_r1.__file__, reaches=())
    assert file_only_a["hash"] == file_only_b["hash"], (
        "the instrument's own bytes did not move between these two calls, so "
        "a file-only hash cannot see the helper edit below -- that is the gap "
        "this test exists to show"
    )

    with_v1 = identity_manifest.instrument_manifest(
        v15_r1.__file__, reaches=(helper_v1.batch_draw,))
    with_v2 = identity_manifest.instrument_manifest(
        v15_r1.__file__, reaches=(helper_v2.batch_draw,))

    assert with_v1["file"] == with_v2["file"], "the instrument's file did not move"
    assert with_v1["reach"] != with_v2["reach"], (
        "the ONE edited character in the helper must move `reach`"
    )
    assert with_v1["hash"] != with_v2["hash"]


# ==================================================== 2. THE L-SCOPE CHECK


def _draw(fn, n, s, d, *, d_model, seed, device):
    return fn(n, s, d, d_model=d_model, seed=seed, device=device)


def test_l_scope_shared_registry_object_not_two_copies():
    """The weakest possible form of 'same batch path' -- that it is not even
    a SECOND implementation -- checked first because everything below is
    only meaningful if this holds. `M3_TASKS["e3_t2"]` must be the identical
    tuple object in both files' namespaces, not two dicts that merely agree
    by value today."""
    from scripts import v15_r1
    from scale import r10_capacity_sweep as r10

    assert v15_r1.M3_TASKS is r10.M3_TASKS is negation_scope.M3_TASKS
    task = f"e3_t{v15_r1.T_STAR}"
    assert v15_r1.M3_TASKS[task] is r10.M3_TASKS[task]
    assert (v15_r1.S, v15_r1.D) == (r10.S, r10.D)
    assert v15_r1.D_MODEL is r10.D_MODEL  # scale.m3_capability.D_MODEL, one import


def test_l_scope_a_mismatched_seed_is_caught(tmp_path):
    """NON-DEGENERACY FOR THE CHECK BELOW. Before trusting a `torch.equal`
    that says two draws match, show it can say NO -- otherwise a check that
    always passes is exactly the shape of the 15 vacuous controls this repo
    has already struck."""
    from scripts import v15_r1

    task = f"e3_t{v15_r1.T_STAR}"
    fn = v15_r1.M3_TASKS[task][0]
    x_a, y_a, _, _ = _draw(fn, 32, v15_r1.S, v15_r1.D, d_model=v15_r1.D_MODEL,
                           seed=0, device=None)
    x_b, y_b, _, _ = _draw(fn, 32, v15_r1.S, v15_r1.D, d_model=v15_r1.D_MODEL,
                           seed=1, device=None)
    assert not torch.equal(x_a, x_b), "two different seeds drew the same tensor"
    x_c, y_c, _, _ = _draw(fn, 32, v15_r1.S, v15_r1.D, d_model=v15_r1.D_MODEL,
                           seed=0, device=None)
    assert torch.equal(x_a, x_c) and torch.equal(y_a, y_c), (
        "the SAME seed must redraw the SAME tensor -- make_equilibrium_batch's "
        "generator is local to the call (scale/negation_scope.py), not global "
        "state a second call could have advanced"
    )


def test_l_scope_batch_path_equals_r10_capacity_sweep_production_path(
        tmp_path, monkeypatch):
    """THE CHECK RULING 5 ASKS FOR. `scripts/v15_r1.py`'s `main()` and
    `scale/r10_capacity_sweep.py`'s `main()` -- the script this repo's other
    capability cells are produced by -- are BOTH RUN, at matched shapes and
    the SAME seeds, with the shared `M3_TASKS` entry wrapped by a recording
    spy so every batch draw either file makes is captured with its actual
    args AND its actual returned tensors. The two files' draws are then
    compared BY VALUE, not by asserting `batch_fn is batch_fn` (already done
    in the identity test above) and not by reading source text.

    `r10_capacity_sweep.py` never constructs `arm_pl` or `arm_smprime`
    (`Arm(str)` only -- see that file's own docstring), so the comparison is
    scoped to what the two files DO share: the batch construction and the
    control arm (`softmax`), which is the only arm both files can run. That
    scoping is the honest reading of 'batch path', not a narrowing to dodge
    the check -- the arms are new constructions and were never claimed to be
    on a shared path.
    """
    from scripts import v15_r1 as v15
    from scale import r10_capacity_sweep as r10

    task = f"e3_t{v15.T_STAR}"
    assert task == "e3_t2"
    real_fn, oracle_fn, feature_fn, fd_fn = negation_scope.M3_TASKS[task]

    calls = []

    def spy(*a, **kw):
        x, y, f, p = real_fn(*a, **kw)
        calls.append(dict(args=a, kwargs=dict(kw), x=x, y=y))
        return x, y, f, p

    monkeypatch.setitem(negation_scope.M3_TASKS, task,
                        (spy, oracle_fn, feature_fn, fd_fn))

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    monkeypatch.setattr(v15, "ROOT", tmp_path)
    monkeypatch.setattr(r10, "ROOT", tmp_path)

    monkeypatch.setattr(sys, "argv", [
        "v15_r1.py", "--arms", "softmax", "--seeds", "0", "1",
        "--n-train", "64", "--n-eval", "4096", "--steps", "0",
        "--threads", "2", "--tag", "g14_lscope_v15",
    ])
    assert v15.main() == 0
    v15_calls = list(calls)
    calls.clear()

    monkeypatch.setattr(sys, "argv", [
        "r10_capacity_sweep.py", "--t-star", "2", "--n-train", "64",
        "--max-steps", "150", "--seeds", "0", "--threads", "2",
        "--tag", "g14_lscope_r10",
    ])
    assert r10.main() == 0
    r10_calls = list(calls)
    calls.clear()

    def find(rows, seed, n):
        #: keyed on (n, seed), not seed alone: `calibrate_bar` draws its OWN
        #: internal batch at `n=a.n_eval` with ITS default `seed=0`
        #: (`scale/negation_scope.py::calibrate_bar`, no `seed=` threaded from
        #: the cell), which collides with a genuine `seed=0` TRAIN draw on
        #: `seed` alone -- `n` (4096 for the calibration/eval draw, 64 for the
        #: train draw here) is what tells them apart.
        hits = [r for r in rows
                if r["kwargs"].get("seed") == seed and r["args"][0] == n]
        assert len(hits) == 1, (
            f"expected exactly one draw at (n={n}, seed={seed}), got {len(hits)}: "
            f"{[(r['args'], r['kwargs']) for r in rows if r['kwargs'].get('seed') == seed]}"
        )
        return hits[0]

    v15_eval, r10_eval = find(v15_calls, 12345, 4096), find(r10_calls, 12345, 4096)
    v15_train, r10_train = find(v15_calls, 0, 64), find(r10_calls, 0, 64)

    for name, a, b in (("eval", v15_eval, r10_eval), ("train", v15_train, r10_train)):
        assert a["args"] == b["args"], f"{name} draw: positional (n, s, d) differ"
        assert a["kwargs"]["d_model"] == b["kwargs"]["d_model"], f"{name}: d_model differs"
        assert a["kwargs"]["seed"] == b["kwargs"]["seed"], f"{name}: seed differs"
        assert torch.equal(a["x"], b["x"]), f"{name} draw: x tensors differ by value"
        assert torch.equal(a["y"], b["y"]), f"{name} draw: y (label) tensors differ by value"

    # DEVICE HANDLING: named explicitly rather than folded into the loop
    # above, because it is the one field that differs in FORM without
    # differing in EFFECT, and collapsing that distinction would be the
    # 'names match' shortcut RULING 5 forbids.
    assert v15_eval["kwargs"]["device"] == "cpu"
    assert r10_eval["kwargs"]["device"] is None
    #: `scale/negation_scope.py::make_batch`: `dev = device or torch.device("cpu")`
    #: -- a truthy string and a bare `None` resolve to the same `.to(...)`
    #: target on cpu, which is exactly why the batch tensors above compared
    #: equal despite this literal difference. Proved, not assumed: redraw
    #: through both device spellings directly and require bitwise agreement.
    fn = negation_scope.M3_TASKS[task][0]
    xa, ya, _, _ = fn(8, v15.S, v15.D, d_model=v15.D_MODEL, seed=99, device="cpu")
    xb, yb, _, _ = fn(8, v15.S, v15.D, d_model=v15.D_MODEL, seed=99, device=None)
    assert torch.equal(xa, xb) and torch.equal(ya, yb), (
        "device='cpu' and device=None must draw the identical tensor on a "
        "cpu box, or the two files' device handling is NOT the same path "
        "despite the args above matching"
    )


# ======================================================= 3. THE CITATION


def test_cell_records_carry_the_instrument_hash(tmp_path, monkeypatch):
    """'Wired where cells are written, not as a manual step.' Runs the real
    `main()` at a throwaway shape and reads the journal it wrote back off
    disk -- the citation has to survive a JSON round trip, not merely exist
    as a Python attribute while the run is live."""
    import json
    from scripts import v15_r1 as v15

    (tmp_path / "results").mkdir()
    monkeypatch.setattr(v15, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "v15_r1.py", "--arms", "softmax", "--seeds", "0", "1",
        "--n-train", "64", "--n-eval", "512", "--steps", "0",
        "--threads", "2", "--tag", "g14_citation",
    ])
    assert v15.main() == 0

    rows = [json.loads(line) for line in
            (tmp_path / "results" / "g14_citation.jsonl").read_text().splitlines()]
    cells = [r for r in rows if r.get("t") == "cell"]
    headers = [r for r in rows if r.get("t") == "header"]
    assert cells and headers

    expected = v15.INSTRUMENT_MANIFEST["hash"]
    for c in cells:
        assert c.get("instrument_hash") == expected, (
            f"cell {c.get('cell')!r} does not carry the instrument digest"
        )
    for h in headers:
        assert h.get("instrument_hash") == expected

    # NON-DEGENERACY: the digest actually depends on the file it was computed
    # from -- an empty-reach manifest of a DIFFERENT file must not collide.
    other = identity_manifest.instrument_manifest(__file__, reaches=())["hash"]
    assert other != expected
