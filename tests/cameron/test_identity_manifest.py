"""X-R1 -- MEASURED-OBJECT != SHIPPED-OBJECT. The identity manifest.

THE DEFECT, in this repository's own instances. `tgate` measured under `sgate`'s
name; a ParaFormer arm silently running softmax; `impact_hetero` registering
`make_impact_batch` so a heterogeneous-plant arm drew the homogeneous corpus;
the published intervals splitting into two families because the estimator was
dropped at render time. Every one is the same shape: **a number was measured
under one object and reported under another's name**, and nothing in the tree
could tell.

WHAT THE JOURNAL KEY ALREADY CARRIES, AND WHAT IT DOES NOT.
`m3_quintuple._key` encodes `cell, k, s, d, steps, n_train, n_eval, t_max, seed,
task`. That is the config half and it is already load-bearing -- three consumers
parse it. It does NOT encode `beta`, `n_neumann`, `d_model` or `reserve_query`,
all four of which `load_unit` names as the arguments "that make a `QuintArm` the
cell it is". Two runs differing only in `n_neumann` therefore land on the SAME
key, and the second silently overwrites the first's number. That is not a
hypothetical: `test_the_key_cannot_separate_two_different_arms` draws it.

Nor does the key carry the CODE PATH. `beta` is a module constant (`BETA = 0.5`)
rather than a per-unit argument, so changing it moves every published cell with
no change to any key, any record field, or any file the reader can see.

THE FOUR COMPONENTS, one per thing the contract names.

  config  the declared scalars, including the four the key drops
  code    module:qualname + BYTECODE digest of each callable the arm dispatches
          to -- bytecode, not source, so a docstring fix does not invalidate a
          published cell while a changed branch does
  shapes  the parameter shapes in construction order
  rng     the seeding plan AND the untrained parameter values it produces --
          the plan alone is a claim, the values are the measurement of it

The RNG half is the part with no precedent and X-R6 names why it matters:
`paired_arm.train_and_predict` calls `torch.manual_seed(seed)` and then
constructs the arm, so initialisation draws from the GLOBAL generator. Anything
that consumes from that generator in between -- or any change to the order in
which parameters are created -- moves every published number with no change to
config, key or code. Hashing the untrained tensors is what makes that visible,
because it is the only artifact that records what the generator actually did.

G2: NO FIX MAY MOVE A PUBLISHED NUMBER. The manifest is a pure read. The
bitwise-reproduction half of this file is the proof, and it is the half that
matters most.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import identity_manifest as IM                        # noqa: E402

#: A PUBLISHED cell: the e3_t1 ladder rung whose reading is on record.
UNIT = ("results/m3_quintuple_v2_weights/"
        "settled_k8_s64_d24_st150_ntr2048_nev2048_b21_sd0_taske3_t1.pt")
JOURNAL = "results/m3_quintuple_v2.jsonl"


@pytest.fixture(scope="module")
def record():
    return torch.load(ROOT / UNIT, weights_only=True)


@pytest.fixture(scope="module")
def journal_nrmse(record):
    """The journalled number for this key, from the COMMITTED blob -- the
    working file is being appended to by another agent's run."""
    raw = subprocess.run(["git", "show", f"HEAD:{JOURNAL}"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    for line in raw.splitlines():
        if line.strip():
            r = json.loads(line)
            if r["key"] == record["key"]:
                return r["value"]["eval_nrmse"]
    pytest.skip(f"{record['key']} is not in the committed journal")


def _man(rec, params=None, callables=None):
    return IM.manifest(
        rec,
        callables=[IM.manifest] if callables is None else callables,
        params=rec["state_dict"] if params is None else params)


# ------------------------------------------------- G2: the number cannot move
def test_the_manifest_is_a_pure_read_and_moves_no_published_number(record, journal_nrmse):
    """THE HALF THAT MATTERS MOST. Computing an identity must not touch the
    thing it identifies."""
    before = record["eval_nrmse"]
    before_sd = {k: v.clone() for k, v in record["state_dict"].items()}
    m = _man(record)
    assert record["eval_nrmse"] == before                    # bitwise
    assert record["eval_nrmse"] == journal_nrmse, (record["eval_nrmse"], journal_nrmse)
    for k, v in record["state_dict"].items():
        assert torch.equal(v, before_sd[k]), k
    print(f"\n  {record['key']}")
    print(f"    eval_nrmse {record['eval_nrmse']!r} unchanged, journal agrees")
    print(f"    manifest {m['hash'][:16]}")


def test_the_manifest_is_deterministic_and_not_a_constant(record):
    """A hash that never moves refuses nothing; a hash that moves on its own
    refuses everything. Both halves asserted."""
    a, b = _man(record), _man(record)
    assert a["hash"] == b["hash"]
    other = dict(record)
    other["cell"] = "twin"
    assert _man(other)["hash"] != a["hash"], "the hash is constant"


# --------------------------------------------- the must-fire: mutate one byte
@pytest.mark.parametrize("field,new", [
    ("beta", 0.5000001),          # the module constant the key never sees
    ("n_neumann", 20),            # named by load_unit, absent from the key
    ("d_model", 32),              # absent from the key
    ("cell", "twin"),             # tgate-under-sgate's-name, the headline case
    ("task", "e3_t2"),            # a different corpus under the same arm
])
def test_mutating_one_config_field_makes_the_refusal_fire(record, field, new):
    """RED-first, must-fire seen: mutate one field, the refusal must fire and
    must NAME the component that moved. A bare "hash differs" is not actionable
    and would be a refusal nobody can act on."""
    stored = _man(record)
    moved = dict(record)
    moved[field] = new
    with pytest.raises(IM.IdentityMismatch) as e:
        IM.refuse_if_changed(stored, _man(moved))
    msg = str(e.value)
    print(f"  {field}: {msg.splitlines()[0][:100]}")
    assert "config" in msg, msg
    assert field in msg, msg


def test_restoring_the_field_reproduces_the_manifest_bitwise(record):
    """The other half of the must-fire. Mutate, watch it refuse, restore, and
    the SAME hash comes back -- otherwise the refusal is noise rather than a
    measurement."""
    stored = _man(record)
    moved = dict(record)
    moved["beta"] = 0.9
    with pytest.raises(IM.IdentityMismatch):
        IM.refuse_if_changed(stored, _man(moved))
    moved["beta"] = record["beta"]
    IM.refuse_if_changed(stored, _man(moved))                 # must not raise
    assert _man(moved)["hash"] == stored["hash"]


# ------------------------------------- what the key cannot do and this can
def test_the_key_cannot_separate_two_different_arms(record):
    """THE REASON THIS MODULE EXISTS, drawn rather than argued.

    Two records differing in `beta`, `n_neumann` and `d_model` -- three of the
    five arguments `load_unit` calls identity-defining -- produce the SAME
    journal key, so the second overwrites the first's number and no consumer can
    tell. The manifest separates them."""
    from scale import m3_quintuple as Q
    p = dict(cell=record["cell"], k=record["k_piv"], s=record["s"],
             d=record["d"], steps=record["steps"], n_train=record["n_train"],
             n_eval=record["n_eval"], t_max=record["t_max"],
             seed=record["seed"], task=record["task"])
    other = dict(record, beta=0.9, n_neumann=3, d_model=32)
    assert Q._key(p) == record["key"], "the key recipe drifted"
    print(f"\n  one key, two arms: {Q._key(p)}")
    print(f"    beta {record['beta']} / {other['beta']}, "
          f"n_neumann {record['n_neumann']} / {other['n_neumann']}, "
          f"d_model {record['d_model']} / {other['d_model']}")
    assert _man(record)["hash"] != _man(other)["hash"], (
        "the manifest cannot separate two arms the key already conflates")


# ----------------------------------------------------- the code-path component
def test_a_different_callable_under_the_same_name_moves_the_code_digest(record):
    """`tgate` measured under `sgate`'s name, reduced to its mechanism. Bytecode
    is hashed rather than source, so the digest is stable under a comment or
    docstring edit and moves on a changed branch."""
    #: All three are compiled under the SAME name into the same module, so
    #: `module:qualname` is identical and only the body varies. Defining them as
    #: three differently-named `def`s would have compared the names instead of
    #: the logic and proved nothing -- which is what the first draft of this
    #: test did, and it failed for that reason rather than for a real one.
    def build(body):
        g = {"__name__": "identity_probe"}
        exec(f"def gate(x):\n{body}\n", g)                      # noqa: S102
        return g["gate"]

    sgate = build("    return x + 1")
    tgate = build("    return x - 1")
    sgate_recommented = build('    "a docstring that did not exist before"\n'
                              "    return x + 1")

    assert sgate.__qualname__ == tgate.__qualname__ == sgate_recommented.__qualname__
    a = _man(record, callables=[sgate])
    b = _man(record, callables=[tgate])
    c = _man(record, callables=[sgate_recommented])
    print(f"  sgate {a['code'][:12]}  tgate {b['code'][:12]}  "
          f"re-commented {c['code'][:12]}")
    assert a["code"] != b["code"], "a changed implementation did not move the digest"
    assert a["code"] == c["code"], (
        "a docstring edit moved the digest -- this would invalidate every "
        "published cell on a comment fix, which is a refusal nobody can obey")


# ----------------------------------------------------------- the RNG component
def test_a_different_rng_plan_moves_the_rng_digest_with_config_fixed(record):
    """X-R6: `paired_arm` seeds the GLOBAL generator and then constructs the
    arm, so a change in what the generator does between those two points moves
    every published number with no change to config, key or code. Same config,
    same code, different untrained tensors -- the digest must move."""
    same_shapes = {k: torch.zeros_like(v) for k, v in record["state_dict"].items()}
    a = _man(record)
    b = _man(record, params=same_shapes)
    assert a["shapes"] == b["shapes"], "shapes should be identical here"
    assert a["config"] == b["config"] and a["code"] == b["code"]
    assert a["rng"] != b["rng"], (
        "identical shapes and config with different initial tensors produced "
        "the same rng digest -- the seeding plan is not being measured")
    assert a["hash"] != b["hash"]
    print(f"  rng {a['rng'][:12]} vs {b['rng'][:12]} at identical shapes")


def test_the_refusal_names_every_component_that_moved(record):
    """A refusal that says only "differs" sends the reader back to bisecting.
    When several components move at once, all of them are named."""
    stored = _man(record)
    moved = dict(record, cell="twin")
    cur = _man(moved, params={k: torch.zeros_like(v)
                              for k, v in record["state_dict"].items()})
    with pytest.raises(IM.IdentityMismatch) as e:
        IM.refuse_if_changed(stored, cur)
    msg = str(e.value)
    print(f"\n  {msg.splitlines()[0][:120]}")
    assert "config" in msg and "rng" in msg, msg

def test_it_catches_the_registry_defect_this_repository_actually_shipped(record):
    """THE PLANTED POSITIVE, on a REAL defect rather than a toy.

    `scale/negation_scope.py` registered `impact_hetero` against
    `make_impact_batch` -- the same callable `impact` binds -- so a
    heterogeneous-plant arm drew the homogeneous corpus and was its own baseline
    by construction. Two registry names, one implementation, and no artifact in
    the tree could tell them apart. That is X-R1's headline instance and it is
    the reason this module exists, so the mechanism is checked against it and
    not only against functions written to be caught.

    The pre-fix state is reconstructed by pointing both names at one callable,
    which is exactly what the registry line did."""
    from scale import negation_scope as NS

    homo = NS.M3_TASKS["impact"][0]
    het = NS.M3_TASKS["impact_hetero"][0]

    # AS SHIPPED TODAY: two names, two implementations, two digests.
    a = _man(record, callables=[homo])
    b = _man(record, callables=[het])
    print(f"\n  impact        {getattr(homo, '__name__', homo)}  {a['code'][:12]}")
    print(f"  impact_hetero {getattr(het, '__name__', het)}  {b['code'][:12]}")
    assert a["code"] != b["code"], (
        "the two registered builders share a code digest -- the manifest cannot "
        "tell the plants apart and would not have caught the defect it was "
        "built for")

    # THE DEFECT, reconstructed: the registry line as it stood, both names bound
    # to the homogeneous builder. The digests collapse, which is the reading a
    # stored manifest would have refused.
    as_shipped_before = _man(record, callables=[homo])
    assert as_shipped_before["code"] == a["code"], "reconstruction is not faithful"
    with pytest.raises(IM.IdentityMismatch) as e:
        IM.refuse_if_changed(b, as_shipped_before)
    assert "code" in str(e.value)
    print("  a stored `impact_hetero` manifest REFUSES the homogeneous builder")
