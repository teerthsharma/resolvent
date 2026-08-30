"""A1: `impact_hetero` must draw a different corpus from `impact`.

WHAT THIS BINDS. `M3_TASKS` is a plain dict literal and nothing validates an
entry on import (FINDINGS F). Two of its entries -- `impact` and
`impact_hetero` -- were written to name two different plants: the homogeneous
MP-cleaned graph and the two-block degree-heterogeneous one. Both entries bound
`make_impact_batch`, so a heterogeneous-plant arm drew the homogeneous corpus
and could not differ from its own baseline by construction. It is the LIVE
vacuous control in this repository (MISTAKES.md, class V-1). No ordinal is
claimed for it: the struck count on record is fourteen (`STATE.md:53`) and two
separate later items were each already labelled "the fifteenth"
(`DONE.md:1228`, `PIVOT_EXCLUSION_FALSIFIER.md:50-52`).

THE CLAIM, as a property. For every admissible draw size and seed, the batch
the registry returns for `impact_hetero` differs from the batch it returns for
`impact` -- in the tensor, in the label, and in the query node the plant
selects -- and the heterogeneous label is non-degenerate on its own terms.

THE NON-DEGENERACY HALF. The fourteenth strike was a PASS case whose label was
constant (`SupercriticalDense_S2Rips_256`, one component, sd 0.0, NRMSE nan).
"The two labels differ" is worthless if one of them is constant, so the count of
discriminating instances is asserted AND reported, and both labels are required
to have `sd > 0`.

THE ORACLE HALF. Binding the heterogeneous builder while leaving
`impact_oracle` rebuilding `build_impact_graph(..., heterogeneous=False)` would
trade a vacuous control for a mislabelled one: `calibrate_bar`'s clause 3
(`nrmse(oracle(x,f,p), y) < 1e-6`) would read the homogeneous kernel against the
heterogeneous label. The registry entry is only repaired when the oracle it
carries reproduces the label its builder plants, so that is asserted here too.

ADMISSION. Both builders enforce `s >= IMPACT_MIN_NODES` (1024) --
`scale/impact.py:596` and `:633` -- so every draw below is at s=1024. The graph
build is the cost (about 2.7 s each on this machine), so the draws are module
scoped and shared.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import impact as IMP                               # noqa: E402
from scale import negation_scope as NS                        # noqa: E402

S_NODES = IMP.IMPACT_MIN_NODES      # 1024, the admission floor, read not copied
N_DRAWS = 8
D_HOUSE = 24                        # the house distance default
SEED = 0


def _entry(name):
    """The registry entry, read through the registry -- not the module. The
    defect was in the binding, so a test that imports the builders directly
    cannot see it."""
    assert name in NS.M3_TASKS, f"{name} is not registered"
    return NS.M3_TASKS[name]


@pytest.fixture(scope="module")
def drawn():
    """One draw per registered key, at the admission floor, same seed."""
    out = {}
    for name in ("impact", "impact_hetero"):
        batch_fn, oracle_fn, feature_fn, _fd = _entry(name)
        x, y, f, p = batch_fn(N_DRAWS, S_NODES, D_HOUSE,
                              d_model=IMP.D_MODEL, seed=SEED)
        out[name] = dict(x=x, y=y, f=f, p=p,
                         oracle_fn=oracle_fn, feature_fn=feature_fn)
    return out


def test_the_two_registered_keys_do_not_bind_the_same_builder():
    """The binding itself, before any tensor is drawn. This is the cheap half
    and it names the defect exactly: two dict values that are the same function
    object cannot produce two corpora."""
    homo_fn = _entry("impact")[0]
    het_fn = _entry("impact_hetero")[0]
    assert homo_fn is not het_fn, (
        "impact and impact_hetero bind the SAME batch builder "
        f"({getattr(homo_fn, '__name__', homo_fn)}) -- the heterogeneous arm "
        "draws the homogeneous corpus and is its own baseline"
    )
    assert het_fn is IMP.make_impact_hetero_batch, (
        f"impact_hetero binds {getattr(het_fn, '__name__', het_fn)}, "
        "not make_impact_hetero_batch"
    )


def test_the_drawn_tensors_differ_on_a_counted_number_of_instances(drawn):
    """The expensive half: the corpora are drawn, not reasoned about.

    Reported: the count of instances on which the label differs, the max
    absolute tensor gap, and both label sds. A control that cannot say how many
    draws it discriminated on has not measured anything."""
    homo, het = drawn["impact"], drawn["impact_hetero"]

    assert homo["x"].shape == het["x"].shape == (N_DRAWS, S_NODES, IMP.D_MODEL)

    gap = (homo["x"] - het["x"]).abs()
    label_differs = int((homo["y"] != het["y"]).sum())
    sd_homo = float(homo["y"].std(unbiased=False))
    sd_het = float(het["y"].std(unbiased=False))

    print(f"\n  draws={N_DRAWS} s={S_NODES} seed={SEED}")
    print(f"  label differs on {label_differs}/{N_DRAWS} drawn instances")
    print(f"  max |x_homo - x_het| = {float(gap.max()):.6f}")
    print(f"  query node: impact f={homo['f']}  impact_hetero f={het['f']}")
    print(f"  label sd: impact {sd_homo:.6f}  impact_hetero {sd_het:.6f}")

    assert not torch.equal(homo["x"], het["x"]), (
        "the two registered keys draw a BYTE-IDENTICAL tensor"
    )
    assert label_differs == N_DRAWS, (
        f"the labels agree on {N_DRAWS - label_differs} of {N_DRAWS} draws"
    )


def test_neither_label_is_constant(drawn):
    """The non-degeneracy half. `sd == 0` makes NRMSE nan and every downstream
    reading uninterpretable -- that is how the fourteenth control was struck."""
    for name in ("impact", "impact_hetero"):
        y = drawn[name]["y"]
        sd = float(y.std(unbiased=False))
        assert sd > 0.0, f"{name} label is CONSTANT (sd={sd}) -- NRMSE is nan"
        assert torch.isfinite(y).all(), f"{name} label has non-finite entries"


def test_the_registered_oracle_reproduces_the_label_it_is_registered_beside(drawn):
    """`calibrate_bar` clause 3 requires `nrmse(oracle(x,f,p), y) < 1e-6`. An
    oracle that rebuilds the wrong plant fails it, so a repair that binds the
    heterogeneous builder without teaching the oracle which plant it is reading
    ships a mislabelled task in place of a vacuous one."""
    for name in ("impact", "impact_hetero"):
        d = drawn[name]
        got = d["oracle_fn"](d["x"], d["f"], d["p"])
        err = NS.nrmse(got, d["y"])
        print(f"  {name}: nrmse(oracle, y) = {err:.3e}")
        assert err < 1e-6, (
            f"{name}: registered oracle does not reproduce the registered "
            f"builder's label (NRMSE {err:.6f}) -- the entry is mislabelled"
        )


def test_the_registered_features_are_not_the_same_numbers_for_both_plants(drawn):
    """The feature hook is the decoder gate's FAIL probe. If it rebuilds the
    homogeneous graph for both keys, the heterogeneous gate is measuring the
    homogeneous plant's local structure and the gate is vacuous for the same
    reason the builder was."""
    homo, het = drawn["impact"], drawn["impact_hetero"]
    f_homo = homo["feature_fn"](homo["x"], homo["f"], homo["p"])
    f_het = het["feature_fn"](het["x"], het["f"], het["p"])
    rows_differ = int((f_homo != f_het).any(dim=-1).sum())
    print(f"  features differ on {rows_differ}/{N_DRAWS} rows")
    assert rows_differ == N_DRAWS, (
        f"the two plants produce identical local features on "
        f"{N_DRAWS - rows_differ} of {N_DRAWS} rows"
    )


def test_the_plant_bit_is_load_bearing_not_decoration(drawn):
    """THE REPAIR MUST BE SHOWN TO CHANGE THE OBJECT IT REPAIRS.

    `exclude=(0,)` writing `-inf` into a top-k that selected `s-1` in 0 of 32
    draws was shipped as the repair for a vacuous-control problem and was
    itself vacuous. The guard against repeating that here is to delete the
    repair on a live tensor and watch the number move: erase CH_HET and the
    oracle rebuilds the homogeneous plant, which is exactly the pre-repair
    code path. If the reading does not move, CH_HET is decoration and the
    rebinding shipped a mislabelled task."""
    d = drawn["impact_hetero"]
    honoured = NS.nrmse(d["oracle_fn"](d["x"], d["f"], d["p"]), d["y"])
    erased_x = d["x"].clone()
    erased_x[:, :, IMP.CH_HET] = 0.0
    erased = NS.nrmse(d["oracle_fn"](erased_x, d["f"], d["p"]), d["y"])
    print(f"  oracle with plant bit honoured: {honoured:.3e}")
    print(f"  oracle with plant bit erased  : {erased:.6f}")
    assert honoured < 1e-6
    assert erased > 1.0, (
        f"erasing CH_HET leaves the oracle at NRMSE {erased:.6f} -- the plant "
        "bit is not doing any work and the two plants are not distinguishable "
        "from the tensor"
    )
