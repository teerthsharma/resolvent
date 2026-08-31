"""ARM PL -- the three binds, the four planted negatives, the diagnostic column.

The arm is `V15_JUPITER2_FORK.md` §8.3, which is `lean/CEQ/V15Fork.lean`'s
`Asink` in float64:

    l_ij = q_ij - C_j + s_j        j <= i,   C = scan(g)
    O_i  = sum_{j<=i} softmax_j(l_i.) * V_j

Three binds, and this file is the whole of what this node certifies.

  BIND 1  (P) at (g, s) = (0, 0): BITWISE standard causal attention, against a
          reference this file did not write.
  BIND 2  (L) at the oracle setting: O_i = y_i, the chain path-product label.
  BIND 3  the general-DAG form: the resolvent (I - A)^-1 on edge gates equals
          the brute-force sum over directed paths.

WHAT BIND 1 DOES AND DOES NOT SAY (MISTAKES.md V-24, closing paragraph, which
is binding here). Its information content is: **the modification enters only
through the key logit, additively, and vanishes at zero.** It is NOT "bitwise
standard attention" unqualified -- that is what §S-M's original clause claimed
and what `CEQ.V15.gate_zero_not_stochastic` refuted. The rejection region is
non-empty (`test_bind1_has_an_occupied_rejection_region` occupies it: any
non-zero head moves the operator) and it is narrower than the refuted original,
because the parity point is "both auxiliary heads zero" rather than the gate's
own identity point.

THE REFERENCE, and why it is not `ceq.bench._softmax_operator` alone.
`bench._causal_mask_pair` is `tril(-1)` -- STRICTLY causal, diagonal excluded,
so row 0 is empty and A_ii = 0. ARM PL needs `P_ii = 1`, i.e. the diagonal.
`ceq.lm.Attention("softmax_x").operator` is the repo's INCLUSIVE-causal control
-- `lm.py`:39 states "SMX_TAU=1, SMX_RHO=1, SMX_HOPS=1, SMX_ID=False is EXACTLY
standard causal attention" -- and it is what the bitwise comparison is against.
`bench._softmax_operator` is used too, at the strict convention, so the bind is
run against TWO independently written references rather than one.

PLANTED NEGATIVES ARE PART OF THE BIND, NOT AN EXTRA (MISTAKES.md rule 8,
V-24). Four deliberate mutilations of the (L) setting, each of which must fail
at O(1). Expected from `V15_JUPITER2_FORK.md` §4: 0.9749, 0.9165, 1, 0.4845.
"""
from __future__ import annotations

import math

import pytest
import torch

from ceq import arm_pl, bench, lm

DT = torch.float64
S = 8                       # positions 0..S; 0 is the BOS sink
SEED = 15                   # the fork probe's draw, so the numbers are comparable


def _qk(n: int = S + 1, d: int = 4, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g, dtype=DT),
            torch.randn(n, d, generator=g, dtype=DT),
            torch.randn(n, 2, generator=g, dtype=DT))


# ------------------------------------------------------------------ BIND 1

def test_bind1_parity_is_bitwise_against_the_inclusive_causal_control():
    """(P): at (g, s) = (0, 0) the arm IS the shipped softmax control, bitwise.

    Content: the modification enters only through the key logit, additively,
    and vanishes at zero.
    """
    q, k, v = _qk()
    zero = torch.zeros(q.shape[0], dtype=DT)
    ref_op = lm.Attention("softmax_x", 4, 1).operator(q, k)
    mine = arm_pl.operator(q, k, zero, zero)
    assert torch.equal(mine, ref_op)
    assert torch.equal(arm_pl.readout(q, k, v, zero, zero), ref_op @ v)


def test_bind1_parity_is_bitwise_against_bench_at_the_strict_convention():
    """The same statement against a SECOND reference written for another node.

    `bench._softmax_operator` excludes the diagonal, so the comparison is at
    `diagonal=-1`. Same claim, different mask convention, different author.
    """
    q, k, _ = _qk()
    zero = torch.zeros(q.shape[0], dtype=DT)
    assert torch.equal(arm_pl.operator(q, k, zero, zero, diagonal=-1),
                       bench._softmax_operator(q, k))


def test_bind1_holds_for_the_drop_in_module_and_not_only_the_function():
    """The same bind on `ArmPL`, the object the harness would actually hold.

    A parity bind on a free function that the shipped module does not dispatch
    through is a reading of a non-shipped operator -- the defect
    `tests/loop/test_m3_harness_operator_is_shipped.py` exists for. At
    `zero_heads()` the module's own operator is bitwise the softmax control.
    """
    arm = arm_pl.ArmPL(s=S + 1, d_model=4, hidden=8).zero_heads()
    x = torch.randn(3, S + 1, 4, generator=torch.Generator().manual_seed(5))
    q, k = arm.wq(x), arm.wk(x)
    g, s = arm.heads(x)
    assert torch.equal(g, torch.zeros_like(g)) and torch.equal(s, torch.zeros_like(s))
    assert torch.equal(arm._operator(q, k, g, s),
                       lm.Attention("softmax_x", 4, 1).operator(q, k))
    assert arm(x).shape == (3,)


def test_bind1_has_an_occupied_rejection_region():
    """A non-zero head MOVES the operator. Without this the bind is a V-24."""
    q, k, _ = _qk()
    zero = torch.zeros(q.shape[0], dtype=DT)
    ref = lm.Attention("softmax_x", 4, 1).operator(q, k)
    gen = torch.Generator().manual_seed(7)
    for name, g, s in (("g only", torch.randn(S + 1, generator=gen, dtype=DT), zero),
                       ("s only", zero, torch.randn(S + 1, generator=gen, dtype=DT))):
        moved = (arm_pl.operator(q, k, g, s) - ref).abs().max().item()
        assert moved > 1e-3, f"{name} did not move the operator: {moved}"


def test_the_family_never_leaves_the_softmax_class():
    """V-24's class-closure test: rows sum to 1 and are strictly positive at a
    RANDOM theta, not only at the parity point."""
    q, k, _ = _qk()
    gen = torch.Generator().manual_seed(11)
    tri = torch.ones(S + 1, S + 1, dtype=torch.bool).tril(0)
    for _ in range(3):
        a = arm_pl.operator(q, k, torch.randn(S + 1, generator=gen, dtype=DT) * 0.7,
                            torch.randn(S + 1, generator=gen, dtype=DT) * 0.7)
        assert (a.sum(-1) - 1.0).abs().max().item() < 1e-15
        assert a[tri].min().item() > 0.0


# ------------------------------------------------------------------ BIND 2

def test_bind2_the_oracle_setting_reproduces_the_chain_label():
    """(L): max |O - y| at the oracle setting. Contract bar is 1e-6; the
    construction is an identity and must land at machine precision."""
    a, b = arm_pl.draw(seed=SEED, s=S)
    cell = arm_pl.label_cell(a, b, seed=SEED)
    assert cell["residual"] <= 1e-6            # the contract's bar
    assert cell["residual"] < 1e-14            # what an identity owes


def test_bind2_the_normalizer_telescopes_to_one():
    """Z_i = 1 EXACTLY is the mechanism (`Asink_row_sum`, no hypothesis on a)."""
    a, b = arm_pl.draw(seed=SEED, s=S)
    g, s, _ = arm_pl.oracle_heads(a, b)
    z = arm_pl.normalizer(g, s)
    assert (z - 1.0).abs().max().item() < 1e-15


# ------------------------------------------------------------------ BIND 3

@pytest.mark.parametrize("lo,hi,density", [(0.05, 0.45, 0.6), (0.30, 0.90, 1.0)])
def test_bind3_the_dag_resolvent_matches_brute_force_path_sums(lo, hi, density):
    """(I - A)^-1 on edge gates against explicit path enumeration.

    Two routes that share no code: an LU solve, which knows nothing about
    paths, and a sum over every increasing vertex sequence. `CEQ_V15_CONTRACT.md`
    records 1.1e-16 `[INHERITED]`; L-TIME requires the citing iteration to
    re-measure it, so this is a measurement, not a quotation.

    Two gate regimes, because an ABSOLUTE residual on a resolvent whose entries
    never leave 1 is cheap. The heavy regime drives them to O(10), and the
    tolerance is relative to the quantity actually being reproduced.
    """
    A = arm_pl.draw_dag(n=9, seed=SEED, density=density, lo=lo, hi=hi)
    r = arm_pl.dag_resolvent(A)
    brute = arm_pl.brute_force_path_sums(A)
    scale = r.abs().max().item()
    assert scale >= 1.0
    assert (r - brute).abs().max().item() / scale < 1e-14


# ------------------------------------------------ THE PLANTED NEGATIVES

@pytest.mark.parametrize("mutation,expected", [
    ("drop_key_bias", 0.9749),
    ("drop_value_rescale", 0.9165),
    ("drop_bos_sink", 1.0),
    ("half_key_bias", 0.4845),
])
def test_every_planted_negative_fires(mutation, expected):
    """Four deliberate mutilations, four failures at O(1). A bind whose
    rejection region is empty is a V-24; this is the evidence it is occupied."""
    a, b = arm_pl.draw(seed=SEED, s=S)
    r = arm_pl.label_cell(a, b, mutation=mutation, seed=SEED)["residual"]
    assert r > 1e-6, f"plant {mutation!r} did not fire: {r}"
    assert abs(r - expected) < 5e-4, f"{mutation}: {r} != published {expected}"


def test_the_mutation_list_is_the_one_the_fork_settled():
    assert arm_pl.MUTATIONS == ("none", "drop_key_bias", "drop_value_rescale",
                                "drop_bos_sink", "half_key_bias")


# --------------------------------------- THE DIAGNOSTIC COLUMN (§8.4 cost 6)

def test_the_dynamic_range_column_is_populated_and_grows_as_a_goes_to_one():
    """§8.4 cost 6, filed as a prediction BEFORE the arm runs: the value head's
    range must grow like 1/(1 - a). The column is `v_max` against
    `1/(1 - a_max)`, per cell, and it is not optional."""
    seen = []
    for a_max in (0.5, 0.9, 0.99):
        a, b = arm_pl.constant_gate_draw(a_max, s=S)
        cell = arm_pl.label_cell(a, b, seed=SEED)
        for col in ("a_max", "v_max", "dyn_range_bound"):
            assert col in cell and math.isfinite(cell[col]), f"{col} not populated"
        assert cell["v_max"] > 0.0
        assert cell["dyn_range_bound"] == pytest.approx(1.0 / (1.0 - a_max))
        seen.append(cell["v_max"])
    assert seen[0] < seen[1] < seen[2], f"column does not grow: {seen}"


# ------------------------------------------------------ THE IDENTITY MANIFEST

def test_the_manifest_records_the_variant_the_heads_the_bos_and_the_column():
    a, b = arm_pl.draw(seed=SEED, s=S)
    m = arm_pl.label_cell(a, b, seed=SEED)["manifest"]
    for f in arm_pl.PL_FIELDS:
        assert f in m["pl_values"], f"manifest does not carry {f}"
    assert m["pl_values"]["variant"] == arm_pl.VARIANT
    assert m["pl_values"]["bos_value"] == 0.0
    assert "device" in m["values"], "device is a declared CONFIG_FIELD"


def test_the_manifest_hash_moves_when_a_planted_negative_moves_the_cell():
    """A manifest that cannot tell the mutilated cell from the honest one is
    not an identity. Both the head tensors and the PL block must move it."""
    a, b = arm_pl.draw(seed=SEED, s=S)
    honest = arm_pl.label_cell(a, b, seed=SEED)["manifest"]["hash"]
    for mutation in arm_pl.MUTATIONS[1:]:
        other = arm_pl.label_cell(a, b, mutation=mutation, seed=SEED)["manifest"]["hash"]
        assert other != honest, f"{mutation} left the manifest hash unmoved"
