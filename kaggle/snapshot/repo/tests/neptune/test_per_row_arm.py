"""The per-row cells: the bind that must hold, and the RED that must move bits.

`ROW_CELLS` writes the pivot reading at every query row instead of at row `s-1`
alone. Two claims carry the lane and both are tested here against drawn data:

  BIND. Under the shipped scalar readout the output depends only on `z[:, s-1]`
  (the MLP and the readout are position-wise), so a per-row cell must be
  **bitwise** the shipped cell. If it is not, the extra rows are perturbing the
  one row the published readings were taken at.

  RED. Under `vector_readout` the per-row cells must differ from `softmax` at
  positions other than `s-1`, and the shipped cells must NOT. A bind that only
  ever saw the correct path could not tell the two apart -- `scale/arm_s.py`
  states the requirement and this is its instance.

The RED is also the measurement that scores `R9_IRENE_PREDICTION.md` falsifier
5, which asks for "a demonstration that the pivot term reaches a position other
than `s-1` in the R9 arm".
"""
from __future__ import annotations

import pytest
import torch

from scale import eprocess as EP
from scale import m3_flops as F
from scale import m3_quintuple as MQ
from scale import negation_scope as NS

S, D, DM, N = 16, 8, 16, 256
K = 8


def _batch(n=N, s=S):
    x, _y, _f, _p = NS.make_batch(n, s, D, d_model=DM, seed=0)
    return x


def _run(cell, x, *, vector=False, s=S):
    torch.manual_seed(0)
    m = MQ.QuintArm("softmax", s, cell=cell, k_piv=K, vector_readout=vector)
    m.eval()
    with torch.no_grad():
        return m(x)


def test_cells_tuple_is_untouched_and_row_cells_sit_beside_it():
    assert MQ.CELLS == ("softmax", "glance", "settled", "twin", "argmax")
    assert set(MQ.ROW_CELLS).isdisjoint(MQ.CELLS)
    assert set(MQ.ROW_CELLS) <= set(MQ.ALL_CELLS)


def test_row_cell_names_survive_the_journal_key_parsers():
    """The suffix-collision trap, checked rather than asserted in a comment.

    `eprocess._parse_key` reads the cell as everything before the FIRST
    underscore, so a cell name carrying its own underscore is misparsed. The
    `_plus` cells already fail this; the row cells are named without one so they
    cannot, and that is the whole reason for the ugly spelling.
    """
    for cell in MQ.ROW_CELLS:
        key = MQ._key(dict(cell=cell, k=K, s=S, d=D, steps=1, n_train=8,
                           n_eval=8, t_max=21, seed=3))
        assert EP._parse_key(key) == (cell, "s%d_d%d_st1_ntr8_nev8_b21" % (S, D), 3)
        assert MQ.task_of(key) == MQ.SHIPPED_TASK
        for bad in ("_sd", "_task", "_k", "_b"):
            assert bad not in cell


def test_parameter_count_is_4769_on_every_cell_and_both_readouts():
    for cell in MQ.ALL_CELLS:
        for vec in (False, True):
            m = MQ.QuintArm("softmax", S, cell=cell, k_piv=K,
                            vector_readout=vec)
            assert sum(p.numel() for p in m.parameters()) == 4769, (cell, vec)
            assert not list(m.buffers()), cell


def test_bind_per_row_is_bitwise_the_shipped_cell_under_a_scalar_readout():
    x = _batch()
    for row, base in (("twinrow", "twin"), ("settledrow", "settled")):
        got, want = _run(row, x), _run(base, x)
        assert torch.equal(got, want), (
            row, (got - want).abs().max().item())


def test_red_only_the_per_row_cells_move_bits_away_from_s_minus_1():
    """Irene falsifier 5. Shipped cells: identical to softmax off `s-1`."""
    x = _batch()
    ref = _run("softmax", x, vector=True)
    assert ref.shape == (N, S)

    for cell in ("twin", "settled"):
        got = _run(cell, x, vector=True)
        assert torch.equal(got[:, :S - 1], ref[:, :S - 1]), cell
        assert not torch.equal(got[:, S - 1], ref[:, S - 1]), cell

    for cell in MQ.ROW_CELLS:
        got = _run(cell, x, vector=True)
        assert not torch.equal(got[:, :S - 1], ref[:, :S - 1]), cell
        moved = (got[:, :S - 1] - ref[:, :S - 1]).abs().max(dim=0).values > 0
        # A head row must move exactly when some example can write it. Rows 0
        # and 1 are pivotless for every example and must stay put; every row
        # that any example can write must move, or the per-row path is not
        # reaching the rows it claims.
        torch.manual_seed(0)
        arm = MQ.QuintArm("softmax", S, cell=cell, k_piv=K)
        with torch.no_grad():
            _, valid = MQ.batched_row_gates(
                arm.wq(x), arm.wk(x), MQ.batched_pivots(arm.wk(x), K))
        assert torch.equal(moved, valid[:, :S - 1].any(dim=0)), cell
        assert not bool(moved[0]) and not bool(moved[1]), cell


def test_the_pivotless_early_rows_are_reported_and_not_written():
    """The all-masked-row guard: `log_softmax` over an all -inf row is `nan`."""
    x = _batch()
    torch.manual_seed(0)
    m = MQ.QuintArm("softmax", S, cell="twinrow", k_piv=K)
    with torch.no_grad():
        q, k = m.wq(x), m.wk(x)
        piv = MQ.batched_pivots(k, K)
        gate, valid = MQ.batched_row_gates(q, k, piv)
    assert gate.shape == (N, S, K) and valid.shape == (N, S)
    # Query row p reads j <= p-1, so row p is usable exactly when the smallest
    # SELECTED pivot is causally visible to it. Validity is therefore per
    # example, not a fixed prefix: it is checked against that relation computed
    # independently of the mask, not against a guessed row count.
    rows = torch.arange(S).unsqueeze(0)
    assert torch.equal(valid, piv.min(dim=1).values.unsqueeze(1) <= rows - 1)
    # Rows 0 and 1 are structurally pivotless for every example: pivot 0 is
    # excluded, so no pivot can be <= 0.
    assert not bool(valid[:, :2].any())
    # The read position is always usable, or the bind above could not hold.
    assert bool(valid[:, S - 1].all())
    assert bool(torch.isnan(gate[:, 0]).all()), "row 0 should be the nan case"
    # and the nan must not survive into the arm's output
    assert bool(torch.isfinite(_run("twinrow", x, vector=True)).all())


def test_gradients_are_finite_through_the_per_row_path():
    x = _batch(n=8)
    torch.manual_seed(0)
    m = MQ.QuintArm("softmax", S, cell="settledrow", k_piv=K,
                    vector_readout=True)
    m(x).square().mean().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name


def test_flops_price_the_row_cells_above_their_single_row_cell():
    n, s, dm, h, t, nn = 2048, S, DM, 128, 21, 21
    for row, base in (("twinrow", "twin"), ("settledrow", "settled")):
        hi = sum(F.cell_terms(row, n, s, dm, h, K, t, nn).values())
        lo = sum(F.cell_terms(base, n, s, dm, h, K, t, nn).values())
        assert hi > lo, (row, hi, lo)


C1 = "c1_propagate_t2"
_UNIT = dict(k=8, seed=0, s=S, d=D, steps=0, n_train=128, n_eval=128,
             t_max=21, n_neumann=21)


def test_a_c1_vector_unit_runs_end_to_end_at_the_shipped_parameter_count():
    """Saturn's `[n, s - t*]` label against this arm, through `_unit`."""
    for cell in ("softmax", "twinrow"):
        v = MQ._unit(dict(_UNIT, cell=cell, k=0 if cell == "softmax" else K,
                          task=C1))
        assert v["n_params"] == 4769
        assert v["eval_nrmse"] == v["eval_nrmse"], "nan eval_nrmse"
        assert v["nrmse0_eval"] == v["nrmse0_eval"], "nan nrmse0_eval"


def test_the_label_support_guard_raises_when_the_width_and_the_dial_disagree():
    """Both branches, or the guard is decoration.

    A slice that merely happens to line up reads as working until `t*` changes,
    so the offset implied by the label's width is checked against the task's
    declared dial. The honest task must pass the same guard that the doctored
    one trips, otherwise the check is testing nothing.
    """
    real = NS.M3_TASKS[C1][0]

    def narrow(n, s, d, **kw):
        x, y, f, p = real(n, s, d, **kw)
        return x, y[:, 1:], f, p

    NS.M3_TASKS["c1_neptune_liar"] = (narrow,) + tuple(NS.M3_TASKS[C1][1:])
    NS.E_T_STAR["c1_neptune_liar"] = lambda s: 2
    try:
        with pytest.raises(ValueError, match="declares"):
            MQ._unit(dict(_UNIT, cell="softmax", k=0, task="c1_neptune_liar"))
        MQ._unit(dict(_UNIT, cell="softmax", k=0, task=C1))   # must NOT raise
    finally:
        NS.M3_TASKS.pop("c1_neptune_liar", None)
        NS.E_T_STAR.pop("c1_neptune_liar", None)


def test_the_per_row_arm_covers_most_of_the_label_support():
    """The dilution residue, measured rather than assumed.

    The shipped arm can write exactly ONE position of the label's support, so
    its share is `1 / (s - t*)`. The per-row arm writes every position with a
    causally visible pivot. Both numbers belong beside any margin this lane
    ever reports, which is the obligation `R9_IRENE_PREDICTION.md` row omega
    creates.
    """
    s = 64
    torch.manual_seed(0)
    m = MQ.QuintArm("softmax", s, cell="twinrow", k_piv=K)
    for t, floor in ((2, 0.85), (8, 0.90), (32, 0.99)):
        x, y, _f, _p = NS.M3_TASKS["c1_propagate_t%d" % t][0](
            256, s, 24, d_model=DM, seed=0)
        assert y.shape[1] == s - t
        with torch.no_grad():
            q, k = m.wq(x), m.wk(x)
            _, valid = MQ.batched_row_gates(q, k, MQ.batched_pivots(k, K))
        frac = float(valid[:, t:].float().mean())
        assert frac >= floor, (t, frac)
        # and it must beat the single-row arm's share, which is the whole point
        assert frac > 1.0 / (s - t)


def test_an_unregistered_cell_still_dies_at_accounting_time():
    try:
        F.cell_terms("notacell", 8, S, DM, 128, K, 21, 21)
    except ValueError:
        return
    raise AssertionError("cell_terms accepted an unregistered cell")
