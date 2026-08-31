"""MARS / MORIARTY, R9 iteration 2 -- the bind on the straight-through cell.

WHAT THIS CELL IS FOR. Iteration 1 measured that `argmax`'s alpha carries no
`grad_fn`: `d(alpha)/d(log_gate)` L1 is exactly 0.0 against 52.3193 for `twin`
and 55.1514 for `settled`, on the same batch at the same init. So which pivot
`argmax` reads is never trained, and the published `argmax - softmax =
-0.118456` confounds mixture-against-lookup with trained-against-untrained
selection. `argmaxste` holds the forward fixed and changes only the gradient,
which is the only way to separate the two.

THE BIND IS BITWISE, AND IT SHIPS WITH ITS RED. `torch.equal`, not `allclose`.
Paired with a control that MUST move bits, per `scale/arm_s.py:341-346` -- a
bind that only ever saw the correct path could not tell the two apart.

NO READING IS TAKEN HERE. This file builds the cell, binds it and prices it.
The five-seed ladder belongs to whoever holds the execute seat.
"""
from __future__ import annotations

import pytest
import torch

import scale.m3_flops as FL
import scale.m3_quintuple as Q
import scale.negation_scope as NS

S, D, D_MODEL, K = 64, 24, 16, 8
SHAPES = [(8, 0), (16, 1), (32, 2)]          # (n, seed)


def _arm(cell, seed=0):
    torch.manual_seed(seed)
    return Q.QuintArm("softmax", S, cell=cell, k_piv=K)


def _batch(n, seed):
    return NS.make_batch(n, S, D, d_model=D_MODEL, seed=seed)[0]


# ------------------------------------------------------- 1. the bitwise bind --
@pytest.mark.parametrize("n,seed", SHAPES)
def test_the_ste_forward_is_bitwise_the_argmax_forward(n, seed):
    """Same function, different gradient. If this is `allclose` and not `equal`,
    the cell is a new arm rather than a gradient path, and the contrast against
    `argmax` stops isolating the estimator.
    """
    x = _batch(n, seed)
    a_hard, a_ste = _arm("argmax", seed), _arm("argmaxste", seed)
    with torch.no_grad():
        out_hard, out_ste = a_hard(x), a_ste(x)
    assert torch.equal(out_hard, out_ste), (
        f"max abs diff {float((out_hard - out_ste).abs().max())}")


@pytest.mark.parametrize("n,seed", SHAPES)
def test_the_red_control_moves_bits_so_the_bind_can_tell_them_apart(n, seed):
    """THE RED. `twin` is the same arm with the soft alpha left in place -- the
    one thing the estimator must NOT collapse to. It must fail the bind that
    `argmaxste` passes, on the same instances, or the bind above is reading
    nothing.
    """
    x = _batch(n, seed)
    with torch.no_grad():
        out_hard, out_soft = _arm("argmax", seed)(x), _arm("twin", seed)(x)
    assert not torch.equal(out_hard, out_soft)
    assert float((out_hard - out_soft).abs().max()) > 0.0


def test_the_unparenthesised_estimator_is_not_bitwise_and_is_the_second_red():
    """SECOND RED, on the arithmetic rather than the arm. `hard + soft -
    soft.detach()` rounds `hard + soft` before subtracting and is not bitwise;
    `hard + (soft - soft.detach())` is, because the inner difference is exactly
    `+0.0`. The shipped branch uses the parenthesised form. This draws a gate on
    which the two forms disagree, so the parentheses are asserted by value.
    """
    torch.manual_seed(0)
    log_gate = torch.randn(4096, K, dtype=torch.float64) * 40.0
    soft = (log_gate - torch.logsumexp(log_gate, -1, keepdim=True)).exp()
    hard = torch.zeros_like(log_gate)
    hard.scatter_(1, log_gate.argmax(-1, keepdim=True), 1.0)
    assert torch.equal(hard + (soft - soft.detach()), hard)
    assert not torch.equal(hard + soft - soft.detach(), hard)


# ------------------------------------------------- 2. the gradient it exists for --
def test_the_ste_restores_the_gate_gradient_argmax_drops():
    """The defect and the repair, side by side on one batch at one init.

    `argmax` is asserted at EXACTLY zero rather than small: its alpha has no
    autograd path at all, so the number is not a magnitude but an absence.
    """
    x = _batch(64, 0)
    got = {}
    for cell in ("twin", "settled", "argmax", "argmaxste"):
        arm = _arm(cell)
        q, kk = arm.wq(x), arm.wk(x)
        piv = Q.batched_pivots(kk, K)
        log_gram, log_gate, _av = Q.batched_log_pivot_context(
            q, kk, x, piv, need_gram=(cell == "settled"))
        log_gate = log_gate.detach().requires_grad_(True)
        if cell == "settled":
            alpha = Q.BatchedSettled.apply(
                log_gate, log_gram.detach(), Q.BETA, 21, 21).exp()
        elif cell == "twin":
            alpha = (log_gate
                     - torch.logsumexp(log_gate, -1, keepdim=True)).exp()
        else:
            hard = torch.zeros_like(log_gate)
            hard.scatter_(1, log_gate.argmax(-1, keepdim=True), 1.0)
            if cell == "argmax":
                alpha = hard
            else:
                soft = (log_gate
                        - torch.logsumexp(log_gate, -1, keepdim=True)).exp()
                alpha = hard + (soft - soft.detach())
        if not alpha.requires_grad:
            got[cell] = 0.0
            continue
        torch.manual_seed(1)
        probe = torch.randn_like(alpha)
        got[cell] = float(
            torch.autograd.grad((alpha * probe).sum(), log_gate)[0].abs().sum())

    assert got["argmax"] == 0.0, got
    assert got["argmaxste"] > 1.0, got
    #: The estimator's gradient IS the twin's, by construction -- the whole
    #: point is that the forward changed and the backward did not.
    assert got["argmaxste"] == pytest.approx(got["twin"], rel=1e-12), got


def test_the_repair_moves_the_object_it_repairs():
    """Vacuity rule 6: a repair must be shown to change the thing it repairs.

    THE STATISTIC IS PARAMETER DIVERGENCE, AND THE OBVIOUS ONE IS CONFOUNDED.
    Counting how many examples change their selected pivot does NOT measure
    trained selection here, because `select_pivots` is a top-k over `kk.norm()`
    and `wk` moves for reasons that have nothing to do with the gate: measured
    over 30 Adam steps at lr 1e-2, n=256, the pivot SET changes for 256 of 256
    examples in BOTH cells. Against that churn the argmax slot changed 200/256
    for `argmax` and 183/256 for `argmaxste` -- the estimator moved FEWER, which
    is noise in a statistic that is mostly reading candidate-set reshuffling.

    So the assertion is on the parameters, which nothing else can confound: from
    an identical init on an identical batch, the only difference between the two
    runs is the gradient that reaches `log_gate`. If that gradient did nothing,
    the two parameter vectors would be identical.
    """
    x, y, _f, _p = NS.make_batch(256, S, D, d_model=D_MODEL, seed=0)

    def train(cell, steps=30):
        arm = _arm(cell)
        opt = torch.optim.Adam(arm.parameters(), lr=1e-2)
        first = None
        for _ in range(steps):
            opt.zero_grad()
            loss = ((arm(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
            first = first if first is not None else float(loss)
        theta = torch.cat([p.detach().flatten() for p in arm.parameters()])
        return theta, first

    t_hard, loss0_hard = train("argmax")
    t_ste, loss0_ste = train("argmaxste")

    #: The forward bind again, this time through the training loop: step 0's
    #: loss is taken before any update, so the bitwise-equal forwards must give
    #: bitwise-equal losses.
    assert loss0_hard == loss0_ste

    divergence = float((t_hard - t_ste).norm())
    assert divergence > 0.0, "the extra gradient path changed nothing"
    #: Not a marginal effect: ~32% of the parameter norm at 30 steps.
    assert divergence / float(t_hard.norm()) > 0.05, divergence


# ------------------------------------------------------- 3. price and registry --
def test_the_cell_adds_no_parameter():
    """Hard rule 3. An STE is a gradient path, not a parameter. If this moves,
    the Adam trajectory changes and the paired bootstrap stops being paired.
    """
    for cell in ("softmax", "twin", "settled", "argmax", "argmaxste"):
        assert sum(p.numel() for p in _arm(cell).parameters()) == 4769, cell


def test_the_cell_is_priced_and_does_not_die_at_accounting_time():
    """`scale/m3_flops.py` raises `ValueError` on an unregistered cell. The STE
    is priced as `twin`: it materialises the same normalised gate before taking
    the one-hot, and `alpha @ av` runs as a real contraction rather than
    degenerating to `argmax`'s gather.
    """
    args = dict(n=8192, s=64, dm=16, h=128, k=8, t=21, nn_terms=21)
    spec = FL.cell_terms("argmaxste", **args)
    assert spec == FL.cell_terms("twin", **args)
    #: And it is strictly dearer than the cell whose forward it reproduces.
    assert sum(spec.values()) > sum(FL.cell_terms("argmax", **args).values())


def test_the_registry_holds_and_a_default_run_is_untouched():
    """The name must survive the journal key parsers and must not join `CELLS`.
    `capability_table.read_journal` does `key.partition("_")` and
    `eprocess._parse_key` splits on `_sd`, so an underscore in the cell name
    would silently re-read this cell as `argmax`.
    """
    assert Q.STE_CELLS == ("argmaxste",)
    assert "argmaxste" in Q.ALL_CELLS and "argmaxste" not in Q.CELLS
    fams = (Q.CELLS, Q.PLUS_CELLS, Q.ROW_CELLS, Q.STE_CELLS)
    assert set(Q.ALL_CELLS) == set().union(*map(set, fams))
    assert sum(len(f) for f in fams) == len(set(Q.ALL_CELLS))
    assert Q._argparser().parse_args([]).cells == list(Q.CELLS)
    for bad in ("_sd", "_task", "_k", "_b", "_"):
        assert bad not in "argmaxste"
    #: The key round-trips: the cell survives `partition("_")` intact.
    key = Q._key(dict(cell="argmaxste", k=8, s=64, d=24, steps=150,
                      n_train=8192, n_eval=512, t_max=21, seed=0))
    assert key.partition("_")[0] == "argmaxste", key

    import scale.e_ladder as EL
    assert EL.HOP_BUDGET["argmaxste"] == EL.HOP_BUDGET["argmax"]
