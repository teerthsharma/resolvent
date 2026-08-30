"""The pivot-exclusion lift: does `av` actually contain softmax's own row?

THE CONFOUND. `ceq/bench.py:154` builds the causal mask as `tril(-1)`, so row
`i` reads `j <= i-1`. `scale/m3_quintuple.py` selects pivots with
`exclude=(0, s-1)`, so the largest legal pivot is `s-2`, whose row reads
`j <= s-3`. `v[s-2]` is therefore invisible to every pivot row, while softmax's
own row `s-1` reads `j <= s-2` and sees it directly. At `t* = 1` the label is
concentrated on exactly that token.

THE LIFT THAT DOES NOT WORK, KEPT AS A TEST SO IT CANNOT BE RE-PROPOSED.
Merely permitting `s-1` (`exclude=(0,)`) changes nothing: `select_pivots` is a
top-k over the key-norm and `s-1` does not rank. `test_permitting_the_query_row_
is_not_enough` measures that, and it is the reason the shipped lift RESERVES the
slot instead.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ceq import bench                                             # noqa: E402
from scale import m3_capability as M3                             # noqa: E402
from scale import m3_quintuple as Q                               # noqa: E402
from scale import negation_scope as NS                            # noqa: E402
from scale.pivot_probe import select_pivots                       # noqa: E402

S, D, N, K = 64, 24, 8, 8


def _batch(seed=0):
    x, y, f, p = NS.M3_TASKS["e3_t1"][0](N, S, D, d_model=M3.D_MODEL, seed=seed)
    torch.manual_seed(seed)
    arm = M3.Arm("softmax", s=S)
    with torch.no_grad():
        q, kk = arm.wq(x), arm.wk(x)
    return x, q, kk


def test_the_shipped_pivot_set_cannot_see_the_answer_token():
    """The confound itself, asserted rather than described."""
    m, _nm = bench._causal_mask_pair(S, 0, "cpu")
    assert int(m.diagonal().sum()) == 0, "mask is not strictly causal"
    assert int(m[S - 2].nonzero().max()) == S - 3
    assert int(m[S - 1].nonzero().max()) == S - 2
    _x, _q, kk = _batch()
    assert int(Q.batched_pivots(kk, K).max()) <= S - 2


def test_permitting_the_query_row_is_not_enough():
    """`exclude=(0,)` leaves `s-1` ELIGIBLE and UNSELECTED.

    This is the measurement that killed the first proposed lift. Without it, a
    change that moves no number ships as a fix.
    """
    _x, _q, kk = _batch()
    permitted = torch.stack([select_pivots(kk[i], min(K, S - 2), exclude=(0,))
                             for i in range(N)])
    assert not bool((permitted == S - 1).any()), (
        "s-1 was selected by content here; re-measure the rank claim")
    assert torch.equal(permitted.sort().values,
                       Q.batched_pivots(kk, K).sort().values), (
        "permitting changed the pivot set, so it is not inert after all")


def test_the_reserved_slot_puts_the_query_row_in_every_example():
    _x, _q, kk = _batch()
    plus = Q.batched_pivots(kk, K, reserve_query=True)
    assert plus.shape == (N, K), f"k changed: {plus.shape}"
    assert bool((plus == S - 1).all(dim=1).any() or
                (plus == S - 1).any(dim=1).all()), "s-1 missing from some example"
    assert (plus == S - 1).sum(dim=1).min() == 1, "s-1 must appear exactly once"
    # the other k-1 are still content-selected, and still exclude 0 and s-1
    rest = plus[:, :-1]
    assert int(rest.max()) <= S - 2 and int(rest.min()) >= 1


def test_the_reserved_slot_reproduces_the_softmax_row():
    """THE MECHANISM BIND. `av` at the reserved slot must BE the softmax cell's
    own row applied to `v` -- that is what makes softmax an interior point of
    `twin_plus`'s function class.

    THE CONTROL IS THE SHIPPED SET: the same assertion must FAIL there, or this
    is testing arithmetic that was always true and the lift is decorative.
    """
    x, q, kk = _batch()
    with torch.no_grad():
        a = bench._softmax_operator(q, kk)
        want = (a @ x)[:, S - 1].double()

    plus = Q.batched_pivots(kk, K, reserve_query=True)
    _lg, _lgate, av = Q.batched_log_pivot_context(q, kk, x, plus,
                                                  need_gram=False)
    slot = int((plus[0] == S - 1).nonzero()[0])
    got = av[:, slot, :]
    assert torch.allclose(got, want, atol=1e-9), (
        f"reserved slot is not the softmax row: "
        f"max abs {float((got - want).abs().max()):.3e}")

    ship = Q.batched_pivots(kk, K)
    _l2, _g2, av2 = Q.batched_log_pivot_context(q, kk, x, ship, need_gram=False)
    assert not any(torch.allclose(av2[:, j, :], want, atol=1e-9)
                   for j in range(K)), (
        "the SHIPPED pivot set already contains the softmax row; the confound "
        "as described does not exist and the lift is unnecessary")


def test_the_lift_restores_sensitivity_to_the_answer_token():
    """Perturb `x[s-2]` and require the lifted `av` to move like softmax does
    and the shipped `av` not to. Controls seen to fire, both halves."""
    x, q, kk = _batch()

    def av_move(piv):
        _l, _g, av = Q.batched_log_pivot_context(q, kk, x, piv, need_gram=False)
        x2 = x.clone()
        x2[:, S - 2, :] += 100.0
        torch.manual_seed(0)
        arm = M3.Arm("softmax", s=S)
        with torch.no_grad():
            q2, k2 = arm.wq(x2), arm.wk(x2)
        _l2, _g2, av2 = Q.batched_log_pivot_context(q2, k2, x2, piv,
                                                    need_gram=False)
        return float((av2 - av).abs().max())

    shipped = av_move(Q.batched_pivots(kk, K))
    lifted = av_move(Q.batched_pivots(kk, K, reserve_query=True))
    assert lifted > 10 * shipped, (
        f"the lift did not restore value access: shipped {shipped:.6f}, "
        f"lifted {lifted:.6f}")


def test_the_plus_cells_exist_and_the_shipped_default_is_untouched():
    """New cells must not change what a default run measures."""
    assert Q.CELLS == ("softmax", "glance", "settled", "twin", "argmax")
    assert Q.PLUS_CELLS == ("twin_plus", "settled_plus")
    assert Q.ROW_CELLS == ("twinrow", "settledrow")
    #: Every extra cell family is a NEW TUPLE BESIDE `CELLS`, never a name added
    #: to it, so the tuples must be pairwise disjoint and must exhaust
    #: `ALL_CELLS`. Stated this way the check keeps working as families are
    #: added, which the previous `CELLS | PLUS_CELLS` equality could not.
    fams = (Q.CELLS, Q.PLUS_CELLS, Q.ROW_CELLS)
    assert set(Q.ALL_CELLS) == set().union(*map(set, fams))
    assert sum(len(f) for f in fams) == len(set(Q.ALL_CELLS)), "families overlap"
    ap = Q._argparser()
    assert ap.parse_args([]).cells == list(Q.CELLS), (
        "a default run would now measure the plus or row cells too")
    for c in Q.PLUS_CELLS + Q.ROW_CELLS:
        Q.QuintArm("softmax", S, cell=c)          # must construct
    with pytest.raises(ValueError):
        Q.QuintArm("softmax", S, cell="twin_minus")
