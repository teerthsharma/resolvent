"""Three binds M2 has to survive before it can be marked GREEN. ALL RED.

Each asserts the SAFE property, so each FAILS while the defect stands. Nothing
here reimplements the probe: every operator comes out of `scale.pivot_probe`
and `ceq.bench`, the same code path the journalled units ran.

1. THE s=8 COLUMN IS ONE ARM REPORTED TWICE (G3, the ParaFormer class).
   `select_pivots` excludes i and j, so at s=8 only 6 indices remain and P is
   ALL of them. `A[:,P] A[P,:]` then differs from `A @ A` only by the p = i and
   p = j terms, and both vanish because A is strictly lower triangular. So the
   pivot arm and the dense CONTROL compute a bitwise-identical operator at s=8.
   results/m2.jsonl agrees to every digit:
       pivot_signed__in_P/s8       k=101 n=4096 rate=0.024658203125
       dense_signed__at_pivots/s8  k=101 n=4096 rate=0.024658203125
   That shared point is the LEFTMOST point of both log-log fits, so it anchors
   the claim slope and the control slope to the same measurement.

2. |P| IS 6 AT s=8, NOT THE k=8 EVERY ARTIFACT REPORTS.
   `results/m2_run.txt` and `scale/m2_units.py` both state k=8 pivots at every
   size. At s=8 the probe silently uses 6. A "flat in s" curve whose leftmost
   point has a different k is not measuring flatness in s alone.

3. THE M2 ARM'S OPERATOR SHIPS NOWHERE.
   `pivot_signed` / `dense_signed` build A with `_causal_tgate_operator`,
   A_ij = g_i * tanh(qhat_i . khat_j / tau) -- denominator-free. The HuggingFace
   package ships `sgate`, rho*(softmax(w) - lam*softmax(-w))/(1+lam), and
   `ceq/attention.py` ships `ceq_operator`, rho*w/||w||_1. Both shipped forms
   carry the denominator that `ceq/bench.py` itself names as "the measured cause
   of the 1/s death". A flatness result on tgate is not a result about either.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ceq import bench
from scale.pivot_probe import build_arm, pivot_hop2, select_pivots

D, TAU, SEED = 16, 1.0, 0


def _draw(s: int, seed: int = SEED):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, generator=g)
    wq, wk = rnd(D, D), rnd(D, D)
    x0 = rnd(s, D)
    gvec, bet = torch.sigmoid(rnd(s)), torch.sigmoid(rnd(s))
    i, j = s - 1, max(1, s // 4)
    pivots = select_pivots(x0 @ wk, 8, exclude=(i, j))
    return x0 @ wq, x0 @ wk, gvec, bet, pivots, i, j


def test_the_pivot_and_dense_arms_are_distinct_at_s8():
    """SAFE PROPERTY, CURRENTLY RED: the claim arm and its control must not be
    the same operator at any size in the table."""
    qq, kk, gvec, bet, pivots, i, j = _draw(8)
    _, h_pivot = build_arm("pivot_signed", qq, kk, gvec, bet, pivots, tau=TAU)
    _, h_dense = build_arm("dense_signed", qq, kk, gvec, bet, pivots, tau=TAU)
    # [i, j] is the ONLY entry the flip measurement reads.
    delta = abs(float(h_pivot[i, j]) - float(h_dense[i, j]))
    # The probe's own significance floor. Anything under it is summation-order
    # noise, not two different operators.
    assert delta > 1e-6, (
        "at s=8 the pivot arm and the dense CONTROL produce the SAME "
        "hop2[i=%d, j=%d] to float32 summation noise (delta = %r, the probe's "
        "own floor is 1e-6), so the two s=8 cells in the M2 table are one "
        "measurement under two names -- results/m2.jsonl records k=101 n=4096 "
        "rate=0.024658203125 for BOTH. |P| = %d covers all 6 usable indices; "
        "the dropped p = i and p = j terms both vanish on a strictly lower "
        "triangular A. At s=32 the same delta is 1.3e-02."
        % (i, j, delta, int(pivots.numel())))


def test_the_two_arms_stay_distinct_where_the_table_says_they_differ():
    """CONTROL FOR THE ABOVE, EXPECTED GREEN: at s=32 they really do differ, so
    test 1's failure is a fact about s=8 and not a broken comparison."""
    qq, kk, gvec, bet, pivots, i, j = _draw(32)
    _, h_pivot = build_arm("pivot_signed", qq, kk, gvec, bet, pivots, tau=TAU)
    _, h_dense = build_arm("dense_signed", qq, kk, gvec, bet, pivots, tau=TAU)
    assert abs(float(h_pivot[i, j]) - float(h_dense[i, j])) > 0.0


def test_k_is_the_reported_8_at_every_size_in_the_table():
    """SAFE PROPERTY, CURRENTLY RED: every artifact says 'k=8 pivots'."""
    got = {}
    for s in (8, 32, 128):
        *_, pivots, _, _ = _draw(s)
        got[s] = int(pivots.numel())
    assert got[8] == 8, (
        "|P| by size = %r. `select_pivots` takes topk(min(k, #finite)) and "
        "excludes i and j, so at s=8 only 6 candidates survive. The leftmost "
        "point of the M2 flatness curve was measured at a different k from "
        "every other point." % (got,))


def test_the_m2_signed_arm_uses_an_operator_that_ships():
    """SAFE PROPERTY, CURRENTLY RED: the operator M2 measures must be one the
    package actually ships, or M2 is a result about a harness-only operator."""
    qq, kk, gvec, bet, pivots, _, _ = _draw(32)
    a_m2, _ = build_arm("pivot_signed", qq, kk, gvec, bet, pivots, tau=TAU)

    shipped = {
        "sgate (ceq/hf/modeling_ceq.py default, PARITY_POINT rho=1.5 lam=0.10)":
            bench._causal_sgate_operator(qq, kk, rho=1.5, lam=0.10),
        "signed / ceq_operator (ceq/attention.py, rho*w/L1)":
            bench._causal_signed_operator(qq, kk),
    }
    deltas = {name: float((a_m2 - a).abs().max()) for name, a in shipped.items()}
    assert min(deltas.values()) == 0.0, (
        "M2's `pivot_signed` base operator matches NO shipped operator. "
        "max|delta| against each: %r. M2 measures `_causal_tgate_operator`, "
        "A_ij = g_i*tanh(qhat_i.khat_j/tau), which exists only in ceq/bench.py "
        "and is imported by nothing under ceq/hf/ or ceq/attention.py."
        % (deltas,))


def test_the_windowed_arm_actually_windows_the_operator():
    """SAFE PROPERTY, CURRENTLY RED. M2's frozen Test field names three arms:
    `c in P` vs `c not-in P` vs `windowed placement`, and the frozen baseline
    reads "windowed arms buy flatness only by surrendering reach".

    `build_arm` never passes `window=` to any operator, so no M2 arm is windowed.
    The `windowed` label selects only the POOL that `c` is drawn from --
    `range(i-8, i)` -- while hop 2 stays routed through the same global P. What
    that cell measures is the chance a window token happened to be a pivot:
    88/320 at s=32, 18/320 at s=128, 3/320 at s=512. It decays as k/s and says
    nothing about reach.
    """
    qq, kk, gvec, bet, pivots, _, _ = _draw(32)
    a_open, _ = build_arm("pivot_signed", qq, kk, gvec, bet, pivots, tau=TAU)
    a_win = bench._causal_tgate_operator(qq, kk, gvec, TAU, window=8)
    assert float((a_open - a_win).abs().max()) == 0.0, (
        "M2's arms are built with window=0 at every placement, including the "
        "one labelled `windowed`; a true windowed operator differs from the "
        "one M2 measures by max|delta| = %r."
        % float((a_open - a_win).abs().max()))
