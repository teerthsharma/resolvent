"""Where does the bucketed form of `ceq.arm_smprime.operator` break?

THE OBJECT. `W_ij = G_ij exp(qk q_i.k_j) / Z_i^beta`, `G_ij = prod_{k=j+1..i}
m_k e^{i theta_k}`, shipped dense at `ceq/arm_smprime.py:252-257`. The bucketed
form under test splits the key axis into blocks and merges:

    m_b = max_j log|w_ij|,  l_b = sum_j |w_ij| e^{-m_b},  o_b = sum_j w_ij v_j e^{-m_b}
    M = max_b m_b,  l = sum_b l_b e^{m_b-M},  o = sum_b o_b e^{m_b-M}
    out_i(beta) = e^{(1-beta)M} o / l^beta

Three merges live in this file because no bucketed implementation of this
operator is in the tree: the two blockwise merges that do exist
(`ceq/mz_kernel.py:68-121` and `tests/chase/k22.py:186-229`) are Triton, carry
no gate and no `beta`, and do not run on this CPU-only box.

    `_merge_spec`           the phase's formula VERBATIM
    `_merge_content_shift`  shift from the content term only, gate multiplicative
    `_merge_guarded`        shift from `log R + logit` with a live mask

THE PLANTED NEGATIVES, which must be SEEN to fire:

  (a) on a row with no dead entry and no overflow, all three merges must
      reproduce the dense operator to 1e-12. A guard that changes a healthy row
      is a guard that is reading something else.
  (b) the closed-gate bed must be the ONLY thing that separates them, so the
      same bed with `m_4 = 1` must be clean under all three.
  (c) at a logit of +80 rather than +800 the dense operator is finite, so the
      overflow tests are measuring the overflow and not the bed.

THE RED RUN, verbatim. Five assertions stating the property the round WANTS,
run against `ceq/arm_smprime.py` and the phase's formula as they stand, before
`_merge_guarded` was allowed to differ from `_merge_spec`
(`python -m pytest test_red_first.py -q`, `5 failed in 2.69s`):

    AssertionError: spec-verbatim merge at block=4 returned
      [1.0, 1.5, 2.0, 2.5, nan, nan, nan, nan] against dense
      [1.0, 1.5, 2.0, 2.5, 5.0, 5.5, 6.0, 6.5]
    AssertionError: dense readout at a planted logit of +800 is (nan+nanj)
    AssertionError: the log-domain merge is nan at beta=0.11277, finite only for
      beta > 0.11277160888326998
    AssertionError: ceq/arm_smprime.py:244 says Z is STRICTLY POSITIVE; at a
      logit of -800 it is [0.0, 0.0, 0.0, 0.0] and the operator is nan
    AssertionError: block=1 and block=4 differ; max|d| 2.2377e-16
      At index 5 diff: (-0.10724960132718507+0.0728031045997474j)
                    != (-0.10724960132718507+0.07280310459974738j)

Each assertion below pins the MEASURED break rather than the wish, so the file
is green against the operator as it stands and goes red again the moment any of
the behaviours changes. The numbers are this box's: torch 2.14.0+cpu, float64,
`python -m pytest tests/curvature/test_bucket_numerics.py`.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ceq import arm_smprime as A

DT = torch.float64

def _bisect_exp_edge(dtype):
    """The largest `x` for which `torch.exp(torch.tensor(x, dtype=dtype))` is
    still finite, found by bisection on `exp` itself -- not `log(finfo.max)`.
    The two are not the same number: on this box (torch 2.14.0+cpu) the
    float32 edge sits `3.51e-6` below `log(finfo(float32).max)` --

        python -c "import torch; print(torch.log(torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float64)).item())"

    prints `88.72283905206835`, `3.51129688e-06` above the bisected edge below.
    Using the `log(finfo.max)` form would put the `beta` floor on the wrong
    side by `4.4e-9`, and a floor that is wrong in the unsafe direction is not
    a floor. The edge is computed here, once, instead of pasted as a literal,
    so this file cannot quote a value that its own bisection disagrees with.
    """
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if torch.isfinite(torch.exp(torch.tensor(mid, dtype=dtype))):
            lo = mid
        else:
            hi = mid
    return lo


#: The overflow edge of each dtype -- see `_bisect_exp_edge` above.
LOG_MAX_F64 = _bisect_exp_edge(torch.float64)
LOG_MAX_F32 = _bisect_exp_edge(torch.float32)

#: The relative bar the round asks the bucketed form to hold. `1e-12` is a
#: float64-only number: `docs/canon/08_ARCHITECTURE.md:729` already ruled that a
#: single `1e-12` on the shipped float32 path is `5.96e4` below one rounding and
#: names `eps_32 = 1.5e-4` in its place.
BAR = 1e-12
BAR_F32 = 1.5e-4


# ---------------------------------------------------------------------------
# 0. THE THREE MERGES
# ---------------------------------------------------------------------------

def _pieces(q, k, u, theta):
    """`(G, R, logit, live)` -- everything the merges read, from the shipped module."""
    s = q.shape[-2]
    m = A.magnitude(u)
    G = A.path_product(A.gate(m, theta))
    R = A.path_product(m)
    logit = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    causal = ~torch.ones(s, s, dtype=torch.bool, device=q.device).triu(1)
    return G, R, logit, (R > 0) & causal


def _merge_spec(q, k, v, u, theta, beta, block):
    """The phase's formula VERBATIM: `m_b = max_j log|w_ij|` on a formed `|w_ij|`."""
    s = q.shape[-2]
    G, R, logit, _ = _pieces(q, k, u, theta)
    e = torch.exp(logit.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), A.NEG))
    w = G * e.to(G.dtype)
    out = []
    for i in range(s):
        mb, lb, ob = [], [], []
        for j0 in range(0, i + 1, block):
            js = list(range(j0, min(j0 + block, i + 1)))
            wij = w[i, js]
            m_b = torch.log(wij.abs()).max()
            mb.append(m_b)
            lb.append((wij.abs() * torch.exp(-m_b)).sum())
            ob.append((wij * torch.exp(-m_b).to(wij.dtype) * v[js, 0].to(wij.dtype)).sum())
        out.append(_finish(torch.stack(mb), torch.stack(lb), torch.stack(ob), beta))
    return out


def _merge_content_shift(q, k, v, u, theta, beta, block):
    """THE VARIANT A KERNEL ACTUALLY WANTS: shift from the CONTENT term only,
    `m_b = max_j s_ij`, with the gate riding multiplicatively in `R_ij`.

    It dodges the dead-block NaN for free -- `s_ij` never sees a closed gate, so
    `m_b` is `-inf` only where the causal mask already is, and no `log m` is
    taken anywhere, so `V16Domain.lean::no_prefix_scan_represents_a_zero_gate`
    is not touched. But the shift is now chosen by an entry that may contribute
    NOTHING: when the row max of `s` falls on a `j` whose `R_ij = 0`, every live
    term is scaled by `e^{s_ij - m_b}` and they can underflow together, leaving
    `l = 0` and `o / l^beta = 0/0`.
    `test_a_shift_chosen_by_a_dead_entry_empties_the_row` is that bed.
    """
    s = q.shape[-2]
    G, R, logit, _ = _pieces(q, k, u, theta)
    sc = logit.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), A.NEG)
    out = []
    for i in range(s):
        mb, lb, ob = [], [], []
        for j0 in range(0, i + 1, block):
            js = list(range(j0, min(j0 + block, i + 1)))
            m_b = sc[i, js].max()
            e = torch.exp(sc[i, js] - m_b)
            mb.append(m_b)
            lb.append((R[i, js] * e).sum())
            ob.append((G[i, js] * e.to(G.dtype) * v[js, 0].to(G.dtype)).sum())
        out.append(_finish(torch.stack(mb), torch.stack(lb), torch.stack(ob), beta))
    return out


def _merge_guarded(q, k, v, u, theta, beta, block, reverse_blocks=False):
    """The repair, two changes and no more.

    ONE: `m_b` is taken in the LOG domain from `log R_ij + logit_ij`, never from
    a formed `|w_ij|` -- an overflowed `|w_ij|` has already lost the row before
    the max can subtract anything -- and never from the content term alone, so
    the shift cannot be set by an entry that contributes nothing.
    TWO: an entry with `R_ij == 0` is DEAD and contributes exactly `0`; a block
    with no live entry is dropped before the max, so `e^{-m_b}` never meets
    `m_b = -inf` and `l_b` is never `0 * inf`. This is the guard
    `ceq/mz_kernel.py:104,121` already ships and `tests/chase/k22.py:212,229`
    does not.

    `log R` is a logarithm of the gate product, which is why clause TWO is not
    optional: the log form is legal only ON THE LIVE SET, where `R_ij > 0`. Off
    it the live mask carries the zero, not an extended-real convention.
    """
    s = q.shape[-2]
    G, R, logit, live = _pieces(q, k, u, theta)
    tiny = torch.finfo(R.dtype).tiny
    lg = torch.where(live, torch.log(R.clamp_min(tiny)) + logit,
                     torch.full_like(logit, float("-inf")))
    #: the unit-modulus phase of `w_ij`, read off `G/R` so no `log` touches it.
    ph = torch.where(live, G / R.clamp_min(tiny).to(G.dtype), torch.zeros_like(G))
    out = []
    for i in range(s):
        mb, lb, ob = [], [], []
        for j0 in range(0, i + 1, block):
            js = list(range(j0, min(j0 + block, i + 1)))
            alive = live[i, js]
            if not bool(alive.any()):
                continue
            m_b = lg[i, js].max()
            sc = torch.where(alive, torch.exp(lg[i, js] - m_b),
                             torch.zeros_like(logit[i, js]))
            mb.append(m_b)
            lb.append(sc.sum())
            ob.append((sc.to(G.dtype) * ph[i, js] * v[js, 0].to(G.dtype)).sum())
        if not mb:
            out.append(REFUSE)
            continue
        if reverse_blocks:
            #: same blocks, same boundaries -- only the order `_finish` sums
            #: them in changes. Isolates MERGE ORDER from block SIZE.
            mb, lb, ob = mb[::-1], lb[::-1], ob[::-1]
        out.append(_finish(torch.stack(mb), torch.stack(lb), torch.stack(ob), beta))
    return out


#: A row with no live entry has `l = 0` and `o/l^beta` is `0/0`. The house
#: convention for that is a value in the output space and never a `nan`
#: (`ceqjepa/curvature.py:198-210`: "a threshold test applied to a Refusal
#: fails loudly (TypeError) instead of passing the way `nan < tol` silently
#: does"). No bed here reaches it -- `R_ii` is the empty product, so `j = i` is
#: always live -- so the branch is stated and not tested.
REFUSE = "REFUSE"


def _finish(mb, lb, ob, beta):
    M = mb.max()
    sc = torch.exp(mb - M)
    l = (lb * sc).sum()
    o = (ob * sc.to(ob.dtype)).sum()
    return (torch.exp((1 - beta) * M).to(o.dtype) * o / (l ** beta).to(o.dtype)).item()


ALL_MERGES = (("spec", _merge_spec), ("content", _merge_content_shift),
              ("guarded", _merge_guarded))


# ---------------------------------------------------------------------------
# 1. THE BEDS
# ---------------------------------------------------------------------------

def _gate_bed(closed=4, s=8):
    """`s` tokens, all logits `0`, every gate open except `closed`.

    `closed = 4` puts the zero EXACTLY on the boundary of a `block = 4` split,
    which is the case the round asks for; `block = 3` puts it in a block's
    interior, and the tests run both so the finding is not about the boundary.
    """
    q = torch.zeros(s, 1, dtype=DT)
    k = torch.zeros(s, 1, dtype=DT)
    v = torch.arange(1.0, s + 1, dtype=DT).unsqueeze(-1)
    u = torch.ones(s, dtype=DT)
    if closed is not None:
        u[closed] = 0.0
    return q, k, v, u, torch.zeros(s, dtype=DT)


def _logit_bed(big=800.0):
    """One planted logit of `big` at `(i, j) = (3, 0)`, every other logit `0`."""
    q = torch.tensor([[0.0], [0.0], [0.0], [big]], dtype=DT)
    k = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=DT)
    v = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=DT)
    return q, k, v, torch.ones(4, dtype=DT), torch.zeros(4, dtype=DT)


def _dead_shift_bed(big=700.0):
    """The row max of the CONTENT term sits on an entry the GATE killed.

    Three tokens, gate closed at index 2, so `R_2j = 0` for `j < 2` and
    `R_22 = 1` (the empty product). The `+big` logit is planted at `(2, 0)`,
    one of the dead entries, and the one LIVE entry `(2, 2)` carries `-big/7`,
    so the spread across row 2 is `8 big / 7`.

    `big = 700` is chosen against both edges at once: `e^{700} = 1.01e304` is
    finite, so the dense operator survives the bed, while the spread
    `800 > 746` is past the float64 underflow edge, so a merge that shifts by
    the content-term max loses the live entry entirely.
    """
    q = torch.tensor([[0.0], [0.0], [big]], dtype=DT)
    k = torch.tensor([[1.0], [0.0], [-1.0 / 7.0]], dtype=DT)
    v = torch.tensor([[1.0], [2.0], [3.0]], dtype=DT)
    u = torch.ones(3, dtype=DT)
    u[2] = 0.0
    return q, k, v, u, torch.zeros(3, dtype=DT)


def _live_bed(s=16, d=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(s, d, generator=g, dtype=DT),
            torch.randn(s, d, generator=g, dtype=DT),
            torch.randn(s, 1, generator=g, dtype=DT),
            torch.rand(s, generator=g, dtype=DT),
            torch.rand(s, generator=g, dtype=DT) * 6.0)


def _dense(q, k, v, u, theta, beta):
    return [c for c in A.readout(q, k, v, u, theta, beta=beta).flatten().tolist()]


def _rel(got, ref):
    return max(abs(a - b) / max(abs(b), torch.finfo(DT).tiny) for a, b in zip(got, ref))


# ---------------------------------------------------------------------------
# 2. THE PLANTED NEGATIVES. These must pass or nothing below means anything.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("block", [1, 2, 3, 4, 8, 16])
def test_every_merge_reproduces_the_dense_operator_on_a_live_row(beta, block):
    """(a) No dead entry, no overflow: all three merges ARE the dense operator."""
    q, k, v, u, th = _live_bed()
    ref = _dense(q, k, v, u, th, beta)
    for name, fn in ALL_MERGES:
        rel = _rel(fn(q, k, v, u, th, beta, block), ref)
        assert rel <= BAR, "%s merge misses dense by %.4e at block=%d beta=%s" % (
            name, rel, block, beta)


@pytest.mark.parametrize("block", [1, 2, 3, 4, 8])
def test_the_bed_with_no_closed_gate_is_clean_under_every_merge(block):
    """(b) The separation below is the closed gate and nothing else about the bed."""
    q, k, v, u, th = _gate_bed(closed=None)
    ref = _dense(q, k, v, u, th, 1.0)
    for name, fn in ALL_MERGES:
        assert _rel(fn(q, k, v, u, th, 1.0, block), ref) <= BAR, \
            "%s merge broke a bed with every gate open" % name


def test_a_logit_of_80_is_the_control_for_the_overflow_beds():
    """(c) At +80 the dense operator is finite, so +800 measures the overflow."""
    q, k, v, u, th = _logit_bed(big=80.0)
    out = A.readout(q, k, v, u, th, beta=1.0)
    assert bool(torch.isfinite(out.real).all()), out
    assert out[3].item() == 1 + 0j, out[3].item()


# ---------------------------------------------------------------------------
# 3. FIRED: A DEAD BLOCK POISONS THE ROW WITH NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block", [2, 3, 4])
def test_a_dead_block_makes_the_spec_verbatim_merge_return_nan(block):
    """The kill. `m_b = max_j log|w_ij|` is `-inf` on a block the gate zeroed,
    so `l_b = sum 0 * e^{+inf}` is `0 * inf = nan`, and the merge's
    `e^{m_b - M} = 0` then multiplies a `nan` and cannot recover it.

    The dense operator gets this bed exactly right, so the NaN is manufactured
    by the bucketing and by nothing else. `block = 3` puts the closed gate in a
    block's INTERIOR and fails identically: the break is any wholly-dead block,
    not a boundary coincidence.
    """
    q, k, v, u, th = _gate_bed(closed=4)
    ref = _dense(q, k, v, u, th, 1.0)
    assert [c.real for c in ref] == [1.0, 1.5, 2.0, 2.5, 5.0, 5.5, 6.0, 6.5], ref
    got = _merge_spec(q, k, v, u, th, 1.0, block)
    bad = [i for i, c in enumerate(got) if c != c]
    assert bad == [4, 5, 6, 7], "rows %r went nan at block=%d, expected [4,5,6,7]" % (
        bad, block)


@pytest.mark.parametrize("block", [1, 2, 3, 4, 8])
def test_the_two_repaired_merges_survive_the_closed_gate_bitwise(block):
    """The replacement, measured: keep the gate out of the shift, or drop a
    wholly-dead block before the max, and the merge is BITWISE the dense
    operator on this bed at every block size."""
    q, k, v, u, th = _gate_bed(closed=4)
    ref = _dense(q, k, v, u, th, 1.0)
    for name in ("content", "guarded"):
        fn = dict(ALL_MERGES)[name]
        got = fn(q, k, v, u, th, 1.0, block)
        assert got == ref, "%s merge at block=%d: %r vs dense %r" % (name, block, got, ref)


# ---------------------------------------------------------------------------
# 4. FIRED: A SHIFT CHOSEN BY A DEAD ENTRY EMPTIES THE ROW
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block", [1, 2, 3])
def test_a_shift_chosen_by_a_dead_entry_empties_the_row(block):
    """The second kill, and it is the price of the first repair. Taking the
    shift from the content term alone makes `m_b` blind to the gate, so the
    row's max can sit on an entry with `R_ij = 0`. Here `l` underflows to
    exactly `0.0` and `o / l^1` is `0/0`.

    The dense operator is finite on this bed and the log-domain merge is exact,
    so the NaN belongs to the shift choice and not to the bed.
    """
    q, k, v, u, th = _dead_shift_bed()
    ref = _dense(q, k, v, u, th, 1.0)
    assert [c.real for c in ref] == [1.0, 1.5, 3.0], ref
    got = _merge_content_shift(q, k, v, u, th, 1.0, block)
    assert got[2] != got[2], "row 2 should be nan at block=%d, got %r" % (block, got[2])
    assert _merge_guarded(q, k, v, u, th, 1.0, block) == ref


def test_the_dead_shift_bed_is_clean_at_a_logit_small_enough_not_to_underflow():
    """The control for the bed above: the same dead maximum, a logit of +80
    instead of +700, and the content-shift merge is exact. The break is the
    UNDERFLOW the shift forces, not the presence of a dead maximum."""
    q, k, v, u, th = _dead_shift_bed(big=80.0)
    ref = _dense(q, k, v, u, th, 1.0)
    assert _rel(_merge_content_shift(q, k, v, u, th, 1.0, 2), ref) <= BAR


def test_an_overflowing_logit_on_a_DEAD_entry_contributes_exactly_zero():
    """A third dense break, now repaired -- pin the CORRECT behaviour.
    `ceq/arm_smprime.py:236` returns `rh * e` with `rh = R_ij` and
    `e = exp(logit)`. On an entry the gate killed, `R_ij` is exactly `0`; the
    gate-kill mask now runs BEFORE the complex multiply (masking the dead set
    ahead of the exponential), so a dead entry contributes exactly `0.0`,
    bitwise, no matter how far its own logit overflows -- `0 * inf` never
    gets formed, because the `inf` never gets multiplied against the `0`.
    The mask-before-exp comment at `:230-231` is about the causal triangle;
    this is the separate gate-kill mask, moved earlier by the same fix that
    made `tests/arm_smprime/test_bf16_ceiling.py` need inverting this
    morning and made `tests/arm_smprime/test_gate_kill_mask.py` grow cases --
    this is the third instance of that one class, not a new one.

    RED AGAIN IF: the gate-kill mask is removed, or reordered to run after
    the exponential instead of before it -- `mod[2, 0]` would go back to
    manufacturing `nan` and the first assertion below would fail exactly the
    way the old, opposite-signed version of this test used to pass."""
    q, k, v, u, th = _dead_shift_bed(big=800.0)
    ref = _dense(q, k, v, u, th, 1.0)
    mod = A.numerator(q, k, u, th)[1]
    assert mod[2, 0] == 0.0, mod[2].tolist()
    #: the live, gradually-underflowed entry is untouched by the mask and
    #: still carries its tiny but nonzero magnitude -- this is the third
    #: value from the failing run, `2.3245822930325816e-50`, and it is fine.
    assert mod[2, 2] == 2.3245822930325816e-50, mod[2].tolist()
    out = A.readout(q, k, v, u, th, beta=1.0).flatten().tolist()
    assert out[2] == ref[2], (out, ref)
    #: and the control is the bed one edge lower, where `e` is finite even on
    #: the dead entry, so the mask ordering makes no visible difference there.
    q7, k7, v7, u7, th7 = _dead_shift_bed(big=700.0)
    assert A.numerator(q7, k7, u7, th7)[1][2].tolist() == [0.0, 0.0,
                                                           math.exp(-100.0)]


# ---------------------------------------------------------------------------
# 5. FIRED, PARTLY: THE +800 RESCUE EXISTS BUT ONLY ABOVE A beta FLOOR
# ---------------------------------------------------------------------------

def test_the_dense_operator_and_the_spec_verbatim_merge_both_die_at_a_logit_of_800():
    """`ceq/arm_smprime.py:232` exponentiates the logit with no max subtracted,
    so `exp(800) = inf`, `Z = inf`, and `inf/inf` is `nan`. The spec-verbatim
    merge inherits it: `m_b = max_j log|w_ij|` reads `log(inf) = inf` off a
    value that overflowed before the max was taken."""
    q, k, v, u, th = _logit_bed()
    dense = A.readout(q, k, v, u, th, beta=1.0)[3].item()
    assert dense != dense, "dense row 3 should be nan, got %r" % dense
    spec = _merge_spec(q, k, v, u, th, 1.0, 2)[3]
    assert spec != spec, "spec merge row 3 should be nan, got %r" % spec


def test_the_repaired_merges_rescue_800_at_beta_one():
    """Subtract a max that was chosen before the exponential and the row is
    exact: `out_3 = v_0 = 1`."""
    q, k, v, u, th = _logit_bed()
    assert _merge_guarded(q, k, v, u, th, 1.0, 2)[3] == 1 + 0j
    assert _merge_content_shift(q, k, v, u, th, 1.0, 2)[3] == 1 + 0j


@pytest.mark.parametrize("beta,finite", [(1.0, True), (0.9, True), (0.5, True),
                                         (0.12, True), (0.11277, False),
                                         (0.05, False), (0.0, False)])
def test_the_800_rescue_dies_below_a_beta_floor(beta, finite):
    """THE BOUNDARY, and it is arithmetic, not a bug. `out_i(beta)` carries
    `e^{(1-beta)M}` with `M = 800`, so the ANSWER itself exceeds float64 once
    `(1-beta) * 800 > 709.782712893384`. No merge can return a finite value for
    a quantity that is not representable: the round's requirement "the bucketed
    form must return a finite output where the dense form overflows" holds only
    for `beta > 1 - LOG_MAX/M`, which is 0.11277160888326998 in float64 and
    0.8890964555740357 in float32."""
    floor = 1 - LOG_MAX_F64 / 800.0
    assert (beta > floor) is finite, "floor is %.17g" % floor
    got = _merge_guarded(*_logit_bed(), beta, 2)[3]
    assert (got == got) is finite, "beta=%s gave %r" % (beta, got)


def test_the_dtype_edges_are_where_the_beta_floor_comes_from():
    """The two constants above, bisected rather than quoted."""
    for dtype, want in ((torch.float32, LOG_MAX_F32), (torch.float64, LOG_MAX_F64)):
        lo, hi = 0.0, 1000.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if torch.isfinite(torch.exp(torch.tensor(mid, dtype=dtype))):
                lo = mid
            else:
                hi = mid
        assert lo == want, "%s edge is %.17g, not %.17g" % (dtype, lo, want)


# ---------------------------------------------------------------------------
# 6. FIRED: Z IS NOT STRICTLY POSITIVE, AND THE DOCSTRING SAYS IT IS
# ---------------------------------------------------------------------------

def test_Z_underflows_to_exactly_zero_and_the_operator_returns_nan():
    """`ceq/arm_smprime.py:244-247` claims `Z_i` "is real and STRICTLY POSITIVE
    ... So `Z^beta` needs no positivity side condition". The claim is true in
    the reals and FALSE in float64: `R_ii exp(w_ii) > 0` requires `exp(w_ii)`
    not to underflow, and at `w_ii = -800` it is exactly `0`. Every other term
    underflows too, so `Z = 0`, `Z^1 = 0`, and `num/Z` is `0/0 = nan` --
    reached with ORDINARY large-negative logits and no mask anywhere.

    Line 255 is also the only normalizer in this repo with no floor on it: the
    same division is written `.clamp_min(torch.finfo(w.dtype).tiny)` at
    `ceq/attention.py:189`, `ceq/lm.py:162`, `ceq/hopcache.py:72`,
    `ceq/bench.py:180`, `scale/arm_s.py:128` and `scale/foreman_hilbert.py:139`.
    """
    s = 4
    q = torch.full((s, 1), -800.0, dtype=DT)
    k = torch.ones(s, 1, dtype=DT)
    v = torch.arange(1.0, s + 1, dtype=DT).unsqueeze(-1)
    u, th = torch.ones(s, dtype=DT), torch.zeros(s, dtype=DT)
    Z = A.numerator(q, k, u, th)[1].sum(-1)
    assert Z.tolist() == [0.0] * s, Z.tolist()
    out = A.readout(q, k, v, u, th, beta=1.0).flatten().tolist()
    assert all(c != c for c in out), out
    #: the control: ten times smaller a logit and the row is ordinary.
    Zc = A.numerator(q / 10, k, u, th)[1].sum(-1)
    assert all(z > 0.0 for z in Zc.tolist()), Zc.tolist()
    assert bool(torch.isfinite(A.readout(q / 10, k, v, u, th, beta=1.0).real).all())
    #: and the merges that subtract a max get the row right.
    assert _merge_guarded(q, k, v, u, th, 1.0, 2) == [1 + 0j, 1.5 + 0j, 2 + 0j, 2.5 + 0j]
    assert _merge_content_shift(q, k, v, u, th, 1.0, 2) == [1 + 0j, 1.5 + 0j, 2 + 0j,
                                                            2.5 + 0j]


# ---------------------------------------------------------------------------
# 7. CLEARED: UNDERFLOW IN THE MERGE, MERGE ORDER, AND float32
# ---------------------------------------------------------------------------

def test_a_block_more_than_746_nats_below_the_global_max_drops_to_exactly_zero():
    """`e^{m_b - M}` underflows to `0` at a gap of 746 nats in float64, and the
    block it deletes was worth at most `n_b * e^{-745} = 4.94e-324` of `l`
    relative -- 307 decades under the 1e-12 bar. Underflow in the merge is
    CLEARED, and the number is why."""
    assert torch.exp(torch.tensor(-745.0, dtype=DT)).item() == 4.9406564584124654e-324
    assert torch.exp(torch.tensor(-746.0, dtype=DT)).item() == 0.0
    assert 8 * 4.9406564584124654e-324 < BAR


@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_merge_order_moves_the_last_bits_but_not_the_1e_12_bar(beta):
    """Non-associativity, measured. Block size changes the summation order, so
    the result is NOT bitwise stable across block sizes -- which matters,
    because `docs/canon/08_ARCHITECTURE.md:635-653` requires bitwise forwards
    for "every deciding forward cell". The gap is 4 decades under the 1e-12
    bar, so the BAR survives and the BITWISE regime does not: block size has to
    join `seed` and `rng_plan` in the manifest."""
    q, k, v, u, th = _live_bed()
    by = {b: _merge_guarded(q, k, v, u, th, beta, b) for b in (1, 4, 8)}
    assert by[1] != by[4], "expected the last bits to move between block 1 and 4"
    worst = max(_rel(by[b], by[1]) for b in (4, 8))
    assert 1e-16 <= worst <= 1e-14, "merge-order relative gap is %.4e" % worst
    assert worst <= BAR


def test_float32_cannot_hold_a_1e_12_bar_and_the_canon_already_said_so():
    """The dtype question, answered with the repo's own replacement bar.
    `docs/canon/08_ARCHITECTURE.md:729` rules that a single `1e-12` on the
    shipped float32 path is `5.96e4` below one rounding and names
    `eps_32 = 1.5e-4`. Measured here: the shipped operator's float32-vs-float64
    relative gap is 2.3x float32 eps, five decades over 1e-12 and three decades
    under 1.5e-4. `ceq/arm_smprime.py:108-109` is the trap -- `_ctype` silently
    drops to `complex64` for a float32 caller, with no warning."""
    q, k, v, u, th = _live_bed()
    for beta in (1.0, 0.5, 0.0):
        o64 = A.operator(q, k, u, th, beta=beta)
        o32 = A.operator(q.float(), k.float(), u.float(), th.float(), beta=beta)
        assert o32.dtype == torch.complex64, o32.dtype
        rel = ((o32.to(torch.complex128) - o64).abs()
               / o64.abs().clamp_min(1e-300)).max().item()
        assert rel > BAR, "float32 met the 1e-12 bar at beta=%s: %.4e" % (beta, rel)
        assert rel < BAR_F32, "float32 missed the canon's own bar: %.4e" % rel
        assert rel < 4 * torch.finfo(torch.float32).eps, rel


def test_the_operator_has_an_absolute_floor_a_relative_bar_cannot_see():
    """The bar's denominator, and this is a repair to the BAR and not the code.
    On a row whose phases cancel, `|out|` falls to the instrument's own
    `polar(1, pi)` residue -- `6.123233995736766e-17`, the number
    `ceq/arm_smprime.py:143` already names as `1.2246e-16` of imaginary part.
    A bar relative to `|out|` there is asking for `6e-29` absolute. Stated
    relative to the row's absolute mass `sum_j |w_ij| |v_j|` it is well posed,
    and the merge-order gap on that row is `2.47e-32`, i.e. `3.1e-33` of mass."""
    s = 8
    q = torch.zeros(s, 1, dtype=DT)
    k = torch.zeros(s, 1, dtype=DT)
    v = torch.ones(s, 1, dtype=DT)
    u = torch.ones(s, dtype=DT)
    th = torch.tensor([0.0] + [math.pi] * (s - 1), dtype=DT)
    got = _merge_guarded(q, k, v, u, th, 1.0, 4)
    assert abs(got[7]) == 6.123233995736767e-17, abs(got[7])
    mass = float((A.path_product(A.magnitude(u))[7] * v[:, 0]).abs().sum())
    assert mass == 8.0, mass
    gap = max(abs(a - b) for a, b in zip(got, _merge_guarded(q, k, v, u, th, 1.0, 1)))
    assert gap <= 2.5e-32, gap
    assert gap / abs(got[7]) > BAR * 1e-4          # relative to |out|: 4.0e-16
    assert gap / mass < BAR                        # relative to mass: 3.1e-33


# ---------------------------------------------------------------------------
# 8. THREE LEVERS ON THE BITS, AND WHAT THE MANIFEST DOES NOT YET CLAIM
# ---------------------------------------------------------------------------
#
# `docs/canon/08_ARCHITECTURE.md:639` (08.22's Statement) requires "bitwise
# ... for every deciding forward cell", against the manifest of SHAPE §A.10 --
# `scale/identity_manifest.CONFIG_FIELDS`, per the Evidence line at `:643`.
# The test above, `test_merge_order_moves_the_last_bits_but_not_the_1e_12_bar`,
# already pins BLOCK SIZE as a lever that moves the last bits and stays under
# BAR. The two levers below isolate THREAD COUNT and MERGE ORDER the same
# way, on the same bed, same beta, same block size, so the three numbers are
# directly comparable:
#
#   lever          moves bits?  measured gap (rel, `_live_bed`, beta=1, block=4)
#   block (1 vs 4)      yes     3.693947489657612e-16
#   threads (1 vs 20)   no      0.0 -- identical, `got == got_one_thread`
#   merge order         yes     2.376607112048788e-16
#
# Thread count is REPORTED as a non-mover, not assumed one: at `s=16, d=4`
# every matmul and reduction here is far below whatever size threshold this
# box's BLAS would parallelize at, so `torch.set_num_threads` has nothing to
# reorder. That is a statement about THIS shape, not a proof threads can
# never move a bit at a shape this file does not measure.

def test_thread_count_and_merge_order_measured_the_same_way_as_block_size():
    """Isolate each lever from the other two: same `_live_bed`, same beta,
    same block=4 grouping. Only the lever named in each branch changes."""
    q, k, v, u, th = _live_bed()
    beta = 1.0
    forward = _merge_guarded(q, k, v, u, th, beta, 4)

    prior = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one_thread = _merge_guarded(q, k, v, u, th, beta, 4)
    finally:
        torch.set_num_threads(prior)
    assert one_thread == forward, (
        "thread count moved a bit at s=16 -- update the table above, gap %.4e"
        % _rel(one_thread, forward))

    reordered = _merge_guarded(q, k, v, u, th, beta, 4, reverse_blocks=True)
    order_gap = _rel(reordered, forward)
    assert reordered != forward, "reversing block order changed nothing"
    assert 1e-17 <= order_gap <= BAR, "merge-order gap is %.4e" % order_gap


def test_config_fields_correctly_excludes_bucketed_merge_execution_params():
    """GREEN, correctly -- the RED predecessor of this test was itself the
    defect, not `scale/identity_manifest.py`. It asked `CONFIG_FIELDS` to
    declare `block_size` and `merge_order` because the lever test above
    measured both moving the last bits of a bucketed `arm_smprime` merge.
    Neither the module nor the canon ever scoped `CONFIG_FIELDS` to cover
    that, so the old test was inventing a requirement, not finding one.

    `scale/identity_manifest.py`'s own module docstring names what the field
    is for: `beta, n_neumann, d_model` and `kind` on top of `_key`'s fields
    -- the arguments that "make a `QuintArm` the cell it is", i.e. the
    construction/training identity of an `m3_quintuple`/`paired_arm` arm.
    Block size and merge order are properties of HOW an already-built arm's
    forward pass is evaluated, not of which arm was trained, and the
    module's own STATUS note says it is "standalone by design" and not yet
    integrated with any cell type -- `arm_smprime` included.

    `docs/canon/08_ARCHITECTURE.md`'s 08.22 Statement -- the one this repo's
    comments cite (`:643`) as the reason `CONFIG_FIELDS` would need to grow,
    and the `CORRECTIONS.md`-gated, exhaustively enumerated mandatory-field
    list for "every deciding forward cell" under Ruling 1's bitwise regime
    (`:639`) -- names dozens of fields, including route-conditional ones
    (`route` `+K, +C`), and names neither `block_size` nor `merge_order`
    anywhere in that list (grepped: zero hits in `docs/canon/`). No
    `CORRECTIONS.md` row adds them either. The old test cited `:639`/`:643`
    for a requirement that is not actually written at either location.

    RED IF: a future `CORRECTIONS.md` row extends 08.22's mandatory list to
    include `block_size`/`merge_order` and `CONFIG_FIELDS` is edited to
    match, without this test being re-scoped to require them instead of
    exclude them."""
    from scale.identity_manifest import CONFIG_FIELDS
    out_of_scope = {"block_size", "merge_order"}
    overreach = out_of_scope & set(CONFIG_FIELDS)
    assert not overreach, (
        "CONFIG_FIELDS gained a bucketed-merge execution field with no "
        "CORRECTIONS.md row sanctioning the scope change: %r"
        % sorted(overreach))


# ---------------------------------------------------------------------------
# 9. MEASURED, NOT ARGUED: THE BLAST RADIUS OF THE :255 CLAMP
# ---------------------------------------------------------------------------
#
# The candidate repair for `ceq/arm_smprime.py:255` is the same clamp the six
# siblings already ship (`ceq/attention.py:189`, `ceq/lm.py:162`,
# `ceq/hopcache.py:72`, `ceq/bench.py:180`, `scale/arm_s.py:128`,
# `scale/foreman_hilbert.py:139`): clamp `Z = mod.sum(-1)` to
# `finfo(dtype).tiny` BEFORE `** beta`, not after. `arm_smprime.py` is another
# lane's file this round (READ-ONLY here), so the candidate is applied to a
# LOCAL copy of the division only -- `numerator()` itself is untouched and
# imported straight from the shipped module -- and compared against
# `A.operator` as it stands.
#
# THE RED RUN, verbatim, against the WISH that the clamp rescues the -800 bed
# to the row `_merge_guarded` already gets right
# (`test_Z_underflows_to_exactly_zero_and_the_operator_returns_nan`):
#
#     AssertionError: [0j, 0j, 0j, 0j]
#     assert [0j, 0j, 0j, 0j] == [(1+0j), (1.5+0j), (2+0j), (2.5+0j)]
#       At index 0 diff: 0j != (1+0j)
#     1 failed in 3.93s
#
# It does not rescue anything: `num` already underflowed to exact `0.0` at
# `exp(-800)` (line 232, upstream of the clamp), so `0.0 / tiny` is `0.0`, not
# `v_i`. The clamp's only measured effect anywhere in this file's beds is
# turning that one `nan` into a silent, wrong `0` -- everywhere the dense
# operator is currently finite (softmax corner, every beta on the live bed,
# the open and closed gate beds, +-80, +800, the 700-nat dead-shift bed), the
# clamp changes NOTHING, bitwise.

def _operator_255_clamped(q, k, u=None, theta=None, *, beta=1.0, qk=1.0,
                          g=1.0, route="product"):
    """The candidate patch, applied locally. Same `numerator()` the shipped
    `operator()` calls; the only change is `Z.clamp_min(tiny)` ahead of
    `** beta`, in the sibling files' order and not `arm_smprime.py`'s own."""
    num, mod = A.numerator(q, k, u, theta, qk=qk, g=g, route=route)
    zb = mod.sum(-1, keepdim=True).clamp_min(torch.finfo(mod.dtype).tiny) ** beta
    return torch.complex(num.real / zb, num.imag / zb)


def _finite_subset_bitwise_unchanged(before, after):
    finite = torch.isfinite(before.real) & torch.isfinite(before.imag)
    if not bool(finite.any()):
        return True, 0
    same = (torch.equal(before.real[finite], after.real[finite])
            and torch.equal(before.imag[finite], after.imag[finite]))
    return same, int(finite.sum())


def test_the_255_clamp_is_bitwise_a_noop_on_the_softmax_corner():
    """(a) `beta=1, qk=1, g=0` -- `#5a`'s softmax corner. `Z` there is a sum of
    causal `exp` terms, `1.0866...` at this seed and never near `tiny`, so the
    clamp cannot fire and the two operators must be bitwise IDENTICAL, not
    merely close: `0.000e+00` against both the shipped operator and the
    literal statement of `#5a` that `test_bind2_softmax_corner_is_bitwise...`
    already pins bitwise."""
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(8, 4, generator=gen, dtype=DT)
    k = torch.randn(8, 4, generator=gen, dtype=DT)
    before = A.operator(q, k, beta=1.0, qk=1.0, g=0.0)
    after = _operator_255_clamped(q, k, beta=1.0, qk=1.0, g=0.0)
    assert torch.equal(before.real, after.real), float((before.real - after.real).abs().max())
    assert torch.equal(before.imag, after.imag)
    w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    up = torch.ones(8, 8, dtype=torch.bool).triu(1)
    e = torch.exp(w.masked_fill(up, A.NEG))
    ref = e / e.sum(-1, keepdim=True)
    assert torch.equal(after.real, ref)
    assert float(e.sum(-1).min()) > 1e300 * torch.finfo(DT).tiny, "Z is nowhere near tiny"


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_the_255_clamp_is_bitwise_a_noop_on_a_normal_draw(beta):
    """(b) The everyday bed, all three `beta` corners the round names. `Z`'s
    row minimum is `0.536...`, twenty-two hundred orders of magnitude above
    `tiny`, so `clamp_min` is a no-op at every entry and both dtype-`beta`
    settings agree bitwise."""
    q, k, v, u, th = _live_bed()
    before = A.operator(q, k, u, th, beta=beta)
    after = _operator_255_clamped(q, k, u, th, beta=beta)
    same, n_finite = _finite_subset_bitwise_unchanged(before, after)
    assert n_finite == before.numel(), "expected every entry finite before the clamp"
    assert same
    assert torch.equal(before.real, after.real) and torch.equal(before.imag, after.imag)


@pytest.mark.parametrize("bed,beta", [
    ("gate_closed4", 1.0), ("gate_closed4", 0.5), ("gate_closed4", 0.0),
    ("gate_open", 1.0), ("logit_80", 1.0), ("logit_800", 1.0),
    ("dead_shift_700", 1.0),
])
def test_the_255_clamp_touches_no_currently_finite_entry_in_this_files_own_beds(bed, beta):
    """Sweep every OTHER bed this file already draws its findings from -- the
    closed and open gate beds, the +80 and +800 logit beds, the 700-nat
    dead-shift bed -- and check the same property one bed at a time rather
    than asserting it once and hoping it generalizes: wherever the dense
    operator is finite now, `clamp_min(tiny)` on `Z` changes not one entry."""
    if bed == "gate_closed4":
        q, k, v, u, th = _gate_bed(closed=4)
    elif bed == "gate_open":
        q, k, v, u, th = _gate_bed(closed=None)
    elif bed == "logit_80":
        q, k, v, u, th = _logit_bed(big=80.0)
    elif bed == "logit_800":
        q, k, v, u, th = _logit_bed(big=800.0)
    else:
        q, k, v, u, th = _dead_shift_bed(big=700.0)
    before = A.operator(q, k, u, th, beta=beta)
    after = _operator_255_clamped(q, k, u, th, beta=beta)
    same, n_finite = _finite_subset_bitwise_unchanged(before, after)
    assert same, "%s beta=%s: clamp moved a currently-finite entry" % (bed, beta)


def test_the_255_clamp_does_not_rescue_the_800_bed_it_only_silences_the_nan():
    """(c) The exact bed `test_Z_underflows_to_exactly_zero_and_the_operator_
    returns_nan` measures: `q = full(-800)`, `k = ones`. There `Z` is `[0.0]*4`
    and the dense operator is `nan` everywhere. The clamp turns `Z` into
    `[tiny]*4` -- but `num` underflowed to exact `0.0` at the SAME `exp(-800)`
    call, one step upstream of line 255, so `num / Z` goes from `0/0 = nan` to
    `0/tiny = 0`. That is not the row `_merge_guarded` gets right on this same
    bed (`[1+0j, 1.5+0j, 2+0j, 2.5+0j]`, `v` itself since the diagonal alone
    survives); it is a second, silent wrong answer, and the RED run above
    is what proves it is not the first one."""
    s = 4
    q = torch.full((s, 1), -800.0, dtype=DT)
    k = torch.ones(s, 1, dtype=DT)
    v = torch.arange(1.0, s + 1, dtype=DT).unsqueeze(-1)
    u, th = torch.ones(s, dtype=DT), torch.zeros(s, dtype=DT)

    num, mod = A.numerator(q, k, u, th)
    assert mod.sum(-1).tolist() == [0.0] * s
    assert num.real.diagonal().tolist() == [0.0] * s, "numerator already 0, upstream of :255"

    before = A.readout(q, k, v, u, th, beta=1.0).flatten().tolist()
    assert all(c != c for c in before), before

    def _readout_255_clamped(q, k, v, u, th, beta):
        a = _operator_255_clamped(q, k, u, th, beta=beta)
        return a @ v.to(a.dtype)

    after = _readout_255_clamped(q, k, v, u, th, 1.0).flatten().tolist()
    assert after == [0j, 0j, 0j, 0j], after
    #: the row a max-subtracting merge already gets right, on this same bed --
    #: the clamp lands on a *different* number than the correct one, not on it.
    rescued = _merge_guarded(q, k, v, u, th, 1.0, 2)
    assert rescued == [1 + 0j, 1.5 + 0j, 2 + 0j, 2.5 + 0j], rescued
    assert after != rescued

    for beta in (0.0, 0.5):
        assert _readout_255_clamped(q, k, v, u, th, beta).flatten().tolist() == [0j] * s
