"""BAR 1-2 for THE FIX at `ceq/arm_smprime.py:236`: an overflowing logit on a
GATE-KILLED entry must not poison the numerator, and the mask that stops it
must be a bitwise no-op everywhere the gate did not kill the entry.

THE BUG, verbatim, before this fix (`ceq/arm_smprime.py:236`):

    return gh * e.to(gh.dtype), rh * e

`rh` (the modulus row `R_ij`) is exactly `0` wherever the gate killed an
entry. If that same entry also carries a masked QK logit past float64's
overflow edge (`~709.78`), `e` there is `inf`, and `0 * inf` is `nan` --
sited on the one entry that was supposed to have been annihilated, not on
a live one. `block_summary`/`read_summary` never take this route (they stay
in log domain, shifted by a per-block max before any exponential), so they
read the same row correctly; this suite checks the dense `numerator` route
now agrees with them, bitwise, once the dead entry stops mattering.

`_dead_shift_bed` is bitwise the fixture `tests/curvature/test_bucket_numerics.py`
uses for the same finding; it is reproduced here rather than imported, so
this suite carries no dependency on `tests/curvature` (a lane this task does
not touch).
"""
from __future__ import annotations

import math

import torch

from ceq import arm_smprime as smp

DT = torch.float64


def _dead_shift_bed(big=700.0):
    """Three tokens, gate closed at index 2 (`u[2] = 0`): `R_2j = 0` for
    `j < 2` (the gate killed them) and `R_22 = 1` (the empty product). The
    `+big` logit is planted at `(2, 0)`, one of the dead entries."""
    q = torch.tensor([[0.0], [0.0], [big]], dtype=DT)
    k = torch.tensor([[1.0], [0.0], [-1.0 / 7.0]], dtype=DT)
    v = torch.tensor([[1.0], [2.0], [3.0]], dtype=DT)
    u = torch.ones(3, dtype=DT)
    u[2] = 0.0
    return q, k, v, u, torch.zeros(3, dtype=DT)


def _live_bed(s=8, seed=1, scale=1.0):
    """No dead gate anywhere -- every entry of the bed is LIVE, so bar 2's
    no-op claim has a bed where the mask should never fire at all."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(s, 4, generator=g, dtype=DT) * scale
    k = torch.randn(s, 4, generator=g, dtype=DT) * scale
    u = torch.rand(s, generator=g, dtype=DT) * 0.9 + 0.1  # never exactly 0
    th = (torch.rand(s, generator=g, dtype=DT) * 2 - 1) * math.pi
    return q, k, u, th


def _scattered_dead_bed(s=12, seed=2, n_dead=3, scale=6.0):
    """A bed with several dead gates scattered through the row, and bigger
    logits than the live bed, at random rather than at one planted index."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(s, 4, generator=g, dtype=DT) * scale
    k = torch.randn(s, 4, generator=g, dtype=DT) * scale
    u = torch.rand(s, generator=g, dtype=DT)
    dead = torch.randperm(s, generator=g)[:n_dead]
    u[dead] = 0.0
    th = (torch.rand(s, generator=g, dtype=DT) * 2 - 1) * math.pi
    return q, k, u, th


def _reference_unmasked_numerator(q, k, u, theta, *, qk=1.0, g=1.0):
    """THE PRE-FIX FORMULA, reproduced independently of `smp.numerator` (which
    this suite is checking), so bar 2's comparison is against the actual old
    arithmetic and not against a second copy of the fix."""
    n = q.shape[-2]
    w = qk * ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]))
    up = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
    e = torch.exp(w.masked_fill(up, smp.NEG))
    gh, rh = smp.hop(u, theta, g=g)
    return gh * e.to(gh.dtype), rh * e


def test_the_masked_numerator_matches_block_summary_on_the_gate_kill_bed():
    """BAR 1. At `big=800`, dense's own numerator overflows: `e[2, 0] =
    exp(800) = inf`, `rh[2, 0] = 0`, and `0 * inf` is `nan` at row 2 -- this
    is the RED the fix removes. After the mask, dense's numerator must read
    finite at every entry, and the readout must match `read_summary` on the
    SAME bed, bitwise, at `[1.0, 1.5, 3.0]`."""
    q, k, v, u, th = _dead_shift_bed(big=800.0)

    mod = smp.numerator(q, k, u, th)[1]
    assert torch.isfinite(mod[2]).all(), (
        "gate-killed entry still poisons the numerator: %r" % mod[2].tolist())

    dense = smp.readout(q, k, v, u, th, beta=1.0).flatten()
    m, l, o, carry = smp.block_summary(q, k, v, u, th, block=1)
    bucketed = smp.read_summary(m, l, o, carry, beta=1.0).flatten()

    assert torch.equal(dense.real, bucketed.real), (
        dense.tolist(), bucketed.tolist())
    assert torch.equal(dense.imag, bucketed.imag), (
        dense.tolist(), bucketed.tolist())
    assert dense.real.tolist() == [1.0, 1.5, 3.0], dense.tolist()


def test_the_mask_is_a_bitwise_no_op_on_every_live_entry(capsys):
    """BAR 2. `torch.where(rh > 0, e, 0)` must return the ORIGINAL `e`
    wherever the gate did not kill the entry -- any change there kills the
    fix. Measured against the pre-fix formula at every LIVE position
    (`rh > 0`), across five beds, at both `g` settings the round names."""
    dead800 = _dead_shift_bed(big=800.0)
    dead700 = _dead_shift_bed(big=700.0)
    dead80 = _dead_shift_bed(big=80.0)
    beds = [
        (dead800[0], dead800[1], dead800[3], dead800[4]),
        (dead700[0], dead700[1], dead700[3], dead700[4]),
        (dead80[0], dead80[1], dead80[3], dead80[4]),
        _live_bed(),
        _scattered_dead_bed(),
    ]
    worst = 0.0
    for q, k, u, th in beds:
        for g in (1.0, 0.0):
            ref_num, ref_mod = _reference_unmasked_numerator(q, k, u, th, g=g)
            new_num, new_mod = smp.numerator(q, k, u, th, g=g)
            _, rh = smp.hop(u, th, g=g)
            live = rh > 0
            for ref, new in ((ref_mod, new_mod), (ref_num.real, new_num.real),
                              (ref_num.imag, new_num.imag)):
                diff = (new[live] - ref[live]).abs()
                if diff.numel():
                    worst = max(worst, float(diff.max()))
    with capsys.disabled():
        print(f"\n  bar 2, worst live-entry diff across {len(beds)} beds x "
              f"g in {{1.0, 0.0}} = {worst:.3e}")
    assert worst == 0.0, worst


def _reference_pre_imag_fix_numerator(q, k, u, theta, *, qk=1.0, g=1.0, route="product"):
    """THE PRE-FIX FORMULA FOR *THIS* FIX -- the return line as it read right
    before the imaginary-lane guard below (`gh * e.to(gh.dtype), rh * e`,
    WITH the dead-entry guard already applied to `e`), reproduced locally so
    bar 3's no-op comparison is against the actual immediately-prior
    arithmetic. `_reference_unmasked_numerator` above is an OLDER reference
    (the formula before the dead-entry guard even existed) and is not reused
    here for that reason -- it has no `live`-masking of `e` at all, so it
    disagrees with the current module everywhere `rh == 0`, which is not
    this fix's question."""
    n = q.shape[-2]
    du, dev = q.dtype, q.device
    w = qk * ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]))
    up = torch.ones(n, n, dtype=torch.bool, device=dev).triu(1)
    e = torch.exp(w.masked_fill(up, smp.NEG))
    gh, rh = smp.hop(u, theta, g=g, route=route)
    live = rh > 0
    e = torch.where(live, e, torch.zeros_like(e))
    return gh * e.to(gh.dtype), rh * e


def _cositing_bed(seed=3, n=8, s=64, d=16, scale=40.0, dead_thresh=0.55):
    """The co-siting bed: a dead entry (`u == 0`) shares a causal cell with an
    overflowing masked logit, at float64's own overflow edge (`~709.78`) and
    NOT `bfloat16`'s (`~88.72`) -- this suite stays on the dense route only.
    Draw order is exactly q, k, u, theta; `dead_thresh` zeros `u < 0.55` AFTER
    the draw, which is why the occupancy below is not a free parameter."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, s, d, generator=g, dtype=DT) * scale
    k = torch.randn(n, s, d, generator=g, dtype=DT) * scale
    u = torch.rand(n, s, generator=g, dtype=DT)
    th = (torch.rand(n, s, generator=g, dtype=DT) * 2 - 1) * math.pi
    u = torch.where(u < dead_thresh, torch.zeros_like(u), u)
    return q, k, u, th


def test_the_complex_multiply_artifact_becomes_inf_not_nan_on_the_cositing_bed(capsys):
    """BAR 1-3 for THE SECOND FIX at `ceq/arm_smprime.py`'s `numerator` return
    line: a LIVE entry (`rh > 0`) whose gate lands on a real value
    (`gh.imag == 0`, true of the diagonal, where the empty path product is
    `1 + 0j`) still poisons itself when the same entry's masked logit
    overflows. `gh * e.to(complex)` runs the general complex multiply and
    evaluates `0 * e` in the imaginary lane, which is `nan` whenever `e` is
    `inf` -- manufactured by the MULTIPLY, after the exponential, on an entry
    the dead-entry guard above never touches because it is not dead. Reproduce
    the mechanism directly: `torch.tensor([1+0j]) * torch.tensor([inf+0j])`
    is `inf+nanj` (checked below, complex128). Max-subtraction acts on the
    exponent and cannot reach a nan made after `exp` returns.

    The fix forces the imaginary lane to `0` wherever `gh.imag == 0` exactly,
    which is a bitwise no-op on every finite entry of both lanes (the
    component-wise product there already equals the general complex
    product), and turns the 173 diagonal `nan`s this bed carries into `inf`
    -- an honest overflow -- without moving the non-finite COUNT, which must
    stay at exactly 339 of 16,640 causal cells. `torch.complex(gh.real * e,
    gh.imag * e)` alone (no `torch.where`) is NOT the fix: it still reads
    `nan` on the diagonal, because `gh.imag` there is exactly `0.0` and
    `0.0 * inf` is `nan` regardless of the component-wise split.

    This is one instance of a bed-independent scalar fact, checked first so a
    changed torch build cannot silently retire the mechanism this test
    exists to pin."""
    bare = (torch.tensor([1 + 0j], dtype=torch.complex128)
            * torch.tensor([float("inf") + 0j], dtype=torch.complex128))
    assert torch.isnan(bare.imag).all() and torch.isinf(bare.real).all(), bare

    q, k, u, th = _cositing_bed()

    #: sanity: a drifted bed must fail loudly, not silently re-baseline.
    assert float((u == 0).double().mean()) == 0.509765625, float((u == 0).double().mean())
    tri = torch.ones(64, 64, dtype=torch.bool).tril(0)
    causal = tri.unsqueeze(0).expand(8, 64, 64)
    assert int(causal.sum()) == 16640, int(causal.sum())

    ref_num, ref_mod = _reference_pre_imag_fix_numerator(q, k, u, th)
    ref_nonfinite = causal & (~torch.isfinite(ref_num.real) | ~torch.isfinite(ref_num.imag))
    ref_nan = causal & (torch.isnan(ref_num.real) | torch.isnan(ref_num.imag))
    _, rh = smp.hop(u, th)
    #: the bed must actually EXERCISE the branch being guarded -- a no-op
    #: proved where the branch never fires proves nothing.
    diag = torch.eye(64, dtype=torch.bool).unsqueeze(0).expand(8, 64, 64)
    guarded = causal & diag & (rh > 0)
    assert int(guarded.sum()) > 0, "bed never reaches a live, gh.imag==0 cell"
    assert int(ref_nan.sum()) == 173, int(ref_nan.sum())
    #: every pre-fix nan sits AT rh == 1.0 (the diagonal) -- none off it, not
    #: that every diagonal cell is nan (most diagonal cells have a finite
    #: masked logit and stay finite even pre-fix).
    assert not bool((ref_nan & ~diag).any()), (
        "a pre-fix nan sits off the diagonal", int((ref_nan & ~diag).sum()))

    new_num, new_mod = smp.numerator(q, k, u, th)
    new_nonfinite = causal & (~torch.isfinite(new_num.real) | ~torch.isfinite(new_num.imag))
    new_nan = causal & (torch.isnan(new_num.real) | torch.isnan(new_num.imag))
    new_inf_only = new_nonfinite & ~new_nan

    with capsys.disabled():
        print(f"\n  co-siting bed: pre-fix nonfinite={int(ref_nonfinite.sum())} "
              f"nan={int(ref_nan.sum())}; post-fix nonfinite="
              f"{int(new_nonfinite.sum())} nan={int(new_nan.sum())} "
              f"inf-only={int(new_inf_only.sum())}; guarded live diagonal "
              f"cells exercised={int(guarded.sum())}")

    #: BAR 1: the non-finite COUNT does not move.
    assert int(new_nonfinite.sum()) == 339 == int(ref_nonfinite.sum())
    #: BAR 2: every one of those 339 is inf-only now; zero are nan.
    assert int(new_nan.sum()) == 0, int(new_nan.sum())
    assert int(new_inf_only.sum()) == 339, int(new_inf_only.sum())

    #: BAR 3: bitwise no-op on every causal cell finite in BOTH lanes pre-fix.
    ref_finite = causal & torch.isfinite(ref_num.real) & torch.isfinite(ref_num.imag)
    assert bool(torch.equal(new_num.real[ref_finite], ref_num.real[ref_finite]))
    assert bool(torch.equal(new_num.imag[ref_finite], ref_num.imag[ref_finite]))
    assert bool(torch.equal(new_mod, ref_mod))

    #: BAR 4: the dead-shift bed is untouched by this edit.
    dead = _dead_shift_bed(big=700.0)
    dense = smp.readout(dead[0], dead[1], dead[2], dead[3], dead[4], beta=1.0).flatten()
    assert dense.real.tolist() == [1.0, 1.5, 3.0], dense.tolist()


def test_operators_own_larger_nan_problem_is_unmoved_by_the_numerator_fix(capsys):
    """WHAT THE FIX DOES NOT REACH, recorded so the repair is not oversold.
    One call downstream of `numerator`, `operator()` divides by
    `Z_i = mod.sum(-1)`, and on 258 of this bed's 512 rows `Z_i` is itself
    `inf` (the row's own overflowing logits, summed in the LINEAR domain).
    `num / inf` is `inf / inf`, which is `nan` for a reason this edit does
    not touch: an INF/INF OVERFLOW, not a complex-multiply artifact, and one
    max-subtraction would fix -- which is exactly what `block_summary` /
    `read_summary` already do by staying in log domain instead of calling
    `numerator` at all. The count is measured identical before and after the
    `numerator` edit, because `operator()`'s division reads `mod` (`rh * e`),
    a quantity this edit never changes, and the edit's only touch is the
    imaginary lane of `num` on entries that are `0 * 0` regardless. The 173
    diagonal cells this edit repairs are a strict 4.9% subset of this 3,517;
    the novelty claim (a nan made by the multiply, after `exp`) holds only
    for the 173, and the repair claim does not extend past `numerator`."""
    q, k, u, th = _cositing_bed()
    mod = smp.numerator(q, k, u, th)[1]
    rows_inf = torch.isinf(mod.sum(-1))
    with capsys.disabled():
        print(f"\n  rows with Z_i = mod.sum(-1) == inf: {int(rows_inf.sum())} of "
              f"{mod.sum(-1).numel()}")
    assert int(rows_inf.sum()) == 258, int(rows_inf.sum())

    op = smp.operator(q, k, u, th)
    tri = torch.ones(64, 64, dtype=torch.bool).tril(0)
    causal = tri.unsqueeze(0).expand(8, 64, 64)
    nan = causal & (torch.isnan(op.real) | torch.isnan(op.imag))
    with capsys.disabled():
        print(f"  operator() nan causal cells: {int(nan.sum())} of {int(causal.sum())}")
    assert int(nan.sum()) == 3517, int(nan.sum())


def test_gradient_finiteness_stays_finite_at_the_measured_dead_occupancy(capsys):
    """BAR 4. `self.g` is a plain `nn.Parameter`, never frozen anywhere in
    `ceq/`, `ceqjepa/` or `scale/`. At the module's own default shape
    (`n=512, s=64, seed=0`), `frac_positions_at_m_zero` reads
    `0.555816650390625` -- over half of every trained row is in the dead
    set -- and the 40-step probe must still see zero non-finite gradient
    entries there after the fix, with no freeze added to reach it."""
    r = smp.gradient_finiteness(dtype=DT, device=None)
    with capsys.disabled():
        print(f"\n  gradient_finiteness at frac_positions_at_m_zero="
              f"{r['frac_positions_at_m_zero']} -> first_non_finite="
              f"{r['first_non_finite']}")
    assert r["frac_positions_at_m_zero"] > 0.5, r
    assert r["first_non_finite"] is None, r
