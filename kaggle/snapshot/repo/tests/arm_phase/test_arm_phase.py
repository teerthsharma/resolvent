"""ARM PHASE -- five binds, five planted negatives, and the diagnostic control.

The arm is `CEQ_V15_3_DELTA.md` X36:

    a_j = m_j * exp(i theta_j),  m_j = clamp(u_j, 0, 1)  CLOSED
    C_j = sum log m  +  i sum theta                      prefix-PHASE
    l_ij = q_ij - Re C_j + s_j        (real, the softmax row)
    A_ij = softmax_j(l_i.) * exp(i (Im C_i - Im C_j))    (complex, the twist)

THE FAILURE THIS REPLACES. `V15_R1.md` section 7: three of eight seeds trained
ARM PL's gate to `a_hat_max` of `20.31`, `49.66` and `285.07`, above `1`, where
the `1/(1-a)` value rescale is undefined; the five crossing seeds sat at
`[1.10, 1.51]`. BIND 1 is the statement that the same parameter values cannot
produce `|a| > 1` here, and it is a PROPERTY test over a wide range rather than
three hand-picked points, because three points is what `V-24` calls an answer
key.

EVERY BIND SHIPS ITS REJECTION REGION (MISTAKES.md V-24). Five binds, five
deliberate mutilations, each of which must fail at `O(1)`. A bind with no
demonstrated rejection region does not ship.

WHAT IS NOT CLAIMED. The standard-attention identity is at
`m = 1, theta = 0, s = 0` -- i.e. `a = 1`, the closed cap's upper endpoint --
and its content is `V-24`'s wording, not more: **the modification enters only
through the key logit (additively) and through a unimodular twist, and both
vanish at that setting.** The label bind is at `|a| < 1` and FAILS on the band
`m = 1`; that is measured in `test_bind5_...on_the_closed_band` and it is the
cost this construction pays.

THE REFERENCE IS `ceq.lm.Attention("softmax_x").operator` AND NOT
`ceq.bench._softmax_operator`. `bench` masks with `tril(-1)`: strictly causal,
diagonal excluded, so `A_ii = 0` and row 0 is empty. This arm needs the
diagonal (`P_ii = 1`). Binding the parity claim to one object and the label
claim to another is `V-24`'s shared-carrier defect, and ARM PL's node already
found it; `bench` is therefore not used here at all.
"""
from __future__ import annotations

import functools
import math
import os

import pytest
import torch

from ceq import arm_phase, lm
from scale import negation_scope as ns

DT = torch.float64
S = 8                       # positions 0..S; 0 is the BOS sink
SEED = 15                   # the fork probe's draw, so the numbers are comparable

#: THE DEVICE THE WHOLE SUITE RUNS ON. `ARM_PHASE_DEVICE=cuda python -m pytest
#: tests/arm_phase` re-runs the IDENTICAL suite on the GPU -- same test ids,
#: same count, so the two runs diff line for line. Default `cpu`, so a bare
#: `pytest` is the run that produced `V15_ARM_PHASE.md`.
#:
#: An env var and not a `params=` fixture ON PURPOSE: parametrizing would rename
#: every test id and double the collected count, and both `V15_ARM_PHASE.md` and
#: the loop-level collection guards quote these ids.
#:
#: EVERY DRAW IS MADE ON THE HOST AND THEN MOVED. `torch.Generator()` is a CPU
#: generator and a cuda generator draws different bytes; the corpus builders in
#: `scale/negation_scope.py` already construct-then-`.to()` for that reason, and
#: `V16_BAR_RECERT.md` 6.1 verified the draw byte-identical on both devices. So
#: `.to(DEV)` always sits AFTER the generator, never inside it -- otherwise this
#: file would be measuring a different corpus rather than a different device.
DEV = torch.device(os.environ.get("ARM_PHASE_DEVICE", "cpu"))
ON_CUDA = DEV.type == "cuda"

#: `V15_R1.md` section 7's own `a_hat_max` column, all eight seeds. These are
#: gate magnitudes a TRAINED ARM PL actually produced, five crossing and three
#: divergent. Fed to this arm's parametrization they must all read `|a| <= 1`.
R1_A_HAT_MAX = (1.4107265932952324, 1.2868, 20.3090, 49.6613,
                1.4390, 1.5051, 1.1029, 285.0719)


def _qk(n: int = S + 1, d: int = 4, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g, dtype=DT).to(DEV),
            torch.randn(n, d, generator=g, dtype=DT).to(DEV),
            torch.randn(n, 2, generator=g, dtype=DT).to(DEV))


# ================================================================== BIND 1
# |a| <= 1 BY CONSTRUCTION -- the bind the whole delta rests on.

def test_bind1_the_magnitude_cap_holds_over_a_wide_parameter_range():
    """A PROPERTY test, not three points: `2e5` draws with `|u|` log-uniform
    over `[1e-12, 1e12]` at both signs and `theta` uniform on `[-1e3, 1e3]`.

    The range is chosen to CONTAIN the failure: `V15_R1.md`'s divergent seeds
    reached `a_hat_max = 285.07`, which is `2.85e2`, ten orders inside the
    upper end of this sweep.
    """
    g = torch.Generator().manual_seed(36)
    n = 200_000
    u = (torch.rand(n, generator=g, dtype=DT) * 48.0 - 24.0).exp()   # 1e-10..1e10
    u = u * torch.where(torch.rand(n, generator=g, dtype=DT) < 0.5, -1.0, 1.0)
    th = (torch.rand(n, generator=g, dtype=DT) * 2000.0) - 1000.0
    u, th = u.to(DEV), th.to(DEV)
    mod = arm_phase.gate(u, th).abs()
    assert float(mod.max()) <= 1.0, f"worst |a| = {float(mod.max())!r}"
    assert bool(torch.isfinite(mod).all())


def test_bind1_holds_at_the_exact_gate_magnitudes_r1_measured():
    """The eight `a_hat_max` values `V15_R1.md` section 7 printed, including the
    `285.07` that put the `1/(1-a)` rescale out of its domain."""
    u = torch.tensor(R1_A_HAT_MAX, dtype=DT, device=DEV)
    th = torch.zeros_like(u)
    assert float(arm_phase.gate(u, th).abs().max()) <= 1.0


def test_bind1_holds_at_the_endpoints_of_the_float_line():
    """Infinities and the largest finite doubles included, because a cap that
    is only tested on ordinary numbers is a cap on ordinary numbers."""
    u = torch.tensor([-math.inf, -1e308, -1.0, 0.0, 1.0, 1e308, math.inf],
                     dtype=DT, device=DEV)
    th = torch.zeros_like(u)
    m = arm_phase.magnitude(u)
    assert float(m.max()) <= 1.0 and float(m.min()) >= 0.0
    assert bool(torch.isfinite(arm_phase.gate(u, th)).all())


def test_bind1_zero_and_one_are_ATTAINED_and_not_approached():
    """The delta's word is CLOSED: `0` and `1` are attainable VALUES.

    The contrast is LRU (Orvieto 2023), `lambda = exp(-exp nu + i theta)`, whose
    magnitude is OPEN -- it reaches neither endpoint at any ordinary `nu`. Both
    are evaluated on the same parameter grid so the difference is measured and
    not asserted.
    """
    u = torch.linspace(-2.0, 2.0, 401, dtype=DT, device=DEV)
    m = arm_phase.magnitude(u)
    assert float(m.max()) == 1.0 and float(m.min()) == 0.0
    lru = torch.exp(-torch.exp(u))                      # LRU's open magnitude
    assert float(lru.max()) < 1.0 and float(lru.min()) > 0.0


def test_bind1_planted_negative_dropping_the_cap_restores_the_r1_divergence():
    """PLANTED NEGATIVE 1. `cap=False` is the open magnitude ARM PL had. The
    bind must fail at the size R1 measured, not merely fail."""
    u = torch.tensor(R1_A_HAT_MAX, dtype=DT, device=DEV)
    th = torch.zeros_like(u)
    worst = float(arm_phase.gate(u, th, cap=False).abs().max())
    assert worst > 1.0, "the planted negative did not fire"
    assert worst == pytest.approx(285.0719, abs=1e-3), worst


# ================================================================== BIND 2
# |path product| = 1 EXACTLY on the band.

def test_bind2_the_band_path_product_has_modulus_one_over_1e4_phases():
    """`CEQ_V15_3_DELTA.md`: `[RUN: 1e4 phases -> modulus 1.000000000000]`.

    Two routes that share no code: the prefix-PHASE construction
    `exp(C_i - C_j)`, and a direct cumulative complex product. Both are read,
    over every prefix and not only the endpoint.
    """
    r = arm_phase.band_modulus(n=10_000, seed=SEED, device=DEV)
    assert r["prefix_route"] == pytest.approx(1.0, abs=1e-12)
    assert r["product_route"] == pytest.approx(1.0, abs=1e-12)
    assert r["max_abs_dev"] < 1e-12
    #: the bound is `<= 1`, and it survives float64: every one-ulp miss over the
    #: 1e4 prefixes is a miss DOWNWARD, so no prefix reads above the band.
    assert r["n_above_one"] == 0
    #: `n_exactly_one` IS NOT A BIND, AND THE DEVICE SPLIT IS WHAT PROVES IT.
    #: The four assertions above ARE the bind and all four hold unchanged on
    #: both devices. This line is a CENSUS of how many of the 1e4 prefixes land
    #: on `1.0` with no ulp of error at all, which is a property of the
    #: SUMMATION ORDER inside `cumsum` -- sequential on cpu, a parallel scan on
    #: cuda -- and not of the construction. Measured, exact on both, neither
    #: relaxed: cpu `9767`, cuda `7713` of 10000. `n_above_one == 0` on both, so
    #: the extra 2054 prefixes are still misses DOWNWARD and the `<= 1` bound
    #: the bind actually claims is untouched. Pinned per device rather than
    #: widened to a range: a range would stop being able to say which device
    #: moved.
    assert r["n_exactly_one"] == (7713 if ON_CUDA else 9767), r["n_exactly_one"]


def test_bind2_the_modulus_is_bounded_by_one_off_the_band_too():
    """`|path product| <= 1` by construction everywhere, `= 1` only on the
    band. The inequality is the half `Lean #6` gave; the equality is `#16`."""
    g = torch.Generator().manual_seed(2)
    for _ in range(5):
        u = (torch.rand(64, generator=g, dtype=DT) * 3.0 - 1.0).to(DEV)  # spans the cap
        th = (torch.randn(64, generator=g, dtype=DT) * 4.0).to(DEV)
        c = arm_phase.scan_phase(arm_phase.magnitude(u), th)
        assert float(torch.exp(c.real).max()) <= 1.0 + 1e-15


def test_bind2_planted_negative_a_magnitude_leak_kills_the_modulus():
    """PLANTED NEGATIVE 2. `m = 0.9` instead of `1` on the band -- the smallest
    possible departure from the cap's upper endpoint. Over `1e4` positions the
    modulus must collapse, i.e. the bind must be a statement about `m` and not
    a statement about `exp(i theta)` alone."""
    r = arm_phase.band_modulus(n=10_000, seed=SEED, band_magnitude=0.9, device=DEV)
    assert abs(r["prefix_route"] - 1.0) == pytest.approx(1.0, abs=1e-9)


# ================================================================== BIND 3
# parity mask = Z2 winding.

def test_bind3_theta_in_zero_pi_reproduces_the_signed_parity_mask_exactly():
    """`lean/CEQ/V15.lean` `parity_sign`:
    `chi(pscan p i - pscan p j) = prod_{k=j+1}^{i} chi(p k)`, `chi` the sign
    character of `ZMod 2`. Setting `theta_k = pi * p_k` makes the Z2 winding
    `(Phi_i - Phi_j)/pi` carry exactly that XOR, and `(-1)^winding` is the mask
    BITWISE -- `torch.equal`, not a tolerance.
    """
    g = torch.Generator().manual_seed(3)
    p = torch.randint(0, 2, (64,), generator=g).to(DEV)
    th = math.pi * p.to(DT)
    w = arm_phase.winding_matrix(th)
    assert float(w["residual"]) < 1e-12, "non-integer winding is an instrument defect"
    assert torch.equal(w["parity"], arm_phase.parity_sign_mask(p))


def test_bind3_the_continuous_exponential_agrees_to_machine_precision():
    """The same statement read through `exp(i(Phi_i - Phi_j))` rather than
    through the rounded winding. It is NOT exact -- `cumsum` of `k` copies of
    `pi` is not `fl(k*pi)` -- and the size of that gap is reported rather than
    hidden behind the integer route."""
    g = torch.Generator().manual_seed(4)
    p = torch.randint(0, 2, (64,), generator=g).to(DEV)
    th = math.pi * p.to(DT)
    twist = arm_phase.phase_factor(th)
    mask = arm_phase.parity_sign_mask(p).to(twist.dtype)
    assert float((twist - mask).abs().max()) < 1e-13


def test_bind3_the_per_instance_winding_is_an_integer():
    """`CEQ_V15_3_DELTA.md` X37 (a): the winding is printed per instance and is
    an INTEGER; a non-integer reading is an instrument defect."""
    g = torch.Generator().manual_seed(5)
    p = torch.randint(0, 2, (128,), generator=g).to(DEV)
    w, res = arm_phase.winding(math.pi * p.to(DT))
    assert res < 1e-12
    assert torch.equal(w, torch.cumsum(p, 0))


def test_bind3_planted_negative_a_quarter_turn_breaks_the_winding():
    """PLANTED NEGATIVE 3. `theta in {0, pi/2}` is a `Z4` gate, not a `Z2` one.
    The winding stops being an integer and the mask stops agreeing."""
    g = torch.Generator().manual_seed(6)
    p = torch.randint(0, 2, (64,), generator=g).to(DEV)
    th = (math.pi / 2.0) * p.to(DT)
    w = arm_phase.winding_matrix(th)
    assert float(w["residual"]) > 0.4, "quarter turn read as an integer winding"
    disagree = (w["parity"] != arm_phase.parity_sign_mask(p)).double().mean()
    assert float(disagree) > 0.1, f"mask still agreed on {1 - float(disagree):.3f}"


# ================================================================== BIND 4
# the standard-attention identity.

def test_bind4_the_identity_setting_is_bitwise_standard_attention():
    """`m = 1, theta = 0, s = 0`: the key bias is exactly `0.0` and the twist is
    exactly `1 + 0j`, so the operator is BITWISE `lm.Attention("softmax_x")`.

    Content, in V-24's words and no wider: the modification enters only through
    the key logit, additively, and through a unimodular twist, and both vanish
    at this setting. `m = 1` is `a = 1`, the cap's upper endpoint -- the same
    boundary ARM PL's `g = 0` sat on.
    """
    q, k, v = _qk()
    ref = lm.Attention("softmax_x", 4, 1).operator(q, k)
    op = arm_phase.operator(q, k,
                            *arm_phase.identity_setting(q.shape[0], DT, device=DEV))
    assert torch.equal(op.real, ref)
    assert torch.equal(op.imag, torch.zeros_like(op.imag))
    assert torch.equal(op.real @ v, ref @ v)


def test_bind4_the_complex_readout_costs_one_ulp_and_the_size_is_named():
    """WHAT THE BIND DOES NOT REACH, measured rather than skipped.

    The OPERATOR is bitwise. `readout` contracts it against `V` with a COMPLEX
    gemm, which is a different reduction than the real one -- `Re(sum a_ij v_j)`
    accumulates through `Re*Re - Im*Im` even when every `Im` is `0.0` -- so the
    read-out is NOT bitwise. The gap is one ulp and it is named here so nobody
    later quotes "bitwise" for the wrong tensor.
    """
    q, k, v = _qk()
    ref = lm.Attention("softmax_x", 4, 1).operator(q, k)
    out = arm_phase.readout(q, k, v,
                            *arm_phase.identity_setting(q.shape[0], DT, device=DEV))
    gap = float((out.real - (ref @ v)).abs().max())
    #: THE DEVICE-INDEPENDENT HALF FIRST: whatever the gemm does, it costs AT
    #: MOST the one ulp the docstring names. Neither device may exceed the
    #: published figure, and that assertion is not per-device.
    assert gap <= 1.1102230246251565e-16, gap
    #: AND THE EXACT VALUE, PER DEVICE, NEITHER RELAXED. cpu
    #: `1.1102230246251565e-16`, the published number, untouched. cuda `0.0`:
    #: cuBLAS returns this read-out BITWISE, so on cuda the gap this test exists
    #: to name does not open at all. That is the device being STRICTER than the
    #: published claim, so the claim is preserved and the cuda row is pinned at
    #: `0.0` rather than admitted by a widened tolerance.
    assert gap == pytest.approx(0.0 if ON_CUDA else 1.1102230246251565e-16,
                                abs=1e-18), gap


def test_bind4_holds_for_the_drop_in_module_and_not_only_the_function():
    """The same bind on `ArmPhase`, the object a harness would actually hold --
    a parity bind on a free function the shipped module does not dispatch
    through is a reading of a non-shipped operator."""
    arm = arm_phase.ArmPhase(s=S + 1, d_model=4, hidden=8).identity_heads().to(DEV)
    x = torch.randn(3, S + 1, 4, generator=torch.Generator().manual_seed(5)).to(DEV)
    u, th, s = arm.heads(x)
    assert torch.equal(arm_phase.magnitude(u), torch.ones_like(u))
    assert torch.equal(th, torch.zeros_like(th))
    assert torch.equal(s, torch.zeros_like(s))
    op = arm._operator(arm.wq(x), arm.wk(x), u, th, s)
    assert torch.equal(op.real, lm.Attention("softmax_x", 4, 1).operator(arm.wq(x),
                                                                        arm.wk(x)))
    assert arm(x).shape == (3,)


def test_bind4_planted_negative_every_head_moves_the_operator():
    """PLANTED NEGATIVE 4. Three mutilations of the identity setting, one per
    head. Without an occupied rejection region this bind is a V-24."""
    q, k, _ = _qk()
    n = q.shape[0]
    ref = lm.Attention("softmax_x", 4, 1).operator(q, k)
    gen = torch.Generator().manual_seed(7)
    one = torch.ones(n, dtype=DT, device=DEV)
    zero = torch.zeros(n, dtype=DT, device=DEV)
    cases = {
        "m only": ((torch.rand(n, generator=gen, dtype=DT) * 0.8).to(DEV), zero, zero),
        "theta only": (one, torch.randn(n, generator=gen, dtype=DT).to(DEV), zero),
        "s only": (one, zero, torch.randn(n, generator=gen, dtype=DT).to(DEV)),
    }
    for name, (u, th, s) in cases.items():
        moved = float((arm_phase.operator(q, k, u, th, s) - ref).abs().max())
        assert moved > 1e-3, f"{name} did not move the operator: {moved}"


def test_the_modulus_row_never_leaves_the_softmax_class():
    """V-24's class-closure test at a RANDOM theta, not only at the identity
    point: the row of MODULI sums to 1 and is strictly positive. The complex row
    does not sum to 1 -- the twist is the parity carrier, not a probability --
    and that is stated rather than quietly dropped."""
    q, k, _ = _qk()
    gen = torch.Generator().manual_seed(11)
    tri = torch.ones(S + 1, S + 1, dtype=torch.bool, device=DEV).tril(0)
    for _ in range(3):
        a = arm_phase.operator(
            q, k,
            torch.rand(S + 1, generator=gen, dtype=DT).to(DEV),
            torch.randn(S + 1, generator=gen, dtype=DT).to(DEV),
            (torch.randn(S + 1, generator=gen, dtype=DT) * 0.7).to(DEV))
        assert (a.abs().sum(-1) - 1.0).abs().max().item() < 1e-14
        assert a.abs()[tri].min().item() > 0.0


# ================================================================== BIND 5
# the chain path-product label.

def test_bind5_the_oracle_setting_reproduces_the_complex_chain_label():
    """(L) on the OPEN interior `|a| in (0.15, 0.85)`: `O_i = y_i` for the
    COMPLEX recurrence `y_i = a_i y_{i-1} + b_i`. The contract bar is `1e-6`;
    ARM PL reached `6.6613381477509392e-16` on the real chain at this shape."""
    a, b = arm_phase.draw(seed=SEED, s=S, device=DEV)
    cell = arm_phase.label_cell(a, b, seed=SEED)
    assert cell["residual"] <= 1e-6
    assert cell["residual"] < 1e-14
    #: BITWISE THE PUBLISHED FIGURE ON BOTH DEVICES -- a stronger reading than
    #: either bar above, which is why it is asserted rather than described.
    assert cell["residual"] == 9.155133597044475e-16, repr(cell["residual"])
    #: and the cell says where it was measured. Until V16 this field was the
    #: literal `"cpu"` (`V16_BAR_RECERT.md` F4).
    assert cell["device"] == DEV.type


def test_bind5_the_normalizer_telescopes_to_one():
    """`Z_i = 1` EXACTLY is the mechanism, and it is a statement about the
    MAGNITUDES only -- the twist has modulus 1 and cannot move it."""
    a, b = arm_phase.draw(seed=SEED, s=S, device=DEV)
    u, th, s, _ = arm_phase.oracle_heads(a, b)
    z = arm_phase.normalizer(arm_phase.magnitude(u), s)
    assert (z - 1.0).abs().max().item() < 1e-15


def test_bind5_the_label_bind_FAILS_on_the_closed_band():
    """THE COST, MEASURED. On BED-M's band `|a| = 1`, so the value rescale
    `b/(1 - m)` is `inf` and the key bias `log(1 - m)` is `-inf`. The gate
    itself is now exactly representable -- `m in {0, 1}`, `theta in {0, pi}` --
    where ARM PL's `g = log a` read `nan/-inf/inf` (`V15_R1.md` section 1). The
    VALUE path is what stays out of range, and this test is the receipt.
    """
    a, b = arm_phase.band_draw(seed=SEED, s=S, device=DEV)
    cell = arm_phase.label_cell(a, b, seed=SEED)
    assert not (cell["residual"] <= 1e-6), "the band cell silently passed"
    assert math.isnan(cell["residual"])          # inf value against a 0 weight
    assert not math.isfinite(cell["dyn_range_bound"])
    assert not math.isfinite(cell["v_max"])
    _, _, s, v = arm_phase.oracle_heads(a, b)
    assert float(s[1]) == -math.inf and not math.isfinite(float(v[1]))


def test_bind5_the_cost_is_the_VALUE_path_and_the_ladder_prices_it(capsys):
    """The label bind survives to `m = 1 - 1e-12` at machine precision and dies
    only AT the endpoint, with `v_max` tracking `1/(1 - m)` the whole way. So
    the cost is not "the phase arm cannot do the label" -- it is section 8.4
    cost 6, unchanged: the value rescale, priced, at the one magnitude the
    closed cap made attainable."""
    rows = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-9, 1e-12):
        c = arm_phase.label_cell(*arm_phase.band_draw(seed=SEED, s=S,
                                                      magnitude=1.0 - eps,
                                                      device=DEV),
                                 seed=SEED)
        rows.append((eps, c["residual"], c["v_max"], c["dyn_range_bound"]))
        assert c["residual"] < 1e-13, rows[-1]
        assert c["v_max"] > 0.5 * c["dyn_range_bound"]
    with capsys.disabled():
        print("\n  m = 1 - eps     residual      v_max        1/(1-m)")
        for eps, r, vm, dr in rows:
            print(f"  eps={eps:<8g}  {r:.6e}  {vm:.6e}  {dr:.6e}")


@pytest.mark.parametrize("mutation", ["drop_key_bias", "drop_value_rescale",
                                      "drop_bos_sink", "half_key_bias",
                                      "drop_phase"])
def test_bind5_planted_negatives_all_fire(mutation):
    """PLANTED NEGATIVE 5. ARM PL's four mutilations plus `drop_phase`, which is
    this arm's own: deleting the twist leaves a magnitude-only chain, and the
    complex label is not reproduced by any real one."""
    a, b = arm_phase.draw(seed=SEED, s=S, device=DEV)
    r = arm_phase.label_cell(a, b, mutation=mutation, seed=SEED)["residual"]
    assert r > 1e-6, f"plant {mutation!r} did not fire: {r}"


def test_the_mutation_list_is_the_settled_one():
    assert arm_phase.MUTATIONS == ("none", "drop_key_bias", "drop_value_rescale",
                                   "drop_bos_sink", "half_key_bias", "drop_phase")


def test_the_prefix_route_is_undefined_once_a_magnitude_hits_zero():
    """A LIMIT of prefix-PHASE, stated because `m = 0` is attainable by design.
    Past a zero gate `Re C = -inf` at both endpoints and `C_i - C_j` is `nan`.
    The cumulative-product route returns the true `0`. Reported, not repaired:
    ARM PL has the identical hole at `a = 0` (`g = log 0 = -inf`)."""
    m = torch.tensor([1.0, 0.5, 0.0, 0.5], dtype=DT, device=DEV)
    th = torch.zeros(4, dtype=DT, device=DEV)
    c = arm_phase.scan_phase(m, th)
    assert torch.isnan(c[3] - c[2])
    assert float(torch.cumprod(m, 0)[3]) == 0.0


# ================================================== THE IDENTITY MANIFEST

def test_the_manifest_hash_moves_when_a_planted_negative_moves_the_cell():
    a, b = arm_phase.draw(seed=SEED, s=S, device=DEV)
    honest = arm_phase.label_cell(a, b, seed=SEED)["manifest"]
    for f in arm_phase.PHASE_FIELDS:
        assert f in honest["pl_values"], f"manifest does not carry {f}"
    assert "device" in honest["values"]
    for mutation in arm_phase.MUTATIONS[1:]:
        other = arm_phase.label_cell(a, b, mutation=mutation,
                                     seed=SEED)["manifest"]["hash"]
        assert other != honest["hash"], f"{mutation} left the hash unmoved"


# =========================================== THE DIAGNOSTIC, AND ITS CONTROLS
# MISTAKES.md M-18 and its correction: a diagnostic is run on the CORPUS ALONE
# and against a ZERO-STEP control BEFORE it is trusted. `V15_R1.md` section 6
# found the prescribed `sign(a)` probe saturated at init (`p = 0.9178` trained,
# `p0 = 0.9437` at zero steps -- HIGHER). The instrument that discriminated was
# the gate `R^2`. This node carries that one forward and nothing else, and it
# TRAINS NOTHING (L-LEAN): only the two controls are read here.

def _probe(f_tr, y_tr, f_ev, y_ev):
    """`scripts/v15_r1.py::probe`, same form: least squares fit on train,
    scored out of sample on eval."""
    a_tr = torch.cat([torch.ones(f_tr.shape[0], 1, dtype=DT, device=f_tr.device),
                      f_tr.double()], 1)
    a_ev = torch.cat([torch.ones(f_ev.shape[0], 1, dtype=DT, device=f_ev.device),
                      f_ev.double()], 1)
    w = torch.linalg.lstsq(a_tr, y_tr.double().unsqueeze(1)).solution
    pred = (a_ev @ w).squeeze(1)
    yv = y_ev.double()
    sst = float(((yv - yv.mean()) ** 2).sum())
    return (float("nan") if sst == 0.0 else
            1.0 - float(((yv - pred) ** 2).sum()) / sst)


def test_the_gate_r2_instrument_is_read_on_the_corpus_alone_and_at_zero_steps(capsys):
    """The two readings `V15_R1.md`'s LIMITS say must precede trusting any
    diagnostic. Nothing is trained: (a) is a probe on the corpus's own input
    channels with NO arm in the loop, (b) is the same probe on an UNTRAINED
    `ArmPhase`'s gate. If (a) is already ~1.0 the target is visible in the
    input and a trained gain is not evidence about the arm."""
    batch = functools.partial(ns.make_equilibrium_batch, t_star=2)
    s, d, d_model, t_star = 64, 24, 16, 2
    live = list(range(s - t_star, s))
    x_tr, _, _, _ = batch(512, s, d, d_model=d_model, seed=0, device=DEV)
    x_ev, _, _, _ = batch(512, s, d, d_model=d_model, seed=12345, device=DEV)
    a_tr = x_tr[:, live, ns.CH_DRIVE].reshape(-1)
    a_ev = x_ev[:, live, ns.CH_DRIVE].reshape(-1)

    corpus_r2 = _probe(x_tr[:, live, :].reshape(-1, d_model), a_tr,
                       x_ev[:, live, :].reshape(-1, d_model), a_ev)

    zero_step = []
    for seed in range(8):
        torch.manual_seed(seed)
        arm = arm_phase.ArmPhase(s, d_model=d_model).to(DEV)
        zero_step.append(_probe(arm.gate_feature(x_tr, live), a_tr,
                                arm.gate_feature(x_ev, live), a_ev))
    z = sum(zero_step) / len(zero_step)
    with capsys.disabled():
        print(f"\n  gate R^2, CORPUS ALONE (no arm)  = {corpus_r2:.6f}")
        print(f"  gate R^2, ZERO-STEP ArmPhase x8  = {z:.6f}"
              f"   per seed {[round(v, 6) for v in zero_step]}")
    assert corpus_r2 > 0.99, corpus_r2      # the target IS the input (M-18)
    assert z < corpus_r2                    # the arm's gate is not saturated
    assert all(math.isfinite(v) for v in zero_step)
