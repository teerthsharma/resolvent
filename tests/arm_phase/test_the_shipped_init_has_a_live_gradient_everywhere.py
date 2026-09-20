"""The same guard as `tests/curvature/test_the_shipped_init_has_a_live_gradient_everywhere_smprime.py`, pointed at ARM PHASE.

`MISTAKES.md` V-29 closes by naming this module as the LIVE half of the defect:
*"THE DEFECT IS LIVE IN A SIBLING ARM, unfixed at the time of writing."* The
sibling's guard is per-PARAMETER and asserts `|grad| > 0`, not `isfinite`,
because zero is finite and that is exactly how `arm_smprime.gradient_finiteness`
passed a gradient that was identically `0.0`.

THREE THINGS THIS FILE MEASURES THAT THE SIBLING'S DOES NOT.

1. THE BATCH IS THE CORPUS, NOT A DRAW. The sibling censuses on `torch.randn`.
   A gradient that is zero for a STRUCTURAL reason is zero on any input, but a
   guard that reports on synthetic data cannot say the arm is dead *on the
   corpus it is scored on*, so the census here runs `make_equilibrium_batch`
   at `t* = 2` -- BED-M, the cell `V15_R1.md` section 7 ran and the cell
   `scripts/v15_r1.py` trains this arm at.

2. THE COUNT IS FOUR, NOT FIVE, AND `s_head.bias` IS NOT ONE OF THEM.
   V-29 names five: `m_head.weight`, `m_head.bias`, `theta_head.weight`,
   `theta_head.bias`, `s_head.bias`. Measured per-parameter at this shape,
   `s_head.bias` reads `8.673617e-19` -- NOT exactly zero, and therefore not
   what the `== 0.0` guard catches. It is worse than dead at the corner: it is
   dead EVERYWHERE. `key_bias` adds `s_j` to the logit row and a softmax is
   invariant under a constant added across `j`, so the head's BIAS -- constant
   across `j` by construction -- is an exact GAUGE. Its analytic gradient is
   zero at every parameter value, and `8.673617e-19` is the float64 residue of
   a cancellation that is exact in the reals. A `> 0` guard would pass it for
   the wrong reason, which is V-29's own shape one level down, so it is bound
   here by INVARIANCE of the forward instead -- moving the bias a whole unit
   moves the read-out by `2.775558e-17`, below the same `DELTA_P` floor
   `p_spread` must clear from above -- and declared non-trainable.

3. LoRA INIT IS REFUTED BY MEASUREMENT, NOT DECLINED BY OPINION. These heads
   are `nn.Linear(d_model, 1)` -- a single factor, not a product -- so there is
   no pair of factors to split. Forced anyway (head output identically zero,
   which is what one-zero-factor buys), `m = clamp(0, 0, 1) = 0.0` lands on the
   clamp's OTHER endpoint, `log m = -inf` makes the key bias `+inf`, and the
   loss is `nan` with `m_head` still exactly dead.

RED FIRST. Written against a `trainable_heads` that did not exist, it failed
verbatim at commit `03adf7f` on `WIN-16QAL06O9GB`, python 3.11.9,
torch 2.14.0+cpu, exit 1, under
`python -m pytest tests/arm_phase/test_the_shipped_init_has_a_live_gradient_everywhere.py -q`:

    E  AttributeError: 'ArmPhase' object has no attribute 'trainable_heads'

and the census test that names the corner failed with the finding itself:

    E  AssertionError: ArmPhase.identity_heads() declares 14 trainable
    E  tensors and 4 of them receive an EXACTLY ZERO gradient: m_head.bias,
    E  m_head.weight, theta_head.bias, theta_head.weight
    E  assert not ['m_head.bias', 'm_head.weight', 'theta_head.bias',
    E  'theta_head.weight']

L-NULL. VARIES: the initialiser named by `_census`'s argument, and nothing
else. PINNED: the module shape (`S=8, d_model=16, hidden=8`), the construction
seed, the corpus draw (`make_equilibrium_batch(4, 8, 4, t_star=2,
d_model=16, seed=12345)`), the loss, and the dtype (float64, the module's own).
"""
from __future__ import annotations

import pytest
import torch

import ceq.arm_phase as arm
import ceq.arm_smprime as sib
from scale.negation_scope import make_equilibrium_batch

SEED, N, S, D, D_MODEL, H, T_STAR = 0, 4, 8, 4, 16, 8, 2

#: The four that are dead AT THE CORNER, named so a regression is a named diff
#: and not a count. `s_head.bias` is deliberately absent -- see the header, and
#: `test_the_key_bias_head_carries_an_exact_softmax_gauge` below.
DEAD_AT_THE_CORNER = ["m_head.bias", "m_head.weight",
                      "theta_head.bias", "theta_head.weight"]

#: The float64 floor below which a displacement of a softmax row is arithmetic
#: and not a fact. STATED BEFORE THE MEASUREMENT. A row of `S = 8` unmasked
#: terms carries at most `S * eps = 8 * 2.220446e-16 ~ 1.8e-15` of accumulated
#: rounding, and the worst cancellation this module's own gauge probe produced
#: -- shifting `s_head.bias` to `1e6` and reading the forward, which is exactly
#: invariant in the reals -- was `1.387e-12`. `delta_p = 1e-12` sits at that
#: worst observed cancellation and about three decades above the unit-scale
#: floor, so `p_spread > delta_p` cannot be manufactured by rounding: it is a
#: real displacement of the attention law. Below it, the new init would be a
#: no-op and any gradient it appeared to wake would be noise -- which is the
#: one failure mode a repair of V-29 has to rule out.
DELTA_P = 1e-12

#: And the paired UPPER bound: the corner is a correctness point and the
#: training door is meant to start NEAR it. `arm_smprime.trainable_heads`
#: publishes `5e-2` for the same offset; the same ceiling is asserted here so
#: the two arms' doors are comparable rather than each calibrated to itself.
CORNER_TOL = 5e-2


def _batch():
    """BED-M at `t* = 2`, the corpus `scripts/v15_r1.py` trains this arm on.

    Not `torch.randn` and not `torch.ones`: a ones-tensor makes every position
    identical, which collapses the very `j`-dependence the key-bias head lives
    on, and a guard that cannot distinguish a dead head from a degenerate input
    is not a guard.
    """
    x, y, _, _ = make_equilibrium_batch(N, S, D, t_star=T_STAR,
                                        d_model=D_MODEL, seed=12345)
    return x.to(torch.float64), y.to(torch.float64)


def _arm(init: str | None = None, **kw):
    torch.manual_seed(SEED)
    a = arm.ArmPhase(S, d_model=D_MODEL, hidden=H).to(torch.float64)
    if init is not None:
        getattr(a, init)(**kw)
    return a


def _census(init: str) -> dict:
    """`{name: |grad|max}` over every parameter the module DECLARES trainable,
    after ONE backward pass at `init` on the corpus batch.

    `requires_grad` is the filter, so a parameter the repair declares a
    non-trainable gauge leaves the census rather than sitting in it at zero.
    """
    a = _arm(init)
    x, y = _batch()
    ((a(x) - y) ** 2).mean().backward()
    return {n: (0.0 if p.grad is None else float(p.grad.abs().max()))
            for n, p in a.named_parameters() if p.requires_grad}


def _assert_all_live(init: str) -> dict:
    """THE GUARD ITSELF. Not `isfinite` -- `> 0`."""
    c = _census(init)
    dead = sorted(n for n, gmax in c.items() if gmax == 0.0)
    assert not dead, (
        "ArmPhase.%s() declares %d trainable tensors and %d of them receive "
        "an EXACTLY ZERO gradient: %s" % (init, len(c), len(dead), ", ".join(dead)))
    return c


def test_every_declared_trainable_parameter_has_a_nonzero_gradient(capsys):
    """GREEN on the shipped training init, on a REAL batch.

    Varies: nothing. Pins: everything `_census` pins.
    """
    c = _assert_all_live("trainable_heads")
    with capsys.disabled():
        print("\n  ArmPhase.trainable_heads(off=%g) on BED-M t*=%d"
              % (arm.ArmPhase.HEAD_INIT_OFF, T_STAR))
        for n in sorted(c):
            print("    %-18s |grad|max %.6e" % (n, c[n]))
    #: 14 tensors exist; 13 are DECLARED trainable, and the fourteenth is the
    #: gauge below. The count is asserted so a silently added head cannot slip
    #: past the guard by never being censused.
    assert len(c) == 13, sorted(c)
    assert len(list(_arm("trainable_heads").parameters())) == 14


def test_the_same_guard_still_fails_on_the_corner_it_was_written_against(capsys):
    """RED preserved as a finding, and the count corrected while preserving it.

    `identity_heads()` is a CORRECTNESS point -- `tests/arm_phase/test_arm_phase.py`
    `test_bind4_holds_for_the_drop_in_module_and_not_only_the_function` asserts
    `torch.equal(magnitude(u), ones)`, `torch.equal(th, zeros)` and
    `torch.equal(s, zeros)` on exactly what it returns, and
    `scripts/v16_device_probe.py` reads the same exact corner -- so it must KEEP
    sitting on both critical points and the guard must keep rejecting it.

    Varies: the initialiser. Pins: the guard, the shape, the seed, the batch.
    """
    with pytest.raises(AssertionError) as e:
        _assert_all_live("identity_heads")
    with capsys.disabled():
        print("\n  the guard against ArmPhase.identity_heads():\n    %s"
              % str(e.value).replace("\n", " "))
    for name in DEAD_AT_THE_CORNER:
        assert name in str(e.value), name
    assert "4 of them" in str(e.value), str(e.value)
    assert "s_head.bias" not in str(e.value), str(e.value)


def test_the_key_bias_head_carries_an_exact_softmax_gauge(capsys):
    """V-29's fifth name, re-derived and re-classified.

    `s_head.bias` is not dead at the corner; it is dead at EVERY point, because
    `key_bias(m, s) = s - cumsum(log m)` enters the logit row and a softmax is
    invariant under a constant added across `j`. Measured two ways, because a
    `> 0` guard reads its rounding residue as life:

      (a) INVARIANCE: moving the bias moves the forward by less than `DELTA_P`,
          the SAME floor `p_spread` has to clear from above. Not bitwise -- the
          cancellation is exact in the reals and rounded in float64, and at a
          unit shift it leaves one ulp of the read-out. Claiming bitwise here
          would be claiming more than the arithmetic gives;
      (b) CENSUS: its `|grad|max` is at the float64 floor both at the corner
          and at an interior point where every other head is live.

    Varies: `s_head.bias`, and nothing else. Pins: the shape, seed, batch.
    """
    x, _ = _batch()
    base = _arm("trainable_heads")(x).detach().clone()
    shifted = []
    for val in (0.0, 1.0, -1.0):
        b = _arm("trainable_heads")
        with torch.no_grad():
            b.s_head.bias.fill_(val)
        o = b(x).detach()
        d = float((o - base).abs().max())
        shifted.append((val, d))
        assert d < DELTA_P, (val, d)
    #: and at the corner it is float64 dust, not a gradient
    a = _arm("identity_heads")
    x, y = _batch()
    ((a(x) - y) ** 2).mean().backward()
    g = float(a.s_head.bias.grad.abs().max())
    #: AND AWAY FROM THE CORNER, which is what separates a gauge from a
    #: critical point. `m` strictly inside the clamp and `theta` off zero, so
    #: every OTHER head is live at 1e-2 scale: the gauge is still at the floor.
    b = _arm()
    gen = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for h in (b.m_head, b.theta_head, b.s_head):
            h.weight.copy_(torch.randn(h.weight.shape, generator=gen,
                                       dtype=torch.float64) * 0.02)
        b.m_head.bias.fill_(0.5)
        b.theta_head.bias.fill_(0.3)
        b.s_head.bias.fill_(0.0)
    ((b(x) - y) ** 2).mean().backward()
    interior = {n: float(p.grad.abs().max()) for n, p in b.named_parameters()}
    with capsys.disabled():
        print("\n  s_head.bias shift -> max|d out| %s  (floor %g)" % (shifted, DELTA_P))
        print("  s_head.bias |grad|max at identity_heads: %.6e  (< %g)" % (g, DELTA_P))
        print("  at an INTERIOR point (m=0.5, theta=0.3): s_head.bias %.6e, "
              "m_head.bias %.6e, theta_head.bias %.6e"
              % (interior["s_head.bias"], interior["m_head.bias"],
                 interior["theta_head.bias"]))
    assert 0.0 <= g < DELTA_P, g
    assert interior["s_head.bias"] < DELTA_P, interior["s_head.bias"]
    assert interior["m_head.bias"] > 1e-3, interior["m_head.bias"]
    assert interior["theta_head.bias"] > 1e-5, interior["theta_head.bias"]
    #: DECLARED, not hidden: the training door takes it out of the trainable set
    #: rather than leaving it in it at zero. V-29's own rule, verbatim.
    assert _arm("trainable_heads").s_head.bias.requires_grad is False
    assert _arm("identity_heads").s_head.bias.requires_grad is True


def test_the_new_init_moves_the_attention_law_off_the_corner(capsys):
    """`p_spread > DELTA_P`, and `<= CORNER_TOL`.

    `p_spread` is `max |P_new - P_identity|` over the real causal-softmax
    probability matrix on the corpus batch, the two runs differing ONLY in the
    initialiser. A repair that woke gradients without moving `P` would be
    waking them on a no-op.

    Varies: the initialiser. Pins: the shape, the seed, the batch, the weights.
    """
    x, _ = _batch()

    def p_of(init):
        a = _arm(init)
        with torch.no_grad():
            u, th, s = a.heads(x)
            return a._operator(a.wq(x), a.wk(x), u, th, s).real

    spread = float((p_of("trainable_heads") - p_of("identity_heads")).abs().max())
    with capsys.disabled():
        print("\n  p_spread %.6e   delta_p %g   corner_tol %g" % (spread, DELTA_P, CORNER_TOL))
    assert spread > DELTA_P, spread
    assert spread <= CORNER_TOL, spread


def test_lora_init_does_not_apply_and_would_make_it_worse(capsys):
    """The owner's proposed repair, measured rather than argued.

    LoRA init is one random factor and one zero factor, which is a statement
    about a PRODUCT parametrisation. `m_head`/`theta_head`/`s_head` are each one
    `nn.Linear(d_model, 1)` -- `W x + b`, a single factor -- so there is no pair
    to split. Its only realisable content here is the head OUTPUT being
    identically zero, and at that point `m = clamp(0, 0, 1) = 0.0` is the
    clamp's LOWER endpoint: `clamp'` is zero there too, `log m = -inf` puts
    `+inf` in the key bias, and the loss is `nan`. `scripts/v15_r1.py`
    `identity_point` already names `m = 0` the drop-in trap for this arm.

    Varies: the head init. Pins: the shape, the seed, the batch.
    """
    a = _arm()
    with torch.no_grad():
        for h in (a.m_head, a.theta_head, a.s_head):
            h.weight.zero_()
            h.bias.zero_()
        m = arm.magnitude(a.heads(_batch()[0])[0])
    x, y = _batch()
    loss = ((a(x) - y) ** 2).mean()
    loss.backward()
    c = {n: float(p.grad.abs().max()) for n, p in a.named_parameters()}
    dead = sorted(n for n, g in c.items() if g == 0.0)
    with capsys.disabled():
        print("\n  LoRA-zero: m absmax %.1f, loss %s, still dead: %s"
              % (float(m.abs().max()), float(loss), dead))
    assert float(m.abs().max()) == 0.0
    assert not torch.isfinite(loss), float(loss)
    assert "m_head.weight" in dead and "m_head.bias" in dead, dead
    #: and the clamp's backward, at both endpoints and the interior, so the
    #: mechanism is read off torch rather than off this docstring.
    t = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64, requires_grad=True)
    torch.clamp(t, 0.0, 1.0).sum().backward()
    assert t.grad.tolist() == [0.0, 0.0, 1.0], t.grad.tolist()


def test_the_offset_is_this_modules_own_constant_and_equals_the_siblings():
    """One number per module, and the two are bound to each other HERE.

    `ceq/arm_smprime.py` imports `ceq/arm_phase.py` (`band_draw`,
    `chain_label`), so `arm_phase` cannot import `GATE_INIT_OFF` back without a
    cycle. The value is therefore restated as `HEAD_INIT_OFF` and the two are
    asserted equal in a test, which can import both -- a drift between the arms'
    doors shows up as a red line rather than as nothing.
    """
    assert arm.ArmPhase.HEAD_INIT_OFF == sib.GATE_INIT_OFF == 1e-3
    a = _arm("trainable_heads")
    assert float(a.m_head.bias) == 1.0 - arm.ArmPhase.HEAD_INIT_OFF
    assert float(a.theta_head.bias) == arm.ArmPhase.HEAD_INIT_OFF
    assert float(a.s_head.bias) == 0.0
    assert float(a.m_head.weight.abs().max()) == 0.0
    assert float(a.theta_head.weight.abs().max()) == 0.0
    assert float(a.s_head.weight.abs().max()) == 0.0


def test_the_woken_gradient_survives_a_hundred_adam_steps(capsys):
    """CONFIRMED AT SCALE RATHER THAN AT A POINT, which is what V-29 needed on
    the sibling and did not have until the repair.

    A door that wakes every gradient at step 0 and lets the clamp re-absorb the
    magnitude at step 1 is not a repair -- `m` climbing back to `1.0` puts it on
    the same endpoint. Measured: the clamped fraction of `u` does rise to `0.969`
    by step 5 and then falls to `0.188` by step 100 as the head learns to spread,
    and NO declared-trainable tensor is dead at any checkpoint. The optimiser is
    `scripts/v15_r1.py`'s own construction, `Adam(model.parameters())`, which
    also proves the frozen gauge does not break it.

    Varies: the number of Adam steps. Pins: the shape, the seed, the batch, the
    optimiser, the lr.
    """
    x, y = _batch()
    rows = []
    for steps in (0, 5, 100):
        a = _arm("trainable_heads")
        opt = torch.optim.Adam(a.parameters(), lr=1e-3)
        for _ in range(steps):
            opt.zero_grad()
            ((a(x) - y) ** 2).mean().backward()
            opt.step()
        opt.zero_grad()
        loss = ((a(x) - y) ** 2).mean()
        loss.backward()
        c = {n: float(p.grad.abs().max())
             for n, p in a.named_parameters() if p.requires_grad}
        with torch.no_grad():
            u = a.heads(x)[0]
            clamped = float(((u >= 1.0) | (u <= 0.0)).double().mean())
        rows.append((steps, float(loss), clamped,
                     sorted(n for n, g in c.items() if g == 0.0)))
        assert float(a.s_head.bias) == 0.0, float(a.s_head.bias)
    with capsys.disabled():
        print("\n  steps / loss / frac u on the clamp / dead")
        for s, l, cl, dead in rows:
            print("    %3d  %.6f  %.3f  %s" % (s, l, cl, dead or "none"))
    for s, l, cl, dead in rows:
        assert not dead, (s, dead)
    #: the loss falls -- the arm is training, not merely differentiable
    assert rows[-1][1] < rows[0][1] / 1.5, [r[1] for r in rows]
