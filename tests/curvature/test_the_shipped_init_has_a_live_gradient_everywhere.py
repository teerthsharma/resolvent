"""The guard `gradient_finiteness` cannot be: NON-ZERO, not merely finite.

RED-FIRST, and the RED is `MISTAKES.md` V-29 caught by a check that can see it.
`ceq/arm_smprime.py:451 gradient_finiteness` asserts only that
`first_non_finite is None`, and **zero is finite**, so a gradient that is
identically `0.0` passes it by construction. V-29's own rule is the repair: for
every parameter a module declares trainable, one backward pass at the shipped
initialisation must give `|grad| > 0`.

Written first against `identity_heads`, which was the initialisation arms were
being trained from, it failed verbatim at commit `62cb8e0` on `WIN-16QAL06O9GB`,
python 3.11.9, torch 2.14.0+cpu, exit 1, under
`python -m pytest tests/curvature/test_the_shipped_init_has_a_live_gradient_everywhere.py -q`:

    E  AssertionError: ArmSMPrime.identity_heads() declares 15 trainable
    E  tensors and 5 of them receive an EXACTLY ZERO gradient: g,
    E  m_head.bias, m_head.weight, theta_head.bias, theta_head.weight
    E  assert not ['g', 'm_head.bias', 'm_head.weight', 'theta_head.bias',
    E  'theta_head.weight']

FIVE, NOT FOUR, AND THE FIFTH IS A SWITCH. V-29 names the two gate heads. The
`g` switch -- one of the three `CEQ_V16_CONTRACT.md` PART I names as settable --
is dead at that same point for three reasons at once, and any one of them
suffices: `dm/d(lerp)` is the clamp's zero at the endpoint, `d(lerp)/dg` is
`u - 1 = 0`, and `d(theta * g)/dg` is `theta = 0`. A guard written per-mechanism
would have found two; a guard written per-PARAMETER finds all three.

THE REPLACEMENT ROUTE, which this file now binds: `trainable_heads()`, the same
corner stepped `GATE_INIT_OFF` off each critical point. `identity_heads()` keeps
the exact corner its correctness callers need and is asserted here to STAY dead,
so the defect stays measurable and cannot be re-introduced as a training init
without this file going red.
"""
import pytest
import torch

import ceq.arm_smprime as arm

SEED, N, S, D, H = 0, 4, 8, 4, 8

#: V-29's five, named so a regression is a named diff and not a count.
DEAD_AT_THE_CORNER = ["g", "m_head.bias", "m_head.weight",
                      "theta_head.bias", "theta_head.weight"]


def _census(init: str) -> dict:
    """`{name: |grad|max}` over every parameter the module declares trainable,
    after ONE backward pass at `init` on random data.

    VARIES: the initialiser named by `init`, and nothing else. PINS: the module
    shape (`S=8, d_model=4, hidden=8`), the construction seed, the draw, the
    loss, and the dtype (float64, the module's own).
    """
    torch.manual_seed(SEED)
    a = arm.ArmSMPrime(S, d_model=D, hidden=H).to(torch.float64)
    getattr(a, init)()
    g = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(N, S, D, generator=g, dtype=torch.float64)
    y = torch.randn(N, generator=g, dtype=torch.float64)
    ((a(x) - y) ** 2).mean().backward()
    return {n: (0.0 if p.grad is None else float(p.grad.abs().max()))
            for n, p in a.named_parameters() if p.requires_grad}


def _assert_all_live(init: str) -> dict:
    """THE GUARD ITSELF, one call, reusable by any module with an initialiser.
    Not `isfinite` -- `> 0`."""
    c = _census(init)
    dead = sorted(n for n, gmax in c.items() if gmax == 0.0)
    assert not dead, (
        "ArmSMPrime.%s() declares %d trainable tensors and %d of them receive "
        "an EXACTLY ZERO gradient: %s" % (init, len(c), len(dead), ", ".join(dead)))
    return c


def test_every_declared_trainable_parameter_has_a_nonzero_gradient(capsys):
    """GREEN on the shipped training init. Varies: nothing. Pins: everything
    `_census` pins."""
    c = _assert_all_live("trainable_heads")
    with capsys.disabled():
        print("\n  ArmSMPrime.trainable_heads(off=%g)" % arm.GATE_INIT_OFF)
        for n in sorted(c):
            print("    %-18s |grad|max %.6e" % (n, c[n]))
    assert len(c) == 15, len(c)


def test_the_same_guard_still_fails_on_the_corner_it_was_written_against(capsys):
    """RED preserved as a finding. `identity_heads()` is a correctness point and
    must KEEP sitting on both critical points, so the guard must keep rejecting
    it -- a guard that went green here would mean the corner had been moved.

    Varies: the initialiser. Pins: the guard, the shape, the seed, the draw.
    """
    with pytest.raises(AssertionError) as e:
        _assert_all_live("identity_heads")
    with capsys.disabled():
        print("\n  the guard against ArmSMPrime.identity_heads():\n    %s"
              % str(e.value).replace("\n", " "))
    for name in DEAD_AT_THE_CORNER:
        assert name in str(e.value), name
    assert "5 of them" in str(e.value), str(e.value)


def test_the_finiteness_probe_passes_the_dead_gradient_it_cannot_see(capsys):
    """WHY THE SHIPPED GUARD DID NOT CATCH IT, asserted rather than argued.
    `gradient_finiteness(init="identity")` reports `first_non_finite = None` on
    the very point the guard above rejects. Zero is finite."""
    r = arm.gradient_finiteness(init="identity", steps=2, n=8, s=8, d_model=4)
    with capsys.disabled():
        print("\n  gradient_finiteness(init='identity') -> first_non_finite=%r"
              % r["first_non_finite"])
    assert r["first_non_finite"] is None
    assert sorted(n for n, g in _census("identity_heads").items() if g == 0.0) \
        == DEAD_AT_THE_CORNER


def test_the_offset_is_the_modules_own_constant_not_a_harness_local_one():
    """One number, one place. `ceqjepa/headtohead.py` builds its own gate heads
    and must read the offset from here, or the bed's init and the module's init
    can drift apart silently -- which is the shape of the defect itself."""
    assert arm.GATE_INIT_OFF == 1e-3
    a = arm.ArmSMPrime(S, d_model=D, hidden=H).to(torch.float64).trainable_heads()
    assert float(a.m_head.bias) == 1.0 - arm.GATE_INIT_OFF
    assert float(a.theta_head.bias) == arm.GATE_INIT_OFF
    assert float(a.m_head.weight.abs().max()) == 0.0
    assert float(a.theta_head.weight.abs().max()) == 0.0
