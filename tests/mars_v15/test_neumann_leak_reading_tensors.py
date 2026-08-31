"""MARS standing attack #1 -- floor crossing by feature leak.

FILED AT it.0, bound by Saturn, against `CEQ_V15_CONTRACT.md` PART IV: "R1 BED-M,
t* = 2 ... PREDICTION: PL < floor_1 = 0.7071 -- the first floor crossing in the
campaign." The contract itself names the instrument: "a Neumann-term census on
the reading tensors."

THE MECHANISM, worked out from the contract's own algebra (PART I, S-M).
`(I - A)^-1 = sum_{t=0}^N A^t` is the Neumann series of the causal operator; the
t-hop term of the label is the path product `a_{s-1} ... a_{s-t} . b`
(WHERE WE ARE, "THE ONE SURVIVING ALGEBRAIC FACT"). ARM PL computes its gate as a
strictly PER-TOKEN read: `g(x_i) = log a_i`, `g = -softplus(W x_i)` -- a function
of `x_i` ALONE, order 0 in the Neumann sense (it carries no path history). The
composition -- order 1, 2, ..., t* -- is supposed to happen ONLY in the scan
(`C = scan(g)`) and the resolvent that follows.

A LEAK is a reading tensor built to be order-0 that turns out to depend on input
positions other than its own -- i.e. `d(reading_i) / d(x_j) != 0` for some `j !=
i`. If that dependence reaches as far as `t*` positions back, the "reading" step
has already done part of the composition FOR the model, and R1 crossing
`floor_1` would be evidence of a leaky corpus/reading-tensor pipeline, not of the
scan's compositional power. THE NUMBER THAT WOULD BE WRONG: `PL`'s NRMSE at
t*=2, reported as evidence the architecture composes 2-hop paths, when it is
actually evidence the reading tensor was handed the 2-hop answer.

CLASS. MISTAKES.md is M (measurement failure): the measured quantity (floor
crossing) is contaminated by a pipeline defect, in the shape of M-1 ("train and
eval saw different preprocessing" -- here, the reading step sees more than its
declared receptive field) and adjacent to D-2 ("an oracle that is the arm's own
resolvent" -- here the READING supplies what the RESOLVENT was supposed to earn).
NEW MECHANISM: no existing entry names a Jacobian-detectable receptive-field leak
in a per-token reading tensor; this is that census, written as a test.

STATUS, run at time of filing. `ARM PL` does not exist in this tree --
`grep`-equivalent module search (`ceq.arm_pl`, `scale.arm_pl`, `prefix_logit`)
returns nothing, `V15_LEDGER.md` NEXT is `it.3-4` and the Lean train-gate
(`it.5`) that must go green before ARM PL trains has not been reached. The census
below therefore SKIPS against the real arm and FIRES against a planted leak, so
it is live the day ARM PL lands rather than a promise about a file that isn't
there yet.
"""
from __future__ import annotations

import importlib
import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.manual_seed(0)

S, D = 6, 8          # short sequence, small width -- a census, not a training run
T_STAR = 2            # R1's own t*


# ============================================================ THE CENSUS ITSELF
def neumann_term_census(reading_fn, x: torch.Tensor) -> dict[int, float]:
    """For every output position `i`, the max |d reading_fn(x)[:, i] / d x[:, j]|
    over `j`, bucketed by `order = |i - j|`.

    `order` is the Neumann-series order a nonzero entry at that offset would
    represent: `order = 0` is the position reading itself (legal for a per-token
    gate); `order >= 1` is the reading tensor depending on a token `order` hops
    away, i.e. a term of that order baked into a tensor the architecture says
    should carry none.

    This is a real Jacobian on real tensors -- `torch.autograd.grad`, not a check
    on a variable's name or a module's docstring.
    """
    x = x.clone().requires_grad_(True)
    r = reading_fn(x)                                  # [n, s]
    assert r.dim() == 2 and r.shape[1] == x.shape[1], (
        f"reading_fn must return one scalar per position, got shape {tuple(r.shape)}")
    s = x.shape[1]
    by_order: dict[int, float] = {}
    for i in range(s):
        (grad,) = torch.autograd.grad(r[:, i].sum(), x, retain_graph=True)
        # grad: [n, s, d] -- d r[:,i] / d x[:,j,:], summed over the batch already
        per_pos = grad.abs().amax(dim=(0, 2))           # [s] -- worst case over batch, channel
        for j in range(s):
            o = abs(i - j)
            by_order[o] = max(by_order.get(o, 0.0), float(per_pos[j]))
    return by_order


def assert_reading_tensor_clean(census: dict[int, float], *, allowed_order: int = 0,
                                 tol: float = 1e-6) -> None:
    """The assertion a real run makes: no order above `allowed_order` may be
    nonzero. Raises with the offending order and magnitude, not a bare False."""
    leaks = {o: v for o, v in census.items() if o > allowed_order and v > tol}
    assert not leaks, (
        f"Neumann-term census found order(s) {sorted(leaks)} present in a reading "
        f"tensor declared order<={allowed_order}: {leaks}. The reading step has "
        f"already composed a path of that length; a floor crossing downstream is "
        f"evidence of this leak, not of the scan.")


# ============================================================ REFERENCE READERS
def spec_reading_g(x: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    """`g(x_i) = -softplus(W x_i)`, exactly S-M's declared reading tensor.
    Strictly per-token: the matmul is over the channel dim only, so its true
    Jacobian off the diagonal (`j != i`) is exactly zero by construction. This is
    the clean-case control -- a census that fires on THIS function would be a
    census that cries wolf on a leak-free implementation."""
    torch.manual_seed(seed)
    w = torch.randn(x.shape[-1], x.shape[-1]) / x.shape[-1] ** 0.5
    return -torch.nn.functional.softplus(x @ w).mean(dim=-1)


def leaky_reading_g(x: torch.Tensor, *, t_leak: int, seed: int = 0,
                     leak_weight: float = 0.7) -> torch.Tensor:
    """The planted positive: the same per-token read, PLUS a term that mixes in
    the token `t_leak` positions back -- an order-`t_leak` term sitting inside
    what is declared an order-0 tensor. Modelled on a lookahead/shortcut bug
    (a conv tap, a cached neighbour feature) rather than named as one, so the
    census has to find it from the tensor, not from a comment.
    """
    base = spec_reading_g(x, seed=seed)
    shifted = torch.roll(x, shifts=t_leak, dims=1)
    shifted[:, :t_leak, :] = 0.0        # no wraparound artefact at the boundary
    leak = leak_weight * shifted.mean(dim=-1)
    return base + leak


# =================================================================== THE TESTS
def _sample_x(n: int = 4) -> torch.Tensor:
    return torch.randn(n, S, D)


def test_census_is_silent_on_the_contract_specified_reading_function():
    """Sanity: the instrument does not fire on a leak-free implementation.
    Without this, a PASS on ARM PL later would be uninterpretable -- a census
    that always fires (or always clears) is a V-10 (threshold satisfied by
    construction) wearing a Neumann label."""
    census = neumann_term_census(lambda x: spec_reading_g(x), _sample_x())
    assert_reading_tensor_clean(census, allowed_order=0)
    #: and the order-0 entry itself must be genuinely nonzero, i.e. the census
    #: can see SOMETHING -- a census blind at order 0 could also be blind at
    #: order t and this asserts it is not the reason nothing else fired.
    assert census[0] > 1e-6, census


def test_census_MUST_FIRE_on_a_planted_order_2_leak():
    """FIRES. `leaky_reading_g` with `t_leak=2` mixes R1's own t* directly into
    the reading tensor. The census must find a nonzero order-2 entry and the
    real assertion must raise on it -- demonstrated, not claimed."""
    census = neumann_term_census(lambda x: leaky_reading_g(x, t_leak=T_STAR), _sample_x())
    assert census[T_STAR] > 1e-3, (
        f"planted leak not detected: order-{T_STAR} entry is {census.get(T_STAR)}; "
        f"the fixture failed to plant what this test claims it plants")
    with pytest.raises(AssertionError, match=r"Neumann-term census found order"):
        assert_reading_tensor_clean(census, allowed_order=0)


def test_census_MUST_FIRE_at_every_leak_distance_it_is_asked_to_find():
    """A second must-fire, at a distance other than t*=2 (R2's t*=8 territory),
    so the census is shown general to the offset rather than tuned to one
    number."""
    for t_leak in (1, 3, 4):
        census = neumann_term_census(
            lambda x, tl=t_leak: leaky_reading_g(x, t_leak=tl), _sample_x())
        assert census[t_leak] > 1e-3, (t_leak, census)
        with pytest.raises(AssertionError):
            assert_reading_tensor_clean(census, allowed_order=0)


def _find_arm_pl():
    """Best-effort locate ARM PL's reading-tensor constructor. Tries the two
    module names the contract's own naming (`ARM PL`, S-M) would produce.
    Returns the callable or None."""
    for mod_name, attr in (
        ("ceq.arm_pl", "reading_gate"),
        ("scale.arm_pl", "reading_gate"),
        ("ceq.arm_pl", "log_a"),
        ("scale.arm_pl", "log_a"),
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    return None


def test_neumann_census_on_arm_pl():
    """The real attack. SKIPS -- ARM PL has not been built (`V15_LEDGER.md`
    NEXT is `it.3-4`; the Lean train-gate at `it.5` must go green before ARM PL
    trains, per L-LEAN, and R1 is scheduled `it.9-10`). This is not a promise of
    a future PASS: the two tests above show the same assertion function catches
    a real leak, so this test fires the day `ceq.arm_pl.reading_gate` (or
    `scale.arm_pl.reading_gate`) exists, without being rewritten.

    Per the coordinator's own instruction (never let an absent subject read as
    a pass, MISTAKES.md V-6/V-7): this SKIPS rather than passing vacuously.
    """
    fn = _find_arm_pl()
    if fn is None:
        pytest.skip(
            "ARM PL not found (tried ceq.arm_pl / scale.arm_pl reading_gate / "
            "log_a). R1 has not run: V15_LEDGER.md NEXT=it.3-4, Lean train-gate "
            "verdict is it.5, ARM PL build is it.6-7, R1 is it.9-10. This attack "
            "is live once that module exists; see test_census_MUST_FIRE_* above "
            "for proof the assertion catches a real leak on a stand-in function.")
    census = neumann_term_census(fn, _sample_x())
    assert_reading_tensor_clean(census, allowed_order=0)


def demo() -> None:
    """Self-check, runnable standalone (`python test_neumann_leak_reading_tensors.py`)."""
    census = neumann_term_census(lambda x: spec_reading_g(x), _sample_x())
    assert_reading_tensor_clean(census, allowed_order=0)
    leaked = neumann_term_census(lambda x: leaky_reading_g(x, t_leak=2), _sample_x())
    assert leaked[2] > 1e-3
    try:
        assert_reading_tensor_clean(leaked, allowed_order=0)
    except AssertionError:
        pass
    else:
        raise AssertionError("must-fire fixture did not fire")
    print("demo OK: clean reading tensor cleared the census; planted order-2 "
          f"leak found at magnitude {leaked[2]:.6f} and the assertion raised on it")


if __name__ == "__main__":
    demo()
