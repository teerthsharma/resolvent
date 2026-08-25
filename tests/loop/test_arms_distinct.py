"""ARMS-DISTINCT: no arm may report another arm's number under its own name.

WHY THIS EXISTS. `ceq/bench.py::sign_flip_rate` built the `gam` vector for its
`paraformer` arm and never referenced it -- `op_kind == "paraformer"` was absent
from the path-sum branch, so that arm executed plain single-hop softmax. The
published claim "0 flips in 2048 draws, Clopper-Pearson ceiling 0.00146, >= 84x
better than ParaFormer" was a SOFTMAX number wearing ParaFormer's name, and it
stood as this project's last surviving novelty claim until it was caught.

That is stopping condition G3. This file is the standing bind against it.

TWO INDEPENDENT CHECKS, because one is not enough:

  1. BEHAVIOURAL. Each arm is fingerprinted by its sign-flip rate across several
     seeds. Two arms that are genuinely different operators disagree on at least
     one seed. Two arms that are secretly the same code path agree on ALL of
     them -- which is exactly how the bug presented (both read 0.02734375,
     bitwise, at hops=0).

     Fingerprinting on a VECTOR of seeds rather than one number is deliberate.
     `sign_flip_rate` returns k/n, so single-seed collisions between genuinely
     distinct arms are common at these draw counts; agreement across six
     independent seeds is not.

  2. STRUCTURAL. The `op_kind` literal for each arm must appear inside
     `sign_flip_rate`'s dispatch chain. The bug was invisible behaviourally
     until someone thought to compare arms, but it was always visible
     structurally: the string "paraformer" simply did not occur in the branch
     that computed its number.

CALIBRATED BEFORE USE. `test_the_bind_fires_on_a_known_duplicate` feeds the
check two arms that ARE the same operator and requires it to RAISE. An
instrument that has never been observed failing is not an instrument -- eleven
in this project were internally consistent and externally wrong, including the
calibration gate itself, which until 2026-08-25 printed its targets as strings
and always exited 0.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import torch

from ceq import bench

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Every arm `sign_flip_rate` accepts. Kept explicit rather than introspected:
#: an arm added to the dispatch and forgotten here is exactly the failure this
#: file guards, so the list must be maintained by hand and reviewed.
ARMS = ["signed", "softmax", "sgate", "paraformer", "signmag", "deltanet",
        "tgate", "tgatex"]

SEEDS = (0, 1, 2, 3, 4, 5)


def fingerprint(kind: str, *, hops: int = 2, s: int = 16, n_draws: int = 64,
                device=None) -> tuple[float, ...]:
    """Behavioural signature of an arm: its rate at each of several seeds."""
    return tuple(
        bench.sign_flip_rate(kind, n_draws=n_draws, s=s, hops=hops, seed=seed,
                             device=device)
        for seed in SEEDS
    )


def assert_arms_distinct(kinds, *, device=None, **kw) -> dict:
    """RAISE if any two arms produce an identical fingerprint.

    Returns the fingerprints so a caller can record them as evidence.
    """
    fps = {k: fingerprint(k, device=device, **kw) for k in kinds}
    collisions = [
        (a, b, fps[a])
        for i, a in enumerate(kinds)
        for b in kinds[i + 1:]
        if fps[a] == fps[b]
    ]
    if collisions:
        lines = [
            f"  {a!r} and {b!r} agree on ALL {len(SEEDS)} seeds: {fp}"
            for a, b, fp in collisions
        ]
        raise AssertionError(
            "ARMS-DISTINCT FAILED (G3). At least one arm is reporting another "
            "arm's number under its own name:\n" + "\n".join(lines) +
            "\nThis is the ParaFormer bug class. No per-arm number may be "
            "published until the dispatch is fixed and this bind is GREEN."
        )
    return fps


# ==========================================================================
# CALIBRATION -- the bind must be seen to FAIL before any green is trusted
# ==========================================================================

def test_the_bind_fires_on_a_known_duplicate():
    """RED-first. Two names, one operator: the check MUST raise.

    This reconstructs the exact shape of the ParaFormer bug -- two arm labels
    resolving to the same code path -- and requires detection. If this test
    passes without raising, every green below is worthless.
    """
    with pytest.raises(AssertionError, match="ARMS-DISTINCT FAILED"):
        # `sgate` against itself: same dispatch branch, same random stream,
        # therefore identical on every seed, therefore must be caught.
        assert_arms_distinct(["sgate", "sgate"])


def test_the_bind_does_not_fire_on_two_genuinely_different_arms():
    """The other half of calibration: it must not cry wolf.

    A check that raises on everything would also 'catch' the bug and would be
    equally useless.
    """
    assert_arms_distinct(["softmax", "sgate"])


# ==========================================================================
# THE BIND ITSELF
# ==========================================================================

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_no_arm_reports_another_arms_number(device):
    """Every arm in `ARMS` must be behaviourally distinct from every other."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    fps = assert_arms_distinct(ARMS, device=torch.device(device))
    print(f"\n  ARMS-DISTINCT on {device}: {len(ARMS)} arms, "
          f"{len(ARMS)*(len(ARMS)-1)//2} pairs, 0 collisions")
    for k, fp in fps.items():
        print(f"    {k:>10} {fp}")


#: The ONE arm allowed to be the `else` fallthrough. Declared here, in the test,
#: rather than inferred -- an inferred default would silently absorb whichever
#: arm happened to be last, which is the bug this file exists to catch.
DEFAULT_ARM = "softmax"


def missing_from_dispatch(dispatch: str, arms, default: str) -> list[str]:
    """Arms with no literal in `dispatch`, excusing exactly one declared default.

    Pure function of its text so it can be exercised against a KNOWN-BAD
    dispatch, which is the only way to show the check can fail.
    """
    return [a for a in arms if a != default and f'"{a}"' not in dispatch]


def _real_dispatch() -> str:
    src = (ROOT / "ceq" / "bench.py").read_text(encoding="utf-8")
    body = src[src.index("def sign_flip_rate"):]
    return body[body.index("if op_kind =="):body.index("if not h.requires_grad")]


# --- RED-first: the sharpened check must still catch the real bug -----------

def test_excusing_the_default_does_not_excuse_a_missing_arm():
    """The repair must not blunt the instrument.

    Iteration 1's version flagged `softmax`, which is the intentional `else`
    fallthrough -- a false positive. The fix excuses ONE declared default. This
    test proves the excusal is narrow: on a verbatim reconstruction of the
    pre-fix dispatch, with `softmax` now excused, `paraformer` is STILL flagged.

    If this ever passes vacuously the repair has disabled detection of the
    exact bug the file was written for.
    """
    buggy = '''
                if op_kind == "deltanet":
                    a = _causal_deltanet_operator(kk, bet, window=window)
                elif op_kind == "signed":
                    a = _causal_signed_operator(qq, kk, window=window)
                else:
                    a = _softmax_operator(qq, kk, window=window)
    '''
    got = missing_from_dispatch(buggy, ["paraformer", "deltanet", "signed",
                                        "softmax"], DEFAULT_ARM)
    assert got == ["paraformer"], (
        f"the sharpened check no longer detects the historical bug shape; "
        f"it reported {got!r}, expected ['paraformer']"
    )


def test_a_newly_added_arm_cannot_inherit_the_default_silently():
    """The forward-looking half of the same guarantee.

    An arm added to the accept-list and forgotten in the dispatch executes the
    `else` branch and publishes softmax's number under its own name. Excusing
    `softmax` must never extend to it.
    """
    got = missing_from_dispatch(_real_dispatch(),
                                ARMS + ["a_new_arm_nobody_wired_up"],
                                DEFAULT_ARM)
    assert got == ["a_new_arm_nobody_wired_up"], (
        f"a hypothetical unwired arm was not flagged; got {got!r}"
    )


# --- the bind itself --------------------------------------------------------

def default_is_the_else_branch(dispatch: str, default_call: str) -> bool:
    """True if the FIRST statement of some `else:` block calls `default_call`.

    Reads the first non-blank, non-comment line of each `else:` block rather
    than a fixed line offset. The offset version discriminated correctly on
    today's formatting and would have read the wrong line after any reflow --
    a check that depends on whitespace is a check that fails silently later.
    """
    for seg in dispatch.split("else:")[1:]:
        for line in seg.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            if default_call in line:
                return True
            break                      # only the block's first statement counts
    return False


def test_the_default_is_else_check_can_fail():
    """RED-first for the excusal guard, kept in the file, not in a shell.

    BAD case: `_softmax_operator` appears in the dispatch but NOT as the `else`
    fallthrough -- precisely the hole that excusing `softmax` would open.
    """
    good = ('\n                if op_kind == "deltanet":\n'
            '                    a = _causal_deltanet_operator(kk, bet)\n'
            '                else:\n'
            '                    a = _softmax_operator(qq, kk)\n')
    bad = ('\n                if op_kind == "deltanet":\n'
           '                    a = _softmax_operator(qq, kk)\n'
           '                else:\n'
           '                    a = _causal_signed_operator(qq, kk)\n')
    assert default_is_the_else_branch(good, "_softmax_operator") is True
    assert default_is_the_else_branch(bad, "_softmax_operator") is False, (
        "the guard passes a dispatch whose `else` does not default to softmax; "
        "excusing the default is then unsafe"
    )


def test_the_declared_default_really_is_the_else_branch():
    """`softmax` is excused only because it IS the documented fallthrough.

    Verified, not assumed: if the `else` ever stops calling `_softmax_operator`,
    the excusal in `missing_from_dispatch` becomes a hole, and this closes it.
    """
    assert default_is_the_else_branch(_real_dispatch(), "_softmax_operator"), (
        f"the declared default {DEFAULT_ARM!r} is excused from the literal "
        f"check because it is supposed to be the `else` fallthrough, but no "
        f"`else` block calls `_softmax_operator`. The excusal is now a hole."
    )


def test_every_arm_literal_occurs_in_the_dispatch_that_computes_it():
    """STRUCTURAL half. The bug was always visible here.

    `paraformer` was accepted by the guard clause -- so it did not raise
    ValueError -- while never appearing in the path-sum dispatch, so it silently
    fell through to the softmax branch. Accepting a name is not implementing it.
    """
    missing = missing_from_dispatch(_real_dispatch(), ARMS, DEFAULT_ARM)
    assert not missing, (
        f"arms accepted by sign_flip_rate but absent from the branch that "
        f"computes their number: {missing}. This is the ParaFormer bug class "
        f"(G3) -- the arm will silently execute whichever branch it falls "
        f"through to, and report that operator's number under its own name."
    )
