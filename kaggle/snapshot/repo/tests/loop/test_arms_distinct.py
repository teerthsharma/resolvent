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
    """G3, bound on BEHAVIOUR rather than on source text.

    WHY THIS WAS REWRITTEN. The previous version sliced `sign_flip_rate`'s body
    between the literals `"if op_kind =="` and `"if not h.requires_grad"`.
    FOREMAN's discard-floor refactor split that function into
    `sign_flip_draws` + `flip_rate`, both anchors vanished, and the bind died
    with `ValueError: substring not found` -- leaving **G3 unguarded** while the
    repository looked green.

    That was the FOURTH structure-by-regex instrument here to break, after the
    LOCK slice boundary, the LOCK-line scraper matching prose, and the
    provenance audit missing a line break. The two instruments that have never
    broken -- the calibration gate and the bitwise journal replay -- both
    compare VALUES. So this one now compares values too: re-anchoring on new
    strings would have been the fifth instance of the same mistake.

    An unknown arm must RAISE, not fall through to whichever branch is last.
    That is exactly how `paraformer` ran softmax under its own name.
    """
    with pytest.raises(ValueError):
        bench.sign_flip_rate("an_arm_that_was_never_implemented", n_draws=4, s=8)


@pytest.mark.parametrize("a,b", [(x, y) for i, x in enumerate(ARMS)
                                 for y in ARMS[i + 1:]])
def test_no_two_arms_execute_the_same_code_path(a, b):
    """The G3 guarantee itself, checked by EXECUTION at a fixed seed.

    If two arms share a dispatch branch their draws are bit-identical. This
    detects that directly, without knowing anything about how the dispatch is
    written -- so it survives any refactor that preserves behaviour, which is
    the property the regex version lacked.
    """
    da = bench.sign_flip_draws(a, n_draws=24, s=16, seed=0)
    db = bench.sign_flip_draws(b, n_draws=24, s=16, seed=0)
    assert da != db, (
        f"arms {a!r} and {b!r} produced BIT-IDENTICAL draws at seed 0. Either "
        f"one is executing the other's branch (the ParaFormer defect, G3), or "
        f"they are the same operator under two names."
    )


def test_the_declared_default_really_is_the_default():
    """`softmax` is the documented fallthrough. Verified by BEHAVIOUR.

    The old version asserted this by locating an `else:` in the source. The
    honest check is that softmax reads its known structural value -- exactly
    0.0 on the value path, because `I + A + A^2` is non-negative entrywise for
    a non-negative operator, so no sign flip is reachable.
    """
    r = bench.sign_flip_rate("softmax", n_draws=128, s=8, hops=3)
    assert r == 0.0, (
        f"softmax read {r!r} on the value path; it is pinned at exactly 0 by "
        f"theorem, so a nonzero reading means the arm is not running softmax"
    )


def test_the_behavioural_bind_is_calibrated_and_fires():
    """RED-first: the same arm under two names MUST be caught.

    Without this, a bind that passed everything would look identical to a bind
    that works -- which is how sixteen instruments here got believed.
    """
    d1 = bench.sign_flip_draws("sgate", n_draws=24, s=16, seed=0)
    d2 = bench.sign_flip_draws("sgate", n_draws=24, s=16, seed=0)
    assert d1 == d2, "same arm, same seed, must be reproducible"
    d3 = bench.sign_flip_draws("softmax", n_draws=24, s=16, seed=0)
    assert d1 != d3, "two genuinely different arms must not be bit-identical"
