"""MARS / MORIARTY, R9 iteration 3 -- attacks filed on the standing GREENs.

Each test below is a VALUE test with the number that would prove it. Two attacks
are filed; one fires and one is refuted by its own planted null, and both are
kept, because an adversary who only reports the hits is running an uncontrolled
search.

NO WALL CLOCK IS TAKEN ANYWHERE IN THIS FILE. Every number is either exact
combinatorics or a re-run of a pure function on committed per-seed readings.
"""
from __future__ import annotations

import itertools
from collections import Counter

import pytest

import scale.arm_s as A
from scale.m3_synthetic_settled import contrast

#: Per-seed eval NRMSE, read out of `results/m3_quintuple_v2.jsonl` at
#: `key = <cell>_k{8,0}_s64_d24_st150_ntr8192_nev512_b21_sd{0..4}`. Inlined so
#: the attack does not depend on a journal Mercury is writing this round.
PER_SEED = {
    "softmax": [0.8771677350487059, 0.8895233704358247, 0.9191477552755476,
                0.8901746162956505, 0.8856028755736706],
    "twin":    [0.7674026300641029, 0.7843970173263923, 0.7945053565599524,
                0.7980011632282340, 0.7603277678907548],
    "settled": [0.7535814923070905, 0.7688018924834016, 0.8746579711474933,
                0.8160713667359690, 0.7063165309644458],
    "argmax":  [1.0229100603025747, 0.9948669621484079, 1.0117045251109624,
                1.0108039523469563, 1.0136094994756117],
}


def _exact_lattice(a, b):
    """Every achievable bootstrap mean of the paired delta, with multiplicity.

    `contrast(a, b)` reports `mean(a) - mean(b)` and resamples the FIVE paired
    per-seed deltas, so the resample space is 5**5 = 3125 ordered tuples and the
    statistic takes at most C(9, 4) = 126 distinct values.
    """
    d = [x - y for x, y in zip(a, b)]
    c = Counter(sum(t) / len(d) for t in itertools.product(d, repeat=len(d)))
    return d, sorted(c), c


# ------------------------------------------------------------------ ATTACK G1 --
def test_G1_the_shipped_interval_endpoints_are_a_function_of_the_bootstrap_seed():
    """FIRES. The published intervals are not determined by the data.

    `scale/m3_synthetic_settled.py:193-200` draws 10,000 resamples of five
    values and reads `reps[250]` and `reps[9750]`. The bootstrap statistic has
    exactly 126 atoms whose masses are computable in closed form, and for
    `settled - softmax` the 2.5% target falls in the gap between cumulative mass
    0.024320 (at 0.066232) and 0.025920 (at 0.068181) -- a window of width
    0.0016. Which atom `reps[250]` lands on is then a Binomial(10000, 0.024320)
    coin flip.

    Measured over bootstrap seeds 0..99 at the shipped `n_boot=10000`:

        settled - softmax  ci_lo in {0.066232: 32, 0.068181: 40,
                                     0.071449: 23, 0.071582: 5}
        argmax  - softmax  ci_hi in {-0.102204: 36, -0.102786: 36,
                                     -0.103194: 19, -0.103286: 9}

    The `argmax - softmax` row is the sharp one: `-0.102204` is what
    `ceq/hf_artifact/README.md` publishes and `-0.102786` is what `README.md`
    and `CHECKLIST.md` publish, and the SHIPPED Monte Carlo alone produces both,
    36 seeds each. For that contrast the two "interval families" are not
    separated by the number.

    NO VERDICT MOVES -- every endpoint stays on the same side of zero at every
    seed tested, so this is an attack on the reproducibility of published digits
    and not on any conclusion.
    """
    spread = {}
    for a, b in (("twin", "softmax"), ("settled", "softmax"),
                 ("settled", "twin"), ("argmax", "softmax")):
        lo = Counter()
        hi = Counter()
        for s in range(40):
            r = contrast(PER_SEED[b], PER_SEED[a], seed=s)
            lo[round(r["ci_lo"], 6)] += 1
            hi[round(r["ci_hi"], 6)] += 1
        spread[f"{a} - {b}"] = (dict(lo), dict(hi))

    #: The headline F-green is the STABLE one and is not struck by this.
    lo_f, hi_f = spread["twin - softmax"]
    assert max(lo_f.values()) >= 30 and 0.100873 in lo_f, lo_f

    #: The other three are not.
    lo_s, _ = spread["settled - softmax"]
    assert len(lo_s) >= 3, lo_s
    assert {0.066232, 0.068181} <= set(lo_s), lo_s

    _, hi_a = spread["argmax - softmax"]
    assert {-0.102204, -0.102786} <= set(hi_a), (
        "the shipped Monte Carlo no longer reproduces both published values")

    #: No verdict moves: every sampled endpoint keeps its sign.
    for name, (lo, hi) in spread.items():
        assert len({v > 0 for v in lo}) == 1, (name, lo)
        assert len({v > 0 for v in hi}) == 1, (name, hi)


def test_G1b_the_exact_percentile_is_cheaper_than_the_sample_it_replaces():
    """The repair, and the reason the defect is gratuitous: enumerating the
    whole resample space costs 3,125 sums against the 10,000 the shipped
    function draws, and it is deterministic.
    """
    for a, b in (("twin", "softmax"), ("settled", "softmax"),
                 ("argmax", "softmax"), ("settled", "twin")):
        d, vals, c = _exact_lattice(PER_SEED[b], PER_SEED[a])
        assert sum(c.values()) == 5 ** 5 == 3125
        assert len(vals) == 126, len(vals)
    #: `twin - softmax`: the published endpoints ARE the exact ones.
    d, vals, c = _exact_lattice(PER_SEED["softmax"], PER_SEED["twin"])
    tot = sum(c.values())
    cum = 0
    lo = None
    for v in vals:
        cum += c[v]
        if lo is None and cum / tot > 0.025:
            lo = v
            break
    assert round(lo, 6) == 0.100873, lo


# ------------------------------------------------------------------ ATTACK G2 --
@pytest.mark.parametrize("beta,fires", [(0.5, True), (1e-3, True),
                                        (1e-6, False), (1e-9, False)])
def test_G2_the_settled_birth_gate_has_a_rejection_region(beta, fires):
    """REFUTED, and kept for that reason.

    The attack was V-10: `gate1_event`'s discriminating contrast is
    `d_H(settled fixed point, ONE step)` against a threshold, and a gate whose
    quantity cannot approach the threshold from below has no rejection region.
    The planted null is `beta -> 0`, which makes the settle the identity and
    must collapse `d_onestep`.

    It collapses. Measured at s=64, d=24, k=8, seeds 0-2, threshold 1e-6:

        beta 0.5    d_onestep 7.139e+00 .. 1.539e+01   fired 3/3
        beta 1e-2   d_onestep 2.135e-03 .. 4.417e-03   fired 3/3
        beta 1e-6   d_onestep 2.113e-11 .. 4.379e-11   fired 0/3
        beta 1e-9   d_onestep 0.000e+00                fired 0/3

    `d_onestep` scales as roughly `beta**2`, so the gate does discriminate. The
    honest qualification, which the gate does not state: its rejection region
    begins near `beta = 7e-4`, so it separates "settling happens at all" from
    "settling does not happen", over four orders of magnitude of beta. It is not
    evidence that settling is MATERIAL at the shipped `beta = 0.5`.
    """
    rows, _g, (n_fired, _ci), n = A.gate1_event(
        [64], [24], [8], [0, 1, 2], beta, 21, 1e-6)
    assert (n_fired == n) is fires, (beta, n_fired, n,
                                     [r["d_onestep"] for r in rows])
