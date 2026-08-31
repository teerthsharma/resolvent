"""Both published interval families reproduce from the journal.

THE FINDING THIS CLOSES. Three headline contrasts are published in two versions
that agree on every point estimate and every upper bound but differ on lower
bounds (and on one upper). Family A ships in `ceq/hf_artifact/README.md` and
`results/capability_table_v{0,1}.md`; family B is in the root `README.md`,
`CHECKLIST.md` and `DONE.md`, and `DONE.md` carries both.

    contrast            family A                     family B
    settled - twin      [-0.048587, +0.031557]       [-0.042903, +0.031557]
    settled - softmax   [+0.066232, +0.147110]       [+0.068181, +0.147110]
    argmax  - softmax   [-0.134115, -0.102204]       [-0.134115, -0.102786]

Neither is wrong. At five seeds the paired resample space is FINITE -- `5**5 =
3125` tuples, 126 distinct means -- so the exact percentile is computable, and
it is a different estimator from a `B=10000` Monte-Carlo draw of the same
distribution. Family A is `m3_synthetic_settled.contrast(..., n_boot=10000,
seed=0)`. Family B is the exact enumeration. Both are defensible; only one was
ever labelled.

WHAT IS ALREADY BOUND, AND WHAT THIS ADDS. `tests/chase/test_capability_table.py`
already binds family A to the journal
(`test_the_interval_is_the_journal_bootstrap_and_carries_its_estimator_name`)
and already requires each row to carry `n_boot` and `boot_seed`. Family B is
bound NOWHERE -- it exists only as transcribed digits in three prose documents,
which is how it came to look like a discrepancy rather than an estimator. This
file binds that half and nothing else.

WHY IT MATTERS THAT THIS WAS ALREADY KNOWN. The generator's own limits paragraph
(e) stated the whole thing -- in `ceq/hf_artifact/README.md:85`, the file that
ships -- while the table above it printed a bare `95% CI` with no estimator and
the JSON behind it already carried `n_boot: 10000, boot_seed: 0`. The provenance
existed and was dropped at render time. That is MISTAKES.md P-8 exactly: a
caveat that is present, correctly worded, in the right place, and below the
number that gets quoted.

SINCE FIXED, and the description above is kept because it is why this file
exists. The contrast table now names its estimator in the column header and
prints the journal records the rows were computed from, so the caveat and the
number no longer live a render apart. This test binds the arithmetic regardless
of how the card chooses to render it.

NOTHING HERE IS RUN AGAINST A LIVE FILE. The per-seed values are read from the
COMMITTED blob of `results/m3_quintuple_v2.jsonl` via `git show`, never from the
working tree, because that journal is being appended to by another agent's run.
No wall clock is read and no arm is trained; both procedures are deterministic
arithmetic over ten floats.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale.m3_synthetic_settled import contrast                    # noqa: E402

JOURNAL = "results/m3_quintuple_v2.jsonl"
#: The published geometry, verbatim from the journal keys the reading used.
STEM = "_k{k}_s64_d24_st150_ntr8192_nev512_b21_sd{sd}"
CELLS = {"argmax": 8, "settled": 8, "softmax": 0, "twin": 8}
SEEDS = range(5)

#: (first arg, second arg, family A, family B). `contrast(a, b)` reports
#: `mean(a) - mean(b)`, so the pair is ordered to make the published sign.
PUBLISHED = {
    "settled - twin":    ("twin", "settled", (-0.048587, +0.031557), (-0.042903, +0.031557)),
    "settled - softmax": ("softmax", "settled", (+0.066232, +0.147110), (+0.068181, +0.147110)),
    "argmax - softmax":  ("softmax", "argmax", (-0.134115, -0.102204), (-0.134115, -0.102786)),
}


def _exact(a, b):
    """The exact percentile over ALL 5**5 paired resamples.

    Same percentile convention as `contrast`, so the two differ only in whether
    the resample space is enumerated or sampled.
    """
    d = [x - y for x, y in zip(a, b)]
    reps = sorted(sum(t) / len(d) for t in itertools.product(d, repeat=len(d)))
    n = len(reps)
    return reps[int(0.025 * n)], reps[min(n - 1, int(0.975 * n))]


@pytest.fixture(scope="module")
def per_seed():
    """Per-seed eval NRMSE from the COMMITTED journal blob."""
    raw = subprocess.run(["git", "show", f"HEAD:{JOURNAL}"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in raw.splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["key"]] = r["value"]["eval_nrmse"]
    out = {}
    for arm, k in CELLS.items():
        out[arm] = [rows[arm + STEM.format(k=k, sd=s)] for s in SEEDS]
    return out


def test_the_journal_holds_all_five_seeds_of_all_four_cells(per_seed):
    """The instrument before the reading. If a cell were missing the fixture
    would raise, but a silently short list would make every interval below a
    different statistic -- so the lengths are asserted, not assumed."""
    for arm, vals in sorted(per_seed.items()):
        print(f"\n  {arm:>8} {[round(v, 6) for v in vals]}")
        assert len(vals) == 5, (arm, len(vals))
        assert len(set(vals)) == 5, f"{arm} has duplicate seeds -- not five runs"


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_family_A_is_the_shipped_monte_carlo_bootstrap(name, per_seed):
    """Family A -- what `ceq/hf_artifact/README.md` ships."""
    x, y, fa, _fb = PUBLISHED[name]
    r = contrast(per_seed[x], per_seed[y], n_boot=10000, seed=0)
    got = (round(r["ci_lo"], 6), round(r["ci_hi"], 6))
    print(f"\n  {name}: B=10000 seed=0 -> {got}   published A {fa}")
    assert got == fa, (got, fa)


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_family_B_is_the_exact_enumeration_over_all_3125_resamples(name, per_seed):
    """Family B -- what the root `README.md` and `CHECKLIST.md` print. This is
    the half nothing else in the tree binds."""
    x, y, _fa, fb = PUBLISHED[name]
    got = tuple(round(v, 6) for v in _exact(per_seed[x], per_seed[y]))
    print(f"\n  {name}: exact 3125    -> {got}   published B {fb}")
    assert got == fb, (got, fb)


def test_the_resample_space_is_finite_and_small_enough_to_enumerate(per_seed):
    """The fact that makes family B an estimator rather than a transcription
    error: at five seeds there are 3125 tuples and only 126 distinct means, so
    the exact percentile is not an approximation of anything."""
    d = [t - s for t, s in zip(per_seed["twin"], per_seed["settled"])]
    reps = [sum(t) / 5 for t in itertools.product(d, repeat=5)]
    print(f"\n  {len(reps)} resamples, {len(set(round(v, 12) for v in reps))} distinct means")
    assert len(reps) == 3125
    assert len(set(round(v, 12) for v in reps)) == 126


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_the_two_families_are_not_the_same_estimator(name, per_seed):
    """THE PLANTED NEGATIVE, and the reason this file is not circular.

    If the two procedures agreed everywhere, both tests above would pass with
    either mapping and the file would prove nothing about WHICH document uses
    WHICH. They must be shown to disagree on the real data, and the mis-pairing
    must be shown to fail."""
    x, y, fa, fb = PUBLISHED[name]
    mc = contrast(per_seed[x], per_seed[y], n_boot=10000, seed=0)
    ex = _exact(per_seed[x], per_seed[y])
    mc_r = (round(mc["ci_lo"], 6), round(mc["ci_hi"], 6))
    ex_r = tuple(round(v, 6) for v in ex)
    assert mc_r != ex_r, (
        f"{name}: the two procedures agree, so this file cannot tell which "
        f"document uses which and the labelling it argues for is unfalsifiable")
    # the mis-pairing must fail
    assert mc_r != fb and ex_r != fa, (mc_r, ex_r, fa, fb)
    print(f"\n  {name}: MC {mc_r} vs exact {ex_r} -- distinct, mis-pairing rejected")


def test_the_point_estimate_is_common_to_both_families(per_seed):
    """Both procedures resample the SAME paired deltas, so the point estimate
    cannot differ. If it ever does, the two families are reading different runs
    and this whole entry is the wrong diagnosis."""
    for name, (x, y, _fa, _fb) in sorted(PUBLISHED.items()):
        r = contrast(per_seed[x], per_seed[y], n_boot=10000, seed=0)
        d = [a - b for a, b in zip(per_seed[x], per_seed[y])]
        assert round(r["delta"], 6) == round(sum(d) / 5, 6), name
        print(f"  {name}: point {r['delta']:+.6f}")
