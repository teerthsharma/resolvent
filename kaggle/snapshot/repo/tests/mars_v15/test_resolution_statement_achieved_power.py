"""MARS standing attack #2 -- parity by underpowering.

FILED AT it.0, bound by Saturn, against `CEQ_V15_CONTRACT.md` PART I: "TOST
retires to `N >= 23` runs; below that, resolution statements only: 'excludes a
difference beyond `Delta = t_{.975, N-1} . sd / sqrt(N)` and nothing smaller'."

THE MECHANISM. A resolution statement bounds what a design COULD have detected;
it says nothing about whether the compared arms actually agree. Read casually --
"excludes a difference beyond Delta" -- it sounds like a parity claim, and the
smaller Delta looks, the more reassuring it reads. But Delta shrinking with N
says nothing about the design's ACHIEVED POWER to detect a difference smaller
than Delta, and at the round's own `N = 8` and in the `[23, 70)` band the
contract now licenses, achieved power against a modest true effect is low enough
that "no difference beyond Delta" is close to "this design could not have told
you either way." THE NUMBER THAT WOULD BE WRONG: any published resolution row
read as "PARITY" by a downstream table or prose sentence, when its own (sd, N)
imply single-digit-percent power to detect a plausibly-sized true difference.

THIS IS NOT HYPOTHETICAL ARITHMETIC. `V15_CONTRACT_ARITHMETIC_AUDIT.md` A-1,
filed in this same round, computes achieved power at `N=23` for two
BIT-IDENTICAL arms (the most favourable case that can exist) at **6.69%** under
the OLD TOST/two-sample framing, and states outright: "Mars's standing attack #2
... requires exactly that achieved-power column. The attack and the clause it
attacks are both in the same contract." `V15_LEDGER.md`'s own NEXT block, item 4,
instructs the coordinator to wire the two together. This test is that wiring.

CLASS. MISTAKES.md M (measurement failure). A SHARPER INSTANCE of M-13 ("an
equivalence margin registered without a reachability check") and M-9 ("a verdict
whose finest achievable p cannot reach the alpha it quotes"), not a new
mechanism: M-13 was filed against TOST; the contract's own fix for M-13 is to
retire TOST below N=23 and substitute resolution statements, and this attack
shows the substitution reopens the SAME disease in the new instrument's clothing
unless every row carries its achieved-power column, exactly as M-13's own
"Check" paragraph already prescribes ("must be divided by the design's own
standard error before it is registered").

STATUS, run at time of filing. No v15 resolution-statement row exists yet on
disk -- R4 (the two-sided parity table this format is for) is scheduled
`it.15-17`; `V15_LEDGER.md` NEXT is `it.3-4`. The census over `results/*.jsonl`
and the round's markdown below therefore finds zero rows and SKIPS the "every
published row" test rather than passing vacuously. `resolution_delta` and
`achieved_power` are exercised, cross-checked against an independent Monte Carlo
estimate, and must-fired against a planted row below -- live the day R4 lands.
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import random
import re
import statistics

import pytest
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]

ALPHA = 0.05
#: the house's own established "meaningful margin" convention -- `Delta_eq = 0.5
#: * sd` in `scale/it11_verdict.py:delta_eq` (M-13's own subject). Reused here
#: rather than invented, so the achieved-power column asks the same question the
#: repository already asks elsewhere: "power against a half-sigma true effect."
REFERENCE_EFFECT_IN_SD = 0.5


# ================================================================ THE INSTRUMENT
def resolution_delta(sd: float, n: int, alpha: float = ALPHA) -> float:
    """`Delta = t_{.975, N-1} . sd / sqrt(N)`, the contract's own formula,
    verbatim (PART I)."""
    if n < 2:
        raise ValueError(f"resolution_delta needs N>=2 and got {n}")
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return tcrit * sd / math.sqrt(n)


def achieved_power(sd: float, n: int, effect: float, alpha: float = ALPHA) -> float:
    """Analytic power of the one-sample two-sided t-test (df=N-1, the same test
    `resolution_delta` inverts) to detect a true shift of size `effect`, via the
    noncentral t distribution. `effect` and `sd` share units; `sd` cancels when
    `effect` is expressed as a multiple of `sd`.
    """
    if n < 2:
        raise ValueError(f"achieved_power needs N>=2 and got {n}")
    if sd <= 0:
        raise ValueError(f"achieved_power needs sd>0 and got {sd}")
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df=df)
    ncp = effect * math.sqrt(n) / sd
    return 1.0 - (stats.nct.cdf(tcrit, df, ncp) - stats.nct.cdf(-tcrit, df, ncp))


def achieved_power_at_reference(sd: float, n: int, alpha: float = ALPHA) -> float:
    """The achieved-power column this attack requires: power against the
    house's own `0.5*sd` reference effect. `sd` cancels algebraically, so this
    is a function of `N` (and `alpha`) alone -- exactly the shape of M-13's own
    power table, reused rather than re-derived."""
    return achieved_power(sd, n, REFERENCE_EFFECT_IN_SD * sd, alpha=alpha)


def _mc_power(sd: float, n: int, effect: float, alpha: float = ALPHA,
              sims: int = 20_000, seed: int = 0) -> float:
    """Independent Monte Carlo estimate: draw N normal(effect, sd) samples,
    run the one-sample t-test, count rejections. Exists so `achieved_power`
    is checked against a SECOND method sharing no code with it -- the
    discipline `V15_CONTRACT_ARITHMETIC_AUDIT.md` A-0d used to catch its own
    60k-draw error against a 4M-draw one, applied here before this instrument
    is trusted."""
    rng = random.Random(seed)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    hits = 0
    for _ in range(sims):
        xs = [rng.gauss(effect, sd) for _ in range(n)]
        m = statistics.fmean(xs)
        s = statistics.stdev(xs)
        se = s / math.sqrt(n)
        if se == 0:
            continue
        t = m / se
        if abs(t) > tcrit:
            hits += 1
    return hits / sims


# ==================================================================== SELF-CHECK
def test_achieved_power_at_zero_effect_is_alpha():
    """At the null (true effect 0) power must equal the test's own alpha --
    a two-sided t-test at the null rejects at exactly its stated rate by
    construction. This is a necessary, not sufficient, correctness check: it
    would not catch a formula error that happened to be symmetric, which is
    why the Monte Carlo cross-check below exists too."""
    for n in (8, 23, 40, 70):
        p = achieved_power(1.0, n, 0.0)
        assert abs(p - ALPHA) < 1e-9, (n, p)


@pytest.mark.parametrize("n,effect_in_sd", [(8, 0.5), (23, 0.5), (70, 0.5), (23, 1.0)])
def test_achieved_power_matches_independent_monte_carlo(n, effect_in_sd):
    """The formula and a from-scratch simulation must agree within MC noise.
    `sims=20_000` gives `se(power) <= 0.5/sqrt(20000) ~= 0.0035`; the bound
    below is 6 se, generous the way the repository's own MC-vs-analytic checks
    are (`V15_CONTRACT_ARITHMETIC_AUDIT.md` A-0d)."""
    sd = 1.0
    analytic = achieved_power(sd, n, effect_in_sd * sd)
    mc = _mc_power(sd, n, effect_in_sd * sd, sims=20_000, seed=n * 1000 + int(effect_in_sd * 10))
    assert abs(analytic - mc) < 0.03, (n, effect_in_sd, analytic, mc)


def test_achieved_power_at_the_rounds_registered_n8_is_alarmingly_low():
    """The number that matters most, because N=8 is not a hypothetical -- it is
    what PART IV actually registers: "N = 8 seeds; resolution statements." Every
    R1/R2/R5/R6 cell's resolution row is computed at this N.

    NOT the same quantity as `V15_CONTRACT_ARITHMETIC_AUDIT.md` A-1's 6.69% --
    that is the TWO-SAMPLE TOST framing (`Delta_eq=0.5*sigma`, SE has a factor
    `sqrt(2)`, df=2N-2, and TOST requires BOTH one-sided tests to reject, a
    strictly harder bar than a plain two-sided difference test). This module is
    the ONE-SAMPLE resolution-statement framing PART I actually licenses
    (df=N-1). The two numbers are expected to differ and do
    (6.69% vs the value asserted below) -- what they AGREE on is the point of
    this attack: at the round's registered N, a plausible half-sigma true
    difference is caught well under even odds.
    """
    p = achieved_power_at_reference(1.0, 8)
    assert 0.10 < p < 0.40, (
        f"achieved_power_at_reference(N=8) = {p:.4f}; expected roughly a "
        f"1-in-4-ish chance, not near-certain (>0.8) or effectively zero (<0.05)")


def test_achieved_power_climbs_toward_070_as_n_grows_toward_the_tost_floor():
    """Monotonicity sanity, and a floor at the contract's own retirement point:
    power at the reference effect should be materially higher by N=70 (where
    the contract's two-sample TOST first clears 0.80) than at N=23, though the
    one-sample number here is not required to hit 0.80 itself."""
    p23 = achieved_power_at_reference(1.0, 23)
    p70 = achieved_power_at_reference(1.0, 70)
    assert p70 > p23 + 0.3, (p23, p70)


# =========================================================== THE ROW-LEVEL RULE
class MissingAchievedPowerError(AssertionError):
    pass


def require_achieved_power_column(row: dict) -> float:
    """The rule this attack files: a published resolution row MUST carry a
    numeric `achieved_power` field, and that field must equal what
    `achieved_power_at_reference` computes from the row's OWN `sd` and `n` --
    not an arbitrary number, and not absent. Returns the verified power.
    """
    for key in ("sd", "n"):
        if key not in row:
            raise MissingAchievedPowerError(
                f"row {row!r} is missing '{key}'; achieved power cannot even be "
                f"computed, let alone checked")
    if "achieved_power" not in row or row["achieved_power"] is None:
        implied = achieved_power_at_reference(row["sd"], row["n"])
        raise MissingAchievedPowerError(
            f"row {row!r} publishes a resolution statement with no achieved-power "
            f"column. At its own (sd={row['sd']}, n={row['n']}) the achieved power "
            f"against a {REFERENCE_EFFECT_IN_SD}*sd true effect is {implied:.4f} "
            f"-- a reader has no way to tell whether 'excludes a difference beyond "
            f"Delta' means 'the arms agree' or 'this design is nearly blind'.")
    implied = achieved_power_at_reference(row["sd"], row["n"])
    published = float(row["achieved_power"])
    if abs(published - implied) > 1e-3:
        raise MissingAchievedPowerError(
            f"row {row!r} publishes achieved_power={published:.4f} but the row's "
            f"own (sd={row['sd']}, n={row['n']}) imply {implied:.4f}; the column "
            f"exists but does not answer for its own numbers")
    return published


def test_require_achieved_power_column_MUST_FIRE_on_a_planted_row_without_one():
    """FIRES. A resolution row shaped exactly like the contract's own example --
    N=8 (the round's registered seed count), a plausible sd -- with a Delta
    computed and printed, and no power column. Demonstrates both that the check
    raises and what it would have told a reader had it been there."""
    sd, n = 0.021, 8
    delta = resolution_delta(sd, n)
    row = {"cell": "R1 t*=2", "sd": sd, "n": n, "delta_res": delta}
    #: the row LOOKS complete -- it has its own margin printed -- which is
    #: exactly the shape that gets misread as a parity claim.
    assert "delta_res" in row and "achieved_power" not in row
    with pytest.raises(MissingAchievedPowerError, match="no achieved-power column"):
        require_achieved_power_column(row)
    implied = achieved_power_at_reference(sd, n)
    assert implied < 0.30, (
        f"fixture too weak to make the point: implied power {implied:.4f} is not "
        f"alarmingly low at N={n}")


def test_require_achieved_power_column_MUST_FIRE_on_a_planted_wrong_number():
    """FIRES. A row that HAS an `achieved_power` field, but one that does not
    follow from its own sd/n -- e.g. copy-pasted from a different cell. The
    column existing is not sufficient; it must answer for its own row."""
    row = {"cell": "R1 t*=2", "sd": 0.021, "n": 8, "achieved_power": 0.80}
    with pytest.raises(MissingAchievedPowerError, match="do not answer|imply"):
        require_achieved_power_column(row)


def test_require_achieved_power_column_passes_on_a_correctly_labelled_row():
    """The clean case: same row, honestly computed column attached."""
    sd, n = 0.021, 8
    row = {"cell": "R1 t*=2", "sd": sd, "n": n,
           "achieved_power": achieved_power_at_reference(sd, n)}
    got = require_achieved_power_column(row)
    assert 0.0 <= got <= 1.0


# =========================================================== THE LIVE REPO SCAN
#: the CONTRACT'S OWN WORDING, verbatim, plus the two field names a producer of
#: this exact row shape would plausibly use. Deliberately NOT the bare word
#: "resolution" -- measured against the live tree, that alone false-positives on
#: `results/journal_census.jsonl`'s "the resolution limit on any difference
#: between two cells swept at different thread counts", an unrelated float-
#: reduction-order finding that has nothing to do with TOST or parity. A search
#: keyed on a word this common is a V-4 waiting to happen (fired, wrong cause).
_RESOLUTION_ROW_HINT = re.compile(
    r"excludes a difference beyond|resolution_delta|delta_res\b", re.I)
_RESOLUTION_ROW_KEYS = {"delta_res", "resolution_delta", "resolution_statement"}


def _candidate_jsonl_rows() -> list[dict]:
    """Every JSON object under `results/*.jsonl` (parsed, never grepped -- this
    is a different file family from `house-events.jsonl` but the same
    discipline applies: `json.loads` per line, skip what doesn't parse) that
    carries the contract's own resolution-statement wording or field names."""
    out = []
    for path in sorted(glob.glob(str(ROOT / "results" / "*.jsonl"))):
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if (_RESOLUTION_ROW_KEYS & obj.keys()) or any(
                isinstance(v, str) and _RESOLUTION_ROW_HINT.search(v)
                for v in obj.values()
            ):
                out.append(obj)
    return out


def test_every_published_resolution_row_carries_a_correct_achieved_power_column():
    """The real attack, run against the live repository. SKIPS -- v15's R4 (the
    two-sided parity table this format is scoped to) has not run:
    `V15_LEDGER.md` NEXT is `it.3-4`, R4 is scheduled `it.15-17`. A scan of
    every `results/*.jsonl` file on disk for a resolution-shaped row finds none,
    which this test treats as an honest absence (MISTAKES.md V-7: a search that
    could not have found anything must not be read as a clean bill of health) --
    it SKIPS rather than passing on zero rows.
    """
    rows = _candidate_jsonl_rows()
    if not rows:
        pytest.skip(
            "no resolution-statement-shaped row found under results/*.jsonl; "
            "R4 has not run (V15_LEDGER.md NEXT=it.3-4, R4 scheduled it.15-17). "
            "See test_require_achieved_power_column_MUST_FIRE_* above for proof "
            "this check catches a missing/wrong power column on a planted row.")
    for row in rows:
        require_achieved_power_column(row)


def demo() -> None:
    p_null = achieved_power(1.0, 23, 0.0)
    assert abs(p_null - ALPHA) < 1e-9
    p23 = achieved_power_at_reference(1.0, 23)
    p70 = achieved_power_at_reference(1.0, 70)
    mc23 = _mc_power(1.0, 23, REFERENCE_EFFECT_IN_SD, sims=20_000, seed=1)
    row = {"sd": 0.021, "n": 8, "delta_res": resolution_delta(0.021, 8)}
    try:
        require_achieved_power_column(row)
    except MissingAchievedPowerError:
        pass
    else:
        raise AssertionError("must-fire fixture did not fire")
    print(f"demo OK: power(null)={p_null:.4f}==alpha, power(N=23,0.5sd)={p23:.4f} "
          f"(mc {mc23:.4f}), power(N=70,0.5sd)={p70:.4f}, planted row without a "
          f"power column correctly raised MissingAchievedPowerError")


if __name__ == "__main__":
    demo()
