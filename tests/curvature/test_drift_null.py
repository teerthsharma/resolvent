"""RED-FIRST tests for ceqjepa/drift_null.py (PHASE C it.1, the T-DRIFT null).

WHAT THESE TESTS ARE FOR. The module under test claims to say whether a drift
number is large. A drift number without a null is not a number, and every test
below is a way for the null to be WRONG while the code still returns a p-value:

  (i) THE NULL IS NOT REPRODUCIBLE. A null that moves between runs at the same
     seed cannot be audited, and any threshold read off it is a story.
  (ii) THE FALSE-ALARM RATE IS NEVER MEASURED. A detector whose false-alarm rate at
     its stated threshold is asserted rather than counted is not a detector. The
     rate is counted here, over many repetitions, with a Wilson interval, and a
     MISCALIBRATED detector is planted and must be caught by the same counter --
     otherwise the counter is the thing that is broken.
  (iii) "IT FIRED" IS NOT POWER. One planted effect that fires says nothing about the
     effects that do not. The floor -- the smallest planted drift caught at 80% --
     is what the module owes, and a zero-size plant must NOT reach it.
  (iv) THE CHEAP NULL IS NOT THE EXPENSIVE NULL. The whole refusal of the cost
     trade-off rests on the cached null being the SAME distribution as the one a
     recompute-per-permutation loop produces, not a close one. Tested as exact
     equality plus a call count, and a perturbed cache must break it.
  (v) THE RECORDED NEGATIVE IS QUIETLY DELETED. The negative-fraction-only detector
     failed (0.251 against 0.219 in the owner's lineage). A failure that is removed
     from the file once the file goes green is a failure that gets re-shipped.

THE IMPORT IS GUARDED ON PURPOSE, matching tests/curvature/test_curvature_instrument.py:
a module-level import turns the RED phase into one collection error with no test
names in it. Each test must be seen to FAIL BY NAME before the module exists.

RUN: python -m pytest tests/curvature/test_drift_null.py -v
"""

import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

try:
    from ceqjepa import drift_null as dn
except Exception as _exc:                      # noqa: BLE001 -- RED phase carries it
    dn = None
    _IMPORT_ERROR = _exc


def _dn():
    """The module under test, or a named failure saying it is not there yet."""
    if dn is None:
        raise AssertionError(
            "ceqjepa/drift_null.py did not import: %r" % (_IMPORT_ERROR,))
    return dn


ALPHA = 0.05
N_PERM = 199


# ---------------------------------------------------------------------------
# Synthetic statistics with KNOWN behaviour. Nothing here needs curvature: the
# module takes the statistic as a callable, so it is tested today against
# statistics whose null and whose response to a plant are both known in advance.
# ---------------------------------------------------------------------------

def _iid_blocks(rng, n_blocks=16, size=8, shift=0.0, n_new=4):
    """n_blocks exchangeable blocks; the LAST n_new get a mean shift of `shift`."""
    blocks = [rng.normal(size=size) for _ in range(n_blocks)]
    for b in range(n_blocks - n_new, n_blocks):
        blocks[b] = blocks[b] + shift
    return blocks


def _mean_gap(new, ref):
    """|mean(new) - mean(ref)|. Large under a shift, exchangeable without one."""
    return abs(float(np.mean([np.mean(b) for b in new]))
               - float(np.mean([np.mean(b) for b in ref])))


# ---------------------------------------------------------------------------
# 1. SEEDS AND DETERMINISM
# ---------------------------------------------------------------------------

def test_the_null_is_reproducible_from_a_pinned_seed():
    m = _dn()
    rng = np.random.default_rng(0)
    blocks = _iid_blocks(rng)
    a = m.permutation_null(_mean_gap, blocks, n_new=4, n_perm=N_PERM, seed=20260913)
    b = m.permutation_null(_mean_gap, blocks, n_new=4, n_perm=N_PERM, seed=20260913)
    c = m.permutation_null(_mean_gap, blocks, n_new=4, n_perm=N_PERM, seed=20260914)
    assert np.array_equal(np.asarray(a), np.asarray(b)), \
        "the same seed gave two different nulls: the null is not reproducible"
    assert not np.array_equal(np.asarray(a), np.asarray(c)), \
        "two different seeds gave the identical null: the seed is not wired in"
    assert len(a) == N_PERM, "the null does not hold n_perm draws (%d)" % len(a)


# ---------------------------------------------------------------------------
# 2. CALIBRATION, WITH A MISCALIBRATED DETECTOR PLANTED
# ---------------------------------------------------------------------------

def test_the_false_alarm_rate_is_measured_with_a_ci_at_a_stated_threshold():
    m = _dn()

    def trial(rng):
        blocks = _iid_blocks(rng, shift=0.0)
        dist = m.permutation_null(_mean_gap, blocks, n_new=4, n_perm=N_PERM,
                                  seed=int(rng.integers(1 << 30)))
        obs = _mean_gap(blocks[-4:], blocks[:-4])
        return bool(m.detect(obs, dist, alpha=ALPHA).fired)

    far, lo, hi, n_rep = m.false_alarm_rate(trial, n_rep=300, seed=20260913)
    print("\n  same-distribution FAR at alpha=%.2f: %.4f  95%% CI [%.4f, %.4f]  (%d reps)"
          % (ALPHA, far, lo, hi, n_rep))
    assert lo <= ALPHA <= hi, (
        "measured false-alarm rate %.4f CI [%.4f, %.4f] does not cover the nominal "
        "%.2f" % (far, lo, hi, ALPHA))


def test_a_miscalibrated_detector_is_caught_by_the_same_measurement():
    """PLANTED NEGATIVE for the calibration check: a detector thresholding at the
    null's MEDIAN instead of its upper tail fires about half the time on
    same-distribution data. The same counter must reject it, or the counter is
    what is broken and test 2's green means nothing."""
    m = _dn()

    def broken_trial(rng):
        blocks = _iid_blocks(rng, shift=0.0)
        dist = np.asarray(m.permutation_null(_mean_gap, blocks, n_new=4,
                                             n_perm=N_PERM,
                                             seed=int(rng.integers(1 << 30))))
        obs = _mean_gap(blocks[-4:], blocks[:-4])
        return bool(obs > np.median(dist))          # the plant: wrong threshold

    far, lo, hi, _ = m.false_alarm_rate(broken_trial, n_rep=300, seed=20260913)
    print("\n  PLANTED miscalibration (threshold at the null median): FAR %.4f "
          "CI [%.4f, %.4f]" % (far, lo, hi))
    assert not (lo <= ALPHA <= hi), (
        "the FAR measurement ACCEPTS a detector thresholding at the null median "
        "(FAR %.4f CI [%.4f, %.4f]): the measurement cannot fail" % (far, lo, hi))


# ---------------------------------------------------------------------------
# 3. POWER, REPORTED AS A FLOOR
# ---------------------------------------------------------------------------

def test_the_detection_floor_is_reported_not_just_that_it_fired():
    m = _dn()

    def power_at(shift, n_rep=150, seed=7):
        def trial(rng):
            blocks = _iid_blocks(rng, shift=shift)
            dist = m.permutation_null(_mean_gap, blocks, n_new=4, n_perm=N_PERM,
                                      seed=int(rng.integers(1 << 30)))
            obs = _mean_gap(blocks[-4:], blocks[:-4])
            return bool(m.detect(obs, dist, alpha=ALPHA).fired)
        return m.false_alarm_rate(trial, n_rep=n_rep, seed=seed)[0]

    sizes = (0.0, 0.25, 0.5, 1.0, 2.0)
    powers = {s: power_at(s) for s in sizes}
    for s in sizes:
        print("\n  planted shift %.2f -> power %.3f" % (s, powers[s]))
    mde = m.detection_floor(powers, target=0.8)
    print("  detection floor at 80%% power: %s" % mde)

    assert powers[0.0] < 0.15, (
        "a ZERO-size plant reaches power %.3f: the 'power' being reported is the "
        "false-alarm rate" % powers[0.0])
    assert powers[2.0] > 0.8, \
        "a 2.0-sigma planted drift is missed (power %.3f): the check is vacuous" % powers[2.0]
    assert powers[0.0] < powers[1.0] <= 1.0, "power does not rise with effect size"
    assert mde is not None and 0.0 < mde <= 2.0, \
        "no detection floor is reported for a sweep that reaches power 1"


# ---------------------------------------------------------------------------
# 4. SAME-DISTRIBUTION INJECTIONS MUST NOT FIRE
# ---------------------------------------------------------------------------

def test_same_distribution_injections_do_not_fire():
    """An injection at a dated index t0 drawn from the SAME distribution is the
    thing a drift detector is most likely to call. The rate is published."""
    m = _dn()
    fires, n_rep = 0, 200
    rng = np.random.default_rng(4242)
    for _ in range(n_rep):
        base = _iid_blocks(rng, shift=0.0, n_new=0)
        inject = [rng.normal(size=8) for _ in range(4)]      # same distribution
        blocks = base[:12] + inject                          # dated index t0 = 12
        dist = m.permutation_null(_mean_gap, blocks, n_new=4, n_perm=N_PERM,
                                  seed=int(rng.integers(1 << 30)))
        obs = _mean_gap(blocks[12:], blocks[:12])
        fires += int(m.detect(obs, dist, alpha=ALPHA).fired)
    far = fires / n_rep
    lo, hi = m.wilson(fires, n_rep)
    print("\n  same-distribution INJECTION at t0=12: %d/%d fired, rate %.4f "
          "CI [%.4f, %.4f]" % (fires, n_rep, far, lo, hi))
    assert lo <= ALPHA <= hi or far < ALPHA, (
        "same-distribution injections fire at %.4f CI [%.4f, %.4f] against a "
        "nominal %.2f" % (far, lo, hi, ALPHA))


# ---------------------------------------------------------------------------
# 5. THE REFUSED TRADE-OFF: the cached null IS the recomputed null
# ---------------------------------------------------------------------------

def test_the_cached_null_is_the_recomputed_null_exactly_and_costs_B_calls_not_n_perm():
    """The claim that makes the cheap null admissible is EQUALITY, not similarity.

    A permutation loop that recomputes the expensive per-block map every draw
    calls it n_perm * B times. The cached null calls it B times. If the two nulls
    are not identical the saving was bought with an approximation, which the
    owner's rule strikes."""
    m = _dn()
    rng = np.random.default_rng(11)
    blocks = _iid_blocks(rng)

    calls = {"expensive": 0, "cheap": 0}

    def block_map_expensive(b):
        calls["expensive"] += 1
        return float(np.mean(b))

    def block_map_cheap(b):
        calls["cheap"] += 1
        return float(np.mean(b))

    def combine(new_s, ref_s):
        return abs(float(np.mean(new_s)) - float(np.mean(ref_s)))

    slow = m.permutation_null(
        lambda new, ref: combine([block_map_expensive(b) for b in new],
                                 [block_map_expensive(b) for b in ref]),
        blocks, n_new=4, n_perm=N_PERM, seed=20260913)
    fast = m.cached_null(block_map_cheap, combine, blocks, n_new=4,
                         n_perm=N_PERM, seed=20260913)

    print("\n  expensive per-block-map calls: %d    cached: %d    ratio %.1fx"
          % (calls["expensive"], calls["cheap"],
             calls["expensive"] / max(calls["cheap"], 1)))
    assert np.array_equal(np.asarray(slow), np.asarray(fast)), (
        "the cached null is NOT the recomputed null: max |diff| = %.3e"
        % float(np.abs(np.asarray(slow) - np.asarray(fast)).max()))
    assert calls["cheap"] == len(blocks), (
        "the cached null called the expensive map %d times for %d blocks"
        % (calls["cheap"], len(blocks)))
    assert calls["expensive"] == N_PERM * len(blocks), (
        "the recompute baseline is not paying full price (%d calls)" % calls["expensive"])

    # PLANTED NEGATIVE for this equality: perturb one cached summary and the two
    # nulls must part. An equality check that cannot fail proves nothing.
    seen = []

    def block_map_perturbed(b):
        seen.append(b)
        # only the FIRST block's summary is wrong, by 0.5
        return float(np.mean(b)) + (0.5 if len(seen) == 1 else 0.0)

    bad = m.cached_null(block_map_perturbed, combine, blocks, n_new=4,
                        n_perm=N_PERM, seed=20260913)
    assert not np.array_equal(np.asarray(slow), np.asarray(bad)), \
        "perturbing one cached block summary left the null unchanged: it is not used"


# ---------------------------------------------------------------------------
# 6. THE RECORDED NEGATIVE
# ---------------------------------------------------------------------------

def test_the_negative_fraction_alone_does_not_separate_a_new_phase():
    """The owner's recorded negative, reproduced. A new phase adds a COMPONENT,
    not a bridge, so the fraction of negative-curvature edges barely moves while
    beta_0 moves by one. The composite residual on the SAME data is the control:
    without it, a null result here would only say the bed is weak."""
    m = _dn()
    out = m.recorded_negative(seed=20260913)
    b0 = out["beds"][0]
    print("\n  bed 0: negative fraction reference %.3f  drifted %.3f  (owner's "
          "lineage 0.219 -> 0.251)" % (b0["neg_frac_ref"], b0["neg_frac_new"]))
    print("  bed 0: beta_0 %d -> %d, negfrac null sd %.4f"
          % (b0["beta0_ref"], b0["beta0_new"], b0["negfrac_null_sd"]))
    print("  POWER over %d beds: negfrac %.2f | unit-weight %.2f | studentised %.2f"
          % (out["n_beds"], out["power_negfrac"], out["power_unit"],
             out["power_stud"]))

    # Reported as POWER, not as one p-value: a single bed where the negative
    # fraction happens not to fire would be seed shopping.
    assert out["power_negfrac"] <= 0.5, (
        "the negative-fraction-only detector reaches power %.2f: the recorded "
        "negative no longer reproduces" % out["power_negfrac"])
    assert out["power_stud"] >= 0.8, (
        "the composite also fails on these beds (power %.2f): the beds carry no "
        "drift, so the negative-fraction null result is not evidence"
        % out["power_stud"])
    assert out["power_stud"] > out["power_negfrac"], \
        "the two detectors are indistinguishable: nothing is being separated"
    assert b0["beta0_new"] > b0["beta0_ref"], \
        "the planted new phase did not add a component: the bed is not the mechanism"


def test_the_recorded_negative_stays_in_the_module_source():
    """A failure deleted once the file goes green is a failure that gets
    re-shipped. The lineage pair and the reason must be IN the module."""
    src = (ROOT / "ceqjepa" / "drift_null.py")
    assert src.exists(), "ceqjepa/drift_null.py does not exist"
    text = src.read_text(encoding="utf-8")
    assert "0.251" in text and "0.219" in text, \
        "the owner's lineage pair 0.251 / 0.219 is not recorded in the module"
    assert re.search(r"RECORDED NEGATIVE", text), \
        "the module does not carry a RECORDED NEGATIVE section"
    assert re.search(r"component", text, re.I) and re.search(r"bridge", text, re.I), \
        "the module does not state the REASON (a new phase adds a component, not a bridge)"


# ---------------------------------------------------------------------------
# 7. EVERY NUMBER IN PROSE MUST COME FROM A RUN
# ---------------------------------------------------------------------------

# Adapted from tests/curvature/test_curvature_instrument.py::
# test_every_measured_number_in_a_docstring_is_printed_by_the_demo, then repaired
# twice, IN THIS ORDER, because the order is not stylistic.
#
#   FIRST the matcher. The original asks `tok not in out`, which is a SUBSTRING
#   test, so a token is satisfied by any longer number that happens to contain it:
#   the Inspector planted 1.14522 and the guard passed, because the run prints the
#   arXiv id 2111.14522. Every check below is only as good as this matcher.
#
#   ONLY THEN the pattern. Widening to bare integers before fixing the matcher
#   would have added checks that cannot fail -- measured across the four modules,
#   of 98 bare integers already in docstrings 0 are absent as a SUBSTRING while 22
#   are absent as a TOKEN. Widen first and all 98 pass, and the widening looks free
#   precisely because it is doing nothing.
_MEASURED = re.compile(r"[+-]?\d+\.\d+e[+-]\d+"       # 1.250e-01
                       r"|[+-]?\d+\.\d+%"              # 12.4%
                       r"|[+-]?\d+\.\d+x"              # 199.0x
                       r"|[+-]?\d+\.\d+"               # 0.0149
                       r"|[+-]?\d{1,3}(?:,\d{3})+"      # 40,317
                       r"|[+-]?\d+")                    # 40317


def _token_in(tok, out):
    """True when `tok` stands alone in `out`, not buried inside a longer number.

    The lookarounds separate a planted token from a longer number the run
    legitimately prints and that happens to contain it -- a fake null spread
    sitting inside an arXiv id, or a short decimal inside the real one. A digit,
    a dot or a thousands comma on either side means the run printed a DIFFERENT
    number. Both cases are asserted as planted negatives in the guard test below,
    which is a stronger form of example than one narrated here: no exemption list
    means this docstring may not carry a numeral the demo does not print either.
    """
    return re.search(r"(?<![\d.,])" + re.escape(tok) + r"(?![\d.,]*\d)",
                     out) is not None


def _docstrings_of(mod):
    """The module docstring plus every callable DEFINED IN IT, public or private.

    By __module__ rather than __all__: __all__ cannot see demo() itself, and
    demo() is where an unprinted number survives longest once the public surface
    is clean.
    """
    docs = [("module", mod.__doc__ or "")]
    for name in sorted(dir(mod)):
        obj = getattr(mod, name)
        if (callable(obj) and getattr(obj, "__module__", None) == mod.__name__
                and getattr(obj, "__doc__", None)):
            docs.append((name, obj.__doc__))
    return docs


def _missing(docs, out):
    """Every docstring token the run does not print AS A TOKEN. One code path,
    used by the real check and by both planted negatives, so the plants exercise
    the thing that ships."""
    return [(where, tok) for where, doc in docs
            for tok in _MEASURED.findall(doc) if not _token_in(tok, out)]


def test_every_measured_number_in_a_docstring_is_printed_by_the_demo(capsys):
    """The check whose absence let three reported numbers come from no run.

    Every decimal in the module docstring, and in the docstring of every public
    callable, must appear VERBATIM in the output of `python -m ceqjepa.drift_null`.
    A number that no run prints cannot be checked by anyone, its author included,
    and drifts silently from the moment the bed changes. There is no exemption
    list: a citation or a constant that wants to live in prose has to be printed
    like everything else.
    """
    m = _dn()
    m.demo()
    out = capsys.readouterr().out

    # Scanned by __module__, not by __all__. Scanning __all__ misses demo() itself
    # and every private helper, which is exactly where an unprinted number hides
    # once the public surface is clean. The Inspector's cross-module sweep found
    # three different guards across four modules, and the narrowest of them is how
    # a stray 1.06 escaped in chess_steps.
    # P5: the guard scans THE FILE IT LIVES IN as well. No guard in this round
    # did, so a measured number in a test docstring was unbound in all four
    # modules -- including in the tests that police the other numbers.
    docs = _docstrings_of(m) + _docstrings_of(sys.modules[__name__])
    missing = _missing(docs, out)
    checked = sum(len(_MEASURED.findall(doc)) for _, doc in docs)
    print("\n  %d numbers across %d docstrings (module + this test file), "
          "%d printed by no run" % (checked, len(docs), len(missing)))
    assert checked >= 40, \
        "only %d numbers found: the pattern is not biting" % checked
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))

    # PLANTED NEGATIVE 1, SUBSTRING VACUITY. 1.14522 is printed by no run, but it
    # sits inside the arXiv id 2111.14522 that IS printed. Under the original
    # `tok not in out` the guard accepted it. The same _missing used above must
    # reject it, or every green in this test is a collision away from meaningless.
    collision = [("planted", "a null sd of 1.14522 that no run produced")]
    assert "1.14522" in out, "the collision plant needs 2111.14522 in the output"
    assert _missing(collision, out), \
        "SUBSTRING VACUITY: the guard accepts 1.14522 because 2111.14522 is printed"

    # PLANTED NEGATIVE 2, a bare INTEGER absent as a token. This is the check the
    # widening buys, and it only buys it because the matcher was fixed first.
    assert _missing([("planted", "over 40317 permutations")], out), \
        "the guard does not see bare integers: the widening is doing nothing"
    assert _missing([("planted", "a spread of 77.31")], out), \
        "the guard does not see an absent decimal at all"
    print("  planted negatives fired: 1.14522 (collision), 40317, 77.31")


_PRICE_LINE = re.compile(r"MEASURED IN THIS RUN:\s*([\d.]+)\s*ms")


def _price_varies(fn, reps=60):
    """(varies, a, b) from two timings in ONE run. A constant returns varies=False.

    Used by the real check and by the planted literal alike, so the plant
    exercises the shipped code path rather than a copy of it.
    """
    a, b = float(fn(reps=reps)), float(fn(reps=reps))
    return a != b, a, b


def _published_price(out):
    """The price the demo actually printed, or None."""
    hit = _PRICE_LINE.search(out)
    return hit.group(1) if hit else None


def test_the_retired_figure_is_named_on_the_page_and_printed_by_the_run(capsys):
    """A retirement that does not name the retired number is a deletion with a note.

    The withdrawal of 0.0018 -- the null spread the group-mean negative-fraction
    form reported under a bed configuration this file no longer ships -- has to
    survive as a FIGURE a reader can match against the record, not as a paragraph
    saying that some earlier number was wrong.

    This collides with the docstring guard above, which requires every decimal in
    prose to be printed by the run: writing 0.0018 into a docstring makes that
    guard fail. The resolution is not to weaken either check but to satisfy both --
    the demo PRINTS the retirement, so the figure is on the page, in stdout, and
    beside the measured pair that replaced it.
    """
    m = _dn()
    src = (ROOT / "ceqjepa" / "drift_null.py").read_text(encoding="utf-8")
    assert "0.0018" in src, \
        "the retired figure is named nowhere in the module: that is a deletion"
    m.demo()
    out = capsys.readouterr().out
    assert "RETIRED" in out, "the run does not mark a retirement"
    assert "0.0018" in out, \
        "the retired figure 0.0018 is not printed by the run, so a reader cannot " \
        "tell which number was withdrawn"
    for replacement in ("0.0047", "0.0149"):
        assert replacement in out, (
            "the measured replacement %s is not printed beside the retired figure"
            % replacement)
    print("\n  retirement printed: 0.0018 named, replaced by 0.0047 / 0.0149")


def test_the_lp_price_behind_the_reprice_is_measured_in_the_run(capsys):
    """STRUCK 1's repair, bound. The reprice arithmetic must be derived from a
    price this process measured, not from a literal sitting in a print string.

    The module must expose the timing entry point, the demo must print a measured
    price, and the two must agree in order of magnitude on the same box -- a
    hardcoded constant cannot follow the machine and would drift the moment the
    run moved.
    """
    m = _dn()
    assert hasattr(m, "measure_lp_price"), \
        "the module has no measure_lp_price: the reprice rests on a literal again"

    # THE DISCRIMINATING PART. A wall clock never returns the same value twice; a
    # constant cannot help but. The old form bounded the price by 0.05 < p < 50.0,
    # a thousandfold band that ANY literal in three orders of magnitude satisfies,
    # and the Inspector passed it with 1.1 -- two significant figures of the real
    # measurement. Variation is the property a literal cannot fake.
    varies, a, b = _price_varies(m.measure_lp_price)
    # PLANTED NEGATIVE: the Inspector's own defeat of the previous version.
    planted_varies, pa, _ = _price_varies(lambda reps=200: 1.1)

    # The PUBLISHED price must be the measured one, not a constant printed beside
    # a measured one, so two runs of the demo must print different prices. capsys
    # is DRAINED first and every diagnostic printed AFTER the last readouterr: a
    # print issued before it is swallowed into the captured buffer and nobody ever
    # sees it, which is precisely the failure this round keeps striking.
    capsys.readouterr()
    m.demo()
    out1 = capsys.readouterr().out
    m.demo()
    out2 = capsys.readouterr().out
    p1, p2 = _published_price(out1), _published_price(out2)

    print("\n  two timings in one run : %.6f ms and %.6f ms  (differ = %s)"
          % (a, b, varies))
    print("  PLANTED literal 1.1 ms : %.6f twice, differ = %s -- rejected"
          % (pa, planted_varies))
    print("  published across 2 runs: %s ms and %s ms" % (p1, p2))

    assert varies, (
        "two calls to measure_lp_price returned the IDENTICAL value %.6f: that is "
        "a constant, not a measurement" % a)
    assert 0.2 < a < 30.0, \
        "the measured LP price %.4f ms is not plausible for an exact transport LP" % a
    assert not planted_varies, \
        "a hardcoded 1.1 ms 'measurement' still satisfies the variation check"
    assert "MEASURED IN THIS RUN" in out1, \
        "the demo does not mark its LP price as measured in this run"
    assert p1 is not None and p2 is not None, "no published price line to read"
    assert p1 != p2, (
        "the demo published the identical price %s on two runs: the printed "
        "number is not the measured one" % p1)


# ---------------------------------------------------------------------------
# 8. THE SELF-CHECK
# ---------------------------------------------------------------------------

def test_the_self_check_runs_under_120s_and_ends_with_the_banner():
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "ceqjepa.drift_null"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    dt = time.time() - t0
    print("\n  python -m ceqjepa.drift_null -> exit %d in %.1f s" % (p.returncode, dt))
    assert p.returncode == 0, "the self-check failed:\n%s\n%s" % (p.stdout[-3000:],
                                                                  p.stderr[-3000:])
    assert dt < 120.0, "the self-check took %.1f s, over the 120 s budget" % dt
    assert p.stdout.strip().splitlines()[-1].strip() == "ALL SELF-CHECKS PASSED", \
        "the self-check does not end with the banner:\n%s" % p.stdout[-800:]
