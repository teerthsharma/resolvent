"""Pre-registered claims from scale/pivot_probe.py's module docstring, tested
against (1) the frozen results/m2.jsonl snapshot and (2) a live re-draw via
scale/mech_attack.py's draws() replay of run_arm's exact RNG stream.

Claims under test (quoted from scale/pivot_probe.py's docstring):

  TEST 1: "a dense two-hop sum spreads over ~s intermediates and gives 1/s"
          -> one token's mean |term_c| should fall off like s^-1.
  TEST 2: "The dense prediction is a-b near -1.389 (measured)" / the project
          quotes an exponent of -0.746 for the observed flip-rate decay ->
          if that is a real power law, the exponent should be the same in
          the first half of the s-range as in the second half.
  TEST 3: "One token's share of the two-hop sum becomes 1/k" / the plateau is
          set by "a k-term background [that] anti-concentrates as k^-1/2" ->
          deleting the k-term background from the flip statistic should move
          the flip rate by a lot.
  TEST 4: contrapositive of TEST 3 -> deleting the ONE-HOP term (not part of
          the k-term background) should barely move the flip rate.

TEST 1 and TEST 2 read results/m2.jsonl (frozen snapshot, not touched here)
and use scale.pivot_probe.loglog_slope (imported, not reimplemented) for every
slope. TEST 3 and TEST 4 share ONE live call to scale.mech_attack.draws (see
the module-scoped `mech_draws` fixture below) so the 90s wall-clock budget is
only charged once for the expensive draw loop.

DRAW COUNT: the spec asked for n_draws=4096. A standalone timing check (same
kind="pivot_signed", s=128, k=8, fast=True) measured ~37.6s wall clock for one
such call, plus ~13s one-time torch import -- about 51s total against TEST 3
and TEST 4's shared fixture, leaving headroom under the 90s ceiling. 4096 was
kept; NOT dropped to 2048. If a future run of this file blows the budget,
that is a regression to report, not something to silently paper over by
lowering n_draws without updating this docstring.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scale.pivot_probe import loglog_slope, clopper_pearson
from scale import mech_attack

ROOT = pathlib.Path(__file__).resolve().parents[2]
M2_JSONL = ROOT / "results" / "m2.jsonl"

N_DRAWS = 4096
FLOOR = 1e-6


# --------------------------------------------------------------------------
# results/m2.jsonl loading + aggregation (TEST 1, TEST 2)
# --------------------------------------------------------------------------

def _load_m2_rows():
    rows = []
    with open(M2_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _aggregate_by_s(rows, key_prefix):
    """Group journal rows whose `key` starts with `key_prefix + '/s'` by their
    s value. A key like '<prefix>/s2048/b3' is one of several batches of the
    same s (b0, b1, ...); a key like '<prefix>/s512' with no /bN suffix is a
    single batch. Returns {s: [value_dict, ...]}.
    """
    groups = defaultdict(list)
    prefix = key_prefix + "/s"
    for r in rows:
        k = r["key"]
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        s = int(rest.split("/")[0])
        groups[s].append(r["value"])
    return groups


def _weighted_mean_term(values):
    """Draw-weighted mean of value.term, weights value.n."""
    total_w = sum(v["n"] for v in values)
    total = sum(v["term"] * v["n"] for v in values)
    return total / total_w


def _pooled_rate(values):
    """rate = sum flips / sum draws, pooled across batches of the same s."""
    total_k = sum(v["k"] for v in values)
    total_n = sum(v["n"] for v in values)
    return (total_k / total_n if total_n else float("nan")), total_k, total_n


def _loglog_r2(xs, ys):
    """R^2 of the same log-log OLS fit loglog_slope() performs (same y>0
    filter, same points) -- loglog_slope returns only the slope, so this
    reimplements just the goodness-of-fit half of the identical regression,
    not the slope itself.
    """
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y > 0]
    if len(pts) < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((a - mx) ** 2 for a, _ in pts)
    syy = sum((b - my) ** 2 for _, b in pts)
    sxy = sum((a - mx) * (b - my) for a, b in pts)
    if sxx == 0 or syy == 0:
        return float("nan")
    return (sxy * sxy) / (sxx * syy)


def test_dense_term_dilutes_as_one_over_s():
    """CLAIM (pivot_probe.py docstring): "a dense two-hop sum spreads over ~s
    intermediates and gives 1/s" -> mean|term_c| should fall off like s^-1,
    i.e. loglog_slope(s, term) <= -0.5.
    """
    rows = _load_m2_rows()
    groups = _aggregate_by_s(rows, "dense_signed__at_pivots")
    ss = sorted(groups)
    terms = [_weighted_mean_term(groups[s]) for s in ss]

    slope, npts = loglog_slope(ss, terms)
    r2 = _loglog_r2(ss, terms)

    size_bytes = M2_JSONL.stat().st_size
    sha256 = hashlib.sha256(M2_JSONL.read_bytes()).hexdigest()

    per_s = ", ".join(f"s={s}:term={t!r}" for s, t in zip(ss, terms))
    msg = (
        f"CLAIM slope(term vs s) <= -0.5 (1/s dilution). "
        f"per-s draw-weighted-mean term: {per_s}. "
        f"loglog_slope={slope!r} (n_points_used={npts}/{len(ss)}) R^2={r2!r}. "
        f"results/m2.jsonl size_bytes={size_bytes} sha256={sha256}"
    )
    assert slope <= -0.5, msg


def test_dense_decay_is_a_power_law():
    """CLAIM (pivot_probe.py docstring): the dense arm's flip-rate decay has
    ONE exponent (the project quotes -0.746), i.e. a real power law, not a
    curve whose local slope changes across the s-range. Tested by comparing
    the OLS slope over the first three nonzero-rate s values against the OLS
    slope over the last three: abs(ratio) should sit near 1, generously
    bounded to [0.5, 2.0].
    """
    rows = _load_m2_rows()
    groups = _aggregate_by_s(rows, "dense_signed__at_pivots")
    ss_all = sorted(groups)
    pooled = {s: _pooled_rate(groups[s]) for s in ss_all}
    nonzero_ss = [s for s in ss_all if pooled[s][0] > 0]
    nonzero_rates = [pooled[s][0] for s in nonzero_ss]

    slope_all, n_all = loglog_slope(nonzero_ss, nonzero_rates)

    first3_s, first3_r = nonzero_ss[:3], nonzero_rates[:3]
    last3_s, last3_r = nonzero_ss[-3:], nonzero_rates[-3:]
    slope_first3, n_first3 = loglog_slope(first3_s, first3_r)
    slope_last3, n_last3 = loglog_slope(last3_s, last3_r)

    local_slopes = []
    for (s1, r1), (s2, r2) in zip(
        zip(nonzero_ss, nonzero_rates), zip(nonzero_ss[1:], nonzero_rates[1:])
    ):
        local_slopes.append((s1, s2, math.log(r2 / r1) / math.log(s2 / s1)))

    ratio = abs(slope_first3 / slope_last3) if slope_last3 not in (0, float("nan")) else float("nan")

    all_s_n = {s: (pooled[s][1], pooled[s][2]) for s in ss_all}
    msg = (
        f"CLAIM: same exponent in first half and second half of s-range. "
        f"all s -> (flips,n): {all_s_n}. "
        f"nonzero (s,rate): {list(zip(nonzero_ss, nonzero_rates))!r}. "
        f"(a) slope_all={slope_all!r} over s={nonzero_ss} (n_points={n_all}). "
        f"(b) slope_first3={slope_first3!r} over s={first3_s} rates={first3_r!r} (n_points={n_first3}). "
        f"(c) slope_last3={slope_last3!r} over s={last3_s} rates={last3_r!r} (n_points={n_last3}). "
        f"(d) local slopes ln(r2/r1)/ln(s2/s1): "
        + "; ".join(f"{s1}->{s2}:{ls!r}" for s1, s2, ls in local_slopes) + ". "
        f"ratio=abs(slope_first3/slope_last3)={ratio!r}"
    )
    assert 0.5 <= ratio <= 2.0, msg


# --------------------------------------------------------------------------
# live re-draw via scale.mech_attack.draws (TEST 3, TEST 4)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mech_draws():
    """ONE shared call to scale.mech_attack.draws, reused by TEST 3 and
    TEST 4 to stay inside the 90s wall-clock budget (see module docstring for
    the timing measurement behind N_DRAWS=4096).
    """
    return mech_attack.draws(
        kind="pivot_signed", s=128, n_draws=N_DRAWS, k=8, d=16,
        placement="in_P", protocol="SCALING", seed=0, fast=True,
    )


def _flip_stats(recs, formula):
    """formula(r) -> (lo, hi) already scaled by wo_sum. Floor applied to
    those wo_sum-scaled values, matching run_arm/mech_attack's own flip
    convention.
    """
    flips = 0
    for r in recs:
        lo, hi = formula(r)
        if lo * hi < 0 and min(abs(lo), abs(hi)) > FLOOR:
            flips += 1
    n = len(recs)
    return flips, n, (flips / n if n else float("nan"))


def _full(r):
    lo = (r["a_ij"] + r["bg"] + r["t1"]) * r["wo_sum"]
    hi = (r["a_ij"] + r["bg"] + r["t2"]) * r["wo_sum"]
    return lo, hi


def _no_background(r):
    lo = (r["a_ij"] + r["t1"]) * r["wo_sum"]
    hi = (r["a_ij"] + r["t2"]) * r["wo_sum"]
    return lo, hi


def _no_onehop(r):
    lo = (r["bg"] + r["t1"]) * r["wo_sum"]
    hi = (r["bg"] + r["t2"]) * r["wo_sum"]
    return lo, hi


def test_the_two_hop_background_is_what_the_plateau_measures(mech_draws):
    """CLAIM: the flat ~0.03 plateau is a property of the k-term two-hop
    background. Deleting the ENTIRE background (bg) from the flip statistic
    should move the flip rate by a lot: abs(FULL-NO_BACKGROUND)/FULL > 0.25.
    """
    flips_full, n, rate_full = _flip_stats(mech_draws, _full)
    flips_nobg, n2, rate_nobg = _flip_stats(mech_draws, _no_background)
    assert n == n2

    cp_full = clopper_pearson(flips_full, n)
    cp_nobg = clopper_pearson(flips_nobg, n2)
    rel_diff = abs(rate_full - rate_nobg) / rate_full if rate_full else float("inf")

    msg = (
        f"n_draws_requested={N_DRAWS} n_draws_used={n} "
        f"FULL: rate={rate_full!r} flips={flips_full} CP95={cp_full!r}. "
        f"NO_BACKGROUND: rate={rate_nobg!r} flips={flips_nobg} CP95={cp_nobg!r}. "
        f"rel_diff=abs(FULL-NO_BACKGROUND)/FULL={rel_diff!r}"
    )
    assert rel_diff > 0.25, msg


def test_the_one_hop_term_is_not_what_the_plateau_measures(mech_draws):
    """CLAIM (contrapositive of TEST 3): the plateau is NOT a property of the
    one-hop term a_ij. Deleting a_ij from the flip statistic should barely
    move the flip rate: abs(FULL-NO_ONEHOP)/FULL <= 0.25.
    """
    flips_full, n, rate_full = _flip_stats(mech_draws, _full)
    flips_noonehop, n2, rate_noonehop = _flip_stats(mech_draws, _no_onehop)
    assert n == n2

    cp_full = clopper_pearson(flips_full, n)
    cp_noonehop = clopper_pearson(flips_noonehop, n2)
    rel_diff = abs(rate_full - rate_noonehop) / rate_full if rate_full else float("inf")

    msg = (
        f"n_draws_requested={N_DRAWS} n_draws_used={n} "
        f"FULL: rate={rate_full!r} flips={flips_full} CP95={cp_full!r}. "
        f"NO_ONEHOP: rate={rate_noonehop!r} flips={flips_noonehop} CP95={cp_noonehop!r}. "
        f"rel_diff=abs(FULL-NO_ONEHOP)/FULL={rel_diff!r}"
    )
    assert rel_diff <= 0.25, msg
