"""it.17 MERCURY — the Phase C arena rig behind
`tests/mercury/test_v20_r15_it15_arena_rig.py`.

Three things this module refuses to do, each one a priced finding carried IN
rather than filed in a report a reader may not open:

1. It never emits one tail of clause (1) alone. `cp_lower_both_tails` returns
   both readings labelled, with `ruling="UNRULED"`, so no office can pick the
   tail after seeing which one wins. That temptation is `M-2`.
2. Every row states the run-order confound in its own output. `secs` is
   confounded with run order (published `rho = +0.7029`, this loader `+0.6971`)
   AND with arm directly, because `early_warning` runs ~16 arm-dependent
   forward passes inside the timed window (`scripts/v15_r1.py:712,:747`).
3. Absent columns are `None` WITH A REASON, never a plausible number.
"""
import json
import math
import pathlib
import statistics

FLOOR_1 = math.sqrt((2 - 1) / 2)  #: scripts/v15_r1.py:586, T_STAR = 2.

#: Spearman rho of seed against `secs` over the 16 `arm_smprime` cells, from
#: THIS loader (retake-first), recomputed — not quoted. The published figure is
#: `+0.7029` (V20_R15_IT8_JUPITER.md:130), which reproduces only under a
#: fresh-file-first loader that exists nowhere in the tree. Every committed
#: loader is retake-first (tests/jupiter/test_v20_r15_it12_constants.py:61-69),
#: and retake-first gives +0.6971. Both are carried; neither is preferred here.
RUN_ORDER_RHO = 0.697059
PUBLISHED_RUN_ORDER_RHO = "+0.7029"
GATE_RHO = "+0.5197"

#: The 40-cell bank. Order is retake-first, matching every committed loader.
BANK = ("results/v17k_r4_retake.jsonl",
        "results/v20_r15_it6_seeds8_15.jsonl",
        "results/v20_r15_it8_armpl_b.jsonl")

RUN_ORDER_CONFOUND = (
    f"secs is confounded with run order (published rho = "
    f"{PUBLISHED_RUN_ORDER_RHO} seed-vs-secs, this loader {RUN_ORDER_RHO:+.4f}), "
    f"stronger than the gate correlation {GATE_RHO} it must be separated from; "
    f"and confounded with arm directly, because early_warning runs ~16 "
    f"arm-dependent forward passes inside the timed window "
    f"(scripts/v15_r1.py:712,:747). There is no torch.cuda.synchronize() in the "
    f"runner, so every secs on CUDA is un-synchronised host wall clock."
)


class Reading:
    """A number, or `None` and the reason there is no number.

    The reason is the point. `dist_to_skyline` is `None` on every banked cell
    and a rig that filled it with a plausible float would send the leap chasing
    a ghost at it.35.
    """

    __slots__ = ("value", "why", "first_crossing_index")

    def __init__(self, value=None, why=None, first_crossing_index=None):
        if (value is None) == (why is None):
            raise ValueError("a Reading is a value XOR a reason, never both")
        self.value, self.why = value, why
        self.first_crossing_index = first_crossing_index

    def __repr__(self):
        return (f"Reading({self.value!r})" if self.why is None
                else f"Reading(None, why={self.why!r})")


# ------------------------------------------------------------------ THE RULE
def crosses(cell):
    """The runner's own rule at `scripts/v15_r1.py:909`, per cell.

    `crosses = bool(m + half < floor1)`. For a cell the interval is the
    bootstrap, so `m + half` is `boot_hi`. This is NOT `eval_nrmse < floor`:
    that lazy twin fires on the near miss and is wrong.
    """
    return bool(cell["boot_hi"] < FLOOR_1)


def gpu_seconds_to_floor(cells):
    """Cumulative `secs` through the FIRST crossing cell, or `None` and why."""
    total = 0.0
    for i, c in enumerate(cells):
        total += c["secs"]
        if crosses(c):
            r = Reading(total)
            r.first_crossing_index = i
            return r
    r = Reading(None, why=(
        f"no cell crosses floor_1={FLOOR_1:.4f} under the runner's own rule "
        f"(m + half < floor1, v15_r1.py:909): 0 of {len(cells)} cells. The "
        f"instrument declines rather than interpolate a crossing that is not "
        f"in the data."))
    r.first_crossing_index = None
    return r


# --------------------------------------------------------------- THE PLANTS
def _cell(nrmse, half, secs):
    return {"eval_nrmse": nrmse, "boot_lo": nrmse - half, "boot_hi": nrmse + half,
            "secs": secs, "kind": "planted", "dist_to_skyline": None,
            "dist_to_skyline_why": "no v15/v16 scan-skyline module exists (R-SKY)"}


def plant_crossing(at, secs):
    """A crossing planted at index `at`. Every other cell sits above the floor.

    The instrument has never been exercised on a positive it did not find for
    itself. One that cannot detect a crossing it was TOLD is there measures
    nothing (V-16).
    """
    out = []
    for i, s in enumerate(secs):
        out.append(_cell(FLOOR_1 - 0.10, 0.01, s) if i == at
                   else _cell(FLOOR_1 + 0.10, 0.01, s))
    return out


def plant_non_crossing(secs, near_miss=False):
    """The other half of the control. An instrument that fires on everything
    also detects a crossing it was told is there.

    `near_miss=True` is the sharp negative: the point estimate sits BELOW the
    floor and only the bootstrap upper limit reaches above it. The plants then
    differ from `plant_crossing` in the bootstrap width alone.
    """
    if near_miss:
        return [_cell(FLOOR_1 - 0.10, 0.20, s) for s in secs]
    return [_cell(FLOOR_1 + 0.10, 0.01, s) for s in secs]


# ------------------------------------------------------------------ CLAUSE 1
def _beta_ppf(p, a, b):
    """Inverse regularised incomplete beta by bisection on `math.lgamma`.

    No scipy dependency for one number, and no numerical surprise: the CDF is
    monotone, so 200 bisections give full double precision.
    """
    def cdf(x):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta) / a
        # ponytail: Lentz continued fraction, 300 terms; plenty for n <= 2000.
        f, c, d = 1.0, 1.0, 0.0
        for i in range(300):
            m, num = i // 2, 0.0
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            c = 1.0 + num / c
            c = 1e-30 if abs(c) < 1e-30 else c
            f *= c * d
            if abs(1.0 - c * d) < 1e-15:
                break
        return front * (f - 1.0)

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cp_lower_both_tails(x, n, alpha=0.05, target=0.5):
    """BOTH readings of clause (1), side by side, always.

    Clause (1) is `BED-M crossing CP-lower > 0.5`
    (CEQ_V20_R15_CONTRACT.md:125). The contract never fixes the tail, and at
    12/16 the two readings DISAGREE — two-sided fails by 0.0238, one-sided
    clears. Emitting one alone would let an office pick the tail after seeing
    which wins. So both ship, labelled, with `ruling="UNRULED"`: this office
    prices and builds, it does not rule.
    """
    two = 0.0 if x == 0 else _beta_ppf(alpha / 2.0, x, n - x + 1)
    one = 0.0 if x == 0 else _beta_ppf(alpha, x, n - x + 1)
    return {"x": x, "n": n, "alpha": alpha, "target": target,
            "two_sided": two, "one_sided": one,
            "verdict_two_sided": "CLEARS" if two > target else "FAILS",
            "verdict_one_sided": "CLEARS" if one > target else "FAILS",
            "ruling": "UNRULED",
            "why_unruled": ("the contract fixes no tail for clause (1) and the "
                            "two readings disagree at this count; MERCURY "
                            "prices and builds, it does not rule (M-2)")}


# ------------------------------------------------------------------ THE BANK
def load_cells(root):
    """The banked Phase C cells, retake-first, de-duplicated by cell id.

    Retake owns seeds 0-7; the it.6 and it.8 journals add seeds 8-15 and repeat
    seeds 0/1 as in-run reproduction controls, which are the SAME cell and not
    a 17th. This is the loader every committed test already uses
    (tests/jupiter/test_v20_r15_it12_constants.py:61-69). It reads only; it
    writes nothing.
    """
    root = pathlib.Path(root)
    seen = {}
    for rel in BANK:
        for line in (root / rel).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("t") == "cell" and "dist_to_skyline_why" in r:
                seen.setdefault(r["cell"], r)
    return list(seen.values())


def spearman_rho(x, y):
    """Spearman rho with midranks. Recomputed from the cells, never quoted."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    a, b = ranks(list(x)), ranks(list(y))
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den


# ------------------------------------------------------------------- THE ROW
#: Columns the contract names at CEQ_V20_R15_CONTRACT.md:119-124, each with the
#: reason it has no object. A column with no object gets `None` and this text.
_ABSENT = {
    "dist_to_skyline": ("no v15/v16 scan-skyline module exists (R-SKY); the "
                        "cells carry dist_to_skyline=None on every row and this "
                        "rig will not manufacture a distance to a skyline that "
                        "was never computed"),
    "w1_to_oracle": ("both wings emit a POINT prediction, so there is no state "
                     "distribution for a Wasserstein-1 distance to have as its "
                     "object; the column has no object, not a missing value"),
    "peak_bytes": ("the runner journals peak_wset_gib once per run at the t=wall "
                   "record, host-wide and not per cell; no per-cell peak bytes "
                   "was ever measured"),
    "cert_grade": ("no mask certificate is emitted by either arm in this bed, so "
                   "there is no certificate for L-CERT to grade"),
}


def arena_row(cells, arm):
    """One arena row. Every column the contract names, and the confound."""
    n = len(cells)
    x = sum(crosses(c) for c in cells)
    below = [c["eval_nrmse"] for c in cells if crosses(c)]
    row = {
        "arm": arm,
        "n": n,
        "crossings": x,
        "cp_lower": Reading(cp_lower_both_tails(x, n)),
        "conditional_nrmse": (Reading(statistics.fmean(below)) if below else
                              Reading(None, why=(
                                  "conditional on crossing, and no cell in this "
                                  "arm crosses, so the conditioning set is empty"))),
        "dist_to_floor": Reading(statistics.fmean(
            c["eval_nrmse"] - FLOOR_1 for c in cells)),
        "gpu_seconds_to_floor": gpu_seconds_to_floor(cells),
        "run_order_confound": RUN_ORDER_CONFOUND,
    }
    for col, why in _ABSENT.items():
        row[col] = Reading(None, why=why)
    return row
