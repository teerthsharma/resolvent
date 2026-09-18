"""Wald's sequential probability ratio test, and the K1 slope-to-rate mapping.

======================================================================
PART 1 OF THE PRE-REGISTRATION: THE MAPPING, WRITTEN BEFORE ANY DRAW
======================================================================

THE PROBLEM. K1's flip clause is a statement about a LOG-LOG SLOPE:

    CHECKLIST.md:371 -- "DUAL SLOPE. `flip(s)` slope <= -0.4 on X4 AND `D_FR`
    slope >= -0.1 on the SAME draws."

The sweep variable in the code that reads this clause is the PIVOT COUNT `k`,
not the context length `s`: `scale/arm_a_k1.py` fits `loglog_slope(ks, ...)`
over `--ks 8 16 32 64 128 256` at a single `--s 1024`. The clause is written
`flip(s)` and measured `flip(k)`. This document uses `k`, which is what the
instrument actually varies, and records the discrepancy rather than resolving
it: nothing here re-reads the clause at varying `s`.

An SPRT, by contrast, is a statement about a PER-DRAW BERNOULLI RATE. A slope is
not a rate, so a mapping is required, and the contract (1.4) requires it in
writing before a single datum lands.

THE MAPPING. Assume the flip rate follows a power law in the pivot count:

    r(k) = r_ref * (k / k_ref) ** b                                        (M1)

with `b` the log-log slope the clause is about. (M1) is the ONLY functional form
under which "the slope" is a well-defined single number, and it is the form
`loglog_slope` already fits, so adopting it here changes no object that has been
measured -- it names the object the existing fit assumes.

(M1) has two free parameters and the clause constrains only one of them, so an
ANCHOR is required. The anchor is pre-registered as round 5's k=8 cell:

    k_ref    = 8
    r_ref    = 6 / 400 = 0.015          (CHECKLIST.md:573, "6/0/0/1/1/0 per 400")

Under (M1) the anchor cell is FIXED under every slope: r(k_ref) = r_ref for all
`b`. Two consequences, both load-bearing:

  * the k=8 cell carries ZERO information about `b`, so re-using round 5's count
    of it as the anchor does not re-use that count as evidence; and
  * every informative cell is k > k_ref.

THE TWO HYPOTHESES. K1's bar is `b <= -0.4`. The two slopes that bound the
question are therefore

    b_MET  = -0.4     the bar exactly met
    b_FLAT =  0.0     no decay at all

and (M1) turns them into a pair of per-draw rates at each cell:

    r0(k) = r_ref * (k / 8) ** (-0.4)      "K1's flip clause is met"
    r1(k) = r_ref * (k / 8) ** ( 0.0)      "the rate does not decay"
          = r_ref

Because -0.4 < 0, r0(k) < r1(k) at every k > 8, which is what the contract's
`H0: r <= r0` vs `H1: r >= r1` requires. NOTE THE LABELLING, because it is the
opposite of the intuitive one: H0 is the hypothesis that the CLAUSE IS MET, and
accepting H1 is the outcome in which the clause FAILS. Both directions are worth
+1 under K-H; the labels decide only which sentence gets written.

At the primary cell k = 256 the mapping gives, in closed form,

    r0(256) = 0.015 * 32 ** (-0.4) = 0.015 / 4 = 0.00375
    r1(256) = 0.015
    r1 / r0 = 4 exactly, because 32 ** 0.4 = 2 ** 2 = 4.

WHAT THIS TEST DOES *NOT* SAY -- four limits, collected here and not repeated:

  1. It is a test between TWO POINTS, not between two regions. Accepting H0
     means the stream favours `b = -0.4` over `b = 0.0` at the stated error
     rates. It does NOT establish `b <= -0.4`. Every slope strictly between the
     two is an INDIFFERENCE REGION in which Wald controls neither error rate;
     the calibration table below measures what the procedure actually does
     there rather than assuming it behaves.
  2. The anchor is a point estimate (6 events in 400 draws, itself CP95
     [0.0055, 0.0323]) used as a CONSTANT. An anchor that is too high inflates
     both hypothesised rates together; because the test compares them to each
     other, a common multiplicative error moves the decision less than it moves
     either rate, but it is not neutral and it is not modelled here.
  3. The power-law form (M1) is assumed, not tested. A rate that falls off a
     cliff between two cells satisfies no slope.
  4. Round 5's counts at k > 8 (0, 0, 1, 1, 0 per 400) have been SEEN by the
     author of this mapping. They are used to set nothing, and the anchor cell
     is uninformative by construction, but the design was chosen by someone who
     had read them. The SPRT therefore runs on FRESH draws under a fresh
     declared seed; replaying round 5's journal through these boundaries would
     be a test of data that were generated before the boundaries existed.

IMMUTABILITY. Every constant in the block below is fixed from the moment the
first fresh draw lands. `tests/chase/test_sprt_k1.py` pins each one by value.

======================================================================
PART 2: THE MACHINERY
======================================================================

    Lambda_n = x * log(r1/r0) + (n - x) * log((1-r1)/(1-r0))

    accept H1 at  Lambda_n >= log((1-beta)/alpha)
    accept H0 at  Lambda_n <= log(beta/(1-alpha))

At (alpha, beta) = (0.05, 0.05) both thresholds reduce to +-log(19), because
(1-beta)/alpha = 0.95/0.05 = 19 and beta/(1-alpha) = 0.05/0.95 = 1/19.

G8. The stopping decision is two comparisons of a sum against a constant. No
product enters it, so the `lo*hi` underflow class cannot reach the boundary.

NO THREAD DEPENDENCE, AND NO TORCH. Nothing here calls BLAS; the simulation is
pure Python integer and float arithmetic, so the module imports only the
standard library and a consumer of `Sprt` takes no torch dependency (issue #3).
"""
from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# THE PRE-REGISTERED CONSTANTS. Immutable once the first fresh draw lands.
# ---------------------------------------------------------------------------
K_REF = 8                                #: anchor cell, round 5's k=8
K_GRID = (8, 16, 32, 64, 128, 256)       #: scale/arm_a_k1.py's --ks default
R_ANCHOR = 6.0 / 400.0                   #: 0.015 -- CHECKLIST.md:573
B_MET = -0.4                             #: K1's bar, CHECKLIST.md:371
B_FLAT = 0.0                             #: the alternative: no decay
ALPHA = 0.05
BETA = 0.05
K_PRIMARY = 256                          #: the cell the K1 SPRT runs at

#: Round 5's fixed-n price for the same question, for the saving line.
FIXED_N_PER_CELL = 20000


def rate_at(k: int, b: float, *, k_ref: int = K_REF,
            r_ref: float = R_ANCHOR) -> float:
    """r(k) = r_ref * (k / k_ref) ** b -- equation (M1) of the mapping."""
    return r_ref * math.exp(b * math.log(k / k_ref))


def hypotheses(k: int) -> tuple[float, float]:
    """(r0, r1) at cell `k`: the rate under K1's bar, and under no decay.

    r0 < r1 at every k > K_REF, and r0 == r1 == R_ANCHOR at k == K_REF, where
    the two hypotheses are indistinguishable by construction.
    """
    return rate_at(k, B_MET), rate_at(k, B_FLAT)


def verdict_text(decision: str | None) -> str:
    """The sentence each outcome licenses. Pre-registered so it cannot drift."""
    return {
        "H1": "K1 FLIP CLAUSE FAILS: slope shallower than -0.4",
        "H0": "K1 FLIP CLAUSE MET: slope at least as steep as -0.4",
        None: "UNDECIDED: no boundary crossed within the cap",
    }[decision]


def thresholds(alpha: float = ALPHA, beta: float = BETA) -> tuple[float, float]:
    """(lower, upper) = (log(beta/(1-alpha)), log((1-beta)/alpha))."""
    return (math.log(beta / (1.0 - alpha)),
            math.log((1.0 - beta) / alpha))


class Sprt:
    """One sequential test of `H0: r <= r0` against `H1: r >= r1`.

    `update(x)` folds in one Bernoulli observation and returns the running
    Lambda. `decision` is "H0", "H1", or None while no boundary is crossed.
    """

    def __init__(self, r0: float, r1: float, *, alpha: float = ALPHA,
                 beta: float = BETA):
        if not (0.0 < r0 < 1.0 and 0.0 < r1 < 1.0):
            raise ValueError(f"rates out of range: r0={r0} r1={r1}")
        self.r0, self.r1 = r0, r1
        self.lo, self.hi = thresholds(alpha, beta)
        #: the two per-draw log-likelihood-ratio increments, formed once
        self.inc_one = math.log(r1 / r0)
        self.inc_zero = math.log((1.0 - r1) / (1.0 - r0))
        self.n = 0
        self.x = 0
        self.lam = 0.0

    def update(self, x: int) -> float:
        self.n += 1
        if x:
            self.x += 1
            self.lam += self.inc_one
        else:
            self.lam += self.inc_zero
        return self.lam

    @property
    def decision(self) -> str | None:
        if self.lam >= self.hi:
            return "H1"
        if self.lam <= self.lo:
            return "H0"
        return None


def _oc_root(p: float, r0: float, r1: float) -> float:
    """The nonzero h solving E[exp(h*z)] = 1 for the per-draw increment z.

    Used only by `expected_n` for rates other than r0 and r1, where Wald's
    acceptance probability has no closed form. Bisection, because the function
    is convex with exactly two roots and one of them is h = 0.
    """
    a, b = r1 / r0, (1.0 - r1) / (1.0 - r0)

    def g(h):
        return p * a ** h + (1.0 - p) * b ** h - 1.0

    drift = p * math.log(a) + (1.0 - p) * math.log(b)
    if abs(drift) < 1e-15:
        return 0.0
    lo, hi = (0.0, 60.0) if drift < 0 else (-60.0, 0.0)
    # walk off the h = 0 root before bisecting
    step = 1e-6 if drift < 0 else -1e-6
    lo_or_hi = step
    if g(lo_or_hi) * g(hi if drift < 0 else lo) > 0:
        return 0.0
    a_, b_ = (lo_or_hi, hi) if drift < 0 else (lo, lo_or_hi)
    for _ in range(200):
        m = 0.5 * (a_ + b_)
        if g(a_) * g(m) <= 0:
            b_ = m
        else:
            a_ = m
    return 0.5 * (a_ + b_)


def expected_n(p: float, r0: float, r1: float, alpha: float = ALPHA,
               beta: float = BETA) -> float:
    """Wald's E[N] at true rate `p`. Ignores overshoot, so it is a lower-ish
    approximation; `calibrate` reports the simulated mean beside it."""
    lo, hi = thresholds(alpha, beta)
    drift = p * math.log(r1 / r0) + (1.0 - p) * math.log((1.0 - r1) / (1.0 - r0))
    if abs(drift) < 1e-15:
        return float("inf")
    if p == r0:
        p1 = alpha
    elif p == r1:
        p1 = 1.0 - beta
    else:
        h = _oc_root(p, r0, r1)
        if h == 0.0:
            return float("inf")
        aa, bb = math.exp(hi) ** h, math.exp(lo) ** h
        p1 = (1.0 - bb) / (aa - bb)
    return (p1 * hi + (1.0 - p1) * lo) / drift


def calibrate(p: float, r0: float, r1: float, *, n_rep: int = 4000,
              max_n: int = 200000, seed: int = 0, alpha: float = ALPHA,
              beta: float = BETA) -> dict:
    """Run the procedure against a KNOWN answer: Bernoulli(p) streams.

    Returns the realised decision frequencies and mean sample size. A sequential
    test that has never been run against a known answer is not an instrument, so
    every number this module produces is downstream of this function.
    """
    rng = random.Random(seed)
    n_h0 = n_h1 = n_trunc = 0
    total_n = 0
    for _ in range(n_rep):
        t = Sprt(r0, r1, alpha=alpha, beta=beta)      # module-global on purpose
        d = None
        for _i in range(max_n):
            t.update(1 if rng.random() < p else 0)
            d = t.decision
            if d is not None:
                break
        total_n += t.n
        if d == "H0":
            n_h0 += 1
        elif d == "H1":
            n_h1 += 1
        else:
            n_trunc += 1
    return dict(p=p, r0=r0, r1=r1, n_rep=n_rep, max_n=max_n, seed=seed,
                accept_h0=n_h0 / n_rep, accept_h1=n_h1 / n_rep,
                truncated=n_trunc / n_rep, mean_n=total_n / n_rep)


# ---------------------------------------------------------------------------
# the plan, printed
# ---------------------------------------------------------------------------

def print_plan() -> None:
    lo, hi = thresholds()
    print("=== K1 SPRT PRE-REGISTRATION (mapping fixed before any fresh draw) ===")
    print(f"  anchor: k_ref={K_REF}  r_ref={R_ANCHOR!r} (6/400, CHECKLIST.md:573)")
    print(f"  H0 slope b_MET={B_MET}   H1 slope b_FLAT={B_FLAT}")
    print(f"  alpha={ALPHA} beta={BETA}   thresholds "
          f"[{lo:.6f}, {hi:.6f}]   +-log(19)={math.log(19.0):.6f}")
    print(f"  |upper - log(19)| = {abs(hi - math.log(19.0)):.3e}   "
          f"(contract 1.4 requires < 5e-5)")
    print(f"  primary cell k={K_PRIMARY};  fixed-n price for the same question "
          f"= {FIXED_N_PER_CELL} draws/cell")
    print()
    print(f"{'k':>6} {'r0 (b=-0.4)':>14} {'r1 (b=0)':>12} {'r1/r0':>8} "
          f"{'inc(x=1)':>11} {'inc(x=0)':>12} {'E[N|H0]':>11} {'E[N|H1]':>11} "
          f"{'saving vs 20000':>16}")
    for k in K_GRID:
        r0, r1 = hypotheses(k)
        if r0 == r1:
            print(f"{k:>6} {r0:>14.8f} {r1:>12.8f} {1.0:>8.4f} "
                  f"{0.0:>11.6f} {0.0:>12.6f} {'inf':>11} {'inf':>11} "
                  f"{'anchor: no info':>16}")
            continue
        t = Sprt(r0, r1)
        n0 = expected_n(r0, r0, r1)
        n1 = expected_n(r1, r0, r1)
        print(f"{k:>6} {r0:>14.8f} {r1:>12.8f} {r1 / r0:>8.4f} "
              f"{t.inc_one:>11.6f} {t.inc_zero:>12.6f} {n0:>11.1f} {n1:>11.1f} "
              f"{FIXED_N_PER_CELL / max(n0, n1):>15.1f}x")


def print_calibration(n_rep: int = 4000, max_n: int = 200000) -> None:
    r0, r1 = hypotheses(K_PRIMARY)
    mid = math.sqrt(r0 * r1)
    print(f"\n=== SPRT CALIBRATION at k={K_PRIMARY}  r0={r0:.8f} r1={r1:.8f}  "
          f"n_rep={n_rep} ===")
    print(f"{'stream':>22} {'p':>12} {'accept H0':>10} {'accept H1':>10} "
          f"{'truncated':>10} {'mean N':>10} {'Wald E[N]':>10} {'expected':>28}")
    rows = (("far below r0", r0 / 2.0, "accept H0"),
            ("at r0 (H0 true)", r0, f"accept H1 <= alpha={ALPHA}"),
            ("between r0 and r1", mid, "indifference: both occur"),
            ("at r1 (H1 true)", r1, f"accept H0 <= beta={BETA}"),
            ("far above r1", r1 * 2.0, "accept H1"))
    for i, (name, p, want) in enumerate(rows):
        c = calibrate(p, r0, r1, n_rep=n_rep, max_n=max_n, seed=100 + i)
        w = expected_n(p, r0, r1)
        print(f"{name:>22} {p:>12.8f} {c['accept_h0']:>10.4f} "
              f"{c['accept_h1']:>10.4f} {c['truncated']:>10.4f} "
              f"{c['mean_n']:>10.1f} {w:>10.1f} {want:>28}")


def print_must_fire(n_rep: int = 500) -> None:
    """The control. A decider with the boundary logic DELETED must fail this
    table, or the table above measures nothing."""
    global Sprt
    r0, r1 = hypotheses(K_PRIMARY)
    real = Sprt

    class Blind(real):
        @property
        def decision(self):
            return "H0"

    print(f"\n=== MUST-FIRE: the same calibration with the decision deleted ===")
    good = calibrate(r1, r0, r1, n_rep=n_rep, max_n=2000, seed=900)
    try:
        Sprt = Blind
        bad = calibrate(r1, r0, r1, n_rep=n_rep, max_n=2000, seed=900)
    finally:
        Sprt = real
    print(f"  stream at r1, real test    accept_h0={good['accept_h0']:.4f}  "
          f"accept_h1={good['accept_h1']:.4f}  "
          f"(beta bar {BETA}: {'PASS' if good['accept_h0'] <= BETA + 0.02 else 'FAIL'})")
    print(f"  stream at r1, data-blind   accept_h0={bad['accept_h0']:.4f}  "
          f"accept_h1={bad['accept_h1']:.4f}  "
          f"(beta bar {BETA}: {'PASS' if bad['accept_h0'] <= BETA + 0.02 else 'FAIL'})")
    print("  CONTROL SEEN TO FIRE" if bad["accept_h0"] > BETA + 0.02 else
          "  CONTROL DID NOT FIRE -> the calibration measures nothing")


def main() -> int:
    print_plan()
    print_calibration()
    print_must_fire()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
