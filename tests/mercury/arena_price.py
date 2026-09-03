"""Pure pricing arithmetic for the v20 R15 arena. MERCURY, it.3.

Nothing here reads a file, opens a device or trains anything: rows in, numbers
out. That is deliberate -- every statistic below can then be run against a
PLANTED input, which is the only way to show a statistic returns a reading
rather than a constant (`tests/saturn/test_v20_r15_wing_rubric.py:433-448`
makes the same argument for `seconds_to_floor`, and that function is copied
here VERBATIM so the order-invariance comparison is against the shipped one and
not a paraphrase of it).

MEASURED BASIS. `PER_CELL_150_STEPS` and `FIXED_PROCESS_SECONDS` are READ from
`V17_R4_RETAKE_PRICE.md:194-196` and `:198-200` -- six CUDA cells, one process,
on the certified RTX 4060 Laptop (`V16_DEVICE_CERT.md:112,117`, sm_89,
torch 2.5.1+cu121). Everything `arena_seconds` does to them is multiplication,
and that is the whole extrapolation: N seeds x the per-cell second, plus one
fixed process cost. It assumes cells run sequentially in one process, which is
how the retake was taken.
"""
import math

from scipy.stats import beta, binom, fisher_exact

#: `V17_R4_RETAKE_PRICE.md:194-196` -- seconds per cell at 150 steps, mean of
#: two seeds, BED-M shape (t*=2, n_train=2048, s=64, d_model=16) on cuda.
PER_CELL_150_STEPS = {"arm_smprime": 15.970, "arm_pl": 1.614, "softmax": 1.497}

#: `V17_R4_RETAKE_PRICE.md:198-200` -- a 1-cell run carries 3.5 s of non-cell
#: time; per-cell overhead outside the training loop is under 0.05 s and is
#: NOT modelled here (it is inside the measured per-cell seconds already).
FIXED_PROCESS_SECONDS = 3.5


# ------------------------------------------------------------------ crossings

def crossing_table(cells, floor):
    """{kind: (crossings, n)} using the shipped `<= floor` comparison."""
    out = {}
    for r in cells:
        x, n = out.get(r["kind"], (0, 0))
        out[r["kind"]] = (x + (1 if r["eval_nrmse"] <= floor else 0), n + 1)
    return out


def clopper_pearson_lower(x, n, alpha=0.05):
    """Two-sided 95% Clopper-Pearson LOWER bound. Exactly 0.0 at x == 0."""
    return 0.0 if x == 0 else float(beta.ppf(alpha / 2.0, x, n - x + 1))


def n_for_cp_lower(rate, target, rule="round", alpha=0.05, n_max=2000):
    """Smallest (N, k) with CP-lower(k, N) > target, holding the point rate.

    `rule` is the integer rule used to turn `rate * N` into a count, and it is
    a parameter rather than a constant because THE ANSWER DEPENDS ON IT.
    """
    f = {"round": lambda v: int(round(v)), "ceil": math.ceil,
         "floor": math.floor}[rule]
    for n in range(1, n_max):
        k = f(rate * n)
        if 0 <= k <= n and clopper_pearson_lower(k, n, alpha) > target:
            return n, k
    raise ValueError("no N below %d" % n_max)


def power_for_cp_lower(n, true_rate, target, alpha=0.05):
    """P(CP-lower(K, n) > target) with K ~ Binomial(n, true_rate).

    The N that a point estimate "clears" is a coin flip; this is what the
    design actually buys.
    """
    kmin = next((k for k in range(n + 1)
                 if clopper_pearson_lower(k, n, alpha) > target), None)
    return 0.0 if kmin is None else float(binom.sf(kmin - 1, n, true_rate))


# ------------------------------------------------- the restated clause (EXIT B)

def fisher_vs_skyline(arm, skyline, alternative="greater"):
    """Fisher exact on 2x2 [[x_arm, miss_arm], [x_sky, miss_sky]].

    Fisher rather than a two-proportion z: the skyline cell is an exact ZERO,
    where the normal approximation's variance term p(1-p)/n vanishes and the
    statistic is undefined. Fisher conditions on the margins and needs no
    variance estimate, so it is valid at 0/N where the z-test is not.
    """
    x_a, n_a = arm
    x_s, n_s = skyline
    return float(fisher_exact([[x_a, n_a - x_a], [x_s, n_s - x_s]],
                              alternative=alternative)[1])


def power_fisher_vs_skyline(n, true_rate, alpha=0.05, alternative="greater"):
    """P(reject) at true arm rate `true_rate` against a skyline fixed at 0/n."""
    return float(sum(
        binom.pmf(k, n, true_rate)
        for k in range(n + 1)
        if fisher_vs_skyline((k, n), (0, n), alternative) < alpha))


# ------------------------------------------------------------------- criterion (3)

def seconds_to_floor(rows, floor):
    """VERBATIM from `tests/saturn/test_v20_r15_wing_rubric.py:433-448`.

    Copied rather than imported so the order-invariance comparison is against
    the shipped statistic. Sums `secs` in ITERATION order to the first crossing.
    """
    out = {}
    for r in rows:
        kind, secs = r["kind"], r.get("secs", 0.0)
        acc, hit = out.get(kind, (0.0, None))
        acc += secs
        if hit is None and r["eval_nrmse"] <= floor:
            hit = acc
        out[kind] = (acc, hit)
    return {k: hit for k, (acc, hit) in out.items()}


def expected_cost_to_crossing(rows, floor):
    """MARS's repriced statistic: mean cost of a CROSSING cell x (n / crossed).

    Order-invariant by construction -- a mean and two counts are symmetric
    functions of the multiset of rows, so no permutation moves it. `None` for
    an arm with no crossing: UNDEFINED, not large, same convention as above.
    """
    agg = {}
    for r in rows:
        n, x, s = agg.get(r["kind"], (0, 0, 0.0))
        crossed = r["eval_nrmse"] <= floor
        agg[r["kind"]] = (n + 1, x + (1 if crossed else 0),
                          s + (r.get("secs", 0.0) if crossed else 0.0))
    return {k: (None if x == 0 else (s / x) * (n / x))
            for k, (n, x, s) in agg.items()}


# ------------------------------------------------------------------- the price

def arena_seconds(n_seeds, per_cell=None, fixed=FIXED_PROCESS_SECONDS):
    """GPU-seconds for `n_seeds` seeds x every arm in `per_cell`, one process.

    The extrapolation, stated in full: sum(per-cell seconds) * n_seeds + fixed.
    `n_seeds` is the arena's N; cells run sequentially so PEAK BYTES do not
    move with N and the 8188 MiB device bound is unaffected.
    """
    per_cell = PER_CELL_150_STEPS if per_cell is None else per_cell
    return sum(per_cell.values()) * n_seeds + fixed
