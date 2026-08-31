# V15 n4 -- BED-K and the interventional channel

CAMERON/SATURN, node n4 of the CEQ v15 composition round (D-1 parallel-safe
set: n1 Jupiter prior art, n2 Saturn's D-3 read, n3 Jupiter's Lean items,
n4 this file). Scope per dispatch: `ceq/beds/**`, `tests/beds/**`, and this
file. No existing module touched -- confirmed by `git status --porcelain`
before and after: the only new paths are `ceq/beds/`, `tests/beds/`, and
`V15_N4_BEDK.md`.

## Post-GREEN revision: three cross-node findings applied

The coordinator relayed three findings from parallel nodes after the section
below was already green (10/10). All three were in-scope (`ceq/beds/**`,
`tests/beds/**`) and are now applied; the suite is re-run at the end of this
section and stands at 15/15. What follows states what changed and what each
finding's own report request asked for, verbatim where it matters.

**1. Positional channel -- now explicit, "yes."** `build()`'s manifest gained
a `pos = arange(n)` field, documented in its own docstring paragraph. Before
this change, position was only implicit in array order (available to
anything that bothered to use the index, but not a field a caller could
point to). `hard_delay_attention` now takes `pos` as an explicit parameter
and reads ONLY `b` and `pos` -- the two fields a real arm could actually be
handed -- rather than assuming index order or reaching into `bed["K"]`. The
must-fire (test 2) now calls it as `hard_delay_attention(bed["b"], d,
pos=bed["pos"])`, so "attention-reachable" is a claim about a channel a real
arm could be given, not a claim about what only this test script can see.
Numerically nothing changed (`pos` still equals `arange(n)`), but the
channel is now a stated field of the corpus rather than an assumption buried
in a test helper. **What is NOT addressed, and is out of this node's scope:**
whether `scale/m3_capability.py`'s existing arms actually read a positional
feature at all. That file is untouched (hard constraint: write only under
`ceq/beds/**`, `tests/beds/**`). If an arm is fed `bed["b"]` alone, a
delay-bed failure at R3 will still be confounded between "no positional
feature" and "no memory kernel" exactly as flagged -- BED-K now exposes the
channel needed to resolve that, but does not and cannot force a downstream
arm to consume it.

**2. Grunwald-Letnikov weights -- implemented via the ratio recurrence, not
`scipy.special.binom`.** Added `gl_weights(alpha, k_max)`:
`psi_0=1; psi_k=psi_{k-1}*(alpha+k-1)/k`. Verified independently before
using it (`scipy.special.binom(-alpha, k)` vs the recurrence):

```
alpha=0.3:  mine  [1.0, 0.3, 0.195, 0.1495, 0.1233375, 0.10607025]
            scipy [1.0, 0.30000000000000004, 0.19500000000000003,
                   0.14949999999999988, 0.12333749999999989, 0.10607024999999991]
alpha=1.0:  mine  [1. 1. 1. 1. 1. 1.]
            scipy [nan, nan, nan, nan, nan, nan]
```

Matches to float equality at alpha=0.3 (differences are last-bit rounding
noise from the two different recurrences, not a disagreement); at alpha=1
scipy returns all-NaN (the `binom(-1, k)` negative-integer branch) where the
recurrence returns the correct `[1,1,1,...]` (coefficients of
`(1-z)^{-1} = sum z^k`). `_powerlaw_kernel_matrix` now builds
`K[i, i-k] = psi_k` for `k=1..i` (still strictly `j<i`, no self-term, per the
contract's own formula) instead of the crude `(i-j)^(H-1.5)` applied
directly at every lag. `gl_weights`' docstring states explicitly that this is
the FRACTIONAL INTEGRAL kernel (`(-1)^k C(-alpha,k)`, coefficients of
`(1-z)^{-alpha}`), not the GL derivative, which has the opposite sign --
BED-K's power-law bed is long-memory/integrated (`H>0.5`), so the integral
kernel is the correct one and this was checked before writing the sign, not
after.

**3. The alpha box -- the generator now raises, under test.** `alpha = H -
0.5` must lie in `(0, 0.5)`, i.e. `H` strictly in `(0.5, 1.0)`:
`_powerlaw_kernel_matrix` raises `ValueError` outside that range, and
`test_powerlaw_bed_refuses_outside_the_stationarity_box` checks it at
`H in {0.3, 0.5, 1.0, 1.2}` (all four must raise) alongside
`test_powerlaw_bed_builds_inside_the_stationarity_box` (`H=0.75` must not,
per V-8's rule applied to a refusal: a gate that rejects everything is as
vacuous as one that accepts everything, so the non-empty interior is checked
on the same code path). The weight-energy divergence figures the coordinator
cited (15.9 at alpha=0.5, 26.9 at alpha=0.6) were not independently
re-derived here -- accepted on the coordinator's report, since the box's
enforcement doesn't depend on the exact divergence rate, only on divergence
occurring at `alpha>=0.5`, which is the standard ARFIMA stationarity
condition (`|d|<0.5`) and was cross-checked against that literature fact,
not against the two specific numbers.

**Effect on the Hurst-recovery numbers.** Re-measured after switching to the
GL-weight kernel (same `H=0.75`, `n=8192`, seeds 100-104):
`ests=[0.7319, 0.7114, 0.8737, 0.8839, 0.8869]`, `mean=0.8176` -- bias +0.068,
essentially unchanged from the naive-kernel reading of +0.062 reported
below. The GL-weight fix corrects the SHORT-LAG coefficients (`psi_1=alpha`
instead of always `1`) and was the mathematically correct thing to do
regardless, but it did not meaningfully shrink the DFA crossover bias in
practice: a direct autocovariance check (bypassing DFA entirely) on the
naive kernel already read an ACF-implied `H~0.72-0.735` at `H_true=0.7`
using only lags 10-2000 of a 20000-length realization, meaning the finite-
lag effective correlation structure runs hot relative to the asymptotic `H`
for BOTH kernel constructions -- this looks like a genuine finite-sample DFA
characteristic of this causal, growing-window generator, not an artifact
the GL-weight fix was positioned to remove. The existing test tolerance
(`0.15`) was set from measurements before this fix and still holds with
room to spare after it; it was not loosened or tightened in response.

```
$ python -m pytest tests/beds/test_bed_k.py -v
collected 15 items
... (10 tests from before) + 5 new:
test_powerlaw_bed_refuses_outside_the_stationarity_box[0.3] PASSED
test_powerlaw_bed_refuses_outside_the_stationarity_box[0.5] PASSED
test_powerlaw_bed_refuses_outside_the_stationarity_box[1.0] PASSED
test_powerlaw_bed_refuses_outside_the_stationarity_box[1.2] PASSED
test_powerlaw_bed_builds_inside_the_stationarity_box PASSED
============================= 15 passed in 1.03s ==============================
```

The rest of this file (below) describes the state at the first GREEN, before
these three findings arrived; numbers in the "six must-fires" and "Hurst
estimator" sections below are from the naive-kernel version except where the
paragraph above states the re-measured value. The RED/GREEN transcripts
below are the original 10-test run and remain the valid TDD record; they
were not re-staged for the 5 added tests since those were written directly
against an already-passing `build_powerlaw` (their own red/green cycle was
run interactively above rather than transcribed a second time).

## What was built

- `ceq/beds/__init__.py` -- package marker, one line of `__all__`.
- `ceq/beds/bed_k.py` -- BED-K's two variants and the interventional channel:
  - `kernel_matrix`, `build`, `build_delay`, `build_powerlaw` -- the corpus.
    `K(i,j)=1` iff `j=i-d` (pure delay); `K(i,j)=(i-j)^(H-1.5)` for `j<i`,
    `H>0.5` (power-law / fBm-type). `z = K @ b`, `b` iid `N(0,1)` from
    `numpy.random.default_rng(seed)` -- one seed, one RNG, matching
    `ceq/corpus.py`'s discipline (that file uses `random.Random` because its
    payload is token ids; this module's payload is real-valued, so
    `default_rng` is the equivalent instrument). `build()` returns a
    manifest dict (`kind, n, seed, params, K, b, z`), not an object, matching
    `corpus.py`'s `build()` shape.
  - `rebuild`, `bump`, `finite_diff_jacobian_column` -- the interventional
    channel. `bump` reads `K[:, p]` directly (exact, since `z` is linear in
    `b`); `finite_diff_jacobian_column` central-differences `rebuild` (the
    generator's own forward pass) at `eps=1e-6`. The two never call each
    other, so their agreement is a check, not an identity.
  - `fit_first_order_recurrence`, `hard_delay_attention` -- the two
    instruments Lean #12 (`first_order_cannot_delay`) needs: a least-squares
    scan fit and a hand-set single attention head. `hard_delay_attention`
    reads an explicit `pos` channel (see the post-GREEN revision section
    above), not implicit array order.
  - `estimate_hurst_dfa` -- DFA, order 1, with a window range restricted to
    >=3% of the series length (justified below).
  - `gl_weights` -- Grunwald-Letnikov / fractional-integral weights via the
    stable ratio recurrence, used by `_powerlaw_kernel_matrix` and gated by
    an explicit stationarity-box check (see post-GREEN revision above).
- `tests/beds/test_bed_k.py` -- the six must-fires, 10 collected tests
  (two are parametrized over both bed kinds).

## RED, before `ceq/beds/` existed

```
$ python -m pytest tests/beds/test_bed_k.py -v
collecting ... collected 0 items / 1 error
=================================== ERRORS ====================================
__________________ ERROR collecting tests/beds/test_bed_k.py __________________
ImportError while importing test module '...\tests\beds\test_bed_k.py'.
Traceback:
tests\beds\test_bed_k.py:39: in <module>
    from ceq.beds import bed_k
E   ModuleNotFoundError: No module named 'ceq.beds'
=========================== short test summary info ===========================
ERROR tests/beds/test_bed_k.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 1.49s ===============================
```

## GREEN, after `ceq/beds/bed_k.py`

```
$ python -m pytest tests/beds/test_bed_k.py -v
collecting ... collected 10 items

tests/beds/test_bed_k.py::test_delay_bed_scan_blind_but_ar1_control_reaches_0_99 PASSED [ 10%]
tests/beds/test_bed_k.py::test_delay_bed_is_attention_reachable_to_1e_12 PASSED [ 20%]
tests/beds/test_bed_k.py::test_hurst_estimator_white_noise_must_fire PASSED [ 30%]
tests/beds/test_bed_k.py::test_hurst_estimator_ar05_is_not_the_rs_bias PASSED [ 40%]
tests/beds/test_bed_k.py::test_hurst_estimator_recovers_known_H_on_the_powerlaw_bed PASSED [ 50%]
tests/beds/test_bed_k.py::test_jacobian_oracle_matches_independent_finite_difference[delay-params0] PASSED [ 60%]
tests/beds/test_bed_k.py::test_jacobian_oracle_matches_independent_finite_difference[powerlaw-params1] PASSED [ 70%]
tests/beds/test_bed_k.py::test_bump_moves_z_exactly_where_kernel_nonzero_and_nowhere_else[delay-params0] PASSED [ 80%]
tests/beds/test_bed_k.py::test_bump_moves_z_exactly_where_kernel_nonzero_and_nowhere_else[powerlaw-params1] PASSED [ 90%]
tests/beds/test_bed_k.py::test_seeded_reproducibility_both_beds PASSED   [100%]

============================= 10 passed in 4.63s ==============================
```

## The six must-fires, with real numbers

1. **Delay bed is scan-blind.** `fit_first_order_recurrence` on the delay
   bed (`n=2048, d=5, seed=0`), using the label's own true previous value as
   the recurrence state (the most generous state a scan could ever get):
   `alpha=-0.0109, beta=-0.0216, R2=-0.000170`. Threshold `0.05`, justified
   two ways: population `R2` is exactly 0 (`b` iid, so `z_{i-1}=b_{i-1-d}`
   and `z_i=b_{i-d}` draw from different noise positions for any `d>=1`),
   and a direct 30-seed sweep at `d=4, n=2048` measured `R2` in
   `[-0.0027, 0.0028]`. The AR(1) control (`a=0.8`, `n=2048`, same fitting
   routine) recovered `alpha=0.8000, beta=1.0000, R2=1.000000` -- exact,
   because the control series is generated by exactly this recursion, which
   is the point: the fitting code is not the thing reading near-zero.
2. **Delay bed is attention-reachable.** One-hot positional query/key
   coding offset by `d=7`, temperature 45, `n=500`: max abs error against
   the true label over `i>=d` was `8.67e-19`, against a `<=1e-12`
   requirement.
3. **Power-law bed's Hurst.** Estimator: **DFA** (Peng et al. 1994),
   order-1 detrending, over the reasons in the next section.
4. **Jacobian oracle exact.** `bump` (analytic `K[:,p]`) vs
   `finite_diff_jacobian_column` (central difference of an independent
   `rebuild` call, `eps=1e-6`), 6 positions each, both bed kinds:
   delay max err `2.88e-11`, power-law max err `4.88e-10`. Both `<=1e-9`.
5. **Bumps move the label, both halves.** `n=200`, bump at `p=n//2`,
   `eps=1e-3`, both bed kinds: `diff != 0` at every position where
   `K[:,p] != 0`, and `diff == 0` (bitwise, not "small") at every position
   where `K[:,p] == 0`. Nonzero-mask size checked non-degenerate
   (`0 < count < n`) before either half is read.
6. **Seeded reproducibility.** Same seed: `b` and `z` arrays bitwise equal
   across two independent `build_delay`/`build_powerlaw` calls. Different
   seed: both arrays differ. Checked for both bed kinds.

All six pass, both halves, on the current implementation.

## The Hurst estimator: DFA, its white-noise reading, and its AR(0.5) reading

**Chosen: Detrended Fluctuation Analysis (DFA), order-1 detrending.**
`CEQ_V15_CONTRACT.md` PART III states the constraint by name: "the Hurst
covariate ships ONLY with its white-noise must-fire (the author's R/S read
0.75 on AR(0.5): biased)." R/S's bias on short-memory AR processes is a
known failure mode (short-range correlation inflates the rescaled range);
DFA is the standard calibrated alternative and was picked over a
wavelet/Whittle estimator only because it needed no extra dependency beyond
`numpy` (`np.polyfit`/`np.polyval` do the per-window detrending) --
ponytail's rung 5 (already-installed dependency) covers it without adding
`statsmodels` or a custom wavelet transform for one estimator.

**White noise (true H=0.5), n=8192, 5 seeds (0-4):**

```
ests = [0.4180, 0.6060, 0.4615, 0.3423, 0.5596]
mean = 0.4775
```

**AR(0.5) (true H=0.5, short memory only), n=8192, 5 seeds (0-4):**

```
ests = [0.4250, 0.6148, 0.4700, 0.3495, 0.5682]
mean = 0.4855
```

Both land within ~0.02-0.03 of the true 0.5, and the AR(0.5) reading in
particular sits nowhere near the author's documented R/S reading of 0.75 --
this DFA implementation does not carry that bias, on this exact case.

**Known-H recovery, power-law bed at H=0.75, n=8192, 5 seeds (100-104):**

```
ests = [0.7267, 0.7068, 0.8683, 0.8774, 0.8813]
mean = 0.8121   (bias +0.062)
```

A second, independent seed block (0-4, used during calibration rather than
in the committed test) gave mean `0.7867` (bias +0.037) -- both comfortably
inside the test's `0.15` tolerance, and the bias is consistently positive
and shrinks as `H_true` grows toward 1 (measured across H in
`{0.6, 0.7, 0.75, 0.8, 0.9}` during calibration: biases of roughly
`+0.10, +0.056, +0.037, +0.02, -0.002`).

**Where the bias comes from, and why the estimator isn't tuned to hide it.**
The bed's kernel has weight exactly `1` at lag 1 for *every* `H`
(`(i-j)^(H-1.5)` at `i-j=1` is `1` regardless of `H`), which is a genuine
short-range crossover baked into this specific causal, truncated
construction -- confirmed independently by measuring the process's own
autocovariance decay (not through DFA at all): at `H_true=0.7`, fitting
`log(acf) ~ slope * log(lag)` over lags 10-2000 gave `slope=-0.529`, implying
`H=1+slope/2=0.735`, matching DFA's read on the same process closely and
confirming the bias is a property of the *generator*, not a DFA artifact
that a different window range would make disappear. A naive DFA fit from
`min_win=8` reads `H_hat` in `[0.86, 1.03]` for true `H` in `[0.6, 0.9]`
(compressed toward 1, useless for discriminating `H`); restricting the fit
to windows `>=3%` of `n` -- past the crossover -- is what gets the bias down
to the `+0.02..+0.10` range reported above. `estimate_hurst_dfa`'s
`min_win` default encodes this restriction and its docstring states the
measurement it is calibrated against.

## Is the delay bed genuinely attention-reachable and scan-blind?

**Yes, both halves, and both were measured, not assumed.** Scan-blind: the
best-case first-order recurrence (using the true previous label, which no
real scan actually gets) reaches `R2=-0.00017`, indistinguishable from the
population value of exactly 0, while the identical fitting code recovers a
true AR(1) process to `R2=1.000000` -- so the near-zero reading on the delay
bed is not the fitting routine being broken. Attention-reachable: a single
hand-set head reproduces the label to `8.67e-19`, seven orders of magnitude
inside the `1e-12` bar. Neither half was assumed from the algebra alone;
both are asserted against numbers a bug would move.

## Tests I was tempted to write, and did not, because they would have been vacuous

- **A dedicated "K is lower-triangular" / "K has the right shape" test.**
  Would have re-derived the same condition from the same `np.where` call
  that builds `K` -- MISTAKES.md V-3, an algebraic identity of its own
  construction. The structural property is exercised indirectly by every
  other test that reads `K` (the Jacobian and bump-movement tests would fail
  immediately if causality were wrong), so a standalone check would add
  nothing a bug could dodge that those tests don't already catch.
- **Implementing R/S myself and asserting it reads ~0.75 on AR(0.5).**
  Tempting, to "reproduce" the contract's cited number -- but a self-written
  R/S calibrated to match a number already stated in the contract tests
  nothing about *this* corpus; the contract's R/S claim is taken as given,
  and the only thing worth measuring is whether the estimator actually
  shipped (DFA) avoids that bias, which is what the AR(0.5) must-fire does.
- **A bump-movement test that only checks `np.any(diff != 0)`.** Would pass
  even if the bump moved the wrong position, or moved every position
  regardless of the kernel -- the exact shape of MISTAKES.md V-9. The
  committed test checks the movement is nonzero at *every* kernel-nonzero
  position and *exactly* zero at every kernel-zero position, individually.
- **A Hurst-recovery tolerance tight enough to look impressive (e.g.
  +/-0.02).** The measured single-block bias is `+0.037` to `+0.10`
  depending on `H_true` and seed block; a tolerance that tight would be
  flaky on its own calibration runs and would fail the honest-threshold
  check (V-10: compute the expected value before setting the bar). `0.15`
  is set from the measured spread, not from what would look best.
- **A specific numeric target for `beta_hat` on the delay-bed recurrence
  fit.** `alpha` and `beta` have no meaningful population target beyond
  "small" when the regressors are independent of the label; `R2` already
  carries the full "no linear relationship" content, and pinning `beta_hat`
  to a specific value would just be re-asserting sampling noise as if it
  were a prediction.
- **Tightening the attention-reachability bound to float64 machine
  epsilon (~1e-16) instead of the contract's stated `1e-12`.** The measured
  error (`8.67e-19`) would pass either way, but over-tightening past what
  the contract actually asks for makes the test brittle to legitimate
  implementation choices (a different temperature, a different softmax
  stabilization) that would still satisfy the real requirement.

## Distance to the north star

Unchanged by this node: BED-K registers a corpus whose label contains a
delayed cause and demonstrates the two-sided attention/scan split the
contract's WHERE WE ARE section names as the campaign's blind spot, plus an
exact (not fit) interventional oracle for it. It does not itself move ARM PL,
softmax, or any floor-crossing number -- those depend on the Lean train-gate
(n3) and R1/R2 (it.9-10), still ahead on the DAG. BED-K's own deciding
measurement (R3: `alpha-hat` recovered, `H-hat` within CI of this DFA
estimator) is downstream of this node, not answered by it.
