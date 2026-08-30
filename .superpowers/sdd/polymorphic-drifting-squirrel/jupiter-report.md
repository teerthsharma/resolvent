# JUPITER / MYCROFT — iteration 1 of 15

Branch `feat/r9-causal-consequence`, worktree `agent-ade92ee4d5e164d8f`.
Worktree started at `ac47049`, fast-forwarded to `74e5590` before work.
One commit: `22036de`.

Role: derive only. No training run. No adjudication of any arm.

---

## M3 — second exact oracle, Kirchhoff. DONE.

`scale/kirchhoff.py`, 347 lines. `tests/jupiter/test_kirchhoff_agreement.py`,
11 tests, all pass.

**The derivation.** `L = D − A`. `det(L_g) = tau(G)` — spanning trees.
`det(L_xy) = F(x|y)` — 2-forests splitting `x` from `y`. Ground at `x`, Cramer:
`R(x,y) = F(x|y)/tau(G)`. Resistance is a forest ratio. Then unit current `a→b`,
polarisation, and **ground at `b`**, which collapses the three-resistance form to

    omega_x = M_xa / M_aa = F(x,a|b) / F(a|b)

One solve `L_b z = e_a`. No inverse. No cancellation of big resistances.

**Why it is independent.** Chain path inverts `I − Q`, `Q = D^-1 A` on transient
nodes only. Kirchhoff path solves symmetric `D − A` on whole component, `a` an
ordinary interior column, only `b` removed. Different matrix, different
normalisation, different RHS, different index map. `L_T` and `I − Q` are
diagonally related (`I − Q = D^-1 L_T`), so the Kirchhoff route does not form
`L_T`. That is the whole point.

**Instrument law.** `e4_harmonic.measure()` now runs both oracles and asserts
agreement under `1e-10` **before** any number leaves the instance. Gap goes out
in the returned dict as `kirchhoff_gap`. Law refuses `kill > 0` — the forest
identity is the undamped identity, and comparing two different objects and
calling the difference a defect is worse than not checking.

**Measured gaps, RUN, float64, this machine:**

| instance | merged | transient | gap |
|---|---|---|---|
| 6 drawn Erdos-Renyi, n=9 | 9 | 7 | `2.220446e-16` |
| SMALL_CASE | 62 | 60 | `8.992806e-15` |
| SHIPPED_CASE | 1202 | 1200 | `9.636736e-14` |

**Planted defect fires.** `scratch_chain_with_off_by_one` divides by
`len(neighbours) + 1` — the bug from counting a node among its own neighbours.
Survivable by construction: rows stay strictly sub-stochastic, solve succeeds,
output stays in `[0,1]`, nothing downstream would notice. Moves omega by
`1.749951e-01` to `3.156170e-01`. Check rejects **6 of 6** drawn instances, and
also through `assert_oracles_agree`, the function the law actually calls.

Tolerance `1e-10` sits **3 orders above the worst clean gap, 9 below the
smallest defect**. Not chosen after seeing a result.

**Both ends checked against brute force.** Cofactor determinant vs enumeration
of every spanning tree on drawn 7-node graphs. Double minor vs enumeration of
2-forests. Forest ratio vs grounded solve, node by node. Label non-degenerate:
`omega.std() = 0.499989` on SHIPPED, mean `0.503333`, min 0, max 1.

---

## M4 — the verdict as a statistic. DONE. Verdict is RISES.

`scale/page_trend.py`, 520 lines. `tests/jupiter/test_page_trend.py`, 9 tests,
all pass. Artifact `results/e_ladder_trend.txt`.

**What it replaces.** `scale/e_ladder.py:217` uses
`mono = all(b >= a for a, b in zip(deltas, deltas[1:]))`. No null. No error
rate. Four noisy numbers land ordered by chance 1 time in 24.

**The exact null.** Rank within block, `L = sum_j j R_j`. Under exchangeability
the `k!` rank assignments are equally likely per block and blocks independent,
so the null of `L` is the `n`-fold convolution of the `k!`-point distribution.
`k=4`, `n=5`: `24^5 = 7962624` assignments enumerated exactly. Support
`[100,150]`, mean `125`, symmetric.

**Pre-registered branch, frozen before the statistic ran, exhaustive over all
four clause states:** `RISES` = both clauses. `FLAT` = neither. `SPLIT` =
exactly one. `SPLIT` exists because "both/neither" leaves a hole and row H
already cost this repo one.

**THE READING, RUN:**

    rank sums R_j    8.0  12.0  13.0  17.0
    L                139
    exact p          0.016724
    permutation p    0.016255   se 0.000283, 200000 draws, seed 0
    isotonic fit     -0.036025  -0.017215  -0.004284  +0.016035
    top rung         +0.016035   CI [+0.002826, +0.033167]

**VERDICT: RISES.** Not softened. The two p-value paths differ by `4.7e-4`,
which is `1.7` standard errors.

**Three things that verdict rests on. All measured. All printed beside it.**

1. **The size clause is the constraint, not the data.** Same bootstrap without
   the monotone constraint: `[-0.004711, +0.033167]` — **covers zero**. Its
   lower bound reproduces `results/e_ladder_reading.txt`'s shipped `ci_lo` of
   `-0.004711` **exactly**, so it is the same bootstrap. PAVA pooled the top in
   `13.07%` of resamples and lifted the bound by `+0.007537`. Pooling only ever
   raises a low top. That lift is estimator, not data.

2. **Trend clause survives 2 of 5 single-seed deletions.** drop seed 1 →
   `p = 0.003864`; seed 2 → `0.021741`; seed 0 or 4 → `0.050411`; seed 3 →
   `0.072401`.

3. **ROW G OUTRANKS THE WHOLE STATISTIC.** 3 of the 4 rungs carry a cell at or
   above predict-the-mean and are credited nothing in either direction:

   | rung | settled | twin | credited |
   |---|---|---|---|
   | e3_t1 | 0.994399 | 0.958373 | yes |
   | e3_t2 | 1.013958 | 0.996743 | no |
   | e3_t8 | 1.096009 | 1.091725 | no |
   | e3_t32 | 1.103711 | 1.119745 | no |

   A trend in the difference between two arms that both lose to the mean is an
   **ordering, not a capability**. `RISES` is a statement about the contrast as
   a number. It is **not** a claim that settling bought anything at depth.

**But the conjunction is calibrated even though one clause is not.** 200 drawn
tables per arm, `5x4`, `sd 0.03`, seed 11, `n_boot 300`:

    flat truth      trend 0.065   size 0.325   RISES 0.040
    rising truth    trend 0.810   size 1.000   RISES 0.810

Size clause alone fires on a third of flat tables — 6x nominal. Conjunction
reads `0.040` against nominal `0.05` because the trend clause gates it. That is
the measured reason the branch needs both. Both arms from one generator, verdict
separates them, so the branch is not vacuous.

---

## M2 — RIP sample-complexity line. DONE, with a correction.

`scale/rip_line.py`, 317 lines. Moon-built, audited and corrected by me.
`scale/impact.py` **read only, zero diff** — Mars owns it.

**Correction to the dispatch's framing.** The planted `B` has fixed sparsity
**per ROW, not per column**. `SUPPLIERS_PER_NODE + COMPETITORS_PER_NODE = 4`
nonzeros per row (`scale/impact.py:88-89, 405-410`, READ). A column's nonzero
count is however many rows chose it — unbounded, instance-dependent.
`impact_attribution` recovers `B[query, :]`, a row (`scale/impact.py:1021`,
READ). The line is stated for the row.

**The line.** `m >= C * s * ln(n/s)`. `verdict()` returns the exact strings
`UNDER-SAMPLED` and `ADMISSIBLE`, so a cell prints a regime not a loss.

**The must-fire fires.** Every grid `0.00` at the bottom, `>= 0.96` at the top
for basis pursuit. Both ends non-degenerate. Two solvers, failing differently.

| n | s | n/s | ln(n/s) | m_50 BP | C BP | m_50 OMP | C OMP |
|---|---|---|---|---|---|---|---|
| 64 | 4 | 16 | 2.772589 | 16 | 1.44270 | 24 | 2.16404 |
| 128 | 8 | 16 | 2.772589 | 32 | 1.44270 | 48 | 2.16404 |
| 256 | 4 | 64 | 4.158883 | 24 | 1.44270 | 28 | 1.68314 |
| 64 | 16 | 4 | 1.386294 | 40 | **1.80337** | 64 | 2.88539 |

**MOON DEFECT I CAUGHT AND FIXED.** Moon shipped 2 beds, both at `n/s = 16`,
and reported `C` "exactly stable — not a coincidence". It is a coincidence of
the design: at fixed `n/s`, `ln(n/s)` is one number, so `m_50` tracking `s`
**forces** `C` to repeat whatever the truth is. That is a vacuity-rule-2 hit —
an algebraic identity of the design read as a measurement. Two beds varying the
ratio were added. `C = 1.44270` holds at `n/s = 16` and `n/s = 64`, rises to
`1.80337` at `n/s = 4` — **25% higher**. `C` is a per-bed measurement. Reusing
`1.44270` elsewhere is a GUESS. Docstring, `SETTINGS` and a test now say so.

Shipped corpus: `n_samples = 256`, `N = 1024`, `s = 4` (`impact.py:973`, READ).
`rip_line(1024, 4, 1.44270) = 32.01`. Oversamples by ~8x. Floor is the
deliverable, not a claim about today's default.

---

## M1 + M5 — coherence floor. DONE. The author's figure is 15.8% low.

`scale/coherence_floor.py`, 229 lines.
`tests/jupiter/test_coherence_and_rip.py`, 7 tests, all pass.

**Welch.** `max coherence >= sqrt(max(0, (k-d)/(d(k-1))))`. **Exactly 0 for
k <= d**, because an orthonormal set of `k <= d` vectors exists and attains it.
Tight above the dimension: `d=2, k=3` bound reads `0.5`, and three unit vectors
at 120 degrees attain `|cos 120| = 0.5` to `1e-12`.

**M1, d=256, k=16: floor is exactly 0.** Random roles carry max coherence
`0.174795`. A crosstalk-shaped failure at 16 roles in 256 dimensions is a
**training or design defect, not dimension starvation**.

**M5. Four routes to the max coherence:**

    sqrt(2 ln k / d)            0.147176   author's form, union over k
    sqrt(2 ln(k(k-1)/2) / d)    0.193397   union over C(k,2)=120 pairs, a BOUND
    Monte Carlo 20000 trials    0.174795   95% CI [0.174460, 0.175131]
    order-statistic quadrature  0.174499   integral_0^1 (1 - F(x)^120) dx

Quadrature is inside the Monte Carlo CI. **Both closed forms are outside it.**
Union bound is above, as a bound must be. **The author's form is below, because
it counts `k` events when the max runs over `C(k,2) = 120` pairs.**

Mean overlap confirmed on both paths: exact
`Gamma(d/2)/(sqrt(pi) Gamma((d+1)/2)) = 0.049917` vs Monte Carlo `0.049917`, CI
`[0.049869, 0.049964]`; limit `sqrt(2/(pi d)) = 0.049868`, agrees to `4.9e-5`.

**M5 consequence, and the error is in the safe direction.** Scrambled roles at
`d=256, k=16` carry expected mean overlap `0.049917` and expected worst-case
overlap `0.174795`, both by chance, both computable before the control runs. A
scramble control calibrated to zero is vacuous, and it is **more** vacuous than
the amendment's own figure implies, not less. Fifteenth pattern, killed
pre-birth.

---

## CLAIM LEDGER

Every claim, class, and check. No claim without a check.

| # | Claim | Class | Check |
|---|---|---|---|
| J1 | `omega_x = M_xa/M_aa = F(x,a\|b)/F(a\|b)` | DERIVED+RUN | polarisation derivation shown in `MATHEMATICS.md` §11; `test_the_harmonic_measure_is_the_spanning_forest_ratio` vs brute-force 2-forest enumeration, 3 drawn 7-node graphs, `< 1e-9` |
| J2 | `det(L_g) = tau(G)` | RUN | `test_the_laplacian_cofactor_is_the_spanning_tree_count`, 4 drawn graphs vs full edge-subset enumeration |
| J3 | `det(L_xy) = F(x\|y)` | RUN | `test_the_two_forest_count_is_the_double_minor_determinant`, 3 drawn graphs vs enumeration |
| J4 | `R(x,y) = F(x\|y)/tau(G)` | RUN | `test_effective_resistance_agrees_with_the_determinant_ratio`; grounded-inverse form vs determinant ratio, `< 1e-8` relative |
| J5 | Oracles agree `< 1e-10` on every Dirichlet instance | RUN | gaps `2.220446e-16` / `8.992806e-15` / `9.636736e-14` at n=9 / 62 / 1202; `test_..._drawn_rips_dirichlet_instance` parametrised over SMALL and SHIPPED |
| J6 | Agreement check fires on a planted defect | RUN | 6/6 drawn instances; defect magnitude `1.749951e-01` to `3.156170e-01`; also via `assert_oracles_agree` raising |
| J7 | Law is wired into the instance builder | READ+RUN | `scale/e4_harmonic.py:measure` calls `kirchhoff.assert_oracles_agree`; `measure(1024,2.0,0x33960005)["kirchhoff_gap"] = 8.992806e-15` |
| J8 | The `e4_harmonic` edit broke nothing | RUN | `tests/cameron/test_e4_harmonic.py` + `test_harmonic_attribution.py`: 11 failed / 10 passed, **FAILED set byte-identical** with and without the edit, by stash-diff |
| J9 | Page null at k=4,n=5 is exact on `[100,150]`, mean 125, symmetric | RUN | `test_the_exact_null_is_a_distribution_on_the_right_support`; sums to 1 within `1e-12`, `dist == dist[::-1]` |
| J10 | `L = 139`, exact `p = 0.016724` | RUN | two paths: convolution `0.016724`, permutation `0.016255` (se `0.000283`, 200000 draws) — `1.7` se apart |
| J11 | isotonic top `+0.016035`, CI `[+0.002826, +0.033167]` | RUN | `test_the_real_ladder_reading_is_reproduced_exactly` pins all four to `1e-6` |
| J12 | Unconstrained CI covers zero: `[-0.004711, +0.033167]` | RUN | lower bound reproduces shipped `results/e_ladder_reading.txt` `ci_lo` `-0.004711` **exactly** — independent confirmation the bootstrap matches `m3_synthetic_settled.contrast` |
| J13 | PAVA lift `+0.007537`, pooled in `13.07%` of resamples | RUN | `isotonic_top_ci` returns both; printed in `report()` |
| J14 | Trend survives 2/5 seed deletions | RUN | `leave_one_block_out`, p `0.003864 / 0.021741 / 0.050411 / 0.072401 / 0.050411` |
| J15 | Conjunction FPR `0.040`, power `0.810`; size clause alone FPR `0.325` | RUN | 200 drawn tables per arm, seed 11; asserted at 60 tables in `test_the_size_clause_alone_is_miscalibrated_and_the_conjunction_is_not` |
| J16 | Row G credits nothing at 3 of 4 rungs | READ+RUN | `e_ladder.read()["ladder"][i]["credited"]`; cell means `1.013958 / 1.096009 / 1.103711` all above `1.0` |
| J17 | Planted `B` sparsity is per-row, 4 nonzeros | READ | `scale/impact.py:88-89, 405-410`; recovery target `B[query,:]` at `:1021` |
| J18 | Oracle transitions across the RIP line | RUN | 4 beds, both solvers, `0.00` at every grid bottom, `>= 0.96` at every BP top |
| J19 | `C` is NOT bed-invariant | RUN | `1.44270` at `n/s = 16, 16, 64`; `1.80337` at `n/s = 4`. Moon's "exactly stable" was forced by fixed `n/s` |
| J20 | Welch bound is exactly 0 for `k <= d` | DERIVED+RUN | closed form; attained by the 120-degree frame at `d=2,k=3` to `1e-12`; asserted for `k in (2,8,16,255,256)` at `d=256` |
| J21 | `E[max coherence] = 0.174795` at `d=256,k=16` | RUN | two paths: Monte Carlo 20000 trials CI `[0.174460, 0.175131]`, and order-statistic quadrature `0.174499` — inside the CI |
| J22 | Author's `0.147176` understates by 15.8% | DERIVED+RUN | union counts `k` events; max runs over `C(k,2)=120`. Asserted: `expected_max > jl * 1.15` |
| J23 | `E[mean coherence] = 0.049917` | RUN | exact Gamma ratio, quadrature of the same density (`< 1e-9`), and Monte Carlo CI `[0.049869, 0.049964]` |
| J24 | No `nn.Parameter` added | READ | `git show HEAD` touches no arm, no `forward`, no module; no `nn.Parameter` in any of the five new files |

---

## ADVERSARIAL PASS — what was attacked, and what broke

- **Would the Kirchhoff check pass with its logic deleted?** No. Deleting the
  planted-defect test leaves an agreement test that has never seen a
  disagreement, which is why the plant is in the suite and asserts `6 of 6`.
- **Do the two oracles share an assumption?** They share the graph and the
  degree convention. A wrong `adjacency` moves both. `laplacian()` raises if an
  edge leaves the node set, which is the one place that assumption is checkable.
  **Not covered:** a wrong `case_graph` would fool both. Named in §15.
- **Does the isotonic CI mean what it says?** **NO — and this is the finding.**
  It excludes zero only because PAVA's pooling is one-sided. Caught by running
  the same bootstrap unconstrained and getting `[-0.004711, ...]`, which matches
  the shipped `ci_lo` exactly. Now printed permanently.
- **Is `RISES` robust to one seed?** **NO.** 2 of 5. Printed permanently.
- **Is the conjunction itself miscalibrated?** Checked, and it is not:
  `0.040` under a flat truth. The bad clause is gated by the good one.
- **Is the moon's `C` stability real?** **NO.** Forced by fixed `n/s`. Fixed.
- **Is the author's `0.147` right?** **NO.** Wrong by 15.8%, and the wrongness
  strengthens M5 rather than weakening it.
- **Was a pre-existing test broken?** No. 11 failing in
  `tests/cameron/test_harmonic_attribution.py` and `test_e4_harmonic.py`,
  FAILED set byte-identical before and after the `e4_harmonic` edit. Not touched.

---

## WHAT COULD NOT BE VALIDATED

The instrument law only guards `e4_harmonic.measure()`; `absorbing_chain` and
`fixed_point` stay callable directly and unguarded, so a caller bypassing
`measure` gets no cross-check, and both oracles consume the same `case_graph`,
so a defect in the graph builder itself would fool both identically. The
agreement margin is three sizes on one machine in float64 with no conditioning
bound proved, and nothing establishes it holds above 1202 nodes or in float32 on
the CUDA lane. The `RISES` verdict is real under a branch fixed before the
statistic ran, but its size clause fires on a one-sided estimator bias that can
be quantified and not removed, its trend clause survives only 2 of 5 seed
deletions, the percentile bootstrap behind both has 3125 distinct atoms at five
seeds, and the calibration sweep uses Gaussian noise at one scale rather than
the ladder's real noise; on top of all that, row G already credits three of the
four rungs nothing, so the statistic sharpens a reading the pre-registration had
declined to make. `C` in the RIP line is fit on coarse grids — `m_50` at
`n/s = 4` is pinned only to `(32, 40]` — under a Gaussian ensemble that
`news_mat` has not been shown to be; the line is necessary and never sufficient,
and nothing here wires `UNDER-SAMPLED` into an IMPACT cell, which is Mars's file
and out of scope. The coherence arithmetic is about independent uniform unit
vectors and says nothing about vectors any trained arm holds; the
order-statistic quadrature treats the 120 pairwise products as independent when
they are not, so it corroborates the sampled figure rather than proving it. The
full suite was not run, nothing was trained, and it was not verified that the
`tests/cameron` failures are the same 146 the record names — only that the edit
did not change them.
