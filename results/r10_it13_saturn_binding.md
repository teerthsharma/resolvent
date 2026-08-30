# R10 it.13 — SATURN binds MARS attack #1

**Measured, not argued.** MARS filed every number in
`results/r10_it13_mars_attack1.md` before this measurement existed. Nothing here
adjusted his threshold, his CI or his falsifier. The binding is
`scale/it13_mars_binding.py` — section 3 of his file with an error count, a JSONL
record and a falsifiability arm added; the measurement lines are his.
Per-block rows: `results/r10_it13_saturn_binding.jsonl`.

Run: K=200 draws per block, seeds 20000–20199 (disjoint from grid seeds 0–7 and
12345), shipped eval shape n=4096, s=64, d=24, d_model=16. 600 draws total,
**600 completed, 0 errored**. No training, one process, 73.0 s.

---

## 1. VERDICT: WITHDRAWN

MARS's rule is three clauses, all three required in all three blocks. Applied
mechanically:

| `t*` | mean `Var(y)` | `delta_1` | 95% CI | 1. sign<0 | 2. CI excludes 0 | 3. dev from -1 < 0.05 | block |
|---|---|---|---|---|---|---|---|
| 2  | 2.002719  | **-0.997281** | `[-1.003067, -0.991496]` | yes | yes | 0.002719 yes | FIRES |
| 8  | 7.984942  | **-1.015058** | `[-1.041212, -0.988903]` | yes | yes | 0.015058 yes | FIRES |
| 32 | 32.069818 | **-0.930182** | `[-1.024814, -0.835550]` | yes | yes | **0.069818 NO** | **MISS** |

Section 4 of the attack: *"MARS also withdraws if `|delta_1 + 1.000| >= 0.05` in
any block."* At `t*=32` it is 0.069818. **Attack #1 is WITHDRAWN by its author's
own filed falsifier.** Not "fires at two of three"; there is no partial credit in
the filed rule and none is taken here.

**What the data nevertheless says, reported without being used to rescue the
attack.** The `t*=32` CI `[-1.0248, -0.8356]` covers -1.000, and the measured
mean `Var(y)` = 32.0698 has a CI covering 32.0 = `t*` and excluding 33.0 = `t*+1`
by 9.8 CI half-widths. The shortfall at `t*=32` is one unit of variance to within
the estimator's own noise. The clause that failed is a point-estimate clause
filed tighter than the estimator MARS himself priced — see §5. That observation
changes the verdict by exactly nothing: WITHDRAWN is the mechanical reading and
it is the reading.

---

## 2. FALSIFIABILITY — the harness can report `delta_1` ~ 0

A measured -1.000 is worthless if the harness subtracts 1 somewhere. The arm
restores the pre-fix builder on the *same draw*: `b[:, s-1]`, which
`make_equilibrium_batch` zeroes (`scale/negation_scope.py:404`), is refilled with
a fresh independent `N(0,1)` and the label is recomputed through the shipped
`equilibrium_oracle`. That corpus has `Var(y) = t*+1` by construction. Same
`y.var(unbiased=False)`, same `mean - (t*+1)`, same `stat()`.

| `t*` | pre-fix mean `Var(y)` | pre-fix `delta_1` | 95% CI | reads ~0? |
|---|---|---|---|---|
| 2  | 3.006387  | **+0.006387** | `[-0.002982, +0.015756]` | yes, covers 0 |
| 8  | 8.996501  | **-0.003499** | `[-0.032784, +0.025787]` | yes, covers 0 |
| 32 | 33.093519 | **+0.093519** | `[-0.004759, +0.191796]` | yes, covers 0 |

Paired against the shipped corpus, draw for draw, the arm difference is
**+1.003669 / +1.011559 / +1.023701** at `t* = 2 / 8 / 32`: restoring that one
zeroed driver adds exactly one unit of label variance and moves the harness's
reading from -1 to 0. **The instrument can fail. It reports 0 where 0 is true and
-1 where -1 is true, on the same code path.**

---

## 3. CONTROL A — did not fire; F-2/F-3 survive

`delta_2 = mean nrmse(equilibrium_hop_reading(x,1), y) - sqrt((t*-1)/t*)`, filed
at 0.000 with `|delta_2| < 0.002`.

| `t*` | 1-hop NRMSE | closed form | `delta_2` | 95% CI | within +/-0.002 |
|---|---|---|---|---|---|
| 2  | 0.707183 | 0.707107 | +0.000076 | `[-0.001028, +0.001181]` | yes |
| 8  | 0.935095 | 0.935414 | -0.000320 | `[-0.001035, +0.000395]` | yes |
| 32 | 0.984139 | 0.984251 | -0.000112 | `[-0.000493, +0.000268]` | yes |

Every CI covers 0 and lies inside +/-0.0012. The 1-hop ceiling the grid is scored
against is the closed form, measured. **Control A did not fire. F-2 and F-3 of
`R10_ITERATION_08_09.md` are not the casualty of this iteration**, and the branch
in which they would have been is closed on the data, not on argument.

## 4. CONTROL B — concession holds, eval-draw axis stays closed

Pinned training-free predictor `c * equilibrium_hop_reading(x,1)`, `c = 0.248958`,
`t*=8`, 200 independent eval draws:

| quantity | filed | measured |
|---|---|---|
| mean pinned NRMSE | 0.9723723935 (by construction) | 0.972365 |
| across-draw sd | 0.00129 | **0.001306** |
| 95% draw half-width | 0.0025 | 0.002561 |
| crossing margin | 0.027628 | 10.8x the half-width |

Withdrawal trigger was sd above 0.014; measured 0.001306, a factor 10.7 below it.
**The `t*=8` crossing is not an eval-draw artifact.** The axis stays conceded.

---

## 5. WHERE THE PRE-REGISTRATION BROKE, since the round scores mechanisms

The measurement performed exactly as MARS priced it. His filed CI half-widths
were `t* * 0.0030625` = 0.006 / 0.025 / 0.098; measured 0.005785 / 0.026155 /
0.094632 — within 3.5% at every block. The estimator is not the problem.

The problem is that at `t*=32` he filed a **point-estimate clause (0.05) tighter
than the CI he filed in the same table (+/-0.098)**. Under his own mechanism being
exactly true, clause 3 at `t*=32` had probability

`P(|delta_1 + 1| < 0.05 | true delta_1 = -1, se = 0.0483) = 0.6996`

of passing — so the attack carried a **30% chance of self-withdrawing while being
entirely correct**, and it took it. Clauses 1 and 2 scale with `t*` because the
sampling spread of a variance does; clause 3 was filed as an absolute constant.
A clause 3 stated relatively (e.g. `|delta_1 + 1| < 0.05 * max(1, t*/8)`) would
have been the same mechanism claim, correctly powered. **That is a defect in the
filing, not in the code under attack, and it is recorded here as MARS's cost.**

---

## 6. THE TRAP — REAL, AND EXECUTABLE

Verified two ways: the arithmetic, and by running `bar_verdict` on the shipped
`t*=2` journal row.

| `t*` | journal `flipper_dependence` | `2/sqrt(t*)` | dev | `2/sqrt(t*+1)` | dev | verdict under `t*+1` |
|---|---|---|---|---|---|---|
| 2  | 1.4012436552 | 1.4142135624 | 0.012970 | 1.1547005384 | **0.246543** | **FAILS** (4.93x tol) |
| 8  | 0.7130855231 | 0.7071067812 | 0.005979 | 0.6666666667 | 0.046419 | passes, at 92.8% of tol |
| 32 | 0.3687045648 | 0.3535533906 | 0.015151 | 0.3481553119 | 0.020549 | passes |

MARS's two numbers check exactly: 0.0129699 and 0.2465431. The tolerance is
`flipper_tol = 0.05`, `bar_verdict`'s default, and
`scale/r10_capacity_sweep.py:153` calls `bar_verdict` **without overriding it**.

Not arithmetic — executed, on the shipped `t:bar` row of
`results/r10_it8_capacity_softmax_t2.jsonl`:

```
as shipped, fd=2/sqrt(2): (True,  'BAR CALIBRATED')
after repair, fd=2/sqrt(3): (False, 'flipper_dependence=1.401244 is not the value this
                             task predicts in closed form, 1.154701 (tol 0.05)
                             -- this is NOT the task')
```

`r10_capacity_sweep.main` prints `ABORT: calibration bar failed; crediting
nothing.` and returns 1 at `:157`, before the eval corpus is drawn at `:159`.

**The trap is real.** A consistency pass that repaired `chain_flipper_dependence`
to match the `:367` docstring — the pass this repo runs, and the docstring at
`:419` is the only thing standing in its way — kills the `t*=2` block outright,
and the test guarding it
(`tests/cameron/test_m3_etasks.py::test_the_chain_flipper_dependence_is_its_closed_form`)
would be "fixed" in the same pass and still pass.

**One correction to MARS's framing of the cost.** `R10_ITERATION_08_09.md:149-153`
reports N=8 seed CIs at (150, 2048) for **all three** blocks, so all three are
it.11-compliant *readings*; `t*=8` `[1.1162, 1.1314]` and `t*=32` `[1.1307,
1.1837]` sit above the bar. What is unique to `t*=2` is that it is the round's
**only it.11-compliant reading that excludes 1.0** — the only compliant LEARNABLE
verdict anywhere in the grid. That is what the repair would destroy. The cost is
as MARS priced it; the word "reading" was doing more work than the record allows.

**And what the measurement adds to the trap.** §1 and §2 settle which law the
corpus obeys: `Var(y) = t*`, with `t*+1` excluded by 9.8 half-widths at the
loosest block. So the repair the trap punishes is the repair that would have
moved the *executable* line to the *wrong* law. The six `t*+1` prose sites are
what should move, not `chain_flipper_dependence`.

---

## 7. WHAT THIS ITERATION SETTLED

- `Var(y) = t*`, measured, 600 draws, at the shipped eval shape. `t*+1` is
  excluded in all three blocks. The confound C is resolved in favour of the
  executable lines.
- Attack #1 is **WITHDRAWN** on its author's filed falsifier, at `t*=32`,
  clause 3.
- Control A did not fire: the 1-hop ceiling is its closed form to +/-0.0012.
  F-2/F-3 stand.
- Control B holds: the `t*=8` crossing is not an eval-draw artifact.
- The trap is real and was executed, not argued.
- Nothing was trained, no tracked source was edited, no cell of the grid moves.
