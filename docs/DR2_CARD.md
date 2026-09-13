# DR-2 CARD

Four claims, each scored against the baseline that owns its class, on beds built so the
baseline has somewhere to win. Every number below was read from the stdout of a run
executed for this card. No number is transcribed from a report, a prior round, or another
machine.

## Provenance

| item | value | how it was obtained |
| --- | --- | --- |
| commit | `c9a9434` | `git rev-parse --short=7 HEAD` |
| machine id | `60b8cf943ee0` | `python -c "import platform,hashlib;print(hashlib.sha256(platform.node().encode()).hexdigest()[:12])"` |
| node | `WIN-16QAL06O9GB` | printed by R1 and R2 as part of their own provenance lines |
| platform | Windows 11; Python `3.11.9`, numpy `2.4.6`, scipy `1.17.1`, torch `2.14.0+cpu` | `python -c "import torch,numpy,scipy,sys; print(...)"`, exit 0 |

Both the commit and the machine id were recomputed here rather than copied, and both agree
with the values the runs print for themselves. R1 prints
`PROV commit c9a9434 machine 60b8cf943ee0 node WIN-16QAL06O9GB python 3.11.9 numpy 2.4.6`;
R2 prints `both verified, not copied`.

## The runs

| label | command | exit | wall clock | last line |
| --- | --- | --- | --- | --- |
| R1 | `python -m ceqjepa.pi_assign` | 0 | 2.11 s | `ALL SELF-CHECKS PASSED` |
| R2 | `python -m ceqjepa.t_length` | 0 | 129.98 s | `ALL SELF-CHECKS PASSED` |
| R3 | `python -m ceqjepa.intent_do` | 0 | 3.30 s | `ALL SELF-CHECKS PASSED` |
| R4 | `python -m pytest tests/curvature/test_pi_assign.py tests/curvature/test_t_length.py tests/curvature/test_intent_do.py -q` | **1** | 301.06 s | `1 failed, 77 passed in 296.60s (0:04:56)` |

Wall clock is the shell's own measurement around each command. The modules report a smaller
internal figure — R1 `elapsed 0.17s against the 300 s bar`, R2
`128.1 s elapsed against the 300 s bar`, R3 `elapsed 1.13 s against the 180 s bar` — because
the internal clock excludes interpreter and import time.

### R4 is red, and this is what it is

`R4` exits 1. One test of the 78 collected fails:

```
FAILED tests/curvature/test_t_length.py::test_the_module_imports_no_assignment_rule
E       AssertionError: importing t_length dragged in the assignment rule
E       assert 'ceqjepa.pi_assign' not in { ... } = sys.modules
tests\curvature\test_t_length.py:204: AssertionError
```

Collection counts, from `--collect-only -q` on each file: `test_pi_assign` 26,
`test_t_length` 34, `test_intent_do` 18, total `78 tests collected`, which matches
`1 failed, 77 passed`.

Two further runs locate the cause and are reported because a card that leaves a red run
unexplained is as useless as one that hides it:

- `python -m pytest tests/curvature/test_t_length.py -q -k "imports_no_assignment_rule"`
  → exit 0, `1 passed, 33 deselected in 1.96s`.
- `python -m pytest tests/curvature/test_pi_assign.py tests/curvature/test_t_length.py -q -k "digest or imports_no_assignment_rule"`
  → exit 1, `1 failed, 1 passed, 58 deselected in 2.45s`, same assertion.

The failing assertion is the fourth of four in that test. The three that precede it — the AST
scan of `ceqjepa/t_length.py` for any `import` naming `pi_assign` or `hbucket` — pass in both
orderings. Only the `sys.modules` line fails, and it fails only when `test_pi_assign.py` is
collected first in the same interpreter, having already imported `ceqjepa.pi_assign` for its
own use. The test measures interpreter-global state and is therefore order-dependent; the
independence property it was written to check (`t_length.py` imports no assignment rule) holds
in every ordering.

This is a defect in the test's mechanism, not a finding about the module, and not a repair
this card is permitted to make. It is recorded here as an unresolved red run under the exact
command specified. Every number in the four rows below comes from R1, R2 or R3, all of which
exit 0.

## Summary table

| # | claim | theorem | hypothesis census | planted result | control result | verdict by class |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | the assignment rule | `three_corners_containment` | checked, R1 (h) | rule serves 7/9, breaks 0, refuses 2 | `all_softmax` 3/3 inside INTENSIVE | INTENSIVE **tie** · EXTENSIVE **tie** vs `all_linear`, **win** vs `all_softmax` · cross-class **win** · FLOOR **void** · NOT-A-POWER-LAW **void** |
| 2 | T-length | representational collapse / hull bound | checked, R2 (a) and (j) | learned-beta 2.0322 worst cell | assigned-oracle 0.0804 worst cell, 58 vs 60 params | **win** over learned-beta · **void** against `all_softmax` and `all_linear` (bed calibration) |
| 3 | the refusal channel | none — a cost measurement | checked, R2 (l) | MIXED conservative, NOT-A-POWER-LAW protective | solo corner 0.0526 / 0.2884 at 4x | MIXED-SCALING **win** (refusal survives the flip test) · NOT-A-POWER-LAW **win** (ranking flips) |
| 4 | the consequence swap | symmetry of `\|dq(i)-dq(j)\|` is an algebraic identity | checked, R3 (b) and (f) | density ratio 0.9 | oracle 0.170141 vs exactly 0.000000 | exact read **win** · estimated read **void** |

The three words, as used here. **Win**: the instrument beats the baseline on a bed built
inside that baseline's own class, with the baseline's hypotheses checked on the data.
**Tie**: the baseline matches the instrument on that class, or the comparison is unmatched
(parameter counts, training budget). **Void**: the comparison cannot be scored, because the
quantity it is scored against is one the same run refuses as meaningless.

---

## Row 1 — THE ASSIGNMENT RULE

Source: R1, `python -m ceqjepa.pi_assign`, exit 0, 2.11 s.

**THEOREM.** `three_corners_containment`, read at `lean/CEQ/V16Domain.lean:433`, with
`corners_are_distinct` at `:445`. R1 (h) prints the source string
`READ lean/CEQ/V16Domain.lean:433 (corners_are_distinct :445)`. The file exists in this tree
and line 433 reads `theorem three_corners_containment {g₀ : ℕ → ℝ} (hg : ∀ k, g₀ k = 0)`,
concluding that `Hop 1 g₀ qk = softmaxAttn`, `Hop 0 g₀ qk = linearAttn`, and
`Hop 0 g (fun _ _ => 0) = CEQ.V15.Wc g` — one family, three named operators at three settings
of `(β, g, QK)`. `corners_are_distinct` at `:445` establishes that at `g ≡ 0, QK-off` the
`β = 1` corner reads `1/2` at `(i,j) = (1,0)` and the `β = 0` corner reads `1`, so the
containment describes a family with interior rather than three names for one operator.

`β = 1 − α` is derived, not chosen. R1 (e) states the derivation and then measures it:
`W = e^l/Z^beta` carries `N^-beta`, so a read carries `N^(1-beta)`.

| β | predicted α_out | measured α_out | CI | gap |
| --- | --- | --- | --- | --- |
| 0.00 | +1.0000 | +1.0089 | [+0.9931, +1.0248] | +0.0089 |
| 0.25 | +0.7500 | +0.7593 | [+0.7451, +0.7734] | +0.0093 |
| 0.50 | +0.5000 | +0.5093 | [+0.4968, +0.5217] | +0.0093 |
| 0.75 | +0.2500 | +0.2589 | [+0.2483, +0.2695] | +0.0089 |
| 1.00 | +0.0000 | +0.0082 | [−0.0006, +0.0169] | +0.0082 |

Worst gap over the sweep: 0.0093. The statement that the softmax read is a mean and the linear
read is a total is therefore measured across five settings of β rather than asserted at two.

**HYPOTHESIS CENSUS.** R1 (h) checks the theorem's hypotheses on the bed that is actually used:

- gate `g == 0`: `True`
- logits finite: `True` over 65536 entries
- `n = 64`
- `β = 1` row sums `1.000000000000`, worst `|sum − 1|` `2.220e-16`
- `β = 0` row sums `104.966973`, corners distinct: `True`

A second, independent hypothesis is checked in R1 (a): only a dimensionless group may enter
`exp`. Unit exponents as numbers — `logit z_value V^+0 U^+0 N^+0 sum 0 exp accepted: True
max|l| 4.3327`; `logit log_ratio_u V^+0 U^+0 N^+0 sum 0 exp accepted: True max|l| 2.2016`. The
planted violation `logit raw_value` carries `V^+1 U^+0 N^+0` and is refused as
`DIMENSIONFUL-EXPONENT`; a second plant carrying `V^+1 U^+0 N^-1` sums to zero and is refused
all the same, so sum-zero is shown on the data to be necessary and not sufficient.

The power-law test is checked against the alternative that would have been natural. R1 (c):
a design refusing at `R^2 < 0.95` would refuse 2 of the 8 planted power laws, `prob_above` and
`ratio_vu`; χ² separates them, power laws at worst `4.5849` against the plant's `7.827e+06`,
critical `20.0`, which on 2 dof is `p = 4.540e-05` per coordinate and `4.085e-04` over the 9
tested.

**PLANTED RESULT.** The sweep is `n = 16, 32, 64, 128`, 1024 replicates in 32 blocks, seed
3301. R1 (b):

| coordinate | α | CI (2 dof) | resid | R² | χ² | verdict | planted truth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_value | −0.0021 | [−0.0034, −0.0007] | 0.000491 | 0.9567 | 0.4204 | β=1 INTENSIVE | INTENSIVE |
| prob_above | −0.0330 | [−0.0645, −0.0015] | 0.011569 | 0.9104 | 4.585 | β=1 INTENSIVE | INTENSIVE |
| ratio_vu | −0.0041 | [−0.0073, −0.0010] | 0.001151 | 0.9416 | 1.559 | β=1 INTENSIVE | INTENSIVE |
| count_above | +0.9670 | [+0.9355, +0.9985] | 0.011569 | 0.9999 | 4.585 | β=0 EXTENSIVE | EXTENSIVE |
| sum_value | +0.9979 | [+0.9966, +0.9993] | 0.000491 | 1.0000 | 0.4204 | β=0 EXTENSIVE | EXTENSIVE |
| total_u | +1.0000 | [+0.9983, +1.0017] | 0.000634 | 1.0000 | 0.697 | β=0 EXTENSIVE | EXTENSIVE |
| half_walk | +0.5180 | [+0.4703, +0.5657] | 0.017537 | 0.9991 | 1.385 | REFUSED MIXED-SCALING | MIXED |
| osc_shape | −0.0000 | [−3.3460, +3.3459] | 1.229569 | 0.0000 | 7.827e+06 | REFUSED NOT-A-POWER-LAW | NOT-A-POWER-LAW |
| log_shape | +0.0387 | [+0.0369, +0.0406] | 0.000676 | 0.9998 | 3.073 | β=1 INTENSIVE | FLOOR |

All four must-fires fire: INTENSIVE on three coordinates, EXTENSIVE on three, MIXED-SCALING on
`half_walk`, NOT-A-POWER-LAW on `osc_shape`. The ninth coordinate is the planted floor and is
treated in Limits.

`half_walk`'s refusal has two independent routes, not one. R1 (f) compares declared units
against measured α per coordinate; `half_walk` declares `N^+1.0` and measures `+0.5180`, a
disagreement of `+0.4820`, against `total_u`'s `+0.0000` and `sum_value`'s `+0.0021`.

**CONTROL RESULT.** R1 (g) scores the rule against the three simplest things that could have
been done on the same bed. A coordinate counts as *served* when the corner's output exponent
`1 − β` matches that coordinate's own measured α within the `0.10` tolerance.

| arm | served | broken | refused | EXTENSIVE | INTENSIVE | FLOOR | MIXED | NOT-A-POWER-LAW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule | 7/9 | 0 | 2 | 3/3 | 3/3 | 1/1 | 0/1 | 0/1 |
| all_softmax | 5/9 | 4 | 0 | 0/3 | 3/3 | 1/1 | 0/1 | 1/1 |
| all_linear | 3/9 | 6 | 0 | 3/3 | 0/3 | 0/1 | 0/1 | 0/1 |
| random | 5/9 | 4 | 0 | 2/3 | 3/3 | 0/1 | 0/1 | 0/1 |

R1 states the control result in its own words: `L-FAIL both directions: all_softmax serves 3/3
INTENSIVE (the rule 3/3, a tie) and 0/3 EXTENSIVE (the rule 3/3). It wins inside its class and
loses outside it.`

The attack in R1 (j) is the sharper control. A normalisation layer that hides `n` moves the
total-norm exponent from `+0.9982` clean to `+0.0000` attacked, and under it the rule alone
silently reclassifies 3 of 3 planted extensive coordinates as intensive
(`count_above +0.9670 → −0.0271`, `sum_value +0.9979 → −0.0005`, `total_u +1.0000 → +0.0018`).
The two-leg census catches it: on the clean bed `fires=False`, total norm spread `8.74e-01`
not pinned, declared-minus-measured offset median `+0.0031` with IQR `0.0314` inside ±0.25;
on the attacked bed `fires=True`, spread `0.00e+00` pinned, offset median `+1.0024` with IQR
`0.0271` over 8 coordinates. The refusal is structural: `assign()` returns a dict whose verdict
is set and whose every reachable β is that refusal, so 0 of the 9 coordinates hand a caller
back a float.

**VERDICT BY CLASS.**

- INTENSIVE — **tie**. `all_softmax` 3/3, rule 3/3, matched. The run says so itself.
- EXTENSIVE — **tie** against `all_linear` (3/3 each); **win** against `all_softmax` (0/3
  against 3/3).
- Across the two classes jointly — **win**. No single-corner arm serves both: `all_softmax`
  5/9, `all_linear` 3/9, rule 7/9 with 0 broken. That the rule breaks nothing while both
  corners break 4 and 6 is the measured difference.
- FLOOR — **void**. Both the rule and `all_softmax` are scored as serving `log_shape`, and both
  are wrong in kind: the coordinate is not a power law. See Limits.
- NOT-A-POWER-LAW — **void**. `all_softmax` is scored 1/1 here only because the degenerate fit
  returns α = −0.0000, which `1 − β = 0` matches; the same run refuses that α at χ² `7.827e+06`
  against a critical `20.0`. A serve scored against an α the run itself refuses is not a score.
- MIXED-SCALING — no arm serves it; the cost of the refusal is Row 3.

---

## Row 2 — T-LENGTH

Source: R2, `python -m ceqjepa.t_length`, exit 0, 129.98 s.

**THEOREM.** Two, both other people's, both used as calibration rather than as findings.
arXiv:2410.01104 (Veličković et al., ICML 2025) — softmax circuits must disperse as the item
count grows. arXiv:2406.04267 (Barbero et al., NeurIPS 2024) — Theorem B.3 gives
representational collapse of the last-token representation in L1 under four hypotheses,
Corollary B.10 turns it into the counting failure, Proposition B.9 is the non-asymptotic
version with no positional encodings. The operative consequence on this bed is the hull bound:
a normalised read lands inside the convex hull of the values it pools, so its reachable
interval is fixed while an extensive target is not.

**HYPOTHESIS CENSUS.** Two hypothesis sets are checked on the data, both in R2.

The bed's own structure, R2 (a). One data set, two bars, the extensive target exactly `n` times
the intensive one: `n=16` intensive mean `0.4682` and extensive `7.4915`; `n=64` `0.4635` and
`29.6645`; `n=128` `0.4674` and `59.8210`; worst `|extensive − n * intensive|` over all three
lengths `0.000e+00`. The two bars are therefore one experiment and not two. The class
hypotheses on the data: the intensive mean drifts `0.0009` across the lengths, the extensive
mean grows `7.9852x`; count exponents from a length sweep over `(16, 32, 64, 128)` are
`−0.0042` intensive and `+0.9958` extensive, with no R² gate anywhere.

The theorem's own hypothesis, on the model's values rather than on the target, R2 (j):
`max|u|` for the softmax head is `5.1883` at `n=16`, `5.1973` at `n=64`, `5.2104` at `n=128` —
a growth of `1.0043x` while the extensive target grows `8.0x`. R2's phrasing: the reachable
interval is fixed and the target is not, which is the whole of the theorem. The hull property
itself is checked on the pooled outputs in R2 (b): `the beta=1 column lies inside the convex
hull of u at every row: True`.

The guard that would void the whole row is exercised, R2 (c): testing at the training length
only is refused as a value carrying its reason —
`length_refusal(16, (16,)) -> no test length exceeds the training length 16: (16,) is the
training length again, which measures nothing about length`, and the report publishes no table
when it fires.

**PLANTED RESULT.** 3 seeds × 2000 Adam steps at lr 0.03 on 512 sequences of length 16, torch
float64, predictions pooled over seeds before the error is taken. Errors are nrmse.

| arm | params | 1x int | 1x ext | 4x int | 4x ext | 8x ext |
| --- | --- | --- | --- | --- | --- | --- |
| lookup-table | 0 | 0.1464 | 0.1464 | 0.1128 | 0.7548 | 0.8793 |
| marginal-frequency | 0 | 0.4190 | 0.4190 | 0.3972 | 0.7895 | 0.8964 |
| all-softmax | 58 | 0.0026 | 0.0019 | 0.0019 | 0.7500 | 0.8750 |
| all-linear | 58 | 0.0555 | 0.0952 | 3.0229 | 0.2541 | 0.2889 |
| learned-beta | 60 | 0.0160 | 0.0123 | 1.1477 | 0.2757 | 0.3739 |
| assigned-oracle | 58 | 0.0026 | 0.0025 | 0.0014 | 0.0702 | 0.0804 |
| assigned-measured | 58 | 0.0026 | 0.0025 | 0.0014 | 0.0702 | 0.0804 |

The 8x intensive column, which R2 prints separately: all-softmax `0.0018`, all-linear `7.0671`,
assigned-oracle `0.0011`.

**CONTROL RESULT.** The only baseline that could have discovered the assignment by itself is
the learned-β head, and the parameter counts are matched in the direction that favours it.
R2 (g): assigned-oracle 58 parameters, learned-beta 60 — the difference is exactly 2, one
exponent logit per output coordinate, and it is the learned head that carries the extra.

The learned head landed at `(0.5762, 0.2054)` against the assigned mask `(1.0, 0.0)`; per seed
`(0.54,0.29) (0.60,0.20) (0.59,0.13)` — never near the mask. It was given the best of 3
configurations by training loss at each seed (`(2000, 0.03)`, `(2000, 0.1)`, `(4000, 0.03)`),
so under-training is ruled out by construction: training loss `learned 0.002096, assigned
0.000055`. The identifiability diagnostic says why it is under-determined instead —
`log(sum w)` moves sd `0.0015` across instances at `n=16` and `1.3863` between `n=16` and
`n=64`, a ratio of `900`, so the training set carries three orders of magnitude less signal
about the exponent than the test asks of it.

R2's verdict line: `VERDICT WIN -- worst cell over the test lengths: learned-beta 2.0322 at 60
parameters, assigned-oracle 0.0804 at 58, band 1.50`.

The single-corner arms are calibration and are pre-registered as such. R2 (e) states the
direction and magnitude before reading the numbers: predicted softmax on extensive `0.7500` at
`n=64` and `0.8750` at `n=128`, linear on intensive `3.0000` and `7.0000`. Measured: softmax
`0.7500` and `0.8750` — agreement `+0.0000` absolute at both lengths; linear `3.0229` and
`7.0671` — `+0.76%` and `+0.96%` relative. Both failures grow with length. The intensive
control, where the softmax arm must win, fires too: softmax `0.0019` at `n=64` against the best
of the arms it must beat, `0.1128`, and `0.0018` at `n=128` against `0.1399`.

The confound is censused rather than assumed. R2 (i): R² for predicting `n` from per-token
feature means `0.0019`, from the feature extremes `0.7697`, and on a bed with a planted
position feature `1.0000`; a raw position feature moves the softmax extensive bar from `0.7500`
to `0.7185`, `4.21%`. The positive control fires — hand the same head the true `n` and multiply
and the extensive bar collapses `0.7500 → 0.0030`, so the missing quantity is exactly the
factor `n` and this bed does not carry it.

The measured mask, not the oracle one, is what the harness is fed in R2 (k): axis `'D'` length
9, 7 of 9 coordinates assigned, 2 refused (`half_walk`, `osc_shape`); measured `(1.0, 0.0)`
against oracle `(1.0, 0.0)`, `agree True`; cells identical at `n=64` (`0.0014 / 0.0702`) and
`n=128` (`0.0011 / 0.0804`). R2 refuses to present that identity as confirmation: `presenting
it as independent confirmation would be a tautology dressed as a measurement`.

**VERDICT BY CLASS.**

- Against learned-beta, matched to within 2 parameters in the learned head's favour — **win**.
  `2.0322` worst cell against `0.0804`.
- Against `all-softmax` and `all-linear` — **void as findings**. Both replicate published
  theorems and are declared bed calibration before the run; the softmax arm's agreement is
  `+0.0000` absolute at both lengths, which is what licenses the bed, not what the bed shows.
- Against the zero-parameter floors (lookup-table, marginal-frequency) — **win** at 4x and 8x
  extensive (`0.7548`/`0.8793` and `0.7895`/`0.8964` against `0.0702`/`0.0804`), but these
  carry 0 parameters and so are a floor, not a contest.
- The exponent is **not** the whole assignment, and the run corrects itself on this. R2 (h):
  with the output constant free, assigned reads `0.0702` at 4x and `0.0804` at 8x; with it
  zeroed, `0.0012` and `0.0009` — a factor of `58.6`. The residual above is the constant, not
  the length.

---

## Row 3 — THE REFUSAL CHANNEL

Source: R2 (l), `python -m ceqjepa.t_length`, exit 0, 129.98 s.

**THEOREM.** None is claimed. This row is a cost measurement: what a reader loses by honouring
a refusal instead of forcing a corner. The construction is stated rather than asserted — each
target is built the way the coordinate it maps onto is planted.

**HYPOTHESIS CENSUS.** Both refused targets are characterised on the data before they are
scored. MIXED-SCALING maps onto `half_walk`: the sum of a zero-mean per-token feature, a signed
running total whose magnitude grows like the square root because terms cancel, measured
`alpha +0.5024`, between the corners. NOT-A-POWER-LAW maps onto `osc_shape`: a per-sequence
quantity scaled by an amplitude that is a function of the context length alone, norm `0.5083`
at `n=16`, `1.2618` at `n=64`, `0.5092` at `n=128` — the same at 8x as at the training length
and different at 4x, so its slope is `-0.0006` and describes nothing. Every arm in both tables
carries 58 parameters; 1 seed, 1500 steps, 256 sequences.

**PLANTED RESULT.** MIXED-SCALING, assignable / refused bar:

| arm | 4x | 8x |
| --- | --- | --- |
| force-softmax | 0.2261 / 0.9099 | 0.2200 / 0.9601 |
| force-linear | 0.0764 / 4.5369 | 0.0673 / 7.1865 |
| solo-softmax | not trained / 0.7500 | not trained / 0.8750 |
| solo-linear | not trained / 0.0526 | not trained / 0.0694 |
| refuse | 0.0006 / REFUSED | 0.0004 / REFUSED |

NOT-A-POWER-LAW:

| arm | 4x | 8x |
| --- | --- | --- |
| force-softmax | 0.0314 / 0.5995 | 0.0297 / 0.0144 |
| force-linear | 0.1891 / 0.5564 | 0.2111 / 6.6509 |
| solo-softmax | not trained / 0.5999 | not trained / 0.0017 |
| solo-linear | not trained / 0.2884 | not trained / 5.1177 |
| refuse | 0.0006 / REFUSED | 0.0004 / REFUSED |

The bump's cost is pre-registered before the run: a model that learns the amplitude at `n0` and
cannot learn that it moved costs `|1 - A(n0)/A(n)|`, `0.6000` at 4x and `0.0000` at 8x.
Measured on the softmax corner trained alone: `0.5999` and `0.0017`.

**CONTROL RESULT.** The two refusal reasons do not cost the same, and the solo rows are the
control that separates *does a corner represent this coordinate* from *what does forcing it
cost in a table*. Trained alone, the best corner costs `0.0526` on MIXED-SCALING and `0.2884`
on NOT-A-POWER-LAW at 4x; sharing a trunk the same corners cost `0.9099` and `0.5564`. On
MIXED-SCALING the linear corner alone reads `0.0526` while the same corner sharing a trunk
reads `4.5369` — a factor of 86, and that factor is the trunk, not the exponent. R2 records
that its own prediction was wrong here and the run corrected it.

The discriminator is not a threshold but whether one corner wins at every test length:

- MIXED-SCALING — winner per length `['solo-linear']`. One corner wins at both (`0.0526` and
  `0.0694` against the other corner's `0.7500` and `0.8750`), so a corner exists and the
  refusal is **conservative** about it. Honouring it costs a reader a model that works when the
  coordinate is not sharing a value net.
- NOT-A-POWER-LAW — winner per length `['solo-linear', 'solo-softmax']`, the ranking **flips**.
  force-softmax reads `0.5995` at 4x and `0.0144` at 8x, force-linear `0.5564` and `6.6509`,
  solo-softmax `0.5999` and `0.0017` — not monotone. A reader testing only at 8x would certify
  a corner that is wrong at 4x. The refusal is **protective**, and no finite set of test lengths
  substitutes for it.

The price of the careless default is measured in the same units. A caller writing
`b if b is not None else 1.0` turns the refusal into the softmax corner with no trace; on the
real vector, mapping the extensive target onto `half_walk`, the careless path returns
`(1.0, 1.0)` while the harness returns `None` carrying its reason. That careless number is the
force-softmax row, `0.9099` at 4x.

The softmax row on MIXED-SCALING carries a second result: its error is `0.7500` and `0.8750`
against the closed form `1 − n0/n = 0.7500` and `0.8750` — the same law as for a plain count,
although this coordinate's norm exponent is about `0.50` and not 1. The exponent the rule
measures governs the magnitude; the one that governs the readout error is the declared count
exponent.

**VERDICT BY CLASS.**

- MIXED-SCALING — **win**. A corner exists, one corner wins at both lengths, and the refusal is
  correctly labelled conservative with the cost of honouring it stated as `0.0526` at 4x.
- NOT-A-POWER-LAW — **win**. The ranking flip between 4x and 8x is measured, which is the
  statement that no finite set of test lengths substitutes for the refusal.
- Against a forced corner in a shared table — **win** for the refusal on both reasons
  (`0.0006` and `0.0004` on the assignable bar against `0.2261`/`0.0764` and `0.0314`/`0.1891`).

---

## Row 4 — THE CONSEQUENCE SWAP

Source: R3, `python -m ceqjepa.intent_do`, exit 0, 3.30 s.

**THEOREM.** One negative result, proved algebraically rather than measured. R3 (f): the field
equation's own form `T = |dq(i) - dq(j)|` is symmetric as an algebraic identity, so no bed can
make it carry a direction. Measured over every read and story, `|T_field - T_field^T| worst =
0.0`. The replacement is stated in the same breath: the directed source needs the two-index
form `T(i,j) =` movement at `j` under an intervention at `i`.

**HYPOTHESIS CENSUS.** The bed is synthetic and exact — 12 slots, 8 outcomes, branch slots
`[2, 4, 5, 8, 10]` with outcome pools `[(0,1), (1,2), (2,3), (3,4), (4,0)]` forming a cycle,
every other slot decor, seed 20260913. The admitted-future set is the intersection of the
tokens' masks, so the ground truth of every swap is known by construction; the read is the
uniform distribution over that set, a count of futures, not a language model and not a trained
probe. Admitted-set sizes over the first 8 stories: `[4, 5]`.

The hypothesis the planted negative depends on is checked token by token and not on average,
R3 (b): 20 branch tokens, consequence partners at Hamming `[3]`, surface-only partners at
Hamming `[3]` — matched everywhere. Without this the surface-only control would be measuring
distance rather than consequence.

**PLANTED RESULT.** Three swap families, each with a CI and the worst single swap carried
beside the mean so that a handful of corrupted swaps cannot hide in an average.

| read | family | mean | CI | worst | swaps |
| --- | --- | --- | --- | --- | --- |
| oracle | consequence | 0.170141 | [0.165219, 0.175062] | 0.240909 | 1000 |
| oracle | surface_only | 0.000000 | [0.000000, 0.000000] | 0.000000 | 1000 |
| oracle | null | 0.000000 | [0.000000, 0.000000] | 0.000000 | 2400 |
| counts | consequence | 0.163626 | [0.158706, 0.168546] | 0.296877 | 1000 |
| counts | surface_only | 0.011893 | [0.011632, 0.012154] | 0.028126 | 1000 |
| counts | null | 0.000000 | [0.000000, 0.000000] | 0.000000 | 2400 |
| density | consequence | 0.261442 | [0.253217, 0.269668] | 0.685914 | 1000 |
| density | surface_only | 0.287905 | [0.280290, 0.295520] | 0.627901 | 1000 |
| density | null | 0.000000 | [0.000000, 0.000000] | 0.000000 | 2400 |

Gaps: oracle difference `0.170141`, ratio undefined because surface-only is exactly `0.0`;
counts difference `0.151733`, ratio `13.8`; density difference `-0.026463`, ratio `0.9`.

Directedness, R3 (g): `|T(i,j)-T(j,i)|` over 66 pairs, median `0.159167`, p90 `0.173333`, max
`0.177500`, against a symmetrised control that reads median `0.0` and max `0.0` — zero by
construction, since `(T + T^T)/2` is symmetric for any `T`, and named in the run as the scale
the asymmetry is read against rather than as an independent arm. `0.6818` of pairs are
asymmetric above tolerance.

**CONTROL RESULT.** The planted negative fires. An alikeness read — what self-attention scores,
arXiv:1706.03762 — has no access to consequence, so on matched-surface swaps its gap must
collapse, and it does: density ratio `0.9`, against oracle undefined and counts `13.8`. The
density read scores its surface-only family *higher* than its consequence family.

The density confound is then closed on its own terms, R3 (d2). The generator splits the
consequence swaps into those that leave the number of admitted futures identical and those that
do not. On the entropy-preserving half no density, surprisal or entropy can move at all:

| read | subset | swaps | mean dq | CI | mean abs dH | max abs dH |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | entropy_preserving | 492 | 0.132431 | [0.123717, 0.141145] | 0.000000 | 0.0 |
| oracle | entropy_moving | 508 | 0.206663 | [0.205186, 0.208140] | 0.239024 | 0.287682072451781 |
| counts | entropy_preserving | 492 | 0.140814 | [0.131536, 0.150093] | 0.002077 | 0.015725166561852078 |
| density | entropy_preserving | 492 | 0.259780 | [0.247853, 0.271706] | 0.281408 | 1.2289925075649644 |

The `max abs dH` column is quoted at the precision R3 prints it, which is full float repr in
that column and 6 decimals elsewhere; the widths are the run's, not this card's.

R3 names which half is the measurement and which is an identity: the exact read is uniform, so
its entropy is the log of the count and equal counts force `0.0`; the finding is the other
half, `dq = 0.132431` over 492 swaps where every entropy, surprisal and density is pinned at
zero.

The forward-only question is answered with the tautology labelled as one, R3 (h): the causal
prefix read has backward max `0.0` over 3960 entries, which is zero by the mask; the
bidirectional read has backward mean `0.074954` with 1391 of 3960 entries nonzero, so
consequence on a context window is not forward-only once the mask is removed, and the direction
lives in the asymmetry rather than in the order.

The refusal is scored with sensitivity and specificity kept separate and both trivial criteria
in the table, R3 (j), over 1920 cases labelled by the generator's constraint sets, which the
criterion never touches:

| read | criterion | sensitivity | specificity | tp/fn/tn/fp |
| --- | --- | --- | --- | --- |
| oracle | criterion (ours) | 100.00% | 100.00% | 105/0/1815/0 |
| oracle | refuse everything | 100.00% | 0.00% | 105/0/0/1815 |
| oracle | answer everything | 0.00% | 100.00% | 0/105/1815/0 |
| counts | criterion (ours) | 0.00% | 100.00% | 0/105/1815/0 |
| density | criterion (ours) | 0.00% | 100.00% | 0/105/1815/0 |

**VERDICT BY CLASS.**

- Exact read against the similarity read, on matched-surface swaps — **win**. `0.170141`
  against exactly `0.000000`, with the density arm's ratio collapsing to `0.9` on the same
  swaps.
- Exact read against the density/entropy explanation — **win**. `0.132431` over 492 swaps where
  `|dH|` is exactly `0.000000`.
- Directedness — **win** against the symmetrised control, with the control's zero declared as
  by-construction rather than as evidence.
- The algebraic-symmetry kill — **holds**, with a replacement. `|T_field - T_field^T| worst =
  0.0` kills the one-index form outright; the two-index form is the stated replacement and is
  what produces the asymmetry above.
- Estimated read — **void**. At the pinned tolerance the counts row is cell for cell the
  answer-everything row (`0/105/1815/0`): an estimated read cannot certify that nothing moved,
  so it cannot refuse. R3 (k) measures whether any threshold recovers it —
  `1e-12`, `1e-06`, `1e-04` all give 0.00% sensitivity; `1e-02` gives 20.95% sensitivity at
  82.20% specificity with 323 false positives; `1e-01` gives 100.00% / 100.00% — and then
  disqualifies that row itself: it is chosen against the labels, an upper bound on what a
  threshold could do, not an operating point anyone could have set in advance.
- Forward-only — **void as a finding** on the causal read, by the run's own labelling: backward
  max `0.0` is zero by the mask.

---

## What is cited, not claimed

Two arms in Row 2 are prior art reproduced as bed calibration. They are not findings of this
round, and the card does not count them as wins.

**arXiv:2410.01104** — Veličković, Perivolaropoulos, Barbero, Pascanu, *Softmax is not Enough
(for Sharp Size Generalisation)*, ICML 2025. Already marked `[V]` in this tree. The claim that
`docs/sources/sweep/sweep_expressivity.md:325-330` says what it is said to say was verified by
reading those lines. Line 325 is the entry header, carrying the arXiv id and the `[V]` mark;
lines 326-327 are the `Owns` field:

> **Owns.** Softmax circuits must disperse as item count grows; a size-generalisation failure
> of sharpness.

Lines 328-329 are the `Leaves open` field — a length-generalisation limitation of the row
mixture itself, supporting obstruction 1 at the level of sharpness and not joint determination
— and line 330 is `- **Found by.** Memory; abs fetched.` The brief's description is accurate:
the entry is present, marked `[V]`, and its `Owns` statement is the dispersion result the
softmax arm reproduces.

**arXiv:2406.04267** — Barbero et al., NeurIPS 2024. R2 (m) records what is being borrowed:
Theorem B.3 (representational collapse of the last-token representation in L1, under four
hypotheses), Corollary B.10 (the counting failure), Proposition B.9 (the non-asymptotic version
with no positional encodings).

What that buys, and what it does not. The softmax arm's job here is to show the bed reproduces
a known theorem to `+0.0000` absolute at both test lengths, which is what licenses the one
measurement on unoccupied ground: Row 2's per-coordinate assignment against the learned head at
matched parameter counts. A bed that failed to reproduce the theorem would not be a bed.

Three further citations are relayed rather than verified here, and R1 and R2 both mark them as
relayed. arXiv:2202.04643 (Bakarji et al., *Nature Computational Science* 2022) is the nearest
prior work: BuckiNet learns one exponent per input coordinate, but through a single shared
layer against a null-space loss, and the exponents are fitted rather than swept. arXiv:2006.
16236 (Katharopoulos et al. 2020) is the normalised/unnormalised axis itself; arXiv:2108.12409
(Press et al.) is the length-generalisation line. Dimensional consistency is not claimed as
new.

Every "cannot" in Row 2 is scoped, because arXiv:2511.20038 Theorem 4.3 proves
length-generalizable softmax chain-of-thought transformers Turing-complete with relative
positional encodings. The scope of the statements here is one forward pass, no scratchpad, and
the representation of one readout row. Unscoped, they are refuted.

---

## The honest sentence

> On one synthetic bed at commit `c9a9434`, a per-coordinate pooling exponent measured from a
> four-point context-length sweep recovers both corners where each single corner recovers one
> (7/9 served and 0 broken, against `all_softmax` 5/9 and 4 broken and `all_linear` 3/9 and 6
> broken), beats a learned exponent head at 58 parameters against 60 by 0.0804 against 2.0322
> nrmse in the worst cell over 4x and 8x the training length, refuses two coordinates of the
> nine, on one of which the forced-corner ranking flips between those same lengths (solo-linear
> 0.2884 against solo-softmax 0.5999 at 4x, and 5.1177 against 0.0017 at 8x), and — on an exact
> count-of-futures read, not an estimated one — separates consequence from matched-surface
> similarity at 0.170141 against exactly 0.000000, including 0.132431 over the 492 swaps where
> every entropy is pinned at zero.

Not in that sentence, and why:

- The per-class comparisons on INTENSIVE and EXTENSIVE, which are ties (`all_softmax` 3/3
  against the rule's 3/3; `all_linear` 3/3 against the rule's 3/3). Only the cross-class
  statement is a win.
- The `all-softmax` and `all-linear` arms of Row 2, which are prior art reproduced as
  calibration.
- The FLOOR coordinate, which both the rule and `all_softmax` are scored as serving and both
  get wrong in kind.
- The NOT-A-POWER-LAW serve count, which is scored against an α the same run refuses.
- The estimated (`counts`) read of Row 4, whose criterion is cell for cell the answer-everything
  row.
- The forward-only result on the causal read, which is zero by the mask.
- The measured-mask-versus-oracle agreement of R2 (k), which is identical by construction.

---

## Limits

**The four-point sweep has a floor, and it mislabels a logarithm.** R1 (d) gives it in closed
form: four points leave 2 dof, a quadratic departure `q` leaves residuals `q*(+1,-1,-1,+1)`, and
the smallest refusable curvature is `q_floor = sqrt(20.0/4)*se = 2.2361*se = 0.001866` in log2
units. The planted logarithm carries curvature `-0.000375`, five times below that floor, so it
passes at χ² `3.0730` and R² `0.9998` and is assigned `beta=1` at `alpha +0.0387`. It is not a
power law and this instrument cannot say so. R2 (k) checks whether the floor bites its own bed
and reports that it does not — the two targets there measure `alpha -0.0042` and `+0.9958`
against the trap's `+0.0387` — but records that it would bite any coordinate whose growth is
logarithmic, and that the harness could not tell.

**The measured mask was checked against the oracle on two coordinates, and they are the two
easiest.** R2 (k) compares the vector, not the row: measured `(1.0, 0.0)` against oracle
`(1.0, 0.0)`, `agree True`. Those two are `prob_above` at `alpha -0.0330` and `count_above` at
`+0.9670` — the coordinates furthest from either boundary, out of the 9 the rule assigns. The
resulting cells are identical at `n=64` (`0.0014 / 0.0702`) and `n=128` (`0.0011 / 0.0804`)
because they are the same vector through the same pipeline at the same seeds. R2 says so
itself: presenting it as independent confirmation would be a tautology dressed as a
measurement. Nothing here tests the mask on the 5 remaining assigned coordinates or on either
refused one.

**The tiebreak that would convert a refusal into an assignment was killed by a construction the
rule cannot distinguish.** R1 (l) treats it as a hypothesis and runs it: `half_walk` declares
`N^+1.0` and measures `+0.4890`; `sqrt_total` — a sum over the first `ceil(sqrt(n))` tokens, a
top-k pool at `k = sqrt(n)` — declares `N^+1.0` and measures `+0.5152`. The two differ by
`0.0262`, inside the `0.10` tolerance, and the tiebreak reads the same inputs for both. It would
hand `sqrt_total` the linear corner, whose read carries `n^1` against a coordinate scaling as
`n^0.5152`, missing by `0.4848` — `4.8` times the tolerance. Refused as a rule; the
MIXED-SCALING refusal stands and the declared-minus-measured gap ships as a diagnosis instead.
R2 (l) proposes the route that would revive it (prefer the declared exponent where declared and
measured disagree and declared is a corner) and that route is exactly what R1 (l) kills on
`sqrt_total`. The two runs disagree about whether this refusal is convertible; R1 is the one
that ran the counterexample.

**One fabrication class survives the guard.** The published-value guards pin every printed
number to an expression that recomputes it from the module's own functions in the same run,
position by position for vectors so that a single frozen entry cannot hide inside an aggregate
tolerance. Run under `-s`, they report their own surface: `24 published values re-derived
against the module in this run` (`pytest tests/curvature/test_pi_assign.py -k rounded_literals`),
`22` (`pytest tests/curvature/test_t_length.py -k recomputed_not_literals`, 131.44 s), and
`76 published values re-derived against the module in this run, 1 exempt`
(`pytest tests/curvature/test_intent_do.py -k recomputed_not_literals`). The guard is not
vacuous — `pytest tests/curvature/test_intent_do.py -k vacuous_against_planted` prints
`5 planted fabrications, each seen by both counters and absent from the run`, covering a plain
decimal, a fragment of a number the run really prints, a bare integer, a number at the end of a
sentence, and a dotted identifier. What survives is stated in the guard's own docstring and is a
property of equality, not a defect: equality cannot separate a live computation from a constant
typed at full precision. A value frozen at every digit it has passes all 122 pinned values
above; the same value frozen as the figure the run prints fails. Only a different kind of check
moves that ceiling — perturbing an input and requiring the published value to move, which R1
gets from the freeze digest (`digest 854dc3aae168dd98e0e631e189aa30c0800cd8d1a61e69433b2b2ea21fad504a
check ok=True`, and a planted refit at seed 7717 → `a3180fe6b4807220 fires: True`) and the other
two modules do not.

**The normalizer census has a hole its own run names.** R1 (j): a normalizer dividing by an
independent noisy estimate of `n` pins no norm, so leg 1 misses it and only leg 2 catches it —
and leg 2 needs declared units, which not every consumer supplies.

**Row 2's residual is a constant, not a length.** R2 (h) corrects the instance that predicted
the assigned arm exact on both bars: output constant free gives `0.0702` at 4x and `0.0804` at
8x, zeroed gives `0.0012` and `0.0009`, a factor of `58.6`.

**Scope.** One synthetic bed per row, one training length (16) in Row 2, 2 targets, 4 masks, 8
arms scored; 3 seeds in the main Row 2 table and 1 seed in the Row 3 tables; 12 slots and 8
outcomes in Row 4, seed 20260913. Nothing here is measured on natural data, on a trained
language model, or on more than one machine. The `ceq/arm_smprime.py` wiring the modules point
at (`:527` for the 0-dim β parameter, `:250` where it is applied, `:264` for the value
contraction, `tests/gate0/test_g13_beta_learnable.py:130` for the empty-shape assertion) was
confirmed to exist in this tree but was not run or read for this card.

**One red run.** `R4` exits 1 on `test_the_module_imports_no_assignment_rule`, which fails only
when `test_pi_assign.py` is collected ahead of it in the same interpreter and passes alone in
1.96 s. The mechanism is documented above under *The runs*. No number in Rows 1-4 depends on
that test.
