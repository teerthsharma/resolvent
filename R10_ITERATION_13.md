# R10 — iteration 13: attack #1, and a strike on the attack

**Script task.** *"MARS: attack #1 = a confound C with predicted effect δ_C;
SATURN binds: measured δ̂_C with CI; attack fires iff CI excludes 0 in the
confound's direction."*

Predictor and measurer were deliberately different agents, and MARS's δ_C was
committed to a file before any measurement that could settle it.

## The confound

**The label's variance law.** Every reference line the grid is scored against —
the 1-hop ceiling `sqrt((t*-1)/t*)` and the calibration band `2/sqrt(t*)` — is
derivable only from `Var(y) = t*`. The round record, the producer, the pricing
tool, the README and a guarding test's docstring all state `N(0, t*+1)`.

**The mechanism MARS gives.** The operator is nilpotent of index `t*+1`, so the
variance was read off the nilpotency index. But `b[s-1] = 0` — added to pass the
0-step RED gate — kills the `m=0` term, dropping the true count to `t*`. Nobody
decremented the docstring.

## The filed prediction

| | |
|---|---|
| δ₁(t\*) | `mean Var(y) − (t*+1)` = **−1.000**, negative in all three blocks |
| filed CIs | t\*=2 [−1.006, −0.994], t\*=8 [−1.025, −0.975], t\*=32 [−1.098, −0.902] |
| fires iff | sign negative **and** CI excludes 0 **and** \|δ₁ + 1.000\| < 0.05 |
| control A | δ₂ = 1-hop NRMSE minus closed form, filed at 0.000 ± 0.002 |

The `−1` is the mechanism's signature, so a negative δ of any *other* size is
scored a miss — the prediction cannot be satisfied by being directionally right.
Binding is training-free: K=200 draws per block, seeds 20000–20199, disjoint from
the grid's 0–7 and 12345, ~350 MiB.

## Verified independently

`negation_scope.py` states the law **six times, and the split is 5 to 1**:

| form | lines |
|---|---|
| `N(0, t_star + 1)` | **367** |
| `N(0, t*)` | 419, 421, 528, 649, 651 |

So `:367` is a single stale sentence, not half of a genuine ambiguity. MARS filed
it as "two docstrings disagreeing"; the measurement settles the repair direction
without needing the data to arbitrate.

**The arithmetic, checked against the shipped journal.** `t*=2` records
`flipper_dependence = 1.4012436552018552`:

| target | value | gap | against `flipper_tol = 0.05` |
|---|---|---|---|
| 2/√2 | 1.4142135624 | 0.0129699 | passes |
| 2/√3 | 1.1547005384 | **0.2465431** | **4.93× over — ABORT** |

Making the code agree with its own `:367` docstring aborts the `t*=2` block, and
`t*=2` at n=2048 is the **only it.11-compliant LEARNABLE verdict in the round**.

## STRUCK: the severity claim

MARS wrote that
`tests/cameron/test_m3_etasks.py::test_the_chain_flipper_dependence_is_its_closed_form`
**would still pass** through that repair. That is what made the trap silent, and
it is false.

Run as shipped: **5 passed in 8.52 s**, over `t* ∈ {1, 2, 8, 32, 63}`. Its first
assertion is

```python
assert abs(want - 2.0 / math.sqrt(t_star)) < 1e-12
```

a hard-coded `2/sqrt(t*)` at 1e-12 — any change to the closed form fails it
immediately. Its second assertion compares that form against
`cal["flipper_dependence"]`, which `negation_scope.py:1460` computes as
`moved / scale`: **measured from data**, and therefore unmoved when a constant
changes. The repair fails both assertions *and* aborts the sweep at
`bar_verdict` (`negation_scope.py:1566`).

**The trap is loud and self-announcing.** What survives is smaller but real: the
wrong form is written down as correct in three places — `negation_scope.py:367`,
`test_m3_etasks.py:38` and `:278` — so a maintainer has three documented
invitations to attempt the change and discovers the cost only by breaking the
round's central block.

## Self-strike: propagation

The author copied MARS's unverified clause into the docstring of
`tests/loop/test_the_variance_law_is_stated_once.py` — the guard written to bind
the attack — and relayed it as established in the same turn.

Four of MARS's claims had been checked and all four held. The fifth was accepted
on the strength of the first four. **Accuracy is a property of claims, not of
reports**, and it does not transfer between them. Cost to check: one pytest
invocation, 8.5 seconds.

This round strikes inherited citations as a standing rule, and the same hour the
author handed MERCURY's `m3_capability.py:271` hop-budget citation to the
Inspector *precisely because* it was load-bearing and unread. The rule was
applied to the claim that was convenient to doubt and skipped for the report that
was impressive. Catalogued as instance 19.

## The binding, shipped RED

`tests/loop/test_the_variance_law_is_stated_once.py` — **1 RED, 3 passing**:

- RED: the law is stated both ways (the defect).
- Two must-fires proving the reader can see **both** forms; a pattern blind to one
  would report unanimity and pass by vacuity.
- One passing test making the trap executable, which fails if the two laws ever
  become indistinguishable at `flipper_tol` — i.e. it tells the reader when this
  guard has stopped guarding anything.

**Route**, neither edit moving a number: correct `:367` to `N(0, t*)` with the
reason (`b[s-1] = 0` kills the `m=0` term, so the nilpotency index overcounts the
drivers by one), and give `test_the_chain_flipper_dependence_is_its_closed_form`
a docstring deriving the form it actually asserts.

## The binding (SATURN): WITHDRAWN, mechanism CONFIRMED

600/600 draws completed, 0 errored, 73 s, one process, no training.

| t\* | mean Var(y) | δ₁ | 95% CI | clause 3 |
|---|---|---|---|---|
| 2  | 2.002719  | −0.997281 | [−1.003067, −0.991496] | 0.002719 ✓ |
| 8  | 7.984942  | −1.015058 | [−1.041212, −0.988903] | 0.015058 ✓ |
| 32 | 32.069818 | −0.930182 | [−1.024814, −0.835550] | **0.069818 ✗** |

**Mechanical verdict: WITHDRAWN.** Clause 3 fails at `t*=32`, and MARS's own
section 4 withdraws the attack if it fails in *any* block. No partial credit, no
"fires at two of three".

**The science is nonetheless settled the way MARS said.** At `t*=32` the CI on
mean Var(y) covers `t*` and **excludes `t*+1` by 9.8 half-widths**. `Var(y) = t*`
across all three blocks. So the five `t*` prose sites are correct and `:367` is
the site that must move — no longer a matter of which docstring to trust.

**Why the attack died.** MARS filed a `t*=32` CI half-width of ±0.098 and a
clause-3 acceptance threshold of 0.05 — **narrower than his own stated
uncertainty**. P(clause 3 passes | mechanism exactly true) = **0.6996**. The
filing carried a 30% chance of self-withdrawing while correct and landed there.
A pre-registered rule whose acceptance region is tighter than the filer's own
interval is a defect in the filing, not in the finding.

### The falsifiability arm is what makes −1.000 believable

Restoring `b[:, s-1]` to a fresh N(0,1) on the **same draw** — the pre-fix
builder, `Var(y) = t*+1` by construction — through identical var/subtract/CI
lines gives δ₁ = **+0.006387 / −0.003499 / +0.093519**, every CI covering 0.
Paired arm difference: **+1.003669 / +1.011559 / +1.023701**. The same code path
reports 0 where 0 is true and −1 where −1 is true, so the −1 is not a subtraction
bug.

**Control A did not fire**: δ₂ = +0.000076 / −0.000320 / −0.000112, all CIs
covering 0 and inside ±0.0012 against the filed 0.002. F-2 and F-3 of
`R10_ITERATION_08_09.md` survive. **Control B holds**: across-draw sd 0.001306
against the filed 0.00129, half-width 0.002561 against the 0.027628 margin.

### The trap, executed rather than argued

`bar_verdict` run on the shipped `t:bar` row of
`results/r10_it8_capacity_softmax_t2.jsonl` returns `(True, 'BAR CALIBRATED')` at
`fd = 2/sqrt(2)` and `(False, '…1.401244 is not the value this task predicts in
closed form, 1.154701 (tol 0.05)')` at `fd = 2/sqrt(3)`.
`r10_capacity_sweep.main` then aborts at `:157`, **before** the eval corpus is
drawn at `:159`. `flipper_tol = 0.05` is not overridden at `:153`.

Since the measurement settles `Var(y) = t*`, the repair the trap punishes would
move the **executable** line to the **wrong** law. The six `t*+1` prose sites are
what should move.

### A second self-strike, caught by SATURN

This record and three others said `t*=2` at n=2048 is the *only it.11-compliant
reading* in the round. All three blocks carry N=8 seed CIs at (150, 2048) and are
therefore all compliant; `t*=2` is the only one **excluding 1.0** — the only
compliant **LEARNABLE** verdict. Compliance was conflated with the verdict, which
made the round read as one measurement from collapse when three compliant
readings exist and only the direction of one is at stake. Corrected in all four
files.

## Open

- SATURN's δ₁ binding, with the requirement that it demonstrate its harness can
  report δ₁ ≈ 0 on a synthetic case where the variance genuinely is `t*+1` —
  otherwise a measured −1.000 is indistinguishable from a harness that subtracts
  one somewhere.
- Control A firing instead of the attack would make F-2 and F-3 of
  `R10_ITERATION_08_09.md` the casualty, and attack #1 is then scored a miss
  rather than reinterpreted.
