# JUPITER (MYCROFT) — CEQ v20 ROUND 15, it.4 — THE FREEZE

Three it.2 claims were struck (C4, C7, C9). All three are answered here by test,
not by argument. Zero moons: every input was journalled and the wall clock was
better spent on the deletion probe than on fan-out.

**NEW** `tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py` — 1 RED by
design, 5 GREEN.
**EDITED** `tests/jupiter/test_v20_r15_it2_ldom_census.py` — the C7 planted
negative now invokes the real node; two dominated assertions deleted from that
node with the derivation.

`[RUN]` `python -m pytest tests/jupiter/ -q` — **1 failed, 107 passed in 46.41s**
(it.2 read 101 passed). The one failure is this file's RED-by-design node.

---

## THE VERDICT, IN ONE LINE

**W1 and W3 are one primitive and two arena entries. `N = 2` measured.** The
isomorphism is conceded and it is **unexercised**: on the trained record no W3
cell lies in W1's image, and on the half of the gate mass where W1 is defined W3
is `nan`. A map with an empty exercised set cannot retire an entry the
scoreboard already sorts on.

---

## TASK A — `N = 1`, RESTATED ON MARS's REROUTE, AND BOUND

### A.1 The RED, first, verbatim

The node asserts the **it.2 verdict itself** — that the merge covers the trained
record — and nothing else. It runs against the tree as it stands and fires.

```
[RUN] python -m pytest tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py -q

E  AssertionError: the it.2 merge verdict claims a coordinate change between W1
   and W3, but W1's image under g=log m is g<=0 (clamp(u,0,1),
   ceq/arm_smprime.py:113) and 0 of 8 trained W3 cells have a_hat_max <= 1.0;
   g = log a_hat_max per seed: {0: 0.3439, 1: 0.2522, 2: 2.5469, 3: 3.9052,
   4: 0.3741, 5: 0.4089, 6: 0.098, 7: 4.7536}
E  assert 0 >= 1
E   +  where 0 = len([])

tests\jupiter\test_v20_r15_it4_merge_is_unexercised.py:94: AssertionError
1 failed, 5 passed in 2.02s
```

Logged `t:"test"`, `status:"red"` before any finding event this iteration.
**It is kept RED**, as the record of the strike against its own author — the
same discipline SATURN applied to his struck `0.30` floor.

The algebra is not touched. `exp ∘ cumsum ∘ log = cumprod` on `m ∈ (0,1]` is
**conceded**, on MARS's control, which the Inspector probed at `1 + 1e-6` and
made fail at `8.999999999703689e-07` against a `1e-12` bar. This file asserts the
**verdict**.

### A.2 The restated verdict, GREEN — and one number MARS did not have

MARS proved W3 leaves W1's image. He did not read W1's own side of the same
field. `[READ]` `results/v17k_r4_retake.jsonl`, `a_hat_max` at top level, the
**same field for both wings**, so the two are on one scale:

| | seed 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **W1** `arm_smprime` `a_hat_max` | 1.0 | 1.0 | 0.5400443 | 0.8454326 | 1.0 | 1.0 | 1.0 | 1.0 |
| **W3** `arm_pl` `a_hat_max` | 1.4104527 | 1.2868506 | 12.7675161 | 49.6605225 | 1.4536346 | 1.5051768 | 1.1029453 | 116.0060730 |

`test_the_restated_verdict_the_common_image_holds_no_trained_w3_cell` GREEN:
W1 is inside the common image `g ≤ 0` on **8 of 8** (max exactly `1.0`, attained
on six cells — the clamp is not merely a bound, it is *reached*); W3 is outside
on **8 of 8** (min `1.102945`). The intersection of the two in-image seed sets
is **empty**.

So the reroute is not a weakened form of the it.2 claim. It is exact:
**one primitive on the common image `g ≤ 0`, and the record contains no trained
W3 cell inside that image.** The `1.0` boundary is not a tolerance — it is the
clamp endpoint of `ceq/arm_smprime.py:113`, where `g = log m` changes sign, and
the nearest cell from above is `0.103` away, so no float bar is load-bearing.

`test_the_unnamed_boundary_sorts_the_scoreboard_with_zero_overlap` GREEN
re-derives MARS's partition at `a_hat_max = 2.0`: 5 cells score `≤ 0.662128`,
3 score `≥ 1.113339`, zero overlap. (`2.0` is the partitioning threshold; `1.0`
is the cap and partitions nothing, since every W3 cell exceeds it — the
Inspector's precision note, carried into the constant's comment.)

`test_planted_negative_the_image_predicate_returns_nonzero_when_a_cell_moves`
GREEN: the mutation is named and exact — move the boundary-nearest W3 cell
(seed 6, `1.102945`) to `0.9` and the **same** `in_common_image` predicate the
verdict uses returns `1`, not `0`. The zero is a measurement.

### A.3 The question the freeze turns on

> If two wings are one primitive but no trained cell of either lies in the
> other's image, are they one arena entry or two?

**Two.** As a consequence of a stated principle, not a preference:

> **The exercised-map principle.** An arena entry is a thing the record can be
> scored on. A reduction retires an entry only if the record **exercises** the
> map: some trained cell of each side must lie in the domain where the map is
> defined and where both sides are actually evaluated. A map whose exercised set
> is empty is a claim about a region the record never visits, and a claim about
> an unvisited region cannot retire an entry the scoreboard sorts on.

This is the round's own binding kill, applied in the other direction. *A wing
named rather than found is struck* refuses an entry the record does not produce.
The same rule refuses a **merge** the record does not exercise. In both
directions the record decides and the algebra does not. Granting the merge here
would be the exact mirror failure the brief names — erasing a distinction that
already sorts `eval_nrmse` with zero overlap, on the strength of an identity
that holds only where neither wing was trained.

The principle is falsifiable and cheap to falsify: **one** trained `arm_pl` cell
with `a_hat_max ≤ 1.0` collapses W1 and W3 to one entry, and the RED node above
turns GREEN the moment such a cell lands. It costs no new instrument.

**Wing count: `N = 2` measured (W1 `arm_smprime`, W3 `arm_pl`).** W2 `arm_phase`
is **UNPRODUCED**, not a wing — SATURN's it.3 C.4 retire route, which this
verdict does not contest and now independently supports: W2's magnitude is W3's
identically, so W2 sits on W3's side of the same boundary and adds no entry.

---

## TASK B — THE TWO INSTRUMENT DEFECTS

### B.1 C7 — the planted negative that did not invoke its node

**Root cause, and it is deeper than non-invocation.** The Inspector's probe was
*delete the upper-side assert and confirm the planted negative now fails*. That
probe is **unreachable by construction** on this node, and the reason is
arithmetic, not bookkeeping.

`[DERIVED]` The node carried three assertions. With `exact = 1/27 = 0.037037…`:

- one-sided lines: `|crude − exact| ≤ TOL = 1e-11`;
- ratio line: `|crude/exact − 1| ≤ TOL` ⟺ `|crude − exact| ≤ TOL·exact = 3.70e-13`.

The ratio bound is **27× tighter on both sides**. No mutation exists that a
one-sided line catches and the ratio does not, so deleting either one-sided line
removes nothing the node still asserts — and no planted negative, however
written, could have detected that deletion.

**The repair, two parts.**

1. The two dominated lines are **deleted** from
   `test_the_repo_crude_phi_EQUALS_the_exact_minimum_two_sided`. What survives is
   one line, two-sided and strictly stronger than the pair it replaces, with the
   derivation in the docstring. C20's kill (one-sidedness admits `crude = 10·exact`)
   is satisfied more tightly than before, not relaxed.
2. The planted negative **calls the real node.** It monkeypatches the
   module-global `_crude_and_exact` the node itself calls and then invokes
   `test_the_repo_crude_phi_EQUALS_the_exact_minimum_two_sided()`. Nothing is
   re-declared. Parametrised over two named mutations: `crude × 10.0` (C20's own
   counterexample) and `crude + TOL/2` — the second clears the *deleted*
   upper-side line and still fires the survivor, which is the demonstration that
   the survivor is what does the work.

**Verified by the Inspector's own test.** `[RUN]` the surviving assertion deleted
from the real node:

```
>       with pytest.raises(AssertionError):
E       Failed: DID NOT RAISE <class 'AssertionError'>

FAILED ...::test_planted_negative_the_two_sided_node_fires_when_crude_is_inflated[crude_x10-10.0-0.0]
FAILED ...::test_planted_negative_the_two_sided_node_fires_when_crude_is_inflated[crude_plus_half_tol-1.0-5e-12]
2 failed, 8 passed in 2.93s
```

Restored and re-run: **10 passed in 3.07s**. Source md5
`4f0059ee9e2e2828e46ab56962d9be12`, identical before the probe and after it.
The file now holds **10** nodes (9 at it.2; the planted negative is parametrised
into two).

### B.2 C4 — the RED count, retracted rather than restated

**No count is published, because the tool emits none.** A module-level
`ImportError` is a **collection** error: pytest reports `Interrupted: 1 error
during collection` and `1 error in 0.59s`, zero tests run, **no per-node result
of any kind**. `8 nodes uncollectable` in the it.2 log and `all 9 RED` in the
it.2 report were both descriptions of a thing the tool did not report. Both are
withdrawn. The file held 9 test functions; that is a `[READ]` of the source, and
it is not a RED count.

The correction generalises, and it is the mechanism behind all three of this
node's it.2 strikes: **a collection RED has no per-node count, and describing one
is describing the instrument rather than reading it.**

it.4's RED is not of that class. It is a per-node assertion failure, so its count
*is* tool-emitted and is published verbatim: `1 failed, 5 passed` in the file,
`1 failed, 107 passed` over `tests/jupiter/`.

---

## TASK C — THE ATOM, PRICED

`test_at_the_atom_w1_is_defined_and_w3_is_not` GREEN and
`test_the_atom_is_not_a_measure_zero_corner_it_carries_half_the_gate_mass` GREEN.

`[READ]` per W1 cell, `manifest.smp_values.n_zero_gates` and
`frac_gate_annihilated`, with `n_zero_gates / 8192 == frac_gate_annihilated`
asserted **bitwise on all eight** (the denominator is measured, not assumed):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `n_zero_gates` | 4123 | 4123 | **0** | 4069 | 4123 | 4123 | 4123 | 4123 |
| atom mass | .5032959 | .5032959 | **0** | .4967041 | .5032959 | .5032959 | .5032959 | .5032959 |

**Correction of fact to SATURN's it.3, not a strike.** The published figure is
"50.3% of the gate mass on seven of eight cells". Seven cells carry a nonzero
atom, but only **six** are at `0.5032958984375`; seed 3 is `0.4967041015625`.
The mean over the seven is `0.5023542131696429`. The claim's substance — half the
gate mass, seven of eight cells — is exactly right; the single figure covers six
of the seven, not seven. The GREEN node asserts the mean and the `> 0.49` floor
rather than the modal figure.

### C.1 What the atom does to the merge

`m = 0` is the **only** exclusion the it.2 verdict named — `0 ∉ ℝ_{>0}` — and it
named it as a point. It is not a point on this record. On 7 of 8 trained W1 cells
it carries **~50% of the gate mass**, and at it W1's `cumprod` returns a true `0`
while W3 routes `log 0 = −∞` into a softmax logit and returns `nan`.

An isomorphism that fails on half the gate mass is not a coordinate change on that
half. Say it precisely: **it is a domain difference, and a domain difference is a
distinction between the objects, not between descriptions of one object.** A
coordinate change is a bijection between two representations of the same value.
Where one side has a value and the other has no value, there is nothing to be in
bijection with — `nan` is not a coordinate of `0`. The set where the map is
defined is therefore not `ℝ_{>0}` minus a null set; it is the complement of a set
carrying half the measure the record actually draws from.

### C.2 The wing count on each side of the atom

| region | mass | W1 | W3 | count | why |
|---|---|---|---|---|---|
| **at the atom**, `m = 0` | ~50.2% on 7/8 cells | finite | `nan` | **1** — W1 alone | one side has no value; there is no map to write |
| **off the atom**, `m ∈ (0,1]` | the rest | defined | defined | **2** | the map exists and is exact, but **no trained W3 cell is in this region** — exercised set empty |

The two halves fail the merge for two different reasons and they do not overlap.
Where the map exists, the record never puts both wings in it; where the record
puts W1, the map does not exist. **On no part of the draw does the merge both
apply and have two trained things to merge.** That is the whole of the answer to
Task A, arrived at from the gate instead of from the cap — the fourth direction
onto one object, after MARS's STRIKE 4, STRIKE 5 and STRIKE 7.

**Arena count, frozen: `N = 2` measured (W1, W3), W2 UNPRODUCED.**

---

## WITHDRAWN THIS ITERATION

| # | claim | disposition |
|---|---|---|
| C9 | `N = 1 primitive`, W1 ≡ W3 as one arena entry | **RETIRED.** Replaced by A.2/A.3: one primitive, two entries, on the exercised-map principle. The algebra it rested on is conceded and untouched. |
| C4 | the it.2 L-DOM RED took 8 nodes / all 9 | **WITHDRAWN, no replacement number.** A collection RED emits no per-node count. |
| C7 | the two-sided **node** fires at `crude × 10` | **REPAIRED**, by invocation, and verified by the deletion probe that struck it. |
| — | it.2's three-assertion two-sided node | **two lines deleted** as dominated 27:1, with the derivation. Not a weakening. |

## LIMITS

The exercised-map principle is stated and applied here; it is not itself bound by
a test, and no test could bind a criterion. What is bound is its input — the two
`a_hat_max` columns and the empty intersection. The atom node re-derives the
definedness structure on **one** draw (`s=8`, `seed=0`, an exact zero inserted at
index 0), not on all eight per-cell gates; SATURN's it.3 A.5 has the eight-cell
measurement and this file does not duplicate it. Every number is float64 on this
box's CPU; nothing ran on CUDA. The `eval_nrmse` partition is re-derived, not
re-trained — it inherits whatever the retake journal inherits. W2's retirement is
argued from W3's identity with it and from SATURN's it.3, not measured here.
No moons were sent, so no independent read of the journal backs the two
`a_hat_max` columns beyond the Inspector's it.2 reproduction of the W3 half.

**No git write of any kind. Nothing touched Kaggle.** One source file was mutated
for the C7 deletion probe and restored, md5-verified identical.
