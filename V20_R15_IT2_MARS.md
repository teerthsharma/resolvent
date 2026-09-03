# MARS — CEQ v20 ROUND 15, PHASE A, it.2

MORIARTY, pulled forward a second time because the wing list is being merged by
primitive this iteration. **Five strikes filed**, each with a RED test that was
RED against the tree as it stands. P-12 is refiled with the Inspector's
correction applied and is now BOUND. The it.1 unpulled thread is pulled and it
resolves into a single fact. Attacks that did not fire are at the foot.

Tests: `tests/mars_v20/`. **16 RED / 14 GREEN** across the directory — 10 RED and
10 GREEN new this iteration, the rest standing from it.1. No git write. Nothing
touched Kaggle.

**MOONS: 3 of 4 used.** One raw journal dump, one raw `STRUCK.md`/`MISTAKES.md`
fetch, one poller for the it.2 planet reports. `V20_R15_IT2_SATURN.md` landed at
11 min and is attacked in TASK B. **`V20_R15_IT2_JUPITER.md` landed at 14 min and
is attacked in the ADDENDUM at the foot of this file** — that section supersedes
the gap this header originally recorded.

---

## TASK A — P-12: BOUND, with the wording corrected

The Inspector struck P-12 UNBOUND (C19) and added one correction of fact:
`STRUCK.md` carries **no evidence *command*, only prose describing a search**.
He is right and the it.1 wording was wrong. The corrected proposition names
something that is in the file:

> `STRUCK.md`'s row for `0.743864` records an **absence proof in prose** — "`git
> log -S` across ALL refs returns zero commits containing a definition of any of
> them" — and performing that search as described now returns **hits**.

**The verdict on `0.743864` is untouched and is not in question. It stays
struck.** What is defective is the reproducibility of the recorded proof.

### The RED, verbatim `[RUN]`

`tests/mars_v20/test_p12_absence_proof_falsified_by_recording_it.py`

```
E  AssertionError: STRUCK.md records 'git log -S across ALL refs returns zero
   commits containing a definition' of absorbing_boundary_kernel; run as
   described it returns 5 commits: 9ce3048 Keep the regenerated code snapshot
   out of the tree it snapshots; 45a715d Keep the regenerated code snapshot out
   of the tree it snapshots; 208cf69 Bridge the local and Kaggle stacks, and
   refuse a GPU torch cannot use; 27038a3 Bridge the local and Kaggle stacks,
   and refuse a GPU torch cannot use; 7abb325 Strike the U1/N3
   harmonic-attribution clause: no producer has ever existed for it

E  AssertionError: the absence proof's own recording sits inside the absence
   proof's search reach: ['.superpowers/sdd/polymorphic-drifting-squirrel/
   progress.md', '.../saturn-report.md', 'PREREGISTRATION_HOLE_AUDIT.md',
   'STRUCK.md', 'tests/cameron/test_harmonic_attribution.py',
   'tests/deimos/DEIMOS_REPORT.md', 'tests/deimos/test_deimos_r9_iteration1.py',
   'tests/loop/test_no_struck_constant_ships.py']

E  AssertionError: STRUCK.md's 0.743864 row publishes the bare search with no
   path exclusion and no control symbol

3 failed, 3 passed in 4.04s
```

**`7abb325` is the strike commit itself.** The commit that recorded the absence
is one of the five commits the absence search now returns. Eight files are in
reach, including `STRUCK.md` and the registry
`tests/loop/test_no_struck_constant_ships.py` that `STRUCK.md` names as the
single source of truth.

### Proving the search works — V-7's own rule, applied to this file

`MISTAKES.md:131` is the standard: "Zero results is a claim about the search
before it is a claim about the world, and it must be paid for with a positive
the search is required to find." Two planted positives, both GREEN, both in the
**same invocation style**:

| control | invocation | result |
|---|---|---|
| bare form finds a real definition | `git log -S"def path_product(" --all --oneline` | **5**, non-zero |
| **repaired** form keeps its reach | `git log -S"def path_product(" --all --oneline -- "*.py" ":!tests/"` | **non-zero** |

The second is the one that matters: a repair that silenced the search would be
V-7 introduced by the fix for P-12. It does not.

### The class

**P-12 — an absence proof falsified by the act of recording it.** The highest
existing P-number is **P-11** (`MISTAKES.md:1597`), so P-12 is free. It is the
**mirror of V-7** (`MISTAKES.md:117`): V-7's search was structurally incapable
of finding and reported zero; this search **was capable, was correct when run,
and publishing it destroyed its own reproducibility.**

**REPLACEMENT ROUTE — REPRICE, one line.** An absence command written into a
document ships (a) the path exclusion that keeps the document out of its own
result and (b) a control symbol the same invocation must still find. Concretely
for this row, and it is a one-line edit to `scripts/render_struck.py`'s template:

```
git log -S"def absorbing_boundary_kernel(" --all --oneline -- "*.py" ":!tests/"  -> 0
git log -S"def path_product("             --all --oneline -- "*.py" ":!tests/"  -> 5  (control)
```

Both verified. **No leap is needed; this is not routed to the leap gate.**

---

## TASK C — the unpulled thread, pulled: THE ESCAPE AND THE CONTROL ARE ONE FACT

`tests/mars_v20/test_seed2_escape_and_the_control_are_one_fact.py` — 3 RED,
3 GREEN.

### The one fact

`arm_smprime:t2:n2048:seed2` is the **unique** cell whose learned magnitude head
never reaches **either** endpoint of its own closed cap `[0,1]`.

```
a_hat_min per cell: seed0 0.0  seed1 0.0  seed2 0.340760201215744  seed3 0.0
                    seed4 0.0  seed5 0.0  seed6 0.0  seed7 0.0
m_max     per cell: 1.0, 1.0, 0.5400443077087402, 0.8454325795173645,
                    1.0, 1.0, 1.0, 1.0
```

The mechanism, asserted as a **biconditional and GREEN 8/8**:
`(a_hat_min == 0.0)` **iff** `(n_zero_gates > 0)`. A gate that reaches the cap's
lower endpoint annihilates; one that does not, cannot.

**That single fact produces all three it.1 observations:**

| observation | why the one fact produces it |
|---|---|
| escapes MARS STRIKE 1 | with no annihilated gate, `frac_gate_annihilated` **cannot** equal a corpus sign count — it is `0.0` |
| carries MARS STRIKE 2(b) | `product` and `exp_scan` differ **only** where some `m_k == 0`; there are none, so the planted negative's region is empty |
| is the sole reason the it.1 control is GREEN (Inspector C12) | the `n_zero_gates` escape value `0` is contributed by this cell and no other |

**Answer to the question as asked: YES, they are one fact.**

### The consequence, and it is the strike

**RED `[RUN]`**

```
E  AssertionError: with arm_smprime:t2:n2048:seed2 removed, every remaining
   n_zero_gates is a corpus sign count: seen=[4069, 4123] corpus=[4069, 4123].
   The it.1 GREEN and the it.1 RED rest on the same cell.
```

The it.1 control
`test_the_manifest_identity_field_n_zero_gates_carries_more_than_the_corpus` is
**not independent evidence**. Its entire margin is the datum its own strike
exempts. Delete the exempted cell and the identity field is **100% corpus**. The
Inspector called the margin "one element wide"; it is one element wide **by
construction**, not by luck — the escape value exists *because* the cell is
anomalous, and the cell is anomalous *because* it has no annihilating gate.

**REPLACEMENT ROUTE — RETIRE the control, REROUTE the question.** A control that
can only pass on the exception is retired, not repaired. The question it was
asked to answer — *does `SMP_FIELDS` carry arm information beyond the corpus?* —
reroutes onto fields that are not gate-derived: `beta`, `qk`, `g` take eight
distinct trained values apiece and are already in `manifest.smp_values`. That is
a live discriminator and it needs no run.

### Thread 4, pulled: ARM PL SHIPS NO CAP

**RED `[RUN]`**

```
E  AssertionError: ARM PL a_hat_max partitions eval_nrmse with a gap:
   unbounded-gate cells [seed2, seed3, seed7] all score >= 1.113339;
   bounded-gate cells [seed0, seed1, seed4, seed5, seed6] all score <= 0.662128.
   dyn_range_bound per cell: {all eight: inf}.
   a_hat_max: {seed0 1.4104527, seed1 1.2868506, seed2 12.7675161,
               seed3 49.6605225, seed4 1.4536346, seed5 1.5051768,
               seed6 1.1029453, seed7 116.0060730}
```

`manifest.pl_values.dyn_range_bound` reads **`inf` on all eight `arm_pl` cells**,
against `arm_smprime`'s `m_max <= 1.0`. `a_hat_max` then partitions the
tournament **with zero overlap on either side** — 8.5x gap in the gate, 1.68x in
the score. The `arm_pl` identity field `a_max` **equals** `a_hat_max` on every
cell: it is the observed maximum, not a setting, so the identity hash cannot
distinguish a converged run from a diverged one — **it is the divergence.**

**This is the sharpest measured distinction between W1 and W3 found this round,
and it is a CAP, not a formula.** GREEN control: `arm_pl`'s `a_hat_min` is
strictly positive on all eight cells (`0.0318`–`0.3408`), so `a_hat_min == 0.0`
on `arm_smprime` is measured, not a journal default.

---

## TASK B — ATTACK ON `V20_R15_IT2_SATURN.md`, AS IT LANDED

`tests/mars_v20/test_saturn_it2_distinctness_gate_and_cost.py` — 2 RED, 2 GREEN.

**Credit first, because it bears on the strike's weight.** SATURN discloses both
defects below in his own `Limits` paragraph, and he declines to claim W2/W3 —
which is the ParaFormer kill correctly refused. The strike is that the disclosed
limits are **load-bearing enough to void the headlines they sit under**, and
they are quantified here for the first time.

### STRIKE 5 — A.6's distinctness is measured on a gate W1 cannot occupy

**RED `[RUN]`**

```
E  AssertionError: W1 distinctness is measured with u ~ U(0.0318116, 116.006073)
   then clamped to [0,1]: only 0.83% of draws land below the cap, so ~99.17% of
   the gate is exactly 1.0. The trained W1 gate spans [0.0, 1.0] and reaches the
   annihilating endpoint 0.0 on 7/8 cells, which u never attains since
   lo=0.0318116 > 0.
```

`tests/saturn/test_v20_r15_wings_distinct.py:20` and `:179` build W1's gate from
**W3's** journalled range. `ceq/arm_smprime.py:109` clamps to `[0,1]`. So:

- **99.17% of the gate is exactly `1.0`** — the identity gate. That is MARS's
  own it.1 corner in the `m` dial, with only `beta`/`qk`/`g` varying. The
  `>= 0.30` separation is attributable to the **switches on a saturated gate**,
  not to the gate content the wings differ in.
- `lo = 0.0318116 > 0`, so the draw **never reaches `m = 0`** — the exact
  endpoint 7 of 8 trained W1 cells do reach, and the endpoint that carries
  `no_prefix_scan_represents_a_zero_gate`, the one thing W1 provably has that W2
  provably does not.

**A distinctness measurement that excludes the endpoint the wings differ at is
the mirror of the merge that hides a distinction.** It happens to report
separation; it reports it for the wrong reason.

**REROUTE, and nothing new needs training.** The gate is not unjournalled in the
way `Limits` implies: `a_hat_min` and `a_hat_max` are journalled **per W1 cell**,
and `n_zero_gates` gives the mass at the endpoint. Draw `u` per cell on **that
cell's own** `[a_hat_min, a_hat_max]` with an atom at `0.0` of mass
`n_zero_gates/8192`. Every input is read from the record.

### STRIKE 6 — criterion (3)'s `25.0x` is a row-order artifact

**RED `[RUN]`**

```
E  AssertionError: seconds-to-floor in FILE order {'W3': 1.884, 'W1': 47.048}
   ranks W3 ahead of W1 by 25.0x; the identical statistic with seeds visited
   best-first is {'W3': 1.759, 'W1': 14.852}, a ratio of 8.4x. Per-cell cost
   barely varies ({'W3 secs': (1.726, 1.916), 'W1 secs': (14.852, 16.66)}), so
   the figure is (index of the first crossing seed) x (per-cell cost).
   W3 crosses on its FIRST journal row and W1 on its THIRD.
```

`47.048 = 16.164 + 16.032 + 14.852` — W1's first three rows. `1.884` is W3's
first row alone. Per-cell cost is near constant inside each arm, so the statistic
is `(index of the first crossing seed) x (per-cell cost)`. **The ranking
direction survives; the magnitude does not** — `25.0x` becomes `8.4x` under a
permutation of the same cells.

**REPRICE.** The order-invariant form is expected cost to a crossing:
`per-crossing-cell cost x (cells run / cells crossed)` — W3 `1.78 x 8/5 = 2.85`
GPU-s, W1 `14.85 x 8/1 = 118.8` GPU-s. Also flagged: **W3 fails to cross on 3/8
seeds and W1 on 7/8**, so a first-crossing statistic is survivorship on both
arms, and the three W3 seeds that never cross are exactly the three
unbounded-gate cells of STRIKE 4.

---

## ATTACKS THAT DID NOT FIRE

**1. "The skyline is smuggled in as a contender."** NOT FOUND. The deciding
journal's header does read `"arms": ["arm_pl", "arm_smprime", "softmax"]` and
eight `softmax` cells exist (`eval_nrmse` `0.9388`–`0.9734`, median `0.948612`),
so the pressure is real. But SATURN's criterion (3) excludes `softmax` — it
crosses on **0 of 8** seeds and so has no computable value — and nowhere in
`V20_R15_IT2_SATURN.md` is it counted toward N. **N is discussed as W1/W2/W3
throughout.** No strike.

**2. "A multiplier promoted to an arm."** NOT FOUND in SATURN's it.2. No
interventional channel, no curriculum, no method-of-training candidate appears as
a wing; the wing list is unchanged at W1/W2/W3.

**3. "The merge invents a distinction — W2 and W3 are one code path."** NOT
STRUCK, because **SATURN struck it himself first** and correctly refused to claim
otherwise: `|W2| = W3` identically, the whole separation rides on `theta`,
`theta` reads bitwise `0.0` at the corner, and `arm_phase` has **no trained cell
anywhere in `results/`**. His A.7 is the ParaFormer kill applied to his own
merge. Nothing for an adversary to add.

**4. "The journal's DISTANCE line is false."** OPEN, NOT FILED.
`V20_R15_JOURNAL.md:220` reads "No arm crosses BED-M's floor₁ `0.7071`", and six
cells of `results/v17k_r4_retake.jsonl` are below `0.7071067811865476` with the
journal's own `dist_to_floor` negative on each. SATURN's §B already resolves the
apparent contradiction — the cited `V15_R1.md:250` is a different (v15 CPU)
round — so this is a **carried-forward citation in the journal's own summary
line, not a false measurement.** It is named here rather than struck: no RED, so
no strike.

---

## WHAT THIS NODE COULD NOT VALIDATE

1. **`V20_R15_IT2_JUPITER.md` arrived at 14 minutes and got six.** It IS attacked
   (ADDENDUM, STRIKE 7), but on **§A alone**. His §D (the two struck it.1 claims,
   bound) and §E (the Wilson reconciliation) were not read, let alone tested.
   Absence of a strike against those sections is absence of search, not evidence
   of a clean merge (`MISTAKES.md` V-7, applied to this report).

2. **STRIKE 3's mechanism is measured; its *cause* is not.** That seed 2's
   `m_head` never approaches either endpoint is read off the journal 8/8. *Why*
   that seed alone trained to an interior gate is not established — it would need
   the trained weights, which are not in the tree, and a run this node did not
   perform (L-LEAN).

3. **STRIKE 5's `99.17%` is exact arithmetic on the disclosed range, but the
   downstream effect is inferred.** That a gate saturated at `1.0` on 99% of
   entries *weakens* the reported separation follows from the operator algebra
   filed at it.1; it was **not** re-measured by re-running SATURN's node with the
   corrected per-cell draw. That re-run is the one measurement that would settle
   whether `>= 0.30` survives, and it is cheap — every input is journalled.

4. **Nothing was run on CUDA.** Every number is float64 or journal arithmetic on
   this box's CPU.

5. **Three moons, not four, and one of the three returned only a poll receipt on
   its first pass.** The other ~60 result files and the 129 KB `AUDIT.md` were not
   swept this iteration either.

---

## ADDENDUM — `V20_R15_IT2_JUPITER.md` landed at 14 min and IS attacked

Supersedes the "not attacked" line at the head of this file and item 1 of
`WHAT THIS NODE COULD NOT VALIDATE`.
`tests/mars_v20/test_jupiter_it2_merge_hides_the_upper_cap.py` — 2 RED, 2 GREEN.

### STRIKE 7 — the merge that hides a distinction

JUPITER's A.1/A.4 merges W1 and W3 into **one primitive** on
`cumprod = exp . cumsum . log` with `g := log m`, and names the isomorphism's
domain exclusion as **exactly one point**: "The isomorphism has a domain:
`0 ∉ R_{>0}`" — the **lower** endpoint.

**The algebra is correct and is not what is struck.** GREEN control: on the
shared image `m ∈ (0,1]`, `exp(cumsum(log m)) == cumprod(m)` to `< 1e-12`.

**The domain claim is what is struck. RED `[RUN]`**

```
E  AssertionError: W1 is clamp(u,0,1) so its image under g=log m is g<=0.
   W3 exceeds the cap on 8/8 trained cells (g=log a_hat_max > 0):
   {seed0 0.3439, seed1 0.2522, seed2 2.5469, seed3 3.9052,
    seed4 0.3741, seed5 0.4089, seed6 0.0980, seed7 4.7536}.
   V20_R15_IT2_JUPITER.md names only the lower exclusion '0 not in R_{>0}'
   and does not name the upper cap, which is the boundary the trained record
   actually separates on.

E  AssertionError: the cap boundary JUPITER's merge does not name partitions
   W3's own eval_nrmse with zero overlap: 5 cells at a_hat_max<=2.0 score
   <= 0.662128; 3 cells above score >= 1.113339.
```

There is a **second** domain difference. W1's magnitude is `clamp(u, 0, 1)`
(`ceq/arm_smprime.py:109`), so W1's image under the isomorphism is `g ≤ 0`. W3
ships `dyn_range_bound = inf` and its trained `a_hat_max` exceeds `1.0` on **8 of
8** cells — `g > 0` on every one, up to `g = 4.7536`. **The two wings are not in
a common image on any trained cell.** A merge is entitled to an isomorphism only
where both sides are in its image.

And the unnamed boundary is the one that **sorts the scoreboard** — the STRIKE 4
partition, zero overlap. A boundary that partitions `eval_nrmse` is not a
coordinate change.

**REROUTE — the verdict is not refuted, it is UNPRICED.** `N = 1` is defensible
restated as: *one primitive on the common image `g ≤ 0`, and the record contains
no trained W3 cell inside that image.* The operative criterion between W1 and W3
is then the **cap**, which is decidable at **zero training cost** from the
journalled `a_hat_max`, not by the reduction. This is the same object as
STRIKE 4 and STRIKE 5, reached from a third direction.

**Not struck, and worth saying:** JUPITER's §B lists softmax as a floor and not a
contender (`1.497 s / 150 steps`), and his §C refuses to list curricula as a
takeable multiplier for want of a generator. Both forgery pressures this node was
sent to find are, in his file, already refused.
