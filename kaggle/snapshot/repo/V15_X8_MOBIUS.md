# V15 X8′ — MÖBIUS INVERSION, AGAINST THE REPO'S ESTIMATOR

Filed by JUPITER-5, R11 it.11 (2026-08-31), against `CEQ_V15_1_DELTA.md`
§X₈′. Companion script: `scripts/v15_x8_mobius_probe.py` (run it; output
pasted verbatim in each section below). No existing module was modified;
both output files are new. Nothing trains; float64 throughout.

**Answer up front** (detail follows):

| question | answer |
|---|---|
| k=2 identity holds? | **Yes, exactly.** `max\|diff\| = 0.000e+00` over 8 drawn geometries, both signed and unsigned. |
| k=3 identity holds? | **Only up to a sign.** Direct comparison: `max\|diff\| = 9.127e-09` (real, not noise). After correcting for the `(−1)^k` parity the repo's convention differs by: `max\|diff\| = 3.081e-33` (machine epsilon). |
| k=4 | **No repo estimator exists.** Nothing to compare against; reported for completeness only, plus an internal self-consistency check of the general formula. |
| must-fire | Additive landscape, max \|interaction\| over k∈{2,3,4}, 5 seeds: **1.998401e-15** (machine epsilon — fires cleanly). |
| positive control | Multiplicative landscape, mean \|degree-2 interaction\|: **6.002222e-01**, vs additive's **6.032212e-17**. |
| delta's cited `[RUN: 0.600]` and `[RUN: 0.0375 vs 0.0000 additive]` | **Neither reproduces from any Möbius/Walsh/epistasis code in this repo.** Both trace to unrelated numbers elsewhere in the repo's own history (provenance below). |
| rename plan | 13 sites across 2 code files, 3 test files, 1 doc, plus a dangling Lean-inventory name. **No collision with `scale/r10_capacity_sweep.py` or `scale/identity_manifest.py`** (grepped for `walsh`/`epistasis`/`mobius`/`four-point`/`leakage_ratio`/`twodof` in both — zero matches). |

---

## PART 1 — THE IDENTITY

### 1.0 What was searched, and what was found

Grepped `scale/` and `ceq/` (and, once nothing turned up there, the whole
repo) for `walsh`, `epistasis`, `four-point`/`four_point`,
`two-point`/`two_point`, `mobius`/`möbius`. The only estimator of this shape
anywhere in the codebase is `scale/twodof.py::leakage_ratio` (defined at
`scale/twodof.py:215-248`), which computes exactly the four-point and
eight-point finite differences the delta describes:

```python
inter = f(()) - f((c,)) - f((j,)) + f((c, j))                          # k=2
l3 = (f(()) - f((c,)) - f((j,)) - f((m,))
      + f((c, j)) + f((c, m)) + f((j, m)) - f((c, j, m)))               # k=3
```

`scale/trained_projections.py:215` calls it in production (`TD.leakage_ratio`,
`TD = scale.twodof`), reading `abs_I`/`abs_L3` per example on trained and
untrained attention geometries. `tests/cameron/test_r6_twodof_red.py` and two
standalone reimplementations (`tests/loop/test_two_dof_lemma.py`,
`tests/loop/test_l3_leakage.py`) exercise the same arithmetic under generic
names (`interaction`, `leakage`, `ratio` — never branded "Walsh" or "Möbius"
in code, only in prose). `scale/r10_capacity_sweep.py` and
`scale/identity_manifest.py` were grepped specifically per the task's
collision check: **zero matches** for any of the search terms in either file.
No Lean file defines `walsh_two_point` — see §3.4.

### 1.1 The poset definition, transcribed

Möbius inversion is a theorem on any **locally finite poset** `(P, ≤)`
(Rota, 1964): the zeta function `ζ(S,T) = 1` if `S ≤ T` else `0` is invertible
in the incidence algebra, and its inverse `μ` satisfies

```
μ(S,S) = 1
μ(S,T) = − Σ_{S ≤ U < T} μ(S,U)         for S < T
```

Inversion: if `g(T) = Σ_{S ≤ T} f(S)`, then `f(T) = Σ_{S ≤ T} μ(S,T) g(S)`.

**On the Boolean lattice** `(2^[n], ⊆)` specifically, this recurrence has the
closed form

```
μ(S, T) = (−1)^(|T| − |S|)          for S ⊆ T
```

which turns inversion into ordinary inclusion–exclusion:

```
g(T) = Σ_{S ⊆ T} (−1)^(|T|−|S|) f(S)
```

This is a **specialization**, not the whole theorem: the recurrence holds on
any locally finite poset (chains, divisor lattices, partition lattices,
Rota's own W9/cache-eviction use elsewhere in this repo at `CONTRACT.md:166`,
which is a *different, unrelated* application of the same machinery to
retroactive-eviction repair, not to epistasis). The alternating-sign formula
is what it reduces to specifically on `2^[n]`.

Both derivations are implemented independently in
`scripts/v15_x8_mobius_probe.py`:

* `mobius_boolean(S,T)` — the closed form, `(-1)**(len(T)-len(S))`.
* `mobius_boolean_recursive(S,T)` — Rota's raw recurrence, with **no
  reference** to the closed-form sign rule.

Cross-checked against each other over every `(S,T)` with `S ⊆ T ⊆ {0,1,2,3}`:
**`max |closed_form − recurrence| = 0.000000e+00`** (script §PART 0). This
confirms the closed form used below is not a typo before it is used to judge
the repo's code.

### 1.2 V-3 guard, stated

MISTAKES.md's V-3 is an assertion (`pooled < tail`) that "passed 400/400 and
could not fail" because both sides were restrictions of the same construction
— agreement proved self-consistency, not correctness. The guard applied here:
`mobius_inversion(f, T)` (script) is built from the bare subset-sum
definition above, with **no import from, or reference to, `scale/twodof.py`
in its body**. It is compared against `scale.twodof.leakage_ratio`, imported
and **called unmodified** — not reimplemented, not paraphrased. Where no repo
estimator exists (k=4), that is reported as an absence, not papered over with
a second copy of the same formula standing in for both sides.

### 1.3 k = 2

Geometry: `s=32, d=8, seed=20260826, c=5, j=11`, row `i=31`, `target=3` (the
repo's own defaults, `scale/twodof.py:268-276`). `f(S)` = row `i`'s weight on
`target` after masking exactly `S`, via `scale.twodof.masked_row` — the
repo's own primitive, unmodified — with the `(c,j)` perturbation zeroed.

Swept over 8 drawn geometries (`seed = 20260826..20260833`), per MISTAKES.md's
rule to draw instances rather than hand-build one:

```
example (t=0):
  mobius_inversion(f, {c,j})       = 5.913828454718e-12
  repo signed (twodof.py:245-247)  = 5.913828454718e-12
  repo abs (leakage_ratio, live)   = 5.913828454718e-12

over all 8 geometries:
  max|diff| signed        = 0.000e+00
  max|diff| of abs values = 0.000e+00
```

**k=2: the repo's estimator equals the Möbius inversion exactly, bitwise, on
every geometry drawn.** This is expected once the two formulas are written
side by side — `leakage_ratio`'s `inter` line *is* the k=2 inclusion–exclusion
sum, term for term — but it is now a measured identity rather than a read
formula, per the delta's own instruction not to assert it.

### 1.4 k = 3 — matches only up to a sign, and this is a real finding

Same sweep, `T = {c,j,m}`, `m=17`:

```
example (t=0):
  mobius_inversion(f, {c,j,m})       =  6.245004513517e-17
  repo signed L3 (twodof.py:245-247) = -6.245004513517e-17
  repo abs (leakage_ratio, live)     =  6.245004513517e-17

over all 8 geometries:
  max|diff| DIRECT   (mobius − L3)          = 9.127e-09
  max|diff| AFTER (−1)^k sign correction    = 3.081e-33
  max|diff| of abs values                    = 3.081e-33
```

The direct-diff column is not noise — `9.127e-09` is large relative to the
values involved and stays that way across all 8 geometries with a consistent
sign. **The repo's `L3` and the canonical Möbius inversion of `f` on
`T={c,j,m}` are not the same number: they are negatives of each other.**

The reason is structural, not a bug in either place. Expanding both formulas
by parity of `|S|`:

```
μ(S,T) = (−1)^(|T|−|S|) = (−1)^|T| · (−1)^|S|
```

so `Σ_S μ(S,T) f(S) = (−1)^|T| · Σ_S (−1)^|S| f(S)`. The repo's `inter`/`L3`
lines are literally `Σ_S (−1)^|S| f(S)` (leading term `+f(∅)`, alternating).
For `|T|=2` the prefactor `(−1)^|T| = +1`, so the two coincide — which is
exactly what §1.3 measured. For `|T|=3`, `(−1)^|T| = −1`, so they are
negatives. `leakage_ratio` only ever exposes `abs()` of both quantities
(`scale/twodof.py:248`, `return abs(inter), abs(l3)`), which is why this
sign flip has never been visible from the outside: every existing caller
(`scale/trained_projections.py:215`, `tests/cameron/test_r6_twodof_red.py:175`)
consumes `abs_i`/`abs_l3` and the sign cancels before it is ever read. The
delta's claim "the four-point estimator IS the Möbius inversion" is **exactly
true at k=2 and true only up to an alternating sign at k=3**, and the general
statement needs that qualifier or it overclaims.

### 1.5 k = 4 — no repo estimator to compare against

Grepped `scale/`, `ceq/`, and the whole repo: nothing computes a 16-point
(k=4) finite difference. `T = {c,j,m,n}` with `n=23` (a 4th token, distinct
from `c,j,m`, `target`, and row `i`):

```
max |mobius_inversion(f,T)| over the same 8-geometry sweep = 2.362e-10
```

reported for completeness only — **there is nothing in this repo to check it
against**, and per the task's own instruction this absence is the finding for
k=4, not a comparison. What *is* checked at k=4 is internal: §1.1's
closed-form-vs-recurrence cross-check was run for every `S ⊆ T` with
`|T| ≤ 4`, agreeing exactly, so the general formula used to produce the
`2.362e-10` figure is at least self-consistent under two independent
derivations of `μ`. That is a correctness check on the instrument, not a
validation of the delta's k=4 claim, and it is reported as such — computing
Möbius inversion two ways and comparing them to each other is the same shape
of error V-3 warns against if it were presented as evidence the *repo*
generalizes to k=4. It isn't presented that way here: it establishes that
*this script's* general formula is not internally broken, nothing more.

### 1.6 Provenance of the delta's cited `[RUN: 0.600]`

The delta's X₈′ section states the identity holds "`[RUN: 0.600]`" with no
attached geometry, seed, or method. That figure does not come out of
`scripts/v15_x8_mobius_probe.py`, and grepping the repo for `0.600` turns up
its actual source: `MISTAKES.md:1219-1220` (entry M-13), the two-sample TOST
equivalence-margin half-width at `N=16`: *"at `N=16` it is `0.6001`"* —
confirmed independently at `V13_CLAIM_AUDIT.md:27` and
`V13_DAG_TASKLIST.md:162`. That number is about statistical power for an
equivalence test, unconnected to Möbius inversion, Walsh coefficients, or
`scale/twodof.py`. **This repo has a documented history of exactly this
failure mode** — see the many "rotation" entries in `DONE.md`/`CHECKLIST.md`
where a published number from one measurement gets read against an unrelated
claim — and `[RUN: 0.600]` in the delta appears to be another instance of it,
not a real Möbius-inversion run. The real number for the same comparison this
figure claims to summarize is the k=2 row in §1.3: **`max|diff| = 0.000e+00`**.

---

## PART 2 — MUST-FIRE AND POSITIVE CONTROL

Per the task and `CEQ_V15_CONTRACT.md` PART III's EPISTASIS positive control
and PART II item #9. These use synthetic Boolean-lattice landscapes
(`f: 2^[n] → ℝ`, `n=6`), independent of the attention machinery in §1, because
the contract's claim is about `f` in general, not about attention rows
specifically:

```python
additive:        f(S) = c0 + Σ_{i∈S} c_i
multiplicative:   f(S) = c0 · Π_{i∈S} (1 + c_i)
```

(seeded `torch.Generator`, float64; 5 seeds each). Expanding the k=2 Möbius
inversion of the multiplicative form by hand gives exactly `c0·c_a·c_b` for
`T={a,b}` — generically nonzero, with no additive decomposition — which is
what a "multiplicative landscape has nonzero degree-2 Walsh coefficients"
means concretely.

### 2.1 Must-fire: additive ⇒ 0

```
max |interaction| over k ∈ {2,3,4} and 5 seeds: 1.998401e-15
  k=2  additive mean|I| = 6.032212e-17
  k=3  additive mean|I| = 1.382228e-16
  k=4  additive mean|I| = 2.694141e-16
```

**1.998401e-15 is machine epsilon for float64 accumulation over ~15-30 terms
(`|T| choose·` sums with cancellation); this is exact 0 in the sense the
task asked for, not a small residual effect.** This holds at k=2, 3, *and* 4
— an additive set function has zero Möbius transform at every order ≥ 2, not
only order 2, which the task's must-fire only required at k=2 but the
general theorem gives for free and the sweep confirms.

### 2.2 Positive control: multiplicative ⇒ nonzero degree-2

```
k=2  additive mean|I| = 6.032212e-17    multiplicative mean|I| = 6.002222e-01
```

Clean separation: 15 orders of magnitude between the additive null and the
multiplicative signal, at the same `n`, same seeds, same instrument. This is
the must-fire and the positive control **together**, per the task: the
instrument reads exactly 0 when there is no interaction to see (§2.1) and
reads a large, stable nonzero number when there is (§2.2) — neither alone
would distinguish "correctly blind" from "just broken."

### 2.3 Provenance of the delta's cited `[RUN: 0.0375 vs 0.0000 additive]`

As with §1.6, this figure does not reproduce from any epistasis/Walsh/Möbius
code in the repo. `0.0375` traces to `MISTAKES.md:1297-1299` (entry M-15):
*"random Gaussian logits at the same measured logit standard deviation
(`0.0375`)"* — a null-distribution parameter for an **unrelated** common-mode
energy decomposition (`cm_full(a@a)`), confirmed at `V13_DAG_TASKLIST.md:713`
(`0.1127, 3.0025, 0.0375`) and `:866`. M-15 is itself the mistake-log entry
about a **baseline being reported as a result** — a null value from an
untrained operator, mistaken for structure. Citing that same `0.0375` as an
epistasis positive-control *result* would repeat the shape of M-15's error one
level up: a number belonging to one measurement, asserted as belonging to a
different one, without a run to back the second claim.

The real number for this claim is §2.2: **`0.6002` vs `6.03e-17`**, both
reproducible from `scripts/v15_x8_mobius_probe.py`, seeded, 5-seed mean, both
sides of the comparison run through the identical instrument.

**Limit.** The multiplicative landscape's `0.6002` is a synthetic-`f`
construction chosen to have a clean, provable nonzero degree-2 term
(`c0·c_a·c_b`); it is not a measurement on a trained or untrained attention
operator the way §1's `f` is, and it should not be read as an attention-model
result. `scale/trained_projections.py` already has the wiring
(`TD.leakage_ratio` on `Pair(trained=True/False)`) to run this same
must-fire/positive-control contrast on real trained vs. untrained geometries
instead of a synthetic landscape; that is a natural next probe and is not
what was run here.

---

## PART 3 — RENAME PLAN (not applied)

Per the task: report only. No file outside
`V15_X8_MOBIUS.md`/`scripts/v15_x8_mobius_probe.py` was modified.

**RUL-5 constraint, checked first.** `V15_LEDGER.md:42` (RUL-5):
`CEQ_V15_CONTRACT.md` and `attic/**` are **verbatim-of-record and are never
edited**; corrections live in a separate audit doc, not in the frozen file.
`CEQ_V15_1_DELTA.md:4` and `CEQ_V15_2_DELTA.md:4` say the same of themselves.
**This means the two sites that most directly name the estimator
(`CEQ_V15_CONTRACT.md:158,194` and `CEQ_V15_1_DELTA.md:88-94`) are not legal
rename targets at all** — renaming there would violate RUL-5. The correction
belongs in a new document (this one, or a future
`V15_CONTRACT_ARITHMETIC_AUDIT.md`-style file) that names the frozen line and
states the new name beside it, the way that file already does for other
contract corrections. `LOOP_PROMPT_ROUND6_ARCHIVE.md` is not explicitly named
by RUL-5 but follows the same `_ARCHIVE` convention as the other six frozen
round-prompt files; flagged below as **probably frozen by convention**,
coordinator to confirm rather than assumed.

| # | file | line(s) | current name | proposed name | notes |
|---|---|---|---|---|---|
| 1 | `scale/twodof.py` | 215 | `def leakage_ratio(...)` | `def mobius_interaction(...)` | primary definition; returns `(abs_I, abs_L3)` → rename to `(abs_k2, abs_k3)` or keep and document the `(-1)^k` convention (see §1.4) |
| 2 | `scale/twodof.py` | 218–223 | docstring: "Walsh coefficients", `fhat({c,j})` | update to "Möbius/inclusion–exclusion coefficients", `ĝ({c,j})` | prose only |
| 3 | `scale/twodof.py` | 357 | `leakage_ratio(...)` call in `main()` | `mobius_interaction(...)` | call site |
| 4 | `scale/trained_projections.py` | 65 | `from scale import twodof as TD` | unchanged (module name, not the estimator) | no action unless module itself is renamed |
| 5 | `scale/trained_projections.py` | 215 | `TD.leakage_ratio(...)`, locals `abs_i, abs_l3` | `TD.mobius_interaction(...)`, `abs_k2, abs_k3` | production call site — **flagged: this file is NOT `r10_capacity_sweep.py` or `identity_manifest.py`, no collision, but it is a live production path another node may also be touching; coordinator should sequence** |
| 6 | `scale/trained_projections.py` | 218–220, 314–315 | dict keys `abs_I`, `abs_L3`; report labels `"E\|I\| degree-2"`, `"E\|L3\| degree-3"` | `abs_k2`, `abs_k3`; `"E\|g(T)\| k=2"`, `"E\|g(T)\| k=3"` | changes on-disk `results/*.jsonl` schema if regenerated — **breaking change for downstream readers of that schema** |
| 7 | `tests/cameron/test_r6_twodof_red.py` | 152, 161, 175, 187 | `leakage_ratio`, "8-point mask", `abs_i`/`abs_l3` locals | mirror #1/#6 | test asserts the sign relationship already (line ~161-179); update alongside #1, not before — the test would need its `i_without - i_with` comparison re-derived for whichever sign convention is kept |
| 8 | `tests/loop/test_l3_leakage.py` | 22–39 | `interaction()`, `leakage()`, `ratio()`, module docstring | leave as-is, OR rename to `mobius_k2()`/`mobius_k3()`/`leakage_ratio()` | standalone reimplementation, not a repo-estimator call site; lower priority — never imports `twodof` |
| 9 | `tests/loop/test_two_dof_lemma.py` | 37 | `def interaction(...)` | leave as-is, OR `mobius_k2` | same as #8; also standalone |
| 10 | `LOOP_PROMPT_ROUND6_ARCHIVE.md` | 244, 248, 250 | "THE FOUR-POINT WALSH PROBE", `I(c,j) = ...`, `f̂({c,j})` | **flag only — probably frozen by `_ARCHIVE` convention (see RUL-5 note above); do not edit without coordinator confirmation** | |
| 11 | `CEQ_V15_CONTRACT.md` | 158, 193–195 | `walsh_two_point`, `multiplicative_epistasis_nonzero`, "Walsh degree-2" | **not a legal rename target — RUL-5, verbatim-of-record** | correction must be a new doc entry, not an edit |
| 12 | `CEQ_V15_1_DELTA.md` | 88–94 | "four-point estimator", "Möbius inversion", "Renamed in code and docs" | **not a legal rename target — RUL-5, verbatim-of-record** | this file is the one that ordered the rename and is itself frozen; the instruction can be carried out, the sentence that gave it cannot be edited |
| 13 | `scale/doc_readings.py` | 26 | prose mention `"scale/twodof.py" by -3` (a doc-drift census example) | update only if `twodof.py` the **file** is renamed, not for the estimator rename alone | lowest priority, cosmetic, not a functional call site |

**Dangling name, not a rename site but adjacent.** `CEQ_V15_CONTRACT.md:158`
lists Lean item `#9 walsh_two_point + multiplicative_epistasis_nonzero` as
`[M]` (must-prove-before-train). Grepped `lean/CEQ/*.lean` (excluding the
vendored `.lake/` packages): **no file defines `walsh_two_point` or anything
matching `epistasis`/`mobius`/`walsh`.** The Lean-side name has nothing to
rename because nothing was ever built under it — this is the "no such
estimator exists" finding the task asked to surface if applicable, applied to
the Lean inventory specifically rather than to `scale/`/`ceq/`.

**Collision check, explicit.** `scale/r10_capacity_sweep.py` and
`scale/identity_manifest.py` were grepped for `walsh`, `epistasis`, `mobius`,
`möbius`, `four-point`, `four_point`, `two_point`, `leakage_ratio`, `twodof`,
case-insensitive: **zero matches in either file.** Neither imports
`scale.twodof`. **No collision with the node editing those two files; this
rename plan touches neither.**

**Site count: 13** (rows above), of which **2 are not legal rename targets at
all** (RUL-5), **1 is flagged frozen-by-convention pending confirmation**, and
**1 is a dangling name with no code to rename**. The remaining 9 are live-code
or live-doc sites a coordinator can sequence.

---

## REPRODUCE

```
python scripts/v15_x8_mobius_probe.py
```

Self-contained: imports `scale.twodof` unmodified, writes nothing, trains
nothing, exits 0 iff every check above (μ cross-check, k=2 exact match, k=3
match-up-to-sign, additive must-fire, multiplicative positive control) holds.
