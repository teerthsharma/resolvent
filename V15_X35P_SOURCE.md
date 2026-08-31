# V15 X₃₅′ — THE EXACT SOURCE SOLVE, AND WHAT IT LOSES TO

Node JUPITER-6, CEQ v15.2 composition round. Spec: `CEQ_V15_2_DELTA.md`
§INSTRUMENTS (a) and (b), §LEAN #15, §MUST-FIRES 1–3, §KILLS.

Code: `ceq/x35p/source.py`, `lean/CEQ/V15Source.lean`. Tests:
`tests/x35p/test_source.py`. `ceq/x35p/__init__.py` is **not edited** — the
coordinator's docstring stands and `from ceq.x35p import source` resolves the
submodule without an export line, so the parallel node writing `kk.py`/`crb.py`
inherits an unmodified file.

Provenance for every number below: measured against the working tree at parent
commit `e0066b1` and re-confirmed green at `d8ad268` (15 passed, 11.82s, and the
Lean build unchanged) after the coordinator landed the parallel node's
`kk.py`/`crb.py`. Python 3.11.9,
numpy 1.26.4, Windows-11-10.0.26200, float64, single machine. Bed and seed
blocks are **X₃₅a's, verbatim** — `delay d=5, n=128`, one latent at true index
36 with magnitude 1.0, calibration seeds `[0, 2000)`, evaluation seeds from
`1e6` — so every comparison against `V15_X35A_RESIDUAL.md` is direct rather
than approximate.

---

## VERDICT — four calls, in the order the kills bind

1. **X₃₅′ is NOT VOID.** The no-plant production-path residual is flat:
   `corr = −0.1215` against a null sd of `0.089`, `half_ratio = 0.9747` against
   a null sd of `0.0125`, false-alarm rate `0.0120` on a fresh 2000-run block
   against a calibrated `α = 0.01`. Reproduced bit-for-bit from
   `V15_X35A_RESIDUAL.md` §5 on the same seeds; the X₃₅ kill binds first and it
   does not fire.
2. **(a) reproduces the delta's figure exactly.** Two planted sources, exact
   solve `ĥ = (I − A) r` on the propagated field: **`8.882e-16`**, against the
   delta's `[RUN 8.9e-16, two sources]`. It is machine epsilon, not a
   tolerance, because nothing is solved.
3. **(b) as literally written — `Wᵀ r`, unnormalized — cannot localize
   anything on this bed**, and the reason is structural rather than statistical
   (§6). Normalized by the column norms it is the strongest localizer measured
   here at every noise level under white noise, and it is beaten only by the
   Wiener form at heavily colored noise (`φ = 0.9`, §8c).
4. **THE EXACT INVERSE IS RETIRED TO THE NOISELESS REGIME.** It is strictly
   better than every alternative measured at `sd = 0` and at `sd = 0` only; it
   **ties** the Wiener adjoint at `sd = 0.05` and `0.10` on the production-path
   source; and it loses from `sd = 0.15` on the point-source task and from
   `sd = 0.20` on the production-path source, by a margin that grows
   monotonically thereafter. The delta's second kill fires and is called here in
   writing rather than argued around (§9).

**No CRB number and no KK number appears anywhere in this file** (§10).

---

## 1. LEAN #15 — `lean/CEQ/V15Source.lean`

### The statement, with its hypotheses

```lean
theorem source_is_first_order_difference
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : StrictlyLower A) (h r : Fin n → ℝ)
    (hfwd : r = CEQ.Occupancy.occupancy A n *ᵥ h) :
    h = r - A *ᵥ r
```

`StrictlyLower A` is `CEQ.Nilpotent.StrictlyLower` — `A i j = 0` whenever
`i ≤ j`, diagonal excluded — and it is the **only** hypothesis. `occupancy A n`
is `CEQ.Occupancy.occupancy`, the terminating sum `∑_{k<n} Aᵏ`. So the theorem
reads: *if the residual field is what the resolvent carrier produced from an
unobserved source, the source is recovered by a first-order difference of the
field, exactly.*

**It composes rather than restates.** `CEQ.Nilpotent.occupancy_is_exact_inverse`
already proves `(1 − A) · occupancy A n = 1` with the nilpotency hypothesis
discharged by `CEQ.Nilpotent.pow_card_eq_zero`; this file uses that as a
one-line `have` and adds the vector-level consequence the operator-level
theorem does not state. Nothing about `Aⁿ = 0` is re-derived.

**The trivial reading is refused explicitly, not silently avoided.**

```lean
theorem inverse_identity_is_vacuous {R : Type*} [Ring R] (u : Rˣ) :
    (u : R) * ((u⁻¹ : Rˣ) : R) = 1 := u.mul_inv
```

It is in the file, and then not used by anything. `M · M⁻¹ = 1` holds for every
unit of every ring, so it separates this carrier from no other and licenses no
claim about the cost or the sparsity of the inverse — which is the entire
argument the delta rests on the carrier.

**What makes "first order" a claim rather than a phrasing.** Two more theorems,
and the second is the one that does the work:

```lean
theorem no_fill_in {R : Type*} [Ring R] {A : Matrix (Fin n) (Fin n) R}
    {i j : Fin n} (hij : i ≠ j) (hA : A i j = 0) : (1 - A) i j = 0

theorem forward_map_fills_in :
    StrictlyLower shift3 ∧ shift3 2 0 = 0 ∧ (1 - shift3) 2 0 = 0 ∧
      (CEQ.Occupancy.occupancy shift3 3) 2 0 = 1
```

`no_fill_in` says the inverse's sparsity pattern is the hop's plus a diagonal —
at most `nnz A + n` entries, however long the paths through `A` are.
`forward_map_fills_in` is the witness that this is a real asymmetry: the
sub-diagonal shift on `Fin 3` has `A 2 0 = 0`, so the inverse is zero there,
while the forward map `∑_{k<3} Aᵏ` reads `1` there — the length-2 path
`0 → 1 → 2` that `A²` counts. Sparsity of an inverse is a claim only against
something denser, and that is the something.

Two supporting theorems: `two_sources_recovered` (linearity of the forward map,
so the superposition must-fire costs nothing extra and needs no assumption that
the sources have disjoint support) and `source_entry` (the cost written out at
one index: `(r − A ·ᵥ r)_i = r_i − ∑_j A_ij r_j`, one weighted sum over `A`'s
own row support).

### Build

```
$ cd lean && lake env lean CEQ/V15Source.lean
'CEQ.V15Source.inverse_identity_is_vacuous' does not depend on any axioms
'CEQ.V15Source.source_is_first_order_difference' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Source.two_sources_recovered' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Source.no_fill_in' depends on axioms: [Quot.sound, propext]
'CEQ.V15Source.shift3_strictlyLower' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Source.forward_map_fills_in' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Source.source_entry' depends on axioms: [propext, Classical.choice, Quot.sound]
```

No errors, no warnings, no `sorry` (`grep -n sorry lean/CEQ/V15Source.lean`
returns line 60 only, which is the module docstring saying there is none), and
no `sorryAx` in any axiom set. The three axioms listed are Mathlib's standard
three. The `#print axioms` lines are in the file, at the foot, one per theorem.

**FOR THE COORDINATOR — the import line is**

```lean
import CEQ.V15Source
```

`lean/CEQ.lean` was **not** touched by this node. `CEQ.V15Source` imports
`CEQ.Occupancy` and `CEQ.Nilpotent` and nothing else of the repo's, so it may
be placed anywhere after those two and does not order against `CEQ.V15Kernel`.

---

## 2. WHERE THE PROPAGATOR COMES FROM — the scope statement, measured first

`ceq/beds/bed_k.py` adds the planted latent `u` to `z` **directly, after the
kernel**. `V15_X35A_RESIDUAL.md` §9 files that as a limit in those words. With
the oracle visible model the production-path residual is therefore

```
r  =  u + observation noise
```

identically. **The forward map from the hidden source to the residual field is
the identity, and the resolvent's `A` is exactly 0 on that path.** Measured, not
assumed (`test_production_path_propagator_is_the_identity`):

```
[SCOPE] production-path ||r - u||_inf = 2.220e-16 (propagator = I, A = 0)
```

Two consequences, both stated rather than worked around:

1. On the production path the exact solve reduces to `r`, and the recovery
   error is the float64 rounding of `(K b + u) − K b`. **That is what `8.9e-16`
   is**, and it is why the delta says the figure is "not a convergence
   tolerance". It is also, by itself, not a test of an inverse.
2. An identity forward map has no inverse worth testing, and two estimators
   that are both the identity cannot be compared at all. So the comparison runs
   on the **propagated field**: the same production-path source
   `bed["plant"]["u"]`, the same production-path observation stream
   (`ceq.x35.residual.observe` — same salt, same seed), with the architecture's
   own carrier between them:

   ```
   A = bed_k.kernel_matrix(kind, n, **params)      the bed's own memory kernel
   W = occupancy(A, n) = ∑_{k<n} Aᵏ                the resolvent, exact
   y = W h + noise
   ```

   Every component is the shipped one; the composition is the single new line
   and it is named (`ceq/x35p/source.py::wave_field`). `L-SCOPE` is satisfied on
   the must-fires — which run on the production-path residual itself — and the
   sweep says exactly what it is measured on.

The carrier is bitwise exact and its inverse is bitwise sparse
(`test_occupancy_is_the_exact_two_sided_inverse`,
`test_the_inverse_is_a_first_order_difference_and_the_forward_map_is_not`):

```
[CARRIER] ||(I-A)W - I||_max=0.000e+00 ||W(I-A) - I||_max=0.000e+00 cond(W)=33.69
[FIRST-ORDER] nnz(A)=123 nnz(I-A)=251 nnz(W=sum A^k)=1703 fill-in=+1452
```

`0.000e+00` on both sides is the nilpotency doing the work: the sum terminates,
there is no tail to bound, and `cond(W) = 33.69` never enters any computation
because nothing is solved. `nnz(I − A) = 251` against `nnz(W) = 1703` is
`forward_map_fills_in` with numbers in it.

---

## 3. RED

Test file authored and run first, before `ceq/x35p/source.py` existed:

```
$ python -m pytest tests/x35p/test_source.py -q
=================================== ERRORS ====================================
_________________ ERROR collecting tests/x35p/test_source.py __________________
ImportError while importing test module 'C:\Users\seal\Desktop\New folder (32)\tests\x35p\test_source.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\x35p\test_source.py:69: in <module>
    from ceq.x35p import source as sx
E   ImportError: cannot import name 'source' from 'ceq.x35p' (C:\Users\seal\Desktop\New folder (32)\ceq\x35p\__init__.py)
=========================== short test summary item ===========================
ERROR tests/x35p/test_source.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.47s
```

## 4. GREEN

```
$ python -m pytest tests/x35p/test_source.py -q -s

[CARRIER] ||(I-A)W - I||_max=0.000e+00 ||W(I-A) - I||_max=0.000e+00 cond(W)=33.69
[FIRST-ORDER] nnz(A)=123 nnz(I-A)=251 nnz(W=sum A^k)=1703 fill-in=+1452
[SCOPE] production-path ||r - u||_inf = 2.220e-16 (propagator = I, A = 0)
[MUST-FIRE 1] one source at 36: production-path err=2.220e-16 propagated-field err=6.661e-16
[MUST-FIRE 2 / superposition] sources at [36, 80]: production-path err=4.441e-16 propagated-field err=8.882e-16
[MUST-FIRE 3 / flatness] corr=-0.1215 (3sd=0.2662) half_ratio=0.9747 (3sd=0.0375) mean_energy=0.00249003 (sd^2=0.0025)
[MUST-FIRE 3 / FAR] tau=0.0382564 measured FAR=0.0120 (24/2000) vs alpha=0.01
[ADJOINT FLATNESS] raw W^T corr=-0.9916  column-normalized corr=+0.1582
[SWEEP sd=0.00] exact=1.000 adjoint=1.000 wiener_white=1.000 wiener_colored=1.000
[SWEEP sd=0.20] exact=0.720 adjoint=0.990 wiener_white=0.780 wiener_colored=0.780
[SWEEP sd=0.30] exact=0.350 adjoint=0.905 wiener_white=0.415 wiener_colored=0.415
[COLORED phi=0.7 sd=0.3] exact=0.455 adjoint=0.885 wiener_white=0.535 wiener_colored=0.650
[WHITE   phi=0.0 sd=0.3] exact=0.350 adjoint=0.905 wiener_white=0.415 wiener_colored=0.415
[ONSET SWEEP sd=0.00] exact_mae=0.000 wiener_mae=36.000 paired=-36.000 exact_wins=200 ties=0 wiener_wins=0
[ONSET SWEEP sd=0.30] exact_mae=7.720 wiener_mae=6.020 paired=+1.700 exact_wins=2 ties=165 wiener_wins=33

15 passed in 68.06s (0:01:08)
```

`python -m pytest tests/x35 tests/beds -q` → **42 passed in 108.50s**,
unchanged: nothing in `ceq/x35/**` or `ceq/beds/**` was modified.

---

## 5. MUST-FIRE 3 — NO PLANT ⇒ FLAT RESIDUAL. IT BINDS FIRST.

> *"no-plant residual fails flatness ⇒ X₃₅ VOID. The detector cannot outrun the
> model it subtracts."*

Measured on the **production-path residual**, oracle visible model, `sd = 0.05`,
pooled over 200 fresh no-plant runs from the evaluation block:

| statistic | measured | flat means | null sd | deviation |
|---|---|---|---|---|
| energy-vs-index Pearson `corr` | **−0.1215** | 0 | 0.089 (`1/√(n−1)`) | 1.4 sd |
| second-half / first-half energy ratio | **0.9747** | 1 | 0.0125 | 2.0 sd |
| mean residual energy | **0.00249003** | 0.0025 (`sd²`) | — | — |

False-alarm rate, threshold from the calibration block `[0, 2000)`, scored on
the disjoint evaluation block `[1e6, 1e6+2000)`:

```
tau=0.0382564   measured FAR=0.0120 (24/2000)   vs calibrated alpha=0.01
```

**X₃₅′ is NOT VOID.** These are `V15_X35A_RESIDUAL.md` §5's numbers reproduced
on the same seeds, which is the point: this node scores the identical field
that node's detector read, and `test_scores_the_field_the_x35_detector_read`
asserts `run_detect(...)["residual"]` is `np.array_equal` to
`run_residual(...)[1]`, not merely close to it.

The scope is X₃₅a's scope and inherits its condition unchanged: the visible
model is the **oracle**, so the no-plant residual is exactly the observation
noise and flatness is exact rather than estimated. A trained arm carries
approximation error the oracle does not, and this gate **re-arms** the moment
`z_model` changes. The rejection region that makes the gate non-vacuous is
`V15_X35A_RESIDUAL.md` §5's `truncated_visible` control (FAR `0.2667` at
`L = 64`, `1.0000` at `L = 8`); it was not re-run here because nothing in this
node changes the visible model.

---

## 6. INSTRUMENT (b) — WHY `Wᵀ r` SHIPS NORMALIZED

`Wᵀ` is a **reverse cumulative sum along the carrier's hop lattice**:
`(Wᵀ r)_i` sums `r` over the indices reachable *from* `i`. Its null variance is
therefore `σ² ‖W e_i‖²`, which falls monotonically with `i`. Measured under no
plant at `sd = 0.05`, 200 runs
(`test_the_raw_adjoint_is_structurally_not_flat_under_no_plant`):

```
[ADJOINT FLATNESS] raw W^T corr=-0.9916   column-normalized corr=+0.1582
```

`−0.9916` is not noise and is not a bad threshold. It is the operator's shape,
and it has a hard consequence: **a first-crossing rule on the raw field calls
index 0 at every seed and every noise level** (onset MAE pinned at 36.0 = |0 −
36| across the whole sweep, in the pre-registration probes), and `argmax |Wᵀ r|`
is dominated by `i = 0` whatever the source did. Instrument (b) as literally
written in the delta is void on this bed for that reason and no other.

Dividing by the column norms `‖W e_j‖` — the geometric-spreading correction of
time-reversal imaging, and exactly the normalization that makes the null
variance index-independent — restores flatness (`+0.1582`, 1.8 null sd) and
with it the focusing property. `adjoint_source(..., normalize=True)` is the
default for that reason; `normalize=False` returns the raw field for the record.

**Where the Wiener form is needed.** The delta's qualifier is "with Wiener
deconvolution when the noise is colored". `wiener_source` implements
`ĥ = (Wᵀ C⁻¹ W + σ_h⁻² I)⁻¹ Wᵀ C⁻¹ r` with `C` defaulting to `σ² I`. Two things
it is handed that the exact solve is not, stated because they favour it:
`noise_var` and `source_var` are supplied from outside and the runs below supply
their **true** values.

---

## 7. MUST-FIRES 1 AND 2 — THE EXACT SOURCE SOLVE

`ĥ = (I − A) r = r − A r`. One mat-vec against the hop and a subtraction:
`exact_source` takes `A` and never takes `W`, asserted by
`test_exact_solve_never_forms_the_forward_map`, because a source solve that
needed `W` would already have paid for what the closed form exists to avoid.

| plant | production-path `r` (`A = 0`) | propagated field `y = W h` |
|---|---|---|
| one source @ 36 | `2.220e-16` | `6.661e-16` |
| **two sources @ 36, 80** | `4.441e-16` | **`8.882e-16`** |

`8.882e-16` against the delta's `[RUN 8.9e-16, two sources]`. Both sources are
recovered index-by-index, not merely their sum: `ĥ` is **bitwise zero** on the
whole pre-plant prefix `[0, 36)` and both switch-ons are readable off it. That
is the thing X₃₅a's Shewhart onset structurally cannot do — it calls the
earliest source only, because the second switches on inside a residual already
above threshold (`V15_X35A_RESIDUAL.md` §4) — and it is what instrument (a) buys
over the detector at zero noise.

Superposition needed no separation step and no disjoint-support assumption:
`two_sources_recovered` in the Lean file is the reason, and it is one line off
linearity of the forward map.

---

## 8. THE NOISE SWEEP — THE COMPARISON, AND IT IS THE DELIVERABLE

Two sweeps, because the two estimators answer different questions and the kill
has to survive both. Raw localization error only; **no floor is quoted beside
any of it** (§10).

### 8a. Point source, localized by the peak of the recovered source

The task time reversal is *for* (Fink): a source at a single index, found by
where the back-propagated field focuses. Source `h = e_p` with `p` drawn per
seed from a stream disjoint from the noise (`POINT_SALT`), field `y = W h + n`,
white `n`, 500 evaluation seeds from `1e6`. Score = fraction of seeds whose
`argmax |ĥ|` is exactly `p`.

| noise `sd` | exact `(I−A)y` | raw `Wᵀy` | **normalized `Wᵀy`** | Wiener |
|---|---|---|---|---|
| 0.00 | **1.000** | 0.044 | **1.000** | 1.000 |
| 0.05 | **1.000** | 0.224 | **1.000** | 1.000 |
| 0.10 | **1.000** | 0.222 | **1.000** | 1.000 |
| 0.15 | 0.960 | 0.222 | **1.000** | 0.964 |
| 0.20 | 0.736 | 0.220 | **0.990** | 0.790 |
| 0.25 | 0.516 | 0.218 | **0.952** | 0.578 |
| **0.30** | **0.344** | 0.218 | **0.892** | 0.410 |
| 0.40 | 0.172 | 0.206 | **0.742** | 0.252 |
| 0.50 | 0.112 | 0.198 | **0.596** | 0.162 |
| 1.00 | 0.038 | 0.146 | **0.258** | 0.074 |

**The crossover is at `sd = 0.15`** — the first level at which the exact solve
is not perfect and the adjoint still is. It is not a marginal crossing: by
`sd = 0.20` the gap is `0.736` against `0.990`, and by `sd = 0.30` it is
`0.344` against `0.892`. The delta's `[RUN: exact at sd 0.3]` for instrument (b)
reproduces in shape at `0.892`, while the exact solve has collapsed to a third.

The mechanism is not statistical luck and was predicted before the run:
`(I − A)` **differences** the observation noise, so `ĥ = h + (I − A)n` carries
noise of variance `σ²(1 + γ²)` — worse than the field it started from — while
`Wᵀ` **averages** it over the `≈ n/d = 26` indices in the carrier's reachable
set. The exact inverse is exact about the signal and destructive about the
noise, in the same operation.

The raw `Wᵀ` column is the §6 finding, and note its shape: it is **worse at
`sd = 0` (0.044) than at `sd = 0.05` (0.224)**. An estimator that improves when
noise is added is not reading the source; it is reading its own column-norm
profile, and 0.044 is what that profile scores when nothing perturbs it.

### 8b. Production-path step source, localized by onset

The source `bed_k` actually plants: switch-on at index 36 with iid `N(0, 1)`
drive after it. Localized by the **same memoryless comparator X₃₅a uses**
(`residual.detect`), each estimator's threshold calibrated on its own no-plant
runs over `[0, 2000)` at `α = 0.01`, scored on 500 disjoint seeds from `1e6`.
Errors are **paired on seeds** — the same field goes to both — and a miss is
scored as error `n = 128` rather than dropped, so an estimator cannot look
accurate by alarming rarely. The raw adjoint is not swept here: it is not an
onset estimator on this field at all (§6).

| `sd` | exact MAE | Wiener MAE | paired diff | exact wins | ties | Wiener wins | exact misses | Wiener misses |
|---|---|---|---|---|---|---|---|---|
| 0.00 | **0.000** | 36.000 | **−36.000** | **500** | 0 | 0 | 0 | 0 |
| 0.05 | 0.268 | 0.272 | −0.004 | 2 | 498 | 0 | 0 | 0 |
| 0.10 | 0.712 | 0.710 | +0.002 | 1 | 496 | 3 | 0 | 0 |
| 0.20 | 2.568 | **2.432** | +0.136 | 3 | 471 | 26 | 0 | 0 |
| 0.30 | 6.812 | **5.348** | +1.464 | 5 | 408 | 87 | 0 | 0 |
| 0.50 | 43.626 | **18.784** | +24.842 | 9 | 225 | 266 | 67 | 5 |
| 1.00 | 118.212 | **67.756** | +50.456 | 4 | 206 | 290 | 442 | 174 |

`sd = 0.05` and `sd = 0.10` are **ties, and are reported as ties**: 498 and 496
of 500 seeds give the identical onset, and the 2-vs-0 and 1-vs-3 splits on the
remainder are not a result in either direction. The sign flips for good at
`sd = 0.20` and the margin grows monotonically at every step after. The miss
columns are where the difference stops being cosmetic: at `sd = 0.5` the exact
solve fails to call anything on **67 of 500** runs against Wiener's 5.

**The `sd = 0.00` row is the exact inverse's one clean win, and it is a real
one.** `(I − A)y` is a subtraction, so the pre-onset prefix is **bitwise** zero
and the calibrated threshold of `0.0` fires at exactly index 36 on every seed.
The Wiener arm at `λ → 0` is ordinary least squares — a linear **solve** —
whose `~1e-17` rounding a zero threshold turns into an onset at index 0, every
time. That is the closed form's advantage at its sharpest: not accuracy, but
the absence of a solve.

### 8c. Colored noise — instrument (b)'s qualifier

AR(1) observation noise, marginal `sd = 0.3`, point-source task, 500 seeds.
`wiener(σ²I)` is handed the wrong covariance; `wiener(true C)` is handed
`C_ij = σ² φ^|i−j|`.

| `φ` | exact | normalized `Wᵀ` | Wiener `(σ²I)` | Wiener (true `C`) |
|---|---|---|---|---|
| 0.0 | 0.344 | **0.892** | 0.410 | 0.410 |
| 0.3 | 0.376 | **0.888** | 0.436 | 0.458 |
| 0.7 | 0.442 | **0.884** | 0.514 | 0.630 |
| 0.9 | 0.818 | 0.842 | 0.842 | **0.906** |

The covariance argument earns its place: `0.630` against `0.514` at `φ = 0.7`,
`0.906` against `0.842` at `φ = 0.9`. The `φ = 0` row is the control that makes
the rest readable — the two Wiener columns are **identical**, not merely close,
because `ar1_cov(n, sd, 0)` is bitwise `sd² I` and the estimator takes one code
path. Without that row, "the colored version is better" could have been a
second implementation being better at something else.

### 8d. Where the O(nnz) claim holds, and where it does not

| BED-K kind | `nnz(A)` | `nnz(I − A)` | `nnz(W = ∑Aᵏ)` | max abs of `(I−A)W − I` |
|---|---|---|---|---|
| `delay d=5` | 123 | **251** | 1703 | **0.000e+00** |
| `powerlaw H=0.75` | 8128 | 8256 | 8256 | 2.274e-13 |

**The delay carrier is where "sparse and local" is true**: two nonzeros per row,
the solve reads `ĥ_i = r_i − γ r_{i−5}`, and the identity is exact to the last
bit because the sum terminates in 26 terms. **On the power-law carrier the
fill-in advantage is exactly zero** — `A` is already dense lower-triangular, so
`nnz(I − A) = nnz(W) = 8256` and the inverse is no sparser than the forward map.
What survives there is the weaker half of the claim, and the two halves are
worth separating: the solve is still one mat-vec with no factorization and no
conditioning question (`O(n²)` against a solve's `O(n³)`), but it is **not
local, and the 251-against-1703 figure is a property of the delay hop rather
than of the resolvent carrier in general**. The `2.274e-13` in the last column
is the same distinction from the other side: 128 accumulated terms instead of
26.

---

## 9. THE KILL — THE RETIRE CALL, EXPLICIT

> *"Detector worse than the adjoint at high noise ⇒ the exact inverse is retired
> to the noiseless regime, stated. The exact solve is not defended past where it
> stops winning."*

**THE EXACT INVERSE IS RETIRED TO THE NOISELESS REGIME.** The kill fires, and
these are the numbers it fires on:

- It is **strictly better than every alternative measured at `sd = 0` only**:
  bitwise-exact recovery (`8.882e-16` with two sources), a bitwise-zero
  pre-onset prefix, 500/500 onsets exact, and no linear solve anywhere in it.
- It **ties** the Wiener adjoint at `sd = 0.05` and `sd = 0.10` on the
  production-path source (498/500 and 496/500 identical onsets).
- It **loses** from `sd = 0.15` on the point-source task (`0.960` against
  `1.000`) and from `sd = 0.20` on the production-path source, and the margin
  grows monotonically at every step after: `0.344` against `0.892` at
  `sd = 0.30`, MAE `43.6` against `18.8` at `sd = 0.50`, 442 misses of 500
  against 174 at `sd = 1.00`.

The scope it keeps is not nothing and is not a consolation prize. At `sd = 0`
it is the only estimator here that is exact rather than accurate; it recovers
**superposed** sources index-by-index where X₃₅a's onset detector structurally
reports only the earliest; and it costs one mat-vec against a hop with two
nonzeros per row, against the adjoint's dense `Wᵀ` and Wiener's `O(n³)` solve.
An exact method that is exact only at zero noise is a real and useful thing.
What it is not is dominant, and this node does not claim it is.

**The practical reading, stated so the next node does not have to re-derive
it.** Use `(I − A)r` where the residual is a computation rather than a
measurement — model-against-model differences, exact-arithmetic checks, the
noiseless-oracle configuration this round is calibrated in. Use the normalized
adjoint, Wiener-deconvolved with the true covariance when the noise is colored,
the moment a real observation channel is in front of it.

---

## 10. WHAT IS NOT COMPUTED HERE

**No Cramér–Rao number, and none is implied.** `CEQ_V15_2_DELTA.md` strikes any
CRB or Kramers–Kronig number quoted before its own must-fire; both instruments
belong to a parallel node. Every accuracy figure above is **raw** — a hit rate
or a mean absolute index error — with no floor beside it. In particular this
node makes **no optimality or near-optimality claim** for any estimator: "near
the floor" is a CRB claim by implication and is struck the same way an explicit
number would be. That the normalized adjoint wins every noisy row above is a
statement about the three estimators compared, not about the estimators
possible.

**No Kramers–Kronig number.** Nothing in `ceq/x35p/source.py` or
`tests/x35p/test_source.py` touches a causality residual.

**No CUSUM number.** The onset comparator is `ceq.x35.residual.detect`, the
memoryless Shewhart chart, unchanged; the settled memoryless-versus-CUSUM
result is not relitigated here.

---

## 11. LIMITS

The propagator on the production path is the **identity**, measured at
`2.220e-16`, because `bed_k` adds the latent to `z` after the kernel — so
must-fires 1 and 2 as run on the production-path residual are float64
subtractions, and every number that involves an actual inverse is measured on
the propagated field `wave_field` composes from the same source, the same
observation stream and the bed's own kernel. That composition is the one step
this node adds and it is named rather than hidden. The flatness gate at §5 is
X₃₅a's, at the **oracle's scope only**, and re-arms the moment `z_model` becomes
a trained arm whose approximation error is kernel-dependent and is not zero.
The Wiener arm is handed the **true** `noise_var` and `source_var` throughout,
which favours it against the exact solve and against any deployed version of
itself; its prior `σ_h² I` is also mis-specified for the step-onset source,
which is non-stationary, so its numbers are neither an upper nor a lower bound
on what a tuned version would do. The point-source task in §8a and §8c is a
diagnostic and is **not** a production-path bed: `bed_k`'s plant is a step-onset
extended source and cannot express a single-index source, so that table says
what time reversal does on the task it was designed for and is labelled as
such. `γ = 1` throughout, and no `γ < 1` carrier was measured. The `sd = 0.05`
and `sd = 0.10` rows of §8b are ties and are reported as ties. All numbers are
single-machine, float64, at the seed blocks named in each section; the crossover
in §8a is located to the grid `{0.10, 0.15, 0.20}` and was not bisected further.
