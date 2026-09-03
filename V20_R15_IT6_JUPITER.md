# V20 R15 — it.6 — JUPITER (MYCROFT) — PHASE B OPENS: Q1 and Q2, both wings

**Branch `v17k-gate0`, HEAD `207e7b9`.** Contract slot `CEQ_V20_R15_CONTRACT.md:94-113`
(PHASE B — ARMING). Frozen list `V20_R15_WING_MANIFEST.md:20-21`: **N = 2**,
W1 `arm_smprime`, W3 `arm_pl`.

---

## 0. WHICH CELLS THIS ITERATION FILLED, AND WHICH IT DID NOT

Phase B is twelve cells (six questions x two wings) across it.6-it.14. Six
questions per wing inside one 20-minute wall clock produces twelve shallow
answers, and the round has already paid once for a theorem asserted on a domain
the corpus never enters — M14, struck by V-25 rather than by its own kill.

| | W1 `arm_smprime` | W3 `arm_pl` |
|---|---|---|
| **Q1 EXACT CLASS** | **FILLED** — §1 | **FILLED** — §2 |
| **Q2 OUTSIDE THE CLASS** | **FILLED** — §3 | **FILLED** — §4 |
| Q3 LEARNABILITY | **NOT FILLED** — seed recorded §6 | **NOT FILLED** — seed recorded §6 |
| Q4 COST LAW | NOT FILLED | NOT FILLED |
| Q5 INFORMATION FLOOR | NOT FILLED | NOT FILLED |
| Q6 STATE METRIC | NOT FILLED | NOT FILLED |

**Four filled, eight not.** Nothing in §6 answers Q3; it records a seed with its
citation, which is what it.5 asked for.

**Citation discipline, restated because it is the failure mode this office
created.** Six Lean citations were withdrawn at it.2 after WILSON's census found
no source declaration behind their numbers — `Lean #21 [S]`, the annex's own
authority for the scan-computable class, among them. **Every theorem below is
named by DECLARATION and `path:line` and by nothing else.** An annex number is
an index, not evidence.

---

## 1. Q1 / W1 `arm_smprime` — THE EXACT CLASS

### 1.1 The statement

One hop of the path product represents with **zero error** exactly the family

> `y_i = SUM_{j<i} ( PROD_{k=j+1..i} m_k ) * exp( i * SUM_{k=j+1..i} theta_k ) * b_j`,
> **`m` in `[0,1]` closed at BOTH endpoints**, `theta` in `R` unrestricted.

### 1.2 The theorems that carry it, by declaration and line

| declaration | `path:line` | hypothesis, verbatim | what it contributes |
|---|---|---|---|
| `pathProd_polar` | `lean/CEQ/V16Domain.lean:105` | **none** on `m`, none on `theta` | the factorization itself: magnitude from `m`, unit modulus from `theta`, **no cross term** |
| `pathProd_eq_zero_iff` | `lean/CEQ/V16Domain.lean:129` | **none** | closes the `m = 0` endpoint as an **iff** — the class is CLOSED, not open |
| `pathProd_abs` | `lean/CEQ/V16Domain.lean:121` | `(h0 : forall k, 0 <= m k)` | `abs(pathProd) = PROD m` |
| `pathProd_eq_Wp` | `lean/CEQ/V16Domain.lean:176` | `(hm : forall k, 0 < m k)`, `(hij : j <= i)` | on the strictly-positive part the class **coincides** with the exponential-scan form |
| `bedM_gate_exact` | `lean/CEQ/V16Domain.lean:251` | `(ha : a = -1 or a = 0 or a = 1)` | `gateOf (magOf a) (argOf a) = a` **exactly in C** on each of BED-M's three support points |
| `negative_draw_is_on_the_band` | `lean/CEQ/V16Domain.lean:273` | **none** | `gateOf 1 pi = (-1 : C)` — the `a = -1` draw is an interior point, not a boundary case |

**The two hypothesis-free declarations are the ones that carry the class
statement.** `pathProd_polar` and `pathProd_eq_zero_iff` quantify over all
`m theta : N -> R`. The two carrying hypotheses (`pathProd_abs`,
`pathProd_eq_Wp`) are refinements, and §5 prices exactly what their hypotheses
cost against the beds.

**Toolchain.** `lean/lean-toolchain` reads `leanprover/lean4:v4.7.0`; `lake build`
in `lean/` exits 0 with no output — `[CITED]` `V20_R15_IT1_WILSON.md:131,:137`.
**Not re-run this iteration; that is a gap and it is graded in §7.**

### 1.3 Grade

**Q1 / W1 = F0**, for the hop.

The class statement rests on two machine-checked declarations with **no
hypotheses at all**, and both endpoints are characterized rather than dodged —
`m = 0` by an iff (`:129`), `a = -1` by an explicit evaluation (`:273`).

### 1.4 HOW-BAD gap, with its number

**The theorems cover the HOP. They do not cover the ARM.**
`ceq/arm_smprime.py`'s `hop()` returns `(G, R)` and the trained arm carries the
`beta`, `QK` and `g` switches plus a normalizer `Z`. The configuration the F0
statement covers is named in the code itself:

> `ceq/arm_smprime.py:259` — *"At `beta = 0` with QK off this is `sum_j (prod_{k>j} a_k) b_j` -- BED-M's ..."*

**`[RUN]` the record, and the gap is exact: of the 24 `t="cell"` records in
`results/v17k_r4_retake.jsonl`, ZERO journal `beta`, `qk`, `route`, or `g`.** A
cell carries 58 keys — `a_hat_max ... z_winding_residual` — and the switch
configuration is not among them. So **0 of 24** arena cells can be shown to sit
in the configuration the F0 theorem's domain is conditioned on. Not because they
did not; because **the record cannot say**.

**HOW-BAD = 0/24 cells attributable.** A theorem exact on a configuration the
journal does not record is exact about nothing the arena will price.

**REPLACEMENT ROUTE, and it costs 0 GPU-seconds:** journal `beta`, `qk`, `route`
and `g` as four scalar fields per cell at the next run. Nothing already banked
needs re-running — these are configuration, not measurement, and Phase C has not
opened. Filed now rather than discovered at it.29 when the theory table is asked
which cell its F0 refers to.

---

## 2. Q1 / W3 `arm_pl` — THE EXACT CLASS

### 2.1 The statement

One causal-softmax hop with the real prefix scan entering **key-side only**
(`ceq/arm_pl.py:88` `scan`, `:93` `key_bias`, `VARIANT = "key_only"`) represents
with zero error exactly

> `y_i = SUM_{j<=i} A_ij b_j`, with
> `A_ij = exp(s_j - C_j + q_i.k_j) / SUM_{j'<=i} exp(s_j' - C_j' + q_i.k_j')`, `C = cumsum(g)`

and with the QK term off this is the scan-computable family with `phi = exp`:

> `y_i = SUM_{j<=i} exp(C_i - C_j) b_j`

via the telescoping normalizer — `ceq/arm_pl.py` `normalizer`: *"at the oracle
setting `exp(-C_j)(1 - a_j) = exp(-C_j) - exp(-C_{j-1})` telescopes to
`exp(-C_i) - 1`, so `Z_i = 1` EXACTLY."*

### 2.2 THE POSITIVE STATEMENT HAS NO THEOREM, AND THIS OFFICE IS WHY

The annex's authority for this class was `Lean #21 [S]`. **That citation was
withdrawn at it.2** — WILSON's census found no source declaration behind the
number, and this office withdrew six such citations rather than defend them.
**No replacement declaration exists.** What `lean/` carries for the scan side is
the corner family (`corner_softmax` `:394`, `corner_linear` `:401`,
`corner_path_product` `:408`) and nothing that states W3's representable class.

**What W3 does have, machine-checked, is the NEGATIVE boundary of its class:**

| declaration | `path:line` | hypothesis | what it contributes |
|---|---|---|---|
| `no_prefix_scan_represents_a_zero_gate` | `lean/CEQ/V16Domain.lean:165` | `(hz : exists k in Ico (j+1) (i+1), m k = 0)`; **`C : N -> C` universally quantified** | **no prefix scan whatsoever** — not `scan(log m)`, not any repaired or extended-real variant — represents a hop that annihilates |
| `lean_log_junk_makes_the_scan_form_silently_false` | `lean/CEQ/V16Domain.lean:147` | **none** | the same fact in the form a proof assistant takes it: the naive relaxation returns `1` where the path product is `0`, and does it silently |
| `sixteen_is_silent_on_the_zero_draw` | `lean/CEQ/V16Domain.lean:355` | **none** | the census's own guard: a hypothesis-free theorem has total overlap and can still be decoration; the **second column** is what fails it |

The universal quantifier at `:165` is the load-bearing part. It is not a claim
about this implementation's `C`. It is a claim about every `C`.

### 2.3 Grade

**Q1 / W3 = F1**, delta = the instance tolerance. **Not F0.**

Q1's bar is *"which labels does the primitive represent with zero error, and by
which theorem? (F0 or it isn't exact.)"* W3's positive class statement is
carried by a `<=1e-6` numerical instance and a code-level derivation, and by no
theorem. **An instance at `1e-6` is not zero error.** F1 is the honest grade;
F0 would be the it.2 mistake repeated with a different number on it.

**F1 is not F4. The wing is NOT withdrawn.** Q1 is answered: the class is stated,
its boundary is machine-checked in the strongest available form (universal over
`C`), and it is instanced. What is missing is a positive theorem, and that is a
**priced** gap, not an absent answer.

### 2.4 HOW-BAD gap and replacement route

**Gap 1 — the missing theorem, priced rather than papered over.** The wanted
statement is a telescoping lemma for `Znorm` (`lean/CEQ/V16Domain.lean:370`)
over `CEQ.V15.scan`, of the shape *"at `s_j - C_j` with `qk == 0`,
`Znorm g qk i = 1`"*. Both ingredients are already in the file.
**Route: attempt it at it.7 as a Lean obligation with a named declaration, or
carry Q1/W3 at F1 into the it.14 theory table and say so in the leap dossier.**
The third route — citing a number — is refused.

**Gap 2 — the class exclusion is not marginal, it is the bed.** Quantified in §5.

---

## 3. Q2 / W1 — OUTSIDE THE CLASS, AS A BOUND WITH A CONSTANT

Instrument: the **Hankel best-k-state** bound, `R2_k = SUM_{i<=k} sigma_i^2 /
SUM_i sigma_i^2`, inducing `err_k >= sqrt(1 - R2_k)`. Implementation
`ceq/hankel.py`, already in the tree, and already separating the three columns
that get conflated (`rank_R` / `rank_+` / Myhill-Nerode).

Measured constants: §8.

---

## 4. Q2 / W3 — OUTSIDE THE CLASS, AS A BOUND WITH A CONSTANT

**W3's bound is not an energy fraction, and saying so is the content of this
cell.** `ceq/hankel.py` returns the `NEG_ENTRY` sentinel — its own docstring:
*"no nonnegative factorisation exists at ANY size"* — when the Hankel matrix
carries a negative entry.

W3's operator row is a **softmax**: `A_ij >= 0` elementwise and `SUM_j A_ij = 1`.
So `y_i` is a **convex combination** of `{b_j : j <= i}`, for every
`(g, s, q, k)` whatsoever, and therefore

> `min_{j<=i} b_j <= y_i <= max_{j<=i} b_j`

A target leaving that interval is unreachable, and the **excess is the constant**:

> `err_i >= dist( t_i , [ min_{j<=i} b_j , max_{j<=i} b_j ] )`

This is a **nonexistence** bound, not a truncation bound. For W1, "how many
states" has a finite answer `k` with a residual. For W3 on a sign-alternating
path product **there is no `k` at any size**. That is the sharpest form Q2 takes
on this wing and the round has not previously stated it.

Measured constants: §8.

---

## 5. THE L-DOM CENSUS — TWO COLUMNS, FOR EVERY THEOREM CITED IN §1-§4

The it.2 census found Cheeger's hypotheses satisfied by **0 of the registered
beds**, which is why M14 dies by V-25 rather than by its own kill. The same
census is run here for every theorem above, and the result that matters is that
**for W1 it is already machine-checked, by `decide`, inside the same Lean file.**

### 5.1 The census, already proved

`lean/CEQ/V16Domain.lean:288-312`, the support as a decidable literal list and
the overlap counts as theorems, not as a script:

| declaration | `path:line` | statement | reads |
|---|---|---|---|
| `bedM` | `:288` | `List Z := [-1, 0, 1]` | BED-M's gate support, sourced at `scale/negation_scope.py:428-429` |
| `bedMProp` | `:290` | `List Z := [-1, 1]` | the second builder, `make_propagate_batch` |
| `satOldTwo` | `:293` | `0 < a` | the **exponential-scan form's** domain — W3's |
| `satNewTwo` | `:296` | `0 <= abs a and abs a <= 1` | the **path product's** domain — W1's |
| `satSix` | `:300` | `0 < a and a < 1` | the softplus-gate family's image |
| `bedM_overlap_old_two` | `:302` | `bedM.countP satOldTwo = 1` `by decide` | **W3's domain meets BED-M at 1 of 3** |
| `bedM_overlap_new_two` | `:304` | `bedM.countP satNewTwo = 3` `by decide` | **W1's domain meets BED-M at 3 of 3** |
| `bedM_overlap_six` | `:306` | `bedM.countP satSix = 0` `by decide` | **0 of 3** |
| `six_misses_every_bedM_value` | `:321` | no `w` sends `exp(-softplus w)` to `-1`, `0` or `+1` | the 0/3 is not a counting artifact; it is the whole parametrized family |

**THE Q1 HEADLINE, and it is machine-checked in both directions.**

> **W1's exact class strictly contains W3's.**
> Containment on BED-M's support: **3/3 versus 1/3**, `bedM_overlap_new_two`
> (`:304`) against `bedM_overlap_old_two` (`:302`), both `by decide`.
> **Strictness**: `no_prefix_scan_represents_a_zero_gate` (`:165`), universally
> quantified over `C`, so the containment cannot be closed by any repair to the
> scan's parametrization.

The two excluded support points are the two that matter. `a = 0` is excluded by
`:165`. `a = -1` is excluded because a softmax row is nonnegative and sums to
one, so `y_i` is a convex combination and cannot carry a sign that alternates
with the window (§4, instanced §8). W3 represents BED-M's `+1` point and no other.

### 5.2 The second column, which is what the it.2 lesson actually was

`sixteen_is_silent_on_the_zero_draw` (`:355`) is in the file precisely to stop
this census being read as sufficient:

> *"A census that counts only hypothesis overlap passes this theorem; the second
> column — what the theorem constrains on the drawn value — is what fails it."*

So the counts above are reported **with** their second column:

| theorem | col 1: hypothesis overlap with BED-M | col 2: what it constrains on the drawn value |
|---|---|---|
| `pathProd_polar` `:105` | total (no hypothesis) | the factorization, on every draw including `0` and `-1` — **not decoration** |
| `pathProd_eq_zero_iff` `:129` | total (no hypothesis) | annihilation, exactly and only, at the `0` draw — **not decoration** |
| `pathProd_abs` `:121` | `0 <= m`: satisfied by the shipped `m`, whose closed cap is a property of the arithmetic at every switch value (`ceq/arm_smprime.py:120-127`) | the modulus law |
| `pathProd_eq_Wp` `:176` | `0 < m`: **fails at the `0` draw**, which is where BED-M's causal pairs concentrate | the coincidence with the scan form — **inapplicable on the majority of BED-M's pairs**, see §8 |
| `no_prefix_scan_represents_a_zero_gate` `:165` | total over `C`, plus the zero-window hypothesis, which BED-M supplies | a nonexistence — **the strongest column-2 content in the file** |

### 5.3 BED-K's column, and the hole in it

`delay_zero_is_first_order` (`lean/CEQ/V16Domain.lean:339`) records that
`ceq/beds/bed_k.py`'s `_delay_kernel_matrix` **rejects only `d < 0`**, so
`build_delay(n, 0, seed)` is a legal BED-K(a) cell, and at `d = 0` the delay is
a first-order recurrence. **The Hankel constant used in §3 is `1/d`. It is
undefined at `d = 0`, which is a legal cell.** The arena's registered cell is
`d = 20` (`CEQ_V20_R15_CONTRACT.md:116`), so nothing in Phase C trips this —
but the constant's domain has a hole in it and the census is what found it.
Recorded, not repaired.

---

## 6. Q3 SEED — RECORDED WITH ITS CITATION, NOT ANSWERED

`V20_R15_IT5_VENUS.md` §1 and `V20_R15_JOURNAL.md` (it.5 entry) both carry it:

> W1's seven failing cells are the predicted descent to the exact corner
> (`lambda_hat = -inf`, gate dead, landing at `0.88-0.93` inside the skyline's
> own neighbourhood). W3's failures are open-range divergence (`a_hat_max`
> blowing to `12.77 / 49.66 / 116.01`, `gate_r2` collapsing).
> **"A norm cap arrests M2; a cap does not undo M1's descent direction."**

**This is a learnability claim. It belongs in Q3 and it is not answered here.**

**What Q1 adds to the seed, and this is the only thing this iteration says about
it.** The asymmetry has a class-level restatement that Q1 now licenses:

> **W1's failures land INSIDE its own exact class; W3's land OUTSIDE its class.**
> `lambda_hat = -inf` with the gate dead is `m = 0`, which is a **representable**
> point of W1's class — `pathProd_eq_zero_iff` (`:129`) makes it an interior
> point, not a limit. W3's `a_hat_max -> 116.01` leaves the finite-`C` domain in
> which its `exp(C_i - C_j)` form is the class at all.
> A norm cap returns W3 **to** its class. A norm cap on W1 moves it **within**
> its class, and the corner is a point of that class, not an escape from it.

That sharpens the seed without answering it. Whether a cap changes the descent
direction is a landscape question and it is Q3's. **Not answered at it.6.**

---

## 7. F-GRADES AND HOW-BAD GAPS — THE FOUR FILLED CELLS

| cell | grade | HOW-BAD gap | replacement route |
|---|---|---|---|
| **Q1 / W1** | **F0** (hop) | **0 of 24** arena cells journal `beta`/`qk`/`route`/`g`, so no banked cell is attributable to the configuration the theorem covers | journal four scalar config fields per cell — **0 GPU-s** |
| **Q1 / W3** | **F1**, delta = instance tolerance | positive class statement has **no theorem**; `Lean #21` withdrawn at it.2, no replacement declaration exists | attempt a `Znorm` telescoping lemma at it.7, or carry F1 into the it.14 table |
| **Q2 / W1** | **F1 with the constant** | §8 | §8 |
| **Q2 / W3** | **F1 with the constant** | §8 | §8 |

**Standing gap on all four, named once and not repeated per row:** `lake build`
was **not re-run this iteration**. Every Lean citation above rests on WILSON's
it.1 verification (`V20_R15_IT1_WILSON.md:131,:137`) and on the file's contents
read at HEAD `207e7b9`, not on a green build observed at it.6. **Cheapest
killer: `cd lean && lake build; echo $?` — one command, no GPU.**

---

## 8. INSTANCES, RED-FIRST, AND THE MEASURED CONSTANTS

Nodes under `tests/jupiter/`. RED captured verbatim before green in every file.

### 8.1 L-DOM census node — `tests/jupiter/test_v20_r15_it6_ldom_census.py`

**RED, verbatim** (line `105` deliberately mutated to `106` for `pathProd_polar`):

```
FAILED tests/jupiter/test_v20_r15_it6_ldom_census.py::test_the_six_census_declarations_are_still_at_their_cited_lines[106-pathProd_polar]
E       AssertionError: (106, 'pathProd_polar', '    pathProd m θ i j = ((∏ k in Ico (j + 1) (i + 1), m k : ℝ) : ℂ)')
1 failed, 8 passed in 0.45s
```

**GREEN, verbatim:** `9 passed in 0.23s`.

### 8.2 THE CENSUS NUMBER THAT DECIDES §1 vs §2

`[READ]` `ceq/arm_smprime.py:23` — *"`126,976 / 131,072` entries and NO sequence
satisfies `forall k, 0 < m k`"*. That is **96.875%**, and
`ceq/arm_smprime.py:429` states the same figure rounded: *"on BED-M `96.9%` of
causal pairs annihilate"*.

**Therefore:**

> **`pathProd_eq_Wp` (`lean/CEQ/V16Domain.lean:176`), the theorem that says the
> path product COINCIDES with the exponential-scan form, has hypothesis
> `forall k, 0 < m k` — and it is satisfied by 0 of the 3 registered beds.**
> `ceq/kdata.py:473,:475,:480` register `bed_m`, `bed_k`, `bed_1`. On `bed_m`
> **no sequence at all** satisfies it (`ceq/arm_smprime.py:23`); `bed_k`'s field
> is `b = rng.standard_normal(n)` (`ceq/beds/bed_k.py:225`, signed and unbounded)
> and `bed_1`'s label is a committor probability — neither is a gate magnitude,
> so the hypothesis is not merely unmet but inapplicable.

**This is the it.2 shape exactly, and it is why §1 states Q1/W1 through
`pathProd_polar` and `pathProd_eq_zero_iff` rather than through
`pathProd_eq_Wp`.** M14 died by V-25 because Cheeger's hypotheses were satisfied
by 0 of the registered beds. `pathProd_eq_Wp` is in that same position — **0 of
3** — and if Q1/W1 had been stated on it, the F0 would have been decoration.
The two hypothesis-free declarations are **not** in that position: their column-2
content is non-vacuous on the very draw (`m = 0`) where the bed concentrates,
which is the test `sixteen_is_silent_on_the_zero_draw` (`:355`) exists to apply.

**Shipped `m` range, quoted:** `ceq/arm_smprime.py:110` — *"`m = clamp(u, 0, 1)`.
The X36 cap, CLOSED: `0` and `1` are attainable"*. `m` in `[0,1]` **closed**,
which is what lets the parametrization reach BED-M's `{-1, 0, +1}` at all.

**Full census table:**

| declaration | `path:line` | hypothesis | BED-M support `{-1,0,+1}` | shipped `m` in `[0,1]` closed | registered beds satisfying (of 3) |
|---|---|---|---|---|---|
| `pathProd_polar` | `:105` | none | YES | YES | 3/3 vacuously; **column 2 non-vacuous on `bed_m`** |
| `pathProd_abs` | `:121` | `forall k, 0 <= m k` | YES (`m = abs a` in `{0,1}`) | YES | 1/3 applicable, satisfied |
| `pathProd_eq_zero_iff` | `:129` | none | YES | YES | 3/3 vacuously; **column 2 non-vacuous on `bed_m`** |
| `pathProd_eq_Wp` | `:176` | `forall k, 0 < m k` | **NO** | **NO** (`0` attainable) | **0/3** |
| `no_prefix_scan_represents_a_zero_gate` | `:165` | `exists k, m k = 0` | **YES, and triggered on 96.875% of pairs** | YES | 1/3 applicable, **triggered** |
| `lean_log_junk_makes_the_scan_form_silently_false` | `:147` | none (concrete at `m == 0`) | YES | YES | demonstrates `bed_m`'s actual regime |

### 8.3 Q1 instances — `tests/jupiter/test_v20_r15_it6_q1_exact_class.py`

**RED, verbatim** (test 1's `assert diff <= TOL` temporarily replaced by
`assert diff <= 0.0`):

```
>       assert diff <= 0.0  # DELIBERATE-RED-PROBE: restored to `TOL` before shipping
E       assert 1.3877787807814457e-16 <= 0.0
FAILED tests/jupiter/test_v20_r15_it6_q1_exact_class.py::test_w1_path_product_matches_the_polar_closed_form_to_1e_6
1 failed, 3 passed in 2.04s
```

**GREEN, verbatim:** `4 passed in 1.96s`.

| instance | measured | bar |
|---|---|---|
| **Q1/W1** — `path_product` against `pathProd_polar`'s closed form, computed by an independent double loop, float64, S=16 | **`1.387779e-16`** | `<= 1e-6` **MET by 10 orders** |
| **Q1/W3** — `arm_pl.readout` against the `exp(C_i - C_j)` class member, `A` built by an explicit loop, QK exactly off | **`1.110223e-16`** | `<= 1e-6` **MET by 10 orders** |

**The tolerance and its justification.** The bar is the contract's
`<= 1e-6`. Both instances land at **float64 machine-epsilon scale**, so the
tolerance is not doing any work: nothing between `1e-16` and `1e-6` is being
absorbed by it. **Q1/W3's grade is F1 despite this**, because the number is a
measurement over one draw and Q1 asks for a theorem — see §2.3. A `1e-16`
instance is a very good instance; it is still not `zero error, by which theorem`.

### 8.4 Planted negatives, named and seeded

**PN-1 — `hop_scan`, the shipped planted negative** (`ceq/arm_smprime.py`),
S=16, `torch.manual_seed(0)`, `m[7] := 0.0` exactly.

- **63** entries whose window straddles index 7 have `G` **exactly** `0`.
- **45** later-pair entries have `G` **finite** while `hop_scan` is **non-finite**.

The moon found and corrected a flaw in this office's own probe design, and it is
recorded rather than smoothed over: on the *straddle* set `hop_scan` also reads
exactly `0`, because only `c_i` is `-inf` there and IEEE754 gives
`exp(-inf + i*finite) = 0` unambiguously. **That is a sentinel artifact, not the
scan representing the gate**, and a probe aimed there would have passed for the
wrong reason. The module's own docstring names the real region — *"every later
pair, not only the pairs that straddle it"* — where both `c_i` and `c_j` are
`-inf` and `-inf - (-inf) = nan`. **The assertion was re-aimed at the 45-entry
set.** `no_prefix_scan_represents_a_zero_gate` (`:165`) is what the 45 measure.

**PN-2 — `sign_flip_gate`, seed 0**, S=8, `a` in `{-1,+1}`, 200 random `(g,s)`
draws. Every draw's softmax row is nonnegative and sums to 1 at `1e-12`, so
`y_i` is confined to the convex hull of the values. **The sign-alternating target
escapes it at `i = 1`: `t_1 = -2.483118`, hull `[-1.398595, -1.084522]`.**
That is §4's constant, instanced: `dist = 1.084523`.

### 8.5 Q2 instances — `tests/jupiter/test_v20_r15_it6_q2_outside_bound.py`

**RED, verbatim** (the annex-comparison assertion deliberately conjoined with
`False`; the measured value on both sides is real):

```
E       AssertionError: measured R2_1=0.050000000000000 vs annex 1/d=0.050000000000000
E       assert (0.0 < 1e-06 and False)
FAILED tests/jupiter/test_v20_r15_it6_q2_outside_bound.py::test_m16_hankel_best_k_state_bound_on_the_delay_task_d20
1 failed, 3 passed in 1.81s
```

**GREEN, verbatim:** `4 passed in 1.32s`.

#### Q2 / W1 — THE ANNEX ANCHOR REPRODUCES EXACTLY

`[RUN]`, `ceq/hankel.py`, delay block `21 x 20`, `rank_real = 20`:

```
singular values (first 5) = [1.0, 1.0, 1.0, 1.0, 1.0]
R2_k, k = 1..8            = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
measured R2_1             = 0.050000000000000   annex 1/d = 0.050000000000000
```

**THE CONSTANT: `err_1 = sqrt(1 - R2_1) = 0.9746794345`** — the best relative
`L2` error achievable at `d = 20` with **one** state.

And the `1/d` law is exact rather than approximate:

| `d` | `R2_1` | `R2_1 * d` |
|---|---|---|
| 5 | `0.200000000000000` | `1.000000000000000` |
| 10 | `0.100000000000000` | `1.000000000000000` |
| 20 | `0.050000000000000` | `1.000000000000000` |

**max-min of `R2_1 * d` = `0.000e+00`.** `CEQ_V20_R15_CONTRACT.md:248`'s
`[RUN: 1/d for delays]` is **CONFIRMED to machine precision**. The number was
right; §9 shows only its attribution was wrong.

#### Q2 / W3 — THE NONEXISTENCE, WITH ITS WITNESS

`[RUN]` on the sign-alternating path-product series, `31 x 31` block,
minimum entry `-1.0`:

```
rank_plus_lower -> RankPlusBound(bound=NEG_ENTRY,
    method='negative entry: no nonnegative factorisation of any size',
    detail={'witness': (0, 2), 'value': -1.0})
rank_real(H) = 1        gap_ratio = inf     singular values = [31.0, 0.0, 0.0, 0.0]
```

**THE SEPARATION, and it is as sharp as this question gets:**

> **The multiplicative path-product series needs ONE ring-weighted state and NO
> number of nonnegative states.** `rank_R = 1`; `rank_+` does not exist at any
> size, with an exhibited witness at `H[0,2] = -1.0`.

**DISCREPANCY WITH THE ANNEX HINT, recorded rather than forced:** the annex
side expected `rank_R = 2`; measured is **1**, because `H[u,v] = f(u)*f(v)` is
an exact rank-1 outer product for a pure multiplicative series. **Smaller than
the hint, not larger.** The `NEG_ENTRY` impossibility is unaffected, and the
separation is *wider* than the annex thought, not narrower.

**The second constant for W3** is the convex-hull excess of §4, instanced in
§8.4: the seed-0 `sign_flip_gate` target escapes the hull at `i = 1`,
`t_1 = -2.483118` against `[-1.398595, -1.084522]`, **`dist = 1.084523`**.

**THE DELTA ON Q2/W1'S CONSTANT, stated because F1 means "with the constant"
and a constant carries its construction.** The `d = 20` block is built from the
impulse response — a baseline word plus one single-position flip per offset,
against one representative suffix per length `0..d-1` — not from full
enumeration over `2^d` words, which is intractable at `d = 20` and, approximated
at small `d`, is mean-dominated rather than flat. **`R2_1 = 1/d` is exact on
this construction and is not claimed beyond it.** The construction is the
delta; the `0.000e+00` spread across `d in {5,10,20}` is the evidence it is the
right one.

#### M3 EVEN-TARGET — THE CONSTANT, AND ITS PLANTED NEGATIVE FIRES

`[RUN]` `t(x) = x^2` on `x = linspace(-1, 1, 401)`, best nondecreasing `L2` fit
(isotonic, sklearn present):

> **`||t - iso(t)||_2 / ||t||_2 = 0.5303282490`** — that is M3's constant.

**PN-3 — `squared_feature`, seed 0:** refit as a monotone function of `x^2` and
the residual drops to **`1.031e-16`**. The annex's *"one squared feature => 0"*
**FIRES**: `0.530` to `1e-16`, and the unmutated residual is far above the
`0.1` floor the probe requires, so the planted negative is a real discriminator
and not a tautology.

---

## 9. THE CONTRACT-SCHEDULED CITATION RAISE, AND IT IS REFUSED

`CEQ_V20_R15_CONTRACT.md:247-249` states M16 with the bracket
**`(AAK/Glover [U -> V-eq at it.6])`** — the raise from UNVERIFIED to
VERIFIED-EQUIVALENT is scheduled for **this iteration**, and this office owns
the annex. **The raise is REFUSED, and the reason is a misattribution.**

### 9.1 Everything resolves. That is not the problem.

`[CITED]`, each DOI fetched and its returned title compared against the claimed
title (Scholar Sidekick returned `not subscribed` / `too many requests` on every
call; CrossRef `api.crossref.org/works/<doi>` was substituted and each record
re-fetched independently):

| citation | title | year | identifier |
|---|---|---|---|
| AAK | Analytic properties of Schmidt pairs for a Hankel operator and the generalized Schur-Takagi problem | 1971 | `10.1070/sm1971v015n01abeh001531` |
| Glover | All optimal Hankel-norm approximations of linear multivariable systems and their L-infinity error bounds | 1984 | `10.1080/00207178408933239` |
| Fliess | Matrices de Hankel | 1974 | **no DOI on record** (pre-DOI Gauthier-Villars); title corroborated via EUDML / Numdam / an AMS Bulletin review |
| Carlyle & Paz | Realizations by stochastic finite automata | 1971 | `10.1016/S0022-0000(71)80005-3` |
| Yannakakis | Expressing combinatorial optimization problems by linear programs | 1991 | `10.1016/0022-0000(91)90024-Y` |
| Eckart & Young | The Approximation of One Matrix by Another of Lower Rank | 1936 | `10.1007/BF02288367` |
| Mirsky | Symmetric Gauge Functions and Unitarily Invariant Norms | 1960 | `10.1093/qmath/11.1.50` |
| Schmidt | Zur Theorie der linearen und nichtlinearen Integralgleichungen (Teil I) | 1907 | `10.1007/BF01449770` |

**Eight resolved, title-matched, `[V-eq]`.** Items 3-5 are the three the
`ceq/hankel.py` docstring itself flags as *"upstream and unverified in this
repository"*; they are now resolved.

### 9.2 THE KILL — M16 cites the wrong theorem

**AAK and Glover state an OPERATOR-norm result: the minimal Hankel-norm error of
approximating a Hankel operator by one of rank <= k equals `sigma_{k+1}` — one
singular value, and the optimal approximant is constrained to be itself Hankel.**

**M16 computes `R2_k = SUM_{i<=k} sigma_i^2 / SUM_i sigma_i^2`, hence
`err_k >= sqrt(SUM_{i>k} sigma_i^2) / sqrt(SUM_i sigma_i^2)`. That is a
normalized FROBENIUS-norm truncation bound.** A sum of squared singular values
in both numerator and denominator is the signature of an energy statement.
**AAK/Glover's `sigma_{k+1}` bound contains no sum of squares at all.**

The correct source is **Eckart-Young-Mirsky** — Schmidt 1907 (compact
operators) / Eckart-Young 1936 (Frobenius) / Mirsky 1960 (the general
unitarily-invariant-norm statement containing both). All four resolved above.

**And the repo already said so, from an independent pass.**
`V20_R15_IT1_WILSON.md:591` records *"nothing in it is named for an AAK/Glover
bound"*, and `ceq/hankel.py`'s own reference list — Fliess, Carlyle-Paz,
Yannakakis, Cohen-Rothblum, Vavasis, Hrubes — never mentions AAK or Glover
anywhere. **The citation was attached to a contract line, not to any code or
theorem that uses the result.** That is the it.2 numbering failure in a second
costume: an authority named at the index, with nothing behind it at the source.

### 9.3 REPLACEMENT ROUTE — reprice, not retire

The bound is **correct**; only its attribution is wrong. M16 is not struck.

> **M16 HANKEL BOUND (L-BOUND).** Best-k-state fit
> `R2_k = SUM_{i<=k} sigma_i^2 / SUM sigma_i^2`
> (**Eckart-Young-Mirsky Frobenius-norm truncation `[V-eq]`**;
> Schmidt 1907 `10.1007/BF01449770` / Eckart-Young 1936 `10.1007/BF02288367` /
> Mirsky 1960 `10.1093/qmath/11.1.50`)
> `[RUN: 1/d for delays]`. **F1 by construction.**

**If the annex separately wants the true AAK/Glover operator-norm bound
`sigma_{k+1}`, that is a DIFFERENT quantity and needs its own annex line, its
own F-grade and its own producer.** Per the it.1 audit no producer for it exists
anywhere in this tree. It is not to be readmitted under M16's number.

**Only the author amends the contract.** This office files the refusal, the
correct attribution and the corrected line; it does not edit
`CEQ_V20_R15_CONTRACT.md`.

### 9.4 What this does to the dossier bar

Phase B's bar is *">=3 papers `[V-eq]`"*. **Eight are resolved and title-matched
above, five of them load-bearing for Q2** (Eckart-Young-Mirsky x3 for the
truncation constant; Yannakakis and Carlyle-Paz for the nonnegative-rank side of
§4). **The bar is met, and it is met by refusing the raise the contract
scheduled rather than by granting it.**


---

## 10. WHAT it.7 OWES, AND WHAT THIS ITERATION DID NOT DO

1. **Q3 for both wings**, from the seed in §6, with the class-level restatement
   Q1 now licenses. Ahead of it, the it.5 experiment — eight fresh
   `arm_smprime` seeds 8-15, ~129 GPU-s — is still first in the queue and this
   office did not run it (**KAGGLE / GPU launches require the author's explicit
   yes; none was sought or given at it.6**). `results/v20_r15_it6_seeds8_15.jsonl`
   and `tests/mercury/test_v20_r15_it6_seeds8_15.py` appeared in the tree during
   this iteration and are **another office's**; nothing in this report reads them
   and no claim here depends on them.
2. **The Q2/W1 constant**, §8.5. `ceq/hankel.py` SVD on `d in {5,10,20}`, no GPU.
3. **`lake build`**, one command, closing §7's standing gap.
4. **The author's ruling on M16's attribution** (§9.3). Only the author amends
   the contract; this office files the refusal and the corrected line.
5. **Q4-Q6**, unstarted, per §0.

**Not done and named as such:** eight of twelve Phase B cells; the Q2/W1
measurement; any GPU work; any git write; anything touching Kaggle.

## 11. SCOREBOARD

**Nothing claimed for the annex's `+4`.** That target is `>=12 of M1-M16 at
F0/F1 with M14 resolved`. This iteration **raised zero annex items** and
**refused one scheduled raise**. M16's bound survives at F1 with a corrected
source; its `[V-eq]` now rests on Eckart-Young-Mirsky rather than on a
misattribution, which is a repair, not a point.

**Dossier bar, Phase B:** `>=3 papers [V-eq]` — **8 resolved and title-matched**
(§9.1). `>=5 pages [V]` — the Lean pages carrying §1/§2/§5 are
`lean/CEQ/V16Domain.lean:105, :121, :129, :147, :165, :176, :251, :273, :288-312,
:339, :355` — **eleven declarations across the one file**, each read at HEAD
`207e7b9` and bound by `tests/jupiter/test_v20_r15_it6_ldom_census.py`.
**Q1 attempted in Lean AND instanced at `1.39e-16` / `1.11e-16`** — both routes,
where the contract asks for either.

**Carried: the it.5 figure, unmoved by this iteration.** Nothing here earns a
scoreboard item and nothing here is claimed to.
