# V20 R15 it.2 — JUPITER (MYCROFT)

Merge by primitive; the native skyline as cost floor; multipliers not arms; the
two struck claims bound; the annex's Lean numbers reconciled against WILSON.

Evidence classes are marked on every claim. `[RUN]` carries the command.
**WILSON is the reference and is not re-verified** — §E reconciles against him.

---

## A. THE MERGE BY PRIMITIVE

SATURN's N=3 is a list of three **modules** (`ceq/arm_smprime.py`,
`ceq/arm_phase.py`, `ceq/arm_pl.py`). Merged by primitive it is **not three**.

### A.0 The primitive, stated once

Every one of the three computes the same object: a **prefix-differenced
homomorphism of the token sequence into the multiplicative group `C*`**,

```
G_ij  =  Π_{k=j+1..i} a_k ,     a_k ∈ C*
```

`C*` factors as the **direct product** `C* ≅ R_{>0} × U(1)`, `a = m·e^{iθ}`, and
the factorisation is what the three modules disagree about — not the primitive.

### A.1 W1 vs W3 — **ONE primitive.** The algebraic fact: `cumprod = exp ∘ cumsum ∘ log`

`[DERIVED]`, steps shown. On the `R_{>0}` factor, `log` is a group isomorphism
`(R_{>0}, ×) → (R, +)`. Therefore for strictly positive magnitudes

```
Π_{k=j+1..i} m_k  =  exp( Σ_{k=j+1..i} log m_k )  =  exp( R_i − R_j ),   R = cumsum(log m)
```

W1's **path product as the hop** (`ceq/arm_smprime.py:144`, `def path_product`)
and W3's **real prefix scan** (`ceq/arm_pl.py:88`, `def scan(g)`) are the two
sides of that one identity, with `g := log m`. They are **not two primitives**;
they are one primitive in two coordinate systems, and `ceq/compat.py:393-394`'s
`cumsum` vs `cumprod` — the source argument SATURN's list rests on
(`V20_R15_JOURNAL.md:158`) — is a **coordinate difference, not an operator
difference.**

`[CITED]` MARS measured exactly this: `arm_pl(g=log m, s=0)` vs
`arm_smprime(θ=0, β=1)` at `1.110223e-16` / `5.551115e-16` against a control dial
move of `1.11e-01` / `2.37e-01` (`V20_R15_IT1_MARS.md` STRIKE 2 table; Inspector
C15 **clean**, four REDs at `house-events.jsonl:11356-11359` preceding the
finding). Fifteen orders of separation.

**Where they are genuinely two.** The isomorphism has a domain: `0 ∉ R_{>0}`.
`log 0` is undefined, so on any draw with `m_k = 0` the scan coordinate does not
exist and the two objects separate. This is not a footnote — it is the *only*
algebraic content distinguishing W1 from W3, and the tree already names it:
`lean/CEQ/V16Domain.lean:165`, `theorem no_prefix_scan_represents_a_zero_gate`,
and `:147`, `lean_log_junk_makes_the_scan_form_silently_false` `[READ]`. MARS's
shipped planted negative (`arm_smprime` `product` vs `exp_scan`,
`ceq/arm_smprime.py:89`) separates on the same set and nowhere else — `3.14e-16`
off it.

### A.2 W1 vs W2 — **ONE primitive on the interior; TWO only at a closed endpoint**

`[DERIVED]`. W1's hop is `Π m_k e^{iθ_k}`; under the direct product it splits as

```
G_ij = exp(R_i − R_j) · exp( i(Φ_i − Φ_j) ),      Φ = cumsum(θ)
```

The second factor **is** W2's gate `a = m·e^{iθ}` composed along the prefix
(`ceq/arm_phase.py:96`, `def gate(u, theta, *, cap=True)`). Machine-checked in
this tree: `lean/CEQ/V16Domain.lean:105` `pathProd_polar`, `:121` `pathProd_abs`,
`:176` `pathProd_eq_Wp` — *the path product equals the phase module's `Wp` on
`m > 0`* `[READ]`. That is the merge, already proved.

The one thing W2 adds is **not a primitive, it is a domain**: `m = clamp(u,0,1)`
makes `0` and `1` **attainable** rather than limits (SATURN's own clause,
`V20_R15_IT1_SATURN.md:21`). `m = 0` leaves `C*`. So:

> **W1 and W2 are one primitive wherever the primitive is defined, and two
> objects only on the boundary the clamp makes reachable.**

`[CITED]` MARS: `arm_phase(s=0)` vs `arm_smprime(β=1, product)` at `2.775558e-16`
/ `8.106339e-16` (STRIKE 2; C15 clean).

### A.3 W2 vs W3 — the two factors of the same direct product

`[DERIVED]`. W3 is the `R_{>0}` factor with `θ ≡ 0`; W2 is the same product with
the magnitude clamped. Neither is a primitive the other lacks. `[READ]`
`lean/CEQ/V15Phase.lean:216`, `theorem phase_modulus_is_the_real_carrier` — the
phase channel carries no magnitude; `:181-188` `C_re` / `C_im` / `C_sub` give the
real and imaginary prefix scans as *separate* cumsums.

**This pair is NOT measured and I do not assert it as bound.** MARS ran
phase↔smprime and pl↔smprime; he did **not** run pl↔phase. Identity is transitive
in exact arithmetic but the *tolerance* is not, and the derivation routes both
legs through the `β=qk=g=1` corner, which `V20_R15_JOURNAL.md:149` records the
trained models never occupy (`β` reads `0.588 … 1.509`, never `1`).

> **OPEN, and it is the it.3 owe:** one node comparing `arm_pl(g, s=0)` against
> `arm_phase(m=exp(g), θ=0, s=0)` directly, at the **trained** `β`, not at the
> corner. Until it exists the merge of W2 and W3 is `[DERIVED]`, not `[RUN]`.

### A.4 The merge, as a table

| pair | one primitive or two | **by what algebraic fact** | class |
|---|---|---|---|
| W1 ↔ W3 | **ONE** | `cumprod = exp ∘ cumsum ∘ log`; `log : (R_{>0},×) ≅ (R,+)` | `[DERIVED]` + MARS `[RUN]` 1.11e-16 |
| W1 ↔ W2 | **ONE** on `C*` | `C* ≅ R_{>0} × U(1)`; W1's `U(1)` factor **is** W2's gate. `lean/CEQ/V16Domain.lean:176` | `[DERIVED]` + MARS `[RUN]` 2.78e-16 |
| W2 ↔ W3 | **ONE** (the two factors) | W3 = the `R_{>0}` factor at `θ≡0`; W2 = the same with clamped `m` | `[DERIVED]` only — **OPEN** |
| **the genuine separator** | — | **the attainable zero.** `0 ∉ R_{>0}`, so `log` has no value there; `lean/CEQ/V16Domain.lean:165` | `[READ]` + MARS's planted negative |

**THE MERGE VERDICT.** By primitive the wing list is **N = 1 primitive carried in
three coordinate systems, plus one boundary** — the attainable zero. The three
"wings" differ by (i) which factor of `R_{>0} × U(1)` they retain, (ii) whether
the magnitude interval is closed, and (iii) the insertion site (W3 enters
key-side only over a value-zero BOS sink, `V20_R15_IT1_SATURN.md:22`) — and (iii)
is **wiring, not a primitive at all.**

This does not refute SATURN. It reprices his own flag: he wrote that distinctness
is *argued from source*, and the source difference he names (`cumsum` vs
`cumprod`) is the exact thing the isomorphism dissolves. **The it.4 freeze should
not freeze N=3 as three primitives.** It may freeze three *restrictions* of one.

---

## B. THE NATIVE SKYLINE AS COST FLOOR (not a contender)

**The task's citation is off by two lines and I am correcting it rather than
repeating it.** `V17_R4_RETAKE_PRICE.md:194` is `arm_smprime`, `:195` is
`arm_pl`. The softmax line is **`:196`** `[READ]`:

```
V17_R4_RETAKE_PRICE.md:192 | arm | s / 150 steps, mean of 2 | per-cell observations |
V17_R4_RETAKE_PRICE.md:194 | `arm_smprime` | **15.970** | 17.839, 14.101 |
V17_R4_RETAKE_PRICE.md:195 | `arm_pl`      | **1.614**  | 1.622, 1.605   |
V17_R4_RETAKE_PRICE.md:196 | `softmax`     | **1.497**  | 1.496, 1.497   |
```

**Units, verified before use** `[READ]` `:189-192`: **seconds per 150 training
steps**, mean of **2 seeds**, CUDA, from a six-cell single-process run (3 arms ×
seeds 0,1) with `t="wall" = 41.842 s`, process wall `44.533 s`. Not per step, not
per token. Per-cell overhead outside the training loop is **under 0.05 s/cell**
`[MEASURED by difference, :198-201]`, so the floor is the arm and not the harness.

### The cost-floor line

> **FLOOR: `softmax` = 1.497 s / 150 steps on CUDA** (`V17_R4_RETAKE_PRICE.md:196`).
> Spread across its two observations is `0.001 s` — the tightest of the three rows.

Two independent corroborations, neither substituted for the floor:

- `[READ]` `V17_R4_RETAKE_PRICE.md:229` — the fitted CUDA law `exp(−12.1852)·n^0.9963`
  predicts `1.524 s` against the measured `1.497 s`, ratio **0.982 (−1.8 %)**.
- `[READ]` `V16_DEVICE_CERT.md:1221` — per-arm multipliers on the softmax law,
  `softmax ×0.999`, `arm_pl ×1.042`, `arm_phase ×2.085`.

### What the floor prices

`[DERIVED]` from `:194-196`: `arm_smprime / softmax = 15.970 / 1.497 = **10.67×**`;
`arm_pl / softmax = 1.614 / 1.497 = **1.078×**`. SATURN reports the same figure
against `arm_pl` as **10.68×** (`V20_R15_IT1_SATURN.md:187-191`) and puts W1 at
**9.90× the incumbent, a 99× gap** to the 0.1× target.

**The floor is listed as a floor.** Softmax is not entered as an arm, is not
scored, and does not compete for the wing list. It is the number every arm's price
is quoted against. The north-star sentence — *attention equal to self-attention on
its own ground* — means the floor is what an arm must **reach**, so an arm cheaper
than 1.497 s that does not also cross BED-M's floor₁ `0.7071` (`V15_R1.md:250`,
`crosses? NO`) has bought nothing.

---

## C. MULTIPLIERS, NOT ARMS

A **multiplier** is a method-of-training or instrumentation choice that any arm
may take and that changes an arm's reading without being an operator. Each is
cited or it is not listed.

| # | multiplier | what it multiplies | citation `file:line` | status |
|---|---|---|---|---|
| **X1** | **Interventional channel** — per-position `do()` bumps with oracle-measured outcomes | the *training draw*; `ζ > 0` vs `ζ = 0` at fixed `n` | design `CEQ_V16_CONTRACT.md:179`; earlier form `CEQ_V15_CONTRACT.md:188`; **implemented** `ceq/beds/bed_k.py:35, :245, :261`; corpus `ceq/corpus.py:1, :89`; driver `ceq/diagnose.py:121` | **LIVE — takeable by any arm today** |
| **X2** | **Interventional ablation R7 / R6** — the experiment that prices X1 | prediction: **≥2×** at fixed `n` | `CEQ_V16_CONTRACT.md:230`; `CEQ_V15_CONTRACT.md:224` | **specified, not run this round** |
| **X3** | **`bump` vs `rebuild`** — two independent routes to the same intervention | exactness of the channel | `ceq/beds/bed_k.py:35` | **LIVE, and it is its own control** |
| **X4** | **Device / arm cost multiplier** on the fitted softmax law | *cost*, not capability: `softmax ×0.999`, `arm_pl ×1.042`, `arm_phase ×2.085` | `V16_DEVICE_CERT.md:493, :498, :1221`; `V16_PRICING.md:158-159` | **LIVE — see `Limits`** |
| **X5** | **X₃₅′ hidden-cause detector** — an *instrument* for the label class, not a primitive | what an arm is *measured with* | reclassified by SATURN himself at `V20_R15_IT1_SATURN.md:76`, under contract `:87-88` | **reclassify, do not close** |
| **X6** | **Curricula** | — | **`CEQ_V20_R15_CONTRACT.md:87` and nowhere else** | **NAMED, NOT IMPLEMENTED** |

**X6 is the finding in this section.** `[RUN]` `grep -rin "curricul" .` over the
whole tree returns **exactly one line**, and it is the contract sentence that
names curricula as a multiplier. There is no generator, no scheduler, no test, no
result. **Listing it as a takeable multiplier would be a `GUESS`.** It is listed
here as *named by the contract and unimplemented*, which is what can be cited.

**Everything above is a multiplier, and none of them is an arm.** They do not
change `G_ij`; they change the draw the arm sees (X1, X2, X3, X6), the price the
arm is quoted at (X4), or the instrument reading it (X5). Contract `:87-88` is
therefore satisfiable as written — with one entry hollow.

---

## D. TASK B — THE TWO STRUCK CLAIMS, BOUND

New file: `tests/jupiter/test_v20_r15_it2_ldom_census.py` (**9 nodes**).
Producer patched: `scripts/v20_m14_cheeger.py`, new `_h1_h4` predicate and
`domain_census_facts()`, added to `__all__`.

### D.1 THE VERBATIM RED, against the tree as it stands

The census **was** already computable and true — the producer had it. It was
struck because the producer returns a **formatted string**, and a string is not an
assertion. So the node was written **first**, against a producer entry point that
did not exist, and the RED is the tree refusing it:

```
$ python -m pytest -q tests/jupiter/test_v20_r15_it2_ldom_census.py

ERROR collecting tests/jupiter/test_v20_r15_it2_ldom_census.py
tests\jupiter\test_v20_r15_it2_ldom_census.py:30: in <module>
    from scripts.v20_m14_cheeger import (  # noqa: E402
E   ImportError: cannot import name 'domain_census_facts' from
    'scripts.v20_m14_cheeger' (C:\Users\seal\Desktop\New folder (32)\scripts\v20_m14_cheeger.py)
=========================== short test summary info ===========================
ERROR tests/jupiter/test_v20_r15_it2_ldom_census.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.59s
```

All nine nodes RED. Logged to `house-events.jsonl` as
`{"t":"test","agent":"Jupiter","status":"red",...}` **before** the producer was
touched.

### D.2 THE GREEN

```
$ python -m pytest -q tests/jupiter/test_v20_r15_it2_ldom_census.py -rf
.........                                                                [100%]
9 passed in 2.78s

$ python -m pytest -q tests/jupiter/
101 passed in 43.74s
```

`tests/jupiter/` read **92 passed** at it.1 (Inspector C2) and reads **101** now:
`+9`, nothing broken by the producer patch.

### D.3 C21 — the L-DOM census, now asserted as numbers

| node | asserts |
|---|---|
| `test_ldom_census_bed_m_satisfies_the_hypotheses_on_zero_of_384_draws` | `n_draws == 384`, `H1 is False`, **`0`** satisfying |
| `test_ldom_census_the_ternary_object_satisfies_them_on_zero_of_2048_draws` | `n_draws == 2048`, support ⊆ `{−1,0,+1}`, `negative_entries > 0` (so no `π ≥ 0` exists), **`0`** |
| `test_ldom_census_bed_k_is_strictly_lower_triangular_and_nilpotent_on_all_8` | `n_kernels_swept == 8`, **`strictly_lower_triangular is True`**, **`nilpotent is True`**, `row_sums_are_one is False`, **`0`** |
| `test_ldom_census_verdict_zero_registered_beds_...` | `registered_beds_satisfying_H1_H4 == 0`; Rips `== 2`; `Rips.is_a_registered_bed is False` |
| `test_planted_negative_the_census_predicate_can_return_nonzero` | **V-16** |

**PLANTED NEGATIVE, named exactly.** The zero is a *reading*, not a constant,
because the **same function** `_h1_h4` is applied to every object, and on the Rips
chain — the one object the census reports as satisfying all four — it returns
**nonzero (2)**. A predicate that returns `0` on three inputs and `> 0` on a
fourth cannot be a hardwired zero. The node also asserts
`_predicate_is_the_same_function is True`, so the shared-predicate property is
itself bound and cannot be silently refactored away.

`0` is **not** the parameters' fault: BED-K is **swept** over
`(delay d=1, delay d=4, powerlaw H=0.6, powerlaw H=0.9) × n ∈ {32,128}` = 8
kernels, all strictly lower triangular ⇒ `K^n = 0` ⇒ **nilpotent** ⇒ **not
irreducible**, so H3 fails structurally, and `row_sums_are_one is False` fails H2
independently. Two hypotheses out on every kernel by construction.

**The consequence, which the journal asked for and can now be written.** With
Cheeger's hypotheses satisfied by **0 of the campaign's registered beds**
(`ceq/kdata.py:472-482` registers three: `bed_m`, `bed_k`, `bed_1`), **M14 dies by
V-25 rather than by its own kill.** The annex's own kill
(`CEQ_V20_R15_CONTRACT.md:262-263`, "cited before it passes its own check") does
not fire — the check passes, 43/43. V-25 does. **The theorem is machine-true and
domain-empty**, and those are different deaths that were being scored as one.

### D.4 C20 — the two-sided node

The struck node asserts `crude >= exact - TOL` only
(`tests/jupiter/test_m14_cheeger.py:243`). Four nodes replace it:

- `test_the_repo_crude_phi_EQUALS_the_exact_minimum_two_sided` — **both** sides
  plus `crude/exact == approx(1.0, abs=TOL)`.
- `test_the_repo_crude_phi_is_the_planted_rational_one_over_twentyseven` —
  `exact == approx(1/27)` **and** `crude == approx(1/27)`.
- `test_planted_negative_the_two_sided_node_fires_when_crude_is_inflated` —
  **the mutation is named and exact: `crude * 10.0`**, the very counterexample C20
  names. The one-sided assertion **still passes** under it; the upper side raises.
  The node asserts the `AssertionError` via `pytest.raises`, so the strike's own
  sentence is now a green test.
- `test_planted_negative_the_sweep_path_is_not_hardwired_to_the_exact_path` —
  **the permutation is seeded and named:
  `numpy.random.default_rng(20152).permutation(n)`**, applied as `W[ix_(p,p)]`.
  Both paths must still return `1/27`; the sweep's reported *ordering* must change,
  and non-identity of the permutation is asserted inside the node. (This is the
  mistake the Inspector struck at C7 — "a scrambled ordering" specifies no
  permutation, so the count did not reproduce. Seed `20152`, generator named.)

**TOLERANCE.** `TOL = 1e-11`, **inherited unchanged** from
`tests/jupiter/test_m14_cheeger.py:72`, whose docstring `:37-46` derives it as
`>5×` the dominating error source on these same matrices. Reusing the existing bar
rather than choosing one means the file cannot be accused of picking a tolerance
that admits its own reading. The census assertions use **no float tolerance at
all** — they are integer counts and booleans.

### D.5 Nothing withdrawn

Both struck claims bind. No withdrawal is filed under Task B.

---

## E. TASK C — THE ANNEX LEAN NUMBERS, RECONCILED AGAINST WILSON

**WILSON §2 is the reference and is not re-verified.** `lean/` builds (toolchain
`leanprover/lean4:v4.7.0`; `lakefile.lean` requires mathlib at `v4.7.0`) `[READ]`,
so this is checkable rather than a matter of opinion.

`[RUN]` a declaration census over `lean/` (13 `.lean` files, excluding
`.lake/packages`) confirms the shape of Wilson's finding without re-deriving it:
the numbered comments in `lean/` run **#1 … #16, #5a, #5b**. **No declaration or
comment anywhere in `lean/` mentions #17, #17′, #18, #19, #20, #21, #22 or #23**,
and **no file under `lean/` contains the word "annex".**

| annex item | number | assignment | reconciliation | verdict |
|---|---|---|---|---|
| M1 | **#18 [M]** | `CEQ_V20_R15_CONTRACT.md:179`, share-weighted mixture, "decides corner usage" | The corner theorems that exist — `lean/CEQ/V16Domain.lean:394` `corner_softmax`, `:401` `corner_linear`, `:408` `corner_path_product`, `:433` `three_corners_containment`, `:445` `corners_are_distinct` — are **#5a's** three corners (`CEQ_V16_CONTRACT.md:97`). **None states a share-weighted mixture.** | **CITATION WITHDRAWN.** #18 is a target, not a declaration. |
| M2 | **#19 [M]** | `CEQ_V20_R15_CONTRACT.md:185`, `1.3e-3`, "w still growing at 400 steps" | No declaration in `lean/` concerns a training-step trajectory. Compounded: SATURN K6 and WILSON §7 both read `1.3e-3` as **NOT FOUND IN TREE** (Inspector C24, clean) — so the *number* has no producer and the *theorem* has no declaration. | **CITATION WITHDRAWN**, and the item is doubly unbacked. |
| M9-F0 | **#22 [M]** | `CEQ_V20_R15_CONTRACT.md:250` | **THE DECLARATION EXISTS UNDER A DIFFERENT NAME.** M9-F0's content is the X35′ source solve, `(a) h_hat = (I − A) r [RUN 8.9e-16, two sources]` (`ceq/x35p/source.py:7`; WILSON §7 M9-F0 — **FOUND, with a live producer**). Its machine-checked statement is `lean/CEQ/V15Source.lean:128`, `theorem two_sources_recovered`, with `:108` `source_is_first_order_difference`, `:188` `source_entry`, `:143` `no_fill_in`. | **RECONCILED.** Cite `lean/CEQ/V15Source.lean:128` **by name, not by number.** |
| — | **#17′ [S]** | `CEQ_V20_R15_CONTRACT.md:188` | no declaration | **WITHDRAWN as a citation**; survives as an `[S]` target |
| — | **#20 [S]** | `:192` | no declaration | **WITHDRAWN as a citation** |
| — | **#21 [S]** | `:202` | no declaration | **WITHDRAWN as a citation** |
| — | **#23 [S]** | `:219`, "the union-bound mask theorem" | The mask theorems that exist — `lean/CEQ/V15.lean:128` `prefix_logit_mask`, `lean/CEQ/V16Domain.lean:221` `prefix_logit_mask_restated` — are **prefix-logit masks, not union bounds**. Different proposition. | **WITHDRAWN as a citation** |

**The distinction that makes six withdrawals honest rather than destructive.**
`[M]` numbers are assignments of work; `[S]` numbers are stretch targets. Neither
tag entitles the annex to *cite* a number as though it names a machine-checked
declaration, and `CEQ_V20_R15_CONTRACT.md:250`'s "LEAN LEDGER this round" line
reads as if all three `[M]` numbers were ledger entries. **One of the three is
(#22, under the name `two_sources_recovered`); two are not.**

**REPLACEMENT ROUTE for the six withdrawals** (every kill ships one):

- **REROUTE #22** — cite `lean/CEQ/V15Source.lean:128` by declaration name. Costs
  nothing, is checkable by `lake build`, and survives any renumbering. **The round
  should cite declarations by name and drop the numbering scheme entirely** — the
  numbers are the failure mode: eleven of them resolve to nothing.
- **RETIRE #18, #19** — as *citations*. Keep them as assignments, in the schedule
  column, with the word "target" instead of a `#`.
- **RETIRE #17′, #20, #21, #23** — `[S]` already means unproved; the fix is to stop
  printing them next to `[M]` numbers in the same ledger sentence.

I do not claim a Lean number for M14. `V20_R15_IT1_JUPITER_M14.md` §8 declined the
ledger this round and §D.3 above is the reason it should stay declined: the
theorem is true and its domain is empty.

---

## Limits

- **W2 ↔ W3 is `[DERIVED]`, not `[RUN]`.** MARS measured phase↔smprime and
  pl↔smprime; nobody measured pl↔phase. Identity is transitive; the *tolerance* is
  not, and both measured legs route through the `β=qk=g=1` corner that
  `V20_R15_JOURNAL.md:149` records the trained models never occupy. §A.3 names the
  exact node it needs.
- **The whole §A merge is an algebra claim resting on MARS's measurement,** which
  the Inspector cleared (C15) but which was taken **at a corner**, not at trained
  settings. The it.4 freeze blocker SATURN named — extend `assert_arms_distinct`
  over W1/W2/W3 at *trained* configurations — is **not discharged by this report**
  and remains the blocker.
- **§B's floor is `N = 2`.** Per-arm seconds are the mean of two seeds
  (`V17_R4_RETAKE_PRICE.md:318` states this itself), and `:337` records that the
  throughput law was **not verified past 150 steps**. The floor is a floor at one
  ladder rung, on one device.
- **X4 is not carried forward by the tree's own later work.**
  `V17_G06_G07_CERT_COST.md:199-200` states the V16 arm multipliers are **not**
  carried and that `arm_smprime` is fitted directly, because a multiplier measured
  at one shape is a claim. It is listed as a multiplier *class*, not a live
  constant.
- **X6 (curricula) is hollow.** One citation, to the contract sentence naming it.
- **§D.3's V-25 consequence is a consequence, not a grade.** M14's grade stays
  where it.1 put it (**F1, HOW-BAD 108×**); this report supplies the assertion the
  journal said had to exist before the annex is graded, and does not re-grade.
- **`domain_census()` was not rewritten** on top of `domain_census_facts()`; the
  formatted string keeps its own arithmetic. Marked in-file with a `ponytail:`
  comment. Merge them when `results/v20_m14_cheeger.txt` is next regenerated —
  until then the string and the facts are two computations of one census, which is
  a small V-7 risk and is named here rather than left silent.
- **No git write, no Kaggle contact.** Working tree modified: one new test file,
  one patched producer, `house-events.jsonl` appended.
