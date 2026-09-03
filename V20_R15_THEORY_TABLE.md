# V20 R15 — THE THEORY TABLE, FROZEN AT it.14

**N = 2 wings × Q1–Q6. Twelve cells. `1×F0 / 5×F1 / 1×F2 / 2×F3 / 3×F4`.**

W1 = `arm_smprime` · W3 = `arm_pl`. Both are **BED-M** arms. The wing list froze at it.4
(`V20_R15_WING_MANIFEST.md:3`).

**This file has two readers and is written for the second.** The arena (it.15–30) reads it
as a ticket and can ask a follow-up question. **The leap model (it.35) reads it as its
primary input, gets ONE call, and cannot.** So every defect the round knows about is
carried *inside* a cell rather than in prose around the table, and every claim that would
send the leap chasing a ghost is marked in the cell that would have sent it.

**Three things a reader must know before reading a single grade.** They are §0.

---

## §0 — WHAT THE GRADES ARE MEASURED WITH, AND WHY THAT IS NOT SAFE

### 0.1 THE SCALE HAS NO TEXT. THE RUBRIC BELOW IS A RECONSTRUCTION FROM USAGE.

`L-GRADE (F0–F4 + HOW-BAD gap)` is cited once, at `CEQ_V20_R15_CONTRACT.md:58`, on the
**standing** side of `LAWS: all standing +` — and **no canonical F0–F4 rubric exists
anywhere in this repository.** The Inspector affirms the finding and strikes its
instrumentation: *the narrow claim is BOUND, the generalization is UNBOUND — including
when I made it* (`V20_R15_IT12_INSPECTOR.md:54`). Two reconstructions were filed. **This
office's was ruled BOUND** to its stated usage and to no other
(`V20_R15_IT12_INSPECTOR.md:247`); **SATURN's was ruled NOT BOUND**, refuted by eight
counterexamples he supplied himself (`V20_R15_IT12_INSPECTOR.md:233`).

**The rubric the twelve grades below were actually assigned under**, banked as an event on
the day it was used, self-labelled a reconstruction (`house-events.jsonl:12784`,
`V20_R15_IT12_JUPITER.md:173`):

| token | as used in this round | NOT |
|---|---|---|
| **F0** | exact, no tolerance, carried by a machine-checked declaration | — |
| **F1** | a bound **with its constant** on the graded object | matches `L-CERT`'s mask sense (`CEQ_V20_R15_CONTRACT.md:64`), which is a *different* object |
| **F2** | a direction or an ordering, **no constant** | — |
| **F3** | the claim is a fact about the **harness**, not about the theory | SATURN's *"failed instance"* |
| **F4** | the theory's **domain is empty** for the graded wings | SATURN's *"unattempted"* — M16 was attempted, run, and reproduced to machine precision |

**THE COUNTEREXAMPLES TO THIS RUBRIC, NAMED, NOT BURIED:**

1. **`F4` is overloaded three ways** — *unattempted* (`V20_R15_LEAP_LEDGER.md:28`, L-7),
   *withdrawn* (`V20_R15_LEAP_LEDGER.md:25`, L-4), *domain-empty*
   (`V20_R15_LEAP_LEDGER.md:24`, L-3). **A reader cannot tell which from the token.** The
   rubric above covers only the third sense; the ledger uses all three.
2. **`F2` and `F3` are not ordered.** F3 as used is not "worse than F2", it is a
   *different kind* of failure. **Twelve grades were assigned on a five-token scale that is
   not a scale** (`V20_R15_IT12_JUPITER.md:189`).
3. **`M14` carries three grades in three files for one item** — `F3` in the contract
   (`CEQ_V20_R15_CONTRACT.md:241-244`), `F4` in the ledger (`V20_R15_LEAP_LEDGER.md:24`), and
   `F1` in the journal (`V20_R15_JOURNAL.md:645`). Confirmed and enumerated at
   `V20_R15_IT12_INSPECTOR.md:260`. **RESOLVED at it.17 — `M14 = F4`.** The `F1` is an it.3 SCOREBOARD reading on the far side of the it.4 freeze (`V20_R15_JOURNAL.md:650`); the contract's `F3` is pre-registration, self-conditioned; `L-3` is the live verdict (RULING J-17d).
4. **An `F4` row carries a measured constant.** `V20_R15_LEAP_LEDGER.md:131` (L-14) is
   graded `F4` and reports `1.421901019003236` from a live run. Under *any* reading where
   F4 means "unattempted", that row is a contradiction.
5. **This office filed two contradictory grades for one object** (Q2/W1 at it.6 and it.8)
   **and neither report noticed** (`V20_R15_IT12_JUPITER.md:229`). No node compared grades
   across reports until it.12.

> **INSTRUCTION TO THE LEAP.** Read the twelve grades as **ordinal within a column and
> nominal across columns.** `F1 → F0` on one cell is a real improvement. `F2` vs `F3` on
> two different cells is **not a comparison** and must not be treated as one. Do not spend
> the one call deriving a rubric; the rubric's absence is a **0 GPU-second contract edit**
> (`V20_R15_IT12_JUPITER.md:215`) and is not leap material.

### 0.2 `floor₁` IS NOT AN INFORMATION FLOOR, AND EVERY Q5 CELL SAYS WHICH FLOOR IT MEANS

`floor₁ = sqrt((t*−1)/t*)`, computed @ `scripts/v15_r1.py:586` from `T_STAR = 2` @
`scripts/v15_r1.py:138`, evaluating to `0.7071067811865476` — **a value no line of that file contains**; identity @ `scripts/v15_r1.py:17`. It is defined by
`h_hat = t*(1 − NRMSE²)`, reaching `h_hat = 1` exactly at `NRMSE = floor₁`. **It is a
one-hop CAPABILITY THRESHOLD.** A lower bound is never violated; this one is.

**`[RUN]` re-measured this iteration on the current record, not carried from it.8** —
`python -c "<dedupe results/v17k_r4_retake.jsonl + results/v20_r15_it6_seeds8_15.jsonl +
results/v20_r15_it8_armpl_b.jsonl on (kind,seed); count eval_h_hat > 1.0>"`:

| quantity | it.8 figure, now stale | **measured at it.14** |
|---|---|---|
| banked cells carrying `eval_nrmse` | 34 | **40 unique** (43 rows; **3 exact re-emissions** — `arm_pl` seed 0, `arm_smprime` seeds 0 and 1, identical `eval_nrmse` to 16 digits under one `instrument_hash`) |
| cells **violating** `floor₁` | 6 of 34 | **13 of 40** — 12 `arm_pl`, 1 `arm_smprime` |
| `s = 64` | 34 of 34 | **40 of 40** |
| `dist_to_skyline` populated | 0 of 34 | **0 of 40** |

**Of four candidate floors, ZERO bind both frozen wings as information floors.** M11 Fano
and M12 rate–distortion: **0 of 3 beds, 0 producing `.py`** — prose. M16 Hankel: **BED-K
only**, and both wings are BED-M. `floor₁`: BED-M, and not a floor. **The real information
floor on BED-M is the exact oracle at `0.0`, and no annex theorem predicts either wing's
distance to it.**

### 0.3 Q4's `F3` IS A FACT ABOUT THE HARNESS, NOT A FAILURE OF EITHER WING

Neither wing failed a cost law. **No cost law was ever measurable.** `S, D = 64, 24` is a
**module constant** at `scripts/v15_r1.py:137`; there are **nine sibling argparse flags**
at `scripts/v15_r1.py:547-558` and **zero override paths** for `S`; `s = 64` on **40 of 40**.
The independent variable has `n = 1`, so the exponent in `S` is unidentified — not
poorly estimated, **unidentified**. Compounding, and separately: the per-cell timer is
un-synchronised CUDA host wall clock (`scripts/v15_r1.py:249`, `scripts/v15_r1.py:267`) and
the strongest correlate of `secs` is **run order** (`ρ = +0.7029`), above the gate
correlation `+0.5197`.

> **INSTRUCTION TO THE LEAP.** Q4's two `F3`s are the same fact about `scripts/v15_r1.py`
> written twice. **They are not two failures and not evidence about the primitives.**
> Repair is one `argparse` line plus two `torch.cuda.synchronize()` calls, priced at
> **206–537 GPU-s** (`V20_R15_IT13_MERCURY.md:146`) — and MERCURY's limit stands: the
> run-order confound survives `synchronize()` and **the randomised-execution edit that
> would remove it has never been priced by any office.**

---

## §1 — THE TABLE AT A GLANCE

| | **W1 `arm_smprime`** | **W3 `arm_pl`** |
|---|---|---|
| **Q1 EXACT CLASS** | **F0** | **F1** |
| **Q2 OUTSIDE** | **F4** (re-graded it.12) | **F1 + const** |
| **Q3 LEARNABILITY** | **F2** | **F1 + const** |
| **Q4 COST LAW** | **F3** | **F3** |
| **Q5 INFORMATION FLOOR** | **F1 + const** | **F1 + const** |
| **Q6 STATE METRIC** | **F4** | **F4** |

---

## §2 — THE TWELVE CELLS

Each cell carries **GRADE · GAP · DECLARATION (by name and `path:line`, never by annex
number) · CENSUS · ROUTE · ARENA**.

### CELL Q1/W1 — EXACT CLASS, `arm_smprime`
- GRADE: F0
- GAP: The theorems cover the **HOP**; they do not cover the **ARM**. `0 of 24` arena cells journal `beta`, `qk`, `route` or `g`, so **no banked cell is attributable** to the configuration the theorem is exact on. Instance `1.387779e-16`.
- DECLARATION: `pathProd_polar` @ `lean/CEQ/V16Domain.lean:105` and `pathProd_eq_zero_iff` @ `lean/CEQ/V16Domain.lean:129` — both hypothesis-free; supported by `pathProd_abs` @ `lean/CEQ/V16Domain.lean:121`, `bedM_gate_exact` @ `lean/CEQ/V16Domain.lean:251`, `negative_draw_is_on_the_band` @ `lean/CEQ/V16Domain.lean:273`. Instance @ `tests/jupiter/test_v20_r15_it6_q1_exact_class.py:*`.
- CENSUS: **3 of 3** BED-M support points `{−1,0,+1}`, by `bedM_overlap_new_two` @ `lean/CEQ/V16Domain.lean:304` (`by decide`). **EXCLUDED and stated:** `pathProd_eq_Wp` @ `lean/CEQ/V16Domain.lean:176` is **0 of 3 registered beds** and **0 of 2048 rows** satisfy its hypothesis `∀k, 0 < m k` — cause at `scale/negation_scope.py:429`. It is not cited in this cell.
- ROUTE: journal `beta`/`qk`/`route`/`g` as four scalars per cell — **0 GPU-s**. This is the whole repair; it is bookkeeping, not theory.
- ARENA: bears on clause (1) and clause (3). Scoreable on both.

### CELL Q1/W3 — EXACT CLASS, `arm_pl`
- GRADE: F1, δ = instance tolerance
- GAP: **The positive class statement has NO THEOREM.** `Lean #21 [S]` was withdrawn at it.2 — this office withdrew six citations rather than defend them — and **no replacement declaration exists.** The class is carried by a `≤1e-6` numerical instance (measured `1.110223e-16`) and a code-level derivation. **An instance at `1e-6` is not zero error**, which is why this is F1 and not F0.
- DECLARATION: **negative boundary only** — `no_prefix_scan_represents_a_zero_gate` @ `lean/CEQ/V16Domain.lean:165` (universally quantified over `C`), `lean_log_junk_makes_the_scan_form_silently_false` @ `lean/CEQ/V16Domain.lean:147`, `sixteen_is_silent_on_the_zero_draw` @ `lean/CEQ/V16Domain.lean:355`. Positive side: `scan` @ `ceq/arm_pl.py:88`, `key_bias` @ `ceq/arm_pl.py:93`, telescoping `normalizer` giving `Z_i = 1` exactly. **No Lean declaration states the positive class. Do not read one into this cell.**
- CENSUS: **1 of 3** BED-M support points — W3 represents `+1` only, by `bedM_overlap_old_two` @ `lean/CEQ/V16Domain.lean:302` (`by decide`).
- ROUTE: a telescoping lemma for `Znorm` @ `lean/CEQ/V16Domain.lean:370` over `CEQ.V15.scan`, shape *"at `s_j − C_j` with `qk == 0`, `Znorm g qk i = 1`"*. Proposed at it.6, **never attempted at it.7–it.13.**
- ARENA: bears on clause (1) and clause (3). Scoreable on both.

### CELL Q2/W1 — OUTSIDE THE CLASS, `arm_smprime`
- GRADE: F4 — **re-graded from `F1 + const` at it.12; the cell loses its `+ const`**
- GAP: **The bound was graded on the wrong bed.** The `1/d` Hankel law binds **BED-K only** and W1 is BED-M, so the domain intersection is **EMPTY**. This office filed two contradictory grades for one object (it.6 `F1 + const`, it.8 `F4 for these wings`) and neither report noticed. **The it.8 row is correct and the it.6 grade is withdrawn** (`V20_R15_IT12_JUPITER.md:135`). **No BED-M bound was manufactured to save the cell.**
- DECLARATION: `hankel_block` @ `ceq/hankel.py:108`, `rank_real` @ `ceq/hankel.py:131`, `rank_plus_lower` @ `ceq/hankel.py:279`, `NEG_ENTRY` @ `ceq/hankel.py:75`, imported at `tests/jupiter/test_v20_r15_it6_q2_outside_bound.py:43`; the `1/d` curve is computed **only inside that test**, `tests/jupiter/test_v20_r15_it6_q2_outside_bound.py:75`. Constant `err_1 = 0.9746794345`, bound to `5e-11` @ `tests/jupiter/test_v20_r15_it12_constants.py:12` — **and that node states its own weakness: the block is `I_d` with a zero row, which is arithmetic and not evidence.** The annex's AAK/Glover authority was **REFUSED** at it.6; the correct source is Eckart–Young–Mirsky.
- CENSUS: **1 of 3 beds (BED-K).** `[RUN]` — *"Q2 domain census: BED-K only; intersection with W1/W3 domain = EMPTY"*. `1/d` is exact on `d ∈ {5,10,20}` (max−min of `R²₁·d` = `0.000e+00`), and has a **domain hole at `d = 0`**, which is a legal BED-K cell — `delay_zero_is_first_order` @ `lean/CEQ/V16Domain.lean:339`.
- ROUTE: **none on this wing.** The `1/d` law stays `F1` **on BED-K**, where nothing this round is graded. Repairing the cell means running a BED-K cell, and **zero cells of BED-K's shape have ever been run** (`V20_R15_IT13_MERCURY.md:63`). Ledger-level repair: a node asserting each object carries exactly one live grade.
- ARENA: **this is the cell that voids clause (2).** See §3.

### CELL Q2/W3 — OUTSIDE THE CLASS, `arm_pl`
- GRADE: F1 + const — **kept at it.12; the `rank₊` half struck**
- GAP: W3's bound is a **NONEXISTENCE**, not a truncation. A softmax row is `A_ij ≥ 0` with `Σ_j A_ij = 1`, so `y_i` is a convex combination of `{b_j : j ≤ i}` for **every** `(g,s,q,k)` whatsoever, and `err_i ≥ dist(t_i, [min_{j≤i} b_j, max_{j≤i} b_j])`. **For W1, "how many states" has a finite answer `k` with a residual; for W3 on a sign-alternating path product there is no `k` at ANY size.** The `rank₊` half is struck — *the theorem is sound, the `n = 4` test citing it is two identities* (`V20_R15_IT11_INSPECTOR.md:502`).
- DECLARATION: measured instance — planted negative **PN-2 `sign_flip_gate`, seed 0**, `S = 8`, 200 draws, on `ceq/arm_pl.py:*`: `t_1 = −2.483118`, hull `[−1.398595, −1.084522]`, excess **`1.0845223424`**, bound to `5e-10` @ `tests/jupiter/test_v20_r15_it12_constants.py:11`. **The constant was published as `1.084523` and withdrawn as wrong** — a prose constant formed by arithmetic on another prose constant. Second instance: `rank_plus_lower → NEG_ENTRY`, witness `(0,2)`, value `−1.0`, `rank_real(H) = 1` — **measured 1, smaller than the annex's hint of 2.**
- CENSUS: **structural, not bed-counted.** The convex-combination argument holds on every `(g,s,q,k)`, so the census is the arm's whole parameter space rather than a bed list. That is why this cell survived it.12's bed audit and Q2/W1 did not.
- ROUTE: the constant is kept because it is measured on `ceq/arm_pl.py:*` — **W3's own arm** — and not on a Hankel block. Outstanding: a repo-wide scan for prose constants formed by arithmetic on other prose constants.
- ARENA: bears on clause (1) and clause (3). Scoreable on both. **Does not touch clause (2)** — this bound is W3's own, not the Hankel ceiling.

### CELL Q3/W1 — LEARNABILITY, `arm_smprime`
- GRADE: F2
- GAP: **The window is NECESSARY on sixteen cells and DEMONSTRABLY NOT SUFFICIENT.** Seed 3 sits **inside** the qualitative window at `λ̂_live = −0.4411` and does **not** cross (`0.916258`). A condition with a counterexample in its own sixteen cells is F2, not F1. **Second, independent gap:** the partition is stated in **post-training** coordinates (`manifest.smp_values`) — a description, not a criterion. **No constant** — which is exactly what F2 means here.
- DECLARATION: `lambda_hat = float(lg.mean())` @ `scripts/v15_r1.py:383`, `lambda_hat_live` @ `scripts/v15_r1.py:384`, `frac_gate_annihilated` @ `scripts/v15_r1.py:386`; theory side `pathProd_eq_zero_iff` @ `lean/CEQ/V16Domain.lean:129` and `pathProd_abs` @ `lean/CEQ/V16Domain.lean:121`. **Instrument defect carried in the cell:** `lambda_hat` is the mean of `log m` over **every** position, so one `m_k == 0` sends it to `−inf`; it carries exactly one bit (*"at least one gate is exactly zero"*) and **every mechanism sentence in the round that reads it as a magnitude is reading a mixture.** `lambda_hat_live` is the magnitude column and is journalled on all cells.
- CENSUS: **16 of 16** cells partitioned by trained `qk` with **zero overlap**; the `−inf` law holds on **32 cells** with `bad == []`. Spearman `qk = +0.717647`, `beta = −0.032353`, both bound @ `tests/jupiter/test_v20_r15_it12_constants.py:13-14`.
- ROUTE: journal `manifest.smp_values` @ `ceq/arm_smprime.py:409` on the `_0step` record @ `scripts/v15_r1.py:804` — **0 GPU-s**, the same four fields Q1/W1 needs.
- ARENA: bears on clause (1). Scoreable, and it is the cell that explains W1's `1/16`.

### CELL Q3/W3 — LEARNABILITY, `arm_pl`
- GRADE: F1 + const (margin `0.451211`)
- GAP: **A correlation over `n = 8` with the causal direction UNTESTED.** The separation is equally consistent with the cap being causal and with `λ̂`'s sign and the divergence sharing a common cause in initialisation. **No capped run exists.** The gap narrowed but did not close at it.9: the predictor held **out of sample on 16 of 16** — but *held out of sample* is still not *causal*.
- DECLARATION: measured — `::test_w3_lambda_hat_sign_separates_all_eight_arm_pl_cells` @ `tests/jupiter/test_v20_r15_it7_q3.py:106`; producers `lambda_hat` @ `scripts/v15_r1.py:383`, `a_hat_max`/`gate_r2` journalled per cell. **The theorem half is cited by ANNEX NUMBER ONLY** — the it.7 text names *"M2 GATE-LANDSCAPE THEOREM"* and `Lean #19 [M]` and **gives no declaration name.** This cell therefore rests on the instance alone, and the leap should read it that way.
- CENSUS: **8 of 8 → 16 of 16 out of sample**, zero overlap, gap `0.451211`; `max(nrmse | λ̂<0) = 0.662128` against `min(nrmse | λ̂>0) = 1.113339`; `frac_gate_annihilated == 0.0` on all eight. **Caveat kept in the cell:** the eight new `arm_pl` cells are **uncontrolled** — softmax exists only for seeds 0–7.
- ROUTE: three **capped** cells at seeds 2, 3, 7 — **~6 GPU-s** at `1.884 s`/cell. It settles Q3/W3 causality, Q5/W3 crossing and pre-registerability **at once**. **Priced three times across it.7, it.8 and it.9 and taken zero times. It is the cheapest live experiment in the round.**
- ARENA: bears on clause (1) directly — it is the mechanism behind W3's `12/16`.

### CELL Q4/W1 — COST LAW, `arm_smprime`
- GRADE: F3 — **a fact about the harness (see §0.3), not about the primitive**
- GAP: The exponent in `S` is **unidentified**: `n = 1`. The constant is known only to **±1.49×**. The timer is un-synchronised. The strongest correlate of `secs` is **run order** at `ρ = +0.70`.
- DECLARATION: `[DERIVED]` from source, steps in `V20_R15_IT8_JUPITER.md:105` — `path_product` @ `ceq/arm_smprime.py:144` is `O(n·S²)`, giving `T_W1(n,S) = Θ(n·S²·d_model)`. Harness constant `S, D = 64, 24` @ `scripts/v15_r1.py:137`. **Sparsity certificate is F0-EXACT and carried by a declaration**, `pathProd_eq_zero_iff` @ `lean/CEQ/V16Domain.lean:129` — it needs no Cantelli, no Azuma and no union bound, which is fortunate, since **all four of those have 0 producing `.py`.** Uncalled producer: `ArmSMPrime.zero_hop_mask` @ `ceq/arm_smprime.py:559` — **never called by the runner**, and `path_product` materialises the dense `[S,S]` block **before** producing the zeros.
- CENSUS: `s = 64` on **40 of 40** (`[RUN]` this iteration, deduped). **0 of 2 wings** lie in the domain of any of the three shipped cost models — `ceq/sizing.py:145`, `scale/m3_flops.py:207`, `ceq/mz_kernel.py:170`. `frac_gate_annihilated` measures **2 of 64 positions**, not the `64×64` operator.
- ROUTE: one `--seq-len` flag, one substitution at `scripts/v15_r1.py:137`, two `torch.cuda.synchronize()` at `scripts/v15_r1.py:249` and `scripts/v15_r1.py:267`; `S ∈ {32,64,128}`. **Price: 206–537 GPU-s, band only — NO POINT** (`V20_R15_IT13_MERCURY.md:146`; the endpoints are `_sweep(1.0)`/`_sweep(3.0)` and `e ∈ [1,3]` is unmeasured) — this office's published `~275` **does not evaluate to 275** (the formula as written gives `309.015`; the withdrawn `309.047` is `_sweep(2.0)` on it.13's re-measured means — a different object, and quoting it as the band's centre would claim an independent centre the band does not have) and its `960` honest total is **34.8% high**. **A fifth edit — randomised or blocked execution order — is required to remove the `ρ = +0.70` confound and NO OFFICE HAS PRICED IT.**
- ARENA: **clause (3) is not scoreable on this cell until the repair lands.** See §3.

### CELL Q4/W3 — COST LAW, `arm_pl`
- GRADE: F3 — **the same harness fact as Q4/W1, written twice**
- GAP: There is no law in `S` **because `S` never varied**. An arm that costs `1.0589×` softmax at one length is **a ratio, not a cost law**. W3-specific and separate: `brute_force_path_sums` @ `ceq/arm_pl.py:304` is `O(2^S)` and **unguarded**.
- DECLARATION: `[DERIVED]` from `ceq/arm_pl.py:113` ⇒ `Θ(n·S²·d)`, same order as softmax; gate is an additive log-domain bias @ `ceq/arm_pl.py:93`. Measured ratio `1.0589×`, range `0.987×–1.154×`. Harness constant `scripts/v15_r1.py:137`, as W1.
- CENSUS: `s = 64` on **40 of 40**; `frac_gate_annihilated == 0.0` on **8 of 8** `arm_pl` cells at the time of measurement; same **0 of 3** cost-model domain.
- ROUTE: W3's share of the sweep ≈ **21 GPU-s**. Identical repair to Q4/W1 and taken at the same time.
- ARENA: **clause (3) not scoreable until the repair lands.** See §3.

### CELL Q5/W1 — INFORMATION FLOOR, `arm_smprime`
- GRADE: F1 + const **on `floor₁`, a one-hop CAPABILITY THRESHOLD — NOT on an information floor.** The bound is real and the object is the wrong one; the GAP below is a scope note, **not a withdrawal of the grade** (RULING J-17a, `V20_R15_IT17_JUPITER.md`). Ledger agrees: `L-11` reads *"F1 + const on both wings, but on the wrong object"* (`V20_R15_LEAP_LEDGER.md:70`).
- GAP: **THE FLOOR GRADED AGAINST IS NOT AN INFORMATION FLOOR.** `floor₁` is a **one-hop capability threshold** and is **violated by 13 of 40 cells** (`[RUN]` this iteration; the published `6 of 34` is stale). Distance to the **real** information floor — the exact oracle at `0.0` — is `0.203920` at best and `0.852` modally, and **no annex theorem predicts either.**
- DECLARATION: **WHICH FLOOR THIS CELL IS MEASURED AGAINST:** `floor1 = math.sqrt((T_STAR-1)/T_STAR)` @ `scripts/v15_r1.py:586`, second producer `hop_floor` @ `scale/it11_verdict.py:451`, identity @ `scripts/v15_r1.py:17`. **It is `floor₁`, a capability threshold — NOT Fano, NOT rate–distortion, NOT Hankel.** Nodes `::test_q5_floor_1_is_a_ONE_HOP_THRESHOLD_not_an_information_floor` @ `tests/jupiter/test_v20_r15_it8_q4_q5.py:127`; constant band `min=0.144954 max=0.219610 (n=12)` bound @ `tests/jupiter/test_v20_r15_it12_constants.py:122`.
- CENSUS: **four candidate floors, zero reach this wing as information floors.** M11 Fano **0 of 3 beds, 0 `.py`**. M12 rate–distortion **0 of 3 beds, 0 `.py`**. M16 Hankel **1 of 3 (BED-K)** — does not bind a BED-M wing. `floor₁` **1 of 3 (BED-M)**, journalled on **40 of 40**, and not a floor. `dist_to_skyline` is `None` on **40 of 40**, reason recorded.
- ROUTE: journal `I(X;Y)` against BED-M's **already-computed exact oracle** — an instrument line, **0 GPU-s**. The annex imported M11/M12's conclusions and left the estimator out. Open sub-gap: the predicted `+0.20…+0.27` distance is **still prose with no node**.
- ARENA: bears on clause (3)'s *"-to-floor"* half — **and that half inherits this cell's defect**: GPU-seconds-to-`floor₁` is seconds to a capability threshold, not to a floor.

### CELL Q5/W3 — INFORMATION FLOOR, `arm_pl`
- GRADE: F1 + const (`ci_hi = 0.6582633033`, clears `floor₁` by `0.0488`) — **on `floor₁`, the same capability threshold, NOT on an information floor** (RULING J-17a).
- GAP: **The obstruction is not the floor — it is the pooled statistic.** 12 of 16 `arm_pl` cells sit below `floor₁` and **every one clears it with its whole bootstrap interval** (`max(boot_hi | below) = 0.705184`, seed 12), yet the registered verdict is `crosses: false`, because **four divergent cells drag the mean to `0.7784`, above the threshold before variance is considered.** A mean over a bimodal population is not a statement about either mode. The it.8 five-cell CI was **post-hoc**; at it.9 it stopped being post-hoc — **7 of 8 new cells fell on the predicted side.**
- DECLARATION: **WHICH FLOOR: `floor₁` @ `scripts/v15_r1.py:586`, the same capability threshold, with the same defect.** Measured @ `tests/jupiter/test_v20_r15_it8_q4_q5.py:151`, node `::test_q5_no_arm_crosses_and_W3_is_blocked_by_variance_not_by_the_floor`; out-of-sample confirmation @ `V20_R15_IT9_JUPITER.md:265`. Registered verdict `crosses: false` on all three arms.
- CENSUS: 5 of 8 below floor at it.8 → **12 of 16** at it.9 → **12 `arm_pl` of 16** confirmed by this iteration's dedupe. **Caveat kept in the cell:** the eight new `arm_pl` cells are uncontrolled — softmax exists only for seeds 0–7.
- ROUTE: the same **~6 GPU-s** capped run at seeds 2, 3, 7. **And a scoring-rule change that costs nothing**: component-wise CIs under the declared mixing variable `sign(lambda_hat)`, which is pre-registerable and already journalled on every cell.
- ARENA: **this is the cell clause (1) actually turns on**, and it is why `⟨CLAUSE_1_TAIL⟩` is decisive. See §3.

### CELL Q6/W1 — STATE METRIC, `arm_smprime`
- GRADE: F4 — **domain empty: the metric has no object on this bed**
- GAP: **The wing returns a point prediction. There is no state distribution to put a measure on.** The contract's own clause is *"W1 to the oracle **where a state distribution exists**"* (`CEQ_V20_R15_CONTRACT.md:123`) and **the condition is not met.** Second: **0 of 40** banked cells journal a prediction vector, histogram, quantile or density, so the marginal is **not computable from the record without a re-run.**
- DECLARATION: `ArmSMPrime.forward` returns `readout(h).squeeze(-1)[:, seq-1]`, shape `[n]` @ `ceq/arm_smprime.py:577`; `equilibrium_oracle` returns shape `[n]` @ `scale/negation_scope.py:304` (declared @ `scale/negation_scope.py:286`). **M13 Wasserstein is cited by ANNEX NUMBER ONLY and has NO PRODUCER IN THIS TREE** — a keyword sweep returns no `.py` for `wasserstein` and none for `kantorovich`; the annex's `[RUN: 0.492 vs KL 0.519]` resolves to `0.492188` from `scale/foreman_consequence.py:12`, **an unrelated quantity.** Nodes @ `tests/jupiter/test_v20_r15_it9_q6.py:*` and `tests/jupiter/test_v20_r15_it11_q6_oracle.py:*`.
- CENSUS: **0 of 40** — re-run at it.11 over 60 union field names with 3 false-positive `dist` hits and **0 semantic hits**, journalled as `t="census"` in `results/v20_r15_it11_jupiter_census.jsonl`. No parameter named `*logits` or `*prob*` on either arm.
- ROUTE: **(1)** journal `pe` and `y_ev`, or 101-quantile summaries — 0 GPU-s, but this buys a **calibration** reading, not the score. **(2)** a distributional head scored by CRPS or pinball loss — **that is an ARM CHANGE and the wing list is frozen.** No theorem turns a point prediction into a distribution.
- ARENA: **bears on no clause.** Q6 is not in the lexicographic order.

### CELL Q6/W3 — STATE METRIC, `arm_pl`
- GRADE: F4 — **same mechanism; and the metric is additionally REFUTED on this bed**
- GAP: `ArmPL.forward` returns the same `[n]` scalar. **And the marginal W1 is permutation-blind on the real oracle**: a predictor returning the oracle's own values **in the wrong order** scores `W1 = 0.0` exactly while `NRMSE = 1.421901019003236` (against `√2 = 1.4142135623730951`, off by `+7.687e-03`) — **worse than predict-the-mean.** Against `oracle + 0.1σ` noise, `W1` reads `0.0` vs `0.012049103155732155` (**prefers the permutation**) while `NRMSE` reads `1.421901` vs `0.098296619951725` (**prefers the noise**). **The metric and the bed rank two predictors in OPPOSITE orders, by `14.465410797679917×`.**
- DECLARATION: `::test_q6_marginal_W1_is_permutation_blind_on_the_real_oracle` and `::test_q6_marginal_W1_inverts_the_ranking_against_nrmse` @ `tests/jupiter/test_v20_r15_it11_q6_oracle.py:*`, on `equilibrium_oracle` @ `scale/negation_scope.py:286` at `n=4096, s=64, d=24, d_model=16, t_star=2, seed=4096`, CPU. **The it.9 node that stood here until it.11 is STRUCK and KILLED** — it used `torch.randn` in place of the oracle, measured `1.4060346618513293`, and its `W1 == 0.0` was a sorted-difference identity that **could not fail** (`V20_R15_IT89_INSPECTOR.md:47`; kill @ `tests/jupiter/test_v20_r15_it9_q6.py:155`). Controls that fire: `W1(2z,z) > 0`, `W1(z+1,z) = 1.0`.
- CENSUS: **0 of 40** banked cells journal a distribution — same census as Q6/W1, one instrument. Geometry-bound: `t_star=None → 1.413115257241014`, `seed=0 → 1.411094757351561`.
- ROUTE: **none on the frozen wings.** The one registered bed with a categorical state space would be the chess witness — **and it is NOT REGISTERED**: `[RUN] python -c "from ceq import kdata; print(list(kdata.BED_SPECS))"` returns `['bed_m', 'bed_k', 'bed_1']` (`V20_R15_IT13_MERCURY.md:189`). **The ledger's claim at `V20_R15_LEAP_LEDGER.md:131` that it is "the one registered bed" is FALSE and is corrected here.**
- ARENA: **bears on no clause.** Q6 is not in the lexicographic order.

---

## §3 — THE ARENA TICKET, PER WING. IT IS PARTLY VOID.

**The lexicographic criterion** (`CEQ_V20_R15_CONTRACT.md:125`): (1) BED-M crossing
CP-lower > 0.5; (2) BED-K(a) within resolution of the attention-native ceiling; (3) lowest
GPU-seconds-to-floor; (4) witness tiebreak.

| clause | **W1 `arm_smprime`** | **W3 `arm_pl`** | status |
|---|---|---|---|
| **(1)** BED-M CP-lower > 0.5 | **SCOREABLE.** `1/16`, rate `0.0625`, CP-lower `0.0016` two-sided / `0.0032` one-sided | **SCOREABLE.** `12/16`, rate `0.7500`, CP-lower **`0.4762` two-sided (FAILS by `0.0238`) / `0.5156` one-sided (CLEARS)** | **blocked on `⟨CLAUSE_1_TAIL⟩`, an author ruling costing 0 GPU-s** |
| **(2)** BED-K(a) within `1/d` | **NOT SCOREABLE** | **NOT SCOREABLE** | **VOID — see below** |
| **(3)** lowest GPU-s-to-floor | **NOT SCOREABLE AS WRITTEN** | **NOT SCOREABLE AS WRITTEN** | needs 4 instrument edits + a 5th unpriced one; and *"floor"* here means `floor₁`, a capability threshold (§0.2) |
| **(4)** witness tiebreak | **NOT SCOREABLE** | **NOT SCOREABLE** | no arm consumes a chess corpus; no chess bed is registered; eval-Δ oracle has **0 producers** and sits behind a 34.4 GB **Kaggle attach** |

### 3.1 CLAUSE (2) IS VOID, AND IT IS A GAP IN THE ORDER — NOT A TIEBREAK

**A ticket to a clause no entrant can enter is not a ticket.** Both frozen wings are
**BED-M arms**; the ceiling clause (2) names is the Hankel `1/d` bound, which **binds
BED-K only** — *"do not cite it on W1/W3 at all"* (`V20_R15_IT8_JUPITER.md:473`). **That is
this office's own ruling, and it is the ruling that re-graded Q2/W1 to F4 above.** MERCURY
prices the clause at **nothing**, because **no cell of BED-K's shape has ever been run**:
the generator is registered at `ceq/kdata.py:475` and **zero files under `scripts/` call
it** (`V20_R15_IT13_MERCURY.md:63`).

**A tiebreak that cannot be evaluated still orders the arms — it simply never fires.
A GAP in the order does not.** Clause (2) is a gap: when the arms reach it, the order has
nothing to say and the arena returns no winner rather than an arbitrary one.

**AND WHETHER THE GAP BITES IS DECIDED BY `⟨CLAUSE_1_TAIL⟩`, WHICH IS UNRULED. THIS IS THE
LOAD-BEARING CONSEQUENCE AND IT IS STATED HERE FOR THE FIRST TIME:**

- **One-sided reading** (`0.5156`): **W3 clears clause (1) alone, W1 does not.** The order
  terminates at clause (1). **Clause (2) is never reached and its void is LATENT.**
- **Two-sided reading** (`0.4762`): **NO entrant clears clause (1).** All arms tie at the
  first clause, the order **falls through to clause (2)** — and clause (2) cannot be
  scored. **The void becomes FATAL and Phase C returns no winner at any price.**

**So `⟨CLAUSE_1_TAIL⟩` is not a bookkeeping convention. It decides whether the arena has a
structural hole or merely an unused one.** It remains the author's ruling and this office
will not pick the tail: choosing the tail that makes the round score is the catalogued
failure of selecting a threshold after seeing the data it judges (`MISTAKES.md:451`).

### 3.2 WHAT EACH WING CAN ACTUALLY BE SCORED ON, IN ONE SENTENCE EACH

- **W1 `arm_smprime`** — scoreable **today** on clause (1) only, where it reads `1/16` and
  fails under **both** tails. Clause (3) after a 206–537 GPU-s repair that still carries the
  run-order confound. Clauses (2) and (4): **not an entrant**.
- **W3 `arm_pl`** — scoreable **today** on clause (1), where it reads `12/16` and its verdict
  **flips with the tail**. Clause (3) same repair. Clauses (2) and (4): **not an entrant**.
- **Separation, which does not depend on the tail ruling:** Fisher exact one-sided,
  `arm_pl` vs `arm_smprime`, **`p = 8.544e-05`**.

---

## §4 — WHAT THE LEAP GETS, AND WHAT IT MUST NOT BE ASKED

### 4.1 THE F4 RULING, AND THE RULE IT WAS MADE UNDER

The it.35 gate sorts **F1/F2/F3** failures into LEAPABLE or TERMINAL. **The gate does not
take F4.** This table has **three F4 cells**: Q2/W1, Q6/W1, Q6/W3. The ledger separately
carries **six F4 rows** already stamped LEAPABLE or TERMINAL — L-3, L-4, L-7, L-8, L-13,
L-14 (`V20_R15_IT12_INSPECTOR.md:272`; SATURN counted five and omitted **L-8**).

> **RULING J-14. An `F4` cell is NOT leap material as an `F4`.**
>
> **The rule applied:** *TERMINAL* means a **bound** — a claim about every instrument in
> the class, which no theorem removes. *LEAPABLE* means a **missing statement** a theorem
> could supply. **An empty domain is neither.** It is a claim about **this round's
> instruments**, and it is repaired by a bed, a head, or a journalled field — never by a
> theorem. Grading it TERMINAL overstates it (it is not a bound); grading it LEAPABLE
> invites the leap to search for a theorem whose hypotheses nothing here satisfies.
>
> **Therefore each F4 cell is filed NOT-PUT, with an ADMISSION CONDITION**: the named
> instrument change that makes the domain non-empty. **It re-enters the gate at whatever
> grade it earns once that condition holds, and not before.**

| F4 cell | ADMISSION CONDITION | grade it would re-enter at |
|---|---|---|
| **Q2/W1** | one cell of BED-K's shape actually run — the generator is `build_delay` @ `ceq/beds/bed_k.py:236`, registered @ `ceq/kdata.py:475`, and has **two callers under `scripts/`** — `scripts/v20_m14_cheeger.py:347` and `:462`, both at the registered `(n=500, d=4, seed=7)`, both consuming the kernel `b["K"]` and never the bed's data, so no cell of BED-K's shape has been run (`:152`) | `F1 + const` (the `1/d` law is already true there) |
| **Q6/W1** | a distributional head on the arm — **an arm change, and the wing list is frozen** | unknown; the metric is untested on any distribution this round produces |
| **Q6/W3** | the same head **plus** a metric that is not permutation-blind — W1-marginal is **refuted** on this bed, not merely inapplicable | unknown; and CRPS/pinball must be re-derived, not assumed |

**The three F4 cells therefore reach the leap as ADMISSION CONDITIONS, not as failures.**
The leap is being told *what this round could not ask*, not *what it asked and failed*.

**And the six F4 ledger rows were graded on a scale the gate does not take.** They are not
withdrawn — the underlying findings stand — but **their LEAPABLE/TERMINAL stamps are
formally out of scope for the it.35 gate** and must not be counted as gate input. Naming
that here is the point: an unstated answer is what the leap would have spent its one call
on.

### 4.2 THE NINE CELLS THAT DO REACH THE GATE

| cell | grade | gate class | FIELD (LEAPABLE only) — a discipline, never a want |
|---|---|---|---|
| Q1/W3 | F1 | **LEAPABLE** | **realization theory for linear time-invariant systems** (Hankel/Kronecker realization) |
| Q2/W3 | F1 + const | **TERMINAL** | none — the convex-hull excess is a **bound**, holding for every `(g,s,q,k)` |
| Q3/W1 | F2 | **LEAPABLE** | **transfer-operator / Koopman spectral theory** — the part relating a leading exponent to approximation error |
| Q3/W3 | F1 + const | **LEAPABLE, but ~6 GPU-s buys it outright** | **bifurcation theory / gradient-flow convergence** — and the run settles it cheaper than the reading |
| Q4/W1 | F3 | **TERMINAL as a leap target** | none — **no theorem supplies a slope from one point.** §0.3 |
| Q4/W3 | F3 | **TERMINAL as a leap target** | none — same harness fact |
| Q5/W1 | F1 + const | **LEAPABLE** | **rate–distortion / channel-capacity theory applied to the DETERMINISTIC oracle case** |
| Q5/W3 | F1 + const | **LEAPABLE — and it is a SCORING RULE, not a theorem** | **finite-mixture inference** — when a pooled statistic is inadmissible and what replaces it |
| Q1/W1 | F0 | **not a failure** | — (the `0 of 24` attributability gap is bookkeeping, 0 GPU-s) |

### 4.3 THE LEDGER ROWS THIS OFFICE OWNS, RE-CHECKED AGAINST THE FIELD RULING

The rule (`V20_R15_IT567_INSPECTOR.md:482`): *naming a field and then saying what you want
from it is compliant; naming the want and calling it a field is not.*

**`[RUN] python -m pytest tests/saturn/test_v20_r15_it12_saturn.py::test_the_FIELD_ruling_is_enforced_row_by_row -q` → `1 passed`.**
SATURN's detector — `inadmissible_leapable_rows()` @ `tests/saturn/test_v20_r15_it12_saturn.py:319` —
names **five** rows: `V-it7` (VENUS's), and **`L-2`, `L-9`, `L-13`, `L-14` — ALL FOUR OF
THEM THIS OFFICE'S.** The brief named one. It is four.

**And the four fire on a different arm of the predicate than the report's framing
suggests.** The framing is *"FIELD names a want rather than a discipline"*, which describes
**only `V-it7`**. The four JUPITER rows trip `head.startswith("none")`
(`tests/saturn/test_v20_r15_it12_saturn.py:329`) — and the ledger's own convention says
naming no field is **correct** for a TERMINAL row (`V20_R15_LEAP_LEDGER.md:99`). What
actually pulls them in is the substring test `"LEAPABLE" not in verdict` reading a **hybrid
verdict** as LEAPABLE:

| row | the verdict as written | what it is |
|---|---|---|
| `L-2` | *TERMINAL as a leap target, LEAPABLE by 0 GPU-s of bookkeeping* | hybrid |
| `L-9` | *TERMINAL as a leap target, LEAPABLE by 206–537 GPU-s, band only, no point (`V20_R15_IT13_MERCURY.md:146`) and one argparse line* | hybrid |
| `L-13` | *TERMINAL as stated; LEAPABLE only after a bed change* | hybrid |
| `L-14` | *TERMINAL for BED-M and BED-K; LEAPABLE only on the chess witness* | hybrid |

> **RULING J-14b. A HYBRID VERDICT IS NOT A VERDICT.** The it.35 gate sorts each failure
> into LEAPABLE **or** TERMINAL. A row saying both hands the leap a fork and no rule for
> taking it. **All four hybrids are this office's, and all four are resolved here to a
> single token.** The prose that made them hybrids was accurate; it belonged in the gap
> column, not the verdict column.

| row | **resolved verdict** | ground |
|---|---|---|
| `L-2` | **TERMINAL** | no theorem repairs a record that never wrote the field. Journalling four scalars fixes it; a leap cannot. Field: none. |
| `L-9` | **TERMINAL** | no theorem supplies a slope from one point (§0.3). Field: none. |
| `L-13` | **NOT-PUT** (F4) | superseded by RULING J-14; **FIELD column withdrawn** — *"the nearest LEAPABLE restatement is calibration / probabilistic forecasting (CRPS, pinball loss)"* (`V20_R15_LEAP_LEDGER.md:130`) is a **want** introduced by a hedge, under a column that must lead with a discipline |
| `L-14` | **NOT-PUT** (F4) | same defect, same column, same ground |

**The findings in all four rows stand** — `L-2`'s `0 of 24` attributability, `L-9`'s
`s = 64` constant, `L-13`'s `[n]` shape and `0 of 40` census and M13's missing producer,
`L-14`'s permutation-blindness. Each is carried in its cell in §2. **Only the verdict
tokens and `L-13`/`L-14`'s FIELD columns are withdrawn.**

**`L-3`, `L-4`, `L-7`, `L-8`** are the remaining F4 rows; their LEAPABLE/TERMINAL stamps
are out of scope for the gate under J-14, and none is withdrawn as a finding.

**Two limits on the detector itself, recorded rather than smoothed.** `KNOWN_INADMISSIBLE`
(`tests/saturn/test_v20_r15_it12_saturn.py:343`) is a **frozen literal**, so the node is a
change-detector, not an independent derivation of five. And its **planted negative
(`tests/saturn/test_v20_r15_it12_saturn.py:353`) exercises only the `ROUND_NOUNS` arm** —
**the `none` arm that produces four of the five findings has no control at all.** This
office is the party those four findings are against and says so anyway.

**The correction to the record this makes:** `V20_R15_LEAP_LEDGER.md:131` asserts the chess
witness is *"the one registered bed with a categorical state space."* **It is not
registered.** `BED_SPECS` returns `['bed_m', 'bed_k', 'bed_1']`. That sentence is
withdrawn here.

---

## §5 — LIMITS

Collected once, at the end.

**`[RUN] cd lean && lake build; echo exit=$?` → `exit=0`, zero output lines**
(`lean/lakefile.lean:*`, toolchain `leanprover/lean4:v4.7.0`, Lake `5.0.0-6fce8f7`).
**It was a silent cache hit, not a recompile** — the 13 `.olean` artifacts under
`lean/.lake/build/lib/` all date to Aug 31 and forcing a from-scratch build would require
deleting `.lake`, which is a write. **`grep -n sorry lean/CEQ/V16Domain.lean` returns one
hit and it is prose in a comment** (`lean/CEQ/V16Domain.lean:75`); every declaration carries
`#print axioms` at the foot of the file. **All 14 citations re-verified at HEAD, 14/14, zero
moved**, including `pathProd_eq_zero_iff` @ `lean/CEQ/V16Domain.lean:129`.

**Two citation mismatches found by that check and recorded here rather than smoothed.**
(i) `scripts/v15_r1.py:17` is **module-docstring prose**, not executable code — the `floor₁`
identity is documented there and *computed* at `scripts/v15_r1.py:586`. Every citation of
`:17` in this table is a citation of a comment. (ii) `zero_hop_mask` @
`ceq/arm_smprime.py:559` has **no production caller** anywhere — its only real call site is
`tests/arm_smprime/test_arm_smprime.py:559`, and `tests/jupiter/test_v20_r15_it8_q4_q5.py:222`
merely asserts the string `"def zero_hop_mask"` is present in the source. **A test that
asserts a function's text exists is not a test that the function runs.**

**The checker that verifies this table's citations cannot read semantics** — it confirms
the file exists and the line is in range, not that the declaration named is the declaration
at that line. That gap is why every DECLARATION field also carries the declaration's
*name*, and it is the honest ceiling of
`tests/jupiter/test_v20_r15_it14_theory_table.py:*`.

**The grades are ordinal within a column and nominal across columns** (§0.1) — the scale
has no text and this office's reconstruction is BOUND only to its own usage.

**`floor₁` is a capability threshold on every Q5 cell** (§0.2). The `13 of 40` violation
count is measured this iteration on the deduplicated record; the published `6 of 34`
predates eight `arm_pl` cells landing. **Three of the 43 rows are exact re-emissions and
the round has never journalled that fact.**

**Q4's two `F3`s are one harness fact** (§0.3), and the run-order confound at `ρ = +0.70`
**survives** the priced repair.

**Every arena figure in §3 is MERCURY's, not re-derived here**; his `secs` basis is
un-synchronised CUDA host wall clock, so `206–537 GPU-s` is an order-of-magnitude price and
not a budget. This office's own independent sum over the 40 deduplicated cells is
**`316.954 s`**, against MERCURY's `317.092` over 40 rows — the `0.138 s` difference is the
duplicate accounting and neither figure is wrong.

**The `12/16` and `1/16` crossing counts are pooled over cells whose control is
incomplete** — softmax exists only for seeds 0–7, so the eight newest `arm_pl` cells are
uncontrolled.

**Nothing in this file was measured on BED-K or on the chess witness**, because no cell of
either shape has ever been run in this tree.
