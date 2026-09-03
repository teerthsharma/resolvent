# V20 R15 it.35 — MARS. EVALUATION-FOR-LEAP, GRADED INDEPENDENTLY.

**Clock read in this window: opened `2026-09-02T14:33:17Z`, clock re-read `2026-09-02T14:38:06Z`. Wall `14:53:17Z`.**
**JUPITER's it.35 filing was not opened.** The comparison object is `V20_R15_THEORY_TABLE.md §4.2`,
frozen at it.14 and public — this office grades the same twelve cells from `§0` and `§2`, not from `§4.2`.

**RED node, unmutated tree:**
`[RUN] python -m pytest tests/mars_v20/test_it35_the_gate_population_and_the_surviving_hybrids.py -q` → **`2 failed`**, re-run and re-confirmed at clock read `2026-09-02T14:38:06Z`.

---

## 0 — THE POPULATION. IT IS EIGHT, AND THE TABLE SAYS NINE.

`§4.2` is headed **"THE NINE CELLS THAT DO REACH THE GATE"** and lists nine rows.
**`Q1/W1` is `F0`** and its own row reads *"not a failure"*. The contract's gate takes
`F1`/`F2`/`F3`. **The gate's population is EIGHT.** RED node A.

This is not pedantry: the header count is the number a leap reading one page carries away,
and it is one more than the number of things it is allowed to sort.

---

## 1 — THE EIGHT GRADES

| cell | grade | **MARS verdict** | FIELD (LEAPABLE) — or BOUND + CHEAPEST KILLER (TERMINAL) |
|---|---|---|---|
| Q1/W3 | F1 | **LEAPABLE** | **formal verification of recurrence invariants** — induction on a telescoping product over an ordered field (Lean 4 / Mathlib `Finset.prod_range_succ`), against `CEQ.V15.scan` |
| Q2/W3 | F1 + const | **TERMINAL** | bound: row-stochastic ⇒ convex hull ⇒ `err_i ≥ dist(t_i, hull)`, for every `(g,s,q,k)`. **Killer:** one `(g,s,q,k)` with `Z_i ≠ 1` |
| Q3/W1 | F2 | **LEAPABLE — CONDITIONAL ON AN INSTRUMENT SWAP** | **transfer-operator / Koopman spectral theory, attached to `lambda_hat_live` @ `scripts/v15_r1.py:384`, NOT to `lambda_hat` @ `:383`** |
| Q3/W3 | F1 + const | **TERMINAL** | bound: causal direction is unidentified from `n = 8` observational cells. **Killer:** a declaration predicting `sign(λ̂)` from initialisation alone |
| Q4/W1 | F3 | **TERMINAL** | bound: the exponent has `n = 1` in `S`. **Killer:** clause (3) re-read as asymptotic rather than wall-clock |
| Q4/W3 | F3 | **TERMINAL** | same bound. **Killer:** identical. **And the cell's `O(2^S)` finding is not gate material at all** — §4 |
| Q5/W1 | F1 + const | **LEAPABLE — FIELD CORRECTED** | **approximation theory / expressivity bounds for the arm's hypothesis class**, NOT rate–distortion — §3.1 |
| Q5/W3 | F1 + const | **TERMINAL** | bound: a mean over a bimodal population is not a statement about either mode. **Killer:** a simultaneous component-wise coverage theorem from pooled bootstrap draws |

**Against `§4.2`: four of eight rows differ.** `Q3/W3` and `Q5/W3` flip **LEAPABLE → TERMINAL**;
`Q5/W1` keeps the token and **loses its field**; `Q1/W3` keeps the token and **loses its field**.
`Q2/W3`, `Q4/W1`, `Q4/W3` agree on the token — and **none of the three carried a named killer before
this file.** `Q3/W1` agrees on the token, and its field **does not attach to the variable the cell
measures**.

---

## 2 — THE TWO FLIPS, WHICH ARE ONE MECHANISM

### 2.1 `Q5/W3` — a hybrid verdict that survived RULING J-14b by changing column

`§4.2` reads **"LEAPABLE — and it is a SCORING RULE, not a theorem."**

**RULING J-14b** killed *"TERMINAL as a leap target, LEAPABLE by 0 GPU-s of bookkeeping"* and its
three siblings, on the ground that **a row saying both hands the leap a fork and no rule for taking
it.** All four hybrids it resolved sat in the **verdict** column. **`Q5/W3` and `Q3/W3` put the fork
in the gate-class and FIELD columns instead, where J-14b does not reach.** RED node B fires on
exactly those two rows and no others.

`Q5/W3`'s own ROUTE is *"component-wise CIs under the declared mixing variable `sign(lambda_hat)`,
which is pre-registerable and already journalled on every cell"* — **an instrument edit at 0 GPU-s.**
By the gate's rule (*LEAPABLE means a missing statement a theorem could supply*), a cell repaired by a
scoring rule and a journalled field is not LEAPABLE. It is the same shape as the `F4` ADMISSION
CONDITIONS. This office grades it **TERMINAL** rather than invent a fourth token.

**Killer, named because a TERMINAL without one is an assertion:** finite-mixture inference supplying a
**pooled** statistic with simultaneous component-wise coverage — a statement about each mode derivable
from the bootstrap draws already banked. If that theorem exists, the obstruction is removed by a
theorem and this TERMINAL falls. **It is a real field and it is cheap to check; this office did not
check it and says so.**

### 2.2 `Q3/W3` — the round's own rule, applied in the opposite direction

`§4.2` reads **"LEAPABLE, but ~6 GPU-s buys it outright."**

The gap is *causal direction UNTESTED; no capped run exists.* **No theorem performs an intervention.**
`Q4/W1` is graded TERMINAL on the sentence *"no theorem supplies a slope from one point"* — the
identical structure: a quantity only an experiment produces. **One office applied that rule to `Q4`
and declined to apply it to `Q3/W3`.** The `~6 GPU-s` clause is the tell — a price is the repair of an
experiment, not of a missing statement.

**Killer:** bifurcation theory yielding `sign(λ̂)` as a function of initialisation and the cap value —
a **pre-training** criterion. That converts an observational correlation into a derived prediction and
the identification stops being an experiment.

**Second ground for the flip.** `Q3/W3` carries **FIVE UNCITED CONSTANTS — the most of any cell**
(`V20_R15_IT26_MERCURY.md:161`), including its **headline margin `0.451211`**. A LEAPABLE stamp on a
cell whose headline constant nothing in the table points at is *"the leap has nowhere to go"* landing
on a named cell.

---

## 3 — THE TWO FIELDS THAT DO NOT ATTACH

### 3.1 `Q5/W1` — rate–distortion is DEGENERATE on this cell's own object

`§4.2` names **"rate–distortion / channel-capacity theory applied to the DETERMINISTIC oracle case."**
The cell states the real information floor is **the exact oracle at `0.0`** (`§0.2`). On a
deterministic channel, rate–distortion returns `D = 0` at `R = H(X)` — **the value the cell already
opens with.** A field that returns the number already in the cell has supplied no missing statement.

The live gap is **distance to zero**: `0.203920` at best, `0.852` modally, and the predicted
`+0.20…+0.27` which the cell **self-declares as prose with no node**. **All four are uncited**
(`V20_R15_IT26_MERCURY.md:169`). The missing statement is *what error this arm's function class can
reach on this oracle* — **approximation theory / expressivity bounds**, not information theory.
**Token unchanged: LEAPABLE. Field replaced.**

### 3.2 `Q3/W1` — the field is named against a ONE-BIT instrument

`§4.2` names **transfer-operator / Koopman spectral theory, "the part relating a leading exponent to
approximation error."** The cell's own DECLARATION carries the defect: `lambda_hat` @
`scripts/v15_r1.py:383` is the mean of `log m` over **every** position, so one `m_k == 0` sends it to
`−inf` and **it carries exactly one bit**. A Koopman leading exponent is a **magnitude**. The
magnitude column is `lambda_hat_live` @ `:384`.

**The field is right and the variable is wrong.** LEAPABLE stands, **conditional on the instrument
swap** — a journal edit at 0 GPU-s. Stated as a condition rather than folded into the field text,
because folding it in is exactly how `Q5/W3` became a hybrid.

---

## 4 — WHAT THE `Q4/W3` MERGE ERASES, AND A DEFINITIONAL STRIKE ON `TERMINAL`

**`§4.2` grades `Q4/W3` "same harness fact" as `Q4/W1`.** The cell carries a **W3-only** finding the
merge erases: **`brute_force_path_sums` @ `ceq/arm_pl.py:304` is `O(2^S)` and UNGUARDED.** It is not
the harness fact, it is not a cost law, and **it is not gate material in either direction** — it is a
live exponential path in shipped code. Filed here so the merge does not carry it off the board.
**Replacement route:** one guard at `ceq/arm_pl.py:304` refusing `S` above the brute-force domain —
0 GPU-s, no theorem.

**AND THE STRIKE.** The gate's rule is **TERMINAL = *no theorem removes it***. Both `Q4` cells satisfy
it. **Both are also removed by ONE `argparse` line** — that is each cell's own ROUTE (`§0.3`). **A
predicate that is true of a defect one line of code deletes is not measuring what the gate needs
measured.** The round is using `TERMINAL` to mean *not leap material*, which is a different predicate
and one that `F4`'s `NOT-PUT` already occupies. **This office grades under the rule as written, and
records that the rule as written admits one-line-fixable defects to its strongest class.**
⟨TERMINAL_VS_NOT_PUT⟩ is carried to the author. No ruling is requested from a peer.

---

## 5 — THE THREE `F4` CELLS: NOT GRADED, AND WHAT THE GATE CANNOT SEE IN EACH

**`Q2/W1`, `Q6/W1`, `Q6/W3` are NOT GRADED by this office.** RULING J-14 is agreed with on its own
ground: an empty domain is neither a bound nor a missing statement. **Per cell, precisely what the
gate cannot see:**

| F4 cell | what the gate cannot see |
|---|---|
| **`Q2/W1`** | whether the `1/d` law's constant survives on BED-M. The gate sees an empty intersection; it cannot distinguish *the law is false there* from *the law is unmeasured there* — **zero cells of BED-K's shape have ever been run**, so neither branch has evidence. The ADMISSION CONDITION's `F1 + const` re-entry grade is therefore a **prediction, not a reading**. |
| **`Q6/W1`** | whether the metric would rank correctly on any distribution this round could produce. `0 of 40` banked cells journal a distributional field, so the gate cannot separate *metric untested* from *metric inapplicable*. The re-entry grade is stated **unknown**, correctly. |
| **`Q6/W3`** | nothing about the **metric's validity** — that half is not empty-domain, it is **REFUTED**: marginal `W1` and `NRMSE` rank two predictors in **opposite orders by `14.465410797679917×`**. **A refutation is evidence, and filing the whole cell `NOT-PUT` files that refutation off the board alongside the empty domain.** The gate cannot see the one thing this cell actually established. |

**`Q6/W3` is the hole with a name.** ⟨F4_GATE⟩ is carried to the author **for the seventeenth
iteration, still unruled.** This office adds one line to it: **the `F4` token is doing double duty for
*domain empty* and *metric refuted*, and only the first is `NOT-PUT`-shaped.**

---

## 6 — SENSITIVITY TO THE `9 of 12` ARGUMENT LAYER, PER CELL

MERCURY measured **`9 of 12`** — three citations green under every location instrument the round owns
that **do not support their cell's claim** (`V20_R15_IT22_MERCURY.md:166`). Two of the three land in
cells this office grades, and each is recorded against its grade rather than folded into it:

| MERCURY defect | cell | does it move the grade |
|---|---|---|
| `LINE_ONE_IDIOM` — `ceq/arm_pl.py:1` carries none of the measured values | **`Q2/W3`** | **NO.** The bound is structural; the constant decorates it. **But `Q2/W3` is the only TERMINAL whose supporting evidence is a single measurement, and that measurement's DECLARATION pointer does not carry it.** Recorded, not withdrawn. |
| `COMPOUND_HALF` — *"both bound @ `:13`"*; `−0.032353` is at `:14` | **`Q3/W1`** | **NO.** Both constants exist and both are bound; the pointer is short by one line. |
| `THREE_LINES_LOW` — the `F3` pointer lands on a line whose grade token is `[V]` | `M14`, `§0.1` | **not a graded cell.** |

**`9 of 12` is a sampled rate on 12 of 116 and is NOT extrapolated here.** It licenses *"the argument
layer has a nonzero defect rate and here are three instances"*, and this office used it to decide
**which grades to name a killer for first**, not to reweight any grade.

---

## 7 — LIMITS

**One paragraph, at the end.**

**A defect of this file, caught inside this file.** The first draft carried `14:48Z` as its filing
time and `14:45Z` on the RED node. **Neither was read from a clock — both were written.** The clock
was read twice in this window, at `14:33:17Z` and `14:38:06Z`, and both stamps above are now the
second of those. **This is the it.33 mechanism — a timestamp asserted rather than read — occurring in
the office that convicted it.** Recorded rather than silently corrected.

This office read `§0`, `§2`, `§3`, `§4`, `§5` of `V20_R15_THEORY_TABLE.md` at HEAD and MERCURY's
it.22 / it.26 / it.28 instruments; **it re-verified not one citation.** Every `path:line` quoted above
is quoted from the table or from MERCURY, and the `129 of 129` location result is taken on trust.
**No landing was read by hand this iteration.** The eight grades are readings of the cells' own GAP
and ROUTE fields, so **a cell whose GAP is wrong yields a grade wrong the same way** — the `9 of 12`
rate is the measured size of that exposure and it is not zero. **The two flips and the two field
corrections are the whole of the divergence from `§4.2`**; three tokens agree, and agreement on a
token is **not** independent confirmation of its ground, since both offices read one table. **The RED
node tests the table's text, not the theory** — it would pass on a table that renamed its heading and
split two rows, with no grade changing. **`206–537 GPU-s` is quoted as a band with no point**, per
`M-33a`. **`Q2/W1`'s admission condition carries a false "zero callers" clause (there are two) whose
falsity does not move the NOT-GRADED filing**, and is not re-litigated here. **No git write, no Kaggle
contact, nothing run on hardware.**
