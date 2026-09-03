# V20 R15 — it.12 — JUPITER (MYCROFT)

**Node:** `tests/jupiter/test_v20_r15_it12_constants.py` — six nodes, six planted
negatives, 6 of 6 mutations kill the node they aim at and no mutation is silent.

| | |
|---|---|
| **A — five unasserted constants** | **four bound, one WITHDRAWN as WRONG** |
| **B — Q2 graded on the wrong bed** | **Q2/W1 re-graded `F1 + const` → `F4`**, on this office's own it.8 row |
| **C — the F0–F4 rubric** | **IT DOES NOT EXIST.** Concurs with the Inspector; adds two facts against the contract |

---

## 1. TASK A — THE FIVE CONSTANTS

`[RUN]` `python -m pytest tests/jupiter/test_v20_r15_it12_constants.py -q -p no:randomly`
→ **`6 passed in 2.25s`**.

| constant | published at | status | tolerance |
|---|---|---|---|
| `1.084523` | `V20_R15_IT6_JUPITER.md:427,494` | **WITHDRAWN — WRONG** | `5e-10` on `1.0845223424` |
| `0.9746794345` | `V20_R15_IT6_JUPITER.md:466` | **BOUND** (and worth little — §1.2) | `5e-11` |
| `+0.717647` | `V20_R15_IT7_JUPITER.md:91` | **BOUND** | `5e-7` |
| `−0.032353` | `V20_R15_IT7_JUPITER.md:92` | **BOUND** | `5e-7` |
| `+0.145 … +0.220` | `V20_R15_IT8_JUPITER.md:344,352` | **BOUND** | `5e-4` each endpoint |

Node names, in order: `::test_sign_flip_gate_seed0_hull_excess_is_1_0845223424`,
`::test_q2_w1_err_1_is_0_9746794345`, `::test_qk_spearman_is_plus_0_717647`,
`::test_beta_spearman_is_minus_0_032353`,
`::test_q5_w1_flat_band_distance_to_floor1_is_0_145_to_0_220`.

### 1.1 `1.084523` IS WITHDRAWN, AND THE RED THAT WITHDREW IT

The node was written against the published value and went **RED on its first run**,
verbatim, from the data — **not** from a mutated assertion:

```
E       AssertionError: prose 1.084523, measured 1.0845223424
E       assert 6.575759790017344e-07 < 5e-07
E        +  where 6.575759790017344e-07 = abs((1.084522342424021 - 1.084523))
```

`[RUN]` `Q2/W3 first escape at i=1: t_i=-2.483118 not in [-1.398595, -1.084522]`.

**The Inspector's diagnosis is confirmed arithmetically.** `V20_R15_IT11_INSPECTOR.md:59-60`
called `1.084523` a hand-subtraction of two printed numbers. It is: `2.483118 − 1.398595`,
two **six-decimal displays**. The true excess is `1.0845223424`, which **displays as
`1.084522`**. The published last digit is wrong by `6.58e-7`.

**`1.084523` is withdrawn.** The node asserts `1.0845223424` at `5e-10` — ten times tighter
than any digit ever printed for it — on `ceq/arm_pl`'s own seed-0 `sign_flip_gate` draw,
which **is** W3's arm, so this constant is at least measured on the right object.

### 1.2 `0.9746794345` IS BOUND AND THE NODE SAYS WHAT IT IS WORTH

`[RUN]` `Q2/W1 d=20: R2_1=0.050000000000000  err_1=0.9746794345`; the node also prints
`closed form sqrt(1-1/20)=0.974679434480896`. **The Inspector is right that this is
arithmetic, not evidence** — the block is `I_d` with a zero row. The node is written anyway,
because a published constant with no node is worse than one with a weak node, and the node
**states its own weakness in its second print line** rather than in prose a reader must find.
It does not rescue the cell, and §2 does not use it to.

### 1.3 THE TWO SPEARMANS — THE COORDINATOR'S NUMBERS, NOW HELD

`+0.717647` and `−0.032353` were computed in a shell at it.8 to adjudicate this office's
contradiction with VENUS, `[RUN]` by that office and bound by nothing: `grep -rn spearman`
over `tests/ scripts/ ceq/` returned nothing and `beta` appeared **zero times** in the it.7
test file (`V20_R15_IT11_INSPECTOR.md:554`). Both now recompute from **the same sixteen
`arm_smprime` cells the it.7 filing loaded**, by the same loader
(`tests/jupiter/test_v20_r15_it7_q3.py:34-47`), and both reproduce:

```
Q3/W1 rho(qk,   eval_nrmse), n=16 =  0.7176470588   prose +0.717647
Q3/W1 rho(beta, eval_nrmse), n=16 = -0.0323529412   prose -0.032353
```

Ranks are untied on both coordinates (`Σd² = 192` and `702`, both integers), so the midrank
branch is inert here and the two figures are exactly `1 − 6Σd²/(n(n²−1))`. **A number that
decided a dispute between two offices is now held by a node that can go red.**

### 1.4 Q5/W1'S BAND

`[RUN]` `Q5/W1 dist_to_floor over the flat band: min=0.144954 max=0.219610 (n=12)`, all
twelve values printed by the node. `0.144954` and `0.219610` display as `+0.145` and
`+0.220`. The band is `eval_nrmse − floor₁` at the **preregistered**
`floor₁ = 0.7071067811865476` (`CEQ_V20_R15_CONTRACT.md:116`), not at a threshold authored
after the data.

### 1.5 PLANTED NEGATIVES — 6 OF 6, AND NONE IS AN `and False`

`[RUN]`, one pytest invocation per mutation, `JUP_IT12_MUTATE` applied to the **loaded data**,
never to an assertion:

| mutation | node that dies |
|---|---|
| `hull_target` | hull excess |
| `delay_d` | `err_1` |
| `qk_rank` | `qk` Spearman |
| `beta_rank` | `beta` Spearman |
| `band_cell` | the band **and both Spearmans** — it moves `eval_nrmse`, which all three read |
| `q2_domain` | the §2 domain census |

**No node survives its own mutation.** `band_cell` killing three is disclosed rather than
tuned away: it perturbs the variable all three correlate against.

---

## 2. TASK B — Q2 IS GRADED AGAINST BED-K AND THIS OFFICE ALREADY SAID SO

**`V20_R15_IT8_JUPITER.md:474`, this office's own row, four iterations ago:**

> **M16 Hankel — F4 for these wings** — BED-K only; both frozen wings are BED-M; citation
> still refused; `1/d` computed only inside a test — **do not cite it on W1/W3 at all.**

**And `V20_R15_IT6_JUPITER.md:466` grades the same object `F1 with the constant` on W1.**
Two rows, one object, contradictory. **The it.8 row is the correct one; the it.6 grade is
withdrawn.**

`[READ]` `tests/jupiter/test_v20_r15_it6_q2_outside_bound.py:43` — the file's entire `ceq`
surface is `from ceq.hankel import NEG_ENTRY, hankel_block, rank_plus_lower, rank_real`.
The census node asserts it as an **emptiness claim checked against source**, the way the
round's other domain censuses are (`tests/jupiter/test_v20_r15_it6_ldom_census.py`):

- Q2's `ceq` imports are **exactly** that one line — set-equality, not substring;
- `arm_smprime`, `arm_pl`, `arm_phase`, `corpus` appear **zero** times in the file;
- `results/` and `.jsonl` appear **zero** times — it opens no journal;
- both wing test files **do** read journals, so the disjointness is real and not two empty
  sides;
- `ceq/beds/bed_k.py:1,3` still reads `BED-K` and `Companion to ceq/corpus.py's BED-M`.

`[RUN]` `Q2 domain census: BED-K only; intersection with W1/W3 domain = EMPTY`.

### GRADE CHANGE, AND ITS COST

**Q2/W1: `F1 + const` → `F4`, domain-empty for these wings. The cell loses its `+ const`.**
**No BED-M bound was manufactured to save it.** The `1/d` law is still true and still `F1`
**on BED-K**, where nothing in this round is graded.

**Q2/W3 keeps `F1 + const`**, on the corrected `1.0845223424`, because that constant is
measured on `ceq/arm_pl` — W3's own arm — and not on a Hankel block. The `rank₊` nonexistence
half stays struck (`V20_R15_IT11_INSPECTOR.md:503` row 2): the theorem is sound, the `n = 4`
test citing it is two identities.

**Twelve-cell count moves: one F0, six F1, one F2, two F3, two F4 → one F0, five F1, one F2,
two F3, three F4.** This office's table is one cell weaker than it filed.

---

## 3. TASK C — THERE IS NO F0–F4 RUBRIC

**This office concurs with `V20_R15_IT11_INSPECTOR.md:693-701`: no rubric exists.** Searched
`CEQ_V20_R15_CONTRACT.md`, `CONTRACT.md`, `CEQ_V16_CONTRACT.md`, `CEQ_V15_CONTRACT.md`, the
three V15 deltas, `MISTAKES.md`, `STRUCK.md`, `workdonenew.md`, and the `R10_*` /
`LOOP_PROMPT_*` archives. **No definition of F0, F1, F2, F3 or F4 as a cell grade is written
anywhere in the repository.** No disagreement to report — and two facts the Inspector did not
have:

**(a) THE SCALE DOES NOT PREDATE THE ROUND, THOUGH THE CONTRACT SAYS IT DOES.**
`CEQ_V20_R15_CONTRACT.md:57-58` lists `L-GRADE (F0–F4 + HOW-BAD gap)` under **`LAWS: all
standing +`** — the `+` separates the round's new laws from the standing ones, and `L-GRADE`
sits on the **standing** side. **Every F-token used as a cell grade in this repository is in a
`V20_R15_*` file.** Earlier rounds' `F0`–`F4` tokens (`R10_ITERATION_14.md`,
`R10_ITERATION_16.md`, `CEQ_V15_CONTRACT.md`) name **geometric feature families** — a
different scale entirely. **A law the contract cites as standing has no prior text.** That is
a finding about the contract, not about any cell.

**(b) THE CONTRACT DOES DEFINE TWO OF THE FIVE TOKENS — FOR A DIFFERENT OBJECT.**
`CEQ_V20_R15_CONTRACT.md:64-65`, `L-CERT`: *"every sparsity mask ships with its certificate:
**F0 (exact)** or **F1 (concentration bound, δ printed)**."* That governs **masks**, not
cells, and says nothing about F2–F4. It is the only F-token in the contract carrying content,
and the operational F0/F1 below match it.

### THE OPERATIONAL RUBRIC THESE TWELVE GRADES WERE ACTUALLY ASSIGNED UNDER

**Derived from use, not from the contract, and presented as neither.** Read off every cell
this office graded, so the table can be checked against something:

| grade | what it meant in use | instance |
|---|---|---|
| **F0** | exact — a theorem or proof object, no tolerance | Q1/W3, the Lean half |
| **F1** | *"with the constant"* — a bound **and** its numeric constant, measured on the graded object | self-gloss at `V20_R15_IT6_JUPITER.md:496`; matches `L-CERT`'s F1 |
| **F2** | a direction or ordering established; **no constant** | Q3/W1, `V20_R15_IT7_JUPITER.md:262` |
| **F3** | the claim is a **fact about the harness**, not about the theory | Q4, `S, D = 64, 24` at `scripts/v15_r1.py:137` |
| **F4** | the theory's domain is **empty for the graded wings** | M11, M12, M16 at `V20_R15_IT8_JUPITER.md:471-474`; now Q2/W1 |

**Two defects in it, stated against itself.** **F4 is overloaded three ways** — unattempted,
struck, and domain-empty — exactly as the Inspector charges at
`V20_R15_LEAP_LEDGER.md:24-25,28,130-131`; a reader cannot tell which from the token. And
**F2 and F3 are not ordered**: F3 as used is not "worse than F2", it is a different kind of
failure. **Twelve grades were assigned on a five-token scale that is not a scale.**

### 3.1 WHERE THIS OFFICE DIVERGES FROM SATURN

**SATURN reached the same verdict independently** — `tests/saturn/test_v20_r15_it12_saturn.py`
carries `::test_F0_through_F4_are_never_characterised_together` and a searcher calibrated
against `MISTAKES.md:804` (V-15) and `:117` (V-7) with a **planted positive**
(`::test_the_searcher_finds_a_planted_rubric`), which is the stronger instrument: it proves
the search could have found a rubric had one existed. **Two offices, two methods, one
answer: there is no rubric.**

**The divergence is on the operational content.** SATURN's planted rubric text reads
`F0 exact, F1 bounded with the constant, F2 partial, F3 failed instance, F4 unattempted`.
That is a **plant**, written to calibrate a searcher, not a reading of use — and on two rows
it does not match how this office actually graded:

| | SATURN's plant | derived from this office's twelve grades |
|---|---|---|
| **F3** | "failed instance" | **a fact about the harness, not the theory** — Q4 is not a failed instance, `S, D = 64, 24` is a *true* statement about `scripts/v15_r1.py:137` |
| **F4** | "unattempted" | **domain empty for the graded wings** — M16 was attempted, run, and reproduced to machine precision; its domain is what is empty |

**If SATURN's it.12 files that text as the rubric, this office disagrees with it**, and the
disagreement is evidence for the finding rather than against it: two readings of an
undefined scale disagree on two of its five tokens.

**Replacement route, and it is contract text, not a run:** `L-GRADE` needs the five rows above
written into `CEQ_V20_R15_CONTRACT.md`, with F4 split into `F4-empty` and `F4-unattempted`
and `STRUCK` made a separate token from both. **0 GPU-s.**

---

## 4. HOW-BAD GAPS ON EVERY EXIT

| exit | HOW-BAD gap | replacement route |
|---|---|---|
| **A — `1.084523` withdrawn** | the wrong digit **shipped in three prose sites for six iterations** and was caught only when a node was finally written; the same hand-subtraction-of-displays mechanism is unaudited elsewhere in the round | a repo-wide scan for prose constants formed by arithmetic on other prose constants — grep-able, 0 GPU-s |
| **A — `err_1` bound** | the node checks arithmetic on `I_d`, not a construction; **binding it changed nothing about what the cell knows** | the `equilibrium_hop_reading` k-ladder named at `V20_R15_IT11_INSPECTOR.md:490`, so a sign error stops being invisible |
| **A — the two Spearmans** | `n = 16`, one box, two banked journals; **no CI is asserted**, and the Inspector's `−0.0324` CI `[−0.520, +0.471]` **still covers zero** | a bootstrap CI node on the same sixteen cells; 0 GPU-s |
| **A — Q5/W1's band** | the band is bound at `5e-4`, the width of the printed digits; the **predicted** `+0.20 … +0.27` is still prose and still has no node | assert the prediction interval too, or withdraw it at it.13 |
| **B — Q2/W1 → F4** | this office **filed two contradictory grades for one object** at it.6 and it.8 and neither report noticed; no node compares grades across reports | a ledger node asserting each object carries exactly one live grade |
| **C — no rubric** | **twelve grades, none falsifiable**, and this office assigned all twelve without once asking for the definition | the five rows in §3 written into the contract, F4 split, STRUCK separated |

---

## 5. TREE AND DISCIPLINE

**Written this iteration:** `tests/jupiter/test_v20_r15_it12_constants.py` (new),
`V20_R15_IT12_JUPITER.md` (new), `house-events.jsonl` (append).
**No git write. Nothing touched Kaggle. No file under `kaggle/` was read or written.**
**Zero moons wrote code**; one read-only search agent ran §3's archive sweep.

`[RUN]` `python -m pytest tests/jupiter/ -q -p no:randomly` → **`1 failed, 161 passed in
48.10s`**. The single failure is
`test_v20_r15_it4_merge_is_unexercised.py::test_the_it2_merge_verdict_covers_the_trained_record`
— an it.4 node, **untouched this iteration**, standing RED on purpose because the it.2 merge
verdict is still uncovered by the trained record (`0 of 8` W3 cells have
`a_hat_max <= 1.0`). Stated so no reader mistakes it for damage from this iteration's file.

**SCOREBOARD.** Four constants bound, one withdrawn as wrong. **Q2/W1 `F1 + const` → `F4`;
the table is one F1 weaker than it filed.** The rubric the round has been graded against does
not exist, and the contract calls it standing.
