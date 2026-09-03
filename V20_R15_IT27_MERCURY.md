# V20 R15 · it.27 · MERCURY — THE UNCITED CLASS OUTSIDE §2, AND THE SCOPE WHERE THE INSTRUMENT READS ZERO

**All readings 2026-09-02, 12:33–12:50Z, local CPU, branch `v17k-gate0`.**
**This office wrote exactly two files: this one and `tests/mercury/test_v20_r15_it27_uncited_class.py`.**
Other modifications visible in `git status` predate this iteration and are not this office's.

**Content digests of every file read or written (sha256[:16], raw bytes, at the time of reading):**

| file | digest | read at |
|---|---|---|
| `V20_R15_THEORY_TABLE.md` | `3c4d1b270049f6ff` | 12:33Z **and** 12:37Z — the subject of every count below |
| `V20_R15_THEORY_TABLE.md` | `942e4208893444cd` | **12:40Z — JUPITER's edit, landed mid-iteration. See §0 below.** |
| `ceq/kdata.py` | `c810c887956af343` | 12:36Z |
| `tests/saturn/test_v20_r15_it12_saturn.py` | `4ffd05cd434df811` | 12:36Z |
| `V20_R15_JOURNAL.md` | `aa128cb7ecb8ca94` | 12:36Z |
| `tests/mercury/test_v20_r15_it27_uncited_class.py` | `9e9d55adcc92386a` | written this iteration |

**No HEAD SHA is published here** — `0 of 22` files under `tests/mercury/` are known to git.

## 0 — THE TABLE MOVED UNDER THE SWEEP. IT MOVED AT 12:40Z AND EVERY COUNT SURVIVED.

The sweep ran against `3c4d1b270049f6ff`, **the same digest it.26 measured**; JUPITER's three
replacements were absent at 12:33Z and still absent at 12:37Z. At **12:40Z the digest was
`942e4208893444cd`** — the fourth mid-iteration edit of this table this round. **All three
it.26 replacements landed**, verified by their new text and not by his report:

```
[RUN] sha256 V20_R15_THEORY_TABLE.md            12:33Z / 12:37Z -> 3c4d1b270049f6ff
[RUN] sha256 V20_R15_THEORY_TABLE.md            12:40Z          -> 942e4208893444cd
[RUN] grep -c "ceq/beds/bed_k.py:236"   -> 1   (M-25c landed)   12:40Z
[RUN] grep -c "v15_r1.py:547-558"       -> 1   (M-25a landed)   12:40Z
[RUN] grep -c "T_STAR = 2"              -> 1   (M-25b landed)   12:40Z
[RUN] mercury it.27 screen re-run on 942e4208893444cd            12:40Z
      §0 118/9   §1 22/0   §2 382/64   §3 72/16   §4 117/5   §5 42/9  — IDENTICAL
```

**THE SWEEP AND HIS EDITS DO NOT DISAGREE, AND THAT IS ITSELF THE READING.** Every
float-shaped token in §0, §3 and §4 is the same token at the same line **except one**:
`0.7071067811865476` moved from `:71` to `:72`. **All four counts below hold on both
digests. Only the `M-25b` line number in §2 of this report changes: `:71` → `:72`.**

**And `M-25b` is still CITED under this instrument after the repair** — it now cites
`scripts/v15_r1.py:586` **and** `:138`, which is a landing repair, not a citation added
where none was. **A landing repair does not move the uncited count.** That is the disjointness
in §2 of this report, demonstrated on a live edit rather than argued.

The three REDs were re-run against `942e4208893444cd` at 12:40Z and are **still RED**: his
edit did not touch §3.2 or §4.3.

**Every count below is against `3c4d1b270049f6ff`, confirmed to hold on `942e4208893444cd`,
and against no other table.**

| line | it.27 |
|---|---|
| **UNCITED, §0** | **`2` across `1` of `3` subsections**, denominator **`6`** |
| **UNCITED, §3** | **`8` across `4` of `6` units**, denominator **`13`** |
| **UNCITED, §4** | **`0` across `0` of `3` subsections**, denominator **`0`** — the instrument reads zero because **§4 asserts no float-shaped constant at all** |
| **UNCITED NAMED ARTIFACTS, §4** | **`2` across `1` of `3` subsections**, denominator **`5`** |
| MARKER NODE | `tests/mercury/test_v20_r15_it27_uncited_class.py` — **3 RED / 3 GREEN**, RED against the unmutated table |
| LANDINGS CENSUS | **`21 → 24 of 129`**, one clause opened, **`0` new defects** |

**These four numbers are NOT pooled and no rate is offered.** Neither is it.26's `29 of 62`
on §2. Five denominators, five populations, five separate readings.

---

## 1 — THE SCREEN, AND THE ONE PLACE IT DISAGREES WITH it.26

The screen was **re-implemented from the it.26 description, not reused**, and on §2 it
returns `382` numeric / `64` float-shaped where it.26 recorded `240` / `63`.

```
[RUN] mercury it.27 screen, all six sections, table 3c4d1b270049f6ff          12:35Z
      §0  118 numeric / 9 float      §1  22 / 0       §2  382 / 64
      §3   72 numeric / 16 float     §4 117 / 5       §5  42 / 9
```

**That disagreement is a finding about the procedure and is reported as one.** The `240`/`63`
figures were never expressed as a regex in a file, so **the it.26 denominator is not
reproducible from the it.26 report** — an independent office reading the same table under
the same stated rule lands `64`, not `63`. **it.26's `62` hand-read denominator is not
withdrawn** (the hand count, not the screen, is the number there), but the screen behind it
is an office-local artefact and **no future office should quote `63` as reproducible.**
This office's screen rule is stated here in full:

> A token counts if it matches a float shape (`d.d`, `d.dEd`) **outside** every backticked
> span containing a source-file extension. Section references (`§0.2`, `4.1`) are then
> **rejected by hand** as pointers, not constants.

---

## 2 — SCOPE §0. **`2` UNCITED ACROSS `1` OF `3` SUBSECTIONS, DENOMINATOR `6`.**

Unit = subsection (`§0.1`, `§0.2`, `§0.3`). Screen `9` → **`3` rejected as section
references** (`0.1` @ `:20`, `0.2` @ `:69`, `0.3` @ `:93`; unmoved by the 12:40Z edit) → **denominator `6`**.

| line | constant | ruling |
|---|---|---|
| `:57` | `1.421901019003236` | **CITED** — `V20_R15_LEAP_LEDGER.md:131`, same clause |
| `:71` (`:72` after JUPITER's 12:40Z edit) | `0.7071067811865476` | **CITED, and this is the point** — see below |
| `:78` | `1.0` | excluded, definitional (`h_hat = 1` is the definition of the threshold) |
| `:90` | `0.0` | excluded, definitional — the same exclusion it.26 made on the exact oracle |
| `:101` | `ρ = +0.7029` | **UNCITED** |
| `:102` | `+0.5197` | **UNCITED** |

> ### THE TWO CLASSES ARE DISJOINT, AND §0.2 PROVES IT.
> The task named `0.7071067811865476` @ `:71` as a same-shape constant to sweep. **It is
> not in this class.** `:71` offers `scripts/v15_r1.py:586`, so under the it.26 rule it is
> **CITED**; what is wrong with it is that the pointer **does not land** — the value is in
> no line of that file. That is `M-25b`, the LANDING class. **A constant can fail the
> landings census and pass the uncited screen, and the reverse.** The two instruments do
> not overlap and neither subsumes the other. Sweeping §0 was the only way to see that.

**`+0.7029` and `+0.5197` are the same claim §2's Q4/W1 carries at lower precision** — it.26
counted `ρ = +0.70` there as uncited. **The claim is uncited in two sections at two
precisions, and the two counts are independent.** `scripts/v15_r1.py:249` / `:267` are cited
in the same paragraph and are the **timer**, the producer of `secs` — not of a correlation.
Under the `M-25c` ruling (a registry entry and its artefact are two citations, not one), an
input's producer is not the statistic's producer.

---

## 3 — SCOPE §3. **`8` UNCITED ACROSS `4` OF `6` UNITS, DENOMINATOR `13`.**

Unit = the four clause rows of the ticket table, plus `§3.1`, plus `§3.2`. Screen `16` →
**`3` rejected as section references** (`0.2` @ `:239`, `3.1` @ `:242`, `3.2` @ `:270`) →
**denominator `13`**.

| unit | uncited | the constants |
|---|---|---|
| header `:231-233` | 0 | `0.5` carries `CEQ_V20_R15_CONTRACT.md:125` in its own sentence |
| **clause (1) row** | **4** | `0.0016`, `0.0032` (W1 CP-lower), `0.4762`, `0.5156` (W3 CP-lower) — **no pointer in the row** |
| clause (2) row | 0 | no constant |
| clause (3) row | 0 | no constant |
| **clause (4) row** | **1** | `34.4` GB Kaggle attach |
| **§3.1** | **2** | `0.5156` @ `:259`, `0.4762` @ `:261` — restated to carry the load-bearing tail argument, still no producer |
| **§3.2** | **1** | `p = 8.544e-05` |

**A RULING THIS SCOPE FORCES, OFFERED FOR THE INSPECTOR — PARENT YES, SIBLING NO.**
`0.5` recurs at `:237` inside the clause (1) row with no pointer of its own. It is ruled
**CITED** because the sentence introducing the table, two lines above, cites the criterion.
it.26 ruled the opposite way for Q2/W3's `−1.0`, whose pointer sat in a **sibling** cell.
**A block's own header is the parent of its rows and its citation reaches them; a peer's is
not and does not.** Without this rule the §3 number is `9`, not `8`.

**THREE EXCLUSIONS, STATED SO THEY CAN BE REVERSED.** `0.0625` (= `1/16`), `0.7500`
(= `12/16`) and `0.0238` (= `0.5 − 0.4762`) are **exact re-expressions of numbers printed
beside them**, not separate claims. Ruling them claims moves §3 from **`8` to `11`**.

**THE ONE THAT MATTERS.** `0.4762` and `0.5156` are the two readings of `⟨CLAUSE_1_TAIL⟩`,
and §3.1 says in its own text that the choice between them decides whether the arena has a
structural hole or merely an unused one. **The most consequential pair of numbers in the
arena section is asserted four times across two units and pointed at zero times.** Neither
is a landing defect; nothing claims to land them. The census cannot see this.

---

## 4 — SCOPE §4. **THE INSTRUMENT READS `0` OF `0`, AND THAT IS THE RESULT.**

```
[RUN] mercury it.27 screen, §4, lines 282-392                                12:35Z
      5 float-shaped tokens:  4.1 @ :284,  4.2 @ :319,  0.3 @ :327,
                              4.3 @ :333,  0.3 @ :367
```

**All five are section references. After hand rejection the denominator is `0`.**

> ### `0 uncited across 0 of 3 subsections, denominator 0`.
> **§4 — the section that rules what the leap gets and what it must not be asked — asserts
> not one numeric constant.** Its claims are files, functions, symbols and arms.
> **The numeric instrument is completely blind to the section that decides the round's
> output**, and it.26's `29` would have stayed `29` no matter what §4 said. This is the
> argument for the named-artifact extension, and it is not an argument from analogy: it is
> a measured zero denominator.

---

## 5 — NAMED ARTIFACTS IN §4. **`2` UNCITED ACROSS `1` OF `3` SUBSECTIONS, DENOMINATOR `5`.**

**POPULATION RULE, stated so it can be attacked.** A backticked token in §4 that names a
**code artifact** — a module-level symbol, a function, a bed, or an arm of a detector — and
is not itself a `path:line`. **Rejected from the population, by name:** `scripts/` (a bare
directory in "zero callers under `scripts/`", a path not an artifact), `1/d` (a
mathematical bound), `(g,s,q,k)` (a key tuple), and the three code fragments quoted inside
`[RUN]` blocks. Grade tokens (`F4`), ledger row ids (`L-13`) and censuses (`0 of 24`) are
not artifacts. **Denominator `5`.**

| line | artifact | ruling |
|---|---|---|
| `:339` | `inadmissible_leapable_rows()` | **CITED** — `tests/saturn/test_v20_r15_it12_saturn.py:319` |
| `:379` | `KNOWN_INADMISSIBLE` | **CITED** — `:343` |
| `:382` | `ROUND_NOUNS` | **CITED, weakly** — `:353` is the *exerciser*, not the definition. `M-25c`-shaped; ruled CITED under the it.26 rule (*some* pointer attaches) and flagged here rather than counted. |
| `:383` | the **`none` arm** | **UNCITED** |
| `:388` | **`BED_SPECS`** | **UNCITED** |

**BOTH LAND ON THE SAME PARAGRAPH, AND IT IS THE PARAGRAPH THAT WITHDRAWS A LEDGER LINE.**

1. **`BED_SPECS`.** §4.3 withdraws `V20_R15_LEAP_LEDGER.md:131`'s "the one registered bed"
   sentence on the strength of what `BED_SPECS` returns, and offers **no pointer for
   `BED_SPECS` and no `[RUN]`**. The same claim at `:224` in §2 carries **both** — a `[RUN]`
   and `V20_R15_IT13_MERCURY.md:189`. **The identical claim is provenanced in §2 and bare in
   §4, in one file, and the bare copy is the one doing the withdrawing.**
2. **The `none` arm.** §4.3's own text says this arm "produces four of the five findings"
   and "has no control at all". It is named with no pointer while its **sibling
   `ROUND_NOUNS` arm carries one on the very same line.** A reader told which arm is
   uncontrolled is given no way to open it.

**This is the class the task predicted and it costs the same as `M-25c`:** an artifact named
without a pointer sends the leap nowhere, and here the artifact is the evidence for a
withdrawal.

---

## 6 — MARKER NODE. **3 RED / 3 GREEN, RED AGAINST THE UNMUTATED TABLE.**

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it27_uncited_class.py -q    12:37Z
      3 failed, 3 passed in 0.50s
      RED  : m27a_sec4_bed_specs_correction_cites_the_symbol
             m27b_sec4_none_arm_is_cited
             m27c_sec3_2_fisher_p_cites_its_record
      GREEN: m27a_control_bed_specs_is_defined_at_kdata_472
             m27b_control_none_arm_is_at_saturn_329
             m27c_control_fisher_p_is_banked_in_the_journal
```

Every RED asserts `V20_R15_THEORY_TABLE.md`; every GREEN asserts the unmutated repo. **All
three go GREEN on a table edit alone and on no code change** — if a control ever goes RED,
the defect was "repaired" by moving the code, which is not the repair.

### THREE REPLACEMENTS — JUPITER'S TO APPLY, NAMED NOT MADE

```
[RUN] sed -n 472p ceq/kdata.py                            ->  BED_SPECS = {          12:36Z
[RUN] sed -n 329p tests/saturn/test_v20_r15_it12_saturn.py
          ->  if head.startswith("none") or not head:                                12:36Z
[RUN] grep -n 8.544e-05 V20_R15_JOURNAL.md
          ->  1552:Fisher exact, one-sided - arm_pl vs arm_smprime: p = 8.544e-05    12:36Z
```

> **`M-27a`, `V20_R15_THEORY_TABLE.md:388`:**
> from `` `BED_SPECS` returns `['bed_m', 'bed_k', 'bed_1']` ``
> to `` `BED_SPECS` @ `ceq/kdata.py:472` returns `['bed_m', 'bed_k', 'bed_1']` (`[RUN]` banked @ `V20_R15_IT13_MERCURY.md:189`) ``
>
> **`M-27b`, `V20_R15_THEORY_TABLE.md:383`:**
> from `` the **`none` arm** `` to `` the **`none` arm** (`tests/saturn/test_v20_r15_it12_saturn.py:329`) ``
>
> **`M-27c`, `V20_R15_THEORY_TABLE.md:278`:**
> from `` **`p = 8.544e-05`** `` to `` **`p = 8.544e-05`** (banked @ `V20_R15_JOURNAL.md:1552`) ``

**`M-27c` is a repair of an UNCITED constant whose producer already existed in the record.**
It cost one `grep`. **Of the `10` uncited constants counted in §0 and §3, this is the only
one whose producing record this office located inside the wall clock** — the other nine are
priced, not repaired, and no claim is made that they are equally cheap.

---

## 7 — LANDINGS CENSUS. **`21 → 24 of 129`. ONE CLAUSE. `0` NEW DEFECTS.**

The clause opened, named as the task required: **§4.3's detector clauses, table lines
`339`, `379`, `382`** — the three `tests/saturn/test_v20_r15_it12_saturn.py` occurrences.
it.25's `21` came from table lines `71, 95, 104, 163–170, 171, 203, 304`; **none of these
three is in that set.**

| table line | citation | what is there | verdict |
|---|---|---|---|
| `:339` | `…it12_saturn.py:319` | `def inadmissible_leapable_rows() -> dict[str, str]:` | **LANDS** |
| `:379` | `…it12_saturn.py:343` | `KNOWN_INADMISSIBLE = {"V-it7", "L-2", "L-9", "L-13", "L-14"}` | **LANDS** |
| `:382` | `…it12_saturn.py:353` | `def test_the_FIELD_detector_fires_on_a_planted_violation():` | **LANDS** |

**`24 of 129` is a count, not a rate. No extrapolation to `129` is offered** and the
`~5 office-iteration` remainder price from it.25 is unchanged. **The three that land are in
the very paragraph carrying both uncited artifacts** — `129 of 129` landing would not have
found `M-27a` or `M-27b`, which is the INSPECTOR's it.25 ruling restated as a measurement.

---

## 8 — WHAT WAS NOT REACHED

- **Named artifacts were swept in §4 only.** §2's and §3's named-artifact populations
  are **unmeasured**; it.26's `29` remains numeric-constants-only for §2 and this iteration
  did not change that. **No §4 result is carried to any other section.**
- **§5 (`42` numeric / `9` float-shaped, lines `393-443`) was screened and NOT hand-read.**
  It is the table's own limits section; the screen ran, the ruling did not.
- **§1 was screened and is empty of float-shaped constants** (`22` numeric, `0` float) —
  a real zero, reported so the six sections sum.
- **Integer-shaped uncited constants remain outside every denominator on this page**, as at
  it.26. A bare integer asserted with no pointer is not counted anywhere in this round.
- **The `~275` / `309.047` derivation chain in Q4/W1 was not re-opened**; it stays as it.26
  left it.

## 9 — LIMITS

Five counts, five denominators, five populations, **and they must not be added.** All are
one office's hand ruling against a table whose digest is `3c4d1b270049f6ff` at both readings
this iteration and **re-screened against `942e4208893444cd` after his 12:40Z edit landed**,
where every token population is identical and one line number moves (§0 of this report).
**A count quoted against any third digest is a different count.** The §3 number depends on
the parent-not-sibling ruling in §3 of this report (`8` with it, `9` without) and on three
named exclusions (`8` with them, `11` without); both are stated so they can be reversed.
The §4 named-artifact denominator of `5` is a population this office defined this iteration
and no other office has ruled on it. The screen behind every denominator is float-shaped and
this office's own re-implementation, which **disagrees with it.26's on §2 by one token** —
reported in §1 rather than reconciled, because reconciling it would have cost the sweep.
Three replacements are named, not applied: the table is JUPITER's artifact and this office
did not edit it. No git writes. Nothing touched Kaggle.
