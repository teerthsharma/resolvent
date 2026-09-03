# V20 R15 it.22 — MERCURY: THE SECOND CENSUS, AND THE ARGUMENT LAYER PRICED

**Office:** MERCURY (arena and measurement).
**Deliverable:** an INDEPENDENT re-take of the citation census in
`V20_R15_THEORY_TABLE.md`, with this office's own digest published beside JUPITER's,
plus a measured argument-support rate.
**Test:** `tests/mercury/test_v20_r15_it22_independent_census.py` — 4 GREEN, 3 RED
against unmutated code.

---

## §0 — WHAT WAS BOUGHT, IN ONE LINE EACH

| | |
|---|---|
| **Independent recount** | **129 occurrences, 103 unique pointers, 37 distinct files, 0 unresolvable.** **AGREES with JUPITER on every total.** |
| **Independent split** | **116 scored / 13 refused.** AGREES with J-20b, and the 13 were derived here, not taken. |
| **Independent digest** | `MERCURY-CENSUS-R1 = 7b1e67cd9e5d049fff93a7795cd6cce1aabd5b95b6a42892870ceb0ad1f6e9b4`, recipe named in §2. **Not comparable to `WANT_SEAL` by construction** — different object, named as such. |
| **Argument-support rate** | **9 of 12.** Twelve occurrences drawn at MERCURY seed 2209, each read as a reader. **Three do not support their cell's claim, and all three are GREEN under every location instrument the round owns.** |
| **Not reached** | §6. |

---

## §1 — WHAT WAS RE-TAKEN, AND WHAT WAS DELIBERATELY NOT IMPORTED

JUPITER priced this iteration himself at it.20:

> `WANT_SEAL` is tamper-evident, not tamper-proof — the seal sits in the file the
> audited office writes. *"The real price of tamper-proof is **one iteration of a
> second planet re-taking the census and publishing an independent digest.**"*

`tests/mercury/test_v20_r15_it22_independent_census.py` **imports nothing from
`tests/jupiter/`**: no `CENSUS`, no `WANT_SEAL`, no `HEADING_CENSUS`, no `REANCHORS`,
no `LIVE_FILES`, no `table_citations`, no `line_at`. The population is rebuilt from
the table by one regex. The independence is the deliverable; a recount that imports
the manifest it is checking asserts nothing, which is JUPITER's own MANIFEST
PROVENANCE rule turned back on his file.

**The 116/13 split was reached, not taken.** This office extracted the 129, grouped
by file, and found exactly three files that the round appends to every iteration
(`V20_R15_LEAP_LEDGER.md` ×10, `V20_R15_JOURNAL.md` ×2, `house-events.jsonl` ×1 =
**13**). That is J-20b's refusal set, arrived at from the other side.

---

## §2 — RECIPE `MERCURY-R1`, NAMED SO THE DIGESTS CAN DISAGREE HONESTLY

The precedent is it.19: SATURN and JUPITER published **different**
`THEORY-CELLS-SHA256` and **both were correct, because each named its recipe.**
Mine, in full:

1. A CITATION is a backtick-delimited `path:lineno` token in
   `V20_R15_THEORY_TABLE.md`. One regex, no manifest:
   `` `<path with a dot-extension>:<digits>` ``.
2. OCCURRENCES counts tokens. POINTERS counts distinct `(path, lineno)`.
3. `MERCURY_CENSUS_R1` = `sha256` over, for each unique pointer in sorted order,
   `f"{path}:{lineno}\t{current_line_text.strip()}"`, joined by newlines.

**`MERCURY-CENSUS-R1 = 7b1e67cd9e5d049fff93a7795cd6cce1aabd5b95b6a42892870ceb0ad1f6e9b4`**
`MERCURY-POINTERS-R1 = 3bc81fd1a00ce8540237edcaceef0281fb7ec7cea21136a7883337787eb66b7a`
(the same census with the line CONTENT dropped — pointers only.)

### 2.1 THE TWO SEALS DO NOT SUBSUME EACH OTHER, AND THAT IS WHAT THE SECOND CENSUS BUYS

**This is the finding of §2, and it is not "the digests agree" — they cannot agree,
because they digest different objects.**

| | `WANT_SEAL` (JUPITER) | `MERCURY-CENSUS-R1` |
|---|---|---|
| digests | the substrings **JUPITER chose** | the cited **lines as they stand in the repo** |
| lives in | the file the audited office writes | this file, a second office |
| goes RED when | JUPITER's manifest is edited | **a cited file moves under the table** |
| **blind to** | **a cited file moving, if the manifest is moved with it in the same edit** | a `want` being re-chosen without the line changing |

**The blind spot in the left column is exactly the one J-20a was built to cover and
covers only by making the edit two-line instead of one-line.** `MERCURY-CENSUS-R1`
closes it from outside: an office that recomputes `WANT_SEAL` to follow a moved line
still leaves this digest RED, because this digest never asked the manifest anything.
Neither seal is tamper-proof alone. **Together they require an office to edit two
files in two directories owned by two planets, which is the practical article the
round was buying.**

---

## §3 — RECOUNT: THE NUMBERS, WITH `[RUN]`

`[RUN] python -m pytest tests/mercury/test_v20_r15_it22_independent_census.py -q`
→ **`3 failed, 4 passed in 0.62s`**, the three failures being §4's findings.

| quantity | MERCURY it.22 | JUPITER it.20–21 | |
|---|---|---|---|
| occurrences | **129** | 129 | **AGREE** |
| unique pointers | **103** | 103 | **AGREE** |
| distinct files | **37** | not published as a total | — |
| unresolvable (missing file / out of range) | **0** | 0 | **AGREE** |
| scored by line | **116** | 116 | **AGREE** |
| refused under J-20b | **13** | 13 | **AGREE** |

**The agreement is evidence and it was not assumed.** It is also worth naming what
agreement does and does not buy: **it certifies the POPULATION, not the LANDINGS.**
This office reproduced JUPITER's counts; it did not re-derive his 103 `want` strings,
and does not claim to have.

### 3.1 A CORRECTION TO MY OWN FIRST TAKE, BANKED RATHER THAN QUIETLY FIXED

The first extraction returned **110**, not 129, because the extension list omitted
`.lean`. `lean/CEQ/V16Domain.lean` carries **18** occurrences and `lean/lakefile.lean`
**1** — `110 + 19 = 129`. **A census whose recipe under-specifies the path grammar
under-counts silently and reports a clean total.** It is in this report because the
error is the same class as the ones §4 catches, committed by the office doing the
catching.

### 3.2 THE LIVE-FILE CRITERION IS ASSERTED IN BOTH OFFICES, AND MY ATTEMPT TO MEASURE IT FAILED

I tried to derive LIVE from `git log --oneline -- <file> | wc -l`. It does not
separate the set:

| file | commits touching it | J-20b status |
|---|---|---|
| `house-events.jsonl` | **66** | LIVE |
| `MISTAKES.md` | 22 | not live |
| `V20_R15_LEAP_LEDGER.md` | **0** | LIVE |
| `V20_R15_JOURNAL.md` | **0** | LIVE |
| `CEQ_V20_R15_CONTRACT.md` | **0** | not live |

**The two files J-20b refuses hardest have zero commits, because they are UNTRACKED
working-tree files** — `git ls-files --error-unmatch` fails on
`V20_R15_LEAP_LEDGER.md`, `V20_R15_JOURNAL.md`, `CEQ_V20_R15_CONTRACT.md` **and
`V20_R15_THEORY_TABLE.md` itself.** So commit history cannot see the growth J-20b
refuses, and `CEQ_V20_R15_CONTRACT.md` is untracked-and-not-refused while the ledger
is untracked-and-refused, on a distinction no measurement in this repo currently
makes. **`MERCURY_LIVE` in my test is therefore the same asserted three-name tuple
JUPITER has, reached independently but not MEASURED.** Named as a limit, not repaired.

---

## §4 — THE ARGUMENT LAYER: **9 OF 12**

J-21c retired ARGUMENT as uncertifiable by instrument and was right to. **What was
missing was a number.**

**Sampling frame.** The 116 scored occurrences. **Seed 2209** (`v20 r15 it22`) —
**deliberately not the INSPECTOR's 1520**, so this sample and his 12-at-1520 are
disjoint draws from the same frame. 12 occurrences drawn, each read as a reader: the
cell's claim beside the cited line. No instrument was built; J-21c already retired
that route.

**Ruling rule.** Does the cited line **support** the proposition the cell attributes
to it, or does it merely **contain the words**?

| # | citation | cell's claim | ruling |
|---|---|---|---|
| 1 | `V20_R15_IT12_INSPECTOR.md:233` | SATURN's rubric ruled NOT BOUND | **SUPPORTS** — line is the STRIKE itself |
| 2 | `CEQ_V20_R15_CONTRACT.md:239` | the contract grades M14 **F3** here | **NO** — §4.1 |
| 3 | `V20_R15_IT13_MERCURY.md:146` | repair priced at **206–537 GPU-s** | **SUPPORTS** — row carries the range and the point |
| 4 | `lean/CEQ/V16Domain.lean:165` | `no_prefix_scan_...`, universally quantified over `C` | **SUPPORTS** — the `(C : ℕ → ℂ)` binder is on the line |
| 5 | `lean/CEQ/V16Domain.lean:147` | `lean_log_junk_...` declared here | **SUPPORTS** |
| 6 | `V20_R15_IT12_JUPITER.md:135` | the it.6 grade is withdrawn | **SUPPORTS** — line records `F1 + const → F4` |
| 7 | `tests/.../it12_constants.py:12` | `0.9746794345` bound to `5e-11` | **SUPPORTS**, weakly — §4.4 |
| 8 | `ceq/arm_pl.py:1` | measured `t_1 = −2.483118`, excess `1.0845223424` | **NO** — §4.3 |
| 9 | `tests/.../it12_constants.py:13` | `+0.717647` **and** `−0.032353` **both** bound here | **NO** — §4.2 |
| 10 | `tests/jupiter/test_v20_r15_it7_q3.py:106` | the named `::test_w3_...` is here | **SUPPORTS** |
| 11 | `V20_R15_IT8_JUPITER.md:105` | `path_product` is `O(n·S²)`, `[DERIVED]` | **SUPPORTS** — §4.5 |
| 12 | `scale/negation_scope.py:286` | `equilibrium_oracle` declared here | **SUPPORTS** |

> ### **ARGUMENT-SUPPORT RATE: `9 of 12`.**
> **Denominator is 12. This is a SAMPLED RATE ON 12 OF 116 AND IS NOT EXTRAPOLATED
> TO 129.** The INSPECTOR's `8-in-20` was withdrawn at it.20 partly for being
> presented as one thing and used as another; this number is presented as one thing
> and is to be used as that thing. **What it licenses is "the argument layer has a
> nonzero defect rate and here are three instances", not a count over the table.**

**All three failures are GREEN under every location instrument the round owns**, and
each is a different mechanism.

### 4.1 `THREE_LINES_LOW` — the grade is three lines below the pointer

`V20_R15_THEORY_TABLE.md:53` — *"`F3` in the contract (`CEQ_V20_R15_CONTRACT.md:239`)"*.

`CEQ_V20_R15_CONTRACT.md:239` reads
`M14 CHEEGER STRATIFICATION. φ²/2 ≤ 1−λ₂ ≤ 2φ [V] — corpus`.

**There is no `F3` on the line.** The grade token on it is `[V]`. `F3` first appears
at **`:242`** — *"my crude φ FAILED the sanity check — grade F3 pending an exact
sweep-cut conductance"*. The pointer lands on the right ITEM and the wrong
PROPOSITION, and any `want` anchored on `M14 CHEEGER` — the obvious choice — scores
GREEN forever.

**Second-order, found while checking:** the table says M14 was *"RESOLVED at
**it.17**"*; `CEQ_V20_R15_CONTRACT.md:244` says *"SUPERSEDED **it.19**, RULING
J-17d"*. Both name J-17d. **The iteration disagrees between the cell and the file the
cell cites.** Filed as Open — no `[RUN]` was produced for which is right, so under
the INSPECTOR's it.21 rule it is not a correction.

### 4.2 `COMPOUND_HALF` — two constants asserted at one line; the line carries one

`V20_R15_THEORY_TABLE.md:167` — *"Spearman `qk = +0.717647`, `beta = −0.032353`,
**both** bound @ `tests/jupiter/test_v20_r15_it12_constants.py:13`"*.

`:13` reads `+0.717647           V20_R15_IT7_JUPITER.md:91          5e-7  (prose: 6 dp)`.
**`−0.032353` is at `:14`.** A landing check anchored on `+0.717647` is GREEN, and
the word doing the damage is **"both"** — the cell makes a two-part claim and cites
one line, so *half of a compound claim is certified and the instrument reports the
whole cell green.*

### 4.3 `LINE_ONE_IDIOM` — a file pointer wearing a line number

`V20_R15_THEORY_TABLE.md:158` — *"200 draws, on `ceq/arm_pl.py:1`: `t_1 = −2.483118`,
hull `[−1.398595, −1.084522]`, excess `1.0845223424`"*.

`ceq/arm_pl.py:1` is the module docstring's opening line. **It carries none of those
numbers, and cannot** — the measurement is not in the file at all, it is a run.
Here `:1` means **THE FILE**, and read that way the cell's ROUTE sentence
(*"measured on `ceq/arm_pl.py:1` — W3's own arm"*) is fine. The DECLARATION sentence
is not: it attributes measured values to a line.

**This is a class, not an instance, and the class is measured:** `[RUN]` **8
occurrences of `:1` across 6 files** — `ceq/arm_pl.py`, `lean/lakefile.lean`,
`tests/jupiter/test_v20_r15_it11_q6_oracle.py`, `.../it14_theory_table.py`,
`.../it6_q1_exact_class.py`, `.../it9_q6.py`. **6.2% of the census is a pointer whose
line number is not meant to be read as one**, and the round has no notation that
distinguishes a file pointer from a line pointer, so every line-landing instrument
scores all 8 as lines. GREEN under all of them.

### 4.4 THE ONE I DECLINED TO FAIL, AND WHY IT IS RECORDED ANYWAY

`tests/.../it12_constants.py:12` carries **both** `0.9746794345` and `5e-11`, so the
cell's claim is on the line. **But `:12` is inside the MODULE DOCSTRING** — the file
opens `"""` at `:1` and the constants table at `:10–:15` is prose. The cell says
*"bound to `5e-11` @ `:12`"*; **`:12` asserts nothing — it is the file's index of
what it asserts elsewhere.** Ruled **SUPPORTS** because the proposition is present
and true, and recorded because *"the citation points at documentation of the
assertion rather than the assertion"* is a fourth mechanism that a stricter reader
would count against the rate. **On the strict reading the rate is `8 of 12`.** Both
readings are published; this office scores the loose one.

### 4.5 THE ONE THAT LOOKED WRONG AND IS NOT

`V20_R15_IT8_JUPITER.md:105` names `ceq/arm_smprime.py:163-172` for the `[..., S, S]`
build, while the citing cell says `path_product` @ `ceq/arm_smprime.py:144`. **Both
are right** — `:144` is the `def`, `:163–172` is the body. Checked and cleared rather
than filed, because a correction with no defect behind it is worse than none.

---

## §5 — STANDING ITEM, MENTIONED AND NOT SPENT ON

**This office's it.18 escalation is still open.** The lazy predicate
`eval_nrmse < floor` and the runner's `m + half < floor1` give **identical `(12,16)`
and `(1,16)` on all 40 banked cells** — the round's crossing counts have never
distinguished them. Untouched at it.22 by instruction. It remains the case that
**every crossing count the round has published is invariant to which predicate
produced it**, and no iteration has yet produced a cell that separates them.

---

## §6 — WHAT I DID NOT REACH

1. **The 103 `want` strings were not independently re-derived.** This census
   certifies the POPULATION (129/103/116/13) and the CURRENT LINE CONTENT (via
   `MERCURY-CENSUS-R1`). It does **not** independently confirm that each of
   JUPITER's 103 chosen substrings is the right anchor for its cell. That is a
   second iteration's work and it is the larger half of a true re-take.
2. **The 13 heading-anchored occurrences were not re-taken at all.** J-21a's
   resolver was read but not re-implemented; the 13 are excluded from my sampling
   frame, not judged. **`HEADING_SEAL` has no second census.**
3. **The LIVE criterion is still asserted in both offices** (§3.2). My measurement
   attempt failed and is published as a failure.
4. **The argument sample is 12.** A rate on 12 is what the wall clock bought. **30
   would separate the four mechanisms of §4 from each other; 12 shows they exist.**
5. **No correction was filed against the theory table.** Three defects are banked as
   RED tests with `[RUN]` behind them; the repairs (re-point `:239`→`:242`, split the
   compound at `:167`, adopt a file-pointer notation for the 8 `:1`s) are **named and
   not performed** — they are edits to a file this office does not own.
