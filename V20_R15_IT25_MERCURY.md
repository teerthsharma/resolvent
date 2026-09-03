# V20 R15 — it.25 — MERCURY (arena and measurement)

**THE LANDINGS CENSUS, INDEPENDENTLY TAKEN.**
Wall clock armed `12:03:01Z`. All readings below dated. Branch `v17k-gate0`.
No git writes. Nothing touched Kaggle.

Theory table read at mtime `2026-09-02 11:51:49Z`, `grep -c "" = 443`.
**Unchanged for the whole of this iteration** — the mid-iteration edit that
corrupted it.24 did not recur.

---

## 1 — THE HEADLINE

```
LANDINGS RULED BY HAND      21 of 129 occurrences        3 DEFECTS
POPULATION RE-COUNTED       129 of 129                   0 disagreements
```

**The population census is closed by a second office. The landings census is
NOT, and this report does not claim it is.** 108 of the 129 were screened but
not read; the screen's verdict on them is not evidence and is named as such
in §5.

---

## 2 — THE RECOUNT. `129 of 129`, INDEPENDENT, AND IT MOVES A PRIOR FINDING

Recipe `MERCURY-R2`: three regexes over the table, one per notation, **no
manifest, no import from `tests/jupiter/`**. `WANT_SEAL`, `CENSUS`,
`HEADING_CENSUS`, `REANCHORS`, `REISSUED` were not read.

```
[RUN] MERCURY-R2 step 1                                          12:05:14Z
   occurrences 129   unique pointers 103   files 37
   notation  plain 120  +  file 8  +  range 1  = 129
   extension .md 40  .py 69  .lean 19  .jsonl 1 = 129
```

**Every figure matches JUPITER's `129 / 103 / 37` and both histograms, reached
without touching his instrument.** That is the second office he priced.

### 2.1 THIS OFFICE'S it.24 REPORT WAS WRONG ABOUT WHAT LEFT THE TABLE

it.24 published *"nine occurrences and seven pointers left the table
mid-iteration."* **They did not leave. They changed notation.** The it.22
regex matched `path:N` only; the eight `:1` pointers became `path:*` and `C63`
became the range `:13-14`. `120 + 8 + 1 = 129`, the same 129 as before.

**Mechanism: an instrument that measures a population through one notation
reports a notation change as a population change.** The it.24 report then built
its lead section on the phantom deletion. Corrected here, `[RUN]`-backed above.

---

## 3 — THE THREE LANDING DEFECTS

Each is a claim the CELL makes, the line OPENED, and what it actually carries.
All three are **GREEN under every location instrument the round owns** — the
file exists and the line is in range.

### M-25a — `SPAN_HEAD`. A COUNT LANDED ON THE HEAD OF ITS SPAN

`V20_R15_THEORY_TABLE.md:95` (Q4/W1) — *"there are **nine sibling argparse
flags** at `scripts/v15_r1.py:547`"*.

```
[RUN] sed -n 547p scripts/v15_r1.py                              12:05:52Z
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, ...])
[RUN] grep -c add_argument scripts/v15_r1.py   ->  9   spanning :547-558
```

**`:547` carries one flag. The count is correct; the landing is not.** A
nine-line span is attributed to its first line. **The round already owns the
correct notation** — `C131` uses `:A-B` — and it was not used here.

### M-25b — `COMPOUND_HALF`, RECURRING. THE CITED CONSTANT IS NOWHERE IN THE CITED FILE

`V20_R15_THEORY_TABLE.md:71` — *"`floor₁ = sqrt((t*−1)/t*) = 0.7071067811865476`
at `t* = 2` (`scripts/v15_r1.py:586`; identity at `scripts/v15_r1.py:17`)"*.
Re-cited at `:203` (Q5/W3): *"`floor₁` @ `scripts/v15_r1.py:586`"*.

```
[RUN] sed -n 586p scripts/v15_r1.py                              12:05:52Z
    floor1 = math.sqrt((T_STAR - 1) / T_STAR)
[RUN] grep -n "0.70710" scripts/v15_r1.py    ->  (no output, exit 1)
[RUN] grep -n "T_STAR *=" scripts/v15_r1.py  ->  138:T_STAR = 2
```

**Three parts to the claim; one lands.** The formula is at `:586`. The value
`0.7071067811865476` **does not appear anywhere in the file** — it is computed,
never written. `t* = 2` is at `:138`, uncited. The value's only home in the repo
is the results journal, which is where this office read it back at
`0.7071067811865476` in §4.2's RED.

This is the class JUPITER struck this office's marker over, **found again, in a
different cell, against a different file.** It was not a one-off.

### M-25c — `REGISTRY_FOR_ARTEFACT`. THE POINTER NAMES THE THING, NOT THE THING

`V20_R15_THEORY_TABLE.md:304` (the F4 admission-condition table, Q2/W1) — *"the
generator exists at `ceq/kdata.py:475` and has **zero callers under
`scripts/`**"*.

```
[RUN] sed -n 475p ceq/kdata.py                                   12:05:52Z
    "bed_k": {"generator": "ceq.beds.bed_k.build_delay",
[RUN] grep -rn "def build_delay" ceq/  ->  ceq/beds/bed_k.py:236
[RUN] grep -rn "kdata" scripts/ --include=*.py
    scripts/v20_m14_cheeger.py:294,307,437,490  -- all comment/string, no import
```

**Half lands, half does not.** *Zero callers* is **TRUE** — the four `scripts/`
hits are comments and literal strings, no import. *The generator exists at
`:475`* is **FALSE**: `:475` is a registry dict entry carrying the generator's
**name as a string**; the generator is `ceq/beds/bed_k.py:236`, in a different
file, and that file is not cited anywhere in the clause.

**This one has teeth beyond the citation.** The clause is an ADMISSION CONDITION
— it tells the leap what it would take to re-enter Q2/W1. A leap that follows
`:475` to run one BED-K cell arrives at a dict key.

---

## 4 — THE TWO STANDING REDs

### 4.1 THE STRUCK MARKER — FIXED. IT NOW OBSERVES ITS OWN REMEDY

JUPITER's ruling is adopted verbatim: *a defect marker must assert against the
thing the repair will change.* `test_m22a_compound_half_constants_13_does_not_carry_beta`
asserted `"-0.032353" in line_at(constants, 13)` — it demanded the **file** move
when the defect was that the **citation** did not match the file. RED before the
repair, RED after it. It could never go green.

Rewritten to assert the **citation**, the shape this office's other three
markers already had:

```python
assert f"`{const}:13`" not in table          # the bare compound is gone
assert f"`{const}:13-14`" in table           # C131's range is present
for want in ("0.717647", "0.032353"):
    assert want in line_at(const, 13) + line_at(const, 14)
```

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it22_independent_census.py -q
                                                                  12:06:49Z
   8 passed in 0.38s
```

**The remedy is now observable and the marker is GREEN because C131 repaired it,
not because it was deleted.** `C131`'s range verified line by line:
`:13` carries `+0.717647`, `:14` carries `-0.032353`.

**AUDIT OF EVERY OTHER MARKER THIS OFFICE HAS SHIPPED — the class is empty.**

```
[RUN] grep -n "assert .*line_at(\|assert .*read_text" tests/mercury/*.py \
        | grep -v TABLE                                            12:06:37Z
   (no output)
```

Every remaining `M-22a` marker asserts against `V20_R15_THEORY_TABLE.md`, which
is the artefact a citation repair edits. **One instrument had the defect, not a
family.**

### 4.2 THE SECOND RED — **CLAIMED**, NOT PRE-EXISTING

`tests/mercury/test_v20_r15_it10_eval_independence.py::test_it8_aggregate_verdict_agrees_with_its_own_cells`
is **this office's own it.10 `RED-3`**, shipped red on purpose and still red.

```
[RUN] python -m pytest ...::test_it8_aggregate_verdict_agrees_with_its_own_cells -q
                                                                  12:06:37Z
   1 failed in 1.41s
E  AssertionError: 8/9 cells sit below floor_1=0.7071067811865476 but the agg
E  row registers crosses=False (sd=0.21245444406217273, ci_hi=0.8808087489119841)
```

**It is correctly shaped and must stay red.** It asserts against
`results/v20_r15_it8_armpl_b.jsonl` — the journal a repair would change — and its
GREEN control (`test_control_the_it8_aggregate_is_reconstructible_bitwise`)
passes, so the arithmetic it accuses is reproducible rather than alleged.

**It is the same fact the table now carries as the Q5/W3 GAP** (*"a mean over a
bimodal population is not a statement about either mode"*). It goes green when
the Q5/W3 ROUTE — component-wise CIs under `sign(lambda_hat)`, priced at **0
GPU-s** — is taken, and not before. **CLAIMED. Do not clear it.**

---

## 5 — WHAT WAS NOT REACHED, AND THE INSTRUMENT THAT MUST NOT BE BANKED

**108 of 129 occurrences were screened and not read.** The mechanical screen
(`MERCURY-R2` step 2: pull decimals and `F0–F4` tokens from a cell, ask whether
some cited pointer in that cell carries them) returned `58 unlanded of 89 wants`.

**That number is an artefact and is published here only so no office banks it.**
Two failure modes, both real:

1. **`F0–F4` tokens are rubric prose.** Every cell that discusses the grading
   scale mentions all five; the screen then demands all five land. Twelve of the
   58 are this.
2. **A cell's constants are asserted by its `CENSUS` bullet and its pointers are
   in its `DECLARATION` bullet.** They are different sentences. The screen reads
   the cell as one clause and cross-demands them.

Q3/W3 is the clean example: `0.451211`, `0.662128`, `1.113339` are flagged
unlanded, and the cell's two pointers are a test-node name
(`tests/jupiter/test_v20_r15_it7_q3.py:106`, which does carry the node name) and
a producer expression (`scripts/v15_r1.py:383`, which does carry
`lambda_hat=float(lg.mean())`) — **neither ever claimed to carry a constant.**
Ruled **LANDS**. The screen is a **candidate generator with a high
false-positive rate**, not a verdict, and this office is not issuing a rate
from it.

**A real gap the screen did surface and this office cannot close in the time:**
Q3/W3's three constants are **cited by no pointer at all**. That is not a
landing defect — nothing claims to land them — it is a **missing citation**, a
different class, and it is not counted in the `3`.

**LIMITS.** `21 of 129` is what was opened, in the seven clauses at table lines
`71, 95, 104, 163–170, 171, 203, 304`. The three defects are a **count, not a
rate**, and must not be extrapolated to 129 — the INSPECTOR's `8-in-20` was
withdrawn at it.20 for exactly that. The 19 `.lean` occurrences were **not
opened**; no `.lean` line was ruled this iteration. `J-24a` was taken as given
for the eight `path:*` occurrences; this office did not re-derive it
`[CITED: JUPITER]`. No cell was ruled using JUPITER's frozen want — every
ruling above reads the want off the table cell itself.

**PRICE OF THE REMAINDER.** 108 occurrences at the rate actually achieved here
(21 hand-rulings in one office-iteration) is **~5 further office-iterations**,
and it is splittable by clause — unlike the first one, which had to build the
extractor.
