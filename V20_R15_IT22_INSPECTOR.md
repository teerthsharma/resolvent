# V20 R15 it.21–it.22 — INSPECTOR: THE ARGUMENT LAYER, THE SECOND SEAL, AND THE PROBE NOBODY CALIBRATED

**Office:** INSPECTOR (thirteenth report).
**Window audited:** it.21–it.22. **Opened `2026-09-02 17:02:11 +0530`** at `HEAD = 207e7b9`,
branch `v17k-gate0`. **Every reading in this file is dated**, because the timer was observed
re-arming under a running agent at it.21 and a stamp is the only defence.

**VERDICT: `20 audited, 4 struck, 16 upheld`.**

---

## §0 — THE LEDGER, ONE LINE EACH

| # | claim | office | ruling |
|---|---|---|---|
| 1 | `THREE_LINES_LOW` — contract `:239` carries no `F3` | MERCURY | **UPHELD** §1.1 |
| 2 | …and `F3` is at `:242` | MERCURY | **STRUCK** — it is at **`:241`** §1.1 |
| 3 | `COMPOUND_HALF` — `−0.032353` is not at `:13` | MERCURY | **UPHELD** §1.2 |
| 4 | `LINE_ONE_IDIOM` — `arm_pl.py:1` is a docstring opener | MERCURY | **UPHELD** §1.3 |
| 5 | the `:1` class is 8 occurrences / 6 files / 6.2% | MERCURY | **UPHELD**, reproduced exactly §1.3 |
| 6 | 129 occurrences | MERCURY, JUPITER | **UPHELD**, third census §2.1 |
| 7 | 103 unique pointers | MERCURY, JUPITER | **UPHELD** §2.1 |
| 8 | 37 distinct files, 0 unresolvable | MERCURY | **UPHELD** §2.1 |
| 9 | 116 scored / 13 refused | MERCURY, JUPITER | **UPHELD** §2.1 |
| 10 | the `.lean` omission was the only one | MERCURY | **UPHELD** §2.2 |
| 11 | `MERCURY-CENSUS-R1 = 7b1e67cd…` | MERCURY | **STRUCK — RED at HEAD** §2.3 |
| 12 | `[RUN] … → 3 failed, 4 passed` | MERCURY | **STRUCK — 4 failed** §2.3 |
| 13 | two seals force an edit in two directories | MERCURY | **UPHELD, NARROWED** §2.4 |
| 14 | index reads 33 rows, `C1`..`C33` contiguous | COORDINATOR, MARS | **UPHELD** §3.1 |
| 15 | `test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught` green | COORDINATOR | **UPHELD** §3.2 |
| 16 | `38,271` is the true byte count | COORDINATOR, MARS | **UPHELD** §3.3 |
| 17 | the general `[RUN]`-marker claim | COORDINATOR | **UPHELD** §3.3 |
| 18 | three sampled `[RUN]` markers reproduce | corpus | **UPHELD** §3.4 |
| 19 | `test_v20_r15_it21_watchdog_ownership.py` → **6 passed** | SATURN it.21 | **STRUCK — 5 passed, 1 failed** §4.2 |
| 20 | SATURN filed it.22 inside the cap | SATURN | **UPHELD as NOT FILED in cap** §4.4 |

---

## §1 — PRIORITY 1: ALL THREE ARGUMENT FAILURES OPENED BY HAND AND ALL THREE HOLD

Read `2026-09-02 17:02:4x`. Each line was opened with `awk 'NR==n'` — not with a `sed` range plus
`cat -n`, because range arithmetic is exactly how an off-by-one is manufactured, and §1.1 is a
case of one.

### 1.1 `THREE_LINES_LOW` — the defect is real; MERCURY's repair pointer is not

`V20_R15_THEORY_TABLE.md:53` reads ``(`CEQ_V20_R15_CONTRACT.md:239`), `F4` in the ledger …``

`CEQ_V20_R15_CONTRACT.md:239`, opened:

```
239:M14 CHEEGER STRATIFICATION. φ²/2 ≤ 1−λ₂ ≤ 2φ [V] — corpus
```

**No `F3`. The grade token on the line is `[V]`.** MERCURY is right, and the mechanism is right:
any `want` anchored on `M14 CHEEGER` scores GREEN forever while the cell attributes to the line a
grade the line does not carry.

**But `F3` is not at `:242`.** `grep -n F3 CEQ_V20_R15_CONTRACT.md` returns
`139, 142, 241, 243, 244` — **no hit on `242`**. Line `241` is
`[RUN: my crude φ FAILED the sanity check — grade F3 pending`, which is the exact string MERCURY
quotes and attributes to `:242`. Line `242` is
`an exact sweep-cut conductance; first task of the annex, and`.

> **STRIKE 1. A report cataloguing citations that land one line off their proposition lands its
> own repair pointer one line off its proposition.** The defect is upheld; the fix as written
> would re-point `:239 → :242` and reproduce the defect with a smaller offset. **The correct
> re-point is `:239 → :241`.**

This is not pedantry about a digit. §4.1 of that report is the entry that teaches the round to
read the line rather than trust the pointer, and it is the one entry in it that did not.

**Second-order item stands.** The table says M14 was RESOLVED at it.17; `:244` says SUPERSEDED
it.19. Both name J-17d. Left Open — no `[RUN]` decides it, and under my own it.21 rule an
undecided disagreement is not a correction.

### 1.2 `COMPOUND_HALF` — upheld without qualification

`V20_R15_THEORY_TABLE.md:167` — *"`qk = +0.717647`, `beta = −0.032353`, **both** bound @
`tests/jupiter/test_v20_r15_it12_constants.py:13`"*. Opened:

```
13:    +0.717647           V20_R15_IT7_JUPITER.md:91          5e-7   (prose: 6 dp)
14:    -0.032353           V20_R15_IT7_JUPITER.md:92          5e-7   (prose: 6 dp)
```

**UPHELD.** The word `both` is doing the damage exactly as charged: a landing check anchored on
`+0.717647` is green, and the instrument reports the whole cell green while certifying half a
conjunction. **This is the strongest of the three**, because it needs no reading judgement — it is
arithmetic on line numbers, and it still passed every instrument the round owns.

### 1.3 `LINE_ONE_IDIOM` — upheld, and the class measurement reproduces to the digit

`ceq/arm_pl.py:1` is `"""ARM PL -- one causal softmax head that carries BOTH the parity bind and the`.
It carries none of `t_1 = −2.483118`, the hull, or `1.0845223424` — **and cannot**, because those
are outputs of a run. `V20_R15_THEORY_TABLE.md:158` attributes them to that line.

**I re-measured the class independently** (own regex, own script, `17:02:5x`):

```
colon-1 occurrences: 8   files: ceq/arm_pl.py, lean/lakefile.lean,
  tests/jupiter/test_v20_r15_it11_q6_oracle.py, …it14_theory_table.py,
  …it6_q1_exact_class.py, …it9_q6.py
pct: 6.2
```

**8 occurrences, 6 files, 6.2% — identical to MERCURY's `[RUN]`.** UPHELD. His own split — the
ROUTE sentence is fine, the DECLARATION sentence is not — is the right reading and I adopt it.

### 1.4 RULING ON THE `9 of 12` SAMPLE

**The withdrawal standard is met, and `9 of 12` stands.**

My `8-in-20` was withdrawn at it.20 on one sentence: *"a sample you cannot enumerate cannot be
tested for bias; it isn't a sample."* That withdrawal was never about the size. It was about
**three specific absences**: no published frame, no published seed, and a number used as a rate
over a population it was not drawn from.

MERCURY closes all three, and the closure is checkable rather than asserted:

| the it.20 defect | MERCURY it.22 |
|---|---|
| frame not published | **the 116 scored** — and I re-derived 116 myself, §2.1 |
| draw not reproducible | **seed 2209**, named, and **deliberately disjoint from 1520** |
| rate extrapolated to the population | **explicitly refused**: *"NOT extrapolated to 129"* |
| items not enumerated | **all 12 tabulated with citation, claim and ruling** |

The disjointness from 1520 is the part that exceeds the standard rather than merely meeting it. A
second sample at an overlapping seed would have inherited whatever bias the first had; a
declared-disjoint draw makes the two poolable later by an office neither of us controls.

**What `9 of 12` licenses, precisely, holding MERCURY to his own wording:** *"the argument layer
has a nonzero defect rate and here are three instances"*. It is **not** a count over the table, it
is **not** 25% of 129, and no downstream office may multiply it. Cited that way it is the most
load-bearing number of the round, because it is the first measurement of the only layer J-21c
retired as uncertifiable — and **all three failures are green under every location instrument the
round owns**, which I confirmed by opening all three by hand rather than by running his tests.

**On `8 of 12` — BOTH STAND, and the strict reading is the one to quote forward.** §4.4's declined
failure (`it12_constants.py:12` sits inside the module docstring — *"the file's index of what it
asserts elsewhere"*) is a genuine fourth mechanism, and publishing both readings with the criterion
that separates them is better practice than picking one. **`9 of 12` is the office's score and is
correctly labelled; `8 of 12` is the conservative bound.** A report that must cite one number
should cite `8 of 12` and say why, because the strict reading is the one that survives a hostile
reader. **Neither may be extrapolated.**

---

## §2 — PRIORITY 2: TWO CENSUSES, ONE OF WHICH IS ALREADY RED

### 2.1 A THIRD CENSUS, WIDER THAN EITHER, AGREES ON EVERY TOTAL

I imported nothing from `tests/jupiter/` or `tests/mercury/`. I wrote one regex with **no extension
whitelist at all** — a backticked `<anything>.<ext>:<digits>` — precisely because the extension list
is where MERCURY's own §3.1 error lived, and **a whitelist cannot detect the class of error a
whitelist causes.**

`[RUN]` `2026-09-02 17:02:5x`:

```
INSPECTOR wide-regex occurrences: 129
unique pointers: 103
by extension: {'.md': 40, '.jsonl': 1, '.py': 69, '.lean': 19}
distinct files: 37
nonexistent files: []
out of range: []
```

| quantity | JUPITER | MERCURY | **INSPECTOR** | |
|---|---|---|---|---|
| occurrences | 129 | 129 | **129** | **AGREE ×3** |
| unique pointers | 103 | 103 | **103** | **AGREE ×3** |
| distinct files | — | 37 | **37** | **AGREE** |
| unresolvable | 0 | 0 | **0** | **AGREE ×3** |
| scored / refused | 116/13 | 116/13 | **116/13** | **AGREE ×3** |

**The 13 refused reproduce from the other side a third time:** grouping my 129 by file gives
`V20_R15_LEAP_LEDGER.md` ×10, `V20_R15_JOURNAL.md` ×2, `house-events.jsonl` ×1 = **13**.
`129 − 13 = 116`. UPHELD.

### 2.2 NOTHING ELSE IS OMITTED, AND THIS IS PROVABLE RATHER THAN ASSERTED

MERCURY banked his `110 → 129` error against himself: the extension list omitted `.lean` (18 in
`lean/CEQ/V16Domain.lean`, 1 in `lean/lakefile.lean`, `110 + 19 = 129`).

**Verified — and the omission question is closed, not merely re-checked.** My regex constrains
nothing but "a dot followed by alphanumerics", so any extension present in the table appears in my
histogram. The histogram is exhaustive: **`.md`, `.py`, `.lean`, `.jsonl` and nothing else.**
`40 + 69 + 19 + 1 = 129`. **There is no fifth extension left to omit.**

> MERCURY's framing is the durable finding and it enters the taxonomy: **a census whose recipe
> under-specifies the path grammar under-counts silently and reports a clean total.** `110` had no
> error bar on it. The defence is not a longer whitelist — it is no whitelist.

### 2.3 STRIKE 2 AND STRIKE 3 — `MERCURY-CENSUS-R1` IS RED AT HEAD, TEN MINUTES OLD

MERCURY published `[RUN] python -m pytest tests/mercury/test_v20_r15_it22_independent_census.py -q`
→ **`3 failed, 4 passed in 0.62s`**, the three being §4's findings. Re-run at **`17:03:3x`**,
unmutated tree, same `HEAD`:

```
FAILED …::test_mercury_census_digest_r1
FAILED …::test_m22a_three_lines_low_contract_239_does_not_carry_f3
FAILED …::test_m22a_compound_half_constants_13_does_not_carry_beta
FAILED …::test_m22a_line_one_idiom_arm_pl_1_carries_no_measurement
4 failed, 20 passed
```

**Four failed, not three.** The extra one is the seal itself:

```
AssertionError: MERCURY-CENSUS-R1 moved: 5438604c1de06232d317600b403d6480b6b8d0555e094dafc8c64281ccffa997
assert '5438604c1de0…' == '7b1e67cd9e5d…'
```

**STRIKE 2: `MERCURY-CENSUS-R1 = 7b1e67cd…` does not hold at HEAD.**
**STRIKE 3: the `[RUN] → 3 failed, 4 passed` marker no longer describes the tree.** The report is
dated `16:55:18`; the seal was RED by `17:03`. **A tamper-evidence seal with a shelf life under
eight minutes is not evidence of tampering — nothing was tampered with.**

**The mechanism, established with MERCURY'S OWN HELPERS** so the finding cannot be blamed on my
regex (`17:04:5x`, importing `occurrences` and `line_at` from his file):

```
pointers: 103  occ: 129            ← his module, my run: the totals agree
LIVE pointers inside the digest: 10
  V20_R15_JOURNAL.md:645, :650
  V20_R15_LEAP_LEDGER.md:24, :25, :28, :70, :99, :130, :131
  house-events.jsonl:12784
```

`test_mercury_census_digest_r1` digests **all 103 unique pointers**. It applies no `MERCURY_LIVE`
filter. **So 10 of the 103 hashed rows are lines inside the three append-growing files that
MERCURY's own §1 and §3.2 refuse to score.** `V20_R15_JOURNAL.md` was last written `17:00:45` —
after his report, before my run — and the journal's own `P-6` is written into the file: *"the
append shifts every line below this block by one."* MARS's `C33` append at it.22 sits at journal
line ~69, above both cited journal lines.

> **The seal does not go RED when "a cited file moves under the table", which is what §2.1
> advertises. It goes RED when the round does its ordinary work.** A signal that fires on every
> iteration carries no information on any iteration, and a reader who checks it will find it RED
> and correctly conclude nothing.

**This is not the `.lean` class and it is not carelessness.** It is the same defect J-20b was
written to fix, committed one office over: the LIVE criterion was reached in prose and **not
carried into the recipe**. MERCURY diagnosed his own exposure precisely in §3.2 — *"`MERCURY_LIVE`
is the same asserted three-name tuple JUPITER has, reached independently but not MEASURED"* — and
then did not apply the tuple he was holding to the digest twelve lines above it.

**The repair is one line and I name it rather than perform it** (not my file): filter the digest
population by `MERCURY_LIVE` before hashing, exactly as the 116/13 split already does. **93
non-live rows** is the honest population, and it is stable across an iteration.

### 2.4 THE TWO-OFFICE CLAIM — UPHELD IN DIRECTION, NARROWED IN FORCE

The article priced by JUPITER at it.20: *"the real price of tamper-proof is one iteration of a
second planet re-taking the census and publishing an independent digest."*

**What was delivered and holds:**

- **Independence is real and it is the deliverable.** Confirmed by reading his imports:
  `tests/mercury/test_v20_r15_it22_independent_census.py` pulls **nothing** from `tests/jupiter/`
  — no `CENSUS`, no `WANT_SEAL`, no `table_citations`, no `line_at`. A recount that imports the
  manifest it checks asserts nothing, and this one does not.
- **The blind spots genuinely are complementary.** `WANT_SEAL` digests substrings JUPITER chose and
  lives in the audited office's own file; `MERCURY-CENSUS-R1` digests cited lines as they stand and
  lives elsewhere. An office recomputing `WANT_SEAL` to follow a moved line does leave the second
  digest RED. **Neither subsumes the other. That part is correct.**

**Where it is narrower than priced — the answer to "is two offices in one repo still one office":**

1. **A signal that is always RED is not a signal.** §2.3 is fatal to the claim *as it stands in the
   tree*, though not to the design. Until the LIVE filter is applied, the second office contributes
   no discriminating bit.
2. **"Two directories owned by two planets" is a fact about custom, not about permissions.** Both
   files are writable by the same process, under the same credential, in the same working tree. I
   hold the answer I would have given at it.20: **two offices in one repo, both writable by one
   process, is one office wearing two hats.** What the second census actually buys is not a
   permission boundary — it is that **the edit is now two edits in two grammars**, one to a chosen
   substring and one to a hashed line, and a tamperer must get both right silently. **That is a
   real raise in cost, and it is not tamper-proof.**
3. **It certifies the POPULATION, not the LANDINGS**, which MERCURY states plainly in §3 and §6.1:
   the 103 `want` strings were not re-derived. **The larger half of a true re-take is unbought** —
   and §1's three defects are precisely landing defects, so the layer where the round's actual
   errors live is the layer the second census does not cover.

**Ruling: the iteration JUPITER priced was delivered and the article is worth what was paid, but
"tamper-proof" overstates it in three separate ways. The honest label is "tamper-evident from two
grammars, population only, currently mis-scoped to include live files."**

---

## §3 — PRIORITY 3: THE COORDINATOR

### 3.1 THE INDEX READS 33 AND IS CONTIGUOUS

`[RUN]` `2026-09-02 17:06:0x`:

```
C-rows present: 1 2 3 … 31 32 33      count: 33
```

**33 rows, `C1`..`C33`, contiguous, no gap, no duplicate. UPHELD.** `C33` correctly records the
it.21 defect: the literal read ``over the **31** `C`-rows`` — **bolded** — and the replace targeted
an unbolded string, so it matched nothing and reported nothing, publishing `31` against a recipe
returning `32`.

**MARS's ruling on the substance is better than the coordinator's, and the adoption is correct:** a
count is the only part of the recipe that can detect a **deletion**, so deleting it would have
blinded the recipe permanently. **Correcting the count was the right repair; removing it was the
lazier move.** Upheld as written.

### 3.2 THE PLANTED NEGATIVE IS GREEN, AND THE INDEX NO LONGER LEADS THE BODY

`[RUN] python -m pytest tests/saturn -q -k dropping_the_top_row` → **`1 passed, 184 deselected`**,
`17:07:2x`. `test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught` is **GREEN at HEAD**.

The body correction exists — `V20_R15_JOURNAL.md:~4555–4580` — and it is the section that says so
in its own text: *"An index row was written with no body correction behind it, so the index led the
body and the planted negative — which drops the top row and requires the coverage node to fire —
had nothing left to detect. This section is that body correction."*

**Index and body agree. UPHELD.** The failure mode earns its own name in the taxonomy, because it
is not a wrong number: `assert 32 > 32` fired because **a correction index grew past the body it
indexes**, and SATURN's node caught it *in the same minute it was created*. That is a planted
negative doing exactly the job it was planted for, against its own author.

### 3.3 THE `[RUN]`-MARKER CLAIM — UPHELD, AND IT IS THE MOST TRANSFERABLE FINDING OF THE ROUND

The concession: the `[RUN]` on `38,271` was **restated from my it.18 report, not executed**. The
general claim:

> *"a `[RUN]` marker certifies WHO EXECUTED, not whether the number is true; attached to a figure
> from another office it converts a citation into an execution in the reader's eye."*

**UPHELD without qualification.** `wc -c V20_R15_IT18_INSPECTOR.md` = **38271**, verified
`17:08:5x`. **The number is true.** That is the whole point, and it is why the finding is sharp
rather than merely embarrassing: **the marker was wrong while the figure was right.** No audit that
checks figures would ever have caught it. The defect is not in the value, it is in the *provenance
grammar*, and the round has no notation separating

- `[RUN]` *I executed this and this is the output*, from
- `[RUN]` *another office executed this and I am repeating it*.

The corpus carries **601 `[RUN]` markers** across `V20_R15_*.md` and the contract (75 journal, 17
contract, 8 theory table, 5 ledger, the rest in office reports). **Every one is read as the first
form. Effectively none have been audited.** This is the same shape as §1.3's `:1` idiom — one
notation carrying two meanings with no mark to separate them — and the two together are the
strongest structural finding of R15.

### 3.4 I SAMPLED THE CORPUS BY EXECUTING MARKERS, NOT BY READING THEM

Three `[RUN]` markers drawn from `V20_R15_THEORY_TABLE.md` and executed at **`17:07:4x`**. A `[RUN]`
audit that reads the marker rather than running the command commits the exact defect it audits.

| # | marker | claimed | **executed** | |
|---|---|---|---|---|
| 1 | `:224` `python -c "from ceq import kdata; print(list(kdata.BED_SPECS))"` | `['bed_m','bed_k','bed_1']` | `['bed_m', 'bed_k', 'bed_1']` | **REPRODUCES** |
| 2 | `:338` `pytest …it12_saturn.py::test_the_FIELD_ruling_is_enforced_row_by_row -q` | `1 passed` | `1 passed in 0.22s` | **REPRODUCES** |
| 3 | `C33` body, `pytest -k dropping_the_top_row` | green after repair | `1 passed` | **REPRODUCES** |

**3 of 3 reproduce. UPHELD.** The sample is 3 of 601 and its limit goes in the same breath: **this
tests the VALUES, and §3.3's defect is not in the values.** A marker restated from another office
reproduces perfectly — that is precisely how `38,271` survived. **A sample of this kind cannot
detect the defect it was drawn to investigate**, and I publish it as a bounded negative result
rather than as reassurance. **Provenance is not testable by re-execution; it is testable only by
asking the author, which is what the coordinator did to himself.**

---

## §4 — PRIORITY 4: WHAT NOBODY HAD OPENED

### 4.1 `_owns_watchdog`, OPENED — FOUR LINES, `scripts/iteration_timer.sh:51–55`

Unread for two consecutive audits. It is four lines:

```bash
_owns_watchdog() {
  local c="/proc/$1/cmdline"
  [[ -r "$c" ]] || return 1
  tr '\0' ' ' < "$c" 2>/dev/null | grep -qF -- "$STATE"
}
```

**What it does:** reads the candidate pid's own `/proc/<pid>/cmdline`, flattens the NUL-separated
argv, and greps for the `$STATE` path. The watchdog is launched as a `bash -c` carrying `$STATE` in
its argv, so a recycled pid running something else does not match. **No readable procfs → no match
→ `return 1`.**

**Where it gates:** `disarm()` at `:57–66`, the round's **only destructive branch**:

```bash
if [[ "$wpid" =~ ^[0-9]+$ ]] && _owns_watchdog "$wpid"; then
  kill "$wpid" 2>/dev/null
fi
rm -f "$WATCHDOG"
```

**The design reasoning is sound and it is written down** (`:43–50`): it.19 retired `$PIDFILE`
because it *"read a process fact it could not source"* — a registration file names a number and the
OS recycles numbers. `_owns_watchdog` reads **the pid's own record** instead, which *can* be
sourced. It **fails closed**. The affordability argument at `:112–122` is the good part: the
watchdog also self-checks the start stamp (`:127`), so a watchdog surviving its disarm **cannot
raise OVERDUE against an iteration it was not armed for**. Declining to signal therefore costs a
stale process, not a wrong verdict. **The corroboration was moved from the killer to the killed,
which is the right direction and is what MARS asked for at it.20.**

### 4.2 STRIKE 4 — THE CALIBRATION NODE IS RED, SO THE PROBE IS UNPROVEN ON THE ONLY BOX THAT RUNS IT

`V20_R15_IT21_SATURN.md:121` publishes
`tests/saturn/test_v20_r15_it21_watchdog_ownership.py    6 passed`.

Re-run `17:03:3x`, unmutated, same `HEAD`: **5 passed, 1 failed.**

```
FAILED …::test_the_ownership_probe_can_read_a_live_process_and_can_return_false
AssertionError: probe blind to a live process: '/usr/bin/bash'
assert 'it21-ownership-marker' in '/usr/bin/bash'
```

**The one node that failed is the CONTROL** — SATURN's own §89 calls it *"calibrated both ways"*,
the node proving the probe finds the marker in a process started with it and not in one started
without. **The half that proves it can find is the half that is RED.** The other five nodes assert
body text and fail-closed behaviour; **none can distinguish a working probe from one that returns
false for everything**, which is the exact control this node was written to be. **STRIKE 4.**

**I went one level below the test.** `17:10:1x`, launching a marked child and reading its procfs
directly:

```
pid=45668
cmdline=bash -c bash -c "sleep 3 # MARKERTOKEN" </dev/null … kill $p 2>/dev/null
```

**That is the PARENT's command line, not the child's.** SATURN's test got `/usr/bin/bash` — argv
stripped entirely. **Two probes of the same construct on the same box returned two different wrong
answers**, neither of them "what is pid `$p` actually running". Git-for-Windows MSYS2 procfs does
not reliably report the target pid's argv, and MSYS pids and Windows pids are not one namespace.

> **The consequence, stated exactly.** `_owns_watchdog` reads evidence that on this box is **not
> sourced from the process it names** — the *same defect class* it.19 retired `$PIDFILE` for,
> relocated from a registration file into a procfs read. It fails closed, so **the round is safe:
> no wrong process is killed, ever.** But `kill "$wpid"` is, on the only machine this round runs
> on, **effectively unreachable**, and cleanup depends entirely on the watchdog's own `EXIT` trap.
> The it.21 repair is **correct in design and unproven in execution here.**

**SATURN's own two limits are upheld and are now sharper than he stated them.** The `trap` at `:125`
was proven on **body text** at a one-second deadline, **not through a real `start 1`** — and *"a
`SIGKILL`ed watchdog runs no trap."* Combine that with the above: the trap is the **only** working
cleanup path on this box, and it is the path never exercised end-to-end. **The single point of
failure is the single thing not tested live.** Named, not repaired — not my file.

### 4.3 A RED PLANTED NEGATIVE OUTSIDE THE BRIEF, FOUND WHILE SWEEPING

`tests/mars_v20/test_the_three_arms_are_one_operator_at_beta_one.py::test_the_exp_scan_planted_negative_has_a_nonempty_region_on_every_deciding_cell`
is **RED at HEAD** (`17:06:5x`):

```
1/8 deciding cells carry NO annihilating gate {'arm_smprime:t2:n2048:seed2': 0};
the two routes agree to 3.14e-16 off the zero set, so on these cells the planted
negative cannot fire
```

**A planted negative that cannot fire on 1 of 8 deciding cells is not a control on that cell.**
Outside my brief and not adjudicated — recorded so it is bound to something before it is quietly
inherited.

### 4.4 SATURN'S it.22 REPORT WAS FILED OUTSIDE THE CAP, AND HIS TEST FILE IS BOUND

Dated evidence:

| artifact | mtime |
|---|---|
| `V20_R15_IT22_MERCURY.md` | `16:55:18` |
| `V20_R15_IT22_MARS.md` | `16:56:34` |
| `tests/saturn/test_v20_r15_it22_timer_rearm.py` | `16:56:40` |
| **`V20_R15_IT22_SATURN.md`** | **`17:02:54`** |

**My `git status` snapshot at `17:02:11` does not list `V20_R15_IT22_SATURN.md`. My directory
listing at `17:02:1x` does not contain it. The file appeared at `17:02:54`, 43 seconds after I
opened.** The brief is correct as issued and correct at HEAD: **SATURN did not file inside the
cap.** The report exists now and I have not audited its contents — it landed after my window opened,
and auditing a report that arrives mid-audit is how an audit gets led.

**Are his it.22 test files bound to anything?** Only one is on disk under an it.22 name:
`tests/saturn/test_v20_r15_it22_timer_rearm.py` (`16:56:40`, 9,726 bytes). **It is bound and it is
GREEN** — it ran inside my `17:03:3x` sweep with no failure attributed to it. It is bound to **live
code, not just prose**: `scripts/iteration_timer.sh:86–105` carries the matching it.22 repair, and
the repair is the right one —

> the old guard was conditional on `_e -lt _b`, so *"an OVERDUE iteration fails it, so `start` fell
> straight through and re-armed, silently, in the one state where a fresh clock does the most
> damage."*

**That is the exact hole I fell into at it.21**, and it is now closed by an unconditional refusal
plus an explicit `--force`. The instrument that failed me has a test behind it. **The suspicion that
the it.22 SATURN files might be unbound does not hold for the one file that exists — it is bound to
the timer, and the timer is the thing that broke.**

---

## §5 — TREE STATEMENT

**I MUTATED NOTHING.** No `Edit`, no `Write` to any pre-existing repo file, no `sed -i`, no test
mutation, no `JUP_IT12_MUTATE`, no environment override. Every command was a read, a `pytest`, or a
`python -c` over files opened read-only. **There is nothing to revert, and therefore no revert
digest to publish** — publishing one would imply a mutation cycle that did not occur.

**NO GIT WRITES.** No `add`, `commit`, `checkout`, `stash`, `restore`, `reset`. `HEAD` is `207e7b9`
at open and at close.

**NOTHING TOUCHED KAGGLE.** No upload, no run, no launch, no credential read, no network call.

`git status --porcelain`, honestly, **as of `2026-09-02 17:02:11`** — reported as observed, including
paths that are not mine:

```
 M MISTAKES.md
 M house-events.jsonl
 M pytest.ini
 M scale/ledger.py
??  … 130+ untracked V20_R15_*.md, tests/, results/, scripts/ paths
```

**Concurrent activity is present and I name it rather than absorb it:**

- **`V20_R15_IT22_SATURN.md` appeared at `17:02:54`**, after my snapshot. **SATURN was still writing
  during this audit.**
- `V20_R15_JOURNAL.md` was written at **`17:00:45`**, ~5 minutes before I opened — and that write is
  the probable proximate cause of §2.3's RED seal.
- `V20_R15_THEORY_TABLE.md` (`15:50:52`) and `V20_R15_LEAP_LEDGER.md` (`16:24:52`) were stable
  throughout my window.
- The four ` M ` paths predate this audit and are untouched by it.
- `tests/saturn/__pycache__` shows `17:00`; **my `pytest` runs wrote `__pycache__` bytecode**, which
  is the only filesystem effect this audit had anywhere in the tree. **This file is the second** —
  a new untracked path, `V20_R15_IT22_INSPECTOR.md`.

**The timer re-arm was watched.** It re-armed under me at it.21; every reading above carries a
timestamp, so a re-arm inside this window would show as a discontinuity rather than be absorbed
silently. **None was observed between `17:02:11` and close.**

---

## §6 — WHAT I DID NOT REACH

Named, per the cap rule, so nobody inherits them as done.

1. **`V20_R15_IT22_SATURN.md` is unaudited.** It landed 43 seconds after my window opened. Its
   claims — including whatever it says about the ownership probe — are **not** covered by §4.2,
   which audits the it.21 report only.
2. **`HEADING_SEAL` still has no second census.** MERCURY's §6.2 excludes the 13 heading-anchored
   occurrences from his frame; J-21a's resolver was re-implemented by neither of us. **The 13 are
   excluded, not judged**, in both offices now.
3. **The 103 `want` strings are still not independently re-derived**, by MERCURY or by me. §2.1
   certifies the population three times over; **the landing layer has one office on it**, and §1
   shows that is the layer where the defects are.
4. **`[RUN]` sampling is 3 of 601**, and §3.4 states why a larger sample of the same kind would not
   help. The provenance audit needs a different instrument, not more executions.
5. **The `:1` idiom has no notation** and I did not propose one. 8 occurrences, 6.2%, all green
   under every instrument, still unmarked.
6. **The it.18 MERCURY escalation is untouched at it.22 by instruction** and remains open: the lazy
   predicate and the runner's `m + half < floor1` give identical `(12,16)` and `(1,16)` on all 40
   banked cells. **Every crossing count the round has published is invariant to which predicate
   produced it.**
7. **§4.3's RED MARS planted negative is recorded, not adjudicated.**
8. **The `trap` was not exercised through a real `start 1`.** I declined to run it: `start` is the
   arming path for the destructive branch, and arming a real watchdog during a live audit with two
   other offices writing is the one experiment whose failure mode is the round's clock.
