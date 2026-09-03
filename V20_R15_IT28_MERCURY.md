# V20 R15 · it.28 · MERCURY — NAMED ARTIFACTS IN §2 AND §3, AND §5 HAND-READ

**All readings 2026-09-02, 12:49–12:56Z, local CPU, branch `v17k-gate0`.**
**This office wrote exactly three files: this one, `tests/mercury/screen_v20_r15_it28_named_artifacts.py`,
and `tests/mercury/test_v20_r15_it28_named_artifacts.py`.** No git write. Nothing touched Kaggle.

**Content digests (sha256[:16], raw bytes, at time of reading):**

| file | digest | read at |
|---|---|---|
| `V20_R15_THEORY_TABLE.md` | `942e4208893444cd` | 12:50Z **and** 12:54Z — the subject of every count below, and **it did not move** |
| `ceq/arm_pl.py` | control, read 12:53Z | `class ArmPL` @ `:358`, `def forward` @ `:405` |
| `scripts/v15_r1.py` | control, read 12:53Z | `r["secs"]` @ `:1006` |
| `tests/mercury/screen_v20_r15_it28_named_artifacts.py` | `e8089da725cc18f9` | written this iteration |
| `tests/mercury/test_v20_r15_it28_named_artifacts.py` | `30072e547b12f3d2` | written this iteration |

`942e4208893444cd` is the digest it.27 recorded at 12:40Z after JUPITER's three replacements
landed. **The table has been still for one iteration** — the first time this round that a
MERCURY count did not have to survive a mid-iteration edit.

## RESULT — THREE SCOPES, THREE DENOMINATORS, NOT POOLED

| scope | uncited named artifacts | units touched | **denominator** |
|---|---|---|---|
| **§2 — the twelve cells** | **`29`** | **`11` of `12` cells** | **`81`** |
| **§3 — the arena ticket** | **`6`** | **`3` of `3` units** | **`7`** |
| **§5 — limits** | **`2`** | **`1` of `1` unit** | **`5`** |

**No rate is offered and the three are not added.** Three populations, three readings.
`29` here is **not** it.26's `29 of 62` on §2 floats — a different instrument on a
different population that happens to return the same integer, recorded so nobody merges them.

**A second column, because the per-cell rule overstates the cost.** The task's rule is
*no `path:line` anywhere in that cell's `DECLARATION`*. Applied literally it counts an
artifact that the **neighbouring** cell cites. Split:

| §2 | count |
|---|---|
| uncited **within its own cell** | **`29` of `81`** |
| of those, uncited **anywhere in the whole table** — the leap has nowhere to go | **`18`** |
| recoverable by reading a sibling cell | `11` |

**The `18` is the number that costs what `M-25c` cost.** The `11` is a navigation tax, not a
dead end, and is reported separately rather than folded in.

**LANDINGS CENSUS: `24 → 34 of 129`.** One clause, named in §6: **the Q6 pair, table lines
`211–228`.** `10 of 10` land. **`1` weak landing and `1` new defect of a class no previous
iteration has filed — a withdrawal that misquotes the sentence it withdraws.** See §6.

---

## 1 — THE SCREEN, WRITTEN TO A FILE, WITH THE COMMAND THAT REPRODUCES IT

it.27 struck it.26's `63` because the screen behind it *was never written to a file*. That
finding applies to this office first. **This iteration's screen is a file and its denominator
survives this report:**

```
[RUN] python tests/mercury/screen_v20_r15_it28_named_artifacts.py V20_R15_THEORY_TABLE.md
      12:52Z   ->  digest 942e4208893444cd, 43461 bytes
```

The screen **does not rule CITED/UNCITED** — it fixes the *population* only: it splits every
backticked token per unit into `path:line`-shaped and artifact-candidate, and prints the
candidates with line numbers. **The ruling below is a hand read.** Screen output, per unit:

| unit | artifact-candidates | pointer-shaped |
|---|---|---|
| §2 Q1/W1 · Q1/W3 · Q2/W1 · Q2/W3 | `20` · `20` · `22` · `23` | `9` · `7` · `10` · `4` |
| §2 Q3/W1 · Q3/W3 · Q4/W1 · Q4/W3 | `21` · `15` · `25` · `12` | `8` · `2` · `12` · `4` |
| §2 Q5/W1 · Q5/W3 · Q6/W1 · Q6/W3 | `22` · `16` · `17` · `27` | `6` · `3` · `7` · `6` |
| **§2 total** | **`240`** | `84` |
| §3 preamble · 3.1 · 3.2 | `14` · `6` · `7` = **`27`** | `1` · `4` · `0` |
| §5 (one unit) | **`27`** | `10` |

**`240` candidates screen down to a hand-read population of `81` in §2.** The gap is the
rejection rule below, and it is the whole reason the screen cannot be the instrument.

**One defect in the screen, recorded not smoothed.** Its first run died with
`UnicodeEncodeError` on `U+2212` against cp1252 stdout — the table is UTF-8 and Windows
stdout is not. It was fixed by one `sys.stdout.reconfigure` line, and **the fix is in the
published file**, so the command above reproduces the counts on this machine. An unfixed
screen would have truncated at Q1/W1 and returned `1` unit instead of `16`.

### POPULATION RULE, STATED SO IT CAN BE ATTACKED

**IN:** a backticked token naming a **file, function, class method, bed, arm, test node,
journalled field, constant name, ruling id or annex id**, that is not itself a `path:line`.

**OUT, by name** (inherited from it.27 and extended; every extension is listed so the
denominator can be recomputed):

- **numbers, censuses and rates** — `1/16`, `0 of 24`, `316.954 s`, `ρ = +0.70`
- **mathematics** — `1/d`, `Θ(n·S²·d)`, `I(X;Y)`, `√2`, `A_ij ≥ 0`
- **grade tokens** — `F3`, `F4`; **markers** — `[RUN]`, `[DERIVED]`
- **key tuples and shapes** — `(g,s,q,k)`, `[S,S]`, `[n]`
- **bare directories and extensions** — `scripts/`, `.lake`, `.olean`, `.py`
- **code fragments quoted inside `[RUN]` blocks** — `kdata`, `['bed_m', 'bed_k', 'bed_1']`
- **external library APIs** — `torch.randn`, `torch.cuda.synchronize()`
- **NEW, and it removes count in this office's favour: artifacts a ROUTE proposes but that do
  not yet exist** — `--seq-len`. A missing pointer to a thing that does not exist is not the
  defect class. `pe` and `y_ev` are **kept**: they are existing runner quantities the ROUTE
  says to journal, not new ones.
- **NEW: un-backticked names are out of population.** `BED-K`, `M13 Wasserstein`,
  *"M2 GATE-LANDSCAPE THEOREM"* and *"the chess witness"* are asserted with no pointer and
  are **not counted**, purely because the screen keys on backticks. **§5's closing sentence —
  "Nothing in this file was measured on BED-K or on the chess witness" — is the `BED_SPECS`
  shape exactly, and this instrument is blind to it because of a formatting convention.**
  That is a limit of the instrument, filed in §7 below, not a finding.

---

## 2 — §2. **`29` UNCITED ACROSS `11` OF `12` CELLS, DENOMINATOR `81`.**

| cell | uncited / population | uncited table-wide | the artifacts |
|---|---|---|---|
| Q1/W1 | **`4` / `11`** | `2` | `beta` `qk` `route` `g` — the four journal fields the whole ROUTE turns on |
| Q1/W3 | **`3` / `11`** | `3` | `Lean #21 [S]`, `normalizer`, `CEQ.V15.scan` |
| **Q2/W1** | **`0` / `6`** | `0` | **the only clean cell in §2** |
| Q2/W3 | **`4` / `5`** | `1` | **`sign_flip_gate`**, `rank_plus_lower`, `NEG_ENTRY`, `rank_real` |
| Q3/W1 | **`1` / `10`** | `1` | `bad` (in `bad == []`) |
| Q3/W3 | **`5` / `7`** | `3` | `a_hat_max`, `gate_r2`, `Lean #19 [M]`, `frac_gate_annihilated`, `arm_pl` |
| Q4/W1 | **`2` / `6`** | `1` | **`secs`**, `frac_gate_annihilated` |
| Q4/W3 | **`1` / `3`** | `0` | `frac_gate_annihilated` |
| Q5/W1 | **`2` / `6`** | `2` | **`RULING J-17a`** (file, no line), `dist_to_skyline` |
| Q5/W3 | **`4` / `6`** | `2` | **`crosses: false`**, `⟨CLAUSE_1_TAIL⟩`, `lambda_hat`, `arm_pl` |
| Q6/W1 | **`2` / `5`** | `2` | `pe`, `y_ev` |
| Q6/W3 | **`1` / `5`** | `1` | **`ArmPL.forward`** |
| **total** | **`29` / `81`** | **`18`** | **`11` of `12` cells carry at least one** |

### 2.1 THE THREE THAT REPEAT it.27's MECHANISM — THE SIBLING CITES, THE CELL DOES NOT

**(a) `ArmPL.forward`, Q6/W3 `:221`.** Q6/W1 `:214` states the cell's central fact —
*the wing returns a point prediction* — and cites `ArmSMPrime.forward` @ `ceq/arm_smprime.py:577`.
Q6/W3 states the identical fact about `ArmPL.forward` and **cites nothing at all**. Both
cells are graded **F4 domain-empty**, and F4 is the grade RULING J-14 files **NOT-PUT with an
admission condition**. **The admission condition on W3 names a method the leap cannot open**,
while the same condition on W1 is provenanced. This is `M-25c` with the roles swapped: there
the pointer led to a dict key, here there is no pointer beside a sibling that has one.
`ArmPL.forward` is at `ceq/arm_pl.py:405`, `class ArmPL` at `:358` — **measured, not assumed.**

**(b) `sign_flip_gate`, Q2/W3 `:158`.** The cell's kept constant `1.0845223424` — the one that
survived it.12's bed audit when Q2/W1's did not — is measured on **planted negative PN-2
`sign_flip_gate`, seed 0**. The pointer on that line, `ceq/arm_pl.py:*`, is *the arm the draws
ran on*, not the negative. **And the negative has no producer to point at:**

```
[RUN] grep -rn sign_flip_gate --include=*.py .                              12:53Z
      tests/jupiter/test_v20_r15_it12_constants.py:138       (a test name)
      tests/jupiter/test_v20_r15_it6_q1_exact_class.py:11,140 (a docstring)
      -> 0 non-test producers
```

**The name resolves to a test function name and two docstrings.** The cell carrying the
surviving `F1 + const` rests on a planted negative that exists only as a string in test prose.

**(c) `crosses: false`, Q5/W3 `:205` and `:206`.** The cell says *"the **registered** verdict
is `crosses: false`"* and *"Registered verdict `crosses: false` on all three arms"* — twice,
with **no record location either time.** A verdict asserted as *registered* and not addressed
is the one artifact class where the missing pointer is self-contradicting: the word
*registered* is itself a claim that a register exists.

### 2.2 `RULING J-17a` — CITED TO A FILE, WITH NO LINE

Q5/W1 `:196` and Q5/W3 `:204` both carry the grade on the authority of **RULING J-17a**,
attributed to `V20_R15_IT17_JUPITER.md` — **a filename with no line number.** §2's own preamble
at `:128-129` sets the standard the cell fails: *"DECLARATION (by name and `path:line`,
**never by annex number**)"*. A whole-file citation is one step above an annex number, not on
the other side of the line the preamble draws. Same shape at Q6/W1 `:215`, where the census
address is a `.jsonl` **plus a record discriminator** `t="census"` — that one is **ruled
CITED**, because a discriminator is the correct addressing scheme for a JSON-lines file and a
line number would be the wrong instrument. The two are distinguished here rather than pooled.

### 2.3 THE FOUR FIELDS IN Q1/W1, WHICH ARE THE WHOLE REPAIR

Q1/W1's GAP is *"`0 of 24` arena cells journal `beta`, `qk`, `route` or `g`"* and its ROUTE is
*"journal `beta`/`qk`/`route`/`g` as four scalars per cell — **0 GPU-s**. This is the whole
repair."* **Neither line carries a pointer for any of the four.** `qk` and `beta` are
recoverable — Q3/W1 `:167` bounds both @ `tests/jupiter/test_v20_r15_it12_constants.py:13-14`.
**`route` and `g` are recoverable nowhere in the table.** The cheapest repair in §2 names four
fields and leaves two of them unaddressable.

---

## 3 — §3. **`6` UNCITED ACROSS `3` OF `3` UNITS, DENOMINATOR `7`.**

| unit | population | uncited |
|---|---|---|
| §3 preamble `:229-241` | `4` — `arm_smprime`, `arm_pl`, `⟨CLAUSE_1_TAIL⟩`, `floor₁` | **`3`** — `floor₁` is **CITED weakly** via `§0.2`, a section reference, which does resolve to `scripts/v15_r1.py:586` |
| §3.1 `:242-269` | `1` — `⟨CLAUSE_1_TAIL⟩` | **`1`** |
| §3.2 `:270-281` | `2` — `arm_smprime`, `arm_pl` | **`2`** |
| **total** | **`7`** | **`6`** |

**The small denominator is the reading, not a weakness of it.** §3 is the arena ticket and it
names almost no code: `7` artifact slots against §2's `81`. It carries exactly **one**
`path:line` in its preamble (`CEQ_V20_R15_CONTRACT.md:125`) and **zero** in §3.2. **§3.2 — the
sentence-per-wing summary a leap reads to learn what can be scored today — contains no pointer
of any kind.**

### 3.1 `⟨CLAUSE_1_TAIL⟩` IS UNCITED IN ALL THREE UNITS AND EVERYWHERE ELSE IN THE TABLE

It appears at `:237`, `:256`, `:265` in §3 and again at `:209` in Q5/W3 — **four times, no
pointer once.** §3.1 states its consequence in the table's own words: one-sided, the void in
clause (2) is **LATENT**; two-sided, it is **FATAL and Phase C returns no winner at any
price.** The token that decides whether the arena has a structural hole is a token the reader
cannot open.

```
[RUN] grep -rn CLAUSE_1_TAIL --include=*.md .                               12:53Z
      12 hits across 6 reports; 0 hits under --include=*.py
      V20_R15_IT13_INSPECTOR.md:524  'Filed <CLAUSE_1_TAIL> NOT RULED at :1524-1525'
```

**It is not uncited because code moved — it is uncited because it is a sentence nobody has
written.** INSPECTOR filed it NOT RULED at it.13 and the table cites neither the filing nor
the open ruling. **The correct repair is one pointer to `V20_R15_IT13_INSPECTOR.md:524`, not a
theorem.** That is a `0 GPU-s` edit and this office does not make it: the table is JUPITER's.

### 3.2 THE WING NAMES, AND WHY THEY ARE COUNTED

`arm_smprime` and `arm_pl` are Python modules (`ceq/arm_smprime.py`, `ceq/arm_pl.py`) and the
subject of every §3 row. **They carry no pointer anywhere in §3.** They are counted because
the rule is the rule, and flagged here because a reader who has reached §3 has passed §2,
where both are cited many times. **`4` of §3's `6` are the two wing names; `2` are
`⟨CLAUSE_1_TAIL⟩`. Anyone quoting `6 of 7` without that split is quoting a formatting fact.**

---

## 4 — §5 HAND-READ. **`2` UNCITED, DENOMINATOR `5`, ONE UNIT.**

it.27 screened §5 at `42` numeric / `9` float-shaped and did not hand-read it. Hand-read:

| line | artifact | ruling |
|---|---|---|
| `:404` | `pathProd_eq_zero_iff` | **CITED** — `lean/CEQ/V16Domain.lean:129` |
| `:407` | `floor₁` | **CITED** — `scripts/v15_r1.py:586`, and `:17` named as the prose site |
| `:409` | `zero_hop_mask` | **CITED** — `ceq/arm_smprime.py:559`, plus both call sites |
| `:426` | `arm_pl` | **UNCITED** |
| `:432` | **`secs`** | **UNCITED** |

**`secs` is the one that matters, and §5 is where it does most damage.** §5's penultimate
paragraph rules *"his `secs` basis is un-synchronised CUDA host wall clock, so `206–537 GPU-s`
is an order-of-magnitude price and not a budget"* — **the sentence that downgrades every arena
price in the round to an order of magnitude, resting on a field it does not locate.** `secs` is
read at `scripts/v15_r1.py:1006` (`r["secs"]`) and printed at `:886`; **neither is cited here
or at Q4/W1 `:181`, where the same field carries the `ρ = +0.70` run-order confound.** Two of
the round's largest caveats rest on one unaddressed field name.

**§5 is otherwise the best-cited unit in the table** — `10` pointers against `5` artifact slots,
and it is the only unit that cites a *toolchain* (`leanprover/lean4:v4.7.0`, Lake `5.0.0-6fce8f7`).
The two mismatches it volunteers against itself — `scripts/v15_r1.py:17` being docstring prose,
and `zero_hop_mask` having no production caller — are **exactly the class this instrument hunts,
found and published by the table's own author before this office arrived.** That is recorded in
his favour and it is why `2 of 5` is the honest reading of §5 rather than a larger one.

---

## 5 — MARKER NODE. **3 RED / 3 GREEN, RED AGAINST THE UNMUTATED TABLE.**

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it28_named_artifacts.py -q   12:54Z
      3 failed, 3 passed in 1.34s
      RED  : m28a_q6_w3_armpl_forward_carries_a_pointer
             m28b_q2_w3_planted_negative_cites_its_generator
             m28c_sec5_secs_cites_its_producer
      GREEN: m28a_control_armpl_forward_is_at_arm_pl_405
             m28b_control_sign_flip_gate_has_no_producer_py
             m28c_control_secs_is_a_journalled_field
```

Every RED asserts `V20_R15_THEORY_TABLE.md` and **goes GREEN on a table edit alone, on no code
change.** Every GREEN asserts the unmutated repo — if a control goes RED the defect was
"repaired" by moving the code, which is not the repair. **`m28b`'s control is the unusual one:
it passes by asserting that `sign_flip_gate` has NO non-test producer.** It goes RED the day
someone writes the generator, which is the correct signal — at that point the citation becomes
possible and the RED above becomes a two-word fix.

### THREE REPLACEMENTS — JUPITER'S TO APPLY, NAMED NOT MADE

```
[RUN] grep -n "class ArmPL|def forward" ceq/arm_pl.py      -> 358, 405        12:53Z
[RUN] grep -rn sign_flip_gate --include=*.py .             -> 3 hits, all tests  12:53Z
[RUN] grep -n r["secs"] scripts/v15_r1.py                  -> 1006            12:53Z
```

1. Q6/W3 `:221` — `ArmPL.forward` @ `ceq/arm_pl.py:405`.
2. Q2/W3 `:158` — either cite PN-2's generator, or state in the cell that **`sign_flip_gate`
   has no producer in this tree**, which is the true and more useful sentence.
3. §5 `:432` and Q4/W1 `:181` — `secs` @ `scripts/v15_r1.py:1006`.

---

## 6 — LANDINGS CENSUS. **`24 → 34 of 129`. ONE CLAUSE. `10 of 10` LAND.**

**The clause opened, named as the task requires: the Q6 pair, table lines `211–228`** — the
two cells §2 above finds the `ArmPL.forward` defect in. it.25's `21` came from table lines
`71, 95, 104, 163–170, 171, 203, 304`; it.27's three were `339, 379, 382`. **None of these ten
is in either set.** `34` is a count. **No rate and no extrapolation to `129` is offered.**

| table line | citation | what is there | verdict |
|---|---|---|---|
| `:213` | `CEQ_V20_R15_CONTRACT.md:123` | *"W1 to the oracle where a state distribution exists"* | **LANDS** — the quoted clause, verbatim |
| `:214` | `ceq/arm_smprime.py:577` | `return self.readout(h).squeeze(-1)[:, seq - 1]` | **LANDS** |
| `:214` | `scale/negation_scope.py:286` | `def equilibrium_oracle(x, f, p) -> torch.Tensor:` | **LANDS** |
| `:214` | `scale/negation_scope.py:304` | `return z` | **LANDS WEAKLY — see below** |
| `:214` | `scale/foreman_consequence.py:12` | the line carrying `0.492188` | **LANDS** |
| `:222` | `scale/negation_scope.py:286` | the declaration | **LANDS** |
| `:222` | `V20_R15_IT89_INSPECTOR.md:47` | `### 1.3 STRIKE I-1 — the Q6 planted negative is not the experiment…` | **LANDS** |
| `:222` | `tests/jupiter/test_v20_r15_it9_q6.py:155` | `# KILLED at it.11. …` | **LANDS** |
| `:224` | `V20_R15_IT13_MERCURY.md:189` | the `BED_SPECS` `[RUN]` row | **LANDS** |
| `:224` | `V20_R15_LEAP_LEDGER.md:131` | ledger row `L-14` | **LANDS — and the quotation of it does not. See below.** |

### 6.1 THE WEAK LANDING — `scale/negation_scope.py:304`

The table asserts *"`equilibrium_oracle` returns shape `[n]` @ `scale/negation_scope.py:304`"*.
`:304` is `return z`. **It does not establish the shape.** The shape is fixed at `:301` —
`z = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)` — and `:302-303` only
accumulate into it. **This is the `ROUND_NOUNS` shape from it.27:** a pointer that lands on
the exerciser rather than on the line carrying the asserted fact. Ruled **LANDS**, not
FAILS — the function does return `[n]` — and **flagged, not counted as a failure**. The
correct citation is `scale/negation_scope.py:301`.

### 6.2 NEW DEFECT — THE WITHDRAWAL MISQUOTES THE SENTENCE IT WITHDRAWS

Q6/W3 `:224` reads: **"The ledger's claim at `V20_R15_LEAP_LEDGER.md:131` that it is 'the one
registered bed' is FALSE and is corrected here."**

**The ledger does not say that.** `V20_R15_LEAP_LEDGER.md:131` says: *"The one registered bed
**whose output is categorical** is the chess witness"*. The table drops the qualifier, and
the qualifier is the whole proposition. **`BED_SPECS` returning `['bed_m', 'bed_k', 'bed_1']`
refutes the truncated quote and says nothing about the qualified one** — three registered
beds existing is entirely consistent with exactly one of them having categorical output.

**The conclusion survives on other grounds and the reasoning does not.** The ledger's claim
*is* false, because the chess witness is absent from `BED_SPECS` altogether — so it is not a
registered bed at all, categorical or otherwise. **The table reaches a right answer by
attacking a clause the ledger never wrote.**

**This is a class no iteration in this round has filed: a correction whose target is a
paraphrase of its source.** It costs more than an uncited artifact, because a reader
checking the withdrawal against `:131` finds a sentence that does not match the one quoted,
and cannot tell whether the ledger was edited or the table misread it. **Filed as `M-28a`.
JUPITER's to apply; this office does not edit the table.** The repair is to quote `:131` in
full and give the actual ground — *the chess witness is not in `BED_SPECS` at all*.

**`0` of the ten citations FAIL. `1` lands weakly, `1` carries `M-28a`.**

---

## 7 — WHAT THIS OFFICE DID NOT REACH

- **The CORRECTIONS INDEX `overturns:` field was not consulted.** The task asked whether a
  grep-based census should subtract what the index declares dead. **It should be checked
  before the next census and was not checked here**, so **no count above subtracts anything**
  and all three are gross, not net. If any of the `29`/`6`/`2` names an artifact the index has
  killed, these numbers are high — **that is an untested direction of error and it is
  one-sided.**
- **Un-backticked named artifacts are unmeasured** (§1 rejection list). `BED-K` and *"the chess
  witness"* in §5's closing sentence are the `BED_SPECS` shape and this instrument cannot see
  them.
- **§0, §1 and §4 named artifacts.** §4 was swept at it.27 (`2 of 5`). **§0 and §1 have never
  been swept for named artifacts by anyone.**
- **No office has hand-read §2 for named artifacts before, so `81` has no prior to disagree
  with.** Unlike it.26's `63`, it is reproducible: the screen is a file and the command is
  published in §1. **If a later office recomputes `81` differently, the rejection list in §1 is
  where the disagreement will be, and it is written down to make that possible.**
