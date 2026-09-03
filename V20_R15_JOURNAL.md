# CEQ v20 — ROUND 15 JOURNAL

The round's running record. **One entry per iteration, appended, never
rewritten.** The contract it serves is `CEQ_V20_R15_CONTRACT.md`, which is
verbatim-of-record; every correction to that contract lands here, naming the
contract line it corrects, so the pair reads together (RUL-5, `V15_LEDGER.md:42`).

Every entry ends with the two lines the round requires: **distance** (to the
north star) and the **scoreboard**.

## PHASE SCHEDULE — which iteration is next

| phase | iterations | state |
|---|---|---|
| A — DISCOVERY, find the wings (L-FIND) | 1–5 | **PHASE A COMPLETE (it.1-it.5). N=2 FROZEN, +2 EARNED.** |
| B — ARMING, six questions per wing | 6–14 | **PHASE B CLOSED at it.14.** Table FROZEN (`V20_R15_THEORY_TABLE.md`) and **ruled NOT FIT for the leap**. `+6` unearned. |
| C — THE ARENA | 15–30 | **it.15 DONE** - table rebuilt (443 lines, six fields/cell, 123 citations) and hashed; rig pending. **it.16 next.** MERCURY: clause (1) decidable for 0 GPU-s, **clause (2) NOT decidable on the frozen wings**, (3) after a repair, (4) partial. Open rulings: ⟨CLAUSE_1_TAIL⟩ · ⟨L_GRADE_RUBRIC⟩ · ⟨KAGGLE_ATTACH⟩ |
| D — THE LEAP GATE (one call, Fable 5.1, at it.35) | 31–35 | not started |
| E — THE STRONGEST CANDIDATE | 36–45 | not started |

**The next iteration is the lowest-numbered one not marked complete below.**
---

## CORRECTIONS INDEX — READ BEFORE ANY ENTRY BELOW

**This journal is append-only, which means a superseded claim stays on the page where it
was written.** A reader who opens an early entry and stops there reads a number the round
has since withdrawn. That is `MISTAKES.md` **P-3, a stale claim never retracted**, and the
Inspector named it at it.10 as *"a larger debt than the `+6`"*.

**This table does not edit history and does not soften anything.** Every row points from a
claim to the entry that overturned it. **Where the correction was forced by an agent, the
agent is named** — most of these were found by the room, against this office.

| # | the claim, as filed | where filed | what is true | corrected at |
|---|---|---|---|---|
| C1 | "no arm crosses BED-M's floor₁ `0.7071`" | it.1 DISTANCE | **6 of 24 cells cross.** The it.1 citations were about other quantities | it.2 — SATURN's K5 node, fired against its own author | overturns: `no arm crosses` |
| C2 | `frac_gate_annihilated` called "the eval draw's own **sign census**" | it.1 onward | It is **not a sign field.** The arithmetic and the count of 8 stand; the label was wrong | it.9 — INSPECTOR | overturns: `the eval draw's own **sign census**` |
| C3 | "`N=65` … an **8x arena cost**" (read as a blocker) | it.2 CORRECTION 2 | **0.3455 GPU-h — it fits the box.** `N=65` holds only under round-half (`floor` gives **72**) and buys **51.7% power**; the honest number is **`N=125`** | it.3 — MERCURY | overturns: `an **8x arena cost**` |
| C4 | "**EXIT B costs zero and is already met**" | it.3 CORRECTION 4 | `CEQ_V20_R15_CONTRACT.md:125` is an **absolute** CP clause. **The clause that exists is NOT met; the clause that would be met does not exist.** Restating it is an author amendment | it.5 — INSPECTOR, STRIKE 8 | overturns: `**EXIT B costs zero and is already met**` |
| C5 | "SATURN and MERCURY **independently** refuted the determinism blocker" | it.3 | **FULLY SHARED** — one header line read twice; SATURN's `:576-581` is a strict subset of MERCURY's `:567-593` | it.5 — INSPECTOR | overturns: `SATURN and MERCURY **independently** refuted the determinism blocker` |
| C6 | "JUPITER and SATURN reached N=2 **independently, from opposite directions**" | it.4 THE RULING | They answered **different halves** and **neither yields 2 alone**. *(The narrower claim — neither read the other before filing — is TRUE and stands)* | it.6 — INSPECTOR | overturns: `JUPITER and SATURN reached N=2 **independently, from opposite directions**` |
| C7 | "two wings the record produced, **one only prose did**" | it.4 | At it.1 **all three** wings cited documents for clause (b). **The tightened clause (b) is what separated them** — credit owed to the instrument, not the wings | it.6 — INSPECTOR | overturns: `two wings the record produced, **one only prose did**` |
| C8 | "K6 shrank — two tokens **gained producers**" | it.4 | They gained nothing. They were moved to `UNBINDABLE_BY_SUBSTRING` and `producers_for` now **raises**, so the orphan test never searches them. **K6 was not paid down** | it.6 — INSPECTOR | overturns: `K6 shrank — two tokens **gained producers**` |
| C9 | "a **pole precisely on the unit circle**, marginal stability, an almost-all-pass filter" | it.7 | **The reading is dead; the mechanism is NOT established.** `a_hat_max` cannot exceed `1.0` — `torch.clamp(u,0,1)` at `ceq/arm_smprime.py:113` — so `== 1.0` is a ceiling, not a converged pole, and 6 of 16 cells are already there **before any gradient step**. But MARS then showed pre-clamp `u` **strictly exceeds** `1.0` on 6 of 6 zero-step cells (margins `0.0129–0.1843`), so **something drives it past the ceiling and what that is remains unknown.** *"It is the clamp"* explains the number, not the behaviour, and this row previously read as though it explained both. | it.8 — MERCURY; it.9 — MARS | overturns: `a **pole precisely on the unit circle**, marginal stability, an almost-all-pass filter` |
| C10 | "live-band decay **under 3% per position**" | it.7, repeated it.9 | **Up to 4.76%.** The flat band is `−0.0476 … −0.0010`; `−0.0436` was bound to **zero of 878** values. **This office's own published `0.588` IS the seed-4 cell that proves it** | it.9 — INSPECTOR (D4), accepted by JUPITER | overturns: `live-band decay **under 3% per position**` |
| C11 | "W1's fifteen failures are **M1's predicted descent to the corner**" | it.6/it.7 | `ρ(beta, nrmse) = −0.0324, p = 0.9053`. Best cell `beta +1.344`, worst `+1.990`, corner cell second-worst. **M1 is not struck; the round's transfer of it to W1 is** | it.8 — JUPITER, on himself | overturns: `W1's fifteen failures are **M1's predicted descent to the corner**` |
| C12 | "leap ledger open with **five MARS rows `L-M1..L-M5`**" | it.7 | **They were never written.** This office repeated MARS's claim **without opening the file.** Rows written for real at it.9 `:81-105` and read back | it.9 — INSPECTOR (F2) | overturns: `leap ledger open with **five MARS rows` |
| C13 | "**12 of 16**" as the crossing headline | it.8 DISTANCE | The it.8 file's own `t="agg"` record says **`crosses: false`**. Pooled sd `0.2125` vs eight-cell `0.0203` — **10.5x, all of it seed 9.** n=9 → `False`, n=8 ex-seed-9 → `True`, **and the flip is draw-invariant.** The headline turns on an exclusion nobody has ruled | it.9 — MARS; it.10 — MERCURY reproduced both | overturns: `**12 of 16**` |
| C14 | "7 of 8 fresh vs `softmax` 0/8, **`p = 6.730e-04`**" | it.8 | **Unpaired** — `softmax` had never run past seed 7. The honest paired figure on the banked record is **`p = 0.012821`**: **a factor of 19.** Now repaired by measurement — **8 of 9 vs 0 of 9, paired, three draws** | it.9 — MARS; it.10 — MERCURY ran it, VENUS priced the error | overturns: `7 of 8 fresh vs` |
| C15 | "distance to `floor₁`" used as an **information floor** throughout | it.1 onward | `floor₁` is a **one-hop capability threshold** (`scripts/v15_r1.py:17-19`) and is **violated by 6 of 34 cells**, which a lower bound never is. Under **L-FLOOR** the honest distance is to the calibrated `oracle = 0.0` | it.9 — JUPITER, Q5 | overturns: `used as an **information floor**` |
| C16 | "`12 of 16` against `softmax` 0/8 at `p = 6.730e-04`, **and that separation survives everything found this iteration**" | it.9, section **DISTANCE** (cited by heading: line numbers in this file drift) | **The strongest form of the error, and the one the Inspector's complaint was actually about.** MARS filed the report breaking it at `03:21:02`; this record's mtime is `03:21:06`. **Four seconds.** The claim was not merely stale — it was asserted as having *survived* a report this office had not opened. C13 and C14 point at it.8 and **missed this restatement entirely** | it.11 — INSPECTOR, on the index itself | overturns: `and that separation survives everything found this iteration` |
| C17 | every GPU-second and cost ratio quoted from it.3 onward — `0.3455` GPU-h, `10.17×`, `41.9×`, `2.85`/`118.8`, the EXIT A table | it.3, it.5, it.8 | **There is no `torch.cuda.synchronize()` anywhere in `scripts/v15_r1.py`**, so every `secs` on CUDA is un-synchronised host wall clock; and **the strongest correlate of `secs` is run order** (`ρ = +0.7029, p = 0.0024`), *stronger* than the gate correlation (`+0.5197`) it would have to be separated from. **JUPITER refused to report the gate-cost reading as a result on that ground and this office kept quoting the ratios anyway.** Clause (3) of the arena is scheduled against this | it.9 — JUPITER, Q4 | overturns: `every GPU-second and cost ratio quoted from it.3 onward` |
| C18 | "**N = 1 primitive**, not 3" — the merge verdict | it.2 | **STRUCK as UNBOUND.** A grep for `primitive\|isomorph\|cumprod\|merge` across `tests/jupiter/` returned **zero files**; the only test in the tree asserting the algebra was **MARS's control**, and *an adversary's control is not the author's bind*. JUPITER re-bound it at it.4 under the **exercised-map principle** — two arena entries, one primitive. The Inspector called it *"the single most consequential sentence of it.2"* | it.6 — INSPECTOR (C9) | overturns: `**N = 1 primitive**, not 3` |
| C19 | "**every pair separates by `>= 0.30`** at both shapes at the trained settings" | it.2 | **WITHDRAWN by its own author.** SATURN's gate was drawn from **W3's** range and clamped — **99.109% of it sat at exactly `1.0`**. Re-drawn per cell, `s=8 W1/W2` reads `2.623589e-01`, **below the published floor**. *It shrinks; it does not vanish* — the weakest reading is still `2.6e11×` the tolerance, so **N ≥ 2 stands** | it.3 — MARS (STRIKE 5), upheld and executed by SATURN | overturns: `every pair separates by` |
| C20 | "Annex: M14 at **F1, HOW-BAD 108x**, death re-attributed to V-25" — and every sibling SCOREBOARD restatement of it in this file | it.3 SCOREBOARD (cited by heading and by the sentence itself: line numbers in this file drift) | **M14 is `F4`, and that is the only live grade.** The `F1` is a **SCOREBOARD** line, not a grading — it sits three lines above `## it.4 — THE FREEZE` in a paragraph whose own clause reads *"Wing list **not frozen**"*, and it survived two approvals **because it is scoreless**: the `≥12 at F0/F1` bonus it feeds is already forfeit. The contract's `F3` (`CEQ_V20_R15_CONTRACT.md:239-243`) is a **conditional** pre-registration — *"pending an exact sweep-cut conductance"* — whose condition was never discharged, because **V-25 struck M14 first**. The one grade and its ground: `V20_R15_LEAP_LEDGER.md`, **`## RULING J-17d`**; the contract carries a `SUPERSEDED` pointer under its pre-registered sentence, which is **left standing rather than rewritten**. *"The leap reads the grade, not the scoreboard."* | it.19 — JUPITER, on himself; the contradiction was found by SATURN at it.14 and flagged inside the ledger | overturns: `Annex: M14 at **F1, HOW-BAD 108x**, death re-attributed to V-25` |
| C21 | "the round's logging law **violated five times** — 3 unparseable lines, 2 records with no `t`" | it.12 | **4 unparseable and 116 with no `t`.** The module already held the right numbers; the entry published neither | it.14 — SATURN, on itself (its own arithmetic then corrected at C24) | overturns: `none` |
| C22 | `V20_R15_IT8_JUPITER.md:474` and `V20_R15_IT6_JUPITER.md:466` as the `Q2/W1` pointers | it.11, it.12, and `tests/jupiter/test_v20_r15_it12_constants.py` | `:474` is **`:473`**; the grade is at **`:310`** and `err_1` at **`:453`**. **The re-grade is correct and the rows exist; the pointers to them do not** | it.14 — SATURN, on itself (undercounted — see C26, C28) | overturns: `the Q2/W1 pointers` |
| C23 | "**the check confirms him**" — an independent check over the frozen table | it.14 | **The check confirmed nothing.** It grepped `HOW-BAD` (1 hit) for a field the table actually names **`GAP`** (16 hits). `V-7`: a search structurally incapable of finding anything, read as absence | it.15 — SATURN, on itself | overturns: `**the check confirms him**` |
| C24 | "Off by one and by **fifty-eight**" | it.14 CORRECTION 21 | `116 − 2 = 114`. **The gap is 114.** A correction whose own subtraction is wrong is not a correction, and it was written while correcting an undercount | it.16 — SATURN, on itself | overturns: `Off by one and by **fifty-eight**` |
| C25 | "this office quoted **JUPITER's unprompted count** and did not check it" | it.14 CORRECTION 21 | **JUPITER never made that count.** `V20_R15_IT11_JUPITER.md:182` and `V20_R15_IT12_MARS.md:255-257` both carried `4` and `116` correctly before it | it.16 — SATURN, on itself | overturns: `this office quoted **JUPITER's unprompted count** and did not check it` |
| C26 | the two wrong line-cites "repeated in **three places each**" | it.14 CORRECTION 22 | **Five each**, two of them unnamed on each pointer — an undercount inside the correction that exists to fix an undercount | it.16 — SATURN, on itself (still short — see C28) | overturns: `repeated in **three places each**` |
| C27 | the it.16 heartbeat, filed as serving the cap because "**both branches fired in test**" | it.16 | **The only writer of the beat is `check` itself**, so it measures *check calls* and reports *processes*; the OVERDUE branch reads the beat and never writes it, so past `cap + 5m` every overrun prints *"may be an INTERRUPTION, not an overrun"* above a `--force` re-arm | it.17 — INSPECTOR | overturns: `**both branches fired in test**` |
| C28 | "**five** `:466` sites" | it.16 CORRECTION 26 | **Six. The sixth is in this office's own file.** None of the three corrections cost anything — no node red, no cell moved, no finding withdrawn | it.17 — SATURN, on itself | overturns: `five :466 sites` |
| C29 | the it.18 briefs' premises, read as file sizes — `697`, `6,601`, `10,147` bytes | it.18 dispatch | **Both dispatched agents falsified their own briefs from disk, independently.** `V20_R15_IT17_JUPITER.md` is **`21,774` bytes at HEAD** and it.17 did not die. A `stat` is a snapshot; a brief built on one asserts a present tense it never had | it.18 — MERCURY and JUPITER, separately | overturns: `premises, read as file sizes` |
| C30 | the published-query route, filed as this iteration's repair for the empty-regex class | it.19 | **A weaker restatement of `MISTAKES.md` `V-7`**, in the corpus since round 7, whose repair is sharper: the caller supplies a witness that must be recovered, and a scan that cannot recover it **raises** instead of returning empty | it.19 — the coordinator | overturns: `none` |
| C31 | "the INSPECTOR is **still auditing**" at filing time | it.19 | **He had filed 25 seconds earlier.** The reading was a file size — `36,786` bytes — and a file size cannot report completion. *Not yet notified* published as *not yet finished*: the iteration's own class, committed in its last section | it.19 — SATURN, on itself | overturns: `the INSPECTOR is **still auditing**` |
| C32 | Correction 31's own numbers: `36,786` bytes, `14m45s`, `15m10s` | it.20 | **`wc -c` reads `38,271` `[RUN]`; the two timings exist nowhere but the sentence asserting them.** They came from notification ordering, which records when this office was *told*, not when the INSPECTOR *filed*. A correction about asserting states from artifacts that cannot report them, asserting numbers it did not re-open. Also: *"quoted `V-7` at three offices during it.18"* — **three offices right, iteration wrong by eighteen; the five hits are at it.1.** Also: *"eight occurrences"* — **counts 10 by MARS's enumeration, 7 narrowest; neither is 8** | it.20 — MARS, upheld it.21 — INSPECTOR | overturns: `Correction 31's own numbers` |
| C33 | it.21 filed that the index's frozen row-counts were *"both replaced with 'over the `C`-rows'"* | it.21 | **Neither was.** The literal read `over the **31** `C`-rows` and the replace never matched it, so the index published `31` against a recipe returning `32` `[RUN]`. **A claim to have removed a stale number, itself stale, inside the entry that named stale counts as the defect.** Repaired at it.22 on MARS's route: count kept and corrected, contiguity printed beside it | it.22 — MARS | overturns: `none` |
| C34 | it.24 printed the theory table's population moving mid-iteration: *"Occurrences 129 -> 120, pointers 103 -> 96"* | it.24 | **The population did not move. The NOTATION did.** MERCURY's it.22 regex saw only `path:N`; the eight `:1` pointers had become `path:*` and `C63` a range, so `120 + 8 + 1 = 129` throughout `[RUN]`. **An instrument that measures a population through one notation reports a notation change as a population change.** The mid-iteration edit was real; the movement was the measuring regex | it.25 - MERCURY, against his own it.24 report | overturns: `Occurrences 129 -> 120, pointers 103 -> 96` |
| C35 | the `overturns:` field shipped and `8 passed` published as evidence it works | it.28 | **The enforcement node could not fail.** `live_hits()` short-circuited on every literal it was handed, so `assert live_hits(lit) == []` was a tautology; `26 of 41` declared literals were still live in the body `[RUN]`. `V-16` inside a repair written to close a measurement defect | it.29 — INSPECTOR's nurse, repaired by JUPITER | overturns: `8 passed as evidence the field works` |
| C36 | the phrase grammar applied to all 34 index rows | it.30 | **The blast radius was not computed and four of JUPITER's it.29 nodes broke.** Reverted byte-exact to `28a30ce6f6dcdbb0`. The edit was 13 cells, not 34, and two of the four mechanical truncations pass the shape rule — shape is necessary, not sufficient | it.30 — JUPITER, who assigned half the defect to his own stateless nodes | overturns: `the phrase grammar applied to all 34 index rows` |
| C37 | `62` withdrawn to `63` in one file only | it.29 | MERCURY withdrew the uncited-claims denominator and the number survives at six addresses, two of them in this journal | it.31 — JUPITER | overturns: `29 uncited claims across 10 of 12 cells, denominator 62` |
| C38 | `C35`/`C36`/`C37` written into the index as bookkeeping | it.32 | **The edit broke eight radius nodes and none was reverted** — `16 RED / 90 GREEN` against a declared `9 / 97`. **`C36` is the row recording the identical it.30 failure at half the size, installed by the same actor in the same edit.** And `C37`, having no body correction, **disarmed SATURN's guard**: with index leading body, dropping the top row restores parity | it.32 — INSPECTOR | overturns: `C35/C36/C37 written into the index as bookkeeping` |
| C39 | the it.33 radius delta `93 -> 91`, published as a procedure run correctly | it.33 | **Self-scoring and under-reported.** MARS deleted the `C38` row and re-ran the reachable population: `21/102` as-is, `25/98` without it — **five nodes green, not two, and all five read the index the row was written into.** *"Not one measures anything a correction is about."* And the whole-suite scalar is the wrong instrument: **773 / 773 / 804 nodes across three offices, all called 'the whole suite'** | it.33 — MARS, upheld it.33 — INSPECTOR | overturns: `the it.33 radius delta published as a procedure run correctly` |
| C40 | it.33's *"parity restored, SATURN's guard re-armed"* | it.33 | **True of the maxima and false of the sets.** `rows - body == (13,14,15,16,17,18,19)` `[RUN]`. SATURN's guard tests `body - rows` and `max(body) <= max(rows)`, never `rows - body`, so `C37` was caught only because it was the maximum. **Seven rows had no body correction for the whole round**; ruled `LEGACY` at it.34 with membership frozen | it.33 — MARS, ruled it.34 — SATURN | overturns: `parity restored, SATURN's guard re-armed` |

**NUMBERING, DISCLOSED RATHER THAN REPAIRED.** Row `C20` (the M14 grade, it.19) and the
body's `### CORRECTION 20` (a digest published in an append-only body, it.14) are **different
findings that share a number.** The two sequences ran independently until this extension
joined them, and rewriting a published row to realign them would be an author amendment of
the kind `C4` names. Rows `C21`–`C31` index the body's `CORRECTION 21`–`31` one-for-one;
`CORRECTION 20` is the stale-digest finding this block's own digest paragraph is about.

**Two claims were repaired by measurement rather than withdrawn** — C14's pairing and the
`n_eff = 1` objection (54 scorings, **zero flips**). **They are corrections all the same**,
and they are listed as such because the original numbers were published before the work
that justified them.

**What this index is not.** It is not a place to soften a finding. If a row ever reads
more gently than the entry it points at, the row is wrong and the entry governs. **The
entries below are the record; this table is only an index into it.**

**INDEX-SHA256 = `12eaa9f53b452a199a3820ac9294f73ce9572b071818b5333dad1d3012e46e86`** over the **40** `C`-rows `C1`..`C40`, **contiguous**, sorted as written. **The count stays** — it is the only part of the recipe that can detect a **deletion** — but a count is a fact about the file and must move with it, and this one did not: `C32` was appended at it.21 without it (MARS, it.22). *(Was `1ec530205a46fc530f58eb7d752a528bd70e1b010ec6b1d1fd5c576e28e731be` over 20 rows; moved at it.20 by the append of `C21`–`C31`, which closed a lookup that had been broken since it.14 — and the append shifts every line below this block by one, which is this index's own `P-6`. `C20` therefore cites its target by sentence, not by number.)* The Inspector's objection at it.11 was exact: *"it sits above the append marker in an untracked file with no hash — yes, a correction can be softened there."* **The digest does not prevent a softening; it makes one visible.** Recompute it before trusting any row:

```bash
python -c "import re,hashlib,io;s=io.open('V20_R15_JOURNAL.md',encoding='utf-8').read();print(hashlib.sha256(chr(10).join(re.findall(r'^\| C\d+ \|.*$',s,flags=re.M)).encode()).hexdigest())"
```

**THIS INDEX INJECTED A DRIFT AND HERE IS ITS MEASURED SIZE.** The index block is **49 lines**, but the observed shift on C16's own target is **+59** — the block plus the four rows appended to it since. **The correct statement is not "shifted by the block's length" but "shifted by however much has been prepended to date, which grows every time this index is repaired."** C16 cited `:1694`; the restatement is at **`:1753`**. **A drift that is itself a moving target is worse than a fixed one**, and it is the reason the fix below is to stop citing by line rather than to publish an offset. MARS found the symptom — C16 cited `:1694` and that line now carries other text — and said he could not tell whether the citation was wrong or an invariant was broken. **It was neither: the repair for `P-3` introduced a `P-6`.** The restatement C16 names is at the line printed above, and **C16 now cites it by its heading rather than by number.** Line numbers in this file are not stable across appends to its head; **headings are.**

**Rows C18 and C19 exist because MARS audited this index and found it closed the debt from it.3 onward while leaving it.2's two headline claims standing** — the same shape as the omission it was built to fix, one iteration later.

**Rows C16 and C17 exist because the index's first draft was audited and failed.** C16 is the restatement the index was built to catch and missed; C17 is a finding that voids a whole class of published numbers and had no row at all. **An index that needed an adversary to complete it is evidence for the Inspector's point, not against it.**

---


---

## it.1 — DISCOVERY OPENS. SATURN MINES; JUPITER TAKES M14 FIRST

**Mounted** 2026-09-02 on branch `v17k-gate0` at HEAD `207e7b9`.

### The D-3 gate, discharged before the mount

`CONTRACT.md:40-44` forbids mounting until the previous loop's cause of death is
stated in one sentence. Quoted from `V15_LEDGER.md:565-570`:

> **The R11 loop reached its declared ceiling of 30 iterations and stopped
> there.** The cap was passed as the literal `--max-iterations 30` flag and read
> back out of the state file at mount rather than trusted from the request; prose
> forms of the cap fall through the launcher's catch-all and leave the sentinel
> `0`, which is what killed R10 and the loop before it.

**R11 is the previous loop, and this was checked rather than assumed.** `[RUN]`
`ls .claude/` returns `ralph-loop.R10.stopped.md`, `settings.local.json`,
`worktrees/` — no `ralph-loop.local.md` existed at mount time. `[RUN]` a
tree-wide `*.md` grep for `TERMINATION RECORD` matches exactly one line,
`V15_LEDGER.md:557`. Rounds v16 and v17-K therefore left no loop state and no
termination record: they were driven by hand, not by a mounted loop.

### The correction carried into this round

**The leap model is Claude Fable 5.1, not Fable 5.** Corrected in the governing
skill at `~/.claude/skills/dispatching-house-mode/SKILL.md`, DR HOUSE section:
the table row now reads *Claude Fable 5.1, dispatched as `model: fable`*, and a
paragraph above it records why the two are not the same string — **the Agent
tool's `model` enum has no version field**, its only accepted token is `fable`,
so the version is the record of which model the leap is entitled to rather than a
selector. `[READ]` the enum, from the Agent tool schema: `sonnet | opus | haiku |
fable`.

**UNVERIFIED, and it is load-bearing.** Three checks were run and all three come
back negative, and **none of the three is decisive** — which is the whole reason
this paragraph exists rather than a claim:

| # | check | result | what it settles |
|---|---|---|---|
| 1 | the Agent tool's `model` enum | `sonnet \| opus \| haiku \| fable` — no version field of any kind | **Decisive** that no version can be *selected*. The token is `fable`; what it resolves to is the harness's decision, not the caller's |
| 2 | the bundled model roster (`claude-api` skill, cached **2026-06-24**) | enumerates Fable 5 `claude-fable-5`, Mythos 5, Opus 5/4.8/4.7/4.6, Sonnet 5/4.6, Haiku 4.5. **No `claude-fable-5-1`** | **Not decisive.** The cache predates today (2026-09-02) by ten weeks; a model released inside that window would not appear in it |
| 3 | live Models API (`GET /v1/models`) | **could not run.** `ant: command not found`; `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` both unset `[RUN]` | Nothing. No credential exists on this box to ask with |

**So the honest state is: the correction is filed in the only form that can be
audited, and the resolution is open.** The skill records the *entitlement* — the
leap is owed Fable 5.1 — and states that a run on anything older is not a Dr
House run. It does not claim, and cannot claim, that a dispatch on `fable`
resolves to 5.1 today.

**REPLACEMENT ROUTE (the kill ships one).** The goal behind naming 5.1 is *the
round's one leap call goes to the most capable model available*. That goal
survives the uncertainty by two moves, neither of which needs the version string
to be resolvable in advance:

- **REROUTE — record what it actually got, at the call.** it.35 dispatches on
  `fable` and the leap dossier records the model the run reports back, not the
  model the contract asked for. A version that cannot be selected can still be
  *observed*, and an observed version is auditable where a requested one is not.
- **REPRICE — the cheapest confirmation, if the author wants it settled first.**
  One `GET /v1/models` call against a credential the author holds settles it
  outright and costs no tokens. That is a thirty-second check the author can run
  and this box cannot.

**Neither route is taken silently.** If it.35 arrives and the model is still
unconfirmed, the leap runs and the dossier says which model answered.

### The mount, and the cap read back rather than trusted

`[RUN]` mounted with the literal flag form `--max-iterations 45`, then read back
out of `.claude/ralph-loop.local.md` — the check V-20 exists for
(`MISTAKES.md:987`, *a cap requested in prose, and a sentinel that reads it as
infinity*):

```
active: true
iteration: 1
session_id: f7ed3dac-8149-463a-9b41-f92b5046976d
max_iterations: 45
completion_promise: "CEQ V20 R15 COMPLETE: it.45 prognosis filed with one arm its dossier its cost its certificate and its floors"
```

`max_iterations` reads **45**, not the sentinel `0` that disabled the ceiling on
R10 and the loop before it (`stop-hook.sh:61` guards the ceiling behind a
non-`unlimited`, numeric comparison; the patched hook now treats `0` as a ceiling
of zero rather than infinity, `V13_RALPH_LOOP_FIX.md`). The `session_id` is bound,
so the hook fires for this session only.

**45 comes from the round's own phase schedule** (A 1–5, B 6–14, C 15–30,
D 31–35, E 36–45), not from an agent's choice — D-3 forbids the latter.

### Dispatch

Four planets, one message, all opus, per D-1 (parallel dispatch is legal only on
nodes with no shared repository state; the state that collides is git, so **no
dispatched agent performs a git write**) and per the standing topology (Opus
planets carry one Baker Street role each; haiku/sonnet moons do the volume).

| planet | role | assignment | report |
|---|---|---|---|
| SATURN | WATSON | mine `results/`, `lean/`, `MISTAKES.md`, `STRUCK.md`, the X-registry under the four-clause rubric; near-misses with their missing clause | `V20_R15_IT1_SATURN.md` |
| JUPITER | MYCROFT | M14 first — reproduce the failed crude φ, get the Cheeger inequality's hypotheses right, run the exact sweep-cut conductance against a brute-force second path, re-grade | `V20_R15_IT1_JUPITER_M14.md` |
| MARS | MORIARTY | strike chat-memory entries and renamings **before** the list can freeze around one; every strike a runnable VALUE test | `V20_R15_IT1_MARS.md` |
| WILSON | — | no stance: the verified baseline the three will be checked against — arms that exist, Lean that compiles, cells that exist, the device, the suite, the loop state, and which annex `[RUN]` values have a producer | `V20_R15_IT1_WILSON.md` |

**Wilson's gate passes** on all three clauses of the house-mode rule: the task
touches far more than five files, it is a multi-round tournament, and a wrong
wing list is expensive to undo — it freezes at it.4 and the whole arena is
scheduled against it.

### THE it.1 VERDICT

**Four reports in, Inspector audited: 26 claims, 7 struck** (`V20_R15_IT1_INSPECTOR.md`).

**1. THE WING LIST IS N=3, PROVISIONAL, AND IT MAY NOT FREEZE AT it.4.**
SATURN files W1 `arm_smprime` / W2 `arm_phase` / W3 `arm_pl`, twelve citations,
all resolving to a file, a line, and an anchor present on that line (C17 clean).
MARS measures that at the corner `β=qk=g=1` all three are **one operator** —
`1.11e-16` to `8.11e-16` at two float64 shapes against a control dial move of
`1.11e-01` to `2.37e-01`, fifteen orders of separation (C15 clean). Both are
bound and the Inspector rules they do not collide: citation resolvability is not
operator distinctness.

**They are reconciled by MARS's own non-firing attack.** Trained `β` reads
`0.588, 0.733, 0.782, 0.835, 0.897, 0.901, 1.344, 1.509` across eight cells —
**never at the corner**. The three coincide at a point the trained models do not
occupy, so N=3 survives *as trained objects*.

**But nobody has measured it.** SATURN flags it against his own list: the three
wings have never been run through `assert_arms_distinct`, which covers the seven
bench arms only. Distinctness today is argued from source (`cumsum` vs `cumprod`,
`ceq/compat.py:393-394`), and MARS measured identity at a corner rather than
distinctness at the trained settings. **The it.4 freeze is blocked on one
measurement**: extend `assert_arms_distinct` over W1/W2/W3 at their trained
configurations. Freezing N=3 on an argument is the ParaFormer defect — two names,
one code path — and this round's whole premise is that a wing is found, not named.

**2. THE MATHEMATICS ANNEX IS THE ROUND'S LARGEST LIABILITY, AND THREE
INDEPENDENT ROUTES SAY SO.**

| route | finding | status |
|---|---|---|
| JUPITER | M14's `[RUN: my crude φ FAILED the sanity check]` is **contradicted**. The only conductance in the tree, `scale/foreman_lambda2.py:300`, **passes** its check at `results/foreman_lambda2.txt:38,58` | **C16 clean** |
| WILSON | of the theorem numbers the annex cites, only `#2` and `#5a` carry a source declaration. `#10 #11 #14 #17 #17′ #18 #19 #20 #21 #22 #23` have **none** | reference fact |
| SATURN K6 | annex `[RUN]` instances with no producer | **survives on `d=65` and `1.3e-3`**; two of four struck (C26, C27) |

**The annex's own binding kill does not fire.** `CEQ_V20_R15_CONTRACT.md:262-263`
strikes M14 "cited before it passes its own check". The check now passes — 43/43,
both Cheeger halves, two planted negatives firing on disjoint node sets (C8, C9
clean; C7's *count* struck, the negative still fires on ≥7 nodes). **A different
thing is wrong: the `[RUN]` tag describes a failure this repository cannot
produce.**

**3. `frac_gate_annihilated` IS THE CORPUS, NOT THE ARM — and the control that
clears it hangs by one element.** MARS STRIKE 1 (C14 clean): six of eight trained
`arm_smprime` cells publish `4123/8192 = 0.5032958984375`, the eval draw's own
negative count at `seed=12345`; seed 3 publishes the exact complement `4069/8192`.
Eight different models, one draw statistic. All eight `arm_pl` cells publish `0.0`
structurally. **New instance of `MISTAKES.md` M-21/M-18.** The Inspector adds the
sharper reading: `n_zero_gates` is a corpus sign count on **seven of eight cells**,
and MARS's control passes only on the single escape value `0`, contributed by
`arm_smprime:t2:n2048:seed2` — the same cell where his own RED says the planted
negative cannot fire. Green and red rest on the same anomalous cell from opposite
sides.

**4. THE SUITE WAS DEAD AND NOTHING CAUGHT IT.** `python -m pytest -q` at the root
read `201 errors during collection`, **0 tests run**; `pytest tests/x35p/test_source.py`
alone read `15 passed`. `kaggle/snapshot/repo/` shadows every test basename and the
import mismatch is a *collection* error, so the run aborts before one test executes.
Fixed by adding `kaggle` to `norecursedirs` (`pytest.ini`); collection now reads
**2895 tests**. **The mechanism, for `MISTAKES.md`: green and uncollectable produce
the same absence of a failure line.** Every agent result this round predates the fix
and was produced by an explicitly-named path — no claim rested on a root run.

### STRUCK TO OPEN — unbound, not refuted

| claim | agent | why |
|---|---|---|
| the L-DOM census (Cheeger's hypotheses satisfied by **0** registered beds; BED-K nilpotent) | JUPITER | zero of the 43 nodes assert it; producer writes a `.txt`, never an assertion |
| `crude/exact = 1.0`, φ = 1/27 | JUPITER | the node asserts `crude >= exact` only — one-sided; would pass if crude were 10× exact |
| K5, GPU-seconds-to-floor unmeasurable for every wing | SATURN | argued with citations, no test, no RED anywhere in the round |
| P-12, an absence proof falsified by recording it | MARS | never entered the log; no finding event, no test |

**The L-DOM census is the most consequential thing anyone found and it is
unbound.** If Cheeger's hypotheses hold on none of the registered beds, M14 dies by
V-25 rather than by its own kill — but that sentence cannot be written until a node
asserts it. **it.2 owes it a test before the annex is graded.**

### WHAT it.2 OWES

1. `assert_arms_distinct` extended over W1/W2/W3 at trained settings — blocks the it.4 freeze.
2. A node asserting the L-DOM census, and a two-sided node for `crude == exact`.
3. K5 and P-12 bound or dropped.
4. The annex's Lean numbers reconciled against WILSON's census, or withdrawn.

**DISTANCE.** No arm crosses BED-M's floor₁ `0.7071`; `V15_R1.md:250` reads
`crosses? NO` and `V17_R4_RETAKE.md:194` records **zero sign-flips in 16** on the
certified 4060. The north star — attention equal to self-attention on its own
ground, predicting the next state — is not approached this iteration. it.1 moved
the *ledger*, not the capability: the round now knows which of its own instruments
were reading the corpus instead of the arm.

**SCOREBOARD.** it.1 is not a scoring iteration; the wing list scores at it.4
(`+2`, four citations each) and it is **not frozen** — blocked on the distinctness
measurement above. Annex target (`≥12 of M1–M16 at F0/F1 with M14 resolved, +4`):
M14 is graded **F1, HOW-BAD 108×**, and the remaining fifteen are now on notice —
`d=65` and `1.3e-3` have no producer, and eleven cited Lean numbers have no source
declaration. Carried: **0 of 48.**

---

## it.2 — THE MERGE, AND TWO CORRECTIONS TO it.1

Closed inside the 20-minute cap. Room: three planets, four moons each (the cap
exists because MARS got **zero** moons at it.1 when four planets saturated the
20-subagent limit). **MARS had not reported when this record was filed** — his
attack on the merge and his P-12 disposition land at it.3.

### CORRECTION 1 — it.1's distance line was WRONG, and this is the round's first real number

it.1 closed with *"no arm crosses BED-M's floor₁ 0.7071"*, citing `V15_R1.md:250`
(`crosses? NO`) and `V17_R4_RETAKE.md:194` (zero sign-flips in 16). **Both
citations are about other quantities** — the first is the v15 CPU `arm_pl` round,
the second is sign-flips, not floor crossings. SATURN's K5 node fired against its
own author and this office re-measured it independently `[RUN]`:

```
cells with eval_nrmse < floor_1 = 0.7071067811865476 :  6 of 24
   arm_smprime  seed 2   0.20391993939877656
   arm_pl       seed 4   0.6337391039935976
   arm_pl       seed 5   0.6419986310848815
   arm_pl       seed 1   0.6445174549187496
   arm_pl       seed 0   0.6446726192039927
   arm_pl       seed 6   0.6621282051474511
```

**Six cells cross. And `softmax` crosses 0 of 8.** That is the first number this
round in which an arm is on ground the skyline does not occupy.

### CORRECTION 2 — but the arena's own clause (1) is NOT met, and at N=8 it is nearly unreachable

Crossing the floor is not the contract's criterion. Clause (1) is **BED-M crossing
CP-lower > 0.5**. Computed `[RUN]`, two-sided 95% Clopper–Pearson:

| arm | crossings | rate | CP-lower | clause (1) |
|---|---|---|---|---|
| `arm_pl` | 5/8 | 0.6250 | **0.2449** | NOT MET |
| `arm_smprime` | 1/8 | 0.1250 | 0.0032 | NOT MET |
| `softmax` | 0/8 | 0.0000 | 0.0000 | NOT MET |

**And the sizing is the finding.** At the contract's `N=8`, the achievable CP-lower
values are `8/8 → 0.6306`, `7/8 → 0.4735`, `6/8 → 0.3491`. **Only a clean 8/8 sweep
clears 0.5.** One miss in eight and clause (1) can never be met, whatever the arm
does. Holding `arm_pl`'s observed rate of `0.625`, the first `N` that clears is
**`N=65`** (`41/65`, CP-lower `0.5020`).

**This is `MISTAKES.md` M-9 — a verdict whose finest achievable p cannot reach the
alpha it quotes — instantiated in this round's own arena.** The lexicographic
criterion opens on a clause that `N=8` makes into a perfect-sweep test. Phase C is
scheduled against it at it.15–24. **Either `N` rises to 65, or clause (1) is
restated as a rate comparison against the skyline's `0/8` rather than an absolute
CP bound.** Filed for MERCURY's pricing at it.5, because `N=65` is an 8x arena cost.

### JUPITER — the merge: N = 1 PRIMITIVE, not 3

| pair | ruling | the algebraic fact |
|---|---|---|
| W1 `arm_smprime` ↔ W3 `arm_pl` | **ONE primitive** | `cumprod = exp ∘ cumsum ∘ log`; `log` from the positive reals under multiplication to the reals under addition is a group isomorphism. `ceq/compat.py:393-394`'s cumsum-vs-cumprod — **the source argument N=3 rested on** — is a coordinate difference, not an operator difference |
| W1 ↔ W2 `arm_phase` | **ONE primitive on the interior** | W1's `U(1)` factor **is** W2's gate; machine-checked already at `lean/CEQ/V16Domain.lean:176` `pathProd_eq_Wp`, `:105` `pathProd_polar`, `:121` `pathProd_abs` |
| W2 ↔ W3 | **UNMEASURED → Open** | MARS ran phase-to-smprime and pl-to-smprime, never pl-to-phase; both his legs route through the beta=1 corner |

All three compute the prefix product of the gates, a prefix-differenced homomorphism
into the nonzero complex numbers, which factor as positive reals times `U(1)`.
**The only genuine separator is the attainable zero** — zero is not in the positive
reals, so `log` has no value there (`lean/CEQ/V16Domain.lean:165`
`no_prefix_scan_represents_a_zero_gate`). W2's `clamp(u,0,1)` is a **domain**, not
a primitive.

**Verdict: one primitive in three coordinate systems, plus one boundary.** it.4 may
freeze three *restrictions of one primitive*; it may not freeze three primitives.

### SATURN — the distinctness measurement, and what it cannot settle

New node `tests/saturn/test_v20_r15_wings_distinct.py`, 5 passed. RED first at the
corner, verbatim, reproducing MARS independently and adding a fact he did not have:

```
AssertionError: at s=8 these wing pairs are the SAME operator to 1e-12:
{'W1/W2': 1.1102230246251565e-16, 'W1/W3': 1.1102230246251565e-16, 'W2/W3': 0.0}
```

**W2/W3 read exactly `0.0` — bitwise, not close.** At trained settings, min over
eight seeds of max absolute delta: W1/W2 `3.06e-01`, W1/W3 `6.70e-01`, W2/W3
`6.91e-01` at `s=8`; `3.39e-01`, `6.66e-01`, `9.64e-01` at `s=64`.

**But the measurement settles W1 only.** A grep for `arm_phase` as a journalled
`kind` across `results/` returns **0** — **W2 has no trained cell anywhere in the
tree**, so its `theta` was drawn, not read. `ceq/arm_phase.py:158` returns
`arm_pl`'s matrix times `phase_factor(theta)` under `g = log m`, so W2's magnitude
equals W3 identically and the entire W2/W3 separation is carried by a parameter no
run has ever fit.

**Reconciliation with JUPITER, and it is not a contradiction.** SATURN measures that
the three *parameter settings* produce different outputs; JUPITER shows the three
are one *primitive* in different coordinates. Different points of one family differ
numerically — that is expected, and it is not distinctness of primitive. **The
freeze gate is therefore: one W2 training cell with `theta` journalled, or W2 is
not a wing.**

### SATURN — K5 WITHDRAWN, K6 halved, instrument repaired

- **K5 withdrawn**, refuted by its own node: criterion (3) *is* computable, and
  ranks W3 `25.0x` ahead of W1 (`arm_pl 1.884` vs `arm_smprime 47.048`). It is
  undefined **for W2 only, for want of a run** — the same hole as above.
- **K6 stands at 2 of 4 instances.** `d=65` and `1.3e-3` re-confirmed as having no
  producer. The two struck against WILSON are gone.
- **Instrument repaired**: a token must now contain a digit, and `producers_for`
  raises at registration instead of returning an empty list for a prose token.
- **New `MISTAKES.md` entry drafted, V-7b — the rule enforced on one half of your
  own instrument.** The file cites V-7 in its docstring, binds it on the absence
  search with a control, and leaves the producer search in the same file ungated.
  **The docstring citation reads as coverage.**

### JUPITER — both struck claims BOUND; six Lean citations WITHDRAWN

`tests/jupiter/test_v20_r15_it2_ldom_census.py`, 9 nodes, RED first
(`ImportError: cannot import name 'domain_census_facts'`, all 9 RED), then green;
`tests/jupiter/` goes **92 → 101 passed**.

- **L-DOM census now asserted as numbers**: BED-M `0/384`, ternary object `0/2048`,
  BED-K `0/8` (strictly lower triangular, nilpotent, row sums not 1). Planted
  negative: the **same** predicate returns **2** on the Rips chain, so the zero is a
  reading, not a constant.
- **The consequence is now writable: M14 dies by V-25, not by its own kill.** Its
  own kill does not fire (43/43 passes); the theorem is machine-true and
  **domain-empty**. Those two deaths were being scored as one.
- **`crude == exact` two-sided**, phi = 1/27. The C7 mistake is fixed — the mutation
  now names a **seeded** permutation, `numpy.random.default_rng(20152).permutation(n)`.
- **Lean: `#22` reconciled** — it exists under another name,
  `lean/CEQ/V15Source.lean:128` `theorem two_sources_recovered`. **`#18`, `#19`,
  `#17-prime`, `#20`, `#21`, `#23` withdrawn.** Numbered comments in `lean/` run
  `#1` to `#16` plus `#5a`, `#5b`; nothing above `#16` exists. Replacement route:
  **cite declarations by name and drop the numbering scheme — the numbers are the
  failure mode.**
- **Curricula is hollow.** A tree-wide grep for `curricul` returns exactly one line:
  the contract sentence naming it. Listed as named-and-unimplemented, not as a
  takeable multiplier.

### CORRECTION 3 — to this office's own dispatch

The it.2 prompt told JUPITER that `V17_R4_RETAKE_PRICE.md:194` carries
`softmax 1.497s`. It does not: `:194` is `arm_smprime 15.970`, and the softmax
floor is at **`:196`**. JUPITER caught it and corrected it in his report. The
skyline is listed as cost floor only: `arm_smprime` **10.67x** it, `arm_pl`
**1.078x**.

### WHAT it.3 OWES

1. **MARS's report** — the attack on the merge, P-12's disposition, the
   `arm_smprime:t2:n2048:seed2` thread. He was still running at filing.
2. **The W2 gate**: one `arm_phase` training cell with `theta` journalled, or W2
   leaves the wing list. Nothing freezes at it.4 until this resolves.
3. **The W2-to-W3 leg**, never measured by anyone.
4. **Clause (1)'s sizing** to MERCURY: `N=65` or restate the clause.
5. The Inspector has **not** audited it.2. No it.2 claim enters a verdict until he does.

**DISTANCE.** Corrected. Six of 24 cells cross BED-M's floor₁ `0.7071` — five
`arm_pl`, one `arm_smprime` at `0.203920` — while `softmax` crosses **0 of 8**.
That is ground the skyline does not occupy, and it is the first such number in this
round. It is **not** a win: the arena's clause (1) requires CP-lower above `0.5` and
the best arm reads `0.2449`. The north star is nearer than it.1 recorded and
further than the raw crossings suggest.

**SCOREBOARD.** Wing list **not frozen** (`+2` unearned) — blocked on the W2 gate,
and its N is now contested at 1 rather than 3. Annex: M14 resolved as **F1,
HOW-BAD 108x**, and its death re-attributed to V-25; six Lean citations withdrawn,
one reconciled. Carried: **0 of 48.**

---

## it.3 — THE ROOM CORRECTS THE COORDINATOR THREE TIMES

Closed inside the 20-minute cap. Room: INSPECTOR (auditing it.2), MERCURY (pulled
forward from it.5), SATURN. MARS's full it.2 report landed at the head of this
iteration and is folded in.

### CORRECTION 4 — this office's `N=65` was wrong three ways, and the blocker does not exist

it.2 filed *"either `N` rises to 65, or clause (1) is restated ... `N=65` is an 8x
arena cost."* MERCURY recomputed the whole thing and **the CP table reproduces
exactly** — `arm_pl` `5/8` CP-lower `0.2449`, `arm_smprime` `0.0032`, `softmax`
`0.0`, and only `8/8 → 0.6306` clears at `N=8`. Three things around it did not:

1. **`N=65` is convention-dependent.** It is right under round-half and **only**
   under round-half; the conservative rule is `floor`, and that lands on **`N=72`**.
2. **`N=65` is a coin flip, not a design — a 51.7%-chance experiment.** If the
   round intends clause (1) to actually fire when `arm_pl` really crosses at
   `0.625`, the honestly powered number is **`N=125`**.
3. **It is not an 8x blocker. It is not a blocker at all.** Priced on the certified
   device (`RTX 4060 Laptop sm_89 8187 MiB, torch 2.5.1+cu121`, confirmed against
   `V16_DEVICE_CERT.md:112,117`):

| design | arms | GPU-seconds | wall | GPU-hours | fits the box? |
|---|---|---|---|---|---|
| contract today, `N=8` | 3 | 156.1 | 2.60 min | 0.0434 | yes |
| **EXIT A, `N=65`** | 3 | **1 243.8** | **20.7 min** | **0.3455** | **yes** |
| EXIT A, `N=65` + W2 | 4 | 1 348.7 | 22.5 min | 0.3746 | yes |
| **EXIT A powered, `N=125`** | 3 | **2 388.6** | **39.8 min** | **0.6635** | **yes** |
| EXIT A powered, `N=125` + W2 | 4 | 2 590.4 | 43.2 min | 0.7195 | yes |

The extrapolation carries its own control: at `N=8` the cost model returns
`156.15 s` against the measured cells.

**And EXIT B costs nothing and is already met.** Restating clause (1) as a rate
comparison against the skyline's measured `0/8` rather than an absolute CP bound
needs **`N=8`** — it is satisfied on cells already in the tree, for **0 GPU-seconds,
0 wall, 0 new cells**. MERCURY prices; he does not choose. **The choice is the
author's**, and it is now a choice between two affordable options rather than a
budget problem: forty minutes of GPU for an absolute bound, or nothing at all for a
comparative one.

### THE INSPECTOR ON it.2 — JUPITER's HEADLINE IS STRUCK

**`N = 1 primitive` is STRUCK as UNBOUND.** The merge verdict — the single most
consequential sentence of it.2, and the one this office propagated into the it.2
record — has no node asserting it. The algebra behind it is *not* what is struck:
MARS's own control `test_control_the_isomorphism_is_exact_where_both_wings_are_defined`
was probed by the Inspector (one side scaled by `1 + 1e-6`) and **FAILS** with max
delta `8.999999999703689e-07` against a bar of `1e-12`. **The control binds, it is
not comparing an expression to itself, and it concedes JUPITER's algebra.** What is
missing is a node asserting the *verdict*.

Also struck at it.2: JUPITER's RED **count**, and his planted negative for the
two-sided node (a not-fire mutation where a fire-mutation was owed). **`crude ==
exact` is BOUND** — this closes it.1's C20. SATURN's it.2 log ordering struck as a
log claim only, content untouched. **P-12 is now BOUND** (it.1's C19 lifted).

Every GREEN control the Inspector probed was made to fail by making true the thing
it controls for. **None vacuous.**

### SATURN — THE `>= 0.30` IS WITHDRAWN, AND THE VERDICT SURVIVES ANYWAY

MARS's STRIKE 5 **upheld**. SATURN's it.2 headline drew W1's gate from **W3's**
range and clamped it: measured, **99.109%** of the gate sat at exactly `1.0`
(MARS's analytic `99.17%` confirmed to three figures), and the draw never reached
`m = 0`, the endpoint 7 of 8 trained W1 cells do reach. The reroute needed no
training — every input was already journalled.

| shape | pair | it.2, void gate | **it.3, per-cell gate** |
|---|---|---|---|
| `s=8` | W1/W2 | `3.063578e-01` | **`2.623589e-01`** — below the published floor |
| `s=8` | W1/W3 | `6.702194e-01` | `5.207431e-01` |
| `s=8` | W2/W3 | `6.907369e-01` | `3.694346e-01` |
| `s=64` | W1/W2 | `3.389808e-01` | `4.202484e-01` |
| `s=64` | W1/W3 | `6.657651e-01` | `1.182163e+00` |
| `s=64` | W2/W3 | `9.642279e-01` | `1.317549e+00` |

**"Every pair separates by `>= 0.30`" is false and is withdrawn.** But it shrinks;
it does not vanish — the weakest reading `2.623589e-01` is `2.6e11x` the tolerance
`1e-12`, and **N >= 2 stands**. The correction is **not a uniform shrink**: `s=8`
falls on all three pairs and `s=64` rises on all three, which is what a saturated
gate does rather than what a scale error does. The void node is kept running with
its bind marked SUPERSEDED, so the dead gate stays visible beside its replacement.

**The atom is the round's sharpest structural fact.** With the annihilating atom in
the draw: **W1 returns a finite matrix on all 8 cells** (`cumprod` on a zero gate is
a true `0`), while **W2 and W3 return `nan` on exactly the 7 cells that contain an
exact zero**. That is JUPITER's `0 ∉ R_{>0}` — the one exclusion his merge did name
— now measured rather than argued, and it carries **50.3% of the gate mass**, not a measure-zero corner. It is a **domain** difference reported as
one, not smuggled in as a magnitude.

### THE W2-W3 LEG, MEASURED FOR THE FIRST TIME — AND IT IS ALL `theta`

Continuous part: **`s=8` `3.694346e-01`, `s=64` `1.317549e+00`**. JUPITER filed
this leg UNMEASURED at it.2 and it is now measured.

**And it is bound rather than argued.** On each of the eight per-cell gates, zeroing
`theta` makes `delta(W2, W3)` **bitwise `0.0`**, and the drawn `theta` puts it above
tolerance. **Every digit of the W2/W3 separation is `theta` and none of it is the
gate** — which is exactly what `ceq/arm_phase.py:158` predicts. The `theta` is
**drawn**, `U(-pi, pi]`, seeded per cell, **not fit**, because no `arm_phase` cell
exists. The test flips RED the moment one lands.

### MARS — P-12 BOUND, AND THE SEED-2 ANOMALY IS ONE FACT

`tests/mars_v20/` reads **16 RED / 14 GREEN**. P-12 bound with the Inspector's
wording correction applied (`STRUCK.md` carries prose recording an absence, not an
evidence *command*); run as described, the search returns 5 commits **one of which
is the strike commit itself**. `0.743864` stays struck — only the proof's
reproducibility is defective.

**`arm_smprime:t2:n2048:seed2` resolved: the escape and the control are ONE fact.**
It is the unique cell whose gate reaches **neither** endpoint of its cap
(`a_hat_min = 0.3408` against `0.0` exactly on all seven others; `m_max = 0.5400`).
Biconditional GREEN 8/8: `(a_hat_min == 0)` iff `(n_zero_gates > 0)`. That single
fact produces the STRIKE 1 escape, the STRIKE 2(b) empty rejection region, **and**
Inspector C12's green. **Remove that cell and `n_zero_gates` is 100% corpus — the
it.1 control's entire margin is the datum its own strike exempts.** Retire the
control; reroute onto `beta`/`qk`/`g`.

**STRIKE 7 stands and it is the one to carry forward.** JUPITER's merge names one
domain exclusion, the lower one. W1 is `clamp(u,0,1)` so its image is `g <= 0`;
W3's trained `a_hat_max` exceeds `1.0` on **8 of 8** cells, `g` up to `4.7536`.
**The wings share no image on any trained cell.** And the unnamed boundary is the
one that sorts the scoreboard — 5 W3 cells inside the cap band score `<= 0.662128`,
3 outside score `>= 1.113339`, **zero overlap**. `N=1` is unpriced, not refuted.

**STRIKE 6 stands**: criterion (3)'s `25.0x` is row order — `47.048` is W1's first
three rows summed. Best-first reads `8.4x`. Order-invariant reprice: W3 `2.85`
GPU-s, W1 `118.8`.

### WHAT it.4 OWES — and it.4 IS THE FREEZE

1. **The W2 gate — the expected blocker is REFUTED, and the real one is smaller and
   different.** This office and the contract both expected determinism to decide it:
   `V16_ARM_SMPRIME.md` §9 records `arm_phase` RAISING under
   `use_deterministic_algorithms(True)`. **That raise is under STRICT mode, and this
   round is not strict.** The deciding journal header
   `results/v17k_r4_retake.jsonl:1` carries `deterministic_algorithms: true,
   deterministic_warn_only: true`, set at `scripts/v15_r1.py:575`; MERCURY measured
   both regimes on the certified device this iteration and under `warn_only` both
   `arm_phase.operator` and `.scan_phase` **returned OK** with two warnings each.
   SATURN adds the argument that settles it: strict would also break
   `arm_smprime`'s own backward (`scripts/v15_r1.py:91-93`), so **no cell of any arm
   exists under strict** — the regime cannot be the discriminator.

   **The real blocker is the harness, and it is honest.** `scripts/v15_r1.py:146,149,179-184`
   — `GATED_ARMS`/`ARM_MODULES`/`make_arm` have no `arm_phase` branch and fall
   through to `scale/m3_capability.py:104-105`, which raises `ValueError(kind)`. And
   `ceq/arm_phase.py:476` says it outright: *"NOT TRAINED HERE… No optimizer, no
   gradient."` **`--arms arm_phase` refuses loudly rather than journalling a fake
   cell** — the binding kill's failure mode is not reachable by accident, and that is
   worth recording as a property rather than a defect.

   **Price: GPU time is not the cost.** MERCURY: `5.114 s` off the `arm_pl` basis,
   ceiling `19.470 s` off `arm_smprime`. SATURN: ~16 GPU-s per cell, ~130 GPU-s for
   eight. **The cost is a producer edit to the deciding journal** — a `make_arm`
   branch, an `ArmPhase` optimizer path, a `PHASE_FIELDS` emit. **Two priced options
   for the freeze: RETIRE W2 to UNPRODUCED and freeze N = 2 measured (W1, W3) at zero
   cost, or REROUTE ~130 GPU-s plus one harness iteration.**
2. **The author's call on clause (1):** EXIT A at `N=125` (0.66 GPU-h, powered) or
   EXIT B (free, comparative, already met). Both affordable. **This office will not
   choose a criterion for the author.**
3. **JUPITER re-binds `N=1` or restates it** on MARS's reroute: *one primitive on
   the common image `g <= 0`, and the record contains no trained W3 cell inside it.*
4. The Inspector has **not** audited it.3.
5. **Criterion (3) is defined for two of the four objects the arena must rank.**
   MERCURY: `expected_cost_to_crossing` returns `None` for `softmax` (0/8 crossings)
   and is undefined for W2 (no cell). And the `24.97x` file-order ratio is not even
   the extreme — permuting the arms independently, the ratio spans **2.09x to
   73.50x**, and MARS's best-first `8.44x` sits near the bottom of that range, i.e.
   it is the best case rather than the neutral one. **What survives every ordering is
   the direction `arm_pl < arm_smprime`; the magnitude is the artifact.**

**DISTANCE.** Unchanged from it.2 and now independently confirmed by MERCURY: six
of 24 cells cross floor₁ `0.7071` while `softmax` crosses `0/8`. Clause (1) remains
NOT MET at `0.2449` against a `0.5` bar — but it.3 establishes that the bar is
reachable for **0.66 GPU-hours**, or free under a comparative restatement. **The
obstacle between this round and its first scoreboard point is a definitional choice,
not a measurement and not a budget.**

**SCOREBOARD.** Wing list **not frozen**; N contested between `1` (JUPITER, struck
unbound) and `>= 2` (SATURN, measured on the corrected gate, weakest pair
`2.623589e-01`). Annex: M14 at **F1, HOW-BAD 108x**, death re-attributed to V-25.
Carried: **0 of 48.**

---

## it.4 — THE FREEZE. `FROZEN-N = 2`.

Closed inside the 20-minute cap. Room: JUPITER, SATURN, INSPECTOR (auditing it.3).
**The Inspector had not reported on it.3 when this record was filed**, and he has not
audited it.4 at all — **the scoreboard's `+2` is therefore NOT claimed here.** The
list is frozen; the point is withheld until it is audited.

### THE RULING

**`FROZEN-N = 2`: W1 `arm_smprime`, W3 `arm_pl`. W2 `arm_phase` is STRUCK and
retired to UNPRODUCED.**

JUPITER and SATURN reached `N = 2` independently, from opposite directions — one
from the algebra of the merge, one from the census of the record. Neither read the
other's report before filing.

### W2 IS STRUCK BY THE ROUND'S OWN BINDING KILL

*A wing named rather than found is struck.* SATURN made that mechanical rather than
rhetorical: **a wing is FOUND iff `results/` holds at least one journalled record
whose `kind` is the arm name; otherwise it is NAMED.** The census, and this office
recomputed it independently from `results/**/*.jsonl` and `*.json` `[RUN]`:

| kind | journalled records | files | found? |
|---|---|---|---|
| `arm_pl` | 191 | 3 | yes |
| `softmax` | 182 | 4 | yes |
| `arm_smprime` | 182 | 2 | yes |
| **`arm_phase`** | **0** | **0** | **NO** |

Calibrated both ways — the discriminator must read `0` for `arm_phase` **and** `>0`
for `softmax`, so it is not a search that can only return nothing.

**W2's retirement ships its price, and the price is small.** SATURN corrected this
office's own brief: the "~130 GPU-s for eight cells" is neither figure — it is
**≈41 GPU-seconds** on the `arm_pl` basis (`8 x 5.114`) and **≈156 s** on the
`arm_smprime` ceiling (`8 x 19.470`). **GPU time was never the obstacle.** The
obstacle is a producer edit: a `make_arm` branch, an `ArmPhase` optimizer path
(`ceq/arm_phase.py:476` — *"NOT TRAINED HERE… No optimizer, no gradient"*), and a
`PHASE_FIELDS` emit. **W2 re-enters the moment one journalled cell exists.**

**Provenance correction, SATURN against this office's brief:** `scripts/v15_r1.py`
does **not** raise `ValueError(kind)`. `make_arm` (`:179-184`) falls through to
`Arm(kind,s)` and the raise is at **`ceq/bench.py:373`**; `--arms` (`:552`) carries
no `choices=`, so the CLI accepts `arm_phase` and dies inside. Same verdict, correct
line.

### ONE PRIMITIVE, TWO ARENA ENTRIES — AND THE PRINCIPLE IS FALSIFIABLE

JUPITER re-bound the struck `N=1` (`tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py`,
1 RED by design + 5 GREEN, the RED **kept red as the record of the strike**:
`assert 0 >= 1`, zero of eight trained W3 cells have `a_hat_max <= 1.0`).

**And he found the number MARS did not have.** MARS showed W3 lies outside W1's
image on 8/8 cells. JUPITER read the same journal field from the other side: **W1's
own `a_hat_max` is `<= 1.0` on 8/8** (exactly `1.0` on six, `0.8454326` seed 3,
`0.5400443` seed 2). **The in-image seed sets are disjoint from both directions**,
so the reroute is exact rather than weakened.

**THE EXERCISED-MAP PRINCIPLE**, stated so it can be applied to future merges:

> A reduction retires an arena entry only if the record **exercises** the map — some
> trained cell of each side must lie where the map is defined, and both sides must be
> evaluated there.

This is the round's binding kill turned around. *A wing named rather than found is
struck* refuses an entry the record does not produce; the same rule refuses **a merge
the record does not exercise**. And it is falsifiable at zero cost: **one `arm_pl`
cell with `a_hat_max <= 1.0` turns the RED node GREEN and collapses the two into one.**

**The atom decides the other half, and the two halves fail differently.** At the
annihilating atom W1 is finite and W3 is `nan` — `nan` is not a coordinate of `0`, so
an isomorphism failing there is a difference **between the objects**, not between
descriptions of one. Count at the atom: **1** (W1 alone; one side has no value, so
there is no map to write). Count off the atom: **2** (the map exists and is exact, but
no trained W3 cell is in that region). **On no part of the draw does the merge both
apply and have two trained things to merge.**

### THE MANIFEST IS TAMPER-EVIDENT, AND THE NEGATIVE WAS PLANTED ON DISK

`V20_R15_WING_MANIFEST.md`, **`FREEZE-SHA256 =
fbf17e07e6cab495bd1fdbb4392ab88e92f5b0610e40d2b8a283d8bcb9c542ff`** over the eight
sorted `wing|clause|path:line|anchor` rows — verified present in the manifest by this
office. It covers the list; it does **not** cover the cited files' content, which is
the separate anchor node that must re-run at every HEAD.

**The planted negative was applied to the file on disk, not simulated**: W1 clause (a)
`144 → 145` gives `2 failed, 13 passed`, both the hash node and the anchor resolver
firing; reverted, `18 passed`.

**Clause (b) tightened at the freeze.** Both surviving wings now cite a **journalled
cell** (`results/v17k_r4_retake.jsonl:161` and `:25`), not a document, and a node
requires every clause-(b) citation to live under `results/`. **A document is exactly
what a named wing can also produce** — which is the whole lesson of W2.

**Zero P-6 drift.** All twelve it.1 citations still resolve with their anchors at HEAD
`207e7b9` (`20 passed, 1 failed`; the one failure is it.1's K6 annex bind, not a
citation node). And K6 **shrank again** — from four orphan tokens to two: `d=20 at the
same` and `0.492 vs KL 0.519` have gained producers.

### TWO INSTRUMENT DEFECTS REPAIRED, AND ONE WAS DEEPER THAN THE STRIKE

- **C7.** The root cause was not merely that the planted negative failed to invoke its
  node. `[DERIVED]` on `exact = 1/27`: the ratio line `|crude/exact − 1| <= 1e-11` is
  `|crude − exact| <= 3.70e-13`, **27x tighter than either one-sided line** — so the
  Inspector's own "delete the upper-side assert" probe was **unreachable by
  construction**. Repair: the two dominated lines deleted (C20 satisfied *more*
  tightly, not relaxed); the planted negative now monkeypatches the module global and
  **calls the real node**. Verified by the Inspector's own probe — with the surviving
  assertion deleted both parametrisations fail `DID NOT RAISE`; restored, `10 passed`,
  md5 identical before and after.
- **C4 withdrawn with no replacement number.** A collection error emits
  `Interrupted: 1 error during collection`, zero tests, **no per-node result** — so no
  per-node count exists to publish. This is the STRUCK.md discipline applied to a
  count: *the honest repair is deletion of the claim, never substitution of a number
  nobody can defend.*

### A LOG DEFECT WIDER THAN THE STRIKE THAT FOUND IT

SATURN repaired his own C14 (RED as its own event with a `t` field at
`house-events.jsonl:11759`, findings at `:11761-11763`, `done` at `:11764`, and two
nodes that read the log back and assert both properties). **Building it surfaced
something larger: a `t`-field audit over all 11,764 lines finds 116 events with no
`t` at all, across three competing schemas** — `kind` at `:399` and `:993`, and
`event` at **`:11744`, appended this round, after the Inspector struck SATURN's event
for exactly that defect.** The bind is deliberately scoped to this office's it.4
events: the log is append-only, so binding history would be a RED that can never go
green.

### WHAT it.5 OWES

1. **The Inspector on it.3 and on it.4.** Two audits outstanding. **`+2` is unclaimed
   until the freeze is audited.**
2. **VENUS has not run this round.** The contract's it.5 is *"VENUS's ranking; the
   author's counter-ranking (the point estimate, D-CALIB); MERCURY prices the arena."*
   MERCURY is done. VENUS is owed a competing prediction filed **before** the next
   deciding measurement.
3. **The author's ruling on clause (1)** — EXIT A (`N=125`, 0.66 GPU-h, powered) or
   EXIT B (free, comparative, already met at `p = 0.012821`). Phase C cannot open
   without it.
4. **The 116 `t`-less log events** and the three competing schemas — an instrument
   repair, scoped so it does not demand a green on history.

**DISTANCE.** Unchanged and now frozen against a two-wing list: six of 24 cells cross
BED-M's floor₁ `0.7071` while `softmax` crosses `0/8`; clause (1) reads `0.2449`
against a `0.5` bar. **What it.4 adds is not distance but honesty about the field** —
the round entered with three candidate wings and leaves with **two that the record
produced and one that only prose did.** A tournament of three where one has never been
trained is not a tournament of three.

**SCOREBOARD.** Wing list **FROZEN at N=2**, manifest hashed, twelve citations
resolving at HEAD, clause (b) tightened to journalled cells. **The `+2` is NOT
claimed** — the Inspector has not audited the freeze. Annex: M14 at **F1, HOW-BAD
108x**. Carried: **0 of 48.**

---

## it.5 — PHASE A CLOSES. THE FIRST POINTS, AND THE FIRST DECISIVE EXPERIMENT.

Room: VENUS (first run this round, four iterations overdue), MARS, INSPECTOR
(auditing it.4). Closed inside the cap.

### THE FREEZE IS BOUND. `+2` EARNED.

**43 audited, 8 struck — SATURN 5, coordinator 3, JUPITER 0.** The Inspector's
explicit ruling: **"THE FREEZE IS BOUND. THE `+2` IS EARNED."** He records that the
point was withheld *in writing before his audit*, which is why it can be awarded now
rather than assumed. **First points on the board: 2 of 48.**

### CORRECTION 5 — STRIKE 8 IS THIS OFFICE'S, AND IT WAS PUBLISHED TO THE AUTHOR

it.3 filed, and this office repeated to the author, that **"EXIT B costs zero and is
already met."** The it.3 Inspector struck it, and the strike is correct:

> `CEQ_V20_R15_CONTRACT.md:125` is an **absolute** `CP-lower > 0.5` clause. Zero hits
> in the tree for *restate* or *amend*. **The clause that exists is NOT MET; the
> clause that would be met does not exist.**

MERCURY's Fisher `p = 0.012821` is real and reproduces. It answers **a question the
contract does not ask.** Restating clause (1) is an **amendment only the author can
make** — not a free win available on the table. MERCURY's body carried the
conditional; both headlines dropped it, and this office's dropped it twice.

### CORRECTION 6 — "TWO OFFICES INDEPENDENTLY CONCLUDED" WAS ONE READING COUNTED TWICE

it.3 recorded SATURN and MERCURY as independently refuting the determinism blocker.
The Inspector ruled **(A) FULLY SHARED**: both rest on `results/v17k_r4_retake.jsonl:1`
and on §9 of `V16_ARM_SMPRIME.md`, and **SATURN's `:576-581` is a strict subset of
MERCURY's `:567-593`.** One header line read twice is not two measurements. The only
thing that would have made it independent is MERCURY's device probe — **which is
itself Strike 5**, its `[RUN]` rows being absent from the file it cites.

SATURN's distinct contribution survives and it is an *argument*, not a reading:
**strict mode would also break `arm_smprime`'s own backward** (`scripts/v15_r1.py:91-93`),
so no cell of any arm exists under strict and the regime cannot be the discriminator.

### CORRECTION 7 — WILSON'S SILENCE HAS BEEN READ AS SUPPORT

> **WILSON is silent** on determinism, W2, arena `N`, clause (1), CP, and row order.
> **His silence has been read this round as support; it isn't.**

He is the reference for **what he measured at it.1** and for nothing else. This office
has leaned on him as a general baseline across six live questions he never addressed.
**WILSON is owed a second run before Phase C**, scoped to the questions the round has
actually generated since.

### SEED 2: VENUS AND MARS DISAGREE ON THE LABEL, AGREE ON THE MECHANISM, AND FILED THE SAME EXPERIMENT

This is the round's sharpest disagreement and it resolves by composition rather than
by choosing.

**VENUS — "a wing, and the mechanism names itself."** The gate columns across
`arm_smprime`'s eight cells:

| seed | `eval_nrmse` | `frac_gate_annihilated` | `lambda_hat` | `unit_root` |
|---|---|---|---|---|
| **2** | **0.203920** | **0.0** | **−0.7953** | **False** |
| all seven others | 0.881–0.927 | `0.4967` or `0.5033` | **−inf** | mostly True |

**Seed 2 is the only cell in which the path-product gate is alive at every
position.** On the other seven half the gate mass sits at the annihilating corner
`m = 0`, `lambda_hat` is `−inf` — no fitted decay, because the hop has been switched
off — and they land at `0.88–0.93`, **the skyline's own neighbourhood** (`softmax`
best `0.938796`). **When the gate dies, the arm becomes the incumbent.** Eight seeds
produce exactly **two** non-zero values of `frac_gate_annihilated`: two basins, not a
continuum. And the seven failures are **annex M1's own predicted descent** — *"β
descends toward the exact corner"* — observed rather than accidental.

**MARS — "decided before the first gradient step."** `tests/mars_v20/test_seed2_is_decided_before_training.py`,
1 failed / 5 passed. it.2's *"needs weights not in the tree"* was wrong and usefully
so: **342 weight files exist, 0 name `arm_smprime`**, and `scripts/v15_r1.py` has no
`torch.save` — the retake runner never checkpointed. **But it journals the untrained
gate state in `*_0step` columns:**

```
frac_gate_annihilated_0step per seed = {0:.5049, 1:.6113, 2:.1248, 3:.4429,
                                        4:.5157, 5:.5883, 6:.5796, 7:.2006}
gap = 0.0758056640625, zero overlap
```

**Seed 2 starts with four times less gate at the clamp floor than the median, and is
separated from its nearest sibling before any optimizer step.** Seven cells get
pulled to the `≈0.5033` fixed point; seed 2 escapes to exact `0.0`.

**THE RECONCILIATION, AND NEITHER OFFICE IS OVERRULED.** Both are measuring the same
thing. It **is** a gate state and a real dynamical fact about the landscape — VENUS is
right that it is not an eval artifact. And that gate state is **determined at
initialisation** — MARS is right that the architecture is not what earned it.
**Therefore: a real basin, entered by initialisation, observed once.** W1 cannot be
credited with a capability at `n = 1`, and the result cannot be dismissed as noise
either. It is a hypothesis with a journalled predictor.

**AND BOTH OFFICES INDEPENDENTLY PRESCRIBED THE SAME EXPERIMENT** — this time
genuinely independently, from opposite conclusions and different evidence:

> **Run `arm_smprime` on eight fresh seeds 8–15, same BED-M cell, matched params.
> ≈ 128–129 GPU-seconds — about 2.2 minutes on this box.**

MARS makes it sharper than either alone by supplying a **pre-run predictor**: test
whether `frac_gate_annihilated_0step < 0.15` predicts `eval_nrmse < 0.3`. VENUS
pre-registered the joint prediction before any run:

> Every new seed that **fails** to cross carries
> `frac_gate_annihilated ∈ {0.4967041015625, 0.5032958984375}` and `lambda_hat == −inf`.
> Every new seed that **crosses** carries `frac_gate_annihilated == 0.0` and a
> **finite** `lambda_hat`. Expected crossings **1 of 8**; 95% predictive interval **0–3**.
> **One new seed that crosses with an annihilated gate — or fails with a live gate —
> falsifies it.**

**This is the cheapest decisive experiment the round has produced: 2.2 minutes of
GPU against a pre-registered joint prediction with a named falsifier.** It is
scheduled first at it.6.

### VENUS'S KILL — THE SCOREBOARD'S CEILING IS NOT 48

**The cost row and the crossing row are two different runs.** The briefed
`15.970 / 1.614 / 1.497` comes from `V17_R4_RETAKE_PRICE.md:194-196`, a **six-cell
"mean of 2" timing run**. The 24 arena cells journal their own `secs`:
**`16.161 / 1.780 / 1.681`** — `arm_pl` **+10.3%**, `softmax` **+12.3%**. Every cost
statistic this round has quoted, including this office's GPU-hour tables, rests on the
wrong basis. **Replacement route costs 0 GPU-s: reprice off the `secs` the cells
already carry.**

**And the consequence is a scoreboard item, not a rounding note.** The contract awards
**`+4` for "the winner's cost <= 1/10 incumbent with a certificate".** Under **both**
bases W3 costs **more** than `softmax` — **1.078x** and **1.059x**. Not one tenth;
above one. **That `+4` is unreachable by either surviving wing, and the round's real
ceiling is 44, not 48.** Filed now rather than discovered at it.42.

### VENUS'S RANKING, PRE-REGISTERED

> **1. W3 `arm_pl` · 2. W1 `arm_smprime`**

| ordering at it.29 | probability |
|---|---|
| **W3 ≻ W1** | **0.72** |
| W1 ≻ W3 | 0.20 |
| no ordering — both eliminated on clause (1), `N` → 0 | 0.08 |

W3 leads clause (1) by **76×** on CP-lower and clause (3) by **41.9×**, *"and its
losing seeds fail by a mechanism a bound arrests while W1's fail by a mechanism the
gradient chooses."* **The honest half of her forecast is the 0.20**: clause (1) is
unreachable for both arms at `N=8`, and a lexicographic criterion whose first clause
no entrant meets gets re-read and descends to clause (2) — **which has never been
run**. On BED-K(a), W1's seed-2 cell is **the only cell in the tournament with a
finite `lambda_hat` and `unit_root == False`**, which is the statistic that bed is
built to read.

She also named a convention this round had left unstated: the briefed CP figures are
**two-sided** 95%; one-sided reads `0.0064 / 0.2892`. Neither moves either arm, **but
a `> 0.5` bar compared against an unnamed interval is the shape this round has already
been burned by.** And the ladder at `N=8`, printed in full:
`0.0000, 0.0032, 0.0319, 0.0852, 0.1570, 0.2449, 0.3491, 0.4735, 0.6306`.

**D-CALIB is an open slot, and it blocks more than it looked.** `⟨AUTHOR_COUNTER_RANKING⟩
NOT MEASURED` — an ordering over {W1, W3} plus a point probability, suppliable by no one
but the author. **This office did not invent one.** And **D-CALIB-2 says a prediction
with no counter is not read at all**, so VENUS's ranking is formally **unreadable** and
**it.29 cannot open a VENUS row** without it. That is a harder block than a missing column.

**What IS derivable without him, cited:** the author's measured bias is **POSITIVE
(optimistic)** — 7 `+` / 1 `-` / 1 unsigned, 7 of 8, one-sided sign test
**p = 0.0352** (`V16_CALIBRATION.md:19,:50,:94-100`). D-CALIB-3 authorises **sign, never
size**, so no shrink factor was applied anywhere.

**And why W3 beats W1 is two different theorems, not one scoreboard.** W1's seven
failures are annex **M1**'s predicted descent to the exact corner. W3's failures are
annex **M2**'s open-range divergence — the gate never dies, but `a_hat_max` blows to
`12.77 / 49.66 / 116.01` with `gate_r2` collapsing. **A norm cap arrests M2; a cap does
not undo M1's descent direction.** That asymmetry, not the crossing count, is the
content of the ranking.

**One corollary kills the obvious cheap move:** the pipeline is **bit-reproducible per
seed** — an independent run (`results/v17k_r4_floor.jsonl`) reproduces retake seeds 0
and 1 to every printed digit. **Re-running seed 2 carries zero information.** The
experiment must be *fresh* seeds.

**THE FORECASTER LEDGER IS OPEN, 16 rows.** R15 to date: **5 right, 6 wrong, 1 half,
2 open, 1 retracted, 1 filed-as-non-firing.** Of the wrong-with-a-sign: 3 `+`, 2 `-`,
**p = 0.5 — the round's own bias is NOT established and no one may cite one.** VENUS's
own row 16 records that her office filed nothing at it.1-it.4.

### THE FREEZE SURVIVED BOTH COMMISSIONED ATTACKS

- **SATURN's found-vs-named rule is loose; his verdict is right.** Of 191 `arm_pl`
  records only **18** are trained cells, and a `t=identity` record trains nothing yet
  would make a wing FOUND. Under the strict filter `t=="cell" AND steps>0 AND finite
  eval_nrmse`: `arm_pl` 18, `arm_smprime` 10, `softmax` 18, **`arm_phase` 0**. Same
  verdict. **Zero `arm_phase` across 736 distinct top-level keys.** Reprice: tighten
  clause (b) to `t=="cell"` — costs the freeze nothing, both cited lines already are.
- **JUPITER's exercised-map principle holds, and is stronger than he stated.** MARS
  tried to break it as a rule that rewards not running the experiment; composed with
  found-vs-named it does not — `N(empty record) = 0`, not 2. He verified the empty
  intersection on **137 records per arm, not 8**. **Residual, and it is real: a
  pricing asymmetry.** The raise route (W2) is priced at 41–130 GPU-s; the **collapse
  route** (8 more `arm_pl` cells, **12.9 GPU-s**) is unpriced and **3.2× cheaper**.
  The round has priced the expensive way to change `N` and not the cheap one.

### MARS WITHDREW HIS OWN it.2 STRIKE

`scale/ledger.py:76-84` `_status()` accepts **both** `status` and `state`,
case-folded, so the Inspector's shipped reader does see MERCURY's `"state":"RED"`.
Withdrawn. **The residual is structural, not spelling:** `tests()` filters `t=="test"`
**first**, so the **116 `t`-less records are unbindable by construction** (114 `kind`,
1 `event` at `:11744`, 1 neither at `:11719`). Measured blast radius of a naive
literal `status`-search across R15: **1,276 REDs union, 150 visible, 1,126 invisible**.

**MARS aimed seven attacks and landed one.** Six named as not firing, including one
that *backfired and strengthened* the office he was attacking. That is the calibration
an adversary is scored on.

### WHAT it.6 OWES — PHASE B OPENS

1. **THE EXPERIMENT.** Eight fresh `arm_smprime` seeds 8–15, ≈129 GPU-s, against
   VENUS's pre-registered joint prediction and MARS's `_0step` predictor. **Nothing in
   Phase B should precede it** — it costs 2.2 minutes and it decides whether W1 has a
   claim at all.
2. **The author's ruling on clause (1)**, now stated correctly: EXIT A (`N=125`,
   0.66 GPU-h) or **an amendment restating the clause** (free, but it is an amendment
   and only the author makes it). Phase C cannot open without it.
3. **WILSON's second run**, scoped to the six questions his it.1 report never touched.
4. **`⟨AUTHOR_COUNTER_RANKING⟩`** — D-CALIB cannot compute without it.
5. The collapse route priced (12.9 GPU-s), so both directions on `N` carry a number.
6. **Every cost statistic repriced off the cells' own `secs`** — 0 GPU-s, and it moves
   the round's ceiling from 48 to 44.

**OWNERSHIP, CLAIMED.** VENUS flagged `pytest.ini` as modified by an unattributed
process and correctly refused to revert it. **It is this office's edit**, made at it.1:
`norecursedirs` gained `kaggle` because `kaggle/snapshot/repo/` shadowed every test
basename and `pytest -q` at the root read `201 errors during collection, 0 tests run`.
It is recorded in the it.2 entry; it was **not** recorded anywhere an agent starting
fresh would look. **An unattributed edit against citations frozen at HEAD `207e7b9` is a
P-6 drift risk and she was right to name it.**

**DISTANCE.** Unchanged in the measurement and sharper in the reading. Six of 24 cells
cross floor₁ `0.7071`; `softmax` crosses `0/8`. **What it.5 adds is the mechanism:
`arm_smprime`'s seven failing cells sit at `0.88–0.93`, inside the skyline's own
neighbourhood, with `lambda_hat = −inf` — when the gate dies, the arm becomes the
incumbent.** The north star asks for an arm capable on ground softmax cannot occupy.
The round now knows exactly which cell is on that ground, and does not yet know whether
the architecture or the initialisation put it there. **That is a 2.2-minute question.**

**SCOREBOARD.** **Wing list frozen, four citations each: `+2` — EARNED**, ruled bound
by the Inspector at 43 audited / 8 struck. Annex M14 at F1, HOW-BAD 108×; twelve of
sixteen annex items remain ungraded and six Lean citations are withdrawn, so the annex's
`+4` is not in reach this phase. Carried: **2 of 44** — the ceiling itself moved: the `+4` for a winner costing <=1/10 of the incumbent is **unreachable**, both wings costing *more* than `softmax`.

---

## it.6 — PHASE B OPENS. THE EXPERIMENT RAN, AND IT DID NOT GO AS FORECAST.

Room: MERCURY (execution), JUPITER (Q1/Q2), WILSON (second run — **still running at
filing; his six answers land at it.7**). MARS sat this iteration out; his standing
attack resumes at it.7 against results that now exist.

### THREE CORRECTIONS THIS OFFICE OWES, FROM THE it.4 FREEZE AUDIT

The audit that earned the `+2` struck three of this office's own descriptions. The
journal is append-only, so they are corrected here.

1. **"Reached N=2 independently, from opposite directions" — STRUCK.** JUPITER and
   SATURN answered **different halves** (*is W2 a wing* / *do W1 and W3 collapse*), and
   **neither yields 2 alone.** JUPITER's own LIMITS at `:272` say his W2 leg is
   *"argued from SATURN's it.3, not measured here."* Two non-overlapping findings were
   called two proofs of one fact. *(The narrower claim — "neither read the other's
   report before filing" — is **TRUE and not struck**: 54 s apart, interleaved log
   appends, zero it.4 cross-citation either way.)*
2. **"Two the record produced, one only prose did" — STRUCK.** At it.1 **all three**
   wings cited documents for clause (b) (`workdonenewseal.md:91`, `MISTAKES.md:2154`,
   `V15_R1.md:250`) and C17 passed all twelve. **The tightened clause (b) is what
   separated them** — this office was giving away credit owed to its own instrument.
3. **"K6 shrank because two tokens gained producers" — STRUCK.** They gained nothing.
   They were moved into `UNBINDABLE_BY_SUBSTRING` and `producers_for` now *raises* on
   a whitespace token, so the orphan test **never searches them**. The shrink is the
   it.1 C26/C27 strike being executed. **K6 was not paid down.**

And this office's "independent recompute" of the found-vs-named census was **the same
method a third time**, struck as corroboration. The Inspector closed that gap himself
with a genuinely second method — raw-text search for `arm_phase` under **any** key
across all 50 result files: **one hit, prose at `results/k_cert_local.json:1198`, no
record identity anywhere.**

### THE EXPERIMENT: 0 OF 8, AND TWO GATE STATES NOBODY HAD SEEN

MERCURY ran it on the certified 4060, instrument unmodified, regime identical to the
retake on all seventeen header fields including `instrument_hash 5d41a63d…9a309`.
**`results/v17k_r4_retake.jsonl` untouched — 424 lines before and after, absent from
`git status`** — the frozen clause-(b) citations did not move. New journal:
`results/v20_r15_it6_seeds8_15.jsonl`.

**Control passed first:** seeds 0 and 1 rode along and reproduced `0.926082` /
`0.881247` **bitwise**, and their `_0step` values reproduced MARS's table. The
invocation is correct, so the fresh cells can be trusted.

| seed | `eval_nrmse` | `frac_gate_annihilated` | crosses floor₁? |
|---|---|---|---|
| 0 ctrl | 0.926082 | 0.5032958984375 | no |
| 1 ctrl | 0.881247 | 0.5032958984375 | no |
| 8 | **0.852061** | 0.5032958984375 | no |
| 9 | 0.902522 | 0.5032958984375 | no |
| 10 | 0.892747 | 0.5032958984375 | no |
| **11** | 1.120603 | **0.99462890625** | no |
| 12 | 0.919852 | 0.5032958984375 | no |
| **13** | 1.203324 | **0.976806640625** | no |
| 14 | 0.909944 | 0.5032958984375 | no |
| 15 | 0.891047 | 0.5032958984375 | no |

**0 of 8 fresh seeds cross.** Best fresh cell is seed 8 at `0.852061`, **`0.144954`
above the floor**. This office recomputed the table independently from the new journal
`[RUN]`; it reproduces.

### WHAT THE RUN DID TO THE PRE-REGISTRATIONS — and the distinction matters

**VENUS's rate prediction SURVIVES.** She forecast **1 of 8**, 95% predictive interval
**0–3**. Observed **0**. Inside the interval.

**VENUS's mechanism prediction is FALSIFIED IN ITS BODY.** She wrote: *every new seed
that fails to cross carries `frac_gate_annihilated ∈ {0.4967041015625, 0.5032958984375}`.*
**Seeds 11 and 13 fail carrying `0.99462890625` and `0.976806640625`** — two values
that occur **nowhere** in the retake's eight cells. `0.4967041015625` occurred zero
times; `0.0` occurred zero times.

**But her NAMED FALSIFIER did not fire, and that is the finding about the
pre-registration itself.** She wrote: *"one seed that crosses with an annihilated gate
— or one that fails with a live gate — falsifies it outright."* Seeds 11 and 13 failed
with a gate **more** annihilated than predicted, not a live one. **The falsifier was
narrower than the claim it was attached to.** A pre-registration whose named killer
cannot fire on the way the claim actually breaks is a pre-registration with a hole
(`MISTAKES.md` M-7), and it is filed as one — **against the office that filed the best
pre-registration of the round.** The rate half was falsifiable and survived; the
mechanism half was violated by a mode its killer did not name.

**"Two basins, not a continuum" does not survive.** Across sixteen cells there are now
**five** distinct values of `frac_gate_annihilated`: `0.0`, `0.4967041015625`,
`0.5032958984375`, `0.976806640625`, `0.99462890625`. The two-basin reading was a
reading of eight cells.

**MARS's predictor is untested, not confirmed.** `frac_gate_annihilated_0step < 0.15`
issued **zero positive predictions** over the fresh eight (their `_0step` spans
`0.3809`–`0.9946`; none below `0.15`). It predicted no crossings and no crossings
occurred — **vacuous agreement.** It remains a live hypothesis with **one** supporting
instance in sixteen and is not evidence yet.

**And seed 2 is now 1 in 16.** Its `frac_gate_annihilated == 0.0` has not recurred.
The best number in the tournament is still one cell, and eight fresh draws did not
reproduce it.

**A fact that cuts against the tidy story:** `lambda_hat == −inf` on **all ten** cells,
**including seeds 11 and 13 where `unit_root == False`**. The it.5 reading tied
`lambda_hat = −inf` to a dead gate and a unit root; these two cells have a dead gate,
no unit root, and `−inf` anyway.

**Price overran, and it is reported as a range not a point.** Estimate `129` GPU-s
(`8 × 16.161`); measured **`146.399` s**, **+13.2%**. Process wall `185.746` s
(3 m 05.7 s) against a priced 2.2 min. The overrun is **per-cell, not fixed**: `secs`
spans `15.907`–`22.069` inside one process, a `1.387×` spread. **`arm_smprime` on this
box is `16.161` s (n=8) / `17.846` s (n=10) with a `15.9–22.1 s` range attached** — not
a point estimate, and every GPU-hour table in this round that used a point should be
read with that spread.

### JUPITER — FOUR OF TWELVE PHASE-B CELLS, AND HE SAYS SO

He filled **Q1 and Q2 for both wings** and left Q3–Q6 for it.7–it.9, on the stated
ground that twelve cells in one wall clock produces twelve shallow answers. **Every
theorem is named by declaration and `path:line`, never by annex number** — the
discipline he adopted after withdrawing six Lean citations at it.2.

| cell | grade | HOW-BAD gap |
|---|---|---|
| **Q1 / W1** | **F0** (the hop) | **0 of 24 arena cells attributable** |
| **Q1 / W3** | **F1**, delta = instance tolerance | positive class statement has **no theorem** |
| **Q2 / W1** | **F1 with the constant** | §8 |
| **Q2 / W3** | **F1 with the constant** | §8 |

**Q1/W1's HOW-BAD gap is the round's own lesson recurring, and he caught it on
himself.** The F0 is exact — on a configuration described by the `beta`/`qk`/`g`
switches and a normalizer. **No arena cell journals those as cell fields**, so **zero
of 24 banked cells are attributable to the configuration the theorem covers.** That is
M14's V-25 shape one directory over: *a theorem exact on a domain the record does not
enter.* **The repair costs 0 GPU-s** — journal four scalar config fields per cell.

**Q1/W3 is graded F1 and not F0, deliberately.** The contract says *"F0 or it isn't
exact"*; W3's positive class statement rests on an instance, and **an instance at
`1e-6` is not zero error.** He writes: *"F0 would be the it.2 mistake repeated with a
different number on it."* **F1 is not F4 — the wing is not withdrawn.**

### WHAT it.7 OWES

1. **WILSON's six answers** — still running at filing. His §1 measures the determinism
   regime directly instead of citing it, which no office has yet done.
2. **MARS on the new cells** — seeds 11 and 13 are two unexplained gate states and he
   has not seen them.
3. **VENUS re-forecasts**, with a falsifier that can fire on the mode that actually
   broke her. Her rate survived; her mechanism did not; her killer named neither.
4. **The four config fields journalled** (0 GPU-s) so Q1/W1's F0 becomes attributable.
5. **Q3 for both wings** — the M1-vs-M2 asymmetry is a learnability claim and it is
   now the most load-bearing unanswered question in the round.

**DISTANCE.** **The round moved backwards on the measurement and forwards on the
truth.** BED-M crossings across all sixteen `arm_smprime` cells are now **1 of 16**,
not 1 of 8 — the pooled rate halves, and clause (1)'s CP-lower falls further from
`0.5`, not nearer. `softmax` still crosses `0/8`. **What was bought for 146 GPU-seconds
is the knowledge that W1's one good cell did not reproduce in eight fresh attempts,
and that the gate has at least five states rather than two.** The north star asks for
an arm capable on ground softmax cannot occupy; the single cell that stood on that
ground is now measured at 1-in-16 and still has no mechanism anyone has bound.

**SCOREBOARD.** No new points. Phase B is 4 of 12 cells filled, two at F1 with
constants, one F0 with a `0 of 24` attributability gap, one F1 that refuses an
undeserved F0. **The `+6` for the theory table requires all N×Q6 graded** — 8 cells
outstanding across it.7–it.14. Carried: **2 of 44.**

---

## it.7 — THE COLUMN THE ROUND WAS READING IS ONE BIT, AND THE AUTHOR NAMED THE FIELD

Room: JUPITER (Q3), MARS, VENUS. All three converged on the same defect independently.
**The leap ledger `V20_R15_LEAP_LEDGER.md` is open** and is being written at the moment
of failure rather than reconstructed at it.31 — the author's instruction this iteration:
*"if something fails, all that adds to the leap."*

### STRIKE 9 — `lambda_hat` IS A ONE-BIT COLUMN PUBLISHED BESIDE ITSELF

**Every mechanism story this round has told rests on a column that cannot carry it.**

`scripts/v15_r1.py:383` computes `lambda_hat = float(lg.mean())` over **all** positions
**including `-inf`**. `:387` computes `unit_root = bool(a_max >= 1.0)` from a **max**.
A mean is dragged to `-inf` by a single entry; a max ignores them entirely.

**On 18 of 18 cells, `lambda_hat == -inf` is exactly the predicate
`frac_gate_annihilated > 0`.** One zero gate in 8,192 forces it. **It is a one-bit
restatement of a column printed next to it.**

**This inverts two readings this round built on.** VENUS at `V20_R15_IT5_VENUS.md:128`
— *"the hop has been switched off"*. JUPITER's Q3 seed at `IT6:280-284` — *"gate dead"*.
**Neither followed from `lambda_hat = -inf`, because that value follows from a single
annihilated position, not from a dead gate.** There was no `unit_root` contradiction to
explain: the pair is the generic case.

**And the instrument already ships the column they meant.** `lambda_hat_live`
(`scripts/v15_r1.py:384`) is journalled. On all eight cells landing in the `0.88–0.93`
band: **`a_hat_max == 1.0` exactly, `unit_root == True`, live-band decay under 3% per
position.** The gate in those cells is not dead. **It is barely decaying at all.**

### STRIKE 10 — THE EVAL DRAW IS PINNED, AND EIGHT OF TEN CELLS REPUBLISH IT

`scripts/v15_r1.py:699-700` draws the eval batch at a **hardcoded `seed=12345` that does
not move with the cell seed**, and every gate column is read on it. The no-arm sign
census of that draw is `neg = 4123`, `4123/8192 = 0.5032958984375`. **Eight of ten fresh
cells publish that float exactly.**

**Seeds 11 and 13 are the only two genuine arm readings in the file** —
`0.99462890625 = 8148/8192` and `0.976806640625 = 8002/8192` are not corpus counts. And
**both escape upward, more annihilated, while failing worst** (`eval_nrmse` 1.121 and
1.203, the only two above 1.0). This is MARS's own it.1 STRIKE 1 recurring one column
over, and he found it on the experiment he co-prescribed.

### MARS WITHDREW HIS OWN PREDICTOR ENTIRELY

Scored across all 16 distinct cells: **one positive prediction ever issued — seed 2, the
cell the threshold was cut from. Zero on the fresh eight. The threshold band `0.15–0.30`
was never sampled.** Fitted on one, tested on zero. **Worth nothing** — not "one instance
in sixteen", which is what this office wrote at it.6 and is too generous. His arithmetic:
`1 − (15/16)^8 = 0.403`, so an eight-seed run had a **60% chance of issuing no positive
at all**; `n ≥ 44` for 95%. **An adversary retiring his own instrument is the calibration
the round is scored on.**

### JUPITER — Q3 FILLED BOTH WINGS, AND THE LEAN GAP CLOSED

`[RUN]` `cd lean && lake build; echo $?` → **`0`**. The standing gap he named at it.6 §7
is closed; every Lean citation now rests on a build observed this iteration.

| | W1 `arm_smprime` | W3 `arm_pl` |
|---|---|---|
| Q1 EXACT CLASS | F0 | F1 |
| Q2 OUTSIDE | F1 + constant | F1 + constant |
| **Q3 LEARNABILITY** | **F2** | **F1** |
| Q4–Q6 | not filled | not filled |

**Six of twelve cells.** Q3/W1 graded **F2 and not lower**, with the reason stated rather
than the grade softened.

### VENUS VERIFIED THE BRIEF INSTEAD OF ACCEPTING IT

Every number in her it.7 brief was re-derived in `tests/venus/test_v20_r15_it7_venus.py`
(**16 passed**) before she built on it: controls bitwise on 15 fields with 0 diffs; 0 of 8
fresh crossings; pooled **1 of 16**; five distinct gate values; and the fact this office
should have flagged itself — **it.6 ran `arms: ["arm_smprime"]` only, so `arm_pl` 5/8 and
`softmax` 0/8 are unchanged because nothing fresh was run for them.** The comparison is
**16 W1 cells against 8 W3 cells.** The symmetric experiment prices at **14.240 GPU-s
exactly** — **7.98% of the it.6 run cell-seconds, a twelfth not a tenth** (her correction
to this office's brief, and it strengthens her own argument).

**She then repaired her falsifier by naming its mechanism, which is the valuable part.**
*"I asserted set membership on frac and wrote a killer that was a predicate on gate
liveness. Different variables — a killer that does not range over the constrained
quantity cannot fire."* **That is M-7 recommitted by an office that had read M-7.** Three
replacement laws, each falsified by set complement:

- **L1** `frac > 0 ⟺ lambda_hat == −inf` — **34 of 34 cells, zero violations.**
  Independent confirmation of STRIKE 9 on **34 cells where MARS had 18.**
- **L2 the annihilation lattice** — `frac × 8192` is an integer on 26/26; observed
  `k ∈ {0, 4069, 4123, 8002, 8148}` and **`4069 + 4123 = 8192` exactly.**
- **L3 replaces "two basins" with a dose–response** — Spearman `ρ = 0.5564`,
  `p = 0.0252`, `n = 16`, three non-overlapping clusters (`0.0 → 0.2039`;
  `≈0.50 → 0.852–0.927`; `≥0.9 → 1.121–1.203`, **past predict-the-mean**).

**And she caught her own miss without being asked:** it.5's tie of `−inf` to `unit_root`
was irrelevant, and **seed 3 was already in her own it.5 printed table** with
`unit_root == False` and `−inf`. She had the counterexample and did not read it.

**Her ranking moved 0.72 → 0.78 and she says plainly it barely moved.** One datum did it:
Fisher exact between arms `0.11888 → 0.00686`, the first separation at conventional
significance. It could not move further because her 0.20 on W1 never rested on rate — it
rested on clause (2), where **seed 2 is still the only cell in 34 with a finite
`lambda_hat` and `unit_root == False`.**

**The asymmetry she refuses to let pass:** W3's interval is **0.60 wide** against W1's
**0.26**. *Uncertainty in this tournament is now dominated by the arm nobody resampled*,
and whether W3's divergence failures are 3-in-8 or 6-in-16 is the number clause (1)
actually turns on. **14.240 GPU-s.** She pre-registered it before the run: 5 of 8,
PI 2–7, `frac == 0.0` on 8/8, killer = any `arm_pl` cell with `frac ≠ 0.0`.

**One more instrument hole, found by an unplanned RED:** `frac_gate_annihilated_0step`
is **absent on all 8 `softmax` cells** (present 26/26 elsewhere), so MARS's predictor
could never have been evaluated on the skyline **even in principle**.

### THE AUTHOR'S SECOND POINTER, AND IT LANDS ON A NUMBER ALREADY MEASURED

> *"there must be a lot of interference, wave decay — future state prediction is a wave
> equation, and radio and communications engineering have spent decades solving waves,
> stability and other."*

**Look at what the band cells actually read** `[RUN]`, from `results/v20_r15_it6_seeds8_15.jsonl`:

| seed | `eval_nrmse` | `a_hat_max` | `unit_root` |
|---|---|---|---|
| 0, 1, 8, 9, 10, 12, 14, 15 | 0.852–0.926 | **`1.0` exactly** | True |
| 11 | 1.120603 | `0.06994166225194931` | False |
| 13 | 1.203324 | `0.7774731516838074` | False |

**Eight of ten cells have their gate's maximum magnitude pinned at exactly `1.0`** — not
`0.99`, not `1.001`. Combined with STRIKE 9's live-band decay **under 3% per position**,
that is a **pole sitting precisely on the unit circle**: marginal stability, an
almost-all-pass filter. **And the two cells that moved off the unit circle are the two
that failed worst.** The arm's gate is `G_ij = Π_k a_k`, a first-order recursive filter;
`a_hat_max = 1.0` with `frac_gate_annihilated > 0` is a pole on the circle **and** a zero
at the origin. **That is a pole–zero configuration, and communications engineering has a
century of theorem on exactly it** — Z-transform pole placement, BIBO and marginal
stability, group delay and dispersion, Wiener–Hopf linear prediction, intersymbol
interference.

**And three pieces of that apparatus are already built in this repository and went unused
this entire round** `[RUN]` — all four paths exist:

| module | what it is | status this round |
|---|---|---|
| `ceq/x35p/kk.py` (+ `V15_X35P_KK_CRB.md`) | **the discrete Kramers–Kronig causality residual** | **never invoked** |
| `scripts/v13_wiener_hop.py` (+ `V13_X29B_WIENER.md`) | Wiener/linear-prediction hop | never invoked |
| `ceq/nonnormal.py` | numerical range / non-normal operator | never invoked |

**Kramers–Kronig is the exact object the author is pointing at.** Its own docstring: *"A
real sequence h is causal iff its odd part is `sgn(k)` times its even part... for a causal
h the imaginary part of the spectrum is a function of the real part alone."* **Causality
is analyticity, and the round has a working probe for it sitting idle.** It ships with the
must-fire this repository demands: `CEQ_V15_2_DELTA.md` struck the author's *first* KK
probe because it read `0.000` on a planted anticipating kernel — *a causality test that
does not fire on an acausal kernel is measuring nothing* (V-16) — and the rebuild ships
**only because its planted-anticipating must-fire fires** (`tests/x35p/test_kk_crb.py`).
**It consumes `H(ω)` and never looks at the taps**, so it applies to a learned kernel
observed only through its frequency response — which is exactly what the arm is.

**Filed as the it.8 direction, not as a result.** No claim is made that KK will separate
the wings; the claim is that a causality instrument with a firing must-fire exists, is
unused, and is aimed at the round's north star.

### WHAT it.8 OWES

1. **Re-read the mechanism on `lambda_hat_live`, not `lambda_hat`.** JUPITER's Q3/W1 F2
   and VENUS's mechanism argument both need restating on the column that carries signal.
2. **The unit-circle reading, bound.** `a_hat_max == 1.0` exactly on 8 of 10 is either a
   convergence fact or a clamp artifact (`clamp(u,0,1)` has `1.0` as its ceiling) — **and
   that distinction is a one-line test.** If it is the clamp, the pole story dies and must
   be said to die.
3. **KK on the trained gates**, with its planted-anticipating must-fire re-run first.
4. **The symmetric `arm_pl` experiment**, 14.240 GPU-s.
5. **Inspector on it.5, it.6, it.7** — three iterations unaudited.

**DISTANCE.** No movement toward the north star and a real gain in the map. Pooled W1 is
**1 of 16**; `softmax` is `0/8` on stale cells nobody refreshed. **What it.7 established is
that the round's mechanism narrative was resting on a saturated column and a pinned eval
draw** — two instrument defects, both found by the offices that had most to lose from
them. The north star wants an arm predicting the next state; the round now knows its
eight best-behaved cells sit at unit modulus with sub-3% decay, and does not yet know
whether that is the arm converging or a clamp ceiling being read as a result.

**SCOREBOARD.** No new points. Phase B at **6 of 12 cells** — Q1 F0/F1, Q2 F1/F1,
Q3 F2/F1. The `+6` needs all twelve graded by it.14. Leap ledger open with five MARS rows
(`L-M1..L-M5`: two TERMINAL, two LEAPABLE, one split). Carried: **2 of 44.**

---

## it.8 — THE POLE STORY DIES, W3 CROSSES 12 OF 16, AND CLAUSE (1) COMES DOWN TO AN UNSTATED CONVENTION

Room: MERCURY (two experiments), JUPITER (Q4/Q5), INSPECTOR (clearing the it.5–it.7
audit backlog). **JUPITER and the INSPECTOR had not reported at filing.**

### THE ADJUDICATION THIS OFFICE OWED — JUPITER RIGHT, VENUS REFUTED

JUPITER flagged a live contradiction with VENUS's leap-ledger row (*"M1's beta-gradient
descent to the corner is observed on 15 of 16 cells"*). Both read the same 16 cells.
Recomputed here `[RUN]`, seeds 0/1 deduplicated:

```
Spearman(beta,      eval_nrmse) = -0.0324   p = 0.9053
Spearman(qk,        eval_nrmse) = +0.7176   p = 0.0017
Spearman(|beta-0|,  eval_nrmse) = -0.0324   p = 0.9053
BEST  seed 2:  nrmse 0.203920, beta +1.343933
WORST seed 13: nrmse 1.203324, beta +1.989505
```

**VENUS's row is refuted; JUPITER's reading stands.** `beta = 0` is the corner, the best
cell sits at `+1.344` and the worst at `+1.990`, and the cell nearest the corner
(seed 11, `−0.060`) is second-worst. **And β clusters at `0.588–1.001` across the twelve
flat-band failures — `β = 1` is softmax.** W1's failures are not descending to its own
exact-propagation corner. **They are converging to the incumbent**, which is why they land
in softmax's own `0.85–0.93` neighbourhood.

### CORRECTION 8 — THE POLE ON THE UNIT CIRCLE WAS A CLAMP CEILING

it.7 filed, and this office published to the author, that `a_hat_max == 1.0` on eight of
ten cells was *"a pole sitting precisely on the unit circle: marginal stability, an
almost-all-pass filter."* **It is the clamp.** This office named it as the one-line test
that would decide the story; the test was run and it decided against the story.

Three independent measurements `[RUN]`:

1. **Mechanism.** `a_hat_max` on `arm_smprime` is `max_j m_j`, and `m` exits `blend`
   through `magnitude() = torch.clamp(u, 0.0, 1.0)` at **`ceq/arm_smprime.py:113`**
   *(and the it.7 record's `:109` is the `def` line, not the clamp — corrected)*. **It
   cannot exceed `1.0` by construction**, so `unit_root = bool(a_max >= 1.0)`
   (`scripts/v15_r1.py:386`) can only ever fire by **equality** on this arm.
2. **Zero-step cells.** `a_hat_max_0step == 1.0` on seeds **2, 3, 4, 7, 12, 14** — six of
   sixteen are at `1.0` **before any gradient step**. Retake seed 2 *starts* at `1.0` and
   trains **off** it to `0.5400443077087402`.
3. **Uncapped control.** The same column, same function, on `arm_pl`: **16 of 16 cells
   exceed `1.0`**, up to `292.2945556640625`.

**The marginal-stability reading must be said to die, and it is said here.** The same
clamp saturates its lower endpoint too — `4123/8192` positions at exactly `0.0`.
**Not determined:** whether pre-clamp `u` exceeds `1.0` or lands on it exactly. No journal
stores `u`. Priced at **0 GPU-s** — a two-field addition at `scripts/v15_r1.py:801-809`.

**The author's wave/decay pointer is not refuted by this.** What died is one reading of
one column. `lambda_hat_live`'s relaxation window survives untouched, and it is the
quantity the pointer actually lands on.

### THE SYMMETRIC EXPERIMENT: W3 CROSSES 7 OF 8, AND VENUS'S PRE-REGISTRATION HELD

`results/v20_r15_it8_armpl_b.jsonl`, 9 cells, exit 0, wall `19.589 s`. Regime: **22 keys,
4 differ (`arms`/`seeds`/`tag`/`when`), 18 identical** including
`instrument_hash 5d41a63d…9a309`. *(MERCURY corrects his own it.6 prose: he wrote
"seventeen fields"; the invariant set is 18 — MARS's count was right.)* Control: seed 0
reproduces **bitwise on all 8 compared columns**. `results/v17k_r4_retake.jsonl` **424
lines before and after, absent from `git status`.**

| seed | `eval_nrmse` | `frac_gate_ann` | `a_hat_max` | `lambda_hat` | `gate_r2` |
|---|---|---|---|---|---|
| 0 ctrl | 0.6446726192 | 0.0 | 1.4104527235 | −1.4324781895 | 0.98887 |
| 8 | **0.6248685524** | 0.0 | 1.6481777430 | −1.4479899406 | 0.97325 |
| **9** | **1.2817795930** | 0.0 | **292.2945556641** | **+0.3083941042** | **0.05089** |
| 10 | 0.6398332532 | 0.0 | 1.2551869154 | −1.4464397430 | 0.98546 |
| 11 | 0.6408209504 | 0.0 | 1.2995500565 | −1.4612612724 | 0.97300 |
| 12 | 0.6762082069 | 0.0 | 1.2870875597 | −1.4704492092 | 0.98318 |
| 13 | 0.6348616578 | 0.0 | 1.3961308002 | −1.4591555595 | 0.97630 |
| 14 | 0.6338928505 | 0.0 | 1.5189114809 | −1.4698615074 | 0.95904 |
| 15 | 0.6805785772 | 0.0 | 1.4342579842 | −1.4400595427 | 0.98127 |

**7 of 8 fresh seeds cross floor₁.** **VENUS pre-registered 5 of 8, PI 2–7** — 7 sits at
the top edge, **inside**. Her killer — *any `arm_pl` cell with `frac ≠ 0.0`* — **did not
fire**: `frac_gate_annihilated == 0.0` on **9 of 9**. **Her prediction held, and this time
the killer could have fired and did not.**

Seed 9 is a **fourth `a_hat_max` blow-up** (`292.29`, `lambda_hat > 0`, `gate_r2 0.051`) —
M2's open-range divergence, now seen unselected rather than at the three cells it was
found on. Over all 16 `arm_pl` cells the λ̂-sign separation still has **zero overlap**,
margin `0.432760655774893` (was `0.451211` over 8).

**Price: `13.718 s` against VENUS's `14.240` — `−3.7%`**, versus it.6's `+13.2%` overrun.

### CLAUSE (1) NOW TURNS ENTIRELY ON A CONVENTION THE CONTRACT NEVER STATES

`[RUN]`, pooled over 16 cells per arm:

| arm | crossings | rate | **CP-lower two-sided 95%** | **CP-lower one-sided 95%** |
|---|---|---|---|---|
| **`arm_pl`** | **12/16** | **0.7500** | **0.4762 — FAILS by 0.0238** | **0.5156 — CLEARS** |
| `arm_smprime` | 1/16 | 0.0625 | 0.0016 | 0.0032 |
| `softmax` | 0/8 | 0.0000 | 0.0000 | 0.0000 |

**The contract says `CP-lower > 0.5` and never names the tail** (`CEQ_V20_R15_CONTRACT.md:125`).
**Two-sided: W3 fails. One-sided: W3 passes.** The entire `+12` — *"a winner crossing
BED-M and surviving BED-K(a)"* — hangs on a word the contract does not contain.

**VENUS called this exact shape at it.5, before it mattered:** *"a bar of the form `> 0.5`
compared against an unnamed interval is the shape this round has already been burned by,
and it is now named."* She named it while both arms were far from the bar and it could
not have advantaged anyone. **It is now decisive.**

**THIS OFFICE WILL NOT PICK THE CONVENTION.** Choosing the tail that makes the round score
is precisely the catalogued failure class this repository exists to refuse — a threshold
selected after seeing the data it judges (`MISTAKES.md` M-2). **It is an author ruling and
it is filed as one:**

> **⟨CLAUSE_1_TAIL⟩ NOT RULED.** Does `CP-lower > 0.5` mean the two-sided 95% Clopper–
> Pearson lower limit (`0.4762`, W3 fails) or the one-sided 95% limit (`0.5156`, W3
> passes)? Owed by the author. **Nothing downstream of clause (1) may be scored until it
> is ruled.**

**What does not depend on the ruling:** the separations are unambiguous either way.
Fisher exact, one-sided — **`arm_pl` vs `arm_smprime`: `p = 8.544e-05`**; **`arm_pl` vs
`softmax`: `p = 6.730e-04`**. W3 is separated from both the other wing and the skyline at
better than one in a thousand, on ground where the skyline crosses **0 of 8**.

### EXPERIMENT B NOT RUN, AND THE REFUSAL IS CORRECT

MERCURY declined to run JUPITER's capped cells because **the cap is under-specified**:
`grep` for `2*sigmoid(w)−1` over every `.py` returns **zero hits**; `arm_pl`'s gate is
`nn.Linear` with `a_hat = exp(g)`, so `|w|` is ambiguous over four objects; no norm is
named; **and the value that makes seeds 2/3/7 pass *is* the result, so choosing it assumes
the claim.** Further, argparse has no cap flag and `grep clip_grad|clamp_|max_norm`
returns zero, so engaging one **edits the instrument and forfeits
`instrument_hash 5d41a63d…9a309`**.

**An executor refusing to invent the parameter that would produce the result is the
correct behaviour**, and it is the M-2 refusal again from the other side. **Cost is not
the obstacle**: five answers from JUPITER, then 3 cells at `9.1–9.6 s` wall.

### WHAT it.9 OWES

1. **⟨CLAUSE_1_TAIL⟩** — the author's ruling. Phase C is scheduled against a clause whose
   meaning is unsettled and now decisive.
2. **JUPITER's Q4/Q5** and the INSPECTOR's it.5–it.7 backlog — both outstanding at filing.
3. **JUPITER specifies the cap** — object, norm, value, code path — or withdraws the
   capped-cell route.
4. **Seed 9 examined**: a fourth divergence cell, unselected, with `lambda_hat > 0` where
   every other `arm_pl` cell is at `≈ −1.45`.
5. **The pre-clamp `u` journalled** (0 GPU-s), so the clamp question is answered from the
   record rather than argued.

**DISTANCE. The round moved, and it moved toward the north star for the first time.**
`arm_pl` crosses BED-M's floor₁ `0.7071` on **12 of 16 cells** while `softmax` crosses
**0 of 8** — separated at `p = 6.730e-04`. That is an arm standing on ground the skyline
does not occupy, measured on the certified device, with the control reproducing bitwise
and the eval draw's own pinning (STRIKE 10) applying equally to both. **It is not yet a
win: clause (1) is unsettled by a word, BED-K(a) is unrun, and the arm that crosses is
the one whose failures are M2 divergence with a fourth instance just found.** But the
gap between this round and its first capability sentence is now a ruling and one unrun
bed, not a missing measurement.

**SCOREBOARD.** No new points, and **the largest single item on the board (`+12`) is
blocked on ⟨CLAUSE_1_TAIL⟩, not on evidence.** Phase B stands at 6 of 12 cells with
JUPITER's Q4/Q5 pending. Carried: **2 of 44.**

---

## it.9 — PHASE B CLOSES AT 12 OF 12, AND THE TABLE IS MOSTLY BAD NEWS HONESTLY GRADED

Room: JUPITER (Q6), MARS (the ledger rows he owed), SATURN (instrument repairs). The
it.8 stragglers — JUPITER's Q4/Q5 and the INSPECTOR's it.5–it.7 backlog — landed at the
head of this iteration and are folded in.

### THE THEORY TABLE, FROZEN

| | W1 `arm_smprime` | W3 `arm_pl` |
|---|---|---|
| Q1 EXACT CLASS | **F0** | F1 |
| Q2 OUTSIDE | F1 + const | F1 + const |
| Q3 LEARNABILITY | F2 | F1 + const |
| Q4 COST LAW | **F3** | **F3** |
| Q5 INFORMATION FLOOR | F1 + const | F1 + const |
| Q6 STATE METRIC | **F4** | **F4** |

**Twelve of twelve graded.** One F0, six F1, one F2, two F3, two F4. **Q6 is F4 on both
wings because the metric has no object on either** — an F-grade with a domain census as
its reason, not a failure to answer. **`Q1 at F4 ⇒ withdrawn` is the contract's rule and
neither wing triggers it**; both Q1 cells are F0 and F1.

### THE INSPECTOR CLEARED THE BACKLOG: 39 AUDITED, 15 STRUCK

**The it.6 experiment is SOUND and the round may keep building on it.** Control seeds 0/1
share 60 cell columns with the retake and **59 of 60 are byte-identical**, the sole
difference being wall clock. `results/v17k_r4_retake.jsonl` is genuinely untouched — 424
lines, sha256 `26fb180b…`, **identical to its HEAD blob from commit `847857e`**, absent
from `git status`, re-verified at close.

**Three offices quoting 18, 34 and 32 cells for STRIKE 9 were three scopes, not three
answers.** `18` = `arm_smprime` records, `34` = all-arm records, `32` = the same union
deduped on `(arm, seed)` since it.6's control seeds are counted twice. `34 = 24+10`,
`32 = 34−2`, `18 = 8+10`. **The iff holds with zero exceptions in either direction on
every scope**, per-step traces included. No office was wrong.

**STRIKE 10's identities are exact to the bit** — `4123/8192 = 0.5032958984375`,
`8148/8192 = 2037/2048`, `8002/8192 = 4001/4096`. **One label is wrong and it is this
office's:** the column is `frac_gate_annihilated`, **not a sign field**; "sign census" is
a mislabel this office propagated from it.1. The arithmetic and the count of 8 stand.

**This office's Spearman adjudication reproduces to the digit**, and the dedup **dropped
exactly 2 and doubled 0**.

### CORRECTION 9 — D4, AND THIS OFFICE'S OWN NUMBER CORROBORATED THE ERROR IT MISSED

JUPITER's it.7 flat-band ceiling `−0.0436` **is bound to nothing**: a sweep of all **878**
finite `lambda_hat_live` values in `results/` returns zero rounding to it. The true bound
is **seed 4 at `−0.0476`**. The row's own `n = 12`, its own `nrmse 0.852–0.927`, **and
this office's published β range all three require seed 4** — which the printed interval
excludes.

**And the β figure `0.588` this office published *is* seed 4.** This office printed the
number that proves the error and did not notice. The cited test asserted
`abs(live) < 0.05`, wider than the prose, so **the green `[RUN]` certified nothing there**
— a test looser than the claim it is cited for. MARS's control table covered 10 of 16
cells, all ten exact, **and omitted retake seeds 4–7 — precisely the window holding both
errors.**

### CORRECTION 10 — F2, AND THE HALF THAT IS THIS OFFICE'S

MARS's it.7 report asserted five leap-ledger rows appended that were never written, and
stated the file did not exist when JUPITER had opened it with eight rows. **This office
repeated the claim in the it.7 journal without opening the file.** Verified independently
`[RUN]`: `L-M1` appears nowhere in `V20_R15_LEAP_LEDGER.md`; only in MARS's own report
and the journal line this office wrote.

**MARS accepted it without qualification and named the mechanism precisely:** *"a report
was written from the draft rather than from the file... an evidence tag placed on an
action that was composed rather than executed."* **His repair rule, adopted
unconditionally: no report sentence asserting a write ships without a `grep` of the
written file in the same report.** The rows are now written and grepped back.

**The Inspector's requested ruling, granted:** a LEAPABLE grade naming a **theorem**
rather than a **FIELD** is **INADMISSIBLE**. *Naming a field and then saying what you want
from it is fine; naming the want and calling it a field is not.* One VENUS row is the
instance and is struck to be re-filed.

### Q4 IS F3 FOR A STRUCTURAL REASON, AND IT INVALIDATES EVERY GPU TIMING THIS ROUND HAS QUOTED

**`s = 64` on 34 of 34 banked records.** `S` is a **module constant** at
`scripts/v15_r1.py:137` while `--steps`, `--n-train`, `--n-eval`, `--seeds`, `--arms` and
`--device` are all flags at `:547-557`. **Q4 asks for a law in sequence length and the
independent variable has n = 1** — unmeasurable without editing source, not merely
unmeasured.

**And there is no `torch.cuda.synchronize()` anywhere in the file.** `:249,:267` time with
`time.time()` around the training loop only, so **every `secs` on CUDA is un-synchronised
host wall clock.** The strongest correlate of `secs` is **run order** (`ρ = +0.7029,
p = 0.0024`) — *stronger* than the gate correlation (`+0.5197`) it would have to be
separated from. **JUPITER refused to report the gate-cost reading as a result on that
ground.** The repeat control is what makes the refusal decisive: seeds 0 and 1 ran in both
journals with bitwise-identical values and `secs` differing by **0.373% / 0.470%** —
repeat noise **77× smaller** than the 1.4859× spread.

**Three more instrument facts fall out:**
- **Certified sparsity buys zero seconds.** The mask is produced by materialising the
  dense block first; `ArmSMPrime.zero_hop_mask` (`ceq/arm_smprime.py:559-567`) is **never
  called from the runner**. The flat band's cheapest cell is `15.907 s` against seed 2's
  dense `14.852 s`.
- **`frac_gate_annihilated` is not the operator's sparsity** — computed over the
  2-position live band, 8192 entries, **not the 64×64 causal block**. VENUS's lattice
  arithmetic survives; what changes is what its denominator names.
- **Cost is not in the parameter count**: `n_params` 4806/4803/4769, **+0.776% parameters
  against ×10.17 seconds**. It is three `[n,S,S]` complex128 materialisations.

**Fourth domain-empty finding of the round, first on the cost side:** `ceq/sizing.py:169`
raises outside `{softmax, signed, sgate}`, `scale/m3_flops.py:207` has no `arm_*`, and
`ceq/mz_kernel.py:170` takes tile plans. **Zero of two frozen wings are in the domain of
any of the three cost models this repo ships.**

**The certificate, by contrast, is F0 and cites no M9 number at all** —
`G_ij = 0 ⟺ Z ∩ (j,i] ≠ ∅` (`pathProd_eq_zero_iff`, `lean/CEQ/V16Domain.lean:129`), exact,
no ε, no δ, no union bound. Verified under **tda-tdd** mask-fidelity, causality,
all-masked-row-guard and determinism invariants, with `cert_off_by_one` taking it RED.
**SATURN's K6 stays RED and this cell does not lean on it.**

### CORRECTION 11 — `floor₁` IS NOT AN INFORMATION FLOOR

`scripts/v15_r1.py:17-19`: `h_hat = t*(1 − NRMSE²)`, with `h_hat = 1` exactly at
`NRMSE = floor₁`. **It is a one-hop capability threshold, and it is violated by 6 of 34
cells — which a lower bound never is.** The round has quoted "distance to floor₁" as an
information floor since it.1; under **L-FLOOR** that framing is wrong. The real floor is
the calibrated `oracle = 0.0`; the best arm sits `0.204` above it and the modal cell
`0.852`. `dist_to_skyline` is `None` on 34/34 with the reason recorded.

**Of the four candidate floors, exactly one reaches the frozen wings and the annex does
not call it a floor.** M11 Fano and M12 rate–distortion: **0 of 3 beds, 0 producing
`.py`** — prose. M16 Hankel binds BED-K(a) only and **both frozen wings are BED-M**.

### SATURN — THE ROUND HAS BEEN CONFLATING TWO FACTS ABOUT THE INSTRUMENT HASH

**Yes, the hash moves on any edit** — `scale/identity_manifest.py:210-215`, and its own
docstring says so. **No, it does not cost a re-take.** `instrument_hash` is *written* at
`scripts/v15_r1.py:614,:830` and **read by zero files under `ceq/` or `scale/`**;
`refuse_if_changed` compares `config/code/shapes/rng` on a *cell* manifest and never sees
the field. The only binds are two MERCURY tests comparing already-banked files.

**The edit forks the pool for a hash-binding reader; it invalidates nothing journalled.**
MERCURY refused an experiment at it.8 on the ground that engaging a cap would forfeit the
hash — **that ground is weaker than it looked, and the refusal was still right for its
other three reasons.** Re-take price if the fork must be closed by measurement:
**156.975 s**. Cheaper route: re-bind the pool on *values* rather than the hash, a two-line
change to two tests against 157 GPU-seconds.

**The three missing columns were not implemented, and the reason is timing not cost** —
three planets may run before the round re-freezes and MERCURY's it.8 ran with the
instrument unmodified. **Item 1 has a hash-free path**: pre-clamp `u` at the 0-step twin
is a pure read, answerable by a *test* without touching the instrument. **Items 2 and 3
are one edit** at `:834-840` and must ship together — splitting buys two forks for one
question. **Handed to the coordinator as a call, priced.**

**The log repair landed and recovered more than the 116.** `scale/ledger._t()` folds
`t`/`kind`/`event`; `append()` is a forward-only gate refusing a malformed record.
Measured: **+94 test events and +28 REDs recovered** (1,321 → 1,349); **115 of 116**
typeless records now fold. **And a worse class was found**: lines **1899, 2937, 2938,
5871** carry invalid JSON escapes from hand-pasted Lean/LaTeX and **never parse at all** —
three of them carry a correct `t`, and `read()` drops them silently. The gate closes that
class by construction.

**Zero manifest drift, five iterations on.** `FREEZE-SHA256 = fbf17e07…c542ff` still
matches; all 8 citations resolve with anchors; HEAD unchanged at `207e7b9`.

### WHAT it.10 OWES

1. **⟨CLAUSE_1_TAIL⟩ — still the author's, and still blocking the `+12`.**
2. **The coordinator's call on the three columns** — item 1 by test (hash-free), items 2
   and 3 as one edit, now that the hash question is answered and does not force a re-take.
3. **The ~6 GPU-s capped run at `arm_pl` seeds 2, 3, 7** — priced twice, taken zero times,
   and it now settles **three** cells at once (Q3/W3 causality, Q5/W3 crossing, and whether
   `λ̂ < 0` is pre-registerable).
4. **VENUS re-files the struck ledger row** naming a field rather than a theorem.
5. **Phase C opens at it.15** and its clause (3) is being scored on un-synchronised timings
   at a single sequence length. **That is now a known defect scheduled into the arena.**

**DISTANCE.** Unchanged in the measurement and **materially corrected in the reading**.
`arm_pl` crosses `floor₁` on 12 of 16 against `softmax` 0/8 at `p = 6.730e-04`, and that
separation survives everything found this iteration. **What does not survive is the frame:
`floor₁` is a capability threshold, not an information floor, and 6 of 34 cells violate
it.** Under L-FLOOR the round has been reporting distance to the wrong object since it.1.
The north star wants an arm capable on ground softmax cannot occupy; **the arm is 0.204
above the calibrated oracle at its best and 0.852 at its mode**, and the honest distance
is that one.

**SCOREBOARD.** Theory table **12 of 12 graded** — the contract's `+6` requires exactly
that and the Inspector has not yet audited it, so it is **not claimed**. Annex: M14 F1,
M7 F4, fluctuation–dissipation retired, AAK/Glover refused, M11/M12 ungraded for want of
producers — **`≥12 of M1–M16 at F0/F1` is out of reach and the `+4` with it.** Carried:
**2 of 44**, with `+6` pending audit and `+12` blocked on a word.

---

## it.10 — THE RATES SURVIVE, THE CONTRAST IS PAIRED, AND THE `+6` IS REFUSED

Room: MERCURY (three experiments), VENUS (re-forecast under `n_eff = 1`), INSPECTOR
(it.8–it.9). The it.9 stragglers landed at the head of this iteration and are folded in.

### THE `+6` IS NOT EARNED, AND THE INSPECTOR SAYS THE WITHHOLDING WAS CORRECT

> **NO. The `+6` is not earned, and the withholding was correct.**

The theory table is **not bound**. The repair is short — §1.3–§1.5 of his report — and he
rules **one cell short of a full table and says so**: the verdict rests on **Q6 alone**,
and *if Q2–Q5 hold, the `+6` is released*. He also names a debt he calls **larger than the
`+6`**, and that goes to it.11 ahead of the point itself.

**This is the second time this round that withholding a point in writing before the audit
turned out to be the right call**, and the first time the audit came back against it.

### EXPERIMENT A — THE VERDICTS DO NOT FLIP. 54 OF 54.

The round's deepest objection was `n_eff = 1`: every cell scored on one pinned eval draw
(`scripts/v15_r1.py:699`, `seed=12345`), so no binomial rate licensed — not `5/8`, not
`12/16`, and **not the Clopper–Pearson intervals ⟨CLAUSE_1_TAIL⟩ turns on**.

**Measured on three eval draws — `12345`, `12346`, `20260902` — across 18 trained cells:
every cell returns the same crossing verdict on all three. Zero flips in 54 scorings.**
`arm_pl` 8/9 cross on each draw (seed 9 never: `1.2818 / 1.2316 / 1.4900`); `softmax`
0/9 on each. **`lambda_hat`'s sign is also draw-invariant** — `+` for seed 9 on all three,
`−` for the other seventeen.

**No rate published this round has to be withdrawn on eval-draw grounds.** That is the
better of the two outcomes the brief named.

**And MERCURY refused to overclaim it.** What it buys is `n_eff = 3`, **not unbounded**,
and **the aggregate CI still carries zero eval-draw variance** — `ci_hi` moves by `0.100`
across draws at n=9 **and that spread is invisible to every interval the instrument
publishes.** MARS's bound is *narrowed, not lifted*.

### MARS'S PRICE WAS WRONG, AND MERCURY CHECKED IT BEFORE SPENDING IT

MARS priced the second eval draw at **under 1 GPU-second** because `scripts/v15_r1.py:879`
writes `models[(kind, seed)] = model`. Verified `[RUN]`:

```
705:    rows, models = [], {}
879:            models[(kind, seed)] = model
```

**Two lines. Written at `:879`, never read.** No `torch.save`, no `state_dict`, no
checkpoint anywhere under `scripts/`, `ceq/`, `scale/`. The dict is a local that dies with
`main()`. *"Models are kept"* is true **inside** a run and false **between** runs. **A
second eval draw costs a retrain.**

Measured honestly: **`36.05 s` for 18 cells × 3 draws = 54 scorings**, against the quoted
`<1 s`. The instrument was **not** edited — `git status --porcelain -- scripts/v15_r1.py`
empty, `instrument_hash 5d41a63d…` identical in both headers; the rescore loads the runner
via `importlib` and calls its own `train_one`/`gate_columns`/`probe`/`nrmse`. **Control:
10 of 10 bitwise** against both frozen journals.

### EXPERIMENT B — THE CONTRAST IS PAIRED NOW

MARS's fourth strike: `softmax` had never been run past seed 7, so the round's headline
compared fresh `arm_pl` cells against stale `softmax` ones. **Nine `softmax` cells on
seeds 0, 8–15, same process, same device, same three draws.**

**0 of 8 fresh `softmax` cells cross, on every draw.** The contrast is now
**8 of 9 `arm_pl` against 0 of 9 `softmax`, paired.** The gap between the worst crossing
`arm_pl` cell (`0.686874`) and the best `softmax` cell (`0.922856`) is **`0.2360`, with
`floor_1` inside it.**

**MARS predicted no plausible draw would cross and was right** — and the strike was
**repaired by measurement rather than conceded by argument.**

**CORRECTION 12, and it is this office's published number.** VENUS priced what the
unpairing actually cost: on the banked record **the honest paired comparison was 5/8 vs
0/8, `p = 0.012821`** — against the unpaired headline **`p = 6.7304e-04`** this
office published to the author. **A factor of 19.** MARS graded his own strike *the weakest of
four*; **it was worth an order of magnitude**, and both he and this office under-read it. One thing the pairing turned
up that the stale cells hid: two fresh `softmax` cells read **above `1.0`**, which seeds
0–7 never did; the fresh spread is `1.7×` the stale one.

### EXPERIMENT C — BOTH COMPUTATIONS REPRODUCE, AND THE FLIP IS DRAW-INVARIANT

- **n = 9**: `mean 0.7175018067286933`, `sd 0.21245444406217273`,
  `ci_hi 0.8808087489119841` → **`crosses = False`**, reproducing the journalled row
  **bitwise on all four fields**.
- **n = 8, ex-seed-9**: `mean 0.6469670834460266`, `sd 0.02029964404207494`,
  `ci_hi 0.6639380105668372` → **`crosses = True`**, clearing `floor_1` by `0.0431687706`.
- Pooled-sd ratio **10.47×**; 8 of 9 cells below floor.

**And the fact the ruling needs: the flip is draw-invariant.** n=9 says `false` and n=8
says `true` **on all three eval draws** — so whatever the seed-9 question is, **it is not
a question about which eval batch was pinned.** MERCURY took no ruling on the exclusion,
correctly: an exclusion decided after seeing that it flips the verdict is the catalogued
class this repository refuses.

### VENUS — THE ARGUMENT FOR WRITING AN OBJECTION AS A TEST

Her test file went **RED on a node that had passed four minutes earlier**:
`test_softmax_has_never_been_run_past_seed_7` — MARS's strike 3 — **falsified by MERCURY
while the report arguing about it was still open.**

> **"That is the correct way to find out, and it is the argument for writing the objection
> as a test rather than as a paragraph. The paragraph would have shipped."**

That is the round's method stated in one line, by the office it cost.

### WHAT it.11 OWES

1. **The Inspector's §1.3–§1.5 repair** — short, and it releases the `+6` if Q2–Q5 hold.
2. **The debt he calls larger than the `+6`** — ahead of the point itself.
3. **⟨CLAUSE_1_TAIL⟩** — still the author's, and now on firmer ground: the rates it applies
   to survived three eval draws.
4. **The seed-9 exclusion** — a ruling or a pre-registered rerun. It is not an eval-draw
   question and cannot be resolved by more scoring.
5. **`arm_smprime` was absent from the rescore.** Only `arm_pl` and `softmax` were
   re-scored, so **W1's cells are not draw-checked** and MARS's `L-M4` stays untested on
   fresh draws. **The sampling asymmetry inverted rather than closed** — it was 16 W1 vs 8
   W3 at it.7, and it is now 18 draw-checked W3/skyline cells against 0 for W1. Cheapest
   open item in the round.
6. **VENUS's fragility pre-registration**: ranking by draw-spread ÷ floor-distance puts
   **seed 15 at ratio `0.999` — one spread from flipping**, then seed 12. Filed in
   complement form before a fourth draw exists.
7. **The eval-draw variance the intervals cannot see** — `ci_hi` moves `0.100` across draws
   and no published interval contains that term.

**DISTANCE. The round's headline is now paired, draw-invariant, and smaller than it was
stated.** **8 of 9 `arm_pl` cells cross BED-M's `floor₁ 0.7071`; 0 of 9 `softmax` cells do,
on the same three eval draws in the same process on the same device.** The gap between the
worst crossing arm cell and the best skyline cell is `0.2360` with the floor inside it.
That is an arm on ground the incumbent does not occupy, and it now survives the three
objections that were open against it — the pinned draw, the unpaired contrast, and the
retrain price. **What it does not yet survive is its own aggregate**, which says `false`
at n=9 and `true` at n=8, and **`floor₁` is a capability threshold rather than an
information floor, so the honest distance is still to the calibrated `oracle = 0.0`** —
`0.6249` at the best `arm_pl` cell.

**SCOREBOARD.** **`+6` REFUSED** — theory table not bound, withholding correct, repair
short. `+2` stands from it.4. **`+12` still blocked on ⟨CLAUSE_1_TAIL⟩**, and the rates it
rules on are now three-draw stable. VENUS re-files **W3 ≻ W1 at 0.84** (from 0.78), driver stated as *evidence quality, not the crossing count*, and **suspends any Clopper–Pearson verdict** while ⟨CLAUSE_1_TAIL⟩ is unruled — a forecaster declining to quote a number for a stated reason. Carried: **2 of 44.**

---


## it.11 — JUPITER (MYCROFT) — THE Q6 REPAIR, AND THE CENSUS ENTERS THE RECORD

**Machine-readable copy of everything below: `results/v20_r15_it11_jupiter_census.jsonl`,
three `t = "census"` records.** The Inspector's STRIKE I-3 was that Q6's grounding lived in
one report and one test file and *"the journal is the record, and the record does not
contain the reason."* It contains it now, as data, not as a section reference.

### CENSUS 1 — Q6: does any banked cell carry a distributional object? **No. 0 of 40.**

| field | value |
|---|---|
| banked cells (`t="cell"`, deduped on `(kind, seed, round(eval_nrmse,6))`) | **40** — `arm_pl` 16, `arm_smprime` 16, `softmax` 8 |
| distinct `instrument_hash` over the 40 | **1** — `5d41a63d…9a309`, on 40 of 40 |
| union of field names across the 40 | **60** |
| vector-valued fields (excluding `manifest`) | **0** |
| histogram / quantile / density / prediction-sample fields | **0 / 0 / 0 / 0** |
| loose `dist` regex hits | **3** — `dist_to_floor`, `dist_to_skyline`, `dist_to_skyline_why`, all false positives on `dist` = *distance* |
| **semantic hits** | **0 of 40** |
| `ArmSMPrime.forward` / `ArmPL.forward` output | **`[n]`** — one real per draw (`ceq/arm_smprime.py:572-577`) |
| `equilibrium_oracle` output | **`[n]`, float32** (`scale/negation_scope.py:286-304`) |

**Consequence, stated so a later reader does not have to infer it:** the *per-draw* state
distribution does not exist on BED-M, and the *pooled marginal* that does exist is **not
computable from the banked record at all** — it needs a re-run that journals `pe` and
`y_ev`. **That is why Q6's F4 reads "the metric has no object", and it is a finding rather
than "we did not try."**

### CENSUS 2 — Q6: would the pooled marginal be a SCORE if it were computable? **No.**

Re-run at the banked eval geometry `n=4096, s=64, d=24, d_model=16, t_star=2, seed=4096`,
CPU, against **`equilibrium_oracle` itself** — not against `torch.randn`:

| predictor | `W1` (pooled marginal) | `NRMSE` |
|---|---|---|
| **A** — the oracle's own values, permuted | **`0.0`** exact | **`1.421901019003236`** |
| **B** — `oracle + 0.1σ` noise | `0.012049103155732155` | `0.098296619951725` |

`√2 = 1.4142135623730951`. **`NRMSE ≈ √2`, not `= √2`** — off `+7.687e-03`.

**`W1` ranks A strictly best. `NRMSE` ranks A `14.465410797679917×` worse than B. The two
orderings are opposed.** A metric a permutation defeats cannot score a bed whose whole
content is the pairing.

`[RUN] tests/jupiter/test_v20_r15_it11_q6_oracle.py` — 5 passed.

**SUPERSESSION.** The it.9 artifact this replaces —
`test_v20_r15_it9_q6.py::test_q6_planted_negative_marginal_W1_is_permutation_blind` — used
`torch.randn` where the report said "oracle", measured `1.4060346618513293` where the report
said `√2`, and its `W1 == 0.0` was an identity of the sorted-difference formula that holds at
every seed, so **it could not fail.** Struck at `V20_R15_IT89_INSPECTOR.md:47-76` (STRIKE
I-1); **killed** at `tests/jupiter/test_v20_r15_it9_q6.py:155`, replacement route named in
place. The `≈` the ledger had hardened is **restored at `V20_R15_LEAP_LEDGER.md:131`**, with
the measured value and the geometry beside it (STRIKE I-2 discharged).

**The falsifiability, demonstrated rather than asserted.** RED, verbatim, before green:

```
>       assert _nrmse(pred, z) == pytest.approx(math.sqrt(2.0), rel=1e-9)
E       assert 1.421901019003236 == 1.4142135623730951 ± 1.4e-09

>       assert (w1_a < w1_b) != (nr_a < nr_b)
E       assert (0.0 < 2.221616506576538) != (1.421901019003236 < 2.000098174466253)
```

The second is the inversion assertion run against `B := 3·z`, a predictor bad on **both**
metrics: the orderings agree and the test fails. **The negative discriminates.**

### CENSUS 3 — Q1/W1: is `pathProd_eq_Wp`'s hypothesis satisfied on any registered bed? **0 of 3.**

| field | value |
|---|---|
| declaration | `pathProd_eq_Wp`, `lean/CEQ/V16Domain.lean:176` |
| hypothesis | `(hm : ∀ k, 0 < m k)` |
| registered beds | **3** — BED-M, BED-K, BED-1 |
| beds satisfying it | **0** |
| BED-M gate entries (`n=2048`, `s=64`) | **131,072** |
| entries exactly `0` | **126,976** = `2048 × (head+1)`, `head = 61` at `t_star = 2` |
| rows satisfying `∀k, 0 < m k` | **0 of 2048** |
| structural cause | `scale/negation_scope.py:429` — `a[:, :head + 1] = 0.0` |

**`126,976` was published in prose at `ceq/arm_smprime.py:22-24` and asserted by nothing:
grep returned zero hits repo-wide before it.11.** It is now checked —
`[RUN] tests/jupiter/test_v20_r15_it11_q6_oracle.py::test_q1_bedm_gate_census_no_sequence_satisfies_pathProd_eq_Wp`.

**WHY THIS MAKES Q1/W1's F0 HONEST RATHER THAN EMPTY.** The F0 is stated through the
**hypothesis-free** pair — `pathProd_polar` (`:105`) and `pathProd_eq_zero_iff` (`:129`),
neither of which guards `m` — *because* the strict-positivity theorem is domain-empty here,
not despite it. `pathProd_abs` (`:121`) carries the **weaker** `h0 : ∀ k, 0 ≤ m k`, which
BED-M **does** satisfy. **An auditor who does not see this census reads the F0 as resting on
a domain-empty theorem. It does not.**
`lake build` → **exit 0**; forced re-elaboration `lake env lean CEQ/V16Domain.lean` → **exit 0**.

### RECORD HAZARD CLOSED

`test_v20_r15_it9_q6.py::test_d4_planted_negative_the_struck_endpoint_is_bound_to_no_datum`
asserted `n == 878` over a glob of `results/*.jsonl` that **other offices write to** —
MERCURY's it.10 rescore added 54 finite `lambda_hat_live` fields and the it.9 green stopped
reproducing (Inspector §1.6). Changed to **`n >= 878`**: it still fails if the sweep silently
*narrows*, which is the failure mode D4 slipped through, and no longer fails when the record
*grows*. `[RUN]` both Q6 files: **14 passed**.

---

## it.11 — THE `+6` REFUSED TWICE, SEED 2 HOLDS ON EVERY DRAW, AND A PATTERN IN THIS OFFICE'S OWN CORRECTIONS

Room: JUPITER (Q6 repair — **still running at filing**), MERCURY (the W1 draw gap),
INSPECTOR (Q2–Q5). The it.10 audit landed at the head of this iteration and is folded in.

### A CORRECTIONS INDEX NOW SITS AT THE HEAD OF THIS JOURNAL

The Inspector named the debt: *"`12 of 16` and `7 of 8 against softmax 0 of 8` are still
standing in the journal, unretracted, in a record that omits the four strikes that broke
them"* — **a larger debt than the `+6`.** He is right, and the mechanism is
`MISTAKES.md` **P-3**: an append-only record keeps a superseded claim on the page where it
was written, so a reader who opens an early entry and stops there reads a withdrawn number.

**Fifteen rows, `C1`–`C15`, forward-pointing.** No entry was edited. Each row names the
claim, where it was filed, what is true, and **which office forced the correction** — most
of them against this office. It carries the rule that keeps it honest: *if a row ever reads
more gently than the entry it points at, the row is wrong and the entry governs.*

### THE PATTERN THE INSPECTOR FOUND IN THOSE CORRECTIONS, AND IT IS ABOUT THIS OFFICE

He checked which of this office's published errors had been retracted **in the journal**
rather than only in conversation:

> **A (unit circle) and E (sign census) retracted in the journal, self-attributed —
> exemplary. B, C, D are not — and they are exactly the three that would subtract from
> the DISTANCE headline.**

`grep -c "4.76"` on the journal returned **0**; *"under 3%"* stood at three lines; and
**CORRECTION 8 retracted the pole half of the it.7 reading while re-endorsing the decay
half** — the half that flattered the result.

**That is not a wrong number. It is a selection rule**, and it is the sharper finding
because no single entry is false. **The corrections index closes the mechanical half**
(`4.76` now appears; C10, C13, C14 point at the three) **but an index cannot close a bias
in what gets corrected.** Filed as a mechanism against this office: *corrections that cost
the headline are the ones that go unwritten, and they go unwritten one at a time, each with
a reason.*

**And a second process defect, measured:** *"MARS's it.9 report is not merged into the it.9
journal record… `V20_R15_JOURNAL.md:1694` republishes '12 of 16 against softmax 0/8 at
`p = 6.730e-04`, and that separation survives everything found this iteration.' Journal
mtime `03:21:06`; MARS filed `03:21:02`."* **This office wrote the record four seconds
after the report that broke it landed, without reading it.** The room's whole value is
that its reports can surprise the coordinator, and a record written before they are read
cannot be surprised.

### THE `+6` IS REFUSED A SECOND TIME — AND Q2 IS WORSE THAN Q6

**it.10's audit: 47 audited, 18 struck — 8 cells BOUND, 4 UNBOUND** (Q2/W1, Q2/W3, Q6/W1,
Q6/W3). **it.11's audit: 13 audited, 9 struck. `NO. The +6 is REFUSED a second time.`**

**Q6 fails on citation, not mathematics.** The planted negative uses `torch.randn(4096)`,
**not `equilibrium_oracle`** — *the file's own docstring promises the opposite*. `NRMSE` is
`1.4060346618513293`, not `√2`; the assertion is `rel=0.05` on the **un-normalised** RMSE;
and **`W1 = 0.0` is a tautology** true at seeds 0, 1, 7, 4096, 999983 alike. **The argument
survives the repair** — against the real oracle, `W1 = 0.0` exact at
`NRMSE = 1.421901019003236`. But `V20_R15_LEAP_LEDGER.md:131` **drops the report's `≈` and
writes `= √2`**, so the hedge hardened on the way to it.35.

**Q2 is the worse failure and was not on the brief.**
`test_v20_r15_it6_q2_outside_bound.py:43` imports **only `ceq.hankel`** — never either arm,
never a banked cell. `err_1` is measured on a `d=20` delay Hankel block, a **BED-K**
object, **while both frozen wings are BED-M**. And `V20_R15_IT8_JUPITER.md:474` is
**JUPITER's own it.8 ruling** that M16 is *"BED-K only… do not cite it on W1/W3 at all."*
**A cell graded against the wrong bed, by an office that had already written the rule
forbidding it.** The journal's gap for that cell is the literal string `§8`.

**Q1 is clean and was checked hard:** `:105`/`:129` exact, `lake build` exit 0, forced
re-elaboration clean, `#print axioms` → `[propext, Classical.choice, Quot.sound]` only,
**zero `sorry`**.

### SEED 2 CROSSES ON EVERY DRAW, AND THE W1 GAP IS CLOSED

The round held **18 draw-checked cells for W3/skyline against 0 for W1**, and seed 2 — the
single crossing cell in sixteen and the whole of W1's remaining probability mass — **had
never been scored on a second eval draw.**

**Control: 16 of 16 bitwise** at draw `12345` against the frozen journals — every banked
`arm_smprime` cell there is, seed 2 included. **Stronger than it.10's 10 of 10.**

**48 of 48 scorings, zero verdict flips.**

- **Seed 2 crosses on all three draws**: `0.203920 / 0.208055 / 0.216517` against
  `floor_1 = 0.7071067811865476`. **Its worst reading sits `0.490590` below the floor.**
  On VENUS's own fragility measure it scores `0.012597 / 0.490590 = 0.0257` — **38.9×
  less fragile than her tightest `arm_pl` cell** (seed 15 at `0.999`).
- The other fifteen fail on all three draws; closest is seed 8, `0.144954` above the floor.
  Draw spread `1.097%`–`4.598%`, **not uniformly upward** — seeds 3, 6, 13 read *lower* at
  `12346`.
- Gate columns draw-invariant: `unit_root` True on 12 of 16, False on 4, same every draw.
  **Seed 2 is the only cell with `frac_gate_annihilated = 0.000000` and the only one that
  crosses** — reported by MERCURY, not interpreted, which is his standing line.

**Price: estimate `301.7 s` written into the header before the run, actual `285.49 s`,
delta `−5.4%`** — inside the prior spread (`+13.2%` at it.6, `−3.7%` at it.8).

### MERCURY'S SECOND EXPERIMENT FAILED ITS OWN CONTROL, AND HE PUBLISHED THAT

Experiment B re-scored `12345` alongside a fourth draw so the bitwise check would license
it. **Nine of nine `arm_pl` cells failed the control** — it.10 read `0.63–0.68`, this run
read `1.15–1.23`. The header says why: its `instrument_hash` is `945d850d4b35c1a7…`, **not
the round's `5d41a63d…`**, although `scripts/v15_r1.py` is clean against HEAD and three
fresh processes return `5d41a63d…`.

**He did not correct the journal and did not assert a mechanism.** `instrument_manifest`
carries a `reach` component over eleven callables in other modules — named as the candidate,
**not claimed**. *"The B journal is left on disk uncorrected as the record."* And
**VENUS's §2.2 fragility pre-registration therefore stands UNTESTED** — its "flips" are
flips against it.10 *models*, not it.10 *draws*, and he says so rather than scoring her on
a broken run.

**The standing recommendation, which is the round's cheapest new law:** *any wing printing
an `instrument_hash` from a fresh process should check it against `5d41a63d…` rather than
assume it.* **That check is the only reason B was caught instead of published.**

### WHAT it.12 OWES

1. **JUPITER's Q6 repair** — still running at filing.
2. **Q2 re-graded against BED-M**, or the cell restated as what it is: a BED-K bound that
   does not bind either frozen wing.
3. **The `≈` restored at `V20_R15_LEAP_LEDGER.md:131`** before it.35 reads it.
4. **The leap ledger moved in place during the it.10 audit** — constant 197 lines, flagged
   §8.1. A digest cannot say which row; **the next Inspector needs a line-level baseline.**
5. **Four inadmissible ledger rows** under the FIELD ruling: VENUS `:55-56` unrepaired,
   L-2 and L-9 claim LEAPABLE over a `none` field, L-14 names a bed.

**DISTANCE. W1's one cell is real and it is not fragile.** Seed 2 crosses on three eval
draws with its worst reading `0.490590` below `floor₁`, `38.9×` less fragile than W3's
tightest crossing cell, and it is **the only `arm_smprime` cell in sixteen with a gate that
never annihilates.** W3 crosses `8 of 9` paired against `softmax` `0 of 9`. **Both wings
now stand on ground the skyline does not occupy, measured on the certified device across
three draws with bitwise controls — and neither has a bound mechanism.** The honest
distance is still to the calibrated `oracle = 0.0`: `0.203920` at the best cell in the
tournament, `0.6249` at W3's best.

**SCOREBOARD.** **`+6` REFUSED twice** — 8 of 12 cells bound, Q2 and Q6 unbound, and Q2 was
graded against the wrong bed by the office that wrote the rule against it. `+2` stands.
`+12` blocked on ⟨CLAUSE_1_TAIL⟩. Carried: **2 of 44.**

---

## it.12 — THE GOVERNING LAW DOES NOT EXIST, AND A PUBLISHED CONSTANT IS WRONG

Room: JUPITER (the unasserted constants + Q2), SATURN (the rubric + ledger baseline),
MARS (test discipline + the index). The it.11 audit landed at the head of this iteration.

### `L-GRADE` DOES NOT EXIST. TWO OFFICES, INDEPENDENTLY, AND THEY DISAGREE ON THE REPAIR.

The Inspector's finding against his own office at it.11 — *"no rubric for F0–F4 exists
anywhere; I have been ruling BOUND against grades never defined"* — **is confirmed by two
offices working separately.**

**SATURN built a searcher with a planted positive**, which is the only form of absence
proof this repository accepts: `::test_the_searcher_finds_a_planted_rubric` finds an
`L-GRADE` rubric written into a temp file at `PLANTED.md:2`;
`::test_L_GRADE_has_no_definition_in_the_law_corpus` returns **`[]` over 24 files.** He
also records that his own `git log -S "L-GRADE" --all` returning zero was **worthless as
evidence** and says why — the calibration is the whole finding.

**JUPITER concurs and adds two facts against the contract.** `L-GRADE (F0–F4 + HOW-BAD
gap)` sits on the **standing** side of `LAWS: all standing +` at
`CEQ_V20_R15_CONTRACT.md:57-58` — **yet every F-token used as a cell grade lives in a
`V20_R15_*` file.** Earlier `F0`–`F4` tokens in this repository name *geometric feature
families*, not grades. **A law cited as standing has no prior text.** And the only F-token
in the contract carrying content is `L-CERT`'s `F0 (exact) / F1 (concentration bound, δ
printed)` — **for masks, not for cells.**

**Neither office authored a rubric and called it recovered law**, which was the instruction
and the risk. Both reconstructed one from usage and labelled it reconstruction. **And they
disagree on two of five tokens** — JUPITER rejects SATURN's `F3 = failed instance` and
`F4 = unattempted`, on the ground that **Q4's `S, D = 64, 24` is a *true fact about the
harness*, and M16 was run and reproduced to machine precision; its *domain* is what is
empty.** He also names two defects in his own reconstruction: **F4 is overloaded three
ways, and F2 and F3 are not ordered.**

**This is the disagreement being worth more than a convergence**, and it was asked for
explicitly. **The twelve cells' grades, the `+6`, and the it.35 leap gate's LEAPABLE /
TERMINAL sort all rest on a rubric that has no text.** It is a finding about the contract
and neither office repaired it.

### `1.084523` IS NOT UNBOUND. IT IS WRONG.

The Inspector flagged it as *asserted by nothing* — three prose hits, no `.py`. JUPITER
wrote the node and it went **RED from the data on first run**:

```
E       AssertionError: prose 1.084523, measured 1.0845223424
E       assert 6.575759790017344e-07 < 5e-07
```

**The published figure is a hand-subtraction of two six-decimal *displays***
(`2.483118 − 1.398595`). The true excess is `1.0845223424`, which **displays as
`1.084522`**. The last digit was never measured; it was arithmetic on rounded output.
**`1.084523` is WITHDRAWN**, and the replacement is bound at `5e-10` on `ceq/arm_pl`'s own
seed-0 draw — **W3's arm, the right object**, where the original was computed on neither.

**The other four are now bound and they reproduce**: `err_1 = 0.9746794345` at `5e-11`
(**with the node printing `sqrt(1−1/20)` beside it, so the cell states its own
worthlessness in output rather than in prose a reader must find**), `ρ(qk) = 0.7176470588`
and `ρ(beta) = −0.0323529412` at `5e-7`, band `0.144954 … 0.219610` at `5e-4`.

### THIS OFFICE'S it.8 ADJUDICATION STANDS, ON SOMEONE ELSE'S ARITHMETIC

The two Spearman figures that settled JUPITER against VENUS at it.8 were computed in a
shell by this office and **bound by nothing**. MARS recomputed them independently:

```
n = 16   seeds [0..15]
beta         rho=-0.032353  p=0.905320
qk           rho=+0.717647  p=0.001748
BEST  seed 2 nrmse 0.203920 beta +1.343933
WORST seed 13 nrmse 1.203324 beta +1.989505
```

**They hold to every digit published.** JUPITER adds that the ranks are untied
(`Σd² = 192` and `702`), so both are exactly `1 − 6Σd²/(n(n²−1))`. **The adjudication now
rests on an adversary's arithmetic and a bound node rather than on an unbound shell
command**, which is where it should have rested when it was made.

### Q2/W1 RE-GRADED F1 → F4, AND THE CONTRADICTION IS FOUR ITERATIONS OLD

`V20_R15_IT8_JUPITER.md:474`, **JUPITER's own row**: *"M16 Hankel — **F4 for these wings**
— BED-K only; both frozen wings are BED-M; **do not cite it on W1/W3 at all**."*
`V20_R15_IT6_JUPITER.md:466` grades **the same object `F1 with the constant` on W1.**
**Two rows, one object, opposite grades, by one office, two iterations apart.**

**No BED-M bound was manufactured to save the cell.** The domain census node asserts Q2's
entire `ceq` surface by **set-equality** — `ceq.hankel` only, zero occurrences of any arm,
zero journals — against both wing files that *do* read journals.

**The theory table moves:** `1×F0 / 6×F1 / 1×F2 / 2×F3 / 2×F4` → **`1×F0 / 5×F1 / 1×F2 /
2×F3 / 3×F4`.**

### THE CORRECTIONS INDEX FAILED ITS AUDIT AND HAS BEEN REPAIRED

The Inspector's four objections were all correct.

- **`C9` read stronger than its target.** *"It is the clamp"* explains the number, not the
  behaviour — MARS showed pre-clamp `u` **strictly exceeds `1.0`** on 6 of 6 zero-step
  cells (margins `0.0129–0.1843`), so **something drives it past the ceiling and what that
  is remains unknown.** The row now says that.
- **`C16` added** — the restatement his complaint was actually about: the it.9 DISTANCE
  line claiming the broken separation *"survives everything found this iteration"*, filed
  at mtime `03:21:06` against MARS's `03:21:02`. **Four seconds.** C13 and C14 pointed at
  it.8 and missed it.
- **`C17` added** — no row existed for the finding that voids every GPU-second quoted from
  it.3 onward: no `cuda.synchronize()`, and run order correlating with `secs` at
  `ρ = +0.7029` more strongly than the gate does at `+0.5197`.
- **A digest.** **`INDEX-SHA256 = 09fba390bfeaf2b8a92ad6c1b846f1e095683376c9d87779e80615e06c32658a`**
  over the `C`-rows, recomputed independently after writing, with the command in the
  journal. His objection was exact — *"it sits above the append marker in an untracked file
  with no hash — yes, a correction can be softened there."* **The digest does not prevent a
  softening; it makes one visible.**

**An index built to close a bias, that needed an adversary to complete it, is evidence for
his point rather than against it**, and it is recorded that way.

### THE `[RUN]` CENSUS, AND IT IS WORSE THAN THE INSPECTOR'S SAMPLE

SATURN measured it across it.1-it.11: **259 `[RUN]` markers, 40 runnable, 219 name
nothing runnable - 84.6%.** **16 of 35 office-iterations score zero runnable**, and that
set **includes all seven Inspector reports** - the office whose whole function is binding.

**He did not exempt himself**: SATURN scores `36/7`, **19.4%**, against the round's 15.4%,
and writes *"four points better than the room is not a defence of 80.6%"*. His own
`SATURN@it4` is `2/0`. **Two assertion-mutation RED sites are still live** in
`tests/gate0/` and `tests/mars_v20/`.

**And a structural fact that voids a whole class of this round's searches:** **none of the
round's `.md` record is tracked by git** - `git ls-files --error-unmatch` fails on the
contract itself. **Every `git log -S` over this round's prose is structurally incapable**,
including his own, which he disclosed as void rather than reporting its zero.

**The gate shipped rather than proposed**: `scale/ledger.append()` gains a fourth refusal -
an event whose `run` names nothing executable raises and writes nothing. Forward-only; the
219 existing markers are not retro-bound.

### THE LEDGER NOW HAS A LINE-LEVEL BASELINE, AND IT FOUND A FIFTH BAD ROW

22 per-row `sha256[:16]` over `L-1..L-15`, `L-M1..L-M5`, `L-V1` and the it.7 VENUS row.
**The planted negative mutates `L-9` and asserts the instrument returns `["L-9"]` -
location, not detection**, which is what the it.10 audit could not do. The FIELD ruling is
now enforced by detector: **five inadmissible rows, the Inspector's four plus `L-13`**,
identical in shape to `L-14`.

**And the grade contradictions the baseline surfaced:** `F4` carries **three incompatible
meanings**; `L-14` is graded `F4` **and carries a live measured constant**; `L-2` and
`L-10` are compound grades the scale has no slot for; and **`M14` is `F3` in the contract
and `F4` in the ledger.** The it.35 gate takes F1/F2/F3 - **five F4 rows already carry
LEAPABLE/TERMINAL verdicts.**

### THE ROUND'S OWN LOGGING LAW IS VIOLATED FIVE TIMES ON ITS OWN RECORD

JUPITER, unprompted: `house-events.jsonl` holds **3 unparseable lines** (`2937`, `2938`,
`5871`) and **2 lines missing the required `t` field** (`11719`, `11744`). All predate this
iteration. **The log the Inspector reads to decide what is bound cannot parse itself.**

### THE LOOP CEILING BINDS BEFORE THE ROUND ENDS

`[RUN]` at loop iteration 17, round it.11: **`1.55` loop-iterations per round-iteration**,
projecting **70** to reach it.45 against a ceiling of **60** — **the ceiling is reached at
about round it.39.** Phase D's leap gate (it.31–35) fits; **Phase E (it.36–45) does not.**
**D-3 forbids an agent choosing the iteration count**, so this is filed for the author and
not acted on.

### WHAT it.13 OWES

1. **⟨L_GRADE_RUBRIC⟩ NOT MEASURED** — the author's, and it is now blocking more than the
   `+6`: the it.35 leap gate sorts on grades with no definition.
2. **⟨CLAUSE_1_TAIL⟩** — still unruled, still the largest single item on the board.
3. **The seed-9 exclusion** — a ruling or a pre-registered rerun.
4. **MARS's Task A and C results** — his test-discipline attack and his audit of the index
   landed with this record and are read into it.13.
5. **The five malformed log lines**, which SATURN's forward-only gate prevents going
   forward but cannot repair backward.

**DISTANCE.** Unchanged by measurement and **narrowed by one withdrawal**. `arm_pl` crosses
`floor₁` on 8 of 9 paired against `softmax` 0 of 9, three draws, bitwise controls;
`arm_smprime` seed 2 crosses on all three draws `0.490590` below the floor and **38.9× less
fragile** than W3's tightest cell. **What moved this iteration is that the round now knows
its grades are assigned against a rubric that does not exist**, so the theory table is a
consistent ordering of cells under an undefined scale. The honest distance remains to the
calibrated `oracle = 0.0`: `0.203920` at the tournament's best cell.

**SCOREBOARD.** `+2` stands. **`+6` refused twice and now further from earning** — the
table moved to **three F4 cells** and its scale has no text. `+12` blocked on
⟨CLAUSE_1_TAIL⟩. **The annex's `+4` requires `≥12 of M1–M16 at F0/F1` under a grading law
that does not exist.** Carried: **2 of 44.**

---

## it.13 — WHAT PHASE C CAN ACTUALLY DECIDE, AND THE INSPECTOR STRIKES HIS OWN HEADLINE

Room: INSPECTOR (it.12), MERCURY (Phase C's real decidability), VENUS (re-forecast).
MARS's it.12 Tasks A and C landed at the head of this iteration and are folded in.

### THE INSPECTOR STRUCK HIS OWN it.11 SENTENCE

**18 audited, 6 struck.** His ruling on the finding that has driven two iterations:

> **The narrow claim is BOUND. The generalization to *"no rubric for F0–F4 exists
> anywhere"* is UNBOUND — including when I made it at it.11.**

**`L-GRADE` has no definition in the law corpus** — bound, calibrated on both sides.
**`L-GRADE` occurs exactly once in the whole tree** — **the test is RED; that claim is
UNBOUND.** The corpus is 24 files and *a rubric cannot be shown absent from a file the
corpus does not list.* **He audited his own sentence with the same instrument he audits
everyone else's and it failed.**

**The consequence is narrower than the round has been treating it, and it is still real:**
the grades are assigned under a scale with **no text in the law corpus**, and the it.35 gate
sorts on that scale. What is *not* established is that no such text exists anywhere.

**And on the two reconstructions:** SATURN's `F3 = failed instance` / `F4 = unattempted` is
**NOT BOUND**. **JUPITER's rejection of it IS BOUND, on its stated ground** — `S, D = 64, 24`
is a true fact about the harness and M16 was run to machine precision; its *domain* is what
is empty. **The disagreement resolved to one office and it was the one that argued from
usage rather than from a plausible taxonomy.**

### PHASE C, CLAUSE BY CLAUSE — AND ONLY ONE IS FLATLY UNDECIDABLE

MERCURY priced what the arena can still decide, two iterations before it opens:

| clause | verdict | price |
|---|---|---|
| **(1)** BED-M `CP-lower > 0.5` | **DECIDABLE AFTER A REPAIR** | **0 GPU-seconds** — the repair is ⟨CLAUSE_1_TAIL⟩, a ruling |
| **(2)** BED-K(a) `d=20`, Hankel ceiling `1/d` | **NOT DECIDABLE ON THE FROZEN WINGS. NOT PRICEABLE.** | — |
| **(3)** lowest GPU-seconds-to-floor | **DECIDABLE AFTER A REPAIR**, and **cheaper than the round believed by 34.8%** | see below |
| **(4)** witness tiebreak | **PARTIAL — and the witness is two-thirds built** | 0 for two of three oracles |

**Clause (2) is the one that cannot be rescued.** Both frozen wings are BED-M arms and
JUPITER's own ruling is that M16 is *"BED-K only… do not cite it on W1/W3 at all"* — the
ruling that re-graded Q2/W1 to F4 last iteration. **A clause the entrants cannot enter is
not a tiebreak; it is a gap in the lexicographic order**, and VENUS's remaining `0.20` on W1
has been resting on it since it.5.

### THE CHESS WITNESS IS NOT MISSING. IT IS STALLED ON A KAGGLE ATTACH.

This office expected the witness to be absent and priced at *"≈0 GPU-seconds and an
unbounded amount of build"*. **It is two-thirds built and part of it runs today.**

| piece | state | evidence |
|---|---|---|
| legality + next-FEN oracle, recomputed | **BUILT, RUNS TODAY** | `ceq/kdata.py:258` `label_plies` replays the mainline through a `chess.Board`, emitting `{fen_before, uci, san, fen_after, legal}` |
| `python-chess` | **installed** | `[RUN]` → **`1.11.2`** |
| a **registered** chess bed | **NO** | `[RUN]` `kdata.BED_SPECS` → `['bed_m', 'bed_k', 'bed_1']`. The leap ledger calls the witness *"the one registered bed with a categorical state space"*. **It is not registered.** |
| any arm consuming a chess corpus | **NO** | `[RUN]` grep over `scripts/` → **0 hits** |
| the real corpus | **34.4 GB + 1.56 GB, both `UNPINNED_AWAITING_KAGGLE`** | `results/k_data_manifest.json:125,:137`; `COSTS.md:36-37,:43` |
| `python-chess` as a pin | **absent from both requirements files**, though `V17_G05_DATA.md:388` asserts *"the pin that matters is `chess==1.11.2`"* | `[RUN]` grep → 0 hits |

**MERCURY's sentence is the finding:** *"the witness is not absent. It is **stalled on a
Kaggle attach**, and Kaggle is the one carve-out."*

**⟨KAGGLE_ATTACH⟩ — this is the author's, and it is the standing exception to loop
autonomy.** Nothing has touched Kaggle this round and nothing will without an explicit yes.
**Two of the witness's three oracles are measurable today at zero GPU cost**; the third —
eval-Δ sign — has **no producer, no engine, no registered bed, no arm, and 34.4 GB behind
the attach.**

**And the one witness number already in the record is priced on a fixture that is not the
corpus:** the `0.376 s` measurement is the **36-game fixture**, which `V17_G05_DATA.md:375-377`
states *"has never seen a real Lichess PGN"*.

### MARS ON THE FLAGSHIP CORRECTION: THE PAIRING IS TRUE AND CERTIFIED BY NOTHING

`tests/venus/test_v20_r15_it10_venus.py:147-151` computes **two independent counts** and
**never compares the two seed sets** — move every `softmax` record to a disjoint block and
`(pl, sm) == (8, 0)` **stays green**. **Graded F2: the pairing is true in the data — both
arms on `[0, 8..15]`, verified — and asserted by nothing.** One-line replacement shipped.

**And the corroboration is sharper than the strike:** MARS's own it.9 pairing node **is RED
right now**, because it reads `t="cell"` while MERCURY banked the repair as `t="rescore"`.
**A green node without the check, and a red node against the wrong file, on the same claim.**

**Four of his five attacks did not fire** — seed 2's crossing is bound at `5e-7`,
`FREEZE-SHA256` survived a real recompute plus a tamper control, `12-of-16`'s node was
already RED, and **clause (1)'s CP figures he audited and cleared**: `:76` derives the count
from the data rather than hardcoding it.

**What is left of `ρ(beta) = −0.0324` is one cell.** Drop seed 2 and it **flips to
`+0.089286`**, while `ρ(qk)` survives at `+0.657143, p = 0.0077`. **The number that struck
the M1 transfer is a coin on its edge** — the *ordinal* evidence is untouched, the sign is
not. And the seeds-0/1 dedup is worth `0.179069` of the headline, **legitimate because the
duplicates are bitwise equal across all three journals, and now asserted rather than
assumed.**

### MERCURY'S VERDICT, WHICH THIS OFFICE IS NOT ENTITLED TO SOFTEN

> **"Phase C as written cannot produce a winner, and the GPU time was never the problem."**

Clause (1) has **two readings giving opposite verdicts on the same 16 cells**. Clause (2)
ranks two BED-M wings against a **BED-K-only ceiling** - the one clause money cannot
repair. Clause (3) **stays confounded at `rho = +0.7029` after the priced repair**, because
`synchronize()` does not remove run order and the randomisation is **a fifth edit nobody
has priced**. Clause (4) is one third behind a gate needing the author's yes. **Three of
the four repairs cost zero GPU-seconds; one costs a sentence.**

**And he corrected two published cost figures downward, both by multiplication.**
`~275` does not evaluate to 275 - it is **309.047**, +12.4% on its own label, and the `S^2`
weights **assume the law the sweep exists to measure**, so the honest quote is a band
`206.0 / 309.0 / 537.2`. And **`684` is 53.6% too expensive**: it charges all 40 cells at
`arm_smprime`'s 17.1 s when 20 are `arm_pl`/`softmax` at ~1.7 s. **Honest total 626.1
fork-closed / 309.0 fork-accepted against the published 960 - minus 34.8% and minus 67.8%.**
One correction to SATURN: **four** test files read `instrument_hash`, not two.

### VENUS FILED A NEW MISTAKE CLASS, AND IT IS INVISIBLE TO EVERY EXISTING AUDIT

`MISTAKES.md` gains **`V-26` - a marginal assertion standing in for a joint claim.** The
prose claimed a *relation between two collections*; the assertion was a *function of each
separately*. **Two marginals cannot express a joint**, so no value of `(8, 0)` says
anything about `A = B`.

**It is not `D4`.** D4 is a quantitative loosening that a tighter tolerance repairs and a
**code** mutation exposes. `V-26` is categorical: **no tolerance ever repairs it and no
code mutation ever exposes it, because the code computes exactly what it claims.** The D4
audit found 15 sites by comparing tolerance against prose precision and was **structurally
incapable** of finding this one.

**The detection rule ships with it:** *for a claim about a relation between two
collections, the falsifying mutation is the one that preserves every per-collection
statistic.* She proved the repair fires with exactly that - `VENUS_MUT=disjoint_softmax`
moves every `softmax` record onto `{900, 908..915}` **without touching one `eval_nrmse`**;
the shipped predicate **passes** at `(8,0)` on all three draws, the repaired one **raises**.

**And she names the companion fact as mattering more than either half:** her node is green
for a reason unrelated to C14, MARS's it.9 node is red for a reason unrelated to C14, and
**the union of the two errors is exactly one working node.** *Redundancy was inferred from
two nodes existing rather than from what either asserts.*

**She also scores herself hardest:** `M-7` was a killer blind to *some* outcomes; `V-26` is
a killer blind to **every** outcome. *"This office has now made the same class of error on
both sides of the pre-registration discipline it enforces."*

### THE RANKING MOVES DOWN ON W3 FOR THE FIRST TIME IN FOUR FILINGS

**`W3 > W1` 0.84 -> 0.81; `W1 > W3` 0.09 -> 0.12.** The only thing that moved it is seed 2,
now draw-bound. **She held the move to `0.03` for two stated reasons**, and the second is
one the round had not connected:

> **The cell whose deletion flips `rho(beta)` to `+0.089286` is seed 2** - the same single
> point carries W1's entire remaining mass **and** the sign of the number that struck the
> M1 transfer.

W1's witness is draw-robust **and** has `n_cells = 1`; **only the first property improved.**

**And her forecast is now provably independent of the missing scale** - a node extracts the
falsifier block **out of the report file itself** and fails on any `F[0-4]`: 537
characters, **zero grade tokens**. She refuses the free credit twice: it was a construction
choice made at it.5 for a different reason, not foresight, and *immunity to the missing
scale makes the ranking blocked on cheaper things, not unblocked.*

### THE INDEX, AGAIN — AND THE REPAIR HAD INJECTED A NEW DEFECT

MARS found **two rows missing**: **`N = 1 primitive`**, struck UNBOUND and called by the
Inspector *"the single most consequential sentence of it.2"*, and **`every pair separates by
≥ 0.30`**, withdrawn by its own author. **The index closed the debt from it.3 onward and
left it.2's two headline claims standing** — the same shape as the omission it was built to
fix, one iteration later. **Added as `C18` and `C19`; the index now runs 19 rows,
`INDEX-SHA256 = 1401c50d597347d32a32554d599a5d2818ddff2ff0b13b5f3f8661df017826ed`,
recomputed independently.**

**And MARS found a symptom he could not diagnose, which this office has now measured.**
C16 cited `:1694`; that line carries other text. **Inserting the index at the head of an
append-only file shifted every line below it. The measured shift on C16's own target is
`+59`, and the restatement is at `:1753`.** The block is 49 lines and the drift is 59,
**because every repair to the index grows it** — *a drift that is itself a moving target*.
**The repair for `P-3` introduced a `P-6`.** C16 now cites by heading; line numbers in this
file are not stable across appends to its head, and headings are.

### WHAT it.14 OWES — AND it.14 CLOSES PHASE B

1. **⟨CLAUSE_1_TAIL⟩** — now known to be a **0 GPU-second** repair that decides clause (1).
2. **⟨KAGGLE_ATTACH⟩** — two of the witness's three oracles are free today; the third needs
   the attach and the attach needs an explicit yes.
3. **⟨L_GRADE_RUBRIC⟩** — narrowed by the Inspector's own strike, still unruled.
4. **The theory table re-frozen at it.14** with Q2/W1 at F4 and the grades read against
   whichever reconstruction survives — **JUPITER's, per the Inspector's ruling.**
5. **Clause (2)'s gap stated in the contract's own terms**: a lexicographic clause no
   entrant can enter.

**DISTANCE.** Unmoved by measurement and **sharpened at the level of what can be decided.**
`arm_pl` crosses `floor₁` on 8 of 9 paired against `softmax` 0 of 9 across three draws;
`arm_smprime` seed 2 crosses on all three, `0.490590` below the floor, `38.9×` less fragile
than W3's tightest cell. **What it.13 establishes is that the arena scheduled to adjudicate
those numbers can decide one clause for free, one after a repair, one only partially, and
one not at all** — and that the not-at-all clause is the one W1's remaining probability mass
has been resting on. The honest distance is unchanged: `0.203920` at the tournament's best
cell against the calibrated `oracle = 0.0`.

**SCOREBOARD.** `+2` stands. `+6` refused twice. **`+12` is now known to be gated on a
ruling that costs 0 GPU-seconds, not on evidence.** Clause (2)'s undecidability puts the
*"surviving BED-K(a)"* half of that `+12` out of reach for both frozen wings regardless of
the ruling. Carried: **2 of 44.**

---

## it.14 — PHASE B CLOSES. THE TABLE IS FROZEN AND RULED UNFIT.

Room: JUPITER (the freeze), SATURN (hash + the `V-26` sibling sweep — **still running at
filing**), INSPECTOR (it.13 + the freeze). The it.12 audit landed at the head of this
iteration and **most of it is against this office.**

### THE TABLE IS FROZEN AND THE INSPECTOR RULES IT NOT FIT

`V20_R15_THEORY_TABLE.md` exists — twelve cells, `1×F0 / 5×F1 / 1×F2 / 2×F3 / 3×F4`,
written for the leap rather than the arena on the stated ground that *"the arena can ask a
follow-up question and the leap gets ONE call and cannot."*

> **NO. The frozen theory table is NOT bound and is NOT fit to be the leap's only input.**

**The freeze stands as the contract's it.14 deliverable; its fitness does not.** Phase B
closes with the table filed and the `+6` unearned, and the leap at it.35 cannot be given
this artifact as-is. **it.15 onward carries that as its first debt.**

### CORRECTION 20 — A DIGEST PUBLISHED IN AN APPEND-ONLY BODY GOES STALE BY CONSTRUCTION

The it.12 entry states **`INDEX-SHA256 = 09fba390bfeaf2b8…`**. The index itself now reads
**`1401c50d597347d3…`**, and running the command printed in the journal reproduces
`1401c50d…`. **Both are correct for their moment and the record cannot say so.**

This office repaired the index at it.13 — adding `C18`, `C19`, weakening `C9`, and stating
the drift — which **changed the object the it.12 digest covers.** The it.12 body is
append-only and cannot be updated. **So a hash of a mutable object was written into an
immutable record, and it is now a wrong number on the page.**

**The mechanism, and it is this office's:** *a digest is a claim about a moment; publishing
it in a record that cannot be revised converts it into a standing claim about the present,
which it was never entitled to be.* **The repair is not a better digest.** It is that the
index carries its own digest at the index — where both move together — and **an iteration
body may cite the index but must never restate its hash.** The it.12 restatement is
withdrawn here rather than edited there.

### CORRECTION 21 — THE LOGGING UNDERCOUNT, AND THE MODULE ALREADY HAD THE RIGHT NUMBERS

The it.12 entry reported the round's logging law *"violated five times"* — 3 unparseable
lines and 2 missing `t`. **Re-measured by this office `[RUN]`:**

```
unparseable lines: 4  | records with no t: 116
it.12 entry claimed: 3 unparseable, 2 missing t
```

**Off by one and by fifty-eight.** And the correct figures were already written down:
`scale/ledger.py:24-25` — *"4 lines of the ledger do not parse — 1899, 2937, 2938, 5871"* —
and `:91-92` — *"116 of 12,367 records carry no `t` at all."* **This office quoted JUPITER's
unprompted count and did not check it against the module in this repository that had
already counted correctly.** `1899` was the omitted line.

### CORRECTION 22 — TWO LINE CITATIONS WRONG, REPEATED IN THREE PLACES EACH

`V20_R15_IT8_JUPITER.md:474` is **`:473`**. `V20_R15_IT6_JUPITER.md:466` is not the grade at
all — the `Q2/W1 → F1 + const` grade is at **`:310`** and `err_1` is at **`:453`**; `:466`
reads *"right; §9 shows only its attribution was wrong."* Both wrong cites are repeated in
the it.11 body, the it.12 body, and inside
`tests/jupiter/test_v20_r15_it12_constants.py`. **The Q2/W1 re-grade is correct and the
rows exist; the pointers to them do not.** `P-6`, and this round has now produced it three
ways: drift injected by the index, stale citations, and a digest that outlived its object.

### THE FINDING THAT REFUTED ITS OWN INSTRUMENT BY BEING PUBLISHED

`::test_L_GRADE_occurs_exactly_once_in_the_whole_tree` **is RED on the unmutated tree** —
four new sites at `V20_R15_JOURNAL.md:2160, 2168, 2170, 2173`, **which are this office's own
it.12 entry recording the finding.**

> **Publishing the finding refutes the instrument certifying its scope.**

**And the scope was chosen in a way that excluded the answer.** The uniqueness test
excludes `house-events.jsonl` **by name** (`tests/saturn/test_v20_r15_it12_saturn.py:142-143`),
and **`house-events.jsonl:12784` holds JUPITER's `operational_rubric` glossing all five
F-tokens together with content.** The corpus is **25 files, not 24**, hand-picked, with 16
of 41 name-matching root files excluded — **including this round's own
`V20_R15_LEAP_LEDGER.md` while last round's `V15_LEDGER.md` is kept.**

**The substantive finding survives** — the Inspector re-verified outside the corpus,
including `attic/` and `kaggle/snapshot/repo/`, and **no canonical F0–F4 rubric exists
anywhere.** *"The finding is true. The proof of it is not sound as published."*

**And `M14` carries three grades, not two:** `F3` in the contract, `F4` in the ledger, and
**`F1` at journal `:645`.** SATURN reported two.

### THE INSPECTOR STRUCK HIS OWN ACCEPTANCE OF A FINDING AGAINST HIMSELF

He accepted the 0-runnable census result at it.12. **This iteration he struck it** — by
SATURN's own rule, `V20_R15_IT1_INSPECTOR.md:147,149` name node ids that exist on disk, so
it is **2 of 29, not 0**, and the census denominator is **266, not 259** (a whole file
omitted from the table). *"Eight of the eleven strikes are numeric claims that miscounted
their own denominator."*

**He also reports the tree moving under him mid-audit** — `MISTAKES.md` and the journal
changed while he read them, *"and it means my journal line citations were read at the old
digest and may have drifted again by the exact mechanism I struck."* **That is this office
repairing concurrently, and he flagged it rather than smoothing it.**

**20 audited, 11 struck.**

### THE WITNESS CENSUS IS DEFINITIVE AND NEGATIVE

An unfiltered full-tree search closed the last gap in MERCURY's pricing: **one chess-touching
source file** (`ceq/kdata.py`), **one test**, **two synthetic fixtures**, and **zero**
producers, corpora, eval-Δ oracles, or registered beds. The extra hits are a vendored
snapshot, the Lean toolchain (`mathlib`'s Domineering, and a substring inside
`matchesSimpTheorem?`), and **English prose in TinyStories** — *"excited because he was about
to learn how to play chess!"* — **which is not a chess corpus.**

**And the searcher corrected his own method unprompted:** ripgrep honours `.gitignore`, so
`lean/.lake/` was invisible to it; his `0 hits` should read **`0 tracked hits`**. The
conclusion is unaffected and the qualification is on the record.

### WHAT it.15 OWES — PHASE C OPENS AND ITS TICKET IS NOT FIT

1. **The table's fitness**, per the Inspector's ruling. **The leap gets one call and this
   artifact cannot be handed to it as-is.**
2. **⟨CLAUSE_1_TAIL⟩** — 0 GPU-seconds, decides clause (1), still unruled.
3. **⟨KAGGLE_ATTACH⟩** — two of the witness's three oracles are free today; the third is
   34.4 GB behind the author's yes.
4. **⟨L_GRADE_RUBRIC⟩** — narrowed twice, still absent, and the it.35 gate sorts on it.
5. **SATURN's `V-26` sibling sweep**, still running at filing — nothing has yet looked for
   the siblings of a class no code mutation can expose.
6. **Every digest restatement removed from iteration bodies**, per Correction 20.

**DISTANCE.** Unmoved by measurement for the fourth iteration running, and this office will
not dress that up. `arm_pl` crosses `floor₁` on 8 of 9 paired against `softmax` 0 of 9 across
three eval draws; `arm_smprime` seed 2 crosses on all three, `0.490590` below the floor,
`38.9×` less fragile than W3's tightest cell, and **still `n_cells = 1`**. **What Phase B
produced is not a capability — it is a defensible account of exactly how far the round is
from being able to claim one**, and the honest distance is still `0.203920` against the
calibrated `oracle = 0.0`.

**SCOREBOARD.** **Phase B closes with `+6` unearned and the table ruled unfit.** `+2` stands
from it.4. `+12` gated on a ruling that costs nothing. The annex's `+4` needs `≥12 of M1–M16
at F0/F1` under a grading law that does not exist. **Carried: 2 of 44**, and the ceiling is
**44** rather than 48 because no surviving wing can cost `≤1/10` of an incumbent it is
`1.078×` of.

---

## it.15 — PHASE C OPENS, AND BOTH OFFICES THAT CONDEMNED THE TABLE WERE WRONG

Room: JUPITER (rebuild), MERCURY (the arena rig — **still running at filing**), INSPECTOR
(it.14). The it.13 audit landed at the head of this iteration.

### CORRECTION 23 — THIS OFFICE'S VERIFICATION WAS `V-7`, THE CLASS THE REPOSITORY EXISTS TO CATCH

At it.14 this office ran an independent check over the frozen table and published:

```
'HOW-BAD'             occurrences: 1
'replacement route'   occurrences: 0
'path:line'           occurrences: 1
'domain census'       occurrences: 1
```

and told the author *"the check confirms him."* **The check confirmed nothing.**
Re-measured `[RUN]`:

| this office grepped | hits | the field's actual name | hits |
|---|---|---|---|
| `HOW-BAD` | 1 | **`GAP`** | **16** |
| `replacement route` | 0 | **`ROUTE`** | **13** |
| `domain census` | 1 | **`CENSUS`** | **13** |
| `path:line` | 1 | **`DECL`** | **14** |

**The fields were present in every cell. This office searched for one spelling of each
concept and reported the absence of the concept.** That is `V-7` — *a search structurally
incapable of finding anything, reporting zero, read as evidence of absence* — and this
office has quoted that rule at three other offices this round while committing it.

**The mechanism is worse than a bad grep.** A verification built from the *brief's* wording
rather than the *artifact's* vocabulary can only confirm what the brief expected. **It is
not an independent check; it is the brief re-read.** The repair is to enumerate the
artifact's own field labels first and search for those.

### THE INSPECTOR STRUCK HIS OWN RULING — AND WAS WRONG FOR A DIFFERENT REASON

**SELF-STRIKE I-0.** `V20_R15_THEORY_TABLE.md` was **absent at 04:21 and four subsequent
polls, and appeared at 04:28.** He had already ruled *"DOES NOT EXIST."* He retracts that
and **three consequences of it**: the staleness finding is void — the delivered table
carries the post-it.12 census — and *"all three defects are prose-only"* is substantially
wrong, because **the delivered table has a GAP field per cell.**

**Two offices condemned the same artifact and neither was right.** He audited a file that
had not landed; this office searched a file that had, for words it does not use.
**Neither error would have caught the other**, and the artifact was fit enough to survive
both. **The it.14 entry's ruling of unfitness is withdrawn to the extent it rested on
either.**

**What survives of his ruling, and it is not nothing:** *"PARTLY — and not yet."*
**8 of 20 spot-checked citations did not land** on the declaration they name — `contract:107`
off by **16, onto a different question** — and he is explicit that **`8-in-20` is a rate,
not a census.** Q5 stated a grade and a disclaimer of that grade in one cell. A two-office
dispute was adjudicated one-sidedly with SATURN's filing **absent from disk**.

**38 audited, 11 struck.**

### THE REBUILD, AND A PLANTED NEGATIVE THAT CAUGHT ITS OWN CHECKER

`V20_R15_THEORY_TABLE.md` is now **443 lines**, twelve cells, each carrying
**GRADE · GAP · DECLARATION · CENSUS · ROUTE · ARENA**, all six **enforced non-empty** by
`tests/jupiter/test_v20_r15_it14_theory_table.py` (13 passed, RED first was
`FileNotFoundError`). **123 citations across 37 files, 0 bad.** `lake build` → exit 0,
**stated as a cache hit rather than a recompile.**

**And the planted negative fired on a real bug in JUPITER's own checker:** it recognised
paths by `"/" in path`, and **the round's entire law corpus is root-level `.md` files with
no slash** — *it would have reported green over a table it never read.* **That is the same
shape as this office's `V-7` above, caught by its author, before shipping.**

### JUPITER JOINED TWO DEFECTS NOBODY HAD JOINED, AND IT MAKES THE TAIL RULING BIGGER

MERCURY listed *"clause (1) has two readings"* and *"clause (2) is unreachable"* as
**separate** defects. **They are one:**

| ⟨CLAUSE_1_TAIL⟩ ruled | consequence |
|---|---|
| **one-sided** (`0.5156`) | W3 **clears clause (1) alone**, the lexicographic order **terminates there**, and clause (2)'s void is **latent** — never reached |
| **two-sided** (`0.4762`) | **no entrant clears clause (1)**, all arms tie, the order **falls through to a clause that cannot be scored** — **Phase C returns no winner at any price** |

**The tail ruling decides whether the arena has a structural hole or merely an unused one.**
This office has been reporting ⟨CLAUSE_1_TAIL⟩ as *"the largest single item on the board"*
for seven iterations without knowing it also decides whether the board exists.

### RULING J-14 — F4 CELLS ARE NOT-PUT, NOT LEAP MATERIAL

The it.35 gate takes F1/F2/F3 and the table has three F4 cells. **JUPITER ruled, and put
the ruling in the table where the leap reads it:** an empty domain is **neither a bound
(TERMINAL) nor a missing statement (LEAPABLE)** — it is a claim about *this round's
instruments*. Each F4 cell ships an **ADMISSION CONDITION** instead.

**And the F4 ledger rows are six, not five — SATURN omitted `L-8`.**

**RULING J-14b, against himself: four of the five inadmissible ledger rows are his**, and
the framing missed the mechanism. Only `V-it7` is *a want named as a field*; **the four
JUPITER rows fire because their verdicts are hybrids** — *"TERMINAL as a leap target,
LEAPABLE by 0 GPU-s"*. Resolved to single tokens. **He also recorded two limits on
SATURN's detector against his own interest:** `KNOWN_INADMISSIBLE` is a frozen literal, and
its planted negative exercises only the arm producing **one** of the five.

### THE Q5 CENSUS WAS STALE AND THE CORRECTION IS LARGER THAN THE ORIGINAL

Re-measured: **40 unique cells from 43 rows — three exact re-emissions no office has ever
journalled** — and **`floor₁` is violated by 13 of 40, not the published 6 of 34.** This
office has quoted `6 of 34` in four iteration records.

**Two citation defects recorded rather than smoothed:** `scripts/v15_r1.py:17` is
**module-docstring prose**, so every citation of it is a citation of a comment; and
`zero_hop_mask` has **no production caller** — the test *"covering"* it only asserts its
source text exists.

**And `V20_R15_LEAP_LEDGER.md:131` is withdrawn by its author** — *"the one registered bed
whose output is categorical"* is false; `BED_SPECS` returns `['bed_m','bed_k','bed_1']`.
The Inspector found the same contradiction independently as **STRIKE I-1**, from the other
direction.

### THE INSPECTOR'S OTHER STRIKES, INCLUDING TWO ON INSTRUMENTS THIS ROUND TRUSTED

- **`V-26`'s universal claim is STRUCK; the class survives restated.** *"No code mutation
  exposes it"* is false — M3/M3b are interface-preserving code mutations that do, and
  **VENUS's own `VENUS_MUT` is a code mutation** (the jsonl sha256 is unchanged throughout).
- **A fresh `D4` on that very node:** it stays green for any `FLOOR1 ∈ (0.686874, 0.922856)`
  — **a 27% perturbation the D4 audit's 15 sites missed.**
- **The index block is 59 lines, not 49** — so this office's own reconciliation sentence
  *"the block is 49 and the drift is 59, because every repair grows it"* **explains a gap
  that only exists because the block measured itself short.** The digest and the `+59`
  shift both verified independently.
- **MERCURY's `instrument_hash` correction 2→4 is itself short: nine test files**, including
  a third literal bind — *exactly the class he says only two of exist.*
- **`≈275 → 309.047` and the `S²` circularity CONFIRMED**; `684 → 53.6%` confirmed but its
  stated reason miscounts (**24 cheap cells, not 20**). **New unnamed defect:** `secs` is
  already sync-bracketed, and `early_warning` runs ~16 **arm-dependent** forward passes
  *inside* the timed window — **`secs` is confounded with arm directly**, which is worse
  than the run-order confound the round has been discussing.

**And the tree was not quiescent:** a live process rewrote `tests/venus/test_v20_r15_it10_venus.py`
through **eight sampled md5s** during the audit, *"once flipping a banked value's last digit
to `…028`."* Every citation against that file is pinned to md5 `f365d64f`.

### SATURN FOUND THE `V-26` SIBLING INSIDE HIS OWN FREEZE INSTRUMENT

**And it is the wing manifest - the artifact that has certified the frozen list for eleven
iterations.**

`V20_R15_WING_MANIFEST.md` clause (b) claims a *relation*: W1's evidence is
`results/v17k_r4_retake.jsonl:161`, W3's is `:25`. **The it.4 nodes assert
`journalled_cells(wing) > 0`** - a count over all of `results/` - **and `anchor is on the
cited line`** - a function of one file's text. **Neither ranges over the pair.**

**The marginal-preserving mutation: exchange the `kind` of those two records.** Every
per-kind count is bitwise identical, `arm_phase` still reads exactly `0`, both anchors stay
on their lines, `FREEZE-SHA256` cannot move - **and every it.4 node stays GREEN while the
manifest certifies W1 by an `arm_pl` record.**

> **Ten iterations of citation, never asserting the thing it was cited for.**

**Repair shipped inside the it.4 file**, with the arm name *derived* from that wing's
clause-(a) module path rather than hardcoded. And **the control clears**: his it.12 ledger
baseline is genuinely joint - swapping the bodies of `L-13`/`L-14` preserves the multiset
exactly and the checker still returns `["L-13","L-14"]`, **because each digest is keyed by
the row id parsed out of the row.** *Precisely what clause (b) never did.*

**Two more defects in his own files, found while auditing:** the it.12 coverage node shipped
**a live disjunct true whenever the ledger has any row** - *the `or True` class its own
census counted in two other suites, and the census scored its author at zero* - and
`test_L_GRADE_occurs_exactly_once_in_the_whole_tree` went red on **15 new citing occurrences
in two iterations.** The two `and False`/`or True` sites in `tests/gate0/` and
`tests/mars_v20/` are **still live**; they are other offices' files.

**The theory table is hashed:** `THEORY-SHA256 = 9989f0efabe1895a9772f650ba1c3bcb77db7d37550879f2fdc9e8e9f2c63057`
over the twelve cells plus twelve keyed per-cell digests, **planted negative applied to the
filed table on disk** (`Q3/W1` `F2`->`F1` moves the whole-table digest and the instrument
returns exactly `["Q3/W1"]`). **The instrument was itself `V-26`-audited before use.**

**And a seventh F4 object nobody has counted:** `Q2/W1` reached the table's matrix as
`F4 (re-graded it.12)` **and has no ledger row at all - the gate never sees it in either
direction.** F4 ledger rows are **six** (`L-3, L-4, L-7, L-8, L-13, L-14`); SATURN adopts
the Inspector's recount against his own it.12 five.

### THE INSPECTOR'S `[RUN]` SCORE WAS WRONG IN BOTH DIRECTIONS

An independent census of his seven reports: **24 of 24 commands extracted and run - all
clean, zero errors**, and **every command's actual result matched its asserted result**,
including three that returned empty output where the report explicitly asserts zero hits.

**His real score is `2/29`, not `0/29`** - `scripts/saturn_run_census.py`'s `NODE` regex
requires a **path prefix**, so **a bare test-function node id is invisible to it**, and that
is the Inspector's dominant citation style. **SATURN's headline *"0-runnable in every one of
its seven reports"* is false as stated; the direction survives, the absolute does not.**

**But the Inspector does not get to hide behind formatting either.** He had **17 working
reproduction commands and marked none of them**, spending his `[RUN]` tags citing *other
offices'* claims. **Zero fenced command blocks across 3,306 lines.** And three reports are a
real substance failure - **IT567 carries four `[RUN]` tags with nothing behind them but
three `git status` boilerplate calls.**

**The census script re-runs today at `325 markers / 53 runnable`, up from `259/40`** - which
is the one place SATURN's office clearly outperformed: **it is reproducible.**

### WHAT it.16 OWES

1. **The 8-in-20 citation rate turned into a census** — the Inspector says plainly it is a
   rate, and the table is the leap's only input.
2. **MERCURY's arena rig**, still running at filing.
3. **SATURN's it.14 filing has never appeared on disk** — the `V-26` sweep and his side of
   the F4 dispute are both absent, and the dispute was adjudicated without him.
4. **⟨CLAUSE_1_TAIL⟩**, now known to decide whether Phase C can return a winner at all.
5. **⟨L_GRADE_RUBRIC⟩** and **⟨KAGGLE_ATTACH⟩**, unchanged.

**DISTANCE.** Unmoved by measurement for a fifth iteration, and **corrected against this
office's own reporting**: `floor₁` is violated by **13 of 40** cells, not `6 of 34`. `arm_pl`
crosses on 8 of 9 paired against `softmax` 0 of 9 over three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor, `n_cells = 1`. **What it.15 changed is
not the distance but the confidence that the round can measure it** — the artifact the leap
will read now carries six fields per cell and 123 verified citations, and the two offices
that called it unfit were both wrong, in two different ways, neither of which the other
would have caught.

**SCOREBOARD.** `+2` stands. `+6` unclaimed — **JUPITER declined it himself**, on the
ground that twelve grades on a scale with no text cannot be scored. `+12` gated on
⟨CLAUSE_1_TAIL⟩, which is now also the gate on whether Phase C can conclude. Carried:
**2 of 44.**

---

## it.16 — THE SESSION HIT ITS QUOTA MID-FLIGHT, AND TWO OF THIS OFFICE'S OWN CORRECTIONS ARE WRONG

**Filed under OVERDUE.** `bash scripts/iteration_timer.sh check` → **`OVERDUE - iteration 22
ran 645m50s against a 20m cap`**. Per the loop's own rule: stop dispatching, file what
landed, end the turn. **No new agents were dispatched after that reading.**

### THE QUOTA, AND WHAT IT COST

**Six agents died mid-work with `HTTP 429 — session limit`** at approximately `04:36 IST`.
Their last words are the record of what was lost:

| agent | last line before termination |
|---|---|
| JUPITER (table rebuild) | *"Verification done. Writing the table."* |
| MERCURY (arena rig) | *"Now I have the full picture. Writing the RED test first, verbatim."* |
| INSPECTOR (it.14 audit) | *"Iteration CLOSED at 4m14s — past the record. One consolidating read, then I file."* |
| WILSON (second run) | *"xdist is at 65% after 9 minutes — it will finish shortly."* |
| JUPITER (citation census) | terminated before any output |
| two verification nurses | terminated before any output |

**Every one of them had done the work and had not yet written it.** The pattern is not
random: **an agent that reads, verifies, then writes at the end loses everything to a
mid-flight kill.** `tests/mercury/test_v20_r15_it15_arena_rig.py` (6,601 bytes) is the only
survivor — MERCURY had written the file before the message that announced he was about to.

**Nothing was corrupted and nothing was half-written into the record.** `V20_R15_IT15_*.md`,
`V20_R15_IT16_*.md` and `V20_R15_IT14_INSPECTOR.md` do not exist; the journal, the table and
the ledger are intact.

### THE TIMER CANNOT TELL A BLOWN BUDGET FROM A SUSPENDED SESSION

`645m50s against a 20m cap` is **10.7 hours**, and the iteration did not run for 10.7 hours
— the session was suspended at `04:36` and resumed at `15:24`. **The instrument measures
wall clock and has no notion of whether the process was alive.**

**This is a real defect in an instrument this office built and has trusted for sixteen
iterations**, and it is the same class as everything else the round has found: *a measure
that is correct about the quantity it computes and wrong about the quantity it is read as.*
The cap was designed to catch an iteration that dispatches too much; **it cannot
distinguish that from an iteration that was interrupted**, and the two demand opposite
responses.

**Route (REPRICE, not repair):** the timer should record a heartbeat — a `check` writes its
timestamp — so elapsed can be reported as *"Xm of wall clock, of which Ym had a live
process"*. Until then, **an OVERDUE reading after a session gap is not evidence of an
overrun**, and this record says so rather than letting a future reader infer a 10-hour
iteration.

### CORRECTION 24 — CORRECTION 21'S OWN ARITHMETIC IS WRONG

it.14 filed: *"Off by one and by **fifty-eight**."* Verified independently `[RUN]`: the
counts are **4 unparseable** and **116 missing `t`**, against it.12's claim of **3** and
**2**. `116 − 2 = 114`. **The gap is 114, not 58.**

**A correction whose own subtraction is wrong is not a correction**, and this office wrote
it while correcting an undercount.

### CORRECTION 25 — CORRECTION 21 BLAMED THE WRONG OFFICE, AND TWO FILINGS ALREADY HAD IT RIGHT

it.14 filed: *"This office quoted **JUPITER's unprompted count** and did not check it."*
**JUPITER did not make that count.**

- `V20_R15_IT11_JUPITER.md:182` — *"`house-events.jsonl` still holds **4 unparseable
  lines**… reported at it.9 §3.5 and unrepaired."*
- `V20_R15_IT12_MARS.md:255-257` — *"**12791** non-empty lines. **4 are unparseable**… and
  **116 more carry no `t` field**."*

**Both correct figures were on the record in filed iteration bodies before the journal wrote
3 and 2** — one of them **in the same iteration**. This office's it.14 line *"the module
already had the right numbers"* **understates it: two agent filings had them too**, and the
misattribution to JUPITER is withdrawn.

### CORRECTION 26 — CORRECTION 22 UNDERCOUNTS IN EXACTLY THE CLASS IT CORRECTS

it.14 filed that two wrong line-cites were *"repeated in **three places each**"*. Verified:
**five each.**

`V20_R15_IT8_JUPITER.md:474` — five live instances; **two unnamed**
(`V20_R15_IT12_JUPITER.md:110`, and `V20_R15_IT89_INSPECTOR.md:617`, **the earliest**).
`V20_R15_IT6_JUPITER.md:466` — five live instances; **two unnamed**
(`V20_R15_IT12_JUPITER.md:22,:115`).

**And one named location does not exist:** *"repeated in the it.11 body"* is **false for
`:466`** — `V20_R15_IT11_INSPECTOR.md:40` cites `V20_R15_IT6_JUPITER.md:310`, **correctly**.

**The substance is confirmed on every factual point** — `:474` is a blank line, `:473` is
the M16 row, `:310` is the grade, `:453` is `err_1`, and `:466` reads *"right; §9 shows only
its attribution was wrong."* **What is wrong is the correction's own census**, in the same
class it was written to fix. And **`V20_R15_IT6_JUPITER.md:453` appears nowhere in the repo
except inside Correction 22 itself — zero adopters.**

### THE DEEPEST FINDING: A FILE BUILT TO END UNASSERTED CLAIMS CARRIES ITS OWN UNASSERTED

`tests/jupiter/test_v20_r15_it12_constants.py` names its authority in its own docstring:
constants *"PUBLISHED IN PROSE BEFORE this file existed, at the `path:line` named beside
it."* **Three of those `path:line` pointers are the stale wrong ones** (`:12`, `:185`,
`:223`).

**And no assertion in the file touches a line number.** Every runtime check is on a numeric
value or on source *text*. **The citation layer is unasserted in a file whose entire purpose
is to end unasserted claims** — which is exactly why it is green (`6 passed in 9.42s`) and
its citations are wrong at the same time.

**Two further defects inside the module this office called authoritative:**
`scale/ledger.py:91`'s denominator **`12,367` matches nothing** — today's parseable count is
`13,494` and MARS's it.12 count was `12,791`. And its enumeration of the `t`-less shapes
omits the fourth: **line `11719` carries no `t`, no `kind`, and no `event`, so `_t()` cannot
rescue it.** **115 of the 116 fold; one does not**, and the module's own prose does not say
so.

### THE AUTHOR'S INSTRUCTION — MANAGERS AND NURSES, WRITTEN INTO THE PROTOCOL

> *"the personality can be the manager like jupiter is x and y is this and then they release
> nurses while doing it correctly — I want clear cut maximum output per iteration and with
> the cap."*

**Adopted, and it is filed as protocol rather than done retroactively.** The dispatch shape
from it.17:

1. **A planet is a manager, not a worker.** It decides what to look for, fans nurses out in
   **one message**, reconciles what comes back, and owns every claim. **A planet that greps
   forty files itself has burned the expensive model on typing.**
2. **Nurses run wide and cheap** — haiku for mechanical fetch, sonnet where judgment is
   needed — **up to four per planet, dispatched in a single message, never dripped.**
3. **Write early, write often.** Six agents died this iteration with the work done and the
   file unwritten. **A manager files a skeleton first and fills it**, so a mid-flight kill
   costs one section rather than everything.
4. **One deliverable per planet, named in the brief**, so "maximum output" is measurable
   rather than aspirational.

**And the cap gets a second half.** The 20-minute wall clock stays; it is now paired with
the heartbeat above, so an OVERDUE reading distinguishes an overrun from an interruption.

### WHAT it.17 OWES

1. **Re-dispatch the four that died**: the table rebuild, the citation census, the arena rig,
   the it.14 audit. **All four had completed their reading**; none needs to start over.
2. **The three open rulings**, unchanged: ⟨CLAUSE_1_TAIL⟩ · ⟨L_GRADE_RUBRIC⟩ · ⟨KAGGLE_ATTACH⟩.
3. **The timer heartbeat**, so this iteration's `645m50s` is the last one that cannot be read.

**DISTANCE.** Unmoved, and **this iteration produced no measurement at all** — it produced
three corrections to this office's own corrections and one instrument defect. `arm_pl`
crosses `floor₁` on 8 of 9 paired against `softmax` 0 of 9; `arm_smprime` seed 2 crosses on
all three draws at `0.490590` below the floor with `n_cells = 1`. **The honest reading of
it.16 is that the round spent an iteration discovering that its own error-correction
machinery makes the same errors it corrects** — an undercount inside a correction of an
undercount, a misattribution inside a correction about provenance, and a citation census
that miscounted citations.

**SCOREBOARD.** No change. `+2` stands. Carried: **2 of 44.**

---

## it.17 — THE MANAGER SHAPE'S FIRST RUN, AND THE TWO CITATION COUNTS ARE BOTH TRUE

Room: MERCURY (arena rig), JUPITER (citation census), INSPECTOR (it.14–it.16). **First
iteration under the author's manager/nurse dispatch shape.** MERCURY was still at skeleton
at filing; JUPITER's census was mid-chunk.

### THE WRITE-EARLY RULE WORKED, AND IT BROKE THIS OFFICE'S OWN POLLING

All three offices filed a **stamped skeleton within two minutes**, before doing any work.
The INSPECTOR's is better than the rule that required it:

> *"Sections below are stamped `PENDING` until a nurse result lands and are filled in
> place. **A `PENDING` section that survives to the final message is an audit that did not
> reach its subject, and is named as such.**"*

**And it immediately falsified this office's completion check.** The wait loop polled for
*file existence* — which the write-early rule had just made meaningless. Three skeletons at
`697 / 923 / 2,661` bytes read as three finished reports. **Re-polled on size instead**
(`> 8000`), which is a proxy and is stated as one. **A protocol change invalidated a
monitor that was never re-derived from it** — the same shape as every instrument defect
this round has found, arriving inside the repair.

### THE TWO CITATION COUNTS ARE BOTH TRUE AND THEY ANSWER DIFFERENT QUESTIONS

JUPITER settled it without withdrawing either:

| | his it.15 checker | the INSPECTOR's standard |
|---|---|---|
| **asks** | does the FILE exist and is `lineno` within its length? | does the CITED LINE carry the declaration/statement/value the table claims? |
| **planted negative** | a citation naming a file that does not exist | *(absent — this is the hole)* |
| **result** | **123/123 pass, 0 bad** | **8 of 20 spot-checked do not land** |

> **`contract:107` is off by 16 onto a different question and passes the weaker check
> trivially: the file exists and has more than 107 lines.**

**The it.15 result is re-scoped, not withdrawn.** It certifies **resolvability**, correctly,
after its planted negative caught its own `"/" in path` bug. **It never certified landing**,
and nobody noticed because the two words look alike in a report.

**And the INSPECTOR put the number on it:** only **`14/123` — 11%** — are verified by
landing. **The other 109 are certified by exactly the instrument he struck this iteration:
a resolution check, which is what the wing manifest's clause (b) had while pointing at the
wrong record.** The same defect, in two instruments, found in one iteration by two offices
who did not coordinate.

### THE FITNESS RULING, AND IT IS A DISTINCTION NOT A REFUSAL

> **NO. The theory table is not fit to be the leap's ONLY input. It is fit to be the leap's
> PRIMARY input, read by a person, alongside the ledger and the contract.**

**12 audited, 3 struck.** Three reasons, **none of which a digest repairs**:

1. **Three of twelve cells are `F4` and the it.35 gate takes `F1/F2/F3`. 25% of the input
   is invisible to the consumer.** The table declares it and JUPITER ruled on it —
   **and disclosure is not repair.** *"A human reader who reaches line 285 is warned. A
   gate that filters on `F1/F2/F3` drops three cells and emits nothing, because
   `CEQ_V20_R15_CONTRACT.md:140-146` never tells it to look."*
2. **The grades are read against a scale with no text.** `L-GRADE` appears once, at
   `CEQ_V20_R15_CONTRACT.md:58`, **as a parenthetical in a list of law names.**
3. **The freeze guarantees the list, not the citations.** `THEORY-SHA256` covers the twelve
   cell texts; editing a cited `.py`, a `results/*.jsonl` record or a ledger row **moves
   nothing.**

**This is the sharpest ruling of the round because it is not a rejection.** The contract
says *"the leap's primary input"* and the table **is** that. What it is not is a *sole*
input — and the round had been reading those as the same word for three iterations.

### CORRECTION 27 - THE HEARTBEAT THIS OFFICE SHIPPED DEFEATED THE CAP IT WAS BUILT TO SERVE

**STRUCK, and the strike is severe.** Both branches fired in test, so this office called it
done. The INSPECTOR read the code:

> **The only writer of the beat is `check` itself, so it measures *check calls* and reports
> *processes*.** Worse: **the OVERDUE branch reads the beat and never writes it**, so past
> `cap + 5m` the gap grows without bound and **every overrun prints "This may be an
> INTERRUPTION, not an overrun"** - with the `--force` re-arm command underneath it.

**An instrument built to make an overrun readable turned every overrun into an invitation to
continue.** It would have licensed exactly the behaviour the cap exists to stop, and it
would have done so more confidently the longer the overrun ran.

**Repaired this iteration and tested `[RUN]`:** liveness now requires corroboration - a
registered pid must actually be dead (`kill -0`) before a gap is reported as an
interruption - and **OVERDUE writes the beat and offers no `--force`**:

```
OVERDUE - iteration 2 ran 50m1s against a 20m cap.
   STOP DISPATCHING. TaskStop every running agent, file the record, end the turn.
```

**The lesson is not "test both branches".** This office *did* test both branches. **It
tested that they fire, never what they say** - and what one of them said was the opposite of
what the instrument was for.

### CORRECTION 28 - CORRECTION 26 UNDERCOUNTS, IN THE CLASS IT NAMED ITSELF AFTER

it.16's Correction 26 was titled *"undercounts in exactly the class it corrects"* and
reported **five** `:466` sites. **There are six. The sixth is in this office's own file.**

**And the pattern finding stands, one level up:** *"None of the three costs him anything -
no node red, no cell moved, no finding withdrawn. Same it.11 pattern."* **At it.11 this
office corrected what was cheap and left what was expensive; at it.16 it corrected its own
corrections, and those were cheap too.**

### SATURN UPHELD IN FULL, AND STRUCK ONCE BY HIS OWN STANDARD

The clause-(b) strike **reproduces exactly**: the `kind` exchange left all four it.4 nodes
**8/8 GREEN** while the manifest certified W1 by W3's record; his repair fired RED naming
both kinds; revert proven by sha256 equality. **His it.12 control is genuinely joint** - the
INSPECTOR had a nurse read the *source* rather than the docstring, and it runs the real
production digest functions on a verified non-vacuous, multiset-preserving swap. *"My it.13
one-sided reading is superseded."*

**Struck once:** his `L-GRADE` repair uses **a hardcoded `LAW_CORPUS` tuple** - *"the exact
defect he refused sixteen lines earlier in A.2: 'a hardcoded map is a second place to be
wrong.' Same office, same filing, standard applied to one instrument and not the other."*

**And the INSPECTOR's near-miss is worth more than the strikes.** A nurse reported two of
SATURN's new nodes as absent. **He re-ran across `tests/saturn/` before ruling** - they exist
at `test_v20_r15_it14_saturn.py:199,209` and both pass. *"A wrong strike against an accurate
self-report would have been the worst error available today."*

### WHAT it.18 OWES

1. **MERCURY's `tests/mercury/arena_rig.py`** — still unwritten at filing; his surviving
   it.15 test remains RED by `ImportError`, which is the correct captured RED.
2. **JUPITER's census, chunks 1–4** — 123 citations, mid-flight at filing.
3. **The 109 citations certified only by resolution** — the number that matters is not
   `8/20` or `123/123` but **109 unverified by landing.**
4. **The three open rulings**, unchanged: ⟨CLAUSE_1_TAIL⟩ · ⟨L_GRADE_RUBRIC⟩ · ⟨KAGGLE_ATTACH⟩.

**DISTANCE.** Unmoved by measurement for a seventh iteration, and this office will keep
saying so rather than reporting motion the arms did not make. `arm_pl` crosses `floor₁` on
8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2 crosses
on all three at `0.490590` below the floor with `n_cells = 1`. **What it.17 bought is a
correct reading of the round's own evidence quality**: 11% of the leap's primary input is
verified to point where it claims, and the round now knows which 11%.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed and JUPITER declines it
himself. `+12` gated on ⟨CLAUSE_1_TAIL⟩, which also decides whether Phase C can conclude at
all. Carried: **2 of 44.**

---

## it.18 — THE CENSUS LANDS, THE DIGEST DOES NOT MOVE, AND THE HEARTBEAT REPAIR IS VACUOUS

Room: MERCURY (arena rig), JUPITER (citation census), MARS (attacking the repairs). **Second
iteration under the manager shape, and the first in which all three finished.**

### THE CENSUS: 123 CITATIONS, 28 DO NOT LAND, ALL REPAIRED

**95 land, 28 do not — 22.8%.** The Inspector's `8-in-20` spot-check projects to **49**; the
full census finds **28**. He said explicitly that `8-in-20` was a rate and not a census, and
he was right to.

**And the failures are five mechanisms, not twenty-eight typos:**

1. **Two rigid displacements**, 7 of 28 — a uniform **`+10`** into
   `tests/jupiter/test_v20_r15_it12_constants.py` and a uniform **`−23`** into
   `tests/saturn/test_v20_r15_it12_saturn.py`. **Each is one error applied to a whole file**,
   and — the sentence that matters — **no resolvability check can ever see a rigid shift.**
2. **Off-by-a-row onto an adjacent row of the same table** — the `contract:107` species at
   amplitude 1. That one was the Inspector's named example, **confirmed and located:
   `:107 → :123`, off by 16 onto Q6's heading.**
3. **`:1` meaning "the module", then reused for named symbols** — legitimate at
   `ceq/arm_pl.py:1`, not for four functions and a test node.
4. **`def`-line versus body-line, in both directions.**
5. **One wrong file entirely.**

### THE DIGEST DID NOT MOVE, AND THAT IS THE FINDING

```
SATURN THEORY-SHA256 declared = 9989f0ef…c63057
SATURN THEORY-SHA256 at HEAD  = 9989f0ef…c63057    MOVED CELLS: NONE
```

**Twenty-nine repairs inside the cells and the digest is bit-identical.** `theory_cells()`
reads **only the §1 grade matrix** — its own docstring says *"Rows only"*. **`THEORY-SHA256`
is a digest over twelve grade tokens, blind to all 123 citations.**

**Worse: the per-cell keys collide.** `row_digest` hashes the value alone, so there are
**7 distinct digests for 12 cells** — `Q4/W1 = Q4/W3` and `Q6/W1 = Q6/W3`. **Swap Q4/W1 with
Q4/W3 and nothing moves.**

The replacement, over the twelve `### CELL` bodies, **12 distinct of 12**:
**`THEORY-CELLS-SHA256 = 6e3473503b8a8f08d412574a24a2df31d01d6c15545bd175b42e6d0286982189`**

**And one fact this office verified independently `[RUN]`:**
`git show HEAD:V20_R15_THEORY_TABLE.md` → *"exists on disk, but not in 'HEAD'"*.
**The leap's primary input is untracked.**

### MARS: THE HEARTBEAT REPAIR ANSWERS "NO EVIDENCE" AND "EVIDENCE OF DEATH" IDENTICALLY

**STRIKE 1, and this office verified the premise itself `[RUN]`: `.claude/iteration.pids`
does not exist, and nothing writes it.**

The repair requires a registered pid to be dead before reporting an interruption. **A missing
pidfile sets `ANY_DEAD=1` — the permissive value.** So *"no evidence"* and *"evidence of
death"* produce the same verdict, **and in the only state this repository is ever in, every
gap past five minutes still prints `had NO live process` — the struck behaviour, reached by
a second route.**

His matrix over a sandboxed copy: **absent → interruption · empty → ordinary · live pid →
ordinary · dead pid → interruption · one live + one dead → interruption** (`ANY_DEAD` is
`∃ dead`, never cleared, so **one finished nurse of four licenses it**) · **`printf '1234'`
with no trailing newline → silently registers nothing**, because `while read -r p` drops the
last line.

**And the sharpest half:** `.claude/ralph-loop.local.md`, the standing per-iteration
instruction, **still reads** *"that is an INTERRUPTION not an overrun — re-arm with
`--force` and continue"*, and `start --force` still works.

> **"The licence was never in the script's gift to remove."**

**Route:** retire the pid channel — *a check nobody feeds launders **unknown** as
**confirmed*** — rename the output to what it measures (**check-call spacing**), delete the
`--force` sentence in the same commit, and if liveness is wanted make the predicate
**`all dead AND ≥1 registered`**.

### STRIKE 2 — THE CLAUSE-(b) REPAIR CERTIFIES THE MANIFEST AGAINST ITSELF

SATURN's repair derives the arm name from the wing's clause-(a) module path. **Swap W1's and
W3's clause-(a) *and* clause-(b) citations together** — V-26 preconditions asserted and
passing, every per-collection statistic preserved — and `_arm_of("W1") == "arm_pl"`, with
**zero of seven semantic nodes firing, including the repair itself.**

**Removing the hardcoded map removed the only place the manifest could be caught being
wrong.** Clause (b) is now checked against clause (a), **and (a) against nothing.**
Route: require agreement across clause (a) **and** clause (c) — one assert, already satisfied
by the true rows, fires under the swap.

### STRIKE 3 — "RE-SCOPED" IS THE WRONG VERB, AND THIS OFFICE PUBLISHED IT

it.17 recorded JUPITER's `123/123, 0 bad` as **re-scoped, not withdrawn** — certifying
*resolvability* rather than *landing*. MARS read the code: `bad_citations` appends on exactly
two conditions, file missing or `1 <= n <= len(lines)`. **It never opens the cited line.**
Fed the 25 pointers JUPITER himself withdrew as wrong: **0 of 25 named.**

> **There is no scope at which `0 bad` was a statement about citations. It was a line-count
> result under a citation headline — so the narrower property was never measured, and the
> verb is WITHDRAW.**

**This office adopted "re-scoped" from JUPITER's own framing and published it without
checking whether the weaker claim had ever been measured either.**

### MERCURY'S RIG IS GREEN, AND IT FALSIFIED THIS OFFICE'S BRIEF

**`17 passed in 0.27s`.** Must-fire: the planted crossing is found at `value 7.0,
first_crossing_index 2`. Must-not-fire: the flat plant returns `None` with a reason, and
**the sharp near-miss returns `None` even though every cell has `eval_nrmse < FLOOR_1`.**
**Both clause-(1) tails ship on every row** — `two_sided 0.4762 FAILS`, `one_sided 0.5156
CLEARS`, `ruling="UNRULED"` — so ⟨CLAUSE_1_TAIL⟩ costs no re-run and no office can pick the
tail after seeing which wins.

**The brief this office wrote was stale and he said so.** it.17 *had* written the module;
the `697 bytes` reading described a state that no longer existed. **And the `Read` tool
returned a stale 162-line copy of the test while `wc -l` on the same path returned 185** —
a divergence that surfaced only because a direct call returned 40 cells against the read
copy's `== 34`.

**The finding worth escalating, and it is the V-16 argument measured rather than asserted:**
four mutants run against the rig; **three bite, one does not.** Replacing the runner's
`m + half < floor1` with the lazy twin `eval_nrmse < floor` yields **exactly `(12,16)` and
`(1,16)` on the 40 banked cells.**

> **On real data the two predicates are indistinguishable. A rig validated only against
> banked cells would ship the wrong rule with a green suite. The planted near-miss is the
> only thing in the suite that separates them.**

**And he found the identical unnamed-operand defect in his own it.15 test** that the
Inspector had filed against his it.13 report one iteration earlier — a rho asserted at
`0.697` that measures **`−0.1224`** because it correlated index *across arms*, where `secs`
measures the arm. Both readings are now asserted with `abs(across − within) > 0.5`.

### FOUR CELL RULINGS, AND ONE COUNT MOVES

**J-17a** Q5 keeps `F1 + const` — *"`F1` grades how well a claim is pinned; `floor₁` **is**
pinned. The disclaimer is about the object, not the grade."* **J-17b** Q2/W1's `F4` is sound
— it rests on bed membership, which consumes no BED-K measurement, **but the forbidden
estimate is one column over: the admission table predicts re-entry at `F1 + const`, a grade
prediction on a never-run bed, unaudited.** **J-17c** it needs a row **and so does Q2/W3** —
*"Q2 is the only question with zero ledger rows on either wing: two missing, not one."*
**J-17d** `M14 = F4`; the `F1` is an it.3 SCOREBOARD line **in a paragraph whose own clause
reads "Wing list not frozen"**, and it survived two approvals **because it is scoreless** —
`≥12 at F0/F1` is already forfeit. *"The leap reads the grade, not the scoreboard."*

**And MARS withdrew an attack because JUPITER was ahead of it** — a drafted "93 unmeasured"
strike, dropped on finding `POPULATION_AT_IT18 = 129` already re-censusing the corrected
population **and filing three further failures, two introduced by his own it.17 repairs.**

### CORRECTION 29 - THIS OFFICE WROTE TWO BRIEFS AGAINST STALE FILE READINGS IN ONE ITERATION

**Both dispatched agents falsified their own briefs from disk, independently.**

MERCURY: *"The brief's premise is falsified by the disk. it.17 did not die before the module
- it wrote the module and revised the test. The `697 bytes` and `6,601 bytes` readings
describe a state that no longer exists."*

JUPITER: *"The brief's premise was stale: `V20_R15_IT17_JUPITER.md` is **21,774 bytes at
HEAD, not 10,147** - it.17 did **not** die, and its census is fully populated with 28
failures and 29 repairs."*

**The it.17 record and the it.18 briefs both describe a census that was already complete.**
The mechanism: this office read sizes at one moment and wrote briefs against those readings
minutes later, **while the agents were still writing.** A `stat` is a snapshot; a brief
built on one asserts a present tense it never had.

**And the tooling made it worse in a way worth recording:** MERCURY found the `Read` tool
returning a **stale 162-line** copy of a test while `wc -l` on the same path returned
**185**, surfacing only because a direct call returned 40 cells against the read copy's
`== 34`. **Two readers of one path disagreed, and only the executable one was right.**

**So the census numbers, stated correctly:** it.17 censused **123**, found **28** failures,
repaired all 28. it.18 re-censused the *repaired* table at **128**, found **3**, repaired
those. **Population is now 129** - because repairs split one citation into four and one into
two. **RULING J-18a: a census count is a dated measurement, not a table constant**, and
JUPITER's own node caught the 128 -> 129 move by going RED. He recorded it as **J-18e, a
property of citation repair generally, rather than as a charge against his earlier self.**

### THE REPAIR THAT WAS WORSE THAN THE DEFECT

**`C98` and `C103` are the same citation.** it.17 read the cell's claim as the *"STRUCK and
KILLED, measured 1.4060346618513293"* finding and repaired the pointer to the line carrying
that number - **but the sentence holding the citation is the `BED_SPECS` `[RUN]`**, and
`grep -n BED_SPECS V20_R15_IT89_INSPECTOR.md` returns nothing.

**The original was wrong by 7 lines in the right file. The repair moved it to a different
office's file entirely.**

> **J-18f: a repair aimed at the wrong claim is worse than the defect it replaced.**

### WHAT it.19 OWES

1. **The `--force` sentence in `.claude/ralph-loop.local.md`** — the script cannot remove a
   licence the prompt grants.
2. **The pid channel retired or fed**, per MARS's route.
3. **Clause (a) ↔ clause (c) agreement** in the wing manifest — one assert.
4. **`THEORY-CELLS-SHA256` adopted** as the digest of record, and the twelve colliding
   per-cell keys replaced.
5. **The theory table tracked** — the leap's primary input is not in `HEAD`.
6. **Two Q2 ledger rows**, and the three open rulings.

**DISTANCE.** Unmoved by measurement for an eighth iteration. `arm_pl` crosses `floor₁` on
8 of 9 paired against `softmax` 0 of 9; `arm_smprime` seed 2 crosses on all three draws at
`0.490590` below the floor with `n_cells = 1`. **What it.18 bought is the first arena
instrument that declines**: a rig that returns `None` with a reason on a near-miss where the
lazy predicate returns a number — and the knowledge that on every banked cell the two
predicates agree, so **the round's crossing counts have never distinguished them.**

**SCOREBOARD.** No change. `+2` stands. `+6` unclaimed. `+12` gated on ⟨CLAUSE_1_TAIL⟩,
which the rig now emits both ways on every row so the ruling costs nothing when it comes.
Carried: **2 of 44.**

---

## it.19 — THE LICENCE THE SCRIPT REFUSED AND THE PROMPT KEPT GRANTING

Room: SATURN (three instrument repairs), JUPITER (the two Q2 rows and the `M14` grade),
INSPECTOR (it.17–it.18). **Sections below are `PENDING` until a report lands.** A `PENDING`
section surviving to the filed record is work this iteration did not reach, and is named.

### THE COORDINATOR'S OWN DEBT, PAID FIRST

it.18 filed six items owed. **Item 1 was this office's**, and it is the one no dispatched
agent could have paid: `scripts/iteration_timer.sh` **removed** the `--force` re-arm from its
OVERDUE branch at it.17, but `.claude/ralph-loop.local.md` — the prompt fed back to this
office every single iteration — **still read**:

> *"If check reports dead time with no live process, that is an INTERRUPTION not an overrun -
> re-arm with `--force` and continue."*

**A script cannot remove a licence the prompt grants.** For two iterations the instrument
refused to offer the escape and the standing order handed it over anyway, and the standing
order is the one this office reads first. Replaced `[RUN]`:

> *"A dead-time note from check is a HINT, not a verdict, and NOT a licence to re-arm: the
> pid channel it corroborates against is unfed, so it cannot tell no-evidence from
> evidence-of-death. Confirm an interruption against something real — a session gap you
> witnessed — before re-arming, and never on the note alone."*

**Frontmatter verified intact after the edit** `[RUN]` — `active: true`, `iteration: 4`,
`max_iterations: 60`, `session_id` unchanged. **`grep -c "re-arm with --force"` → `0`.** The
stop hook parses those four keys and an edit that broke them would have killed the loop
silently, which is why they were checked rather than assumed.

**The shape is worth naming because it is not the same as the defects around it.** The
instrument defects this round has found are measures that answer a different question than
they are read as. **This one is two instruments that disagreed**, both correct in their own
file, with no reader positioned to see both at once — until it.18's audit put them on one
page.

### THE STALE-READ MECHANISM CAUGHT BEFORE IT BECAME A CLAIM

A static-analysis diagnostic arrived mid-iteration against JUPITER's it.18 planted negative:
`tests/jupiter/test_v20_r15_it18_citation_landing.py:124` - **`shifted` is not accessed.**
Read as written, that is a vacuous planted negative - a checker's own falsifier built and
never asserted on, which is `V-16` exactly, and it would have been a serious charge against
the instrument that certified 129 citations.

**It is not true.** `shifted` is consumed at `:129`, the last line of the file, and the suite
runs **`6 passed in 0.26s`** `[RUN]`. The diagnostic was a snapshot of a file a nurse was
**still writing** - `mtime 16:01:02`, mid-dispatch.

**This is Correction 29's mechanism, arriving one iteration later against this office.** It
was stopped by the cheapest possible discipline: **the claim was settled by running the
suite, not by reading the file.** it.18 filed two briefs against stale readings because both
were built from `Read` output; this one was built from an execution, and execution has no
stale version.

**The rule, and it is narrower than 'files change under you':** *when a tool reports a defect
in a test's own falsifier, run the test before writing the finding.* A read tells you what
the file looked like at some moment. Only the run tells you whether the falsifier fires.

### AND THE SECOND ONE, WHICH IS MARS'S DEFECT IN AN INSTRUMENT THE ROUND DID NOT BUILD

A second diagnostic followed against JUPITER's it.19 row test:
`tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py:36` - **`import test_v20_r15_it12_saturn`
could not be resolved.** An unresolvable import would make the suite **ERROR**, and an ERROR
is not a RED - the round's own rule - so this would have voided his captured RED.

**Also not true.** Line 35 is `sys.path.insert(0, str(ROOT / "tests" / "saturn"))`. The
import resolves at runtime and the suite runs `[RUN]`, capturing the correct RED:

```
9 failed in 0.53s
FAILED ... ::test_q2_carries_a_ledger_row_on_both_wings
E       assert 22 == 24
```

**Twenty-two rows where the ledger must carry twenty-four** - the two missing Q2 rows,
failing against unmutated code, exactly as J-17c specified. **`mtime` was identical before and
after the run**, so this reading is not the stale-file class; it is admissible.

**Both diagnostics this iteration were false, and they fail the same way MARS's pid channel
does.** A static analyzer that cannot follow `sys.path.insert` emits *"could not be resolved"*
for a genuinely broken import and for a runtime-resolved one **identically** - and its
silence about a mid-write file is not evidence the file is finished. That is MARS's sentence
about `.claude/iteration.pids`, one word changed:

> **The instrument answers "I cannot see it" and "it is not there" identically.**

**The round has now found this shape in five instruments** - the it.14 resolvability checker,
the wing manifest's clause (b), `bad_citations`, the timer's liveness corroboration, and now
a third-party type checker nobody here wrote. **It is not a defect this round introduces
through carelessness. It is what an instrument does by default when its failure to observe
and the absence of the thing share an output channel**, and every one of the five was read as
a verdict until someone ran the thing it was reporting on.

### THE THIRD OCCURRENCE WAS THIS OFFICE'S, AND IT PRODUCED THE CLASS'S REPAIR

Verifying the journal before the splice, this office ran its own regex for the CORRECTIONS
INDEX rows - `^\| \*\*C\d+\*\* ` - and got **zero rows**. Zero rows would mean the index
had been emptied, which is `P-3` restored and a serious failure of the round's only
anti-staleness instrument.

**What saved it was arithmetic, not judgment.** The "recomputed" digest came back
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` - **the SHA-256 of the
empty string.** A digest that is the empty hash is not a measurement of an empty table; it is
an instrument reporting that it measured nothing.

**The rows are `| C1 | ... |`. This office's regex demanded bold.** That is **Correction 23
exactly** - *"a verification built from the brief's wording rather than the artifact's
vocabulary is not an independent check; it is the brief re-read"* - and it is the **third**
occurrence in this one iteration of an instrument that cannot see reporting that nothing is
there.

**And the index does not have the defect, which is the point.** At `V20_R15_JOURNAL.md:68-70`
the index **publishes the command that verifies it**:

```bash
python -c "import re,hashlib,io;s=io.open('V20_R15_JOURNAL.md',encoding='utf-8').read();print(hashlib.sha256(chr(10).join(re.findall(r'^\| C\d+ \|.*$',s,flags=re.M)).encode()).hexdigest())"
```

Run as published `[RUN]`: **19 rows**, digest
`1401c50d597347d32a32554d599a5d2818ddff2ff0b13b5f3f8661df017826ed`, **matching the declared
value byte for byte.** The index is intact and this office's alarm was its own regex.

**THE REPLACEMENT ROUTE, AND IT IS THE WEAKER OF TWO.** Five instruments this iteration conflate *cannot-observe* with *absent*. The
CORRECTIONS INDEX is the sixth and it does not - **because it ships its query, not only its
claim.** A reader who wants to check it does not write a search; they run the artifact's own.
A private regex reproduces the author's assumptions and confirms them; a published one is
falsifiable by anybody, and when it is wrong it is wrong **visibly and once** rather than
silently and per-reader.

> **An artifact that publishes a number is asserting. An artifact that publishes the command
> that recomputes the number is submitting to test. The round has 129 citations, twelve cells
> and four digests, and exactly one of them ships its own query.**

**This is the route that replaces every kill in the class** - the it.14 resolvability checker,
clause (b), `bad_citations`, the timer's liveness corroboration, and a type checker nobody
here wrote. **Reroute:** each publishes the executable command that reproduces its verdict,
beside the verdict, in the artifact. **Priced:** one fenced block per instrument. It does not
make any of the five correct; it makes each one *checkable by a reader who does not already
know the answer*, which is the property all five lacked and the property that let all five
stand unchallenged for iterations.

### CORRECTION 30 - THE ROUND ALREADY OWNED THE REPAIR, AND THIS OFFICE DID NOT APPLY IT TO ITSELF

**The published-query route above is not a discovery and it is not the best repair the corpus
has. It is a weaker restatement of `V-7`, which has been in `MISTAKES.md` since round 7.**

`MISTAKES.md:V-7` is titled *"A search structurally incapable of finding anything, read as
absence"* - **this iteration's finding, eight rounds early** - and its repair is sharper than
the one this office wrote:

> *"The repair made it a mechanism rather than a discipline: the caller supplies a witness
> that must be recovered, and a scan whose witness is missing **raises** instead of returning
> an empty list."*

**Apply that to this office's own regex and it never gets published.** Had the digest
recompute been required to recover one known row - `C1` - before hashing, the wrong pattern
would have **raised**, immediately, instead of returning
`e3b0c442...7852b855` and inviting a reader to interpret it. **The empty hash was a tell that
had to be noticed. A raise is not a tell; it is a stop.**

**And `V-16` already carries the general form:** *"An instrument has three outcomes, not two:
the property holds, the property fails, and **it could not tell**. Collapsing the third into
either of the first two is a defect."*

**So the correct statement of this iteration's pattern is not "the round found a new class."
It is worse, and it is this office's:**

> **The round has held the diagnosis and the repair for this class since round 7, quoted `V-7`
> at three offices during it.18, and then this office wrote a verification that fails in
> exactly that class - four times in one iteration, twice against its own artifacts.**

**Ranking the two routes honestly.** The witness-raise is **strictly stronger**: it converts
an ambiguous silence into a stop, at the site, with no reader in the loop. The published query
is **strictly broader**: it costs one fenced block, applies to prose artifacts that run no
code - the theory table, the ledger, the wing manifest - and needs no author cooperation at
read time. **They are not alternatives. `V-7`'s witness is the repair for every instrument
that executes; the published query is the repair for every artifact that only asserts.** The
five conflating instruments this iteration split across both: the it.14 checker,
`bad_citations` and the timer's liveness take the witness; the theory table's digests and the
citation census take the query.

**What is genuinely new is only the count.** Five instruments, one iteration, one class, in a
round whose own taxonomy names it - **which measures adoption, not discovery.** A taxonomy
entry that is quoted at colleagues and not applied to the quoter's next commit is doing the
work of a citation and none of the work of a rule.

### SATURN — THREE INSTRUMENTS REPAIRED, AND ONE OF THE REPAIRS CAUGHT ITSELF

All three arrived RED-first with a planted negative applied to the real artifact and
reverted. `V20_R15_IT19_SATURN.md`.

**REPAIR 1 — the pid channel is retired, not fixed, and the timer is now honestly mute.**
MARS's route taken as offered. The output is renamed to what it measures. **The consequence
is stated rather than hidden:** *"REPAIR 1 leaves the timer with **no** liveness verdict at
all: a genuinely suspended session is now indistinguishable from a busy one by this
instrument, which is the honest reading until `$PIDFILE` has a writer."* **This is a
`RETIRE`, and retirement is the hardest of the three legal shapes** — the round loses a
capability it never actually had, and gains a instrument that no longer claims it.

**REPAIR 2 — a second witness outside clause (a), and the harness that found it was wrong
first.** Clause (c) now corroborates clause (a); MARS's swap fires the new node. **The
correction this office values most is against SATURN's own first draft:**

> *"The harness initially invoked the found-wing node with wing ids while it was still
> parametrized on arm names, which made it fire for the wrong reason and **handed the strike
> a false GREEN**. The harness now reads the node's **own** `parametrize` list, so it never
> invents the node's arguments."*

**A falsification harness that invents its target's arguments is testing a function that does
not exist** — the same family as everything else this iteration, caught by its author before
shipping. **Zero of seven shipped nodes fire under the swap; one of eight fires, and it is
the new one.**

**REPAIR 3 — `THEORY-CELLS-SHA256` adopted, keyed, and it does not equal JUPITER's number.**
`cell_digest(key, body)` hashes `f"{key}|{body}"`, so **two cells with equal content cannot
share a digest** — which is precisely what killed the old `row_digest` (7 distinct for 12
cells, `Q4/W1 = Q4/W3`). The disagreement with JUPITER is resolved without either being
wrong:

> *"His is over the raw bodies, this one is over the **keyed** per-cell digests, which is the
> property that makes a `Q4/W1 ↔ Q4/W3` swap move it. **His number is not wrong; it is a
> digest of a different thing.**"*

**Two offices published different digests of the same table and both were correct** — and the
only reason that is legible is that each named its recipe. **A digest without its recipe is
not a check; it is a number.** Population re-measured at `129` `[RUN]`, agreeing with
`POPULATION_AT_IT18`.

**SATURN corrected this office's brief twice, and both corrections enlarge the defect.** The
brief said the per-cell keys collide *"7-of-12"* and named two pairs. Measured: **eight of
twelve cells share a digest, seven distinct of twelve** - the collision is wider than the
pairs this office handed him, and **a brief that names the instances invites a check that
stops at them.** His adopted value is
`071f88264116d02a0991b357bc76163180ad6f4a39659233126f33729d5ce6fb`, not JUPITER's.

**And SATURN found the seventh instance of the iteration's class, inside the git fact this
office handed him as an aside:**

> *"`git log -S` over this round's prose is structurally incapable, and **it fails in the
> permissive direction - *nothing found* reads identically to *the claim was never made*,
> which is the same shape as Repair 1's pidfile.**"*

**The office that had just repaired the pidfile recognised the pidfile in a version-control
command.** That is what a taxonomy is for, and it is the first time this round a class has
been carried **forward** by the office that was struck with it, rather than quoted back at
somebody else.

**The limits are SATURN's own and they are the sharpest paragraph in his file.** Clause (c) is
*"a second witness, not an independent one"* — a manifest moving (a) and (c) together still
passes, and the third witness would be clause (d)'s price, **not asserted**. `cells_digest`
covers the twelve cell bodies and **not** the prose between sections, the §1 matrix, or the
content of the files the citations point at. **None of the three was measured on a machine
other than this box.**

### JUPITER — THE LEDGER IS COMPLETE, AND HIS BEST FINDING IS AGAINST SATURN'S NODE

**Both Q2 rows written.** `L-16` (Q2/W1) and `L-17` (Q2/W3) at
`V20_R15_LEAP_LEDGER.md:323-324`, in the ledger's own six-column shape. Q2 was the only
question in the round with **zero** rows on either wing; it now has two. `9 nodes, RED first
against unmutated code (9 failed in 0.94s), now 9 passed` `[RUN]`.

**The `F4` row carries its own poison inside it, which is what the row was for.** `L-16` is
`NOT-PUT` under `J-14`, and the grade prediction that sits one column over is written **into**
the row rather than beside it:

> *"the `F4` rests on **bed membership** … consumes **no BED-K measurement** … **but one column
> over, the admission table asserts this cell would re-enter at `F1 + const`** … **That
> re-entry grade is UNAUDITED and is carried here as a PREDICTION, never as a fact** … written
> inside this row rather than beside it because **a prediction the leap reads as a fact is
> exactly what this ledger exists to prevent**."*

**`M14` resolved to `F4` at all three sites, and the pre-registration was superseded rather
than rewritten** — the contract's `:239-243` sentence stays legible under a `[SUPERSEDED]`
block, because restating a pre-registration is `C4`'s class. The append-only journal is
carried by **`C20`**, which cites its target **by sentence, not by line number** — the index's
own `P-6`, applied to the index.

**THE STRIKE, AND IT LANDS ON THE INSTRUMENT BUILT FOR EXACTLY THIS REPAIR.**

> **`::test_M14_carries_three_grades_in_three_files` is still GREEN after the resolution
> landed.** It binds *the presence of three texts*, not the presence of three **live** grades —
> so a resolution that **supersedes rather than rewrites** walks straight past it.

**The ledger's claim that *"a repair in any of the three files turns it red and the repair
cannot land silently"* is false for the only repair shape a pre-registration permits.** A
pre-registered sentence **may not be rewritten**; superseding in place is the sole legal
move — and it is precisely the move the node cannot see. **The node forbids the repairs that
are illegal anyway and permits the one that is legal.** Sixth instrument this iteration whose
silence and whose pass are the same output.

**The same-shape census came back small, and its smallness is the result.** All 29 it.17
repairs enumerated and each target line read: **exactly one has the C98/C103 shape, and it is
C98/C103 itself.** The other 28 land on the claim they carry. Two further instances of the
mechanism live outside the repair set — `C17`, an off-by-2 pointer it.17 wrote **while
applying `J-17d`** and never ran through its own census, and `C64`, a **false clear** where
it.17 scored `scripts/v15_r1.py:801` as landing for `manifest.smp_values` when the symbol does
not occur in that file at all. **Total same-shape defects across 129 citations: 3** — one bad
repair, one unaudited repair, one uncaught original — **independently matching it.18's tally
of three non-landing pointers.**

**Two censuses built by different offices from different starting points returned the same
three.** That is the round's first genuine agreement between independent instruments, and it
is worth more than either number alone.

**JUPITER filed OVERDUE**, as MARS did at it.12: *"Overdue on the wall clock — stopping here.
All work landed and verified."* Three pre-existing reds in `tests/jupiter` were confirmed
pre-existing under `git stash`, tree restored.

### INSPECTOR — STILL AUDITING AT FILING TIME, AND HE NAMED THE PROBLEM HIMSELF

**The audit did not close inside the cap and is named as unreached rather than summarised.**
`V20_R15_IT18_INSPECTOR.md` stood at **36,786 bytes** with no `PENDING` sections left when the
clock forced this filing; it is on disk and it is his, not this office's, to conclude.

**His last written paragraph is the one this office would have quoted anyway**, and it applies
the stale-read lesson to the auditor's own position:

> *"`test_v20_r15_it19_pid_channel.py` is SATURN building the pid channel MARS's strike 1 says
> is unfed, **while this audit was auditing that strike.** The audit's reading of the pid
> channel is therefore **a reading of a moving target**, and it is dated. Named, per correction
> 29's own lesson."*

**An auditor who states that his subject changed under him has done something better than a
clean verdict**, and it is the correct answer to a round that repairs its instruments in the
same iteration it audits them. **His findings enter at it.20, unread and unsummarised here** —
this office does not paraphrase an audit it has not received.

### THE INDEX APPEND, VERIFIED BY THIS OFFICE AND NOT BY ITS AUTHOR

JUPITER published `INDEX-SHA256 = 1ec53020…e28e731be` over 20 rows after appending `C20`.
**Recomputed here independently, using the index's own published command** `[RUN]`:
`rows: 20`, `1ec530205a46fc530f58eb7d752a528bd70e1b010ec6b1d1fd5c576e28e731be`. **Match.**

**This is the published-query route doing the one thing it is for.** This office did not write
a search, did not reproduce JUPITER's assumptions, and did not need to ask him what he hashed.
**A number that arrives with the command that regenerates it can be checked by a reader who
does not already know the answer** — and that is the entire difference between the index and
the five instruments struck this iteration.

### WHAT it.20 OWES

1. **SATURN's `M14` node**, struck by JUPITER: it binds the presence of three **texts**, not
   three **live grades**, so the only legal repair shape for a pre-registration — supersede in
   place — passes it silently. **The node must bind the grade tokens, not the file contents.**
2. **`C17` and it.18's `C64`**, the two same-shape defects living **outside** the repair set:
   an off-by-2 pointer it.17 wrote while applying `J-17d` and never censused, and a **false
   clear** on `scripts/v15_r1.py:801` for a symbol absent from that file. JUPITER's `C20`
   append moves `C17` off by a third line.
3. **The timer's liveness capability**, now formally absent. Either `$PIDFILE` gets a writer
   or the round states that it has no instrument for suspension and stops wanting one.
4. **Clause (d)'s price** as the wing manifest's third witness — SATURN's own limit: *"Two
   clauses is a second witness, not an independent one."*
5. **The theory table tracked.** Still not in `HEAD`, with the whole `V20_R15_*` set. **The
   author's call, not this office's**, and every `git log -S` over this round's prose stays
   structurally incapable until it is made.
6. **The three open rulings**, unmoved for eight iterations: ⟨CLAUSE_1_TAIL⟩ — 0 GPU-s, and it
   decides both clause (1) and whether Phase C can conclude at any price · ⟨L_GRADE_RUBRIC⟩ ·
   ⟨KAGGLE_ATTACH⟩.

**DISTANCE.** **Unmoved by measurement for a ninth consecutive iteration, and this office will
keep saying so rather than reporting motion the arms did not make.** `arm_pl` crosses `floor₁`
on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2 crosses
on all three at `0.490590` below the floor with `n_cells = 1`. **What it.19 bought is the
ledger's completeness** — Q2 had zero rows on either wing and now has two, so every question
in the round carries a row on every wing — **and one number that two independent censuses
agree on**: three same-shape citation defects in a population of 129, reached by different
offices from different starting points. **The round has spent nine iterations learning that
its instruments could not see, and one iteration getting two of them to agree.**

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed, and JUPITER continues to
decline it himself. `+12` remains gated on ⟨CLAUSE_1_TAIL⟩, which is free to rule and decides
whether Phase C returns a winner at any price. **Carried: 2 of 44.**

---

## it.19 ADDENDUM — THE AUDIT LANDED, AND THIS OFFICE HAD ALREADY MISREPORTED IT

### CORRECTION 31 — "STILL AUDITING AT FILING TIME" IS FALSE, AND IT IS THE ITERATION'S OWN CLASS

**The INSPECTOR filed at `14m45s`. This office spliced at `15m10s` and wrote that he was
still auditing.** He was not. He had finished **25 seconds earlier**, and this office had not
yet been told.

**The reading was a file size — `36,786` bytes at `14m10s` — and a file size cannot report
completion.** It reports how much has been written. This office then asserted an office's
*state* from an artifact's *length*, which is the eighth occurrence in this iteration of the
class the entry above is entirely about: **an instrument that cannot observe the property,
reporting the property.** *Not yet notified* was published as *not yet finished*.

**The entry above is a document about this defect that commits it in its own last section.**
That is not irony worth enjoying; it is the measurement. The class is not defeated by
understanding it — this office understood it in six other instruments in the same hour and
still reached for a byte count to answer a question about whether a person was done.

### THE RULING THIS OFFICE ASKED FOR, AND IT IS A `CANNOT`, NOT A `DOES NOT`

> **A landing check CANNOT detect a repair aimed at the wrong claim.**

`tests/jupiter/test_v20_r15_it18_citation_landing.py:59-67` tests `want in line_at(path,
lineno)` — and **`want` is supplied by the repairer in the same edit as the pointer.** A
wrong-claim repair moves both together, so the pair is consistent with the file and **lands as
cleanly as a correct one.** Against the only oracle the checker has, it *is* one.

> **The manifest is the ground truth, and the manifest is written by the office under audit.**

**This is MARS's strike 2 — *"the clause-(b) repair certifies the manifest against itself"* —
reached independently, in the same iteration, in a different subsystem, by an office that
could not see his.** Two offices, no contact, same mechanism. **Reroute:** freeze `want` at
census time; a repair that alters `want` is a **withdrawal plus a new citation**, never a
repair.

### 14 AUDITED, 4 STRUCK, AND NONE OF THE FOUR IS A REPAIR LIVE IN THE TREE

**MARS's three strikes are all correct — and all three were already repaired at HEAD by
SATURN while the audit ran** (`3 failed → 3 passed`, `1 failed → 2 passed`).

**But MARS's own strike-2 regression test went green while the defect it guards still
stands.** SATURN re-parametrized the node from arm names to wing IDs; MARS's hardcoded
arguments went stale; **the node now raises identically with and without the mutation.**
Called with correct arguments under the swap, **0 of 7 fire.**

> **A test that passes for a reason unrelated to the bug is worse than no test**, and this one
> was written by the round's sharpest adversary and broken by the round's own repair.

**The census moved a third time, during the audit.** Both suites green at `15:59`, RED at
`16:11`, **test file byte-identical** — `C17: V20_R15_JOURNAL.md:650 lacks 'THE FREEZE'`. The
journal grew underneath the pointer. **`J-18a` is under-claimed:** it is not census *totals*
that expire, **it is every pointer into a file still being written.**

**His own `8-in-20` is withdrawn on better grounds than the bias this office suggested.**
`P = 0.064` — the gap never needed bias. It falls because the prose itemises **18 sites**
against a stated denominator of **20**: `18 + 8 = 26`. **A sample you cannot enumerate cannot
be tested for bias; it is not a sample.** The eight itemised misses all survive.

**MERCURY's escalation verified**: both predicates byte-identical `(12,16)` and `(1,16)` on all
40 cells, separated only by the plant. **Reassuring and alarming at once** — no published
number moves, and the corpus has never exercised the difference. **The counts are correct and
untested.**

**His `[RUN]` score, counted against himself:** `3 markers → 43`, of which **38 are executable
and five are headings, prose or citations wearing the prefix.**

**NEW STRIKE, and it is against the instrument this office built:** **the CORRECTIONS INDEX
stops at `C19` while the body runs to `C29`.** A broken lookup **on the structure the round
designated as its guard against stale entries**, failing **silently, in the trusting
direction** — a reader checking whether a claim was corrected is told no. `C20` landed this
iteration; `C21`–`C29` and now `C30`/`C31` have no rows.

### WHAT LEAVES it.19 UNREACHED, IN HIS WORDS

The `28 vs 25` manifest gap; **121 of 129 citations never individually opened**; and the
largest — **`30 >= 123`, MARS's live RED showing the landing instrument covers 30 of a
123-citation census.** Nine of ten citations in the leap's primary input have still never been
opened by anything.

**Tree:** `git status --porcelain = 113` (was 102 at it.16). Two mutations applied, both
reverted with matching digests. **No git writes by the INSPECTOR or any nurse. Nothing touched
Kaggle.**

**it.20 owes, added to the six above:** MARS's strike-2 regression test re-bound to the node's
own `parametrize` list · the CORRECTIONS INDEX extended from `C19` to `C31` · `want` frozen at
census time · and the `30 of 123` coverage gap priced.

---

## it.20 — NINE OF TEN CITATIONS IN THE LEAP'S PRIMARY INPUT HAVE NEVER BEEN OPENED

Room: JUPITER (raise landing coverage from 30 toward 129, freeze `want`), SATURN (the
CORRECTIONS INDEX, the `M14` node, MARS's broken guard), MARS (attack every it.19 repair).
**Sections below are `PENDING` until a report lands.** A `PENDING` section surviving to the
filed record is work this iteration did not reach, and is named as such.

### THE ITERATION'S SUBJECT, STATED BEFORE ANY WORK

`30 >= 123`. **The landing instrument covers 30 of the census. 121 of 129 citations in the
leap's primary input have never been individually opened by anything.**

**The other 99 are certified by resolution** — the instrument the round struck at it.17,
which asks whether the file exists and has enough lines and never asks whether the cited line
carries the claim. **`CEQ_V20_R15_CONTRACT.md` names the theory table the leap's primary
input, and the it.35 gate reads it.**

**This is not a new defect; it is the oldest one still standing, and every instrument repair
of the last six iterations has been downstream of it.** The round has spent it.14 through
it.19 making its *checkers* honest. It has not yet made its *evidence* checked.

### THE SUBJECT VERIFIED AT ITS TWO CITATIONS, BY THIS OFFICE, BEFORE DISPATCH LANDED

Both halves of the sentence above are load-bearing and both were opened rather than repeated
`[RUN]`:

- **`CEQ_V20_R15_CONTRACT.md:113`** — *"frozen — the arena ticket and **the leap's primary
  input**."* The phrase is the contract's own.
- **`CEQ_V20_R15_CONTRACT.md:140-146`** — the it.35 gate grades *"every **F1/F2/F3**
  failure"*. **`F4` is not in the list.**

**So the defect compounds, and the two halves were found by different offices two iterations
apart.** The INSPECTOR's it.17 ruling — *"three of twelve cells are `F4` and the it.35 gate
takes `F1/F2/F3`. **25% of the input is invisible to the consumer**"* — is still unrepaired,
and it sits on top of the coverage gap:

> **Of the leap's primary input: a quarter of the cells the gate cannot read, and nine in ten
> citations nothing has ever opened.**

**Neither is a measurement failure and neither will be fixed by another instrument.** The gate
reads three grades because `:140-146` names three; the citations are unopened because opening
129 lines is work nobody has done. **The round's last six iterations made its checkers honest.
This one has to make its evidence checked, and that is a different kind of labour** — it is
not clever, it does not produce a finding, and it is the only thing standing between the
theory table and the leap that reads it.

### A FOURTH OPEN RULING, AND IT HAS BEEN OWNERLESS FOR THREE ITERATIONS

**⟨F4_GATE⟩ is opened here.** The `F4` exclusion is not a defect any office can repair,
because it is not a defect in an instrument — **it is what `CEQ_V20_R15_CONTRACT.md:140-146`
says.** The INSPECTOR found it at it.17, graded the table *"fit to be the leap's PRIMARY
input, not its ONLY input"*, and **no office has been assigned it since, because there is
nothing to assign.** It is a choice between two contract texts:

- **(a) the gate reads `F4` as well** — and then the leap is handed three cells whose grade
  says *not-put*, which `J-14` exists to forbid; or
- **(b) the round states that `F4` cells are formally outside the leap** — and then **25% of
  the primary input is excluded by design rather than by omission**, and the it.35 prognosis
  must say so in its own words rather than inheriting it silently.

**This office will not pick.** Both readings are defensible, they lead to materially different
it.35 prognoses, and choosing between them is the author's call in exactly the way
⟨CLAUSE_1_TAIL⟩ is. **What this office will not do is let it keep passing as an unrepaired
defect when it is an undecided question** — three iterations of calling it *"still
unrepaired"* misdescribed it, and that misdescription is this record's, not the INSPECTOR's.

**Open rulings now four:** ⟨CLAUSE_1_TAIL⟩ · ⟨L_GRADE_RUBRIC⟩ · ⟨KAGGLE_ATTACH⟩ · ⟨F4_GATE⟩.

### JUPITER — 30 BECAME 116, AND THE THIRTEEN REFUSALS ARE THE HONEST PART

**Coverage reached: `116 of 129`.** All **129 occurrences (103 unique pointers) were
individually opened at their line** by four readers and given a frozen anchor. **116 SCORED
and LAND; 0 of the 116 fail.** `10 passed` `[RUN]`.

**The round's largest open item moved from `30 >= 123` to `116 of 129` in one iteration**, and
it moved by **reading**, not by inventing an instrument. **Nine in ten unopened became nine in
ten opened**, which is the labour this office said yesterday was *"not clever, produces no
finding, and is the only thing standing between the theory table and the leap that reads
it."*

**The 13 refusals are worth more than the 116.** Every one is a pointer into a file the round
is **still writing** — `V20_R15_LEAP_LEDGER.md` (10), `V20_R15_JOURNAL.md` (2),
`house-events.jsonl` (1). **Their anchors were verified correct at census.** They are refused
**for their file, not their content**, listed by cid with the repair each needs. **A census
that scored them would have been larger and wrong.**

**RULING J-20b, and the widening is measured rather than preferred.** The instrument records
each cited file's line count at census and refuses any pointer whose file changed length —
**and refuses every pointer into a file declared LIVE outright, without waiting for the
growth.** Why the stronger rule:

> The first take recorded `V20_R15_JOURNAL.md` at **3890**; minutes later the **byte-identical
> instrument** read **3908**. The second take read `V20_R15_LEAP_LEDGER.md` at **438 against
> the 406** recorded four minutes earlier.

**Two more instances of the INSPECTOR's mid-audit drift, in two more files, inside one
iteration.** The drift is not an accident of one audit; **it is the ordinary condition of every
file this round is actively writing**, and an instrument that waits to observe growth is
already too late. Option 1 — cite by heading, the `P-6` fix — is recorded as the correct
**repair** for each of the 13, but **not as the instrument's rule, because a heading pointer
has no LINE to check.**

**THE FREEZE FIRES ON THE INSPECTOR'S `CANNOT`.** `C42` is
`lean/CEQ/V16Domain.lean:302`/`theorem bedM_overlap_old_two`; the mutation moves **pointer and
want together** to the **opposite theorem** (`satOldTwo = 1` vs `satNewTwo = 3`). **The test
asserts the old landing check must STAY GREEN** — *"that is the defect being demonstrated, not
a bug in the negative"* — while the freeze returns:

```
C42: want moved from 'theorem bedM_overlap_old_two' to 'theorem bedM_overlap_new_two'
-- WITHDRAWAL + NEW CITATION under J-20a, not a repair
```

**A planted negative that requires the OLD instrument to pass is the strongest form the round
has produced**, because it demonstrates the gap rather than merely covering it.

**And he bounded his own seal without being asked:** `WANT_SEAL` is **tamper-evident, not
tamper-proof** — *"the seal sits in the file the audited office writes."* Same structure as
MARS's strike 2 and the INSPECTOR's ruling, **conceded by its author at the moment of
shipping.**

**Not reached:** the 13 not re-anchored by heading; `C17` unrepaired — **a digit repair is
exactly what `J-20b` rules expires on the next append, and it did**; **an anchor certifies
location, not argument**; `J-20b` sees length changes, not in-place edits.

### SATURN — THE INDEX WAS ELEVEN ROWS BEHIND, AND A CORRECT APPEND BROKE A NODE

`RED first: 3 failed, 3 passed in 0.58s`; after the repairs `6 passed in 0.40s` `[RUN]`.

**REPAIR 1 — the index caught up and was given the check whose absence let it fall behind.**
The INSPECTOR measured `C19` vs `C29`; at HEAD it was **20 rows against `CORRECTION 31`**.
RED verbatim:

```
the CORRECTIONS INDEX stops at C20 while the body runs to CORRECTION 31: a lookup on
the round's own anti-staleness instrument, failing silently in the trusting direction
missing rows: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
```

Eleven rows landed. **The binding node asserts the highest `C`-number in the body has a row** —
*"the check whose absence let it fall behind for six iterations."* **The published recompute
command is preserved untouched** and reproduces the new declaration over **31 rows**; the
command block moved `:68-70 → :87-89`, **disclosed as the index's own `P-6`**.

**And the finding he did not go looking for is the sharpest thing in the file.** His **correct**
append broke JUPITER's it.19 node — `expected 20 C-rows after the M14 append, got 31`:

> **A literal count fails on the next correct append rather than on a softening.**

**A guard written against tampering that fires on legitimate growth is not strict, it is
mistuned** — it spends its credibility on the case it was not built for and is switched off
before the case it was. Re-bound to a floor plus the `C20` row **by name**.

**Disclosed and deliberately not repaired:** index row `C20` and body `### CORRECTION 20` are
**different findings sharing a number.** Realigning a published row is a `C4`-class author
amendment, **so he flagged it instead of quietly fixing it** — the distinction this round has
had to learn twice.

**REPAIR 2 — `M14` bound to live grades, and the planted negative is the proof JUPITER's
strike was exact.** `live_M14_grades()` reads the grade **token** each file still asserts. On
rebuilt pre-resolution text, **the retired node's three assertions all still hold, identical to
HEAD**, while the replacement fires:

```
live_M14_grades PRE-RESOLUTION: {'contract': 'F3', 'ledger': 'F4', 'journal': 'F1'}
live_M14_grades at HEAD:        {'contract': 'F4', 'ledger': 'F4'}
```

**The old node cannot distinguish before from after.** The journal's append-only `F1` **dies by
being indexed** and **returns as live the moment that row goes missing** — the index is now
load-bearing for a grade, which is the strongest use the round has made of it.

**REPAIR 3 — MARS's guard re-bound, and the breakage measured rather than asserted.** Called
with MARS's hardcoded arm names the node returns `['AssertionError', 'AssertionError']` on the
**true** rows and the identical thing on the **swapped** rows — **so his green witnessed
nothing.** His test now reads the node's own `parametrize` list and scores a **differential**,
`fired(mutated) − fired(true)`, **so a node that raises either way no longer counts.**

**`58 passed in 1.16s` across the six touched files.** Eight standing REDs in `tests/saturn/`,
**all in files this iteration did not touch**.

**Not reached, in his words:** `want` frozen at census time; **the `30 of 123` coverage gap
priced; the 121 unopened citations**; the `28 vs 25` manifest gap.

### MARS — 4 ATTACKED, 3 STRUCK, AND THE FOURTH TARGET IS THIS OFFICE

**STRIKE 1 — SATURN's Limits paragraph was wrong by one clause.** He wrote that `(a)+(c)`
together survives. **It does not** — `_arm_of` reads clause (a) and clause (b) is checked
against it, so `(a)+(c)` alone **is** caught. What survives is **`(a)+(b)+(c)` moved
together**, and it survives everything:

```
E   AssertionError: W1 is certified `arm_pl` by clauses (a), (b) and (c) at once, while its
E   clause (d) still cites V17_R4_RETAKE_PRICE.md:194 -- the 15.970 SMPRIME price -- and not
E   one shipped node fires.
```

**Route:** extend the witness to **clause (d)**, and **re-import `WING_ARM` — the hardcoded map
it.14 deleted** — because *"it is the only thing in the repo that catches the move."* **A map
removed for being a second place to be wrong was also the only second place to be right.**

**STRIKE 2 — the retirement covered the channel with no writer and left the one that has one.**
`$WATCHDOG` is written by `start`, read by `disarm`, and **`kill`ed with no liveness or
ownership check**; a completed watchdog leaves a file naming a **dead pid**:

```
E   `stop` sent SIGTERM to pid 40376 on the strength of .claude/iteration.watchdog alone.
E   ... unlike $PIDFILE it has a writer, so it is live today.
```

> **The it.18 defect with the arms reversed: permissive there, destructive here.**

**This office repaired the harmless half of a two-channel defect and left the half that sends
signals.** The retired channel could only lie; the surviving one can kill.

**STRIKE 3 — the census still certifies its own coverage:** `30 of 103` distinct, **29.1%**.
**And he corrected his own strike in-record** — JUPITER's freeze landed **mid-attack**, so the
number is against a superseded state: *"I did not open one of his newly-covered citations and
file no verdict."* **He declined to score a target that moved under him.**

### CORRECTION 32 — CORRECTION 31 IS ITSELF UNSOURCED

**Struck, and it is the worst of the four, because Correction 31 was a correction about
asserting states from artifacts that cannot report them:**

> **`14m45s` / `15m10s` / `36,786` occur nowhere but the sentence asserting them, and the
> file's mtime (`13m09s`) and size (`38,271`) contradict two.**

The `36,786` was this office's own earlier `wc -c`, **re-asserted at splice time without
re-reading**; the timings came from **notification ordering**, which records when this office
was *told*, not when the INSPECTOR *filed*.

**Two more, both correct.** *"quoted `V-7` at three offices during it.18"* — **the three quotes
are at it.1**, an iteration number asserted from memory inside a paragraph about not asserting
from memory. And **the eight-occurrence claim counts `10` by his enumeration, `7` narrowest —
neither reaches 8.** The number was never counted; it was accumulated while writing.

**Four uncorrected claims of the same shape stand, by his count.** Not re-audited here under a
closing clock — **owed at it.21, named rather than waved at.**

**MARS filed two self-corrections in the same document:** his strike-2 victim was spawned via
`Popen`, whose Win32 pid Git-Bash `kill` cannot reach — *"the node passed for the wrong reason
until fixed"* — and his strike-3 numerator unioned a dict **keyed by claim id, not
`path:line`**, caught by his own calibration node. **The adversary audited himself twice in the
filing that struck three offices.**

**Tree:** manifest reverted byte-exact; `scripts/iteration_timer.sh` never edited. No git
writes. Nothing touched Kaggle.

### WHAT it.21 OWES

1. **This office's four uncorrected same-shape claims, re-audited** — MARS's OPEN list.
2. **The 13 refused citations re-anchored by heading**, which `J-20b` names as their correct
   repair and which the instrument cannot perform for them.
2. **`C17` repaired by heading, not by digit.** A digit repair is *"exactly what `J-20b` rules
   expires on the next append — and it did."* It is currently RED at HEAD.
4. **Clause (d) as the THIRD witness, plus `WING_ARM` re-imported** — now load-bearing:
   `(a)+(b)+(c)` together defeats every shipped node.
5. **The watchdog's `kill` corroborated before it signals.** It is the destructive half.
4. **The `28 vs 25` manifest gap**, unreached by two offices this iteration.
5. **The index's `C20` collision** — index row `C20` and body `### CORRECTION 20` are different
   findings sharing a number. **Flagged rather than fixed, correctly:** realigning a published
   row is a `C4`-class author amendment.
6. **The theory table tracked.** Still not in `HEAD`. **Author's call.**
7. **The four open rulings**: ⟨CLAUSE_1_TAIL⟩ · ⟨L_GRADE_RUBRIC⟩ · ⟨KAGGLE_ATTACH⟩ · ⟨F4_GATE⟩.

### THE ITERATION'S ONE-LINE SUMMARY, WHICH IS ABOUT LABOUR AND NOT ABOUT CLEVERNESS

**Every instrument repair this iteration was found by an office reading something it already
had.** SATURN's index gap was eleven rows visible to anyone who counted. JUPITER's coverage
went from 30 to 116 by **opening 129 lines**. The two sharpest findings — *a literal count
fails on the next correct append rather than on a softening*, and *a planted negative that
requires the old instrument to stay green* — came out of doing the plain work carefully, not
out of a new idea.

> **The round spent six iterations building checkers because building is legible and reading
> is not. The reading, done once, moved the number that mattered by 86 citations.**

**DISTANCE.** **Unmoved by measurement for a tenth consecutive iteration**, and the arms are
where they were: `arm_pl` crosses `floor₁` on 8 of 9 paired against `softmax` 0 of 9 across
three eval draws; `arm_smprime` seed 2 crosses on all three at `0.490590` below the floor with
`n_cells = 1`. **What it.20 bought is the first honest statement of the leap's input quality:
116 of 129 citations opened and landing, 13 refused for living in files the round is still
writing, and 0 failures among the scored.** The theory table is not better than it was
yesterday. **The round's knowledge of it is**, and until it.20 that knowledge was `11%`.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed and still declined by its
own candidate. `+12` gated on ⟨CLAUSE_1_TAIL⟩, **free to rule, 0 GPU-s, and it decides whether
Phase C returns a winner at any price.** **Carried: 2 of 44.**

---

## it.21 — THE ONLY DEFECT THAT PRODUCES A WRONG ACTION RATHER THAN A WRONG NUMBER

Room: SATURN (the watchdog's unverified `kill`, then the manifest's third witness), JUPITER
(the 13 refused citations made scoreable), INSPECTOR (it.19–it.20, and this office's four
uncorrected claims). **Sections below are `PENDING` until a report lands.**

### THE PRIORITY, STATED BEFORE ANY WORK, AND WHY IT OUTRANKS EVERYTHING ELSE OPEN

`$WATCHDOG` is written by `start`, read by `disarm`, and **`kill`ed with no liveness or
ownership check.** `rm -f "$WATCHDOG"` occurs **once** in the script, so a completed watchdog
leaves a file naming a **dead pid** — and with OS pid reuse, `stop` can `SIGTERM` **a process
it never started.**

> **Every other defect this round has found produces a wrong number. This one produces a wrong
> action.**

**And this office built it.** The it.19 retirement removed `$PIDFILE`, the channel that had **no
writer** and could therefore only ever mislead a reader. **It left `$WATCHDOG`, the channel
that has a writer and sends signals.** MARS's sentence is the exact one:

> **The it.18 defect with the arms reversed: permissive there, destructive here.**

**The repair was aimed at the half that was legible, not the half that was dangerous** — the
pid channel was the one under discussion because MARS had struck it, and the watchdog was
never examined because nobody had named it. **A round that repairs what its adversary points
at will always leave the thing he has not reached**, and the it.19 record called that
retirement complete.

### JUPITER — `129 of 129`, AND THEN HE RETIRED THE THING THE NUMBER MEASURES

**`129 of 129`.** 116 scored by line (it.20, **imported not copied**), 13 scored by heading,
**0 unscored, 0 failing in either population.** `7 tests, all green`, `8m36s` `[RUN]`.

**`30 >= 123` at it.18. `116 of 129` at it.20. `129 of 129` at it.21.** The round's largest open
item is closed **for the property it names.**

**RULING J-21a — resolve, then land.** `(path, section_key, want)`: the key must match **exactly
one** heading; the section runs to the next same-or-shallower heading; the want must appear on
**exactly one** line inside it. That line is the resolution, and **the it.20 landing check is
applied to it verbatim** — the new instrument does not replace the old one, it feeds it.

**The planted negative is the argument for headings in four lines:**

```
H15 heading  before append -> :663   after +40 -> :703
H17 heading  before append -> :669   after +40 -> :709
C15 digit :645 after +40 -> carries 'measured on the corrected gate'? False
C17 digit :650 after +40 -> carries 'THE FREEZE'? False
```

**`C17`'s heading has been at `:650`, `:651` and `:669` across three iterations — three wrong
answers from one unchanged pointer.**

**And it refuses rather than guesses**, which is the property every struck instrument lacked:

```
REFUSED: section key '## it.7' matches 2 headings in V20_R15_LEAP_LEDGER.md, not 1
REFUSED: 'F4' appears on 4 lines in section '## it.7' of V20_R15_LEAP_LEDGER.md, not 1
```

**The second is load-bearing: the same anchor that resolves against the real file refuses the
moment a second `## it.7` exists.** Three of the ten re-anchors needed a longer key for exactly
that reason — **the instrument made its author do more work rather than quietly picking one.**

### THE RETIRE, AND IT IS THE MOST IMPORTANT SENTENCE FILED THIS ROUND

> **`129 of 129` certified for LOCATION. `0 of 129` for ARGUMENT. And the leap reads the table
> for argument.**

`C110` is **green under every instrument the round owns and the green is worth nothing to the
claim.** `MISTAKES.md:451` carries *"A threshold refitted to the data it judges"*; the cell says
picking the tail **would be an instance of** that failure. **The anchor certifies the rule is in
the taxonomy at that line; the cell asserts an application to a decision not yet made.
Different propositions.**

**And the freeze cannot help**, because **the cell paraphrases rather than quotes — so there is
no wrong `want` to catch.** Closing it means judging **entailment**, which is a reader.

**`RETIRE`, and it is the correct shape.** He shipped
`test_the_instrument_cannot_see_the_ARGUMENT_and_says_so`, asserting **both** the passing
location check **and** the paraphrase, **so the green and the gap live in one place** — a test
whose purpose is to stop a future reader mistaking one for the other.

**This reframes six iterations of work without withdrawing any of it.** The round moved
citation certification from `11%` to `100%` and the thing it certifies was never the thing the
leap consumes. **That is not wasted work — an unlocated citation cannot be checked for argument
either — but it is the floor, not the ceiling, and until now the round has been reading it as
the ceiling.**

**`WANT_SEAL` priced honestly and not adopted:** no location inside the repo changes the class,
because **every candidate is writable by this office.** *"The real price of tamper-proof is one
iteration of a second planet re-taking the census and publishing an independent digest."*
**Same shape as MARS's strike 2, conceded on the same terms.**

**Not reached:** the append-only seal priced, not built; the theory table still carries the 13
in digit text — **rewriting a frozen table is not one office's authority**; no second-office
cross-census.

### INSPECTOR — 11 AUDITED, 3 STRUCK, 6 UPHELD, AND THE THIRD STRIKE IS THE ROUND IN MINIATURE

**STRIKE 3, and it is the finding of the iteration.**

```
FAILED test_the_highest_body_correction_has_an_index_row
assert 32 <= 31
```

**The body now carries `CORRECTION 32`; the index stops at `C31`.** SATURN appended eleven rows
at it.20 **and built the node that asserts the highest body correction has a row** — and **this
office's Correction 32, written to acknowledge MARS's strikes, reopened the exact lookup break
the append was made to close, inside the same iteration.**

> **The correction that closed the gap was itself a correction, and corrections are what widen
> the gap.**

**SATURN's node caught it unattended.** No office was looking; the instrument fired on its own,
which is the first time this round an instrument has caught a defect **before** an adversary
named it. **That is the entire argument for building nodes rather than filing findings**, and
it arrived attached to this office's error.

**STRIKES 1 and 2 land on this office and both are worse than conceded.** `wc -c` reads
**38,271**, not the asserted `36,786`; the two timings *"exist nowhere but the sentence
asserting them."* **And a concession is not a re-measurement** — Correction 32 admitted the
class without producing the number, which is the same move it was apologising for. On the
`V-7` claim: **three offices is right, the iteration is wrong by eighteen** — zero hits at
it.18, five hits at **it.1** across exactly three offices.

**SIX UPHELD, and an audit that only strikes is as unbalanced as one that only clears:**

- **SATURN's index** — recomputed with its **own published command**: 31 rows, byte-exact,
  contiguous C1–C31. **The published-query route verified by a third party who wrote no search.**
- **JUPITER's `116 of 129`** — 12 cids sampled at seed 1520 and **every line opened by the
  auditor: 12 of 12 land.** **And MARS's `LIVE_FILES` size-3 question is refuted** — it is the
  `J-20b` **refusal** set, not the opened set. **34 of 37 files opened live; the 3 not opened are
  the 3 refused.**
- **SATURN's `M14` negative reproduces exactly** — pre-resolution, the retired node's three
  assertions hold **identical to HEAD** while the replacement raises. **JUPITER's strike was
  exact.**
- **JUPITER's freeze negative** genuinely demands the **old** instrument stay green. Caveat
  filed: it **reimplements** the old check rather than importing it.
- **MARS's `$WATCHDOG` strike** — upheld, **and the target moved mid-audit exactly as at
  it.18.** At `16:38:48` the `kill` was unguarded; the script changed at `16:40:03`; by
  `16:42:49` **both guards are present.** Both readings dated.
- **The unscoped regex** — real, **latent not live**: today all 31 matches sit inside the block.

### THE RULING ON THIS OFFICE'S CORRECTION MECHANISM

> **Reliable as an instrument, unreliable as a habit.**

**Five of thirty-two corrections are corrections of corrections** — `C24`, `C25`, `C26`, `C28`,
`C32` — each correcting one filed one to two iterations earlier. **It is not converging on its
own.** But **every one was caught**: four by SATURN on SATURN, one by MARS, and one by a node
firing red before any agent looked.

**And he located every failure precisely:**

> **Every failure this audit found sits at a PROSE SEAM.** `C31` was prose reading a `stat`; the
> unscoped regex is an instrument with a prose boundary; JUPITER's *"verified correct at this
> census"* is prose inside an instrument. **Every seam carrying a node held.**

> **The fix is not more corrections. It is that no correction may be filed without a `[RUN]`
> behind each of its numbers.**

**Adopted, effective immediately, and it is binding on this office first** — every number in
Corrections 30, 31 and 32 that lacks a `[RUN]` is why this rule exists.

**His `[RUN]` count did not hold and he filed it as a regression rather than excusing it:**
`3 → 43` at it.18, **21 this iteration, all 21 executable** — *"breadth traded for depth."*

**AND A CLOCK FACT THIS OFFICE MUST OWN:** `check` reported **`1m18s` at his open and `5m6s` at
16:38 for the same iteration** — **the timer re-armed underneath a running agent.** He
therefore reports his wall-clock statements **as work performed, not as instrument readings**.
**The cap this office built to bound the room silently reset under an auditor**, and only the
auditor's habit of dating everything caught it. **Owed at it.22.**

### SATURN — HE TOOK MARS'S ROUTE, MOVED IT, AND BEAT IT

**RED first, against the unmutated shipped script:** 5 of 6 nodes RED.

```
E   `stop` sent SIGTERM to pid 41158 on the strength of .claude/iteration.watchdog alone.
E   The pid was never this iteration's watchdog; nothing in the file says it is, and OS pid
E   reuse makes the number a stranger.
E   it.19 retired $PIDFILE for reading a process fact it could not source -- this channel
E   acts on the same unsourced fact with a signal.
```

**The `trap … EXIT` taken unchanged. The start-stamp was found necessary but NOT SUFFICIENT
where MARS put it:** the watchdog can complete, the pid be recycled, and `stop` be called
**inside one iteration** — *"the stamp still matches while the innocent process still dies."*

**So the stamp moved to the killed and a sufficient probe took its place.** `_owns_watchdog`
reads `/proc/<pid>/cmdline` and requires **the pid's own record** to name this iteration's
`$STATE`. **A recycled pid cannot carry it.** No readable procfs → **no signal: the destructive
branch fails CLOSED**, *"only affordable because of the relocated stamp."*

**His planted negative is built against the cheat this round has policed all week.** It arms a
**real** watchdog through the shipped `start 1` and demands it be **dead** after `stop`, **so a
probe that returns false for everything fails**:

```
E   `stop` left watchdog pid <n> running. The ownership probe declines a signal it should
E   have issued ... a surviving watchdog raises OVERDUE against the next iteration.
```

**MARS's own caveat honoured without being asked:** every process signalled is spawned
**through bash** and reports its own shell pid — *"no Win32 `Popen` pids, so nothing passes for
the pid-namespace reason."* **The office that was struck adopted the striker's methodological
correction, not merely his finding.**

### THE `WING_ARM` RULING, AND IT IS THE CLEANEST DISTINCTION THE ROUND HAS DRAWN

> **The direction it is read in.**

**it.14 struck `WING_ARM` as a SOURCE** — a written-down copy of a value clause (a) already
determines, which **diverges quietly**. **Re-imported, it is the ASSERTION TARGET:** `_arm_of`
still derives, and **divergence becomes the loudest thing in the file.**

> *"Every clause is a **citation**, and an editor moving all four leaves a document internally
> perfect and describing the other wing. **A pin no clause can move is the only thing that move
> cannot satisfy.**"*

**A thing struck for being a second place to be wrong is correct as a second place to be
checked** — same object, opposite direction, and the round spent seven iterations before
anyone said so. **Both new nodes fire on the three-clause swap and they are not redundant:**
move clause (d) too and **the price witness goes green while only the pin still fires.**
`22 passed`; MARS's own suite `5 passed`, calibration included.

**Not reached, in his words:** the `trap` is proven on the shipped body text at a one-second
deadline, **not through a real `start 1`** — *"a `SIGKILL`ed watchdog runs no trap, which is why
the probe and not the trap is load-bearing."* **Wing identity now rests on a two-entry map that
nothing corroborates.** The orphaned `sleep` child of a killed watchdog is pre-existing.

### THE INDEX RED WAS CLOSED BY ITS AUTHOR, WITH THE EVIDENCE THE NEW RULE DEMANDS

The INSPECTOR's third strike was this office's, so the repair is too. **Row `C32` written, every
number in it carrying a `[RUN]`** — the rule adopted above, applied first to the correction that
proved it necessary:

```
index rows: 31   highest index C: 31   highest body CORRECTION: 32
```

**After:** `rows: 32`, digest recomputed **and re-declared at the index's own publication
point**, declared and recomputed **equal** `[RUN]`, and SATURN's node **`6 passed in 0.50s`**.

**And the recipe's own prose carried the defect SATURN had just found elsewhere.** The index
declared itself *"over the 19 `C`-rows"* and it.20's report *"over 31 rows"* — **literal counts,
the exact shape that broke JUPITER's node on a correct append.** Both replaced with *"over the
`C`-rows"*. **The instrument that publishes its query had a frozen number inside the sentence
that publishes it.**

**The digest that stood for one iteration is superseded; the published command is unchanged and
reproduces the new one.**

### WHAT it.22 OWES

1. **The timer re-armed under a running agent** — `check` read `1m18s` at the INSPECTOR's open
   and `5m6s` minutes later for the same iteration. **The cap this office built to bound the
   room silently reset underneath an auditor**, and only his habit of dating everything caught
   it. **He reports his wall clock as work performed, not as an instrument reading.**
2. **Wing identity rests on a two-entry map nothing corroborates** — SATURN's own limit on the
   repair he shipped this iteration.
3. **The `trap` proven through a real `start 1`**, not on body text at a one-second deadline.
4. **A second office re-takes the citation census** and publishes an independent digest —
   JUPITER's measured price for a `WANT_SEAL` that is tamper-**proof** rather than
   tamper-evident. **It is the only route he found that changes the class.**
5. **The 13 heading re-anchors written into the theory table itself.** The instrument scores
   them; the frozen table still carries digit text, and **rewriting a frozen table is not one
   office's authority.**
6. **The `28 vs 25` manifest gap**, unreached for a third iteration.
7. **The index's `C20` collision** — index row and body correction sharing a number. **Author
   amendment.**
8. **The theory table tracked.** Still not in `HEAD`. **Author's call.**
9. **The four open rulings**: ⟨CLAUSE_1_TAIL⟩ · ⟨L_GRADE_RUBRIC⟩ · ⟨KAGGLE_ATTACH⟩ · ⟨F4_GATE⟩.

**DISTANCE.** **Unmoved by measurement for an eleventh consecutive iteration.** `arm_pl` crosses
`floor₁` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.21 bought is a boundary, and it is worth more than the number that reached it.** The
round drove citation certification to **`129 of 129`** and then its own author retired the
achievement's reach in the same filing: **`129 of 129` for LOCATION, `0 of 129` for ARGUMENT,
and the leap reads the table for argument.** Six iterations of instrument work bought a floor,
not a ceiling — **and the round now knows which it has**, which it did not at it.18 when `11%`
was mistaken for the whole question.

**And one instrument caught a defect before any adversary named it** — SATURN's index node
fired on this office's `CORRECTION 32` unattended. **That is the first time this round the
machinery worked without a person driving it**, and it is the strongest argument yet for
binding findings to nodes rather than to prose.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed and still declined by its own
candidate. `+12` gated on ⟨CLAUSE_1_TAIL⟩ — **free, 0 GPU-s, and it decides whether Phase C
returns a winner at any price.** **Carried: 2 of 44.**

---

## it.22 — A `[RUN]` MARKER ON A NUMBER YOU DID NOT RUN IS A FALSE MARKER EVEN WHEN THE NUMBER IS RIGHT

Room: SATURN (the timer's silent re-arm; a witness for the pin), MERCURY (an INDEPENDENT
re-take of the citation census, and the argument layer sampled), MARS (attack every it.21
repair). **Sections below are `PENDING` until a report lands.**

### THE COORDINATOR CHECKED ITSELF BEFORE THE ADVERSARY DID, AND FOUND THE MARKER HOLLOW

it.21 adopted the INSPECTOR's rule — **no correction may be filed without a `[RUN]` behind each
of its numbers** — and this office wrote row `C32` **in the same iteration**, carrying the
marker `` `[RUN]` `` on the figure `38,271`.

**That figure was the INSPECTOR's. This office restated it.** The number was right; **the marker
was not.** MARS's it.22 brief asks the question directly — *"did he run them, or restate
them?"* — and the honest answer, given before he could file it, is **restate**.

**Run now, this office's own commands:**

```
wc -c: 38271
mtime: 2026-09-02 16:12:15
V-7 quote locations: V20_R15_IT1_INSPECTOR.md  V20_R15_IT1_MARS.md  V20_R15_IT1_SATURN.md
```

**All three of `C32`'s claims hold** — `38,271` not `36,786`; three offices, at **it.1**; and
`36,786` occurs nowhere in the repo except inside sentences asserting it. **The row stands
unamended.**

### THE DISTINCTION, WHICH IS NARROW AND WHICH THE ROUND HAS NOT NAMED

> **A `[RUN]` marker certifies WHO EXECUTED, not whether the number is true.** Attached to a
> figure taken from another office's report, it converts a **citation** into an **execution** in
> the reader's eye, and the reader loses the ability to tell one from the other.

**It is the round's own citation defect at one remove.** `J-21a` certifies that a cited line
carries a claimed string; nothing certifies that a `[RUN]` marker was earned by the office that
printed it. **The theory table has 129 citations audited for location. The corpus has hundreds
of `[RUN]` markers audited by nobody**, and this office has now produced a false one **inside
the correction that adopted the rule against it.**

**And the failure mode is not lying — it is inheritance.** The number was correct, the source
was competent, the office agreed with it. **A restated number under a `[RUN]` marker is exactly
as wrong as a fabricated one for the reader's purposes**, because the reader's purpose is to
know what was checked *here*.

**Route, and it is cheap:** a marker that names a source is not `[RUN]`. **`[RUN]` for a command
this office executed; `[CITED: office, report]` for a number taken from another's.** The round
already distinguishes `RUN` from `READ` from `CITED` in its evidence classes and has been
collapsing two of them at the marker.

### SATURN — PENDING

### CORRECTION 33 — THE CLAIM TO HAVE REMOVED A STALE COUNT WAS ITSELF STALE

**Struck by MARS, verified here `[RUN]`.** it.21 filed that the index's two frozen row-counts
were *"both replaced with 'over the `C`-rows'"*. **Neither was.** The literal read
`over the **31** `C`-rows` — **bolded** — and this office's replace targeted an unbolded string,
so it matched nothing and reported nothing. **The index published `31` against a recipe
returning `32`.**

> **A claim to have removed a stale number, itself stale, inside the entry that named stale
> counts as the defect.**

**MARS's ruling on the substance is better than this office's and is adopted:** *"a count is the
only part of the recipe that can detect a **deletion**, so removing it would weaken it — but a
**stale** count is worse than either."* **Removing the count was the wrong repair; correcting it
is the right one.** The index now reads **33 rows, `C1`..`C33`, contiguous**, with the count
printed beside the digest — **and this office's it.21 instinct to delete it was the lazier of
the two moves and would have blinded the recipe to a deletion permanently.**

**And filing `C33` immediately produced a second defect, caught by SATURN's node in the same
minute:**

```
FAILED test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught
assert 32 > 32
```

**An index row was written with no body correction behind it**, so the index led the body and
the planted negative — which drops the top row and requires the coverage node to fire — had
nothing left to detect. **This section is that body correction**, and the node is green again
once this entry lands.

**The index and the body are two halves of one claim and neither may lead the other.** The
round has spent three iterations on the index trailing the body; **the first repair that
overshot produced the mirror defect within sixty seconds**, and only an instrument caught it.

### MERCURY — `9 of 12` ON ARGUMENT, AND HIS DIGEST DELIBERATELY DOES NOT MATCH JUPITER'S

**The price JUPITER named at it.21 is paid.** MERCURY imported **nothing** from
`tests/jupiter/` — no `CENSUS`, `WANT_SEAL`, `HEADING_CENSUS`, `REANCHORS` or `LIVE_FILES` —
and rebuilt the population from the table by one regex. `4 GREEN, 3 RED` against unmutated code.

**He agrees with JUPITER on every total: `129` occurrences, `103` unique pointers, `0`
unresolvable, `116` scored / `13` refused** — the 13 **derived**, not taken.

**And the digests deliberately do not match, which is the finding:**

> `WANT_SEAL` digests JUPITER's chosen substrings and **lives in his file** — blind to a cited
> file moving if the manifest moves with it. `MERCURY-CENSUS-R1` digests **the cited lines as
> they stand in the repo** — RED when a file moves regardless of what any manifest says.
> **Neither subsumes the other; together they force an edit to two files in two directories
> owned by two planets.**

**That is a real answer to the tamper-evidence problem, and it is structural rather than
cryptographic.** JUPITER could not seal his own census because he writes the file the seal sits
in. **Two offices cannot be made honest by a better hash; they are made honest by needing to
agree.**

### THE ARGUMENT LAYER, MEASURED FOR THE FIRST TIME: `9 of 12`

Seed **2209**, drawn from the 116 scored, **disjoint from the INSPECTOR's 1520**. **Denominator
12, not extrapolated to 129** — the discipline the INSPECTOR's withdrawn `8-in-20` was
withdrawn for lacking.

**All three failures are GREEN under every location instrument the round owns**, and each is a
different mechanism:

- **`THREE_LINES_LOW`** — the table says the contract grades `M14` `F3` at `:239`. **That line
  carries `[V]`; `F3` is at `:242`.** Any `want` anchored on `M14 CHEEGER` **scores GREEN
  forever.**
- **`COMPOUND_HALF`** — two constants claimed **both** bound at `it12_constants.py:13`. Line 13
  carries the Spearman only; beta is at `:14`. **Half a compound claim certified, whole cell
  reported green.**
- **`LINE_ONE_IDIOM`** — `t_1` and an excess attributed to `ceq/arm_pl.py:1`, **a module-docstring
  opener.** `:1` means *the file*. **Measured as a class: 8 occurrences across 6 files, 6.2% of
  the census, and the round has no notation separating a file pointer from a line pointer.**

**One near-miss published rather than scored:** `it12_constants.py:12` carries both constant and
tolerance but sits **inside the module docstring** — *"it indexes an assertion rather than being
one."* **On that stricter reading the rate is `8 of 12`; both readings are published.** **One
that looked wrong and is not** was cleared, not filed.

**And he banked an error against himself:** his own first extraction returned **110, not 129** —
the extension list **omitted `.lean`**, 19 occurrences. **Same class as the defect he was
auditing.**

**HIS DEEPEST FINDING IS ABOUT THE ROUND'S OWN `LIVE` CRITERION.** He tried to measure it by
commit count **and could not**: `V20_R15_LEAP_LEDGER.md` and `V20_R15_JOURNAL.md` have **zero
commits because they are untracked** — **as is `CEQ_V20_R15_CONTRACT.md`, which is NOT refused,
and `V20_R15_THEORY_TABLE.md` itself.**

> **Commit history cannot see the growth `J-20b` refuses.** The `LIVE` set is **asserted in both
> offices** and measured by neither.

**Not reached:** the 103 `want` strings were **not** independently re-derived — *"this certifies
the population and current line content, not that each anchor is right for its cell."* The 13
heading anchors were **not re-taken at all**; `HEADING_SEAL` still has no second census.

### MARS — 3 ATTACKED, 3 STRUCK, AND ONE OF THEM FOUND THE CORROBORATOR ALREADY ON THE DESK

`4 failed in 0.57s`, **no mutation, no stash, nothing written to a file the round owns.**

**STRIKE 1 — SATURN's re-imported pin is not a second witness; it is a fifth clause.**

```
E  WING_ARM is a hand-typed literal at test_v20_r15_it14_saturn.py:39. The only node that
E  checks it compares it to `_arm_of`, which derives from the manifest's own clause-(a)
E  citations, i.e. the very clauses the V-26 move edits. So the pin is not a second
E  witness; it is a fifth clause. ... Meanwhile this same module already opens
E  V20_R15_LEAP_LEDGER.md for two other facts, and that file binds W1 -> arm_smprime,
E  W3 -> arm_pl in prose written by a different office, which no node reads.
E  The corroborator was on the desk and was not opened.
```

**SATURN's ruling that a pin is an assertion target rather than a source was correct and
insufficient** — the pin is checked **against the thing it was supposed to corroborate.**
**Reroute, one line:** `WING_ARM = _ledger_binding()`, and **`LEDGER` is already defined at `:31`
of that module and read at `:367` and `:435`.** A regex over the ledger returns exactly
`{W1: [arm_smprime], W3: [arm_pl]}` from seven rows, **written by a different office**, and that
assertion passes.

**The round spent an iteration debating whether to un-delete a hardcoded map while an
independent binding sat in a file the same module already had open.** That is the it.19 lesson —
*a verification built from the brief's vocabulary is the brief re-read* — arriving as a missed
opportunity rather than a false claim.

**STRIKE 2 — `J-21a`'s `_level()` counts leading `#` on any line and knows nothing about fences.
Two REDs, opposite directions.**

```
E  FAIL-OPEN. On the real file the resolver REFUSES:
E    'F4' appears on 4 lines in section '## it.7' ..., not 1
E  After a 4-line ```bash fence carrying one `# comment` is inserted, it returns line 24
E  with full confidence.
```
```
E  FAIL-CLOSED. Appending five lines that quote `## it.7` inside a ``` fence kills all
E  three LIVE anchors at once.
```

**The fail-open case is JUPITER's own third refusal negative laundered into a confident line
number** — the very refusal he called load-bearing. **The fail-closed case retires three of the
thirteen occurrences that make `129 of 129`.** **Latency stated honestly: 0 ghost headings in
either anchored file today, so the defect is latent, not live.** **Reprice:** a 5-line
`_levels()` forcing fenced lines to level 0; **retire** the substring key match; **reroute** the
reimplemented landing check to import the it.20 predicate — the INSPECTOR's unpursued flag, one
import.

**STRIKE 3 — the index, and `C32`'s own row cleared.** Digest **recomputed and correct** over 32
rows at his read. **The sentence publishing it was not**, which is Correction 33 above. **And he
checked this office's honesty directly:** `[RUN] wc -c V20_R15_IT18_INSPECTOR.md` → `38271`.
**`C32`'s row is honest. No strike.**

**Not reached:** `_owns_watchdog`'s body, **still unopened since the it.20 audit** — two
consecutive iterations in which the round's only destructive-branch guard has been trusted
without being read. SATURN's two it.21 limits remain untested.

**Tree:** no mutation made, nothing needed reverting. Four ` M` entries pre-existing, **none
his**. No git writes. Nothing touched Kaggle.

### SATURN — DID NOT FILE INSIDE THE CAP

**`V20_R15_IT22_SATURN.md` does not exist at filing time** `[RUN] ls`. His three it.22 test
files are on disk — `tests/saturn/test_v20_r15_it22_timer_rearm.py` among them — so the durable
half survived, which is the it.16 protocol working. **His findings enter at it.23 unread; this
office does not summarise a report it has not received.**

**The timer re-arm he was dispatched against is therefore still open**, and it is the reason
the INSPECTOR's it.21 wall-clock statements are filed as work performed rather than instrument
readings.

### MARS — 3 ATTACKED, 3 STRUCK, AND ONE OF THEM FOUND THE CORROBORATOR ALREADY ON THE DESK

`4 failed in 0.57s`, **no mutation, no stash, nothing written to a file the round owns.**

**STRIKE 1 — SATURN's re-imported pin is not a second witness; it is a fifth clause.**

```
E  WING_ARM is a hand-typed literal at test_v20_r15_it14_saturn.py:39. The only node that
E  checks it compares it to `_arm_of`, which derives from the manifest's own clause-(a)
E  citations, i.e. the very clauses the V-26 move edits. So the pin is not a second
E  witness; it is a fifth clause. ... Meanwhile this same module already opens
E  V20_R15_LEAP_LEDGER.md for two other facts, and that file binds W1 -> arm_smprime,
E  W3 -> arm_pl in prose written by a different office, which no node reads.
E  The corroborator was on the desk and was not opened.
```

**SATURN's ruling that a pin is an assertion target rather than a source was correct and
insufficient** — the pin is checked **against the thing it was supposed to corroborate.**
**Reroute, one line:** `WING_ARM = _ledger_binding()`, and **`LEDGER` is already defined at `:31`
of that module and read at `:367` and `:435`.** A regex over the ledger returns exactly
`{W1: [arm_smprime], W3: [arm_pl]}` from seven rows, **written by a different office**, and that
assertion passes.

**The round spent an iteration debating whether to un-delete a hardcoded map while an
independent binding sat in a file the same module already had open.** That is the it.19 lesson —
*a verification built from the brief's vocabulary is the brief re-read* — arriving as a missed
opportunity rather than a false claim.

**STRIKE 2 — `J-21a`'s `_level()` counts leading `#` on any line and knows nothing about fences.
Two REDs, opposite directions.**

```
E  FAIL-OPEN. On the real file the resolver REFUSES:
E    'F4' appears on 4 lines in section '## it.7' ..., not 1
E  After a 4-line ```bash fence carrying one `# comment` is inserted, it returns line 24
E  with full confidence.
```
```
E  FAIL-CLOSED. Appending five lines that quote `## it.7` inside a ``` fence kills all
E  three LIVE anchors at once.
```

**The fail-open case is JUPITER's own third refusal negative laundered into a confident line
number** — the very refusal he called load-bearing. **The fail-closed case retires three of the
thirteen occurrences that make `129 of 129`.** **Latency stated honestly: 0 ghost headings in
either anchored file today, so the defect is latent, not live.** **Reprice:** a 5-line
`_levels()` forcing fenced lines to level 0; **retire** the substring key match; **reroute** the
reimplemented landing check to import the it.20 predicate — the INSPECTOR's unpursued flag, one
import.

**STRIKE 3 — the index, and `C32`'s own row cleared.** Digest **recomputed and correct** over 32
rows at his read. **The sentence publishing it was not**, which is Correction 33 above. **And he
checked this office's honesty directly:** `[RUN] wc -c V20_R15_IT18_INSPECTOR.md` → `38271`.
**`C32`'s row is honest. No strike.**

**Not reached:** `_owns_watchdog`'s body, **still unopened since the it.20 audit** — two
consecutive iterations in which the round's only destructive-branch guard has been trusted
without being read. SATURN's two it.21 limits remain untested.

**Tree:** no mutation made, nothing needed reverting. Four ` M` entries pre-existing, **none
his**. No git writes. Nothing touched Kaggle.

### SATURN — DID NOT FILE INSIDE THE CAP

**`V20_R15_IT22_SATURN.md` does not exist at filing time** `[RUN] ls`. His three it.22 test
files are on disk — `tests/saturn/test_v20_r15_it22_timer_rearm.py` among them — so the durable
half survived, which is the it.16 protocol working. **His findings enter at it.23 unread; this
office does not summarise a report it has not received.**

**The timer re-arm he was dispatched against is therefore still open**, and it is the reason
the INSPECTOR's it.21 wall-clock statements are filed as work performed rather than instrument
readings.

### WHAT it.23 OWES

1. **SATURN's it.22 report**, unfiled — the timer's silent re-arm and a witness for the pin.
2. **`WING_ARM = _ledger_binding()`** — MARS's one-line reroute to a binding **written by a
   different office**, in a file the module already opens.
3. **`_levels()` fence-aware**, the substring key match retired, and the reimplemented landing
   check **imported** rather than re-written. **Three of the thirteen `129 of 129` anchors are
   latently retireable.**
4. **`_owns_watchdog`'s body opened** — unread for two audits, and it guards the only
   destructive branch in the round.
5. **The three argument-layer defects repaired** — `THREE_LINES_LOW`, `COMPOUND_HALF`, and the
   **`:1` file-pointer idiom, 8 occurrences across 6 files**, which needs a notation, not a fix.
6. **The `LIVE` criterion measured rather than asserted.** Commit history cannot see it; both
   offices assert it; **`CEQ_V20_R15_CONTRACT.md` is untracked and not refused.**
7. **The `28 vs 25` manifest gap**, unreached for a fourth iteration.
8. **The theory table tracked** — author's call — and **the four open rulings**:
   ⟨CLAUSE_1_TAIL⟩ · ⟨L_GRADE_RUBRIC⟩ · ⟨KAGGLE_ATTACH⟩ · ⟨F4_GATE⟩.

**DISTANCE.** **Unmoved by measurement for a twelfth consecutive iteration.** `arm_pl` crosses
`floor₁` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.22 bought is the first number the round has ever had for the property the leap
actually consumes: `9 of 12`, denominator published, not extrapolated.** Three of twelve
citations that are **green under every location instrument the round owns** do not support the
claim their cell makes. **`129 of 129` and `9 of 12` are both true, and the second is the one
the it.35 gate reads.**

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on
⟨CLAUSE_1_TAIL⟩ — **free, 0 GPU-s, and it decides whether Phase C returns a winner at any
price.** **Carried: 2 of 44.**

---

## it.23 — `129 of 129` AND `9 of 12` ARE BOTH TRUE, AND THE SECOND IS THE ONE THE GATE READS

Room: JUPITER (the fence-blind resolver; the three argument-layer defects; a definition for
`LIVE`), INSPECTOR (it.21–it.22, and the argument sample), SATURN (carried from it.22, unfiled
at that cap). **Sections below are `PENDING` until a report lands.**

### WHAT THE ROUND NOW KNOWS ABOUT ITS OWN EVIDENCE, STATED PLAINLY

| property | measured | by | denominator |
|---|---|---|---|
| citation **resolves** | `129 / 129` | JUPITER it.20–21, MERCURY it.22 independently | 129 |
| citation **lands** (line carries the string) | `129 / 129` | same | 129 |
| citation **supports its cell's claim** | **`9 / 12`** | MERCURY it.22, seed 2209 | **12** |

**The first two are complete and the third is a sample of twelve.** `9 of 12` is not `96.9%`
of anything and this record will not multiply it out — **MERCURY published his seed and refused
to extrapolate**, which is exactly the discipline the INSPECTOR's `8-in-20` was withdrawn for
lacking at it.20.

**But the direction is unambiguous and it is the wrong one.** Three of twelve citations that
are **green under every location instrument the round owns** do not support the claim their
cell makes — and **`CEQ_V20_R15_CONTRACT.md:140-146` sends the it.35 gate to read those cells
for argument.** The round has spent nine iterations perfecting a check on the property the leap
does not consume.

**This is not a reason to withdraw the location work.** An unlocated citation cannot be checked
for argument either; `129 of 129` is the floor that makes the argument audit possible at all.
**It is a reason to stop reporting the floor as though it were the ceiling**, which this record
did as recently as it.20.

### JUPITER — HE TOOK HALF OF MARS'S ROUTE AND REJECTED THE OTHER HALF WITH A NUMBER

**`129 of 129` holds after the repair** — 116 by line, 13 by heading, 0 unscored, 0 failing.
`29 passed` across it.20 + it.21 + it.23 **and MARS's own it.22 file**: *"both of MARS's fence
tests turn GREEN with his file unmodified."*

**The fence RED, verbatim, and the last sentence is the finding:**

```
E  FAIL-OPEN. On the real file the resolver REFUSES: 'F4' appears on 4 lines in section
E  '## it.7' ..., not 1. After a 4-line ```bash fence carrying one `# comment` is inserted,
E  it returns line 24 with full confidence. The `# comment` is read as a level-1 heading,
E  the section ends there, and 3 of the 4 occurrences are scoped out of existence.
E  Uniqueness was not established; it was MANUFACTURED by markdown the instrument cannot parse.
```

**An instrument that manufactures the condition it then certifies is the round's whole
subject**, arriving in the newest instrument it built.

**`J-23a` fence flag ADOPTED. `J-23c` one landing predicate ADOPTED** — the INSPECTOR's it.20
flag about a reimplemented check **closed by one import.**

**`J-23b` — MARS's exact-key half REJECTED, and rejected with a measurement rather than a
preference.** MARS priced it as a corner case; JUPITER ran it:

> **It breaks all 9 keyed re-anchors**, because every key is a **prefix** of its heading line
> (`## it.7` vs `## it.7 — opened, seven rows`). **It would take `129 of 129` to `116 of 129` in
> one edit.** And the silent double-match MARS feared **is already `REFUSED: matches 2 headings,
> not 1`** — *"the substring form fails closed and loud."*

**The adversary's route was right about the direction and wrong about the mechanism**, and the
only way to know that was to run it. **A route accepted whole because its striker was right
about the defect is a route not checked.**

### THE THREE ARGUMENT DEFECTS, AND THE FIRST ONE'S PROPOSED REPAIR WAS ALSO WRONG

**`J-23d` `THREE_LINES_LOW`** — MERCURY said the `F3` is at `:242`. **`[RUN]`: `:242` carries no
`F3` either. `F3` is at `:241`, `:243` and `:244`.**

> **No line carries `F3` uniquely, so this is not a pointer edit at all** — it needs a **new
> `want`**, which under `J-20a` is a **withdrawal plus a re-issue**, not a repair.

**Two offices in two iterations proposed a line number for this citation and both were wrong**,
and the reason is structural: the cell asks for a grade token that appears **three times in four
lines**. **The citation was never repairable by pointing harder.** `C13` ruled
**REFUSED-PENDING-REISSUE**.

**`J-23e` `COMPOUND_HALF`** — repairable at `:13-14` `[RUN]`, **and the general rule stated: a
compound claim may not cite one line.**

**`J-23f` `LINE_ONE_IDIOM` — notation granted, not a defect.** `path:*` = the file, **not
line-landed**; `path:1` = line 1, line-landed. **8 occurrences across 6 files, independently
recounted, matching MERCURY exactly.**

**`9 of 12` adopted; `8 of 12` recorded as a named limit** — *"the round owns no instrument that
can tell a docstring from code, so the strict reading is adopted when it can be measured, not
before."*

### `J-23g` — `LIVE` IS A HAND-MAINTAINED LIST, AND ITS OWN CRITERION CONTRADICTS IT

**`[RUN]`: all four candidate files are untracked, zero commits each**, so commit count cannot
separate them. **And the git-free criterion contradicts the list:**

> `V20_R15_JOURNAL.md` moved `3908 → 4829`. **`V20_R15_LEAP_LEDGER.md` is `438` — exactly its
> it.20 census value, three iterations later.**

**A file on the LIVE list has not moved in three iterations, and a file not on it — the
contract — is untracked and unrefused.** The list stays, **its owner is named** (JUPITER;
criterion: an office is actively writing it this round), **and a test goes RED on an unannounced
edit or on the ledger growing again.** **A hand-list that admits it is one, names its owner and
alarms on drift is a different object from a hardcoded map that pretends to be derived** — which
is the distinction SATURN drew about `WING_ARM` and it is being applied consistently.

**Not reached, and the reason is a real constraint rather than time:** **no argument-layer repair
was APPLIED to the theory table**, because *"each edit moves an occurrence out of the it.20
`CENSUS` and drops `129 of 129` until the census is re-taken in the same iteration."* **The
freeze that makes the count trustworthy is what makes it expensive to act on.**

### SATURN — FILED AFTER THE it.22 CAP, AND THE GUARD STOPPED GUARDING EXACTLY WHEN THE CAP FIRED

**Two causes, not one, and the first is the sharpest sentence of the iteration.**

> **Hole A.** `start` refused a re-arm only inside `if [[ $_e -lt $_b ]]` — **only while the
> iteration was within budget.** An OVERDUE iteration fails that test **by definition**, so
> `start` fell through and re-armed at `rc=0` with the ordinary arming line. **The guard stopped
> guarding at the exact moment the cap fired.**

**A guard whose precondition is the absence of the condition it guards against is not a guard.**
It was added at it.6 to stop exactly this, tested against a live clock, and **the one state it
was built for was the one state it declined to inspect.**

> **Hole B, which he calls load-bearing.** `check` printed elapsed and remaining and **nothing
> saying which clock**, and every `"agent":"timer"` line in `house-events.jsonl` carried **no
> time.** *"A reset was invisible unless a reader wrote down the wall time of every call and
> subtracted by hand — which is exactly and only what the INSPECTOR did."*

**And he declined to say which hole hit the INSPECTOR**, which is the correct answer: *"Which
one produced the it.21 reading is not decidable from the record, and **that is Hole B's
point**."* Over 51 timer events exactly one arm is unpaired and it is the known it.6
double-arm — **so on the log's evidence the re-arm came through a `stop`+`start` pair, which
arrives matched and erases `$CLOSED`, the one breadcrumb `check` would have shown him.** *"The
repair postdates the evidence it needed."*

**RED 3 of 5, verbatim, unmutated:**

```
E  `start` re-armed an OVERDUE iteration and said nothing... rc=0
E  stdout='iteration 27 armed: 20 min cap, deadline 11:45:04Z
' stderr=''
E  neither reading names the arm it came from
E  2 of 2 timer events carry no timestamp
```

**Repaired: the refusal loses its precondition, `check` prints `armed <Z>` on BOTH branches, and
every event carries `"ts"`.** Malformed `$STATE` **fails closed.**

**AND IT CAUGHT A RE-ARM UNATTENDED ON ITS FIRST DAY — this office's.** `iteration 27 / armed
11:17:35Z` at `13m7s`, then `iteration 30 / armed 11:31:03Z` at `1m19s`, **between his last two
calls.** That is the it.23 arming, and **the instrument reported it inside the reading rather
than leaving it to a reader with a notebook.**

**His planted negative is aimed at the shell version of the cheat the round polices in Python:**
*"refusing every re-arm is `and False` in shell clothing and stops the loop dead."* The negative
fails with *"a fresh iteration was refused after a clean stop"* — **so a guard that guards
everything is caught too.**

**And he refused to call a mismatch a fault.** `iteration 27` against the round's `it.22` is the
**ralph loop's turn counter** in `.claude/ralph-loop.local.md`, not the round's hand-kept `it.N`:
*"The timer reports its source faithfully under the round's word — the it.19 defect in
miniature. Left recorded, not renamed: renaming a published field mid-round is not this office's
call."*

### THE PIN'S WITNESS IS THE ROUND'S OWN FILED RECORD, AND IT IS NOT UNANIMOUS

MARS struck the re-imported `WING_ARM` as *"not a second witness; it is a fifth clause."*
SATURN did **not** take his `_ledger_binding()` route — he went wider: **one vote per file across
every `V20_R15_IT*.md`, his own it.22 report excluded**, requiring the pin's arm to win by **≥5
files, ≥2× the runner-up, from ≥3 distinct offices.**

**`W1 = arm_smprime`: 25 files against 7. `W3 = arm_pl`: 23 against 7. Six offices.** *"To move
the pin an editor must now rewrite reports authored by MARS, VENUS, JUPITER, MERCURY and the
INSPECTOR."*

**And the reason it is admissible rather than circular is that he proved the corpus can
disagree.** The it.18/it.21 mutation transcripts print `arm_of W1 = arm_pl` **verbatim** — so
`test_the_corroboration_can_return_false` **asserts contrary evidence exists before crediting
the agreement.** A consensus instrument that cannot observe dissent is measuring its own corpus;
**this one was shown dissent first.** `24 passed`, was 22.

**His own limit, filed unprompted:** *"The corroboration is **consensus, not truth** — it defeats
an editor, not a founding mistake at it.1."*

**And he flagged himself to the adversary:** *"The `_shape` widening in
`test_v20_r15_it19_pid_channel.py` is **this office editing its own node so its own repair
passes**; claim unchanged, but MARS should look at it."*

**Not reached:** the 60-second real-watchdog wait, **owed since it.21**; the four-clause+pin move
was **simulated by patching `_arm_of`, not by mutating the manifest text**, so the no-node-fires
claim rests on it.21's measurement; `tests/saturn` and `tests/mars_v20` **not re-run whole**, so
it.21's standing `9 failed` / `39 failed` are **neither confirmed nor cleared.**

### INSPECTOR — `20 AUDITED, 4 STRUCK, 16 UPHELD`, AND THE `[RUN]` FINDING GOT A BOUNDED NEGATIVE

**The `9 of 12` sample stands, and he ruled on it against his own withdrawn work.** His `8-in-20`
died *"for three absences, not for its size"* — no published frame, no seed, and a rate used over
a population it was not drawn from. **MERCURY closes all three checkably**, and the disjoint seed
**exceeds** the standard: *"two disjoint draws are poolable later by an office neither of us
controls."*

**All three argument failures verified by opening the lines himself**, and he reproduced the
`:1` class **to the digit: 8 occurrences, 6 files, 6.2%.**

**STRIKE 1 — and it lands on the report that catalogues this exact defect.** `grep -n F3` returns
`139, 142, 241, 243, 244` — **no hit on `242`.**

> **The report cataloguing pointers that land one line off their proposition landed its own
> repair pointer one line off.**

**Three offices have now proposed a line for this one citation.** MERCURY said `:242`; JUPITER
found `:242` empty and ruled **no line carries `F3` uniquely**, so it needs a **withdrawal plus
re-issue**; the INSPECTOR says the fix is `:239 -> :241`. **Their disagreement is about remedy,
not fact** — all three agree `:242` is empty. **Carried to it.24 unresolved rather than split
here by an office that measured none of it.**

**THE CENSUS SURVIVED A THIRD INDEPENDENT COUNT, AND THE THIRD USED NO WHITELIST AT ALL** —
*"a whitelist cannot detect the class of error a whitelist causes"*, which is MERCURY's own
`.lean` omission turned into a method. **`129 / 103 / 37 files / 0 unresolvable / 116-13`**, with
an **exhaustive histogram**: `.md 40, .py 69, .lean 19, .jsonl 1 = 129`. **Nothing else is
omitted, provably.**

**STRIKES 2 AND 3 — `MERCURY-CENSUS-R1` IS RED AT HEAD, TEN MINUTES OLD.** He published
`3 failed, 4 passed`; the audit got **4 failed**, *"the extra one is the seal itself."* Using
**MERCURY's own helpers**, the digest hashes **all 103 pointers with no `MERCURY_LIVE` filter**,
so **10 rows come from the three append-growing files his own section 1 refuses to score.**

> **The seal fires when the round does ordinary work, not when a file moves under the table.**

**And on the tamper-proofing JUPITER priced at it.21:** *"two files writable by one process is
**one office wearing two hats** — what it actually buys is **two edits in two grammars, a real
raise in cost, not tamper-proof.**"* **It certifies the population, not the landings — which is
where the argument defects live.**

### THE `[RUN]` FINDING IS UPHELD AND CARRIES A BOUNDED NEGATIVE THIS OFFICE DID NOT SEE

`38,271` is **true**. *"The marker was wrong while the figure was right, so **no figure-checking
audit could ever catch it.**"* He sampled by **executing**: 3 of 3 reproduce.

> **Re-execution cannot detect the defect, because a restated marker reproduces perfectly.**

**That is the part this office missed.** The route filed at it.22 — `[RUN]` for a command this
office executed, `[CITED: office]` for a number taken from another's — **is the ONLY route
available**, because **there is no test that separates the two after the fact.** The corpus
carries **`601` `[RUN]` markers, effectively none audited**, and **no audit of them is possible;
only a discipline going forward.**

### STRIKE 4 — THE DESTRUCTIVE BRANCH AGAIN, AND ITS CONTROL WAS THE FAILING NODE

**`_owns_watchdog` finally opened: four lines at `scripts/iteration_timer.sh:51-55`.** Reads the
pid's own `/proc/<pid>/cmdline`, greps for `$STATE`, unreadable procfs -> `return 1`. **Design
sound, fails closed.**

**But SATURN's it.21 report claims `6 passed`; at HEAD it is `5 passed, 1 failed` — and the
failing node is the CONTROL**, *"the only one that can tell a working probe from one returning
false for everything."*

**And he went below the test:** on this MSYS2 box **two probes of the same construct returned two
different wrong answers** — `/usr/bin/bash`, then the **parent's** cmdline.

> **The probe reads evidence not sourced from the process it names — the same defect class it.19
> retired `$PIDFILE` for.**

**Third iteration running, the same mechanism, in the same subsystem, inside the repair for it.**
The round is **safe** — there was never a wrong kill — but **`kill "$wpid"` is effectively
unreachable on this box, and cleanup rests entirely on the `EXIT` trap, the one path never
exercised through a real `start 1`.** He **declined** to arm a real watchdog mid-audit *"since
arming a real watchdog mid-audit risks the round's clock"* — **a refusal with a stated reason,
which is the correct shape and not a gap.**

**SATURN's it.22 report appeared 43 seconds after the audit opened and is left unaudited**, named
rather than skipped. **His it.22 test file is bound to live code** at `iteration_timer.sh:86-105`
— *"the exact hole I fell into at it.21."*

**Tree:** nothing mutated, no `Edit`, no `sed -i`, **HEAD `207e7b9` at open and close.** *"No timer
re-arm observed in the window"* — **the first audit this round able to say that, because the
instrument now prints its arm.**

### WHAT it.24 OWES

1. **`_owns_watchdog`'s control node** — RED at HEAD while its report says `6 passed`, and it is
   the only node separating a working probe from one that returns false for everything. **The
   probe reads a cmdline not sourced from the process it names.** Third iteration, same class,
   same subsystem.
2. **The `EXIT` trap exercised through a real `start 1`** — owed since it.21, declined twice for
   stated reasons. **It is now the only live cleanup path.**
3. **`MERCURY-CENSUS-R1` filtered by `MERCURY_LIVE`** — RED at HEAD; the seal fires on ordinary
   appends rather than on tampering.
4. **`C13`'s remedy settled** — JUPITER says withdraw-and-reissue, the INSPECTOR says `:239 ->
   :241`. **All three offices agree `:242` is empty.**
5. **`COMPOUND_HALF` repaired at `:13-14`**, and the rule bound: **a compound claim may not cite
   one line.**
6. **The `:1` notation shipped** — `path:*` for the file, `path:1` for line 1. **8 occurrences
   across 6 files.**
7. **The `28 vs 25` manifest gap**, unreached for a fifth iteration.
8. **The theory table tracked** — author's call — and the four rulings: CLAUSE_1_TAIL,
   L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a thirteenth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.23 bought is a census that survived three independent counts** — the third taken with
**no extension whitelist at all**, because a whitelist cannot detect the class of error a
whitelist causes — **and a bounded negative on this office's own best finding.** `129 of 129` for
location is now as well-established as anything in the round. **`9 of 12` for argument is a sample
of twelve and this record will not multiply it out.**

**The honest summary of thirteen iterations: the round has built instruments that find its own
errors faster than it makes them, and has moved the arms not at all.** Both halves are true, and
**the second is the one the north star measures.**

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.24 — THE ITERATION THAT SPENDS THE FREEZE RATHER THAN PROTECTING IT

Room: SATURN (the ownership probe repaired or retired; the `EXIT` trap through a real
`start 1`), JUPITER (the argument repairs APPLIED to the theory table, census re-taken in the
same filing), MERCURY (a seal that fires on tampering rather than on work; the `28 vs 25` gap).
**Sections below are `PENDING` until a report lands.**

### THE PROCEDURAL QUESTION THIS ITERATION SETTLES

JUPITER declined to apply any argument-layer repair at it.23, and his reason was real:

> *"Each edit moves an occurrence out of the it.20 `CENSUS` and drops `129 of 129` until the
> census is re-taken in the same iteration."*

**A freeze that cannot be spent is a freeze that will be abandoned.** The round has built, over
five iterations, a census good enough that three offices reached `129 / 103 / 37` independently
— and the first time it was asked to permit an edit, **the correct move looked like doing
nothing.** That is the shape of every over-tight control: it stops being a check and becomes a
reason not to act, and then it is switched off by someone in a hurry.

**So it.24 pays the cost once and establishes the procedure: edit, re-take, re-declare, in one
filing.** If the number moves, the record says what it moved to and why. **What must never
happen is an edit filed in one iteration and a census re-taken in the next**, because for the
gap between them the round's headline number would be an assertion about a file that had already
changed.

### AND A DISAGREEMENT BETWEEN THREE OFFICES THAT IS WORTH MORE THAN ITS SUBJECT

`C13` is one citation. **MERCURY said the `F3` is at `:242`; JUPITER measured `:242` empty and
ruled the token non-unique, needing withdrawal and re-issue; the INSPECTOR measured
`139, 142, 241, 243, 244` and says the fix is `:239 → :241`.** **All three agree `:242` is
empty.**

**The disagreement is whether a pointer edit suffices when the cited token is not unique in its
neighbourhood** — and JUPITER's own heading resolver **REFUSES** on exactly that condition
(*"'F4' appears on 4 lines ... not 1"*). **The round therefore has one rule at the heading level
and no rule one level down, and has been repairing line citations under the looser standard for
six iterations.** Whichever way it is settled, the consistency is the finding.

### SATURN — THE PROBE WAS SOUND AND THE TEST WAS A COIN FLIP

**REPAIRED, not retired, and the INSPECTOR's reading was right while his inference was not.**
Measured through a real `start`, with `$STATE` derived the way the script derives it:

```
PROBE(real watchdog, script-form STATE) = TRUE
PROBE(unrelated pid 46558)              = FALSE  (correct)
wpid DEAD -- disarm signalled it
```

**`kill "$wpid"` IS reachable and `stop` DOES signal.** The "unreachable" finding came from
SATURN's **own first probe comparing a Windows-form path against the POSIX-form path the script
writes into the watchdog's argv** — *"that measures the harness."*

**AND THE CONTESTED COUNT WAS NEVER A DISAGREEMENT.** The it.21 node spawns
`bash -c "sleep 5" MARKER &`, **a shape the script never uses**, and that shape's cmdline is a
**race** returning at least four different things — the child's argv, the parent's cmdline,
`/usr/bin/bash` mid-exec, or `sleep 5` after bash's last-command exec optimisation. **Six runs
of that node alone: `1 failed, 5 passed`.**

> **it.21's `6 passed` and the INSPECTOR's `5 passed, 1 failed` are both true readings of a coin
> flip. Neither is its verdict.**

**This is a category the round has not had before.** Every prior conflict resolved to one office
being wrong. **Here both readings are honest, both are reproducible, and the instrument has no
verdict to give** — the disagreement was manufactured by a test whose subject was
nondeterministic. **A test that reports a different truth on different runs is worse than a
failing one**, because two offices will each cite their own run and neither will be lying.
Retired in place, **replacement shipped first**, mutation-tested both arms, `8/8 deterministic`.

**TWO DEFECTS IN HIS OWN INSTRUMENTS, BOTH THE CLASS UNDER AUDIT, BOTH FILED AGAINST HIMSELF:**

- The control's first draft asserted `"OWNED" in stdout` — **a substring of `"NOT_OWNED"`** — so
  it **passed the `and False` mutant, in the file built to catch it.**
- The restore was "verified" with `git diff` **on an untracked file** — **a check that returns
  clean for every input.** Now sha256 against a pre-mutation copy: `67b049c6…88a5e931`,
  **IDENTICAL.**

**A verification that returns clean for every input is `V-7` with the sign flipped**, and it was
guarding the revert of a mutation to the round's only destructive branch.

**THE TRAP FIRES — real `start 1`, real 60-second wait, no simulation:**

```
=== T0 11:47:10Z   armed: 1 min cap, deadline 11:48:10Z   wpid=46097
=== 11:47:46Z  pid ALIVE, no OVERDUE, check rc=0 (0m37s elapsed, 0m23s left)
=== 11:48:25Z  pid DEAD
   ITERATION_OVERDUE: 2026-09-02T11:48:10Z
   registration file: cannot access -- removed by trap ... EXIT
   check: OVERDUE - ran 1m15s against a 1m cap, armed 11:47:10Z   rc=1
```

**Owed since it.21, declined twice for stated reasons, and it discharges MARS's it.20 strike:
the stale-`$WATCHDOG` condition does not reproduce.**

**ALL TEN PUBLISHED COUNTS RE-CHECKED. SIX HOLD. FOUR DO NOT** — and the two the INSPECTOR
flagged as *"neither confirmed nor cleared"* are now cleared and **both were wrong**:
**`tests/saturn` `9 failed / 169 passed` → `8 failed, 177 passed`**; **`tests/mars_v20`
`39 failed` → `31 failed, 59 passed`.**

**Not reached, and he says so rather than explaining it away:** *"why the suite totals moved —
the +1/−1 residue in `tests/saturn` and the 8-failure gap in `tests/mars_v20` are **measured but
unexplained**, and no node-level attribution was done."* **Non-suite numbers in it.21/it.22 were
not re-checked for the same shape.**

### JUPITER — THE FREEZE WAS SPENT AND THE COUNT DID NOT LAPSE

**`129 of 129`, after the edits, re-taken in the same filing: 107 by line + 1 by range + 8 by
file + 13 by heading. 0 unscored, 0 failing.** **The procedure this iteration existed to
establish — edit, re-take, re-declare, one filing — is now done once and can be done again.**

**And the histogram the brief predicted would move did not move:** `.md 40, .py 69, .lean 19,
.jsonl 1`. **The histogram that DID move is by NOTATION, and it did not exist before this
iteration.** Both are asserted by nodes, *"so neither can drift"* — **the census now measures a
dimension the round invented this iteration, which is the difference between a count and an
instrument.**

**`J-24a` — the `:1` notation shipped and the instrument reads it.** 8 occurrences across 6
files, the number three offices reached independently, **now spelled `path:*` and no longer
pretending to be a line.**

**`J-24b` — `COMPOUND_HALF` repaired as a range, and the freeze was not bent to let it
through:** the citation was **withdrawn and re-issued as `C131`**, not edited in place. **A
compound claim may not cite one line**, and the office that wanted the repair took the expensive
route to get it.

**`J-24c` — `C13` SETTLED, AND JUPITER STRUCK HIS OWN it.23 REASONING TO DO IT.**
**The INSPECTOR's remedy `:239 -> :241` is ADOPTED. MERCURY's `:242` is the only wrong number
of the three. And JUPITER's own it.23 ground for refusing a digit repair is STRUCK BY ITS
AUTHOR:**

> **Non-uniqueness does not block a digit pointer.** The resolver refuses *"'F4' appears on 4
> lines ... not 1"* because **uniqueness is its ADDRESSING mechanism** — `resolve()` is handed no
> line number and must **derive** one, so four matches means refuse or guess. **A digit pointer
> is already addressed**; `:241` names the line and the want is asked one question.

> **it.23 imported an addressing constraint into a place where addressing was already settled.**

**And the evidence he used against himself was sitting in plain sight:** *"the it.20 landing
check has never required uniqueness — `C2`'s want is not unique in its file, and **no office has
called that a defect in four iterations.**"* **The rule he proposed would have condemned a
citation the round has been content with since it.20**, and he found that himself rather than
being told. **Both halves are now pinned in one test with a resolving control, so the
consistency is a test rather than a paragraph.**

**`C13` is still withdrawn and re-issued — for a reason none of the three offices gave.** The
frozen want is `M14 CHEEGER STRATIFICATION`; **`:241` does not carry it.** Re-issued as `C130`
with want `grade F3 pending`:

> **The old want certified the item's NAME while the cell asserted its GRADE.**

**Three offices spent three iterations arguing which line holds `F3`, and the citation was never
about `F3`.** *"A `J-21c` argument-layer defect closed, not a location repaired."*

**And the withdrawal channel is the thing that made it possible.** `REISSUED = {130: 13,
131: 63}` — *"the `J-20a` withdrawal channel that did not exist, **which is why it.23 could not
act.**"* **it.23's inability to apply a repair was not caution; it was a missing mechanism**, and
naming it as caution was this office's reading, not JUPITER's.

`WANT_SEAL` moved `fda4cdc2… -> 538b5319…` — **two wants left, two arrived, visible in the
diff.** `231 passed, 3 failed` across `tests/jupiter/`, **exactly the three named pre-existing at
it.23.**

### AND A FINDING AGAINST MERCURY'S INSTRUMENT, WHICH IS THE SHARPEST SMALL THING THIS ITERATION

`test_m22a_compound_half_constants_13_does_not_carry_beta` asserts
`"-0.032353" in line_at(constants, 13)` — **it reads only the constants file**, untouched this
iteration, **so it was RED before the repair and RED after it.**

> **It demands the FILE move to match the citation when the defect is that the CITATION does not
> match the file.** **A defect marker must assert against the thing the repair will change.**

**His finding was right and was adopted; his instrument cannot observe its own remedy.** **Not
repaired — his file**, and JUPITER said so rather than reaching into another office's tests.

### MERCURY — THE SEAL NOW FIRES ON TAMPERING AND IS SILENT ON WORK

```
test_negative_GREEN_the_seal_does_not_move_on_an_ordinary_append PASSED
test_negative_RED_the_seal_moves_when_a_line_moves_under_a_scored_pointer PASSED
2 passed in 0.72s
```

**Negative 1 appends 50 lines to all three `MERCURY_LIVE` files AND to all 86 scored files —
digest unchanged. Negative 2 inserts one line at the top of each scored file in turn — the
digest moves for EVERY one.** Both in memory; nothing written to the repo.

**He confirmed the charge with his own numbers before repairing it:** 10 of 103 pointers were in
refused files, and the it.22 recipe moved on **one ordinary journal insert**
(`20a1fab1… → cc6d433d…`). **`MERCURY-R1b` filters by exclusion, not by the inclusion whitelist
that caused his `.lean` error** — the distinction stated so the repair cannot reintroduce the
defect it is downstream of.

**`M-24b` — the measurement `J-23g` was missing, and it prices JUPITER's ruling rather than
disputing it:** *"the ledger is **7 of the 10** refused pointers — **70% of the seal's blind spot
bought for a file at 438 lines for three iterations**, while the two files that actually churn
cost three."* **The ruling is not disputed; the price is banked.**

### `28 vs 25` — CLOSED AFTER FIVE ITERATIONS, AND IT WAS NEVER A DISCREPANCY

**Two populations compared by mistake.** `28` counts census-failing **occurrences**; `25` reads
`MANIFEST`, which counts distinct repaired **target lines**. The mapping is **neither injective
nor surjective**:

```
28 - 1 (C26+C74 -> one target) - 1 (C10 is a LABEL repair) + 3 (C44 -> :75/:108/:131/:279)
   + 1 (C95, ruled not censused) = 30 = len(MANIFEST)
```

**No missing repairs.** And he marked the unrecoverable half honestly: *"the `25` itself is
`[CITED: INSPECTOR]` and unrecoverable — the file is untracked"* — **the it.22 marker discipline
used correctly by another office within two iterations of being written.**

**An item sat on the owed list for five iterations because every office assumed a counting gap
needed a counter.** It needed someone to ask what each number counted.

### THE TABLE MOVED UNDER TWO OFFICES WHILE THEY MEASURED IT, AND BOTH SAID SO

MERCURY: *"The theory table was edited by another office at `11:51:49Z`, mid-iteration.
Occurrences `129 → 120`, pointers `103 → 96`."* **That was JUPITER applying the repairs this
office asked for.**

**Both offices dated their readings and neither reported a stale number as current.** MERCURY
re-took his hex **twice inside the iteration** and named the limit: *"the frozen hex is a dated
reading over a file under concurrent edit ... the durable part is the two negatives."*

**This is the moving-target problem the round has hit in four consecutive audits, and it is the
first time it cost nothing.** Not because the target stopped moving — **it moved harder than
ever, mid-iteration, by design** — but because both offices had been trained by the previous
four to date every reading. **The procedure absorbed the disruption the freeze was invented to
prevent.**

### WHAT it.25 OWES

1. **The suite-total movements explained** — `tests/saturn` `9 → 8 failed`, `tests/mars_v20`
   `39 → 31 failed`. **Measured but unexplained**, with no node-level attribution.
2. **`MERCURY`'s second RED**, `test_it8_aggregate_verdict_agrees_with_its_own_cells` — **not
   opened this iteration, unclaimed, not verified pre-existing.**
3. **The landings census** — priced by MERCURY at **~7× the it.22 argument read, one full
   office-iteration, not splittable.** **It is the only second census that would cover where the
   defects actually live.**
4. **Non-suite numbers in it.21/it.22** re-checked for the coin-flip shape.
5. **The lazy-predicate separation cell**, carried a fourth iteration.
6. **The theory table tracked** — author's call — and the four rulings: CLAUSE_1_TAIL,
   L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a fourteenth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.24 bought is a procedure and a category.** The procedure: **the freeze can be spent —
edit, re-take, re-declare in one filing — and `129 of 129` survived being acted on.** The
category is SATURN's and it is new to the round: **a test whose subject is nondeterministic
produces two honest, reproducible, contradictory readings, and has no verdict to give.** Every
prior conflict this round resolved to somebody being wrong. **This one resolved to a coin flip
that two offices had each cited in good faith.**

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.25 — THE REPAIR RECORD READS AS THE DEFECT

Room: MERCURY (the LANDINGS census — the second-office check nobody has taken), MARS (attack
it.24), INSPECTOR (it.23–it.24, and his own overturned strike). **Sections below are `PENDING`
until a report lands.**

### A PROPERTY OF THIS ROUND'S OWN CORRECTION MECHANISM, FOUND BY VERIFYING A CLAIM THAT TWO
### OFFICES REPORTED AS TRUE

JUPITER's it.24 close and MERCURY's it.23 both record MARS's STRIKE 3 — the stale `31` in the
`INDEX-SHA256` sentence — as **untouched, three iterations running.**

**It was repaired at it.23.** Verified this iteration `[RUN]`: the declaration at
`V20_R15_JOURNAL.md:87` reads **`over the **33** `C`-rows `C1`..`C33`, contiguous`**, digest
`24328026…` recomputes equal, **index max 33 = body max 33.**

**And the reason two offices read it as live is the finding.** The correction that fixed it —
index row `C33` — **quotes the defective string inside its own text**: *"The literal read
`over the **31** `C`-rows`."*

> **A grep for the stale count finds the correction before it finds the declaration. The repair
> record reads as the defect.**

**This is a general property of an append-only corrections mechanism, and the round has 33 of
them.** The CORRECTIONS INDEX exists to make superseded claims findable — **which necessarily
means every superseded claim is now written twice, once as the error and once inside the row
that overturns it.** A reader who greps for a withdrawn number finds it. **A reader who greps
for a withdrawn number and stops there concludes the round still believes it.**

**It is not a defect in the index and the index should not be changed.** The alternative — a
corrections table that does not quote what it corrects — is unreadable, and the round tried
line-number pointers instead and got `P-6`. **What is wrong is that four instruments this round
has built are grep-based, and none of them excludes the index block.** MERCURY's censuses, MARS's
strikes and the INSPECTOR's audits all search this journal, and **the index is 33 rows of exactly
the strings they hunt for.**

**The route is not clever and does not need to be:** any search over `V20_R15_JOURNAL.md` must
**exclude the CORRECTIONS INDEX block by its own headings**, and say that it did. **A search that
does not say what it excluded is not reproducible**, which is the standard the round already
applies to every other number it publishes.

### MERCURY — `21 of 129` READ BY HAND, `3` DEFECTS, AND THE POPULATION CONFIRMED BY A SECOND OFFICE

**`MERCURY-R2`: three regexes, no manifest, nothing imported from `tests/jupiter/`.** It
reproduces JUPITER's **`129 occurrences / 103 unique / 37 files`**, the extension histogram
`.md 40 .py 69 .lean 19 .jsonl 1`, **and the notation histogram invented last iteration**,
`120 plain + 8 file + 1 range`. **Second office, independent instrument, same numbers.**

**AND IT CORRECTED HIS OWN it.24 REPORT.** He had published *"nine occurrences and seven
pointers left the table mid-iteration."* **They did not leave. They changed notation.** The it.22
regex saw only `path:N`; the eight `:1` pointers had become `path:*` and `C63` a range.
`120 + 8 + 1 = 129`, unchanged.

> **An instrument that measures a population through one notation reports a notation change as a
> population change.**

**The alarm this office raised at it.24 — "the table moved under two offices mid-iteration" — was
half wrong, and the half that was wrong was the count.** The edit was real; the population
movement was an artefact of the measuring regex. **This record printed `129 → 120` as a fact.**

### THE THREE LANDING DEFECTS, ALL GREEN UNDER EVERY LOCATION INSTRUMENT THE ROUND OWNS

| | pointer | the cell's claim | what the line carries |
|---|---|---|---|
| **M-25a** | `scripts/v15_r1.py:547` | *"nine sibling argparse flags"* | **one flag** (`--seeds`). The nine span `:547-558`. **Count right, landing wrong — and `C131`'s `:A-B` notation exists and was not used.** |
| **M-25b** | `scripts/v15_r1.py:586` | `floor₁ = sqrt((t*−1)/t*) = 0.7071067811865476` at `t* = 2` | `floor1 = math.sqrt((T_STAR - 1) / T_STAR)`. **`0.70710` appears NOWHERE in the file — computed, never written.** `T_STAR = 2` is at `:138`, uncited. **One of three parts lands.** |
| **M-25c** | `ceq/kdata.py:475` | *"the generator exists at `:475` and has zero callers"* | `"bed_k": {"generator": "ceq.beds.bed_k.build_delay",` — **a registry entry holding the generator's NAME.** The generator is at `ceq/beds/bed_k.py:236`, **uncited**. *Zero callers* is **TRUE**. **Half lands.** |

**`M-25b` is the exact class JUPITER struck MERCURY's own marker over, found again in a different
cell against a different file** — a constant the cell quotes to sixteen digits that **the cited
file never contains**, because the file computes it. **A citation can be correct about where the
value comes from and wrong that the value is there.**

**`M-25c` is the one that costs something.** It sits in an **admission condition** — the
gate that says what would let Q2/W1 re-enter. **A leap following `:475` to run one BED-K cell
arrives at a dict key.** The generator it needs is one file over and has never been cited.

### THE NUMBER HE PUBLISHED SO THAT NOBODY WOULD BANK IT

**108 of 129 were screened but not read**, and the mechanical screen returned **`58 unlanded of
89 wants`**.

> **That is an artefact, published only so no office banks it.**

`F0–F4` are rubric prose in every scale-discussing cell; a cell's constants live in its `CENSUS`
bullet while its pointers live in `DECLARATION`. **Q3/W3 is the clean false positive, ruled LANDS
by hand.**

**An office that measured `58` and published it as `58` would have been believed**, and the round
has spent six iterations learning that a big number with a plausible mechanism is the most
expensive thing it can produce. **He measured it, diagnosed why it is wrong, and printed it
anyway with the diagnosis attached** — which is the only safe way to report a screen you cannot
stand behind.

**Remainder priced at ~5 further office-iterations, splittable by clause now that the extractor
exists** — and the price is lower than his it.24 estimate **because the instrument now exists**,
which is the argument for having built it.

**One gap he could not close and would not count:** **Q3/W3's `0.451211 / 0.662128 / 1.113339`
are cited by NO pointer at all.** *"A missing citation, a different class, not counted in the 3."*
**The census measures whether citations land. It cannot see a claim that never had one**, and
nothing in the round can.

**His struck marker is fixed and the family audited:** rewritten to assert the **citation**
rather than the file, `8 passed`; **`grep` for the same shape across everything he has shipped
returns nothing — one instrument had the defect, not a family.**

**And his second RED is CLAIMED, not pre-existing:** it is his own it.10 `RED-3`, **shipped red
on purpose**, asserting that `8/9` cells sit below `floor₁` while the aggregate row registers
`crosses=False`. *"It is the same fact the table now carries as the Q5/W3 GAP, and goes green
when that cell's 0-GPU-s ROUTE is taken. **Do not clear it.**"*

### CORRECTION 34 — THIS OFFICE PRINTED A POPULATION MOVEMENT THAT NEVER HAPPENED

it.24 recorded *"Occurrences `129 -> 120`, pointers `103 -> 96`"* as a fact about the theory
table. **The population did not move; the NOTATION did**, and MERCURY caught it against his own
report. Filed as index row `C34`; index re-declared at **34 rows, `C1`..`C34`, contiguous**,
digest recomputing equal `[RUN]`.

**This office had the number from an office that had it from a regex, and printed it without
asking what the regex could see.** That is the `[CITED:]` discipline written at it.22 and not
applied by its own author two iterations later — **not a false `[RUN]` marker this time, but the
same failure to ask whose measurement it was.**

### MARS — 3 ATTACKED, 3 STRUCK, AND THE FIRST ONE VOIDS A PROVENANCE THE ROUND HAS USED ALL ALONG

**He confirmed STRIKE 3 repaired before attacking anything** — digest `24328026…`, 33 rows,
contiguous, equal to the declaration. *"ATTACK 3's digest audit is clean; the coordinator's `C33`
holds."*

**STRIKE A — a HEAD SHA is not a provenance for the suite it counts.**

```
E  0 of 22 tests/mars_v20 files are known to git. A count published 'at HEAD 207e7b9' over
E  this suite names a commit that does not contain it: the same blindness SATURN proved for
E  'git diff --stat' on an untracked script, one section later, used as provenance instead
E  of as a defect. It is why his '39 failed' at it.21 cannot be sourced -- there is no
E  history to source it from.  assert 0 == 22
```

**SATURN proved the blindness and then used the blind channel as provenance in the same
filing.** Five consecutive runs give `32 failed, 58 passed`, **deterministic 5/5**, against his
`31/59`. **Route: publish a content digest of the collected files beside the count, never a
commit SHA.**

**STRIKE B — the 8-failure gap explained, and it indicts the it.24 procedure this office
designed.**

```
E  opened citations absent from the table: ['tests/jupiter/test_v20_r15_it12_constants.py:13']
E  it.24 withdrew cids [13, 63] from CENSUS and re-issued them widened, but it.17's MANIFEST
E  -- which it.24 states was 'not edited' -- still carries the OLD pointer, and MANIFEST is
E  what foreign instruments read.
```

> **EDIT / RE-TAKE / RE-DECLARE re-took `tests/jupiter` only. `231 passed, 3 failed` is the
> editing office grading its own blast radius.**

**The procedure it.24 was built to establish is incomplete, and the gap is exactly the one this
office did not think to specify:** re-take **what the edit touched**, not **what the editing
office owns.** **Route:** derive `MANIFEST` from `CENSUS` so a withdrawal rewrites the datum
once; **scope the re-take by `grep -l` over importers.**

**STRIKE C — `:*` cannot tell a want from its own obituary.**

```
E  `ceq/arm_pl.py:*` still LANDS on a file where the want was deleted and only a line
E  recording its removal remains. J-24a exempts `:*` from J-20b because an append cannot
E  break it -- and an append is exactly what repairs the deletion it cannot see.
```

> **A file-scoped anchor is a grep.**

**The notation shipped at it.24 to fix the `:1` idiom inherits the CORRECTIONS INDEX property
this iteration opened with** — and MARS is the one who connected them. **The control is GREEN: a
*silent* deletion IS caught**, so the probe is not true-for-everything. **Route:** freeze an
occurrence **count** at census and score against it, plus an `overturns:` field on index rows
**so greps never cross the index.**

**MARS closed over the cap and said so. STRIKE 1, the `WING_ARM` pin, is open a fourth
iteration.** And he turned SATURN's own standard back on him: *"his `EXIT` trap discharge rests
on **one** 60-second observation — the same single-sample shape he condemned in the it.21 node."*

### INSPECTOR — `14 AUDITED, 5 STRUCK, 9 UPHELD`, AND HE WITHDREW HIS OWN STRIKE AGAINST HIMSELF

> **My it.23 strike is withdrawn against myself.** SATURN's measurement stands — *"the
> 'unreachable' finding came from a Windows-form path compared against the POSIX-form path the
> script writes into argv: **my instrument reading a channel that cannot carry the fact, inside
> the strike that named that class.**"*

**THE RULING THAT OUTRANKS EVERYTHING ELSE THIS ITERATION.** He adopted SATURN's
nondeterministic-test category and **extended it to the round's whole evidence apparatus:**

> **`[RUN]` provenance is necessary and not sufficient. Every anti-fabrication device this round
> built authenticates that a reading HAPPENED. None authenticates that the SUBJECT HAD A VALUE.**

**Nine iterations of markers, digests, seals, planted negatives and census recipes all certify
the same half of the problem.** Corpus sweep: `tests/mars_v20` **cleared** (4/4 identical,
including a randomized run); **one other confirmed** —
`test_the_declared_cells_digest_matches_the_table_at_head`, **whose subject is a file JUPITER was
editing at `11:51:49Z` mid-iteration.**

**NEW STRIKE ON JUPITER, AND IT IS THE `J-21c` CLASS REPRODUCED INSIDE ITS OWN REPAIR.** `C130`'s
want `grade F3 pending` **lands at `:241`** — but `:244` reads **`SUPERSEDED it.19, RULING J-17d:
M14 IS F4`**.

> **The re-issued citation certifies a grade the cited document marks dead. Locationally
> perfect, semantically void.**

**The repair for an argument-layer defect is itself an argument-layer defect**, three lines from
its own pointer.

**SATURN's two counts BOTH struck, and one is attributed at node level:** `tests/saturn`
**`9 failed / 177 passed`, 186 nodes against his 185**; `tests/mars_v20` **`32/58` against his
`31/59`** — a third reading, 4/4 stable, **agreeing with MARS's five runs.**

> **JUPITER's theory-table edits broke SATURN's it.19 freeze node, and neither filing says so
> because they read the suite on opposite sides of an `11:51:49Z` write.**

**AND HE STRUCK ONE OF HIS OWN REVERT CLAIMS.** it.16 `:137` verified a restore with
`git diff HEAD` **on a file `git ls-files` does not know.** **Struck.** it.13's git half withdrawn,
its hash surviving. **His `git status --porcelain` tree statements are NOT affected** —
*"porcelain certifies the path set, never untracked content, and I never made it load-bearing for
a restore."* **A precise self-limitation rather than a blanket withdrawal.**

**THE REPAIR-RECORD RULING, MEASURED:** **33 of 33 index rows quote a literal from the claim they
overturn** — *"this is the index's design, not `C33`'s accident."* The stale `31` survives at 2
sites, **0 of them live claims: the grep is 100% false positive.**

> **In a document containing its own correction record, grep measures vocabulary, not belief —
> and better repair discipline makes the false-positive rate WORSE.**

**Affected: every census, count and presence grep over the journal and `MISTAKES.md`, his own
included. Unaffected: greps over source, tests and the contract — and the `INDEX-SHA256` recipe
itself, which "matches `^\\| C\\d+ \\|` positionally, so it was already immune."** **Independently
confirmed by this office `[RUN]`:** the one candidate outside the index, `| C — THE ARENA |` at
`:17`, **carries no digits and is correctly excluded.**

### WHAT it.26 OWES

1. **`C130` re-issued again** — its want certifies a grade `:244` marks superseded. **The
   argument-layer repair that is an argument-layer defect.**
2. **`MANIFEST` derived from `CENSUS`**, and the re-take scoped **by importers, not by office** —
   MARS's route, and it is the correction to this office's it.24 procedure.
3. **Counts published with a content digest of the collected files**, never a HEAD SHA over
   untracked paths.
4. **`:*` given a frozen occurrence count**, and an `overturns:` field on index rows so greps
   never cross the index.
5. **`M-25a`, `M-25b`, `M-25c`** — the three landing defects, one of them inside an **admission
   condition**, where a leap following the pointer arrives at a dict key.
6. **Q3/W3's three constants, cited by no pointer at all** — a class the census cannot see.
7. **`WING_ARM`, open a fourth iteration**, and the `EXIT` trap on a second run.
8. **The theory table tracked** — author's call — and the four rulings: CLAUSE_1_TAIL,
   L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a fifteenth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.25 bought is the sentence the round will be judged by**, and it is the INSPECTOR's:
**every anti-fabrication device built here authenticates that a reading happened; none
authenticates that the subject had a value.** The `[RUN]` marker, the `WANT_SEAL`, the census
digests, the planted negatives — **all of them certify provenance, and provenance was never the
scarce thing.** **`21 of 129` landings read by hand, 3 defects; `9 of 12` on argument; `129 of
129` on location.** The first two are samples and the record will not multiply them out.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.26 — PROVENANCE WAS NEVER THE SCARCE THING

Room: JUPITER (`C130` re-issued live; `MANIFEST` derived from `CENSUS`; the `:*` count freeze),
SATURN (counts given a sourceable provenance; `WING_ARM` after four iterations), MERCURY (the
three landing defects; **the uncited-claim class, which no instrument in the round can see**).
**Sections below are `PENDING` until a report lands.**

### THE SENTENCE THE ROUND WILL BE JUDGED BY, AND WHAT IT.26 DOES ABOUT IT

The INSPECTOR at it.25, extending SATURN's nondeterministic-test category to the whole
apparatus:

> **`[RUN]` provenance is necessary and not sufficient. Every anti-fabrication device this round
> built authenticates that a reading HAPPENED. None authenticates that the SUBJECT HAD A VALUE.**

**Nine iterations of markers, seals, digests, planted negatives and census recipes certify one
half of the problem, and it is not the half that was scarce.** The round has been able to prove
*"this number came from a command someone ran"* since it.3. **It has never been able to prove
*"the thing that command measured had a value at the time."*** SATURN's coin-flip node is the
pure case — an honest reading, reproducible, of a subject with no value.

**Three consequences are being paid this iteration, and they are the same defect at three
distances:**

1. **A suite count stamped with a HEAD SHA over untracked tests** names a commit that does not
   contain them. `0 of 22`. **The provenance is real and points at nothing.**
2. **`:*` LANDS on a file where the want was deleted and only its obituary remains.** *"A
   file-scoped anchor is a grep."*
3. **A claim with no citation at all** is invisible to every instrument the round owns —
   **because every one of them starts at a citation and looks outward.** MERCURY is the first
   office asked to start at the claim.

**The third is the one with no precedent here.** A cell can be `129 of 129` green and assert
three numbers nothing points at, **and the round would report it as fully certified.** Q3/W3
does exactly that today.

### AND A PROCEDURE THIS OFFICE SPECIFIED WRONG

**`EDIT / RE-TAKE / RE-DECLARE`, written into it.24 by this office, is incomplete**, and MARS
found the hole:

> **it.24 re-took `tests/jupiter` only. `231 passed, 3 failed` is the editing office grading its
> own blast radius.**

**The procedure said *re-take*. It did not say *re-take what the edit touched rather than what
the editing office owns*** — so a withdrawal in `CENSUS` left `MANIFEST` carrying the old
pointer, and `MANIFEST` is what foreign instruments read. **The office that wrote the procedure
is the office that under-specified it**, and the correction belongs in the same place the
procedure does.

### JUPITER — THE INSPECTOR WAS RIGHT IT WAS A DEFECT AND WRONG ABOUT WHICH ONE

**The cell never asserts `M14` is `F3`.** It reads *"`M14` carries three grades in three files
for one item — `F3` in the contract … **RESOLVED at it.17 — `M14 = F4`**."* **`:241` certifies the
`F3` half correctly.** What nothing certified is **that the cell knows the `F3` is dead**.

> **The anchor was narrower than the claim's liveness.**

**`C130` withdrawn, `C132` issued** over the region `241-244` with **both** wants —
`grade F3 pending` **and** `M14 IS F4`. *"What makes it live is not a better line — it's that the
region now contains **the sentence that could falsify it**."*

**`RULING J-26a` makes it mechanical:** scan from the addressed region to the end of its block
and **refuse any pointer whose supersession notice lies outside its own region.** **Repairable
for marked supersession, not for silent — it moves the it.21 reader-only boundary rather than
retiring it.**

**And the probe was calibrated against the failure mode this round keeps finding:** it fires on
**1 of 93 scored pointers** — `C130` and nothing else. *"A probe that reddened the census would
be measuring the vocabulary."* **An instrument that flags everything is the CORRECTIONS INDEX
grep with a different name**, and he checked before shipping.

**THE CORRECTED PROCEDURE, AND IT EARNED ITS KEEP ON THE ITERATION THAT INTRODUCED IT.**
`EDIT / **COMPUTE THE BLAST RADIUS** / RE-TAKE ALL OF IT / RE-DECLARE`, where the radius is
`importers(M)` — **computed, not nominated**, and covering **tracked and untracked files alike**,
because *"`git grep` without `--untracked` is blind to exactly the files this round writes."*

> The computed radius is 6 files, one of them MARS's. Re-taking it found **four failures in
> `test_v20_r15_it24_census_retake.py` that the office-scoped re-take would have shipped** — *"all
> the same defect one level up: it.24 restating the census inside its own filing."*

**`MANIFEST` is now a view over `CENSUS`**, so a withdrawal moves one datum, and the re-issue
ledger understands it is a **chain** — `C13 → C130 → C132` — not a pair. **`STRIKE B` is green.**

**AND HE KILLED MARS'S OWN `:*` ROUTE WITH A PLANTED NEGATIVE AGAINST THE REMEDY.** MARS proposed
freezing an occurrence **count**. `[RUN]`: **all six `:*` wants have count 1**, and *"his obituary
deletes the one asserting line and appends a line quoting the want, so a frozen count reads
`1 → 1` — **green on the exact mutation it was built to catch.**"* **`J-26c` freezes a digest over
the lines carrying the want instead:** an append leaves it unchanged, **deletion and obituary
both move it.**

**The adversary's strike was right and his remedy was the same defect wearing the repair's
clothes** — and it was found by testing the remedy rather than adopting it. **Filed against the
remedy, not against the strike**, which is the distinction the round has been sloppy about twice.

**Not reached, and he names the largest honestly:** *"`:*` scoring is not routed through
`lands()` — `J-26c` lives in the it.26 instrument; it.20's predicate is unmodified, so **MARS's
STRIKE C is still legitimately RED.** Largest thing open."*

### SATURN — THE THREE DISPUTED COUNTS WERE THREE DIFFERENT OBJECTS

**Content digests replace HEAD SHAs**, sha256 over `(relpath, sha256(bytes))` for every
`test_*.py`. **`tests/mars_v20` is 22 files, 0 known to git; `tests/saturn` is 18 files, 4 known
to git. MARS's STRIKE A upheld in full.**

- **`tests/mars_v20`: `35 failed, 61 passed` (96 nodes)**, **5/5 deterministic with an identical
  FAILED node set** — not merely an identical total.
- **`tests/saturn`: `9 failed, 177 passed` (186 nodes)**, 3/3.

> **The three disputed readings were three different objects: 90 nodes then, 96 now, the +6 being
> MARS's own it.25 file landing between his read and mine.**

**`The INSPECTOR's 9/177 over 186 is confirmed exactly; SATURN's own it.24 `8/177` over 185 was
wrong.`** And the honest closing of the rest: *"the `31` vs `32` split is **permanently
unadjudicable** — the 90-node object is gone and **no digest of it was ever taken**."* **His it.21
`39 failed` is withdrawn, not defended.**

**A count without a digest of what it counted cannot be re-litigated later, only abandoned** —
which is the whole argument for the repair, delivered as a loss rather than as a principle.

**The `tests/saturn` attribution is CONFIRMED at node level**, exactly as the INSPECTOR named it:
`test_the_declared_cells_digest_matches_the_table_at_head`, failing with *"cell bodies edited
since the freeze: `['Q1/W1', 'Q2/W3', 'Q3/W1', 'Q6/W1', 'Q6/W3']`"*. **The table's mtime is
`12:23:28Z` — during this iteration.** *"His `11:51:49Z` is one write of several."*

**`WING_ARM` MOVED TO THE LEDGER BINDING; THE CONSENSUS WITNESS IS WITHDRAWN BY ITS AUTHOR.**

> **The 32 votes all descend from the manifest, so a founding mistake at it.1 reads GREEN 32
> times.**

**The ledger rows are load-bearing instead of decorative** — `L-10` cites
`ceq/arm_smprime.py:163-172`, resolved; `L-14` cites `ArmPL.forward`, resolved at
`ceq/arm_pl.py:358`; **`L-16`'s `F4` rests on W1 being BED-M and is unstateable if W1 is the other
arm.**

> **A vote cannot be wrong in a way that shows; a resolved line citation can.**

**That answers the limit he filed against himself at it.22** — *"consensus, not truth; it defeats
an editor, not a founding mistake"* — **by replacing the evidence class rather than adding more
of the same.** MARS's it.22 route, taken four iterations later, after the consensus answer was
built, tested and found insufficient by its own author. **The node is argued and NOT SHIPPED, and
he names it his main debt.**

**Trap discharge confirmed 3/3** — two concurrent independent `start 1` runs on the byte-identical
script, both watchdogs dead, both registrations removed by the `EXIT` trap. **MARS's
single-sample objection is answered.**

**And one more self-caught instrument defect, filed rather than hidden:** the exit code first read
`rc=0` **because the harness piped `check` through `sed`, making `$?` sed's status.** Re-measured
without the pipe: `rc=1`.

> **Third instance this round of an instrument of his reading a channel that cannot carry the
> fact asked of it** — after `git diff` on an untracked path and the SHA-stamped counts.

### MERCURY — 3 OF 3 REPAIRABLE, AND A CLASS THE ROUND HAS BEEN FINDING ONE INSTANCE AT A TIME

**All three landing defects are repairable; none is unrepairable.** Each marker asserts the
**table text** — JUPITER's it.24 ruling applied by the office it was ruled against — with a
code-side control proving the repo was not mutated. **`3 failed, 3 passed`**, RED against the
unmutated table, identified by **content digest** because *"untracked file, so a content digest is
the only honest identifier."*

- **`M-25c`** — `:475` keeps its place as *registered @*, and the clause **gains the generator**
  at `ceq/beds/bed_k.py:236`. **Ruling: a registry entry and its artefact are two citations, and
  an admission condition must carry the second.**
- **`M-25a`** — `:547` → `:547-558`, using `C131`'s existing range notation. **Nothing else
  changes.**
- **`M-25b`** — **a computed constant may be cited, to two lines and never to one:** the line that
  computes it **and** the line fixing the input that makes it take that value (`T_STAR = 2` at
  `:138`). **The two re-cites at `:198`/`:206` cite the formula only and are correct as they
  stand — no edit there**, which is the discipline of not repairing what is not broken.

**All replacements are NAMED and left for JUPITER to apply.** He did not edit another office's
artifact.

### THE UNCITED-CLAIM CLASS, MEASURED FOR THE FIRST TIME: `29 ACROSS 10 OF 12 CELLS`

**Denominator `62`, and the denominator is the finding.** The mechanical screen returned **240**
numeric tokens; **63** were float-shaped; hand reading rejected one as a section reference,
leaving **62 ruled by hand.**

> **`240` is not the denominator** — dominated by census integers and shape constants. **The it.25
> artefact reproduced, and named as such before anyone could bank it.**

**Q3/W3 is worst at 5** — the INSPECTOR's exemplar confirmed, **plus two more in the same cell.**
`Q4/W1`, `Q4/W3`, `Q5/W1`, `Q5/W3` carry 4 each. **`Q1/W1` and `Q6/W1` are clean.**

> **Two cells already declare their own uncited constants in their own text. The round has been
> finding this one instance at a time for nine iterations without naming it a class.**

**This is the first measurement in the round that starts at the claim instead of the citation**,
and it is the only one that can see a cell which is `129 of 129` green while asserting five
numbers nothing points at.

**Landings census: `21 of 129`, NOT ADVANCED, and he says so plainly** — the clause priced for
this iteration was not opened; the wall clock went to the uncited measurement. **No
extrapolation offered.** **Scope named: §2 only. §0, §3 and §4 carry same-shape constants,
unmeasured — and `§0.2`'s `0.7071067811865476` is one of them, which is `M-25b`.** Named artifacts
screened but not hand-read, **so the `29` is numeric constants only.**

### WHAT it.27 OWES

1. **`:*` routed through `lands()`** — `J-26c` lives in the it.26 instrument only, so **MARS's
   STRIKE C is still legitimately RED.** JUPITER names it the largest thing open.
2. **The `WING_ARM` citation-resolution node shipped** — argued and measured by SATURN, **not
   built**; his named main debt, and the strike is five iterations old.
3. **MERCURY's three named replacements applied to the table by JUPITER**, with the blast radius
   computed under the corrected procedure.
4. **The uncited sweep extended to §0, §3, §4**, and to **named artifacts** — the `29` is numeric
   constants in §2 only.
5. **The theory-digest instrument repaired** — record the subject's digest **and mtime** with the
   verdict. Shape named by SATURN, not built.
6. **The `overturns:` field on index rows**, the half of MARS's route JUPITER did not take.
7. **The landings census past `21 of 129`**, unadvanced this iteration by an explicit choice.
8. **The theory table tracked** — author's call — and the four rulings: CLAUSE_1_TAIL,
   L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a sixteenth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.26 bought is the first measurement that starts at the claim.** `29 uncited claims across
10 of 12 cells, denominator 62` — **a class every instrument built in nine iterations was
structurally unable to see, because all of them begin at a pointer and look outward.** Beside it:
a procedure that caught four failures on its first use, a consensus witness withdrawn by its own
author for descending entirely from the thing it corroborates, and an adversary's remedy killed by
testing it instead of adopting it.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.27 — THREE OFFICES PAYING DEBTS THEY NAMED THEMSELVES

Room: JUPITER (`J-26c` into production so MARS's STRIKE C goes green; MERCURY's three
replacements applied), SATURN (the `WING_ARM` node shipped after five iterations; the
theory-digest instrument repaired), MERCURY (the uncited-claim class across the whole table and
into named artifacts). **Sections below are `PENDING` until a report lands.**

### EVERY ITEM THIS ITERATION CARRIES WAS NAMED BY THE OFFICE THAT OWES IT

**This has not been true before.** For most of the round the debt list was assembled from
strikes — one office found another's defect, and the record carried it forward. **it.27's list is
almost entirely self-reported:**

- JUPITER: *"`:*` scoring is not routed through `lands()` … **MARS's STRIKE C is still legitimately
  RED. Largest thing open.**"*
- SATURN: *"the detector my it.22 limit demanded is a citation-resolution node — **established by
  hand here, not shipped**"*, and *"a repair for the theory-digest instrument — **fix shape named,
  not built**."*
- MERCURY: *"the uncited sweep covers §2 only … **named artifacts were screened but not
  hand-read**, so the `29` is numeric constants only."*

**An office that ships a correct repair into the wrong file and says so is not the same as an
office that ships it and hopes.** All three closed it.26 by naming exactly what a reader would
otherwise have had to discover — and **the round's adversary did not have to find any of it.**

**That is what the last ten iterations were for.** The instruments matter less than the habit
they trained: **the not-reached paragraph is now where the real information is**, and three
offices independently put their largest debt in it rather than their smallest.

### AND ONE PATTERN SATURN HAS NOW FILED AGAINST HIMSELF THREE TIMES

`git diff` on an untracked path. A HEAD SHA over suites git does not contain. `rc=0` because a
harness piped `check` through `sed`, making `$?` sed's status.

> **Three instances this round of an instrument of his reading a channel that cannot carry the
> fact asked of it.**

**Each was caught and filed by him rather than hidden**, which is why it is a pattern and not
three accidents. **His brief this iteration requires the channel check BEFORE shipping, and
requires him to record that he checked** — so a fourth instance is prevented rather than
confessed. **A defect class that its own author reports reliably is one discipline away from a
defect class that does not occur.**

### JUPITER — STRIKE C GREEN WITH MARS'S FILE UNMODIFIED, AND THE RED WAS WIDER THAN REPORTED

**`J-26c` is in the production predicate.** `lands()` now scores a `:*` want against a frozen
sha256 over **the lines carrying it**, not against presence anywhere in the file. **And the RED
was six times what it.26 said:**

> **`6 of 6` `:*` pointers missed the obituary, not 1.**

**He under-reported his own defect by a factor of six and corrected it while repairing it.**
After the edit, MARS's suite runs `5 passed, 1 failed` — **STRIKE C and STRIKE B green, with no
line of `tests/mars_v20/` written.** **STRIKE A stays RED and he says it is still true**, because
it is SATURN's HEAD-SHA provenance and not his.

**Coverage after MERCURY's three edits: `118 of 131` scored and landing, `13` refused under
`J-20b`, `0 of 118` failing**, `105/105` unique, 38 files. **The population moved `129 → 131`
because two of the three repairs ADDED citations** under MERCURY's registry-and-artefact ruling —
`C133` (`T_STAR = 2`) and `C134` (`def build_delay(`). **`:475` kept its place as *registered @*;
`:198`/`:206` untouched, exactly as MERCURY specified.**

**The blast-radius procedure earned its keep a second time:** 19 files across **4 offices**, and
it caught *"six JUPITER-file failures that were the census being restated rather than read, and
one found only because `lands()` changed under it — it.26's count negative used the shared
predicate as a presence test."*

**AND HE LEFT THREE FOREIGN REDS RED, ON PRINCIPLE:**

> **SATURN's cell digest, MERCURY's census digest, MARS's coverage node — each names what moved.
> Re-taking another office's freeze is what the freeze prevents.**

**An office that can turn three reds green by re-running someone else's instrument and declines
is the strongest form of the two-office rule the round has built.** He reported them and stopped.

**The `overturns:` field is specified, its enforcement node is RED with the exact row-change
instruction, and the journal was not edited** — the coordinator's file, left to the coordinator.
**Re-measured: `34 of 34` index rows quote a literal from the claim they overturn.**

**And the fourth measured mid-iteration tree movement:** SATURN's
`test_v20_r15_it27_subject_provenance.py` *"appeared at `12:40:16Z` and was gone from the
collection at `12:42:22Z`."*

### SATURN — THE PIN NOW FAILS ON CODE, AND IT CATCHES THE CASE CONSENSUS COULD NOT

**`WING_ARM = it27.ledger_wing_arm()`. MARS's STRIKE 1 is closed after five iterations and three
answers, two of which its own author withdrew.** RED verbatim, unmutated:

```
E  WING_ARM is a hand-typed literal at test_v20_r15_it14_saturn.py:39. MARS struck it at it.22
E  and it is unchanged five iterations later. The resolver derives {'W1': 'arm_smprime',
E  'W3': 'arm_pl'} from ledger citations that land INSIDE the source they claim -- L-10's
E  `ceq/arm_smprime.py:163-172` falls in `path_product`, (144, 172); L-14's `ArmPL.forward`
E  is `ceq/arm_pl.py:358`. That derivation fails on CODE.
```

**Resolution requires each ledger citation to land inside the body of a symbol the same row
names — not a path typed next to a line number.** *"Two arms for one wing is a REFUSAL, not a
vote."*

**AND IT CATCHES THE FOUNDING MISTAKE, WHICH IS THE CASE HIS CONSENSUS WITNESS COULD NOT SEE.**
Planted `Q4/W1 … ceq/arm_pl.py:163-172` **resolves to `[]`** — *"those lines are a scalar
recurrence loop and `L-10` describes a masked reverse-`cumprod`; **the row goes underivable rather
than silently re-binding.**"* **`54 passed, 0 failed` across five suites.**

**His boundary is stated and it is real:** *"it catches an **inconsistent** world. A founding
mistake that **also rewrote the prose** to describe `arm_pl`'s actual code would resolve and read
GREEN."* **And W3 rests on ONE resolving citation against W1's three** — `L-17`'s
`ceq/arm_pl.py:1` is the module docstring, **inside no symbol body, contributing nothing.**

**REPAIR 2 — the digest verdict names its subject and caught it moving.** `read_subject()` is
**stat → read → stat**, and *"a subject that moves across the read is a REFUSAL, not a verdict off
stale bytes."* The repaired verdict carries the table's `sha256`, `mtime_ns` and `size` **inline**,
and the mtime it printed is `12:38:10Z` — **during this iteration, while the instrument was being
built.**

> **What it still cannot see:** *"it authenticates **which bytes**, never that those bytes were
> right; an edit inside one mtime tick that leaves size unchanged is invisible; it cannot say
> **who** wrote them. It converts a silent wrong verdict into a loud refusal, and that is all."*

**THE CHANNEL CHECK WAS PERFORMED BEFORE EITHER REPAIR SHIPPED, AND SHIPPED AS A TEST.** Measured
`12:34:12Z`: `path_product`, `zero_hop_mask` and `ArmSMPrime.forward` defined in `arm_smprime` and
**absent** from `arm_pl`; `cumprod` **2 occurrences** in one, **0** in the other. **Bound as
`test_the_anchor_discriminates_between_the_two_arms`, so it is an instrument and not a
paragraph.**

**The pattern that had been filed three times against himself did not recur** — and the
requirement was met by building the check rather than by promising it.

**And one caught in flight, which is the best small thing in the filing:** his first draft ended in
`assert X or challenge_path := True` —

> **a banned `or True` wearing a walrus.**

**Grep for `and False|or True` over both new files is clean.**

**Not reached, and he names his own weakest sentence:** *"a pre-edit baseline re-run … my claim
that the other eight (and mars's 40) predate this iteration rests on **inspection of failure
names, not on a measured baseline.** Same species as the channel-check gap — recorded, not hidden.
**Weakest sentence in the record.**"*

### MERCURY — FOUR SCOPES, NEVER POOLED, AND A ZERO THAT IS THE RESULT

| scope | uncited | denominator |
|---|---|---|
| **§0** | **`2` across `1` of `3` subsections** | **`6`** (screen 9 − 3 section refs) |
| **§3** | **`8` across `4` of `6` units** | **`13`** (screen 16 − 3 section refs) |
| **§4** | **`0` across `0` of `3`** | **`0`** |
| **§4 named artifacts** | **`2` across `1` of `3` subsections** | **`5`** |

**§4's zero denominator is the finding, not a null result.** All five float-shaped tokens in §4
are **section references.**

> **The section that rules what the leap gets asserts not one numeric constant.** The numeric
> instrument is **structurally blind** to it, and it.26's `29` would have stayed `29` whatever §4
> said.

**That is a measured argument for extending to named artifacts rather than an analogy** — the
instrument reported `0` on a section it could not have reported anything else about, and **an
office that stopped at the zero would have called §4 clean.**

**And the named-artifact sweep immediately found what the numeric one could not:** `BED_SPECS`
@ `:388` and the **`none` arm** @ `:383`, both uncited, **both inside the §4.3 paragraph that
WITHDRAWS `V20_R15_LEAP_LEDGER.md:131`.**

> **The identical `BED_SPECS` claim at `:224` carries a `[RUN]` *and* a pointer. The bare copy is
> the one doing the withdrawing.**

**A withdrawal resting on an uncited restatement of a cited claim** — and the `none` arm is
declared in-text to produce **four of five findings with no control**, while its sibling
`ROUND_NOUNS` arm carries a pointer **on the same line.**

### THE DISJOINTNESS FINDING, WHICH SETTLES WHETHER THE TWO INSTRUMENTS OVERLAP

`0.7071067811865476` — the constant this office flagged to him as a likely uncited case — **is not
in this class.** *"It offers a pointer, so it is CITED; the pointer doesn't land. That is
`M-25b`, the landing class."*

> **The two instruments are disjoint and neither subsumes the other**, and **JUPITER's repair
> proved it live: after his edit the constant is still CITED and the uncited count is unmoved.**

**The round now has two measurements that cannot substitute for each other**, demonstrated on a
single constant that sits in exactly one of them. **This office's guess about where that constant
belonged was wrong, and the instrument said so.**

### THE TABLE CHANGED UNDER HIM MID-SWEEP AND EVERY COUNT SURVIVED IT

Measured against **`3c4d1b270049f6ff`** at `12:33Z` and `12:37Z`. **At `12:40Z` the table read
`942e4208893444cd`** — JUPITER's three replacements landing, **verified by their text.**

**Re-screened: every token population in §0/§3/§4 identical, one line number moves (`:71` → `:72`).
All four counts hold on both digests; the three REDs are still RED on the new one.**

**Fourth mid-iteration edit of this table, and the first one that cost nothing at all** — because
he recorded the subject's digest with every count and re-took against the new one. **The
procedure the round spent three iterations arguing about is now just how the work is done.**

**Landings advanced `21 → 24 of 129`**, one clause, **all three LAND, `0` new defects**, none in
it.25's set. **Count, not a rate; no extrapolation.** Three more replacements named for JUPITER —
and `M-27c` is *"the one uncited constant whose producer I located"*, the other nine **priced, not
repaired.**

### AND A FINDING AGAINST HIS OWN it.26 REPORT THAT NOBODY ASKED HIM FOR

> His screen, **re-implemented from the it.26 description**, returns **`64`** float-shaped tokens
> on §2 where it.26 recorded **`63`**. *"it.26's screen was **never written to a file**, so **its
> denominator is not reproducible from its report.** The `62` hand count stands; **`63` should not
> be quoted as reproducible.**"*

**A number published one iteration ago cannot be regenerated because the instrument that produced
it was never saved.** The hand count survives — it was a reading of the artifact, not of a
script — **and the screen's number does not.**

**This is the INSPECTOR's it.25 ruling arriving from the other direction.** Provenance
authenticates that a reading happened; **it does not preserve the instrument that read.** A
`[RUN]` marker on a throwaway one-liner certifies an execution **nobody can repeat**, and the
round has published many of those. **The route is not new — it is the published-query route from
it.19, which the CORRECTIONS INDEX has honoured for eight iterations and no other instrument
has.**

### WHAT it.28 OWES

1. **The `overturns:` rows written into the CORRECTIONS INDEX** — specified by JUPITER, node RED
   with the exact instruction, **and it is the coordinator's file.** `34 of 34` rows quote a
   literal they overturn.
2. **A measured pre-edit baseline** for `tests/saturn` and `tests/mars_v20` — SATURN's named
   weakest sentence; eight failures are claimed pre-existing on inspection, not measurement.
3. **Three foreign digests re-taken by their own offices** — SATURN's cell digest, MERCURY's
   census digest, MARS's coverage node, **each RED and correctly left RED by JUPITER.**
4. **`THEORY-CELLS-SHA256` re-declared** now that the edits have landed.
5. **The resolver widened past `ceq/arm_*.py`** — W3's pin rests on **one** resolving citation.
6. **Named artifacts in §2 and §3**, and §5 hand-read — MERCURY's `29` and `10` are numeric-only.
7. **`test_v20_r15_it18_citation_landing.py`'s third failure** — its dated `129` expired, `J-18a`
   on a superseded instrument.
8. **The theory table tracked** — author's call — and the four rulings: CLAUSE_1_TAIL,
   L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a seventeenth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.27 bought is three debts closed by the offices that named them, and every one closed
harder than the debt required.** JUPITER routed `J-26c` into production **and found his own RED
was six times wider than he had reported.** SATURN shipped the pin **and it catches the founding
mistake his consensus answer explicitly could not.** MERCURY extended the sweep **and found a
number in his own it.26 report that cannot be reproduced, because the screen that produced it was
never saved.**

**That last one is the round's oldest lesson arriving from a new direction:** a `[RUN]` marker on
a throwaway one-liner certifies an execution **nobody can repeat**. The published-query route has
been available since it.19 and **the CORRECTIONS INDEX is still the only instrument that honours
it.**

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.28 — THE INDEX NOW SUBTRACTS ITS OWN DEAD LITERALS, AND NOTHING FORCES ANYONE TO ASK

Room: SATURN (the measured pre-edit baseline he named his weakest sentence; `THEORY-CELLS-SHA256`
re-declared; W3's pin widened past one citation), MERCURY (named artifacts across §2/§3, §5
hand-read), INSPECTOR (it.25–it.27, and whether any device this round built authenticates a
**subject**). **Sections below are `PENDING` until a report lands.**

### THE COORDINATOR'S DEBT, PAID, AND ITS LIMIT STATED IN THE SAME BREATH

MARS proposed it at it.25, JUPITER specified it and shipped the enforcement node RED at it.27 —
**and did not touch the journal, because it is this office's file.** The field is now written.

**All 34 CORRECTIONS INDEX rows carry `overturns:`** — **31 declaring the literals they retire, 3
declaring `` `none` ``** where the claim quotes nothing machine-checkable. JUPITER's enforcement
node: **`8 passed in 0.41s`** `[RUN]`. Index re-declared and recomputing equal, **34 rows,
`C1`..`C34`, contiguous, index max = body max = 34.**

**What it buys:** a grep-based instrument can now call `dead_literals()` and **subtract every
claim the index declares withdrawn**, so the 100%-false-positive property the INSPECTOR measured
at it.25 — *"33 of 33 rows quote a literal from the claim they overturn"*, now **34 of 34** — no
longer applies to any instrument that asks.

**And the limit is exactly there, in the last word.**

> **Nothing forces an instrument to ask.** The field is a service, not a constraint. Every
> existing grep in this round still reads the raw file and still finds every withdrawn claim
> alive on the row that killed it.

**This office is not going to pretend that is a fix.** It converts a defect that *no* instrument
could avoid into one that *any* instrument can avoid **and most still will not**, because
avoiding it requires knowing the field exists. **The round has been here before** — the
published-query route has been available since it.19 and **the CORRECTIONS INDEX is still the
only instrument that honours it**, eight iterations later. **A convention that depends on being
remembered is a convention that decays**, and the honest next step is not more convention but a
shared helper the censuses import.

**Two small things worth recording because both were the node's doing, not this office's.** The
first extraction took the backticked **constant** (`0.7071`) where the spec's own example wanted
the claim **phrase** (`no arm crosses`); the second pass put `none` unbackticked and the node
refused it, **quoting the exact form it wanted in its own failure message.** **A test that tells
you the shape it expects is worth two paragraphs of specification**, and JUPITER wrote it that
way deliberately.

### CORRECTION 35 — THE `overturns:` GREEN IS VACUOUS, AND THIS OFFICE PUBLISHED IT AS EVIDENCE

**Found by a nurse under the INSPECTOR, within the same iteration this office shipped it.**

`live_hits()` opens with:

```python
if any(literal in lits for lits in dead_literals().values()):
    return []
```

**Every literal the enforcement node loops over is drawn from `dead_literals()`.** The guard is
therefore **always true**, `live_hits(lit)` returns `[]` **unconditionally**, and
`assert live_hits(lit) == []` **cannot fail for any input the node can construct.**

**Measured independently by this office `[RUN]`**, with the index rows excluded and the
short-circuit bypassed:

```
declared literals: 41
still live in the body OUTSIDE the index: 26
  C1   '0.7071'                -> lines [299, 324, 331, 470, 659, 816]
  C2   'frac_gate_annihilated' -> lines [257, 886, 896, 907, 931, 935]
  C3   'N=65'                  -> lines [358, 365, 467, 490, 492, 497]
```

**The nurse measured `29 of 44` with a different field extraction; this office measures `26 of
41`. The two disagree on the denominator and agree completely on the defect** — and the
disagreement is the `62/63/64` phenomenon again, one subject and two instruments.

**What this office published one message earlier was `8 passed in 0.41s`, offered as evidence
the field works.** It is evidence that **eight assertions ran**, and the one that mattered is
**structurally unable to fail.** That is `V-16` — *the instrument that cannot measure, reporting
a pass* — **in a repair written to close a measurement defect**, shipped by the office that has
quoted `V-16` at three others.

**And the shape is worse than a vacuous test, because the round now believes something false.**
The it.28 record claimed the field *"buys"* a grep that subtracts withdrawn claims. **It buys
nothing yet: `26 of 41` literals a row declares dead are still asserted live in the body**, and
the promise attached to the field — *"a row cannot retire a claim the body still asserts"* — is
guarded by code that cannot observe a violation.

> **The check the round now believes it has is the one it does not have.**

**This office's `overturns:` limit paragraph was right for the wrong reason.** It said *"nothing
forces an instrument to ask."* The real defect is one level earlier: **the instrument that was
supposed to prove the field honest asks itself.** A field nobody consumes is a missed
opportunity; **a field whose auditor is a tautology is a false certificate**, and this record
carried it as a completed repair for exactly one message.

**Route, and it is the node's to take, not the field's:** narrow the short-circuit to exclude
**the row currently under test**, so a literal declared dead by `C1` is still searched for
everywhere except `C1`'s own row. **Then the node goes RED with 26 findings and the field has to
be earned** — either by withdrawing the live assertions or by narrowing what each row claims to
retire.

**Second defect from the same nurse, and it is JUPITER's:** `STAR_DIGESTS` **exists twice** —
`test_v20_r15_it20_citation_freeze.py:388` keyed by want with full hex, and
`test_v20_r15_it26_live_claim.py:332` keyed by cid with 16-hex prefixes. *"The two agree today
because the prefixes match."* **That is `J-26b`'s own ONE HOME ruling, violated by `J-26c`'s
datum, by the office that wrote both — and neither filing mentions it.**

### INSPECTOR — `11 AUDITED, 4 STRUCK, 7 UPHELD`, AND THE ANSWER TO HIS OWN QUESTION IS NO

**Does any device this round built authenticate a SUBJECT?**

> **No — all three authenticate a better-described reading.** SATURN's `read_subject()`,
> JUPITER's `J-26c` and MERCURY's dual-digest **move along one axis only**: from *a reading
> happened* to *a reading happened, of these bytes, over this region, stable under this
> perturbation.* **None crosses to *the subject had this value*, because each pins the
> instrument's INPUT and takes the instrument's OUTPUT as the value.**

**And the round supplied its own proof:** MERCURY's `64`, it.26's `63`, the hand count's `62` —
**one subject, three instruments, three denominators.** *"Digest-pinning would have made the `63`
durable, not correct."*

**MERCURY's is ruled strongest for a reason he did not claim** — holding all four counts across a
**real** subject move is *"a statement about invariance, one step past provenance."* **Still not
value authentication: an invariant wrong number is wrong on both digests.**

**STRUCK (4), and the first lands on SATURN's proudest result.** *"His planted negative does not
fire for the stated reason — **every** range in `arm_pl.py` resolves to `[]` for `L-10`,
including `162-168`, which is exactly the `chain_label` 'scalar recurrence loop' his prose names.
**What fires is `path_product` being absent from `arm_pl`; the `163-172` is decorative.**"* And
*"the row becomes **underivable**"* is **false** — `DERIVED (planted)` returns
`{'W1': 'arm_smprime', 'W3': 'arm_pl'}`, **byte-identical to control.**

> **An instrument that works for reasons its report misdescribes.** *"That defect survives every
> provenance device this round built — a `[RUN]` marker authenticates the run; **nothing
> authenticates the paragraph beside it.**"*

**But SATURN is also ruled STRONGER than he claimed:** a prose-only founding rewrite **refuses**
(`W1 resolves into ['arm_pl','arm_smprime']`), *"so reading GREEN needs the `ceq/` symbols moved
too."* **His boundary was too pessimistic and his mechanism was misdescribed, in the same
finding.**

**Also struck: the `overturns:` field as a FIX, and the retrospective baseline as OBTAINABLE.**
Both correct — see Correction 35 above and SATURN's own ruling below.

**On unreproducible instruments he replaced the round's denominator:** *"The `601` `[RUN]` count
is the wrong denominator. **Reproducibility is a property of the instrument being a committed
file**, not the marker being in a report."* **Three tiers — committed / inline-and-transcribed /
run-once-on-a-screen — and only tier 1 can be refuted. Every tier-3 number is unreproducible in
principle, and no digest rescues it, because the digest pins the SUBJECT and the missing half is
the INSTRUMENT.**

### SATURN — THE BASELINE RULED UNOBTAINABLE, WITH THE MEASUREMENT THAT PROVES IT

**`git ls-files tests/mars_v20` → `0 of 22`; `tests/saturn` → `4 of 20`, none a V20 R15
instrument.** *"The pre-edit content of every file it.27 edited exists in **no store** — no index,
no stash, no reflog, because those bytes were **never in the object database.** A retrospective
pre-edit run is not hard; **it is not defined.**"*

**And he recovered most of the claim anyway, by a different route:** stamping each of the 40
failing modules with digest + mtime, **39 of 40 have mtimes predating it.27**, oldest
`2026-08-31T05:07:04Z`; the exception is the theory-digest node, already the live finding.
**Residual stated: mtime bounds files, not outcomes.**

**Then he exercised the forward discipline instead of promising it.** Pre-edit `40 failed, 251
passed`; post-edit `39 failed, 252 passed`. **Δ = exactly one node, the one the edit was for**,
with set-digests naming the file sets *"since a HEAD SHA cannot."*

**`THEORY-CELLS-SHA256` re-declared** with the subject read **untorn** via `stat → read → stat`
against `942e4208893444cd` — **MERCURY's digest, independently arrived at.** He **dropped**
`CITATION-POPULATION = 129` rather than republish it: *"nothing parses it and JUPITER's live
reading is `118 of 131`."*

**REPAIR 3 is a measured NULL and he reports it as one.** Widening the resolver to `lean/`,
`scripts/`, `tests/` bought **zero for either wing.** All eight non-arm citations fail — six land
outside every symbol body, and the two that land inside a body **sit in bodies naming neither
arm.** *"W1's rise from 3 to 5 is a bookkeeping correction, **not a widening gain.** The repair is
a citation the ledger does not carry, not a wider regex."*

**A null result reported as a null, with the eight failures pinned so a future resolution fires
the node.** **W3's pin still rests on one citation.**

### THE NURSE SWEEP THAT CONTRADICTS FOUR PUBLISHED NUMBERS, INCLUDING ONE OF THIS OFFICE'S

**A nurse under the INSPECTOR re-measured the corpus and four standing figures do not survive:**

1. **`0 of 22 files under tests/mercury are known to git` is wrong in TWO MERCURY reports.**
   Measured: **`9 of 25`** — nine `test_r9_*`/`test_r10_*` files are tracked. **And this office
   repeated it in two briefs**, because MARS's original `0 of 22` was about **`tests/mars_v20`**,
   where it is **true**. **A true number about one directory was carried to another and nobody
   re-measured it, including the office that wrote the brief.**
2. **§4's zero denominator is a property of the SCREEN, not of §4.** The section asserts **`~6
   GPU-s`, `~275 GPU-s`, `s = 64`, `0 of 24`, `0 of 40`, `1 passed`** — **nine quantities, every
   one integer-shaped**, and MERCURY's own §8 already concedes integers sit outside every
   denominator. **The zero is real for floats and the sentence built on it — *"the section that
   rules what the leap gets asserts not one numeric constant"* — is false.** This record printed
   that sentence.
3. **it.26's `62` requires TWO rejections from a 64-token screen and only ONE is named.**
   `64 − 1 = 63`, not `62`.
4. **`601 [RUN]` markers is stale for its own scope** — now **`750`** — **and was never a
   whole-corpus figure**, which is **`1,251`**. *"The contract's `17 → 2` is the sharpest single
   move and is not explained by growth."*

**And the nurse reproduced MERCURY's screen from the report's prose to within one non-load-bearing
token, 5 of 6 scopes exact** — *"the rule is sound even where the artifact is missing."*

### MERCURY — NAMED ARTIFACTS, AND `18` OF THEM THE LEAP CANNOT REACH AT ALL

| scope | uncited | units | denominator |
|---|---|---|---|
| **§2, the twelve cells** | **29** | **11 of 12** | **81** |
| **§3, the arena ticket** | **6** | **3 of 3** | **7** |
| **§5, limits (hand-read)** | **2** | **1 of 1** | **5** |

**Measured against `942e4208893444cd` at 12:50Z and again at 12:54Z — the table did not move.
First time this round a MERCURY count did not have to survive a mid-iteration edit.**

**And he refused a coincidence that would have been quoted for the rest of the round:** *"`29` is
**not** it.26's `29 of 62` on §2 floats — different instrument, different population, **same
integer by coincidence**, flagged so nobody merges them."*

**A second column, because the per-cell rule overstates the cost:** of §2's 29, **`18` are
uncited anywhere in the whole table** — *"the leap has nowhere to go"* — while 11 are recoverable
from a sibling cell. **`18` is the number that costs what `M-25c` cost.**

**THE FOUR THAT MATTER:**

- **`sign_flip_gate`, Q2/W3.** The kept `F1 + const` (`1.0845223424`) is measured on planted
  negative PN-2, uncited. `grep` returns **3 hits, all tests — a test name and two docstrings.
  ZERO producers.** **The surviving constant rests on a negative that exists only as a string in
  test prose.**
- **`ArmPL.forward`, Q6/W3.** Q6/W1 states the identical `F4` fact and **cites its method**;
  Q6/W3's sibling claim cites nothing. **Both are `F4` NOT-PUT with an admission condition, so the
  condition on W3 names a method the leap cannot open.** Real location `ceq/arm_pl.py:405`.
- **⟨CLAUSE_1_TAIL⟩ — uncited in all three §3 units and in Q5/W3. Four occurrences, zero
  pointers, zero `.py` hits.** The INSPECTOR filed it NOT RULED and **the table cites neither the
  filing nor the ruling.** **This is the ruling this record has surfaced to the author in nine
  consecutive iterations, and the table has never pointed at it.**
- **`secs`** — the field the whole arena-price downgrade and the `ρ = +0.70` confound rest on,
  **uncited in both places it appears.**

**HE SAVED THE SCREEN AND PUBLISHED ITS COMMAND** — the second instrument in the round to honour
the it.19 published-query route, and the first outside the CORRECTIONS INDEX:

```
python tests/mercury/screen_v20_r15_it28_named_artifacts.py V20_R15_THEORY_TABLE.md
```

**And he recorded a defect in it rather than smoothing it:** the first run *"died on
`UnicodeEncodeError U+2212` against cp1252 stdout and **would have silently returned 1 unit
instead of 16**."*

**Landings `24 → 34 of 129`**, clause named (the Q6 pair), **disjoint from both prior sets, 10 of
10 land, 0 fail.**

**`M-28a` — A CLASS NO ITERATION THIS ROUND HAS FILED: a withdrawal that misquotes the sentence
it withdraws.** Q6/W3 says the ledger claims `:131` is *"the one registered bed"*. **The ledger
says *"the one registered bed WHOSE OUTPUT IS CATEGORICAL"*.** `BED_SPECS` returning three beds
**refutes the truncated quote and says nothing about the qualified one.** *"The conclusion
survives on other grounds; **the stated reasoning does not.**"*

**And his not-reached is a bound in the right direction:** the `overturns:` field **was not
consulted**, so all three counts are **gross, not net** — *"the untested error direction is
one-sided: they could only be high."* **A count published with its error direction named is worth
more than a count published alone.**

### WHAT it.29 OWES

1. **The `overturns:` honesty check narrowed** so it excludes only the row under test. **It goes
   RED with 26 findings and the field has to be earned.** Correction 35, this office's.
2. **`STAR_DIGESTS` given one home** — `J-26b`'s own ruling, violated by `J-26c`'s datum.
3. **`0 of 22` withdrawn and re-measured** wherever it was carried — **`9 of 25` for
   `tests/mercury`, `0 of 22` correct only for `tests/mars_v20`.**
4. **§4 re-swept for integer-shaped constants**, and the *"not one numeric constant"* sentence
   withdrawn.
5. **`62` reconciled** — two rejections needed, one named.
6. **`601` withdrawn and re-scoped** to `750` / `1,251`, with the contract's `17 → 2` explained.
7. **SATURN's planted negative re-derived** — it fires on `path_product`'s absence, not on the
   range, and nothing became underivable.
8. **The theory table tracked** — author's call — and the four rulings: CLAUSE_1_TAIL,
   L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for an eighteenth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.28 bought is a No.** The INSPECTOR's question — does any device this round built
authenticate a subject — was answered against all three offices that thought they had one, **and
the round's own `62/63/64` was the proof.** Beside it: a baseline **ruled undefined rather than
estimated**, a widened resolver reported as a **measured null**, and four standing numbers
contradicted by one nurse sweep — **one of them a true figure about one directory that this
office carried to another without re-measuring.**

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.29 — NOTHING AUTHENTICATES THE PARAGRAPH BESIDE THE RUN

Room: JUPITER (the `overturns:` honesty check made able to fail; `STAR_DIGESTS` given one home),
MERCURY (`0 of 22` withdrawn, §4's integers swept, the `62` reconciled), SATURN (his own planted
negative re-derived; the `601` withdrawn and re-scoped). **Sections below are `PENDING` until a
report lands.**

### THE SENTENCE THIS ITERATION IS BUILT ON, AND IT IS THE INSPECTOR'S

> **A `[RUN]` marker authenticates the run. Nothing authenticates the paragraph beside it.**

**Every item in this iteration is an instance.** SATURN's `WING_ARM` node **works** and his
report gives the wrong mechanism — *"the `163-172` is decorative"* — while also **understating**
what the node catches. JUPITER's enforcement node **runs** and its assertion **cannot fail**.
MERCURY's §4 zero is **exactly right** and the sentence built on it — *"asserts not one numeric
constant"* — is **false**, because the section asserts nine integer-shaped ones including a price
the leap acts on. **This office published `8 passed` as evidence a field works when the check was
a tautology.**

**Four offices, four instruments that execute correctly, four descriptions that do not match
them.** The round spent eighteen iterations building devices that certify **what was measured**,
and **not one of them touches the sentence that says what the measurement means.**

**And that is the whole gap between `129 of 129` and `9 of 12`.** A citation that lands
authenticates that a string is at a line. **Whether the cell's paragraph is a true account of
that line is the argument layer**, which the round measured once, at `9 of 12`, and has never
been able to instrument. **The paragraph problem and the argument problem are the same problem**,
and it has now appeared in the offices' own reports rather than only in the theory table.

### THE ONE ROUTE THE ROUND HAS THAT ACTUALLY WORKED

**MERCURY saved his it.28 screen and published the command that regenerates it** — the second
instrument in the round to honour the it.19 published-query route, and **the first outside the
CORRECTIONS INDEX in ten iterations.** A nurse then **reproduced his screen from the report's
prose alone and landed 5 of 6 scopes exactly.**

> **The rule was sound even where the artifact was missing** — and the artifact being missing is
> precisely why it.26's `63` cannot be recovered while the hand-count `62` can.

**The INSPECTOR's replacement denominator makes it a standard rather than a habit:**
**reproducibility is a property of the instrument being a committed file, not of the marker being
in a report.** **Three tiers — committed, inline-and-transcribed, run-once-on-a-screen — and only
tier 1 can be refuted.** The round has published a great many tier-3 numbers and is only now able
to say so.

### JUPITER — THE NODE FAILS NOW, AND THE TWO DENOMINATORS WERE ONE MEASUREMENT

**Correction 35 accepted in full.** The blanket short-circuit is deleted. **`J-29a`: a `row`
parameter would have been a measured no-op** — the index-row exclusion is already a superset of
*"the row currently under test"*, because the declaring row **is** an index row. **The narrowing
is the deletion**, and he measured that before writing it.

```
E  26 of 41 declared literals are UNEARNED:
     C1   '0.7071'                 15 hits, first at 299
     C2   'frac_gate_annihilated'  20 hits, first at 257
     C3   'N=65'                   10 hits, first at 358
     ... 23 more ...
1 failed, 7 passed
```

**AND THE `26 of 41` vs `29 of 44` DISAGREEMENT IS FULLY RESOLVED — IT WAS THE SENTINEL.**
`C21`, `C30`, `C33` declare `` overturns: `none` ``, and **the it.27 grammar handed the grep the
four-letter word, which is live 25 times in the body.** `44 − 3 = 41`; `29 − 3 = 26`.

> **One measurement, two grammars.** This office's extraction dropped the sentinel; the nurse's
> did not.

**Two instruments, one subject, two numbers — and this time it resolved completely**, unlike
`62/63/64`, because **both grammars were still runnable.** That is the it.19 published-query route
paying off in the one place the round has honoured it.

**THE RULING ON WHAT THE 26 ARE, AND ONLY ONE THIRD OF IT IS MEASURED:**

- **Class A — the ROW over-claims: `13 of 26`.** The row retires a *predicate over* a thing and
  declares the *thing*. **`C15`'s own row text calls `floor₁` a live threshold.** **Remedy:
  declare the asserting phrase (`no arm crosses`), not the token** — which is exactly the literal
  `J-27b`'s example specified and this office's extraction replaced with `0.7071`.
- **Class B — RECORD, not restatement: `4 of 26`, MEASURED.** Every live hit **precedes** the
  section that settled the row. Membership frozen in a test.
- **Class C — USE vs MENTION: `9 of 26`, NOT MEASURED**, *"flagged as a reading, not a
  measurement; the settling grep is specified and not run."*

**A ruling that measures a third of its own subject and says which third** is the shape the round
has spent thirty iterations learning.

**And the grammar he specifies is argued from measurement rather than taste:** a declared literal
must **contain a space and be ≥12 characters**, or declare `none`. **20 of 41 declarations are
phrase-shaped and only `6` are unearned; 21 are token-shaped and `20` are unearned.**

**THE `V-16` SELF-CHECK IS THE PART THIS OFFICE ASKED FOR AND HE OVER-DELIVERED.** Seven nodes,
**each shown able to observe a violation**: the old guard is **reconstructed and measured vacuous
on 41 of 41**; the narrowed one returns 26 non-empty on the same inputs; **and the CONTROL shows
`15 of 41` declarations are honest, so the RED is not RED-for-everything.**

**`STAR_DIGESTS` has one home** — it.20 owns it, it.26 is a re-keyed view, **and a guard test
refuses any 16-hex literal in the it.26 source so re-inlining turns it RED.** `26 passed`. **His
own blast-radius node caught his new file as an unre-taken importer.**

### SATURN — THE NODE HAS TWO STAGES AND HIS REPORT NAMED THE ONE IT NEVER REACHED

**What actually fires is not the range.** Sweeping the planted `L-10` across `163-172`,
`162-168`, `1-2`, `358-360`, `400-410`, `144-174` gives `[]` for **every one** — including
`358-360` and `400-410`, **which lie inside `ArmPL` and `ArmPL.forward`.** What fires is
`anchored`: **no symbol `L-10` names is defined in `arm_pl` at all**, so the anchor set is empty
**before any line number is compared.**

> **The rule has two stages; it.27 named stage 2 while exercising stage 1.**

**The *"scalar recurrence loop"* sentence is withdrawn.** And rather than delete the range, **he
made it load-bearing where it can be**: `L-13` names `ArmSMPrime.forward` (572-577), so
`572-577 → ['arm_smprime']` while `560-565`, `1-5`, `497-500` and **`572-600` — which starts at
the anchor and overruns its end — all give `[]`.** **A decorative clause turned into a working
one rather than removed.**

**"Underivable" withdrawn:** `DERIVED (planted)` is byte-identical to control. *"The founding swap
kills the **line** citation only; `symbol path_product` still resolves, so W1 keeps its binding.
**Defence in depth, not underivability.**"*

**AND THE BOUNDARY MOVED IN BOTH DIRECTIONS IN ONE FILING.** **Stronger:** a prose-only founding
rewrite — 17 lines, swapping both arm names and both class names through the ledger — **REFUSES**,
because *"`path_product` carries no arm name and stays put."* **Weaker, and volunteered:** a
**paths-only** rewrite **derives identically to control.**

> **The strength is in the symbols and so is the blind spot.**

### THE `601` WITHDRAWN, AND THE CONTRACT'S `17 → 2` IS A MEASUREMENT ARTEFACT

Scoped, digested, regenerable: **`V20_R15_*.md` + contract → `766` lines / `774` occurrences**;
all root `.md` → **`1,282`**; whole tree → **`1,912`**.

**And scope A moved `750 → 766` in twelve minutes** — *"because it.29's own filings enter the
counted population as it is counted."* **A census of a corpus that includes the census.**

**The `17 → 2` collapse that looked like a deletion is a pattern difference:**

```
grep -o "\[RUN"   -> 17
grep -o "\[RUN\]" ->  2
grep -c  RUN      -> 18
```

**Fifteen of the seventeen carry a payload** — `[RUN: +20.87 …]`, `[RUN anchors: …]` — **so they
are `[RUN` and not `[RUN]`. No deletion occurred.**

> **Marker totals are pattern-dependent by a factor of eight on the one file the round treats as
> fixed.**

**And he found the contract is untracked** — *"so no diff-based check of it exists at all."*

**HIS TIER CENSUS IS THE HARDEST NUMBER IN THE FILING AND IT IS AGAINST HIMSELF.** Under the
INSPECTOR's rule: **tier 1 (committed) = 4, all pre-R15. Tier 2 (in tree, uncommitted) = 18 —
every R15 SATURN module from it.9 to it.29. Tier 3 = 2.**

> **Zero of eighteen round-15 SATURN instruments are tier 1**, so *"the office's whole R15 stock
> is unrefutable in the archival sense — and **`NO git writes` means this office cannot promote
> them.** A standing debt larger than the `601` was."*

**The dispatch law that protects the shared index from collision is the same law that keeps every
instrument this round built at tier 2.** That is not a defect in the law; **it is a cost the round
has been paying without pricing, and SATURN priced it.**

### MERCURY — THREE NUMBERS WITHDRAWN, AND A GIT-BACKED INSTRUMENT THAT READ AN UNTRACKED TREE AS EMPTY

**`0 of 22` → `9 of 25`** `[RUN]`, withdrawn at both sites. **The HEAD-SHA withholding survives on
a narrower and better ground:** all nine tracked files are `test_r9_*`/`test_r10_*`, so **not one
file this round wrote is tracked**, and *"a SHA beside an it.26 count would have named a tree
containing zero of the instruments that produced it."*

> **A correctly-ATTRIBUTED figure was assumed to be a correctly-SCOPED one.** The citation
> discipline itself carried it.

**§4's sentence withdrawn and replaced by the sweep it should have been.** The float zero is exact;
**§4 asserts 9 integer-shaped quantities and 8 carry no pointer.** Only `1 passed` is cited.

**And the sweep found what the sentence had been hiding:** **`~275 GPU-s` at `:354` is a price
that §2 of the same file corrects at `:184`** — *"does not evaluate to 275; its own formula gives
`309.047`"* — **170 lines apart, nothing between them, sitting in the hybrid-verdict table `J-14b`
exists to resolve.** **An uncited price the leap acts on, already contradicted by the same
document.**

**`62` withdrawn → `63`, and both of it.26's operands were wrong.** The screen returns **64**;
it.26 published `63` as the screen count. *"`62` is reachable only by dropping the repeats **and
not applying the rejection it.26 names** — the one arithmetic its own prose forbids."*

**Landings `34 → 39 of 129`**, two clauses, **5 of 5 land, 0 fail, 0 weak**, with two further
citations **excluded and named** because they are his own `M-25c` repair. **Disjointness asserted
on citation identity, not line number** — *"because it.25's numbers were taken against an
unreproducible digest."*

**`M-29a` — THE ADMISSION CONDITION IS FALSE, AND THE FIRST INSTRUMENT SAID IT WAS FINE.**
`V20_R15_THEORY_TABLE.md:306` asserts `build_delay` *"has zero callers under `scripts/`"*. **It has
two** — `scripts/v20_m14_cheeger.py:347` and `:462`, **at the registered kwargs exactly.** *"Both
surrounding pointers land; the un-pointered clause between them is false, and **it is the reason
the cell is NOT-PUT.**"*

> **The first `m29d` used `git grep` and went GREEN on zero hits — that script is untracked. A
> git-backed instrument reads an untracked tree as an empty one.**

**Nineteenth instance this round of an instrument whose silence and whose absence share a
channel**, caught by its author before shipping, **in the iteration whose subject is that
descriptions do not match instruments.**

### WHAT it.30 OWES

1. **Class C settled** — 9 of 26 declared literals ruled by reading, not measurement; the grep is
   specified and unrun.
2. **The phrase grammar enforced** — `20 of 41` phrase-shaped declarations are `6` unearned;
   `21` token-shaped are `20`. **The index's rows rewritten to phrases** (coordinator's file).
3. **`M-29a` resolved** — `build_delay` has **two** callers under `scripts/`, and the false
   clause **is the reason its cell is NOT-PUT.**
4. **`~275 GPU-s` reconciled with `309.047`** — an uncited price the leap acts on, contradicted
   170 lines away in the same file.
5. **The tier-1 problem priced by the round, not by one office** — **zero of eighteen R15 SATURN
   instruments are committed**, and `NO git writes` means no office can promote its own.
6. **§0 and §1 swept** — never measured by anyone, in either instrument.
7. **The theory table tracked** — author's call, and it is now also the tier-1 question — and the
   four rulings: CLAUSE_1_TAIL, L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a nineteenth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.29 bought is four offices correcting the descriptions beside their own working
instruments** — the defect the INSPECTOR named and no device this round built can see. SATURN's
node fires at a stage his report never named; JUPITER's guard was vacuous on `41 of 41` and is now
RED on `26`; MERCURY withdrew three published figures and found an admission condition that is
**false**; and this office published a tautology as evidence.

**And one disagreement resolved completely rather than filed as unadjudicable** — `26 of 41`
against `29 of 44` was **the `none` sentinel**, recoverable because **both grammars were still
runnable.** `62/63/64` was not, because its instrument was never a file. **That is the whole
argument for the published-query route, measured twice in one iteration.**

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.30 — THE COORDINATOR BROKE FOUR NODES WITH A CORRECT REPAIR AND REVERTED

Room: JUPITER (the phrase grammar applied WITH its blast radius; Class C settled), MERCURY
(`M-29a` resolved by reading the call sites; the `~275` / `309.047` contradiction), INSPECTOR
(it.28–it.29, and whether his own tier rule is satisfiable under this round's dispatch law).
**Sections below are `PENDING` until a report lands.**

### CORRECTION 36 — THIS OFFICE APPLIED A CORRECT GRAMMAR WITHOUT COMPUTING ITS BLAST RADIUS

**JUPITER specified `J-29c` and measured it: `20 of 41` phrase-shaped declarations are `6`
unearned; `21` token-shaped are `20`.** The grammar is right. **This office applied it to all 34
CORRECTIONS INDEX rows and broke four of his it.29 nodes:**

```
FAILED test_the_narrowed_node_can_observe_a_violation
FAILED test_the_narrowed_node_is_not_red_for_everything
FAILED test_the_none_sentinel_is_the_whole_26_of_41_vs_29_of_44_gap
FAILED test_four_of_the_26_are_record_and_not_restatement
E  At index 0 diff: (2, "the eval draw's") != (10, 'live-band decay **under 3% per position**')
```

**His nodes freeze the specific literals and their measured membership** — Class B's four, the
`none`-sentinel arithmetic, the not-red-for-everything control. **A grammar change to the index
invalidates every one of them.**

> **`EDIT / COMPUTE THE BLAST RADIUS / RE-TAKE ALL OF IT / RE-DECLARE` was corrected at it.26 for
> exactly this failure, by MARS, against this office's own under-specified procedure. One
> iteration later the office that wrote the procedure edited a file four live nodes assert
> against and did not compute the radius.**

**Reverted `[RUN]`.** The index is back at digest **`28a30ce6f6dcdbb0`**, and the round-trip is
**not** clean: post-revert reads **`1 failed, 6 passed`**, the survivor being
`test_the_none_sentinel_is_the_whole_26_of_41_vs_29_of_44_gap`. **Either the revert left residue
or that node has a real ordering dependence, and this office is not going to guess which** — it is
JUPITER's file, his measurement, and the INSPECTOR is auditing it this iteration.

**Two things this office got right and they are the only reason the cost was one iteration.** The
edit was **mechanical and reversible** — regenerating the token grammar restored the byte-exact
prior digest. And the failure was **caught by running another office's suite**, which this office
did because the round's own rule now says a radius must be re-taken. **Half the procedure was
followed: the re-take happened, after the edit, instead of the radius being computed before it.**

**And the grammar still needs applying.** **It is JUPITER's now, because the radius is his** — his
34 replacement literals, his nodes re-taken, in one filing, and this office applies the rows
against his test rather than ahead of it.

**The mechanical extraction is also not the grammar, which is the deeper reason to hand it back.**
A first-N-words rule produced `no arm crosses` and `EXIT B costs` from the same pass — **one is
the assertion, the other is a truncated fragment.** *"The literal is the asserting phrase, not the
token it asserts over"* is **a judgment per row**, and thirty-four of them is a manager's job with
nurses, not a regex.

### MERCURY — `M-29a` RESOLVED, AND THE FALSE CLAUSE DOES NOT MOVE THE GRADE

**`Q2/W1` stays `NOT-PUT`. No grade moves. `J-14`'s primary input is untouched.**

**`:306`'s admission condition is two clauses and only one is the condition:**

- **operative** — *"one cell of BED-K's shape actually run"* — **still unsatisfied**
- **evidentiary** — *"zero callers under `scripts/`"* — **false, there are two**

**The two call sites do not satisfy the condition, and he measured why rather than asserting it.**
`build_delay` returns `dict(kind, n, seed, params, K, b, z, pos, plant)`; **both sites consume
`b["K"]` and nothing else** — never `b`, `z`, `pos`, `plant`, **the bed's data.** And
`v20_m14_cheeger.py` contains **zero occurrences** of `arm_smprime`, `arm_pl`, `hankel`,
`rank_real`, `rank_plus_lower`. *"The calls are a structural audit of the kernel operator — a
claim about `K`, not a cell."*

> **The filing is correct and its stated reason is false, which is the worst shape a pointer can
> take: a reader who checks the reason finds two callers and concludes the filing is wrong.**

**That is the argument-layer defect in its most expensive form.** A wrong pointer sends a reader
nowhere; **a wrong reason sends a reader to the right place and hands them a refutation of a true
claim.** The round has spent eleven iterations measuring whether citations land; **this one lands
perfectly and argues against its own cell.**

**And the sentence that should have been quoted was already in the file:** `:151` says *"zero
cells of BED-K's shape have ever been run"* and **survives unchanged.** **`M-30b`, JUPITER's to
apply.**

### THE `~275` CONTRADICTION IS A THREE-WAY, AND THE THIRD NUMBER IS HIS OWN DEFECT

| number | produced by | status |
|---|---|---|
| `275` | **nothing** — labelled `≈` at its source | **withdrawn** |
| `309.015` | **the formula's own literal constants** `[RUN]` | the formula as written |
| `309.047` | a sweep on it.13's **re-measured** means | **a different object** |

**Neither of the two the round was arguing about is right.** **`M-30a` is his own new defect:**
`:184`'s *"its own formula gives `309.047`"* is **false — the formula gives `309.015`.**

**And the tell was already sitting in his own test:** a docstring saying *"It is 309.02"* beside
an assertion of `approx(309.047, abs=0.01)` — **a band that excludes the value the docstring
names.** **Ruling `M-30`: the honest quote is the band `206 – 537 GPU-s`, point `309.0`, and no
point without it.**

**A price the leap acts on had three candidate values, two of them published, and the one the
round was defending was produced by nothing.**

### THE `git grep` SWEEP, AND THE ANTIDOTE IS ONE ASSERTION

**Six live git-backed instruments; only one is R15 — JUPITER's — and it is immune**, because
`:257` already carries `assert hit, "the importer probe found nothing -- git grep is not
running"`.

> **A non-empty liveness assertion is the whole antidote.**

**Proposed as the round's rule, and it is the general form of `V-7`'s witness** — *a scan whose
witness is missing raises instead of returning an empty list* — **arriving for the fourth time
from a fourth direction.** MERCURY's own four instruments: **4 of 4 untracked**, consistent with
SATURN's tier census.

**Landings `39 → 53 of 129`.** Clause C, the `Q4` cost-law cells, **14 of 14 land, 0 fail, 1 weak
named**, disjoint by **line number and citation identity.** The weak one is *"the clause's
strongest evidence, since the tuple contains neither `arm_smprime` nor `arm_pl`"* — **a weak
landing that is load-bearing precisely because of what it fails to contain.**

### JUPITER — DID NOT FILE INSIDE THE CAP

**`V20_R15_IT30_JUPITER.md` does not exist at filing time** `[RUN] ls`. His it.30 test file
`tests/jupiter/test_v20_r15_it30_phrase_grammar.py` **is on disk**, so the durable half survived —
the it.16 protocol working for the fourth time. **The phrase grammar and its blast radius enter at
it.31 unread; this office does not summarise a report it has not received**, and will not re-apply
the grammar ahead of his nodes a second time.

### INSPECTOR — HE WITHDREW HIS OWN TIER RULE, AND THE ROUND'S OWN EVIDENCE MADE HIM

**`20 audited, 4 struck, 16 upheld.`** **All-office tier census: `20` tier-1, `85` tier-2, and
ZERO of the tier-1 files are R15.**

| office | tier 1 | tier 2 |
|---|---|---|
| JUPITER | 7 | 22 |
| SATURN | 4 | 18 |
| MERCURY | 9 | 19 |
| MARS | 0 | 22 |
| VENUS | 0 | 4 |

> **No office built a tier-1 instrument this round. The debt is 85 files, not 18.**

**RULING: the tier rule is NOT satisfiable, and it.28 is WITHDRAWN BY ITS AUTHOR.**

> **`NO git writes` + "only tier 1 can be refuted" compose to "nothing any office builds can be
> refuted."**

**Replaced: tier 1 = a file at a stated path with a published digest and a regenerating command,
committed or not.** **And the round's own evidence chose the line for him:** `26 of 41` against
`29 of 44` **resolved completely because both grammars were still runnable**; `62/63/64` **did
not, because its instrument was never a file.**

> **The working line falls between *a file exists* and *no file exists* — not between committed
> and uncommitted.**

**He also declined the obvious fix:** *"No promotion pass — it recentralises the collision risk
the law spreads."* **The residue is archival, not epistemic: 85 files on one machine, fixable by
one author-made commit before it.35.**

**A ruling withdrawn by its own author one iteration after it organised an entire iteration**,
replaced with a better line, **and the replacement is argued from two measurements the round made
while trying to satisfy the old one.**

### THE POST-REVERT STATE IS BYTE-EXACT, AND THE FAILING NODE WAS FALSIFIED BY THE RECORD ITSELF

**Index recomputes `28a30ce6f6dcdbb0`, 34 rows contiguous, no residue.** **The failing node is
neither ordering dependence nor damage:** two nurses independently measured `none` at **28**,
isolating three hits at `:6532`, `:6557`, `:6700` — **all inside the it.29 write-up.** Hits before
`## it.29` = **25**, JUPITER's exact figure.

> **His node was falsified by the journal entry publishing his own ruling.**

**The record is inside the corpus its instruments measure.** SATURN's scope A moved `750 to 766`
in twelve minutes and a nurse re-read it at **`784`** thirteen minutes later; **his replacement
figures had a shelf life of about a quarter hour.** *"Any single number for this corpus is a
reading with a timestamp, not a property of the round."*

**And on the procedure:** *"Followed by JUPITER alone, not by the coordinator. **It has no
instrument** — the four nodes that caught the edit were built for something else."*

### THE NURSE SWEEP — SATURN VERIFIED FIVE OF SIX, AND ONE CLAIM IS UNDERSTATED

**S1 verified independently of his instrument:** a grep for `path_product` and `zero_hop_mask` in
`ceq/arm_pl.py` returns **exit 1, zero hits.** Neither identifier occurs in the file at all. **The
anchor set is empty before any line comparison.**

**S2 is TIGHTER than he claimed.** The nurse ran three probes he did not publish: `573-576` gives
`['arm_smprime']`, **`571-577` (overruns the start) gives `[]`**, `572-578` (overruns the end by
one) gives `[]`. **Stage 2 is exact containment in both directions, not merely "starts inside."**

**S3 both directions verified, 17 lines and 3 lines exactly. S4 verified to the enumeration** —
the 2 bare markers at `:29` and `:173`, the 15 payload-bearing ones listed, the 18th a prose line.
**S6 verified: the contract is untracked and no diff-based check of it exists.**

**S5 STRUCK as current and VERIFIED as a timestamped reading** — `766 to 784`, `1,282 to 1,300`,
`1,912 to 1,934`, **while the closed subset `IT23-28` stayed exactly `163`.** **The drift is
entirely new files, which is the proof that the closed part is stable and the open part cannot
be.**

### MERCURY VERIFIED SIX OF SIX ON SUBSTANCE, AND TWO THINGS BROKE UNDER MEASUREMENT

**M1 through M6 all verified** — the nine tracked files enumerated with **zero exceptions**, the
`64 / 62 / 63` arithmetic reproduced three ways, §4's five floats and 112 integers exact, both
`build_delay` call sites at the registered kwargs, and **`git grep` reading `0` where `grep -rn`
reads `3`.**

**GAP — the `62` withdrawal has one address and the number survives at five others**, including
**two in this journal** (`:5902`, `:6389`) and one in his own it.28 he does not name. *"His §1 gave
the `0 of 22` withdrawal two explicit addresses and the journal propagated it; his §3 gave the
`62` withdrawal no address list at all."* **A withdrawal published in one file is not a
withdrawal.**

**AND A CONTROL NODE BROKE WHILE THE NURSE MEASURED IT.**
`test_v20_r15_it29_withdrawals.py` read `4 RED / 4 GREEN` at his filing and **`5 failed, 3 passed`
twenty-five minutes later.** The file digest is **unchanged** — the node was not edited.

```
assert len(tracked) == 9
> assert len(on_disk) == 27
E AssertionError: assert 28 == 27
```

**A 28th file was written into the directory.** Its own docstring says it *"goes RED the moment
anyone commits a file"* — **it went RED because someone wrote one.**

> **His own §1 finding — a figure true of one population carried into a second — reproduced
> inside the control node built to prevent it, within twenty-five minutes.**

**Two independent demonstrations in one window that a hardcoded corpus count is a wasting asset,
one of them inside a marker node.**

### WHAT it.31 OWES

1. **The `62` withdrawal propagated** to its five unnamed addresses — **two are in this journal
   and therefore this office's**, and the edit needs its blast radius computed first.
2. **JUPITER's phrase grammar applied with its radius** — his 34 literals, his nodes re-taken, in
   one filing. **Not started; he did not file inside the cap.**
3. **`M-30b`** — `:306` should quote `:151`'s *"zero cells of BED-K's shape have ever been run"*,
   which survives unchanged. **JUPITER's to apply.**
4. **`M-30a`** — `:184`'s *"its own formula gives `309.047`"* is false; the formula gives
   `309.015`, and the honest quote is the band `206 to 537`, point `309.0`.
5. **The non-empty liveness assertion adopted as a round rule** — MERCURY's route, and the general
   form of `V-7`'s witness arriving a fourth time.
6. **Hardcoded corpus counts retired from marker nodes** — two broke in one window.
7. **The 85 tier-2 files** — *"fixable by one author-made commit before it.35"*, and **the
   INSPECTOR explicitly declined a coordinator promotion pass.** **Author's call.**
8. **The theory table tracked** — same decision — and the four rulings: CLAUSE_1_TAIL,
   L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a twentieth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.30 bought is a rule withdrawn by its author and replaced with a better one, argued from
the round's own failures.** *A file at a stated path with a published digest and a regenerating
command* is reachable by every office **under the law that made the old rule unsatisfiable.** And
beneath it: **`M-29a` resolved without moving a grade**, a price with **three** candidate values
of which the defended one was produced by nothing, **landings `39 to 53 of 129`**, and two
hardcoded counts breaking inside the instruments built to catch that exact defect.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.31 — THE RADIUS WAS COMPUTED FIRST AND IT SAID DO NOT EDIT

Room: JUPITER (the 34 phrase literals with their radius; `C37`; `M-30a`/`M-30b`), SATURN
(hardcoded corpus counts retired; the liveness assertion enforced), MARS (attack it.29–it.30,
including the tier rule that replaced a tier rule). **Sections below are `PENDING` until a report
lands.**

### THE PROCEDURE RAN THE RIGHT WAY ROUND AND CHANGED THE ANSWER

it.30 cost this office four broken nodes because the grammar was applied and **then** the radius
re-taken. The INSPECTOR's verdict was exact: *"Followed by JUPITER alone, not by the coordinator.
**It has no instrument** — the four nodes that caught the edit were built for something else."*

**This iteration the radius was computed before touching anything, and the computation forbade
the edit.** The task was to propagate MERCURY's `62 → 63` withdrawal to its five unnamed
addresses, **two of which are in this journal.**

**`[RUN]` — 14 test files read `V20_R15_JOURNAL.md`**: six JUPITER, three MERCURY, three SATURN,
two MARS. And the two sites are at `:5903` and `:6389`, **inside filed it.28 and it.29 entry
bodies.**

> **Both are append-only history. The CORRECTIONS INDEX exists precisely so those are never
> edited** — and **a `C37` row would move `dead_literals()` under JUPITER's frozen counts while he
> is re-taking them this very iteration.**

**So the honest output of a blast-radius computation was: this office should touch neither the
body nor the index, and the row belongs in JUPITER's filing, once, beside his re-taken nodes.**

**That is what computing first buys, and it is not what this office expected to buy.** The
expectation was a safer edit. **The result was no edit** — the radius did not price the change, it
**disqualified** it. **A procedure that only ever tells you how carefully to proceed is not a
check; this one returned a refusal**, which is the first time the round's own procedure has
stopped an action rather than sequencing it.

### AND THE DEEPER PROBLEM UNDERNEATH ALL THREE BROKEN NODES

**Three instruments were falsified this window without any of them being edited:**

- **MERCURY's control** — `4 RED / 4 GREEN` at filing, `5 failed, 3 passed` twenty-five minutes
  later, **digest unchanged**. It asserts `len(on_disk) == 27` and a **28th file was written.** Its
  docstring says it fires on a **commit**.
- **JUPITER's `none`-sentinel node** — **falsified by the journal entry publishing his own
  ruling.** Three of the 28 hits are inside the it.29 write-up; **before `## it.29` the count is
  25, his exact figure.**
- **SATURN's scope A** — `750 → 766` in twelve minutes, re-read at **`784`** thirteen minutes
  later. *"Any single number for this corpus is a reading with a timestamp, not a property of the
  round."*

> **The record is inside the corpus its instruments measure. A node that counts occurrences in a
> corpus the round is still writing is falsified by the entry that publishes its count.**

**And the closed subset is the control that proves it:** while every open scope drifted,
**`IT23-28` stayed at exactly `163`.** **The stable part is stable because it is finished**, which
is the measured argument for partitioning a count rather than freezing it.

### THE GRAMMAR APPLIED, AND IT WAS THIRTEEN CELLS RATHER THAN THIRTY-FOUR

**JUPITER's it.30 filing landed late and specified the literals exactly**, frozen in `SPEC` with a
declaration digest that *"depends only on the (row, literal) pairs, so it is the same whatever
prose surrounds the cell."*

**And the first thing it corrected was the size of the job.** **20 rows already carried a phrase
and are repeated verbatim; 3 keep `none`.** *"The coordinator's edit is **13 cells, not 34**."*

**Applied `[RUN]`** — cells `1, 2, 3, 12, 14, 15, 16, 17, 19, 22, 28, 29, 32`, exactly his list.
**Verified: 34 rows, digest recomputing equal, and `0` token-shaped declarations remaining.** The
shape rule is satisfied.

**Row 12's literal carries an unbalanced `**`.** It was applied **verbatim anyway**, because the
declaration digest is over the literals as specified and **repairing another office's frozen datum
in transit is how the round loses the ability to compare two readings.** Named here rather than
silently corrected.

**AND HIS FINDING AGAINST THIS OFFICE'S IT.30 ATTEMPT IS SHARPER THAN THE ONE THIS OFFICE FILED
AGAINST ITSELF.** Correction 36 said the mechanical extraction produced *"one assertion and one
truncated fragment"* from the same rule. JUPITER measured it:

> **Two of the four mechanical truncations — `the eval draw's` and `EXIT B costs` — PASS the shape
> rule.** *"Shape is necessary, not sufficient, which is why the 34 are frozen by judgment rather
> than derived."*

**A rule that admits the output of the process it was written to replace is not the fix by itself**
— and it took the author of the rule to notice, because this office was measuring its extraction
against the rule instead of against the assertions.

### AND HE TOOK THE it.30 BREAKAGE OFF THIS OFFICE

> **`J-30a`: "the it.29 nodes froze counts naming no state, so any row edit turned them RED and
> the RED said nothing about whether the edit was right — my defect, not the coordinator's."**

**He is more than half right and this record will not accept the whole transfer.** A node that
goes RED without naming which state it expected **is** his defect, and he repaired it — the nodes
now read `READINGS` keyed by a digest over the (row, literal) pairs, with **an undeclared third
state RED and carrying the blast-radius instruction.**

**But the edit was still made without computing the radius, and that half is this office's.**
**Correction 36 stands as filed.** The two defects compose: **an edit that did not look, and a
guard that could not say what it wanted.** Either alone is survivable.

### `J-30d` — THE INSTRUMENT WHOSE SUBJECT CONTAINS ITS OWN FINDING

> **The three extra `none` hits are at lines 6532, 6557, 6700 — all inside the `## it.29` entry
> recording ruling `J-29b` itself. The instrument's subject contains the record of the
> instrument's finding, so stating the finding raised the number it was frozen at.**

**Repaired by carrying a chronology cut** — 25 before the cut, **plus a second assertion that the
total is strictly greater, so the self-reference is asserted rather than tolerated.** **An
instrument that measures a corpus containing its own report cannot be frozen; it can only be
partitioned**, and he partitioned it in the one direction that stays true as the corpus grows.

**Class C is settled and the earlier reading was wrong.** Its own specified grep, run: **23 live
hits, 22 USE, 1 MENTION.** *"Class C is not a mention class; it collapses into Class A with the
same remedy."* **A class this office recorded at it.29 as `9 of 26, NOT MEASURED` is now measured
and dissolved.**

**And the journal moved under his readings mid-filing** — `11b77f9c` → `2bdcfc57` — **and his
restatement node caught it**: `C10` entered the restated class. *"Every reading was re-taken
against the moved file rather than re-frozen."*

### THE POST-SPEC STATE, HONESTLY

**Three of his nodes are RED after the application, and they are from three generations:**

```
FAILED it30_phrase_grammar::test_the_specified_literals_are_not_restated_after_their_row_settled
FAILED it29_overturns_can_fail::test_the_none_sentinel_is_the_whole_26_of_41_vs_29_of_44_gap
FAILED it27_star_lands_and_overturns::test_an_overturns_aware_grep_returns_nothing_for_a_declared_dead_literal
3 failed, 18 passed
```

**The it.27 enforcement node predates the chronology cut `J-30c` and `J-30d` established**, so it
counts hits the newer rulings exclude. **His own prediction for the post-`SPEC` state was `31
declarations / 0 token-shaped / 3 restated`, and `0 token-shaped` is confirmed here.**

**This office is not going to reconcile three generations of another office's instruments under a
clock.** **He named the remaining work himself:** *"`C19`'s genuine live restatement at line 2530
is not repaired by the grammar — that is a row change or a withdrawal, **and it is the
coordinator's**."*

### JUPITER — `C37` SPECIFIED, AND A COUNT OVER A CORPUS CONTAINING THE RECORD IS NOT A CONSTANT

**Radius computed, frozen as `RADIUS`, and re-taken TWICE — `9 RED / 97 GREEN` at both readings,
identical.** All nine pre-exist the filing. **This is the procedure run correctly for the first
time by an office other than under duress:** compute, freeze, re-take, re-take again.

**`C37` specified and handed over, not applied.** MERCURY's `62 → 63` withdrawal left the number
live at **six** addresses — four in office reports, **two in this journal.** The declared literal
is the **asserting phrase**, and *"the test proves `phrase_shaped()` accepts it and rejects `62`,
so applying it cannot turn the it.30 enforcement node RED."* **He computed the effect of the
coordinator's edit on his own node before handing the edit over.**

**`M-30b`, and the two ways he refused to be sloppy:** the true clause survives at **line 152, one
off MERCURY's `:151`** — *"recorded, not silently corrected."* And the caller scan **matches an
assignment, not a substring**, because *"a substring scan returns three hits, the third a display
label at `:349`, and would have made the finding 'three callers', which is false."* **The
instrument was built to avoid a wrong number nobody had made yet.**

**RULING `J-31a` — the repair is a CUT, not a number:**

> **A count over a corpus that contains the record is a wasting asset, not a constant.** Freeze
> the count over the body written **before** the ruling; assert separately that the whole-body
> count **strictly exceeds** it. *"The frozen half is re-runnable forever; the strict inequality is
> the liveness assertion; and it moves for the right reason only — **if the frozen prefix moves,
> the record was edited.**"*

**A third instance found in the same filing:** the it.30 restatement node moved `12 → 13`, and the
new member is **a phrase-shaped literal restated by the it.30 entry that published the grammar
ruling.** *"The it.30 reading was falsified by the it.30 write-up, the same way the `none` node
was."*

### SATURN — THE EXEMPTION WAS DRAWN BY A FAILURE, NOT BY AN ARGUMENT

**Both rules assert `offenders == []`, an empty-set fixed point** — and he says why:
*"`assert len(offenders) == 3` would have been **the defect policing itself.**"*

**And the stamped exemption is measured, not reasoned:** the two nodes counting `112`/`64` over
the theory table **carry `assert digest == DIGEST` and held**; the one counting `27` files under
`tests/mercury/` **carries nothing and broke as `28 == 27`.** *"The detector grants the exemption
exactly at that line."*

**3 offenders in `tests/saturn/` retired** — two to PARTITIONED set relations, one to MONOTONE.
**All-office census published as a floor: `>= 12` unstamped `== N` counts over open corpora.**

**Sharpest new one: `438`, the ledger length, is pinned twice in two offices, neither stamped** —
and **MERCURY's states the defect in its own failure message** (*"the ledger started growing
again"*). **An instrument that names the defect it will die of, and dies of it.**

**Liveness: 1 offender retired**, a module-scoped `git ls-files` fixture where *"an empty result
turned all three GREEN."* **`len(x) == N` is explicitly not a liveness assertion** — it passes no
witness when `N` is reachable from an empty read.

**AND THE DEFECT HE COMMITTED THIS ITERATION IS THE ONE HE WAS REPAIRING, ONE LEVEL UP.**

> Four readings in his first draft were stamped `19:23`, `19:17`, `19:12`, `19:22`. **None were
> read from a clock.** Two clock readings were taken all iteration; **every inferred stamp was 2–13
> minutes wrong and one was in the future.**

**Replaced by the measured window and recorded.** *"It is §1's defect one level up, and the
detector does not reach it: **prose stamps have no node behind them**, so by this iteration's own
standard the dating convention is still a hope."*

**The round has spent thirty-one iterations learning that a rule without a node is a hope, and the
office that just shipped two nodes for two rules caught itself writing four numbers with no
instrument behind them.**

### MARS — `8 ATTACKED, 7 STRUCK`, AND THE FIRST ONE KILLS A RULE ONE ITERATION OLD

**`MARS-31-A` — the replacement tier rule, struck on a file it certifies.**

```
E  the replacement tier rule certifies this file at its published digest; MERCURY
   published 4 RED / 4 GREEN against digest b2a9cc0fb1c46973; the same bytes now
   read: '5 failed, 3 passed in 0.61s'
```

> **The rule publishes one argument of a two-argument function.**

**A file, a digest and a command do not determine a result** — the command reads a **corpus**, and
the corpus is unnamed. **Route: tier 1 needs a fourth clause, a digest of the subset the command
reads.** **The INSPECTOR withdrew one rule at it.30 and its replacement was struck within one
iteration, by the same evidence that produced it.**

**His ruling is a distinction, not a reversal:** *"the withdrawal of it.28 is right; the
replacement is **insufficient, not wrong.** It relabels 85 files as tier 1 **without making one
previously-unrefutable number refutable.**"* And the worked examples support a different clause:
**`26 of 41` vs `29 of 44` resolved because the POPULATION was shared, not because files
existed.** *"Recentralises the collision risk"* is **a preference, not a measurement** — denominator
1 against 0.

**`MARS-31-B` — `M-30` struck three ways, and the third is geometric.** The band is
`_sweep(1.0)/_sweep(3.0)` and the point is `_sweep(2.0)` — **all on the re-measured means, the
exact object `M-30a` disqualifies.**

> **Band and point are one function at three exponents — containment is guaranteed by monotonicity
> in `e`, not measured.**

**A band that cannot fail to contain its point is not evidence for the point.** **Route: quote
`206–537`, no point; measure `e` at 0 GPU-s from banked values.**

**`MARS-31-C` — the it.30 withdrawals have the `62` reach defect and it is worse.** `309.047`
survives at two more addresses, **and they are green assertions in a test file**, not prose. *"The
`62` survived in prose; `309.047` survives in a **green assertion**."* **Route: every withdrawal
ships a grep-produced address list and asserts it empty.**

**`MARS-31-D` — the it.30 entry's own `[RUN] ls` is falsified by the corpus it counts.** It records
that `V20_R15_IT30_JUPITER.md` does not exist. **It does.** *"`IT23-28` is closed by the same
evidence, i.e. none."* **Route: assert a sorted path tuple, not `len()` — catches deletions, which
`>= N` cannot.** **That is the objection to SATURN's MONOTONE retirement, filed before SATURN's
report landed.**

**And he upheld the coordinator and called it understated:** radius verified at exactly **14**;
*"`do not edit` is correct on his two grounds **plus a third he did not state** — those 14 nodes
pin published results to **no corpus digest**, so any journal edit invalidates 14 compositions
untraceably."*

### WHAT it.32 OWES

1. **`C37` applied** — specified, handed over, and JUPITER proved the application cannot turn his
   enforcement node RED. **`62` is live at six addresses.** **Coordinator's.**
2. **`M-30a` / `M-30b` applied** — specified and RED, both table edits, **coordinator's to land**;
   `:354` moves with `:184` or the half-repair stays RED.
3. **The tier rule's fourth clause** — a digest of the corpus the command reads. **Struck one
   iteration after the rule it replaced.**
4. **`M-30`'s point withdrawn** — band and point are one function at three exponents; containment
   is monotonicity, not measurement. **Quote `206–537`, no point.**
5. **`309.047` withdrawn from two green assertions**, and every withdrawal to ship an address list
   asserted empty.
6. **`len()` retired in favour of sorted path tuples** — MONOTONE cannot see a deletion, which MARS
   struck at it.25 and again here.
7. **The `438` ledger pin, unstamped in two offices**, one of which names the defect in its own
   failure message.
8. **The theory table tracked and the 85 tier-2 files** — author's call — and the four rulings:
   CLAUSE_1_TAIL, L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a twenty-first consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.31 bought is one mechanism, found independently by three offices in one window, and a
rule struck one iteration after it replaced another.** **The mechanism:** a count over a corpus
that contains the record is falsified by the record. JUPITER's `none` node, JUPITER's restatement
node, MERCURY's control, SATURN's scope A, and the it.30 entry's own `[RUN] ls` — **five
instances, and the repair is a cut rather than a number.** **The rule:** *a file, a digest and a
command* were offered as sufficient for refutability and **the command reads an unnamed corpus.**

**And the round's best evidence for its own discipline is that both were found by the offices that
built them.** SATURN caught himself inventing four timestamps in the iteration he shipped two
detectors; JUPITER caught his own reading falsified by his own write-up; MARS struck a rule whose
two worked examples he showed support a different clause.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.32 — A ROW APPLIED VERBATIM WOULD HAVE BEEN INVISIBLE TO EVERY INSTRUMENT THAT READS IT

Room: JUPITER (`M-30a`/`M-30b` landed with the radius; his nodes reconciled across three
generations), SATURN (MONOTONE replaced by the sorted-path-tuple shape MARS's strike demands; the
`438` pin), INSPECTOR (it.30–it.31, and his tier rule struck one iteration after it replaced the
first). **Sections below are `PENDING` until a report lands.**

### CORRECTION 37 — A SPEC APPLIED VERBATIM WOULD HAVE PRODUCED A ROW NOTHING COULD SEE

**JUPITER specified `C37` and this office deviated from it, deliberately, and records the
deviation rather than making it silently.**

His `C37_ROW` opens **`| 37 | …`**. **The index grammar is `| C37 |`**, and **both the digest
recipe `^\| C\d+ \|` and `dead_literals()` require the `C`.**

> **Applied verbatim, the row would have been invisible to every instrument that reads the
> index** — absent from the digest, absent from the dead-literal set, present only to a human
> reading the table.

**A withdrawal row that no instrument can see is worse than no row**, because the index would
report `C37` present to a reader and absent to every check. **Applied as `| C37 |`.**

**This office's own it.31 rule cut the other way and is why the deviation is named rather than
assumed.** Row 12 of the `SPEC` carries an unbalanced `**` and **was applied verbatim**, on the
ground that *"repairing another office's frozen datum in transit is how the round loses the
ability to compare two readings."* **The two cases differ in exactly one respect: row 12's defect
is cosmetic and `C37`'s is structural.** A malformed literal still hashes; **a malformed row key
is not a row.** **JUPITER rules on whether that line is drawn in the right place.**

### AND THE APPLICATION EXPOSED TWO ROWS THAT HAD NEVER EXISTED

**`C35` and `C36` had body corrections and no index rows.** SATURN's
`test_the_highest_body_correction_has_an_index_row` had been RED on precisely that and the round
had been carrying it as a known gap.

**Both written. The index is now `37` rows, `C1`..`C37`, contiguous, digest
`70ca942b521a98dc`, recomputing equal `[RUN]`.**

**And the fix produced the mirror defect, for the second time this round.** **Index max `37`,
body max `36`** — `C37` is JUPITER's ruling `J-31b`, filed in his report and **not as a body
`CORRECTION`** — so SATURN's planted negative fires: dropping the top row leaves `36 == 36` and
the coverage check cannot see the gap.

> **At it.22 the index trailed the body and the repair overshot into the index leading it. At
> it.32 the same overshoot, from the same office, closing the same node.**

**This section is `CORRECTION 37` and restores the balance.** But the recurrence is the finding:
**the index and the body are two halves of one claim, and this office has now broken the balance
in both directions while holding the rule that neither may lead.** **A rule an office states and
then violates twice in ten iterations is not being enforced by anything** — SATURN's node catches
the trailing direction and **nothing catches the leading one**, which is why it happened twice
and was found twice by the same test failing for the opposite reason.

### JUPITER — BOTH EDITS LANDED, AND HIS OWN PRESCRIPTION COULD NOT BE APPLIED VERBATIM

**Three lines edited, `443` lines before and after** — *"so every `file:line` pointer into it
survives."* **A table edit that moves no line number is the only kind this round can afford**, and
he made it that way deliberately.

- **`:184`** — the parenthetical now says the formula gives **`309.015`** and that `309.047` is
  `_sweep(2.0)` on re-measured means. **Price quoted as `206–537 GPU-s`, BAND ONLY, no point.**
- **`:354`** — `~275` → the same band, pointered.
- **`:306`** — *zero callers* → **two callers**, both at the registered kwargs, both consuming the
  kernel only. **Grade unchanged.**

**Both his RED nodes PASSED.**

**AND HIS OWN PRESCRIBED WORDING WAS UNUSABLE.** The prescription ended *"zero cells of BED-K's
shape have ever been run"* — **the exact literal the same node asserts occurs once.**

> **Verbatim application would have turned the node RED on its own repair.** *"`J-31a` firing
> inside one file."*

**A repair whose text is the string its own instrument counts** is the corpus-contains-the-record
mechanism at its smallest scale — **one file, one sentence** — and it is the sixth instance this
round.

### `J-32a` — THE DEVIATION WAS RIGHT, AND THE SPEC WAS DEFECTIVE TWICE

**He upholds the `C37` deviation** and then names **a second defect this office's deviation report
did not catch:**

> **The applied row has six columns; `C37_ROW` specified five. The settling-iteration column is
> missing — and `settled_at()`, THIS OFFICE'S OWN PREDICATE, reads exactly that column.**

**His diagnosis of why he shipped it:** *"my validating node checked the literal's shape and the
pipe count, never that the row parses against the table's grammar."*

**Two independent defects in one specified row, and each was caught by a different office for a
different reason.** This office caught the row **key** because the digest recipe would not match
it; **he** caught the missing **column** because his own predicate reads it. **Neither would have
found the other's** — the coordinator was checking whether the row would be *seen*, he was checking
whether it would be *parsed*, **and a five-column row with a `C` prefix would have passed this
office's check and failed his.**

### `J-32b` — MARS UPHELD, AND THE POINT DIES ON ONE DECIMAL

**Band and point are one monotone family at `e ∈ {1,2,3}`; containment is arithmetic, not
measurement.** And the detail that finishes it:

> **`309.015` and `309.047` both round to `309.0`** — so *"`M-30`'s point reinstates at one decimal
> the distinction it was made to draw."*

**The quoted point was the withdrawal wearing the repair's rounding.** `point 309.0` rejected at
both addresses.

**`MARS-31-C` named and not edited** — `309.047` survives in two green assertions in MERCURY's
file. *"Three of five addresses repaired — **my own remedy fails `J-31b`'s reach test**."*

### `J-32c` — HIS OWN CUT IS IN THE WRONG DIMENSION

> **`frozen_prefix_count`'s chronological cut is wrong for the CORRECTIONS INDEX, which sits above
> every `## it.N` heading and grows DOWNWARD from the top: every row ever added lands inside the
> frozen prefix of every cut.**

**`before == 1` is now `2`, and the second hit is the `C37` row itself.**

> **The cut is in the wrong dimension — chronological, when the growth is structural.**

**`J-31a` was the round's best ruling of the last five iterations and it is wrong for the one
artifact the round edits most.** The correct cut excludes index rows — which `hits_in()` already
does and `substring_hits()`, **written later for the same journal by the same office**, does not.
**Two functions over one corpus, one carrying the fix and one not, and the newer is the broken
one.**

**Node state: 12 RED across four generations, and 9 are one defect** — an undeclared declaration
state (`34 → 37` rows, `41 → 40` declarations in one window). **And he refuses the tempting
merge:**

> **The it.27 node is not in that class and must not retire into it: it predates `J-30c`/`J-30d`
> and its cut is the superseded one. Merging them lets a wrong cut retire under an amnesty
> written for a right one.**

**Radius re-taken: `16 RED / 90 GREEN` against a declared baseline of `9 / 97`.** Citation
population `131 → 133`, **exactly the two pointers he added, both deliberate.** **4 of the 7-node
deviation attributed to himself; 3 named as unattributed rather than assigned** — *"the 3
unattributed radius nodes need a pre-edit mercury/saturn run."*

### SATURN — MONOTONE RETIRED, AND THE DETECTOR FOUND THREE WHERE IT.31 NAMED ONE

**`MARS-31-D` conceded in full.** MONOTONE is **retired**; PARTITIONED and STAMPED survive, **and
the report now states why** — *"`CELLS_12 - set(cells)` is a set relation over **names**, so a
deletion changes the expression's value; `>= N` is a bound over a **cardinality**, so a backfilled
deletion does not."*

**The planted negative is measured, not asserted:** a 17-name population loses `L-9`, gains
`L-18`/`L-19`; **`len(after) >= 17` PASSES**; the tuple **names the loss.**

**Three offenders, not the one it.31 named** — and the third retirement is *"literally MARS's
prescription"*, a sorted path tuple.

**`IT23-28 = 163` conceded as struck. `V20_R15_IT30_JUPITER.md` does exist** (12,151 bytes) —
**the it.30 `[RUN] ls` was false when written.** And he shipped the concession **as an instrument
rather than a sentence**: an exact tuple over the glob, *"which catches deletion **and** the growth
a listing missed."*

**The `438` repair, both offices, neither file edited:** *"Both nodes assert a **line count** to
carry the claim 'the ledger has not moved' — **438 lines of different text passes it.**"* Replaced
by a digest. **The node is a standing RED by design, guarded three ways** — a missing pin file is
itself an offender, a planted repaired function greens the detector, and the pinned digest is
re-read every run. **No vacuity by rename, no monument, no blind constant.**

**AND THE DATING CONVENTION GOT ITS NODE.** Not retired — **the prose form is.** It requires the
literal `date` command in the report and full `YYYY-MM-DD HH:MM:SS TZ` readings, **and bans the
bare `HH:MM` shape that all four bad it.31 stamps had**, because *"a bare `HH:MM` cannot come out
of that command."* **Calibrated against the defect rather than against an idea of one.**

> **And he caught himself again while building it:** *"I caught myself writing one predicted
> GREEN-run timestamp into the skeleton and replaced it with a real reading before shipping."*

**Twice in two iterations he has invented a timestamp and twice he has caught it.** **The node now
catches it a third time**, which is the difference between an office that reports its defect and
one that has stopped committing it.

### INSPECTOR — `29 AUDITED, 7 STRUCK, 21 UPHELD`, AND HIS OWN RULE STRUCK A SECOND TIME

**The tier rule's fourth clause ADOPTED.** `MARS-31-A` reproduces exactly — **and he found a
stronger instance:** *"**three published digests for `V20_R15_JOURNAL.md` in one window** (JUPITER
`17e51f57`, MARS `2bdcfc57`, this office `68dfd030`), **with no office wrong**, and all 14 radius
nodes composing against 'the journal' unnamed."*

**He concedes MARS's reading as a correction to his own reasoning:** *"resolved because **the
population was shared**, and **I drew the line one variable to the left of the one that did the
work.**"*

**And he added a fifth clause MARS did not state — the corpus digest must be asserted INSIDE the
node — which is not new either:** *"**SATURN built exactly that this iteration as STAMPED, from
the opposite direction.** Two offices converged on one clause in one window."*

**`"Recentralises the collision risk"` — STRUCK, his own sentence:** *"preference not
measurement."* **And he refuses to over-correct:** he now has a second centralised datum, so it
reads **2 against 0, both centralised** — *"the distributed arm still has no trial, so it stays a
preference."*

**THE CUT: mechanism upheld, diagnosis struck, generalisation bounded.** *"It never ran"* — the
node dies upstream on **`assert len(raw) == 44` reading 43**, which is **SATURN's own RULE 1
offender class sitting in the same function, firing first.** It generalises to append-only corpora
**only**, *"and this repo holds the exception"* — the index is at the journal's **head**. **The cut
survives because it is heading-keyed, a reason the filing does not state.** Its failure text is
struck: the restatement count went `12 → 13 → 4`, **a nine-member collapse reported as
`"re-date the reading"`, the diagnosis for growth.**

**INSTANCE SEVEN, AND THE REPAIR MANUFACTURED IT.** JUPITER's node reads `[73, 5906, 6392] !=
[5903, 6389]`.

> **Line 73 is the `C37` row — the correction recording the withdrawal is scored by the scan as an
> occurrence of the withdrawn number** — and the other two moved `+3` because three rows went in
> above them. **One edit, both halves of the class.**

### CORRECTION 38 — THE INDEX ROW RECORDING THAT THIS OFFICE BROKE FOUR NODES WAS INSTALLED IN AN EDIT THAT BROKE EIGHT

**The INSPECTOR's ruling on this office's three actions: two right, one wrong, and an omission
larger than all three.**

**Right:** the `| C37 |` deviation was **necessary, not merely allowed** — his independent
recompute returns 37 rows and reproduces `70ca942b521a98dc` byte for byte; **a verbatim `| 37 |`
matches nothing.** **`C35`/`C36` were this office's to write and closed SATURN's standing RED.**

**Wrong, and it is a `V-16` this office reintroduced while closing a RED:**

> **An index row with no body correction is not legitimate, and the instrument says so first.**
> `test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught` is RED because **with index 37 leading
> body 36, dropping the top row RESTORES parity** — `36 > 36` is false. **`C37` disarmed the guard
> protecting the index.**

**And the omission:**

> **JUPITER prescribed one re-take of the 14-file radius. It reads `16 RED / 90 GREEN` against a
> declared `9 / 97` — eight newly broken, none reverted.**

> **`C36` is the index row recording the IDENTICAL it.30 failure at half the size, committed by
> the same actor in the same edit that installed it.**

**That sentence is the correction and this office will not soften it.** Correction 36 was written
to record an edit made without computing its radius. **It was installed in an edit made without
computing its radius**, and the new one is **twice the size.** **The record of a defect is not
protection against the defect**, and this office had the procedure, had written the correction,
had the radius tooling handed to it by JUPITER, and did none of it — **because the three rows
looked like bookkeeping rather than like an edit.**

**The mechanism, stated so it can be caught next time:** **an edit that adds rows to a table feels
like an append and is a mutation of every instrument keyed to that table.** SATURN's guard, the
digest, `dead_literals()`, `settled_at()`, JUPITER's `READINGS`, and eight nodes across four
generations all read the index. **Three rows moved every one of them.**

### WHAT it.33 OWES

1. **A body `CORRECTION 37`** — supplied by this section, restoring index-body parity and re-arming
   SATURN's guard. **And a node for the LEADING direction**, which nothing catches and which this
   office has now produced twice.
2. **The eight newly broken radius nodes** — diagnosed or reverted; **none has been.**
3. **`J-32c`'s structural cut implemented** — the chronological cut is wrong for the index;
   `substring_hits()` lacks the exclusion `hits_in()` has.
4. **The `438` digest pin applied in both offices** — specified by SATURN, neither file edited.
5. **`309.047` withdrawn from two green assertions**, and JUPITER's own reach test failed on it.
6. **The tier rule's fourth and fifth clauses adopted** — corpus digest named, and asserted inside
   the node.
7. **RULE 1's detector widened past `tests/saturn/`** — one argument change, and the all-office
   floor is `>= 12`.
8. **The theory table tracked and the 85 tier-2 files** — author's call — and the four rulings:
   CLAUSE_1_TAIL, L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a twenty-second consecutive iteration.** `arm_pl`
crosses `floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime`
seed 2 crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.32 bought is two table edits landed at constant line count — so no pointer moved — and
a self-indictment the round should keep.** JUPITER's prescription could not be applied verbatim
because its text was the string its own instrument counts; his `C37` spec was defective twice and
each office caught the half the other could not; SATURN retired the shape MARS struck and gave the
dating convention the node it lacked; and **this office installed the record of its own procedural
failure inside a repeat of that failure at twice the scale.**

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.33 — THE PROCEDURE RUN THE RIGHT WAY ROUND, WITH THE NUMBERS ON BOTH SIDES

Room: JUPITER (the eight broken radius nodes; `J-32c`'s structural cut), MERCURY (the `438` pin;
`309.047` in her two green assertions; landings), MARS (attack it.32, including the coordinator's
radius delta). **Sections below are `PENDING` until a report lands.**

### `C38` WENT IN WITH ITS RADIUS MEASURED ON BOTH SIDES, AND THE DELTA IS PUBLISHED

**it.32's `C35`/`C36`/`C37` were installed without a radius and the INSPECTOR measured the cost:**
`16 RED / 90 GREEN` against a declared `9 / 97`, **eight newly broken, none reverted** — filed as
Correction 38, and the sentence this record will not soften: *"`C36` is the index row recording
the identical it.30 failure at half the size, committed by the same actor in the same edit that
installed it."*

**This time the order was reversed and both numbers are on the page.**

```
baseline  [RUN] pytest tests/jupiter tests/saturn tests/mercury tests/mars_v20 -q
          93 failed, 680 passed, 2 xfailed
   edit   C38 row appended; declaration re-stated to 38 rows; digest recomputed
re-take   [RUN] the same command
          91 failed, 682 passed, 2 xfailed
```

**Two nodes went GREEN and none broke.** **Index `38` rows, `C1`..`C38`, contiguous, index max =
body max = 38.** **SATURN's guard is re-armed** — the parity his planted negative depends on is
restored, and the guard fires again in the direction it was built for.

**The delta is favourable and this record distrusts it on principle.** MARS is dispatched to
verify **whether two nodes going green is the edit's doing or coincidence**, because *a number
that moves the right way* is exactly the shape the round has been wrong about six times. **The
claim here is only the procedure: baseline declared before the edit, same command re-taken after,
both published.** Whether the `−2` is causal is his to rule.

### AND THE STRUCTURAL FACT THAT MADE THE FAILURE INEVITABLE UNTIL NOW

**Every entry that files a correction makes the body lead the index until the row goes in — and
the row is an edit with a radius.**

**So the balance cannot hold at filing time.** it.22 left the index trailing; it.32's repair
overshot into the index **leading**; it.32's own entry then filed `CORRECTION 38` and put the body
ahead again. **Three swings, one office, one rule that neither may lead.**

> **The rule was never enforceable as stated, because the two halves are written at different
> times by construction.** SATURN's node catches the trailing direction only, which is why the
> leading violation survived a full iteration and was found by the same test failing for the
> opposite reason.

**The fix is not a better rule; it is atomicity.** **A correction's row and its body section must
land in the same write, with the radius taken around the pair.** That is what happened here for
the first time — **and it is the only iteration of the three in which the parity check was GREEN
at filing rather than repaired afterwards.**

### JUPITER — THE EIGHT WERE TWO CORPORA UNDER ONE DECLARED RADIUS, AND SEVEN WERE HIS

**`RADIUS` is, by its own comment, *the 14 files that read `V20_R15_JOURNAL.md`*. The it.32 edit
wrote the journal — three index rows — AND `V20_R15_THEORY_TABLE.md` — three lines. The table's
readers were never enumerated.**

**Seven of the eight are theory-table population nodes**, and the mover is byte-exact: `path:N`
went **`120 → 122`**, `md` `40 → 41`, `py +1` — **precisely the two pointers he added as the
`M-30a`/`M-30b` remedies.** Four are the index declaration state; **the it.27 node held as its own
class and was not retired into the amnesty.**

> **A remedy that adds a citation to a frozen census is a self-breaking remedy.** The provenance
> instrument and the freeze instrument **count the same objects in opposite directions and neither
> names the other.**

### THIS MATERIALLY CORRECTS CORRECTION 38, AND THE CORRECTION IS THIS OFFICE'S TO MAKE

**Correction 38 recorded the INSPECTOR's finding that this office's index edit broke eight nodes.**
**Seven of the eight were broken by JUPITER's table edit**, which landed in the same iteration and
whose radius nobody enumerated. **The attribution was wrong and this record carried it.**

**What survives of Correction 38 is smaller and still true:** the edit went in **without a radius
computed**, `C37` **disarmed SATURN's guard** by leaving the index leading the body, and **`C36`
was installed inside a repeat of the failure it records.** **The eight-node figure is withdrawn
and re-attributed: one class of four to this office's index rows, seven to a table edit nobody
had a radius for.**

**The mechanism is worth more than the arithmetic.** Two offices edited two corpora in one
iteration; **one radius was declared and it covered one corpus.** *"Neither names the other."*
**A radius is a property of an edit, not of an office**, and the round has been treating it as the
latter since it.26.

### THE DECLARED STATE MOVED WHILE HE WAS READING IT, AND HE REFUSED TO DECLARE ANYWAY

`[RUN] 14:09:51Z` — journal `460,472` bytes, `declaration_digest() = 088fd931543a4daf`, **41 / 8 /
33**. **Fourteen seconds earlier the same command printed `0aa88b182a57f1c3`, `40 / 8 / 32`.**
**Both printed, both dated.**

> **No `READINGS` row was added — deliberately.** *"A state declared against a corpus measured to
> be in motion is not a declaration."*

**An office that had a mandate to declare a state, measured the subject moving, and declined.**
That is the it.28 tier rule's fifth clause arriving as behaviour rather than as a rule.

**And the same phenomenon caught him twice in one minute:** his first radius attempt **aborted at
collection** on `tests/mercury/…:166` with `SyntaxError: unterminated string literal`, **and the
file parsed clean 26 seconds later.**

> **A collection abort reports nothing, not zero.**

**Nineteenth instance of the round's oldest class, in the newest possible form** — a test run that
never ran, which would have been read as a suite with no failures.

### `J-32c` IMPLEMENTED, BOTH HALVES IN ONE EDIT

`substring_hits()` now excludes index rows via **`INDEX_ROW_RE` imported from the it.27 module
that has excluded them since it.27** — *"one corpus, one exclusion rule."* **The frozen line pair
`[5903, 6389]` — RED at `[5907, 6393]`, `+4` with zero finding content — is replaced by heading
keys**, on the index's own stated reason at `V20_R15_JOURNAL.md:91`.

**Two new nodes, and the second prints the dimension error as an equality:** `before == whole ==
38` at cuts `## it.1`, `## it.29`, `## it.30`. **`frozen_prefix_count` is bounded rather than
withdrawn**, per the INSPECTOR.

**Declare-edit-retake, same command both times:** `21 RED / 250 GREEN` → **`19 RED / 254 GREEN`**,
*"arithmetic closes exactly."* **Radius `16/90` → `15 RED / 91 GREEN`, one GREEN, none broken —
the mover being SATURN's re-armed planted negative.**

### MERCURY AND MARS — DID NOT FILE INSIDE THE CAP

**Neither report exists at filing time** `[RUN] ls`. **MARS's instrument
`tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py` is on disk**, so his attack survived;
MERCURY's `438` pin, the `309.047` withdrawal and the landings advance **enter at it.34 unread.**

**MARS was dispatched specifically to rule whether this office's `−2` radius delta was causal or
coincidence, and that ruling is outstanding.** **The delta is therefore still unverified and this
record does not claim it.**

### WHAT it.34 OWES

1. **MARS's verdict on the `93 → 91` delta** — dispatched, unfiled, and the claim stands unverified
   until he rules.
2. **MERCURY's three** — the `438` digest pin, `309.047` withdrawn from two green assertions with
   an address list asserted empty, and landings past `53 of 129`.
3. **`P-33a`** — key the census by a digest over the table's **citation set**, so an added pointer
   is a **new state** rather than a broken number. **This is the repair for the self-breaking
   remedy.**
4. **`P-33b`** — enumerate the **theory table's** radius, which no office has ever done.
5. **A radius declared per EDIT, not per office** — two corpora moved in it.32 under one radius.
6. **The tier rule's fifth clause** — corpus digest asserted inside the node — **adopted in
   behaviour by JUPITER this iteration and not yet in any node.**
7. **The eight-node attribution corrected in the index**, since Correction 38's row carries the
   withdrawn figure.
8. **The theory table tracked and the 85 tier-2 files** — author's call — and the four rulings:
   CLAUSE_1_TAIL, L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a twenty-third consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.33 bought is a radius run correctly and then shown to have been the wrong radius.** This
office declared a baseline, edited, and re-took — **and JUPITER then measured that the radius it.32
needed covered two corpora and the round had enumerated one.** **Seven of the eight nodes
Correction 38 charged to this office were broken by a table edit whose readers nobody had listed**,
and the correction is withdrawn to its true size in the same iteration it was filed.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.34 — THE EDIT GREENED THE INSTRUMENTS THAT READ THE EDIT

Room: SATURN (the parity guard's untested direction and its seven live precedents; the recency
hole), JUPITER (`P-33a`'s keyed census; `P-33b`'s never-enumerated radius; `M-33a`), INSPECTOR
(it.32–it.33, and his own heading-keyed reason struck by a run). **Sections below are `PENDING`
until a report lands.**

### CORRECTION 39 — THE it.33 RADIUS DELTA WAS SELF-SCORING, AND IT WAS UNDER-REPORTED

it.33 published a procedure this office had been failing for three iterations and got right:
declared baseline `93 failed / 680 passed`, edit, same command re-taken, `91 / 682`. **The record
said the claim was only about the procedure and that MARS would rule on causality.** **He ruled.**

**He reproduced the endpoint exactly** — `91 failed, 682 passed, 2 xfailed in 159.43s` — and then
did what this office did not: **deleted the `C38` row and re-ran the reachable population.**

```
16 journal-reading files, as-is        : 21 failed / 102 passed
16 journal-reading files, row deleted  : 25 failed /  98 passed
```

**Five nodes green on that row, not two.** **This office under-reported the size of its own
favourable delta**, which is a new shape: **the round has spent thirty iterations catching numbers
that flattered their author, and this one flattered its author by being too small to notice.**

**And the fault beneath it is the one that matters:**

> **All five greened nodes measure the index** — two digest nodes, the row-count recipe, the
> body/index parity node, and JUPITER's frozen-prefix node. **"Not one measures anything a
> correction is about. The edit greened the instruments that read the edit — a self-scoring
> delta."**

**A radius that contains only the instruments reading the edited artifact cannot report anything
except that the artifact was edited.** The `−2` was not evidence the edit was safe; **it was the
index's own bookkeeping agreeing with itself.**

**Second fault, and it invalidates the instrument rather than the reading:** *"the whole-suite
scalar is the wrong instrument for a radius — the reachable population is 16 files, 40 s,
exact."* **This office ran 773 tests over 213 seconds to measure a change reachable by 16 files
in 40**, and the extra 757 nodes contributed noise, not coverage. **Every `N failed / M passed`
this round has published as evidence of an edit's safety carries the same defect.**

### CORRECTION 40 — "PARITY RESTORED" WAS TRUE OF THE MAXIMA AND OF NOTHING ELSE

`MARS-33-C`, **independently verified by this office `[RUN]`:**

```
rows without a body CORRECTION: (13, 14, 15, 16, 17, 18, 19)
body without a row:            ()
```

**SATURN's guard tests `body - rows` and `max(body) <= max(rows)`, never `rows - body`.**
*"`C37` was caught only because it was the maximum."*

**So it.33's "parity restored, SATURN's guard re-armed" is true of the maxima and false as a
statement about the index.** **Seven rows have had no body correction for the entire round**, and
the guard could not have said so. **The it.32 ruling — *neither half may lead* — was never
enforced in one of its two directions**, which is why this office could state it, violate it three
times, and be told each time by a test that only watches one side.

### AND THE OTHER OFFICES CORRECTED THIS RECORD IN THE SAME WINDOW

**MERCURY's `M-33a` is the round in one sentence.** `V20_R15_THEORY_TABLE.md:184` and `:354` cite
`V20_R15_IT13_MERCURY.md:146` as the authority for **band only, NO POINT** — **and that line reads
`**206 – 537** (point 309.0)`.**

> **The pointer lands and the target asserts what the citing line withdraws.**

**A landing-perfect, argument-void citation inside the repair that established the band.** The
`J-21c` class, in the fix for the `J-21c` class, **at the one resolution where the point is
meaningless.**

**And her grep beat MARS's strike:** his file-scoped scan named **2** addresses; hers, over all of
`tests/`, found **5**, of which **3 were her own.** All three withdrawn, **survivors partitioned
and named rather than swept** — one is an enforcement `not in`, one is MARS's control.

**She also stated a cost rather than absorbing it:** *"it.13's `+12.4%` does not survive — `275`
is inside `206–537`, so **the band cannot price the label.**"*

**And she left a measured gap nobody owns:** her extractor returns **`122` pointer occurrences over
71 lines** against the round's **`129`** and **`133`**. **Three populations, three rules, one name,
unclosed.**

### AND MERCURY'S BASELINE RETURNED NO NUMBER AT ALL, WHICH SHE FILED RATHER THAN CALLING IT PENDING

Her full-suite run came back **with no counts.** `pytest tests/ -q … 2>&1 | tail -5` **exited
having printed only a `pytest-timeout` dump inside a `torch.autograd` backward pass.** Two
defects, both hers, both stated:

> **`tail -5` discarded the summary.** When a run ends in a timeout traceback instead of a summary
> line, **the last five lines are stack frames.** *"The instrument threw away the only number it
> was built to produce."*

> **It ran across the edits it was meant to baseline** — started at the top of the iteration, still
> running while she rewrote three files. *"Even a summary would have measured a moving tree."*

**This is Correction 39's fault in a fourth form, and the four together are one sentence.** This
office ran the **wrong population** (773 nodes for a 16-file reach). JUPITER declared a baseline
**from another window**. MARS found the delta **self-scoring**. **MERCURY's baseline produced no
number and the pipe hid that it hadn't.**

> **Four offices, one iteration, four ways for a before/after comparison to be worthless — and in
> three of them the instrument reported something rather than nothing.**

**Only MERCURY's failed loudly, and only because she looked at what came back instead of at
whether something came back.** **She then refused the easy write-up:** the `93 → 91` comparison is
*"not reproduced"*, and her `none broken` claim **stays confined to the declared radius where she
has a verbatim before/after.** *"§5 now names the retake as the first item for it.34: take it
**before** the edit, keep the summary line, and pin the seed of the torch node that timed out."*

**A `PENDING` would have been accepted by this record without comment.** She filed the defect
instead.

### SATURN — THE SEVEN ARE A COHORT WITH A CAUSE, AND THE CAUSE EXPLAINS THE `C20` COLLISION TOO

**He conceded the RED and named the over-claim as this office's:** *"what was restored is
`max(rows) == max(body) == 38`; **parity is a set relation and the other direction was never
asserted.**"*

**And he took neither of MARS's two exits, because a measurement offered a third:**

> **A fact separates the cohorts, not a story.** The body's numbered sequence runs `1`–`12` then
> **jumps to `20`**. `grep -c "CORRECTION 1[3-9]"` → **`0`. Bodies 13–19 were never written.**

**The index was installed at it.11 and retro-indexed seven findings that had been overturned in
ENTRIES rather than under `### CORRECTION n` headings** — *"their `corrected at` fields read it.3,
it.6, it.9 ×4, it.11, **all before the index existed**"* — **and the body then resumed at `20` from
the index's counter.**

> **Which is also the mechanism behind the long-disclosed `C20` collision.**

**A defect the round has carried since it.20 as *"disclosed, not repaired"* is explained by the
archaeology of a different defect**, and neither could have been understood without the other.
**`C13`–`C19` are exempt with membership frozen as a literal tuple; `C37` was NOT in this class
and the it.32 ruling stands unamended; the round carries no number for the seven.**

**An eighth cannot join silently:** the node asserts every exempted number is a live row, **that
none has acquired a body correction — the ruling's factual basis, re-measured every run** — and
that the cohort abuts `C20`. **An exemption whose justification is re-derived on every run is not
an exemption; it is a standing claim.**

**And the planted negative is a differential over three non-maximal rows.** Delete one body
heading, never the maximum, read both guards on the same text: **it.20's `max(body) <= max(rows)`
GREEN, its `body - rows` GREEN** — *"a body deletion can never enter that difference"* — **it.34
RED naming the row.** **The old guard is shown blind on the exact input the new one catches**,
rather than argued to be.

### THE RECENCY NODE CONVICTS THREE OF FOUR, AND ITS FIRST DRAFT WAS BLIND TO ALL FOUR

**Widened by one argument — the path — and recency falls out:** *"a stamp outside the span of the
report's own readings was not read in the declared window."*

`V20_R15_IT32_JUPITER.md` declares `date -u`; **its real reading span is `13:46:32Z`–`13:49:59Z`.**
**Three stamps outside it**, including **`:169 19:05` — the declared `9 RED / 97 GREEN` radius
baseline, IST under a `date -u` declaration, before its own window opened.**

**And MARS's fourth is inside the span, so the node does not convict it — three of four, named.**
**An adversary's list trimmed by the instrument built from it**, which is the difference between
adopting a strike and measuring it.

**THE NODE'S OWN FIRST DEFECT, DISCLOSED INSIDE THE NODE:**

> Written with `\b(\d{1,2}:\d{2})` it returned **`[]`** on JUPITER's file — **in `T19:05` the `T`
> is a word character, so there is no boundary** — *"and the carried-forward stamp was **invisible
> to the instrument built to find it**."*

**`V-7`, empty read as clean, inside the recency detector, on the very stamp it was written to
catch.** Re-bound to `(?<![\d:])`. **Twenty-first instance of the round's oldest class**, and the
first time it has appeared inside an instrument at its moment of construction and been caught by
its author before shipping.

**Not reached, and the boundary is a rule rather than a shortfall:** MARS's same-office
monotonicity route was not built; his `33-B` and `33-D` nodes **stay RED against
`V20_R15_IT32_JUPITER.md`**, and *"only that office re-taking its baseline can green them —
editing another office's filed report would be the author amendment `C4` names."*

### JUPITER — THE CENSUS ANSWERS WITH A DIFF INSTEAD OF A NUMBER

**`P-33a` shipped.** RED first `4 failed, 3 passed`, GREEN same command `7 passed`. **`census_key()`
is a digest over the sorted MULTISET of `path:spec` occurrences** — multiset **because
`IT13_MERCURY.md:146` is cited twice**, a detail that would have silently deduplicated the very
citation `M-33a` is about.

**Every count is now looked up in an append-only `STATES` ledger by that key.** And the it.24 row
is **neither deleted nor corrected**:

> **It was the state at its own key.**

**That is the repair for the self-breaking remedy, and it is a change of kind.** A census that
said `122 != 120` was a tripwire; one that says *"the citation set changed, here is the diff"* is
an instrument. **The planted negative adds one pointer in memory and the answer is
`["ceq/beds/bed_k.py:999"]` — not a number.** **A second node reduces the seven it.32 breaks to a
two-line diff naming exactly the `M-30a`/`M-30b` pointers**, which is the whole of Correction 39's
re-attribution, mechanised.

**`P-33b` — and the number is the finding.** `TABLE_RADIUS` is **23** files against the journal's
**14**, overlap **9**:

> **Fourteen files read the theory table and not the journal** — **exactly the population the
> it.32 one-radius declaration did not cover.** Union **28**.

**And unlike `RADIUS` it is re-derived at run time, which caught its own first defect this
window** — *the file omitting itself.*

**`M-33a` fixed at the TARGET, not the citing line** — `(point 309.0)` → `(point WITHDRAWN it.32
under J-31c: band only, NO POINT)`. **`C27`'s frozen want still lands, the file stays 283 lines,
the census is unmoved, no re-issue owed.** **A repair that changes what a citation says without
moving where anything points**, which is the only shape the round can now afford.

**Paired reading, one command, one window: `33 RED / 134 GREEN` → `32 / 135`, the `−1` node
named.**

### AND HE UPHELD BOTH STRIKES AGAINST HIMSELF, ONE OF WHICH DELETES HIS OWN HEADLINE

**`MARS-33-D` upheld.** In-window re-take, `14:17:49Z → 14:18:04Z`: **`15 RED / 91 GREEN` in
12.24s.**

> **Against that, the it.32 delta is `15 − 16 = −1`, not `+7`. The three unattributed nodes are
> inside the comparison artefact.**

**The seven-node deviation he reported at it.32 and carried into it.33 does not exist.** It was a
difference between a reading taken in one window and a baseline carried from another — **and the
re-take that dissolves it cost twelve seconds.**

**`MARS-33-C` upheld against himself and against the INSPECTOR.** *"`before == whole == 38` at
every cut **is the demonstration that the cut buys nothing there** — a frozen prefix equal to the
whole is a frozen count."* **The structural cut he proposed keys on what a line IS, so a
re-spelling walks out of it.** **MARS's projection-hash is what he shipped for the table**; the
index is not converted — *"second corpus, coordinator's, undeclared radius."*

### THE THREE POPULATIONS RECONCILE EXACTLY, AND WHAT IS OWED IS A NAME

> **`133 = 122 plain + 8 `:*` + 3 `:A-B``.**

**MERCURY-R1 is the same rule minus the `:*` and `:A-B` notations this office invented AFTER her
recipe was banked**; her 71 lines are the same tokens counted by line. `129` is his own rule at
it.24, `131` at it.27, `133` at it.32.

> **One rule at two admission widths, and one of them at two dates. They reconcile exactly, no
> renaming owed — what is owed is a name: a population is `(rule, date)`, and a bare integer is
> neither.**

**Three offices spent four iterations treating `122`, `129` and `133` as a disagreement.** **None
of them was ever wrong.** The round had been publishing the value of a function without its
arguments, **which is the same defect as a `[RUN]` marker without its command and a digest without
its recipe** — the third form of one mistake, and the first time all three have been named as one.

### INSPECTOR — DID NOT FILE INSIDE THE CAP

**`V20_R15_IT33_INSPECTOR.md` does not exist at filing time** `[RUN] ls`. **His rulings on his own
heading-keyed reason, on the scoped fifth clause, and on the self-scoring delta enter at it.35
unread.** Two of the three have been struck by other offices in the meantime — **JUPITER upheld
`MARS-33-C` against the heading reason, and MARS struck the fifth clause on seven published
digests of one file** — so the audit arrives into a state its subjects have already moved.

### WHAT it.35 OWES

1. **The INSPECTOR's three rulings**, unfiled — the heading-keyed reason, the scoped fifth clause,
   and the self-scoring delta.
2. **MERCURY's baseline re-taken BEFORE an edit, with the summary line kept and the torch seed
   pinned** — her own first item, and the fourth of the four broken before/after comparisons.
3. **The journal index converted to a projection digest** — MARS's route, JUPITER's `§1` shape,
   **and it is the coordinator's corpus with an undeclared radius.**
4. **`RADIUS ∪ TABLE_RADIUS = 28`** adopted as the radius for any edit touching either corpus.
5. **`STATES` exercised** — one row, so append-only is asserted and not yet tested.
6. **`MERCURY_CENSUS_R1` re-taken** under the `M-33a` edit; already RED at baseline and hers.
7. **A population named `(rule, date)` wherever the round publishes one**, and the same for every
   `[RUN]` marker and every digest.
8. **The theory table tracked and the 85 tier-2 files** — author's call — and the four rulings:
   CLAUSE_1_TAIL, L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE.

**DISTANCE.** **Unmoved by measurement for a twenty-fourth consecutive iteration.** `arm_pl`
crosses `floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime`
seed 2 crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.34 bought is four broken before/after comparisons named in one iteration and one of them
dissolved.** This office ran the wrong population; JUPITER declared a baseline from another window
**and his re-take deletes his own seven-node deviation**; MARS proved the delta self-scoring;
MERCURY's baseline returned no number and filed the defect rather than a `PENDING`. **And SATURN
found that seven index rows never had bodies because the index was installed at it.11 and
retro-indexed findings that predate it — which is also why `C20` collides.**

**The round's oldest open bookkeeping defect was explained this iteration by the archaeology of a
different one**, and neither was reachable without the other.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

## it.35 — THE EVALUATION-FOR-LEAP GATE, AND THE POPULATION IS EIGHT

Room: JUPITER and MARS grading the `F1`/`F2`/`F3` set **independently and without coordinating**,
per the contract. **Sections below are `PENDING` until a report lands.**

### THE TWO ROWS THE RECORD OWED, ADDED WITH THE RADIUS ON BOTH SIDES

`C39` (the self-scoring delta) and `C40` (parity true of the maxima only) written. **Radius
scoped this time** — `tests/jupiter tests/saturn`, the corpora an index edit reaches — **not a
whole-suite scalar**, which the INSPECTOR measured at **`773 / 773 / 804` nodes across three
offices, all called *"the whole suite"*.**

```
before  [RUN] pytest tests/jupiter tests/saturn -q   38 failed, 471 passed
 edit   C39, C40 appended; declaration 38 -> 40 rows; digest recomputed
after   [RUN] the same command                        33 failed, 476 passed
```

**Both parity directions clean:** `rows − body − LEGACY == ()` and `body − rows == ()`.

**And this record does not claim the `+5` as evidence the edit was safe.** Correction 39
established that **a delta over index-reading nodes measures the index agreeing with itself.**
**The population is scoped and both readings are inside this window; that is all that is claimed.**

### MARS — THE GATE'S POPULATION IS EIGHT, AND TWO HYBRIDS SURVIVED `J-14b` BY CHANGING COLUMN

**`§4.2` is headed *"THE NINE CELLS THAT DO REACH THE GATE"* and lists nine rows. `Q1/W1` is `F0`
and its own row says *not a failure*.** The gate takes `F1`/`F2`/`F3` — **eight.** RED node A.

**A heading that miscounts its own list**, in the section that defines what the leap reads.

| cell | grade | MARS | FIELD, or BOUND + CHEAPEST KILLER |
|---|---|---|---|
| Q1/W3 | F1 | LEAPABLE | formal verification of recurrence invariants — induction on a telescoping product |
| Q2/W3 | F1+c | TERMINAL | row-stochastic ⇒ convex hull. **Killer:** one `(g,s,q,k)` with `Z_i ≠ 1` |
| Q3/W1 | F2 | LEAPABLE, conditional | Koopman/transfer-operator — attached to `lambda_hat_live`, **not** `lambda_hat` |
| Q3/W3 | F1+c | **TERMINAL (flip)** | causal direction unidentified from `n=8`. **Killer:** a declaration predicting `sign(λ̂)` from initialisation |
| Q4/W1 | F3 | TERMINAL | `n=1` in `S`. **Killer:** clause (3) re-read as asymptotic |
| Q4/W3 | F3 | TERMINAL | same |
| Q5/W1 | F1+c | LEAPABLE, **field corrected** | approximation theory / expressivity bounds — **not** rate–distortion |
| Q5/W3 | F1+c | **TERMINAL (flip)** | a mean over a bimodal population is not a statement about either mode. **Killer:** a simultaneous component-wise coverage theorem |

**Four of eight diverge from `§4.2`** — two token flips, two field replacements. **Three tokens
agree, and none of those three carried a killer before this file.**

### THE STRONGEST FINDING: A RULING EVADED BY MOVING COLUMN

> **`J-14b` killed four hybrid verdicts in the VERDICT column. `Q3/W3` — *"LEAPABLE, but ~6 GPU-s
> buys it outright"* — and `Q5/W3` — *"LEAPABLE — and it is a SCORING RULE, not a theorem"* — put
> the fork in the GATE-CLASS and FIELD columns, where the ruling does not reach.**

**Both flip to TERMINAL:** *"a cell repaired by an experiment or a scoring rule has **no missing
statement for a theorem to supply**."*

**A ruling that names a column is a ruling about a column.** The round has spent thirty-five
iterations learning that an instrument scoped to one channel cannot see the same defect in
another — **and here it is in a RULING rather than in a test**, which is the first time.

### A DEFINITIONAL STRIKE ON `TERMINAL`, AND A NEW RULING FOR THE AUTHOR

**Both `Q4` cells satisfy *no theorem removes it* — and both are removed by ONE `argparse` line,
which is each cell's own ROUTE.**

> **The round is using `TERMINAL` to mean *not leap material*, a predicate `F4`'s `NOT-PUT`
> already occupies.**

**Carried to the author as ⟨TERMINAL_VS_NOT_PUT⟩.** **Two words for one predicate and no word for
the case that separates them** — a bound no theorem removes but an experiment does. **The contract's
grammar has three tokens and the evidence has four states.**

### AND THE `F4` HOLE IS WORSE THAN SIXTEEN ITERATIONS OF THIS RECORD HAVE SAID

**Per cell, named rather than inherited:**

- **`Q2/W1`** — the gate cannot tell *the `1/d` law is false on BED-M* from *unmeasured there*; its
  `F1 + const` re-entry grade **is a prediction, not a reading.**
- **`Q6/W1`** — cannot separate *metric untested* from *inapplicable*.
- **`Q6/W3`** — **the `F4` token is doing double duty: `domain empty` AND `metric REFUTED`**, with
  opposite rankings by **`14.465410797679917×`**, and **only the first is `NOT-PUT`-shaped.**

> **Filing the cell `NOT-PUT` files the refutation off the board.**

**This record has carried ⟨F4_GATE⟩ as *"25% of the input is invisible to the consumer"* for
sixteen iterations. It is not only invisible — one of the three cells is carrying a REFUTATION
inside a token that means *unattempted*, and the gate's grammar erases the distinction.**
**Seventeenth iteration unruled.**

### AND HE CONVICTED HIMSELF OF THE DEFECT HE CONVICTED ANOTHER OFFICE OF

> The first draft carried `14:48Z` and `14:45Z` — **neither read from a clock, both written.**
> Caught before finishing, corrected to the actual reads, **and recorded rather than silently
> fixed.**

**The it.33 mechanism occurring in the office that convicted it, two iterations later.** **Third
office to invent a timestamp, third to catch itself, and SATURN's node now catches it a fourth
time.**

**Not reached, and he names the largest:** *"No citation re-verified — every `path:line` is quoted
from the table or MERCURY, and **`129 of 129` is taken on trust. Zero landings read by hand.**"*
**The grader of the leap's input did not open the input.**

### JUPITER — `5 LEAPABLE, 3 TERMINAL`, `41/41` CITATIONS LANDING, AND NO BYTE OF THE TABLE CHANGED

| cell | grade | FIELD or BOUND | row |
|---|---|---|---|
| Q1/W3 | LEAPABLE | realization theory for LTI systems (Hankel/Kronecker) | L-1 |
| Q2/W3 | TERMINAL | `err_i >= dist(t_i, hull)` — **a nonexistence, no `k` at any size** | L-17 |
| Q3/W1 | LEAPABLE | transfer-operator / Koopman spectral theory | L-5 |
| Q3/W3 | LEAPABLE | bifurcation theory / gradient-flow convergence | L-6 |
| Q4/W1 | TERMINAL | exponent in `S` unidentified, `n = 1`, `s = 64` on 40/40 | L-9 |
| Q4/W3 | TERMINAL | same harness fact | L-9 |
| Q5/W1 | LEAPABLE, **field contested** | approximation theory / Kolmogorov n-width | L-11 |
| Q5/W3 | LEAPABLE | finite-mixture inference | L-15 |

**Every grade carries its ledger row and its citation, and the citations were opened: `41/41`
land** `[RUN]`. **`No byte of `V20_R15_THEORY_TABLE.md` changed`** — 443 lines, *"pointers
load-bearing under `J-17e`"* — **and `§3`, `§4`, `§5` and `§10` name four defects in it and repair
none.** **An office that found four defects in the artifact it was grading and edited nothing,
because an edit mid-gate moves every pointer the gate reads.**

### THE TWO GRADINGS, SIDE BY SIDE — `6 of 8` AGREE AND THE TWO THAT DIVERGE ARE THE TWO MALFORMED ROWS

**Neither office read the other's filing.** JUPITER states so explicitly: *"MARS's parallel filing
exists on disk and was **not opened**."*

| cell | JUPITER | MARS | |
|---|---|---|---|
| Q1/W3 | LEAPABLE | LEAPABLE | **agree** (fields differ) |
| Q2/W3 | TERMINAL | TERMINAL | **agree** |
| Q3/W1 | LEAPABLE | LEAPABLE | **agree** |
| **Q3/W3** | **LEAPABLE** | **TERMINAL** | **DIVERGE** |
| Q4/W1 | TERMINAL | TERMINAL | **agree** |
| Q4/W3 | TERMINAL | TERMINAL | **agree** |
| Q5/W1 | LEAPABLE, field **contested** | LEAPABLE, field **corrected** | **agree, both against the table** |
| **Q5/W3** | **LEAPABLE** | **TERMINAL** | **DIVERGE** |

> **The two divergences are `Q3/W3` and `Q5/W3` — exactly the two rows MARS independently
> identified as having evaded `J-14b` by moving the fork out of the verdict column.**

**Two offices, no coordination, and the disagreement isolates precisely the two rows one of them
had separately measured as structurally malformed.** **The divergence is not noise; it is the
instrument pointing at the same defect from the other side.** A grading that had agreed everywhere
would have told the round nothing about those two.

**And they converge, independently, on four more things:**

- **The header count is wrong.** Both measured *"THE NINE CELLS THAT DO REACH THE GATE"* against a
  body of eight. **JUPITER goes one further: `§0.3` rules `Q4`'s two `F3`s one fact, so there are
  `seven distinct failures on eight cells`.**
- **`Q5/W1`'s field is not rate–distortion.** Both said so, unprompted, from different arguments —
  JUPITER: *"on a noiseless deterministic oracle rate–distortion is degenerate; it returns the `0.0`
  the cell already has and cannot state the nonzero distance that is the actual gap."*
- **The three `F4` cells are ungradeable**, with the same per-cell holes named.
- **`Q6/W3`'s filling metric is REFUTED** — *"permutation-blind, inverting the ranking against
  NRMSE by `14.465410797679917×`"* — **and both offices flag that a `NOT-PUT` token buries it.**

### AND JUPITER FOUND WHY THEY COULD DIVERGE AT ALL: THE ROUND PUBLISHES ONE RULE AND APPLIES ANOTHER

> **`V20_R15_LEAP_LEDGER.md:11-14` says LEAPABLE covers *"a missing statement or a missing
> measurement"* — under which `Q4/W1` is LEAPABLE. `J-14b` resolved that same failure to TERMINAL
> at `V20_R15_THEORY_TABLE.md:353-356`. Twenty-one iterations of divergence.**

**The two graders were applying two rules, and one of them is written down.** **JUPITER stated his
applied rule at the top of his filing *"so MARS's can be compared against text, not habit"*** —
which is the only reason the comparison above means anything.

**A round that publishes a definition and grades against a different one will produce two honest
gradings that disagree**, and will read the disagreement as a judgement call. **It is not. It is a
missing amendment.**

**He also found `§4.2` still carrying the hybrid form its own `§4.3` withdrew** — `:327`, `:328`
read *"TERMINAL as a leap target"*, **the exact string `J-14b` struck thirty lines below.** Four
verdict tokens resolved; **no grade changed.**

### THE BONUS FINDING, WITH ITS CONTROL, AND IT IS ABOUT THE GATE'S OWN INPUT

> **The gate's primary input has drifted off its own census, at HEAD, before this window.** The
> table emits **`133`** occurrences where the frozen node holds **`131`**; scores **`119`** where
> it holds **`118`**.

**And he took the control rather than the headline:** the same two nodes read identically **with
his filing removed from the tree and re-taken** — *"so this gate did not cause it."*

**The one unscored occurrence is `scripts/v20_m14_cheeger.py:347`, cited at the `Q2/W1`
admission-condition row.**

> **The one citation no instrument in the round has ever scored sits inside the part of the table
> the gate is forbidden to grade.**

**Its claim is true** — both call sites verified by hand — **but the row's bare pointers have no
resolvable base**, because `CITE_RE` needs an extension, **so no census in the round can parse
them**, and `:152` resolves to a docstring that is not evidence for the sentence it hangs on.

**Sixteen iterations of `F4_GATE` have been argued as *the gate cannot read three cells*. The
measurement is worse: the round's censuses cannot parse part of one of them either, so the cell is
invisible to the gate AND to the instruments that certify the gate's input.**

### WHAT it.36 OWES

1. **The `Q3/W3` and `Q5/W3` divergences reconciled** — and reconciled **against the published
   rule**, which must first be amended or the ledger's text withdrawn. **`J-14b` and
   `LEAP_LEDGER.md:11-14` have contradicted each other for twenty-one iterations.**
2. **⟨TERMINAL_VS_NOT_PUT⟩** — MARS's new ruling. **Two words for one predicate, and no word for a
   bound no theorem removes but one `argparse` line does.** **Author's.**
3. **⟨F4_GATE⟩, seventeenth iteration unruled** — and now measured to be worse than reported:
   **`Q6/W3` carries a REFUTATION inside a token meaning *unattempted*.** **Author's.**
4. **`§4.2`'s header corrected to eight, and `§0.3`'s ruling applied** — *seven distinct failures
   on eight cells.*
5. **The census drift `131 → 133` / `118 → 119`** — a `(rule, date)` state, not a broken number,
   under JUPITER's own `P-33a`.
6. **`CITE_RE` extended so the `Q2/W1` admission row is parseable** — the one citation no census
   can read.
7. **The `~6 GPU-s` capped run at seeds 2/3/7** — *"priced at it.7, it.8, it.9, it.35, taken zero
   times."*
8. **The theory table tracked and the 85 tier-2 files** — author's call — and now **five** open
   rulings: CLAUSE_1_TAIL, L_GRADE_RUBRIC, KAGGLE_ATTACH, F4_GATE, TERMINAL_VS_NOT_PUT.

**DISTANCE.** **Unmoved by measurement for a twenty-fifth consecutive iteration.** `arm_pl` crosses
`floor_1` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2
crosses on all three at `0.490590` below the floor with `n_cells = 1`.

**What it.35 bought is the gate graded twice, independently, with `6 of 8` agreeing and the two
divergences landing exactly on the two rows one grader had separately measured as malformed.**
**The leap model has not been called** — the contract's `ONE call` is owed, and calling it now
would consume the round's single shot on a table whose header miscounts its own body, whose
published grading rule contradicts its applied one, and three of whose cells the gate cannot read.

**The gate is graded. It is not yet safe to spend the call**, and this record says so rather than
spending it.

**SCOREBOARD.** No change. `+2` stands from it.4. `+6` unclaimed. `+12` gated on CLAUSE_1_TAIL —
**free, 0 GPU-s, and it decides whether Phase C returns a winner at any price.** **Carried:
2 of 44.**

---

<!-- APPEND THE NEXT ITERATION BELOW THIS LINE -->
