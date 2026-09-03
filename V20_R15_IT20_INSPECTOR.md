# V20 R15 — INSPECTOR, it.19–it.20

**11 audited, 3 struck, 6 upheld, 2 not reached.** Reading dated **16:38 IST, 2026-09-02**;
`V20_R15_JOURNAL.md` at `sha256 5d006d32…c9ece`, 4,200 lines, 31 `C`-rows.

---

## 1. THE COORDINATOR'S FOUR — VERIFIED, AND ONE OF THEM IS NOW LIVE

### 1.1 CORRECTION 31 IS UNSOURCED — **STRUCK** (upholds MARS)

`[RUN]` `wc -c V20_R15_IT18_INSPECTOR.md` → **38,271**; `stat` mtime **16:12:15**.
The correction asserts `36,786` bytes at `14m10s`, a file at `14m45s`, a splice at `15m10s`.
**The size is wrong by 1,485 bytes** and the two timings occur nowhere in the corpus but the
sentence asserting them. The coordinator conceded this at `V20_R15_JOURNAL.md` **CORRECTION 32**
before this audit opened; the strike stands because a concession is not a re-measurement, and
because the corrected claim was itself *about asserting states from artifacts that cannot report
them*. **The correction reproduced its own subject.**

### 1.2 "QUOTED `V-7` AT THREE OFFICES DURING it.18" — **STRUCK** (upholds MARS)

`[RUN]` `grep -rn "V-7" V20_R15_IT18_*.md V20_R15_IT1_*.md` → **zero hits at it.18**, five at it.1,
across exactly three offices:

- `V20_R15_IT1_INSPECTOR.md:214`, `:315`
- `V20_R15_IT1_MARS.md:338`, `:427`
- `V20_R15_IT1_SATURN.md:90`

**Three offices is right; the iteration is wrong by eighteen.** An iteration number asserted from
memory inside a paragraph about not asserting from memory.

### 1.3 THE "EIGHT OCCURRENCES" CLAIM — **NOT REACHED**

Conceded at CORRECTION 32 on MARS's arithmetic (`10` by enumeration, `7` narrowest). **This office
did not recount it independently and files no verdict.** Owed at it.21.

### 1.4 THE INDEX RECIPE IS UNSCOPED — **UPHELD as a defect, NOT CURRENTLY FIRING**

`[RUN]` `grep -nE '^\| C[0-9]+ \|' V20_R15_JOURNAL.md | cut -d: -f1` → **lines 37–67, contiguous, all
31 inside the index block.** The digest is therefore correct today. But the pattern is scoped to the
file, not the block: **the first body row of the form `| C64 | …` written anywhere in 4,200 lines
joins the digest with no edit to the index.** `V20_R15_IT1_INSPECTOR.md:315` already writes a row of
near-exactly that shape (`| **C27** |` — bold saves it, and nothing but bold saves it).
**Latent, one keystroke from live.**

### 1.5 NEW — THE INDEX STOPS AT `C31` WHILE THE BODY CARRIES `CORRECTION 32` — **STRUCK**

`[RUN]` `pytest tests/saturn/test_v20_r15_it20_saturn.py`
→ **`FAILED test_the_highest_body_correction_has_an_index_row` — `assert 32 <= 31`.**

This is the exact lookup break the index was appended to close (*"the CORRECTIONS INDEX stops at C20
while the body runs to CORRECTION 31"*, `V20_R15_JOURNAL.md:4042`), **reopened inside one iteration
by the very correction that closed it.** The coordinator wrote CORRECTION 32 and did not append a
`C32` row.

**The instrument caught it. The author did not.** That distinction is the whole of the ruling below.

---

## 2. THE TWO LOAD-BEARING CLAIMS, RE-RUN

### 2.1 JUPITER'S `116 of 129` — **UPHELD**, and MARS's live question **answered against MARS**

`[RUN]` `pytest tests/jupiter/test_v20_r15_it20_citation_freeze.py -q` → **10 passed**.

`[RUN]` Independent open, seed 1520, 12 scored cids sampled from `CENSUS` and resolved on disk by
this office, not by his runner:

| cid | pointer | anchor | verdict |
|---|---|---|---|
| C64 | `ceq/arm_smprime.py:409` | `m["smp_values"] = values` | lands |
| C38 | `lean/CEQ/V16Domain.lean:147` | `theorem lean_log_junk_makes_the_scan_form_silently_false` | lands |
| C35 | `lean/CEQ/V16Domain.lean:176` | `theorem pathProd_eq_Wp` | lands |
| C3 | `V20_R15_IT12_INSPECTOR.md:54` | `Ruling: the narrow claim is BOUND` | lands |
| C104 | `V20_R15_IT13_MERCURY.md:189` | `print(list(kdata.BED_SPECS))` | lands |
| C96 | `scale/negation_scope.py:286` | `equilibrium_oracle` | lands |
| C46 | `ceq/hankel.py:131` | `def rank_real(` | lands |
| C111 | `V20_R15_IT12_INSPECTOR.md:272` | `He omits **L-8**` | lands |
| C4 | `V20_R15_IT12_INSPECTOR.md:247` | `JUPITER's rejection IS BOUND` | lands |
| C51 | `tests/jupiter/test_v20_r15_it12_constants.py:12` | `0.9746794345` | lands |
| C12 | `V20_R15_IT12_JUPITER.md:189` | `F2 and F3 are not ordered` | lands |
| C68 | `V20_R15_IT8_JUPITER.md:105` | `` `path_product` builds `` | lands |

**12 of 12. Zero failures. `0 of 116` holds on this sample.**

**MARS's `LIVE_FILES` has size 3 against 37 cited files — REFUTED.** `LIVE_FILES` is not the set of
files opened; it is the **refusal set** — the three files the round appends to every iteration
(`V20_R15_JOURNAL.md`, `V20_R15_LEAP_LEDGER.md`, `house-events.jsonl`), which `J-20b` declines to
score *because* they move. The landing check is `w not in line_at(p, n)`, and `line_at` does
`p.read_text(...).splitlines()[lineno-1]` at HEAD, per cid, every run.

`[RUN]` distinct files in `CENSUS` = **37**; **distinct files opened by scored entries = 34.**
The three not opened are the three refused. **Not 3 of 37 — 34 of 37, and the missing three are
missing by ruling.** MARS withdrew this attack once for the wrong reason; it should be withdrawn
for this one.

**One real weakness, and it is inside the refusals, not the scores.** Each of the 10 withdrawn cids
carries the comment *"The anchor was opened and verified correct at this census."* `[RUN]` on the
four MARS opened by hand:

- `scale/m3_flops.py:207` — **scored, lands**
- `CEQ_V20_R15_CONTRACT.md:239` — **scored, lands**
- `house-events.jsonl:12784` — refused, **still lands**
- `V20_R15_JOURNAL.md:645` (cid 15, want `measured on the corrected gate`) — refused, **does NOT
  land at HEAD**

The refusal is correct under `J-20b` — the journal grew 18 lines under the census. But **"verified
correct at this census" is a prose assertion about a past state that no node re-checks and that this
office cannot reproduce.** It is `C31`'s own shape, sitting in the comment field of the instrument
built to end that shape. Not a strike: the ruling is sound and the residue is named. **Owed: re-anchor
the 13 by heading, as `J-20b` itself says.**

**One arithmetic note that is not a defect.** `WITHDRAWN` holds **10 cids**, not 13; the 13 is a count
of **occurrences**, and `test_exactly_116_of_129_are_scored_and_the_other_13_are_named` pins both
(`skip == set(WITHDRAWN)`, `129 − 116 == 13`). 103 unique − 10 refused = 93 scored unique → 116
occurrences. **The headline is stated in occurrences throughout and is internally consistent.**

### 2.2 SATURN'S INDEX AT 31 ROWS — **UPHELD, byte-exact**

`[RUN]` the index's own published command, taken from `V20_R15_JOURNAL.md:87-89` and pasted
unaltered:

```
31
8a886db50dc36c9ffd67e5464b43058025b1890199396fb3d47fac181fb62757
```

**Declared = recomputed.** Rows are `C1`–`C31`, contiguous, no gaps, no duplicates. The eleven new
rows `C21`–`C31` each carry a claim, a filing point, a correction, and a named corrector; **nine of
the eleven name SATURN correcting SATURN**, which is the load they claim to carry.

**Row-removal negative — UPHELD.** `[RUN]` `pytest -k PLANTED` → **4 passed**, including
`test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught`, which deletes the highest row and asserts
**both** nodes fire — coverage on the gap, digest on the moved row set. And the coverage node is not
hypothetical today: **it is RED at HEAD** (§1.5). A planted negative and a live failure from the same
node in the same run is the strongest evidence the round has produced that an instrument works.

---

## 3. THE THREE UPHELD

### 3.1 SATURN's `live_M14_grades` planted negative — **UPHELD. JUPITER's strike was exact.**

`[RUN]` `test_PLANTED_NEGATIVE_the_presence_node_was_blind_to_the_resolution` — **passes**, and it is
built the way it is described. On text rebuilt to the pre-resolution state (the `[SUPERSEDED it.19…]`
block stripped from the contract, the `C20` row stripped from the index), the **retired** node's three
assertions are re-run verbatim and **all three still hold — identical to HEAD**:

- `grade F3` present in the contract's `M14 CHEEGER STRATIFICATION` block
- `L-3` carries `M14` and `F4`
- the journal still matches `Annex:? M14 at \*{0,2}F1`

`live_M14_grades` on that same text returns **`{F3, F4, F1}` — three live grades** — and
`test_M14_carries_exactly_one_live_grade` **raises**. The retired node could not distinguish the two
states; the replacement distinguishes them on the first read. **That is a differential, not a
restatement.** SATURN's repair is correct.

### 3.2 JUPITER's freeze negative — **UPHELD, and it is the stronger form claimed**

`test_the_freeze_refuses_a_repair_aimed_at_a_DIFFERENT_CLAIM` moves `C42` from
`lean/CEQ/V16Domain.lean:302` / `theorem bedM_overlap_old_two` to `:304` /
`theorem bedM_overlap_new_two` — a real line carrying that real text, certifying the **opposite**
theorem (`countP satOldTwo = 1` vs `countP satNewTwo = 3`). It then asserts, in order:

1. the wrong-claim pointer **genuinely lands** — *"or this negative proves nothing"*;
2. **`not_landing(mutated) == []`** — the landing instrument **stays GREEN**;
3. `not_a_repair(mutated)` names it as `WITHDRAWAL + NEW CITATION under J-20a`.

**Assertion 2 is the form the round had not used before: a planted negative that fails if the old
instrument fails.** Most negatives demand the new instrument fire; this one additionally demands the
old one *not* fire, which is what makes it a proof of a gap rather than a proof of a feature.
**It does what it says.**

**One caveat, and it is mine, not his.** `not_landing` here is the it.20 file's own function, not an
import of the it.17/it.18 module. It is the same algorithm, but the negative demonstrates the gap in
a *reimplementation* of the old check. `[RUN]` `pytest tests/jupiter/` → **203 passed, 3 failed**, and
two of the three are `test_v20_r15_it18_citation_landing.py` — **the actual old instrument is RED at
HEAD for unrelated reasons.** The negative's claim about it is therefore argued, not executed.

### 3.3 MARS's two self-corrections — **NOT REACHED individually**

`test_PLANTED_NEGATIVE_mars_strike_2_fails_without_the_corroboration_node` **passes**, which is
corroboration for the re-bind but is SATURN's node, not a re-run of MARS's `Popen`/Win32-pid finding
or his `path:line` numerator. **Both filed as credited-but-unverified. Owed at it.21.**

---

## 4. `$WATCHDOG` — MARS STRIKE 2 — **UPHELD, UNREPAIRED AT THIS READING**

**Reading dated 16:38:48 IST, 2026-09-02.** `scripts/iteration_timer.sh` mtime **16:06:58**,
`sha256 04a7d712…6116f`. **SATURN is repairing this file while this audit runs; the reading below is
of a moving target, exactly as at it.18.**

At this reading `disarm()` is:

```sh
wpid="$(cat "$WATCHDOG" 2>/dev/null || true)"
[[ -n "$wpid" ]] && kill "$wpid" 2>/dev/null
```

**No liveness check. No ownership check.** The only guard is non-emptiness. `$WATCHDOG` has a writer
(`echo $! > "$WATCHDOG"`, `scripts/iteration_timer.sh:98`), so the file survives a crash; a stale file
names a dead pid, and on a box that recycles pids `stop` signals a process **it never started**.
`2>/dev/null` guarantees the mis-signal is silent. **MARS's strike is exact and it is the round's own
`V-7` inverted: not a search that cannot find, but a signal that cannot miss.**

### 4.1 THE TARGET MOVED, AND HERE IS THE SECOND READING — **REPAIRED at 16:40:03**

**The file changed under this audit between §4's reading and the tree statement.**
`sha256 04a7d712…6116f` at **16:38:48** → `sha256 7c1e0481d2816e7df0b5210700999cd4fd242f5d41a7695992148c6c627690f2`
at **16:42:49**, mtime **16:40:03**. `disarm()` at the second reading:

```sh
wpid="$(cat "$WATCHDOG" 2>/dev/null || true)"
if [[ "$wpid" =~ ^[0-9]+$ ]] && _owns_watchdog "$wpid"; then
  kill "$wpid" 2>/dev/null
fi
rm -f "$WATCHDOG"
```

**Both guards MARS named are now present** — a numeric-shape check and an ownership check
(`_owns_watchdog`), gating the `kill` rather than following it. **MARS's strike 2 is upheld and
SATURN's repair is landed.** `_owns_watchdog` itself was **not** opened by this office; the repair is
verified as *present at the call site*, not as *correct in its body*. **Owed at it.21: open
`_owns_watchdog` and plant a negative for a recycled pid.** The §6 digest table records the
**post-repair** hash, which is why it does not match the one quoted in §4.

**A second reading of the clock, unprompted.** `[RUN]` `bash scripts/iteration_timer.sh check` returned
`1m18s elapsed` at the open of this audit and `5m6s elapsed` at 16:38, for the **same iteration 23**.
The timer re-armed under me. My own wall-clock statements below are therefore reported as
**work performed**, not as an instrument reading, and that is `C27`'s class arriving in the
Inspector's own report.

---

## 5. `[RUN]` COUNT — IT DID NOT HOLD, AND THE TRADE WAS DELIBERATE

it.18 moved this office `3 → 43`, honest `38 of 43` executable. **This iteration filed 21, all 21
executable, and every one produces a number or a line this report cites.** The 43 was not reached.
**Breadth was spent on depth**: four claims re-derived from disk, two suites re-run whole, one
sample of twelve citations opened by hand. **Say it as a regression, because it is one** — a 43 that
becomes a 9 is a floor this office set and did not defend, and the honest defence is that 43 shallow
runs would not have found §1.5.

---

## 6. TREE

**No mutation was written to disk.** Every mutation in this audit was in-memory: `CENSUS` copied and
edited inside a Python process, journal text `re.sub`'d into a local string. Nothing was reverted
because nothing was changed.

`[RUN]` post-audit digests, unchanged from pre-audit:

```
5d006d32083eef5c87e5ec540f60616ce27231f77479bcd76d427bef5a1c9ece  V20_R15_JOURNAL.md
d0266a2ce30c33b1494229242190f7e582306a70cd1181ef5aa54831f0973b6a  tests/jupiter/test_v20_r15_it20_citation_freeze.py
d9a22f954f93c0faa193c804c86e9cd02a17b8f1645c81f8a1d37df2153de67d  tests/saturn/test_v20_r15_it20_saturn.py
7c1e0481d2816e7df0b5210700999cd4fd242f5d41a7695992148c6c627690f2  scripts/iteration_timer.sh
```

**The fourth digest moved and this office did not move it.** `scripts/iteration_timer.sh` went
`04a7d712…` → `7c1e0481…` at mtime **16:40:03**, mid-audit, when SATURN's strike-2 repair landed
(§4.1). **Named rather than smoothed:** a digest table that quietly published the post-repair hash
under a pre-repair reading would be exactly `C31`'s class. The other three are unchanged from
pre-audit.

`[RUN]` `git status --porcelain`, tracked, verbatim and complete:

```
 M MISTAKES.md
 M house-events.jsonl
 M pytest.ini
 M scale/ledger.py
```

**All four are concurrent it.21 paths, none of them this office's.** `house-events.jsonl` is the
append-only event log every agent writes; `MISTAKES.md`, `pytest.ini` and `scale/ledger.py` are under
active write by the offices repairing this iteration's findings. 118 untracked paths at close (116 at open), all round
reports, plus this one. **No git writes. Nothing touched Kaggle.**

---

## 7. RULING — IS THE COORDINATOR'S CORRECTION MECHANISM RELIABLE?

**Not yet, and the evidence is arithmetic rather than rhetorical.**

Thirty-two corrections filed. **Struck: `C24` (its own subtraction wrong), `C26` (an undercount inside
the correction fixing an undercount), `C28` (still short by one, and the sixth site was in its own
file), `CORRECTION 30`, `CORRECTION 31` (three numbers, one measurably false), and now
`CORRECTION 32`'s own missing index row.** The corrections that correct corrections are `C24`, `C25`,
`C26`, `C28`, `C32` — **five of thirty-two, and every one of them corrects a correction filed one to
two iterations earlier.** A mechanism whose sixth-most-recent output needed correcting by its most
recent output is not converging on its own.

**But the diagnosis "introduces errors at the rate it removes them" is the wrong reading, and the
distinction matters more than the count.**

- **Every one of the five self-corrections of corrections was caught.** `C24`, `C25`, `C26`, `C28` by
  SATURN on SATURN. `C32` by MARS. §1.5 by **SATURN's own node, firing red, unattended, before any
  agent looked.**
- **The index digest is byte-exact in both published states**, recomputed here with the index's own
  command, and MARS confirmed it independently at both. **The one instrument the round holds up as
  clean is clean.**
- **Where the mechanism has an instrument, it is reliable. Where it has only prose, it is not.**
  `C31` was prose reading a `stat`. §1.4's unscoped regex is an instrument with a prose boundary.
  §2.1's *"verified correct at this census"* is prose inside an instrument. **Every failure this
  audit found is at a prose seam; every seam that carries a node held.**

**The ruling.** The correction mechanism is **reliable as an instrument and unreliable as a habit.**
The error rate is not in the corrections — it is in the *filing* of corrections from memory and from
`stat` under a closing clock, which is the identical mechanism as `MISTAKES.md` `P-3` and `V-7` and
which this round has now catalogued six times without changing the behaviour that produces it.
**Two iterations of evidence say the fix is not more corrections. It is that a correction may not be
filed without a `[RUN]` behind each of its numbers** — the standard this office is held to, applied
to the office that sets it.

**Two audits, opposite verdicts, and that is the point.** it.18 struck four and landed no repair.
This one strikes three and upholds six, including two planted negatives that do exactly what their
authors claimed and one instrument that caught its author unprompted. **An audit that only strikes is
as unbalanced as one that only clears, and the round is better instrumented today than it was
two iterations ago.**

---

## 8. WHAT THIS AUDIT DID NOT REACH

1. **The eight-occurrence claim, recounted independently.** Conceded by its author; unverified here.
2. **MARS's two self-corrections** — the `Popen` Win32 pid and the claim-id numerator — re-run
   individually rather than via SATURN's corroboration node.
3. **The 3 red nodes in `tests/jupiter/`** diagnosed to a cause. Two are the superseded it.18 landing
   instrument, one is `test_v20_r15_it4_merge_is_unexercised.py:94`. **Named, not diagnosed.**
4. **`_owns_watchdog`'s body.** The repair is verified present at the `kill` call site (§4.1); the
   ownership predicate itself was not opened and carries no planted negative for a recycled pid.
5. **The `[RUN]` floor of 43, not defended.** Named as a regression in §5 rather than excused.
