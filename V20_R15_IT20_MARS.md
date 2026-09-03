# V20 R15 — it.20 — MARS (ADVERSARY)

Three it.19 repairs attacked, three struck. Each strike carries its RED as an
executable event against unmutated shipped code, and each ships a replacement route.
A fourth attack, on the coordinator, returned documentary evidence and no RED; it is
filed to **OPEN**, not to the strike count.

Machine: the author's Windows 11 box, Git-for-Windows `bash`, `python -m pytest`.
Tree: branch `v17k-gate0`, working tree at the it.20 dispatch, SATURN and JUPITER
both live in it.

---

## STRIKE 1 — CLAUSE (c) IS NOT A WITNESS OUTSIDE CLAUSE (a)

### What the instrument does

`tests/saturn/test_v20_r15_freeze_manifest.py::test_every_wings_arm_is_corroborated_outside_clause_a`
derives the wing's arm from clause (a)'s module stem and requires clause (c) — the
accepted kill — to name it. SATURN priced it in his own Limits: *"It would not catch
a manifest in which clause (a) and clause (c) were moved together. The third would be
clause (d)'s price, and it is not asserted here."*

The price is understated by one clause and wrong in kind.

**(a)+(c) alone does not survive** — and not because of the new witness. `_arm_of`
reads clause (a); `test_every_clause_b_line_journals_that_wings_own_arm` then checks
clause (b) against it. Move (a) without (b) and the joint node fires. The manifest
looks defended against the two-clause move for a reason that is not the repair.

**(a)+(b)+(c) moved together survives everything the freeze file ships.**

### RED — `t`, before the finding

```
$ python -m pytest tests/mars_v20/test_it20_two_clauses_is_not_two_witnesses.py -q
FAILED ::test_the_new_second_witness_does_not_catch_the_three_clause_swap
E   AssertionError: W1 is certified `arm_pl` by clauses (a), (b) and (c) at once,
E   while its clause (d) still cites V17_R4_RETAKE_PRICE.md:194 -- the 15.970
E   SMPRIME price -- and not one shipped node fires. Clause (c) is not a witness
E   OUTSIDE clause (a); it is a third seat in the same closed loop.
E   assert []
1 failed, 4 passed in 1.20s
```

`assert []` is the whole finding: the fired list is empty. The four that pass are the
calibration (`the_true_rows_pass_every_shipped_node`), the V-26 precondition, the
harness self-guard, and the two-clause control — all asserted **before** the
falsification is claimed. The harness reads each node's **own** `parametrize` list
off `pytestmark` and never invents arguments, which is the it.19 failure this office
was handed.

Confirmed independently on disk, manifest mutated then reverted:

```
$ python /tmp/mars_swap.py a b c    # swap W1<->W3 clauses a,b,c; re-freeze the sha
swapped clauses: ['a','b','c'] -> FREEZE-SHA256 = 9a9ae0a9c42897bc...d7e517a0
$ python -m pytest tests/saturn/test_v20_r15_freeze_manifest.py -q
20 passed in 0.52s
```

**20 of 20 GREEN**, the new witness included, on a manifest that reads:

```
W1|a|ceq/arm_pl.py:88|def scan(g: torch.Tensor)
W1|b|results/v17k_r4_retake.jsonl:25|0.6446726192
W1|c|V15_LEDGER.md:786|ARM PL with the parity claim RETIRED
W1|d|V17_R4_RETAKE_PRICE.md:194|15.970          <- the SMPRIME price, unmoved
```

The mutation is V-26 marginal-preserving: 8 rows, 2 wings, the same
`(wing, clause)` pairs, the same multiset of citations and anchors, every anchor
still on its exact cited line, `FROZEN-N = 2`. The declared `FREEZE-SHA256` is
**recomputed**, because freezing is tamper-evidence against an edit made *after* the
freeze and says nothing about a wrong freeze; leaving the stale digest in place would
strike the wrong instrument.

### The one thing in the repo that does catch it

`WING_ARM = {"W1": "arm_smprime", "W3": "arm_pl"}` at
`tests/saturn/test_v20_r15_it14_saturn.py:39` — **the hardcoded map the it.14 repair
deleted from the manifest test** on the reasoning that it "would be a second place for
the manifest to be wrong". Measured over the four wing-bearing files, baseline 3
standing REDs, under the swap 6:

```
+ it14_saturn::test_the_clause_b_pairing_is_true_on_the_unmutated_tree
+ it14_saturn::test_the_it4_freeze_is_two_marginals_and_stays_green_under_the_joint_break
+ mars_v20/test_it18_wing_identity_is_self_certified::test_the_clause_b_repair_does_not_catch_the_identity_swap
        (double-swap artifact of a pre-mutated manifest -- NOT a catch)
```

Both genuine catches read `WING_ARM`. The first is a **calibration** node whose own
docstring says it exists so "the strike below would [not] be a bug report about the
data" — it fires with a message about the ledger, not about wing identity. A reader
would file it as data drift and re-pin the citation.

### Replacement route — REPRICE, then REROUTE

1. **Reprice now, one line.** Extend the it.19 witness to clause (d). The price rows
   name their arm verbatim: `V17_R4_RETAKE_PRICE.md:194` is
   `` | `arm_smprime` | **15.970** | 17.839, 14.101 | `` and `:195` is
   `` | `arm_pl` | **1.614** | 1.622, 1.605 | ``. Same `_norm` substring test,
   `"c"` → `("c", "d")`. Cost: the move goes from 3 clauses to 4, in a third file.
2. **Reroute, because four clauses of the same kind is still one loop.** The binding
   `W1 = arm_smprime` must come from **outside the manifest**, and it already exists:
   import `WING_ARM` from `tests/saturn/test_v20_r15_it14_saturn.py` and assert
   `_arm_of(w) == WING_ARM[w]`. One import, one assert, no new artifact. The it.14
   reasoning is inverted — a second place that can disagree is the only place a lie
   can be caught, and this repo already pays for that second place.

Not speculative: route 2 is the mutation's own detector, promoted from a calibration
premise in a neighbouring file to a guard in the file that certifies the manifest.

---

## STRIKE 2 — THE RETIREMENT COVERED THE CHANNEL WITH NO WRITER AND LEFT THE ONE THAT HAS ONE

### What the instrument does

REPAIR 1 retired `$PIDFILE` because "a check nobody feeds launders unknown as
confirmed". Correct, verified, and incomplete. `scripts/iteration_timer.sh` holds a
**second** pid channel — `$WATCHDOG` = `.claude/iteration.watchdog` — and it differs
in the way that matters: **it has a writer.** `echo $! > "$WATCHDOG"`, the `start`
branch, line 97.

`disarm()` reads it and issues `kill "$wpid"` with no liveness, ownership or
start-time check. And `rm -f "$WATCHDOG"` occurs **exactly once in the whole script**,
inside `disarm` itself — the detached watchdog body never clears its own
registration. A watchdog that runs to completion therefore leaves a file naming a
dead pid, and the next `start` or `stop` signals whatever the OS has since given that
number to.

This is the it.18 defect with the arms reversed. There, a channel with no writer
answered in the **permissive** value. Here, a channel with a writer answers in the
**destructive** value: an unsourceable process fact acted on with a signal.

### RED — `t`, before the finding

```
$ python -m pytest tests/mars_v20/test_it20_the_retired_channel_left_a_live_one.py -q
FAILED ::test_disarm_kills_a_pid_it_never_verified
E   AssertionError: `stop` sent SIGTERM to pid 40376 on the strength of
E   .claude/iteration.watchdog alone. The file names a pid; it does not establish
E   that the pid is still the watchdog that was armed, and nothing clears it when
E   the watchdog exits on its own. REPAIR 1 retired $PIDFILE for reading a process
E   fact it could not source. $WATCHDOG reads the same unsourced fact and acts on
E   it with a signal, and unlike $PIDFILE it has a writer, so it is live today.
E   assert False
1 failed, 2 passed in 1.89s
```

The two that pass are the premises, asserted before the falsification: `$WATCHDOG`
has a writer and `$PIDFILE` does not, and the watchdog body does not touch its own
registration. The shipped script is copied **unmodified** into a scratch tree; a
`sleep` is registered in place of a stale watchdog pid; `stop` kills it. Reproduced
on the command line before the node existed: `RESULT: victim2 WAS KILLED by disarm`.

One correction against this office's own first draft. The victim was initially
spawned with `subprocess.Popen`, whose pid is a Win32 pid that Git-Bash `kill` cannot
address. **The node passed** — measuring the pid-namespace gap, not the defect. It now
spawns the victim through `bash` and reads back the shell's own `$!`, which is the pid
`disarm` actually signals. A guard that passes for the wrong reason is the failure
this office was handed at it.19; it was in my own harness first.

### Replacement route — REROUTE, four lines, no new artifact

The watchdog channel is worth keeping: unlike `$PIDFILE` it is fed, and it is the only
thing that enforces the cap. Two changes, both in `scripts/iteration_timer.sh`:

1. Give the detached watchdog body `trap 'rm -f "$8"' EXIT` on its own registration
   path, so a completed **or** killed watchdog deregisters itself.
2. In `disarm`, corroborate before signalling. Write `"$NOW $!"` at `start`; refuse to
   signal when the recorded stamp predates `$STATE`'s `start=`, and require
   `kill -0 "$wpid"` first. When it cannot corroborate, say so in the output and do
   not kill.

The predicate is the mirror of REPAIR 1's own `ponytail:` note: never signal on
*file exists*, only on *registered AND corroborated*.

---

## STRIKE 3 — THE CENSUS STILL CERTIFIES ITS OWN COVERAGE

### What the instrument does

The landing instrument is the pair `tests/jupiter/test_v20_r15_it17_citation_landing.py`
+ `…it18_citation_landing.py`. it.19 shipped no successor —
`test_v20_r15_it19_q2_rows_and_m14.py` opens no citation.

> **CORRECTION, filed against this section before it left the office.** JUPITER
> landed `tests/jupiter/test_v20_r15_it20_citation_freeze.py` **while this attack
> ran** — it was absent from the tree at dispatch and present at filing. It declares
> `CENSUS` of **103** entries, `POPULATION_AT_IT20 = 129`, `UNIQUE_AT_IT20 = 103`,
> and runs **10 passed**. On its own declaration the ratio this office was sent to
> attack is **103/103**, not 30/103. **This office did not independently open a
> single one of his newly-covered citations, and files no verdict on them.** The RED
> below therefore measures the it.17/it.18 pair only, and its finding — that the
> *old* pair never asserted its own coverage — stands as history, not as a live
> price on JUPITER's number. What survives as live work for it.21 is named at the
> foot of this section.

**Neither file names coverage.** `POPULATION_AT_IT18 = 129` is compared only against
itself (`len(cites) == POPULATION_AT_IT18`). it.17's sensitivity probe builds
`shifted` **from** `MANIFEST` and compares `len(named)` **to** `len(MANIFEST)` — a
legitimate line-sensitivity calibration that is arithmetically incapable of failing
for a coverage reason. The two numbers live in different files and are never brought
into contact.

### RED — `t`, before the finding

```
$ python -m pytest tests/mars_v20/test_it20_the_census_still_certifies_its_own_coverage.py -q
FAILED ::test_the_landing_instrument_opens_most_of_the_census
E   AssertionError: 30 of 103 distinct citations (30/129 occurrences) are
E   individually opened -- 29.1%. The remaining 73 have never been read at the line
E   they name by anything. Nothing in the repo asserts this ratio: POPULATION_AT_IT18
E   is compared only against itself, and it.17's +10 shift probe is built from
E   MANIFEST and compared to len(MANIFEST). `30 >= 123` is now `30 >= 129`.
E   assert 0.2912621359223301 >= 0.9
1 failed, 2 passed in 0.56s
```

The two that pass are the premise (neither instrument file mentions coverage) and the
calibration (every opened citation is a real member of the census — the numerator
carries no phantoms). One correction in this office's own draft: the numerator first
unioned `IT18.REPAIRS` directly, which is keyed by **claim id** (`C17`, `C64`, `C103`),
not by `path:line`. That inflated the count to 33, and **the calibration node caught
it** — the numerator now pulls `path:line` out of the record shape, and the three
it.18 repairs prove to be a **subset** of it.17's 30.

**The numerator did not move at it.19. The denominator did: 123 → 129.**

### The instrument is RED at HEAD on a failure it did not declare

```
$ python -m pytest tests/jupiter/test_v20_r15_it1{7,8}_citation_landing.py -q
FAILED it18::test_every_true_location_carries_what_the_table_claims
E   AssertionError: C17: V20_R15_JOURNAL.md:650 lacks 'THE FREEZE'
```

`## it.4 — THE FREEZE` is now at journal line **669**; the M14-at-F1 line at **664**.
Both are **+19** — a rigid displacement of the whole journal, from an append above
line 645: the `C20` CORRECTIONS INDEX row now at `V20_R15_JOURNAL.md:56`, added at
it.19. Third instance of the mechanism it.17 already named as
`RIGID_SHIFTS = {CONSTS: +10, SAT: -23}`. The instrument caught **one of the two**
journal citations, because `:650` happened to be in `REPAIRS` and `:645` did not —
while `test_v20_r15_it19_q2_rows_and_m14.py:47` asserts outright that *"nothing in
this file cites `:645`"* and the theory table cites `:645`.

### Sampled misses — uncovered citations, opened by hand this iteration

| citation | what the table claims | what is on that line | verdict |
|---|---|---|---|
| `V20_R15_JOURNAL.md:645` | "`F1` in the journal" | `unbound) and \`>= 2\` (SATURN, measured on the corrected gate…` | **MISS** (+19 drift) |
| `scale/m3_flops.py:207` | one of "the three shipped cost models" | `CELLS = ("softmax","glance","settled","twin","argmax")` | **MISS** — a name tuple; the models are at `:150`/`:154` |
| `CEQ_V20_R15_CONTRACT.md:239` | "`F3` in the contract" | `M14 CHEEGER STRATIFICATION. φ²/2 ≤ 1−λ₂ ≤ 2φ [V] — corpus` | **MISS** — `F3` is at `:241`/`:243`; the journal cites `:239-243` correctly, the table narrowed it |
| `house-events.jsonl:12784` | "self-labelled a reconstruction" | `{"event":"finding_contract","finding":"no F0-F4 rubric exists anywhere…"}`, no `reconstruct*` on the line | **MISS** |
| `V20_R15_IT9_JUPITER.md:265` | "out-of-sample confirmation" | a results-table row; no verbatim anchor | AMBIGUOUS |
| `lean/CEQ/V16Domain.lean:105/129/304/75` | four symbol claims | all four present verbatim | LANDS ×4 |
| `scripts/v15_r1.py:137/547/586` | `S, D = 64, 24`; nine sibling flags; `floor1` | all present; `:547` heads exactly 9 `add_argument` calls | LANDS ×3 |
| `V20_R15_LEAP_LEDGER.md:24/25/28/70/99/130/131` | seven L-row claims | all seven verbatim | LANDS ×7 |

Four clear misses in a hand sample of the uncovered set. Every cited **file** exists;
no missing-file miss in the census.

### Replacement route — REPRICE, plus one free coverage lift, plus a retirement

1. **Assert the ratio.** The shipped RED is the route: one node comparing
   `len(opened)` to `len(census)` against a declared floor. A census whose coverage
   is unstated cannot be argued about. Set the floor where the office will defend it
   and let it fail until it is met — *that* is the number, not `129`.
2. **The 34 machine-checkable citations are free.** The table's
   `` `SYMBOL` @ `path:line` `` form needs no hand manifest: derive the anchor from
   the backticked symbol and open the line. All 34 land today, so this is coverage at
   zero repair cost, and it grows with the table instead of going stale.
3. **Retire the drift class; do not re-pin it.** Three rigid shifts in three
   iterations is a mechanism, not an accident. For `V20_R15_*` prose the citation
   should carry a **marker string**, not a line number — the route this repo's own
   commit `df8892f` already took for the code snapshot ("find it by walking for its
   marker, not by naming its mount"). It has not been applied to the table, and
   JUPITER's it.20 `CENSUS` re-pins line numbers rather than retiring them, so the
   next append above line 645 moves them again.

### THE LIVE WORK FOR IT.21, NAMED

JUPITER's 103/103 is a **declaration by a hand-written `CENSUS` dict, checked against
the table it is drawn from**. That is the shape this office struck at it.9 and it.18.
Three specific questions, none answered here:

- Does `CENSUS` open each of the 103 at the line it names, or does it assert presence
  as a string? `test_v20_r15_it20_citation_freeze.py` carries a `WANT_SEAL` and
  `LIVE_FILES` of size **3** — if only 3 of 37 cited files are opened live, the other
  100 entries are sealed against a snapshot, not resolved at HEAD.
- Do the four misses this office opened by hand — `V20_R15_JOURNAL.md:645`,
  `scale/m3_flops.py:207`, `CEQ_V20_R15_CONTRACT.md:239`,
  `house-events.jsonl:12784` — appear in `CENSUS` with anchors that land, or with
  anchors weakened until they do? His suite is green; the question is which.
- `POPULATION_AT_IT20 = 129` is still compared to `len(cites)` drawn from the same
  table. The coverage floor this office's RED asserts is still asserted nowhere.

---

## OPEN — THE COORDINATOR. NO RED, SO NOT A STRIKE

Four documentary findings, measured, filed OPEN because none is backed by a runnable
RED inside the wall clock. **The record moved under this audit** — `V20_R15_JOURNAL.md`
went from 3,890 lines / a 20-row index to 3,908 lines / a 31-row index at
`mtime 16:20:35`, `sha256 09b4f316…0403e`. All citations below are the later state.

**O-1. The CORRECTIONS INDEX digest verifies — in both states — and the recipe is
unscoped.** Declared `1ec53020…e731be` over 20 rows recomputed **byte-exact**;
declared `8a886db5…62757` over **31** rows (C1–C31, contiguous, no gaps) recomputed
byte-exact. The brief's "20 rows" was true at dispatch and is now stale; the
instrument is sound. **But** the stated recipe's pattern `^\| C\d+ \|` is unscoped to
the index block — it hashes any matching line in all 3,908. The it.19 body already
discusses `C64`, `C98`, `C103`, and `V20_R15_IT1_INSPECTOR.md:315` already writes a
body row of that shape. The first `| C64 | …` row written into the journal body joins
the digest silently, moving it with no edit to the index. The round's own class,
latent inside the one instrument it holds up as clean.

**O-2. Correction 30's quote is exact; its provenance is not.** `MISTAKES.md:122-125`
matches the block quote word for word. But *"quoted `V-7` at three offices during
it.18"* cannot be sourced: `grep -c "V-7"` over the four it.18 office files returns
`0 0 0 0`, and the it.18 journal section returns zero. Three offices did quote V-7 —
**at it.1** (`IT1_INSPECTOR.md:214`, `IT1_MARS.md:338`, `IT1_SATURN.md:90`). The count
of three is right; the iteration is off by seventeen. And the correction calls the
witness-raise *"strictly stronger"* while stopping one sentence short of
`MISTAKES.md:125-129`, which records that the witness mechanism **itself shipped with
two bugs, the worse one in exactly this class** — *"a witness that cannot fail on a
wrong selector is precisely the disease it was written to cure."*

**O-3. Correction 31's diagnosis is sound and its three numbers cannot be sourced.**
`14m45s`, `15m10s`, `14m10s` and `36,786` occur **nowhere in the repo except the
sentence that asserts them**; `house-events.jsonl` carries no `ts` after
`2026-09-02T04:43:41` and no it.19 INSPECTOR filing event. Against the only checkable
artifact: `V20_R15_IT18_INSPECTOR.md` mtime is `16:12:15` = **13m09s**, not 14m45s,
and the file is **38,271 bytes**, not 36,786 — so a stat at 14m10s returns 38,271. The
headline "25 seconds earlier" is the difference of two unsourceable numbers. **A
correction about asserting state from a proxy is itself resting on unwitnessed
proxies.** Its index row at `:67` also misattributes it to *"SATURN, on itself"* when
the body says *"This office spliced"*.

**O-4. Four uncorrected claims of correction 31's own shape stand in the it.19
entry.** `:3488` asserts a nurse was *"still writing"* from an **mtime** — six lines
after diagnosing the class, and it is the premise that dismisses the first
static-analyzer diagnostic. `:3518` admits a reading because *"mtime was identical
before and after the run"*, the weaker instrument the INSPECTOR had already refused in
the same hour in favour of an MD5 (`V20_R15_IT18_INSPECTOR.md:589`). `:3468` reads
`grep -c "re-arm with --force" → 0` as *the licence no longer exists* — V-7 verbatim,
unrepaired by the very repair correction 30 proposes. `:3898` reports tree state as
`git status --porcelain = 113`, a line count.

**O-5. The eight-occurrence claim is LOW, and unreconstructible from its own
numbering.** Enumerated inside `:3443-3908`, this office counts **10** in-class
occurrences; on the narrowest reading (the 7 the record itself places in the class) it
is 7 and the claim is high by one. Neither route reaches 8. The record's own numbers
are 3, 6, 7, 8 — **skipping 4 and 5**, printed **out of order** (*"seventh"* 52 lines
before *"Sixth"*), and resting on three mutually incompatible enumerations of "the
five" at `:3529`, `:3565` and `:3620`, of which the first is explicitly
round-cumulative (*"four of those five predate it.19"*) and the second restates the
same list as iteration-local. The paragraph contradicts itself two lines later:
`:3828` says *"eighth"*, `:3834` says *"six other instruments"* — six others plus
itself is seven.

**Route for all five: retire the running tally.** A hand-maintained ordinal that
skips numbers, prints out of order and draws on three incompatible base sets is an
instrument that cannot observe the property it reports — the class it is counting.
Replace it with a derived count: tag each in-class paragraph with a literal marker and
let a node count the markers, the way the CORRECTIONS INDEX digest is derived rather
than declared. And scope that digest's regex to the index block before the first body
`| C64 |` row lands.

---

## TREE, AND THE REVERT

Two mutations were applied; both were reverted.

| target | mutation | reverted | proof |
|---|---|---|---|
| `V20_R15_WING_MANIFEST.md` | W1↔W3 clauses (a,b,c), `FREEZE-SHA256` recomputed | yes | `c0439c156119a43a033165589ebc9f04241041d887cbd513b2b3bdb52d196137` — byte-identical to the pre-mutation digest |
| scratch tree under `mktemp -d` | copy of `iteration_timer.sh`, a `sleep` victim | n/a | removed; no repo file touched |

`scripts/iteration_timer.sh` was **never** edited — the STRIKE 2 RED copies it into a
scratch tree. Its digest is unchanged at
`5dc094f1ddafd83005609eacf739ef73c50dff895ee9a408f109164962fa42da`.

Three files were added, all under `tests/mars_v20/`:

```
tests/mars_v20/test_it20_two_clauses_is_not_two_witnesses.py
tests/mars_v20/test_it20_the_retired_channel_left_a_live_one.py
tests/mars_v20/test_it20_the_census_still_certifies_its_own_coverage.py
```

No file outside `tests/mars_v20/` and this record was written. **No `git add`, no
`git commit`, no git write of any kind. Nothing touched Kaggle.** The
`git status --porcelain` at filing carries concurrent it.20 paths belonging to SATURN
and JUPITER, which this office did not author and did not touch.

---

## LIMITS

STRIKE 1's replacement route 2 makes `tests/saturn/test_v20_r15_it14_saturn.py` a
dependency of the freeze test; if that file is ever deleted the witness goes with it,
so the route should be read as "the binding lives outside the manifest", not "the
binding lives in that file". STRIKE 2's RED is measured only on Git-for-Windows and
turns on `kill` reaching an MSYS pid; the defect is in the script's logic and is
platform-independent, but the *demonstration* is not, and it passed for a wrong reason
before that was fixed. STRIKE 3's 30/103 counts **distinct** `path:line` pairs; by
occurrence it is 30/129, and the two differ because 26 citations repeat. The four
sampled misses are four of 73 uncovered citations — a sample, not a census of the
misses, and no claim is made about the other 69. The rigid-shift finding names `C20`
at `V20_R15_JOURNAL.md:56` as the displacing append on an arithmetic match (+19 in two
places) and a date, not on a `git log -S`, which SATURN established at it.19 is
structurally incapable over this round's untracked prose. **The five OPEN items carry
no RED and are not strikes**; O-3's contradictions rest on filesystem mtimes and sizes,
which is the same proxy class the item is about, and they are offered as *the numbers
cannot be sourced*, not as *the office is wrong about what happened*. None of the
three REDs requires a GPU, and none was measured on a machine other than the author's
box.
