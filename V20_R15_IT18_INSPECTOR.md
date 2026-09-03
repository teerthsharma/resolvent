# V20 R15 it.17-it.18 — HEALTH INSPECTOR LEDGER

Filed: 2026-09-02. Branch `v17k-gate0`. Wall clock start 15:59 IST, wall 20 min.
Eleventh inspector report. Skeleton filed first; sections filled as nurse evidence landed.

## VERDICT LINE

**14 audited, 4 struck.** Every strike is against a *claim or an instrument*, none
against a repair that is live in the tree.

| # | subject | ruling |
|---|---|---|
| 1 | `J-18a` - census is a dated measurement, `128 -> 129` | **UPHELD**, and **UNDER-CLAIMED** - it broke a pointer during this audit. See LIVE. |
| 2 | the planted negative fires on a still-resolving wrong line | **UPHELD**, and it is the suite's **only** load-bearing discriminator: blinding the checker fails 1 of 6. |
| 3 | the 28 + 3 repairs carry what their cells claim | **UPHELD** on a 6-of-129 sample. `28 claimed vs 25 in the it.17 manifest` **OPEN**. |
| 4 | `J-18f` - C98/C103, repair aimed at the wrong claim | **UPHELD**, and already reversed by JUPITER at it.18. Nothing live carries it. |
| 5 | this office's `8-in-20` | **STRUCK** - not for bias (`P = 0.064`; no bias needed), but because **the twenty are not itemised to twenty**. The eight misses survive; the rate does not. |
| 6 | MERCURY's predicate twin, indistinguishable on 40 cells | **UPHELD** - byte-identical output, separated only by the plant. |
| 7 | MERCURY's self-charge, *"0.697 measures -0.1224"* | **STRUCK** - two assertions, two populations, **both reproduce exactly**. Real defect is an unnamed operand at `:111`. |
| 8 | MARS strike 1 - the heartbeat corroborates against nothing | **UPHELD IN FULL**, and **already rerouted**: the channel is retired, `3 failed -> 3 passed`. His charge understates it - the predicate was `exists dead`, so one finished nurse of four licensed the verdict. |
| 9 | MARS strike 2 - the (a)+(b) swap leaves 0 of 7 firing | **UPHELD** - 0 of 7 confirmed under correct arguments. SATURN shipped the repair at it.19 mid-audit. |
| 10 | MARS's own strike-2 regression test | **STRUCK** - it reports `2 passed` while the defect stands, because SATURN re-parametrized the node from arm names to wing IDs and MARS's args went stale. **A test that went green for a reason unrelated to the bug.** |
| 11 | MARS strike 3 - *"re-scoped"* is the wrong verb | **UPHELD. `withdraw` is correct.** `bad_citations` reads `len(lines)` and never indexes a line; **0 of 26** withdrawn pointers named (MARS said 25 - one stronger than he claimed). |
| 12 | Correction 29 - two briefs on stale readings | **UPHELD.** No surviving claim rests on a stale reading. Its own `21,774` has since drifted to `21,897` - filed, not struck. |
| 13 | the CORRECTIONS INDEX | **STRUCK** - stops at C19 while the body runs to C29. The round's designated guard against stale entries **fails silently in the trusting direction.** |
| 14 | the loop file after the `--force` removal | **INTACT.** All four keys present, timer runs, parser reads only `iteration:`. Licence withdrawn, capability retained at `iteration_timer.sh:72`. |

**Two things this iteration got right that outrank any single strike.** MARS's three
strikes were **all correct**, and **all three were already repaired at HEAD by SATURN's
it.19 work while this audit ran** - the round is now closing defects faster than it
audits them. And the two mutations this office applied are reverted with matching
digests.

### THE RULING ASKED FOR

**A landing check CANNOT detect a repair aimed at the wrong claim - not "does not",
cannot.** The checker tests `want in line_at(path, lineno)`, where `want` is supplied by
the repairer **in the same edit as the pointer**. A wrong-claim repair moves both
together, so the pair is consistent with the file and **lands as cleanly as a correct
repair - because against the only oracle the checker has, it IS one.** The manifest is
the ground truth, and the manifest is written by the office under audit.

**This is the same mechanism as MARS's strike 2** - *"the clause-(b) repair certifies the
manifest against itself"* - reached independently, in the same iteration, in a different
subsystem, by an office that never saw the other's work. **Two offices found one defect:
an instrument whose ground truth is authored by the party it audits.** That convergence
is the iteration's structural result and it outranks every individual strike above.

**The reroute:** freeze `want` at census time; a repair that alters `want` is not a
repair but a **withdrawal plus a new citation**. That is the only change that converts an
undetectable class into a bookkeeping one.

## P1 — THE CENSUS THAT MOVED (123 -> 28 -> 128 -> 3 -> 129) — **AUDITED, UPHELD**

**Both suites green at HEAD, unmutated, run by this office:**

```
[RUN] python -m pytest tests/jupiter/test_v20_r15_it17_citation_landing.py                       tests/jupiter/test_v20_r15_it18_citation_landing.py -v
   ->  12 passed in 0.30s        (it.17: 6 nodes, it.18: 6 nodes)
```

**The population is re-derived, not asserted.** `POPULATION_AT_IT18 = 129` at
`tests/jupiter/test_v20_r15_it18_citation_landing.py:76`; an independent regex sweep of
the frozen table by this office's nurse returns **129**. The `128 -> 129` move is real
and JUPITER's `J-18a` is correct on its own terms: **his node went RED on a number he
had published one iteration earlier, and he filed the mechanism rather than the
embarrassment.** A census count is a dated measurement. UPHELD.

### 1.1 THE PLANTED NEGATIVE — IT FIRES, AND IT IS THE ONLY THING THAT DOES

The planted negative shifts `CEQ_V20_R15_CONTRACT.md:123` to `:133` — a line that
**still resolves** in a file that **still exists** — and requires the checker to name it.
That is exactly the shape the it.14 checker cannot see.

**A nurse mutation that commented the assertion out is not a test of it** — deleting an
assertion trivially passes and proves nothing. This office ran the discriminating
mutation instead: **blind the checker to line content**, turning it back into an
it.14-shaped resolve-only checker.

```
[RUN] sed -i 's/elif want not in line_at(path, lineno):/elif False:/'         tests/jupiter/test_v20_r15_it18_citation_landing.py ; python -m pytest ... -q
   ->  FAILED test_the_checker_fires_on_a_wrong_line_inside_a_file_that_exists
   ->  E  AssertionError: the checker missed a wrong line
   ->  E  assert [] != []
   ->  1 failed, 5 passed in 0.56s
[RUN] md5sum tests/jupiter/test_v20_r15_it18_citation_landing.py   (after revert)
   ->  f846c839476e5bcd6e92d1759241c642    == pre-mutation digest. REVERTED CLEAN.
```

**`1 failed, 5 passed` is the whole finding.** The planted negative is load-bearing and
it discriminates precisely the property it claims. **And five of the other six nodes —
including `test_every_true_location_carries_what_the_table_claims`, the census itself —
stay GREEN against a checker that never opens a line.**

**RULING: the planted negative is UPHELD and is the suite's only load-bearing
discriminator.** This is not a criticism of the other five; it is a structural fact.
**A census run over a table whose pointers are correct cannot detect its own blindness**
— every entry passes under either standard. Only a deliberately wrong entry separates
them. **JUPITER built the one node that could fail, and it is the one node that did.**

### 1.2 THE REPAIRS, SAMPLED — 6 OF 129, ALL LAND

Sampled across the table (first, middle, last): `V20_R15_WING_MANIFEST.md:3`,
`CEQ_V20_R15_CONTRACT.md:58`, `ceq/arm_smprime.py:409` (`m["smp_values"] = values`),
`scripts/v15_r1.py:804` (`sign_acc_0step`), `tests/jupiter/test_v20_r15_it8_q4_q5.py:222`
(`assert "def zero_hop_mask" in src`), `tests/jupiter/test_v20_r15_it14_theory_table.py:1`.
**Six of six carry what the cell claims.** No strike.

### 1.3 ONE ARITHMETIC GAP, OPEN

it.17 reports **28 repairs**; its `MANIFEST` dict at
`tests/jupiter/test_v20_r15_it17_citation_landing.py:40-82` holds **25 entries**.
**Three repairs are claimed in prose and absent from the instrument that certifies them.**
This is the it.17 side of the same `28 vs 25` bookkeeping that MARS's strike 3 reaches
from the withdrawal direction. **OPEN — not struck, because the sampling above found no
bad pointer and the gap may be grouping rather than omission.** It is named so it cannot
be lost.


## P1b — THE INSPECTOR'S OWN `8-in-20`: WITHDRAWN, AND ON A BETTER GROUND THAN BIAS

**The charge as put:** the `8-in-20` projects to 49 where the census found 28; check
whether the twenty were a biased sample, because if they were, the rate should be
withdrawn rather than superseded.

**First, a correction to the charge, in this office's disfavour on one point and its
favour on another.**

*Favour:* **no 49 was ever published.** The only `49` in this office's reports is
`V20_R15_IT13_INSPECTOR.md:347`, an unrelated line-count refutation.

```
[RUN] grep -rn "49" V20_R15_IT16_INSPECTOR.md V20_R15_IT13_INSPECTOR.md
   ->  IT13:347  **"The block is 49 lines": REFUTED. It is 59.**   (a different subject)
   ->  IT16      no hit
```

The rate was labelled a rate at both sites where it appeared:
`V20_R15_IT13_INSPECTOR.md:917` — *"Eight misses in twenty is a rate, not a census"* —
and `V20_R15_IT16_INSPECTOR.md:450` — *"on the sample drawn so far ... may move in
either direction."* **The 49 is the brief's arithmetic, not this office's claim.**

*Disfavour, and it is worse than bias:* **the twenty cannot be reconstructed from the
published itemisation, so the sample cannot be tested for bias at all.**

`V20_R15_IT13_INSPECTOR.md:758` states `Twelve land exactly. EIGHT DO NOT.` The eight
are itemised in a table, one row each — that side is clean and reconstructible. The
twelve are itemised as prose at `:760-764`, and **the prose lists eighteen distinct
`path:line` sites**:

```
[RUN] sed -n '758,765p' V20_R15_IT13_INSPECTOR.md
   -> v15_r1.py:137, :586, :249, :267, :383, :384, :386      (7 sites)
   -> ceq/kdata.py:475                                        (1)
   -> CEQ_V20_R15_CONTRACT.md:58                              (1)
   -> lean/CEQ/V16Domain.lean:105/129/165/304                 (4)
   -> ceq/arm_smprime.py:559, :572                            (2)
   -> ceq/arm_pl.py:93, IT8_JUPITER.md:473, IT13_MERCURY.md:63 (3)
                                                       TOTAL: 18 sites
```

18 exact sites + 8 missing = **26**, against a stated denominator of **20**. The prose
groups some sites into one citation (*"both timer sites"*, *"all four Lean
declarations"*, `:383/384/386`), but **no consistent grouping rule yields twelve.**
Maximal grouping yields 11; no grouping yields 18. **Twelve is reachable from neither.**

**RULING — `8-in-20` is WITHDRAWN, on defective itemisation, not on bias.**

This is the stronger ground and it is against this office. A sample whose membership
cannot be recovered from its own report is not a biased sample — it is **not a sample**.
Bias is a property you can only charge against a draw you can enumerate. This office
published a denominator it did not itemise, and the numerator's eight rows are the only
part of the finding that survives.

**What survives, and it is not nothing:** the eight itemised misses each carry the
cited target, the actual content found, and the true line. **All eight are real defects
and none is withdrawn** — the census confirmed the class. What is withdrawn is the
**rate** `8/20 = 40%` and every inference drawn from its denominator.

**The bias test, run anyway, and it exonerates on that count.** Had the twenty been a
clean random draw, `8/20` against the census `28/123 = 22.8%` is not a significant
excess:

```
[RUN] python -c "from math import comb; p=28/123; n=20; print(sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(8,21)))"
   ->  0.064            expected bad in 20 = 4.55, observed 8
```

**P = 0.064.** The gap between the rate and the census **never required bias to explain
it.** So the brief's hypothesis is refuted on its own terms — and the rate falls anyway,
for a reason the brief did not reach. **Bias was the wrong charge; arithmetic was the
right one.**

## P2 - C98/C103, THE REPAIR AIMED AT THE WRONG CLAIM - **UPHELD, AND ALREADY REVERSED**

**The facts, verified at HEAD.**

it.17 moved C98 from `V20_R15_IT13_MERCURY.md:196` to `V20_R15_IT89_INSPECTOR.md:63`
(`V20_R15_IT17_JUPITER.md:75`). The target line reads:

```
[RUN] sed -n '63p' V20_R15_IT89_INSPECTOR.md
   -> | `NRMSE = sqrt2` | `1.4060346618513293` against `sqrt2 = 1.4142135623730951`, off ...
[RUN] grep -n BED_SPECS V20_R15_IT89_INSPECTOR.md      ->  (empty, 0 matches)
[RUN] grep -rn BED_SPECS --include=*.py .              ->  ceq/kdata.py:472  BED_SPECS = {
```

The repaired line **carries the number `1.4060346618513293`** the cell mentions. It
carries **no `BED_SPECS`** and **no `[RUN]` marker**. The claim the cell actually makes
lives at `V20_R15_IT13_MERCURY.md:189` - the `[RUN] python -c "from ceq import kdata;
print(list(kdata.BED_SPECS))"` row, in the office the citation started in.
**The original was wrong by 7 lines in the right file; the repair was wrong by a whole
office.** `J-18f` UPHELD.

**And JUPITER already reversed it, unprompted.**
`tests/jupiter/test_v20_r15_it17_citation_landing.py:66` - *"C98, RE-REPAIRED at it.18."*
`tests/jupiter/test_v20_r15_it18_citation_landing.py:39` now reads
`"C103": ("V20_R15_IT13_MERCURY.md", 189, "print(list(kdata.BED_SPECS))")`, with
`V20_R15_IT89_INSPECTOR.md:63` moved to the WITHDRAWN list at `:47`. **No live pointer
carries the defect. Nothing to strike; the repair of the repair is correct.**

### 2.1 THE RULING ASKED FOR - **NO. A LANDING CHECK CANNOT DETECT THIS CLASS.**

**Not "does not"; cannot.** The reason is structural and it is visible in the checker's
own signature:

```
tests/jupiter/test_v20_r15_it18_citation_landing.py:59-67
def not_landing(text, manifest: dict[str, tuple[str, int, str]]) -> list[str]:
    for cid, (path, lineno, want) in manifest.items():
        if f"`{path}:{lineno}`" not in text:      ...
        elif want not in line_at(path, lineno):   ...
```

**The checker compares the line against `want`. It never compares `want` against the
cell.** `want` is a substring supplied by the repairer, in the same edit as the pointer.
A repair aimed at the wrong claim changes the pointer *and* the `want` together - C98's
`want` became the number `1.4060346618513293` instead of the `BED_SPECS` run - and the
pair is then perfectly consistent with the file. **It lands as cleanly as a correct
repair because, against the only oracle the checker has, it IS a correct repair.**

**The manifest is the oracle, and the manifest is written by the office being audited.**
No amount of line-opening fixes this: the checker's whole strength is that it opens the
line, and opening the line is exactly what a wrong-claim repair survives.

**This is the same defect as MARS's strike 2** - *"the clause-(b) repair certifies the
manifest against itself."* Two offices, two subsystems, **one mechanism: an instrument
whose ground truth is authored by the party it audits.** That convergence is the
iteration's most important structural result, and neither office saw the other's.

**What CAN detect it - three things, none of them a landing check:**

1. **A symbol-absence node**, which JUPITER already built:
   `test_v20_r15_it18_citation_landing.py:111` -
   `assert "BED_SPECS" not in (ROOT / "V20_R15_IT89_INSPECTOR.md").read_text(...)`.
   This catches mechanism 5 - the cited file lacks the symbol **entirely**. It is real
   and it works. **It does not generalise:** a wrong-claim repair into a file that
   *does* contain the symbol passes it untouched.
2. **Provenance.** A repair that changes `want` as well as the pointer is a different
   act from a repair that changes only the pointer. An instrument that diffs `want`
   across iterations catches every instance of this class, mechanically.
3. **A second office** - which is what happened here, and is not a scalable instrument.

**RECOMMENDED, and this is the reroute `J-18f` owes:** freeze `want` at census time and
require any repair that alters it to be filed as a **withdrawal plus a new citation**,
never as a repair. **That converts an undetectable class into a bookkeeping one.**


## P3 - MARS'S THREE STRIKES - **ALL THREE UPHELD; ONE OF HIS OWN TESTS STRUCK**

### 3.1 THE HEARTBEAT - **UPHELD IN FULL, AND ALREADY REROUTED IN HEAD**

The premise is confirmed by direct observation:

```
[RUN] ls .claude/
   ->  iteration.beat  iteration.start  iteration.watchdog
   ->  ralph-loop.R10.stopped.md  ralph-loop.local.md  settings.local.json  worktrees
   ->  .claude/iteration.pids DOES NOT EXIST
```

**The matrix the brief asks this office to verify cannot be run against HEAD, and that
is the finding.** `ANY_DEAD` no longer exists in `scripts/iteration_timer.sh`. The
corroboration channel has been **retired**, and the retirement comment at `:147-157`
states MARS's charge more completely than the brief does:

```
scripts/iteration_timer.sh:147-157
# THE BEAT MEASURES CHECK-CALL SPACING AND NOTHING ELSE. it.18 tried to promote
# it to a liveness verdict by corroborating against $PIDFILE -- but nothing in
# this repo writes $PIDFILE, and the missing-file arm set ANY_DEAD=1, the
# PERMISSIVE value, so "no evidence" and "evidence of death" printed the same
# sentence. The predicate was `exists dead` and never cleared besides, so one
# finished nurse of four licensed the interruption verdict while three ran.
# A check nobody feeds launders unknown as confirmed.
```

**MARS is upheld on the charge he filed and on one he did not.** The brief names the
missing pidfile and the permissive default. The code names a **second, worse** defect:
the predicate was **`exists dead`, not `all dead`** - so with four nurses registered,
**the first one to finish would have licensed an interruption verdict while three were
still running.** A pidfile with a writer would not have fixed this; it would have made
it fire constantly.

**The matrix, executed against a byte-identical sandbox copy** (md5
`31002af76170cb413c1e1b6a07ebfb37`, equal to the repo file):

```
(a) no pidfile          ->  prints CHECK-CALL SPACING notice, EXIT=0. No verdict.
(b) one live + one dead ->  same shape, EXIT=0. NO interruption verdict.
(c) printf '1234' vs printf '1234\n'  ->  byte-identical output both ways.
[RUN] python -m pytest tests/mars_v20/test_it18_timer_liveness_defaults_permissive.py -q
   ->  3 passed in 0.88s          (MARS's report: 3 failed in 1.18s)
```

**Every arm of MARS's matrix is now inert because the channel is gone.** The
no-trailing-newline case is moot rather than refuted: no arm consumes it.
`grep -n "force" .claude/ralph-loop.local.md` is empty - the
`"re-arm with --force and continue"` string MARS quotes is **no longer in the file**,
and `test_force_rearm_licence_is_actually_gone` passes on its absence. It remains live guidance for whoever feeds the channel next - and the
retirement comment already carries the upgrade path:

```
scripts/iteration_timer.sh:156-157
# ponytail: no liveness verdict at all. If one is ever genuinely wanted, give
# $PIDFILE a writer FIRST, then predicate on `all dead AND >=1 registered` --
# never `exists dead`, and never with a missing file as the permissive default.
```

**RULING: strike 1 UPHELD, reroute ALREADY SHIPPED as `retire`.** The output now names
the quantity it actually holds - `check` prints *"That is CHECK-CALL SPACING, not
liveness"* and *"Not evidence, and not a licence."* **This is the cleanest repair in the
iteration: a channel that could not distinguish no-evidence from evidence was deleted
rather than patched, and the instrument now reports the smaller true thing instead of
the larger false one.**

**Concurrent, and declared:** SATURN is writing `tests/saturn/test_v20_r15_it19_pid_channel.py`
at this moment. **This office audited a subsystem that is being rebuilt underneath it**;
the reading above is HEAD at audit close.

### 3.2 THE CLAUSE-(a)/(b) SWAP - **UPHELD, AND MARS'S OWN TEST HAS GONE GREEN ON A LIE**

**The charge is confirmed.** Called with the arguments the node actually takes today,
under the (a)+(b) swap:

```
arm_of W1 = arm_pl   arm_of W3 = arm_smprime
PASS clause_b_journals_that_wings_own_arm      PASS every_frozen_wing_is_found ('W1',)
PASS every_frozen_wing_is_found ('W3',)        PASS every_frozen_citation_resolves ('a'..'d')
PASS the_frozen_list_names_only_found_wings    PASS every_frozen_wing_cites_all_four_clauses
PASS no_wing_cites_the_same_line_for_two_clauses  PASS the_declared_N_equals_the_wing_rows
   ->  0 of 7 FIRE under the swap.   MARS IS RIGHT.
```

V-26, verbatim, `tests/saturn/test_v20_r15_it14_saturn.py:6-7`: *"for a claim about a
relation between two collections, the falsifying mutation is the one that PRESERVES
EVERY PER-COLLECTION STATISTIC."* The (a)+(b) swap is exactly that mutation.

**And now the finding MARS could not have made about himself.** His regression test
`tests/mars_v20/test_it18_wing_identity_is_self_certified.py` reported `1 failed,
1 passed` in his report. **At HEAD it reports `2 passed` - and the defect is still
there.**

```
[RUN] python -m pytest tests/mars_v20/test_it18_wing_identity_is_self_certified.py -v
   ->  test_the_clause_b_repair_does_not_catch_the_identity_swap PASSED
   ->  2 passed in 0.34s          (MARS's report: 1 failed, 1 passed)
```

The cause is at `tests/saturn/test_v20_r15_freeze_manifest.py:128-133`: SATURN
re-parametrized the node **from arm names to wing IDs**. MARS's test at
`tests/mars_v20/test_it18_wing_identity_is_self_certified.py:78-80` still hardcodes
`[("arm_smprime",), ("arm_pl",)]`. Called with those stale arguments the node raises
**identically whether the swap is applied or not**:

```
FIRE arm_smprime AssertionError arm_smprime cites no clause (a); its arm name is underivable
FIRE arm_pl      AssertionError arm_pl cites no clause (a); its arm name is underivable
   ->  the same error with and without the mutation. NOT SENSITIVE TO THE SWAP.
```

**STRIKE, and it is the most dangerous defect this office found all round: a regression
test that went green because its arguments went stale, while the defect it guards
stands.** A RED that turns GREEN for a reason unrelated to the bug is worse than no test
- it actively reports safety. **MARS's charge is UPHELD; MARS's instrument is STRUCK.**

**SATURN has already shipped the repair, at it.19, concurrently with this audit:**
`tests/saturn/test_v20_r15_it19_wing_identity.py` - *"SATURN it.19 REPAIR 2 - the
clause-(b) repair certified the manifest against itself. MARS's it.18 STRIKE 2, and it
lands."* -> `4 passed`.

### 3.3 "RE-SCOPED" vs "WITHDRAWN" - **MARS IS RIGHT. WITHDRAW IS THE CORRECT VERB.**

**`bad_citations` never opens the cited line.** Verbatim,
`tests/jupiter/test_v20_r15_it14_theory_table.py:53-70`:

```
67	        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
68	        if not (1 <= int(lineno) <= len(lines)):
69	            bad.append(f"{path}:{lineno} - file has {len(lines)} lines")
```

`lines` is read **only to compute `len(lines)`.** There is no `lines[lineno-1]` anywhere
in the function. It tests *the file exists* and *the file is long enough*. **That is
all.**

```
[RUN] scratchpad probe: feed every WITHDRAWN key to bad_citations
   ->  count of withdrawn keys: 26        named bad by bad_citations: 0
[RUN] python -m pytest tests/mars_v20/test_it18_census_zero_bad_was_never_a_citation_claim.py -v
   ->  AssertionError: the it.14 checker named 0 of 26 pointers that JUPITER himself
   ->                  withdrew as wrong.   assert 0 == 26
   ->  AssertionError: the it.17 landing instrument holds 30 hand-verified citations
   ->                  against a census of 123.   assert 30 >= 123
   ->  2 failed, 1 passed in 0.93s          <- MARS'S REDs ARE LIVE AT HEAD
```

**MARS's number is wrong in his own disfavour: it is 0 of 26, not 0 of 25.** `WITHDRAWN`
holds 26 entries. The finding is one pointer stronger than he claimed.

**RULING: "re-scoped" is WRONG and "withdrawn" is RIGHT.** The coordinator published
*"The it.15 result is re-scoped, not withdrawn"* (`V20_R15_JOURNAL.md:3105`, `:3303`,
adopted from `V20_R15_IT17_JUPITER.md:17`) and himself flagged the adoption at `:3312`.

**A re-scope says the claim was always outside the instrument's remit. That is only
available if some scope ever made `0 bad` a statement about citation landing - and none
did.** `bad_citations` is a file-existence and file-length check that never reads a
line. **`0 bad` never meant the citations were right; it meant the files were there.**
The office that published it as a citation result set the scope broadly, and **a
checker's narrowness cannot retroactively narrow a claim its author made broadly.**
`0/123` is **WITHDRAWN**. The coordinator should correct the verb at both journal sites.


## P4 - MERCURY'S RIG AND THE INDISTINGUISHABLE PREDICATES - **VERIFIED, AND ONE STRIKE**

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it15_arena_rig.py -v   ->  17 passed
   ->  ::test_MUST_FIRE_the_instrument_finds_a_crossing_it_was_told_is_there PASSED
   ->  ::test_MUST_NOT_FIRE_the_near_miss_that_separates_the_rule_from_its_lazy_twin PASSED
```

**The escalation, UPHELD.** Runner predicate `scripts/v15_r1.py:909`
(`crosses=bool(m + half < floor1)`); rig copy `tests/mercury/arena_rig.py:78`; corpus
`BANKED_40` at `tests/mercury/phase_c_price.py:18`. Run **without touching the repo** -
both rules imported in-process from a scratchpad driver, `md5sum arena_rig.py` =
`8a857eab8198c4de0bc90a4da08bb1d8` before and after:

```
ORIGINAL  (m + half < floor1) :  arm_pl (12, 16)   arm_smprime (1, 16)
LAZY TWIN (eval_nrmse < floor):  arm_pl (12, 16)   arm_smprime (1, 16)
[RUN] diff orig_predicate_result.txt lazy_predicate_result.txt  ->  empty, exit 0
NEAR-MISS plant, ORIGINAL  crosses(): [False, False] -> any: False
NEAR-MISS plant, LAZY TWIN crosses(): [True,  True ] -> any: True
```

**RULING: REASSURING about the numbers, ALARMING about the instrument, and the two are
not in tension.**

*Reassuring:* every crossing count published this round - `12 of 16`, `1 of 16` - is
**invariant to which rule produced it.** No published number moves. An office that used
the lazy twin by accident would have shipped the same figures.

*Alarming:* **the real data never exercised the difference.** The 40 banked cells contain
no case where the interval straddles the floor. So the round's crossing counts are
correct **but untested** - produced by a rule whose distinguishing behaviour the corpus
never reached. **The only evidence the right rule was used is a case MERCURY had to
invent.** Correctness by coincidence of corpus is not correctness by instrument.

**Same structure as P1.1**, where five of six census nodes stayed green against a blinded
checker. **In both places the only load-bearing evidence is a planted negative. A suite
run over well-behaved data cannot certify the rule it used.**

### 4.1 STRIKE - MERCURY'S SELF-CHARGE IS FALSE AS STATED

He charges himself with *"a rho asserted at `0.697` that measures `-0.1224`"*. **It does
not survive execution.**

```
[RUN] python -c "from tests.mercury.arena_rig import load_cells, spearman_rho, RUN_ORDER_RHO ..."
   ->  within arm_smprime seed-vs-secs rho = 0.6970588235294117
   ->  RUN_ORDER_RHO constant            = 0.697059        <- MATCHES
   ->  across-arms index-vs-secs rho     = -0.12244898296078643
```

There are **two** assertions in that one function, over two different populations:
`test_v20_r15_it15_arena_rig.py:107` (`c["seed"]` vs `c["secs"]`, `arm_smprime` only) and
`:111` (`list(range(len(every)))` vs `c["secs"]`, all arms). **Both reproduce exactly.
Neither is wrong.** `0.697` does not "measure `-0.1224`". **STRUCK as worded.**

**What is actually defective:** at `:107` both operands are named; at `:111` the first is
`list(range(len(every)))` - **a bare enumeration index with no name and no guarantee it
is run order.** Real, unnamed-operand family, at `:111` only, and a *naming* defect, not
a *value* defect.

**It cuts both ways.** Volunteering a defect against himself in the family this office
filed against him one iteration earlier is the behaviour the round wants. **But an
overstated self-charge still puts a wrong number in the ledger, and the next reader
cannot tell it was wrong in the author's own favour to fix.**

## P5 - THE COORDINATOR - **ONE UPHELD, ONE STRUCK, LOOP FILE CLEAN**

### 5.1 CORRECTION 29 - MECHANISM RIGHT; NO SURVIVING CLAIM RESTS ON A STALE READING

`V20_R15_JOURNAL.md:3356-3375`. Both dispatched agents falsified their own briefs from
disk - MERCURY on `697 / 6,601 bytes`, JUPITER on *"21,774 bytes at HEAD, not 10,147."*
The mechanism (sizes read at one moment, dispatched minutes later while agents were still
writing) is correct and self-reported.

**Checked:** the stale figures appear at `:3086`, `:3304`, `:3306`, `:3325`, `:3365` -
**every one inside the correction that withdraws it.** No load-bearing claim survives on
a stale size. **UPHELD, no strike.**

**One residual.** The correction's own number has already drifted:

```
[RUN] wc -lc V20_R15_IT17_JUPITER.md  ->  339 lines, 21,897 bytes   (correction says 21,774; +123)
```

**A size written into an immutable record is the same dated-measurement class as
`J-18a`'s census count** - the correction reproducing, in miniature, the mistake it
corrects. **OPEN, filed, not struck.**

### 5.2 **STRIKE - THE CORRECTIONS INDEX IS TEN CORRECTIONS STALE**

The loop prompt makes it load-bearing: *"read ... its CORRECTIONS INDEX at the head
before trusting any early entry."*

```
CORRECTIONS INDEX at head of V20_R15_JOURNAL.md  ->  contains C1 .. C19 only
Correction 29 lives at :3356, in the it.18 body, UNINDEXED
```

**Corrections 20-29 are in the body and absent from the index.** A reader following the
loop's own instruction **will not find them, and will then trust an entry that has been
corrected.** Not a prose defect: a broken lookup on the structure the round designated as
its guard against stale entries, **failing silently in the trusting direction.**
**STRUCK. Bring the index to 29 before the next prognosis reads the journal.**

### 5.3 THE LOOP FILE - INTACT, `--force` REMOVAL CLEAN

```
[RUN] head -8 .claude/ralph-loop.local.md
   ->  active: true | iteration: 4 | session_id: f7ed3dac-... | max_iterations: 60
[RUN] grep -n -- "--force" .claude/ralph-loop.local.md    ->  (empty)
[RUN] scripts/iteration_timer.sh:52  sed -n 's/^iteration: *//p' "$f" | head -1
[RUN] bash scripts/iteration_timer.sh check  ->  runs, reports elapsed/left. NOT BROKEN.
```

**All four keys present; the timer runs.** The parser reads **only `iteration:`** -
`active`, `max_iterations`, `session_id` are never read by it, so the edit could not have
broken the loop through them even had it damaged them.

**Worth recording:** the `--force` *licence* is gone from the prose; the *capability*
remains in the code at `scripts/iteration_timer.sh:72`. **Permission withdrawn, mechanism
retained** - the right order, since a removed capability cannot be audited and a removed
permission can. **OPEN deliberately: this office does not recommend removing the flag.**

**Note, and it is this office's own:** `.claude/ralph-loop.local.md` changed md5 during
the audit window (`68d9277616094685aa134ab27177fa3b` -> `fb9d194e0dba0d258af82e201b1f2392`)
with **no write by this office or any of its nurses.** The frontmatter reading above is
dated to audit close.

## SELF - THE `[RUN]` SCORE: **YES, IT CHANGED. 38 EXECUTABLE OF 43 MARKED.**

The concession stands: `2/29` at it.16, conceded, and this office declined to use
SATURN's `NODE`-regex defect as a defence. **The question was whether it moved.**

```
[RUN] grep -c "\[RUN\]" on the three most recent inspector reports
   ->  V20_R15_IT13_INSPECTOR.md :   3 markers  (920 lines, 31 fences)
   ->  V20_R15_IT89_INSPECTOR.md :   3 markers  (968 lines,  4 fences)
   ->  V20_R15_IT16_INSPECTOR.md :  43 markers  (686 lines, 32 fences)
```

**Three markers to forty-three, in the shorter report**, naming real commands -
`sha256sum` before/mutate/after, `pytest -k` by node name, `git status --porcelain` at
open and close, `git rev-parse HEAD`, `sed -n` at cited lines.

**Five of the forty-three are not commands and this office counts them against itself:**
`:472` is a heading containing the literal `[RUN]`; `:484` and `:525` are prose about
markers; `:516` is a `path:line` citation wearing a `[RUN]` prefix; `:379` quotes another
office. **Honest score: 38 executable of 43 marked.**

**This report:** every `[RUN]` above names a command run in this audit, and both mutations
ship digests before and after.

## LIVE AT AUDIT CLOSE - THE CENSUS MOVED A THIRD TIME, DURING THIS AUDIT

**Both citation suites were GREEN at audit open (`12 passed`, 15:59) and RED at audit
close (16:11) with the test file BYTE-IDENTICAL** (`f846c839476e5bcd6e92d1759241c642`,
unchanged; mtime 16:03 predates neither run's result):

```
[RUN] python -m pytest tests/jupiter/test_v20_r15_it18_citation_landing.py -q
   ->  E  AssertionError: C17: V20_R15_JOURNAL.md:650 lacks 'THE FREEZE'
   ->  E  assert 'THE FREEZE' in ''
   ->  E   +  where '' = line_at('V20_R15_JOURNAL.md', 650)
   ->  2 failed, 4 passed
```

**Nothing about the test changed. `V20_R15_JOURNAL.md` grew underneath it**, and line 650
now holds a different line - in fact an empty one. `C17` did not become wrong; **it
became stale.**

**This is `J-18a` firing a third time, in real time, against a live target.** JUPITER
filed *a census count is a dated measurement, not a table constant* at `128 -> 129`. This
office has now watched the same property break a **pointer**, not a count: **any citation
into an append-only file is a dated measurement too, and the journal is appended every
iteration.**

**RULING: `J-18a` should be GENERALISED and it is under-claimed as filed.** The unit that
expires is not the census total - it is **every pointer into a file that is still being
written.** `V20_R15_JOURNAL.md` is cited by the census and appended by the coordinator
every iteration, so **a fraction of the table rots on a schedule.**

**The reroute this office recommends:** citations into append-only files must anchor on a
**stable marker** (a heading, a correction id, a `FREEZE-SHA256` line) and never on a
line number - the mechanism `df8892f` already adopted for the code snapshot, *"find the
code snapshot by walking for its marker, not by naming its mount."* **The same fix, one
subsystem over, is already in this repo's history.**

**This RED is NOT charged against JUPITER.** It appeared after his report was filed, from
a write by another office, and it is evidence for his thesis rather than against it.

## NOT REACHED - NAMED, NOT BURIED

1. **121 of 129 repaired citations** were not individually opened. Sample is 6, plus the
   2 in C98/C103. The census is trusted on its instrument, which P1.1 verified; it is
   **not independently re-enumerated** by this office.
2. **The `28 claimed vs 25 in the it.17 MANIFEST` gap** (P1.3). Named, unresolved.
3. **The `30 >= 123` coverage failure** surfaced by MARS's own test is quoted in 3.3 but
   **not adjudicated** - the landing instrument covering 30 of 123 is a live RED this
   office did not have time to rule on. **It is the largest open item leaving it.18.**

## TREE - HONEST, INCLUDING CONCURRENT it.19 PATHS

```
[RUN] git status --porcelain | wc -l   ->  113   (it.16 audit closed at 102)
[RUN] git status --porcelain | grep "^ M"
   ->   M MISTAKES.md    M house-events.jsonl    M pytest.ini    M scale/ledger.py
[RUN] ls .claude/
   ->  iteration.beat  iteration.start  iteration.watchdog
   ->  ralph-loop.R10.stopped.md  ralph-loop.local.md  settings.local.json  worktrees
   ->  NO iteration.pids                      <- MARS's strike-1 premise CONFIRMED
```

**The four `M` entries pre-date this audit** and are unchanged by it. **No git write of
any kind was performed by this office or by any of its four nurses.** Nothing touched
Kaggle.

**Two mutations were applied and both are reverted with digests shown in-section:**

| mutation | file | digest before | digest after |
|---|---|---|---|
| blind the checker to line content (P1.1) | `tests/jupiter/test_v20_r15_it18_citation_landing.py` | `f846c839476e5bcd6e92d1759241c642` | **`f846c839476e5bcd6e92d1759241c642`** |
| predicate twin (P4.2) | `tests/mercury/arena_rig.py` | `8a857eab8198c4de0bc90a4da08bb1d8` | **`8a857eab8198c4de0bc90a4da08bb1d8`** |

The predicate-twin comparison was run **entirely from the scratchpad**, importing both
rules in-process; **the repo file was never opened for writing.** The census mutation
was written to disk and reverted from a pre-edit backup.

**CONCURRENT it.19 WORK, declared because it moved the tree under this audit.** Four
it.19 paths appeared during the audit window and are not this office's:

```
?? tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py
?? tests/saturn/test_v20_r15_it19_pid_channel.py        <- SATURN answering MARS strike 1
?? tests/saturn/test_v20_r15_it19_theory_digest.py
?? tests/saturn/test_v20_r15_it19_wing_identity.py
```

**`test_v20_r15_it19_pid_channel.py` is SATURN building the pid channel MARS's strike 1
says is unfed, while this audit was auditing that strike.** The audit's reading of the
pid channel is therefore **a reading of a moving target**, and it is dated: the `ls`
above is HEAD at audit close. Named, per correction 29's own lesson.

