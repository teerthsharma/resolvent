# V20 R15 it.1 — HEALTH INSPECTOR

Repository `C:\Users\seal\Desktop\New folder (32)`, branch `v17k-gate0`, HEAD `207e7b9`.
Audit run 2026-09-02 01:19–01:35 IST from the repository root, under the FIXED
`pytest.ini`.

**This document files no finding of its own.** It has no opinion about whether any
claim is *correct*. Every ruling below is about the LOG: whether a claim is BOUND.
Where a ruling reads `struck`, the claim leaves the round's verdict and goes to Open.

**26 claims audited, 7 struck.**

---

## 0. THE REGIME EVERY CLAIM WAS MEASURED UNDER

`pytest.ini` gained `kaggle` to `norecursedirs` at mtime **`Sep 2 01:03:51`**. The four
reports were written at:

| artefact | mtime |
|---|---|
| `V20_R15_IT1_WILSON.md` | `00:55:28` |
| `V20_R15_IT1_MARS.md` | `01:01:06` |
| `V20_R15_IT1_SATURN.md` | `01:01:11` |
| `V20_R15_IT1_JUPITER_M14.md` | `01:02:37` |
| **`pytest.ini` fix** | **`01:03:51`** |

**Every agent test result in this round was therefore produced BEFORE the fix**, which
means every one of them was produced by an explicitly-named path, never by a root
`pytest -q`. No claim in this round rests on a root run, and none of them could have.
That is the benign reading and it is the correct one: no agent claimed a root run.

The two regimes, measured now:

| command | result |
|---|---|
| `python -m pytest -q --collect-only` (fixed) | `2895 tests collected in 7.78s`, EXIT=0 |
| same with `kaggle` removed from `norecursedirs` (pre-fix) | `2828 tests collected, 1 error in 6.65s`, EXIT=1 |

Wilson's `200 errors during collection` is not contradicted — the pre-fix re-run above
carried `-x`, which halts at the first import mismatch. The direction is confirmed:
the pre-fix root tree errors at collection, the fixed tree does not.

**Everything below was re-run under the FIXED tree.**

---

## 1. CHECK 1 — EVERY TEST CLAIMED GREEN, RE-RUN

### 1.1 The three suites

| # | agent | claim in `house-events.jsonl` | re-run | verdict |
|---|---|---|---|---|
| C1 | Jupiter | `m14_cheeger_suite` green, `43 passed in 0.63s` | `python -m pytest -q tests/jupiter/test_m14_cheeger.py` → EXIT=0, **`43 passed in 0.51s`** | **clean** |
| C2 | Jupiter | whole directory unbroken, `92 passed` | `python -m pytest -q tests/jupiter/` → EXIT=0, **`92 passed in 42.66s`** | **clean** |
| C3 | Saturn | `13 passed; 1 standing RED` | `python -m pytest -q tests/saturn/test_v20_r15_wing_rubric.py -rf` → EXIT=1, **`1 failed, 13 passed in 4.08s`** | **clean** |
| C4 | Saturn | the standing RED is intentional, on annex producers | the single FAILED node is `test_every_annex_run_instance_has_a_producer_in_the_tree` | **clean** |
| C5 | Mars | 6 RED / 4 GREEN across two files | `python -m pytest -q tests/mars_v20/ -rA` → EXIT=1, **`6 failed, 4 passed in 1.48s`**, node-for-node identical to the ten logged events | **clean** |

Saturn's suite defines 11 functions and generates 14 nodes (one is parametrised over
the four clauses). 13 + 1 = 14. The arithmetic closes.

### 1.2 Saturn's first RED, and why it cannot be re-run

| # | claim | verdict |
|---|---|---|
| C6 | Saturn: the suite went RED first, `7 failed, 5 passed`, "`V20_R15_IT1_SATURN.md` does not exist" | **clean, on the log — unreproducible by construction** |

The RED is recorded at `house-events.jsonl:11342` and it is the correct shape: the
7 REDs are the bind (the report did not exist), the 5 greens are the calibrations.
It **cannot** be re-executed today for two independent reasons, both structural rather
than evasive: the report it binds now exists, and the file has since grown from 12
nodes to 14 (the producer-search bind and its own calibration landed after the first
RED — `8 failed, 6 passed` at `:11369` is the 14-node intermediate). 7+5=12 and
8+6=14 are both internally consistent with the file's history. Accepted on the log.

### 1.3 Jupiter's two planted negatives, re-applied

Both mutations were re-applied by hand to `scripts/v20_m14_cheeger.py` and reverted.
The file is untracked, so `git checkout` could not have restored it; a byte-exact
backup was taken first and the SHA256 verified equal afterwards
(`fd75917307f216e489a77bc376a443df71ba156c733f165a8e4165179db86f77` before and after).

| # | claim | re-applied | verdict |
|---|---|---|---|
| C7 | mutation A: Fiedler coordinate scrambled → **`7 failed, 36 passed`** | **`8 failed, 35 passed`** | **STRUCK as to the count** |
| C8 | mutation B: `min(vol S, vol Sᶜ)` → `vol S` in both paths → **`6 failed, 37 passed`** | **`6 failed, 37 passed`**, node list identical to the report's | **clean** |
| C9 | the two failure sets are DISJOINT | intersection empty under Jupiter's 7-node A-set *and* under the observed 8-node A-set | **clean** |

**C7, struck.** Re-application fired on a **superset** of Jupiter's list — the same
seven nodes plus `test_the_sweep_cut_obeys_its_own_quadratic_guarantee[dumbbell_4x4]`.
The cause is that "a scrambled ordering" names no particular permutation; the report
does not say which one, so `7 failed, 36 passed` is not a reproducible figure. The
*instrument* verdict survives intact and is not struck: **the planted negative fires**,
on at least seven nodes, and its failure set is disjoint from B's. This is not the
`V-16` class. Only the exact count leaves the record.

Mutation B reproduced exactly, node for node:

```
test_cheeger_two_sided_holds_with_the_exact_phi[complete_8]
test_cheeger_two_sided_holds_with_the_exact_phi[cycle_8]
test_cheeger_two_sided_holds_with_the_exact_phi[hypercube_16]
test_cheeger_two_sided_holds_with_the_exact_phi[star_10]
test_the_repo_crude_phi_is_an_upper_bound_on_the_exact_minimum
test_the_repo_crude_phi_is_bracketed_by_the_certificate_free_bounds
```

### 1.4 Mars's four GREENs — a control that passes vacuously is worthless

Each was probed by making true the thing it controls for.

| # | node | probe | verdict |
|---|---|---|---|
| C10 | `test_control_the_census_is_not_degenerate_and_the_journal_is_readable` | real census `n=8192, neg=4123, pos=4069, frac_neg=0.5032958984375`. On a degenerate (all-positive) draw, `0.4 < frac_neg < 0.6` evaluates **False** | **clean — live** |
| C11 | `test_control_a_column_that_is_genuinely_the_arms_does_not_match_the_census` | substituting `frac_gate_annihilated` for `eval_nrmse` in the *same expression* makes it **FAIL**, intersection `{0.5032958984375, 0.4967041015625}`. Nearest real `eval_nrmse` sits `0.2928` away from either census value | **clean — the strongest of the four** |
| C12 | `test_the_manifest_identity_field_n_zero_gates_carries_more_than_the_corpus` | `seen = {0, 4123, 4069}`, `corpus = {4123, 4069}`. Passes **only** because of `0` | **clean — margin flagged** |
| C13 | `test_control_a_genuine_dial_move_is_visible_at_these_shapes` | with no dial move the reading is **exactly `0.0`**; with the real move it is `0.11135276160324367` and `0.23697569219106063`, matching the report's `1.11e-01 to 2.37e-01` | **clean — live** |

**C12 carries a note for the coordinator, not a strike.** The assertion is live — it
can fail and it very nearly does. **Seven of the eight cells publish an
`n_zero_gates` that IS a corpus sign count**; the suite passes on the single escape
value `0`, contributed by cell `arm_smprime:t2:n2048:seed2`. That is the same cell
Mars's own RED `test_the_exp_scan_planted_negative_has_a_nonempty_region_on_every_deciding_cell`
cites as the one where the planted negative cannot fire. The green and the red rest
on the same anomalous cell from opposite sides. Binding is intact; the margin is one
element wide and the coordinator should know it.

Mars's STRIKE 1 arithmetic reproduces exactly:
`frac_gate_annihilated` across the eight cells reads
`[0.5033, 0.5033, 0.0, 0.4967, 0.5033, 0.5033, 0.5033, 0.5033]` at full precision —
six cells at `4123/8192`, seed 3 at the exact complement `4069/8192`, one at `0.0`.

---

## 2. CHECK 2 — EVERY FINDING HAS A MATCHING RED BEFORE IT

Walked in log order over `house-events.jsonl:11338-11393` (17 `finding` events this
round).

### 2.1 Bound

| # | finding | the RED that binds it | verdict |
|---|---|---|---|
| C14 | **MARS STRIKE 1** — `frac_gate_annihilated` equals the eval draw's own sign census, 4123/8192 | `:11351` and `:11352`, both `status:red`, both precede the finding at `:11361`; both RED again today | **clean** |
| C15 | **MARS STRIKE 2** — the three arms are one operator at the `β=qk=g=1` corner | `:11356`–`:11359`, four REDs, all precede the finding at `:11362`; all RED again today | **clean** |
| C16 | **JUPITER** — the annex's `[RUN: my crude φ FAILED the sanity check]` is CONTRADICTED | `test_the_repo_crude_phi_is_an_upper_bound_on_the_exact_minimum` asserts **both** Cheeger halves on `LargestJoin_S2Rips_64` (`exact²/2 ≤ g+TOL` and `g ≤ 2·exact+TOL`); it sits inside the suite logged RED at `:11341` and GREEN at `:11363`, and mutation B fires it | **clean** |
| C17 | **SATURN** — the N=3 wing list, twelve citations resolving | `:11342` RED first (`V20_R15_IT1_SATURN.md does not exist`), GREEN now: `test_every_wing_cites_all_four_clauses`, `..._carries_its_anchor[a-d]`, `..._same_line_for_two_different_clauses` all PASSED | **clean** |
| C18 | **SATURN K6** — four annex `[RUN]` instances have no producer | `:11369` RED (`8 failed 6 passed`) precedes the finding; `test_every_annex_run_instance_has_a_producer_in_the_tree` is RED again today | **clean as to binding** (two of its four instances are struck under CHECK 3 — §3.2) |

**On C16 and the collection-error RED.** Jupiter's first RED is
`ModuleNotFoundError: No module named 'scripts.v20_m14_cheeger'` — a collection error,
not a failed assertion. It is logged under the same name `m14_cheeger_suite` as the
later green, so the letter of the rule is met, and it is the honest TDD shape: the
test file existed before its producer. **But a collection-error RED binds the SUITE,
not any particular proposition.** It certifies that the file could not run; it
certifies nothing about what any individual node asserts. So it binds exactly those
Jupiter claims that some node in the suite actually asserts, and nothing else. That
discriminator does the work in §2.2 below.

### 2.2 Unbound — struck to Open

| # | finding | why | verdict |
|---|---|---|---|
| C19 | **MARS's proposed new class P-12** (`STRUCK.md`'s evidence command no longer reproduces) | **`grep -c 'P-12' house-events.jsonl` returns `0`.** No `finding` event was ever logged for it, and no test in `tests/mars_v20/` mentions it. There is no RED, and there is no claim in the log for a RED to bind | **STRUCK — UNBOUND → Open** |
| C20 | **JUPITER** — `crude/exact = 1.0`, φ = 1/27, the crude φ is EXACT | the node that could bind it asserts `crude >= exact - TOL` — **one-sided**. It binds "crude is an upper bound". It would pass identically if crude were ten times exact. **No node in the suite asserts the equality.** The figure lives in `results/v20_m14_cheeger.txt` as printed output, not as an assertion | **STRUCK — UNBOUND → Open** |
| C21 | **JUPITER** — the L-DOM census (0 of the registered beds satisfy Cheeger's hypotheses; BED-M 0/384, the `{−1,0,+1}` object 0/2048, BED-K 0/8 and nilpotent) | `grep` over `tests/jupiter/test_m14_cheeger.py` for `domain_census`, `census`, `BED`, `bed_k`, `nilpotent` returns **zero hits**. The producer is `scripts/v20_m14_cheeger.py::domain_census`, whose output is written to a `.txt` and never asserted. **Not one of the 43 nodes binds this** | **STRUCK — UNBOUND → Open** |
| C22 | **SATURN K5** — GPU-seconds-to-floor is unmeasurable for every wing | the suite contains **no** occurrence of `K5`, `GPU-second`, `gpu_second` or `seconds-to-floor`, and defines no node for it. The finding at `:11372` has no RED anywhere in the round | **STRUCK — UNBOUND → Open** |

**C19 note, so the coordinator is not misled about the substance.** Mars's evidence
commands *do* reproduce exactly — `9`, `5`, `0`, `5` respectively, and the five hits
for `git log -S"def absorbing_boundary_kernel"` are indeed
`PREREGISTRATION_HOLE_AUDIT.md` + `tests/cameron/test_harmonic_attribution.py` at
`7abb325` plus four `kaggle/snapshot/repo/` copies of both. Mars is also careful to
label P-12 "filed, not struck". None of that changes the ruling: **a claim that never
entered the log cannot be bound by it.** One correction of fact for whoever picks it
up: `STRUCK.md:29` contains no evidence *command*, only prose describing a search, so
P-12 as worded ("`STRUCK.md`'s own evidence command") names something that is not in
that file.

**C22 note.** Saturn's K5 arithmetic is stated with citations in the report
(`V17_R4_RETAKE_PRICE.md:194-196`, 15.970s / 1.614s / 1.497s, 9.90×). The kill is
argued, not bound. It goes to Open as an unbound kill, not as a refuted one.

---

## 3. CHECK 3 — NO AGENT CONTRADICTS WILSON

Wilson was not re-verified. His report is the reference.

### 3.1 Clean

| # | topic | ruling |
|---|---|---|
| C23 | **The Lean ledger.** Wilson: only `#2` and `#5a` of the queried set carry source declarations; `#10 #11 #14 #17 #17′ #18 #19 #20 #21 #22 #23` have none | **No planet cites a Lean theorem number at all.** Jupiter §8 goes the other way and explicitly declines the ledger: *"M14 should not enter the Lean ledger this round."* Mars and Saturn make no Lean claim. Nothing to contradict | **clean** |
| C24 | **The annex `[RUN]` values `d=65` and `1.3e-3`.** Wilson §7: both **NOT FOUND IN TREE**, zero hits outside the annex | Saturn's K6 names both as unproduced. **Agrees with Wilson** | **clean** |
| C25 | **The arms census.** Wilson §1.1 lists `arm_phase`, `arm_pl`, `arm_smprime` as three modules with three written formulas, all wired | Mars measures operator identity **at the `β=qk=g=1` corner**, a question Wilson never addresses; Saturn's `N=3` counts wings with resolving citations, and Wilson agrees three arm modules exist. **Neither contradicts Wilson**, because Wilson made no distinctness claim | **clean** |

### 3.2 Struck against Wilson

| # | claim | Wilson's fact | verdict |
|---|---|---|---|
| C26 | **SATURN K6, instance `0.492 vs KL 0.519`** — "no `.py`/`.json`/`.jsonl`/`.txt`/`.log` producer in tree" | Wilson §7 M13: `0.492` is **FOUND**, every hit `0.492188`, with **producers `scale/foreman_consequence.py:12` and `scale/foreman_signfloor.py:382`** — both `.py`, both in-tree, neither under `attic` nor `kaggle`, so neither is excluded by Saturn's own filter | **STRUCK** |
| C27 | **SATURN K6, instance `d=20 at the same`** | the searched token is annex **prose**, not a figure. `ANNEX_RUN_TOKENS` maps M9-F1′ to the literal string `"d=20 at the same"`, and no `.py` or `.json` emits an English phrase | **STRUCK** |

Both strikes are the same mechanism and it is a mechanism Saturn's own instrument was
built to catch. `producers_for` does a literal substring search over
`.py/.json/.jsonl/.txt/.log`. Its calibration node
`test_the_producer_search_finds_a_number_that_does_have_one` calibrates on a **number**,
`0.7071067811865476`, and asserts the hit lands under `results/`. That calibration
certifies the search **for numeric tokens**. Two of the four bound tokens are not
numeric: `"d=20 at the same"` and `"0.492 vs KL 0.519"` are phrases lifted from annex
prose, and a search for a phrase in machine output is structurally incapable of
returning a hit. That is `MISTAKES.md:117` **V-7**, cited by name in Saturn's own
docstring at line 139 and enforced there for the *absence* search — but not extended
to the producer search's non-numeric tokens.

`d=65` and `1.3e-3` are genuine figures and Wilson independently confirms both are
absent. **K6 survives on two of its four instances.** The RED is real, it was RED
first, and it stays RED; it now stands on `d=65` and `1.3e-3` alone.

---

## 4. THE RULING ON THE SATURN / MARS TENSION

The question put to this office is narrow and this is the whole of the answer.

> **Is each of the two claims BOUND by a RED test that was RED first?**

**Both are. Neither is struck. The two do not collide at the binding level.**

| | SATURN: N=3 wings | MARS: the three are one operator |
|---|---|---|
| RED first | yes — `:11342`, `7 failed, 5 passed`, before the report existed | yes — `:11356`, `:11357`, before the STRIKE 2 finding at `:11362` |
| RED reproduces today | n/a (green now, by design) | **yes** — both nodes FAILED on re-run |
| GREEN reproduces today | **yes** — 13 passed | n/a |
| the proposition actually bound | *each of three named wings carries four citations that resolve to a file, a line inside it, and an anchor present on that line* | *`arm_phase(s=0)`, `arm_pl(g=log m, s=0)` and `arm_smprime(β=qk=g=1)` agree to `1.11e-16 … 8.11e-16` at two float64 shapes, against a control dial move of `1.11e-01 … 2.37e-01`* |

These are different propositions. Citation resolvability is not operator distinctness,
and operator identity at one corner is not a citation defect. **A log can carry both
without inconsistency, and it does.** Mars's own non-firing attack — trained β reading
`0.588–1.509` across eight cells, never at the corner — is filed honestly under
"ATTACKS THAT DID NOT FIRE" and it is exactly the datum that lets both stand.

**Whether N is 3 or 1 is the coordinator's call. This office does not make it and has
not made it.**

---

## 5. THE TREE

Every mutation applied during this audit was reverted. `scripts/v20_m14_cheeger.py`
was backed up byte-exact before Jupiter's two mutations and restored after each;
SHA256 `fd75917307f216e489a77bc376a443df71ba156c733f165a8e4165179db86f77` before and
after, and `43 passed` reproduced after each restore. The Mars control probes were run
as throwaway scripts in the scratchpad and touched no repository file.

`git status --porcelain` at close is **identical to its state at open**, modulo this
report and the audit events this office is required to append:

```
 M house-events.jsonl        <- audit events appended by this office, as required
 M pytest.ini                <- pre-existing; the kaggle fix, made before this audit
?? CEQ_V20_R15_CONTRACT.md
?? V20_R15_IT1_INSPECTOR.md  <- this file
?? V20_R15_IT1_JUPITER_M14.md
?? V20_R15_IT1_MARS.md
?? V20_R15_IT1_SATURN.md
?? V20_R15_IT1_WILSON.md
?? V20_R15_JOURNAL.md
?? results/v20_m14_cheeger.txt
?? scripts/iteration_timer.sh
?? scripts/v20_m14_cheeger.py
?? tests/jupiter/test_m14_cheeger.py
?? tests/mars_v20/
?? tests/saturn/test_v20_r15_wing_rubric.py
```

**No git write command was run. Nothing touched Kaggle.**

---

## 6. THE LEDGER

**26 claims audited, 7 struck.**

| # | agent | claim | verdict |
|---|---|---|---|
| C1 | Jupiter | `m14_cheeger_suite`, 43 passed | clean |
| C2 | Jupiter | `tests/jupiter/` whole directory, 92 passed | clean |
| C3 | Saturn | wing rubric, 13 passed / 1 RED | clean |
| C4 | Saturn | the standing RED is the annex-producer bind | clean |
| C5 | Mars | `tests/mars_v20/`, 6 RED / 4 GREEN | clean |
| C6 | Saturn | first RED, 7 failed / 5 passed | clean (on the log) |
| **C7** | **Jupiter** | **mutation A count `7 failed, 36 passed`** | **STRUCK** — re-application reads `8 failed, 35 passed`; the permutation is unspecified so the count is not reproducible. The negative *fires*; only the count is struck |
| C8 | Jupiter | mutation B, `6 failed, 37 passed` | clean — exact, node for node |
| C9 | Jupiter | the two failure sets are disjoint | clean |
| C10 | Mars | control: census not degenerate | clean |
| C11 | Mars | control: `eval_nrmse` ≠ census | clean |
| C12 | Mars | `n_zero_gates` carries more than the corpus | clean — passes on one element of three; 7/8 cells ARE corpus counts |
| C13 | Mars | control: a genuine dial move is visible | clean |
| C14 | Mars | STRIKE 1 | clean |
| C15 | Mars | STRIKE 2 | clean |
| C16 | Jupiter | the annex's failed-sanity-check `[RUN]` is contradicted | clean |
| C17 | Saturn | the N=3 wing list | clean |
| C18 | Saturn | K6 | clean as to binding |
| **C19** | **Mars** | **P-12, the proposed new class** | **STRUCK** — no finding event, no test, no RED → Open |
| **C20** | **Jupiter** | **`crude/exact = 1.0`, φ = 1/27** | **STRUCK** — the node asserts `crude >= exact` only; the equality is unbound → Open |
| **C21** | **Jupiter** | **the L-DOM census** | **STRUCK** — zero of the 43 nodes assert it → Open |
| **C22** | **Saturn** | **K5, GPU-seconds-to-floor** | **STRUCK** — no test, no RED anywhere in the round → Open |
| C23 | all | the Lean ledger vs Wilson | clean — no planet cites a theorem number |
| C24 | Saturn | K6 instances `d=65`, `1.3e-3` vs Wilson | clean — agrees with Wilson |
| C25 | Mars, Saturn | the arms census vs Wilson | clean — different propositions |
| **C26** | **Saturn** | **K6 instance `0.492 vs KL 0.519`** | **STRUCK** — Wilson names `.py` producers `scale/foreman_consequence.py:12`, `scale/foreman_signfloor.py:382` |
| **C27** | **Saturn** | **K6 instance `d=20 at the same`** | **STRUCK** — the token is prose; no producer emits prose (V-7 shape) |

A struck claim is struck. It leaves the verdict and goes to Open, and the agent who
filed it does not get to appeal.
