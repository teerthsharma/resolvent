# V20 R15 it.28 — SATURN (WATSON, instruments)

Three repairs were asked for. **Two shipped as measurements and one shipped as a measured
null**, which is the honest outcome and is filed as one rather than dressed as a repair.

The weakest sentence of it.27 — *"my claim that the other eight (and mars's 40) predate this
iteration rests on inspection of failure names, not on a measured baseline"* — is closed
below. A retrospective **pre-it.27** baseline is ruled **UNOBTAINABLE**, with the reason
measured rather than asserted, and the recording discipline that would have made it obtainable
is stated **and exercised**: an it.28 baseline was taken **before** this office's only edit and
re-taken after it, and the two differ by exactly the one node the edit was for.

Every count below is dated and carries a content digest of its subject. **No HEAD SHA appears
in this record.**

| subject | sha256 (16) |
|---|---|
| `V20_R15_THEORY_TABLE.md` (freeze subject) | `942e4208893444cd` |
| `V20_R15_IT19_SATURN.md` (the re-declared freeze) | `63d30b04ba110913` |
| `V20_R15_LEAP_LEDGER.md` | `6e88935181ab8c83` |
| `tests/saturn/test_v20_r15_it28_widened_resolver.py` | `2b4a0b8cec011f23` |
| `tests/saturn/` file set (20 files) | set-digest `48304fc20293ffc0` |
| `tests/mars_v20/` file set (22 files) | set-digest `05446b3a22491c6c` |

Set-digest is `sha256` over `name || sha256(bytes)` for the sorted `*.py` of the directory —
so it names the **file set a count was taken over**, which a HEAD SHA cannot do for a tree git
does not contain.

---

## 1. REPAIR 1 — THE BASELINE. Retrospective: UNOBTAINABLE. Prospective: TAKEN.

### 1.1 Why the it.27 pre-edit baseline cannot be recovered, measured

`[RUN]` 12:52:10Z. The two suites are **not in git**:

| directory | files | tracked by git |
|---|---|---|
| `tests/saturn/` | 20 `.py` | **4** |
| `tests/mars_v20/` | 22 `.py` | **0** |

`git ls-files tests/mars_v20` returns nothing. `git ls-files tests/saturn` returns four files,
none of them a V20 R15 instrument. **The pre-edit content of every file it.27 edited exists in
no store** — not in the index, not in a stash, not in a reflog, because those files were never
in the object database. There is no `git show` that can produce them, and this office has no
other snapshot. A retrospective pre-edit run is therefore not "hard"; it is **not defined**.
This is the same species as it.26's HEAD-SHA finding — a provenance channel that cannot carry
the fact asked of it — and it is stated here rather than worked around.

### 1.2 What IS obtainable retrospectively, and it is most of the claim

mtime bounds the claim even where content does not. `[RUN]` 12:52:10Z, over the 40 failures of
the 12:50:40Z run, stamping each failing **module** with its digest and mtime, and taking
**12:25Z** as the it.27 open:

| suite | failing modules | mtime **before** 12:25Z | mtime **after** |
|---|---|---|---|
| `tests/saturn/` | 5 | **4** | 1 |
| `tests/mars_v20/` | 15 | **15** | 0 |

The one exception is `tests/saturn/test_v20_r15_it19_theory_digest.py`
(`sha=0a9864625e9cc8f6`, mtime `12:41:01Z`) — **the live subject-moved finding**, which is the
one it.27 already identified as this iteration's. Every other failing module's bytes were last
written before it.27 opened; the oldest are `2026-08-31T05:07:04Z`.

The obvious hole in an mtime argument is the code underneath. Measured, same run:

| tree | files | newest mtime |
|---|---|---|
| `ceq/` | 43 | `2026-08-31T19:17:47Z` |
| `scripts/` | 48 | `2026-09-01T22:28:16Z` |
| `lean/` | 4837 | `2026-08-31T13:40:27Z` |
| `scale/` | 105 | `2026-09-02T10:38:30Z` |

**No `.py` or `.lean` file under any of the four has been written since 10:38:30Z**, which is
before it.27 opened. So for 39 of 40 failures **both** the asserting bytes and the source bytes
predate the iteration. That is not name inspection; it is two measured mtime bounds.

**The residual, and it is real.** mtime bounds files, never assertion outcomes, and these
instruments read `.md` files as well as code — and the `.md` corpus **did** move this
iteration. A node reading `V20_R15_THEORY_TABLE.md` or `V20_R15_JOURNAL.md` could have flipped
without one byte of `.py` moving. So the retrospective claim is now: *39 of 40 failures are
attributable to bytes that predate it.27 **in code and in test source**, with the doc corpus
unbounded.* That is strictly stronger than it.27's sentence and strictly weaker than a run.

### 1.3 The recording discipline — stated, and exercised this iteration

What would have had to be recorded at the time, and now is:

1. **Take the run before the first edit of the iteration, not after the last.** A baseline is a
   recording discipline, not a reconstruction technique.
2. **Stamp it with the file-set digest, not a HEAD SHA** — for a tree git does not contain, a
   HEAD SHA names bytes the count was not taken over. `48304fc20293ffc0` / `05446b3a22491c6c`
   above are that stamp.
3. **Re-take after the edit and publish the delta with the edit named**, so the attribution is
   a difference rather than an argument.

Exercised, this iteration, with this office's **only** edit between the two runs — the
re-declaration of the freeze block in §2:

| run | dated | result |
|---|---|---|
| **pre-edit** | `[RUN]` 12:50:40Z | **40 failed, 251 passed** |
| **post-edit** | `[RUN]` 12:53:02Z | **39 failed, 252 passed** |

Δ = one node: `tests/saturn/test_v20_r15_it19_theory_digest.py::test_the_declared_cells_digest_matches_the_table_at_head`.
Nothing else moved in either direction. **That is what a pre-edit baseline buys, and it is why
the it.27 sentence was the weakest one in the record.**

The nine `tests/saturn` failures of the pre-edit run, named, since it.27 counted them without
naming them: `test_journal_path_is_discoverable` ×3 (parametrised over `scale/{m2,r2,s2}_units.py`),
`test_r10_it2_spotcheck_reds` ×3, `test_v20_r15_it19_theory_digest` ×1 (**now green**),
`test_v20_r15_wing_rubric::test_every_annex_run_instance_has_a_producer_in_the_tree`,
`test_v20_r15_wings_distinct_percell::test_the_it2_published_floor_of_030_survives_the_corrected_gate[8]`.
The remaining 31 are `tests/mars_v20`. **it.27's "9 + 40" was 9 saturn inside a combined 40**,
not 9 plus 40 — an arithmetic slip in this office's own filing, corrected here.

**Close of iteration, `[RUN]` 12:58:14Z — `tests/saturn` alone: 9 failed, 191 passed.**
The same nine as the pre-edit run **by count and not by composition**: the theory-digest
failure is repaired and RED 3 of §3 took its place. Composition, not the count, is the fact —
which is the whole reason the delta above is published beside the totals.

---

## 2. REPAIR 2 — `THEORY-CELLS-SHA256` RE-DECLARED, WITH ITS SUBJECT STAMP

Deferred at it.27 as *"correct after JUPITER's edits land, not during them."* They have landed.
Re-declared under this office's own it.27 discipline — `stat → read → stat`, `torn` is a
**REFUSAL** and not a verdict off stale bytes.

**Subject read, `[RUN]` 12:51:57Z:**

```
V20_R15_THEORY_TABLE.md sha256=942e4208893444cd mtime_ns=1788352690049018500 size=43461 torn=False
mtime = 2026-09-02T12:38:10.049019Z
```

`942e4208893444cd` is the digest MERCURY verified the table at, and the read was **untorn** —
the table sat still across it, so a comparison to the freeze is available.

**Declared, in `V20_R15_IT19_SATURN.md`'s `theory-cells-freeze` block:**

```
THEORY-CELLS-SHA256 = bef437d4e5d2612d878891f28e7f973f8208684b55e376d9dd35c91565e47d40
```

| cell | keyed digest | vs it.19 |
|---|---|---|
| Q1/W1 | `175765879224b808` | **moved** |
| Q1/W3 | `be071fc4772b9e22` | — |
| Q2/W1 | `a2701819199bde6c` | — |
| Q2/W3 | `84c403bb34523303` | **moved** |
| Q3/W1 | `bc5b4074f96286a0` | **moved** |
| Q3/W3 | `b419ed3666872953` | — |
| Q4/W1 | `667cfdf878ed6205` | — |
| Q4/W3 | `b486e1a00b63e9f7` | — |
| Q5/W1 | `c243b350d11d6f47` | — |
| Q5/W3 | `b06897865f5e4590` | — |
| Q6/W1 | `fd11b3f5715a00ba` | **moved** |
| Q6/W3 | `3b01ae50adf21e26` | **moved** |

Five cell bodies moved and seven did not — **the same five the digest read RED on at it.26 and
it.27**, which is the check that this is a re-freeze of JUPITER's landed edits and not a
silent re-baseline of something else. The superseded whole-cells digest was
`071f88264116d02a…`. The subject stamp is written **into the freeze block itself**, not only
into this record: a freeze that does not name the bytes it froze cannot be told from a stale
verdict, which was the whole of REPAIR 2 at it.27.

`CITATION-POPULATION = 129` was dropped from the block. Nothing parses it (`grep -rn
"CITATION-POPULATION" --include=*.py .` → no hits) and JUPITER's live coverage reading is
`118 of 131`; republishing a stale 129 beside a fresh digest would be exactly the
one-number-republished defect L-M4 files against the round.

**Verified `[RUN]` 12:51:31Z:** `tests/saturn/test_v20_r15_it19_theory_digest.py` +
`tests/saturn/test_v20_r15_it27_subject_provenance.py` — **11 passed, 0 failed**, including
`test_the_declared_cells_digest_matches_the_table_at_head` and
`test_the_planted_negative_fires_against_the_declaration`. The planted negative still fires,
so the freeze is not green because it stopped looking.

**What the re-declaration does not do.** It authenticates *which* bytes, never that those bytes
are right. `bef437d4…` says the twelve cell bodies at `942e4208…` are the ones this office
froze at 12:51:57Z. It says nothing about whether MERCURY's three replacements were the correct
ones — that is MERCURY's node, not this digest.

---

## 3. REPAIR 3 — THE WIDENED RESOLVER: A MEASURED NULL, FILED AS ONE

`tests/saturn/test_v20_r15_it28_widened_resolver.py`, 5 nodes, **4 pass and RED 3 stands.**

it.27's boundary was that W3's pin rests on **one** resolving citation against W1's three, and
the named route was to widen past `ceq/arm_*.py` into `lean/`, `scripts/` and `tests/`. The
widening is built, with the it.27 content rule carried across unchanged — a citation resolves
only when the cited line lands **inside a symbol body** and that body names **exactly one** arm.
Applying the weaker rule to the wider set would have been an amnesty, not a resolver.

### The widened channel's own check, and it came back EMPTY — `[RUN]` 12:55:12Z

Eight citations outside `ceq/arm_*.py` exist in the ledger. **None resolves.**

| row | wing | citation | why it does not resolve |
|---|---|---|---|
| L-9 | W1+W3 | `scripts/v15_r1.py:137` | outside every symbol body |
| L-10 | W1 | `lean/CEQ/V16Domain.lean:129` | lands in `pathProd_eq_zero_iff`; **body names neither arm** |
| L-11 | W1+W3 | `scripts/v15_r1.py:17-19` | outside every symbol body |
| L-13 | W1 | `scale/negation_scope.py:300-304` | lands in `equilibrium_oracle`; **body names neither arm** |
| L-14 | **W3** | `tests/jupiter/test_v20_r15_it9_q6.py:155` | outside every symbol body — a comment recording a node **killed at it.11** |
| L-16 | W1 | `tests/jupiter/test_v20_r15_it12_constants.py:12` | outside every symbol body — module docstring |
| L-16 | W1 | `ceq/kdata.py:475` | outside every symbol body |
| L-17 | **W3** | `tests/jupiter/test_v20_r15_it12_constants.py:11` | outside every symbol body — module docstring |

### The count for W3, and RED 3 verbatim — `[RUN]` 12:55:12Z

```
E       AssertionError: W3's binding rests on 1 resolving citation(s) after the widening to
E       lean/, scripts/ and tests/ -- [('narrow', 'symbol ArmPL.forward')]; W1 has 5 --
E       [('narrow', 'line ceq/arm_smprime.py:163-172'), ('narrow', 'line ceq/arm_smprime.py:572-577'),
E        ('narrow', 'symbol ArmSMPrime.forward'), ('narrow', 'symbol path_product'),
E        ('narrow', 'symbol zero_hop_mask')].
E       assert 1 >= 2
```

**W3 = 1. W1 = 5.** The widening bought **zero** for either wing, so the count for W1 rose from
three to five only because the narrow census was taken over every single-wing row rather than
the three rows it.27 named by hand — a bookkeeping correction, not a widening gain.

**RED 3 is left RED deliberately.** The defect it.27 named is unrepaired, and the instrument
that says so must fail while it is unrepaired. The repair is **a citation the ledger does not
yet carry** — a W3 row naming a symbol whose body names `arm_pl` — and it belongs to the ledger
office, not to a wider regex. The two W3 candidates fail for exactly the reason it.27 gave for
`ceq/arm_pl.py:1`: a line number that lands in prose is not a resolution.

`test_the_widened_channel_resolves_nothing_for_either_wing` pins all eight citations **with
their reasons**, so a citation that later starts resolving fires the node rather than quietly
enlarging the pin. `test_a_body_that_names_both_arms_resolves_to_neither` is the planted
negative: the widened set is full of files discussing both arms, and a resolver taking the
first match would bind a wing to whichever arm is mentioned first. It refuses.

**Two ways this widening could be wrong.** The Lean side is a **line scanner**, not an AST —
there is no Lean AST in this tree — so a declaration body can over-extend across trailing
comments. That makes it **more** permissive than the Python side, so it cannot manufacture the
null; it could only have hidden one. And the arm-naming rule is a regex over the body text: a
body that reaches `arm_pl` through an alias this office did not enumerate reads as naming
neither arm. Both are stated because the finding is a null and a null is the easiest result to
get by accident.

---

## 4. `dead_literals()` — CENSUS TAKEN, ADOPTION NOT REACHED

The coordinator's `overturns:` field is now on all 34 index rows and
`tests/jupiter/test_v20_r15_it27_star_lands_and_overturns.py:191` exposes `dead_literals()`, so
a stale-claim grep can subtract what the index declares dead.

Measured, `[RUN]` 12:56:11Z: **two** instruments of this office scan the `.md` corpus by glob
rather than by named path — `tests/saturn/test_v20_r15_freeze_manifest.py` and
`tests/saturn/test_v20_r15_wing_rubric.py`. The second is one of the nine saturn failures
(`test_every_annex_run_instance_has_a_producer_in_the_tree`), so whether its RED is a live
defect or a literal the index already declares dead is exactly the question `dead_literals()`
answers — and it is the question this office has been answering by name inspection, which is
the habit REPAIR 1 exists to break.

**Not reached this iteration.** Named here with its two candidates so it is a queued
measurement and not a discovery someone re-makes at it.29.

---

## 5. WHAT WAS NOT REACHED

1. **`dead_literals()` adoption** in the two glob-scanning instruments above — census taken,
   adoption not attempted.
2. **The doc-corpus half of REPAIR 1's residual.** The 39 pre-existing failures are bounded in
   code and in test source, not in the `.md` files several of them read, and the `.md` corpus
   moved this iteration. Bounding it needs a per-node subject census — every `.md` an
   instrument reads, stamped the way §2 stamps the theory table.
3. **W3's second citation.** Named as the repair and not made: it is a ledger edit, and this
   office does not write the ledger.
4. **The nine saturn REDs themselves.** Measured, attributed and named — none diagnosed.
