# Pre-registration hole audit — all five documents, before R9

**Written 2026-08-30, R9 iteration 1, by the IRENE seat. No R9 number exists.**

`E_LADDER_PREREGISTERED_READING.md:183-188` records the failure mode this audit
looks for, in the author's own words:

> Row H was added at 13:35, after rows A–G and before any number from this run
> existed, because A–G did not cover it. [...] A pre-registration with a hole in
> it is not a pre-registration, and the hole is patched before the data rather
> than after.

Row H was found by asking one question: **construct the outcome that falls
through every row.** This document asks that question of all five
pre-registrations and of the one place where a table is enforced in code.

**What this document does not do.** It does not edit any of the five
pre-registrations. Three of them are post-data: the E ladder is complete
(`results/e_ladder_reading.txt` ends `ladder complete: True`), M2's control
finished, and the M3 quintuple's money run is journalled. Patching a table
after its data has landed is the thing the whole discipline exists to prevent,
and the patch rows below are therefore written **beside** the documents, marked
for whether they are still live. Their owners decide whether to adopt them.

---

## Ranking — worst fall-through first

The rank is how badly a fall-through would be misread, not how likely it is.
A hole that prints a **false sentence** outranks a hole that prints nothing,
because nothing is visibly nothing.

| rank | # | document | the outcome that falls through | what gets printed instead |
|---|---|---|---|---|
| 1 | H-1 | `E_LADDER` §6 C/D + `e_ladder.py:280-293` | every CI covers zero; `\|delta\| ≥ 0.027260` at `t*=1` or `t*=2` **only** | row **C**, with the sentence *"every \|delta\| is below the 13-seed resolution"* — which is false |
| 2 | H-2 | `E_LADDER` §6 A + `e_ladder.py:248-262` | CI excludes zero and is **negative** at `t*=1`, excludes zero and is **positive** at `t*=32` | row **A** or **B**, with A's sentence *"Compatible with zero at t\*=1"* — which is false |
| 3 | H-3 | `M2_TRAINED` §readings B/C | trained slope `> −0.3` **and the control is flatter than the arm** | nothing — and this is the ordering the existing random-init table already shows |
| 4 | H-4 | `M2PRIME` §outcomes A/B/C | determined fraction neither `≈ 0` nor "high" | nothing; "high" is never given a number anywhere in the file |
| 5 | H-5 | `M2_TRAINED` §"What would convict the instrument" | the cheap instrument test **convicts** the instrument | nothing; rows A–D all presuppose a surviving instrument |
| 6 | H-6 | `M3_QUINTUPLE` §5 rows 1–2 | `settled` beats `twin` and **loses to** `argmax` | nothing |
| 7 | H-7 | `M2` §readings A/B/D | control decays but does not reach "near zero" | nothing; no threshold for "near zero" is stated |
| 8 | H-8 | `M2PRIME` §outcomes | R2's second kill clause, *"or constraint destroys training"* | nothing; quoted once at `:21` and never given a row |

Four **drifts** — a row whose enforcing branch does not match it — are listed
separately in §2. They are not fall-throughs; they are rows that fire on the
wrong condition.

---

## 1. The holes, one at a time, with the patch row

### H-1 — `E_LADDER` row C's universal is enforced as an existential

**The rows.** §6 row **C** fires on *"CI covers zero at every rung **and**
`|delta| < 0.027` at every rung"*. Row **D** fires on *"CI covers zero at every
rung but `|delta| ≥ 0.027` at `t* = 8` or `t* = 32`"*.

**The fall-through.** Every CI covers zero, `|delta| ≥ 0.027260` at `t* = 1`
(or `t* = 2`) and `< 0.027260` at both `t* = 8` and `t* = 32`. Row C's
universal is violated, so C does not fire. Row D names only the two deep rungs,
so D does not fire. Rows A, B, E, F and H all need a CI that excludes zero.
Row G is about the absolute bar and is silent. **Nothing fires.**

**What the code does with it.** `scale/e_ladder.py:280-281` builds

    big = [t for t in ("e3_t8", "e3_t32")
           if t in live and abs(live[t]["delta"]) >= RESOLUTION_13]

and falls through to `return ("C", ...)` at `:288`, whose sentence asserts
*"every `|delta|` is below the 13-seed resolution 0.027260"*. The shallow rungs
were never inspected. The verdict prints a universal the branch did not check.

**How close this came to firing.** In the completed ladder, `e3_t1` reads
`delta = −0.036025` with CI `[−0.118936, +0.062209]` — a CI covering zero and a
`|delta|` of `0.036025`, above the `0.027260` floor. The C branch was not
reached only because `e3_t2`'s CI excluded zero, and G pre-empted before that.
The trap is one rung's CI away from having been sprung, in the real data.

**Patch row — still live for any future ladder, NOT retrofitted to the complete one.**

> **D′** | CI covers zero at every credited rung, and `|delta| ≥ 0.027260` at
> **any** rung | **UNDERPOWERED, NOT A KILL, and the rung is named.** Identical
> licence to D. D's restriction to `t* = 8, 32` was a statement about where an
> effect was expected, not about where the resolution floor applies; the floor
> is a property of `N = 5` and applies at every rung. Route: seeds 5 → 13 at the
> named rungs only. Row C requires `|delta| < 0.027260` at **every** rung and
> may not be reported unless that has been checked at every rung and printed.

The one-line code repair that makes D′ mechanical is to widen `:280` from
`("e3_t8", "e3_t32")` to `RUNGS`.

---

### H-2 — `E_LADDER` row A's "covers zero at `t*=1`" is never checked

**The row.** §6 row **A** fires on *"CI excludes zero and is positive at
`t* = 32`, **and covers zero at `t* = 1`**, and the point estimate is monotone
non-decreasing"*.

**The fall-through.** CI excludes zero and is **negative** at `t* = 1` (settled
strictly loses inside every arm's hop budget), and excludes zero and is
**positive** at `t* = 32`. Then: A needs `t*=1` to cover zero — no. B needs a
non-monotone ladder — not implied. E and F need a *positive* exclusion at
`t*=1` — no. H needs settled to be positive nowhere — it is positive at
`t*=32`. C and D need every CI to cover zero — no. G is silent. **Nothing fires.**

**What the code does with it.** `verdict()` reaches the `won("e3_t32")` block at
`:250` whenever `won("e3_t1")` is false. `won` is `ci_lo > 0.0`; a *negative*
exclusion at `t*=1` leaves `won("e3_t1")` false, so the block is entered and
returns A (if monotone) with the sentence *"Compatible with zero at t\*=1"*.
The branch never distinguishes "covers zero" from "excludes zero downward".

**Why this outcome matters more than row A's own.** A negative exclusion at
`t*=1` plus a positive exclusion at `t*=32` is a **cleaner** dose-response than
row A asks for: the iteration costs where it cannot help and pays where the
budget binds. Row A demands the weaker signal at `t*=1` and has no row for the
stronger one, so the best result the ladder could have produced would have been
reported under a sentence contradicting it.

**Patch row — still live for any future ladder.**

> **I** | CI excludes zero and is **negative** at `t* = 1`, and excludes zero and
> is **positive** at `t* = 8` or `t* = 32` | **CROSSOVER.** Not row A and not row
> F. Settling is a measurable *cost* inside the hop budget and a measurable
> *gain* outside it. This is a stronger dose-response than A, because the
> shallow end is not merely compatible with zero but significantly against, and
> the sign change is itself the dose. It licenses what A licenses, and the claim
> sentence must carry the shallow-rung cost in the same sentence as the deep-rung
> gain.

Row A must additionally check its own antecedent: `A` requires
`live["e3_t1"]["ci_lo"] <= 0.0 <= live["e3_t1"]["ci_hi"]`, not merely
`not won("e3_t1")`.

---

### H-3 — `M2_TRAINED` has no row for a control flatter than the arm

**The rows.** §readings **B**: *"Trained slope `> −0.3` AND the unsigned/softmax
control is steeper in the same table"*. **C**: *"Trained slope `> −0.3` but the
control is comparably flat"*.

**The fall-through.** Trained slope `> −0.3` and the control is **flatter** than
the arm. Not "steeper" (B), not "comparably flat" (C). A clears the bar in the
other direction. D is about seed count. **Nothing fires.**

**Why this is the third-worst hole and not the eighth.** The document's own
measured table already exhibits that ordering. `:30` reads *"Fitted slope
**−1.298** [...] Dense control **−1.088**, so routing is worse than not
routing."* The control is flatter than the arm at random init by `0.21`. The
outcome with no row is the one the existing data points at.

**Patch row — LIVE. No trained operator has been measured
(`M2_TRAINED_PREREGISTERED_READING.md:3`), so this table is still pre-data.**

> **B′** | Trained slope `> −0.3` **and the control is flatter than the arm by
> more than the seed-to-seed spread of either** | **ROUTING IS A COST.** Both
> clear the bar, so neither is dilution-limited, and the arm is the worse of the
> two at matched parameters. This is not outcome C — C says the flatness is a
> property of the probe, shared. Here it is not shared: the probe is flatter
> without the routing. It reads as a regression in the arm, it belongs to the new
> item, and it reproduces the random-init ordering at `:30` rather than
> overturning it.

"Comparably flat" in row C must additionally be given a number before the run.
The document supplies no threshold, and §2 below records that as a separate
defect.

---

### H-4 — `M2PRIME` splits a continuous quantity into two branches

**The rows.** **A** fires on *"Determined fraction ≈ 0"*. **B** and **C** both
open with *"Determined fraction high"*. **D** is *"high but so is softmax's"*.

**The fall-through.** Determined fraction `0.45`. Not `≈ 0`; not "high" by any
reading that would also let 1.0 be "high". **Nothing fires**, and no number
anywhere in the file separates the two.

This is the cleanest hole in the five: a three-valued predicate with two
branches and no cut point. R2's own kill text at `:21` says *"determined
fraction ~0"*, so the low end inherits the same missing number.

**Patch row — LIVE if R2 has not run.** The file says *"no R2 number yet in
existence"* at `:3`; `results/` holds no `r2_*` reading, and
`scale/r2_units.py` exists.

> Fix the cut before the run. Row **A** fires on `determined ≤ 0.05`; rows
> **B/C** fire on `determined ≥ 0.60`; and:
>
> **E** | `0.05 < determined < 0.60` | **NOT READABLE.** The sign pattern
> determines *some* of the block and not the rest, which is neither the
> magnitude-independence R2 asks for nor its absence. No GREEN, no RED. The
> reading owed is the **distribution** of the determined fraction across draws,
> not its mean, because a bimodal 0.45 and a unimodal 0.45 are different objects
> and the mean cannot tell them apart.

Both cut points must be fixed against the §CALIBRATION exhibits — softmax at
`1.0` and a random sign pattern at the same density — **before** the first R2
number, or they are cut points chosen to flatter the data.

---

### H-5 — `M2_TRAINED`'s rows all presuppose an instrument that survived

**The commitment.** `:124` — *"The instrument test runs first, and it is cheap.
If the instrument survives, M2's RED stands regardless of any trained number."*
`:68-74` defines what would convict it: the flip statistic being dominated by
the Gaussian draw's scale rather than the operator's structure, *"so it would
read −1.298 for any operator including ones known to be context-stable."*

**The fall-through.** The instrument **is** convicted. Rows A–D are all
statements about a trained slope read by that instrument, and every one of them
is void. Nothing in §readings says so, and the natural misreading — the one this
whole file exists to prevent — is to keep reading A–D on a convicted instrument
because the rows are still sitting there.

**Patch row — LIVE, and it costs nothing: the instrument test has not run.**

> **0** | The instrument test at `:68-74` **convicts** the instrument — the flip
> slope reads at or near `−1.298` for an operator known to be context-stable |
> **ROWS A–D ARE VOID AND ARE NOT READ.** A statistic that returns the same
> slope for a stable operator is not measuring the operator. M2's RED is
> overturned by the contract's own clause at `:59-60` — *"a RED is overturned
> only by convicting the INSTRUMENT"* — and the trained run is not authorised
> until a replacement statistic has its own calibration in both directions. This
> row is evaluated **first**, before any of A–D is looked at.

---

### H-6 — `M3_QUINTUPLE` has no row for `argmax` beating `settled`

**The rows.** §5 row 1: settled beats twin **and** beats argmax. Row 2: settled
beats twin **but ties** argmax. Row 3: settled ties twin. Row 4: settled loses
to twin. Row 5: the absolute bar.

**The fall-through.** Settled beats twin and **loses to** argmax — the CI
excludes zero in argmax's favour. Rows 1 and 2 both condition on settled beating
or tying argmax. Rows 3 and 4 condition on the twin contrast going the other
way. **Nothing fires**, and §4 names `settled` vs `argmax` as the **headline
caveat**, so the fall-through is on the axis the document says is decisive.

`results/m3_quintuple_v2.jsonl` makes this unlikely — `argmax − softmax` reads
`−0.118456` — but "unlikely" is what row H was too, at 13:35.

**Patch row — POST-DATA for the `k=8` money run. Live for the `k=16`/`k=32`
follow-on bucket at §7.2, which has not run.**

> | `settled` beats `twin` but **loses to** `argmax` | the mixture is worse than
> its own one-hot collapse while still beating the unsettled mixture. That is not
> a routing claim and not an equilibrium claim: it says the settling improves the
> gate's *ranking* while the averaging destroys what the ranking bought. Reported
> as such, with `argmax` as the shipped cell if it also clears the absolute bar,
> and no positive claim for `settled`. |

§5's last row (the absolute bar) also needs its **precedence** stated: it says a
failing cell is *"credited with nothing, whatever its contrast says"* but not
whether it pre-empts rows 1–4. `E_LADDER`'s row G has the same ambiguity and
`e_ladder.py` resolved it silently in code — see §2, D-3.

---

### H-7 — `M2`'s three readings do not partition the control's range

**The rows.** **A**: control finishes *"near zero"* at `s=2048`. **B**: control
*"rises"* at `s=2048`. **D**: control *"flat and nonzero"*.

**The fall-through.** The control lands at, say, `0.008` — down from `0.024658`
at `s=8` and up from `0.000488` at `s=1024`. It has not risen above its own
trend (not B), it is not flat (not D), and calling `0.008` "near zero" against a
claim arm at `0.029907` is a judgment, not a reading. **Nothing fires
mechanically.**

No threshold for "near zero" appears in the file. The material to fix it is
already there — `:42` records the current trajectory as `0/1024`, CP upper
`0.00292` — so the cut could have been the CP upper bound and was not.

**POST-DATA.** The control finished. Recorded as a structural defect for the
next document of this shape, not as a patch to this one.

---

### H-8 — `M2PRIME` never gives R2's second kill clause a row

R2 as quoted at `:21`: *"**Kill:** determined fraction ~0 in trained blocks, **or
constraint destroys training**."* Outcomes A–D are all about the determined
fraction. The training-destruction clause is quoted once and never returns.
`:104` re-quotes only the first clause when the defect section discusses the
"trained" qualifier, so the second clause is not covered by that discussion
either.

**Patch row — LIVE.**

> **F** | The SNS constraint is imposed and the constrained block's training
> loss fails to reach the unconstrained block's within its own seed spread |
> **R2 RED by its second kill clause**, independent of the determined fraction,
> and the determined fraction is not reported as a screening number from a run
> whose training did not converge. The comparison is against the *unconstrained*
> block at matched parameters and matched steps, both trained in the same run.

---

## 2. Drift — rows whose enforcing branch does not match them

`E_LADDER_PREREGISTERED_READING.md` §6 is the only outcome table in the five
that is enforced in code, by the `if`/`elif` chain in
`scale/e_ladder.py:217 verdict()`. The correspondence was checked row by row.
**Four rows have drifted.** Two of them are H-1 and H-2 above and are not
repeated here.

The file's own docstring at `:23` says *"`OUTCOMES` below is a transcription of
[§6]"*. **There is no `OUTCOMES` symbol in the file.** `grep -n OUTCOMES
scale/e_ladder.py` returns that docstring line and nothing else. The promised
machine-readable transcription does not exist, and the only encoding of the
table is the imperative chain — which is where all four drifts live. That is
`FINDINGS.md` B7's line-reference rot in a load-bearing position.

| # | row | document says | code does | direction |
|---|---|---|---|---|
| D-1 | **F** | §4 `:130-132`: *"rows **A**, **C** and **F** all quantify over every rung and are unavailable until every rung is in"*. Docstring `:29-31` repeats it verbatim | `:241-244` fires F on `won("e3_t1") and (lost("e3_t8") or lost("e3_t32"))` with **no `cur["complete"]` guard**. A is guarded at `:252`, C at `:276`. F is not | **unsafe** — F is a theory-death verdict and can be claimed on two rungs |
| D-2 | **H** | *"CI excludes zero and is negative at some rung, and **is never positive at any rung**"* — a universal over rungs | `:267` fires H on `any(lost(t)) and not any(won(t))` with **no `complete` guard**. On a partial ladder an unrun rung could have been positive, and the universal is unverifiable | **unsafe** — same class as D-1, in the row that was added to patch a hole |
| D-3 | **G** | *"**that rung** is credited nothing in either direction [...] Printed, **never read as a verdict**"* — per-rung, and explicitly not a verdict | `:226-231` returns `("G", ...)` as **the** verdict for the whole ladder as soon as any single rung is uncredited, pre-empting every other row | **scope change**, defensible under §7b.1 (*"the ladder is unreadable at this budget"*) but contradicting row G's own last sentence. §7b.1 and row G disagree inside the document, and the code picked one |
| D-4 | **D** | §4 `:131-132`: *"If only the endpoints land, the strongest reachable readings are **B**, **D** and **E**"* — D is reachable on a partial ladder | `:276-279` returns `"--"` on any incomplete ladder before D is reached, so D is complete-only | **safe** — the code is stricter than the document. Reported so the pair is not later mistaken for agreement |

D-1 and D-2 answer deliverable 1's fourth question directly. `scale/e_ladder.py`
`:29-31` and `:252`/`:276` make rows **A** and **C** refuse to fire on partial
data, exactly as claimed. **Row F does not, and row H does not.** Two of the four
rows carrying a universal over rungs are unguarded, and one of them is row H
itself.

### Rows that are not mechanically evaluable

Deliverable 1's third question, applied to the four unenforced documents. None
of the twenty-odd rows in `M2`, `M2PRIME`, `M2_TRAINED` and `M3_QUINTUPLE` has an
enforcing branch anywhere in the tree — `grep -rln` over `*.py` for the four
filenames returns citations in docstrings and report strings only
(`scale/m3_quintuple.py:3,591`, `scale/capability_table.py:256,303`,
`scale/eprocess.py:89`, `scale/s2_units.py:16`, `inspector.py:271`,
`scale/chase_struck_coverage.py:125-132`). `inspector.py:335-372` parses
`M2_TRAINED_PREREGISTERED_READING.md`, but it parses the **measured-state table**
in §"What is already measured" and re-derives the slope from those counts. It
does not touch rows A–D.

The predicates that cannot be evaluated without a judgment call, each needing a
number fixed before its run:

| document | predicate | missing |
|---|---|---|
| `M2` A | "near zero" | a threshold. `:42`'s CP upper `0.00292` is the obvious candidate and is not used as one |
| `M2` B | "rises" | rises against what — the `s=1024` point, the fitted trend, or its own CI |
| `M2` D | "flat and nonzero" | both a flatness tolerance and a nonzero floor |
| `M2PRIME` A/B/C | "≈ 0" / "high" | two cut points — see H-4 |
| `M2PRIME` C vs D | which applies | D's antecedent (*"softmax's is also high"*) is **always true**: `:80` and `:91` state softmax's determined fraction is `1.0` by construction. So D fires whenever B or C's antecedent holds, and no precedence is stated. C is unreachable as written |
| `M2_TRAINED` B/C | "steeper" / "comparably flat" | a tolerance; see H-3 |
| `M2_TRAINED` A vs D | which applies at one seed | A's condition does not exclude a single seed and D's does not exclude A. No precedence. D taking precedence is the safe reading and is not written down |
| `M3_QUINTUPLE` §5 all | "beats" / "ties" | **evaluable** — §4 `:104-108` fixes the trichotomy `ci_lo > 0` / `ci_hi < 0` / NO DIFFERENCE. This is the one set of rows in the four that is mechanically decidable, and it is decidable because the statistic was written into the same document |
| `M3_QUINTUPLE` §5 row 5 | precedence over rows 1–4 | unstated; see H-6 |

`M3_QUINTUPLE` §4 is the pattern the other three should have copied: define the
verdict function in the pre-registration, then let the rows refer to it.

---

## 3. What this audit could not check

The `E_LADDER` correspondence was read against `scale/e_ladder.py` at
`worktree-agent-adec25a08af78a4c8`; `verdict()` was **not executed** on
synthetic ladders that exercise each branch, because this seat does not run
cells. Every drift in §2 is a `READ` of the branch condition against the row
text, not a `RUN` that watched the wrong row fire. D-1 and D-2 are the two where
that matters most: the guard is *absent*, which a read establishes, but whether
some earlier branch happens to catch the partial case first was traced by hand
through the chain rather than by executing it. The four unenforced documents
have no branch to compare against at all, so every judgment about them is a
reading of English against English. The four `M2`/`M2PRIME` patch rows are
marked LIVE on the strength of each file's own statement that its run had not
happened plus the absence of a matching `results/` artifact; neither is proof
that no number exists somewhere off-journal.
