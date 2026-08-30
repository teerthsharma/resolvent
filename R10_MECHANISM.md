# The surface-proxy defect — R10's one mechanism, twenty-four times

Round 10 was budgeted as a triage: audit the tree, spot-check the audit, move the
dead rows. It found one defect, repeatedly, in instruments written by four
different seats across three iterations — including in four instruments written
specifically to catch earlier instances of it.

This file is the catalogue. It exists because the instances are scattered across
per-iteration reports and the pattern is worth more than any of them.

## The shape

> A rule keys on a **surface feature that correlates with** the property it names,
> instead of on the property itself. It then fires — or fails to fire — on the
> correlation rather than the thing, and because the correlation is usually right,
> the rule looks sound until something lands in the gap.

The correlation is what makes it survive review. Every one of these rules is
*mostly* correct, and several were correct on the day they were written. What they
share is that nothing in them can distinguish "the property holds" from "the proxy
happens to be present", so the first case where the two come apart is silent.

## The first fourteen

| # | rule | keyed on | the property it names | found by | cost |
|---|---|---|---|---|---|
| 1 | P1 vacuity | *who* built the input | what path the input takes | MARS it.1 | would have attic'd the V-1 binding control |
| 2 | P3-literal | a literal path string in the source | the path the code constructs | SATURN it.1 | classed 2 live producers ORPHAN |
| 3 | P1′ vacuity | *where* the plant enters | what the instrument claims | SATURN it.1 | classed the prior-art defence vacuous |
| 4 | P2/P3 orphan | *who imports it* | whether it is reachable | NEPTUNE it.1 | none — survived audit on a 3-part conjunction |
| 5 | refutation classifier | the name prefix `test_claim_*` | the in-band `CLAIM AS WRITTEN` banner | MERCURY it.2 | 3 banner files invisible; 2 of 3 routed failures |
| 6 | "no python importer" | a search that could not find what it sought | the import graph | SATURN it.2 | wrong on 12 of 18 rows, 135 edges; 0 dispositions |
| 7 | "ran ALL-GREEN" | the process exit status | pass vs recorded finding | SATURN it.2 | **7 rows** — `xfail(strict=True)` exits 0 |
| 8 | hankel ATTIC conjunction | the same blind prefix as #5 | `test_mustfire_*` nodes | HOUSE it.2 | would have removed `ceq.hankel`'s only calibration |
| 9 | "0 results artifacts" | absence of a **file** | absence of a **producer** | HOUSE it.2 | 1 row; the module was unrun, not dead |
| 10 | HOUSE's import guard | git **index** membership | presence in the **tree** | Inspector it.2 | the guard stopped reporting a live defect |
| 11 | argmin-equality test | argmin over a **truncated grid** | the location of the minimum | MERCURY it.3 | the axis reported its own boundary as an answer |
| 12 | HOUSE's draw header | the **stratum** count | the **rule** count | MARS it.3 | advertised 1.0000 coverage; delivered 20 of 61 |
| 13 | HOUSE's ledger check | a literal-string **grep** | the parsed **event** | Inspector it.3 | reported 0 of 16 reds present |
| 14 | agent-name filter | one **capitalisation** | the agent | HOUSE it.3 | `Cameron` 108 against `cameron` 9,054 — a filter on the capitalised form sees 1.2% |

> The `cameron` figure above is a **live count** and grows every time any seat logs.
> Measured 8,940 when the instance was found and 9,054 twenty minutes later; the
> ratio is the stable part, not the numerator. Quoting a live count in the catalogue
> of this very defect is instance 14 committed inside its own row — a number keyed on
> "what the file says now" standing in for "what was measured". Left visible rather
> than pinned, because the correction is the shortest available statement of the
> shape.

## What the catalogue is actually for

**Four of the last five were committed by instruments written to catch the earlier
ones.** #10, #12, #13 and #14 are all HOUSE, writing guards against #1–#9 while
reproducing their shape. #8 is HOUSE applying MERCURY's own blind predicate in the
same iteration MERCURY proved it blind. That is not carelessness distributed
randomly; it is the defect being genuinely hard to see from inside the rule.

**The ones that cost nothing are as informative as the ones that cost rows.** #4
and #6 were both real instances and neither moved a disposition, because the rows
they touched were carried by a second independent leg. A conjunction survives a
blind conjunct; a single-predicate rule does not. That is the only structural
defence in the list that worked.

**The expensive one was #7**, and it is the cleanest illustration of the shape:
`xfail(strict=True)` returns exit code 0. "All tests passed" and "twelve findings
were recorded" are *the same byte*. A rule reading that byte cannot be made more
careful; it can only be replaced by one that reads the report.

## The three repairs that held

Not every instance was repaired by widening the proxy. The ones that stuck replaced
it:

- **#1 → P1′ → P1″.** P1 keyed on the author of the input, P1′ on where it enters,
  P1″ on what the instrument claims. Each replacement moved *closer to the property*
  rather than adding exceptions to the proxy.
- **#5 → T-b route 3.** `scale/red_by_design.py` keys on a present-tense declaration
  in band, in three bands narrowest-first, with a narrower non-declaring band
  overriding a wider declaring one. It carries two exclusions that are themselves
  the lesson: `RED-first` is authoring order, not present state; and a form inside
  backticks is a citation, not an utterance.
- **#10 → union of index and tree.** The proxy was not widened; the population was
  corrected to the one the hazard actually lives in.

## The tell

Every instance answers "how do I detect X?" with something cheap and adjacent. The
question that catches it before it ships is not "is this rule right?" — it is
almost always right — but:

> **Name a case where the proxy is present and the property is absent, or the
> property is present and the proxy is absent. If you cannot construct one, you have
> not yet understood which of the two you are measuring.**

Three of the instances above were caught by exactly that question being asked of a
rule that had already passed review: MARS on P1, MERCURY on the prefix classifier,
MARS on HOUSE's coverage claim.

## What `MISTAKES.md` already names, and the direction it does not

`MISTAKES.md` carries fourteen V-entries, V-1 to V-14, and they are a real taxonomy
— V-14 opens by distinguishing itself from V-7 (*"the one that survives rule 5 being
obeyed, which is what makes it a new type rather than another instance of V-7"*).
Two of them are surface-proxy defects:

- **V-7** — `journal_scan` read top-level keys of records nesting payloads under
  `value`, reported "zero readings above 1.0" against a true 22 of 68, and the false
  absence was used to strike a colleague's evidence. Keyed on *what the walk visited*
  standing in for *what the records contain*.
- **V-14** — `chase_struck_coverage` shipped a must-fire control, ran it first, and
  refused to proceed without it — and the scanner reached nothing, because the
  exclusion list was tested against absolute paths and `.claude` is a component of
  every worktree path. 366 candidate files became 0. Keyed on *the matcher firing*
  standing in for *the reach being non-empty*.

**Every V-entry describes an instrument that wrongly PASSES.** The class is vacuity:
a control with no rejection region, producing a false GREEN. That is the whole
registry, and the rules it ships are all of the form "pay for an absence with a
planted positive the search is required to find."

**The surface-proxy defect produces false REDS as well, and in round 10 that was the
expensive direction.** A rule keyed on a proxy does not merely fail to condemn the
guilty; it condemns the innocent, and nothing in the V-class rules catches that:

| instance | direction | cost |
|---|---|---|
| #1 P1 | false RED | would have attic'd the control binding V-1 itself |
| #3 P1′ | false RED | classed the prior-art defence vacuous; 12 of 27 condemned rows were live |
| #5 prefix classifier | false RED | 3 banner files invisible; 2 of the 3 routed "unpriced" failures |
| #7 exit status | false RED | **7 rows** — `xfail(strict=True)` exits 0, so recorded findings read as passes |
| #8 hankel conjunction | false RED | would have removed `ceq.hankel`'s only calibration |
| #9 "0 results artifacts" | false RED | attic'd a producer that was unrun, not dead |
| #10, #12, #13 | false GREEN | guards that stopped reporting live defects |

Seven of the first fourteen condemn rather than excuse. The registry's planted-positive rule
is powerless against those: a planted positive proves the instrument *can* fire, and
every one of these fired — enthusiastically, on the wrong thing.

**The missing rule, stated in the registry's own voice:**

> A rule that CONDEMNS needs a planted NEGATIVE: an instance that carries the proxy
> and is innocent of the property. A rule that has never spared anything is not
> discriminating, it is convicting, and its pass rate is a measurement of the tree
> rather than of the rule.

That is exactly what HOUSE demanded of T-b route 3 and what SATURN measured before
committing it — 1,141 of 1,201 refused, 13 of 22 refused on a pre-registered sweep.
The demand was made ad hoc in a dispatch prompt. It belongs in the registry.

## RESOLVED: the registry now names them

The section above was written when `MISTAKES.md` had fourteen V-entries and seven
checks, and it argued that the registry's planted-positive rule could not catch
half of what this catalogue records. **That argument has been acted on.** Four
entries were added, each from a measured failure in this round rather than
proposed abstractly, and the checklist went from seven checks to ten:

| entry | catches | derived from |
|---|---|---|
| **V-14a** | a scope test that condemns every refusal guard; `R ∩ D = ∅` is necessary, not sufficient | it.18's census: 19 of 31 fired, 17 were not defects |
| **V-15** | a **condemning** rule with no planted *negative* | instances 3, 5, 7, 9, 11, 13, 14 — seven of the first fourteen |
| **V-16** | an instrument that cannot measure, reporting a pass; checked with a planted *unreadable* | instance 18, measured, and it fired |
| **V-17** | a threshold imported out of its units | C-F's `δ = 0.5`, unreachable for `\|B\| > 7.5` by construction |

**The division of labour is deliberate.** This file **diagnoses** — it names a
mechanism and shows it twenty-four times. `MISTAKES.md` **prescribes** — it gives
the check that catches each class before it ships. A catalogue without the second
half is a list of regrets.

**V-14a also repairs a contradiction this round created.** Instance 21's fix —
*bind a must-fire to a constructed input so it cannot expire* — directly opposes
V-14's rule that a control whose reach misses production certifies nothing. Both
are right; they apply to different roles. Without the taxonomy, a reader obeying
both has no consistent action available.

**What is still not caught.** The reader-side entries — 19 and 23, and the
severity-scored-once case at 24 — have **no plant**, because nothing is wrong with
the instrument to plant against. They are caught only by questions: *is this claim
itself verified, or only the report it arrived in?*, *what did this guard actually
scan?*, *what did we correctly file as harmless, back when it was?* No registry
entry converts those into a mechanical check, and this round did not find one.

## Standing consequence for the round

`AUDIT.md`'s 314 KEEP rows are produced by **61 distinct classing rules**, and the
iteration-3 stratified draw samples 20 of them; the 44 singleton rules pool into one
stratum drawn at p = 3/44 = 0.0682. So **41 rows carry rules no draw at this budget
can falsify**. The catalogue above is the reason that matters: the rules on this
sheet have a measured failure rate against exactly one mechanism, and the sampling
scheme does not cover the strata where a fresh instance of it would land.

MARS's reprice is the route: unpool the singletons and budget 61 of 314 (19.4%).
Until then, the honest statement of the census is the one SATURN gave — the sheet is
**≥90.5% right at 95% confidence** on the drawn rows, and that is all it licenses.

---

## Where the substitution happens

The claim in the title is that these are one mechanism. That holds, but the
mechanism has **two loci**, and conflating them would make the catalogue less
useful rather than more.

**Instrument-side** — the rule itself keys on the proxy. A CI over the wrong
population (16), occupancy standing for room (17), unreadable standing for idle
(18), a roster standing for the reach (20), a conditional's reachability standing
for a guarantee's enforcement (21), a fit validated inside its range standing for
one validated outside it (22). Repairing the instrument fixes these.

**Reader-side** — the instrument is correct and reports correctly; the substitution
is in what someone takes the report to mean. A report's credibility standing for a
claim's truth (19). A green guard standing for a repaired repo (23). A severity
scored at filing standing for severity now (24). **No repair to
the instrument prevents either**, and that is what makes them worth separating: a
reader who fixes only the instrument-side entries will keep committing the other
two, and will believe the class is closed.

The two are not equally easy to catch. An instrument-side defect can be planted
against — that is what V-15's planted negative and V-16's planted unreadable are
for. A reader-side defect has no plant, because nothing is wrong with the
instrument to plant against. The only checks that caught the two here were
questions: *is this claim itself verified, or only the report it arrived in?* and
*what did this guard actually scan?*

## A note on the numbering

Instances are numbered 1–16 and 16–21. **There is no instance 15**: the numbering skipped it when the later entries were appended, and the title read "nineteen times" over eighteen entries until the Inspector's Phase 1a audit counted them. The gap is left in place rather than closed, because the ledger and several round records already cite instances by number, and silently renumbering them would break every reference to make a count look tidy.

**24 instances, numbered to 25.**

This paragraph has been wrong three times, each in a different way, and the third is the instructive one. Twice a literal string replacement failed silently because the sentence WRAPS across a line break. Then it was rewritten to DERIVE the count by scanning the file -- and the derivation counted `^| N |` across the whole document, matching a second table further down, and published 22 for a file holding 20. Deriving a number is not the same as deriving it over the right region; an automated count with the wrong scope is more confident and no more correct than the hand-written one it replaced.

## Instance 16 — a confidence interval over the wrong population

**Where.** `scale/r10_capacity_sweep.py:101`, and every cell in the three
`results/r10_it8_capacity_softmax_t*.jsonl` journals.

**The rule it has to satisfy.** The v-main.3M script, it.11, pre-registers the
open-region verdict as: *region learnable iff ≥1 cell < 1.0 **with N=8 seed CI
excluding 1.0** (bootstrap, B=10⁴)*.

**The proxy.** Each cell ships `boot_lo`/`boot_hi` from
`bootstrap_ci(pe, y_eval, seed=seed)` — a resample of **eval examples inside one
trained model**. It is a real interval, correctly computed, and it excludes 1.0
for `t*=8` at n=32768: `[0.9641, 0.9814]`.

**Why it is the wrong number.** It answers "is this model's error below the bar
on this eval set", not "does a randomly-seeded training run land below the bar".
The rule names the second. At n=32768 there is exactly one seed, so the quantity
it.11 asks for does not exist in the journal at all.

**The bias has a direction.** The eval-example CI is 0.0173 wide. The measured
seed-to-seed range at n=2048 (seeds 0–7) is 0.0345 — twice as wide. So the
shipped interval is not merely the wrong interval, it is systematically *tighter*
than the right one, in the direction that makes an unreplicated crossing look
established.

**Why it would have survived review.** Nothing is broken. The field is named what
it is, the bootstrap is correct, the interval genuinely excludes 1.0, and the
verdict string `LEARNS` is computed from the point estimate against a documented
bar. A reader satisfying it.11 by checking "does the shipped CI exclude 1.0"
gets `True` and moves on. The defect is entirely in the gap between the
population the rule names and the population the interval is drawn over.

**Direction.** This one EXCUSES rather than condemns — it makes a passing cell
look better-established than it is. That puts it in the half of the catalogue
that a planted-positive rule *can* catch, unlike instances 3, 5, 7, 9, 11, 13 and
14: plant a seed whose training run lands above the bar and the N=1 CI cannot
notice. The planted NEGATIVE this document argues for is not what finds this one;
an ordinary planted positive would have.

**Correction, filed against this entry's own author.** MERCURY did not commit
this one. His shipped table marks every single-seed cell `n=1` in the cell
itself, computes real N=8 seed CIs at (150, 2048) where eight seeds exist
(`t*=2` [0.9462, 0.9591], `t*=8` [1.1162, 1.1314], `t*=32` [1.1307, 1.1837]),
and states plainly that the n=32768 seeds "cost 8x600 s to seed and were not
bought." The hazard in the `boot_lo`/`boot_hi` field is real and the entry above
stands as a description of it. The implication that the round's table rested on
it does not: the instrument invites the substitution and the analyst declined it.
An entry in a mistake catalogue that assigns a mistake to someone who did not
make it is itself the catalogue's failure mode, and is corrected here rather than
quietly edited.

---

## Instance 17 — occupancy now, standing in for room later

**Where.** The scheduler this session, not any instrument: two queued jobs each
waited on the predicate *"no `r10_capacity_sweep` process is running"* before
starting. Seeds 4-7 needed 4.3 GiB; the it.10 octave at n=65536 needed an
extrapolated 7.9 GiB.

**The proxy.** Current occupancy stands in for *will there be room when I
allocate*. Both jobs wake on the same edge, and the second one's preflight runs
BEFORE the first has allocated -- so it prices against free memory that is
already spoken for, passes, and then 4.3 + 7.9 = 12.2 GiB lands on a 16.1 GiB
box.

**Caught before it fired, in that form.** The octave was stopped and re-queued
with an explicit dependency on a durable completion line rather than on
occupancy. But a milder form DID fire, and measurement caught it: the seeds job
launches `t*=8` and `t*=32` as two subprocesses, and between them the count is
legitimately 0 for a second or two. An edge-triggered wait cannot distinguish
that gap from the job ending. Two 4.3 GiB seats went live and the host fell to
2719 MiB of 16091; had the first job then reached `t*=32` there would have been
three seats wanting 12.9 GiB.

**What it costs to know this.** `scale/vram_gate.py:19` already says it:

> Nothing on this machine coordinates the card, and no per-job gate can see
> aggregate occupancy.

The gate prices a resource; it cannot sequence work. Pricing correctly at the
moment of asking is not the same as pricing correctly at the moment of spending,
and every job that passes a gate then becomes part of what the next job must be
priced against. Repaired by making the wait level-triggered -- three consecutive
idle readings, 20 s apart -- and, for the expensive job, by depending on a
durable fact in a file instead of a live reading at all.

---

## Instance 18 — an unreadable measurement reported as a passing one

**Where.** The idle-check in both queued waiters this session:

```python
return int(out) if out.isdigit() else 0   # unreadable -> "nothing is running"
```

**The proxy.** A PowerShell CIM query returns the live sweep-process count. `0`
means the box is idle and the next job may start. Any result that does not parse
as a digit — an empty string under memory pressure, a timeout, a transient CIM
failure — was mapped to `0`.

**So an absence of evidence was returned as evidence of absence**, in the exact
field a scheduler uses to decide whether to allocate 7.9 GiB.

**It fired.** The seeds 4–7 waiter woke and launched a second seat while PID
35396 was still running seeds 1–3 with 2.2 GiB resident. The debounce added
minutes earlier — three consecutive idle readings, 20 s apart — could not help,
because three consecutive *unreadable* answers are three consecutive zeros. A
retry loop over a fail-open predicate is still fail-open.

**What makes it the campaign's own defect, not a generic bug.**
`scale/vram_gate.py:47-50` states the rule this violates, in its own docstring:

> It degrades to a stated UNKNOWN rather than a false GREEN when `nvidia-smi` is
> absent, there is no CUDA device, or `psutil` is not installed, because a gate
> that silently passes when it cannot measure is the vacuous-control shape this
> campaign catalogues.

The fail-open version was written into the checker that decides whether to *call*
that gate. The rule was known, written down, and applied one layer too shallow.

**Cost.** None to the data: the two live jobs carried disjoint seed sets (1–3 and
4–7) at `t*=8`, so both produced needed cells, and the host held at 3283 MiB. The
octave waiter, which had 7.9 GiB at stake, was protected by a second condition —
a durable completion line in a file — and not by this predicate.

---

## Where this round's author-committed defects actually live

Instances 16–18 plus the noise-floor strike are four defects committed by HOUSE
in this round. Their distribution is the finding:

| # | defect | where it lives |
|---|---|---|
| — | a resource reading proxying for a job state ("starved, cannot start") | monitoring |
| — | one thread-pair reported as the noise floor (4.911e-4 for 2.345e-3) | analysis |
| 17 | occupancy-now proxying for will-there-be-room | scheduling |
| 18 | an unreadable measurement proxying for idle | scheduling |

**Three of four are in the scaffolding, not the science.** The capacity sweep,
the frontier fit, the seed CI and the E4′ spread were all built with pre-declared
rules, must-fires, and refusals. The code deciding *when those may run* was
written quickly, with fail-open defaults, and was never given a must-fire of its
own — `alive()` had no test that it reports busy when it cannot see.

That asymmetry is worth stating plainly, because it generalises past this repo:
**the guardrails were reasoned about less carefully than the thing they guard.**
An instrument gets a rejection region because everyone can see it produces a
number someone will believe. A scheduler produces no number, so nothing about it
looks like a claim — and it silently decides which numbers get produced at all.

The direction is also uniform: every one of the four failures is permissive. The
resource reading declared work blocked that was running; the noise floor was
quoted 4.8x too tight; occupancy said there was room; unreadable said idle. None
of them ever refused something that should have proceeded. A defect class whose
errors all point the same way is not noise, and the planted-negative rule this
document argues for would catch none of them — they need a planted *unreadable*,
an input the instrument cannot measure, to check that it says so.

---

## Instance 19 — a report's credibility, standing in for a claim's truth

**Where.** `tests/loop/test_the_variance_law_is_stated_once.py`, in the docstring
of the guard written to bind MARS's attack #1 — and in the message relaying that
attack to the author, in the same turn.

**The proxy.** MARS filed a detailed report. Its mechanism was precise (nilpotency
index `t*+1`, `b[s-1] = 0` killing the `m=0` term), its line citations were exact,
and every number checked: `negation_scope.py:367` versus `:419` verified by
reading both; `1.4012436552` verified against the shipped journal; the 0.2465431
gap reproduced to seven decimals. Four checks, four confirmations.

The fifth claim — *"and the test would still pass"* — was copied without checking.

**Why it mattered more than the others.** It was the claim that made the finding
serious. A repair that aborts the round's central block *loudly* is a
documentation inconsistency. The same repair passing a green suite is a silent
catastrophe. Severity lived entirely in that one unverified clause.

**What it cost to check.** One `pytest` invocation, 8.5 seconds:

    tests/cameron/test_m3_etasks.py::test_the_chain_flipper_dependence_is_its_closed_form
    5 passed in 8.52s

and one read of the assertion, which hard-codes `2.0 / math.sqrt(t_star)` at a
1e-12 tolerance — so any change to the closed form fails it immediately — plus
one line of `negation_scope.py:1460`, where `flipper_dependence` is `moved /
scale`, measured from data and therefore unmoved by editing a constant.

**The proxy named plainly.** *This report has been right four times* stood in for
*this particular sentence is true*. Accuracy is a property of claims, not of
reports, and it does not transfer between the two.

**Direction.** EXCUSING, and self-serving in a specific way: the unverified clause
made the round's own adversary look sharper and the finding more valuable, so
nothing about accepting it felt like lowering a standard.

**BY THE END OF THE ROUND THIS HAD A MEASURED RATE, AND IT IS NOT LOW.** Five
inherited claims were struck, all accepted without opening the source:

| inherited | from | what was wrong |
|---|---|---|
| `m3_capability.py:271`, hop budget | MERCURY | line is `:273`, and it is a *print statement*; the enforced constant is `e_ladder.py:55` |
| `m3_capability.py:120`, "no .cuda()" | MERCURY | line is `:261` — and it was written into a tracked module's docstring, then quoted onward into a third agent's brief |
| "the test would still pass" | MARS | it fails twice and aborts the sweep; the clause carried the finding's entire severity |
| three static-analysis leads | an analyzer | all three false positives, forwarded to two agents, costing each a verification pass |
| "within 10 MiB" fit residual | the it.8 record | 10.1 MiB, rounded in the flattering direction |

Five for five, across three sources, in one round, by an author who struck others
for the same thing in between. The rate is what makes this a mechanism rather than
a lapse: **a claim arriving inside good work inherits that work's credibility**,
and nothing about reading it feels like accepting an unverified assertion. The
cost of checking any one of them was under a minute; the cost of not checking put
a wrong line number into a shipped module and a false severity into a published
record.

**THE SAME MECHANISM APPLIES TO CONSTANTS, NOT ONLY CLAIMS.** Four briefs this
round told agents to IMPORT thresholds that already exist rather than pick ones
that flatter the result. Agents complied, and well: SATURN imported
`kirchhoff.AGREEMENT_TOL` by **object identity**, so a local re-pick fails the
self-check, and imported `rips_gate.FAIL_BAR = 0.9` for C-B. Both correct, because
both sources measure the same quantity.

MARS imported `delta = 0.5` from `negation_scope.py:1587` — the **chain task's**
`flipper_dependence > 0.5` clause — and applied it to a harmonic corpus whose
causal effect scales as `1/|B|`. Measured from his own rows, `fd_max × |B|` is
near-constant at **3.741**, so `fd ≥ 0.5` needs `|B| ≤ 7.5` while the corpus runs
`|B| = 4…80`. Ten of twelve rungs fail **by construction**. The chain threshold is
worse than merely inapplicable: `2/sqrt(t*)` drops below 0.5 at `t* > 16`, so that
clause would fail on the chain itself at `t*=32`.

**The rule as stated is incomplete.** "Import thresholds that exist rather than
re-picking one that flatters your result" does not say *the source must measure the
same physical quantity*, and without that clause it licenses transporting a number
across task families where it means something else. An imported constant carries
the authority of having been used before — which is this instance's mechanism
exactly, with a number in place of a claim.

**The rule that already existed and was not applied.** This round strikes
inherited citations — the author had, the same hour, handed MERCURY's
`m3_capability.py:271` hop-budget citation to the Inspector precisely because it
was load-bearing and unread. The rule was applied to the seat whose claim was
convenient to doubt and skipped for the seat whose report was impressive.

---

## Instance 20 — the defect that concealed a second defect in the same instrument

**Where.** `tests/loop/test_conftest_import_is_order_dependent.py`, written by this
document's author to catch order-dependent collection.

**Defect one: a roster standing in for the reach.** The guard carried a hardcoded
list of five offending files. By the time it was re-read the tree had moved past
it in *both* directions — three of the five had been retired to `attic/` at
iteration 4, and five live offenders had never been listed at all. It condemned
five files while five more went unreported. That is MISTAKES.md V-14 exactly,
committed in a guard.

Replacing the roster with a scan over `tests/` and `attic/tests/` found **ten**
offenders, not five. The defect was twice the size the instrument reported.

**Defect two, which only became visible once defect one was fixed.** The scan
matched `^\s*from conftest import`, catching indented function-local imports. The
per-file assertion used `line.startswith(...)` with no `lstrip()`, so it saw only
UNINDENTED imports. Five of the ten newly-collected cases were therefore
parametrised in and **could never fail**.

**Why the second hid behind the first.** The original hardcoded list happened to
contain only module-level importers. Against that roster the assertion's blindness
to indented imports had nothing to miss, so it never showed. Fixing the roster is
what exposed it — and had the roster been fixed by someone who did not then re-run
and count, the guard would have gone from reporting 5 of 10 to reporting 5 of 10
while *appearing* to have been repaired.

**The proxy named plainly.** A guard's OUTPUT stood in for its COVERAGE. Five
failures look exactly like all failures. Nothing in a red test distinguishes "this
found everything" from "this found half", and the half it found was the half its
own two blind spots happened to agree on.

**Direction.** Excusing, twice over, and compounding: each defect independently
under-reported, and their intersection under-reported by more than either alone.

**The check that catches it.** Not a planted positive — the guard fires. Not a
planted negative — the files it names are guilty. It needs the count of things
EXAMINED alongside the count of things FOUND, which is the second half of the
registry's rule 5 and the half that is easiest to skip: *"check the count of things
examined as well as the count of things found."* Ten collected, ten failing, is a
different statement from five failing, and only the first says the instrument
looked everywhere.

---

## Instance 21 — the must-fire with an expiry date

**Where.** `scale/it11_verdict.py::demo()`, the self-check on the gate that decides
whether the open region is learnable.

**What it looked like when written, and it was correct then:**

```python
v = verdict(8)
if v.n_seeds < REQUIRED_SEEDS:
    assert v.mean is None and "INSUFFICIENT" in v.reading
```

At the time, `t*=8` at n=32768 had four seeds. The branch executed, the refusal was
exercised, the check was live and passing for the right reason.

**Then the measurement made progress.** Seeds 4, 5, 6 and 7 landed. `v.n_seeds`
became 8, the branch stopped executing, and the guarantee — *never report a CI on
an unregistered N* — was no longer checked by anything. Nothing in the output
changed. `demo()` kept printing "demo OK".

**Proof it had gone vacuous**, from `test_phase1a_modules_are_bound.py`, which
plants a failure and requires the copy to exit non-zero: with
`if n < REQUIRED_SEEDS:` disabled, `demo()` **still exited 0**, reporting a CI
`over N=7` and printing its OK line.

**Why this is not the same as the other vacuities here.** Instances 16–20 were
static: a branch that never ran, an interval over the wrong population, an
assertion that could not fail, a roster that had drifted. This one was **genuinely
live when written, correct when tested, and decayed afterwards** — and it decayed
*because the work succeeded*. The seeds arriving is the thing the round wanted; it
is also what silently disarmed the check.

**The proxy named plainly.** A conditional's REACHABILITY stood in for the
guarantee's ENFORCEMENT. `if <transient data state>:` around a must-fire means the
must-fire is only a must-fire while the data cooperates, and no run announces the
day it stops.

**The repair generalises.** Bind the must-fire to a CONSTRUCTED input that cannot
drift — here `verdict(8, n_train=-1)`, matching no journal row, so the cell has
zero seeds by construction and the refusal fires on every run forever. If a
must-fire's premise can be satisfied by real data that may change, it is not a
premise; it is a coincidence with a schedule.

**Direction.** Excusing, and self-concealing: the check reported success both
before and after it stopped checking anything.

**What catches it.** Not a planted positive, not a planted negative, not a planted
unreadable — all three pass. Only planting a failure in the GUARANTEE and requiring
the self-check to notice. That is what
`tests/loop/test_phase1a_modules_are_bound.py::test_every_demo_asserts_something`
does, and it is the only reason this was found rather than shipped.

---

## Instance 22 — a held-out point that tested the wrong thing

**Where.** The host-RSS model used to price every memory refusal in iterations 10
and 11, including the one that declined the cell which would settle whether
`t*=32` ever crosses.

**The model.** Three measured points — 901.2, 1567.9, 4275.1 MiB at
n = 2048, 8192, 32768 — fitted as `RSS = 665.5 + 0.1102·n`.

**The validation, and why it convinced.** The fit was built on the two large
points and then checked against the held-out n=2048: predicted 891.1 against a
measured 901.2, an error of **10 MiB, 1.1%**. That is a genuinely good number, it
was reported honestly, and it was the evidence on which the model was believed and
used to price `n=65536`.

**Why it was the wrong test.** n=2048 lies **inside** the fitted range. Holding it
out tests INTERPOLATION. The model was then used at n=49152 and n=65536 —
**outside** that range, where nothing had been checked. A model can interpolate
perfectly and extrapolate badly, and this one does.

**Measured, once a point beyond the range existed.** Peak RSS at n=49152 is
**6584 MiB** against a predicted 6082 — under by **502 MiB (8.2%)**. The
per-example slope is not constant:

| segment | MiB per example |
|---|---|
| 8192 → 32768 | 0.1102 |
| 32768 → 49152 | **0.1409** |

Growth is **superlinear**, and the linear model degrades fastest exactly where it
was being used.

**Direction: permissive.** Re-extrapolating with the local slope puts `n=65536` at
~8892 MiB actual and ~11115 with margin, against the 9859 the gate was told. The
model quoted a **smaller** requirement than the truth. Had the box had 10 GiB
free, the gate would have approved a job that did not fit, on a number that looked
validated.

**Nothing was lost only because the gate refused for an independent reason** — the
box had 8808 MiB and even the understated figure exceeded it. The model's error
and the gate's refusal were not connected; the second happened to cover the first.

**The proxy named plainly.** *Validated somewhere* stood in for *validated here*.
A held-out point inside the fitted range is a real check of a real property — just
not the property the extrapolation needed.

**The check that catches it.** Hold out a point **beyond** the range you intend to
use, or state in the same breath as the fit that the model is unvalidated outside
its span and that its errors there are unbounded in an unknown direction. The
first is often impossible — you cannot measure the cell you cannot afford — which
makes the second obligatory rather than optional.

---

## Instance 23 — a green guard read as a repaired repo

**The first instance in this catalogue where the instrument was not defective.**
Every entry above is a rule keying on the wrong thing. This one is a correct rule,
correctly reporting, read as saying more than it said.

**What happened.** `tests/loop/test_the_variance_law_is_stated_once.py` was written
to check that `scale/negation_scope.py` states the label's variance law
consistently. It was RED, correctly. The law was repaired at that file, the guard
went green, and the finding was recorded as cleared.

**Four other files still asserted the wrong law:**

| site | what it was |
|---|---|
| `README.md:189` | live documentation |
| `tests/cameron/test_m3_etasks.py:38` | module header |
| `tests/cameron/test_m3_etasks.py:278` | a docstring deriving `2/sqrt(t*+1)` above an assertion of `2/sqrt(t*)` |
| `scale/r10_capacity_sweep.py:16` | **the producer of the round's data** |
| `R10_ITERATION_08_09.md:173` | the round record itself |

The guard never claimed otherwise. Its docstring names its source file in the
second line. It scans one file because the defect it was built for was an
*internal* inconsistency in one file — that scope is correct and was correct when
written.

**The proxy named plainly.** A test's PASSING is a statement about what it checks.
It was read as a statement about the world. Green means "this file is now
consistent", and it was heard as "the wrong law is gone".

**Why it is worth a separate entry.** Instance 20 is an instrument whose scope
did not match its own claim — a defect *in* the guard. This is a guard whose scope
matched its claim exactly, and a reader who took the claim to be larger. The
failure has moved from the instrument to the person holding it, and no repair to
the instrument would have prevented it.

**What actually caught it.** Not a test. A `grep` for the wrong form across the
repo, run because the it.13 finding had listed five sites by name and only one had
been touched. Four of the five stated `N(0, t*+1)` **within a line or two of a bar
of `2/sqrt(t*)`** — a contradiction visible inside a single paragraph, which
survived because nobody read the two halves against each other.

**And the sweep could not be automated.** Six files still carry the wrong form
deliberately: MARS's attack report, the guard's own docstring, SATURN's binding,
the it.13 record, a script that computes both forms to compare them, and a
correction note. A repo-wide replace would have destroyed the description of the
defect while fixing the defect. Telling those apart took reading each one.

**The check.** When a guard goes green, ask what it scanned — and if the finding
named sites the guard does not cover, the finding is not closed by the guard's
colour.

---

# A DIFFERENT FINDING: the script's own rules

Everything above is an instrument keying on the wrong thing. This is not that, and
it is filed separately so the instance count is not inflated by it.

**Three of the round's pre-registered rules were checked against their own designs.
All three survive as decision procedures. All three fail as justifications.**

| rule | the threshold | what was wrong with its reasoning |
|---|---|---|
| **it.11** — *region learnable iff ≥1 cell < 1.0 with N=8 seed CI excluding 1.0* | met, and correct | the interval a reader would naturally check — the `boot_lo`/`boot_hi` every cell ships — resamples eval examples inside one model. It excludes 1.0 and answers a different question. The seed CI the rule names did not exist at the deciding rung until it was bought. |
| **it.12** — *strike stands iff sign holds in ≥18/20, binomial p = C(20,≤2)·0.5²⁰ ≈ 2.0e-4* | met, k=19/20 | that p is a **sign-test** p-value. Nothing makes 0.5 the null crossing rate for a continuous NRMSE against a fixed bar, and the 20 draws share one graph, one partition, one split, so they are not 20 replications. The decision-relevant quantity is P(score ≥ 0.5) = 1/20, Clopper–Pearson [0.0013, 0.2487]. **The rule quotes a p three orders of magnitude tighter than its design supports.** |
| **it.16 C-C** — *peak activation bytes = 4·n·s·d·heads vs card bytes* | GREEN, 25/25 shapes | the formula prices the **projections**; the shipped peak is the `[n, s, s]` operator, so it under-prices by exactly `s/d = 64`. At batch 8192, s=1024, it reads 512 MiB where the real tensor is 32,768 MiB — four times the card. And VRAM never binds anyway: host refuses at batch 8,832, VRAM not until 101,837. |

**What pre-registration bought, and what it did not.** It bought the thing it is
for: nobody chose a threshold after seeing the data, and each verdict above was
reached mechanically. It did not make any threshold's *justification* correct, and
in two of the three cases the justification is the part a reader would rely on —
a p-value they would quote, a formula they would reuse.

**The asymmetry that makes this worth recording.** A wrong threshold is caught the
first time someone disagrees with a verdict. A wrong justification beside a correct
threshold is never caught, because the verdicts keep coming out right. `k ≥ 18`
is a fine decision rule and needs no null at all; the null was decoration that
looked like rigour. `4·n·s·d·heads` returns a number, the number is under the card,
and the clause passes — for as long as nobody asks which tensor it priced.

**The check.** For every pre-registered rule, ask separately: *is the threshold
sensible?* and *is the arithmetic beside it true?* This round was three for three
on the first and zero for three on the second.

---

## Instance 24 — a severity scored once, for a dependency that moved

**Reader-side, like 19 and 23.** Nothing is wrong with the finding, the report, or
the score it was given. What went wrong is that the score was correct **when
written** and nobody re-read it when the thing it touched became load-bearing.

**The finding.** MARS, iteration 4, attack 3: `calibrate_bar` trains its
two-feature positive control on `feats` and scores it on the same `feats`, while
every arm the bar gates is scored held-out. He scored it, accurately:

> "FIRES as a method defect; no verdict moves."

**He was right.** At iteration 4 the round had no verdict resting on the bar. The
defect was real, the severity was correctly assessed as low, and the entry went
into the record with that severity attached.

**Thirteen iterations later** the round published `t*=32` **NOT LEARNABLE** on a
seed CI of [1.0009, 1.0057]. That verdict rests on the bar having certified the
task as learnable in the first place — and the certification came from an
in-sample control. The verdict stands as a measurement; its *interpretation* now
splits two ways ("softmax is the limitation" / "the bar overstated the task") and
nothing separates them.

**The score did not change. The graph under it did.**

**The proxy named plainly.** *Severity at filing* stood in for *severity now*. A
findings registry stores the first and is read as if it were the second, and the
gap widens silently every time new work lands on top of an old open item.

**Why no instrument repair helps.** MARS's report is accurate and remains accurate.
The score was right. The catalogue entry is right. There is nothing to fix in any
of them — which is what puts this with 19 and 23 rather than with the
instrument-side entries.

**The check.** When a result becomes load-bearing, re-read the open findings that
touch it — not for new defects, but to re-score the ones already recorded. The
question is not *"what did we miss?"* but *"what did we correctly file as harmless,
back when it was?"*

**Measured cost here:** thirteen iterations, and it was found by accident — while
surveying journal record types for something unrelated, noticing a
`trained_two_feature` field no analysis this round had read.

## Instance 25 — two decimal digits, standing in for the same subject

**Where.** `scale/coherence_floor.py:18-30` publishes the random-coherence cell
at `d = 256, k = 16`. The script line that consumes it prices S1's scramble
control, and S1 runs at `d_model = 16` (`scale/m3_capability.py:79`).

**The proxy.** *Numeric proximity* stood in for *same quantity*. The published
`0.174795` and `welch(16, 32) = 0.179605` agree to within **2.7%** and share no
argument: one is an expectation over random draws at `(256, 16)`, the other a
worst-case floor over codes at `(16, 32)`. **Five distinct quantities in this
repo read within 15% of 0.175**, `ceq/diagnose.py:15`'s
`content_conditional_sign_decay` at s=8 (`0.17480`) among them.

**What it cost.** The constant transfers to a width the code does not run. At the
shipped `(16, 8)` the measurement is `0.547180` [0.545987, 0.548373] against
`0.174795` [0.174460, 0.175131] — **3.130×**, CIs disjoint by 0.371. Real key
directions read `0.719784`, 4.12× the published figure, because real keys are
correlated and so *more* coherent than random ones.

**Which side.** **Reader-side.** The instrument is correct and says what it
measured, at `:18` — *"at d=256, k=16, seed=0"*. Nothing in the file claims the
number travels. A reader carried it to S1 and the arithmetic still looked right,
because the wrong answer was close to a right-looking one.

**Why the catalogue's usual repair does not apply.** Instances 16–18 and 20–22
were repaired by making the instrument state its scope. This instrument already
does. The failure is entirely downstream, which is why it is filed with 19, 23
and 24 rather than with the instrument-side group.

**Route.** Print the closed **form** per row, not the evaluated constant, so a
row cannot be lifted without its arguments. Bound at
`scale/r10_it20_coherence.py`, whose `demo()` must-fires on the two dimensions'
CIs being disjoint — the check that would have caught the transfer.

**The dual, stated because the same tell hides it.** `mu_jl(256,16) = 0.147176`
is a **union bound** on `E[max]` and sits *below* the measured `0.174795`. A
union bound below its quantity is not a loose bound, it is **not a bound**.
`.superpowers/sdd/polymorphic-drifting-squirrel/progress.md:454` prices this as
*"15.8% low"* — an accuracy error. It is a validity error, and the percentage is
what disguises it.
