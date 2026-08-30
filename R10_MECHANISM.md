# The surface-proxy defect — R10's one mechanism, fourteen times

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

## The fourteen

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
| 14 | agent-name filter | one **capitalisation** | the agent | HOUSE it.3 | `Cameron` 108 vs `cameron` 8,940 — an 88× miss |

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
