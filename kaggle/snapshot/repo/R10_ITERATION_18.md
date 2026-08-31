# R10 — iteration 18: the scope clause over-fires 17 times out of 19

**Script task.** *"MARS: attack #2 scope-hunt — for each control, the reachability
set R(control) vs production domain D; fires iff R ∩ D = ∅ for any control (the
14th class, mechanized as a set census)."*

| | |
|---|---|
| controls examined | **31** |
| firing (`R ∩ D = ∅`) | **19** |
| firing, **proven** empty | 18 |
| firing, **sampled** only | 1 |
| **firing AND a defect** | **2** |
| errored | 0 |

Scored rows: 29 passed, 2 failed, 0 errored, 0 inapplicable, 1 not-reached,
1 unscored.

## The amendment is the result

**`R ∩ D = ∅` is necessary, not sufficient.** It flags 19 controls and 17 of them
are not defects, because **a refusal guard is supposed to be exercised on inputs
production cannot emit** — that is what a refusal guard is *for*. The clause has to
be conditioned on a **declared role**: certifying, excluding, or illustrating,
filed before scoring.

**And the repair carries its own hole, which MARS states rather than hides:** role
is *declared, not measurable*, so a mislabelled role gets a wrong verdict silently.
That is the shape of every entry in `R10_MECHANISM.md`, sitting inside the fix for
the clause.

## How R and D were made finite

A literal intersection of two infinite input spaces is not computable, so both
sides become named finite objects — **and the replacement is the thing to attack,
not the arithmetic**, which is why it is stated first.

**`R` is exact, no proxy.** All 31 controls are called at fixed arguments, so the
set of inputs each is actually handed is enumerable. The narrowing is declared:
`R` as *"could be handed"* is larger than `R` as *"is handed"*.

**`D` is two objects, with different logical force:**

| object | construction | what it can prove |
|---|---|---|
| `D_shipped` | 28 front-door instances, membership by `adm.labelled_key` SHA-256 | exact, both directions |
| `D_invariant` | five **necessary** conditions, each cited — n-grid, edge count, connectivity, `g ∈ [0,1)`, λ₂-band | **violation proves out; satisfaction proves nothing** |

Undecided cases fall to **200 draws** imported from `naive_yield(200, …)` at
`r10_corpus_spec.py:742`, seeds `0x3a180000+i`. Three further domains carry
specific loads: `D_label` (a least-squares residual proving *no* `g` produces the
label), `D_resource` (36 preflight arguments read **via AST — a line scan would
have found zero**), and `D_file`.

That `D_invariant` line is the correct logical form and it is what lets 18 firings
be called **proven** while one stays **sampled**.

## Proven versus sampled, exercised

**18 proven** — 11 by a violated invariant, 3 by label residual, 2 by
resource-literal, 2 by path.

**1 sampled** — `plant-relabelled`, a relabelled real instance that violates
nothing and was not found in 400 production draws. For that one the census
**cannot** distinguish `R ∩ D = ∅` from *`R ∩ D` small and unsampled*, and says so.

It exists **so that branch actually runs**. A sampled-only code path with no case
reaching it would be precisely the unexercised-path defect this census is about.

## The two real defects, and their price

| control | where | what rests on it |
|---|---|---|
| `it14/path_case` | `r10_corpus_spec.py:478` | three of it.14 §2's six agreement rows |
| `it15/probe-ring-B2` | `r10_dual_oracle.py:390` | the "route is not broken" control, run only on a 6-ring |

Both **certifying**, both **proven** empty. And both **redundant with in-domain
certifications** at 9.99e-16 and 1.554e-15, so the verdicts they support stand on
other evidence. MARS's phrasing is the precise one: *"not load-bearing — but as
written unable to bear load."*

## The conflict with this round's own law

> **"The round law 'bind a must-fire to a constructed input' and this clause point
> in opposite directions."**

That law is this record's author's repair for **instance 21** — the must-fire that
expired when the eighth seed landed and its guarded branch stopped executing. The
fix was to bind it to a constructed input so it could not decay.

**A must-fire bound to a constructed input is, by construction, a control whose `R`
may miss `D`.** One rule was invented this round; the other was already in the
script; they contradict.

**The role taxonomy reconciles them, and the reconciliation is not a fudge.** A
control whose job is to *exercise an instrument* — does the refusal fire when its
condition is violated? — legitimately uses constructed inputs, because its purpose
is to reach a branch, not to describe production. A control that *certifies a
property of production* must enter through the production path. Same mechanism,
opposite requirements, and the deciding fact is the role.

Which is exactly why the census cannot infer it.

## Side finding, carried rather than repaired

`house-events.jsonl` lines **1899, 2937, 2938, 5871** do not parse — `Invalid
\escape`, from earlier rounds and other agents. MARS identifies the shape
correctly: **shell-quoting corruption**, the same failure this record's author
committed today writing a ledger entry through backticks in a double-quoted shell
string, where the write reported success while destroying part of its payload.

Left unrepaired and logged, which is correct for append-only evidence.
