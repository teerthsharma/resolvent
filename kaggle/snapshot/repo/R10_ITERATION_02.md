# R10 — v-main.3M Iteration 2 — the spot-check

Phase 0, TRIAGE. Room dispatched in parallel: SATURN (execute the draw), MARS
(rebind + draw power), MERCURY (priced-or-regression), NEPTUNE (provenance,
untracked tests, `make verify`). Health Inspector audited before this record.

## Verdict

**The census is REJECTED at p < 0.09 and it re-runs at iteration 3.**

SATURN measured 1 KEEP failure and reported the census stands. He also reported,
unprompted, that his own protocol was ambiguous in a way that decided the verdict,
gave the strict reading that rejects his work, and asked to be overruled. The
Health Inspector read the committed protocol and overruled him: **the count is 3.**

That exchange is the iteration's result. The instrument that certifies which rows
may carry a claim was itself certified, by someone other than its author, against
a text neither could edit after the fact.

## The deciding ruling

The protocol was committed at `06a180c` **before any row was measured**;
`git diff 06a180c HEAD -- results/r10_it2_protocol.md` is empty and one commit ever
touched it, so the governing-version question is clean.

Section 0 defines a disposition failure as "the row's KEEP is wrong". Section 2
supplies the operational test T-b: an outcome is red-by-design only via the chase
`KNOWN_RED` ledger or a `test_claim_*` refutation instrument, and closes
`Any other red is a FAIL.`

The Inspector's grounds, both textual:

1. **The drafter knew how to subordinate a section to section 0 and did it once —
   not here.** Section 5 line 121 reads "Any leg that fails makes the ATTIC wrong,
   and **per section 0** that is a finding and an amendment, outside the binomial."
   That express cross-reference appears in section 5 and nowhere in T-b. A drafter
   who reaches for "per section 0" when he wants it to control, and omits it, has
   drawn a line rather than left a gap.
2. **The class-cell rule would be surplusage.** Section 2 line 53 imports section
   0's disposition principle but scopes it expressly to class-cell errors. It has
   work only if T-a and T-b are *not* disposition-filtered.

And the direction of error is the cheap one: T-b as drafted is over-inclusive,
making the instrument conservative — it can reject a sound census at the cost of a
re-run. The opposite reading certifies a census on a rule silently widened after
the count came in. Section 6's own guard, "A second failure is not argued into
being one", bites the reading that argues.

| row | measured | leg | |
|---|---|---|---|
| `workdone2.md` | 0 of 4 readings reproduce at abs=5e-7 | D1, expressly load-bearing | **FAIL 1** |
| `tests/cameron/test_r5_aggregator_red.py` | 8 failed / 1 passed; zero failing names carry `test_claim_`; `tests/cameron/conftest.py` has no ledger and no xfail | T-b, neither route | **FAIL 2** |
| `tests/chase/test_structural_zero_guard.py` | 2 failed / 2 passed; neither failure prefixed, neither on the chase ledger | T-b, neither route | **FAIL 3** |

Rejection is robust: `test_r5_aggregator_red.py` plus the undisputed `workdone2.md`
already makes 2.

**The re-run cannot reuse the draw.** `scale/spotcheck_draw.py` reads `AUDIT.md`
live from the working tree. SATURN's amendments moved the sheet 307/40 → 314/33, so
the urn moved with it: 3 of 10 KEEP and 3 of 5 ATTIC paths differ at HEAD.
Iteration 3 must draw against a pinned revision or re-pre-register.

## The mechanism — ten instances, one shape

Every one keyed on a **surface proxy** instead of the thing itself.

| # | rule | keyed on | the thing itself | found by |
|---|---|---|---|---|
| 1 | P1 | who built the input | what path it takes | MARS it.1 |
| 2 | P3-literal | a literal path string | the constructed path | SATURN it.1 |
| 3 | P1' | where input enters | the instrument's claim | SATURN it.1 |
| 4 | P2/P3-orphan | who imports it | whether it is reachable | NEPTUNE it.1 (survived audit) |
| 5 | refutation classifier | a name prefix `test_claim_*` | the `CLAIM AS WRITTEN` banner | MERCURY it.2 |
| 6 | "no python importer" | a search that could not find what it sought | the import graph | SATURN it.2 |
| 7 | "ran ALL-GREEN" | an exit status | pass vs recorded finding | SATURN it.2 |
| 8 | hankel ATTIC conjunction | the same blind prefix | `test_mustfire_*` nodes | HOUSE it.2 |
| 9 | "0 results artifacts" | absence of a file | absence of a producer | HOUSE it.2 |
| 10 | HOUSE's own import guard | tracked-ness (`git ls-files`) | presence in the tree | Inspector it.2 |

Instance 10 was committed by the guard written to catch instances 1-9, and the
Inspector struck its count. `scale/p1prime.py` left the git index between two runs;
the module-scope `open(mode='w')` at line 63 never moved, and the guard stopped
reporting it. Repaired to walk the union of index and tree.

Instance 5 has a live consequence beyond the classifier: T-b's second route reaches
only prefixed files. `tests/foreman/test_ppnp_parity.py`'s reds *are* prefixed and
*do* pass T-b. The route works; it stops at the `tests/chase` and `tests/cameron`
doorstep where both disputed rows live. That is why the protocol rejected two rows
whose KEEP is substantively correct.

## What each seat returned

**SATURN.** Committed the protocol before measuring. Ran the 15 rows. Found the
"no python importer" absence claim wrong on **12 of 18** rows asserting it, missing
**135 importer edges** (52 on `pivot_probe.py`, 45 on `negation_scope.py`) — V-7
again, one level above iteration 1's instance, and it cost zero dispositions
because every affected row was carried by its journal leg; the orphan class
survives at 0 of 19. Found **7 of the 12** chase `KNOWN_RED` files classed ATTIC:
`xfail(strict=True)` returns exit 0, so "ALL-GREEN" and "twelve recorded findings"
are the same byte. Generalised his own `workdone2.md` failure and found a second
undrawn row, `LOOP_PROMPT_ROUND7_ARCHIVE.md` at 0/2. Sheet moved 307/40 → 314/33.
Routed three `tests/chase` files declaring "RED on purpose" off-ledger to CHASE
rather than editing a ledger that governs CI.

**MARS.** Rebound the iteration-1 strike in the correct order (the strike was on
the binding, never the fact). Attack **FIRES**: a uniform draw of 10 from 307
misses the measured c=12 cluster with **P = 0.6671**, and did miss — 0 of 12 drawn.
Needs k=67 for 95%. Deeper: KEEP 307 carries **46 distinct classing rules**, the
draw touches **7**, leaving **39 rules over 151 rows (49.2%) never sampled** and 10
of 16 multi-row strata at zero draws. A rule the draw never samples cannot be
falsified, probability 1. Reported the limit that cuts against him: stratification
does **not** dominate — against a size-12 cluster placed adversarially in the
largest stratum, uniform at matched budget wins 0.7157 vs 0.4951. Clean non-hit
reported plainly: the presumption axis is fully powered, P(>=1 presumed drawn) =
1.0000 exactly, since all 36 presumed rows are ATTIC and only 4 ATTIC rows are not.

**MERCURY.** **Zero regressions. Three of three priced. No G2 event.** R1 settling
is red-by-design (`test_r1_settling.py:57` `# CLAIM AS WRITTEN -- RED`, root commit
`dfc1591`, never modified). APPNP's claim was already withdrawn (`DONE.md:10441`,
"Max-plus is dead", 9.95e-14) — and the group is **14 nodes, not 10**;
`tests/foreman/test_ppnp_parity.py` contributes 4 that SATURN's routing omitted,
Inspector-confirmed at `14 failed, 21 passed`. The 300M group is **neither** — all
ten failures are one data-presence guard at `test_scale_axes.py:84`, 3 of 20 points
present, **zero of the ten ever reads a ratio**; a witnessed journal scan over 809
records found `n_params` present only as 4769/4770 and zero hits at >=1e8. The 300M
claim was never made: `MODEL_CARD.md:210` pins parity to 3.3M.

**NEPTUNE.** Rescued the provenance out of the session temp directory into
`scale/p1prime.py` + `results/p1prime_*`. Settled the nesting question by showing it
**ill-posed, not undecidable**: the 46 is a test-file count, the 123 is an
importer-graph target count, **intersection 0** — disjoint universes, which was
House's error to put on one axis. The 22 is **permanently unreproducible**, marked
rather than reconstructed. Refuted `AUDIT.md:106-108`'s own reconciliation:
excluding the four named tokens gives **38, not 22**; 16 files unexplained.
Measured that **`make` is not on PATH**, so it.4's `make verify` cannot execute at
all here; the replacement is one pytest invocation carrying four node ids,
**8.823 s** against 15.435 s for three separate invocations.

## The rho axis — measured, complete

Launched by HOUSE against MERCURY's dated exposure: it.9's capacity sweep runs at
`rho=1.5`, a constant tuned at 3.3M and never checked at a second size, and that
conditioning is not removable after the fact.

| rho | 0.9 | 1.2 | 1.5 | 2.0 | argmin |
|---|---|---|---|---|---|
| **small** (3,319,296 params) | 1.0723 | 1.0433 | 1.0280 | **1.0228** | rho = 2.0 |
| **large** (12,784,128 params) | 1.1613 | 1.1404 | 1.1097 | **1.0974** | rho = 2.0 |

ratio = sgate / softmax, lower is better parity. 2 seeds per point.

**The axis cannot answer its own question with this grid.** It asks whether the
argmin over rho *moves* between sizes. Both argmins sit at **rho = 2.0, the grid
boundary**, and the ratio is monotone decreasing across the whole grid at both
sizes. A quantity pinned to the edge at both ends cannot be shown to move or not
move. Running the axis does not clear it.9's conditioning; **extending the grid
one octave does**, and it costs one point at each size, ~42 s + ~150 s ≈ 3.3 min.
`axes.py`'s grid is a fixed literal, so that edit belongs to CHASE.

**What the run does resolve, and it is not comfortable.** At the best rho on the
grid, parity is measurably worse at the larger size: small `1.0228 [1.0183, 1.0274]`
against large `1.0974 [1.0803, 1.1145]`, **CIs disjoint**. The degradation holds at
every rho on the grid. Two seeds, so this is a reading and not a verdict, and it
belongs to the seats that own the parity claim.

**Two corrections to MERCURY's cost model, both measured this iteration:**

| | his model | measured |
|---|---|---|
| rho-axis wall-clock | 12,391 s = 3.44 h (INHERITED) | 39+40+42+43+150+162+155+146 = **777 s = 12.95 min** |
| size gap | "8x apart in parameters" (from `axes.py`'s docstring) | **3.85x** — 12,784,128 / 3,319,296 |

The INHERITED label is retired: 15.95x over. His scheduling conclusion inverts —
the axis is not a seven-iteration budget item, it finishes inside one iteration.

## Health Inspector — 32 audited, 7 struck

1. **SATURN's reading** of section 0 vs T-b — overruled on the committed text.
2. **SATURN's count and verdict** — 3, not 1; REJECTED, not STANDS.
3. **HOUSE's "2 failed, 127 passed"** — re-ran as 1 failed. Cause: instance 10 above.
4. **HOUSE's four guards, all UNBOUND** — zero ledger occurrences of any test name.
   HOUSE wrote the logging rule into six dispatch prompts and logged nothing. All
   twelve reds and five findings have since been logged with re-runnable node ids
   and a note labelling them as retroactive re-runs, not the original runs.
5. **NEPTUNE's "reproduces `AUDIT.md:100` verbatim"** — prints 193/46/147 against
   191/46/145, and the 191 lives on line 99. The +2 is plausibly this iteration's
   own new test files inflating a live-tree count, which makes it instability
   rather than fabrication, but the claim as posted does not reproduce.
6. **MERCURY's "only 10 use the `test_claim_*` prefix"** — re-measures 12. The
   mechanism survives untouched: 3 banner files carry no prefix, so banner and
   prefix remain different sets. Only the count is struck.
7. **The Inspector's own log entry at `9182`** — named the wrong node as the
   retroactive binder. UNBOUND verdict stands; citation corrected.

**Ledger hazard, recorded by the Inspector and worth more than several findings:**
MERCURY encodes every red as `"state":"RED"`, NEPTUNE as `"status":"RED"`. The
convention is `"status":"red"`. A binding audit keyed on the convention sees
**none** of their reds. That is instance 10's shape again, in the audit trail
itself.

## Open

1. **The census re-runs at iteration 3**, and it cannot reuse the draw. Draw against
   a pinned revision or re-pre-register.
2. **T-b needs a third route** before the re-run: reds a file declares as its
   deliverable, the `tests/foreman` precedent. Without it the protocol will reject
   the same two substantively-correct rows again.
3. **MARS's stratified draw** is specified and executable — budget 30 KEEP with a
   floor of 1 on every stratum holding >=2 rows, ATTIC unchanged at 5, giving 1.0000
   rule coverage against all 46. It does not dominate uniform on adversarial
   cluster placement; both together cost budget 40+.
4. **The rho grid needs one octave** to make the axis answerable. ~3.3 min. CHASE.
5. **Parity degrades with size** on the measured grid, CIs disjoint at the best rho.
   Two seeds. Belongs to the parity-claim seats.
6. **`ceq.hankel` has no replacement calibration** named for its ATTIC'd must-fire
   battery; two production modules import it.
7. **`data/README.md`'s derivation is wrong by 10.6x** and cannot be repaired by
   guessing. Whoever fetched the corpus states the exact upstream slice.
8. **`scale/vgpe_flops.py`'s ATTIC row has lost its basis** — leg 2 of its
   conjunction is now false, `results/vgpe_flops_run.txt` exists, and 29 of 29 of
   `V12_PRICING.md`'s numbers reproduce from it at exit 0. SATURN's amendment.
9. **`kind` and `torch_version` have no must-fire** in the identity manifest.
10. **G2 baseline taken** for it.3's attic move: `results/G2_BASELINE_it2.sha256`,
    256 files, all 256 verify, no self-reference.

## North Star

> "A consequence-understanding, path-understanding, causality-understanding attention
> architecture, as good as self-attention on a calibrated bar, that predicts the NEXT
> BEST ACTION toward equilibrium — not the next best token. The novelty is the WHOLE
> ARCHITECTURE; components may be pre-existing, always cited."

**Distance: zero metres toward the architecture, and one real step on the bar.** The
census that decides which rows may carry a claim was rejected by its own
pre-registered rule, which is the instrument working rather than failing. Against
that, the rho run is the first measurement this round that touches the calibrated
bar itself, and what it says is that parity is worse at 12.8M than at 3.3M with
disjoint CIs — a reading the sheet could not have produced at any KEEP count.
