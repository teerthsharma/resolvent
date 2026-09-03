# V20 R15 — it.13 — MERCURY (LESTRADE). WHAT PHASE C CAN STILL DECIDE, AND WHAT IT COSTS

Branch `v17k-gate0`. **No git writes. Nothing touched Kaggle. No training run started.**
Nothing under `results/` was opened for write; the only files created are
`tests/mercury/phase_c_price.py`, `tests/mercury/test_v20_r15_it13_phase_c_price.py` and
this report.

**RED first, verbatim** — the node was written before its module existed:

```
tests\mercury\test_v20_r15_it13_phase_c_price.py:9: in <module>
    from tests.mercury.phase_c_price import (
E   ModuleNotFoundError: No module named 'tests.mercury.phase_c_price'
=========================== short test summary info ===========================
ERROR tests/mercury/test_v20_r15_it13_phase_c_price.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.94s
```

Green: `[RUN] python -m pytest tests/mercury/test_v20_r15_it13_phase_c_price.py -q` ->
**`8 passed in 5.66s`**. Suite: `[RUN] python -m pytest tests/mercury/ -q` ->
**`1 failed, 116 passed in 55.96s`**; the one failure is
`test_v20_r15_it10_eval_independence.py::test_it8_aggregate_verdict_agrees_with_its_own_cells`,
**pre-existing and left RED on purpose since it.10** — it is a claim about a written
journal and no measurement repairs it.

**Two figures other offices published do not survive their own arithmetic, and both are
corrected downward.** Neither correction is a ruling; both are multiplications.

---

## TASK A — THE FOUR CLAUSES, PRICED

### A.1 CLAUSE (1) — BED-M `CP-lower > 0.5`. **DECIDABLE-AFTER-A-REPAIR. THE REPAIR COSTS 0 GPU-SECONDS.**

The missing artifact is a **sentence**, not a cell: `⟨CLAUSE_1_TAIL⟩`, open since it.11
(`V20_R15_JOURNAL.md:1524-1525`), still open at `:2323` where the journal calls it *"the
largest single item on the board"*. The pooled 12/16 reads
**`0.4762` two-sided (fails by `0.0238`)** and **`0.5156` one-sided (clears)**
(`V20_R15_JOURNAL.md:1506`).

**Price of the ruling: 0 GPU-seconds, 0 instrument edits, 0 cells.** Both numbers are
already computed on cells already banked. The clause is decidable the moment the tail is
named and is undecidable for any amount of GPU time until it is.

**Price if the ruling goes two-sided** — the only branch that costs anything. From this
office's it.3 model `GPU-s = sum(per-cell secs) x N + 3.5`, control at `N=8` **1.63% low**
against `V17_R4_RETAKE_PRICE.md:207`: `N=72` (rate-preserving) **~1 378 GPU-s**,
`N=125` (0.80 power) **~2 389 GPU-s = 0.66 GPU-hours**. Both fit this box. **On BED-M,
`N` has never been the constraint.**

### A.2 CLAUSE (2) — BED-K(a), `d = 20`, Hankel ceiling `1/d`. **NOT DECIDABLE ON THE FROZEN WINGS. NOT PRICEABLE.**

Two independent facts, and the first is fatal on its own.

**(i) Neither frozen wing is on this bed.** JUPITER's own ruling on M16 is *"BED-K only…
do not cite it on W1/W3 at all"*, and it is what re-graded Q2/W1 to F4 this iteration.
`V20_R15_JOURNAL.md:1703-1704`: *"M16 Hankel binds BED-K(a) only and **both frozen wings
are BED-M**."* A clause that ranks the frozen wings against a ceiling neither wing is
measured against ranks nothing. **The wing list froze at it.4; changing it is not a price,
it is a different round.**

**(ii) No cell of BED-K's shape has ever been run.** The bed generator is registered
(`ceq/kdata.py:475`, `bed_k` -> `ceq.beds.bed_k.build_delay`) and **zero files under
`scripts/` call it** — `[RUN] grep -rn "label_plies\|iter_games\|split_by_game" scripts/`
returns nothing, and the BED-K builder has no consumer in any runner. `scripts/v15_r1.py`
is BED-M synthetic end to end.

**This office therefore prices clause (2) at NOTHING, and that is the finding.** The
standing rule — *no estimate without its measured basis* — bites here exactly as it bit at
it.3: there is no per-cell second for BED-K's shape anywhere in this tree, and this office
does not extrapolate a cost across a shape it has never seen run. **The it.3 limit
paragraph (`V20_R15_IT3_MERCURY.md:334`) said this ten iterations ago and nothing has
moved it.**

### A.3 CLAUSE (3) — LOWEST GPU-SECONDS-TO-FLOOR. **DECIDABLE-AFTER-A-REPAIR, AND THE REPAIR IS CHEAPER THAN THE ROUND BELIEVES BY 34.8%.**

Three defects stack on this one clause, and they are independent:

| # | defect | named artifact |
|---|---|---|
| 1 | **no `torch.cuda.synchronize()` anywhere in the runner** | `scripts/v15_r1.py:249,:267` — every `secs` on CUDA is un-synchronised host wall clock |
| 2 | **the strongest correlate of `secs` is run order** | `rho = +0.7029, p = 0.0024`, against the gate correlation `+0.5197` |
| 3 | **the cost law's variable has n = 1** | `S, D = 64, 24` is a module constant at `scripts/v15_r1.py:137`; `s = 64` on 40 of 40 cells, nine sibling flags at `:547-557`, zero override paths |

**VERIFIED, NOT RE-DERIVED — and the verification moves both published numbers.**

**(a) The `~275 GPU-s` does not evaluate to 275.** The formula is written out in full at
`V20_R15_IT8_JUPITER.md:216`: `3 x (0.25 + 1 + 4) x (16.16 + 1.78 + 1.68)`.
`[RUN] python -m pytest tests/mercury/test_v20_r15_it13_phase_c_price.py::test_jupiters_own_it8_formula_does_not_evaluate_to_275 -q`
-> **`309.047`**. The label is **11.0% low against its own arithmetic**, and the formula is
**+12.4%** on the label.

**(b) And 309.047 is one point on a band, because the exponent is the unknown the sweep
exists to measure.** The `0.25 / 1 / 4` weights *assume* `S²` — the very law the runs are
supposed to establish. `[RUN] ...::test_the_s_sweep_price_is_a_band_because_the_exponent_is_what_is_unknown`,
same 3 seeds x `S in {32,64,128}` x 3 arms, measured means `arm_smprime 16.161`,
`arm_pl 1.780`, `softmax 1.681` (retake subset, read from `results/v17k_r4_retake.jsonl`):

| assumed law | sweep price | with this office's measured estimate spread `[-5.4%, +13.2%]` |
|---|---|---|
| `S^1` | **206.0 GPU-s** | 194.9 – 233.2 |
| `S^2` (the assumed law) | **309.0 GPU-s** | 292.4 – 349.8 |
| `S^3` | **537.2 GPU-s** | 508.1 – 608.1 |

`275` sits inside the band and is none of its points. **The honest quote is `206 – 537
GPU-s`, not a point.**

**(c) The `684 GPU-s` re-take is 53.6% too expensive, and the cells say so.**
`V20_R15_IT9_JUPITER.md:239` charges `40 x 17.1 s`. `17.1` is **`arm_smprime`'s** mean; 20
of the 40 cells are `arm_pl` or `softmax` at **~1.7 s**.
`[RUN] ...::test_the_retake_of_all_40_costs_317_not_684` sums the 40 cells' **own** `secs`:

```
retake 24 (results/v17k_r4_retake.jsonl)              156.975   <- SATURN's figure, exactly
it.6 seeds 8-15 (v20_r15_it6_seeds8_15.jsonl)         146.399
it.8 seeds 8-15 (v20_r15_it8_armpl_b.jsonl)            13.718
THE 40                                                317.092
40 x 17.1                                             684.0
```

**SATURN's `156.975 s` is confirmed to the millisecond and is the retake-24 subset;** the
full-40 figure is **`317.092`**, not 684.

**(d) SATURN's hash split, and what it does to the 960.**
`[RUN] ...::test_instrument_hash_is_read_by_zero_files_under_ceq_and_scale` — the census
returns **`ceq/ -> []`**, **`scale/ -> []`**, and non-empty for `scripts/` (the control
that shows the searcher works: `scripts/v15_r1.py` plus the three MERCURY re-score scripts,
all **write** sites). The hash **moves on any edit** (`scale/identity_manifest.py:210-215`)
and **is read by no consumer** — so the edit **forks the pool for a hash-binding reader and
invalidates nothing journalled.**

**One correction to SATURN's count, recorded rather than smoothed.** He says *"the only
binds are two MERCURY tests"*. `[RUN] grep -rn instrument_hash tests/` returns **four**
files: `tests/mercury/test_v20_r15_it11_smprime_draws.py:33` (a **literal** bind to
`5d41a63d…`), `tests/mercury/test_v20_r15_it8_armpl_and_clamp.py:36` (cross-file equality),
plus `tests/gate0/test_g14_instrument_hash.py:314` and
`tests/jupiter/test_v20_r15_it9_q6.py:196` — the last two bind the field's *presence and
uniqueness*, not the literal, and survive the edit. **Two sites move; four need review.**

**THE THREE PRICES FOR CLAUSE (3), and the round has only ever been quoted the most
expensive one:**

| route | GPU-seconds | instrument edits | test edits | what it gives up |
|---|---|---|---|---|
| **RETIRE the re-take (SATURN's split)** | **206 – 537** (point WITHDRAWN it.32 under J-31c: band only, NO POINT) | 4 — one `argparse` line, one substitution at `:137`, two `cuda.synchronize()` at `:249,:267` | 0 | the 40 banked cells are non-comparable **to a hash-binding reader only**; nothing journalled is invalidated |
| **REPRICE — re-bind the pool on values, not the hash** | **206 – 537** | 4 | **2** (`it11_smprime_draws.py:33`, `it8_armpl_and_clamp.py:36`) | nothing; 2 sites edited, 2 more confirmed unaffected |
| **RETAKE — close the fork by measurement** | **523 – 854** (point 626.1) | 4 | 0 | nothing — and this is JUPITER's route, honestly repriced |

**JUPITER's `960` against the measured route is `626.1`. The difference is `333.9`
GPU-seconds, `34.8%`,** and all of it is the `40 x 17.1` substitution plus the 275-vs-309
label. **Under SATURN's split the re-take is not owed at all and the price is `206 – 537`
— a fifth to a half of the published figure.**

**THE LIMIT THAT SURVIVES THE REPAIR, and it is not a cost.** Defect 2 — run order at
`rho = +0.7029` — is **not fixed by `synchronize()`**. Synchronising removes the
launch-queue artifact; it does not remove thermal drift, clock ramp, or any other
monotone-in-position effect. **Sequential cells in one process cannot separate order from
arm.** A randomised or blocked execution order is a **fifth instrument edit** and it is the
one nobody has priced. Without it, the 206–537 GPU-s buys a synchronised measurement
carrying the same confound.

### A.4 CLAUSE (4) — WITNESS TIEBREAK. **See TASK B.**

---

## TASK B — THE WITNESS. IT IS TWO-THIRDS BUILT, AND THE MISSING THIRD IS THE ONE THAT NEEDS KAGGLE

**The search, shown.** `[RUN] python -m pytest tests/mercury/test_v20_r15_it13_phase_c_price.py -q`
(nodes `::test_the_chess_witness_is_two_thirds_built_and_registers_as_no_bed` and
`::test_the_two_oracles_that_do_exist_cost_zero_gpu_seconds`), corroborated by an
independent sweep of `ceq/ scale/ scripts/ tests/ data/ results/ colab/ kaggle/ attic/
haskell/ lean/`.

**What EXISTS — and the round has been asserting it does not:**

| component | status | evidence |
|---|---|---|
| PGN reader, split-by-game, FEN dedupe | **built** | `ceq/kdata.py:206,:225,:288` |
| **legality + next-FEN oracle, recomputed** | **built and RUNS TODAY** | `ceq/kdata.py:258` `label_plies` — replays the mainline through a `chess.Board`, emitting `{fen_before, uci, san, fen_after, legal}` |
| `python-chess` | **installed on this box** | `[RUN] python -c "import chess; print(chess.__version__)"` -> **`1.11.2`** |
| fixture corpus | **present** | `tests/gate0/fixtures/games.pgn` 7 624 B, **36 games**; `evals_shard.jsonl` 111 058 B, **512 rows** |
| measured cost of the two oracles | **`[RUN]` this iteration** | **512 oracle rows from 36 games in `0.376 s` of CPU. `0.000` GPU-seconds.** |

**What DOES NOT EXIST:**

| component | evidence |
|---|---|
| **a registered chess bed** | `[RUN] python -c "from ceq import kdata; print(list(kdata.BED_SPECS))"` -> **`['bed_m', 'bed_k', 'bed_1']`**. The leap ledger calls the witness *"the one registered bed with a categorical state space"* (`V20_R15_LEAP_LEDGER.md:131`). **It is not registered.** |
| **the eval-Δ sign oracle** | `[RUN] grep -rn "eval_delta\|cp_delta\|centipawn" ceq/ scripts/ scale/ tests/` -> **0 hits** |
| **an engine to recompute it with** | `stockfish\|SimpleEngine\|popen_uci` -> the only non-snapshot hit is **this office's own search regex**; the exclusion is written in the source, not silent |
| **any arm that consumes a chess corpus** | `[RUN] grep -rn "label_plies\|iter_games\|split_by_game" scripts/` -> **0 hits** |
| **the real corpus** | `results/k_data_manifest.json:125,:137` — `lichess/chess-evaluations` (34.4 GB) and `arevel/chess-games` (1.56 GB), both `UNPINNED_AWAITING_KAGGLE` (`COSTS.md:36-37,:43`) |
| **`python-chess` as a pin** | `[RUN] grep -i chess requirements.txt requirements-kaggle.txt` -> **0 hits**, though `V17_G05_DATA.md:388` asserts *"the pin that matters is `chess==1.11.2`"* |

And `ceq/kdata.py:11-13`'s own docstring rules the module out as a bed:
*"WHAT THIS MODULE IS NOT. It builds no corpus and defines no task."*

**THE PRICE OF THE WITNESS, in three rows:**

| what | GPU-seconds | build | blocker |
|---|---|---|---|
| legality + next-FEN on the **fixture** | **0.000** (0.376 s CPU, 512 rows) | **0** — it runs today | none |
| the same on a **real** corpus | 0.000 | corpus attach | **KAGGLE** — 1.56 GB, unpinned |
| **eval-Δ sign, oracles recomputed** | unmeasurable | **an engine integration, a bed registration, and an arm that consumes it — none of which exist** | **KAGGLE** — 34.4 GB, unpinned |

**The brief's `~0 GPU-seconds and an unbounded amount of build` is right in structure and
wrong in one direction: two of the three oracles are already built and already run.** The
witness is not absent. It is **stalled on a Kaggle attach**, and Kaggle is the one carve-out
from loop autonomy that requires the author's explicit yes. **This office did not touch it.**

---

## TASK C — THE HONEST TOTAL

**Phase C as scheduled, clause by clause.**

| clause | it. | **decidable?** | GPU-seconds | edits | what it returns as written |
|---|---|---|---|---|---|
| **(1)** BED-M `CP-lower > 0.5` | 17–24 | **AFTER A REPAIR — and the repair is a sentence** | **0** for the ruling; 1 378 at `N=72`, 2 389 at `N=125` if two-sided | 0 | one-sided: **fires**. two-sided: **fails by 0.0238** and needs `N >= 72`. Undecidable at any price until `⟨CLAUSE_1_TAIL⟩` is ruled |
| **(2)** BED-K(a) within `1/d` | 25–27 | **NO** | **unpriceable — no cell of this shape has ever run** | unknown | **nothing.** Both frozen wings are BED-M; M16 is *"BED-K only, do not cite it on W1/W3 at all"*. The clause ranks the wings against a ceiling neither is measured against |
| **(3)** lowest GPU-s-to-floor | 17–27 | **AFTER A REPAIR** | **206 – 537** (fork accepted) · **523 – 854** (fork closed) | **4**, +2 test edits on the reprice route | a **synchronised** cost law over three lengths — **still confounded with run order** unless a 5th edit randomises execution |
| **(4)** witness tiebreak | 28 | **PARTIAL** | **0.000** measured for 2 of 3 oracles | 0 for those two | legality + next-FEN, **today**. eval-Δ sign: **no producer, no engine, no registered bed, no arm — and 34.4 GB behind a Kaggle attach** |

**THE TOTAL, and it is smaller than the round has been quoting:**

| | published | **measured / verified here** | delta |
|---|---|---|---|
| the `S`-sweep runs | `~275` | **206 – 537**, point **309.0** | its own formula is **+12.4%** on the label |
| the re-take of 40 banked cells | `684` | **317.092** | **-53.6%** |
| the honest total | **`960`** | **626.1** fork-closed · **309.0** fork-accepted | **-34.8%** · **-67.8%** |

**AND THE ANSWER TO THE QUESTION THE BRIEF ASKED.**

**PHASE C AS WRITTEN CANNOT PRODUCE A WINNER, and it is not the cost that stops it.**
Lexicographic order means clause `k` is reached only if clause `k-1` ties.

1. **Clause (1) has no text.** It is not that the answer is unknown — the two candidate
   readings give **opposite verdicts on the same 16 cells** (`0.4762` fails, `0.5156`
   clears). An arena whose first clause has two readings has two winners.
2. **Clause (2) is unreachable on the entrants.** Both frozen wings are BED-M; the bound
   the clause names binds BED-K only, by JUPITER's own ruling. **No amount of GPU time on
   BED-K(a) ranks two BED-M arms.** This is the clause money cannot repair.
3. **Clause (3) is measurable after 206–537 GPU-s and still will not be clean**, because
   `synchronize()` does not remove a run-order confound at `rho = +0.7029` and no office
   has priced the randomisation that would.
4. **Clause (4) is one third missing, and the missing third is behind a Kaggle attach this
   round is not authorised to make.**
5. **And every F-grade the ranking is expressed in is scored on a scale with no text** —
   `L-GRADE (F0–F4)` occurs **exactly once** in the whole tree, at
   `CEQ_V20_R15_CONTRACT.md:58`, as a citation (SATURN it.12 §A, with a searcher shown to
   find both a real law and a planted one).

**Two iterations before Phase C opens, the arena has one clause with two readings, one
clause its entrants are not on, one clause with a confound nobody has priced, one clause
two-thirds built behind a gate that needs the author's yes, and a grading scale that was
never written down. The GPU time was never the problem. `626.1 GPU-seconds` buys clause
(3) and nothing else.**

**This office does not recommend and does not rule.** The four repairs are priced above;
three of them cost **zero GPU-seconds** and one of them costs **a sentence**.

---

## LIMITS

Every `secs` in every figure above is **un-synchronised CUDA host wall clock** — the cost
basis carries the exact defect the clause-(3) repair exists to remove, so the 206–537 band
is an order-of-magnitude price and not a budget. The `S^1/S^2/S^3` band brackets the
exponent but does not contain it: the true law could sit outside `[1, 3]`, and the band is
a sensitivity, not a confidence interval. `MEAN_SECS` is the retake-24 subset on this box's
certified RTX 4060 in one process; this office's prior estimate spread on that basis is
`[-5.4%, +13.2%]` (it.11, it.8, it.6) and is applied as a range, never as a point. The
`317.092` re-take assumes one process and the same sequential order that produces the
`rho = +0.7029` confound. The `0.376 s` witness measurement is the **36-game fixture**,
which `V17_G05_DATA.md:375-377` states *"has never seen a real Lichess PGN"* — it prices
the oracle's shape, not a corpus. `python-chess 1.11.2` is installed on this box and is
pinned in **neither** requirements file, so the witness's zero-GPU-second price is
contingent on an unpinned dependency. The clause-(1) `N` prices reuse the it.3 model
(`sum(secs) x N + 3.5`), whose control is 1.63% low at `N=8`. The chess census's engine
scan excludes `kaggle/snapshot/` and this office's own regex file; the exclusion is a
prefix filter written in `tests/mercury/phase_c_price.py`, not a silent one. **No git
write, no Kaggle contact, no training run, no frozen file opened for write.**
