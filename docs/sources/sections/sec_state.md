# §S — WHERE ROUND 15 LEFT THE CAMPAIGN, AND WHAT BINDS THE NEW PROGRAMME

*Prepared by the MERCURY/LESTRADE reader, 2026-09-03, against HEAD `207e7b9` on branch `v17k-gate0` `[RUN: git log -1]`. Every load-bearing sentence carries an evidence class: `[READ path:line]` quotes the record at HEAD; `[RUN]` was executed in this session; `[DERIVED]` shows its steps. No external source was fetched this session, so no claim below is `CITED`; every external identifier that appears is `READ` from the record and is inherited with the record's own tag.*

**Reading rule for this section.** The round's own journal is append-only and its head block grows, so its line numbers drift (`V20_R15_JOURNAL.md:100`, *"headings are"* stable, lines are not). Journal citations below therefore carry both the line at HEAD and the heading; a reader who finds the line moved should search the heading. This is the record's own repair for `MISTAKES.md` P-6 (line-reference drift, `MISTAKES.md:364`) and it is adopted here rather than re-invented.

---

## S.1 THE SCOREBOARD AND THE PHASE STATE AT it.35

**The round.** CEQ v20 Round 15 is a 45-iteration local tournament on the certified RTX 4060 with one scheduled leap call at it.35, five phases — A discovery it.1–5, B arming it.6–14, C arena it.15–30, D leap gate it.31–35, E strongest candidate it.36–45 — and a scoreboard ceiling of 48 `[READ CEQ_V20_R15_CONTRACT.md:34-37, :76-169, :275-283]`. The contract file is verbatim-of-record and is never edited; corrections live in the journal and in `MISTAKES.md` `[READ CEQ_V20_R15_CONTRACT.md:3-6]`. The leap model is Claude Fable 5.1, one call, dossier in, dossier out; a leap on anything older does not count `[READ CEQ_V20_R15_CONTRACT.md:39-43, :72-73]`.

**Where the scoreboard stands.** After thirty-five iterations the carried score is **2 of 44** `[READ V20_R15_JOURNAL.md:8242-8243, heading "## it.35 — THE EVALUATION-FOR-LEAP GATE"]`. The two points are the it.4 freeze (*wing list frozen, four citations each +2*) and nothing else has been banked `[READ V20_R15_JOURNAL.md:15]`. The denominator is 44, not the contract's 48, because VENUS's it.5 kill showed the *winner's cost ≤ 1/10 incumbent* bonus (+4) is unreachable by either surviving wing: on both timing bases W3 costs **more** than softmax, `1.078×` and `1.059×` `[READ V20_R15_JOURNAL.md:951-966, heading "### VENUS'S KILL — THE SCOREBOARD'S CEILING IS NOT 48"]`.

**The `+6` outcome.** The theory table (+6, *N×Q6 all graded*) was frozen at it.14 and **ruled NOT FIT for the leap**; the `+6` was refused at it.10, refused a second time at it.11, and is carried as *unclaimed* through it.35 `[READ V20_R15_JOURNAL.md:16, :1801, :2088, :8242]`. The `+12` (a winner crossing BED-M and surviving BED-K(a)) is gated on a single author ruling, `⟨CLAUSE_1_TAIL⟩`, *"free, 0 GPU-s, and it decides whether Phase C returns a winner at any price"* `[READ V20_R15_JOURNAL.md:8241-8242]`.

**Distance to the north star, as the record states it.** *"Unmoved by measurement for a twenty-fifth consecutive iteration."* `arm_pl` crosses `floor₁` on 8 of 9 paired against `softmax` 0 of 9 across three eval draws; `arm_smprime` seed 2 crosses on all three at `0.490590` below the floor with `n_cells = 1` `[READ V20_R15_JOURNAL.md:8228-8230]`. Since `floor₁` is a capability threshold and not an information floor (S.3 below), the sentence is a crossing count against a threshold, not a distance.

**The phase state, exactly.** Phases A and B closed (A at it.5, B at it.14). Phase C ran it.15–30 and returned no winner: the arena's four lexicographic clauses read, per wing, (1) scoreable but blocked on the tail ruling, (2) **VOID**, (3) not scoreable as written, (4) not scoreable `[READ V20_R15_THEORY_TABLE.md:235-240]`. Phase D ran it.31–35; the gate was graded twice, independently, at it.35 (S.3), and **the leap was not called**: *"calling it now would consume the round's single shot on a table whose header miscounts its own body, whose published grading rule contradicts its applied one, and three of whose cells the gate cannot read"* `[READ V20_R15_JOURNAL.md:8234-8238]`. Phase E has not begun. Iterations it.36–45 are unspent.

**A stale table at the journal's head, flagged.** The journal's PHASE SCHEDULE still reads *"it.15 DONE … it.16 next"* for Phase C and *"not started"* for Phases D and E `[READ V20_R15_JOURNAL.md:17-19]`, while the body carries entries through it.35 `[READ V20_R15_JOURNAL.md:8011]`. By the journal's own definition this is P-3, a stale claim never retracted (`MISTAKES.md:316`), sitting in the block built to catch P-3. The paper's programme must not read its start state from that table; it reads it from the it.35 entry.

**The it.35 arena ticket, per clause and per wing** (the numbers the next programme inherits, not re-derived here):

| clause | W1 `arm_smprime` | W3 `arm_pl` | status |
|---|---|---|---|
| (1) BED-M CP-lower > 0.5 | `1/16`, rate `0.0625`, CP-lower `0.0016` two-sided / `0.0032` one-sided | `12/16`, rate `0.7500`, CP-lower **`0.4762` two-sided (fails by `0.0238`) / `0.5156` one-sided (clears)** | blocked on `⟨CLAUSE_1_TAIL⟩` |
| (2) BED-K(a) within `1/d` | not scoreable | not scoreable | **VOID** — zero cells of BED-K's shape have ever been run |
| (3) lowest GPU-s-to-floor | not scoreable as written | not scoreable as written | four instrument edits plus a fifth unpriced one; "floor" here means `floor₁` |
| (4) witness tiebreak | not scoreable | not scoreable | no chess bed registered; eval-Δ oracle has 0 producers, behind a 34.4 GB Kaggle attach |

`[READ V20_R15_THEORY_TABLE.md:235-240]`. Separation that does not depend on the tail: Fisher exact one-sided `arm_pl` vs `arm_smprime`, `p = 8.544e-05` `[READ V20_R15_THEORY_TABLE.md:277-278]`. The crossing counts pool cells whose control is incomplete — softmax exists only for seeds 0–7 `[READ V20_R15_THEORY_TABLE.md:438-440]`.

**The corrections that void published numbers.** The journal's CORRECTIONS INDEX runs `C1`–`C40`, contiguous, `INDEX-SHA256 = 12eaa9f53b452a199a3820ac9294f73ce9572b071818b5333dad1d3012e46e86`, with the recomputing one-liner printed beside it `[READ V20_R15_JOURNAL.md:94-98]`. The rows a reader of any earlier R15 document must apply before quoting it:

| row | the claim as filed | what is true |
|---|---|---|
| C1 | *no arm crosses BED-M's `floor₁`* | 6 of 24 cells cross `[READ V20_R15_JOURNAL.md:37]` |
| C13 / C14 / C16 | *12 of 16* as the W3 crossing headline; *7 of 8 vs softmax 0/8, `p = 6.730e-04`* | the file's own `agg` record says `crosses: false`; the flip is draw-invariant on seed 9; the `p` was unpaired — honest paired figure `p = 0.012821`, a factor of 19; repaired by measurement to 8 of 9 vs 0 of 9 paired `[READ V20_R15_JOURNAL.md:49-50, :52]` |
| C15 | *distance to `floor₁`* as an information floor | `floor₁` is a one-hop capability threshold; under L-FLOOR the honest distance is to the exact oracle at `0.0` `[READ V20_R15_JOURNAL.md:51]` |
| C17 | every GPU-second and cost ratio from it.3 onward (`0.3455` GPU-h, `10.17×`, `41.9×`) | no `synchronize()` in the harness; run order is the strongest correlate of `secs` `[READ V20_R15_JOURNAL.md:53]` |
| C18 | *N = 1 primitive, not 3* | STRUCK as UNBOUND — the only test asserting the algebra was the adversary's control; re-bound at it.4 as two arena entries, one primitive `[READ V20_R15_JOURNAL.md:54]` |
| C20 | M14 at F1 / F3 / F4 in three files | M14 is `F4`, the only live grade (J-17d); the `≥12 at F0/F1` annex bonus is already forfeit `[READ V20_R15_JOURNAL.md:56; V20_R15_LEAP_LEDGER.md:367-372]` |
| C35 | the `overturns:` enforcement node, `8 passed` | the node could not fail (`live_hits()` short-circuited); V-16 inside a repair `[READ V20_R15_JOURNAL.md:71]` |

Two rows were repaired by measurement rather than withdrawn (C14's pairing; the `n_eff = 1` objection, 54 scorings, zero flips) and are listed as corrections all the same `[READ V20_R15_JOURNAL.md:85-88]`. **Design-against:** the paper's own numbers ship with an index of this shape from the first draft, so a superseded figure is carried to its retraction (P-3, `MISTAKES.md:316`).

---

## S.2 THE FROZEN WINGS, THE STRUCK WING, AND THE FOUND-vs-NAMED DISCRIMINATOR

**Frozen at it.4.** `FROZEN-N = 2`, `FREEZE-SHA256 = fbf17e07e6cab495bd1fdbb4392ab88e92f5b0610e40d2b8a283d8bcb9c542ff`, `RETIRED = arm_phase PRICE 5.114 s` `[READ V20_R15_WING_MANIFEST.md:11-13]`. The digest is `sha256` over the eight `wing|clause|path:line|anchor` rows, normalised and sorted; it covers which wings, which lines, which anchor strings, and does **not** cover the content of the cited files, which is why the anchor check is a separate node re-run at every HEAD `[READ V20_R15_WING_MANIFEST.md:134-146]`.

| wing | primitive | (a) distinct primitive | (b) journalled cell | (c) accepted kill | (d) cost footprint |
|---|---|---|---|---|---|
| **W1** ARM S-M′ `arm_smprime` | path product `G_ij = Π m_k e^{iθ_k}` with `β`/`QK`/`g` switches | `ceq/arm_smprime.py:144` `def path_product` `[RUN: sed]` | `results/v17k_r4_retake.jsonl:161`, seed 0, `eval_nrmse 0.9260818361`, `secs 16.164` | `V16_ARM_SMPRIME.md:529` | `V17_R4_RETAKE_PRICE.md:194`, `15.970` |
| **W3** ARM PL `arm_pl` | real prefix scan `C = scan(g)` key-side only over a value-zero BOS sink | `ceq/arm_pl.py:88` `def scan` `[RUN: sed]` | `results/v17k_r4_retake.jsonl:25`, seed 0, `eval_nrmse 0.6446726192`, `secs 1.884` | `V15_LEDGER.md:786` | `V17_R4_RETAKE_PRICE.md:195`, `1.614` |

`[READ V20_R15_WING_MANIFEST.md:20-48]`. Wing ids are held at their it.1 values so the strike of W2 is visible as a gap, not hidden by renumbering `[READ V20_R15_WING_MANIFEST.md:25-26]`. Clause (d) carries the it.1 reservation unchanged: these are GPU-seconds per cell at a matched 150-step budget, not GPU-seconds-to-floor, because no arm crosses the floor and criterion (3)'s statistic does not exist for any wing — K5, unrepaired `[READ V20_R15_WING_MANIFEST.md:50-53]`.

**The discriminator, stated so it can be run.** *A wing is FOUND iff `results/` holds at least one journalled record whose `kind` field is the wing's arm name. Otherwise it is NAMED — it exists in source and in prose, and the ledger has never seen it* `[READ V20_R15_WING_MANIFEST.md:59-63]`. That is clause (b) with the charity removed: (d) asks what a cell costs, (b) whether a cell was ever produced `[READ V20_R15_WING_MANIFEST.md:65-69]`.

**The census that struck W2.** Records by `kind` over every `results/**/*.jsonl`: `arm_pl` **191** (3 files), `softmax` 182 (4), `arm_smprime` **182** (2), `arm_phase` **0** (0) `[READ V20_R15_WING_MANIFEST.md:73-78]`. The counter is calibrated on both sides — it must read `0` for `arm_phase` and `> 0` for `softmax` — so a zero is a measurement, not a broken search (V-7, `MISTAKES.md:117`) `[READ V20_R15_WING_MANIFEST.md:80-83]`. Four corroborating reads: `theta` drawn `U(−π,π]` and never fit; `make_arm` at `scripts/v15_r1.py:179-184` has branches for `arm_pl` and `arm_smprime` only, anything else dies at `ceq/bench.py:373 raise ValueError(kind)`; `GATED_ARMS` and `ARM_MODULES` at `:146`/`:149` exclude it; the module says of itself *"NOT TRAINED HERE … No optimizer, no gradient"* (`ceq/arm_phase.py:476`) `[READ V20_R15_WING_MANIFEST.md:88-100]`. **W2 is STRUCK and RETIRED to UNPRODUCED — not refuted** `[READ V20_R15_WING_MANIFEST.md:105-108]`.

**The route back, priced.** One W2 cell `5.114 s` on the `arm_pl` basis, `19.470 s` on the expensive basis; eight cells ≈ 130 GPU-s on the `arm_smprime` ceiling, ≈ 41 GPU-s on the `arm_pl` basis; a three-line `make_arm` branch, an optimizer path, a `PHASE_FIELDS` emit; the cell is `cumsum`-based and non-deterministic under `warn_only=False` `[READ V20_R15_WING_MANIFEST.md:115-123]`. *"The W2 gate is under half a minute of wall clock on this box. It was never a cost decision"* `[READ V20_R15_WING_MANIFEST.md:125-126]`.

**Planted negative on the freeze.** `test_moving_one_citation_by_one_line_fires_both_binds` adds `1` to the clause-(a) line and demands two failures — digest moves and `resolve()` raises; one without the other would show which check is asleep; GREEN `[READ V20_R15_WING_MANIFEST.md:148-152]`. All twelve it.1 citations, W2's four included, still resolve at HEAD `207e7b9`; zero P-6 drift `[READ V20_R15_WING_MANIFEST.md:154-159]`. **Design-against for the new programme:** any wing or arm the paper proposes enters under this discriminator (V-7 calibrated on both sides; L-FIND) — a `kind` in `results/` or it is NAMED.

---

## S.3 THE THEORY TABLE AND THE EIGHT GATE CELLS, BOTH OFFICES' VERDICTS

**At a glance** — `N = 2 × Q1–Q6`, twelve cells, `1×F0 / 5×F1 / 1×F2 / 2×F3 / 3×F4` `[READ V20_R15_THEORY_TABLE.md:3]`:

| | W1 `arm_smprime` | W3 `arm_pl` |
|---|---|---|
| Q1 exact class | **F0** | F1 |
| Q2 outside the class | **F4** (re-graded it.12) | F1 + const |
| Q3 learnability | F2 | F1 + const |
| Q4 cost law | F3 | F3 |
| Q5 information floor | F1 + const | F1 + const |
| Q6 state metric | **F4** | **F4** |

`[READ V20_R15_THEORY_TABLE.md:115-122]`. Three facts precede any grade. (i) The F0–F4 scale has no canonical text; the rubric used is a reconstruction ruled BOUND to its own usage and to no other — F0 exact by machine-checked declaration, F1 a bound with its constant, F2 a direction with no constant, F3 a fact about the harness, F4 the theory's domain is empty for the graded wings; `F4` is overloaded three ways in the ledger and `F2`/`F3` are not ordered `[READ V20_R15_THEORY_TABLE.md:20-51]`. (ii) `floor₁ = √((t*−1)/t*) = 0.7071067811865476` is a one-hop capability threshold, violated by **13 of 40** deduplicated cells, and *of four candidate floors zero bind both frozen wings as information floors*; the real BED-M floor is the exact oracle at `0.0` and no annex theorem predicts either wing's distance to it `[READ V20_R15_THEORY_TABLE.md:71-91]`. (iii) Q4's two `F3`s are one fact about the harness: `S, D = 64, 24` is a module constant at `scripts/v15_r1.py:137` `[RUN: sed -n 137p]`, `s = 64` on 40 of 40, nine sibling argparse flags `[RUN: grep -c add_argument = 9]` and no override for `S` `[RUN: grep seq_len → no hit]` `[READ V20_R15_THEORY_TABLE.md:95-102]`.

**The gate's population is eight, not nine.** `§4.2` is headed *"THE NINE CELLS THAT DO REACH THE GATE"* and its ninth row is `Q1/W1`, `F0`, *"not a failure"* `[READ V20_R15_THEORY_TABLE.md:319-331]`; both graders measured eight `[READ V20_R15_IT35_JUPITER.md:61-64; V20_R15_IT35_MARS.md:14-16]`. JUPITER goes further: `§0.3` rules Q4's two `F3`s one fact, so the gate graded *seven distinct failures on eight cells* `[READ V20_R15_IT35_JUPITER.md:52-55]`.

**The eight cells, both offices, no coordination** (JUPITER: *"MARS's parallel filing exists on disk and was not opened"*; MARS: *"JUPITER's it.35 filing was not opened"* `[READ V20_R15_IT35_JUPITER.md:150-151; V20_R15_IT35_MARS.md:4]`):

| cell | grade | JUPITER | JUPITER's field / bound | MARS | MARS's field / bound + killer |
|---|---|---|---|---|---|
| Q1/W3 | F1 | LEAPABLE | realization theory for LTI systems (Hankel/Kronecker) | LEAPABLE | formal verification of recurrence invariants — induction on a telescoping product (Lean 4 / Mathlib `Finset.prod_range_succ`) against `CEQ.V15.scan` |
| Q2/W3 | F1+c | TERMINAL | bound `err_i ≥ dist(t_i, hull)`: row-stochastic ⇒ convex combination for every `(g,s,q,k)`; a nonexistence, no `k` at any size | TERMINAL | same bound; killer: one `(g,s,q,k)` with `Z_i ≠ 1` |
| Q3/W1 | F2 | LEAPABLE | transfer-operator / Koopman spectral theory | LEAPABLE, conditional on an instrument swap | same field, attached to `lambda_hat_live` @ `scripts/v15_r1.py:384`, not `lambda_hat` @ `:383` (one bit) |
| **Q3/W3** | F1+c | **LEAPABLE** | bifurcation theory / gradient-flow convergence | **TERMINAL** | causal direction unidentified from `n = 8`; killer: a declaration predicting `sign(λ̂)` from initialisation alone |
| Q4/W1 | F3 | TERMINAL | exponent in `S` unidentified, `n = 1`; no theorem supplies a slope from one point | TERMINAL | same; killer: clause (3) re-read as asymptotic rather than wall-clock |
| Q4/W3 | F3 | TERMINAL | same harness fact; `brute_force_path_sums` @ `ceq/arm_pl.py:304` is `O(2^S)` unguarded — an implementation guard, not a theorem | TERMINAL | same; the `O(2^S)` finding is not gate material in either direction |
| Q5/W1 | F1+c | LEAPABLE, **field contested** | approximation theory / Kolmogorov n-width — not `§4.2`'s rate–distortion | LEAPABLE, **field corrected** | approximation theory / expressivity bounds — not rate–distortion |
| **Q5/W3** | F1+c | **LEAPABLE** | finite-mixture inference | **TERMINAL** | a mean over a bimodal population is not a statement about either mode; killer: a simultaneous component-wise coverage theorem from pooled bootstrap draws |

`[READ V20_R15_IT35_JUPITER.md:41-48; V20_R15_IT35_MARS.md:27-34]`. JUPITER's tally: 5 LEAPABLE, 3 TERMINAL `[READ V20_R15_IT35_JUPITER.md:50]`. MARS's: 4 LEAPABLE (one conditional), 4 TERMINAL `[DERIVED from V20_R15_IT35_MARS.md:27-34]`. Agreement `6 of 8` on the token; both against the table on Q5/W1's field `[READ V20_R15_JOURNAL.md:8131-8166, heading "### THE TWO GRADINGS, SIDE BY SIDE"]`.

**The two divergences and why they could occur at all.** The two rows that diverge, `Q3/W3` and `Q5/W3`, are exactly the two MARS separately measured as having evaded RULING J-14b by moving the fork out of the verdict column: `§4.2` reads *"LEAPABLE, but ~6 GPU-s buys it outright"* and *"LEAPABLE — and it is a SCORING RULE, not a theorem"* — hybrids placed in the gate-class and FIELD columns where J-14b (*a hybrid verdict is not a verdict*, `V20_R15_THEORY_TABLE.md:358`) does not reach `[READ V20_R15_IT35_MARS.md:46-60, :68-76]`. JUPITER found the mechanism beneath the disagreement: the ledger publishes one rule — *LEAPABLE when the gap is a missing statement **or a missing measurement*** (`V20_R15_LEAP_LEDGER.md:11-14`) — while J-14b resolved `L-9` to TERMINAL against that rule at `V20_R15_THEORY_TABLE.md:353-356`; *"the round has published one grading rule and applied another — twenty-one iterations, it.14 to it.35"* `[READ V20_R15_IT35_JUPITER.md:23-30]`. Under JUPITER's stated rule (TERMINAL iff a bound **or a datum the record never wrote**; LEAPABLE iff the measurements exist and a statement is missing) `Q3/W3` stays LEAPABLE; under MARS's application of the same rule a cell repaired by an experiment or a scoring rule *has no missing statement for a theorem to supply* and is TERMINAL `[READ V20_R15_IT35_JUPITER.md:14-21; V20_R15_IT35_MARS.md:57-60, :72-76]`. The divergence is a missing amendment, not a judgement call `[READ V20_R15_JOURNAL.md:8167-8184, heading "### AND JUPITER FOUND WHY THEY COULD DIVERGE AT ALL"]`.

**The definitional strike on TERMINAL.** Both Q4 cells satisfy *no theorem removes it* and both are removed by one `argparse` line, their own ROUTE. *"The round is using TERMINAL to mean not leap material, a predicate F4's NOT-PUT already occupies"* — carried to the author as `⟨TERMINAL_VS_NOT_PUT⟩` `[READ V20_R15_IT35_MARS.md:127-133]`. The contract's grammar has two tokens; the evidence has four states `[READ V20_R15_JOURNAL.md:8069-8080, heading "### A DEFINITIONAL STRIKE ON TERMINAL"]`.

**The three cells the gate cannot see.** Under RULING J-14 an `F4` is NOT-PUT — neither bound nor missing statement — and ships an ADMISSION CONDITION `[READ V20_R15_THEORY_TABLE.md:291-302]`: `Q2/W1` needs one cell of BED-K's shape run (`build_delay` @ `ceq/beds/bed_k.py:236`, whose only two callers under `scripts/` consume the kernel and never the bed's data); `Q6/W1` needs a distributional head — an arm change under a frozen wing list; `Q6/W3` needs the head **plus** a metric that is not permutation-blind, because marginal `W1` is REFUTED on this bed — a predictor returning the oracle's own values in the wrong order scores `W1 = 0.0` at `NRMSE = 1.421901019003236`, and metric and bed rank two predictors in opposite orders by `14.465410797679917×` `[READ V20_R15_THEORY_TABLE.md:304-308, :219-225]`. Both offices flag that filing `Q6/W3` NOT-PUT files a refutation off the board; `⟨F4_GATE⟩`, opened at `V20_R15_JOURNAL.md:3965`, is in its seventeenth iteration unruled `[READ V20_R15_IT35_JUPITER.md:130-140; V20_R15_IT35_MARS.md:139-151]`.

**Two facts about the gate's own input, measured at it.35.** The table has drifted off its frozen census at HEAD before the window opened — `133` occurrences against a frozen `131`, `119` scored against `118` — and the single unscored citation sits inside the `Q2/W1` admission row the gate is forbidden to grade, in a bare-pointer notation (`:462`, `:152`) no census in the round can parse `[READ V20_R15_IT35_JUPITER.md:229-233, :242-259]`. And the grader of the leap's input did not open it: MARS *"re-verified not one citation … `129 of 129` is taken on trust"* `[READ V20_R15_IT35_MARS.md:183-185]`. **What the new programme inherits from S.3:** Q5/W1's field is approximation theory on both readings; the `~6 GPU-s` capped run at seeds 2, 3, 7 has been priced at it.7, it.8, it.9 and it.35 and taken zero times `[READ V20_R15_IT35_JUPITER.md:157-159]`; the Q6 pair says the arms return one real per draw and no state axis exists — which is the D-1 diagnosis (`MISTAKES.md:677`) arriving from the metric's side.

**The leap ledger's inventory.** Opened at it.7 on the author's instruction *"if something fails, all that adds to the leap"*; rows are appended when the failure is produced `[READ V20_R15_LEAP_LEDGER.md:3-9]`. At HEAD it carries 24 digested units — `L-1`..`L-17`, `L-M1`..`L-M5`, `L-V1`, `V-it7` — each with a `sha256[:16]` row digest so an in-place edit is named, not merely noticed; the planted negative mutates `L-9` in memory and demands the instrument report `["L-9"]` `[READ V20_R15_LEAP_LEDGER.md:202-251]`. Seven F4 objects, six of them carrying a LEAPABLE/TERMINAL stamp the gate cannot consume (`L-3`, `L-4`, `L-7`, `L-8`, `L-13`, `L-14`); `L-16` is filed NOT-PUT and is the seventh `[READ V20_R15_LEAP_LEDGER.md:285-299, :431-436]`. The four hybrid verdicts that were JUPITER's own were resolved to single tokens at it.14 — `L-2` TERMINAL, `L-9` TERMINAL, `L-13` and `L-14` NOT-PUT with FIELD withdrawn `[READ V20_R15_THEORY_TABLE.md:364-369]`. The fields the ledger names for the new programme's benefit, discipline first: realization theory (L-1), Koopman spectral theory (L-5), bifurcation / gradient-flow convergence (L-6, L-12), structured-sparsity kernel scheduling for the certified banded-suffix mask (L-10), finite-mixture inference (L-15), anti-concentration / small-ball probability for the seed-2 crossing (L-M3), error-statistical severity for one-tailed falsifiers (L-M1), non-convex optimisation's implicit-bias branch for corner descent (L-V1) `[READ V20_R15_LEAP_LEDGER.md:22-29, :68-71, :85-89, :160, :181]`. The field-naming rule that governs them: *naming a field and then saying what you want from it is compliant; naming the want and calling it a field is not* `[READ V20_R15_THEORY_TABLE.md:335-336]`.

**The annex M1–M16, as graded.** The contract pre-registered sixteen items with `[RUN]` instances `[READ CEQ_V20_R15_CONTRACT.md:172-263]`. What the round did to them: M14 (Cheeger) is `F4`, struck by V-25 before its own sweep-cut condition was discharged, with the contract's pre-registered `F3` left standing under a `SUPERSEDED` pointer `[READ CEQ_V20_R15_CONTRACT.md:244-253]`; M11 Fano and M12 rate–distortion have `0 of 3` beds and `0` producing `.py` — prose `[READ V20_R15_THEORY_TABLE.md:87-90]`; M13 Wasserstein is cited by annex number only and has no producer in the tree, its `[RUN: 0.492 vs KL 0.519]` resolving to an unrelated quantity in `scale/foreman_consequence.py:12` `[READ V20_R15_THEORY_TABLE.md:214]`; M16 Hankel binds BED-K only, where nothing this round was graded `[READ V20_R15_THEORY_TABLE.md:152]`; M9's F1/F1′ machinery (Cantelli, Azuma, union bound) has `0` producing `.py` and is not needed on W1, whose zero-gate segmentation is F0-exact by `pathProd_eq_zero_iff` `[READ V20_R15_THEORY_TABLE.md:182]`; M1's transfer to W1 was struck (`ρ(β, nrmse) = −0.0324, p = 0.9053`) while M1 itself stands `[READ V20_R15_JOURNAL.md:47]`. The annex's `+4` (*≥12 at F0/F1 and M14 resolved*) is forfeit `[READ V20_R15_LEAP_LEDGER.md:367-372]`. For the new programme the annex items with a producer and an instance in the tree — M1 (`β`-gradient), M2 (gate landscape), M4 (identifiability from `do()` pairs, `a_i = Δz_i/δ` at `1e-9`), M9-F0 (exact segmentation), M10 (influence persistence `4096 → … → 1`) — are the ones that can be cited as `[V-eq]`; the rest re-enter only with a bed in their domain (L-DOM).

---

## S.4 THE LAWS IN FORCE — ONE LINE EACH, WITH SOURCE

The round-15 contract binds *all standing* laws plus its own two `[READ CEQ_V20_R15_CONTRACT.md:57-67]`. The ones the new programme must design under:

| law | one line | source |
|---|---|---|
| **D-1** dependency | Work is a DAG; parallel dispatch only on nodes with no shared repository state; git is the shared state; dispatched agents never `git add`/`commit` | `[READ CONTRACT.md:27-33]` |
| **D-2** skills are modes | Planet names label responsibilities inside documents, not concurrent processes | `[READ CONTRACT.md:35-38]` |
| **D-3** loop gate | No autonomous loop mounts until the deactivation commit, `MISTAKES.md` and the standing loop failures are read and the previous loop's cause of death is one sentence; the iteration count comes from the DAG's critical path, never chosen by an agent | `[READ CONTRACT.md:40-44]`; discharged for R15 at `[READ CEQ_V20_R15_CONTRACT.md:10-37]` |
| **D-4** order | Contracts that schedule work behind an unreached round are staged, not started | `[READ CONTRACT.md:55-57]` |
| **L-DOM** | Every theorem that gates a run ships a domain census — the corpus's value support against the theorem's hypotheses; no overlap ⇒ decoration | `[READ CEQ_V16_CONTRACT.md:54-56]`; pays for V-25 `[READ MISTAKES.md:1954-1956]` |
| **L-SIGN** | A counter-prediction of equal specificity beside every prediction; a calibration column across rounds; a one-sided sign discounts the author | `[READ CEQ_V16_CONTRACT.md:58-61]`; pays for D-7 `[READ MISTAKES.md:2037-2039]` |
| **L-DIAG** | A contract prescribes what a diagnostic must distinguish, never which statistic does it | `[READ CEQ_V16_CONTRACT.md:63-66]`; M-18 `[READ MISTAKES.md:2103]` |
| **L-FLOOR** | Every capability number ships beside its information floor (hop floor, Hankel ceiling, Fano, rate–distortion) so "how good" reads as distance-to-floor | `[READ CEQ_V20_R15_CONTRACT.md:60-63]`; consequence: `floor₁` is not one, C15 `[READ V20_R15_JOURNAL.md:51]` |
| **L-CERT** | Every sparsity mask ships its certificate, F0 exact or F1 with δ printed; a mask without one is refused in the arena | `[READ CEQ_V20_R15_CONTRACT.md:64-67]` |
| **L-EQ** | `[V]` (page exists, intro matches) is inadmissible for any load-bearing statement; `[V-eq]` requires the theorem with hypotheses and one numeric instance run | `[READ CEQ_V15_CONTRACT.md:51-55]` |
| **L-LEAN** | The arm may not be trained before its identity theorems are green; binds become theorems, tests confirm theorems | `[READ CEQ_V15_CONTRACT.md:57-58; CEQ_V16_CONTRACT.md:50-51]` |
| **L-FIRST** | Listed as standing; usage: predictions and counters are filed before each run, "the correct order" | `[READ CEQ_V16_CONTRACT.md:49, :191; CEQ_V15_1_DELTA.md:115]` — **definitional text not located this session** |
| **L-G2** | Journals never move and are never deleted; superseded cells stay with a supersede marker | `[READ V17K_RULINGS.md:62-64; attic/ATTIC_MANIFEST.md:16; AUDIT.md:44]` |
| **FOUND-not-NAMED** | A wing named rather than found ⇒ struck; FOUND iff `results/` holds a record with the wing's `kind` | `[READ CEQ_V20_R15_CONTRACT.md:49, :266; V20_R15_WING_MANIFEST.md:59-63]` |
| **R-SKY** | The native (scan) skyline is read beside every bed; `Δ_sky` column mandatory; "beats softmax" is not licensed where softmax is a fellow approximator | `[READ CEQ_V16_CONTRACT.md:209]` |
| **Ruling 1** | Determinism: CUDA with `warn_only=True`; bitwise for replay and every deciding forward cell; training compared to a measured noise floor; CPU-strict refused | `[READ V17K_RULINGS.md:39-45]` |
| **Ruling 2 / 2a** | The corner: `β` learnable, init 1, per-instance `β` logged; the card sentence splits; no sentence transfers across corners without a bind at the corner it describes; "pinned" is per-parameter, three branches | `[READ V17K_RULINGS.md:47-54, :275-325]` |
| **Ruling 3** | Matched params: `0.032 %` residual is matched; exact counts in every table header; do not re-architect to close it | `[READ V17K_RULINGS.md:56-59]` |
| **Ruling 4** | CPU cells re-taken on the certified 4060, never widened; CLOSED, 24 cells, 2.79 GPU-min | `[READ V17K_RULINGS.md:61-64, :24]` |
| **Ruling 5** | `scripts/v15_r1.py` is the named Q1/Q2 instrument, hashed into the identity manifest; CLOSED | `[READ V17K_RULINGS.md:66-69, :25]` |
| **Ruling 6** | Envelope text 6a–6f: stale = two polls, session time is Kaggle's, tier precedence, per-root lineage, OOM keyed per shape, deciding-cell list frozen at launch | `[READ V17K_RULINGS.md:71-81]` |
| **Ruling 7** | BED-M carried as generator + seed + hash, regenerated in-notebook; the hash assertion is the provenance | `[READ V17K_RULINGS.md:83-86]` |
| **Ruling 8** | Q2 (R2 reproduction, `t* = 8`, `n = 16,384`) DROPPED — `10.578 GiB` against the 4060's `7.996 GiB` | `[READ V17K_RULINGS.md:328-339]` |
| **Ruling 9** | R1′ cells fall under the measured-floor clause; no equality statement without its floor printed beside it | `[READ V17K_RULINGS.md:354-379]` |
| **Ruling 10′** | "Pinned" by likelihood ratio: `Λ = 2[LL(β_final) − LL(β ≡ 1)]` on held-out; `≤ 3.841` PINNED, `> ln n` MOVED, between prints the interval verdict; the minimum detectable departure printed; supersedes Ruling 10's ulp criterion | `[READ V17K_RULINGS.md:389-436]`; annex M15 `[READ CEQ_V20_R15_CONTRACT.md:254-256]` |
| **Rulings 11–12** | Cited as binding (*"Rulings 1–12"*) | `[READ CEQ_V20_R15_CONTRACT.md:59]` — **no Ruling 11 or 12 text exists in the tree** `[RUN: grep -rniE "ruling 1[12]\b" *.md → 0 hits outside the two citing lines]` |
| **RULING J-14 / J-14b** | An F4 is NOT-PUT with an admission condition; a hybrid verdict is not a verdict | `[READ V20_R15_THEORY_TABLE.md:291-302, :358-362]` |
| **RULING J-17d / J-17e** | M14 is F4 and that is the only live grade; every citation repair is made in place so pointers do not move | `[READ V20_R15_LEAP_LEDGER.md:353-357; V20_R15_IT17_JUPITER.md:268]` |
| **The R15 kills** | named-not-found ⇒ struck; theory exit without F-grade+gap ⇒ not filed; mask without certificate ⇒ refused; capability number without floor ⇒ not a number; second leap call ⇒ breach; leap output acted on before its instance runs ⇒ struck; absorbed component without licensing theorem ⇒ dropped; deletion without manifest commit first ⇒ breach | `[READ CEQ_V20_R15_CONTRACT.md:265-273]` |

**Two laws whose text is thinner than their citation.** `L-GRADE (F0–F4 + HOW-BAD gap)` is cited once, on the standing side, and no rubric exists anywhere in the repository `[READ V20_R15_THEORY_TABLE.md:22-24]`; Rulings 11–12 are in the same condition (row above). The paper's programme must not cite either as if it had text; it states the rubric it uses and marks it a reconstruction, which is what the table did and was ruled BOUND for.

---

## S.5 THE OPEN RULINGS AND THE DEBTS CARRIED

**Five author rulings open at it.35** `[READ V20_R15_JOURNAL.md:8225-8226, heading "### WHAT it.36 OWES", item 8]`:

| ruling | what it decides | source |
|---|---|---|
| `⟨CLAUSE_1_TAIL⟩` | one- vs two-sided CP-lower on clause (1): one-sided, W3 clears alone and clause (2)'s void is latent; two-sided, no entrant clears, the order falls through to a void clause and Phase C returns no winner at any price | `[READ V20_R15_THEORY_TABLE.md:256-268]` |
| `⟨L_GRADE_RUBRIC⟩` | the F0–F4 text that does not exist | `[READ V20_R15_THEORY_TABLE.md:22-29]` |
| `⟨KAGGLE_ATTACH⟩` | the chess witness is stalled on a 34.4 GB Kaggle attach; no chess bed registered | `[READ V20_R15_JOURNAL.md:2418 heading; V20_R15_THEORY_TABLE.md:240]` |
| `⟨F4_GATE⟩` | 3 of 12 cells invisible to the gate; one of them carries a refutation inside a token meaning *unattempted* | `[READ V20_R15_IT35_MARS.md:149-151]` |
| `⟨TERMINAL_VS_NOT_PUT⟩` | two words for one predicate and no word for a bound an experiment removes | `[READ V20_R15_IT35_MARS.md:127-133]` |

The table chose not to pick the tail because *choosing the tail that makes the round score is the catalogued failure of selecting a threshold after seeing the data it judges* (M-2, `MISTAKES.md:451`) `[READ V20_R15_THEORY_TABLE.md:265-268]`. The paper's programme fixes every threshold before its data for the same reason.

**Debts inherited from v16/v17-K, still on the ledger** `[READ workdonenewseal.md:508-515, §11; CEQ_V16_CONTRACT.md:245-252]`:

| debt | one line |
|---|---|
| **D-APPROX** | Lean #17 approximation bound. Lean #12 is an exact-identity result; the power-law bed is not scan-blind at `R² = 0.604` and that does not contradict it. *"Every 'X cannot represent Y' claim needs a bound before it reads as 'X cannot fit Y'"* — the largest unclosed gap in the round's own logic `[READ workdonenewseal.md:510]` |
| **D-R3** | The two inverse registrations: PART III and PART IV of the v16 contract predict opposite outcomes for the same bed (M-20, `MISTAKES.md:1723`); the author picks one and the cell cannot run first `[READ workdonenewseal.md:511; CEQ_V16_CONTRACT.md:249]` |
| **R2's `n`** | `32768` does not fit; `16,384` is the last power of two `[READ workdonenewseal.md:512]`; Ruling 8 then dropped the reproduction at `16,384` too (`10.578 GiB` vs `7.996 GiB`) and flagged that dropping the reproduction does not supply the reading `[READ V17K_RULINGS.md:330-348]` |
| **`[0,1]` vs `[0,1)`** | Three nodes measured the closed endpoint as the defect; a fourth refuted that — *the endpoint is not the failure, `log` of it is*; `[0,1)` would cost reachability of `a = ±1`, two-thirds of BED-M's support. Filed two-sided for the author `[READ workdonenewseal.md:513]` |
| **the scan skyline** | Refused in R11 as a new construction; legal now as the native control `[READ workdonenewseal.md:514]` |
| **HF package** | Scheduled, unbuilt; no trained checkpoint ships `[READ workdonenewseal.md:515; MODEL_CARD.md:73-75]` |

**Harness debts the new programme must not inherit silently** (all `[RUN]` this session against `scripts/v15_r1.py` at HEAD, corroborated by the record):

- `S, D = 64, 24` is a module constant at `:137` `[RUN: sed -n 137p]`; `add_argument` appears **9** times `[RUN: grep -c]`; the strings `seq_len`/`seq-len` do not occur `[RUN: grep → no hit]`. So `s = 64` on 40 of 40 banked cells and the exponent of any cost law in `S` has `n = 1` `[READ V20_R15_THEORY_TABLE.md:95-99]`.
- `torch.cuda.synchronize` does not occur in the file `[RUN: grep → no hit]`; the per-cell timer is `t0 = time.time()` at `:249` and `secs = time.time() − t0` at `:267` `[RUN: sed]` — un-synchronised host wall clock on CUDA `[READ V20_R15_THEORY_TABLE.md:99-101]`.
- The strongest correlate of `secs` is **run order**, `ρ = +0.7029, p = 0.0024`, above the gate correlation `+0.5197` that it would have to be separated from; every GPU-second and cost ratio quoted from it.3 onward is void (C17) `[READ V20_R15_JOURNAL.md:53]`. The priced repair — one `--seq-len` flag, one substitution, two `synchronize()` calls, `S ∈ {32,64,128}` — is `206–537 GPU-s`, **band only, no point**; the run-order confound *survives* it and the fifth edit (randomised or blocked execution order) has never been priced by any office `[READ V20_R15_THEORY_TABLE.md:104-109, :184]`.
- `lambda_hat = float(lg.mean())` at `:383` averages `log m` over every position, so one `m_k == 0` sends it to `−inf` and it carries one bit; `lambda_hat_live` at `:384` is the magnitude column `[RUN: sed; READ V20_R15_THEORY_TABLE.md:166]`.
- `brute_force_path_sums` at `ceq/arm_pl.py:304` is `O(2^S)` and unguarded `[RUN: sed; READ V20_R15_IT35_MARS.md:120-125]`.

**Bookkeeping debts the record names against itself**, carried so the paper does not repeat them: the C20 numbering collision and the seven index rows `C13–C19` that never had body corrections (LEGACY, membership frozen) `[READ V20_R15_JOURNAL.md:78-83, :76]`; every whole-suite `N failed / M passed` published as evidence of an edit's safety is the wrong instrument — the reachable radius is `RADIUS ∪ TABLE_RADIUS = 28` files `[READ V20_R15_JOURNAL.md:7896-7919, heading "### JUPITER — THE CENSUS ANSWERS WITH A DIFF"]`; a population is `(rule, date)` and a bare integer is neither, the same defect as a `[RUN]` marker without its command and a digest without its recipe `[READ V20_R15_JOURNAL.md:7950-7966, heading "### THE THREE POPULATIONS RECONCILE EXACTLY"]`.

---

## S.6 THE KAGGLE AND HUGGINGFACE STATE

**Nothing has launched.** *"Nobody who wrote these files ran the `kaggle` CLI, pushed a kernel, or touched `~/.kaggle`; launch is the author's own call, after Gate 0 is green"* `[READ kaggle/README.md:9-11]`. Gate 0 closes only at open-rulings = 0, and G0.10 carries a circularity the register does not resolve: rows 1 and 2 are blocked on numbers only the Kaggle run produces `[READ V17K_RULINGS.md:3-17, :21-22]`. The HF package ships *on explicit say-so* at the end of the launch order `[READ V17K_RULINGS.md:92-94; CEQ_V16_CONTRACT.md:295]`, and the record has already refused a relayed instruction as that consent: *"a coordinator message is not that say-so"* `[READ CHECKLIST.md:1275]`. The author's own standing instruction (memory: *Kaggle launch requires yes*) is the same rule from the other side.

**What exists.** `ceq/hf/` holds `configuration_ceq.py`, `modeling_ceq.py`, `train.py`, `smoke.py` `[RUN: ls]`. There is no trained checkpoint — `⟨SLOT Q3_CHECKPOINT⟩ NOT MEASURED` `[READ MODEL_CARD.md:73-75]`; parameter counts are not equal, arm `25,736,232` against control `25,728,000`, `+8,232 = +0.03200 %`, the arm carrying more `[READ MODEL_CARD.md:76-79]`; the card's own header states the shipped operator does not work — sign-flip slope `−1.298` against a bar of `−0.3`, zero flips at `s ≥ 128` `[READ MODEL_CARD.md:18-24]`; `README.md:301` reads *"No trained checkpoint ships"* `[RUN: grep]`.

**The pins.** `results/k_data_manifest.json` `[RUN: json.load]`: `bed_m`, `bed_k`, `bed_1` are `kind: generator, status: PINNED`; `enwik8` is `kind: kaggle_attach_sliced, PINNED`; `lichess_chess_games`, `lichess_chess_evaluations`, `tinystories_cdla` are `UNPINNED_AWAITING_KAGGLE`. The bed digests, cited from the manifest rather than retyped `[READ kaggle/README.md:86-97]`:

| bed | generator | kwargs | sha256 |
|---|---|---|---|
| BED-M | `ceq.corpus.build` | `n_train=384, n_test=128, seed=0` | `2f282a5d0e9ac412b9644e19969590c0f855e5ccb3db73436364bcb925f7d24d` |
| BED-K | `ceq.beds.bed_k.build_delay` | `n=500, d=4, seed=7` | `15de94b47ba07e8b2d118deac26306689a7c55d0b243699e7d04e0881a5c56e4` |
| BED-1 | `ceq.beds.bed_1.build` | `T=0.25, jitter=0.05, seed=11` | `f73ca0e60712dec446165131c07576d03173653b14cfdeec9ce1f581bb6881d2` |

`enwik8` is the first `100,000,000` bytes of `jamesmcguigan/hutter-prize`'s `enwik9` (CC-BY-SA-3.0), hashing to `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`, byte-identical to the existing pin; the 90/5/5 split lands at `90,042,869` / `95,000,818` `[READ kaggle/README.md:31-43]`. The three unpinned sources make the notebook's hash cell **print** and then **raise** `kdata.MissingPin`, never a silent skip `[READ kaggle/README.md:99-105]`. The notebook's code pin is `PINNED_SHA = "eea1aa25245c8a98f891a7f7e6bdae6aa26b4885"` `[RUN: grep on kaggle/ceq_v17k.ipynb]` — the code snapshot the run executes, which HEAD's own message says is allowed to differ from the branch tip `[RUN: git show 207e7b9]`. The certified-device laws `scripts/k_cert.py` refits: `arm_smprime` throughput `s/step = exp(−10.0187)·n^1.0026`, `R² = 1.000000`; the fit refused `n = 16384` and `32768`, where the allocator pages over PCIe `[READ README.md:288-297]`.

**What this means for the paper's programme.** Every Kaggle line in the plan is priced but not launchable by any agent; the author's explicit yes is a node on the critical path, not a formality. The chess witness — the one bed with a categorical state space — is behind an attach nobody has made and is not registered (`BED_SPECS` returns `['bed_m', 'bed_k', 'bed_1']`) `[READ V20_R15_THEORY_TABLE.md:224]`.

---

## S.7 THE RECORD'S OWN LEAP-GATE PROMPT, AND THE AUTHOR'S SUPERSESSION

**The prompt, as the contract fixes it** `[READ CEQ_V20_R15_CONTRACT.md:142-151]`:

> it.35 EVALUATION-FOR-LEAP: JUPITER + MARS grade every F1/F2/F3 failure LEAPABLE (name the FIELD likely to hold the missing theorem, not the theorem) or TERMINAL (a bound; no theorem removes a bound). ONE call to the leap model with the fixed prompt (v19-Q, unchanged: repair graded failures with named theorems + runnable instances; compose ONE candidate from the winner's primitive plus licensed components; predictions AND counters; every untested claim with its cheapest killer; no new primitive that is not a repair). Output filed [LEAP-UNTESTED]; its instances RUN before anything else.

Its consumer-side kills: *a second leap call ⇒ breach; leap output acted on before its instance runs ⇒ struck; an absorbed component without a licensing theorem ⇒ dropped* `[READ CEQ_V20_R15_CONTRACT.md:268-272]`. The theory table was written as the leap's primary input, *"gets ONE call, and cannot"* ask a follow-up `[READ V20_R15_THEORY_TABLE.md:8-12]`. No separate `v19-Q` document exists in the tree; the parenthetical above is the whole fixed text `[RUN: ls | grep -i v19 → none; grep v19-Q → only the two contract lines]`.

**What the it.35 gate handed forward.** The call was not spent `[READ V20_R15_JOURNAL.md:8234-8238]`. Had it been, the record's own grading says the leap would have received: seven distinct failures on eight cells, two of them with verdicts the two offices could not agree on under a rule that was published one way and applied another; a header miscounting its body; three cells it could not read, one carrying a refutation; a primary input drifted off its census with one citation no instrument can parse `[READ V20_R15_JOURNAL.md:8226-8238]`.

**The supersession, stated where it matters.** On 2026-09-03 the author licensed a new primitive — *"rethink the shape"* — and instructed *"do not write any code only docs"* `[READ BRIEF.md:11-30, :40-42]`. The contract's clause *no new primitive that is not a repair* is therefore superseded by the author for this document, and the paper says so here rather than by editing the contract, which is verbatim-of-record and never edited `[READ CEQ_V20_R15_CONTRACT.md:3-6]`. What the supersession does **not** lift, because the author did not lift it: the leap's remaining clauses — named theorems with runnable instances, predictions **and** counters (L-SIGN), every untested claim with its cheapest killer, instances RUN before anything is acted on — and the standing laws of S.4. What it changes in kind: the iteration count of the new programme is derived from its DAG's critical path (D-3), not inherited from the 45 the R15 schedule named, and its Phase-E-shaped work (candidate built with identity binds, manifests, certificates, param-matched, full arena, verdict by the same clauses, purge, prognosis) `[READ CEQ_V20_R15_CONTRACT.md:153-169]` is the template the plan inherits for the shape, with the beds and labels replaced.

**Mechanisms this section is designed against**, named per the brief's rule 6: P-3 stale claim never retracted (`MISTAKES.md:316`) — the stale phase table and the unspent call are stated, not inherited; P-6 line drift (`:364`) — journal cites carry headings; V-7 empty search read as absence (`:117`) — every `no hit` above names the command and the calibrating positive (`add_argument` = 9 on the same file); M-2 refitted threshold (`:451`) — the tail ruling is left to the author; D-1 racing a proven optimum (`:677`) and D-2 the arm's own resolvent as oracle (`:710`) — the Q6 pair and the BED-M label class are carried as the reason the arena could not return a winner, which is the paper's starting premise; V-20 cap in prose (`:987`) — the R15 count came from a literal flag read back; D-7 prediction without counter (`:2037`) — no prediction is made in this section.

---

## S.8 LIMITS (collected once)

This section reads the record at HEAD `207e7b9`; the Lean build was not re-run and its green at `V20_R15_THEORY_TABLE.md:397-400` is a cache hit `[READ]`. Every arena and cost figure is MERCURY's of the round, on an un-synchronised timer, and is reproduced as the record states it, not re-measured. Line numbers in `V20_R15_JOURNAL.md` are true at HEAD and will drift on the next head-block append; headings are given beside each. `L-FIRST`'s definitional text and Rulings 11–12 were not located; they are listed as cited by the record, not as read. The `MISTAKES.md` mechanism count is **66** `###` headings including `V-14a` `[RUN: grep -cE '^### (V|P|M|D)-[0-9]+']`, against the brief's *65*; the difference is the lettered sub-entry at `MISTAKES.md:770`. JUPITER's it.35 file says *35 of 35* citations land `[READ V20_R15_IT35_JUPITER.md:204]` while the journal's digest of it says *41/41* `[READ V20_R15_JOURNAL.md:8112, :8125-8126]`; the discrepancy is recorded, not resolved. No external source was fetched; the `[V]` tags on Wilks and BIC in Ruling 10′ are the record's, inherited. No number in this section was produced by new code; no code, git write, Kaggle contact or hardware run was made.
