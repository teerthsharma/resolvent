# DONE — Round 2, under CONTRACT.md and LOOP_PROMPT.md

Started 2026-08-25. Round 1's 5,922 lines are preserved verbatim in
`DONE_ARCHIVE_ROUND1.md` and are still binding evidence — nothing here
overrides them, and nothing recorded there may be re-derived.

This file is **append-only**. A recorded result is never deleted or edited.

---

## CARRIED FORWARD — the facts Round 2 starts from

Everything below is journalled, replay-verified, and must **never be
re-measured**. Re-deriving a recorded number is a wasted iteration.

### The surviving measurements (PROTOCOL: SCALING, CP intervals, calibration bit-identical)

**M2 claim arm `pivot_signed`, wrt=v, 16384 draws at the tail:**

| s | 8 | 32 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| rate | 0.024658 | 0.028564 | 0.031006 | 0.028809 | 0.029663 | 0.029907 |
| k/n | 101/4096 | 117/4096 | 127/4096 | 118/4096 | 486/16384 | 490/16384 |

Slope **+0.0270**, R² 0.5222, across a **256× context growth**.

**Control `dense_signed__at_pivots`** — same c, same operator, only hop-2
differs (`A@A` instead of `A[:,P]A[P,:]`):

| s | 8 | 32 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| rate | 0.024658 | 0.021240 | 0.010254 | 0.003174 | 0.000488 | 0/2048 so far |

Slope **−0.746**. Separation 1.00× / 1.3× / 3.0× / 9.1× / **61×** / ≥20.5×.

**S2 ablation on `wrt="x"`** — the only channel where the comparison is
non-vacuous, since on `wrt=v` a non-negative operator is pinned at exactly 0 by
theorem (`I + A + A²` non-negative entrywise):

| s | 8 | 32 | 128 | 512 | slope |
|---|---|---|---|---|---|
| `pivot_unsigned__x` | **0.102539** | 0.014160 | 0.001221 | **0.000000** | **−1.598** (R² 0.9962) |
| `pivot_signed__x` | 0.026367 | 0.034912 | 0.024902 | 0.031250 | −0.021 |

Separation at s=512 **≥ 42.7×** (CP upper on 0/4096 = 0.000731).

### The reframing that Round 2 inherits

**At s=8 the UNSIGNED arm reads 0.1025 against signed's 0.0264** — softmax
flips a sign FOUR TIMES MORE OFTEN at short range, then dies. The signed
operator is **not better at s=8; it is better at HOLDING.** "Softmax cannot do
this" is FALSE and this project's own measurement disproves it. The defensible
claim is **PERSISTENCE IN CONTEXT** plus depth/parameter efficiency, never
short-range capability.

Reinforcing this: **one GELU between two softmax layers restores the sign
flip** (`test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`).
The semiring theorem covers non-negative operators with LINEAR value paths, not
real transformers.

### M2 is a reported defect, not a pass

M2's kill clause 2 (`c ∉ P is ALSO flat`) is **unevaluable by construction** —
`hop2[i,j] = Σ_{p∈P} A[i,p]A[p,j]` has no term with index c when c ∉ P. The
verdict code mapped its NaN slope to "control decays as required": a **FALSE
GREEN**, instrument #15. Item text frozen under `LOCK M2 efadc390c93f`. Not to
be edited, not to be silently fixed.

### Sixteen instruments, fifteen already caught

Internally consistent, externally wrong: parity vs the repo's own
`stock_attention` (89,400.180 vs 1.667) · a `sorry` detector firing on the
sentence "No `sorry` anywhere" · an eviction test asserting bitwise equality
where it is provably false · a corpus split by literal value · an uncalibrated
probe reading 0.0000 everywhere · multizoom with `c = s//2` in the schedule by
construction · a random-schedule control redrawing between its own arms · the
dfloor probe keeping `min(4, nblk)` of `nblk` blocks · the ParaFormer arm
running softmax under another name · an LO probe at fixed `i=7,j=1,c=4` with ~5
intermediates at every s · `run_calib.py` printing targets as strings and
always exiting 0 · M2 dying silently at 588 s with `| tee` reporting exit 0 ·
**M2's own control being zero by construction.**

**Assume the next one is yours.**

### Standing facts

- Whole-suite `pytest tests/ -q` has **never completed** — nine attempts, three
  agents, one 1-hour monitor. `--collect-only` gives 829. **No total pass/fail
  count exists. Never quote one.**
- The one capability comparison at matched parameters went **against** this
  operator: COGS-gen softmax **0.0293** (15/512) vs sgate **0.0000** (0/512),
  one-sided Fisher **p = 2.7502788939e-05**, and behind **in-distribution** too
  (0.9258 vs 0.7734). ARC-AGI never scored. No Turing-style eval file exists.
- Nothing has trained above **3.65M** parameters against a **300M** gate.
  "912 A100-hours" is **unsupported**; the real figure is **130–257 A100-h**
  (~$194–385). The gate is blocked by **~15 lines of missing checkpoint/resume**
  in `ceq/hf/train.py::train()`, not by money.
- Lean: **27 theorems**, `lake build CEQ` exit 0, zero `sorry`, no `sorryAx`.
  `CEQ.Refcount` is the live provenance candidate, gated by
  `tests/chase/test_lean_refcount_binding.py`.
- Prior art, every pair in the triple occupied: Star-Transformer 1902.09113
  (relay hop-2, **unsigned**, 2019) · Perceiver 2103.03206 (pivots+multi-hop) ·
  NSA 2502.11089 / MoBA (content-selected, single-hop). On the operator itself:
  SimA 2206.08898 · SDA 2606.04833 (**is** the sgate matrix) · DeltaNet
  2406.06484 · ParaFormer 2512.14619 · SignGT 2310.11025 · Cog 2411.07176 ·
  RetNet 2307.08621.
- The **"84× vs ParaFormer"** number is DEAD — it was softmax under another
  name. Never cite it.

### DEAD — do not revive without new evidence of the stated kind

DEQ / Hopfield settling (Howard PI beat fixed-point search 7.9–21.4×) ·
multigrid / spectral decimation / mixed curvature (R5 deleted) · sheaf-Hodge
contradiction energy (averaging won by 0.0793 AUROC) · Nash / QRE / ESS / MFG
(W7: 1/5 seeds) · Lyapunov-spectrum shaping (vacuous on spectrum {0}; migrated
into A2, numerical range) · IFS-collage compression · free-probability hop
scaling · Lorentzian polynomials · ΔFloor-by-eviction as a selector (both
pre-registered kills fired, CI at s=1024 entirely below chance) ·
refcount-priced schedule as a selector (killed by its own Lean proof — refcount
is constant within a sequence, so the score carries zero bits).

**Max-plus is dead**: the star measured identical to the APPNP of its own
greedy policy at 9.95e-14, gradient at 1.65e-08. Route R3 must state what makes
the symmetrized/valuation form a different object before it is built.

---

## THE OPEN CONTRADICTION — Round 2's first job

`CONTRACT.md` states naive flatness of a flip probability at global reach is
**impossible** (Littlewood–Offord lower bounds). This project **measured flat**:
slope **+0.0270** across a 256× growth, 16,384 draws at the tail,
replay-verified 11×, calibration bit-identical.

Both cannot be true. Exactly one holds:

1. the flatness measurement is **instrument #16**; or
2. **pivot routing is already an escape** — the background is `k` terms, not
   `s`, which is R1's hypothesis-break ("generic token") achieved
   *structurally* rather than by a decoder.

If (2), R1 is partly done and the cost order changes. One derivation plus one
probe settles it, and it is worth more than any single route.

---

## CHEAP WINS AVAILABLE NOW — ranked by value per hour

1. **~15 lines of checkpoint/resume** in `ceq/hf/train.py::train()` (no
   optimizer state, no step counter, no load path). Unblocks every long
   training run including the 300M gate. Highest value-to-cost on the board.
2. **The free 25.7M T4 Colab run** — the notebook's own default shape is
   25,707,520 params, **7.0× above the 3.65M ceiling this project has ever
   trained**, and it fits a free T4 (2.45 GiB against 14.5) in one 12-hour
   session. As shipped the notebook runs 1.6% of a Chinchilla budget. Costs
   nothing but wall-clock.
3. **M5 is nearly GREEN already** — 27 theorems, `lake build CEQ` exit 0, zero
   `sorry`, no `sorryAx`. What is missing is the grep-bind from theorem
   hypothesis to the shipped tensor (`.tril(-1)`), and precedent for that bind
   already exists in the repo.
4. **M1 is probably GREEN from existing data** — the signed path sum reaches a
   negative influence Jacobian entry and softmax reads exactly 0.000000e+00.
   Needs the frozen-zero-gradient-cell check (the −ρ at (1,0) trap) and it is
   done.
5. **Resolve the open contradiction above** — one derivation, one probe.

---

## ROUND 2 LOG

<!-- Iterations append below this line. Never delete, never edit. -->

---

# CAMERON — R5 PRE-REGISTRATION, WRITTEN BEFORE ANY R5 NUMBER EXISTS

**Timestamp discipline:** this block is appended while the arm-building nurse is
still running and has reported nothing. No R5 number has been seen by anyone.
That is the entire point of writing it here — `M2_PREREGISTERED_READING.md`
exists because several of this project's fifteen instrument failures were
*interpretations* that hardened after the number arrived.

## The route

**R5 — PUT A READOUT ON IT.** Stop measuring the operator; measure what the
operator lets a model DO. Train the M3 negation-scope arms on this CPU.

R1-R4 each produce a number *about the operator*. `CHECKLIST.md`'s preamble
already rules on that class: "Statistics are not capabilities." R5 is the only
route on the table whose output is the thing the checklist calls M3.

## Why this is not a fifth expense

Both expensive halves are already built and calibrated:

- `scale/negation_scope.py` — the M3 task, with an ABSOLUTE bar that has been
  seen to fire: `results/iter04_m3_bar_calibration.txt` reads
  `predict_the_mean 1.000000 / payload_only 1.414204 / oracle 0.000000`.
  It has **no arms**. `bootstrap_ci` has zero callers
  (`house-events.jsonl:1926`, Chase RED).
- `scale/pivot_probe.build_arm` — the operator builder that is the single
  source of truth for every M2 number published.

The delta between them is a readout head and an Adam loop.

## THE KILL — frozen here, and every clause checked for STRUCTURAL FIRE

M2 clause 2 could never fire (`c ∉ P` is zero by construction) and the verdict
code turned its NaN slope into a false GREEN. Each clause below carries the
argument for why it can reach both sides of its threshold.

**K1. `pivot_signed` held-out NRMSE ≥ 1.0 at d = 256 → RED.**
FIRES. NRMSE = RMSE/std(y) is ≥ 0 and exactly 1.0 for the mean predictor, so
both sides are reachable. The instrument must DEMONSTRATE this by reading an
untrained (0-step) arm at ≥ 1.0 before any trained arm is credited.

**K2. `softmax` beats `pivot_signed` on held-out NRMSE with non-overlapping
bootstrap CIs → RED.**
FIRES, and there is precedent that it fires: the one capability comparison ever
run at matched parameters went AGAINST this operator — COGS-gen softmax 0.0293
(15/512) vs sgate 0.0000 (0/512), one-sided Fisher p = 2.7502788939e-05, and
behind in-distribution too (0.9258 vs 0.7734).

**K3. `pivot_unsigned` within the CI of `pivot_signed` → G4 fires, the
contribution is routing, and the claim sentence is rewritten before work
continues.**
FIRES **ONLY ON AN x-PATH READOUT**, and this is the clause that inherits
instrument #15's defect if it is coded carelessly. On the v-path,
`out = v + Av + A²v` gives influence Jacobian `I + A + A²`, non-negative
entrywise for non-negative `A` — so an unsigned arm reads exactly 0 by THEOREM,
not by measurement. That is precisely what `results/s2.jsonl` shows:
`pivot_unsigned__at_pivots` = 0/4096 at s = 8, 32, 128 and 512.
**A K3 evaluated on a v-path readout is not a test, it is the semiring theorem
restated.** On the x-path the unsigned arm is demonstrably non-zero — it reads
0.1025390625 at s = 8, which is 3.889x the signed arm — so K3 is evaluable
there and only there.

**K4. Any arm whose NRMSE is NaN or inf is RED, never GREEN.**
FIRES only if coded first: `float('nan') >= 1.0` is False in Python, so a NaN
sails through a kill written as `if nrmse >= 1.0: FAIL`. `nrmse()` returns NaN
whenever `std(y) == 0`, which its own `sd == 0.0` branch makes reachable.

## WHAT R5 SACRIFICES — written before the numbers, so it cannot be trimmed

1. ~10⁴ parameters of synthetic regression. It says **nothing** about "working
   as an LLM", and nothing about the 300M gate.
2. It inherits M2's geometry. If the random/learned projection family is the
   confound, R5 is confounded the same way.
3. It settles **no** novelty question. Star-Transformer 1902.09113 still owns
   pivots + multi-hop with an unsigned operator, from 2019.
4. One task, one metric. NRMSE on one synthetic regression is not "understands
   consequences", and R5 must never be written up as if it were.

## THE OUTCOME NOBODY HAS COSTED, AND THE REASON R5 IS WORTH RUNNING

`pivot_signed` flips sign on **3.0%** of intervened draws. Flat, yes — but flat
at three percent. **No one has asked what rate a downstream task needs.** If
sign information is present on 3% of (i, j, c) triples and a task needs it on
every example, both arms fail and the flat statistic everyone is defending is a
flat *failure*.

R1-R4 cannot discover that. R5 is the only route that can, and it is the reason
a route that merely adds another statistic is not a substitute.


### ITERATION 1 — 2026-08-25 — M2' appended verbatim. M2 superseded, not passed.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

ACTION (one): appended **M2'** verbatim to `CHECKLIST.md` as a new MANDATORY
item, with its four routes and the binding cost order 2 -> 4 -> 1 -> 3.

**GOVERNANCE CALL, stated rather than fudged.** M2's kill had three clauses:
  * clause 1 (`slope(c in P) < -0.3`) **did NOT fire** — measured **+0.0270**;
  * clause 3 (G2, published number moves) did not fire — calibration held
    bit-identical across 31 iterations and three concurrent fellows;
  * **clause 2 is UNEVALUABLE BY CONSTRUCTION** (instrument #15).

So M2 is **neither GREEN nor RED-by-kill — it is DEFECTIVE**, and a claim whose
safeguard cannot be evaluated has not survived its safeguard. Marked
**SUPERSEDED** with the defect report attached permanently, because the contract
names M2' as the mandatory item in its place. **The work-stopping clause does
not fire**, since M2 is no longer a live mandatory item — recorded explicitly so
nobody later reads "superseded" as a quiet pass.

**LOCK VERIFIED [RUN].** Appending M2' moved the slice boundary used to hash M2's
text, and the naive check reported `1e56334a6b9c != efadc390c93f`. Checked
against the archived copy instead of the moving endpoint:

    archived text hash : efadc390c93f
    live slice hash    : efadc390c93f
    archived == live   : True

**M2's frozen text is byte-identical.** Worth recording that the first hash
mismatch of this project was an artifact of the checker, not of the file — which
is the sixteenth time an instrument here has been wrong before the thing it
measures was.

CHECKLIST: M2 -> SUPERSEDED; **M2' added, UNTESTED**. No other status changed.

### ITERATION 2 — 2026-08-25 — THE CONTRADICTION IS RESOLVED. Pivot routing IS an LO escape.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

ACTION (one): settled the contradiction between CONTRACT.md's impossibility
claim and this project's measured flatness. One derivation, one probe.

**THE DERIVATION.** Littlewood-Offord's lower bound applies to a background
that sums over ~s terms: with per-term scale sigma, the sum has spread
sigma*sqrt(s) while one term is sigma, so the small-ball probability
P(|background| < |term_c|) ~ 1/sqrt(s) and MUST decay. **Pivot routing makes
the background sum over exactly the pivots, independent of s**, so the ratio is
1/sqrt(k) and does not move. The impossibility's hypothesis is "the background
is a sum over ~s terms". Routing breaks that hypothesis STRUCTURALLY.

**[RUN] THE PROBE — count the hop-2 background terms.** PROTOCOL: SCALING.

    s        dense nnz   part.ratio      pivot nnz   part.ratio
       8         4           2.16            4          2.16
      32        22          11.91            6          4.52
     128        94          37.90            6          4.10
     512       382         150.28            6          2.50
    1024       766         305.94            7          4.38
    2048      1534         637.48            5          2.63

    dense nnz     ~ s^**+1.062**   (LO's hypothesis HOLDS: ~s terms)
    pivot nnz     ~ s^**+0.048**   (FLAT at 5-7 terms, never grows)
    dense part.r  ~ s^+1.001
    pivot part.r  ~ s^+0.015

**THE QUANTITATIVE CHECK, and it is the point.** Feed those term counts back
through LO's 1/sqrt(terms):

    predicted dense  s^-0.531      MEASURED  **-0.746**
    predicted pivot  s^-0.024      MEASURED  **+0.027**

The pivot prediction and the pivot measurement agree **to within 0.05**. The
dense prediction has the right sign and order; the residual (-0.746 vs -0.531)
is the share term, which LO's small-ball factor does not contain.

**VERDICT: option (2). The flatness is NOT instrument #16.** It is exactly what
Littlewood-Offord predicts once the background terms are counted correctly.
There was never a contradiction — the contract's impossibility is a statement
about DENSE aggregation, and this operator is not dense at hop 2.

**CONSEQUENCES, and they change the plan:**

1. **R1's hypothesis-break ("generic token") is ALREADY ACHIEVED, structurally,
   with no group-testing decoder.** R1 as written asks for a d-disjunct decoder
   to recover k causal tokens so the post-recovery background has k terms. The
   background ALREADY has k terms, by construction, because
   `hop2[i,j] = sum_{p in P} A[i,p]A[p,j]` ranges over P and nothing else.
   What R1 would add is a *combinatorial guarantee* that P contains the right
   tokens — which is a SELECTION claim, not a background-size claim.
2. **The cost order 2 -> 4 -> 1 -> 3 should be revisited**, since R1 is partly
   done and its remaining half (is P the right set?) is exactly what S2's
   ablation and M3's capability test already measure.
3. **A1 (anti-concentration) is now [V] rather than [U] for the part that
   matters here** — its hook was "M2' exponent accounting", and the exponent
   now accounts: dense terms grow as s^+1.062, pivot terms do not, and the
   two measured slopes follow.

**WHAT THIS DOES NOT DO.** It does not make anything a capability. M3 is
UNTESTED and `CHECKLIST.md`'s preamble is explicit that statistics are not
capabilities. It does not touch prior art — Star-Transformer 1902.09113 routes
hop-2 through a relay UNSIGNED, and the flatness argument above is
sign-agnostic, so it explains routing, not signedness. Signedness is carried by
the S2 ablation (unsigned routed dies at s=512, signed holds), not by this.

CHECKLIST: no status changed. M2' remains UNTESTED.

### ITERATION 3 — 2026-08-25 — R2's reading pre-registered. A DEFECT IN R2 FOUND BEFORE IT RAN.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

ACTION (one): wrote `M2PRIME_PREREGISTERED_READING.md` while `scale/r2_units.py`
is still under construction and **no R2 number exists**. Not editable once the
first real number lands.

**THE TRAP, fixed in advance.** Sign-determinacy is insensitivity to magnitude.
The property M2 measured is a third token FLIPPING a sign. **These pull in
opposite directions** — a fully sign-determined operator cannot flip at all, so
R2 pushed to its limit drives the flip rate to ZERO, which is softmax's number
and the death of the only property this project has.

So the naive reading — "determined fraction high => R2 GREEN" — **is wrong, and
wrong in the direction that feels like winning.** Same shape as the multizoom
`c = s//2` artifact, the dfloor `min(4, nblk)` ceiling, and S2's vacuous
unsigned arm.

**R2 is GREEN only if the determined fraction is high AND the flip rate
survives — both, same run, same table.** Recorded before the run so a
one-sided report cannot be read as a pass.

**THE DEFECT IN R2 AS WRITTEN — the more valuable finding.** R2's kill says
*"determined fraction ~0 in TRAINED blocks"*. **This project has no trained
pivot blocks.** Nothing has trained above 3.65M parameters, the pivot operator
has never been trained at all, and every M2/S2 number on record is on RANDOM
projections. **Read literally, R2's kill cannot be evaluated today** — the same
class of defect as M2's clause 2, which is why M2 is superseded.

Two honest options, recorded so the choice cannot be made silently later:
  1. Run R2 on random projections, state plainly that the "trained" qualifier
     is unmet, and treat it as a SCREENING number that cannot close M2' alone.
  2. Train a small pivot block first — which pulls the ~15 lines of
     checkpoint/resume and the free 25.7M T4 run AHEAD of R2 and changes the
     cost order.
Option 1 is cheaper and probably right first, **but a screening number must not
quietly become a GREEN.**

**Outcomes fixed:** (A) fraction ~0 -> RED, replacement owed. (B) fraction high
AND flip rate collapses -> **RED, and the informative death** — sign-determinacy
and sign-sensitivity genuinely incompatible for this operator class, which is a
real result about the design space; replacement is **path coherence**, the
weaker correct condition (the j->i paths through c SHARE a sign so c's
contribution adds instead of cancelling). (C) both hold -> GREEN, and still only
a statistic; it buys the right to run M3, nothing more. (D) softmax also high ->
VACUOUS, that is the calibration not a comparison.

**Calibration required before belief:** softmax must read **1.0 by
construction** (all-positive pattern is trivially sign-determined) or the
instrument is broken; a random pattern at the same density must read low; and
the kill must be shown able to fire in BOTH directions before any value between
is trusted.

CHECKLIST: no status changed. M2' remains UNTESTED; R2 not yet run.

### ITERATION 4 — 2026-08-25 — R2's "trained blocks" defect RULED ON. Screening can kill, cannot pass.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

ACTION (one): ruled on the defect found in iteration 3 — R2's kill names
**"determined fraction ~0 in TRAINED blocks"** and this project has no trained
pivot blocks.

**THE RULING: option 1, run on random projections, under an ASYMMETRIC rule.**

    A screening run on random projections may produce **RED**.
    It may **NEVER** produce **GREEN**.

The asymmetry is not a convenience, it is the actual inferential content:
  * a determined fraction near **zero** on random projections is strong evidence
    it is also near zero once trained — the sign pattern of a learned block is
    not going to acquire magnitude-independence that the random one lacks, and
    R2 dies cheaply and honestly;
  * a determined fraction near **one** on random projections says **nothing**
    about trained blocks, because R2's kill names them explicitly. Passing a
    kill whose stated condition was never met is exactly how M2's clause 2
    became a FALSE GREEN.

So R2 is a cheap FILTER now and a closeable item only after training. Recorded
so a screening number cannot be promoted later by anyone, including me.

**Consequence for the ship path, and it is a good one.** The only way R2 ever
reaches GREEN is through a trained pivot block — which requires the **~15 lines
of checkpoint/resume in `ceq/hf/train.py::train()`** and the **free 25.7M T4
Colab run** that are already ranked #1 and #2 on the cheap-wins list. **R2's
defect and the terminal deliverable want the same next piece of work.** That is
the strongest argument yet for pulling the training work forward rather than
treating it as Phase 3.

**[READ] CHASE's in-flight `scale/r2_units.py` honors both calibration ends**
(scale/r2_units.py:84-93):
    `softmax` must read 1.000 by construction or the instrument is broken
    `softmax_broken` is the deliberately-wrong pattern that must read low
    `random` is the chance baseline at the same density
It also carries a **path-coherence** measure judged against a same-N
same-magnitude random-sign null (scale/r2_units.py:198-208), which is the exact
replacement `M2PRIME_PREREGISTERED_READING.md` named for outcome B — and it
controls for the fact that N random-sign terms give ~N^(-1/2) for free, so the
null is doing real work rather than decorating.
It contains **no reference to "trained"**, so the defect above is unaddressed
in code and this ruling is what governs it.

CHECKLIST: no status changed. M2' remains UNTESTED. R2 is now a SCREENING item
until a trained block exists.

### ITERATION 5 — 2026-08-25 — HEALTH INSPECTOR PASS. All four checks CLEAN.

Mandatory every 5th iteration per `LOOP_PROMPT.md`. Four checks, all run.

**CHECK 1 — calibration [RUN].** `run_calib.py --self-test` -> exit 0.
The gate rejected a deliberately-wrong target first, then read 4/4
bit-identical. **CLEAN.**

**CHECK 2 — replay a journalled bucket, bitwise [RUN].**
`pivot_signed__in_P/s32` recomputed from a fresh process:

    journal    {'k': 117, 'n': 4096, 'rate': 0.028564453125,
                'sigma': 0.043463709101146546, 'term': 0.006959808408699891}
    recompute  {'rate': 0.028564453125, 'k': 117, 'n': 4096,
                'term': 0.006959808408699891, 'sigma': 0.043463709101146546}

**MATCH**, every field including `sigma` to full precision. **CLEAN.**

**CHECK 3 — re-run published numbers from a SECOND journal [RUN].** Chosen from
`s2` rather than `m2` so the audit does not exercise one code path twice:

    pivot_unsigned__x/s128   BIT-IDENTICAL   rate=0.001220703125   k=5/4096
    pivot_signed__x/s8       BIT-IDENTICAL   rate=0.0263671875     k=108/4096

**CLEAN.** These are two of the numbers the S2 verdict rests on, and they
reproduce from scratch.

**CHECK 4 — every LOCK, against the ARCHIVED copy [RUN].**

    LOCK M2 efadc390c93f -> efadc390c93f   archived==live: True   CLEAN

Verified against `results/m2_item_text.txt`, **not** against a slice boundary
that moves when a neighbouring item is appended. That distinction is why
iteration 1's naive check cried wolf, and it is now the standing method.

**AUDIT VERDICT: 4/4 CLEAN, 0 struck.** No claim leaves the verdict.

**ONE INSTRUMENT NIT, recorded rather than ignored.** The LOCK-line scraper in
this pass used `re.findall(r"^LOCK (\S+) (\S+)", ...)` against STATE.md and
returned `[('hash', 'against')]` — a false positive matched out of the PROSE
"verify every LOCK hash against the archived copy", not a real lock declaration.
Harmless here because M2 is the only LOCK and it was verified directly, but it
is a checker that would silently miscount locks once there are several. Fixing
it is not this iteration's action; it is recorded so it is not rediscovered.

CHECKLIST: no status changed. M2' UNTESTED; R2 screening not yet run.

---

## CHASE — R2 (sign-determinacy) BUILT, RUN, AND ATTACKED

Instrument: `scale/r2_units.py` (new, own journal `results/r2.jsonl`, 36/36 units,
budget-bucketed). `PROTOCOL: SCALING` (i=s-1, j=s/4, c drawn from P), k=8, 32
draws/cell, 10^4 magnitude resamples/draw, CPU. `run_calib.py --self-test` exit 0,
4/4 bit-identical, gate observed rejecting a wrong target. G2 clean.

### THE REDs, all three fired BEFORE any real number was read

`python scale/r2_units.py --red`, exit 0.

**RED 1 — the naive determined fraction is 0.70 by construction.** `A` is
strictly lower triangular, so all 36 of the 64 entries of `(I-A_P)^{-1}` with
i <= j are EXACTLY 0 at every resample: bitwise-constant sign, hence "determined"
under a naive reading. Measured on a pattern whose real determinacy is 0.3214:
`frac_det_naive = 0.7031`. Same class as the M2 control that was zero by
construction; worth 0.56 of a headline number for free.

**RED 2 — a narrow magnitude band fakes determinacy.** 50 random dense k=8
patterns, 10^4 resamples each:

| magnitude band | empirical determined | exact (combinatorial) | disagreements |
|---|---|---|---|
| `exp U(-0.1, +0.1)` | **0.7321** | 0.3886 | **481** |
| `exp U(-6.91, +6.91)` (3 decades) | **0.3886** | 0.3886 | **0** |

Disagreement is one-sided: a narrow band OVER-reports determinacy. Independently
reproduced by a nurse at k=6 with an implementation sharing no code with mine
(exhaustive `itertools` chain enumeration, 200 patterns, 20 000 resamples): wide
band 1525/1525 exact-determined also empirically determined and **0** false
positives; narrow band **714 false positives out of 1475**.

**RED 3 — R2's pre-registered kill CANNOT FIRE.** For a dense k x k
strictly-lower sign pattern every entry at distance i-j = 1 has exactly ONE chain,
so it is sign-determined unconditionally. Hence

    frac_det >= (k-1) / (k(k-1)/2) = 2/k = 0.2500 at k=8

20 000 random dense patterns, exact: mean **0.3811**, **min 0.2500**, max 0.8214.
The minimum is attained and equals the floor. Nurse, independent implementation,
20 000 patterns: mean **0.382352**, min count **7/28 = 0.2500**. R2's kill is
"determined fraction ~0". **It is unreachable by construction.**

Chain counts and per-distance determinacy (nurse, exhaustive, k=8, 20 000
patterns): d=1 1 chain frac 1.000000 | d=2 2 chains 0.502883 | d=3 4 chains
0.125470 | d=4 8 chains 0.014650 | d=5 16 chains 0.000817 | d=6 32 chains
0.000075 | d=7 64 chains 0.000000.

### The GREENs the task names, both observed

all-positive (softmax) pattern **1.0000**; deliberately-broken pattern (half the
signs negated) **0.2857**. The instrument separates them.

### THE CURVE — determined fraction vs s (the deliverable)

Empirical (10^4 resamples, bitwise constant) and exact (combinatorial DP) agree
on **every one of the 36 units, 0 disagreements**. CP interval on pooled entries;
`d[...]` is the draw-level interval, which is the honest one because entries of
one block share edges and are not independent.

| arm | s=8 | 32 | 128 | 512 | 1024 | 2048 | slope |
|---|---|---|---|---|---|---|---|
| tgate | 0.4729 | 0.3917 | 0.3895 | 0.3850 | 0.3906 | 0.4062 | **-0.023** |
| softmax | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000 |
| softmax_broken | 0.4750 | 0.3633 | 0.3934 | 0.4728 | 0.4643 | 0.5220 | +0.031 |
| deltanet | 0.4542 | 0.3304 | 0.3315 | 0.3214 | 0.3047 | 0.2980 | -0.062 |
| sgate | 0.4708 | 0.3862 | 0.3873 | 0.3783 | 0.3728 | 0.3969 | -0.028 |
| **random (chance)** | 0.4958 | 0.3772 | 0.3750 | 0.3795 | 0.3605 | 0.3527 | -0.048 |

CP at s=2048: tgate 0.4062 [0.374, 0.439], random 0.3527 [0.321, 0.385].

**The curve is FLAT — and flat carries no information here.** The block is k x k
at every s, so flatness is guaranteed by the geometry, exactly the defect that
voided the LO probe at fixed i=7,j=1,c=4. Worse, every signed arm sits ON the
random-pattern baseline: tgate 0.39-0.41 against random 0.35-0.38, deltanet
BELOW it. The measurement does not distinguish the trained-operator class from a
coin flip, cannot reach 0, and reaches 1.0 only for the all-positive pattern
where it is true by construction.

**VERDICT: R2's kill cannot structurally fire. The route is decoration.**

### THE TRAP — does determinacy kill the flip M2 measures? PARTLY. 35%, not 100%.

`python scale/r2_units.py --trap 32 128 512 --trap-draws 4096`.

BOUND TO THE SHIPPED INSTRUMENT, not a second implementation: same draw sequence
as `pivot_probe.run_arm`, flip via `grad[j].sum() == (A+hop2)[i,j] * wo.sum()`.
The bind is that `k` reproduces the journalled M2 cells exactly --
**117 / 127 / 118 at s = 32 / 128 / 512**, rates 0.02856 / 0.03101 / 0.02881
against journalled 0.028564 / 0.031006 / 0.028809. Bit-identical, so this is M2's
own flip event being decomposed, not a lookalike.

Perturbing `x[c]` changes EXACTLY ONE term of the readout, `t_c = A[i,c]*A[c,j]`:
`A[i,j]` and every `p != c` term are functions of tokens other than c.

| s | n | flips | t_c sign moved | flip & t moved | flip & t FIXED | share fixed |
|---|---|---|---|---|---|---|
| 32 | 4096 | 117 | 0.3540 | 75 | **42** | **0.3590** |
| 128 | 4096 | 127 | 0.3635 | 82 | **45** | **0.3543** |
| 512 | 4096 | 118 | 0.3706 | 77 | **41** | **0.3475** |

`flip & t FIXED` is the flip driven by MAGNITUDE alone at an unchanged sign --
the event sign-determinacy forbids. It is **35%** of M2's flips and **flat over a
16x context growth**. The other 65% need `sign(t_c)` to move, which a determinacy
condition on a FIXED pattern does not forbid at all.

**So the standing caveat is right in spirit and wrong in mechanism.** Determinacy
does not drive the flip rate to zero; it costs 35% of it and leaves the flatness
intact. The rate that drives to zero is softmax's, and softmax gets there by a
STRICTLY STRONGER property -- an all-positive pattern, where `sign(t_c)` can
never move AND no term can ever oppose the sum. R2's own constraint is survivable;
R2's own test is not informative.

### PATH COHERENCE, the named replacement — MEASURED, AND IT DOES NOT SAY WHAT C1 HOPED

`coh = |sum w| / sum|w|` over the j->i paths through c at hop budget 3, against a
same-N same-magnitude random-sign null (N random-sign terms give ~N^{-1/2} free,
so a raw coherence without its null measures the term COUNT).

| arm | quantity | s=32 | 128 | 512 | 1024 | 2048 | slope |
|---|---|---|---|---|---|---|---|
| tgate | coherence, DENSE bundle | 0.3550 | 0.2675 | 0.1885 | 0.2825 | 0.2741 | **-0.163** |
| tgate | its null | 0.3605 | 0.1500 | 0.0761 | 0.0512 | 0.0353 | **-0.553** |
| tgate | coherence, PIVOT bundle | 0.6995 | 0.7556 | 0.5675 | 0.6105 | 0.7610 | -0.013 |
| tgate | its null | 0.6742 | 0.6812 | 0.6323 | 0.6208 | 0.6676 | -0.025 |
| random | coherence, DENSE | 0.3345 | 0.1298 | 0.0523 | 0.0342 | 0.0237 | -0.603 |
| random | its null | 0.2587 | 0.1248 | 0.0583 | 0.0421 | 0.0305 | -0.539 |

Terms in the bundle: dense 22 / 94 / 382 / 766 / **1534** (slope +1.062); pivot
5.78 / 5.94 / 6.06 / 6.34 / **6.03** (slope +0.064). The `random` arm reads at its
own null at every size (0.0237 vs 0.0305 at s=2048), so the null is calibrated.

**THE RESULT IS BACKWARDS FROM C1's HOPE.** The DENSE bundle is already coherent
far beyond chance -- **7.8x its null at s=2048** (0.2741 vs 0.0353) and decaying
3.4x slower than chance (-0.163 vs -0.553). deltanet dense is **13.4x** its null
(0.4384 vs 0.0326) and RISING (+0.022). The PIVOT bundle's excess is only
**1.14x** (0.7610 vs 0.6676). Routing does not create coherence; it raises the
absolute number by cutting N and REDUCES the excess over chance.

Mechanically: c's own path bundle was never the thing that cancels. The dense
arm's death (-1.009) therefore cannot be internal cancellation of c's bundle --
it has to be the BACKGROUND outgrowing `t_c`, which is A1's small-ball account and
is already measured as `absmag` (pivot -0.046 vs dense -0.930). **Path coherence
as a pivot justification is RED.**

### WHY EVERY SIGN-BASED ROUTE IS ONE-SIDED [DERIVED from the bit-identical bind]

The readout is `A[i,j] + sum_{p in P} A[i,p]A[p,j]`, and c enters exactly one
term. "Does c flip the sign" is therefore exactly "can `|t_c|` beat `|the rest|`"
-- a magnitude comparison. A sign-pattern condition can FORBID the flip (when all
terms share a sign, as in softmax) but can never PROTECT it. So sign-based
routes have power in one direction only: the direction that destroys the claim.
R2 is not merely uninformative here, it is structurally incapable of supporting
M2'; and this applies to any successor built on qualitative matrix theory.

### REPLACEMENT PROPOSED — R2b, COUNT-AND-SCALE, with a kill that fires

Not sign-based, because the paragraph above says sign-based cannot work. The flip
predicate decomposes exactly into two factors that are each checkable WITHOUT
measuring a flip rate:

  1. **COUNT** -- the number of background terms in the readout. Combinatorial,
     magnitude-free. Measured here: routed **6.03 at s=2048 (slope +0.064)**,
     dense **1534 (slope +1.062)**.
  2. **SCALE** -- whether entry magnitude shrinks with s, i.e. whether the
     operator carries a denominator that sums over context. Already journalled as
     `term`/`sigma` in `results/m2.jsonl` and as the `absmag` slopes.

PRE-REGISTERED KILL, and it can fire because it is a PREDICTION with four
independent chances to be wrong: predict `slope(flip rate)` for each of
`tgate-pivot / tgate-dense / sgate-pivot / softmax` from COUNT and SCALE ALONE,
then compare against the journalled +0.027 / -1.009 / -1.826 / exactly 0.
**Kill: any arm's predicted slope misses the measured slope by more than 0.3**
(the same threshold M2's own kill uses). Softmax is the free calibration point --
COUNT and SCALE must predict its exact 0, and if they do not, the decomposition is
broken before it is applied to anything.

### RECONCILIATION WITH FOREMAN — two R2 readings, one contradiction to settle

`scale/route_dependency.py` (FOREMAN, not touched here) implements R2 on the
SHIPPED READOUT (`r2_bruteforce` / `r2_closed_form`), where this file implements
it on the RESOLVENT of the k x k block. Different objects, both legitimate
readings of the route; his is the one that binds to what M2 reads. Three notes,
raw:

  * His sampler is `torch.rand` = magnitudes in **(0,1)**, so his closed form (the
    reachable set is the open interval `(-N, P)`) is exact FOR THAT PRIOR. The
    qualitative class of Brualdi-Shader is ALL positive magnitudes, `(0, inf)`,
    under which the predicate collapses to "every block term shares one sign" --
    his `r2_all_agree`. **R2's answer depends on a magnitude prior the route never
    specified**, and the two priors give different numbers.
  * We reached the same one-sidedness independently: his "the resampling protocol
    OVER-reports determinacy relative to the exact predicate" is my RED 2 measured
    at 481 disagreements narrow / 0 wide.
  * **PROSE-CODE DIVERGENCE in his file, not corrected by me:** `r2_closed_form`'s
    docstring says "A >= P_mass (when A > 0) or -A >= N_mass (when A < 0)"; the
    code is `if A > 0: return int(A >= N)` / `if A < 0: return int(-A >= P)`.
    P and N are swapped between prose and code. The CODE is the correct one (a
    positive A must survive the most negative excursion, `-N`). His file, his fix.

### OPEN

  * `M2PRIME_PREREGISTERED_READING.md` is owed by Phase 0 and does not exist. Not
    written here (root docs are not mine to create).
  * R2b is specified and its ingredients are measured; the prediction itself is
    NOT yet made. Until it is, R2b is a design, not a result.
  * Trap decomposition ran at s=32/128/512 only (4096 draws each, to bind against
    the journalled M2 cells). s=1024/2048 need 16 384 draws to bind and were not
    run: ~340 s and ~1300 s per cell.
  * The determined-fraction CP intervals treat the 28 entries of a block as
    independent. They are not. Believe the `d[...]` draw-level intervals.
  * Every number here is on RANDOM projections, not trained checkpoints -- same
    limitation C2 already carries.

### ITERATION 6 — 2026-08-25 — R2's KILL CANNOT FIRE. Caught before a single real number.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

ACTION (one): ran R2's RED-first calibration (`scale/r2_units.py --red`, k=8,
10^4 resamples per pattern) **before** any R2 measurement, as
`M2PRIME_PREREGISTERED_READING.md` requires.

**THREE REDs FIRED, and the third is the M2-clause-2 defect repeating.**

**RED 1 — structural zeros inflate the naive fraction.**
    frac_det_naive = 0.7031 on all 64 entries,
      of which **36 are exactly 0 at every resample**
    frac_det (i>j only) = **0.3214**
The operator is strictly lower triangular, so the upper entries are zero by
construction and are trivially "sign-determined". Counting them more than
DOUBLES the reported fraction. Any R2 number that does not restrict to i>j is
measuring triangularity, not sign-solvability.

**RED 2 — a narrow magnitude band FAKES determinacy.**
    narrow (+-0.1):    empirical 0.7321   exact 0.3886   **481 disagreements**
    wide   (+-6.91):   empirical 0.3886   exact 0.3886   0 disagreements
The 10^4-resample empirical method is itself biased unless the magnitude range
is wide enough to actually explore the qualitative class. R2 as specified says
"10^4 resamples" and says nothing about the band — so R2 as specified can report
0.73 where the truth is 0.39.

**RED 3 — THE KILL IS UNREACHABLE.**
    20000 random dense sign patterns, EXACT determined fraction:
      mean 0.3811   **min 0.2500**   max 0.8214   **floor 2/k = 0.2500**

**R2's kill is "determined fraction ~0". The minimum achievable is 0.25.**
The kill **cannot fire**. This is precisely the defect that superseded M2 —
a kill clause whose stated condition is structurally unreachable — and it is
sitting in the route the contract ranked FIRST by cost.

**The calibration ends themselves are CLEAN**, which is what licenses trusting
the three REDs above:
    all-positive (softmax) pattern : **1.0000** (required 1.0000)
    deliberately broken pattern    : **0.2857** (required low)

**VERDICT: R2 is DEFECTIVE, not RED-by-kill and certainly not GREEN.** Its
clause A ("frac ~ 0") is unreachable; its clause B ("constraint destroys
training") is untestable because no trained pivot block exists (iteration 3).
**Both halves of R2's kill are unevaluable today.**

**THE FLOOR IS ITSELF A RESULT, and it is worth keeping.** `2/k = 0.25` is not
noise — on a k x k causal block a fixed fraction of entries is sign-determined
no matter what the pattern is. That is a real statement about sign-solvability
in this geometry, and it means "how much determinacy is there" was always the
wrong question: the interesting quantity is determinacy **above the structural
floor**, which R2 never specified.

**REPLACEMENT, per the standing order (the idea is the patient).** Named in
advance by `M2PRIME_PREREGISTERED_READING.md` outcome B and already built by
CHASE at `scale/r2_units.py:198-208`: **path coherence** — not "the sign is
determined regardless of magnitude" but **"the j->i paths through c SHARE a
sign, so c's contribution adds instead of cancelling"** — judged against a
same-N same-magnitude random-sign null, because N random-sign terms give
~N^(-1/2) for free. It attacks the measured death mechanism directly, it has no
2/k floor problem, and its kill (coherence at or below the null) **can** fire.

CHECKLIST: **M2'/R2 -> DEFECTIVE** (kill unreachable, floor 2/k = 0.25).
M2' itself remains UNTESTED — three routes are untried and the R2 replacement
is specified. The work-stopping clause does NOT fire: M2' is the mandatory item
and it is not RED; R2 is one route within it.

### ITERATION 7 - 2026-08-25 - CHASE reports. Path coherence RED. The sign branch is closed by argument.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical (his run too).
Journal results/r2.jsonl, 36/36 units, own journal, replay MATCH.

**THE STRUCTURAL ARGUMENT THAT CLOSES THE BRANCH - the most valuable thing in
the report, and it is a derivation, not a measurement.**

The readout is `A[i,j] + sum_{p in P} A[i,p] A[p,j]`. Perturbing `x[c]` changes
**exactly one term**, `t_c = A[i,c] * A[c,j]`. So "does c flip the sign of j's
influence on i" is exactly **"can |t_c| beat |the rest|"** - a MAGNITUDE
comparison.

**Sign structure can FORBID the flip. It can never PROTECT it.** Sign-based
routes therefore have power only in the direction that destroys the claim. That
closes **R2 and arsenal item C1 (sign-solvability) as a BRANCH**, not merely as
a failed test - and it explains why the standing caveat was right in spirit:
the rate a fully sign-determined operator reaches is softmax's zero.

**THE TRAP, MEASURED - determinacy costs 35% of M2's flips, not 100%.**
Bound bit-identically to the shipped instrument: reproduces the journalled M2
cells **k = 117 / 127 / 118 at s = 32/128/512**, rates 0.02856 / 0.03101 /
0.02881.

    s      flips   flip & t_c sign FIXED   share
    32      117            42             **0.3590**
    128     127            45               0.3543
    512     118            41               0.3475

Magnitude-only flips - the event determinacy forbids - are **35%, flat over a
16x context growth**. The other 65% need `sign(t_c)` itself to move, which no
fixed-pattern condition forbids. So the caveat was **right in spirit, wrong in
mechanism**: R2's constraint is survivable; R2's TEST is what is uninformative.

**R2's CURVE IS FLAT - AND FLAT IS UNINFORMATIVE HERE.**

    arm        8       32      128      512     1024     2048    slope
    tgate    0.4729  0.3917  0.3895  0.3850  0.3906  0.4062   -0.023
    softmax  1.0000 at every s                                 +0.000
    deltanet 0.4542  0.3304  0.3315  0.3214  0.3047  0.2980   -0.062
    sgate    0.4708  0.3862  0.3873  0.3783  0.3728  0.3969   -0.028
    random   0.4958  0.3772  0.3750  0.3795  0.3605  0.3527   -0.048

**The block is k x k at EVERY s, so flatness here is geometry** - the same
defect that voided the LO probe (instrument #14). CP at s=2048: tgate 0.4062
[0.374, 0.439] against random 0.3527 [0.321, 0.385]. **Every signed arm sits on
the chance baseline; deltanet sits below it.** Empirical == exact on all 36
units, 0 disagreements.

**REPLACEMENT 1 - PATH COHERENCE: MEASURED, AND IT COMES OUT BACKWARDS. RED.**
`|sum w| / sum|w|` on the j->i paths through c at hop 3, against a same-N
same-magnitude random-sign null:

    tgate dense  s=2048:  0.2741 vs null 0.0353  = **7.8x above chance**
                          decaying -0.163 against chance's -0.553
    deltanet dense:       **13.4x** and rising
    **pivot bundle excess: only 1.14x** (0.7610 vs 0.6676)

The `random` arm reads at its own null everywhere, so the null is calibrated.
**Routing does not CREATE coherence - it cuts N and REDUCES the excess.** c's
bundle was never cancelling in the first place, so the dense death (-1.009) is
**background magnitude, not internal cancellation.** The replacement my own
pre-registration named is therefore RED, and RED for a reason that inverts its
premise.

**REPLACEMENT 2 - R2b, COUNT-AND-SCALE. Not sign-based, and its kill fires.**
Predict slope(flip rate) for four arms from COUNT (background terms: routed
6.03, slope +0.064; dense 1534, slope +1.062) and SCALE (entry magnitude vs s,
already journalled as `term`/`sigma`) ALONE. **Kill: any arm missing its
journalled slope by more than 0.3.** Targets: tgate-pivot **+0.027**,
tgate-dense **-1.009**, sgate-pivot **-1.826**, softmax **exactly 0**.
**Four chances to be wrong, and softmax's exact zero is a free calibration
point.** This is the successor and it is adopted.

**FOREMAN COLLISION - reconciled, not edited.** `scale/route_dependency.py`
implements R2 on the SHIPPED readout; Chase's is the block resolvent. Two notes:
his sampler is `torch.rand` in (0,1), so his closed form is exact **for that
prior**, while Brualdi-Shader's qualitative class is (0, inf), under which the
predicate collapses to his `r2_all_agree`. **R2's answer depends on a magnitude
prior the route never specified** - a third way R2 was underdetermined. Also
`r2_closed_form`'s docstring swaps P and N against its own code (code correct,
prose wrong).

**OPEN, carried:** the trap ran at s=32/128/512 only - binding at 1024/2048
needs 16384 draws (~340 s / ~1300 s per cell). CP intervals treat a block's 28
entries as independent; they are not. All numbers on random projections.

CHECKLIST: **R2 DEFECTIVE and now also closed by argument; path coherence RED;
R2b adopted as successor.** M2' remains UNTESTED - R4/R1/R3 untried.

### ITERATION 8 - 2026-08-25 - R2b RED. It fails on exactly the arm its own derivation excludes.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): made R2b's prediction with the formula fixed BEFORE any number
was read, then checked it.

**THE FORMULA, fixed a priori.** The flip event is |t_c| > |background|.
Small-ball gives rate ~ E|t_c| / sigma(background), hence

    slope(rate) = slope(E|t_c|) - slope(sigma_background)

Both inputs are already journalled per unit as `term` and `sigma`. **Nothing is
fitted** - there is no free parameter anywhere in this prediction.
KILL: any of four arms missing its journalled slope(rate) by more than 0.30.

**[RUN] THE RESULT.**

    arm            slope(term)  slope(sigma)  PREDICTED  JOURNALLED    err  verdict
    tgate-pivot         +0.029       +0.067     -0.038      +0.027    0.065  PASS
    tgate-dense         +0.025       +0.936     -0.911      -0.769    0.142  PASS
    sgate-pivot         +0.041       +0.092     -0.050      +0.012    0.063  PASS
    softmax             -3.550       -0.623     -2.927      -1.598  **1.329  KILL**

    worst error 1.329  ->  **R2b IS RED**

**WHAT PASSED IS NOT NOTHING.** The model predicts all three SIGNED arms to
within 0.15, from `term` and `sigma` alone, with no fitted parameter - and those
three span a slope range of 0.8 (from -0.769 to +0.027). Getting a decaying arm
and a flat arm both right from the same two-factor model is the substantive
content, and it stands.

**WHY IT FAILED, and it is diagnosable rather than mysterious.** The
derivation assumes **c enters exactly one term** - which is Chase's structural
argument from iteration 7, and it is TRUE on `wrt="v"`. Softmax was measured on
`wrt="x"`, because that is the ONLY channel where a non-negative operator can
flip at all (on `wrt="v"` it is pinned at exactly 0 by theorem). **On `wrt="x"`
perturbing c also moves `A[c,:]` and `A[:,c]`, so MANY terms change and the
premise is false.**

So R2b was applied outside its own domain, and the arm it failed on is the one
its derivation excludes.

**THE CALL, and I am making it after seeing which arm failed, which is stated
rather than hidden.** A domain-restricted successor is available - the same
model, scope limited to `wrt="v"` where c enters one term. The restriction is
**principled** (it is a derivational fact, not an exclusion chosen to rescue a
number). But it is being proposed AFTER the failure, and re-scoring the same
four arms under a narrower scope would be **tuning a dead arm back into its
test**, which the loop forbids.

**Therefore: R2b is RED and stays RED.** A successor R2c may be pre-registered
with the domain stated in advance and **NEW arms**, not re-scored on these.

**AND THE FAILURE CONFIRMS THE ITERATION-7 ARGUMENT.** Softmax's zero on
`wrt="v"` is structural - it comes from non-negativity, not from magnitudes -
so **no count-and-scale model can ever predict it.** A magnitude-only model must
fail on the one arm whose behaviour is set by sign structure. R2b failing
exactly there is evidence FOR "sign structure forbids, magnitude decides", not
against it.

CHECKLIST: **R2b RED.** M2' still UNTESTED - R4/R1/R3 untried, and the sign
branch (R2, C1) remains closed by argument.

# ============================================================
# FOREMAN — M2' ROUTE-DEPENDENCY AUDIT. R1–R4 ARE ONE MECHANISM.
# ============================================================

**2026-08-25. Question: are R1–R4 four routes, or one mechanism in four
costumes? Answer: one mechanism. "Any ONE suffices" is one shot with three
decoys.**

New files, neither in the live loop: `scale/route_dependency.py` (probe),
`tests/foreman/test_m2prime_routes_are_independent.py` (the RED-first bind).
`run_calib.py --self-test` after the work: gate rejected its wrong target,
then **4/4 bit-identical, exit 0**. G2 does not fire.

## THE RED, BEFORE THE FINDING

`pytest tests/foreman/test_m2prime_routes_are_independent.py -q` —
**8 failed in 17.57 s.** Each test asserts M2''s own premise; each fails.

| test | assertion (M2' premise) | measured |
|---|---|---|
| `test_r2_determinacy_is_not_also_maximal_on_a_sign_free_operator` | `abs(A)` twin scores < 0.99 on R2 | **1.0000** |
| `test_r2_and_the_m2_flip_rate_are_independent_events` | CIs overlap | **0.0046 CP[0.0001,0.0255] vs 0.1750 CP[0.0734,0.3278], disjoint** |
| `test_r1_decoder_recall_can_fail` | off-support perturbation moves the readout | **0.000e+00, 48/48 bitwise identical** |
| `test_r3_balanced_ambiguity_can_fire_on_a_non_negative_operator` | ambiguity on `abs(A)` > 0 | **exactly 0.0000** |
| `test_r4_theta_is_injective_over_its_own_range[32/128/512]` | k(0.7) != k(1.0) | **29=29, 125=125, 509=509** |
| `test_r4_slope_theta_is_defined_everywhere_on_the_grid` | slope non-NaN on the grid | **NaN at theta=1.0, rates [0.0391, 0.0]** |

## THE INSTRUMENT, AND ITS DECLARED CHEATS

The flip statistic is computed algebraically as `sign(A[i,j] + hop2[i,j])`
rather than through autograd, because with `wrt="v"` the leaf is `v` and
`grad[j].sum() = (I + A + hop2)[i,j] * wo.sum()` exactly. **CONTROL:**
`route_dependency.py check` ran pivot_probe's real autograd path on the same
draws — **flip-indicator agreement 40/40, max|autograd − algebraic| = 1.06e-06**
on values of order 1e-1.

Second cheat: `hop2[i,j] = w.sum()`, so `build_arm` is called with a one-element
pivot tensor (`A` does not depend on pivots at all) — O(s^2) instead of O(s^2 k).
Verified against the full matmul on the first draw of every call: **max relative
deviation 5.360e-06 over 45 checks, float32 matmul reassociation only, `A[i,j]`
bit-identical.** Third: draw counts are 256–512, not M2's 16,384; every rate
below carries a Clopper–Pearson interval and **none of it may be journalled as a
bucket**.

## R2 — IT IS A DOMINANCE TEST ON THE ONE-HOP EDGE, IN CLOSED FORM

Over `m in (0,1)^k` the reachable set of `sum_p m_p w_p` is the open interval
`(-N, P)`. So `sign(A[i,j] + sum m_p w_p)` is constant iff

> `A[i,j] >= N_mass` (when `A[i,j] > 0`) or `-A[i,j] >= P_mass` (when `< 0`)

Three numbers. **The 10^4 resamples compute a closed form.** And the deciding
term, `A[i,j]`, is *not in the block being randomized*.

| s | n | D_block (10^4 resamples) | D_closed | randomize one-hop too | all signs agree | flip rate |
|---|---|---|---|---|---|---|
| 128 | 512 | 56/64 | **0.8555** | 0.0215 | 0.0469 | 0.0293 [0.0165,0.0479] |
| 512 | 512 | 55/64 | **0.8438** | 0.0508 | 0.0938 | 0.0273 [0.0150,0.0455] |

Determinacy is 0.84; sign agreement is 0.02–0.05. The other 0.80 is the one-hop
edge dominating. **R2's kill is "determined fraction ~0"; it reads 0.84 and
cannot fire** — not from a structural zero this time, but because the term that
decides the verdict sits outside the set being varied. Same defect class as
clause 2, one abstraction up.

The resampling protocol is also **biased toward not firing**: at s=512 it reports
55/64 determined where the exact predicate gives 53/64. 10^4 uniform draws cannot
reach the endpoints of an open interval, so the error is one-sided.

## R2 AND M2 ARE ONE MEASUREMENT READ TWICE — the partition

Same draws, split by R2's own predicate:

| s | n given determined | flip given determined | n given undetermined | flip given undetermined |
|---|---|---|---|---|
| 128 | 438 | 3 -> **0.0068** CP[0.0014,0.0199] | 74 | 12 -> **0.1622** CP[0.0867,0.2661] |
| 512 | 432 | 3 -> **0.0069** CP[0.0014,0.0202] | 80 | 11 -> **0.1375** CP[0.0707,0.2327] |

Intervals disjoint, ratio 24x and 20x. `0.855*0.0068 + 0.145*0.1622 = 0.0293`,
which is the total rate exactly. **80% of M2's flip mass lives in the 15% of
draws R2 calls undetermined.** R2 is not a second route to the property; R2 is
the complement of the property. **A GREEN on R2 is M2's flip rate going to zero
— which is softmax's number.** `ARSENAL.md:246` said this in prose ("sign-
DETERMINACY ... pushed to its limit drives the flip rate to ZERO"); this is the
number.

## THE TWIN TEST — all four are passed by an operator with NO SIGNS

`scale/s2_probe.py::absmag` runs the identical measurement on `abs(A)`: same
scores, same tau, same gate, magnitudes entrywise identical, signs gone.

| route | criterion | on `A` | on `abs(A)` | softmax |
|---|---|---|---|---|
| R2 | determined fraction (want high) | 0.8555 / 0.8438 | **1.0000 / 1.0000** | 1.0 by theorem (`A = abs(A)`) |
| R3 | balanced ambiguity (want <10%) | 0.0410 / 0.0410 | **0.0000 / 0.0000** | 0.0 by theorem |
| R3 | argmax frozen across branches | 0.8574 / 0.8477 | 0.8574 / 0.8477 | — |
| R1 | off-support influence | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| R4 | slope(R = term/sigma), pivot | −0.04 (Cameron) | **−0.046** | — |
| R4 | slope(R = term/sigma), dense | −0.92 (Cameron) | **−0.930** | — |

n = 512 per cell at s = 128 and 512 for the R2/R3 rows. The R4 rows are **already
in `results/s2_ablation.txt`, at n=1024/512, and the string `absmag` appears
ZERO times in this file** — the twin arm was run and never reported.

**Every one of the four criteria is met at least as well by the sign-free twin.**
A route a non-negative operator passes cannot be evidence for signed influence.

## R1 — the kill cannot fire, and it is clause 2 wearing a decoder

Group testing's defective items are tokens whose perturbation moves the readout.

| arm | s | draws | max abs-delta OFF-support | median abs-delta ON-support | exact zeros OFF |
|---|---|---|---|---|---|
| pivot_signed | 128 | 96 | **0.000e+00** | 3.666e-03 | **96/96** |
| pivot_signed | 512 | 96 | **0.000e+00** | 3.800e-03 | **96/96** |
| dense_signed | 128 | 96 | 8.795e-02 | 3.666e-03 | 17/96 |
| dense_signed | 512 | 96 | 5.201e-02 | 3.800e-03 | 31/96 |

Off-support tokens are **bitwise inert** on the pivot readout, so the decoder
faces a noiseless separation and **recall is 1.0 for any decoder, at any s, at
zero overhead**. "Recall >= 1-eps within polylog overhead" is a theorem about
`sum_{p in P}`, not a measurement. The kill fires only on `dense`, i.e. only if
the pivot mechanism is given up. (The dense arm's 17/96 and 31/96 exact zeros are
a *third* instance of the same defect: tokens with `t < j` contribute
`A[t,j] = 0` by causality, ~25% of the off pool at `j = s/4`.)

**CORRECTION TO MY OWN FIRST VERSION, recorded because it is this repo's bug
class.** The first R1 probe compared hop-2 mass through an 8-token off-support
pool against a 1-token on-support pool and read 1.386e-01 vs 2.910e-02
("off-support is bigger"). It varied POOL SIZE between the arms. It measured
pool size. Fixed by perturbing tokens and reading influence at matched size.

## R4 — theta is not injective and slope(theta) is not defined

`k(s,theta) = 8*s^theta`, clipped at `s-3`. Measured, n=512, 3 seeds,
s = 32/128/512:

| theta | slope, seed 0/1/2 | D_closed (R2) | rho = t_c/abs(A) |
|---|---|---|---|
| 0.0 | −0.091 / −0.146 / −0.000 | 0.8464 | 0.074 |
| 0.3 | −0.104 / −0.316 / −0.604 | 0.4147 | 0.069 |
| 0.5 | −0.542 / −0.279 / −0.354 | 0.1842 | 0.072 |
| 0.7 | **+nan** / −0.925 / −0.170 | 0.1680 | 0.072 |
| 1.0 | **+nan** / −0.925 / −0.170 | 0.1680 | 0.072 |

Three separate failures, all structural:

1. **theta >= ~0.65 is one arm.** k(0.7) = k(1.0) = 29 / 125 / 509 at
   s = 32/128/512. The upper third of the range is a single point.
2. **slope is NaN where the kill must be read.** The rate hits exactly 0 at the
   dense end and `loglog_slope` returns NaN rather than clamping (correctly).
   Instrument #15 died of a NaN slope mapped to a false GREEN. R4 rebuilds one.
3. **theta=0's sign is inside its own noise.** −0.091/−0.146/−0.000 here at
   n=512 against the journal's +0.0270 at n=4096–16384 (R^2 0.5222). The crossing
   R4 must locate is inside the error bar of the endpoint that brackets it.
   **This is R4's one clause that CAN fire — "theta* unstable over 3 seeds" — and
   on this evidence it fires.**

Note the fourth column: **D_closed(theta) falls monotonically 0.846 -> 0.414 ->
0.184 and then saturates exactly where theta saturates.** R2's statistic and R4's
statistic are the same function of the pivot budget.

## R3 — revival or new object? NEITHER: it inherits the corpse and adds a frozen bit

`grep -rni "signed tropical|signed max-plus|S_max|smax"` across the tree:
**zero hits.** R3's object does not exist here. Its two halves:

* **Magnitude channel = the deleted object, unchanged.** `tests/foreman/_ceq.py::
  bellman`, `greedy_policy`, `policy_affine`. `test_r2_maxplus_reduction.py:46`
  asserted `worst > 1e-8` and **measured 9.95e-14** — the star IS the linear
  resolvent of its own greedy policy, whose `E` is 0/1 row-stochastic
  (`_ceq.py:76-77`), i.e. **non-negative**. Gradient: asserted `gap > 1e-4`,
  **measured 1.65e-08**.
* **Sign channel = the argmax path's sign, and it is frozen.** Measured here,
  n=512: the argmax INDEX is unchanged across both c-branches in **0.8574 /
  0.8477** of draws. A locally-constant argmax has zero gradient — the same
  reason the max-plus Jacobian equalled APPNP's at 1.65e-08. So the sign bit
  is piecewise constant, and **M1's own kill covers it: "the most negative entry
  is a frozen zero-gradient cell."**

And R3's two success criteria oppose each other. Balanced-ambiguity < 10% means
the top path dominates in >=90% of positions; but cancellation — the thing a
signed operator is for — is *exactly* the balanced case in the symmetrized
tropical semiring. **R3 asks the operator to cancel and to never be in a
position to cancel.** Its ambiguity criterion is met perfectly (0.0000) by
`abs(A)`.

R3's `<=1.10` training bar is the one clause here that can fire. It fires on a
non-negative object with 3.3x the parameters of the arm it must beat
(`test_r2_signed_consequence_fit.py:22-26`).

## WHICH KILLS CAN STRUCTURALLY FIRE

| route | kill | can it fire? | why |
|---|---|---|---|
| **R1** | recall < 1-eps at s=2048 | **NO** | off-support is bitwise inert, 96/96 exact zeros; recall is 1.0 for every decoder. Clause 2 again. |
| **R2** | determined fraction ~0 | **NO** | reads 0.8438; the deciding term `A[i,j]` is outside the randomized block. And `abs(A)` reads 1.0000. |
| **R2** | loss blowup under the constraint | yes | untested; but the constraint's GREEN is M2's death (partition above). |
| **R3** | balanced-ambiguity >= 10% | **NO** | reads 0.0000 on `abs(A)`; a non-negative operator passes it perfectly. |
| **R3** | trains above 1.10 | **yes** | a real bar on a real optimizer. |
| **R4** | no crossing | **NO** | theta non-injective above 0.65 and slope NaN there; the question is not well-posed on the upper range. |
| **R4** | theta* unstable over 3 seeds | **yes — and it fires** | −0.091 / −0.146 / −0.000 at theta=0. |

**Four of the seven clauses cannot fire. Two of the three that can, fire.**

## THE GENERALIZATION — one line above DONE.md:5885

DONE.md already has: *"the harness varies a quantity in the KERNEL of the map it
measures."* R2 is not that — nothing here is in a kernel; `A[i,j]` is varied by
nothing because it is never varied. The covering statement is:

> **The verdict is decided by a term outside the set the protocol varies.**

Kernel-zero (`c not in P`, M4's `exclude=keep`, `t < j` causality) is the special
case where the outside term is the *only* term. R2's one-hop dominance is the
general case where it is merely the *biggest* term. Both make a criterion read as
rigour while being a fact about the instrument.

## THE ROOT CAUSE, ONE SENTENCE

> **All four routes are thresholds on one scalar — the intervened token's
> two-hop path weight measured against the one-hop offset and the background
> spread — and that scalar is sign-blind (measured: `abs(A)` matches `A` at
> −0.046 vs −0.04 and −0.930 vs −0.92) and homogeneous of degree 1 under per-row
> rescale (the rescale lemma), so no one of them can separate signed influence
> from magnitude bookkeeping.**

## SUCCESSOR — required by the standing rule

**M2''. THE SIGN-BLIND TWIN DIFFERENTIAL.** Not a new mechanism: a *reporting
rule* that every route must satisfy before it counts.

> For any candidate readout `F`, publish `F(A)`, `F(abs(A))` and `F(softmax)` in
> the same table. `abs(A)` is `scale/s2_probe.py::absmag` — magnitudes
> bit-identical, signs stripped, nothing else moved.
> **KILL: if `F(A)` and `F(abs(A))` agree within their intervals, `F` is
> sign-blind and dies, whatever its slope.**

* **Can it structurally fire?** Yes, in both directions, and the instrument is
  calibrated on both. Known-positive: M1's influence-Jacobian minimum,
  **−9.000e-01 on `A`, exactly 0.000e+00 on `abs(A)`** over 40 max-plus and 160
  APPNP instances — a maximal gap. Known-negative: `R = term/sigma`, gap
  **0.006** in slope. There is no term outside the varied set, because the two
  runs differ *only* by the sign pattern of every entry.
* **Softmax's number in the same table, and it is a theorem, not a measurement:**
  softmax is its own twin (`A = abs(A)` entrywise), so its gap is exactly 0. That
  makes softmax the correct floor *and* means any route with a zero twin gap has
  reproduced softmax.
* **Cost:** cheaper than what it replaces. `absmag` is 17 lines and already
  written; one extra forward per draw. R1–R4 need a group-testing decoder, a 10^4
  resample loop, a 3-seed theta sweep and a trained tropical arm.
* **Conversion to M3, because a statistic is not a capability:** run the twin on
  `scale/negation_scope.py` — arm `A` vs arm `abs(A)` at matched magnitudes and
  matched parameters, softmax's failure distance recorded first per M3's text.
  Kill: **if `abs(A)` solves negation-scope at d >= 256 as well as `A` does,
  signedness is decoration and the claim sentence goes.** That is the only
  instrument in the repo where the designated token comes from the TASK and not
  from the selector — Cameron's repair, already on the board.

**WHAT THE SUCCESSOR DOES NOT RECOVER, stated plainly.** It tests the SIGN half
of "signed influence that context cannot dilute". It does not test the DILUTION
half, and the rescale lemma already showed that half is equivalent to
lambda = Theta(1), an unbounded hop-2 norm. The honest successor claim is
sign-dependence at fixed magnitude, not context-invariance. Anyone wanting both
is asking for a bounded operator with a scale-invariant sign readout, and the
lemma says pick one.

## WHAT I COULD NOT VERIFY

* **R3's `<=1.10` training bar was not run.** No signed-tropical arm exists to
  train; building and training one is not a CPU-minute job. My R3 verdict rests
  on the frozen-argmax measurement (0.857/0.848) plus the journalled 9.95e-14 /
  1.65e-08, not on a training curve.
* **R2's second kill clause ("loss blowup under the constraint") is untested** —
  no training was run under a sign-determinacy constraint.
* **Nothing here was measured at s=2048**, where R1's and R2's kills are written.
  The structural arguments (bitwise-inert off-support; `A[i,j]` outside the
  randomized block) are s-independent; the *rates* are not, and are reported at
  s <= 512 only.
* **All measurements are on `tgate`, on random projections, not trained
  checkpoints** — and DONE.md's SCOPE RED stands: `tgate` ships nowhere.
* **theta=0's slope sign is unresolved at these draw counts.** I read
  −0.091/−0.146/−0.000 where the journal reads +0.0270 at 8–32x the draws. I do
  not claim the journal is wrong; I claim the sign is not resolvable at n=512,
  which is itself R4's problem.
* **No whole-suite pytest was run** and no pass/fail total is quoted.

### ITERATION 9 - 2026-08-25 - R4's kill is ILL-POSED. Fourth in a row. But the mechanism is CONFIRMED.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): checked whether R4's kill can fire BEFORE building R4 - the check
that has caught something in every route so far.

**THE DERIVATION, made before the measurement.** Dyson tree with level coupling
2^(-theta*l): level l holds ~2^l terms, so the background variance is
sigma^2 ~ sum_l 2^l * 2^(-2 theta l) = sum_l 2^((1-2theta) l). Over
L = log2(s) levels:

    theta < 1/2 : the sum is dominated by the top level, sigma ~ s^((1-2theta)/2)
    theta > 1/2 : the sum CONVERGES, sigma ~ constant in s

With slope(rate) = slope(E|t_c|) - slope(sigma) this gives

    **slope(theta) = theta - 0.5  for theta < 0.5,  and 0 for theta >= 0.5**

**[RUN] THE MEASUREMENT confirms it.**

    theta     seed0     seed1     seed2    spread   predicted
     0.00    -0.850    -0.411    -0.392    0.458     -0.500
     0.20    -0.254    -0.320    -0.332    0.078     -0.300
     0.35    -0.095    -0.118    -0.218    0.123     -0.150
     0.50    -0.055    -0.061    -0.048    0.013     +0.000
     0.65    -0.015    -0.039    +0.005    0.044     +0.000
     0.80    -0.005    -0.017    +0.001    0.018     +0.000
     1.00    +0.015    -0.029    +0.014    0.044     +0.000
     1.50    +0.023    -0.019    +0.007    0.042     +0.000

    slopes at theta >= 0.5:  0.50:-0.055  0.65:-0.017  0.80:-0.007
                             1.00:+0.000  1.50:+0.004
    **max |slope| above 0.5 = 0.055**

**THE DEFECT: theta* IS A HALF-LINE, NOT A POINT.** Above theta = 0.5 the slope
is identically zero for EVERY theta tested, out to 1.5. R4's kill says
*"no zero crossing of slope(theta) in range, OR theta* unstable over 3 seeds"*.
There IS a crossing, so clause 1 cannot fire; and **theta* is not identifiable
at all**, because any theta >= 0.5 gives slope 0. **Stability of an
unidentifiable parameter is not a test.** Fourth route in a row with a kill that
cannot do its job - after R2 (floor 2/k = 0.25 unreachable), path coherence
(backwards), and R2b (out of its own domain).

**BUT THE MECHANISM IS REAL, AND IT IS NOT THE SAME AS PIVOT ROUTING.** This is
the first genuinely NEW escape the project has confirmed:

    pivot routing   : flatness by CUTTING THE TERM COUNT to k (measured
                      s^+0.048 against dense s^+1.062)
    Dyson coupling  : flatness by KEEPING every term but making the sum
                      CONVERGE - sum_l 2^((1-2theta)l) converges for theta > 1/2

Those break different hypotheses. Pivot routing breaks "generic token" (R1's
break, achieved structurally - iteration 2). Dyson coupling breaks **"flat
aggregation"**, which is exactly what R4 advertises. **Two independent escapes,
now both derived and both measured.**

And the quantitative agreement is the evidence: predicted -0.300 at theta=0.20
against measured -0.254/-0.320/-0.332; predicted -0.150 at theta=0.35 against
-0.095/-0.118/-0.218. The seed spread tightens sharply in the flat region
(<=0.044 for theta >= 0.5 against 0.458 at theta=0), which is what a genuine
plateau looks like.

**REPLACEMENT: R4b, a SHAPE kill that is well-posed.** Do not test "theta* is
stable" - test the shape, at two pre-registered points that can each fail:

    slope(theta = 0.20) must be <= -0.20   (the decaying regime is real)
    slope(theta = 0.80) must be within 0.10 of 0   (the flat regime is real)
    both over 3 seeds, both reported with spread

**Both clauses can fire in both directions**, neither depends on locating a
point inside a plateau, and the two together are the actual content of
"hierarchical criticality". Pre-registered here, before R4b is built.

CHECKLIST: **R4 kill ILL-POSED; R4b pre-registered as its replacement.**
M2' still UNTESTED - R1/R3 untried.

## CORRECTION AND SHARPENING — R2's VERDICT IS SET BY A PRIOR NOBODY CHOSE

**Prompted by the reconciliation note in this file's R2b section, which reviewed
`scale/route_dependency.py` and found a real defect in it. Both points accepted;
one is a bug I own, the other makes the finding above stronger, not weaker.**

**1. Prose–code divergence in `r2_closed_form`, MINE, now fixed.** The docstring
read "A >= P_mass (when A > 0) or -A >= N_mass (when A < 0)"; the code reads
`if A > 0: return int(A >= N)`. The **code was right** — a positive `A` must
survive the most negative excursion, which is `-N` — and it is unchanged, so no
number moves. The docstring is corrected in place with a note recording that it
was wrong. The journal entry above already carried the correct form; only the
source comment was swapped.

**2. The prior. This is the sharper version of "R2's kill cannot fire".**
`r2_bruteforce` samples `torch.rand`, i.e. magnitudes in **(0,1)**. Brualdi &
Shader's qualitative class — which R2 cites — is **(0, ∞)**. R2's own text says
"10⁴ resamples" and names no prior at all. Added `r2_closed_form_unbounded`:
under (0,∞) the reachable set of `Σ m_p w_p` is all of R whenever both signs
appear, so no `A[i,j]` can dominate and the predicate collapses to *the block
terms share a sign AND the one-hop edge agrees with it*.

Same draws, same operator, same geometry, n=512:

| s | determined, m~U(0,1) | determined, m~(0,∞) | R2's kill under each |
|---|---|---|---|
| 128 | **0.8555** | **0.0215** | CANNOT fire / **FIRES** |
| 512 | **0.8438** | **0.0508** | CANNOT fire / **FIRES** |

**R2 returns GREEN or RED depending on a magnitude prior nobody selected.** That
is worse than a kill that cannot fire, and it is a third variety of the same
disease: the bounded prior makes `A[i,j]` decisive (a term outside the block),
the unbounded prior makes it irrelevant. Either way the verdict is carried by
something the route did not name.

RED for it: `test_r2_gives_the_same_verdict_under_both_magnitude_priors` —
**9 failed in 11.20 s** on the file now.

**Which prior is right?** The unbounded one, if R2 means what its citation
means. Under it R2 is RED on arrival at 0.0215/0.0508, and outcome **A** of
`M2PRIME_PREREGISTERED_READING.md` applies — "the sign pattern determines
nothing". That is consistent with, and independent of, the partition above
showing that R2 GREEN would have been M2's death anyway. **R2 is RED under its
own citation's prior and vacuous under the other. There is no prior on which it
is informative.**

### ITERATION 10 - 2026-08-25 - M2' IS RED. All four routes are SIGN-BLIND. M2'' adopted.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical (both runs).

**FOREMAN'S TWIN TEST KILLS THE WHOLE PROGRAM, not one route.** Run the
identical criterion on `A` and on `|A|` - magnitudes bit-identical, signs
stripped (`scale/s2_probe.py::absmag`, 17 lines, already existed and appeared
ZERO times in DONE.md):

    route  criterion                        on A            on |A|      softmax
    R2     determined fraction              0.8555/0.8438   **1.0000**  1.0 by thm
    R3     balanced ambiguity (<10% wanted) 0.0410          **0.0000**  0.0 by thm
    R1     off-support influence            0.000e+00       0.000e+00   0.000e+00
    R4     slope(term/sigma) pivot / dense  -0.04 / -0.92   **-0.046 / -0.930**

**EVERY CRITERION IS MET AT LEAST AS WELL BY AN OPERATOR WITH NO SIGNS AT ALL.**
M2' does not fail because a route failed; it fails because all four routes
measure a sign-blind scalar.

**ROOT CAUSE, his sentence:** all four routes are thresholds on ONE scalar - the
intervened token's two-hop path weight against the one-hop offset `A[i,j]` and
the background spread - and that scalar is sign-blind (measured) and homogeneous
of degree 1 under per-row rescale.

**R2 AND M2 ARE THE SAME MEASUREMENT READ TWICE, WITH OPPOSITE POLARITY.**
Same 512 draws, split by R2's own predicate:

    s      flip | R2-determined            flip | R2-undetermined
    128    0.0068 CP[0.0014,0.0199] n=438  0.1622 CP[0.0867,0.2661] n=74
    512    0.0069 CP[0.0014,0.0202] n=432  0.1375 CP[0.0707,0.2327] n=80

Intervals disjoint, **24x / 20x**. And
`0.855*0.0068 + 0.145*0.1622 = 0.0293` - **M2's total rate exactly.**
**80% of M2's flip mass sits in the 15% of draws R2 calls undetermined.**
**R2 GREEN is M2's flip rate going to zero, which is softmax's number.** The
trap `M2PRIME_PREREGISTERED_READING.md` fixed in prose at iteration 3 is now a
number.

**FOUR OF SEVEN KILL CLAUSES CANNOT FIRE.**
  * **R1** - perturbing an off-support token moves the pivot readout by
    **0.000e+00, 96/96 bitwise identical** at s=128 and 512. `hop2[i,j]` has no
    term with index t not in P, so recall is 1.0 for ANY decoder at zero
    overhead. **Clause 2 wearing a decoder.** (Independently confirms
    iteration 2.) On `dense_signed` the same probe reads 8.795e-02 - the kill
    fires only if the pivot mechanism is abandoned.
  * **R2** - its verdict is set by a magnitude prior it never states: 0.8438
    under m~U(0,1) (cannot fire) against 0.0508 under m~(0,inf), which is
    Brualdi-Shader's actual qualitative class (fires).
  * **R3's ambiguity clause** - exactly 0.0000 on `|A|`; its two criteria also
    oppose each other, since cancellation IS the balanced case in the
    symmetrized tropical semiring.
  * **R4's "no crossing"** - `k = 8*s^theta` clips at s-3, so theta=0.7 and
    theta=1.0 are the IDENTICAL arm (k=29/125/509 at s=32/128/512), and slope is
    NaN there because the rate hits exactly 0. **Instrument #15's NaN-as-GREEN,
    rebuilt.**

**GENERALIZATION, one line above the kernel statement:** *the verdict is decided
by a term OUTSIDE the set the protocol varies.* Kernel-zero is the special case
where that term is the only term; R2's dominance is the case where it is merely
the biggest.

**R3 INHERITS THE MAX-PLUS CORPSE.** No signed-tropical object exists here
(`S_max` and "signed tropical": zero hits). The magnitude channel is
`bellman`/`greedy_policy` unchanged - star equals the resolvent of its own 0/1
row-stochastic NON-NEGATIVE greedy policy at 9.95e-14, gradient 1.65e-08. The
sign channel is the argmax path's sign, and the argmax **index is frozen across
both c-branches in 0.8574 / 0.8477 of draws** - piecewise constant, no gradient.
**That is M1's own kill clause: "the most negative entry is a frozen
zero-gradient cell."**

**MY R4b AGREES FROM THE OTHER SIDE, after fixing instrument #16.**
INSTRUMENT #16, MINE: I weighted `a` and the RETURNED `hop2` separately, which
is a POSITIVE ELEMENTWISE RESCALE of an already-summed quantity - and a positive
rescale **cannot change a sign**. theta=0.20 and theta=0.80 gave BIT-IDENTICAL
rates (0.0250/0.0000/0.0000 at both). Fixed so the coupling enters the hop-2
SUM, where per-p weights differ. [RUN] after the fix, `pivot_signed` at s=128,
256 draws: **rate 0.000000 at theta = 0.0, 0.20, 0.80, 1.5.** The proxy said
flat; **the real operator says the coupling kills the property outright.**
Calibration held throughout: theta=0 reproduces the unweighted operator bitwise
(True), softmax reads exactly 0.000000 on the value path.

**SUCCESSOR ADOPTED - M2'', THE SIGN-BLIND TWIN DIFFERENTIAL.**
For every candidate readout F, publish **F(A), F(|A|), F(softmax) in ONE
table**. **KILL: if F(A) and F(|A|) agree within intervals, F is SIGN-BLIND and
dies.** It fires in both directions and is calibrated at both ends:
  known-positive - M1's influence-Jacobian min **-9.000e-01 on A, exactly
                   0.000e+00 on |A|**
  known-negative - `R = term/sigma`, gap 0.006
  softmax's column is a **theorem** (A = |A|).
The instrument already exists (17 lines) and is cheaper than any of R1-R4. It
converts to M3 on `scale/negation_scope.py`, the only instrument where the
designated token comes from the TASK rather than the selector. **It recovers the
SIGN half only** - the dilution half is equivalent to lambda = Theta(1) by the
rescale lemma.

**RED EVIDENCE:** `tests/foreman/test_m2prime_routes_are_independent.py` -
**9 failed in 11.20 s**, each asserting M2's own premise. Controls: algebraic
flip indicator vs real autograd **40/40, max dev 1.06e-06**; fast-path hop2 max
rel dev 5.360e-06 (float32 reassociation, `A[i,j]` bit-identical).

**COULD NOT VERIFY (his):** R3's <=1.10 training bar and R2's "loss blowup" -
no arm exists to train. Nothing at s=2048 where R1's and R2's kills are written;
the structural arguments are s-independent, the rates are not. All on `tgate`,
random projections, not trained checkpoints. theta=0's slope sign unresolved at
n=512.

CHECKLIST: **M2' RED** - all four routes sign-blind by the twin test.
**M2'' adopted as successor** per the standing order. Work-stopping applies to
M2' and is discharged by the successor, exactly as M2 -> M2' was.

### ITERATION 11 - 2026-08-25 - M2'' appended. Calibrated at ONE end, not two.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): appended **M2''** verbatim to CHECKLIST.md and ran its own
calibration.

**FLAG RAISED BEFORE RUNNING, AND IT WAS JUSTIFIED.** M2'''s proposed
known-positive anchor was **-9.000e-01**. The Round 1 record says of that exact
number: *"-9.000e-01 is -rho, read off the zero-gradient entry A[1,0]"* - the
frozen-zero-gradient-cell trap that **M1's own kill clause names**. Calibrating
a new instrument against a discredited number would poison it at the root, so
the anchor was measured fresh instead of inherited.

**[RUN] KNOWN-POSITIVE END: WORKS.** Influence-Jacobian min over STRICTLY CAUSAL
entries (the restriction matters - unrestricted, the identity diagonal floors
it):

    arm                     F(A)            F(|A|)      differs?
    pivot_signed  s=64   **-6.050431e-01**   4.522256e-05   YES
    pivot_signed  s=256  **-5.960416e-01**   1.024215e-06   YES
    dense_signed  s=64     -7.095125e-01     1.023862e-03   YES
    dense_signed  s=256    -3.447314e+00     6.117090e-04   YES
    pivot_unsigned s=64     0.000000e+00     0.000000e+00   no (SIGN-BLIND)
    pivot_unsigned s=256    0.000000e+00     0.000000e+00   no (SIGN-BLIND)

**The real anchor is -6.050e-01, NOT -9.000e-01.** The flag was correct: the
inherited figure was the artifact. M2'' is now anchored on a fresh measurement.
Softmax reads sign-blind at both sizes, which is the behaviour the test must
show for a non-negative operator.

Note in passing: `dense_signed` at s=256 reads **-3.447e+00** against
`pivot_signed`'s -5.96e-01 - the unbounded row-L1 growth of the dense arm
showing up in the Jacobian, consistent with the O(s*g) magnitude risk on record.

**[RUN] SOFTMAX COLUMN IS A THEOREM: CONFIRMED.**
    max|A - |A|| = **0.000e+00**    min A = **0.000e+00**
A = |A| entrywise, exactly, so the softmax column must agree and does.

**[RUN] KNOWN-NEGATIVE END: DOES NOT REPRODUCE.** FOREMAN's anchor was
`R = term/sigma` with gap **0.006** (i.e. sign-blind). Measured here:

    pivot_signed s=64 : F(A)=2.143791  F(|A|)=0.344951  gap **1.798840**
    pivot_signed s=256: F(A)=1.826988  F(|A|)=1.306226  gap **0.520762**

It reads sign-SENSITIVE, by two to three orders of magnitude more gap than
claimed. Either his `R` and mine are different quantities - plausible, since
`R = term/sigma` is under-specified as written and I take `term` as
|A[i,c]*A[c,j]| and `sigma` as the std of the pivot-bundle weights - or the
0.006 is wrong.

**VERDICT: M2'' IS CALIBRATED AT ONE END, WHICH IS NOT CALIBRATED.** The
standing doctrine is *"calibrate every instrument against a case where it must
fire and one where it must not, before believing it"*. The must-fire end is
verified; the must-not-fire end is not. **No F may be judged by this test until
a known-negative reproduces**, because without it the test cannot be
distinguished from one that calls everything sign-sensitive.

**This is the correct outcome to record rather than a setback.** A twin test
with only a must-fire anchor would have called R2, R3 and R4 sign-sensitive too
- the exact opposite of what Foreman measured with `absmag` - and the
disagreement would have surfaced as a contradiction between two of my own
results instead of as a missing calibration.

CHECKLIST: **M2'' UNTESTED, calibration INCOMPLETE (1 of 2 ends).**

### ITERATION 12 - 2026-08-25 - M2'' CALIBRATED AT BOTH ENDS, on theorem-backed anchors.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): resolved M2'''s missing known-negative end. Rather than chase
FOREMAN's under-specified `R = term/sigma` (his gap 0.006 against my 1.799 /
0.521 - iteration 11), used anchors that are sign-blind **BY THEOREM**: any
readout that is a function of |A| alone satisfies F(A) == F(|A|) IDENTICALLY,
so the anchor is a proof rather than a measurement and cannot drift.

**[RUN] KNOWN-NEGATIVE END: VERIFIED, and exactly.**

    readout            arm                 F(A)             F(|A|)        gap
    row-L1 (max)       pivot_signed    23.071105957     23.071105957   0.000e+00
    row-L1 (max)       dense_signed    24.398654938     24.398654938   0.000e+00
    Frobenius          pivot_signed    12.154439926     12.154439926   0.000e+00
    Frobenius          dense_signed    10.858698845     10.858698845   0.000e+00
    nnz                pivot_signed  8128.000000000   8128.000000000   0.000e+00
    nnz                dense_signed  8128.000000000   8128.000000000   0.000e+00
    particip. ratio    pivot_signed  4668.108398438   4668.108398438   0.000e+00
    particip. ratio    dense_signed  4714.416015625   4714.416015625   0.000e+00

    **worst gap across all four theorem-backed anchors: 0.000e+00**

**[RUN] MUST-FIRE END, on the SAME draws** (so the two ends are not measured on
different data): influence-Jacobian min **F(A) = -4.776115e-01**,
**F(|A|) = +8.638127e-05**, gap **4.777e-01**. FIRES as required.

**M2'' IS NOW A USABLE INSTRUMENT** - calibrated against a case where it must
fire and one where it must not, which is the standing doctrine's requirement and
the thing eleven of the sixteen broken instruments here lacked.

**WHY THE THEOREM-BACKED ANCHOR IS STRICTLY BETTER than the one it replaces.**
It returns EXACTLY zero rather than a small number; it cannot drift with seed,
size or arm; and it needs no agreement about what an under-specified quantity
means. The iteration-11 disagreement (0.006 vs 1.799) simply cannot arise for
`row-L1` - both sides are the same expression.

**THE DOMAIN NOTE, and it decides how M2'' should be used.** Applying M2'' to
the FLIP RATE itself is trivially "sign-sensitive": a non-negative operator has
`I + A + A^2` non-negative entrywise, so `F(|A|) = 0` **by theorem**, and the
gap is guaranteed. That is the must-fire anchor restated, not new information.

**M2'''s value is as a SCREEN FOR PROPOSED CRITERIA before a route is built on
them** - which is exactly how FOREMAN used it to kill all four M2' routes at
once (`absmag`: determined fraction 0.8555 -> 1.0000 on |A|; balanced ambiguity
0.0410 -> 0.0000; off-support influence 0.000e+00 both; slope -0.04/-0.92 ->
-0.046/-0.930). Run it on a criterion BEFORE spending a route on it, and the
four dead routes of M2' cost one table instead of ten iterations.

CHECKLIST: **M2'' calibration COMPLETE (2 of 2 ends).** Status remains UNTESTED
as an ITEM - the instrument is ready; no candidate readout has yet been judged
by it and converted to M3.

### ITERATION 13 - 2026-08-25 - NO readout carries persistence AND separation AND sign-sensitivity.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): asked the question M2'' exists to answer - **is there any readout
that (a) is flat in s, (b) separates pivot from dense, and (c) passes M2''
non-trivially?** If not, the project's claim rests on the flip rate alone.

**[RUN] PROTOCOL: SCALING, 12 draws/cell, s = 32/128/512/2048.**

    readout               arm            32       128       512      2048   slope   F(|A|)
    min J (causal)   pivot_signed    -0.4480   -0.5413   -0.6905   -0.7981  +0.142   0.0000
    min J (causal)   dense_signed    -0.5243   -1.7964   -8.4884  -40.9220  +1.055   0.0000
    frac J < 0       pivot_signed     0.5123    0.4974    0.4999    0.5001  -0.005   0.0000
    frac J < 0       dense_signed     0.5156    0.4993    0.4997    0.5001  -0.007   0.0000
    neg mass/total   pivot_signed     0.5043    0.4996    0.4995    0.5001  -0.002   0.0000
    neg mass/total   dense_signed     0.5070    0.5007    0.4994    0.4998  -0.003   0.0000

**THE PATTERN IS CLEAN AND IT IS NEGATIVE.**
  * readouts that SEPARATE pivot from dense are **not flat** - `min J` grows
    +0.142 (pivot) against +1.055 (dense), with dense reaching **-40.92** at
    s=2048;
  * readouts that are FLAT do **not separate** - `frac J < 0` and
    `neg mass/total` sit at **0.5001 for BOTH arms** at s=2048;
  * and every one reads **F(|A|) = 0.0000 for a THEOREM reason** - the min of a
    non-negative matrix is 0, the negative fraction of a non-negative matrix is
    0 - so M2'' "passes" trivially, not informatively.

**CONCLUSION: no readout was found that carries all three.** The project's
persistence claim therefore rests on **the flip rate alone**, and the flip rate
is sign-sensitive only because `I + A + A^2` is non-negative entrywise for
non-negative A - a THEOREM, not a discovery. That is the property SimA
(2206.08898), SDA (2606.04833), SignGT (2310.11025) and Cog (2411.07176)
already have, and the one that **a single GELU between two softmax layers
restores** (`test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`).

**TWO RESULTS WORTH KEEPING FROM A NEGATIVE ITERATION.**

1. **`frac J < 0` = 0.5001 at s=2048 for BOTH arms, flat from s=32.** The
   operator's signs are balanced - independently consistent with the Zaslavsky
   frustration index of **0.5056** measured on `tgate` in Round 1 by a
   completely different instrument. Two unrelated measurements agreeing on
   "balanced signs" is a genuine cross-check.

2. **The pivot-vs-dense separation lives entirely in MAGNITUDE GROWTH**
   (+0.142 against +1.055, dense hitting -40.9). That is the O(s*g) row-L1
   growth on record, seen through the Jacobian. It is a **magnitude** story, so
   by M2'''s own logic it is the kind of criterion that would read the same on
   |A| once measured with a statistic that is not floored at zero - which is
   exactly how all four M2' routes died.

**WHAT THIS DOES NOT SAY.** It does not say the pivot flatness is false - that
measurement stands, journalled and replay-verified 11x. It says the flatness is
carried by ONE readout whose sign-sensitivity is definitional, and that no
second, independent readout was found to corroborate it. A claim resting on a
single theorem-driven readout is thinner than the record has been treating it.

CHECKLIST: no status changed. M2'' remains a calibrated instrument with no
readout yet passing it non-trivially.

### ITERATION 14 - 2026-08-25 - M3: SOFTMAX BASELINE RECORDED FIRST. It FAILS the absolute bar.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): ran M3's softmax arm ALONE and journalled it **before any pivot
number exists**, which is M3's own protocol requirement ("softmax failure
distance recorded FIRST"). Artifact `results/m3_capability.txt`,
timestamp 2026-08-25 15:47:05.

**[RUN] BAR CALIBRATION - passes at both ends, so the instrument is sound.**

    predict_the_mean  NRMSE **1.000000**   (must be exactly 1)
    payload_only      NRMSE   1.398922
    oracle            NRMSE **0.000000**   (must be exactly 0)
    BAR CALIBRATED

    data: train n=512 seed=0, eval n=256 seed=12345 (DIFFERENT seed),
          flipper@103 payload@126, same batches reused across every arm

**[RUN] SOFTMAX, 4769 params, 300 steps.**

    RED  0-step   train NRMSE 1.003200   eval NRMSE 1.004525   [OK]
    POST 300      train NRMSE **0.531924**  eval NRMSE **1.725106**
    eval bootstrap CI (n_boot=400) = **[1.374256, 2.103521]**
    wall clock 65.192 s

**SOFTMAX IS AT 1.725, WELL ABOVE THE 1.0 BAR - WORSE THAN PREDICTING THE
MEAN.** Train 0.532 against eval 1.725 is memorisation without generalisation,
at 4769 parameters on 512 examples. The RED-first 0-step check reads 1.0032 /
1.0045, i.e. an untrained model sits exactly at the mean-predictor - so the
harness is measuring what it claims.

**THIS IS THE W4 DEATH RECURRING**, and the record already names it: W4's
absolute bar FAILED with all arms above 1.0 = worse than predict-the-mean.
Same shape, new task.

**THE CONSEQUENCE, stated BEFORE spending pivot runs on it.** M3's kill is
written against the PIVOT arm ("pivot arm above NRMSE 1.0"). But the BASELINE is
already above it. Two readings, and they need separating before more compute:

  1. **the bar is unreachable at this scale** - 4769 params, 300 steps, 512
     train examples. Then M3 is **a termination clause wearing a gate's
     clothes**, which is precisely the objection CAMERON raised against the
     300M gate ("a gate whose only outcome is OOM is a termination clause
     wearing a gate's clothes"), and it would fail every arm identically
     regardless of the operator;
  2. **the bar is reachable but needs more budget** - in which case softmax's
     1.725 is a budget artifact and the comparison is not yet meaningful.

**Running the pivot arms now would produce numbers that cannot be interpreted
either way**, so the next action is to establish whether ANY arm can get below
1.0 at ANY budget on this task. That is the same discipline as "check the kill
can fire", applied to a bar instead of a kill - and it has caught something in
every route it has been applied to.

**WHAT IS ALREADY CREDITED, and it is not nothing.** The M3 harness itself is
calibrated at both ends by construction: `predict_the_mean` reads exactly
1.000000 and `oracle` reads exactly 0.000000. Those are theorems about the
metric, not measurements, which is the same standard that made M2'''s
known-negative trustworthy in iteration 12. **The instrument is good; the
question is whether the task is winnable at this size.**

CHECKLIST: **M3 UNTESTED - softmax baseline recorded (1.725106, CI [1.374,
2.104]) and it fails the absolute bar.** No pivot arm has been run.

### ITERATION 15 - 2026-08-25 - HEALTH INSPECTOR PASS. 4/4 CLEAN, 0 struck.

Mandatory every 5th iteration per LOOP_PROMPT.md.

**CHECK 1 - calibration [RUN].** `run_calib.py --self-test` -> exit 0. Gate
rejected its deliberately-wrong target first, then 4/4 bit-identical. **CLEAN.**

**CHECK 4 - LOCK against the ARCHIVED copy [RUN].**

    LOCK M2 efadc390c93f -> efadc390c93f   archived==live: True   **CLEAN**

Verified against `results/m2_item_text.txt`, not against a slice boundary that
moves when neighbouring items are appended - and **two items have been appended
since iteration 1** (M2' and M2''), which is exactly the situation that made the
naive check cry wolf. The archived-copy method held.

**CHECKS 2+3 - replay a journalled unit, bitwise, from a journal NEVER
REPLAYED BEFORE [RUN].** Previous passes used `m2` and `s2`; this one used
**`r2`** (36 units, CHASE's, written in this round):

    r2_units exposes units()/compute(): True
    replay tgate/s8: **BITWISE MATCH**

Choosing an un-replayed journal each pass is deliberate - replaying `m2` a
fourth time exercises one code path repeatedly and audits nothing new.

**AUDIT VERDICT: 4/4 CLEAN, 0 struck.** No claim leaves the verdict. Three
independent journals (`m2`, `s2`, `r2`) have now each been shown to reproduce
bitwise from a fresh process.

**ONE THING THE PASS SURFACED, recorded not acted on.** `r2`'s journalled values
include `coh_dense` and `coh_pivot` - the path-coherence numbers that were RED
in iteration 7. At `tgate/s8` they are **identical to 16 digits**
(0.7044318334094577 both), then diverge: s=32 gives 0.3550 dense against 0.6995
pivot, s=128 gives 0.2675 against 0.7556. **The identity at s=8 is expected and
is a correctness signal** - at s=8 nearly every token is a pivot, so routed and
dense hop-2 are the same computation, which is the same structural reason the
M2 claim arm and its dense control read identically at s=8 (0.024658 both). Two
unrelated instruments agreeing on that boundary condition is a cross-check
neither was designed to provide.

CHECKLIST: no status changed.

## CAMERON — R5 OUTCOME AGAINST `LOCK R5 d8491f67bf6b`

Scored against the kill exactly as frozen above, before any R5 number existed.

**K1 — `pivot_signed` held-out NRMSE ≥ 1.0 at d = 256 → RED. FIRES, on every
configuration that has run.** 21 logged runs at `n_train=128`
(`results/m3_capability.txt`, s = 64/80/112/128/160, d = 24..54, steps 150,
seeds 0 and 1) plus one data-repaired run at `n_train=1024`, s=64: **no arm ever
got below 1.0 on held-out.** Representative (s=128, d=42, seed 1):

| arm | params | 0-step eval | train | held-out | bootstrap CI |
|---|---|---|---|---|---|
| softmax | 4769 | 1.007122 | 0.212803 | **1.723800** | [1.507765, 1.995668] |
| pivot_signed | 4930 | 1.012668 | 0.039843 | **1.184561** | [1.092116, 1.293018] |
| pivot_unsigned | 4769 | 1.007125 | 0.234064 | **1.663397** | [1.485817, 1.887009] |

**K2 — softmax beats `pivot_signed` → RED. DOES NOT FIRE.** `pivot_signed` beats
softmax with non-overlapping CIs in every run.

**K3 — `pivot_unsigned` matches `pivot_signed` → G4. DOES NOT FIRE.**
`pivot_unsigned` tracks *softmax*, not `pivot_signed`. This is the S2
decomposition M2 structurally could not produce, and it is evaluable here
because the readout is on the x-path.

**K4 — NaN/inf → RED. Correctly implemented** at `scale/m3_capability.py:172-176`,
checked before any threshold comparison. No NaN occurred.

**RED-FIRST FIRED, which is the part that matters most.** Every arm read ≥ 1.0
at 0 steps in every run (e.g. 1.007122 / 1.012668 / 1.007125 above). Unlike M2
clause 2 and unlike the M4 kill, this kill's quantity demonstrably reaches both
sides of its threshold.

**NOT ESTABLISHED, and must not be assumed:** no arm has completed at
d ≥ 256, and the pivot arms have not completed at `n_train=1024` — the forward
pass runs an `n`-iteration Python loop (`m3_capability.py:100-107`) that must be
vectorised before the route is costed honestly.

**Verdict: R5 is PARTIALLY RUN. Not GREEN, not RED.** What has run says the arms
do not clear the absolute bar, while producing the cleanest arm ordering this
project has ever had. Both halves get reported or neither does.
