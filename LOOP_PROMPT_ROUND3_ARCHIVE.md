# CEQ v5 — ROUND 3 LOOP PROMPT. Read in full, every iteration, follow exactly.

**30 iterations. Promise word `SCALEFREE`.** Supersedes v4, which the room
amended before a line was built. Round 2's prompt is archived at
`LOOP_PROMPT_ROUND2_ARCHIVE.md`. `CONTRACT.md` holds the arsenal; this file
outranks it; `CHECKLIST.md` outranks both.

## WHAT THE ROOM CHANGED, AND WHY IT IS BINDING

v4 scheduled **R8 first, as the cheapest one-line change.** Foreman and Chase
independently asked which route changes the **EVENT** rather than the
**STATISTIC**. Measured, 20,000 draws:

    common positive rescale (R5, R8)   event changed in      0 / 20000
    ECDF rank transform      (R7)      event changed in    723 / 20000  (3.6%)

**R5 and R8 are provable no-ops and are struck before build.** This repository
already proved it once as instrument #16 — *a positive elementwise rescale
cannot change a sign* — which is part of why R4 died. Building R8 would have
produced a guaranteed null and read as evidence.

**A published claim is withdrawn.** Foreman demanded the interval on the
comparison that declared pivot routing dead. Bootstrap B=20000:
**pivot − dense = −0.2099, 95% CI [−0.7497, +0.2651] — does not exclude zero.**
*"Routing makes it worse than dense"* is unsupported and appears in every
document since round 2. Pivot routing stays dead (both arms far past −0.3);
that sentence does not.

## ORDER OF WORK — M3 FIRST. THIS IS THE ROOM'S MAIN AMENDMENT.

**Phase 0 (iterations 1–4): CASH THE ARM ON THE SHELF.**
F4's windowed w=8 signed multi-hop is **already flat to s=2048** — the exact
property four routes are chasing — and has **never been capability-tested**. It
is a live counterexample to the sufficiency of the M2″ gate, sitting inside the
facts table. Two rounds built statistics first and produced zero capability.

Run **M3 on the windowed arm**: negation-scope flip at d ∈ {256, 512, 1024},
executable oracle with no answer key in the corpus file, 5 seeds, matched params,
matched lr sweep, **softmax measured and timestamped FIRST** in `results/`.

Before the first reading, **M3's kill must be seen firing on a deliberately
broken arm** (Chase). A criterion never observed failing is a rule, not a test.

**Outcome A — windowed M3 GREEN:** the capability exists in the flat regime.
Everything after is about extending reach, and the round has a result.
**Outcome B — windowed M3 RED:** flatness does not produce capability. **The
M2″ gate is then void as a proxy** and R6/R7 must not be built on it — say so
and re-scope. This is the cheapest possible test of the round's own premise.

**Phase 1 (5–18): R7, and R6 only if R7 dies.**
**Phase 2 (19–26):** M6 guard on the survivor, then M7 trained reading at 25.7M.
**Phase 3 (27–30):** write-up, honest limits, HF package.

## R7 — RANK / COPULA AGGREGATION, with the corrected pre-registration

    t_ij = g_i · tanh((q̂_i · k̂_j)/τ),  j < i
    ã_ij = sign(t_ij) · F_i(|t_ij|)^γ            F_i = causal ECDF of row i
    P_i  = top-k of F_i(|t_ij|)                  selection unchanged
    out_i = v_i + Σ_j ã_ij v_j + Σ_{p∈P_i} ã_ip Σ_{j<p} ã_pj v_j

**The ratio kill from the fable run is REPLACED — it tested a finite-size
correction at s₀, not the mechanism.** All of its numbers are one formula at
different boundaries: ratio from s₀ = `(1 − γ(k+1)/(2(s₀+1)))⁻¹` reproduces
2.000 / 1.636 / 1.360 against measured 1.996 / 1.633 / 1.357.

**The corrected kill, pre-registered [DERIVED, verified this session]:**

- `A_8 = 2.187500` **exactly**, by 256-term sign enumeration. No Khintchine slack.
- `E|B_k(s)| ↑ A_k` monotonically — a theorem, not a trend: `m_r(s) ↑ 1` and
  `|B_k| ≤ k` always, so dominated convergence applies. Verified:
  **0.7476 / 0.9653 / 0.9912 / 0.9978 / 1.00000** at s = 16/128/512/2048/2²⁰.
- **KILL 1 (sup bound):** measured `E|B_k(s)|` exceeds `A_k·(1+ε_Jensen)` at any s.
- **KILL 2 (terminal slope):** local log-log slope on **s=512→2048 alone** with
  `|slope| ≥ 0.01`. Predicted **−0.00476**; the dense arm at −1.088 cannot pass.
  This is the discriminating test — the −0.1246 slope over 8→2048 is a
  **transient**, and a bar fitted across the transient would pass arms that
  should fail.
- **γ admissible range is a formula, not a sweep:** the sup-ratio crosses 1.9 at
  **γ = 1.789** at (s₀,k) = (16,8). Above that the kill can fire from geometry alone.
- **Baseline is s=16, never s=8.** At s=8 `select_pivots` excludes i and j, so
  |P| = 6 not 8 — a documented degenerate point (round 2, iteration 23).
- **Sign-independence is TESTED, not assumed.** If sign–rank dependence is found,
  re-derive `A_k` under the measured sign copula; boundedness survives
  unconditionally since `|B_k| ≤ k` is sign-free — only the constant moves.

## THE TWIN TEST IS REBUILT — v4's WAS VACUOUS

Foreman: in `ã = sign(t)·F(|t|)^γ` the sign is a **multiplicative prefactor**, so
sign-sensitivity is a property of the algebra, not the aggregation.
**`ã = sign(t)·1` — pure sign, all magnitude destroyed, plainly useless — passes
the v4 twin test identically.** It certified formula shapes.

The rebuilt test has **three** arms and the control must be seen losing:

    scale-insensitive   monotone magnitude randomization must not change the decision
    sign-sensitive      sign randomization MUST change it
    NOT-VACUOUS         `sign(t)·1` must score STRICTLY WORSE than `sign(t)·F^γ`
                        on the same draws — if it ties, the magnitude channel
                        carries nothing and R7 is sign(t) in a costume

## OPEN, AND IT IS THE DEEPEST THING HERE (Cameron)

F1 says softmax sits at exactly `0.000000` on negation **by theorem**. The only
matched-parameter capability measurement says softmax **15/512** on COGS against
this operator's **0/512**, p = 2.75e-05, and behind in-distribution too (0.7734
vs 0.9258). **Both numbers cannot be about the same thing.** Nothing in two
rounds resolves it. Any iteration that can design the experiment separating them
should take it over the scheduled work and say why.

Related and unanswered (Foreman, Cameron): **no result in either round shows a
change in flip-rate moving any capability number in either direction.** F7
concedes val-loss under 3% does not predict capability; the same standard has
never been turned on flip-slope. Phase 0 is the first honest test of it.

## STANDING POLICY - NEVER BLOCK ON A MEASUREMENT

**If a measurement is running, something is being BUILT alongside it.** An
iteration that spends its wall clock watching a probe finish has spent it.

**WILSON MANAGES THE NURSES.** The nurses are ENGINEERS - inference engineers and
senior compiler engineers - and they always have something to optimise. Wilson
has no stance and no angle, which is exactly why he owns them: he assigns
mechanical work and judges what came back by whether it is verifiably true, not
by whether it is interesting.

    long measurement launched   ->   Wilson + nurses dispatched in the SAME turn
    measurement lands           ->   reconcile both, record both

**Nurse work is engineering, not opinion.** Vectorise a Python loop over a batch;
kill a quadratic; cache what is recomputed; fix an instrument that reports FAIL
where the honest verdict is INDETERMINATE. Every optimisation ships a **bitwise
equivalence bind** against the implementation it replaces - a faster path that
changes a number is not an optimisation, it is a new arm, and this repository has
published one of those before.

**Declared cheats.** A nurse that subsamples, caches, or approximates DECLARES it
with its cost. An undeclared shortcut is a fabricated result.

## INSTRUMENT LAW — nine failures, three survivors

**COMPARE VALUES, NEVER STRUCTURE.** Nine structure-comparing instruments (regex,
slices, substrings, shell exit codes) all gave false readings; three
value-comparing ones never have.

- **No decision downstream of a shell pipeline.** `$?` after a pipe is the last
  stage's. `| tail`, `| head`, `| grep`, `| wc` mask identically. Iteration 40
  wrote this rule; iteration 45 broke it anyway. **`inspector.py` is the pattern:
  run it, do not retype it.**
- **Every checker ships a must-fire control SEEN to fire.** A green whose control
  stayed silent is blind, and that exits nonzero.
- **Pin exact values at `abs=5e-7`. Inequalities protect nothing** — a fabricated
  magnitude survived fourteen passing tests because the assertion was `> 1e-3`.
- **Every zero carries its CP interval and its floor-discard count.** The 1e-6
  floor discards 3/44 → 10/19 → 2/2 as s grows. Report `floor=0` beside every
  floored arm; it is a **new arm**, never a replacement (G2 stands).
- **The journal replays bitwise only at `OMP_NUM_THREADS=2`** — stated in the
  journal header, pinned in `inspector.py::JOURNAL_THREADS`.
- **G3:** every new arm ships the `hops=0` bitwise-identity RED bind before its
  first reading.

## G1 — FETCH BEFORE BUILD, AND ONE IS OWED NOW

**TACTiS (Drouin et al., ~ICML 2022)** — transformer-attentional copulas for time
series — must be fetched **before the name "Copula Attention" is written
anywhere**. Believed to use copulas as the *output distribution* rather than the
*attention aggregation*, which is a different cell, but that must be established
by fetch. Asserting it from memory is the ParaFormer costume again.

Also owed: soft ranks / Blondel–Cuturi (R7's differentiable surrogate — **known
sign-blind in native form**, so the rebuilt twin test runs on the soft arm before
any training number is believed); Chernoff–Savage and van der Waerden scores
`Φ⁻¹(F̂)` (a second R7 arm, one line of delta, scores growing like `√(2 log s)` —
matched to the promotion rate they must cancel); DKW/Massart (the causal ECDF at
position i estimates F from i samples, so early positions are noisy and DKW
bounds it in closed form — every probe reading carries a finite-sample band).

## THE TRAINING CORNER, FLAGGED NOW

**The ECDF is piecewise constant — gradient zero almost everywhere.** R7 is not
trainable as written. The soft surrogate is the Blondel–Cuturi lineage, which G1
marks sign-blind natively. **Any iteration that reports an R7 training number
without the rebuilt twin test having passed on the soft arm has reported
nothing.**

## GOVERNANCE

One iteration = exactly ONE of: write a RED test / turn one RED test GREEN by the
minimum change / repair one instrument with proof / one G1 fetch / write-up. Then
update `STATE.md`, `DONE.md`, `CHECKLIST.md` status column, and stop.

Any MANDATORY item RED stops build work; legal moves are instrument repair
(RED-first on a synthetic with known ground truth), G1 fetch, or write-up. **A RED
is overturned only by convicting the INSTRUMENT, never by adjusting the arm.**

Every 5th iteration: run `python inspector.py`. Any failure replaces that
iteration's plan with the repair.

## COMPLETION — line 1 of `DONE.md`

`SCALEFREE: KEPT` requires **M3 GREEN with softmax timestamped first in the same
table**, one route GREEN by its own kill, the rebuilt twin test passed including
the not-vacuous arm, and a trained checkpoint at ≥ 25.7M with the 300M gate
stated. **A slope is not a capability. KEPT on a statistic is a false promise.**

`SCALEFREE: BROKEN — <item> <which kill fired>` after the write-up with its
numbers is complete. **An honest BROKEN outranks an unfinished KEPT.**

Emit `<promise>SCALEFREE</promise>` only when one is completely true.
