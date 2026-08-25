# CEQ v6′ — ROUND 4 LOOP PROMPT. The chosen-sign round. Read in full, follow exactly.

**Promise word `CHOSENSIGN`.** Supersedes v5, archived at
`LOOP_PROMPT_ROUND3_ARCHIVE.md`. Rounds 1–3 facts carry over.
`CHECKLIST.md` outranks this file; this file outranks `CONTRACT.md`.

**THE GOAL, restated by the user and narrower than "attention that works":** not
the best token predictor — the **next-equilibrium predictor**. Attention that
understands causality and consequences on Turing-grade problems at the smallest
scale, and **the module must survive equal to self-attention or supersede it.**

## THE ALGEBRA THIS ROUND EXECUTES

    flip(s) = P[ |t_c| > |Σ_{p≠c} ε_p t_p| ]

Rounds 1–3 attacked **`t_p`** — the term count, the scale, the rank. All three
died. **Round 4 attacks `ε`.**

- **X₁** reachability holes are a *counting necessity*, cured exactly by
  (v,k,λ)-difference sets. Co-prime dilation was the empirical shadow of this.
- **X₂** signs may be **CHOSEN**: `|Σ ε t| ≤ 6√k` (Spencer [U]).
- **X₃** selection-coupled signs **MUST** align (FKG [V]).
- **X₄** the instrument must compare **valuations**, not floats (Levi-Civita [V]).

**X₃ RETRO-EXPLAINS ROUND 3's OWN MEASUREMENT AS A THEOREM.** Iteration 6 read
the top-k selected set at **99.98% positive, uncentered pairwise product
0.9993**, and attributed it to the operator's `λ`. FKG says it is forced by the
coupling between selection and sign. It was never an accident.

**X₄ EXPLAINS THE FLOOR.** At depth, median `|grad|` is **2.8e-32** and
`floor = 1e-6` discarded **100%** of a live arm's flips. A float comparison
cannot straddle 30 orders of magnitude.

## ROUND-3 FACTS THAT BIND THIS ROUND

**F16 — THE SIGNED ARM WAS NOT SIGNED, AND THIS IS THE ROUND'S ENTRY POINT.**
At the harness geometry (`make_batch` scales `x` by 0.1; logits `|w|` mean
**2.682399e-03**), `_causal_sgate_operator(lam=0.10)` is **entrywise
non-negative** — min entry exactly `0.000e+00`, bulk negative fraction
`0.000000`. **λ is a THRESHOLD at 1.0, not a dial**: at `w ≈ 0`, `pp ≈ pm`, so
`A ∝ (1−λ)/i` stays positive until λ crosses 1, where row sums are exactly zero
(absmax 2.403e-07). **Every prior signed-vs-unsigned comparison at this geometry
compared two non-negative operators and is void as sign evidence.**

**F17 — EVERY OPERATOR AUDIT MUST STATE ITS LOGIT SCALE.** Round 3's frustration
audit read **0.2779** on unit-scale `torch.randn` projections and **0.000000** at
harness scale. Same operator, opposite answers. **A sign measurement without its
logit scale is not a measurement.**

**F18 — ROUTING BEATS SOFTMAX, and that comparison SURVIVES F16** because
`pivot_unsigned` was always the non-negative arm. At `n_train=8192`, 4769 params
each, softmax first: softmax **0.877168** [0.830455, 0.924226], pivot_unsigned
**0.747528** [0.696849, 0.797716] — **disjoint**.

**F19 — THE BUDGET IS THE BAR.** No arm had ever passed the absolute bar in three
rounds because every reading was at `n_train=128`, where 4769 params memorise:
`128 → 2.116579`, `512 → 1.316514`, `2048 → 0.949529`, `8192 → 0.877168`.
**M3 readings below n_train=8192 rank overfitting, not capability.**

**F20 — CO-PRIME DILATION.** `[1,3,5,7]` cuts severance `0.5745 → 0.1277` at
s=128 with gradient support unchanged at exactly **1.0000**; power-of-two severs
41–57%. Severance is **INERT, never UNREACHED** — an influence defect, not a
reach defect. Identical under weight sharing (0.065217 / 0.127660).

**F21 — THE BATCHED PATH IS FREE.** `batched_pivot_hop2` / `batched_select_pivots`
are **bitwise** equal to the loop (`torch.equal`, forward 12 cases, gradients
maxdiff 0.0 at n=8) and **374×** faster on forward+backward at n=2048.
Declared limit: gradients bound only to n ≤ 64; at 2048/8192 the bind is
forward-only.

**F22 — THE BAR IS REPAIRED.** `calibrate_bar` + `bar_verdict` carry
`flipper_dependence` (exactly **2.0** real / **0.0** flipper-blind) and a
**trained** two-feature control (**0.047149**). Two of the old three checks were
algebraic identities. One gate, one copy.

## ARMS — TWO. NO THIRD WITHOUT AN AMENDMENT NAMING ITS X.

**A. DIFFERENCE-SET SCHEDULE (solves X₁).** Hop offsets = cyclic Singer
(v,k,1)-difference set, `k ≈ √s`.
*Birth gates:* `|D−D| = v−1` asserted as a **VALUE**; flipper placed **uniformly
at random**, never on the schedule; off-schedule flip rate within CI of
on-schedule — **the exact test that killed the dilation, promoted to a birth
gate**. Lean target: the coverage lemma, finite and decidable — this round's M5.
*Kill:* any bitwise-identical gradient pair (severing), OR `|slope| ≥ 0.01` over
s=512→2048 on the X₄ instrument, OR M3 fail at n=8192.

**B. DISCREPANCY-STEERED SIGNS (solves X₂, certified against X₃).** Sign head
carries a discrepancy regulariser minimising `|Σ_{p≠c} ε_p t_p|` over the
background while the task loss owns `ε_c`. Reference assignment from a
constructive discrepancy pass (Lovett–Meka [U]); **Spencer's `6√k` printed beside
the measured background at every s**.
*Birth gates (M8):* `P(+) ∈ [0.35, 0.65]`, pairwise corr `< 0.2`, measured
`E|Σ ε t| ≤ c√k` with `c` pinned; **FKG-escape documented** — the sign source must
be decoupled from the salience order, stated and tested (shuffle the selection
order; **signs must not move**).
*Kill:* trained background exceeds **3×** the Spencer reference, OR the
regulariser degrades task loss past the **1.10** tuning bar, OR M3 fail at n=8192.

## INSTRUMENT UPGRADE (X₄) — PRECEDES BOTH ARMS

`sign_flip` rebuilt on **(valuation, mantissa)** pairs. **The floor is DELETED,
not defaulted.** Calibration: reproduces every published `floor=0` number
exactly. Must-fire: a planted **30-order-spread** arm must be read correctly
where the float instrument **provably misreads it**. All prior floored numbers
remain quoted as **historical instrument readings**.

## CHECKLIST DELTA

- **M2‴** measured **only** on the X₄ instrument.
- **M3** unchanged: `n_train=8192`, **softmax pass reproduced first**, 5 seeds,
  unsigned ablation, CIs excluding zero.
- **M5** = the difference-set coverage lemma in Lean — this round's proof
  deliverable, replacing the N=n mismatch item.
- **M8** = the FKG-escape certificate, replacing the bare diversity gate.
- **G1 fetches owed PRE-BUILD:** LongNet + the severing test (D1b stands);
  difference-set / Sidon attention; discrepancy-in-ML (**KNOWN NEAR-MISS:
  herding / kernel herding [U] uses discrepancy for SAMPLE selection — the cell
  here is SIGN assignment inside the operator; establish the distinction by
  fetch**); balanced-colouring networks.
- **G7** event-change **≥ 1%** pre-build, both arms — the test that struck R5/R8
  at 0/20000 before they cost anything.

## DELIVERABLES

- **D1** (unconditional): the negative result, the instrument taxonomy, the Lean
  core, plus the round-3 chapter and the X₁–X₄ algebra.
- **D1b**: the LongNet severing note if the test fires.
- **D2** (iff birth gates + M2‴ + M3): *"coverage by difference set [theorem],
  background bounded by chosen-sign discrepancy [Spencer], measured flat on a
  valuation instrument, capability at the budget where softmax passes"* — four
  clauses, each with its own kill already survived.

## STANDING POLICY — NEVER BLOCK ON A MEASUREMENT

**If a measurement is running, something is being BUILT alongside it.**
**WILSON MANAGES THE NURSES** — inference engineers and senior compiler
engineers, who always have something to optimise. Long measurement launched →
Wilson + nurses dispatched in the **same turn**. Every optimisation ships a
**bitwise equivalence bind**; a faster path that changes a number is a **new
arm**, not an optimisation. Declared cheats only.

## INSTRUMENT LAW — nine structure failures, three value survivors

**COMPARE VALUES, NEVER STRUCTURE.** No decision downstream of a shell pipeline
(`$?` after a pipe is the last stage's). Every checker ships a must-fire control
**seen to fire**. Pin exact values at `abs=5e-7` — **inequalities protect
nothing**. Every zero carries its CP interval and its floor-discard count. The
**"journal replays bitwise at `OMP_NUM_THREADS=2`" is WITHDRAWN as unproven [RUN, r4 iter 5]** — generalised from ONE unit in round 2. Measured on `dense_signed__at_pivots/s1024/b0`: MATCH at 1 and 4 threads, **DRIFT at 2**, exactly backwards. `rate` (integer, and what every published number rests on) matches at every count; only `sigma`/`term` drift. **No single count is KNOWN to replay the whole journal.** Census owed, bucketed. Long runs go through
`scale/bucket.py` — **ADR-001 was bypassed once and cost two 0-byte files.**
`inspector.py` is the pattern: run it, do not retype it. It is now **tri-state** —
an INDETERMINATE check exits nonzero, because an unmeasured check is not a clean
one.

## GOVERNANCE

One iteration = exactly ONE of: write a RED test / turn one RED test GREEN by the
minimum change / repair one instrument with proof / one G1 fetch / write-up. Then
update `STATE.md`, `DONE.md`, `CHECKLIST.md` status, and stop. Any MANDATORY item
RED stops build work. **A RED is overturned only by convicting the INSTRUMENT,
never by adjusting the arm.** Every 5th iteration: `python inspector.py`.

## COMPLETION — line 1 of `DONE.md`

`CHOSENSIGN: KEPT` requires **M3 GREEN at n_train=8192 with softmax reproduced
first in the same table, 5 seeds, the unsigned ablation, and CIs excluding
zero**, plus one arm's birth gates passed and M2‴ on the X₄ instrument.
**A slope is not a capability. KEPT on a statistic is a false promise.**

`CHOSENSIGN: BROKEN — <item> <which kill fired>` after the write-up with its
numbers. **An honest BROKEN outranks an unfinished KEPT.**

Emit `<promise>CHOSENSIGN</promise>` only when one is completely true.
