# CEQ-NOVELTY CHECKLIST — WORK-STOPPING CLAUSE

Status of every item is one of: UNTESTED / GREEN / RED.
Any MANDATORY item RED after its test runs => ALL BUILD WORK STOPS.
Only instrument repair, prior-art search, or write-up may continue.
**A checklist item may never be edited after its test has run.**

## PREAMBLE — what counts

A capability counts only if (a) baseline softmax attention PROVABLY or
MEASURABLY cannot do it, with the baseline number in the same table,
(b) it survives its own pre-registered kill number, (c) it is measured
on a calibrated instrument whose published numbers still reproduce
(nine instruments in this project were internally consistent and
externally wrong — calibration is not optional), and (d) prior art has
been swept by direct fetch, not memory, before the claim is written.
Taxonomy cells count for nothing. Val-loss deltas under ~3% count for
nothing (arXiv:2605.20798). Statistics are not capabilities.

---

## MANDATORY — the module must do ALL of these; softmax can do NONE

| id | item | status |
|---|---|---|
| M1 | Negative influence on the value path | **GREEN, SCOPED [RUN, r3 iter 8]** — the theorem holds on a LINEAR value path and softmax reads exactly **0.0 at depth 1 AND 2**. But every shipped block has `nn.Linear→nn.GELU→nn.Linear` (`ceq/lm.py:214`, four of them), and with that nonlinearity **softmax_gelu depth=2 reads 0.0546875** while **sgate depth=2 reads 0.09375** — **1.71×, not exclusivity**. sgate gets WORSE with depth (0.1484375→0.09375). The 0.0546875 has been asserted at `tests/cameron/test_parity_is_the_wrong_target.py:61` since round 1 and was never carried into the capability framing. |
| M2 | Context-stable signed influence at global reach | **RED** — MEASURED on the shipped operator: sgate-pivot slope **-1.298** vs bar **-0.3**, and **0 flips at s>=128**. The +0.0270 / 61x numbers were `tgate`, which ships nowhere (instrument #17). **SCOPE [RUN, iter 39]: measured at RANDOM INITIALIZATION** — `pivot_probe.py:145` draws q, k and the gate from `torch.randn`; no trained weights are loaded anywhere in `scale/`. **INSTRUMENT-CLEARED [RUN, iter 42] and the RED is now PERMANENT**: at `floor = 0` the slope is **−1.115** (R² 0.9350) — the kill still fires with the magnitude gate removed, and the same instrument reads −0.034 for `tgate` vs −1.298 for `sgate`, so it is not reading a constant. Per `M2_TRAINED_PREREGISTERED_READING.md`, written before the test, **no trained number can overturn this**. |
| M2' | Signed influence that context cannot dilute (4 routes) | **RED** — all four routes SIGN-BLIND (twin test F(A) vs F(\|A\|)); **G1 FIRES for R3** — Tropical Attention arXiv:2505.17190 is R3, published |
| M2'' | Sign-blind twin differential — F(A), F(\|A\|), F(softmax) in one table | UNTESTED — **instrument CALIBRATED 2/2 ends** (theorem anchors gap 0.000e+00; must-fire gap 4.777e-01); no readout judged yet |
| M3 | Long-range sign capability, absolute bar | UNTESTED — softmax first: **1.725106** (n=512) -> **1.304590** (n=2048); **bar REACHABLE** (learnability control 0.0071, routing free) |
| M4 | Exact eviction | UNTESTED — **kill now EVALUABLE**: must-fire arm added (kept-token perturbation moves 8/8); crushed-token 0.0 re-labelled **structural**; gating moves 8/8 under BOTH placements |
| M5 | Certified finite computation | **RED** — `occupancy_is_exact_inverse` is stated at **N=n**; the module truncates at **hops=2..4**, where **||A^hops||=0.880500 at s=128**, not 0 (value now PINNED by test; the 1.47 first printed here was struck by the iteration-35 audit). `A^n=0` IS confirmed. The theorems are sound; they certify a computation the module does not perform; violation now **pinned by test** (`test_theorem_hypotheses_hold_at_shipped_settings.py`, 14 passed). **SCOPE [RUN, iter 39]: measured at RANDOM INITIALIZATION**, same as M2. |
| M6 | Denominator-free magnitude certificate | UNTESTED |

**M1. NEGATIVE INFLUENCE ON THE VALUE PATH**
Claim: influence Jacobian d(out_i)/d(v_j) reaches negative values.
Baseline impossibility: measured +0.000000e+00 exactly for softmax, APPNP,
max-plus star; semiring theorem — non-negative operators cannot represent
negation.
Test: existing Jacobian probe, both devices.
Kill: min influence >= 0, OR the most negative entry is a frozen zero-gradient
cell (the -rho at (1,0) trap from Foreman Q3).
KNOWN RISK: SignGT/SDA/Cog also pass M1. M1 alone is NOT novelty — it is a
precondition. Novelty lives in M2xM3.

**M2. CONTEXT-STABLE SIGNED INFLUENCE AT GLOBAL REACH**
Claim: sign-flip rate for pivot-routed tokens is flat in s while reach stays
global (no window).
Baseline impossibility: softmax exactly 0 at every s; every dense signed arm
decays s^-1.1..-1.7; windowed arms buy flatness only by surrendering reach.
Test: bench.py pivot arm, stream-isolated, s = 8..2048, 16,384 draws at the
tail, c in P vs c not-in P vs windowed placement.
Kill: slope(c in P) < -0.3, OR c not-in P is ALSO flat (mechanism story false
even if the number looks good), OR any previously published bench.py number
moves.

**M2'. SIGNED INFLUENCE THAT CONTEXT CANNOT DILUTE (route-dependent)**
Naive flatness of a flip PROBABILITY at global reach is impossible
(Littlewood-Offord lower bounds). Four escapes, each breaking one hypothesis of
the impossibility; ANY ONE GREEN suffices. Test in cost order 2 -> 4 -> 1 -> 3:
  R2: sign-determinacy (breaks "magnitude statistic") -- enforce a
      sign-nonsingular pattern on the k x k pivot block; test is
      magnitude-randomization invariance, 10^4 resamples, sign pattern
      bitwise constant. Kill: determined fraction ~0 in trained blocks,
      or the constraint destroys training.
  R4: hierarchical criticality (breaks "flat aggregation") -- Dyson-tree
      aggregation, level coupling 2^(-theta*l); slope in s is a function
      of theta. Kill: no zero crossing of slope(theta) in range, or
      theta* unstable over 3 seeds.
  R1: certified selection (breaks "generic token") -- group-testing /
      d-disjunct decoder recovers the k causal tokens with combinatorial
      guarantee at polylog overhead; post-recovery background has k
      terms. Kill: recall < 1-eps at s=2048, or overhead exceeds the
      stated polylog envelope.
  R3: non-Archimedean routing (breaks "Archimedean sum") -- symmetrized
      max-plus / valuation aggregation; minimal-valuation term dominates
      independent of s; Newton-polygon certificate of dominance. Kill:
      annealed training > 1.10 of softmax, or balanced-ambiguity
      decisions > 10%.
Standing clauses inherited: stream isolation; published bench.py numbers
immutable; softmax baseline in every table; any route GREEN must still convert
to M3's capability or it is a statistic.

---

**M2 IS SUPERSEDED, NOT PASSED -- the defect report, attached permanently.**

M2's kill had three clauses. Clause 1 (slope(c in P) < -0.3) did NOT fire: the
measured slope is **+0.0270** over a 256x context growth. Clause 3 (G2) did not
fire; calibration held bit-identical throughout. **Clause 2 is UNEVALUABLE BY
CONSTRUCTION** -- hop2[i,j] = sum_{p in P} A[i,p] A[p,j] contains no term with
index c when c is not in P, so that arm is identically zero and cannot be "also
flat" in the sense the clause intends. The verdict code mapped its NaN slope to
"control decays as required": a FALSE GREEN, broken instrument #15.

So M2 is neither GREEN nor RED-by-kill. It is DEFECTIVE, and a claim whose
safeguard cannot be evaluated has not survived its safeguard. It is superseded
by M2', which the contract names as the mandatory item in its place. **The item
text stays frozen under LOCK M2 efadc390c93f and is never edited.** Its
measurements survive and are carried in DONE.md.

**M2''. SIGN-BLIND TWIN DIFFERENTIAL**
For every candidate readout F, publish **F(A), F(|A|), F(softmax) in ONE
table** - the same measurement on the operator, on its entrywise absolute value
(magnitudes bit-identical, signs stripped), and on softmax.
**KILL: if F(A) and F(|A|) agree within intervals, F is SIGN-BLIND and dies.**
Calibration required at BOTH ends before any F is believed:
  known-positive : a readout that IS sign-sensitive must show F(A) != F(|A|)
  known-negative : a readout that is sign-blind must show F(A) == F(|A|)
  softmax column : a THEOREM (A = |A| entrywise), must agree EXACTLY
Standing clauses inherited: stream isolation; published bench.py numbers
immutable; softmax baseline in every table; **a GREEN F must still convert to
M3's capability or it is a statistic.**

**M3. LONG-RANGE SIGN CAPABILITY, ABSOLUTE BAR**
Claim: a downstream task with ground truth — negation-scope flip at distance d
— is solved at d >= 256 where softmax fails.
Baseline impossibility: to be MEASURED, not assumed; softmax arm runs first and
its failure distance is recorded before ours.
Test: new capability corpus (oracle = executable, no answer key in file), 5
seeds, matched params, matched lr sweep.
Kill: pivot arm above NRMSE 1.0 (predict-the-mean — the W4 death), OR CIs
overlap softmax at every d >= 256, OR the unsigned-pivot ablation matches it
(then the routing is the contribution and the claim sentence must be rewritten
before work continues).

**M4. EXACT EVICTION**
Claim: removing a token renormalizes survivors exactly; settled state of
unrelated content moves by 0.0.
Baseline impossibility: post-softmax gating leaks 2.3e-03 row-sum deficit
permanently — the denominator remembers.
Test: W3 instrument, keep-set held fixed across arms, 24 draws.
Kill: max change > 1e-12 in the evict-before-read window; scope MUST state the
W9 limit (eviction after dependents read is not retroactive — claiming
otherwise is RED by dishonesty).

**M5. CERTIFIED FINITE COMPUTATION**
Claim: the multi-hop operator's resolvent is EXACT in finitely many terms,
machine-proved, over the shipped matrix class.
Baseline impossibility: softmax has no finite exact form and no certificate;
even DeltaNet's own codebases state it unproved (0 grep hits for
"nilpotent"/"Neumann").
Test: lake build CEQ exit 0, zero sorry, calibrated sorry-detector; grep-binding
between theorem hypothesis and shipped code (.tril(-1) check) GREEN.
Kill: any hypothesis of the Lean theorem not satisfied by the tensor that
actually ships.

**M6. DENOMINATOR-FREE MAGNITUDE CERTIFICATE**
Claim: hop growth bounded by ||A^h|| <= 2 w(A)^h via numerical radius
w(A) <= rho — one scalar, no sum over s, so nothing for context to dilute.
Baseline impossibility: softmax's magnitude control IS the global denominator —
the measured cause of its own x-path sign decay and sgate's death.
Test: enforce w(A) <= rho in forward; measure row-L1 drift over 800 steps (the
1.5e4x spread is the known hazard) and re-run M2 with the guard on.
Kill: guard reintroduces decay in M2, OR training diverges under the constraint
at every lr in the sweep.

---

## SUPPORTING — valuable, not work-stopping alone

**S1.** Bounded decode: multi-hop at K slots + K rows per token (hopcache),
prefill/decode parity <= 1e-12 relative. Softmax multi-hop has no incremental
form at all.
**S2.** Selection ablation table: pivots+unsigned vs pivots+signed vs
dense+signed — the honest decomposition of WHERE the capability lives. Required
before any claim sentence is published.
**S3.** Derived hop coefficients (Carnot eps^h grading) beat learned scalars
gamma_k at matched params — else learned scalars ship and the geometry is
reported as structure, not advantage.

---

## GLOBAL STOPPING CONDITIONS (any one fires => STOP)

**G1.** Step-0 prior-art sweep (direct fetch: Nystromformer, landmark, NSA,
MoBA, ParaFormer, DeltaNet, SDA, SignGT, Cog) finds content-selected signed
multi-hop routing already published.
**G2.** Any bench.py calibration number moves when a new arm is added.
**G3.** Any instrument found reporting another arm's number under this arm's
name (the ParaFormer bug class) — stop until the RED-first bind exists.
**G4.** M1-M6 all GREEN but S2 shows the capability lives entirely in the
unsigned ablation — stop, rewrite the claim, re-enter at M3.

## THE CLAIM SENTENCE

**May only be written when M1-M6 are GREEN and S2 is done.**

Restated by the user 2026-08-25 and AUTHORITATIVE in this form. It replaces the
prose target recorded earlier the same day; that sentence had no baseline column
and the preamble forbids publishing in that form. This one is measurable
end to end, which is why it is the one that ships:

> Fixing multi-hop path count by content-selected pivot routing rather than
> locality holds signed influence flat in context at global reach and converts it
> into capability X at distance d, with exact eviction and a machine-checked
> finite resolvent -- where softmax attention is at exactly zero, dense signed
> operators decay, and windowed operators surrender reach.

### The vision it serves, kept as context and NOT as the claim

> an attention module capable of understanding consequences, trained locally then
> on Colab, working as an LLM that understands consequences and choices

Every word of that maps onto an item, which is the only reason it is allowed on
the page at all:

| the words | the measurable thing | item |
|---|---|---|
| "consequences, not similarity" | content-conditional sign -- a THIRD token decides whether j HELPS or HURTS i. Softmax is at exactly 0.000000 by THEOREM: `I + A + A^2` is non-negative entrywise for non-negative A, so a similarity kernel cannot represent "j hurts i" at all. | **M1** |
| "at LLM scale" | survives s = 8..2048 at GLOBAL reach | **M2** |
| "understands" | converts to a ground-truth task softmax fails, softmax's failure distance recorded FIRST | **M3** |
| "choices" | the intervention is PERFORMED, not modelled: 0.000000e+00 under eviction against 2.154868e-05 under gating | **M4** |
| whose result it is | pivot routing with a NON-NEGATIVE operator is Star-Transformer 1902.09113 (2019) | **S2** |

### RELEASE GATE — added 2026-08-25, the user's bar

**A Turing-style evaluation must come out ABOVE self-attention**, and the
HuggingFace release carries that comparison. This is a gate on SHIPPING, not on
the claim sentence, and it is stated separately because it is stronger than
anything M1-M6 asks for.

It is not satisfiable today and the reasons are on record:
  * **No Turing-style eval has ever been attempted -- no file for one exists.**
  * Nothing has trained above **3,652,096 parameters**; the gate the user set is
    300M, priced at **129.7-257.2 A100-hours** (`ceq/sizing.py`), blocked by ~15
    lines of missing checkpoint/resume in `ceq/hf/train.py` rather than by money.
  * The ONE capability comparison ever run at matched parameters went AGAINST the
    operator: COGS-gen softmax **0.0293** (15/512) vs sgate **0.0000** (0/512),
    one-sided Fisher **p = 2.7502788939e-05**, and behind IN-DISTRIBUTION too
    (0.9258 vs 0.7734).

**A Turing comparison run before M3 is GREEN would be measuring an operator that
has not yet shown it can do the thing at all.** Order: M1-M6 and S2 first, then
scale, then Turing. Any other order spends A100-hours to learn something a
3.65M CPU run already knows.

---|---|---|---|
| "consequences, not similarity" | **content-conditional sign**: a THIRD token c decides whether j HELPS or HURTS i. Softmax is at exactly 0.000000 here and it is a THEOREM, not a measurement -- with `out = v + Av + A^2 v` the influence Jacobian is `I + A + A^2`, which is non-negative entrywise for non-negative A. A similarity kernel cannot represent "j hurts i" at all. | **M1** | UNTESTED |
| "at llm scale" | the property survives **s = 8..2048 at global reach**, where every dense signed arm decays s^-1.1..-1.7 and windowed arms buy flatness only by giving up reach | **M2** | UNTESTED |
| "understands" | it converts into a **downstream task with ground truth** that softmax fails, with softmax's failure distance recorded FIRST | **M3** | UNTESTED |
| "choices" | the intervention is **PERFORMED, not modelled** -- `do(c)` on a real token, measured at 0.000000e+00 for eviction against 2.154868e-05 for post-softmax gating, because a gate cannot leave the denominator | **M4** | UNTESTED |

Plus **S2**, which decides whose result it is: pivot routing with a NON-NEGATIVE
operator is Star-Transformer 1902.09113, published 2019. If the capability lives
in the unsigned ablation, **G4 fires** and this sentence may not be written.

### WHAT IS STILL MISSING, stated so nobody has to ask

- **M3 is the bridge and it has never run.** M1 and M2 are properties of an
  operator; only M3 turns a property into "understands" anything.
- **Nothing has trained above 3,652,096 parameters.** "working as a llm" is not
  in evidence at any size. The 300M gate is priced at 129.7-257.2 A100-hours
  (`ceq/sizing.py`), and what blocks it is ~15 lines of missing checkpoint/resume
  in `ceq/hf/train.py`, not money.
- **The one capability comparison ever run went AGAINST the operator**: COGS-gen
  softmax 0.0293 (15/512) vs sgate 0.0000 (0/512), one-sided Fisher
  p = 2.7502788939e-05, at matched parameters, and behind in-distribution too
  (0.9258 vs 0.7734).
- **ARC-AGI has never been scored. A Turing-style eval has never been attempted.**

### THE TECHNICAL SENTENCE THAT M1-M6 ACTUALLY LICENSE

When the six go GREEN, this is what the evidence supports, and it is what goes in
the paper next to the user's sentence rather than instead of it:

> Fixing multi-hop path count by content-selected pivot routing rather than by
> locality holds signed influence flat in context at global reach, and converts
> it into capability X at distance d, with exact eviction and a machine-checked
> finite resolvent -- where softmax attention sits at exactly zero by theorem,
> dense signed operators decay, and windowed operators surrender reach.

---

# G1 SWEEP — RUN 2026-08-25, BY DIRECT FETCH. VERDICT: DOES NOT FIRE.

G1 fires only if **content-selected signed multi-hop routing** is already
published. It is not. But every PAIR in that triple is occupied, and the
nearest miss is closer than the plan assumed.

| construction | pivots | multi-hop | content-selected | signed |
|---|---|---|---|---|
| Set Transformer 1810.00825 | inducing points | no | no | no |
| **Star-Transformer 1902.09113** | relay node | **YES, 2-step** | no (virtual hub) | no |
| BigBird 2007.14062 / Longformer 2004.05150 | global tokens | no | no | no |
| Nystromformer 2102.03902 | landmarks (segment-means) | no | no | no |
| **Perceiver 2103.03206** | latent array | **YES, stacked latent self-attn** | no (learned latents) | no |
| **NSA 2502.11089** / MoBA | blocks | no | **YES, trainable** | no |
| signed GNN propagation 2301.08918, HopGAT | — | **YES** | — | **YES** |
| ParaFormer 2512.14619 | — | yes | — | coefficients only |
| DeltaNet 2406.06484 | — | yes | — | yes, but KEY-KEY |
| SDA 2606.04833 / SignGT 2310.11025 / Cog 2411.07176 | — | no | — | yes |

**Star-Transformer is the near-miss and MUST be named in the claim sentence.**
Verbatim from 1902.09113: the relay node is "a virtual hub to gather and scatter
information from and to all the satellite nodes", and "every two non-adjacent
satellite nodes are two-hop neighbors and can receive non-local information with
a two-step update."

That is the routing mechanism of this plan, published 2019, with k = 1.

**The consequence for the claim sentence, which is not optional:** the draft says
"windowed, normalized, and dense signed baselines each fail one of the two."
Star-Transformer and Perceiver fail NEITHER — they have fixed path count AND
global reach. They fail only on signedness. The sentence must name them.

**The consequence for the ladder:** the selection ablation (S2) is not a
step-4 decomposition, it is the entire claim. Star-Transformer already shows
pivot routing works with a NON-NEGATIVE operator. So S2 runs FIRST, before M2.
If pivots+unsigned is flat too, signedness contributes nothing and the result
belongs to 2019.

**The reframe that is worth more than the cell:** Star-Transformer is not prior
art that steals the claim, it is independent published evidence that the
mechanism WORKS. The 1/k-vs-1/s derivation and the inverse-Littlewood-Offord
k^-1/2 reading both predict what a 2019 architecture already demonstrates. That
is the first external corroboration this mechanism story has ever had.

## ROUND 3 — CEQ v5. Items freeze on first test. Status column only.

| id | item | status |
|---|---|---|
| M3w | **BLOCKED [RUN, r3 iter 7, Chase]** — `windowed_signed` reach is `2w = 16` while the harness default is `d=24` and its sweep runs `d=24..54`: `d(out)/d(x[flipper])` is **exactly 0.0**. Its support `[47,63]` contains the payload and excludes the flipper, i.e. it IS the `payload_only` predictor the bar calibrates as a failure — so the reading would have been indistinguishable from "no capability". **BAR REPAIRED [RUN, r3 iter 9]** — `flipper_dependence` (exactly 2.0 real / 0.0 flipper-blind) and `trained_two_feature` (a MODEL at the harness's own budget, 0.036698) added; `bar_verdict()` is now the single gate in `negation_scope.py`. Chase's broken task is REFUSED by name. **[RUN, r3 iter 17] ALL THREE ARMS PASS THE ABSOLUTE BAR at n_train=8192** — softmax 0.877168 [0.830455, 0.924226], pivot_unsigned 0.747528 [0.696849, 0.797716], pivot_signed 0.673762 [0.632559, 0.715564], n_params=4769 each, softmax first. **Routing beats softmax with DISJOINT CIs.** **G4 VOID [RUN, r3 iter 19]** — at the harness geometry (logits |w| mean 2.68e-03) `_causal_sgate_operator(lam=0.10)` is **ENTRYWISE NON-NEGATIVE**, min entry exactly `0.000e+00`. The paired test compared two non-negative operators; `pivot_signed` was `pivot_unsigned` wearing a name. **WITHDRAWN, not overturned.** λ is a threshold at 1.0, not a dial. Previously read: **G4 CONFIRMED [RUN, r3 iter 18] by the PAIRED test** — the instrument that FAVOURS the signed arm, since both see the identical eval batch: paired mean difference **−0.019901**, 95% CI **[−0.045847, +0.006275]**, B=20000, **includes zero**; signed wins on **55.3%** of examples. Capability is ROUTING. Previously: signed vs unsigned per-arm CIs overlap by 0.0188, so the capability is ROUTING, not SIGN — rewrite as a routing result. 1 seed against M3's 5; d=24, not M3 proper. **[RUN, r3 iter 10] SOFTMAX PASSES THE ABSOLUTE BAR at n_train=8192: eval 0.877168, CI [0.830455, 0.924226], entirely below 1.0.** The budget sweep reads 128→2.1166 / 512→1.3165 / 2048→0.9495 (CI straddles) / 8192→0.8772. *"No arm has ever passed"* was a **DATA-BUDGET fact, not an operator fact**, and every prior M3 reading here was taken at n_train=128 where the harness could not produce a pass.** Previously: **two of the bar's three checks were algebraic identities** (`nrmse(y.mean(),y)≡1.0`, `nrmse(t,t)≡0.0`), and the bar printed CALIBRATED on a task with no flipper dependence at all. **No arm has ever passed the bar**, and there is no model-level positive control, so "arm failed" and "harness cannot pass" are the same printout. **[RUN, r3 iter 1] HARNESS WAS OPERATOR-UNBOUND** — `m3_capability.py`'s only signed arm resolves to `_causal_tgate_operator`, which ships nowhere, and round 2's operator bind covered `pivot_probe.py` only. So round 2's M3 numbers (softmax 1.855584 / pivot_signed 1.342215 / pivot_unsigned 1.956147) are instrument-#17 numbers, **and every one is ABOVE the 1.0 absolute bar**. New bind `test_m3_harness_operator_is_shipped.py` **GREEN [RUN, r3 iter 2]** — `pivot_signed` moved onto shipped `sgate`; the tgate-only `g[s]+tau` params went with it, so all three arms now report **n_params=4769 exactly** and the matched-params requirement is structural rather than tuned. Round 2's M3 numbers are **re-scoped, not corrected**. **[RUN, r3 iter 3] `windowed_signed` ADDED** — shipped sgate at `window=8` (one argument, not a new operator), hop 2 dense within the band, bound by four value assertions: bitwise-equal to sgate_w8, NOT equal to unbounded sgate, zero mass outside the band, and hop 2 reaching exactly 2w **with** nonzero mass in the w..2w ring. All four arms n_params=4769. F4's windowed arm must be built on **sgate**, not tgate [archive:1272]. **CAPABILITY ON THE WINDOWED ARM (Phase 0, FIRST).** Negation-scope flip at d in {256,512,1024}, executable oracle, 5 seeds, matched params + lr sweep, softmax timestamped FIRST. Kill: NRMSE > 1.0; or CIs overlap softmax at all d >= 256; or the unsigned ablation matches. **The kill must be SEEN firing on a deliberately broken arm before the first reading.** | UNTESTED |
| M2n | **SCALE-FREE SIGNED INFLUENCE (R7).** KILL 1: measured E\|B_k(s)\| exceeds A_k(1+eps_Jensen) at any s, A_8 = 2.187500 exact by 256-term enumeration — **STRUCK FOR sgate [RUN, r3 iter 6]: the independence assumption is FALSE. Measured E|Σε| = 7.997 at s=512, P(+) = 0.9998, pairwise correlation 0.9993 — 3.66× the enumerated constant, and equal to the all-aligned control (8.000). Boundedness survives (|B_k| ≤ k is sign-free); only the constant moves. KILL 1 must be re-derived under the measured sign law.** KILL 2: local log-log slope on **s=512->2048 alone** with \|slope\| >= 0.01 (predicted -0.00476; dense reads -1.088 and cannot pass). Baseline s=16, never s=8. Requires identical-draws pairing, floor=0 companion arm, CP intervals + discard counts on every zero, rebuilt twin test passed, no published number moved. | UNTESTED |
| TWIN | **REBUILT TWIN TEST — three arms.** scale-insensitive (monotone magnitude randomization must NOT change the decision); sign-sensitive (sign randomization MUST change it); **NOT-VACUOUS** (`sign(t)*1` must score STRICTLY WORSE than `sign(t)*F^gamma` on the same draws). v4's two-arm version was vacuous: sign is a multiplicative prefactor, so `sign(t)*1` passed it identically. | UNTESTED |
| M6 | Numerical-radius guard compatible with the surviving route. Kill: guard reintroduces decay, or training diverges at every lr. | UNTESTED |
| M7 | Trained reading at >= 25.7M under a pre-registered reading doc. 300M remains the gate and its absence is stated in every document. | UNTESTED |
| R5 | Self-normalized / studentized sign statistic | **STRUCK BEFORE BUILD** — a common positive rescale changed the event in **0/20000** draws. Divides both sides by the same scalar; the event set is invariant. Already proved once here as instrument #16. |
| R8 | Extreme-value-calibrated threshold | **STRUCK BEFORE BUILD** — same 0/20000. Was scheduled FIRST as "cheapest"; would have produced a guaranteed null and read as evidence. |
| R6 | Inverse-propensity (Horvitz-Thompson) | UNTESTED — per-token 1/pi_i is NOT a common rescale, so it may change the event, but this is **unmeasured**. Run the 20000-draw event test before building. Held behind R7. |

**WITHDRAWN CLAIM.** *"Pivot routing makes it worse than dense"* (-1.298 vs
-1.088) is **not statistically supported**: bootstrap B=20000 gives
pivot - dense = -0.2099, 95% CI **[-0.7497, +0.2651]**, which does not exclude
zero. Pivot routing stays dead (both arms far past the -0.3 bar); that sentence
does not, and it appears in every document written since round 2.


## ROUND 4 - CEQ v6'. THE CHOSEN-SIGN ROUND. Items freeze on first test.

| id | item | status |
|---|---|---|
| X4 | **VALUATION INSTRUMENT, precedes both arms.** `sign_flip` on (valuation, mantissa) pairs; **floor DELETED, not defaulted**. Calibration: reproduces every published `floor=0` number exactly. Must-fire: a planted **30-order-spread** arm read correctly where the float instrument **provably misreads it**. | **GREEN [RUN, r4 iter 1]** — `scale/valuation.py`. Calibration 4/4 exact at the published floor; must-fire PROVABLE: `np.float32(1e-30) * np.float32(-1e-30)` = `-0.000e+00` exactly, so `lo*hi < 0` is False and a real flip is missed, while `v_opposite_signs` reads True. **The defect is the MULTIPLY, not only the floor** — deleting the floor does not fix it. Found: the published calibration table is itself a FLOORED reading (`sign_flip_rate` defaults `floor=1e-06`), discarding **57.1%** of `sgate h1` and **0.0%** of two others. |
| A | **DIFFERENCE-SET SCHEDULE (X1).** Cyclic Singer (v,k,1), k~sqrt(s). Birth gates: `\|D-D\| = v-1` as a VALUE; flipper **uniformly at random**, never on the schedule; off-schedule flip rate within CI of on-schedule. Kill: any bitwise-identical gradient pair, OR `\|slope\| >= 0.01` over s=512->2048 on X4, OR M3 fail at 8192. | UNTESTED |
| B | **DISCREPANCY-STEERED SIGNS (X2, certified vs X3).** Sign head minimises `\|sum eps_p t_p\|`; Lovett-Meka reference; **Spencer 6*sqrt(k) printed beside the measured background at every s**. Birth gates: P(+) in [0.35,0.65], corr < 0.2, `E\|sum eps t\| <= c*sqrt(k)` with c pinned, **FKG-escape tested** (shuffle selection order; signs must NOT move). Kill: background > 3x Spencer, OR task loss past the 1.10 bar, OR M3 fail at 8192. | UNTESTED |
| M5' | **Difference-set coverage lemma in Lean** - finite, decidable. Replaces the N=n mismatch item as this round's proof deliverable. | UNTESTED |
| M8 | **FKG-escape certificate.** Replaces the bare diversity gate. | UNTESTED |
| G7 | **Event-change >= 1% pre-build, both arms** - the test that struck R5/R8 at 0/20000 before they cost anything. | **GREEN [RUN, r4 iter 2]** — arm B event-change **20.93%** (harness) / **14.43%** (unit), far above 1%. Arm A `|D-D| = v-1` exact for all five Singer sets. **BUT Spencer is VACUOUS here**: scaled by max|t| it is **4.039e-06** against a natural background of **3.000e-07** — already **7.4× below the bound**. One term carries **50.7%** of background mass (equal terms would be 14.3%), so optimal sign choice buys only **18%**. Arm B is buildable; its stated warrant is not operative. |

**F16 BINDS THIS ROUND.** At the harness geometry (logits `|w|` mean **2.682399e-03**)
`_causal_sgate_operator(lam=0.10)` is **entrywise non-negative**, min entry exactly
`0.000e+00`. lambda is a **threshold at 1.0**, not a dial. **Every prior
signed-vs-unsigned comparison at this geometry is void as sign evidence.**

**F17.** A sign measurement without its **logit scale** is not a measurement:
frustration read **0.2779** at unit scale and **0.000000** at harness scale, same operator.

**F18 SURVIVES F16.** `pivot_unsigned` was always the non-negative arm, so routing beating
softmax - **0.747528 [0.696849, 0.797716]** against **0.877168 [0.830455, 0.924226]**,
disjoint, 4769 params each - stands.

**F19.** M3 readings below `n_train=8192` rank overfitting: 128->2.116579, 512->1.316514,
2048->0.949529, 8192->0.877168.

**F20 — G1 FIRED [CITED, r4 iter 3].** Co-prime spacing is PRIOR ART: arXiv 2606.28560 compares a **"coprime (anti-gridding) reassignment"**. Round 3's severance repair is not novel. Difference sets / Sidon sets are **NOT MENTIONED** there and NOT FOUND in search — arm A proceeds as the **difference-set coverage theorem**, never as "co-prime spacing".

## ROUND 5 - CEQ v7. THE TWO-SPHERES ROUND. Items freeze on first test.

| id | item | status |
|---|---|---|
| K1 | **DUAL SLOPE.** `flip(s)` slope <= -0.4 on X4 AND `D_FR` slope >= -0.1 on the SAME draws. **If `D_FR` slope < -0.3, displacement dies with flip and the answer was "no leap".** | UNTESTED |
| K2 | **FILLER TWIN.** `D_FR(causal c)` vs `D_FR(filler c)` must separate with DISJOINT CIs. A displacement statistic that cannot lose to a filler is M2-clause-2 again and **VOIDS the table**. | UNTESTED |
| K3 | **GEOMETRY EARNS ITSELF.** `theta` must separate causal-vs-filler with a LARGER standardized effect than raw TV on identical draws, **else the sphere is notation and TV ships**. | UNTESTED |
| M2q | Scale-free consequence: K1-K3 GREEN on X4, floor deleted, CP intervals everywhere, every zero with its discard count. | UNTESTED |
| M3 | **THE DECIDING ITEM.** ARM B vs softmax at n=8192, softmax first, 5 seeds, CIs excluding zero. **An arm passing M3 but failing M2q is a ROUTING result - claim rewritten, still shipped.** | UNTESTED |
| M5p | Lean: additive-basis coverage lemma; truncation identity at shipped hops; torque-symmetry as the equilibrium condition. | UNTESTED |
| M8 | Integrity: reproducible-summation journals bitwise at ANY thread count; `gamma_r` printed beside every shadow number; must-fire controls seen firing. | UNTESTED |
| G8 | **NEW. No multiplication inside any sign/flip decision** - the `lo*hi` underflow class, mechanised as a lint over probe code with a must-fire example. | UNTESTED |

**X6 IS PRE-REGISTERED TO CUT ITSELF.** If ONE pass already gives
`||tau||_F ~ 0`, the equilibrium clause is **CUT** and every document says so.
That is the R1 lesson written in before the measurement.

**F-green BINDS:** `pivot_unsigned` **0.747528** [0.696849, 0.797716] vs softmax
**0.877168** [0.830455, 0.924226], DISJOINT, 4769 params each, softmax first.
The one positive result in four rounds, and it is UNSIGNED.

**G1 [CITED, r5 iter 1] - 4 of 6 fetched.**

| cell | verdict |
|---|---|
| reachability / severance in causal sparse attention | **OCCUPIED** - arXiv 2606.02680, *"two adjacent tokens can be disconnected in the attention graph at every depth"*, with coverage laws and a Boundary Bridge repair. **Round 3/4 severance is reproduction, not discovery.** |
| simplex -> sphere Fisher-Rao map | **TEXTBOOK** - Cencov uniqueness, square-root transform, isometry to the positive orthant. Use it; never claim it. |
| Procrustes in deep nets | **OCCUPIED AS ANALYSIS** (RSA, GPA, GCPA, latent alignment). **NOT FOUND as a routing objective inside an operator.** |
| TV ablation probe | **OCCUPIED as interpretability practice.** `D_FR` may be new; "ablate and measure the attention change" is not. |
| additive basis / sumset schedule for attention offsets | **NOT FOUND** - 2606.02680 explicitly has no additive bases, sumsets, difference sets or Sidon sets. |
| Karcher-mean pooling; Star-Transformer delta | **OWED** |

**What is left to claim:** the Procrustes torque as a ROUTING OBJECTIVE, and the
additive-basis COVERAGE THEOREM. Both still owe their kills.

**G1 COMPLETE [CITED, r5 iter 2] - 6/6 attempted.**

| cell | verdict |
|---|---|
| Karcher-mean pooling | **OCCUPIED** - Riemannian Mean Pooling on the SPD manifold; Frechet-mean aggregation *"such as attention"* is existing practice. **GIFT:** arXiv 2003.00335 solves differentiating through the Frechet mean, which is exactly ARM B's blocker. |
| Star-Transformer delta | **NOT ESTABLISHED** - the abstract gives *"shared relay node"*, which reads as fixed rather than content-selected, but selection, causality and evaluation cannot be read from an abstract. **Must not be claimed.** |

**FIVE of seven cells occupied before a line was built.** The two not found -
additive-basis schedule, and Procrustes torque as a ROUTING OBJECTIVE - are what
is left, and **a not-found cell is permission to test, not a result.**

**ARM A FIRST READING [RUN, r5 iter 4]** - 120 draws/cell declared, k in {8,32,128}.

| kill | verdict |
|---|---|
| K1 dual slope | **NOT EVALUABLE.** `flip` = 0.00000 at every k **by F1** - `pivot_unsigned`'s Jacobian `I+A+hop2` has **no negative entry, min exactly 0.000000e+00** - so the flip half was never alive on an unsigned arm. `D_FR` slope **-0.3061** misses the -0.3 line by **0.006** on a 3-point fit. **Untested, not failed.** |
| K2 filler twin | **GREEN** - causal vs filler CIs **DISJOINT at every k**, ~**10x** separation (0.030850 vs 0.003317 at k=8). The statistic can lose to a filler. |
| K3 geometry earns itself | **GREEN, THIN** - theta beats TV at every k but by **2.3% / 2.1% / 4.7%**. Passes the stated criterion; a 2% edge is not "clearly", and the contract's point was that TV ships otherwise. |

**INSPECTOR PASS [RUN, r5 iter 5] - exit 0, CLEAN, 10 checks / 17 controls.**

| item | status |
|---|---|
| `inspector.py` *"published: M2 two-point slope"* | **REPAIRED.** Was `log10(0.02732/0.16511)/log10(4)` compared to `-1.2977` - **three constants the file held itself**, verifying `math.log10`, unfailable by any edit to any shipped file. Now parses `M2_TRAINED_PREREGISTERED_READING.md`, re-derives the slope from the counts, checks the doc's rate column equals `k/n`, and requires README + MODEL_CARD to carry the rate string those counts produce. |
| its controls | **15 -> 17.** One vacuous control removed (*"rejects the struck -1.826"*, also true of a literal); three added, each perturbing a different input the check reads. **All fired.** |
| what the repair found | Published **-1.2977** came from the **rounded** rates; from the raw counts it is **-1.2976**. Delta **4.96e-05** against a **-0.3** bar - **no verdict moves, not a G2 event.** The old check could not have seen it. |
| the M2 counts themselves | **STILL UNJOURNALLED.** `results/` holds no unit with n=751 or n=549. The check now prints `NOT journalled` every run rather than passing silently. |

**GATE 3 AUDIT [RUN, r5 iter 6] - `scale/gate3_audit.py`, exit 0.**

| item | status |
|---|---|
| is gate 3's `0/384` a theorem? | **NO. The cell is LIVE.** 109/384 off-schedule draws have `lo != hi`, max separation **1.505102e-02** = **3.7%** of the on-schedule maximum **4.044973e-01** and ~13 orders above float64 rounding. Real influence. |
| gate 3 verdict | **FAILS, and the FAIL STANDS.** Off-schedule `c` moves the gradient 109 times and the sign never flips; on-schedule it flips 17/384. **The flip capability tracks lattice placement, not content** - the placement-artifact class that killed the dilation arm. |
| audit control 1 (must see influence) | **FIRED** - on-schedule live 269/384. |
| audit control 2 (must read EQUAL) | **WRONG FIRST.** Used `c = i`, which is the *opposite* of inert - it moves `q[i]` and the whole row - and read **live 96/96**. Repaired with the condition derived from the code: `c != i`, `(i-c) not in D`, and no `p` with `(i-p in D and p-j in D)` has `p == c` or `(p-c) in D`. **Now FIRES: live 0/48, max\|lo-hi\| exactly 0.000000e+00.** |
| side effect | That control **binds Cameron's severance closed form** in the direction it claims - structural disconnection implies bitwise identity. One-directional; a single-visible-key row softmaxes to 1.0 and cannot move even when connected. |

**WILSON [RUN, r5 iter 6] - TASK 1 and TASK 4 in, TASK 2/3 still bucketing.**

| item | status |
|---|---|
| flip instrument calibration (TASK 4, ran first) | **PASS.** Planted sign change reads `flip=True` (`+2.849876e+00` / `-2.849876e+00`); softmax reads `flip=False` with `min(I+A+hop2)=0.000000e+00`; at `1e-200` the float path reads `lo*hi = -0.0` and misses a flip the valuation catches. |
| `scale/valuation.py` docstring | **DEFECT.** Its worked example argues the underflow with float32's `1.18e-38`, but the function takes a **Python float64**, where `1e-30 * -1e-30` is represented fine. **The instrument is right; its stated reason does not reproduce at the stated magnitude.** |
| **F16 (`lam` is a threshold at 1.0)** | **DOES NOT GENERALISE.** At ARM A's geometry `_causal_sgate_operator` is signed at **every** lam including 0.10 - **30/30** readings, `min A = -1.363636e-01`, negative fraction 0.2322-0.2495, `I+A+pivot_hop2` negative in 5/5 seeds at every `(s, lam)`. Cause is **logit scale**: mean causal \|w\| **1.171e+01** / **1.320e+01** vs the harness's **2.682399e-03**. F16 was true of one geometry and I generalised it - **ninth appearance of correct statement, wrong object.** |
| K1's flip half | **NOW EVALUABLE** - the signed arm's Jacobian is genuinely negative here, so `flip` is a measurement rather than a theorem. TASK 2/3 numbers **not in**; nothing claimed. |

**PROCESS DEFECT AND REPAIR [r5 iter 7] - raised by the user, third occurrence.**

| item | status |
|---|---|
| house mode, iterations 1-6 | **NOT RUN.** Agents were dispatched **serially, one at a time**, and each report written up on its own. That is sequential delegation. **No parallel fellow tier, no reconciliation, no Prognosis, no Chart.** |
| `inspector.py` vs the Health Inspector | **CONFLATED.** The former is a command; the latter is a named agent who re-runs claimed GREENs and strikes unbound claims. Running one is not running the other. |
| Dr House | **NEVER RELEASED this round**, despite standing instruction to release him when the fellows have gone without a mathematical leap. |
| repair | **3 fellows dispatched in ONE message on the SAME question**, each test-bound with nurse rules. Foreman: is theta a reparametrisation of TV in the linearised regime. Chase: is D1 complete, has any published number moved, what is the blast radius. Cameron: is the MEAN the weak part rather than the sphere, and are `\|\|tau\|\|`/`gamma_1` free separators nobody tested. |
| **escalation chain** | **WRITTEN INTO `LOOP_PROMPT.md` AS BINDING** - fellows -> Wilson -> Health Inspector -> **DR HOUSE (`model: fable`, 5 min hard stop, no nurses, one leap or "no leap", HYPOTHESIS not finding, must be bound by a fellow's RED test or it stays in Open)**. |
| autonomy | **Loop runs to iteration 30 without check-in.** Decisions go through the chain and are recorded with their reasoning in `DONE.md`. |

**X6 EQUILIBRIUM CLAUSE [RUN, r5 iter 8] - `scale/equilibrium_probe.py`, exit 0, first run in five rounds.**

| item | status |
|---|---|
| X6 pre-registered kill | **SURVIVES - the clause is NOT cut.** Residual at the glance **0.599101** [0.58661, 0.61330] / **0.388587** / **0.321843**, CIs nowhere near zero; **100% converged** in **28.43 / 44.02 / 52.55** steps to `1e-8`. The glance is not the fixed point, so "equilibrium" is doing work. |
| must-fire controls | **4/4 FIRED** - identical readings give residual `0.000e+00`; spread readings give `0.603907` and the iterate moves `0.673630`; uniqueness guard fires at `theta_max=3.141593`; `\|\|tau\|\|` is `0.000e+00` when `Y==X` and `5.635273` otherwise. |
| **uniqueness precondition** | **NEW HARD LIMIT.** `theta_max < pi/2` holds on **0.9333 / 0.8167 / 0.5167** of draws at k = 8 / 32 / 128. **At k=128 the Karcher mean is not unique on 48.3% of draws**, so "the settled reading" is not well defined there. The contract sells uniqueness as *"an existence-and-uniqueness statement the DEQ era never had"*. |
| direction of travel | **BOTH TRENDS WRONG FOR ARM B.** More pivots = **slower settling** (28.43 -> 52.55) **and weaker uniqueness** (0.9333 -> 0.5167). ARM B wants k pivots. |
| probe defect 1 | **UNREACHABLE TOLERANCE, mine.** `tol=1e-8` sits below float32's floor - `float32 eps = 1.1920928955078125e-07`, iteration floors at **1.8546e-08**, float64 reaches **5.4944e-13**. `converged` could never be true. Fixed: iteration runs float64 as a **declared analysis choice**. **[STRUCK r5 iter 21 - 5.4944e-13 was asserted `[RUN]` with NO LIVE PRODUCER; it lived only in a comment and in prose. The probe reads 7.307e-13 to 8.405e-13 at `--tol 1e-15 --steps 400`. In the STRUCK registry now.]** |
| probe defect 2 | **CENSORED STEPS REPORTED AS A TIME, mine.** First table printed `steps 64.00`, which was the cap. Now averaged over **converged draws only**, with the converged fraction printed beside it. |
| scope | X6 names the **`\|\|tau\|\|_F` trajectory**; what is measured is the **Karcher residual** trajectory, with `\|\|tau\|\|` at the glance only (0.270901 / 0.175964 / 0.145850). **The `\|\|tau\|\|` trajectory under iteration is NOT MEASURED.** |

**X6 `||tau||` TRAJECTORY [RUN, r5 iter 9] - `scale/tau_trajectory.py`, exit 0 unpiped.**

| item | status |
|---|---|
| the trajectory X6 names | **MEASURED.** `\|\|tau\|\|` falls **29.9x / 19.0x / 18.3x** and **plateaus NONZERO** at **7.255e-02 / 2.962e-01 / 1.019e+00**, while the Karcher residual at the same fixed point reaches **5.4944e-13**. **[STRUCK r5 iter 21 - 5.4944e-13 was asserted `[RUN]` with NO LIVE PRODUCER; it lived only in a comment and in prose. The probe reads 7.307e-13 to 8.405e-13 at `--tol 1e-15 --steps 400`. In the STRUCK registry now.]** |
| **contract inconsistency** | **FOUND, and it is structural.** `\|\|tau\|\|=0` requires `m` parallel to `xbar`, the normalised **Euclidean** mean; the settled reading is the **geodesic** mean. **"equilibrium <=> tau = 0" and "the settled reading is the Karcher mean" are DIFFERENT FIXED POINTS**, separated by **0.031648 rad [0.026318, 0.037550]** at k=8, CI excluding zero, at every k. **Whichever ARM B is built on, the other clause is false of it, and the document asserts both.** |
| geodesic vs flat mean | Gap is **2.75% / 3.90% / 3.90%** of the pivot spread; CIs exclude zero at every k, so the two means are **distinguishable**. |
| my 1% threshold | **ARBITRARY AND UNJUSTIFIED, mine.** The printed verdict line ("machinery EARNS") rests on a cutoff nothing supports. **The raw numbers are the finding; the verdict line is not.** |
| pattern, not a result | K3 read the geometry's edge over TV at **2.1-4.7%**; this reads the geodesic/flat displacement at **2.75-3.90%**. Recorded as a pattern only. |
| probe defect | **FAKE CONTROL, mine, DELETED.** "C3 instrument SEES a Karcher/Euclid gap" passed `True` **unconditionally** and chose orthonormal axes whose two means coincide **by symmetry** - zero either way. Third instrument-#15 in this project and the first in my own file. C1/C2 are real and fired (`0.000e+00`/`0.000e+00`; `2.089918`/`0.635487`). |
| scaling caveat | `tau` carries a factor of `k`, so settled `\|\|tau\|\|` is **NOT comparable across k**. The **drop ratio** is. |

**EQUILIBRIUM REPAIR + INSPECTOR [RUN, r5 iter 10].**

| item | status |
|---|---|
| `inspector.py`, 5th-iteration pass | **CLEAN, exit 0**, 10 checks / 15 controls. Rotation selected `M5 tail norms` -> `s128h2=0.880500 s512h2=1.292741`; its control **rejected the struck 1.471448**. |
| the two-equilibria contradiction | **REPAIRED BY MEASUREMENT.** `\|\|tau\|\|=0` has a **one-pass closed form**: `m = xbar/\|\|xbar\|\|` reads **2.454507e-16 / 9.675157e-16 / 5.176001e-15** vs **8.067740e-02 / 2.823547e-01 / 1.054581e+00** at the Karcher mean. **X6's own kill therefore fires on the `tau=0` branch** - one pass already gives it, so that branch DELETES the clause. **Forced, not chosen.** |
| what the contract now says | Equilibrium certificate = **the Karcher residual**. `tau` = **a displacement statistic**, the job ARM A already gave it. **`tau = 0` may not be written as "equilibrium" anywhere.** |
| the price, carried not buried | Uniqueness holds on **0.9333 / 0.8167 / 0.5167** of draws. **ARM B may not be built on a Karcher mean until it says what happens on the 48.3% at k=128 where the mean is not unique.** |

**FOREMAN [REPORTED, r5 iter 10 - NOT YET A VERDICT; awaiting Wilson, then Inspector].**
Replayed ARM A's own draw stream (published `D_FR causal` reproduced to 6 dp at all three k).

| finding | status |
|---|---|
| **F1 theta IS TV** | **RED-bound.** One-token masking leaves one degree of freedom, so `TV_i = m_i` (residual **2.980e-07**, the float32 floor) and **`theta_i = arcsin(sqrt(TV_i))` IDENTICALLY**. Not a tight sandwich - a construction. Spearman **0.9930-0.9989**. **No draw count separates them.** |
| **F2 the 2% is real, and it is not geometry** | **RED-bound.** Exact float64 route gives `d_theta = +1.0890` vs `+1.0888` - stable. But it is the generic gain of **any** row-wise concave transform, and theta is a **mediocre** one: `TV^0.20` beats TV by **15.8%**, `TV^0.15` by **32.7%** - **3.7x-7.0x theta's margin**. |
| **F3 curvature is a LIABILITY** | **RED-bound.** chord beats geodesic by **+0.86/+0.65/+1.25%**; `sqrt(TV)` by **+3.97/+2.87/+5.70%**. The sphere's only distinguishing feature **costs** standardized effect. |
| **F4 `theta_rows` vs its docstring** | **RED-bound.** Docstring says dead rows give 0; code gives **pi/2**. Constant **0.0015340 rad** floor = **46.2/50.0/55.1% of the FILLER baseline**. K2 survives but is **understated**: 9.30/5.90/4.74x -> **16.44/10.79/9.33x**. K3's d unaffected. |
| **F5 K1's slope is contaminated** | **RED-bound, and LIVE.** **-0.3061** as computed / **-0.3323** live-only / **-0.3346** exact. Trip wire **-0.30**: 2% past becomes **11% past**. **FORWARDED TO WILSON** to verify at his geometry with CIs on both slopes. |
| **F6 theta reads the float32 grid** | **RED-bound.** 83-88% of nonzero rows sit on **five values = `arccos(1-n*2^-24)`** to 1.12e-07. **80.6-85.5% of rows wrong by >100%** vs exact. `p50 = 0`; **top 1% carry 52.8/80.3/90.1%**. Cohen's d unmoved; **every per-row use of theta, including `gamma_1` and `Xi`, is affected.** |
| his OPEN | `gamma_1` as reparametrisation (**not ruled out, not shown**); `\|\|tau\|\|` rank-2 (**instrument too blunt**); **"ship `TV^0.2`" is NOT supported** - 8 exponents searched on the draws that scored them, no CI, no multiplicity correction. |

**K3 CONCAVITY CONTROL [RUN, r5 iter 11] - `scale/k3_concavity_control.py`, exit 0, FIT/SCORE split 50/50.**

| item | status |
|---|---|
| Foreman's identity, re-measured here | **HOLDS.** `max\|theta - arcsin(sqrt(TV))\|` = **8.457e-04 / 9.779e-04 / 8.457e-04** over **20460 live rows** (float32); **20 dead rows** at exactly **1.570796**. **Three independent measurements agree** - mine 8.457e-04 (f32), Cameron 3.375e-10 (f64), Foreman 2.980e-07. |
| theta vs **raw** TV | **theta WINS at every k, CI excludes zero** - **+0.0157 / +0.0282 / +0.0387**. |
| theta vs **`TV^p`** (held-out) | **theta LOSES at k=8** (**-0.0643**, CI **[-0.0981,-0.0159]**, excludes zero) and **TIES at k=32 and k=128**. |
| **K3 as written** | **NOT EVIDENCE FOR THE GEOMETRY.** It measures row-wise concavity, which a plain power of TV supplies with **no sphere, no Fisher-Rao, no Chentsov**. Pre-registered: **rewrite against this control or drop. Not softened.** |

**CORRECTIONS TO MY OWN RECORD [r5 iter 11, both raised by Chase].**

| item | was | is |
|---|---|---|
| the `D_FR` slope verdict | *"misses the -0.3 line by 0.006 ... **Untested, not failed**"* | **The CI is [-0.4314, -0.1860] and K1's bar is >= -0.1, so the ENTIRE CI FAILS THE BAR** - the miss is **0.2061**, and it fails by **0.0860 even at the favourable end**. Against the **-0.3** line the CI **straddles**, so that clause is **unresolved with the point estimate inside the kill region**. **"Untested, not failed" was too generous, and it was my sentence.** |
| *"controls 15 -> 17"* | stated as a property of the Inspector | **A property of `iteration % 3`** (`inspector.py:311`): the M2-slope branch carries 3 controls, the M5 branch 1. **Rotation, not drift.** |

**FELLOW TIER [REPORTED r5 iter 11 - one rung of four; NOT verdicts].**

| fellow | headline |
|---|---|
| **Chase F2** | **K2's 10x is the pivot selector reading its own score back.** `select_pivots` ranks by `key.norm(dim=-1)`, a pure function of the token itself. Filler from ranks k+1..2k - still outside P - collapses **9.39 DISJOINT to 1.10 OVERLAP**; two fillers separate from **each other at 8.52**. **ARM B's B1 ("shadow mass instead of salience") may be the SAME criterion** - `xi` comes from `theta`, `theta` is a monotone read of the salience score. |
| **Chase F3/F4** | **A struck constant (`-1.389`, `0.9938`) is still PINNED by `tests/chase/test_hub_package_hardening.py:496-499`** - two shipped tests make opposite demands, and the Inspector cannot see it. **Every shipped `COSTS` number lost its provenance**: `1.44x`/`1.0334`/`0.379x`/`3,319,296` appear **0 times** in `DONE.md`, **2/7/2/4** times in the round-1 archive. |
| **Chase F5** | **The Inspector's clean bill covers 107 of 1278 tests - 8.4%.** Two files probed inside the blind spot yielded **8 live failures**. |
| **Chase F6** | **His K3 attack FAILED.** Paired bootstrap excludes zero at every k. **12/12 published ARM A fields reproduce at delta exactly `+0.000e+00` - G2 holds.** Retraction blast radius: **2 documents, 1 journal, ZERO tests.** |
| **Cameron F1** | **THE AGGREGATOR WAS THE EFFECT.** `theta.max()` \|d\|=**2.3275** vs `theta.mean()` **1.0888** on the identical published draws - delta **+1.2394 [+0.7962,+1.8565]**, winning in **6/6 cells**. **The incumbent is 46.8% of the best available**, while the whole K3 quarrel is **+0.0241**. `.mean()` divides by 1024 a signal carried by ~503 rows. |
| **Cameron F3** | **`\|\|tau\|\|` beats theta at every k, free** (**+0.1932/+0.2018/+0.2026**, CIs exclude zero) - already computed on every draw and never compared. Unlike theta it is **not a function of the per-row TVs**. |
| **Cameron / G-c** | **`gamma_1` is a TIE** - CI spans zero in **5 of 6 cells**, negative at k=32. **This is the G-c kill condition: `gamma_r` does not separate from draw noise.** |
| **Cameron F5** | theta's float32 active-row mean is **2.43% wrong** vs float64 against TV's **0.000076%**; 53-58% of moved rows read **exactly 0.0**. **Her own limit: does NOT change the verdict** - float64 moves `d` by <= 0.0005. |

**B1 COLLAPSE TEST [RUN, r5 iter 12] - `scale/b1_collapse_test.py`, exit 0. A PROPOSAL tested before it is built.**

| item | status |
|---|---|
| Chase's premise about the selector | **CONFIRMED BY ME [READ] `scale/pivot_probe.py:88`** - `score = key.norm(dim=-1).clone()`, docstring: *"a pure function of the token's own representation"*. The incumbent ranks by **magnitude**. |
| **ARM B's B1 ("shadow mass instead of salience")** | **NOT A RENAME.** `rho(salience, shadow) = -0.024620 [-0.032497, -0.017230]`, **k-independent**; top-k overlap **0.003125 / 0.015625 / 0.093652** against chance **0.007843 / 0.031373 / 0.125490** - **below chance at every k**. **Chase's objection does not land on ARM B.** Not a claim that B1 is *better* - only that it is not the old criterion. |
| controls | **3/3 FIRED**, including the one aimed at the failure mode: **a monotone transform `a^0.3*7+2` reads `rho=1.000000`**, so the instrument is *shown* to catch a relabelling. |
| the tie objection, raised against my own probe | **66.3% of `xi` rows are EXACTLY ZERO**, so near-chance overlap is also what ranking *noise* produces. Settled by the added column: **`top-k inside nonzero xi = 1.000000` at every k** - B1's top-k never selects a zero-score token. |
| corroboration | Cameron measured **53-58% of moved rows** reading exactly 0.0 through theta; this reads **66.3% of all legal candidates** with zero shadow mass. **Same defect, two instruments.** |
| my reporting defects, both fixed | (1) `rho` was printed as **three identical numbers** because neither score depends on `k` - `piv` was computed and never used. **One measurement shown three times.** (2) The tie/noise ambiguity was **not distinguishable** in the first run. Both repaired before the verdict was written. |

**MAX-ROW MECHANISM [RUN, r5 iter 13] - `scale/max_row_mechanism.py`, exit 0.**

| item | status |
|---|---|
| Cameron's mechanism claim (*"the max row is at or near i = c+1"*) | **REFUTED.** argmax at `c+1` on **0.0000 / 0.0167 / 0.0250** of draws against chance **0.005728 / 0.012310 / 0.004483**; **median offset 103 / 115 / 156**. No consistent lift. **She disclaimed it and it stays disclaimed.** |
| the O(1) shortcut it implied | **DOES NOT EXIST.** Reading row `c+1` alone gives \|d\| = **0.1864 / 0.2421 / 0.3489** against the max's **2.0133 / 2.0555 / 2.0078** - gap **+1.83 / +1.81 / +1.66**, and **far worse than the mean** too. |
| the aggregator win itself | **REPRODUCES, and is UNEXPLAINED.** `max` beats the active-row mean by **+0.4503 / +0.8648 / +1.0547**. |
| is it a sphere result? | **NO.** `tv_max` reads **1.9850 / 1.9804 / 1.9730** against theta's **2.0133 / 2.0555 / 2.0078**. **The aggregator lifts both statistics.** |
| controls | **3/3 FIRED** - planted max at row 377 found at 377; planted max at row 42 with c=100 **correctly excluded** (returns 483); `arcsin(sqrt(.))` strictly increasing forces a shared argmax, asserted not assumed. |
| **CONTRADICTION WITH CAMERON - OPEN** | Same s, draws and seeds: `d(th_max)` reads **2.0133** here vs her **2.3275** at k=8, and **2.0078** vs her **1.4806** at k=128 - **an opposite trend in k**. The `mean_all` vs active-row-mean difference cannot explain it, because `d(th_max)` depends on neither. Active-row means nearly agree (**1.5630** vs **1.5304**). **Her probe replays the published generator; mine writes a fresh draw loop and claims no such bind - the likely source. SENT TO WILSON. Neither `th_max` figure may be quoted as the value until he settles it.** |

**DRAW-STREAM BIND [RUN, r5 iter 14] - `scale/max_row_mechanism.py`, exit 0.**

| item | status |
|---|---|
| the bind | **PASSES at `abs=5e-6` on all six published ARM A fields** - causal 0.030850 / 0.018089 / 0.013203 and filler 0.003317 / 0.003068 / 0.002784. Asserted **before** any other output; the probe returns 1 rather than reporting on an unbound stream. |
| **the contradiction I reported at iter 13** | **WITHDRAWN. It was mine.** `arm_a_run.one` draws **three** `d x d` matrices and a `v0` before `j`, and **two more `randn(d)`** after `c`; my loop drew two, none, and none. Every draw past the first sat at a different generator position. |
| the corrected numbers | `d(th_max)` = **2.3275 / 1.8900 / 1.4806**, `d(th_mean)` = **1.5304**, `d(tv_max)` = **2.2586 / 1.8530 / 1.4341** - **Cameron's values to every printed digit.** My unbound figures (2.0133 / 2.0555 / 2.0078) were wrong. |
| the "opposite k-trend" I claimed | **ALSO AN ARTIFACT.** Bound, the `max` advantage reads **+0.7971 / +0.5003 / +0.5088** - shrinking then flat, **her direction**. |
| Cameron's mechanism, retested on the correct population | **STILL REFUTED.** argmax at `c+1` on **0.0250 / 0.0167 / 0.0083** vs chance **0.008959 / 0.007750 / 0.004493** (2.8x / 2.2x / 1.9x chance but **under 3% of draws**), **median offset 137 / 117 / 130**. Row `c+1` alone scores **0.2923 / 0.0583 / 0.1977** against the max's **2.3275 / 1.8900 / 1.4806**. **No O(1) shortcut.** |
| the aggregator win | **REAL, LARGE, UNEXPLAINED** (+0.7971 / +0.5003 / +0.5088 vs a K3 quarrel of +0.0241) - **and shared with TV**, so not a sphere result. |

**WILSON [RUN, r5 iter 15] - the rung whose facts settle disputes. 400 draws/cell, 6 k, bucketed, replay MATCH.**

| item | verdict |
|---|---|
| **K1 `D_FR` clause** | **RESOLVED AND FAILED.** Slope **-0.4137 [-0.4579, -0.3704]** as-computed and **-0.4654 [-0.5173, -0.4160]** live-rows-only. **Both intervals entirely below -0.30**, so the pre-registered *"displacement dies with flip -> no leap"* trigger **FIRES**. Both also fail K1's bar of `>= -0.10`. **ARM A has NOT survived K1; ARM B is NOT authorized.** |
| K1 flip half | **ALIVE, UNRESOLVED.** 6/0/0/1/1/0 per 400 -> pooled **8/2400 = 0.003333 CP95 [0.001440, 0.006557]**. Slope **-0.6960 CI [-1.0000, +0.0000]** straddles the -0.4 bar; the CI is **count discreteness, not an interval**. The unsigned arm's CP95 **contains** the signed arm's rate - **the two arms are not separated at these sample sizes.** Needs 20000 draws/cell (~8 h). |
| K2 / K3 | **K2 passes at all 6 k** before and after correction. **K3 theta wins at all 6 k**; the dead-row correction is an affine map on both groups so **Cohen's d is invariant by construction** (deltas ~1e-16). |
| **the identity** | **A THEOREM, derived and measured.** `theta_i = arcsin(sqrt(TV_i))` exactly. float32 residual **8.457280e-04** (reproducing my figure exactly), float64 **6.828570e-08**. *"K3 can only ever win by Jensen on a concave map."* |
| **my `TV^p` control** | **VINDICATED, his words:** *"this is WHY the coordinator's held-out `TV^p` control beat theta at k=8 - that outcome is expected, not anomalous, and it is the correct control."* |
| Chase F3(a) | **REFUTED.** Shipped dict already repaired (`slope: None`). Registry test **exit 0, 12 passed**; hardening test **exit 1**. **One stale RED, not a contradiction.** |
| Chase F4 | **REFUTED ON THE LETTER, and the refutation is a HAZARD.** `DONE.md` now reads **1/1/1/1** - and all four sit in **F4's own text**. **Substance stands**; the hazard is that provenance could be satisfied by the report of its absence. |
| Cameron `gamma_1` tie | **REFUTED at his geometry - G-c's kill does NOT fire.** **0.9123 / 0.9766 / 0.8507**, all CIs excluding zero, none negative. He did not run her pipeline. |
| Chase F2 | **REFUTED in degree.** K2 is **DEGRADED, NOT VOIDED**: band-filler **0.4301 / 0.4586 / 0.5425**, all excluding zero; key-norm matching removes **~65%** at k=8. **The filler pool IS two populations** (band-vs-tail **1.0716 / 0.8782 / 0.4868**). |
| Cameron F1 | **CONFIRMED.** max-mean **+1.1347 / +0.9355 / +0.4959**, all excluding zero. Sphere-vs-TV at the same aggregator is **+0.06 to +0.11, ~5% of the aggregator's +1.13**. |
| dead-row floor | **VERIFIED** (1 in 1200/1200 draws, 0.0015340 rad) **with a refinement**: rows reading exactly pi/2 reach **2, 3, 4** and those extras are **LIVE**. **Detecting dead rows by `theta == pi/2` over-subtracts by up to 3.** |
| his own AUC prediction | **REFUTED by him.** Deltas +1.06e-03 / +2.50e-05 / -9.75e-04; float64 gap = **1 discordant pair in 3600**. |
| **B1 - OPEN, he and I disagree** | He infers B1 collapses onto salience from `rho(\|\|k_c\|\|, theta) = 0.45-0.58`. **That is key-norm vs the masked token's DISPLACEMENT, not vs SHADOW MASS per candidate.** My direct measurement of the quantity B1 proposes reads **`rho = -0.024620`, overlap below chance at every k**, three controls firing. **Needs one probe, not an argument.** |

**INSPECTOR [RUN, r5 iter 15] - exit 0, CLEAN, 11 checks / 16 controls.**

| item | status |
|---|---|
| **Chase F5** | **EXACT, verified twice independently.** `--collect-only` = **1278**; `check_suites` = **107**. **8.37%.** |
| the repair | `inspector.py` **now prints its own coverage** with the sentence *"a CLEAN result above is a statement about these 107 tests and about NOTHING ELSE"*, a **must-fire control that the fraction is measured not assumed 100%**, and an **INDETERMINATE** if collection fails. **The misreading cannot recur.** |

**B1 DECOMPOSITION [RUN, r5 iter 16] - `scale/b1_decomposition.py`, exit 0, 3/3 controls fired.**

| item | status |
|---|---|
| the Wilson/coordinator disagreement | **SETTLED: different objects, both readings correct.** R1 `rho(\|\|k_p\|\|, \|\|xi_p\|\|)` per candidate = **-0.025318 [-0.0322,-0.0181]** (reproduces my iter-12 -0.024620); R2 `rho(\|\|k_c\|\|, D_FR)` per draw = **+0.570980** (inside Wilson's +0.45 to +0.58). **Measured on the same draws.** |
| **my own hypothesis** | **REFUTED.** I predicted `\|R3\| > 0.6` - that shadow mass is a read of attention-to-c. It reads **+0.144930 [+0.1174,+0.1728]**. **What `\|\|xi_p\|\|` tracks is NOT established**, and the probe prints **AMBIGUOUS** by my own pre-registration rather than claiming the branch I wanted. |
| causality | **CONFIRMED, asserted not assumed.** `\|\|xi_p\|\| = 0` for **68.04%** of candidates; max below `c` = **4.371e-07** (float dust). Row `p` moves only if it attended to `c`, which needs `p > c`. Corroborates Cameron's 53-58% from a third direction. |
| **B1 IS ILL-POSED AS WRITTEN** | **NEW, and neither Wilson nor Chase raised it.** `xi` is defined **relative to a chosen `c`**, but `select_pivots` is documented *"USES ONLY CONTENT -- never `c`, never `i`, `j`"*. Top-k overlap across a change of `c` on the SAME draw: **0.054167 / 0.196094 / 0.631510** (chance 0.007835 / 0.031342 / 0.125367). **At k=8 the selection retains 5.4% of its picks.** A defect in the **contract sentence**, found before ARM B exists. |
| my verdict-rule defect | **CAUGHT BY THE PROBE'S OWN OUTPUT.** The first gate used **bulk rank stability (+0.891716)** and printed *"well-posedness survives"*. The bulk Spearman is high **because ~68% of the vector is exact zeros tying with zeros regardless of `c`**. `select_pivots` takes a **top-k**, not a bulk ranking. Gate moved; **verdict reverses**. |
| k-independence, noted not hidden | R1, R3 and bulk R4 **do not depend on k**, so three identical rows print for them. **Only the top-k overlap is k-dependent, and it is the deciding one.** |

**AGGREGATOR MECHANISM [RUN, r5 iter 17] - `scale/aggregator_mechanism.py`, exit 0, bound to the published stream.**

| item | status |
|---|---|
| **the aggregator win, four iterations "UNEXPLAINED"** | **EXPLAINED.** `max_i theta_i = arcsin(sqrt(max_i A^c[i,c]))` -- **`theta.max()` is a strictly-increasing function of the largest attention weight any row places on `c`.** Not a geometric statistic. |
| M1 identity on VALUES | **HOLDS.** `max\|max th - arcsin(sqrt(max A))\|` = **6.697e-04 / 6.074e-04 / 6.471e-04**, the float32 floor, same order as Foreman's 8.457e-04. |
| my M1 test, first version | **WRONG TEST, mine.** It checked **argmax**, read 0.84-0.94, and printed *"BROKEN"*. argmax is preserved only without **ties**, and F6 measured theta quantised onto **five float32 levels holding 83-88% of nonzero rows**. Reported as a tie-break artifact now. |
| M2 does the sphere contribute? | **NO.** \|d\| max theta **2.3275 / 1.8900 / 1.4806** vs \|d\| max A **2.2586 / 1.8530 / 1.4341**; **AUC gap ~1e-03**. The win is visible with **no Fisher-Rao, no Chentsov, no arccos**. |
| M4 the mechanism | **ATTENTION CONCENTRATION.** max A causal **0.881909 / 0.843779 / 0.714204** vs filler **0.161140 / 0.207901 / 0.173796**, **ratio 5.473 / 4.059 / 4.109**. A pivot captures ~88% of some row's whole attention; a filler 16%. |
| M3 is it the selector's score returning? | **NOT SIMPLY** - within-arm `rho(\|\|k_c\|\|, max A)` = **+0.25 to +0.45**. **The POOLED +0.76 is NOT evidence** - pooling mixes high-key-norm pivots with the low-key-norm tail, so it **is the separation under test**. |
| **the control nobody has run** | **KEY-NORM-MATCHED FILLER vs THE AGGREGATOR: NOT YET RUN.** Wilson showed matching key-norm removes **~65%** of the mean-based K2 effect (1.2267 -> 0.4301). Until that control is applied to the aggregator, *"the win is not the selector"* rests on a within-arm rho of 0.45, **not on a matched control**. |

**AGGREGATOR vs KEY-NORM-MATCHED FILLER [RUN, r5 iter 18] - `scale/aggregator_matched_filler.py`, exit 0, bound on the first 120 draws.**

| item | status |
|---|---|
| **the aggregator finding** | **SEVERELY DEGRADED AT EVERY k.** causal-vs-band \|d\| on `max A` = **0.0677 [0.0040,0.2978] / 0.2241 [0.0274,0.4494] / 0.8366 [0.6020,1.1082]**, keeping **2.9% / 13.3% / 58.9%** of the unmatched-tail effect. |
| the concentration ratio | **1.023 / 1.116 / 1.997** against a matched band. The **5.473** reported at iter 17 was against an **unmatched tail**; a key-norm-matched filler reaches **0.864755** peak attention against causal's **0.884602**. |
| **the comparison that matters** | Wilson's **MEAN**-based K2 kept **35%** under this same control. The **aggregator keeps 2.9%**. **The aggregator is MORE confounded by the selector than the mean it replaced, not less.** |
| what still stands | Cameron's F1 - `theta.max()` beats `theta.mean()` by +1.13 with CIs excluding zero - **is real and unaffected**. What dissolves is the **interpretation**: the contrast it amplifies is largely the selector's own score at small k. **A bigger effect on a confounded contrast is a bigger confounded effect.** |
| my verdict gate | **TESTED HALF ITS OWN PRE-REGISTRATION.** It checked only "CI excludes zero" and printed **SURVIVES**; the pre-registration reads *"CI excludes zero **and the band ratio stays above ~2x**"*, and the ratio is **1.023**. Fixed; verdict reverses to SEVERELY DEGRADED. |
| **the pattern, now three** | iter 16 gated on **bulk rank** when selection takes a **top-k**; iter 17 tested the identity on **argmax** when it claims **values**; iter 18 tested **half** a two-part gate. **Each measured something ADJACENT to what was pre-registered, and each erred in the flattering direction.** All three self-caught by the probe's own output. |
| bind defect, also self-caught | First read MISMATCH at all three k - comparing a **160-draw** mean to a **120-draw** published one. A statement about denominators, not the stream. Now asserts on the **first 120**. |
| limit travelling with the result | The match is by **RANK, not value**: band/causal key-norm **0.9120 / 0.8900 / 0.8347**. The residual gap is **uncontrolled**, so **2.9% is an UPPER bound** on what survives. |

**D1 [WRITTEN, r5 iter 19] - `D1.md`.**

| item | status |
|---|---|
| D1 as a document | **CREATED.** It did not exist: named twice in `LOOP_PROMPT.md`, defined only in the round-4 archive, with `PROGNOSIS.md` serving as D1 in substance for rounds 1-4 under another name. |
| **acceptance criteria** | **SUPPLIED** - the part that was actually missing. Five conditions: kills named with **intervals** not point estimates; defects recorded **with attribution of who caught them**; withdrawn claims recorded **with replacements**; survivals stated separately and **unpadded**; limits collected **once**. |
| round-5 chapter | **WRITTEN.** Verdict: **the central kill fired, no positive claim survives.** Covers K1 (fired, both intervals below -0.30), K3 (passes and cannot mean what it meant), the aggregator (2.9% retained), gate 3 (failed on a measurement), B1 (not a rename, ill-posed). |
| what survives, unpadded | X6's equilibrium clause; the contract contradiction **repaired by measurement**; two frame-independent facts about softmax attention. |
| defect ledger | **10 defects of mine tabulated with where each was introduced**, plus the shape the last four share. **4 defects found by other agents listed separately** so the record does not read as self-discovery. |
| withdrawn claims | **5 tabulated with what replaced them**, including two of mine that erred in the generous direction. |
| **promise** | **HELD.** The chapter's closing limit states that the Health Inspector had not returned, so **nothing in D1 has passed an independent audit yet**, and a struck claim must not stand in it as though it had survived. |

**INSPECTOR PASS + SCAN COVERAGE [RUN, r5 iter 20].**

| item | status |
|---|---|
| `python inspector.py` | **CLEAN, exit 0, 11 checks / 18 controls all fired.** Coverage line holds at **107/1278 = 8.37%**. |
| **`D1.md` in the struck-constant scan** | **WAS ABSENT, NOW ADDED.** Written at iteration 19, it is the contract's negative-result **deliverable** and it **ships** - and `LEAD_DOCS` did not list it. **A shipped document outside this scan is how `1.471448` survived eighteen iterations.** [RUN] 13 passed, exit 0; **D1.md carries no struck value.** |
| **the "9 documents" count** | **WAS A TYPO.** `LEAD_DOCS` listed **`MODEL_CARD.md` twice**; the params deduplicate via `sorted(set(...))` so no check was doubled, but the **reported count is the param count**. **Chase flagged this at iteration 11 and it had not been repaired.** De-duplicated. |
| full-suite corroboration for Chase F5 | **WEAK, and labelled so.** A nurse's full-suite attempt collected **1278** and was **KILLED at 56% with no summary and no exit code** (`grep -c EXITCODE` = 0). **Not a failure count.** It agrees in direction with Chase's directly-measured 8 failures in two files: **failures exist outside the covered 107.** |
| tree drift during measurement | `scale/*.py` grew **51 -> 52** mid-task with ten new probe files observed. **Several agents write this tree concurrently** - which is why every number this round carries its own bind. |

**HEALTH INSPECTOR [RUN, r5 iter 21] - rung 3 of 4. 37 claims audited, 3 struck. All strikes applied the same iteration.**

| # | struck | why | applied |
|---|---|---|---|
| **S1** | **`5.4944e-13`** (mine) | Asserted `[RUN]` with **no live producer** - it existed only in a code comment and in prose. Probe reads **7.481e-09/8.155e-09/8.405e-09** published, **7.307e-13/7.958e-13/8.405e-13** at `--tol 1e-15 --steps 400`. **Not the mean, min or max at any k.** It came from a throwaway float64 script whose own output printed `nan` for both means. **The 1.471448 class.** | **Added to the `STRUCK` registry**; comment replaced with reproducible figures; `LOOP_PROMPT.md` + `CHECKLIST.md` marked. Registry then caught **two more unmarked assertions I had missed**. [RUN] **13 passed, exit 0.** |
| **S2** | the sentence *"his 1.1019 does not reproduce, nor his 9.3869"* (mine) | **Both reproduce to four decimals** on Chase's bound probe. Wilson measured **Cohen's d**; Chase's is a **D_FR ratio** - different quantities. The sentence claimed a failed reproduction that did not happen. **Wilson's measurement untouched.** | Struck in `CHECKLIST.md`. |
| **S3** | the provenance bind | **Confirmed and worse:** delete two lines and all four assertions go False, **and nine more figures are missing outright** - all present in `DONE_ARCHIVE_ROUND1.md`. | **Repaired at the class**: corpus is every `DONE*.md`; a hit counts only inside a **`[RUN]` paragraph or a table row**. Now **FAILS listing nine real absences** - correct behaviour, left RED. |

| cleared | detail |
|---|---|
| **all six of my probes** | Re-run independently, **every one exit 0**, figures exact - including `k3_concavity_control`'s **-0.0643 [-0.0981,-0.0159]** and `max_row_mechanism`'s **6/6 bind**. |
| claimed-greens | `run_calib` rc 0; `inspector.py` CLEAN; **107/1278 = 8.37%**; `--collect-only` **1278**. **Chase F5 exact.** |
| unbound findings | **None.** Every named RED exists and fails. `chase_k3_ci` rc 0 is correct - a *failed attack* claims no RED. |
| contradictions vs Wilson | **None standing.** All four refutations recorded; **G-c's kill recorded nowhere as fired.** |

| his findings nobody had recorded | detail |
|---|---|
| two independent failures, not one | `test_hub_package_hardening.py` rc 1, **4 failed = 2 tests x 2 devices**. **The second was attributed to the first.** |
| archive count is 5, not 4 | `3,319,296` appears **5 times on 4 lines**. **Chase and Wilson both counted lines, not occurrences.** Substance unaffected. |
| **caveat bearing on K1** | `arm_a_k1.py`/`wilson_probes.py` **replayed cached journals** (*"0 remaining, ran_this_bucket: 0"*). **Replay MATCH is real; a fresh derivation of the slope was not attempted.** |

| my control defect, new shape | The must-fire control fed literal paragraphs to `_measured`, but the evidence filter lived in the **corpus builder**, so **the control could not reach the logic it guarded** and failed. Filter moved into `_measured`. The three earlier gate defects tested the **wrong statistic**; this one **could not reach its subject at all**. |

**D1 AUDIT-CORRECTED [r5 iter 22] - `D1.md`.**

| item | status |
|---|---|
| the struck figure in D1 | **NEVER PRESENT.** [RUN] `grep -n "5.4944e-13" D1.md` -> no hits. It was in a code comment, `CHECKLIST.md`, `LOOP_PROMPT.md` and `DONE.md`, all handled at iter 21. **D1 was clean of it by luck, not by care.** |
| `### The audit` section | **ADDED** - all three strikes stated so they cannot be softened; what the audit **cleared** (six probes re-run, every one exit 0, figures exact); and **its two unrecorded findings** (two independent failures not one; archive count **5 on 4 lines**, both agents counted lines). |
| **the K1 caveat** | **IN LIMITS.** The slope that fired the central kill rests on a **journal replay, not a fresh derivation** - every unit already journalled, none recomputed. **Replay matched bitwise; nobody re-derived the slope from draws this pass.** |
| the closing limit | **CORRECTED.** The inspection has returned and its strikes are applied; **its own two open gaps are named** - the verified-facts pass was outside its mandate, and one claim names a file the log never identifies. |
| acceptance criteria | **5/5 SELF-CHECK OK** against the file, plus the audit section. Registry **13 passed, exit 0**. |
| **completion status** | **KEPT is FALSE** - ARM A did not survive K1 and ARM B was never authorized. **BROKEN requires the write-up with its numbers, which now exists.** One mechanical step remains: line 1 of `DONE.md`. |

**ROUND 5 CLOSES [r5 iter 23] - `TWOSPHERES: BROKEN`.**

| item | status |
|---|---|
| **verdict** | **BROKEN - ARM A, K1's dual slope, displacement clause.** Slope **-0.4137 [-0.4579,-0.3704]** and **-0.4654 [-0.5173,-0.4160]**, both entirely below the **-0.30** trigger, both failing the **>= -0.10** bar. |
| KEPT, checked not assumed | **FALSE on two grounds** - ARM A did not survive K1, **and ARM B was never authorized**. The contract's *"build only if ARM A survives"* was honoured, not relaxed. |
| write-up | **`D1.md`**, 273 lines - round-5 chapter, audit section, defect ledger, withdrawn-claims table, limits. **5/5 acceptance criteria self-check OK.** |
| chain | **COMPLETE.** fellows -> Wilson -> Health Inspector. **37 claims audited, 3 struck, all applied.** |
| Dr House | **NEVER RELEASED - a decision.** Every kill fired from a measurement with an interval. He is for a gap in **innovation**; this round had a gap in **results**. |
| left open | K1's **sign-flip clause** (8/2400, interval is count discreteness, arms not separated - needs 20000 draws/cell); **what peak attention tracks** once the key-norm confound is removed. |


---

# CEQ v8.2 — ROUND 6, THE HILBERT ROUND

| item | status |
|---|---|
| **K-A** `Δ̂ = +∞` → ARM S unbuilt, T2 | **REPAIRED PRE-DATUM.** Was `κ̂_cert ≥ 1`, which `tanh` saturates to exactly `1.0` at `Δ ≥ 76.246190` in float64 — a ratio of `e^76 = 3.73e+32`, ordinary for softmax rows. Now fires on `Δ̂ = +∞` only; `1−κ = 2/(e^(Δ/2)+1)` in closed form, verified against 50-digit `Decimal` to `<1e-40`. UNTESTED on live draws. |
| **K-B** matched re-run dissolves F-green | **UNTESTED** — Cameron's matcher builds at it.0, the re-run is it.1. Decides the round's shape (+3/−5). |
| **K-C** twin matches settled → equilibrium clause cut project-wide | **UNTESTED** — Phase C, RULE 2 pins it to iteration 12. |
| **K-D** ARM P matched separation fails | **UNTESTED** — Phase B. |
| **K-E** leakage ratio > 0.5 voids degree-2 claims | **UNTESTED** — Cameron, Phase B pilot. |
| **K-F** wall clock > 1.5× | **UNTESTED** — gate applies to the SUM `cost(glance) + t*·cost(T) + N·cost(J·v)`. |
| **K-G** any statistic of degree ≤ 1 struck unbuilt | **UNTESTED** — Foreman's degree audit; F-identity mechanised as algebra rather than judgment. |
| **K-H** K1 by SPRT boundary, either side, +1 | **UNTESTED** — Chase derives the slope→rate mapping at it.0 **before any draw**; thresholds `±2.944439` verified two ways. |
| **G1 Birkhoff** [U] | **OWED, FETCHING.** Foreman, it.0. Open question he must settle: Birkhoff is stated for positive **linear** maps and `T` is a normalised weighted combination — the hypotheses may not admit it. |
| **G1 deltas** — Sinkhorn / DEQ / Star / Shapley-interaction | **OWED, IN FLIGHT.** Sinkhorn's convergence proof IS Birkhoff, so the delta must be the *settling-certificate* use, not the contraction. |
| **RULE 2** M3 run by iteration 12 | **CLOCK RUNNING** from iteration 0. |
| **RULE 4** caveman register, all agents | **IN FORCE.** Artifacts exempt and stay normal English. |

**ROUND 6 it.1 — the metric primitive [RUN].**

| item | status |
|---|---|
| `scale/hilbert.py` — `d_H`, `delta_hat`, `kappa_cert`, `one_minus_kappa`, `neumann_terms` | **GREEN.** 30 tests, RED shown first (`ModuleNotFoundError`, exit 2). Written once because three fellows consume it and three implementations give three bugs. |
| projective invariance | **GREEN**, ten decades of scale. Must-fire constructs a scale-sensitive metric and shows the property rejects it — **not vacuous**. |
| overflow at extreme spread | **GREEN** — literal ratio overflows at `e^700`; log form reads `700.0` to `1e-12` rel. |
| boundary → `+inf` | **GREEN** — zero, negative, and both-zero. `delta_hat` goes infinite from **one zero in 96 entries**, never averaged away. |
| closed-form gap vs 50-digit `Decimal` | **GREEN** at Δ = 10 … **1400**, where the float route has read exactly 0 since Δ = 100. |
| **defect, self-caught** | `neumann_terms` computed `math.log(1.0 - gap)` — at Δ=76.5 the gap is `4.889518e-17`, `1.0-gap` rounds to exactly `1.0`, log is `0.0`, **ZeroDivisionError**. The function undid the K-A repair one call after it was made. Fixed to `math.log1p(-gap)`, bound by regression at Δ = 60/76.5/100/200/700. Caught by pairing the closed form against a brute-force loop — two methods that fail differently. |
| **NEW CONSTRAINT on contract 1.2** | **`κ < 1` is not the same as the implicit gradient being computable.** Neumann terms for `1e-6`: Δ=20 → **254,653**; Δ=60 → **2.3e14**; Δ=76.5 → **1.05e18**; Δ=∞ → **−1, no finite N**. The `+2` for `κ<1` buys a **uniqueness certificate, not a trainable arm**. Foreman's live `Δ̂` now decides whether 1.2 is implementable at all. |

**ROUND 6 it.2 — the settling driver, and a certificate that was not one [RUN].**

| item | status |
|---|---|
| `scale/settle.py` — `settle`, `column_diameter`, `predicted_steps` | **GREEN**, 16 tests, RED shown first. Generic over `T`; journals the SUCCESSIVE residual `r_t = d_H(m_{t+1}, m_t)`, which obeys `r_t ≤ κ·r_{t-1}` without needing `m*`. |
| convergence law | **GREEN**, one-sided — observed ratio never exceeds `κ̂_cert`; observed steps never exceed `t* = ⌈log(r₀/tol)/log(1/κ)⌉`. |
| must-fire: non-contracting map | **FIRED** — a permutation is an isometry on the cone, reported unconverged at the cap. |
| must-fire: map leaves the cone | **FIRED** — `left_cone` is set **separately** from `converged=False`; conflating them misattributes K-A. |
| **contract 1.1 `Δ̂` estimator** | **DEFECTIVE, REPAIRED PRE-DATUM.** Defined as *"max over sampled pairs"*. The supremum is attained at the **extreme rays** — `A e_i` is exactly column `i`. Uniform interior sampling reaches **53.1% / 48.5% / 39.0%** of truth; implied κ **0.327189 / 0.309848 / 0.320729** against truth **0.564416 / 0.578982 / 0.692614**. **A lower bound on Δ presented as an upper bound on κ, erring optimistic.** Round 5's adjacent-gate class; would have been the fifth. Chains into 1.2, where it would have under-budgeted `N` and surfaced at Phase D. |
| repair window | **CLOSED.** Both 1.1 repairs (tanh saturation; sampled-Δ̂ optimism) were made before any datum. **No third repair after Foreman's first `Δ̂`.** |
| **open, and now urgent** | **Is `T` linear?** Birkhoff is stated for positive **linear** maps; `T`'s `w_p(m)` depends on `m`. If not linear, the column route is unavailable **and the theorem's hypotheses may not be met at all**. Foreman, answering plainly. |
| RULE 1 | **2 of 3 iterations are instrument work — above the 40% cap.** Next iteration must not be. |
