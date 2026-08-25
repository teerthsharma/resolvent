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
| M3w | **BLOCKED [RUN, r3 iter 7, Chase]** — `windowed_signed` reach is `2w = 16` while the harness default is `d=24` and its sweep runs `d=24..54`: `d(out)/d(x[flipper])` is **exactly 0.0**. Its support `[47,63]` contains the payload and excludes the flipper, i.e. it IS the `payload_only` predictor the bar calibrates as a failure — so the reading would have been indistinguishable from "no capability". **BAR REPAIRED [RUN, r3 iter 9]** — `flipper_dependence` (exactly 2.0 real / 0.0 flipper-blind) and `trained_two_feature` (a MODEL at the harness's own budget, 0.036698) added; `bar_verdict()` is now the single gate in `negation_scope.py`. Chase's broken task is REFUSED by name. **[RUN, r3 iter 17] ALL THREE ARMS PASS THE ABSOLUTE BAR at n_train=8192** — softmax 0.877168 [0.830455, 0.924226], pivot_unsigned 0.747528 [0.696849, 0.797716], pivot_signed 0.673762 [0.632559, 0.715564], n_params=4769 each, softmax first. **Routing beats softmax with DISJOINT CIs.** **G4 FIRES**: signed vs unsigned CIs OVERLAP by 0.0188, so the capability is ROUTING, not SIGN — rewrite as a routing result. 1 seed against M3's 5; d=24, not M3 proper. **[RUN, r3 iter 10] SOFTMAX PASSES THE ABSOLUTE BAR at n_train=8192: eval 0.877168, CI [0.830455, 0.924226], entirely below 1.0.** The budget sweep reads 128→2.1166 / 512→1.3165 / 2048→0.9495 (CI straddles) / 8192→0.8772. *"No arm has ever passed"* was a **DATA-BUDGET fact, not an operator fact**, and every prior M3 reading here was taken at n_train=128 where the harness could not produce a pass.** Previously: **two of the bar's three checks were algebraic identities** (`nrmse(y.mean(),y)≡1.0`, `nrmse(t,t)≡0.0`), and the bar printed CALIBRATED on a task with no flipper dependence at all. **No arm has ever passed the bar**, and there is no model-level positive control, so "arm failed" and "harness cannot pass" are the same printout. **[RUN, r3 iter 1] HARNESS WAS OPERATOR-UNBOUND** — `m3_capability.py`'s only signed arm resolves to `_causal_tgate_operator`, which ships nowhere, and round 2's operator bind covered `pivot_probe.py` only. So round 2's M3 numbers (softmax 1.855584 / pivot_signed 1.342215 / pivot_unsigned 1.956147) are instrument-#17 numbers, **and every one is ABOVE the 1.0 absolute bar**. New bind `test_m3_harness_operator_is_shipped.py` **GREEN [RUN, r3 iter 2]** — `pivot_signed` moved onto shipped `sgate`; the tgate-only `g[s]+tau` params went with it, so all three arms now report **n_params=4769 exactly** and the matched-params requirement is structural rather than tuned. Round 2's M3 numbers are **re-scoped, not corrected**. **[RUN, r3 iter 3] `windowed_signed` ADDED** — shipped sgate at `window=8` (one argument, not a new operator), hop 2 dense within the band, bound by four value assertions: bitwise-equal to sgate_w8, NOT equal to unbounded sgate, zero mass outside the band, and hop 2 reaching exactly 2w **with** nonzero mass in the w..2w ring. All four arms n_params=4769. F4's windowed arm must be built on **sgate**, not tgate [archive:1272]. **CAPABILITY ON THE WINDOWED ARM (Phase 0, FIRST).** Negation-scope flip at d in {256,512,1024}, executable oracle, 5 seeds, matched params + lr sweep, softmax timestamped FIRST. Kill: NRMSE > 1.0; or CIs overlap softmax at all d >= 256; or the unsigned ablation matches. **The kill must be SEEN firing on a deliberately broken arm before the first reading.** | UNTESTED |
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

