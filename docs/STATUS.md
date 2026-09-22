# Status — done against total

Read at 2026-09-22, after the 4x5 grid completed at 20/20 and the first hour of
Phase J (`tests/foreman/phase_j/RECORD.md`). Every percentage
on this page is a count of named gates, not a judgement of effort. Each table lists its own denominator so a reader can
disagree with the denominator rather than with the number.

The north star is **conjunctive**: three conditions that must all hold. A
conjunctive goal is not the average of its legs — it is the weakest leg. Both
numbers are given below, and the weakest leg is the honest one.

---

## Headline

| reading | value |
|---|---|
| gates passed, all three conditions | **9 of 19 — 47%** |
| the goal as actually stated (conjunctive, weakest leg) | **30%** |
| the leg that sets it | condition 3, *carries the weight of self-attention or JEPA* |

---

## Condition 1 — understands causality

**2.5 of 5 gates — 50%**

| # | gate | state | evidence |
|---|---|---|---|
| 1 | An operator whose entries encode path structure, not pairwise affinity | **done** | `W_ij = G_ij e^{s_ij} / Z_i^β` with `G_ij = ∏_{k=j+1..i} m_k e^{iθ_k}`, implemented and exact at the `(β, qk)` corners |
| 2 | A bed that demands order, separated from a commuting control | **done** | H.1 on a generator handed over as bare integers: operator **0.8620** vs DIAG control **0.2860** at matched 404 params, 5/5 seeds, control saturated within 0.031 of its multiset ceiling 0.3110 |
| 3 | The non-commutative matrix separated from recurrence alone | **half** | 84.9% of the gap needs the matrix, 15.1% is recurrence — measured, but the bed cannot separate the effect from its own generator family |
| 4 | A second, independent bed carrying the same property | **not done** | — |
| 5 | The mechanism named and tested | **not done** | both named candidates just failed; see condition 2 gate 7 |

---

## Condition 2 — beats anything before it

**5 of 9 gates — 56%, with three gates now measured as failures rather than open**

| # | gate | state | evidence |
|---|---|---|---|
| 1 | A win exists | **done** | −0.2473 eval NLL, 5 seeds |
| 2 | At matched parameters, excess reported not hidden | **done** | 783 excess per the same per-block formula on all three domains |
| 3 | More than one domain | **done** | TinyStories −0.2473, WikiText-103 −0.3285, codeparrot −0.3301 |
| 4 | Against the strongest baseline lacking the property | **failed** | the strongest such baseline is no longer the softmax twin: a Forgetting Transformer gate on the twin (727,704 params, same count as a2) recovers **100.2 / 100.3 / 110.9%** of `C_win` at split seeds 0 / 1 / 2 |
| 5 | A seed interval on every domain | **not done** | 5 seeds / 5 seeds / **1 seed** — the third domain is a data point, and its record says so |
| 6 | Survives a varied-split protocol | **done** | `C_win` = **+0.2470**, sd 0.0160, 95% CI **[+0.2271, +0.2668]** excludes zero, 5/5 split seeds, CRN digests byte-identical across all four arms at every seed |
| 7 | The mechanism identified | **done — and it is prior art** | the win is a data-dependent forget gate: the FoX twin matches the family's per-token gain in every difficulty quintile to within 0.01. Before that: `C_phase` **−0.0124** (p_BH 0.0177) and `C_mass` **−0.0376** (p_BH 0.0201) — both statistically non-zero and both signed *against* their own mechanism, so neither contributes any part of the win; `C_resid` is **+0.2969, 120.2% of it** |
| 8 | Against a strong published baseline, not only its own twin | **failed** | FoX (arXiv 2503.02130) minus the family: **−0.0004 / −0.0008 / −0.0284** at split seeds 0 / 1 / 2 — a tie, a tie, and a loss, at **131.9 s** against **514.8 s** per cell |
| 9 | Matched on compute, not only on tokens and parameters | **failed** | a twin grown to hidden 512 (9,976,320 params) at the same steps and tokens finishes in **226 s** and beats the family by **0.1672**. Before that: the arms match at 3,538 steps and matched tokens, but wall clock is **514.8s** for (f) against **76.6s** for (a), n = 5 each, ranges 512.4–519.4 and 75.7–77.6 — a ratio of **6.72×** |

**Phase J hour one settled condition 2's loss leg, and against the family.** The
effect reproduces and is now explained — by a published forget gate that
reaches it at 3.9x less wall clock. A plainly larger twin beats it at under half
its clock. Condition 2 can no longer pass through language-model loss; only a
capability the forget gate lacks (the DO and JEPA rows of Phase J) can reopen it.

**Hour two tried DO, and it did not reopen.** Trained with every stream from its
own chain and read by a frozen linear probe, the arms score 0.1054 (twin), 0.0954
(FoX) and 0.0986 (family) on 50 intervened worlds, where a three-line Dirichlet
plug-in reading the same 128 tokens scores **0.0741**. Nobody clears the floor, so
condition 2 stays closed *by absence*, not by a passed kill: "consequence" is not
struck. The end-to-end head (R-DO-E2E) scored 0.1057 / 0.1028 / 0.1032 — on the
null. **Both reads were of arms that never learned the task:** 150 steps, next-token
loss 1.94–1.97 nats against 1.720 for an online Dirichlet count and 2.079 for a
unigram. Ruled **DO untested**, not dead. Admissibility gate for every future DO
row: held-out chain NLL ≤ 1.720 before any read counts. Phase J closes as a kernel
phase; the forget gate is the only survivor.
An earlier DO row trained on a single chain was ruled **void**: every arm sat on
the no-update null 0.07226 to four decimals.

**Correction.** Earlier versions of this page scored condition 2 at 6 of 8, then
6 of 9. Gates 1, 2, 3, 4 and 6 were the only ones marked done — five, not six. The
headline was 9 of 18 (50%), then 9 of 19 (47%), not 10 of 18 (56%) and 10 of 19
(53%). The count was wrong by one from the first version.

**Gate 9 did not exist when this page was first written.** An adversarial check
of the grid found it: `C_win` carries an unpriced confound as load-bearing as
the parameter confound already carried beside `C_mass`. The win is token-matched
and parameter-matched, and the winning arm burns 6.72x the clock to get it. That
is wall clock for an unfused operator against fused SDPA on this implementation,
not a measured FLOP count — but unpriced either way, and it is the exact axis
condition 3 names. The denominator grew from 8 to 9 because of it, which is why
this condition fell from 75% to 67% while nothing about the measurements got
worse.

One arithmetic caution carried from the same check: *`C_resid` is
distinguishable from `C_win`* is not a second finding. `C_resid − C_win ≡
−(C_phase + C_mass) = 0.0500` identically, so it is one statement made twice and
is not counted twice here.

---

## Condition 3 — carries the weight of self-attention or JEPA

**1.5 of 5 gates — 30%**

| # | gate | state | evidence |
|---|---|---|---|
| 1 | Runs at transformer scale | **not done** | 128 hidden / 3 layers / 8 heads / ~725k params |
| 2 | Cost competitive with FlashAttention | **not done** | no kernel; the carry-collapse theorem retired the kernel claim — a single-token path gate telescopes to a per-key vector, one head dimension in stock SDPA at 1.084×. Now carries measured numbers: **6.72×** the twin's wall clock at equal steps (family), and the forget gate alone runs **1.99×** an explicit-mask twin and **2.84×** is_causal per step (R-K5). This torch build has **no flash attention**; every "fused" figure here is the memory-efficient kernel, so K5 is deferred to a flash build. Addendum L's SU(2) fold, measured on an implementation that computes the contract's product, runs at **1.03–1.09×** forward and **1.08–1.09×** forward+backward against that build's fastest SDPA (EFFICIENT_ATTENTION), GPU re-run by the Inspector — a cost with no trained capability behind it yet |
| 3 | A prediction or consequence capability self-attention lacks | **not done** | the prediction leg answered NO: committor resolution tied by eight bins of piece count taken from its own input |
| 4 | The block-summary path proven exact | **done** | dense vs bucketed at 8.9e-16 / 3.2e-13 / 2.6e-10 for β = 1 / 0.5 / 0, block 8, 15 closed gates |
| 5 | Formal backing | **half** | 12 Lean files, **6 `sorry`** remaining |

---

## Workstream breakdown

| stream | done / total | % | note |
|---|---|---|---|
| Paired 4×5 grid | 20 / 20 cells | **100%** | CRN digests verified from the run records, identical across arms at all five seeds; pre-registered KILL did not fire |
| Domains at full seed count | 2 / 3 | **67%** | the third is one seed and is recorded as one |
| Seeds landed across domains | 11 / 15 | **73%** | if every domain carried five |
| Seat producer directories reachable in-tree | 22 | **—** | no denominator; this is a count, not a fraction |
| Lean obligations closed | 6 of 12 files carry a `sorry` | **50%** by file | 6 `sorry` total |
| Commits pushed this run | 45 / 45 | **100%** | nothing held locally |
| Phase I.1 sections written | 22 | **—** | §22 *Rows still out* is stale: R3 landed and Kaggle has since run three domains |
| Abstention decile test | ran, then **retired** | **NEITHER** | decile gradient 0.113 → 0.341, top-3 share 40.6% < 50%; control failed, then R-STRAT showed abstention survives stratification (ρ with difficulty 0.0125) |
| Phase J hour one | 17 rows, 0 struck | **—** | R-FoX PASS ×3, R-TEMP FAIL, R-COMP PASS, R-STRAT survives; three contract figures did not reproduce |
| Phase J hour two | 10 rows, 0 struck | **—** | R-XFER PASS (abstention retires), R-K5 FAIL 1.99×; R-DO-LM, R-DO-IC, R-DO-E2E all **void** — one chain, then 150 steps; DO untested |

---

## What the percentages do not mean

A gate marked **done** means one measurement passed with its control and its
provenance recorded. It does not mean the question behind the gate is closed.
Condition 2 stands at 56% with three gates measured false, which is the
uncomfortable and accurate reading: the effect is well measured, now understood,
and owned by prior art.

Three counts belong beside the percentages and have no denominator:
**6 misattributions** found and corrected, **16 pre-registered checks that could
not have failed** and **3 that could not have passed** catalogued, and **5 beds that failed at their own floors**.
Those are the error rate this project is measuring against itself, and a status
page that omits them reads better than the work deserves.
