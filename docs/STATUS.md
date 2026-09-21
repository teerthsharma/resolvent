# Status — done against total

Read at 2026-09-22. Every percentage on this page is a count of named gates,
not a judgement of effort. Each table lists its own denominator so a reader can
disagree with the denominator rather than with the number.

The north star is **conjunctive**: three conditions that must all hold. A
conjunctive goal is not the average of its legs — it is the weakest leg. Both
numbers are given below, and the weakest leg is the honest one.

---

## Headline

| reading | value |
|---|---|
| gates passed, all three conditions | **10 of 18 — 56%** |
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

**6 of 8 gates — 75%**

| # | gate | state | evidence |
|---|---|---|---|
| 1 | A win exists | **done** | −0.2473 eval NLL, 5 seeds |
| 2 | At matched parameters, excess reported not hidden | **done** | 783 excess per the same per-block formula on all three domains |
| 3 | More than one domain | **done** | TinyStories −0.2473, WikiText-103 −0.3285, codeparrot −0.3301 |
| 4 | Against the strongest baseline lacking the property | **done** | the softmax twin routes through a genuine `scaled_dot_product_attention` call site, monkeypatch-verified |
| 5 | A seed interval on every domain | **not done** | 5 seeds / 5 seeds / **1 seed** — the third domain is a data point, and its record says so |
| 6 | Survives a varied-split protocol | **done** | `C_win` = **+0.2500**, sd 0.0167, 4/4 split seeds positive, CRN-paired |
| 7 | The mechanism identified | **not done** | `C_phase` **−0.0114** and `C_mass` **−0.0327**, both 4/4 seeds *against* their own mechanism; `C_resid` is **+0.2941, 118% of the win** |
| 8 | Against a strong published baseline, not only its own twin | **not done** | every comparison to date is internal |

Gate 7 is the load-bearing failure on this page. The effect reproduces; nothing
yet explains it.

---

## Condition 3 — carries the weight of self-attention or JEPA

**1.5 of 5 gates — 30%**

| # | gate | state | evidence |
|---|---|---|---|
| 1 | Runs at transformer scale | **not done** | 128 hidden / 3 layers / 8 heads / ~725k params |
| 2 | Cost competitive with FlashAttention | **not done** | no kernel; the carry-collapse theorem retired the kernel claim — a single-token path gate telescopes to a per-key vector, which is one head dimension in stock SDPA at 1.084× |
| 3 | A prediction or consequence capability self-attention lacks | **not done** | the prediction leg answered NO: committor resolution tied by eight bins of piece count taken from its own input |
| 4 | The block-summary path proven exact | **done** | dense vs bucketed at 8.9e-16 / 3.2e-13 / 2.6e-10 for β = 1 / 0.5 / 0, block 8, 15 closed gates |
| 5 | Formal backing | **half** | 12 Lean files, **6 `sorry`** remaining |

---

## Workstream breakdown

| stream | done / total | % | note |
|---|---|---|---|
| Paired 4×5 grid | 17 / 20 cells | **85%** | four split seeds complete, fifth running |
| Domains at full seed count | 2 / 3 | **67%** | the third is one seed and is recorded as one |
| Seeds landed across domains | 11 / 15 | **73%** | if every domain carried five |
| Seat producer directories reachable in-tree | 22 | **—** | no denominator; this is a count, not a fraction |
| Lean obligations closed | 6 of 12 files carry a `sorry` | **50%** by file | 6 `sorry` total |
| Commits pushed this run | 36 / 36 | **100%** | nothing held locally |
| Phase I.1 sections written | 22 | **—** | §22 *Rows still out* is stale: R3 landed and Kaggle has since run three domains |
| Abstention decile test | pre-registered, unrun | **0%** | thresholds fixed and committed before its checkpoints exist |

---

## What the percentages do not mean

A gate marked **done** means one measurement passed with its control and its
provenance recorded. It does not mean the question behind the gate is closed.
Condition 2 stands at 75% with its mechanism gate failing, which is the
uncomfortable and accurate reading: the effect is now well measured and not at
all understood.

Three counts belong beside the percentages and have no denominator:
**6 misattributions** found and corrected, **16 pre-registered checks that could
not have failed** catalogued, and **5 beds that failed at their own floors**.
Those are the error rate this project is measuring against itself, and a status
page that omits them reads better than the work deserves.
