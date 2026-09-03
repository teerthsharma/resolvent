# V20 R15 it.3 — MERCURY (LESTRADE). THE ARENA'S PRICE

Contract slot it.5, pulled forward: the it.2 sizing defect blocks all of Phase C.
Three tasks, all arithmetic, all bound by one node —
`tests/mercury/test_v20_r15_it3_arena_price.py`, 11 assertions against
`tests/mercury/arena_price.py`. **RED first, verbatim:**

```
tests\mercury\test_v20_r15_it3_arena_price.py:13: in <module>
    from tests.mercury.arena_price import (
E   ModuleNotFoundError: No module named 'tests.mercury.arena_price'
=========================== short test summary info ===========================
ERROR tests/mercury/test_v20_r15_it3_arena_price.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.52s
```

Then green: `[RUN]` `python -m pytest tests/mercury/ -q` → **89 passed in 9.47s**
(78 pre-existing + 11 new). Machine: this box, the certified device below.

**One correction to this office's own first draft, recorded rather than
overwritten:** `arena_seconds(125)/3600` was written as `0.6636` and the node
read `0.6635`. The expectation was wrong, not the module.

**DEVICE, confirmed against `V16_DEVICE_CERT.md:112,117` `[RUN]`:**
`NVIDIA GeForce RTX 4060 Laptop GPU sm_89 8187 MiB, torch 2.5.1+cu121,
cuda True`. The cert reads `sm_89, 7.996 GiB` — 8187 MiB is the same number in
different units. Certified.

---

## TASK A — CLAUSE (1) AT N=8, AND THE TWO EXITS

### A.0 The coordinator's table reproduces. All of it.

Recomputed from `results/v17k_r4_retake.jsonl` — 24 `t="cell"` rows,
`floor_1 = 0.7071067811865476` read from the file's own header, crossing tested
with the shipped `<= floor` comparison `[RUN]`:

| arm | crossings | rate | CP-lower (two-sided 95% Clopper–Pearson) | CP-upper | clause (1) |
|---|---|---|---|---|---|
| `arm_pl` | 5/8 | 0.6250 | **0.2449** | 0.9148 | NOT MET |
| `arm_smprime` | 1/8 | 0.1250 | 0.0032 | 0.5265 | NOT MET |
| `softmax` | 0/8 | 0.0000 | 0.0000 | 0.3694 | NOT MET |

Achievable CP-lower at `N=8`: `8/8 → 0.6306`, `7/8 → 0.4735`, `6/8 → 0.3491`,
`5/8 → 0.2449`. **Only a clean sweep clears `0.5`.** Confirmed.

### A.1 But `N=65` is convention-dependent, and the convention picked rounds the rate UP

`n_for_cp_lower(0.625, 0.5, rule=...)` `[RUN]` — the smallest `N` clearing 0.5
while "holding the observed rate 0.625" depends entirely on how `0.625 × N` is
turned into a whole count:

| integer rule | first N | k | realised rate | CP-lower |
|---|---|---|---|---|
| `ceil` | **58** | 37 | 0.6379 | 0.5012 |
| `round` | **65** | 41 | 0.6308 | 0.5020 |
| `floor` | **72** | 45 | **0.6250** | 0.5030 |

`N=65` is right under round-half and only under round-half. `41/65 = 0.6308`
is **above** `arm_pl`'s observed rate; the only rule that holds `0.625` exactly
is `floor`, and it lands on **`N=72`**. This is not a correction to the
coordinator's arithmetic — it is a correction to the *claim attached* to it.

### A.2 And `N=65` is a coin flip, not a design

`N` chosen so that the point estimate clears the bar buys **50% power**. With
`K ~ Binomial(N, 0.625)` `[RUN]`:

| N | smallest k clearing 0.5 | P(clause (1) fires) |
|---|---|---|
| 58 | 37 | 0.478 |
| **65** | 41 | **0.517** |
| 72 | 45 | 0.552 |
| 100 | 61 | 0.663 |
| **125** | — | **≥ 0.80** (first such N) |
| 200 | 115 | 0.937 |

**`N=65` is a 51.7%-chance experiment.** If the round intends clause (1) to fire
when `arm_pl` really crosses at 0.625, the number is **`N=125`**, not 65. This
is `MISTAKES.md` M-9's neighbour: sizing on a point estimate as if the estimate
carried no error.

---

### EXIT A — RAISE N. THE PRICE, AND IT IS NOT THE BLOCKER ANYONE THINKS

**Measured basis, `[READ] V17_R4_RETAKE_PRICE.md:194-196`** — verified line by
line, the coordinator's `:194` mis-cite of it.2 does not recur:

```
194  | `arm_smprime` | **15.970** | 17.839, 14.101 |
195  | `arm_pl`      | **1.614**  | 1.622, 1.605   |
196  | `softmax`     | **1.497**  | 1.496, 1.497   |
```

**Units confirmed** from the table header at `:191` — *"s / 150 steps, mean of
2"* — six CUDA cells, one process, `t="wall" = 41.842 s` (`:186`). Fixed
non-cell cost **3.5 s per process** (`:198-200`); per-cell overhead outside the
training loop **under 0.05 s**, already inside the per-cell figure.

**The extrapolation, stated in full and nothing hidden in it:**
`GPU-seconds = (sum of per-cell seconds) × N + 3.5`. Cells run sequentially in
one process, which is how the retake was taken. **Peak bytes do not scale with
`N`** — `N` is seeds, not batch — so the 8187 MiB bound is untouched at every
`N` below.

| design | arms | GPU-seconds | wall | **GPU-hours** | fits the box? |
|---|---|---|---|---|---|
| contract today, `N=8` | 3 | 156.1 | 2.60 min | 0.0434 | yes |
| **EXIT A, `N=65`** | 3 | **1 243.8** | **20.7 min** | **0.3455** | **yes** |
| EXIT A, `N=65` + W2 | 4 | 1 348.7 | 22.5 min | 0.3746 | yes |
| EXIT A powered, `N=125` | 3 | 2 388.6 | 39.8 min | 0.6635 | yes |
| EXIT A powered, `N=125` + W2 | 4 | 2 590.4 | 43.2 min | 0.7195 | yes |

W2 `arm_phase` is priced at `arm_pl`'s 1.614 s — the nearest measured basis,
because `ceq/arm_phase.py:180` returns `arm_pl`'s real softmax matrix times
`phase_factor(theta)`. **[DERIVED, not measured for this arm.]**

**Control on the extrapolation:** at `N=8` the model returns **156.15 s**
against the price doc's own measured-driven total of **158.74 s** (`:207`) —
**1.63% low**, inside the doc's own 20% adjudication threshold. The model is
low because it charges one fixed cost where the doc charged 6.10 s.

**VERDICT ON EXIT A: it fits the local box with room to spare — 0.35 GPU-hours
at `N=65`, 0.72 at the honestly-powered `N=125`.** it.2 filed `N=65` as *"an 8x
arena cost"*. That multiplier is correct and the inference from it is not: 8×
of 2.6 minutes is 21 minutes. **On BED-M, `N` is not a cost constraint at any
value the statistics ask for.**

**THE LIMIT, AND IT IS THE ONLY REAL ONE.** The above prices **BED-M alone**.
Phase C is three beds (`CEQ_V20_R15_CONTRACT.md:116-118`): BED-M, BED-K(a)
`d=20`, and the chess witness. **No measured per-cell second exists in this
tree for either of the other two shapes** — the retake measured BED-M's shape
(`t*=2, n_train=2048, s=64, d_model=16`) only. Their price is **unpriceable
today**, and no estimate is offered for them: this office does not extrapolate
a cost across a shape it has never seen run.

---

### EXIT B — RESTATE THE CLAUSE. IT COSTS ZERO, AND IT IS ALREADY MET

**Instrument: Fisher exact, one-sided.** Chosen, not defaulted to. The skyline
cell is an exact **`0/8`**; a two-proportion z-test's variance term
`p(1−p)/n` **vanishes** there and the statistic is undefined (and every
continuity-corrected patch is a fudge over a zero cell). Fisher conditions on
the margins and needs no variance estimate, so it is valid at `0/N` where the
z-test is not. One-sided because the clause is directional — the arm is claimed
*above* the skyline, not merely different from it.

`[RUN]` on the 2×2 `[[5, 3], [0, 8]]`:

| comparison | one-sided p | two-sided p | fires at α=0.05? |
|---|---|---|---|
| `arm_pl` 5/8 vs `softmax` 0/8 | **0.012821** | **0.025641** | **YES, both** |
| `arm_smprime` 1/8 vs `softmax` 0/8 | 0.500000 | 1.0 | no (**the control**) |

**What `N` does the restated clause need? `N=8`. It is already met, on cells
already in `results/v17k_r4_retake.jsonl`.**

**Price of EXIT B: 0 GPU-seconds, 0 wall, 0 new cells.** No arena re-run, no
device time, no Kaggle. The whole of it is a contract edit.

And it is powered, which the absolute-CP form at `N=65` is not `[RUN]`:

| N | P(reject) at true rate 0.625 vs skyline 0 |
|---|---|
| 6 | 0.596 |
| 7 | 0.757 |
| **8** | **0.863** |
| 10 | 0.962 |

**Planted control on that power figure**: at half the rate (`p = 0.3125`) the
same `N=8` reads **below 0.60**, so `0.863` is a reading of the rate and not a
property of `N=8`.

**Both exits repriced, neither chosen. The numbers, side by side:**

| | EXIT A (`N` to 65) | EXIT A (powered, `N=125`) | EXIT B (restate) |
|---|---|---|---|
| GPU-hours, BED-M | 0.3455 | 0.6635 | **0.0000** |
| new cells | 171 | 351 | **0** |
| power at the observed rate | 0.517 | 0.803 | **0.863** |
| status today | not met | not met | **MET** |
| BED-K(a) + witness | unpriced | unpriced | unpriced |

---

## TASK B — THE W2 GATE. IT CAN RUN, AND IT COSTS UNDER 20 SECONDS

### B.1 Can it run? MEASURED THIS ITERATION, both regimes, on the certified device

`V16_ARM_SMPRIME.md:567-593` §9 is **reproduced exactly** `[RUN]`, and it is
only half the picture:

```
  strict warn_only=False       arm_phase.operator     RAISES  cumsum_cuda_kernel does not have a deterministic implementation...
  strict warn_only=False       arm_phase.scan_phase   RAISES  cumsum_cuda_kernel does not have a deterministic implementation...
  round regime warn_only=True  arm_phase.operator     OK  warns=2
  round regime warn_only=True  arm_phase.scan_phase   OK  warns=2
  150 forwards arm_phase.operator   0.0611 s
```

Warning source, at the two `cumsum` lines: `ceq/arm_phase.py:128`
(`s - torch.cumsum(torch.log(m), dim=-1)`) and `:138`
(`phi = torch.cumsum(theta, dim=-1)`). Note this is worse than `arm_smprime`,
whose forward is `cumprod` (deterministic) and which only meets `cumsum` in the
backward — **W2's raise is in the FORWARD**.

**And the round's regime is not strict.** `results/v17k_r4_retake.jsonl`'s own
header records it:

```
"deterministic_algorithms": true, "deterministic_warn_only": true,
"cublas_workspace_config": ":4096:8", "torch": "2.5.1+cu121"
```

`ceq/hf/modeling_ceq.py:1058` states the same regime in prose: *"under the
round's `warn_only=True` regime it warns and proceeds."*

**RULING: the W2 training cell CAN run.** The determinism regime that produced
every arena cell in hand is `warn_only=True`, under which `arm_phase` warns
twice and returns. The §9 RAISE is real and is measured under
`warn_only=False`, which is `scale/r10_capacity_sweep.py:253`'s regime — a
script whose `--arm` choices are `softmax|pivot_*` and which cannot run
`arm_phase` at all.

**The cell it produces is not strict-deterministic, and that must be journalled
on the cell**, not inferred. Two `cumsum` calls run non-deterministically; the
cell is not bitwise reproducible and may not be pooled with a strict cell.

### B.2 The price: one cell, matched params, 150 steps, BED-M's shape

| basis | per-cell s | + fixed 3.5 s | total |
|---|---|---|---|
| `arm_pl` (`arm_phase` = `arm_pl` magnitude × a phase factor) | 1.614 | | **5.114 s** |
| `arm_smprime` (upper bound, the expensive wing) | 15.970 | | **19.470 s** |

**Under 20 GPU-seconds either way.** The W2 gate — the thing blocking the it.4
freeze and with it the whole wing list — costs **less than half a minute of
wall clock on this box.** It is not a cost decision. It has never been one.

`[DERIVED]` both rows: no `arm_phase` training cell has ever been measured
(a journalled-`kind` grep across `results/` returns 0, SATURN it.2). The
measured forward above — 150 `arm_phase.operator` calls at `s=64` in
**0.0611 s** — says the forward is not the cost driver, so the `arm_pl` basis
is the right one and `arm_smprime` is a ceiling, not an estimate.

### B.3 If the round decides strict IS required — three routes, priced

| route | what it costs | what it gives up |
|---|---|---|
| **REROUTE — run it on CPU.** `cumsum` has a deterministic CPU kernel; the raise is CUDA-only | **unpriced** — no CPU per-cell second exists for `arm_phase` at BED-M's shape. `V17_R4_RETAKE_PRICE.md`'s CPU lane would have to be re-measured | cross-device pooling is refused (`refuse_cross_device_pool`), so a CPU W2 cell cannot sit in a CUDA arena table |
| **REPRICE — replace `scan_phase`'s `cumsum` with a `cumprod` path product of unit-complex gates.** The exact repair that made `arm_smprime` strict-clean (§9: *"cumprod has a deterministic CUDA kernel; cumsum does not"*) | one code change plus one cell, ~5 s | the phase is then represented on the unit circle, not on the additive line — it changes W2's operator, and W2's whole claim is the phase |
| **RETIRE — run under `warn_only=True` and journal the cell as non-strict.** Zero extra cost; the regime every arena cell already ran under | the W2 cell is not bitwise reproducible and may not be pooled with strict cells | — |

**Cheapest route that keeps W2 a wing: RETIRE strictness for this cell.**
5.1 seconds, and it matches the regime the other 24 cells already ran under.

---

## TASK C — MARS's STRIKE 6, ADJUDICATED ON ARITHMETIC ONLY

All three statistics recomputed from the journal, with `seconds_to_floor`
copied **verbatim** from `tests/saturn/test_v20_r15_wing_rubric.py:433-448` so
the comparison is against the shipped instrument `[RUN]`:

| # | statistic | W3 `arm_pl` | W1 `arm_smprime` | ratio | **order-invariant?** |
|---|---|---|---|---|---|
| 1 | `seconds_to_floor`, **file order** | 1.884 | 47.048 | **24.97×** | **NO** |
| 2 | `seconds_to_floor`, **best-first** | 1.759 | 14.852 | **8.44×** | **YES** — it canonicalises |
| 2′ | `seconds_to_floor`, **worst-first** | 7.113 | 129.287 | 18.18× | YES — canonicalises the other way |
| 3 | **expected cost to a crossing** | 2.8358 | 118.8160 | **41.90×** | **YES** |

**Every one of MARS's numbers reproduces.** `47.048 = 16.164 + 16.032 + 14.852`
(W1's first three rows). `25.0×` and `8.4×` both land — 24.97 and 8.44. His
repriced `1.78 × 8/5 = 2.85` reads 1.7724 × 1.6 = **2.8358** (he rounded the
mean); his `14.85 × 8/1 = 118.8` reads **118.8160**. All within his own
rounding.

**THE ADJUDICATION, and it is one sentence.** Statistic 1 is a function of the
**sequence** of rows; statistics 2, 2′ and 3 are functions of the **multiset**.
A permutation moves 1 and cannot move the others. **MARS is right: `25.0×` is a
row-order artifact.**

**Two things this office adds, both arithmetic.**

**First, `24.97×` is not even the extreme.** Under permutation the shipped
statistic spans, per arm: W3 `[1.759, 7.113]` (4.04×), W1 `[14.852, 129.287]`
(8.70×). Permuting the two arms independently, the *ratio* spans
**`2.09×` to `73.50×`** — a factor of 35 between the smallest and largest value
the same cells can be made to report. `24.97×` sits in the middle of that
range, not at its top. Best-first's `8.44×` sits near its bottom and is the
**best case**, not the neutral one.

**Second — and it cuts against the reprice as well as the original.** MARS
flags survivorship on the first-crossing statistic. Statistic 3 does not escape
it: `softmax` crosses `0/8`, so `expected_cost_to_crossing` returns **`None`**
for the skyline. **Criterion (3) cannot rank an arm that never crosses.**
It is undefined for the skyline, undefined for W2 (no cell exists at all), and
defined for exactly two of the four objects the arena means to rank. The node
carries the planted control that proves the `None` is a reading: plant one
crossing into a `softmax` cell and the statistic returns a number.

**WHAT DOES NOT MOVE — and this is the whole of what survives.** The *direction*
holds under every ordering tested: `arm_pl < arm_smprime` on file order,
best-first, worst-first, and on the order-invariant statistic. The magnitude is
an artifact; the ranking is not. **This office prices and does not interpret
whether that ranking "counts" — that is the Inspector's call, and the numbers
are above.**

---

## WHAT THIS OFFICE HANDS OVER, AND DOES NOT DECIDE

1. **`N=65` is arithmetically correct under round-half and is a 51.7% experiment.**
   The honestly-powered figure is `N=125`; the rate-preserving figure is `N=72`.
2. **EXIT A costs 0.35 GPU-hours on BED-M and fits this box.** The "8× arena
   cost" is 21 minutes. The unpriced part is BED-K(a) and the witness, and it is
   unpriced because no cell of either shape has ever been measured.
3. **EXIT B costs nothing and is already satisfied**, one-sided `p = 0.012821`,
   two-sided `p = 0.025641`, powered 0.863 at `N=8`, with a control that
   correctly declines to fire for `arm_smprime`.
4. **The W2 gate costs 5.1 GPU-seconds and CAN run** under the regime every
   arena cell already ran under, measured both ways this iteration. Under strict
   mode it raises in the forward, and three routes are priced above.
5. **STRIKE 6 stands on the arithmetic.** File order is the only non-invariant
   statistic of the three; its span under permutation is 2.09×–73.50×; and
   criterion (3) is undefined for two of the four objects it must rank.

**The exit is not chosen here.** Both are priced.

**LIMITS.** BED-K(a) and the chess witness have no measured per-cell cost in
this tree and are excluded from every total above — EXIT A's full-arena price is
BED-M's figure **plus two unknowns**. Every GPU-hour figure is
`sum(per-cell seconds) × N + 3.5 s`, a multiplication of six measured seconds
taken in a single 6-cell process; it assumes sequential cells in one process and
carries the price doc's own +11.1% measured-vs-law disagreement (`:207-209`)
as its uncertainty floor. `arm_phase`'s per-cell second is derived from
`arm_pl`'s, never measured. The W2 forward probe (0.0611 s / 150 calls) is a
forward only — no backward, no optimiser, no eval — and is evidence about which
basis to use, not a cell cost. Power figures assume the observed rate is the
true rate, which is the same assumption that makes `N=65` a coin flip. **No git
write, no Kaggle contact, no training run started.**
