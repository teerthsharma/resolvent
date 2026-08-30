# The it.15 dual oracle: the Kirchhoff route across the corpus, and a halt that means it

SATURN, v-main.3M script iteration 15. The script line, verbatim:

> SATURN: generator + dual oracle; per instance assert
> `|u_absorbing - u_kirchhoff|_inf <= 1e-10` (matrix-tree cofactor route) --
> disagreement halts the corpus, not the reading.

Code `scale/r10_dual_oracle.py`, rows `results/r10_it15_dual_oracle.jsonl`.
Reproduce every number below with

```
python -m scale.r10_dual_oracle            # demo(), assert-based, ~3 s
python -m scale.r10_dual_oracle --write    # the corpus, 199.9 s, writes the .jsonl
```

Python 3.11.9, numpy 1.26.4, Windows 11 x86-64, branch `feat/r9-causal-consequence`
at `bf2a769`. Arm `reproduce` uses JUPITER's seeds `0x3a140000 + index`; arm
`widen` uses `0x3a150000 + index`; every seed is printed in its row. Re-run and
every MEASURED field is identical -- nothing reads a hash seed or a thread count
-- but the `.jsonl` is not byte-identical, because each row carries a `seconds`
wall-clock field. The two runs behind this report gave the same
`max_gap = 1.5543122344752192e-15` and differed only in timing (182.7 s against
199.9 s, a sibling job on the host between them).

`scale.vram_gate.preflight(0, name='r10-dual-oracle', host_mib=256)` returned
`HOST FITS -- needs 320 MiB, 7033 MiB available` before the corpus ran. One
process, no training, no GPU, no capacity sweep. Peak is one `1024 x 1024`
Laplacian plus one `(|I|+1) x (|I|+1)` LU workspace.

---

## 0. THE HEADLINE, BEFORE THE EVIDENCE

**The dual oracle holds across the whole corpus at `1.554312e-15`, and the most
useful number in this report is the one where it FAILS to fire.** Both routes
carry the `deg + 1` miscount at once and agree to `3.331e-16` — 0 of 17
rejections. That is the pair's blind spot measured rather than argued away, and
it is the honest limit on every agreement figure below.

| count | value |
|---|---|
| instances | **31** |
| **passed** | **29** |
| **failed** | **0** |
| **errored** | **0** |
| **inapplicable** | **2** |
| never generated (own category, §5) | 1 |
| dual-checked | 29 |

1. **Max disagreement `1.554312e-15`** on `reproduce-n1024-d4-t0.95`
   (`n = 1024`, `|B| = 80`, seed `0x3a14000a`), five orders inside the imported
   `1e-10`. §2.
2. **The halt is literal and was demonstrated**: a planted disagreement on
   instance 1 of a 3-instance corpus emitted **0 rows**, not 2. §3.
3. **The planted defect is rejected 29/29 and 17/17** in the two single-route
   directions, gaps `2.916667e-01` to `6.301520e-01`. **The shared plant is
   rejected 0/17.** §4.
4. **The generator reproduces the upper-half clustering exactly** — 12/12 in
   `[0.9434514476, 0.9499393658]`, `lambda_2` delta `0.0` against it.14. Widening
   is possible and cost 1 band miss in 17 attempts plus 64% more second-oracle
   work per rung. §1.

---

## 1. THE GENERATOR — THE CLUSTERING REPRODUCES, AND THE BAND WAS NOT COVERED

### 1.1 The generator is JUPITER's, imported, and that is checked not claimed

`scale/r10_dual_oracle.py:104` is `from scale import r10_corpus_spec as spec`.
The graph draw, `admissible`, `blocks`, `spectrum`, `stratify` and the absorbing
oracle are all his; nothing was rewritten. That distinction is load-bearing — a
second oracle validated against a *reconstruction* of the first one's generator
measures something weaker than it appears to — so it is verified rather than
asserted. `reproduce_check` (`scale/r10_dual_oracle.py:339-371`) diffs arm
`reproduce` against `results/r10_it14_corpus_spec.jsonl` per rung:

```
"kind": "reproduce_check", "rungs": 12,
"max_lambda_2_delta": 0.0, "boundary_sizes_identical": true
```

**Zero, not "small".** Same graphs, same node order, same boundary sets, same `g`
off the same rng stream, and the arm consumed **121 eigensolves** — the same 121
his §3 reports. The only new codepath in this round is the second oracle.

A static-analysis pass flagged `scale/r10_corpus_spec.py` as an unknown import
symbol. It is not: the module resolves, exports every name used, and
`git ls-files` shows it **untracked**, which is why an index-driven analyzer
cannot see it. No workaround was applied and nothing was re-implemented.

### 1.2 The clustering reproduces, and the corpus it.14 shipped does not cover its band

**Yes — the generator reproduces the upper-half clustering exactly.** All twelve
rungs of arm `reproduce` land in `[0.9434514476, 0.9499393658]`:

| `lambda_2` bin | `[0.90,0.91)` | `[0.91,0.92)` | `[0.92,0.93)` | `[0.93,0.94)` | `[0.94,0.95)` |
|---|---|---|---|---|---|
| **arm `reproduce`, 12 rungs** | **0** | **0** | **0** | **0** | **12** |
| arm `widen`, 16 rungs | 3 | 3 | 4 | 3 | 3 |

**So it must be said plainly: a corpus built by that rule does not cover the band
it claims.** It covers the top fifth of it. The mechanism is not a bug —
`stratify` bisects for the smallest `|B|` whose `lambda_2(k)` clears `hi = 0.95`,
`lambda_2(k)` is monotone decreasing in `k`, and the first `k` that clears the
upper edge is still near it. A label carrying "stratified to `lambda_2 in
[0.90, 0.95]`" describes a design intent that the emitted corpus does not
realise, and any downstream reading of a rate across that band rests on 12 points
spanning `0.0065` of it.

### 1.3 The widening, and what it cost

The rule change is **one argument** — `hi = target` instead of `hi = BAND_HI` —
against a declared ladder `(0.905, 0.915, 0.925, 0.935, 0.945)`
(`scale/r10_dual_oracle.py:123`), fixed before any rung was drawn and not
adjusted after seeing where they landed. 17 attempts, 16 rungs,
`[0.9033074288, 0.9449749999]`, all five bins occupied. The cost, in three parts:

| | arm `reproduce` | arm `widen` |
|---|---|---|
| rungs emitted / attempted | 12 / 12 | 16 / 17 |
| eigensolves | 121 (10.1 per rung) | 165 (9.7 per rung) |
| Kirchhoff solves | 324 (27.0 per rung) | 707 (44.2 per rung) |
| wall clock | 53.3 s | 91.7 s |

- **Stratification itself costs nothing extra.** 9.7 eigensolves per rung against
  10.1; bisection depth is `ceil(log2 n)` regardless of the target.
- **One band miss in 17 attempts (5.9%).** `widen-n128-d4-t0.905`, seed
  `0x3a150000`: `lambda_2(k)` stepped from above `0.905` straight to
  `0.8943072442`, below the band. No integer `|B|` lands inside. It is recorded
  `status="band_miss"` under its own `kind` and is **not** an instance — see §5.
- **The real cost is the second oracle, and it is structural.** Lower targets
  need larger `|B|`, and the Kirchhoff route costs one solve per boundary node:
  `n = 1024` needs `|B| = 64` at `t = 0.95` and `|B| = 124` at `t = 0.905`.
  44.2 solves per rung against 27.0 — **64% more** second-oracle work to cover
  the band than to cluster at its top.

---

## 2. THE DUAL ORACLE, PER INSTANCE

### 2.1 The route, and it is the cofactor one

`kirchhoff.harmonic_measure` computes `omega_x = M_xa / M_aa` and applies exactly
where `|B| = 2` and `g = (1,0)`. This corpus has `|B|` from 4 to 124 and a random
`g`, so the identity is used one boundary node at a time
(`scale/r10_dual_oracle.py:170-198`). Ground at `B \ {b}` — delete those rows and
columns of the symmetric unnormalised `L = D - A`, keeping `b` as an ordinary
column — solve `K_b z = e_b` once, and read

```
omega^{(b)}_v = z_v / z_b = F(v ~ b ; B) / F(B)
```

the ratio of spanning `|B|`-forests (one boundary node per tree) putting `v` in
`b`'s tree, to all such forests. `u_kirchhoff = sum_b g_b omega^{(b)}`.

**That the ratio is a forest ratio is tied down against literal enumeration, not
asserted.** `demo()` step 2 runs the route on a 5-node graph and compares against
`kirchhoff.brute_force_separating_forests`: counts `8/13`, `7/13`, `4/13`,
agreement **`0.000000e+00`**. `demo()` step 1 checks the generalisation reduces to
the shipped `harmonic_measure` on the `|B| = 2` path at `1.110223e-16`.

### 2.2 The tolerance, and where it came from

**`1e-10`, imported as `AGREEMENT_TOL = kirchhoff.AGREEMENT_TOL`
(`scale/r10_dual_oracle.py:113`), defined at `scale/kirchhoff.py:106`.** Not
re-picked here, not narrowed after seeing the result. `demo()` step 0 asserts the
identity of the object, so a local re-definition would fail the self-check rather
than pass silently. The window that set it there: largest gap between two correct
oracles `9.636736e-14`, smallest gap the planted degree defect produces
`1.749951e-01`.

The window is **wider** on this corpus than where the constant was calibrated —
largest clean gap `1.554312e-15`, smallest planted gap `2.916667e-01`, a ratio of
`1.876e+14` — so the imported constant is held to rather than tightened. Deriving
a sharper tolerance from numbers already seen would be fitting the instrument to
its own result.

### 2.3 The measured disagreement

**29 instances dual-checked, 29 agree, max `1.554312e-15`.**

| | value |
|---|---|
| **max `\|u_absorbing - u_kirchhoff\|_inf`** | **`1.554312e-15`** |
| **achieved on** | `reproduce-n1024-d4-t0.95`, `n=1024`, `\|B\|=80`, seed `0x3a14000a` |
| min over the corpus | `1.110223e-16` (`reproduce-n64-d4-t0.95`) |
| instances checked | 29 |
| tolerance | `1e-10`, `scale/kirchhoff.py:106` |
| margin | 5 orders |

The gap grows with `n` — `2.775558e-16` at `n=64` to `1.554312e-15` at `n=1024` —
which is the shape a conditioning argument requires and the same shape
`scale/kirchhoff.py:69-71` records for its own pair.

A second, single-route check runs alongside: `sum_b omega^{(b)}_v = 1`, the
absorption probabilities over a full boundary summing to one. Max deviation
**`2.220446e-15`** over the corpus. It matters because it holds the Kirchhoff
route to something *without* consulting the absorbing route, so the second oracle
is not trusted purely on the strength of the first one agreeing with it.

Per-instance rows, all 28 rungs:

| id | `n` | deg | seed | target | `lambda_2` | `\|B\|` | K solves | gap | partition dev | s |
|---|---|---|---|---|---|---|---|---|---|---|
| reproduce-n64-d4 | 64 | 4 | `0x3a140000` | 0.95 | `0.9434514476` | 7 | 7 | `2.775558e-16` | `4.441e-16` | 0.1 |
| reproduce-n64-d6 | 64 | 6 | `0x3a140001` | 0.95 | `0.9466824744` | 4 | 4 | `4.440892e-16` | `4.441e-16` | 0.1 |
| reproduce-n128-d4 | 128 | 4 | `0x3a140002` | 0.95 | `0.9455579636` | 11 | 11 | `5.551115e-16` | `5.551e-16` | 0.3 |
| reproduce-n128-d6 | 128 | 6 | `0x3a140003` | 0.95 | `0.9490558907` | 7 | 7 | `5.551115e-16` | `8.882e-16` | 0.3 |
| reproduce-n256-d4 | 256 | 4 | `0x3a140004` | 0.95 | `0.9489302427` | 19 | 19 | `9.992007e-16` | `1.332e-15` | 2.1 |
| reproduce-n256-d6 | 256 | 6 | `0x3a140005` | 0.95 | `0.9461843337` | 16 | 16 | `7.771561e-16` | `7.772e-16` | 1.7 |
| reproduce-n256-d8 | 256 | 8 | `0x3a140006` | 0.95 | `0.9455168387` | 17 | 17 | `7.771561e-16` | `8.882e-16` | 2.3 |
| reproduce-n512-d4 | 512 | 4 | `0x3a140007` | 0.95 | `0.9499393658` | 41 | 41 | `1.110223e-15` | `1.443e-15` | 4.7 |
| reproduce-n512-d6 | 512 | 6 | `0x3a140008` | 0.95 | `0.9464959515` | 27 | 27 | `8.326673e-16` | `1.665e-15` | 3.6 |
| reproduce-n512-d8 | 512 | 8 | `0x3a140009` | 0.95 | `0.9493823268` | 31 | 31 | `7.771561e-16` | `1.887e-15` | 4.2 |
| **reproduce-n1024-d4** | 1024 | 4 | `0x3a14000a` | 0.95 | `0.9494274830` | 80 | 80 | **`1.554312e-15`** | `2.220e-15` | 18.9 |
| reproduce-n1024-d6 | 1024 | 6 | `0x3a14000b` | 0.95 | `0.9498399931` | 64 | 64 | `1.165734e-15` | `1.776e-15` | 14.8 |
| widen-n128-d4 | 128 | 4 | `0x3a150001` | 0.915 | `0.9110231539` | 17 | 17 | `7.771561e-16` | `6.661e-16` | 0.6 |
| widen-n128-d4 | 128 | 4 | `0x3a150002` | 0.925 | `0.9230052120` | 21 | 21 | `6.661338e-16` | `6.661e-16` | 1.2 |
| widen-n128-d4 | 128 | 4 | `0x3a150003` | 0.935 | `0.9321344552` | 12 | 12 | `3.885781e-16` | `9.992e-16` | 0.5 |
| widen-n128-d4 | 128 | 4 | `0x3a150004` | 0.945 | `0.9424286480` | 10 | 10 | `3.885781e-16` | `7.772e-16` | 0.4 |
| widen-n256-d6 | 256 | 6 | `0x3a150005` | 0.905 | `0.9033074288` | 31 | 31 | `5.551115e-16` | `7.772e-16` | 3.7 |
| widen-n256-d6 | 256 | 6 | `0x3a150006` | 0.915 | `0.9149805570` | 29 | 29 | `6.661338e-16` | `8.882e-16` | 3.0 |
| widen-n256-d6 | 256 | 6 | `0x3a150007` | 0.925 | `0.9245803406` | 27 | 27 | `6.661338e-16` | `8.882e-16` | 3.0 |
| widen-n256-d6 | 256 | 6 | `0x3a150008` | 0.935 | `0.9344738486` | 19 | 19 | `5.551115e-16` | `1.221e-15` | 2.3 |
| widen-n256-d6 | 256 | 6 | `0x3a150009` | 0.945 | `0.9434582438` | 23 | 23 | `7.216450e-16` | `8.882e-16` | 2.7 |
| widen-n512-d4 | 512 | 4 | `0x3a15000a` | 0.905 | `0.9046162255` | 81 | 81 | `1.165734e-15` | `1.443e-15` | 8.3 |
| widen-n512-d4 | 512 | 4 | `0x3a15000b` | 0.915 | `0.9149374573` | 64 | 64 | `8.881784e-16` | `1.110e-15` | 6.8 |
| widen-n512-d4 | 512 | 4 | `0x3a15000c` | 0.925 | `0.9236377783` | 54 | 54 | `8.881784e-16` | `1.554e-15` | 4.8 |
| widen-n512-d4 | 512 | 4 | `0x3a15000d` | 0.935 | `0.9348891066` | 54 | 54 | `8.326673e-16` | `1.776e-15` | 5.2 |
| widen-n512-d4 | 512 | 4 | `0x3a15000e` | 0.945 | `0.9449749999` | 43 | 43 | `8.326673e-16` | `1.221e-15` | 4.8 |
| widen-n1024-d6 | 1024 | 6 | `0x3a15000f` | 0.905 | `0.9047121217` | 124 | 124 | `1.110223e-15` | `2.220e-15` | 25.9 |
| widen-n1024-d6 | 1024 | 6 | `0x3a150010` | 0.925 | `0.9248397288` | 98 | 98 | `1.221245e-15` | `1.554e-15` | 18.4 |

Plus `probe-ring-B2`, the applicable control of §5, at `1.110223e-16`.

---

## 3. HALTING SEMANTICS, WHICH ARE THE POINT

**The rule is implemented literally: on disagreement `build_corpus` raises and
returns nothing at all.** Not the passing rows. Not the rows with the bad one
flagged. Not a corpus with a `status` column a reader might filter on later
(`scale/r10_dual_oracle.py:274-291`). `main` writes no `.jsonl` when it is raised
and prints the refusal as the result.

`demonstrate_halt` (`scale/r10_dual_oracle.py:419-448`) builds the same
three-instance corpus twice:

| build | rows returned |
|---|---|
| clean | **3**, all `status="ok"` |
| `deg + 1` planted on instance 1 | **0** — `CorpusHalted`, gap `3.980916e-01` |

**Two of the three instances in the planted build passed their own dual check and
are still not emitted.** That is the whole content of the rule: the corpus is the
unit of confirmation, so a corpus containing an instance its own oracle cannot
confirm does not get to exist in a reduced form. `demo()` step 6 asserts
`planted_rows_emitted == 0`, so the semantics are covered by the self-check and
not merely by this paragraph.

One deliberate departure, stated rather than hidden: `build_corpus` scores
**every** instance before deciding to halt, instead of failing fast on the first
disagreement. Fail-fast would leave the four counts unknowable for everything
after the first bad instance, and those counts are the deliverable. Nothing is
emitted either way; only the diagnostic is fuller.

---

## 4. MUST-FIRE — AND THE CONTROL THAT DOES NOT FIRE

### 4.1 What is structurally different, in one sentence

**The absorbing route solves the nonsymmetric row-normalised `I - P_II` on the
interior alone against `P_IB g` and reads `u` off directly; the Kirchhoff route
solves the symmetric unnormalised `L = D - A` on `interior + {b}` — a different
matrix, one size larger, differently indexed — against `e_b`, and recovers each
value as the RATIO `z_v / z_b` of two entries of one solved column, a division
the other route never performs.**

**And the honest half of that sentence.** `I - Q = D^-1 L_T` is a *diagonal*
relation, which `scale/kirchhoff.py:57-62` states about its own pair. The two
routes share the adjacency data and the degrees read off it. They are not
independent computations; they are two different factorisations of one problem,
and the claim being made is only that a defect in the normalisation, indexing,
transposition or boundary handling of one moves one and not the other.

### 4.2 The planted defect is the repo's own, and it is rejected

`deg + 1` in place of `deg` — the miscount from counting a node among its own
neighbours, `kirchhoff.scratch_chain_with_off_by_one`
(`scale/kirchhoff.py:225-260`). It is applied as a row-scaling by `d/(d+1)` of
blocks already formed, so there is no second copy of `blocks` to drift out of
step; `demo()` step 4 asserts this reproduces that function's output at
**exactly `0.0`**, so the plant is that function's defect and not a lookalike.

| plant | instances | rejected | gap min | q1 | median | q3 | gap max |
|---|---|---|---|---|---|---|---|
| **absorbing route** (`deg+1` in `P`) | 29 | **29 / 29** | `0.2917` | `0.4315` | `0.4831` | `0.5390` | `0.6302` |
| **Kirchhoff route** (`deg+1` on the `L` diagonal) | 17 | **17 / 17** | `0.2917` | `0.4004` | `0.4497` | `0.5257` | `0.5954` |
| **both routes at once** | 17 | **0 / 17** | `0.000e+00` | — | — | — | **`3.331e-16`** |

Weakest rejection `2.916667e-01` (`probe-ring-B2`), strongest `6.301520e-01`
(`reproduce-n512-d4-t0.95`) — **nine orders above the tolerance at the weakest**,
against a largest clean gap of `1.554312e-15`. The distribution sits where
JUPITER's did (`4.07e-01` to `6.30e-01`), which is expected: it is the same
defect on overlapping instances.

The absorbing-side plant rides on the clean pass at no extra cost, so it is
measured on **every** applicable instance. The Kirchhoff-side plant needs a second
full Kirchhoff pass, so it is measured on the 17 instances with `n <= 256`
(`reverse_below`, `scale/r10_dual_oracle.py:274`) — a subset, and named as one.

### 4.3 The row that matters most: 0 of 17

**Both routes carrying the same `deg + 1` miscount agree to `3.331e-16` and the
pair rejects none of them.** This is not a surprise once the algebra is written
down — `L + I = (D + I) - A`, and the absorbing route with `deg + 1` solves
`((D+I) - A_II)^-1 A_IB`, the same operator — but it is the difference between a
pair whose agreement is evidence and a pair whose agreement is a tautology, so it
is run and reported as a number.

**What the 29/29 agreement therefore does and does not license.** It licenses:
the normalisation, the index maps, the block partition, the boundary handling and
the transposition conventions of the two routes are mutually consistent, because
a defect in any of them moves one route and not the other, and that was checked
in both directions. It does not license: the shared adjacency data, the degree
read off it, or any error committed identically in both. An instance whose
adjacency lists are wrong produces two confidently agreeing wrong answers, and
nothing in this report would notice.

---

## 5. WHERE THE KIRCHHOFF ROUTE CANNOT BE USED

`K_b` is nonsingular iff every node of `interior + {b}` reaches `B \ {b}`, which
is exactly `r10_corpus_spec.admissible(adj, B \ {b})` — so the precondition is
checked with the repo's own guard rather than a second copy of it
(`scale/r10_dual_oracle.py:150-168`).

**2 instances of 31 are inapplicable, and both are recorded `dual_checked=false`,
never as passes.**

| instance | absorbing route | Kirchhoff route | status |
|---|---|---|---|
| `probe-B1` (`n=6` ring, `\|B\|=1`) | answers, `u == g_0` | `\|B\| < 2`: grounding at `B \ {b}` leaves the full singular Laplacian; the ratio is `0/0` | **inapplicable** |
| `probe-split` (two components, one boundary node each) | admissible, answers | every grounding singular: 2 interior vertices cannot reach `B \ {b}` | **inapplicable** |
| `probe-ring-B2` (same ring, `\|B\|=2`) | answers | applies | **ok**, gap `1.110223e-16` |

The third row is the control. Without it, two refusals on constructed graphs are
equally consistent with a broken route; with it, the same graph passes the moment
the precondition is met.

**None of the 28 drawn rungs is inapplicable**, because `draw_graph` is connected
by construction and the stratifier never returned `|B| < 4`. A count of zero taken
from the drawn corpus alone would therefore have said nothing about whether the
category is handled — which is why the three probes exist and why the count is
reported as 2 of 31 rather than 0 of 28.

**Weight positivity, the third precondition, is vacuous here and is not claimed as
a pass.** `kirchhoff.laplacian` builds unit conductances only, so an instance with
a zero or negative weight cannot be expressed by this corpus at all. It is
untested, not satisfied.

### The fourth category: never generated

`widen-n128-d4-t0.905` (seed `0x3a150000`) produced **no instance**: the bisection
crossed from above `0.905` to `0.8943072442`, below the band, so no integer `|B|`
lands inside. It was never reached by either oracle, so it is neither a pass, nor
inapplicable, nor an oracle error. It is emitted under its own `kind`
(`not_generated`) and counted in its own column. The dict-merge that would have
filed it back under `kind: "rung"` was found and fixed
(`scale/r10_dual_oracle.py:578`) precisely because a never-generated rung
sitting in the rung stream is the confusion this round catalogues.

---

## 6. ERRORS AND COUNTS, EACH IN ITS OWN COLUMN

| category | count | what it means |
|---|---|---|
| **passed** | **29** | both oracles ran and agreed within the imported `1e-10` |
| **failed** | **0** | both ran and disagreed — would have halted the corpus |
| **errored** | **0** | an oracle raised — also halts; counted separately from failure |
| **inapplicable** | **2** | the Kirchhoff route's preconditions do not hold; `dual_checked=false` |
| never generated | 1 | no instance existed to check; separate `kind` |
| **total instances** | **31** | 28 rungs + 3 probes |

`dual_checked` is 29, and it is the number the max-disagreement figure is over.
It is reported separately from `passed` so that the two can never silently drift
apart: an instance the second oracle did not reach is not evidence about the
first one.

Cost: 295 eigensolves, 1033 Kirchhoff solves, 199.9 s wall clock, one process.

---

## 7. LIMITS

Collected here rather than scattered.

The two routes are diagonally related (`I - Q = D^-1 L_T`) and share the
adjacency and degree data, so the 29/29 agreement is evidence about
normalisation, indexing and block structure and about nothing else; the shared
plant of §4.3 is the measurement of that boundary, and it is `0 / 17`. The
Kirchhoff-side and shared plants ran on 17 instances (`n <= 256`), not on all 29,
because each costs a full extra Kirchhoff pass. The graph family is JUPITER's one
family — uniform attachment tree plus random chords — so the inapplicable count of
2/31 prices the *constructed* class and not a natural rate; a Rips, grid or
expander corpus could hit it by drawing. The widened arm's 16 rungs are 3–4 per
`0.01` bin, which is coverage of the band but not a sample size for anything.
Weight positivity is untested rather than satisfied, since unit conductances are
the only ones expressible. The band-miss rate of 1 in 17 is measured on one target
ladder and one seed base. The halt was demonstrated on a planted disagreement, not
observed on a real one — the corpus never failed, so the semantics are shown to
work rather than shown to have been needed. Nothing here bears on the dose
theorem, which is it.14's §4 and was not re-run; and no claim is made about what a
model can learn from this corpus.
