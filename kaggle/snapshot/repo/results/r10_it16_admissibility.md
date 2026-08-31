# The it.16 admissibility battery: two clauses fail, and the third is a GREEN nobody earned

SATURN, v-main.3M script iteration 16. The script line, verbatim:

> SATURN: C-A split integrity = train/eval same graph family or graph in x
> (impact died at linear-probe train 4.93e-08 vs eval 1.478); C-B linear probe
> within-split must read NRMSE > bar; C-C peak activation bytes = 4*n*s*d*heads
> computed in writing vs card bytes.

Code `scale/r10_admissibility.py`, rows `results/r10_it16_admissibility.jsonl`.
Reproduce every number below with

```
python -m scale.r10_admissibility            # demo(), assert-based, ~13 s
python -m scale.r10_admissibility --write    # the battery, 12.5 s, writes the .jsonl
```

Python 3.11.9, numpy 1.26.4, no torch, Windows 11 x86-64, branch
`feat/r9-causal-consequence` at `bf2a769`. One process, no GPU work, no training,
no capacity sweep. The only randomness the battery adds is the probe's row
shuffle, `PROBE_SEED = 0x3a160000`, printed in every C-B row; the instances are
the corpus's own and carry its seeds.

`scale.vram_gate.preflight(0, name='r10-admissibility', host_mib=256)` returned
`HOST FITS -- needs 320 MiB, 5229 MiB available` before the battery ran. Peak is
one dense `960 x 960` float64 solve.

The corpus admitted is `scale/r10_corpus_spec.py` (JUPITER, it.14) plus
`scale/r10_dual_oracle.py` (SATURN, it.15) — 12 rungs on JUPITER's seeds, 16
widened rungs, 3 probes, one rung never generated.

---

## 0. THE HEADLINE, BEFORE THE EVIDENCE

**Two of the three clauses fail, the third passes against a resource this corpus
does not spend, and all three were ill-posed as written until an amendment was
declared.** The amendments are stated as amendments, not applied quietly.

| clause | verdict | the number |
|---|---|---|
| **C-A** split integrity | **FAILS on one of the two splits a reader could mean** | **12 shared graphs**, labels bit-identical, `43.66 %` of eval rows |
| **C-B** within-split linear probe | **FAILS, 25 of 28 rungs + both pooled splits** | median NRMSE `0.7639` against the bar `0.9` |
| **C-C** peak activation bytes | **GREEN, and unearned** | `512.0 MiB` against `8,585,740,288` card bytes — **VRAM is not the binding resource** |

1. **The corpus declares no train/eval split at all**, so C-A is ill-posed until
   one is declared. Both partitions a reader could take are measured. §2.
2. **The corpus declares no feature set either**, and C-B's verdict turns
   entirely on that missing declaration: on strictly geometric features the
   corpus is admissible **28 of 28**; adding one local read of the boundary data
   makes it inadmissible **25 of 28**. §3.
3. **`4*n*s*d*heads` under-prices the shipped harness by exactly `s/d = 64`.**
   The peak is the `[n, s, s]` operator, not the projections. At batch 8192,
   `s = 1024`, the clause reads `512 MiB` and the real tensor is `32,768 MiB` —
   four times the whole card. §4.
4. **VRAM never binds.** At the round's declared 2 GiB host budget the host
   refuses at batch **8,832** while VRAM would not refuse until batch
   **101,837** — `11.53x` later, with the card at `9 %` of free when the box is
   already out. §4.3.

Counts, five of them, each in its own column, in §5.

---

## 1. THE INSTANCES ARE RECOVERED, AND THE RECOVERY IS BOUND

Nothing below is about a lookalike corpus, and that is checked rather than
asserted.

`make_rung` consumes its rng in the order graph → node order → `g`, and
`stratify` (`scale/r10_corpus_spec.py:361-397`) takes `adj` and `order` and draws
nothing. Its only output is the boundary SIZE, which every shipped row prints. So
an instance is recovered exactly from `(n, mean_deg, seed, boundary_size)` with
**zero eigensolves** — the 121 + 165 of it.14 and it.15 are skipped, which is why
this battery costs 12.5 s against their 72.7 s and 199.9 s.

**The recovery is bound to the shipped rows.** `u_absorbing_range` is a two-float
fingerprint of the label vector that every it.15 rung carries, and recomputing it
from the recovered instance is a check on the graph, the node order, the boundary
set, the boundary data and the oracle at once:

```
recovery bound to 28 shipped rows, max u-range delta 0.0
```

**Zero, not "small".** `demo()` asserts it at `== 0.0`, so a recovery that drifts
fails the self-check rather than quietly measuring something else.

---

## 2. C-A — SPLIT INTEGRITY

### 2.1 The clause is ill-posed, and the amendment is one sentence

**The corpus declares no train/eval split.** There is no split field in
`results/r10_it14_corpus_spec.jsonl` or `results/r10_it15_dual_oracle.jsonl`, and
no split rule in either report. "Train and eval draw from the same graph family"
has no referent until a partition is named.

> **Amendment (C-A).** *The corpus must declare which of its instances are train
> and which are eval.*

Rather than pick one, both partitions a reader could actually take are measured:

- **SPLIT-ARM** — arm `reproduce` (12) against arm `widen` (16). The only
  partition the corpus itself names.
- **SPLIT-FILE** — `r10_it14_corpus_spec.jsonl` (12) against
  `r10_it15_dual_oracle.jsonl` (28). The partition a reader gets by treating the
  two shipped files as two datasets, which is what they look like.

### 2.2 The rule applied, stated before the counts

- **Same graph family** is decided by **regeneration, not resemblance**: an
  instance is in the family iff `spec.draw_graph(n, extra_for(n, mean_deg),
  default_rng(seed))` reproduces its edge set bit-for-bit. A graph from any other
  process fails on the hash. No degree-distribution statistic is involved and
  none would have been a proof.
- **Graph identity** is `SHA-256` over the sorted edge list on the generator's
  own node labels. Both splits are drawn into the same label space `0..n-1`, so
  an edge-set match means the same object, not merely an isomorphic one — the
  stronger claim, and the cheap one here.
- **Labelled identity** adds the boundary list and the float64 bytes of `g`. Two
  instances agreeing on it have **bit-identical labels `u`**, because `u` is a
  function of exactly those three. That is what makes the count a statement about
  leakage rather than about coincidence.

### 2.3 The counts

| split | same family | **shared graphs** | shared labelled | eval rows | leaked rows | status |
|---|---|---|---|---|---|---|
| **SPLIT-ARM** `reproduce \| widen` | **yes** | **0** | 0 | 5,693 | 0 (`0.00 %`) | **ok** |
| **SPLIT-FILE** `it14 \| it15` | **yes** | **12** | **12** | 10,105 | **4,412 (`43.66 %`)** | **failed** |

**Same graph family: yes, on both, and that half of the clause is satisfied.**
All 40 instances regenerate bit-for-bit from `spec.draw_graph`; `out_of_family`
is empty. The two arms differ only in the stratifier's target argument, which
does not touch the draw.

**SPLIT-ARM shares no graph.** The seed bases are disjoint — `0x3a140000 + i`
against `0x3a150000 + i` — and 0 is what disjoint seed bases buy.

**SPLIT-FILE shares 12 graphs, and all 12 are labelled-identical.** This is not a
coincidence and it is not a bug in either file: arm `reproduce` **is** it.14's
corpus, drawn on his seeds on purpose, which is exactly the property it.15 §1.1
reports as `max_lambda_2_delta: 0.0`. The reproduction that made the second
oracle trustworthy is the same fact that makes the two files an unusable split.

### 2.4 What a shared graph costs, measured rather than argued

**It is not automatically fatal, and here is the price in this corpus's own
units.**

All 12 shared graphs carry the same boundary set AND the same `g`, so their label
vectors are bit-identical between the two sides. **4,412 of the 10,105 eval
interior rows — `43.66 %` — carry a label already present in train.** Any
consumer able to key on the instance reads those rows off memory.

The cost is not symmetric with the failure this clause is named for, and saying
so is the point:

- **`impact` failed the other way.** Train `4.93e-08`, eval `1.478`, worse than
  the mean predictor's `1.000` (`.superpowers/sdd/polymorphic-drifting-squirrel/mars-report.md:82-88`).
  Nothing leaked; the label functional `w = Bᵀ Kᵀ e_q` is per-graph and
  `cos(w0, w1) = -1.077e-08`, so the two graphs' labels are orthogonal and no
  transfer was available at all. The graph was not in `x`.
- **SPLIT-FILE fails toward the optimistic side.** A leaked label inflates eval
  rather than destroying it, so it would present as a *good* result.

They are the same disease at opposite ends: the split does not separate what a
consumer can memorise. The `impact` end is loud and the SPLIT-FILE end is silent,
which makes this end the more expensive one.

**The third reading of the clause — "graph in `x`" — is not answerable on this
corpus and is recorded as such.** `x` does not exist yet: the corpus ships `adj`,
`boundary`, `g` and the label, and declares no tensorisation. Whether the graph
reaches `x` is a decision no downstream consumer has made, and it is exactly the
decision that killed `impact` (channels 2..9 "reserved for sparse B encoding if
needed (not used)", `scale/impact.py:69`; neither `A` nor `B` enters `x`). It is
reported as an open precondition, not as a pass.

### 2.5 C-A is shown able to fail, in both directions

| plant | result |
|---|---|
| **leak**: one train instance copied into eval | `shared_graphs = 1`, `shared_labelled = 1`, **status failed** |
| **family**: a 32-node ring substituted into eval | `same_graph_family = false`, **status failed** |
| **control**: the clean SPLIT-ARM above | `shared_graphs = 0`, **status ok** |

Both plants are asserted in `demo()` (`scale/r10_admissibility.py`, demo steps
2a and 2b), and the control is asserted too: a clause that fires on everything is
worth as little as one that fires on nothing, so the non-firing case is part of
the demonstration rather than a footnote to it.

---

## 3. C-B — THE WITHIN-SPLIT LINEAR PROBE

### 3.1 The bar is imported, and it is the repo's own anti-triviality bar

**`FAIL_BAR = 0.9`, imported by object identity: `FAIL_BAR = rips_gate.FAIL_BAR`,
defined at `scale/rips_gate.py:61`.** `demo()` step 0 asserts
`FAIL_BAR is rips_gate.FAIL_BAR`, so a local re-pick fails the self-check rather
than passing silently — the same discipline as it.15's `AGREEMENT_TOL`.

It is the right constant and not merely an available one. `scale/rips_gate.py:58-59`
declares it as *"the decoder is doing essentially nothing"*, pre-registered before
any number there was read, and `scale/impact.py:1121` already uses it for exactly
this role: `local r0 NRMSE = ... (FAIL_BAR 0.9, must be >=0.9)`. `PASS_BAR = 0.5`
is carried alongside so every reading can be placed against both, but the verdict
is on `FAIL_BAR`, because that is the bar the anti-triviality clause means.

**The corpus declares no bar of its own.** Neither it.14 nor it.15 states an
NRMSE threshold; the only tolerance either declares is the `1e-10` oracle
agreement, which is a different quantity. Importing is therefore not a
convenience here, it is the only honest option.

### 3.2 The feature ladder, declared before any number was read

The corpus declares no feature set either, so one is declared here, as a ladder,
so the verdict cannot be read as a feature set chosen to produce it. Every
feature is static and strictly local; none iterates, none solves, none sees `u`.

| rung | columns |
|---|---|
| **F0** `degree` | intercept, `deg` |
| **F1** `balls` | F0 + `\|ball(v,1)\|`, `\|ball(v,2)\|`, `\|ball(v,3)\|` |
| **F2** `geometry` | F1 + boundary-neighbour count, hop distance to `B` (capped 8), `1/(1+hop)` |
| **F3** `boundary_data` | F2 + `gbar(v) = (1/deg v) * sum of g over boundary neighbours` |

`ball` is `scale/rips_gate.py:83`, reused. The fit is `rips_gate.fit_eval`
(`:163-174`), the repo's own half/half least squares with a `1e-6 I` ridge, on
rows shuffled by `PROBE_SEED` so the halves are not an artefact of the
generator's node labelling. **Both halves come from the same instance (or the
same arm): this is within-split, and it is not reported as a transfer
measurement.**

**The verdict is taken on F3, the most generous rung**, because the clause asks
whether a linear probe on the features ALREADY beats the bar, and withholding the
one local feature that obviously carries signal would be choosing a probe that
fails.

### 3.3 The readings

Held-out NRMSE, 28 rungs, within-instance. Full per-rung table in the `.jsonl`.

| rung | min | **median** | max | **at or under `0.9`** | at or under `0.5` |
|---|---|---|---|---|---|
| F0 `degree` | 0.9991 | 1.0066 | 1.2641 | **0 / 28** | 0 / 28 |
| F1 `balls` | 0.9831 | 1.0060 | 1.5602 | **0 / 28** | 0 / 28 |
| F2 `geometry` | 0.9686 | 1.0192 | 2.0404 | **0 / 28** | 0 / 28 |
| **F3 `boundary_data`** | **0.5670** | **0.7639** | 1.4169 | **25 / 28** | **0 / 28** |

Pooled within-split, all rows of one arm together:

| split | F0 | F1 | F2 | **F3** | status |
|---|---|---|---|---|---|
| `reproduce`, 12 instances, 4,412 rows | 1.0011 | 0.9586 | 0.9588 | **0.7961** | **failed** |
| `widen`, 16 instances, 5,693 rows | 0.9995 | 0.9981 | 0.9976 | **0.7975** | **failed** |

**So the clause fails: 25 of 28 rungs and both pooled splits read NRMSE under the
bar.** Reported plainly, as instructed, rather than by hunting for a probe that
fails: F0, F1 and F2 are exactly such probes, and they are printed above rather
than promoted to the verdict.

### 3.4 The mechanism, which is one line of the spec's own algebra

The harmonic extension satisfies `u = P_IB g + P_II u` exactly. The first term
is, per interior vertex, `(1/deg v) * sum over boundary neighbours of g` —
**`gbar` is literally the first Neumann term of the label's own defining
recursion.** A linear probe handed it is performing one hop of the iteration the
architecture is supposed to perform, and the ladder measures what that one hop is
worth: the median moves from `1.0192` to `0.7639`, roughly `42 %` of the label
variance, on one column.

It is worth being precise about what that does and does not say. **No rung of the
ladder reaches `PASS_BAR = 0.5` — 0 of 28.** The probe is nowhere near reading
the label; the remaining `P_II u` term is a global solve at `lambda_2 ≈ 0.95` and
no local feature touches it. The corpus is not trivial. It is, by the bar this
repo uses for exactly this question, **not clean either** — a probe at `0.76` is
not "doing essentially nothing".

### 3.5 The three that pass, and they are not a rescue

`reproduce-n64-d6` (`1.4169`), `reproduce-n256-d4` (`1.0575`) and
`widen-n128-d4-t0.915` (`0.9187`). The first has 60 interior rows against 9
features — 30 train, 30 eval — and a held-out NRMSE above 1.4 on a mean predictor
of 1.0 is the variance of a 30-row fit, not evidence of a hard instance. They are
counted as passes because that is what the instrument returned, and they are
named here so nobody reads `3 / 28` as a partial acquittal.

### 3.6 The verdict turns on a declaration the corpus does not make

**This is the amendment, and it is the more important finding of §3.**

> **Amendment (C-B).** *The corpus must declare its tensorisation — what goes in
> `x` — before C-B has an answer.*

On F0–F2 the corpus is **admissible 28 of 28**. On F3 it is **inadmissible 25 of
28**. The entire verdict is decided by whether the boundary data `g` is readable
one hop away in whatever `x` a consumer builds, and no document in this corpus
says whether it is. A tensorisation that carries `g` only on boundary rows and
gives interior rows no local view of it lands at F2 and passes; one that
broadcasts a one-hop neighbourhood average lands at F3 and fails.

That is the same missing declaration as §2.4's third reading. `impact` died of it
from the other direction — the graph was NOT in `x`, and the task became
unlearnable across graphs. This corpus has not yet decided, and until it does,
C-B's answer is a function of an undeclared choice rather than of the corpus.

### 3.7 C-B is shown able to fail, in both directions

| plant | reading | against the bar |
|---|---|---|
| **must-fire**: label `3*deg - 1.5*ball1 + 7`, inside the F1 span | **`5.041e-05`** | four orders under `0.9` — fires |
| **must-not-fire**: white noise, same features, same split | **`1.6028`** | above `0.9` — does not fire |

`5.041e-05` rather than machine zero because `fit_eval` carries a `1e-6 I` ridge
(`scale/rips_gate.py:172`) and `ball3` runs to ~60; the assertion is at `1e-3`
and the reason is in the code beside it. Both are asserted in `demo()`.

**The must-not-fire half is the one that matters here.** A probe that read `0.76`
on noise would make the whole of §3.3 an artefact of the fit; it reads `1.6028`,
so the 25 failures are about the labels.

---

## 4. C-C — PEAK ACTIVATION BYTES, IN WRITING, AND THE RESOURCE THAT ACTUALLY BINDS

### 4.1 The formula, computed as the clause asks

`4 * n * s * d * heads`, float32, where for this corpus:

- **`s`** is the graph size, because one graph instance is one sequence:
  `s ∈ {64, 128, 256, 512, 1024}` (`spec.RUNGS`, `scale/r10_corpus_spec.py:506`).
- **`d = 16`** is `m3_capability.D_MODEL`, `scale/m3_capability.py:79`,
  *"fixed per task spec"*.
- **`heads = 1`**, and this is not a convention. `Arm.__init__` builds **one**
  `wq` and **one** `wk` (`scale/m3_capability.py:108-109`) and `_operator`
  returns a single `[n, s, s]` tensor (`:144`). There is no head axis to sum over.
- **`n`** is the batch. The three the it.8 RSS fit was measured at, plus two small
  ones.

**Card bytes: `8,585,740,288`** — 8188 MiB, NVIDIA GeForce RTX 4060 Laptop GPU,
read by `vram_gate.read_card()` (`nvidia-smi`, 7956 MiB free at run time).

| batch `n` | `s` | **`4*n*s*d*heads`** | as MiB | % of card |
|---|---|---|---|---|
| 8 | 1024 | 524,288 | 0.50 | 0.006 % |
| 64 | 1024 | 4,194,304 | 4.00 | 0.05 % |
| 2048 | 1024 | 134,217,728 | 128.00 | 1.56 % |
| **8192** | **1024** | **536,870,912** | **512.00** | **6.25 %** |
| 32768 | 1024 | 2,147,483,648 | 2048.00 | 23.85 % |

**The clause passes at every shape, with 4x headroom at the largest one it was
ever going to be asked about.** 25 of 25 shapes fit.

### 4.2 The formula prices the wrong tensor, by exactly `s/d`

`4*n*s*d*heads` is the size of a `[n, s, d]` projection output. **The shipped
harness's peak is the operator**, `a = self._operator(q, k)` returning `[n, s, s]`
at `scale/m3_capability.py:144`, which `forward` then keeps live through `a @ x`.
That is `4*n*s²*heads` bytes — larger by exactly `s/d`, which is **64** at
`s = 1024`.

The `s²` is not a new claim. `scale/m3_flops.py:260` already carries
`4*n*s^2*d_model` as a base term; it is the same tensor, priced for memory
instead of arithmetic.

| batch `n` | `s` | clause figure | **shipped peak `[n,s,s]`** | under-priced by |
|---|---|---|---|---|
| 8 | 1024 | 0.50 MiB | **32.0 MiB** | **64x** |
| 2048 | 512 | 64.0 MiB | **2,048.0 MiB** | 32x |
| **8192** | **1024** | **512.0 MiB** | **32,768.0 MiB** | **64x** |
| 32768 | 1024 | 2,048.0 MiB | **131,072.0 MiB** | 64x |

**At batch 8192, `s = 1024`, the clause reads 512 MiB and the real tensor is four
times the entire card.** If this harness spent VRAM, the clause as written would
have passed a shape that cannot exist on this machine — the same over-request
shape that killed `impact` at 8192 MiB on an 8188 MiB card, which
`scale/vram_gate.py:4-6` names as the reason this clause exists.

### 4.3 VRAM IS NOT THE BINDING RESOURCE, AND SAYING SO IS THE CLAUSE

**Stated explicitly, as required.** The consumer of this corpus is the capability
harness, and it spends no VRAM at all: `device=cpu (no .cuda() anywhere in this
file)`, `scale/m3_capability.py:261`. *(Provenance correction: the it.8 filing and
`scale/vram_gate.py:37` both cite this string as `m3_capability.py:120`; at
`bf2a769` it is at `:261`. The string is unchanged, the line number is not.)*

So every GREEN in §4.1 is a true statement about a resource this corpus's
consumer does not touch — the same shape it.8 catalogued when
`require(4275, name='capacity-sweep')` returned FITS citing 7162 MiB of free
VRAM. **The 25 VRAM rows are recorded `status="inapplicable"`, not `passed`.**

**The binding resource is HOST RSS.** `pricing.PEAK_RSS_MIB`
(`scale/r10_it8_pricing.py:46`) measured 901.2 / 1567.9 / 4275.1 MiB at
n = 2048 / 8192 / 32768. The published line is reconstructed here rather than
copied, and the reconstruction names the fit rule:

```
RSS(n) = 665.5 + 0.1102 * n   MiB
```

is the line through the **two largest** points with `n = 2048` **held out**;
predicted 891.2 against measured 901.2, **held-out residual 10.1 MiB**. (An
unweighted three-point least squares gives `672.2 + 0.10992*n`, a different line
— which is why the rule is stated. it.8 reported the held-out point as accurate
"to within 10 MiB"; it is 10.1, and the tenth is recorded rather than rounded
off.) `demo()` asserts both coefficients and the residual.

| batch `n` | host peak RSS | × 1.25 margin | against the round's 2048 MiB budget |
|---|---|---|---|
| 8 | 666.4 | 833.0 | ok |
| 64 | 672.5 | 840.6 | ok |
| 2048 | 891.1 | 1113.9 | ok |
| 8192 | 1567.9 | 1959.9 | ok |
| **32768** | **4275.1** | **5343.9** | **FAILS** |

**The crossover, which is the whole answer to C-C.** At `s = 1024` and the
round's declared 2 GiB budget:

| | batch at which it refuses | load on the other resource at that point |
|---|---|---|
| **HOST** | **8,832** | VRAM at **690 MiB = 9 %** of the 7956 MiB free |
| **VRAM** | **101,837** | — **11.53x later** |

**The box is out of memory at batch 8,832 with the card at 9 % occupancy.** A
C-C gate reading the card alone reports GREEN through that entire range and keeps
reporting it for another order of magnitude of batch size. That is a clause
satisfied against the wrong resource, and it is the reason this section exists.

The measured host reading is not stable and is reported separately from the
verdict: `read_host()` returned **1142 / 2351 / 3178 / 4615 / 5229 / 5770 / 6946
MiB available** across seven runs of this module inside twenty minutes on the same
prompt, and inside the last of those runs it moved from **1142 MiB at preflight to
an implied 5590 MiB thirteen seconds later** � the two readings that bracket a
single `demo()`. The crossover therefore swung from batch 2,251 to batch 34,541
within one run of the same code on the same box. Pricing
against that number would make the verdict a statement about the minute it ran
in, so the `.jsonl` carries `host_status` against the declared 2048 MiB budget and
`host_status_measured` against the reading. This is `scale/vram_gate.py:14-30`'s
aggregate-occupancy argument arriving one resource over.

### 4.4 C-C is shown able to fail

| plant | result |
|---|---|
| **host must-fire**: `preflight(0, host_mib = 4x available)` | refused, `fits = False` |
| **card must-fire**: `preflight(10x card total)` | refused, `DOES NOT FIT` |
| **the finding, asserted**: `vram_refuses_at_batch > host_refuses_at_batch` | **101,837 > 8,832** — the VRAM clause cannot fail before the host does, at any shape this corpus reaches |

The third row is the demonstration that matters. The first two show the gate
mechanism can refuse; the third shows that **on this corpus the VRAM clause has
no reachable failure state at all**, which is what makes its GREEN unearned
rather than merely generous.

---

## 5. COUNTS — FIVE CATEGORIES, EACH IN ITS OWN COLUMN

| clause | total | passed | failed | errored | inapplicable | **not reached** |
|---|---|---|---|---|---|---|
| **C-A** | 4 | 1 | **2** | 0 | 0 | **1** |
| **C-B** | 30 | 3 | **27** | 0 | 0 | 0 |
| **C-C** host | 25 | 20 | **5** | 0 | — | — |
| **C-C** VRAM | 25 | — | — | — | **25** | — |

- C-A's 2 failures are SPLIT-FILE and the planted leak; the planted one is tagged
  `role: "must_fire"` and is not a corpus defect.
- C-B's 30 rows are 28 rungs plus 2 pooled splits.
- **C-C's VRAM column is 25 inapplicable and 0 passed.** A clause priced against a
  resource the job does not spend is not a pass, and putting those 25 in the
  passed column is exactly the false GREEN this iteration was sent to look for.
- **`not_reached` is the fifth category, and it is carried forward rather than
  dropped.** `widen-n128-d4-t0.905` (seed `0x3a150000`) produced no instance at
  it.15 — the bisection crossed from above `0.905` to `0.8943072442` — so it
  reaches neither C-A nor C-B here either. It is emitted with
  `status="not_reached"` and counted in its own column, not summed into any pass.

---

## 6. LIMITS

Collected here rather than scattered.

The two split partitions of §2 are the two a reader could plausibly take, not an
exhaustive enumeration; a partition by `n`, by target, or by a random instance
draw would give different counts and none was measured. The graph-identity rule
is label-sensitive, so two isomorphic graphs on different labellings would read as
distinct, and no isomorphism-invariant count is offered — on this corpus the two
agree, because both splits come from one generator into one label space, but that
is a property of this corpus rather than of the rule. The C-B feature ladder is
one declared ladder of four rungs; a fifth rung carrying a two-hop boundary
average would read lower and was not run, and the verdict rung F3 is the most
generous of the four rather than the most generous conceivable. C-B is measured
within-instance and within-arm only — no transfer reading was taken, so nothing
here says what a cross-instance probe would do, and the `impact` failure it names
was a transfer failure. The three passing rungs of §3.5 include one 30-row fit.
`heads = 1` is read off the shipped `m3_capability` module; a future arm with a
head axis changes §4.1 by a factor and changes §4.2 not at all. The `s²` peak of
§4.2 is the largest single live tensor, not a full activation-memory accounting —
autograd retains more, and the it.8 note that a per-example loop's backward was
the memory cost that killed `n_train=8192` (`scale/m3_capability.py:158-162`) is
evidence that the true peak exceeds any forward-tensor formula. The host RSS line
is three measured points on one machine at one thread pin, extrapolated to batch
sizes outside their range at §4.3's crossover; the `1.25` margin is
`vram_gate`'s, imported and not re-picked. The 2048 MiB budget is the round's
declared figure, not a measurement. No claim is made here about what a model can
learn from this corpus, and no clause below C-B was run against a trained arm —
this iteration attacks the corpus, not the architecture.
