# LAW L-REFLECTOR — the audit, the hop gate, and the filter question

Wilson. Ruling on `docs/PHASE_G.md`, `docs/PHASE_H.md`, `docs/PHASE_I.md`,
`docs/PHASE_I1.md` and the `tests/` producers under them.

Three outcomes and no fourth. **SURVIVES**: the fixed structure supports the
sentence. **VOID**: scorer and ceiling are different functionals, or the regime
forbids the sentence. **RESTATED**: the number stands, the sentence changes.

---

## 0. The law, in one line

A scorer and its ceiling must be the same functional. A ratio between two
different functionals is not a small error — it is not a quantity. The row is
void, not inverted, because inverting it would require knowing which functional
was meant, and nothing in the row says.

---

## HALF ONE — THE AUDIT

### 0.1 Machine row, carried by every table below unless overridden

| field | value | provenance |
|---|---|---|
| torch build (CPU rows) | `2.14.0+cpu` | `docs/PHASE_G.md` header |
| torch build (GPU rows) | `2.14.0+cu126` | `tests/foreman/iaut/iaut_results.jsonl` `producer` |
| numpy / python | `2.4.6` / `3.11.9` | same |
| host | Windows 10.0.26200, Intel64 F6 M183, 28 logical CPUs | `docs/PHASE_G.md` header |
| card | RTX 4060 Laptop, 8188 MiB, cc 8.9 | `docs/PHASE_I.md` §4 |
| flash-attn | **never compiled into this Windows build, at any shape or dtype** | `docs/PHASE_I1.md` §5 |

That last row is fixed structure and it is load-bearing: every kernel-race
verdict in the tree is measured against SDPA-math / mem-efficient / eager, never
against flash. It is stated on the pages; it is restated here because it caps
what any throughput sentence in this repository can mean.

---

### G — `docs/PHASE_G.md`

**G§2 — phase fold-in, `2.398082e-13` vs `1e-12`**
Fixed: initializer n/a (position-only construction); parameterization
`phase_factor` on shipped `ceq.arm_phase`; corpus regime n/a; scorer = max abs
fold-in error; ceiling = the same max abs error against float64 truth — **same
functional**; bin scheme none; dtype float64; torch `2.14.0+cpu`; eval n = 5
seeds at n=4096, d=64, no subsample.
→ **RESTATED.** The number survives a re-run through the shipped path. Two
sentences attached to it did not: the provenance sentence (instrumented counter
read `{'phase_factor': 0, ...}` — a local copy), and the cost sentence ("the
phase is free" is not in RoFormer §3.4.2, which says the direct rotation *is not
computationally efficient*). Number stands, both sentences replaced.

**G§2b — the phase never reaches the attention pattern**
Fixed: the parameterization itself is the finding. In `ceq/arm_smprime.py::
numerator`, θ multiplies the score *after* exponentiation; RoPE rotates q and k
*before* the dot product.
→ **SURVIVES**, and it is the cleanest fixed-structure audit in Phase G: modulus
row and normalizer differ by `0.000e+00` between θ=0 and θ~N(0,1). A measured
zero on the fixed path, not a trained result. The `logit` route that closes the
gap is *available, not used* — no measured row in that document runs through it.

**G§3 — half-precision recipe, 61.513× → 4.001× floor**
Fixed: dtype path fp16 / bf16 / fp32 / fp64; the cached fp32 position table is
the fixed structure and the whole finding; torch `2.14.0+cpu`; 5 seeds; no eval
subsample.
→ **SURVIVES.** Scorer and control are the same functional (ratio over the same
dtype floor, with a do-nothing `q@kᵀ` control at `1.612×` / `1.705×`). The page
already states the law that makes it survive: *a ratio over a floor is not
self-interpreting* — the identical code path at float32 reads `72.253×` at an
absolute error of `2.4e-04`.

**G§3b — kernel race, 5 routes, do-nothing floor `0.042 ms`**
Fixed: wall clock is one functional across all five rows including the floor;
dtype path is **not stated for this table**; torch `2.14.0+cpu`; flash and
compiled FlexAttention unreachable by construction.
→ **RESTATED.** The dtype row is missing from the table and must be added. The
conclusion is unaffected because the comparison is within one functional and the
gap is `+1905.1 ms` (155.1×) — not a margin any dtype choice closes.

**G§3c — carry collapse, `1.353e-13` vs control `3.514e+00`**
Fixed: this is an algebraic identity about softmax, not a trained quantity.
float64, n=1024, d=64, 3 seeds. Cost row: n=4096, d=64, float32, CPU,
flush-denormal **on**, interleaved min-of-9.
→ **SURVIVES**, and it is the second cleanest. The denormal flag is named as
fixed structure and priced (`2.341×` with it off, 0.609% of live causal pairs in
the `exp(−104)..exp(−87.3)` band). Four separate artifacts were isolated by name
before the `1.084×` was reported.

**G§4 — Chebyshev degree 11 / 34 / 130**
Fixed: `f(x)=1/(1−γx)`, pole at `1/γ`, Bernstein ellipse; no trainable part at
all; ε = 1e-6; DCT-I over 16,385 extrema on a 1,040,005-point grid; LP over
60,001 points re-verified on 2,000,001.
→ **RESTATED, correctly.** Two functionals are in play — truncated-series sup
bound (11/34/130) and LP-certified minimax floor (11/33/125) — and the page names
both side by side rather than dividing one by the other. That is the correct
handling of exactly the defect this law exists to catch.

**G§4b — the accuracy margin was a property of the bed**
Fixed: `Q` = dense random symmetric rescaled by `0.999/row_max`, which collapses
the spectrum to `[-2.538e-02, 9.627e-01]` with exactly one eigenvalue above 0.1.
→ **RESTATED.** The pass is real (`6.857e-07` on a spectrum-filling path-graph
`Q`), the margin is ~1.5× at γ=0.99 and not 100×. The arithmetic proof that the
old margin was one point: `|p_K(0) − 1| = 2.7394e-07`, worst reported cell
`2.730e-07` — the same number.

**G§4c — the cost lever**
Fixed: density 0.1, N ≥ 1024, `OMP`/`MKL`/`OPENBLAS` pinned to one thread.
→ **RESTATED.** The solve wins 6 of 6 at density 0.1 and 12 of 12 unchanged in
sign under thread pinning. Repriced against `dense_factor_plus_solve` (fresh
Cholesky), where the filter wins 9 of 12. Both baselines reported, stricter one
headlined — same functional throughout.

**G§5 — Fourier reroute, 32-frequency arm**
Fixed: initializer = the schedule the buffer already used, promoted
`register_buffer → nn.Parameter`, nothing else changed; +32 params on 1,937;
5 seeds; corpus streams fresh signals every step so **repetition = 0 by
construction**; scorer = nats, paired.
→ **SURVIVES.** `−0.0138 / −0.0331 / −0.0342` at t = `−3.69 / −12.83 / −14.58`,
5/5 seeds at every budget. The null lived for the first time in the project
(106/160 instances, 0 unscored). The bed's own instrument was validated before
being trusted: a slot-order null is an algebraic no-op on the chirp arm
(`6.664e-08`) and moves the genuinely rotary arm by `3.524e-03`.

**G§5b — the 4-frequency reproduction bar**
Fixed: `f_learn4_randinit` and `e_learn4_ropeinit` carry identical **1,941**
parameters and the same seeds, differing only in start point — parameter count
excluded by construction.
→ **VOID.** On a rebuilt bed the pre-registered 150-step reproduction bar fails
by `0.0063` nats on every matched pair. A row that fails its own pre-registered
bar is void; it is not rescued by the 32-frequency table, which is a different
measurement at a different parameter count. The page already keeps them apart
after an earlier draft merged them.

**G§6 — C10 closed classes, 4,424 subsets**
Fixed: exact `Fraction` arithmetic against float64, 0 disagreements; independent
graph-closure algorithm reproduces the identical split; γ swept at 1−1e−6 and
1−1e−7.
→ **SURVIVES.** The residue reads `1.0666665e-06` and `1.0666667e-07` — a factor
of ten, which *measures* the O(1−γ) claim. One γ could not have distinguished
that from coincidence. The eigenvalue-1-multiplicity half is correctly excluded
from its own bars as a construction property.

**G§6b — C12 flux spectrum**
Fixed: the operator and its closed form both live inside the test file; a
repository-wide search for a magnetic Laplacian returns nothing.
→ **RESTATED.** `6.661e-16` / `4.001e-16` / `1.776e-15` are real and would not
land on the closed form at four fluxes by accident. But the two routes to β₁ are
rank-nullity stated twice, and no shipped code is exercised. It is two
derivations by one author agreeing — stated at that size on the page.

**G§7 — the resolution bound over seven corpses**
Fixed: pigeonhole — `n` behaviours onto `m` observable values — needs no ground
truth and no null to fire.
→ **SURVIVES**, with its refusal intact: it declines two of seven corpses because
no defensible noise scale existed, and the companion `certified_gap` refuses on
the S6 chirp gap and prices it (`n = 116` under t, `n = 406` under
empirical-Bernstein) rather than certifying.

---

### H — `docs/PHASE_H.md`

**H§3 — the fetch ledger, 8 sources**
→ **SURVIVES.** It is a fixed-structure audit by construction: it audits what the
sources *say*, before any table used them. Three of five contradicted the
intended sentence. Unreached sources are named as unreached (Merrill–Sabharwal
TACL, Mamba-2 2405.21060 never fetched at all, four arXiv:2210.02671 quotes
lane-only, the Pāṇini kāraka row auth-gated and resting on nothing).

**H§5 — order-free composition baseline**

| baseline | L2 error |
|---|---|
| bag-of-context | 4.9938 |
| last-operator-only | 5.0922 |
| order-free composition | 5.5367 |
| chance | 5.7160 |

Fixed: planted operators are norm-preserving **SO(8)** rotations; scorer is L2 on
the sphere; the true state lies on that same sphere.
→ **VOID.** Order-free scores at chance *by geometry*: the expected L2 distance
between two uniform points on a sphere is a constant, so a permuted composition
lands at chance however wrong the permutation is. Bag-of-context wins only
because averaging shrinks toward the origin — a norm artefact. Both trained arms
score **worse than chance** (6.3206, 6.7240 vs 5.7160), which is the signature of
a metric that cannot see the property under test. The kill "(a) ties (d) → binding
is dead" presupposes the bed measures something; it does not, so the kill does
not fire and no number from those twins carries information either way.

**H§6a — the gate class keyed on `det = 0`**
Fixed: `R_ε = (1−ε)P + εI` is invertible for every ε > 0; measured down to
`ε = 1e−15`, length 50,000, ~5,000 resets.
→ **RESTATED.** `det = 0` is sufficient but not necessary; the separating
property is **contraction**, not singularity. A load-time gate check rejecting on
"non-idempotent and non-invertible" would admit the wrong objects and reject the
right ones. The earlier structural claim came from measuring L2 on isolated
prefixes and never decoding a task label — a scorer that was not the task's.

**H§6b — the line gate as a noise filter**
Fixed: the index of coincidence is a function of the **letter multiset alone**,
so the gate is blind to word order, spelling and grammar by construction.
→ **VOID as a noise filter, RESTATED as a length filter.** Shuffling letters
within the first 1,000 lines gives **75 rejects against 75** — identical, not
approximately. Real corpus words in random order pass 93.6% of the time. Reject
rate is monotone in line length and nothing else: 41.72% under 40 letters, 0.00%
above 250.

**H§6c — verb-centre recall 1.0000, Wilson [0.9922, 1.0], n = 489**
Fixed: the system under test and the oracle are **the same parser call on the
same string, run twice**.
→ **VOID.** An identity wearing error bars. The replacement is measured and on
the page: cross-tagger agreement `0.8262`, Wilson [0.7901, 0.8572].

**H§8 — the H.1 accuracy table (generator-visible)**

| arm | params | accuracy | std |
|---|---|---|---|
| (a) vectors + softmax + RoPE | 620 | 0.2238 | 0.0115 |
| (b) scalar gate | 620 | 0.2628 | 0.0148 |
| (c) operator gate, invertible | 552 | 0.9570 | 0.0048 |
| (d) operator + projector | 654 | 0.9470 | 0.0059 |

Fixed: arms (c)/(d) forward pass is `state = ops[verb] @ (state + emb[patient])`
— **character-for-character the bed's own generative recursion**.
→ **RESTATED.** 59 σ is real and the per-seed floors agree bitwise across lanes
(`0.3900 / 0.3600 / 0.3190 / 0.3640 / 0.3500`), which is what proves the split is
identical rather than merely distributional. The parameter residue favours the
losing arms. But the fixed structure forbids the sentence "non-commutative
operators win": on this bed *non-commutative operator* and *the generator's exact
operator family* are the same object. The sentence it supports is "the correct
hypothesis class fits its own generator." The 84.9% / 15.1% split against the
matched diagonal control (554 params, `0.3342 ± 0.0161`) is the same restatement.

**H§8b — the confound-removed I-AUT race**

| arm | params | mean | std |
|---|---|---|---|
| a vectors + softmax | 460 | 0.2285 | 0.0176 |
| b scalar gate | 460 | 0.2305 | 0.0190 |
| **c operator, invertible** | **404** | **0.8620** | 0.0556 |
| d operator + projector | 398 | 0.6975 | 0.1034 |
| DIAG commuting control | **404** | **0.2860** | 0.0150 |

Fixed structure, read from the producer artifact rather than the page —
`tests/foreman/iaut/iaut_results.jsonl`:

| row | value |
|---|---|
| initializer | default per-module; no bias pinning; anchors seeded `iaut_anchor\|seed` |
| parameterization | matched by **live `numel()` search over hidden width d**, not a formula |
| corpus regime | synthetic; `n_train = 3000`, `n_eval = 800`, `word_length = 14` → 2¹⁴ = 16,384 possible words; **disjoint train/eval by construction; repetition ≈ 0** |
| scorer functional | argmax final-state-class accuracy on the eval split |
| ceiling functional | **the same** — multiset ceiling fitted `0.2898`, oracle-fit-on-eval `0.3110`, both accuracy on the same 800 items |
| bin scheme | none; 8 discrete classes |
| dtype path | model `float32`, transition table `int64` |
| torch build | `2.14.0+cu126` |
| eval subsample | 800, floor `1/800 = 0.00125` |

→ **SURVIVES**, and it is the strongest row in the repository. Scorer and ceiling
are one functional. The `1/800 = 0.00125` floor is three orders below the
`0.8620 − 0.2860 = 0.5760` separation, so quantization cannot reach it. The kill
was pre-registered and did not fire; an independent re-run from a clean directory
returned `c = 0.8817 ± 0.0869` against `DIAG = 0.2803 ± 0.0104`, 5/5 seeds
pairwise with no overlap, worst case `0.7300` against `0.2937`. The control is
**saturated**, within `0.031` of its own multiset ceiling of `0.3110`, so the
separation is not a training artefact.

One producer-level correction the page already carries and the artifact does not:
`iaut_results.jsonl` ships `"tightest_floor": 0.24875`, superseded by the
multiset ceiling `0.3110` measured later. Against the real floor the operator's
worst seed is `0.7300 / 0.3110 = 2.35×`. Arms (a) and (b) sit at `0.23`, *below*
that ceiling, so they are underfit against a baseline this row did not measure —
their scores bound the vector side from above and do not measure it. Scope in the
signature: **one parameter budget, one word length, no length-generalisation
check.**

**H§9 — census ray radii**

| φ | 0 | π/4 | π/2 | 3π/4 | π |
|---|---|---|---|---|---|
| r(φ) | 0.3678794411714423 | 1.214461 | 1.961309 | 2.513997 | 2.718281828459045 |

→ **SURVIVES.** Same functional at two resolutions; the earlier empirical scan's
`0.3669` / `2.7178` differ by `0.000979` / `0.000482`, both at or under that
scan's own grid step of `0.0009970`. Resolution, not convention. Endpoints exact
at `1/e` and `e`. The substitution `z = a·c` matches at **exactly 0.0** over
twenty trajectories.

**H§9b — the reduction to the head**
→ **VOID.** Two independent routes: under the gate route there is no complex
quantity to reduce (modulus row bitwise phase-independent); under the logit route
the operator output is exactly real (max imaginary part `0.0` across 20 cells).
The direct measurement misses by +7.6% to +49.1% at 15 of 20 cells. The cause is
derived: a static operator application has no self-referential feedback loop, so
its row-sum tracks `e^a`, while `c*` has an essential branch point at `a = 1/e`
*only because* the census recursion feeds `c` back into itself. The Lambert
bifurcation is right about the scalar map and unconnected to this operator.

**H§11a — the Krohn–Rhodes nulls**

| null | per-seed error | correct construction |
|---|---|---|
| shuffled operator assignment | 0.5045 / 0.4970 / 0.5175 / 0.5170 / 0.5160 | 0.0000 |
| orthogonal reset substitute, `det = 1` | 0.5445 / 0.5465 / 0.5445 / 0.5420 / 0.4645 | 0.0000 |

→ **SURVIVES.** Both nulls scored on the task's own argmax label metric — the
same functional as the arm — at n = 2000 per seed. The entire 2×2 orthogonal
family was swept (721 angles, both determinant signs, singular values exactly
`[1, 1]`) and the best member still errs 0.393–0.544. The bed had shipped **no
null at all** before this; under the standing precondition it was void, and it is
now alive.

**H§11b — the replacement line gate**
→ **RESTATED.** It passes both placebo tests the original failed (letter-shuffle
0.00%, word-salad 2.40%, both under the 5% void line) and the length confound is
**worse**: 60.0% rejected under 40 chars, 0.0% above 250 — a spread of 0.60
against the struck gate's 0.4172. Better mechanism, larger symptom. The
within-bucket control that would settle it was not run.

**H§11c — the recall bar**
→ **RESTATED.** `0.8262` (404/489), Wilson [0.7901, 0.8572], clearing four
no-parsing floors without interval overlap (strongest `0.3579` [0.3166, 0.4013];
`0.4013 < 0.7901`). It measures **concordance, not correctness** — two imperfect
taggers disagree on 17.4% when nothing is broken — so a below-bar reading cannot
separate an extraction bug from ordinary tagger ambiguity. Honest number, weak
bar, and the page says so.

**H§12a — the dilation identity**
→ **SURVIVES.** σ = 0 and σ = 1 included **deliberately rather than left to a
random draw** — which is the fixed-structure move, and it found the real corner:
the divergence is at σ = 0 (`arccos(0)=π/2`, `cos(π/2)=6.123233995736766e-17`),
not σ = 1 as an earlier draft claimed. The shipped implementation is the
direct-product route, exact at both corners. Attribution corrected to Halmos 1950
/ Sz.-Nagy 1953; Stinespring 1955 is not the citation.

**H§12b — exactness at absorbing states**

| route | value at the absorbing state |
|---|---|
| resolvent read | `0.0` exact |
| TD(0) after 20,000 sweeps | `2.470e−323` |
| sigmoid gate after 20,000 steps | `0.03817905137506084` |

→ **VOID as a differentiator, SURVIVES as a measurement.** `2.470e−323` is a
subnormal float64 value; no tolerance anyone would set can register the gap. The
sigmoid's `0.038` is gradient starvation on a sigmoid, a fact about gradient
descent and not about value estimation. Exactness at absorbing states buys
nothing measurable against the method it was supposed to beat.

---

### I — `docs/PHASE_I.md`

**I§1a — the Murphy residual `2.776e-17`**
Fixed: the decomposition is checked by substituting back **the same bin
probabilities it was built from**.
→ **VOID.** A bin-substitution tautology. It could not have failed, and the
project's own standing rule — *a bar whose branch never fires is worth nothing* —
applies to it directly. Neither the bed nor the residual exists anywhere in the
tracked tree; both live in an uncommitted scratchpad.

**I§1b — reliability is free**
→ **SURVIVES**, and it is a theorem rather than an observation: temperature
scaling is **monotone on logits**, and resolution scores the ranking, which a
monotone map preserves exactly. Measured: reliability `0.02407 → 0.00012` (×200)
at fitted `T = 2.742` against injected `k = 2.5`, resolution flat across
`0.044–0.046`. The bed's own credibility check fires first: a base-rate
forecaster scores reliability `0.0` — best of every arm — and resolution `0.0` —
worst. The joint bar `reliability_CI_upper ≤ 0.001299 AND resolution_CI_lower >
0.0` is the correct repair. Margin stated: recalibrator drift is 2.0–2.2% and the
sharp-vs-mis-scaled resolution difference is 3.2%, **inside the drift**.

**I§1c — the conformal row**
→ **VOID on its own bar**, correctly declared. Split conformal coverage is
**marginal, not conditional**; the pre-registered bar treated it as a per-seed
floor and it holds 7 of 15. The ruling — *a marginal guarantee licenses a
population-level refusal rate, never a per-instance refusal claim* — is the right
restatement. The NEVER leg fires honestly: far state 12, `0/3000` reached,
rule-of-three upper bound `3/3000 = 0.00100`.

**I§1d — the chess ceiling at `max_plies` 80 → 400**
Fixed: same generator, same seed, one keyword; `max_plies=80` puts **98.67% of
games in SINK**.
→ **SURVIVES.** Ceiling `0.000501` CI95 `[6.09e-05, 1.04e-03]` → `0.101170` CI95
`[0.0708, 0.1329]`, **202×, non-overlapping**, `RES/UNC` 1.9% → 18.0%, and the
same arm goes from `1.11×` at `p = 0.337` to `3.99×` at `p = 0.000`. Reproduced
bit for bit on rebuild: `0.10116955630126778` against the recorded `0.101170`.
Both ceiling readings are the same oracle-resolution functional, so the ratio is
a quantity. This is the row that proves the instrument works.

**I§1e — the committor's own Murphy decomposition**

| against | RES gap | CI95 |
|---|---|---|
| base rate | +0.001469 | [+0.001041, +0.004009] |
| ply-bucket histogram | −0.014518 | [−0.020625, −0.006879] |
| piece-count histogram | −0.000662 | [−0.002380, +0.001812] |

Fixed: 1,973-parameter `TinyCEQ`, 4,000 steps, held out **by game**; scorer is
binned one-vs-rest RES; the piece-count arm reads 8 bins of total piece count
computed from **the operator's own 769-float input** — strictly less information.
→ **RESTATED.** The three RES-vs-RES gaps are one functional and they stand. The
sentence that does not stand is the one comparing this binned RES to the
**unbinned** ceiling; see I1§4 below. The verdict — the operator ties or loses to
one scalar summary of its own input, and its raw Brier `0.6334` is worse than the
marginal predictor's `0.5020` even after recalibration at `T = 2.9395` — rests on
the same-functional rows and survives.
The binned identity residual is `0.00188`, 0.26% of Brier, **not machine
epsilon**, and is reported as a real discretization gap; the marginal-predictor
row on the same code path residuals at `1e-16` because it is piecewise constant.
That is the correct way to report a residual, and it is the exact contrast to
I§1a.

**I§2a — the spurious-zero dtype grid**

| dtype | S | gate | entries reading bitwise `0.0` |
|---|---|---|---|
| bfloat16 | 64 | 0.2002 | 21 |
| float32 | 64 | 0.15 | 45 |
| float64 | 4096 | 0.5 | 4,564,731 |

Fixed: constant gate, **no closed gate anywhere**, so every reported zero is
spurious by construction. 12 of 16 cells wrong.
→ **RESTATED.** The counts are measured on `path_product` **in isolation**. The
full gated forward cannot reach bfloat16 at all — `gate()` calls `torch.polar`,
which rejects it — so the bf16 column describes a primitive and not a shipped
forward. `float16` is accepted and silently returns **complex128**. The float32
and float64 columns are unaffected, and float64 at S=4096 is where the
certificate fails worst. Mechanism named and bounded: not compounding —
`prefix_logit_mask_restated` proves `|pathProd| ≤ 1` so no gain above one exists
— plain underflow.

**I§2b — the `seg` bind test**

| test | shipped `path_product` | `seg` | ground truth |
|---|---|---|---|
| S=64, bf16, no closed gate | 21 refusals | 0 | 0 |
| S=64, one gate at k=32 | — | exactly 1024 pairs | 1024 |
| S=4096, three gates | — | bit-for-bit | — |

→ **SURVIVES.** RED on the float route, GREEN on the segment route, in all three,
against ground truth rather than against agreement. Shipping leg measured as
segment-wise SDPA against dense reachable-key-masked SDPA on the same Q/K/V:
`max_abs_diff = 2.384e-07` against `5e-2`. Prior art conceded exactly: Blelloch,
IEEE TC 38(11) 1989 / CMU-CS-90-190 1990, combinator `⟨a,p⟩ + ⟨b,q⟩ = ⟨a·¬q + b,
p ∨ q⟩`, and the 1989 IEEE paper is recorded **UNREACHED** after two DNS failures
with the attribution resting on the tech report read directly.

**I§3 — the degree law**
→ **SURVIVES.** `K = ceil( ln(2C/((1−ρ)ε)) / ln(1/ρ) ) − 1`, `C = (1+ρ²)/(1−ρ²)`
reproduces 11 / 34 / 130 exactly, no fitted constant, at ε = 1e-6, indexed by γ
and **not by sequence length**. Stated as a trade rather than a speedup. The one
sentence the page adds and must keep: **the filter has never been raced against
attention** — the struck lever raced it against a dense solve, a different
comparison.

**I§4a — the cost table**

| N | D | measured bf16 | fits a 9-hour cap |
|---|---|---|---|
| 12M | 240M (20×) | 0.354 h | yes |
| 25M | 500M (20×) | 1.535 h | yes |
| 50M | 1.0B (20×) | 4.915 h | yes, 45% margin |

→ **RESTATED.** Computed from `6·N·D` against **measured achieved** throughput,
not MFU × attained peak — attained peak drifted 20% between two runs on this card
(`27.349 → 31.762` TFLOPS) while achieved reproduced to 3% (`13.575` vs
`13.205`). The corpus correction is folded in: the "67,108,864-token corpus" is
the `max_bytes` **cap** in `load_open_text`; the file is `18,167,706` bytes,
`3.69×` less. Every MFU figure on that page is qualified by the drift.

**I§4b — the head-dimension cliff**

| `d_head` | bf16 / fp32 | SDPA median | bf16 peak memory |
|---|---|---|---|
| 44 | 0.943× | 14.753 ms | 4,750 MiB |
| 32 | 2.953× | 0.464 ms | 1,879 MiB |

→ **SURVIVES as an ablation.** Parameter count held identical at 12,281,280 and
`d_model = 352` fixed, moving only `n_heads`. Mechanism named and verified live:
at `d_head = 44` both fused backends are rejected
(`can_use_flash_attention` False, `can_use_efficient_attention` False) and the
math kernel materialises `[B, H, S, S]`. It re-explains an earlier MFU `0.1208`
that appeared to fail a `0.20` gate as this artefact; at the default shape
measured MFU is `0.496`. **The 31.8× headline attached to it is separately
RESTATED — see I1§5.**

**I§5 — the gate init / trained table**

| | init | trained |
|---|---|---|
| exactly `0.0` | 49.3% | 59.3% |
| geomean over nonzero | 0.2425 | 0.9024 |
| mean run of consecutive nonzero `m` | 1.023 | 0.685 |
| longest live path anywhere | 17 | 7 |

Fixed structure, verified live in the tree rather than from the page:
`ceq/hf/modeling_ceq.py:439` builds `self.m_head = nn.Linear(d, 1)` and
`_init_weights` (line 554) runs `module.bias.data.zero_()`. The `m = 0.999`
initialisation belongs to `ArmSMPrime.trainable_heads()`, **which the HF path
never calls**. Confirmed present in the working tree today — the I.1 repairs are
proposed as diffs and none is applied to `ceq/`.
→ **VOID as first drawn, RESTATED after the repair.** With `m_head.bias = 0.999`,
changing nothing else: init exactly-zero `0.4915 → 0.0182`, trained
`0.5969 → 0.3121`, trained max run `10 → 42`, and the final loss **falls**
`1.5618 → 1.3122` (last-50 `1.2665 ± 0.0399` vs `1.5018 ± 0.0352`). Pairing
verified by a detrended per-step loss correlation of `0.9832` against a shuffled
control at `−0.0156`. Two further corrections inside the corrected row: `max 7`
was read at `eval_seed=999` while both runs used `12345` (same checkpoint reads
`max 10`), and "stretch" overstates live reach by ~1.7× because `G_ij` runs
backward from `i`. Scope: 1.8% of one epoch on a children's-story corpus with no
held-out channel.

**I§5b — no checkpoint has a gate**
→ **SURVIVES.** Every `results/**/*.pt` loaded with `weights_only=True`; the
union of weight-key names across all of them is **eight names**, none of them a
gate parameter; 158 of 165 carry a `state_dict` and every one is
`kind='softmax', beta=0.5`. A complete enumeration with an exact count is the
right shape for this claim.

**I§6a — the COGS checkpoint bar `0.04930`**

| step | in-distribution | generalization |
|---|---|---|
| 3,000 | 0.93359 | 0.02148 |
| 6,000 | 0.94531 | 0.13672 |
| 9,000 | 0.94531 | 0.14648 |
| 12,000 | 0.94922 | 0.09766 |

Fixed: `ceq/capability.py:39-40` — `RESOLUTION_FLOOR = 0.20`, `WIN_MARGIN = 0.02`
(verified in the tree), over a recorded control generalization of `0.02930` from
a run of 3,000 steps = **3.97 epochs** of the 24,155-item train split.
→ **VOID.** The bar was set from one checkpoint of a non-monotone curve; the
control sits 2.0×–3.0× above its own challenger bar from 6,000 steps on, so
`0.04930` measures training length and not architecture. And **no replacement bar
can be set**: the curve has not flattened by 12,000 and peaks at 9,000, so every
candidate value is a point the run has already crossed in both directions. The
curve itself is a property of the weights — four checkpoints re-scored in a
separate process returned the same counts, paired worst-case McNemar `χ² ≥ 43.0`,
`p < 1e-10`.

**I§6b — the gate on COGS, `0.0000` at all three seeds**
Fixed: `tests/chase/cogs/cogs_control.py:37` — `GATE_EVAL, GEN_EVAL, MAX_NEW =
256, 512, 192`, with the file's own comment recording the quantization floors
`1/256 = 0.0039` and `1/512 = 0.00195` (verified in the tree).
→ **SURVIVES as a null.** 0 of 1,536 items across three seeds. If the arm's true
rate were the bar, `P(0 of 1536) = exp(−1536 × 0.0506) ≈ 1e-34`, so the verdict
does not depend on where the void bar sat. The arm was verified to be the gated
one **at runtime rather than from config**: instrumented `readout` fires exactly
`n_layers` times with no fallthrough to SDPA, all 28 new tensors receive nonzero
gradients, and loading the smprime state dict into the softmax arm shifts logits
by `1.128` against an RMS of `0.579`. Mechanism measured at the COGS shape:
**97.99% of the causal operator's entries annihilated before training starts, each
query seeing 1.93 of 192 keys** — an **underfit** (train loss `0.407–0.485`
against the control's `0.273–0.288`), not a generalization failure.

**I§6c — the `0.113` fixed-seed swing**
→ **RESTATED.** Three of four seed-0 CUDA runs of the identical recipe land
inside `±0.03` and one swings `0.113` absolute, from SDPA kernel selection and
embedding-gradient scatter-add. On `GEN_EVAL = 512` with floor `1/512 = 0.00195`,
`0.113` is **58 items flipping** — which points at the evaluation subsample and
the exact-match scorer, not at training. `ceq/harness.py:313`'s `eval_indices`
**ignores its own seed argument** — verified live at line 323,
`manual_seed(20260825 + n_test)`, still unpatched in the tree — so checkpoint
comparisons are perfectly paired and no subsample-luck estimate is reachable.

---

### I.1 — `docs/PHASE_I1.md`

**I1§1 — the clamp is the freeze**

| start | at exact 0 | at exact 1 | carrying gradient |
|---|---|---|---|
| bias 0 | 0.501 | 0.160 | 33.8% |
| bias 0.999 | — | 0.498 | 34.3% |

Fixed: `magnitude = clamp(u, 0, 1)`, gradient **zero outside the open interval**;
LayerNorm-normalised rows at d = 512, weights `N(0, 1/d)`, 20,000 samples,
float64; closed form `Φ(1) − Φ(0) = 0.34134` and `Φ(0.001) − Φ(−0.999) =
0.34150`; stable to swapping the draw order (`0.50135 / 0.16125`).
→ **SURVIVES**, and it is a model instance of this law: it audits the
*parameterization* and finds the freeze there, then shows the data could not have
produced it — `m_head.bias` moved within `0.02` from **both** starts
(`0 → −0.0172 / −0.0021 / +0.0068 / −0.0101`, `0.999 → 0.9797 / 0.9987 / 0.9908 /
0.9990`).

**I1§2a — the certificate-shape parameter formula**
Fixed: `P = 12·L·d² + 2·vocab·d + seq·d`, reverse-derived and confirmed exact on
**two existing shapes** before being trusted for a new one (`d=512, L=8 →
25,690,112`; `d=512, L=4 → 13,107,200`).
→ **SURVIVES.** That is the correct order: check the fixed formula against known
values first. The sensitivity of the corpus decision is stated rather than
assumed — the 25M row fails only below 565,617 rows, so the fetched count would
have to be overstated 3.79×.

**I1§2b — the repetition table**

| row | params | tokens at 20/param | repetition on the 18.2M file |
|---|---|---|---|
| 12M | 13,107,200 | 262,144,000 | 14.43× |
| 25M | 25,690,112 | 513,802,240 | 28.28× |
| I1§6 LM row | 4,929,536 | — | 5.4267× |

→ **VOID, retroactively, for every LM comparison built on this corpus.** The
regime forbids the sentence: a gate that helps or hurts memorization says nothing
about generalization. This is the largest single VOID in the audit by number of
rows it reaches. The repair is real and derived rather than recalled — full
TinyStories at `2,141,709 × 908.39 = 1.945e9` bytes puts 12M at `0.135×` and 25M
at `0.264×` — and the pin is enforced as a **gate on the run**, not a promise:
the lane prints `bytes_total`, `tokens`, `repetition` and labels the row
`MEMORIZATION REGIME` whenever `repetition ≥ 1`.

**I1§3 — the catalogue of checks that cannot fail, 12 rows**
→ **SURVIVES**, and it is the audit-of-audits this law wants. Two entries deserve
naming because they are live instances of the same defect this document is
ruling on:
- row 1: a **constant** added to every `m_head` weight lay **exactly in
  LayerNorm's null space** (row sum zero, `8.1e-6` absmax); measured `3.5e-6`,
  a random perturbation moves logits `0.716`.
- row 12: the span-containment curve tracks `1 − (1−p)^L` to three decimals, so
  it **measures the density it was handed** rather than a mechanism.
The audit struck its own first pass on the way: a grep for `taskkill` against a
line reading `Killed PIDs` — a search with no true-positive path, found inside
the audit for checks that cannot fail. And the watchdog counted `Stop-Process`
**attempts** rather than verified deaths; PID 9000 was reported killed three
times and was still running.
**Open, and correctly labelled open:** ~140 further tolerance assertions located
and not individually verified; five spot-checked, none decorative.

**I1§4 Verdict 1 — the operator represents non-commutative composition**
→ **SURVIVES, with the scope in the signature.** See H§8b. Signed.

**I1§4 Verdict 2 — the operator predicts. Not signed.**
The stated reason is: *recalibrated resolution `0.001469` against an oracle
ceiling of `0.10117` is 1.45% of the resolution the bed offers.*
→ **RESTATED.** The verdict stands; the reason does not. `0.001469` is scored by
`murphy_binned_onevsrest(..., 10)` — ten bins — and `0.10117` is reused verbatim
from `wil_res_ceiling.py` as an **unbinned** per-class variance
(`CEILING_PROVENANCE` in `tests/foreman/chess_sweep/s1_sweep.py:78`). Those are
different functionals and `1.45%` is not a quantity. The verdict survives on the
rows that are one functional: the piece-count histogram — 8 bins of the operator's
own 769-float input, strictly less information — ties it on resolution with the
point estimate in the histogram's favour and beats it on Brier, `0.5020` against
`0.6334`. **Not signed, for the piece-count reason, not the ceiling reason.**

**I1§5a — the guards table**
| guard | fires | silent |
|---|---|---|
| head dim `% 8` | `d_head` 44, 33, 12 | 32, 64 |
| forced-fused SDPA | `d_head=44` bf16, verified live | 64 bf16; 32/64 fp32 |
| host spill | L=9: peak `8578.1 MiB` vs `8187.5 MiB` card, **no OOM**, status `ran` | L=6, L=7 |
| depth default | refuses L=7 without `acknowledge_headroom`, hard-refuses L=8 | L=6 |
| dtype refusal | bf16 and fp16 at construction | fp32; `sgate`; `signed` |
→ **SURVIVES**, five of six fire under an independent trigger rather than a
report. The dtype guard's placement is itself a fixed-structure finding: it
raises through an `_apply` override because **HF modules are cast after
construction**, so an `__init__` check would never see the dtype that crashes.
The **host-spill guard is signed at L=1–4 only** — the checker could not
re-observe the L=9 fire (card at `7837/8188 MiB` under two sibling lanes, runs
starved past 25 minutes), **killed nothing and waited**, and the L=9 fire is
labelled as resting on one agent's observation.

**I1§5b — three contract numbers that did not reproduce**
→ **RESTATED, all three.**
- Head-dim penalty: `31.8×` is a **bfloat16** defect, not an fp32 one. At fp32
  `d_head=44` is `1.41×` slower because memory-efficient attention *accepts* it
  on this box. The reproducing case is bf16 `d_head=44` at **12.22×**, peak
  `183.4 MiB` against `16.1`. The substitution was made rather than the
  contract's figure reported as a measurement.
- The `exp(S·5.2e-4) − 1` rounding law is wrong in **functional form**:
  re-measured relative error `0.079524 / 0.281715 / 0.734220 / 0.995026` at
  S = 64 / 256 / 1024 / 4096 — it **saturates toward 1.0** rather than growing to
  `7.41`. Trusted over the formula because the S=4096 point matches an
  independently recorded `gate3.md` Cell A figure of `0.99510` to four
  significant figures.
- The crossover is **S = 75, not S = 256**. The first pass reported 256 as the
  first S over the `0.094` bound; that was an artefact of testing only four
  powers of four. Bisected, it is 75. Consequence: the three-channel path is
  **fp32-mantissa permanently**.

**I1§5c — the memory law**
→ **SURVIVES.** Re-fit live from L = 6, 7, 9: `peak_mib = 929.96·L + 208.56`,
R² `0.9999999458`, within `0.06 MiB` per layer of the contract's
`929.9·L + 208.3`. Same functional, independent re-fit, agreement stated as a
residual per layer rather than as "matches".

**I1§6a — the held-out split**
→ **SURVIVES.** 105,095 documents, 94,585 train / 10,510 val, **zero document
overlap**, checked **by searching for the leak rather than by reading the code**:
0 of 96 mid-document 200-character windows appear in train; 2 of 200 whole
held-out documents match verbatim and both are corpus fragments (`"The end."`,
8 chars; `"Are you OK?" Lily asked.`, 25 chars). The leak it closes is priced at
**56 bytes — 0.0031% of the 1,795,595-byte val split**. Corpus-wide duplication
is 704 copies (0.67%), only 2 at ≥100 characters, so duplication is not a second
channel. Both the correctness and the smallness are stated.

**I1§6b — the determinism null**
→ **VOID as a null about this project's problem.** Six runs returned
bit-identical eval-loss trajectories, final `2.249588042497635` in every one —
but the `0.113` swing lives in `ceq.lm`'s literal SDPA path and `CEQForCausalLM`
has **zero `scaled_dot_product_attention` call sites**. The model used cannot
exhibit the mechanism. **The `0.113` swing is untested, not unremoved.** The
must-fire's band was set from repeat-determinism variance because **no seed was
varied across the six runs** — the procedure was sound, the quantity was wrong.

**I1§7a — the R1 kill**

| | (i) live gradient ≥ 0.95 | (ii) bias moves ≥ 0.05 | (iii) init zeros < 0.05 |
|---|---|---|---|
| straight-through | PASS `1.0000 / 1.0000` | **FAIL `0.01923`** | PASS `0.01825` |
| hard-concrete | PASS `0.9978 / 0.9863` | **FAIL `0.02757`** | PASS `0.00000` |

→ **SURVIVES.** Pre-registered before any number, fired cleanly on condition (ii)
for both forms, and the `(ii)` failure was confirmed on **all four layers read
out of the safetensors** rather than the single scalar the race reported. Giving
every gate a live gradient did not unfreeze the bias — both moved inside the same
`~0.02` band the zero-gradient clamp moved in.

**I1§7b — the result the must-fire does not capture**

| | last-50 loss | trained exact-zero | backward reach mean / median / max |
|---|---|---|---|
| repaired clamp | `1.2665 ± 0.0399` | 0.3121 | 2.146 / 1 / 42 |
| straight-through | `1.2999 ± 0.0439` | 0.4050 | 1.209 / 1 / 13 |
| hard-concrete | **`1.1391 ± 0.0401`** | **0.0006** | **241.4 / 234 / 512** |

→ **RESTATED, and the restatement is the finding.** Hard-concrete **does not
achieve** its reach — it **preserves** it: at initialisation hard-concrete already
reads `256.5` and the other two read median `52`, mean `79.07`. Training moves
hard-concrete `256.5 → 234` and moves the other two `52 → 1`. The honest sentence
is *two forms destroy their initial reach and one does not*, which is a different
claim from an earlier draft's. And straight-through gives every gate an unbounded
live gradient and the gate **still** closes, to `0.4050` — worse than the clamp it
was built to rescue — so the collapse is not only the dead zone.

**I1§7c — the zero-density account of gate closure**
Fixed: the three rows above all ran at `m_head.bias = 0.999`
(`tests/chase/gate/r1_gate.py:260`), whose measured init zero densities are
`0.018250 / 0.018250 / 0.000000` — **not 0.50**. The premise "at bias 0 the clamp
puts a zero every second token" describes the **pre-repair** init, measured at
`0.491455`, and the pre-repair run is not in that table.
→ **VOID.** It explains a table it is not about. And the table refutes it without
new training: straight-through's forward is *bitwise* clamp, so at seed 0 it
carries the identical init density `0.018250`, the identical span curve and the
identical init reach `52 / 79.07` — then ends at a different trained zero fraction,
a different max reach and a different loss, off a **bitwise-identical step-0
loss**. Same step-0 density, different endpoint.
The crossed cell built to save it was **void by construction**: clamp at
`bias = +3.0` reaches zero density `0.0` with `m == 1.0` at **100%** of positions
and `frac_grad_nonzero = 0.0000`. At `u_std = 0.443641` the clamp's live interval
is `2.254σ` wide, so the lowest bias reaching hard-concrete's density is `1.336`,
where **77.6% of gates are already dead at m = 1**. No clamp cell holds that
density with a live gradient at this init scale.

**I1§7d — R7, the spurious-zero instrument**
→ **SURVIVES**, and it is the correct order: the instrument was **self-tested
before it was trusted** (synthetic S=16 with one true zero, `n_true_zero = 55`,
exact) and then cross-validated against this project's own recorded defect
(constant `m = 0.5`, S = 4096, float64, **4,564,731 spurious zeros, an exact
match**, with S = 64 float64 correctly reading 0). Zero spurious zeros on every
measured cell of the race.

**I1§8 — S1, the chess re-binning**

| tier 2, seed 0 | fixed-width (10) | equal-count (10 / 200) |
|---|---|---|
| operator | 0.002155 | 0.003769 |
| twin | 0.001102 | 0.001721 |
| **piece-count floor** | 0.006761 | **0.006993** |

Fixed structure, read from the producer: `tests/foreman/chess_sweep/s1_sweep.py:274`
scores `murphy_binned_onevsrest(q, k_te, bed.nA, 10)` — **10 fixed-width
one-vs-rest bins** — against `CEILING = 0.10116955630126778` reused verbatim at
line 78 from `wil_res_ceiling.py`, an **unbinned per-class variance**.
→ **VOID.** Scorer and ceiling are different functionals, and the page's own
evidence shows it is not a small difference: `white_win` and `black_win`
contribute `res_k` of **exactly 0** at both seeds because every recalibrated
forecast lands in bin 0 — the scorer is blind to rare-class resolution while the
ceiling counts all four classes. **Re-binning inverts the ordering**: under
fixed-width the operator sits `2.3×` above the floor; under equal-count it sits
`3.1×` below it, and the floor reaches `0.069` of ceiling — **above the 0.05 bar
the operator fails**. The interval contains the bar at both seeds
(`[0.00415, 0.07521]`, `[0.00436, 0.08228]`), so the kill fires on the point
estimate only. Three further claims do not hold: `RES/ceiling` does not rise with
size (every per-arm trend direction flips between seeds), the arms separate at
**zero of three sizes**, and the operator-minus-twin gap sign-flips at all three
tiers between seeds. "Trained to convergence" means the early stop became
eligible only past `0.8 × budget` and all twelve runs stopped within ~180 steps
of eligibility; at tier 1 the operator's final `L_q` is `0.985` against the
twin's `0.082` on the identical objective. The page's own ruling is correct and I
endorse it: **`fen_to_vec` must not be touched on this evidence.**

**I1§9 — S2, the eval-only prescription**

| readout | annihilated | live keys / 192 | median backward reach |
|---|---|---|---|
| frozen clamp | **98.3579%** | **1.5846** | 2 |
| hard-concrete | **0.0%** | **96.5** | 92 |

→ **VOID.** Frozen clamp and hard-concrete are **different functions over
identical weights**: in-distribution `0.0000` (0 of 256) under the frozen readout
against `0.3125` under the training readout, same parameters. "Freeze to clamp at
eval" is not a freeze, it is a substitution, and it reconstructs the broken arm's
`97.99% / 1.93` almost exactly. The arm is a milder underfit (train loss `0.3494`
vs control `0.2731`) and that is **not** the cause of the zero, since the same
weights score `0.3125` read under their own training form.
The freeze verifier was itself defective and **predicted before the run reached
step 3000**: `blend()` computes `magnitude(lerp(ones, u, g))`
(`ceq/arm_smprime.py:134`) while the verifier compared against
`clamp(u_raw, 0, 1)`, and `g` is a trainable parameter (line 725) drifted to
`0.8699419498443604`. It read `freeze_took = false` at max difference `0.1299`
and **halted the row on a false alarm**. Compared on the blended argument:
`bitwise_equal_to_clamp_forward = true`, max difference `0.0` over n = 768 from a
real eval forward. **One seed, on a bed with a measured `0.113` fixed-seed
swing** — a data point.

**I1§10 — Kaggle seed variance**
```
finals  2.2976 2.2973 2.2973 2.3029 2.2981 2.2982 2.2981 2.3010
swing   0.005563836097717267
stdev   0.001885094358229673
```
→ **RESTATED.** Across-seed swing `0.0056` on the SDPA-literal arm against the
`0.113` fixed-seed COGS swing — and the page correctly calls the comparison
**suggestive rather than decisive**, because these are different seeds on a
byte-level LM scored by loss against the same seed four times on COGS scored by
exact match. Different functionals, different models, and the page says so
instead of taking the 20× ratio. Three defects repaired before the push, the
first of which would have **inverted the finding**: the notebook computed
`repetition = tokens_seen / (20·n_params)`, the **reciprocal** with the wrong
numerator, so a larger model scored a lower factor and would have been labelled
generalization-admissible. The certificate shape is **632,496** parameters, not
the 720,896 the closed form predicted — a 12.3% shortfall, reported.

---

### The tally

| | count |
|---|---|
| SURVIVES | 25 |
| RESTATED | 18 |
| VOID | 13 |

**Seven of the thirteen VOIDs are two mechanisms, and only two.**

*Scorer and ceiling are different functionals* — I1§8 (chess: 10 fixed-width bins
vs unbinned per-class variance), I1§9 (COGS: frozen clamp vs hard-concrete
readout over identical weights), I1§4 Verdict 2's stated reason (the same chess
mismatch inherited), I§1a (Murphy residual: the decomposition checked against its
own inputs), H§6c (recall: the system under test *is* the oracle), H§5 (L2 on a
sphere vs a composition-order question).

*The regime forbids the sentence* — I1§2b (5.4×–28.3× repetition forbids any
generalization sentence), I§6a (a non-monotone curve crosses its own bar in both
directions, so no bar exists), I1§7c (the account describes a `0.50` density
table while the table ran at `0.018250`).

The remaining VOIDs are singletons: G§5b (fails its own pre-registered bar),
H§6b (a letter-multiset statistic asked a word-order question), H§9b (no complex
quantity exists on the route it reduces), H§12b (the gap is a subnormal),
I§1c (a marginal guarantee asked a per-instance question), I1§6b (the model
cannot exhibit the mechanism the null was written for).

---

### (A) — the two live verdicts, ruled

**The I-AUT order result — its fixed structure supports its sentence. SURVIVES.**
The scorer is argmax accuracy on 800 eval items; the ceiling is argmax accuracy
on **the same 800 items**. One functional. The eval floor `1/800 = 0.00125` is
three orders below the `0.5760` separation. The corpus has **no repetition** —
3,000 train and 800 eval words drawn disjointly from 2¹⁴ = 16,384. Parameters
matched by live `numel()` rather than by formula (404 vs 404). The generator
reaches the arms **only as opaque integers**, so no arm's forward pass can be the
generator. The kill was written before the numbers and did not fire. The control
is saturated within `0.031` of its own ceiling. The sentence it supports, exactly:
*at 404 parameters on a generator presented only as symbol ids, a non-commuting
matrix gate reaches `0.8620 ± 0.0556` where a commuting diagonal gate saturates at
`0.2860 ± 0.0150`, at one word length and one parameter budget, with no
length-generalisation check.* Everything in that sentence is licensed.

**The chess prediction result — its fixed structure does not support its sentence.
VOID.** Not inverted: void. `RES/ceiling = 0.02495 < 0.05` divides a 10-fixed-width-bin
one-vs-rest resolution by an unbinned per-class variance. Those are not the same
quantity, so the ratio is not a number and the bar it is compared to is not a
threshold on anything. The page's own re-binning proves the mismatch is
load-bearing rather than cosmetic — it **inverts the pass/fail ordering**, and an
8-bin summary of the operator's own input clears the bar the operator misses.
What survives the void is the same-functional comparison underneath: the operator
ties the piece-count histogram on resolution and **loses to it on Brier**,
`0.6334` against `0.5020`. Verdict 2 stays unsigned. It stays unsigned for that
reason and not for the ratio.

---

## HALF TWO — THE STANDARD FOR A HOP

### The shared shape, stated precisely enough to gate on

Hasenjaeger was handed a 100-character message and audited the part of the Enigma
that **changes with the key** — rotor choice, rotor positions — and found a real
weakness there. He missed the part that **never changes**: the reflector, which
makes the permutation a fixed-point-free involution.

The cost of that fixed part, in the exemplar's own currency:
`26! = 88.4` bits of a free permutation against fixed-point-free involutions
`25!! = 7,905,853,580,625 = 42.8` bits. **45.5 bits lost**, of which the bare
no-self-map rule accounts for `1.44` and the self-reciprocal convenience for the
rest.

Wheeler 1937 and Heisenberg through the 1940s did the opposite. The S-matrix
describes only the fixed asymptotic boundary — in-states to out-states, **with no
account of the path** — and the structure that does not move is unitarity and the
`+iε` prescription. Lippmann–Schwinger `T = V + V G₀ T` with
`G₀ = (E − H₀ ± iε)⁻¹` makes the Born series the Neumann series of the resolvent,
and its convergence is a property of the **fixed** operator: `ρ(G₀V) < 1`, the
same shape as `γ·ρ(P) < 1`. Instance at `E = 4 + 0.05i`: `ρ = 0.2596`, truncation
error `1.9e−4 / 2.0e−7 / 2.7e−13` at 5 / 10 / 20 hops.

Both share three properties, and the third is the reframe:

1. **The audited object is the one that cannot move under the system's own
   rules.** The reflector's involution is wired in; unitarity is not a fitted
   parameter.
2. **The immovability is priced exactly, in bits or in a spectral radius.** Not
   "the reflector is a weakness" but 45.5 bits; not "the series converges" but
   `ρ = 0.2596` with the error at three truncation orders.
3. **The impossibility is then read as a filter, not as a refusal.** Not "the
   machine declines" but "the machine has eliminated." A 20-letter crib survives
   a random alignment with probability `(25/26)²⁰ = 0.456`, and **every surviving
   position is a NEVER-certified hypothesis**. The Bombe searched that reduced
   set. The `+iε` prescription is the same move on the physics side: causality
   turned into a branch selection rule that eliminates one of two solutions.

The leap seat's failure mode, visible across all four of its outputs: **a
plausible mechanism aimed at the wrong object.** The segment bit aimed at a
target that was already Blelloch's. The three-channel gate aimed at a bf16 saving
that the mantissa cannot deliver. The zero-density account aimed at a table that
ran at a density of `0.018250` rather than the `0.50` it assumed. Only the
Chebyshev degree law aimed at the object it named.

Neither exemplar has that failure mode, and the reason is structural: **both
audit the part that does not move, and both check it against the actual machine
before saying what it costs.**

---

### (B) — THE HOP GATE

Five conditions. **C1 and C5 are mandatory; the bar is 4 of 5.**

> **C1 — Fixed before trainable, and checked against the real target.**
> The leap names which quantity is fixed structure and which is trainable, in
> that order, and it **measures the fixed quantity on the actual object it claims
> to explain** before the mechanism is proposed. Assuming the fixed value fails
> this condition even when the mechanism is coherent. *(Mandatory.)*
>
> **C2 — Fetched prior art with exact figures, at dispatch.**
> Page, section, quoted line, exact number — not paraphrase from memory, and not
> a concession made after binding. An unreachable source is named as unreached
> and nothing is cited to it.
>
> **C3 — Reframes an existing object rather than patching one.**
> It states what the thing provably *is*, replacing a story rather than adding a
> special case that keeps an old story alive.
>
> **C4 — Falsifiable by a measurement that already exists.**
> A number already in the tree, or already recorded, would come out differently
> if the leap were wrong — and it is named before it is run. If no such
> measurement exists, the leap names the one that would have to.
>
> **C5 — The proposer ran their own instance and reported it when it failed.**
> In the record, before dispatch. Passing numbers alone do not satisfy this.
> *(Mandatory.)* The author's own S-matrix instance is the standard: the first
> sat `E` inside the band and diverged at `ρ = 1.6688`; it was corrected and
> reported **unscored** rather than the `E` quietly moved.

#### Applied to the four past leaps

| leap | C1 | C2 | C3 | C4 | C5 | score | dispatched? |
|---|---|---|---|---|---|---|---|
| 1. segment bit for exact refusal | ✓ | ✗ | ✓ | ✓ | ✓ | **4/5** | **yes** |
| 2. three-channel gate to evade a complex dtype | ✓ | ✗ | ✓ | ✓ | ✓ | **4/5** | **yes** |
| 3. zero-density account of gate closure | ✗ | ✗ | ✗ | ✓ | ✗ | **1/5** | **no** |
| 4. exact degree law for Chebyshev | ✓ | ✓ | ✓ | ✓ | ✓ | **5/5** | **yes** |

**1. Segment bit — 4/5, dispatched.** C1 ✓ and it is the leap's whole strength:
it names the Boolean zero-ness of `m_k` as the quantity that does not move and
the float product as the one that does, and it measured the float one on the
**shipped** `path_product` across all 16 dtype cells first — 21 spurious refusals
at S=64 bf16, 4,564,731 at S=4096 float64. C4 ✓, RED/GREEN against ground truth
rather than against agreement, plus a shipping-leg bind at `2.384e-07`. C5 ✓ on
the spirit: the limits section names the instance that would kill it — a bf16
gate value rounding to `0.0` without the clamp firing — rather than leaving it to
a checker. C2 ✗ **at dispatch**: the Blelloch concession arrived after binding.
The construction is correct and not novel, and knowing that before dispatch would
have changed what was raced.

**2. Three-channel dtype gate — 4/5, dispatched.** C1 ✓: the dtype path is fixed
structure and the leap names it. C4 ✓ and it fired hard — the `exp(S·5.2e-4) − 1`
form was measured wrong in **functional form**, saturating toward `1.0` rather
than growing to `7.41`. C5 ✓ and this is the exemplary instance: the proposer's
own first pass reported the crossover at `S = 256`, found it to be an artefact of
testing only four powers of four, **bisected it to S = 75 and published the
correction**. C2 ✗: no prior art was fetched for the three-channel decomposition
at all. The leap is half alive exactly where the gate predicts — the mechanism
holds, the bf16 saving it was for does not, because a bf16 mantissa fails below
any sequence length this project runs.

**3. Zero-density account — 1/5, not dispatched.** C1 ✗, and it is the whole
failure: the account named a fixed quantity (init zero density) and **never
measured it on the table it was explaining**. It assumed `0.50`; the table ran at
`0.018250 / 0.018250 / 0.000000`. The `0.50` belongs to the pre-repair init
(`0.491455`), and the pre-repair run is not in that table. C5 ✗: no instance was
run before dispatch; it died in a checker's hands. C3 ✗: it replaced one
unproved mechanism story (gradient starvation) with another, and the crossed
design built to test it was void by construction — clamp at `bias = +3.0` pins
`m == 1.0` at 100% of positions with `frac_grad_nonzero = 0.0000`. C2 ✗: no
prior art. C4 ✓ is its one virtue and it is a real one: the claim was falsifiable
by a measurement already in hand, and it was falsified **without new training** —
straight-through carries a bitwise-identical step-0 density and step-0 loss and
ends at a different endpoint.
*This is Hasenjaeger's error with the polarity reversed.* He audited the right
machine and missed its fixed part. This audited a fixed part and never checked
which machine it was on.

**4. Chebyshev degree law — 5/5, dispatched.** C1 ✓: `K` is a property of the
fixed function `1/(1−γx)` and its pole at `1/γ`, with **no trainable part
anywhere**, and it is `O(1)` in sequence length — the leap says which axis it
does and does not live on. C2 ✓ and it is the model instance: ChebNet was
**fetched and contradicted the claim** (it uses K=25 and K=5 and cites Hammond
et al. 2011; it does not supply 11/32/112), and the replacement is a four-line
Bernstein-ellipse derivation rather than borrowed authority. A lane correctly
declined to substitute Hammond without fetching it. C3 ✓: the degree is not a
fitted constant, and the cost sentence was reframed from "cheaper" to "a trade at
an exchange rate of √horizon". C4 ✓ and it fired against the project's own
published numbers — the old K missed its own stated `1e-6` at two of three gammas
(`2.481441e-06` at γ=0.9, `1.149069e-05` at γ=0.99) — and was re-derived by an
independent route (DCT-I over 16,385 extrema on a 1,040,005-point grid), with the
LP minimax floor 11/33/125 over 60,001 points re-verified on 2,000,001. C5 ✓: the
proposer's own prior formula (prefactor 1) was the thing killed, and the failure
was published **as a table with the exact sup at both K and K−1** rather than
quietly replaced.

**The gate discriminates.** It does not pass all four and it does not fail all
four. The single condition that separates the one failure from the three
survivors is **C1 in its strong form — check the fixed quantity against the
actual target before proposing the mechanism** — which is the reflector's own
lesson applied to this project's own leaps. C5 separates it a second time,
independently.

*Note on the dispatch's own draft scoring:* it gave the zero-density account
"2/5 — it fails only condition 1". Those are inconsistent: 2/5 fails three
conditions, not one. Ruled 1/5, with C4 the only pass, for the reasons above.

---

### (C) — THE HARD ONE: is there a filter reading of this project's NEVER?

The project has **two** structural NEVERs and they behave differently. The answer
is no for the one it sells and yes-but-worthless for the one it does not.

#### The NEVER this project sells: the gate's exact zero. No filter reading.

`G_ij = ∏ m_k = 0` exactly means key `j` is unreachable from query `i`. It is
sold as a refusal a threshold cannot imitate. It cannot be read as a search
filter, for a reason that is structural and not a matter of effort:

**The reflector's NEVER is fixed; this one is trained.** The reflector's law is
wired into the machine and holds for every key, so an attacker knows it before
seeing any ciphertext — which is exactly why it filters. The gate's zeros are a
function of `m_head`'s weights, which move. There is no statement of the form
"this pair can never contribute" that holds independently of the run.

**And the reflector's filter carries a survival guarantee that this one does
not.** The true crib alignment is *guaranteed* to survive the no-self-map test,
because the true plaintext was encrypted by a machine obeying that law. Nothing
guarantees the answer survives the gate. The measurement is on the page: at the
COGS shape, **97.99% of the causal operator's entries are annihilated before
training starts, each query seeing 1.93 of 192 keys, and the arm scored exact
match `0.0000` — 0 of 1,536 items.** The filter fired, eliminated 98% of the
candidates, and eliminated the answer with them. A filter whose survivor set is
not certified to contain the answer is not a filter; it is a guess with a high
rejection rate.

So the honest answer for the gate: **there is no crib to place.** The elimination
is not computable before the model runs, it is not stable across runs, and when
it ran at scale it deleted the answer.

#### The NEVER this project does not sell: closure refusal. A filter reading
#### exists, it already ran, and it bought nothing.

`SingularTransientBlockError` is a genuine fixed-structure NEVER. A closed class
has an exactly-`1.0` diagonal in the transient block, so `I − Q` is exactly
singular and no committor exists — verified by running it on the pinned bed in
`docs/C9_NEVER_RESIDUE.md`. No weight can change that; it is a property of the
partition, not of the numbers.

Read as a filter it is exactly the Bombe's shape, and C10 in `docs/PHASE_G.md` §6
is the instance. Over **4,424 enumerated proper subsets**, the exact-closed to
exact-transient split reads `3/11`, `27/35`, `147/107` and `3123/971` at the four
planted `k`. At the largest, the structural test eliminates **3,123 of 4,094 —
76.3% — before any resolvent is solved**, and it eliminates them exactly: 0
disagreements between `Fraction` arithmetic and float64, with the identical split
reproduced by an independent graph-closure algorithm. The 971 survivors are
NEVER-certified in precisely the reflector's sense — each one is a hypothesis the
structure did not kill.

**And it bought nothing, for one reason: 4,424 is enumerable in full.** The
Bombe's elimination was worth something because `26!` was not. C10 tested every
subset directly, so the filter saved no work it did not also perform. A filter
over a set you can afford to enumerate is a description, not a search.

#### What would have to exist for it to become one

Three conditions, and the third is the one nothing in this tree meets.

1. **A candidate set enumerable in principle and unaffordable in practice.** The
   closest thing already in the tree is the chess bed at `max_plies = 400`, where
   the legal-move set is enumerable and the branching factor is real. Not 4,424.
2. **The elimination computable strictly cheaper than the thing it replaces.**
   Closure is a graph reachability test, `O(edges)`; the resolvent read is a
   triangular solve, `O(n³)`. The asymmetry is already there.
3. **A certificate that the answer survives.** This is the missing one and it is
   the whole gap. The reflector has it for free, from the machine's own law.
   Here it would have to be a theorem of the form *the true absorbing outcome is
   never in a class the closure test eliminates* — and the pinned bed shows why
   that is not free: `committor` **hard-codes every declared absorbing index to
   the one-hot row of `eye(k)` regardless of what the matrix says there**
   (checked directly — setting `W[DRAW, WIN] = 0.5` still returns `0.0` for
   DRAW's committor to WIN). So `DRAW` reads `0.0` by **embedding convention, not
   by dynamics**. Using that as a filter would be using a coding convention as
   evidence, which is the same defect as scoring a tautology.

Until (3) exists, the closure NEVER is a refusal with a filter's shape and no
search to constrain. That is the honest absence, and I am not going to dress it
up: **this project's NEVER has no crib to place, because it has no search that
cannot afford to enumerate itself.**

---

### (D) — one sentence for the page, and the next row

**Page sentence.**

> Every table in this repository is now audited against its own fixed structure
> before its trainable part, and thirteen are void — seven of them by two
> mechanisms only, a scorer measured against a ceiling that is a different
> functional, and a corpus regime that forbids the sentence the row was written
> to make.

**Next row, dispatchable as written.**

`tests/foreman/chess_sweep/s1_sweep.py` scores every arm with
`murphy_binned_onevsrest(q, k_te, bed.nA, 10)` at line 274 and divides by
`CEILING = 0.10116955630126778`, reused verbatim at line 78 from
`wil_res_ceiling.py` as an **unbinned** per-class variance. Recompute the ceiling
through `murphy_binned_onevsrest` on the oracle forecasts, at the same `n_bins`
and the same bin scheme the arms are scored with, for both schemes
— 10 fixed-width and 10 equal-count — and re-report `RES/ceiling` for `operator`,
`twin`, `piece_count_hist` and `base_rate` at all three tiers and both seeds.
**No retraining: it re-scores saved forecasts.** Estimated cost is minutes.

Pre-registered kill, written before the numbers:

> If, under the matched functional, the piece-count floor still clears `0.05`
> while the operator does not, the encoding sentence is dead on its own terms and
> `fen_to_vec` stays untouched. If operator and floor cross the bar **together**,
> then `0.05` was a threshold on the binning and not on the encoding, and the S1
> kill is withdrawn rather than restated.

Default, already chosen: run it. It changes a live verdict, it costs minutes, and
it is the only row in this audit whose void can be lifted without training
anything.
