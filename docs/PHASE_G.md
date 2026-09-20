# Phase G — spectral

Phase G asked whether the operator's phase axis carries anything, whether the
closed-class certificate discriminates, and whether a Chebyshev read of the
resolvent is cheaper than solving it. Three of its rows died. The two that
survived did so at a narrower claim than the one they were dispatched to make.

Every number below carries its producer. Deciding numbers come from this
machine — Windows 10.0.26200, Intel64 Family 6 Model 183, 28 logical CPUs,
Python 3.11.9, torch 2.14.0+cpu, numpy 2.4.6, scipy 1.17.1. Rows are marked
with the checker that signed them and, where a claim was struck, with what
struck it.

---

## 1. Scoreboard

| row | claim dispatched | verdict | what survives |
|---|---|---|---|
| S1 | phase folds into q, k | **holds, float64** | fold-in exact to `2.398082e-13`; the "free" half struck |
| S5 | phase is free in a fused kernel | **repriced** | ~2 ULPs, 0.66% of attention time at n=4096 |
| S6 | learned θ beats fixed schedules | **void, then rerouted** | the fixed schedule is the right *initialization* |
| C10 | closed-class count is recovered | **holds, discriminating** | 4,424 subsets, exact and float64 agreeing |
| C11 | Chebyshev filter beats a dense solve | **cost lever struck** | accuracy holds; the degree formula was wrong |
| C12 | flux spectrum reads the planted holonomy | **holds, self-contained** | three must-fires at 1e-15; no shipped code exercised |

---

## 2. What the phase axis actually does

S1 asked whether the phase gate folds into a rotation of `q` and `k`. It does.
At n = 4096, d = 64, over 5 seeds in float64, the worst absolute fold-in error
is `2.398082e-13` against a bar of `1e-12`, driven through the shipped
`ceq.arm_phase.phase_factor` with an instrumented call count of 5/5.

Two separate things were then struck.

**The provenance sentence.** The lane printed that it had used the shipped
function; an instrumented counter read
`{'phase_factor': 0, 'path_product': 0, 'scan_phase': 0, 'gate_smp': 0}`. It had
built a local copy. The number survived a re-run through the shipped path; the
sentence did not. Two later lanes found the same defect independently in
`phaseg_S1_foldin.py`.

**The cost claim.** "The phase is free" does not appear in the source it was
attributed to. RoFormer §3.4.2 states that applying the rotation directly as in
Equation 16 *is not computationally efficient*, and supplies a separate
element-wise realization because of it. The rotation is efficient, not free.

### The gap that explains the rest of the phase

The autopsy found something neither S1 nor S6 could have seen from inside their
own beds. RoPE rotates `q` and `k` **before** the dot product, so its phase
moves the real attention logit — it changes which tokens attend to which. In
`ceq/arm_smprime.py::numerator`, θ multiplies the score **after** it has been
exponentiated.

Measured directly: the real modulus row `R_ij · e_ij` and the normalizer `Z_i`
are `0.000e+00` different between `θ = 0` and `θ ~ N(0,1)`.

The phase does not touch the attention pattern at all. No setting of β, g or qk
routes it into the real logit, and a repository-wide search for any q/k rotation
returns nothing, including `ceq/hf/modeling_ceq.py`. So the corner claim splits:
**softmax is β=1 exactly, and RoPE is not contained**, because RoPE is softmax
on a rotated score rather than vanilla softmax.

A second lane reached the same wall numerically from the other side: the
imaginary half of the twist is discarded bitwise,
`Re((w·twist) @ v) == (w·Re(twist)) @ v` at max difference `0.0`.

---

## 3. Half precision, and a retracted kill

The first S5 gate computed every step of the phase construction in the half
dtype and read **61.513×** the dtype floor at float16, with bfloat16 unable to
run at all because `torch.polar` refuses `BFloat16`. On that evidence both race
lanes correctly refused to run and built no stand-in baselines.

That kill did not survive a fairer recipe. Computing angles, the cumulative sum
over positions and `cos`/`sin` in float32 — cached once, because they depend on
position and not on data — then casting and applying to half-precision `q` and
`k`:

| recipe | float16 | bfloat16 |
|---|---|---|
| all-half (what was first gated) | 61.513× floor | shipped call fails 0/5 |
| fp32 table, cast, apply | **4.001× floor** | **4.001× floor**, 5/5 succeed |

Against float64 truth rather than route agreement: `3.915×` floor at float16
(relative `1.18e-03`) and `4.079×` at bfloat16 (relative `9.41e-03`). The
control that decides it — plain `q @ kᵀ` with no phase at all, same dtype, same
seeds — costs `1.612×` and `1.705×`. The phase therefore costs roughly two extra
ULPs on top of rounding already being paid.

The 61× has a cause: `Φ = cumsum(θ)` random-walks to about ±60 radians, where
float16 spacing is 0.0625 radians. It was position arithmetic in a storage
dtype, which no production kernel does.

**A ratio over a floor is not self-interpreting.** The identical recipe-B code
path at float32 reads `72.253×` its floor at an absolute error of `2.4e-04`,
because the floor shrinks faster than accumulated arithmetic noise does. No bare
multiple appears in this document without its absolute and relative error and a
do-nothing control.

---

## 4. The Chebyshev row

### The degree formula was wrong

The formula in use took `sup |f − S_K| ≤ ρ^K/(1−ρ)` — prefactor 1. For
`f(x) = 1/(1−γx)` the true bound carries a prefactor that grows as `γ → 1`:

```
sup |f − S_K|  ≤  2C ρ^(K+1) / (1 − ρ),     C = (1+ρ²)/(1−ρ²)
```

| γ | K published | K correct | sup at K | sup at K−1 |
|---|---|---|---|---|
| 0.50 | 11 | **11** | `4.321033e-07` | `1.612631e-06` |
| 0.90 | 32 | **34** | `9.748701e-07` | `1.555340e-06` |
| 0.99 | 112 | **130** | `8.916143e-07` | `1.027669e-06` |

At the published K the old degrees missed their own stated `1e-6` at two of
three gammas: `2.481441e-06` at γ=0.9 and `1.149069e-05` at γ=0.99.

The checker re-derived the corrected degrees by an independent route — a DCT-I
over 16,385 Chebyshev extrema, evaluated on a 1,040,005-point grid — and
obtained the same K. He also corrected the claim of tightness: K−1 failing shows
the *truncated series* is tight, not the *problem*. The certified minimax floor,
the smallest degree at which some polynomial meets `1e-6`, is **11 / 33 / 125**
by linear program over 60,001 Chebyshev-clustered points, re-verified on
2,000,001. Chebyshev truncation sits within 1, 1 and 5 degrees of optimal.

### The cost lever is struck

The pre-registered prediction was that the sparse filter beats a dense solve at
density ≤ 0.1 for N ≥ 1024. At density exactly 0.1 the dense solve wins **6 of 6
cells, by 2.75× to 54×**. The crossover survives only at γ=0.5, density 0.01,
N ≥ 1024 — 2 of 12 target cells. At γ = 0.9 and 0.99, the regimes where a
resolvent read actually needs a degree-K filter, the solve wins at every density
and every N tested. Pinning `OMP`, `MKL` and `OPENBLAS` to one thread to kill
the core-count confound left 12 of 12 unchanged in sign.

The corrected degrees make this worse, not better: filter cost is linear in K,
so it rises ×1.000 / ×1.062 / ×1.161 and pushes an already-absent crossover
further out.

**Replacement route — reprice, not retire.** Against `dense_factor_plus_solve`,
a fresh Cholesky per call, the filter wins 9 of those same 12 cells. If `Q`
changes every training step the factorization cannot be amortized, and that is
the regime where the filter earns its place. Both baselines are reported; the
amortized solve is headlined because it is the stricter one.

### The accuracy margin was a property of the bed

The bed's `Q` was a dense random symmetric matrix rescaled by the single scalar
`0.999/row_max`. That collapses the spectrum: at N=1024 the eigenvalues lie in
`[-2.538e-02, 9.627e-01]` with exactly **one** above 0.1. The reported
"max gap over the spectrum" was therefore the interpolant's error at a single
point near zero. The arithmetic proof is that `|p_K(0) − 1| = 2.7394e-07` and
the worst reported cell over all nine was `2.730e-07` — the same number.

Rebuilt on a spectrum-filling `Q` (path graph, eigenvalues spanning
`[-0.99900, 0.99900]`), the must-fire still passes at worst `6.857e-07`. The
pass is real. The margin is about **1.5×** at γ=0.99, not 100×.

---

## 5. The Fourier bed, and why its headline was withdrawn

S6 was the only model-scoring row in the phase. Its **null lived**, which no bed
in this project had previously managed: arm (a) with `θ_k` shuffled across
positions scored worse on at least one instance at every one of 5 seeds, 106/160
instances, 0 unscored. That precondition — *a bed whose null scores nothing is
void before the model runs* — is the only reason any of the following is
readable.

The row was then withdrawn on five defects, four of which were found after its
numbers had already been reported.

1. **Arm (a) had no phase to learn.** The imaginary half of the twist is
   discarded bitwise. The row was not testing a learned phase.
2. **The two lanes implement different operators under one label.** The
   stationary lane rotates `q`/`k` pre-softmax; the chirp lane reweights
   post-softmax. The 0.130-nat stationary headline is not a learned-versus-fixed
   comparison and is quarantined as unreadable.
3. **The floors are ±logit-pinned**, overstating margins 5–7×. Chirp last-value
   calibrates to `0.5408`, not `1.3900`. Stationary FFT-linear calibrates to
   `0.6725` — worse than chance, an anti-signal rather than a floor.
4. **The headline gap is not significant.** Paired, `b − a` gives `t = −1.59`
   (51/80). Only `c − a` reaches significance at `t = −3.50` (58/80).
5. **The two regimes ran at different widths** — `d_model=64, hidden=128` against
   `d_model=16, hidden=32`. Two beds, not one sweep.

### The reroute

Promoting `ArmRoPE.freqs` from `register_buffer` to `nn.Parameter`, initialized
at the schedule it already used and with nothing else changed, beats every arm
S6 ran, on **5 of 5 seeds at every budget**:

| budget | paired gap vs fixed RoPE-32 | t |
|---|---|---|
| 150 steps | `−0.0138` | `−3.69` |
| 600 steps | `−0.0331` | `−12.83` |
| 2400 steps | `−0.0342` | `−14.58` |

for +32 parameters on 1,937. At the original 150-step budget it scores `0.3867`
against the old winner's `0.4004`.

The effect is initialization, and parameter count is excluded by construction:
`f_learn4_randinit` and `e_learn4_ropeinit` carry identical 1,941 parameters and
the same seeds, differing only in start point. `e` wins at every budget and `f`
loses at every budget.

**Undertrained or unnecessary — answered.** Arm (a) is not undertrained: the
paired `c − a` gap *widens* with budget, and the bed streams fresh signals every
step so it cannot overfit. Arm (a)'s own mechanism decays toward the identity as
it trains, `theta_absmean` falling `0.1776 → 0.1213 → 0.0791` across the three
budgets. Given sixteen times the budget it learns to switch its own phase off.
Learning was not unnecessary; the wrong learnable was.

What replaces the "kernel, not a model" reading is narrower and testable: **the
fixed schedule is the right initialization, not the right answer — a kernel at
init, a model thereafter.** The one datum still arguing the older way is kept
intact: arm (d), with no attention at all and 1,425 parameters, beats arm (a) by
as much as arm (b) does.

---

## 6. Closed classes and flux

**C10 discriminates.** Over 4,424 enumerated proper subsets the exact-closed to
exact-transient split reads 3/11, 27/35, 147/107 and 3123/971 at the four
planted k, with **0 disagreements** between exact `Fraction` arithmetic and
float64, and the identical split reproduced by an independent graph-closure
algorithm. A uniform pass would have proved nothing; the split is what makes it
evidence.

The residue is `1.0666665e-06` at γ = 1−1e−6 and `1.0666667e-07` at γ = 1−1e−7 —
a clean factor of ten, which *measures* the O(1−γ) claim rather than asserting
it. A single γ, which is what the lane first reported, could not have
distinguished that from a coincidence.

The eigenvalue-1-multiplicity half of C10 is excluded from its bars, correctly:
planting k and reading k back is a construction property, and its two stated
must-fires are the same theorem twice by the Laplacian result the contract
itself cites.

**C12 holds and is self-contained.** Cycle `6.661e-16` against a bar of `1e-9`,
path `4.001e-16` and gauge `1.776e-15` against `1e-12`, with β₁ computed rather
than asserted. It exercises no shipped repository code, and it cannot: a search
for any magnetic Laplacian across every `.py` in the tree returns nothing. Both
the operator and its closed form live inside the test file, so the three
must-fires are two derivations by one author agreeing with each other. That is
genuine — a wrong `L^(q)` would not land on the closed form at four fluxes and
on zero at every flux on the path — but it is not independent corroboration, and
its two routes to β₁ are rank-nullity stated twice.

---

## 7. Instruments

Seven instruments have now been withdrawn for scoring a property of the draw
rather than of the model. Three structurally different cures were tried in this
phase.

**A paired null** — score `f(model, draw) − f(null, draw)` on the same seed so
additive draw terms cancel. Untestable: the null fired on zero instances at all
five seeds, leaving precision undefined and no ratio formable.

**A precondition that raises before scoring** — modelled on a guard in a
sibling project. It died on its own false-positive test: run unchanged on a new
draw-property instrument it had never seen *and* on a legitimate instrument that
genuinely measures its model, it raised on **both**, via the same check, for the
same reason — `m=1, n=1000` on the fraud and `m=2, n=1000` on the honest one.
Zero discriminative power. Its backward pass over the seven corpses was honestly
3 clean, 2 only by stretching, and 2 not at all.

**A resolution bound** — and this one works. It is the pigeonhole generalization
of an orbit-error theorem: if `n` distinct behaviours land on only `m` distinct
observable values, at least `n − m` are indistinguishable *to that instrument*,
certified from a single run with no ground truth and no null required to fire.

| corpse | naive reading | resolution bound |
|---|---|---|
| gate flat across four budget decades | `m = 4` (worthless) | **`m = 1`** |
| twenty numeric answers sharing one token | — | **`m = 1`**, `n = 20`, indistinguishable 19 |
| the refusal bed that killed the paired cure | precision `0/0` | **`m = 2`, `n = 10`, zero collisions** |

The third row is the one that matters: model recall `[1.0]×5` against null recall
`[0.0]×5` separates cleanly at a noise scale of exactly 0.0. The paired diff
needed a *ratio*; the separation was there the whole time.

The bound's honesty is in what it refuses. Two corpses received no `m` at all,
because no defensible noise scale existed in the recorded data and the tool
declined to invent one — the same two the precondition also failed to cover.
Two independent methods agreeing on where the boundary lies is worth more than
either claiming all seven.

A companion, `certified_gap`, ports a published frontier rule — compare the
worst case of everything inside a selected set against the best case of
everything outside it, rather than the k-th against the (k+1)-th, which is
unsound — and supplies concentration intervals over seeds in place of the
rounding bound it was built for. It certifies only when a Student-t and an
empirical-Bernstein interval agree, and otherwise refuses while naming the two
arms and the exact deficit. On the S6 chirp gap it refuses and prices the
question: **n = 116** seeds under t, **n = 406** under empirical-Bernstein. A
synthetic control certifies under both, so a run of refusals is not a broken
function.

---

## 8. Citations

Four papers were fetched and checked against what this project claimed of them
before any table cited them. All four contradicted the claim.

| claim | what the source says |
|---|---|
| ChebNet supplies K = 11/32/112 | not in that paper; it uses K=25 and K=5 and cites Hammond et al. 2011 |
| RoPE: the phase is free | the rotation is *efficient*, not free (§3.4.2) |
| FNet is fixed causal mean mixing | it is a 2D DFT, and causality there is an open problem |
| MagNet supplies holonomy and gauge invariance | neither word appears in the paper |

None had reached a published table, which is the only reason this was cheap. The
ChebNet attribution never reached the tracked repository at all, and its
replacement — classical Bernstein-ellipse truncation for a function with a
single real pole at `1/γ` — is a four-line self-checkable derivation rather than
borrowed authority. A lane correctly declined to substitute Hammond et al.
without fetching it first.

The FNet correction arrived after the fact and was unnecessary: both S6 lanes
had already found the causality problem themselves, masked causally, and
labelled the arm a stand-in in their own file headers. The arm is not an oracle.
It is also not FNet, and the results table's `d_fnet` label carries that caveat.

---

## Limits

Every deciding number here is CPU-only on one machine; nothing in this phase ran
on the certified 4060 and nothing was reproduced on Kaggle. The S5 timings are
float32 CPU measurements over 20 repetitions and must be re-timed on whatever
device a fused kernel would actually use; the half-precision conclusions rest on
CPU emulation of those dtypes. C10 and C12 exercise no shipped repository code —
C10 by choice, C12 because the object it tests does not exist in the tree. The
S6 reroute is a pilot at one width on one bed, and its pre-registered
reproduction bar, together with the null it still lacks — shuffling the learned
frequency vector across its slots after training — has not yet been run. The
resolution bound covers five of seven recorded corpses and declines the other
two. The counts of what the process caught are this document's own
classification, not an independent measurement.
