# R9 MATHEMATICS SURVEY — what this project has not tried, and what it need not try

JUPITER / MYCROFT, iteration 4. HEAD `486ae41`.

**No wall-clock measurement was taken.** The `4.560600 s/step` rate below is an
input supplied to this survey, not a reading taken here. Every other number is
combinatorial, closed-form, or read out of a journal already on disk.

---

## 0. THE FRAMING IS WRONG IN A WAY THAT CHANGES THE ANSWER

The brief asks about *"discrete selection that is somehow being trained despite
the selection being non-differentiable."* There are **two** discrete stages in
this arm and they are not the same stage.

| | what is discrete | where | trained? |
|---|---|---|---|
| **Stage A** | **which `k` rows are pivots** | `m3_quintuple.py:169 batched_pivots` → `pivot_probe.py:80 select_pivots` = `topk(key.norm(dim=-1), k)` | **never, in any arm** |
| **Stage B** | the mixture weights `α` over the `k` already-chosen pivots | `m3_quintuple.py:440 _alpha` | yes — softmax / one-hot / STE / settled |

`argmaxste` puts its straight-through estimator on **stage B**. Its forward is
bitwise `argmax`'s — `soft - soft.detach()` is elementwise exactly `+0.0`,
confirmed over 200 drawn gates in float64, and `soft ∈ [0,1]` so no inf/nan path
exists. **Stage A is byte-identical in `argmax`, `softmax`, `argmaxste`,
`settled` and `twin`.** Nothing measured this round bears on stage A at all.

That distinction sorts the entire candidate field, so it is stated first.

---

## 1. THE GRADIENT IS THE EFFECT. THE MECHANISM IS NOT.

Recomputed from `results/m3_quintuple_v2.jsonl` at `ntr8192_nev512`, 5 seeds,
using the repo's own `m3_synthetic_settled.contrast` (paired percentile
bootstrap, `n_boot=10000`, seed 0). Positive = the second cell has lower NRMSE.

| contrast | delta | CI | bound | n+ |
|---|---|---|---|---|
| `argmax` → `argmaxste` | **+0.225760** | `[+0.212433, +0.245886]` | — | **5/5** |
| `softmax` → `argmaxste` | +0.107304 | `[+0.082879, +0.140870]` | — | 5/5 |
| `argmaxste` → `twin` | +0.004092 | `[−0.023107, +0.029187]` | 0.029187 | 3/5 |
| `settled` → `twin` | +0.002959 | `[−0.031557, +0.048587]` | 0.048587 | 2/5 |
| `argmaxste` → `settled` | +0.001133 | `[−0.069616, +0.057312]` | 0.069616 | 3/5 |

Cell means: `argmax 1.010779`, `softmax 0.892323`, `glance 0.892323`,
`settled 0.783886`, `argmaxste 0.785019`, `twin 0.780927`.

**Three structurally different stage-B mechanisms — a Neumann-settled fixed
point, a straight-through one-hot, and a full softmax mixture over 8 rows — land
inside `0.004092` of each other, every CI covering zero.** The one contrast that
excludes zero at 5/5 is the presence or absence of a gradient, worth
`+0.225760`. The mechanism is not the mechanism; the gradient is.

---

## 2. THE INSTRUMENT CANNOT SEE A STAGE-B IMPROVEMENT. THIS IS THE PRICING RULE.

Minimum detectable effect, paired, `α=0.05` two-sided, power `0.80`. Two paths:
the closed-form normal approximation, and an exact solve against the noncentral
`t` (bisection, with a guard — `scipy.stats.nct` returns `nan` at large
noncentrality and an unguarded bisection converges upward to a non-monotone,
wrong answer).

| paired sd | n=5 normal | **n=5 exact** | n=10 | n=20 |
|---|---|---|---|---|
| 0.022345 | 0.027996 | **0.037584** | 0.022256 | 0.014758 |
| 0.034451 | 0.043164 | **0.057946** | 0.034313 | 0.022753 |
| 0.050146 | 0.062828 | **0.084345** | 0.049945 | 0.033119 |
| 0.082152 | 0.102929 | **0.138179** | 0.081824 | 0.054257 |

**At 5 seeds this instrument resolves `0.057946` or larger.** The stage-B
mechanism differences are `0.004092`, `0.002959`, `0.001133` — **14×, 20× and
51× below the floor.** The gradient effect is `3.9×` above it, which is why 5
seeds read it at 5/5.

Seeds required, two paths (normal approximation / exact paired-`t`), and their
cost at the supplied `4.560600 s/step` × 150 steps = `684.09 s` per run:

| contrast | seeds (normal) | seeds (exact t) | runs | lower-bound wall clock |
|---|---|---|---|---|
| `argmax` vs `argmaxste` | 0.1 | **3** | 6 | 1.1 h |
| `argmaxste` vs `twin` | 556.3 | **559** | 1 118 | **8.85 days** |
| `settled` vs `twin` | 2 254.1 | **2 257** | 4 514 | 35.74 days |
| `argmaxste` vs `settled` | 41 263.5 | **41 266** | 82 532 | 653.46 days |

Those are **lower bounds**: the rate was measured for `settledrow` at
`n_train=2048` and the compared cells ran at `ntr8192`, four times the batch.

> **PRICING RULE.** Any candidate whose contribution is a better relaxation of
> **stage B** is competing for an effect this project cannot measure at any seed
> budget it could plausibly run. It is not that the mathematics is bad. It is
> that the result would be **unfalsifiable here**, and an unfalsifiable
> improvement is inadmissible under this project's own evidence rule.

This kills, on measurement rather than on merit, every stage-B candidate:
Gumbel-softmax, IMLE, SIMPLE, and perturbed optimisers applied to `α`.

---

## 3. THE STAGE-A EXPERIMENT IS ALREADY WRITTEN, AND HAS NEVER BEEN RUN

Stage A is the axis nothing has trained. Its search space is real: with `s=64`
and rows `0` and `s-1` excluded, there are **`C(62,8) = 3 381 098 545`** pivot
sets, and `topk(key.norm)` explores exactly **one**, deterministically.

**But it has been ablated once, and the ablation could not have said anything.**
`DONE_ARCHIVE_ROUND1.md:4707` records **K4**: `randpivot_signed`, `k=8` pivots
drawn uniformly and content-blind, slope `+0.081`, every CP interval overlapping
the content-selected arm, and the pre-registered consequence that *"content
selection is not load-bearing for M2"*.

**That null was forced by M2's design, not measured.** `scale/recall_probe.py:3-7`
states the identity: for any content-blind schedule of size `k`,

```
    P(c reachable) × (share | reachable)  =  (k/s) × (1/k)  =  1/s
```

bit-for-bit the dense rate, **independent of `k`**. M2 draws `c` *from* `P`, so
it measures the second factor and conditions the first away. Verified here on two
paths that fail differently — the symbolic factorisation, and a 200 000-draw
Monte Carlo over 12 `(s,k)` pairs, agreeing at `s=16,64,256,1024` (e.g. `s=1024`:
symbolic `0.00097656`, sampled `0.00099187`). **A content-blind schedule
reproduces the dense rate exactly, so K4 was structurally incapable of reading
anything else.**

`recall_probe.py:9-11` then names the quantity that is *not* an artefact:

> whether `P(c selected)` stays `Θ(1)` as `s` grows when selection is
> **content-conditional**. That is routing, and a random schedule provably
> cannot fake it.

**`scale/recall_probe.py` is imported by zero Python files, has no `results/`
artifact, and `DONE_ARCHIVE_ROUND1.md:5833` describes it as "already sitting
unrun".**

> **THE HIGHEST-VALUE MATHEMATICS IN THIS SURVEY IS NOT MISSING. IT IS WRITTEN,
> CORRECT, AND UNRUN.** Every differentiable-top-k, DPP and submodular-greedy
> candidate below is a way to *improve* stage-A selection. None of them is worth
> building until `recall_probe.py` has established that stage-A selection is
> load-bearing at all — on a bed that does not condition the recall factor away.

---

## 4. THE CITATION THE PROJECT'S CENTRAL ARGUMENT RESTS ON IS OVERSTATED THREE WAYS

`FINDINGS.md` C2 reads: *"Softmax is provably Bayes-optimal on exactly that shape.
One softmax layer attains Bayes risk where linear attention provably cannot."*
`arXiv:2410.01537`, ICLR 2025. C3 then makes the novelty claim `UNTESTED` on the
strength of it. The equations were fetched — abstract first (which says only
*"asymptotic Bayes optimality"* and *"a simplified version of a non-linear
self-attention layer"*), then the full text.

**Marion, Berthier, Biau & Boyer, "Attention layers provably solve single-location
regression."** Fetched: `ar5iv.labs.arxiv.org/html/2410.01537`.

| the repo says | the paper says |
|---|---|
| "softmax" | the predictor is `T_λ^{k,v}(𝕏) = erf(λ𝕏k)ᵀ 𝕏v` — an **erf** gate |
| "provably Bayes-optimal" | Corollary 2 gives **asymptotic** Bayes optimality; Theorem 1 is an exact risk formula, not an optimality claim |
| "linear **attention** provably cannot" | Proposition 3 is about **linear regression**: risk `ℛ(β⋆) = ε²+γ²−γ⁴/(γ²+L−1) → ε²+γ²`, the null predictor |

**And the regime.** Corollary 2 assumes *"joint asymptotic scaling where `d→∞` and
`L=o(d)`,"* with `λ√d→∞`, `λ√L→0`. This repo runs `L = s = 64` against
`D_MODEL = 16` (`scale/m3_capability.py:79`; the `d24` in journal keys is
`make_batch`'s flipper **offset**, `scale/negation_scope.py:84-86`, not the width).

```
    theorem needs   L = o(d)      i.e.  L/d → 0
    repo sits at    L/d = 64/16 = 4.00,  the opposite direction
```

**The label shape does match** — `Y = X_{J₀}ᵀv⋆ + ξ`, scalar, one informative
token at a latent position — so the repo's *qualitative* worry is well founded and
C2's instinct is right. But **"provably" is not earned at `L=64, d=16`**, the
result is asymptotic and regime-conditioned, the nonlinearity is `erf` not
softmax, and the impossibility result is against linear regression rather than
linear attention. `FINDINGS.md` C2/C3 should be restated. This is for whoever
holds `FINDINGS.md`; this survey does not edit it.

**Second path incomplete, and declared.** The ICLR proceedings PDF returned
compressed streams and could not be read, so the regime condition rests on **one**
fetch of the ar5iv HTML. Treated as `CITED, single source` rather than confirmed.

---

## 5. THE VECTOR-VALUED READOUT IS ALREADY BUILT

`FINDINGS.md` C's consequence reads as a proposal: *"dropping the `[:, s-1]` index
yields an `[n, s]` vector label for zero new parameters."* It is implemented.

```
    scale/m3_quintuple.py:112   ROW_CELLS = ("twinrow", "settledrow")
    scale/m3_quintuple.py:368   vector_readout: bool = False
    scale/m3_quintuple.py:483   return out if self.vector_readout else out[:, s - 1]
```

`READ`, all four line numbers verified. **Every vector-valued candidate in this
survey has its plumbing already in the tree**, at zero parameter cost. What does
not yet exist is a *label* that is genuinely vector-valued rather than one scalar
broadcast across positions — which is where §8's candidates earn or lose.

---

## 6. THE `nash.py` SHARED-`tau` DEFECT IS REAL, AND TOO SMALL TO BE THE CAUSE

The defect, `READ`:

```
    ceq/nash.py:64   return margin * float(torch.linalg.matrix_norm(m, ord=2).max()) / 4.0
    ceq/nash.py:146  t = tau if tau is not None else safe_tau(game.reshape(-1, s_len, s_len))
```

`.max()` over the batch, cast to a single Python `float`. **One temperature for
every instance, set by the worst-conditioned one.** Real, and correctly diagnosed.

**But it is priced, and the price is far too low.** `stance = 2σ(z/τ) − 1`, so for
small `z/τ`, `stance ≈ z/(2τ)`: amplitude is inversely proportional to `τ`, and an
instance run at `τ_batch` instead of its own `τ_i` is attenuated by `τ_i/τ_batch`.
Measured on drawn games built exactly as `ceq/nash.py:138-144` builds them
(causal-masked softmax, mean-centred, symmetrised), Gaussian `q,k`:

| batch | median shrink | p10 | min |
|---|---|---|---|
| 64 | 0.877526 | 0.839381 | 0.802094 |
| 256 | 0.851143 | 0.809502 | 0.766996 |
| **2048** | **0.789701** | 0.747352 | 0.690213 |

Two paths: the closed-form small-signal ratio, and the exact stance amplitude
through the sigmoid itself — `0.50 → 0.547752`, `0.25 → 0.281624`,
`0.10 → 0.113593`, the exact sitting just above the linearisation as sigmoid
concavity requires.

> **A `1.27×` attenuation on the median instance cannot produce an OOD NRMSE of
> `2.6151` to `5.8198`.** The recorded failure is 2.6× to 5.8× worse than
> predicting the mean; a 21% stance shrinkage is an order of magnitude too small
> to explain it.

**Consequence for the round.** The cheap diagnostic must run *before* the fix and
the rerun: **measure `matrix_norm(game, ord=2).max() / median` on a real batch.**
If that ratio is near the `1.27` a Gaussian surrogate gives, the shared-`tau` fix
provably cannot rescue the ordinal thread and the idea failed on its merits. Only
if the real spread is far wider — near-degenerate instances inflating the max — is
the bug a live candidate cause. That is one line and no training, against a rerun
that is neither.

**Limit, stated.** The draw uses Gaussian `q,k`, not the trained distribution, so
the shrinkage above is what the surrogate gives, not what the corpus gives. The
figure is a **surrogate lower bound on the attenuation**, and the real spectral
spread is exactly what the one-line diagnostic returns.

---

## 7. STAGE-A CANDIDATES — DIFFERENTIABLE TOP-K

Surveyed by a moon, equations fetched per candidate, re-checked by me where the
verdict turned on a single discriminating fact. **All of these are gated behind
§3**: each is a way to *improve* stage-A selection, and none is worth building
until `recall_probe.py` shows stage-A selection is load-bearing at all.

Every entry below passes the zero-`nn.Parameter` gate except where noted, because
each scores off the existing `kk = self.wk(x)` (`m3_quintuple.py:472`) and adds
only hyperparameters.

| candidate | defining object | cost at `s=64,k=8` | verdict |
|---|---|---|---|
| **SIMPLE** (Ahmed et al., ICLR 2023) | exact `k`-subset marginals by DP; **forward draws a real discrete `k`-subset**, backward uses exact marginals | `O(nk)` ≈ 512 ops/example | **strongest fit** |
| **Sander et al.** (ICML 2023) | `f_{φ,R}` reduced to isotonic regression, solved by PAV | `O(n log n)`, no linear solve in the backward | **strongest fit** |
| **Berthet et al.** (NeurIPS 2020) | `y*_ε(θ) = argmax_{y∈𝒞}{⟨y,θ⟩ − εΩ(y)}`, MC estimate `ȳ_{ε,M} = (1/M)Σ y^{(m)}` | `M` extra `topk` calls | viable |
| **SOFT / OT top-k** (Xie et al., NeurIPS 2020) | entropic OT with `k` in the marginal `ν=[k/n,(n−k)/n]ᵀ` | `O(nL)` Sinkhorn | viable |
| **Cuturi et al.** (NeurIPS 2019) | Sinkhorn rank/sort, `P*_ε = argmin⟨P,C⟩ − εH(P)` | `O(nmℓ)`, `ℓ` up to 100 | viable, priciest of the cheap |
| NeuralSort (ICLR 2019) / SoftSort (ICML 2020) | full `n×n` permutation matrix via `softmax` over pairwise differences | `O(s²)` | **expected to fail** |
| LapSum (2025) | scaled-Laplace CDF `F-Top_α` | `O(n log n)` | **fails if `α` is learned** |
| DFTopK (2025) | `σ(x − θ(x))`, `θ` located by Introselect | `O(n)` | **expected to fail** |

**The one verdict I re-checked myself, because it decides the gate.** The moon's
first fetch reported that Berthet's regulariser `Ω` must be *learned* — which
would have failed the parameter gate outright — and its second fetch reversed
that. I fetched it independently: `ar5iv.labs.arxiv.org/html/2002.08676`,
Proposition 2.1, *"Let Ω be the Fenchel dual of F₁, with domain 𝒞. We have that
`y*_ε(θ) = argmax_{y∈𝒞}{⟨y,θ⟩ − εΩ(y)}`"*, and *"εΩ is the Fenchel dual of F_ε"*.
**`Ω` is determined analytically by the choice of noise distribution and is never
fitted.** Berthet PASSES the gate. Appendix B confirms top-k is an instance
verbatim: *"It fits our framework over the set `𝒞 = {y ∈ ℝ^d: 0 ≤ y ≤ 1, 1ᵀy = k}`."*

**Why NeuralSort and SoftSort are expected to fail here**, stated plainly: both
spend `O(s²)` resolving a full permutation's worth of pairwise ordering, and the
only downstream consumer is a gathered `[n,k]` index set. Order among the
never-selected tail, and order *within* the chosen `k`, are both discarded. They
are dominated by construction, not by measurement.

**The architectural trap every one of them shares**, and it is measured in this
repo rather than argued: `scale/m3_flops.py:102-107` records the wall-clock ratio
`settledrow/settled = 4.560600 / 0.407800 = 11.19×` at `s=64, n=2048`, caused by
widening an `[n,k,k]` computation to `[n,s,s]`. **Any of these operators wired so
that a dense `[n,s]` relaxation flows downstream reproduces that `11.19×`
exactly.** All of them must be wired hard-forward / relaxed-backward, the pattern
`BatchedSettled` already uses — which is also what keeps the existing
`_bind_batched_against_arm_s` parity checks valid.

---

## 8. CONSEQUENCE-PROPAGATION CANDIDATES

Ranked on the axis that matters here: does the mathematics **naturally produce a
vector- or function-valued label**, or is it another way to compute one number?
§5 established that the `[n,s]` readout is already built, so a candidate that
supplies a genuinely vector-valued *label* is the scarce thing.

| candidate | defining object | vector-valued? | parameter gate |
|---|---|---|---|
| **SCM intervention as graph surgery** | `X = (I−B)^{-1}Z`; `do(X_i=a)` clamps row `i`. `ACE_i = B_do(1)[i] − B_do(0)[i]` | **yes** — one number per node | **PASS**, reuses `e4_harmonic.fixed_point` verbatim |
| **Persistent homology causal effect** (arXiv:2603.02289) | `ψ_d(t) := E[φ¹_{i,d}(t) − φ⁰_{i,d}(t)]`, a curve in `t` | **yes** — function-valued | PASS as a label |
| **Cellular sheaves** (Hansen & Ghrist 1808.01513; Bodnar et al. 2202.04579) | `Δ⁰_{v,v} = Σ F*F`, `Δ⁰_{u,v} = −F*_{u⊴e}F_{v⊴e}`; `ker Δ^k ≅ H^k` | **yes** — `d`-dim stalk per node | **FAIL as published**: restriction maps `ℱ = Φ(x_v,x_u) = σ(V[x_v‖x_u])` need a new `V` |
| **Hyperbolic entailment cones** (Ganea et al. 1804.01882) | `ψ(x) = arcsin(K(1−‖x‖²)/‖x‖)`; `S_x^{ψ(x)} = {y : Ξ(x,y) ≤ ψ(x)}` | yes, per position | conditional PASS with a fixed reprojection |
| **Shapley / Integrated Gradients** (1705.07874; 1703.01365) | `IG_i(x) = (x_i−x_i')∫₀¹ ∂F(x'+α(x−x'))/∂x_i dα`, completeness `Σ IG_i = F(x)−F(x')` | vector, but **explains an existing scalar** | PASS, needs no network change |
| **Influence functions** (1703.04730) | `ℐ_{up,loss}(z,z_test) = −∇L(z_test)ᵀH⁻¹∇L(z)` | **no** — scalar, or a vector in *parameter* space | PASS but irrelevant |

**Expected to fail, plainly.**

- **Influence functions.** Wrong axis entirely: they propagate consequence from
  *training example* to *prediction*, never from *entity* to *entity within one
  input*. Their vector form indexes the 4769 parameters, not sequence positions,
  so it cannot express "the action caused the travel". And `H` is `4769×4769`, so
  a direct solve is `O(p³) ≈ 1.08e11` flops. Most expensive candidate surveyed and
  the worst fit. **Recommend against.**
- **Cellular sheaves as published.** The one framework built natively around
  vector-valued node data, and it fails the parameter gate on its defining
  construction. The parameter-free repairs — fixing restriction maps to the
  existing attention weights, or reusing `wq`/`wk` as the maps — are `LEAD ONLY,
  NOT VERIFIED`; with a 1-D stalk the first collapses to the weighted graph
  Laplacian `kirchhoff.py` already builds, which is not new.
- **Hyperbolic cones on the shipped E4' case.** `scale/hyperbolic.py` and
  `tests/foreman/test_gromov_delta.py` already measure Gromov `δ̂` exactly
  (verified to `1e-12` against brute force, with tree `=0`, cycle `=n/4` and
  complete-graph `=0` controls). Cones only help on near-tree metrics, and the E4'
  two-lobe motif is not shown to be one. **The value of `δ̂` is that it predicts
  the failure before the run** — which is the untried step on that thread: use the
  already-measured `δ̂` to *gate* whether the cone experiment runs at all, rather
  than running it blind.

**The one with the cheapest decisive pre-check.** SCM graph surgery reuses
`fixed_point(q,r)` — machinery already cross-validated to `1e-10` by
`scale/kirchhoff.py` — and needs only a clamped row, not a new weight. Before any
training: **check whether `ACE_i` correlates near `1.0` with a column of `B`
already computed.** If it does, it is a relabelling of what `e4_harmonic.py`
already reports and is refuted for redundancy, cheaply, without a single training
step. That pre-check is the highest value-per-cost item in this section.

---

## 9. SUBMODULARITY — A PROVABLE GREEDY BOUND EXISTS, AND THE SIGNED ARM VOIDS IT

The coordinator asked whether the pivot objective has diminishing-returns
structure, because if it does the selector has *"a provable quality bound nobody
has claimed."* **It does, on one of the two natural objectives, under one
condition, and this repo's signed arm violates that condition.** Derived here and
checked numerically; not taken from any paper.

**The structure.** `B_P = A[:,P] A[P,:] = Σ_{p∈P} u_p v_pᵀ` with `u_p = A[:,p]`,
`v_p = A[p,:]` — a sum of rank-1 outer products, **one per pivot, each independent
of the rest of `P`**. That is the whole reason set-function structure exists here.
With `M_pq := (u_p·u_q)(v_p·v_q)`,

```
    f(P) := ‖B_P‖_F²  =  Σ_{p,q∈P} M_pq  =  1_Pᵀ M 1_P
```

a quadratic form in the indicator, so its second difference is **constant in `S`**:

```
    f(S∪{a,b}) − f(S∪{a}) − f(S∪{b}) + f(S)  =  2 M_ab
```

**Captured mass is SUPERMODULAR, not submodular.** For `A ≥ 0` — and a softmax
attention operator is nonnegative — every inner product is `≥ 0`, so `M_ab ≥ 0`
and the second difference is `≥ 0`: **increasing** returns. Measured largest second
difference `+21.196398` over 300 drawn matrices and every subset `S`. Greedy has no
`1−1/e` guarantee on this objective. Anyone reaching for submodularity on the
obvious objective gets the opposite of what they came for.

**Reconstruction IS monotone submodular, and the bound is real.** Expanding
`−‖A² − B_P‖²` leaves a constant, a modular term, and `−1_PᵀM1_P`, so with

```
    h(P) := ‖A²‖_F² − ‖A² − B_P‖_F²
```

the second difference is `−2M_ab ≤ 0` (**submodular**, largest observed
`−0.072313`), `h(∅) = 0`, and `h ≥ 0`. Monotonicity, derived rather than hoped:
with residual `R = A² − B_S` and `a ∉ S`,

```
    h(S∪{a}) − h(S) = 2⟨R, u_a v_aᵀ⟩ − ‖u_a v_aᵀ‖²  ≥  ‖u_a v_aᵀ‖²  >  0
```

because for `A ≥ 0` and `a ∉ S`, `R ≥ u_a v_aᵀ` elementwise. Checked over **21 936
`(S,a)` pairs**: `min(gain − ‖u_a v_aᵀ‖²) = −3.55e−15`, i.e. zero to float
precision, and tight — the bound is attained when `S` holds every other pivot.

> **Monotone, nonnegative, submodular, `h(∅)=0` ⇒ greedy attains `1 − 1/e =
> 0.632121` of the optimum** (Nemhauser–Wolsey–Fisher 1978; classical, cited from
> standard knowledge, **not fetched** — flagged as such).

**And this repo's signed arm voids it.** `M_ab ≥ 0` needs `A ≥ 0`. On signed
operators `M` had a negative entry in **297 of 300** draws at `n \in {4,5,6}`, and
in **300 of 300** at fixed `n = 6` (`scale/pivot_selection_theory.py`), so neither
`f` nor `h` is submodular there and the guarantee does not apply. The softmax-family cells
(`softmax`, `glance`, `twin`) keep it; the signed operator does not.

**A prediction of mine that was wrong, recorded.** I expected `h` to be
non-monotone, on the reasoning that `B_P` is a fixed sum rather than a projection
so adding a pivot could increase the residual. It cannot, for `A ≥ 0`: `0` of
21 936 checked `(S,a)` pairs decreased `h`. The derivation above is the corrected
version and it is why the `1−1/e` survives.

**What this is worth, honestly.** It is a bound on how well *greedy* approximates
the *best* pivot set under a reconstruction objective. It says nothing about
whether a better pivot set lowers NRMSE — that is §3's question, and it is
unanswered. **A provable guarantee on an objective nobody has shown to matter is
still a guarantee about nothing.**

---

## 10. DETERMINANTAL POINT PROCESSES

Dispatched to a moon whose report **did not return before this survey was
written**. Nothing about DPPs is claimed here. What can be said without it, from
the structure above: a `k`-DPP needs an eigendecomposition of an `s×s` kernel per
example — `O(s³) = 262 144` scalar ops at `s=64`, per example, per step — and the
natural kernel would be the pivot Gram `m3_quintuple.py` already forms for the
`settled` cell. That is a lead, not a verdict, and it is **gated behind §3 like
every other stage-A candidate.**

---

## 11. RANKING, AND WHAT I EXPECT TO FAIL

Ranked by expected value **per unit of cost to find out**, which given §2 is the
only ranking that matters here.

| # | item | why | gate |
|---|---|---|---|
| **1** | **Run `scale/recall_probe.py`** | The stage-A question's correct experiment, already written, never executed. Costs one run, not a research programme. Everything in §7, §9 and §10 is downstream of its answer. | none — run it |
| **2** | **`matrix_norm(game).max()/median` on a real batch** | One line. Decides whether the `nash.py` shared-`tau` fix can possibly rescue the ordinal thread, before anyone spends a rerun on it. §6 says a `1.27×` surrogate attenuation cannot explain a `2.6–5.8` NRMSE. | none — one line |
| **3** | **`ACE_i` vs the existing `B` columns** | Decides SCM graph-surgery's novelty for the cost of a correlation, reusing an oracle already cross-checked to `1e-10`. Genuinely vector-valued, zero new parameters. | none — one correlation |
| **4** | Restate `FINDINGS.md` C2/C3 | The citation is overstated three ways and the repo's `L/d = 4.00` sits outside the theorem's regime. Costs an edit. | owner of `FINDINGS.md` |
| **5** | Integrated Gradients as a **diagnostic** | Cheap, parameter-free, and answers "is the trained arm reading the graph at all" — attribution should land on the bridge node `e4_harmonic.py`'s decoder already identifies. | none |
| **6** | SIMPLE or Sander for stage A | Cheapest credible differentiable top-k; forward stays an exact `[n,k]` draw so nothing downstream changes shape. | **§3 first** |
| **7** | Greedy pivot selection under `h` | Has the `1−1/e` of §9 — but only on a nonnegative operator, and only if stage A matters. | **§3 first** |
| **8** | Persistent-homology causal effect | Most genuinely function-valued label surveyed; one recent single-source paper (`arXiv:2603.02289`, verified to resolve and title-match: Kim & Lee, 2026-03-02). | §3-independent, but expensive |

**Expected to fail, stated plainly and with reasons rather than hedges:**

- **Every stage-B relaxation** — Gumbel-softmax, IMLE, SIMPLE-on-`α`, perturbed
  optimisers on `α`. Not because the mathematics is weak but because §2 shows the
  effect they compete for is `14×` to `51×` below this instrument's detection
  floor and would need `559` to `41 266` seeds. **Unfalsifiable here.**
- **NeuralSort and SoftSort** — `O(s²)` to resolve a full permutation whose only
  consumer is a gathered `[n,k]` set. Dominated by construction.
- **Influence functions** — wrong axis (training example → prediction, not entity →
  entity), scalar or parameter-space output, and `O(p³)` at `p = 4769`. Recommend
  against.
- **Cellular sheaves as published** — the one framework native to vector-valued
  node data, and its restriction maps `σ(V[x_v‖x_u])` need a new `V`. Fails the
  parameter gate outright; the parameter-free repairs collapse to the Laplacian
  `kirchhoff.py` already builds.
- **LapSum with learned `α`** — literally `+1 nn.Parameter`. Fixed-`α` only.
- **Submodularity on the captured-mass objective** — supermodular. The intuition
  that "pivot selection is obviously a diminishing-returns problem" is **wrong on
  the obvious objective**, and right only on the reconstruction form.
- **Hyperbolic cones on the shipped E4' case** — `δ̂` is already measured and the
  two-lobe motif is not shown to be near-tree. The untried step is to *use* `δ̂`
  as a gate, not to run the cone experiment blind.

**The survey's one-line answer to "what maths are we missing".** For stage B,
none — the question is closed by measurement, not by mathematics. For stage A, the
missing thing is **not a method but a measurement**, and it has been sitting
written and unrun in `scale/recall_probe.py` since round one.
