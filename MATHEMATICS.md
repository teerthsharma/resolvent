# MATHEMATICS.md — the theory of record

Every claim carries an evidence class. `RUN` means executed this session and the output
read. `READ` means a file and line was opened and quoted. `CITED` means an external
source was resolved and its title matched. `DERIVED` means it follows from `RUN`/`READ`
by steps that are written out. Anything that would be `GUESS` is either labelled or
absent.

Numbers carry provenance: which journal, which seed count, and what they were compared
against.

---

## 0. THE THESIS — WHAT THE MODULE IS SUPPOSED TO PREDICT

Self-attention is called a next-token predictor, and that description is accurate: it
guesses the next best symbol. **This module is meant to guess the next best SHAPE** — the
configuration the sequence settles into — so that the next equilibrium can be reached
rather than the next word emitted.

Concretely, and in the author's framing: it does not predict *a* word. Given the words
already present, it predicts **several at once**, jointly, because their values are
determined together. And the way it decides which existing tokens matter is causal:
**inject an existing token, and observe how the whole shape moves** — read through the
embedding geometry and the topology of the sequence, not through a similarity score.

### 0.1 THIS PROJECT HAS NOT BEEN TESTING THAT. `READ`, AND IT IS THE CENTRAL DEFECT.

`scale/negation_scope.py:274`, `equilibrium_oracle`:

> *"`z*_{s-1}`, the last coordinate of `(I - A)^{-1} b`."*

**The label is one coordinate. `z*` is the shape; `z*_{s-1}` is a single point on it.**
And `nrmse` at `scale/negation_scope.py:672` is a root-mean-square error on that scalar.

So every task in this repository — the static ones **and** the equilibrium ones —
**asks for a point prediction.**

**That is why softmax wins, and the reason is a theorem rather than an accident.**
Predicting one scalar from a sequence, where the answer is a payload at a latent
location times a query-side transform, **is exactly the single-location regression
problem on which one softmax layer is provably Bayes-optimal** (`arXiv:2410.01537`).
**The corpus was built as the one task softmax cannot lose.**

### 0.2 WHAT THE THESIS REQUIRES INSTEAD

Three changes, each of which is a build rather than an argument:

**(a) The label becomes the shape.** Predict `z*` — the whole fixed point, a vector of
length `s` — not `z*_{s-1}`. The oracle already computes the entire vector and then
discards all but one entry.

**(b) The readout must not be flat.** RULE 9 applies with force here. A root-mean-square
error over a configuration weights every coordinate independently and identically, which
is precisely the assumption a shape violates: the coordinates of a fixed point are
**coupled**, and a prediction that is right in aggregate but wrong in arrangement is not
a good prediction of a shape. The candidate readouts are the ones that respect that
coupling — Hilbert projective distance on the positive part, Fisher–Rao or total
variation where the object is a distribution, and a Procrustes-style or optimal-transport
cost where only the arrangement matters and scale does not.

**(c) The consequence test becomes vector-valued.** The existing fidelity metric asks
whether `sign(Δŷ)` matches `sign(Δy)` for a scalar. Under the thesis it should ask **how
the whole shape moves** under an intervention — a displacement field, compared by
direction and magnitude across all coordinates, not a sign on one.

### 0.3 WHERE SOFTMAX CANNOT FOLLOW — the novelty claim, stated narrowly

**One softmax step computes a convex mixture per query row, and each row is computed
independently of the others.** There is no mechanism in that step by which the value
assigned to position `i` constrains the value assigned to position `j`. A jointly
determined configuration — where the coordinates must be mutually consistent because
they solve a fixed point together — **is not what a single independent-per-row mixture
computes.**

That is the sharpest available statement of what this module could do that softmax
cannot, and it is **narrow on purpose**:

* It is **not** a claim that softmax cannot predict any single coordinate — the theorem
  says it can, optimally.
* It is **not** a claim about depth — depth composition is separately governed by
  `L = floor(log2 k) + 2` (§2), and stacked softmax layers get that too.
* It **is** a claim about **joint determination in one read**, which is the only place
  the fixed point is doing work a mixture is not.

**And it is currently UNTESTED, because no vector-valued label exists in the
repository.** Recorded as the open claim rather than as a result.

---

## 0. THE ARCHITECTURAL POSITION, AS IT NOW STANDS

**Softmax attention is not the opponent. It is the base case, and the module must
contain it.**

That is a reversal, and the evidence forced it. Three independent results say the same
thing:

* **Softmax is Bayes-optimal on the single-location task shape.** For a target that is a
  payload at one latent location times a query-side transform, one softmax attention
  layer provably attains Bayes risk while linear attention provably falls short.
  `CITED`: Marion, Berthier, Biau, Boyer, ICLR 2025, [arXiv:2410.01537](https://arxiv.org/abs/2410.01537);
  Duranthon et al., [arXiv:2509.21936](https://arxiv.org/abs/2509.21936).
* **Softmax's single step is already a converged fixed-point step.** The attention update
  is the update rule of a modern Hopfield network, which *"converges with one update"*
  with exponentially small retrieval error. `CITED`: Ramsauer et al., ICLR 2021,
  [arXiv:2008.02217](https://arxiv.org/abs/2008.02217).
* **Iterating the normalisation to a fixed point is published and buys little.**
  Sinkformer runs Sinkhorn iterations to a doubly-stochastic fixed point in 3–5 steps
  for small accuracy gains. `CITED`: Sander, Ablin, Blondel, Peyré, AISTATS 2022,
  [arXiv:2110.11773](https://arxiv.org/abs/2110.11773).

**The consequence for design is precise.** If softmax lies *inside* the module's
function class, the module is bounded below by softmax and any equilibrium machinery is
a strict addition. If softmax lies *outside* it — which is what was actually built, see
§4 — then the comparison measures the exclusion rather than the idea.

**Where iteration is known to pay, the iterated object is the representation, not the
mixture weights.** Weight-tied looped transformers with input injection match standard
transformers on algorithmic tasks at under 10 % of the parameters. `CITED`: Yang, Lee,
Nowak, Papailiopoulos, ICLR 2024, [arXiv:2311.12424](https://arxiv.org/abs/2311.12424).

---

## 1. THE EQUILIBRIUM TASK FAMILY

### 1.1 Why the previous corpus could not test anything

Both originally registered oracles are closed-form functions of their input.
`READ`, `scale/negation_scope.py`:

```
negation_scope    x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]      a product of two entries
counter_squared   x[:, :, CH_FLIP].sum(dim=1) ** 2            a sum, squared
```

Neither has a fixed point. An arm that iterates to convergence and then predicts a
product of two numbers is doing the same work as an arm that does not iterate. **The
deciding contrast on that corpus was therefore correct and uninformative.**

### 1.2 The chain family, and the closed form that makes it admissible

The chain label is the `t*`-step iterate of an affine recursion `z' = a·z + b` with `a`
strictly lower triangular, hence nilpotent, so `t*` is **exact rather than a
tolerance**.

**The truncation law.** A reading that uses only `k` hops has normalised error

```
    ceiling(t*, k) = sqrt( max(0, t* − k) / t* )
```

`RUN`, this session:

| `t*` | `k=0` | `k=1` | `k=2` |
|---|---|---|---|
| 2 | 1.000000 | 0.707107 | 0.000000 |
| 8 | 1.000000 | 0.935414 | 0.866025 |
| 32 | 1.000000 | 0.984251 | 0.968246 |

**`k = 0` is exactly 1.000000, which is the mean-predictor bar.** That is what makes the
family admissible: a zero-hop reading is worth nothing, so the label cannot be obtained
without iterating.

**It had to be earned.** `READ`: an earlier encoding gave the query token a driver, so
`1/(t*+1)` of the label was legible at zero hops and the harness's own 0-step gate
aborted `INSTRUMENT BROKEN` on three of five rungs. Setting `b[s-1] = 0` restores the
property. **A third static task was avoided by a gate firing, not by care.**

The formula was checked against drawn batches rather than assumed: max deviation
`+0.012088` at `n = 2048`.

### 1.3 The consequence family

`E2` labels a coordinate of the **new** fixed point after a one-token `do()` shock,
under a contraction with Lipschitz constant `0.800000` exactly on every draw, enforced
at generation with a raise rather than a warning.

Its truncation decays geometrically, bounded by `L^k × (k=0 error)`:
`1.000067 / 0.410387 / 0.203305 / 0.070732 / 0.013666 / 0.001017`.

**Its first design was killed by its own control.** The raw new-fixed-point coordinate
let a shock-blind reading score `0.194150` — **96 % of that label was the un-intervened
game.** The mirror-intervention contrast that replaced it has shock-blind error
identically `sqrt(1 + mean²/var) ≥ 1.0`; measured `1.000067`.

---

## 2. THE DEPTH LAW — WHY THE SHORTFALL IS NOT ABOUT SETTLING

`hop_k` composition requires depth logarithmic in `k`:

```
    L = floor(log2 k) + 2          width O(1)
```

`CITED`: Sanford, Hsu, Telgarsky, ICML 2024,
[arXiv:2402.09268](https://arxiv.org/abs/2402.09268), Theorem 4.2; Corollary 4.3 gives
an `Ω(log k)` lower bound **conditional on the 1-vs-2-cycle conjecture** — conditional,
and it must be cited as such.

`RUN`: `hop_2 → 3`, `hop_4 → 4`, `hop_8 → 5`, `hop_16 → 6`.

**The arms in this project are depth 1.** At `t* = 8` the required depth is about 5.

**And the measured behaviour matches.** At `t* = 8` an arm trains to `0.860972` —
**sitting on its own two-hop ceiling of `0.866025`** — and evaluates at `1.112208`. It
is not failing to settle; **it is memorising, because the term it needs is a three-way
product `a[s-1]·a[s-2]·b[s-3]` that depth-one mixing cannot form.**

---

## 3. THE CONTRACTION AND ITS CERTIFICATE — WHAT SURVIVED

Three contraction certificates were attempted and all three died: the sphere metric on
an identity (`θ = arcsin(√TV)` exactly on a one-token mask), Birkhoff on `T` (the
theorem requires a positive **linear** map and the map is not linear), and Birkhoff on
`G` (failed its own per-seed bind).

**What holds is `κ = β` — chosen, not derived — and it holds in 142/142 journal rows.**
`READ`: `results/hilbert.jsonl`, field `bound_respected`.

**Two numerical facts that were load-bearing and are easy to get wrong:**

* `tanh(Δ/4)` reads **exactly 1.0** in float64 for `Δ ≥ 76.246190`. The representable
  form is `1 − κ = 2/(e^{Δ/2} + 1)`, which stays finite to `Δ ≈ 1400`.
* `d_H(softmax u, softmax v) = osc(u − v)` **exactly** — the log-partition cancels in an
  oscillation, so the Hilbert distance between two softmax outputs needs no `exp`.

---

## 4. THE CONFOUND — AND IT SUPERSEDES THE ARM COMPARISON

**The pivot family structurally cannot place the deciding token on the value path.**

`READ`, `scale/arm_s.py:107`: `select_pivots(kk, min(k_piv, s - 2), exclude=(s - 1,))`,
then `return piv[piv > 0]`. `READ`, `scale/m3_quintuple.py:129`: the same set via
`exclude=(0, s - 1)`.

**So the largest legal pivot is `s − 2`.** Every operator row is `tril(-1)`-strict, so
row `s − 2` reads only `j ≤ s − 3`. **Therefore `v[s-2]` is invisible to every pivot
row**, while softmax's own row `s − 1` reads `j ≤ s − 2` and sees it directly.

**And the `t* = 1` label is `a[s-1] · b[s-2]`** — concentrated on exactly the hidden
position. Measured, perturbing that token by `+100`: the softmax reading moves
`101.6983` (direct value pass-through) while the pivot value path moves `3.263746`.

**Consequence, stated plainly: any arm-versus-softmax number on this corpus measures the
exclusion at least as much as it measures the idea.** The root cause is a documented
rationale rather than a defect — *"a pivot reading of the row being settled is not an
independent reading of it"* — a defensible independence property that became a
capability ceiling.

**The repair is the same as the architectural position in §0:** lift the exclusion so
that `p = s − 1` is legal, at which point `av[s-1]` **is** the softmax row and softmax
becomes an interior point of the family.

---

## 5. THE SETTLED ARM ITERATES THE WRONG UNKNOWN

The settled cell iterates `α` on a simplex over `{α @ av}` — **a reparameterisation
inside the unsettled twin's own family.** It can reallocate weight; it cannot add a
function the twin could not already express.

**That is a structural account of the measured result**, rather than a post-hoc one:

```
settled − twin = −0.002959     exact 95% CI [−0.042903, +0.031557]     covers zero
```

`RUN`, this session: exact enumeration over all `5**5 = 3125` paired resamples, 126
distinct values — **cheaper than `B = 10000` Monte-Carlo by 3.2× and with no sampling
error at all.**

And the variance tells the same story: `sd 0.064106` for settled against `0.016547` for
the twin, **3.874×**, on the same batches, the same initialisation and the same
parameter count. **Settling widens the distribution without moving the mean.**

**The one-hot control separates the two claims.** `argmax − softmax = −0.118456`, CI
`[−0.134115, −0.102786]`, 0/5 seeds — **collapsing the mixture to a lookup is worse than
plain softmax.** So the mixture is worth something and the fixed point over it is not.
Gradient descent on softmax attention converges to the max-margin separator of optimal
tokens (`CITED`: Tarzanagh et al., [arXiv:2306.13596](https://arxiv.org/abs/2306.13596)),
which is why hard selection kills the training signal that soft selection keeps.

---

## 6. CONSEQUENCE FIDELITY — THE CAPABILITY METRIC OF RECORD

For an arm at trained weights, over drawn `do()` interventions with closed-form oracle
effects:

```
    fidelity  =  fraction of interventions where  sign(Δŷ_model) = sign(Δy_oracle)
```

reported with a **Clopper–Pearson exact** interval, alongside the slope of `Δŷ` on `Δy`,
and compared against softmax by **exact McNemar on the same interventions with
byte-identical draws.**

**It is computable for softmax too, which is what makes it a table column rather than a
plea.** Measured at `s=64 d=24 n_train=8192`, **seed 0 only — the file requests five
seeds and contains one**:

| arm | eval NRMSE | fidelity | CP interval | slope |
|---|---|---|---|---|
| `softmax` | 0.877168 | 0.807843 | [0.754044, 0.854329] | 0.319032 |
| `pivot_signed` | 0.673762 | 0.854902 | [0.805590, 0.895735] | 0.565631 |
| `pivot_unsigned` | 0.747528 | 0.835294 | [0.783958, 0.878642] | 0.469549 |
| `windowed_signed` | 1.075830 | 0.360784 | [0.301809, 0.423038] | 0.021058 |

Paired exact McNemar against softmax: `pivot_signed` `b=28 c=40 p=0.18181` **no
reject**; `pivot_unsigned` `p=0.348889` **no reject**; `windowed_signed`
`p=5.45054e-23` **REJECT, in the wrong direction** — its fidelity is *below* chance and
it scores `flipper 0.000000 (0/72)`, never once recovering the flipper token's sign.

**Two readings, and both matter.** Softmax **already has consequence fidelity at
`0.807843`**, well clear of chance — causal awareness is not the thing that
distinguishes the arms. And the signed arm's apparent advantage **does not survive the
paired test**.

---

## 7. THE ABSORBING-CHAIN ORACLE AND ITS RATE LAW

The label is a coordinate of the absorption probabilities of a Markov chain:
`N = (I − Q)^{-1}`, `B = N R`, the fixed point of `z ← Q z + R`.

**`λ₂` is defined, not assumed.** `P = [[Q, R], [0, I]]` is block-triangular, so the
literal second eigenvalue of `P` is `1` and says nothing. **`λ₂ := ρ(Q)`.**

**The engineered dial is exact.** Setting `Q := α · P_TT` gives `ρ(αP_TT) = α·ρ(P_TT)`,
so `α = target / ρ(P_TT)` lands `λ₂ = 0.9250000000` on both graphs, `t_rel = 13.3333`.

**The committor residual has a closed form.** With `u − u_t = Q^t u` and
`v − v_t = Q^t v`:

```
    q_t − q  =  ( u ⊙ (Q^t s)  −  s ⊙ (Q^t u) )  /  ( s ⊙ s_t )
```

verified to `1.082467e-15` and `4.583348e-13`. Both tails carry the same Perron mode, so
the leading term is `λ₂^t · w ⊙ (c(s)u − c(u)s)` over a converging denominator, and
therefore **`f := decay/λ₂ → 1` exactly**: measured `1.00000063` and `1.00158689` at
`t = 320`.

**But only the limit is derived.** A placement sweep on a fixed graph gives
`f(160) = 0.99034549` — **below one** — which no account of late sub-dominant modes
permits. The leading amplitude `c(s)u − c(u)s` is a property of the two boundary
columns, so **the excess is set by both spectrum and geometry, and geometry can flip its
sign.**

**And the operational consequence is the important part.** Every ladder rung
`t* ∈ {1,2,8,32}` sits **below** the mode time (`20.5615`, `169.3116`). **The ladder is
entirely pre-asymptotic, so `λ₂^{t*}` is the wrong predictor at the rungs even though
`λ₂` is the right asymptotic rate.** The closed form above is the right predictor and
costs one solve.

**A Cheeger obstruction bounds what any one-edge bridge can do.** `g ≤ 2φ` with
`φ = 1/vol(S)` gives `t_rel ≥ vol(S)/2 = 1372.50` against a target band of `[10, 20]`;
landing in band would need `vol(S) ≤ 40`, making every label a 3-hop function. **The
α-scaling construction relocates the obstruction rather than evading it** — a radius-3
ball retains `0.7951152826` of the walk's weight and `0.1090161839` survives past the
graph's diameter, so the reach ceiling moves from 3 hops to about 13. **A ceiling at 13
is still a ceiling.**

---

## 8. THE FORMAL RESULTS

`lake build CEQ` exits 0. **39 theorems across six modules.**

**`oracle_ne_resolvent`.** The arm's operator is strictly lower triangular, hence
`A^n = 0`. The oracle's transient block `Q` is non-negative with symmetric support and
at least one positive entry, hence **never nilpotent**. So the two cannot coincide, and
the ladder's creditability does not rest on an assumption.

**`truncation_never_exact`.** The partial sum `Σ_{k<N} Q^k` is not the inverse of
`(I − Q)` at **any** `N` — whereas for the arm's own operator it becomes an equality at
`N = n`. **Every rung therefore leaves a real residual.**

`SymmSupport` is deliberately weaker than symmetry, because `D^{-1}W` is not symmetric
at unequal degrees.

Both are bound to the shipped object rather than to a paper statement: reachability from
the root module, `#print axioms` on five names **with a planted `sorryAx` seen to fire**,
the three hypotheses checked entrywise on the actual matrix, and a must-fire in which the
arm's own `.tril(-1)` operator is **rejected** by the support check.

---

## 9. THE NOVELTY POSITION, BOUNDED

Most of `equilibrium-labels-as-attention-capability-bar` is occupied: CLRS
([arXiv:2205.15659](https://arxiv.org/abs/2205.15659)) and CLRS-Text
([arXiv:2406.04229](https://arxiv.org/abs/2406.04229)) for equilibrium labels as a bar;
Sanford et al. NeurIPS 2024 (OpenReview `AfzbDw6DSp`) for an attention capability
hierarchy; deep-equilibrium algorithmic reasoners
([arXiv:2402.06445](https://arxiv.org/abs/2402.06445),
[arXiv:2410.15059](https://arxiv.org/abs/2410.15059)) for equilibrium-solving arms.

**The nearest contender was checked directly and does not occupy what remains.**
[arXiv:2607.21607](https://arxiv.org/abs/2607.21607), *"Spectral Flow Certificates for
Depth-Aware Long-Range Propagation in Graph Neural Networks"*, is real — verified via
the arXiv API, title matched, formula `SFC(G,k) = 1 − (1 − γ(G))^k` and all three
`R² = 0.910 / 0.881 / 0.863` byte-exact as attributed. **Its date/identifier mismatch is
block-wide across five neighbouring identifiers and does not impeach it.**

**But:** it is a fixed-graph GCN with `Transformer`, `GAT`, `GIN` and `GraphSAGE` each
appearing **zero** times; its headline fits are on **engineered** graphs while its
surveyed graphs fall to `R² = 0.440`; and its label is **trained accuracy** on a
sign-of-sum readout, with `equilibrium` and `fixed point` each appearing **zero** times.

**And its own Proposition 1 (iii) states that at fixed depth `SFC` and `γ` induce
identical rankings** — so the headline values are `λ₂`'s numbers relabelled, and the
paper's real gain is pooled-depth `R² 0.884` against `0.858`.

**What remains unoccupied, stated narrowly:** attention rather than message-passing, and
**the equilibrium itself as the label** rather than an accuracy. Three further absences
are recorded as `NOT FOUND` rather than assumed: no prior work iterating a routed
mixture-weight vector over content-selected, Gram-coupled pivots to a fixed point as the
attention read; none on the signed row-L1 normalisation `ρ·w/‖w‖₁` as attention; and **no
unconditional lower bound** against one-layer softmax on this product chain.

---

## 10. LIMITS

Collected here rather than scattered.

Every task in the original corpus is static, so no reading taken on it bears on
settling. The arm comparison on the equilibrium corpus is confounded by the pivot
exclusion of §4 and must not be reported as evidence about settling until that exclusion
is lifted. The consequence-fidelity table of §6 is **one seed of a requested five**, and
its companion at `d = 256` covers **two of four arms**. The ladder is pre-asymptotic at
every rung, so any curve read against `λ₂^{t*}` will look wrong for arithmetic reasons.
Five seeds cannot decide anything anytime-validly — the attainable evidence ceiling at
five seeds is `3.80169140625` against a threshold of `40`. No weights have been
published. The depth lower bound of §2 is conditional on the 1-vs-2-cycle conjecture.
And `results/m3_capability.txt:1212` is a **stale trap**: it records
`e3_t8 k=2 NRMSE=0.815162` from before `b[s-1] = 0` and describes a corpus that no
longer exists.
