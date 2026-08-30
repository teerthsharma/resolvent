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

**(b) The readout must not be flat — but only where the target is not flat, and that
was measured rather than assumed.** A root-mean-square error over a configuration
weights every coordinate independently and identically, which is the assumption a shape
violates: the coordinates of a fixed point are **coupled**, and a prediction right in
aggregate but wrong in arrangement is not a good prediction of a shape. Candidate
readouts that respect the coupling are Hilbert projective distance on the positive part,
Fisher–Rao or total variation where the object is a distribution, and a Procrustes or
optimal-transport cost where only arrangement matters.

**A QUALIFICATION THAT SURVIVED A CHECK, AND IT NARROWS THE RULE CORRECTLY.** The
objection **does not bite on the chain family**: `equilibrium_oracle` returns
`z*_{s-1}`, which is exactly `N(0, t*)` on the real line — **the target's own geometry
is flat, so a root-mean-square error is the right readout there** and every reading
already taken under it stands.

**It bites on the absorbing-chain corpus**, whose label is a probability. The
lightweight non-flat readout is one transform and no new scoring code:

```
    phi(p) = 2 * arcsin( sqrt(p) )
```

the exact Fisher–Rao geodesic coordinate on the binary simplex, since
`ds = dp / sqrt(p(1-p))`.

**RULE 8 was applied to it before it was measured.** `phi` is smooth and strictly
monotone, so near the fixed point it acts as a diagonal scaling and **cannot move the
leading eigenvalue** — both readouts must therefore share an asymptotic rate. Confirmed
to `6.319e-07` and `4.610e-04`.

**But the price at the rungs is not zero**, which is what makes the choice
load-bearing:

| `t` | Euclidean | Fisher–Rao | ratio |
|---|---|---|---|
| 8 (`_64`) | 0.1815582640 | 0.2576242379 | **1.418962** |
| 32 (`_1024`) | 0.0481326124 | 0.0582150553 | **1.209472** |

The cause is measured: `_1024`'s conditional label puts **`0.7666` of its mass within
one hundredth of a boundary** (`0.3333` below `0.01`, `0.4333` above `0.99`), which is
exactly where the two metrics diverge.

**So the readout is load-bearing where the round reads and inert where it does not, and
it must be DECLARED BEFORE the corpus run.** A 42 % swing at a rung, chosen after the
curve is visible, would be choosing the answer. The assertion in `demo()` requires
**both** rate agreement below `1e-3` **and** a rung ratio above `1.15`, so a readout
that changed nothing would fail the test that says it needs declaring.

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

### 0.35 THE USE CASE, AND IT IS WHY JOINT DETERMINATION IS NOT A METAPHOR

The target application is **embodied systems — robots and anything else that must
understand causality, consequence, pattern and link** rather than emit plausible
continuations. That is not a marketing frame; it is the setting in which the thesis of
§0.3 stops being a preference and becomes a requirement.

**A language model predicting the next token is well served by per-row independence.**
Each position's continuation can be scored against the context on its own, and softmax's
independent-per-query mixture is exactly the right shape for that — which is why it is
Bayes-optimal on the single-location problem and why it has been so hard to beat here.

**A robot predicting what happens next is not.** Joint angles, contact forces and body
positions are **coupled by constraints**, so they are determined *together* or not at
all: moving one joint changes the reachable set of every other, a contact that makes or
breaks reconfigures the whole kinematic chain, and a force applied at one point
propagates through the linkage. **There is no ordering in which these can be read off
independently, because the constraint is simultaneous.** That is a fixed point in the
literal sense, and it is the thing a per-row mixture cannot represent in one read.

**The intervention test of §0.2(c) is the robot's own question, stated exactly.** *"If I
move this, what else moves?"* is a displacement field over a configuration under a
`do()` operation — which is the consequence-fidelity metric of §6 generalised from a
sign on one scalar to a vector over the whole shape. **An embodied system needs the
vector answer; the scalar answer is not a weaker version of it but a different
question.**

**And the provenance is not a coincidence.** This project's own recovered graph corpus
comes from a physics engine, where the merged upstream contribution computes connected
components over **constraint/tree incidence** — which is precisely the coupling
structure above, expressed as a graph. Bodies joined by constraints form islands that
must be solved together; bodies in different islands do not interact. **The corpus and
the use case are the same object seen twice**, which is why an absorbing-chain label on
that substrate is the natural test rather than an arbitrary one.

**A SECOND SETTING, AND IT SHARES THE STRUCTURE RATHER THAN MERELY THE SPIRIT.**
Retrieval-augmented generation, where a model must understand **the consequence of a
false report**. A retrieved passage is not a token to be attended to on its own merits;
it is an **intervention on the context**, and the question that matters is what the rest
of the answer does when that passage is wrong. That is a `do()` operation followed by a
displacement over the whole output — **the same vector-valued consequence measure of
§0.2(c), applied to evidence rather than to joints.**

**And the failure mode there is the one a per-row mixture is worst at.** A retrieved
falsehood does not corrupt one position; it propagates, because downstream claims are
**conditioned on it** and must be mutually consistent with it. Scoring each output
position independently against the context cannot represent that propagation, which is
why a system built on independent-per-row attention can produce an answer that is locally
plausible at every position and globally wrong. **Detecting that requires reading the
shape, not the tokens.**

**What this does NOT license.** No claim is made here about robotic performance, about
retrieval accuracy, about
control, or about any embodied or retrieval benchmark — nothing in this repository has been run on
one. **This section states why the target shape is what it is, and nothing more.**

## 0.4 THE ARCHITECTURAL POSITION THAT FOLLOWS

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

---

## 11. THE SECOND EXACT ORACLE — KIRCHHOFF AGAINST THE ABSORBING CHAIN

§7's label is one computation of the harmonic measure `ω_v = P_v(τ_a < τ_b)`. The
matrix-tree theorem gives a second, and the two agreeing is now an instrument law
rather than an unstated hope. `scale/kirchhoff.py`, `RUN`.

**The chain.** `L = D − A` on the merged component, `τ(G)` its spanning-tree count,
`F(x | y)` the number of spanning 2-forests separating `x` from `y`.

1. **Kirchhoff.** `det(L_g) = τ(G)` for `L_g` the Laplacian with row and column `g`
   deleted. Every cofactor is the same number.
2. **All-minors.** `det(L_{xy}) = F(x | y)`.
3. **Resistance.** Ground at `x`, so `M := (L_x)^{-1}` has `M_xx = M_xy = 0` and, by
   Cramer, `M_yy = det(L_{xy}) / det(L_x)`. Since
   `R(x,y) = (e_x − e_y)ᵀ L⁺ (e_x − e_y)` and `M` differs from `L⁺` only along the
   all-ones direction, which cancels in the difference,
   **`R(x, y) = F(x | y) / τ(G)`** — effective resistance is a ratio of forest counts.
4. **Harmonic measure.** Unit current `a → b` gives `v = L⁺(e_a − e_b)`, and
   polarisation on `(e_x − e_b) − (e_a − e_b) = e_x − e_a` gives
   `v(x) − v(b) = ( R(x,b) + R(a,b) − R(x,a) ) / 2`. Normalising by
   `v(a) − v(b) = R(a,b)` yields the harmonic function with boundary values 1 at `a`
   and 0 at `b`, which is `ω`. **Now ground at `b`**: `R(x,b) = M_xx`,
   `R(a,b) = M_aa`, `R(x,a) = M_xx + M_aa − 2M_xa`, and the three-term expression
   collapses to

```
    ω_x  =  M_xa / M_aa  =  F(x, a | b) / F(a | b)
```

**with no cancellation of large resistances, and one solve — `L_b z = e_a` — rather
than an inverse.**

**It is independent of §7's solve, and that is structural rather than asserted.** The
chain path inverts `I − Q` with `Q = D^{-1}A` restricted to the nodes that are neither
endpoint. The Kirchhoff path solves a symmetric unnormalised `D − A` on the whole
component, in which `a` is an ordinary interior column and only `b` is removed. The
two differ in matrix, normalisation, right-hand side and index map. **`L_T` and
`I − Q` are diagonally related — `I − Q = D^{-1} L_T` — which is exactly why the
Kirchhoff route does not form `L_T`.**

**Both ends of the derivation are checked against brute force, not assumed.** Steps 1
and 2 are compared to enumeration of spanning trees and separating 2-forests over
every edge subset on drawn 7-node graphs; step 4's forest ratio is compared to the
grounded solve node by node. `tests/jupiter/test_kirchhoff_agreement.py`, 11 tests.

**The law, and its measured margin.** `max |ω_kirchhoff − ω_chain| < 1e-10`, enforced
in `e4_harmonic.measure()` before any number is read off an instance. Measured gaps,
float64, this machine:

```
    six drawn Erdős–Rényi instances, n = 9              2.220446e-16
    SMALL_CASE,     62 merged nodes,   60 transient     8.992806e-15
    SHIPPED_CASE, 1202 merged nodes, 1200 transient     9.636736e-14
```

**The tolerance sits three orders above the largest clean gap and nine below the
smallest defect.** A scratch copy of the chain carrying one off-by-one — dividing by
`len(neighbours) + 1`, the bug you get by counting a node among its own neighbours —
moves `ω` by between `1.749951e-01` and `3.156170e-01` over the same six drawn
instances, and the check rejects it on **6 of 6**. The defect is survivable by
construction: every row stays strictly sub-stochastic, so the solve succeeds and
returns plausible numbers in `[0,1]`, and nothing downstream would have noticed.

**Scope.** The forest identity is the identity for the **undamped** walk. At
`kill > 0` the operator is not a random walk on `G` and the law raises rather than
comparing two different objects and reporting the difference as a defect.

---

## 12. THE LADDER VERDICT AS A NAMED STATISTIC — PAGE'S L AND ISOTONIC REGRESSION

`scale/e_ladder.py:217` decides row A against row B by
`mono = all(b >= a for a, b in zip(deltas, deltas[1:]))`. **That is a monotonicity
check on four numbers, and it has no null, so it has no error rate**: four
independent noisy numbers land in non-decreasing order by chance one time in
twenty-four. `scale/page_trend.py` supplies the null. `RUN`.

**Page's L.** Rank the `k` conditions within each of the `n` blocks, sum ranks per
condition into `R_j`, take `L = Σ_j j · R_j`. Under exchangeability within a block
every one of the `k!` rank assignments is equally likely and blocks are independent,
so **the null distribution of `L` is the `n`-fold convolution of the `k!`-point
distribution of `Σ_j j·r_j`** — exact, no table, no normal approximation. At `k = 4`,
`n = 5` that enumerates `24⁵ = 7 962 624` assignments; the support is `[100, 150]`,
the mean `125`, and the distribution is symmetric, which is what makes the
one-sidedness a choice rather than an artefact.

**The pre-registration, exhaustive before the data.**

```
    RISES := page_p < 0.05  AND  isotonic top rung > 0 with CI excluding zero
    FLAT  := neither clause fires
    SPLIT := exactly one clause fires
```

`SPLIT` exists because "both / neither" leaves a hole, and this repo has already paid
for one: row H was added to `E_LADDER_PREREGISTERED_READING.md` because rows A–G all
conditioned on settled winning somewhere. **Neither clause can pick a rung**: `L`
quantifies over every rung by construction and the size clause reads the last rung,
fixed by the ladder's definition.

**THE READING, on the complete CPU ladder — five seeds, four rungs.**

```
    rank sums R_j       8.0   12.0   13.0   17.0
    L = Σ_j j·R_j       139        support [100,150], null mean 125
    exact p             0.016724
    permutation p       0.016255   (se 0.000283, 200 000 draws, seed 0)
    isotonic fit        −0.036025  −0.017215  −0.004284  +0.016035
    top rung            +0.016035  CI [+0.002826, +0.033167]
```

**The pre-registered branch returns `RISES`.** Two independent p-value paths — an
exact convolution and a permutation draw — agree to `4.7e-4`, inside `1.7` standard
errors.

**Two things that verdict rests on, both measured, neither softened.**

**The size clause is carried by the constraint, not by the data.** The same bootstrap
without the monotone constraint gives the top rung at `+0.016035` with CI
`[−0.004711, +0.033167]`, which **covers zero** — and whose lower bound reproduces
`results/e_ladder_reading.txt`'s shipped `ci_lo` of `−0.004711` **exactly**, so the
two bootstraps are the same bootstrap. PAVA pooled the top rung in `13.07%` of
resamples and lifted the lower bound by `+0.007537`. **Pooling only ever raises a low
top, never lowers a high one**, so that lift is a property of the estimator under a
flat truth.

**The trend clause survives 2 of 5 single-seed deletions.** Dropping seed 1 leaves
`p = 0.003864` and seed 2 leaves `p = 0.021741`; dropping seed 0 or 4 leaves
`p = 0.050411` and seed 3 leaves `p = 0.072401`.

**But the conjunction is calibrated even though one of its clauses is not, and that
is the argument for requiring both.** Over 200 drawn tables per arm (`5×4`,
`sd = 0.03`, seed 11, `n_boot = 300`):

```
    flat truth      trend 0.065   size 0.325   RISES 0.040
    rising truth    trend 0.810   size 1.000   RISES 0.810
```

**The isotonic clause alone fires on a third of flat tables — six times nominal. The
conjunction reads `0.040` against a nominal `0.05`, because the trend clause gates
the miscalibrated one.** Both arms are drawn from the same generator and the verdict
separates them, so the branch is not vacuous.

**THE GRANULARITY QUESTION, AND THE TWO CLAUSES ANSWER IT DIFFERENTLY.** At five
seeds a sign-pattern statistic can express no two-sided p finer than
`2/2^5 = 0.0625`, which is **above** the `0.05` this project quotes; measured on
the shipped `contrast()` over 1000 samples, a 5-0 unanimity excludes zero
`385/385` times, a 4-1 split 20-44% and a 3-2 split 0-3.7%, so "the CI excludes
zero" at `N = 5` is very nearly "all five seeds agreed". That floor is real, and
it lands on **one** of the two clauses.

**The trend clause clears it.** Page's `L` ranks `k` conditions within each block
instead of reading one sign, so the outcome space is `(k!)^n = 24^5 = 7 962 624`
rather than `2^n = 32` — a factor of `12^5 = 248 832`, or `log2 24 = 4.585` bits
per block against 1, `22.925` bits against `5`. The consequences are exact:

```
    achievable p-values on the whole support        51
    finest non-zero achievable p                    1 / 24^5 = 1.2558674e-07
    achievable p-values at or below alpha = 0.05    14   (L = 137 .. 150)
    critical value at alpha = 0.05                  L = 137
    TRUE size of the test at that critical value    0.037002877
    next coarser rung, L = 136                      0.052384114  (above alpha)
```

**`alpha = 0.05` is reachable, and `p = 0.016724386` is an exact atom sum of the
discrete null** — the cumulative count `133 170 / 7 962 624` at `L = 139` — not
an interpolation onto it. Being discrete the test is conservative: its true size
is `0.037002877`, not `0.05`. By contrast the sign lattice at `N = 5` has six
achievable one-sided p-values of which exactly **one** clears `0.05` (unanimity,
`1/32 = 0.03125`), and **none** of their two-sided partners does.

Three routes to the null, sharing no arithmetic: float probabilities through
`np.convolve`, integer polynomials through Python big integers, and literal
enumeration of every one of the `(k!)^n` rank assignments with `L` rebuilt from
whole tables. The first two agree to `1.388e-17` at `k=4, n=5`; all three agree
**exactly** at `(3,3)`, `(4,3)` and `(3,4)`, where enumeration is tractable.

**The size clause does not clear it, and cannot.** It IS a paired percentile
bootstrap over five seeds, so `0.0625` is its floor and no interval it prints is
a `0.05`-level statement at this seed count. The ladder's top rung is a **4-1
split** — the regime Venus measured at 20-44% — and its unconstrained interval
duly covers zero at `[-0.004711, +0.033167]`. Only the one-sided PAVA lift
carries it across. **The granularity floor and the estimator bias are two
independent reasons to distrust the same clause, and they push the same way.**
The pre-registered branch is not refitted after the fact: `RISES` stands as
written, with the trend clause sound at its own level and the size clause now
known to be incapable of the level it was written at.

**AND ROW G OUTRANKS ALL OF IT.** `E_LADDER_PREREGISTERED_READING.md` credits a rung
nothing in either direction when either cell sits at or above predict-the-mean, and
it fires on **three of the four rungs this statistic is computed on**:

```
    rung      settled       twin    credited
    e3_t1    0.994399   0.958373    yes
    e3_t2    1.013958   0.996743    no
    e3_t8    1.096009   1.091725    no
    e3_t32   1.103711   1.119745    no
```

**A trend in the difference between two arms that both lose to predict-the-mean is an
ordering, not a capability.** `RISES` is a statement about the contrast as a number.
**It is not a claim that settling bought anything at depth**, and `page_trend.report()`
prints these cell means beside the verdict so that it cannot be read as one. Replacing
an eyeball with a statistic sharpens the reading of a quantity the ladder's own
pre-registration had already declined to credit.

---

## 13. THE ATTRIBUTION CLAIM'S SAMPLE-COMPLEXITY LINE

`impact_attribution` regresses the query row `B[query, :]` of length `N` against
`n_samples` news vectors — the noiseless system `y = A b`
(`scale/impact.py:1013-1021`, `READ`). Each row of the planted `B` carries
`SUPPLIERS_PER_NODE + COMPETITORS_PER_NODE = 4` nonzeros
(`scale/impact.py:88-89, 405-410`, `READ`). **The fixed sparsity is in the ROW, not
the column** — a column's nonzero count is however many rows chose it and is
unbounded — so the line below is stated for the row the recovery actually reads.

**The line.** Recovering an `s`-sparse vector in `R^n` from `m` noiseless linear
measurements is possible at all only in the RIP regime

```
    m  ≥  C · s · ln(n / s)
```

**Below it an arm's attribution failure carries no information about the arm**: there
is provably no procedure separating the planted `B` from an adversarial one
consistent with the same undersampled measurements. `scale/rip_line.py:verdict`
returns the exact string `UNDER-SAMPLED` there and `ADMISSIBLE` at or above, so a
cell prints a regime rather than a loss.

**`C` is measured, and the oracle transitions across the line.** Gaussian `A`, signed
`s`-sparse `b` with uniform support and Rademacher signs, drawn not built; recovery
by L1 basis pursuit (`scipy.optimize.linprog`) and, independently, by orthogonal
matching pursuit. **Every grid reads `0.00` at its bottom and at least `0.96` at its
top for basis pursuit** — the sweep is non-degenerate at both ends, which is the
whole requirement.

```
    n     s    n/s    ln(n/s)     m_50 (BP)   C (BP)     m_50 (OMP)  C (OMP)
     64    4    16    2.772589       16       1.44270       24       2.16404
    128    8    16    2.772589       32       1.44270       48       2.16404
    256    4    64    4.158883       24       1.44270       28       1.68314
     64   16     4    1.386294       40       1.80337       64       2.88539
```

**`C` IS NOT BED-INVARIANT, AND THE FIRST TWO BEDS COULD NOT HAVE SHOWN THAT IT WAS.**
They share `n/s = 16`, so `ln(n/s)` is the same number in both and `m_50` tracking
`s` **forces** `C` to repeat whatever the truth is. Varying the ratio instead:
`C = 1.44270` holds at `n/s = 16` and `n/s = 64`, and rises to `1.80337` at
`n/s = 4` — 25% higher. `s·ln(n/s)` is the asymptotic scaling and its constant creeps
as `n/s` falls toward 1. **The operational consequence is that `C` must be re-swept
per bed; reusing `1.44270` elsewhere is a guess.** The two solvers likewise disagree
by a factor of 1.2 to 1.6 depending on the bed, which is two solvers failing
differently and a second reason `C` is a per-bed measurement.

**Where the shipped corpus sits.** `impact_attribution` defaults to `n_samples = 256`
against `N = 1024`, `s = 4` (`scale/impact.py:973`, `READ`), and
`rip_line(1024, 4, 1.44270) = 32.01`, so the default oversamples the floor by roughly
`8×`. **The floor is the deliverable, not a claim that today's default sits on it.**

---

## 14. THE COHERENCE FLOOR AND THE CALIBRATED SCRAMBLE CONTROL

Stated as functions of `(d, k)` so the arithmetic survives any change to either.

**The Welch bound.** Any `k` unit vectors in `R^d` obey

```
    max_{i<j} |⟨u_i, u_j⟩|  ≥  sqrt( (k − d) / (d (k − 1)) )
```

which is **exactly 0 for `k ≤ d`**, because an orthonormal set of `k ≤ d` vectors
exists and attains it. Above the dimension it is the real constraint and it is tight:
at `d = 2`, `k = 3` the bound reads `0.5` and three unit vectors at 120° attain
`|cos 120°| = 0.5` exactly, agreeing to `1e-12`.

**M1, at `d = 256`, `k = 16`.** `k ≤ d`, so **the floor is exactly 0**, while random
role vectors carry a max coherence of `0.174795`. **A crosstalk-shaped failure at
`k = 16` roles in 256 dimensions is therefore a training or design defect, not
dimension starvation** — the geometry permits perfect separation and nothing in the
dimension count obstructs it.

**M5, and the author's figure is 15.8% low.** The amendment states the random
coherence as `sqrt(2 ln k / d) = 0.147176`. That closed form is a union bound over
`k` events, but the maximum runs over `C(k,2) = 120` **pairs**, not over 16 vectors.
Four routes, of which two fail differently from the sampling:

```
    sqrt(2 ln k / d)              0.147176   author's form, union over k
    sqrt(2 ln(k(k−1)/2) / d)      0.193397   union over C(k,2) pairs — an upper bound
    Monte Carlo, 20 000 trials    0.174795   95% CI [0.174460, 0.175131]
    order-statistic quadrature    0.174499   ∫₀¹ (1 − F(x)¹²⁰) dx, F the exact
                                             |⟨u,v⟩| CDF from (1−t²)^((d−3)/2)
```

**The quadrature lands inside the Monte Carlo interval and both closed forms land
outside it.** The union bound is above, as a bound must be; the author's form is
below, because it counts the wrong number of events. The mean overlap, by contrast,
is confirmed on both paths: the exact `Γ(d/2) / (√π Γ((d+1)/2)) = 0.049917` matches
the Monte Carlo `0.049917` (CI `[0.049869, 0.049964]`) and its
`sqrt(2/(πd)) = 0.049868` limit to `4.9e-5`.

**The consequence, and the error is in the safe direction.** Scrambled roles at
`d = 256`, `k = 16` carry an expected mean overlap of `0.049917` and an expected
worst-case overlap of `0.174795`, **both by chance and both computable before the
control runs**. A scramble control calibrated to expect zero residual separation is
vacuous before it runs, and it is **more** vacuous than the amendment's own figure
implies, not less. That is the fifteenth pattern, killed pre-birth.


---

## 15. A COVERAGE TOOL THAT SCANNED NOTHING

`scale/chase_struck_coverage.py` walks every `.md` and `.py` the shipped
struck-constant test does not cover, and reports any struck constant asserted
without a strike marker in its paragraph. It printed
`SCANNING 0 PATHS THE SHIPPED CHECK DOES NOT COVER`, `uncovered .md: 0,
uncovered .py: 0`, and **exited 0**. `RUN`.

**The root cause is a scope error in one expression.** The exclusion list
`(".git", "__pycache__", ".pytest_cache", ".claude", ".benchmarks")` was tested
against `p.parts` — the components of the **absolute** path. Every agent
worktree in this project is checked out under `<repo>/.claude/worktrees/<name>/`,
so `.claude` was a component of the absolute path of **every file in the tree**
and the filter dropped all of them. Measured on this checkout: `366` candidate
files, `0` surviving the absolute filter, `365` surviving the same filter applied
to `p.relative_to(ROOT).parts`.

**Its own must-fire kept passing throughout**, because that control feeds text
directly to `scan_text` and never exercises target selection. **A matcher control
is not a coverage control**, and this is the cleanest available example of the
difference: the assertion that fired was true and the instrument was blind.

**The fix is the scope correction plus a refusal.** `collect_targets` filters on
the path relative to the scan root, and `main` returns 1 rather than 0 on an
empty target list — a tool that walks zero paths passes every input and must not
report success. `tests/jupiter/test_struck_coverage_scans.py` plants a struck
constant in a file on disk that target selection must reach, requires it to be
found, requires the same text carrying a strike marker **not** to be found, and
requires a file with no struck constant to be silent — three arms, because two
would pass for a scanner that flagged everything. **6 of its 8 tests fail against
the unfixed scanner; the 8th states the root cause without reference to the
refactor so a later rewrite cannot make it vacuous.**

**What the fix exposed.** With `346` paths now scanned the tool exits 1 on `27`
candidates. They split into two classes, and the file's own docstring already
warns that layer 2 is a text scan and text scans cry wolf. Prose **about** a
strike whose paragraph carries no marker: the scanner's own docstring quoting
`1.471448` in its must-fire description, `MISTAKES.md:206` discussing
`5.4944e-13`, `PREREGISTRATION_HOLE_AUDIT.md:424` running a
`git log -S "0.743864"` forensic. Live assertions: `−1.389` with `R² 0.9938` in
a `RESEARCH.md` table row, and the three newly struck U1/N3 constants
`0.743864`, `0.656532`, `0.816955` asserted in
`tests/cameron/test_harmonic_attribution.py:123-124` and mirrored in
`tests/deimos/`. **`−1.389` alone accounts for 17 of the 27** and is a short
enough string to match inside longer numbers, so the tally is a candidate list
and not a finding. Adjudicating it is not this section's business; surfacing it
at all required the scanner to look at a file first.

---

## 16. LIMITS ADDED BY SECTIONS 11–15

§10 predates these sections and does not cover them.

The instrument law of §11 is enforced only where `e4_harmonic.measure()` is the entry
point; `absorbing_chain` and `fixed_point` remain callable directly and are not
guarded, and the law is defined only for the undamped walk. Its agreement margin is
measured at three graph sizes on one machine in float64 and no conditioning bound is
proved. §12's verdict is `RISES` under a branch that was fixed before the statistic
ran, but its size clause fires on the strength of a one-sided estimator bias AND
cannot express a p finer than `2/2^5 = 0.0625` at five seeds, so it is structurally
incapable of the `0.05` level it was written at; its
trend clause survives only 2 of 5 single-seed deletions, and the percentile bootstrap
behind both has `5⁵ = 3125` distinct atoms at five seeds; the calibration sweep is
200 tables per arm under Gaussian noise at one scale and does not establish the error
rate under the ladder's real noise, and the trend clause's own TRUE size is
`0.037002877` rather than `0.05`, so power quoted against a nominal `0.05`
overstates it. §13's `C` is fit at four beds with coarse `m`
grids — `m_50` for the `n/s = 4` bed is bounded only to `(32, 40]` — under a Gaussian
measurement ensemble, and `news_mat` in `scale/impact.py` has not been shown to be
one; the line is a necessary condition, never a sufficient one, and nothing here
wires `UNDER-SAMPLED` into an IMPACT cell. §15's fix is verified on this checkout and on
synthetic roots; it is not verified on a checkout whose absolute path contains
none of the skipped names, where the old and new filters agree and the test
falls through to an equality branch. The 27 candidates the fix exposed are
unadjudicated, and `−1.389` is short enough to match inside longer numerals, so
the count is an upper bound on real hits rather than a finding. §14 is arithmetic about independent
uniform unit vectors and says nothing about the vectors any trained arm actually
holds; the order-statistic quadrature treats the `C(k,2)` pairwise products as
independent, which they are not, and is corroboration for the sampled figure rather
than a proof of it.

---

## 17. WHAT THE STE RESULT ACTUALLY SHOWS, AND WHAT IT PRICES OUT

The full candidate catalogue is `results/r9_maths_survey.md`. This section records
only the derivations it rests on. **No wall-clock measurement was taken for it.**

### 17.1 Two discrete stages, and only one of them is trained

| | discrete object | site | trained |
|---|---|---|---|
| **Stage A** | which `k` rows are pivots | `m3_quintuple.py:169` → `pivot_probe.py:80`, `topk(key.norm(dim=-1), k)` | **never, in any arm** |
| **Stage B** | mixture weights `α` over the chosen `k` | `m3_quintuple.py:440` | yes |

`argmaxste` estimates **stage B**. Its forward is bitwise `argmax`'s:
`soft − soft.detach()` is elementwise exactly `+0.0`, and `soft ∈ [0,1]` admits no
inf/nan path — checked over 200 drawn gates in float64. **Stage A is byte-identical
across `argmax`, `softmax`, `argmaxste`, `settled` and `twin`.** Nothing measured
this round bears on stage A.

### 17.2 The gradient is the effect; the mechanism is not

Recomputed from `results/m3_quintuple_v2.jsonl` at `ntr8192_nev512`, 5 seeds, via
`m3_synthetic_settled.contrast`. Positive = second cell lower.

```
    argmax    -> argmaxste   +0.225760  CI [+0.212433, +0.245886]  5/5
    softmax   -> argmaxste   +0.107304  CI [+0.082879, +0.140870]  5/5
    argmaxste -> twin        +0.004092  CI [-0.023107, +0.029187]  3/5
    settled   -> twin        +0.002959  CI [-0.031557, +0.048587]  2/5
    argmaxste -> settled     +0.001133  CI [-0.069616, +0.057312]  3/5
```

**Three structurally different stage-B mechanisms — a Neumann-settled fixed point,
a straight-through one-hot, and a full softmax mixture — lie within `0.004092` of
each other with every CI covering zero.** The only contrast excluding zero at 5/5
is the presence of a gradient.

### 17.3 The pricing rule: what this instrument can and cannot falsify

Minimum detectable effect, paired, `α=0.05` two-sided, power `0.80`. Two paths: the
normal approximation, and an exact noncentral-`t` solve. (`scipy.stats.nct` returns
`nan` at large noncentrality; an unguarded bisection converges upward and returns a
non-monotone answer — the guard is load-bearing.)

```
    paired sd   n=5 normal   n=5 exact   n=10      n=20
    0.022345    0.027996     0.037584    0.022256  0.014758
    0.034451    0.043164     0.057946    0.034313  0.022753
    0.050146    0.062828     0.084345    0.049945  0.033119
    0.082152    0.102929     0.138179    0.081824  0.054257
```

Seeds required, and their cost at `settledrow`'s `4.560600 s/step`
(`scale/m3_flops.py:107`, wall clock, one step including backward and Adam,
`s=64 n=2048`) × 150 steps = `684.09 s`:

```
    contrast                seeds (normal / exact t)   runs    lower-bound clock
    argmax vs argmaxste          0.1 /      3             6         1.1 h
    argmaxste vs twin          556.3 /    559         1 118         8.85 days
    settled vs twin           2254.1 /   2257         4 514        35.74 days
    argmaxste vs settled     41263.5 /  41266        82 532       653.46 days
```

**The seed counts are exact. The day counts are PROVISIONAL**: `m3_flops.py:117-120`
records the same `settled` unit reading `2.0775` against `3.4372 s/step` in two
sessions on identical code, `1.65×` apart, which is why every clock in this
repository carries that label. They are also lower bounds — the compared cells ran
at `ntr8192`, four times the batch the rate was measured at.

> **PRICING RULE.** At 5 seeds this instrument resolves `0.057946` or larger. The
> stage-B mechanism differences are `14×`, `20×` and `51×` below that floor; the
> gradient effect is `3.9×` above it. **Any candidate whose contribution is a
> better relaxation of stage B is unfalsifiable here**, and an unfalsifiable
> improvement is inadmissible under this project's own evidence rule.

### 17.4 Why the one stage-A ablation on record could not have said anything

`DONE_ARCHIVE_ROUND1.md:4707` records K4: `randpivot_signed`, `k=8` content-blind
pivots, slope `+0.081`, CP intervals overlapping the content-selected arm, and the
pre-registered consequence that content selection is *"not load-bearing for M2"*.

That null was **forced**. `scale/recall_probe.py:3-7` states the identity: for any
content-blind schedule of size `k`,

```
    P(c reachable) × (share | reachable)  =  (k/s) × (1/k)  =  1/s
```

bit-for-bit the dense rate, **independent of `k`**. M2 draws `c` *from* `P`, so it
measures the second factor and conditions the first away. Verified on two paths
that fail differently — the symbolic factorisation, and a 200 000-draw Monte Carlo
over 12 `(s,k)` pairs (at `s=1024, k=8`: symbolic `0.00097656`, sampled
`0.00099187`). A content-blind schedule reproduces the dense rate exactly, so K4
was structurally incapable of reading anything else.

`recall_probe.py:9-11` names the quantity that is not an artefact: whether
`P(c selected)` stays `Θ(1)` as `s` grows under **content-conditional** selection.
**That file is imported by zero Python files, has no `results/` artifact, and
`DONE_ARCHIVE_ROUND1.md:5833` calls it "already sitting unrun".** Stage A's search
space is `C(62,8) = 3 381 098 545` sets, of which `topk` explores exactly one.

### 17.5 A correction to §C of `FINDINGS.md`

`arXiv:2410.01537` (Marion, Berthier, Biau, Boyer, ICLR 2025) is cited there as
*"softmax is provably Bayes-optimal"* against *"linear attention"*. Equations
fetched from `ar5iv.labs.arxiv.org/html/2410.01537`: the predictor is
`T_λ^{k,v}(𝕏) = erf(λ𝕏k)ᵀ𝕏v` — **`erf`, not softmax**; Corollary 2 gives
**asymptotic** optimality under *"`d→∞` and `L=o(d)`"*; and Proposition 3 refutes
**linear regression**, `ℛ(β⋆) → ε²+γ²`, not linear attention. The label shape does
match (`Y = X_{J₀}ᵀv⋆ + ξ`, scalar, latent informative position), so the worry is
sound — but this repo runs `L = s = 64` against `D_MODEL = 16`
(`m3_capability.py:79`; the `d24` in journal keys is `make_batch`'s flipper offset,
`negation_scope.py:84-86`). `L/d = 4.00`, the opposite of `L = o(d)`. **"Provably"
is not earned at this geometry.** Single-source: the ICLR proceedings PDF returned
compressed streams and the second path failed.

### 17.6 The `nash.py` shared-`tau` defect is real and too small to be the cause

`ceq/nash.py:64` returns `margin * float(matrix_norm(m, ord=2).max()) / 4.0` — one
Python float for the batch, from the worst-conditioned instance, applied at `:146`.
Since `stance = 2σ(z/τ) − 1 ≈ z/(2τ)` for small `z/τ`, an instance run at `τ_batch`
is attenuated by `τ_i/τ_batch`. On games built exactly as `nash.py:138-144` builds
them, Gaussian `q,k`: median shrink `0.877526` at `n=64`, `0.851143` at `256`,
**`0.789701` at `2048`** (min `0.690213`). Two paths — the small-signal ratio and
the exact sigmoid amplitude (`0.50 → 0.547752`, `0.25 → 0.281624`,
`0.10 → 0.113593`).

**A `1.27×` attenuation cannot produce the recorded OOD NRMSE of `2.6151` to
`5.8198`**, which is 2.6× to 5.8× worse than predicting the mean. Before the fix
and the rerun, one line settles it: measure `matrix_norm(game, ord=2).max()/median`
on a real batch. Near `1.27` and the bug cannot be the cause. The Gaussian draw is
a surrogate, so this is a lower bound on the attenuation, not the corpus's value.
