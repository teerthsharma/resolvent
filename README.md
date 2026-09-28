<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" />
  <img alt="PyTorch 2.x CPU" src="https://img.shields.io/badge/torch-2.x_CPU-ee4c2c?style=flat-square" />
  <img alt="tests: 119 passing" src="https://img.shields.io/badge/tests-119_passing-success?style=flat-square" />
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" />
</p>

<h1 align="center">rjepa</h1>

<p align="center">
  <b>Where a JEPA planner's candidate ranking loses to the Bayes rule, and which operators can close it.</b><br/>
  <span>Candidate-set resolvents, a D-JEPA-spec relational operator and two exact-truth beds with a Monte-Carlo
  Bayes ceiling. The gap between "pick the candidate predicted nearest the goal" and the Bayes-optimal pick is
  carried by prediction error that every candidate of a start shares; a set operator closes most of it, a
  pointwise head does not, and D-JEPA's bounded correction caps what any operator can reach.</span><br/>
  <i>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></i><br/>
  <sub><a href="mailto:teerths57@gmail.com">teerths57@gmail.com</a> · first written in <a href="https://github.com/teerthsharma/resolvent">github.com/teerthsharma/resolvent</a> (<code>sun/rjepa/</code>)</sub>
</p>

<p align="center">
  <a href="#abstract">Abstract</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#2-theoretical-foundation">Theory</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#4-results">Results</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#5-round-2">Round 2</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#6-what-did-not-hold">What did not hold</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#7-quick-start">Quick start</a>
</p>

<p align="center"><sub>Every number in the results table carries the command that reproduces it and was checked by
an independent re-run before it was allowed here. Observations that were measured but not bound by a pre-registered
failing test are named in <a href="#6-what-did-not-hold">Section 6</a> without numbers.</sub></p>

---

## Abstract

A latent world model plans by scoring K candidate action sequences from one start and executing the one whose
predicted future lies nearest the goal embedding. D-JEPA (Liu et al., arXiv 2609.24749) names the failure of that
rule the decision-local gap: among the few futures competing for execution, the one predicted closest can realise
a worse outcome than an available alternative. This package isolates one mechanism behind such a gap. When the
error that separates prediction from outcome is **shared** by every candidate of a start (it comes from the start
state, which all candidates share), it translates the whole predicted set, and the Bayes-optimal pick becomes a
function of the set's geometry rather than of each candidate's own distance. On an exact-truth bed with a
Monte-Carlo Bayes ceiling, the gap between the Bayes pick and latent-distance planning grows from 0.000 to 0.241
top-1 hit as the shared error grows from 0.1 to 30 candidate spreads (K = 4, 3 seeds × 4,000 starts). A learned
set head with a resolvent message closes 0.947 of the gap at σ = 1; a pointwise head with the same inputs closes
0.028. D-JEPA's bounded correction (|δ| ≤ ε = 0.2) caps any operator obeying it at 0.289 hit against the Bayes
ceiling's 0.381 at σ = 10. The candidate-axis resolvent `(I − A)⁻¹ v` ships with 73 contract tests (permutation
equivariance, pole refusal, padding, dtype, gradients); its linear permutation-equivariant special case is proved
rank-inert, and its learned multi-hop tail adds −0.0002 hit over a single hop.

**Keywords:** JEPA world models, candidate ranking, Bayes decision rule, shared error, common random numbers,
resolvent, permutation equivariance, D-JEPA.

---

## 1. Background

### Why a set operator?

**The plug-in rule ranks candidates one at a time.** Latent-distance planning scores candidate k by
`|ẑ_k − z_g|` alone. If the prediction error were independent per candidate this is the right rule; the Bayes
pick and the distance pick then coincide (test `B0`: agreement ≥ 0.97 at zero shared error).

**Error from the start state is not per-candidate.** Every candidate of a start is rolled out from the same
(misestimated) start. That error moves all K predicted futures together. Ranking under a common unknown
translation is a question about the whole set, the same structure that common random numbers exploit in
simulation-based planning (Yadav et al., arXiv 2605.04732; Klein et al., arXiv 2409.02086), where paired
differences under shared randomness have variance `var X + var Y − 2 cov(X, Y)`.

**The Bayes rule under a large shared translation is a convex-hull property.** As the shared error grows past the
candidate spread, the chance that candidate k is best tends to the exterior angle of its Voronoi cell. A candidate
inside the hull of the set has exterior angle 0, and it is exactly the candidate nearest-to-goal favours when the
planner aims every candidate at the goal. Distance then drops below chance.

**A set operator can express this; a pointwise one cannot.** Any head that scores each candidate from its own
prediction cannot represent a rule that depends on the other candidates. A head that exchanges messages across
the set can, provided the message passing is permutation-equivariant (the set has no order) and bounded.

### Prior art

| System | What it does with the K candidates | Set interaction | Bound on the correction | Where it is stronger than this package |
|---|---|---|---|---|
| D-JEPA (Liu et al. 2026, arXiv 2609.24749; code Apache-2.0) | re-ranks shared candidates with a bounded relational operator over native cost ranks | 2-layer Transformer, no candidate-order encoding | \|δ_i\| ≤ ε = 0.2 (Prop 2, Cor 1) | evaluated on PushT, Reacher, RoboTwin, PiPER and driving; PushT n = 256: 87.11 relational vs 76.95 TD-JEPA native (paper, Table 5) |
| ACPC (An et al. 2026, arXiv 2608.12939; code MIT) | diagnoses JEPA world models: divergence of a clean and a perturbed history rolled forward under the same actions | none (a diagnostic, not a ranker) | n/a | measures robustness of the representation itself, which this package assumes |
| Common random numbers (Yadav et al. 2026, arXiv 2605.04732; Klein et al. 2024, arXiv 2409.02086) | evaluates candidate policies on shared simulator randomness to cut the variance of their differences | the pairing itself | n/a | proven variance reduction in simulators; this package has no simulator-side result |
| resolvent (Sharma 2026, github.com/teerthsharma/resolvent) | causal resolvent read `O = (1 − g) P (I − gP)⁻¹ V` over a sequence, attributed there to ChaCAL (Fagnou et al., EMNLP 2024) [U] | causal (nilpotent off-diagonal) | g < 1 | the sequence operator, with its cost measured against fused attention; this package moves it to an unordered set |
| **rjepa** (this package) | Bayes ceiling, distance floor, pointwise / one-hop / resolvent set heads and a D-JEPA-spec operator on two exact-truth beds | resolvent `(I − A)⁻¹`, one hop, or Transformer | ρ < 1 refused otherwise; ε for the D-JEPA head | toy beds only (K = 4 and K = 8, D = 2) |

[U] marks a source not fetched for this package. No fetched source states that error shared across candidates
cancels in the argmin ranking of a *learned* world model, or measures the shared fraction; the common-random-number
sources establish the variance identity for simulators, not for learned-model prediction error (source-repo
ground-truth file `sun/rjepa/wilson/facts_r2.json`, item 4.8). D-JEPA itself does not decompose its error into
shared and per-candidate parts (same file, item 2.10).

---

## 2. Theoretical foundation

### 2.1 The decision-local gap as a shared-error decision rule

Bed `shift` (`rjepa/shared_error.py`). The planner observes a start estimate $y$; the true start is

$$
s = y - \sigma \xi, \qquad \xi \sim \mathcal N(0, I_D), \qquad\text{so}\quad s \mid y \sim \mathcal N(y, \sigma^2 I_D)\ \text{exactly.} \tag{1}
$$

Candidates aim at the goal $g = 0$ from the estimate, $a_k = -y + \rho\, r_k$ with $r_k \sim \mathcal N(0, I_D)$,
and execute to

$$
z_k = s + a_k + \sigma_e \eta_k = \rho\, r_k - \sigma \xi + \sigma_e \eta_k, \qquad k^\star = \arg\min_k \lVert z_k \rVert. \tag{2}
$$

The term $\sigma\xi$ is the same for every $k$. The plug-in (latent-distance) rule and the Bayes rule are

$$
\hat k_{\text{dist}} = \arg\min_k \lVert \hat z_k \rVert, \qquad
\hat k_{\text{Bayes}} = \arg\max_k\ \Pr\!\left(k = k^\star \mid y, a_{1:K}\right), \tag{3}
$$

with the probability estimated by $M = 1024$ posterior draws of $(\xi, \eta)$ common to all $K$ candidates
(`bayes_pick`). As $\sigma \to \infty$, $\lVert \rho r_k - \sigma\xi \rVert^2 = \sigma^2 - 2\sigma\rho\, \xi^\top r_k + O(1)$,
so $k^\star \to \arg\max_k \xi^\top r_k$ and, for isotropic $\xi$,

$$
\Pr(k = k^\star) \;\to\; \frac{1}{2\pi}\left|\{u \in S^1 : k = \arg\max_j u^\top r_j\}\right|, \tag{4}
$$

the exterior angle of candidate $k$'s Voronoi cell (`hull_angle_share`), zero for a candidate inside the convex
hull of the set. The *closure* of a learned head is $(\text{hit}_{\text{head}} - \text{hit}_{\text{dist}}) /
(\text{hit}_{\text{Bayes}} - \text{hit}_{\text{dist}})$.

Splitting the error variance into a shared share $1 - m$ and a per-candidate share $m$ (`make_bed(mix=m)`) is the
control: at $m = 1$ the gap vanishes (test `B10`).

### 2.2 D-JEPA's bounded correction and its reach

D-JEPA's operator (`rjepa/djepa.py`, reimplemented from arXiv 2609.24749 §3 and App. B.1) scores

$$
h_i = \mathrm{TF}(\mathrm{enc}(v_i)), \qquad
\delta_i = \varepsilon \tanh\!\big(W_{\text{up}} \tanh(W_{\text{down}} h_i)\big), \qquad
s_i = b_i + \delta_i, \tag{5}
$$

where $b_i = (\mathrm{rank}(c_i) - 1)/(K - 1)$ is the normalised base rank and lower $s_i$ is better. Since
$|\delta_i| \le \varepsilon$, the selected $\hat i = \arg\min_i s_i$ satisfies

$$
b_{\hat i} \;\le\; \min_i b_i + 2\varepsilon \tag{6}
$$

(the paper's Prop 2 / Cor 1). At $K = 4$ the base ranks are $\{0, \tfrac13, \tfrac23, 1\}$ and $2\varepsilon = 0.4$,
so only distance ranks 0 and 1 are reachable. The Bayes rule restricted to those ranks
(`bayes_pick(max_rank=1)`) is therefore a ceiling for **every** operator obeying (6), learned or not.

### 2.3 The resolvent on a candidate set

A causal (sequence) resolvent is nilpotent off the diagonal and cannot reach a pole. On an unordered set the
coupling must be diagonal-free and symmetric in the listing order, and it is no longer nilpotent, so the bound
becomes load-bearing (`rjepa/resolvent.py`):

$$
A = \rho\, \frac{\operatorname{offdiag}(q k^\top)}{\operatorname{rowL1} + \epsilon}, \quad
\lVert A \rVert_\infty \le \rho < 1, \quad
\text{out} = (I - A)^{-1} v, \quad
\lVert (I - A)^{-1} \rVert_\infty \le \frac{1}{1 - \rho}. \tag{7}
$$

$\rho \ge 1$ raises `ValueError`. The stochastic form uses $P = \operatorname{softmax}(q k^\top / \sqrt d)$, which is
row-stochastic with $\rho(gP) = g$:

$$
O = (1 - g)\, P\, (I - gP)^{-1} V, \qquad g < 1, \tag{8}
$$

a convex combination of the rows of $V$. The truncated Neumann series $\sum_{h=0}^{H} A^h v$ has error at most
$\rho^{H+1}/(1-\rho)\,\max|v|$.

### 2.4 A linear equivariant resolvent is rank-inert

The only linear permutation-equivariant coupling on $n$ candidates is $W = \alpha I + \beta J$ ($J$ all-ones).
Whenever $\rho(\gamma W) < 1$,

$$
(I - \gamma W)^{-1} s = \frac{1}{1 - \gamma\alpha}\left(s + \frac{\gamma\beta\, \mathbf 1^\top s}{1 - \gamma\alpha - \gamma\beta n}\, \mathbf 1\right), \tag{9}
$$

a positive rescaling ($|\gamma\alpha| < 1$ is an eigenvalue bound) plus a common shift, so the ranks of $s$ are
unchanged. A resolvent can only change a ranking through a **data-dependent** coupling $W(z)$, which is what the
set heads learn.

### 2.5 LIN1: a one-VJP chance ranker

On bed `torus` (`rjepa/standard_map.py`) success is landing in a ball of radius $R$ around $G$ after $T$ steps of
the standard map. Linearising the rollout at the estimate, $z \approx \hat z + \delta J \xi$; projecting on the
unit goal direction $n$ turns the ball into a half-plane and

$$
\Pr(\text{success}_k) \approx \Phi\!\left(\frac{R - d_k}{\delta\, \lVert J_k^\top n_k \rVert}\right), \tag{10}
$$

one vector-Jacobian product per candidate (`lin1_prob`). It is exact in the linear-Gaussian limit up to the
half-plane approximation of the ball; the tests check it against the exact posterior there.

---

## 3. Implementation

| Module | Contents | Source file in `teerthsharma/resolvent` |
|---|---|---|
| `rjepa/resolvent.py` | `build_A`, `set_resolvent`, `neumann_resolvent`, `stochastic_resolvent`, `resolvent_apply`, `linear_equivariant_resolvent` | `sun/rjepa/chase/rjepa_ops.py`, `sun/rjepa/foreman/rjepa.py` |
| `rjepa/djepa.py` | `RelationalOperator` (5), `rank01`, `dj_loss` | spec from `sun/rjepa/wilson/facts.json`; code from `sun/rjepa/cameron/rj.py` |
| `rjepa/heads.py` | `Head("point" \| "hop1" \| "resolvent" \| "djepa" \| "djepa4")`, `dist_rank`, `fit` | `sun/rjepa/foreman/rjepa.py` |
| `rjepa/shared_error.py` | bed `shift` (1)-(2), `bayes_pick`, `dist_pick`, `hull_angle_share` (4), `jepa_predictor`, `run_cell` | `sun/rjepa/foreman/rjepa.py` |
| `rjepa/dial.py` | bed `dial` (5.1): K = 63 shared-error dial, learned predictor, `bayes_P` + A6.1 tie-break `bayes_pick`, `DialHead`, `saturation`, `run_cell` | `sun/rjepa/r2/r2.py`, `sun/rjepa/r3/foreman/r3.py` |
| `rjepa/standard_map.py` | bed `torus`, exact tangent Jacobian, `lin_prob`, `lin1_prob` (10), rankers | `sun/rjepa/cameron/rj.py` |
| `rjepa/reproduce.py` | `python -m rjepa.reproduce gap \| bound \| closure` | new |

**Low precision is promoted, not trusted.** There is no fp16/bf16 LU, and `|q·k| > 65504` overflows fp16 to
`inf`, after which the row normalisation is `inf/inf = NaN`. Both resolvents solve in ≥ float32 and cast back.

**Dense LU, not a triangular solve.** A triangular solve applied to a non-causal `P` silently reads only the lower
triangle; the stochastic form uses `torch.linalg.solve`.

**Padding is exact.** Masked candidates are removed from `A` on both axes (or from the softmax keys), output 0, and
an all-masked row returns 0 instead of NaN. A padded set's real outputs equal the unpadded set's to 1e-12.

**The row-L1 floor keeps gradients bounded.** Without `ε = 1e-3` in the denominator, `|∂A/∂q| ~ 1/|q|` and the
gradient grows without bound as the logit scale goes to zero; the test compares scale 1 with scale 1e-8.

**The D-JEPA head is zero-initialised.** `W_up` starts at zero, so an untrained head returns the base ranking
exactly (tested), and every departure from latent distance is learned.

---

## 4. Results

All rows are round-1 results that an independent inspector re-ran and left bound to a pre-registered failing test.
Bed `shift`: K = 4, D = 2, ρ = 1, σ_e = 0.02, 3 seeds × 4,000 held-out starts, Bayes by M = 1,024 common draws.

| Quantity | Value | Condition | Reproduce |
|---|---|---|---|
| Gap, Bayes − latent distance (top-1 hit) | **0.000 / 0.024 / 0.149 / 0.213 / 0.241** | σ = 0.1 / 1 / 3 / 10 / 30, ideal plug-in ẑ = y + a | `python -m rjepa.reproduce gap` |
| Best hit reachable under D-JEPA's Cor 1 (ε = 0.2) | **0.289** vs Bayes **0.381** | σ = 10, learning-free: Bayes restricted to distance ranks {0, 1} | `python -m rjepa.reproduce bound` |
| Gap closed by the resolvent set head | **0.947** (agreement with the Bayes pick 0.905) | σ = 1, learned predictor, 100,000 training starts, 36 epochs | `python -m rjepa.reproduce closure` |
| Gap closed by the one-hop set head | **0.925** | same cell | same |
| Gap closed by the pointwise head (control) | **0.028** | same cell, same inputs, no set interaction | same |
| Neumann tail: resolvent − one-hop hit | **−0.0002** | same cell | same |
| Linear equivariant resolvent `(I − γ(αI + βJ))⁻¹` | **rank-inert** | proof (9); 300-draw property test | `pytest -q -k rank_inert` |

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  shared error / spread     0.1     1       3       10      30
  Bayes − distance          0.000   0.024   0.149   0.213   0.241
  D-JEPA Cor 1 ceiling at σ = 10:   0.289   (Bayes 0.381)
  σ = 1 closure:   resolvent 0.947   one-hop 0.925   pointwise 0.028   tail −0.0002
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Substrate.** Learning-free rows (gap, bound) are recomputed by this package's own test suite
(`tests/test_reproduce.py` pins them to three decimals) on Windows 11, Python 3.11.9, torch 2.14.0 CPU with
2 threads, and match the source repo's recorded values. The σ = 1 closure rows are the source repo's recorded run
(`sun/rjepa/foreman/results_cell_sigma1_ep36_point-hop1-resolvent.json`, 3 seeds, per-seed resolvent closure
0.990 / 0.859 / 0.992). The port's rerun of that cell (`python -m rjepa.reproduce closure`, 1,157 s on 2 CPU
threads) passes all four registered bars (agreement ≥ 0.90, closure ≥ 0.80, pointwise ≤ 0.50, |tail| < 0.01, also
checked by `pytest -m slow`) and reproduces the source's resolvent closure on seeds 1 and 2 exactly (0.859, 0.992);
seed 0 differs because its predictor initialisation depends on the global RNG state when the cell starts.

**Reading.** At σ = 0.1 the shared error is a tenth of the candidate spread and there is nothing to close. At
σ = 10 the Cor 1 ceiling sits 0.092 hit below Bayes: whatever it learns, an operator obeying (6) stays at least that
far from the Bayes rule once the shared error exceeds the spread, because the Bayes pick is sometimes a candidate
of distance rank 2 or 3, which (6) cannot reach. At σ = 1 the one-hop and
resolvent heads are within 0.0002 of each other: the set interaction does the work, the multi-hop tail does not.

---

## 5. Round 2

Round 2 moved the question off the K = 4 toy onto a learned predictor at K = 63, re-tested LIN1 on fresh seeds, and
built a verifier for rankers. Every number below is the source repository's recorded result, read from the named file
and bound by the round-2 inspector. Bed `dial` is ported as `rjepa/dial.py`: its tests pin the bed, the Bayes ceiling
and its tie-break, the D-JEPA bound and its saturation measure, and the SIGN of 5.1's `dj4L` > `dj02` at f = 1 on a small
fast cell, not the numbers below; the verifier is not ported, and none of these numbers is recomputed by `pytest` here
(Section 9). Seeds, arms and bars were registered before the runs; amendments declared
after a pilot or a run are named as such in the source `BAR.md` files and repeated below where they matter.

### 5.1 The shared-error dial at K = 63 with a learned predictor

Bed `dial` (source `sun/rjepa/r2/`, bar `BAR.md` with amendments A1-A5). Latent D = 8, action 4 per step, horizon 5,
true dynamics $z' = 1.1\,Q z + 0.5\tanh(Cz + Ba)$ with $Q$ Haar-orthogonal, K = 63 candidates per start, and a learned
MLP predictor rolled out for every candidate. The realised start of candidate $k$ is
$x_k = y - \sigma(\sqrt f\, s + \sqrt{1-f}\, u_k)$: at $f = 1$ one unknown start is shared by every candidate, at
$f = 0$ the error is per-candidate. The label is the truly best candidate; scores are normalised as
NS = (hit − 1/K) / (hit_Bayes − 1/K), with Bayes by M = 2,048 common draws under the true dynamics. Each cell is
3 seeds × 20,000 held-out evaluation starts (60,000 training starts); every learned arm sees the same 12-d token and
the same 4,000-step budget, at 67,805-69,377 parameters.

| Arm at (f = 1, σ/ρ = 3) | NS, seed 0 / 1 / 2 | What it is |
|---|---|---|
| Bayes restricted to base ranks within 2ε = 0.4 of the minimum | **0.944 / 0.944 / 0.970** | learning-free ceiling of any ε = 0.2 operator |
| `dj4L` | **0.732 / 0.703 / 0.787** | D-JEPA-spec operator, ε = 4, LIN1 base ranks |
| `hop1` | **0.501 / 0.467 / 0.497** | round-1 one-hop set head |
| `dj02` | **0.352 / 0.354 / 0.358** | D-JEPA as specified, ε = 0.2, distance base ranks |
| `dist` (floor) | **0.338 / 0.343 / 0.335** | latent-distance planning |

**Reach is not the wall.** Amendment A4 (registered after seed 0, read on seeds 1 and 2) asked whether ε = 0.2 is
too narrow to reach the Bayes pick at K = 63. It is not: the Bayes pick restricted to the ε = 0.2 reach scores
0.944-0.970 NS, while the learned ε = 0.2 operator sits 0.011-0.024 NS above the distance floor.

**The cost is saturation of the bounded correction.** On the ε = 0.2 operator the mean |δ|/ε on evaluation is
0.792 / 0.792 / 0.791: the tanh correction runs at 79 % of its bound. Widening ε with net, base ranks, loss and
learning rate unchanged (Amendment A5, claim S1, registered before its run) lifts NS to 0.725 / 0.657 / 0.729 at
ε = 0.5 and 0.727 / 0.662 / 0.737 at ε = 1, level with 0.724 / 0.661 / 0.735 at ε = 4 (mean |δ|/ε at ε = 1:
0.247 / 0.246 / 0.247). Once the correction is off saturation, the size of ε stops mattering.

**The set edge exists only where the error is shared.** The closable gap (Bayes hit − distance hit) is
0.0447 / 0.0458 / 0.0460 at f = 1, σ/ρ = 3 (mean 0.0455). At f = 0.75 it falls to 0.0037 / 0.0021 / 0.0026, and at
f = 0, σ/ρ = 3 to 0.0005 / 0.0003 / −0.0014. At f = 0, σ/ρ = 3 the 3-seed mean edge of the set arms over `dj02` is
+0.0008 hit for `hop1` and −0.0015 for `dj4L` (registered bar F1: ≤ 0.005). This is round-1 test `B10` reproduced
with a learned predictor at K = 63: remove the shared component and the set operator has nothing to exploit.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  bed dial, K = 63, learned predictor, 3 seeds x 20,000 eval starts, (f = 1, sigma/rho = 3), NS per seed
  Bayes within eps = 0.2 reach   0.944  0.944  0.970
  dj4L  (eps 4)                  0.732  0.703  0.787     dj1 0.727 0.662 0.737    dj0.5 0.725 0.657 0.729
  hop1                           0.501  0.467  0.497
  dj02  (eps 0.2)                0.352  0.354  0.358     |delta|/eps 0.792 0.792 0.791
  dist                           0.338  0.343  0.335
  closable gap (hit)    f = 1: 0.0455    f = 0.75: 0.0021-0.0037    f = 0 (sigma/rho 3): -0.0014-0.0005
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.2 LIN1 on fresh seeds, and a call-accounted race

Bed `torus` unchanged from round 1 (source `sun/rjepa/cameron/r2/`, bar `BAR.md`), fresh evaluation seeds 3, 4, 5
(seeds 0-2 never read), band T ∈ {12, 16, 20, 24, 32}, 10,000 evaluation starts per lead per seed. NS is normalised
between blind and the full-rollout Bayes reference (M = 256 posterior members per candidate).

| Quantity | Seed 3 / 4 / 5 | Compared against |
|---|---|---|
| LIN1 band-mean NS | **0.880 / 0.881 / 0.879** | best D-JEPA-spec operator 0.729 / 0.737 / 0.735; distance floor 0.721 / 0.726 / 0.728 |
| B2 race, worst NS over the 5 leads | **0.9923 / 0.9908 / 0.9957** | full rollout = 1 (bar ≥ 0.99) |
| B2 race, mean call saving per decision | **4.52×-7.89×** over all 15 seed-lead rows | full rollout, K (M + 1) T calls per decision |
| B2 race, 90th-percentile call saving | **2.33×-3.84×** over the same rows | same |

**LIN1 rebounds, scoped.** One VJP per candidate beats the D-JEPA-spec operator by ≥ 0.14 band-mean NS and the floor
by ≥ 0.15 on each fresh seed (claims A1, A2). It does not beat the three-member ensemble control everywhere: at
T = 32 the 3-seed NS is 0.7589 for LIN1 against 0.7791 for ENS3, so claim A3 is lost there, and ENS3 (3T calls per
candidate against LIN1's 2T) is the better cheap ranker at the longest lead.

**Bayes quality at 4.5-7.9× fewer predictor calls.** The B2 race (`cascade2`: empirical-Bernstein radius, a settle
threshold on the LIN1 spread, an ε-good stop, candidates raced on shared member draws) stays within 0.01 NS of the
full rollout on all 15 seed-lead rows. The saving shrinks with the lead (3-seed ratio 7.78 at T = 12, 4.53 at T = 32)
and the tail is heavier than the mean: the 90th-percentile decision saves only 2.3-3.8×. The B2 arm was added by
Amendment B2 after tuning seed 9 had shown the registered race would miss, and before any evaluation seed existed;
its configuration was chosen on seed 9 alone.

### 5.3 A verifier for rankers

Source `sun/daedalus/`, results in `results/r2/`. The round-1 verifier, re-run under round-2 code, rejected
**22/22** planted cheats with **0** errors and reproduced **5/5** known verdicts (`m0_r2c.log`,
`m0_r2c_redteam.json`, `m0_r2c_known_verdicts.json`). A new `ranker` contract
(`forward(state_feats[B, K, F]) → scores[B, K]`) caught **5/5** planted ranker cheats at their named stage (label
leak, candidate index order, false equivariance, false bound on probes, false bound on evaluation) with 0 errors
(`rankers_green_final.log`).

Through that contract, on a frozen copy of bed `shift` (Bayes hit 0.380, latent-distance floor 0.322, label-shuffle
null 0.333-0.336), the resolvent set head `fm_resolvent` scored **0.3770** hit and the one-hop head `fm_hop1`
**0.3603**, and both reached `PASS_V2` (`rjepa_arms.json`). Each of those two numbers is **one** secret-seed draw of
the candidate shuffle; replication on three draws is round-3 work (5.5). No ranker can reach `ACCEPT`: no V3 pool or
V5 ladder is registered for rankers.

### 5.4 Killed in round 2

**P2 at f = 0.75.** Registered: at f ≥ 0.75, σ/ρ ≥ 3, `dj4L` and `hop1` each beat `dj02` by ≥ 0.05 NS on 3/3
seeds. It holds at f = 1 and fails at f = 0.75, where `dj4L` scores below `dj02` on every seed; the closable gap there
is 0.002-0.004 hit, too small for any arm to separate.

**M3, the optimisation-speed reading.** Registered: if `dj02` with learning rate ×20 (`dj02f`) recovers at least half
of the ε = 4 gain, the ε = 0.2 cost is optimisation speed. `dj02f` picks exactly what `dist` picks on 3/3 seeds
(NS 0.338 / 0.343 / 0.335). Killed; the replacement route is the saturation reading of 5.1, which was then registered
and passed.

**P1, the LIN1 half.** Registered: at f = 0, σ/ρ = 1, the pointwise head and LIN1 each tie the best set arm within
0.01 NS. The pointwise half holds; LIN1 sits 0.030 NS below the best set arm (3-seed mean 0.940 against 0.970).

**B, the Hoeffding race.** The registered call-accounted race saves only 2.34-3.56× against a ≥ 4× bar and falls to
NS 0.9812 at seed 4, T = 32. Killed; its registered fallback `prune_m` (m ≤ 4) also misses NS ≥ 0.99 at T = 24 and 32.
B2 in 5.2 is the separately registered replacement.

### 5.5 Round 3 (running)

No numbers until they are bound. Running: a gap-onset map at f ∈ {0.85, 0.9, 0.95} on bed `dial`, to locate where
the set edge appears between f = 0.75 and f = 1; B2 re-derived on a second tuning seed, so the frozen race
configuration no longer rests on seed 9 alone; and the `PASS_V2` verdicts of `fm_resolvent` and `fm_hop1` replicated
on three secret draws.

### 5.6 Reproduction (source repository)

These commands run in `github.com/teerthsharma/resolvent`, not in this package.

```bash
# 5.1 bed dial: predictor + calibration, cells as (f, sigma/rho) pairs, extra arms, table, claim tests
cd sun/rjepa/r2
python r2.py pred
python r2.py cell 1 3 1 1 0.75 3 0 1 0 3
R2_KINDS=dj02f,dj02 python r2.py cell 1 3                 # M3 arm   -> results/res_f1.0_q3.0_s*_extra.json
R2_KINDS=dj0.5,dj1 R2_SFX=_eps python r2.py cell 1 3      # S1 sweep -> results/res_f1.0_q3.0_s*_eps.json
python r2.py table && python -m pytest -q test_claims.py

# 5.2 bed torus: tune on seed 9, evaluate seeds 3-5 (split by memory, merged into eval.json), claim tests
cd sun/rjepa/cameron/r2
python r2.py tune2 && python r2.py evnp && python r2.py evdj && python r2.py eval
python -m pytest -q test_r2.py

# 5.3 verifier red team, ranker contract, R-JEPA arms through the contract
python sun/daedalus/engine/m0.py
python sun/daedalus/engine/r2_rankers.py rankers.json
python sun/daedalus/results/r2/rjepa_arms.py
```

---

## 6. What did not hold

The numbers in this section are the source repo's own records (`sun/rjepa/foreman/BAR.md`, amendments A2-A3, and
the round-1 test run); only 0.024 and −0.0002 were independently re-run.

**The pre-registered gap bar missed.** The σ = 1 cell was registered to show a gap ≥ 0.03; it measured 0.024. The
gap is real but smaller than predicted at σ = 1, and the effect lives at σ ≥ 3.

**The Neumann tail was killed.** The hypothesis that the gap is the multi-hop tail of a resolvent predicted
resolvent − one-hop ≥ 0.01 on all seeds; it measured −0.0002. At 12 epochs the resolvent head led by about 0.01,
so the tail bought convergence speed, not ranking, and by (9) no linear equivariant tail can change a ranking at all.

**The first bed was void.** With per-candidate execution noise σ_e = 0.1 the Bayes ceiling at σ = 0.1 read 0.86,
below its own 0.90 validity bar, because irreducible noise was comparable to the nearest-pair gap of four
candidates. σ_e was fixed to 0.02 by that argument before any run at 0.02; every number above uses 0.02.

**The first learned run failed its learn gate.** At 12 epochs the resolvent head agreed with the Bayes pick on
0.8865 of starts against a 0.90 gate, so that run's comparisons were voided and the cell retrained at 36 epochs for
every head alike. The D-JEPA-spec head at ε = 0.2 learned no swap at all in that run and fails its own learn gate;
the learning-free Cor 1 ceiling replaced it as the deciding test of the bound.

**Measured in round 1 but not bound, pending re-run** (no numbers are quoted until a pre-registered test binds them;
round 1's LIN1-against-D-JEPA reading was struck and is replaced by the fresh-seed result in 5.2): the ε = 4 D-JEPA head recovering the horizon rule at σ = 10;
the hull-angle rule matching the Bayes rule at σ = 10 (the test `B3` passes in this suite, but it had no prior
failing test); the resolvent head's latency against the D-JEPA operator at K = 63.

---

## 7. Quick start

```bash
git clone <this repository> rjepa && cd rjepa
python -m pip install -e ".[test]"
python -m pytest -q                      # 119 passed, 1 skipped (CUDA parity, opt-in with RJEPA_CUDA=1)
python -m rjepa.reproduce bound          # {"1.0": {...}, "10.0": {"bayes": 0.381..., "bounded": 0.289...}}
```

```python
import torch
from rjepa import set_resolvent, stochastic_resolvent, Head
from rjepa import shared_error as S

q, k, v = (torch.randn(2, 16, 8) for _ in range(3))           # batch of 2 sets, K = 16 candidates
out = set_resolvent(q, k, v, rho=0.9)                           # (I - A)^-1 v, permutation-equivariant
perm = torch.randperm(16)
assert torch.allclose(out[:, perm], set_resolvent(q[:, perm], k[:, perm], v[:, perm]), atol=1e-5)
mix = stochastic_resolvent(q, k, v, g=0.9)                      # (1-g) P (I-gP)^-1 V, inside the hull of v

bed = S.make_bed(4000, sigma=10.0, seed=0)                      # shared error = 10 x candidate spread
print("Bayes", S.hit(S.bayes_pick(bed), bed),                   # ~0.38
      "distance", S.hit(S.dist_pick(bed["zhat_true"]), bed))    # below chance (0.25)

scores = Head("resolvent")(torch.randn(8, 4, 2))                # untrained set head: (8, 4) scores
```

Tests run on CPU; the learned σ = 1 cell is marked `slow` and deselected by default:
`python -m pytest -q -m slow`.

---

## 8. Requirements

- Python ≥ 3.11 (developed on 3.11.9).
- PyTorch ≥ 2.0; CPU is sufficient for everything, including the slow cell. Developed on torch 2.14.0.
- NumPy ≥ 1.26. No SciPy: the normal CDF is `torch.special.ndtr`.
- pytest ≥ 8 for the suite.

**Measured cost of the default suite** on the development box (shared with other jobs, torch threads 2, pytest
plugin autoload off): 119 passed, 1 skipped, 2 deselected in 140.9-170.1 s over three runs at about 65 % host CPU load
from other jobs; 73-87 s of that is the `dial` fast cell (`tests/test_dial.py`). Peak working set was measured on the earlier
98-test suite only: 682 MB, of which `import torch` alone (the CUDA 12.6 wheel) is 498 MB. A CPU-only torch wheel
lowers the baseline.

---

## 9. Limitations

The beds are toys chosen for exact truth: K = 4 and D = 2 for `shift`, K = 8 on a 2-torus for `torus`. The JEPA
predictor on `shift` uses an identity target encoder, so the shared error is exactly the start-state error by
construction; in a trained world model the shared fraction is unknown and nothing here measures it. The results
say what a ranking operator can and cannot close *given* shared error, not how much shared error a real JEPA has.

The D-JEPA operator is reimplemented from the paper's equations, not from the released code, and has not been
checked against the released checkpoints. It uses single-horizon tokens `[ẑ − g; rank]` (the paper uses two
backbones, 386-dimensional tokens and base `b = 0.42 r^L + 0.58 r^T`), and on bed `shift` the local-margin term is
dropped from its loss. Its failure to learn at ε = 0.2 on this bed is a statement about this reimplementation on
this bed.

The σ = 1 closure numbers are the source repo's recorded run; this package reruns the cell and tests the bars, not
the third decimal, because seed 0's predictor initialisation depends on the global RNG state at the point the cell
starts. Seed 1's one-hop closure also differs between the recorded run and the port's rerun; the cause is not
isolated.

The resolvents are dense `O(K³)` solves; nothing here is a fused kernel, and no speed claim is made.

The Round 2 results (Section 5) are not reproducible from this package's default suite. Bed `dial` is ported
(`rjepa/dial.py`), but its fast test pins only the sign of `dj4L` > `dj02` at f = 1 on 4,000 training and 2,000
evaluation starts; that sign held on seeds 0, 1 and 2 at those sizes by hit margins of 0.0055, 0.0105 and 0.002 (11, 21
and 4 starts of 2,000), so it is a smoke test, not a replication. The full-size cell sits behind `pytest -m slow` and was
not run for this commit (Bayes at M = 2,048 on 20,000 starts is hours on 2 CPU threads). The round-2 `torus` races and
the verifier live only in the source repository, and Section 5 cites their result files. Within those results, the B2 race configuration
was selected on a single tuning seed, and each `PASS_V2` ranker verdict rests on a single secret shuffle draw.

---

## License

Apache-2.0 (see `LICENSE`). `NOTICE` credits the source repository and the D-JEPA paper, from whose equations the
relational operator was reimplemented without copying code. If you use the relational operator, cite D-JEPA
(`CITATION.cff` lists both).

*Invented by [Teerth Sharma](https://teerthsharma.vercel.app/)*
