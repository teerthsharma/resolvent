**Invented by [Teerth Sharma](https://teerthsharma.vercel.app/)** · [github.com/teerthsharma/resolvent](https://github.com/teerthsharma/resolvent) · [teerths57@gmail.com](mailto:teerths57@gmail.com)

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" />
  <img alt="PyTorch 2.x CPU" src="https://img.shields.io/badge/torch-2.x_CPU-ee4c2c?style=flat-square" />
  <img alt="Lean 4 v4.7.0" src="https://img.shields.io/badge/Lean_4-v4.7.0-blue?style=flat-square" />
  <img alt="166 theorems and 41 lemmas" src="https://img.shields.io/badge/theorems_%2B_lemmas-166_%2B_41-success?style=flat-square" />
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" />
  <a href="https://teerthsharma.github.io/resolvent/"><img alt="Docs: GitHub Pages" src="https://img.shields.io/badge/docs-GitHub_Pages-6ee7b7?style=flat-square" /></a>
</p>

<h1 align="center">resolvent</h1>

<p align="center">
  <b>How topological structure helps JEPA world models decide.</b><br/>
  <span>A JEPA planner ranks K candidate futures one at a time, by each one's latent distance to the goal.
  When the prediction error is shared by every candidate of a start, the right pick depends on the shape of the
  whole candidate set: its convex hull, and the paths between its members. A resolvent over the set,
  <code>(I − A)⁻¹</code>, reads that structure; a pointwise head cannot. This repository measures where that
  helps D-JEPA's decision-local ranking, where it does not, and what it costs. It builds on D-JEPA and makes no
  novelty claim.</span>
</p>

<p align="center">
  <a href="#abstract">Abstract</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#what-we-got-wrong">What we got wrong</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#1-background-and-prior-art">Prior art</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#2-theoretical-foundation">Theory</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#4-results">Results</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#5-lineage-the-ceq-attention-family">Lineage</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#6-repository-map">Map</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#7-quick-start">Quick start</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://teerthsharma.github.io/resolvent/"><b>Site</b></a>
</p>

<p align="center"><sub>Every number on this page names the file that records it and was bound by an independent
re-run before it was allowed here. Numbers fitted after the data was read are labelled as such. Everything that
did not hold is in <a href="#what-we-got-wrong">What we got wrong</a>, <a href="docs/FAILS.md">docs/FAILS.md</a> and
<a href="MISTAKES.md">MISTAKES.md</a>.</sub></p>

---

## Abstract

A latent world model plans by scoring K candidate action sequences from one start and executing the one whose
predicted future lies nearest the goal embedding. D-JEPA (Liu et al., arXiv 2609.24749) names the failure of that
rule the *decision-local gap* and closes part of it with a bounded, permutation-equivariant relational operator
over the candidates. This repository isolates one mechanism behind the gap and measures a family of set operators
against it. When the error separating prediction from outcome is **shared** by every candidate of a start, it
translates the whole predicted set, and the Bayes-optimal pick becomes a function of the set's structure (which
candidates sit on its convex hull, how they relate to one another) rather than of each candidate's own distance.
A resolvent over the candidate set, `(I − A)⁻¹ v` with `‖A‖ < 1`, sums every path of messages between candidates in
closed form and can express that rule; a pointwise head with the same inputs cannot.

On an exact-truth dial with K = 63 candidates and a learned predictor, at full shared error, a D-JEPA-spec operator
with its bound widened to ε = 4 scores **0.732 / 0.703 / 0.787** normalised score (NS) on three seeds, against
**0.352 / 0.354 / 0.358** for D-JEPA as specified (ε = 0.2). The edge over D-JEPA is affine in the room left to
close, `edge = −0.0683 + 0.6618 × closable`: a law **fitted after reading the data**, which then predicted an
untouched cell within 0.004 NS on 3 of 3 seeds. It vanishes where the error is not shared. A call-accounted
race reaches Bayes quality within 0.01 NS at **4.52×–7.89×** fewer predictor calls. A sandboxed verifier rejects
**22/22** planted cheats and **7/7** planted ranker cheats (6/7 at their named stage).

**Keywords:** JEPA world models · planning · candidate ranking · decision-local gap · D-JEPA · resolvent ·
Neumann series · permutation equivariance · shared error · convex hull · Bayes decision rule · Lean 4.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 resolvent for JEPA ranking · bound by the round-2..4 inspectors · Windows 11 · Python 3.11.9 · torch 2.14.0 (CPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 bed dial, K = 63, learned predictor, f = 1 (full shared error), sigma/rho = 3, 3 seeds x 20,000 eval starts, NS
   dj4L   D-JEPA-spec operator, eps = 4          0.732   0.703   0.787
   dj02   D-JEPA as specified, eps = 0.2         0.352   0.354   0.358
 affine closable law (P2a, round 4; FITTED AFTER READING DATA on 15 cell-seeds, passed on a fresh cell)
   edge = -0.0683 + 0.6618 x closable            (0.85, 2): predicted 0.037 0.063 0.059
                                                            measured  0.035 0.060 0.062   within 0.004, 3/3
 bed torus, B2 race vs full rollout K(M+1)T calls, 15 seed-lead rows
   predictor calls saved per decision            4.52x - 7.89x   within 0.01 NS of the full rollout
   B4 at budget 656, T = 32, seeds 14/15/16      NS' 0.99101  0.99213  0.99021
 verifier (daedalus/)
   planted cheats rejected                       22 / 22   (round 2, 0 errors)
   planted ranker cheats rejected                 7 / 7    (round 4, r01-r07; 6/7 at their named stage: r06
                                                            is rejected at V2, earlier than its registered V3;
                                                            2/2 controls admissible, 0 ERROR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Sources: `experiments/dial/r2/` (round 2 table), `experiments/dial/r4/foreman/results/table.json` and commit `c1a3e8e`
(round 4), `experiments/torus/r2/`, `experiments/torus/r3/` and `experiments/torus/r4/sharp.json` (races),
`daedalus/results/r2/m0_r2c_redteam.json` and `daedalus/results/r4/rankers_after.json` (verifier).

---

## What we got wrong

**This repository used to claim that no one had built its operator before. That claim is withdrawn.** Until
2026-09-28 the README opened on "Softmax attention and Markov path composition are the same operator" and carried
a table scoring self-attention, JEPA and resolvent row by row, with ✗ ✗ ✓ on eight rows: normalised and
unnormalised reads as settings of one head, gates that close exactly, exact path products, closed-form committors,
`do(a)` as a rank-1 edit, refusal, edge flows with curl, corners proved in Lean. That thesis line and that table are retracted, for three reasons.

1. **D-JEPA owns the combination this repository claimed.** A JEPA world model, a set of competing candidate
   consequences, and a bounded permutation-equivariant operator that re-ranks them: D-JEPA (arXiv 2609.24749,
   submitted 2026-09-21, code Apache-2.0) states all three, with a proof of the bound (Prop 2, Cor 1) and
   results on PushT, Reacher, RoboTwin, PiPER and driving. The facts were fetched in this project and are recorded
   at `experiments/wilson/facts.json`. The resolvent's set heads build on it; its relational operator is
   reimplemented here from the paper's equations (`resolvent/djepa.py`, see `NOTICE`).
2. **An ICLR paper covers the same ground, and it is not yet identified.** The author reports it; a search of
   eleven queries did not find a fetched page stating an ICLR venue for a JEPA + topology / resolvent paper. Its
   row in the prior-art table stays empty until the author supplies the link.
3. **The ✓ column compared the family with its own settings.** Most of its rows were definitional (the family was
   built to have them), and the rows that were measured measured the operator, not a model: the family's
   language-model win turned out to be positional encoding, and its prediction leg tied an eight-bin histogram
   (Section 5).

The canon records the retraction as row C10 of `docs/canon/CORRECTIONS.md`. The remaining failures, each with
the number that killed it:

| What was claimed or registered | What was measured | Where |
|---|---|---|
| The attention family cannot represent a closed class (the structural claim) | sparsemax opens arbitrarily many closed classes by direct construction; retracted 2026-09-20. The operator is a kernel, not a model | `docs/FAILS.md`, `docs/experiment.md` |
| The family has a language-model win over softmax | a zero-parameter ALiBi twin recovers 108.8 / 106.7 / 108.8 % of it and beats the family at 3/3 seeds; a RoPE twin 96.3 / 99.7 / 101.4 %; a published forget gate (FoX) 100.2 / 100.3 / 110.9 %. The win is positional encoding | `tests/foreman/phase_j/N2/foreman/FOREMAN_REPORT.md`, `docs/STATUS.md` |
| The operator predicts (the North Star's prediction leg) | resolution `0.001469` against an oracle ceiling of `0.10117`, tied by an eight-bin histogram of piece count from its own input | `docs/STATUS.md`, `docs/FAILS.md` |
| P2: at f ≥ 0.75 the set arms beat D-JEPA by ≥ 0.05 NS | holds at f = 1, fails at f = 0.75, where `dj4L` scores below `dj02` on every seed; the closable gap there is 0.002–0.004 hit | `experiments/dial/r2/BAR.md` |
| P2n: the edge turns on exactly where the closable room is ≥ 0.25 NS | killed on the untouched cell (0.9, 2): closable 0.234 with edges +0.106 / +0.080 / +0.109 NS. The proportional replacement P2r is killed at (0.8, 2) seed 1 | commit `c1a3e8e`, `experiments/dial/r4/foreman/BAR.md` |
| The gap is the multi-hop tail of the resolvent (round 1) | resolvent − one-hop = −0.0002 hit; by (9) below no linear equivariant tail can change a ranking | `experiments/shift/BAR.md` |
| The σ = 1 gap is ≥ 0.03 (round 1 bar) | 0.024; the effect lives at σ ≥ 3 | `experiments/shift/BAR.md` |
| The ε = 4 operator is a *bounded* operator | 2ε = 8 exceeds the base range of 1, so the bound constrains nothing; the verifier measured max \|score − base\| of 2.26 / 2.40 / 2.29. Its pass is a ranking result, not evidence for a bound | `daedalus/results/r3/r3_runs.json` |
| The verifier's V2 null cannot be gamed | plant r06 passed V2 in round 3, r07 passed V3; fixed in round 4 by running the null in its own sandbox process | commit `c1a3e8e` |
| LIN1 loses to a 3-member ensemble at T = 40 (round 3) | does not replicate in round 4 (gap +0.0056 on the sharp instrument); both regime explanations killed | commit `c1a3e8e` |

---

## 1. Background and prior art

### Why a set operator?

**The plug-in rule ranks candidates one at a time.** Latent-distance planning scores candidate k by
`‖ẑ_k − z_g‖` alone. If prediction error were independent per candidate this is the right rule, and the Bayes pick
and the distance pick coincide (test `B0`: agreement ≥ 0.97 at zero shared error).

**Error from the start state is not per-candidate.** Every candidate of a start is rolled out from the same
misestimated start, so that error moves all K predicted futures together. Ranking under a common unknown
translation is a question about the whole set, the structure common random numbers exploit in simulation-based
planning.

**Under a large shared translation the Bayes rule is a convex-hull property.** As the shared error grows past the
candidate spread, the chance that candidate k is best tends to the exterior angle of its Voronoi cell (4). A
candidate inside the hull of the set has exterior angle 0, and it is exactly the candidate nearest-to-goal
favours when the planner aims every candidate at the goal. Distance then drops below chance.

**A set operator can express this; a pointwise one cannot.** A head that scores each candidate from its own
prediction cannot represent a rule that depends on the others. A head that exchanges messages across the set can,
provided the messages are permutation-equivariant (the set has no order) and bounded. The resolvent is the
bounded, closed-form sum of every such message path.

### Prior art

| System | What it does with the K candidates | Set interaction | Bound on the correction | Where it is stronger than this repository |
|---|---|---|---|---|
| D-JEPA (Liu et al. 2026, arXiv 2609.24749; code Apache-2.0) | re-ranks shared candidates with a bounded relational operator over native cost ranks | 2-layer Transformer, no candidate-order encoding | \|δ_i\| ≤ ε = 0.2 (Prop 2, Cor 1) | real tasks: PushT n = 256, 87.11 relational vs 76.95 TD-JEPA native (paper, Table 5); Reacher, RoboTwin, PiPER, driving. This repository has toy beds only |
| ICLR paper — **to be identified; author to supply** | — | — | — | — |
| ACPC (An et al. 2026, arXiv 2608.12939; code MIT) | diagnoses JEPA world models by rolling a clean and a perturbed history forward under the same actions | none (a diagnostic, not a ranker) | n/a | measures the robustness of the representation itself, which this repository assumes |
| ChaCAL (Fagnou et al., EMNLP 2024) [U] | the causal resolvent read `(1 − g) P (I − gP)⁻¹ V` over a sequence, to which the CEQ lineage attributes its sequence operator | causal (nilpotent off the diagonal) | g < 1 | a sequence operator on language; this repository moves the read to an unordered set |
| Common random numbers (Yadav et al. 2026, arXiv 2605.04732; Klein et al. 2024, arXiv 2409.02086) | evaluates candidate policies on shared simulator randomness | the pairing itself | n/a | proven variance reduction in simulators; no simulator-side result here |
| **resolvent** (this repository) | Bayes ceiling, distance floor, pointwise / one-hop / resolvent set heads and a D-JEPA-spec operator on three exact-truth beds, a call-accounted race, a sandboxed verifier | resolvent `(I − A)⁻¹`, one hop, or Transformer | ρ < 1, refused otherwise; ε for the D-JEPA head | measures *where* a set operator helps (shared error), not *whether* it helps on a real task |

[U] marks a source not fetched for this repository. No fetched source states that error shared across candidates
cancels in the argmin ranking of a *learned* world model, or measures the shared fraction; D-JEPA does not decompose
its error into shared and per-candidate parts (`experiments/wilson/facts_r2.json`, items 2.10 and 4.8).

---

## 2. Theoretical foundation

### 2.1 The decision-local gap as a shared-error decision rule

Bed `shift` (`resolvent/shared_error.py`). The planner observes a start estimate $y$; the true start is

$$
s = y - \sigma \xi, \qquad \xi \sim \mathcal N(0, I_D), \qquad\text{so}\quad s \mid y \sim \mathcal N(y, \sigma^2 I_D). \tag{1}
$$

Candidates aim at the goal $g = 0$ from the estimate, $a_k = -y + \rho\, r_k$ with $r_k \sim \mathcal N(0, I_D)$,
and execute to

$$
z_k = s + a_k + \sigma_e \eta_k = \rho\, r_k - \sigma \xi + \sigma_e \eta_k, \qquad k^\star = \arg\min_k \lVert z_k \rVert. \tag{2}
$$

The term $\sigma\xi$ is the same for every $k$. The plug-in rule and the Bayes rule are

$$
\hat k_{\text{dist}} = \arg\min_k \lVert \hat z_k \rVert, \qquad
\hat k_{\text{Bayes}} = \arg\max_k\ \Pr\!\left(k = k^\star \mid y, a_{1:K}\right), \tag{3}
$$

with the probability estimated from $M = 1024$ posterior draws of $(\xi, \eta)$ common to all $K$ candidates. As
$\sigma \to \infty$, $k^\star \to \arg\max_k \xi^\top r_k$ and, for isotropic $\xi$,

$$
\Pr(k = k^\star) \;\to\; \frac{1}{2\pi}\left|\{u \in S^1 : k = \arg\max_j u^\top r_j\}\right|, \tag{4}
$$

the exterior angle of candidate $k$'s Voronoi cell (`hull_angle_share`): zero for a candidate inside the convex
hull. This is the structure a pointwise ranker cannot see. Splitting the error into a shared share $1 - m$ and a
per-candidate share $m$ is the control: at $m = 1$ the gap vanishes (test `B10`).

### 2.2 D-JEPA's bounded correction and its reach

D-JEPA's operator (`resolvent/djepa.py`, reimplemented from arXiv 2609.24749 §3 and App. B.1) scores

$$
h_i = \mathrm{TF}(\mathrm{enc}(v_i)), \qquad
\delta_i = \varepsilon \tanh\!\big(W_{\text{up}} \tanh(W_{\text{down}} h_i)\big), \qquad
s_i = b_i + \delta_i, \tag{5}
$$

with $b_i$ the normalised base rank. Since $|\delta_i| \le \varepsilon$, the selected $\hat i = \arg\min_i s_i$ satisfies

$$
b_{\hat i} \;\le\; \min_i b_i + 2\varepsilon \tag{6}
$$

(the paper's Prop 2 / Cor 1). The Bayes rule restricted to ranks within $2\varepsilon$ of the minimum is a ceiling
for every operator obeying (6), learned or not. When $2\varepsilon$ exceeds the base range, (6) constrains nothing.

### 2.3 The resolvent on a candidate set

A causal resolvent is nilpotent off the diagonal and cannot reach a pole. On an unordered set the coupling must be
diagonal-free and symmetric in the listing order, it is no longer nilpotent, and the bound becomes load-bearing
(`resolvent/resolvent.py`):

$$
A = \rho\, \frac{\operatorname{offdiag}(q k^\top)}{\operatorname{rowL1} + \epsilon}, \quad
\lVert A \rVert_\infty \le \rho < 1, \quad
\text{out} = (I - A)^{-1} v = \sum_{h \ge 0} A^h v, \quad
\lVert (I - A)^{-1} \rVert_\infty \le \frac{1}{1 - \rho}. \tag{7}
$$

$\rho \ge 1$ raises `ValueError`. $(A^h)_{ij}$ sums the weights of every length-$h$ message path from candidate $j$
to candidate $i$, so the resolvent reads the path structure of the whole set in one solve. The stochastic form uses
$P = \operatorname{softmax}(q k^\top / \sqrt d)$, row-stochastic with $\rho(gP) = g$:

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

a positive rescaling plus a common shift, so the ranks of $s$ are unchanged. A resolvent changes a ranking only
through a **data-dependent** coupling $W(z)$, which is what the set heads learn.

### 2.5 LIN1: a one-VJP chance ranker

On bed `torus` (`resolvent/standard_map.py`, the Chirikov standard map on the 2-torus) success is landing in a ball
of radius $R$ around $G$ after $T$ steps. Linearising the rollout at the estimate and projecting on the unit goal
direction $n$,

$$
\Pr(\text{success}_k) \approx \Phi\!\left(\frac{R - d_k}{\delta\, \lVert J_k^\top n_k \rVert}\right), \tag{10}
$$

one vector-Jacobian product per candidate (`lin1_prob`).

---

## 3. Implementation

| Module | Contents |
|---|---|
| `resolvent/resolvent.py` | `build_A`, `set_resolvent` (7), `neumann_resolvent`, `stochastic_resolvent` (8), `resolvent_apply`, `linear_equivariant_resolvent` (9) |
| `resolvent/djepa.py` | `RelationalOperator` (5), `rank01`, `dj_loss`: D-JEPA reimplemented from the paper |
| `resolvent/heads.py` | `Head("point" \| "hop1" \| "resolvent" \| "djepa" \| "djepa4")`, `dist_rank`, `fit` |
| `resolvent/shared_error.py` | bed `shift` (1)-(2), `bayes_pick`, `dist_pick`, `hull_angle_share` (4), `run_cell` |
| `resolvent/dial.py` | bed `dial`: K = 63 shared-error dial, learned predictor, `bayes_P` with the A6.1 tie-break, `DialHead`, `saturation` |
| `resolvent/standard_map.py` | bed `torus`, exact tangent Jacobian, `lin_prob`, `lin1_prob` (10), rankers |
| `resolvent/reproduce.py` | `python -m resolvent.reproduce gap \| bound \| closure` |

**Low precision is promoted, not trusted.** There is no fp16/bf16 LU, and `|q·k| > 65504` overflows fp16 to `inf`,
after which the row normalisation is `inf/inf = NaN`. Both resolvents solve in ≥ float32 and cast back.

**Dense LU, not a triangular solve.** A triangular solve applied to a non-causal `P` silently reads only the lower
triangle; the stochastic form uses `torch.linalg.solve`.

**Padding is exact.** Masked candidates are removed from `A` on both axes, output 0, and an all-masked row returns 0
instead of NaN. A padded set's real outputs equal the unpadded set's to 1e-12.

**The D-JEPA head is zero-initialised.** `W_up` starts at zero, so an untrained head returns the base ranking
exactly, and every departure from latent distance is learned.

The experiment records live beside the library, filed by bed (Section 6); the verifier that grades candidate
rankers is `daedalus/`.

---

## 4. Results

### 4.1 Round 1: the gap and what closes it (bed `shift`)

K = 4, D = 2, ρ = 1, σ_e = 0.02, 3 seeds × 4,000 held-out starts, Bayes by M = 1,024 common draws. Re-run by an
independent inspector and bound to a pre-registered failing test.

| Quantity | Value | Condition | Reproduce |
|---|---|---|---|
| Gap, Bayes − latent distance (top-1 hit) | **0.000 / 0.024 / 0.149 / 0.213 / 0.241** | σ = 0.1 / 1 / 3 / 10 / 30 | `python -m resolvent.reproduce gap` |
| Best hit reachable under D-JEPA's Cor 1 (ε = 0.2) | **0.289** vs Bayes **0.381** | σ = 10, learning-free | `python -m resolvent.reproduce bound` |
| Gap closed by the resolvent set head | **0.947** | σ = 1, learned predictor, 36 epochs | `python -m resolvent.reproduce closure` |
| Gap closed by the one-hop set head | **0.925** | same cell | same |
| Gap closed by the pointwise head (control) | **0.028** | same cell, same inputs, no set interaction | same |
| Linear equivariant resolvent | **rank-inert** | proof (9); 300-draw property test | `pytest tests/resolvent -k rank_inert` |

The learning-free rows are pinned to three decimals by `tests/resolvent/test_reproduce.py`. The σ = 1 closure rows
are the recorded run in `experiments/shift/`; the package's rerun passes all four registered bars.

### 4.2 Round 2: the shared-error dial at K = 63 (bed `dial`)

Latent D = 8, horizon 5, true dynamics $z' = 1.1\,Qz + 0.5\tanh(Cz + Ba)$, K = 63 candidates, a learned MLP
predictor, 3 seeds × 20,000 held-out starts, M = 2,048 Bayes draws. NS = (hit − 1/K) / (hit_Bayes − 1/K). Every
learned arm sees the same 12-d token and 4,000-step budget at 67,805–69,377 parameters. Source `experiments/dial/r2/`.

| Arm at (f = 1, σ/ρ = 3) | NS, seed 0 / 1 / 2 | What it is |
|---|---|---|
| Bayes restricted to base ranks within 2ε = 0.4 | **0.944 / 0.944 / 0.970** | learning-free ceiling of any ε = 0.2 operator |
| `dj4L` | **0.732 / 0.703 / 0.787** | D-JEPA-spec operator, ε = 4, LIN1 base ranks |
| `hop1` | **0.501 / 0.467 / 0.497** | one-hop set head |
| `dj02` | **0.352 / 0.354 / 0.358** | D-JEPA as specified, ε = 0.2 |
| `dist` (floor) | **0.338 / 0.343 / 0.335** | latent-distance planning |

**Reach is not the wall; saturation is.** The Bayes pick within the ε = 0.2 reach scores 0.944–0.970 NS, while the
learned ε = 0.2 operator sits 0.011–0.024 above the floor, its tanh correction running at 79 % of its bound
(mean |δ|/ε 0.792 / 0.792 / 0.791). Widening ε to 0.5, 1 or 4 with everything else fixed lifts NS to the same level.

**The set edge exists only where the error is shared.** The closable gap is 0.0455 hit at f = 1, 0.002–0.004 at
f = 0.75, and within ±0.0014 of zero at f = 0 (σ/ρ = 3), where no set arm beats `dj02` by more than 0.0008.

### 4.3 Rounds 3 and 4: where the edge turns on, and the affine law

Round 3 (`experiments/dial/r3/foreman/results/table.json`) measured the edge `dj4L − dj02` at f = 0.85 / 0.9 / 0.95:
+0.105 / −0.023 / +0.149, +0.170 / +0.120 / +0.079 and +0.158 / +0.244 / +0.254 NS. Round 4
(`experiments/dial/r4/foreman/`, n_eval 60,000 × 3 seeds) killed the frozen-threshold law P2n and its proportional
replacement, then registered **P2a**, `edge = −0.0683 + 0.6618 × closable`, fitted on 15 cell-seeds **after
reading round-4 data**, before the cell (0.85, 2) existed. On that cell it predicted 0.037 / 0.063 / 0.059 against
measured 0.035 / 0.060 / 0.062. One fresh cell passed; the law is a fit that survived one test, not a derivation.

### 4.4 Fewer predictor calls (bed `torus`)

| Quantity | Value | Compared against |
|---|---|---|
| LIN1 band-mean NS, seeds 3 / 4 / 5 | **0.880 / 0.881 / 0.879** | best D-JEPA-spec operator 0.729 / 0.737 / 0.735 |
| B2 race, mean call saving per decision | **4.52×–7.89×**, within 0.01 NS on all 15 seed-lead rows | full rollout, K (M + 1) T calls |
| B2 re-derived on tuning seed 10, eval seeds 6 / 7 / 8 | same configuration selected; NS ≥ 0.9922, saving 4.54×–7.87× | same |
| B4 at budget 656, T = 32, seeds 14 / 15 / 16 | **NS′ 0.99101 / 0.99213 / 0.99021**, p90 call ratio 3.06 | the 128-member full rollout, beaten on 3/3 seeds |

Sources: `experiments/torus/r2/`, `experiments/torus/r3/eval.json`, `experiments/torus/r4/sharp.json`.

### 4.5 A verifier for rankers (`daedalus/`)

Candidate code runs in a sandbox process that sees no evaluation labels and may not read anything in the
repository outside its own directory. The round-1 verifier rejected **22/22** planted cheats with 0 errors under
round-2 code; a `ranker` contract caught **5/5** planted ranker cheats at their named stage. Round 3 added a
verifier-owned pool (V3): the resolvent set head and the ε = 4 operator pass it on 3/3 secret draws, the one-hop head
and a stochastic resolvent do not. Round 4 closed the V2 null hole; r01–r07 are **7/7** rejected, 6/7 at their named stage (r06, registered to be caught at V3, is now
rejected earlier, at V2, by the null fix), both controls
admissible, 0 ERROR (`daedalus/results/r4/rankers_after.json`).

---

## 5. Lineage: the CEQ attention family

The resolvent read comes from an earlier programme in this repository: a causal attention family with three
switches (`β`, `g`, `qk`) in which softmax attention, unnormalised-kernel attention and the exact path product of a
Markov chain are settings of one head, proved in Lean 4 (13 files, **166 theorems + 41 lemmas**, 0 `sorry`,
`python scripts/lean_count.py`) and matched bitwise in code (`ceq/arm_smprime.py`, `ceq/arm_pl.py`,
`ceqjepa/operator.py`). Its bound verdicts, which are also why the programme moved to the candidate set:

| Verdict | Number | Source |
|---|---|---|
| The operator represents order | on S5 handed over as bare integers: operator **0.8620** vs commuting control **0.2860** at matched 404 parameters, 5/5 seeds | `docs/STATUS.md` |
| **The ALiBi twin beats the family** | a zero-parameter ALiBi twin recovers 108.8 / 106.7 / 108.8 % of the family's language-model win and beats it by 0.016–0.023 nats at 3/3 seeds | `tests/foreman/phase_j/N2/foreman/FOREMAN_REPORT.md` |
| **FoX recovers the win** | a published forget gate recovers 100.2 / 100.3 / 110.9 % | same |
| **The win is positional encoding** | a zero-parameter RoPE twin recovers 96.3 / 99.7 / 101.4 % (mean 99.2 %) | same; `docs/STATUS.md` row 1 |
| The operator does not predict | resolution 0.001469 vs ceiling 0.10117, tied by an eight-bin histogram | `docs/STATUS.md` |
| The structural claim | retracted: sparsemax opens n closed classes by construction; what survives is cross-row consistency at 31.875× vocabulary cost | `docs/experiment.md` |

The family's code stays where it was (`ceq/`, `ceqjepa/`, `scale/`, `scripts/`, `lean/`, `results/`, most of
`tests/`), its ledgers stay at the root, and its paper and canon are on the [site](https://teerthsharma.github.io/resolvent/).

---

## 6. Repository map

| Path | What it is |
|---|---|
| `resolvent/` | **the library**: set resolvents, D-JEPA-spec operator, beds `shift`, `dial`, `torus` (Section 3) |
| `tests/resolvent/` | the library's contract tests (7 test files and a conftest) |
| `experiments/shift/` | bed `shift`, round 1 (K = 4) |
| `experiments/dial/r2/`, `r3/foreman/`, `r4/foreman/` | bed `dial`, rounds 2–4 (K = 63 shared-error dial); each round has its `BAR.md` pre-registration. r3 and r4 keep a `foreman/` level because round 4's code resolves it and its results pin that code's sha256 |
| `experiments/torus/` (with `r2/ r3/ r4/`) | bed `torus`, rounds 1–4 (standard map, LIN1, the B2/B4 races) |
| `experiments/cost/` | quality and per-decision cost bench of the set resolvent |
| `experiments/wilson/` | fetched facts about D-JEPA and ACPC used by the rounds |
| `experiments/plan_a/` | Plan A ("Canon"), **killed at P2**; the negative is kept |
| `daedalus/` | the verifier: sandboxed candidate execution, planted cheats, controls, registry of bars |
| `ceq/`, `ceqjepa/`, `scale/`, `scripts/`, `lean/` | the CEQ attention family (Section 5) and its Lean proofs |
| `results/` | the family's recorded runs; recorded paths inside are provenance and are never rewritten |
| `tests/` (other folders) | the family's tests, filed by who ran them, not by subject: `foreman/`, `chase/`, `cameron/`, `wilson/`, `watson/`, `house/` are the roles of the review team that designed, attacked, proposed alternatives to, fact-checked and audited each round (`tests/foreman/phase_j`, `phase_k` hold Phases J and K); `mercury/`, `venus/`, `mars*/`, `jupiter/`, `saturn/`, `neptune/`, `deimos/` and `w2/`–`w15/` are campaign and iteration names; `arm_*`, `beds/`, `certs/`, `curvature/`, `gate0/`, `loop/`, `x35*/` are named by subject |
| `docs/` | the site and the paper; `docs/canon/` is append-only (`CORRECTIONS.md` is its only door) |
| `MISTAKES.md`, `STRUCK.md`, `MATHEMATICS.md`, `MODEL_CARD.md`, `COSTS.md`, `V16_CALIBRATION.md`, `V17K_RULINGS.md` | ledgers, published on the site under `/ledgers/` |
| `MOVED.md` | every path this restructure moved, old → new, and the commit where the old layout is readable |

---

## 7. Quick start

```bash
git clone https://github.com/teerthsharma/resolvent && cd resolvent
python -m pip install -e ".[test]"
python -m pytest tests/resolvent -q -m "not slow"   # 120 passed, 1 skipped (CUDA parity, opt-in with RJEPA_CUDA=1), 2 deselected
python -m resolvent.reproduce bound                 # {"1.0": {...}, "10.0": {"bayes": 0.381..., "bounded": 0.289...}}
```

```python
import torch
from resolvent import set_resolvent, stochastic_resolvent, Head
from resolvent import shared_error as S

q, k, v = (torch.randn(2, 16, 8) for _ in range(3))           # batch of 2 sets, K = 16 candidates
out = set_resolvent(q, k, v, rho=0.9)                           # (I - A)^-1 v, permutation-equivariant
perm = torch.randperm(16)
assert torch.allclose(out[:, perm], set_resolvent(q[:, perm], k[:, perm], v[:, perm]), atol=1e-5)
mix = stochastic_resolvent(q, k, v, g=0.9)                      # (1-g) P (I-gP)^-1 V, inside the hull of v

bed = S.make_bed(4000, sigma=10.0, seed=0)                      # shared error = 10 x candidate spread
print("Bayes", S.hit(S.bayes_pick(bed), bed),                   # ~0.38
      "distance", S.hit(S.dist_pick(bed["zhat_true"]), bed))    # below chance (0.25)
```

The lineage runs from the same checkout: `pip install -r requirements.txt`, then
`python -m pytest tests/arm_smprime tests/arm_pl -q` (the Lean corners, in code) and
`(cd lean && lake build CEQ)`.

## 8. Requirements

- Python ≥ 3.11 (developed on 3.11.9); PyTorch ≥ 2.0, CPU is enough for the library (developed on torch 2.14.0);
  NumPy ≥ 1.26. The lineage pins its own stack in `requirements.txt`.
- Lean 4 v4.7.0 for the proofs; the first `lake build` downloads about 4.2 GB of mathlib.

## 9. Limits

The beds are toys chosen for exact truth: K = 4 at D = 2 (`shift`), K = 8 on a 2-torus (`torus`), K = 63 at D = 8
with a known 5-step map (`dial`). The results say what a ranking operator can and cannot close *given* shared error,
not how much shared error a trained JEPA has; nothing here measures the shared fraction of a real world model, and
no result is on a D-JEPA task. The D-JEPA operator is reimplemented from the paper's equations, not the released
code, and has not been checked against the released checkpoints. The affine law P2a was fitted after reading the
data it summarises and has passed one fresh cell. The resolvent set head and the ε = 4 operator remain
inseparable on the existing beds. The B2 race configuration rests on two tuning seeds, and its tail saving at T = 32
is not bound. The verifier's sandbox is an in-process audit hook, not an OS boundary, and a candidate can still
detect its null through label/distance correlation. The ICLR paper that covers this ground is unidentified, so the
prior-art table is incomplete. The site's pages still tell the CEQ family's story; they are regrouped, not rewritten.

## Citation

```bibtex
@software{sharma2026resolvent,
  author = {Sharma, Teerth},
  title  = {resolvent: how topological structure helps JEPA world models decide},
  year   = {2026},
  url    = {https://github.com/teerthsharma/resolvent}
}
```

`CITATION.cff` lists D-JEPA; cite it for the relational operator.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) (third-party code, and the D-JEPA
reimplemented-from-paper notice).

*Invented by [Teerth Sharma](https://teerthsharma.vercel.app/)*
