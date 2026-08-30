# V13 TIER-6 PRIOR ART — the five lineages owed to the Q1/Q2/Q3 head

Round 13, Tier-6 fetch node. Normal English by the artifact exemption.

**Marking convention.** `[V]` = the actual source text was fetched and read in
this session, and every equation quoted below was taken from that text. `[V-t]`
= only the bibliographic record (title/authors/venue, sometimes abstract) was
confirmed; no equation was read. `[U]` = unfetched and still owed. Nothing is
marked `[V]` on the strength of recall.

**Locator convention.** Every entry carries a DOI, an arXiv ID, or venue+year.
Where a PDF was reached but its text layer defeated the fetcher's own extractor,
the text was recovered locally from the same fetched bytes; those entries are
still `[V]` because the source text, not a description of it, was read.

---

## Verdict

The kill clause does **not** fire on its literal conjunction — no published work
was found that scores barriers over discrete exits from a detected metastable
state *and* uses the result to condition autoregressive token emission. It fires
on everything except that last clause: Henkelman and Jónsson, *J. Chem. Phys.*
115, 9657 (2001) already detect the current basin, enumerate its exits on the
fly by dimer saddle search, score each exit by `k^hTST = ν exp(−ΔE‡/k_BT)`, and
select one with probability proportional to that rate — which is
`softmax_a(−ΔE‡_a / k_BT)` written in chemistry notation — then condition the
next state on the exit chosen. The single most dangerous overlap is therefore
that Q3 is adaptive kinetic Monte Carlo with a learned `ΔÊ‡` and a token
decoder bolted on, and the residual claim is the substrate swap, not the
mechanism. The second most dangerous is Sutton–Precup–Singh Eq. (7),
`p^o_{ss'} = Σ_{k=1}^∞ p(s′,k) γ^k`, which is the same discounted-resolvent
object the project names as its own propagation law, published in 1999 as the
option model. The third is that `committor = absorbing resolvent` is Grinstead
and Snell Theorem 11.6 (`B = NR`, `N = (I−Q)^{-1}`), undergraduate textbook
material, so the four-names-one-object sentence is an exposition of standard
facts and must never be written as a discovery.

---

## 1. Markov state models — Noé / Pande / Chodera lineage

### 1.1 Prinz, Wu, Sarich, Keller, Senne, Held, Chodera, Schütte, Noé (2011) `[V]`

*"Markov models of molecular kinetics: Generation and validation."*
*J. Chem. Phys.* **134**, 174105. DOI `10.1063/1.3565032`.

**Implied timescales, Eq. (18):**

```
t_i = −τ / ln λ_i
```

used inside the spectral decomposition, Eq. (19):

```
u_{t+kτ}(x) = 1 + Σ_{i=2}^{m} exp(−kτ / t_i) ⟨u_t, ψ_i⟩_μ ψ_i(x) + T_fast(kτ) ∘ u_t(x)
```

**The Chapman–Kolmogorov test, Eq. (60):**

```
[T̂(τ)]^k ≈ T̂(kτ)
```

The paper does not stop at Eq. (60); it states that comparing `n × n` matrix
elements is unusable and defines the test on set-probabilities instead.
Reference distribution restricted to a set `A`, Eq. (61):

```
w^A_i = π_i / Σ_{j∈A} π_j   for i ∈ A ;   w^A_i = 0 otherwise
```

Trajectory-side observable, Eqs. (62)–(63):

```
p_MD(A, A; kτ) = Σ_{i∈A} w^A_i p_MD(i, A; kτ)
p_MD(i, A; kτ) = ( Σ_{j∈A} c^obs_ij(kτ) ) / ( Σ_{j=1}^{n} c^obs_ij(kτ) )
```

Model-side observable, Eq. (64):

```
p_MSM(A, A; kτ) = Σ_{i∈A} [ (w^A)^T T^k(τ) ]_i
```

The statistic tested, Eq. (65), is `p_MD(A,A;kτ) = p_MSM(A,A;kτ)`.

**Error bars, Eq. (66)** — one-sigma standard error on the trajectory side, and
this is the exact form MSM practice uses:

```
ε_MD(A, A; kτ) = sqrt( [ k · p_MD(A,A;kτ) − p_MD(A,A;kτ)^2 ] / Σ_{i∈A} Σ_{j=1}^{n} c^obs_ij(kτ) )
```

The paper states that the uncertainty of `p_MSM` may also be computed by the
transition-matrix sampling of its Sec. IV E, but "this is not necessary if the
test already succe[eds]" within the trajectory-side error alone.

**At what n is CK conventionally checked.** Three explicit rules are given.
(i) The test is only sensitive "in ranges of `k` greater one and smaller than
the global relaxation time of the system", because `k = 1` is trivially exact
for the nonreversible maximum-likelihood estimator and `k ≫ t_2/τ` is always
good. (ii) "The test should be done for all times `kτ` for which trajectory data
are available", and a test comparing only `τ` against `2τ` is called "likely to
be unreliable". (iii) The worked figure reports the total error as the 2-norm of
`p_MD − p_MSM` over `kτ ∈ [1, 128]`. So the conventional check is a decade or
two of `k`, not a single multiple.

**Load-bearing caveat for the project's model-selection step.** The paper states
outright that "observing convergence of the slowest implied timescales in τ is
not a test of Markovianity. While Markovian dynamics implies constancy of
implied timescales in τ, the reverse is not true and would require the
eigenvectors to be constant as well."

**Delta.** Prinz et al. select `m` and `τ` for a partition of a molecular
configuration space and validate with CK on set-probabilities carrying Eq. (66)
error bars; CEQ intends to select `m` from the spectral gap of a lag-τ transition
matrix built over per-position readings `r_t` of text and to use CK as an
admissibility gate on a learned state head; the difference is that the object
being discretised is a learned representation whose lag-τ statistics are a
property of the model, not of a physical process with a fixed generator, so the
CK gate tests self-consistency of the head rather than Markovianity of a
substrate — and Prinz's own warning shows the project must gate on CK, never on
ITS flatness.

### 1.2 Weber (2018), the χ = XA exposition of PCCA+ `[V]`

*"Implications of PCCA+ in Molecular Simulation."* *Computation* **6**(1), 20.
DOI `10.3390/computation6010020`.

**The membership construction, in three steps quoted from the text:**

```
χ = X A
```

1. Find an `n`-dimensional invariant subspace `X` of the transfer operator
   (real eigenspaces or real Schur spaces).
2. "linearly transform the basis of this invariant subspace `X` in such a way,
   that all basis vectors form a non-negative partition of unity", i.e. find
   `A ∈ R^{n×n}` with `χ = XA`, `χ ≥ 0`, rows of `χ` summing to one.
3. Build the projected transfer operator from `χ`.

**The simplex-vertex step, quoted:** given `X ∈ R^{m×n}` of the `n ≪ m` leading
eigenvectors of the `m × m` transition matrix `P`, "the `m` rows of `X` — plotted
as `n`-dimensional points — seem to always form an `(n−1)`-simplex"; the points
at the `n` vertices "represent the thermodynamically stable conformations" and
the points on the edges "represent transition regions". The **inner simplex
algorithm (ISA)** "identifies those `n` points out of the `m` rows of `X` which
are supposed to be the vertices", and `A` maps the found simplex `σ1` onto the
standard simplex `σ0` spanned by the unit vectors.

**The objective.** PCCA+ is posed as an unconstrained maximisation `I[g(A)]`
where `I[A]` "accounts for the 'crispness' of `χ`" — memberships as close as
possible to characteristic functions — and "the optimum of a convex maximization
problem on a linearly bounded compact set is attained at a vertex of `F`". Many
local maxima; local optimisers give no global guarantee.

**Delta.** PCCA+ transforms the dominant eigenvectors of an *estimated* `P` by a
single linear map `A` chosen for crispness, with the simplex geometry supplying
both the vertex identification and the partition-of-unity constraint; CEQ's state
head is a trained nonlinear map `r_t ↦ χ_t ∈ Δ^{m−1}` whose simplex output is
imposed by a softmax rather than derived from eigenvector geometry; the
difference is that CEQ has no ISA step and therefore no guarantee that its
vertices are the metastable representatives — the geometric content PCCA+ buys
with `χ = XA` must be re-earned by CEQ some other way, or conceded.

### 1.3 Deuflhard and Weber (2005), the PCCA+ original `[V-t]`

*"Robust Perron cluster analysis in conformation dynamics."*
*Linear Algebra and its Applications* **398**, 161–184.
DOI `10.1016/j.laa.2004.10.026`. Crossref record read; no equation read.

**Delta.** Owed as the primary source for the ISA construction restated in 1.2;
the delta is identical to 1.2 and must be re-stated against this text once read.

### 1.4 Röblitz and Weber (2013) `[V-t]`

*"Fuzzy spectral clustering by PCCA+: application to Markov state models and
data classification."* *Adv. Data Anal. Classif.* **7**(2), 147–179.
DOI `10.1007/s11634-013-0134-6`. Crossref record read; no equation read.

### 1.5 Mardt, Pasquali, Wu, Noé (2018), VAMPnets `[V]`

arXiv:1710.06012; *Nat. Commun.* **9**, 5. DOI `10.1038/s41467-017-02388-1`.

**Score maximised:**

```
R̂_2[χ_0, χ_1] = || C_00^{-1/2} C_01 C_11^{-1/2} ||_F^2
```

**Koopman matrix from network outputs:** `K = C_00^{-1} C_01`, covariances taken
over the transformed outputs `χ_0`, `χ_1`.

**Output layer:** softmax output nodes, and the network "effectively performs
featurization, dimension reduction and finally a fuzzy clustering to metastable
states", with "the activation of an output node ... interpreted as a probability
to be in state `i`".

**Validation:** `K(nτ) ≈ K^n(τ)`, i.e. CK at the level of the learned Koopman
matrix.

**State count:** "To resolve `k−1` relaxation timescales, at least `k` output
nodes or metastable states are required."

**Delta — this is the closest published object to Q1 and the delta is thin.**
VAMPnets already learn a neural map from a raw configuration to a softmax
membership over `m` metastable states, already validate with CK, and already tie
`m` to the number of resolved timescales; CEQ's state head does the same three
things over a per-position reading `r_t` of text; the difference is only the
substrate and the downstream use — VAMPnets stop at the membership and the
Koopman matrix, CEQ feeds `χ_t` to a fan-out detector and an exit head. **Q1
alone is not novel. Any sentence claiming a novel "state head" must be deleted
and replaced by a citation to VAMPnets plus a statement of what is composed on
top.**

---

## 2. Energy-based models — LeCun lineage

### 2.1 LeCun, Chopra, Hadsell, Ranzato, Huang (2006) `[V]`

*"A Tutorial on Energy-Based Learning."* In *Predicting Structured Data*, MIT
Press, 2006. Read from `cs.nyu.edu/~sumit/publications/assets/ebmtutorial.pdf`.

**Inference, Eq. (1):**

```
Y* = argmin_{Y ∈ Y} E(Y, X)
```

**Gibbs distribution, Eq. (2):**

```
P(Y|X) = e^{−β E(Y,X)} / ∫_{y ∈ Y} e^{−β E(y,X)}
```

with `β` "an arbitrary positive constant akin to an inverse temperature", and
the denominator named the partition function.

**Training without a normaliser.** The family is `E = { E(W,Y,X) : W ∈ W }`,
Eq. (3). "Since EBMs have no requirement for proper normalization, this problem
is naturally circumvented"; the transformation of energies into probabilities is
"only possible if the integral ... converges", and computing the partition
function is "intractable ... or outright impossible", so "probabilistic modeling
comes with a high price, and should be avoided when the application does not
require it."

**The contrastive shape.** Definition 1, Eq. (8), the most offending incorrect
answer:

```
Ȳ^i = argmin_{Y ∈ Y, Y ≠ Y^i} E(W, Y, X^i)
```

and the generalised margin loss, Eq. (10):

```
L_margin(W, Y^i, X^i) = Q_m( E(W, Y^i, X^i), E(W, Ȳ^i, X^i) )
```

with training described as "pushing down" on `E(W,Y^i,X^i)` and "pulling up" on
the incorrect energies.

**Delta.** LeCun's EBM assigns an energy to a *configuration* `(Y, X)` and trains
by pushing the observed configuration's energy below its competitors'; CEQ's Q3
head assigns an energy *difference* `ΔÊ‡_a` to a transition between two
configurations and trains it against a barrier target; the difference is that
CEQ's scalar is a saddle height relative to a basin floor, not a compatibility
score, so the pull-up/push-down contrastive machinery does not transfer without
a supervision signal for `ΔÊ‡` that the EBM lineage does not supply.

### 2.2 Grathwohl, Wang, Jacobsen, Duvenaud, Norouzi, Swersky (2019), JEM `[V]`

arXiv:1912.03263, *"Your Classifier is Secretly an Energy Based Model and You
Should Treat It Like One."*

```
E_θ(x, y) = −f_θ(x)[y]
E_θ(x)    = −LogSumExp_y ( f_θ(x)[y] )
p_θ(y|x)  = exp(f_θ(x)[y]) / Σ_{y'} exp(f_θ(x)[y'])
log p_θ(x,y) = log p_θ(x) + log p_θ(y|x)
∂ log p_θ(x) / ∂θ = E_{p_θ(x')}[ ∂E_θ(x')/∂θ ] − ∂E_θ(x)/∂θ      (SGLD estimator)
```

**Delta — this is the formal identity the project must concede.** JEM states
plainly that a softmax over `K` logits *is* a Boltzmann distribution over `K`
energies with `E = −logit`; CEQ's `p(a) = softmax_a(−ΔÊ‡_a / T_temp)` is
therefore not a new kind of head, it is a softmax whose logits have been renamed
`−ΔÊ‡_a / T_temp`; the difference is zero at the level of the functional form,
and whatever novelty exists lies entirely in what `ΔÊ‡_a` is *supervised
against*, not in the softmax.

### 2.3 Bengio, Bengio, Jain, et al. (2021), GFlowNet `[V]`

arXiv:2106.04399, *"Flow Network based Generative Models for Non-Iterative
Diverse Candidate Generation."*

Flow-matching condition at a state `s'`:

```
Σ_{s,a : T(s,a) = s'} F(s,a) = R(s') + Σ_{a' ∈ A(s')} F(s', a')
```

Policy:

```
π(a|s) = F(s,a) / F(s)
```

**Does any EBM score transitions rather than states?** Yes — this one. Flows are
assigned to *edges* `F(s,a)`, not to states, and the sampling policy is the
normalised edge flow out of the current state. The terminal reward `R(x)` is an
arbitrary positive scalar in this paper, with `R(x)^β` used to tune selectivity;
the exponential-of-negative-energy reading is not written here.

**Delta.** GFlowNet scores transitions by a *flow* that must balance in and out
at every node and normalises it into a policy; CEQ scores exits by a *barrier
height* and normalises `exp(−ΔÊ‡_a/T)`; the difference is the conservation law —
GFlowNet's edge scores are constrained globally by flow matching so that terminal
sampling is proportional to reward, while CEQ's exit scores are local to one
basin and constrained by nothing, which is a weaker object and must not be
described as a flow.

### 2.4 Shao, Li, Meng, Zhou (2025), CALM `[V]`

arXiv:2510.27688, *"Continuous Autoregressive Language Models."*

```
h_{i−1} = Transformer(z_{1:i−1}),      z_i ∼ p( · | h_{i−1} )                    (7)
S(P, y) = E_{x',x'' ∼ P} [ ||x' − x''||^α ] − 2 E_{x ∼ P} [ ||x − y||^α ]        (9)
```

**Delta, and a name collision worth naming.** CALM's "energy-based generative
head" is a head trained with the *energy score* of Székely — a strictly proper
scoring rule over continuous vectors, `α ∈ (0,2)`, likelihood-free — and it emits
a continuous latent `z_i`, not a choice among discrete exits; CEQ's energy is a
physical barrier over an enumerated discrete exit set; the difference is total,
and the word "energy" is doing unrelated work in the two papers, so any CEQ text
adjacent to CALM must say which energy it means.

### 2.5 Blondel, Sander, Vivier-Ardisson, Liu, Roulet (2025) `[V-t]`

arXiv:2512.15605, *"Autoregressive Language Models are Secretly Energy-Based
Models: Insights into the Lookahead Capabilities of Next-Token Prediction."*
Abstract page read; no equation read. Claims "an explicit bijection between ARMs
and EBMs in function space" corresponding to a special case of the soft Bellman
equation in maximum-entropy RL.

**Delta.** Owed at equation level. If the bijection is what the abstract says,
then a barrier-scored softmax over exits is a soft-Bellman backup over exits with
`−ΔÊ‡` in the role of the soft value, and the RL delta in §5 tightens further.
This entry is the single highest-value remaining fetch.

---

## 3. Committor learning

### 3.1 Grinstead and Snell, *Introduction to Probability*, Ch. 11 `[V]`

American Mathematical Society, 2nd revised edition, 1997. Chapter 11, "Markov
Chains", §11.2 "Absorbing Markov Chains".

Canonical form of an absorbing chain with `t` transient and `r` absorbing states:

```
P = [ Q  R ]
    [ 0  I ]
```

**Definition 11.3.** "For an absorbing Markov chain `P`, the matrix
`N = (I − Q)^{-1}` is called the fundamental matrix for `P`."

**Theorem 11.6.** "Let `b_ij` be the probability that an absorbing chain will be
absorbed in the absorbing state `s_j` if it starts in the transient state `s_i`.
Let `B` be the matrix with entries `b_ij`. Then `B` is a `t`-by-`r` matrix, and

```
B = N R
```

where `N` is the fundamental matrix and `R` is as in the canonical form." The
proof given is `B_ij = Σ_n Σ_k q^{(n)}_{ik} r_{kj} = Σ_k n_{ik} r_{kj} = (NR)_ij`,
i.e. a Neumann series in `Q` resummed to `(I − Q)^{-1}`.

**Delta — and this one is close to zero, which is the point of reporting it.**
The project's novelty spine says the committor is the absorbing resolvent
`(I − P_II)^{-1} P_IB`, and that this is the same operator family as
`Σ_t γ^t P^t = (I − γP)^{-1}`; Theorem 11.6 *is* that statement, at `γ = 1` on
the sub-stochastic block, in an undergraduate textbook. **The difference is
zero.** What remains available to the project is not the identity but the
*use*: adopting `B = NR` as a supervision target for a text model whose forward
pass already computes a resolvent. Every sentence in the project that presents
the identification itself as new must be rewritten to present it as exposition.

### 3.2 Khoo, Lu, Ying (2018/2019) `[V]`

arXiv:1802.10275; *Res. Math. Sci.* **6**, 1.
*"Solving for high-dimensional committor functions using artificial neural
networks."*

**Boundary value problem:**

```
0 = −β^{-1} Δq(x) + ∇U(x) · ∇q(x)   in Ω \ (A ∪ B),
q(x)|_{∂A} = 0,   q(x)|_{∂B} = 1
```

**Variational loss, boundary conditions as soft constraints with hardness `ρ`:**

```
argmin_{θ ∈ R^p} E_ν ( |∇q_θ(x)|^2 χ_{Ω\A∪B}(x)
                        + (ρ/α) q_θ(x)^2 χ_{∂A}(x)
                        + (ρ/α) (q_θ(x) − 1)^2 χ_{∂B}(x) )
```

Harmonic-function language is present but only through Laplace's equation and
Green's functions in the high-temperature limit; **no harmonic measure, no
resolvent.**

**Delta.** Khoo–Lu–Ying minimise a Dirichlet energy against a known potential
`U` with the boundary sets given as regions of configuration space; CEQ has no
`U`, no `∇`, and its `A`/`B` are basins discovered by the state head; the
difference is that CEQ's committor problem is on a learned discrete graph, where
the Dirichlet form is a quadratic form in the graph Laplacian and the minimiser
is the linear solve of §3.1 rather than a variational neural fit — so this loss
is not the one CEQ should import.

### 3.3 Thiede, Giannakis, Dinner, Weare (2019) `[V]`

arXiv:1810.01841; *J. Chem. Phys.* **150**, 244111.
*"Galerkin approximation of dynamical quantities using trajectory data."*

```
q₊(x) = P[ τ_B < τ_A | Ξ^{(0)} = x ]
L q₊(x) = 0   for x ∈ (A ∪ B)^c,    q₊|_A = 0,   q₊|_B = 1
Σ_{j=1}^{M} L_ij a_j = h_i − r_i ,   L_ij = ⟨φ_i, L φ_j⟩,  h_i = ⟨φ_i, h⟩,  r_i = ⟨φ_i, L r⟩
```

The Neumann-series language appears, but for the integrated autocorrelation time
via "the Neumann series representation of the appropriate pseudo-inverse of `L`",
**not** for the committor, and no `(I − P)^{-1}` is written. No harmonic-measure
sentence.

**Delta.** Thiede et al. solve a Galerkin projection of `Lq = 0` in a fixed basis
`{φ_i}` chosen from trajectory data; CEQ would solve the same equation on the
graph induced by its own state head with the resolvent it already computes; the
difference is that the basis is learned end to end rather than fixed, which
removes the projection error Thiede et al. control and replaces it with a
representation-learning error nobody controls.

### 3.4 Strahan, Finkel, Dinner, Weare (2023) `[V]`

*"Predicting rare events using neural networks and short-trajectory data."*
*J. Comput. Phys.* (2023); read via PMC10270692.

**Stopped transition operator, Eq. (5), and the committor equation, Eq. (4):**

```
T^{D^c}_τ [f](x) = E_x[ f(X_{τ ∧ T}) ],       T = inf{ t > 0 : X_t ∈ A ∪ B }
( T^{D^c}_τ − I )[q](x) = 0 ,    q(x) = 0 for x ∈ A,   q(x) = 1 for x ∈ B
```

**Loss — Feynman–Kac regression, Eqs. (9)–(10):**

```
θ* = argmin_θ [ C_FKE + λ C_BC ]
C_FKE = || ( (T^{D^c}_τ − I) u_θ + E_x[ ∫_0^{τ ∧ T} h(X_s) ds ] ) 1_D ||²_μ
C_BC  = || ( u_θ − g ) 1_{D^c} ||²_μ
```

**No `(I − T)^{-1}`, no Neumann series, no harmonic-measure sentence.** The
operator equation is used directly; it is never inverted.

**Delta — and this is the sharpest available statement of what is actually
unclaimed.** Strahan et al. write the stopped-operator equation
`(T^{D^c}_τ − I) q = 0` and then *avoid* inverting it, regressing on the residual
instead; CEQ proposes to invert it, because the inverse is the operator its
forward pass already computes; the difference is that CEQ can afford the solve
and they cannot, which is a computational claim about the architecture, not a
mathematical claim about the committor.

### 3.5 Pigeon, Stoltz, Lelièvre (2026) `[V]`

arXiv:2607.21425, *"Approximating committor functions: Objective functions and
enhanced sampling."*

Boundary value problem with the overdamped Langevin generator
`L_ovd = −∇_q V · ∇_q + β^{-1} Δ_q`:

```
L_ovd χ(q) = 0  ∀ q ∈ (R̄ ∪ P̄)^c ,   χ|_R̄ = 0 ,   χ|_P̄ = 1
```

Objectives catalogued, each read from this text:

```
(1) variational, generator:   inf_f ∫_{(R̄∪P̄)^c} |∇f(q)|² e^{−βV(q)} dq
(2) residual, generator:      inf_θ ∫ |L_ovd f_θ(q)|² μ(dq)
(3) residual + soft BC:       inf_θ { ∫ |L_ovd g_θ|² μ + α ( ∫_R̄ g_θ² + ∫_P̄ (g_θ−1)² ) }
(4) residual, transfer op.:   inf_f ∫ [ (Id − P^i_ϑ) f(q) − (P^b_ϑ 1_P)(q) ]² μ(dq)
(5) log-residual:             inf_f ∫ [ ln f(q) − ln( P^i_ϑ f(q) + P^b_ϑ 1_P(q) ) ]² μ(dq)
(6) variational, transfer op: inf_f { ½ ∫ f (Id − P^i_ϑ) f e^{−βV} dq − ∫ f P^b_ϑ 1_P e^{−βV} dq }
(7) time-lag (Roux):          inf_f lim_n (T−ϑ)^{-1} ∫_0^{T−ϑ} ( f(q_{t+ϑ}) − f(q_t) )² dt
(8) Itô-based (this work):    a stochastic-integral residual, quoted in full in the source
```

The discrete-time fixed point is written `f(q) = Q_ϑ f(q)` on `(R̄ ∪ P̄)^c`, and
the text states the committor is **harmonic with respect to the infinitesimal
generator**, and connects it to an **absorbing Markov chain** where trajectories
terminate on reaching `R` or `P`.

**Does the identification already exist in print?** Partly, and the project must
say so. "Harmonic" and "absorbing Markov chain" are both stated here; objective
(6) is `½ f^T (I − P^i) f − f^T b`, whose stationary point is exactly
`(I − P^i)^{-1} b` — the absorbing resolvent — even though the inverse is never
written. The phrase *harmonic measure* is not used, and no paper found in this
session writes `q = (I − P_II)^{-1} P_IB` as such. **The novelty sentence is
therefore weaker than it thinks: three of the four names are in print together
in this 2026 review, and the fourth is Grinstead–Snell Theorem 11.6. The
sentence survives only as a bridge, never as a discovery.**

**Delta.** Pigeon et al. catalogue eight objectives for fitting a committor to a
known Langevin generator; CEQ proposes no objective at all, because it intends
to *solve* rather than fit; the difference is that CEQ's `L` is a learned
transition matrix whose entries are outputs of the model, so the solve is
differentiable and the committor is a function of the parameters, which none of
the eight objectives assume.

### 3.6 Roux (2022) `[V]`

*"Transition rate theory, spectral analysis, and reactive paths."*
*J. Chem. Phys.* (2022). DOI `10.1063/5.0084209`.

```
q(z) = ∫ dz′ q(z′) P_τ(z′|z),      q = 0 on A,   q = 1 on B                     (22)
∂/∂z [ e^{−W(z)/k_B T} ∂q(z)/∂z ] = 0                                            (36)
q(z) ≈ [ −ab/(b−a) ] ψ_1^L(z) + [ 1/(b−a) ] ψ_2^L(z)                             (30)
```

Eq. (30) is the committor written in the leading left eigenvectors, with "the
second eigenmode `ψ_2^R` ... represent[ing] the global transfer of probability
between metastable basins `A` and `B`". No harmonic-measure sentence, no
resolvent.

**Delta.** Roux links the committor to the *same* dominant spectrum that PCCA+
uses for the state decomposition, showing Q1 and the Q3 supervision target are
two readings of one eigenproblem; CEQ asserts the same link but proposes to
learn both ends; the difference is zero at the level of the mathematical
relation and lies only in the estimator, so the project may cite Roux for the
relation and must not claim it.

### 3.7 Li, Lin, Ren (2019) `[V-t]`

*"Computing committor functions for the study of rare events using deep
learning."* *J. Chem. Phys.* **151**, 054112. DOI `10.1063/1.5110439`. Crossref
record and abstract read; no equation read.

---

## 4. Saddle search — dimer, NEB, and learned saddle finders

### 4.1 Henkelman and Jónsson (1999), the dimer method `[V]`

*J. Chem. Phys.* **111**, 7010–7022. DOI `10.1063/1.480097`.

Dimer images about midpoint `R` with separation `ΔR` along unit vector `N̂`,
Eq. (1):

```
R_1 = R + ΔR N̂ ,     R_2 = R − ΔR N̂
```

**Curvature along the dimer, Eq. (2):**

```
C = ( F_2 − F_1 ) · N̂ / (2 ΔR) = ( E − 2 E_0 ) / (ΔR)²
```

with `E = E_1 + E_2`, and the midpoint energy recovered from forces alone,
Eq. (3):

```
E_0 = E/2 + (ΔR/4) ( F_1 − F_2 ) · N̂
```

**Rotation.** Rotational force `F^⊥ = F_1^⊥ − F_2^⊥` where
`F_i^⊥ ≡ F_i − (F_i · N̂) N̂`; scalar rotational force `F = F^⊥ · Q̂ / ΔR` in the
rotation plane spanned by `N̂` and `Q̂`; the analytic rotation angle, Eq. (13):

```
Δθ = θ_0 = −½ arctan( 2 F_0 / F_0′ )
```

**Translation — the update rule owed.** The effective force inverts the component
along the dimer, Eq. (18), and is switched on curvature sign, Eq. (21):

```
F† = F_R − 2 F^∥                         (18),  F^∥ = (F_R · N̂) N̂
F† = −F^∥                if C > 0
F† = F_R − 2 F^∥         if C < 0        (21)
```

so that in convex regions the dimer climbs the lowest mode until the curvature
turns negative. No Hessian is ever formed; only first derivatives are used.

### 4.2 Henkelman, Uberuaga, Jónsson (2000), climbing-image NEB `[V]`

*J. Chem. Phys.* **113**, 9901–9904. DOI `10.1063/1.1329672`.

**The harmonic-TST rate, Eq. (1)** — the exponential that Q3's softmax is:

```
k^hTST = [ Π_{i=1}^{3N} ν_i^init / Π_{i=1}^{3N−1} ν_i^‡ ] · e^{ −(E^‡ − E_init)/k_B T }
```

**NEB force projection, Eqs. (2)–(4):**

```
F_i = F_i^{s,∥} − ∇E(R_i)|_⊥                                                     (2)
∇E(R_i)|_⊥ = ∇E(R_i) − ( ∇E(R_i) · τ̂_i ) τ̂_i                                    (3)
F_i^{s,∥} = k ( |R_{i+1} − R_i| − |R_i − R_{i−1}| ) τ̂_i                          (4)
```

**Climbing image, Eq. (5)** — full potential force with the band-parallel
component inverted:

```
F_{imax} = −∇E(R_{imax}) + 2 ( ∇E(R_{imax}) · τ̂_{imax} ) τ̂_{imax}
```

The climbing image feels no spring force at all.

**Delta for 4.1 and 4.2 together.** Dimer and NEB *find* an index-1 saddle by
first-derivative optimisation on a real potential energy surface — dimer without
knowing the final state, NEB given both endpoints; CEQ does not search for
saddles at all, it is *given* candidate exits by the corpus/graph and only
regresses their heights `ΔÊ‡_a`; the difference is that CEQ has no potential to
differentiate and therefore cannot verify that a candidate exit is an index-1
saddle rather than an arbitrary pair of basins — the verification step that
Henkelman's method gets for free (quench on both sides of the saddle and confirm
the MEP connects back) has no CEQ analogue, and its absence is a hole in Q3.

### 4.3 Henkelman and Jónsson (2001), adaptive kinetic Monte Carlo — **the kill-clause entry** `[V]`

*"Long time scale kinetic Monte Carlo simulations without lattice approximation
and predefined event table."* *J. Chem. Phys.* **115**, 9657–9666.
DOI `10.1063/1.1415500`.

Same rate expression, Eq. (1), as 4.2. Then, verbatim from §II B:

> "If a list of possible transitions for a given initial state is available, a
> random number can be used to choose one of the processes and evolve the system
> to a new state. **The probability of choosing a certain transition is
> proportional to its rate, `r_i`.**"

```
Δt = 1 / Σ r_i                                                                   (2)
Δt = −ln μ / Σ r_i ,      μ ~ Uniform(0,1)                                       (3)
```

And §II C, the composition: "For each state of the system, characterized by a
local minimum on the potential energy surface, multiple searches for saddle
points are carried out using random initial directions. The dimer method is used
for the saddle point searches and the rate for each transition mechanism is
estimated using harmonic transition state theory. Transitions are selected and
the clock advanced according to the kinetic Monte Carlo algorithm." Exits are
enumerated **per visited state** (25 or 50 dimer searches per basin in the
reported runs), each candidate is quenched on both sides to verify it lies on a
minimum energy path from the current minimum, and unverified saddles "are
discarded from the list of possible transitions".

**Why this is the kill-clause entry.** `p(a) ∝ r_a = ν_a exp(−ΔE‡_a / k_B T)` is
`softmax_a( −ΔE‡_a / k_B T )` up to the prefactor `ν_a`. So the composite
"detect the current metastable state → enumerate its discrete exits → score each
by a barrier → softmax over the barriers → condition what happens next on the
chosen exit" is published, in full, in 2001, and it is not even the paper's own
novelty (kMC selection is cited there to Bortz–Kalos–Lebowitz and successors).

**Delta, stated without softening.** Adaptive kMC scores exits with a *computed*
`ΔE‡` from a real potential and conditions the *next physical state* on the
chosen exit; CEQ scores exits with a *learned* `ΔÊ‡` from text and conditions
*token emission* on the chosen exit, `p(token | context, chosen a)`; the
difference is (i) the barrier is regressed rather than computed, so there is no
saddle verification step, and (ii) the consequent is an autoregressive decoder
rather than a state update. **That is the entire delta of Q3. It is a
substrate-and-consequent delta, not a mechanism delta, and the claim must be
written that way or it is false.**

### 4.4 Karwounopoulos, De Landsheere, Galustian, Jechtl, Heid (2025) `[V]`

*"Graph-based prediction of reaction barrier heights with on-the-fly prediction
of transition states."* *Digital Discovery* (2025). DOI `10.1039/d5dd00240k`.

Directed message-passing neural network over a condensed graph of reaction;
hidden directed edge features updated by message passing over `T` steps;
transition-state geometries generated on the fly per reaction by TSDiff
(diffusion) or GoFlow (flow matching), converted to 3D descriptors (MACE,
Equiformer V2) and concatenated with the 2D graph features; the predicted
quantity is the barrier height in kcal/mol.

**Delta — the closest published object to the Q3 *regressor*.** This is a learned
`ΔÊ‡` predictor over enumerated candidate transitions, generating the saddle on
the fly, exactly the role CEQ's barrier head plays; the difference is that the
exit set here is a chemically well-posed reaction with a ground-truth DFT barrier
for supervision, whereas CEQ has no ground-truth barrier for a text transition
and must manufacture one — **so the open question for Q3 is not architecture, it
is what supervises `ΔÊ‡`, and no fetched work answers it.**

---

## 5. The nearest RL neighbour — options, HRL, decision transformers, world models

### 5.1 Sutton, Precup, Singh (1999), the options framework `[V]`

*"Between MDPs and semi-MDPs: A framework for temporal abstraction in
reinforcement learning."* *Artificial Intelligence* **112**(1–2), 181–211.
DOI `10.1016/S0004-3702(99)00052-1`.

**The triple, §2, verbatim:** "Options consist of three components: a policy
`π : S × A → [0,1]`, a termination condition `β : S⁺ → [0,1]`, and an initiation
set `I ⊆ S`. An option `⟨I, π, β⟩` is available in state `s_t` if and only if
`s_t ∈ I`." A Markov option continues from `s_{t+1}` with probability
`1 − β(s_{t+1})`. "When the option terminates, the agent has the opportunity to
select another option."

**Multi-time model, Eqs. (6)–(7):**

```
r^o_s      = E{ r_{t+1} + γ r_{t+2} + ··· + γ^{k−1} r_{t+k} | E(o,s,t) }         (6)
p^o_{ss′}  = Σ_{k=1}^{∞} p(s′, k) γ^k                                            (7)
```

**Bellman equations over options, Eqs. (8)–(10):**

```
V^μ(s)    = Σ_{o ∈ O_s} μ(s,o) [ r^o_s + Σ_{s′} p^o_{ss′} V^μ(s′) ]              (8)
Q^μ(s,o)  = r^o_s + Σ_{s′} p^o_{ss′} Σ_{o′ ∈ O_{s′}} μ(s′,o′) Q^μ(s′,o′)         (9)
V*_O(s)   = max_{o ∈ O_s} [ r^o_s + Σ_{s′} p^o_{ss′} V*_O(s′) ]                  (10)
```

**Selection at a decision point.** The framework fixes only that a policy over
options `μ(s,o)` is applied when the previous option terminates; Eq. (10) uses
`max`, and the paper prescribes no particular exploration rule.

**Three delta sentences, as owed, and none of them is comfortable.**

*Initiation set vs. fan-out detector.* An option's `I ⊆ S` is a fixed,
hand-specified or learned *set membership* test on the current state, evaluated
before the option may be taken; CEQ's fan-out detector is a threshold on a
predictive statistic, `H_t − H̄_basin > τ_H`, evaluated on next-token entropy
relative to a per-basin baseline; the difference is that CEQ's gate is a property
of the *predictive distribution* rather than of the state, so it fires on model
uncertainty and can fire in states an option-style `I` would admit
unconditionally — this is a real difference and it is the strongest single
distinction the project holds against options.

*Termination condition vs. saddle crossing.* An option's `β(s) ∈ [0,1]` is a
learned or specified stochastic stopping probability with no geometric content;
a saddle crossing is a specific event — passage through an index-1 point between
two basins — defined by the landscape, not by a free function; the difference is
that `β` is a *parameter* and the crossing is a *consequence*, so CEQ constrains
what options leaves free. But CEQ inherits the burden with it: nothing in the
architecture verifies that the crossing is index-1 (see §4.2 delta), so in
implementation `ΔÊ‡` risks degenerating into an unconstrained `β`-like free
function, at which point the difference collapses.

*Option-value selection vs. barrier-scored softmax.* Eq. (10) selects by
`max_o [ r^o_s + Σ p^o_{ss'} V*_O(s') ]`, a value; CEQ selects by
`softmax_a(−ΔÊ‡_a / T)`, a Boltzmann distribution over barriers; by §2.2 the
softmax is a Boltzmann distribution over energies `E_a = ΔÊ‡_a`, so the two
differ only in **which scalar** is used, `−ΔÊ‡_a` versus `Q(s,o)`, and in max
versus soft-max. A Boltzmann policy over option values is standard practice.
**The difference here is close to zero and must be conceded in writing:** the
project's claim is that the barrier is a *different scalar with a physical
identity and its own supervision*, not that the selection rule is new.

**And the resolvent.** Eq. (7), `p^o_{ss′} = Σ_{k=1}^{∞} p(s′,k) γ^k`, is a
discounted sum over exit times — the same `Σ γ^k P^k` family the project names as
its proved propagation law. The options paper published a discounted-resolvent
transition model over temporally extended exits in 1999. Any CEQ sentence
implying that a discounted resolvent over multi-step transitions is unusual is
refuted by this equation.

### 5.2 Bacon, Harb, Precup (2017), option-critic `[V]`

arXiv:1609.05140; AAAI 2017.

```
A Markovian option ω ∈ Ω is a triple (I_ω, π_ω, β_ω)
Q_Ω(s, ω) = Σ_a π_{ω,θ}(a|s) Q_U(s, ω, a)                                        (1)
U(ω, s′)  = (1 − β_{ω,ϑ}(s′)) Q_Ω(s′, ω) + β_{ω,ϑ}(s′) V_Ω(s′)                   (3)
```

Intra-option policy gradient, Theorem 1:
`Σ_{s,ω} μ_Ω(s,ω | s_0, ω_0) Σ_a ( ∂π_{ω,θ}(a|s) / ∂θ ) Q_U(s,ω,a)`.
The architecture "does not prescribe how to obtain `π_Ω`"; the experiments use an
ε-greedy policy over options with `ε = 0.01`.

**Delta.** Option-critic learns `π_ω` and `β_ω` end to end by policy gradient
from reward; CEQ learns `χ_t` and `ΔÊ‡_a` from corpus statistics with no reward
signal at all; the difference is the training signal — option-critic needs an
MDP with rewards, CEQ needs only text — and that is a genuine difference, but it
is a difference of *supervision*, not of the head's structure.

### 5.3 Machado, Bellemare, Bowling (2017), eigenoptions `[V]`

arXiv:1703.00956, *"A Laplacian framework for option discovery in reinforcement
learning."*

```
r^e_i(s, s′) = e^T ( φ(s′) − φ(s) )     ;   tabular:  e[s′] − e[s]
M^e_i = ⟨ S, A ∪ {⊥}, r^e_i, p, γ ⟩
I = { s : ∃ a ∈ A with q*^e(s,a) > 0 }
terminate when  q^e_χ(s,a) ≤ 0  for all a ∈ A
π^e(s) = argmax_{a ∈ A ∪ {⊥}} q^e_π(s,a)
```

**Delta — the reinvented-options-with-chemistry-vocabulary test, run
honestly.** Eigenoptions derive `I` and the termination rule from the *eigen-
vectors of the graph Laplacian* of the transition structure, i.e. from exactly
the spectral decomposition CEQ uses to pick `m` and extract `χ_t`; a spectral
decomposition of a transition operator yielding discrete temporally extended
exits is therefore already published. The differences that survive: eigenoptions
build one option per eigenvector and select greedily by `argmax q`, while CEQ
builds a membership simplex over *all* dominant eigenvectors jointly (PCCA+
shape) and selects by a Boltzmann distribution over barrier heights; and
eigenoptions' termination is `q ≤ 0`, a value condition, not a landscape
condition. **The vocabulary charge is partly upheld: the spectral→discrete-exits
half of CEQ is eigenoptions with different words. What is not eigenoptions is the
barrier scalar and the entropy-triggered gate.**

### 5.4 Chen, Lu, Rajeswaran, Lee, Grosse, Laskin, Abbeel, Srinivas, Mordatch (2021), Decision Transformer `[V]`

arXiv:2106.01345.

```
τ = ( R̂_1, s_1, a_1, R̂_2, s_2, a_2, …, R̂_T, s_T, a_T ) ,    R̂_t = Σ_{t′=t}^{T} r_{t′}
loss (continuous actions) = mean( (a_pred − a)² )
```

At test time a target return is specified, actions are sampled autoregressively,
and the target return is decremented by the achieved reward each step.

**Delta.** Decision Transformer conditions autoregressive emission on a *scalar
return-to-go* supplied by the user and updated by arithmetic; CEQ conditions
autoregressive emission on a *discrete exit* sampled from a barrier-scored
softmax computed by the model; the difference is that the conditioning variable
is chosen by the model from an enumerated set with a physical score, not handed
in as a scalar knob — this is the clearest structural difference in the whole RL
lineage and is worth stating explicitly whenever "conditioned generation" is
claimed.

### 5.5 Mikhaylovskiy (2025), states of LLM-generated text `[V]`

arXiv:2503.06330, *"States of LLM-generated Texts and Phase Transitions between
them."*

```
C(τ) = (1/(N−τ)) Σ_{i=1}^{N−τ}  ( V_i · V_{i+τ} ) / ( ||V_i|| ||V_{i+τ}|| )
```

Three "states" — periodic, critical/amorphous, gas — separated by decoding
temperature around `T ≈ 0.7–0.8`, detected from autocorrelation decay of GloVe
embeddings and an FFT/GAPELMAPER statistic. **No transition matrix is estimated,
no metastability, no barrier, and generation is not conditioned on anything.**

**Delta.** This paper names phases of generated text by a decoding hyper-
parameter and stops; CEQ estimates a lag-τ transition matrix over per-position
readings and conditions emission on the result; the difference is that the
"states" here are properties of a sampling temperature, not basins of a dynamics,
so the paper is a naming collision rather than prior art, and is listed only so
that a reviewer's search for "LLM metastable state" resolves here and not into a
priority dispute.

---

## Owed — entries still `[U]`

1. **Blondel et al., arXiv:2512.15605** — read at abstract only (`[V-t]`, §2.5).
   The equation linking next-token logits to an energy, and the claimed soft-
   Bellman correspondence, are **the single highest-value unread item**: if the
   correspondence is exact, §5.1's third delta sentence shrinks further and the
   Q3 head becomes a soft-Bellman backup over exits. Owed at equation level.
2. **Bortz, Kalos, Lebowitz (1975), the n-fold way / BKL algorithm.** *J. Comput.
   Phys.* **17**, 10–18. The kMC selection rule quoted in §4.3 is cited there;
   the primitive form of `p(a) ∝ r_a` therefore predates 2001 and the primary
   source was not read this session. Identifier believed resolvable; unverified.
3. **E and Vanden-Eijnden (2006), "Towards a theory of transition paths",**
   *J. Stat. Phys.* **123**, 503; and **Metzner, Schütte, Vanden-Eijnden (2009),
   "Transition path theory for Markov jump processes",** *Multiscale Model.
   Simul.* **7**, 1192. Both fetch attempts failed (dead NYU link, 301 to a
   department landing page). These are the two places most likely to state the
   discrete committor and the reactive flux in the exact `Lq = 0` matrix form
   the project wants, and possibly the harmonic-measure sentence. Owed.
4. **Deuflhard and Weber (2005)** and **Röblitz and Weber (2013)** — §1.3, §1.4,
   bibliographic records only. The ISA vertex-selection step was read in Weber
   (2018)'s restatement, not in either original.
5. **Li, Lin, Ren (2019)**, §3.7 — abstract only; the loss form is unread.
6. **Rotskoff, Mitchell, Vanden-Eijnden (2022), active importance sampling for
   variational committor objectives** (PMLR/MSML). Not fetched; would add a
   sampling-side objective to the §3.5 catalogue.
7. **Du and Mordatch (2019), arXiv:1903.08689**, implicit generation with EBMs —
   not fetched. The Langevin-based contrastive estimator is represented in this
   file only through JEM (§2.2).
8. **Covering options / successor options** (Jinnai et al. 2019 and successors),
   which build options from the Fiedler vector of the graph Laplacian — not
   fetched. This is the most likely place for a *tighter* eigenoptions-style hit
   than §5.3, because Fiedler-vector options are explicitly bottleneck-seeking,
   and a bottleneck is a saddle. **Treat as an open kill-clause risk for Q1+Q3
   until read.**
9. **World models** (Ha and Schmidhuber 2018, arXiv:1803.10122; Dreamer line) —
   not fetched. Owed a delta sentence on latent-state rollout versus basin
   membership.
10. **A published Boltzmann-policy-over-options citation.** §5.1's third delta
    asserts that `softmax` over option values is standard practice; the two
    option papers read here use `argmax` (§5.3) and ε-greedy (§5.2). The claim
    that softmax-over-options is standard is therefore **unsupported by any
    source read this session** and must either be cited or dropped.
11. **What supervises `ΔÊ‡` for text.** §4.4 shows the learned-barrier regressor
    exists, with DFT ground truth. No source found in this session supplies or
    even discusses a ground-truth barrier for a transition between two states of
    a text corpus. This is not a citation gap; it is an unanswered design
    question, recorded here because the Tier-6 fetch is where it surfaced.

---

## Count

- `[V]` — 22 entries: §1.1, §1.2, §1.5, §2.1, §2.2, §2.3, §2.4, §3.1, §3.2,
  §3.3, §3.4, §3.5, §3.6, §4.1, §4.2, §4.3, §4.4, §5.1, §5.2, §5.3, §5.4, §5.5.
- `[V-t]` — 4 entries: §1.3, §1.4, §2.5, §3.7.
- `[U]` — 11 items, listed above.

**Kill clause: did not fire on the literal conjunction; fired on every clause
except autoregressive emission (§4.3).**
