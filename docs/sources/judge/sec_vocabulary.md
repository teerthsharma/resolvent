# Appendix B. Vocabulary — the field's name for each construction

Every object in this paper was built or reached inside the repository before it was
named. This appendix gives the standard term for each, the field the term comes from,
and what the term commits its user to, so that a sentence about this work can be read
by someone who has never opened the repository. It is a naming table, not a claim
table: a term appearing here says the construction belongs to a known family, never
that the family's results transfer to it. Where two fields use one word for different
objects, both are given, because that ambiguity has cost this campaign real numbers.

## B.1 The operator and its read

| the construction | the field's term | field of origin | what the term commits you to |
|---|---|---|---|
| $z=(I-\gamma P)^{-1}V$ | **resolvent** of $P$ at $1/\gamma$; equivalently the **Neumann series** $\sum_t (\gamma P)^t$ | operator theory, numerical linear algebra | convergence needs $\|\gamma P\|<1$ or nilpotency; the object is linear in $V$ |
| the same, for row-stochastic $P$ and a reward $V$ | **policy evaluation**; $(I-\gamma P)^{-1}$ is the **successor representation** | reinforcement learning (`bellman-1957-markovian`, `dayan-1993-successor`) | $\gamma$ is a discount; the entries are expected discounted visit counts, so they are non-negative and sum to $1/(1-\gamma)$ |
| the same on a graph adjacency | **personalised PageRank** / **Katz index** | network science (`katz-1953-status`, `brin-1998-anatomy`, `gasteiger-2019-appnp`) | teleport probability $1-\gamma$; the read is a centrality unless $V$ carries content |
| $\Pi_\gamma=(1-\gamma)P(I-\gamma P)^{-1}$ | the **discounted occupancy kernel**; a **Markov kernel** in its own right | Markov chains | it is row-stochastic, so the read is a convex mixture and cannot leave the value hull |
| solving $(I-\gamma P)z=V$ by substitution rather than inverting | **forward substitution** on a **triangular system** | numerical linear algebra | cost $\Theta(s^2 d)$, sequential depth $s$; the inverse is never formed |
| computing it in blocks | the **chunked** or **UT-transform / WY** form | linear-attention kernels (`yang-2024-deltanet`, `dao-2024-ssd`) | the intra-chunk term is a small dense triangular inverse; inter-chunk is a matmul |
| $\gamma$ swept from $0$ to $1$ | the **discount**, or the **horizon**; $\gamma\uparrow1$ is the **undiscounted limit** | RL, Markov chains | at $\gamma=1$ with an absorbing row the resolvent of the full matrix does not exist; the limit is taken on the transient block |

## B.2 The boundary rows and what they read

| the construction | the field's term | field of origin | what the term commits you to |
|---|---|---|---|
| a row set to $e_a$ | an **absorbing state**; the set is an **absorbing class**; imposing it is a **Dirichlet boundary condition** | Markov chains; potential theory | once entered, never left; the chain is then a **killed** or **absorbed** walk |
| the block form $\begin{pmatrix}Q&R\\0&I\end{pmatrix}$ | **canonical form** of an absorbing chain; $Q$ is the **transient block** | Markov chains (`kemeny-1960-finitemarkov`) | $\rho(Q)<1$ is *absorption almost surely*, and it is a hypothesis, not a given |
| $N=(I-Q)^{-1}$ | the **fundamental matrix** | Markov chains | $N_{ij}$ is the expected number of visits to $j$ from $i$ before absorption |
| $B=NR$, its column $k$ | the **absorption probability**; for two sets, the **committor** or **splitting probability**; in control, the **reach-avoid probability** | Markov chains; transition-path theory (`metzner-2009-tpt-markov-jump`); stochastic reachability (`summers-2010-reach-avoid`) | the columns sum to one across *all* absorbing sets, which is why a goal set must exist |
| the same quantity solved as $Lq=0$ with $q$ clamped on the boundary | the **discrete Dirichlet problem**; $q$ is a **harmonic function**; the machine-learning instance is **label propagation** | potential theory (`doyle-1984-electric`); semi-supervised learning (`zhu-2003-harmonic`) | harmonicity is on the *interior* only; the boundary values are data |
| the set $\{q=\tfrac12\}$ | the **isocommittor surface**; in chemistry, the **transition state** | transition-path theory (`e-2010-tptreview`) | it is a level set of a function, not a partition of the state space |
| a position that absorbs mass without being asked to | an **attention sink** (a *column* device) — distinct from an absorbing *row* | transformer analysis (`xiao-2023-attentionsinks`) | a sink receives weight; an absorbing row emits none. The two coexist and are not each other |

## B.3 Interventions and decisions

| the construction | the field's term | field of origin | what the term commits you to |
|---|---|---|---|
| rewriting a row of $P$ and re-solving | an **intervention**, $\mathrm{do}(a)$; on a linear system, **graph surgery** on a **structural causal model** | causal inference (`pearl-2009-causality`, `shimizu-2006-lingam`) | the model must be the data-generating one for the word "causal" to be earned; on a learned $P$ it is an intervention on the *model* |
| $\Delta z$ after that rewrite | the **total effect**; for an equilibrium, the **equilibrium displacement** | causal inference; comparative statics (`mooij-2013-ode2scm`, `bongers-2021-cyclic`) | it is defined relative to a stated pre-intervention state |
| updating the solve after a rank-one row change | the **Sherman–Morrison** formula; for several rows, **Woodbury**; on a Markov chain, **fundamental-matrix perturbation** | numerical linear algebra (`sherman-1950-inverse-adjustment`); Markov chains (`schweitzer-1968-perturbation`) | the update is exact, not an approximation; it costs one solve's worth of back-substitution |
| whether "intervene then settle" equals "settle then intervene" | **equilibration–manipulation commutability** | causal modelling of equilibria (`dash-2005-emc`) | it can fail; it holds here by triangularity, which is a property of the operator and not of the world |
| $\arg\max_a q^{(0)}(\mathrm{do}\,a)$ | a **safety filter** or **least-restrictive filter**; the value is a **reach-avoid value function** | safe control (`hsu-2023-safetyfilter`, `fisac-2019-bridging`) | it is a decision rule over *evaluated* candidates, not a learned policy |
| $\arg\min_a\max_k q^{(k)}$ | **Chebyshev scalarisation** of a multi-objective problem | multi-objective optimisation (`vanmoffaert-2013-chebyshev`) | the max-form and the weighted-sum form give different optima; the choice is a modelling decision and must be declared |
| ranking constraints instead of maxing them | **lexicographic** ordering; bounding each is a **constrained MDP** | multi-objective RL (`yang-2026-lexisafe`); CMDPs (`altman-1999-cmdp`) | Bellman's principle can fail on multichain safety-constrained problems (`misra-2023-safety-constrained-mdp`) |

## B.4 The gate, the segmentation and the topology

| the construction | the field's term | field of origin | what the term commits you to |
|---|---|---|---|
| a multiplicative gate $m_k$ hitting exactly zero | a **reset gate**; the segment it opens is a **chunk boundary** | gated linear attention (`yang-2023-gla`, `lin-2025-forgetting-transformer`) | implementations that carry the gate as $\log m$ cannot represent the exact zero; that is the repository's own theorem |
| the resulting zero cross-block | **block-diagonal** structure; the solve is **decoupled** | linear algebra | it is exact, not sparse-approximate; approximate masks fill in under the inverse |
| filtering an influence graph by a threshold and counting components | the **$\beta_0$ barcode** of a **filtration**; the construction is **persistent homology** | topological data analysis | stability constants for *directed* networks differ from the classical ones (`turner-2019-quasimetric-rips`) |
| covering a space by overlapping cells and taking the nerve | the **Mapper** graph; the object it estimates is a **Reeb graph** | topological data analysis (`singh-2007-mapper`, `carriere-2018-mapper-statistics`) | the cover's parameters must be fixed before the data, or the summary is chosen rather than measured |
| the record's own $S^2$ point clouds | a **Vietoris–Rips** complex; the parameter sweep crosses a **percolation transition** | computational topology | connectivity is $\beta_0$; nothing about $\beta_1$ follows from a $1$-skeleton |

## B.5 The instruments and the statistics

| the construction | the field's term | field of origin |
|---|---|---|
| the running evidence value that licenses stopping at any time | an **e-process**; the guarantee is **Ville's inequality**; the family is **anytime-valid inference** |
| declaring two arms equivalent rather than failing to distinguish them | **equivalence testing**, specifically **TOST** (`schuirmann-1987-tost`) |
| the smallest effect a design can see | the **minimum detectable effect**; its inputs are **power** and the **paired standard deviation** |
| an exact interval on a proportion | the **Clopper–Pearson** interval; the paired test on two arms is **McNemar's** |
| the floor below which no predictor can go | **Fano's inequality** (for a discrete answer); **rate–distortion** (for a continuous one) |
| a lower bound on error from the label's own entropy at zero information | the **zero-information** or **majority-class** floor |
| a bound with no sum over the sequence length | a **uniform** or **dimension-free** bound |

## B.6 Two words this campaign used in two senses, and what each cost

**"Causal."** In a transformer, *causal* names the triangular mask: position $i$ reads
only $j\le i$. In causal inference, *causal* names invariance under intervention. The
repository's operator is causal in the first sense throughout, and the second sense is
earned only on a bed where the label is generated by an intervention on a stated model.
Every sentence in this paper that uses the second sense names the bed. This is the
term-slide the campaign paid for most often.

**"State."** In linear algebra a state is a vector; in Markov-chain language a state is
a *position with a transition law*; in state-space models the state is the recurrence's
carried summary, whose *dimension* is a capacity bound. The repository's
"next state toward equilibrium" is the Markov sense; its `d_model` is the linear-algebra
sense; the delay theorems of `V15Kernel` are about the third. A sentence that slides
between them proves nothing, and the record has one such sentence per round.

**"Depth."** Depth in a transformer is the number of parameter blocks; depth in
circuit complexity is the length of the longest computation path. The shape has one
parameter block and a solve whose sequential depth is the sequence length. Both are
said, in the same paragraph, every time either is claimed.
