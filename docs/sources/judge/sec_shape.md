# 2. The shape

JUPITER (MYCROFT), judge of the theory, 2026-09-03, HEAD `207e7b9`. The paper's §2 as final
prose. Inputs: the brief, the notes, both corrections, the six `sections/`, the eight `sweep/`,
`references.bib` (canonical keys only), the three designs and all six refutations
(`refute_{theory,instrument,falsify}_{occvac,math}.md`; none is missing). Rule: every KEEP adopted,
every REPAIR applied in the refuter's wording where right, every KILL deleted unless answered with
a number; where refuters disagree the ruling carries its RUN. Evidence classes: `RUN[J]` — one
numpy float64 draw by this judge (seed 0, $s=32$, $\gamma=0.6$, causal softmax $P$ with the
diagonal, sink $\{0\}$, goal $\{5\}$, constraints $\{9,10\},\{15\}$; computed and printed only);
`RUN[coord]`/`RUN[M]`/`RUN[I]`/`RUN[F]` — the coordinator's, MARS's, SATURN's, VENUS's runs as
their files record them; `READ path:line`; `CITED [V]`/`[U]` by canonical key; `DERIVED` with
steps. Every proposition carries its `MISTAKES.md` mechanism and a status in
`{LEAN, CLASSICAL, DERIVED, RUN, OPEN}`. Nothing here is a result; the paper's body is the plan
(`THESIS_CORRECTIONS_2.md` §0) and this section is its load-bearing wall.

## 2.0 Occupancy — cited before the shape is named

The read operator on a causal row-stochastic softmax matrix, solved as a triangular system and
reverting to standard attention at $\gamma = 0$, is **ChaCAL** (`fagnou-2024-chacal`, EMNLP 2024,
Eq. 5, Eq. 7, Thm 1; CITED [V]; `0` files in the tree cite it — a V-7-on-search finding for §3).
One detail decides two controls below: ChaCAL removes the diagonal inside the inverse
(`refute_falsify_math.md` §3.15, `[V-fetched]`), so its resolvent is a finite regime-N sum and its
read is **sub-stochastic** — `RUN[J]` row sums `0.400 … 0.765` against `1.000` diagonal-kept. Its
blockwise evaluation is `zhao-2026-structuredsparse` [V]. The discounted Neumann closure of a
content-adaptive attention with learnable per-head $\gamma$, read as the fundamental matrix of an
absorbing chain with one leak state, is `roffo-2026-infsa` [V] (ReLU Frobenius base, non-causal).
Multi-hop PageRank on attention: `wang-2021-magna`, `feng-2022-diffuser`, `yuan-2025-paraformer`
(lineage `chien-2020-gprgnn`); the implicit resolvent step of an attention flow
`chamberlain-2021-grand`; the exact resolvent layer on a fixed graph `gasteiger-2019-appnp`;
$(I-\gamma P)^{-1}$ as policy evaluation, successor representation, Katz/PageRank index
`bellman-1957-markovian`, `dayan-1993-successor`, `katz-1953-status`, `brin-1998-anatomy`. The three
corners: `vaswani-2017-attention` ($\beta=1$), `katharopoulos-2020-linearattention` ($\beta=0$,
$g\equiv0$), `dao-2024-ssd` Def. 3.1 (the 1-semiseparable mask); the chunked triangular solve
`yang-2024-deltanet` Eq. 10; no finite-state dual for row-softmax `hu-2025-ssdtheory` [V].

Boundary rows: $N=(I-Q)^{-1}$, $B=NR$ — `kemeny-1960-finitemarkov` [V-cat] (the bib carries
the same book as `kemeny-1976-finite` and `kemeny-1976-finitemarkov`; the assembler collapses the
three) and `grinstead-1997-probability` Thm 11.6 [U]; clamped labelled nodes as absorbing boundary
with a harmonic read *as a learning layer* — `zhu-2003-harmonic` [V], `zhou-2003-consistency`,
`wu-2012-partially-absorbing`, `begga-2023-diffusion-jump`, `azad-2022-harmonic-extension`; the
discrete Dirichlet committor — `metzner-2009-tpt-markov-jump`, `e-2006-transition-paths`,
`e-2010-tptreview`, `doyle-1984-electric`; one absorbing row inside attention (`do()` with the
token fixed and forbidden from attending) — `karbalayghareh-2026-doformer` [V]; the attention
*sink*, a column device distinct from a row condition — `xiao-2023-attentionsinks`,
`gu-2024-sinkemerges`, `ranmilo-2026-attentionsinks`. Decision: reach-avoid probability —
`summers-2010-reach-avoid`, `abate-2008-reachability`, `kariotoglou-2014-lp-reach-avoid`; several
target sets in one linear system with a verdict — `baier-2008-modelchecking`; Chebyshev
`vanmoffaert-2013-chebyshev`, lexicographic `yang-2026-lexisafe`, max–min `park-2026-maxmin`, CMDP
`altman-1999-cmdp`, argmin over evaluated values `bellman-1957-markovian`, a move from a linear
first-exit solve `todorov-2006-lmdp` / `todorov-2009-efficient`, a transformer choosing the
lowest predicted violation probability `jeddi-2021-lyapunovsafe` (NEAR-MISS), safety filter
`hsu-2023-safetyfilter` / `borquez-2023-lrf`, the discount that buys a contraction
`fisac-2019-bridging` / `hsu-2021-reachavoidrl`. Intervention: row surgery on a linear SCM
`pearl-2009-causality`, `shimizu-2006-lingam`; consequence as equilibrium displacement
`mooij-2013-ode2scm`, `bongers-2021-cyclic`, first-order `bottou-2013-counterfactual` §7.3; rank-one
re-solve `sherman-1950-inverse-adjustment`, `hager-1989-updating`; fundamental-matrix perturbation
`schweitzer-1968-perturbation`; barrier re-solve on a fixed chain `piray-2021-linearrl` Eq. 5; the
SCM as the fixed point of a causally masked transformer with `do()` by clamp-and-re-solve
`scetbon-2024-fip` [V]; order of intervention and equilibration `dash-2005-emc`; oracle-scored
effect with a no-change floor `vakalis-2026-interventiongap`, `lin-2026-scratchworld`. Neumann
tail: textbook (`meyer-2000-matrix`, `horn-2013-matrix`). Segmentation mechanism:
`lin-2025-forgetting-transformer`, `yang-2023-gla`, `hwang-2025-hnet`; converse
`yang-2026-boundary-repair` Thm 1. Topology: thresholded attention barcodes
`kushnareva-2021-tda-attention`, `kushnareva-2022-betti`; intervention effects on persistence
`kim-2026-topological-causal`; directed stability `turner-2019-quasimetric-rips`,
`chowdhury-2017-path-homology`; Mapper `singh-2007-mapper` (records [V], landing [U]),
`carriere-2018-mapper-statistics`; cluster covers `cho-2022-sbm-attention`,
`roy-2021-routing-transformer`, `kitaev-2020-reformer`, `yuan-2025-nsa`; the persistence-derived
CSR schedule in an attention kernel — the author's own `sharma-2026-kernels-22` [V].

**The delta, stated narrowly.** No fetched source combines the following in one attention
operator, and each part is owned above (eight sweeps, queries recorded; absence is bounded by
those queries, V-7): (e) $K\ge2$ absorbing constraint sets, a goal set and a declared sink as
identity rows of a *content-dependent, causal, row-stochastic* read; (g) the interventional
re-solve with the displacement as a jointly scored vector label at two stated prices; (h) the
committor vector and the safest-move rules over candidate moves placed in the context; (i) a
printed certificate with a planted negative and the $1/(1-\gamma)$ mask amplification; (j) a
learnable discount in a causal LM read pinned by a boundary-corrected LR test; (k) machine-checked
containment and the zero-gate iff; (l) restricted to the F0 endpoint; (m) the cover-to-schedule
path. The name is the repository's, **Consequence–Equilibrium Attention (CEQ)**; its operator is
ChaCAL's read with the diagonal kept in the inverse, on the record's three-corner base.

## 2.1 Definitions

**Definition 1 (base family).** For $s$ positions with logits $qk_{ij}=q_i\cdot k_j$, magnitudes
$m_k\in[0,1]$, phases $\theta_k$, and a switch $\beta\in\{0,1\}$,
$$W_{\beta,qk,m,\theta}[i,j]=\mathbb 1[j\le i]\;\mathrm{pathProd}(m,\theta)_{ij}\;e^{qk_{ij}}/Z_i^{\beta},\qquad
\mathrm{pathProd}(m,\theta)_{ij}=\prod_{k=j+1}^{i} m_k e^{i\theta_k},$$
`READ lean/CEQ/V16Domain.lean:92-97`, `ceq/arm_smprime.py:144`. The exponential prefix-scan form
of `V16Domain.lean:366-378` (`Hop`) is its restriction to $m>0$ (`prefix_logit_mask_restated`
clause 5, `:221`) and carries no zero (`no_prefix_scan_represents_a_zero_gate`, `:165`). Corners:
$\beta=1,m\equiv1$ softmax; $\beta=0,m\equiv1$ linear attention; $\beta=0$, QK off, the path product
(`three_corners_containment` `:433`, `corners_are_distinct` `:445`). Ruling: the base carries the
multiplicative gate (`refute_theory_math` item 1, `refute_theory_occvac` §1.2 row 9) so that
Proposition 6 has a subject. Mechanism: V-25, P-7.

**Definition 2 (two regimes, one solve).** Regime S: $\beta=1$, $W=P$ row-stochastic with
$P_{ii}>0$ for $i\ge1$ and $P_{00}=1$; $\gamma\in[0,1)$; $(I-\gamma P)^{-1}=\sum_{t\ge0}(\gamma P)^t$
by $\|\gamma P\|_\infty=\gamma<1$. Regime N: $W=A$ strictly causal ($A_{ij}=0$ for $j\ge i$), any
$\gamma$, $A^s=0$ (`Nilpotent.pow_card_eq_zero` `:77`, `occupancy_is_exact_inverse` `:96`). The
boundary is a theorem (`one_not_nilpotent` `:105`), and an absorbing row is not `StrictlyLower`:
regime N admits no absorbing row, and at $\gamma=1$ one makes $I-A'$ singular (`RUN[M]`
$\det(I-A')=0.0$). The north star lives in regime S, corner 3 in regime N; they do not meet on a
boundary row. Mechanism: V-25, C8.

**Definition 3 (boundary rows).** Disjoint position sets: the sink $\mathcal A_{\rm sink}\ni0$,
the goal $\mathcal A_0$, the constraints $\mathcal A_1,\dots,\mathcal A_K$;
$\mathcal A=\bigsqcup\mathcal A_\bullet$, $T=[s]\setminus\mathcal A$, the query position in $T$.
$P_{a,\cdot}=e_a$ for $a\in\mathcal A$; canonical form $P=\begin{pmatrix}Q&R\\0&I\end{pmatrix}$,
$Q=P_{TT}$, $R_k=P_{T\mathcal A_k}$. Two construction facts (`THESIS_CORRECTIONS_2.md` §1,
re-run `RUN[J]`): **F1** row 0 of any causal softmax at $\beta=1$ is $e_0$, so position 0 is
absorbing whether or not declared; undeclared, $\det(I-Q)=0.0$. **F2** walks descend, so a
boundary set after the query has committor exactly $0.0$; every set precedes the query.
Ruling on BOS (`refute_theory_math` C8(b), `refute_falsify_math` §3.11 against `design_theory`
§1.3): BOS is its **own sink set with value $0$ on every indicator channel**, not a goal member —
otherwise "reach the goal" collapses to "descend to 0 without a constraint" and the two rules of
Definition 7 become one by construction (V-3, C4); the sink share ("fell off the prompt") is printed
beside every label. The value-zero column sink of `V15Fork.Asink` (`V15Fork.lean:67-70`) and the
sink row are two distinct conditions that coexist on any causal softmax (`sweep_resolvent.md`
§2.14; P-7). Mechanism: V-25, V-8, V-12, D-3.

**Definition 4 (state, mixing matrix, read).**
$$z(\gamma)=(I-\gamma W)^{-1}V,\qquad \Pi_\gamma=(1-\gamma)W(I-\gamma W)^{-1},\qquad O(\gamma)=\Pi_\gamma V.$$
The read carries the $(1-\gamma)$ factor. For any row-stochastic $W$ — boundary rows or not —
the bare $W(I-\gamma W)^{-1}$ has every row sum $1/(1-\gamma)$ (`RUN[J]`, `RUN[M]`: $2.5$ at
$\gamma=0.6$ with and without the sets); the factor is a property of the class, not of the
boundary rows. Mechanism: V-17, V-23.

**Definition 5 (horizon dial).** $\gamma\in[0,1)$ is one learnable scalar per head: the discount of
`bellman-1957-markovian`, learnable as in `roffo-2026-infsa`, the $\gamma\uparrow1$ limit of
`kemeny-1960-finitemarkov`. At $\gamma=1$ with any absorbing row $I-P$ is singular
(`RUN[M]`); the endpoint is a limit (Proposition 9).

**Definition 6 (interventional channel, two objects).** A candidate move $a$ is an intervention.
(i) On the *oracle's* latent chain: a row clamp $P'_{v_a,\cdot}\leftarrow e_{u_a}$, rank one,
$u^\top\mathbb 1=0$. (ii) On the *arm's* operator: a token rewrite at position $i$, which changes
$q_i$ and $k_i$, hence every softmax row $j\ge i$ through $qk_{ji}$ and $Z_j$ — `RUN[J]` at $i=12$,
$s=32$: $\mathrm{rank}(\Delta P)=20=s-i$. Two objects, two prices (Proposition 7); `design_theory`
§1.6 ("supported on the intervened rows") is KILLED for (ii), kept for (i). Mechanism: M-8, P-8.

**Definition 7 (labels and rules).** Per move, the reach-avoid vector
$(q^{({\rm sink})},q^{(0)},q^{(1)},\dots,q^{(K)})(a)$ with $\sum=1$ on $T$ (Proposition 8). Two
rules as columns of one bed (V-1): $a^\star=\arg\max_a q^{(0)}(\mathrm{do}\,a)$ (reach the goal
before any constraint) and the Chebyshev $a^\dagger=\arg\min_a\max_{k\ge1}q^{(k)}(\mathrm{do}\,a)$;
lexicographic as a third column; the disagreement fraction printed. Two heads: (H-q) the committor
vector by the **exact triangular solve at $\gamma=1$ on the transient block**,
$\hat q=(I-\hat Q)^{-1}\hat R_k\mathbb 1$; (H-z) the state/displacement channel at the trained
$\hat\gamma$. Ruling (`refute_instrument_occvac` FATAL-2, `refute_falsify_math` item 16): an
undiscounted label and a discounted read at $\hat\gamma<1$ pull $\gamma$ in opposite directions —
$q$ to $1\%$ at $\tau=8$ needs $\hat\gamma\ge0.99857$, $1/(1-\hat\gamma)\approx697$ (`RUN[F]`) —
so the committor head does not carry $\gamma$; the discounted read $E_i[\gamma^{\tau-1}\mathbb 1_k]$
is a separate registered label at a bed constant $\gamma_{\rm env}$ (`fisac-2019-bridging`) for the
horizon-dial bet only. Mechanism: V-17, D-2, D-3, M-20.

## 2.2 Propositions

**Proposition 1 (parity at $\gamma=0$, with a rejection region).** For every causal $W$ and $V$,
$O(0)=WV$; in IEEE-754 forward substitution at $\gamma=0$, $z=V$ bitwise and the read is the same
matmul as attention. *Proof.* $(I-0\cdot W)=I$; $0\cdot x=0$, $v-0=v$, $v/1=v$ exact for finite
entries. $\square$ Carrier `V16Domain.corner_softmax` `:394`; target `gamma_zero_is_softmax` [M].
*Evidence.* `RUN[coord]` `torch.equal(O(0),PV)=True` at $s=64,d=16$; rejection region
$\max|O(0.5)-O(0)|=2.3002850040264393$; the record's own softmax corner is $1.110223\times10^{-16}$
off `ceq/lm.py` on $19/64$ entries (`READ V16_ARM_SMPRIME.md:266-293`), so "bitwise" is against the
lane's own routine. *Battery* (`refute_theory_occvac` §2.1): the plant "non-causal $W$" is deleted
(it passes at $\gamma=0$, `RUN[M]` `array_equal True`); kept: $\gamma\ne0$ and $\beta=0$ at the
softmax corner ($\max|{\rm gap}|>0.5$, `READ V16_ARM_SMPRIME.md:336-339`); declared-empty row:
"ChaCAL at the same $\gamma$" — a reproduction of ChaCAL's Eq. 5 at $\gamma=0$, not a
contribution. Mechanism: V-24, V-3. Status: **RUN + LEAN (corner)**.

**Proposition 2 (corner 3 is a resolvent, regime N).** For $A$ strictly lower bidiagonal with
$A_{i,i-1}=a_i$: $[(I-A)^{-1}]_{ij}=\prod_{k=j+1}^{i}a_k$ for $j\le i$, $0$ above. *Proof.* One walk
per $(i,j)$ of length $i-j$; $A^s=0$; $\sum_{m<s}A^m=(I-A)^{-1}$ (`occupancy_is_exact_inverse`),
entry $(i,j)$ receiving the $m=i-j$ term. $\square$ Consequence: BED-M's label
$y_{s-1}=((I-A)^{-1}b)_{s-1}$ (`READ scale/negation_scope.py:286-304`) is $z(1)_{s-1}$, so any arm
containing corner 3 reproduces BED-M's label as its own forward — BED-M is contained, not won.
*Evidence.* `RUN[coord]` $\max|G-(I-A)^{-1}|=0.0$, last row vs `equilibrium_oracle`
$6.217248937900877\times10^{-15}$, $A^{64}=0$ exactly. Mechanism: D-2. Status: **LEAN** (two halves)
+ target `pathprod_is_chain_resolvent` [S].

**Proposition 3 (the committor is the resolvent read with absorbing rows — classical).** $P$
row-stochastic including its absorbing rows, $\rho(Q)<1$, $V=\mathbb 1_{\mathcal A_k}$; for $i\in T$,
$$(1-\gamma)z(\gamma)_i=E_i[\gamma^{\tau_k}\mathbb 1_k],\qquad O(\gamma)_i=E_i[\gamma^{\tau_k-1}\mathbb 1_k],\qquad
\lim_{\gamma\uparrow1}O(\gamma)_i=q^{(k)}_i=[(I-Q)^{-1}R_k\mathbb 1]_i,$$
and, when the declared sets exhaust the absorbing states, $\sum_\bullet q^{(\bullet)}=\mathbb 1$ on
$T$; at $i\in\mathcal A_k$ the read is $1$, not $\gamma^{-1}$. *Proof.* Exchange sums in
$z_i=\sum_t\gamma^tP_i(\tau_k\le t,k)$; first-step analysis; monotone convergence;
$\sum_kR_k\mathbb 1=\mathbb 1-Q\mathbb 1$. $\square$ Every clause is Kemeny–Snell; the *read* form on an
attention $P$ is the composition. Discounted reads sum below one: `RUN[I]` state-form
$[0.136,0.340]$, read-form $[0.227,0.567]$ at $\gamma=0.6$, ratio exactly $\gamma$; the delay share
printed beside every safest-move reading is $1-\sum_kE[\gamma^{\tau-1}\mathbb 1_k]$. *Evidence.*
`RUN[J]`/`RUN[M]`: read vs $E[\gamma^{\tau-1}]$ to $8.33\times10^{-17}$; BED-1's real sets
$A=[0],B=[1],|T|=9$: resolvent read vs `bed["q"]` $0.0$, residual $1.04\times10^{-17}$,
$\rho(Q)=0.9408612510154677$; Kirchhoff $<10^{-10}$ (`READ MATHEMATICS.md:542-600`). *Ruling on
the $3.8\times10^{-7}$ line.* `refute_theory_math` item 8 derived a universal gap $\ge1-\gamma$; the
derivation dropped the indicator — $q_i-E_i[\gamma^{\tau}\mathbb 1_k]=E_i[(1-\gamma^\tau)\mathbb 1_k]\ge(1-\gamma)q_i$,
small where $q_i$ is. `RUN[J]` at $\gamma=1-10^{-6}$ on channel $\{15\}$: $\max_T=1.42\times10^{-7}$.
The line stands with $(1-\gamma)q_i$ printed beside it. *Read-side plant* (ADD): declare
$\mathcal A_k$ on the wrong set $\Rightarrow q$ moves by $O(1)$. Mechanism: D-2, V-12, V-25.
Status: **CLASSICAL, RUN on the arm's $P$**; targets `committor_is_resolvent_read` (a) [M],
(b-causal) [M] via Proposition 10, (b-chain) [S], `hitting_time_transform` [D].

**Proposition 4 (Neumann tail: attained on the class, and what a mask does).** (i) $P\ge0$
row-stochastic (identity rows included), $\gamma\in[0,1)$, $K\ge0$:
$$\Big\|(I-\gamma P)^{-1}-\sum_{k\le K}(\gamma P)^k\Big\|_\infty=\frac{\gamma^{K+1}}{1-\gamma}\quad(\text{equality});$$
for $\|P\|_\infty\le1$ the same quantity is $\le$ the right side; for $\|P\|_\infty>1$ the bound
*can* fail (a nilpotent $A$ with $\|A\|_\infty=2$ satisfies it at $K=s-1$, `RUN[M]`
$1.98\times10^{-10}$ vs $3.68\times10^{-5}$). *Proof.* The tail is entrywise non-negative with row
sums $\sum_{k>K}\gamma^k$; the $\infty$-norm of a non-negative matrix is its largest row sum
(`meyer-2000-matrix`, `horn-2013-matrix`). $\square$ On the vector: $\le\gamma^{K+1}\|V\|_\infty/(1-\gamma)$,
never the bare $\delta$ (V-17). `err = bound` is a declared V-3 identity; the bind is the planted
non-stochastic $P$ — quote the convergent plant (rows $1.5$, $\gamma=0.6$, $K=2$: $7.29$ vs $0.54$,
`RUN[M]`); the coordinator's $\gamma=0.7$ plant ($119.37$ vs $1.143$) is a divergent series
($\gamma\cdot1.5=1.05$) and is labelled so. (ii) **Mask amplification** (ADD,
`refute_instrument_math` row 8): for a sparsified $P_m$ with dropped row mass
$\varepsilon=\|P-P_m\|_\infty$, $\|O_{\rm full}-O_{\rm mask}\|_\infty\le\varepsilon\|V\|_\infty/(1-\gamma)$,
attained up to a constant: `RUN[J]` $s=3$, $\gamma=0.9$, $\varepsilon=0.1$: $0.5263157894736843$
against the naive $0.1$. Every F1 union bound carries $1/(1-\gamma)$; the exact solve on an F1 mask
is refused (L-CERT). *Census.* $\mathrm{rowsum}(\hat P)$ on every trained cell; at $\hat\beta\ne1$
rows sum $1.31\dots10.29$ (`READ V16_ARM_SMPRIME.md:28-32`) and no certificate is printed (V-25).
The `≈7.6e-05` vector bound of `sec_cost.md` §4.x.3 carries `[ASSUMED ‖V‖_∞ ≈ 5]`. Mechanism:
V-3, V-10, V-17, V-24, L-CERT.
Status: **DERIVED (textbook) + RUN**; targets `neumann_truncation_bound` [S],
`neumann_tail_attained` [S], `mask_amplification` [S].

**Proposition 5 (one triangular solve; $\Pi_\gamma$ is a mixture; the support does not move).**
(a) For causal $W$, $I-\gamma W$ is lower-triangular with diagonal $1-\gamma W_{ii}\in[1-\gamma,1)$
in regime S (row 0 and every absorbing row attain $1-\gamma$; `RUN[M]` min $0.4$ at $\gamma=0.6$)
and $=1$ in regime N; $z$ is a forward substitution costing $+s^2d/2$ MACs and depth $s$ **over**
the $s^2d$ causal MACs of the softmax head it contains at $\gamma=0$ ($+50\%$; $+25\%$ if $QK^\top$
is counted dense) — an increment, never a stand-alone figure (`sec_cost.md` §4.x.1, M-8).
(b) In regime S, $\Pi_\gamma=(1-\gamma)\sum_t\gamma^tP^{t+1}$ is row-stochastic and non-negative,
with or without absorbing rows (`RUN[F]` row sums $1.000000000000000$), so $O_i\in\mathrm{hull}\{V_j:j\le i\}$
for every $\gamma$: the record's Q2/W3 hull bound applies verbatim, and labels outside the hull
need the value rescale of `V15Fork.Asink_computes_chain` or regime N; committors live in $[0,1]$.
(c) On a dense causal softmax, $\mathrm{supp}\,\Pi_\gamma=\mathrm{supp}\,P$ with or without boundary
rows (`RUN[J]` `True`; lower-triangle minimum on transient rows $2.1\times10^{-3}>0$): the
resolvent adds **no support** in regime S; absorbing rows change *weights* downstream (mass
redirected to $a$); the "transitive closure" sentence has content only under F0 zeros. The claim
"identity rows change the support" (`design_falsify` §3.1) is KILLED (`refute_falsify_math` item
13, `refute_instrument_math` row 5). *Evidence.* `RUN[coord]` `solve_triangular` vs dense inverse
$1.7763568394002505\times10^{-15}$; `RUN` (NEPTUNE, certified RTX 4060, float32, $n=2048,s=64,d=16$;
producer owed to `scripts/k_cert.py`): solve$+Pz$ fwd+bwd $2.514$ ms vs $PV$ $1.473$ ms vs one
Neumann hop $3.001$ ms — truncation never wins in MACs. Kernel boundary: no finite-state dual for
softmax $P$ (`hu-2025-ssdtheory`); the exact path is the full $s\times s$ solve or Zhao et al.'s
$\tilde O(n^{4/3}d)$. Mechanism: M-8, M-3, V-17, P-7.
Status: **DERIVED + RUN**; targets `lower_triangular_isUnit`, `diag_one_sub_smul_pos` [M],
`resolvent_is_triangular_solve`, `mixing_matrix_rowStochastic` [S].

**Proposition 6 (segmentation is exact, and a zero gate is a boundary condition).** (i) In the
Definition-1 base, $m_c=0$ gives $W_{ij}=0$ for all $j<c\le i$ (`pathProd_eq_zero_iff` `:129`, no
hypothesis); then $I-\gamma W$ is block lower-triangular with a zero sub-block and inverts
blockwise; an identity row respects every cut, so absorbing rows are consistent with the block
structure (`refute_instrument_math` row 15). In regime S an exact zero is a $-\infty$ logit — F0
by mask, not by gate — and the iff has no producing theorem on the softmax corner
(`no_prefix_scan_represents_a_zero_gate`); `segmentation_blockdiag` [M] is stated for
`StrictlyLower` and `resolvent_fromBlocks` [M] is the half that transfers. (ii) **ADD** (`refute_theory_math`
item 11, `RUN[J]`): after a cut at $c$ the row $c$ has window $\{c\}$ and renormalises to
$P_{cc}=1.0$ — a new undeclared absorbing state; with the sets of Definition 3 and $c\notin\mathcal A$,
$\rho(Q)=1.0$ and the committor solve is singular; declaring $c$ absorbing severs every boundary
set before the cut from every $i\ge c$ ($q=0.0$, `RUN[M]`); segmentation and the reach-avoid read
coexist only if every segment carries its own sink, goal and constraint sets, with the cut position
a registered dial. (iii) The dividend is stated in pair-count units, $D=s(s+1)/\sum_mL_m(L_m+1)$:
BED-M reads $31.04\times$ from the live-pair fraction $0.0322$ (`READ V16_ARM_SMPRIME.md:518-522`;
the source line's pair counts disagree by $2\times$, flagged); $(C5)$'s $s^2/\sum L_m^2$ would read
$58.5$ (V-17); a corpus property, $1\times$ on the softmax corner. Mechanism: L-CERT, V-25, D-3,
V-22, V-24 (the $10^{-300}$ plant reads F0 by underflow after $\ge80$ gates at $0.5$; the safe
plant is a $-30$ logit, $e^{-30}=9.36\times10^{-14}$, `refute_instrument_math` row 14).
Status: **LEAN (annihilation) + DERIVED + RUN**; targets `cut_makes_segment_head_absorbing` [M],
`masked_softmax_blockdiag` [S].

**Proposition 7 (consequence as a displacement field).** (i) For $(P,V)\to(P',V')$,
$\Delta z=(I-\gamma P')^{-1}(\Delta V+\gamma\,\Delta P\,z)$ exactly — one extra solve (four lines;
`bottou-2013-counterfactual` §7.3 is its linearisation). (ii) For lower-triangular $P'$ with
$P'_{<i,\cdot}=P_{<i,\cdot}$, $V'_{<i}=V_{<i}$: $z'_{<i}=z_{<i}$ and the suffix re-solve equals
the full re-solve — by *triangularity*, not nilpotency (`RUN[F]` $5.6\times10^{-17}$ on the
non-nilpotent softmax corner); consequences propagate forward only, $\Delta z_0=0$ on every draw.
(iii) For a **row clamp** (Definition 6(i)) $P'=P+e_iu^\top$, $u^\top\mathbb 1=0$, $M=(I-\gamma P)^{-1}$:
$\Delta z=\gamma(Me_i)(u^\top z)/(1-\gamma u^\top Me_i)$ with denominator
$(1-\gamma p'_{ii})/(1-\gamma P_{ii})>0$ for every $\gamma<1$, absorbing rows included
(`RUN[I]` $1.45\times10^{-15}$; `RUN[F]` $0.511918$); the column $Me_i$ costs one substitution
($s^2/2$ MACs), the update $O(sd)$; $m$ candidates cost $m\cdot s^2/2$ against the solve's $s^2d/2$,
ratio $m/d=0.5$ at $(8,16)$. For a **token rewrite** (Definition 6(ii)) $\Delta P$ has rank $s-i$
and the price is the suffix re-solve $(s-i)^2d/2$ — pricing the arm at the oracle's rank-one rate
is M-8 literally. (iv) $V\equiv\mathbb 1\Rightarrow\Delta z\equiv0$ exactly; in floats $\le10^{-15}$
(`RUN[F]` $1.1\times10^{-16}$; a bitwise $0.0$ is a one-code-path accident); the region is a
Gaussian $V$ ($0.1096$ at $s=32$, $1.127$ at $s=64$). (v) A value-only move leaves every $q^{(k)}$
constant across moves — the $m(K+1)$-RHS route is legal for $\Delta z$ only. *Metric* (VENUS):
position-matched per-coordinate NRMSE over coordinates $\ge i_{\min}$, field cosine, magnitude
ratio, harmonic residual $r(\hat z)=\|(I-\gamma P_{\rm env})\hat z-V\|_\infty/\|V\|_\infty$ with
$\sigma_{\min}$ and $\|I-\gamma P_{\rm env}\|_\infty$ printed (its units), the sign column kept
(softmax reads $0.807843$, `READ MATHEMATICS.md:396-406`); McNemar on the sign column only, paired
$t$/Wilcoxon on the cosine. Mechanism: V-24, D-5, V-26, M-8, P-8, V-17.
Status: **DERIVED + RUN**; targets `displacement_identity` [M], `causal_forward_only` [S],
`sherman_morrison_row` [S] (docstring: row clamp only).

**Remark (EMC, retired to a remark).** For $\gamma<1$ the map $x\mapsto V+\gamma P'x$ is a
contraction with one fixed point, so "settle then intervene" and "intervene then solve" coincide
for every row-stochastic $P'$, cyclic or not (`RUN[J]` $8.9\times10^{-16}$ on a dense feedback
$P$; `RUN[I]` $1.33\times10^{-15}$; `RUN[F]` $4.4\times10^{-16}$). `dash-2005-emc` Thm 1 concerns a
reduced model whose equilibrated form hides feedback; an explicit linear fixed point has no
such gap. The "planted feedback instance" exhibits loss of forward-only propagation
($\Delta z_{<i}\ne0$ on a non-causal $P$) and is filed as a **triangularity** bind under
Proposition 7(ii). `momennejad-2017-sr` / `russek-2017-predictive` (mechanism [U]) remain the
external argument that a cached resolvent needs the re-solve under a transition change.
Ruling: two refuters KEEP-with-label, two KILL; the KILL stands because the rejection region was
an artefact of a wrong algorithm (V-24, V-3).

**Proposition 8 (the safest move; degeneracy; the reach-avoid identity with a sink).** With
Definition 3, $q^{({\rm sink})}+q^{(0)}+\sum_{k\ge1}q^{(k)}=\mathbb 1$ on $T$ (Proposition 3; `RUN[I]`
$[0.9999999999999993,1.0]$). *Degeneracy lemma.* If the constraint sets exhaust the absorbing
states, $\sum_{k\ge1}q^{(k)}=\mathbb 1$ and $\max_kq^{(k)}\ge1/K$ by pigeonhole — "avoid every
constraint" is unattainable (V-12 generalised). Its planted negative is "BOS declared inside a
constraint set"; the plant "drop $\mathcal A_0$" cannot run (dropping BOS makes $I-Q$ singular,
`refute_instrument_math` row 3), and the coordinator's $0.605$ was computed *with* the goal
counted. *Floors.* The zero-information floor on the argmin is chance, $1-\max_a\pi(a^\star)=1-1/m$
uniform: $0.75/0.875/0.9375$ at $m=4/8/16$; the weak Fano $1-\ln2/\ln m$ ($0.5/0.6667/0.75$) is
struck as a floor (an arm at error $0.60$, $m=4$, would print above it while worse than chance);
tight Fano $(\ln m-\ln2)/\ln(m-1)$ ($0.6309/0.7124/0.7679$, `RUN[J]`) only where
$I(X_{\le k};a^\star)>0$ is computed; $m=2$ has floor $0.5$. The zero-hop view carries the move
tokens and membership flags, so $I(X_{\le0};a^\star)>0$ and the plug-in is computed (V-25 on a
floor). *Census.* Label sd over the admitted query region (the prefix before the first constraint
is constant); every class frequency of $a^\star$ in $(0.05,0.95)$ (the "argmin-unique" gate was
inverted); rule-disagreement fraction $>0$; discard count; $q^{({\rm sink})}$ apart from $q^{(0)}$.
*Bed.* The environment chain must lie inside the arm's operator class or the identity licenses
nothing (`refute_theory_occvac` §2.4, `refute_instrument_occvac` FATAL-1): an undirected
Rips/`bed_1` chain has `SymmSupport` and no causal $\hat P$ equals it (one line; target
`lowerTriangular_ne_symmSupport`); a causal chain recoverable from edge tokens makes the label the
arm's own resolvent (D-2). Ruling: BED-S's environment is a random DAG in token order with
self-loops, adjacency rows as multi-hot node features so $\hat P=P_{\rm env}$ is representable at
$d_{\rm model}\ge n_{\rm nodes}$; the VOID list is pre-registered — `shape − softmax` and
`shape − skyline` are reproduction-vs-non-reproduction contrasts; creditable are
`shape − ChaCAL-diag-with-sink` (boundary mechanism), `shape − Neumann-K` (exactness) and
$\|\hat P-P_{\rm env}\|_\infty$ per seed (identification); leak clause C2(a) is dropped (the graph
is the bed's input), C2(c) kept; `bed_1`/E4′ remain oracle cross-checks (Kirchhoff at $K=2$).
Hazard carried to Limits: `misra-2023-safety-constrained-mdp`.
Mechanism: V-8, V-12, V-25, D-2, D-3, M-2, V-10, L-FLOOR. Status: **OPEN** (no cell; discharged by
PLAN items S-1 to S-4 of the ledger).

**Proposition 9 (the horizon dial).** $\gamma=0$: next step, softmax bitwise (P1); $0<\gamma<1$:
the hitting-time transform $E[\gamma^{\tau-1}]$ (P3); $\gamma\uparrow1$: absorption probabilities,
basin membership, the $q=\tfrac12$ isocommittor for $K=2$ (`ceq/beds/bed_1.py:5-8,:382`,
`e-2010-tptreview`); corner 3 is $\gamma=1$ on a nilpotent chain (P2). *Pinning.* Ruling 10′'s
$\Lambda=2[LL(\hat\gamma)-LL(\gamma=0)]$ on held-out, null and held-out set declared for $\gamma$
(Ruling 10′ was stated for $\beta$); with $\gamma\in[0,1)$ the null sits on a boundary and the
$95\%$ point is $\tfrac12\chi^2_0+\tfrac12\chi^2_1=2.7055$, not $3.8415$ (`RUN[J]`; the citation is
owed, not in `references.bib`); the $|\hat\gamma|<0.05$ rule is deleted (two verdicts for one
cell); the mirror kill $\hat\gamma>0.99$ prints $1/(1-\hat\gamma)$ beside every $\delta$
(`READ V15_R1.md:56`). Mechanism: V-9, M-2, V-17. Status: **DERIVED (corollary)**; instrument OPEN.

**Proposition 10 (invertibility on the causal class is a diagonal read).** For the arm's causal
$P$ with finite logits, $Q=P_{TT}$ is lower-triangular, $\rho(Q)=\max_{i\in T}P_{ii}$ (`RUN[J]`
$0.6926596893360386$ both), and $I-Q$ is invertible **iff** $0\in\mathcal A$ ($\det(I-Q)=0.0$
undeclared): "absorption a.s." is the single census line "BOS declared", and
`committor_is_resolvent_read`(b) is [M] via `lower_triangular_isUnit`; the Perron certificate is
for the oracle's non-causal chain only (`SymmSupport`, $\rho(Q)=0.9409$). The $\|Q\|_\infty$ Neumann
bound on the committor is vacuous ($\|Q\|_\infty=0.958$, bound $19.2$ at $K=4$ vs error
$2.5\times10^{-3}$, `RUN[I]`): the committor head is the exact solve and claims no truncation
certificate. Mechanism: V-10, V-25, L-CERT. Status: **DERIVED + RUN**; `isUnit_one_sub_of_perron`
split into (b-causal) [M] and (b-chain) [S].

**Proposition 11 (regime boundary).** The softmax corner keeps $P_{ii}>0$, so $\gamma P$ is not
nilpotent (`one_not_nilpotent`) and "all $s$ hops by nilpotency" holds in regime N only. The
discriminating evidence is $\max_{i\ge1}(\gamma P)^{32}_{ii}=6.27\times10^{-13}$ beside
$\gamma^{32}=7.96\times10^{-8}$ attained at $(0,0)$ (`RUN[M]`; the design's number was V-4). The
masked-diagonal route is not a matrix identity ($\|(I-\gamma P_m)^{-1}-\sum_{t<s}(\gamma P_m)^t\|=1.99\times10^{-7}$);
it is exact on a value channel iff $V_0=0$, which the Definition-3 sink satisfies on every
indicator channel (target `masked_sink_finite_sum` [S]). Default: keep the diagonal and carry
Proposition 4. Mechanism: V-25, P-3, V-4. Status: **LEAN + RUN**.

## 2.3 What the shape computes

Consequence: $\Delta z$ (Proposition 7), one extra solve, scored as a field. Committor vector: $\hat q$
by the exact solve on $\hat Q$ (Propositions 3, 10). Safest move: the two rules of Definition 7 with
their disagreement fraction; the threshold form $\max_k\hat q^{(k)}+\delta\|V\|_\infty\le\delta_{\rm thr}$
bounds the solve, never the model error $\hat P\ne P_{\rm env}$ — no move is "admitted" by it (V-17).
Next transient state / phase / equilibrium: the dial (Proposition 9); "phase" is metastable
membership (`prinz-2011-msm`, `ramsauer-2021-hopfield`, `erel-2025-attentionchains`).

## 2.4 The topological layer (restricted)

(a) The F0 endpoint is a theorem (Proposition 6) on gated corners or masked $P$; on the parity
corner the $\varepsilon=0$ influence graph is one weak component on $100\%$ of draws ($\Pi_{i0}>0$),
so the $\beta_0$ barcode row is KILLED on BED-S (three refuters agree) and survives only as the
thresholded barcode with a permutation null on BED-M (corner 3, zero gates on $3$ of $3$ values):
`NOT MEASURED — needs a digraph β₀ instrument` (`beta0_interleaving` consumes point clouds,
`READ ceq/certs/topological.py:476-505`); Turner's hypothesis is stated as a predicate on the
filtered object first. (b) Isocommittor surfaces: BED-1's $q=\tfrac12$ guard inside the read;
NOT MEASURED. (c) Mapper cover $\to$ tiles $\to$ schedule: the do-nothing control is both the
0D-salience schedule and Zhao et al.'s blockwise resolvent; the F1 union certificate carries
$1/(1-\gamma)$ (Proposition 4(ii)); the dense control is not runnable at $n=2048$, $s=4096$
($\approx137$ GB against $7.996$ GiB), so $n$ is stated per $s$; `kernels#22` has no backward.
Mechanism: V-3, V-25, M-15, M-2, V-9, L-CERT, P-4, P-8.

## 2.5 The cost law and the kernel path

Per head, MACs, increments over the softmax head contained at $\gamma=0$ (`sec_cost.md` §4.x.9):
serial substitution $+s^2d/2$, depth $s$, $O(s^2)$ memory ($P$ explicit; no bf16 path for
`solve_triangular` [U]); chunked $+s^2d/2+sCd/2+sC^2/3$, depth $s/C$, `NOT MEASURED — needs a
chunked kernel` (pattern `yang-2024-deltanet`); Neumann $+Ks^2d/2$, depth $K$, never wins in MACs;
F0-segmented $+\sum_mL_m^2d/2$; CSR two-stage $+\,$units$\cdot B^2d/2$, forward-only. Measured
increment $2.514-1.473=1.041$ ms/step at $n=2048$ ($2.312$ at $4096$, $4.542$ at $8192$), a per-op
floor under a dispatch gap of $2.0\times$–$6.6\times$ (`READ scale/m3_flops.py:101-121`). One
bed-cell pair on the softmax corner $\approx34$ s; seven arms $7\times8\times1.680+4.0\approx98$ s
$\approx1.6$ GPU-min (`refute_instrument_math` row 28 corrects the design's $4$); the corner-3
base ($\approx133$ s) has no certificate domain ($\hat\beta=0$ rows sum $1.31$–$10.29$).
Determinism: `solve_triangular` bitwise forward and backward over 8 repeats under strict mode, no
torch-documented guarantee; `cumsum` raises; the shape on the softmax corner is the first *gated
wing* whose training step runs under `use_deterministic_algorithms(True)`. Circuit class: DET
(`cook-1985-taxonomy`, §3.3). Must-fire checks: parity at $\gamma=0$ with the $\gamma=0.5$ region;
solve vs dense inverse with a non-triangular plant; certificate in vector units with the convergent
plant; segmentation `torch.equal` zeros with the $-30$ plant; CSR fill-in guard; three-outcome
determinism; cell price within the law-vs-measured band. Mechanism: M-8, M-3, L-CERT, V-17, P-1,
P-8, V-16, V-23.

## 2.6 Lean targets (build order; every `[M]` is pending `lake build`; a missing Mathlib lemma moves it to `[S]`; nothing is cited as proved before it builds — L-LEAN, P-11)

| # | target | grade | carries | refusal shipped | mechanism |
|---|---|---|---|---|---|
| 1 | `gamma_zero_is_softmax` | [M] | P1 | `gamma_half_is_not_softmax` witness | V-3, V-24 |
| 2 | `bos_row_is_absorbing` at $\beta=1$ | [M] | F1 | at $\beta=0$ row 0 is $e^{qk_{00}}$ | V-25 |
| 3 | `later_boundary_unreachable` | [M] | F2 | — | V-8 |
| 4 | `softmax_corner_not_nilpotent` ($0<\gamma$, lower-triangular) | [M] | P11 | `pow_card_eq_zero` | V-25 |
| 5 | `lower_triangular_isUnit`, `diag_one_sub_smul_pos` (interval $[1-\gamma,1)$) | [M] | P5a, P10 | `zero_diag_not_unit` | M-8 |
| 6 | `resolvent_fromBlocks`; `segmentation_blockdiag` (`StrictlyLower`) | [M] | P6(i) | `tiny_gate_does_not_cut` at a $-30$ logit | L-CERT |
| 7 | `cut_makes_segment_head_absorbing`, `cut_severs_boundary_sets` | [M] | P6(ii) | — | V-25, D-3 |
| 8 | `displacement_identity` | [M] | P7(i) | `const_value_zero_displacement` | V-24, D-5 |
| 9 | `lowerTriangular_ne_symmSupport` | [M] | P8 (D-2) | `zero_not_a_counterexample` pattern | D-2 |
| 10 | `committor_is_resolvent_read` (a), (b-causal) | [M] | P3, P10 | — | D-2 |
| 11 | `subdiag_pow_entry`, `pathprod_is_chain_resolvent` | [S] | P2 | `forward_map_fills_in` at general $n$ | D-2 |
| 12 | `neumann_truncation_bound`, `neumann_tail_attained`; `mask_amplification` ($\varepsilon/(1-\gamma)$) | [S] | P4 | `nonstochastic_breaks_bound` (rows $3/2$, $\gamma\cdot3/2<1$); the $s=3$ witness | V-24, V-10, L-CERT |
| 13 | `resolvent_is_triangular_solve`; `mixing_matrix_rowStochastic` | [S] | P5 | `bare_read_row_sum` $=1/(1-\gamma)$ | V-17 |
| 14 | `causal_forward_only`; `suffix_resolve_eq_full` (prefix invariance); `sherman_morrison_row` (row clamp only) | [S] | P7 | `dense_P_displaces_backward`; `token_rewrite_rank` remark | V-24, M-8 |
| 15 | `isUnit_one_sub_of_perron` (b-chain); `reach_avoid_sum_one` (sets exhaust the absorbing states) | [S] | P3, P8 | `unit_needs_rho_lt_one`; `no_goal_forces_max_ge_inv_K` | V-25, V-12 |
| 16 | `masked_sink_finite_sum` ($V_0=0$) | [S] | P11 | — | V-25 |
| 17 | `hitting_time_transform`; `f1_cantelli_union` | [D] | P3, L-CERT | — | — |

Rows 1–10 are one evening of independent `[M]` items; 11–16 name their dependencies; 17 is off the critical path. Price: $0$ GPU-s each.

## 2.7 Limits carried to §8

Every `RUN[J]` number is one numpy float64 draw at $s=32$, $\gamma=0.6$, seed 0, CPU, an identity
or counterexample check, not a statistic; other planets' runs are quoted at their own geometries.
The NEPTUNE timings are one box, float32, a per-op floor, producer owed to `scripts/k_cert.py`.
The ChaCAL diagonal convention rests on one HTML fetch (`refute_falsify_math` §3.15) and is re-read
against the PDF before typesetting. Theorem numbers inside cited sources inherit the sweeps' `[U]`
marks; `grinstead-1997-probability` is `[U]`; the boundary-null citation is owed. Proposition 8 has
no cell, no realised sd, no measured $t^\star$; its DAG bed is written, not run. The barcode and
Mapper rows have no instrument. Lean grades are statements about statements; nothing was compiled.
No code file, no git write, no external fetch by this judge.
