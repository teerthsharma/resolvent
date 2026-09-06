# 01 — THEORY: ACCURACY

JUPITER (MYCROFT), derivations. Written 2026-09-05 under `CHARTER.md` at HEAD `99777ab`.
Every `READ` pin below is `path:line @ 99777ab` unless another commit is named; every
`RUN` number is quoted from the record with its producer named, and no code was written.

## Preface — what this book decides, and the verse that decides it first

This book decides one question: on which labels the resolvent read
$O=(1-\gamma)P(I-\gamma P)^{-1}V$ with absorbing boundary rows is *exact*, and at what
error a per-row softmax of matched parameters *cannot fit* those labels — so that "more
accurate" in forms A1 and A2 (`CHARTER.md` §1) is a rate, not a hope. Verse **01.1** decides
it first: the read is exact on the four vector labels iff the arm's $\hat P$ equals the
environment chain on the reachable set, with the propagation constants printed. Verse
**01.4** writes the $\mathrm{hop}_k\to$ committor reduction the record left unwritten; verses
**01.5–01.7** print every obstruction's vacuity line at the planned geometry
($s=64$, $d=16$, $h=1$, $p=32$; BED-S at $t^\star=8$, $m=8$, $K=2$, $N=8$, $4{,}769$
parameters) and the geometry at which each becomes non-vacuous. Verses **01.10–01.11**
resolve the Cheeger and Bellman hazards. Verses **01.13–01.17** state the ChaCAL delta as
theorems under the standing kill K-E1. Verses **01.18–01.22** turn the LEAPABLE fields into
named theorem targets with Lean grades. The honest sentence of `CEQ_SHAPE.md` §3.8 is the
Terminal of every chain.

**Notation.** Positions $[s]=\{0,\dots,s-1\}$; $P$ causal row-stochastic with $P_{ii}>0$
(regime S, `READ docs/CEQ_SHAPE.md:286-293`); boundary sets
$\mathcal A=\mathcal A_{\rm sink}\sqcup\mathcal A_0\sqcup\mathcal A_1\sqcup\dots\sqcup\mathcal A_K$,
$0\in\mathcal A_{\rm sink}$, $T=[s]\setminus\mathcal A$, $Q=P_{TT}$, $R_k=P_{T\mathcal A_k}$
(`:295-310`); $z=(I-\gamma P)^{-1}V$, $\Pi_\gamma=(1-\gamma)P(I-\gamma P)^{-1}$,
$O=\Pi_\gamma V$ (`:311-317`); the committor $q^{(k)}=(I-Q)^{-1}R_k\mathbb 1$ on $T$;
$P_{\rm env}$ the bed's latent chain; $\hat P$ the arm's. $|T|\in\{56,\dots,60\}$ at the
design point (DERIVED: four sets $\mathcal A_{\rm sink},\mathcal A_0,\mathcal A_1,
\mathcal A_2$ at $K=2$, each of size $1$ or $2$, so $|\mathcal A|\in\{4,\dots,8\}$ and
$|T|=64-|\mathcal A|$; the earlier $\{57,60\}$ was printed as DERIVED with no derivation and
excluded $|\mathcal A|=5,6,8$, which the stated geometry admits). Mechanism ids are `MISTAKES.md`
headings; laws are `CHARTER.md` §4.

**$K$ and $K'$ are two objects, and this book writes both.** $K$ is the number of constraint
sets, fixed at $K=2$ at the design point above, giving $K+2$ value channels. $K'$ is the
**Neumann truncation order** — the number of reused hops a stack takes before it is read —
with registered support $K'\in\{1,2,4,8,16\}$ (`READ 08_ARCHITECTURE.md` verse 08.11, whose
Hypotheses print the same support and the same separation; cited by verse id because book 08's
line numbers move under its own repairs, and at repair time the certificate row is at `:46`).
`CHARTER.md` §6's scope sentence writes both with one letter; book 08 does not, and from this
repair neither do verses 01.7 and 01.16, whose certificate is $\gamma^{K'+1}/(1-\gamma)$ and
never $\gamma^{K+1}/(1-\gamma)$.

**Instrument routing, restated at repair (`CHARTER.md` §3, P-4).** `CHARTER.md:179` requires
a `NOT MEASURED` clause to name *"the instrument, specified in the instruments book"*. Book
04 carries verses 04.1–04.21 and the card ids S-10, S-12, S-13, S-24, S-28, S-35, S-61,
S-62, S-73; a grep of `04_BEDS_AND_INSTRUMENTS.md` counts $0$ for "collision instrument",
"Hankel-rank", "margin probe", "Schur" and "SVD", and its one "Dirichlet" hit is 04.15's
manifest field `committor_route ∈ {dirichlet_gamma1, discounted_gamma}`, a route and not an
instrument. **Six instruments this book needs are therefore unspecified in the canon** — the
margin probe (01.4), the collision instrument (01.6), the Dirichlet-conductance instrument
(01.10), the Hankel-rank probe (01.18), the Schur instrument (01.20), the SVD of
$\hat\Pi_\gamma$ (01.21) — and every clause naming one now reads "unspecified in the canon"
with the shape written out here as a specification book 04 does not yet carry, and the kill
that depends on it marked **unreachable** until it does. Separately, this book routed kills
to the record's card ids S-02, S-17, X-2, X-4, R-20 and J-L18, none of which is a verse of
book 04; every such routing now carries a `READ docs/CEQ_SHAPE.md:<line> @ 99777ab` pin
instead, and says that the card is the record's and not book 04's. (The finding that
convicted this also named S-31, S-33, S-64 and N-15; a grep of this book at repair time
finds none of the four, and the same defect additionally covers J-D3, J-D6, J-D8, J-L0,
J-L6 and R-07, which the finding did not name and which appear here in Evidence position
only.) No verse of this book adds an instrument to book 04: that is book 04's own repair,
and until it lands the reachability column of the kills table below reads what it reads.

---

## Part I — The exact class

### 01.1 — The read is exact on four labels, and only through $\hat P$

**Statement.** Let $\hat P$ be the arm's causal row-stochastic operator with rows
$e_a$ on $a\in\mathcal A$. The read represents, with zero error as functions of
$\hat P$, exactly the linear images of its resolvent:
$$z^\star=(I-\gamma\hat P)^{-1}V,\qquad
\Delta z=(I-\gamma\hat P')^{-1}\big(\Delta V+\gamma\,\Delta\hat P\,z\big),\qquad
q^{(k)}=(I-\hat Q)^{-1}\hat R_k\mathbb 1,\qquad
a^\star=\arg\max_a q^{(0)}(\mathrm{do}\,a).$$
**One direction only.** If $\hat P=P_{\rm env}$ on the set of positions reachable from the
query, all four labels are exact. The converse holds for $z$ alone, and only at
$\gamma>0$: if $(I-\gamma\hat P)^{-1}V=(I-\gamma P_{\rm env})^{-1}V$ for **every** $V$ then
$\hat P=P_{\rm env}$ on the reachable set, by inverting both resolvents and dividing by
$\gamma$. The converse is **false** at $\gamma=0$, a value these Hypotheses admit:
$z^\star=(I-0)^{-1}V=V$ for every $\hat P$, so the label matches with
$\|\hat P-P_{\rm env}\|_\infty=1$ on the reachable set. It is false for the committor at
**every** $\gamma$: with $\mathcal A_{\rm sink}=\{0\}$, $\mathcal A_0=\{1\}$, $T=\{2\}$ and
row $2=(\alpha,1-\alpha-\beta,\beta)$, $q^{(0)}_2=(1-\alpha-\beta)/(1-\beta)$ reads $0.5$ at
both $(\alpha,\beta)=(0.5,0)$ and $(\alpha,\beta)=(0.25,0.5)$ — two chains $0.25$ apart in
sup norm with identical labels (DERIVED). The argmax clause inherits that many-to-one map,
so no "iff" is written for $q$ or for $a^\star$. Where $\hat P\ne P_{\rm env}$ the errors
are bounded, not zero:
$$\|\hat z-z\|_\infty\le\frac{\gamma\,\|\hat P-P_{\rm env}\|_\infty\,\|z\|_\infty}{1-\gamma},\qquad
\|\hat q^{(k)}-q^{(k)}\|_\infty\le\kappa\big(\|\hat Q-Q\|_\infty+\|\hat R_k-R_k\|_\infty\big),\quad
\kappa:=\|(I-\hat Q)^{-1}\|_\infty .$$
The argmin inherits the committor bound: $a^\star$ is read correctly whenever the gap
between the best and second-best goal committor exceeds $2\kappa(\|\hat Q-Q\|_\infty+\|\hat R-R\|_\infty)$,
and from this repair the clause is stated at the $\kappa$-free upper bound
$\kappa\le1/\varepsilon_{\rm esc}$ derived in the Hypotheses, because $\kappa$ itself is
`NOT MEASURED` and the record's $7.9$ is a condition number in different units.

**Hypotheses.** $\gamma\in[0,1)$ for $z$ and $\Delta z$; $\gamma=1$ on the transient block
for $q$ with $0\in\mathcal A$ so that $I-\hat Q$ is a unit (verse 01.10; `READ
docs/CEQ_SHAPE.md:539-548`); every boundary set precedes the query and the query lies in
$T$ (`:295-310`, facts F1 and F2). Non-vacuous at every geometry: the bounds are identities
of the class. At the judge's draw ($s=32$, $\gamma=0.6$, seed 0, `RUN[J]`) $\max_T\hat P_{ii}=0.6927$
and the **sup-norm condition number** of $I-\hat Q$ is $7.9$ (`READ docs/CEQ_SHAPE.md:2282`,
a line the record opens *"Withdrawn candidate, recorded so it is not re-found (V-7
discipline)"* — the provenance is printed because it is a withdrawn-candidate paragraph and
not a card). That number is **not** $\kappa$: $\mathrm{cond}_\infty(I-\hat Q)=\|I-\hat
Q\|_\infty\,\|(I-\hat Q)^{-1}\|_\infty$, so $\kappa=7.9/\|I-\hat Q\|_\infty$ and $\kappa\le7.9$
holds only if $\|I-\hat Q\|_\infty\ge1$, which is printed nowhere in the record.
**$\|I-\hat Q\|_\infty$ is printed here as the interval the class forces, and $\kappa$ is
derived from the quotient with both endpoints.** For a lower-triangular non-negative $\hat Q$
with row sums $r_i=\sum_j\hat Q_{ij}\le1$ the $i$th row sum of $|I-\hat Q|$ is
$(1-\hat Q_{ii})+(r_i-\hat Q_{ii})=1+r_i-2\hat Q_{ii}$, so
$$1-\max_T\hat P_{ii}\ \le\ \|I-\hat Q\|_\infty\ \le\ 2\qquad(\text{DERIVED; } r_i\ge\hat Q_{ii}
\text{ gives the left, } r_i\le1,\ \hat Q_{ii}\ge0 \text{ the right}),$$
which at the judge's draw reads $\|I-\hat Q\|_\infty\in[0.3073403106639614,\,2]$ from
$\max_T\hat P_{ii}=0.6926596893360386$ (`RUN[J]`), and therefore
$$\kappa=\frac{7.9}{\|I-\hat Q\|_\infty}\in[3.95,\ 25.70]\qquad(\text{DERIVED from the
quotient}).$$
The interval's **upper** endpoint is a bound in the direction the argmin clause needs and its
width is a factor $6.51$; the earlier text printed only the lower endpoint $\kappa\ge3.95$,
which is the opposite direction, and printing one endpoint of a two-sided interval was itself
the defect. Neither endpoint is a read: $\kappa$ is
`NOT MEASURED — needs a direct sup-norm read of $\|(I-\hat Q)^{-1}\|_\infty$ on the drawn
$\hat P$, unspecified in the canon (no verse of book 04 carries it)`; on BED-S at the design
point it is `NOT MEASURED` for the same reason and for want of a cell (B5).

**The $\kappa$-free bound the argmin clause is restated on, derived here.** $\hat Q\ge0$ with
$\|\hat Q\|_\infty=\max_{i\in T}r_i<1$ gives the Neumann bound
$\|(I-\hat Q)^{-1}\|_\infty\le1/(1-\|\hat Q\|_\infty)$, and on the causal class
$1-r_i=\sum_{a\in\mathcal A}\hat P_{ia}$ is the one-step escape mass of row $i$, so
$$\kappa\ \le\ \frac1{\varepsilon_{\rm esc}},\qquad
\varepsilon_{\rm esc}:=\min_{i\in T}\sum_{a\in\mathcal A}\hat P_{ia}\qquad(\text{DERIVED}),$$
and $\varepsilon_{\rm esc}>0$ for every finite-logit softmax row, since $0\in\mathcal A$ and a
causal softmax has full support on $j\le i$. This is a **different object** from the condition
number — the operator's own rows, not a norm of its inverse — read by a **different
instrument**, the row-sum pass verse 01.19 registers as reachable today at $0$ GPU-s, and it
returns a **different number**, $\varepsilon_{\rm esc}$ and not $7.9$. Every clause of this
verse that multiplies by $\kappa$ is stated at $1/\varepsilon_{\rm esc}$ from this repair, with
$\varepsilon_{\rm esc}$ `NOT MEASURED — needs the row-sum pass of 01.19's frozen census
extended to print $\min_{i\in T}\sum_{a\in\mathcal A}\hat P_{ia}$, $0$ GPU-s on a saved
$\hat P$`; the argmin gap clause reads
$2(\|\hat Q-Q\|_\infty+\|\hat R-R\|_\infty)/\varepsilon_{\rm esc}$ and is licensed once that
one number is printed, rather than waiting on an inverse-norm probe the canon does not
specify. The bound is honest about its own failure mode: a row whose escape mass is
$O(e^{-L})$ sends $1/\varepsilon_{\rm esc}$ to $O(e^{L})$, so the clause is a measured
quantity and never an assumption.

**Evidence.** DERIVED (four steps): (i) $(I-\gamma\hat P)(\hat z-z)=\gamma(\hat P-P_{\rm env})z$
and $\|(I-\gamma\hat P)^{-1}\|_\infty\le1/(1-\gamma)$ by the Neumann series on a
row-stochastic matrix (verse 01.16); (ii) $(I-\hat Q)(\hat q-q)=(\hat R-R)\mathbb 1+(\hat Q-Q)q$
with $\|q\|_\infty\le1$; (iii) the displacement identity `READ docs/CEQ_SHAPE.md:460-470`,
three independent RUNs at $1.03\times10^{-15}$, $1.2\times10^{-15}$, $1.36\times10^{-15}$;
(iv) the committor identity on BED-1's real sets, max-abs $0.0$, harmonic residual
$1.0408340855860843\times10^{-17}$ (`RUN`, `READ docs/CEQ_SHAPE.md:919-921`). The
"reachable set" clause is F2 as a theorem target (`later_boundary_unreachable` [M],
`READ docs/CEQ_SHAPE.md:1451-1456`).

**Mechanism.** D-2 (`MISTAKES.md:710`, the label is the arm's own resolvent — the verse says
so and files the shape-versus-softmax contrast VOID by registration); D-1 (`:677`, the
label class leaves the scalar-readout regime); V-24 (`:1658`, every identity ships its
mutilation battery); V-17 (`:855`, the bounds carry their units, $\kappa$ printed).

**Kill.** On the bed's own generator (Ruling 7: generator + seed + hash), with
$\hat P:=P_{\rm env}$ substituted, any of the four label identities reading above
$10^{-12}$ in max-abs on any of $512$ draws at the design point; instrument: the identity
script specified at `READ docs/CEQ_SHAPE.md:1606-1628 @ 99777ab` (card S-02, raise on a
missing key — a card of the record, not a verse of book 04, which carries no such
instrument); price $0$ GPU-s, one evening. Planted
negatives (each must fail at $O(1)$): declare $\mathcal A_k$ on the wrong set ($q$ moves
by $O(1)$, `READ docs/CEQ_SHAPE.md:387`); a non-causal $P$ (backward displacement
$0.0761$, $0.0868$, `RUN[M]`, `:1454`); a Gaussian $V$ against the constant-value plant
($\max|\Delta z|=0.1096$ at $s=32$, $1.127$ at $s=64$, `:474-476`).

**If killed.** The identity fails only if the generator's chain is not row-stochastic on
its absorbing rows or a set is undeclared (then $\det(I-Q)=0.0$, `:303`). Replacement
formula: the reach-avoid read on the *declared* sets with the undeclared absorbing
position added to $\mathcal A_{\rm sink}$, $q^{(\rm sink)}$ printed apart from $q^{(0)}$
(`:498-500`). Its Hypotheses: $0\in\mathcal A_{\rm sink}$ on $100\%$ of draws (census X-2,
`READ docs/CEQ_SHAPE.md:1441 @ 99777ab` and `:2335`).
Its Evidence: $\rho(Q)=0.692660$ declared against $1.000000$ undeclared (`RUN[P]`,
`READ docs/CEQ_SHAPE.md:1440 @ 99777ab`). Its Kill: a declared-sink draw with $\rho(Q)=1$ — a second undeclared absorbing
row (a zero gate, verse 01.17's `cut_makes_segment_head_absorbing`) — $0$ GPU-s. If that
fires: the exact class is stated per segment, with each segment carrying its own sink,
goal and constraint sets (`:445-450`), and its Kill is the same census line re-read per
segment; being a census line it is strictly cheaper than the identity battery.

**Terminal.** The read is a resolvent of the arm's own operator (ChaCAL's Eq. 5 with the
diagonal kept, `READ docs/CEQ_SHAPE.md:205-212`); exactness is a property of the class
and licenses no capability number; what the canon still licenses is the identity table
with its plants, and it withdraws every sentence in which "exact" is read as "accurate on
the bed".

---

### 01.2 — What $\hat P=P_{\rm env}$ costs in the softmax class

**Statement.** A causal softmax row with finite logits has full support on $j\le i$, so
$\hat P=P_{\rm env}$ holds exactly only when $P_{\rm env}$ has no zero entry below the
diagonal. For a DAG environment with non-edges, the best row-wise approximation with
logit margin $L$ between edges and non-edges leaks at most
$$\varepsilon_{\rm row}(L)\le s\,e^{-L}\qquad(\text{DERIVED: at most } s-1 \text{ non-edge terms, each } \le e^{-L}\text{ relative to an edge term}),$$
which is $6.0\times10^{-12}$ at $s=64$, $L=30$ ($e^{-30}=9.36\times10^{-14}$); and the
logit matrix has rank $\le d_{\rm model}$, so an arbitrary DAG pattern on $s$ positions is
representable at $d_{\rm model}\ge s$ (one-hot keys $k_j=e_j$, $q_i=\log P_{\rm env}(i,\cdot)$
on edges, $-L$ off them). The decidable claim at the planned geometry, and the only one this
verse asserts, is: **the DAG family drawn by BED-S at S-13 is not representable at
$d_{\rm model}=16$**, decided by the sign-rank of the drawn edge sets printed per seed —
representable at seed $\sigma$ iff $\mathrm{signrank}(E_\sigma)\le16$. The per-seed sign-rank
census is `NOT MEASURED — needs the sign-rank read on the drawn edge sets, unspecified in the
canon (no verse of book 04 carries it)`; the hedge "not guaranteed representable" is struck
under `CHARTER.md:161` and appears nowhere in this book. **The universal form is struck with
it, and the reason is printed:** a sentence of the form "not guaranteed representable for all
DAG families" is compatible with K-D2 firing on one family, so its firing would leave that
sentence standing and the If-killed's assertion would not follow from the kill — a kill that
cannot refute its own statement (`CHARTER.md:145-146`, V-3). What this verse asserts is the
per-family claim above and nothing wider.

**Hypotheses.** Regime S, $\beta=1$; node tokens carry out-adjacency as a multi-hot feature
(`READ docs/CEQ_SHAPE.md:1677-1683`). The leak bound is non-vacuous at every $L>\log s$;
the rank sentence is non-vacuous exactly when $d_{\rm model}<s$, which is the planned
geometry ($16<64$).

**Evidence.** DERIVED (two lines above). The record's own arithmetic that a zero is a
$-\infty$ logit and no finite gate produces one: `V16Domain.no_prefix_scan_represents_a_zero_gate`
(`lean/CEQ/V16Domain.lean:165`, `READ docs/CEQ_SHAPE.md:441-444`). The identification
reading $\|\hat P-P_{\rm env}\|_\infty$ per seed is `NOT MEASURED — needs the arena
(book 04, S-13)`.

**Mechanism.** V-25 (`MISTAKES.md:1954`, the hypothesis $\hat P=P_{\rm env}$ must be
censused on drawn cells, not assumed); D-2; V-10 (`:168`, K-D2 had an empty rejection
region on the undirected substrate and a non-empty one on the DAG).

**Kill.** K-D2 frozen: $\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on at least $6$ of $8$
seeds at the design point ($N=8$, seeds $0$–$7$, the arena's realised draws) — the arm
copied the environment and every contrast is a copy-versus-no-copy statement; instrument
S-13; price inside the $\approx34$ s pair (`[FITTED]`, `READ docs/CEQ_SHAPE.md:2295`), one
evening. Control: a plant with $d_{\rm model}=s$ and the one-hot construction above must
read below $10^{-3}$ (the copy is possible), and the $d_{\rm model}=16$ arm on a random DAG
must read above it on at least one seed, else the metric cannot see the rank wall.

**If killed** (the arm copies at $d_{\rm model}=16$ on $\ge6/8$ seeds). K-D2 firing refutes
the Statement as now written, because the Statement is the per-family claim and not a
universal negative: the drawn DAGs have sign-rank $\le16$ on those seeds and the family is
representable there.

**The replacement is re-derived at repair, because the earlier one died with the verse
(V-9).** The earlier replacement re-drew the family at higher sign-rank and moved the
capability sentence to the committor head with kill "$\kappa\varepsilon$ at or above the label
sd on $6$ of $8$ seeds". Both of its numbers are absent: $\kappa$ is `NOT MEASURED` (verse
01.1, and its condition-number surrogate is in different units), the label sd is `NOT
MEASURED` (B5, no BED-S cell), and the re-draw itself needs the sign-rank read this verse has
just marked unspecified in the canon — so the replacement waits on the same missing
identification reading K-D2 waits on and cannot fire where K-D2 does. It is replaced, not
renamed, by a reading on a **different object** — the bed's own generator specification, not
the arm and not the drawn edge sets — decided by a **different instrument** and returning a
**different number**:

> **Replacement — the feature-content reading.** BED-S's registered generator hands the arm
> the environment: *"Node tokens carry the out-adjacency as a multi-hot feature … the
> transition matrix is a deterministic function of the edge tokens"*
> (`READ docs/CEQ_SHAPE.md:1677-1683 @ 99777ab`, card S-11). Under that specification
> $P_{\rm env}(i,\cdot)$ is a function of the token at $i$ alone, so a copy at
> $d_{\rm model}=16$ is a **decoding** of a feature the bed supplies and not a
> **learnability** number about a rank wall, and the identification reading
> $\|\hat P-P_{\rm env}\|_\infty$ is reported as a decoding error with the feature named
> beside it.

*Its Hypotheses:* the arena draws from S-11's generator as registered, with the out-adjacency
multi-hot present in the token and no channel derived from a solve (S-11's own KILL line, D-2).
*Its Evidence:* the generator specification as quoted, `READ` at the pin; the decoding error
itself `NOT MEASURED — needs the first BED-S cell (book 04, 04.21)`.
*Its Kill:* the generator emitting tokens **without** the out-adjacency multi-hot — a read of
the registered spec and the arena's own `producer_cmd`, one line, no solve — while K-D2 still
fires on $\ge6$ of $8$ seeds: the copy is then not explained by the feature, the decoding
reading is withdrawn, and the sign-rank route is the only one left. *Its plant:* a generator
variant with the multi-hot deleted must read the spec line absent; a spec read that cannot
tell the two variants apart is not a spec read.
*Price:* $0$ GPU-s, one document read — **strictly cheaper than K-D2**, which needs a trained
cell, and it is decidable today on the registered specification whereas K-D2 is not (B5).
If that kill fires, the chain ends at the Terminal.

**Terminal.** Representability of the environment is a property of $(d_{\rm model},s)$
and of the DAG family, never of the read; the canon licenses the identification reading
as a learnability number and withdraws "the shape recovers the environment" wherever
$\|\hat P-P_{\rm env}\|_\infty$ is not printed beside it.

---

### 01.3 — The committor is the read at $\gamma\uparrow1$, with the discounted gap printed

**Statement.** For $i\in T$, $V=\mathbb 1_{\mathcal A_k}$:
$$O(\gamma)_i=E_i\big[\gamma^{\tau_k-1}\mathbb 1_k\big],\qquad
\lim_{\gamma\uparrow1}O(\gamma)_i=q^{(k)}_i .$$
**Two gaps, two objects, never interchanged.** The $(1-\gamma)q_i$ lower bound belongs to
the *undiscounted-by-one* read $z_i:=E_i[\gamma^{\tau_k}\mathbb 1_k]$, which is $\gamma
O(\gamma)_i$ and not $O(\gamma)_i$:
$$0\le q^{(k)}_i-E_i[\gamma^{\tau_k}\mathbb 1_k]=E_i\big[(1-\gamma^{\tau_k})\mathbb 1_k\big]\ge(1-\gamma)\,q^{(k)}_i$$
(DERIVED: $1-\gamma^{\tau}\ge1-\gamma$ for $\tau\ge1$). For the read $O(\gamma)$ the verse
asserts only what is true of it,
$$q^{(k)}_i-O(\gamma)_i=E_i\big[(1-\gamma^{\tau_k-1})\mathbb 1_k\big]\ \ge\ 0,$$
**with no non-trivial lower bound**: $\tau_k=1$ has positive probability whenever an
$\mathcal A_k$ position is one step from $i$, and on that event the integrand is $0$.
Counterexample to the $(1-\gamma)q_i$ form read on $O$: $P_{ii}=0.1$, $P_{ia}=0.9$ with
$a\in\mathcal A_k$, $\gamma=0.6$ gives $q_i=1$, $O_i=0.9/(1-0.06)=0.957447$, gap
$0.042553$, while $(1-\gamma)q_i=0.4$ — the inequality fails by a factor $9.4$ (DERIVED).
The committor head is the exact solve at $\gamma=1$ on the transient block and carries no
$\gamma$; the discounted read is a separate registered label at a bed constant
$\gamma_{\rm env}$.

**Hypotheses.** $P$ row-stochastic including absorbing rows; $\rho(Q)<1$ (verse 01.10);
$\tau_k=0$ on $\mathcal A_k$, $\gamma^\infty=0$. Non-vacuous at every $\gamma<1$; the gap
lower bound $(1-\gamma)q_i$ is $10^{-6}q_i$ at $\gamma=1-10^{-6}$ **on $E[\gamma^{\tau_k}]$
and on nothing else**, and the RUN gap on channel $\{15\}$ reads $1.42\times10^{-7}$ at
$\max_T$ (`RUN[J]`, `READ docs/CEQ_SHAPE.md:384-388`), which is the record's own line for
$E[\gamma^\tau]$ and is never quoted for the read.

**Evidence.** CITED `kemeny-1960-finitemarkov` [V-cat] and `grinstead-1997-probability`
Thm 11.6 [U] own $N=(I-Q)^{-1}$, $B=NR$; the read form is DERIVED (exchange of sums,
first-step analysis, monotone convergence; `READ docs/CEQ_SHAPE.md:1570-1576`, card J-D3);
read against $E[\gamma^{\tau-1}]$ to $8.33\times10^{-17}$ (`RUN[J]`, `RUN[M]`, `:385`).
Ruling: the committor head does not carry $\gamma$ because $q$ to $1\%$ at $\tau=8$ needs
$\hat\gamma\ge0.99857$, $1/(1-\hat\gamma)\approx697$ (`RUN[F]`, `:336-343`).

**Mechanism.** D-2; V-12 (`MISTAKES.md:189`, one absorbing target makes the label
constant — $K=1$ is illegal, `READ docs/CEQ_SHAPE.md:1681`); V-25; P-10 (`:1382`, the
owners are cited before the object).

**Kill.** The read-side plant: declare $\mathcal A_k$ on the wrong set and read
$\max_T|q^{\rm wrong}-q|<10^{-3}$ on any draw — the instrument cannot see its own label;
$0$ GPU-s, half an evening, on BED-1's real sets ($A=[0]$, $B=[1]$, $|T|=9$) and on one
BED-S draw. The must-fire perturbation on BED-1 drives the residual from $1.04\times10^{-17}$
to $10^{-6}$, ratio $9.6\times10^{10}$ (`READ docs/CEQ_SHAPE.md:918`).

**If killed.** A wrong-set read that moves the label by less than $10^{-3}$ means the two
sets are unreachable from the query (F2) — the bed, not the identity, is at fault.
Replacement: the reachability census X-4 (`READ docs/CEQ_SHAPE.md:3153 @ 99777ab`; every set
before the query, query in $T$) is
made an admission line with the discard rule "a set after the query reads $q=0.0$ exactly,
discard" (`:1670-1673`). Its Kill: X-4 below $100\%$ on $512$ draws ($0$ GPU-s). If that
fires, the placement convention S-10 (the one card of this list book 04 does carry) is
re-drawn with sets forced before the query and the
same line re-read; cheaper again, being one generator flag.

**Terminal.** The committor is Kemeny–Snell's absorption probability read through the
arm's $\hat P$; the canon licenses it as the exact head at $\gamma=1$, licenses
$(1-\gamma)q_i$ as a lower bound on the gap of $E_i[\gamma^{\tau_k}\mathbb 1_k]$ and on no
other object, and withdraws every sentence that binds that bound to the read
$O(\gamma)_i=E_i[\gamma^{\tau_k-1}\mathbb 1_k]$, for which the only licensed statement is
$q^{(k)}_i-O(\gamma)_i\ge0$.

---

## Part II — D-APPROX: from "cannot represent" to "cannot fit below $\varepsilon$"

### 01.4 — The $\mathrm{hop}_k\to$ committor reduction, written

**Statement.** Every $\mathrm{hop}_k$ instance on $n$ tokens (each token $j$ carries a
pointer $\pi(j)<j$ to an earlier token; $\mathrm{hop}_k(i)=\pi^{k}(i)$) is a committor
instance on a DAG of $s'=(k+1)n+1$ positions with $K=n$ singleton absorbing sets **and one
dedicated sink**, and
$$\mathrm{hop}_k(i)=\arg\max_{j\in[n]}\;q^{(j)}_{(i,0)},\qquad q^{(j)}_{(i,0)}\in\{0,1\}\ \text{exactly on } P_{\rm env}.$$
*Construction.* Position $0$ is the sink and nothing else: $\mathcal A_{\rm sink}=\{0\}$,
$P_{00}=1$. Positions $(j,\ell)$ for layers $\ell=k,k-1,\dots,0$ are laid out after it, so
layer $k$ occupies $[1,n+1)$ and layer $0$ occupies $[kn+1,(k+1)n+1)$; edges
$(j,\ell)\to(\pi(j),\ell+1)$ point to earlier positions; layer-$k$ positions are absorbing,
$\mathcal A_j=\{(j,k)\}$ at index $1+j$; the walk from $(i,0)$ is deterministic and reaches
$(\pi^k(i),k)$ in exactly $k$ steps. **The sink is disjoint from every $\mathcal A_j$ by
construction**, which is what `Definition 3` requires
($\mathcal A=\mathcal A_{\rm sink}\sqcup\mathcal A_0\sqcup\dots\sqcup\mathcal A_K$,
`READ docs/CEQ_SHAPE.md:295-297 @ 99777ab`): the earlier layout, which put $\mathcal A_0$ at
position $0$, gave position $0$ two disjoint labels at once and made the answer channel and
the leak channel the same set on the $1/64$ of instances whose answer is token $0$ under
uniform $\pi^k$. On the arm's softmax class with logit margin $L$ (edge logit $0$, every
other earlier position and the self entry at $-L$),
$$q^{(\mathrm{hop}_k(i))}_{(i,0)}\;\ge\;1-\frac{k\,s'\,e^{-L}}{1-s'e^{-L}},$$
so the argmax reads $\mathrm{hop}_k$ whenever the right side exceeds $\tfrac12$: at
$n=64$, $k=8$, $s'=577$, $L=30$ the loss is $\le4.4\times10^{-10}$ (DERIVED:
$577\cdot9.36\times10^{-14}=5.40\times10^{-11}$ per step, $\times8=4.32\times10^{-10}$).

**Hypotheses.** Two readings, kept apart because they need different substrates. **(a) The
exact reading at $\gamma=1$** is on the *strict* pointer DAG: no self entries, every row of
a transient position supported on its single pointer target, so the walk is exactly $k$
steps and $q\in\{0,1\}$; this is the reading `hop_layered_committor` is stated on (verse
01.17), and it is regime N on the transient block with the absorbing rows added, not regime
S. **(b) The $\varepsilon$-clause** is on the softmax class, regime S, with the self entry at
$-L$ and the dedicated sink $\{0\}$ absorbing the leak; the arm's logits attain margin $L$ —
which the arm **learns**, so the clause is a statement about a *representable* $\hat P$, not
about a trained one. The two are never mixed in one sentence: (a) supplies $q\in\{0,1\}$ and
(b) supplies the $\varepsilon$ around it. Non-vacuous for every
$L>\log(2ks')=\log(2\cdot8\cdot577)=9.131$ at the instance above. The stall mass
(J-D8's blocker, `READ docs/CEQ_SHAPE.md:2156-2162`) is bounded because on a deterministic
pointer graph a self-loop delays the walk and never changes the absorbing target; only
leak to non-pointer positions does, and each step's leak is $\le s'e^{-L}$ with the expected
number of steps $\le k/(1-s'e^{-L})$.

**Evidence.** DERIVED (the construction and the union bound above). The reduction was
`NOT FOUND` in the literature by `sweep_expressivity.md` §3.5 (`READ
docs/sources/sweep/sweep_expressivity.md:495-511`) and "unwritten" at `READ
docs/CEQ_SHAPE.md:797`; it is written here. Direction: committor $\supseteq\mathrm{hop}_k$
(a solver for the committor on this family solves $\mathrm{hop}_k$); the converse is not
claimed and not needed.

**Mechanism.** P-4 (`MISTAKES.md:332`, claimed scaffolding that did not exist — the
reduction is now text); V-3 (`:72`, the stall-mass bound is stated, not assumed away);
P-10; D-2 (the instance family is the oracle's, and BED-S's random DAG at $K=2$ is **not**
this family — the skyline for BED-S remains by analogy, verse 01.8).

**Kill.** On $512$ random $\mathrm{hop}_8$ instances at $n=64$ (seed rule: instance seed
$=$ draw index), the exact oracle solve on $P_{\rm env}$ returning $\arg\max_jq^{(j)}\ne\pi^8(i)$
on any instance; $0$ GPU-s, one evening, float64 CPU. Planted negative: reverse one
pointer to point forward — F2 fires, the channel reads $q=0.0$ exactly and the argmax is
undefined; the instrument must raise (V-16, `MISTAKES.md:828`).

**If killed** (the exact solve mis-reads the pointer). The layered layout is wrong — an
edge points forward. Replacement: the same construction with layer order verified by the
census line "every edge $(j,\ell)\to(\pi(j),\ell+1)$ has a smaller position index", which
is a $0$ GPU-s decidable predicate on the layout; its Kill is that predicate reading false
on any instance, strictly cheaper than the solve. If the $\varepsilon$-clause alone fails
(a trained arm's margin $L<\log(2ks')$), the reduction stands on $P_{\rm env}$ and the
depth statements of verses 01.5 and 01.8 transfer to the *oracle's* family only; the
replacement is the sentence "obstruction 2 is a theorem on the layered family and a
lineage argument on BED-S", with the margin $\hat L=\min_{\rm edges}\ell_{ij}-\max_{\rm non\text{-}edges}\ell_{ij}$
printed per trained cell (`NOT MEASURED — needs the margin probe, **unspecified in the
canon**: book 04 carries verses 04.1–04.21 and no margin probe among them, so this kill is
unreachable until book 04 ships one`).

**Terminal.** The committor class contains pointer chasing at $K=n$ and $s'=(k+1)n+1$; on
BED-S ($K=2$, $s=64$) the canon licenses only "hops need depth" as lineage with the
inequalities of verse 01.5 printed, and withdraws the phrase "theorem about the shape's
bed".

---

### 01.5 — **OPEN** — Obstruction 3 through the reduction: Peng's Theorem 1 with its vacuity line

*Marked OPEN at repair: the verse's only kill is an act book 04 registers as outside this
canon's no-fetch rule, so no link of the chain can fire inside the canon. The Terminal below
is the sentence in force today.*

**Statement.** Function composition $f(g(x))$ on $n$-element domains is $\mathrm{hop}_2$;
by verse 01.4 it is a committor instance at $K=n$, $s'=3n+1$. If `peng-2024-transformer-limitations`
Thm 1 holds as recorded — a single $H$-head softmax layer with embedding $d$ at $p$ bits
errs with probability $\ge R/(3n\log n)$, $R=n\log n-H(d+1)p>0$ — then a depth-1
single-head softmax at $d=16$, $p=32$ errs on the 2-hop committor family with probability
at least the table's value, while the exact solve reads $0$:

| $n$ | $n\log_2n$ | $H(d+1)p$ | $R$ | error $\ge$ | status at $(d,p)=(16,32)$ |
|---|---|---|---|---|---|
| $64$ | $384$ | $544$ | $-160$ | — | **vacuous** (the planned geometry) |
| $128$ | $896$ | $544$ | $352$ | $0.1310$ | non-vacuous, $s'=385$, $K=128$ |
| $256$ | $2048$ | $544$ | $1504$ | $0.2448$ | non-vacuous, $s'=769$, $K=256$ |
| $64$, $p=16$ | $384$ | $272$ | $112$ | $0.0972$ | non-vacuous at half precision |
| $64$, $d=8$ | $384$ | $288$ | $96$ | $0.0833$ | non-vacuous at half width |

**Hypotheses.** Peng's model class (single softmax layer, communication-complexity
argument), read at the abstract: `CITED [V]`, `[V-eq]` **owed** — the theorem's own
hypotheses on how $f,g$ are presented were not read at the body (`READ
docs/sources/sweep/sweep_expressivity.md:125-137`). Under L-EQ this verse is therefore a
DERIVED conditional ("if Thm 1 as recorded") and is not load-bearing until the body is
read. The geometry column is DERIVED with $\log_2$.

**Evidence.** DERIVED arithmetic; the vacuity at $s=64$ is the record's own (`READ
docs/CEQ_SHAPE.md:710-719`, `:2952` — the earlier pin `:2947-2950` did not resolve to
$384<544$ at `99777ab` and is corrected here); the reduction is verse 01.4.

**Mechanism.** V-25 (`MISTAKES.md:1954`) — the inequality is printed so the reader sees
$0\%$ of the planned geometry admitted; P-10 (`:1382`) — the theorem is not cited beyond
its recorded statement; V-17 — the error is in the theorem's own units (misclassification
probability on its instance family), never carried to BED-S's NRMSE.

**Kill — UNREACHABLE, and the verse stands OPEN on that account.** The only condition that
refutes the conditional is reading Thm 1 at the arXiv LaTeX source and finding its
hypothesis to differ from the recorded $n\log n>H(d+1)p$, or its instance presentation to
be incompatible with the layered DAG of 01.4. That act is **outside this canon's rules**:
`04_BEDS_AND_INSTRUMENTS.md` verse **04.12**'s Evidence (`:193` at repair time; cited by
verse id because book 04's line numbers move under its own repairs) closes with *"No LaTeX source has been
read: `NOT MEASURED` — needs one fetch of the arXiv source, an author action outside this
canon's no-fetch rule"*. It is therefore repriced from the earlier "$0$ GPU-s, one evening"
— which read as if the act were inside — to `NOT MEASURED — needs one author fetch of
arXiv:2402.08164, an act no writer, refuter or repairer of this canon may perform`. Under
`CHARTER.md:145-148` a kill that cannot fire on the bed as registered is not a kill, so this
verse carries **no reachable kill today** and its Terminal is in force now, not after a
run. The fetch is registered as an author action with its own row in book 05 and not as
this verse's kill. Control, for whenever the fetch happens: the same re-read on
`fagnou-2024-chacal` Eq. 5, whose recorded form is already `[V-fetched]` — a re-read that
cannot reproduce a known equation is a broken instrument.

**If killed — one lineage link, then the Terminal.** The two unconditional theorems are
collapsed into a **single** link, because neither was strictly cheaper nor strictly more
decisive than the Peng re-read (`CHARTER.md:139-141`) and both died to the same act.
*The link:* `sanford-2024-inductionheads` Thm 1 and `chen-2024-multilayer` Thm 1.1 are
cited together as **lineage only**, with their vacuity lines printed and no kill of their
own. Sanford Thm 1 is a $\mathrm{hop}_1$ bound and is labelled $k=1$, never $t^\star=2$:
through 01.4 at $k=1$ it is a committor instance at $s'=2n+1$, and $h\,m\,p=\Omega(n)$ bites
at $n>h\,m\,p/c$ with $c$ the theorem's own unstated constant. At $h=1$, $m=16$, $p=32$ the
product is $512$, so the geometry at which it bites is $n>512/c$, i.e. $s'>1024/c+1$, and
$c$ is `NOT MEASURED — needs the constant read from Thm 1's body, the same author fetch`;
the earlier "first violated at $n>512$" silently set $c=1$ and is struck, as is the earlier
"i.e. $s\ge1024$ at $t^\star=2$", which carried the hop index $k$ into the bed's $t^\star$
and set $c=1$ in the same breath. The position count is $s'=2n+1$ and **not** $2n$: verse
01.4's construction lays $(k+1)n$ layer positions after a **dedicated sink** at position $0$,
so at $k=1$ the count is $2n+1$; the extra position is the sink the reduction's repair added
to keep the answer channel and the leak channel disjoint, and printing $2n$ would understate
the geometry by one position. Chen Thm 1.1 at
$L=1$ needs $n\ge512^{16}=2^{144}$, vacuous at every geometry any device can run.
*Its Hypotheses:* none beyond citation. *Its Evidence:* `[V]`, `[V-eq]` owed for both.
*Its Kill:* none — lineage carries no kill, which is why the chain terminates here in one
link rather than three.

**Terminal — in force now.** At $s=64$, $d=16$, $p=32$ no unconditional "softmax cannot" is
licensed for the 2-hop committor, and no verse of this canon can change that by any act it
is permitted to perform. The canon licenses the table above as the geometry at which an
A1-form statement would become theorem-shaped ($s'\ge385$ at $K=n$), and withdraws every
"cannot" sentence at the planned geometry in favour of Bet D (verse 01.8) with the counter
as the point estimate (D-CALIB).

---

### 01.6 — D-APPROX by capacity: the per-position sketch, and the hypothesis it needs

**Statement.** A depth-1 softmax reads, at query $i$, a $d$-dimensional aggregate
$O_i=\sum_{j\le i}P_{ij}V_j$ whose weights depend on $(x_i,x_j)$ and whose values on
$x_j$ alone; at $p$ bits the aggregate carries $d\,p=512$ bits at the planned geometry.
The readout sees $(x_i,O_i)$, not $O_i$ alone, and $x_i$ carries the out-adjacency multi-hot
of 01.2's Hypotheses, at most $s=64$ further bits, so the **joint** sketch carries at most
$d\,p+s=576$ bits. The downstream graph of $i$ on $s=64$ positions has
$2^{s(s-1)/2}=2^{2016}$ configurations, so at least $2^{2016-576}=2^{1440}$ graphs share
each joint sketch value ($2^{1504}$ is the count for $O_i$ alone and is the wrong budget for
a readout that also sees $x_i$). Any readout of
$(x_i,O_i)$ therefore misfits the committor by at least $\varepsilon$ on some instance iff
the label separates the sketch fibres at resolution $\varepsilon$:
$$\text{err}_\infty\ \ge\ \varepsilon\quad\text{whenever}\quad
\Pr_{G,G'\,\text{same sketch}}\big[|q_i(G)-q_i(G')|>2\varepsilon\big]>0 .$$
By pigeonhole on the *label* alone the bound is vacuous: the per-move tensor has
$m(K+2)=32$ coordinates and $(1/\varepsilon)^{32}>2^{512}$ needs $\varepsilon<2^{-16}=1.5\times10^{-5}$;
the full vector on $T$ has $|T|(K+2)$ coordinates with $|T|\in\{56,\dots,60\}$ — at
$|\mathcal A_\bullet|\in\{1,2\}$ over the four sets $\mathcal A_{\rm sink},\mathcal A_0,
\mathcal A_1,\mathcal A_2$ the total $|\mathcal A|$ ranges over $\{4,\dots,8\}$, so
$|T|=64-|\mathcal A|$ ranges over $\{56,\dots,60\}$ and the earlier $\{57,60\}$ excluded
values the stated geometry admits — hence $|T|(K+2)\in\{224,\dots,240\}$ and the requirement
is
$$\varepsilon<2^{-512/224}=0.2050\ \ \text{to}\ \ 2^{-512/240}=0.2278,$$
each endpoint printed against the expression that produces it (the earlier line transposed
them). This bites **only if** the committor image is full-dimensional at that scale, which
is not a theorem.

**Hypotheses.** Depth 1, one head, per-position readout, $d=16$, $p=32$; the sketch
model excludes positional side channels wider than $x_i$. Non-vacuous exactly when the
fibre-separation probability above is positive — a measured quantity.

**Evidence.** DERIVED (counting). The fibre-separation probability is
`NOT MEASURED — needs the collision instrument, **unspecified in the canon**: book 04
carries verses 04.1–04.21 and no collision instrument among them (a grep for "collision
instrument" in `04_BEDS_AND_INSTRUMENTS.md` counts $0$), so the shape stated here — draw DAG
pairs, compute the depth-1 sketch at fixed random weights, read the fraction of same-sketch
pairs with $|\Delta q_i|>2\varepsilon$ — is a specification this verse writes and book 04
does not carry`. That existence of collisions does not by itself force label
error is the content of Peng's communication argument (verse 01.5), not of counting.

**Mechanism.** B7 (`CHARTER.md` §5) — the record's largest unclosed gap was that no bound
followed "cannot represent"; this verse states the bound with the one hypothesis that
makes it bite, so it cannot be claimed by accident (V-10, `MISTAKES.md:168`: the label
pigeonhole is satisfied by construction and is printed as vacuous); V-17.

**Kill.** The collision instrument reading fibre-separation at $\varepsilon=0.05$ below
$5\%$ of same-sketch pairs on $512$ drawn BED-S pairs at the design point: the capacity
bound is vacuous at the planned geometry; $0$ GPU-s (CPU float64), one evening — but
**unreachable until book 04 carries the instrument** (Evidence above). **Control, and its
two defects named.** The must-fire plant is a *calibration* plant and not an import from
01.5: the instrument is run on a synthetic pair family with a **constructed** separation
fraction $\phi_0$ (two DAGs agreeing on the sketch by construction and differing in $q_i$ by
a chosen $\Delta$), and must read $\phi_0$ to within $\pm0.02$ on $512$ pairs; an instrument
that cannot recover a planted separation fraction cannot certify a measured one. The earlier
control — "separation above $13\%$ on the layered $\mathrm{hop}_2$ family at $n=128$" —
imported 01.5's cell $0.1310=352/2688$ across two boundaries and both are recorded here
rather than repaired away. *(i) Log base.* That cell is DERIVED with $\log_2$; under natural
log the same row reads $R=128\ln128-544=77.4$ and error $\ge77.4/(3\cdot128\ln128)=0.0415$,
so a working instrument reading $0.05$ would be declared broken by a factor $3.2$. 01.5's
Hypotheses grade Peng `[V]` with `[V-eq]` **owed** — the body was not read — so the base is
not settled, and if that control is ever run it is run as "under $\log_2$ the threshold is
$0.131$; under $\ln$ it is $0.0415$", never as one number. *(ii) Units.* A collision
fraction over same-sketch DAG pairs and a lower bound on the misclassification probability
of the best readout are two different quantities, and **no theorem in this canon relates
them**; the import is therefore withdrawn as a control and kept only as the sentence just
written.

**If killed.** The replacement bound is the trained one: Bet D, registered at
`READ docs/CEQ_SHAPE.md:2236 @ 99777ab` with **prediction** "the depth-5 stack within
$\mathrm{MDE}_8$ of the shape's argmin accuracy — the honest control holds" and **counter**
"short by more than $\mathrm{MDE}_8$, sign logged: the bed needs something the hop
construction does not supply". Its Kill is
$\mathrm{acc}_{\rm shape}-\mathrm{acc}_{\rm sky}$ at $N=8$ against the realised paired sd
($\approx64.8$ s `[ASSUMED]`) — more decisive than the counting bound because it is measured
on the bed. **The two branches, in the direction the registration gives them** (the earlier
text read them backwards and contradicted 01.8's Kill on the same registered bet): if Bet D
reads its **prediction** — the skyline within $\mathrm{MDE}_8$ — then the depth skyline
**fits**, the D-APPROX sentence for BED-S is "**what survives on BED-S is exactness
($\delta=0$) and the cost sentence, and no fit sentence is licensed**" — the word "advantage"
appears nowhere, being a comparative outside A1/A2/T1/T2 with no bed, floor, $N$ or matched
count attached (`CHARTER.md:106-109`) — the **A2 sentence is withdrawn** and the
A1 sentence is reduced to verse 01.7. If Bet D reads its **counter** — the skyline short by
more than $\mathrm{MDE}_8$ — the A2 form has its first evidence on this bed and the counting
bound is **superseded, not replaced**: the measured shortfall stands in front of it and the
counting line is kept only as the reason the measurement was worth taking. Under D-CALIB the
counter is the point estimate, so the second branch is the one the canon plans against.

**Terminal.** "Cannot represent" becomes "cannot fit below $\varepsilon$" only where a
fibre-separation number or a communication-complexity theorem supplies the constant; at
$s=64$, $d=16$, $p=32$ neither is in the record, and the canon withdraws every
"cannot fit" sentence there while licensing the two instruments that would earn one.

---

### 01.7 — The iteration skyline computes the resolvent with error exactly $\gamma^{K'+1}/(1-\gamma)$

**Statement.** A stack of $K'$ blocks that reuses one head $P$ with a two-channel residual
stream (channel $C^{(t)}=(\gamma P)^tV$, channel $S^{(t)}=\sum_{u\le t}(\gamma P)^uV$;
each block computes $C^{(t+1)}=\gamma PC^{(t)}$ and $S^{(t+1)}=S^{(t)}+C^{(t+1)}$, a
linear map of the attention output and the residual) outputs $S^{(K')}$, the $K'$-truncated
Neumann sum, and
$$\Big\|(I-\gamma P)^{-1}-\sum_{t\le K'}(\gamma P)^t\Big\|_\infty=\frac{\gamma^{K'+1}}{1-\gamma}\quad(\text{equality}),\qquad
\|z-S^{(K')}\|_\infty\le\frac{\gamma^{K'+1}\|V\|_\infty}{1-\gamma}.$$
**The depth count, corrected at repair, and the letter it is written in.** $S^{(K')}$ needs
$C^{(0)}=S^{(0)}=V$ in the residual stream at initialisation — $V$ is written into the stream,
not computed by a block — and then exactly $K'$ blocks, each computing
$C^{(t+1)}=\gamma PC^{(t)}$ and accumulating. The stack is therefore **$K'$ blocks**, and the
earlier "depth-$(K+1)$ stack" was off by one in the direction that flattered the read (it
charged the skyline one block more than the construction needs) *and* wrote the truncation
order in the letter this book's Notation reserves for the constraint-set count, fixed at $K=2$
at the design point. Both are corrected: the order is $K'$ throughout this verse and verse
01.16, matching `08_ARCHITECTURE.md` verse 08.11, which writes the certificate
$\delta=\gamma^{K'+1}/(1-\gamma)$ and states in its own Hypotheses that $K'$ and $K$ are two
objects. The title's exponent, which read $\gamma^{L}/(1-\gamma)$ against a body reading
$\gamma^{K+1}/(1-\gamma)$ — a third letter for the same order — now reads $\gamma^{K'+1}/(1-\gamma)$.
This is the A1-form statement in theorem shape: one solve does what $K'$ reused hops do,
with the hops' shortfall printed:

| $\gamma$ | $K'=2$ | $K'=4$ | $K'=6$ | $K'=16$ |
|---|---|---|---|---|
| $0.6$ | $0.540$ | $0.194$ | $0.0700$ | $4.23\times10^{-4}$ |
| $0.7$ | $1.143$ | $0.560$ | $0.2745$ | $7.75\times10^{-3}$ |
| $0.9$ | $7.29$ | $5.90$ | $4.78$ | $1.67$ |

(matrix $\infty$-norm; multiply by $\|V\|_\infty$ for the vector bound; DERIVED, one
convention for every cell: $\gamma^{K'+1}/(1-\gamma)$, the quantity the Statement's equality
names.) **The $\gamma=0.7$ row is reprinted at repair.** Three of its four cells previously
carried $\gamma^{K'+1}$ — the $\Pi_\gamma$ residual — while the fourth and both other rows
carried $\gamma^{K'+1}/(1-\gamma)$: $0.343$, $0.168$, $0.0824$ are $0.7^3$, $0.7^5$, $0.7^7$
and are short of the Statement's quantity by the factor $1/(1-\gamma)=3.3\overline{3}$,
whereas $7.75\times10^{-3}=0.7^{17}/0.3$ was already divided. The record prints the correct
value for the identical $(\gamma,K')=(0.7,2)$: bind B-I's $\gamma=0.7$ plant reads
$119.37/1.143$ (`READ docs/CEQ_SHAPE.md:3116 @ 99777ab`), $1.143=0.7^3/0.3$, against the
book's former $0.343$. The corrected row is $0.7^3/0.3=1.143333$, $0.7^5/0.3=0.560233$,
$0.7^7/0.3=0.274514$, $0.7^{17}/0.3=7.754\times10^{-3}$ (DERIVED). At $\gamma=0.7$, $K'=2$ the
truncation shortfall therefore **exceeds** $\|V\|_\infty$: the two-hop iteration skyline is
not a fellow approximator at that horizon, which the mis-divided cell hid.

**Hypotheses.** $P\ge0$ row-stochastic including identity rows, $\gamma\in[0,1)$; the
stack reuses $P$ (the *iteration* skyline). A freely trained depth-$L$ stack with $L$
different heads is **not** covered: for it the only bounds are verse 01.5's (vacuous at
$s=64$) and Bet D. Non-vacuous at every $K'$ and $\gamma>0$; at $\gamma=0.6$, $K'=16$ the
shortfall $4.23\times10^{-4}$ is below any label sd the bed will show, so the InfSA-style
$K'=16$ control is a fellow approximator there and the exactness delta is invisible at that
$\gamma$ (the honest reading of R-SKY).

**$\gamma_{\rm env}$ carries no number, and this verse no longer decides anything "at
$\gamma_{\rm env}$".** The record names it and never prints it: `READ
docs/CEQ_SHAPE.md:341-342 @ 99777ab` reads *"the discounted read $E_i[\gamma^{\tau-1}\mathbb
1_k]$ is a separate registered label at a bed constant $\gamma_{\rm env}$"*, and a grep for a
value finds none in the record or in any book of this canon; it is
`NOT MEASURED — needs the BED-S registration triple to fix $\gamma_{\rm env}$ before the first
draw (book 04, the D-4 admission block), a $0$ GPU-s registration act`. The quantity this
verse decides varies over that unset constant by a factor $13.5$ — $0.540$ at
$(\gamma,K')=(0.6,2)$ against $7.29$ at $(0.9,2)$, both cells of the table above (DERIVED) —
so a threshold "at $\gamma_{\rm env}$" is not frozen in the sense M-2 requires. Every clause
of this verse is therefore registered at the **two corners the record has run**, $\gamma=0.6$
and $\gamma=0.9$, with both shortfalls printed, and at no other $\gamma$ until the constant
is fixed.

**Evidence.** DERIVED (`meyer-2000-matrix`, `horn-2013-matrix`: the tail is entrywise
non-negative with row sums $\sum_{t>K}\gamma^t$; the $\infty$-norm of a non-negative matrix
is its largest row sum); `RUN` residual $0.007754350466241788$ against
$\gamma^{17}/(1-\gamma)=0.007754350466240224$ at $K'=16$, $\gamma=0.7$
(`READ docs/CEQ_SHAPE.md:853-855`); the layer-by-layer TD reading is lineage only:
`wang-2024-incontext-td` [V], `xie-2026-softmax-rl` [V] at the abstract
(`READ docs/CEQ_SHAPE.md:773-777`), `yang-2024-looped`, `gasteiger-2019-appnp` [V].

**Mechanism.** R-SKY (`CHARTER.md` §4; `READ CEQ_V16_CONTRACT.md:209 @ 99777ab`) — the
skyline is stated beside the read; V-3 (`MISTAKES.md:72`) — the bound is an attained
identity and is labelled so; D-1 — this is the "one solve versus $L$ hops" sentence in
the only form the record licenses, a rate on the reused head.

**Kill — the learned-stack condition is struck as unable to refute the Statement; the
arithmetic kill the Statement admits replaces it.** The previous kill read: "On BED-S at
$\gamma_{\rm env}$, the depth-$5$ *learned* stack (Bet D's arm) reading a $z$-channel NRMSE
below the exact solve's by more than $\mathrm{MDE}_8$ on at least $6$ of $8$ seeds". It is
struck on two counts, both printed. **(1) It cannot refute the Statement.** The Statement is
an **attained matrix identity** about the truncated Neumann sum of $(I-\gamma P)^{-1}$; no
outcome of a training run — a depth-$5$ learned stack reading any NRMSE whatever — can make
$\|(I-\gamma P)^{-1}-\sum_{t\le K'}(\gamma P)^t\|_\infty$ differ from $\gamma^{K'+1}/(1-\gamma)$,
because the two quantities are computed from $(\gamma,K',P)$ and never from a weight
(`CHARTER.md:145-146`: a kill that cannot fire against the statement it is attached to is not
a kill). **(2) Its threshold is not frozen.** It was registered "at $\gamma_{\rm env}$", and
$\gamma_{\rm env}$ carries no value anywhere in the record or the canon (Hypotheses above), so
the number the kill decides ranges over a factor $13.5$ — $0.540$ at $(0.6,2)$ to $7.29$ at
$(0.9,2)$ — over an unset constant (M-2). The learned-stack contrast is not deleted: it is
**Bet D's contrast and belongs to verse 01.8**, which registers it against the record's own
threshold, and it is cited there rather than duplicated here.

**The form the contrast is written in, and the bare comparative that is struck with it
(`CHARTER.md` §1, R-SKY, D-1).** The struck kill's phrasing — a learned stack reading below
the exact solve, the bed and $\mathrm{MDE}_8$ named and nothing else — is a comparative outside
A1/A2/T1/T2: it printed neither the floor nor the matched parameter count, which
`CHARTER.md:106-109` strikes at refutation. The sentence this verse licenses in its place is
the **A1** form, and it carries four things in one clause or it is not written: the bed
(BED-S at $t^\star=8$, $m=8$, $K=2$, $N=8$); the floor (**L-FLOOR**, the argmin
zero-information floor $1-\max_a\hat\pi(a^\star)$ at the realised class distribution,
`NOT MEASURED` because $\hat\pi$ has no cell (B5) — $0.875$ is the uniform-$\hat\pi$ instance
at $m=8$ and nothing wider); the **matched parameter count** ($4{,}769$ for the shape at the
design point, against book 04 clause 3b's reduced-width count fixed within $0.0960$ per cent,
which is `NOT MEASURED` because 3b has no width and no cell); and $\mathrm{MDE}_8$ at the
realised paired sd, itself `NOT MEASURED` until 04.21 ships $\sigma_d$. Three of the four are
`NOT MEASURED`, so **no A1 value is printed by this verse and no fit sentence is licensed
here**. What this verse prints instead is the truncation shortfall
$\gamma^{K'+1}/(1-\gamma)$, which is a function of $(\gamma,K',P)$ and is an identity rather
than a comparison — $0.540$ at $(0.6,2)$, $7.29$ at $(0.9,2)$ — beside which the A1 clause
above is a route and not a number.

> **Frozen.** Form $P$ as a drawn causal row-stochastic operator at $s=64$ (regime S,
> identity rows on $\mathcal A$), compute $(I-\gamma P)^{-1}$ and $\sum_{t\le K'}(\gamma P)^t$
> in float64 at **both registered corners** $\gamma=0.6$ and $\gamma=0.9$ and at every
> $K'\in\{2,4,6,16\}$, and read
> $\big|\;\|(I-\gamma P)^{-1}-\sum_{t\le K'}(\gamma P)^t\|_\infty-\gamma^{K'+1}/(1-\gamma)\;\big|$
> **relative** to $\gamma^{K'+1}/(1-\gamma)$. Any of the eight $(\gamma,K')$ pairs reading a
> relative departure above $10^{-12}$, or any of the twelve printed table cells differing from
> $\gamma^{K'+1}/(1-\gamma)$ by more than $10^{-12}$ relative: the word "equality" is withdrawn
> from the Statement, the certificate is demoted to the Neumann upper bound, and the A1 rate is
> withdrawn as a rate.

Instrument: float64 formation of a $64\times64$ resolvent and a partial sum, CPU, no cell, no
mask, no bed, no arm; price $0$ GPU-s, half an evening. **It is reachable today and it fires on
a real defect: the table's $\gamma=0.7$ row as this book printed it before repair reads $0.343$
against $1.143$, a relative departure of $0.7$** — the second clause would have fired on this
book's own page. This kill is a different object from verse 01.16's — 01.16 recomputes the
**scalar** closed form against the record's two **banked residual numbers**, while this reads
the **matrix** $\infty$-norm of a formed resolvent against the same closed form on a drawn
$64\times64$ operator — and the two are kept apart on that ground. Planted negatives, each of
which must fire: the rows-$1.5$ non-stochastic $P$ at $\gamma=0.6$, $K'=2$ must **exceed**
$0.54$, reading $7.29$ (`RUN[M]`, verse 01.16's Evidence) — row stochasticity is the
hypothesis, and an instrument that reports equality on rows summing $1.5$ is reading the
formula and not the matrix; and a $P$ with one row scaled by $1+10^{-6}$ must read a relative
departure above $10^{-12}$ at $\gamma=0.9$, $K'=2$, the smallest perturbation the gate must
see. Must-not-fire half: the record's own banked pair at $K'=16$, $\gamma=0.7$, relative
$2.0179\times10^{-13}$ (verse 01.16), inside the gate.

**If killed** (the matrix equality departs by more than $10^{-12}$ relative at either
corner). The A1 sentence is replaced by the cost sentence: one solve of depth $s$ at
$+s^2d/2$ MACs against $L$ layers (`READ docs/CEQ_SHAPE.md:583-590`), a T1 statement owned
by book 03; its Kill is the cost law's exponent interval (K-9, $206$–$537$ GPU-s, band),
more decisive because it is measured on the certified device rather than on a formed matrix.
If that dies, the sentence "the shape's hops are exact and certified rather than learned"
survives as a property with $\delta$ printed (verse 01.16), and the chain ends.

**Terminal.** A deeper softmax stack computes the same resolvent by iteration with a
printed rate; "beats softmax" is not a sentence this canon writes; the licensed sentence
is exactness against a truncated read, and it is withdrawn at any $\gamma$ where the
truncation shortfall is below the label sd.

---

### 01.8 — Obstruction 2 restated: the skyline depth is a theorem on the layered family and an analogy on BED-S

**Statement.** `sanford-2024-logdepth` Thm 4.2 gives $\mathrm{hop}_k$ at depth
$\lfloor\log_2k\rfloor+2$ ($3/5/7$ at $t^\star=2/8/32$); by verse 01.4 that depth also
solves the committor on the layered family at $K=n$. On BED-S ($K=2$, random DAG) the
skyline depth $5$ at $t^\star=8$ is a **choice by analogy**. The lower bounds are
conditional (Cor. 4.3, the one-versus-two-cycle conjecture; `sanford-2024-graph-algorithms`
Thm 3/19) or vacuous (verse 01.5); `yehudai-2025-depthwidth` removes depth at linear
width, so the matched control fixes width too, and at $d_{\rm model}=16<s=64$ the wide
skyline's matched instance does not exist.

**Which skyline A2 is read against, named at repair, and what follows from the wide one's
absence.** `CHARTER.md:88-91` defines A2 as "at matched parameters against the depth skyline"
and requires "the depth skyline's number printed in the same row"; `CHARTER.md:80` fixes that
skyline's depth at $\lfloor\log_2t^\star\rfloor+2$, which is **$5$ at $t^\star=8$**. Book 04
splits the depth-$5$ row into two arms and only one of them can carry an A2 sentence
(`04_BEDS_AND_INSTRUMENTS.md` verses **04.9** clause 3 and **04.20**, cited by verse id
because book 04's line pins move under its own repairs; at repair time `:149` and `:302`):
**(3a)** the depth-$5$ stack at **full width and unmatched parameters**, whose number is
printed as $\Delta_{\rm sky}$, which satisfies R-SKY and **licenses no A2 sentence at all**;
and **(3b)** the depth-$5$ stack at **reduced width**, the width fixed before any draw as the
largest per-layer width landing within $0.0960$ per cent of the shape's count — and **A2 is
read against 3b and against nothing else**. Of the four skylines book 04 registers, the
depth-$5$ full-width arm (3a) is the one it requires FOUND at the design point; the
matched-width arm (3b) has **no width, no count and no cell** (`NOT MEASURED`), and the wide
constant-depth stack and the chain-of-thought decoder have settings and no cell — book
04.20's own title reads "one FOUND, two with settings and no cell (B11)" and its Limits
paragraph states in full that **no A2 number exists anywhere in that book**. The consequence
for this verse, printed rather than implied: the linear-width skyline of
`yehudai-2025-depthwidth` is **out of scope at $d_{\rm model}=16$** — it has no
matched-parameter instance there — so **no A2 sentence is licensed against it**, by this book
or any other; and the A2 sentence that *is* licensed in principle, against 3b at depth $5$,
has no number today because 3b has no width and no cell. This verse therefore prints an A2
route and no A2 value.

**Hypotheses.** Thm 4.2's model ($m=O(1)$, $H=1$, causally masked) `[V]`, `[V-eq]` owed
(`READ docs/CEQ_SHAPE.md:667-694`). The skyline is a control's depth, not a load-bearing
bound, so `[V]` is admissible for it.

**Evidence.** CITED as above; the reduction DERIVED (01.4); the analogy declared at
`READ docs/CEQ_SHAPE.md:689-694`.

**Mechanism.** P-10, V-25, P-3 (`MISTAKES.md:316`, conditional bounds never stated as
settled), D-7 (`:2037`, Bet D carries its counter).

**Kill — Bet D at the threshold the record registers, with the seed-count clause struck.**
The verse read the kill as "short of the shape's argmin accuracy by more than
$\mathrm{MDE}_8$ **on at least $6$ of $8$ seeds** at $N=8$". That clause is not Bet D's:
`READ docs/CEQ_SHAPE.md:2236 @ 99777ab` registers Bet D's deciding number as
$\mathrm{acc}_{\rm shape}-\mathrm{acc}_{\rm sky}$ with SPLIT band **"none"** and **no
seed-count clause**; the $6$-of-$8$ form belongs to Bets **C**, **F** and **H** in the same
table, and importing it thickened a registered threshold, which M-2 forbids. It is struck:

> **Frozen (Bet D as registered), in A2 form with its floor and its matched count in the
> same clause (`CHARTER.md` §1, L-FLOOR).** On **BED-S** ($t^\star=8$, $m=8$, $K=2$, $N=8$),
> the paired contrast $\mathrm{acc}_{\rm shape}-\mathrm{acc}_{\rm sky}$ at $N=8$ against the
> **realised paired sd**, both arms read as distance to the **argmin zero-information floor**
> $1-\max_a\hat\pi(a^\star)$ (`NOT MEASURED`, $\hat\pi$ has no cell, B5; $0.875$ only as the
> uniform-$\hat\pi$ instance at $m=8$), at **matched parameters** — $4{,}769$ for the shape
> against book 04 clause 3b's reduced-width depth-$5$ count fixed within $0.0960$ per cent
> before any draw (`NOT MEASURED`: 3b has no width and no count) — with the depth skyline's
> own number printed in the same row: the skyline short by more than $\mathrm{MDE}_8$ means
> the bed needs something the hop construction does not supply and the analogy is wrong in the
> direction that flatters the shape. SPLIT band: none. No seed-count clause. **No bare
> comparative is licensed anywhere in this book**: a sentence of the form "the learned stack
> does better than iteration", naming the bed and $\mathrm{MDE}_8$ but neither the floor nor
> the matched count, is struck under `CHARTER.md:106-109` and appears nowhere.

Instrument S-35 (a card of the record); $\approx64.8$ s `[ASSUMED]`, one evening; the
$\mathrm{MDE}_8$ is the realised-sd clause and is `NOT MEASURED` until book 04 verse 04.21
ships $\sigma_d$ (B5). Should a seed-count clause ever be wanted on Bet D, it enters through a
`CORRECTIONS.md` row against the registration and not through a book. Control: on
the layered family of 01.4 at $k=8$, $n=64$ the depth-$5$ stack must reach the argmin
within $\mathrm{MDE}_8$ (Thm 4.2's own instance); a skyline that fails where the theorem
says it succeeds is mis-built.

**If killed** (the skyline falls short on BED-S). Replacement: the depth dial is escalated
to $7$ and then to the chain-of-thought decoder at $t^\star$ steps (`merrill-2024-cot`
[V]; `READ docs/CEQ_SHAPE.md:2508-2510`), each a registered skyline row; its Kill is the
same $\mathrm{MDE}_8$ comparison at the next depth ($\approx7.6$ s per cell `[ASSUMED]`,
cheaper than the first because the arm exists). If every depth falls short, the sentence
licensed is "on BED-S the hop construction does not reach the label at any registered
depth" — an observation about the control, filed with its sign, never "softmax cannot".

**Terminal.** Hops need depth is lineage; the canon licenses the skyline row with its
$\Delta_{\rm sky}$ interval — the depth-$5$ full-width control of book 04's clause 3a, which
carries no A2 sentence — and withdraws every sentence that reads the analogy as a
theorem about BED-S. A2 is licensed only against the matched-parameter depth-$5$ skyline of
book 04's clause 3b, which has no width, no count and no cell, so **no A2 number is licensed
by this book today**; and it is licensed against the linear-width skyline **never**, since at
$d_{\rm model}=16<s=64$ that arm has no matched-parameter instance at all.

---

## Part III — Per-row independence, Cheeger, Bellman, degeneracy

### 01.9 — Per-row independence is a definition; joint consistency is a measured ratio

**Statement.** One softmax step is $O_i=f(P_{i\cdot},V)$ and the reads are independent
across $i$; the shape's $z$ is the unique fixed point of the $\gamma$-contraction
$z=V+\gamma Pz$ in $\|\cdot\|_\infty$, so its reads are mutually constrained. **The claim
about the literature is replaced at repair by the decidable half.** The verse read "No theorem
converts this into an error separation" — a universal negative about what exists in the
literature, which no measured ratio can refute, so no kill of this verse could reach it. What
this verse asserts instead: **the joint-consistency claim at `MATHEMATICS.md:106-127` is
UNTESTED**, the ratio below with its registered counter is what would test it, and until a
cell is read the counter is the point estimate (D-CALIB). The measurable statement is the
harmonic residual
$r(\hat z)=\|(I-\gamma P_{\rm env})\hat z-V\|_\infty/\|V\|_\infty$, and the licensed
prediction is the ratio $r_{\rm softmax}/r_{\rm shape}$ with the counter $\le2$ (Bet E,
SPLIT in $(2,10)$).

**Hypotheses.** $\gamma<1$, $P$ row-stochastic (`Contraction.rowStochastic_perron`,
`lean/CEQ/Contraction.lean:118`); $r$ printed beside $\sigma_{\min}(I-\gamma P_{\rm env})$
and $\|I-\gamma P_{\rm env}\|_\infty\le1+\gamma$ (its units). Non-vacuous whenever both
arms reach marginal NRMSE within $\mathrm{MDE}_8$ of each other — otherwise the ratio
compares two different errors.

**Evidence.** `READ docs/CEQ_SHAPE.md:652-665`; the claim "joint determination in one
read" UNTESTED at `READ MATHEMATICS.md:106-127`; nearest per-query fixed points
`ramsauer-2021-hopfield` [V]; the sharpness failure `velickovic-2025-softmaxnotenough` [V].

**Mechanism.** V-26 (`MISTAKES.md:2190`, a marginal is not a joint); V-17; D-7.

**Kill.** K-H1: $r_{\rm softmax}/r_{\rm shape}\le2$ at marginal NRMSE within
$\mathrm{MDE}_8$ on $6$ of $8$ seeds — "joint determination in one read" is withdrawn;
inside the $\approx34$ s pair (Bet E's one bed-cell pair, `04_BEDS_AND_INSTRUMENTS.md` verse
04.6; at repair time `:102`), one evening.

**Registration clause, added at repair: K-H1 and verse 01.2's K-D2 cannot both be decided on
the same seeds.** $r_{\rm shape}$ is not free of the identification error — it *is* it. The
shape solves $\hat z=V+\gamma\hat P\hat z$ exactly, so
$$(I-\gamma P_{\rm env})\hat z-V=(V+\gamma\hat P\hat z)-\gamma P_{\rm env}\hat z-V
=\gamma(\hat P-P_{\rm env})\hat z,\qquad
r_{\rm shape}=\frac{\gamma\|(\hat P-P_{\rm env})\hat z\|_\infty}{\|V\|_\infty}
\ \le\ \frac{\gamma\,\|\hat P-P_{\rm env}\|_\infty}{1-\gamma}$$
(DERIVED, using $\|\hat z\|_\infty\le\|V\|_\infty/(1-\gamma)$). K-D2 fires when
$\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on $\ge6$ of $8$ seeds; **on exactly those seeds
$r_{\rm shape}\le1.5\times10^{-3}$ at $\gamma=0.6$** ($0.6\cdot10^{-3}/0.4$, DERIVED), so
K-H1's "$\le2$" would need $r_{\rm softmax}\le3\times10^{-3}$ — the softmax arm identifying
$P_{\rm env}$ almost as well as the shape — and the ratio otherwise diverges. The two frozen
kills of this book are therefore mutually exclusive in that direction, and the registration is
made explicit: **$r_{\rm shape}$'s floor and the identification reading
$\|\hat P-P_{\rm env}\|_\infty$ are printed beside every K-H1 evaluation, and K-H1 is
registered only on the seeds where K-D2 does not fire**, with the excluded seed count printed
in the row. A K-H1 verdict quoted over a seed set that includes a K-D2-firing seed is VOID,
never a reading. Both numbers are `NOT MEASURED — needs the first BED-S cell (B5; book 04
verse 04.21)`.

Control: an arm emitting the oracle $z$ itself
must read $r=0$ to $10^{-12}$, and the same arm with coordinate $i_{\min}$ perturbed by
$+0.1\sigma$ must read $r>0$ — a residual blind to a one-coordinate move is not a
residual (B-G2's plants, `READ docs/CEQ_SHAPE.md:3115`).

**If killed.** Replacement: the residual as a diagnostic and "one read" as a cost statement
(one solve, depth $s$), owned by book 03; its Kill is K-9's band. The chain ends there.

**Terminal.** Per-row independence is a definition and licenses nothing about error; the
canon keeps the residual as an instrument, records the sentence at `MATHEMATICS.md:106-127`
as **UNTESTED** rather than as unsupported by the literature, withdraws it whenever Bet E
reads its counter, and withdraws every K-H1 verdict quoted over seeds on which K-D2 fires.

---

### 01.10 — **OPEN** — The Cheeger sentence: withdrawn on the causal $P$, restated on the reversible oracle chains

*Marked OPEN at repair: branch (a)'s kill was satisfied by construction and is struck, and
branch (b)'s kill needs an instrument the canon does not specify, so the verse carries no
reachable kill today and the Terminal below is in force now. Both branches keep their
decidable content as admission lines, which are instruments and not kills.*

**Statement.** (a) On the arm's causal class, $Q=\hat P_{TT}$ is lower-triangular, so
$$\rho(Q)=\max_{i\in T}\hat P_{ii}\quad\text{and}\quad I-Q\ \text{is a unit iff}\ 0\in\mathcal A,$$
there is no bottleneck spectrum to bound, and every conductance sentence about $\hat P$ is
withdrawn; the truncation residual of a $t$-hop read is $Q^t q$ with $\|Q^t\|_\infty\le1$
(sub-stochastic) and no termwise $\rho(Q)^t$ bound (a triangular $Q$ is non-normal). (b) On
the oracle cross-check chains (BED-1, E4′: random walks $D^{-1}W$ on undirected weighted
graphs, reversible with respect to degree), `levin-2017-markov-mixing` Thm 13.10 applies
as `[V-eq]`: for a reversible $P$ with spectral gap $\gamma_{\rm gap}$ and bottleneck ratio
$\Phi_\star$, $\Phi_\star^2/2\le\gamma_{\rm gap}\le2\Phi_\star$ (read at p. 183 of the
authors' PDF, `READ docs/references.bib:2579`), with the record's instance
$g\le2\varphi$, $\varphi=1/\mathrm{vol}(S)$, $t_{\rm rel}\ge\mathrm{vol}(S)/2=1372.50$ at
`READ MATHEMATICS.md:455-456` — an instance on E4′'s own reversible chain, admissible
there and inadmissible on $\hat P$. (c) The non-reversible replacement is not a
symmetrisation of $\hat P$ (the additive symmetrisation's spectrum is unrelated to
$\mathrm{diag}\,\hat P$) but the exact eigenvalue read of (a).

**Hypotheses.** (a) finite logits, $\beta=1$; (b) reversibility, which holds for
$D^{-1}W$ and fails for any causal $P$ with $P_{i0}>0$, $P_{0i}=0$. Non-vacuous: (a) at
every draw of the class; (b) on BED-1 and E4′ only, with $\rho(Q_{\rm BED\text{-}1})=0.9408612510154677$
(`RUN`, `READ docs/CEQ_SHAPE.md:890`).

**Evidence.** (a) DERIVED (eigenvalues of a triangular matrix are its diagonal) and
`RUN[J]` $0.6926596893360386$ both ways (`READ docs/CEQ_SHAPE.md:539-543`); (b) CITED
`levin-2017-markov-mixing` [V-eq] via `lawler-1988-bounds`, `sinclair-1989-approximate`,
`cheeger-1970-lower` [V]; the correction filed at `READ docs/CEQ_SHAPE.md:1122-1127` and
card J-D6 (`:1584-1590`). The Dirichlet-form Cheeger bound for the killed reversible
chain, $\lambda_{\min}(I-Q)\ge h_D^2/2$, is textbook and its citation is **owed** (not in
`references.bib`); it is `[ASSUMED textbook]` and load-bearing nowhere in this book.

**Mechanism.** V-25 (`MISTAKES.md:1954`) — a reversible theorem on a non-reversible chain
is decoration with a proof attached; P-10; V-17 (the ergodic-chain gap is not the
transient-block Perron root, `READ docs/sources/sweep/sweep_methods.md:413-415`).

**Kill (a) — struck at repair; not replaced by another gate of the same kind.** The
previous kill (a) read "any drawn trained cell with $|\rho(\hat Q)-\max_T\hat
P_{ii}|>10^{-12}$", and the verse's own If-killed conceded in one line that it "cannot fail
on a triangular operator". It cannot: the eigenvalues of a triangular matrix **are** its
diagonal entries, exactly, so the quantity being thresholded is the float rounding of an
eigensolver against a number read straight off the diagonal, and $10^{-12}$ is a
numerical-noise gate, not a refutation of the Statement (`CHARTER.md:145-146`, V-10). The
repair does **not** repoint it at the strict-upper `torch.equal` census either: that census
is itself satisfied by construction on a causally masked softmax, whose strict upper block
is bitwise $0.0$ because the mask writes $-\infty$ before the exponential (verse 01.19's
repair, same defect). Kill (a) is therefore deleted, and the triangularity census is filed
as an **admission line** — a cell whose $\hat P$ fails `torch.equal` on the strict upper
block is refused at admission, before any certificate is read off it — owned by book 04,
which does not carry it: `NOT MEASURED — needs the triangularity admission line,
**unspecified in the canon**` (`04_BEDS_AND_INSTRUMENTS.md` carries verses 04.1–04.21 and no
such line; the manifest of 04.15 journals `route` and `boundary_sets`, not the strict-upper
read). What branch (a) asserts — $\rho(Q)=\max_T\hat P_{ii}$, and no termwise $\rho(Q)^t$
bound on the non-normal $Q$ — is algebra, and this verse states it as algebra with no kill
attached.

**Kill (b) — reachability restated, and the missing plant supplied.** On BED-1, $h_D^2/2$
computed from the bed's own conductance exceeding $1-\rho(Q)=0.0591$: the Dirichlet Cheeger
instance fails and the `[ASSUMED textbook]` bound is struck. Price $0$ GPU-s; but
`NOT MEASURED — needs the Dirichlet-conductance instrument on `bed_1`, **unspecified in the
canon**`: a grep for "Dirichlet" in `04_BEDS_AND_INSTRUMENTS.md` returns one hit, inside
04.15's manifest field `committor_route ∈ {dirichlet_gamma1, discounted_gamma}`, which names
a route and not an instrument, so no verse of book 04 supplies the conductance read. The
kill is **unreachable until book 04 ships it**, and the verse stands OPEN on that account.
**Control, supplied at repair — this branch shipped with no planted negative, the only kill
in the book with none (V-15).** The must-fire plant is a non-reversible transient block
whose Dirichlet conductance is computed by the same instrument: a two-state block with
$Q=\begin{pmatrix}0&0.9\\0&0.9\end{pmatrix}$ has $\rho(Q)=0.9$, $1-\rho(Q)=0.1$, and a
Dirichlet conductance whose $h_D^2/2$ the instrument must report **above** $0.1$ — a chain
that leaves state $1$ only to absorption at rate $0.1$ while the conductance sees the
$0.9$ edge (DERIVED; reversibility fails because $Q_{12}=0.9$ and $Q_{21}=0$). An instrument
that certifies BED-1 without first firing on this plant has never been shown to fire at all,
and the `[ASSUMED textbook]` bound would then be certified by a probe with an empty
rejection region. The plant is priced with the instrument, $0$ GPU-s, and is `NOT MEASURED`
for the same reason.

**If killed.** (a) has no kill to fire. (b) If the Dirichlet instance fails once the
instrument exists, the Cheeger sentence is withdrawn from the oracle chains too, and the
replacement is the hop-ladder rate read off the ladder (`READ MATHEMATICS.md:420-462`: the
ladder is pre-asymptotic and $\lambda_2^{t^\star}$ is the wrong predictor at every rung), a
measured decay with no spectral prediction — a different object (a decay read at $t$, not a
conductance), a different instrument (the ladder, which the record has already run), and a
different number; its Kill is the ladder's own $f=\mathrm{decay}/\lambda_2$ reading outside
$[0.99,1.01]$ at $t\ge320$, which the record already read inside it ($1.00000063$,
$1.00158689$, `:441`), so that link is reachable today at $0$ GPU-s and is strictly cheaper
than the conductance read. The chain ends there.

**Terminal — in force now.** No conductance sentence is written about the causal $P$; the
canon licenses $\rho(\hat Q)=\max_T\hat P_{ii}$ as the exact rate on the arm's class — as
algebra, with no kill and therefore no capability weight — and licenses the reversible
Cheeger bound on the oracle chains only, as `[ASSUMED textbook]` with its citation owed and
its instrument unspecified; it withdraws `MATHEMATICS.md:455` as a sentence about the shape,
and withdraws the claim that either branch of this verse has been tested.

---

### 01.11 — The Bellman hazard resolved: one-clamp optimality, DAG backward induction, and where scalarisation breaks

**Statement.** (i) *One clamp.* With one intervention chosen from $m$ candidates and the
environment otherwise fixed, $a^\star=\arg\max_aq^{(0)}(\mathrm{do}\,a)$ maximises
$\Pr(\text{reach }\mathcal A_0\text{ before any }\mathcal A_k)$ by definition of the
objective; no principle of optimality is invoked. (ii) *Sequential goal rule on the DAG.*
If a clamp may be chosen at every transient state visited, the optimum is
$$V^\star(i)=\max_{a\in\mathcal M(i)}\sum_jP^{(a)}_{ij}V^\star(j),\qquad V^\star|_{\mathcal A_0}=1,\ V^\star|_{\mathcal A\setminus\mathcal A_0}=0,$$
solved exactly by backward induction in topological (position) order in at most $s$
steps. **The chain is multichain, and the justification never says otherwise.** With $K+1$
boundary sets plus a sink, every absorbing state is its own closed recurrent class, so at
the design point ($K=2$, $|\mathcal A_\bullet|\in\{1,2\}$ over
$\mathcal A_{\rm sink},\mathcal A_0,\mathcal A_1,\mathcal A_2$, `READ
docs/CEQ_SHAPE.md:295-297 @ 99777ab`) the chain carries $|\mathcal A|\ge4$ absorbing states
and hence $\ge4$ recurrent classes; a chain is unichain iff it has exactly one. The earlier
clause "the class is unichain-absorbing and Misra's multichain hazard is absent for this
objective" is **deleted**: it closed census row B10 by asserting that the hazard's own
hypothesis — several unsafe sets, multichain (`CHARTER.md:236`) — was absent, when the
design point is that hypothesis. What licenses backward induction is a different fact
about a different criterion: **the objective here is a total-reward absorption probability
on a transient DAG block with $\rho(Q)<1$, not an average-reward criterion**, and Misra's
hazard is a hazard of the average-reward constrained LP, where a multichain structure makes
the single-chain LP's optimal value unattainable by a stationary policy. On a total-reward
absorption objective the value recursion above is a finite backward substitution: the
position order is a topological order of the DAG, $V^\star$ at every $\mathcal A$ state is
fixed by the terminal condition, and each transient row depends only on strictly later
positions, so the recursion terminates in $|T|\le s$ substitutions with no fixed-point
argument, no ergodicity assumption, and no appeal to a unique recurrent class. *The
hypothesis under which that argument holds, printed:* (H1) every clamp $a\in\mathcal M(i)$
keeps $P^{(a)}$ causal, so the substrate stays a DAG under every policy; (H2)
$0\in\mathcal A$ and every policy's chain absorbs with probability $1$ in at most $s$ steps
(`later_boundary_unreachable`; verse 01.10's branch (a)); (H3) the objective is additive in
the absorption indicator — which (iii) below shows the Chebyshev objective is **not**, and
that is where the principle of optimality genuinely fails here. The one-step rule with the
fixed continuation is a lower bound on $V^\star$. (iii) *Scalarisation.* The Chebyshev objective
$\min_\pi\max_{k\ge1}q^{(k)}_\pi$ is a non-additive function of the absorption vector, and
the principle of optimality fails for it even on a DAG. **The witness is rewritten at repair,
because the earlier one exhibited no second epoch and used an action the bed does not carry.**
The earlier witness had one state $u$, two clamps $a$ and $b$ absorbing with probability $1$
and **no continuation**: deterministic value $1$, mixture $\tfrac12a+\tfrac12b$ value
$\tfrac12$. The principle of optimality is a statement about sub-policies of a **multi-epoch**
problem, so a one-epoch instance cannot exhibit its failure; and the safest-move read is an
argmin over $m=8$ discrete candidate moves realised as row clamps with targets restricted to
$T$ (R-07, `READ docs/CEQ_SHAPE.md:1736-1742 @ 99777ab`), in which the mixture
$\tfrac12a+\tfrac12b$ is not a candidate, so the exhibited gap was **unreachable on the bed as
registered** (V-11). What that instance does prove is kept, and stated as what it is: *for a
Chebyshev objective the optimum over randomised clamps ($\tfrac12$) is strictly below the
optimum over deterministic clamps ($1$)* — an aside about the objective's action set, carrying
no principle-of-optimality content and no bed reachability.

**The two-epoch witness, derived here, on the bed's own action set.** Eight positions in
causal order, every edge to a strictly smaller index: $\mathcal A_{\rm sink}=\{0\}$,
$\mathcal A_1=\{1\}$, $\mathcal A_2=\{2\}$ absorbing ($K=2$); transient
$x=3$ with row $e_1$, $y=4$ with row $0.6e_1+0.4e_2$, $w=5$ with row $e_2$, $v=6$ clampable,
and the query $u=7$ with row $\tfrac12e_6+\tfrac12e_5$. Two candidate moves at $v$, both
R-07 row clamps to **transient** targets: $c:\;v\leftarrow e_3$ and $d:\;v\leftarrow e_4$.
Then $q(x)=(1,0)$, $q(y)=(0.6,0.4)$, $q(w)=(0,1)$ and

| policy at $v$ | $q(v)=(q^{(1)},q^{(2)})$ | Chebyshev value at $v$ | $q(u)$ | Chebyshev value at $u$ |
|---|---|---|---|---|
| $c$ | $(1,\ 0)$ | $1$ | $(0.5,\ 0.5)$ | $\mathbf{0.5}$ |
| $d$ | $(0.6,\ 0.4)$ | $\mathbf{0.6}$ | $(0.3,\ 0.7)$ | $0.7$ |

(DERIVED, exact rationals.) The Chebyshev-optimal sub-policy at $v$ is $d$ ($0.6<1$); the
Chebyshev-optimal full policy uses $c$ ($0.5<0.7$). **The locally optimal prefix is not the
prefix of any optimal full policy, and choosing it costs $0.7-0.5=0.2$ at the root** — the
principle of optimality fails, with two epochs, deterministic clamps only, and every action in
the read's own set. The Chebyshev and lexicographic columns are therefore **one-clamp
scalarisations of evaluated vectors** and never sequential policies.

**Hypotheses.** (i) none beyond the label's definition; (ii) H1, H2, H3 as printed in the
Statement, with moves as row clamps $u_a\in T$ (R-07's restriction, `READ
docs/CEQ_SHAPE.md:1736-1742 @ 99777ab`) — and **not** unichain, which is false at the design
point and is nowhere assumed; (iii) $K\ge2$, at least two transient positions below the
clampable one, and clamp targets in $T$ (R-07) — the witness above uses exactly those and no
randomised action. Non-vacuous: (ii) at every BED-S draw, where
the recurrent-class count is $|\mathcal A|\in\{4,\dots,8\}$ and the average-reward criterion
Misra's theorem is about is never formed; (iii) on any draw carrying a clampable transient
position with two distinct transient targets whose committor vectors are not proportional —
which the eight-position witness realises inside $s=64$, $K=2$, $m\ge2$, so it is reachable on
the bed as registered rather than only in a mixture the action set excludes. Census row B10 is
closed by the criterion distinction printed above, not by denying the hazard's hypothesis.

**Evidence.** DERIVED (backward induction on a finite DAG; the eight-position two-epoch
witness of (iii), exact rationals, and the one-epoch randomisation aside kept apart from it).
CITED `misra-2023-safety-constrained-mdp` [V] for the hazard's class (multichain,
several unsafe sets — carried, `READ docs/CEQ_SHAPE.md:2960-2963`); `hsu-2023-safetyfilter`,
`borquez-2023-lrf` [V] for the one-step least-restrictive filter vocabulary;
`vanmoffaert-2013-chebyshev`, `yang-2026-lexisafe`, `altman-1999-cmdp` [V] for the
scalarisations; `bellman-1957-markovian` for the principle whose failure (iii) exhibits.
R-20's instrument (`READ docs/CEQ_SHAPE.md:1758-1763`) measures the one-step-versus-two-step
disagreement fraction.

**Mechanism.** P-7 (`MISTAKES.md:375`, "safest move" had no referent — it now has three,
named); V-17; P-10 (Misra is cited for its class, not beyond it); V-1 (`:34`, the two
rules are columns of one bed with the disagreement fraction printed).

**Kill — one per clause, because R-20 refutes none of the three.** The verse carried R-20
alone: *"the one-step goal rule and the two-step optimum (apply $a$, re-solve, best second
clamp) disagreeing on more than $0.5$ of $512$ admitted oracle draws at the design point"*
(`READ docs/CEQ_SHAPE.md:1758-1763 @ 99777ab`, `:2275`; a card of the record — book 04
carries no R-20 verse, so the routing is to the pin and not to a book-04 id). A disagreement
fraction between two heuristics cannot refute **(i)**, which is true by the definition of the
objective; nor **(ii)**, which is a backward-induction theorem on a finite DAG; nor **(iii)**,
which is closed arithmetic on eight positions. R-20 is not deleted — it is repositioned as the
**replacement's** kill below, where the claim it can decide actually lives — and each clause
now carries the kill that can refute it.

> **Kill (i) — the label is not constant across the candidate moves.** Frozen: on the $512$
> admission draws of `04_BEDS_AND_INSTRUMENTS.md` verse **04.2**, the realised effective move
> count $m_{\rm eff}$ (that verse's census clauses 9 and 11: the no-op screening rate is the
> fraction of candidate targets whose clamp moves the query's committors by $>10^{-12}$)
> reading $m_{\rm eff}\le1$ on more than $0.05$ of draws. A **value-only** move leaves every
> $q^{(k)}$ constant across moves (`READ docs/CEQ_SHAPE.md:475 @ 99777ab`), so where every
> candidate is value-only the argmax label is constant (V-12) and (i)'s "maximises by
> definition of the objective" is a tie-break over one identical vector rather than a decision
> rule. Instrument: 04.2's census, a verse of book 04 that carries it; price $0$ GPU-s, no
> arm, no cell. Planted negative: a draw whose $m$ candidates are constructed value-only (the
> $V$-only plants of verse 01.15, which read $\le10^{-15}$) must read $m_{\rm eff}=0$ and fire.

> **Kill (ii) — the DP value against exhaustive policy enumeration.** Frozen: on draws
> restricted to $|T|\le12$ and $m\le3$ candidate clamps per state,
> $\max_i|V^\star_{\rm DP}(i)-V^\star_{\rm enum}(i)|>10^{-12}$ on any of $512$ such draws,
> where $V^\star_{\rm enum}$ is the maximum over all $m^{|T|}\le3^{12}=531{,}441$ deterministic
> stationary policies of the exact absorption probability of that policy's chain. Instrument:
> the exact solve applied once per enumerated policy on a $\le12$-state transient block, CPU
> float64; price $0$ GPU-s, one evening. This is the only clause that tests whether backward
> induction **is** the oracle, which is what (ii) asserts. Planted negative: replace the
> $\max$ in the recursion with the $\min$ — enumeration must then exceed the DP value by
> $O(1)$ on at least one draw, and an identity that survives an inverted recursion is
> comparing a routine with itself. Second plant, aimed at H3: run the same enumeration against
> the DP applied to the Chebyshev objective $\min_\pi\max_kq^{(k)}_\pi$ — the two **must**
> differ, by $0.2$ at the root of (iii)'s eight-position witness, because H3 fails there.

> **Kill (iii) — the witness re-evaluated.** Frozen: recomputing the five numbers of (iii)'s
> table in float64 — $\max q(v\mid c)=1$, $\max q(v\mid d)=0.6$, $\max q(u\mid c)=0.5$,
> $\max q(u\mid d)=0.7$, root gap $0.2$ — and finding the local and global argmins **agree**,
> or the gap reading below $10^{-12}$. Instrument: one $8\times8$ exact solve pair, CPU
> float64, no bed and no arm; price $0$ GPU-s, minutes. Planted negative: replace the
> Chebyshev $\max_k$ with the linear scalarisation $q^{(1)}$ alone — local at $v$ then reads
> $c=1$, $d=0.6$ and global at $u$ reads $c=0.5$, $d=0.3$, so both argmins are $d$ and
> **must** agree; an instrument reporting disagreement under an additive objective is reading
> its own bug and not a failure of optimality.

The three are **strictly ordered by price**: (iii) is minutes on an $8\times8$ pair, (i) is a
census column on draws the bed already takes, (ii) forms up to $531{,}441$ solves per draw.

**If killed** (any of the three clauses). The phrase "safest move" is replaced by "one-step
safety filter" in every sentence, the sequential problem is withdrawn, and what ships is the
one-clamp filter of (i) alone — the argmin over $m$ candidate moves of an evaluated committor
vector, with no policy-level claim attached. **The replacement's kill is R-20, repositioned
here from the verse's Kill field, where it could refute none of the three clauses and here
refutes exactly what the replacement asserts.**

> **Frozen (R-20).** The one-step goal rule and the two-step optimum (apply $a$, re-solve,
> best second clamp) disagreeing on more than $0.5$ of $512$ admitted oracle draws at the
> design point (`READ docs/CEQ_SHAPE.md:1758-1763 @ 99777ab`, `:2275`).

Then the one-step filter is not a stand-in for the sequential optimum on more than half the
bed, "safest move" is withdrawn as a phrase in every reading, and the chain ends at the
Terminal with the evaluated vector label alone licensed. Instrument: $m$ re-solves per draw on
the bed's own generator, $0$ GPU-s, one evening. It is **strictly cheaper than Kill (ii)** by
the ratio of solves per draw, $3^{12}/m=531{,}441/8=66{,}430$, and it decides a different
object — the adequacy of a *filter*, not the exactness of a *recursion*. Control: on the
layered family of verse 01.4 (deterministic pointers) the two rules must agree on $100\%$ of
instances — every clamp there is a full redirect and the continuation carries no choice;
disagreement on that family is an instrument bug. Planted negative: a draw in which the second
clamp is forced (one candidate at the successor) must read agreement; a rule ladder that
disagrees where the second step has no choice is measuring its own re-solve and not the
horizon. The earlier text made this replacement's kill "the three-step optimum disagreeing
with the two-step on more than $0.5$ of draws — the same instrument one level deeper, equally
priced", which is the same number on the same object measured by the same instrument and is a
V-9 replacement (`CHARTER.md:133-134`) as well as a violation of the Depth rule
(`CHARTER.md:139-141`, which admits no "equally priced" link); it is struck and is not
reinstated.

**Terminal.** The argmin is a decision rule over an evaluated vector label and claims no
policy-level optimality; the canon licenses (i) unconditionally, (ii) as the oracle for
the sequential extension, (iii) as the eight-position two-epoch witness that the Chebyshev
objective's locally optimal prefix costs $0.2$ at the root — a statement about the objective
on the read's own action set — and, separately and as an aside only, that the Chebyshev
optimum over randomised clamps ($\tfrac12$) lies strictly below the optimum over
deterministic ones ($1$) on a one-epoch instance whose mixture the bed's action set does not
contain. It withdraws the Chebyshev and lexicographic columns from any sequential reading, and
withdraws the earlier claim that the one-epoch mixture instance exhibited a failure of the
principle of optimality.

---

### 01.12 — The reach-avoid identity with a sink, and the degeneracy lemma

**Statement.** With $\mathcal A$ exhausting the absorbing states and $\rho(Q)<1$,
$$q^{(\rm sink)}+q^{(0)}+\sum_{k\ge1}q^{(k)}=\mathbb 1\ \text{on } T,\qquad
\text{and if the constraint sets alone exhaust absorption, }\ \max_kq^{(k)}\ge1/K .$$
The exact zero-information floor on the argmin is
$$\mathrm{floor}_{\rm zeroinfo}=1-\max_a\hat\pi(a^\star),$$
computed from **the realised class distribution $\hat\pi$ of the argmin label on the batch the
cell is scored on**, and $1-1/m=0.875$ at $m=8$ is its **uniform-$\hat\pi$ instance and
nothing more. The unconditional form is struck at repair.** The record carries the conditional
form — *"exact zero-information floor $1-\max_a\pi\text{-hat}(a^\star)$ from the realised class
distribution; at uniform $\pi$ this is $1-1/m$"* (`READ docs/CEQ_SHAPE.md:2493 @ 99777ab`) —
and so do three neighbouring books of this canon: `04_BEDS_AND_INSTRUMENTS.md` verse 04.3
($\mathrm{floor}_{\rm zeroinfo}=1-\max_a\hat\pi(a^\star)$, $=1-1/m_{\rm eff}$ at uniform
$\hat\pi$; at repair time `:59`), `06_PREDICTIONS.md` verse 06.18 ("$0.875$ at the realised
class distribution when it is uniform at $m=8$"; at repair time `:256`) and
`09_CHESS_AND_MARKETS.md` verse 09.8 ("at uniform $\pi$ and $m=7$ this is $1-1/7=0.857143$; at
$m=8$, $0.875$"; at repair time `:91`) — all four cited by verse id because their line pins move
under their own repairs. This book printed $0.875$ flat and thereby contradicted the record and
three books at once. **How far the flat form can be wrong, printed:** the D-4 admission gate
admits any argmin class frequency in $(0.05,0.95)$ (`READ docs/CEQ_SHAPE.md:1668 @ 99777ab`),
so an admitted cell may realise $\max_a\hat\pi(a^\star)=0.95$ and a true floor of $0.05$ — an
error of $0.825$ against the printed $0.875$, in the direction that credits an arm scoring at
chance with a capability. $\hat\pi$ itself is
`NOT MEASURED — needs the realised argmin class distribution on the first BED-S batch (B5; book
04 verse 04.21's cell, journalled in 04.15's field `floor_zeroinfo`)`, so **no value of this
floor is printed for BED-S by this book**, and $0.875$ appears only with "at uniform $\hat\pi$"
attached. The weak Fano $1-\ln2/\ln m$ ($0.6667$ at $m=8$) is not a floor; the tight Fano
$(\ln m-I-\ln2)/\ln(m-1)$ ($0.7124$ at $m=8$, $I=0$) sits below the uniform-$\hat\pi$ floor and
is retired by book 04 verse 04.3 on that ground, and no value of it is printed here.

**Hypotheses.** Row sums $1$ including absorbing rows; $0\in\mathcal A_{\rm sink}$;
sets exhaust the absorbing states. Non-vacuous at every draw with the sink declared; the
sink share read $[0.362,1.000]$ on the one draw run (`RUN`, `READ docs/CEQ_SHAPE.md:1671`),
which is the admission hazard of Tree C.

**Evidence.** DERIVED: $\sum_\bullet R_\bullet\mathbb 1=\mathbb 1-Q\mathbb 1$ and
$N(I-Q)\mathbb 1=\mathbb 1$; pigeonhole. `RUN[I]` conservation row
$[0.9999999999999993,1.0]$; with BOS inside a constraint set and no goal,
$\min_T\max_kq^{(k)}=0.548718\ge\tfrac12$ (`RUN[P]`, `READ docs/CEQ_SHAPE.md:1531`).
Floors `RUN` at `READ docs/CEQ_SHAPE.md:2493`.

**Mechanism.** V-12, V-8 (`MISTAKES.md:136`), V-23 (`:1626`), V-10 (the weak Fano sits
$0.208$ below chance and would pass an arm worse than chance), L-FLOOR.

**Kill — two clauses, one per half of the Statement.** *(a) The identity half.* The
conservation row off $\mathbb 1$ by more than $10^{-12}$ on any of $512$
draws: a set was left undeclared and the solve must raise (V-16); $0$ GPU-s. Control:
drop BOS from every set — the solve must raise with $\rho(Q)=1.000000$, $\det=0$
(`READ docs/CEQ_SHAPE.md:3116`). *(b) The floor half, which clause (a) cannot decide.* A
conservation residual says nothing about a floor, so the floor clause is decided by the kill
of the book that owns L-FLOOR: `04_BEDS_AND_INSTRUMENTS.md` verse **04.3** — an uninformed
control (the $0$-hop, `random` or `majority` arm) printing an error below
$\mathrm{floor}_{\rm zeroinfo}$ computed on the batch it was scored on, on $\ge2$ of $8$
seeds — routed there by verse id and not restated here, plus this book's own assembly clause:
**any cell of this canon printing $0.875$ as the argmin floor without "at uniform $\hat\pi$"
beside it, or printing an argmin capability number without the realised $\hat\pi$ in the same
row.** That clause is decidable today by reading the page, $0$ GPU-s, and it fired on this
verse before repair.

**If killed** (clause (a): the conservation row is off $\mathbb 1$). Replacement: the sink set
is enlarged to every undeclared absorbing position found by the census
"$\hat P_{ii}=1$ bitwise and $i\notin\mathcal A$", and the read is re-registered on the
enlarged partition. **Its Kill is not the same residual re-read — that was a V-9 rename and is
struck.** A horizon of one more solve cannot refute the enlargement that produced it; the
replacement is decided by a different object, a different instrument and a different number:

> **Frozen.** The undeclared-absorbing count $n_{\rm abs}:=\#\{i\notin\mathcal A:\hat
> P_{ii}=1\ \text{bitwise}\}$ reading $0$ on a draw whose conservation row is still off
> $\mathbb 1$ by more than $10^{-12}$.

Then the missing mass is not an undeclared absorbing row, the enlargement cannot recover it,
and the defect is in the generator's row normalisation rather than in the partition: the
identity is withdrawn on that bed and the chain ends at the Terminal. Instrument: one bitwise
pass over the diagonal — an integer count, not a sup-norm of a solve — $0$ GPU-s, **strictly
cheaper than the conservation read**, which forms $(I-Q)^{-1}R\mathbb 1$. Planted negative: a
draw with one transient row overwritten by $e_i$ must read $n_{\rm abs}\ge1$; a census that
cannot see a planted identity row cannot certify the absence of one.

**Terminal.** The identity is Kemeny–Snell's; the canon licenses the printed conservation row
and the exact zero-information floor **in its conditional form $1-\max_a\hat\pi(a^\star)$ with
the realised class distribution named**, licenses $0.875$ only as that floor's uniform-$\hat\pi$
instance at $m=8$, and withdraws every floor not printed beside its class distribution —
including every unconditional $1-1/m$ this book printed before repair.

---

## Part IV — The ChaCAL delta as theorems

### 01.13 — ChaCAL-published and the diagonal-kept read coincide only at $\gamma=0$

**Statement.** Let $A$ be a causal softmax and $A_s=A-\mathrm{diag}(A)$. ChaCAL's published
read is $Y=(1-\gamma)A(I-\gamma A_s)^{-1}V$ (diagonal removed inside the inverse,
`fagnou-2024-chacal` Eq. 5 with the "remove the diagonal part" clause, `[V-fetched]`);
the shape's is $O=(1-\gamma)A(I-\gamma A)^{-1}V$. Then
$$O-Y=(1-\gamma)\,\gamma\,A(I-\gamma A)^{-1}\mathrm{diag}(A)(I-\gamma A_s)^{-1}V,$$
which vanishes for all $V$ iff $\gamma=0$ or $\mathrm{diag}(A)=0$; on the softmax corner
$A_{ii}>0$, so the two reads differ at every $\gamma>0$, ChaCAL-published is
sub-stochastic (row sums $0.400\dots0.765$ against $1.000$, `RUN[J]`,
`READ docs/CEQ_SHAPE.md:207-211`) and its resolvent is a finite regime-N sum inside a
regime-S read.

**Hypotheses.** Finite logits, $\beta=1$. Non-vacuous at every $\gamma>0$.

**Evidence.** DERIVED (resolvent identity $M_1^{-1}-M_2^{-1}=M_1^{-1}(M_2-M_1)M_2^{-1}$
with $M_2-M_1=\gamma\,\mathrm{diag}(A)$); `RUN[J]` row sums as cited; the convention
rests on one HTML fetch (`READ docs/sources/design/refute_falsify_math.md:327-344`).

**Mechanism.** P-10 (a source's equation read from a rendered page); V-3 (the parity
plant is declared as a construction); B22 (`CHARTER.md` §5) — the convention decides two
controls (C-CHD, C-CHP, `READ docs/CEQ_SHAPE.md:3128-3129`).

**Kill — the arXiv re-read is repriced as an author action and is not this verse's kill;
the operator-level kill replaces it.** The previous kill was "the source-level re-read of
Eq. 5 and Eq. 7 at the arXiv LaTeX source (`2410.05565`) … price $0$ GPU-s, one evening".
Book 04 registers that act as outside this canon's rules:
`04_BEDS_AND_INSTRUMENTS.md` verse **04.12**'s Evidence (`:193` at repair time) closes with *"No LaTeX source has been
read: `NOT MEASURED` — needs one fetch of the arXiv source, an author action outside this
canon's no-fetch rule"*. Pricing it "$0$ GPU-s, one evening" read as if the act were inside;
it is `NOT MEASURED — needs one author fetch of arXiv:2410.05565, an act no writer, refuter
or repairer of this canon may perform`, registered as an author action with its own row in
book 05 and not as a kill here (the same disposition verse 01.5 gives arXiv:2402.08164).
**The convention — which arithmetic the published paper ran — is undecided until that fetch
happens, and no sentence of this book asserts it.** The kill that decides the Statement is
the one book 04.12 actually ships, decided by arithmetic and not by the citation:

> **Frozen.** On the same drawn causal softmax $A$ at $\gamma=0.9$ and $\beta=1$, the two
> arms failing to differ by $O(1)$: $\|O-Y\|_\infty/\|V\|_\infty<10^{-3}$ on any draw, or
> the row sums of $A_s(I-\gamma A_s)^{-1}$-normalised ChaCAL-published reading $1.000$
> within $10^{-12}$.

Either reading refutes the Statement's "the two reads differ at every $\gamma>0$" and
collapses the two arms into one, at which point the convention question stops deciding
anything. Instrument: the lane's own two arms of 04.11, both FOUND under their own `kind`
(04.14); price $0$ GPU-s, half an evening, $512$ draws. It fires or does not on the bed as
registered — the record's instance is on the firing side, row sums $0.400\dots0.765$ against
$1.000$ (`RUN[J]`, `READ docs/CEQ_SHAPE.md:207-211 @ 99777ab`) — which is why the arithmetic
and not the fetch is the decidable half. Control: at $\gamma=0$ the two arms must agree
**bitwise** ($O-Y=(1-\gamma)\gamma A(\cdots)$ carries the factor $\gamma$), and a probe
reporting a non-zero difference at $\gamma=0$ is broken before it is read at $\gamma=0.9$.

**If killed** (the two arms do not differ by $O(1)$ at $\gamma=0.9$). The "diagonal kept"
clause of the delta is withdrawn — not renamed — and with it every sentence that makes
component (k) turn on the diagonal's position. Replacement, on a different object measured
by a different instrument at a different number: the **row-sum census on the two arms'
operators**, read before any resolvent is formed. ChaCAL-published's operator is
sub-stochastic by construction ($A_s\mathbb 1=\mathbb 1-\mathrm{diag}(A)<\mathbb 1$, 04.12's
Hypotheses), so its inverse is a finite regime-N sum
(`READ docs/CEQ_SHAPE.md:286-289 @ 99777ab`: regime N is $A^s=0$), while the shape's is a
regime-S resolvent; that is a statement about the operators' rows, not about their reads,
and it survives whatever the reads do. *Its Hypotheses:* $\beta=1$, finite logits, so
$\mathrm{diag}(A)>0$ entrywise. *Its Evidence:* `RUN[J]` row sums $0.400\dots0.765$ against
$1.000$. *Its Kill:* any drawn cell where ChaCAL-published's row sums read $1.000$ within
$10^{-12}$ — then $\mathrm{diag}(A)=0$ numerically, the two operators are the same operator,
and the regime distinction is withdrawn as well; decided by one pass over the rows, strictly
cheaper than forming either resolvent, with the diagonal-kept arm reading $1.000$ to
$10^{-12}$ as the must-fire plant. If **that** fires, the delta loses its operator clause
too and reads: CEQ's operator **is** ChaCAL's read, and the delta is (e), (g), (h), (i), (k)
of `READ docs/CEQ_SHAPE.md:260-271 @ 99777ab` alone. The InfSA-style base
(`roffo-2026-infsa` [V], ReLU Frobenius, non-causal) is **not** carried as a link of this
chain: it is `NOT MEASURED — needs the base wired, unspecified in the canon` (no verse of
book 04 wires it), so it cannot be a replacement whose kill fires, and it is kept only as
the plant named at `READ docs/CEQ_SHAPE.md:3110 @ 99777ab` for bind B-E1.

**Terminal.** The read is occupied (`READ docs/CEQ_SHAPE.md:2891-2937`, the record's own
V-7 on a literature search); the canon licenses "ChaCAL's read with the diagonal kept" as
a convention statement pending the re-read, and withdraws every sentence calling the
resolvent new (K-11).

---

### 01.14 — Absorbing rows are the limit of a self-spike: the boundary mechanism is an exactness delta, not a representability delta

**Statement.** For $a\in\mathcal A$ let $P^{(L)}$ be the causal softmax whose row $a$ has
self logit $L$ above every other entry; then $\|P^{(L)}_{a\cdot}-e_a\|_1\le2s\,e^{-L}$ and,
for the reads with every other row shared,
$$\|\Pi_\gamma(P_{\rm abs})-\Pi_\gamma(P^{(L)})\|_\infty\ \le\ \frac{2s\,e^{-L}}{1-\gamma},\qquad
\|O_{\rm abs}-O^{(L)}\|_\infty\le\frac{2s\,e^{-L}\|V\|_\infty}{1-\gamma},$$
by the mask-amplification bound (`READ docs/CEQ_SHAPE.md:404-409`). Hence every
**discounted** read with absorbing rows, at $\gamma$ bounded away from $1$, is within
$\varepsilon$ of a ChaCAL-diag read with logit margin
$L\ge\log(2s/((1-\gamma)\varepsilon))$ on the boundary rows — at $\gamma=0.9$,
$\varepsilon=10^{-6}$, $s=64$ this is
$$L\ \ge\ \ln\!\Big(\frac{2\cdot64}{0.1\cdot10^{-6}}\Big)=\ln(1.28\times10^{9})=20.9701,
\quad\text{printed as } \mathbf{20.97}\ \text{(or }21.0\text{ rounded up)},$$
**not the $20.7$ this verse printed before repair.** The $1.28$ factor had been dropped from
the logarithm: $20.7233=\ln(10^{9})$, and $20.7$ is that value truncated. The misprint is not
cosmetic — it breaks the bound it is quoted for. At $L=20.7$ the containment radius reads
$2\cdot64\,e^{-20.7}/0.1=1.31\times10^{-6}$, and at $\ln(10^9)=20.7233$ it reads exactly
$1.28\times10^{-6}$; both **exceed** the $\varepsilon=10^{-6}$ the margin is chosen to meet.
At $L=20.9701$ the radius reads $10^{-6}$ exactly (DERIVED). Because the inequality is a
**lower** bound on $L$, any rounding of this number rounds **up** and never down; $20.7$
rounded the wrong way. The verse carries the **blow-up printed**: the
containment radius is $2s\,e^{-L}/(1-\gamma)$, which at fixed $L=30$, $s=64$ reads
$1.20\times10^{-11}$ at $\gamma=0.9$, $1.20\times10^{-10}$ at $\gamma=0.99$,
$1.20\times10^{-8}$ at $\gamma=0.9999$ and **diverges as $\gamma\uparrow1$** (DERIVED:
$2\cdot64\cdot e^{-30}=1.198\times10^{-11}$, divided by $1-\gamma$). Component (e) of the
delta is therefore **not** a representability statement **on the $z$ and $\Delta z$
channels at $\gamma<1$**: there it is (1) exactness at finite parameters ($\delta=0$ against
$\varepsilon>0$) and (2) a training statement — whether AdamW finds the spike (book 02). A
column sink (K-E1's control) is a further parameterisation inside the same
$\varepsilon$-class **on those channels**.

**The committor head at $\gamma=1$ is outside the finite-logit class entirely, and this
verse's containment does not reach it.** Two readings, both against it, neither a bound:
*(a) with $\mathcal A$ declared*, the boundary rows lie outside $T$, so $\hat Q=Q$ and
$\hat R=R$ **exactly** — the spike rows never enter the transient block — and
$\|\hat q^{(L)}-q\|_\infty=0$ identically; the former clause "$\|\hat
q^{(L)}-q\|_\infty\le\kappa\cdot2s\,e^{-L}$" is then satisfied by construction (V-10) and
decides nothing. *(b) with $\mathcal A$ not declared* — the reading the class-containment
claim needs, since ChaCAL-diag declares no boundary — regime S fixes $P_{00}=1$ and
$P_{ii}<1$ for $i\ge1$ (`READ docs/CEQ_SHAPE.md:286-287 @ 99777ab`), so position $0$ is the
only absorbing state; absorption into any $\mathcal A_k$ not containing $0$ has probability
**exactly $0$**, and at $\gamma=1$ the resolvent $(I-P)^{-1}$ does not exist at all on the
full chain ($1$ is an eigenvalue of a row-stochastic $P$). The error on that reading is
$\|q\|_\infty$, an $O(1)$ quantity — the record's own instance reads
$\min_T\max_kq^{(k)}=0.548718$ (`RUN[P]`, `READ docs/CEQ_SHAPE.md:1531 @ 99777ab`), so
$\|q\|_\infty\ge0.548718$ — and **not** $\le\kappa\cdot2s\,e^{-L}$, which at $L=30$ reads
$\le9.5\times10^{-11}$ at the $\kappa\le7.9$ the earlier text assumed and
$\le3.1\times10^{-10}$ at the upper endpoint $\kappa\le25.70$ verse 01.1 derives from the
quotient at repair — wrong by ten orders and by nine, so the conclusion is unchanged and the
number it rests on is now the derived interval rather than the mis-united $7.9$. The clause is
deleted. What replaces it: the committor head is a read of a **declared partition** of an
operator, and the declaration is an architectural act, not a weight; no finite-logit
softmax realises absorption into a set other than its own $P_{ii}=1$ rows, and no margin
$L$ closes the gap because the gap is not $O(e^{-L})$. Whether component (e) is retired on
that head is therefore decided **separately** from the $z$ channel, by the kill below and
not by this verse's $\varepsilon$-class.

**Hypotheses.** Regime S, and $\gamma$ **bounded away from $1$**: the containment radius
$2s\,e^{-L}/(1-\gamma)$ is below a target $\varepsilon$ only for
$\gamma\le1-2s\,e^{-L}/\varepsilon$, which at $s=64$, $L=30$, $\varepsilon=10^{-6}$ admits
every $\gamma\le1-1.2\times10^{-5}$ and no $\gamma$ nearer $1$ than that. The committor head
at $\gamma=1$ is **excluded** by the Statement's second paragraph and inherits nothing
through verse 01.1's $\kappa$; the clause asserting that inheritance is deleted, being
either vacuous (with $\mathcal A$ declared: the error is $0$ identically) or false by $O(1)$
(without: the error is $\|q\|_\infty\ge0.548718$). Non-vacuous at every $L$ **on the $z$ and
$\Delta z$ channels only**; the class containment is exact at $L=\infty$ only, and at
$\gamma=1$ it does not hold at any $L$.

**Evidence.** DERIVED (softmax row $a$: mass off $a$ is $\le(s-1)e^{-L}/(1+(s-1)e^{-L})\le s\,e^{-L}$,
$\ell_1$ distance to $e_a$ at most twice that; then Prop. 4(ii)); the amplification
constant attained up to a constant at $s=3$, $\gamma=0.9$, $\varepsilon=0.1$ — **not the
judge's $s=32$, $\gamma=0.6$ draw**, and the draw is printed because the book's Limits
paragraph no longer covers it by a blanket sentence — (`RUN[J]` $0.5263157894736843$ against
naive $0.1$, `READ docs/CEQ_SHAPE.md:405-406 @ 99777ab`; the earlier pin `:407-408` resolves
at `99777ab` to the row-sum census and the $\hat\beta\ne1$ rows and never to these two
numbers, and is corrected here). The sink literature (`xiao-2023-attentionsinks`, `gu-2024-sinkemerges` [V])
owns the column device; `karbalayghareh-2026-doformer` [V] owns one clamped row.

**Mechanism.** V-24 (`MISTAKES.md:1658`) — an identity bind whose rejection region is
empty: the theorem says the rejection region of "boundary rows versus a spike" is empty
in representability and non-empty only in $\delta$ and in training; M-13 (`:1210`) — no
equivalence at $N=8$; D-7 — Bet F's counter is the point estimate, and this verse says
why it is representable.

**Kill.** K-E1 frozen (`READ docs/CEQ_SHAPE.md:2274`): ChaCAL-with-sink-token's committor
$\varphi$-NRMSE within $\mathrm{MDE}_8$ of the shape's on at least $6$ of $8$ seeds **and**
harmonic residual within $2\times$, holding at the escalation to $N=16$.

**The price, reconciled against the record and against book 04.** K-E1's price is
$\approx52$ s with escalation $\approx44$ s, which is the record's own Bet F row
(`READ docs/CEQ_SHAPE.md:2238 @ 99777ab`, the Bet F line), and the class is `[ASSUMED]` and
not `[FITTED]`: the record marks Bet F's seconds `[ASSUMED]` on the `arm_smprime` cell time,
and this verse printed `[FITTED]`, which claimed a fit no cell supports. Both are corrected
here. The $\approx34$ s a reader may meet in book 04 is **a different kill's price**: it is
Bet E's one bed-cell pair at `04_BEDS_AND_INSTRUMENTS.md` verse **04.6** (at repair time
`:102`, `:357`), the pair verse 01.9's K-H1 runs inside, and it is not K-E1's. K-E1 runs the
shape **and** ChaCAL-with-sink — two arms, not one bed-cell pair — which is why the two
numbers differ. Book 04's own K-E1 registration now reads $\approx52$ s `[ASSUMED]`
(verse **04.11**, at repair time `:185`), so **the two books agree at repair time and no
`CORRECTIONS.md` row is owed**; the finding is answered by reconciliation before birth and not
by a correction after it.

**And K-E1 is registered UNREACHABLE by the book that owns the arm.**
`04_BEDS_AND_INSTRUMENTS.md` verse 04.11 records that neither ChaCAL-with-sink nor
ChaCAL-diag nor ChaCAL-published exists — `grep -rniE 'chacal|infsa' --include='*.py' .`
returns $0$ lines and the $24$-kind census holds $0$ records for each — so under
`CHARTER.md:145-148` K-E1 is not a kill today, at any price. This verse's kill therefore reads
**UNREACHABLE**, its price is the price it would carry if the two arms existed, and the verse
is carried by its two links below, both of which are re-registered at repair to need no
ChaCAL arm at all.

Control: ChaCAL-diag with $\mathcal A=\emptyset$
must read $\|\Pi_{\rm shape}-\Pi_{\rm ChaCAL\text{-}diag}\|_\infty=O(1)$ on downstream
rows whenever $\mathcal A\ne\emptyset$ (rows changed $=[3]$ on the judge's draw,
`RUN[MARS]`, `:3110`) — a comparison that cannot see the boundary rows on the plant cannot
see them on the arm.

**If killed — link 1, on the $z$ channel.** Component (e) is retired **on the discounted
channels** to "a parameterisation of a column sink"; what survives by theorem is the
exactness clause — $\delta=0$ on the exact route against
$\varepsilon=2s\,e^{-\hat L}/(1-\gamma)$ printed from the trained margin $\hat L$, with the
$1/(1-\gamma)$ factor now carried into the printed number. **Its Kill has two clauses, and the
first is reachable today.**

> **Frozen (α), the margin arithmetic.** Recompute $L_{\rm req}=\ln(2s/((1-\gamma)\varepsilon))$
> at every registered $(s,\gamma,\varepsilon)$ triple this book prints, and the containment
> radius $2s\,e^{-L}/(1-\gamma)$ at every registered $(s,L,\gamma)$ grid point, in float64;
> **any printed value departing by more than $10^{-12}$ relative, or any printed $L$ at which
> the radius exceeds the $\varepsilon$ it is quoted to meet.**

> **Frozen (β), the label comparison.** $\varepsilon\|V\|_\infty$ below the label sd on $6$ of
> $8$ seeds — the exactness delta is invisible at that margin.

Clause (α) is decided by four logarithms and four exponentials on the page, CPU, no cell, no
arm, no bed: $0$ GPU-s, minutes, and **it fires on this book as printed before repair**, where
$L=20.7$ carried a radius of $1.31\times10^{-6}$ against its own $\varepsilon=10^{-6}$. Clause
(β) is `NOT MEASURED — needs the label sd (B5, book 04 verse 04.21)` and is not reachable
today. Planted negative for (α): substitute $\ln(10^9)=20.7233$ for $\ln(1.28\times10^9)$ and
the radius must read $1.28\times10^{-6}>\varepsilon$ and fire; a recomputation that reports the
dropped-factor margin as sufficient is evaluating the formula it was given rather than the
bound. Link 1 is cheaper than K-E1 on both clauses and, on (α), needs no cell at all.

**If killed — link 2, on the committor head, a different object and a different
instrument.** Link 1 says nothing about $\gamma=1$, so the head needs its own link rather
than the same one renamed. **The object is moved off the arm at repair, because book 04
registers every ChaCAL arm as non-existent** (verse 04.11: $0$ source lines, $0$ records), so
a link read on "the ChaCAL-with-sink arm's own learned operator" would die exactly where K-E1
dies and would be a replacement that does not survive its verse (V-9). The decidable object is
the **non-sink zero-fraction read on the environment chain the bed already builds**: on each
of 04.2's admitted draws, take $P_{\rm env}$ with a **value-zero sink column** appended in the
record's own `V15Fork.Asink` form (`READ lean/CEQ/V15Fork.lean:67-70`) and **no partition
declared**, solve the $K+1$ absorption channels by the same route the shape uses, and read
$\phi_0:=\Pr_{\rm draw}[\,q^{(k)}_i=0\text{ exactly for every }k\ne\text{sink and every }i\,]$.
*Its Hypotheses:* the bed's own generator as registered, $\mathcal A$ undeclared, $512$
admission draws at the design point; no training, no seeds, no matched parameters, no arm.
*Its Evidence:* DERIVED — with no absorbing row declared, the only self-absorbing position of
the drawn chain is the one regime S fixes (`READ docs/CEQ_SHAPE.md:286-287 @ 99777ab`), so
$\phi_0=1$ is the predicted reading; the column device is the record's own
(`READ lean/CEQ/V15Fork.lean:67-70`), and the two-read comparison is the column
`04_BEDS_AND_INSTRUMENTS.md` verse 04.11's link 1 already specifies for the admission census
at $0$ GPU-s with no `make_arm` branch, so this link is `NOT MEASURED — needs that census
column, $0$ GPU-s, no arm` and **not** `needs the sink arm's cells (B5)`, which is what it
read before repair. *Its Kill:* $\phi_0<1$ on any draw — some non-sink channel carries positive mass, a
finite-logit operator does produce absorption into a set it was not given, and component
(e) is retired on the committor head too. *Its plant:* an operator with a literal identity
row planted at one non-sink position must read $\phi_0<1$; a census that cannot see a
planted identity row cannot certify the absence of one. This link reads an exact-zero
predicate over draws — a **fraction of $512$ draws**, not an $\mathrm{MDE}_8$ comparison of
two NRMSEs over $8$ seeds — so it is decided by a different instrument at a different number
on a different object from K-E1, and it is strictly more decisive **and strictly cheaper**:
K-E1 asks whether two **arms** are within a detectable margin and needs both arms to exist,
while this asks whether one **chain's** channel is identically zero on draws the bed already
takes, at $0$ GPU-s and with no arm. If **both** links fire, the delta is (g), (h), (i), (k) and Tree B's sentence
(`READ docs/CEQ_SHAPE.md:2377-2381 @ 99777ab`), and the chain ends.

**Terminal.** Boundary rows are the $L\to\infty$ limit of the class the arm already contains
**on the discounted channels at $\gamma$ bounded away from $1$**, with the containment radius
$2s\,e^{-L}/(1-\gamma)$ printed and diverging at $\gamma\uparrow1$; on the committor head at
$\gamma=1$ they are not a limit of that class at all, and the head is a read of a declared
partition rather than of a learned operator. The canon licenses "exact at finite parameters,
with the margin **and the $1/(1-\gamma)$ factor** printed" on the $z$ and $\Delta z$
channels; it withdraws "a mechanism softmax cannot express" there; and on the committor head
it licenses neither that phrase nor its negation until link 2's $\phi_0$ is read — the
declaration is an architectural act, and an act is not a capability number.

---

### 01.15 — The interventional re-solve, and when the cached mixture is exact

**Statement.** For $(P,V)\to(P',V')$,
$\Delta z=(I-\gamma P')^{-1}(\Delta V+\gamma\,\Delta P\,z)$ exactly; the cached mixture
$O_{\rm cached}=\Pi_\gamma(P)V'$ equals the true post-intervention read iff
$\Delta P\,z'=0$, i.e. on value-only moves, and otherwise
$$\|O_{\rm true}-O_{\rm cached}\|_\infty=(1-\gamma)\,\big\|P'(I-\gamma P')^{-1}V'-P(I-\gamma P)^{-1}V'\big\|_\infty
\le\frac{\|\Delta P\|_\infty\,\|V'\|_\infty}{1-\gamma}\quad(\text{DERIVED, four steps}).$$
**The $(1+\gamma)$ factor is dropped at repair, and the derivation that has no room for it is
printed.** (i) $\Pi_\gamma(P)=(1-\gamma)P(I-\gamma P)^{-1}=\tfrac{1-\gamma}{\gamma}\big((I-\gamma P)^{-1}-I\big)$;
(ii) so $\Pi_\gamma(P')-\Pi_\gamma(P)=\tfrac{1-\gamma}{\gamma}\big((I-\gamma P')^{-1}-(I-\gamma P)^{-1}\big)$
and the constant $I$ cancels; (iii) the resolvent identity
$(I-\gamma P')^{-1}-(I-\gamma P)^{-1}=(I-\gamma P')^{-1}\,\gamma\,\Delta P\,(I-\gamma P)^{-1}$
turns this into $(1-\gamma)(I-\gamma P')^{-1}\Delta P\,z'$ with $z'=(I-\gamma P)^{-1}V'$;
(iv) $\|(I-\gamma P')^{-1}\|_\infty\le1/(1-\gamma)$ and $\|z'\|_\infty\le\|V'\|_\infty/(1-\gamma)$
give the display, the two $(1-\gamma)$ factors cancelling once. No step introduces $(1+\gamma)$,
and no step of the earlier text produced one; the factor was carried with no derivation behind
it and inflated the bound by up to $2\times$ at $\gamma\uparrow1$.
For a row clamp $P'=P+e_iu^\top$, $u^\top\mathbb 1=0$: $\Delta z=\gamma(Me_i)(u^\top z)/(1-\gamma u^\top Me_i)$
with denominator $(1-\gamma p'_{ii})/(1-\gamma P_{ii})>0$; for a token rewrite
$\mathrm{rank}(\Delta P)\le s-i$ — an **inequality**, corrected at repair from the equality
this book and the record's own line (`READ docs/CEQ_SHAPE.md:471 @ 99777ab`) both printed:
rewriting the token at position $i$ changes the $s-i$ rows at or after $i$ and changes no
row before it, so $\Delta P$ has at most $s-i$ non-zero rows and its rank is at most that,
with equality failing on every rewrite that leaves some later row's logits fixed (a rewrite
whose token is unattended by rows $>i$ up to the leak of 01.2). The suffix re-solve price
$(s-i)^2d/2$ is supported by the inequality alone — it is charged on the $s-i$ **rows** that
may move and not on the rank — and is unchanged.

**Hypotheses.** Row-stochastic $P,P'$, $\gamma<1$; triangularity for the suffix
identity ($P'_{<i}=P_{<i}$, $V'_{<i}=V_{<i}$). Non-vacuous whenever a move changes $P$
(census line S-17, `READ docs/CEQ_SHAPE.md:1722-1727 @ 99777ab` — a card of the record, not
a verse of book 04: both plant classes non-empty, at least $64$ draws each,
`READ docs/CEQ_SHAPE.md:1722-1727`).

**Evidence.** DERIVED; `RUN` identities $1.03\times10^{-15}$–$1.36\times10^{-15}$;
Sherman–Morrison against re-solve $1.42\times10^{-15}$ and $1.2339847026143769\times10^{-15}$
with denominator $1.9$ (`READ docs/CEQ_SHAPE.md:1521-1526`); CITED
`sherman-1950-inverse-adjustment`, `hager-1989-updating`, `schweitzer-1968-perturbation`,
`bottou-2013-counterfactual` §7.3 (the linearisation) [V]; the cached-representation
mechanism `momennejad-2017-sr`, `russek-2017-predictive` [U].

**Mechanism.** M-8 (`MISTAKES.md:544`, the arm is never priced at the oracle's rank-one
rate); D-5 (`:755`, the constant-value plant reads $\le10^{-15}$, Gaussian $V$ reads
$O(1)$); D-2; V-9 (`:154`, the re-solve must move a number: the cached control is the
repair-that-changes-nothing plant).

**Kill.** K-G1: the cached mixture within one seed sd of the shape on $P$-changing plants
on at least $6$ of $8$ seeds — the re-solve is a per-row control wearing a name;
$\approx35$ s, one evening. Control: on $V$-only plants the cached arm must match the
shape to $10^{-12}$ (the identity is exact there); a cached arm that fails on $V$-only
plants is mis-wired.

**If killed.** Component (g)'s re-solve sentence is retired; the displacement identity
survives as algebra and the channel as a cost statement ($m$ candidates at $m\cdot s^2/2$
MACs against one solve's $s^2d/2$, ratio $m/d=0.5$ at $(8,16)$, Bet G's band
$(0.5\times,8\times)$); its Kill is the measured ratio at or above $8\times$ (seconds,
V-17 of the ledger). If that fires, a re-solve per candidate is what ships and the price
is quoted as measured.

**Terminal.** Consequence as a displacement field is one extra solve, exact; the canon
licenses the identity and its price and withdraws "the re-solve is needed" wherever
K-G1 reads its counter.

---

### 01.16 — The Neumann certificate $\gamma^{K'+1}/(1-\gamma)$ is attained, in vector units, and dies at $\gamma=1$

**Statement.** For $P\ge0$ row-stochastic (identity rows included), $\gamma\in[0,1)$:
the matrix residual of the $K'$-truncated series is exactly $\gamma^{K'+1}/(1-\gamma)$ for
$(I-\gamma P)^{-1}$ and exactly $\gamma^{K'+1}$ for $\Pi_\gamma$; the printed certificate
is $\delta\,\|V\|_\infty$ with $\|V\|_\infty$ and $1/(1-\hat\gamma)$ beside it; a
sparsified $P_m$ with dropped row mass $\varepsilon$ costs $\varepsilon\|V\|_\infty/(1-\gamma)$
more; and at $\gamma=1$ on the transient block the certificate is $\infty$ — the
committor head ships on the exact route only ($\delta=0$) and no truncated committor
carries a certificate ($\|Q\|_\infty=0.958$ gives $19.2$ at $K'=4$ against a true error
$2.5\times10^{-3}$, `RUN[I]`).

**Hypotheses.** As stated; for $\|P\|_\infty>1$ the bound *can* fail (the $\hat\beta\ne1$
rows summing $1.31\dots10.29$ print no certificate, `READ V16_ARM_SMPRIME.md:28-32 @ 99777ab`).
Non-vacuous at every $\gamma>0$; at $\hat\gamma>0.99$ the factor $1/(1-\hat\gamma)>100$ and
the mirror kill K-J prints it. $K'$ is the **Neumann truncation order** with registered
support $K'\in\{1,2,4,8,16\}$, journalled per cell; it is a different object from the $K$
constraint sets of this book's Notation, which is fixed at $K=2$ at the design point, and the
two share a letter only in `CHARTER.md` §6's scope sentence. This verse and verse 01.7 wrote
$K$ for both until repair; both now write $K'$, matching `08_ARCHITECTURE.md` verse 08.11.

**Evidence.** DERIVED + `RUN` (`READ docs/CEQ_SHAPE.md:391-413`, `:853-855`); the
convergent plant rows $1.5$, $\gamma=0.6$, $K'=2$: $7.29$ against $0.54$ (`RUN[M]`); the
divergent $\gamma=0.7$ plant labelled as such.

**Mechanism.** V-3, V-10 (`err = bound` is a declared identity; the bind is the planted
non-stochastic $P$); V-17 (vector units); L-CERT.

**Kill — the reachable one, registered at repair; K-I is not this verse's kill.** K-I as
previously written ("one exceedance of $\delta\|V\|_\infty$ on any of $1{,}024$ draws, or
$\delta\|V\|_\infty$ at or above the label sd; $\approx2.6$ s") **cannot fire on the bed as
registered**, and both of its branches need an object the canon records as absent: branch 1
needs a shipped mask, and `CHARTER.md:241` (B15) records *"the kernels do not exist: chunked
solve, CSR path, Mapper schedule are `NOT MEASURED`"*; branch 2 needs the label sd, and
`CHARTER.md:231` (B5) records *"BED-S has no cell, no realised sd"*. The record itself prices
it dormant — bind B-I is *"REPAIR, **dormant** … dormant until row M ships a mask"*
(`READ docs/CEQ_SHAPE.md:3116 @ 99777ab`), $\approx2.6$ s *when a mask exists*. Under
`CHARTER.md:145-148` that is not a kill. K-I is therefore **a dormant row of book 03**, named
here and owned there, and this verse's kill is the one decidable at $0$ GPU-s today:

> **Frozen.** Recomputing the $K'$-truncated residual on the record's own registered
> $(\gamma,K')$ pairs and finding it depart from $\gamma^{K'+1}/(1-\gamma)$ by more than
> $10^{-12}$ in **relative** units on any pair — the word "attained" is withdrawn and the
> certificate is demoted from an equality to an upper bound.

Instrument: float64 recomputation of a $\le64\times64$ Neumann partial sum, CPU, no cell, no
mask, no bed; price $0$ GPU-s, half an evening. The pair the record already banks passes:
residual $0.007754350466241788$ against $\gamma^{17}/(1-\gamma)=0.007754350466240224$ at
$K'=16$, $\gamma=0.7$ — absolute departure $1.5647\times10^{-15}$, **relative
$2.0179\times10^{-13}$**, inside the frozen $10^{-12}$ (`RUN`, `READ
docs/CEQ_SHAPE.md:853-855 @ 99777ab`; the relative figure DERIVED from the printed pair).
Must-fire plant: the rows-$1.5$ non-stochastic $P$ at $\gamma=0.6$, $K'=2$ must **exceed** the
bound, $7.29$ against $0.54$ (`RUN[M]`) — an instrument that reports equality on a plant
whose row sums are $1.5$ is reading the formula and not the matrix. Second plant, in the
other direction: $\gamma=0.7$, $K'=2$ must now read $1.143$ and not $0.343$; a recomputation
returning $0.343$ has dropped the $1/(1-\gamma)$ factor, which is the defect verse 01.7's
table carried until repair.

**If killed** (the residual departs from the equality by more than $10^{-12}$ relative). The
equality clause is withdrawn and what survives is the inequality
$\|\cdot\|_\infty\le\gamma^{K'+1}/(1-\gamma)$, which is the Neumann tail bound and needs only
non-negativity and row sums $\le1$. Replacement, with a different object and a different
instrument: the certificate is re-registered as the **dropped-mass** bound
$\varepsilon\|V\|_\infty/(1-\gamma)$ read off the sparsified $P_m$'s own row sums, a
census over the operator's rows rather than a norm of its resolvent; its Hypotheses are
$P_m\ge0$ and $P_m\mathbb 1\le\mathbb 1$ alone (no equality, no attainment); its Evidence is
the row-sum census; its Kill is any row of $P_m$ with $\sum_jP_{m,ij}>1+10^{-12}$ — the
dropped-mass reading is then negative and the bound is not a bound — decided by one pass
over the rows, strictly cheaper than the resolvent recomputation. If **that** fires, no
certificate ships on any route and the mask is refused: the exact solve ships, priced
against one hop at $s=64$ (`READ docs/CEQ_SHAPE.md:2395`), and the chain ends at the
Terminal with K-9 (book 03) owning the price.

**Terminal.** The certificate is textbook (`meyer-2000-matrix`, `horn-2013-matrix`),
claimed as an instrument and never as mathematics; the canon licenses it in vector units
on Neumann routes and withdraws it on every route at $\gamma=1$.

---

### 01.17 — Lean containment: the delta's theorems with their grades

**Statement.** The delta's clauses are carried by named targets, graded `[M]` (a Mathlib
route confirmed at the pinned revision `a45ae637`, toolchain `leanprover/lean4:v4.7.0`),
`[S]` (a supporting lemma supplied in-file), `[D]` (no object in this Mathlib); every
`[M]` is pending `lake build` and nothing is cited as proved before it builds. **No row of the
table below carries `[D]` after repair**: the one that did — the hitting-time row — was graded
on a repository `grep`, and Mathlib at the book's own pin carries the object (note below the
table), so `[D]` remains defined here and is claimed nowhere in this table. The one `[D]` left
in the book is verse 01.22's, which asserts no Lean target at all.

| clause | target | grade | Mathlib name (`[V-name]` at the pin) | refusal shipped |
|---|---|---|---|---|
| parity (k) | `gamma_zero_is_softmax` | [M] | `Matrix.inv_one` | `gamma_half_is_not_softmax` |
| BOS absorbing (e) | `bos_row_is_absorbing` at $\beta=1$ | [M] | `Finset.sum_range_one` | at $\beta=0$ row 0 is $e^{qk_{00}}=2.0138$ |
| F2 (e), 01.1 | `later_boundary_unreachable`, `causal_forward_only` | [M] | `Matrix.blockTriangular_inv_of_blockTriangular` (`Block.lean:346`) | `dense_P_displaces_backward` |
| unit (e), 01.10 | `lower_triangular_isUnit`, `diag_one_sub_smul_pos` | [M] | `Matrix.det_of_lowerTriangular` (`Block.lean:265`), `Matrix.isUnit_iff_isUnit_det` (`NonsingularInverse.lean:151`) | `zero_diag_not_unit` |
| committor (h), 01.3 | `committor_is_resolvent_read` (a), `isUnit_one_sub_transient_causal` | [M] | as above | `bos_undeclared_singular` |
| displacement (g), 01.15 | `displacement_identity` | [M] | `Matrix.sub_mulVec`, `mulVec_sub` | `const_value_zero_displacement` |
| segment cut, 01.1 | `cut_makes_segment_head_absorbing`, `cut_severs_boundary_sets` | [M] | — | — |
| D-2 separation, 01.2 | `lowerTriangular_ne_symmSupport` | [M] | — | `zero_not_a_counterexample` pattern |
| certificate (i), 01.16 | `neumann_truncation_bound`, `neumann_tail_attained`, `mask_amplification` | [S] | — | `nonstochastic_breaks_bound` at $\gamma\cdot3/2<1$ |
| spike limit, 01.14 | `absorbing_row_is_spike_limit` (new: $\|P^{(L)}_{a\cdot}-e_a\|_1\le2se^{-L}$) | [S] | `Real.exp_pos`, `Finset.sum_le_card_nsmul` | `finite_logit_never_absorbing` |
| conservation, 01.12 | `reach_avoid_sum_one`, `no_goal_no_sink_forces_max_ge_inv_K` | [S] | — | `unit_needs_rho_lt_one` |
| Chebyshev prefix failure, 01.11(iii) | `chebyshev_prefix_not_optimal` (new: the two-epoch witness on `Fin 8`, $\max q(v)$ locally $d$, globally $c$, root gap $0.2$) | [M] | `decide` / `norm_num` over `ℚ` | additive plant: on $q^{(1)}$ alone both argmins are $d$ |
| Chebyshev randomisation, 01.11(iii) aside | `chebyshev_randomised_below_deterministic` (the one-epoch `Fin 4` instance: $\tfrac12$ against $1$) | [M] | `decide` / `norm_num` | — |
| reduction, 01.4 | `hop_layered_committor` (new: $q\in\{0,1\}$ and the argmax on the layered DAG) | [S] | — (see the note below the table) | `forward_pointer_reads_zero` |
| hitting time, 01.3 | `hitting_time_transform` | [S] | `MeasureTheory.hitting` (`Mathlib/Probability/Process/HittingTime.lean:51`) exists at the pin; the **absorption identity** does not | `forward_pointer_reads_zero` (01.4's, re-used: a chain with no path to $\mathcal A_k$ must read $E_i[\gamma^{\tau_k}]=0$) |

**Note on the hitting-time row, regraded at repair.** The row read `[D]` — "no object in this
Mathlib" — on the evidence of a repository `grep` returning zero files. The grade is **false at
the book's own pin**: `lean/lake-manifest.json:61` pins Mathlib at
`a45ae63747140c1b2cbad9d46f518015c047047a`, and that revision carries
`Mathlib/Probability/Process/HittingTime.lean:51`,
`noncomputable def hitting [Preorder ι] [InfSet ι] (u : ι → Ω → β) (s : Set β) (n m : ι) : Ω → ι`
(`RUN` this session: `grep -n 'noncomputable def hitting'` on the vendored package at the
pinned revision returns that line). A `grep` of the **repository's own** `lean/CEQ/` tree is
not evidence about Mathlib, and it was the only evidence the `[D]` carried (P-1); every other
row of this table cites a Mathlib `path:line`, and the four checked resolve exactly
(`Block.lean:265`, `Block.lean:346`, `NonsingularInverse.lean:151`,
`SchurComplement.lean:202`). The row is regraded **`[S]`** — a supporting lemma supplied
in-file — and the gap is named precisely rather than as an absence: Mathlib's `hitting` is a
**stopping time on a filtered stochastic process**, indexed by a `Preorder` with `InfSet` and
defined pathwise; what verse 01.3 needs is the **finite-state absorption identity**
$E_i[\gamma^{\tau_k}\mathbb 1_k]=\big((I-\gamma Q)^{-1}\gamma R_k\mathbb 1\big)_i$ on a
row-stochastic matrix with an absorbing partition, and **no route from `hitting` to that
identity exists in this Mathlib** — there is no finite-chain first-step-analysis lemma
connecting the pathwise object to a matrix resolvent. That is a statement about a missing
bridge, not about a missing object, and it is what the row now says.

**Note on the 01.4 row, rewritten at repair.** The column is headed "Mathlib name
(`[V-name]` at the pin)" and previously held `Nilpotent.pow_card_eq_zero`
(`lean/CEQ/Nilpotent.lean:77`), which is a **repository** lemma of this tree and not a
Mathlib name; it is moved out of that column. Its statement is
`theorem pow_card_eq_zero {A : Matrix (Fin n) (Fin n) R} (hA : StrictlyLower A) : A ^ n = 0`,
and `StrictlyLower` is regime N, which *"admits no absorbing row"*
(`READ docs/CEQ_SHAPE.md:288-289 @ 99777ab`). Verse 01.4's construction at the
$\varepsilon$-clause is regime S with self entries at $-L$, so $P_{ii}>0$ and the hypothesis
fails there. The route is therefore admissible on **reading (a) only** — the strict pointer
DAG at $\gamma=1$, no self entries, the walk exactly $k$ steps — which is the reading 01.4's
Hypotheses already separate out and the one `hop_layered_committor` is stated on; on reading
(b), the softmax $\varepsilon$-clause, the nilpotency route does not apply and the clause is
carried by the union bound of 01.4's Statement instead. The row's grade stays `[S]`, its
supporting lemma is supplied in-file, and the repository citation is recorded here rather
than in a column reserved for Mathlib.

**Hypotheses.** Grades are statements about statements; nothing was compiled this
session; `lake build` last exited $0$ on a warm cache with $169$ declarations across
$12$ files and zero `sorry` (`READ docs/CEQ_SHAPE.md:816-830`). Non-vacuous on every
row: a green build with a $0\%$ domain census is decoration on that bed (L-DOM), so each
row inherits the census of the verse it carries.

**Evidence.** `READ docs/CEQ_SHAPE.md:604-627`, `:2600-2626`, `:2328-2352`; Mathlib
names as recorded by card J-L0's census (`:1425-1431`); the three new targets are
DERIVED here and carry no name beyond the ones listed.

**Mechanism.** P-11 (`MISTAKES.md:1597`, an `[M]` grade cited as a build); L-LEAN;
V-25 (census beside every green tag); P-4 (a target with a `decide` witness on `Fin 4`
exists as text, not as scaffolding).

**Kill.** K-K: any `[M]` target not building by the Lean milestone (J-L18,
`READ docs/CEQ_SHAPE.md:823 @ 99777ab`, `:863`, `:1368` — a card of the record; book 04
carries no J-L18: exit $0$,
zero `sorryAx`, axioms `[propext, Classical.choice, Quot.sound]`); $0$ GPU-s, CPU
minutes, one evening. Control: a planted `sorryAx` in one file must turn the axiom print
red (the record's own must-fire, `READ MATHEMATICS.md:465-486`).

**If killed.** The row is `[S]` and never cited as proved; what survives is every float
instance as `RUN` with its plant (verse-by-verse above); its Kill is the instance itself
reading above $10^{-12}$, cheaper than a build. If an `[S]` row also fails, the clause is
DERIVED text and the sentence "machine-checked" is withdrawn for that clause only.

**Terminal.** "Machine-checked containment" is licensed exactly for the rows that build
with the three-axiom set and a non-zero census; the canon withdraws it elsewhere.

---

## Part V — The LEAPABLE fields as named theorem targets

### 01.18 — Realisation theory (Hankel–Kronecker): no finite-state realisation of the softmax resolvent read

**Statement.** *Target* `no_finite_state_dual_of_softmax_resolvent` [S]: for the causal
softmax $P$ with logits of rank $d$ and the map $V\mapsto z=(I-\gamma P)^{-1}V$, the
Hankel block $H_i=[(I-\gamma P)^{-1}]_{\ge i,<i}$ has rank exceeding any fixed $e$ for
generic logits as $s$ grows, so no linear recurrence of state dimension $e$ reproduces
the read on all inputs; the minimal state at position $i$ is $\mathrm{rank}\,H_i$
(Kronecker–Fliess, `fliess-1974-hankel` [V]; the $L^\infty$ error of the best rank-$e$
Hankel approximation is $\sigma_{e+1}(H_i)$, `glover-1984-hankel` [V]). The consequence
for the canon: the chunked-WY native skyline (`yang-2024-deltanet`, `dao-2024-ssd` [V])
is closed to the softmax corner by `hu-2025-ssdtheory` [V] (rank $T$), and the read is
priced as a full triangular solve (`READ docs/CEQ_SHAPE.md:583-590`).

**Hypotheses — the geometry at which the statement first bites, printed.** Generic logits;
"generic" is a measured predicate — the numerical Hankel rank at tolerance $\tau$ on drawn
$\hat P$. $H_i$ is the $(s-i)\times i$ lower-left block of a lower-triangular matrix, so
$$\mathrm{rank}\,H_i\ \le\ \min(i,\,s-i)\ \le\ \lfloor s/2\rfloor\quad\text{for every }i,$$
and the non-vacuity clause "that rank exceeds $2d=32$" therefore requires
$\lfloor s/2\rfloor\ge2d+1=33$, i.e. $s\ge4d+2=66$ at $d=16$ (DERIVED). **At the planned
geometry $s=64$ the clause is unsatisfiable**: $\max_i\min(i,s-i)=32=2d$ exactly, so
$\mathrm{rank}\,H_i\le32$ at every $i$ and every draw. At $s=4d+1=65$ it is still
unsatisfiable — $\lfloor65/2\rfloor=32=2d$ — so the first geometry at which the statement
bites is $s=66$, $i=33$, where $H_{33}$ is $33\times33$ and its rank may exceed $2d$. The
probe is therefore registered at $s=66$, $i=33$, $1{,}024$ draws, and not at $s=64$,
$i=s/2$. It is `NOT MEASURED — needs the Hankel-rank probe on drawn $\hat P$,
**unspecified in the canon**: a grep for "Hankel-rank" in `04_BEDS_AND_INSTRUMENTS.md`
counts $0$, and its two "Hankel" hits are BED-K's ceiling $\sqrt{1-1/d}$ in verse 04.3, a
different object; the shape stated here — SVD of $H_{33}$ at $s=66$ over $1{,}024$ draws,
numerical rank at $\tau=10^{-6}$ — is a specification this verse writes and book 04 does not
carry`. The record's `1/d` Hankel floor is BED-K's and binds BED-K only
(`READ V20_R15_LEAP_LEDGER.md:323 @ 99777ab`).

**Evidence.** CITED as above (`[V]`, `[V-eq]` owed for Fliess and Glover); the rank-$T$
closure of `hu-2025-ssdtheory` [V]; the field named at `READ docs/CEQ_SHAPE.md:1220`
(Q1/W3, LEAPABLE by both offices).

**Mechanism.** P-7 (the field now names the object it is about); V-25; M-8 (a
finite-state dual would re-price the solve, so its absence is a pricing hypothesis).

**Kill — the $s=64$ form is struck as satisfied by construction; the kill is registered at
$s=66$.** The previous kill read "a linear recurrence of state dimension $e\le2d=32$
reproducing $z$ to $10^{-12}$ on $1{,}024$ drawn $(\hat P,V)$ at $s=64$", with the probe
specified as the SVD of $H_i$ at $i=s/2$. At $s=64$ that $H_{32}$ is $32\times32$, so
$\mathrm{rank}\,H_i\le32=2d$ on **every** draw, a time-varying linear realisation with
$e=32$ exists for every $(\hat P,V)$, and the kill fires on $1{,}024$ of $1{,}024$ draws
whatever the operator is — a gate satisfied by construction (V-10) whose own non-vacuity
predicate is unsatisfiable at the same geometry (V-11). Frozen replacement:

> **Frozen.** At $s=66$, $d=16$, $i=33$: the numerical rank of $H_{33}$ at
> $\tau=10^{-6}$ reading $\le2d=32$ on any of $1{,}024$ draws — the read has a $32$-state
> dual at the first geometry where the question is open, and the non-realisability sentence
> is withdrawn together with the pricing that rests on it.

Instrument: SVD of a $33\times33$ block, CPU float64, no cell; price $0$ GPU-s, one
evening — but **unreachable until book 04 carries the probe** (Hypotheses above). Control:
the linear corner ($\beta=0$, $g\equiv0$) at $d=16$ must read numerical rank $\le d=16$ at
the same $\tau$ (it has a finite-state dual by construction); a probe that reads full rank
on the linear corner cannot certify the softmax corner's non-realisability. Second control,
against the defect just repaired: run the probe at $s=64$, $i=32$ and it **must** read rank
$\le32$ on every draw — a probe that reports rank $33$ at $s=64$ is reading noise above
$\tau$, not rank.

**At the planned geometry the statement is not decidable in either direction**, and that,
not a measured rank, is what this verse licenses at $s=64$.

**If killed.** The statement moves to numerical rank at tolerance $\tau$: the read is
$\sigma_{e+1}(H_i)$-approximable by an $e$-state recurrence, and T1's "one solve versus
$L$ layers" is re-priced against the $e$-state scan (book 03); its Kill is the cost-law
band K-9. If that dies, the read is priced as the full solve and no finite-state claim is
made in either direction.

**Terminal.** The canon licenses "no finite-state dual for row softmax" as a citation
with its rank statement, and withdraws every sentence pricing the shape against a scan
it has not been shown to lack. At $s=64$, $d=16$ it licenses **neither** the statement nor
its negation: $\mathrm{rank}\,H_i\le\lfloor s/2\rfloor=2d$ holds identically there, so the
question is closed by arithmetic before any draw is taken, and the field enters the canon
only at $s\ge4d+2=66$.

---

### 01.19 — Koopman spectral theory reduces to the diagonal on the causal class

**Statement.** *Target* `spectrum_of_lowerTriangular_eq_diag` [M]: the Koopman operator
of the affine map $T:x\mapsto V+\gamma Px$ on linear observables is
$\gamma P^\top$, **affine by $c^\top V$** — not $\gamma P$, which the verse asserted before
repair. The one-line derivation, printed because the identification and not the conclusion
was wrong: for $f(x)=c^\top x$,
$$(Uf)(x)=f(T(x))=c^\top(V+\gamma Px)=\gamma\,(P^\top c)^\top x+c^\top V,$$
so $U$ acts on the coefficient vector as $c\mapsto\gamma P^\top c$ plus the constant
$c^\top V$, and the linear observables are an invariant subspace only modulo constants
(DERIVED). The **spectral conclusion is unchanged**, and this is why: $\mathrm{spec}(P^\top)
=\mathrm{spec}(P)$, the characteristic polynomials being equal by
$\det(\lambda I-P^\top)=\det((\lambda I-P)^\top)=\det(\lambda I-P)$, so the diagonal read
below is read off $P$ exactly as before
(`koopman-1931-hamiltonian`, `mezic-2005-spectral`, `brunton-2022-koopman` [V]); on the
causal class $\gamma P$ is lower-triangular ($\gamma P^\top$ upper-triangular, same
diagonal) and the spectrum is $\{\gamma P_{ii}\}$, so the
field's content on the arm is the diagonal read of verse 01.10(a) and nothing more. The
record's Q3/W1 field (transfer-operator / Koopman, `READ docs/CEQ_SHAPE.md:1222`) is
therefore attached to a one-line theorem, and the instrument swap it required —
`lambda_hat_live` at `scripts/v15_r1.py:384`, not `lambda_hat` at `:383`, which averages
$\log m$ and reads $-\infty$ on one $m_k=0$ — is a harness fact (B16, book 05).

**Hypotheses.** Regime S, finite logits. Non-vacuous at every draw of the class; on the
oracle chains (non-triangular $Q$) the spectrum is not the diagonal and Koopman has
content there only.

**Evidence.** DERIVED (the characteristic polynomial of a triangular matrix is
$\prod(\lambda-P_{ii})$; Mathlib route `Matrix.det_of_lowerTriangular` `[V-name]`, the
charpoly form `[U]`); `RUN[J]` $\rho(Q)=\max_TP_{ii}=0.6926596893360386$.

**Mechanism.** P-7; V-25; V-17 (the `lambda_hat` one-bit defect, `READ
docs/CEQ_SHAPE.md:1280`).

**Kill — the strict-upper check is struck as this verse's kill; the row-sum census
replaces it.** The previous kill ("any trained cell whose $\hat P$ has a non-zero entry
above the diagonal by `torch.equal` on the strict upper block") is **satisfied by
construction** (V-10) and its firing set is empty: a causally masked softmax writes
$-\infty$ into the strict upper block before the exponential, so $\exp(-\infty)=0.0$ exactly
and the block is bitwise zero on every draw of the class as registered. The verse's own
former If-killed conceded it ("the arm is mis-masked") and then re-ran the identical check,
which is a V-9 replacement. Both are struck. The frozen kill is the condition that can fire
on this class and has already fired once in the record:

> **Frozen.** The row-sum census on every trained cell: any cell with
> $|\sum_j\hat P_{ij}-1|>10^{-12}$ for some $i$, or with $\gamma\max_T\hat P_{ii}\ge1$.

The gate is decidable by one pass over the saved $\hat P$; $0$ GPU-s, half an evening,
$N=8$ cells. **It fires on the registered class:** the arm at $\hat\beta\ne1$ reads row sums
$1.31$ to $10.29$ (`READ V16_ARM_SMPRIME.md:28-32 @ 99777ab`), so $\gamma\hat P$ is not
sub-stochastic, the Neumann series does not converge from row-stochasticity, and the
diagonal read of the spectrum bounds nothing about $\|(\gamma\hat P)^t\|$ — the Koopman
sentence needs $\rho(\gamma\hat P)<1$ *and* an operator whose powers the spectral radius
controls, and $\hat\beta$ is learnable per instance (Ruling 2/2a). Triangularity survives
the census; the spectral *identity* $\mathrm{spec}(\gamma\hat P)=\{\gamma\hat P_{ii}\}$ is
algebra and no draw refutes it, which is exactly why it is not the kill. Control: the
$\beta=1$ masked-softmax arm must read row sums $1.000$ to $10^{-12}$ on every cell — a
census blind to the $10.29$ rows would also be blind to the $1.000$ rows and is not a
census. Second plant: a cell with $\hat\gamma\ge1/\max_T\hat P_{ii}$ must fire the second
clause; the record's judge draw ($\max_T\hat P_{ii}=0.6927$, `RUN[J]`) needs
$\hat\gamma\ge1.4436$ to fire it and does not, so the clause is registered as reachable
only through the $\hat\gamma$ dial, whose corner descent is book 02's B12.

**If killed** (a cell reads row sums off $1$, or $\gamma\max_T\hat P_{ii}\ge1$). The
spectrum statement stands as algebra and every *resolvent* sentence at that cell is
withdrawn — the field is decoration there, not merely uninformative. Replacement, on a
different object and with a different number: the **admitted-cell fraction**, the count
$n_{\rm adm}/N$ of cells passing the census, printed beside every Koopman sentence, so the
field's reach is a number and not an assumption. Its Hypotheses: the census is run on all
$N=8$ cells before any Koopman sentence is written. Its Evidence: the census counts (`NOT
MEASURED — needs a BED-S cell, B5`). Its Kill: $n_{\rm adm}/N=0$ — the class as *trained* is
empty and the Koopman field is withdrawn from BED-S entirely rather than restricted;
decided by reading one integer already computed by the census, strictly cheaper than the
census itself. The chain ends there: at $n_{\rm adm}=0$ there is nothing left to restrict.

**Terminal.** Koopman spectral theory is decoration on the causal class; the canon
licenses the diagonal read and withdraws the field from every BED-S sentence.

---

### 01.20 — Projection-operator formalism: absorbing elimination is memoryless, transient elimination is a Schur complement

**Statement.** *Target* `absorbing_projection_memoryless` [S]: projecting the linear chain
onto $T$ with $\mathcal A$ absorbing yields the closed equation $q=Qq+R\mathbb 1$ with no
memory kernel, because $P_{\mathcal AT}=0$ (Mori–Zwanzig's memory term
$\sum_tQ_{T\mathcal A}P_{\mathcal A\mathcal A}^tP_{\mathcal AT}$ vanishes identically;
`mori-1965-transport`, `zwanzig-1961-memory` [V]). Eliminating a *transient* subset
$T'\subset T$ instead gives the exact Schur complement
$$Q_{\rm eff}=Q_{T''T''}+Q_{T''T'}(I-Q_{T'T'})^{-1}Q_{T'T''},\qquad
R_{\rm eff}=R_{T''}+Q_{T''T'}(I-Q_{T'T'})^{-1}R_{T'},$$
which is the exact coarsening of the read and the object any Mapper/CSR schedule
approximates (book 03); dropping the coupling block $Q_{T''T'}$ costs
$\|Q_{T''T'}\|_\infty\|(I-Q_{T'T'})^{-1}\|_\infty(\|Q\|_\infty+\|R\|_\infty)$ in the
effective operator (DERIVED).

**Hypotheses.** $I-Q_{T'T'}$ a unit. **The condition this was written on was satisfied by
construction and is replaced by the reachable one (V-10).** The earlier text read "holds on the
causal class by verse 01.10's branch (a) whenever $0\notin T'$"; this book's Notation declares
$0\in\mathcal A_{\rm sink}$ (`READ docs/CEQ_SHAPE.md:295 @ 99777ab`), so $0\notin T$ and hence
$0\notin T'$ for **every** $T'\subset T$ at **every** draw — the antecedent could not fail and
the If-killed below repaired a failure that could not occur. The reachable condition it stood
in for is derived here: on the causal class $Q_{T'T'}$ is lower-triangular, so
$$\det(I-Q_{T'T'})=\prod_{i\in T'}(1-\hat P_{ii}),$$
and $I-Q_{T'T'}$ fails to be a unit **exactly when some $i\in T'$ has $\hat P_{ii}=1$**
(DERIVED). Finite softmax logits forbid that — $\hat P_{ii}=1$ needs an infinite logit gap —
and the one mechanism in this canon that produces it is a **saturated gate**, verse 01.17's
`cut_makes_segment_head_absorbing`, which writes an exact $1.0$ on the diagonal and makes the
segment head absorbing without a declaration. That is the census line this verse registers:
> **Census (reachable, $0$ GPU-s).** $\max_{i\in T}\hat P_{ii}<1$ on every cell, read by the
> row-sum and diagonal pass of verse 01.19 on the saved $\hat P$. It reads
> $0.6926596893360386$ at the judge's draw (`RUN[J]`), so it passes there and is registered as
> firing only on a saturated-gate cell.

Non-vacuous at every draw; the coupling norm is the certificate
input and is `NOT MEASURED — needs the Schur instrument on drawn $\hat P$,
**unspecified in the canon**: a grep for "Schur" in `04_BEDS_AND_INSTRUMENTS.md` counts $0$
across verses 04.1–04.21, so the shape stated here — form $Q_{T''T'}$, $(I-Q_{T'T'})^{-1}$,
$Q_{T'T''}$ on a drawn $\hat P$ and read $\|Q_{T''T'}\|_\infty\|(I-Q_{T'T'})^{-1}\|_\infty$
— is a specification this verse writes and book 04 does not carry; this verse's kill is
unreachable until it does`.

**Evidence.** DERIVED (block elimination; Mathlib route `inv_fromBlocks_zero₂₁_of_isUnit_iff`
at `SchurComplement.lean:202`, `[V-name]` per J-L6); the far-field mean-pool bound is the
only printed coarsening bound in the tree (`READ ceq/multizoom.py:38-49`, `READ
docs/CEQ_SHAPE.md:2637`).

**Mechanism.** P-7; L-CERT (a dropped tile without its Schur bound is refused); V-10.

**Kill.** The Schur identity off $(I-Q)^{-1}R\mathbb 1$ restricted to $T''$ by more than
$10^{-12}$ on any of $512$ draws — the elimination is mis-indexed; $0$ GPU-s. Control:
eliminating $T'=\emptyset$ must return $Q$ exactly, and eliminating all of $T\setminus\{i\}$
must return the scalar $q_i$.

**If killed** (the Schur residual exceeds $10^{-12}$ on some draw). **The earlier replacement
is struck, not renamed (V-9).** It read "the identity re-derived with the sink block kept
explicit … its Kill is the same residual with $0\notin T'$ enforced by a census line": its
kill was the *same residual* on the *same object* read by the *same instrument*, and its
enforcement clause was the condition just shown to hold at every draw — it therefore died to
this verse's kill exactly as the verse did, and it repaired a failure that cannot occur.

*Replacement, derived here on a different object, with a different instrument and a different
number.* The residual has two separable causes and the census line above separates them at
$0$ GPU-s. Where the census **fires** — some $i\in T'$ with $\hat P_{ii}=1$ — the defect is not
the elimination but the operator: a saturated gate has made position $i$ absorbing without a
declaration, $\det(I-Q_{T'T'})=0$ and no elimination on that cell is exact. The route is then
the **declared-sink repair** of verse 01.1's If-killed: the saturated position is added to
$\mathcal A_{\rm sink}$, the transient block is re-formed, and the Schur complement is read on
the re-declared $T$. *Its object* is the boundary declaration and the diagonal of $\hat P$,
not the eliminated blocks. *Its instrument* is verse 01.19's row-sum and diagonal pass, not
the $512$-draw Schur residual battery. *Its number* is $\max_{i\in T}\hat P_{ii}$ against $1$,
which reads $0.6926596893360386$ at the judge's draw (`RUN[J]`) and $\rho(Q)=0.692660$
declared against $1.000000$ undeclared (`RUN[P]`, `READ docs/CEQ_SHAPE.md:1440 @ 99777ab`) —
not a residual at all. *Its Hypotheses:* the census runs on the saved $\hat P$ before any
elimination, on all $N=8$ cells. *Its Evidence:* the two `RUN` numbers above; the per-cell
census itself `NOT MEASURED — needs a BED-S cell (B5)`. *Its Kill:* the census reading
$\max_{i\in T}\hat P_{ii}<1$ on **every** cell while the Schur residual still exceeds
$10^{-12}$ — the operator is clean, no saturated gate exists, and the declared-sink route
explains nothing; decided by comparing one already-computed float against $1$, strictly
cheaper than the residual battery it follows. *Its plant:* a $\hat P$ with one diagonal entry
set to $1.0$ must fire the census, and the $\beta=1$ masked-softmax draw must not; a census
blind to an exact $1.0$ is not a census. If that kill fires the chain ends at the Terminal and
the Schur complement is withdrawn as a certificate input, the coarsening reverting to the full
transient solve with L-CERT refusing every dropped tile.

**Terminal.** The projection formalism adds one exact identity (the Schur complement) and
one vanishing (the memory kernel on absorbing sets); the canon licenses both as
certificate inputs and withdraws "memory" as a mechanism of the read.

---

### 01.21 — Kolmogorov $n$-width: the read's approximability by a rank-$d$ bottleneck

**Statement.** *Target* `nwidth_of_linear_image` [D]: for fixed $P,\gamma$ the set
$\mathcal K_\gamma=\{\Pi_\gamma V:\|V\|_2\le1\}$ is a linear image of the unit ball, so its
Kolmogorov $n$-width in $\ell_2$ is $d_n(\mathcal K_\gamma)=\sigma_{n+1}(\Pi_\gamma)$ —
**DERIVED at repair, in two lines, so the equality no longer rests on an owed citation.** Let
$\Pi_\gamma=\sum_j\sigma_ju_jv_j^\top$ be the singular value decomposition. For any $n$-dimensional
subspace $X_n\subset\ell_2$, $\sup_{x\in\mathcal K_\gamma}\mathrm{dist}(x,X_n)
=\sup_{\|V\|_2\le1}\|(I-\Pi_{X_n})\Pi_\gamma V\|_2=\|(I-\Pi_{X_n})\Pi_\gamma\|_2$, the operator
norm of $\Pi_\gamma$ composed with a rank-$\le n$ orthogonal complement projection; minimising
over $X_n$ is Schmidt–Eckart–Young in the operator norm, whose minimum is $\sigma_{n+1}$ and
whose minimiser is the span of $u_1,\dots,u_n$. The grade is therefore **DERIVED**, not
`[ASSUMED textbook]`; Pinkus, *n-Widths in Approximation Theory*, is recorded as the
standard reference and its citation is still **owed** in `references.bib`, but no clause of
this book now rests on it. The Q5/W1 field (`READ docs/CEQ_SHAPE.md:1226`,
"approximation theory / Kolmogorov $n$-width, not rate–distortion") therefore reduces to
the singular-value tail of $\Pi_\gamma$: a depth-1 softmax whose value path has rank
$d=16$ loses at most $\sigma_{17}(\Pi_\gamma)$ on the $z$ channel relative to the exact
read *if* it could realise $\Pi_\gamma$'s weights — which verse 01.5 says it cannot below
the composition floor — so the width bound is an upper bound on what the rank wall costs
and never a lower bound on softmax's error.

**Hypotheses.** Fixed $P$; the $z$ channel only (the committor head is not rank-limited).
Non-vacuous where $\sigma_{d+1}(\Pi_\gamma)/\sigma_1(\Pi_\gamma)$ exceeds the label sd in
relative units; `NOT MEASURED — needs the SVD of $\hat\Pi_\gamma$ on trained cells,
**unspecified in the canon**: a grep for "SVD" in `04_BEDS_AND_INSTRUMENTS.md` counts $0$
across verses 04.1–04.21, so the shape stated here — form $\hat\Pi_\gamma$ from a saved cell
and read $\sigma_{17}/\sigma_1$ — is a specification this verse writes and book 04 does not
carry; and the label sd it is compared against is itself absent (B5). This verse's kill is
unreachable on both counts until book 04 ships the probe and 04.21 ships the first cell`.

**Evidence.** DERIVED (the two-line singular-value derivation in the Statement, replacing the
`[ASSUMED textbook]` grade the verse carried); the field correction at
`READ docs/CEQ_SHAPE.md:1226` (both offices against the table's rate–distortion). Pinkus is
recorded as the standard reference and is **owed** in `references.bib`; nothing here is
load-bearing on it, and that sentence is now true rather than asserted alongside an
`[ASSUMED]` equality it contradicted.

**Mechanism.** P-7; V-10 (an upper bound on the loss is never read as a separation).

**Kill.** $\sigma_{17}(\hat\Pi_\gamma)/\sigma_1(\hat\Pi_\gamma)<10^{-3}$ on at least $6$ of
$8$ trained cells at the design point — the read is rank-$16$ approximable and A1 on the
$z$ channel is unreachable at $d=16$: the rank wall costs nothing and only the weight
realisation (01.5) can separate the arms; $0$ GPU-s from saved cells.

**Control — the pass-plant is demoted and the must-fire plant supplied.** The verse's only
control was "the identity $\Pi_0=P$ at $\gamma=0$ must give the same tail as the softmax
matrix itself". At $\gamma=0$, $\Pi_0=(1-0)P(I-0\cdot P)^{-1}=P$ **identically**, so that
control passes on every input, including a probe whose singular values are noise: it has an
empty rejection region and is exactly the defect V-15 names — no condemning rule without a
planted negative. It is kept, relabelled as what it is, a **wiring check** that the probe
forms $\Pi_\gamma$ from the right matrix, and it is not the plant. The plant that makes the
kill fire, added at repair:

> **Must-fire.** Feed the probe a matrix with a **known** rank-$16$ truncation error:
> $M=R+10^{-2}E$ with $R$ of exact rank $16$ and $E$ random, normalised to
> $\|E\|_2=\|R\|_2$. The probe must read $\sigma_{17}(M)/\sigma_1(M)$ **near $10^{-2}$ and
> never below $10^{-3}$**; a probe reporting the frozen kill's $<10^{-3}$ on a matrix whose
> rank-$16$ residual is $10^{-2}$ by construction cannot certify the reading it is asked for.

Second plant, in the direction the kill fires: an exactly rank-$16$ matrix must read
$\sigma_{17}/\sigma_1$ at the float64 noise level and **fire** the kill — a gate that cannot
fire on a matrix that is rank-$16$ by construction is not the gate this verse registers. Both
plants are $0$ GPU-s and CPU-only, and both are `NOT MEASURED` for the same reason the kill
is: the SVD probe is unspecified in the canon (Hypotheses above).

**If killed.** The A1 sentence on the $z$ channel is withdrawn at $d=16$ and the
capability claim moves to the committor head (verse 01.1's $\kappa$-bound, no rank
bottleneck); its Kill is Bet F / K-E1 (verse 01.14). The chain ends.

**Terminal.** The $n$-width names what a rank bottleneck costs and never what softmax
cannot do; the canon licenses the singular-value tail as a printed number and withdraws
the field as a separation argument.

---

### 01.22 — Finite-mixture inference: a pooled mean over a bimodal seed population is not a component statistic

**Statement.** *Target* none in Lean (`[D]`, no object); the theorem is arithmetic. If
$N$ seeds split into two clusters with means $\mu_1,\mu_2$, weights $\pi,1-\pi$ and
within-cluster variance $\sigma_w^2$, the pooled variance is
$\sigma_{\rm pool}^2=\sigma_w^2+\pi(1-\pi)(\mu_1-\mu_2)^2$, so an $\mathrm{MDE}_N$ computed
from $\sigma_{\rm pool}$ overstates each component's resolution by the factor
$\sigma_{\rm pool}/\sigma_w$. The record's R1 instance: `arm_pl` at $N=8$ split
$5$ crossed ($0.634002$ to $0.662021$) and $3$ NO READING ($1.113403$, $1.139404$,
$1.152430$), nothing between $0.663$ and $1.113$, pooled sd $0.253673$
(`READ docs/CEQ_SHAPE.md:927-940`). **Both cluster bounds are computed by one method at
repair.** The maximum sample sd (denominator $n-1$) of $n$ points in an interval of width $w$
is attained by splitting them between the endpoints, $\lceil n/2\rceil$ at one and
$\lfloor n/2\rfloor$ at the other, and equals
$$\sigma_w^{\max}(n,w)=w\,\frac{\sqrt{\lceil n/2\rceil\lfloor n/2\rfloor/n}}{\sqrt{n-1}}
=\begin{cases}w/\sqrt3=0.5774\,w,&n=3\\ w\sqrt{0.3}=0.5477\,w,&n=5\end{cases}\qquad(\text{DERIVED}).$$
Three points in width $0.039027$ therefore have sample sd at most
$0.039027/\sqrt3=0.0225322$, and **five** in width $0.028019$ at most
$0.5477226\times0.028019=0.0153454$ — **not** $0.0280$, which is the interval width itself and
not a sample sd, and which the earlier line used for the five-point cluster while using the
correct $w/\sqrt3$ form for the three-point one: one sentence, two methods. With one method,
$$\frac{\sigma_{\rm pool}}{\sigma_w}\ \ge\ \frac{0.253673}{0.0153454}=16.53\ \text{ for the
crossed cluster }(n=5),\qquad
\ge\ \frac{0.253673}{0.0225322}=11.26\ \text{ for the NO-READING cluster }(n=3),$$
against the earlier $9.06$ and $11.3$. The direction was safe — the claim is a lower bound and
the old number was the looser one — and the defect is the split method, which is recorded here
rather than repaired away. The pooled resolution $\Delta=0.215326$ is at least **eleven**
times the within-component resolution on either cluster and at least sixteen on the crossed
one, and the mean $0.829151$ is a value no seed produced.

**Hypotheses.** Two clusters, identified by a gap wider than both cluster widths ($0.450$
against $0.039$ and $0.028$). Non-vacuous on R1 and on any BED-S cell whose seeds split.

**Evidence.** `READ` as above; DERIVED. The Q5/W3 field (`READ docs/CEQ_SHAPE.md:1227`,
finite-mixture inference against "a mean over a bimodal population is not a statement
about either mode") is thereby a scoring rule, which is what MARS ruled it.

**Mechanism.** V-26 (`MISTAKES.md:2190`) — the pairing and the split are joint claims;
M-9 (`:580`, $N=8$ verdicts are sign tests); M-3 (`:464`, the realised spread is the
pooled one only if the population is unimodal); D-CALIB.

**Kill.** On any arena cell, a headline mean printed without $n^+$ (the seed-agreement
count) and without the cluster membership beside it; assembly rule, $0$ GPU-s. The
decidable statistic: the marginal-preserving permutation of seed labels between the two
arms leaving the pairing assertion green (V-26's own check, one line). Control: the R1
cell must read $5/8$ and $3/8$ with the gap $[0.663,1.113]$ empty.

**If killed** (a cell prints a pooled mean). The number is struck and re-filed as a split
with $n^+$; the replacement's Kill is the same permutation check. The chain ends.

**Terminal.** Every capability number in the canon is a split with $n^+$ beside it; the
canon withdraws every pooled mean over a bimodal seed population.

---

## Census rows closed

| row | what is broken (`CHARTER.md` §5) | closed by verse |
|---|---|---|
| B4 | the label class: one scalar at one position; Q6 F4 for want of a state axis | 01.1 (the four vector labels and their exactness conditions), 01.2 (what the environment costs), 01.3 (the committor read); the bed itself is book 04 |
| B7 | D-APPROX: "cannot represent" without a bound | 01.5 (Peng through the reduction, with its table), 01.6 (the capacity form with its measured hypothesis), 01.7 (the iteration rate, the only theorem-shaped A1 at the planned geometry) |
| B8 | obstruction theorems vacuous at the record's geometry | 01.5 (the vacuity table and the geometry at which each bites), 01.8 (Sanford/Chen/Yehudai with their status) |
| B9 | the Cheeger sentence on a non-reversible $P$ | 01.10 — **OPEN**: branch (a)'s kill struck as satisfied by construction, branch (b)'s instrument unspecified in book 04; the Terminal is in force |
| B10 | Bellman optimality in multichain constrained MDPs | 01.11 — closed by the **criterion** distinction (total-reward absorption on a transient DAG block versus the average-reward LP Misra's hazard is about), **not** by denying the hazard's hypothesis: the chain at the design point has $\ge4$ recurrent classes and is multichain |
| B11 | the $\mathrm{hop}_k\to$ committor reduction unwritten; the TD-skyline claim on two abstracts | 01.4 (written, with the stall-mass bound), 01.7 (the iteration construction DERIVED, the abstracts demoted to lineage), 01.8 |
| B21 | the read is occupied by ChaCAL; the delta and K-E1 | 01.13, 01.14, 01.15, 01.16, 01.17 |
| B22 | the bibliography verified at the identifier; ChaCAL's diagonal on one fetch | 01.13 (the theorem that makes the convention decisive, and the re-read as its kill), 01.5 and 01.8 (`[V-eq]` owed, stated as conditional) |

## Kills, cheapest first

Reachability is a column, not a footnote: a kill whose instrument the canon does not specify
is marked **UNREACHABLE** and is not walkable this evening, whatever its price reads.

| verse | kill | price | reachable today | replacement verse or route |
|---|---|---|---|---|
| 01.19 | row-sum census: $\lvert\sum_j\hat P_{ij}-1\rvert>10^{-12}$, or $\gamma\max_T\hat P_{ii}\ge1$ | $0$ GPU-s, half an evening | yes — it has already fired (rows $1.31$–$10.29$) | the admitted-cell fraction $n_{\rm adm}/N$, then $n_{\rm adm}=0$ |
| 01.10(a) | **struck**: satisfied by construction on a triangular $\hat P$ | — | no kill; the triangularity census is an admission line, UNSPECIFIED in book 04 | none — branch (a) is algebra |
| 01.12(a) | conservation row off $\mathbb 1$ by $>10^{-12}$ | $0$ GPU-s | yes | sink enlarged, killed by $n_{\rm abs}=0$ on a still-failing draw (a diagonal count, not the residual) |
| 01.12(b) | $0.875$ printed as the argmin floor without "at uniform $\hat\pi$", or a capability number without $\hat\pi$ in the row | $0$ GPU-s (assembly) | yes — it fired on this verse before repair | 04.3's floor kill, routed by verse id |
| 01.20 | Schur identity off by $>10^{-12}$ | $0$ GPU-s | **UNREACHABLE** — Schur instrument unspecified in book 04 | sink block kept explicit |
| 01.22 | a pooled mean without $n^+$ | $0$ GPU-s (assembly) | yes | the split with $n^+$ |
| 01.3 | wrong-set plant moving $q$ by $<10^{-3}$ | $0$ GPU-s, half an evening | yes | X-4 as an admission line, then S-10 re-drawn |
| 01.16 | residual departing from $\gamma^{K'+1}/(1-\gamma)$ by $>10^{-12}$ relative | $0$ GPU-s, half an evening | yes — the banked pair reads $2.0179\times10^{-13}$ | the dropped-mass bound by row-sum census; then the exact solve |
| 01.7 | the formed matrix identity departing from $\gamma^{K'+1}/(1-\gamma)$ by $>10^{-12}$ relative at $\gamma\in\{0.6,0.9\}$, or any printed table cell doing so | $0$ GPU-s, half an evening | yes — the pre-repair $\gamma=0.7$ row would have fired it at relative $0.7$ | the cost sentence (T1, book 03), killed by K-9's band |
| 01.1 | any label identity $>10^{-12}$ on $512$ draws | $0$ GPU-s, one evening | yes | per-segment sets (01.17's cut theorem) |
| 01.4 | oracle argmax $\ne\pi^8(i)$ on $512$ instances | $0$ GPU-s, one evening | yes | layout predicate; then oracle-family-only transfer |
| 01.13 | two arms not differing by $O(1)$ at $\gamma=0.9$: $\|O-Y\|_\infty/\|V\|_\infty<10^{-3}$ | $0$ GPU-s, half an evening | yes (the arXiv re-read is repriced as an author action, book 05) | row-sum census on the two operators; then the delta loses its operator clause |
| 01.11(iii) | the two-epoch witness's local and global argmins agreeing, or the root gap $0.2$ reading $<10^{-12}$ | $0$ GPU-s, minutes | yes | — (clause-level) |
| 01.11(i) | $m_{\rm eff}\le1$ on $>0.05$ of the $512$ admission draws (04.2 census clauses 9, 11) | $0$ GPU-s | yes — 04.2 carries the census | — (clause-level) |
| 01.11(ii) | DP against exhaustive enumeration at $\lvert T\rvert\le12$, $m\le3$: $>10^{-12}$ on any of $512$ draws | $0$ GPU-s, one evening | yes | the one-clamp filter, killed by R-20 ($>0.5$ disagreement), $66{,}430\times$ cheaper in solves per draw |
| 01.17 | any `[M]` target not building (K-K) | $0$ GPU-s, CPU minutes | yes | `[S]`, then RUN instances |
| 01.6 | fibre separation $<5\%$ at $\varepsilon=0.05$ | $0$ GPU-s, one evening | **UNREACHABLE** — collision instrument unspecified in book 04 | Bet D (01.8) |
| 01.5 | **none**: the arXiv re-read is an author action outside the canon | `NOT MEASURED` | **UNREACHABLE**; verse OPEN | Sanford $\to$ Chen as lineage, no kill; Terminal in force |
| 01.18 | numerical rank of $H_{33}$ at $s=66$ reading $\le32$ on any of $1{,}024$ draws | $0$ GPU-s, one evening | **UNREACHABLE** — Hankel-rank probe unspecified; and undecidable at $s=64$ | numerical-rank statement; re-price against the scan (book 03) |
| 01.21 | $\sigma_{17}/\sigma_1<10^{-3}$ on $6$ of $8$ cells, with the rank-$16$-plus-$10^{-2}$ plant as the must-fire | $0$ GPU-s from saved cells | **UNREACHABLE** — SVD probe unspecified; no cell (B5) | the committor head; then K-E1 (itself UNREACHABLE, 04.11) |
| 01.10(b) | Dirichlet Cheeger instance failing on BED-1 | `NOT MEASURED`, $0$ GPU-s | **UNREACHABLE** — Dirichlet instrument unspecified; verse OPEN | the ladder's measured rate ($f\in[0.99,1.01]$ at $t\ge320$) |
| 01.9 | $r_{\rm softmax}/r_{\rm shape}\le2$ (K-H1), **registered only on seeds where K-D2 does not fire** | inside the $\approx34$ s pair (Bet E, 04.6) | needs a cell (B5) | the residual as diagnostic; cost sentence (book 03) |
| 01.2 | $\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on $6$ of $8$ (K-D2) | inside the pair | needs a cell (B5); S-13 is a card of the record | the feature-content reading on S-11's generator spec, killed by the multi-hot absent while K-D2 fires — $0$ GPU-s, one document read, reachable today |
| 01.15 | cached mixture within one seed sd on $P$-plants, $6$ of $8$ (K-G1) | $\approx35$ s | needs a cell (B5) | the identity as algebra; Bet G's band |
| 01.14 | K-E1 at $N=8$, escalated to $N=16$ | $\approx52$ s $+\approx44$ s `[ASSUMED]` (**not** the $\approx34$ s of Bet E's pair in 04.6) | **UNREACHABLE** — 04.11 registers $0$ ChaCAL source lines and $0$ records | link 1($\alpha$): the margin arithmetic, $0$ GPU-s, fires on the pre-repair $L=20.7$; link 1($\beta$): $\varepsilon\|V\|_\infty$ against the label sd (B5); link 2: $\phi_0$ on the **environment chain** with a value-zero sink column, no arm (04.11 link 1's census column); then Tree B |
| 01.8 | Bet D as registered: $\mathrm{acc}_{\rm shape}-\mathrm{acc}_{\rm sky}$ at $N=8$ against the realised paired sd, SPLIT band none, **no seed-count clause** | $\approx64.8$ s `[ASSUMED]` | needs a cell (B5) | deeper skylines, then CoT |
| 01.16 (K-I) | one exceedance of $\delta\|V\|_\infty$ on $1{,}024$ draws | $\approx2.6$ s *when a mask exists* | **dormant**, owned by book 03; needs a mask (B15) and a label sd (B5) | not a link of 01.16's chain |

**Limits.** Every `RUN` number in this book is quoted from `docs/CEQ_SHAPE.md` at the
pin or from the round reports at `99777ab`; none was re-executed this session, and every
`RUN[J]`/`RUN[M]` figure **carries its own draw**. The judge's draw is $s=32$,
$\gamma=0.6$, seed 0, and the exceptions this book itself cites are printed here rather than
covered by a blanket sentence: the **mask-amplification instance** of 01.14 at $s=3$,
$\gamma=0.9$, $\varepsilon=0.1$ (`RUN[J]` $0.5263157894736843$ against the naive $0.1$,
`READ docs/CEQ_SHAPE.md:405-406 @ 99777ab`); the **BED-1 chain** of 01.10(b) at $|T|=9$
(`RUN` $\rho(Q_{\rm BED\text{-}1})=0.9408612510154677$), which is not $s=32$ at all; and the
**Gaussian-$V$ plant** of 01.1's Kill at $s=64$ (`RUN` $\max|\Delta z|=1.127$, beside
$0.1096$ at $s=32$). The former blanket claim was contradicted by all three on this book's own
pages. Verses 01.5,
01.8 and 01.18 rest on `[V]` citations whose `[V-eq]` is owed and are stated as
conditionals until the source-level re-read of book 04 runs. **One** citation is owed and
not in `references.bib`: the Dirichlet-form Cheeger inequality (01.10), load-bearing
nowhere. The second — the $n$-width of a linear image (01.21) — was `[ASSUMED textbook]`
against a Limits sentence claiming it load-bearing nowhere while the verse's Statement rested
on it; at repair the equality is **DERIVED** in two lines from the singular value
decomposition and Schmidt–Eckart–Young, Pinkus is recorded as the standard reference with its
citation still owed, and the two sentences no longer contradict each other. **One further
constant carries no value anywhere in the record or the canon**: $\gamma_{\rm env}$
(`READ docs/CEQ_SHAPE.md:341-342 @ 99777ab` names it and prints no number), over which verse
01.7's shortfall ranges by a factor $13.5$; every clause of that verse is registered at the two
run corners $\gamma\in\{0.6,0.9\}$ instead. **K-E1 (01.14) is UNREACHABLE**, not merely
uncelled: `04_BEDS_AND_INSTRUMENTS.md` verse 04.11 records $0$ ChaCAL source lines and $0$
records under any ChaCAL `kind`, so no ChaCAL arm exists to compare; 01.14 is carried by link
1($\alpha$), which is decidable today at $0$ GPU-s, and by link 2, moved at repair off the
non-existent arm onto the bed's own environment chain. **No A2 number is licensed by this
book**: A2 is read against the matched-parameter depth-$5$ skyline of book 04's clause 3b,
which has no width, no count and no cell, never against the depth-$5$ full-width control and
never against the linear-width skyline, which has no matched instance at $d_{\rm model}=16$.
**No value of the argmin zero-information floor is printed for BED-S**: the floor is
$1-\max_a\hat\pi(a^\star)$ at the realised class distribution, $\hat\pi$ is `NOT MEASURED`
(B5), and $0.875$ appears only as the uniform-$\hat\pi$ instance at $m=8$ — the unconditional
form this book printed before repair contradicted the record and books 04, 06 and 09 at once.
The Lean grades are statements
about statements; nothing was compiled. **One Lean grade was factually wrong at the book's own
pin and is corrected**: `hitting_time_transform` read `[D]` on the evidence of a repository
grep, while Mathlib at `a45ae637` carries `MeasureTheory.hitting`
(`Mathlib/Probability/Process/HittingTime.lean:51`, `RUN` this session); the row is `[S]` and
the gap is the missing bridge from a pathwise stopping time to the finite-state absorption
identity, not a missing object. No BED-S cell exists, so every $\mathrm{MDE}_8$ in
a kill is the realised-sd clause of `CHARTER.md` §2 and is a placeholder until S-62 fills
it. Six instruments this book's kills depend on are **unspecified in the canon** — the
margin probe (01.4), the collision instrument (01.6), the Dirichlet-conductance instrument
(01.10), the Hankel-rank probe (01.18), the Schur instrument (01.20), the SVD of
$\hat\Pi_\gamma$ (01.21) — so **six** kills read UNREACHABLE in the table above for want of an
instrument or of a permitted act (01.6, 01.10(b), 01.18, 01.20, 01.21, and 01.5's author
fetch), and a **seventh**, K-E1 (01.14), reads UNREACHABLE because the arm it compares does not
exist in the tree at all. Two verses
stand **OPEN** with their Terminals in force: 01.5, whose only kill is an author fetch, and
01.10, whose branch (a) was satisfied by construction and whose branch (b) has no
instrument. Verse 01.18's statement is additionally **undecidable at the planned geometry**:
$\mathrm{rank}\,H_i\le\lfloor s/2\rfloor=2d$ at $s=64$, $d=16$, so it enters the canon only
at $s\ge4d+2=66$. No code file, no git write, no external fetch.

---

## Attacks answered

MARS's findings, repair round 1, batch 1 of 3. Every finding appears **verbatim** — verse,
flaw, mechanism, number, severity — beside the repair applied or the sentence now in force.
Nothing here reads more gently than when it was filed (`CHARTER.md` §7); where the number a
finding cites has moved (book 04's line pins move under its own repairs), the moved pin is
printed and the finding's original is kept.

### 01.7 — `strike` · V-17, P-1 · repaired

> **flaw.** The shortfall table's gamma=0.7 row prints gamma^{K+1} where the Statement's
> equality demands gamma^{K+1}/(1-gamma); three of twelve cells are wrong by a factor 3.333,
> and the record prints the correct value for the identical (gamma,K).
>
> **number.** Printed 0.343 / 0.168 / 0.0824 at K=2/4/6, gamma=0.7. Correct: 0.7^3/0.3 =
> 1.14333, 0.7^5/0.3 = 0.560233, 0.7^7/0.3 = 0.274514. The K=16 cell (7.75e-3 = 0.7^17/0.3)
> is divided and the other three are not. docs/CEQ_SHAPE.md:3113 (B-I row) prints the
> gamma=0.7 plant as '119.37 / 1.143' -- bound 1.143 = 0.7^3/0.3 at K=2 -- against the
> book's 0.343. The gamma=0.6 and gamma=0.9 rows are correct throughout.
>
> **replacement_survives.** true.
>
> **required_repair.** Reprint the gamma=0.7 row as 1.143 / 0.560 / 0.2745 / 7.75e-3, or
> relabel the whole table as the Pi_gamma residual gamma^{K+1} and reprint the gamma=0.6 and
> gamma=0.9 rows accordingly. One convention per table.

**Repair.** 01.7's table: the $\gamma=0.7$ row is reprinted $1.143$ / $0.560$ / $0.2745$ /
$7.75\times10^{-3}$, the first branch of the required repair, so one convention —
$\gamma^{K+1}/(1-\gamma)$, the quantity the Statement's equality names — governs all twelve
cells. The paragraph under the table names the defect, the factor
$1/(1-\gamma)=3.3\overline{3}$, the three cells that carried $\gamma^{K+1}$ and the fourth
that did not, and pins the record's own $1.143$ at `READ docs/CEQ_SHAPE.md:3116 @ 99777ab`.
One consequence the wrong cell hid is now printed: at $\gamma=0.7$, $K=2$ the shortfall
**exceeds** $\|V\|_\infty$, so the two-hop iteration skyline is not a fellow approximator at
that horizon. Verse 01.16's new kill cites the same $(\gamma,K)=(0.7,2)$ cell as its second
plant, so a recomputation returning $0.343$ now fires a kill.

### 01.1 — `strike` · V-3, V-25 · repaired (carried from an earlier batch, verified at this repair)

> **flaw.** The Statement's 'iff' is false in the only-if direction at gamma=0, a value its
> own Hypotheses admit (gamma in [0,1)), and false for the committor label at every gamma.
>
> **number.** At gamma=0, z* = (I-0)^{-1}V = V for every P-hat, so the label matches with
> ||P-hat - P_env||_inf = 1 on the reachable set. For q: take A_sink={0}, A_0={1}, T={2},
> row 2 = (alpha, 1-alpha-beta, beta); q^(0)_2 = (1-alpha-beta)/(1-beta) reads 0.5 at both
> (alpha,beta)=(0.5,0) and (0.25,0.5), two chains differing by 0.25 in sup norm with
> identical labels.
>
> **replacement_survives.** false.
>
> **required_repair.** Weaken to the implication that holds: P-hat = P_env on the reachable
> set implies the four labels are exact. State the converse only for z as a map on all V at
> gamma>0, where resolvent equality does force matrix equality, and delete the 'iff' from the
> committor and argmax clauses.

**Repair.** 01.1 carries the required repair in full and it is verified at this repair
rather than re-derived: the Statement's "One direction only" paragraph asserts the forward
implication alone, states the converse for $z$ on **every** $V$ at $\gamma>0$ by inverting
both resolvents, prints the $\gamma=0$ failure with $\|\hat P-P_{\rm env}\|_\infty=1$, and
prints the committor counterexample with the two chains $0.25$ apart reading $0.5$ at both
$(\alpha,\beta)=(0.5,0)$ and $(0.25,0.5)$; the argmax clause is stated as inheriting that
many-to-one map, and no "iff" is written for $q$ or $a^\star$. On `replacement_survives:
false`: 01.1's chain is not the "iff" — its Kill is the four-identity battery with
$\hat P:=P_{\rm env}$ substituted, its replacement is the reach-avoid read with the
undeclared absorbing position moved into $\mathcal A_{\rm sink}$ (killed by a declared-sink
draw reading $\rho(Q)=1$, a census line), and its second link is the per-segment statement
(killed by the same census re-read per segment). Three links, three different numbers, and
none of them the weakened implication.

### 01.14 — `strike` · V-10, V-24 · repaired

> **flaw.** The Hypotheses' clause that the committor head at gamma=1 inherits the 2s e^{-L}
> bound through 01.1's kappa is either vacuous or false by O(1); it is the clause that
> carries the verse's conclusion that boundary rows are not a representability delta.
>
> **number.** With A declared, the boundary rows lie outside T, so Q-hat = Q and R-hat = R
> exactly and ||q-hat^(L) - q||_inf = 0 identically: the bound kappa*2s*e^{-L} is satisfied
> by construction and decides nothing. With A not declared -- the reading the
> class-containment claim needs, since ChaCAL-diag declares no boundary -- regime S fixes
> P_00 = 1 and P_ii < 1 for i >= 1 (READ docs/CEQ_SHAPE.md:286-287 @ 99777ab), so position 0
> is the only absorbing state, absorption into any A_k not containing 0 has probability
> exactly 0, and the error is ||q||_inf >= 0.548718 (the record's own instance, RUN[P], READ
> docs/CEQ_SHAPE.md:1531 @ 99777ab), not <= kappa*2s*e^{-L}.
>
> **replacement_survives.** false.
>
> **required_repair.** Restrict the epsilon-class containment to the discounted read at gamma
> bounded away from 1 and print the bound's blow-up 2s e^{-L}/(1-gamma). State that the
> committor head at gamma=1 is outside the finite-logit class entirely, and re-derive whether
> component (e) is retired on that head separately from the z channel.

**Repair.** The clause is **deleted**. 01.14's Statement restricts the containment to the
discounted channels at $\gamma$ bounded away from $1$ and prints the blow-up
$2s\,e^{-L}/(1-\gamma)$ with four values at fixed $L=30$, $s=64$: $1.20\times10^{-11}$ at
$\gamma=0.9$, $1.20\times10^{-10}$ at $\gamma=0.99$, $1.20\times10^{-8}$ at $\gamma=0.9999$,
diverging at $\gamma\uparrow1$. The Hypotheses print the admissible range,
$\gamma\le1-2s\,e^{-L}/\varepsilon=1-1.2\times10^{-5}$ at $\varepsilon=10^{-6}$. A second
paragraph states that the committor head at $\gamma=1$ is outside the finite-logit class
entirely and carries **both** readings the finding names: with $\mathcal A$ declared the
error is $0$ identically and the bound decides nothing (V-10); without, position $0$ is the
only absorbing state under regime S, $(I-P)^{-1}$ does not exist at $\gamma=1$ on the full
chain, and the error is $\|q\|_\infty\ge0.548718$ against a bound of $\le9.5\times10^{-11}$
at $L=30$, $\kappa\le7.9$ — wrong by ten orders, printed as such. Component (e) is now
decided **separately** per channel. **On `replacement_survives: false`: a second link is
derived, not renamed.** Link 1 (the $z$ channel) keeps the exactness clause with the
$1/(1-\gamma)$ factor carried into the printed $\varepsilon$. Link 2 is a different object
measured by a different instrument at a different number: the non-sink zero-fraction
$\phi_0$ — solve the $K+1$ absorption channels on the sink arm's own operator with **no
partition declared** and read whether $q^{(k)}_i=0$ exactly for every non-sink $k$ and every
$i$ over $512$ draws — killed by $\phi_0<1$ on any draw, planted against a literal identity
row at a non-sink position. K-E1 compares two NRMSEs over $8$ seeds against
$\mathrm{MDE}_8$; link 2 reads an exact-zero predicate over $512$ draws. The Terminal now
licenses the containment on the discounted channels only and, on the committor head,
licenses neither "a mechanism softmax cannot express" nor its negation until $\phi_0$ is
read.

### 01.18 — `strike` · V-10, V-11 · repaired

> **flaw.** The Kill is satisfied by construction at the planned geometry, and the
> Hypotheses' own non-vacuity predicate is unsatisfiable there.
>
> **number.** The probe is specified as 'SVD of H_i at i=s/2 over 1,024 draws' at s=64, so
> H_i is the 32x32 lower-left block of a lower-triangular matrix and rank H_i <= 32 = 2d at
> every draw. A time-varying linear realisation with e = 32 therefore exists for every
> (P-hat,V), so 'an e <= 2d = 32 realisation to 1e-12' fires on 1,024 of 1,024 draws. The
> Hypotheses' clause 'non-vacuous only where that rank exceeds 2d = 32' can never hold at
> s=64.
>
> **replacement_survives.** true.
>
> **required_repair.** Either move the probe to i where min(i, s-i) > 2d -- impossible at
> s=64, d=16, since max_i min(i,s-i) = 32 -- or raise s to 4d+1 = 65 minimum and state the
> geometry at which the statement first bites, or drop the Kill and register only the
> numerical-rank statement at tolerance tau, which is what the If killed already contains.

**Repair.** The Hypotheses now print $\mathrm{rank}\,H_i\le\min(i,s-i)\le\lfloor s/2\rfloor$
and derive the non-vacuity threshold $\lfloor s/2\rfloor\ge2d+1=33$. **The finding's
"4d+1 = 65 minimum" is carried and then tightened, not softened:** at $s=65$,
$\lfloor65/2\rfloor=32=2d$ and the kill still fires by construction, so the first geometry at
which the statement bites is $s=4d+2=66$, $i=33$. The probe is re-registered there — SVD of
the $33\times33$ block $H_{33}$, numerical rank at $\tau=10^{-6}$, $1{,}024$ draws — and the
kill is reprinted as "the numerical rank of $H_{33}$ at $s=66$ reading $\le2d=32$ on any of
$1{,}024$ draws", the third branch of the required repair combined with the second. Two
controls ship: the linear corner ($\beta=0$, $g\equiv0$) must read numerical rank $\le d=16$,
and running the probe at $s=64$, $i=32$ must read rank $\le32$ on every draw, so the repaired
defect itself becomes a plant. The Terminal states that at $s=64$, $d=16$ the canon licenses
**neither** the statement nor its negation, the question being closed by arithmetic before
any draw is taken.

### 01.11 — `strike` · P-10, V-25 · repaired

> **flaw.** Clause (ii) dismisses B10's hazard by classifying the arm's chain as unichain;
> with K+1 boundary sets plus a sink every absorbing state is its own closed recurrent class,
> so the chain is multichain by definition -- exactly the class B10 names.
>
> **number.** At the design point (K=2, |A_bullet| in {1,2}, docs/CEQ_SHAPE.md:295-297 @
> 99777ab) the chain has |A| >= 4 absorbing states, hence >= 4 recurrent classes. A chain is
> unichain iff it has exactly one recurrent class. The census row B10 (CHARTER.md:236) names
> 'multichain constrained MDPs with several unsafe sets' as the hazard, and 01.11 closes it
> by asserting the hazard's own hypothesis is absent.
>
> **replacement_survives.** false.
>
> **required_repair.** Delete 'the class is unichain-absorbing'. Justify backward induction
> on the correct grounds -- the objective is a total-reward absorption probability on a
> transient DAG block with rho(Q) < 1, not an average-reward criterion, and Misra's hazard is
> a hazard of the average-reward LP -- and print the hypothesis under which that argument
> holds.

**Repair.** The clause "the class is unichain-absorbing and Misra's multichain hazard is
absent for this objective" is **deleted**. 01.11's Statement now opens clause (ii) by
conceding the count — $|\mathcal A|\ge4$ absorbing states, hence $\ge4$ recurrent classes at
the design point, and a chain is unichain iff it has exactly one — and then justifies
backward induction on the criterion: a total-reward absorption probability on a transient
DAG block with $\rho(Q)<1$, where the recursion is a finite backward substitution in
topological order with no fixed-point argument, against Misra's hazard, which is a hazard of
the **average-reward** constrained LP. Three hypotheses are printed: (H1) every clamp keeps
$P^{(a)}$ causal so the substrate stays a DAG; (H2) $0\in\mathcal A$ and every policy absorbs
in at most $s$ steps; (H3) the objective is additive in the absorption indicator — and (iii)
is now read as the place H3 fails, which is where the principle of optimality genuinely
breaks. The census table's B10 row records that the row is closed by the criterion
distinction and **not** by denying the hazard's hypothesis. **On `replacement_survives:
false`:** the old replacement's kill was "the three-step optimum disagreeing with the
two-step on more than $0.5$ of draws — the same instrument one level deeper, equally
priced", a V-9 renaming, and it is named as such. It is replaced by an exactness identity
against **exhaustive policy enumeration**:
$\max_i|V^\star_{\rm DP}(i)-V^\star_{\rm enum}(i)|>10^{-12}$ on any of $512$ draws restricted
to $|T|\le12$, $m\le3$, enumerating all $\le3^{12}=531{,}441$ deterministic stationary
policies. Different object (the DP value against the true optimum, not a disagreement
fraction between two heuristics), different instrument, different number, strictly more
decisive. Two plants: an inverted recursion ($\min$ for $\max$) must be exceeded by
enumeration, and a Chebyshev DP must **disagree** with enumeration by $\tfrac12$ on the
two-clamp witness, which is H3's own failure used as a control.

### book-level (01.4, 01.6, 01.10b, 01.18, 01.20, 01.21) — `strike` · P-4 · repaired

> **flaw.** Six verses ship 'NOT MEASURED -- needs <instrument> (book 04)' for instruments
> that book 04 does not specify, violating the CHARTER §3 requirement that the instrument be
> specified in the instruments book.
>
> **number.** 04_BEDS_AND_INSTRUMENTS.md contains verses 04.1-04.21 and card ids S-10, S-12,
> S-13, S-24, S-28, S-35, S-61, S-62, S-73. Missing entirely: the collision instrument
> (01.6), the margin probe (01.4), the Dirichlet-conductance instrument (01.10b), the
> Hankel-rank probe (01.18), the Schur instrument (01.20), the SVD of Pi-hat_gamma on trained
> cells (01.21) -- grep counts 0 for 'collision instrument', 'Hankel-rank', 'margin probe',
> 'Schur instrument', 'Dirichlet' in that file. Book 01 also routes kills to cards S-02,
> S-17, S-31, S-33, S-64, X-2, X-4, N-15, R-20, J-L18, none of which appear in book 04 (they
> appear only in docs/CEQ_SHAPE.md).
>
> **replacement_survives.** false.
>
> **required_repair.** Either add the six instruments as verses in book 04 with shapes,
> draws, seed rule and price, or restate each NOT MEASURED clause as 'needs <instrument>,
> unspecified in the canon' and mark the six kills unreachable until book 04 carries them.
> Replace the card-id routing with book 04 verse ids or with a READ pin into
> docs/CEQ_SHAPE.md at 99777ab.

**Repair.** The second branch, since a writer of book 01 may not edit book 04 (the task
names one file). A new **Instrument routing** paragraph in the Preface states the rule, the
counts, and the six instruments by name and owning verse. Each of the six `NOT MEASURED`
clauses now reads "unspecified in the canon" with the grep count that convicts it and the
probe's shape written out as a specification this book supplies and book 04 does not carry:
the margin probe (01.4), the collision instrument (01.6), the Dirichlet-conductance
instrument (01.10's branch (b) — book 04's one "Dirichlet" hit is 04.15's manifest field
`committor_route ∈ {dirichlet_gamma1, discounted_gamma}`, a route and not an instrument), the
Hankel-rank probe (01.18 — book 04's two "Hankel" hits are BED-K's ceiling $\sqrt{1-1/d}$ in
04.3, a different object), the Schur instrument (01.20), the SVD of $\hat\Pi_\gamma$ (01.21).
The kills table now carries a **reachable today** column and marks five UNREACHABLE; the
Limits paragraph repeats the six by name and records the two verses that stand OPEN in
consequence. Card-id routing: S-02, X-2, X-4, R-20, S-17 and J-L18 each now carry a
`READ docs/CEQ_SHAPE.md:<line> @ 99777ab` pin and say the card is the record's and not book
04's; S-10 is named as the one card of the list book 04 does carry. The finding
additionally named S-31, S-33, S-64 and N-15: a grep of this book at repair time finds none
of the four, which is recorded in the Preface rather than quietly dropped, together with six
further card ids the finding did not name — J-D3, J-D6, J-D8, J-L0, J-L6, R-07 — which carry
the same defect in Evidence position.

### 01.6 — `strike` · D-7, V-17 · repaired (carried from an earlier batch, verified at this repair)

> **flaw.** The If killed reverses the direction of Bet D as registered, and withdraws the A2
> sentence on the outcome that would license it; the same bet is read the other way in 01.8,
> so two verses of one book contradict each other on one registered bet.
>
> **number.** docs/CEQ_SHAPE.md:2237 @ 99777ab registers Bet D's prediction as 'the depth-5
> stack within MDE_8 of the shape's argmin accuracy -- the honest control holds' and its
> counter as 'short by more than MDE_8, sign logged'. 01.6 writes 'If Bet D reads the counter,
> the D-APPROX sentence for BED-S is the depth skyline fits ... the A2 sentence withdrawn'.
> The counter is the skyline NOT fitting. 01.8's Kill reads the same counter as 'the analogy
> is wrong in the direction that flatters the shape', i.e. the skyline is short. Under
> D-CALIB the counter is the point estimate, so the book's stated default outcome and its
> stated consequence are inverted.
>
> **replacement_survives.** false.
>
> **required_repair.** Swap the branches: if Bet D reads its prediction (skyline within
> MDE_8) the A2 sentence is withdrawn and the D-APPROX sentence for BED-S is 'the depth
> skyline fits'. If it reads its counter, the A2 form has its first evidence and the counting
> bound is superseded, not replaced.

**Repair.** 01.6's If-killed carries the required repair in full and it is verified at this
repair: the branches are printed "in the direction the registration gives them", with the
registration quoted at `READ docs/CEQ_SHAPE.md:2236 @ 99777ab` — prediction is the skyline
within $\mathrm{MDE}_8$, counter is short by more than $\mathrm{MDE}_8$. On the prediction
the depth skyline **fits**, the A2 sentence is **withdrawn** and the A1 sentence reduces to
01.7; on the counter the A2 form has its first evidence and the counting bound is
**superseded, not replaced**. The text names the inversion it repairs and the contradiction
with 01.8's Kill, which reads the same counter the same way. Under D-CALIB the counter is
the point estimate, so the second branch is the one the canon plans against, and that is
printed.

### 01.13 — `strike` · V-11, P-4 · repaired

> **flaw.** The only Kill is an act book 04 registers as outside the canon's rules, and it is
> priced as if inside.
>
> **number.** 01.13 Kill: 'The source-level re-read of Eq. 5 and Eq. 7 at the arXiv LaTeX
> source (2410.05565) ... price 0 GPU-s, one evening'. 04_BEDS_AND_INSTRUMENTS.md:175 closes
> 04.12's Evidence with 'No LaTeX source has been read: NOT MEASURED -- needs one fetch of the
> arXiv source, an author action outside this canon's no-fetch rule'. A kill that cannot fire
> on the bed as registered is not a kill (CHARTER.md:145-148).
>
> **replacement_survives.** false.
>
> **required_repair.** Reprice the kill as NOT MEASURED with the author action named, or
> replace it with the operator-level kill book 04.12 actually ships -- the two arms must
> differ by O(1) at gamma=0.9 on the same P, decided by arithmetic and not by the citation --
> and state the convention as undecided until the fetch happens.

**Repair.** Both branches, the two halves being compatible. The re-read is repriced
`NOT MEASURED — needs one author fetch of arXiv:2410.05565, an act no writer, refuter or
repairer of this canon may perform`, registered as an author action with its own row in book
05, and the convention — which arithmetic the published paper ran — is stated **undecided
until that fetch happens**, with no sentence of this book asserting it. The verse's kill
becomes the operator-level one book 04.12 ships: on the same drawn $A$ at $\gamma=0.9$,
$\beta=1$, the two arms failing to differ by $O(1)$ —
$\|O-Y\|_\infty/\|V\|_\infty<10^{-3}$ on any draw, or ChaCAL-published's normalised row sums
reading $1.000$ within $10^{-12}$ — decided by arithmetic, $0$ GPU-s, $512$ draws, with the
record's own instance ($0.400\dots0.765$ against $1.000$, `RUN[J]`) on the firing side and a
$\gamma=0$ bitwise-agreement control. The book-04 pin the finding gives as `:175` reads
`:193` at repair time and is now cited by **verse id** (04.12) so it survives book 04's own
repairs; verse 01.5's copy of the same pin is corrected the same way. **On
`replacement_survives: false`:** the old replacement routed through the InfSA-style base,
which is `NOT MEASURED — needs the base wired` and therefore cannot fire; it is demoted to
the plant it is at `READ docs/CEQ_SHAPE.md:3110 @ 99777ab` and is **not** a link of the
chain. The new replacement is the **row-sum census on the two operators** — a statement about
the operators' rows, not their reads, which survives whatever the reads do — with its own
Hypotheses ($\beta=1$, finite logits, $\mathrm{diag}(A)>0$), its own Evidence (`RUN[J]` row
sums), its own Kill (any cell reading $1.000$ within $10^{-12}$, one pass over the rows,
strictly cheaper than forming either resolvent) and the diagonal-kept arm as its must-fire
plant.

### 01.5 — `strike` · V-11, P-3 · repaired as OPEN, Terminal in force (carried from an earlier batch, verified at this repair)

> **flaw.** Same defect: the Kill is the arXiv-source re-read book 04 registers as outside
> the canon's no-fetch rule, and both pre-written replacements are killed by 'the same
> re-read', so the entire chain is unreachable.
>
> **number.** 01.5 Kill priced '0 GPU-s, one evening'; Replacement 1's Kill: 'the same
> re-read, on a second source'; Replacement 2's Kill: 'trivial (any finite s)'.
> 04_BEDS_AND_INSTRUMENTS.md:175: 'an author action outside this canon's no-fetch rule'. Three
> links, none of which can fire inside the canon.
>
> **replacement_survives.** false.
>
> **required_repair.** Mark the chain unreachable and let the Terminal stand as the licensed
> sentence today: at s=64, d=16, p=32 no unconditional 'softmax cannot' is licensed for the
> 2-hop committor. Register the re-read as a NOT MEASURED author action with its own row in
> book 05, not as this verse's kill.

**Repair.** 01.5 carries the required repair and it is verified at this repair: the verse is
headed **OPEN**, its Kill is headed "UNREACHABLE, and the verse stands OPEN on that account"
and repriced `NOT MEASURED — needs one author fetch of arXiv:2402.08164, an act no writer,
refuter or repairer of this canon may perform`, registered as an author action with its own
row in book 05. The two unconditional theorems are collapsed into **one lineage link with no
kill of its own**, because neither was strictly cheaper nor strictly more decisive than the
re-read and both died to the same act — so the chain terminates in one link rather than
three, and the Terminal is headed "in force now": at $s=64$, $d=16$, $p=32$ no unconditional
"softmax cannot" is licensed for the 2-hop committor, and no verse of this canon can change
that by any act it is permitted to perform. Sanford's constant $c$ is `NOT MEASURED` and the
earlier "first violated at $n>512$", which silently set $c=1$, is struck. The book-04 pin is
corrected to verse-id routing at this repair; the kills table's 01.5 row now reads "**none**:
the arXiv re-read is an author action outside the canon" with UNREACHABLE in the
reachability column.

### 01.16 — `strike` · V-10, V-11 · repaired

> **flaw.** K-I cannot fire on the bed as registered: both of its branches need an object the
> canon records as absent, and the book's own Kills table labels it '(dormant)'.
>
> **number.** Branch 1 needs a shipped mask; CHARTER.md:241 (B15) records 'the kernels do not
> exist: chunked solve, CSR path, Mapper schedule are NOT MEASURED'. Branch 2 needs the label
> sd; CHARTER.md:231 (B5) records 'BED-S has no cell, no realised sd'.
> docs/CEQ_SHAPE.md:2280 @ 99777ab prices R-10/K-I at ~2.6 s and Bet J at ':2237' is
> registered '(dormant)', '~2.6 s when a mask exists'.
>
> **replacement_survives.** true.
>
> **required_repair.** Register the kill that is reachable today at 0 GPU-s: the
> attained-equality check err = gamma^{K+1}/(1-gamma) against the RUN pair
> 0.007754350466241788 / 0.007754350466240224, with the rows-1.5 plant at gamma=0.6 as the
> must-fire. Keep K-I as a dormant row of book 03 and say so in the Kill field.

**Repair.** Exactly as required. 01.16's Kill field now opens by naming K-I unreachable, with
both absent objects quoted at `CHARTER.md:241` (B15) and `CHARTER.md:231` (B5) and the
record's own dormancy quoted from bind B-I ("REPAIR, **dormant** … dormant until row M ships
a mask"); it states that K-I is **a dormant row of book 03**, named here and owned there. The
verse's kill is the attained-equality check, frozen at a relative departure above $10^{-12}$
from $\gamma^{K+1}/(1-\gamma)$ on the record's registered $(\gamma,K)$ pairs, priced $0$
GPU-s and half an evening on a float64 recomputation with no cell, no mask and no bed. The
banked pair is printed on the passing side with the relative figure derived:
$0.007754350466241788$ against $0.007754350466240224$ is an absolute departure of
$1.5647\times10^{-15}$ and a **relative** departure of $2.0179\times10^{-13}$, inside the
frozen threshold. The rows-$1.5$ plant at $\gamma=0.6$, $K=2$ is the must-fire ($7.29$
against $0.54$), and a second plant reads the $(0.7,2)$ cell: a recomputation returning
$0.343$ has dropped the $1/(1-\gamma)$ factor, which is 01.7's repaired defect turned into a
control. The If-killed is re-derived onto a different object — the dropped-mass bound read
off $P_m$'s own row sums, killed by any row summing above $1+10^{-12}$ — rather than left on
the resolvent. K-I keeps its own row in the kills table, marked dormant and owned by book 03.

### 01.19 — `strike` · V-10 · repaired

> **flaw.** The Kill cannot fire on the bed as registered: the arm's P-hat is a causally
> masked softmax whose strict upper block is exactly zero by construction, and the verse's own
> If killed concedes it ('the arm is mis-masked').
>
> **number.** Kill: 'Any trained cell whose P-hat has a non-zero entry above the diagonal by
> torch.equal on the strict upper block'. A masked softmax writes -inf before the exponential,
> so the upper block is bitwise 0.0 on every draw; the kill's firing set is empty. The If
> killed replacement ('the masked corner with the check re-run') is the identical check.
>
> **replacement_survives.** false.
>
> **required_repair.** Replace with a kill that can fire on the class as registered: the arm
> at beta not equal to 1, whose rows sum 1.31 to 10.29 (READ V16_ARM_SMPRIME.md:28-32 @
> 99777ab), makes gamma*P non-substochastic and the diagonal read of the spectrum no longer
> bounds anything -- register the rowsum census on every trained cell as the decidable
> condition.

**Repair.** Exactly as required. The strict-upper check is struck by name, with the mechanism
printed ($\exp(-\infty)=0.0$ exactly, so the firing set is empty, V-10) and the old If-killed
named as the V-9 renaming it was. The frozen kill is the **row-sum census on every trained
cell**: $|\sum_j\hat P_{ij}-1|>10^{-12}$ for some $i$, or $\gamma\max_T\hat P_{ii}\ge1$. It
has already fired once on the registered class — rows summing $1.31$ to $10.29$ at
$\hat\beta\ne1$, `READ V16_ARM_SMPRIME.md:28-32 @ 99777ab` — so $\gamma\hat P$ is not
sub-stochastic, the Neumann series does not converge from row-stochasticity, and the diagonal
read bounds nothing about $\|(\gamma\hat P)^t\|$. The verse records that the spectral
identity itself is algebra no draw refutes, which is why it is not the kill. Controls: the
$\beta=1$ masked arm must read row sums $1.000$ to $10^{-12}$ on every cell; the second
clause needs $\hat\gamma\ge1.4436$ at the judge's draw ($\max_T\hat P_{ii}=0.6927$, `RUN[J]`)
and is registered as reachable through the $\hat\gamma$ dial, book 02's B12. **On
`replacement_survives: false`:** the replacement is now the **admitted-cell fraction**
$n_{\rm adm}/N$ printed beside every Koopman sentence, killed by $n_{\rm adm}/N=0$ — a
different object (a count over cells, not one cell's rows), decided by reading an integer the
census has already produced, so strictly cheaper, and terminating, since at $n_{\rm adm}=0$
there is nothing left to restrict.

### 01.10 (branch a) — `strike` · V-10 · repaired; verse now OPEN, Terminal in force

> **flaw.** Branch (a)'s Kill cannot fire and the verse states so in its own If killed field.
>
> **number.** If killed: '(a) cannot fail on a triangular operator'. Kill (a): '|rho(Q-hat) -
> max_T P-hat_ii| > 1e-12'. Eigenvalues of a triangular matrix are its diagonal exactly; the
> residual is float rounding on a value read off the diagonal, so the threshold 1e-12 is a
> numerical-noise gate, not a refutation of the Statement. CHARTER.md:145-146: a kill that
> cannot fire is not a kill.
>
> **replacement_survives.** false.
>
> **required_repair.** Delete kill (a) or repoint it at the reachable condition it actually
> guards -- a trained cell whose P-hat is not lower-triangular, decided by the strict-upper
> torch.equal census that the If killed already names -- and file that census as book 04's
> admission line rather than as this verse's kill.

**Repair.** Kill (a) is **deleted**, the first branch of the required repair. The second
branch is deliberately not taken and the reason is printed in the verse: repointing at the
strict-upper `torch.equal` census would repoint at another gate satisfied by construction,
since a causally masked softmax's strict upper block is bitwise $0.0$ — the same defect
01.19's finding convicts in this same batch, so repairing one verse into the other's defect
would be a repair that changes nothing. The triangularity census is instead filed as an
**admission line** — a cell failing it is refused before any certificate is read off it —
owned by book 04, which does not carry it: `NOT MEASURED — needs the triangularity admission
line, unspecified in the canon`. Branch (a)'s content is stated as algebra with no kill
attached. Consequence, carried honestly rather than hidden: with (a) struck and (b)
unreachable (next finding), **01.10 carries no reachable kill today**; the verse is headed
**OPEN**, its Terminal is headed "in force now" and now states that neither branch has been
tested, and the census table's B9 row records the same.

### 01.10 (branch b) — `repair` · V-15 · repaired

> **flaw.** Branch (b)'s Kill ships with no planted negative, the only kill in the book with
> none; V-15 requires the control that would make the kill fire on a known-bad plant.
>
> **number.** Kill (b): 'On BED-1, h_D^2/2 computed from the bed's own conductance exceeding
> 1 - rho(Q) = 0.0591: ... NOT MEASURED -- needs the Dirichlet-conductance instrument on bed_1
> (book 04), 0 GPU-s.' No Control clause follows. Every other Kill in the book (01.1-01.9,
> 01.10a, 01.11-01.22) carries one.
>
> **replacement_survives.** true.
>
> **required_repair.** Add the must-fire plant: a non-reversible transient block whose
> Dirichlet conductance is computed by the same instrument must read a bound that exceeds 1 -
> rho(Q), so an instrument that never fires is visible before it certifies BED-1.

**Repair.** The plant is supplied and the omission is named in the verse as "the only kill in
the book with none (V-15)". The must-fire plant is a non-reversible transient block
$Q=\begin{pmatrix}0&0.9\\0&0.9\end{pmatrix}$: $\rho(Q)=0.9$, $1-\rho(Q)=0.1$, reversibility
fails because $Q_{12}=0.9$ and $Q_{21}=0$, and the same instrument must report $h_D^2/2$
**above** $0.1$ on it. The verse prints the consequence of skipping it: an instrument that
certifies BED-1 without first firing on this plant has never been shown to fire at all, and
the `[ASSUMED textbook]` bound would be certified by a probe with an empty rejection region.
The plant is priced with the instrument and is `NOT MEASURED` for the same reason — book 04
carries no Dirichlet-conductance instrument — so the branch is marked UNREACHABLE in the
kills table.

### 01.3 — `repair` · V-17 · repaired (carried from an earlier batch, verified at this repair)

> **flaw.** The printed gap lower bound belongs to a different object than the read the
> Statement defines; for the read O(gamma) the inequality is false.
>
> **number.** The Statement defines O(gamma)_i = E_i[gamma^{tau_k - 1} 1_k] and then asserts
> 0 <= q - E[gamma^{tau_k} 1_k] = E[(1-gamma^{tau_k}) 1_k] >= (1-gamma) q. The middle object
> is (1-gamma) z_i, not O_i. Counterexample for O: P_ii = 0.1, P_ia = 0.9 with a in A_k, gamma
> = 0.6 gives q_i = 1, O_i = 0.9/(1-0.06) = 0.95745, gap 0.04255, while (1-gamma) q_i = 0.4.
> The record's own line (docs/CEQ_SHAPE.md:384-386 @ 99777ab) states the inequality for
> E[gamma^tau], never for the read.
>
> **replacement_survives.** true.
>
> **required_repair.** State the gap chain for (1-gamma) z_i explicitly, or print the correct
> bound for O: q_i - O_i = E[(1 - gamma^{tau_k - 1}) 1_k] >= 0, with no non-trivial lower
> bound, and rewrite the Terminal so it binds the object it names.

**Repair.** Both branches are taken and verified at this repair. 01.3's Statement carries a
"Two gaps, two objects, never interchanged" paragraph: the $(1-\gamma)q_i$ chain is stated
explicitly for $z_i:=E_i[\gamma^{\tau_k}\mathbb 1_k]=\gamma O(\gamma)_i$, and for the read
$O(\gamma)$ the verse asserts only
$q^{(k)}_i-O(\gamma)_i=E_i[(1-\gamma^{\tau_k-1})\mathbb 1_k]\ge0$ **with no non-trivial lower
bound**, with the reason printed ($\tau_k=1$ has positive probability and the integrand is
$0$ on that event). The counterexample is printed with the finding's own numbers —
$P_{ii}=0.1$, $P_{ia}=0.9$, $\gamma=0.6$, $q_i=1$, $O_i=0.957447$, gap $0.042553$ against
$(1-\gamma)q_i=0.4$ — with the failure factor $9.4$. The Hypotheses state that the
$10^{-6}q_i$ reading at $\gamma=1-10^{-6}$ holds "on $E[\gamma^{\tau_k}]$ and on nothing
else", and the Terminal binds the bound to $E_i[\gamma^{\tau_k}\mathbb 1_k]$ and to no other
object, withdrawing every sentence that binds it to the read.

### 01.4 (construction) — `repair` · V-8, V-12 · repaired (carried from an earlier batch, verified at this repair)

> **flaw.** The construction gives position 0 two disjoint boundary labels at once, so the
> answer channel and the leak channel are the same set on every instance whose answer is token
> 0.
>
> **number.** The construction declares A_j = {(j,k)} for j in [n] with layer k at indices
> [0,n), so A_0 = {position 0}; the Hypotheses simultaneously declare 'sink 0 declared as
> A_sink absorbing the leak'. docs/CEQ_SHAPE.md:295-297 @ 99777ab requires A = disjoint union
> of A_sink, A_0, A_1, ..., A_K. At n=64 the collision hits 1/64 of instances by uniform pi^k,
> and on those the epsilon-clause q^(hop_k(i)) >= 1 - k s' e^{-L}/(1 - s' e^{-L}) cannot be
> separated from the leak the sink is declared to absorb.
>
> **replacement_survives.** false.
>
> **required_repair.** Add one dedicated sink position at index 0 and shift the layers, giving
> s' = (k+1)n + 1 and A_sink = {0}, A_j = {(j,k)} at indices [1, n+1). Reprint the loss bound
> at s' = 577 and the non-vacuity threshold L > log(2 k s').

**Repair.** Exactly as required, and verified at this repair. 01.4's Statement reads
$s'=(k+1)n+1$ throughout; position $0$ is the sink **and nothing else**
($\mathcal A_{\rm sink}=\{0\}$, $P_{00}=1$); layer $k$ occupies $[1,n+1)$ with
$\mathcal A_j=\{(j,k)\}$ at index $1+j$ and layer $0$ occupies $[kn+1,(k+1)n+1)$; the
disjointness is asserted against `Definition 3` at `READ docs/CEQ_SHAPE.md:295-297 @
99777ab`, and the earlier layout is named as the defect it was, including the $1/64$
collision rate under uniform $\pi^k$. The loss bound is reprinted at $s'=577$:
$577\cdot9.36\times10^{-14}=5.40\times10^{-11}$ per step, $\times8=4.32\times10^{-10}$, so
$\le4.4\times10^{-10}$; the non-vacuity threshold is reprinted as
$L>\log(2ks')=\log(2\cdot8\cdot577)=9.131$.

### 01.4 (Lean route) — `repair` · P-10, V-25 · repaired

> **flaw.** The Lean route cited for the reduction target has a hypothesis the construction
> violates: StrictlyLower is regime N, and the construction is regime S with self entries at
> -L, i.e. P_ii > 0.
>
> **number.** 01.17's table routes hop_layered_committor through
> Nilpotent.pow_card_eq_zero (lean/CEQ/Nilpotent.lean:77), whose statement reads 'theorem
> pow_card_eq_zero {A : Matrix (Fin n) (Fin n) R} (hA : StrictlyLower A) : A ^ n = 0'. 01.4's
> Hypotheses: 'regime S with self entries at -L'. docs/CEQ_SHAPE.md:288-289 @ 99777ab: 'regime
> N admits no absorbing row'. The row also files a repository theorem in a column headed
> 'Mathlib name ([V-name] at the pin)'.
>
> **replacement_survives.** true.
>
> **required_repair.** Either state the reduction on the strict pointer DAG at gamma=1 with no
> self entries, where pow_card_eq_zero applies and the walk is exactly k steps, and treat the
> -L self entry only in the softmax-class epsilon-clause; or drop the Nilpotent route and grade
> the target [S] with its own supporting lemma. Move the repository citation out of the
> Mathlib-name column.

**Repair.** The first branch plus the column fix. 01.4's Hypotheses separate the two readings
and that separation is verified at this repair: **(a)** the exact reading at $\gamma=1$ is on
the *strict* pointer DAG with no self entries, the walk exactly $k$ steps and $q\in\{0,1\}$,
regime N on the transient block with the absorbing rows added; **(b)** the
$\varepsilon$-clause is on the softmax class, regime S, self entry at $-L$; and the two are
never mixed in one sentence. 01.17's table row for the reduction now carries `—` in the
column headed "Mathlib name (`[V-name]` at the pin)", and a note below the table moves the
repository lemma out of that column: its statement is quoted in full, `StrictlyLower` is
identified as regime N which *"admits no absorbing row"*
(`READ docs/CEQ_SHAPE.md:288-289 @ 99777ab`), the route is declared admissible on **reading
(a) only**, and reading (b) is carried by the union bound of 01.4's Statement instead. The
grade stays `[S]` with its supporting lemma supplied in-file.

---

**Batch 1 disposition, counted exactly.** Sixteen findings filed — twelve `strike`, four
`repair` — and sixteen answered. **Eleven were repaired at this batch**: 01.7, 01.14, 01.18,
01.11, the book-level P-4 row, 01.13, 01.16, 01.19, 01.10's branch (a), 01.10's branch (b),
01.4's Lean route. **Five were verified as already carrying the required repair** from an
earlier batch and are recorded as verified rather than re-claimed: 01.1, 01.6, 01.5, 01.3,
01.4's construction. No finding is left unanswered and no verse was deleted to escape one.
Two verses now stand **OPEN** with their Terminals in force — 01.5 (its only kill an author
fetch) and 01.10 (branch (a) struck as satisfied by construction, branch (b)'s instrument
unspecified) — and five kills read **UNREACHABLE** in the kills table for want of an
instrument book 04 does not carry. Four replacements were re-derived onto a different object
and a different instrument because the old one died with its verse (V-9): 01.14's link 2 (the
non-sink zero-fraction $\phi_0$), 01.11's (DP against exhaustive enumeration), 01.13's (the
row-sum census on the two operators), 01.19's (the admitted-cell fraction $n_{\rm adm}/N$);
01.16's was re-derived onto the dropped-mass row census for the same reason.

---

MARS's findings, repair round 1, batch 2 of 3. Every finding appears **verbatim** — verse,
flaw, mechanism, number, severity — beside the repair applied or the sentence now in force.
Nothing here reads more gently than when it was filed (`CHARTER.md` §7). Where a finding's
line pin has moved because the book it cites was itself repaired between filing and answer,
the moved pin is printed **and the finding's original is kept unaltered**.

### 01.7 — `repair` · M-2, V-3 · repaired

> **flaw.** The Kill cannot refute the Statement, and the threshold it does decide is not
> frozen because gamma_env carries no value anywhere in the canon.
>
> **mechanism.** M-2, V-3.
>
> **number.** The Statement is an attained matrix identity; a learned depth-5 stack reading a
> z-channel NRMSE below the exact solve's cannot make ||(I-gamma P)^{-1} - sum_{t<=K}(gamma
> P)^t||_inf differ from gamma^{K+1}/(1-gamma). The Kill is registered 'On BED-S at
> gamma_env', which docs/CEQ_SHAPE.md:341-342 @ 99777ab names as 'a bed constant gamma_env'
> with no number printed anywhere in the record or the canon. The shortfall the verse decides
> varies from 0.540 (gamma=0.6, K=2) to 7.29 (gamma=0.9, K=2), a factor 13.5, over the
> unspecified constant. The verse's own Hypotheses concede that at gamma=0.6, K=16 the
> shortfall 4.23e-4 is 'below any label sd the bed will show'.
>
> **replacement_survives.** true.
>
> **required_repair.** Freeze gamma_env with a number and an evidence class, or register the
> kill at the two corners the record has run (gamma=0.6 and gamma=0.9) with both shortfalls
> printed. Add the arithmetic kill the Statement actually admits -- any table cell differing
> from gamma^{K+1}/(1-gamma) by more than 1e-12 -- which fires today.
>
> **severity.** repair.

**Repair.** The second branch of the required repair, both halves. **(1)** $\gamma_{\rm env}$
is **not** frozen with a number, because none exists to freeze: a new paragraph in 01.7's
Hypotheses prints `NOT MEASURED — needs the BED-S registration triple to fix
$\gamma_{\rm env}$ before the first draw (book 04, the D-4 admission block), a $0$ GPU-s
registration act`, quotes `docs/CEQ_SHAPE.md:341-342 @ 99777ab` naming the constant without a
value, prints the factor $13.5$ span ($0.540$ at $(0.6,2)$ against $7.29$ at $(0.9,2)$), and
registers **every clause of the verse at the two corners the record has run, $\gamma=0.6$ and
$\gamma=0.9$, and at no other $\gamma$**. **(2)** The learned-stack kill is **struck**, with
both grounds printed in the verse: it cannot refute an attained matrix identity computed from
$(\gamma,K',P)$ and never from a weight, and its threshold was hung on the unset constant
(M-2). It is not deleted from the canon — it is Bet D's contrast, and it is cited to verse
01.8, which registers it against the record's own threshold. The frozen replacement is the
arithmetic kill the finding names, widened to the object the Statement is about: form
$(I-\gamma P)^{-1}$ and $\sum_{t\le K'}(\gamma P)^t$ in float64 on a drawn causal
row-stochastic $P$ at $s=64$, at **both** corners and at every $K'\in\{2,4,6,16\}$, and fire on
a relative departure above $10^{-12}$ from $\gamma^{K'+1}/(1-\gamma)$ in the formed matrix
$\infty$-norm **or in any of the twelve printed table cells**. **It fires today on this book's
own page as it stood before batch 1**: the $\gamma=0.7$ row read $0.343$ against $1.143$, a
relative departure of $0.7$. It is kept distinct from verse 01.16's kill — 01.16 recomputes the
**scalar** closed form against the record's two **banked residual numbers**, this reads the
**matrix** norm of a formed resolvent — and it carries two must-fire plants (the rows-$1.5$
non-stochastic $P$ at $\gamma=0.6$, $K'=2$ reading $7.29$ against $0.54$; a $P$ with one row
scaled by $1+10^{-6}$) and one must-not-fire (the banked pair at relative
$2.0179\times10^{-13}$). The chain is unchanged, since `replacement_survives` was true: the
cost sentence (T1, book 03), killed by K-9's band, measured on the certified device.

### 01.7 — `repair` · V-17, P-7 · repaired

> **flaw.** The stack depth is off by one and the truncation index K collides with the
> constraint-set count K that the book's own Notation fixes at 2; book 08 disambiguates the
> same object as K'.
>
> **mechanism.** V-17, P-7.
>
> **number.** S^{(K)} = sum_{u<=K} (gamma P)^u V requires C^{(0)} = S^{(0)} = V in the
> residual stream and K blocks that each compute C^{(t+1)} = gamma P C^{(t)}; the Statement
> calls it a 'depth-(K+1) stack'. The title writes gamma^L/(1-gamma) where the body writes
> gamma^{K+1}/(1-gamma). The Notation fixes 'K' as the constraint count with K=2 at the design
> point; 08_ARCHITECTURE.md:39 writes the certificate as delta = gamma^{K'+1}/(1-gamma),
> reserving K for the constraint sets.
>
> **replacement_survives.** true.
>
> **required_repair.** Rename the truncation order K' throughout 01.7 and 01.16 to match book
> 08, fix the title's exponent to gamma^{K'+1}, and state the depth as K' blocks with V
> written into the stream at initialisation, or K'+1 with the first block named.
>
> **severity.** repair.

**Repair.** All three, plus the Notation. The truncation order is renamed **$K'$ throughout
verses 01.7 and 01.16** — titles, statements, table headers, formulas, evidence lines, kills,
plants and the kills-table rows — and the book's **Notation** now carries a paragraph stating
that $K$ is the constraint-set count fixed at $K=2$ at the design point with $K+2$ value
channels, that $K'$ is the Neumann truncation order with registered support
$K'\in\{1,2,4,8,16\}$, and that `CHARTER.md` §6's scope sentence writes both with one letter
while book 08 does not and, from this repair, neither does book 01. 01.7's **title** now reads
$\gamma^{K'+1}/(1-\gamma)$, replacing a third letter ($\gamma^{L}$) for the same order. The
**depth** is corrected to the first branch offered: a paragraph in the Statement derives that
$C^{(0)}=S^{(0)}=V$ is **written into the residual stream at initialisation** and that exactly
$K'$ blocks follow, so the stack is **$K'$ blocks** and the earlier "depth-$(K+1)$ stack" was
off by one **in the direction that flattered the read**, charging the skyline one block more
than the construction needs — which the verse now says in those words. The finding's pin
`08_ARCHITECTURE.md:39` has moved under book 08's own repair; the certificate row is at `:46`
at repair time, and both books are cited by **verse id** (08.11) from here on so the pin cannot
rot again.

### 01.2 — `repair` · P-7, V-10 · repaired; replacement re-derived

> **flaw.** The Statement hedges where a number belongs, and the frozen Kill K-D2 does not
> refute it: 'not guaranteed representable' stays true when the arm copies.
>
> **mechanism.** P-7, V-10.
>
> **number.** Statement: 'is not guaranteed representable at d_model = 16 < s = 64'.
> CHARTER.md:161 forbids 'hedging words in place of a number'. K-D2 fires when ||P-hat -
> P_env||_inf < 1e-3 on >= 6 of 8 seeds; a universal negative of the form 'not guaranteed for
> all DAG families' is compatible with that outcome on one family, so the kill's firing leaves
> the Statement standing and the If killed's assertion ('the rank sentence is false on BED-S's
> DAG family') does not follow from the kill as written.
>
> **replacement_survives.** false.
>
> **required_repair.** Restate as the decidable claim K-D2 refutes: 'the DAG family drawn by
> BED-S at S-13 is not representable at d_model = 16', with the sign-rank of the drawn edge
> sets printed per seed as the number that decides it.
>
> **severity.** repair.

**Repair.** The Statement's restatement was already carried and is **verified**, not
re-claimed: it asserts "the DAG family drawn by BED-S at S-13 is not representable at
$d_{\rm model}=16$", decided per seed by $\mathrm{signrank}(E_\sigma)\le16$, with the per-seed
census `NOT MEASURED — needs the sign-rank read on the drawn edge sets, unspecified in the
canon`, and the hedge struck under `CHARTER.md:161`. Added at this batch: the Statement now
also prints **why the universal form is struck** — a "not guaranteed for all DAG families"
sentence survives K-D2 firing on one family, so its firing could not refute the verse, which
is a kill that cannot refute its own statement. **On `replacement_survives: false`, the
replacement is re-derived onto a different object and is not renamed.** The earlier
replacement re-drew the family at higher sign-rank and moved the capability sentence to the
committor head with kill "$\kappa\varepsilon$ at or above the label sd on $6$ of $8$ seeds":
$\kappa$ is `NOT MEASURED` (01.1) and in different units from the record's condition number,
the label sd is `NOT MEASURED` (B5), and the re-draw needs the same unspecified sign-rank
read K-D2 waits on — so it waited on exactly what its verse waited on. The new replacement is
the **feature-content reading**: BED-S's registered generator hands the arm the environment —
*"Node tokens carry the out-adjacency as a multi-hot feature … the transition matrix is a
deterministic function of the edge tokens"* (`READ docs/CEQ_SHAPE.md:1677-1683 @ 99777ab`,
card S-11) — so a copy at $d_{\rm model}=16$ is a **decoding** of a feature the bed supplies
and not a learnability number about a rank wall. Different object (the generator
specification, not the arm and not the edge sets), different instrument (one read of the
registered spec and the arena's `producer_cmd`), different number (the presence or absence of
the multi-hot, not a sup-norm against $10^{-3}$). Its kill: the generator emitting tokens
without the multi-hot while K-D2 still fires on $\ge6/8$ seeds, at which the decoding reading
is withdrawn and the sign-rank route is the only one left. Price $0$ GPU-s, one document read
— **strictly cheaper than K-D2, and decidable today**, where K-D2 needs a cell that does not
exist (B5). The kills-table row is rewritten to match, and the routing "S-13 is in book 04" is
corrected to "S-13 is a card of the record".

### 01.1 — `repair` · V-17, P-2 · repaired; the $\kappa$ clause re-derived

> **flaw.** kappa is read out of its units: a sup-norm condition number is substituted for a
> sup-norm inverse norm, and the source line is a paragraph the record marks as a withdrawn
> candidate.
>
> **mechanism.** V-17, P-2.
>
> **number.** The Notation defines kappa := ||(I - Q-hat)^{-1}||_inf. The Hypotheses read 'the
> sup-norm condition number of I - Q-hat is 7.9 (READ docs/CEQ_SHAPE.md:2282), so kappa <=
> 7.9'. cond_inf = ||I - Q-hat||_inf * ||(I - Q-hat)^{-1}||_inf, so kappa = 7.9 / ||I -
> Q-hat||_inf and the inequality holds only if ||I - Q-hat||_inf >= 1, which is nowhere
> printed. docs/CEQ_SHAPE.md:2282 @ 99777ab opens '*Withdrawn candidate, recorded so it is not
> re-found (V-7 discipline).*'
>
> **replacement_survives.** false.
>
> **required_repair.** Print ||I - Q-hat||_inf beside the condition number and derive kappa
> from the quotient, or read kappa directly and register it as its own RUN. Note the
> provenance of the 7.9 as a withdrawn-candidate line.
>
> **severity.** repair.

**Repair.** The provenance note and the units correction were already carried and are
**verified**: the Hypotheses quote the record's own opening *"Withdrawn candidate, recorded so
it is not re-found (V-7 discipline)"*, state that $7.9$ is **not** $\kappa$, and give
$\kappa=7.9/\|I-\hat Q\|_\infty$ with the inequality $\kappa\le7.9$ holding only if
$\|I-\hat Q\|_\infty\ge1$, which is printed nowhere. What was missing and is supplied at this
batch: **$\|I-\hat Q\|_\infty$ is now printed as the interval the class forces, and $\kappa$
is derived from the quotient with both endpoints.** For lower-triangular non-negative $\hat Q$
the $i$th row sum of $|I-\hat Q|$ is $1+r_i-2\hat Q_{ii}$, so
$1-\max_T\hat P_{ii}\le\|I-\hat Q\|_\infty\le2$, which at the judge's draw reads
$[0.3073403106639614,\,2]$ from $\max_T\hat P_{ii}=0.6926596893360386$ and gives
$\kappa\in[3.95,\,25.70]$ — a two-sided interval of width factor $6.51$, where the earlier text
printed the lower endpoint alone, which is the direction the argmin clause cannot use, and
printing one endpoint of a two-sided interval was itself the defect. **On
`replacement_survives: false`, the clause that dies is re-derived rather than renamed.** The
$\kappa$-multiplying clauses are restated on a $\kappa$-free upper bound derived here:
$\kappa\le1/\varepsilon_{\rm esc}$ with
$\varepsilon_{\rm esc}=\min_{i\in T}\sum_{a\in\mathcal A}\hat P_{ia}$, from the Neumann bound
$\|(I-\hat Q)^{-1}\|_\infty\le1/(1-\|\hat Q\|_\infty)$ on a sub-stochastic $\hat Q$. That is a
**different object** (the operator's own rows, not a norm of its inverse), read by a
**different instrument** (verse 01.19's row-sum pass, which this book already registers as
reachable today at $0$ GPU-s), returning a **different number** ($\varepsilon_{\rm esc}$, not
$7.9$), and it bounds $\kappa$ from **above**, which is the direction the argmin gap clause
needs. The verse prints the bound's own failure mode — a row with escape mass $O(e^{-L})$
sends $1/\varepsilon_{\rm esc}$ to $O(e^{L})$ — so it stays a measured quantity. $\kappa$
itself remains `NOT MEASURED`. Verse 01.14's paragraph that quoted "$\kappa\le7.9$" is
corrected to carry both endpoints ($\le9.5\times10^{-11}$ and $\le3.1\times10^{-10}$ against
$\|q\|_\infty\ge0.548718$ — wrong by ten orders and by nine), so its conclusion no longer rests
on the mis-united number.

### 01.12 — `repair` · V-17, L-FLOOR · repaired; replacement re-derived

> **flaw.** The argmin floor is printed without the uniform-distribution hypothesis that makes
> it true, contradicting the record and three neighbouring canon books.
>
> **mechanism.** V-17, L-FLOOR.
>
> **number.** 01_THEORY_ACCURACY.md:658 prints 'the zero-information floor on the argmin is
> chance, 1-1/m (0.875 at m=8)' flat. docs/CEQ_SHAPE.md:2493 @ 99777ab: 'exact zero-information
> floor 1 - max_a pi-hat(a*) from the realised class distribution; at uniform pi this is 1 -
> 1/m'. 04_BEDS_AND_INSTRUMENTS.md:43, 06_PREDICTIONS.md:246 and 09_CHESS_AND_MARKETS.md:75 all
> carry the conditional form. The admission gate admits any argmin class frequency in (0.05,
> 0.95) (docs/CEQ_SHAPE.md:1668 @ 99777ab), under which the true floor can be 0.05 -- an error
> of 0.825 against the printed 0.875, and no BED-S cell exists to measure pi-hat (B5).
>
> **replacement_survives.** false.
>
> **required_repair.** Print the floor as 1 - max_a pi-hat(a*) with the realised class
> distribution named, and 0.875 only as its uniform-pi instance, matching books 04, 06 and 09.
>
> **severity.** repair.

**Repair.** The Statement now prints
$\mathrm{floor}_{\rm zeroinfo}=1-\max_a\hat\pi(a^\star)$ **from the realised class
distribution of the argmin label on the batch the cell is scored on**, with $1-1/m=0.875$
named as its uniform-$\hat\pi$ instance at $m=8$ **and nothing more**; the unconditional form
is struck in those words. The record's conditional line is quoted at
`READ docs/CEQ_SHAPE.md:2493 @ 99777ab`, and the three neighbouring books are cited by verse
id with their moved pins printed beside the finding's originals: 04.3 (filed at `:43`, at
repair time `:59`), 06.18 (filed at `:246`, at repair time `:256`), 09.8 (filed at `:75`, at
repair time `:91`). The admission gate's band is printed with the arithmetic the finding gives:
$(0.05,0.95)$ admitted (`READ docs/CEQ_SHAPE.md:1668 @ 99777ab`), so an admitted cell may
realise a true floor of $0.05$, **an error of $0.825$ against the printed $0.875$, in the
direction that credits an arm scoring at chance with a capability**; $\hat\pi$ is
`NOT MEASURED` (B5), so **no value of this floor is printed for BED-S by this book**. The
retired Fano forms are stated as retired by book 04 verse 04.3 with no value printed here.
**On `replacement_survives: false`, two repairs.** First, the verse's Kill is split so the
floor clause has a kill at all: clause (a) is the conservation residual, clause (b) routes the
floor to `04_BEDS_AND_INSTRUMENTS.md` verse 04.3's own kill by verse id **and** adds this
book's assembly clause — any cell printing $0.875$ as the argmin floor without "at uniform
$\hat\pi$", or an argmin capability number without the realised $\hat\pi$ in the same row —
which is decidable today by reading the page and **fired on this verse before repair**.
Second, the identity half's replacement no longer re-reads the same residual: that was a V-9
rename and is struck; it is killed instead by the **undeclared-absorbing count**
$n_{\rm abs}=\#\{i\notin\mathcal A:\hat P_{ii}=1\ \text{bitwise}\}$ reading $0$ on a draw whose
conservation row is still off $\mathbb 1$ — an integer count over the diagonal, not a sup-norm
of a solve, strictly cheaper than the read it replaces, with a planted identity row as its
must-fire. The Terminal carries the conditional form.

### 01.5 — `repair` · P-3, P-10 · already repaired at batch 1; verified and tightened here

> **flaw.** The chain's second link is neither strictly cheaper nor strictly more decisive
> than the first, violating the Depth rule, and the theorem it invokes is a hop_1 bound
> registered against t* = 2.
>
> **mechanism.** P-3, P-10.
>
> **number.** 01_THEORY_ACCURACY.md:315: 'equally priced and decisive only for t* = 2'.
> CHARTER.md:139-141: 'each link's kill must be strictly cheaper or strictly more decisive than
> the one before it'. Sanford Thm 1 is cited in the same sentence as 'hop_1 over a three-symbol
> alphabet needs h m p = Omega(n)'; through 01.4's construction k=1 gives s' = 2n, so 'i.e. s
> >= 1024 at t* = 2' conflates the hop index k with the bed's t*. Omega(n) also carries an
> unstated constant that 'first violated at n > 512' silently sets to 1.
>
> **replacement_survives.** false.
>
> **required_repair.** Reorder the chain so each link is cheaper or more decisive, or collapse
> Sanford and Chen into one lineage link with a single kill. Label the Sanford row k=1 and
> print the geometry at which it bites as s' = 2n with n > h m p / c and c named or NOT
> MEASURED.
>
> **severity.** repair.

**Repair.** The second branch was taken at batch 1 and is **verified at this repair, not
re-claimed**: Sanford and Chen are collapsed into **one lineage link** carrying **no kill at
all**, the phrase "equally priced and decisive only for $t^\star=2$" appears nowhere in the
book (`grep`, this session, zero hits outside this verbatim finding), Sanford Thm 1 is labelled
$k=1$ and never $t^\star=2$, and the constant is named: $h\,m\,p=\Omega(n)$ bites at
$n>h\,m\,p/c$, at $h=1$, $m=16$, $p=32$ the product is $512$, the geometry is $n>512/c$, and
$c$ is `NOT MEASURED — needs the constant read from Thm 1's body, the same author fetch`, with
"first violated at $n>512$" struck for silently setting $c=1$. Added at this batch, and it
makes the finding's number **larger**, not smaller: the phrase "i.e. $s\ge1024$ at $t^\star=2$"
is now named and struck in the verse itself, and the position count is printed as
**$s'=2n+1$ and not $2n$** — verse 01.4's repaired construction lays $(k+1)n$ layer positions
after a **dedicated sink** at position $0$, so at $k=1$ the count is $2n+1$; the finding's
$2n$ predates that repair, and printing $2n$ would understate the geometry by one position.
**On `replacement_survives: false`: there is no replacement, and the verse says so.** 01.5
stands **OPEN**, its only conceivable kill is an author fetch of arXiv:2402.08164 that no
writer, refuter or repairer of this canon may perform, the lineage link carries no kill by
construction, and the Terminal is **in force now**: at $s=64$, $d=16$, $p=32$ no unconditional
"softmax cannot" is licensed for the 2-hop committor, and no verse of this canon can change
that by any act it is permitted to perform.

### 01.11 — `repair` · P-3, V-3 · repaired; kill split three ways, R-20 repositioned

> **flaw.** The If killed link is declared 'equally priced', violating the Depth rule, and the
> Kill refutes none of the three clauses of the Statement.
>
> **mechanism.** P-3, V-3.
>
> **number.** 01_THEORY_ACCURACY.md:643: 'the same instrument one level deeper, equally
> priced'. CHARTER.md:139-141 requires each link strictly cheaper or strictly more decisive.
> R-20's disagreement fraction above 0.5 on 512 draws cannot refute (i), which is true by
> definition of the objective; nor (ii), which is a backward-induction theorem; nor (iii),
> which is closed arithmetic on Fin 4.
>
> **replacement_survives.** false.
>
> **required_repair.** Give each clause its own kill: (i) the census that the m candidate moves
> are not all value-only, since a value-only move leaves every q^(k) constant
> (docs/CEQ_SHAPE.md:475 @ 99777ab) and the argmax label is then constant (V-12); (ii) the
> backward-induction residual against the exact V* on 512 draws at 1e-12; (iii) the Fin 4
> witness re-evaluated. Reprice the deeper link so it is strictly cheaper.
>
> **severity.** repair.

**Repair.** All four parts. The verse's Kill field now carries **three frozen kills, one per
clause**, and R-20 is removed from it with the reason printed: a disagreement fraction between
two heuristics refutes none of (i), (ii) or (iii). **Kill (i)** is the value-only census the
finding names: on the $512$ admission draws of `04_BEDS_AND_INSTRUMENTS.md` verse **04.2**, the
realised $m_{\rm eff}$ (that verse's census clauses 9 and 11, the no-op screening rate) reading
$m_{\rm eff}\le1$ on more than $0.05$ of draws, since a value-only move leaves every $q^{(k)}$
constant (`READ docs/CEQ_SHAPE.md:475 @ 99777ab`) and the argmax label is then constant (V-12);
instrument in book 04, price $0$ GPU-s, no arm, planted against $V$-only moves that must read
$m_{\rm eff}=0$. **Kill (ii)** is the exactness identity against exhaustive enumeration —
$\max_i|V^\star_{\rm DP}-V^\star_{\rm enum}|>10^{-12}$ over all $m^{|T|}\le3^{12}=531{,}441$
deterministic stationary policies at $|T|\le12$, $m\le3$ — **promoted from the If-killed, where
batch 1 had placed it, into clause (ii)'s own Kill**, because it is the only statement that
tests whether backward induction *is* the oracle, which is what (ii) asserts; both plants (the
inverted $\max\to\min$ recursion, and the Chebyshev DP that must differ) are carried. **Kill
(iii)** is the witness re-evaluated: the five numbers of the new two-epoch table
($1$, $0.6$, $0.5$, $0.7$, gap $0.2$) recomputed in float64, firing if the local and global
argmins agree or the gap reads below $10^{-12}$, with the linear scalarisation $q^{(1)}$ as the
plant under which both argmins **must** agree. **The deeper link is repriced by inverting the
chain.** R-20 is repositioned as the **replacement's** kill, where it refutes exactly what the
replacement asserts — that the one-clamp filter is a stand-in for the sequential optimum — and
it is **strictly cheaper** than Kill (ii) by the ratio of solves per draw,
$3^{12}/m=531{,}441/8=66{,}430$. The phrase "the same instrument one level deeper, equally
priced" is struck and named as both a V-9 rename and a violation of `CHARTER.md:139-141`, which
admits no "equally priced" link, and is not reinstated. The three kills are printed in strict
price order (iii) $<$ (i) $<$ (ii), and the kills-table rows are split to match.

### 01.11 — `repair` · V-3, V-11 · repaired; a two-epoch witness derived

> **flaw.** Clause (iii)'s witness shows that randomised clamps beat deterministic ones at a
> single decision epoch; it exhibits no second epoch, so it does not exhibit a failure of the
> principle of optimality, and the mixture it uses is not in the read's action set.
>
> **mechanism.** V-3, V-11.
>
> **number.** The witness has one state u, two clamps a and b absorbing with probability 1, and
> no continuation: deterministic value 1, mixture (1/2, 1/2) value 1/2. The principle of
> optimality is a statement about sub-policies of a multi-epoch problem. The safest-move read
> is an argmin over m = 8 discrete candidate moves (docs/CEQ_SHAPE.md:1736-1742 @ 99777ab,
> R-07's row-clamp restriction), and the 1/2 a + 1/2 b mixture is not a candidate, so the
> exhibited failure is unreachable on the bed as registered.
>
> **replacement_survives.** false.
>
> **required_repair.** Replace with a two-epoch witness on a DAG of depth 2 in which the
> Chebyshev-optimal prefix is not the prefix of any Chebyshev-optimal full policy, or restate
> (iii) as what the witness proves -- for a Chebyshev objective the optimum over randomised
> clamps is strictly below the optimum over deterministic clamps -- and regrade the Lean target
> chebyshev_no_bellman to match.
>
> **severity.** repair.

**Repair.** The **first** branch — the stronger of the two — is taken, and the second is
carried beside it. A two-epoch witness is **derived here** on eight positions in causal order:
$\mathcal A_{\rm sink}=\{0\}$, $\mathcal A_1=\{1\}$, $\mathcal A_2=\{2\}$ absorbing ($K=2$);
transient $x=3$ with row $e_1$, $y=4$ with row $0.6e_1+0.4e_2$, $w=5$ with row $e_2$, $v=6$
clampable, query $u=7$ with row $\tfrac12e_6+\tfrac12e_5$. The two candidate moves at $v$ are
**R-07 row clamps to transient targets** — $c:\,v\leftarrow e_3$ and $d:\,v\leftarrow e_4$ —
so both are in the read's own action set and no mixture is used. Then $q(v\mid c)=(1,0)$ with
Chebyshev value $1$ and $q(v\mid d)=(0.6,0.4)$ with value $0.6$, so **locally $d$ is optimal**;
while $q(u\mid c)=(0.5,0.5)$ with value $0.5$ and $q(u\mid d)=(0.3,0.7)$ with value $0.7$, so
**globally $c$ is optimal**. The Chebyshev-optimal prefix is not the prefix of any
Chebyshev-optimal full policy, and taking the locally optimal one costs $0.7-0.5=0.2$ at the
root — a failure of the principle of optimality with two epochs, deterministic clamps only, and
every action reachable on the bed as registered. The old one-epoch instance is **kept and
restated as what it proves**, the second branch of the required repair: for a Chebyshev
objective the optimum over randomised clamps ($\tfrac12$) is strictly below the optimum over
deterministic clamps ($1$) — an aside about the objective's action set, with the verse printing
in full that a one-epoch instance cannot exhibit a failure of the principle of optimality and
that the mixture is not a candidate under R-07, so that gap was unreachable on the bed (V-11).
The Lean target is regraded to match: 01.17's table row is renamed
`chebyshev_prefix_not_optimal` on `Fin 8` with `decide`/`norm_num` over `ℚ` and the additive
$q^{(1)}$ plant in its refusal column, and a second row carries
`chebyshev_randomised_below_deterministic` for the one-epoch `Fin 4` aside;
`chebyshev_no_bellman` appears nowhere. **On `replacement_survives: false`:** the clause's kill
is now Kill (iii) above, an arithmetic re-evaluation of the new witness at $0$ GPU-s, which the
old witness could not carry because the quantity it computed was not the quantity clause (iii)
claimed. 01.11's Hypotheses and Terminal are rewritten to carry the two objects apart.

### 01.6 — `repair` · V-17, P-10 · already repaired at batch 1; verified here

> **flaw.** The Control's must-fire threshold is imported from a log base the verse's own
> Hypotheses declare unread, and it compares a collision fraction with a
> misclassification-probability lower bound, two different quantities.
>
> **mechanism.** V-17, P-10.
>
> **number.** Control: 'separation above 13% on the layered hop_2 family at n=128' comes from
> 01.5's table cell 0.1310 = 352/2688, computed 'DERIVED with log_2'. Under natural log the
> same row reads R = 128*ln128 - 544 = 77.4 and error >= 77.4/(3*128*ln128) = 0.0415, so a
> working instrument reading 0.05 would be declared broken by a factor 3.2. 01.5's Hypotheses
> grade Peng '[V], [V-eq] owed -- the theorem's own hypotheses ... were not read at the body'.
> Separately, no theorem relates the fraction of same-sketch DAG pairs at fixed random weights
> to a lower bound on the error of the best readout.
>
> **replacement_survives.** true.
>
> **required_repair.** State the log base as a condition of the control ('under log_2; under ln
> the threshold is 0.0415') until the [V-eq] is owed, and derive the control threshold from the
> collision instrument's own calibration -- a plant with a known separation fraction -- not from
> Peng's error probability.
>
> **severity.** repair.

**Repair.** Both halves were applied at batch 1 and are **verified at this repair**, not
re-claimed. 01.6's Kill carries a section headed "Control, and its two defects named": the
must-fire plant is a **calibration** plant — a synthetic pair family with a constructed
separation fraction $\phi_0$, which the instrument must recover to within $\pm0.02$ on $512$
pairs, on the ground that an instrument that cannot recover a planted separation fraction
cannot certify a measured one — and the imported threshold is recorded with both of its defects
rather than repaired away. *(i) Log base:* the verse prints that $0.1310=352/2688$ is DERIVED
with $\log_2$, that under natural log the same row reads $R=128\ln128-544=77.4$ and error
$\ge77.4/(3\cdot128\ln128)=0.0415$, that a working instrument reading $0.05$ would therefore be
declared broken by a factor $3.2$, and that because 01.5 grades Peng `[V]` with `[V-eq]` owed
the base is not settled — so if that control is ever run it is run as "under $\log_2$ the
threshold is $0.131$; under $\ln$ it is $0.0415$", never as one number. *(ii) Units:* the verse
states that a collision fraction over same-sketch DAG pairs and a lower bound on the
misclassification probability of the best readout are two different quantities, that **no
theorem in this canon relates them**, and that the import is therefore **withdrawn as a
control** and kept only as that sentence.

### 01.8 — `repair` · R-SKY, D-1 · repaired

> **flaw.** The verse declares that the wide-skyline matched instance does not exist at the
> planned geometry, which leaves A2 -- the form that requires the depth skyline's number in the
> same row -- unmeasurable there, and the book never says which skyline A2 uses.
>
> **mechanism.** R-SKY, D-1.
>
> **number.** Statement: 'yehudai-2025-depthwidth removes depth at linear width, so the matched
> control fixes width too, and at d_model = 16 < s = 64 the wide skyline's matched instance
> does not exist.' CHARTER.md:88-91 defines A2 as 'at matched parameters against the depth
> skyline, where the skyline's learned hops fall short of the exact solve by more than MDE_N'
> and requires 'the depth skyline's number printed in the same row'. The book's only skyline
> cells are the depth-5 and depth-7 rows; 04_BEDS_AND_INSTRUMENTS.md:274 records 'two with
> settings and no cell' (B11).
>
> **replacement_survives.** false.
>
> **required_repair.** Name the skyline A2 is measured against -- depth 5 at t* = 8 by
> CHARTER.md:80 -- and state explicitly that the linear-width skyline is out of scope at
> d_model = 16, so no A2 sentence is licensed against it. Print which of the two skyline rows
> has a cell.
>
> **severity.** repair.

**Repair.** All three, and the naming is **narrower** than the required repair rather than
wider, because `CHARTER.md:88-91` requires matched parameters and book 04 splits the depth-$5$
row on exactly that. A new paragraph in 01.8's Statement names: `CHARTER.md:80` fixes the
skyline depth at $\lfloor\log_2t^\star\rfloor+2=5$ at $t^\star=8$; book 04 verse **04.9**
clause 3 splits that depth into **(3a)** the full-width, unmatched-parameter arm whose number is
printed as $\Delta_{\rm sky}$ and which **licenses no A2 sentence at all**, and **(3b)** the
reduced-width, matched-parameter arm, against which **A2 is read and against nothing else**
(the finding's pin `:274` has moved; at repair time 04.9 clause 3 is `:149` and 04.20 is
`:302`, and both are cited by verse id). The linear-width skyline of `yehudai-2025-depthwidth`
is stated to be **out of scope at $d_{\rm model}=16$** — no matched-parameter instance exists
there — so **no A2 sentence is licensed against it, by this book or any other**. Which rows
have cells is printed: of book 04's four skylines, the depth-$5$ full-width arm (3a) is the one
04.20 requires FOUND at the design point (its title reads "one FOUND, two with settings and no
cell (B11)"), while 3b has **no width, no count and no cell**, and the wide and
chain-of-thought arms have settings and no cell — and book 04's own Limits paragraph states
that **no A2 number exists anywhere in that book**. The consequence is printed in the verse and
again in its Terminal and in this book's Limits: **this book prints an A2 route and no A2
value.** **On `replacement_survives: false`:** the chain is untouched by this finding — its
links are the depth dial escalated to $7$ and then the chain-of-thought decoder, each a
registered skyline row killed by the same $\mathrm{MDE}_8$ comparison at the next depth — but
the A2 licence those links would carry is now stated as absent for all of them, since every one
is a depth row and A2 lives only on 3b's matched-width arm.

### 01.8 — `repair` · M-2 · repaired

> **flaw.** The Kill thickens Bet D's registered threshold with a seed-count clause the record
> does not carry.
>
> **mechanism.** M-2.
>
> **number.** 01.8 Kill: 'the depth-5 stack short of the shape's argmin accuracy by more than
> MDE_8 on at least 6 of 8 seeds'. docs/CEQ_SHAPE.md:2237 @ 99777ab registers Bet D's deciding
> number as 'acc_shape - acc_sky' with SPLIT band 'none' and no seed-count clause; the 6-of-8
> form belongs to Bets C, F and H in the same table. M-2 freezes thresholds where they are
> registered.
>
> **replacement_survives.** true.
>
> **required_repair.** Restate the kill as the registered paired contrast acc_shape - acc_sky
> at N = 8 against the realised paired sd, or file a corrections row adding the 6-of-8 clause
> to Bet D before using it.
>
> **severity.** repair.

**Repair.** The first branch. 01.8's Kill is restated as the registered paired contrast
$\mathrm{acc}_{\rm shape}-\mathrm{acc}_{\rm sky}$ at $N=8$ against the **realised paired sd**,
SPLIT band **none**, and **no seed-count clause**, with the struck clause quoted and the reason
printed: the $6$-of-$8$ form belongs to Bets C, F and H in the same table
(`READ docs/CEQ_SHAPE.md:2236 @ 99777ab`), and importing it thickened a registered threshold,
which M-2 forbids. The verse states that a seed-count clause on Bet D, should one ever be
wanted, enters through a `CORRECTIONS.md` row against the registration and not through a book.
The kills-table row is rewritten with the same three qualifiers, and the row that previously
paired 01.7 with 01.8 on Bet D is split, since 01.7's kill is no longer Bet D at all (the
01.7 · M-2, V-3 finding above). Verse 01.6's If-killed already cited Bet D without the seed
clause and is unchanged.

### 01.9 — `repair` · V-10, V-3 · repaired

> **flaw.** K-H1 and 01.2's K-D2 are mutually exclusive in one direction, so the book's two
> measured kills cannot both be decidable on the same draws; and the Statement is a universal
> negative its own kill cannot refute.
>
> **mechanism.** V-10, V-3.
>
> **number.** r_shape = ||(I - gamma P_env) z-hat - V||_inf / ||V||_inf and the shape solves z
> = V + gamma P-hat z exactly, so r_shape = gamma ||(P-hat - P_env) z-hat||_inf / ||V||_inf.
> K-D2 (01.2) fires when ||P-hat - P_env||_inf < 1e-3 on >= 6 of 8 seeds; on exactly those
> seeds r_shape is driven toward 0 and r_softmax/r_shape diverges, so K-H1 ('ratio <= 2')
> cannot fire there. The Statement 'No theorem converts this into an error separation' is a
> claim about the literature that no measured ratio can refute.
>
> **replacement_survives.** true.
>
> **required_repair.** Print r_shape's floor and the identification reading ||P-hat -
> P_env||_inf beside every K-H1 evaluation, and register K-H1 only on seeds where K-D2 does not
> fire. Restate the Statement as the decidable half: the joint-consistency claim at
> MATHEMATICS.md:106-127 is UNTESTED, and the ratio with its counter is what would test it.
>
> **severity.** repair.

**Repair.** Both halves. The Statement's universal negative is **struck** and replaced by the
decidable half the finding names: the joint-consistency claim at `MATHEMATICS.md:106-127` is
**UNTESTED**, the ratio with its registered counter is what would test it, and until a cell is
read the counter is the point estimate (D-CALIB). A new registration clause in the Kill carries
the finding's derivation in the verse: $(I-\gamma P_{\rm env})\hat z-V=\gamma(\hat
P-P_{\rm env})\hat z$ from $\hat z=V+\gamma\hat P\hat z$, so
$r_{\rm shape}=\gamma\|(\hat P-P_{\rm env})\hat z\|_\infty/\|V\|_\infty
\le\gamma\|\hat P-P_{\rm env}\|_\infty/(1-\gamma)$, and on a K-D2-firing seed
($\|\hat P-P_{\rm env}\|_\infty<10^{-3}$) that caps $r_{\rm shape}$ at $1.5\times10^{-3}$ at
$\gamma=0.6$ — so K-H1's "$\le2$" would need $r_{\rm softmax}\le3\times10^{-3}$ and the ratio
otherwise diverges. It follows, and the verse says so: **$r_{\rm shape}$'s floor and the
identification reading $\|\hat P-P_{\rm env}\|_\infty$ are printed beside every K-H1
evaluation, K-H1 is registered only on the seeds where K-D2 does not fire, the excluded seed
count is printed in the row, and a K-H1 verdict quoted over a seed set including a K-D2-firing
seed is VOID.** The Terminal records the sentence as UNTESTED rather than as unsupported by the
literature and withdraws every K-H1 verdict quoted over K-D2-firing seeds; the kills-table row
carries the restriction and names the $\approx34$ s pair as Bet E's (04.6).

### 01.14 — `repair` · M-2, V-17 · repaired; link 2 re-derived off the non-existent arm

> **flaw.** The verse prices K-E1 at a number that contradicts book 04's price for the
> identical frozen kill.
>
> **mechanism.** M-2, V-17.
>
> **number.** 01.14 Kill: 'approx 52 s, escalation approx 44 s ([FITTED])'.
> 04_BEDS_AND_INSTRUMENTS.md:169 (Kill (K-E1)): 'Instrument: the pair of 04.9, approx 34 s'.
> docs/CEQ_SHAPE.md Bet F @ 99777ab: 'approx 52 s, escalation approx 44 s'. Two canon books
> give two prices for one kill with one frozen threshold.
>
> **replacement_survives.** false.
>
> **required_repair.** Reconcile against the record: K-E1 runs the shape and ChaCAL-with-sink,
> not one bed-cell pair, so 52 s is the price and book 04's 34 s is the pair's. A corrections
> row must name which book is wrong before birth.
>
> **severity.** repair.

**Repair.** Reconciled exactly as the finding directs, and one further error of class is
corrected beside it. 01.14's Kill now prints: K-E1's price is $\approx52$ s with escalation
$\approx44$ s, the record's own Bet F row (`READ docs/CEQ_SHAPE.md:2238 @ 99777ab`), because
K-E1 runs the shape **and** ChaCAL-with-sink — two arms, not one bed-cell pair; the
$\approx34$ s a reader meets in book 04 is **Bet E's one bed-cell pair** at verse **04.6** (at
repair time `:102`, `:357`), the pair verse 01.9's K-H1 runs inside, and it is not K-E1's. The
class is corrected from `[FITTED]` to `[ASSUMED]`, which is what the record marks Bet F's
seconds and what this verse claimed a fit for. **No corrections row is owed**: book 04's K-E1
registration was itself repaired between the filing of this finding and its answer and now
reads $\approx52$ s `[ASSUMED]` (verse **04.11**, at repair time `:185`; the finding's pin
`:169` has moved), so the two books agree before birth and the reconciliation is a book repair
rather than a post-birth correction. **On `replacement_survives: false`, one further fact the
finding's pin exposes and this repair carries:** book 04 verse 04.11 registers K-E1
**UNREACHABLE** — `grep -rniE 'chacal|infsa' --include='*.py' .` returns $0$ lines and the
$24$-kind census holds $0$ records for every ChaCAL arm — so under `CHARTER.md:145-148` K-E1 is
not a kill today at any price, and this verse now says so. Link 2, which batch 1 had read "on
the ChaCAL-with-sink arm's own learned operator", would have died exactly where K-E1 dies and
is **moved off the arm entirely**: $\phi_0$ is now read on **the environment chain the bed
already builds** — $P_{\rm env}$ on 04.2's admitted draws with a value-zero sink **column**
appended in the record's own `V15Fork.Asink` form (`READ lean/CEQ/V15Fork.lean:67-70`) and no
partition declared — which needs **no arm, no training, no seeds and no matched parameters**,
is the census column book 04 verse 04.11's own link 1 already specifies at $0$ GPU-s, and reads
`NOT MEASURED — needs that census column` instead of `needs the sink arm's cells (B5)`. Link 2
is therefore now both strictly more decisive **and** strictly cheaper than K-E1, and the
kills-table row carries the price contrast, the UNREACHABLE mark and the moved object.

### 01.14 — `repair` · P-1 · repaired

> **flaw.** The required logit margin is misprinted: the 1.28 factor is dropped from the
> logarithm, and at the printed margin the bound exceeds the epsilon it is meant to meet.
>
> **mechanism.** P-1.
>
> **number.** L >= ln(2s / ((1-gamma) epsilon)) = ln(128 / 1e-7) = ln(1.28e9) = 20.9701, not
> the printed 20.7 (= ln(1e9) = 20.7233). At L = 20.7 the bound reads 2*64*e^{-20.7}/0.1 =
> 1.28e-6 > epsilon = 1e-6.
>
> **replacement_survives.** false.
>
> **required_repair.** Print 20.97 (or 21.0 rounded up, never down, since the bound is a lower
> bound on L).
>
> **severity.** repair.

**Repair.** $\mathbf{20.97}$ is printed, with $21.0$ named as the rounded-up form and the rule
stated — the inequality is a **lower** bound on $L$, so any rounding rounds up and $20.7$
rounded the wrong way. The verse now prints
$L\ge\ln(2\cdot64/(0.1\cdot10^{-6}))=\ln(1.28\times10^{9})=20.9701$, names the dropped $1.28$
factor, and prints the failure the misprint causes at **both** readings of the old number: at
$L=20.7$ the containment radius reads $2\cdot64\,e^{-20.7}/0.1=1.31\times10^{-6}$ and at
$\ln(10^{9})=20.7233$ it reads exactly the finding's $1.28\times10^{-6}$ — both **exceeding**
the $\varepsilon=10^{-6}$ the margin is chosen to meet — while at $20.9701$ the radius reads
$10^{-6}$ exactly. **On `replacement_survives: false`:** link 1's kill read
"$\varepsilon\|V\|_\infty$ below the label sd on $6$ of $8$ seeds", which computes
$\varepsilon$ from the same formula and needs a label sd that does not exist (B5), so it could
neither catch this defect nor fire at all. Link 1's Kill now carries **two frozen clauses**, and
the first is reachable today: clause $(\alpha)$ recomputes
$L_{\rm req}=\ln(2s/((1-\gamma)\varepsilon))$ at every registered triple and the radius
$2s\,e^{-L}/(1-\gamma)$ at every registered grid point, firing on a relative departure above
$10^{-12}$ **or on any printed $L$ at which the radius exceeds its own $\varepsilon$** — four
logarithms and four exponentials, $0$ GPU-s, no cell, no arm, and **it fires on this book as
printed before repair**; clause $(\beta)$ is the label-sd comparison, marked `NOT MEASURED` and
not reachable today. The plant for $(\alpha)$ is the misprint itself: substituting
$\ln(10^9)$ must produce $1.28\times10^{-6}>\varepsilon$ and fire.

### 01.21 — `repair` · V-15, L-EQ · repaired

> **flaw.** The Control is a pass-plant that holds by construction and cannot make the kill
> fire, and the Statement's own equality rests on a citation the book declares owed while the
> Limits paragraph asserts it is load-bearing nowhere.
>
> **mechanism.** V-15, L-EQ.
>
> **number.** Control: 'the identity Pi_0 = P at gamma = 0 must give the same tail as the
> softmax matrix itself'. At gamma = 0, Pi_0 = (1-0) P (I)^{-1} = P identically, so the control
> passes on every input including a broken SVD probe. V-15 requires the plant that makes the
> kill fire. The Statement is 'd_n(K_gamma) = sigma_{n+1}(Pi_gamma)' graded '[ASSUMED
> textbook]; the citation -- Pinkus, n-Widths in Approximation Theory -- is owed, not in
> references.bib', while the Limits paragraph says both owed citations are 'load-bearing
> nowhere'.
>
> **replacement_survives.** true.
>
> **required_repair.** Add a must-fire plant: a matrix with a known rank-16 truncation error (a
> rank-16 matrix plus 1e-2 times a random perturbation) must read sigma_17/sigma_1 near 1e-2,
> not below 1e-3. Either add Pinkus to references.bib or grade the Statement DERIVED from the
> singular-value decomposition of a linear image of a ball, which is a two-line derivation.
>
> **severity.** repair.

**Repair.** Both, and the second by the branch that does not require touching another file —
no file but this book is edited at this repair. **Control:** the $\gamma=0$ identity is demoted
in place, with the arithmetic printed — $\Pi_0=(1-0)P(I-0\cdot P)^{-1}=P$ identically, so it
passes on every input including a probe whose singular values are noise, an empty rejection
region, the exact defect V-15 names — and is relabelled a **wiring check** that the probe forms
$\Pi_\gamma$ from the right matrix. The **must-fire plant the finding specifies is added
verbatim in substance**: $M=R+10^{-2}E$ with $R$ of exact rank $16$ and $E$ normalised to
$\|E\|_2=\|R\|_2$ must read $\sigma_{17}(M)/\sigma_1(M)$ near $10^{-2}$ and **never below
$10^{-3}$**. A second plant is added in the firing direction: an exactly rank-$16$ matrix must
read the ratio at the float64 noise level and **fire** the kill. **Grade:** the Statement is
regraded **DERIVED** by the two-line derivation the finding names —
$\sup_{x\in\mathcal K_\gamma}\mathrm{dist}(x,X_n)=\|(I-\Pi_{X_n})\Pi_\gamma\|_2$, minimised
over $n$-dimensional $X_n$ by Schmidt–Eckart–Young at $\sigma_{n+1}$ with minimiser
$\mathrm{span}(u_1,\dots,u_n)$ — so `[ASSUMED textbook]` is gone, nothing in this book rests on
Pinkus, and the Limits paragraph's "load-bearing nowhere" is now true of what remains rather
than contradicting a Statement that leaned on the owed citation. Pinkus stays recorded as the
standard reference with its citation owed in `references.bib`, and the Limits paragraph is
rewritten to say **one** citation is owed and load-bearing nowhere (01.10's Dirichlet-form
Cheeger) and to name the contradiction that has been removed.

### 01.17 — `repair` · P-1, P-10 · repaired

> **flaw.** The [D] grade for hitting_time_transform is evidenced by a repository grep, and the
> object it claims absent is present in the vendored Mathlib at the book's own pinned revision.
>
> **mechanism.** P-1, P-10.
>
> **number.** Table row: 'hitting time, 01.3 | hitting_time_transform | [D] | no finite-chain
> hitting-time object (RUN[P] grep, zero files)'. lean/.lake/packages/mathlib at rev
> a45ae63747140c1b2cbad9d46f518015c047047a (the book's pin, lean/lake-manifest.json:61) carries
> Mathlib/Probability/Process/HittingTime.lean:51, 'noncomputable def hitting [Preorder iota]
> [InfSet iota] (u : iota -> Omega -> beta) (s : Set beta) (n m : iota) : Omega -> iota'. Every
> other row of the table cites a Mathlib file:line and all four checked resolve exactly
> (Block.lean:265, Block.lean:346, NonsingularInverse.lean:151, SchurComplement.lean:202).
>
> **replacement_survives.** true.
>
> **required_repair.** Regrade the row to [S] with the gap named precisely -- Mathlib's hitting
> is a stopping time on a filtered process, and no finite-state absorption identity
> E_i[gamma^{tau}] = ((I - gamma Q)^{-1} R 1)_i is derived from it -- and cite
> HittingTime.lean:51 as the object that exists.
>
> **severity.** repair.

**Repair.** The row is regraded **`[S]`**, its Mathlib column now cites
`MeasureTheory.hitting` at `Mathlib/Probability/Process/HittingTime.lean:51` as the object that
**exists**, and its refusal column carries `forward_pointer_reads_zero` re-used from 01.4 (a
chain with no path to $\mathcal A_k$ must read $E_i[\gamma^{\tau_k}]=0$). A note below the table
records the whole finding in the verse: the pin is confirmed by `RUN` this session — the rev at
`lean/lake-manifest.json:61` is `a45ae63747140c1b2cbad9d46f518015c047047a` and
`grep -n 'noncomputable def hitting'` on the vendored package at that revision returns line
$51$ — and the struck evidence is named for what it was, a grep of the **repository's own**
`lean/CEQ/` tree offered as evidence about Mathlib (P-1), which is the only evidence the `[D]`
carried, in a table where every other row cites a Mathlib `path:line` and all four checked
resolve exactly. The gap is stated precisely rather than as an absence: Mathlib's `hitting` is
a **pathwise stopping time on a filtered process**, indexed by a `Preorder` with `InfSet`, and
what verse 01.3 needs is the **finite-state absorption identity**
$E_i[\gamma^{\tau_k}\mathbb 1_k]=((I-\gamma Q)^{-1}\gamma R_k\mathbb 1)_i$ on a row-stochastic
matrix with an absorbing partition; **no route from the first to the second exists in this
Mathlib**, which is a missing bridge and not a missing object. This book's Limits paragraph
carries the correction as one Lean grade that was factually wrong at the book's own pin.

---

**Batch 2 disposition, counted exactly.** Sixteen findings filed — sixteen `repair`, zero
`strike` — and sixteen answered. **Fourteen were repaired at this batch**: 01.7 twice (the
unfrozen $\gamma_{\rm env}$ kill; the $K\to K'$ rename with the depth off-by-one), 01.2, 01.1,
01.12, 01.11 twice (the three-way kill split with R-20 repositioned; the two-epoch witness),
01.8 twice (the A2 skyline naming; Bet D's thickened threshold), 01.9, 01.14 twice (the K-E1
price with link 2 moved off the non-existent arm; the logit margin), 01.21, 01.17. **Two were
verified as already carrying the required repair** from batch 1 and are recorded as verified
rather than re-claimed — 01.5 and 01.6 — with 01.5 additionally tightened at this batch
($s'=2n+1$ and not $2n$, and the struck "$s\ge1024$ at $t^\star=2$" named in the verse). No
finding is left unanswered, no verse was deleted to escape one, and nothing in this section
reads more gently than when it was filed.

**Six replacements were re-derived onto a different object and a different instrument because
the old one died with its verse (V-9).** 01.2's, from a re-drawn family killed by
$\kappa\varepsilon$ against an absent label sd, to the **feature-content reading** on S-11's
generator specification, killed by one document read at $0$ GPU-s. 01.1's $\kappa$-multiplying
clauses, from an unread inverse-norm probe to the **escape-mass bound**
$\kappa\le1/\varepsilon_{\rm esc}$ read by verse 01.19's row-sum pass. 01.12's, from the same
conservation residual re-read to the **undeclared-absorbing count** $n_{\rm abs}$, a bitwise
diagonal pass. 01.11's, from R-20 one level deeper to a chain in which the enumeration identity
is clause (ii)'s **own kill** and R-20 becomes the **replacement's** kill at $66{,}430\times$
fewer solves per draw. 01.14's link 2, from **the ChaCAL-with-sink arm** — which book 04
registers as $0$ source lines and $0$ records — to **the bed's own environment chain** with a
value-zero sink column, needing no arm at all. 01.14's link 1, from a label-sd comparison that
cannot fire (B5) to the **margin arithmetic**, which fires today on this book's own printed
$L=20.7$.

**What this batch withdrew and did not replace.** No A2 number is licensed by this book: A2
lives only on book 04's matched-width depth-$5$ skyline, which has no width, no count and no
cell, and never on the linear-width arm, which has no matched instance at $d_{\rm model}=16$.
No value of the argmin zero-information floor is printed for BED-S: the floor is
$1-\max_a\hat\pi(a^\star)$ and $\hat\pi$ is `NOT MEASURED`. No clause is registered at
$\gamma_{\rm env}$: the constant carries no value in the record or the canon, and verse 01.7 is
registered at $\gamma\in\{0.6,0.9\}$ instead. K-E1 is UNREACHABLE, not merely uncelled — the
arm it compares does not exist in the tree. Verse 01.5 stands **OPEN** with no replacement at
all, its Terminal in force now: at $s=64$, $d=16$, $p=32$ no unconditional "softmax cannot" is
licensed for the 2-hop committor, and no verse of this canon can change that by any act it is
permitted to perform.

---

MARS's findings, repair round 1, **batch 3 of 3**. Every finding appears **verbatim** — verse,
flaw, mechanism, number, severity — beside the repair applied or the sentence now in force.
Nothing here reads more gently than when it was filed (`CHARTER.md` §7).

### book-level — `repair` · P-1, V-17 · repaired

> **verse.** book-level
>
> **flaw.** The Limits paragraph's claim about every RUN[J]/RUN[M] figure is contradicted by
> three figures the book itself cites.
>
> **mechanism.** P-1, V-17
>
> **number.** Limits: 'every RUN[J]/RUN[M] figure is one float64 draw at s = 32, gamma = 0.6,
> seed 0'. 01.14 cites RUN[J] 0.5263157894736843 which the source records at s = 3, gamma =
> 0.9 (docs/CEQ_SHAPE.md:405 @ 99777ab). 01.10(b) cites rho(Q_BED-1) = 0.9408612510154677 on
> BED-1 with |T| = 9, not s = 32. 01.15 cites RUN 1.127 at s = 64.
>
> **replacement_survives.** false.
>
> **required_repair.** Restate as 'every RUN[J]/RUN[M] figure carries its own draw; the
> judge's draw is s = 32, gamma = 0.6, seed 0, and the exceptions are the mask-amplification
> instance at s = 3, gamma = 0.9, the BED-1 chain at |T| = 9, and the Gaussian-V plant at
> s = 64.'
>
> **severity.** repair.

**Repair.** The Limits paragraph now reads that every `RUN[J]`/`RUN[M]` figure **carries its
own draw**, names the judge's draw as $s=32$, $\gamma=0.6$, seed 0, and prints all three
exceptions with their pins: the mask-amplification instance at $s=3$, $\gamma=0.9$,
$\varepsilon=0.1$ (`READ docs/CEQ_SHAPE.md:405-406 @ 99777ab`), the BED-1 chain at $|T|=9$
($\rho(Q_{\rm BED\text{-}1})=0.9408612510154677$), and the Gaussian-$V$ plant at $s=64$
($\max|\Delta z|=1.127$, beside $0.1096$ at $s=32$). Verse 01.14's own Evidence line now
prints its draw ($s=3$, $\gamma=0.9$, $\varepsilon=0.1$) beside the number rather than relying
on a blanket paragraph. **On `replacement_survives: false`:** the Limits paragraph is prose and
carries no chain, so there was nothing to replace; the defect was a universal quantifier over
three known counterexamples on this book's own pages, and the repair is the quantifier's
removal, not a substitute claim.

### book-level (01.6, 01.7) — `repair` · D-1, R-SKY · repaired

> **verse.** book-level (01.6, 01.7)
>
> **flaw.** Two comparative sentences appear outside forms A1/A2/T1/T2 without the bed,
> floor, N and matched count CHARTER §1 requires.
>
> **mechanism.** D-1, R-SKY
>
> **number.** 01_THEORY_ACCURACY.md:429 'the learned stack does better than iteration' --
> names the bed and MDE_8 but no floor and no matched count.
> 01_THEORY_ACCURACY.md:375 'the shape's advantage on BED-S is exactness and cost, not fit' --
> 'advantage' with no form named. CHARTER.md:106-109 strikes such a sentence at refutation.
>
> **replacement_survives.** false.
>
> **required_repair.** Rewrite 429 as the A1-form comparison with the floor (L-FLOOR) and the
> matched parameter count printed in the same clause, and 375 as 'what survives on BED-S is
> exactness (delta = 0) and the cost sentence, and no fit sentence is licensed'.
>
> **severity.** repair.

**Repair, in three places, because the contrast moved between verses during the round.** The
line the finding numbers 429 is verse 01.7's learned-stack kill, which an earlier batch of this
round struck on a different ground — it cannot refute an attained matrix identity, and its
threshold was registered at the valueless $\gamma_{\rm env}$ — and whose contrast it forwarded
to verse 01.8. That strike removed the kill and **left the bare comparative unaddressed**,
which is this finding's charge and is now answered where the sentence lives:

1. **01.7** gains the paragraph *"The form the contrast is written in, and the bare
comparative that is struck with it"*, which names the struck phrasing as a comparative outside
A1/A2/T1/T2, states the **A1** form with all four required items in one clause — the bed
(BED-S at $t^\star=8$, $m=8$, $K=2$, $N=8$), the floor (**L-FLOOR**: the argmin
zero-information floor $1-\max_a\hat\pi(a^\star)$, `NOT MEASURED`, $0.875$ only as the
uniform-$\hat\pi$ instance at $m=8$), the **matched parameter count** ($4{,}769$ against book
04 clause 3b's count within $0.0960$ per cent, `NOT MEASURED`) and $\mathrm{MDE}_8$ at the
realised paired sd (`NOT MEASURED`) — and then states that three of the four are `NOT
MEASURED`, so **no A1 value is printed and no fit sentence is licensed there**.
2. **01.8**'s Frozen Bet D block, which is where the contrast now lives, is rewritten in **A2**
form with the floor and the matched count inside the same clause and the depth skyline's own
number required in the same row, and closes with the flat prohibition: a sentence of the form
*"the learned stack does better than iteration"*, naming the bed and $\mathrm{MDE}_8$ but
neither the floor nor the matched count, **appears nowhere in this book**.
3. **01.6**'s Bet-D-prediction branch is restated exactly as the finding requires — *"what
survives on BED-S is exactness ($\delta=0$) and the cost sentence, and no fit sentence is
licensed"* — with the word "advantage" named as the comparative it is and struck.

**On `replacement_survives: false`:** neither sentence was a verse chain, so neither carried a
replacement; what stood behind them was a licensing claim, and the repair withdraws the claim
rather than re-siting it. No A1 and no A2 **value** is added: both remain routes whose floor
and matched count are `NOT MEASURED`.

### book-level (01.5, 01.14) — `note` · P-2 · repaired, and twelve further pins corrected

> **verse.** book-level (01.5, 01.14)
>
> **flaw.** Two READ pins do not resolve to the number they carry at the book's own commit.
>
> **mechanism.** P-2
>
> **number.** 01.14 cites ':407-408' for 'RUN[J] 0.5263 against naive 0.1'; at 99777ab those
> numbers sit at docs/CEQ_SHAPE.md:405-406 and lines 407-408 carry the rowsum census and the
> beta-not-1 rows. 01.5 cites ':2947-2950' for the vacuity inequality; '384 < 544' appears at
> docs/CEQ_SHAPE.md lines 128, 1421, 1580, 1614, 2559, 2593, 2952, 3071, 3072 and 3164, and in
> none of 2947-2950.
>
> **replacement_survives.** false.
>
> **required_repair.** Repin to :405-406 and :2952. Re-verify every READ range in the book by
> grepping the cited string at 99777ab; docs/CEQ_SHAPE.md is byte-identical between 99777ab
> and HEAD, so the check is mechanical.
>
> **severity.** note.

**Repair, and the wider census the required check produced.** 01.14 is repinned to
`:405-406` with the failed pin named in the text: `:407-408` resolves at `99777ab` to the
row-sum census and the $\hat\beta\ne1$ rows, and to neither number. 01.5's `:2947-2950` was
already repinned to `:2952` at an earlier batch of this round and is re-verified here —
`git show 99777ab:docs/CEQ_SHAPE.md | sed -n '2952p'` (`RUN`, this session) prints *"this reads
$384 < 544$ (DERIVED, $\log_2$), so the inequality is **vacuous**"*. The mechanical sweep the
finding asks for was then run over every `docs/CEQ_SHAPE.md` pin in this book against the
$3{,}311$-line file at `99777ab` (`RUN`: `git show 99777ab:docs/CEQ_SHAPE.md`, one `sed -n` per
cited range). **Twelve further pins did not resolve and are corrected**, each to the line that
carries the cited string:

| claim | pin as printed | pin at `99777ab` |
|---|---|---|
| $\mathcal A_k$ on the wrong set moves $q$ by $O(1)$ (01.1) | `:388` | `:387` |
| $\rho(Q)=0.692660$ declared against $1.000000$ undeclared (01.1) | `:1436`, a blank line | `:1440` |
| the must-fire perturbation, ratio $9.6\times10^{10}$ (01.3) | `:921` | `:918` |
| Bet D's registration (01.6, 01.8) | `:2237`, Bet E's row | `:2236` |
| Bet F's price row (01.14) | `:2237`, Bet E's row | `:2238` |
| bind B-I, the $\gamma=0.7$ plant $119.37/1.143$ (01.7, 01.11, 01.17) | `:3113`, B-G2's row | `:3116` |
| K-E1 as frozen (01.14) | `:3180`, a table header | `:2274` |
| $\mathrm{rank}(\Delta P)$ for a token rewrite (01.15) | `:473` | `:471` |
| Q3/W1, the Koopman field (01.19) | `:1199` | `:1222` |
| the `lambda_hat` one-bit defect (01.19) | `:1275-1278` | `:1280` |
| Q5/W3, finite-mixture inference (01.22) | `:1204` | `:1227` |
| Q1/W3, realisation theory (01.18) | `:1197` | `:1220` |
| Q5/W1, the $n$-width field (01.21) | `:1203` | `:1226` |

(Thirteen rows for twelve pins: `:2237` was wrong twice, once for Bet D and once for Bet F, and
resolves to two different lines.) The two occurrences of `:2237` and the one of `:3113` that
sit **inside** MARS's own quoted findings above are left exactly as MARS wrote them; only this
book's own prose is repinned.

**What the sweep did not do, stated so the check is not overclaimed.** Ranges naming a card
(S-02, S-11, S-17, R-07, R-20, J-D3, J-D8, J-L4, J-L14) were verified at the line carrying the
card header and, where the range spans a card body, at the line carrying the quoted string —
`:1677-1683` was checked to the quoted multi-hot clause at `:1679` — but not string by string
through every card body; a pin that lands inside the right card on the wrong line of it would
survive this sweep. `docs/CEQ_SHAPE.md` is byte-identical between `99777ab` and HEAD, so the
sweep is repeatable exactly as filed. **On `replacement_survives: false`:** a pin carries no
chain; the defect is a false citation and the repair is the true one, with the false pin kept
in the text beside it so the correction is legible.

### 01.19 — `note` · P-10 · repaired

> **verse.** 01.19
>
> **flaw.** The Koopman operator on linear observables is gamma P transpose plus an affine
> shift, not gamma P; the spectral conclusion survives, the stated identification does not.
>
> **mechanism.** P-10
>
> **number.** For T(x) = V + gamma P x and f(x) = c^T x, (Uf)(x) = f(T(x)) = gamma (P^T c)^T x
> + c^T V. The Statement reads 'the Koopman operator ... on linear observables is gamma P
> itself'. spec(P^T) = spec(P), so 'the spectrum is {gamma P_ii}' stands.
>
> **replacement_survives.** true.
>
> **required_repair.** Write 'is gamma P^T on linear observables, affine by c^T V', and note
> spec(P^T) = spec(P) so the diagonal read is unchanged.
>
> **severity.** note.

**Repair.** The Statement now reads that the Koopman operator of $T:x\mapsto V+\gamma Px$ is
$\gamma P^\top$ on linear observables, **affine by $c^\top V$**, and prints MARS's own
computation as the derivation: $(Uf)(x)=c^\top(V+\gamma Px)=\gamma(P^\top c)^\top x+c^\top V$,
so $U$ acts on the coefficient vector as $c\mapsto\gamma P^\top c$ and the linear observables
are invariant only modulo constants. The spectral conclusion is kept **with the reason
printed**: $\det(\lambda I-P^\top)=\det((\lambda I-P)^\top)=\det(\lambda I-P)$, so
$\mathrm{spec}(P^\top)=\mathrm{spec}(P)$ and $\{\gamma P_{ii}\}$ is unchanged; the clause
"$\gamma P$ is lower-triangular" now carries "$\gamma P^\top$ upper-triangular, same diagonal"
beside it. The verse's Kill, its row-sum census, its replacement and its Terminal are untouched
— the defect was the identification of the operator, not the spectrum the verse reads off it,
and the verse says which of the two was wrong rather than calling the error immaterial.

### 01.6 — `note` · V-17 · already repaired at an earlier batch; verified here

> **verse.** 01.6
>
> **flaw.** The pigeonhole count omits the bits the readout carries in x_i, which 01.2's
> Hypotheses give the out-adjacency multi-hot.
>
> **mechanism.** V-17
>
> **number.** The readout is of (x_i, O_i); 01.2's Hypotheses: 'node tokens carry
> out-adjacency as a multi-hot feature'. That adds up to s = 64 bits, so the joint sketch
> carries at most d p + s = 576 bits and the fibre size is 2^{2016-576} = 2^{1440}, not the
> printed 2^{2016-512} = 2^{1504}. The conclusion is unaffected; the exponent is not.
>
> **replacement_survives.** true.
>
> **required_repair.** Print 2^{1440} with the x_i budget named, or state the count for O_i
> alone and say the readout also sees x_i.
>
> **severity.** note.

**Repair, verified at this batch.** 01.6's Statement carries the joint budget
$d\,p+s=576$ bits with the $x_i$ term named as 01.2's out-adjacency multi-hot, prints
$2^{2016-576}=2^{1440}$, and keeps $2^{1504}$ only in the parenthesis saying it is the count
for $O_i$ alone and the wrong budget for a readout that also sees $x_i$. The finding stands as
filed; being answered before this batch opened does not soften it.

### 01.6 — `note` · P-1 · already repaired at an earlier batch; verified here

> **verse.** 01.6
>
> **flaw.** The interval of admissible epsilon is printed with its endpoints transposed, and
> |T| in {57,60} is declared DERIVED with no derivation and excludes values the stated
> geometry admits.
>
> **mechanism.** P-1
>
> **number.** 2^{-512/240} = 0.2278 and 2^{-512/228} = 0.2107, so the text 'epsilon <
> 2^{-512/240} ... 2^{-512/228} = 0.211 ... 0.228' maps each endpoint to the wrong expression.
> The Notation asserts |T| in {57,60} at |A_bullet| in {1,2}, K=2, s=64; four sets of size 1 or
> 2 give |A| in {4,...,8} and |T| in {56,...,60}.
>
> **replacement_survives.** true.
>
> **required_repair.** Print '2^{-512/228} = 0.2107 to 2^{-512/240} = 0.2278'. Either derive
> |T| in {57,60} by naming the set-size convention that forbids |A| = 5, 6 and 8, or print |T|
> in {56,...,60} and the corresponding coordinate range {224,...,240}.
>
> **severity.** note.

**Repair, verified at this batch, on the finding's second branch.** The book took the branch
the finding offers second: the Notation and 01.6 both print
$|\mathcal A|\in\{4,\dots,8\}$ and $|T|=64-|\mathcal A|\in\{56,\dots,60\}$ with the derivation
in the Notation, and both state that the earlier $\{57,60\}$ was DERIVED with no derivation and
excluded values the stated geometry admits. The coordinate range is therefore
$|T|(K+2)\in\{224,\dots,240\}$ and **not** $\{228,240\}$, so the interval the finding computes
for $\{57,60\}$ is superseded by the wider one: the book prints
$\varepsilon<2^{-512/224}=0.2050$ to $2^{-512/240}=0.2278$, **each endpoint against the
expression that produces it**, with the transposition named. The finding's own endpoints stand
as the interval for the geometry it was filed against, and the book does not present the two as
the same number.

### 01.15 — `note` · V-3 · repaired

> **verse.** 01.15
>
> **flaw.** The printed bound carries an unexplained (1+gamma) factor absent from its own
> derivation, and a rank equality is asserted where only an inequality follows.
>
> **mechanism.** V-3
>
> **number.** O_true - O_cached = (1-gamma)(I - gamma P)^{-1} Delta P z', giving ||O_true -
> O_cached||_inf <= ||Delta P||_inf ||V'||_inf/(1-gamma) with no (1+gamma). The display
> multiplies by (1+gamma). 'For a token rewrite rank(Delta P) = s - i' -- rewriting token i
> changes s-i rows, so rank(Delta P) <= s - i; the record's own line (docs/CEQ_SHAPE.md:473 @
> 99777ab) carries the same overclaim.
>
> **replacement_survives.** true.
>
> **required_repair.** Drop the (1+gamma) or show the step that introduces it. Write
> rank(Delta P) <= s - i and keep the suffix re-solve price (s-i)^2 d/2, which the inequality
> already supports.
>
> **severity.** note.

**Repair, both halves.** The $(1+\gamma)$ is **dropped**, and the four-step derivation with no
room for it is printed in the Statement:
$\Pi_\gamma(P)=\tfrac{1-\gamma}{\gamma}\big((I-\gamma P)^{-1}-I\big)$, so the constant cancels
in the difference; the resolvent identity turns the difference into
$(1-\gamma)(I-\gamma P')^{-1}\Delta P\,z'$; and $\|(I-\gamma P')^{-1}\|_\infty\le1/(1-\gamma)$
with $\|z'\|_\infty\le\|V'\|_\infty/(1-\gamma)$ give
$\|\Delta P\|_\infty\|V'\|_\infty/(1-\gamma)$, the two $(1-\gamma)$ factors cancelling once.
The text records that the factor had no derivation behind it and inflated the bound by up to
$2\times$ as $\gamma\uparrow1$. The rank equality becomes $\mathrm{rank}(\Delta P)\le s-i$ with
its reason: a token rewrite at $i$ changes the $s-i$ rows at or after $i$ and no earlier row,
so $\Delta P$ has at most $s-i$ non-zero rows, and equality fails on any rewrite that leaves a
later row's logits fixed. The finding's pin for the record's own copy of the overclaim is
`:473`; the string *"$\Delta P$ has rank $s-i$"* sits at `:471` at `99777ab` (`RUN`, this
session), the verse cites `:471`, and the finding's `:473` is kept above as filed. The suffix
re-solve price $(s-i)^2d/2$ is unchanged and is now stated as charged on the $s-i$ **rows**
that may move rather than on the rank, which is what the inequality supports.

### 01.20 — `note` · V-10 · repaired; replacement re-derived

> **verse.** 01.20
>
> **flaw.** The Hypotheses' condition holds by construction, so the If killed repairs a
> failure that cannot occur.
>
> **mechanism.** V-10
>
> **number.** Hypotheses: 'I - Q_{T'T'} a unit ... whenever 0 not in T''. With 0 in A_sink
> declared (docs/CEQ_SHAPE.md:295 @ 99777ab), 0 is not in T and hence not in any T' subset of
> T, so the condition is satisfied at every draw. If killed: 'the identity re-derived with the
> sink block kept explicit ... its Kill is the same residual with 0 not in T' enforced by a
> census line'.
>
> **replacement_survives.** false.
>
> **required_repair.** Drop the vacuous condition or state the reachable one it stands in for:
> I - Q_{T'T'} fails to be a unit exactly when some i in T' has P-hat_ii = 1, which finite
> logits forbid and a saturated gate (01.17's cut_makes_segment_head_absorbing) produces.
> Register that as the census line.
>
> **severity.** note.

**Repair.** The vacuous condition is named as vacuous in the Hypotheses — $0\in\mathcal
A_{\rm sink}$ is declared (`READ docs/CEQ_SHAPE.md:295 @ 99777ab`), so $0\notin T'$ for every
$T'\subset T$ at every draw and the antecedent could not fail — and is replaced by the reachable
one the finding derives, with the determinant that makes it decidable printed:
$\det(I-Q_{T'T'})=\prod_{i\in T'}(1-\hat P_{ii})$ on the lower-triangular causal class, so
$I-Q_{T'T'}$ fails to be a unit **exactly when some $i\in T'$ has $\hat P_{ii}=1$**, which
finite logits forbid and 01.17's `cut_makes_segment_head_absorbing` saturated gate produces.
That is registered as the verse's census line — $\max_{i\in T}\hat P_{ii}<1$ on every cell,
$0$ GPU-s, read by 01.19's row-sum and diagonal pass, $0.6926596893360386$ at the judge's draw
(`RUN[J]`), so it passes there and fires only on a saturated-gate cell.

**On `replacement_survives: false`, and the replacement re-derived rather than renamed
(V-9).** The old If-killed re-derived the same identity with the sink block explicit and set
its Kill to *"the same residual with $0\notin T'$ enforced by a census line"* — the same
number, the same instrument and the same object, enforced by the condition just shown to hold
always; it died with the verse. The new replacement is the **declared-sink repair** of 01.1's
If-killed, and its three coordinates are each different: its **object** is the boundary
declaration and the diagonal of $\hat P$, not the eliminated blocks; its **instrument** is
01.19's row-sum and diagonal pass, not the $512$-draw Schur residual battery; its **number** is
$\max_{i\in T}\hat P_{ii}$ against $1$ — $0.6926596893360386$ (`RUN[J]`), with $\rho(Q)=0.692660$
declared against $1.000000$ undeclared (`RUN[P]`, `READ docs/CEQ_SHAPE.md:1440 @ 99777ab`) —
and not a residual at all. It carries its own Hypotheses, Evidence and plant, and a Kill
strictly cheaper than the battery it follows (one float compared against $1$): the census
passing on every cell **while the Schur residual still exceeds $10^{-12}$**, at which the chain
ends at the Terminal and the Schur complement is withdrawn as a certificate input. The verse's
own kill remains UNREACHABLE for want of the Schur instrument, which book 04 does not carry;
that stands in the Limits paragraph and this repair does not change it.

### 01.22 — `note` · M-3 · repaired

> **verse.** 01.22
>
> **flaw.** The two within-cluster spread bounds use different methods in one sentence, and
> the five-point bound is the interval width itself rather than the maximum sample sd.
>
> **mechanism.** M-3
>
> **number.** For n = 3 the verse uses w/sqrt(3) = 0.039027/1.732 = 0.0225, the correct
> maximum. For n = 5 it uses w = 0.0280; the maximum sample sd of five points in an interval
> of width w is w*sqrt(0.3) = 0.5477*0.028019 = 0.01535. The printed ratio sigma_pool/sigma_w
> >= 0.253673/0.0280 = 9.06 is therefore loose; the correct bound is >= 16.5. The direction is
> safe (the claim is a lower bound) but the two halves of one sentence are computed
> differently.
>
> **replacement_survives.** true.
>
> **required_repair.** Use w*sqrt((n-1)/n)/sqrt(n-1) consistently -- w/sqrt(3) = 0.0225 for
> the three-point cluster and w*sqrt(0.3) = 0.01535 for the five-point one -- and reprint the
> ratios as >= 11.3 and >= 16.5.
>
> **severity.** note.

**Repair.** One method is written out and both clusters are computed from it: the maximum
sample sd (denominator $n-1$) of $n$ points in an interval of width $w$ is attained by
splitting them between the endpoints, giving
$\sigma_w^{\max}(n,w)=w\sqrt{\lceil n/2\rceil\lfloor n/2\rfloor/n}\big/\sqrt{n-1}$, which is
$w/\sqrt3=0.5774\,w$ at $n=3$ and $w\sqrt{0.3}=0.5477\,w$ at $n=5$ (DERIVED). The crossed
cluster ($n=5$, $w=0.028019$) therefore has $\sigma_w\le0.0153454$ and **not** $0.0280$, which
is the interval width and not a sample sd; the NO-READING cluster ($n=3$, $w=0.039027$) has
$\sigma_w\le0.0225322$. The reprinted ratios are
$\sigma_{\rm pool}/\sigma_w\ge0.253673/0.0153454=16.53$ for the crossed cluster and
$\ge0.253673/0.0225322=11.26$ for the other, against the earlier $9.06$ and $11.3$, and the
closing clause reads "at least **eleven** times" where it read "at least nine times". The text
records that the direction was safe — the claim is a lower bound and the old number was the
looser one — and that the defect was one sentence computed by two methods, which is what M-3
names. The verse's Kill, replacement and Terminal are unchanged: none of the three depends on
the value of $\sigma_w$, only on the split existing and the gap $[0.663,1.113]$ being empty.

### Batch 3 — what changed, and what it did not buy

**Replacements re-derived because the pre-written one died with its verse (V-9).** One:
01.20's, from the same Schur residual with a vacuous enforcement clause to the declared-sink
repair on the diagonal of $\hat P$ — a different object, read by 01.19's row-sum pass rather
than the residual battery, returning $\max_{i\in T}\hat P_{ii}$ rather than a residual, at a
strictly cheaper kill. The three other `replacement_survives: false` findings of this batch —
the two book-level ones and the pin census — attach to prose that carries no chain; each is
answered by withdrawing the false claim rather than by substituting a new one, and each says so
in its own row.

**What this batch withdrew and did not replace.** The blanket draw sentence in Limits: three of
this book's own figures are not at $s=32$, $\gamma=0.6$, seed 0, and each now carries its own
draw. The bare comparative in every form: no sentence of this book compares the shape to a
stack outside A1/A2/T1/T2, and the A1 and A2 clauses that remain are **routes with three of
four terms `NOT MEASURED`**, not numbers. The $(1+\gamma)$ in 01.15's cached-mixture bound,
which no step produced. The rank equality for a token rewrite, now an inequality. Fourteen
`READ` pins that did not resolve at `99777ab`, corrected with the failed pin kept beside each.
Nothing in this batch produced a measured number: every repair is arithmetic, a pin, or a
withdrawal, and the six UNREACHABLE kills, the seventh (K-E1) and the two OPEN verses (01.5,
01.10) stand exactly as the earlier batches left them.
