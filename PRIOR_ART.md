# PRIOR ART — the Hankel program's citation of record

Round 7, Foreman, iteration 0. Normal English by RULE 4's artifact exemption.

This file exists because the contract requires the delta to *"Transformers learn
shortcuts to automata"* stated **in writing before the Krohn–Rhodes framing
appears in any claim** (§1.1). It also carries the citations for the rest of the
Hankel program, because those citations turned out to need corrections that
belong next to the delta rather than scattered.

**Locator convention.** Every quotation below carries a source and a locator and
was taken from the primary text. Where a source could not be reached, that is
said in the line rather than papered over. Nothing here is quoted from a
description of a paper.

**Three of this file's findings contradict the contract.** They are marked
**CONTRACT CORRECTION** and each says what to write instead.

---

## 1. THE NEAR-MISS — what it is, and what it is not

**Liu, Bingbin; Ash, Jordan T.; Goel, Surbhi; Krishnamurthy, Akshay; Zhang,
Cyril. "Transformers Learn Shortcuts to Automata."** arXiv:2210.10749, v1 2022-10-19,
v2 2023-05-02. DOI `10.48550/arXiv.2210.10749`. Affiliations as printed: CMU,
Microsoft Research NYC, University of Pennsylvania.

**Venue caveat.** The arXiv abs page has no `Comments:` and no `Journal ref:`
field, and the PDF carries no venue line. "ICLR 2023" is attested only by the
third-party DBLP record `conf/iclr/LiuAGKZ23`. Cite it as arXiv unless the
OpenReview record is read directly.

### 1.1 What the paper owns, and we concede entirely

The **solvable / non-solvable structure ceiling is theirs**, held centrally, with
a conditional lower bound we cannot improve on.

Section 6, Conclusions:

> "Our theoretical results show that shortcuts are ubiquitous, and characterize
> extremely shallow ones (with $L$ independent of the context length $T$) using
> algebraic machinery (Krohn-Rhodes theory)."

Theorem 4 ("Transformer Barrington"), main body §3.2, PDF p. 6:

> "Let $\mathcal{A}$ be a non-solvable semiautomaton. Then, for sufficiently
> large $T$, no $O(\log T)$-precision Transformer with depth independent of $T$
> and width polynomial in $T$ can continuously simulate $\mathcal{A}$ at length
> $T$, unless $\mathsf{TC}^0 = \mathsf{NC}^1$."

§3.2, PDF p. 6: *"The smallest example of a non-solvable semiautomaton has
$|Q| = 60$ states, whose transitions generate $A_5$ (all of the even
permutations)."*

**CEQ claims nothing on this axis.** Any sentence of ours about solvability,
Krohn–Rhodes decomposition, or constant-depth simulability of semiautomata is
their result and must cite them.

### 1.2 Two textual hazards for anyone citing this paper

**(a) Theorem 4 has two non-identical statements inside one paper.** Appendix C.5,
PDF p. 67:

> "Let $\mathcal{A}$ be a non-solvable semiautomaton. Then, for sufficiently
> large $T$, no fixed-precision Transformer with depth independent of $T$ and
> width polynomial in $T$ can simulate $\mathcal{A}$ at length $T$, unless
> $\mathsf{TC}^0 = \mathsf{NC}^1$."

The main body says `$O(\log T)$-precision` and `continuously simulate`; the
appendix says `fixed-precision` and `simulate`. Continuous simulation is their
defined notion (Appendix C.1: there exist $E, W$ such that $W \circ f \circ E$
simulates). **We cite the main-body statement and say which one we cite.**

**(b) Theorem numbering was disputed between two independent nurse reports.**
One numbered Barrington as Theorem 3, the other as Theorem 4. The second worked
from the compiled 68-page PDF and the ar5iv render rather than from raw LaTeX
with unresolved `\ref`, was internally consistent, and gives
T1 = log-depth simulation, T2 = Transformer Krohn–Rhodes, T3 = gridworld,
T4 = Barrington. **This file uses that numbering. The quoted text is identical in
both reports; only the number was in dispute. Re-verify against the compiled PDF
before publication.**

### 1.3 The "constant depth" claim is not in the theorem

**Theorem 2 (Transformer Krohn–Rhodes), formal statement, Appendix C.3, PDF p. 45**,
states depth **$O(|Q|^2 \log |Q|)$** — not $O(1)$. The phrase "$O(1)$-depth"
appears in the abstract and the §1 contribution bullet. "Constant depth" means
constant **in $T$**; the $|Q|$ dependence is severe: embedding dimension
$O(2^{|Q|}|\mathcal{T}(\mathcal{A})|)$, MLP width
$|Q|^{O(2^{|Q|})} + O(2^{|Q|}|Q||\mathcal{T}(\mathcal{A})|T)$.

And §3.1, PDF p. 5, on the construction:

> "It is non-constructive (much like how the existence of prime factorizations
> doesn't entail a procedure to *find* them). Computationally, these solutions
> still have to be found by a search procedure."

A non-constructive existence result with width $|Q|^{O(2^{|Q|})}$ is not a
competing capability number.

### 1.4 THE DELTA — what the near-miss did not measure

Exhaustive literal string search over two independent full-text copies (ar5iv
HTML render, 272,145 characters of extracted text; and the arXiv LaTeX e-print,
47 files including `arxiv.bbl`), appendices verified present in both.

| string | occurrences |
|---|---|
| `Hankel` | **0** |
| `nonnegative rank` / `non-negative rank` | **0** |
| `rank_+` / `rank_{+}` | **0** |
| `weighted automat` | **0** |
| `rational series` | **0** |
| `Fliess` | **0** |
| `Carlyle` | **0** |
| `Yannakakis` | **0** |
| `extension complexity` | **0** |
| `Myhill` / `Nerode` | **0** |
| `minimal state` / `state complexity` | **0** |
| `anytime` / `sequential test` | **0** |
| `confidence interval` / `error bar` / `bootstrap` | **0** |
| `p-value` / `hypothesis test` | **0** |
| `parameter-matched` / `same number of parameters` | **0** |
| `wall-clock` / `throughput` / `latency` / `training time` | **0** |
| `state-space` / `linear attention` / `S4` / `GRU` | **0** |

The string `rank` occurs **exactly once** in the entire paper, Appendix C, and
it is about non-invertible transitions collapsing the rank of transformations.
All 15 occurrences of the substring `flop` are "flip-flop", the two-state memory
unit; none is FLOP-as-cost.

**The paper has no Limitations section.** Its complete statistical methodology,
Appendix B, is:

> "Our primary goal is to understand if gradient-based training can find shortcut
> solutions at all, rather than whether such training is stable. Accordingly,
> unless otherwise noted, we report the performance of the *best* model among 20
> replicates."

plus mean/median over 20–25 replicates with standard-error shading. The headline
">99% in-distribution accuracy for *every* finite-state semiautomaton" is a
best-of-20 statement, and they say so.

**The max-versus-median spread is the substance of this delta.** From the figure
PDFs in the arXiv source, depths 1–16:

| task | Figure 8 (**max** over 20 runs) | Figure 9 (**median** over 20 runs) |
|---|---|---|
| $S_5$ @ depth 11 | `90.2` | `60.4` |
| $S_5$ @ depth 9 | `86.3` | `40.1` |
| $C_2 \times C_2 \times C_2$ @ depth 16 | `80.6` | `48.4` |

Architecture configurations are openly unmatched (Appendix B.3): *"For GPT-2
models, we fix the embedding dimension and MLP width to 512 and the number of
heads to 8 ... For LSTM, we fix the embedding dimension to 64, the hidden
dimension to 128, and the number of layers to 1."*

### 1.5 The delta stated, on four axes

1. **Question.** They ask which automata a transformer *can represent* and
   whether SGD finds it. We ask whether one attention operator *beats another at
   matched parameters* under anytime-valid statistics. Our harness holds
   `n_params=4769` exactly across arms as a structural property; they do not
   match parameters anywhere.
2. **Difficulty axis.** They grade tasks by **Krohn–Rhodes solvability**, a
   group-theoretic axis. The Hankel program grades by **rank vs nonnegative
   rank**, a rank/semiring axis absent from their paper by literal count. These
   are not the same axis and do not order tasks the same way: see §2.3, where a
   task that is trivially solvable on their axis is the one our axis was
   supposed to call hard.
3. **Comparison.** They compare **depth against sequence length $T$**. We compare
   **operator against operator at fixed depth**.
4. **Statistics.** They report best-of-20 with no interval, no test, and no
   sequential method. The e-process in `METHODS.md` is the whole difference.

**The overlap is Krohn–Rhodes-as-ceiling, and we concede it wholesale.**

### 1.6 What the near-miss hands us, unprompted

Their *measured* failures of shortcut solutions are all on terrain a settling arm
occupies, and an LSTM beats the transformer on every one:

- **Out-of-distribution under input-distribution shift**, Appendix B.2.3: *"The
  performance of the Transformer degrades sharply as the test distribution
  changes away from training, failing at out-of-distribution generalization."*
  … *"In contrast, an LSTM recurrent network maintains perfect accuracy when
  evaluated on all values of $\Pr[\sigma=1]$."*
- **Length generalization**, §5.2: *"In contrast to LSTM's perfect performance on
  all scenarios, Transformer's accuracy drops sharply as we move to lengths
  unseen during training."*
- **Incomplete supervision**, §5.1: *"Transformers may be unable to find good
  solutions when the labels become sparser, whereas LSTM's performance stays
  robust across all choices of $p_{\mathrm{reveal}}$."*

And their own mitigation costs the win, Appendix B.2.3: *"Notice that this
mitigation completely foregoes the computational advantage of a shallow
shortcut."*

**This is a task-selection note for Cameron, not a claim and not a certificate
proposal.** It says where a bounded-depth recurrent-flavoured operator would be
measured against a real published weakness rather than against nothing.

---

## 2. THE HANKEL PROGRAM'S CITATIONS

### 2.1 Fliess / Carlyle–Paz — **it requires a FIELD**

The contract asked whether the theorem needs a field or holds over any
commutative ring, because L4 depends on the answer. **It needs a field, or a
division ring. It does not hold over an arbitrary commutative ring.**

Berstel & Reutenauer, *Noncommutative Rational Series with Applications*
(quotations from the author-hosted draft of 9 April 2010; chapter and theorem
numbers match CUP 2011, printed pagination may not):

- Ch. 2 §1, p. 27, opening line: *"We start by assuming that K is a commutative
  ring."*
- Ch. 2 §1, p. 30, marginal 643, **immediately before** the definitions of rank
  and of the Hankel matrix: *"We suppose from now on that K is a field."*
- **Theorem 1.6 (Carlyle and Paz 1971, Fliess 1974a), p. 31** — carries no
  hypothesis of its own and inherits the field: *"The rank of a formal series S
  is equal to the codimension of its syntactic right ideal, and is equal to the
  rank of its Hankel matrix."*

**The hypothesis switches exactly at the definition of rank.** That is the whole
answer.

Independent confirmation — Balle & Mohri, "Learning Weighted Automata", CAI 2015
(LNCS 9270), §3 preamble: *"From here on, we will assume that the semiring S is
in fact a field. This enables us to define the rank of a matrix with entries in
S."* And their Theorem 1 (Fliess), §3.2:

> "Let S be a field. Then, the rank of the Hankel matrix Hf associated to a
> function f : Σ* → S is finite if and only if f is rational. In that case, there
> exists a WFA A representing f with rank(Hf) states and no WFA representing f
> admits fewer states."

**How far it does generalise: sideways, to division rings — not upward to rings.**
Sakarovitch, "Rational and recognisable power series", Handbook of Weighted
Automata Ch. 4 (author's draft, 19-Sep-2008), §5.2 p. 50: *"We now suppose that K
is a field, not necessarily commutative, hence a skew field, or division ring."*
Theorem 15, p. 50: *"A series s over A* with coefficients in a division ring is
recognisable if and only its rank is finite."* (sic, "if and only its").

**Over semirings, rank is replaced rather than weakened.** Berstel & Reutenauer
Ch. 1 Prop. 5.1 p. 11 uses a *"stable finitely generated left K-submodule"*
instead, and the Notes to Ch. 1, p. 25, give the provenance: *"taken from Jacob
(1975) who extends to semirings a Hankel-like property given by Fliess (1974a)
for fields."* Labai & Makowsky, arXiv:1512.02430, §4: *"notions of orthogonality,
rank, and norms do not readily transfer to the semiring setting."* And the
concrete failure, Berstel & Reutenauer Exercise 5.5, p. 24: the series
$S = \sum_{n\ge 0} n a^n$ is recognizable over $\mathbb{N}$, but the smallest
stable $\mathbb{N}$-submodule containing it is **not finitely generated**.

**The principal-ideal case, and its open problem.** Berstel & Reutenauer Ch. 7
§1, p. 121, under the hypothesis *"Let K be a commutative principal ring and let
F be its quotient field"* — Theorem 1.1 (Fliess 1974a): *"Let S ∈ K⟨⟨A⟩⟩ be a
series which is rational of rank n over F. Then S is rational over K and has a
linear representation over K of dimension n."* **The rank is taken over the
quotient field F and then realised over K.** The load-bearing proof step, p. 122,
is *"μ(K⟨A⟩) is a submodule of a free K-module of rank n, hence is also free and
has rank ≤ n"* — submodule-of-free-is-free, which needs a PID and fails over a
general commutative ring. Whether the Hankel-rank statement itself has a PID
form is open: van Heerdt, Kupke, Rot & Silva, "Learning Weighted Automata over
Principal Ideal Domains", FoSSaCS 2020, arXiv:1911.04404v2, p. 16: *"The second
part of Lemma 23 would follow from a PID variant of Fliess' theorem [12]. We are
not aware of such a result, and leave this for future work."*

**NOT FOUND:** a displayed counterexample of the form "rank $(H_f) = r$ over a
ring $K$, but every $K$-weighted automaton for $f$ needs more than $r$ states".
The nearest items are Berstel & Reutenauer Exercise 4.3 p. 132 (asserts a
non-weak-Fatou ring, left as an exercise with a hint) and Chabert's Theorem 1.5
p. 123 (characterises the failure class rather than exhibiting an automaton gap).

**NOT REACHED:** Fliess (1974a), "Matrices de Hankel", *J. Math. Pures Appl.*
53:197–222 — primary text not retrieved. Every statement attributed to Fliess
above is a secondary citation.

#### CONTRACT CORRECTION 1 — L4

Contract §1.6 sets L4 as *"the Hankel worked example, `rank(H_f) = 2` over any
`CommRing`, with the nonnegative-impossibility on raw entries as its negative
control."* **Both halves are wrong.** Rank is not well-defined over a general
commutative ring (modules need not be free), and Fliess needs a field or a
division ring. The negative control is vacuous — see §2.3.

**Write instead**, and it is a better Lean target because it is constructive:

- **L4a (ring-general, no rank notion required).** $H_f$ **factors through**
  $R^2$ over any `CommRing`: $H = \begin{bmatrix} f & \mathbf{1}\end{bmatrix}
  \cdot \begin{bmatrix} \mathbf{1} & f \end{bmatrix}^{T}$, i.e.
  $H[u,v] = f(u)\cdot 1 + 1\cdot f(v)$. True over any `CommRing`, including
  $\mathbb{N}$.
- **L4b (minimality, separately, with the field hypothesis).** Over a field, 2 is
  minimal, citing Berstel & Reutenauer Theorem 1.6.

Splitting them is not a weakening. The existence half is ring-general and the
minimality half is exactly what the field buys, so the split is the theorem's
actual shape.

### 2.2 Yannakakis, and the nonnegative-rank machinery

**Yannakakis, Mihalis. "Expressing Combinatorial Optimization Problems by Linear
Programs."** *Journal of Computer and System Sciences* **43**(3), 441–466, 1991.
DOI `10.1016/0022-0000(91)90024-Y`. Received December 28, 1988; revised April 1,
1990. Conference version: STOC '88, pp. 223–228, DOI `10.1145/62212.62232`
(**not accessed**; no quotation offered from it).

**Theorem 3, p. 457, verbatim:**

> "Let m be the smallest number such that SM can be written as the product of two
> nonnegative matrices of dimensions f × m and m × v. The minimum of the number
> of variables plus number of constraints over all LP's expressing P is
> Θ(m + n)."

(The bound is capital **Θ**. A raw `pdftotext` layer renders it as "O(m+n)"; that
is an extraction artifact.)

#### CONTRACT CORRECTION 2 — attribution

**Yannakakis 1991 does not contain the string "nonnegative rank", does not
contain "extension complexity", and does not state $xc(P) = \mathrm{rank}_+(S)$.**
He calls the quantity **positive rank** and his statement is the Θ(m+n) above.
Contract §1.1's *"super-polynomial separations known (Yannakakis /
extension-complexity)"* misattributes on both counts, because Yannakakis states
neither the equality nor a rank-versus-rank₊ separation.

Yannakakis, p. 458, naming it and immediately disclaiming any method:

> "Let us call the smallest number *m* of the theorem, the *positive rank* of the
> matrix *SM*. We do not know of any techniques for estimating or deriving bounds
> for the positive rank of a matrix."

and his open-problem list, p. 465, item (1): *"Find techniques for computing or
bounding the positive rank of a matrix."*

**The crisp equality is later and belongs to others.** Fiorini, Kaibel,
Pashkovich & Theis, arXiv:1111.0444v2, §2.3 p. 7, Theorem 2.6, with its own
qualifying sentence on the same page: *"he proved that extension complexity and
nonnegative rank are within a factor of two of each other, when the size of an
extension is defined as the sum of the number of variables and number of
constraints defining the extension."* And Fiorini, Massar, Pokutta, Tiwary & de
Wolf, arXiv:1111.0837v5, §3.1 p. 8: *"It can be stated succinctly as:
xc(P) = rank₊(S) whenever P is a polytope and S a slack matrix of P."*

**The rank-versus-rank₊ separation has a different owner again.** Kwan, Sauermann
& Zhao, *Trans. AMS* 375(6):4209–4250, 2022, arXiv:2006.08836v3, §3 p. 6,
Theorem 3.1:

> "For every n ∈ ℕ, there is a nonnegative n × n matrix M satisfying
> rank₊ M/ rank M = n^{1−o(1)}."

Answering *"a question of Hrubeš"*. **NOT FOUND:** a sentence explicitly phrased
as a super-polynomial separation *between rank and nonnegative rank* in
Yannakakis 1991, in Fiorini et al. arXiv:1111.0837v5, or in Rothvoß
arXiv:1311.2369v3 — those state lower bounds on rank₊ / extension complexity, and
the one thing FMPTdW label an "exponential separation" is nonnegative rank versus
**PSD** rank (Corollary 19, p. 18).

The exponential extension-complexity bounds, for completeness: FMPTdW Theorem 12
p. 15, TSP polytope $2^{\Omega(n^{1/2})}$; Rothvoß arXiv:1311.2369v3 Theorem 1
p. 3, perfect matching polytope $2^{\Omega(n)}$.

### 2.3 Cohen–Rothblum, hardness, and **the decrement wall's death**

**Cohen, Joel E. & Rothblum, Uriel G. "Nonnegative ranks, decompositions, and
factorizations of nonnegative matrices."** *Linear Algebra and its Applications*
**190**, 149–168, 1993. DOI `10.1016/0024-3795(93)90224-C`.

Definition, p. 152: *"we define the nonnegative rank of A, denoted rank_+(A), as
the integer q for which the four equivalent conditions of the above corollary
apply"*, the conditions being the factorization $A = VU$ with $V, U$ nonnegative
of inner dimension $q$, and the sum-of-$q$-nonnegative-rank-one form.

**Lemma 2.3 (H. Robbins), p. 152:** *"Let A be a nonnegative matrix in G^{m×n}.
Then rank(A) ⩽ rank_+(A) ⩽ min(m, n)."* — this is the `rank₊ ≥ rank` the contract
wanted, and it lives here, not in Yannakakis.

**Lemma 2.4, p. 152**, which is the usable lower bound: *"Let A ∈ G^{m×n} be
nonnegative. If A contains a set of q pairwise independent entries, then
rank_+(A) ⩾ q."* — with independence defined on the same page as
$A_i^j A_k^p > 0$ and $A_i^p A_k^j = 0$. **This is a fooling-set bound and it is
cheap to compute.**

**THE TWO THEOREMS THAT KILL THE CONTRACT'S WORKED EXAMPLE**, pp. 157–158:

> "THEOREM 4.1. Let A ∈ G^{m×n} be a nonnegative matrix with rank(A) ⩽ 2. Then
> rank_+ (A) = rank(A)."

> "COROLLARY 4.2. Let A ∈ G^{m×n} be a nonnegative matrix. If either
> m ∈ {1, 2, 3} or n ∈ {1, 2, 3}, then rank_+(A) = rank(A)."

**Gap results in this paper are thin:** exactly one 4×4 example (H. Robbins,
p. 153) with rank 3 and rank₊ 4. An infinite family, or any quantitative bound on
the size of the gap: **NOT FOUND** in Cohen–Rothblum.

**Cohen–Rothblum does NOT pose the complexity of computing rank₊ as an open
problem.** Its one stated open problem, p. 163, is about the ground field:
*"Show that the nonnegative ranks of a rational matrix over the reals and over
the rationals coincide, or provide an example where the two ranks are
different."* What it gives on complexity is a **positive** result — a finite
(super-exponential) algorithm via quantifier elimination, Theorem 5.1, pp. 161–162.

**The NP-hardness is Vavasis, and it is prose, not a numbered theorem.**
Stephen A. Vavasis, "On the Complexity of Nonnegative Matrix Factorization",
*SIAM J. Optimization* **20**(3), 1364–1377, 2009, DOI `10.1137/070709967`
(SIAM text paywalled, **abstract only**); quotations below from arXiv:0708.4149v2.
The numbered result is about a geometric problem:

> "**Theorem 4.** The instance of 3-SAT is a yes-instance if and only if the above
> instance of INTERMEDIATE SIMPLEX is a yes-instance."

reduced from 3-SAT with polynomially-bounded integers, hence *"strong"* NP-hardness
(p. 7). The transfer to nonnegative rank is prose, p. 2: *"Since nonnegative rank
determination is a generalization of EXACT NMF, our result shows that it is also
NP-hard."* **A numbered theorem reading "computing the nonnegative rank is
NP-hard" was NOT FOUND in arXiv:0708.4149v2.**

Refinement — Arora, Ge, Kannan & Moitra, STOC '12, arXiv:1111.0952v1, Theorem 3.19
p. 10: deciding whether an $n\times m$ nonnegative matrix has nonnegative rank
$r$ runs in $O((nm)^{O(r^2 2^r)})$ — **polynomial for fixed $r$** — and
Theorem 1.5 p. 3 rules out $O((nm)^{o(r)})$ under the Exponential Time Hypothesis.

#### CONTRACT CORRECTION 3 — the decrement wall has NO GAP

Contract §1.1 states, for $f(w) = (\#a) - (\#b)$ on $\{a,b\}^*$:

> "the shifted comparison on a length-`n` block gives `rank_ℝ = 3` while the
> nonnegative side must track `n` distinct count-levels — **a gap growing with
> `n`. Rank 2–3 vs Ω(n).**"

**Measured, independently, before Cameron's instrument existed** (numpy SVD,
tolerance `1e-9 * s[0]`, `torch.set_num_threads(2)`):

| $n$ | $|W|$ | rank $H$ | shift $c$ | rank $H + cJ$ | distinct levels of $f(uv)$ |
|---|---|---|---|---|---|
| 3 | 15 | **2** | 6 | **2** | 13 |
| 4 | 31 | **2** | 8 | **2** | 17 |
| 5 | 63 | **2** | 10 | **2** | 21 |
| 6 | 127 | **2** | 12 | **2** | 25 |

**The shifted rank is 2, not 3.** Because $H = f\mathbf{1}^T + \mathbf{1}f^T$ and
$H + cJ = f\mathbf{1}^T + \mathbf{1}(f + c\mathbf{1})^T$ — the shift **folds into
the second rank-one term** rather than adding a third, since $\mathbf{1}f^T$ and
$\mathbf{1}\mathbf{1}^T$ share the column space $\mathrm{span}\{\mathbf{1}\}$.

**And rank₊ = 2 as well, so the gap is exactly zero.** Two independent proofs:

1. **By theorem.** The shifted matrix is nonnegative with rank 2, so
   Cohen–Rothblum Theorem 4.1 gives $\mathrm{rank}_+ = \mathrm{rank} = 2$.
2. **By construction.** Every column of $H + cJ$ is $f + t_v\mathbf{1}$ with
   $t = f + c$. Taking $a = \min t$, $b = \max t$,
   $U = [\,f + a\mathbf{1},\; f + b\mathbf{1}\,]$ and
   $V = \big[\tfrac{b-t}{b-a};\ \tfrac{t-a}{b-a}\big]$ gives $H + cJ = UV$ at
   inner dimension 2. Measured: `U.min=0.000000`, `V.min=0.000000` (both
   nonnegative), $\max|UV - H_s|$ = `1.776e-15`, `0.000e+00`, `1.776e-15`,
   `3.553e-15`, `3.553e-15` at $n = 3,4,5,6,7$.

**Where the §1.1 argument went wrong**, named exactly: it conflated *"the
function takes $n$ distinct values"* with *"rank₊ is $\Omega(n)$"*. These are
unrelated — a rank-2 **nonnegative** matrix takes arbitrarily many distinct
values, and here it does: 13/17/21/25 distinct levels alongside rank₊ = 2. The
other half, *"$H_f$ takes negative values, so it admits no nonnegative
factorisation of any size on raw entries"*, is true but vacuous: a matrix with a
negative entry is not a nonnegative matrix, so rank₊ is not defined on it.
Neither half survives as a gap.

**Consequence for K-5.** K-5 reads *"rank/rank₊ machinery disagrees with the
worked example ⇒ the instrument is broken."* On this evidence the instrument is
fine and **the worked example is broken**. Note also that K-5 is unfireable as
written if one agent produces both the instrument and the reproduction of the
worked example, since it will reproduce whatever it computes; the check above was
run independently from the contract text alone, which is what made the
disagreement visible.

**Replacement route (RULE 5), three shapes:**

- **REPRICE — free, and it ships immediately.** Cohen–Rothblum Theorem 4.1 and
  Corollary 4.2 give a **necessary condition for any gap task, computable before
  any training: $\mathrm{rank}_\mathbb{R}(\hat H) \ge 3$, or the task provably
  has no gap.** One SVD per candidate. The instrument survives intact; only its
  worked example dies, and the cheap half of the instrument now screens
  candidates for free. What the corpse taught: exact $\mathrm{rank}_\mathbb{R}$
  already decides the gap question in the small-rank regime, so the expensive
  half (rank₊ lower bounds via Lemma 2.4's fooling sets) is only ever needed at
  rank ≥ 3.
- **REROUTE — needs a leap, named rather than guessed.** Real separations exist
  and are cited in §2.2 (Kwan–Sauermann–Zhao $n^{1-o(1)}$; the slack-matrix
  families at $2^{\Omega(n)}$). **None is known to be the Hankel matrix of a
  formal series on a finite alphabet**, and a Hankel matrix is heavily
  constrained. Making a separated family Hankel is the leap. **Dr House's
  trigger, not a Foreman guess.**
- **RETIRE — and this is the sharp one.** The prediction ladder needs the **arms**
  to differ in signedness, not only the **tasks**. The repo already measured that
  they do not: `CHECKLIST.md:333` records *"**G4 VOID [RUN, r3 iter 19]** — at the
  harness geometry (logits |w| mean 2.68e-03) `_causal_sgate_operator(lam=0.10)`
  is **ENTRYWISE NON-NEGATIVE**, min entry exactly `0.000e+00`. The paired test
  compared two non-negative operators; `pivot_signed` was `pivot_unsigned`
  wearing a name."* Building a signed-versus-nonnegative task corpus while both
  arms are nonnegative at the measured geometry tests nothing, whatever the
  tasks' Hankel ranks say. **Replacement goal, a precondition on the whole
  ladder: measure the sign structure of every arm operator at the geometry it
  will be measured at, and exhibit a negative entry, before one gap task is
  built.** What the corpse taught: "signed" is a claim about the **operator**, and
  the project designed tasks around a property of the arms that had already been
  voided.

---

## 3. Status of the fetch batch

| item | status |
|---|---|
| Near-miss delta (§1) | **CLOSED** — full text reached twice, string search exhaustive |
| Fliess / Carlyle–Paz, field vs ring (§2.1) | **CLOSED** — field or division ring; primary Fliess 1974a NOT REACHED, secondary only |
| Yannakakis 1991 (§2.2) | **CLOSED** — with attribution correction |
| Cohen–Rothblum + NP-hardness (§2.3) | **CLOSED** — with two corrections |
| Robbins–Siegmund, Doob, Ville | **CLOSED in `METHODS.md` §6** — Ville 1939 and Durrett V5 Thm 4.2.12 reached in full; Robbins–Siegmund 1971 original NOT REACHED, restatement used and labelled |

### 3.1 Bibliographic data for the primary Hankel sources

Verified against the zbMATH Open API, which was the reachable authority.

- **Fliess, Michel. "Matrices de Hankel."** *Journal de Mathématiques Pures et
  Appliquées, Neuvième Série*, **volume 53** (1974), pages **197–222**.
  ISSN 0021-7824. Elsevier (Masson), Paris. zbMATH id 3494308, Zbl 0315.94051.
  **Language: French.** **Erratum:** Fliess, M., "Erratum: Matrices de Hankel",
  *J. Math. Pures Appl. (9)* **54** (1976), page **481** — a separate zbMATH
  record, and the erratum is flagged in Beimel et al.'s bibliography.
  **PRIMARY TEXT NOT REACHED** (JMPA vol. 53 is not open access; zbMATH `links`
  array empty). The only internal locator obtained is second-hand: STACS 2020
  cites it as *"[9, Th. 2.1.1]"*, unverified against the paper.
  **Page-range discrepancy, reported rather than resolved:** 197–222 in zbMATH,
  Berstel–Reutenauer, Beimel et al., Rabusseau et al., Thon & Jaeger, Labai &
  Makowsky and Fijalkow et al.; **197–224** in Duchamp et al.
  (arXiv:math/0607412 ref [6]); and one outlier printing the volume as **5**
  and the title as singular "Matrice" (Lacroce, Panangaden, Rabusseau,
  arXiv:2206.00172).
- **Carlyle, J. W. & Paz, A. "Realizations by stochastic finite automata."**
  *Journal of Computer and System Sciences*, **volume 5**, issue 1 (1971),
  pages **26–40**. DOI `10.1016/S0022-0000(71)80005-3`. Zbl 0236.94042.
  **PRIMARY TEXT NOT REACHED** — ScienceDirect returned HTTP 403; Unpaywall
  reports `is_oa: false` with zero OA locations; Semantic Scholar's record notes
  the abstract was elided by the publisher. **Not even ABSTRACT ONLY was
  obtainable.**

**Every statement attributed to Fliess or to Carlyle–Paz in this file is a
secondary citation.** That is a real limitation on §2.1's verdict, and it is
stated here rather than buried: the *field* hypothesis is attested by four
independent secondary sources that agree (Berstel–Reutenauer p. 30, Balle–Mohri
Theorem 1, Beimel et al. Theorem 2.4, Sakarovitch §5.2), and by **zero** sources
stating it over a commutative ring — but not by the originals.

Berstel & Reutenauer's own convention matters for reading their theorem, Ch. 1
§1 p. 4: *"In this text, a field is always commutative."* So their Theorem 1.6
is the commutative-field case, and Sakarovitch's division-ring version is the
strictly more general one.

Attribution of the notion itself, Berstel & Reutenauer, Notes to Chapter 2,
p. 42: *"The notions of Hankel matrix and rank of a formal series, which are
classical in the case of one variable, were introduced by Carlyle and Paz (1971)
and Fliess (1974a)."*

---

## 4. THE ROUND-8 DELTA — *equilibrium labels as an attention capability bar*

Fetched before any round-8 build, per RULE 7. Eight nurses, four of them on the
literature, twenty-plus distinct searches. **Absence below is recorded as NOT FOUND,
never as unoccupied.**

### 4.1 The claim under test

`LOOP_PROMPT.md` §2 states the delta as
**`equilibrium-labels-as-attention-capability-bar`, NOT GNN regression** — the point
being that the absorbing-chain solve is used to *measure what an architecture can
represent*, not to *compute the solution faster*.

### 4.2 THE VERDICT — the delta as stated does NOT survive

Every one of its three parts is separately occupied, and by named work.

**Equilibrium-shaped labels used to measure architecture capability.**
*The CLRS Algorithmic Reasoning Benchmark*, Veličković, Badia, Budden, Pascanu,
Banino, Dashevskiy, Hadsell, Blundell, ICML 2022, **arXiv:2205.15659**, carries
Bellman–Ford, Floyd–Warshall and Kosaraju SCC — all fixed points reached by
iterating — and frames itself as *"evaluating algorithmic reasoning learnt by neural
network models"*. Its language-model form is *The CLRS-Text Algorithmic Reasoning
Language Benchmark*, **arXiv:2406.04229**. **PageRank is NOT FOUND in CLRS-30.**

**The same, for attention specifically rather than message passing.**
*Understanding Transformer Reasoning Capabilities via Graph Algorithms*, Sanford,
Fatemi, Hall, Tsitsulin, Kazemi, Halcrow, Perozzi, Mirrokni, NeurIPS 2024, OpenReview
`AfzbDw6DSp`, states *"logarithmic depth is necessary and sufficient"* for graph
connectivity and gives a *"representational hierarchy that separates 9 algorithmic
reasoning problems into classes"*. See also Sanford, Hsu, Telgarsky, *Transformers,
parallel computation, and logarithmic depth*, ICML 2024, **arXiv:2402.09268**.

**Equilibrium-solving architectures evaluated against those labels.**
*The Deep Equilibrium Algorithmic Reasoner*, Georgiev, Liò, Buffelli,
**arXiv:2402.06445**, and *Deep Equilibrium Algorithmic Reasoning*, Georgiev, Wilson,
Buffelli, Liò, NeurIPS 2024, **arXiv:2410.15059** — the latter *"requires no
information on the ground-truth number of steps of the algorithm, both during train
and test time"*, evaluated on roughly ten CLRS-30 tasks.

**The Dirichlet half is occupied too.** *Learning Label Initialization for
Time-Dependent Harmonic Extension*, Azad, IJCAI 2022, **arXiv:2205.01358** — *"Node
classification on graphs can be formulated as the Dirichlet problem on graphs"*.
*Inverse Boundary Value and Optimal Control Problems on Graphs*, Garrousian,
Nouranizadeh, **arXiv:2206.02911** — *"system identification problems on graphs with
Dirichlet and Neumann boundary conditions"*. *Diffusion-Jump GNNs*, Begga, Escolano,
Lozano, Hancock, **arXiv:2306.16976** — *"is formulated as a Dirichlet problem"*, and
it names *"absorbing random walks"* explicitly. Solver-side: *Learning the Solution
Operator of Boundary Value Problems using Graph Neural Networks*, Lötzsch, Ohler,
Otterbach, AI4Science @ ICML 2022, **arXiv:2206.14092**.

**Even the "architecture provably cannot" framing on a linear-algebra target is
occupied.** *Message-Passing GNNs Fail to Approximate Sparse Triangular
Factorizations*, Trifonov, Muravleva, Oseledets, TMLR 2026, **arXiv:2502.01397**:
*"message-passing GNNs are fundamentally incapable of approximating sparse triangular
factorizations"*. And *Affinity-Aware Graph Networks*, Velingker, Sinop, Ktena,
Veličković, Gollapudi, NeurIPS 2023, **arXiv:2206.11941**, Theorem C.1: *"it is
impossible for a GNN to compute single-source effective resistances"* — effective
resistance being a Dirichlet problem with two boundary nodes, which is exactly the
oracle proposed here.

**The absorption function itself is a studied regression target**, under the name
*committor*: Khoo, Lu, Ying, **arXiv:1802.10275**; Li, Lin, Ren, *J. Chem. Phys.*,
**arXiv:1906.06285**; Contreras Arredondo et al., *Nature Computational Science*, DOI
`10.1038/s43588-026-00958-2`, **arXiv:2507.17700**, which uses *"a graph-neural-network
architecture built on geometric vector perceptrons to predict the committor function"*.

### 4.3 What was NOT FOUND, stated as not found

* **`B = N R` with `N = (I − Q)^{-1}` as a supervised regression label for a neural
  network — NOT FOUND.** The committor papers above regress the same mathematical
  object from molecular configurations, not from the fundamental matrix of a graph
  chain.
* **A GNN expressivity or receptive-field study whose label class is absorption
  probability — NOT FOUND.**
* **A benchmark measuring transformer capability against PageRank or linear-system
  fixed-point ground truth — NOT FOUND.**
* **The spectral gap, or `t_rel = 1/(1 − λ₂)`, used as a DIFFICULTY DIAL that sets how
  hard a learning benchmark is — NOT FOUND.** Every retrieved use of the spectral gap
  is diagnostic (oversquashing: Topping et al. **arXiv:2111.14522**, Cheeger constant
  `h_G`; Di Giovanni et al. **arXiv:2302.02941**, commute time; Black et al.
  **arXiv:2302.06835**, total effective resistance; Karhadkar, Banerjee, Montúfar
  **arXiv:2210.11790**, spectral gap of the normalized Laplacian) or prescriptive for
  rewiring — never a knob turned to set task difficulty.
* **Synthetic graphs built with a PRESCRIBED spectral gap to produce a dose-response
  curve in depth or iteration count — NOT FOUND.** The Long Range Graph Benchmark,
  Dwivedi et al., NeurIPS 2022 D&B, **arXiv:2206.08164**, establishes its long-range
  property with *shortest-path length and diameter*; a full-text search of it for
  "spectral gap", "Cheeger" and "eigenvalue" returned **no matches**.
* **The phrases "equilibrium labels as a capability bar" and "next-equilibrium
  prediction", in any machine-learning sense — NOT FOUND.**

### 4.4 The surviving delta is NARROW, and it is CONTESTED

What survives the fetch is not the framing but one mechanism: **a benchmark whose
difficulty is set by engineering a spectral quantity of the substrate, so that the
shape of the error curve across a truncation ladder is predicted in advance rather
than described afterwards.**

Two papers sit directly on that, and neither is a comfortable distance away.

* Fesser, Weber, *Performance Heterogeneity in Graph Neural Networks*,
  **arXiv:2503.00547**: *"we propose to use 1/λ2∗ as a heuristic for the GNN depth"*,
  and *"the ideal GCN depth turned out to be the integer closest to 1/λ2∗"*. This
  predicts a depth from an independently measured spectral quantity, which is the same
  move, one step short of a curve.
* Veerabhadraswamy, Emerson, *Spectral Flow Certificates for Depth-Aware Long-Range
  Propagation in Graph Neural Networks*, **arXiv:2607.21607**:
  `SFC(G,k) = 1 − (1 − γ(G))^k` with `γ(G) = λ₂(L_norm)`, reporting
  `R² = 0.910, 0.881, 0.863` at depths `k = 2, 3, 4`. That is a quantitative
  prediction of accuracy from an independently measured spectral quantity, at several
  depths — i.e. a dose-response curve in depth predicted by `λ₂`. **This is the closest
  occupant found and it may occupy the surviving delta outright.** Provenance caveat,
  recorded as observed and not resolved: its abstract page reports *"Submitted on 16
  May 2026"* against a July-2026 identifier prefix.

What is left after those two is thin and must be stated as thin: the same construction
for **attention rather than message passing**, on a substrate whose spectral quantity
is **engineered rather than surveyed**, against a label that is **itself the
equilibrium** rather than a downstream accuracy. It is a delta. It is not the delta
`LOOP_PROMPT.md` §2 claimed, and no round-8 number should be written as though it were.

### 4.5 A second correction, from the same fetch batch

`LOOP_PROMPT.md` §1.2 states that *"Cheeger's inequality bounds `λ₂` by the conductance
from both sides, so the engineering target is reachable by construction rather than by
search"*. The primary source is Levin, Peres, Wilmer, *Markov Chains and Mixing Times*,
2nd ed., **Theorem 13.10**, attributed there to Sinclair & Jerrum (1989) and Lawler &
Sokal (1988):

> `Φ⋆² / 2 ≤ γ ≤ 2Φ⋆`

with `γ = 1 − λ₂` and `λ₂` the second largest eigenvalue of a **reversible** transition
matrix, and `Φ⋆` the **minimum** conductance over all cuts with `π(S) ≤ 1/2`. Two things
follow that the contract sentence does not allow for, and `scale/foreman_lambda2.py`
measures both:

1. The theorem is about the **ergodic** chain. The quantity a `t*` ladder truncates is
   the Perron root of the **transient block** of the absorbing chain, which is a
   different number — measured at `0.9964078857` against `0.9984623637` on
   `LargestJoin_S2Rips_1024`.
2. A single named cut gives an **upper** bound on `Φ⋆`, and an upper bound on `Φ⋆`
   composes only with `γ ≤ 2Φ⋆`. The lower half needs `Φ⋆` itself, a minimum over
   `2^(n−1)` cuts. So even for the ergodic chain, one cut buys "not too fast" and never
   "not too slow".
