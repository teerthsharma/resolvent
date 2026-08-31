# V15 / n1 — COMPONENT PRIOR ART AT EQUATION LEVEL

JUPITER, node `n1` of the CEQ v15 composition round. Discharges `CEQ_V15_CONTRACT.md`
PART V (`n1` all component prior-art fetches at equation level) against the standing
law **L-EQ** (contract lines 51-55): `[V]` is inadmissible for a load-bearing statement;
`[V-eq]` requires the statement WITH HYPOTHESES plus one numeric instance run.

## Admissibility legend

| tag | meaning |
|---|---|
| `[V-eq]` | defining equation transcribed from the source, hypotheses stated as the source states them, and a numeric instance under `scripts/v15_n1_probes/` executed with its real output pasted below |
| `[V]` | page reached, equation NOT verified — **INADMISSIBLE for any load-bearing statement**, flagged inline |

Every numeric instance carries a **control**: a deliberately mis-transcribed variant of
the same equation whose output differs by O(1). A probe that only prints `~1e-16` proves
nothing about transcription; a probe that prints `~1e-16` for the equation and `~1e0`
for the off-by-one variant does.

Reproduce: `python scripts/v15_n1_probes/pNN_*.py` (numpy 1.26.4, scipy 1.17.1, CPython 3.11,
Windows 11, this box). Outputs below are pasted verbatim from those runs.

**Scoreboard for this node: 10 of 10 components reach `[V-eq]`.** Four sub-statements
inside those components remain `[V]` and are named inline.

---

## 1. SSD / Mamba-2 — the structured state-space duality and its decay mask `[V-eq]`

**Citation.** Tri Dao, Albert Gu, *Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality*, ICML 2024, arXiv:2405.21060.
Fetched: `arxiv.org/html/2405.21060v1`.

**Equations, as transcribed.**

- Definition 3.1 (semiseparable): a lower-triangular matrix `M` is `N`-semiseparable if
  every submatrix contained in the lower-triangular portion has rank at most `N`.
- Definition 3.2 / Eq (3) (sequentially semiseparable, SSS): the SSM
  `y = SSM(A,B,C)(x)` is the matrix transformation `y = M x` with

  ```
  M_{ji} = C_j^T A_j A_{j-1} ... A_{i+1} B_i  =  C_j^T A_{j:i}^x B_i
  ```

  where `A_{j:i}^x := A_j x A_{j-1} x ... x A_{i+1}`.
- Eq (6), scalar-identity SSD case `A_t = a_t I`: `M` collapses to the **1-semiseparable**
  matrix, the scalar recurrence `y_t = a_t y_{t-1} + x_t`.
- §2.4, the dual ("attention-like", quadratic) form:

  ```
  (L o Q K^T) V,   L_{ij} = a_i x ... x a_{j+1}  if i >= j,   0 if i < j
  ```

**Hypotheses as the paper states them.** State size `N ∈ N`; sequence length `T`; head
dimension `P ≥ 1`; `B_i, C_i ∈ R^N`, `A_i ∈ R^{N×N}`; causal / lower-triangular structure
(`j ≥ i`); for the SSD case `A_t = a_t I` with **input-dependent scalars `a_i ∈ [0,1]`**;
and — load-bearing — **no softmax normalization** in the dual form.

**Numeric instance.** `scripts/v15_n1_probes/p01_ssd_mamba2.py`, T=7, N=3, P=2.

```
scan vs M_ji form        : 8.881784197001252e-16
scan vs dual (L o CB^T)  : 8.881784197001252e-16
off-by-one control       : 1.6401285245414234
a_t == 1 -> L is all-ones causal: True
```

**What this occupies.** The whole of contract §S-M's carrier. `W_ij = exp(C_i - C_j)` with
`C = scan(g)`, `g = log a ≤ 0` is *definitionally* the 1-semiseparable mask `L` of Eq (6),
written in log space; the "ONE unnormalized causal hop on values `V(x)`" is Dao & Gu's
quadratic dual form. `g == 0 → causal all-ones mask` is their `a_t = 1` case, printed
`True` above. Nothing in S-M's carrier is new relative to this paper.

---

## 2. GLA — the gating form `[V-eq]`

**Citation.** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim,
*Gated Linear Attention Transformers with Hardware-Efficient Training*, ICML 2024,
arXiv:2312.06635. Fetched: `arxiv.org/html/2312.06635v5`.

**Equations, as transcribed.**

- Eq (1), ungated linear attention: `S_t = S_{t-1} + k_t^T v_t`, `o_t = q_t S_t`.
- Eq (3), GLA: `S_t = (alpha_t^T 1) o S_{t-1} + k_t^T v_t = Diag(alpha_t) S_{t-1} + k_t^T v_t`.
- Gate parameterization: `alpha_t = sigmoid(x_t W_{a1} W_{a2})^{1/tau}`.
- Eq (4), parallel form with `b_t := prod_{j=1}^{t} alpha_j`:

  ```
  P_{ij} = sum_k Q_{ik} K_{jk} exp(log B_{ik} - log B_{jk}),  i >= j
  O = (P o M) V
  ```

- Chunkwise form: `S_[i+1] = (gamma_{i+1}^T 1) o S_[i] + (K_[i+1] o Gamma_[i+1])^T V_[i+1]`.

**Hypotheses as the paper states them.** `alpha_t ∈ (0,1)^{d_k}`, so `G_t = alpha_t^T 1 ∈
(0,1)^{d_k × d_v}`; matrix-valued state of size `d_k × d_v` accumulated by outer products;
causal mask `M`; log-space evaluation of Eq (4) for numerical stability (the gate products
underflow otherwise).

**Numeric instance.** `scripts/v15_n1_probes/p02_gla.py`, T=6, d_k=4, d_v=3.

```
GLA recurrent vs parallel Eq(4): 4.996003610813204e-16
no-cumprod control            : 2.6705073132902735
alpha==1 vs linear attention  : 8.881784197001252e-16
```

**What this occupies.** Contract §S-M's exponentiated prefix-sum form is Eq (4) verbatim.
`exp(C_i - C_j)` with `C = scan(log a)` **is** `exp(log B_i - log B_j)` with
`B = cumprod(alpha)`. GLA's gate is a per-channel vector `alpha_t ∈ (0,1)^{d_k}`; the
contract's is the scalar (`d_k = 1`) special case. The contract's `g = -softplus(Wx)`
differs from GLA's `sigmoid(...)^{1/tau}` only in the link function, and both enforce the
same constraint `log a ≤ 0` for the same stability reason. The log-space parallel
evaluation — the contract's `C = scan(g)`, `O(s)` — is GLA's, not new.

---

## 3. RetNet — the retention decay mask `[V-eq]`

**Citation.** Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue,
Jianyong Wang, Furu Wei, *Retentive Network: A Successor to Transformer for Large Language
Models*, 2023, arXiv:2307.08621. Fetched: `arxiv.org/html/2307.08621v4`.

**Equations, as transcribed.**

- Eq (1): `s_n = A s_{n-1} + K_n^T v_n`, `o_n = Q_n s_n = sum_{m=1}^{n} Q_n A^{n-m} K_m^T v_m`,
  with `A ∈ R^{d×d}`, `K_n, Q_n ∈ R^{1×d}`.
- Eq (2): `Q = X W_Q`, `K = X W_K` (content-aware projections).
- Eq (5), parallel: `Retention(X) = (Q K^T o D) V` with
  `D_{nm} = gamma^{n-m}` for `n ≥ m`, `0` for `n < m` — "causal masking and exponential
  decay along relative distance".
- Eq (8), multi-scale decay: `gamma = 1 - 2^{-5 - arange(0,h)} ∈ R^h`.

**Hypotheses as the paper states them.** `A` is diagonalizable and, after absorbing the
relative-position (xPos) rotation, reduces to the scalar `gamma`; `gamma` is a **constant
per head**, i.e. input-INDEPENDENT — this is the point of difference from GLA/Mamba-2 and
from the contract's carrier; causality `n ≥ m`; no softmax.

**Numeric instance.** `scripts/v15_n1_probes/p03_retnet.py`, T=8, d=4, d_v=3, gamma=0.9.

```
recurrent vs parallel (Eq 5) : 1.7763568394002505e-15
off-by-one exponent control  : 1.0495833364016214
Eq(8) gamma schedule h=4     : [0.96875    0.984375   0.9921875  0.99609375]
gamma==1 -> causal all-ones  : True
```

**What this occupies.** The decay-mask idea itself, and the `gamma = 1` → causal-all-ones
identity bind that the contract calls Lean #5. RetNet occupies the **fixed** decay mask;
Mamba-2 and GLA occupy the **input-dependent** one. Between the three there is no
remaining ground under "decay mask" for the contract to claim.

---

## 4. Negative- / complex-eigenvalue SSMs — the signed variant's occupant `[V-eq]`

**Citation.** Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K.H. Franke, Frank Hutter,
Massimiliano Pontil, *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*,
ICLR 2025, arXiv:2411.12537 (v1 2024-11-19, v3 2025-03-18).
Fetched: `arxiv.org/html/2411.12537v3`.

**Equations and theorems, as transcribed.**

- Eq (1): `H_i = A(x_i) H_{i-1} + B(x_i)`, `yhat_i = dec(H_i, x_i)`.
- **Theorem 1 (Parity)** — quoted: "A finite precision LRNN with finitely many layers ...
  can solve parity for arbitrary input lengths ... only if in at least one layer, there
  exist **x** such that **A(x)** has at least one eigenvalue `lambda ∉ {x ∈ R : x ≥ 0}`."
- Reparametrization, diagonal (Mamba): `A^{diag-}(x) := Diag(2 s(x) - 1)`, mapping the
  original `[0,1]` diagonal to `[-1,1]`.
- Reparametrization, generalized Householder (DeltaNet):
  `A^{GH-}(x) := I - 2 phi(x) v(x) v(x)^T`, eigenvalue `1 - 2 phi(x) ∈ [-1,1]`.
- Theorem 3 / Theorem 4 (products of GH matrices with eigenvalues in `[-1,1]` implement any
  finite-state automaton whose transition monoid is a group / recognize any regular
  language) — **`[V]`, statement obtained but no numeric instance run here. INADMISSIBLE
  as a load-bearing statement.** Only Theorem 1 and the two reparametrizations are `[V-eq]`.

**Hypotheses as the paper states them.** Finite precision; finitely many layers; arbitrary
input alphabet; diagonal transitions for `A^{diag-}`; products of generalized-Householder
transitions for Thm 3/4.

**Numeric instance.** `scripts/v15_n1_probes/p04_negative_eigenvalues.py`.

```
sign(prod a) vs (-1)^(P_i XOR P_j): 0.0
signed LRNN state           : 1.0  true parity: 0  decoded: 0  match: True
nonneg-eig LRNN: even/odd state ranges overlap by: 0.905454 -> no threshold readout separates parity (Thm 1)
```

**What this occupies.** Contract §S-M's signed variant in full. The probe shows the
contract's parity mask `(-1)^{P_i XOR P_j}`, `P` = prefix-XOR of sign bits, is **bit-exactly**
the sign of the semiseparable product `prod_{k=i+1..j} a_k` (residual `0.0`, not `1e-16`) —
i.e. the signed mask is not an extra mechanism, it is Mamba-2's `L` with the eigenvalue
range extended per `A^{diag-}(x) = Diag(2 s(x) - 1)`. Lean #3 (`parity_sign`) proves an
identity that Grazzi et al. already use as their construction.

---

## 5. MWU / Hedge — the replicator occupant `[V-eq]`

**Citations.**
(a) Sanjeev Arora, Elad Hazan, Satyen Kale, *The Multiplicative Weights Update Method: a
Meta-Algorithm and Applications*, Theory of Computing 8 (2012) 121-164, article v008a006.
Text extracted locally from the journal PDF (the WebFetch summarizer could not read it;
see FETCH LOG).
(b) Fryderyk Falniowski, Panayotis Mertikopoulos, *On the discrete-time origins of the
replicator dynamics: from convergence to instability and chaos*, arXiv:2402.09824 (v2,
2024-02-26); Int. J. Game Theory 2025.

**Equations and theorems, as transcribed (verbatim from the extracted text).**

- AHK Figure 1, Eq (2.1): `w_i(t+1) = w_i(t)(1 - eta m_i(t))`, with
  `p(t) = {w_1(t)/Phi(t), ..., w_n(t)/Phi(t)}`, `Phi(t) = sum_i w_i(t)`.
- AHK **Theorem 2.1**: "Assume that all costs `m_i(t) ∈ [-1,1]` and `eta ≤ 1/2`. Then the
  Multiplicative Weights algorithm guarantees that after T rounds, for any decision i,
  `sum_t m(t)·p(t) ≤ sum_t m_i(t) + eta sum_t |m_i(t)| + ln n / eta`."
- AHK Eq (2.6), Hedge (Freund-Schapire): `w_i(t+1) = w_i(t) · exp(-eta m_i(t))`;
  **Theorem 2.3** with `eta ≤ 1`:
  `sum_t m(t)·p(t) ≤ sum_t m_i(t) + eta sum_t (m(t))^2·p(t) + ln n / eta`.
- FM Eq (RD): `xdot_i = x_i [u_i(x) - u(x)]`, `u(x) = x · v(x) = sum_i x_i u_i(x)`.
- FM Eq (EW): `y_i(t+delta) = y_i(t) + delta u_i(x(t))` with
  `x_i(t) = exp(y_i(t)) / sum_j exp(y_j(t))`, hence Eq (III):
  `x_i(t+delta) = x_i(t) exp(delta u_i(x(t))) / sum_j x_j(t) exp(delta u_j(x(t)))`.
- FM Eq (5): a first-order Taylor expansion gives
  `x_i(t+delta) - x_i(t) = delta x_i(t)[u_i(x(t)) - u(x(t))] + O(delta^2)`, so (EW) is an
  Euler discretization of (RD). FM footnote 6: "(EW) is referred to as the multiplicative
  weights update (MWU)"; the Hedge instantiation is the same object.

**Hypotheses as the sources state them.** AHK: `n` decisions, `T` rounds, costs in `[-1,1]`
chosen adversarially with full knowledge of `p(t)`, `eta ≤ 1/2` (Thm 2.1) or `eta ≤ 1`
(Thm 2.3). FM: nonatomic population on the simplex `X = Delta(A)`, Lipschitz payoff
functions `u_i : X → R_+`; the Euler-limit statement is `O(delta^2)` and **only** valid as
`delta → 0` — FM's whole paper is that for non-infinitesimal `delta` the discrete model
and (RD) diverge qualitatively (Li-Yorke chaos in a 2×2 congestion game).

**Numeric instance.** `scripts/v15_n1_probes/p05_mwu_replicator.py`, n=5, T=300, eta=0.4.

```
MWU Thm 2.1  LHS=-5.335874  RHS=49.691038  holds=True
EW iterate vs softmax(cumulative payoff): 1.3322676295501878e-15
Fisher: dubar/dt=0.2700185304  Var(f)=0.2700185307  rel.err=1.11e-09
EW-vs-RD one-step error ratio delta/(delta/2) = 3.992 (expect ~4)
```

**What this occupies.** All of §S3-R, and Lean #8 in particular. `replicator_eq_cumsoftmax`
is FM Eq (EW) as written: the softmax of the cumulative payoff vector *is* the EW/MWU
iterate, and the probe reproduces that identity to `1.3e-15`. `dP/dt = Var(f)` is Fisher's
identity for (RD), reproduced above to `1.1e-9`, **under the hypothesis that the payoff
vector is frequency-independent** (constant `u`); see contradiction (c) below. The
"normalization × injection" mechanism the contract claims for S3-R is the normalization in
Eq (EW) plus a payoff field; neither half is new.

---

## 6. Mori-Zwanzig in ML — learned memory kernels from the projection formalism `[V-eq]`

**Citations.**
(a) Kevin K. Lin, Fei Lu, *Data-driven model reduction, Wiener projections, and the
Koopman-Mori-Zwanzig formalism*, J. Comput. Phys. 424 (2021) 109864, arXiv:1908.07725.
(b) Chao Ma, Jianchun Wang, Weinan E, *Model Reduction with Memory and the Machine Learning
of Dynamical Systems*, arXiv:1808.04258 (Commun. Comput. Phys. 2019).

**Equations, as transcribed.**

- Lin & Lu Eq (2.2a-b), discrete Mori-Zwanzig:

  ```
  x_{n+1} = P F(x_n) + sum_{k=1}^{n} Gamma_k(x_{n-k}) + xi_{n+1}(X_0),
  Gamma_k = P (xi_k o F),   P xi_n = 0
  ```

  the three terms being (i) Markov, (ii) memory, (iii) orthogonal / noise.
- Lin & Lu Eq (2.5), **Dyson's formula**, the identity the whole derivation rests on:

  ```
  M^{n+1} = sum_{k=0}^{n} M^{n-k} P M (Q M)^k + (Q M)^{n+1}
  ```

- Koopman operator: `M phi(X) = phi(F(X))`; Hilbert space `H = L^2(mu)`,
  `<f,g> = ∫ f g dmu`; `P` an orthogonal projection, `Q = I - P`.
- Wiener projection `P_W` onto `W = span(Psi ∪ M^{-1}Psi ∪ M^{-2}Psi ∪ ...)`, giving the
  stronger orthogonality `<xi_n, Psi(x_m)> = 0` for `n > m`.
- Ma, Wang & E Eq (11), the GLE form used for the ML reduction:

  ```
  d/dt phihat_j(x,t) = R_j(phihat(x,t)) + ∫_0^t K_j(phihat(x,t-s), s) ds + F_j(x,t)
  ```

  "The first term ... is a Markovian term; the second term ... is a memory term. The third
  term ... will be viewed as a noise term."

**Hypotheses as the sources state them.** A measure-preserving map `F` with invariant
probability measure `mu` (`F_* mu = mu`); observables in `H = L^2(mu)`; `P` an **orthogonal**
projection (`P = P^T = P^2`); the memory sum runs over the full history, and Ma-Wang-E
state the load-bearing caveat: "Solving the GLE accurately is almost equivalent to solving
the full system, because the memory kernel and noise terms contain the full information for
the unresolved variables."

**Numeric instance.** `scripts/v15_n1_probes/p06_mori_zwanzig.py`, dim 9, rank-3 projection.

```
P idempotent, P=P^T          : 3.3306690738754696e-16 0.0
Dyson (2.5) residual n=0     : 1.110e-16
Dyson (2.5) residual n=1     : 2.220e-16
Dyson (2.5) residual n=2     : 1.665e-16
Dyson (2.5) residual n=5     : 3.331e-16
Markov+noise WITHOUT memory  : 4.520e-01  (this is the size of the memory kernel)
```

**What this occupies.** The contract's THEORY OF THE COMPOSITION (lines 79-83) — "projection
of history onto a low-dimensional state yields Markov term + MEMORY KERNEL + noise ... the
composition is forced by the projection theorem". That is Lin & Lu Eq (2.2) and Ma-Wang-E
Eq (11), and both papers already draw the ML conclusion the contract draws: Ma-Wang-E
motivate recurrent architectures from exactly this split. The contract's `[V-field]` tag on
this line is upgraded here to `[V-eq]`, but the upgrade removes any novelty in the
*motivation*: the argument "attention plus recurrence is forced by Mori-Zwanzig" is
published.

---

## 7. ARFIMA / fractional-order nets — the Grünwald-Letnikov power-law kernel `[V-eq]`

**Citations.**
(a) Nabil Mlaiki, *VORT: Adaptive Power-Law Memory for NLP Transformers*, arXiv:2605.08966v1
(2026-05-09). Text extracted locally from the PDF.
(b) Javier E. Contreras-Reyes, Wilfredo Palma, *Statistical Analysis of Autoregressive
Fractionally Integrated Moving Average Models*, arXiv:1208.1728 (Comput. Stat. 2013).
(c) Origins: C.W.J. Granger & R. Joyeux (1980), J. Time Ser. Anal. 1:15-29; J.R.M. Hosking
(1981), *Fractional differencing*, Biometrika 68(1):165-176 — **`[V]` for these two, cited
through (b) and through the long-memory history paper arXiv:1406.6018; their own text was
not fetched. INADMISSIBLE as load-bearing on its own.**

**Equations, as transcribed (VORT, verbatim from the extracted PDF text).**

- Eq (2): `(I^alpha_GL f)(t) := sum_{j=0}^{t-1} w_j^{(alpha)} f_{t-j}`,
  `w_j^{(alpha)} := Gamma(j+alpha) / (Gamma(alpha) Gamma(j+1))`, with `w_j^{(alpha)} > 0`
  for all `j ≥ 0`, `alpha > 0`.
- Eq (3): `w_j^{(alpha)} ~ j^{alpha-1} / Gamma(alpha)` as `j → ∞`; the kernel is heavy-tailed,
  `sum_j w_j^{(alpha)} = ∞`.
- Eq (4): `W_alpha(z) = sum_j w_j^{(alpha)} z^j = (1-z)^{-alpha}`, `|z| < 1`.
- VORT's non-Markov lemma (§2): the GL sum has **no** exact two-term recurrence
  `S_t = c S_{t-1} + b f_t`, because such a recurrence has rational `z`-transform
  `b/(1-cz)` while `(1-z)^{-alpha}` is irrational for `alpha ∉ Z` (branch point at `z = 1`).
- Eq (5): Laplace representation `w_j^{(alpha)} = ∫_0^∞ e^{-lambda j} rho_alpha(lambda) dlambda`,
  `rho_alpha(lambda) = e^{-alpha lambda}(1-e^{-lambda})^{-alpha} e^{-lambda} / (Gamma(alpha)Gamma(1-alpha))`.
- Contreras-Reyes & Palma Eq (1): `Phi(B) y_t = Theta(B)(1-B)^{-d} eps_t`; Eq (2):
  `(1-B)^{-d} = sum_j [Gamma(j+d)/(Gamma(j+1)Gamma(d))] B^j`; Eq (4): `eta_j ~ j^{d-1}/Gamma(d)`;
  Eq (18): `gamma(h) ~ c_gamma |h|^{2d-1}`; Eq (6) spectral density
  `f(lambda) = (sigma^2/2pi)(2 sin(lambda/2))^{-2d} |Theta|^2/|Phi|^2`.

**Hypotheses as the sources state them.** VORT: `alpha ∈ (0,1)`, per-token learnable
`alpha_i ∈ [delta, 1]`; the exact GL sum is non-Markovian and is approximated by a
sum-of-exponentials with `S = O(log(T/eps))` terms (Theorem 3.1). Contreras-Reyes & Palma
Theorem 2.1: **stationarity requires `d ∈ (-1, 1/2)`** with the roots of `Phi(·)` outside
the unit circle.

**Numeric instance.** `scripts/v15_n1_probes/p07_arfima_gl.py`.

```
contract (-1)^k C(-a,k) vs VORT Eq(2) Gamma form: 1.6167622796103842e-15
Eq(4): sum w_k z^k  vs  (1-z)^-a                : 2.220446049250313e-16
Eq(3): w_k / (k^{a-1}/Gamma(a)) at k=1e4        : 0.9999883450108296 (-> 1)
alpha->0 (identity)        w_0..w_4 = [1.00e+00 1.00e-09 5.00e-10 3.33e-10 2.50e-10]
alpha->1 (cumulative sum)  w_0..w_4 = [1. 1. 1. 1. 1.]
(1-B)^d * (1-B)^-d, first 4 taps               : [ 1.  0.  0. -0.]
  d=0.20 -> H=d+1/2=0.70 | stationary (Thm 2.1, d<1/2)=True  | Hurst in (0,1)=True
  d=0.45 -> H=d+1/2=0.95 | stationary (Thm 2.1, d<1/2)=True  | Hurst in (0,1)=True
  d=0.80 -> H=d+1/2=1.30 | stationary (Thm 2.1, d<1/2)=False | Hurst in (0,1)=False
  d=1.00 -> H=d+1/2=1.50 | stationary (Thm 2.1, d<1/2)=False | Hurst in (0,1)=False
```

**What this occupies.** All of §S-K's fractional head (X34), including both of its binds.
The contract's `w_k = (-1)^k C(-alpha, k)` is **the same sequence** as VORT Eq (2) to
`1.6e-15`; the `alpha → 0` identity and `alpha → 1` cumulative-sum limits are printed above
and are immediate from Eq (4) (`(1-z)^0 = 1`, `(1-z)^{-1} = sum z^k`) — they are properties
of the binomial series, not findings. VORT additionally publishes the *architectural* move
the contract proposes: a learnable fractional order per unit driving a GL power-law
retention kernel **inside a transformer**, with the ARFIMA `pi_j ~ j^{d-1}/Gamma(d)`
optimal-predictor argument as its motivation. Its Section 2 also *proves* that the GL sum
admits no exact fixed-dimension recurrence — which is a sharper, published version of the
contract's Lean #12 (`first_order_cannot_delay`).

---

## 8. Symbolic dynamics / generating-partition estimators — the Pesin-deficit occupant `[V-eq]`

**Citations.**
(a) Erik M. Bollt, Theodore Stanford, Ying-Cheng Lai, Karol Życzkowski, *What symbolic
dynamics do we get with a misplaced partition? On the validity of threshold crossings
analysis of chaotic time-series*, Physica D 154 (2001) 259-286, doi:10.1016/S0167-2789(01)00242-1.
Text extracted locally from the authors' PDF.
(b) Maryam Contractor, *The Pesin Entropy Formula*, U. Chicago REU 2023 (following Mañé's
proof), for the theorem statements of Ruelle's inequality and Pesin's formula.
Primary sources named therein: Ya. B. Pesin (1977); D. Ruelle (1978); F. Ledrappier,
J.-M. Strelcyn, L.-S. Young — **`[V]` for the primaries; their own text was not fetched.**

**Equations and theorems, as transcribed.**

- Definition 2.10 (Kolmogorov-Sinai): `h(T; alpha) := lim_n (1/n) H(vee_{i=0}^{n-1} T^{-i} alpha)`;
  `h(T)` is the supremum over all partitions.
- Eq (7.1), Margulis-Ruelle inequality: `h_mu(f) ≤ ∫_M sum_i lambda_i^+ m_i dmu`.
- Notation 7.2: `chi := sum_i lambda_i^+ m_i` (positive exponents times multiplicities).
- **Theorem 7.15 / Eq (7.16), Pesin's entropy formula**: `h_mu(f) = ∫_M chi dmu`.
- Theorem 8.1: Pesin's formula holds **iff** `mu` has absolutely continuous conditional
  measures on unstable manifolds — the definition of an SRB measure (Def 8.2).
- Bollt et al., verbatim: "The generating partition is defined as a partition for which the
  topological entropy achieves its supremum. Thus the entropy of a symbolic sequence
  generated by a misplaced partition cannot be larger than the topological entropy of the
  dynamical system. In general, the misplacement of the partition leads to diminishing of
  the computed entropy of the system." Their result: that entropy is a devil's-staircase-like
  but **non-monotone** function of the amount of misplacement.

**Hypotheses as the sources state them.** Pesin: `f` a diffeomorphism of a compact manifold
`M` with `f'` Hölder continuous (`C^{1+alpha}`); `mu` `f`-invariant and **absolutely
continuous with respect to Lebesgue**. Without absolute continuity only the Ruelle
inequality (7.1) holds, i.e. the deficit is **≥ 0 by theorem, and its vanishing is a
statement about the measure as much as about the partition**. Bollt et al.: deterministic
(noiseless) map, finite alphabet, threshold-crossing partitions; the benchmark is the tent map.

**Numeric instance.** `scripts/v15_n1_probes/p08_pesin_symbolic.py` — tent map iterated in
**exact integer arithmetic** on `x = s/q`, `q = 15485863` prime, N = 4e6 (a float64 tent orbit
loses one bit per step, collapses after ~52 iterations, and falsely reports `h_sym = 0` at
the generating partition; the integer orbit is required for the probe to mean anything).

```
orbit revisited its seed within N steps (0 = no short period): 0
tent map lambda = 0.693147180560   log 2 = 0.693147180560
generating  x=0.5   h_sym = 0.6929   DEFICIT = lambda - h_sym = 0.0003
misplaced   x=0.4   h_sym = 0.5544   DEFICIT = lambda - h_sym = 0.1387
misplaced   x=0.25  h_sym = 0.4849   DEFICIT = lambda - h_sym = 0.2083
misplaced   x=0.6   h_sym = 0.5547   DEFICIT = lambda - h_sym = 0.1385
misplaced   x=0.75  h_sym = 0.4698   DEFICIT = lambda - h_sym = 0.2233
```

**What this occupies.** All of §S-G's X33 instrument. The contract's
`DEFICIT = lambda-hat - h_sym ≈ 0 iff the guards are a generating partition` is the Ruelle
inequality plus the Bollt et al. sentence quoted above — the direction (deficit ≥ 0) is a
theorem from 1978 and the estimator behaviour is the 2001 paper's subject. The contract's
own `[RUN: 0.0003 at the right guard, 0.156 at a wrong one]` is **independently reproduced
here** (0.0003 at `x = 1/2`; 0.139-0.223 at four wrong guards), so the instrument is sound —
but it is an instrument off the shelf, not a contribution. Bollt et al.'s non-monotonicity
result is a live hazard for the contract: a *smaller* deficit does not imply a *less
misplaced* guard.

---

## 9. Hamiltonian / flow-conserving nets — the conservation-census occupant `[V-eq]`

**Citation.** Sam Greydanus, Misko Dzamba, Jason Yosinski, *Hamiltonian Neural Networks*,
NeurIPS 2019, arXiv:1906.01563. Fetched: `ar5iv.labs.arxiv.org/html/1906.01563`.

**Equations, as transcribed.**

- Eq (2): `dq/dt = ∂H/∂p`, `dp/dt = -∂H/∂q`; symplectic gradient
  `S_H = (∂H/∂p, -∂H/∂q)`, "the direction of motion preserving the Hamiltonian value
  exactly constant".
- Eq (3): `L_HNN = || ∂H_theta/∂p - ∂q/∂t ||_2 + || ∂H_theta/∂q + ∂p/∂t ||_2`.

**Hypotheses as the paper states them.** The state is given in **canonical coordinates**
`(q, p)`; `H` is time-independent; the network outputs the scalar `H_theta` and its
gradients are taken by automatic differentiation in-graph (the network does not output the
vector field). Conservation is a property of the *field*, and holds up to the integrator's
own error — it is not exact for a non-symplectic integrator.

**Numeric instance.** `scripts/v15_n1_probes/p09_hamiltonian.py`, `H = (p^2+q^2)/2`,
`H_theta = theta (p^2+q^2)/2`, RK4 at `h = 1e-3` for 2e5 steps.

```
L_HNN(theta=1)    = 0.000e+00
L_HNN(theta=1.01) = 0.103956
L_HNN(theta=0.99) = 0.103956
min over theta of the sign-flipped loss = 9.798152 at theta=1.000 (never 0)
symplectic S_H : max |H - H_0| over 2e5 steps = 1.577e-14
unconstrained  : max |H - H_0| over 2e5 steps = 3.179e+00
```

The sign control matters: writing Eq (3) with `- ∂p/∂t` instead of `+ ∂p/∂t` gives a loss
with a strictly positive minimum (9.798) — the sign convention is load-bearing and is
verified here rather than assumed.

**What this occupies.** All of §S-C's framing. "Carriers conserve mass; readouts print
their dissipation budget" is the HNN inductive bias plus its accounting. HNN occupies the
"conserve by construction" half; the contract's contribution can only be the *census* —
printing the drift as a column on every table — which is bookkeeping discipline, not a
mechanism. Note the contract's `1e-12` conservation target is looser than the `1.6e-14`
that an exact symplectic field with RK4 gives here, so the target is not itself evidence
of anything.

---

## 10. Perturb-seq / GRN interventional identification `[V-eq]`

**Citations.**
(a) Atray Dixit, Oren Parnas, Biyu Li, Jenny Chen, Charles P. Fulco, Livnat Jerby-Arnon
et al., *Perturb-Seq: Dissecting Molecular Circuits with Scalable Single-Cell RNA Profiling
of Pooled Genetic Screens*, Cell 167(7):1853-1866.e17 (2016),
doi:10.1016/j.cell.2016.11.038. Fetched via PMC5181115.
(b) Alain Hauser, Peter Bühlmann, *Characterization and greedy learning of interventional
Markov equivalence classes of directed acyclic graphs*, JMLR 13 (2012) 2409-2464,
arXiv:1104.2808. Fetched: `ar5iv.labs.arxiv.org/html/1104.2808`.

**Equations and theorems, as transcribed.**

- Dixit et al. (MIMOSCA): `Y = X beta + eps` — "the model predicts each gene's (log)
  expression level (expression matrix **Y**) as a linear combination of the effects of guides
  (design matrix **X**), yielding the regulatory effect of each guide on each gene
  (coefficient matrix **beta**)"; fitted with elastic net, `l1_ratio = 0.5`, `alpha = 0.0005`.
- Hauser & Bühlmann Definition 6 (conservative family of targets): `I` is conservative if
  "for all `a ∈ [p]`, there is some `I ∈ I` such that `a ∉ I`".
- Hauser & Bühlmann **Theorem 10**: `D_1` and `D_2` are `I`-Markov equivalent iff for all
  `I ∈ I` the intervention graphs `D_1^{(I)}` and `D_2^{(I)}` are observationally Markov
  equivalent; equivalently, iff `D_1` and `D_2` have the same skeleton and the same
  v-structures **and** `D_1^{(I)}`, `D_2^{(I)}` have the same skeleton for all `I ∈ I`.
  Consequence: `I`-Markov equivalence is a **strictly finer** partition of DAGs than
  observational Markov equivalence.
  *(Provenance note: Thm 10 and Def 6 are as returned by the ar5iv fetch, paraphrased in
  the fetch rather than character-verbatim; the numeric instance below exercises the
  consequence, not the wording. The wording itself is `[V]`.)*

**Hypotheses as the sources state them.** Hauser & Bühlmann: causal sufficiency (no hidden
confounders), a DAG model, **perfect** (structural / `do`) interventions, and a
**conservative** target family — without conservativeness the parent set of a variable that
is intervened in every experiment is never identified. Dixit et al.: linearity and additivity
of guide effects on log-expression, with the covariates (cell state, guide dosage) entered
into the same design matrix.

**Numeric instance.** `scripts/v15_n1_probes/p10_perturb_seq.py`, linear SEM
`z → x → y` with confounding edge `z → y`, N = 4e5.

```
observational OLS  y ~ x : 1.135449   (true direct effect a = 0.80)
closed-form confounded plim a + b c sz^2/(c^2 sz^2 + se^2) = 1.135052  |diff| = 3.98e-04
interventional  do(x)    : 0.800267   |error vs a| = 2.67e-04
confounding bias removed by the intervention: 0.3354 -> 2.67e-04

observational: v~u = 0.9000 , u~v = 0.9277  (both nonzero -> same MEC)
after do(u): v~u = 0.8984 (survives) ; after do(v): u~v = 0.0022 (-> 0)
=> one intervention on a conservative target family separates the two DAGs
```

**What this occupies.** The contract's INTERVENTIONAL CHANNEL sentence — "interventions
identify `a` directly; observations confound `a` with `b`" — in full, with the closed form
of the confounding term (`a + b·Cov(z,x)/Var(x)`, matched to `4e-4` above) and the
identifiability theorem that licenses it. The contract tags this `[V]`; it is upgraded here.
What is **not** occupied by these two papers is the specific loss
`L = L_obs + zeta · L_jac` with gates as Jacobians — Perturb-seq fits a regression, it does
not add a Jacobian-matching term to a sequence model's objective. That is the one live piece
of the interventional channel (see below).

---

## WHAT REMAINS UNOCCUPIED

The honest answer is: **very little, and none of the pieces the contract currently treats
as its architecture.** Component by component, the fetches above put an existing equation
under every named block of PART I:

| contract block | occupant | left over |
|---|---|---|
| S-M carrier `W_ij = exp(C_i - C_j)`, `C = scan(g)` | GLA Eq (4) (log-space, per-channel); Mamba-2 Eq (6) `L` | the scalar-gate restriction and the `-softplus` link — parameter choices inside a published family |
| S-M signed variant, parity mask | Grazzi et al. `A^{diag-} = Diag(2s-1)`, Thm 1 | nothing; probe residual is `0.0` |
| S-M DAG resolvent `(I-A)^{-1}` "IS the successor representation" | Dayan 1993 SR, `Psi^pi = (I - gamma P_pi)^{-1}` `[V]` | the contract concedes the identification itself; feeding **edge gates** rather than similarities is a wiring choice |
| S3-R replicator carrier | AHK Eq (2.1)/(2.6); FM Eq (EW)/(III)/(RD) | nothing; Lean #8 restates Eq (EW) |
| S-K fractional head | VORT Eq (2)-(5), learnable per-unit `alpha`, in a transformer | nothing, including both binds |
| S-K "attention kept as it is, plus a memory kernel" | Mori-Zwanzig (Lin & Lu Eq 2.2, Ma-Wang-E Eq 11) as the *motivation*; Griffin (arXiv:2402.19427) and Samba (arXiv:2406.07522) as the *architecture* | Griffin and Samba interleave gated-recurrence and attention **layer-wise**; VORT composes power-law retention with linear-attention retrieval **within** a block. A within-layer composition of an input-dependent decay carrier with a softmax head is not among the fetched papers |
| S-G Pesin deficit | Ruelle (7.1) + Bollt et al. 2001 | nothing as an instrument |
| S-C conservation census | HNN Eq (2)-(3) | the census as a reporting discipline |
| interventional channel | Dixit et al.; Hauser & Bühlmann Thm 10 | `L = L_obs + zeta · L_jac` — a Jacobian-matching auxiliary loss on a sequence model's gates. Not found in the fetched literature |

**Precisely what is left, stated so it can be attacked:**

1. **The composition target, not the components.** No fetched paper builds a single model
   whose *label* is the next equilibrium state — basin index, decision-point flag, transition
   identity — from a committor/isocommittor guard vocabulary, and evaluates it on both a
   scan-native and an attention-native bed with a two-sided parity requirement (contract R4).
   Griffin/Samba/VORT are language models measured by perplexity. This is a **claim about
   the evaluation protocol and the label**, not about the operator, and it should be written
   that way.
2. **`L = L_obs + zeta · L_jac`.** Gates as Jacobians recovered from per-position `do()` bumps,
   trained as an auxiliary term. The identifiability half is occupied (Hauser & Bühlmann);
   the *loss* is the only piece of PART III that no fetch reached. R6 (intervention budget
   halved) is therefore the most defensible novelty claim in the round.
3. **The two-sided native test itself** — one arm required to sit within `Delta_res` of the
   scan skyline on BED-M *and* the attention skyline on BED-K. Hybrid papers report one
   aggregate benchmark; none of them reports both native skylines with floors and a
   distance-to-skyline column.

**Everything else in PART I should be written as composition, and every component sentence
should carry its citation before its name, exactly as CLAIM SHAPE requires.** In particular
the phrase "the composition is forced by the projection theorem" cannot be offered as the
round's insight: Ma, Wang & E offer it, in those words, in 2018.

---

## WHERE THE CONTRACT CONTRADICTS ITS SOURCES

**(a) "`g == 0` gives the causal all-ones mask = standard attention" (line 96; Lean #5,
line 140).** Dao & Gu's dual form is `(L o Q K^T) V` with **no softmax** (§2.4), and the
contract's own S-M says "ONE **unnormalized** causal hop `W_ij = exp(C_i - C_j)` on values
`V(x)`" (lines 88-89). Setting `g == 0` therefore lands on **causal linear attention**, not
on softmax self-attention. Probe p01 confirms `a_t = 1 → L = tril(ones)`; probe p02 confirms
`alpha = 1 → ungated linear attention` to `8.9e-16`. Both are the *masked-kernel-attention*
object. "PARITY WITH SELF-ATTENTION is by IDENTITY BIND" (line 139) is only true against a
softmax arm if a softmax sits between the mask and `V`; as written, `g == 0` binds to linear
attention, and the parity claim against softmax reverts to a measurement, i.e. back to TOST
and its `N ≥ 23`. **This is the single most consequential discrepancy found at this node.**

**(b) BED-K's `H-hat = alpha-hat + 1/2` (line 215) has a domain the contract does not state.**
`H = d + 1/2` comes from ARFIMA, where Contreras-Reyes & Palma Thm 2.1 requires
`d ∈ (-1, 1/2)` for stationarity; long memory needs `d ∈ (0, 1/2)`. The contract's fractional
head parametrizes `alpha` over `(0,1]` (VORT uses `[delta, 1]`), and probe p07 prints the
consequence: `alpha = 0.8 → H = 1.3`, outside the Hurst range `(0,1)` and outside
stationarity. Either BED-K's power-law bed must be generated with `alpha < 1/2`, or the
`H-hat = alpha-hat + 1/2` readout must be restricted to that range and the head's `alpha`
clipped for the estimator column. As stated, R3's `alpha-hat`/`H-hat` agreement test is
undefined over half its own parameter range.

**(c) S3-R's `dP/dt = Var(f)` (line 105) is missing its hypothesis.** Fisher's identity for
the replicator holds when the payoff/fitness vector is **frequency-independent** — probe p05
reproduces it to `1.1e-9` with constant `u`. For a game payoff `f(x) = J x` the identity is
false in general, and for the **antisymmetric `J`** the contract invokes in the same clause,
`f-bar = x·Jx = 0` identically, which cannot equal `2P` for `P ≠ 0`. Either `f` is constant
(and `J` plays no role) or `J` is antisymmetric (and `f-bar ≡ 0`); the clause asserts both.
Also note contract line 105 already carries the correction "`sqrt(2)` was the random-matrix
value" — this is the same class of defect one clause later, and it is the class L-EQ exists
to catch.

**(d) Naming hazard, not an error.** The contract attributes `w_k = (-1)^k C(-alpha, k)` to
Grünwald-Letnikov / Riemann-Liouville. Probe p07 confirms these are the coefficients of
`(1-z)^{-alpha}`, i.e. the fractional **integral** of order `alpha` (VORT Eq 2/4). The GL
fractional **derivative** has weights `(-1)^k C(alpha, k)`, symbol `(1-z)^{+alpha}` — the
other sign. Anyone re-deriving from "Grünwald-Letnikov derivative" will build the inverse
filter and both binds will fail. The contract's own limits (`alpha→0` identity, `alpha→1`
cumulative sum) are only correct for the integral, and are printed correct above.

**(e) Corroboration, filed for completeness.** Contract line 121's
`[RUN: 0.0003 at the right guard, 0.156 at a wrong one]` reproduces independently: probe p08
gives 0.0003 at the generating guard and 0.139 / 0.208 / 0.223 at four wrong ones. The
caveat that must travel with it is Bollt et al.'s: the deficit is **non-monotone** in the
amount of misplacement, so ranking candidate guards by deficit is not sound; only the
"approximately zero" test is.

---

## FETCH LOG (successes and failures, per the no-invention rule)

| target | route | result |
|---|---|---|
| Mamba-2 / SSD | `arxiv.org/abs/2405.21060` | **FETCH FAILED** — abstract page carries no equations |
| Mamba-2 / SSD | `arxiv.org/html/2405.21060v1` | OK (Def 3.1, 3.2, Eq 3, Eq 6, §2.4) |
| GLA | `arxiv.org/abs/2312.06635` | **FETCH FAILED** — abstract only |
| GLA | `arxiv.org/html/2312.06635v5` | OK (Eq 1, 3, 4, chunkwise) |
| RetNet | `arxiv.org/abs/2307.08621` | **FETCH FAILED** — abstract only |
| RetNet | `arxiv.org/html/2307.08621v4` | OK (Eq 1, 2, 5, 8) |
| Negative eigenvalues | `arxiv.org/html/2411.12537v3` | OK (Eq 1, Thm 1, Thm 3/4, both reparametrizations) |
| MWU survey | WebFetch of `theoryofcomputing.org/.../v008a006.pdf` | **FETCH FAILED** — summarizer could not read the PDF stream. Recovered by downloading and extracting text locally with `pypdf`; Thm 2.1, Eq 2.1, Eq 2.6, Thm 2.3 and the full proof read directly |
| EW → replicator | WebFetch of `arxiv.org/pdf/2402.09824` | **FETCH FAILED** — same cause. Recovered by local `pypdf` extraction; Eq (RD), (EW), (I), (III), (5) read directly |
| Mori-Zwanzig (Lin & Lu) | `ar5iv.labs.arxiv.org/html/1908.07725` | OK (Eq 2.2a-b, Eq 2.5 Dyson, Wiener projection) |
| Mori-Zwanzig (Ma-Wang-E) | `arxiv.org/pdf/1808.04258` then ar5iv | ar5iv OK (Eq 11); PDF route failed |
| VORT (fractional head) | WebFetch of `arxiv.org/pdf/2605.08966` | partial; **recovered by local `pypdf` extraction** — Eq (2)-(5) and Thm 3.1 read directly. The WebFetch summary of this paper asserted "`alpha → 0` recovers standard attention, `alpha → 1` first-order memory"; the actual text says `alpha_i ∈ [delta, 1]` and proves a quantisation bound "with correct analysis near `alpha = 0`". The summary was not used |
| ARFIMA | `ar5iv.labs.arxiv.org/html/1208.1728` | OK (Eq 1, 2, 4, 6, 18, Thm 2.1) |
| ARFIMA history | `arxiv.org/pdf/1406.6018` | extracted locally; narrative only, no usable equation — Granger-Joyeux 1980 and Hosking 1981 remain `[V]` |
| Pesin (Scholarpedia) | `scholarpedia.org/article/Pesin_entropy_formula` | **FETCH FAILED** — `connect ECONNREFUSED 173.255.237.117:443` |
| Pesin (REU, Mañé's proof) | `math.uchicago.edu/~may/REU2023/REUPapers/Contractor.pdf` | WebFetch failed; **recovered by local `pypdf` extraction** — Def 2.10, Eq 7.1, Thm 7.15/Eq 7.16, Thm 8.1 read directly |
| Bollt et al. 2001 | `webspace.clarkson.edu/~ebollt/Papers/PhysicaDMisplacedPartition.pdf` | WebFetch failed; **recovered by local `pypdf` extraction** — abstract and the generating-partition definition read directly |
| HNN | `ar5iv.labs.arxiv.org/html/1906.01563` | OK (Eq 2, Eq 3) |
| Perturb-seq | `cell.com/cell/fulltext/S0092-8674(16)31610-5` | **FETCH FAILED** — `getaddrinfo ETIMEOUT`. Recovered via PMC5181115 |
| Hauser & Bühlmann | `ar5iv.labs.arxiv.org/html/1104.2808` | OK in substance (Def 6, Thm 10, Def 11, Thm 18); wording paraphrased by the fetch, so the wording is `[V]` |
| Griffin / Hawk | `arxiv.org/html/2402.19427v1` | OK (RG-LRU Eq 1-4; interleaved-layer composition confirmed) |
| Samba | `arxiv.org/abs/2406.07522` | OK at abstract level for the composition question (`[V]`, sufficient for a negative claim about within-layer composition) |
| Successor representation | search only, Dayan 1993 | `[V]` — `Psi^pi = (I - gamma P_pi)^{-1}`, equation not fetched from the primary. **INADMISSIBLE** until n1 or a later node fetches Dayan (1993), Neural Computation 5(4):613-624 |

---

## FILES

- `scripts/v15_n1_probes/p01_ssd_mamba2.py` … `p10_perturb_seq.py` — one runnable probe per
  component; each prints an equation-agreement residual and an O(1) mis-transcription control.
