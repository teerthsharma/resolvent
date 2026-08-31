# V15 / X₃₆ — PRIOR ART AT EQUATION LEVEL, BEFORE THE NAMING

JUPITER-7, node `X₃₆` of the CEQ v15.3 round. Discharges `CEQ_V15_3_DELTA.md`
§X₃₆ (`PRIOR ART — state before naming`) and §X₃₇ (the certificates) against the
standing law **L-EQ** (`CEQ_V15_CONTRACT.md` lines 51-55): `[V]` — page exists,
intro matches — is INADMISSIBLE for a load-bearing statement; `[V-eq]` requires
the statement WITH HYPOTHESES plus one numeric instance run.

The ordering in the delta's own tag is binding: the art is fetched **before**
the construction is named, so the naming can be corrected by what is found.
§5, §6 and §8 are that correction.

## Admissibility legend

| tag | meaning |
|---|---|
| `[V-eq]` | defining equation transcribed from the source, hypotheses stated as the source states them, and a numeric instance under `scripts/v15_x36_probes/` executed with its real output pasted below |
| `[V]` | page or abstract reached, equation NOT verified — **INADMISSIBLE for any load-bearing statement**, flagged inline |
| `[U]` | named only |

Every numeric instance carries a **control**: a deliberately mis-transcribed
variant of the same equation whose output differs by O(1). A probe that only
prints `~1e-16` proves nothing about transcription; a probe that prints `~1e-16`
for the equation and `~1e0` for the mis-transcribed variant does.

Reproduce: `python scripts/v15_x36_probes/pNN_*.py` (numpy 1.26.4, scipy 1.17.1,
mpmath 1.3.0, gudhi, PyMuPDF 1.27.2.2, pypdf 6.10.2, CPython 3.11.9, Windows 11,
this box). Outputs below are pasted verbatim from those runs. **Nothing trains.**

**Scoreboard for this node: 4 of 4 X₃₆ occupants reach `[V-eq]`, and 4 of 4 X₃₇
certificates reach `[V-eq]`** (Berry upgraded from the delta's `[U]`). One
sub-statement remains `[V]` and is named inline. Two findings in §5 and §7
contradict the delta.

---

## 0. FETCH DISCIPLINE — why the LaTeX source and not the summariser

The brief records that two nodes last round were handed plausible-but-wrong
paraphrases, one of which dropped a paper's `α` range. This node used
`arxiv.org/e-print/<id>` (the author's LaTeX source) for all five ML papers, and
local `pypdf` / `PyMuPDF` rendering for the two scanned or typeset-only sources.
**The precaution was not cosmetic and it earned its keep on the paper that
matters most.** Extracting §3.3 of arXiv:2303.06349 from the rendered PDF with
`pypdf` returns:

```
achieved at𝜈log =\x001 (whilej𝜆j = 0 is achieved at𝜈log =1)
```

The Type-1 font encoding maps `−` to `\x00` and `∞` to the glyph `1`. Read from
the PDF, the sentence says the boundary is reached at `ν^log = −1` and `ν^log =
1` — finite values, i.e. **a closed interval**. Read from the LaTeX source it
says `−\infty` and `\infty` — **an open one**. The single sentence that decides
this entire node is exactly the sentence the PDF extractor corrupts, and it
corrupts it into the delta's own claim. Every quotation below is from the
`.tex`.

---

## 1. Unitary / orthogonal RNNs — Arjovsky, Shah, Bengio 2016 `[V-eq]`

**Citation.** Martin Arjovsky, Amar Shah, Yoshua Bengio, *Unitary Evolution
Recurrent Neural Networks*, ICML 2016, arXiv:1511.06464. Source `uRNN.tex` from
`arxiv.org/e-print/1511.06464`.

**Equations, as transcribed.**

- Lemma (§4, attributed by the paper to a linear-algebra text): *"A complex
  square matrix `W` is unitary if and only if it has an eigendecomposition of the
  form `W = V D V*`, where `*` denotes the conjugate transpose. Here
  `V, D ∈ C^{n×n}` are complex matrices, where `V` is unitary, and `D` is a
  diagonal such that `|D_{j,j}| = 1`."*
- The four building blocks (§4):

  ```
  D    diagonal with D_{j,j} = e^{i w_j},  parameters w_j ∈ R
  R    = I - 2 v v* / ||v||^2,  a reflection in v ∈ C^n
  Π    a fixed random index permutation matrix
  F, F^{-1}   the Fourier and inverse Fourier transforms
  ```
- The composition actually used:

  ```
  W = D_3 R_2 F^{-1} D_2 Π R_1 F D_1
  ```
- modReLU (§5.2), the **only** magnitude-touching map in the architecture:

  ```
  σ_modReLU(z) = (|z| + b) z/|z|   if |z| + b ≥ 0
               = 0                 if |z| + b < 0
  ```
  and the paper's own restatement `σ_modReLU(z) = σ_ReLU(|z| + b) · z/|z|`,
  with `b ∈ R` a learned bias, one per hidden dimension.

**Hypotheses as the source states them.** Each block is unitary and products of
unitary matrices are unitary; `F` is the norm-preserving DFT; the paper's claim
is about **gradients**, not about a gate — its §3 result is *"the first time a
neural network architecture has been mathematically proven to avoid exploding
gradients"*, and it holds because `‖D_k‖ = 1` for ReLU-type nonlinearities, not
because any magnitude is capped. What is **not** claimed: nothing in the paper
bounds the state norm through modReLU, whose gain is unbounded above.

**Numeric instance.** `scripts/v15_x36_probes/p01_urnn_unitary.py`.

```
uRNN W = D3 R2 F^-1 D2 Pi R1 F D1, n = 64
  ||W* W - I||_F                       : 1.0208173135101304e-13
  CONTROL ||W'* W' - I||_F (R w/o the 2): 1.3929922844893616
  max_j | |lambda_j| - 1 |              : 3.774758283725532e-15
  CONTROL max_j | |lambda_j| - 1 |      : 0.9999999999999991
  |lambda| min / max                    : 0.9999999999999973 1.0000000000000038
  | ||W h|| - ||h|| | / ||h||           : 1.017855708109902e-15
  CONTROL same ratio                    : 0.029431300713034014
  ||W^1000 h|| / ||h||                  : 0.99999999999998
modReLU magnitude gain  |sigma(z)|/|z|,  m = 200000 samples
  exactly 0.0 (closed lower endpoint)   : 29391 of 200000 = 0.146955
  max gain observed (NO upper cap)      : 327.3868252506667
  fraction with gain > 1                : 0.49996
  CONTROL modReLU without the ReLU, (|z|+b)/|z| -> min: -239.37113544286652 (negative: magnitudes flip sign, O(1) wrong)
```

**What this occupies, exactly.** The transition eigenvalues are on the unit
circle to `4e-15` — `|λ| ≡ 1` **identically**, a single point, not an interval.
In the delta's coordinates the uRNN transition is `a = 1 · e^{iθ}`: the `m ≡ 1`
face and nothing else. The delta's line *"it constrains the transition to be
exactly norm-preserving; the delta's `m ∈ [0,1]` is not that"* is **correct**.

**What the delta does not say, and needs to.** The uRNN's magnitude degree of
freedom is not absent; it is in modReLU, and modReLU's per-unit gain attains
**exactly 0** on 14.70% of a standard sample — a **closed** lower endpoint,
occupied since 2016 — while having **no upper cap at all** (max gain 327.39 on
this sample, which is the same order as R1's `â_max = 285.07`). So half of the
delta's "closed `[0,1]`" — the closed `0` — is 2016 prior art on the same
architecture the delta names first.

---

## 2. LRU — Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De 2023 `[V-eq]`

**This is the one that matters, and it is the one the delta gets closest to.**

**Citation.** Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando,
Caglar Gulcehre, Razvan Pascanu, Soham De, *Resurrecting Recurrent Neural
Networks for Long Sequences*, ICML 2023, arXiv:2303.06349. Source
`content_main.tex` / `appendix.tex` from `arxiv.org/e-print/2303.06349`.

**Equations, as transcribed.**

- §3.3 *Enforcing stability*, verbatim:

  > *"An important benefit of the exponential parameterization is that it makes
  > it simple to enforce stability on the eigenvalues. To see this, note that at
  > initialization, `|λ_j| = |exp(−ν_j)| ≤ 1` since `ν_j > 0`. Therefore, to
  > preserve stability during training, we can use an exponential or another
  > positive nonlinearity:*
  >
  > ```
  > λ_j := exp(−exp(ν_j^{log}) + i θ_j)
  > ```
  >
  > *where `ν^{log} ∈ R^N` is the parameter we optimize, and we set
  > `ν_j^{log} := log(ν)` at initialization. Note that a similar idea is used in
  > deep SSMs (Gu et al., 2021a) in the context of discretization.* **"We choose
  > an exponential non-linearity over a simple ReLU nonlinearity to increase
  > granularity around `|λ| = 1`, achieved at `ν^{log} = −∞` (while `|λ| = 0` is
  > achieved at `ν^{log} = ∞`)."**

- §3.2 Lemma 3.2 (`\begin{restatable}[]{lem}{sampling}`), verbatim:

  > *"Let `u_1, u_2` be independent uniform random variables on the interval
  > `[0,1]`. Let `0 ≤ r_min ≤ r_max ≤ 1`. Compute
  > `ν = −(1/2) log(u_1 (r_max² − r_min²) + r_min²)` and `θ = 2π u_2`. Then
  > `exp(−ν + iθ)` is uniformly distributed on the ring in `C` between circles of
  > radii `r_min` and `r_max`."*

- §3.4 Proposition (*Forward-pass blow-up*), verbatim:

  > *"Let `Λ` be diagonal with eigenvalues sampled uniformly on the ring in `C`
  > between circles of radii* **`r_min < r_max < 1`**. *Then, under constant or
  > white-noise input and Glorot input projection, ...*
  > `E[‖x_∞‖²] = 1/(r_max² − r_min²) · log((1−r_min²)/(1−r_max²)) · E[‖Bu‖²]`."

  with the paper's own `r_min = r_max = r` limit, its Eq. (6):
  `lim E[‖x_∞‖²]/E[‖Bu‖²] = 1/ρ = 1/(1 − r²)`.

- §3.4 normalisation: *"we use normalization parameter `γ^{log} ∈ R^N`,
  initialized element-wise as `γ_i^{log} ← log(√(1 − |λ_i|²))`"*.

- Appendix A, the paper's own reference implementation, verbatim from the
  source's commented `mintedbox`:

  ```python
  Lambda   = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
  nu_log   = np.log(-0.5*np.log(u1*(r_max**2-r_min**2) + r_min**2))
  theta_log = np.log(max_phase*u2)
  gamma_log = np.log(np.sqrt(1-np.abs(diag_lambda)**2))
  # defaults: r_min=0, r_max=1, max_phase=6.28
  ```

**Hypotheses as the source states them.** `ν^{log}` and `θ` are unconstrained
reals; Lemma 3.2's radii satisfy the **closed** chain `0 ≤ r_min ≤ r_max ≤ 1`;
the blow-up Proposition requires the **strict** chain `r_min < r_max < 1`; the
`γ` initialisation requires `|λ| < 1` for `log(√(1−|λ|²))` to be finite. The
stability argument the whole section rests on is §2's *"a sufficient condition
to ensure stability (i.e. `x_k` does not explode) is therefore `|λ_j| < 1` for
all `j`"* — **strict**, and stated as sufficient, never as necessary.

**Numeric instance.** `scripts/v15_x36_probes/p02_lru_lambda_range.py`.

```
(a) RANGE OF |lambda| UNDER  lambda = exp(-exp(nu_log) + i theta)
  nu_log =     -1000   |lambda| = 1.0
  nu_log =       -37   |lambda| = 0.9999999999999999
  nu_log =       -20   |lambda| = 0.9999999979388464
  nu_log =         0   |lambda| = 0.36787944117144233
  nu_log =         6   |lambda| = 6.2101364865661445e-176
  nu_log =         7   |lambda| = 0.0
  mpmath 600dps  nu_log= -1000:  1-|lambda| = 5.0759589e-435   (0 < |lambda| < 1 : True)
  mpmath 600dps  nu_log=  1000:  1-|lambda| = 1.0              (0 < |lambda| < 1 : True)
  SUP  |lambda| = 1, attained only at nu_log = -infinity  -> NOT attained
  INF  |lambda| = 0, attained only at nu_log = +infinity  -> NOT attained
  => the LRU magnitude range is the OPEN interval (0, 1).
CONTROL: inner exp dropped,  lambda = exp(-nu + i theta)
  nu =  -1.0  |lambda| = 2.718282   >1  LEAVES THE DISK
  max over nu in [-5,5]: 148.4131591025766 -- O(1e2), not O(1e-16).

(b) WHERE float64 CLOSES THE OPEN INTERVAL BY ROUNDING
  smallest nu_log on grid with |lambda| == 1.0 exactly (float64): -37.43
  smallest nu_log on grid with |lambda| == 0.0 exactly (float64): 6.613600000000005
  largest |lambda| strictly below 1 on grid: 0.9999999999999999
  1 - that value: 1.1102230246251565e-16  (= 1 ulp region)

(c) LEMMA 3.2 INITIALISATION: the chain 0 <= r_min <= r_max <= 1 IS CLOSED
  r_min=0.0, r_max=1.0:  |lambda| min=0.0010651767481782857 max=0.9999989390904241  exactly==r_max: 0/400000  CDF at midpoint emp=0.250393 theory=0.250000 |diff|=3.93e-04
  r_min=0.9, r_max=0.999:  |lambda| min=0.9000001185034464 max=0.9989998003483731  exactly==r_max: 0/400000  CDF at midpoint emp=0.487495 theory=0.486967 |diff|=5.28e-04
  r_min=1.0, r_max=1.0:  |lambda| min=1.0 max=1.0  exactly==r_max: 400000/400000
  r_min=0.0, r_max=0.0:  |lambda| min=0.0 max=0.0  exactly==r_max: 400000/400000
  CONTROL Lemma 3.2 with -log(.) instead of -(1/2)log(.), r=[0,1]:
    CDF at 0.5 = 0.500580  vs theory 0.250000   |diff| = 0.2506   (O(1), not O(1e-16))

(d) WHAT A CLOSED CAP DOES TO  1/(1 - a)
  closed cap m = clip(x,0,1):   exactly 1.0: 73766/200000 = 0.3688   exactly 0.0: 0.5002
  LRU open map:                 exactly 1.0: 0/200000 = 0.0000   exactly 0.0: 0.0141
  1/(1-m) under the CLOSED cap: infinities = 73766  max finite = 423463.04208343045
  1/(1-m) under LRU's open map: infinities = 0  max finite = 2442267.0626089503
  LRU gamma init  log(sqrt(1-|lambda|^2))  at |lambda| = 1 : -inf
  d|lambda|/d nu_log at nu_log= -30.0:  0.000000e+00   (|lambda|=1.000000000000)
  d|lambda|/d nu_log at nu_log= -10.0: -4.539791e-05   (|lambda|=0.999954601101)
  d|lambda|/d nu_log at nu_log=  -2.0: -1.182050e-01   (|lambda|=0.873423018493)
  d|lambda|/d nu_log at nu_log=   0.0: -3.678794e-01   (|lambda|=0.367879441171)
  clip cap gradient in the interior: 1.0 ; outside: 0.0 (attains the cap)
```

**The four questions the brief put to this section, answered.**

**(i) Is the delta's transcription right?** Yes. `λ = exp(−exp ν + iθ)` matches
§3.3 exactly, and the delta's tag `OPEN magnitude` is **correct**: in exact
arithmetic the range is the open interval `(0,1)`, verified at 600 decimal
digits (`1 − |λ| = 5.08e-435` at `ν^log = −1000`, still strictly positive). The
one-symbol control — dropping the inner `exp` — puts `|λ|` at `148.4` on the
same sweep, so the inner `exp` is exactly the thing doing the work.

**(ii) Does the paper discuss the boundary?** **Yes — explicitly, in one
sentence, naming both endpoints and the parameter value that reaches each.**
*"increase granularity around `|λ| = 1`, achieved at `ν^{log} = −∞` (while
`|λ| = 0` is achieved at `ν^{log} = ∞`)."* The openness is not an oversight, an
implementation accident, or a detail the authors did not consider. It is a
**stated design choice with a stated reason**: the exponential link is preferred
over ReLU *because* it spends parameter resolution near `|λ| = 1` instead of
reaching it. The delta's framing — that the closed/open distinction is a gap the
LRU authors left — does not survive their own sentence.

**(iii) Would a closed `[0,1]` be a difference LRU's own analysis notices?**
**Yes, and it notices it as a defect, not a gap.** Two of the paper's own results
break at `|λ| = 1`:
- the blow-up Proposition's hypothesis is `r_max < 1` **strict**, and its gain is
  `1/(1 − r²)`, which is the pole itself;
- the `γ` normalisation is initialised at `log(√(1 − |λ|²))`, which the probe
  evaluates at `|λ| = 1` and prints as `-inf`.

A closed magnitude with `1` attainable makes both undefined on a
positive-measure set of parameters. LRU's openness is load-bearing for LRU.

**(iv) The one place the delta is strictly more expressive than shipped LRU.**
The paper's §3.3 formula leaves `θ_j` unconstrained, but its **own reference
implementation** (Appendix A) writes `1j*jnp.exp(theta_log)`, so the shipped
phase is `exp(θ^{log}) ∈ (0, ∞)` and **`θ = 0` is not attainable**. The delta's
`θ ∈ {0, π}` for BED-M's `±1` gates therefore includes a point LRU's code cannot
represent. This is a real difference and it is on the **phase**, not the
magnitude — the opposite axis from the one the delta claims.

---

## 3. Complex diagonal S4 / Mamba `[V-eq]`

**Citations.**
- Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré, *On the Parameterization
  and Initialization of Diagonal State Space Models* (S4D), NeurIPS 2022,
  arXiv:2206.11893. Source `src2/method.tex`, `src2/appendix_experiments.tex`.
- Albert Gu, Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*, arXiv:2312.00752. Source `src/method.tex`, `src/appendix.tex`.

**Equations, as transcribed.**

- S4D §3.2, *Parameterization of `A`*, verbatim:

  > *"Note that the kernel `K(t) = C e^{tA} B` blows up to `∞` as `t → ∞` if `A`
  > has any eigenvalues with positive real part. [Goel et al. 2022] found that
  > this is a serious constraint that affects the stability of the model,
  > especially when using the SSM as an autoregressive generative model. They
  > propose to force the real part of `A` to be negative, also known as the
  > left-half plane condition in classical controls, by parameterizing the real
  > part inside an exponential function*
  > ```
  > A = -exp(A_Re) + i · A_Im.
  > ```
  > **"We note that instead of `exp`, any activation function can be used as long
  > as its range is bounded on one side, such as ReLU, softplus, etc."** *The
  > original DSS does not constrain the real part of `A`, which is sufficient for
  > simple tasks involving fixed-length sequences, but could become unstable in
  > other settings."*

- S4D §3.5, *Eigenvalue constraint*: *"While DSS found that letting them be
  unconstrained has slightly better performance, our experiments find that the
  difference is negligible and we recommend contraining negative real part of `A`
  as is standard practice in control systems."*

- **The ReLU option is not a remark. It is an ablated, reported row.**
  `src2/appendix_experiments.tex`, Table `tab:ablations-real-full`, column
  *Real part* ∈ {Exp, −, ReLU}:

  | Discretization | Real part | sCIFAR | SC (AR) | BIDMC (SpO2) |
  |---|---|---|---|---|
  | Bilinear | Exp | 85.20 (0.18) | 89.52 (0.01) | 0.1193 (0.0069) |
  | Bilinear | — | 85.35 (0.27) | 90.58 (0.37) | 0.1102 (0.0075) |
  | Bilinear | **ReLU** | **85.06 (0.06)** | **90.22 (0.25)** | **0.1172 (0.0063)** |
  | ZOH | Exp | 85.02 (0.24) | 89.93 (0.07) | 0.1303 (0.0014) |
  | ZOH | — | 85.15 (0.13) | 90.19 (0.58) | 0.1289 (0.0035) |
  | ZOH | **ReLU** | **84.98 (0.72)** | **90.03 (0.13)** | **0.1232 (0.0065)** |

- Mamba Theorem 1 (`src/method.tex`), verbatim: *"When `N=1, A=−1, B=1,
  s_Δ = Linear(x)`, and `τ_Δ = softplus`, then the selective SSM recurrence
  (Algorithm 2) takes the form*
  ```
  g_t = σ(Linear(x_t))
  h_t = (1 − g_t) h_{t−1} + g_t x_t ."
  ```
  with the proof in `src/appendix.tex` giving
  `Ā_t = exp(ΔA) = 1/(1 + exp(Linear(x_t))) = σ(−Linear(x_t)) = 1 − σ(Linear(x_t))`
  and `B̄_t = (ΔA)^{-1}(exp(ΔA) − I)·ΔB = 1 − Ā = σ(Linear(x_t))`.
- Mamba §3.5.2: *"`A` ... ultimately affects the model only through its
  interaction with `Δ` via `Ā = exp(ΔA)` (the discretization)"*, and §3.5.1's
  mechanism: *"the model can mechanistically filter out any particular input
  `x_t` ... when `g_t → 0`"*.

**Hypotheses as the sources state them.** `A_Re`, `A_Im` unconstrained reals;
`Δ > 0` (Mamba: `Δ = softplus(·)`, strictly positive); ZOH gives
`|Ā| = exp(Δ·Re A)`; the bilinear (Cayley) map
`Ā = (I + ΔA/2)(I − ΔA/2)^{-1}` sends the imaginary axis to the unit circle
**exactly** and the open left half-plane to the open unit disk. S4D's
recommendation of the constraint is explicitly a recommendation, not a theorem —
DSS ships without it.

**Numeric instance.** `scripts/v15_x36_probes/p03_s4d_mamba_boundary.py`.

```
S4D:  A = -f(A_Re) + i A_Im,  ZOH  Abar = exp(dt A),  dt = 0.01
  real part =     exp :  Re A max = -9.669076e-05   |Abar| max = 0.9999990330928518   exactly 1.0: 0/400000 = 0.0000
  real part =     ReLU:  Re A max =  0.000000e+00   |Abar| max = 1.0000000000000002   exactly 1.0: 131718/400000 = 0.3293
  real part = softplus:  Re A max = -9.668609e-05   |Abar| max = 0.9999990331395942   exactly 1.0: 0/400000 = 0.0000
    A_Re =  -30.0:  |Abar| = 0.9999999999999991   == 1.0 in float64: False
    A_Re =  -34.0:  |Abar| = 1.0   == 1.0 in float64: True

S4D bilinear (Cayley):  Abar = (1 + dt A/2)/(1 - dt A/2)
  Re A = 0 (ReLU branch):  max | |Abar| - 1 | = 4.440892098500626e-16
  CONTROL (1 + dt A) numerator: max | |Abar| - 1 | = 0.021960541860853278  (O(1e-2), not O(1e-16))
  Re A < 0 (exp branch) :  max |Abar| = 0.9999990349862501  all < 1: True

Mamba Theorem 1:  Abar_t = sigma(-Linear(x_t)) = 1 - g_t,  g_t = sigma(.)
  max | sigma(-L) - exp(softplus(L) * (-1)) | = 2.220446049250313e-16
  CONTROL sign flip, |sigma(L) - exp(-softplus(L))| max = 0.9999999951632721 (O(1))
  Abar range on this sample: min = 7.252287281612216e-09  max = 0.999999997581636
  exactly 0.0: 0   exactly 1.0: 0
    Linear =  -37.0:  Abar = 1.0
    Linear =  -36.0:  Abar = 0.9999999999999998
```

**What the actual constraint is, and whether it is open or closed. The answer
is not the same for the two activations S4D itself offers.**

| parametrisation | constraint | endpoint at `m = 1` |
|---|---|---|
| S4D with `exp` (the recommended one) | `Re A < 0` strictly | **open**, never attained |
| S4D with `softplus` | `Re A < 0` strictly | **open**, never attained |
| **S4D with `ReLU`** (the paper's own alternative, ablated and reported) | `Re A = −ReLU(A_Re) = 0` **exactly** for every `A_Re ≤ 0` | **CLOSED, attained on 32.93% of a standard sample** |
| DSS (unconstrained) | none | `\|λ\|` may exceed 1 |
| Mamba's gate (Thm 1) | `σ(·) ∈ (0,1)` | **open**; the paper's own "ignore this token" is `g_t → 0`, a limit |

**The `ReLU` row is the finding.** `−ReLU(·)` is a magnitude link whose range is
the **closed** half-line `(−∞, 0]`, and under either discretization it puts
`|Ā|` at exactly `1.0` on a positive-measure set of parameters — verified to
`4.4e-16` on the Cayley branch against an `O(1e-2)` mis-transcription control.
This is a **closed** magnitude cap with the upper endpoint attainable, in a
published, ablated, numerically reported configuration, in exactly the discipline
the delta is filing into, since 2022. It performs within noise of the open
variant (`85.06 ± 0.06` vs `85.20 ± 0.18` on sCIFAR).

**One arithmetic point the delta should own.** The `ReLU`/ZOH branch's max
`|Ā|` prints as `1.0000000000000002`. On this box, in float64, *"`|a| ≤ 1` by
construction"* is **false at the last bit** for a construction that puts the
gate exactly on the circle. A hard cap in float64 gives `≤ 1 + 1 ulp`, not `≤ 1`.

---

## 4. RoPE — Su, Lu, Pan, Murtadha, Wen, Liu 2021 `[V-eq]`

**Citation.** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng
Liu, *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
arXiv:2104.09864. Source `roformer_arxiv.tex`.

**Equations, as transcribed.**

- §3.2.1, the 2D solution:
  ```
  f_q(x_m, m) = (W_q x_m) e^{i m θ}
  f_k(x_n, n) = (W_k x_n) e^{i n θ}
  g(x_m, x_n, m−n) = Re[ (W_q x_m)(W_k x_n)* e^{i(m−n)θ} ]
  ```
  *"`θ ∈ R` is a preset non-zero constant."*
- §3.2.2, the general form: `f_{q,k}(x_m, m) = R^d_{Θ,m} W_{q,k} x_m` with
  `R^d_{Θ,m}` block-diagonal in `d/2` blocks `[[cos mθ_i, −sin mθ_i],
  [sin mθ_i, cos mθ_i]]`, *"the rotary matrix with pre-defined parameters
  `Θ = {θ_i = 10000^{−2(i−1)/d}, i ∈ [1, 2, …, d/2]}`"*.
- The relative-position identity:
  `q_m^T k_n = (R^d_{Θ,m} W_q x_m)^T (R^d_{Θ,n} W_k x_n) = x^T W_q R^d_{Θ,n−m} W_k x_n`
  *"where `R^d_{Θ,n−m} = (R^d_{Θ,m})^T R^d_{Θ,n}`.* **"Note that `R^d_Θ` is an
  orthogonal matrix, which ensures stability during the process of encoding
  position information."**
- §3.3: *"Since RoPE injects position information by rotation, which keeps the
  norm of hidden representations unchanged, ..."*
- Appendix, long-term decay: with `h_i = q_{[2i:2i+1]} k*_{[2i:2i+1]}` and
  `S_j = Σ_{i=0}^{j−1} e^{i(m−n)θ_i}`, the Abel transformation gives
  `|Σ_i h_i (S_{i+1} − S_i)| ≤ (max_i |h_{i+1} − h_i|) Σ_i |S_{i+1}|`, and
  *"the value of `(1/(d/2)) Σ_{i=1}^{d/2} |S_i|` decay[s] with the relative
  distance `m − n`"*.

**Hypotheses as the source states them.** `d` even; `θ_i` **preset and not
learned**; `R` depends on the position index only, never on content. The decay
claim is a claim about the Abel **bound's** factor, not about the inner product
itself, and it is asserted from a figure, not proved to be monotone.

**Numeric instance.** `scripts/v15_x36_probes/p04_rope_norm.py`.

```
RoPE, d = 128  theta_i = 10000^{-2(i-1)/d},  i = 1..d/2
  max_m ||R^T R - I||_F                : 1.076401158743041e-15
  max_m | ||R x|| - ||x|| |            : 0.0
  max_m max_j | |eig_j(R)| - 1 |       : 2.220446049250313e-16
  CONTROL (block rows use different frequencies):
    ||R'^T R' - I||_F                  : 1.7986227227217224
    | ||R' x|| - ||x|| |               : 0.09473369640720186
  max ||R_m^T R_n - R_{n-m}||_F        : 1.821170142722983e-14
  CONTROL with R_{m-n} instead, (m,n)=(3,11): 9.858670523075176

Long-term decay (the paper's own quantity) vs magnitude, side by side
   m-n |  ||R_{m-n} q||/||q||  |  (1/(d/2)) sum_i |S_i|
      0 |  1.0  |   32.500000
      1 |  1.0  |   31.538166
     16 |  1.0  |   15.774951
    256 |  1.0  |   6.543097
   1024 |  1.0  |   4.024113
   4096 |  0.9999999999999999  |   4.882792
  |prod of 10000 unit phases| - 1 = -4.107825191113079e-15
```

**What RoPE occupies.** The `m ≡ 1` face, exactly and only: norm preserved to
`0.0` absolute at `d = 128`, every eigenvalue of modulus `1` to `2.2e-16`. It is
the delta's `a = 1 · e^{iθ}` with the phase **not learned** and **not
content-dependent**.

**What RoPE does not occupy, stated because the delta's `m ∈ [0,1]` is adjacent
to it.** No magnitude parameter exists in RoPE, learned or otherwise. The
apparent attenuation with distance is interference across the `d/2` frequencies
**at exactly constant norm** — the probe runs the two columns side by side, and
the norm column reads `1.0` to the last bit at every distance while the Abel
factor falls from `32.5` to `4.0`. RoPE therefore **cannot** bound a path
product below 1, and cannot be cited for anything the delta's `m < 1` is
supposed to do. It does confirm the Lean #16 face: `|Π e^{iθ_k}| − 1 =
−4.1e-15` over `10^4` factors.

---

## 5. THE QUESTION THE ROUND NEEDS ANSWERED

### 5.1 Is "closed magnitude" genuinely unoccupied?

**No. Stated plainly: the closed magnitude is occupied at both endpoints, by
published work, in this discipline, and the upper endpoint is occupied by a
configuration the S4D paper itself ablates and reports.**

| endpoint | occupied? | by what, exactly |
|---|---|---|
| `m = 1` attainable | **yes** | uRNN (Arjovsky 2016): `\|λ\| ≡ 1` by construction. RoPE (Su 2021): `\|λ\| ≡ 1` by construction. **S4D with the `ReLU` real part (Gu 2022, §3.2 text + Table `tab:ablations-real-full`): `Re A = 0` exactly on a half-line of parameters, so `\|Ā\| = 1.0` exactly on 32.93% of a standard sample** — a genuine closed `[0,1]`, not a face |
| `m = 0` attainable | **yes** | modReLU (Arjovsky 2016): gain exactly `0.0` on 14.70% of a standard sample |
| both endpoints in one scalar link | **assembled from published parts** | `−ReLU` gives the closed `1`; `ReLU`-of-magnitude gives the closed `0`. The delta's construction is `clip(·, 0, 1)`, i.e. `ReLU(x) − ReLU(x−1)`, the standard clamp |
| polar `magnitude × phase` factorisation itself | **occupied** | LRU §3.3, `Λ = diag(exp(−ν + iθ))`, whose stated purpose is that it *"decouples magnitude and oscillation frequencies, making optimization with Adam easier"* — the delta's `a_i = m_i e^{iθ_i}` verbatim, with a different link on `m` |

What is **not** occupied, and is all that survives: the *particular* route by
which `m` is produced in X₃₆ — a band mask via the even feature with `R² =
1.000000`. That is a claim about **where `m` comes from**, not about the
interval it lives in. It is a much smaller claim than "closed magnitude", and it
is the only one the fetches leave standing.

### 5.2 The claim the author asked to be priced

> *"our closed `m` is the small delta, and it is exactly the delta R1's three
> divergent seeds paid for."*

**Verdict: too generous, on both halves, and the second half is not merely
generous but backwards.**

**Half one — "the small delta".** Too generous. The closed interval is not the
delta; §5.1 shows both endpoints are occupied and the polar factorisation is
LRU's. The residual delta is the band-mask construction of `m`, which the
brief's own framing does not claim.

**Half two — "exactly the delta R1's three divergent seeds paid for".** This is
the load-bearing error. R1's defect, from `V15_R1.md`, is stated as:

> *"`1/(1 − â_max)` is undefined at every one of the eight seeds:
> `â_max ∈ [1.1029, 285.0719] > 1`."*

The delta's remedy is *"X₃₆ makes `|a| ≤ 1` hold by construction"* with `m`
**closed** so that *"`0` and `1` are attainable values, not limits"*. **A closed
cap at `1` does not make `1/(1 − a)` defined.** It makes the pole *attainable*:
the probe's clip on a standard `N(0,3)` feature lands exactly on `1.0` for
`73766 / 200000 = 36.9%` of draws and yields `73766` infinities in `1/(1 − m)`;
LRU's open map yields `0` infinities on the same spread. And the delta's own
phase convention closes the trap: `θ ∈ {0, π}` includes `θ = 0`, so `m = 1,
θ = 0` gives `a = 1` exactly — the identity gate, sitting on the pole.

The measurement in R1 paid for **a cap**. It did not pay for a **closed** cap.
The version of X₃₆ that discharges the defect R1 measured is the **half-open**
one, `m ∈ [0, 1)`. Under `[0,1)` the amended operator's `1/(1 − a)` is finite for
every representable parameter, the pre-registered *"`â_max ≤ 1` by
construction"* still holds, and the delta loses nothing it can name.

**A third finding, on observability.** Even if the closed/open distinction were
novel, it is **not observable in the arithmetic the arm runs on**. In float64,
LRU's own map returns `|λ|` exactly `1.0` for `ν^log ≤ −37.43` and exactly `0.0`
for `ν^log ≥ 6.61`. The gradient at the boundary is `0.000000e+00` at
`ν^log = −30`, so an optimiser that reaches the region cannot leave it either.
A novelty claim resting on open-vs-closed is a claim about a distinction the
stored parameter does not represent.

### 5.3 What the delta gets right, recorded so it is not lost in the ruling

- The transcription `λ = exp(−exp ν + iθ)` is **correct** and the tag `OPEN
  magnitude` is **correct**. No mis-transcription to file.
- The reading of the uRNN — *"it constrains the transition to be exactly
  norm-preserving; the delta's `m ∈ [0,1]` is not that"* — is **correct**.
- The one genuine expressivity gain found is on the **phase**: LRU's shipped
  implementation writes `1j*jnp.exp(theta_log)`, so `θ = 0` is unattainable
  there, and the delta's `θ ∈ {0, π}` includes it. That is a real difference and
  it is not the one the delta claims.

---

## 6. DOES ANY SOURCE ALREADY REPORT THE FAILURE MODE R1 MEASURED? `[V-eq]`

**The mechanism, yes. The fix, yes, and it is the same fix. The magnitude, no.**

**Citation.** Karan Goel, Albert Gu, Chris Donahue, Christopher Ré, *It's Raw!
Audio Generation with State-Space Models*, ICML 2022, arXiv:2202.09729. Source
`src/method.tex`, `src/experiments.tex`. This is the paper S4D §3.2 cites when
it says *"found that this is a serious constraint"*.

**§3.1 *Stabilizing S4 for Recurrence*, verbatim:**

> *"First, unrolling the RNN mode involves powering up `Ā` repeatedly, which is
> stable if and only if all eigenvalues of `Ā` lie inside or on the unit disk.
> Second, the transformation maps the complex left half plane (i.e. negative real
> part) to the complex unit disk. Therefore computing the RNN mode of an SSM
> (e.g. in order to generate autoregressively) requires `A` to be a Hurwitz
> matrix.*
>
> *However, controlling the spectrum of a general DPLR matrix is difficult;*
> **"empirically, we found that S4 matrices generally became non-Hurwitz after
> training."** *We remark that this stability issue only arises when using S4
> during autoregressive generation, because S4's convolutional mode during
> training does not involve powering up `Ā` and thus does not require a Hurwitz
> matrix."*
>
> *Definition. A Hurwitz matrix `A` is one where every eigenvalue has negative
> real part.*
>
> *Proposition (`prop:stable` in the source; the arXiv build numbers it within
> §3). A matrix `A = Λ − p p*` is Hurwitz if all entries of `Λ` have negative
> real part.*

and the closing remark: *"This ... can be enforced by regularization or
reparameteration (e.g. run its entries through an `exp` function). In practice,
we found that not restricting `Λ` and letting it learn freely led to stable
trained solutions."*

**The instrument.** Figure 5, caption verbatim: *"**(S4 Stability)** Comparison
of spectral radii for all `Ā` matrices in a SaShiMi model trained with different
S4 parameterizations. The instability in the standard S4 parameterization is
solved by our Hurwitz parameterization."* The figure is a sorted scatter of the
spectral radius of every `Ā` in the trained model — **the same instrument R1
built.** Read off the plot (`figs/spectral_radii.png`, rendered and inspected;
the paper publishes no table of these numbers, so the following is `[V]` on the
figure, not `[V-eq]`): roughly 900 `Ā` matrices, the *Standard* curve crosses
`1.000` at about index 830 and tops out near **`1.012`**; the *Hurwitz* curve
approaches `1.000` from below and never crosses.

**Table `tab:s4-ablation`, verbatim** — and this table is the answer to the
delta's own counter-prediction:

| Learned | Frozen | NLL | Stable generation |
|---|---|---|---|
| — | `diag + pq*` | 1.445 | ✓ |
| `diag + pq*` | — | **1.420** | **✗** |
| `diag − pp*` | — | **1.419** | **✓** |

**What this settles.**

1. **The mechanism is prior art.** A trained diagonal/DPLR gate leaving the
   stable region is documented, measured with a spectral-radius instrument, and
   fixed by a reparametrisation that makes the constraint hold **by
   construction** — the identical move X₃₆ proposes, published 2022.
2. **The delta's counter-prediction is already answered in the literature, and
   answered against itself.** The delta filed: *"Phase gates converge 8/8 but the
   crossing shrinks to ≤ 3/8, because the magnitude cap removes gain the
   divergent seeds were exploiting."* SaShiMi's own ablation buys stability for
   `1.420 → 1.419` NLL — **0.001, in the improving direction**. The constrained
   parametrisation was not worse. This does not decide the delta's cell, but it
   removes "constraining the magnitude must cost capability" from the list of
   things that can be assumed; if the crossing shrinks in R11's re-run, the
   literature's prior is that the cause is something other than the cap.
3. **The magnitude is not prior art.** Nothing found reports a trained gate
   magnitude of `285`. SaShiMi's excursion is `≈ 1.2%` above the circle; LRU
   reports instability and loss blow-up but publishes no magnitude; R1's is
   `285.07`, i.e. `2.8 × 10^4` percent. **That gap is itself a finding.** The
   published failure mode is a gate that *drifts across* the boundary by a
   fraction of a percent. R1's is a gate that leaves it by two orders of
   magnitude, with an order-of-magnitude gap between the crossing population
   (`[1.10, 1.51]`) and the failing one (`[20.31, 285.07]`) and nothing in
   between. A cap will hide that, but the literature gives no reason to believe
   a cap *explains* it, and X₃₆ as filed would close the question without
   answering it.

**Also documented, in LRU (§3.3, §3.4, verbatim):** *"we also observed that
learning the diagonal model can be more unstable than learning the dense model in
some experiments"*; *"as we moved `r_min` and `r_max` closer to 1, the training
loss also started to blow up at initialization"*; *"without enforcing stability,
performance starts to degrade as we increase `r_max` past 0.9 in the sCIFAR
task. With stability enforced, we can increase `r_max` up to 0.99 and improve
performance."*

---

## 7. X₃₇ — THE CERTIFICATES, PRICED

`MISTAKES.md` P-10 is the reason each of these carries its hypotheses.

### 7.1 Berry phase `[V-eq]` — **upgraded from the delta's `[U]`**

**Citation.** M. V. Berry, *Quantal Phase Factors Accompanying Adiabatic
Changes*, Proc. R. Soc. Lond. A **392**:45-57 (1984), DOI `10.1098/rspa.1984.0023`.
The available copy is a JSTOR scan with **no text layer**; `pypdf` returns empty
strings for every body page. Pages were rendered at 200 dpi with PyMuPDF and read
directly. Equation numbering below is Berry's own.

**Equations, as transcribed (pp. 46-47, 49).**

```
H(R(t))|ψ(t)⟩ = iħ|ψ̇(t)⟩                                                   (1)
H(R)|n(R)⟩ = E_n(R)|n(R)⟩                                                   (2)
|ψ(t)⟩ = exp{(−i/ħ)∫₀ᵗ dt′ E_n(R(t′))} exp(iγ_n(t)) |n(R(t))⟩               (3)
γ̇_n(t) = i⟨n(R(t))| ∇_R n(R(t))⟩ · Ṙ(t)                                    (4)
|ψ(T)⟩ = exp(iγ_n(C)) exp{(−i/ħ)∫₀^T dt E_n(R(t))} |ψ(0)⟩                   (5)
γ_n(C) = i ∮_C ⟨n(R)| ∇_R n(R)⟩ · dR                                        (6)
⟨m|∇n⟩ = ⟨m|∇H|n⟩/(E_n − E_m),   m ≠ n                                      (8)
γ_n(C) = −∬_C dS · V_n(R)                                                   (9)
V_n(R) ≡ Im Σ_{m≠n} ⟨n|∇_R H|m⟩ × ⟨m|∇_R H|n⟩ / (E_m − E_n)²               (10)
H(R) = (1/2)[[Z, X−iY],[X+iY, −Z]]                                         (12)
E_+(R) = −E_−(R) = (1/2)(X²+Y²+Z²)^{1/2} = R/2                             (13)
exp{iγ_±(C)} = exp{∓ (1/2) i Ω(C)}                                         (18)
```
*"where `Ω(C)` is the solid angle that `C` subtends at the degeneracy."*

**Hypotheses as Berry states them, and they are the whole point.**

> *"the excursion of the system between times `t = 0` and `t = T` can be pictured
> as transport round a closed path `R(t)` in parameter space ... such that
> `R(T) = R(0)`. ... **For the adiabatic approximation to apply, `T` must be
> large.**"*
>
> *"the natural basis consists of the eigenstates `|n(R)⟩` (**assumed discrete**)
> ... provided `|n(R)⟩` is **single-valued** in a parameter domain that includes
> the circuit C."*
>
> *"The normalization of `|n⟩` implies that `⟨n|∇_R n⟩` is imaginary, which
> guarantees that `γ_n` is real."*

Eqs (8) and (10) additionally require **non-degeneracy** (`E_n − E_m` and
`(E_m − E_n)²` in the denominators); eq (18) is exact only for the standard
form (12).

**Numeric instance.** `scripts/v15_x36_probes/p05_certificates.py`, block (a):
discrete Wilson loop of the `E_+` eigenvector round a cone of half-angle `θ₀`,
against `−Ω/2` with `Ω = 2π(1 − cos θ₀)`.

```
  theta0=0.3000  Omega=0.280629115  -Omega/2=-0.140314558  Wilson=-0.140314504  err= 5.389e-08
      gauge-randomised (must be invariant) err= 5.389e-08   CONTROL open circuit err=-2.899e+00
  theta0=1.0472  Omega=3.141592654  -Omega/2=-1.570796327  Wilson=-1.570796085  err= 2.422e-07
      gauge-randomised (must be invariant) err= 2.422e-07   CONTROL open circuit err= 5.614e-01
  theta0=1.9000  Omega=8.314473564  -Omega/2=-4.157236782  Wilson= 2.125948338  err=-1.870e-07
      gauge-randomised (must be invariant) err=-1.870e-07   CONTROL open circuit err=-1.858e+00
  CONTROL eq (18) mis-transcribed WITHOUT the 1/2:  -Omega vs -Omega/2 at
    theta0=pi/3 differs by 1.5707963267948961 rad (O(1)).
```

(The `θ₀ = 1.9` row differs from `−Ω/2` by exactly `2π`; phases are defined mod
`2π` and the wrapped error is `1.87e-7`.) Two controls fire at `O(1)`: dropping
the closing overlap of the circuit destroys gauge invariance, and dropping the
`1/2` in eq (18) is off by `π/2` at `θ₀ = π/3`.

**Pricing.** `[V-eq]` as physics. **But not admissible as a certificate for X₃₇
as the delta uses it, and the delta's own tagging is the right register.** Berry
requires (i) a closed circuit in a *parameter* space, (ii) adiabatic transport,
`T` large, (iii) a discrete, non-degenerate spectrum, (iv) a single-valued
eigenbasis. A learned sequence of gate phases is none of these: there is no
Hamiltonian, no adiabatic limit, and no eigenvector family whose overlap is being
parallel-transported. The delta says *"Berry phase `[U]` as the continuous
frame"* — **frame is exactly the right word and it must stay that word.** If
"Berry phase" appears in a load-bearing sentence of any later contract, it is
P-10 again.

### 7.2 Kuramoto `[V-eq]`

**Citation.** F. A. Rodrigues, T. K. DM. Peron, P. Ji, J. Kurths, *The Kuramoto
model in complex networks*, Physics Reports **610**:1-98 (2016), arXiv:1511.07139.
Source: the arXiv PDF, extracted locally with `pypdf`. (Kuramoto's own 1975
proceedings paper and his 1984 book are not fetchable; this review states the
model and the order parameter with its own equation numbers and cites them to
Kuramoto refs [2,3].)

**Equations, as transcribed (§2, eqs 1-3, 8-9).**

```
θ̇_i = ω_i + (λ/N) Σ_{j=1}^N sin(θ_j − θ_i),   i = 1,…,N                     (1)
R e^{iψ(t)} = (1/N) Σ_{j=1}^N e^{iθ_j(t)}                                    (2)
θ̇_i = ω_i + λ R sin(ψ − θ_i)                                                (3)
R = λR ∫_{−π/2}^{π/2} cos²θ · g(λR sin θ) dθ                                 (8)
λ_c^{KM} = 2 / (π g(0))                                                      (9)
```

**Hypotheses as the source states them.** *"Kuramoto considered `g(ω)` to be
unimodal and symmetric centered at `ω = ω̄`"*, shifted so `ω̄ = 0` and
`g(ω) = g(−ω)`; all-to-all coupling with the `1/N` normalisation; eq (3) is
derived *"by multiplying both sides of Eq. 2 by `e^{−iθ_i}` and equating the
imaginary parts"* — an identity, not an approximation; eq (9) is the `N → ∞`,
stationary onset obtained *"by letting `R → 0+` in Eq. 8"*, so it is an
asymptotic threshold, not a finite-`N` one. The paper also records the
feedback structure the delta relies on: *"the effective coupling is now
proportional to the order parameter `R`, creating a feedback relation between
coupling and synchronization."*

**Numeric instance.** `p05_certificates.py`, block (b). `N = 4000`,
Lorentzian `g` with `γ = 0.5` so `g(0) = 1/(πγ)` and eq (9) gives
`λ_c = 2γ = 1`; the exact partially-locked branch is `R = √(1 − λ_c/λ)`.

```
  Lorentzian g(0) = 1/(pi*gamma) = 0.636619772   eq(9) lambda_c = 1.000000000   (= 2*gamma = 1.0)
  lambda= 0.4  R(sim)=0.008783   sqrt(1-lambda_c/lambda)=0.000000   |diff|=0.0088
  lambda= 0.9  R(sim)=0.021542   sqrt(1-lambda_c/lambda)=0.000000   |diff|=0.0215
  lambda= 1.5  R(sim)=0.589111   sqrt(1-lambda_c/lambda)=0.577350   |diff|=0.0118
  lambda= 3.0  R(sim)=0.821588   sqrt(1-lambda_c/lambda)=0.816497   |diff|=0.0051
  eq(1)-integrated R = 0.785008 vs eq(3)-integrated R = 0.785008  (N=400, lambda=3)
  max | eq(1) - eq(3) |            : 3.552713678800501e-15
  CONTROL sin(theta_i - psi)       : 2.839821424841 (O(1))
```

The sub-critical rows sit at the finite-`N` floor `≈ 1/√N = 0.0158`, as eq (9)'s
`N → ∞` hypothesis predicts. The eq(1)≡eq(3) identity holds at `3.6e-15`; the
sign-flipped control is off by `2.84`, a factor of `8 × 10^14`.

**Pricing.** `[V-eq]`, and usable — with its hypotheses carried. The delta's
*"the order parameter `r` is printed"* is well-founded: `R ∈ [0,1]` by
construction from eq (2), it is exactly the modulus of a mean of unit phases, and
it is the right instrument for *"waves coupling around a point"*. Two caveats to
carry: eq (9) is asymptotic in `N`, so a small-`N` reading has a floor of
`1/√N` that is not synchronisation; and eq (1)'s `1/N` all-to-all coupling is not
the coupling a multi-channel gate has unless it is made so.

### 7.3 Euler–Poincaré `[V-eq]`

**Citation.** Allen Hatcher, *Algebraic Topology*, CUP 2002, §2.2, pp. 146-147
(free PDF, `pi.math.cornell.edu/~hatcher/AT/AT.pdf`, extracted with `pypdf`).

**Statement, as transcribed.**

> *"For a **finite CW complex** `X`, the Euler characteristic `χ(X)` is defined
> to be the alternating sum `Σ_n (−1)^n c_n` where `c_n` is the number of
> `n`-cells of `X`, generalizing the familiar formula vertices − edges + faces
> for 2-dimensional complexes."*
>
> **Theorem 2.44.** `χ(X) = Σ_n (−1)^n rank H_n(X)`.

**Hypothesis as the source states it: `X` a finite CW complex.** That is the
whole hypothesis. Nothing about manifolds, vector fields, dynamics or equilibria
enters, and the theorem's content is that `χ` *"depends only on the homotopy
type of `X`"* and *"is independent of the choice of CW structure"*.

**Numeric instance.** `p05_certificates.py`, block (c), the 7-vertex minimal
triangulation of `T²`, with `β_k` computed as GF(2) ranks of the boundary
matrices:

```
  V=7 E=21 F=14   chi = V - E + F = 0
  beta over GF(2): b0=1 b1=2 b2=1   sum (-1)^k b_k = 0
  Thm 2.44 residual |chi_cells - chi_betti| = 0
  CONTROL chi mis-transcribed as V + E + F = 42   (differs by 42, O(1))
```

**Pricing.** `[V-eq]`, standard, and the delta's use of it is fine as long as
the object is a finite complex — which a Rips complex on finitely many points is.

### 7.4 Poincaré–Hopf `[V-eq]` — **and the X₃₇(c) binding as written is a category error**

**Citation.** John W. Milnor, *Topology from the Differentiable Viewpoint*, §6,
p. 35 (`maths.ed.ac.uk/~v1ranick/papers/milnortop.pdf`, extracted with `pypdf`).

**Statement, as transcribed, hypotheses first because they are the finding.**

> *"Let `M` be a **compact manifold** and `w` a smooth vector field on `M` with
> **isolated zeros**. If `M` has a boundary, then `w` is required to **point
> outward at all boundary points**.*
>
> ***Poincaré–Hopf Theorem.** The sum `Σι` of the indices at the zeros of such a
> vector field is equal to the Euler number `χ(M) = Σ_{i=0}^{m} (−1)^i rank
> H_i(M)`. In particular this index sum is a topological invariant of `M`: it
> does not depend on the particular choice of vector field."*

**All four hypotheses are load-bearing: `M` compact, `M` a *manifold*, the zeros
isolated, and `w` outward-pointing on any boundary.**

**Numeric instance, and it is a counterexample to the delta's binding.**
`p05_certificates.py`, block (d). The field is the Hopf normal form
`ṙ = r(1 − r²), θ̇ = 1` on the closed disk `D²` of radius 2 — one zero, at the
origin; the census is complete.

```
  index at the origin (winding of w/|w|) = 1.000000000  -> 1
  on |p| = 2.0: w . n has sign(min)=-1 sign(max)=-1  -> w points INWARD;
    Milnor's hypothesis holds for -w, whose zero set and index are the same.
  Sigma(iota) = 1   chi(D^2) = 1   -> Poincare-Hopf HOLDS.
  Rips carrier of the trajectory (2000 pts): beta = [1, 1, 0]   Sigma(-1)^k beta_k = 0
  X37(c) as written would compare 0 against 1 and declare a CENSUS DEFECT.
```

**The delta writes:**

> *"**(c) Euler–Poincaré.** `Σ(−1)^k β_k` from the toolkit **must equal** the
> Poincaré–Hopf index sum of the equilibrium census. **Mismatch ⇒ census
> defect.**"*

**Here the census is complete, Poincaré–Hopf holds exactly, and the two numbers
differ.** They are invariants of two different spaces: `Σι = 1 = χ(D²)` is an
invariant of the **domain** `M` on which the field lives; `Σ(−1)^k β_k = 0` is
an invariant of the **carrier**, which is the `ω`-limit set the trajectory
samples — a circle. Milnor's hypotheses bind `M`; nothing in the theorem binds
the carrier, and a Rips complex of a trajectory is in general neither `M` nor
homotopy equivalent to it, and is not a manifold at all.

**The registered diagnostic as written would fire on a correct census.** Under
M-18's own correction — *"any diagnostic registered for the re-run must be run on
the corpus alone, and against a zero-step control, before it is trusted"* — this
diagnostic fails that test on a two-dimensional textbook system before it ever
reaches the corpus.

**The repair, stated so the certificate is not lost.** The binding is sound only
if the complex whose `β_k` are taken is homotopy equivalent to the compact
domain `M` carrying the census. Two admissible forms:
- build the complex on a **sample of the domain** (a grid or dense sample of the
  region the census covers), not on a trajectory; or
- keep the trajectory carrier and bind it to something that *is* an invariant of
  a `ω`-limit set — `β₁ ≥ 1` for a recurrent node, which is X₃₇(b) and is
  independently fine — while dropping the equality to the index sum.

X₃₇(a) (integer `Z` winding, *"a non-integer reading is an instrument defect"*)
and X₃₇(b) (`β₁ ≥ 1` iff recurrent) survive unchanged. **X₃₇(c) as written does
not.**

---

## 8. THE NAMING, CORRECTED BY WHAT WAS FOUND

The `PRIOR ART — state before naming` ordering exists so this section can
overwrite the delta. Four corrections, in descending size.

1. **Do not name the family.** `a_i = m_i e^{iθ_i}` is LRU's polar
   parametrisation of a complex diagonal gate, published 2023, adopted for a
   stated reason (Adam-friendly decoupling of magnitude from frequency). Name
   the *link*, not the family: X₃₆ is **a clipped-magnitude variant of the LRU
   polar gate**. Everything claimed must be a claim about the link function.

2. **Do not claim the closed interval.** The closed upper endpoint is occupied by
   S4D's `ReLU` real-part variant — the S4D paper's own alternative, its own
   ablation table, `85.06 (0.06)` sCIFAR — and by uRNN and RoPE as the `m ≡ 1`
   face; the closed lower endpoint is occupied by modReLU (2016). What is left
   is the **band-mask construction of `m` via the even feature**, which is a
   different and much smaller claim, and the only one that survives the fetches.

3. **Change `[0,1]` to `[0,1)` or state why not.** The defect R1 measured is that
   `1/(1 − â_max)` is undefined. A closed cap leaves it undefined at the
   attainable endpoint — `36.9%` of clipped draws land exactly on `1.0` in the
   probe — and the delta's own `θ ∈ {0, π}` puts `a = 1` in range. The half-open
   cap discharges the defect and costs nothing the delta names. **If the closed
   form is kept, the pre-registration must say what `1/(1 − a)` evaluates to at
   `a = 1`,** because the re-run will hit it.

4. **Retire the phrase "by construction" for float64 claims, or bound it.** The
   probe's `ReLU`/ZOH branch — the closest published analogue of a hard cap —
   returns `|Ā|` max `1.0000000000000002`. In float64 a hard cap gives
   `≤ 1 + 1 ulp`. The correct pre-registration reads `â_max ≤ 1 + O(ε)`.

**And one thing to add, because it is the strongest true claim available.** The
`θ = 0` point is unattainable in LRU's shipped implementation
(`1j*jnp.exp(theta_log)` forces `θ > 0`), and the delta's construction reaches
it. That is a small, real, checkable difference on the phase axis. It is not the
difference the delta claims, and it is worth more than the one it does.

---

## 9. WHAT REMAINS INADMISSIBLE

| statement | tag | why |
|---|---|---|
| the numeric values in SaShiMi Figure 5 (crossing index, max spectral radius `≈ 1.012`) | `[V]` | read off a rendered plot; the paper publishes no table. The *caption* and the §3.1 sentence *"S4 matrices generally became non-Hurwitz after training"* are `[V-eq]`-grade text; the numbers are not |
| Berry phase as a **certificate** for a learned gate-phase sequence | inadmissible | the theorem is `[V-eq]` but its hypotheses (adiabatic, closed circuit in parameter space, discrete non-degenerate spectrum) are not met. Admissible only as the delta already tags it: a frame |
| X₃₇(c) `Σ(−1)^k β_k = Σι` as a **must-fire** | inadmissible as written | §7.4 exhibits a complete census with a `1` vs `0` mismatch and no defect |

---

## FETCH LOG (successes and failures, per the no-invention rule)

| target | route | result |
|---|---|---|
| Orvieto et al., LRU | `arxiv.org/e-print/2303.06349`, LaTeX source | **OK** — §3.3 stability paragraph and the boundary sentence, Lemma 3.2, Prop 3.3 (blow-up), §3.4 `γ` init, Appendix A reference implementation |
| Orvieto et al., LRU | `arxiv.org/pdf/2303.06349`, local `pypdf` | **PARTIAL / MISLEADING** — Type-1 encoding maps `−`→`\x00` and `∞`→`1`; the boundary sentence extracts as `"achieved at ν^log = 1"`, i.e. reads as CLOSED. **Not used.** Recorded in §0 |
| Arjovsky, Shah, Bengio, uRNN | `arxiv.org/e-print/1511.06464`, `uRNN.tex` | OK (Lemma, the four blocks, `W = D₃R₂F⁻¹D₂ΠR₁FD₁`, modReLU, initialisation) |
| Gu, Gupta, Goel, Ré, S4D | `arxiv.org/e-print/2206.11893`, `src2/*.tex` | OK (§3.2 `A = −exp(A_Re) + iA_Im` and the "any activation bounded on one side, such as ReLU, softplus" sentence; §3.5; Table `tab:ablations-real-full`) |
| Gu & Dao, Mamba | `arxiv.org/e-print/2312.00752`, `src/*.tex` | OK (Theorem 1, its proof in `src/appendix.tex`, §3.5.2 `Ā = exp(ΔA)`, §3.5.1 `g_t → 0`) |
| Su et al., RoPE | `arxiv.org/e-print/2104.09864`, `roformer_arxiv.tex` | OK (2D solution, `R^d_{Θ,m}`, the orthogonality sentence, the Abel long-term-decay bound) |
| Goel, Gu, Donahue, Ré, SaShiMi | `arxiv.org/e-print/2202.09729`, `src/*.tex` + `figs/spectral_radii.png` | OK for text (§3.1, Def, Prop 2, `tab:s4-ablation`); figure read visually, numbers `[V]` |
| Berry 1984 | `physics.mcgill.ca/~keshav/551/berryphase2.pdf` (JSTOR scan) | **`pypdf` FETCH FAILED** — image-only, no text layer, every body page extracts empty. **Recovered** by rendering pp. 46-47 and 49 at 200 dpi with PyMuPDF and reading them; eqs (1)-(10), (12)-(18) transcribed from the rendered pages |
| Berry 1984 | `royalsocietypublishing.org/doi/10.1098/rspa.1984.0023` | not attempted for full text (paywalled); DOI recorded |
| Berry 1984 | `michaelberryphysics.wordpress.com`, `itp.tu-berlin.de`, `sites.pitt.edu` mirrors | **FETCH FAILED** — HTML error pages, not PDFs |
| Kuramoto model | `arxiv.org/pdf/1511.07139` (Rodrigues et al. review), local `pypdf` | OK (eqs 1, 2, 3, 8, 9 and the hypotheses paragraph) |
| Kuramoto 1975 / 1984 primary | — | **not fetched.** The review's eq numbers and its citations [2,3] are used; no statement here rests on Kuramoto's own wording |
| Hatcher, *Algebraic Topology* | `pi.math.cornell.edu/~hatcher/AT/AT.pdf`, local `pypdf` | OK (Euler-characteristic definition and Theorem 2.44 with the finite-CW hypothesis) |
| Milnor, *Topology from the Differentiable Viewpoint* | `maths.ed.ac.uk/~v1ranick/papers/milnortop.pdf`, local `pypdf` | OK (§6 p.35, Poincaré–Hopf with all four hypotheses). The file is an OCR'd scan; two OCR slips were repaired against context (`L�`→`Σι`, `at such a vector field`→`of such a vector field`) and are flagged here. All other wording verbatim |
| Edelsbrunner, Letscher, Zomorodian 2002 | `link.springer.com/content/pdf/10.1007/s00454-002-2885-2.pdf`, local `pypdf` | OK (`H_k^{ℓ,p} = Z_k^ℓ/(B_k^{ℓ+p} ∩ Z_k^ℓ)`, eq 3) — fetched but **not load-bearing here**; `β₁` in §7.4 is computed by `gudhi`, and the Euler–Poincaré statement comes from Hatcher |
| Ghrist, *Elementary Applied Topology* | `www2.math.upenn.edu/~ghrist/EAT/EATpage.pdf` | **FETCH FAILED** — HTML stub. Not needed; Milnor covers Poincaré–Hopf |
| Xiao, Chang, Niu, RMP 82:1959 | `arxiv.org/pdf/0907.2021` | fetched as a Berry fallback; **not used**, the primary was recovered |
