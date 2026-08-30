# V13 — X₂₉ prior art: deflation, the Wiener equalizer, code multiplexing

Fetched 2026-08-31. Contract item X₂₉ (differential signalling against the
`a @ a` common mode, per-mode MMSE gains `g_i = S_i/(S_i+N_i)`, Walsh-code
multiplexing of hop streams sharing one residual stream). Project law forbids
any X₂₉ claim before this file lands, and permits composition claims only.
Classes: `[V]` source fetched and read this session, `[V-t]` title/abstract or
bibliographic record only, `[U]` owed and unfetched. Counts at the foot of the
Owed section.

---

## Verdict

DC-mode attenuation is a named standard operation in graph signal processing:
the `λ = 0` Laplacian eigenvector is called the constant or DC-spectral
component (Ortega et al., *Proc. IEEE* **106**(5), Fig. 3), any filter with
`h(λ₀) = 0` is an ideal high-pass graph filter under the graph filtering
theorem, and spectral clustering routinely drops `u₀` — so the project's
deflation is a citation and not a discovery, doubly so because self-attention
is already published as a normalized graph adjacency matrix (Shi et al., ICLR
2022) whose rank-1 common mode carries two published names of its own, the
over-smoothing subspace `M = {eC}` and the residual `res(X) = X − 1xᵀ`.
Orthogonal-code multiplexing of superposed streams is the VSA/HRR construction
outright — bind by circular convolution or elementwise product, superpose by
addition, retrieve by correlation, with a crosstalk term analysed since Plate
1994 — and it has already been run inside neural networks twice, by Cheung et
al. (NeurIPS 2019), who modulate a network's internal activations by `±1`
codes, and by DataMUX (NeurIPS 2022), which multiplexes `N` streams into one
`d`-dimensional representation through fixed orthogonal or Hadamard maps.
The project's crosstalk arithmetic is half right: `√(M/L)` is the published VSA
scaling and reduces to `1/√L` for a single random interferer, but "exactly 0
for Walsh codes" holds only when each stream carries a **scalar** and the
readout is a **full-length** correlation, and for vector payloads bound
elementwise the Walsh-derived VSA of Alam et al. (arXiv:2410.22669, Thm. 3.1)
proves a non-zero noise term `η` whenever `ρ ≥ 2` streams share `d` dimensions
without expanding `d`. A transformer residual stream does not meet the
exact-zero condition: it is a shared additive bus read by later layers through
arbitrary learned linear projections onto 64–128-dimensional slices (Elhage et
al. 2021, "Subspaces and Residual Stream Bandwidth"), not a synchronous
equal-power single-path channel terminated in a matched filter of the full code
length, and the per-head slicing alone destroys the full-length correlation the
zero requires. The Wiener equalizer is the least occupied of the three lineages
and still not empty: the per-eigenmode gain
`h̃(λ) = σ²(λ)/(σ²(λ) + σ²ₙ(λ))` is Eq. (51) of Isufi et al., *IEEE TSP*
**72**, and has been the graph Wiener filter since Girault et al., ICASSP 2014.
One control this fetch supplies for free: the common-mode energy fraction of
`a @ a` is a monotone function of attention sharpness alone, and an i.i.d.
Gaussian-logit null at logit `sd ≈ 1.51` reproduces `0.768 ± 0.050` at
`n = 64`, so the measured `0.7550` is not evidence of learned structure until
it is reported against a sharpness-matched null.

---

## Graph signal processing and Wiener filtering on graphs

### E1 `[V]` — Isufi, Gama, Shuman, Segarra, "Graph Filters for Signal Processing and Machine Learning on Graphs", *IEEE Trans. Signal Process.* **72**, 4745–4781 (2024), DOI 10.1109/TSP.2024.3349788 (arXiv:2211.08854)

Full PDF read this session (text recovered with `pymupdf` after the fetch tool
returned only PDF stream bytes).

Graph Fourier transform, quoted at Eq. (4) — eigendecomposition of the graph
shift operator `S = VΛV⁻¹`:

```
x̃ = V⁻¹ x ,        x = V x̃          (Eq. 4)
```

Frequency ordering by the Laplacian quadratic form, Eq. (6):

```
Q_V(v_i) = v_iᴴ L v_i = λ_i ,
0 = Q_V(v₁) ≤ Q_V(v₂) ≤ … ≤ Q_V(v_N)
```

with the source's own consequence: "The lowest graph frequency is λ₁ = 0 which
corresponds to a constant eigenvector for a connected graph."

The MMSE/Wiener solution, Eq. (50), and its graph frequency response, Eq. (51),
verbatim:

```
H* = argmin_H E‖H(y + n) − y‖²₂ = Σ_y (Σ_y + Σ_n)⁻¹              (Eq. 50)

h̃(λ) = σ²_d(λ) / (σ²_d(λ) + σ²_n(λ)) = 1 / (1 + σ²_n(λ)/σ²_d(λ))  (Eq. 51)
```

valid "when the covariance matrices have the same eigenvectors as the GSO",
`Σ_y = V diag(σ²_y(λ)) Vᴴ`, `Σ_n = V diag(σ²_n(λ)) Vᴴ`. The paper's own gloss:
"the response at frequency λ is controlled by the inverse signal-to-noise (SNR)
ratio `SNR⁻¹(λ) := σ²_n(λ)/σ²_d(λ)`", i.e. "a frequency-adaptive regularizer
given by the inverse SNR". (The numerator subscript prints as `d` in Eq. 51 and
as `y` in Eq. 50; the object is the same signal-mode variance.)

Delta: Isufi et al. define per-eigenmode MMSE gains `S(λ)/(S(λ)+N(λ))` on a
graph shift operator's eigenbasis; X₂₉'s Wiener stage proposes per-mode gains
`g_i = S_i/(S_i+N_i)` on the eigenmodes of `a @ a`; **the difference is zero at
the level of the formula.** `a @ a` is a degree-2 polynomial graph filter of a
row-stochastic shift operator (Eq. 1.6 of E4), so the project's equalizer is
Eq. (51) evaluated on a particular GSO.

### E2 `[V]` — Ortega, Frossard, Kovačević, Moura, Vandergheynst, "Graph Signal Processing: Overview, Challenges, and Applications", *Proc. IEEE* **106**(5), 808–828 (2018), DOI 10.1109/JPROC.2018.2820126

Full PDF read this session. Graph Fourier analysis, Eq. (14), and the filtering
theorem, Eqs. (16)–(17):

```
ŝ = F s = V⁻¹ s                                                   (Eq. 14)
s_out = V · diag[h(λ₀) … h(λ_{N−1})] · V⁻¹ s_in                   (Eq. 16/17)
```

described as "the graph Fourier filtering theorem". The naming that settles the
first verdict question, verbatim: "the lowest frequency `Ω₀ = 0` corresponds to
the least varying spectral component, the **constant or DC-spectral
component**"; and in Fig. 3, "In the Laplacian case, the lowest frequency is
`λ = 0`, representing a constant value throughout the graph."

Delta: Ortega et al. name the `λ = 0` mode the DC component and define
attenuating it as the choice `h(λ₀) = 0` inside a standard filtering theorem;
X₂₉ proposes to remove a rank-1 constant mode from a hop operator; **the
difference is zero.** The operator `I − (1/N)11ᵀ` that X₂₉ would apply is the
ideal high-pass graph filter of Eq. (17) with `h(λ₀) = 0, h(λ_k) = 1`.

### E3 `[V]` — Stanković, Daković, Sejdić, "Graph Signal Processing — Part II: Processing and Analyzing Signals on Graphs", arXiv:1909.10325

Full PDF read this session. Establishes that discarding the DC eigenvector is
routine practice rather than a design decision, verbatim: clustering results
are "obtained based on the three smoothest eigenvectors, `u₁, u₂, and u₃`
(**excluding the constant eigenvector, `u₀`**), of the graph Laplacian matrix
`L = D − W`". Also carries the two-channel low-pass/high-pass graph filter bank
`H_L(L)`, `H_H(L)` with perfect-reconstruction conditions (Eqs. 88–90).

Delta: the spectral-clustering pipeline drops `u₀` as a matter of course;
X₂₉ drops the constant mode of `a @ a` and calls it deflation; **the difference
is zero**, and the operation is old enough to be stated parenthetically.

### E4 `[V]` — Zheng, Cheng, Sun, "Wiener filters on graphs and distributed polynomial approximation algorithms", arXiv:2205.04019 (2022-05-09)

Read this session. Graph shift `S`, filtering as `x ↦ y = Hx` (Eq. 1.1), and
the polynomial graph filter of commutative shifts (Eq. 1.6):

```
H = h(S₁,…,S_d) = Σ_{l₁=0}^{L₁} … Σ_{l_d=0}^{L_d} h_{l₁…l_d} S₁^{l₁} … S_d^{l_d}
```

with the graph shift exemplified by "the adjacency matrix `A`, Laplacian matrix
`L = D − A`, and symmetrically normalized Laplacian `L_sym := D^{−1/2} L
D^{−1/2}`". The paper's Wiener filters for stationary and deterministic graph
signals are "essentially the product of a polynomial filter and inverse of
another polynomial filter" (Thms 4.1, 4.4, 5.1).

Delta: this entry supplies the sentence that classifies `a @ a`: a second hop
is `S²` for the shift `S = a`, hence a polynomial graph filter of degree 2, and
its equalization is a rational graph filter; **the difference is zero for the
object and non-zero only for the substrate** — Zheng et al. filter sensor data
on a network, X₂₉ filters a hop's contribution to a residual stream.

### E5 `[V-t]` — Girault, Gonçalves, Fleury, Mor, "Semi-supervised learning for graph to signal mapping: A graph signal Wiener filter interpretation", *ICASSP* 2014, 1115–1119, DOI 10.1109/ICASSP.2014.6853770

Identifier taken from E1's reference list [109] and Crossref-verified. This is
the citation E1 gives for the graph Wiener filter of Eq. (51), and the paper
"shows improvement upon conventional label propagation algorithms" using it.

Delta: the earliest graph-domain home of the exact gain formula X₂₉ proposes;
**zero.**

### E6 `[V-t]` — Perraudin, Vandergheynst, "Stationary Signal Processing on Graphs", *IEEE Trans. Signal Process.* **65**(13), 3462–3477 (2017), DOI 10.1109/TSP.2017.2690388

Crossref-verified. Establishes graph wide-sense stationarity, the condition E1
Eq. (51) requires for the covariance matrices to be diagonal in the GSO
eigenbasis.

Delta: supplies the hypothesis under which X₂₉'s per-mode gains are the optimal
linear estimator at all; **the difference is that X₂₉ has not stated the
stationarity assumption its diagonal gain vector silently makes.** If the
noise covariance is not diagonal in the `a @ a` eigenbasis, `g_i =
S_i/(S_i+N_i)` is not the MMSE solution — Eq. (50) is, and it is not diagonal.

### E7 `[V-t]` — von Luxburg, "A tutorial on spectral clustering", *Statistics and Computing* **17**(4), 395–416 (2007), DOI 10.1007/s11222-007-9033-z

Crossref-verified. The canonical reference for the practice E3 states in
passing.

Delta: as E3; **zero.**

---

## Vector Symbolic Architectures and Holographic Reduced Representations

### E8 `[V]` — Kleyko, Rachkovskij, Osipov, Rahimi, "A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations", arXiv:2111.06077 (*ACM Comput. Surv.* 55(6):130)

Full PDF read this session. HRR binding by circular convolution, Eq. (10):

```
a ∘ b ≡ z_j = Σ_{k=0}^{D−1} b_k a_{j−k mod D}
```

with atomic hypervector components "generated from the normal distribution with
mean 0 and variance `1/D`". Unbinding, Eq. (9):

```
a = b ⊘ (a ∘ b)
```

and the survey's own statement of the crosstalk fact, verbatim: "when the
unbinding operation is applied to a superposition of HVs, the result will not
be exactly equal to the original bound HV. It will also contain **crosstalk
noise**; therefore, the unbinding operation is commonly followed by a clean-up
procedure". Superposition is component-wise addition in HRR, MAP, FHRR, BSC and
MBAT alike (Table 2). Retrieval accuracy, Eq. (14):

```
p_corr = ∫_{−∞}^{∞} dx/(√(2π) σ_h) · exp(−(x − (µ_h − µ_r))²/(2σ_h²)) · (Φ(x, 0, σ_r))^{N−1}
```

with `N` the item-memory size, attributed to Frady et al. (E9).

Delta: X₂₉ proposes to modulate hop streams by orthogonal codes and despread by
correlation, then reason about how many streams fit before retrieval fails;
E8 is the survey of a field that has done exactly that since 1990, with the
correlator readout, the crosstalk term, and the capacity integral all named;
**the difference is zero for the construction and non-zero only for the choice
of code family** — the survey's models use random hypervectors, X₂₉ proposes
deterministic Walsh rows. That difference is addressed by E11, which is also
prior art.

### E9 `[V]` — Frady, Kleyko, Sommer, "A theory of sequence indexing and working memory in recurrent neural networks", arXiv:1803.00412 (*Neural Computation* **30**(6), 1449–1513, 2018)

Full PDF read this session. The definition of readout SNR, Eq. (9), and the
result that settles the project's crosstalk-scaling assertion:

```
r(K) := σ²(a_d) / σ²(n_d)          (Eq. 9)
```

with `n_d` "decoding noise resulting from crosstalk and neuronal noise", and
the paper's own summary of its scope, verbatim: "we will show that all models
perform similarly and the accuracy can be predicted by the **universal
sensitivity formula `s = √(N/M)`**", where `N` is the network/vector dimension
and `M` the number of superposed items. The same `s = √(N/M)` is shown to hold
for HDC bipolar hypervectors, FHRR phasors, and MBAT random unitary matrices.

Arithmetic consequence, derived here from Eq. (9) and `s = √(N/M)`: the
crosstalk standard deviation relative to the retained signal is `√(M/N)`. With
a single interferer (`M = 2`) it is `√(2/L)`; for two independent random unit
codes in `L` dimensions the inner product has standard deviation `1/√L`.

Delta: the project asserts crosstalk "roughly `1/√L`" for random codes; **the
published scaling is `√(M/L)` and the project's figure is its `M = O(1)`
special case.** The difference is that the project's number has no dependence
on the number of simultaneous hop streams, which is the only parameter that
matters once more than two streams share the bus.

### E10 `[V-t]` — Plate, "Holographic reduced representations", *IEEE Trans. Neural Networks* **6**(3), 623–641 (1995), DOI 10.1109/72.377968

Crossref-verified. The origin of the circular-convolution binding of E8 Eq. (10)
and of the capacity analysis of superposition memories that E8 §2.4 credits to
Plate (1994, 2003). Full text owed (U1).

Delta: the founding paper of the construction X₂₉ proposes to reinvent;
**zero.**

### E11 `[V]` — Alam, Oberle, Raff, Biderman, Oates, Holt, "A Walsh Hadamard Derived Linear Vector Symbolic Architecture", arXiv:2410.22669

Full PDF read this session. The single most important entry in this file for
the third verdict question, because it is the Walsh-Hadamard VSA the project's
code-multiplexing plan describes.

The HRR Fourier identity, Table 1, verbatim:

```
HRR:    B(x,y) = F⁻¹(F(x) ⊙ F(y))      B*(x,y) = F⁻¹(F(x) ⊘ F(y))    x_i ~ N(0, 1/d)
MAP-C:  B(x,y) = x ⊙ y                 B*(x,y) = x ⊙ y               x_i ~ U(−1,1)
MAP-B:  B(x,y) = x ⊙ y                 B*(x,y) = x ⊙ y               x_i ∈ {−1,1}
```

The Walsh–Hadamard construction and orthogonality, Eq. (1) and Lemma 3.1:

```
H₁ = [1],   H₂ = [[1, 1], [1, −1]],   H_{2n} = [[H_{2n−1}, H_{2n−1}], [H_{2n−1}, −H_{2n−1}]]

H(Hx) = d·x        (Lemma 3.1)   ⇒  H Hᵀ = d I  ⇒  rows mutually orthogonal
```

Binding in the Hadamard domain, Eq. (2), superposition of `ρ` pairs, Eq. (3):

```
B(x,y) = (1/d) · H(Hx ⊙ Hy)                (Eq. 2)
χ_ρ = Σ_{i=1}^{ρ} B(x_i, y_i)              (Eq. 3)
```

Theorem 3.1 (Inverse Theorem), stated verbatim in substance:

```
B*(B(x₁,y₁) + … + B(x_ρ,y_ρ), y_i†) = { x_i           if ρ = 1
                                       { x_i + η°_i    if ρ > 1
```

with the proof exhibiting the noise explicitly,
`η°_i = (1/d)·H( (1/H y_i) ⊙ Σ_{j≠i} (H x_j ⊙ H y_j) )`, and the projected
variant giving `η^π_i = Σ_{j≠i} x_j y_j / y_i` (Eq. 5). The paper's own
statement of necessity: a noise component "must exist whenever `ρ ≥ 2` items
are bound together **without expanding the dimension `d`**".

Delta: X₂₉ asserts crosstalk exactly `0` for Walsh codes; E11 builds a VSA out
of Walsh–Hadamard and **proves a non-zero crosstalk term for every `ρ ≥ 2` at
fixed `d`.** The two are reconciled only by the payload type: the project's
zero is correct when each code carries a **scalar** and the correlator spans
the **full** code length, in which case `⟨c_j, Σ_k c_k s_k⟩ = L s_j` exactly by
Lemma 3.1; it is false when each code carries a vector payload bound
elementwise, which is the case E11 covers and the case a multi-dimensional hop
stream actually is. **The difference is not zero; it is a payload-type
condition the project has not stated.**

Derived here, and short enough to check: in the scalar case the multiplexer is
`r = (1/√L) Σ_{k≤K} c_k s_k` with `c_k` Walsh rows and `K ≤ L`, which is an
orthogonal change of basis on an at-most-`L`-dimensional signal. Exact-zero
crosstalk in that regime is therefore not a capacity gain over allocating `K`
disjoint coordinates — it is a rotation of the same allocation, and buys
robustness to coordinate-wise corruption rather than bandwidth. Any bandwidth
claim requires `K · (payload dim) > L`, which is exactly the `ρ ≥ 2`-at-fixed-`d`
regime E11 proves is noisy.

### E12 `[V]` — Dhayalkar, "Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning", arXiv:2512.14709

PDF read this session. A position paper that interprets "self-attention and
residual streams as implementing an approximate Vector Symbolic Architecture",
where "queries and keys define role spaces, values encode fillers, attention
weights perform soft unbinding, and **residual connections realize
superposition of many bound structures**", and lists the resulting limitation
as "capacity–interference trade-offs". Carries no measured crosstalk numbers
and no new equations.

Delta: the framing X₂₉ proposes for the residual stream is published, current,
and explicit; **the difference is that this entry asserts the analogy and
measures nothing, so it forecloses the framing as novel without foreclosing a
measurement.**

### E13 `[U]` — Kleyko, Rachkovskij, Osipov, Rahimi, Part II, arXiv:2112.15424

Downloaded this session, not read. Owed for the applications and the extended
capacity results (U2).

---

## Spread spectrum, CDMA, matched filtering, Walsh–Hadamard codes

### E14 `[V]` — Tan, Rasmussen, "Nonlinear MMSE Multiuser Detection Based on Multivariate Gaussian Approximation", arXiv:cs/0502063

PDF read this session. The synchronous-CDMA channel model, Eq. (1), quoted
with the source's own conditions:

```
r = [s₁, …, s_k, …, s_K] d + n = S d + n              (Eq. 1)
```

"assume a **symbol-synchronous** CDMA system with `K` users, binary data
symbols and binary spreading with processing gain `N`", `S ∈ {±1/√N}^{N×K}`,
`d ∈ {±1}^K`, `n` zero-mean AWGN with covariance `σ² I`. The matched-filter
bank output is `Sᵀ r = Sᵀ S d + Sᵀ n`, so the multiple-access interference on
user `j` is `Σ_{k≠j} ⟨s_j, s_k⟩ d_k`, exactly zero if and only if the columns
of `S` are mutually orthogonal.

Delta: the model states the four conditions under which Walsh crosstalk is
exactly zero — symbol-synchronous, a single propagation path (`S` is the
transmitted spreading matrix, unmodified by a channel), a correlator over the
full processing gain `N`, and column orthogonality. Equal power is *not*
required for the orthogonal case; it becomes required the moment orthogonality
is imperfect, which is the near-far problem. **The difference from a
transformer residual stream is that the residual stream satisfies none of
"single path", "full-length correlator", or "linear matched-filter readout":**
downstream reads are arbitrary learned projections onto 64–128-dimensional
slices (E17), layer normalization rescales per token, and a causal mask makes
the effective code support position-dependent.

### E15 `[V-t]` — Pedersen, Mogensen, "Analysis and results for the orthogonality factor in WCDMA downlinks", *IEEE VTC Spring 2002*, vol. 1, 100–104, DOI 10.1109/VTC.2002.1002672

Crossref-verified. The engineering existence proof for the failure mode: "the
presence of multipaths leads to a **loss of orthogonality** between signals
transmitted simultaneously on a WCDMA downlink", quantified by a named
parameter, the **orthogonality factor**, which enters the link SINR equation
and "depends largely on the power delay profile of the multipath channel".

Delta: a whole named parameter exists in deployed CDMA solely to account for
Walsh crosstalk not being zero once the ideal conditions are relaxed. X₂₉'s
"crosstalk exactly 0" is the value that parameter takes at its boundary case.
**The difference is that the deployed systems that actually use Walsh codes
budget for the non-zero case, and X₂₉ currently budgets for the zero.**

### E16 `[U]` — Viterbi, *CDMA: Principles of Spread Spectrum Communication*, Addison-Wesley (1995), ISBN 978-0-201-63374-4; Verdú, *Multiuser Detection*, Cambridge (1998), ISBN 978-0-521-59373-1

Owed (U3, U4). Verdú is the textbook home of the linear MMSE multiuser
detector, `d̂ = (SᵀS + σ²A⁻²)⁻¹ Sᵀ r`, which is the same Wiener object as E1
Eq. (50) specialised to the CDMA channel and which collapses to a scalar
per-stream gain `S/(S+N)` exactly when `SᵀS = I`. That collapse is the sentence
that unifies lineages 1, 3 and 4, and it is not quoted here from a fetched
source.

---

## Communications equalization inside neural architectures

The contract's premise, "the comms-in-ML neighborhood is not empty", is
**confirmed**. What is occupied:

### E17 `[V]` — Elhage, Nanda, Olsson, et al., "A Mathematical Framework for Transformer Circuits", *Transformer Circuits Thread* (2021)

Page fetched and read this session. The residual-stream-as-channel framing,
verbatim: "We generally think of the residual stream as a **communication
channel**, since it doesn't do any processing itself and all layers communicate
through it." Under the section heading "Subspaces and Residual Stream
Bandwidth": "layers can send different information to different layers by
storing it in different subspaces… every individual head operates on
comparatively small subspaces (often 64 or 128 dimensions), and can very easily
write to completely disjoint subspaces and not interact… **dimensions of the
residual stream become something like 'memory' or 'bandwidth'**", and "There
are generally far more 'computational dimensions'… than the residual stream has
dimensions", so layers are "somehow communicating in **superposition**".

Delta: the entire vocabulary X₂₉ proposes to import — channel, bandwidth,
disjoint subspaces, superposition, interference — is already the working
vocabulary for this exact object, published in 2021. **The difference is zero
for the framing.** What is not in E17 is any *equalizer*: the framing is
diagnostic, not corrective, and no per-mode gain is applied anywhere in it.

### E18 `[V]` — Elhage, Hume, Olsson, et al., "Toy Models of Superposition", *Transformer Circuits Thread* (2022)

Page fetched and read this session. The model and its loss:

```
L = ∫_x ‖ I ( x − ReLU(Wᵀ W x + b) ) ‖² dp(x)
```

with `x_i = 0` with probability `S`. The interference statement, verbatim: in a
model without superposition `(WᵀW)₀ = (1, 0, 0, 0, …)`, but with superposition
"it's something like `(WᵀW)₀ = (1, ε, −ε, ε, …)`. The `ε` entries (which are
solely an artifact of superposition '**interference**')". The capacity
justification, verbatim: "Although it's only possible to have `n` orthogonal
vectors in an `n`-dimensional space, it's possible to have `exp(n)` many
'almost orthogonal' vectors in high-dimensional spaces. See the
Johnson–Lindenstrauss lemma."

Delta: this is the crosstalk analysis of lineage 2 rediscovered inside a neural
network, with the same `ε` off-diagonal interference term and the same
almost-orthogonality capacity argument; **the difference is zero for the
analysis and non-zero for the remedy** — E18 relies on the ReLU to filter
interference nonlinearly and never applies a linear per-mode gain.

### E19 `[V]` — Murahari, Jimenez, Yang, Narasimhan, "DataMUX: Data Multiplexing for Neural Networks", arXiv:2202.09318 (NeurIPS 2022)

Full PDF read this session. The multiplexer, Eq. (1), and the demultiplexer,
Eq. (2):

```
x^{1:N} = Φ(x¹,…,x^N) = (1/N) Σ_{i=1}^{N} φ_i(x^i)          (Eq. 1)
h^i = ϑ_i(h^{1:N}),  ∀ i ∈ [1,…,N]                          (Eq. 2)
```

with `φ_i` chosen as either "a linear projection with a fixed **random
orthogonal matrix**" or "the **Hadamard product** with a fixed Gaussian random
vector… equivalent to a linear map using a diagonal matrix", the stated purpose
being to "map instances at different indices into distinguishable regions and
consequently **reduce interference** between their representations". Reported:
20×/40× multiplexing on Transformers with `< 2%` / `< 4%` absolute drops on
MNLI, and a stated "theoretical construction for multiplexing in self-attention
networks".

Delta: X₂₉ proposes to modulate several hop streams by codes so that they share
one residual stream and can be despread; DataMUX modulates several *inputs* by
codes so that they share one representation and are demultiplexed by a learned
head. **The difference is what is multiplexed — separate token streams versus
separate computations on the same stream — and not the mechanism**, which is
Eq. (1) with a diagonal `±`-code special case in both.

### E20 `[V]` — Cheung, Terekhov, Chen, Agrawal, Olshausen, "Superposition of many models into one", arXiv:1902.05522 (NeurIPS 2019)

Full PDF read this session. Retrieval from superposed parameters, Eq. (2), and
the network form, Eq. (7):

```
Ŵ_k = Σ_i W_i (C_i⁻¹ C_k)                      (Eq. 2)
x^{(l+1)} = g( W^{(l)} ( c(k)^{(l)} ⊙ x^{(l)} ) )   (Eq. 7)
```

with "Binary Superposition" defined by constraining the phase to `φ_j(k) ∈
{0, π}`, so that "the context vectors become `c(k)_j ∈ {−1, 1}`", and the
retrieval error stated as `Ŵ_k x = W_k x + ε` with a per-context-family
analysis of `ε`.

Delta: this is elementwise `±1` code modulation of a neural network's internal
activations, with an explicit interference term, published in 2019. X₂₉'s code
multiplexing is Eq. (7) with Walsh rows substituted for random `±1` contexts.
**The difference is the code family and nothing else.** The substitution is
also not obviously an improvement: Walsh rows are closed under elementwise
product, so `c_i ⊙ c_j` is another Walsh row rather than a fresh random vector,
which concentrates the interference on specific code indices instead of
spreading it.

### E21 `[V-t]` — Alam, Raff, Biderman, Oates, Holt, "Recasting Self-Attention with Holographic Reduced Representations", arXiv:2305.19534 (ICML 2023)

Abstract read this session. Replaces the softmax attention kernel with HRR
binding, performing "the same high-level strategy of the standard
self-attention: a set of queries matching against a set of keys, and returning
a weighted response of the values for each key", at `O(T H log H)` time.

Delta: a transformer whose attention *is* a binding/unbinding pair, published
at ICML. **The difference is that Hrrformer replaces attention with binding
while X₂₉ leaves attention in place and codes the hop streams around it** — a
composition difference, and the only kind X₂₉ may claim here.

### The negative result

Searched this session and returning nothing: MMSE/Wiener equalization applied
to a trained network's *internal* representations as a corrective stage;
common-mode rejection as a named operation on neural activations; CDMA-style
multiplexing of a residual stream's *hop streams* (as opposed to its inputs or
its task-specific parameters). The neighbourhood is occupied at the framing
level (E17, E18, E12), the input-multiplexing level (E19), the
parameter-multiplexing level (E20) and the attention-replacement level (E21),
and appears unoccupied at the level of "apply per-eigenmode MMSE gains to a
specific hop's contribution before it is added to the stream". That gap is
narrow and it is the only place a composition claim can stand.

---

## Mean-centering and rank-1 deflation in deep learning

### E22 `[V]` — Mu, Viswanath, "All-but-the-Top: Simple and Effective Postprocessing for Word Representations", arXiv:1702.01417 (ICLR 2018)

Full PDF read this session. Algorithm 1, quoted verbatim:

```
Input: word representations {v(w), w ∈ V}, threshold D
1  µ ← (1/|V|) Σ_{w∈V} v(w) ;   ṽ(w) ← v(w) − µ
2  u₁,…,u_d ← PCA({ṽ(w), w ∈ V})
3  v′(w) ← ṽ(w) − Σ_{i=1}^{D} (u_iᵀ v(w)) u_i
Output: v′(w)
```

The observation motivating it, verbatim: word vectors "share a large common
vector (with norm up to a half of the average norm of word vector)", and "after
removing the common mean vector, the representations are far from isotropic —
indeed, much of the energy of most word vectors is contained in a very low
dimensional subspace (say, **8 dimensions out of 300**)". The rule of thumb for
the only hyperparameter: `D ≈ d/10`.

Delta: X₂₉ removes a rank-1 common mode from a representation and keeps the
remainder; Algorithm 1 removes the mean *and* the top `D` principal components
from a representation and keeps the remainder, and does so as a published,
named, one-hyperparameter postprocessing step with the isotropy justification
already worked out. **The difference is zero for deflation, and X₂₉ is strictly
the `D = 0` special case of a published algorithm** (mean removal only). The
delta sentence in any X₂₉ writeup must say so.

Note also the 8-of-300 figure: it is the published answer to "does a small
number of dominant directions carry most of the variance in learned
representations", and it is yes.

### E23 `[V]` — Dong, Cordonnier, Loukas, "Attention is not all you need: pure attention loses rank doubly exponentially with depth", arXiv:2103.03404 (ICML 2021)

Full PDF read this session. The decomposition that is X₂₉'s decomposition:

```
res(X) = X − 1 xᵀ ,      where x = argmin_x ‖X − 1 xᵀ‖
```

path decomposition, Theorem 2.1:

```
SAN(X) = Σ_{path ∈ [H]^L} P_path X W_path + 1 bᵀ                 (Eq. 1)
```

and the convergence result, Theorem 2.2 (simplified):

```
‖res(SAN(X))‖_{1,∞} ≤ ( 4γβ / √d_qk )^{(3^L − 1)/2} · ‖res(X)‖_{1,∞}^{3^L}   (Eq. 2)
```

"which amounts to a doubly exponential convergence to a rank-1 matrix", with
skip connections and MLPs identified as what "stop the output from
degeneration".

Delta: the split of an attention output into a rank-1 common mode `1 xᵀ` plus a
differential remainder `res(X)` is Dong et al.'s working definition, published
at ICML 2021 for the exact operator X₂₉ is decomposing. **The difference is
zero.** X₂₉'s further step — dropping the common mode rather than merely
measuring it — is E22's Algorithm 1 line 1 applied to that object.

### E24 `[V]` — Shi, Gao, Xu, Liang, Li, Kong, Lee, Kwok, "Revisiting Over-smoothing in BERT from the Perspective of Graph", arXiv:2202.08625 (ICLR 2022)

Full PDF read this session. The bridge that makes lineage 1 apply to lineage 5:
"Intuitively, the **self-attention matrix can be seen as a normalized adjacent
matrix** of a corresponding graph." Definition 1, verbatim:

```
M := { Y ∈ R^{n×d} | Y = e C, C ∈ R^{1×d} },   e = [1,1,…,1]ᵀ ∈ R^{n×1}
d_M(H) := min_{Y ∈ M} ‖H − Y‖_F
```

Delta: `M` is the rank-1 common mode subspace by another name, `d_M(H)` is the
differential remainder's norm, and the paper's remedy is hierarchical fusion of
layer outputs rather than deflation. **The difference is zero for the
decomposition and non-zero for the remedy** — nobody in this lineage subtracts
the mode and continues; they either measure the distance to it or add a skip
path that avoids collapsing into it.

### E25 `[V-t]` — Ethayarajh, "How Contextual are Contextualized Word Representations?", arXiv:1909.00512 (EMNLP 2019)

Abstract read. "the contextualized representations of all words are **not
isotropic in any layer** of the contextualizing model."

### E26 `[V-t]` — Timkey, van Schijndel, "All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality", arXiv:2109.04404 (EMNLP 2021)

Abstract read. "a small number of **rogue dimensions, often just 1–3**,
dominate these measures", with "a striking mismatch between the dimensions that
dominate similarity measures and those which are important to the behavior of
the model", and simple postprocessing offered as the fix.

Delta on E25/E26: the premise that a handful of directions dominate a learned
representation, and that removing them helps, is established and measured.
**The difference is zero**, and E26 adds the warning X₂₉ needs: the dominant
directions are not necessarily the behaviourally important ones, so removing
the common mode may improve a similarity readout without improving the model.

### E27 `[V-t]` — Gao, He, Tan, Qin, Wang, Liu, "Representation Degeneration Problem in Training Natural Language Generation Models", arXiv:1907.12009 (ICLR 2019)

Abstract read. Learned embeddings "degenerate and be distributed into a narrow
cone", with a regularizer proposed to counter it.

### E28 `[V-t]` — Su, Cao, Liu, Ou, "Whitening Sentence Representations for Better Semantics and Faster Retrieval", arXiv:2103.15316

Abstract read. "the **whitening** operation in traditional machine learning can
similarly enhance the isotropy of sentence representations", with a dimension
reduction as a bonus.

### E29 `[V-t]` — Zhao, Akoglu, "PairNorm: Tackling Oversmoothing in GNNs", arXiv:1909.12223 (ICLR 2020)

Abstract read. A normalization layer "based on a careful analysis of the graph
convolution operator, which prevents all node embeddings from becoming too
similar", with no added parameters.

Delta on E27/E28/E29: centering, whitening and anti-collapse normalization are
routine practice under three separate names in three separate literatures.
**The difference from X₂₉'s deflation is zero for the operation.** If X₂₉ ships
a deflation stage, the honest description is "graph high-pass filtering /
mean deflation, applied to the second hop's contribution", not a new mechanism.

---

## Arithmetic control on the `0.7550` figure

Computed this session. `a` is a row-stochastic `n × n` softmax attention
matrix; `frac(M) := ‖1 cᵀ‖²_F / ‖M‖²_F` with `c` the column means of `M`, i.e.
the fraction of `M`'s energy in the rank-1 common mode of E23/E24.

**Structural fact, no simulation needed.** A softmax attention matrix is row
stochastic, `a 1 = 1`, hence `(a @ a) 1 = 1`: the constant vector is an exact
right eigenvector of the second hop with eigenvalue exactly `1`, and by
Perron–Frobenius on a positive matrix it is the Perron eigenvector. Verified
numerically at `n = 64`, logits `~ N(0, 0.125²)`:
`max|(a@a)1 − 1| = 4.44 × 10⁻¹⁶`, and the next eigenvalues are
`|λ₂..λ₄| = 2.24 × 10⁻⁴, 2.21 × 10⁻⁴, 2.21 × 10⁻⁴`. **The common mode is not a
learned pathology; it is the Perron mode of a row-stochastic operator, present
by construction at every temperature.**

**Null model for the energy fraction.** `n = 64`, i.i.d. Gaussian logits, no
causal mask, `frac(a @ a)` over 20 seeds:

| logit sd | `frac(a @ a)` |
|---|---|
| 1.2 | `0.9152 ± 0.0199` |
| 1.4 | `0.8252 ± 0.0400` |
| 1.5 | `0.7680 ± 0.0496` |
| 1.6 | `0.7061 ± 0.0571` |
| 1.8 | `0.5811 ± 0.0638` |

Single-seed sweep over a wider range, same construction: `0.9988` at sd 0.5,
`0.9645` at 1, `0.4798` at 2, `0.1310` at 4, `0.0767` at 8, `0.0499` at 16.

**Consequence.** `frac(a @ a)` is a monotone decreasing function of attention
sharpness and spans the full `[1/n, 1]` range under a null with no learned
structure whatsoever. A random attention matrix at logit `sd ≈ 1.51` reproduces
the project's measured `0.7550`. **The `0.7550` figure therefore carries no
information about the trained arm until it is reported against a
sharpness-matched null** — the same discipline E24 applies when it reports
`d_M(H)` per layer rather than as a single number. The same applies to the
`0.4899` figure for the 8-of-64-column partial routing: sub-selecting columns
changes the effective `n` and hence the null, so `0.4899 < 0.7550` is not by
itself evidence that the routing filtered the common mode.

Caveats on the null: Gaussian logits with no causal mask, no positional
structure, no head-specific low-rank `W_QK`, and `n = 64` fixed. A
sharpness-matched null on the actual bed replaces all four.

---

## Owed

`[U]` entries, and why each is still owed.

**U1 — Plate, *Holographic Reduced Representation: Distributed Representation
for Cognitive Structures*, CSLI (2003), ISBN 978-1-57586-430-3, and the 1994
thesis.** Owed: the original capacity and crosstalk-variance derivations that
E8 §2.4 credits here. E9's `s = √(N/M)` is the modern restatement; the original
constants and their conditions are not fetched.

**U2 — Kleyko et al. Part II, arXiv:2112.15424.** Downloaded this session, not
read. Owed for the applications survey and the extended capacity results, which
is where any prior use of *Walsh* codes specifically as a VSA code family would
be catalogued.

**U3 — Verdú, *Multiuser Detection*, Cambridge (1998), ISBN
978-0-521-59373-1.** Owed: the linear MMSE multiuser detector in closed form,
`d̂ = (SᵀS + σ²A⁻²)⁻¹ Sᵀ r`, and the proof that it reduces to a per-stream
scalar gain under `SᵀS = I`. That reduction is the single sentence that makes
X₂₉'s Wiener stage and its code stage the same object, and it is asserted here
from memory rather than fetched.

**U4 — Viterbi, *CDMA: Principles of Spread Spectrum Communication*,
Addison-Wesley (1995), ISBN 978-0-201-63374-4.** Owed: the spreading and
despreading equations in their textbook form and the explicit statement of the
equal-power/near-far condition, which E14's Eq. (1) implies but does not state.

**U5 — Mehta, Molisch, Zhang, "Analysis and results for the orthogonality
factor in WCDMA downlinks", *IEEE Trans. Wireless Commun.* **2**(6) (2003).**
The journal version of E15; Crossref returned the VTC conference version only,
so the journal DOI is unresolved. Owed for the closed-form orthogonality factor
as a function of the power-delay profile.

**U6 — Hotelling deflation and its modern treatment.** Hotelling (1933), DOI
10.1037/h0071325, and Mackey, "Deflation Methods for Sparse PCA", NeurIPS 2008.
Owed: the statement that rank-1 deflation `A ← A − λ₁ v₁ v₁ᵀ` is a named
century-old operation in numerical linear algebra, at equation level. E22
supplies the deep-learning version; the linear-algebra ancestry is asserted
here without a fetched source.

**U7 — A source that names the DC-attenuating graph filter.** E2 names the DC
component and E3 shows `u₀` being dropped as routine, but neither gives the
operation `h(λ₀)=0, h(λ_{k>0})=1` a proper name beyond "ideal high-pass graph
filter". Owed if the project intends to write "this is the standard X" with a
specific X.

**Counts: `[V]` 17 · `[V-t]` 11 · `[U]` 7.**
`[V]`: E1, E2, E3, E4, E8, E9, E11, E12, E14, E17, E18, E19, E20, E22, E23,
E24, and the arithmetic control section's computations.
`[V-t]`: E5, E6, E7, E10, E15, E21, E25, E26, E27, E28, E29.
`[U]`: U1–U7 (E13 and E16 are the entry-level pointers to U2 and U3/U4).

---

## Claim shape

**Deflation.** The project may claim that it applies an ideal high-pass graph
filter (equivalently, rank-1 mean deflation; equivalently, E22 Algorithm 1 at
`D = 0`) to the second attention hop's contribution to a residual stream, and
report the measured effect on the arm's read against a sharpness-matched null —
and it may not claim the deflation, the decomposition `res(X) = X − 1xᵀ`, the
subspace `M = {eC}`, or the observation that a few dominant directions carry
most of the energy, all four of which are published (E2, E3, E22, E23, E24).

**Wiener equalizer.** The project may claim that it evaluates the graph Wiener
filter `h̃(λ) = σ²(λ)/(σ²(λ)+σ²ₙ(λ))` — E1 Eq. (51), on the books since E5 —
on the eigenmodes of a specific learned operator `a @ a` inside a trained arm,
and report what per-mode SNR estimates it uses and how they are estimated; it
may not claim the gain formula, and it must state the graph-stationarity
assumption (E6) that makes a diagonal gain vector the MMSE solution rather than
E1 Eq. (50)'s full matrix.

**Code multiplexing.** The project may claim only the composition — Walsh-code
modulation of hop streams stacked on top of the deflation and the equalizer, in
a residual stream, with a measured read — because the modulation itself is E20
Eq. (7) with a different code family, the multiplexing is E19 Eq. (1), the
algebra is the VSA construction of E8/E11, and the framing is E17; and it may
not write "crosstalk exactly 0" without stating in the same sentence that the
zero holds only for scalar payloads under a full-length correlator, that E11
Theorem 3.1 proves a non-zero `η` for `ρ ≥ 2` vector payloads at fixed `d`, and
that the published interference scaling is `√(M/L)` and not `1/√L`.
