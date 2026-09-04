# Prior-art sweep — lineage: linear recurrence / semiseparable / delta rule / resolvent reads

Planet: JUPITER-S2. Date: 2026-09-03. Bib: `bib_linrec.bib` (51 entries, same directory).
Repository root read at HEAD `207e7b9`; scratch brief `BRIEF.md` §1 supplies the shape.

Evidence classes used below: `RUN` (executed this session), `READ path:line`, `CITED [V]`
(abs page or full text fetched this session, title matched), `CITED [U]` (search-index
snippet or memory only), `DERIVED` (steps written out). No number is given without one.

## 0. The question this sweep answers

The record proved that its corner 3 — the path product `G_ij = Π_{k=j+1}^{i} a_k`
(`READ ceq/arm_smprime.py:144-171`) — is the resolvent `(I − A)^{-1}` of the sub-diagonal
chain, entrywise exact (`BRIEF.md` I2, `RUN` by the coordinator, max abs `0.0`). The shape
generalises that corner to `z = (I − γP)^{-1} V`, `O = P z`, with `P` the causal
row-stochastic operator of the three-corner family, `γ ∈ [0, 1)` learnable, and absorbing
constraint sets `𝒜_k` on which the rows of `P` are identity, giving the committor read
`q^{(k)} = (I − Q)^{-1} R_k 1` (`BRIEF.md` §1). The question put to this planet: where
exactly does that shape leave SSD / DeltaNet territory; is a dense content-dependent `P` with
a softmax-shaped normaliser and absorbing rows already in this lineage; and which kernel
pattern is the occupied cost path.

The one-paragraph answer, argued in §3: the shape leaves the semiseparable lineage at the
softmax normaliser. Every operator in the SSD / DeltaNet family is `N`-semiseparable for a
fixed state size `N` (`CITED [V]` Dao & Gu 2024, Def. 3.1 / Thm 3.5), and Hu et al. 2025
prove that a row-softmax over `QKᵀ` has no finite-`N` state-space dual because its rank is
`T` even when `QKᵀ` has rank 1 (`CITED [V]`, arXiv:2510.04944). The chunked-WY cost path
(`T = (I + tril(diag(β)KKᵀ, −1))^{-1} diag(β)` solved by forward substitution per chunk,
`CITED [V]` Yang et al. 2024 Eq. 10) is therefore closed to `(I − γP)^{-1}` with softmax `P`:
there is no `d × d` state to carry across chunks, and the cost is the full `s × s`
triangular solve (`BRIEF.md` I5, `≈ s²d`). What the shape adds beyond that is *not* the
resolvent read — that is occupied (§2.2, §2.3) — it is the causal exact solve with a
bitwise-softmax corner at `γ = 0`, the absorbing constraint sets with the committor as the
read, and the interventional re-solve. The largest single prior-art fact for the coordinator
is InfSA (arXiv:2603.00175, `CITED [V]`), flagged as a SURPRISE in §4.

## 1. Queries run (seeds plus 18 widening searches)

| # | query (verbatim, abbreviated where long) | hits used |
|---|---|---|
| S | eleven seed identifiers from the prompt, each abs page fetched | §2.1 |
| Q1 | `attention "resolvent" "(I - γP)^{-1}" transformer successor representation attention layer` | RL successor-representation papers only; no attention layer. NOT FOUND N4 |
| Q2 | `"semiseparable" attention content-dependent dense mask beyond 1-semiseparable "state space duality" generalization` | Hu et al. 2510.04944 |
| Q3 | `"test-time regression" unifying linear attention DeltaNet chunkwise WY "UT transform"` | Wang 2501.12352, EFLA 2512.12602, TTT-KV 2602.21204, Preconditioned DeltaNet 2604.21100 |
| Q4 | `deep equilibrium attention fixed point "absorbing" Markov chain attention committor transformer layer` | Erel 2507.17657, DEQ 1909.01377 |
| Q5 | `"DeltaProduct" OR "Gated DeltaProduct" OR "Log-Linear Attention" OR "PaTH Attention" Householder products linear RNN state-tracking` | 2502.10297, 2506.04761, 2505.16381, 2510.09389, LT2 2605.20670 |
| Q6 | `Neumann series attention "matrix inverse" "(I - A)^{-1}" multi-hop random walk PageRank attention softmax stochastic matrix` | InfSA 2603.00175, Shi & Cao 2511.23239, Vuckovic 2007.02876 |
| Q7 | `"personalized PageRank" propagation "(I - (1-α)" APPNP predict then propagate` | Gasteiger 1810.05997 |
| Q8 | `MesaNet OR "mesa-layer" attention exact least-squares solve conjugate gradient test-time optimal` | 2309.05858, 2506.05233 |
| Q9 | `Sinkformer doubly stochastic attention Sinkhorn normalization` | 2110.11773 |
| Q10 | `"implicit graph neural network" fixed point "(I - " equilibrium infinite depth EIGNN IGNN` | 2009.06211 |
| Q11 | `attention "absorbing state" OR "hitting probability" OR "committor" transformer causal attention boundary condition constraint tokens` | 2411.04990, 2603.11487, 2601.15158; no attention-level absorbing constraint set. NOT FOUND N2 |
| Q12 | `Titans "learning to memorize at test time" OR xLSTM "mLSTM" OR Longhorn OR "Gated Slot Attention"` | 2501.00663, 2407.14207, 2409.07146, 2405.04517 |
| Q13 | `"random walk with restart" OR "Katz" attention transformer multi-hop "attention matrix" diffusion "graph transformer" learned damping` | InfSA again, Diffuser 2210.11794, Lin 2607.10677 |
| Q14 | `"flash-linear-attention" chunkwise parallel kernel DeltaNet WY "UT transform" cost chunk size triangular solve` | FLA repository, TFLA 2503.14376 |
| Q15 | `attention layer "intervention" OR "do-operator" OR "counterfactual" re-solve fixed point "consequence" transformer` | 2310.20307, 2204.07258; no re-solve channel inside an attention operator. NOT FOUND N3 |
| Q16 | `"successor representation" "self-attention" OR "attention layer" transformer in-context "(I - γ"` | nothing on topic. NOT FOUND N4 |
| Q17 | `attention transformer "harmonic extension" OR "Dirichlet problem" OR "Dirichlet boundary" tokens "boundary condition" absorbing constraint set` | 2209.14977 (EIT, not a token chain). NOT FOUND N2 |
| Q18 | `"energy-based" OR "Hopfield" attention fixed point iteration "equilibrium" transformer layer "energy transformer"` | 2511.00907, 2302.07253, 2008.02217 |

Full-text fetches beyond the abs pages: DeltaNet (`arxiv.org/html/2406.06484`), SSD
(`html/2405.21060`), Gated DeltaNet (`html/2412.06464`), InfSA (`html/2603.00175`), Hu et al.
(`html/2510.04944`), RWKV-7 (`html/2503.14456`), APPNP (`ar5iv 1810.05997`), mesa-layer
(`ar5iv 2309.05858`), IGNN (`ar5iv 2009.06211`), Diffuser (`ar5iv 2210.11794`), the FLA
README. Equation-level statements below cite these; abstract-only statements say so.

## 2. Sources, one block each

Format: id — `[V]/[U]` — what it owns (the equation or mechanism, this planet's words) — what
it leaves open relative to the shape — found by.

### 2.1 Seeds

**yang-2024-deltanet** (arXiv:2406.06484, cs.LG, 10 Jun 2024) — `[V]` abs + full text.
Owns: the per-token delta rule `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, `β_t ∈ (0,1)`,
eigenvalues of the transition in `(0,1]`; and the chunkwise WY / UT-transform training form
whose per-chunk kernel is `T_[t] = (I + tril(diag(β_[t]) K_[t] K_[t]ᵀ, −1))^{-1} diag(β_[t])`,
the unit-lower-triangular inverse solved by forward substitution over the `C` tokens of a
chunk (Eq. 10 of the full text); training cost `O(LCd + Ld²)`. The paper removes the linear
attention normaliser explicitly. Leaves open: no row-stochastic `P`, no token-side resolvent
of a normalised operator, no absorbing rows, no certificate, no intervention. What it
occupies for the shape: the *primitive* of I5 — a forward-substitution triangular solve on a
token-side unit-lower-triangular system — is DeltaNet's intra-chunk kernel. The shape's I5
applies that primitive to the whole `s × s` operator because no `d × d` inter-chunk state
exists for softmax `P` (§3). Found by: seed; Q3; Q14.

**yang-2024-gateddeltanet** (arXiv:2412.06464, cs.CL, 9 Dec 2024; ICLR 2025 `[U]`) — `[V]`
abs + full text. Owns: `S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`, the scalar decay
`α_t` composed with the Householder update, and the same per-chunk
`[I + strictLower(diag(β)KKᵀ)]^{-1} diag(β)` with the cumulative-decay ratios folded in.
Leaves open: as DeltaNet; the gate `α_t` is the record's corner-3 magnitude `m_k` *on the
state side*, not a token-side normaliser. Found by: seed; Q3.

**dao-2024-ssd** (arXiv:2405.21060, cs.LG, 31 May 2024) — `[V]` abs + full text. Owns: the
1-semiseparable mask `L_ij = a_i ⋯ a_{j+1}` for `i ≥ j` (cumulative product of
input-dependent scalars `a_t ∈ [0,1]`), the definition of `N`-semiseparable (every submatrix
in the lower triangle has rank `≤ N`, Def. 3.1), and the theorem that an SSM of state size
`N` produces an `N`-SS matrix (Thm 3.5). This is the record's corner 3 exactly: the SSD mask
`L` **is** `path_product` (`READ ceq/arm_smprime.py:144-171`) with real `a_t`, and I2 says
that object is `(I − A)^{-1}` of the sub-diagonal chain. Leaves open: the full text contains
no inverse of a semiseparable matrix, no resolvent, no Neumann series, no content-dependent
dense mask beyond the scalar case, and no statement about softmax being non-semiseparable
(that is Hu et al. 2025). Found by: seed; Q2.

**yang-2023-gla** (arXiv:2312.06635, cs.LG, 11 Dec 2023) — `[V]` abs. Owns: data-dependent
per-dimension decay gates on the linear-attention state with a hardware-efficient chunked
algorithm. Leaves open: the gate is diagonal on the `d × d` state; no token-side stochastic
operator, no boundary rows. Found by: seed.

**sun-2023-retnet** (arXiv:2307.08621, cs.CL, 17 Jul 2023) — `[V]` abs. Owns: retention with
a fixed per-head exponential decay and three computation paradigms (parallel, recurrent,
chunkwise recurrent). Leaves open: the decay is content-independent, which the record's
corner 3 already generalises (`m_k` per position). Found by: seed.

**orvieto-2023-lru** (arXiv:2303.06349, cs.LG, 11 Mar 2023) — `[V]` abs. Owns: the linear
recurrent unit — linearised diagonal complex recurrence with a stable exponential
parameterisation and normalisation. Leaves open: content-independent transition; no operator
on tokens. Found by: seed.

**gu-2022-s4d** (arXiv:2206.11893, cs.LG, 23 Jun 2022) — `[V]` abs. Owns: diagonal
state-space parameterisation and initialisation. The record already notes S4D's ReLU variant
as the prior owner of closed-magnitude gates (`BRIEF.md` §3, `|Ā| = 1` on 32.93 % of a sample,
`READ`). Leaves open: as LRU. Found by: seed.

**gu-2023-mamba** (arXiv:2312.00752, cs.LG, 1 Dec 2023) — `[V]` abs. Owns: selective SSMs
whose parameters are functions of the input. Leaves open: no token-side normalised operator;
the SSD paper is its dual form. Found by: seed.

**katharopoulos-2020-linearattention** (arXiv:2006.16236, cs.LG, 29 Jun 2020) — `[V]` abs.
Owns: attention with kernel feature maps and associativity, `O(N)` autoregressive form.
This is the record's corner 2 (`β = 0, g = 0`, `READ ceq/arm_smprime.py:14`;
`lean/CEQ/V16Domain.lean:483 gate_zero_beta_zero_is_linear_attention`, `READ`). Leaves open:
nothing about resolvents or boundaries. Found by: seed.

**vaswani-2017-attention** (arXiv:1706.03762, cs.CL, 12 Jun 2017) — `[V]` abs (venue not on
the abs page, `[U]`). Owns: corner 1 (`β = 1`, row-softmax); the shape at `γ = 0` is this
operator bitwise (`BRIEF.md` I1, `RUN`). Found by: direct fetch.

**peng-2025-rwkv7** (arXiv:2503.14456, cs.CL, 18 Mar 2025) — `[V]` abs + full text. Owns:
`S_t = S_{t−1}(diag(w_t) − κ̂_tᵀ(a_t ⊙ κ̂_t)) + v_tᵀ k̃_t`, a diagonal-plus-rank-one transition
with vector-valued gating and in-context learning rates; parallel form as cumulative products
of transition matrices, not WY. Leaves open: no row-stochastic token operator, no resolvent,
no absorbing states (full text checked). Found by: seed.

**sun-2024-ttt** (arXiv:2407.04620, cs.LG, 5 Jul 2024) — `[V]` abs. Owns: the hidden state as
a model updated by a self-supervised gradient step per token. Leaves open: no token-side
operator; the "state" is a parameter vector. Found by: seed.

### 2.2 Resolvent / Neumann / PageRank reads — the occupants of `z = (I − γP)^{-1} V`

**roffo-2026-infsa** (arXiv:2603.00175, cs.CV, v1 26 Feb 2026, v5 30 Mar 2026) — `[V]` abs +
full text. Owns: *Infinite Self-Attention*: the layer is read as one diffusion step on a
content-adaptive token graph, and multi-hop interactions are accumulated as a discounted
Neumann series, `Č = (I − γA)^{-1} − I`, with `γ` a learnable per-head scalar constrained by a
sigmoid into `(0, 1/ρ(A))`. In the full text the base is **not** row-softmax: it is
`Â = [QKᵀ]_+ / (‖[QKᵀ]_+‖_F + ε)` (ReLU, Frobenius-normalised). The series is *truncated across
layers*, `S_L = Σ_{l≤L} γ^l Z^{(l)}`, not solved; complexity `O(N²d)`. Section 3.3 gives the
absorbing-chain reading: every token is transient with absorption probability
`R_i = 1 − γ Σ_j Â_ij`, and the Neumann kernel is the fundamental matrix. It links the kernel
to Katz / PageRank / eigenvector centrality and proposes a linear-time variant that
approximates the principal eigenvector. Global bound `‖S_L‖ ≤ γ/(1 − γ)` only. Experiments:
ImageNet-1K and attention-quality metrics; non-causal; no autoregressive variant. Leaves
open relative to the shape: (i) no causal / triangular structure and no exact solve (I5);
(ii) the base is not softmax, so `γ = 0` is not softmax and I1's bitwise parity is not
theirs; (iii) absorption is the *discount* (`1 − γ·rowsum`), uniform over tokens, not a
constraint set with identity rows — the committor read (I3) is absent; (iv) no per-term
truncation certificate of the form `γ^{K+1}/(1 − γ)` (I4), only the global norm bound;
(v) no intervention. Found by: Q6; Q13. **SURPRISE** — see §4.

**gasteiger-2018-appnp** (arXiv:1810.05997, cs.LG, ICLR 2019) — `[V]` abs + full text.
Owns: `Z = softmax(α(I − (1−α)Â)^{-1} H)` with `Â = D̃^{-1/2}ÃD̃^{-1/2}` fixed, and the
power-iteration approximation `Z^{(k+1)} = (1−α)ÂZ^{(k)} + αH` for `K` steps; the teleport
`α` plays the role of `1 − γ`. This is the exact resolvent read with a stochastic-like
operator, seven years old, on a *fixed graph*. Leaves open: `Â` is not content-dependent,
not causal, no absorbing constraint set (the teleport is uniform), no certificate printed.
Found by: Q7.

**feng-2022-diffuser** (arXiv:2210.11794, cs.LG, 21 Oct 2022) — `[V]` abs + full text. Owns:
attention diffusion on a sparse attention graph, `𝒜 = Σ_k θ_k A^k` with personalised-PageRank
coefficients `θ_k = α(1−α)^k`, `α = 0.1` fixed, computed by `K = 5` power-iteration steps
`Z^{(k+1)} = (1−α)AZ^{(k)} + αV`; non-causal. Leaves open: fixed decay, fixed `K`, no exact
solve, no boundary rows, no certificate. Found by: Q13.

**yuan-2025-paraformer** (arXiv:2512.14619, cs.LG, 16 Dec 2025) — `[V]` abs only. Owns: a
PageRank-enhanced attention module on graphs meant to mimic deep transformers as an
adaptive-pass filter. The record already lists it as the nearest multi-hop construction with
signed hop coefficients (`READ RESEARCH.md:46`). Leaves open: abstract does not state exact
vs truncated; graph, not sequence. Found by: record; Q6 context.

**shi-2025-randomwalks** (arXiv:2511.23239, cs.LG, 28 Nov 2025) — `[V]` abs. Owns: a
one-layer transformer trained by gradient descent predicts random walks on a circle by
selecting the predecessor token and applying a one-step transition in the value matrix.
Leaves open: one hop; this is the D-1 regime (single-location label) the paper must not race.
Found by: Q6.

### 2.3 Semiseparable theory and the unifying frames

**hu-2025-ssdtheory** (arXiv:2510.04944, cs.LG, 6 Oct 2025, rev. 23 Dec 2025) — `[V]` abs +
full text. Owns: (i) SSD extended from scalar-identity to general diagonal SSMs; (ii) matching
training-complexity lower bounds; (iii) Thm 4.1, a necessary-and-sufficient condition for an
`N`-dimensional SSM to have a 1-semiseparable masked-attention dual (block structure with at
most `N` "new columns" per diagonal block); (iv) softmax attention has no SSM dual because
`Softmax(QKᵀ)` has rank `T` even when `QKᵀ` has rank 1. This planet's reading of what (iv)
licenses: *exact* duality is impossible for row-softmax `P`; it says nothing about
approximation. Leaves open: no dense content-dependent non-semiseparable masks, no inverse of
a 1-SS matrix. This is the theorem that locates the shape's exit from the lineage (§3).
Found by: Q2.

**wang-2025-testtimeregression** (arXiv:2501.12352, cs.LG, 21 Jan 2025) — `[V]` abs. Owns:
the test-time-regression frame — associative recall as memorisation-then-retrieval regression
— from which linear attention, SSMs, fast-weight programmers, online learners and softmax
attention are derived by three design choices. Leaves open: the frame is about the `d × d`
regressor; no token-side resolvent, no boundaries. Found by: Q3.

**sieber-2025-coefficientdynamics** (arXiv:2510.09389, cs.LG, 10 Oct 2025) — `[V]` abs.
Owns: linear-combination coefficients as outputs of autonomous linear dynamical systems
driven by impulses, unifying transformers, SSMs, gated linear RNNs and softmax. Leaves open:
abstract does not mention resolvents, Neumann series or boundaries. Found by: Q5.

### 2.4 Delta-rule family: transitions and kernels

**siems-2025-deltaproduct** (arXiv:2502.10297, cs.LG, NeurIPS 2025) — `[V]` abs. Owns:
`n_h` Householder steps per token, diagonal-plus-rank-`n_h` transitions, state-tracking in
finite precision. Leaves open: state-side transitions; the record's corner 3 is a *scalar*
abelian product and must not be described in state-tracking vocabulary (P-7). Found by: Q5.

**yang-2025-path** (arXiv:2505.16381, cs.CL, 22 May 2025) — `[V]` abs. Owns: accumulated
data-dependent Householder transforms as position encoding; the record already assigns VGPE
to it (`READ PRIOR_ART.md` §5 per `BRIEF.md` §3). Found by: Q5.

**guo-2025-loglinear** (arXiv:2506.04761, cs.LG, 5 Jun 2025) — `[V]` abs. Owns: a
logarithmically growing set of hidden states (hierarchical) giving log-linear cost;
instantiated on Mamba-2 and Gated DeltaNet. Leaves open: this is the occupied
*multi-resolution* cost path in the lineage; the record's `ceq/multizoom.py` fine-near /
coarse-far schedule (`READ BRIEF.md` §4) sits beside it, not above it. Found by: Q5.

**lei-2025-efla** (arXiv:2512.12602, cs.LG, 14 Dec 2025) — `[V]` abs. Owns: exact
continuous-time solution of the delta-rule dynamics, rank-1 structure, WY/UT-compatible.
Found by: Q3.

**tumma-2026-preconditioneddeltanet** (arXiv:2604.21100, cs.LG, 22 Apr 2026) — `[V]` abs.
Owns: curvature-preconditioned delta rule with chunkwise parallel algorithms and a diagonal
approximation. Found by: Q3.

**liu-2026-tttkvbinding** (arXiv:2602.21204, cs.LG, 24 Feb 2026) — `[V]` abs. Owns: TTT with
KV binding rewritten as a learned linear attention. Found by: Q3.

**yang-2024-fla** (GitHub `fla-org/flash-linear-attention`, software, 2024) — `[V]` README.
Owns: the Triton kernels for RetNet, GLA, Mamba2, DeltaNet, Gated DeltaNet, RWKV7,
DeltaProduct, MesaNet, PaTH, Log-Linear Attention and others — the occupied cost path as
shipped code. Found by: Q14.

**beck-2025-tfla** (arXiv:2503.14376, cs.LG, 18 Mar 2025) — `[V]` abs. Owns: tiling within
chunks so chunk sizes can grow for linear-RNN / mLSTM kernels. Found by: Q14.

### 2.5 Test-time learners and matrix memories

**behrouz-2024-titans** (arXiv:2501.00663, cs.LG, 31 Dec 2024) — `[V]` abs. Owns: a neural
long-term memory updated by gradient descent at test time with surprise-gated momentum.
Found by: Q12.

**liu-2024-longhorn** (arXiv:2407.14207, cs.LG, 19 Jul 2024) — `[V]` abs. Owns: SSM
recurrence as the closed-form solution of an online associative-recall objective. Found by:
Q12.

**zhang-2024-gsa** (arXiv:2409.07146, cs.CL, NeurIPS 2024) — `[V]` abs. Owns: bounded-memory
slots, two gated linear attention layers linked by softmax. Found by: Q12.

**beck-2024-xlstm** (arXiv:2405.04517, cs.LG, 7 May 2024) — `[V]` abs. Owns: mLSTM — matrix
memory, exponential gating, covariance update, normaliser state. Found by: Q12.

**vonoswald-2023-mesa** (arXiv:2309.05858, cs.LG, 11 Sep 2023) — `[V]` abs + full text.
Owns: the mesa-layer, a causal layer that outputs the ridge-regression solution at each step,
`Φ̂_t = (Σ_{t'≤t} v_{t'} k_{t'}ᵀ)(Σ_{t'≤t} k_{t'} k_{t'}ᵀ + λ^{-1} I)^{-1}`, the inverse maintained
by Sherman–Morrison; sequential, not time-parallel. This is the nearest occupant of "an exact
linear solve inside the layer" — but the system is the `d × d` key Gram, not the `s × s`
token operator. Found by: Q8.

**vonoswald-2025-mesanet** (arXiv:2506.05233, cs.LG, ICLR 2026) — `[V]` abs. Owns: the
chunkwise-parallel, numerically stable mesa-layer solved by conjugate gradient to optimality
per time point. Leaves open: as above; the solve is on the key Gram. Found by: Q8.

### 2.6 Fixed-point, equilibrium and energy reads

**bai-2019-deq** (arXiv:1909.01377, cs.LG, NeurIPS 2019) — `[V]` abs. Owns: the layer output
as a root of `z = f(z, x)` found by root-finding with implicit differentiation, applied to
self-attention transformers. Leaves open: nonlinear fixed point, no closed form, no
certificate with a printed δ, no boundary rows. The honest "deeper softmax" skyline of
`BRIEF.md` §1 has this as its infinite-depth limit. Found by: Q4.

**gu-2020-ignn** (arXiv:2009.06211, cs.LG, NeurIPS 2020) — `[V]` abs + full text. Owns:
`X = φ(WXA + b_Ω(U))` on a *fixed* normalised adjacency `A`, well-posed when
`λ_pf(|Aᵀ ⊗ W|) < 1` (Perron–Frobenius), solved by iteration. Leaves open: fixed graph, not
content-dependent, nonlinear. Found by: Q10.

**ramsauer-2020-hopfield** (arXiv:2008.02217, cs.NE, 16 Jul 2020) — `[V]` abs. Owns: one
modern-Hopfield update equals softmax attention; fixed points are global averages,
metastable subset averages, or single patterns; retrieval in one step. This owns the
"metastable state" vocabulary for the author's capability (c). Leaves open: the fixed point
is of a nonlinear retrieval map, not a linear resolvent; no boundaries. Found by: Q18.

**hoover-2023-energytransformer** (arXiv:2302.07253, cs.LG, NeurIPS 2023) — `[V]` abs.
Owns: a single recurrent block minimising an energy whose gradient is attention-like; tokens
iterate to an equilibrium. Found by: Q18.

**ren-2025-intrinsicoptimizers** (arXiv:2511.00907, cs.LG, 2 Nov 2025) — `[V]` abs. Owns:
the forward pass as energy minimisation with attention variants as optimisation steps.
Found by: Q18.

**deng-2026-lt2** (arXiv:2605.20670, cs.LG, 20 May 2026) — `[V]` abs. Owns: looped
transformer layers with linear-time attention, each loop widening the receptive field at
constant cost — a candidate for the *matched-cost* skyline control. Found by: Q5.

### 2.7 Attention as a Markov chain — analyses, not layers

**erel-2025-attentionmarkovchains** (arXiv:2507.17657, cs.CV, 23 Jul 2025) — `[V]` abs.
Owns: the attention matrix read as a discrete-time Markov chain; metastable states from
eigenanalysis; TokenRank as the stationary vector; indirect attention by propagation. No
absorbing sets, no committor, no inverse; analysis of existing layers. Found by: Q4.

**ildiz-2024-attentionmarkov** (arXiv:2402.13512, cs.LG, 21 Feb 2024) — `[V]` abs. Owns:
one-layer self-attention generation as a context-conditioned Markov chain; winner-takes-all
concentration. Found by: Q16 context.

**vuckovic-2020-maththeory** (arXiv:2007.02876, stat.ML, 6 Jul 2020) — `[V]` abs. Owns: a
measure-theoretic, interacting-particle view of attention with Lipschitz results. Found by:
Q6.

**karagodin-2024-causalclustering** (arXiv:2411.04990, cs.LG, 7 Nov 2024) — `[V]` abs.
Owns: causal self-attention as a particle system converging to one cluster; meta-stable
states via the Rényi parking analogy. No absorbing construction. Found by: Q11.

**lin-2026-connectionlaplacian** (arXiv:2607.10677, cs.LG, 12 Jul 2026) — `[V]` abs. Owns:
attention as a connection walk with a non-negative walk matrix and per-edge linear
transport; conditions under which the generator is a random-walk connection Laplacian.
Abstract mentions no resolvent or boundary. Found by: Q13.

**sander-2021-sinkformers** (arXiv:2110.11773, cs.LG, AISTATS 2022) — `[V]` abs. Owns:
Sinkhorn-normalised (doubly stochastic) attention; heat-diffusion / Wasserstein-flow
reading in depth. Relevant because the shape's certificate (I4) needs a row-stochastic `P`;
doubly stochastic is a stricter normaliser already in the literature. Found by: Q9.

**ranmilo-2026-attentionsinks** (arXiv:2603.11487, cs.LG, 12 Mar 2026) — `[V]` abs. Owns:
a proof that simplex normalisation forces a content-independent anchor (the sink) to realise
a default state. The record's value-zero BOS sink (`V15Fork.Asink_*`, `BRIEF.md` §2) is that
object; the shape's absorbing rows are a different object (identity rows on a constraint
set, not a zero-value anchor) and the paper must say so. Found by: Q11.

**ranmilo-2026-outcomerl** (arXiv:2601.15158, cs.LG, 21 Jan 2026) — title `[V]`; the
"long-jump-absorbing chain / absorption-probability derivative" content is known only from
the search snippet `[U]` — the abs page does not carry those terms. Recorded as a possible
near-miss on *absorption probability as an analysed quantity in transformer training*, not
as a layer. Found by: Q11.

### 2.8 Interventional and boundary lineages, checked here for overlap only

**rohekar-2023-causalselfattention** (arXiv:2310.20307, cs.AI, NeurIPS 2023) — `[V]` abs.
Owns: self-attention as an estimator of a structural equation model over the sequence,
enabling zero-shot causal discovery without interventions. Leaves open: no `do(a)` re-solve;
no displacement read. Found by: Q15.

**melnychuk-2022-causaltransformer** (arXiv:2204.07258, cs.LG, ICML 2022) — `[V]` abs.
Owns: counterfactual outcome estimation over time under treatment sequences with three
transformer subnetworks and balanced representations. This is a *task* occupant for
capability (a), outside this lineage's operator question. Found by: Q15.

**guo-2022-boundaryvalue** (arXiv:2209.14977, cs.LG, ICLR 2023) — `[V]` abs. Owns:
harmonic extension of boundary data as a feature map for EIT with attention as a non-local
kernel; the PDE domain, not the tokens, carries the boundary. Found by: Q17.

## 3. Where the shape leaves SSD / DeltaNet territory — the argument

Step 1 (`READ`, `RUN` by coordinator). Corner 3 is `L_ij = Π_{k=j+1}^{i} a_k`
(`ceq/arm_smprime.py:144-171`), which is SSD's 1-SS mask (`CITED [V]` Dao & Gu 2024) and,
by I2, `(I − A)^{-1}` of the sub-diagonal chain. So the record's corner 3 sits *inside* the
semiseparable lineage: owner Dao & Gu 2024 for the mask, GLA / RetNet / Mamba for the gate
variants, DeltaNet / Gated DeltaNet for the rank-1 generalisation on the state side.

Step 2 (`CITED [V]`). Every operator in that lineage is `N`-SS for fixed `N` (Def. 3.1,
Thm 3.5). The chunked-WY path exists *because* the inter-chunk coupling has rank `≤ N` and
can be carried by a `d × d` (or `N × d`) state; the intra-chunk part is a `C × C`
unit-lower-triangular solve by forward substitution (DeltaNet Eq. 10).

Step 3 (`CITED [V]`, Hu et al. 2025 result (iv)). Row-softmax over `QKᵀ` has rank `T` even
for rank-1 logits; it admits no finite-`N` SSM dual. Hence `P` in the shape (β = 1 corner,
the normaliser present) is not `N`-SS for any fixed `N`, and `(I − γP)^{-1}` cannot be
chunked with a bounded state. `DERIVED`: the inverse of a non-semiseparable
lower-triangular matrix is generically dense lower-triangular, so no state pass exists; the
exact path is the full triangular solve, `≈ s²d` (`BRIEF.md` I5, `RUN` `1.8e-15` vs dense
inverse). That is the precise exit point: **the softmax normaliser on the token side**.

Step 4 (`CITED [V]`). The resolvent read itself — `(I − γP)^{-1}` with a normalised,
content-adaptive `P` and a learnable `γ` — is occupied non-causally and by truncation
(InfSA 2026), on fixed graphs exactly (PPNP 2019), and on sparse attention graphs by fixed
PPR truncation (Diffuser 2022). The absorbing-chain *reading* of the Neumann kernel is in
InfSA §3.3. What none of them has: the causal exact solve with a bitwise-softmax corner at
`γ = 0`; absorbing constraint *sets* with identity rows and the committor as the read; a
per-term truncation certificate with a printed δ; and the interventional re-solve
`Δz = z(do a) − z`.

Step 5 (`CITED [V]`). The nearest "exact linear solve inside a causal layer" is the
mesa-layer / MesaNet, and the system solved there is the `d × d` key Gram
`(KKᵀ + λ^{-1}I)^{-1}` per step — a different matrix with a different meaning (least
squares over keys), sequential or CG-chunked. The nearest "fixed point as the read" is DEQ /
IGNN / Energy Transformer / modern Hopfield — nonlinear maps, no closed form, no certificate.

Step 6 (`DERIVED`, the consequence for cost claims). Against the lineage, the shape's cost
sentence must read: *at matched sequence length `s`, the shape pays the DeltaNet intra-chunk
primitive on the whole sequence (`C = s`), because Hu et al.'s rank result removes the
inter-chunk state; the chunked-WY path is the fellow control and is cheaper by the factor
`s/C` on the token-side solve.* Any sentence claiming the shape is "linear" or "chunkable" is
false by Step 3.

## 4. Verdict

**OCCUPIED (owner named).**
- Corner 1, row-softmax `P` — Vaswani et al. 2017 `[V]`.
- Corner 2, `β = 0, g = 0` linear attention — Katharopoulos et al. 2020 `[V]`
  (`lean/CEQ/V16Domain.lean:483`, `READ`).
- Corner 3, the scalar path product = 1-SS mask = chain resolvent — Dao & Gu 2024 `[V]`
  (mask `L`), with GLA 2023 / RetNet 2023 / Mamba 2023 / S4D 2022 / LRU 2023 owning the
  gate parameterisations `[V]`.
- The chunked-WY / UT-transform cost path with a per-chunk unit-lower-triangular forward
  substitution — Yang et al. 2024 (DeltaNet) `[V]`, extended by Gated DeltaNet 2024 `[V]`,
  shipped in FLA 2024 `[V]`, tiled by TFLA 2025 `[V]`, and generalised by DeltaProduct 2025,
  EFLA 2025, Preconditioned DeltaNet 2026 `[V]`.
- The resolvent read `(I − γP)^{-1}` with a stochastic-like operator and a restart /
  discount — PPNP / APPNP 2019 `[V]` (fixed graph, exact), Diffuser 2022 `[V]` (attention
  graph, fixed PPR, `K = 5`), InfSA 2026 `[V]` (content-adaptive Frobenius-normalised base,
  learnable `γ`, layer-truncated, non-causal).
- The absorbing-Markov-chain *interpretation* of the Neumann kernel — InfSA 2026 §3.3 `[V]`
  (absorption = discount, uniform).
- The semiseparable / rank theorem that separates softmax from the lineage — Hu et al. 2025
  `[V]`.
- "Exact linear solve inside a causal layer" — mesa-layer 2023 / MesaNet 2026 `[V]`, on the
  key Gram.
- "Equilibrium as the layer output" — DEQ 2019, IGNN 2020, Energy Transformer 2023, modern
  Hopfield 2020 `[V]`, nonlinear.
- "Metastable / transient states of the attention chain" vocabulary — Ramsauer 2020, Erel
  2025, Karagodin 2024 `[V]`.
- The attention sink as a forced anchor — Ran-Milo 2026 `[V]` (the record's BOS sink).

**NEAR-MISS.**
- Causal exact resolvent read with softmax base: InfSA has the read but non-causal,
  non-softmax, truncated; PPNP has the exact read on a fixed graph. Neither has `γ = 0`
  bitwise softmax parity (I1).
- The truncation certificate: InfSA's global bound `‖S_L‖ ≤ γ/(1−γ)` is the `K = 0` case of
  I4's `γ^{K+1}/(1−γ)`; the per-`K` certificate with a planted non-stochastic negative
  (`BRIEF.md` I4, `RUN` err `119.37` vs bound `1.143`) is not in any fetched source.
- Absorption probability as an analysed quantity: Ran-Milo et al. 2026 `[U]` on the claim.
- Doubly stochastic normalisers (Sinkformers) as a stricter certificate route: not used by
  the shape but available.

**NOT FOUND (with the queries that failed).**
- N1: a causal attention layer computing `(I − γP)^{-1}V` with row-softmax `P`, learnable
  `γ`, exact triangular solve — Q1, Q6, Q13, Q14.
- N2: absorbing constraint sets as identity rows of the attention operator, with the
  committor `q = (I − Q)^{-1} R 1_B` as the layer's read — Q4, Q11, Q17.
- N3: an interventional channel inside the operator (re-solve under `do(a)`, displacement
  `Δz` as the read, `argmin_a max_k q^{(k)}(do a)`) — Q15.
- N4: the successor representation as an attention layer — Q1, Q16.
- N5: a per-term Neumann truncation certificate attached to an attention read — Q6, Q14
  (InfSA gives the global bound only).
- N6: a dense softmax-normalised `P` inside the chunked-WY cost path — Q2, Q3, Q14; closed
  by Hu et al. 2025 result (iv) rather than merely unfound.

**SURPRISE (flag loudly).** InfSA, Roffo et al., arXiv:2603.00175 (Feb–Mar 2026, cs.CV)
`[V]` occupies the *resolvent half* of the shape's public description: discounted Neumann
series over a content-adaptive attention matrix, learnable per-head `γ`, the absorbing-chain
fundamental-matrix reading, Katz / PageRank centrality, and a linear-time approximant. Any
sentence in the paper that presents "attention as `(I − γP)^{-1}V`" as new is false as of
February 2026. The delta survives only in the narrow form of §3 Step 4: causal exact solve,
softmax base with bitwise parity at `γ = 0`, constraint sets with the committor read, the
printed certificate, and the interventional re-solve. A second, smaller surprise runs the
other way: Hu et al. 2025 `[V]` is a *gift* — it proves the exit point (Step 3) that the
record had only asserted.

## 5. Proposals for the paper's prior-art section, each with the mechanism it is designed against

P1. **Cite InfSA, PPNP and Diffuser before the resolvent is named** (rule 7). Designed
against `P-3` (a stale claim never retracted — the record's `THEORY.md:22` still labels the
successor operator "new"; `READ`) and `P-10` (a source's intro cited as its theorem — cite
InfSA §3.3's construction, not its abstract).

P2. **Plant InfSA's base as a negative in the parity bind.** Run the shape's harness with
`Â = [QKᵀ]_+/‖·‖_F` at `γ = 0` and show I1's `torch.equal` fails, then with row-softmax and
show it holds. Designed against `V-24` (an identity bind whose rejection region is empty)
and `V-2` (a hand-built example where right and wrong coincide). `NOT MEASURED — needs the
Frobenius-normalised base wired into $SCRATCH/shape_identities.py`.

P3. **Price the shape against the chunked-WY fellow at matched `s`, stating the `s/C`
factor** (§3 Step 6), never against softmax alone. Designed against `M-8` (pricing every arm
at one arm's rate) and `P-8` (an upper bound stated as a price). `NOT MEASURED — needs a
DeltaNet chunk-size sweep on the certified RTX 4060`.

P4. **State Hu et al.'s result as what it licenses.** "No exact finite-state dual" — not
"cannot be approximated". Designed against `P-10` and `V-23` (a plural claim whose central
member is the counterexample: InfSA's linear-time variant *approximates* the read).

P5. **Label N1–N6 as NOT FOUND with query lists, never as "novel".** Designed against `V-7`
(a search structurally incapable of finding anything read as absence) — the 18 queries and
11 full-text fetches are the search's own coverage record.

P6. **Keep corner 3 in scalar-abelian vocabulary.** DeltaProduct / PaTH / RWKV-7 own
"state tracking" and non-abelian products; corner 3 is a scalar product with a zero gate
(`pathProd_eq_zero_iff`, `READ lean/CEQ/V16Domain.lean:129`). Designed against `P-7`
(vocabulary with no referent).

P7. **Separate the BOS sink from the absorbing rows in one sentence.** Ran-Milo 2026 owns
the sink; the shape's `𝒜_k` rows are identity rows on constraint positions, which the
telescope (`V15Fork.Asink_*`) does not have. Designed against `V-12` (a single absorbing
target makes the label constant): the paper needs `K ≥ 2` constraint sets so the committor
vector is not constant.

P8. **Name DEQ / LT2 as the matched-cost skyline, not softmax alone.** Designed against
`D-1` (racing a baseline at its proven optimum): the deeper / looped stack is the fellow
approximator on every hop bed, per `CEQ_V16_CONTRACT.md` R-SKY (`READ` per `BRIEF.md` §1).

## 6. Limits of this sweep

Full texts were fetched for 11 of 51 sources; the remainder are abs-page reads, and any
equation-level claim about them is marked as abstract-level above. Venues for DeltaNet,
SSD, GLA, LRU, S4D, Katharopoulos, Vaswani and Gated DeltaNet were not on the fetched abs
pages and are either omitted or marked `[U]` in the bib. The InfSA full-text summary was
produced by a fetch-and-summarise tool over the HTML render; the equations quoted from it
(`Č = (I − γA)^{-1} − I`, the Frobenius-normalised base, `R_i = 1 − γΣ_j Â_ij`) should be
re-read by the assembler against the PDF before they appear in the paper. Search was
US-index web search plus arXiv listing pages; OpenReview-only and non-arXiv venues were not
swept. The Ran-Milo et al. 2026 absorbing-chain content is `[U]`. No numbers in this file
were produced by new code; the two `NOT MEASURED` items in §5 say what code they would need.
