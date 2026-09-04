# THESIS NOTES — the propositions the shape rests on, with proof sketches

Coordinator's notes for the design panel, 2026-09-03. Every proposition below is
either (a) a one-line corollary of an existing Lean theorem, (b) elementary and proved
here in full, or (c) RUN on this box (`$SCRATCH/shape_identities.py`, float64). The
panel's job is to formalise, attack, and either keep or kill each one. Nothing here is
to be softened into prose; each becomes a numbered Proposition in the paper with its
evidence class.

Notation. `s` positions, `d` value width. `P ∈ ℝ^{s×s}` causal (`P_ij = 0` for
`j > i`). `V ∈ ℝ^{s×d}`. `γ ∈ [0, 1)` unless stated. `𝒜 = 𝒜_1 ⊔ … ⊔ 𝒜_K ⊂ [s]` the
absorbing (constraint) positions; `T = [s] \ 𝒜` the transient positions. For an
absorbing chain `P` has `P_ii = 1` and `P_ij = 0` (`j ≠ i`) for `i ∈ 𝒜`; `Q = P_TT`,
`R_k = P_{T 𝒜_k}`. `1_S` the indicator vector of `S`.

## THE FAMILY (two regimes, one solve)

```
  base operator   W_{β,qk,g}   = the record's three-corner family (ceq/arm_smprime.py):
                                 W_ij = G_ij · exp(qk · q_i·k_j) / Z_i^β,  j ≤ i
  state           z(γ)          = (I − γ W)^{-1} V                       (the shape)
  mixing matrix   Π_γ           = (1 − γ) W (I − γ W)^{-1}
  read            O(γ)          = Π_γ V  =  (1 − γ) W z(γ)
```

Regime S (softmax regime): `β = 1`, so `W = P` is row-stochastic with the diagonal
included; `γ < 1`; convergence by `‖γP‖_∞ = γ < 1` (Neumann); certificate P4.

Regime N (nilpotent regime): `W` strictly causal (diagonal excluded) — the record's
sub-diagonal chain `A`; any `γ`, in particular `γ = 1`; convergence by nilpotency
(`lean/CEQ/Nilpotent.lean pow_card_eq_zero`, `Occupancy.lean occupancy_is_exact_inverse`);
the sum is exact in `s` terms.

The record's corner 3 lives in regime N (P2). The author's north star lives in regime S
with boundary rows (P3, P8). Both are the same triangular solve (P5).

## P1 — PARITY AT γ = 0 (bitwise, with a rejection region)

Statement. For every `W`, `V`: `O(0) = W V`. In IEEE-754 forward substitution with
`γ = 0` computes `z_i = v_i − Σ_{j<i} (0 · W_ij) z_j = v_i` exactly (for finite entries
`0·x = 0` and `v − 0 = v` are exact), and `(1 − 0) · (W z) = W V` is the same matmul as
attention. So at `β = 1, g ≡ 0, qk` on the read is **bitwise** softmax attention.

Evidence. RUN: `torch.equal(O(0), P V) = True` at `n=4, s=64, d=16`. Rejection region:
`max|O(0.5) − O(0)| = 2.3003` on the same draw (V-24: an identity bind must have a
non-empty rejection region). Lean: `V16Domain.three_corners_containment` gives the
`β = 1` corner; the `γ = 0` clause is a one-line corollary (`(I − 0·W)^{-1} = I`) —
target `gamma_zero_is_softmax` [M].

Design-against. V-24 (planted mutilations: `γ ≠ 0`, a non-causal `W`, a wrong
normaliser must each break the bind at O(1)); the record's own V-24 instance (the
two-branch escape that passed bitwise for the label itself) is the reason the read is
defined as a *single* mixture `Π_γ V` and not as `softmax(q)@V₁ + λ·X@V₂`.

## P2 — THE RECORD'S CORNER 3 IS A RESOLVENT (regime N)

Statement. Let `A` be strictly lower bidiagonal, `A_{i,i−1} = a_i`. Then
`[(I − A)^{-1}]_{ij} = Π_{k=j+1}^{i} a_k` for `j ≤ i` and `0` above the diagonal — i.e.
`(I − A)^{-1} = G`, the record's path product (`ceq/arm_smprime.py:144 path_product`,
`lean/CEQ/V16Domain.lean pathProd`).

Proof. `(A^m)_{ij}` sums products of entries along walks of length `m` from `i` to `j`;
the only walk on a sub-diagonal is `i → i−1 → … → j`, of length `i − j`, with product
`Π_{k=j+1}^{i} a_k`; `A^m = 0` for `m ≥ s` (nilpotent), so `Σ_{m<s} A^m = (I − A)^{-1}`
exactly (`occupancy_is_exact_inverse`), entry `(i,j)` receiving exactly the `m = i−j`
term. ∎

Consequences. (i) BED-M's label `y_{s−1} = ((I − A)^{-1} b)_{s−1}` (`equilibrium_oracle`,
`scale/negation_scope.py:286`) is the shape's state `z(1)_{s−1}` with `W = A`, `V = b`;
and because BED-M sets `b_{s−1} = 0` (the earned admissibility fix, `MATHEMATICS.md`
§1.2), it is also the read `(W z)_{s−1}`. (ii) Every corner-3 certificate the record
banked transfers to the shape unchanged. (iii) The zero-gate theorem
(`pathProd_eq_zero_iff`) is a statement about this resolvent's support: a zero at `k`
cuts `(I − A)^{-1}` into two independent blocks (P6).

Evidence. RUN: `max|G − (I − A)^{-1}| = 0.0` (exact), `max|z(1)_{s−1} − y| = 6.2e-15`,
`A^s = 0` exactly, on BED-M's own draw (`make_equilibrium_batch(8, 64, 24, seed=0)`).
Lean target `pathprod_is_chain_resolvent` [M] (induction on `i − j`).

## P3 — THE COMMITTOR IS THE RESOLVENT READ WITH ABSORBING ROWS (regime S)

Statement. Let `P` be row-stochastic with `𝒜 = ⊔_k 𝒜_k` absorbing and absorption
almost sure from `T` (`ρ(Q) < 1`). Take `V = 1_{𝒜_k}`. Then for `i ∈ T`:

```
  (1 − γ) z(γ)_i  =  E_i[ γ^{τ_k} ]           τ_k = hitting time of 𝒜_k (γ^∞ := 0)
  O(γ)_i          =  E_i[ γ^{τ_k − 1} ]
  lim_{γ↑1} O(γ)_i =  P_i(τ_k < ∞)  =  q^{(k)}_i =  [(I − Q)^{-1} R_k 1]_i
```

and `Σ_k q^{(k)} = 1` on `T`. For `K = 2` this is the committor (splitting
probability); for general `K` the absorption-probability vector `B = N R` of the
fundamental matrix `N = (I − Q)^{-1}` (Kemeny–Snell).

Proof. `z_i = Σ_{t≥0} γ^t (P^t 1_{𝒜_k})_i = Σ_t γ^t P_i(X_t ∈ 𝒜_k) = Σ_t γ^t P_i(τ_k ≤ t)`
because `𝒜_k` is absorbing. Exchange sums: `Σ_u P_i(τ_k = u) Σ_{t≥u} γ^t =
E_i[γ^{τ_k}]/(1 − γ)`. The read: `O_i = (1−γ) Σ_j P_ij z_j = Σ_j P_ij E_j[γ^{τ_k}] =
E_i[γ^{τ_k − 1}]` by first-step analysis (`τ_k ≥ 1` from `i ∈ T`). Monotone convergence
in `γ ↑ 1` gives `P_i(τ_k < ∞)`; first-step analysis `q = R_k 1 + Q q` gives the
`(I − Q)^{-1} R_k 1` form, invertible since `ρ(Q) < 1`. ∎

Reading. `γ` is the **horizon dial**: `γ = 0` reads the next step (softmax), `γ ↑ 1`
reads the equilibrium (which basin), and in between the read is the hitting-time
transform `E[γ^τ]`, which weights *sooner* absorption more — the "next transient
state" the author asked for, with urgency built in. Multi-constraint: `K` value
channels `1_{𝒜_1}, …, 1_{𝒜_K}` through **one** solve give all `K` committors (one
triangular solve, `K` right-hand sides).

Evidence. READ: `ceq/beds/bed_1.py:188-198 committor` solves exactly the Dirichlet form
(interior harmonic, `q = 0` on `A`, `q = 1` on `B`); `harmonic_residual` reads
`0.000000e+00` (`workdonenewseal.md` §5.2); `MATHEMATICS.md` §7 (`N = (I − Q)^{-1}`,
`B = N R`) and §11 (Kirchhoff cross-check to `1e-10`). Lean target
`committor_is_resolvent_read` — the record's #14 `committor_eq_harmonic` [S]; the
`E[γ^τ]` identity is new, [S].

Design-against. D-2 (an oracle that is the arm's own resolvent): the oracle's `P` is
the *latent environment chain*; the arm's `P` is computed from the context tokens by
`q, k`; the two coincide only if the arm learns the chain, and that coincidence is the
capability being tested, not a tautology. Leak guard: the transition probabilities
never appear as input channels (MARS attack 2 in `sec_refuted.md`).

## P4 — THE NEUMANN CERTIFICATE IS EXACT AND HAS NO SUM OVER s

Statement. `P ≥ 0` row-stochastic, `γ ∈ [0,1)`, `K ≥ 0`:

```
  ‖ (I − γP)^{-1} − Σ_{k=0}^{K} (γP)^k ‖_∞  =  γ^{K+1} / (1 − γ)      (equality)
```

For a general (signed or non-stochastic) `P` with `‖P‖_∞ ≤ 1` the same quantity is
`≤ γ^{K+1}/(1 − γ)`; for `‖P‖_∞ > 1` no such bound holds.

Proof. `‖γP‖_∞ = γ < 1` so `(I − γP)^{-1} = Σ_{k≥0} (γP)^k`; the tail
`Σ_{k>K} γ^k P^k` is a nonnegative matrix whose every row sums to `Σ_{k>K} γ^k`
(`P^k` is row-stochastic), so its `∞`-norm equals `γ^{K+1}/(1 − γ)`. The inequality for
`‖P‖_∞ ≤ 1` is the triangle inequality on the same tail. ∎

Evidence. RUN: err = bound to `1e-15` at `K ∈ {1,2,4,8,16}`, `γ = 0.7`, `s = 64`;
planted negative (rows summing to `1.5`): err `119.37` vs bound `1.143` — the
certificate can fail, so it is a certificate (V-24). Lean target
`neumann_truncation_bound` [M] (Mathlib has geometric series and matrix norms).

Why it matters. `IMPOSSIBLE.md` I1 asked for a magnitude certificate with *no sum over
`s`* and no per-row normaliser, for a *signed* operator, and conjectured it impossible
because the absent normaliser was the mechanism of the signed arm's context-stable
sign influence. P4 is the resolution: **the normaliser pays for the certificate**
(`‖P‖_∞ = 1` independent of `s`), and the veto a third token can exercise is moved from
the *sign* of an entry to a **boundary condition** (an absorbing constraint set). The
signed programme is not revived; its one live want (a certificate) is met by the thing
it threw away.

## P5 — THE READ IS ONE TRIANGULAR SOLVE, AND `Π_γ` IS STILL A MIXTURE

Statement. (a) For causal `W`, `I − γW` is lower-triangular with diagonal `1 − γW_ii`;
in regime S the diagonal lies in `(1−γ, 1]`, in regime N it is `1`; so `z` is a
forward substitution, `≈ s²d/2` multiply-adds per head, against `2s²d` for `QKᵀ` and
`PV` together. (b) In regime S, `Π_γ = (1−γ)Σ_{t≥0} γ^t P^{t+1}` is row-stochastic and
nonnegative; hence `O_i` lies in the convex hull of `{V_j : j ≤ i}` for every `γ`.

Proof. (a) triangular structure is inherited from `W`; the diagonal entry of
`(I − γW)` is `1 − γW_ii`; substitution cost `Σ_i i·d`. (b) `(1−γ)Σ γ^t = 1` and
products of row-stochastic matrices are row-stochastic. ∎

Evidence. RUN: `solve_triangular` vs dense inverse `1.8e-15`.

The limitation this exposes (state it in Limits and in §4, not buried). Because `Π_γ`
is a mixture, the shape's expressivity gain over softmax is entirely in the **mixing
weights** — the support of `Π_γ` is the transitive closure of `P`'s support, every hop
at once — and **not in the value range**. The record's Q2/W3 bound (a row-stochastic
read cannot leave the hull of the values; `V20_R15_THEORY_TABLE.md` cell Q2/W3) applies
to the shape in regime S verbatim. Labels that leave the hull (BED-M's chain label with
raw drives) need either the value rescale the record already proved
(`V15Fork.Asink_computes_chain`, `V_j = b_j/(1 − a_j)` with the BOS sink) or regime N.
Committors live in `[0,1] = hull{0,1}`, so the safest-move label is inside the class.

## P6 — SEGMENTATION IS EXACT AND SPLITS THE SOLVE

Statement. In the three-corner base with the path-product gate, a zero magnitude
`m_k = 0` gives `W_ij = 0` for all `j < k ≤ i` (`pathProd_eq_zero_iff`, hypothesis-free).
Then `I − γW` is block lower-triangular with a **zero** sub-block, i.e. block-diagonal
across the cut, and the resolvent factorises into independent per-segment solves. Cost
falls from `s²d/2` to `Σ_seg |seg|² d/2`. Pure softmax has no exact zeros (`exp > 0`), so
this F0 certificate is a property of the family the record built and not of softmax —
it is `CEQ_V20_R15_CONTRACT.md` M9's F0 sparsity, now with a cost meaning.

Proof. Block structure from the zero pattern; a block lower-triangular matrix with a
zero off-diagonal block inverts blockwise. ∎ (Lean target `segmentation_blockdiag` [M].)

Evidence. RUN in the record: BED-M's causal triangle is `96.78 %` zero-hop
(`workdonenewseal.md` §3); `133,120/133,120 NaN` for the prefix-scan route
(`no_prefix_scan_represents_a_zero_gate`).

## P7 — WHAT DEPTH-1 SOFTMAX CANNOT DO, STATED NARROWLY (each clause cited)

(a) *Per-row form.* `O_i = f(P_{i·}, V)`: the read at `i` never depends on the read at
`j`. `z` in the shape solves `z = V + γPz`, the unique fixed point of a
`γ`-contraction in `‖·‖_∞`; the reads are mutually constrained. (Definition; no citation
needed; the record states it at `MATHEMATICS.md` §0.3.)

(b) *Hops need depth.* Composing `k` content-dependent hops needs depth
`⌊log₂ k⌋ + 2` in a transformer (Sanford–Hsu–Telgarsky 2024 Thm 4.2, upper bound); the
matching lower bound is **conditional** on the 1-vs-2-cycle conjecture (Cor. 4.3).
Graph connectivity needs logarithmic depth (Sanford et al. NeurIPS 2024). The shape
computes all hops in one operator. — The sweep S3 fixes the exact statements.

(c) *Composition.* A single attention layer cannot compose two functions on a large
domain (Peng–Narayanan–Papadimitriou 2024, communication-complexity argument) — [U]
until S3 verifies the id and statement. A consequence *is* a composition (state ↦ next
state ↦ …).

(d) *Optimality on the wrong ground.* One attention layer is (asymptotically, `L = o(d)`)
Bayes-optimal for single-location regression (Marion et al. ICLR 2025, with the
record's three caveats at `MATHEMATICS.md` §17.5). Every bed in the record was that
task. `MISTAKES.md` D-1.

The honest control (skyline). A **looped** softmax block with input injection,
`z^{(t+1)} = V + γ P z^{(t)}`, `P` fixed from the input, run `K` times, computes the
truncated Neumann sum; by P4 its error against the shape is **exactly**
`γ^{K+1}/(1−γ)` in `‖·‖_∞` on the mixing matrix. So the shape is the `K → ∞` limit of a
looped softmax layer, computed exactly at the cost of one attention. Claims against
softmax are therefore made at **matched depth and parameters**; a deeper or looped
stack is the skyline, cited (Yang et al. 2024 looped transformers; APPNP's power
iteration), never beaten in a sentence. A nonlinear loop (`P` recomputed from `z`) is a
DEQ (Bai et al. 2019) and is out of scope.

## P8 — THE SAFEST MOVE UNDER K CONSTRAINTS

Statement. Context = a state description plus `m` candidate moves `a_1..a_m`, each
realised as an intervention `do(a)` that rewrites the tokens it controls. For each
move the `K` reads `q^{(k)}(a)` (P3) are the constraint-violation probabilities. The
safest move is

```
  a*  =  argmin_a  max_k  q^{(k)}(a)          (minimax over constraints)
```

equivalently the move maximising the reach-avoid safety probability when the `𝒜_k` are
unsafe sets; the weighted form `Σ_k λ_k q^{(k)}` is the Lagrangian/CMDP relaxation and
the lexicographic form the priority variant. The label is exact from the oracle chain
(the Dirichlet solve the record owns); the information floor on the argmin is Fano over
`m` moves (`CEQ_V20_R15_CONTRACT.md` M11), and the floor on the committor vector is the
exact oracle at `0`. Cost: one solve with `K` right-hand sides per candidate, or one
solve with `mK` right-hand sides when moves only change values.

Why softmax cannot: at depth 1 it reads one step of consequence (`P V`); a constraint
reached in `t* ≥ 2` steps is outside its class unless the hop structure is leaked into
the input (the leak attack, guarded). The deeper skyline can; the claim is at matched
depth (P7).

Design-against. V-8 (constant label: print committor sd, class balance over moves,
discard count at construction); the zero-hop control (the argmin must not be readable
from the move token alone — `t* = 0` control must read chance); D-3 (the dial `t*`, `K`,
`m` must vary); M-2 (the minimax rule is fixed before data).

## P9 — CONSEQUENCE AS A DISPLACEMENT FIELD

Statement. For an intervention that changes the values by `ΔV` and leaves `P` fixed,
`Δz = (I − γP)^{-1} ΔV` — the resolvent column(s) of the intervened position(s) are the
displacement of every other position; with `P` also changed (the intervened token's
`q,k` move), `Δz` is the difference of two solves. In the linear-SCM reading
(`X = (I − B)^{-1} Z`, graph surgery), this is the interventional effect matrix; the
record's `bed_k.bump / rebuild` pair is the exact Jacobian oracle
(`ceq/beds/bed_k.py`), and its consequence-fidelity metric (`MATHEMATICS.md` §6)
generalises from a sign on one scalar to sign agreement per coordinate plus a
position-matched state metric (per-coordinate NRMSE vector; Procrustes/OT with
position matching where only arrangement matters), paired against softmax by McNemar
on byte-identical draws.

## P10 — THE HORIZON DIAL (next token → next transient state → equilibrium)

`γ = 0`: next step, softmax bitwise (P1). `0 < γ < 1`: the hitting-time transform
`E[γ^{τ−1}]` — the transient regime, with time-scale weighting (P3). `γ ↑ 1`: the
equilibrium — absorption probabilities, basin membership `argmax_k q^{(k)}`, and the
decision surface where the top two `q`'s tie (the isocommittor `q = ½` for `K = 2`,
`ceq/beds/bed_1.py` guards). The record's corner 3 is `γ = 1` on a nilpotent chain (P2).
One learnable scalar carries the whole interpolation; the LR-test pinning discipline
(Ruling 10′) decides whether a trained `γ` is distinguishable from `0`.

## WHAT IS NOT CLAIMED

- No claim that the shape beats softmax on any bed where softmax is a fellow
  approximator (BED-M), nor against the looped/deeper skyline at any depth.
- No claim of novelty for any single component: the resolvent of a stochastic matrix
  is the successor representation / personalised PageRank (APPNP owns it on fixed
  graphs); the committor is transition-path theory; the chunked triangular solve is
  DeltaNet's kernel pattern; the horizon dial is the discount factor. The sweep decides
  what, if anything, is unoccupied in the *composition*: a content-dependent causal
  attention matrix, absorbing constraint rows, bitwise softmax at `γ = 0`, the
  committor/safest-move label class, and machine-checked identities.
- No trained number exists for the shape. Every capability sentence in the paper is a
  prediction with a counter and a price, never a result.
