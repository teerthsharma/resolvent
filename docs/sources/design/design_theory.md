# DESIGN — THEORY: the shape formalised so the plan can be built on it

JUPITER (MYCROFT), 2026-09-03. Repository `C:\Users\seal\Desktop\New folder (32)`, branch `v17k-gate0`, HEAD `207e7b9`. Inputs read in full:
`BRIEF.md`, `THESIS_NOTES.md`, `THESIS_CORRECTIONS.md`, `THESIS_CORRECTIONS_2.md`, the six `sections/sec_*.md`, the eight
`sweep/sweep_*.md`, `references.bib` (408 entries) and `bib_aliases.md`. Not read, by instruction: `design/design_instrument.md`,
`design/design_falsify.md`.

Evidence classes. `RUN` — executed this session; two kinds are distinguished throughout: `RUN[coord]` (the coordinator's
`shape_identities.py`, torch 2.5.1 float64, `s = 64`, `d = 16`, as carried by `THESIS_NOTES.md` and `sec_proved.md` §2.8) and `RUN[here]`
(two numpy float64 one-liners run by this planet, `s = 32`, `d = 4`, `γ = 0.6`, seed 0, causal softmax `P` with the diagonal, goal set `𝒜_0
= {0, 5}`, constraints `𝒜_1 = {9, 10}`, `𝒜_2 = {15}`, intervened row `i = 12`; no file written). `READ path:line` — quoted from the tree via
the section that transcribed it (the transcriber is named). `CITED [V]` — a bibliography key whose identifier the sweeps fetched this
session; `[U]` where the sweep says so. `DERIVED` — steps written out. Every load-bearing claim carries one; every proposition carries the
`MISTAKES.md` mechanism it is designed against and one status token from `{LEAN, PROVED-HERE, RUN, CITED, OPEN}`.

Two binding facts from the author (`THESIS_CORRECTIONS_2.md` §0): the paper's body is the future programme, and this file is a *plan input*
— its propositions are the load-bearing walls, its Lean list is the first evening's work, and nothing here is a result.

---

## 1. Definitions

### 1.1 The base family (the record's three corners)

For a context of `s` positions with logits `qk_ij = q_i·k_j`, a gate sequence `g`, and a switch `β ∈ {0, 1}`, the record's operator is

```
  W_{β,qk,g}[i,j] = 𝟙[j ≤ i] · exp( (scan g)_i − (scan g)_j + qk_ij ) / Z_i^β ,
  Z_i = Σ_{j ≤ i} exp( (scan g)_i − (scan g)_j + qk_ij ) ,
```

`READ lean/CEQ/V16Domain.lean:366-378 (Hop, num, Znorm)` via `sec_proved.md` §2.2. Its three corners are theorems: `β = 1, g ≡ 0` is
`softmaxAttn`; `β = 0, g ≡ 0` is `linearAttn`; `β = 0, qk ≡ 0` is the path product `Wc g` — `V16Domain.three_corners_containment` (`:433`),
`corners_are_distinct` (`:445`), `gate_zero_beta_zero_is_linear_attention` (`:483`), `corner_path_product_is_the_gate_product` (`:414`,
hypothesis `∀ k, 0 < a k`). The corners are measured distinct at `|c₁−c₂| = 4.472918`, `|c₁−c₃| = 1.144938`, `|c₂−c₃| = 5.335671` (`READ
V16_ARM_SMPRIME.md:333` via `sec_measured.md` M.3).

Vocabulary fixed here (the mentor rule, `THESIS_CORRECTIONS.md` C10): the `β = 1` corner is a causal, row-stochastic Markov kernel on
positions (`erel-2025-attentionchains` owns the reading of a softmax matrix as a DTMC; `lin-2026-connection-laplacian` owns the
connection-walk reading); the `β = 0, qk ≡ 0` corner is the 1-semiseparable SSD mask (`dao-2024-ssd` Def. 3.1, via `sweep_linrec.md` §3 step
1).

### 1.2 Two regimes, one solve

```
  state          z(γ)  = (I − γW)^{-1} V                     V ∈ ℝ^{s×d}
  mixing matrix  Π_γ   = (1 − γ) W (I − γW)^{-1}
  read           O(γ)  = Π_γ V  =  (1 − γ) W z(γ)
```

**Regime S (softmax regime).** `β = 1`; `W = P` is row-stochastic with `P_ii > 0` (the diagonal is *inside* the causal window `j ≤ i`); `γ ∈
[0, 1)`. Convergence: `‖γP‖_∞ = γ < 1`, so `(I − γP)^{-1} = Σ_{t≥0} (γP)^t` is an *infinite* series with the certificate of P4. The only
theorems in the tree stated for a row-stochastic `P` with a diagonal are `Contraction.weighted_contraction` (`Contraction.lean:72`),
`rowStochastic_perron` (`:118`) and `expander_expands_l2` (`:151`) (`READ` via `sec_proved.md` §2.1).

**Regime N (nilpotent regime).** `W = A` strictly causal (`A_ij = 0` for `j ≥ i`); any `γ`, in particular `γ = 1`. Convergence by
`Nilpotent.pow_card_eq_zero` (`Nilpotent.lean:77`, hypothesis `StrictlyLower A := ∀ i j, i ≤ j → A i j = 0`) and
`Nilpotent.occupancy_is_exact_inverse` (`:96`): the sum is exact in `s` terms, no sign or magnitude hypothesis. The record's corner 3 lives
here (P2).

The boundary between the regimes is a theorem, not a convention (C8 below): `Nilpotent.one_not_nilpotent` (`:105`) records that a diagonal
entry destroys nilpotency, so no theorem of regime N transfers to regime S and the paper chooses per regime and says which theorem carries
which sentence. Design-against: **V-25** (a theorem whose hypothesis no draw satisfies) — the softmax corner never satisfies
`StrictlyLower`.

### 1.3 Boundary rows: `K` constraint sets, a goal set, BOS, and the query

Let `𝒜_0` (the goal set) and `𝒜_1, …, 𝒜_K` (the constraint sets) be pairwise disjoint position sets, `𝒜 = ⊔_{k=0}^{K} 𝒜_k`, `T = [s] \ 𝒜`
the transient positions. The boundary operator is `P` with rows on `𝒜` replaced by identity rows:

```
  P_{a,·} = e_a   for a ∈ 𝒜 ;    canonical form on (T, 𝒜):   P = [[Q, R],[0, I]],
  Q = P_TT (transient block),   R_k = P_{T𝒜_k} (absorbing columns of set k).
```

Vocabulary: absorbing states, fundamental matrix `N = (I − Q)^{-1}`, absorption matrix `B = N R` — `kemeny-1960-finitemarkov` (CITED
[V-cat], `sweep_safety.md` §1.A) and `grinstead-1997-probability` Def. 11.3 / Thm 11.6 (CITED [V] on the host page, `[U]` ISBN). Two facts
fixed by the first design pass and re-run here (`THESIS_CORRECTIONS_2.md` §1):

- **F1 (BOS is absorbing by construction).** On any causal softmax `P`, row 0 is `e_0` because its window is `{0}` (`RUN[here]`:
  `allclose(P[0], e_0) = True`). If constraint and goal sets are declared but position 0 is left in `T`, then `T` contains a closed class
  and `ρ(Q) = 1.0` (`RUN[here]`: `1.0` undeclared; `0.5134` with `0 ∈ 𝒜_0`; the first pass read `1.0` / `0.687` on its own draw). **The
  definition therefore requires `0 ∈ 𝒜_0`** — BOS is declared inside the goal set. The record's value-zero BOS sink (`V15Fork.Asink`,
  `V15Fork.lean:67-70`) is the same object seen from the value side (a column sink); the shape's `e_0` row is the row condition; the two
  coexist and are not conflated (`sweep_resolvent.md` §2.14; `ranmilo-2026-attentionsinks` owns the forced anchor). Design-against:
  **V-25**, **D-3**.
- **F2 (constraints precede the query).** Walks on a causal `P` move to `j ≤ i`; a boundary position placed after the query position is
  unreachable and its committor reads `0.0` exactly. Every `𝒜_k` and `𝒜_0` lies before the query position, and the builder prints
  reachability at construction. Design-against: **V-8** (the PASS half's label is constant).

### 1.4 The read is a mixture and carries the `(1 − γ)` factor

Without boundary rows, `Π_γ` is row-stochastic and non-negative (P5b). With boundary rows the *bare* read `P(I − γP)^{-1}` has row sums
`1/(1 − γ)` on every row (`RUN[here]`: min `2.4999…`, max `2.5` at `γ = 0.6`, `1/(1−γ) = 2.5`) — so the paper's read **is** `Π_γ V` with the
factor, and any table that prints the bare read declares its row sums beside it (V-17, a threshold imported out of its units). This is
ChaCAL's Eq. 5 up to the record's base (`fagnou-2024-chacal`, CITED [V]; `sweep_expressivity.md` §2.9).

### 1.5 The horizon dial

`γ ∈ [0, 1)` is one learnable scalar per head. `γ = 0` reads the next step (P1); `γ ↑ 1` reads the equilibrium (P3); between, the read is
the hitting-time transform `E[γ^{τ−1}]` (P3, P10). Vocabulary: discount / teleport probability (`bellman-1957-markovian`,
`gasteiger-2019-appnp`, CITED [V]); the *safety discount* of `fisac-2019-bridging` (CITED [V]) has the same effect of buying a contraction
and the same caveat that delaying absorption counts.

### 1.6 The interventional channel

A candidate move `a` is an intervention `do(a)` on the context: it rewrites the tokens it controls, hence changes `(P, V) → (P', V')` with
`P' − P` supported on the intervened rows and `V' − V` on the intervened positions. The consequence is the displacement `Δz = z' − z` (P9,
C6). Vocabulary: graph surgery on a linear SCM (`pearl-2009-causality`, `shimizu-2006-lingam`, CITED [V]); perturbation of the fundamental
matrix (`schweitzer-1968-perturbation`); the rank-one re-solve (`sherman-1950-inverse-adjustment`, `hager-1989-updating`).

### 1.7 The decision

With the goal set present, the per-move label is the reach-avoid vector `(q^{(0)}(a), q^{(1)}(a), …, q^{(K)}(a))` with `Σ_k q^{(k)} = 1` on
`T` (P3, P8). The safest move is `a* = argmax_a q^{(0)}(do a)`, with the Chebyshev form `argmin_a max_{k≥1} q^{(k)}(do a)` printed beside it
(`vanmoffaert-2013-chebyshev`, CITED [V]) and the lexicographic form as a column (`yang-2026-lexisafe`). Vocabulary: a one-step safety
filter whose value is the worst-constraint committor (`hsu-2023-safetyfilter`, `borquez-2023-lrf`).

---

## 2. Propositions

Format: statement · hypotheses · proof or Lean carrier · evidence and number · mechanism · status.

### P1 — Parity at `γ = 0`, with a rejection region

**Statement.** For every causal `W` and every `V`, `O(0) = W V`; in IEEE-754 forward substitution at `γ = 0` the computed `z` equals `V`
exactly and `(1 − 0)·(W z)` is the same matmul as attention, so at `β = 1, g ≡ 0` the read is softmax attention bitwise.

**Hypotheses.** `W` causal (lower-triangular); finite entries.

**Proof.** `(I − 0·W) = I`, so `z = V` and `Π_0 = W`. For the float clause: forward substitution computes `z_i = (v_i − Σ_{j<i} (0·W_ij)
z_j)/(1 − 0·W_ii)`; `0·x = 0`, `v − 0 = v` and `v/1 = v` are exact for finite `x, v`, so `z = V` bitwise, and the read is the identical `W @
V` call. ∎ (DERIVED.) Lean carrier for the corner: `V16Domain.three_corners_containment` (`:433`) and `corner_softmax` (`:394`); the `γ = 0`
clause is target `gamma_zero_is_softmax` [M] (§6).

**Evidence.** `RUN[coord]`: `torch.equal(O(0), P V) = True` at `s = 64, d = 16`; rejection region `max|O(0.5) − O(0)| = 2.3002850040264393`.
`RUN[here]`: `array_equal = True`; rejection at `γ = 0.6`: `2.5685`. Reference point for any "bitwise" sentence: the record's own softmax
corner is *not* bitwise against `ceq/lm.py` (`1.110223e-16` on `19/64` entries, `READ V16_ARM_SMPRIME.md:266-293` via `sec_measured.md` M.1)
— the fused kernel subtracts the row max; the paper names the mechanism, never rounds it.

**Mechanism.** **V-24** (an identity bind with an empty rejection region): the bind ships the mutilation battery — `γ ≠ 0`, a non-causal
`W`, a wrong normaliser (`β = 0` at the softmax corner reads `max|gap| > 0.5`, `READ V16_ARM_SMPRIME.md:336-339`) — and, per
`THESIS_CORRECTIONS.md` C2, the planted negative "ChaCAL with the same `γ`" against which P1 alone has an *empty* rejection region (both are
softmax at `γ = 0`). **V-3**: parity holds by construction and certifies nothing without the pair.

**Status.** PROVED-HERE (RUN bitwise; Lean [M] pending).

### P2 — The record's corner 3 is a resolvent (regime N)

**Statement.** Let `A` be strictly lower bidiagonal with `A_{i,i−1} = a_i`. Then for `j ≤ i`, `[(I − A)^{-1}]_{ij} = Π_{k=j+1}^{i} a_k`, and
`0` above the diagonal — i.e. `(I − A)^{-1} = G`, the record's path product (`ceq/arm_smprime.py:144 path_product`; `V16Domain.pathProd`).

**Hypotheses.** None on the sign or size of `a` (regime N).

**Proof.** `(A^m)_{ij}` sums products of entries along walks of length `m` from `i` down to `j`; the only walk on the sub-diagonal is `i →
i−1 → … → j`, of length `i − j`, with product `Π_{k=j+1}^{i} a_k`; every other `(A^m)_{ij}` is `0` (`Nilpotent.pow_entry_zero`,
`Nilpotent.lean:52`, gives `(A^k)_{ij} = 0` for `i < j + k`); `A^s = 0` (`pow_card_eq_zero`, `:77`), so `Σ_{m<s} A^m = (I − A)^{-1}` exactly
(`occupancy_is_exact_inverse`, `:96`), entry `(i,j)` receiving exactly the `m = i − j` term. ∎ The chain identity `V15.chain_path_product`
(`V15.lean:74`) is the last-row form.

**Consequences.** (i) BED-M's label `y_{s−1} = ((I − A)^{-1} b)_{s−1}` (`scale/negation_scope.py:286 equilibrium_oracle`) is `z(1)_{s−1}`
with `W = A, V = b`; since BED-M sets `b_{s−1} = 0` it is also the read. (ii) Every corner-3 certificate the record banked transfers to
regime N unchanged. (iii) `pathProd_eq_zero_iff` is a statement about this resolvent's support (P6).

**Evidence.** `RUN[coord]`: `max|G − (I − A)^{-1}| = 0.0`, last row vs `equilibrium_oracle` `6.217248937900877e-15`, `A^64 = 0` exactly, on
`make_equilibrium_batch(8, 64, 24, seed=0)`. `RUN[here]` (Gaussian `a`, `s = 32`, dense `inv`): `1.78e-15`, `A^32 = 0.0`.

**Mechanism.** **D-2** (an oracle that is the arm's own resolvent): P2 is the D-2 statement made exact — any arm containing corner 3
reproduces BED-M's label as its own forward, so BED-M is *contained*, not won, and no capability number may be read from it; it stays as the
parity bind and the truncation-law must-fire (`sec_beds.md` §6.A).

**Status.** LEAN (the two halves; the entrywise link `pathprod_is_chain_resolvent` is [S], §6).

### P3 — The committor is the resolvent read with absorbing rows (regime S)

**Statement.** Let `P` be row-stochastic on `[s]` with `𝒜 = ⊔_k 𝒜_k` absorbing and `ρ(Q) < 1`. Take `V = 𝟙_{𝒜_k}`. Then for every `i ∈ T`:

```
  (1 − γ) z(γ)_i   =  E_i[ γ^{τ_k} ]              τ_k := hitting time of 𝒜_k, γ^∞ := 0
  O(γ)_i           =  E_i[ γ^{τ_k − 1} ]
  lim_{γ↑1} O(γ)_i =  P_i(τ_k < ∞)  =  q^{(k)}_i  =  [(I − Q)^{-1} R_k 𝟙]_i ,
```

and, when the boundary sets exhaust the absorbing states, `Σ_k q^{(k)} = 𝟙` on `T`.

**Hypotheses.** `P ≥ 0` row-stochastic *including* its absorbing rows (a domain-census item, V-25); `ρ(Q) < 1` (absorption a.s. from `T`) —
which by F1 fails unless `0 ∈ 𝒜`.

**Proof.** `z_i = Σ_{t≥0} γ^t (P^t 𝟙_{𝒜_k})_i = Σ_t γ^t P_i(X_t ∈ 𝒜_k) = Σ_t γ^t P_i(τ_k ≤ t)`, the last equality because `𝒜_k` is
absorbing. Exchanging sums, `Σ_u P_i(τ_k = u) Σ_{t ≥ u} γ^t = E_i[γ^{τ_k}]/(1 − γ)`. For the read, `O_i = (1 − γ) Σ_j P_ij z_j = Σ_j P_ij
E_j[γ^{τ_k}] = E_i[γ^{τ_k − 1}]` by first-step analysis, since `τ_k ≥ 1` from `i ∈ T`. Monotone convergence in `γ ↑ 1` gives `P_i(τ_k < ∞)`;
first-step analysis `q = R_k 𝟙 + Q q` gives the `(I − Q)^{-1} R_k 𝟙` form, `I − Q` invertible since `ρ(Q) < 1`. Sum-to-one: `Σ_k R_k 𝟙 = 𝟙 −
Q 𝟙` (rows of `P` sum to 1), so `(I − Q) Σ_k q^{(k)} = (I − Q) 𝟙`. ∎ (DERIVED.)

**Evidence.** `RUN[here]`: `(1−γ)z_T` vs `Σ_u γ^u (Q^{u−1} R_k 𝟙)` to `8.33e-17`; the read vs `E[γ^{τ−1}]` to `8.33e-17`; `(1−γ)z` at `γ = 1
− 10^{-6}` vs the Dirichlet committor to `3.8e-07` (the `O(1−γ)` gap of the limit, as expected); `Σ_k q^{(k)} − 𝟙` on `T`: `4.4e-16` with
the goal set declared. `READ`: `ceq/beds/bed_1.py:188-198 committor` solves exactly the Dirichlet form; on BED-1's real sets `A = [0], B =
[1], |T| = 9` the resolvent read matches `bed["q"]` at `0.0`, `harmonic_residual = 1.0408340855860843e-17`, `ρ(Q) = 0.9408612510154677`,
`max|Q^{11}| = 0.477 ≠ 0` (`RUN[coord]` corrected one-liner, `sec_proved.md` §2.8 (3); the script's own `0.858` was a V-16 default-key
artefact, `sec_refuted.md` C12). Kirchhoff cross-check `< 1e-10` (`READ MATHEMATICS.md:542-600`).

**Reading.** `γ` is the horizon dial (P10); `K` value channels `𝟙_{𝒜_1}, …, 𝟙_{𝒜_K}` through *one* solve give all `K` reads (one triangular
solve, `K` right-hand sides). At `γ < 1` the read is the *discounted* absorption probability and `Σ_k O^{(k)} < 1` — the delay share is
printed beside every safest-move reading (`THESIS_CORRECTIONS_2.md` §1: `[0.136, 0.340]` at `γ = 0.6` on the first pass's draw).

**Mechanism.** **D-2**: the oracle's `P` is the latent environment chain built inside the bed; the arm's `P` is computed from tokens by `q,
k`; `OracleSeparation.oracle_ne_resolvent` (`:166`) proves the two operator classes disjoint and BED-1 instantiates its hypotheses. **V-12**
(a single absorbing target makes the label constant): `K ≥ 2` and a goal set. **V-25**: F1's BOS declaration.

**Status.** PROVED-HERE (RUN; Lean (a) [M], (b) [S], (c) [S], the `E[γ^τ]` clause [D], §6).

### P4 — The Neumann certificate is attained and has no sum over `s`

**Statement.** `P ≥ 0` row-stochastic, `γ ∈ [0, 1)`, `K ≥ 0`:

```
  ‖ (I − γP)^{-1} − Σ_{k=0}^{K} (γP)^k ‖_∞  =  γ^{K+1} / (1 − γ)      (equality).
```

For general `P` with `‖P‖_∞ ≤ 1` the same quantity is `≤ γ^{K+1}/(1 − γ)`; for `‖P‖_∞ > 1` no such bound holds.

**Hypotheses.** Non-negativity and row-stochasticity of `P` — the *whole* content.

**Proof.** `‖γP‖_∞ = γ < 1`, so the series converges and the tail `Σ_{k>K} (γP)^k` is a non-negative matrix whose every row sums to `Σ_{k>K}
γ^k = γ^{K+1}/(1 − γ)` (`P^k` is row-stochastic); the `∞`-norm of a non-negative matrix is its largest row sum. The inequality for `‖P‖_∞ ≤
1` is the triangle inequality on the same tail. ∎ (DERIVED.) The exact residual form `(I − γP)^{-1}(γP)^{K+1}` follows from
`Occupancy.occupancy_telescope` (`Occupancy.lean:52`) at `N = K + 1`.

**Evidence.** `RUN[coord]`: error equals bound to `1.3e-15` at `K ∈ {1, 2, 4, 8, 16}`, `γ = 0.7`, `s = 64`; planted negative (rows summing
to `1.5`): `119.37486584159647` against `1.1433333333333329`. `RUN[here]` (`γ = 0.6`): `K = 1: 0.9 = 0.9`, `K = 2: 0.54 = 0.54`, `K = 4:
0.1944 = 0.1944`, `K = 8: 0.02519424 = 0.02519424`; planted `1.5`-rows at `K = 2`: `7.29` vs `0.54`.

**Two halves against `IMPOSSIBLE.md` I1.** (i) With a row-stochastic `P` the normaliser pays for a denominator-free magnitude certificate
with no sum over `s` (`‖P‖_∞ = 1` independent of `s`). (ii) The sign capability the signed operator bought by dropping the normaliser is not
recovered; the veto is relocated to boundary rows, which a stochastic row can carry (`V15Fork.Asink_row_sum`, `V15Fork.lean:81`, no
hypothesis on `a`).

**Certificate semantics (C7).** Because the pass `err ≤ bound` is an *identity* on the class, it is a declared **V-3** and carries no
information about `P` beyond its two hypotheses; the bind is carried by the planted non-stochastic negative and by measuring `δ` on the
*shipped* mask in vector units `δ·‖V‖_∞` (`sec_cost.md` §4.x.3: `max|z_solve − z_{K=16}| = 6.13e-05` against bare `δ = 1.526e-05` and vector
bound `≈ 7.6e-05`). The mask certificate for a sparsified operator is a *different* object (F1 union bound, `CEQ_V20_R15_CONTRACT.md` annex
M9) and the exact solve on an F1 mask is refused (L-CERT).

**Mechanism.** **V-24** (planted non-stochastic `P`), **V-10** (the pass is attained by construction — stated as such), **V-17** (units:
`δ·‖V‖_∞`, never `δ`), **L-CERT**.

**Status.** PROVED-HERE (RUN; Lean `neumann_truncation_bound` [S]).

### P5 — One triangular solve; `Π_γ` is still a mixture

**Statement.** (a) For causal `W`, `I − γW` is lower-triangular with diagonal `1 − γW_ii`, which lies in `(1 − γ, 1]` in regime S and equals
`1` in regime N; `z` is a forward substitution, `≈ s²d/2` MACs per head against `≈ s²d` for the softmax head it contains (`sec_cost.md`
§4.x.1, house convention `scale/m3_flops.py:41`). (b) In regime S, without boundary rows, `Π_γ = (1 − γ) Σ_{t≥0} γ^t P^{t+1}` is
row-stochastic and non-negative, so `O_i` lies in the convex hull of `{V_j : j ≤ i}` for every `γ`.

**Hypotheses.** (a) causality; (b) `P ≥ 0` row-stochastic.

**Proof.** (a) triangular structure is inherited; the diagonal entry is `1 − γW_ii` with `W_ii ∈ (0, 1]` at `β = 1`; substitution cost
`d·Σ_i i`. (b) `(1 − γ) Σ_t γ^t = 1` and products of row-stochastic matrices are row-stochastic. ∎ (DERIVED.)

**Evidence.** `RUN[coord]`: `solve_triangular` vs dense inverse `1.7763568394002505e-15`. `RUN[here]`: `Π_γ` row sums to `6.7e-16`, min
entry `−9.9e-17` (round-off zero). `RUN` (NEPTUNE, `sec_cost.md` §4.x.3, certified RTX 4060, float32, `n = 2048, s = 64, d = 16`): solve +
`Pz` fwd+bwd `2.514 ms` against `PV` `1.473 ms` and one Neumann hop `3.001 ms` — the exact solve is cheaper than one hop at this geometry;
MAC crossover for truncation is `K < 1`, none.

**The limitation this exposes (stated here, not buried).** Because `Π_γ` is a mixture, the shape's expressivity gain over softmax in regime
S is entirely in the *mixing weights* — the support of `Π_γ` is the transitive closure of `P`'s support — and not in the value range. The
record's Q2/W3 bound (`V20_R15_THEORY_TABLE.md` cell Q2/W3, TERMINAL on both gradings, `sec_state.md` S.3) applies verbatim. Labels outside
the hull need the value rescale the record already proved (`V15Fork.Asink_computes_chain`, `:140`, `V_j = b_j/(1 − a_j)` with the BOS sink)
or regime N. Committors live in `[0, 1] = hull{0, 1}`, so the reach-avoid label is inside the class.

**Kernel boundary.** `hu-2025-ssdtheory` (CITED [V], `sweep_linrec.md` §3): `softmax(QKᵀ)` has rank `T` even for rank-1 logits, so no
finite-state SSM dual exists; the chunked-WY path of `yang-2024-deltanet` is closed to regime S, and the exact path is the full `s × s`
triangular solve or `zhao-2026-structuredsparse`'s blockwise `O(n^{4/3} d)`.

**Mechanism.** **M-8** (the shape is priced as a forward substitution, as an increment over the softmax head it contains, never as an
inverse); **M-3** (MACs and depth stated separately; the depth-`s` chain is what the chunked form removes).

**Status.** PROVED-HERE (RUN; Lean `lower_triangular_isUnit` [M], `resolvent_is_triangular_solve` [S]).

### P6 — Segmentation is exact and splits the solve

**Statement.** In the base with the path-product gate, `m_c = 0` gives `W_ij = 0` for all `j < c ≤ i` (`V16Domain.pathProd_eq_zero_iff`,
`:129`, no hypothesis). Then `I − γW` is block lower-triangular with a zero sub-block across `c`, hence block-diagonal across the cut, and
the resolvent factorises into independent per-segment solves; the cost falls from `s²d/2` to `Σ_m L_m² d/2` with dividend `D = s²/Σ_m L_m²`
(`sec_cost.md` (C5)). Pure softmax has no exact zeros (`exp > 0`); the certificate belongs to the family the record built.

**Proof.** If `M = [[M₁₁, 0],[M₂₁, M₂₂]]` with `M₂₁ = 0` then `M^{-1} = [[M₁₁^{-1}, 0],[0, M₂₂^{-1}]]` (blockwise inversion of a
block-diagonal matrix). ∎ (DERIVED; `sweep_topology.md` §3(a) states the same and calls it elementary.) The annihilation half is LEAN.

**Evidence.** `RUN[here]`: with the cut block of `P` set to `0` and rows renormalised, `(I − γP)^{-1}[c:, :c]` is *exactly* zero
(`array_equal`, not `allclose`); planted negative — the same block scaled to `1e-300` instead of `0` — leaves the inverse block nonzero
(`max 2.4e-298`): the F0/F1 boundary made visible. `READ`: BED-M's causal triangle is `96.78 %` zero-hop, positives `257,664/266,240`
(`V16_ARM_SMPRIME.md:518-522` via `sec_cost.md` §4.x.4), dividend `31.06×` DERIVED from that fraction — a statement that BED-M is almost
entirely dead gates (`L̄ ≈ 1.1`), not that text segments are short; `133,120/133,120 NaN` for the prefix-scan route
(`no_prefix_scan_represents_a_zero_gate`, `:165`). The contract's `8.9e-16, 595×` F0 instance has no located producer (P-11 potential,
`sec_proved.md` §2.0).

**Occupancy.** The mechanism (a data-dependent cumulative decay on softmax logits) is `lin-2025-forgetting-transformer` (CITED [V], full
text); its computation `D_ij = c_i − c_j` with `c = cumsum log f` is exactly the form the record's theorem excludes at `f = 0`; the converse
(a block mask confines reach) is `yang-2026-boundary-repair` Thm 1 (CITED [V]). The delta is the machine-checked iff and the impossibility
for the prefix-scan family (`sweep_topology.md` §3(a)).

**Mechanism.** **L-CERT** (`δ = 0` printed by the theorem); **V-25** (the cut is a drawn value on BED-M, `bedM_overlap_new_two = 3` of 3,
`V16Domain.lean:304`); **D-3** (the block size is the registered dial); **V-24** (the `1e-300` plant).

**Status.** LEAN (annihilation) + PROVED-HERE (block inverse); Lean `segmentation_blockdiag` [M], `resolvent_fromBlocks` [M].

### P7 — What depth-1 softmax cannot do, stated exactly as the theorems license

Restated in full in §3; the proposition is the conjunction of four licensed clauses:

(a) *Per-row form* — definitional, no citation needed: `O_i = f(P_{i·}, V)` never depends on `O_j`; the shape's `z` is the unique fixed
point of a `γ`-contraction in `‖·‖_∞` (`Contraction.rowStochastic_perron` + `weighted_contraction`) and the reads are mutually constrained.
(b) *Hops need depth* — `sanford-2024-logdepth` Thm 4.2 (upper, unconditional, the skyline) and Cor. 4.3 (lower, **conditional** on the
one-vs-two-cycle conjecture); `chen-2024-multilayer` Thm 1.1 (unconditional, decoder-only, vacuous below `n ≈ 10⁴`);
`sanford-2024-inductionheads` Thm 1 (unconditional, `t* = 2`, hypothesis `h·m·p = Ω(n)` not violated at the record's geometry). (c)
*Composition* — `peng-2024-transformer-limitations` Thm 1 (unconditional; vacuous at `s = 64, d = 16, p = 32`: `384 < 544`);
`kozachinskiy-2025-strassen` Thm 3.4 (infinite precision, asymptotic). (d) *The ground softmax owns* — `marion-2025-single-location` Cor. 2
(erf, `L = o(d)`) and `duranthon-2026-softmax-advantage` Prop. 4.2 (softmax proper): one layer is Bayes-optimal on single-location
regression, which every bed in the record was (D-1).

**Mechanism.** **P-10** (theorem numbers, never abstracts), **V-25** (the vacuity inequalities printed beside every unconditional bound),
**P-8 / V-17** ("depth-1" is one parameter set; the circuit-depth reading is disclaimed in the same paragraph, §3.3).

**Status.** CITED (the `hop_k → committor` reduction is OPEN, §3.3).

### P8 — The safest move under `K` constraints, with a goal set

**Statement.** Context = a state description plus `m` candidate moves `a_1..a_m`, each a `do(a)`. For each move the `K + 1` reads
`q^{(k)}(a)`, `k = 0..K` (P3), are the absorption probabilities into goal and constraints. The label per move is the reach-avoid vector; the
safest move is `a* = argmax_a q^{(0)}(do a)`, with `argmin_a max_{k≥1} q^{(k)}(do a)` printed beside it. Cost: one solve with `K + 1`
right-hand sides per candidate, or one solve with `m(K+1)` right-hand sides when moves change only values; with a single-row change per
move, `O(sd)` per candidate after the first solve (P9).

**Degeneracy lemma (PROVED-HERE).** If the only absorbing sets are the `K` constraint sets and absorption is a.s., then `Σ_{k=1}^{K} q^{(k)}
= 𝟙` on `T` (P3), so `max_k q^{(k)} ≥ 1/K` at every position and every candidate, and "avoid every constraint" is unattainable: the minimax
only redistributes probability among constraints (V-12 generalised; `grinstead-1997-probability` Thm 11.6; `THESIS_CORRECTIONS.md` C5). The
goal set `𝒜_0` is therefore part of the definition, not an option. (`RUN` first pass: `max_k q^{(k)}` min `0.605` at `K = 3` without a goal
set.)

**Floors and census.** Information floor on the committor vector: the exact oracle at `0.0`. On the argmin: Fano over `m` moves against
*restricted views* (`sec_beds.md` §6.C.5): the zero-hop floor `1 − ln 2/ln m` reads `0.5 / 0.667 / 0.75` at `m = 4/8/16` (`RUN` VENUS); `m =
2` refused (Fano vacuous); default `m = 8`. Domain census at construction: label sd, goal reachability from the query, class balance over
moves, discard count.

**Why depth-1 softmax cannot, narrowly.** At depth 1 it reads one step of consequence (`P V`); a constraint reached in `t* ≥ 2` steps is
outside its class *at the asymptotic geometries of P7(b),(c)*, and only there; the deeper skyline can (§3). The claim is at matched depth
and parameters.

**Hazard carried.** `misra-2023-safety-constrained-mdp` (CITED [V]): Bellman's principle can fail for safety-constrained multichain MDPs;
the shape's per-move one-step rule is not a policy and the paper says so in Limits.

**Mechanism.** **V-8 / V-12** (constant-label guard with the printed triple), **D-3** (`t*`, `K`, `m` vary), **M-2** (the rule fixed before
data), **D-2** (the transition matrix never appears in `x`; leak probe `R² ≥ 0.99` kills the bed, `sec_refuted.md` C2), zero-hop control
(`t* = 0` reads chance, `sec_refuted.md` C8).

**Status.** OPEN (the degeneracy lemma is PROVED-HERE; no BED-S cell exists; every number above is a floor formula or a design constant).

### P9 — Consequence as a displacement field

**Statement.** For an intervention changing `(P, V) → (P', V')`, `Δz = (I − γP')^{-1} (ΔV + γ ΔP z)` exactly (C6). For `ΔP = 0`, `Δz = (I −
γP)^{-1} ΔV` — the resolvent columns of the intervened positions are the displacement of every other position. For a single changed row `i`
with `V` fixed, writing `M = (I − γP)^{-1}`, `u = P'_i − P_i`: `Δz = γ (M e_i)(uᵀ z)/(1 − γ uᵀ M e_i)` (Sherman–Morrison). For causal `P`,
`(M e_i)_j = 0` for `j < i`: consequences propagate forward only.

**Proof.** Sherman–Morrison on `I − γP' = (I − γP) − γ e_i uᵀ`; forward-only from the lower-triangularity of `M` (P5a). ∎ (DERIVED;
`sweep_safety.md` §1.A derives the same.)

**Evidence.** `RUN[here]`: Sherman–Morrison vs re-solve `8.9e-16`; `V ≡ 𝟙` gives `z = 𝟙/(1−γ)` (`1.8e-15`) and `Δz = 2.2e-15` (the planted
identity, since `uᵀ𝟙 = 0`); `V ~ N(0,1)` gives `max|Δz| = 0.363` (rejection region); displacement before `i` exactly `0.0`. First pass
(`THESIS_CORRECTIONS_2.md` §1): `1.2e-15`, `0.1096`, `4.4e-16`. `READ`: `ceq/beds/bed_k.py:256-279 bump / rebuild` is the exact Jacobian
oracle by two routes; `MATHEMATICS.md:384-412` consequence fidelity with softmax already at `0.807843` CP `[0.754044, 0.854329]` on the
*scalar* sign task — so the field, not the sign, is the ground.

**Metric (adopted from VENUS, `sec_beds.md` §6.B.3).** Position-matched per-coordinate NRMSE vector (mean, max) plus the harmonic residual
`r(ẑ) = ‖(I − γP_env)ẑ − V‖_∞/‖V‖_∞` as the joint-consistency score, field cosine and magnitude ratio on `Δz`; marginal `W1` refused
(permutation-blind: `0.0` on the permuted oracle against NRMSE `1.421901`, `READ V20_R15_THEORY_TABLE.md:221`).

**Mechanism.** **V-24** (the `V ≡ 𝟙` plant and the Gaussian rejection region), **D-5** (declaring `0.0` without a movement test — the
do()-bit), **V-26** (joint field, never a per-coordinate sign rate), **M-8 / P-8** (the price is one extra triangular solve, DERIVED, to be
measured).

**Status.** PROVED-HERE (RUN; Lean `displacement_identity` [M], `causal_forward_only` [S]).

### P10 — The horizon dial

**Statement.** `γ = 0`: next step, softmax bitwise (P1). `0 < γ < 1`: the hitting-time transform `E[γ^{τ−1}]` — the transient regime with
time-scale weighting (P3). `γ ↑ 1`: absorption probabilities, basin membership `argmax_k q^{(k)}`, and the decision surface where the top
two tie (the `q = ½` isocommittor for `K = 2`, `ceq/beds/bed_1.py:5-8, :382`; TPT vocabulary `e-2010-tptreview`). The record's corner 3 is
`γ = 1` on a nilpotent chain (P2). One scalar carries the interpolation.

**Proof.** Corollary of P1, P3, P2. ∎

**Pinning instrument (OPEN).** Whether a trained `γ̂` is distinguishable from `0` is decided by the LR-test discipline of Ruling 10′ (`READ
V17K_RULINGS.md:389-436` via `sec_state.md` S.4): `Λ ≤ 3.841` PINNED, `> ln n` MOVED, the minimum detectable departure printed. The mirror
kill: `γ̂ → 1` sends `1/(1 − γ̂)` to infinity and the certificate becomes vacuous; `1/(1 − γ̂)` is printed beside every `δ`
(`sec_refuted.md` C5; the record's `1/(1 − â_max)` was undefined at all eight R1 seeds, `READ V15_R1.md:56`).

**Mechanism.** **V-9** (a repair that changes nothing: the ablation `(I − γ̂P̂)^{-1} → I` at trained weights must move NRMSE by more than
one seed sd), **M-2** (thresholds `3.841`, `ln n` fixed before data).

**Status.** PROVED-HERE (as a corollary; the instrument OPEN).

### C6 — Order of intervention and equilibration (EMC), and the rank-one price

**Statement.** (i) *Displacement identity*: `Δz = (I − γP')^{-1}(ΔV + γ ΔP z)` exactly, one extra solve. (ii) *EMC, causal case*: for causal
`P` (both regimes), "intervene then re-solve" (full re-solve with `P', V'`) and "settle then intervene" (freeze the settled `z_{<i}`,
re-solve only rows `≥ i` with `P', V'`) produce the same `z'`. (iii) *EMC, planted failure*: for a non-causal `P` (feedback), the two orders
differ and the displacement is nonzero before the intervened row.

**Hypotheses.** (ii) `P'_{j,·}` for `j < i` depends only on positions `< i` — true for any causal `P'`; no regime hypothesis. (iii) any `P`
with an entry above the diagonal in a row `< i` reachable from `i`.

**Proof.** (i) `(I − γP')z' = V'` and `(I − γP')z = V − γΔP z`; subtract. (ii) Rows `< i` of `(I − γP')z' = V'` involve only `z'_{<i}` and
read `(I − γP)_{<i} z'_{<i} = V_{<i}`, the same system `z_{<i}` solves, with a unique solution (P5a); so `z'_{<i} = z_{<i}` and the suffix
system is the remainder of the full one. (iii) Exhibited numerically. ∎ (DERIVED.) The statement is the linear clamp-row instance of
`dash-2005-emc` (CITED [V], PDF read by MARS, Def. 4, Thms 1–2): Dash's Thm 1 violations need feedback through the manipulated variable; a
causal `P` has none, and the record's nilpotency (`pow_card_eq_zero`) is the regime-N form of the same fact.

**Evidence.** `RUN[here]`: (i) `1.03e-15`; (ii) suffix re-solve vs full re-solve `4.4e-16`, displacement before `i` `0.0`; (iii) planted
dense row-stochastic `P`: suffix vs full `0.0761`, displacement before `i` `0.0761`. First pass: (i) `1.2e-15`. Rank-one price:
Sherman–Morrison `8.9e-16` (P9), `O(sd)` per candidate after the first solve (`piray-2021-linearrl` Eq. 5 is the same update on a fixed
chain, `sweep_resolvent.md` §3.7). External argument for the re-solve over a cached mixture: `momennejad-2017-sr`, `russek-2017-predictive`
(CITED [V], mechanism `[U]`) — a cached resolvent does not adapt to a change of *transition* structure.

**Mechanism.** **V-24 / D-7** (both halves filed, the feedback plant is the rejection region), **D-2** (FiP's interventional answer is the
model's own re-solved fixed point — the hazard the bed's oracle must not repeat, `sweep_causality.md` §3.4).

**Status.** PROVED-HERE (RUN both halves; Lean `emc_causal` [S]).

### C8 — Regime N versus regime S is a theorem boundary

**Statement.** The softmax corner keeps the diagonal (`P_ii > 0`), so `γP` is not nilpotent and neither `pow_card_eq_zero` nor
`occupancy_is_exact_inverse` applies; `(I − γP)^{-1}` is an infinite series and the brief's sentence "all `s` hops in one operator by
nilpotency" holds only in regime N. In regime S the finite sum is replaced by the P4 certificate, or the diagonal is masked (rows attend to
`j < i` with the `j = 0` value-zero sink absorbing row 0) and the finite sum is inherited exactly.

**Proof.** `Nilpotent.one_not_nilpotent` (`Nilpotent.lean:105`): `1^k ≠ 0`; a matrix with a positive diagonal has `(A^k)_{ii} ≥ (A_ii)^k >
0`. ∎ For the masked option, row 0's window is empty and the sink supplies the row (`V15Fork.Asink_row_sum`).

**Evidence.** `RUN[here]`: `max|(γP)^{32}| = 7.96e-08` with the diagonal, `0.0` strictly lower. `RUN[coord]`: BED-1's `Q` non-nilpotent,
`ρ(Q) = 0.9409`, `max|Q^{11}| = 0.477` (`THESIS_CORRECTIONS.md` C8). `LEAN`: `OracleSeparation.truncation_never_exact` (`:180`) — every rung
leaves a residual on the oracle's chain.

**Mechanism.** **V-25** (a regime-N theorem quoted on a regime-S operator has an empty domain), **P-3** (the brief's nilpotency sentence is
corrected here, not carried).

**Status.** LEAN (`one_not_nilpotent`) + RUN.

---

## 3. The four obstructions, restated as `sweep_expressivity.md` licenses them

All theorem numbers are as the sweep read them from the arXiv HTML (`sweep_expressivity.md` §0, §1, §3); the geometry throughout is the
record's `s = 64`, `d = 16`, `h = 1`, float32 (`p = 32`).

**3.1 Obstruction 1 — per-row independence.** Definitional; no theorem is needed and none is cited as one. The nearest published fixed
points are per-query (`ramsauer-2021-hopfield`: the Hopfield retrieval is a per-row object; `sweep_resolvent.md` §1.11) and the sharpness
failure of the row mixture is `velickovic-2025-softmaxnotenough`. Unconditional, but it licenses only "the reads are jointly constrained",
not any accuracy sentence. P3 of `sec_beds.md` §6.E is the prediction that makes it measurable: the harmonic residual separates the arms
while per-coordinate NRMSE does not.

**3.2 Obstruction 2 — hops need depth.** *Skyline (unconditional):* `sanford-2024-logdepth` Thm 4.2, `hop_k` at depth `⌊log₂ k⌋ + 2`;
`merrill-2025-littledepth` Thm 2, connectivity at `⌈log₂ n⌉` unrolls of a uniform block; `fagnou-2024-chacal` Thm 1, `⌈log₂(depth(G)+1)⌉`
layers for entity tracking (the same law, different constants — to be reconciled by whoever holds the depth lineage). *Lower bounds:*
`sanford-2024-logdepth` Cor. 4.3 — `Ω(log k)` **conditional** on the one-vs-two-cycle conjecture, `k = Θ(N^ξ)`, `mH = O(k^{1−ε})`;
`sanford-2024-graph-algorithms` Thm 3/19 — same conjecture, `mH = O(N^ε)`; `chen-2024-multilayer` Thm 1.1 — **unconditional** for
decoder-only softmax at depth `L` against `L`-sequential composition, hypothesis `H·d·p ≤ n^{2^{−4L}}`: at `L = 1`, `n = 64`, `64^{1/16} =
1.30` against `H·d·p = 512` — **vacuous at every geometry the record ran**; `sanford-2024-inductionheads` Thm 1 — unconditional at `t* = 2`,
hypothesis `h·m·p = Ω(n)`: `512 ≥ 64`, **not violated, so silent here**. *Width:* `yehudai-2025-depthwidth` — depth is not necessary at
linear width; the matched control must fix width as well as depth. *Circuit class:* `merrill-2023-parallelism` Thm 2 — log-precision
constant depth `⊆` uniform `TC⁰`, connectivity L-complete, linear systems P-complete, consequences conditional on class separations. What
the four theorems license together: an *asymptotic, lineage* argument that one softmax parameter set cannot compose `t* ≥ 2`
content-dependent hops, and nothing at the record's geometry.

**3.3 The DET-class relocation (what "one operator" may claim).** The exact committor is the solution of `(I − Q)q = R𝟙`; the exact `z` is
`(I − γP)^{-1}V`. Matrix inversion is `DET`-complete and `NL ⊆ DET ⊆ NC²` (`cook-1985-taxonomy`, CITED [V]). So the resolvent as a circuit
sits at or above the class the conditional bounds place reachability in: **the shape relocates the log-depth from the parameter stack into
the linear solve** — `O(s)` sequential rounds by forward substitution, or `O(log s)` parallel rounds by prefix doubling at `O(s³)`-class
work (`NOT MEASURED — needs a parallel-prefix kernel timing`). "Depth-1" in the paper means *one attention parameter set* (`P`, `γ`), and
the circuit-depth reading is disclaimed in the same paragraph (P-8, V-17). The `hop_k` graph is token-defined; the shape's reachability is
in the graph `P` *the layer computes*; the reduction "every `hop_k` instance is a committor instance of some `P`" is plausible and **NOT
FOUND** (`sweep_expressivity.md` §3.5) — until written, obstruction 2 is a lineage argument, not a theorem about the shape.

**3.4 Obstruction 3 — composition.** `peng-2024-transformer-limitations` Thm 1 (unconditional; identifier confirmed): one softmax layer errs
on `f(g(x))` with probability `≥ R/(3n log n)`, `R = n log n − H(d+1)p > 0`. At `n = 64`: `64·6 = 384` against `1·17·32 = 544` — **not
satisfied**, vacuous here; bites at `n ≳ 100` or `p = 16`. `kozachinskiy-2025-strassen` Thm 3.4 removes precision (infinite-precision,
asymptotic `n^{Ω(1)}` size). "A consequence *is* a composition" is licensed as a reduction sketch (`z' = R(P')V'` after `P' = S(P, a)`), not
as a theorem about the committor.

**3.5 Obstruction 4 — non-negativity — WITHDRAWN as written (C3).** For row-stochastic `P` the mixing matrix is non-negative (P5b) and stays
non-negative under absorbing rows (the Neumann series of a non-negative matrix), so `∂O_i/∂V_j ≥ 0` survives the resolvent and
`ceq/attention.py`'s min-entry test will read `0` on the shape. Absorption **redirects** mass — a walk that hits `a ∈ 𝒜_k` stays at `a`, so
the weight on every token the walk would have reached through `a` goes to zero (the F0 mechanism in boundary form). The paper writes "a
boundary row captures the mass that would have reached later positions", never "veto" or "negative influence" (P-7, V-23). What a boundary
condition changes is *where* mass goes — the support and weights of `Π_γ` — and that is the whole of the capability claim.

**3.6 D-1, stated with the right predictor.** `marion-2025-single-location` Cor. 2 is asymptotic (`d → ∞`, `L = o(d)`) with an `erf`
predictor; `duranthon-2026-softmax-advantage` Prop. 4.2 proves Bayes optimality for softmax proper on single-location regression. The record
ran `L/d = 4.00` (`READ MATHEMATICS.md:1044-1057` via `sec_refuted.md` §3.0). Every bed in the record asked for a scalar at one position; no
bed in the paper does.

**3.7 The three skylines and the two fellow approximators (R-SKY).** Skyline row on every bed: (i) the `⌊log₂ t*⌋ + 2` softmax stack (Thm
4.2; `3/5/7` at `t* = 2/8/32`), (ii) the wide constant-depth stack (`yehudai-2025-depthwidth`), (iii) the chain-of-thought decoder
(`merrill-2024-cot`). `wang-2024-incontext-td` and `xie-2026-softmax-rl` (CITED [V]) show linear-then-softmax transformers implement TD
policy evaluation layer by layer, so the deeper stack computes the *same* resolvent by iteration — the looped block `z^{(t+1)} = V +
γPz^{(t)}` run `K` times is the truncated Neumann sum with error exactly `γ^{K+1}/(1−γ)` (P4; `yang-2024-looped`, `gasteiger-2019-appnp`
power iteration). Fellow approximators, controls on every bed where `γ > 0` matters (C2): ChaCAL with the same `γ` and no boundary rows; an
InfSA-style Neumann read without boundaries. A nonlinear loop (`P` recomputed from `z`) is a DEQ (`bai-2019-deq`) and out of scope. "Beats
softmax" and "beats native" (`yang-2024-deltanet` / `dao-2024-ssd` on the linear corner, `sec_refuted.md` Part D) are banned sentences; the
shape's separate claims are exactness with a printed `δ`, one-read joint consistency, and the boundary-row mechanism, stated as properties.

---

## 4. Occupancy (cited before the delta is named)

The read operator `O = (1 − γ)P(I − γP)^{-1}V` on causal row-stochastic softmax `P`, solved as a triangular system, reverting to standard
attention at `γ = 0`, and motivated by the `log` depth law, is **`fagnou-2024-chacal`** (EMNLP 2024, Eq. 5, Eq. 7, Thm 1; CITED [V], abs +
HTML + ACL page; `0` files in the tree cite it — a V-7-on-search finding for the negatives section, since the round-1 filter discarded every
non-negative multi-hop operator during the signed programme). Its subquadratic blockwise evaluation with a reduced cross-block system is
**`zhao-2026-structuredsparse`** (CITED [V]). P1 and P5 of this file are re-derivations of ChaCAL and are presented as such; the
machine-checked containment and the `torch.equal` run are the record's. The Neumann closure `(I − γÂ)^{-1} − I` over a content-adaptive
attention, with a learnable per-head `γ` and the reading of the kernel as the fundamental matrix of an absorbing chain with one leak state,
is **`roffo-2026-infsa`** (CITED [V], full HTML; ReLU Frobenius-normalised base, non-causal, no softmax corner, no named absorbing sets,
global bound `γ/(1−γ)` only). The exact resolvent of a stochastic-like propagation operator as a neural layer on a fixed graph is
**`gasteiger-2019-appnp`**; `(I − γP)^{-1}` as policy evaluation and as the successor representation is **`bellman-1957-markovian`** and
**`dayan-1993-successor`**; the Katz index is **`katz-1953-status`**. An SCM written as the fixed point of a causally-masked transformer
map, with `do()` by clamping and re-solving and counterfactuals by abduction, is **`scetbon-2024-fip`** (CITED [V], full text; the fixed
point is the data-generating SCM over `d` variables, acyclic `d`-fold composition, no `γ`, no boundary sets, and its interventional answer
is the model's own re-solved fixed point — the D-2 hazard). The absorbing-boundary resolvent with a Woodbury re-solve after a barrier, on a
fixed environment chain, is **`piray-2021-linearrl`** Eq. 4–5 (CITED [V], PMC full text); the linear first-exit solve that yields a move is
**`todorov-2009-efficient`**. Harmonic propagation with clamped labelled nodes as absorbing boundary — the committor read as a learning
layer, 2003 — is **`zhu-2003-harmonic`** (CITED [V] DBLP/AAAI; formula `[U]`), with **`zhou-2003-consistency`** and
**`wu-2012-partially-absorbing`** beside it. The fundamental matrix `N = (I − Q)^{-1}` and `B = NR` are **`kemeny-1960-finitemarkov`** and
**`grinstead-1997-probability`** Thm 11.6; the committor as a discrete Dirichlet problem is **`metzner-2009-tpt-markov-jump`** and
**`e-2006-transition-paths`**; the harmonic-measure and electrical reading is **`doyle-1984-electric`**. The reach-avoid probability and its
naming identity with the committor is **`summers-2010-reach-avoid`** (with `abate-2008-reachability`); the Chebyshev action rule is
**`vanmoffaert-2013-chebyshev`**; several target sets in one linear system with a safety verdict is **`baier-2008-modelchecking`**. The
rank-one re-solve is **`sherman-1950-inverse-adjustment`** and **`hager-1989-updating`**; perturbation of the fundamental matrix is
**`schweitzer-1968-perturbation`**; the order of intervention and equilibration is **`dash-2005-emc`**. A `do()` inside attention — the
intervened token's value fixed and the token forbidden from attending, i.e. an absorbing row — is **`karbalayghareh-2026-doformer`** (CITED
[V] bioRxiv; gene perturbation, one intervention, no resolvent, no displacement read). Betti numbers of thresholded attention graphs as
features are **`kushnareva-2021-tda-attention`** and **`kushnareva-2022-betti`** (a preprint, cited as such); an intervention's effect on
persistence summaries as a causal estimand is **`kim-2026-topological-causal`**; directed-network persistence stability is
`turner-2019-quasimetric-rips`; the Mapper cover is `singh-2007-mapper` (records [V], landing [U]) with `carriere-2018-mapper-statistics`
for pre-fixable parameters; the persistence-derived CSR schedule consumed by an attention kernel is the author's own
`sharma-2026-kernels-22`. The occupancy matrices of the eight sweeps agree: the conjunction (content-dependent, causal, softmax, exact
solve, `γ = 0` parity) is ChaCAL's; boxes (e) absorbing constraint rows, (g) the interventional re-solve, (h) the safest-move read, (i) a
printed certificate, (j) learnable `γ` on a causal LM read are held by no attention source found, and (e)(g)(h) are separately owned on
fixed chains.

## 5. The delta, stated narrowly

The paper's object is not the resolvent. It is the composition every sweep reports NOT FOUND (`THESIS_CORRECTIONS.md` C1, verified against
the eight sweep files): (e) `K ≥ 2` absorbing constraint sets plus a goal set as boundary rows inside the content-dependent causal read (BOS
declared in the goal set); (g) the interventional re-solve `do(a)` with the displacement as a trained, jointly scored vector output; (h)
per-constraint committor / reach-avoid reads and the safest-move rule over candidate moves placed in the context; (i) a printed Neumann
certificate with a planted negative and the `δ·‖V‖_∞` units; (j) a learnable discount in a causal LM read, pinned by the LR test; (k)
machine-checked containment (three corners plus `γ = 0`) and the zero-gate segmentation iff; (l) the influence-Jacobian `β₀` barcode with a
directed-stability `δ`; (m) the Mapper cover → causal CSR schedule → certified resolvent. The name is the repository's own —
**Consequence-Equilibrium Attention (CEQ)** — and its operator is ChaCAL's on the record's three-corner base. Each component is a
composition of owned parts; none is "novel"; the absence is bounded by the sweeps' recorded queries (V-7). The pre-registered kill (C2): if
ChaCAL with a sink token matches the committor read within the TOST margin on a bed with prompt-named constraint sets, component (e) is not
a capability and the shape collapses to ChaCAL plus a certificate.

---

## 6. Lean targets — graded, in build order (plan input)

Grades: `[M]` machine-checkable now from what is in the tree plus Mathlib at `lean/lake-manifest.json:7`; `[S]` needs work at the
`V15Source.lean` scale; `[D]` deferred (needs probability or a Mathlib gap). Each line names the proposition it carries, the declaration it
builds on, the mechanism it is designed against, and whether it is independent of the lines above it (so the author may pick nodes in any
DAG-consistent order). Every target ships the refusal named beside it, as the record's files do (`sec_proved.md` §2.6). Price for every
line: `0 GPU-s`; `lake build` on this box.

| # | target | grade | carries | builds on | refusal shipped | mechanism | depends on |
|---|---|---|---|---|---|---|---|
| 1 | `gamma_zero_is_softmax` : `P * (1 − (0:ℝ)•P)⁻¹ * V = P * V` | [M] | P1 | `corner_softmax` (`V16Domain.lean:394`), `Matrix.inv_one` | `gamma_half_is_not_softmax` (a witness `P, V` with `P(1−½P)⁻¹V ≠ PV`) | V-3, V-24 | none |
| 2 | `bos_row_is_absorbing` : causal softmax row 0 `= e₀` | [M] | F1 | `Hop` definition, `Finset.range 1` | — (a definition unpacked) | V-25 | none |
| 3 | `later_boundary_unreachable` : lower-triangular `M`, `j > i` ⇒ `(M⁻¹ *ᵥ e_j) i = 0` | [M] | F2 | `Matrix.inv` of block-triangular; `Nilpotent.pow_entry_zero` pattern | — | V-8 | 5 |
| 4 | `softmax_corner_not_nilpotent` : `0 < P i i` ⇒ `(γ•P)^k i i > 0` | [M] | C8 | `one_not_nilpotent` (`Nilpotent.lean:105`) | `strict_lower_is_nilpotent` (= `pow_card_eq_zero`, already in tree) | V-25 | none |
| 5 | `lower_triangular_isUnit`, `diag_one_sub_smul_pos` | [M] | P5a | `Matrix.det_of_lowerTriangular` [U], `isUnit_iff_isUnit_det` [U] | `zero_diag_not_unit` witness | M-8 | none |
| 6 | `resolvent_fromBlocks`, `segmentation_blockdiag` (`StrictlyLower A`, cut `c`, `A i j = 0` for `j < c ≤ i` ⇒ `occupancy A n` zero across the cut) | [M] | P6 | `pathProd_eq_zero_iff` (`:129`), `pow_entry_zero` (`:52`), Mathlib `inv_fromBlocks_zero₂₁_of_isUnit_iff` [U] | `tiny_gate_does_not_cut` (`1e-300`-style witness: a nonzero cross entry gives a nonzero inverse block) | L-CERT, V-24 | none |
| 7 | `displacement_identity` : `(1 − γ•P') *ᵥ (z' − z) = (V' − V) + γ • ((P' − P) *ᵥ z)` given `(1 − γ•P) *ᵥ z = V`, `(1 − γ•P') *ᵥ z' = V'` | [M] | C6(i), P9 | ring algebra on `Matrix.mulVec` | `const_value_zero_displacement` (`V = 𝟙`, one changed stochastic row ⇒ `Δz = 0`) | V-24, D-5 | none |
| 8 | `subdiag_strictlyLower`, `subdiag_pow_entry`, `pathprod_is_chain_resolvent` | [S] | P2 | `pow_entry_zero`, `occupancy_is_exact_inverse`, `chain_path_product` (`V15.lean:74`) | `forward_map_fills_in` pattern at general `n` (`V15Source.lean:168`) | D-2 | none |
| 9 | `resolvent_sup_bound`, `isUnit_one_sub_smul`, `neumann_truncation_bound` (vector form, bound `γ^{K+1}/(1−γ)·M`) | [S] | P4 | `rowStochastic_perron` (`Contraction.lean:118`), `weighted_contraction` (`:72`), `occupancy_telescope` (`Occupancy.lean:52`) | `nonstochastic_breaks_bound` (row sum `3/2` witness) | V-24, V-10, L-CERT | none |
| 10 | `neumann_tail_attained` : `P ≥ 0` row-stochastic ⇒ the residual's row sums equal `γ^{K+1}/(1−γ)` | [S] | P4 equality | 9 | — (the declared V-3) | V-3 | 9 |
| 11 | `resolvent_is_triangular_solve` : `M⁻¹ *ᵥ v = fwdSub M v` | [S] | P5a | 5; `Fin n` strong induction | — | M-8 | 5 |
| 12 | `mixing_matrix_rowStochastic` : `(1−γ)•P*(1−γ•P)⁻¹` row sums `1`, entries `≥ 0` | [S] | P5b, C3 | 9 | `bare_read_row_sum` (`= 1/(1−γ)`) | V-17 | 9 |
| 13 | `causal_forward_only` : lower-triangular `M`, `(M⁻¹ *ᵥ e_i) j = 0` for `j < i` | [S] | P9 | 5, 11 | `dense_P_displaces_backward` witness | V-24 | 5 |
| 14 | `emc_causal` : suffix re-solve `=` full re-solve for lower-triangular `P'` | [S] | C6(ii) | 7, 13 | `emc_fails_with_feedback` (a `3×3` non-causal witness) | V-24, D-7 | 7, 13 |
| 15 | `committor_is_resolvent_read` (a) `q = Q*ᵥq + r`, `IsUnit (1 − Q)` ⇒ `q = (1−Q)⁻¹ *ᵥ r` | [M] | P3 | one line | — | D-2 | none |
| 16 | `isUnit_one_sub_of_perron` (b) : `PerronCertificate Q w ρ`, `ρ < 1` ⇒ `IsUnit (1 − Q)` | [S] | P3 | `Contraction.lean:24-37` | `unit_needs_rho_lt_one` (`Q = 1`) | V-25 | none |
| 17 | `bed1_committor_eq` (c) : BED-1's generator solve `=` `(1−Q)⁻¹ *ᵥ (R *ᵥ 𝟙_B)` | [S] | P3 | 15, `ceq/beds/bed_1.py:188-198` unpacked | — | D-2, V-3 | 15 |
| 18 | `reach_avoid_sum_one` : rows of `P` sum to `1`, `IsUnit (1−Q)` ⇒ `Σ_k (1−Q)⁻¹ *ᵥ (R_k *ᵥ 𝟙) = 𝟙` | [S] | P8 degeneracy | 15 | `no_goal_forces_max_ge_inv_K` (corollary, `max_k ≥ 1/K`) | V-12 | 15 |
| 19 | `sherman_morrison_row` : the rank-one displacement formula | [S] | P9 | Mathlib `Matrix.det_one_add_col_mul_row` / Sherman–Morrison [U] | — | M-8 | 5 |
| 20 | `hitting_time_transform` : `(1−γ)·[(1−γ•P)⁻¹ *ᵥ 𝟙_{𝒜}] i = E_i[γ^τ]` | [D] | P3 first clause | needs a finite-Markov-chain / hitting-time development not in Mathlib at the pinned rev [U] | — | — | 15 |
| 21 | `f1_cantelli_union` (annex M9's soft-mask certificate) | [D] | L-CERT for F1 masks | Mathlib concentration [U] | — | L-CERT | none |

Build order is the row order: 1–7 are one evening of `[M]` items and are mutually independent; 8–19 are `[S]` and each names its
dependencies; 20–21 are `[D]` and are not on the critical path (P3's probability clause is carried by the RUN and the DERIVED proof until
then). The contract's `#18, #19, #22 [M]` tags name no declaration in the tree (`sec_proved.md` §2.0, P-11 potential); the list above
replaces them and nothing is cited as `[M]` until it builds (`L-LEAN`: no arm is trained before its identity theorems are green).

---

## 7. Limits (collected once)

Every `RUN[here]` number is one numpy float64 draw at `s = 32`, `d = 4`, `γ = 0.6`, seed 0, on this CPU, an identity check and not a
statistic; the coordinator's `RUN[coord]` numbers are one torch draw at `s = 64`, `d = 16`; neither carries an interval. The `E[γ^τ]` check
truncates the hitting-time series at `4000` terms (`ρ(Q) = 0.513`, so the tail is below `10^{-1000}`); the `γ ↑ 1` limit was checked at `1 −
10^{-6}` and read the expected `O(1 − γ)` gap, not `0`. C6(ii)'s "settle then intervene" is the linear clamp-row reading (freeze `z_{<i}`,
re-solve the suffix); `dash-2005-emc`'s theorems concern SCMs obtained by causal ordering of a dynamic system with feedback, and the
correspondence between the two readings is DERIVED here, not read from Dash. The P7 obstruction clauses are all asymptotic and every
unconditional one is vacuous at the record's geometry; the `hop_k → committor` reduction is NOT FOUND and unwritten, so obstruction 2 is a
lineage argument. P8 has no cell, no paired sd, no measured `t*`; the multichain hazard of `misra-2023-safety-constrained-mdp` is carried,
not resolved. The cost line of P5 is a per-op microbenchmark floor (`sec_cost.md` Limits) and the chunked, CSR, Mapper and parallel-prefix
paths have no kernel in the tree. Lean grades `[M]` for lines 5 and 6 (and the `[S]` grade of line 19) rest on Mathlib lemma names marked `[U]` (not opened this
session); a missing lemma moves the line to `[S]`, not to a claim. Bib keys were checked against `references.bib` titles this session for every key
cited here; theorem numbers inside cited sources are the sweeps' HTML readings and inherit their `[U]` marks where the sweeps say so. No
code file, no git write, no external fetch was made by this planet.

