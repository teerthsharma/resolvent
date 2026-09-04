# 2. What the record proved

Evidence classes used below: **RUN** — executed this session (2026-09-03, this box, Lean
`leanprover/lean4:v4.7.0` per `lean/lean-toolchain`, torch 2.5.1 float64); **READ**
`path:line`; **DERIVED** — steps written out; **CITED** with `[V]`/`[U]`. Every theorem
statement in this section was transcribed from the `.lean` text, not paraphrased from
a report; the hypotheses are quoted verbatim.

## 2.0 The inventory

| file | `theorem`+`lemma` | lines | round | carries |
|---|---|---|---|---|
| `lean/CEQ/Contraction.lean` | 5 | 161 | pre-R11 | Perron-certificate contraction; row-stochastic corollary; the `σ_max` witness |
| `lean/CEQ/Nilpotent.lean` | 4 | 114 | pre-R11 | strictly-lower ⇒ nilpotent at the cardinality |
| `lean/CEQ/Occupancy.lean` | 3 | 89 | pre-R11 | finite Neumann telescope; exact inverse under `A^N = 0` |
| `lean/CEQ/OracleSeparation.lean` | 12 | 205 | R8 | `oracle ≠ resolvent`; truncation never exact |
| `lean/CEQ/OrbitBound.lean` | 5 | 84 | pre-R11 | caustic Theorem 1 (not architecture-bearing here) |
| `lean/CEQ/Refcount.lean` | 10 | 183 | pre-R11 | floor = Σ(refcount−1) (KV cache; not architecture-bearing here) |
| `lean/CEQ/V15.lean` | 23 | 328 | R11 | chain path product, prefix-logit mask, parity, the parity-clause refutation, affine scan |
| `lean/CEQ/V15Fork.lean` | 11 | 251 | R11 | the telescope head `Asink`; `no_row_stochastic_with_drive_values` |
| `lean/CEQ/V15Kernel.lean` | 30 | 640 | R11 | scan-blindness of delay and power-law (BED-K) |
| `lean/CEQ/V15Phase.lean` | 26 | 395 | R11 | polar carrier; closed magnitude; `Z₂` winding |
| `lean/CEQ/V15Source.lean` | 7 | 201 | R11 | source recovery is a first-order difference |
| `lean/CEQ/V16Domain.lean` | 33 | 594 | R12 | `#2` on the closed support; three corners; domain census |
| **total** | **169** | 3,245 | | |

Provenance of the counts: `grep -cE '^(theorem|lemma) ' lean/CEQ/*.lean` — **RUN**, 169
total, per-file as tabled; `lean/CEQ.lean` (33 lines, the import root) contributes 0.
`lake build` in `lean/` — **RUN**, exit 0, no output (warm cache; the same reading
`V16_LEAN_DOMAIN.md:394-399` calls "weak evidence" and backs by force-re-elaboration,
which this session did not repeat). `grep -n '\bsorry\b' lean/CEQ/*.lean` — **RUN**,
six hits, every one the prose phrase "No `sorry`" in a file header
(`V15.lean:41`, `V15Fork.lean:44`, `V15Kernel.lean:72`, `V15Phase.lean:72`,
`V15Source.lean:60`, `V16Domain.lean:75`). `#print axioms` directives — **RUN**
`grep -c`: 96 in-file (`V15Phase` 31, `V15Source` 8, `V16Domain` 57, including the
nine-line TRAIN-GATE re-check at `V16Domain.lean:586-594`); the axiom set
`[propext, Classical.choice, Quot.sound]` with no `sorryAx` is **READ** from
`README.md:101-104` and, declaration by declaration for `V16Domain`, from
`V16_LEAN_DOMAIN.md:417-465`. Mathlib pinned at `lean/lake-manifest.json:7`
rev `32983874…` (**READ**).

The record's own status ledger for these theorems is `CEQ_V16_CONTRACT.md` PART II
(`:135-160`, **READ**): items `#1, #2(re-stated), #3, #5a, #5b, #6, #7, #15, #16` are
`[M]`-proved; `#4 resolvent_eq_path_sum`, `#12`, `#13`, `#14 committor_eq_harmonic`,
`#17` are `[S]`; `#10, #11` are `[D]`. `CEQ_V20_R15_CONTRACT.md:260-262` (**READ**)
moves `#14 committor` to `[D]` and adds `#18, #19, #22 [M]` and `#17′, #20, #21, #23
[S]` — none of which exists under `lean/CEQ/` (**RUN**: `grep -rn 'segment' lean/`
returns nothing; the `[M]` tag there is the contract's *grade*, not a build). This is
`MISTAKES.md` P-11 (`:1597`, a contract citing its own `[M]` item as settled) in
potential form, and the paper cites only declarations that build.

## 2.1 (a) The resolvent identities — `Occupancy`, `Nilpotent`, `OracleSeparation`, `V15Source`

**`CEQ.Occupancy.occupancy_telescope`** (`Occupancy.lean:52`). Over any `[Ring R]`,
`(A : R) (N : ℕ) : (1 - A) * occupancy A N = 1 - A ^ N`, with
`occupancy A N := ∑ k in Finset.range N, A ^ k` (`:42`). Its mirror
`occupancy_telescope'` (`:66`) multiplies on the other side. **Certifies:** the
truncated Neumann sum is the inverse of `1 − A` up to the exact tail `A^N`, with no
convergence hypothesis. **Instance:** the tail is what I4 below measures: at `K = 16`,
`γ = 0.7`, `‖·‖_∞` residual `0.007754350466241788` against `γ^{17}/(1−γ) =
0.007754350466240224` (**RUN**, `$SCRATCH/shape_identities.py`). Note the header's
citation correction: Dayan 1993 eq. 3.1 is `(I − Q)^{-1}` with **no** `γ`
(`Occupancy.lean:6-17`, **READ**; the source itself is [U] this session).

**`CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent`** (`:83`). `(A : R) (N : ℕ)
(hA : A ^ N = 0) : (1 - A) * occupancy A N = 1 ∧ occupancy A N * (1 - A) = 1`.
**Certifies:** under nilpotency the finite sum *is* the two-sided inverse, exactly.

**`CEQ.Nilpotent.pow_entry_zero`** (`Nilpotent.lean:52`). For
`StrictlyLower A := ∀ i j : Fin n, (i : ℕ) ≤ (j : ℕ) → A i j = 0` (`:46-47`):
`(hA : StrictlyLower A) : ∀ (k : ℕ) (i j : Fin n), (i : ℕ) < (j : ℕ) + k → (A ^ k) i j = 0`.
**Certifies:** a nonzero entry of `A^k` needs `k` strictly increasing hops. This is the
path-counting induction the segmentation target of §2.8 (6) re-uses verbatim.

**`CEQ.Nilpotent.pow_card_eq_zero`** (`:77`). `(hA : StrictlyLower A) : A ^ n = 0`.
**`CEQ.Nilpotent.occupancy_is_exact_inverse`** (`:96`). `{A : Matrix (Fin n) (Fin n) ℝ}
(hA : StrictlyLower A) : (1 - A) * occupancy A n = 1 ∧ occupancy A n * (1 - A) = 1`.
**Certifies:** for the strictly causal operator `ceq/nonnormal.py` builds
(`.tril(-1)`, asserted by `tests/w3b/test_w3b_lean_nilpotent.py:118`, **READ**), the
resolvent is a terminating sum of exactly `n` terms — *no sign hypothesis, no
magnitude hypothesis, no norm*. **Instance:** I2 — with `A[i, i−1] = a_i` from
`make_equilibrium_batch(8, 64, 24)`, `torch.linalg.matrix_power(A, 64)` reads
max-abs `0.0` and `(I − A)^{-1}` equals `ceq/arm_smprime.py:144 path_product` to
max-abs `0.0` (**RUN**); `tests/w2/test_w2_nonnormal.py:124` asserts `ρ(A) < 1e-12`
(**READ**). **`one_not_nilpotent`** (`:105`) records that `1 ^ k ≠ 0` for `0 < n`:
the strictness is load-bearing, and §2.8 (5) is where it bites the shape.

**`CEQ.OracleSeparation.not_isNilpotent`** (`OracleSeparation.lean:148`). With
`Nonneg Q := ∀ i j, 0 ≤ Q i j` (`:76`) and `SymmSupport Q := ∀ i j, 0 < Q i j → 0 < Q j i`
(`:82`): `(hQ : Nonneg Q) (hsupp : SymmSupport Q) {i j : Fin n} (hij : 0 < Q i j) :
¬ IsNilpotent Q`. **`oracle_ne_resolvent`** (`:166`): `(A Q) (hA : StrictlyLower A)
(hQ : Nonneg Q) (hsupp : SymmSupport Q) {i j} (hij : 0 < Q i j) : Q ≠ A`.
**`truncation_never_exact`** (`:180`): same hypotheses, `(N : ℕ) : (1 - Q) *
occupancy Q N ≠ 1`. **`zero_not_a_counterexample`** (`:193`) records that dropping
`hij` makes the statement false (the zero matrix is non-negative, symmetric-support,
nilpotent) — a hypothesis whose removal leaves a theorem true is a vacuous control in
proof form (header `:187-192`, and `MISTAKES.md` V-11 by analogy). **Certifies:** an
undirected-walk transient block is never the arm's strictly-causal operator, at the
operator level, before any label is drawn (`MISTAKES.md` D-2 `:710`). **Instance:**
BED-1's chain, `ceq/beds/bed_1.py:build(T=0.25, jitter=0.05, seed=11)`: `Q = P_TT` on
9 transient nodes reads `Nonneg True`, `SymmSupport True`, `ρ(Q) = 0.9408612510154677`,
`max|Q^11| = 0.47739216729378814 ≠ 0` (**RUN**). The oracle's operator is, as the
theorem says, not nilpotent, and the arm's is.

**`CEQ.V15Source.source_is_first_order_difference`** (`V15Source.lean:108`).
`{A : Matrix (Fin n) (Fin n) ℝ} (hA : StrictlyLower A) (h r : Fin n → ℝ)
(hfwd : r = CEQ.Occupancy.occupancy A n *ᵥ h) : h = r - A *ᵥ r`. **`two_sources_recovered`**
(`:128`): `(hfwd : r = occupancy A n *ᵥ h₁ + occupancy A n *ᵥ h₂) : h₁ + h₂ = r - A *ᵥ r`.
**`no_fill_in`** (`:143`): `{i j} (hij : i ≠ j) (hA : A i j = 0) : (1 - A) i j = 0`, over
any ring. **`forward_map_fills_in`** (`:168`): the witness `shift3` on `Fin 3` with
`shift3 2 0 = 0`, `(1 - shift3) 2 0 = 0`, `(occupancy shift3 3) 2 0 = 1`.
**`source_entry`** (`:188`): `(r - A *ᵥ r) i = r i - ∑ j, A i j * r j`. **Certifies:**
inverting the resolvent costs one mat-vec against the hop and a subtraction, `O(nnz A)`,
with no linear system — and the claim is against something denser (the forward map
fills in). **Instance:** two planted sources at `[36, 80]`, propagated-field error
`8.882e-16` (`V15_X35P_SOURCE.md:33, :235`, **READ**); `nnz(1 − A) = 251` vs
`nnz(W) = 1703` at `n = 128, d = 5` (`V15Source.lean:40-42` header, citing
`tests/x35p/test_source.py`, **READ**). The file's refused reading
`inverse_identity_is_vacuous` (`:83`, `(u : Rˣ) : (u : R) * ((u⁻¹ : Rˣ) : R) = 1`) is
listed under §2.6.

**`CEQ.Contraction.weighted_contraction`** (`Contraction.lean:72`). For a
`PerronCertificate A w ρ` (`nonneg : ∀ i j, 0 ≤ A i j`, `w_pos : ∀ i, 0 < w i`,
`dominates : ∀ i, ∑ j, A i j * w j ≤ ρ * w i`, `:59-62`): `(hc) (v) (M)
(hM : ∀ j, |v j| ≤ M * w j) (i : n) : |(A.mulVec v) i| ≤ ρ * M * w i`.
**`rowStochastic_perron`** (`:118`): `(hP : RowStochastic P) (γ : ℝ) (hγ : 0 ≤ γ) :
PerronCertificate (γ • P) (fun _ => 1) γ`. **`expander_expands_l2`** (`:151`):
`!![1,0;1,0]` is row-stochastic and sends `(1,0)` to `(1,1)`, so `σ_max ≥ √2`.
**Certifies:** the shape's `γP` contracts in the sup norm at modulus `γ`, and the
2-norm is the wrong metric for that claim. These three are the only theorems in the
tree stated for a row-stochastic `P` with a diagonal, and they are what §2.8 (4) builds
on.

## 2.2 (b) The three-corner family and the telescope — `V15`, `V15Fork`, `V16Domain`

**`CEQ.V15.chain_path_product`** (`V15.lean:74`). For `chain a b y₀` with
`chain (s+1) = a (s+1) * chain s + b (s+1)` (`:58-60`): `(a b : ℕ → ℝ) (y₀ : ℝ) (m n : ℕ)
(h : m ≤ n) : chain a b y₀ n = (∏ k in Ico (m + 1) (n + 1), a k) * chain a b y₀ m +
∑ s in Ico (m + 1) (n + 1), (∏ k in Ico (s + 1) (n + 1), a k) * b s`. **Certifies:** every
coefficient is a *product* of consecutive gates along the path, and this is exactly
the last row of `(I − A)^{-1} b` for the sub-diagonal `A` (I2: last row vs
`scale/negation_scope.py:286 equilibrium_oracle`, max-abs `6.217248937900877e-15`,
**RUN**; the oracle's docstring at `:290-295` says the same in prose, **READ**).

**`CEQ.V15.prefix_logit_mask`** (`:128`). `(a : ℕ → ℝ) (ha : ∀ k, 0 < a k) {i j : ℕ}
(hij : j ≤ i) : W (fun k => Real.log (a k)) i j = ∏ k in Ico (j + 1) (i + 1), a k`, with
`W g i j := Real.exp (scan g i - scan g j)` and `scan g i := ∑ k in range (i+1), g k`
(`:114, :117`). **Domain census against BED-M:** `bedM_overlap_old_two :
bedM.countP satOldTwo = 1` (`V16Domain.lean:302`, `by decide`) — **1 of 3** drawn values
`{−1, 0, +1}` satisfy `0 < a`; `MISTAKES.md` V-25 (`:1954`) is this theorem. Against
BED-K: empty (no gate sequence exists in the bed; `V16_LEAN_DOMAIN.md:184-186`, **READ**).
**`prefix_logit_computes_chain`** (`:140`) binds #1 to #2:
`chain a b 0 i = ∑ s in Ico 1 (i + 1), W (fun k => Real.log (a k)) i s * b s`.

**`CEQ.V15.parity_sign`** (`:190`). `(p : ℕ → ZMod 2) {i j} (hij : j ≤ i) :
chi (pscan p i - pscan p j) = ∏ k in Ico (j + 1) (i + 1), chi (p k)`. **Instance:**
`parity = Z₂ winding`, `torch.equal`, `0 / 4096` disagreements (`workdonenewseal.md:105`,
**READ**).

**`CEQ.V15.gate_zero_logit_identity`** (`:251`). `(g) (hg : ∀ k, g k = 0) (q : ℕ → ℕ → ℝ)
(i j) : q i j + (scan g i - scan g j) = q i j`. **Certifies:** the additive-logit
reading gives bitwise parity at `g ≡ 0`; the multiplicative-unnormalized reading does
not (§2.6, `gate_zero_not_stochastic`).

**`CEQ.V15.bounded_gates_stable`** (`:266`). `(w : ℝ) : 0 < Real.exp (-softplus w) ∧
Real.exp (-softplus w) ≤ 1`, with `bounded_gates_prod_le_one` (`:271`) and
`bounded_gates_antitone` (`:279`, `h1 : j' ≤ j`, `h2 : j ≤ i`). **Domain census:**
`bedM_overlap_six : bedM.countP satSix = 0` (`V16Domain.lean:306`) — **0 of 3**; and for
the whole family `six_misses_every_bedM_value (w : ℝ) : exp(−softplus w) ≠ −1 ∧ ≠ 0 ∧ ≠ 1`
(`:321`). #6 is decoration on BED-M by theorem.

**`CEQ.V15.scan_assoc`** (`:314`). `(p q r : Aff) : affComp (affComp p q) r =
affComp p (affComp q r)` for the *affine* monoid `affComp p q := (q.1 * p.1, q.1 * p.2 + q.2)`
(`:297`), with `affApply_affComp` (`:302`) proving `affComp` is composition and
`chain_step_eq_affApply` (`:308`) that the elements are the recurrence's own steps.
**Certifies:** the log-depth parallel route is licensed by *this* monoid, not by
`add_assoc` (header `:33-39`).

**The telescope — `CEQ.V15Fork`.** With
`Asink a i j := if j = 0 then ∏ k in Ico 1 (i+1), a k else if j ≤ i then (1 - a j) *
∏ k in Ico (j+1) (i+1), a k else 0` (`V15Fork.lean:67-70`) and
`Vsink a b j := if j = 0 then 0 else b j / (1 - a j)` (`:62`):

- **`Asink_row_sum`** (`:81`). `(a : ℕ → ℝ) (i : ℕ) : ∑ j in range (i + 1), Asink a i j = 1`
  — **no hypothesis on `a` at all**; the reweighting telescopes.
- **`Asink_nonneg`** (`:116`). `(h0 : ∀ k, 0 < a k) (h1 : ∀ k, a k < 1) (i j) : 0 ≤ Asink a i j`.
- **`Asink_computes_chain`** (`:140`). `(ha : ∀ k, a k ≠ 1) (i) : ∑ j in range (i + 1),
  Asink a i j * Vsink a b j = chain a b 0 i`.
- **`no_row_stochastic_with_drive_values`** (`:195`). `(ha : ∀ k, 0 < a k) (A : ℕ → ℝ) (n)
  (hstoch : ∑ j in Ico 1 (n + 2 + 1), A j = 1) (hexact : ∀ b, ∑ j in Ico 1 (n + 2 + 1),
  A j * b j = chain a b 0 (n + 2)) : False` — the label `y_{n+2}` on `b ≡ 1` exceeds `1`
  (`chain_one_gt_one`, `:169`) and leaves the convex hull of its drives.
- **`Asink_eq_hop`** (`:245`). `(ha : ∀ k, 0 < a k) {i j} (hj : j ≠ 0) (hij : j ≤ i) :
  Asink a i j = (1 - a j) * W (fun k => Real.log (a k)) i j` — the whole repair is one
  factor `(1 − a_j)` and a value-zero BOS slot.

**Certifies:** one causal row-stochastic head — a softmax row with a key-side bias
`s_j = log(1 − a_j)`, a query-independent scan term, and a value-zero sink at `j = 0` —
reproduces the chain label exactly; the normalizer is a change of value units
`1/(1 − a_j)`, not an obstruction (`workdonenew.md:40, :105`, **READ**). **Instance:**
label bind `2.2e-16` at `s = 8` (`V15_JUPITER2_FORK.md:602, :666`, **READ**); ARM PL
`6.6613381477509392e-16`, ARM PHASE `9.155133597044475e-16`, ARM SMPRIME `5.919777e-16`
with real part exactly `0.000000e+00` (`workdonenewseal.md:91-96`, `V16_ARM_SMPRIME.md`
verdict (a), **READ**). Planted negatives for the bind: drop key bias `0.9749…`, drop
value rescale `0.9165…`, drop BOS sink `1`, half key bias `0.4845…`
(`workdonenewseal.md:114-122`, **READ**) — a rejection region, which the two-branch
form lacked (`MISTAKES.md` V-24 `:1658`). **Domain census:** `Asink_nonneg` needs
`0 < a < 1` (0 of 3 on BED-M); `Asink_computes_chain` needs only `a ≠ 1` (2 of 3; the
band value `+1` is the excluded point, and `V16_ARM_SMPRIME.md` verdict (h) reports the
`β = 1` bind failing at `1.335288` exactly there, **READ**).

**The family — `CEQ.V16Domain`.** `Hop β g qk i j := if j ≤ i then num g qk i j /
(Znorm g qk i) ^ β else 0` with `num g qk i j := exp((scan g i − scan g j) + qk i j)` and
`Znorm g qk i := ∑ j in range (i+1), num g qk i j` (`V16Domain.lean:366-378`; `β` enters
through `Real.rpow`, so `Z^0 = 1` unconditionally).

- **`three_corners_containment`** (`:433`). `{g₀ : ℕ → ℝ} (hg : ∀ k, g₀ k = 0) (g : ℕ → ℝ)
  (qk : ℕ → ℕ → ℝ) : (∀ i j, Hop 1 g₀ qk i j = softmaxAttn qk i j) ∧ (∀ i j, Hop 0 g₀ qk i j
  = linearAttn qk i j) ∧ (∀ i j, Hop 0 g (fun _ _ => 0) i j = CEQ.V15.Wc g i j)`.
- **`corners_are_distinct`** (`:445`). `Hop 1 0 0 1 0 = 1/2 ∧ Hop 0 0 0 1 0 = 1 ∧ … ≠ …`.
- **`corner_path_product_is_the_gate_product`** (`:414`). `(ha : ∀ k, 0 < a k) {i j}
  (hij : j ≤ i) : Hop 0 (fun k => Real.log (a k)) (fun _ _ => 0) i j = ∏ k in Ico (j+1) (i+1), a k`.
- **`gate_zero_beta_zero_is_linear_attention`** (`:483`). `(hg : ∀ k, g k = 0) (qk) :
  (∀ i j, Hop 0 g qk i j = linearAttn qk i j) ∧ ∃ qk' i j, Hop 0 g qk' i j ≠ softmaxAttn qk' i j`.
- **`softmax_row_sum_one`** (`:464`), **`gate_zero_beta_zero_row_not_one`** (`:500`,
  `(hi : 1 ≤ i)`), **`beta_one_row_is_one`** (`:509`): `β`, not `g`, is the switch that
  decides softmax-class membership.

**Certifies:** softmax, linear attention and the path product are three settings of
one operator; the corners are distinct; and the original parity clause cannot be
re-claimed. **Instance:** corners `|c₁−c₂| = 4.472918`, `|c₁−c₃| = 1.144938`,
`|c₂−c₃| = 5.335671`; the `β = 1` corner bitwise against `softmaxAttn`; row sums
`1.000000` at `β = 1` on every row vs `[1.312192, …, 10.293107]` at `β = 0`
(`V16_ARM_SMPRIME.md:28-32`, **READ**). Against `ceq/lm.py`'s own softmax the corner is
**not** bitwise — `1.110223e-16` on `19/64` entries (verdict (c), **READ**) — which is the
reference point for any "bitwise" claim the shape makes (§2.8 (1)).

## 2.3 (c) The segmentation / zero-gate certificate — `V16Domain`

With `gateOf m θ := (m : ℂ) * exp(θ·I)` and `pathProd m θ i j := ∏ k in Ico (j+1) (i+1),
gateOf (m k) (θ k)` (`V16Domain.lean:92, :97`; no `log` anywhere):

- **`pathProd_abs`** (`:121`). `(h0 : ∀ k, 0 ≤ m k) (i j) : Complex.abs (pathProd m θ i j)
  = ∏ k in Ico (j + 1) (i + 1), m k`.
- **`pathProd_eq_zero_iff`** (`:129`). `(i j) : pathProd m θ i j = 0 ↔ ∃ k ∈ Ico (j + 1)
  (i + 1), m k = 0` — no hypothesis.
- **`no_prefix_scan_represents_a_zero_gate`** (`:165`). `(C : ℕ → ℂ) (m θ) {i j}
  (hz : ∃ k ∈ Ico (j + 1) (i + 1), m k = 0) : Complex.exp (C i - C j) ≠ pathProd m θ i j`.
- **`prefix_logit_mask_restated`** (`:221`). `(h0 : ∀ k, 0 ≤ m k) (h1 : ∀ k, m k ≤ 1) {i j}
  (hij : j ≤ i)`: five clauses — magnitude law, `≤ 1`, `= 1 ↔ ∀ k ∈ Ico (j+1) (i+1), m k = 1`,
  `(∃ k ∈ Ico (j+1) (i+1), m k = 0) → pathProd m θ i j = 0`, and
  `(∀ k, 0 < m k) → pathProd m θ i j = CEQ.V15Phase.Wp m θ i j`.
- **`bedM_gate_exact`** (`:251`). `{a : ℝ} (ha : a = -1 ∨ a = 0 ∨ a = 1) : gateOf (magOf a)
  (argOf a) = (a : ℂ) ∧ 0 ≤ magOf a ∧ magOf a ≤ 1`.
- **`path_product_corner_fails_at_a_zero_gate`** (`:424`). The real `β = 0, QK-off`
  corner of `Hop` cannot equal `pathProd` wherever a zero magnitude sits on the path,
  for any `g`.

**Certifies:** a zero gate annihilates the path product exactly and only then; no
exponential prefix scan `C` — none — can carry it; the corpus's `{−1, 0, +1}` are all
ordinary points of the re-stated theorem. **Domain census:** `bedM_overlap_new_two :
bedM.countP satNewTwo = 3` (`:304`) — **3 of 3**; `bedMProp_overlap_new_two = 2`
(`:310`) — 2 of 2. The cost is a *computational route*, not a conclusion: the scan
survives only as a segmented scan that resets at zeros or as the product itself
(header `:47-55`; `ceq/arm_smprime.py:157-161` implements the product and marks the
segmented scan as the upgrade, **READ**). **Instance:** the `exp_scan` planted negative
reads `nan` on **133,120 / 133,120** causal pairs of BED-M at `n = 128, s = 64`
(`workdonenewseal.md:122`; `V16_ARM_SMPRIME.md:519-527`, **READ**); `96.78%` of those
pairs annihilate, and the annihilation MCC reads `1.000000` corpus-alone, `0.639571`
for the zero-step arm, `0.000000` for `exp_scan` (same source). This is the F0 of
`CEQ_V20_R15_CONTRACT.md:212-213` ("zero-gate segmentation (Lean #22 [M]) [RUN: 8.9e-16,
595×]", **READ**); **no Lean declaration with that content exists in the tree** (RUN
grep), and the producing file of the `595×` number was not located — see gaps.

## 2.4 (d) The scan-blindness theorems — `V15Kernel`

With `vecRec F s₀ x` the first-order recurrence over any state type (`V15Kernel.lean:97`),
`firstOrder f y₀ x := vecRec f y₀ x` (state = output, `:110`), `affineRec α β y₀ x`
(`:114`), and `IsDelay d y x := ∀ i, y (i + d) = x i` (`:120`):

- **`first_order_cannot_delay`** (`:140`). `(f : ℝ → ℝ → ℝ) (y₀ : ℝ) {d : ℕ} (hd : 1 ≤ d) :
  ¬ ∀ x : ℕ → ℝ, IsDelay d (firstOrder f y₀ x) x` — `f` arbitrary, not affine.
- **`affine_first_order_cannot_delay`** (`:166`), the instrument's own model.
- **`first_order_delays_constant_input`** (`:177`). `(c y₀ : ℝ) (d : ℕ) : IsDelay d
  (affineRec 0 1 y₀ (fun _ => c)) (fun _ => c)` — the per-sequence reading is false.
- **`delay_realizable_at_dimension_d`** (`:235`). `(d) (x) : IsDelay d (fun i => vecRec
  (shiftStep d) 0 x i (Fin.last d)) x` — the naive reading is false; a shift register is
  first-order and delays at state dimension `d + 1`.
- **`delay_forces_state_injective`** (`:253`). `{S : Type*} (F : S → ℝ → S) (s₀ : S)
  (g : S → ℝ) (e : ℕ) (h : ∀ x, IsDelay (e + 1) (fun i => g (vecRec F s₀ x i)) x) :
  Function.Injective (fun u : Fin (e + 1) → ℝ => vecRec F s₀ (pad u) e)` — no linearity,
  no dimension.
- **`linear_first_order_cannot_delay_beyond_state_dim`** (`:360`). `(e : ℕ) (A : (Fin k → ℝ)
  →ₗ[ℝ] (Fin k → ℝ)) (B : Fin k → ℝ) (s₀ : Fin k → ℝ) (C : (Fin k → ℝ) → ℝ) (hk : k < e + 1)
  (h : ∀ x, IsDelay (e + 1) (fun i => C (vecRec (linStep A B) s₀ x i)) x) : False`.
- **`GL_weights_ratio_recurrence`** (`:405`). `(α : ℝ) (k : ℕ) : glwClosed α k = glw α k` —
  the contract's binomial closed form equals the ratio recurrence `bed_k.py` runs.
  **`GL_weights_alpha_zero`** (`:418`), **`GL_weights_alpha_one`** (`:426`),
  **`GL_weights_pos`** (`:435`, `hα : 0 < α`), **`GL_weights_strictAnti`** (`:447`,
  `h0 : 0 < α`, `h1 : α < 1`).
- **`GL_not_geometric`** (`:489`). `(h0 : 0 < α) (h1 : α < 1) : ¬ ∃ β γ : ℝ, ∀ k, glw α k =
  β * γ ^ k`; **`GL_not_geometric_tail`** (`:593`) from lag 1.
- **`first_order_cannot_powerlaw`** (`:521`). `(h0 : 0 < α) (h1 : α < 1) (γ β y₀ : ℝ) :
  ¬ ∀ (x : ℕ → ℝ) (i : ℕ), affineRec γ β y₀ x i = kconv (glw α) x i`;
  **`first_order_cannot_powerlaw_strict`** (`:624`) at the strictly causal alignment
  `bed_k.py` builds.

**Certifies:** BED-K is scan-blind by theorem, at the two readings the instrument
measures — no scalar-state recurrence delays; delay `d` needs linear state dimension
`≥ d` (bracketed against the shift register at `d + 1`); the power-law kernel is not
geometric on all of `(0, 1) ⊃` the registered box `α = H − ½ ∈ (0, ½)`. What is *not*
proved: the rate `w_k ~ k^{α−1}/Γ(α)` (header `:50-55`). **Instance:** the fitted
first-order recurrence reaches `R² = −0.000170` on the delay bed (`V15_N4_BEDK.md:194`,
**READ**), which the theorem now makes exact for every seed. **Domain census against
BED-K:** `first_order_cannot_delay` covers `d ≥ 1`; `d = 0` is a legal BED-K(a) cell
and is *necessarily* outside — **`CEQ.V16Domain.delay_zero_is_first_order`**
(`V16Domain.lean:339`, `IsDelay 0 (affineRec 0 1 y₀ x) x`) exhibits the recurrence that
realizes it. Against BED-M: empty (BED-M's label is a first-order chain by
construction). **A stale census line, flagged:** `V16_LEAN_DOMAIN.md:236` records "the
scan-blindness of the power-law bed is a separate claim with no theorem in this
round", but `first_order_cannot_powerlaw` was committed the same day (`git log`:
`1db8fc5 2026-08-31`, **RUN**, read-only) and `V15_JUPITER3_KERNEL.md:21` marks it
GREEN (**READ**). The paper cites `V15Kernel.lean:521` and not the V16 census row —
`MISTAKES.md` P-3 (`:316`, a stale claim never retracted) applies to that row.

## 2.5 (e) The phase / closed-magnitude theorems — `V15Phase`, `V16Domain`

With `C m θ i := ∑ k in range (i+1), (log (m k) + θ k · I)` and `Wp m θ i j := exp (C i − C j)`
(`V15Phase.lean:174, :179`) and `cap x := min 1 (max 0 x)` (`:114`):

- **`Wp_polar`** (`:202`). `(hm : ∀ k, 0 < m k) {i j} (hij : j ≤ i) : Wp m θ i j =
  (∏ k in Ico (j+1) (i+1), m k : ℝ) * exp ((∑ k in Ico (j+1) (i+1), θ k : ℝ) * I)`.
- **`phase_modulus_is_the_real_carrier`** (`:216`). `(m θ) (i j) : Complex.abs (Wp m θ i j)
  = CEQ.V15.W (fun k => Real.log (m k)) i j` — no hypothesis.
- **`prefix_phase_modulus`** (`:223`), **`phase_path_le_one`** (`:231`, `h0 : ∀ k, 0 < m k`,
  `h1 : ∀ k, m k ≤ 1`, `θ` universally quantified), **`phase_path_eq_one_iff_band`**
  (`:239`, an iff).
- **`softplus_gate_lt_one`** (`:132`), **`lru_modulus_lt_one`** (`:139`, `(ν θ : ℝ) :
  Complex.abs (exp ((−exp ν : ℝ) + θ·I)) < 1`), **`cap_band_attains`** (`:271`,
  `hx : ∀ k, 1 ≤ x k`), **`divergence_needs_an_open_magnitude`** (`:280`, modulus
  `285.07` at `m ≡ 285.07`, whatever `θ`).
- **`parity_is_Z2_winding`** (`:305`). `(p : ℕ → ZMod 2) {i j} (hij : j ≤ i) :
  exp ((∑ k in Ico (j+1) (i+1), phaseOf (p k) : ℝ) * I) = (chi (pscan p i − pscan p j) : ℂ)`.
  **`phase_sum_is_not_the_parity_phase`** (`:319`) and **`Z2_forgets_the_winding`**
  (`:346`): the correspondence holds only after `exp`; the parity mask is the integer
  winding reduced mod 2 and does not determine it.

**Certifies:** the carrier factors exactly into magnitude and phase with no cross
term; the magnitude is closed at `1` only under a hard cap (both prior
parametrizations, `−softplus` and LRU, are open at `1`); the `{0, π}` phase gate *is*
the record's sign character. **Instance:** `|a| ≤ 1` worst `1.0` exactly over
`2,200,000` draws, `0` exceedances; band modulus `9767/10⁴` exactly `1.0`, `233` one ulp
low, `0` above (`V15_ARM_PHASE.md:32-33`, **READ**; `tests/arm_phase/test_arm_phase.py:179`
asserts the `9767`, **READ**); LRU `1 − |λ| = 5.0759589e-435` at `ν = −1000` in 600-digit
arithmetic (`V15_X36_PRIOR_ART.md:219`, **READ**). **Domain census:** every magnitude
theorem in `V15Phase` carries `∀ k, 0 < m k` and is therefore decoration on BED-M's
`a = 0` draw (`V16_LEAN_DOMAIN.md:151-160`, **READ**); their closed-support replacements
are §2.3's `pathProd_*` clauses. **`sixteen_is_silent_on_the_zero_draw`**
(`V16Domain.lean:355`) records that `unit_phase_product` (`V15Phase.lean:92`, no
hypotheses) has total overlap with every corpus and constrains nothing on the zero
draw — the L-DOM census needs a second column (`V16_LEAN_DOMAIN.md:251-268`).

## 2.6 (f) The refutations proved in-file

| declaration | file:line | what it refutes | witness |
|---|---|---|---|
| `gate_zero_row_sum` | `V15.lean:221` | — | `∑_{j≤i} Wc g i j = i + 1` at `g ≡ 0` |
| `gate_zero_not_stochastic` | `V15.lean:237` | the R11 parity clause "`g ≡ 0` gives bitwise standard attention" (multiplicative reading) | `(hi : 1 ≤ i)`, row sum `≠ 1`; smallest `i = 1`, sum `2` |
| `lean_log_junk_makes_the_scan_form_silently_false` | `V16Domain.lean:147` | relaxing `0 < a` to `0 ≤ a` with the `log` form kept | `Wp 0 0 1 0 = 1 ∧ pathProd 0 0 1 0 = 0` — Lean's `Real.log 0 = 0` is junk, and the error is silent |
| `six_never_reaches_the_band` | `V15Phase.lean:260` | "on the band `m ≡ 1` #6 gives `= 1`" | `(hij : j < i)`: modulus `< 1` strictly for every `w`, every `θ` |
| `inverse_identity_is_vacuous` | `V15Source.lean:83` | Lean #15 read as `M · M⁻¹ = 1` | true of every unit of every ring; depends on no axioms; used nowhere |
| `unit_phase_does_not_bound_the_gate` | `V15Phase.lean:103` | #16 read as a bound | unit phase, modulus `285.07` — R1's measured `â_max` |
| `first_order_delays_constant_input` | `V15Kernel.lean:177` | the per-sequence reading of #12 | `α = 0, β = 1` delays a constant drive at every `d` |
| `delay_realizable_at_dimension_d` | `V15Kernel.lean:235` | the naive reading of #12 | a shift register delays at state dimension `d + 1` |
| `corners_are_distinct` | `V16Domain.lean:445` | a containment whose corners coincide | `1/2 ≠ 1` at `(1, 0)` |
| `zero_not_a_counterexample` | `OracleSeparation.lean:193` | dropping `hij` from `oracle_ne_resolvent` | the zero matrix is nilpotent |
| `expander_expands_l2` | `Contraction.lean:151` | "row-stochastic ⇒ `σ_max ≤ 1`" | `!![1,0;1,0]` |
| `one_not_nilpotent` | `Nilpotent.lean:105` | weakening `StrictlyLower` to lower | `1 ^ k ≠ 0` |
| `forward_map_fills_in` | `V15Source.lean:168` | "the inverse is sparse" without a denser comparand | `shift3` |
| `no_row_stochastic_with_drive_values` | `V15Fork.lean:195` | the two-branch escape and the unrescaled-value reading | `b ≡ 1`, label `> 1` |

Fourteen refusals, each with a witness in the file. The pattern is the record's answer
to `MISTAKES.md` V-3 (`:72`, an assertion that is an identity of one's own
construction) and V-10 (`:168`): a theorem ships with the statement it is *not*.

## 2.7 Not architecture-bearing, listed for completeness

`OrbitBound.orbit_error_bound` (`OrbitBound.lean:71`, `(hR : Function.Injective R) :
Fintype.card E - orbits f ≤ (errors R f).card`) and the `Refcount` chain
(`floor_add_orbits` `:87`, `evict_floor_add_refcount_pred` `:145`,
`free_face_floor_unchanged` `:160`, `shared_plaque_floor_drops` `:174`) formalize the
author's caustic Theorem 1 (DOI `10.5281/zenodo.21997746`, `OrbitBound.lean:5`, [U]) and
its identification with a KV-cache refcount. They are correct and machine-checked;
nothing in the shape below depends on them, and the paper does not spend them.

## 2.8 What these theorems license for the shape

The shape's operator is `z = (I − γP)^{-1} V`, `O = P z`, with `P` the `β = 1` corner of
`Hop` (a causal softmax row, **with a diagonal entry**) and absorbing rows on the
constraint sets. For each identity of the brief (§1, I1–I5) plus the block certificate:
which existing theorem covers it, which is a one-line corollary, and which is a new
Lean target — stated in Lean-like pseudocode, checked against the declarations above,
graded `[M]` (machine-checkable now from what is in the tree), `[S]` (needs work),
`[D]` (deferred), and named against the `MISTAKES.md` mechanism it is designed to
defeat. All five RUN numbers were reproduced this session from
`$SCRATCH/shape_identities.py`; I3 was re-run with BED-1's actual `A`/`B` sets because
the script's key guess (`basin_A`) is not a key of `bed_1.build` (**RUN**: script
reads `0.858…` on the wrong sets; corrected one-liner reads `0.0`).

**(1) `gamma_zero_is_softmax` — I1.** *Covered by:* `V16Domain.corner_softmax` (`:394`)
identifies `P` with `softmaxAttn` at `β = 1, g ≡ 0`; `gate_zero_logit_identity` gives
the logit-level parity. *One-line corollary:*
```
theorem gamma_zero_is_softmax (P V : Matrix (Fin n) (Fin n) ℝ) :
    P * (1 - (0 : ℝ) • P)⁻¹ * V = P * V        -- by simp [zero_smul, sub_zero, Matrix.inv_one]
```
Grade **[M]**. What Lean licenses: the *real* identity. The **bitwise** claim (`torch.equal
True` at `γ = 0` via `solve_triangular` against `P @ V`, **RUN**) is a float fact and
stays RUN-class; the record's reference point is that even the record's own softmax
corner is not bitwise against `ceq/lm.py` (`1.110223e-16`, `19/64`, §2.2). *Mechanism:*
V-3 — parity at `γ = 0` holds by construction, so it certifies nothing unless it ships
with its rejection region: at `γ = 0.5` the read departs from `P V` by max-abs
`2.3002850040264393` (**RUN**), and the paper's parity bind is the pair.

**(2) `pathprod_is_chain_resolvent` — I2.** *Covered by:* `Nilpotent.occupancy_is_exact_inverse`
(the resolvent is the finite occupancy sum) and `V15.chain_path_product` (the chain's
coefficients are the path products); `V16Domain.corner_path_product_is_the_gate_product`
identifies corner 3 with `∏ a_k` on `0 < a`. *Missing link (NEW):* the entry formula of
the occupancy sum for the sub-diagonal matrix.
```
def subdiag (a : ℕ → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => if (i : ℕ) = (j : ℕ) + 1 then a i else 0
theorem subdiag_strictlyLower (a) : StrictlyLower (subdiag a)                       -- [M]
theorem subdiag_pow_entry (a) (k : ℕ) (i j : Fin n) :
    (subdiag a ^ k) i j = if (i : ℕ) = (j : ℕ) + k then ∏ l in Ico ((j:ℕ)+1) ((i:ℕ)+1), a l else 0
theorem pathprod_is_chain_resolvent (a) (i j : Fin n) (hij : j ≤ i) :
    (CEQ.Occupancy.occupancy (subdiag a) n) i j = ∏ l in Ico ((j:ℕ)+1) ((i:ℕ)+1), a l
```
Grade **[S]** — the induction is `pow_entry_zero`'s with the surviving term tracked;
`V15Source.forward_map_fills_in` is its `n = 3` instance already in the tree. Together
with `chain_path_product` at `m = 0` this makes BED-M's `equilibrium_oracle` *literally*
the last row of corner 3's resolvent — measured `0.0` entrywise and
`6.217248937900877e-15` on the label (**RUN**). *Mechanism:* D-2 — the theorem is the
D-2 statement made exact: any arm containing corner 3 reproduces BED-M's label as its
own forward, so BED-M is *contained*, not won, and no capability number may be read
from it. The paper carries this as a proposition, not as a result.

**(3) `committor_is_resolvent_read` — I3 (`CEQ_V16_CONTRACT.md` #14, `[S]`; R15 `[D]`).**
*Covered by:* nothing at the boundary-value level. `Occupancy.occupancy_telescope` and
`OracleSeparation.truncation_never_exact` speak about `(1 − Q)` and its truncations;
`Contraction.rowStochastic_perron` gives the certificate for `γP`, not for a
sub-stochastic `Q` at `γ = 1`. *NEW, in three parts:*
```
-- (a) the read, given invertibility                                                    [M]
theorem committor_is_resolvent_read {t : ℕ} (Q : Matrix (Fin t) (Fin t) ℝ) (r q : Fin t → ℝ)
    (hharm : q = Q *ᵥ q + r) (hunit : IsUnit (1 - Q)) : q = (1 - Q)⁻¹ *ᵥ r
-- (b) invertibility from absorption: a Perron certificate at rate ρ < 1                [S]
theorem isUnit_one_sub_of_perron {Q w ρ} (hc : PerronCertificate Q w ρ) (hρ : ρ < 1) :
    IsUnit (1 - Q)
-- (c) BED-1's generator form is the same solve: I − Q = −L_TT / d                       [S]
theorem bed1_committor_eq (L : Matrix (Fin n) (Fin n) ℝ) (d : ℝ) (hd : 0 < d) (T B) … :
    (bed_1.committor L A B) = (1 - Q)⁻¹ *ᵥ (R *ᵥ 1_B)   with Q = 1 + L_TT / d, R = L_TB / d
```
(a) is `(1 − Q) q = r` multiplied through — one line once `hunit` is in hand. (b) is the
content and is not one line: a row-substochastic `Q` whose *max* row sum is `1` (only
rows adjacent to an absorber lose mass) has `‖Q‖_∞ = 1`, so the L∞ Neumann series does
not apply directly; the weighted certificate with `ρ = ρ(Q) < 1` is the route
(`Contraction.lean:24-37`). (c) is a definitional unpacking of `ceq/beds/bed_1.py:188-198`
(`rhs = −L[int, B].sum`, `solve(L[int,int], rhs)`, **READ**) against `P = K/d` with
`diag = 1 − rowsum(K)/d` (`:173-174`). *Instance:* `q = (I − Q)^{-1} R 1_B` against
`bed["q"]`: max-abs **`0.0`** with `A = [0]`, `B = [1]`, `n = 11`, `|T| = 9`, `T = 0.25`,
`jitter = 0.05`, `seed = 11`; `harmonic_residual = 1.0408340855860843e-17` (**RUN**;
the record's default reads `0.000000e+00`, `workdonenewseal.md:214`, **READ**);
Kirchhoff cross-check `< 1e-10` law with measured gaps `2.220446e-16` to `9.636736e-14`
(`MATHEMATICS.md:542-600`, **READ**). *Mechanism:* D-2 — the oracle runs on the
environment chain `P` built from `V` and `K` inside `bed_1.build`, which the arm never
sees; `oracle_ne_resolvent` proves the two operator classes are disjoint and the RUN
above instantiates its hypotheses (`Nonneg`, `SymmSupport`, `Q^{11} ≠ 0`). Also V-12
(`:189`, a single absorbing target makes the label constant): BED-1 has two absorbing
sets, and the shape's `K ≥ 2` constraint sets are the general case.

**(4) `neumann_truncation_bound` — I4.** *Covered by:* `Occupancy.occupancy_telescope`
gives the exact residual `(1 − γP)^{-1} − ∑_{k≤K} (γP)^k = (1 − γP)^{-1} (γP)^{K+1}`
(**DERIVED**: multiply the telescope at `N = K+1` by the inverse); `rowStochastic_perron`
plus `weighted_contraction` give `‖(γP)^{K+1} v‖_∞ ≤ γ^{K+1} ‖v‖_∞` by iterating the
one-step bound (**DERIVED**; `weighted_contraction_iterate` `:99` is the one-step form
restated, not a `k`-fold iterate — the iterate is an induction the tree does not yet
contain). *NEW:*
```
theorem resolvent_sup_bound (P) (hP : RowStochastic P) (γ) (h0 : 0 ≤ γ) (h1 : γ < 1)
    (u x : Fin n → ℝ) (hx : x = γ • (P *ᵥ x) + u) (M) (hu : ∀ j, |u j| ≤ M) :
    ∀ i, |x i| ≤ M / (1 - γ)                                                          -- [S]
theorem isUnit_one_sub_smul (P) (hP) (γ) (h0) (h1) : IsUnit (1 - γ • P)               -- [S]
theorem neumann_truncation_bound (P) (hP) (γ) (h0) (h1) (K : ℕ) (v M) (hv : ∀ j, |v j| ≤ M) :
    ∀ i, |((1 - γ • P)⁻¹ *ᵥ v - CEQ.Occupancy.occupancy (γ • P) (K + 1) *ᵥ v) i|
          ≤ γ ^ (K + 1) / (1 - γ) * M                                                 -- [S]
```
Route for the first: take `i` at the sup of `|x|` (finite `Fin n`), then
`|x_i| ≤ γ |x|_∞ + M` by `weighted_contraction` with `w = 1`, so `|x|_∞ ≤ M/(1−γ)`.
Injectivity of `1 − γP` follows from the bound at `u = 0`, and injective ⇒ unit on a
finite-dimensional space. Grade **[S]**, short. *What the number says:* at `γ = 0.7`,
`s = 64`, causal softmax `P`, the `‖·‖_∞` residual equals the bound to `1.3e-15` at
every `K ∈ {1, 2, 4, 8, 16}` (**RUN**) — the certificate is **attained**, not slack,
because for a non-negative row-stochastic `P` the residual's row sums are exactly
`γ^{K+1}/(1−γ)` (**DERIVED**: every entry of `(1−γP)^{-1}(γP)^{K+1}` is non-negative and
its row sum is the geometric tail). Consequently a pass on `err ≤ bound` is an
*identity* on this class (V-3) and the bind must be carried by its planted negative:
`P` scaled to row sums `1.5` reads `119.37486584159647` against `1.1433333333333329`
(**RUN**). *Mechanism:* V-24 (the planted non-stochastic `P`), V-10 (the pass is
attained by construction — stated as such), and L-CERT (every truncation depth `K`
ships its printed `δ = γ^{K+1}/(1−γ)`). This is the two-half proposition the brief
asks for against `IMPOSSIBLE.md` I1: (i) with a row-stochastic `P` the normalizer pays
for a denominator-free certificate with no sum over `s`; (ii) the sign capability the
signed operator bought by dropping the normalizer is not recovered — it is replaced by
absorbing rows, which `Asink_row_sum` already shows a stochastic row can carry
(the `j = 0` sink is an absorbing slot).

**(5) `resolvent_is_triangular_solve` — I5.** *Covered by:* nothing directly, and this
is the one place the record's proofs do **not** transfer. `pow_card_eq_zero` needs
`StrictlyLower` (diagonal excluded); the shape's `P` at `β = 1` is a causal softmax
row over `j ≤ i`, so `P_ii > 0` and `γP` is lower-triangular *with* a diagonal —
`one_not_nilpotent` (`Nilpotent.lean:105`) is the file's own warning. `(I − γP)^{-1}`
is then an *infinite* Neumann series (converging by (4)), not the `n`-term sum, and
the brief's sentence "the resolvent computes all `s` hops in one operator
(nilpotency)" holds only for the diagonal-free causal operator. Two honest options,
both to be stated in the paper: mask the diagonal (rows attend to `j < i`, with the
`j = 0` value-zero sink of `V15Fork.Asink` absorbing row 0) and inherit the finite
sum exactly; or keep the diagonal and carry the (4) certificate. *NEW:*
```
theorem lower_triangular_isUnit (M : Matrix (Fin n) (Fin n) ℝ)
    (hM : ∀ i j, (i:ℕ) < (j:ℕ) → M i j = 0) (hd : ∀ i, M i i ≠ 0) : IsUnit M     -- [M]
    -- det = ∏ diag (Mathlib `Matrix.det_of_lowerTriangular` [U]) and `isUnit_iff_isUnit_det` [U]
theorem resolvent_is_triangular_solve (M) (hM) (hd) (v : Fin n → ℝ) :
    M⁻¹ *ᵥ v = fwdSub M v                                                            -- [S]
    where fwdSub M v i = (v i - ∑ j < i, M i j * fwdSub M v j) / M i i   (well-founded on i)
theorem diag_one_sub_smul_pos (P) (hP : RowStochastic P) (γ) (h1 : γ < 1) (i) :
    0 < (1 - γ • P) i i                                                              -- [M]
```
Grade **[M]** for the unit and the diagonal, **[S]** for the substitution recursion
(a `Fin n` strong induction with `Matrix.mulVec` unfolded). *Instance:*
`solve_triangular` against the dense inverse, max-abs `1.7763568394002505e-15` at
`γ = 0.5`, `s = 64`, `d = 16` (**RUN**). The cost line "`s²d` multiply-adds against
`2s²d` for `QKᵀ`" is **DERIVED** from the loop count, not measured — the script
prints its own formula (`65536` vs `131072` at `s = 64, d = 16`) and the paper says so;
`COSTS.md` / `scale/m3_flops.py` are where a measured count would come from (not run
here). *Mechanism:* M-8 (`:544`, pricing every arm at one arm's rate) — the shape's
price is a forward substitution, and the paper prices it as one, not as an inverse.

**(6) `segmentation_blockdiag` — a zero gate splits the resolvent into blocks.**
*Covered by:* `V16Domain.pathProd_eq_zero_iff` (the path product is zero iff a zero
gate lies on the path) and `Nilpotent.pow_entry_zero` (the induction). *NEW:*
```
theorem segmentation_blockdiag {A : Matrix (Fin n) (Fin n) ℝ} (hA : StrictlyLower A) (c : ℕ)
    (hcut : ∀ i j : Fin n, (j:ℕ) < c → c ≤ (i:ℕ) → A i j = 0) :
    ∀ i j : Fin n, (j:ℕ) < c → c ≤ (i:ℕ) → (CEQ.Occupancy.occupancy A n) i j = 0     -- [M]
-- corollary for the chain: a single zero gate a_c = 0 zeroes the whole block below-left of c
theorem chain_zero_gate_cuts (a : ℕ → ℝ) (c : ℕ) (hc : a c = 0) :
    ∀ i j : Fin n, (j:ℕ) < c → c ≤ (i:ℕ) → (occupancy (subdiag a) n) i j = 0        -- [S] via (2)
-- the general-P form: with the cut block of γP zero, (1 − γP)⁻¹ is block-diagonal
theorem resolvent_fromBlocks (M₁₁ M₂₂) (h₁ : IsUnit M₁₁) (h₂ : IsUnit M₂₂) :
    (Matrix.fromBlocks M₁₁ 0 0 M₂₂)⁻¹ = Matrix.fromBlocks M₁₁⁻¹ 0 0 M₂₂⁻¹              -- [M] (Mathlib `inv_fromBlocks_zero₂₁_of_isUnit_iff` [U])
```
Proof of the first: `(A^{m+1}) i j = ∑_l (A^m) i l · A l j`; for `l < c` the first factor
is zero by the induction hypothesis (`i ≥ c > l`), for `l ≥ c` the second is zero by
`hcut` — the same case split as `pow_entry_zero`, so **[M]** now. *Instance:* BED-M's
generator zeroes the prefix (`a[:, :head+1] = 0.0`, `scale/negation_scope.py:428-429`,
**READ** via `V16_LEAN_DOMAIN.md:170`), so `96.78%` of causal pairs annihilate and the
annihilation MCC reads `1.000000` corpus-alone (`V16_ARM_SMPRIME.md:519-527`, **READ**);
the R15 F0 instance `8.9e-16, 595×` (`CEQ_V20_R15_CONTRACT.md:213`, **READ**) has no
located producer — gap. *Mechanism:* L-CERT (the F0 certificate's `δ` is exactly `0`,
printed, by `pathProd_eq_zero_iff`; F1's Cantelli/Azuma cutoffs are the shape's
soft-mask certificates and are *not* in Lean — `#23 [S]`); V-25 (the cut is a drawn
value on BED-M, 3 of 3, so the theorem has a subject); D-3 (`:727`, a difficulty dial
that does not vary — the block size is the dial, and the paper registers it as one).

**Summary of the ledger for the shape.** Covered outright: I1's real identity, I2's
two halves, the exact residual of I4, the annihilation of (6). One-line corollaries:
`gamma_zero_is_softmax` [M], `resolvent_fromBlocks` [M], `lower_triangular_isUnit` [M],
`segmentation_blockdiag` [M]. Genuinely new work: `pathprod_is_chain_resolvent` [S],
`committor_is_resolvent_read` (a) [M] / (b) [S] / (c) [S], `neumann_truncation_bound` [S],
`resolvent_is_triangular_solve` [S]. Nothing here is `[D]`. Every `[S]` item above is
sized at one file of the `V15Source.lean` scale, and each names the refusal it ships
with: (2) ships `forward_map_fills_in`'s pattern at general `n`; (3) ships
`truncation_never_exact` as the statement that the committor is never a finite sum;
(4) ships the attained-bound identity as a *declared* V-3 with its planted negative;
(5) ships `one_not_nilpotent` as the reason the finite sum is gone once the diagonal
is kept; (6) ships `no_prefix_scan_represents_a_zero_gate` as the reason the block
form cannot be reached by any prefix scan.
