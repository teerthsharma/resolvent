/-
  CEQ.V15Fork
  -----------
  Provenance: `V15_JUPITER2_FORK.md`, the node that settles the fork
  `V15_N3_LEAN.md`'s DIAGNOSIS opened. `CEQ/V15.lean` proved

    (L) `prefix_logit_computes_chain` — the MULTIPLICATIVE unnormalized hop
        `W_ij = exp(C_i − C_j)` on values `b` reproduces the chain label; and
    (P) `gate_zero_logit_identity`    — the ADDITIVE-logit hop gives parity with
        standard attention at `g ≡ 0`,

  and `gate_zero_not_stochastic` proved that no single operator can have both
  UNDER §S-M's reading, because §S-M's row sums to `i + 1` and a softmax row
  sums to `1`. `V15_N3_LEAN.md` concluded "No single operator has both".

  That conclusion is TOO STRONG, and this file is the correction. It is stated
  as three theorems about one weight matrix `Asink`.

  | name                                | what it says |
  |-------------------------------------|--------------|
  | `Asink_row_sum`                     | `∑_{j ≤ i} Asink a i j = 1` — the row is stochastic, with NO hypothesis on `a` at all |
  | `Asink_nonneg`                      | `0 < a k < 1` ⇒ every entry is `≥ 0`; with the row sum, the row is a probability vector |
  | `Asink_computes_chain`              | on values `Vsink = b j / (1 − a j)` it reproduces the chain label EXACTLY |
  | `no_row_stochastic_with_drive_values` | and yet, with the DRIVES `b` themselves as values, no row-stochastic operator reproduces it, at any index `≥ 2` |

  Read together: the obstruction `gate_zero_not_stochastic` identified is real,
  but it is an obstruction to reproducing the label WITH THE DRIVES AS VALUES,
  not to reproducing it at all. `Asink_eq_hop` names the whole repair — one
  factor `(1 − a j)` on §S-M's own hop, and one value-zero slot at `j = 0`:

      Asink a i j = (1 − a j) · W (log a) i j    (j ≠ 0)
      Asink a i 0 =             W (log a) i 0

  and `Asink_row_sum` is the statement that this reweighting telescopes. So the
  softmax normalizer is not the obstruction to forming path products; the fixed
  value scale was. `V15_JUPITER2_FORK.md` §5 prices what the normalizer does
  charge, which is value dynamic range `1/(1 − a)` and nothing more.

  `gate_zero_sink_logit_identity` carries the (P) half for the SAME operator, so
  both binds are now theorems about one object rather than about two.

  `CEQ/V15.lean` is not edited by this file; it is imported.

  No `sorry`.
-/

import CEQ.V15

namespace CEQ.V15Fork

open BigOperators Finset CEQ.V15

/-! ## The construction -/

/-- The values: a value-zero slot at `j = 0` (the BOS sink, standing for the
    chain's initial condition `y₀ = 0`), and the drives rescaled by the
    position-local factor `1 / (1 − a j)` everywhere else.

    The rescale is what `V15.lean`'s reading forbade itself. It is a legitimate
    value map for the arm because `a j = a(x_j)` is position-local, so
    `b j / (1 − a j)` is too. -/
noncomputable def Vsink (a b : ℕ → ℝ) (j : ℕ) : ℝ := if j = 0 then 0 else b j / (1 - a j)

/-- The weights: §S-M's own hop `∏_{k=j+1}^{i} a k` reweighted by `(1 − a j)`,
    with the `j = 0` slot left unweighted. `Asink_eq_hop` below states exactly
    that against `V15.W`. -/
noncomputable def Asink (a : ℕ → ℝ) (i j : ℕ) : ℝ :=
  if j = 0 then ∏ k in Ico 1 (i + 1), a k
  else if j ≤ i then (1 - a j) * ∏ k in Ico (j + 1) (i + 1), a k
  else 0

/-! ## The row is stochastic -/

/-- **The telescoping.** `(1 − a j)` times the tail product from `j` is the
    difference of two consecutive tail products, so the row sums to the empty
    product, `1`.

    Note what is NOT assumed: no positivity, no bound, no hypothesis on `a`
    whatsoever. The row sum is `1` as an algebraic identity. Positivity of the
    entries is a separate statement (`Asink_nonneg`) and needs `a k < 1`. -/
theorem Asink_row_sum (a : ℕ → ℝ) (i : ℕ) :
    ∑ j in range (i + 1), Asink a i j = 1 := by
  -- peel the `j = 0` slot off the front
  rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (Nat.succ_pos i)]
  -- the remaining summand is the reweighted tail product
  have hbody : ∀ j ∈ Ico 1 (i + 1),
      Asink a i j = (1 - a j) * ∏ k in Ico (j + 1) (i + 1), a k := by
    intro j hj
    obtain ⟨hj1, hj2⟩ := Finset.mem_Ico.mp hj
    have hj0 : j ≠ 0 := Nat.one_le_iff_ne_zero.mp hj1
    simp [Asink, hj0, Nat.lt_succ_iff.mp hj2]
  rw [Finset.sum_congr rfl hbody, Finset.sum_Ico_eq_sum_range]
  -- reindex to `range i` and telescope with `f t = ∏_{k ∈ Ico (t+1) (i+1)} a k`
  set f : ℕ → ℝ := fun t => ∏ k in Ico (t + 1) (i + 1), a k with hf
  have hstep : ∀ t ∈ range i,
      (1 - a (1 + t)) * (∏ k in Ico (1 + t + 1) (i + 1), a k) = f (t + 1) - f t := by
    intro t ht
    have hti : t < i := Finset.mem_range.mp ht
    have hpeel : f t = a (t + 1) * f (t + 1) := by
      simpa [hf] using
        Finset.prod_eq_prod_Ico_succ_bot (Nat.succ_lt_succ hti) (fun k => a k)
    have h1t : 1 + t = t + 1 := Nat.add_comm 1 t
    rw [h1t, hpeel, hf]
    ring
  rw [Nat.add_sub_cancel, Finset.sum_congr rfl hstep, Finset.sum_range_sub f i]
  have hfi : f i = 1 := by simp [hf]
  have hf0 : f 0 = ∏ k in Ico 1 (i + 1), a k := by simp [hf]
  have h0 : Asink a i 0 = ∏ k in Ico 1 (i + 1), a k := by simp [Asink]
  rw [h0, hfi, hf0]
  ring

/-- With `a k ∈ (0, 1)` — exactly the range `V15.bounded_gates_stable` puts
    `exp (−softplus w)` in — every entry is non-negative. With
    `Asink_row_sum` this makes each row a probability vector, so `Asink` is a
    causal row-stochastic operator and is realizable as a softmax row. -/
theorem Asink_nonneg (a : ℕ → ℝ) (h0 : ∀ k, 0 < a k) (h1 : ∀ k, a k < 1) (i j : ℕ) :
    0 ≤ Asink a i j := by
  unfold Asink
  split
  · exact Finset.prod_nonneg fun k _ => (h0 k).le
  · split
    · exact mul_nonneg (by linarith [h1 j]) (Finset.prod_nonneg fun k _ => (h0 k).le)
    · exact le_refl 0

/-! ## The row reproduces the label -/

/-- `V15.chain_path_product` at `m = 0`, in the shape the read-out needs. -/
theorem chain_eq_sum (a b : ℕ → ℝ) (i : ℕ) :
    chain a b 0 i = ∑ s in Ico 1 (i + 1), (∏ k in Ico (s + 1) (i + 1), a k) * b s := by
  have h := chain_path_product a b 0 0 i (Nat.zero_le i)
  have hc : chain a b 0 0 = 0 := rfl
  rw [h, hc, mul_zero, zero_add]

/-- **The construction, proved.** A causal row-stochastic operator reproducing
    the chain's path-product label exactly.

    This refutes the impossibility `V15_N3_LEAN.md` conjectured. The hypothesis
    that does the work is `a k ≠ 1`: it is what lets the value be rescaled, and
    it is the boundary the `g ≡ 0` parity setting sits exactly on. -/
theorem Asink_computes_chain (a b : ℕ → ℝ) (ha : ∀ k, a k ≠ 1) (i : ℕ) :
    ∑ j in range (i + 1), Asink a i j * Vsink a b j = chain a b 0 i := by
  rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (Nat.succ_pos i)]
  have hV0 : Vsink a b 0 = 0 := by simp [Vsink]
  rw [hV0, mul_zero, zero_add]
  rw [chain_eq_sum a b i]
  refine Finset.sum_congr rfl (fun j hj => ?_)
  obtain ⟨hj1, hj2⟩ := Finset.mem_Ico.mp hj
  have hj0 : j ≠ 0 := Nat.one_le_iff_ne_zero.mp hj1
  have hne : (1 : ℝ) - a j ≠ 0 := sub_ne_zero_of_ne (Ne.symm (ha j))
  simp only [Asink, Vsink, hj0, if_false, Nat.lt_succ_iff.mp hj2, if_true, if_neg hj0]
  field_simp
  ring

/-! ## …and with the DRIVES as values, nothing can -/

theorem chain_one_nonneg (a : ℕ → ℝ) (ha : ∀ k, 0 < a k) (n : ℕ) :
    0 ≤ chain a (fun _ => 1) 0 n := by
  induction n with
  | zero => simp [chain]
  | succ n ih =>
    have hstep : chain a (fun _ => 1) 0 (n + 1)
        = a (n + 1) * chain a (fun _ => 1) 0 n + 1 := rfl
    rw [hstep]
    nlinarith [ha (n + 1)]

/-- With the constant drive `b ≡ 1` the label at index `≥ 2` exceeds `1`. This
    is the numeric core of the obstruction: the label leaves the convex hull of
    its own drives, so no convex combination of them can reach it. -/
theorem chain_one_gt_one (a : ℕ → ℝ) (ha : ∀ k, 0 < a k) (n : ℕ) :
    1 < chain a (fun _ => 1) 0 (n + 2) := by
  have h1 : chain a (fun _ => 1) 0 (n + 2)
      = a (n + 2) * chain a (fun _ => 1) 0 (n + 1) + 1 := rfl
  have h2 : chain a (fun _ => 1) 0 (n + 1)
      = a (n + 1) * chain a (fun _ => 1) 0 n + 1 := rfl
  have h3 := chain_one_nonneg a ha n
  have hpos : 0 < a (n + 1) * chain a (fun _ => 1) 0 n + 1 := by
    nlinarith [ha (n + 1)]
  have := mul_pos (ha (n + 2)) hpos
  rw [h1, h2]
  linarith

/-- **The theorem that does survive.** With the drives `b` themselves as the
    values — which is §S-M's reading, `"one unnormalized causal hop on values
    V(x)"` reproducing `∑_j W_ij b_j` — NO causal row-stochastic operator
    reproduces the chain label, at any index `≥ 2`, for any positive gates.

    `A` is completely arbitrary here: it may read `a`, it may read `b`, it may
    be any function whatever. Only two things are assumed of it, and they are
    the two that define the class. The witness is the constant drive, and the
    mechanism is that the label leaves the convex hull of the values.

    So the incompatibility `V15_N3_LEAN.md` reported IS a theorem — under this
    hypothesis, and only under it. `Asink_computes_chain` above shows the
    hypothesis is the entire content. -/
theorem no_row_stochastic_with_drive_values
    (a : ℕ → ℝ) (ha : ∀ k, 0 < a k) (A : ℕ → ℝ) (n : ℕ)
    (hstoch : ∑ j in Ico 1 (n + 2 + 1), A j = 1)
    (hexact : ∀ b : ℕ → ℝ, ∑ j in Ico 1 (n + 2 + 1), A j * b j = chain a b 0 (n + 2)) :
    False := by
  have h := hexact (fun _ => 1)
  simp only [mul_one] at h
  rw [hstoch] at h
  exact absurd h (ne_of_lt (chain_one_gt_one a ha n))

/-! ## The parity half, for the same operator -/

/-- **(P) for the construction's logit.** The three-term logit
    `q i j + (C i − C j) + s j` collapses to `q i j` at `g ≡ 0, s ≡ 0`, so the
    softmax row and everything downstream of it are bit-identical to standard
    attention.

    This is `V15.gate_zero_logit_identity` extended by the key bias `s` that
    `Asink` needs, and it is stated for the SAME operator that
    `Asink_computes_chain` is about. That is the difference from the two-branch
    escape `V15_JUPITER2_FORK.md` §1 rules vacuous: there, (L) and (P) are
    binds on two summands; here they are two settings of the same coordinates
    `(g, s)` of one softmax row. -/
theorem gate_zero_sink_logit_identity (g s : ℕ → ℝ) (hg : ∀ k, g k = 0)
    (hs : ∀ k, s k = 0) (q : ℕ → ℕ → ℝ) (i j : ℕ) :
    q i j + (scan g i - scan g j) + s j = q i j := by
  have h : ∀ n, scan g n = 0 := fun n => by simp [scan, hg]
  simp [h, hs]

/-- The key-only form. `V15_JUPITER2_FORK.md` §4 measures that the query-side
    scan term is redundant under the softmax — the target factor comes back out
    of the running normalizer, not out of the logit — so the whole construction
    runs with a KEY bias alone. Parity is unaffected. -/
theorem gate_zero_key_logit_identity (g s : ℕ → ℝ) (hg : ∀ k, g k = 0)
    (hs : ∀ k, s k = 0) (q : ℕ → ℕ → ℝ) (i j : ℕ) :
    q i j + (-scan g j + s j) = q i j := by
  have h : ∀ n, scan g n = 0 := fun n => by simp [scan, hg]
  simp [h, hs]

/-! ## The repair, named against §S-M's own hop -/

/-- The `j = 0` slot is §S-M's hop, unweighted. -/
theorem Asink_zero_eq_hop (a : ℕ → ℝ) (ha : ∀ k, 0 < a k) (i : ℕ) :
    Asink a i 0 = W (fun k => Real.log (a k)) i 0 := by
  rw [prefix_logit_mask a ha (Nat.zero_le i)]
  simp [Asink]

/-- Every other slot is §S-M's hop times exactly one factor `(1 − a j)`. The
    whole difference between an operator that fails both binds and one that
    passes both is this factor and the value-zero slot above. -/
theorem Asink_eq_hop (a : ℕ → ℝ) (ha : ∀ k, 0 < a k) {i j : ℕ} (hj : j ≠ 0)
    (hij : j ≤ i) :
    Asink a i j = (1 - a j) * W (fun k => Real.log (a k)) i j := by
  rw [prefix_logit_mask a ha hij]
  simp [Asink, hj, hij]

end CEQ.V15Fork
