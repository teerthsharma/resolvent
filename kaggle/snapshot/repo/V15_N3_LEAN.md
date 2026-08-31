# V15 n3 — the Lean train-gate

Node `n3` of the CEQ v15 composition round (`CEQ_V15_CONTRACT.md` PART V). Items
#1, #2, #5, #6, #7 are the train-gate; #3 rides along with the signed variant.
All statements live in `lean/CEQ/V15.lean`, imported from `lean/CEQ.lean`.
Toolchain `leanprover/lean4:v4.7.0`, mathlib vendored at the matching tag.

---

## STATUS TABLE

| # | statement | status | Lean name |
|---|---|---|---|
| 1 | `chain_path_product` | **GREEN** | `CEQ.V15.chain_path_product` |
| 2 | `prefix_logit_mask` | **GREEN** | `CEQ.V15.prefix_logit_mask` |
| 3 | `parity_sign` | **GREEN** | `CEQ.V15.parity_sign` |
| 5 | `gate_zero_is_attention` | **GREEN on the mask half, FALSE on the parity clause** | `CEQ.V15.gate_zero_is_attention` / `CEQ.V15.gate_zero_not_stochastic` |
| 6 | `bounded_gates_stable` | **GREEN** | `CEQ.V15.bounded_gates_stable` (+ `_prod_le_one`, `_antitone`) |
| 7 | `scan_assoc` | **GREEN** | `CEQ.V15.scan_assoc` |

Train-gate verdict: **five of five green as algebra; #5's PARITY clause is refuted
as written and holds only under the additive-logit reading of the gate.** No
`sorry` anywhere. See DIAGNOSIS below.

Supporting theorems proved in the same file, all green:
`chain_impulse_path_product`, `prefix_logit_computes_chain`, `chi_add`, `chi_sum`,
`pscan_sub_eq_add`, `gate_zero_row_sum`, `gate_zero_not_stochastic`,
`gate_zero_logit_identity`, `softplus_pos`, `bounded_gates_prod_le_one`,
`bounded_gates_antitone`, `affApply_affComp`, `chain_step_eq_affApply`,
`affComp_id`, `id_affComp`, `affComp_gate_only`.

---

## BUILD OUTPUT

```
$ cd lean && lake build; echo "=== EXIT CODE: $? ==="
[1514/1526] Building CEQ.V15
[1525/1526] Building CEQ
=== EXIT CODE: 0 ===
```

No errors, no warnings, no `sorry` warning.

```
$ grep -n sorry lean/CEQ/V15.lean
40:  No `sorry`.
```

The single hit is the sentence `No \`sorry\`.` in the module doc comment. There is
no `sorry` tactic in the file.

Compiling with exit 0 is necessary but not sufficient — a `sorry` inside a
`macro`, or a theorem closed by an axiom, would still build. Every theorem was
therefore checked against the kernel's axiom list:

```
$ lake env lean AxCheck.lean
'CEQ.V15.chain_path_product' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.chain_impulse_path_product' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.prefix_logit_mask' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.prefix_logit_computes_chain' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.parity_sign' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.gate_zero_is_attention' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.gate_zero_row_sum' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.gate_zero_not_stochastic' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.gate_zero_logit_identity' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.bounded_gates_stable' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.bounded_gates_prod_le_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.bounded_gates_antitone' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.scan_assoc' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.affApply_affComp' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.chain_step_eq_affApply' depends on axioms: [propext, Classical.choice, Quot.sound]
```

`propext`, `Classical.choice`, `Quot.sound` are Lean's three standard axioms.
`sorryAx` appears nowhere.

---

## THE STATEMENTS AS THE KERNEL ELABORATED THEM

Pasted from `#check`, not retyped, so no weakening can hide in the transcription.

```
chain_path_product : ∀ (a b : ℕ → ℝ) (y₀ : ℝ) (m n : ℕ),
  m ≤ n →
    chain a b y₀ n =
      (Finset.prod (Finset.Ico (m + 1) (n + 1)) fun k => a k) * chain a b y₀ m +
        Finset.sum (Finset.Ico (m + 1) (n + 1)) fun s =>
          (Finset.prod (Finset.Ico (s + 1) (n + 1)) fun k => a k) * b s

prefix_logit_mask : ∀ (a : ℕ → ℝ),
  (∀ (k : ℕ), 0 < a k) →
    ∀ {i j : ℕ}, j ≤ i →
      W (fun k => Real.log (a k)) i j = Finset.prod (Finset.Ico (j + 1) (i + 1)) fun k => a k

parity_sign : ∀ (p : ℕ → ZMod 2) {i j : ℕ},
  j ≤ i → chi (pscan p i - pscan p j) = Finset.prod (Finset.Ico (j + 1) (i + 1)) fun k => chi (p k)

gate_zero_is_attention : ∀ (g : ℕ → ℝ),
  (∀ (k : ℕ), g k = 0) → ∀ (i j : ℕ), Wc g i j = if j ≤ i then 1 else 0

gate_zero_row_sum : ∀ (g : ℕ → ℝ),
  (∀ (k : ℕ), g k = 0) → ∀ (i : ℕ), (Finset.sum (Finset.range (i + 1)) fun j => Wc g i j) = ↑i + 1

gate_zero_not_stochastic : ∀ (g : ℕ → ℝ),
  (∀ (k : ℕ), g k = 0) → ∀ {i : ℕ}, 1 ≤ i →
    (Finset.sum (Finset.range (i + 1)) fun j => Wc g i j) ≠ 1

bounded_gates_stable : ∀ (w : ℝ), 0 < Real.exp (-softplus w) ∧ Real.exp (-softplus w) ≤ 1

bounded_gates_antitone : ∀ (w : ℕ → ℝ) {j' j i : ℕ},
  j' ≤ j → j ≤ i →
    (Finset.prod (Finset.Ico j' i) fun k => Real.exp (-softplus (w k))) ≤
      Finset.prod (Finset.Ico j i) fun k => Real.exp (-softplus (w k))

scan_assoc : ∀ (p q r : Aff), affComp (affComp p q) r = affComp p (affComp q r)
```

---

## PER-ITEM NOTES

### #1 `chain_path_product` — GREEN

Stated in ABSOLUTE indices rather than lag offsets, deliberately: the coefficient
of the drive at source `s` in the state at target `n` is `∏_{k=s+1}^{n} a_k`, which
is literally the index range `Ico (j+1) (i+1)` that #2's prefix-logit hop produces.
The two theorems can therefore be composed without a reindexing lemma standing
between them, and they are, in `prefix_logit_computes_chain`:

```
chain a b 0 i = ∑ s in Ico 1 (i + 1), W (fun k => Real.log (a k)) i s * b s
```

That is the contract's oracle-gate bind (`[RUN] reproduces the chain label to
4.0e-15`) at exact arithmetic rather than at fp64. It is the strongest single
statement in the file.

`chain_impulse_path_product` is the same fact in the contract's own wording: a lone
drive at `m+1` reaching `n` is multiplied by exactly the intervening gate product
`∏_{k=m+2}^{n} a_k`, and by nothing else. Proof by `Nat.le_induction` on the target,
with `Finset.prod_Ico_succ_top` / `Finset.sum_Ico_succ_top` doing the peel.

The contract's claim that this is a PATH PRODUCT and not a sum of weights survives
formalization unchanged. The sum in the statement ranges over SOURCES; the
composition along each path is multiplicative. This is the one surviving algebraic
fact and it is now a theorem.

### #2 `prefix_logit_mask` — GREEN

`a k > 0` is stated as an explicit hypothesis, as instructed, and it is
load-bearing rather than decorative: `Real.exp` is positive everywhere, so for any
gate sequence with a negative entry the left side stays positive while the right
side flips sign, and the identity is false. This is precisely why the signed
variant needs the separate parity channel of #3 — the magnitude channel cannot
carry a sign.

### #3 `parity_sign` — GREEN

Formalized in `ZMod 2` (the contract permits it). `pscan` is the prefix sum; in
`ZMod 2`, `+` IS exclusive-or, and `pscan_sub_eq_add` records that the prefix-sum
DIFFERENCE that #2 takes and the prefix XOR that §S-M's parity mask takes are the
same element (`CharTwo.sub_eq_add`). The sign character `chi x = if x = 0 then 1
else -1` is proved multiplicative on `+` (`chi_add`) and then lifted to arbitrary
`Finset` sums (`chi_sum`).

The result is that the signed carrier is the PAIR `(W, chi)` over the same index
range `Ico (j+1) (i+1)` — the same path, the same endpoints, the same
multiplicative composition. Sign and magnitude are one mechanism, not two.

### #5 `gate_zero_is_attention` — GREEN on the mask, **FALSE on the parity clause**

See DIAGNOSIS.

### #6 `bounded_gates_stable` — GREEN, and slightly stronger than asked

`softplus_pos` gives `0 < log (1 + exp w)` from `exp w > 0`, so `g = -softplus w`
is strictly negative and `a = exp g ∈ (0, 1)` — strict on the upper end, stronger
than the contract's `(0, 1]`. The `≤ 1` form is what is exported, because that is
what the two consequences need.

Both consequences are proved, not just the membership:

- `bounded_gates_prod_le_one` — every path product over any `Finset` is `≤ 1`.
- `bounded_gates_antitone` — path products are non-increasing in path length:
  `j' ≤ j ≤ i ⇒ ∏_{Ico j' i} ≤ ∏_{Ico j i}`. Lengthening a path cannot amplify.
  Proved by splitting at `j` with `Finset.prod_Ico_consecutive` and
  `mul_le_of_le_one_left`.

The stationarity claim of §S-M holds as stated.

### #7 `scan_assoc` — GREEN, and the trivial version is refused

Read literally, §S-M's scan is a prefix SUM over `(ℝ, +)` and `scan_assoc` would be
`add_assoc`. That version compiles in one token and licenses nothing; it is exactly
the "weakened into triviality" outcome the node was told is worse than a named
failure. It is not what is stated here.

What is stated is associativity of the AFFINE scan element, `(a, b)` denoting
`x ↦ a·x + b`, with `affComp p q = (q.1 * p.1, q.1 * p.2 + q.2)`. Three theorems
keep it honest:

- `affApply_affComp` — `affComp` really is composition of the denoted maps, so
  `scan_assoc` is not associativity of an arbitrary binary operation.
- `chain_step_eq_affApply` — the maps being composed are the recurrence's own
  steps, by `rfl`.
- `affComp_id` / `id_affComp` — `(1, 0)` is a two-sided unit, so it is a monoid and
  a balanced-tree reassociation is well defined.

`affComp_gate_only` records that the `(ℝ, +)` prefix sum of §S-M is the `b = 0`
submonoid read through `log`, which is why `add_assoc` was never the content of #7.

---

## DIAGNOSIS — #5 AND THE PARITY CLAUSE

### What is true

`gate_zero_is_attention` is green and is stated for ALL `i, j`, not only below the
diagonal:

```
Wc g i j = if j ≤ i then 1 else 0
```

`g ≡ 0` makes the masked hop the causal all-ones mask exactly. This is standard
attention's MASK, and the contract's §S-M sentence "`g == 0` gives the causal
all-ones mask" is correct.

Note what the proof shows about where the causality comes from: with `g ≡ 0`,
`W g i j = exp(0 - 0) = 1` ABOVE the diagonal as well. The causal structure is
supplied by `Wc`'s mask, not derived from the gate. Nothing in the prefix-logit
construction is causal by itself.

### What is false

The PARITY WITH SELF-ATTENTION clause of PART I says, and the round's headline
rests on:

> **PARITY WITH SELF-ATTENTION is by IDENTITY BIND, not TOST.** `g == 0` gives
> bitwise standard attention (Lean #5).

**This is false for the operator §S-M actually specifies**, and the refutation is
in the file as a theorem rather than as an argument:

```
gate_zero_row_sum : ∑ j in range (i + 1), Wc g i j = ↑i + 1
gate_zero_not_stochastic : 1 ≤ i → ∑ j in range (i + 1), Wc g i j ≠ 1
```

§S-M specifies "ONE **unnormalized** causal hop `W_ij = exp(C_i - C_j)` on values
`V(x)`" — the word is the contract's own. Every row of every softmax attention
matrix sums to `1`, for every query, every key, and every weight matrix. The
`g ≡ 0` row sums to `i + 1`. Smallest witness: at `i = 1` the row is `(1, 1)` and
sums to `2`. So the `g ≡ 0` operator is not any softmax row of any attention head,
and its output at position `i` is the unnormalized causal cumulative sum
`∑_{j ≤ i} V(x_j)`, not an attention read-out.

Row-normalizing does not rescue it either, and this is the sharper point: the
normalized row is uniform, `1/(i+1)` at every `j ≤ i`. Uniform attention equals
standard attention only when the QK logits happen to be constant across `j`, which
is the degenerate head, not the general one. So no normalization convention turns
`g ≡ 0` into "bitwise standard attention" for §S-M's operator, because §S-M's
operator has no QK term to recover.

### Is this engineering or mathematics?

**Mathematics.** It is not a proof-engineering obstruction — the refuting statement
is three lines and green. The contract asserts an identity between two objects that
differ by a factor of `i + 1` per row, because §S-M's hop replaces the attention
logits rather than adding to them.

### The repair, also proved

The parity bind survives under the ADDITIVE-LOGIT reading: put `C_i - C_j` into the
logits ON TOP of whatever logits `q` the head already computes, i.e.
`softmax_j(q i j + (C_i - C_j))` rather than `exp(C_i - C_j)` alone.

```
gate_zero_logit_identity : q i j + (scan g i - scan g j) = q i j
```

At `g ≡ 0` the logit is `q i j` unchanged, so the softmax row and everything
downstream of it are bit-identical to standard attention. That is a genuine
identity bind and it does deliver parity with no TOST and no seeds.

### What this costs the round

The two readings are not interchangeable, and the round's other claims are attached
to the multiplicative one:

- `prefix_logit_computes_chain` — the arm reproduces the Markov chain label — holds
  for the MULTIPLICATIVE hop, because the chain's path coefficient is a product of
  gates with no additive logit term in it. The additive-logit form does not
  reproduce the chain label; it multiplies each path product by a softmax weight
  and renormalizes.
- `gate_zero_logit_identity` — parity with self-attention — holds for the ADDITIVE
  hop.

**No single operator has both** under the contract as written. §S-M currently
claims both from one construction. Either the arm carries two paths (a gated value
path for the chain, an additive logit bias for parity) and the parity claim is
about the second only, or the parity claim retires and the round is back to
resolution statements, which is the state the contract's own TOST clause already
concedes at `N = 8`.

This is the item to put in front of MARS and SATURN before ARM PL is built, because
it changes what the arm is, not just how it is measured. `[M]` item #5 is the
failing item, and per PART V it.5 that is the point of the gate.

---

## FILES TOUCHED

- `lean/CEQ/V15.lean` — new, 328 lines, 23 top-level `theorem`/`lemma` declarations plus 9 definitions.
- `lean/CEQ.lean` — one added line, `import CEQ.V15`.
- `V15_N3_LEAN.md` — this file.

No other Lean file was read for edit or modified. No git write command was run.
