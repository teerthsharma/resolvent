# V15 JUPITER-2 — the fork the train-gate opened

Node `JUPITER-2` of the CEQ v15 composition round. Settles the question
`V15_N3_LEAN.md`'s DIAGNOSIS left open: whether ONE operator family carries both
of the contract's ARM PL claims.

- **(L)** with oracle gates the arm reproduces the chain path-product label
  exactly. `CEQ.V15.prefix_logit_computes_chain`, green, for the MULTIPLICATIVE
  unnormalized hop `W_ij = exp(C_i − C_j)`.
- **(P)** `g == 0` gives bitwise standard attention. `CEQ.V15.gate_zero_not_stochastic`
  refutes it for that hop; `CEQ.V15.gate_zero_logit_identity` proves it for the
  ADDITIVE-logit hop.

`V15_N3_LEAN.md` concluded "**No single operator has both** under the contract as
written", and recommended either two paths in the arm or retiring the parity
claim.

**That conclusion is too strong. One operator has both, and this node builds it.**
It also rules the escape it was told to adjudicate VACUOUS, states the rule that
separates the escape from the construction, and refutes the impossibility theorem
the node was invited to prove.

**Filed under L-AMEND** (`CEQ_V15_1_DELTA.md`:14). The brief was widened
mid-flight from adjudication to repair: *"A refuted clause may be REPAIRED by
amending its formula, not only recorded as a loss, provided the amendment moves
toward the north star and the amendment is stated as an amendment — with the
original clause, the refutation, the new formula, and what the new formula gives
up."* §8 is that statement in those four parts, including the cost. §7
adjudicates the candidate route the widened brief named — let the readout divide
the normalizer back out — and rejects it under L-AMEND's own proviso that *"an
amendment that restores a clause by making its bind vacuous is a worse outcome
than the refutation it replaces."*

Nothing was trained (L-LEAN). Every number below is a float64 identity residual
or a bitwise comparison, from `scripts/v15_jupiter2_fork_probe.py`, pasted whole
in APPENDIX A. Eleven new Lean declarations and their axiom lists are in
APPENDIX B.

---

## VERDICT TABLE

| asked | verdict |
|---|---|
| (a) the two-branch bind | **VACUOUS.** Rejection region empty. Rule stated in §1 as class `V-23`. |
| (b) can additive-logit reproduce the label | **NO**, and the failure is exact rather than approximate: it computes `y_i / R_i`, the label divided by the multiplicative hop's own row sum. Confirmed to `4.44e-16`. |
| (c) is the incompatibility a theorem | **YES with the drives as values, and it is FALSE without that hypothesis.** The hypothesis is the entire content. Both halves are in Lean. The campaign's hoped-for headline — "softmax's normalizer is the obstruction to path products" — is refuted, by construction. |
| (d) what ARM PL should be | One causal softmax head whose KEY logit carries `−C_j + log(1 − a_j)` and whose values carry `b_j / (1 − a_j)`, with a value-zero BOS sink; the (P) bind at `(g, s) = (0, 0)` and the (L) bind at the oracle setting are then two settings of the same two coordinates of the same softmax row. |
| (e) the widened brief's own candidate — a per-row readout scalar `ρ_i` | **REJECTED, by a dichotomy with no third branch.** Tie `ρ` to the gate and (P) fails, because `g ≡ 0` forces `ρ_i = Z_i = i + 1`. Leave `ρ` free with an off switch and (P) holds and is vacuous — passed bitwise by the answer key. §7. |

Score against the round: `[M]` item #5's PARITY clause does **not** have to retire.
§S-M's OPERATOR does. The amendment and its seven named costs are §8.

---

## 1 — THE RULING ON THE TWO-BRANCH BIND

### The thing being adjudicated

```
out = softmax_causal(q) @ V1  +  lambda * Wc(g) @ V2
```

(P) at `lambda = 0`; (L) at `V1 = 0, lambda = 1`. Both binds appear to pass.

### Ruling: VACUOUS

The `lambda = 0` bind was run against three substitutions for the second branch.
Real output:

```
X = Wc(g) @ V2, the honest gate      (P) bitwise: True   (L) max|out - y|: 1.1102230246251565e-16
X = i.i.d. Gaussian noise            (P) bitwise: True   (L) max|out - y|: 4.3977935722055612
X = the LABEL y itself (answer key)  (P) bitwise: True   (L) max|out - y|: 0
```

The third row is the ruling. An architecture that hardcodes the answer passes the
parity bind bitwise, and passes the label bind to `0` exactly. A gate that the
answer key passes is not a gate.

The proof of the `lambda = 0` bind uses exactly one fact — that `lambda`
multiplies the second summand — and no property of `Wc` whatsoever. Substituting
any `X` leaves the proof intact. So the bind's rejection region over candidate
mechanisms is **empty**, and its expected value is PASS before it executes. That
is `MISTAKES.md` V-10 verbatim: *"Compute the control's expected value before you
run it. If the comparison holds at the extremes of the quantity's own range, the
gate has no rejection region and is vacuous before it executes."*

The second defect is V-9's shape read forward. At the (P) setting the added
mechanism contributes nothing to the object; at the (L) setting standard
attention is not in the object at all (`V1 = 0` deletes it). The two binds are
witnessed by two disjoint operators joined by a straight line along which nothing
is claimed. *"A repair must be shown to change the object it repairs"* — here the
repair changes nothing at the only setting where parity is asserted.

### The general rule

Proposed for `MISTAKES.md` class V. SATURN owns the file; this is the text.

> ### V-23. An identity bind whose family leaves the class it names
>
> An identity bind `O(θ₀) = A` carries information about an architecture only
> through the CLASS `C` that `A` belongs to, and only if `O(θ) ∈ C` for **every**
> `θ` — i.e. the family is a reparametrization of `C` and `θ₀` is one of its
> points. Read that way the bind says "this architecture IS `A`, everywhere, with
> different parameters", which is falsifiable and pins the mechanism.
>
> If `O(θ) ∉ C` off `θ₀`, the bind reports only "the non-`A` part has an off
> switch". That is true of `A + X` for every `X` and of `γ · A` for every `γ`, so
> the bind's rejection region over mechanisms is empty and it certifies nothing.
>
> **Two tests, both necessary, both run before the bind is quoted.**
>
> 1. **Class closure.** Name `C` and its defining invariant. Evaluate the
>    invariant at a random `θ ≠ θ₀`, not only at `θ₀`. For softmax attention the
>    invariant is: rows non-negative and summing to `1`. If it fails off `θ₀`, the
>    family leaves the class and the bind is decoration.
> 2. **Shared carrier.** The coordinates of `θ` that the identity zeroes must be
>    the same coordinates that, at their other setting, deliver the second
>    property. If (P) is at `θ_A = 0` while (L) requires deleting a coordinate
>    that (P) needs alive, the two claims are about two mechanisms and the
>    composition claim is unearned.
>
> **Planted negative** (rule 8). Substitute an absurd mechanism — noise, a
> constant, the label itself — and run the bind. If it still passes, it was never
> a gate. An identity bind with no occupied rejection region is a `[V]`, not a
> `[V-eq]`.

### The rule applied to both objects

| | two-branch | the §4 construction |
|---|---|---|
| class `C` | at `lambda ≠ 0` the object is attention PLUS an unnormalized operator — **outside** `C` | at every `(q, g, s)` it is one causal softmax read-out — **inside** `C`, measured at three random `θ`: `max\|rowsum − 1\| = 2.22e-16`, `min entry > 2.3e-3` |
| shared carrier | (P) at `lambda = 0`, (L) at `V1 = 0` — (L) deletes what (P) certifies | (P) at `(g, s) = (0, 0)`, (L) at `(g, s) = (log a, log(1 − a))` — the **same two coordinates** |
| planted negatives | three tried, **three passed** — none fire | four tried, **four fire**: `0.975`, `0.917`, `1.000`, `0.484` |

Note what the class-closure test does NOT rest on. It is not "the extra term is
inside the softmax rather than outside" as a matter of taste. It is measurable:
the two-branch object's rows do not sum to `1` off `lambda = 0`, and the
construction's do, at every `θ` tried. That is why the ruling is a measurement
and not an aesthetic.

---

## 2 — Q1. CAN THE ADDITIVE-LOGIT FORM REPRODUCE THE LABEL?

### The algebra

Positions `0..s`; position `0` is BOS with no gate and no drive. Gates
`a_1..a_s > 0`, drives `b_1..b_s`, label

```
y_0 = 0,   y_i = a_i y_{i-1} + b_i ,
y_i = sum_{j=1}^{i} P_ij b_j ,   P_ij = prod_{k=j+1}^{i} a_k ,   P_ii = 1 .
```

Log-gates `g_0 = 0`, `g_j = log a_j`; `C = scan(g)` (the inclusive prefix sum of
`CEQ.V15.scan`), so `C_0 = 0` and `P_ij = exp(C_i − C_j)`.

Take the additive-logit hop `l_ij = q_ij + (C_i − C_j)`:

```
softmax_j (q_ij + C_i - C_j)  =  exp(q_ij + C_i - C_j) / sum_{j'<=i} exp(q_ij' + C_i - C_j')
                              =  exp(q_ij - C_j)       / sum_{j'<=i} exp(q_ij' - C_j')
```

`exp(C_i)` is constant across `j` in row `i` and cancels between numerator and
denominator, **identically, for every `q`**. The general statement is stronger
than the cancellation of one factor: any logit of the form `(i-only) + (j-only)`
produces, after the softmax,

```
A_ij = w_j / W_i ,     w_j = exp(s_j - C_j) ,     W_i = sum_{j<=i} w_j .
```

The target-side half of the path product `exp(C_i − C_j)` is **annihilated**. The
row's only remaining dependence on `i` is the RANGE of the running normalizer
`W_i`. What survives normalization is the source-side factor `exp(−C_j)` and
nothing else; the path product does not survive.

With `q ≡ 0` and values `V_j = b_j` this is `A_ij = P_ij / R_i`, where
`R_i = Σ_{j≤i} P_ij` is the unnormalized hop's own row sum. So

```
                out_i  =  y_i / R_i        exactly.
```

**The additive-logit form computes the label divided by the multiplicative form's
own row sum.**

### Numeric instance, `s = 8`, float64

```
  R_i = sum_{j<=i} P_ij (the hop's own row sum)        [1.       1.63492  2.178895 1.852133 1.335952 1.734932 1.437848 1.939116
 1.759648]
  max |out_i * R_i - y_i|                              4.4408920985006262e-16
  max |out_i - y_i|  (the actual error)                0.77110146279863412
  max |out_i - out_i with the C_i term DELETED|        0
  W_i = sum_{j<=i} exp(-C_j)   (s == 0)                [1.000000e+00 2.575001e+00 4.759250e+00 1.034435e+01 4.113549e+01
 9.710737e+01 3.188908e+02 6.584557e+02 1.525246e+03]
  exp(-C_i)   (what (L) would need W_i to be)          [  1.         1.575001   2.184249   5.585104  30.79114   55.971875
 221.783398 339.564887 866.790746]
  max |W_i - exp(-C_i)|                                658.4556546784255
```

Three readings. The claimed identity `out · R = y` holds to `4.44e-16`, two ulp.
The label itself is missed by `0.771`, five orders above any tolerance the round
uses. And the `C_i` term is **dead code**: deleting it from the logit entirely
moves the output by `0` at this draw — the witness for the cancellation, run
rather than argued.

### Verdict on Q1: NO, and no `q` and no rescale of `V` rescues it

Exactness for all `b ∈ R^s` forces `A_ij = P_ij` entrywise (test on the standard
basis), hence `Σ_j A_ij = R_i`, hence the row is stochastic only if `R_i = 1`.
For positive gates `R_i = 1 + Σ_{j<i} P_ij ≥ 1 + a_i > 1` at every `i ≥ 2`. The
argument never mentions `q`, so no query does it.

**The connective fact, and it is the sharpest thing here.** `R_i` is the same
scalar `CEQ.V15.gate_zero_row_sum` evaluates to `i + 1` at `g ≡ 0`. One number is
simultaneously

- the obstruction to (P) for the multiplicative hop — a softmax row needs `R_i = 1`, and it is `i + 1`; and
- the obstruction to (L) for the additive hop — the label needs `R_i = 1`, and it is not.

The multiplicative form has `R_i ≠ 1` and gets (L). The additive form divides by
`R_i` and gets (P). No reweighting of the SAME values can have both. This is why
the fork looked like an impossibility.

---

## 3 — Q2. IS IT A THEOREM?

### It is a theorem, and the hypothesis is the whole content

> **THEOREM (row-stochastic obstruction, value-fixed).**
> Let `a_k > 0` and `y_i = Σ_{j≤i} P_ij b_j` with `P_ij = Π_{k=j+1}^{i} a_k`.
> Let `A` be causal row-stochastic: `A_ij ≥ 0`, `A_ij = 0` for `j > i`,
> `Σ_{j≤i} A_ij = 1`. Let the values be the drives themselves, `V_j = b_j`.
> If `Σ_{j≤i} A_ij b_j = y_i` for every `b ∈ R^s`, then `i ≤ 1`.
>
> *Proof.* Testing on the standard basis gives `A_ij = P_ij`. Summing,
> `1 = Σ_{j≤i} A_ij = R_i = 1 + Σ_{j<i} P_ij ≥ 1 + a_i > 1` for `i ≥ 2`. ∎

> **COROLLARY (the data-dependent version).** Even if `A` may read `a` and `b`
> and be any function whatever, the witness `a ≡ 1, b ≡ 1` gives `y_i = i`, while
> every row-stochastic row over `V ≡ 1` outputs exactly `1`. The failure is a
> RANGE violation — `y_i ∉ conv{V_j}` — not a shortfall of expressiveness.

Both halves are in Lean. `CEQ.V15Fork.no_row_stochastic_with_drive_values` is the
corollary form, which is the stronger of the two because `A` there is an
arbitrary function:

```
no_row_stochastic_with_drive_values : ∀ (a : ℕ → ℝ),
  (∀ (k : ℕ), 0 < a k) →
    ∀ (A : ℕ → ℝ) (n : ℕ),
      (Finset.sum (Finset.Ico 1 (n + 2 + 1)) fun j => A j) = 1 →
        (∀ (b : ℕ → ℝ), (Finset.sum (Finset.Ico 1 (n + 2 + 1)) fun j => A j * b j) = CEQ.V15.chain a b 0 (n + 2)) →
          False
```

Numeric instance:

```
  forced row sums R_i (must be 1.0 to be stochastic)   [1.       1.63492  2.178895 1.852133 1.335952 1.734932 1.437848 1.939116
 1.759648]
  min_i>=2 (R_i - 1)                                   0.3359522744322101
  a == 1, b == 1: label y_i                            [0. 1. 2. 3. 4. 5. 6. 7. 8.]
  any row-stochastic row over V == 1 gives             1.0 at every i
  max_i (y_i - 1)                                      7
```

### The strong form the node was invited to prove is FALSE

The brief asked whether ANY row-stochastic operator can reproduce an unnormalized
path-product label, and noted that a yes-to-impossibility would upgrade the
campaign's one surviving algebraic fact from an observation to a proof.

**It cannot be upgraded, because the impossibility is false.** Drop the
hypothesis `V_j = b_j` — allow the values to be rescaled by a position-local
factor, which is legitimate because `a_j = a(x_j)` is position-local and `V(x)`
is a learned position-local map — and the obstruction dissolves. §4 gives the
operator and Lean gives the theorem.

The solution is moreover essentially unique. Exactness for all `b` forces
`A_ij = P_ij / γ_j`, and row-stochasticity **at every `i`** then forces, by
differencing consecutive rows,

```
gamma_j = 1 / (1 - a_j)
```

with the residual `Π_{k≤i} a_k` — the chain's own initial-condition coefficient —
absorbed by a value-zero slot. This requires `a_j ∈ (0, 1)`, which is **precisely
the stationarity condition `CEQ.V15.bounded_gates_stable` already imposes** via
`g = −softplus(w)`. The construction is not bolted on; it lands exactly on the
range item #6 was already proved into.

**So the honest headline is the opposite of the hoped-for one.** Softmax's
normalizer is not the obstruction to forming path products. It is a change of
units. What it charges is priced in §5.

---

## 4 — Q3. THE CONSTRUCTION

### The operator family

```
        l_ij  =  q_ij  +  (C_i - C_j)  +  s_j            j = 0..i
        O_i   =  sum_{j<=i} softmax_j(l_i.) * V_j
```

Parameters `θ = (W_Q, W_K, W_V, g-head, s-head)`. `g` and `s` are two independent
per-position scalar heads; `C = scan(g)`.

**(P), the identity setting.** `g ≡ 0`, `s ≡ 0`. Then `C_i − C_j = 0` and
`l_ij = q_ij`. The softmax row and everything downstream of it are bit-identical
to standard causal attention. Lean: `CEQ.V15Fork.gate_zero_sink_logit_identity`.

**(L), the oracle setting.** `q ≡ 0`; `g_0 = 0`, `g_j = log a_j`; `s_0 = 0`,
`s_j = log(1 − a_j)`; `V_0 = 0`, `V_j = b_j / (1 − a_j)`. Then `O_i = y_i`
exactly. Lean: `CEQ.V15Fork.Asink_computes_chain`.

### Why (L) works — the telescoping

```
exp(l_i0) = exp(C_i) ,            exp(l_ij) = (1 - a_j) exp(C_i - C_j) .

exp(-C_j)(1 - a_j) = exp(-C_j) - exp(-C_j + g_j) = exp(-C_j) - exp(-C_{j-1}) ,
```

which telescopes over `j = 1..i` to `exp(−C_i) − exp(−C_0) = exp(−C_i) − 1`, so

```
Z_i = exp(C_i) [ 1 + exp(-C_i) - 1 ] = 1   EXACTLY,
```

and the softmax is transparent — not because it was divided out, but because the
running sum of the reweighted source terms telescopes to the reciprocal of the
target factor. The normalizer **reproduces** `exp(C_i)` rather than cancelling
it. That is the single mechanism §2's form lacked.

Then `O_i = exp(C_i)·0 + Σ_{j≥1} (1 − a_j) exp(C_i − C_j) · b_j/(1 − a_j) = Σ_j exp(C_i − C_j) b_j = y_i`.

The whole difference from §S-M's operator is **one factor per entry**, and Lean
states it against §S-M's own hop:

```
Asink_eq_hop : ∀ (a : ℕ → ℝ),
  (∀ (k : ℕ), 0 < a k) → ∀ {i j : ℕ}, j ≠ 0 → j ≤ i → Asink a i j = (1 - a j) * CEQ.V15.W (fun k => Real.log (a k)) i j
```

`Asink_row_sum` — the statement that this one factor makes the row sum to `1` —
needs **no hypothesis on `a` at all`**; it is an algebraic identity.

### Numeric instance, `s = 8`, float64

```
  W_i = sum_{j<=i} exp(s_j - C_j)  (s = log(1-a))      [  1.         1.575001   2.184249   5.585104  30.79114   55.971875
 221.783398 339.564887 866.790746]
  exp(-C_i)                                            [  1.         1.575001   2.184249   5.585104  30.79114   55.971875
 221.783398 339.564887 866.790746]
  max |W_i - exp(-C_i)| / max exp(-C_i)   THE TELESCOPING 2.6231668535454155e-16
  (L) max |Z_i - 1|                                    3.3306690738754696e-16
  (L) min entry of the attention matrix                0.00066336735884623169
  (L) max |row sum - 1|                                0
  (L) max |out_i - y_i|                                2.2204460492503131e-16
  (P) bitwise equal to std_attention(q, V)             True
```

`(P)` is compared against a reference `std_attention` written independently of
the operator under test, so the comparison is a check and not an identity
(`MISTAKES.md` V-3). It is bitwise, not to a tolerance.

`min entry = 6.6e-4 > 0` and `row sum − 1 = 0` say the row is a strictly positive
probability vector, hence a genuine softmax row of some logit vector, hence the
object really is inside the class.

### The bind has a rejection region, and it is occupied

```
  PLANTED NEGATIVES -- this bind has a rejection region and it is occupied.
    drop the key bias s (s == 0)                       max|out - y| = 0.97494590151405114
    drop the value rescale (V_j = b_j)                 max|out - y| = 0.9165274652308163
    drop the BOS sink (V_0 = 1 not 0)                  max|out - y| = 1
    wrong bias s_j = log(1 - a_j) / 2                  max|out - y| = 0.48449311856985267
```

Four deliberate mutilations, four failures at `O(1)`. Contrast §1, where three
substitutions including the answer key all passed. This is the operational
difference between an informative identity bind and a vacuous one.

### An unasked-for consequence: the query-side scan term is redundant

§2's cancellation applies to the construction too — `A_ij = w_j / W_i` has no
`exp(C_i)` in it. The target factor comes back out of the running normalizer, not
out of the logit. So the same construction runs with a **key-only** bias:

```
  key-only logit l_ij = -C_j + log(1 - a_j): max|out - y| 2.2204460492503131e-16
    vs the C_i - C_j version: max|difference|          2.7755575615628914e-16
```

`CEQ.V15Fork.gate_zero_key_logit_identity` carries (P) for this form.

This changes what has to be built. ARM PL does not need `C_i − C_j` in the
logits. It needs **one extra key channel carrying `−C_j + log(1 − a_j)`**, and
that channel's `−C_j` is the scan's output, so the scan is still required — it
just enters key-side only. Under a causal softmax a scan can ONLY enter key-side;
the query-side half is annihilated by construction, and the target factor must be
recovered from the normalizer.

---

## 5 — WHAT THE NORMALIZER ACTUALLY CHARGES

The impossibility is false, but something survives it, and it is a bound rather
than a barrier.

> **THEOREM (the value-scale charge).** For any causal row-stochastic `A` and any
> values `V`, `O_i ∈ conv{V_j : j ≤ i}`, hence `max_i |O_i| ≤ max_j |V_j|`.
> Reproducing the chain label therefore requires
> `max_j |V_j| ≥ max_i |y_i|`.

For constant gates `a` and `b ≡ 1`, `max_i |y_i| = (1 − a^s)/(1 − a)`.

```
           a    max|y_i| (needed)    max|V_j| (used)      ratio      1/(1-a)
    0.500000            1.9921875                  2      1.004            2
    0.900000            5.6953279                 10      1.756           10
    0.990000          7.725530557                100      12.94          100
    0.999000           7.97205593               1000      125.4         1000
    0.999999             7.999972            1000000   1.25e+05        1e+06
```

The construction pays `1/(1 − a)`, which is the `s → ∞` optimum, and is tight
(ratio `1.004`) when the memory is shorter than the context. It over-pays by
`1/(1 − a^s)` when `a^s → 1`.

**A constant rescale `γ = max_i R_i` would pay less at finite `s`, and the
architecture cannot compute it.** Its sink logit would be `log(1 − R_i/γ)`, an
arbitrary function of the whole prefix and not a scan difference. The
`(1 − a_j)` reweighting is the unique choice whose sink logit is `C_i − C_0` —
the scan's own output, with `s_0 = 0` and no special case. That is why it is the
one to build, and it is a stronger argument for it than tightness.

**The preregisterable consequence, filed for VENUS.** The value head's output
range must grow like `1/(1 − a)`. ARM PL's gate learning is therefore predicted
to fail first, and to fail by value saturation rather than by gate error, on
long-memory chains — the regime BED-M's `t* = 8` cell is designed to probe. The
diagnostic is `max_j |V_j|` against `1/(1 − â_max)`, printed per cell, not the
gate probe alone.

---

## 6 — THE SIGNED VARIANT, AND THE ONE ITEM THIS NODE DOES NOT SETTLE

`CEQ.V15.parity_sign` gives `χ(P_i − P_j) = χ(P_i)·χ(P_j)` in `ZMod 2`, so the
sign FACTORS into a source half and a target half. The source half rides in
`V_j`. The target half `χ(P_i)` is constant across `j` in row `i`, so a
row-stochastic — hence non-negative — weight cannot carry it.

```
  max |chi_i * out_i - y_signed_i|                     4.4408920985006262e-16
  max |out_i - y_signed_i| (no output gate)            2.8870627765992714
```

The signed carrier is exact, but only as `O_i = χ(P_i) · (head read-out)`.
`χ(P_i) = 1` at zero sign bits is a MULTIPLICATIVE off switch, which by the rule
of §1 any `O = γ(θ)·A` passes. **So the signed variant's parity bind is vacuous
by this node's own rule**, unless `χ(P_i)` is applied by machinery already inside
the transformer class — a following MLP reading `χ(P_i)` off position `i`'s own
residual would be; an output gate bolted to the head would not.

This node does not settle which. It is the one open item handed back, and it is
handed back deliberately rather than resolved in the flattering direction.

---

## 7 — THE WIDENED BRIEF'S OWN CANDIDATE: A PER-ROW READOUT SCALAR

The widened brief names one candidate amendment to test before inventing others:
keep the additive-logit form, accept that it produces the path product only up to
a per-row scalar, and let the readout's own scale divide `Z_i` back out — *"since
a per-row scalar is exactly what a readout's own scale can absorb"*. It asks
three things of it: is it legitimate, is it still bitwise attention at `g == 0`,
and does it reproduce the label.

### What `ρ_i` is forced to be

Not a free constant. With `q ≡ 0, s ≡ 0` the row is `A_ij = exp(−C_j)/W_i`, so
recovering `P_ij = exp(C_i − C_j)` requires

```
rho_i = exp(C_i) * W_i          (W_i is the head's own softmax denominator)
```

and then `ρ_i A_ij = exp(C_i) W_i exp(−C_j)/W_i = exp(C_i − C_j) = P_ij`.

**Does it reproduce the label? Yes, exactly** — and that is the problem:

```
  (L) max |rho_i * A_ij - P_ij|  entrywise             1.1102230246251565e-16
  (L) max |rho_i * out_i - y_i|                        2.2204460492503131e-16
```

The effective operator `ρ_i A_ij` is §S-M's unnormalized hop **entry for entry**,
to `1.11e-16`. An operator that divides its own normalizer back out is the
unnormalized operator wearing a softmax. So `gate_zero_not_stochastic` applies to
it verbatim, and it inherits the refutation it was supposed to escape.

### Is it still bitwise attention at `g == 0`? No

At `g ≡ 0`, `exp(C_i) = 1` and `ρ_i` is forced to `Z_i`, the head's own
denominator:

```
    q == 0: rho_i at g == 0                            [1. 2. 3. 4. 5. 6. 7. 8. 9.]
      max |tied output - standard attention|           2.0664886349202778
    q = a trained head's logits: rho_i at g == 0       [ 1.3903  6.2303  3.8952  2.677   9.5342  8.3889  5.1397  8.6119 35.4424]
      max |tied output - standard attention|           5.2916669507402556
```

`[1 2 3 4 5 6 7 8 9]` is `i + 1` read off a float64 run — `CEQ.V15.gate_zero_row_sum`
appearing as a measured quantity rather than as a theorem. Parity is missed by
`2.07` at `q ≡ 0` and by `5.29` at a trained head's logits.

### Is it legitimate? Only if `ρ` is a free coordinate — and then it is vacuous

The only way to have `ρ_i = 1` at parity while `ρ_i = exp(C_i) W_i` at the oracle
setting is to make `ρ` a FREE coordinate with an off switch. That is
`O = γ(θ)·A` with `γ(θ₀) = 1`, which is §1's defect in multiplicative form:

```
    rho = R_i, which solves (L)                        (P) at rho == 1 bitwise: True
    rho = i.i.d. positive noise                        (P) at rho == 1 bitwise: True
    rho = the answer key y_i / out_i                   (P) at rho == 1 bitwise: True
```

Empty rejection region again, answer key included. Under L-AMEND's own proviso
this is *"a worse outcome than the refutation it replaces"*.

### The dichotomy, and there is no third branch

> **Tie `ρ` to the gate mechanism ⇒ (P) fails.** `ρ_i` must equal `exp(C_i) W_i`
> for (L); at `g ≡ 0` that is `W_i`, which equals `1` only at `i = 0`.
> **Leave `ρ` free ⇒ (P) holds and is vacuous.** Passed by every `γ·A`.

The brief anticipated the sharp negative — *"if the per-row scalar CANNOT be
absorbed, because the label's `y_s` genuinely needs unnormalized magnitudes and
not just directions, then say that"*. **Say it: it cannot.** `R_i` runs `1.00` to
`2.18` across nine positions in this draw. BED-M scores NRMSE against `y` itself,
so a per-position factor of up to `2.18` is a magnitude error, not a gauge
freedom. The label is not defined up to a per-position scale, and the additive
form delivers only the direction.

**But that is not where the algebra lands, and this is the point of §4.** The
construction does not divide the normalizer out. It arranges for the normalizer
to *telescope to the target factor on its own* — `W_i = exp(−C_i)` to `2.6e-16` —
so no scalar has to be reapplied, no coordinate has to be switched off, and
`Z_i = 1` identically. That is why it escapes the dichotomy while the readout
route cannot.

---

## 8 — THE AMENDMENT, IN L-AMEND'S FOUR PARTS

### 8.1 The original clause

`CEQ_V15_CONTRACT.md` PART I §S-M and the CLAIM SHAPE paragraph:

> ONE unnormalized causal hop `W_ij = exp(C_i - C_j)` on values `V(x)`.
> `[RUN]` reproduces the chain label to `4.0e-15`; `g == 0` gives the causal
> all-ones mask = standard attention.
>
> **PARITY WITH SELF-ATTENTION is by IDENTITY BIND, not TOST.** `g == 0` gives
> bitwise standard attention (Lean #5).

### 8.2 The refutation

`CEQ.V15.gate_zero_row_sum` / `gate_zero_not_stochastic`: at `g ≡ 0` the row sums
to `i + 1`, never `1`, so the operator is not any softmax row of any head. §2 of
this file adds the second half — the additive-logit repair that restores parity
loses the label, computing `y_i / R_i` — and §7 closes the readout route that
would put the missing factor back. The single scalar `R_i` obstructs (P) for one
form and (L) for the other.

### 8.3 The new formula

```
        l_ij  =  q_ij  -  C_j  +  s_j                    j = 0..i,   C = scan(g)
        O_i   =  sum_{j<=i} softmax_j(l_i.) * V_j
```

with `g` and `s` two independent per-position scalar heads.

- **(P)** at `g ≡ 0, s ≡ 0`: `l_ij = q_ij`, bitwise standard attention.
  `CEQ.V15Fork.gate_zero_key_logit_identity`.
- **(L)** at `q ≡ 0`, `g_j = log a_j` (`g_0 = 0`), `s_j = log(1 − a_j)`
  (`s_0 = 0`), `V_0 = 0`, `V_j = b_j/(1 − a_j)`: `O_i = y_i` exactly, `2.2e-16`
  at `s = 8`. `CEQ.V15Fork.Asink_computes_chain`, with `Asink_row_sum` and
  `Asink_nonneg` establishing that the row is a genuine probability vector.

The whole difference from §S-M is `Asink_eq_hop`: one factor `(1 − a_j)` on
§S-M's own hop, plus a value-zero BOS slot.

### 8.4 What the new formula gives up

Seven things, none rhetorical.

1. **The word "unnormalized" goes, and with it §S-M's operator.** The hop is now
   a softmax row. `prefix_logit_mask` (#2) survives as an algebraic fact about
   `W`, but `W` is no longer the arm's operator — it is one factor of it.
2. **`prefix_logit_computes_chain` is no longer the arm's (L) bind.** It stays
   true; it now describes an operator the arm does not use.
   `Asink_computes_chain` replaces it, and the round's `[RUN] 4.0e-15` line must
   be re-measured against the new operator rather than re-quoted.
3. **The query-side scan term is deleted.** §S-M's `C_i − C_j` becomes `−C_j`.
   Under a causal softmax a scan can only enter key-side; the query half is
   annihilated identically (`max |out − out with C_i deleted| = 0`). Anything in
   the round that reasoned about the query half is void.
4. **One extra per-position scalar head `s`.** Not free slack: it is forced
   uniquely by row-stochasticity at every `i`, and dropping it costs `0.975` on
   the (L) bind.
5. **Stationarity stops being a nicety and becomes load-bearing.** §S-M cited
   `a ∈ (0,1]` for stability (#6). The construction needs `a_j ≠ 1` to exist at
   all, and the parity point `g ≡ 0` is `a ≡ 1` — the boundary. The two binds sit
   at opposite ends of one degeneration, and at the parity setting itself the
   oracle values are undefined. Each bind is exact at its own `θ`; there is no
   `θ` at which both hold, and there was never going to be.
6. **Value dynamic range `1/(1 − a)`.** §5 prices it and it is the real cost:
   `1e6` at `a = 1 − 1e-6`. This is a predicted failure mode for ARM PL on
   long-memory chains, and the diagnostic column is `max_j |V_j|` against
   `1/(1 − â_max)`, printed per cell.
7. **A structurally distinguished BOS slot**, value exactly `0`, logit exactly
   the scan difference to `C_0`. Dropping it costs `1.000` on the (L) bind. And
   **the signed variant is not covered** — §6 is open and is handed back rather
   than resolved in the flattering direction.

### 8.5 Does it move toward the north star?

> *Attention that is EQUAL to self-attention on its own ground, built FROM
> softmax and AdamW, and capable on ground they cannot occupy.*

On the first clause the distance goes from **refuted to bitwise**: §S-M's
operator was proved not to be any softmax row; the amended one is standard
attention bit-for-bit at `(g, s) = (0, 0)`, against an independently written
reference. On the second, "built FROM softmax" goes from false to literal — the
old hop was not a softmax at all, the new one is a softmax row at **every** `θ`,
measured at three random draws (`max |rowsum − 1| = 2.22e-16`). The third clause
is untouched by this node and is still owed a floor crossing.

---

## 9 — WHAT ARM PL SHOULD BE, AND THE CONTRACT LINES THAT MOVE

The contract's PARITY WITH SELF-ATTENTION clause does **not** retire. §S-M's
OPERATOR does; §8.3 is its replacement and §8.4 its price.

The `[RUN]` line in §S-M — *"reproduces the chain label to `4.0e-15`; `g == 0`
gives the causal all-ones mask = standard attention"* — must be rewritten. "The
causal all-ones mask" is the thing `gate_zero_not_stochastic` refuted; the new
line is `g == 0, s == 0` gives **bitwise** standard attention, and the label is
reproduced to `2.2e-16` at `s = 8` by the same operator.

**One-sentence consequence for what ARM PL should be:** a single causal softmax
head with one extra key channel carrying `−scan(g)_j + log(1 − a_j)` and values
`b_j/(1 − a_j)` against a value-zero BOS sink — the whole architectural
difference from a standard head is one scan feeding the KEY projection and one
reciprocal-gate factor on the VALUE, and both the parity bind and the label bind
are then binds on that one head.

### Prior art this owes, and does not yet have

The reparametrization `b_j = (1 − a_j) v_j` is the coupled input/forget-gate form
— the EMA / leaky integrator, the GRU's tied gates, and the ZOH discretization of
a diagonal SSM. Under it the chain is `y_i = a_i y_{i-1} + (1 − a_i) v_i`, whose
unrolling is manifestly a convex combination of `v_j` plus `Π a_k · y_0`; the
"sink" is the initial-condition term and the row-stochasticity is not a
coincidence. `scripts/v15_n1_probes/` covers SSD/Mamba-2, GLA and RetNet but none
of them at the `(1 − a)` coupling. **This enters as a POINTER for `n1`, not as
`[V-eq]`** — no equation was fetched for it by this node, and L-EQ forbids
citing it as one.

---

## LIMITS

- Everything above is exact arithmetic and float64 identity, at `s = 8`, one
  draw, seed `15`, gates in `(0.15, 0.85)`. Nothing here is evidence that a
  trained arm finds any of these settings. The binds are expressibility, not
  learnability, exactly as the contract's binds are.
- The (L) setting has `q ≡ 0`. At the oracle setting the head is not doing
  content addressing. This is inherent to the bind's design and is shared with
  §S-M's own oracle-gate bind; it is not a defect introduced here. A separable
  `q_ij = u_i + v_j` also works, with `v_j` absorbing `s_j`.
- `Asink_computes_chain`'s hypothesis is `∀ k, a k ≠ 1`, including `k = 0` which
  the sum never touches. The hypothesis is stronger than needed, which is the
  conservative direction, and is satisfied at every `k` by `a = exp(−softplus w)`.
- §5's value-scale bound is stated for scalar values. For `V_j ∈ R^d` the
  convex-hull argument gives it coordinatewise; the constant is unchanged, but no
  `d > 1` instance was run.
- §6 is not settled and is flagged as not settled.
- §7 rejects the widened brief's candidate on a dichotomy whose second branch
  is a vacuity judgement, not a measurement. The FIRST branch — tie `ρ` to the
  gate and parity fails by `2.07` and `5.29` — is measured. The second rests on
  the V-23 rule stated in §1, which is this node's proposal and is not yet
  ratified in `MISTAKES.md`. SATURN owns that file.
- The amendment of §8 has its identity theorems (L-LEAN) and none of its
  training. It has NOT been shown learnable, and §8.4 item 6 predicts where it
  will not be.
- No git command that writes was run. Files written: the three named in the
  brief, plus the one import line in `lean/CEQ.lean`. `lean/CEQ/V15.lean` was
  read and not modified.

---

## APPENDIX A — `scripts/v15_jupiter2_fork_probe.py`, full output

Run at commit `deee6c4`, Windows 11, python + numpy float64. Exit code `0`;
every `assert` in the file passed.

```
$ python scripts/v15_jupiter2_fork_probe.py
S = 8, positions 0..8, gates a_1..a_8 in (0.15, 0.85), float64
label y (from the recurrence)      : [ 0.       -1.443531 -0.023826 -0.604967  1.984352  1.820317 -0.079593
  0.149503  0.273545]
|y - P @ b| (two routes to the label): 1.1102230246251565e-16

==============================================================================
1  THE TWO-BRANCH BIND -- out = softmax(q) @ V1 + lambda * X
==============================================================================
  Claim under test: "lambda = 0 gives standard attention, so (P) holds;
  V1 = 0, lambda = 1 gives the label, so (L) holds; one family, both binds."
  The gate is run against three X, the third of which is the answer key.
  X = Wc(g) @ V2, the honest gate                      (P) bitwise: True   (L) max|out - y|: 1.1102230246251565e-16
  X = i.i.d. Gaussian noise                            (P) bitwise: True   (L) max|out - y|: 4.3977935722055612
  X = the LABEL y itself (answer key)                  (P) bitwise: True   (L) max|out - y|: 0

  RULING: VACUOUS. The (P) half passed bitwise for all three X, including one
  that is literally the label. The proof of the lambda=0 bind uses exactly one
  fact -- that lambda multiplies the second summand -- and no property of X.
  Its rejection region over candidate mechanisms is EMPTY, so its expected
  value is PASS before it runs: MISTAKES.md V-10 verbatim. And at the (L)
  setting (V1 = 0) standard attention is absent from the object, so the two
  binds are witnessed by two disjoint objects: MISTAKES.md V-9 read forward.

==============================================================================
2  ADDITIVE-LOGIT ONLY -- l_ij = q_ij + (C_i - C_j), values = the drives
==============================================================================
  ALGEBRA. Any logit of the form (i-only) + (j-only) gives, after softmax,
      A_ij = w_j / W_i,     w_j = exp(s_j - C_j),   W_i = sum_{j<=i} w_j,
  because exp(C_i) is constant across j in row i and cancels between numerator
  and denominator. The target-side half of the path product exp(C_i - C_j) is
  therefore ANNIHILATED, and the row's only i-dependence is the range of the
  running normalizer W_i. With s == 0, q == 0, V = b that is A_ij = P_ij / R_i,
  so out_i = y_i / R_i.

  R_i = sum_{j<=i} P_ij (the hop's own row sum)        [1.       1.63492  2.178895 1.852133 1.335952 1.734932 1.437848 1.939116
 1.759648]
  max |out_i * R_i - y_i|                              4.4408920985006262e-16
  max |out_i - y_i|  (the actual error)                0.77110146279863412
  max |out_i - out_i with the C_i term DELETED|        0
  W_i = sum_{j<=i} exp(-C_j)   (s == 0)                [1.000000e+00 2.575001e+00 4.759250e+00 1.034435e+01 4.113549e+01
 9.710737e+01 3.188908e+02 6.584557e+02 1.525246e+03]
  exp(-C_i)   (what (L) would need W_i to be)          [  1.         1.575001   2.184249   5.585104  30.79114   55.971875
 221.783398 339.564887 866.790746]
  max |W_i - exp(-C_i)|                                658.4556546784255

  VERDICT ON Q1: NO. The additive-logit form computes the label divided by the
  MULTIPLICATIVE form's own row sum, exactly. And R_i is the same scalar Lean's
  `gate_zero_row_sum` evaluates to i+1 at g == 0. One number is simultaneously
  the obstruction to (P) for the multiplicative hop (a softmax row needs R_i=1)
  and the obstruction to (L) for the additive hop (the label needs R_i=1).
  No choice of q rescues it: exactness for all b forces A_ij = P_ij entrywise,
  hence row sum R_i, hence R_i = 1, and R_i >= 1 + a_i > 1 for i >= 2.

==============================================================================
3  ROW-STOCHASTIC WITH THE DRIVES AS VALUES -- the theorem
==============================================================================
  THEOREM. a_k > 0; A causal row-stochastic; values V_j = b_j. If
  sum_{j<=i} A_ij b_j = y_i for every b in R^S, then A_ij = P_ij (test on the
  standard basis), so sum_j A_ij = R_i, and row-stochasticity forces R_i = 1.
  But R_i = 1 + sum_{j<i} P_ij >= 1 + a_i > 1 for i >= 2. Contradiction.

  forced row sums R_i (must be 1.0 to be stochastic)   [1.       1.63492  2.178895 1.852133 1.335952 1.734932 1.437848 1.939116
 1.759648]
  min_i>=2 (R_i - 1)                                   0.3359522744322101

  COROLLARY, allowing A to read the data. Take a == 1, b == 1: then y_i = i,
  while every row-stochastic row over V == 1 outputs exactly 1. The failure is
  a RANGE violation, y_i not in conv{V_j}, not a lack of expressiveness.

  a == 1, b == 1: label y_i                            [0. 1. 2. 3. 4. 5. 6. 7. 8.]
  any row-stochastic row over V == 1 gives             1.0 at every i
  max_i (y_i - 1)                                      7

==============================================================================
4  THE CONSTRUCTION -- one softmax head carrying BOTH binds
==============================================================================
  logit  l_ij = q_ij + (C_i - C_j) + s_j          j = 0..i
  value  V_j

  (P) identity setting : g == 0, s == 0            -> l_ij = q_ij, bitwise attention
  (L) oracle   setting : q == 0, g_j = log a_j, g_0 = 0,
                         s_0 = 0, s_j = log(1 - a_j),
                         V_0 = 0, V_j = b_j / (1 - a_j)

  WHY (L) WORKS. exp(l_i0) = exp(C_i); exp(l_ij) = (1 - a_j) exp(C_i - C_j).
  exp(-C_j)(1 - a_j) = exp(-C_j) - exp(-C_j + g_j) = exp(-C_j) - exp(-C_{j-1}),
  which telescopes over j = 1..i to exp(-C_i) - exp(-C_0) = exp(-C_i) - 1, so
      Z_i = exp(C_i) [ 1 + exp(-C_i) - 1 ] = 1
  EXACTLY: the softmax normalizer is transparent, not because it was divided
  out but because the running sum of the reweighted source terms telescopes to
  the reciprocal of the target factor. The normalizer REPRODUCES exp(C_i)
  instead of cancelling it, which is what section 2's form could not do.

  W_i = sum_{j<=i} exp(s_j - C_j)  (s = log(1-a))      [  1.         1.575001   2.184249   5.585104  30.79114   55.971875
 221.783398 339.564887 866.790746]
  exp(-C_i)                                            [  1.         1.575001   2.184249   5.585104  30.79114   55.971875
 221.783398 339.564887 866.790746]
  max |W_i - exp(-C_i)| / max exp(-C_i)   THE TELESCOPING 2.6231668535454155e-16
  (L) softmax normalizer Z_i                           [1.                 1.                 0.9999999999999999
 1.                 1.                 0.9999999999999999
 1.                 0.9999999999999997 0.9999999999999998]
  (L) max |Z_i - 1|                                    3.3306690738754696e-16
  (L) min entry of the attention matrix                0.00066336735884623169
  (L) max |row sum - 1|                                0
  (L) max |out_i - y_i|                                2.2204460492503131e-16
  (P) bitwise equal to std_attention(q, V)             True

  THE QUERY-SIDE SCAN TERM IS REDUNDANT. Section 2's cancellation applies here
  too: A_ij = w_j / W_i has no exp(C_i) in it. The target factor is recovered
  from the RUNNING NORMALIZER W_i = exp(-C_i), not from the C_i term in the
  logit. So the same construction runs with a KEY-ONLY logit bias:
  key-only logit l_ij = -C_j + log(1 - a_j): max|out - y| 2.2204460492503131e-16
    vs the C_i - C_j version: max|difference|          2.7755575615628914e-16

  PLANTED NEGATIVES -- this bind has a rejection region and it is occupied.
    drop the key bias s (s == 0)                       max|out - y| = 0.97494590151405114
    drop the value rescale (V_j = b_j)                 max|out - y| = 0.9165274652308163
    drop the BOS sink (V_0 = 1 not 0)                  max|out - y| = 1
    wrong bias s_j = log(1 - a_j) / 2                  max|out - y| = 0.48449311856985267

  CLASS CLOSURE -- the family never leaves the softmax class.
    random theta #0: max|rowsum - 1|, min entry        2.2204460492503131e-16, 0.0034241234454139295
    random theta #1: max|rowsum - 1|, min entry        2.2204460492503131e-16, 0.0023141463699415839
    random theta #2: max|rowsum - 1|, min entry        2.2204460492503131e-16, 0.0036530167673236367

  VERDICT ON Q3: A CONSTRUCTION EXISTS, and therefore the impossibility the
  node hoped for is FALSE. Softmax's normalizer is not the obstruction to
  forming path products. The obstruction was only ever to forming them with
  the DRIVES as values; rescaling the values by the position-local factor
  1/(1 - a_j) and giving BOS a value-zero slot dissolves it.

==============================================================================
5  THE VALUE-SCALE CHARGE -- what survives of the impossibility
==============================================================================
  THEOREM. For any causal row-stochastic A and any V, out_i lies in
  conv{V_j : j <= i}, so max_i |out_i| <= max_j |V_j|. Reproducing the label
  therefore costs value dynamic range max_j |V_j| >= max_i |y_i|. That is the
  entire residue of the impossibility: a change of units, not a barrier.

           a    max|y_i| (needed)    max|V_j| (used)      ratio      1/(1-a)
    0.500000            1.9921875                  2      1.004            2
    0.900000            5.6953279                 10      1.756           10
    0.990000          7.725530557                100      12.94          100
    0.999000           7.97205593               1000      125.4         1000
    0.999999             7.999972            1000000   1.25e+05        1e+06

  The bound is attained as a^S -> 0 (memory shorter than the context) and is
  loose by 1/(1 - a^S) when a^S -> 1. The construction pays 1/(1 - a), the
  S -> infinity optimum. A constant rescale gamma = max_i R_i would pay less at
  finite S, but its sink logit log(1 - R_i/gamma) is an arbitrary function of i
  and is NOT a scan difference -- so the architecture cannot compute it. The
  (1 - a_j) reweighting is the unique choice whose sink logit is the scan's own
  output, which is why it is the one to build.

==============================================================================
6  THE SIGNED VARIANT -- where the ruling does not save it
==============================================================================
  Lean #3 gives chi(P_i - P_j) = chi(P_i) * chi(P_j) in ZMod 2, so the sign
  FACTORS into a source half and a target half. The source half rides in V_j.
  The target half chi(P_i) is constant across j in row i, so a row-stochastic
  (hence non-negative) weight cannot carry it: it has to be a per-position
  multiplier on the OUTPUT.

  max |chi_i * out_i - y_signed_i|                     4.4408920985006262e-16
  max |out_i - y_signed_i| (no output gate)            2.8870627765992714

  So the signed carrier IS exact -- but only as O_i = chi(P_i) * (head read-out),
  and `chi(P_i) == 1 at zero sign bits` is a MULTIPLICATIVE off switch that any
  O = gamma(theta) * A passes. By the rule stated in section 1 that bind is
  vacuous unless chi(P_i) is applied by machinery already inside the transformer
  class (a following MLP reading chi(P_i) off position i's own residual), not by
  an output gate bolted to the head. This node does not settle which; it is the
  one open item handed back.

==============================================================================
7  THE PER-ROW SCALAR -- 'let the readout divide Z_i back out'
==============================================================================
  The candidate route: keep the additive-logit form (which has (P) by
  `gate_zero_logit_identity`), accept that it produces the path product only up
  to a per-row scalar, and have the readout supply that scalar rho_i.

  WHAT rho_i HAS TO BE. With q == 0, s == 0 the row is A_ij = exp(-C_j) / W_i,
  so recovering P_ij = exp(C_i - C_j) needs exactly

      rho_i = exp(C_i) * W_i        (W_i = the head's own softmax denominator)

  and then rho_i * A_ij = exp(C_i) W_i exp(-C_j) / W_i = exp(C_i - C_j) = P_ij.
  The scalar is not a free constant. It is the normalizer, put back.

  (L) max |rho_i * A_ij - P_ij|  entrywise             1.1102230246251565e-16
  (L) max |rho_i * out_i - y_i|                        2.2204460492503131e-16

  So (L) is recovered -- and the effective operator rho_i * A_ij IS S-M's
  unnormalized hop, entry for entry. An operator that divides its own
  normalizer back out is the unnormalized operator wearing a softmax, and
  `gate_zero_not_stochastic` applies to it verbatim. Measured at g == 0:

    q == 0: rho_i at g == 0                            [1. 2. 3. 4. 5. 6. 7. 8. 9.]
      max |tied output - standard attention|           2.0664886349202778
    q = a trained head's logits: rho_i at g == 0       [ 1.3903  6.2303  3.8952  2.677   9.5342  8.3889  5.1397  8.6119 35.4424]
      max |tied output - standard attention|           5.2916669507402556
  (P) FAILS for the tied scalar: at g == 0 it is forced to Z_i, which is i+1
  at q == 0 -- Lean's `gate_zero_row_sum` exactly -- and is the head's own
  denominator in general. Never 1.

  THE ONLY WAY OUT is to make rho a FREE coordinate switched off at parity.
  That is O = gamma(theta) * A with gamma(theta_0) = 1, and it has section 1's
  defect in multiplicative form:

    rho = R_i, which solves (L)                        (P) at rho == 1 bitwise: True
    rho = i.i.d. positive noise                        (P) at rho == 1 bitwise: True
    rho = the answer key y_i / out_i                   (P) at rho == 1 bitwise: True

  VERDICT ON THE PER-ROW SCALAR: A DICHOTOMY, AND NEITHER BRANCH IS ADMISSIBLE.
  Tie rho to the gate mechanism and (P) fails, because g == 0 forces rho = Z_i
  and Z_i is never 1. Leave rho free with an off switch and (P) holds and is
  VACUOUS, passed by every gamma * A including the answer key. There is no
  third option: rho_i must equal exp(C_i) W_i for (L), and that quantity equals
  1 at g == 0 only if W_i = 1, which happens only at i = 0.

  The label is not defined up to a per-position scale. R_i runs 1.00 to 2.18
  across positions in this draw, so absorbing it is a per-position magnitude
  error, not a gauge freedom. Section 4's construction is the amendment that
  avoids the dichotomy: it does not divide the normalizer out, it arranges for
  the normalizer to telescope to the target factor on its own.

==============================================================================
ALL ASSERTS PASSED.
==============================================================================
```

---

## APPENDIX B — `lean/CEQ/V15Fork.lean`

New file, 11 top-level declarations plus 2 definitions. `lean/CEQ/V15.lean` was
**not** edited; `V15Fork` imports it. One added line in `lean/CEQ.lean`.

### Build

```
$ cd lean && lake build 2>&1 | head -60; echo "=== EXIT: $? ==="
[1516/1527] Building CEQ.V15Fork
[1526/1527] Building CEQ
=== EXIT: 0 ===
```

No errors, no warnings, no `sorry` warning.

```
$ grep -n sorry lean/CEQ/V15Fork.lean
44:  No `sorry`.
```

The single hit is the sentence in the module doc comment.

### Axioms

Exit `0` is necessary and not sufficient — a theorem closed by an axiom still
builds — so every declaration was checked against the kernel's axiom list, the
same discipline `V15_N3_LEAN.md` used.

```
$ lake env lean AxCheckFork.lean
'CEQ.V15Fork.Asink_row_sum' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.Asink_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.chain_eq_sum' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.Asink_computes_chain' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.chain_one_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.chain_one_gt_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.no_row_stochastic_with_drive_values' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.gate_zero_sink_logit_identity' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.gate_zero_key_logit_identity' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.Asink_zero_eq_hop' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Fork.Asink_eq_hop' depends on axioms: [propext, Classical.choice, Quot.sound]
```

`propext`, `Classical.choice`, `Quot.sound` are Lean's three standard axioms.
`sorryAx` appears nowhere.

### The statements as the kernel elaborated them

Pasted from `#check`, not retyped.

```
Asink_row_sum : ∀ (a : ℕ → ℝ) (i : ℕ), (Finset.sum (Finset.range (i + 1)) fun j => Asink a i j) = 1

Asink_nonneg : ∀ (a : ℕ → ℝ), (∀ (k : ℕ), 0 < a k) → (∀ (k : ℕ), a k < 1) → ∀ (i j : ℕ), 0 ≤ Asink a i j

Asink_computes_chain : ∀ (a b : ℕ → ℝ),
  (∀ (k : ℕ), a k ≠ 1) →
    ∀ (i : ℕ), (Finset.sum (Finset.range (i + 1)) fun j => Asink a i j * Vsink a b j) = CEQ.V15.chain a b 0 i

no_row_stochastic_with_drive_values : ∀ (a : ℕ → ℝ),
  (∀ (k : ℕ), 0 < a k) →
    ∀ (A : ℕ → ℝ) (n : ℕ),
      (Finset.sum (Finset.Ico 1 (n + 2 + 1)) fun j => A j) = 1 →
        (∀ (b : ℕ → ℝ), (Finset.sum (Finset.Ico 1 (n + 2 + 1)) fun j => A j * b j) = CEQ.V15.chain a b 0 (n + 2)) →
          False

gate_zero_sink_logit_identity : ∀ (g s : ℕ → ℝ),
  (∀ (k : ℕ), g k = 0) →
    (∀ (k : ℕ), s k = 0) → ∀ (q : ℕ → ℕ → ℝ) (i j : ℕ), q i j + (CEQ.V15.scan g i - CEQ.V15.scan g j) + s j = q i j

Asink_eq_hop : ∀ (a : ℕ → ℝ),
  (∀ (k : ℕ), 0 < a k) → ∀ {i j : ℕ}, j ≠ 0 → j ≤ i → Asink a i j = (1 - a j) * CEQ.V15.W (fun k => Real.log (a k)) i j
```

Two of these are worth reading twice.

`Asink_row_sum` has **no hypothesis on `a`**. The reweighted row sums to `1` as
an algebraic identity; positivity and the bound `a k < 1` are needed only to make
the entries non-negative, which is `Asink_nonneg`'s separate job. So the
telescoping is not a coincidence of the stationary regime.

`no_row_stochastic_with_drive_values` quantifies over an ARBITRARY `A : ℕ → ℝ`.
It may read `a`, it may read `b`, it may be any function at all. Only the two
class-defining properties are assumed. It is `V15_N3_LEAN.md`'s incompatibility,
stated at the strength it actually holds at — and `Asink_computes_chain`, four
declarations above it in the same file, is the proof that the hypothesis
`V_j = b_j` is the whole of its content.

---

## FILES TOUCHED

- `V15_JUPITER2_FORK.md` — this file, new.
- `scripts/v15_jupiter2_fork_probe.py` — new, the float64 probes, exit `0`.
- `lean/CEQ/V15Fork.lean` — new.
- `lean/CEQ.lean` — one added line, `import CEQ.V15Fork`.

`lean/CEQ/V15.lean` was read and not modified. No git command that writes was run.
