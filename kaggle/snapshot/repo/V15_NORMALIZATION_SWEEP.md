# V15 normalization-boundary sweep — MARS register

Adversary sweep for the CEQ v15 composition round. The Lean node
(`V15_N3_LEAN.md`, `lean/CEQ/V15.lean`) refuted the contract's `g == 0`
parity clause: S-M's hop is UNNORMALIZED (`W_ij = exp(C_i - C_j)`), so at
`g == 0` row `i` sums to `i + 1` while every softmax row sums to `1`. This
document sweeps `ceq/*.py`, `scale/*.py`, `tests/**`, and the claim documents
for every other instance of that mechanism — an identity asserted between a
normalized object and an unnormalized one, carried by a matching mask or
support pattern, or hidden behind a scale-invariant statistic — and for the
adjacent "conservation asserted where it is not conserved" reading of the
same defect.

Runnable checks: `scripts/v15_norm_sweep_probe.py`. Ran clean, exit 0, all
assertions passed, output pasted in full below each check. `house-events.jsonl`
was not grepped.

---

## (a) The top-priority claim: softmax vs `pivot_unsigned`

> **VERDICT: SOUND.** `max|softmax − pivot_unsigned| = 0.000000e+00`. Both row
> sets are genuinely row-stochastic (softmax rows 1.0 on every live row).

**Claim.** `workdonenew.md:146,188,282-284`, `CEQ_V15_CONTRACT.md:22` ("WHERE
WE ARE"), `README.md`: *"softmax and `pivot_unsigned` share a BIT-IDENTICAL
operator"*.

**What produces each side, read from the code, not the name.**
`scale/m3_capability.py:119-139`, `Arm._operator`:

```python
if self.kind == "softmax" or self.kind == "pivot_unsigned":
    return bench._softmax_operator(q, k)              # [n,s,s], batched
```

Both arm kinds are the same `if`-branch. There is no second construction to
compare against the first — `kind == "pivot_unsigned"` calls
`ceq.bench._softmax_operator`, the same softmax-normalized, causally-masked
function `kind == "softmax"` calls, on the same `q, k`. The claim is not an
empirical near-equality that could hide a scale factor; it is Python identity
of the call, checked by `tests/loop/test_m3_harness_operator_is_shipped.py`.
The only place the two arms differ is the separately-added `pivot_hop2` term
(`scale/pivot_probe.py`), which is outside the operator the claim is about.

**Runnable check** (`CHECK 1`, float64, `s=6`, causal, batched):

```
max|softmax - pivot_unsigned|      = 0.000000e+00
softmax row sums   (batch 0, all i) = [0.0, 1.0, 0.9999999999999999, 1.0, 1.0, 1.0000000000000002]
pivot_unsigned row sums (batch 0)   = [0.0, 1.0, 0.9999999999999999, 1.0, 1.0, 1.0000000000000002]
row-sum max|.-1| over LIVE rows i=1..5 = 2.220e-16 (row 0 is the documented dead row, sum 0.0)
correlation (causal entries)        = 1.000000
```

Row 0 sums to `0.0` by construction, not by defect: `ceq/bench.py`'s causal
mask is *strictly* causal (`.tril(-1)`, self-excluded), a deliberate,
separately-documented design (`scale/arm_s.py:g3_bind`, "Row 0 of a strictly
causal operator sums to exactly 0.0 ... which has no predecessors"); the
residual term supplies the self-contribution elsewhere
(`z = x + a @ x`, `m3_capability.py:145`). Every live row is row-stochastic to
float64 precision. **This is the load-bearing SOUND finding the pre-v13
census rests on, and it holds.**

---

## (b) CONFIRMED instances

### CONFIRMED #1 — the contract's own PARITY clause (the round's headline defect, independently reproduced)

**`CEQ_V15_CONTRACT.md:139-140`**, verbatim:

> **PARITY WITH SELF-ATTENTION is by IDENTITY BIND, not TOST.** `g == 0`
> gives bitwise standard attention (Lean #5).

**Which side of the boundary each object is actually on**, read from
`CEQ_V15_CONTRACT.md:82-83` (S-M) and `lean/CEQ/V15.lean:116-117,199`:

- `W g i j = exp(C_i − C_j)`, `C = scan(g)` (prefix sum) — **no denominator,
  no `sum(-1)`, no `softmax(...)`**. `Wc` causally masks it: `Wc g i j = W g
  i j` for `j ≤ i`, else `0`. At `g ≡ 0`, `C_i − C_j = 0` for all `i,j`, so
  every visible entry is `exp(0) = 1`. **Unnormalized side.**
- "standard attention" is `softmax(logits)`, `sum(-1, keepdim=True) == 1` by
  construction on every row that has a visible key. **Normalized side.**

Both sides have the *identical support* — `j ≤ i`, the causal lower triangle
— which is exactly the mask-matched-operator-assumed mechanism
(pattern #2): the two objects were declared identical because their nonzero
locations agree, not because their values do.

**Runnable check** (`CHECK 2`, float64, `s=6`, inclusive-diagonal causal
mask, matching `lean/CEQ/V15.lean`'s own `Wc` convention exactly rather than
`ceq/bench.py`'s self-excluding one, to avoid conflating two separate design
choices):

```
Wc(g==0) row sums (i=0..5)  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
expected i+1                    = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
softmax row sums (i=0..5)   = [1.0, 0.9999999999999999, 0.9999999999999999, 1.0, 1.0, 1.0]
max|Wc(g==0) - softmax(q,k)|    = 9.919145e-01
correlation (causal entries)    = nan (Wc's rows are literally constant, zero variance -- see below)
```

Smallest witness (matching `lean/CEQ/V15.lean`'s `gate_zero_not_stochastic`):
at `i = 1` the row is `(1, 1)`, summing to `2` — no softmax row of any head,
on any weights, ever sums to `2`.

**Verdict: CONFIRMED INSTANCE.** Already found and proved false by this
round's own Lean node (`gate_zero_row_sum`, `gate_zero_not_stochastic`,
`V15_N3_LEAN.md` DIAGNOSIS); reproduced here independently in plain
float64/torch, outside the Lean kernel, so the refutation does not rest on
one proof assistant. `workdonenew.md:64-78` has already absorbed this finding
and correctly names the mechanism ("The mask matching is what let the
identity claim stand unexamined") — the contract text at
`CEQ_V15_CONTRACT.md:139-140` itself is the one location that has **not**
been marked corrected as of this sweep.

### CONFIRMED #2 — the S-C conservation census, read as covering S-M

**`CEQ_V15_CONTRACT.md:125`**, verbatim (section S-C, "CONSERVATION CENSUS"):

> Carriers conserve mass to `1e-12` (softmax rows 1.000; replicator by Lean
> #8); readouts print their dissipation budget...

This is the same defect read from the conservation side (pattern #4). The
sentence opens with an unqualified claim about **"Carriers"** — plural,
unscoped — and PART I of the same contract lists S-M as one of the
architecture's carriers, describing it four sections earlier, in the same
document, as **"ONE unnormalized causal hop"** (`CEQ_V15_CONTRACT.md:82-83`).
The parenthetical gives two examples (softmax, replicator) and both really do
conserve mass — but neither example is S-M, and S-M is a carrier the census
sits directly downstream of. Read as a blanket claim over every carrier the
round composes, "Carriers conserve mass to `1e-12`" is false for S-M by the
identical arithmetic as CONFIRMED #1: at `g ≡ 0` its row sums to `i + 1`,
not `1`, and no witnessed value of `g` makes it row-sum to exactly `1`
either — the mask half of S-M's mechanism (Lean's `gate_zero_is_attention`)
supplies a `0/1` indicator, not a probability distribution, at any gate
setting, because §S-M's construction has no normalizing division anywhere in
it (`lean/CEQ/V15.lean:117`, `def W ... := Real.exp (scan g i - scan g j)`
— no `/`, no `softmax`).

**Verdict: CONFIRMED INSTANCE, if "Carriers" is read to include S-M** (the
natural reading, since S-M is the round's central carrier and the sentence
gives no scoping clause). **If the sentence is instead read narrowly as
scoped only to the two carriers it happens to name (softmax, replicator),
then it makes no claim about S-M and is not itself false** — but the census
then omits its own most consequential carrier's conservation status
entirely, which is a hole rather than an error, and either reading leaves the
same fact unstated where a reader would expect to find it: S-M does not
conserve mass, at any gate setting.

---

## (c) UNRESOLVED

### UNRESOLVED — S-C's "replicator by Lean #8" citation

The same sentence (`CEQ_V15_CONTRACT.md:125`) cites `replicator by Lean #8`
as an already-established fact. `#8` is `replicator_eq_cumsoftmax`
(`CEQ_V15_CONTRACT.md:157`, PART II, THE LEAN INVENTORY), status `[M]`
(must-fire, not yet done). It is **not** in `lean/CEQ/V15.lean` (which covers
only items #1, #2, #3, #5, #6, #7 — see that file's own header table) and
does not appear anywhere else under `lean/CEQ/`:

```
$ grep -rn replicator_eq_cumsoftmax lean/ *.md
CEQ_V15_CONTRACT.md:157:| 8 | `replicator_eq_cumsoftmax` | [M] |
V15_N1_PRIOR_ART.md:245:...  Lean #8 in particular. `replicator_eq_cumsoftmax`
```

**Verdict: UNRESOLVED — the code that would produce this side of the claim
does not exist yet.** This is not itself a normalization-boundary instance —
the mathematical content (`V15_N1_PRIOR_ART.md:236-247`) is a comparison
between two genuinely normalized objects (the EW/MWU iterate
`x_i = exp(y_i)/Σ_j exp(y_j)` and `softmax` of the cumulative payoff, both on
the simplex) and is independently reproduced below as SOUND. The defect here
is citation-provenance, not arithmetic: a theorem tagged `[M]` (not proved)
is cited in a **conservation census** as though `[V-eq]` (proved with a
numeric instance), the exact `[V]`-as-theorem failure mode `L-EQ` was written
to forbid (`CEQ_V15_CONTRACT.md:44-45`).

---

## (d) SOUND — checked and genuinely equal, reported as prominently as the hits

### SOUND #1 — softmax vs `pivot_unsigned` (see (a) above, the top-priority ask)

`max|diff| = 0.000000e+00`. Same function, same call, same inputs.

### SOUND #2 — the additive-logit repair, `gate_zero_logit_identity`

`lean/CEQ/V15.lean:251-254`: `q i j + (scan g i − scan g j) = q i j` at
`g ≡ 0`. Reproduced numerically (`CHECK 4`):

```
max|softmax(q) - softmax(q + (C_i-C_j)=0)| = 0.000000e+00
```

Under this reading the gate enters the logits **additively**, on top of
whatever logits the head already computes, rather than **replacing** them
multiplicatively. At `g ≡ 0` the logit is untouched, so both sides of the
comparison are softmax rows — there is no normalization boundary to cross,
and the identity is genuine. `V15_N3_LEAN.md` DIAGNOSIS already names this as
the repair; `prefix_logit_computes_chain` (the chain-label reproduction),
however, needs the *multiplicative* reading, so no single operator is yet
known to carry both — that is the round's own stated open question, not a
new finding of this sweep.

### SOUND #3 — `ceq/hybrid.py`, `alpha = 0` residual gate

`MODEL_CARD.md:235`: *"At α = 0 this IS stock attention, bitwise."*
`tests/w8/test_w8_real_model.py::test_alpha_zero_is_bitwise_stock_attention`.

`out = stock_attention(q,k,v) + Σ_{h=1..K} (alpha·A)^h v`. `ceq/hybrid.py:149`
and `:154` short-circuit to `return stock_attention(...)` directly when
`alpha == 0.0`, rather than computing a sum whose terms happen to vanish —
code-level identity, same shape as SOUND #1. This is the additive-residual
architecture (Check 2's repair pattern), not S-M's multiplicative-replacement
one. Reproduced (`CHECK 5`, `[2,4,6,4]` random q/k/v, float64):

```
max|ceq_hybrid_attention(alpha=0) - stock_attention| = 0.000000e+00
torch.equal (exact bitwise)                          = True
```

### SOUND #4 — the FM/EW replicator identity cited at S-C's "Lean #8"

Independent of the citation problem in (c) above, the underlying arithmetic
the `[M]` item would formalize is already numerically verified in this repo's
own probe, `scripts/v15_n1_probes/p05_mwu_replicator.py` (read-only rerun,
not modified):

```
MWU Thm 2.1  LHS=-5.335874  RHS=49.691038  holds=True
EW iterate vs softmax(cumulative payoff): 1.3322676295501878e-15
Fisher: dubar/dt=0.2700185304  Var(f)=0.2700185307  rel.err=1.11e-09
EW-vs-RD one-step error ratio delta/(delta/2) = 3.992 (expect ~4)
```

Both sides of "EW iterate vs softmax(cumulative payoff)" are normalized —
`x_i(t) = exp(y_i(t))/Σ_j exp(y_j(t))` is itself a softmax by definition, so
this is a same-side-of-the-boundary comparison, correctly identical to
`1.3e-15`. **The theorem just has not been formalized in Lean yet**
(see UNRESOLVED above); the mathematics it would state is sound.

### SOUND #5 (design, not a claim) — `scale/hilbert.py`'s "same support" usage

`scale/hilbert.py:19-24,67-70`: the Hilbert projective metric is *defined* to
read `0` for two vectors that are positive rescalings of one another
(`d_H(p,q) = 0` whenever `p ∥ q`, not only `p = q`) and is documented as
exactly that: "the metric lives on rays through the cone... a scale-sensitive
implementation reports a nonzero distance between two representations of the
same point." This is the deliberate, correctly-labeled use of a
scale-invariant object — the metric is never used to claim two *different*
carriers (one normalized, one not) are the same operator; it is used to
measure distance between representations that are already understood to be
equivalent up to scale. Not an instance of pattern #3; flagged here only
because the search terms ("same support") otherwise read like a hit.

### SOUND #6 — `scale/arm_s.py:g3_bind`, "with settling disabled the arm is BITWISE the glance"

Same code-identity shape as SOUND #1 and #3: `a = bench._softmax_operator(q,
kk)`, then compares `arm_s(..., t_max=0)` against `a @ v` via `torch.equal`.
The function also runs a second, explicit control for the row-0 dead-row NaN
hazard (dividing by a zero row sum) rather than silently excluding it — a
guard against exactly the "control that passes for the wrong reason" failure
this sweep was looking for elsewhere. README.md's parallel claim ("`glance`
bound bitwise to `softmax`", `README.md:323`) rests on the same code path.

---

## Summary

| # | verdict | one-line |
|---|---|---|
| a | **SOUND** | softmax and `pivot_unsigned` are the literal same function call (`m3_capability.py:131`), `max\|diff\| = 0.0`, load-bearing for the whole pre-v13 census |
| b-1 | **CONFIRMED** | `CEQ_V15_CONTRACT.md:139-140`'s "`g == 0` gives bitwise standard attention" is false for S-M's unnormalized hop: row `i` sums to `i+1`, not `1` (reproduces `lean/CEQ/V15.lean`'s own refutation independently in plain torch) |
| b-2 | **CONFIRMED** (reading-dependent) | S-C's "Carriers conserve mass to `1e-12`" (`CEQ_V15_CONTRACT.md:125`), read to include S-M (the contract's own carrier list), is false by the same arithmetic; read narrowly it is a silent omission rather than a false statement |
| c | **UNRESOLVED** | S-C cites "replicator by Lean #8" (`replicator_eq_cumsoftmax`) as settled; that theorem is `[M]`-status, not proved, and appears nowhere under `lean/CEQ/` |
| d-1..6 | **SOUND** | softmax/pivot_unsigned; the additive-logit repair; `ceq/hybrid.py` alpha=0; the FM/EW replicator arithmetic underlying Lean #8; `scale/hilbert.py`'s deliberate scale-invariance; `scale/arm_s.py`'s glance-bitwise-softmax bind |

**Counts.** CONFIRMED: **2** (one already known and independently
reproduced, one newly surfaced by this sweep in the S-C section).
UNRESOLVED: **1**. Checked and found SOUND: **6**, including the
single highest-priority claim named in the brief.

**Conservation claim about an unnormalized carrier (S-C):** yes — see
CONFIRMED #2. The census names softmax and the replicator as the carriers
that conserve mass; it does not name S-M, and S-M does not conserve mass at
any gate setting (`W` has no normalizing division anywhere in its
definition, `lean/CEQ/V15.lean:117`). Whether this is a false statement or an
omission turns on how "Carriers" is scoped, and the contract does not say.

---

## Files touched

- `V15_NORMALIZATION_SWEEP.md` — this file, new.
- `scripts/v15_norm_sweep_probe.py` — new, read-only against the repo
  (imports `ceq.bench`, `ceq.hybrid`; also separately ran, unmodified,
  `scripts/v15_n1_probes/p05_mwu_replicator.py`). No training, no fitting, no
  arms. `python scripts/v15_norm_sweep_probe.py` — exit 0, all assertions
  pass.

No other file was read for edit or modified. No git write command was run.
