# Component ledger: every piece to its own bar before anything trains jointly

Opened at commit `b4c6620`. The method is component-wise validation first and
joint training last. Nothing trains jointly until every row below reads PASS or
carries an explicit, measured waiver saying why the joint run proceeds without
it.

The reason this ledger exists rather than a plan: this project trained a full
model four times before anyone measured whether its encoder aggregated context.
It does not. Every result produced before `b4c6620` used an encoder that is
`x_t -> s_t`, whose docstring claimed `x_{<=t} -> s_t`, and whose scaling
exponent in the token count is therefore zero by construction. That single
unmeasured component is what retired the per-coordinate corner rule
(`docs/CORNER_RULE_RETIREMENT.md`). A joint run cannot tell you which of its
parts is the broken one; only a bar per part can.

## Rules for a row to read PASS

1. **Its own bar, stated before the run.** The bar names a target the component
   is supposed to reach and a baseline it must beat.
2. **A planted negative that fails the bar.** If the shipped component, or a
   trivial one, also clears the bar, the bar measures nothing and the row stays
   OPEN.
3. **Matched parameter counts**, printed beside every arm. An unmatched win is a
   tie.
4. **RED first.** The test fails before the component exists, verbatim failure
   recorded.
5. **A producer for every number**, per `L-PROSE` in `MISTAKES.md`, with the
   report's bound-over-reported ratio on its first line.

## The pieces

| id | component | bar it must clear | status |
|---|---|---|---|
| **P1** | Encoder | aggregate context causally; clear a target a position-wise map provably cannot express, with the shipped `Encoder` shown failing it and a permutation control that moves the new one and leaves the shipped one bitwise identical | **PASS** |
| **P2** | Read / operator | matched head-to-head against a plain softmax attention head at equal parameter count: does containment cost anything when both are trained? | OPEN |
| **P3** | Head / probe | find the binding ceiling among probe rank, read width and encoder; a linear probe from 7 read columns into 64 classes realises at most rank-7 logits, so every negative to date may be a statement about 7 rather than about the operator | OPEN |
| **P4** | Causal machinery | one synthetic-bed causal claim, measured on the 13,388 human decisions carrying an exact distance-to-mate — real positions supply an exact ground truth for "admitted futures" that the planted bed never had | OPEN |

## The joint gate

Joint training starts when P1 through P4 read PASS, or when a row carries a
waiver stating the measured reason it is being carried forward broken and what
that costs the joint result. A waiver is a number and a sentence, not a
judgement call.

Two things the joint run must carry regardless, both learned the hard way:

- **A frozen-random arm**, identical to the trained arm except that its encoder
  receives no gradient. Without it a joint result cannot separate scale from
  luck. On the synthetic bed the frozen arm *beat* the trained one; on 115,628
  real games the trained arm won, 0.5998 nats against 0.2348 with a uniform
  baseline of `2*ln(64) = 8.3178`. Only the control made that difference
  readable.
- **A trivial baseline on the real target.** Every arm of the T4 run lost to
  "guess the most common square" at every context length — 0.0823, 0.0256,
  0.0123 for the trained arm at L = 16, 64, 128 against 0.0942, 0.0398, 0.0247.
  A joint run without that column beside it can report a win that is not one.

## Limits

Statuses are updated only from a report that carries its producer ratio. A row
moved to PASS on prose is a row that is still OPEN.

## P1 — PASS, 2026-09-14

`ceqjepa/causal_encoder.py`, bound by `tests/curvature/test_causal_encoder.py`
(RED first: `ModuleNotFoundError: No module named 'ceqjepa.causal_encoder'`;
now 12 passed in 227.37 s). Reported with a producer ratio of 24/24.

The bar is a nine-channel strict-prefix EMA bank over an i.i.d. Gaussian bed.
A position-wise map cannot clear it, and not as an optimisation accident: the
sum runs over `j < t` strictly and the bed is i.i.d., so `y_t` is independent of
`x_t`, `E[y_t | x_t] = 0`, and the MMSE of *any* measurable function of `x_t`
alone equals `Var(y_t)`. The ceiling is `R² <= 0` by information, not by
training. A plain prefix mean was rejected as the bar because it is
permutation-invariant, so an encoder clearing it perfectly *should* be
order-blind — the bar and the permutation control would have disagreed.

| arm | params | R² eval | R² train | MSE | MSE const |
|---|---:|---:|---:|---:|---:|
| shipped `Encoder` (planted negative) | 1769 | **−0.1936** | 0.1543 | 1.1147 | 0.9339 |
| shipped `Encoder`, 3.2× wider | 5577 | **−0.6086** | 0.4027 | 1.5023 | 0.9339 |
| `CausalEncoder`, span = self (null) | 1657 | −0.0544 | 0.0446 | 0.9847 | 0.9339 |
| `CausalEncoder` | 1657 | **+0.9863** | 0.9872 | 0.0128 | 0.9339 |

The shipped encoder scores *below the constant predictor*, and widening it 3.2×
makes it strictly worse, which is what the argument predicts. The winner carries
112 fewer parameters than the arm it beats. Three seeds, non-overlapping:
shipped −0.1936 / −0.1940 / −0.2040 against causal +0.9863 / +0.9890 / +0.9896.

Permutation control — varies the order of context positions 0..14, pins the
multiset, `x_15`, weights, seed, batch and dtype: shipped `bitwise_identical =
True`, `max_abs_change = 0.000000e+00`; `CausalEncoder` `False`,
`4.647257e+00`. Causality is asserted rather than assumed: perturbing `x_k`
moves every position `t < k` by exactly `0.000000e+00` in all four arms.

The self-span ablation is what isolates aggregation from everything else —
identical 1657 parameters, initialisation asserted parameter-by-parameter with
`torch.equal`, identical bed, optimiser, lr, steps and loss, varying only which
positions a query may see. The `1.041` R² gap is attributable to aggregation
alone, not to depth and not to positional encoding.

**Adoption is deliberately not done yet.** Swapping the class touches
`pi_jepa.py:844-845`, and `:787-788` is a hard break: the collapse check reaches
*through* the encoder and iterates `.net`, so `CausalEncoder` exposes `.net` as
an `nn.ModuleList` for exactly that reason. `:1307-1311` states "Encoder is a
position-wise map", which becomes false and must be rewritten rather than
deleted — it is the premise the corner-rule retirement rests on.

Limits: `S = 16` only, so this says nothing about length generalisation; the bed
is synthetic i.i.d. by choice, because i.i.d. is what makes the position-wise
ceiling a theorem rather than a measurement; and the encoder's own
representational exponent was not measured, which is a different claim from
tracking nine timescales.
