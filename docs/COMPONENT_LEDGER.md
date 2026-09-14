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
| **P1** | Encoder | aggregate context causally; clear a target a position-wise map provably cannot express, with the shipped `Encoder` shown failing it and a permutation control that moves the new one and leaves the shipped one bitwise identical | OPEN |
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
