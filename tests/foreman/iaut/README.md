# The confound escape: a generator handed over as a transition table

Phase H §8 measured an operator arm at `0.9570` against vectors at `0.2238` and
attributed 84.9% of the gap to non-commutative composition. That row carried a
confound it could not remove: arms (c) and (d) had the forward pass
`state = ops[verb] @ (state + emb[patient])`, character-for-character the bed's
own generative recursion. On that bed, *non-commutative operator* and *the
generator's exact operator family* are the same thing by construction.

This directory removes the confound by handing the generator over as a **finite
automaton presented only as integers**. The arms receive symbol ids and nothing
else — no permutation matrices, no transition table, no group structure. The
target is reachable in principle, because Krohn–Rhodes guarantees the automaton
decomposes into groups and resets at some dimension, but nothing tells an arm the
parameterisation. The operator arm must learn a representation of something it
was never given.

## Running it

```bash
python iaut_race.py
```

One command, no arguments, ~650 s. Writes one JSON line per (arm, seed) to
`iaut_results.jsonl` as each finishes; 25 rows for 5 arms × 5 seeds.

## The pre-registered kill, and that it did not fire

Written before the numbers: *if the operator arm does not beat the diagonal
control at matched parameters on a generator it was never handed, the 84.9%
attribution was the arm being the generator, and the last positive dies.*

It did not fire. Parameters are read from `numel()` at run time, not from a
config.

| arm | d | params | mean | std |
|---|---|---|---|---|
| a vectors + softmax | 10 | 460 | 0.2285 | 0.0176 |
| b scalar gate | 10 | 460 | 0.2305 | 0.0190 |
| **c operator, invertible** | 12 | **404** | **0.8620** | 0.0556 |
| d operator + projector | 10 | 398 | 0.6975 | 0.1034 |
| DIAG commuting control | 36 | **404** | 0.2860 | 0.0150 |

The control matches arm (c) at 404 parameters exactly. An independent re-run from
a clean directory returned `c = 0.8817 ± 0.0869` against `DIAG = 0.2803 ± 0.0104`,
with (c) beating the control on **5 of 5 seeds pairwise** and no overlap of the
spreads — worst case `min(c) = 0.7300` against `max(DIAG) = 0.2937`.

## The floor that matters is not the one the row shipped

The row's own floors — constant-majority and last-two-symbols, tightest `0.2250`
— are not the binding constraint on a commuting arm. A commuting gate can only
ever see the **multiset**, which for a length-14 binary word is the count of
symbol 1: fifteen possible values. That ceiling was computed directly on the
producer's own eval splits at **0.2898** fitted on train and **0.3110** as an
oracle fit on eval itself.

DIAG scored `0.2803` — within `0.010` of the honest version and `0.031` of the
absolute upper bound. **The control is saturated, not undertrained**, so the
separation is not a training artefact. Against the real tightest floor of
`0.3110`, the operator arm's *worst* seed is still `2.35×`.

## That no arm saw a matrix, checked rather than asserted

An AST scan of `VectorArm` and `OperatorArm` finds zero references to `delta`,
`ALL_S5`, `INDEX_OF`, `compose`, `ALPHABET`, `IDENTITY`, `run_word` or
`label_of`; the only free globals either class touches are `B` and `C_CLASSES`.
A runtime wrap of every arm's forward confirms the sole input is `torch.int64` of
shape `(B, 14)`, min 0, max 1. The delta table is built once in
`build_delta_table()` and used only to produce labels.

The null — shuffled symbol order, refolded through the table, scored on the class
label rather than on finiteness — was sound at every seed before any arm ran:
`160 / 158 / 164 / 154 / 164` wrong out of 200, against a requirement of ≥ 1.

## What this row does not establish

One parameter budget (400) and one word length (14). No grid across budgets or
lengths, and no length-generalisation check — train and eval share word length,
matching the precedent it replaces. "Invertible" for arm (c) means no projector
composed and generically full-rank at initialisation, verified post-hoc by a
nonzero determinant logged per run, not enforced during training.

Arms (a) and (b) land at `0.23`, which is **below** the `0.3110` multiset ceiling.
They are underfit against a baseline the row never measured, so the vector side's
score is a floor on their performance rather than a measurement of it. That does
not affect the operator-versus-control comparison, which is the row's question.

Two earlier cuts were caught and discarded rather than shipped: at `L=12` the
floor pool asked for more unique words than `2^12` allows, an infinite rejection
loop caught by a smoke test; at `L=20, steps=400` every arm's train loss sat near
the untrained `ln(8) = 2.079`, so the kill would have fired on underfitting rather
than on the question.
