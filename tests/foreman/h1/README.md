# The H.1 producer chain

Every number in `docs/PHASE_H.md` §8 is produced here. Until this directory
existed, those numbers lived in exactly one file — the prose page — with no bed,
no test and no artifact behind them, so a reader could not re-run the one
measurement the phase rests on. That is the defect this directory closes; the
scripts themselves are unchanged from the runs that produced the published
figures.

## Running it

The chain is ordered. Each stage loads the previous one by path relative to its
own `__file__`, so the directory relocates without edits.

```bash
python phaseh_I1_I2_binding.py     # bed primitives: entities, verbs, operators
python phaseh_H3v2_bed.py          # the order-demanding bed
python phaseh_H3v2_floors.py       # the five floors and the null
python phaseh_H3v2_twins_ab.py     # arms (a) vectors+softmax, (b) scalar gate
python phaseh_H3v2_twins_cd.py     # arms (c) operator, (d) operator+projector
python h1_checker_probe.py         # the commuting-diagonal control
```

`phaseh_H3v2_floors.py` gates the rest: it prints `null_discriminates=True` and
the twins may proceed only on that line. Each stage appends to
`house-events.jsonl` at the repository root if that file is present, and creates
it otherwise; nothing downstream reads it.

## Which producer makes which published number

| number | what it is | producer |
|---|---|---|
| `0.2238` ± 0.0115, 620 params | arm (a), vectors + softmax + RoPE | `phaseh_H3v2_twins_ab.py` |
| `0.2628` ± 0.0148, 620 params | arm (b), scalar gate | `phaseh_H3v2_twins_ab.py` |
| `0.9570` ± 0.0048, 552 params | arm (c), operator gate, invertible | `phaseh_H3v2_twins_cd.py` |
| `0.9470` ± 0.0059, 654 params | arm (d), operator gate + projector | `phaseh_H3v2_twins_cd.py` |
| `0.3566` ± 0.0230 | tightest floor, `last_two_ops`, on this split | `phaseh_H3v2_floors.py` |
| `0.1640` | majority-label baseline, the only oracle-free floor | `phaseh_H3v2_floors.py` |
| `0.3080` @400, `0.3342` ± 0.0161 @4000, 554 params | the commuting-diagonal control | `h1_checker_probe.py` |
| `0.7332`, `59 σ`, `84.9%` | derived from the four arms and the control | arithmetic over the above |

Parameter counts are read from `numel()` at run time rather than from a config,
which is why the control lands at 554 against arm (c)'s 552 — a live sweep over
`d` found the tightest available match, and it is a tighter one than the 620 the
(a)/(b) lane could reach.

## What the chain does not settle

The arms' forward pass for (c) and (d) is `state = ops[verb] @ (state + emb[patient])`,
character-for-character the bed's own generative recursion. So `0.9570` measures
the generator's form fitting its own generator, and on this bed
*non-commutative operator* is not separable from *the generator's exact operator
family*. The floors are oracle-assisted — `last_op_only`, `last_two_ops`, the
order-free replay and the bag predictor are all handed the true entities and the
true operators — so clearing them is harder than "trivial" would imply, and only
`constant_predictor` is oracle-free.

The row that would settle the confound supplies the generator as a transition
table instead of as matrices, so that no arm is handed the parameterisation. It
is a separate bed and it is not in this directory.
