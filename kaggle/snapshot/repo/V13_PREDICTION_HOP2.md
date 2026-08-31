# Pre-registered prediction — the hop-2 term does not work

Filed 2026-08-31 while `pivot_unsigned` at `t*=8, n=32768, seeds 0-7,
threads=12` is still running and before any of its cells have been read. The
journal is `results/r10_v13pilot_capacity_pivot_unsigned_t8.jsonl` and carried
only its header, ceiling and bar rows at the time of writing.

## The structural claim

`scale/m3_capability.py:119-165`. `Arm._operator` returns
`bench._softmax_operator(q, k)` for **both** `softmax` and `pivot_unsigned` —
the same operator, not merely a similar one. `Arm.forward` then computes
`z = x + a @ x` for every arm and adds, for every arm except `softmax`, a second
term `hop2 = batched_pivot_hop2(a, batched_select_pivots(k, self.k_pivots))`
followed by `z = z + hop2 @ x`.

So `pivot_unsigned` is exactly `softmax` plus a pivot-routed second hop, at
`K_PIVOTS = 8` against `s = 64`. It is architecturally 2-hop, and the floor it
can reach is `floor_2 = √((t*−2)/t*)`, not `floor_1`.

## The evidence already in hand

At `t*=2`, `floor_2 = √(0/2) = 0.000000`: a working 2-hop arm can read the label
exactly. The measured N=8 cell at `t*=2, n=2048, threads=6` reads
`0.956525` (sd `0.013133`), which is `0.956525` above its own floor and
`0.004176` **worse** than the `0.952349` of the 1-hop softmax it is built from.
Its `ĥ = t*(1 − NRMSE²)` is `0.170` against softmax's `0.186`.

An added term that leaves the reading unchanged to within a seed sd, in the one
place where a working version of it would drive the error to zero, is not
underpowered. It is inert.

## The prediction

The running cell will land near the 1-hop softmax reading and nowhere near its
own 2-hop floor:

- **P1.** Mean `eval_nrmse` in `[0.960, 0.990]`, i.e. within about `±0.015` of
  softmax's `0.975371` at the same cell, thread count and seed set.
- **P2.** The N=8 seed CI will **not** clear `floor_1 = 0.935414`, so
  `cap_verdict` returns "consistent with 1 hop" and `proven_hops = 0`.
- **P3.** The reading will sit at least `0.09` above its own
  `floor_2 = 0.866025`.
- **P4.** `ĥ` will be below `1.0` — the arm will not demonstrate even one full
  hop, let alone the two it is built to have.

## What each outcome means

If P1–P4 hold, the empty C-CAP census is explained by mechanism rather than by
budget, and no increase in `n` will produce a crossing from this arm. The
correct next move is then to fix or replace the hop-2 construction, not to buy
more data for it.

If the cell instead clears `floor_1`, this prediction is wrong, the `t*=2`
result is a task-specific artefact, and the hop-2 term works where the label
actually needs more than one hop.

If the cell lands **above** `1.0` it is a NO READING and adjudicates nothing;
`n=32768` would then be below this arm's learnability threshold at `t*=8`, and
the comparison must move to a cell where it clears the bar.
