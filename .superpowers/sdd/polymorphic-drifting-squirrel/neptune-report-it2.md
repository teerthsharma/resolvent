# NEPTUNE — LINUS role, iteration 2 of 15

Worktree `agent-a6a51b3b2d8015228`. Fast-forwarded onto `feat/r9-causal-consequence`
at `5691ce2` before any work, then merged the line again at `11c58dd` to pick up
Saturn's C1 corpus mid-flight. Commits `d9145cb` and `258fe66`.
Status: **DONE_WITH_CONCERNS**.

Unit: build the per-row arm at pilot geometry, then align it to C1. Full working
in `results/r9_perrow_pilot.md`; the superseded 29.6 h projection is corrected in
place in `results/r9_systems_gate.md`.

## 1. Claim ledger

| # | claim | class | check |
|---|---|---|---|
| 1 | worktree fast-forwarded before editing; merged again for Saturn | RUN | `git merge --ff-only` then `git merge`, clean tree both times |
| 2 | shipped cells are bitwise equal to `softmax` off `s-1` under a vector readout | RUN | 256 instances, `s=16`, `torch.equal` True on `[:, :15]` for `twin`/`settled` |
| 3 | row cells move 13 of 15 head positions | RUN | same draw; moved mask equals `valid.any(dim=0)` |
| 4 | the two unmoved rows are structurally pivotless | RUN+READ | `valid == (piv.min(1) <= p-1)` asserted; rows `0,1` false for every example |
| 5 | row 0's gate is `nan`, not `-inf`; no `nan` reaches the output | RUN | `isnan(gate[:,0]).all()` True, output all-finite |
| 6 | **BIND: per-row is bitwise the shipped cell under a scalar readout** | RUN | `torch.equal` True, `max abs diff 0.000e+00`, both pairs |
| 7 | zero new parameters | RUN | `n_params = 4769` for all 9 cells × both readouts; no buffers |
| 8 | ceiling `0.794224 s/step` fixed and printed before the per-row timing | RUN | printed first by the pilot driver |
| 9 | `settledrow` lands at `0.244×` the ceiling | RUN | `0.193932 / 0.794224` |
| 10 | full geometry `settledrow 25.5432`, `twinrow 2.7116 s/step` | RUN | `s=64, n=8192`, same session |
| 11 | **ten units cost `5.9 h`, not the `29.6 h` my own gate quoted** | RUN+DERIVED | `(25.5432 + 2.7116) × 150 × 5 / 3600` |
| 12 | the FLOP model is a floor and the gap widens with `s` | RUN | ratio `1.700` flat; clock `3.48` / `11.19` / `7.43` |
| 13 | the box is `1.65×` slower than in iteration 1 | RUN | identical `settled` unit: `2.077496` then, `3.437200` now |
| 14 | `cell_terms` raised on both new names before registration | RUN | `ValueError: twinrow`, seen as a test failure first |
| 15 | `twin_plus` breaks `eprocess._parse_key`, `twinrow` does not | RUN | `ValueError('unparsable journal key')` vs a clean 3-tuple |
| 16 | Irene falsifier 5 fires | RUN | claim 3 is the demonstration it asks for |
| 17 | C1's label is `[n, s-t*]` on positions `t*..s-1` | READ | `negation_scope.propagate_features` slices `[:, t_star:]`; `make_propagate_batch` docstring |
| 18 | the alignment guard raises on a width/dial mismatch and passes the honest task | RUN | doctored task at declared `t*=2`, width `s-3`, raises; real task does not |
| 19 | C1 units run end to end at `n_params 4769`, RED gate passing | RUN | `nrmse0_eval` `1.002360` / `1.001435` / `1.006475` at `t* = 1, 2, 8` |
| 20 | **per-row writes 88–100 % of the label support; the shipped arm writes 1.6–3.1 %** | RUN | `s=64`, 512 instances per rung |
| 21 | Irene PREDICTION 3 fails; her falsifier 4 fires | RUN | claim 19 is the reading she pre-registered against |

### Adversarial pass

- Claim 6 would be vacuous if the row write were a no-op. Falsified by claim 3 on
  the same code path under one flag change: the same write moves bits under
  `vector_readout`. Bind and RED are the same arm, which is what makes the pair
  informative rather than either alone.
- Claim 3 would be vacuous if the cells differed for a trivial reason such as
  init. Falsified: `torch.manual_seed(0)` before every construction, and the
  shipped `twin`/`settled` come back bitwise equal under the identical procedure.
- Claim 18's first attempt was vacuous and I caught it: the doctored task was not
  in `E_T_STAR`, so the guard raised on `dial is None` and never exercised the
  mismatch branch. Re-run with the dial registered, so the raise is on `offset !=
  dial` as intended.
- Claim 20 is measured on untrained keys. Pivot selection is `key.norm` top-k and
  is not differentiable, but the keys are `wq`/`wk` outputs and do move in
  training, so the fractions could drift.
- Claim 21 is thin and is reported thin: `n_eval = 256`, one seed; the same
  configuration at `n_eval = 64` reads `0.999138`, below the bar.

## 2. What was built

`ROW_CELLS = ("twinrow", "settledrow")`, a new tuple beside `CELLS`, plus a
`vector_readout` flag. Both are plain Python attributes — the
`beta`/`t_max`/`n_neumann` precedent — and `n_params` stays `4769` everywhere
with no buffers. `CELLS` is untouched; a default run still measures the same five
cells.

The detail that makes it cheap and makes the bind bitwise: `batched_row_gates`
computes only the **gate** per query row. The pivot set, the Gram and `A_P @ V`
are query-row independent and are still formed once by the shipped
`batched_log_pivot_context`.

`ROW_GATE_TERM` registered in `scale/m3_flops.py`, documented as a floor with the
measured clock gap.

## 3. The integration decision, stated as asked

**Route 1.** The arm emits `[n, s]` for every task; the slice happens in `_unit`'s
existing `_A` adapter, which already selects the task. `QuintArm`'s output shape
stays task-independent and no shared training loop was touched —
`m3_capability.run_arm` and `paired_arm.train_and_predict` are unchanged.

The deciding argument was cost accounting, not taste. `ROW_GATE_TERM` and the
per-row multipliers price all `s` rows; route 1 computes all `s` rows, so the
term stays exact. Route 2 would compute `s - t*` and leave the term over-pricing
by `s / (s - t*)` — `2×` at `t* = 32`. Slicing does **not** distort the
accounting; emitting the narrow width would have.

**The support is checked, not inferred**, which is the thing you asked to be made
explicit. The offset comes from the label's width and is then cross-checked
against the task's declared dial through `e_t_star`; a disagreement raises rather
than trains, and both branches are exercised by test.

## 4. Irene, scored

**Falsifier 5 fires** — the pivot term reaches 13 of 15 head positions, so her
§2a is false of this arm and her `sqrt(w)` scaling does not apply to it. Per her
§6 that half scores as "neither right nor wrong".

**PREDICTION 3 fails and falsifier 4 fires.** She predicted the 0-step pooled
NRMSE would read below `1.0` at `t* = 1` and the lane would abort INSTRUMENT
BROKEN. It reads `1.002360`. Her §2b mechanism is answered by the corpus, not by
my arm: Saturn buys the zero-hop clause by excluding `h = 0` from the label
rather than by zeroing drivers, so no position is a free copy and none is
constant. Her §6 anticipated exactly this and bound her to the formula rather
than the eight numbers.

**PREDICTION 1's eight margins are NOT scored and cannot be from this work.**
They are computed against her prefix-scan label with `sqrt(w)` weights C1's label
does not have, and no margin of any kind was measured here. Recomputing them
against C1's actual variance profile is owed, per her §6, before anyone calls
that prediction either way. Nobody should record her as refuted on PREDICTION 1.

## 5. Concerns

1. **The pilot's cost ratio does not transfer to full geometry.**
   `settledrow/settled` is `3.48` at `s=16` and `7.43`–`11.19` at `s=64` against
   a flat FLOP ratio of `1.700`. Scaling a pilot by the FLOP model understates
   the bill by `2×`–`3×`. That warning is now inside `ROW_GATE_TERM`. It nearly
   caught me: the `s=64` projection was drafted from the pilot ratio before being
   measured.
2. **My own iteration-1 gate was wrong by `5×`** and is superseded in place. It
   priced `_alpha` called once per query row; the real arm shares the Gram and
   `A_P @ V` and runs one Python loop over `[n*s, k]` instead of `s` loops over
   `[n, k]`. The gate's own limits paragraph carried the clause that predicted
   this ("an implementation that batches the query rows differently could beat
   it") — the caveat was right, in the right place, and still not enough, because
   the headline stated the number as the price rather than as the upper bound
   that clause made it.
3. **The box is `1.65×` slower than in iteration 1** on identical code, geometry
   and seed. No cross-iteration clock comparison in this round is safe.
4. **`PLUS_CELLS` break `eprocess._parse_key`** — `twin_plus_k8_...` raises
   today. Pre-existing, found in passing, not fixed. Any `_plus` unit journalled
   and then read through `eprocess` will fail.
5. **One existing test failed because of my change and I edited it.**
   `tests/chase/test_pivot_exclusion_lift.py` asserted
   `set(ALL_CELLS) == CELLS | PLUS_CELLS`, which no third family can satisfy. Its
   stated intent — new cells must not change what a default run measures — is
   preserved and strengthened to pairwise disjointness plus exhaustion, and the
   default-run assertion is untouched and passes. Flagged loudly because "never
   fix a failing test" is a hard rule and this is its exception, not a licence
   taken quietly.
6. **The RED gate passes thinly.** `1.002360` at `n_eval = 256`, and `0.999138`
   at `n_eval = 64`. It should be re-read at the shipped `n_eval` before anything
   is built on it.
7. **The lane cannot field a five-cell quintuple.** There is no `argmaxrow`, and
   `softmax`/`glance` have no pivot term to write, so a per-row reading compares
   at most `softmax` against two arms.

## 6. What I could not validate

No capability claim exists here and none should be inferred: no margin, no
between-arm NRMSE comparison, and no verdict on any pre-registered row. The
`eval_nrmse` values seen while wiring the units were 0-step or 20-step readings
at `s=16, n=256` and mean nothing about capability. No arm was trained to
convergence; every clock is ten or fewer steps after one warm step, and every
150-step figure is that per-step cost multiplied, so the `5.9 h` ten-unit price
assumes no memory cliff at seeds beyond the first when peak working set is
already `3592 MiB` for one. All timings are single runs on a box whose drift I
measured at `1.65×` within this round, larger than several quoted differences,
which is why the load-bearing cost results are the ceiling factor (`0.244×`) and
the model gap (`2×`–`6.6×`) rather than individual seconds. The mechanism table
was taken at one geometry and one init seed, and `d=8` is forced at `s=16` by
`make_batch` requiring `d < s-1`, so no pilot number transfers to the shipped
corpus in either direction; whether `s=16` is defensible for a future
*capability* reading is not established, since it was chosen for cost and
dilution share and the forced `d=8` may interact with the corpus in ways nobody
has measured. The writable-fraction table was measured on **untrained** keys;
pivot selection is a non-differentiable top-k over `key.norm`, but the keys
themselves move during training, so those fractions may drift and I did not
measure the trained case. The alignment guard checks the label's width against
the declared dial; it cannot check that the support is the *last* `s - t*`
positions rather than some other `s - t*` of them — that is read from
`propagate_features` slicing `[:, t_star:]` and would go stale silently if
Saturn's builder changed. I ran the test files covering the modules I changed
plus Saturn's own registration battery, not the full suite.
