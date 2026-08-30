# R9 per-row arm — PILOT. Cost and mechanism only, no capability claim.

Role: LINUS (systems), iteration 2. Ruling being executed: *the per-row arm gets
built and measured at REDUCED geometry before any full-geometry commitment.*

**Nothing here is a capability measurement.** No NRMSE is reported, no arm is
compared to another on a task, and no row of any pre-registered table is
evaluated. The vector-valued label does not exist in this repository yet — it is
a separate landing — so the only honest readings available are the arm's
mechanism and its price. Those are what follows.

Machine: shared box, `torch 2.5.1+cu121`, `cpu`, `torch.set_num_threads(2)`.
Every clock is PROVISIONAL, and this round produced the cleanest demonstration
yet of why: the identical `settled` unit at `s=64, n=8192` read `2.077496 s/step`
in iteration 1 and `3.437200 s/step` in this one — **1.65× apart on identical
code, geometry and seed.** Only same-session ratios are quoted below.

---

## 1. The pilot geometry, and its arithmetic before the first number

`s = 16, d = 8, d_model = 16, hidden = 128, k_piv = 8, n_train = 512,
steps = 150, seed = 0`.

* **`s` is shrunk, not just `n_train`, and it pays twice.** The dilution share of
  the one architecturally distinguishable position is `1/s`: `0.015625` at the
  shipped `s = 64`, `0.062500` at `s = 16` — **4× more signal per position** —
  while the per-row work falls with `s` at the same time.
* **`d = 8`, not the shipped `24`, and this is forced.**
  `negation_scope.make_batch` raises unless `1 <= d < s - 1`, so `d = 24` cannot
  be drawn at `s = 16`. **The pilot is therefore not comparable to the shipped
  ladder in either direction** and no number here may be set beside one from
  `results/e_ladder_reading.txt`.
* `k_piv = 8` is preserved: `batched_pivots` takes `min(k_piv, s - 3)`, which is
  `8` at `s = 16` and would silently drop to `5` at `s = 8`. Changing `k` would
  change the arm, so `s = 16` is the smallest geometry that leaves it alone.

### The ceiling, fixed and printed before the per-row arm was timed

    ceiling = softmax_step + s * (settled_step - softmax_step)

A per-row cell cannot honestly cost more than `s` copies of the single-row
cell's pivot work, because that is the implementation it replaces — and it
shares the Gram and `A_P @ V` across rows, which this bound does not. Landing
**above** it would mean the implementation is worse than the naive thing.

| quantity | value |
|---|---|
| `softmax` step | `0.006437 s` |
| `settled` step | `0.055674 s` |
| one `_alpha`, `settled − softmax` | `0.049237 s` |
| **ceiling, per step** | **`0.794224 s`** |
| **ceiling, ×150 steps** | **`119.13 s`** |

---

## 2. Mechanism — the measurement that decides the lane

Drawn instances: 256, `make_batch(256, 16, 8, d_model=16, seed=0)`, identical
parameter init across cells so only the alpha rule differs.

| cell | readout | bitwise equal to `softmax` at `0..s-2` | head rows moved |
|---|---|---|---|
| `twin` | vector | **yes** | 0 / 15 |
| `settled` | vector | **yes** | 0 / 15 |
| `twinrow` | vector | no | **13 / 15** |
| `settledrow` | vector | no | **13 / 15** |

The two head rows that do not move are rows `0` and `1`, and they are
structurally pivotless rather than missed: `ceq/bench.py` masks with `tril(-1)`
so query row `p` reads `j <= p-1`, while `batched_pivots` selects from
`1 .. s-2`. No pivot can be `<= 0`, so those rows have every gate entry masked
and `log_softmax` returns `nan`, not `-inf`. They are excluded through a mask
and left at the shipped `x + a@x` value rather than written.

Validity is **per example**, not a fixed prefix: row `p` is usable exactly when
the smallest *selected* pivot is `<= p-1`, and the pivots are content-selected.
`tests/neptune/test_per_row_arm.py` asserts that exact relation against an
independently computed `piv.min(dim=1) <= p-1` — an earlier version of that test
asserted "rows 2 and up are always valid", which is false, and the test caught
it.

### The bind, and it is bitwise

Under the scalar readout the output depends only on `z[:, s-1]`, so a per-row
cell must reproduce its single-row cell exactly:

| pair | `torch.equal` | `max abs diff` |
|---|---|---|
| `twinrow` vs `twin` | **True** | `0.000e+00` |
| `settledrow` vs `settled` | **True** | `0.000e+00` |

This is bitwise by construction rather than by tolerance, and it holds because
the shared `batched_log_pivot_context` still supplies the Gram and `A_P @ V`
while the row gate reproduces the `s-1` row of the same log-softmax.

**The RED that makes the bind non-vacuous** is the table above: the same
comparison under `vector_readout` must and does move bits for the row cells and
must not and does not for the shipped cells. A bind that only ever saw the
correct path could not tell the two apart.

---

## 3. Cost

### Pilot geometry, `s=16, n_train=512`

| cell | readout | sec/step | ×150 steps | vs ceiling |
|---|---|---|---|---|
| `softmax` | scalar | `0.006437` | `0.97 s` | |
| `twin` | scalar | `0.028719` | `4.31 s` | |
| `settled` | scalar | `0.055674` | `8.35 s` | |
| `twinrow` | vector | `0.035364` | `5.30 s` | `0.045×` |
| `settledrow` | vector | `0.193932` | **`29.09 s`** | **`0.244×`** |

**The ceiling holds with a factor of 4.1 to spare.** `n_params = 4769` on every
cell above and on both readouts, checked per cell; no buffers.

### Full geometry, measured rather than projected

| cell | `s` | `n_train` | sec/step | ×150 steps | peak wset |
|---|---|---|---|---|---|
| `settled` | 64 | 8192 | `3.437200` | `515.6 s` | |
| `settledrow` | 64 | 8192 | `25.543200` | **`3831.5 s` = `1.06 h`** | `3592 MiB` |
| `twinrow` | 64 | 8192 | `2.711600` | `406.7 s` = `0.113 h` | |
| `settledrow` | 64 | 2048 | `4.560600` | `684.1 s` | `1426 MiB` |
| `twinrow` | 64 | 2048 | `0.666200` | `99.9 s` | |

**This corrects my own iteration-1 gate, and by a large factor.** That gate
priced the per-row lane at `≈29.6 h` for ten units, from `64 × 1.092959 s` — the
cost of calling `_alpha` sixty-four times. The shipped implementation does not
do that: the pivot set, the Gram and `A_P @ V` are query-row independent and are
formed once, and the settle runs as **one** Python loop of `t_max` iterations
over an `[n*s, k]` tensor rather than `s` loops over `[n, k]`. Ten units
(`settledrow` + `twinrow`, five seeds) price at **`5.9 h`, not `29.6 h`** —
`5.32 h` of `settledrow` plus `0.56 h` of `twinrow`, both from measured
per-step costs at the shipped geometry, on a box currently running 1.65× slow.
**The earlier number was an over-estimate of a naive implementation and should
not be quoted again.**

### The FLOP model, scored a second time

`ROW_GATE_TERM` is registered in `scale/m3_flops.py` and `cell_terms` now prices
both row cells. It is a **floor**, and the gap grows with `s`:

| geometry | FLOP ratio `settledrow/settled` | clock ratio | model optimistic by |
|---|---|---|---|
| `s=16, n=512` | `1.700` | `3.48` | `2.05×` |
| `s=64, n=2048` | `1.700` | `11.19` | `6.58×` |
| `s=64, n=8192` | `1.700` | `7.43` | `4.37×` |

The `1.5×` spread between the two `s=64` rows is contention, not `n` — see the
`1.65×` drift on identical code above. **A pilot cost measured at small `s` must
not be scaled to full geometry by this term**; that is now written into the
term's own comment, because it is exactly the mistake the pilot was authorised
to prevent and it would have understated the bill by 2–3×.

Memory is the one resource the per-row arm genuinely spends: `3592 MiB` peak
against `2440 MiB` for the scalar `settled` at the same geometry. The `268 MB`
float64 Gram copy is named with its upgrade path in a `ponytail:` comment at the
call site; it fits, so it is left alone.

---

## 4. `R9_IRENE_PREDICTION.md`, scored where this unit touches it

**Falsifier 5 FIRES.** It asks for *"a demonstration that the pivot term reaches
a position other than `s-1` in the R9 arm"*. Section 2 is that demonstration:
13 of 15 head positions move, and the two that do not are pivotless by
causality. Irene's §2a — *"at every position `p ≠ s-1`, `settled`, `twin`,
`argmax` and `softmax` evaluate the same expression"* — is true of the shipped
arm, is re-measured true here, and **is false of the row cells.** The `sqrt(w)`
scaling in her §2c therefore does not apply to this arm.

**The arm does not settle her stronger half; the corpus does.** Her §2b is an
argument about the **label**, not the arm: under the prefix-scan reading she
chose, `s - t* - 1` positions would have `a_p = 0` so `z_p = b_p` exactly, an
input channel readable at zero hops by every arm identically. A per-row arm
changes nothing about that. But her §6 anticipated exactly this and bound her to
the formula rather than the eight numbers: *"if R9 defines the vector label some
other way ... §2c's variance shares are wrong."* It does. Saturn's C1 buys the
zero-hop clause by excluding `h = 0` from the label instead of zeroing drivers,
so no position is a free copy and no position's label is constant — see §5.

Per her §6 this file scores as *"neither right nor wrong"* on PREDICTION 1's
mechanism half and says so, as she required. **PREDICTION 3 is scored in §5 and
fails** — her falsifier 4 fires, thinly. PREDICTION 1's eight margin numbers are
**not** scored here and cannot be: they are computed against her prefix-scan
label with `sqrt(w)` weights that C1's label does not have, and no margin of any
kind was measured in this pilot. Recomputing them against C1's actual variance
profile, per her §6, is owed before that prediction is called either way.

---

## 5. Integration with Saturn's C1 corpus — route 1, and why

C1's label is `[n, s - t*]`, covering positions `t* .. s-1`: a strictly causal
operator raised to `t*` vanishes on the first `t*` coordinates, so labelling
them would ship `t*` entries of `sd 0`. `negation_scope.propagate_features`
slices `[:, t_star:]`, which fixes the support as the **last** `s - t*`
positions. `run_arm` raised `size of tensor a (64) must match tensor b (56)`.

**Route 1 was taken: the arm emits `[n, s]` and the slice happens outside it.**

* `QuintArm.forward` under `vector_readout` returns `[n, s]` for every task. Its
  output shape does not depend on a corpus's difficulty dial.
* **The FLOP accounting stays exactly right.** `ROW_GATE_TERM` and the `×s`
  multipliers price all `s` rows, and under route 1 all `s` rows are computed.
  Route 2 would have computed only `s - t*` of them and left the term
  over-pricing by `s / (s - t*)` — `2×` at `t* = 32`. That is the reason to
  prefer route 1 beyond taste: it is the one that keeps the cost model honest.
* The slice lives in `_unit`'s existing `_A` adapter, which already selects the
  task, so no shared training loop was touched — `m3_capability.run_arm` and
  `paired_arm.train_and_predict` are unchanged.

**The support is checked, not inferred.** The offset is derived from the label's
width and then cross-checked against the task's own dial via
`negation_scope.e_t_star`; a disagreement raises rather than trains. Both
branches are exercised: a doctored task declaring `t* = 2` while emitting width
`s - 3` raises `label width 13 implies offset 3 at s=16, but task declares
t*=2`, and the honest task passes the same guard.

End-to-end at `s=16, d=8, n=256`, through `_unit`, all cells at `n_params 4769`:

| task | `nrmse0_train` | `nrmse0_eval` | RED gate `>= 1.0` |
|---|---|---|---|
| `c1_propagate_t1` | `1.004224` | `1.002360` | **passes** |
| `c1_propagate_t2` | `1.004672` | `1.001435` | **passes** |
| `c1_propagate_t8` | `1.003886` | `1.006475` | **passes** |

**This is where Irene's PREDICTION 3 is scored, and it fails.** She predicted
the 0-step pooled NRMSE would read below `1.0` at `t* = 1` and the vector lane
would abort `INSTRUMENT BROKEN`, because a vector label re-opens the zero-hop
legibility hole at `s - t* - 1` positions. It does not, on this corpus: Saturn
bought the zero-hop clause by excluding `h = 0` from the label rather than by
zeroing drivers, so no position's label is a free copy and no position's label
is constant. Her §2b — the argument I said survived falsifier 5 — is answered by
the corpus, not by the arm. **Her falsifier 4 fires.**

Read that reading narrowly. It is `n_eval = 256` at one seed and one geometry,
and the margin is thin: the same configuration at `n_eval = 64` read
`nrmse0_eval 0.999138`, below the bar. The gate is passed, not passed
comfortably, and it should be re-read at the shipped `n_eval` before anything is
built on it.

### How much of the label the arm can actually write

The shipped arm writes exactly one position of the support, so its share is
`1 / (s - t*)`. The per-row arm writes every position with a causally visible
pivot. Measured at `s = 64`, 512 drawn instances per rung:

| rung | label width | **shipped arm's share** | **per-row writable fraction** | positions never writable |
|---|---|---|---|---|
| `c1_propagate_t1` | 63 | `1.587 %` | **`88.11 %`** | 1 (row 1, pivotless) |
| `c1_propagate_t2` | 62 | `1.613 %` | **`89.53 %`** | 0 |
| `c1_propagate_t8` | 56 | `1.786 %` | **`95.22 %`** | 0 |
| `c1_propagate_t32` | 32 | `3.125 %` | **`99.99 %`** | 0 |

The shortfall from `100 %` is causality, not a defect: row `p` needs a selected
pivot at `<= p-1`, the pivots are content-selected from `1 .. s-2`, and early
rows can miss. It shrinks as `t*` grows because the support starts later. **Both
columns belong beside any margin this lane ever reports** — that is the
obligation row Ω of `R9_IRENE_PREDICTION.md` creates, and the shipped arm's
column is the reason the lane needed a new arm at all.

## 6. Registration status

* `ROW_CELLS = ("twinrow", "settledrow")`, a **new tuple beside** `CELLS`.
  `CELLS` is untouched and still the exact 5-tuple
  `tests/chase/test_pivot_exclusion_lift.py` asserts.
* Names carry **no internal underscore**, and that is load-bearing rather than
  cosmetic. `eprocess._parse_key` reads the cell as everything before the first
  underscore and then requires the next field to start with `k`; `twin_row`
  would be parsed as cell `twin` with `k_field = "row"` and **raise**. The
  existing `PLUS_CELLS` do fail this, which is a pre-existing defect found in
  passing and not fixed here. `twinrow` cannot. `_sd`, `_task`, `_k` and `_b`
  are all absent from both names, and a round-trip through `_key`,
  `eprocess._parse_key` and `task_of` is asserted in the test file.
* `scale/m3_flops.py` prices both cells; an unregistered name still raises
  `ValueError(cell)`, asserted.
* A default run is unchanged: `_argparser().parse_args([]).cells == list(CELLS)`,
  asserted.
* **Zero new parameters.** `vector_readout` and the per-row selection are plain
  Python attributes, the `beta` / `t_max` / `n_neumann` precedent. Every cell,
  both readouts: `n_params = 4769`, no buffers.

---

## 7. What this pilot does NOT establish

No capability claim of any kind is made or implied: no vector-valued label
exists in this repository, so nothing here says the per-row arm is better than
softmax, or than the shipped arm, at anything. The arm has never been trained to
convergence — every timing is 10 or fewer steps after one warm step, and 150-step
figures are that per-step cost multiplied, not a completed unit. All clocks come
from a single run each on a contended shared box, and the `1.65×` drift measured
on identical code between iterations is larger than several of the differences
quoted, which is why the ceiling comparison (`0.244×`, a factor of four) and the
FLOP-model gap (`2×` to `6.6×`) are stated as the load-bearing results while the
individual seconds are not. The `≈6 h` ten-unit projection multiplies a measured
per-step cost by 150 and by 5 seeds; it assumes no memory cliff at seeds beyond
the first, and peak working set was already `3592 MiB` for one. The mechanism
table was taken at one geometry (`s=16, d=8`) and one init seed, and the
bitwise-identity half of it was independently reproduced at `s=64` in iteration 1,
but the per-row half was not. `d = 8` differs from the shipped `d = 24` by
necessity, so nothing here transfers to the ladder's own corpus. Finally, the
row cells exist for `twin` and `settled` only — there is no `argmaxrow`, and
`glance`/`softmax` have no pivot term to write — so the vector lane cannot
currently field a five-cell quintuple, and whether it needs to is a question for
whoever designs its pre-registration.
