# R9 systems gate — the vector-valued readout lane, priced before it is built

Role: LINUS (systems). Standard: kernel/dispatch count, memory layout and
wall-clock honesty against the analytic FLOP model. Every number below is either
`RUN` (executed this session, command given) or `READ` (`path` plus the quoted
text). Nothing here is an estimate unless it says `DERIVED`, and the derivations
show their steps.

Machine: the shared box. `torch 2.5.1+cu121`, device `cpu`,
`torch.set_num_threads(2)` — pinned inside `scale/m3_quintuple.py`, not by the
launcher. **Every wall-clock figure is PROVISIONAL** in the sense this repository
already uses for `meta.seconds`: other agents share this machine and a contended
timing is not a measurement. The FLOP counts are arithmetic on shapes and cannot
be moved by contention, so the argument is carried on those and the clock is
shown as corroboration and as a *lower* bound on cost.

Geometry throughout: `s = 64`, `d = 24`, `d_model = 16`, `hidden = 128`,
`k_piv = 8`, `t_max = 21`, `n_neumann = 21`, `n_params = 4769` on every arm.

---

## 1. What the readout actually is

`scale/m3_quintuple.py:311`:

    return self.readout(h).squeeze(-1)[:, s - 1]

`h` is `[n, s, d_model]`, `self.readout` is `nn.Linear(d_model, 1)`, so
`readout(h).squeeze(-1)` is already `[n, s]` and the index selects one column.
Dropping the index yields `[n, s]` for zero new parameters — that much of the
proposal (FINDINGS C1–C5) is correct and was verified: `n_params` reads `4769`
in both variants of every benchmark below, so hard rule 3 is not at risk.

**But the index is not where the information is lost.** `QuintArm.forward` writes
the pivot correction into exactly one row (`scale/m3_quintuple.py:307-309`):

    if self.base_cell not in ("softmax", "glance"):
        z = z.clone()
        z[:, s - 1] = x[:, s - 1] + self._alpha(q, k, x).to(z.dtype)

and the class docstring (`scale/m3_quintuple.py:262-265`) states this as the
design intent in as many words: *"Only row s-1 can reach the output ... so only
that row is replaced. The other rows are left exactly as the shipped forward
produced them."* The MLP and the readout are position-wise, so no other row can
be touched by any cell.

### 1.1 The measurement that decides the lane

Claim, in the TDD form: *for all drawn instances, and for every cell in
`{twin, settled, argmax}`, the vector readout at positions `0 .. s-2` is
bitwise equal to the `softmax` cell's, and only position `s-1` differs.*

`RUN`, 256 drawn instances, `make_batch(256, 64, 24, d_model=16, seed=0)`,
identical parameter init (`torch.manual_seed(0)`) so the cells differ only in the
`alpha` rule:

| cell | `max abs diff` at positions `0..s-2` | bitwise equal there | `max abs diff` at `s-1` |
|---|---|---|---|
| `twin` | `0.000e+00` | **True** | `0.003448` |
| `settled` | `0.000e+00` | **True** | `0.005143` |
| `argmax` | `0.000e+00` | **True** | `0.011783` |

**63 of the 64 output coordinates carry zero information about which cell is
running.** Two independent paths, failing differently, agree: the code read above
(only `z[:, s-1]` is written; MLP and readout are position-wise) and the numeric
bitwise comparison on drawn data.

### 1.2 Consequence for the loss

`scale/paired_arm.py:74` is the whole training step:

    torch.nn.functional.mse_loss(model(x_train), ystd).backward()

full batch, `reduction='mean'` (the default). Today the reduction is over `n`
terms and every one of them is the settled row. Under an `[n, s]` readout it is
over `n * s` terms, of which `n` are the settled row:

    settled row's share of the loss = 1/s = 1/64 = 1.5625 %

so **98.4375 % of the gradient would flow through a term that is bitwise
identical for all five cells.** A control whose output is 98.4 % constant across
the arms it is meant to separate is vacuous by the repository's own rule 4
(the PASS half's label must be non-degenerate). The measured `settled − twin`
contrast is already `−0.002959`, CI `[−0.042903, +0.031557]`; diluting it by
`1/64` puts the point estimate near `−4.6e-05` `DERIVED`, which no bootstrap at
five seeds can resolve — `capability_table.LIMITS` clause (c) already records
that a real gap below roughly `0.05` NRMSE reads NO DIFFERENCE here.

**Gate finding 1: dropping `[:, s-1]` alone does not open the vector-valued
regime. It dilutes the only signal the arm has by 64× and buys nothing.** The
binding constraint is the single-row write at `:309`, not the readout index.

---

## 2. Cost of the naive change (index dropped, forward otherwise unchanged)

`RUN`. One configuration per process so the peak-working-set reading is not
polluted by a previous configuration's allocator high-water mark;
`psutil.Process().memory_info()`, Windows `peak_wset`. Warm step excluded from
the clock. `sec/step` is forward + `mse_loss` + `backward` + `Adam.step`.

| cell | `n_train` | readout | out numel | sec/step | ×150 steps | peak wset (MiB) |
|---|---|---|---|---|---|---|
| `softmax` | 2048 | scalar | 2,048 | `0.185946` | `27.89` | `888.54` |
| `softmax` | 2048 | vector | 131,072 | `0.176806` | `26.52` | `837.34` |
| `twin` | 2048 | scalar | 2,048 | `0.315290` | `47.29` | `927.54` |
| `twin` | 2048 | vector | 131,072 | `0.280349` | `42.05` | `926.54` |
| `settled` | 2048 | scalar | 2,048 | `0.443522` | `66.53` | `1027.30` |
| `settled` | 2048 | vector | 131,072 | `0.423430` | `63.51` | `1019.26` |
| `softmax` | 8192 | scalar | 8,192 | `0.984537` | `147.68` | `1624.79` |
| `softmax` | 8192 | vector | 524,288 | `0.901435` | `135.22` | `1628.73` |
| `settled` | 8192 | scalar | 8,192 | `2.077496` | `311.62` | `2439.75` |
| `settled` | 8192 | vector | 524,288 | `2.061487` | `309.22` | `2449.18` |

**Wall clock: the vector readout is free, and in five of five pairs it measured
marginally FASTER** (`−0.77 %`, `−4.53 %`, `−4.92 %`, `−8.44 %`, `−11.08 %`).
That is not a paradox: the index is a gather whose backward is a scatter into a
zero tensor of the full `[n, s]` shape, so the sliced version does strictly more
allocator work than the unsliced one. But the differences sit inside what a
contended box can move, so the honest statement is **no measurable cost**, not a
speedup. `twin` at `n_train = 8192` was not run — the pairs are the five in the
table.

**Memory: also free.** At `n_train = 8192` the peak working set moves
`+3.94 MiB` for `softmax` (`+0.24 %`) and `+9.42 MiB` for `settled` (`+0.39 %`);
at `n_train = 2048` it moved *down* in both cells. Analytic check, independent of
the measurement: the readout output grows from `[n]` to `[n, s]`, and the target
must grow with it, so the added residency is `2 × n × s × 4` bytes
`= 2 × 8192 × 64 × 4 = 4.00 MiB` `DERIVED` — which brackets both measured
deltas. The high-water mark is set elsewhere and by a much larger object: the
attention operator `a` is `[n, s, s]` float32, `8192 × 64 × 64 × 4 =
128.0 MiB` per tensor, and the autograd graph holds several. **The readout shape
is three orders of magnitude away from being the memory question.**

---

## 3. Cost of the change that would actually work

The only non-vacuous version of the lane gives every query row its own pivot
settle, so `_alpha` returns `[n, s, d]` instead of `[n, d]`
(`scale/m3_quintuple.py:300` returns `[n, d]` today). Priced term by term against
`scale/m3_flops.py`, at `n = 8192`:

* `select` — pivots come from `kk` alone (`arm_s.pivots_of`), shared by every
  query row. **Unchanged.**
* `setup` — inside `arm_s.log_pivot_context` the logit slice forms `k + 1` rows
  (`k` pivots plus the one query row). Per-row over all `s` queries it forms
  `k + s`. The Gram and `A_P@V` terms depend on pivots only and are shared.
  `2*(k+1)*s*dm → 2*(k+s)*s*dm`: **×4.00**, `3.5232e+08 → 1.4093e+09`.
* `loop` — `log_alpha_step` iterates from the query row's own gate. **×64**,
  `2.2020e+07 → 1.4093e+09`.
* `contract` — `alpha @ AV`, per row. **×64**, `2.0972e+06 → 1.3422e+08`.
* `bwd` — the Neumann VJPs, per row. **×64**, `4.1943e+07 → 2.6844e+09`.

| quantity | current `settled` | per-row `settled` | ratio |
|---|---|---|---|
| total FLOPs, `n = 8192` | `7.4313e+09` | `1.2650e+10` | **`1.702×`** |
| against `softmax` base `6.9961e+09` | `1.062×` | `1.808×` | |
| non-base share of the arm | `5.856 %` | `44.695 %` | |

### 3.1 The FLOP model under-predicts this arm by 2×, and the gap grows

This is the number the round most needs and the one the FLOP model will not give
you. `RUN`, `n_train = 8192`:

    measured wall-clock ratio settled/softmax = 2.077496 / 0.984537 = 2.110
    analytic FLOP ratio      settled/softmax = 7.4313e9 / 6.9961e9  = 1.062
    the model is optimistic by a factor of                            1.987

The reason is dispatch, not arithmetic. The base terms are three large BLAS
calls; the settle is a Python-level `for` over `t_max` steps in
`arm_s.settle_log` plus a second Python loop of `n_neumann - 1` autograd VJPs in
`arm_s.Settled.backward`. Those loops are 5.9 % of the FLOPs and 53 % of the
clock. **A per-row settle multiplies the iteration count of exactly those two
Python loops by `s = 64`** — the ×64 lands entirely on the dispatch-bound half,
so the realised slowdown will be worse than `1.702×`, not better.

### 3.2 Wall clock, measured two ways

Path A, `RUN`, difference of measured steps at `n_train = 8192`:
one `_alpha` call, forward and backward, costs
`2.077496 − 0.984537 = 1.092959 s/step`. Sixty-four of them: `69.95 s/step`.

Path B, `RUN`, direct timing of `_alpha` at `n = 512`, forward only:
`0.036700 s` for one call; eight calls took `0.301400 s`, i.e. `0.037675 s` each,
**linearity `1.026`** — repeated settles do not amortise, so the ×64 is a real
multiplier and not an extrapolation across a regime change. Scaling `n` by 16 to
8192 gives `0.587 s` forward-only, against path A's `1.093 s` forward *and*
backward. The two paths measure different halves and their ratio (`0.54`) is what
a 21-term Neumann backward should cost relative to its forward. They agree.

**Price of the lane, `DERIVED` from path A:**

    per step   ≈ 0.985 (base) + 69.95 (64 settles) = 70.9 s
    per unit   ≈ 70.9 × 150 steps = 10,640 s ≈ 2.96 h
    settled+twin × 5 seeds = 10 units ≈ 29.6 h

against the current cost of the same ten units, `311.62 s × 10 = 0.87 h`.

> ### SUPERSEDED — this projection was wrong by 5×, and the arm now exists
>
> **`29.6 h` priced an implementation nobody wrote, and it must not be quoted
> again.** It assumed the per-row arm calls `_alpha` once per query row, so
> every term was multiplied by `s`. The arm built in iteration 2
> (`ROW_CELLS` in `scale/m3_quintuple.py`) does not: the pivot set, the Gram
> and `A_P @ V` are query-row independent and are formed once, and the settle
> runs as **one** Python loop of `t_max` iterations over an `[n*s, k]` tensor
> rather than `s` loops over `[n, k]`. The Python iteration count therefore
> does not scale with `s` at all, which is where the factor went.
>
> Measured at this section's own geometry (`s = 64`, `n_train = 8192`), same
> session: `settledrow 25.5432 s/step` and `twinrow 2.7116 s/step`, so ten
> units price at **`5.9 h`, not `29.6 h`** — `5.32 h` plus `0.56 h`. Memory,
> not time, is what the lane actually spends: peak working set `3592 MiB`
> against `2440 MiB` for the scalar `settled` arm.
>
> The reasoning in §3.1 survives and is reinforced — the FLOP model is still
> optimistic, by `2.0×` at pilot geometry and `4.4×` to `6.6×` here — but the
> **magnitude** of §3.2's bill was a derivation from an assumed implementation
> where a measurement was affordable, and it should have been labelled as the
> naive upper bound it was rather than as the price of the lane. Full working:
> `results/r9_perrow_pilot.md`.

---

## 4. Registration — the new lane dies at accounting time unless this is done

`scale/m3_flops.py:110` `cell_terms` ends at `:139` with

    raise ValueError(cell)

`RUN`: `cell_terms('vector_twin', 8192, 64, 16, 128, 8, 21, 21)` raises
`ValueError('vector_twin')`. **Any new cell name must be registered there or the
accounting path dies the first time it is asked to price a unit.** (The finding
sheet cites this as `:103-131`; at `74e5590` it is `:110-139`.)

Registration is not a one-line addition:

* **The readout change needs no new term.** `BASE_TERMS`'s `readout(h)` entry is
  already `2*n*s*d_model` — it prices all `s` rows, because all `s` rows are
  built today and only then is one indexed out. At `n = 8192` that term is
  `1.6777e+07` FLOPs, **`0.2398 %` of base**. The FLOP model is already correct
  for a vector readout and needs no edit for it.
* **The per-row settle DOES need its own terms.** No existing term expresses the
  `×s` on `loop`, `contract` and `bwd`, or the `(k+1) → (k+s)` on `setup`. A new
  cell registered by reusing `SETUP_TERM`/`LOOP_TERM` would under-report its own
  cost by `1.702 / 1.062 = 1.60×` against `softmax` and would be a false entry in
  the one table this project uses to argue cost.
* Beyond `m3_flops`, the arm name is enumerated in the places FINDINGS §E lists,
  and `CELLS` at `scale/m3_quintuple.py:86` is asserted as an exact 5-tuple by
  `tests/chase/test_pivot_exclusion_lift.py:139`. The precedent is `PLUS_CELLS` —
  a new tuple beside `CELLS`, not a name added to it.
* A cell name containing `_sd`, `_task`, `_k` or `_b` breaks the journal key
  parsers. `vector_twin` is safe; `twin_vec_k` would not be.

---

## 5. Correction to the finding sheet's stated reason for a separate lane

FINDINGS says changing `forward` *"voids the `PUBLISHED_SOFTMAX_8192`
reproduction gate at `scale/m3_quintuple.py:696-708`."* **The conclusion is right
and the mechanism is wrong**, and the wrong mechanism would let someone talk
themselves out of the lane separation.

`READ`: the gate at `scale/m3_quintuple.py:688-708` calls
`M3.run_arm("softmax", ...)`, and `run_arm` (`scale/m3_capability.py:174-177`)
constructs `Arm(kind, s)` — `m3_capability.Arm`, **not** `QuintArm`. Editing
`QuintArm.forward` therefore does not touch that gate at all. What it does touch:

1. `m3_capability.Arm.forward:166-167` carries its own
   `out = self.readout(h).squeeze(-1)` then `return out[:, s - 1]`. **That** is
   the forward the reproduction gate exercises, and changing it voids the gate.
2. Every one of the 100 rows in `results/m3_quintuple_v2.jsonl` was produced by
   the current `QuintArm.forward`. The journal key encodes cell, `k`, geometry,
   seed and task (`m3_quintuple._key`, parsed by `task_of`) — **it does not
   encode the readout shape.** A changed `QuintArm.forward` writes rows whose
   keys collide with existing rows while meaning something different. That is the
   same failure mode as the `_task` suffix incident, which took down 14 of 16
   tests in `tests/chase/test_capability_table.py` and is documented in
   `task_of`'s own docstring.

**So the separate lane (own journal, own weights directory) is required, for
reason 2 rather than reason 1.** Reason 1 additionally forbids touching
`m3_capability.Arm`.

---

## 6. Verdict

| # | question asked | answer |
|---|---|---|
| 1 | cost of `[n]` → `[n, s]` in the loss | reduction goes from `n` to `n*s` terms; the settled row's share falls to `1/64 = 1.5625 %` |
| 2 | cost in memory high-water | `+3.94 MiB` (`softmax`) / `+9.42 MiB` (`settled`) at `n=8192`, `≤0.39 %`; analytic `4.00 MiB`. **Not a memory question** |
| 3 | cost in wall clock | none measurable; faster in 5 of 5 pairs (`−0.77 %` to `−11.08 %`), differences inside contention |
| 4 | does `cell_terms()` still describe it | **yes for the readout** — `readout(h)` already prices all `s` rows. **No for the per-row settle**, which needs new terms |
| 5 | must a new lane be registered in `m3_flops` | **yes** — `cell_terms` raises `ValueError(cell)` at `:139`, confirmed by execution |

**The gate does not pass the lane as proposed.** The naive change — drop the
index, keep the forward — is free in every resource this gate measures, and that
is exactly the problem: it is free because it computes nothing new. It would ship
a `[n, 64]` output that is bitwise cell-independent in 63 coordinates and hand
`1/64` of the gradient to the one coordinate that is not.

**The gate passes the per-row variant on cost, at a stated price:** `1.702×` the
FLOPs of the current `settled`, and a realised slowdown worse than that because
the extra work falls on the Python loop that is already 53 % of the clock against
5.9 % of the FLOPs. **Ten units cost `5.9 h` against `0.87 h` today** — measured
in iteration 2 on the built arm, superseding the `29.6 h` this section first
carried; see the SUPERSEDED note in §3.2. If that bill is not affordable, the
honest move is to say so now rather than to build the cheap version and read its
zero.

---

## 7. What this gate did NOT validate

The wall-clock figures are single measurements on a contended shared box, 5 or 10
steps each after one warm step, and no repeat was taken — a second run could move
any of them by more than the differences the table reports between scalar and
vector, which is precisely why no speedup is claimed from them. The
`≈29.6 h` price is `DERIVED` by multiplying one measured `_alpha` cost by 64; the
linearity check that licenses the multiplication was run only to eight calls and
only at `n = 512`, forward-only, so it does not exclude a cache-residency cliff
somewhere between 8 and 64 repetitions at `n = 8192`. The per-row FLOP
decomposition in §3 is arithmetic on shapes for a function **that does not
exist** — no per-row `_alpha` was written or run, so the `(k+1) → (k+s)` setup
claim rests on reading `arm_s.log_pivot_context` and reasoning about what a
per-row version would form, not on measuring one; an implementation that batches
the query rows differently could beat it. **That last caveat fired.** The arm
built in iteration 2 batches exactly that way and came in `5×` under the price
this section quoted, which is the SUPERSEDED note in §3.2. The caveat was
correct, was written in the right place, and was still not enough: a derivation
carrying a live "an implementation could beat this" clause is an upper bound and
should have been labelled one in §3.2's own table rather than only in this
paragraph. The `1/64` loss-share figure assumes
`reduction='mean'` over the full `[n, s]` tensor, which is what
`scale/paired_arm.py:74` does today but is not forced — a per-position weighting
or a loss taken only at `s-1` would change it, and neither was priced here. The
bitwise-identity result in §1.1 was measured at one geometry (`s=64`, `d=24`,
`d_model=16`, `k_piv=8`) and one seed for the parameter init; it follows from the
code structure and should hold everywhere, but "should" is the word. Nothing here
measures whether a per-row settle would *help* — this is a cost gate only, and no
accuracy claim of any kind is made or implied.
