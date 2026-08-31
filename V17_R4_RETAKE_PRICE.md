# V17-K RULING 4 — THE RE-TAKE, PRICED

**Scope: pricing only. No cell was re-taken and nothing below is a research
reading.** Three probe runs of `scripts/v15_r1.py` were made to time the
instrument; the instrument's `ROOT` was redirected to a scratch directory in
every one, so **no record was written into `results/v15_r1.jsonl`** (L-G2 — the
journal's superseded records are not this node's to write, and neither are its
replacements). Raw probe output: `results/r4_price_probe.json`.

| | |
|---|---|
| `git rev-parse HEAD` at start | `ab5b48547884e04258276e6e808d5a71ea65f917` |
| `git rev-parse HEAD` at end | `ab5b48547884e04258276e6e808d5a71ea65f917` |
| `git status --porcelain` delta over the run | **one line added: `?? results/r4_price_probe.json`** (plus `?? V17_R4_RETAKE_PRICE.md`, this file). `results/v15_r1.jsonl` does not appear in either listing. `M scripts/v15_r1.py` is another node's edit, present at start and untouched |
| writing git commands run | **none** |
| box | NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 B, `torch 2.5.1+cu121`, `torch.set_num_threads(8)` `[MEASURED]` |

---

## THE ANSWER, FIRST

**The full re-take is MINUTES, not hours: 2.65 GPU-min for all 24 cells, one
invocation** `[MEASURED + PROJECTED]`. The ruling's `~2 GPU-h` over-books it by
**45×**. The estimate is not wrong about the *arm* — it is right about a
different **ladder**: at 9600 steps the same 24 cells cost **2.71 GPU-h**, which
is almost exactly the author's figure. The journal holds 150-step cells, a
9600-step cell cannot be diffed against a 150-step one, and so the re-take that
Ruling 4 actually orders is the cheap one.

**The CPU↔CUDA delta is measured and it is small.** On the arm the re-take
exists for, `|Δ| eval_nrmse ≤ 3.462e-05`; across all six paired cells the worst
is `4.993e-03`, on the **softmax control**, not on either arm. Nothing here
widens anything — Ruling 4 refused widening a tolerance to absorb an
*unmeasured* delta, and the word "unmeasured" is what this file removes.

---

## 1. THE CELL LIST RULING 4 REQUIRES RE-TAKEN

Ruling 4 names two things: **"R1′"** and **"every deciding cell"**. They are
different sets and only one of them is currently in the journal.

### 1.1 Already on the journal as `device:cpu` — MUST be re-taken

`results/v15_r1.jsonl`, header `device: "cpu"`, 16 `t="cell"` records, all at
`task=e3_t2`, `s=64`, `d=24`, `d_model=16`, `n_train=2048`, `n_eval=4096`,
`steps=150`, `threads=8` `[MEASURED by read of the journal]`.

| cell id | arm | seeds | count | device today |
|---|---|---|---|---|
| `arm_pl:t2:n2048:seed{0..7}` | `ceq/arm_pl.py` | `0,1,2,3,4,5,6,7` | **8** | `cpu` |
| `softmax:t2:n2048:seed{0..7}` | `scale/m3_capability.Arm("softmax")` | `0,1,2,3,4,5,6,7` | **8** | `cpu` |

These 16 are the file `kaggle/ceq_v17k.ipynb` cell 14 points `LOCAL_Q1Q2_JSONL`
at, and the cell's own comment already flags the hazard: *"CONFIRM this is the
CERTIFIED CUDA reading, not a stale cpu-tagged file"*. Until they are re-taken,
Q1/Q2's `|Δ|` column differences a Kaggle CUDA cell against a local **CPU** cell
and reports a device delta as a reproduction delta.

### 1.2 Not taken on ANY device yet — R1′ itself

`CEQ_V16_CONTRACT.md` PART IV: *"R1′ THE RE-RUN. BED-M, `t*=2`, `n=2048`, phase
gates, same seeds."* The phase-gate arm that runs is `ceq/arm_smprime.py`
(`V16_ARM_SMPRIME.md`; `arm_phase`'s `operator` and `scan_phase` raise under
strict mode where `arm_smprime`'s do not).

| cell id | arm | seeds | count | device today |
|---|---|---|---|---|
| `arm_smprime:t2:n2048:seed{0..7}` | `ceq/arm_smprime.py` | `0,1,2,3,4,5,6,7` | **8** | *none — never run* |

R1′'s softmax control is **the same 8 cells as §1.1's softmax row**, taken on the
certified device. `V16_PRICING.md` §3.1 prices R1′ at 16 cells for exactly this
reason (*"the softmax control cannot be inherited from R1. R1 is CPU-taken"*);
because §1.1's softmax re-take is on CUDA too, the control is shared and the
union is **24 cells, not 32**.

### 1.3 THE LIST, as one command

```
python scripts/v15_r1.py --device cuda \
  --arms arm_pl arm_smprime softmax \
  --seeds 0 1 2 3 4 5 6 7 --tag <new_tag>
```

**24 cells = 3 arms × 8 seeds.** One process: one bind check per gated arm, one
bar calibration, one eval draw, 24 trained cells, one aggregate. `--n-train`,
`--n-eval` and `--steps` are already the required defaults (`2048 / 4096 / 150`).
`--tag` must **not** be `v15_r1`: the runner opens the journal in append mode, so
re-using the tag would interleave CUDA rows into the superseded CPU file rather
than leaving it intact as L-G2 requires.

### 1.4 ALREADY ON CUDA — no re-take needed

| artifact | what it is | why it is exempt |
|---|---|---|
| `results/m3_quintuple_v2_cuda.jsonl` | Q4's local capability-table reference, 23 records | already a CUDA journal; `V17_NOTEBOOK.md` row 12 diffs Kaggle's Q4 against it |
| `results/k_cert_local.json` | the device certificate itself | `box.device = "cuda"`, taken on this box at HEAD `ab5b485` |

`results/v15_r1.jsonl` is the **only** file in `results/` carrying
`"device": "cpu"` records `[MEASURED — `grep -l` over `results/*.jsonl`]`.
`results/r2.jsonl` carries no `device` field and is not a deciding-cell journal
(it is the older `key`/`meta`/`value` t-gate coherence probe).

### 1.5 CANNOT be re-taken by the adopted instrument — two blockers, both real

**Q2 (`t*=8`, `n=16,384`) cannot be produced by `scripts/v15_r1.py` at all.**

1. **`T_STAR = 2` is a module constant** (`scripts/v15_r1.py:124`), used to build
   `task = f"e3_t{T_STAR}"`, `floor_1`, the live band and every cell id. There is
   **no `--t-star` flag** — the runner's argparse offers `--seeds --n-train
   --n-eval --steps --threads --arms --trace-every --device --tag` and nothing
   else `[MEASURED by read]`. Ruling 5 adopted this file as *the* named
   instrument; as adopted, it emits `t*=2` cells only.
2. **`n = 16,384` does not fit this card under a training loop.** `COSTS.md`
   §1.3 `[INHERITED, measured on this box at this HEAD]`: `arm_smprime` at
   `n = 16384` allocates 6.296 GiB and **reserves 10.578 GiB under a training
   loop** against 7.996 GiB total — the driver pages over PCIe. That is also why
   `COSTS.md` §1.1 fits the arm's law over `n = 2048, 4096, 8192` only, so **no
   law in this repo extrapolates to Q2's `n`** either.

Both are findings for the author, not tasks this node discharged. **Q2 is not
priced below**, because pricing a cell that neither runs nor fits would be a
number with nothing under it.

---

## 2. THE ONE MEASURED CELL

**Cell timed:** `arm_smprime:t2:n2048:seed0` — the R1′ arm, the most expensive of
the three and therefore the one that decides the total. Timed **end to end** as a
whole process, twice, on the certified 4060.

| quantity | value | tag |
|---|---|---|
| **process wall clock, one cell end to end** | **20.021 s** | `[MEASURED]` |
| of which instrument import + `instrument_manifest` | 2.102 s | `[MEASURED]` |
| of which the cell's own training loop (`t="cell"` `secs`) | **14.390 s** | `[MEASURED]` |
| of which fixed setup (identity + label binds, 4 planted negatives, `calibrate_bar` at 600 steps, eval draw) | ≈ 3.5 s | `[MEASURED by difference]` |
| **peak CUDA `max_memory_allocated`** | **0.9224 GiB** of 7.996 | `[MEASURED]` |
| peak process working set (a different instrument, not comparable) | 1.357 GiB | `[MEASURED]` |
| `eval_nrmse` | `0.92608183617103412` | `[MEASURED]` |

**This same cell, re-run inside §3's 6-cell decomposition run, read `17.839 s`
in the training loop and returned `0.92608183617103412` — the identical
`eval_nrmse` to the last bit.** Two independent CUDA runs of one cell: the
result is bitwise-stable, the clock is not (`14.390` vs `17.839` s, +24 %). Seed
1 of the same arm read `14.101 s`, so the arm's three observations on this box
span `14.101 … 17.839 s`. A laptop GPU's clock is not stationary over a run,
which is why `scripts/k_cert.py` takes a median of medians and why §3.2 is quoted
to two decimals in GPU-minutes and not to the second.

### 2.1 Regime the timing ran in — stated because it is not the regime Ruling 1 describes

**`torch.use_deterministic_algorithms` was OFF.** The instrument does not set the
flag and never did; its `t="header"` record reads
`deterministic_algorithms: false` on all three probe runs `[MEASURED]`.

Measured here, directly, on `arm_smprime` on this card
(`CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before process start):

| regime | forward | backward |
|---|---|---|
| flag OFF | **OK** | **OK** |
| `use_deterministic_algorithms(True)` | **OK** | **`RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation`** |

This reproduces `COSTS.md` §1.6 and Ruling 1's *"the hole is in backward"*, and
it carries two consequences the ruling ledger does not yet state:

- **An R1/R1′ cell is not a forward-only cell.** It takes 150 gradient steps, so
  **Ruling 1 B2's strict regime cannot be applied to it at all** — not because
  anyone chose not to, but because the backward raises. The slot
  ⟨`DECIDING_CELL_BITWISE_RETAKE`⟩ in `V17K_RULINGS.md` §A1.3 asks whether each
  cell on the frozen list *"reproduces bitwise under strict mode on the certified
  4060"*; for every cell in §1 of this file the honest answer is that **strict
  mode is not executable on it**, and the cell belongs under A1.2's measured
  floor, not under A1.1's bitwise bar. Which of the two A1 classes the R1 cells
  fall into is the author's call, not this node's.
- **`CUBLAS_WORKSPACE_CONFIG=:4096:8` must be exported before the process
  starts.** Set from inside Python after CUDA is initialised it does not take,
  and strict mode then fails the **forward** as well — measured here both ways.
  A re-take script that sets it in-process would report a false blocker.

---

## 3. THE PROJECTION, BOTH WAYS

### 3.1 Per-cell, measured

Six CUDA cells (3 arms × seeds 0,1), one process, `t="wall"` = **41.842 s**,
process wall 44.533 s `[MEASURED]`.

| arm | s / 150 steps, mean of 2 | per-cell observations |
|---|---|---|
| `arm_smprime` | **15.970** | 17.839, 14.101 |
| `arm_pl` | **1.614** | 1.622, 1.605 |
| `softmax` | **1.497** | 1.496, 1.497 |

Fixed cost is genuinely fixed: 6 cells cost `41.842 − 38.160 = 3.68 s` of
non-cell time against a 1-cell run's `3.5 s`. Per-cell overhead outside the
training loop (eval NRMSE, bootstrap CI, `operator_columns` over 4,096 rows, the
gate probe) is **under 0.05 s/cell** on CUDA `[MEASURED by difference]`.

### 3.2 The total

| basis | per-seed (3 arms) | 24 cells + fixed | **GPU-min** |
|---|---|---|---|
| **measured-driven** (§3.1) | 19.080 s | **158.74 s** | **2.65** |
| **law cross-check** (§3.3) | 17.098 s | 142.88 s | **2.38** |

**Disagreement: +11.1 %.** Under the 20 % threshold, so no adjudication is
needed — but the two are not equally strong and the reason is worth one line.
The **law** describes the shape better (`R² = 0.9999999` for the arm, over three
in-VRAM points, taken as *"the median over 3 independent child processes of the
median of ≥12 timed steps"*); the **measurement** describes the level better
because it includes the whole instrument, not the training step alone. Trust the
measured column for booking the time and the law for extrapolating to another
`n` (within the range it was fitted over — see §1.5).

### 3.3 The law cross-check, arm by arm

`results/k_cert_local.json` `throughput.laws`, quoted as `COSTS.md` §1.1
`[FITTED]`. The law's regime is `s = 64`, `d_model = 16`, forward + backward +
`Adam.step` — **the same regime and the same shape as an R1 cell**, which is what
makes the comparison legitimate rather than a rescale.

| arm | law, `s/step` | ×150 | measured | ratio |
|---|---|---|---|---|
| `arm_smprime` | `exp(−10.0187)·n^1.0026` → **0.093066** | **13.960 s** | 15.970 s | **1.144** (+14.4 %) |
| `softmax` | `exp(−12.1852)·n^0.9963` → **0.010162** | **1.524 s** | 1.497 s | **0.982** (−1.8 %) |
| `arm_pl` | **no law** — the certificate fitted `softmax` and `arm_smprime` only | — | 1.614 s | — |

The arm's `+14.4 %` is on the **two-seed mean**; the single-cell end-to-end run
of §2 read `14.390 s` against the law's `13.960 s`, **+3.1 %**. The law sits
inside the observed spread `[14.101 … 17.839]`. Both arms clear 20 % either way,
so §3.2's two columns stand together and neither is discarded.

`arm_pl` has no law and its 1.614 s is `[MEASURED]` here only. It is 8 of the 24
cells and 12.9 s of the 158.7 s total, so the missing law costs the projection
nothing material.

### 3.4 Where the `~2 GPU-h` actually lives

| ladder | 24 cells, measured-scaled | 24 cells, law-scaled |
|---|---|---|
| **150 steps** — the ladder the journal holds | **0.044 GPU-h** (2.65 GPU-min) | 0.040 GPU-h |
| 9600 steps — the full ladder | **2.714 GPU-h** | 2.432 GPU-h |

The ruling's estimate is a good estimate **of the 9600-step ladder**, and
`V16_PRICING.md` §4 independently books R1′@9600 at `0.6985 h` for 16 cells on
the same reasoning. It is a 45× over-book **of the re-take**, because a re-take
is defined by the cells it supersedes: every `t="cell"` record in
`results/v15_r1.jsonl` carries `"steps": 150`, and a 9600-step cell cannot be
differenced against a 150-step one. Re-taking at 9600 would not be a re-take; it
would be a new measurement wearing the same cell id.

---

## 4. THE CPU↔CUDA DELTA, MEASURED

Ruling 4 refused *"tolerance widening to absorb an **unmeasured** CPU↔CUDA
delta."* **Nothing below widens any tolerance.** It removes the word
"unmeasured".

Six paired cells. For `arm_pl` and `softmax` the CPU side is **the superseded
journal record itself** (`results/v15_r1.jsonl`), so the pairing is exactly the
one Ruling 4's supersede-mark describes. For `arm_smprime` no CPU record has ever
existed, so both sides were taken here — the **same cell, same seed, same
150 steps, same `threads=8`**, once on each device.

| cell | `eval_nrmse` cpu | `eval_nrmse` cuda | **\|Δ\|** |
|---|---|---|---|
| `arm_smprime` seed 0 — **the timed cell** | `0.92611645301729251` | `0.92608183617103412` | **`3.462e-05`** |
| `arm_smprime` seed 1 | `0.88124683187165054` | `0.88124665516924749` | `1.767e-07` |
| `arm_pl` seed 0 *(vs journal)* | `0.645614434672464` | `0.644672619203993` | `9.418e-04` |
| `arm_pl` seed 1 *(vs journal)* | `0.644454142848541` | `0.644517454918750` | `6.331e-05` |
| `softmax` seed 0 *(vs journal)* | `0.971432426855954` | `0.973400229531387` | `1.968e-03` |
| `softmax` seed 1 *(vs journal)* | `0.933802192699265` | `0.938795633249175` | **`4.993e-03`** |

`[MEASURED]`, all twelve readings, this run. Full per-field table
(`train_nrmse`, `eval_h_hat`, `conservation_drift`, `a_hat_max`,
`dist_to_floor`) in `results/r4_price_probe.json`.

**Four things the table says.**

1. **The delta is nowhere near bitwise.** No `allclose` and no `torch.equal`
   survives a device move on these cells. Any re-take diff must be read against a
   stated tolerance, and this file states the size but chooses no tolerance.
2. **The worst delta is on the CONTROL, not on either arm** — `4.993e-03` on
   `softmax` seed 1, `144×` the arm's worst. The arms are the quieter half.
3. **It is the same order as an effect already in the ledger.** `MISTAKES.md`
   M-10, cited in `V15_R1.md` §9, measures **thread count alone** moving
   `eval_nrmse` by `2.345e-03` `[INHERITED]`. The worst device delta is `2.1×`
   that; the timed cell's is `1.5 %` of it. Device is not a new class of
   perturbation here — it is the same class as a pin the round already
   controls for.
4. **No verdict in `results/v15_r1.jsonl` moves.** The largest delta is `2.3 %`
   of R1's own resolution statement (`Δ = 0.21533 NRMSE`); the `arm_pl` seed-0
   delta is `1.5 %` of that cell's `dist_to_floor = −0.06149`, the crossing
   margin. **Every crossing and non-crossing in the journal survives the device
   move at these six cells.** That is six of sixteen — it is not the whole
   journal, and §5 says so.

---

## 5. LIMITS

- **Nothing here is a research reading.** Three probe runs, all with `ROOT`
  redirected to scratch; `results/v15_r1.jsonl` is untouched and appears in
  neither the start nor the end `git status --porcelain`. The cells this file
  timed exist to produce a clock reading and are not offered as R1′.
- **The instrument was timed mid-edit, and that is stated rather than avoided.**
  `scripts/v15_r1.py` is `M` in the working tree: another node has landed
  Ruling 5's `instrument_manifest` hash. It was timed **as it stands**, not at
  HEAD. Instrument hash on every probe record:
  `77429cdbb37dc0f3a6c5ca59910a6842063e9c721831d6fbc2665fcd26535498`. If that
  node edits further, the 2.1 s import cost and the hash both move; the cell
  timings do not, because the edit is import-time only.
- **`N = 2` on the timings.** Per-arm seconds are the mean of two seeds. The
  arm's own spread across three observations is `14.101 … 17.839 s` (±12 % about
  the mean), which is why §3.2's total is quoted to two decimals in GPU-minutes
  and not to the second.
- **The delta is measured on 6 of the 24 cells, at seeds 0 and 1 only.** Seeds
  2–7 are unmeasured on both devices for `arm_smprime`, and unmeasured on CUDA
  for `arm_pl`/`softmax`. Three of R1's eight `arm_pl` seeds never learned
  (`V15_R1.md` LIMITS); **none of those three is in this sample**, and a
  non-learning seed is exactly where a device delta could be larger. The `≤
  4.993e-03` figure is a floor on the worst case, not a bound on it.
- **Fixed overhead is `[MEASURED by difference]`, not instrumented directly.**
  `3.5 s` from a 1-cell run and `3.68 s` from a 6-cell run; the two runs carried
  1 and 3 bind checks respectively, so the split between "per-arm bind" and
  "per-run bar" is not resolved. It is 4 % of the total and is booked at 4.0 s.
- **Q2 is not priced.** §1.5: the adopted instrument has no `t*` flag, and
  `n = 16,384` does not fit the card under a training loop. Both are blockers on
  the author's desk, not estimates.
- **The 9600-step column is scaled, not measured.** It multiplies the measured
  150-step cell by `64` and assumes step cost is constant across the run. That is
  what the throughput law asserts and it was not verified past 150 steps here.

---

## 6. RECOMMENDATION

**Spend it. The full re-take is 2.65 GPU-min — under three minutes for all
24 cells in one invocation, at a peak of 0.92 GiB against a 7.996 GiB card.** It
is not in the class of work the author's "no multi-hour local GPU blocks"
instruction was written to stop, and it is not a candidate for Kaggle under any
reading — the KILL clause forbids that and there is nothing to save.

Two things the author should settle before it runs, because both are decisions
and neither is a measurement:

1. **The determinism regime for a cell that trains.** §2.1: strict mode is not
   executable on an R1 cell. The instrument currently runs with the flag **fully
   off**, not `warn_only=True`. Ruling 1 makes `warn_only=True` the regime for
   the whole run; making the re-take match it is a one-flag decision that costs
   nothing (`V16_DEVICE_CERT.md` §5.3.1, quoted in `V17K_RULINGS.md` §A1: the
   `warn_only`/off ratios read `1.06/1.06/1.26`, `1.06/0.95/1.09`,
   `1.18/1.18/1.32` across three runs — **including a value below 1**, which a
   real cost cannot produce) and it is not this node's to take.
2. **The tolerance the re-take's `|Δ|` is read against.** §4 gives its size and
   deliberately proposes no number. Ruling 4 refused widening to absorb an
   unmeasured delta; choosing a tolerance now that the delta is measured is a
   different act, and it is the author's.
