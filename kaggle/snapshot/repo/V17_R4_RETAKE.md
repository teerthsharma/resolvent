# V17-K RULING 4 — THE RE-TAKE, TAKEN

**These are research readings.** 24 cells on the certified RTX 4060, one
invocation, written into `results/v17k_r4_retake.jsonl`. No `ROOT` redirect: the
journal is the point. `results/v15_r1.jsonl` keeps all 25 of its original lines,
byte for byte, and carries one appended supersede marker.

| | |
|---|---|
| `git rev-parse HEAD` at start | `ab5b48547884e04258276e6e808d5a71ea65f917` |
| `git rev-parse HEAD` at end | `ab5b48547884e04258276e6e808d5a71ea65f917` |
| writing git commands run | **none** |
| `git status --porcelain` delta over the run | three lines added: `?? results/v17k_r4_retake.jsonl`, `?? results/v17k_r4_floor.jsonl`, `?? V17_R4_RETAKE.md`. `M scripts/v15_r1.py` and ` M results/v15_r1.jsonl` — the first was already `M` at start (another node's RULING 5 edit, plus this node's regime edit on top); the second is the appended supersede marker |
| box | NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 B, `torch 2.5.1+cu121`, `torch.set_num_threads(8)` `[MEASURED]` |
| instrument | `scripts/v15_r1.py`, working tree, `instrument_hash` **`5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309`** on all 24 cell records `[MEASURED]` |
| regime | `torch.use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8` **exported before process start** `[MEASURED — journalled on the header]` |

---

## THE ANSWER, FIRST

**The re-take cost 2.79 GPU-min against a 2.65 projection (+5.3 %) and 0.974 GiB
of 7.996 (+5.6 % on peak).** Every cell landed.

**Three findings the pricing probe's 6-cell sample could not see, all three of
them on the seeds it did not sample:**

1. **The worst CPU↔CUDA delta is NOT on the control.** It is `arm_pl` **seed 7**
   at **`9.522e-03`**, `1.91×` the probe's worst (`4.993e-03`, `softmax` seed 1)
   and `4.06×` M-10's thread-count effect. Seed 7 is one of R1's three
   non-learning seeds. The probe's own LIMITS predicted exactly this class and
   said the `4.993e-03` figure was "a floor on the worst case, not a bound on
   it"; it was a floor, and the bound is `1.91×` higher.
2. **"Non-learning" is not the mechanism, and the other two non-learning seeds
   say so.** Seeds 2 and 3 read `1.505e-04` and `6.329e-05` — the quiet half of
   the whole 16. One diverged seed in three moved; two did not.
3. **The `warn_only=True` flag changed no value.** All six cells the pricing
   probe took flag-OFF reproduce **bitwise** flag-ON, and the identical-seed
   repeat under the flag is **bitwise on all six pairs**: RULING 9's
   `δ_nrmse = 0.0`.

**No verdict in `results/v15_r1.jsonl` moves.** Sixteen of sixteen crossings and
non-crossings survive the device move, both per cell and in aggregate. §4.

---

## 1. THE REGIME — WHAT CHANGED IN THE INSTRUMENT, AND ONLY THAT

RULING 1 fixes **CUDA with `warn_only=True`**. The instrument set **no flag at
all** and journalled `deterministic_algorithms: false` — neither of the two ruled
regimes. Four sites in `scripts/v15_r1.py`, nothing else:

| site | change |
|---|---|
| module docstring | the paragraph claiming the flag "is NOT set by this file and never was" is replaced by the regime it now runs under, and by why strict mode is not executable on a cell that trains |
| `import os` | added, for one `os.environ.get` |
| `main()`, before `torch.set_num_threads` | `torch.use_deterministic_algorithms(True, warn_only=True)` — set before the bind, the bar calibration, the eval draw and every cell, so one run is one regime |
| header record + console line | `deterministic_warn_only` and `cublas_workspace_config` journalled beside the existing `deterministic_algorithms` |

**`CUBLAS_WORKSPACE_CONFIG` is READ, never set.** The pricing probe measured that
setting it in-process after CUDA init does not take. The instrument journals what
it inherited, so a run launched without it is legible in its own header rather
than silently mis-described. This run's header reads `":4096:8"` `[MEASURED]`.

**Nothing about what the instrument measures was touched.** The proof is not the
diff — it is §5: all six cells with a flag-OFF reading reproduce to the last bit.

**The regime is doing something, and the run says so out loud.** Under the flag,
`cumsum_cuda_kernel` warns in the **forward** (`ceq/arm_pl.py:90`, `:140`;
`ceq/arm_smprime.py:184`, `:185`; `ceq/arm_phase.py:210`) and in the **backward**
(`torch/autograd/graph.py:825`) `[MEASURED]`. Under `warn_only=False` that last
one is the `RuntimeError` RULING 9 cites; it is why an R1′ cell is a floor cell.

---

## 2. THE 24 CELLS

`results/v17k_r4_retake.jsonl`, 424 lines, `device: "cuda"`, `task=e3_t2`,
`s=64`, `d=24`, `d_model=16`, `n_train=2048`, `n_eval=4096`, `steps=150`,
`threads=8`. Every one carries
`instrument_hash = 5d41a63d…09` `[MEASURED]`.

| arm | seed | `eval_nrmse` | `dist_to_floor` |
|---|---|---|---|
| `arm_pl` | 0 | `0.6446726192039927` | `−0.062434` |
| `arm_pl` | 1 | `0.6445174549187496` | `−0.062589` |
| `arm_pl` | 2 | `1.1522795055459243` | `+0.445173` ‡ |
| `arm_pl` | 3 | `1.1133392329955414` | `+0.406232` ‡ |
| `arm_pl` | 4 | `0.6337391039935976` | `−0.073368` |
| `arm_pl` | 5 | `0.6419986310848815` | `−0.065108` |
| `arm_pl` | 6 | `0.6621282051474511` | `−0.044979` |
| `arm_pl` | 7 | `1.1489267727154717` | `+0.441820` ‡ |
| `arm_smprime` | 0 | `0.9260818361710341` | `+0.218975` |
| `arm_smprime` | 1 | `0.8812466551692475` | `+0.174140` |
| `arm_smprime` | 2 | `0.20391993939877656` | `−0.503187` |
| `arm_smprime` | 3 | `0.9162579895429029` | `+0.209151` |
| `arm_smprime` | 4 | `0.9119259828412737` | `+0.204819` |
| `arm_smprime` | 5 | `0.9181568234297154` | `+0.211050` |
| `arm_smprime` | 6 | `0.8932391322242402` | `+0.186132` |
| `arm_smprime` | 7 | `0.9267169213804543` | `+0.219610` |
| `softmax` | 0 | `0.9734002295313872` | `+0.266293` |
| `softmax` | 1 | `0.9387956332491751` | `+0.231689` |
| `softmax` | 2 | `0.9455472186895695` | `+0.238440` |
| `softmax` | 3 | `0.9515546327151698` | `+0.244448` |
| `softmax` | 4 | `0.9605147968181855` | `+0.253408` |
| `softmax` | 5 | `0.9456702325679535` | `+0.238563` |
| `softmax` | 6 | `0.960665082247913` | `+0.253558` |
| `softmax` | 7 | `0.9443373459209473` | `+0.237231` |

`[MEASURED]`, all 24. ‡ = one of R1's three non-learning `arm_pl` seeds
(`V15_R1.md` LIMITS: *"3 of 8 are NO READING"*).

**The 8 `arm_smprime` cells are R1′ and they supersede nothing** — no CPU record
of that arm has ever existed. They are new readings, not a re-take, and the
supersede marker says so.

**No verdict is offered on them here.** One line of arithmetic, because it is a
reading and not an adjudication: `arm_smprime` seed 2 sits `0.50` below `floor_1`
while the other seven sit above it, and the seed aggregate is
`mean = 0.822193`, `CROSSES = False`. What that means is the author's.

---

## 3. CPU ↔ CUDA, ALL 16 PAIRED CELLS

The CPU side is the superseded journal record itself. Both sides are the same
cell, same seed, 150 steps, `threads=8`.

| arm | seed | `eval_nrmse` cpu | `eval_nrmse` cuda | **\|Δ\|** |
|---|---|---|---|---|
| `arm_pl` | 0 | `0.645614434672464` | `0.644672619203993` | `9.418e-04` * |
| `arm_pl` | 1 | `0.644454142848541` | `0.644517454918750` | `6.331e-05` * |
| `arm_pl` | **2 ‡** | `1.152430028460822` | `1.152279505545924` | `1.505e-04` |
| `arm_pl` | **3 ‡** | `1.113402523560700` | `1.113339232995541` | `6.329e-05` |
| `arm_pl` | 4 | `0.634001508525320` | `0.633739103993598` | `2.624e-04` |
| `arm_pl` | 5 | `0.641880609328137` | `0.641998631084882` | `1.180e-04` |
| `arm_pl` | 6 | `0.662020773414460` | `0.662128205147451` | `1.074e-04` |
| `arm_pl` | **7 ‡** | `1.139404402712210` | `1.148926772715472` | **`9.522e-03`** |
| `softmax` | 0 | `0.971432426855954` | `0.973400229531387` | `1.968e-03` * |
| `softmax` | 1 | `0.933802192699265` | `0.938795633249175` | `4.993e-03` * |
| `softmax` | 2 | `0.945522870654963` | `0.945547218689569` | `2.435e-05` |
| `softmax` | 3 | `0.951173669060836` | `0.951554632715170` | `3.810e-04` |
| `softmax` | 4 | `0.960759326661283` | `0.960514796818186` | `2.445e-04` |
| `softmax` | 5 | `0.945679411014603` | `0.945670232567953` | `9.178e-06` |
| `softmax` | 6 | `0.960282199651425` | `0.960665082247913` | `3.829e-04` |
| `softmax` | 7 | `0.945482338627906` | `0.944337345920947` | `1.145e-03` |

`[MEASURED]`. `*` = one of the 6 cells in the pricing probe's sample; all four
reproduce it exactly. `‡` = non-learning in R1.

**THE THREE SEEDS THE PROBE FLAGGED AND COULD NOT SEE.** The probe's LIMITS:
*"Three of R1's eight `arm_pl` seeds never learned; none of those three is in
this sample, and a non-learning seed is exactly where a device delta could be
larger."* Measured:

| seed | `eval_nrmse` cpu → cuda | **\|Δ\|** | where it ranks in the 16 |
|---|---|---|---|
| **2** | `1.152430` → `1.152280` | `1.505e-04` | 8th |
| **3** | `1.113403` → `1.113339` | `6.329e-05` | 12th |
| **7** | `1.139404` → `1.148927` | **`9.522e-03`** | **1st — the worst of all 16** |

**The hypothesis is half right and the half that fails is the interesting half.**
A non-learning seed *can* carry the largest delta — seed 7 does, at nearly twice
the control's worst. But two of the three are among the quietest cells in the
journal, so "did not learn" is not itself the mechanism.

**What moved on seed 7, field by field** `[MEASURED]`: `a_hat_max` reads
`285.0719` on CPU and `116.0061` on CUDA — the divergent gate's blow-up differs
by `2.5×` — while `sign_acc` is **bitwise identical** on both devices
(`0.6905517578125`) and `gate_r2` moves by `1.5e-04`. The device does not change
what the diverged seed *does*; it changes how far the unbounded quantity got
before the 150 steps ran out. `v_max` is bitwise identical on both.

**Field coverage, stated because it bounds the diff.** The CPU cell records carry
42 fields; the CUDA records carry 60. The 18 extra are V16's gate columns
(`lambda_hat`, `unit_root`, `z_winding_*`, `frac_gate_annihilated`, their
`_0step` controls) and RULING 5's `instrument_hash`. **The CPU journal cannot be
diffed on the gate columns at all** — it predates them. 42 fields are paired; 18
are new on one side and unpairable.

---

## 4. WHETHER ANY VERDICT MOVES — IT DOES NOT

The pricing node's reading, checked against the full 24 rather than 6:

| the probe's claim | on 6 cells | **on all 16 paired cells** | survives? |
|---|---|---|---|
| worst \|Δ\| as % of R1's resolution statement `Δ = 0.21533` | 2.3 % | **4.42 %** | **YES** — the number doubles, the conclusion does not |
| `arm_pl` seed-0 \|Δ\| as % of that cell's crossing margin `−0.06149` | 1.5 % | **1.53 %** (unchanged — same cell, same reading) | **YES** |
| worst \|Δ\| / crossing margin, over every cell | not computed | **2.20 %**, at `arm_pl` seed 7 and `softmax` seed 1 (equal to 3 s.f.) | **YES** |
| worst \|Δ\| vs M-10's thread-count effect `2.345e-03` | 2.1× | **4.06×** | it is still the same class of perturbation as a pin the round controls for, at 4× rather than 2× |

**Per-cell crossings: zero sign flips in 16.** Every `dist_to_floor` keeps its
sign across the device move — `arm_pl` seeds 0/1/4/5/6 stay below the floor,
seeds 2/3/7 stay above it, all 8 `softmax` cells stay above it `[MEASURED]`.

**Aggregate verdicts: unmoved.**

| row | CPU | CUDA | move |
|---|---|---|---|
| `arm_pl` mean | `0.829151` | `0.830200` | `+1.049e-03` |
| `arm_pl` `CROSSES` | `False` | `False` | — |
| `softmax` mean | `0.951767` | `0.952561` | `+7.94e-04` |
| `softmax` `CROSSES` | `False` | `False` | — |
| contrast `arm_pl − softmax` mean | `−0.122616` | `−0.122360` | `+2.56e-04` |
| resolution statement `Δ` | `0.215326` | `0.217498` | `+1.0 %` |

**Plainly: it survives the seeds the probe did not sample.** The worst delta
doubled, and it is still `4.4 %` of the resolution statement and `2.2 %` of the
tightest crossing margin. Nothing crosses that did not, nothing fails to cross
that did.

**No tolerance is chosen here.** RULING 4 refused widening to absorb an
*unmeasured* delta; the delta is now measured on all 16, and picking a number to
read it against remains the author's.

---

## 5. THE `warn_only` FLAG CHANGED NOTHING — MEASURED, NOT ASSUMED

Six cells have a flag-OFF CUDA reading (`results/r4_price_probe.json`, same box,
same day, instrument one edit earlier). Flag-ON, this run:

| cell | flag OFF | flag ON | \|Δ\| |
|---|---|---|---|
| `arm_pl` seed 0 | `0.6446726192039927` | `0.6446726192039927` | **`0.0`** |
| `arm_pl` seed 1 | `0.6445174549187496` | `0.6445174549187496` | **`0.0`** |
| `arm_smprime` seed 0 | `0.9260818361710341` | `0.9260818361710341` | **`0.0`** |
| `arm_smprime` seed 1 | `0.8812466551692475` | `0.8812466551692475` | **`0.0`** |
| `softmax` seed 0 | `0.9734002295313872` | `0.9734002295313872` | **`0.0`** |
| `softmax` seed 1 | `0.9387956332491751` | `0.9387956332491751` | **`0.0`** |

`[MEASURED]` — bitwise, `==` on the float, not `allclose`. **6 of 6.** The flag
selects no different kernel on this card for these ops; it only decides whether
`cumsum_cuda_kernel` warns or raises. That is consistent with
`V16_DEVICE_CERT.md` §5.3.1's `warn_only`/off cost ratios including a value below
1 — there was no second kernel to be slower.

**This is 6 of 24, and the other 18 have no flag-OFF counterpart.** Producing one
would cost another 2.8 GPU-min and was not ordered.

---

## 6. RULING 9 — THE FLOOR, `δ_nrmse`

**This is the identical-seed repeat, and it is NOT the seed-to-seed spread.**
The two answer different questions and this section reports only the first:

- **`δ_nrmse` (this section)** — the same cell, same seed, same device, same
  regime, run twice. It measures **the instrument's own run-to-run noise**, and
  it is the quantity an equality claim is read against.
- **the seed-to-seed spread** (`sd` in §4's aggregate table: `0.2554` for
  `arm_pl`, `0.2503` for `arm_smprime`, `0.0115` for `softmax`) — eight
  *different* cells. It measures **how much the seed matters**, which is a
  property of the task, not of the instrument. It is `≥ 10⁴×` larger and it is
  not a floor.

Second invocation, `results/v17k_r4_floor.jsonl`, same regime
(`warn_only=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuda), seeds 0 and 1 of all
three arms, 44.7 s:

| cell | take a | take b | **`δ_nrmse`** |
|---|---|---|---|
| `arm_pl` seed 0 | `0.6446726192039927` | `0.6446726192039927` | **`0.0`** |
| `arm_pl` seed 1 | `0.6445174549187496` | `0.6445174549187496` | **`0.0`** |
| `arm_smprime` seed 0 | `0.9260818361710341` | `0.9260818361710341` | **`0.0`** |
| `arm_smprime` seed 1 | `0.8812466551692475` | `0.8812466551692475` | **`0.0`** |
| `softmax` seed 0 | `0.9734002295313872` | `0.9734002295313872` | **`0.0`** |
| `softmax` seed 1 | `0.9387956332491751` | `0.9387956332491751` | **`0.0`** |

**`δ_nrmse = 0.0`, bitwise, 6 of 6, under `warn_only=True` on the certified
4060** `[MEASURED]`. The flag-OFF bitwise repeat the pricing node saw **survives
the flag**, and it was measured rather than carried across — §5's table is the
independent half of that statement.

**What this licenses, and what it does not.** Every equality claim about an R1′
cell is now held against a floor of zero, which is the tightest the north star
could have asked for: `9.522e-03` of device delta is not `9.522e-03 ± noise`, it
is `9.522e-03` of real, reproducible difference. **The floor does not make the
device delta small — it makes it certain.**

**A floor of exactly zero is a claim about this box and this shape, and it is one
draw per cell.** Ruling 2a's own words: one pair is one draw of the noise. Six
pairs across three arms all reading exactly zero is stronger than one, and it is
still not a proof that the seventh would.

---

## 7. THE COST, AGAINST THE PROJECTION

| quantity | projected (`V17_R4_RETAKE_PRICE.md` §3.2) | **measured** | delta |
|---|---|---|---|
| instrument `t="wall"` `secs`, 24 cells | 158.74 s | **167.17 s** | **+5.3 %** |
| **GPU-min** | **2.65** | **2.79** | +5.3 % |
| process wall, end to end | — | **170.99 s** | — |
| peak `torch.cuda.max_memory_allocated` | 0.9224 GiB | **0.9739 GiB** | +5.6 %, **12.2 % of the card's 7.996 GiB** |
| peak process working set | 1.357 GiB | 1.386 GiB | +2.1 % |
| `arm_smprime` s/150 steps | 15.970 | **16.161** (mean of 8) | +1.2 % |
| `arm_pl` s/150 steps | 1.614 | **1.780** (mean of 8) | +10.3 % |
| `softmax` s/150 steps | 1.497 | **1.681** (mean of 8) | +12.3 % |

**The projection held.** `+5.3 %` on a figure the pricing node quoted to two
decimals in GPU-minutes precisely because a laptop GPU's clock is not stationary.
The two cheap arms ran `~11 %` slow against an `N = 2` mean and the expensive arm
— the one that decides the total — ran `+1.2 %`.

**Against the ruling's `~2 GPU-h`: the re-take took 2.79 GPU-min, a 43× over-book
`[MEASURED]`.** The pricing node's 45× was computed against its own projection;
this is the same number against the clock.

**The CPU run this supersedes took 1358.3 s** (`results/v15_r1.jsonl`, `t="wall"`)
for 16 cells. The CUDA run took 167.2 s for **24**. `[MEASURED, both]`

---

## 8. L-G2 — THE SUPERSEDE MECHANISM

**Form used: an append-only supersede marker record. No `superseded_by` field was
added to any existing record.**

| | |
|---|---|
| `results/v15_r1.jsonl` **before** | **25 lines**, md5 **`3559fd5f24ba0dfb67ac176c4584f58e`** |
| `results/v15_r1.jsonl` **after** | **26 lines**, md5 **`4735741405e45cf197bb75867b8aa551`** |
| bytes rewritten | **zero** — verified: the first 25 lines of the file after the append are byte-for-byte the 25 lines before it `[MEASURED]` |
| records deleted | **zero** |

**Why append-only and not in place.** Two reasons, and the first is the
journal's own convention rather than a preference:

1. **Every journal in `results/` is written by a process that opens it with mode
   `"a"`.** `scripts/v15_r1.py:558` is `open(jl, "a", encoding="utf-8")`, and
   nothing in the repo re-writes a journal in place. There is **no precedent for
   an in-place edit of a `results/*.jsonl`** and none for a `superseded_by`
   field: `grep` over `results/*.jsonl` finds the string "supersede" in no file.
   The marker record is therefore the form the journal's conventions support.
2. **In-place superseding rewrites protected bytes.** Adding a field to the 16
   cell records would rewrite lines carrying `manifest` hashes — the exact
   artifact L-G2 exists to keep intact. An append leaves the original digest
   quotable, and the marker quotes it (`md5_before_this_record`), so a reader can
   still verify the superseded content is the content that was superseded.

**The marker carries** `t: "supersede"`, `schema: "supersede/1"`, the 16
superseded cell ids, the pre-append line count and md5, the replacing file with
its md5 / line count / device / tag / `instrument_hash` / the three regime
fields, the RULING 9 floor file, `deleted: false`, and a pointer to this report.
It is verbatim-quotable and self-describing.

**It breaks no consumer.** `kaggle/ceq_v17k.ipynb` cell 14 — the only code that
reads this file — selects with `r.get("t") == "cell"`, so an unknown record type
is inert to it `[MEASURED by read]`.

---

## 9. FINDINGS FOR OTHER NODES — FLAGGED, NOT TOUCHED

- **`kaggle/ceq_v17k.ipynb` cell 14 still points `LOCAL_Q1Q2_JSONL` at
  `results/v15_r1.jsonl`.** Its own comment says *"CONFIRM this is the CERTIFIED
  CUDA reading, not a stale cpu-tagged file"*. It is not; it is now the
  superseded CPU file carrying a marker that says so. The certified reading is
  **`results/v17k_r4_retake.jsonl`**. `kaggle/` is not this node's to edit.
- **That cell also invokes the instrument with default `--arms`
  (`arm_pl softmax`)**, so a Kaggle Q1 run would produce no `arm_smprime` row to
  diff against the 8 R1′ cells taken here.
- **⟨SLOT `DECIDING_CELL_BITWISE_RETAKE`⟩ (`V17K_RULINGS.md` §A1.3) cannot be
  filled as written for these cells, and RULING 9 is why.** It asks whether each
  cell reproduces bitwise *under strict mode*; strict mode is not executable on a
  cell that trains. What *is* now measured is §6: these cells reproduce bitwise
  under the ruled `warn_only=True` regime, `δ_nrmse = 0.0` on 6 of 6 pairs. That
  is the A1.2 answer to an A1.1 question and the substitution is the author's to
  accept or refuse.

---

## 10. LIMITS

- **`δ_nrmse` is measured on 6 cells of 24, at seeds 0 and 1.** Seeds 2–7 have no
  identical-seed repeat on any arm. All six read exactly zero; that is six draws,
  not a proof about the other eighteen.
- **The flag ON/OFF comparison is 6 of 24** — only those cells have a flag-OFF
  counterpart, and it comes from another node's probe file rather than from a
  run this node took. The instrument differs by this node's regime edit between
  the two readings, which is precisely why the six reading bitwise-identical is
  evidence the edit changed nothing measured.
- **The 8 `arm_smprime` cells have no CPU counterpart and no `|Δ|` column.**
  R1′ was never taken on CPU. They supersede nothing and their §3 rows do not
  exist.
- **The CPU journal predates 18 of the 60 fields.** The gate columns
  (`lambda_hat`, `unit_root`, `z_winding_*`, `frac_gate_annihilated`) and
  `instrument_hash` cannot be diffed across the device move at all.
- **`arm_pl` seed 7's `9.522e-03` is one cell.** Whether the mechanism is
  "unbounded `a_hat` on a diverged seed accumulates device-dependent error" or
  something narrower is not established by one cell, and seeds 2 and 3 —
  non-learning, `a_hat_max` `20.3` and `49.7` — did not do it. What is measured
  is the delta and the `285.07 → 116.01` move in `a_hat_max` beside it.
- **No tolerance is proposed and no verdict is offered.** §4 states which
  existing verdicts move (none) and stops. The R1′ readings in §2 are printed
  without adjudication.
- **The instrument hash changed** from `77429cdb…` to `5d41a63d…` because this
  node edited the file. Cells taken before this run cite the old hash; the six
  in §5 are the bridge between them and read identically.
