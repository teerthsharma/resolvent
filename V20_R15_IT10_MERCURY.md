# V20 R15 — it.10 — MERCURY (LESTRADE)

Branch `v17k-gate0`. No git writes. Nothing touched Kaggle. Every run LOCAL on the
4060. Wall clock honoured. New journal only:
`results/v20_r15_it10_mercury_rescore.jsonl`. `results/v17k_r4_retake.jsonl` was
**read** and never opened for write.

---

## 0. MARS'S PRICE IS WRONG, AND THE REASON IS ONE LINE OF THE INSTRUMENT

`V20_R15_IT9_MARS.md:370-379` prices the second eval seed at **under one
GPU-second** because *"trained models are kept in `models[(kind, seed)]`"*
`[READ scripts/v15_r1.py:879]`. The read is correct and the inference is not.

```
[RUN] grep -n "\bmodels\b" scripts/v15_r1.py
705:    rows, models = [], {}
879:            models[(kind, seed)] = model

[RUN] grep -n "torch.save|state_dict|torch.load" scripts/v15_r1.py ceq/*.py scale/*.py
(no match in scripts/v15_r1.py; ceq/arm_smprime.py:464 is a comment reading
 "NOTHING IS KEPT (L-LEAN). No cell, no checkpoint, no loss curve")
```

**`models` is written at `:879` and never read at any line of the file.** There is
no `torch.save`, no `state_dict`, no checkpoint path anywhere under `scripts/`,
`ceq/` or `scale/`. The dict is a local that dies when `main()` returns. *Kept*
is true **inside** a run and false **between** runs, and no run is in progress.

So the second eval draw costs a **retrain**, not a forward pass — unless the
instrument is edited to persist models or to loop over eval seeds, which this
iteration is forbidden to do and which SATURN and JUPITER have already priced
separately.

**It ran anyway, because the retrain price is small and it is exactly measurable.**
`[RUN]` `results/v20_r15_it10_mercury_rescore.jsonl` `t="wall"`:
**`36.05 s` for 18 trained cells scored on 3 eval draws each = 54 scorings**,
against MARS's `<1 GPU-s`. **The overrun against the brief's "under 20 GPU-seconds
combined" is 36.05 s, and it is the finding of this section**: the round has been
pricing a measurement off a variable name rather than off the variable's readers.

### 0.1 The route that does not touch the instrument

`scripts/v20_r15_it10_mercury_rescore.py` is a **new file**. It does not edit
`v15_r1.py`; it loads it with `importlib.util.spec_from_file_location` and calls
the runner's **own** `train_one`, `gate_columns`, `gate_features`,
`recovered_gate`, `probe` and `nrmse`. `scripts/v15_r1.py` is byte-identical to
HEAD and `instrument_hash` is unmoved:

```
[RUN] git status --porcelain -- scripts/v15_r1.py
(empty)
[RUN] header instrument_hash, both journals
5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309   (it.8)
5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309   (it.10)
```

---

## 1. CONTROL FIRST — THE RETRAIN REPRODUCES BOTH FROZEN JOURNALS BITWISE

Training is deterministic in the cell seed (`torch.manual_seed(seed)` then
`make_arm` on CPU before `.to(device)`, `[READ scripts/v15_r1.py:230-235]`), so a
retrain that did **not** reproduce the journalled cell to the bit would mean the
re-score was scoring a different model and nothing below would be admissible.

`[RUN] 10/10 bitwise identical` at eval seed 12345:

| cell | frozen journal `eval_nrmse` | it.10 retrain | bitwise |
|---|---|---|---|
| `arm_pl` 0 | `0.6446726192039927` | `0.6446726192039927` | yes |
| `arm_pl` 8 | `0.6248685523735131` | `0.6248685523735131` | yes |
| `arm_pl` 9 | `1.281779592990027` | `1.281779592990027` | yes |
| `arm_pl` 10 | `0.6398332532283637` | `0.6398332532283637` | yes |
| `arm_pl` 11 | `0.6408209504227382` | `0.6408209504227382` | yes |
| `arm_pl` 12 | `0.6762082068688114` | `0.6762082068688114` | yes |
| `arm_pl` 13 | `0.6348616577583769` | `0.6348616577583769` | yes |
| `arm_pl` 14 | `0.6338928504917678` | `0.6338928504917678` | yes |
| `arm_pl` 15 | `0.6805785772206486` | `0.6805785772206486` | yes |
| `softmax` 0 | `0.9734002295313872` (`v17k_r4_retake.jsonl`) | `0.9734002295313872` | yes |

Nine against `results/v20_r15_it8_armpl_b.jsonl`, one against the frozen
`results/v17k_r4_retake.jsonl`. The eight fresh `softmax` cells have no frozen
counterpart by construction — that is section 3's whole point.

### 1.1 REGIME DIFF, PRINTED

```
KEY                            it8 v20_r15_it8_armpl_b        it10 v20_r15_it10_mercury_rescore
DIFF arms                      ["arm_pl"]                     ["arm_pl", "softmax"]
DIFF eval_seeds                <absent>                       [12345, 12346, 20260902]
DIFF tag                       "v20_r15_it8_armpl_b"          "v20_r15_it10_mercury_rescore"
DIFF when                      "2026-09-02 02:57:44"          "2026-09-02 03:25:18"
DIFF why                       <absent>                       "re-score kept-nothing models on 3 eval draws; ..."
-- header keys: 24 union, 19 identical, 5 differ
```

**24 union keys, 19 identical, 5 differ**, and of the 5 only `arms` names a
measurement condition; `eval_seeds` and `why` are new fields this file adds,
`tag` and `when` are identity. `device`, `steps`, `n_train`, `n_eval`, `s`, `d`,
`d_model`, `lr`, `threads`, `torch`, `floor_1`, `instrument_hash`,
`deterministic_algorithms`, `deterministic_warn_only`,
`cublas_workspace_config`, `t_star`, `task`, `seeds` and `t` are all in the
identical 19. The count is printed rather than described, per the standing
correction against this office.

---

## 2. EXPERIMENT A — **THE VERDICTS DO NOT FLIP. 54 OF 54.**

Three eval draws: `12345` (the pinned one), `12346`, `20260902`. Same 18 trained
models, three `torch.no_grad()` forwards each, the runner's `nrmse` with the
cell's own `mu`/`sigma` from its training batch.

| kind | seed | `eval_nrmse` @12345 / @12346 / @20260902 | crosses `floor_1` | stable |
|---|---|---|---|---|
| `arm_pl` | 0 (control) | 0.644673 / 0.639804 / 0.649342 | T T T | **yes** |
| `arm_pl` | 8 | 0.624869 / 0.628694 / 0.636078 | T T T | **yes** |
| `arm_pl` | **9** | 1.281780 / 1.231621 / 1.489955 | **F F F** | **yes** |
| `arm_pl` | 10 | 0.639833 / 0.650118 / 0.646051 | T T T | **yes** |
| `arm_pl` | 11 | 0.640821 / 0.640625 / 0.646954 | T T T | **yes** |
| `arm_pl` | 12 | 0.676208 / 0.658023 / 0.664551 | T T T | **yes** |
| `arm_pl` | 13 | 0.634862 / 0.641536 / 0.645894 | T T T | **yes** |
| `arm_pl` | 14 | 0.633893 / 0.635862 / 0.649289 | T T T | **yes** |
| `arm_pl` | 15 | 0.680579 / 0.666654 / 0.686874 | T T T | **yes** |
| `softmax` | 0, 8..15 (9 cells) | 0.922856 … 1.049890 over all 27 scorings | **F on all 27** | **yes** |

**Every one of the 18 cells returns the same crossing verdict on all three draws.
No verdict flips.** The per-cell reading is a property of the trained model, not
of `seed=12345`.

`lambda_hat`'s sign is likewise draw-invariant: `+` on all three draws for
`arm_pl` seed 9, `-` on all three draws for the other 17 cells. The sign
separator MARS confirmed out-of-sample at `V20_R15_IT9_MARS.md:246-254` survives
a change of eval draw as well as a change of training seed.

### 2.1 What still moves with the draw, and it is the aggregate

The **cell** verdicts are stable; the **aggregate** interval is not constant.
Recomputed with the runner's own estimator (`scripts/v15_r1.py:899-909`):

| kind | eval seed | n | mean | sd | `ci_hi` | `crosses` | cells below floor |
|---|---|---|---|---|---|---|---|
| `arm_pl` | 12345 | 9 | 0.7175018067 | 0.2124544441 | `0.8808087489119841` | **False** | 8/9 |
| `arm_pl` | 12346 | 9 | 0.7103261727 | 0.1958279067 | `0.8608528269350806` | **False** | 8/9 |
| `arm_pl` | 20260902 | 9 | 0.7461099065 | 0.2793292339 | `0.9608213627296274` | **False** | 8/9 |
| `arm_pl` ex-seed 9 | 12345 | 8 | 0.6469670834 | 0.0202996440 | `0.6639380105668372` | **True** | 8/8 |
| `arm_pl` ex-seed 9 | 12346 | 8 | 0.6451643492 | 0.0123750823 | `0.6555101769062579` | **True** | 8/8 |
| `arm_pl` ex-seed 9 | 20260902 | 8 | 0.6531292510 | 0.0157189660 | `0.6662706354217347` | **True** | 8/8 |
| `softmax` | 12345 | 9 | 0.9688518111 | 0.0365855459 | `0.9969739511367557` | False | 0/9 |
| `softmax` | 12346 | 9 | 0.9685033731 | 0.0371578538 | `0.9970654279283372` | False | 0/9 |
| `softmax` | 20260902 | 9 | 0.9755799488 | 0.0354278835 | `1.0028122307540739` | False | 0/9 |

`ci_hi` moves by `0.100` across draws at n=9 and by `0.0108` at n=8. **Both
aggregate verdicts are nevertheless draw-invariant**: n=9 registers `crosses:
false` on all three draws, n=8 registers `true` on all three, clearing `floor_1`
by `0.0432 / 0.0516 / 0.0408`. The it.8 disagreement of section 4 is therefore
**not** an artifact of the pinned draw — it reproduces on two draws that did not
exist when it.8 ran.

### 2.2 The answer to the question the brief posed

The brief named two outcomes. **It is the first one**: the crossing verdict is
stable across eval draws, so the per-cell reading is a property of the trained
model and not a reading of `seed=12345`. **No rate published this round has to be
withdrawn on eval-draw grounds.**

What this does **not** buy: `n_eff` for the eval is now **3, not 1**, and three is
not many. The aggregate CI at `:899-909` still pools over *training* seeds only
and still carries **zero** eval-draw variance component — section 2.1's `0.100`
spread in `ci_hi` is invisible to every interval this instrument publishes.
MARS's section 5.2 bound on what the round may write is narrowed by this
measurement, not lifted, and where it lands in the ranking is not this office's
call.

---

## 3. EXPERIMENT B — **`softmax` IS NOW PAIRED ON SEEDS 8-15, SAME PROCESS, SAME DRAWS**

MARS's fourth strike `[READ V20_R15_IT9_MARS.md:284-312]`: `softmax` had never run
on seeds 8-15, so *"7 of 8 fresh against 0 of 8"* paired fresh cells against stale
ones from a different process and, for `v15_r1.jsonl`, a different device.

Nine `softmax` cells (seed 0 control plus the eight fresh) trained in the **same
process, on the same device, against the same three eval draws** as the `arm_pl`
cells above:

| seed | @12345 | @12346 | @20260902 | crosses |
|---|---|---|---|---|
| 0 (control) | 0.9734002295 | 0.9724055572 | 0.9770939120 | F F F |
| 8 | 1.0468461869 | 1.0498899205 | 1.0472007825 | F F F |
| 9 | 0.9532169464 | 0.9568123060 | 0.9558831429 | F F F |
| 10 | 0.9472752509 | 0.9409352719 | 0.9555039967 | F F F |
| 11 | 1.0043084047 | 0.9972035764 | 1.0060955039 | F F F |
| 12 | 0.9712421074 | 0.9638697662 | 0.9859121295 | F F F |
| 13 | 0.9394588240 | 0.9443989053 | 0.9515144735 | F F F |
| 14 | 0.9546270607 | 0.9681587434 | 0.9754468244 | F F F |
| 15 | 0.9292912891 | 0.9228563109 | 0.9255687737 | F F F |

**0 of 8 fresh `softmax` cells cross, on every draw.** The contrast is now paired:
**8 of 9 `arm_pl` against 0 of 9 `softmax`, same seeds, same draws, same process,
`device="cuda"` on both sides.**

The separation is wide and has no overlap. Worst crossing `arm_pl` cell over all
draws is `0.686874`; best `softmax` cell over all draws is `0.922856`; **gap
`0.235982`**, with `floor_1 = 0.7071067811865476` sitting inside it. MARS called
this the weakest of his four strikes and predicted no plausible draw would
cross — **he was right, and the strike is now repaired rather than merely
conceded.** Two fresh `softmax` cells (seeds 8 and 11) read **above 1.0**, which
the stale seeds 0-7 never did (`0.9388-0.9734`); the fresh spread `0.9229-1.0499`
is `1.7x` the stale spread, so pairing was not free of information even though it
did not change the outcome.

---

## 4. EXPERIMENT C — THE it.8 AGGREGATE, BOTH COMPUTATIONS, REPRODUCED

Recomputed with `statistics.fmean` / `statistics.stdev` and
`scipy.stats.t.ppf(0.975, df=n-1)`, the estimator at `scripts/v15_r1.py:899-909`,
`crosses = bool(m + half < floor1)`:

```
JOURNALLED   {"n":9,"mean":0.7175018067286933,"sd":0.21245444406217273,
              "ci_lo":0.5541948645454026,"ci_hi":0.8808087489119841,"crosses":false}
RECOMPUTED9  {"n":9,"mean":0.7175018067286933,"sd":0.21245444406217273,
              "ci_lo":0.5541948645454026,"ci_hi":0.8808087489119841,"crosses":false}
RECOMPUTED8  {"n":8,"mean":0.6469670834460266,"sd":0.02029964404207494,
              "ci_lo":0.629996156325216, "ci_hi":0.6639380105668372,"crosses":true}
BITWISE mean/sd/ci_hi/ci_lo: True True True True
cells below floor: 8 / 9
n=8 ci_hi clears floor_1 by: 0.04316877061971036
```

**n=9 reproduces the journalled `mean`, `sd`, `ci_lo` and `ci_hi` bitwise and
returns `crosses=False`. n=8 without seed 9 returns `ci_hi =
0.6639380105668372` and `crosses=True`.** Both of MARS's numbers at
`V20_R15_IT9_MARS.md:124-125` are confirmed to the last digit, including the
`0.0432` clearance.

The pooled `sd` ratio is likewise confirmed: eight cells span `0.6249-0.6806` with
`sd = 0.0202996440`; the nine-cell pooled `sd` is `0.2124544441`, a factor of
**10.47**.

**This office does not rule on whether seed 9 should be excluded.** The
arithmetic above is the whole of what MERCURY is entitled to say about it.
Section 2.1 adds one fact the ruling office may want: the flip is
**draw-invariant** — n=9 says `false` and n=8 says `true` on all three eval
draws, so whatever the exclusion question is, it is not a question about which
eval batch was pinned. An exclusion decided **after** seeing that it flips the
verdict is the catalogued class this repository refuses, and that ruling belongs
to JUPITER or the Inspector, not here.

---

## 5. TEST TALLY — RED FIRST, VERBATIM

`tests/mercury/test_v20_r15_it10_eval_independence.py`. First run, **before** any
cell was trained:

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it10_eval_independence.py -q
FAILED ...::test_every_arm_pl_cell_is_scored_on_more_than_one_eval_draw
E  AssertionError: arm_pl seeds scored on ONE eval draw only (n_eff=1, no binomial
E  rate licensed): [0, 8, 9, 10, 11, 12, 13, 14, 15]; eval seeds seen per cell:
E  {0: [12345], 8: [12345], 9: [12345], 10: [12345], 11: [12345], 12: [12345],
E   13: [12345], 14: [12345], 15: [12345]}
FAILED ...::test_the_fresh_seeds_have_a_softmax_control_on_the_same_seeds
E  AssertionError: softmax cells exist for seeds [0, 1, 2, 3, 4, 5, 6, 7]; the fresh
E  seeds [8, 9, 10, 11, 12, 13, 14, 15] have no softmax control anywhere in results/
FAILED ...::test_it8_aggregate_verdict_agrees_with_its_own_cells
E  AssertionError: 8/9 cells sit below floor_1=0.7071067811865476 but the agg row
E  registers crosses=False (sd=0.21245444406217273, ci_hi=0.8808087489119841)
3 failed, 1 passed in 1.88s
```

The one pass is the **GREEN control** —
`test_control_the_it8_aggregate_is_reconstructible_bitwise` — which had to pass
first, because RED-3 is a claim about arithmetic on a journal, and a journal that
cannot be recomputed cannot be caught disagreeing with itself.

After the run:

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it10_eval_independence.py -q
FAILED ...::test_it8_aggregate_verdict_agrees_with_its_own_cells
1 failed, 3 passed in 2.13s
```

**RED-1 and RED-2 are repaired by measurement.** RED-3 stays RED and cannot be
repaired by measurement: `results/v20_r15_it8_armpl_b.jsonl` is a written record,
its `t="agg"` row disagrees with its own `t="cell"` rows, and the only routes are
a re-run under a pre-registered rule or a ruling that the aggregate is the
verdict. Neither is MERCURY's.

---

## 6. WHAT DID NOT HAPPEN

1. **`scripts/v15_r1.py` was not edited.** `git status --porcelain` on it is
   empty and `instrument_hash` is identical across both headers (section 0.1).
2. **`results/v17k_r4_retake.jsonl` was not written.** It was opened read-only
   for the `softmax` seed-0 control at section 1 and for MARS's seed-range check.
   `[RUN] sha256sum results/v17k_r4_retake.jsonl` ->
   `26fb180be89ffd61ecfb1fa08c80472cfd2de0d9b319057217ea6cf4c966457e`, which is
   the `26fb180b...` the brief froze it at; `git status --porcelain` on it is
   empty.
3. **Nothing touched Kaggle. No git write of any kind.** All 18 cells ran LOCAL
   on `cuda`.
4. **No moons were dispatched.** The work was three experiments over one harness
   plus one arithmetic reproduction; splitting it would have cost more context
   transfer than GPU time.

## 7. LIMITS

The three eval draws make `n_eff = 3`, not `n_eff` unbounded; the aggregate
interval at `scripts/v15_r1.py:899-909` still carries **no** eval-draw variance
component, and section 2.1's `0.100` spread in `ci_hi` across draws is invisible
to every interval this instrument publishes — a design fix, not a measurement
one. The two additional eval seeds (`12346`, `20260902`) were fixed before any of
them was scored and are not otherwise privileged; three draws rule out a verdict
that hinges on one draw, not finer structure in the draw distribution. The
re-score retrains rather than reloads, so it is a statement about models
bitwise-identical to the journalled ones (section 1, 10/10) and not about the
literal tensors it.8 held. The `softmax` pairing in section 3 is on the same
process, device and draws as the `arm_pl` cells, but the `arm_pl` numbers
themselves are retrains rather than the it.8 tensors — the bitwise control is
what licenses reading them together. Section 4's n=8 computation removes seed 9
**post hoc** and is reported as arithmetic, not as a verdict; the exclusion
question is untouched by this office. `36.05 s` is wall-clock `time.time()`
around the whole loop and is **un-synchronised** (L-9), so it is an
order-of-magnitude price and not a budget. `frac_gate_annihilated` reads `0.0` on
every `arm_pl` cell on every draw, confirming MARS's section 7.2 — it
discriminates nothing here and is not offered as a coordinate.
