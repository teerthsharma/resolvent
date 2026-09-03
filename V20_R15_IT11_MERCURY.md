# V20 R15 — it.11 — MERCURY (LESTRADE)

Branch `v17k-gate0`. **No git writes. Nothing touched Kaggle.** Every run LOCAL on
the 4060. New journals only:

* `results/v20_r15_it11_mercury_smprime.jsonl` (Experiment A)
* `results/v20_r15_it11_mercury_draw4.jsonl` (Experiment B)

`results/v17k_r4_retake.jsonl` was **read** and never opened for write; its
sha256 is re-verified in §5.

---

## 0. THE GAP WAS IN THIS OFFICE'S OWN it.10 COVERAGE

it.10 rescored `arm_pl` and `softmax`. `arm_smprime` was absent. The round
therefore held **18 draw-checked cells for W3/skyline against 0 for W1**, and the
single crossing cell in sixteen — `arm_smprime` seed 2,
`eval_nrmse = 0.20391993939877656` `[READ results/v17k_r4_retake.jsonl]`, the whole
of W1's remaining probability mass — had **never been scored on a second eval
draw**. The it.10 finding that made the round comfortable, *"the verdicts do not
flip"*, was established on the two arms that are **not** W1. VENUS named this the
cheapest open item in the round and she was right.

### 0.1 The route, unchanged from it.10 and for the same reason

`scripts/v20_r15_it11_mercury_smprime.py` is a **new file**. It does not edit
`scripts/v15_r1.py`; it loads it with `importlib.util.spec_from_file_location`
and calls the runner's **own** `train_one`, `gate_columns`, `gate_features`,
`recovered_gate`, `probe` and `nrmse`. The retrain is real — `models` is written
at `scripts/v15_r1.py:879` and read at no line of the file, established at it.10 —
so a second eval draw costs a retrain, not a forward pass.

### 0.2 PRICE, STATED BEFORE THE RUN

`arm_smprime` is the expensive arm. The estimate was written into the journal
header as `price_estimate_secs` **before** the run started
`[READ scripts/v20_r15_it11_mercury_smprime.py:34]`:

```
[DERIVED] sum of the 16 banked train `secs`
  retake seeds 0..7 : 16.164+16.032+14.852+15.977+16.527+16.464+16.660+16.611 = 129.287
  it.6   seeds 8..15: 16.670+15.938+15.907+17.647+22.069+17.931+21.979+18.258 = 146.399
  train subtotal                                                              = 275.686
  + module load, three eval-draw builds, 48 scorings with gate columns + probes ~ 26
  ESTIMATE                                                                    = 301.7 s
```

Prior spread on this office's estimates: it.6 came in `+13.2 %`, it.8 came in
`−3.7 %`. The actual and the delta are in §2.3.

**Seed order was `2` first, deliberately.** Seed 2 is simultaneously the round's
highest-value cell and one of its own bitwise controls, so a run cut off by the
wall clock would still have answered the question the round most needs answered.
It was not cut off.

---

## 1. CONTROL FIRST — **16 OF 16 BITWISE**, AND THAT IS ALL 16 BANKED CELLS

The 12345 column of the retrain reproduces the frozen journals to the bit. Eight
against `results/v17k_r4_retake.jsonl` (sha256
`26fb180be89ffd61ecfb1fa08c80472cfd2de0d9b319057217ea6cf4c966457e`, re-verified
this iteration), eight against `results/v20_r15_it6_seeds8_15.jsonl`:

```
== CONTROL: draw 12345 vs frozen journals, bitwise ==
seed  0  0.9260818361710341      0.9260818361710341      yes
seed  1  0.8812466551692475      0.8812466551692475      yes
seed  2  0.20391993939877656     0.20391993939877656     yes
seed  3  0.9162579895429029      0.9162579895429029      yes
seed  4  0.9119259828412737      0.9119259828412737      yes
seed  5  0.9181568234297154      0.9181568234297154      yes
seed  6  0.8932391322242402      0.8932391322242402      yes
seed  7  0.9267169213804543      0.9267169213804543      yes
seed  8  0.8520605074827049      0.8520605074827049      yes
seed  9  0.902522218182919       0.902522218182919       yes
seed 10  0.8927474103076622      0.8927474103076622      yes
seed 11  1.1206034735616488      1.1206034735616488      yes
seed 12  0.919852079772707       0.919852079772707       yes
seed 13  1.2033239267331008      1.2033239267331008      yes
seed 14  0.9099444169087955      0.9099444169087955      yes
seed 15  0.8910465662156185      0.8910465662156185      yes
bitwise 16/16

== EXPERIMENT A: arm_smprime, floor_1=0.7071067811865476 ==
```

it.10's control was 10 of 10 on two arms. This one is **16 of 16 on every banked
`arm_smprime` cell there is**, seed 2 included — so the re-score is scoring the
same models the round banked, and §2 is admissible.

---

## 2. EXPERIMENT A — `arm_smprime` ON THREE EVAL DRAWS. **48 OF 48. NO VERDICT FLIPS.**

```
seed | 12345 / 12346 / 20260902 | crossings | lambda_hat | lam_live | annih | a_max | unit_root | stable
   0 | 0.926082 / 0.940317 / 0.948329 | FFF | -inf | -0.019538 | 0.503296 | 1.000000 | True | yes
   1 | 0.881247 / 0.906922 / 0.921766 | FFF | -inf | -0.011799 | 0.503296 | 1.000000 | True | yes
   2 | 0.203920 / 0.208055 / 0.216517 | TTT | -0.795290470123291 | -0.795290 | 0.000000 | 0.540044 | False | yes
   3 | 0.916258 / 0.907486 / 0.917539 | FFF | -inf | -0.441142 | 0.496704 | 0.845433 | False | yes
   4 | 0.911926 / 0.908832 / 0.933585 | FFF | -inf | -0.047636 | 0.503296 | 1.000000 | True | yes
   5 | 0.918157 / 0.925854 / 0.931209 | FFF | -inf | -0.007938 | 0.503296 | 1.000000 | True | yes
   6 | 0.893239 / 0.892048 / 0.906607 | FFF | -inf | -0.000953 | 0.503296 | 1.000000 | True | yes
   7 | 0.926717 / 0.934802 / 0.959270 | FFF | -inf | -0.024266 | 0.503296 | 1.000000 | True | yes
   8 | 0.852061 / 0.863120 / 0.870645 | FFF | -inf | -0.031969 | 0.503296 | 1.000000 | True | yes
   9 | 0.902522 / 0.906232 / 0.919562 | FFF | -inf | -0.003480 | 0.503296 | 1.000000 | True | yes
  10 | 0.892747 / 0.894235 / 0.905551 | FFF | -inf | -0.025377 | 0.503296 | 1.000000 | True | yes
  11 | 1.120603 / 1.119289 / 1.136689 | FFF | -inf | -4.186897 | 0.994629 | 0.069942 | False | yes
  12 | 0.919852 / 0.923567 / 0.936482 | FFF | -inf | -0.013666 | 0.503296 | 1.000000 | True | yes
  13 | 1.203324 / 1.179629 / 1.222329 | FFF | -inf | -2.498878 | 0.976807 | 0.777473 | False | yes
  14 | 0.909944 / 0.918221 / 0.941339 | FFF | -inf | -0.007220 | 0.503296 | 1.000000 | True | yes
  15 | 0.891047 / 0.902008 / 0.910861 | FFF | -inf | -0.007249 | 0.503296 | 1.000000 | True | yes
cells complete 16/16, scorings 48, verdict flips 0
cross on ALL draws: [2]   cross on ANY draw: [2]
WALL 285.49s vs ESTIMATE 301.7s  delta -5.4%

== EXPERIMENT B: fourth eval draw 20260903 on arm_pl ==
rank | seed | VENUS ratio | 3-draw verdicts | draw4 nrmse | draw4 cross | flipped
   1 |   15 | 0.999 | TTT | 1.221843 | False | FLIP
```

**Seed 2 crosses on all three draws.** `0.203920 / 0.208055 / 0.216517` against
`floor_1 = 0.7071067811865476`. Its **worst** of the three, `0.216517`, is
`0.490590` below the floor `[DERIVED 0.7071067811865476 − 0.216517126655 = 0.490590]`;
the floor is `3.266×` that reading. On VENUS's own fragility measure the cell
scores `spread / margin = 0.012597 / 0.490590 = 0.0257` — against her most
fragile `arm_pl` cell at `0.999`, seed 2 is **39× less fragile** than the tightest
cell she ranked. The one crossing cell in sixteen was
single-draw when this iteration opened and it is now `n = 3`, and the crossing
does not depend on the draw.

**The fifteen non-crossing cells also do not flip.** Their draw-to-draw spread,
`(max − min) / value@12345`, runs from `1.097 %` (seed 3) to `4.598 %` (seed 1);
the movement is not uniformly upward — seeds 3, 6 and 13 read **lower** at 12346
than at 12345. Every one of them stays above the floor by at least `0.145`
(`seed 8`, `0.852061 − 0.707107 = 0.144954`). Nothing
in `arm_smprime` is near the floor in the way `arm_pl` seed 15 is: the closest
non-crossing `arm_smprime` cell sits `0.145` out, against seed 15's `0.020`.

**The gate columns are draw-invariant too.** `unit_root` is `True` on **12 of 16**
cells (`0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15`) and `False` on **4**
(`2, 3, 11, 13`), the same value on all three draws for every cell. The count is
printed from the journal, not read off the table by eye. `frac_gate_annihilated` moves in the
fourth decimal (`0.503296 → 0.506226` at seed 0) and never changes which side of
anything it is on. `lambda_hat` is `-inf` on the 15 annihilating cells at every
draw, which is the path-product arm reporting a genuine zero hop, not an
instrument failure — `lambda_hat_live` beside it is finite and stable.

**Seed 2 is the arm's only cell with `frac_gate_annihilated = 0.000000`**, on all
three draws, with `a_hat_max = 0.540044 < 1` and `unit_root = False`. It is the
only `arm_smprime` cell that annihilates nothing, and it is the only one that
crosses. This office reports the coincidence and does not interpret it.

### 2.3 PRICE — ESTIMATE `301.7 s`, ACTUAL `285.49 s`, DELTA `−5.4 %`

```
WALL 285.49s vs ESTIMATE 301.7s  delta -5.4%   [RUN t="wall", 16 cells x 3 draws = 48 scorings]
```

Prior spread on this office's estimates was `+13.2 %` (it.6) and `−3.7 %` (it.8);
`−5.4 %` sits inside it. Against it.10's `36.05 s` for 18 cells, this run is
`7.9×` the cost for `2.7×` the scorings, which is the `arm_smprime` per-cell
price and not a surprise.

### 2.4 REGIME DIFF, PRINTED

```
DIFF arms                       ["arm_pl", "softmax"]              ["arm_smprime"]
DIFF price_estimate_secs        "<absent>"                         301.7
DIFF seeds                      [0, 8, 9, 10, 11, 12, 13, 14, 15]  [2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
DIFF tag                        "v20_r15_it10_mercury_rescore"     "v20_r15_it11_mercury_smprime"
DIFF when                       "2026-09-02 03:25:18"              "2026-09-02 03:38:52"
DIFF why                        "re-score kept-nothing models on 3 eval draws; softmax paired on 8..15" "arm_smprime on 3 eval draws; W1 had 0 draw-checked cells against W3's 18"
-- header keys: 25 union, 19 identical, 6 differ
```

**25 union keys, 19 identical, 6 differ**, and of the 6 only `arms` and `seeds`
name a measurement condition; `price_estimate_secs` and `why` are fields this
file adds, `tag` and `when` are identity. `device`, `steps`, `n_train`, `n_eval`,
`s`, `d`, `d_model`, `lr`, `threads`, `torch`, `floor_1`, `instrument_hash`,
`eval_seeds`, `deterministic_algorithms`, `deterministic_warn_only`,
`cublas_workspace_config`, `t_star`, `task` and `t` are the identical 19.

---

## 3. EXPERIMENT B — **THE CONTROL FAILED. THE CELLS ARE INADMISSIBLE.**

B ran and it is reported as a **failed control, not as a test of VENUS's
pre-registration.** The fourth-draw run re-scored eval seed `12345` alongside
`20260903` precisely so the same bitwise check that licensed §2 would license §3.
It did not.

```
== EXPERIMENT B: fourth eval draw 20260903 on arm_pl ==
rank | seed | VENUS ratio | 3-draw verdicts | draw4 nrmse | draw4 cross | flipped
   1 |   15 | 0.999 | TTT | 1.221843 | False | FLIP
       control 12345: 1.22503261307929 vs it.10 0.6805785772206486 -> DIFFERS
   2 |   12 | 0.589 | TTT | 1.183994 | False | FLIP
       control 12345: 1.1813421380511242 vs it.10 0.6762082068688114 -> DIFFERS
   3 |    9 | 0.493 | FFF | 1.145191 | False | no
       control 12345: 1.1604131035965342 vs it.10 1.281779592990027 -> DIFFERS
   4 |   14 | 0.266 | TTT | 1.232735 | False | FLIP
       control 12345: 1.2003961188593775 vs it.10 0.6338928504917678 -> DIFFERS
   5 |   10 | 0.180 | TTT | 1.174385 | False | FLIP
       control 12345: 1.162950511677486 vs it.10 0.6398332532283637 -> DIFFERS
   6 |   13 | 0.180 | TTT | 1.138583 | False | FLIP
       control 12345: 1.155733256248327 vs it.10 0.6348616577583769 -> DIFFERS
   7 |    0 | 0.165 | TTT | 1.138784 | False | FLIP
       control 12345: 1.1768841536822816 vs it.10 0.6446726192039927 -> DIFFERS
   8 |    8 | 0.158 | TTT | 1.206166 | False | FLIP
       control 12345: 1.2001336448918893 vs it.10 0.6248685523735131 -> DIFFERS
   9 |   11 | 0.105 | TTT | 1.174695 | False | FLIP
       control 12345: 1.1871008720041685 vs it.10 0.6408209504227382 -> DIFFERS
cells flipped on draw 4: [15, 12, 14, 10, 13, 0, 8, 11]
WALL 19.78s vs ESTIMATE 26.0s  delta -23.9%
```

**Nine of nine `arm_pl` cells fail the 12345 control**, and they fail in the same
direction and by the same order of magnitude: it.10 read `0.63`–`0.68` for the
eight crossing cells, this run reads `1.15`–`1.23` for all nine. The header
records why the two runs are not comparable:

```
[RUN] header instrument_hash
  it.8   5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309
  it.10  5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309
  it.11 A (arm_smprime, results/v20_r15_it11_mercury_smprime.jsonl)
         5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309
  it.11 B (arm_pl,      results/v20_r15_it11_mercury_draw4.jsonl)
         945d850d4b35c1a721cb3bab96b682b26b3ad444912a6a73d4eb76fb6afb0045   <- MOVED
```

**A and B are the same file** apart from `EVAL_SEEDS`, `SEEDS`, `kind`, `arms`,
the output path, `tag`, `why` and `PRICE_ESTIMATE` — B was generated from A by
`sed` and the diff was printed to confirm it. Both load `scripts/v15_r1.py` by
the same `importlib` call, and `scripts/v15_r1.py` is clean against HEAD:

```
[RUN] git status --porcelain -- scripts/v15_r1.py
(empty)
[RUN] three fresh processes, same import, hash + its two components
  5d41a63d57671725  file 0f7fc06e9981db1d  reach 4c9c9e2a706dbf4c  ceq/arm_pl.py = repo root
  5d41a63d57671725  file 0f7fc06e9981db1d  reach 4c9c9e2a706dbf4c  ceq/arm_pl.py = repo root
  5d41a63d57671725  file 0f7fc06e9981db1d  reach 4c9c9e2a706dbf4c  ceq/arm_pl.py = repo root
```

So the hash is stable across processes **now**, and it was `945d850d…` in the B
process at `03:44`. `instrument_manifest` has two components
`[READ kaggle/snapshot/repo/scale/identity_manifest.py:171-201]` — `file`, the
sha256 of `scripts/v15_r1.py`'s own bytes, and `reach`, a bytecode fingerprint
over eleven callables that live in **other** modules (`ceq/arm_pl.py`,
`ceq/arm_smprime.py`, `scale/…`). A `reach` component is exactly what moves when
a module the instrument imports changes with the instrument's own bytes
untouched.

**This office does not have the wall clock to establish which module moved, and
does not assert one.** What it asserts is the consequence:

1. **Experiment B answers nothing about VENUS's §2.2 pre-registration.** Its
   nine cells were trained under an instrument the round has never scored
   against. The "flips" the table prints are flips against it.10 **models**, not
   against it.10 **draws**, and reading them as a fourth-draw result would be the
   same error MARS made at it.9 in the other direction. **VENUS's
   pre-registration stands untested; it is her row at it.29 either way.**
2. **Experiment A is unaffected and stays admissible.** Its header carries
   `5d41a63d…` and its 12345 column is bitwise against sixteen banked cells —
   the strongest control this office has run. A cannot have been scored under a
   moved instrument and still reproduce sixteen frozen values to the bit.
3. **Any wing reporting an `instrument_hash` from a fresh process should print it
   against `5d41a63d…` rather than assume it.** That check cost this office
   nothing and it is the only reason B was caught instead of published.

The B journal is left on disk, uncorrected, as the record of the failure.

---

## 4. TEST-BOUND — RED FIRST, VERBATIM

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it11_smprime_draws.py -q   (before the run)
FAILED ...::test_journal_exists_and_is_not_the_frozen_one
FAILED ...::test_header_keeps_the_instrument_hash
FAILED ...::test_control_column_reproduces_banked_bitwise
FAILED ...::test_seed_2_is_scored_on_all_three_draws
FAILED ...::test_every_banked_seed_on_every_draw
FAILED ...::test_crossing_verdict_is_computed_against_the_stated_floor
6 failed in 0.80s

[RUN] same command, after the run
6 passed in 0.31s
```

`test_header_keeps_the_instrument_hash` asserts `5d41a63d…` literally. It is the
assertion that would have caught §3's failure had it been written against the B
journal, and it is why the A journal can be trusted.

---

## 5. HYGIENE, AND WHAT THIS OFFICE DID NOT DO

```
[RUN] sha256sum results/v17k_r4_retake.jsonl
26fb180be89ffd61ecfb1fa08c80472cfd2de0d9b319057217ea6cf4c966457e   (frozen, matches)
[RUN] git status --porcelain -- scripts/v15_r1.py
(empty)
```

No git writes. Nothing touched Kaggle; both runs local on the 4060, `device=cuda`
in every row. `scripts/v15_r1.py` unedited and unimported-by-name. Two new
journals, no frozen file opened for write. Four moons were not needed and were
not spawned.

**Limits.** `n = 3` eval draws is not `n = \u221e`, and this iteration adds no
fourth draw to anything — B's fourth draw is void for the reason in §3, so the
round still has three draws everywhere it has any. The A control proves the
retrain reproduces the banked **models**; it does not prove the training itself
is draw-independent, which no experiment here tested. The `arm_smprime` price is
measured on this box only. The mechanism behind B's moved `instrument_hash` is
**not** established — the two candidate components are named and neither is
asserted, and no claim here rests on which it was. This office reports cells and
does not adjudicate whether any prediction held: VENUS's §2.2 pre-registration is
scored on her row at it.29, and the seed-9 exclusion this office refused to rule
on at it.10 is still refused.
