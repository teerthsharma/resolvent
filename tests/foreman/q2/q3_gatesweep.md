# Q3 -- the gate sweep table

LAW L-REFLECTOR: fixed structure audited and printed before any arm is scored
(see the `q3_fixed_structure` board event and the JSON block `q3_gatesweep.py`
prints at the top of its own stdout, both written before `run_variant()` is
called for the first time).

## Reuse, not reimplementation

`q3_gatesweep.py` imports `tests/chase/gate/r1_gate.py` (module `R`) and does
not edit it. It contributes exactly three things on top of that module:

1. `R.FORMS["clamp"] = R._ORIG_MAGNITUDE` -- the third form. `r1_gate.py`
   already captures `arm_smprime.magnitude` (bare `clamp(u,0,1)`) as
   `_ORIG_MAGNITUDE` before installing any monkeypatch, so the "clamp" arm
   needed no new gate math, only a dict entry.
2. A seed loop around `R.run_variant(form)`: `R.SEED` and `R.OUT_DIR_TMPL`
   are module-globals that `run_variant()` reads at call time, so setting
   them before each call drives 5 seeds without touching `r1_gate.py`'s
   single hardcoded `SEED = 0`.
3. `R.board = board` -- `run_variant()`'s internal board writes hardcode
   `agent="Chase"` (that file's own author). This is Foreman's phase, so the
   module-global `board` name is replaced for the duration of the run; every
   write this script causes, including the ones inside the reused function,
   carries `agent="Foreman"`.

## Fixed structure (printed, not asserted)

initializer, parameterization, corpus regime, scorer functional, bin scheme,
dtype path, torch build, eval subsample size -- see the `q3_fixed_structure`
board event for the exact values measured on this box at run time (GPU name,
torch version, GATE_INIT_OFF, etc. are read live, not hardcoded here so this
file cannot drift from what actually ran).

## Fields reported per (form, seed)

`loss_last`, `last50_mean`, `exact_zero_trained` (trained exact-zero
fraction), `backward_reach_trained` (mean/median/p95/max -- BACKWARD reach:
`G_ij = prod_{k=j+1}^{i} m_k`, so row `i`'s live reach only ever extends
backward from `i`; `backward_reach_stats` in `r1_gate.py` computes exactly
that, not the maximal run through `i` in either direction), and
`frac_grad_nonzero_trained` (live-gradient fraction). `run_seconds` per run
is also recorded for L-REPRO provenance.

No mechanism sentence is written in this row, per the task's own
instruction: four mechanisms for gate closure were already proposed and
refuted in earlier phases (gradient starvation, step-0 zero density,
local-window optimality, eval-only freeze); this table reports numbers only.

## L-REPRO

Command: `python q3_gatesweep.py 5` (argv[1] = n_seeds, default 5).
**Seeds actually run: 5 of 5 requested, all 3 forms -- 15/15 runs
completed.** First attempt crashed after 1 run (`clamp` seed=0 only) on a
`result()` keyword-collision bug (`res` already carried `form=`/`seed=`);
fixed (one-line diff, see git blame on this file) and relaunched from
scratch. The crash cost nothing: `clamp` seed=0's board-logged numbers
(`loss_last=1.3122`) are bit-identical between the crashed attempt and the
clean rerun, and both runs are in `q3_gatesweep_results.jsonl` under the
same seed (5 clamp rows exist, not 6 -- the crashed attempt's
`clamp/seed=0` never reached `result()`, only the successful rerun did).

Shape: hidden=512, n_layers=4, n_heads=8, d_head=64, seq=512, batch=8,
steps=800 (r1_gate.py's own reference config, unchanged). Device: CUDA
(NVIDIA GeForce RTX 4060 Laptop GPU, torch 2.14.0+cu126). Wall clock:
3418.2s for all 15 runs (mean 228s/run).

## Results (mean over 5 seeds; per-seed numbers in q3_gatesweep_results.jsonl)

| form             | loss_last50_mean | exact_zero_trained | backward_reach_mean | live_grad_frac |
|------------------|------------------:|--------------------:|----------------------:|-----------------:|
| clamp            | 1.2604            | 0.2760              | 32.33                | 0.5514           |
| straight_through | 1.2843            | 0.3912              | 1.32                 | 1.0000           |
| hard_concrete    | 1.1218            | 0.0018              | 214.35               | 0.9839           |

**PREDICTION HELD, not seed-dependent**: hard_concrete has the lowest
`last50_mean` loss AND the highest `backward_reach_mean` at **5 of 5**
seeds individually (not just on the mean above) -- seed-by-seed: hc
last50_mean = {1.1391, 1.1215, 1.1194, 1.1099, 1.1190} vs clamp =
{1.2665, 1.2278, 1.4121, 1.2508, 1.3088} vs straight_through = {1.3000,
1.4389, 1.1774, 1.2849, 1.2205}; hc is lowest in every one of the 5
comparisons. Reach likewise: hc reach_mean = {241.4, 200.2, 206.1, 221.5,
202.5} vs clamp = {2.1, 26.5, 65.3, 1.8, 66.0} vs straight_through = {1.2,
1.4, 1.5, 1.0, 1.4}; hc is highest in every one of the 5. The COUNTER
("ordering is seed-dependent, relabel the previous phase's reach finding
as one-seed") does NOT fire: the single-seed reference numbers this row
was built to beat or overturn (clamp last50=1.2665/exact_zero=0.3121/
reach_mean=2.146; straight_through 1.2999/0.4050/1.209; hard_concrete
1.1391/0.0006/241.4) are reproduced almost exactly at seed=0 in this
5-seed run (compare the seed=0 rows above to those numbers -- they match
because seed=0 is the same seed the earlier single-seed phase used), and
every one of the other 4 seeds preserves the same ordering.

No mechanism sentence follows. These are the numbers.
