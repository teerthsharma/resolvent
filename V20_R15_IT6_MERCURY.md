# V20 R15 it.6 — MERCURY (LESTRADE). THE ROUND'S FIRST DECISIVE EXPERIMENT, RUN.

`arm_smprime`, eight fresh seeds **8–15**, plus seeds **0 and 1 as the in-run
reproduction control**, same BED-M cell as the retake, on this box's certified
RTX 4060. **Ten cells, one process, one regime.** The cells are filed to a new
journal; **no frozen file was appended to.**

**This office does not read the outcome.** VENUS's §2.2 prediction and MARS's
`_0step` predictor are scored at it.29 and the row is VENUS's. What follows is
the columns and the price.

**No git write. Nothing touched Kaggle — the run is local, on this box's own
4060.**

---

## 0. THE COMMAND, VERBATIM

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u scripts/v15_r1.py \
  --device cuda --arms arm_smprime \
  --seeds 0 1 8 9 10 11 12 13 14 15 \
  --tag v20_r15_it6_seeds8_15
```

**The instrument was not modified.** `scripts/v15_r1.py:547` already takes
`--seeds` as `nargs="+"`; `:552` takes `--arms`; `:558` takes `--tag`, and
`:578-579` open `results/{tag}.jsonl` in append mode. `--n-train 2048`,
`--n-eval 4096`, `--steps 150` are the defaults (`:548-550`) and are the
retake's values. **No new runner was built and none was needed.** This is the
same command shape `V17_R4_RETAKE_PRICE.md:80-83` prescribes for the 24 retake
cells, with the arm list narrowed and the seed list extended.

**Why seeds 0 and 1 ride along.** Control 1. `scripts/v15_r1.py:231` calls
`torch.manual_seed(seed)` immediately before `make_arm`, and `:801` repeats it
for the 0-step control, so a cell's init is a function of its own seed and not
of the arm list or the seed order. That is the property that makes seeds 0–1
re-runnable inside a one-arm process, and it is the property the control tests.

---

## 1. THE REGIME, AND THAT NO FROZEN FILE MOVED

**New journal:** `results/v20_r15_it6_seeds8_15.jsonl`.
**`results/v17k_r4_retake.jsonl` was opened read-only and is unchanged** — 424
lines before the run and 424 after, `git status --porcelain` does not list it.
Clause (b)'s citations at `:161` and `:25` do not move.

Header of the new file, `results/v20_r15_it6_seeds8_15.jsonl:1` `[READ]`:

```
device: cuda   deterministic_algorithms: true   deterministic_warn_only: true
cublas_workspace_config: ":4096:8"   steps: 150   n_train: 2048   n_eval: 4096
floor_1: 0.7071067811865476   torch: "2.5.1+cu121"   threads: 8
instrument_hash: 5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309
arms: ["arm_smprime"]   seeds: [0,1,8,9,10,11,12,13,14,15]
when: "2026-09-02 02:28:38"
```

**Every regime field is identical to `results/v17k_r4_retake.jsonl:1`**, the
instrument hash included — same build of the script, same flags, same device,
same floor, same shape. The only two header fields that differ are `arms` and
`seeds`, which are what the experiment varies. This is asserted rather than
claimed in prose: `test_regime_matches_the_retake_field_for_field` diffs
seventeen fields and requires an empty diff.

---

## 2. THE TEN CELLS

`arm_smprime`, `t*=2`, `s=64`, `d=24`, `n_train=2048`, `n_eval=4096`,
`steps=150`, `lr=0.02`, `d_model=16`, `route=product` throughout.
`beta`, `qk`, `g`, `route` and `n_zero_gates` are read from the cell's own
identity manifest (`manifest.smp_values`), which `scripts/v15_r1.py` fills from
the **trained** module — not from the `t="bind"` row, which carries the
constructor's values.

| seed | `eval_nrmse` | `frac_gate_annihilated` | `frac_gate_annihilated_0step` | `a_hat_min` | `a_hat_max` | `lambda_hat` | `unit_root` | `n_zero_gates` | `secs` | `beta` | `qk` | `g` | `route` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** (control) | 0.926082 | 0.5032958984375 | 0.5048828125 | 0.0 | 1.0 | −inf | True | 4123 | 16.104 | 0.7325604557991028 | 1.276558756828308 | 1.319505214691162 | product |
| **1** (control) | 0.881247 | 0.5032958984375 | 0.611328125 | 0.0 | 1.0 | −inf | True | 4123 | 15.957 | 0.8969751 | 1.3205688 | 1.234473467 | product |
| **8** | 0.852061 | 0.5032958984375 | 0.901123046875 | 0.0 | 1.0 | −inf | True | 4123 | 16.670 | 0.6961916089 | 1.241755486 | 1.035467148 | product |
| **9** | 0.902522 | 0.5032958984375 | 0.5103759765625 | 0.0 | 1.0 | −inf | True | 4123 | 15.938 | 0.8696166873 | 1.20524013 | 1.304651141 | product |
| **10** | 0.892747 | 0.5032958984375 | 0.62890625 | 0.0 | 1.0 | −inf | True | 4123 | 15.907 | 0.7277029753 | 1.322333574 | 1.134657502 | product |
| **11** | 1.120603 | **0.99462890625** | 0.99462890625 | 0.0 | **0.06994166225** | −inf | **False** | **8148** | 17.647 | **−0.06023420021** | 2.042710781 | 1.0 | product |
| **12** | 0.919852 | 0.5032958984375 | 0.380859375 | 0.0 | 1.0 | −inf | True | 4123 | 22.069 | 0.8679240942 | 1.559713602 | 1.365590453 | product |
| **13** | 1.203324 | **0.976806640625** | 0.9595947265625 | 0.0 | **0.7774731517** | −inf | **False** | **8002** | 17.931 | **1.989504933** | 2.03054595 | 1.120093703 | product |
| **14** | 0.909944 | 0.5032958984375 | 0.3983154296875 | 0.0 | 1.0 | −inf | True | 4123 | 21.979 | 0.7842412591 | 1.652458668 | 1.398792505 | product |
| **15** | 0.891047 | 0.5032958984375 | 0.7315673828125 | 0.0 | 1.0 | −inf | True | 4123 | 18.258 | 1.000811815 | 1.23466444 | 1.048431158 | product |

### 2.1 Mechanical reads off those columns — no adjudication

These are comparisons the file's own header licenses, and they are stated as
readings. **Whether they confirm or refute anything filed at it.5 is VENUS's
row at it.29 and this office does not score it.**

- **Crossings, by the shipped `eval_nrmse <= floor_1` comparison against the
  header's `floor_1 = 0.7071067811865476`: `0` of the eight fresh seeds.**
  The smallest fresh `eval_nrmse` is `0.852061` (seed 8), `0.144954` above the
  floor.
- **`frac_gate_annihilated` on the fresh eight takes three values, two of which
  occur nowhere among the retake's eight `arm_smprime` cells:**
  `0.5032958984375` (six seeds — 8, 9, 10, 12, 14, 15), `0.99462890625`
  (seed 11), `0.976806640625` (seed 13). **`0.4967041015625` occurs zero times
  here and `0.0` occurs zero times here.**
- **`lambda_hat` is `−inf` on all ten cells**, the two `unit_root == False`
  cells included.
- **`a_hat_min` is `0.0` on all ten.** `a_hat_max` is `1.0` on eight and departs
  from it only on the two high-`frac_gate_annihilated` cells.
- **`frac_gate_annihilated_0step` on the fresh eight ranges `0.380859375`
  (seed 12) to `0.99462890625` (seed 11). None is below `0.15`**, so MARS's
  `_0step < 0.15` predictor issues zero positive predictions over this batch.
- **`n_zero_gates` is bimodal at `4123` / `{8002, 8148}`**, tracking
  `frac_gate_annihilated` exactly.

---

## 3. CONTROL 1 — THE INVOCATION REPRODUCES

| seed | retake `eval_nrmse` | this run's `eval_nrmse` | equal? |
|---|---|---|---|
| 0 | `0.926082` | `0.926082` | **bitwise** |
| 1 | `0.881247` | `0.881247` | **bitwise** |

Both reproduce to the digits the round cites, and the test compares the full
stored floats, not the printed ones.
`frac_gate_annihilated_0step` also reproduces MARS's table
(`0.5048828125` / `0.611328125`), as does trained `frac_gate_annihilated`
(`0.5032958984375` on both). The invocation is the retake's instrument under
the retake's regime, and the eight fresh cells sit in the same frame.

---

## 4. THE PRICE — MEASURED, AND IT OVERRAN

**Estimate: `129` GPU-s for eight cells (`8 × 16.161`, VENUS §2.2).
Measured: `146.399` s of cell time for the eight fresh seeds. `+17.112 s`,
`+13.2 %`.** Against the struck six-cell probe basis (`8 × 15.970 = 127.760`)
the overrun is `+18.639 s`, `+14.6 %`.

| quantity | value | source |
|---|---|---|
| fresh-seed cell time (seeds 8–15) | **146.399 s** | sum of eight `secs` fields |
| all ten cells | **178.460 s** | sum of ten `secs` fields |
| mean per cell, ten cells | **17.846 s** | `t="wall"` → `secs_per_run_by_arm.arm_smprime` |
| process wall clock | **185.746 s** (3 m 05.7 s) | `t="wall"` → `secs` |
| non-cell fixed cost | **7.286 s** | `185.746 − 178.460` `[DERIVED]` |
| peak process working set | 1.333 GiB | `t="wall"` |
| `cuda_max_memory_allocated` | 0.972 GiB | `t="wall"` |

**The priced 2.2 minutes was a 3.1-minute run.** The overrun is not fixed cost —
fixed cost came in at `7.286 s` against the retake's `10.196 s`, and the retake
binds three arms to this run's one. It is per-cell: **`secs` is not stationary
inside one process.** The ten cells span `15.907 s` (seed 10) to `22.069 s`
(seed 12), a **1.387×** spread, and the two slowest are both in the back half.
A per-cell mean taken from the front of a run under-prices the back of it,
which is the same shape of error as pricing 24 arena cells off a six-cell
probe. **The per-cell figure for `arm_smprime` on this box now reads `16.161 s`
over the retake's 8 cells and `17.846 s` over these 10, and it should be quoted
with the observed `15.9–22.1 s` range attached rather than as a point.**

---

## 5. TEST-BOUND, RED FIRST

`tests/mercury/test_v20_r15_it6_seeds8_15.py`. Four nodes: regime match against
the retake header field-for-field, the seed-0/1 bitwise control, seed coverage,
and column presence. **RED first, verbatim, taken while the run was still
producing cells (five of ten present):**

```
E       AssertionError: assert [0, 1, 8, 9, 10] == [0, 1, 8, 9, 10, 11, 12, 13, 14, 15]
E
E         Right contains 5 more items, first extra item: 11
E         Use -v to get more diff

tests\mercury\test_v20_r15_it6_seeds8_15.py:63: AssertionError
__________ test_every_cell_carries_the_columns_the_report_publishes ___________
E       AssertionError: {'0': ['n_zero_gates', 'beta', 'qk', 'g', 'route'],
E        '1': ['n_zero_gates', 'beta', 'qk', 'g', 'route'],
E        '10': ['n_zero_gates', 'beta', 'qk', 'g', 'route'],
E        '8': ['n_zero_gates', 'beta', 'qk', 'g', 'route'],
E        '9': ['n_zero_gates', 'beta', 'qk', 'g', 'route']}

tests\mercury\test_v20_r15_it6_seeds8_15.py:72: AssertionError
=========================== short test summary info ===========================
FAILED tests/mercury/test_v20_r15_it6_seeds8_15.py::test_ten_cells_seeds_0_1_and_8_through_15
FAILED tests/mercury/test_v20_r15_it6_seeds8_15.py::test_every_cell_carries_the_columns_the_report_publishes
2 failed, 2 passed in 0.70s
```

**The second RED was a real defect in this office's first draft and is recorded
rather than overwritten:** the node looked for `beta`, `qk`, `g`, `route` and
`n_zero_gates` as **cell-record** fields. They are not. They live on the cell's
identity manifest under `manifest.smp_values`, filled from the trained module.
The nearest wrong answer — reading them off the `t="bind"` row — would have
published the constructor's values as the cell's: that row reads `beta = 0.0`
for every seed, against a trained `0.7326` on seed 0 and `−0.0602` on seed 11.

**GREEN after the run completed** `[RUN]`:

- `python -m pytest tests/mercury/test_v20_r15_it6_seeds8_15.py -q` → **4 passed in 0.24s**
- `python -m pytest tests/mercury/ -q` → **93 passed in 11.81s** (89 at it.3 + 4 new)

---

## 6. PROVENANCE

| | |
|---|---|
| box | NVIDIA GeForce RTX 4060 Laptop GPU, `0 MiB / 8188 MiB` used and `0 %` utilisation immediately before launch `[RUN] nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader` |
| instrument | `scripts/v15_r1.py`, working tree, **unmodified by this node**, `instrument_hash 5d41a63d…9a309` on all ten cell records — the retake's hash |
| process exit | `0` |
| writing git commands | **none** |
| Kaggle | **not touched** — no kernel push, no dataset upload, no CLI call. The run is local |
| files created | `results/v20_r15_it6_seeds8_15.jsonl`, `tests/mercury/test_v20_r15_it6_seeds8_15.py`, `V20_R15_IT6_MERCURY.md` |
| files appended | `house-events.jsonl` (harness convention) |
| frozen files touched | **none.** `results/v17k_r4_retake.jsonl` read-only, 424 lines before and after, absent from `git status --porcelain` |
| not this node's | `M pytest.ini` was already modified at start (another node's `norecursedirs kaggle` edit) and was not touched here |

**Not this office's to call:** whether the eight fresh seeds confirm or refute
VENUS's §2.2 filing or MARS's `_0step` predictor. The columns are above; the
scoring is VENUS's row at it.29.
