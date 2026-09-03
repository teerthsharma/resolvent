# V20 R15 it.18 — MERCURY (LESTRADE, execution)

Branch `v17k-gate0`. No git writes, no Kaggle, no edit to `scripts/v15_r1.py`,
no write to `results/v17k_r4_retake.jsonl`.

## The deliverable was already on disk

`tests/mercury/arena_rig.py` is 11,924 bytes / 286 lines, and
`tests/mercury/test_v20_r15_it15_arena_rig.py` runs **17 passed in 0.28s**.

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it15_arena_rig.py -q
      -> 17 passed in 0.28s
```

The it.18 brief states it.17 "filed a skeleton at 697 bytes and died before the
module." The disk contradicts that on both files. it.17 wrote the module AND
revised the test; the 697-byte reading and the 6,601-byte test reading both
describe a state that no longer exists. **A stale file-size reading is not
evidence of a missing artifact.** This iteration's first act should have been
`wc -c`, and it was: the module was read before anything was written, and
nothing needed writing.

The stale reading survived one more step. The `Read` tool returned the
**162-line / 6,601-byte** it.15 test; `wc -l -c` on the same path returned
**185 lines / 8,006 bytes**. The divergence was caught only because a direct
call to `load_cells` returned 40 cells while the read copy asserted 34, and
because `pytest --collect-only` listed
`test_dist_to_skyline_is_None_on_all_40_...` where the read copy had
`..._all_34_...`. **Two independent readings of the same path disagreed, and
the executable one won.**

## it.17 corrected the test upward, it did not fit the test to the module

Both edits it.17 made to the it.15 RED are documented in the test's own
docstrings and both are strictly more constraining.

| it.15 assertion | Status | it.17 replacement |
|---|---|---|
| `len(cells) == 34` | **impossible** | `len(cells) == 40` |
| `spearman_rho(range(n), secs) ≈ RUN_ORDER_RHO` | **false, returns −0.1224** | seed-vs-`secs` within `arm_smprime`, plus two added assertions |

`34` cannot coexist with the `(1, 16)` `arm_smprime` assertion twelve lines
below it in the same file: 16 `arm_pl` + 16 `arm_smprime` + 8 `softmax` = 40,
and 34 is the it.8 **undeduplicated** `retake + it6` record count, whose
`arm_smprime` count is 18, not 16
(`tests/jupiter/test_v20_r15_it8_q4_q5.py:60-62`). `tests/mercury/phase_c_price.py:18`
already called the corpus `BANKED_40`.

The rho edit is the load-bearing one. The it.15 form correlated
index-**across-arms** with `secs`, which measures the arm rather than the run
order, because `early_warning` runs ~16 arm-dependent forward passes inside the
timed window (`scripts/v15_r1.py:712,:747`). Measured:

```
[RUN] spearman_rho(range(40), secs) over the banked 40   -> -0.1224
[RUN] spearman_rho(seed, secs) within arm_smprime (n=16) -> +0.6971
```

The revised test asserts **both** and asserts `abs(across - within) > 0.5`, so
the arm confound is now a passing assertion rather than a paragraph. Line count
went 4 → 20 and constraint went up, not down.

## Must-fire

```
[RUN] gpu_seconds_to_floor(plant_crossing(at=2, secs=(1,2,4,8,16)))
      value = 7.0    first_crossing_index = 2    why = None
```

The instrument finds a crossing it was TOLD is there, returns the cumulative
seconds **through** that cell (1+2+4), and names the cell. This is the first
time the GPU-seconds-to-floor instrument has been exercised on a positive it
did not find for itself (V-16).

## Must-not-fire

```
[RUN] gpu_seconds_to_floor(plant_non_crossing(secs=(1,2,4,8,16)))
      value = None   first_crossing_index = None
      why   = "no cell crosses floor_1=0.7071 under the runner's own rule
               (m + half < floor1, v15_r1.py:909): 0 of 5 cells. The instrument
               declines rather than interpolate a crossing that is not in the
               data."
[RUN] gpu_seconds_to_floor(plant_non_crossing(secs=(1,2), near_miss=True))
      value = None,  every cell has eval_nrmse < FLOOR_1
```

`None` **with a reason**, not a plausible number.

## Mutation evidence: the plants are not decoration

Four mutants, run in-memory against the shipped module; no file was modified.

| Mutant | Result | Rig requires | Bites |
|---|---|---|---|
| `crosses` → lazy twin `eval_nrmse < floor` | near-miss returns **1.0** | `None` | **yes** |
| `crosses` → lazy twin, **on the real banked arms** | **(12,16) and (1,16)** | (12,16) and (1,16) | **no** |
| `gpu_seconds_to_floor` interpolates the total | returns **3.0** | `None` + reason | **yes** |
| clause (1) emits one tail alone | `set(cp) >= {...}` is **False** | `True` | **yes** |

Row 2 is the finding. **On the 40 banked cells the lazy twin and the runner's
own `m + half < floor1` rule agree exactly** — same 12/16 and same 1/16. A rig
tested only against banked data would have shipped the wrong predicate and no
assertion would have caught it. The planted near-miss is the **only** thing in
the suite that separates the rule from its twin, which is precisely the
argument the test's docstring makes for the negative existing at all.

## Clause (1), both readings, always

```
[RUN] cp_lower_both_tails(12, 16)
      two_sided = 0.4762291801  verdict FAILS
      one_sided = 0.5156035789  verdict CLEARS
      ruling    = UNRULED
```

Both tails ship side by side on **every** row, labelled, with
`ruling="UNRULED"`, so ⟨CLAUSE_1_TAIL⟩ needs no re-run when ruled and no office
can pick the tail after seeing which one wins (`M-2`). The two readings
disagree at this count by 0.0238; that disagreement is the whole reason the
pair is emitted rather than a choice.

## Absent columns

`dist_to_skyline` is `None` on **40/40** banked cells and the rig carries the
cells' own reason verbatim: `no v15/v16 scan-skyline module exists (R-SKY)`.
`w1_to_oracle`, `peak_bytes` and `cert_grade` are `Reading(None, why=...)` with
reasons over 20 characters. `w1_to_oracle` has no object at all: both wings
emit a **point** prediction, so there is no state distribution for a
Wasserstein-1 distance to measure against.

## Regression

```
[RUN] python -m pytest tests/mercury -q                        -> 1 failed, 133 passed
[RUN] python -m pytest tests/jupiter tests/saturn tests/mars_v20 tests/venus -q
                                                               -> 42 failed, 415 passed
```

The single `tests/mercury` failure is
`test_v20_r15_it10_eval_independence.py::test_it8_aggregate_verdict_agrees_with_its_own_cells`
at line 71, and the 42 elsewhere belong to other offices. **Control: zero files
were written this iteration**, so every one of these 43 pre-dates it.18 and
none is attributable to the rig. The frozen journal check passes inside the
rig's own suite: `sha256(results/v17k_r4_retake.jsonl)` still starts `26fb180b`.

## Limits

The 40-cell corpus is one process on one device (RTX 4060 Laptop, sm_89,
torch 2.5.1+cu121), so the seed-vs-`secs` rho of +0.6971 is a single-machine
reading; the published `+0.7029` (`V20_R15_IT8_JUPITER.md:130`) reproduces only
under a fresh-file-first loader that exists nowhere in the tree, and both
figures are carried without preference. There is no `torch.cuda.synchronize()`
in the runner, so every `secs` on CUDA is un-synchronised host wall clock and
the confound above is a lower bound on the contamination, not a measurement of
it. The mutation table is four hand-run mutants, not a mutation-testing sweep;
it shows the assertions that bite, not that no surviving mutant exists. The 43
failing tests outside the rig were observed, not diagnosed. `arena_rig.py` is
untracked, so it has no commit history and the it.15-to-it.17 diff above is
reconstructed from the test's own docstrings rather than from git.
