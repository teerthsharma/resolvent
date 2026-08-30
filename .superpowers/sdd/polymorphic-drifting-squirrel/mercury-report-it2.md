# MERCURY (LESTRADE) — R9 iteration 2

Role: runner. Prices, schedules, executes. **Nothing below is a verdict.**
Section 4 corrects two of Neptune's cost figures; the correction is arithmetic
and does not overturn the coordinator's ruling, which section 4.4 says plainly.

**HEAD check, per the amended `BASE_PROMPT.md`.** Session opened at `3c61181`,
fast-forwarded to `feat/r9-causal-consequence` @ `597f6db` before any file was
touched. `git log --oneline -1` and `git status --short` were run first. Every
citation below is against `597f6db` or later.

Box unchanged from iteration 1: Windows 11, Raptor Lake, RTX 4060 Laptop
8188 MiB, torch 2.5.1+cu121, `torch.set_num_threads(2)` pinned in-module.

---

## 1. Claim ledger

| # | Claim | Class | check |
|---|---|---|---|
| M25 | `Eprocess.value` raises `OverflowError: math range error` at draw 10135 of the shipped 10240-draw pool, after crossing at draw 67 | RUN | Reproduced independently of Deimos: traceback quoted in §2.1, `crossed at 67, crashed at 10135 of 10240` |
| M26 | The adversarial `d = +B` stream crashes at draw 1757 | RUN | Same script; matches Deimos's figure exactly |
| M27 | The repair leaves every representable value bit-identical | RUN | `test_linear_value_is_unchanged_wherever_it_is_representable` — `value == exp(log_value)` and `peak == exp(log_peak)` to `rel=1e-15` over 60 draws |
| M28 | The decision does not move | RUN | Crossing still at draw 67; `Pair.decision == "settled"`; `live()` on the real journal still reads `t=5, decision=None, peak_settled=1.0040499225957493` |
| M29 | `calibrate` never instantiated `Eprocess`; it does now, and the two agree | RUN | Bind gap `8.881784197001252e-16` over 8 replays on the live `NULL_RADEMACHER` calibration |
| M30 | `BIND_TOL = 1e-9` is measured, not chosen | RUN | Worst `\|log E_t\|` gap over four spec/horizon combinations was `3.553e-15`, at horizon 2000. Six orders of margin |
| M31 | The bind check can read FALSE | RUN | With `Eprocess.update` monkeypatched to drift arm 0 by `1e-6` per draw, `calibrate` raises. Vacuity rule 2 |
| M32 | `max_peak`'s clamp is gone and the number survives | RUN | At horizon 2000 on a `+B` spec, `max_peak` reads `inf` and `log10_max_peak` reads the closed-form value to `rel=1e-12` |
| M33 | No pre-registered constant moved and `update` still raises on `\|d\| > B` | RUN | `test_the_preregistered_constants_are_untouched`, `test_update_still_raises_on_an_out_of_bound_difference` |
| M34 | Neptune's guard holds against the LIVE artifact directory, checked before the cut | RUN | `ceq/hf_artifact` copied and regenerated in place: manifest survived, all five checkpoint filenames present, neither denial reintroduced |
| M35 | **The repaired `LIMITS` clause carried three stale numbers.** "60 of the 100 rows ... 15 each" against a journal holding 61 of 86 with `e3_t1` at 16 | RUN | Census recount: `total 86, bare 25, e3_t1 16, e3_t2 15, e3_t8 15, e3_t32 15` |
| M36 | Two of those were true at Neptune's commit; `100` never was | RUN | `git show 25b9cb8:results/m3_quintuple_v2.jsonl` counts 85 rows, 60 e3. The merge with `551d512` made them stale |
| M37 | **A second clobber was still armed.** The plain CLI wrote v0, which `BOARD.md:182-191` records as frozen evidence | READ + RUN | `OUT_MD`/`OUT_JSON` hardcoded `capability_table_v0.*` and were used only in `main()`. BOARD.md documents the hand-edit workaround rather than fixing it |
| M38 | The cut left v0 untouched | RUN | `git diff --stat results/capability_table_v0.md results/capability_table_v0.json` reports no change |
| M39 | No measured cell moved in the cut | RUN | The v1 JSON diff outside prose is: `journal` path, `registry`, `journal_commit`, `head_commit`, and the two new `log10_peak_*` fields. Every arm NRMSE, contrast and interval is identical |
| M40 | Provenance named a temporary worktree | RUN | `provenance.journal` read `...\.claude\worktrees\agent-ab15bf7938b564127\results\...` before the fix; now `results/m3_quintuple_v2.jsonl` |
| M41 | **Neptune's `settled` base rate is confirmed by an independent path** | RUN | His `2.077496 s/step x 150 = 311.62 s` against the journal median `315.65 s` for the same rows: ratio `1.013` |
| M42 | **`twin` costs MORE than `settled` at `ntr8192`, and Neptune priced all ten units at the settled rate** | RUN | `ntr8192 negation_scope`: settled median `315.65 s`, twin median `508.74 s` (`1.61x`); on minima `298.64` vs `366.58` (`1.23x`). Twin is dearer either way |
| M43 | **`twin` has no settle loop, so the `x64` cannot apply to it** | READ | `scale/m3_quintuple.py:293-295` — twin is `(log_gate - logsumexp(log_gate)).exp()`, and `need_gram` is False for it (`:285`). `BatchedSettled.apply` is reached only on the `settled` branch (`:288`) |
| M44 | The corrected per-row lane price is `15.5 h`, not `29.6 h` | DERIVED from M41-M43 | `5 x 10,635 s` (per-row settled) `+ 5 x 508.74 s` (twin, unchanged) `= 55,719 s = 15.48 h`. Steps in §4.3 |
| M45 | The twin/settled ratio does not carry across tasks | RUN | `negation_scope` at ntr8192: twin is `1.61x` dearer. `e3_t1` at ntr2048: twin is `0.47x`, i.e. `2.1x` cheaper. The ordering inverts |

**Adversarial pass.** Of the 14 tests in the e-process battery, 10 failed before
the fix and 4 passed at both ends by design — those four are the guards the
repair must not have bought its way out of (`|d| > B` still raising, the five
pre-registered constants, `Pair.decision`, and `crossed` agreeing with the
linear comparison while the linear one is defined). The table battery's
anti-vacuity test asserts the journal does **not** coincidentally match the
stale literal `100/60/15`, because if it ever did the count tests could not tell
a counted number from a stored one.

---

## 2. Unit 1 — the statistics layer

### 2.1 The crash, reproduced before anything was touched

```
File "scale/eprocess.py", line 330, in update
    v = self.value
File "scale/eprocess.py", line 313, in value
    return math.exp(_logsumexp(self.log_arm) - math.log(len(self.grid)))
OverflowError: math range error

CROSSED at draw 67;  CRASHED at draw 10135 of 10240
adversarial d=+B stream crashes at draw 1757
```

### 2.2 The repair

`log_value` is the primitive; `value` is a view on it through `_exp_or_inf`,
which saturates to `inf` rather than raising. `log_peak` replaces the linear
`peak` accumulator, `peak` becomes the same saturating view, and `crossed`
compares `log_peak >= log(threshold)` — the same comparison, taken where it
cannot overflow. This is `eprocess_perdraw.log10_max_attainable`'s precedent
(`eprocess_perdraw.py:73-83`), including its rule that the original is not
modified where other readers bind it: every representable value is unchanged.

`calibrate`'s `max_peak = math.exp(min(max_log, 700.0))` was the same defect
resolved the other way — a silent clamp. It saturates through the same helper,
and `log10_max_peak` carries what no longer fits. `live()` and
`eprocess_perdraw.run()` gain `log10_peak_*` for the same reason.

### 2.3 The load-bearing half

`_eprocess_log_path` runs the shipped class over `N_BIND = 8` replays of the
first block of every calibration, and `calibrate` refuses the run if the two
paths disagree past `BIND_TOL = 1e-9`. `peek=True` is exempt and reports `None`:
it is the deliberately broken single-arm rule, which `Eprocess` does not
implement, so binding there would compare two different constructions.

### 2.4 Deimos's test

`test_eprocess_value_overflows_deep_inside_its_own_designed_operating_range`
asserted the crash and had to fail once the crash was gone. It is converted,
not deleted: the docstring recording the finding is verbatim, the assertions are
inverted, and the old name is carried in the docstring so `DEIMOS_REPORT.md`
row 1 stays traceable. Its sibling clause `assert "Eprocess" not in
calibrate_src` was written with the note *"if this fires, calibrate() now
exercises the real Eprocess class"*. It fires.

---

## 3. Unit 2 — the table cut

### 3.1 Pre-flight, before regenerating

Neptune's guard was verified against the real `ceq/hf_artifact` rather than a
fixture: the directory was copied and regenerated in place, which is the
operation a cut performs. The manifest survived, all five checkpoint filenames
appeared, and neither denial returned. Kept as
`test_regenerating_over_the_live_artifact_keeps_the_weight_facts`.

### 3.2 Two defects found in the pre-flight

**The repaired clause carried three stale numbers.** `LIMITS` clause (b) stated
"60 of the 100 rows ... 15 each" as stored prose. At `25b9cb8` the journal held
85 rows with 60 e3 and 15 apiece, so two were true and `100` was never right.
The merge made all three wrong. `_task_census` counts them from the journal now.

**The plain CLI wrote the frozen v0.** `BOARD.md:182-191` records v0 as frozen
evidence and v1 as the current build, *"because OUT_MD/OUT_JSON hardcode the v0
paths and the plain command would clobber committed evidence"* — the rule in a
document, the landmine in the tool. `--version` selects the output and defaults
to `v1`; v0 stays writable and takes saying so.

### 3.3 What changed in the cut

Stored strings:

| string | before | after |
|---|---|---|
| `--task` denial | *"`scale/m3_quintuple.py` has no `--task` flag"* | counted census, see below |
| row census | *"60 of the 100 rows ... 15 each"* (stored) | *"61 of the 86 rows ... (e3_t1 16, e3_t2 15, e3_t32 15, e3_t8 15), against 25 bare"* (counted) |
| weights denial | *"carries NO trained weights ... does not exist yet"* | "Weights shipped" section naming all five `taske3_t1` checkpoints |
| consequence fidelity | weights "do not exist" | per-cell weights are saved; 0 of the 25 this table needs are on disk, 6685.3 s to produce |

Numbers that moved: `journal_commit` `b8a9ace` → `551d512`; `head_commit`
`b8a9ace` → `2eadc43`; `provenance.journal` → repo-relative; `registry` from
`negation_scope` alone to the eight names in `M3_TASKS`; two new fields
`log10_peak_settled 0.0017553070010738684` and `log10_peak_twin
0.0040767727940570235`.

**No measured cell moved.** Every arm NRMSE, contrast, interval and weight-table
row is identical to the previous v1 build.

The denial now survives only in `results/capability_table_v0.{md,json}`, which
is correct — BOARD.md says to cite v0 as historical.

---

## 4. The pricing disagreement with Neptune

### 4.1 Where the two pricings agree

Neptune's `settled` base rate is confirmed by a second path that fails
differently. He timed the arm directly: `2.077496 s/step x 150 = 311.62 s` per
unit at `n_train = 8192`. The journal's own `meta.seconds` median over the same
five rows is `315.65 s`. **Ratio 1.013.** His measurement is sound, and his
warning that the FLOP model is optimistic by `1.987x` is a wall-clock
measurement, not a model, so it stands.

My iteration-1 finding that a cell carries a fixed term (`30.7 s` plus
`0.2944 s/step`) does **not** contradict him: at `10,635 s` per per-row unit the
fixed term is 0.3% and vanishes. No disagreement there.

### 4.2 Where they disagree: `twin` was priced as `settled`

`results/r9_systems_gate.md:196` prices the lane as
`settled+twin x 5 seeds = 10 units ≈ 29.6 h`, i.e. `10 x 10,635 s`. All ten
units are charged the `settled` rate. Measured, `ntr8192 negation_scope`:

| arm | n | min | median | max |
|---|---|---|---|---|
| `settled` | 5 | 298.64 | **315.65** | 374.79 |
| `twin` | 5 | 366.58 | **508.74** | 779.17 |

`twin` is **dearer** than `settled` here — `1.61x` on medians, `1.23x` on
minima. So the `0.87 h` baseline under-prices: the real figure is `1.14 h` on
medians, `0.92 h` on minima.

### 4.3 The larger error: the `x64` cannot touch `twin`

`scale/m3_quintuple.py:282-300` dispatches on `base_cell`. `settled` calls
`BatchedSettled.apply(log_gate, log_gram, self.beta, self.t_max,
self.n_neumann)` — the Python `t_max` loop and the Neumann VJP loop that
Neptune's per-row change multiplies by `s = 64`. `twin` is one line:

```python
alpha = (log_gate - torch.logsumexp(log_gate, dim=-1, keepdim=True)).exp()
```

and `need_gram = self.base_cell == "settled"` is False for it, so it skips the
gram as well. **`twin` has no settle loop for the `x64` to multiply.** Charging
the five twin units the per-row settled rate overstates the lane:

```
  5 per-row settled   5 x 10,635 s = 53,175 s
  5 twin, unchanged   5 x  508.74 s =  2,544 s
                                      -------
                                      55,719 s = 15.48 h
```

against `29.6 h`. On minima it is `15.28 h`. The increase factor over the
corrected baseline is **13.5x on medians, 16.5x on minima**, against the
reported `34x`.

### 4.4 What this does and does not change

It does not change the ruling. `15.5 h` is still far outside a round that
iteration 1 priced at roughly 6.5 h of compute under an assumed four buckets per
iteration, so *pilot at reduced `s` rather than spend it* stands on the
corrected number as it did on the original. What changes is the size of the
number the board carries: `15.5 h` and `13.5x`, not `29.6 h` and `34x`.

It is also the third instance of one mistake type this round. `STATE.md:21`
priced a rung as twelve `settled` cells; `results/r9_systems_gate.md:196` prices
ten units as ten `settled` cells; both over-price by charging every arm the
dearest arm's rate. The correction here happens to run the other way on the
baseline, because `twin` inverts and becomes the dearer arm on `negation_scope`
— which is itself the finding in M45: **the twin/settled ratio does not carry
across tasks.** Twin is `1.61x` dearer at `negation_scope`/ntr8192 and `2.1x`
cheaper at `e3_t1`/ntr2048.

---

## 5. What could not be validated

The saturating reader changes what a caller past the double range sees, from an
exception to `inf`; nothing in the tree binds a value that large today and the
two report rows now carry `log10_peak_*`, but a downstream reader testing
`math.isfinite(peak)` would take a different branch, and `json.dumps` renders
`inf` as the non-strict `Infinity` token — neither row is journalled through
`bucket.Journal` today, so nothing writes it. The bind check covers 8 replays of
the first block at whatever horizon the caller asked for, so it binds agreement
and not exhaustiveness, and at the shipped horizons of 400 and 1500 it never
enters the saturating range itself; the overflow is held by the regression
battery, not by the bind check. `BIND_TOL` is calibrated against summation-order
noise on this machine with this numpy. The four IMPACT gates still have never
run as a battery, and `impact` and `impact_hetero` remain in `M3_TASKS` — and so
in the freshly cut card's `registry` — despite `MISTAKES.md:215-228` recording
that those gates never executed; deregistering is an adjudication and was not
done. Regenerating the card replaced hand-written prose with generated prose and
two specific figures left it: the `t* = 1` rung's interval `settled-twin
-0.036025, CI [-0.118936, +0.062209], N=5`, and the export stamps `journal
commit 1cc7900; export HEAD 1c56985`; a third departure is a repair, since the
hand-written claim that the CUDA lane "has produced no journalled figure yet" is
false at 23 cells. `results/capability_table_v1.*` are regenerated only when
this CLI is run, so the census is only as fresh as the last cut. The section 4
correction rests on five journalled `twin` rows whose spread is `366.58` to
`779.17`, a 2.1x range, so the `1.61x` median ratio is the least certain number
in this report and the direction — twin dearer than settled at this geometry —
is what the minima also support; the per-row projection inherits Neptune's
`70.9 s/step`, which was not re-measured here, and assumes the per-row change
touches `settled` only, which is read from the current dispatch and not from an
implementation that does not yet exist.
