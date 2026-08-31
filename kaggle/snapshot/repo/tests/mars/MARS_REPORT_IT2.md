# MARS / MORIARTY — R9 iteration 2

Fast-forwarded from `74e5590` to `5201d78` before touching anything. Iteration
1's commit `3180d34` is present in the history of `feat/r9-causal-consequence`.

**Deliverable: the `argmaxste` cell — the straight-through-estimator control
that separates the two things iteration 1 showed were entangled.** Built, bound
bitwise with two REDs, priced analytically, registered in four places. **No
reading taken.** No wall clock appears anywhere in this report.

Tests: `tests/mars/test_mars_argmax_ste.py`, 12 tests. Run with the file it
amends, `tests/chase/test_pivot_exclusion_lift.py`: **18/18 pass, 4.30 s.**
Iteration 1's `tests/mars/test_mars_r9_iteration1.py` re-run unchanged: **10/10
pass, 55.62 s.**

---

## 1. What the cell is and why the forward had to stay fixed

Iteration 1 measured that `argmax`'s alpha is
`torch.zeros_like(log_gate).scatter_(...)` — no `grad_fn`, `requires_grad` False,
`‖∂α/∂log_gate‖₁` exactly `0.0` against `52.3193` for `twin` and `55.1514` for
`settled`. So the published `argmax − softmax = −0.118456` confounds two things:
*mixture against lookup*, and *trained selection against untrained selection*.

The only way to separate them is a cell whose **forward is the same function**
and whose **backward is not**. `scale/m3_quintuple.py`, `_alpha`:

```python
elif self.base_cell == "argmaxste":
    soft = (log_gate
            - torch.logsumexp(log_gate, dim=-1, keepdim=True)).exp()
    hard = torch.zeros_like(log_gate)
    hard.scatter_(1, log_gate.argmax(dim=-1, keepdim=True), 1.0)
    alpha = hard + (soft - soft.detach())
```

`soft - soft.detach()` is elementwise `x - x`, exactly `+0.0` for every finite
entry, so `alpha` is bitwise `hard`. The backward runs the softmax Jacobian.

**The parentheses are load-bearing and are asserted by value.**
`hard + soft - soft.detach()` rounds `hard + soft` before subtracting and is
*not* bitwise — drawn on a gate scaled by 40.0 where the two forms disagree
(`test_the_unparenthesised_estimator_is_not_bitwise_and_is_the_second_red`).

## 2. The binds, each with the RED it needs

Per `scale/arm_s.py:341-346` — *"A bind that only ever saw the correct path
could not tell the two apart."*

| bind | statistic | result |
|---|---|---|
| forward equality | `torch.equal(arm_ste(x), arm_argmax(x))` at `(n,seed)` ∈ {(8,0),(16,1),(32,2)} | **True 3/3** |
| **RED 1** — the arm | `torch.equal(arm_argmax(x), arm_twin(x))` on the same instances | **False 3/3**, max abs diff > 0 |
| **RED 2** — the arithmetic | `torch.equal(hard + soft - soft.detach(), hard)` on 4096×8 float64 | **False** |
| forward through the loop | step-0 loss, identical init, identical batch | `1.108632` for both, bitwise equal |
| gate gradient | `‖∂α/∂log_gate‖₁` | `argmax` **exactly 0.0**; `argmaxste` **= twin's, to rel 1e-12** |
| parameter count | `sum(p.numel())` | **4769** for all five of `softmax`/`twin`/`settled`/`argmax`/`argmaxste` |

The gradient row is asserted as an *equality against twin*, not as "greater than
zero": the whole design claim is that the forward changed and the backward did
not, so anything other than twin's gradient would mean the estimator is doing
something extra.

## 3. Vacuity rule 6 — and the statistic that failed it, reported

*"A repair must be shown to change the object it repairs."*

**My first statistic was confounded and I am reporting it rather than dropping
it.** Counting how many examples change their selected pivot over training does
not measure trained selection here, because `select_pivots` is a top-k over
`kk.norm()` and `wk` moves for reasons unrelated to the gate. Measured, 30 Adam
steps at lr 1e-2, n=256, identical init, identical batch:

| cell | pivot **set** changed | argmax **slot** changed | selected **node** changed | loss step 0 → 30 |
|---|---|---|---|---|
| `argmax` | **256 / 256** | 200 / 256 | 197 / 256 | 1.108632 → 1.035080 |
| `argmaxste` | **256 / 256** | 183 / 256 | 182 / 256 | 1.108632 → 1.033804 |

The candidate set turns over completely for **both** cells. Against that churn
the estimator moved *fewer* slots — which is noise in a statistic that is mostly
reading candidate-set reshuffling, not evidence about the gradient. Discarded as
a claim; kept as data.

**The unconfounded statistic is parameter divergence.** From identical init on
an identical batch the only difference between the two runs is the gradient
reaching `log_gate`:

```
||theta_ste - theta_argmax||  after 30 steps  = 3.871934
||theta_argmax||                              = 11.981715
relative                                      = 32.3%
```

Asserted at `> 0.05` relative. If the extra gradient path did nothing the two
vectors would be identical.

**This also sharpens iteration 1's wording, which was too strong.** "Which pivot
`argmax` reads is never trained" should read: *the choice among the selected
candidates receives no gradient.* The candidate set itself moves — 256/256 in 30
steps — as a side effect of `wk` training through `av` and the shared softmax
operator. The `0.0` gate-gradient measurement is unchanged and still stands.

The two loss numbers are **not a reading** — 30 steps at n=256 against a shipped
protocol of 150 steps at n=8192 over five seeds. They are printed only because
step 0 being bitwise equal is part of the forward bind.

## 4. Price

Analytic FLOPs only, `scale/m3_flops.py`, at `n=8192, s=64, d_model=16,
hidden=128, k=8, t=21, n_neumann=21`:

| cell | total FLOPs |
|---|---|
| `softmax` / `glance` | 6,996,099,072 |
| `argmax` | 7,365,197,824 |
| **`argmaxste`** | **7,367,294,976** |
| `twin` | 7,367,294,976 |
| `settled` | 7,431,258,112 |

`argmaxste` is priced as `twin`, not as `argmax`, and the difference from
`argmax` is exactly `2,097,152 = n·2·k·d_model` — the `alpha @ av` contract term.
`argmax`'s entry sets `contract=0` because its one-hot contraction degenerates to
a gather; the estimator materialises the same normalised gate `twin` does and
runs a real contraction, so it pays. The one-hot scatter is comparisons and a
write, not counted — the same treatment `argmax`'s own argmax already gets.

## 5. Registration — four sites, and the name

| site | change |
|---|---|
| `scale/m3_quintuple.py` | `STE_CELLS = ("argmaxste",)`; `ALL_CELLS = CELLS + PLUS_CELLS + ROW_CELLS + STE_CELLS`; one `_alpha` branch |
| `scale/m3_flops.py:172` | `if cell in ("twin", "argmaxste")` — otherwise `ValueError` at accounting time |
| `scale/e_ladder.py:55` | `HOP_BUDGET["argmaxste"] = 2`, `argmax`'s exactly, since the forward is bitwise `argmax`'s |
| `tests/chase/test_pivot_exclusion_lift.py` | `fams` extended to include `STE_CELLS`, and the construct loop with it |

The test-file edit is what the file's own comment at `:142-145` invites — *"the
check keeps working as families are added"* — and is the same edit Neptune made
for `ROW_CELLS`. It is not a repair of a finding.

**Naming.** `argmaxste`, no underscore, following `ROW_CELLS`.
`capability_table.read_journal` does `key.partition("_")` and
`eprocess._parse_key` splits on `_sd`, so `argmax_ste` would be silently
re-read as cell `argmax` — the collision that took down 14 of 16 tests once. The
name carries none of `_sd`, `_task`, `_k`, `_b`, and no underscore at all. The
key round-trip is asserted:
`_key(...).partition("_")[0] == "argmaxste"`.

`--cells` default is still `list(CELLS)` — asserted — so a default run measures
exactly what it measured before and no published reading moves.

## 6. Pre-registered reading — rows exhaustive BEFORE the data

Mercury holds the execute seat. The contrast is `argmaxste − argmax` and
`argmaxste − softmax`, paired percentile bootstrap, same five seeds, same
geometry, strict at zero. Softmax's bar is `0.877168` (seed 0) / `0.892323`
(mean of 5); `argmax`'s is `1.010779` (mean), CI `[1.008307, 1.042529]`.

| row | what the data show | what it licenses |
|---|---|---|
| **A** | `argmaxste − argmax` CI covers zero | Training the selection buys nothing. The deficit IS the lookup. `CHECKLIST.md:1206` is **restored** and the mixture claim is earned. |
| **B** | `argmaxste` beats `argmax`, still above softmax's mean | The `−0.118456` splits. Report both halves; the mixture claim survives **in weakened form**, sized by the split, never quoted as "the whole contribution". |
| **C** | `argmaxste` reaches or beats softmax | The deficit was untrained selection. *"Reading one pivot is worse than reading none"* is **refuted**, and the mixture claim falls. |
| **D** | `argmaxste` is WORSE than `argmax` | The straight-through gradient is biased and this estimator is the wrong instrument. Licenses nothing about the mixture in either direction; the question stays open and needs a different control. |
| **E** | `argmaxste` mean is above 1.0 | **Uncreditable by the repo's own rule** (`CHECKLIST.md:1207`), exactly as `argmax` is, regardless of where it sits against `argmax`. A contrast between two arms that both fail predict-the-mean licenses nothing. This row is live: `argmax` reads `1.010779`. |

Row E is the exhaustiveness catch — the analogue of row H in
`E_LADDER_PREREGISTERED_READING.md`, added because A–D all condition on the cell
being a valid arm.

## 7. Reported, not repaired

`_bind_batched_against_arm_s()` returns **False on its DIAGNOSTIC shapes** at
clean `5201d78` — `n=2, s=128, d=16, k=32, seed=2`, max abs diff `1.387779e-17`.
Confirmed pre-existing by `git stash`: identical `False` with my working tree
removed. **It does not block a run**: `main()` gates on `req` — the run's own
shapes — and `DIAG_SHAPES` is explicitly non-gating (`m3_quintuple.py:875-877`).
The gating shapes bind: `s=64, d=24, k=8`, seeds 0 and 1, **BITWISE 2/2**.
Mercury can start. Untouched, per hard rule 2.

## 8. Claim ledger

| # | claim | class | check |
|---|---|---|---|
| 1 | STE forward is bitwise `argmax`'s | `RUN` | `torch.equal`, 3 shapes, 3/3 |
| 2 | the bind discriminates (RED on the arm) | `RUN` | `twin` vs `argmax` not equal, 3/3 |
| 3 | the parentheses are load-bearing (RED on the arithmetic) | `RUN` | 4096×8 float64, unparenthesised form not equal |
| 4 | `argmax` gate gradient is exactly `0.0` | `RUN` | `requires_grad` False, no `grad_fn` |
| 5 | STE gate gradient equals twin's to rel 1e-12 | `RUN` | random-probe VJP, same init and batch |
| 6 | `n_params == 4769` for the new cell | `RUN` | 5 cells asserted |
| 7 | step-0 loss bitwise equal across the two cells | `RUN` | `1.108632` both |
| 8 | pivot SET churns 256/256 in 30 steps for BOTH cells | `RUN` | the confound that killed my first statistic |
| 9 | slot drift 200 vs 183 — estimator moved FEWER | `RUN` | reported as data, **not** claimed |
| 10 | parameter divergence `3.871934` on `11.981715`, 32.3% | `RUN` | identical init, identical batch, 30 steps |
| 11 | `argmaxste` FLOPs = `twin`'s; `− argmax` = `2,097,152` = `n·2·k·d_model` | `RUN` | `cell_terms` at shipped geometry |
| 12 | journal key round-trips; no forbidden substring | `RUN` | `_key(...).partition("_")[0]` |
| 13 | default `--cells` unchanged | `RUN` | `parse_args([]).cells == list(CELLS)` |
| 14 | `HOP_BUDGET["argmaxste"] == HOP_BUDGET["argmax"]` | `RUN` | asserted |
| 15 | DIAG bind failure is pre-existing at `5201d78` | `RUN` | `git stash`, same `False` |
| 16 | gating shapes bind bitwise | `RUN` | `s=64 d=24 k=8`, seeds 0,1, 2/2 |
| 17 | which of rows A–E fires | **not measured** | Mercury |

## 9. What I could not validate

**The question the cell was built to answer is not answered, and cannot be from
here.** I hold no execute seat, so rows A–E are open; everything above is a
construction and a bind, and the report deliberately contains no NRMSE. The
32.3% parameter divergence proves the extra gradient changes the trained object,
not that it changes it for the better — row D is in the table precisely because a
straight-through gradient is a biased estimator and can make a cell worse. The
30-step loss pair (`1.035080` vs `1.033804`) points the right way by a margin of
`0.001276` at n=256, which is far inside anything the shipped five-seed bootstrap
would call zero, and it should not be quoted as a direction. I did not implement
the per-row form of the estimator, so `_alpha_all_rows` still raises
`AssertionError` for `argmaxste`; that path is unreachable because `per_row` is
False for this cell, but a future `argmaxsterow` would need the branch. I priced
the cell analytically and took no clock at all, on the coordinator's calibration
that the box is `1.65×` off iteration 1 on identical code — so I can offer no
wall-clock estimate to whoever schedules the run, and the FLOP ratio is a floor,
not a prediction, on the same grounds `scale/m3_flops.py:101` states for the row
cells. Nothing here was measured at `s=16`. I did not re-audit any part of
`FINDINGS.md` section B, now demoted, beyond the `CHECKLIST.md:1206` / `:1207`
lines I quote, which I re-read at `5201d78` this session.
