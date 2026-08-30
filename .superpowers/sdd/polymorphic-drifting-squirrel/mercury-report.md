# MERCURY (LESTRADE) — R9 iteration 1

Role: runner. Prices, schedules, executes. **Nothing below is a verdict.** The
tables are tables. Where a number licenses an interpretation, the licensing is
left to whoever holds the adjudicating role.

Branch: `worktree-agent-ab15bf7938b564127`, fast-forwarded from `ac47049` to
`feat/r9-causal-consequence` @ `74e5590` at the start of the session. The branch
`feat/r9-causal-consequence` itself is checked out in the shared root
(`C:\Users\seal\Desktop\New folder (32)`), so a second worktree cannot check it
out; the worktree branch holds the identical tree plus this session's commits.

Box, and it matters for every number here: Windows 11, Intel Raptor Lake
(Family 6 Model 183), RTX 4060 Laptop 8188 MiB, torch 2.5.1+cu121, CUDA
available. `torch.get_num_threads()` reads **20** before any `scale` import and
**2** after, because `torch.set_num_threads(2)` is pinned inside
`scale/arm_s.py:86` and eight sibling modules. Every arm timing in this repo,
journalled or fresh, is a 2-thread timing.

---

## 1. Claim ledger

Class per `BASE_PROMPT.md`. No claim above its class.

| # | Claim | Class | check |
|---|---|---|---|
| M1 | `run_bucket` reported `done`/`remaining` over the whole journal, so a 10-unit request against a 25-record journal returned `done=35 > total=10` | RUN | `tests/mercury/test_r9_mechanical_fixes.py::test_accounting_ignores_other_tasks_records` failed `assert 35 <= 10` at 74e5590 |
| M2 | The same defect drove `remaining` negative for a partially-journalled set | RUN | `...::test_accounting_counts_a_partially_journalled_unit_set` failed `assert 29 == 4` at 74e5590 (25 foreign + 4 own) |
| M3 | The fix in the stale worktree `modest-gates-9099f2` is correct as written | READ + RUN | Diffed both files by content — git access to that worktree is refused for a worktree-isolated agent, so `diff -u` on the file bodies stood in for `git diff`. `mine = len(units) - len(todo)` counts requested units present; `n_done = sum(1 for k, _ in units if k in final)` bounds `remaining` at >= 0. Both harvested verbatim; the battery is green |
| M4 | `require_complete` was never wrong and needed no change | READ | `scale/bucket.py:203-217` — `missing = [k for k, _ in units if k not in done]` already filters by `units` |
| M5 | The fix is live on a real run, not only in the test | RUN | `python scale/m3_quintuple.py --task e3_t1 --cells softmax --seeds 0 ...` against `results/m3_quintuple_v2.jsonl` (85 records, 4 tasks, 3 arms) printed `1/1 units already journalled` and `bucket end: 1/1 done, 0 remaining` |
| M6 | `E_T_STAR` raises for all three registered-but-dial-less tasks | RUN | `KeyError('impact')`, `KeyError('impact_hetero')`, `KeyError('e4prime')`; all three confirmed present in `M3_TASKS` in the same snippet |
| M7 | **FINDINGS A3 overstates the `e_ladder` half.** `scale/e_ladder.py:143` is unguarded but unreachable | READ | Its task names come only from `RUNGS = ("e3_t1","e3_t2","e3_t8","e3_t32")` at `scale/e_ladder.py:48`, a module constant no CLI flag touches (`scale/e_ladder.py:391-395` adds `--seeds`, `--n-train`, `--n-eval`, `--json` and nothing else). Every member has an `E_T_STAR` entry. `scale/etask_k5e.py:117` is the reachable one: `--tasks` defaults to `TASKS` but accepts any string (`scale/etask_k5e.py:92`, no `choices=`) |
| M8 | `make_impact_batch` accepted `d=0` silently and returned a full batch | RUN | `make_impact_batch(2, 1024, 0)` returned in 2.56 s with no exception at 74e5590 |
| M9 | Three statements and their comment, after `return` in `make_impact_batch`, were unreachable | READ | At 74e5590: `return x, y, f, p` at `scale/impact.py:643`, then a comment at `:645` and `f = query` / `p = 0` / `return x, y, f, p` at `:646-648`. `inspect.getsource(...).count("return ")` read 2. FINDINGS A6 cites `:641-643`, which is the dead `if`/`pass` pair plus the live return, not the unreachable block; the commit message for this change says "four unreachable statements", which counts the comment line as a statement. Both are off by one line-range or one item — the block is `:645-648` |
| M10 | `impact_decoder_gate`'s `planted_sd` was `label_sd` under another name | READ | `planted_sd = float(y_np.std())` and `label_sd = float(y_np.std())`, `scale/impact.py:816` and `:814` at 74e5590 — byte-identical expressions |
| M11 | The replacement is a distinct quantity and reads non-degenerate on a real draw | RUN | At `n=64, s=64, seed=0`: `planted_sd 1.581254` vs `label_sd 2.905943`, `pass_nondeg True` |
| M12 | The new clause **can read FALSE** (adversarial pass) | RUN | With `impact_planted_features` monkeypatched to a constant matrix, `planted_sd 0.0`, `pass_nondeg False`, `passes False` |
| M13 | The new clause changes no shipped verdict | RUN | Same draw before and after: `passes True` both sides. Consistent with FINDINGS A4 — the gate has no recorded artifact to disturb |
| M14 | The five failures in `tests/foreman/test_journal_thread_binding.py` are pre-existing | RUN | Identical 5 failures with `scale/` stashed to 74e5590 and with the change applied. Untouched, per hard rule 2 |
| M15 | Contention on this box runs 1.75x-2.5x slower than the journalled medians | RUN | Two independent lanes: CPU softmax `e3_t1` measured 30 s against a journal median of 17.13 s (1.75x); CUDA softmax measured 4 s against a journal median of 1.59 s (2.5x) |
| M16 | The two existing `foreman_looped` rows are contention artifacts, not a price | DERIVED from RUN | They read `e3_t2` at 267.8 s / 296.3 s; a fresh `e3_t8` cell of the same arm, steps and sizes measured **74.9 s**. In `m3_quintuple_v2` every arm is *cheaper* at `e3_t2` than at `e3_t8` (twin 35.96 vs 56.05; settled 77.81 vs 104.31), so the ordering in `foreman_looped.jsonl` is inverted relative to the task's own cost curve |
| M17 | Cell cost is the training loop; batch construction is not a term | RUN | `e3_t2/t8/t32` builders take <= 0.05 s at n=4096; one `run_cell` pays <= 0.13 s of build against a 74.9 s cell |
| M18 | IMPACT's builder is flat in `n` at s=1024 | RUN | 3.44 s at n=64, 3.47 s at n=256 — the 1024x1024 solve is paid once per graph |
| M19 | Cost is **not** linear in `steps`; a cell carries a fixed term | RUN | Same arm, task and sizes: 74.9 s at steps=150, 207.4 s at steps=600 — 2.77x for a 4x increase. Two-point fit gives 30.7 s fixed + 0.2944 s/step. The fixed term is not batch build (M17); it is the 0-step control, the `n_eval=4096` eval forward and the `n_boot=10000` bootstrap |
| M20 | `s=1024` costs 135x `s=64` per training step | RUN | `settled` arm forward+backward at batch n=64, 2 threads: 6.45 / 48.74 / 872.59 ms per step at s = 64 / 256 / 1024. Last leg exponent `log(17.9)/log(4) = 2.08`, i.e. quadratic, as attention is |
| M21 | Training is **full-batch**, so `s=1024` OOMs before the clock matters | READ + DERIVED | `scale/paired_arm.py:70-73` — `mse_loss(model(x_train), ystd)` over all `n_train` rows every step, no minibatching in the loop. At `n_train=2048, s=1024` one attention activation is `2048 x 1024 x 1024 x 4 B = 8192 MiB`; the card is `8188 MiB` total (`nvidia-smi`) |
| M22 | `STATE.md:21` over-prices the four-rung ladder by 3.0x | DERIVED from READ | Its unit cost reproduces (379.3 s vs journal median 379.32 s), but ~77 min per rung is ~12 cells at the `settled` rate, and a rung is 5 `settled` + 5 `twin` + 5 `softmax`. Measured: 6,047 s = 1.68 h for all four rungs, against its ~5 h |
| M23 | The GPU is **slower** than the CPU on the `e3_t2` rung | DERIVED from READ | CUDA rung 681.9 s (flat, launch-bound) against CPU 653.2 s = 0.96x. The GPU advantage is 4.23x at `e3_t1` and 2.39x at `e3_t32`, where CPU `settled` costs 379.32 s and 223.68 s |
| M24 | Background Bash invocations are not bound by the 600 s foreground cap | RUN | Two detached `foreman_looped` runs of 79 s and 211 s wall completed across turns |

**Adversarial pass on the battery.** Sixteen of seventeen tests failed at
74e5590 and all seventeen pass after. The one that passed at both ends is
`test_builder_accepts_an_in_range_d`, the positive control that keeps the new
`ValueError` from over-firing; it is supposed to pass at both ends. Two tests
are source-level greps (`test_the_k5e_reader_routes_through_the_guard`,
`test_builder_has_no_unreachable_tail`) and are labelled as such in the file —
`etask_k5e.main` trains cells and cannot be called from a unit test, so a grep
is the smallest thing that fails if the unguarded index returns.

---

## 2. The four fixes

### A2 — `run_bucket` counted the wrong set

Harvested from `modest-gates-9099f2` verbatim after verification, plus the stale
comment in `scale/m3_synthetic_settled.py:278-282` rewritten to past tense.
`require_complete` untouched.

### A3 — `E_T_STAR` KeyError: **guard, not entry**

`negation_scope.e_t_star(task, s)` returns the dial or `None`;
`scale/etask_k5e.py` prints `-` in the `t*` column.

Reason for choosing the guard over a derived value: IMPACT's label is
`e_q^T (I - rho A)^{-1} B n`, a full resolvent over a graph whose diameter is a
property of the *draw*, not a hop count fixed by the task name — the thing
`E_T_STAR` encodes for the `e3_t*` family. Any dial written down for it would be
a number the pre-registered reading then conditions on. A wrong dial reaches the
outcome table; an absent one does not. `e4prime` and `impact_hetero` are the same
shape of object.

`scale/e_ladder.py:143` is deliberately left alone — see ledger M7. Guarding it
would convert an unreachable `KeyError` into a reachable `None` flowing into the
pre-registered ladder table, which is the silent degradation the dispatch warns
against.

### A6 — dead guard and unreachable tail

`if d < 1 or d >= s: pass` now raises `ValueError`. The four statements after the
`return` are deleted. Note for the record: **no sibling builder in `scale/` has a
`d`-range guard at all** — `grep -n "d < 1 or d >= s"` across `scale/*.py` hits
only `scale/impact.py:640`. The message is therefore written in the style of
IMPACT's own two neighbouring guards (`s=... nodes is below IMPACT admissibility
floor ...`, `d_model=... cannot hold IMPACT channels`), which is the closest
available precedent, rather than in a sibling builder's style that does not
exist.

### A5 — the placeholder in a shipped gate

`planted_sd` is now `float(X_planted[:, 1:].std(axis=0).min())` — the smallest
per-column spread of the planted regressors, intercept excluded, since column 0
of `impact_planted_features` is `ones` (`scale/impact.py:732`) and would pin any
un-sliced minimum at exactly 0.

It is returned, and `pass_nondeg` and `passes` both read it. The open questions
at 805-810 are deleted with the code they annotated.

Why not simply delete: the old value could not fail independently of `label_sd`,
which is already in the clause — vacuity rule 2. The new one can, and is shown
to (ledger M12). It sits in the same family as the two checks already there, and
it protects the gate's PASS half specifically: `score_planted < 0.70` is evidence
that the plant is readable only if the planted regressors vary across the draw.

---

## 3. Cost table

See `results/r9_pricing.md` for the full table, the base rates it was built
from, and the arithmetic. Headline, in units of the default 420 s bucket:

| Item | cells left | journal rate | this box (2.5x) |
|---|---|---|---|
| D1 `foreman_looped` | 2 (was 4) | **415 s — 1.0 bucket, already at this-box rate** | |
| D2 CUDA lane | 37 (was 38) | 1,901 s (4.5 buckets) | 4,751 s (11.3 buckets) |
| 5-seed 3-arm rung, CPU, `e3_t2` | 15 | 653 s (1.6) | 1,633 s (3.9) |
| 5-seed 3-arm rung, CPU, `e3_t1` | 15 | 2,882 s (6.9) | 7,205 s (17.2) |
| 5-seed 3-arm rung, CUDA, any task | 15 | 682 s (1.6) | 1,705 s (4.1) |
| Full 4-rung CPU ladder | 60 | 6,047 s (14.4) | 15,117 s (36.0) |

Two of D1's four missing cells were run and journalled this session
(`e3_t8/looped3` at `steps=150` -> `1.101899` in 74.9 s, at `steps=600` ->
`1.447873` in 207.4 s). Those two points also gave the cost model: **2.77x for a
4x step increase**, not 4x, solving to 30.7 s fixed plus 0.2944 s/step.

**Does not fit, at any iteration count: anything at `s=1024`**, which is where
`impact` and `impact_hetero` are floored (`IMPACT_MIN_NODES = 1024`,
`scale/impact.py:73`). Two independent blockers. Time: the `settled` arm's
forward-plus-backward measures 6.45 ms/step at `s=64` and 872.59 ms/step at
`s=1024`, a **135x** ratio, which puts one `settled e3_t1`-shaped cell at
`379.32 x 135 = 51,208 s = 14.2 h`. Memory, and this one is absolute:
`scale/paired_arm.py:70-73` trains **full-batch** with no minibatching anywhere,
so at `n_train=2048, s=1024` a single attention activation is
`2048 x 1024 x 1024 x 4 B = 8192 MiB` against the card's **8188 MiB total**.
One activation exceeds the whole GPU. The IMPACT registration cannot be
exercised as a training task without changing the training loop or the shape.

Two corrections to the record, both arithmetic: `STATE.md:21`'s unit cost
reproduces exactly (journal median 379.32 s against its 379.3 s) but its rung
and ladder figures price all twelve cells at the `settled` rate, and the
measured four-rung ladder is **1.68 h, not ~5 h — over-priced by 3.0x**. And
the GPU is not uniformly faster: it is 4.23x cheaper at `e3_t1` and **0.96x at
`e3_t2`, marginally slower**, because the CUDA lane is launch-bound and charges
a flat ~67 s per `settled`/`twin` cell while CPU `settled` at `e3_t2` costs
77.81 s.

---

## 4. What could not be validated

The four gates in `scale/impact.py` still have never run as a battery, and this
change does not run them — `impact_decoder_gate` was exercised at `n=64, s=64`,
which takes the `_make_impact_batch_unsafe` branch, not the admissible `s>=1024`
one the gate ships for. FINDINGS A4 stands untouched: `scale/impact.py` remains
registered against its own docstring's instruction. The `planted_sd` threshold
`1e-6` is copied from the `label_sd` clause beside it and is not calibrated
against any distribution of planted spreads; it is a floor against exact
degeneracy, not a power calculation. The `-` printed in the `t*` column of
`scale/etask_k5e.py` was never rendered by an actual run of that CLI, because
running it on `impact` would train arms at `s=1024`, which nothing in this repo
has ever done and which the pricing note puts outside the round; the guard is
verified through `negation_scope.e_t_star` directly and through a source grep
instead. Every timing in the pricing note comes from one laptop under variable
load, with a measured spread of 1.75x-2.5x against the same journal's own
medians and a 7.9x worst-case spread inside the journal itself (`settled e3_t1`
median 379.32, max 3004.14); the extrapolations carry that band and should not
be read tighter than it. The CUDA prices for `e3_t2` and `e3_t8` are
interpolated between measured `t1` and `t32` cells, and no GPU cell at those two
tasks has been run for `settled` or `twin`. The whole pricing note assumes the
`n_train=2048 / n_eval=2048 / s=64 / d=24` configuration and does not price any
other shape; the one observation outside it (`softmax` at `n_train=8192`,
102.67 s against 17.13 s at 2048) also moves `n_eval` from 512 to 2048, so that
6.0x factor is confounded across two variables and is used nowhere. The cost
model in M19 rests on two points of one arm on one task and cannot detect
curvature. The `s`-scaling ratios in M20 come from a batch of 64 against a
synthetic loss, not from a real cell — the *ratio* is what carries the
extrapolation and the absolute ms/step figures do not; scaling that
microbenchmark up by batch size under-predicts the real per-step cost by about
12x, which is why only the ratio is used. Buckets-per-iteration is assumed at 4
and is not measured anywhere. Finally, the falsifier in
`scale/foreman_looped.py` still returns `complete: False`; the two cells this
session added are journalled numbers, `beats_at_t2` and `under_one_at_t8` both
read `false` at `steps=150`, and nothing here reads a verdict off any of that.
