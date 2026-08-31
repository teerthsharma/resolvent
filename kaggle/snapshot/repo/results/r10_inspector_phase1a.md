# R10 — Health Inspector, Phase 1a boundary (iterations 8–12)

Scope: `R10_ITERATION_08_09.md`, `R10_ITERATION_10.md`, `R10_ITERATION_11_12.md`,
`R10_MECHANISM.md`. Evidence re-derived from `results/r10_it8_capacity_softmax_t{2,8,32}.jsonl`,
`results/r10_it10_frontier.jsonl`, `results/r10_it12_e4prime_spread.jsonl`, and the modules
named below. The ledger was read with `scale/ledger.py`, never grepped.

**Journal state at audit.** The it.8 journals are live and grew during this audit:
`t32.jsonl` gained a seed-1 cell at n=32768 at 22:08:22 (`1.0039821450630964`). Counts below
are against 54 cell rows, sha256 prefixes `f5ea8769…` (t2, 16 cells), `b8cf9e7a…` (t8, 21),
`1cec72e0…` (t32, 17). Only `r10_it10_frontier.jsonl` pins its sources by hash and read time;
the two markdown records quote counts against a moving file with no pin.

---

## Adjudications

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | it.11 gate MET at (t\*=2, n=2048, steps=150): seed CI [0.9462, 0.9591], N=8, excludes 1.0 | **CLEAN** | Re-derived: mean 0.9523, CI **[0.9462, 0.9591]**, N=8 **distinct** seeds 0–7 all at **threads=6** (threads=8 carries 1 seed and is dropped by `by_seed`), steps filtered to 150, `seed_ci` resamples the 8 per-seed values — not eval examples. Companions [1.1162, 1.1314] and [1.1307, 1.1837] also reproduce. Steps filter is load-bearing: removing it makes `by_seed` raise, exit 1. |
| 2 | Softmax hop budget is 1, cited `m3_capability.py:271` | **STRUCK** (citation) | Line 271 is the E-TASK header `print`. The hop budget is at **line 273**, and it is a `print` of English prose, not a definition. Substance survives elsewhere: `scale/e_ladder.py:55` `HOP_BUDGET = {"softmax": 1, …}`, bound by `tests/chase/test_m3_ladder_task.py:447`. Bad citation repeated in 3 places incl. `scale/r10_capacity_sweep.py:143`. |
| 3a | "15 cells below the bar and all 15 at 150 steps" | **CLEAN-BUT-UNCHECKED** | "All at 150 steps" is unambiguous and true (19/19 rows). The count is definition-dependent: **19** raw rows, **16** dedup by (t\*,n,steps,seed,threads), **15** dedup by (t\*,n,steps,seed). No dedup rule is stated. The round's own `by_seed` treats thread counts as distinct measurements (drift 2.345e-3), which gives 16. |
| 3b | "6 (t\*,n) groups with 2+ step rungs, 0 monotone-worsening violations" | **CLEAN** | 9 (t\*,n) groups, **6** with ≥2 rungs, **12** adjacent pairs, **0** violations. Independently matches it.10's "12 of 12 across 6 of 6". |
| 4 | JUPITER t\*=32: drops 0.126566 → 0.021089, ratio 0.167, asymptote 1.00185, never crosses; margin 6.066e-3 = 2.59× the 2.345e-3 floor | **CLEAN** | drop1 **0.126566**, drop2 **0.021089**, ratio **0.166622**, asymptote 1.0060659 − 0.021089·r/(1−r) = **1.001850**, gap **1.8496e-3**, margin **6.0659e-3**, ratio **2.5868**. Every figure exact. |
| 5 | Noise floor 2.345e-3 from three cross-thread pairs | **CLEAN** | Exactly **3** same-(t\*,n,steps,seed) different-threads pairs exist in the journals. Drifts **2.345020e-03**, **4.910668e-04**, **7.310651e-06**. Max is 2.345e-3. |
| 6 | SATURN: MARS reported population sd; 0.011083 × √(19/20) = 0.010802 | **CLEAN** | Block A: sd(ddof=1) **0.011083**, pstdev **0.010802**; product **0.0108024**. Every other it.12 figure also exact: μ̂ 0.479472, z **−0.7603**, k **19/20**, block B 0.475633 / 0.010301 / 0 reversals, pooled 0.477553 / 0.010738, C(20,≤2)·0.5²⁰ = **2.0123e-4**, CP [0.0013, 0.2487] and [0.0006, 0.1316]. |
| 7 | HOUSE's counted findings, it.8–12 (11 in scope) | **1 STRUCK, 10 CLEAN** | Verified against the journals: grid readings 0.8748/0.9724/1.0061, steps ladders, ceilings 0.7071/0.9354/0.9843, margins 0.0276 and 0.0061, seed range 0.0345, boot CI [0.9641, 0.9814] width 0.0173, drift 4.91e-4, 4.3+7.9=12.2 on 16.1 GiB. Citations `r10_capacity_sweep.py:101` and `:146` are **exact**. **STRUCK: "48 cells"** (below). |
| 7b | `R10_MECHANISM.md` title: "one mechanism, **nineteen** times" | **STRUCK** | The catalogue holds **18** instances: 1–14 in the table, then 16, 17, 18, 19. **There is no instance 15** anywhere in the file or in the ledger. The highest ID standing in for the count — in the title of the file that catalogues that exact mechanism. |
| 8 | The binding rule (finding must be bound by a test RED before repair) | **STRUCK** | **1 of 11** phase-1a findings has any runnable binding; **0** were RED before repair. Detail below. |

---

## What was struck, and what it costs

### S-1. `m3_capability.py:271` does not say what three documents say it says

Called out in the brief as the round's most load-bearing citation, and it is wrong twice over:
the line number is off by two, and the line it *should* point at (273) is a `print` of English
prose — "Model hop budgets, to read the ladder against: softmax 1, …" — not a definition
anything reads. The only machine-readable source is `scale/e_ladder.py:55`.

**Cost: low, and the claim survives.** `HOP_BUDGET["softmax"] == 1` is asserted by a real
pytest node (`tests/chase/test_m3_ladder_task.py:447`, passes in 4.03 s), so `t* > hop` holds for
all three blocks and `t*=2` is inside the open region. But `scale/r10_capacity_sweep.py`
hardcodes the literal `1` at lines 146 and 149 rather than importing `HOP_BUDGET`, so **the hop
budget the grid was actually computed against is bound by nothing.** Change `e_ladder.HOP_BUDGET`
and the test fails while the sweep silently keeps its old ceilings.

### S-2. "The grid, 48 cells" is a row count published as a cell count

`R10_ITERATION_08_09.md` heads its table "The grid, 48 cells", and ledger findings [44] and [49]
repeat "48 cells" / "the 48-cell grid". The grid is **36** (t\*, n, steps) configurations —
**21 run + 15 dropped-and-never-run** — which MERCURY's priced-DAG paragraph states correctly
**in the same document**, two sections away. 48 was the raw journal row count at write time
(16+16+16); it is now **54** and still climbing.

**Cost: the same defect the document confesses to, one section over.** That document already
corrects itself for printing "9 seeds" where it had 9 rows over 8 seeds. It then commits the
identical row-for-cell substitution in its own section heading and ships it to the ledger twice.
The derivation of "15 dropped" is also non-obvious and worth recording: the 16 `dropped` events
carry `steps` as a **list**, expanding to 24 configs, **9 of which were run anyway**; 24 − 9 = 15.
The raw event count is 16, not 15.

### S-3. `R10_MECHANISM.md` claims nineteen instances and holds eighteen

No instance 15 exists. Cost is small in itself and large as a tell: it is the catalogue's own
mechanism — a surface proxy, the highest ID, standing in for the thing, the count — committed in
the catalogue's title.

### S-4. The binding rule, measured

| quantity | count |
|---|---|
| HOUSE findings logged for it.8–12 | **11** |
| red events in the ledger naming any it.8–12 subject | **3** (all MERCURY) |
| …of those, that are runnable pytest node ids | **0** |
| runnable pytest nodes binding any phase-1a claim | **1** |
| phase-1a claims RED before their repair | **0** |

MERCURY's three reds are `scale/vram_gate.py::require`, `scale/r10_capacity_sweep.py::t32` and
`::steps-axis-is-an-overfitting-axis` — module paths plus prose, not collectible node ids. No
test file exists for `it11_verdict`, `idle_gate`, `vram_gate`, `e4prime_spread`,
`r10_it10_frontier` or `r10_capacity_sweep`. HOUSE's own 18 red events all name `tests/loop/test_*`
nodes from iterations 1–7, none from phase 1a. The single real binding
(`test_each_cell_is_scored_against_its_own_hop_budget`) was added **2026-08-26 in round 8**, four
days before this round, and was never red for any claim in it.

**Cost: the phase-1a boundary is bound almost entirely by module-internal `demo()` self-checks,
which pytest does not collect and the ledger never sees.** Below is what those are worth.

---

## Must-fire verification

All three modules exit 0 as shipped. Each was copied to a temp dir and broken there; tracked
source was not touched.

| module | exit | must-fire tested | fires? |
|---|---|---|---|
| `scale/it11_verdict.py` | 0 | steps filter removed from `cells()` | **YES** — `by_seed` raises `ValueError`, exit 1 |
| `scale/it11_verdict.py` | 0 | run against **zero journals** | **NO** — exit 0, "demo OK" |
| `scale/idle_gate.py` | 0 | MUST-FIRE 3, "three consecutive UNKNOWNs must not release" | **NO** — exit 0 |
| `scale/vram_gate.py` | 0 | UNKNOWN-not-GREEN on either resource | **NO** — branch unreachable on this box |

### D-1. `idle_gate.py` MUST-FIRE 3 does not fire — the round's own defect, in the module written to catalogue it

```python
unknowns = iter([UNKNOWN] * 3 + [0, 0, 0])
assert wait_until_idle(lambda: next(unknowns), checks=3, interval=0)
```

This returns `True` whether or not unknowns release the wait, and nothing checks how many
readings were consumed. I reinstated instance 18 **at the loop layer only** —
`run = run + 1 if (is_idle(count) or count == UNKNOWN) else 0` — leaving `is_idle()` correct so
MUST-FIRE 1 still passes:

```
SHIPPED: returned True, consumed 6 readings [-1, -1, -1, 0, 0, 0]
BROKEN : returned True, consumed 3 readings [-1, -1, -1]
         is_idle(UNKNOWN)=False   <- MUST-FIRE 1 correct in both
```

The broken gate **releases on three unreadable probes** — it allocates a 7.9 GiB job on an
unreadable process query, the exact measured failure that took the host to 2719 MiB — and
`demo()` exits 0 printing *"consecutive unknowns do not release"*.

Ledger finding [54] advertises this must-fire by name: *"three consecutive UNKNOWNs must not
release (the retry loop must not silently restore fail-open)"*. It is advertised and absent.
**Fix is one line**, the shape MUST-FIRE 2 already uses: record the readings and
`assert len(seen) == 6`.

### D-2. `vram_gate.py`'s headline rule is asserted only in dead branches

The module's stated purpose is that it *"degrades to a stated UNKNOWN rather than a false GREEN…
because a gate that silently passes when it cannot measure is the vacuous-control shape this
campaign catalogues."* Both assertions for that rule sit inside `if card is None:` / `if host is
None:`. `nvidia-smi` and `psutil` are both present here, so **neither branch executes**. I
replaced both UNKNOWN clauses with a silent GREEN carrying no UNKNOWN token — the precise defect
the docstring refuses — and `demo()` exits 0.

### D-3. `it11_verdict.demo()` passes on an empty journal set

`ROOT` resolves from `__file__`. Run the pristine module from a tree with no `results/`
directory: all six verdicts print `INSUFFICIENT … 0 distinct seeds` and it exits 0 with
"demo OK". Every assertion is on synthetic in-memory rows; the six `print(verdict(...))` lines
assert nothing about real data. **The gate that decides "the region is learnable" cannot
distinguish its own journals from an empty directory.**

---

## Recorded, not struck

- **JUPITER's t\*=32 margin is stale, and correctly provenanced.** 6.066e-3 rests on the single
  seed-0 cell at n=32768. A second seed has since landed (1.0039821450630964); the 2-seed mean
  is 1.005024, moving the margin to **5.02e-3** and the ratio to **2.14×**, not 2.59×. Not a
  strike — `r10_it10_frontier.jsonl` pins source hashes and a read timestamp, the only place in
  the round where that was done, and it is why this reads as staleness rather than error. It also
  strengthens JUPITER's own conclusion: the margin moved by more than a third of itself on one
  extra seed, so "the measured cells do not distinguish crosses-at-49k from never-crosses" is if
  anything understated.
- **Drop 1 mixes populations.** 0.126566 subtracts a single-seed n=8192 point from an 8-seed mean
  at n=2048. Both are the only figures available; the mixing is not stated.
- **`test_e4_rips_gate.py:147` asserts `score > PASS_BAR`, not the value 0.892650.** The record
  says the value is "asserted at" that line. The score is reproduced exactly in the jsonl, so the
  instrument-can-fail demonstration stands; "asserted" overstates what the line does.

---

## Addendum: the tree moved during the audit

Two subjects changed under this audit. Recorded because it bears on every count above.

- **The journals.** `t32.jsonl` gained a seed-1 cell at n=32768 at 22:08 mid-audit. By the last
  read, `t*=8` at n=32768 carries **6** distinct seeds and `t*=32` carries **3** — both still
  short of the N=8 gate, so no verdict flips, but the below-bar count and JUPITER's margin both
  move with every landing seed. Only `r10_it10_frontier.jsonl` pins hashes and a read time. **The
  two markdown records quote counts against a moving file with no pin**, which is why claim 3a is
  clean-but-unchecked rather than clean.

- **`scale/it11_verdict.py`.** HOUSE edited it at 22:17 and again at 22:18 while this audit ran,
  adding `_SKIPPED` / `skipped()` and a V-16 must-fire for unparseable rows. Two transient states
  were observed and are **not** struck, because neither was ever a settled state: at 22:17:37 the
  file did not parse (literal newlines inside a string literal), and its first V-16 assertion was
  `assert skipped() == [] or isinstance(skipped(), list)` — a tautology, since `skipped()` returns
  `list(_SKIPPED)` and is always a list. The 22:18:22 version repairs both: it calls the module's
  own `_cells_in()` and asserts `skipped() == [str(bad)]`, which genuinely fails if the skip is
  not recorded. Good must-fire.

  **D-3 survives that repair.** Re-tested against the settled 22:18:22 file with no `results/`
  directory: exit 0, "demo OK", all six verdicts `0 distinct seeds`. The six
  `print(verdict(...))` lines still assert nothing about real data.
