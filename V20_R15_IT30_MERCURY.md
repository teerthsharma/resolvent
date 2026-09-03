# V20 R15 IT30 — MERCURY (arena and measurement)

Read 2026-09-02, **18:50–19:02 IST**, branch `v17k-gate0` @ `207e7b9`.
**LANDINGS CENSUS: `39 → 53 of 129`.** One clause, `14 of 14` land, `1` weak named.
**`2` new defects, `M-30a` (this office's own) and `M-30b`.** `M-29a` **RESOLVED**: the
clause is false, **the cell is not** — `Q2/W1` stays `NOT-PUT` and no grade moves.
Subject document `V20_R15_THEORY_TABLE.md`, content digest `sha256:942e4208893444cd`,
43,461 bytes, 444 lines. Every count below is against that digest.

One screen, saved and reproducible:

    [RUN] python -m pytest tests/mercury/test_v20_r15_it30_admission.py -v
    screen: results/v20_r15_it30_mercury_screen.txt

`4 passed, 2 xfailed` — the two `xfail(strict=True)` cases are the RED, taken against
unmutated code: they assert the table's own sentences and fail because the sentences
are false.

---

## 1. M-29a RESOLVED — the clause is false, the CELL IS NOT

**M-29a is closed as a POINTER defect and NOT as a grade change. Q2/W1 stays NOT-PUT.**

`:306` gives Q2/W1's ADMISSION CONDITION in two parts, and they are not the same part:

- the **operative clause**: *"one cell of BED-K's shape actually run"*
- the **evidentiary clause**: *"has zero callers under `scripts/`"*

The evidentiary clause is **FALSE**. `build_delay` has **2** callers under `scripts/`,
`scripts/v20_m14_cheeger.py:347` and `:462`, both at `n=500, d=4, seed=7` — the kwargs
registered at `ceq/kdata.py:476` **exactly**, measured by AST rather than by grep.

The operative clause is **STILL UNSATISFIED**, and that is the finding:

| call site | binds | keys of the bed dict consumed | data keys touched |
|---|---|---|---|
| `scripts/v20_m14_cheeger.py:347` | `b` | `["K"]` | none |
| `scripts/v20_m14_cheeger.py:462` | `b` | `["K"]` | none |

`build_delay` returns `dict(kind, n, seed, params, K, b, z, pos, plant)`
(`ceq/beds/bed_k.py:232`). **Both sites read `K` and nothing else.** They never touch
`b`, `z`, `pos` or `plant` — the bed's *data*. And `scripts/v20_m14_cheeger.py` contains
**zero occurrences** of `arm_smprime`, `arm_pl`, `hankel`, `rank_real`, `rank_plus_lower`.

**So the generator is called, the bed is constructed, and no arm is run on it and no Q2
metric is computed from it.** The two calls are a *structural audit of the kernel
operator* — the H1–H4 Cheeger hypotheses — which is a claim about `K`, not a cell.
A cell of BED-K's shape needs the bed's data through a wing; neither call site has one.

**What this means for the grade — JUPITER's to rule, and the measurement he needs:**
Q2/W1's `NOT-PUT` is **CORRECTLY FILED** and its re-entry grade `F1 + const` is
**untouched**. `J-14`'s primary input does not move. What must be repaired is the
sentence, which currently offers a false reason for a true filing — the worst shape a
pointer can take, because a reader who checks the reason and finds two callers concludes
the filing is wrong.

**Proposed replacement for `:306`, no grade change:**

> one cell of BED-K's shape actually run — the generator is `build_delay` @
> `ceq/beds/bed_k.py:236`, registered @ `ceq/kdata.py:476`, and its **only two callers
> under `scripts/`** (`v20_m14_cheeger.py:347,462`, at the registered kwargs) consume
> `b["K"]` alone: **a structural audit of the kernel, not a cell** — no arm is run and
> no Q2 metric is computed.

**M-30b (NEW, JUPITER).** `:151` (Q2/W1 ROUTE) asserts *"zero cells of BED-K's shape have
ever been run"*, citing `V20_R15_IT13_MERCURY.md:63`. That sentence survives this
measurement **unchanged and correct** — it is the operative clause, and it is the one the
`:306` cell should have quoted. The two sentences 155 lines apart say different things
and only one of them is false.

---

## 2. THE `~275` / `309.047` VERDICT — **NEITHER, AND THERE ARE THREE NUMBERS**

`:354` (the `L-9` hybrid-verdict row) quotes `~275 GPU-s`. `:184` (Q4/W1 ROUTE) says the
number *"does not evaluate to 275; its own formula gives `309.047`"*. **Both lines are
wrong, in different directions, and the second is a defect this office introduced.**

The formula, verbatim, `V20_R15_IT8_JUPITER.md:216`:
`3 × (0.25 + 1 + 4) × (16.16 + 1.78 + 1.68)`.

| number | what actually produces it | status |
|---|---|---|
| `275` | **nothing.** Labelled `≈` at IT8:216; no expression in the tree evaluates to it | **withdrawn** |
| `309.015` | the formula's **own literal constants**, `[RUN]` | the formula as written |
| `309.047` | `tests/mercury/phase_c_price.py:_sweep(2.0)` on it.13's **re-measured** means `16.161 / 1.780 / 1.681` | a *different object* |

**RULING M-30. The honest quote is the band `206 – 537 GPU-s`, point `309.0`, and no
point may be quoted without it** — because the `S²` weight is the very exponent the sweep
exists to measure (`V20_R15_IT13_MERCURY.md:96`). `275` sits **inside** that band and is
**none of its three points**, which is why it has survived: it is never obviously wrong.

**Two edits follow, and they are not the same edit:**

- `:354` — `~275 GPU-s` → `206–537 GPU-s (point 309.0)`. A leap acting on `L-9` is
  currently acting on a number nothing computes.
- `:184` — **M-30a (NEW, this office's own defect).** The parenthetical *"its own formula
  gives `309.047`"* is false: the formula's own constants give **`309.015`**. `309.047`
  is it.13's re-measurement, correctly labelled at `V20_R15_IT13_MERCURY.md:106` as one
  point on a band and **mis-attributed to the formula** when carried into the table.
  `tests/mercury/test_v20_r15_it13_phase_c_price.py:47` already carries the tell — its
  docstring says *"It is 309.02"* while its assertion is `approx(309.047, abs=0.01)`, a
  band that **excludes** `309.015`. The docstring is right about the formula and the
  assertion is right about `_sweep(2.0)`; they are describing two objects.

A price the leap acts on that the same document corrects is worse than an uncited one.
This one was corrected **with a third number**, which is worse still.

---

## 3. THE `git grep` SWEEP — one other git-backed instrument, and it is IMMUNE

The lesson: **the first `m29d` used `git grep`, went GREEN on zero hits, and the script it
swept for is untracked. A git-backed instrument reads an untracked tree as an empty one,
and an empty sweep is indistinguishable from a clean one.**

`[RUN] grep -rln "git grep|git ls-files|check_output(\[.git" tests/ scripts/`
→ **6 live instruments** (`__pycache__` excluded):

| instrument | office | carries the antidote? |
|---|---|---|
| `tests/jupiter/test_v20_r15_it26_live_claim.py` | JUPITER, **R15** | **YES** |
| `tests/loop/test_attic_never_removes_the_last_must_fire.py` | pre-R15 | not assessed |
| `tests/loop/test_corpus_is_recoverable_and_verifiable.py` | pre-R15 | not assessed |
| `tests/loop/test_no_module_writes_a_file_at_import.py` | pre-R15 | not assessed |
| `tests/mars_v20/test_p12_absence_proof_falsified_by_recording_it.py` | MARS, pre-R15 | not assessed |
| `scripts/k_cert.py` | — | not assessed |

**The only R15 instrument on the list is JUPITER's, and it already knows.** Its `:230`
comment states the mechanism in the round's own words — *"every test file that names M,
tracked OR untracked, because `git grep` without `--untracked` is blind to exactly the
files this round writes"* — and `:257` carries the guard this office's first `m29d` lacked:
`assert hit, "the importer probe found nothing -- git grep is not running"`.

**A non-empty liveness assertion is the whole antidote.** It converts a silent empty
sweep into a RED. This office proposes it as the round's rule: **any git-backed check
must assert its own probe is non-empty**, and SATURN's measurement — 0 of 18 R15
instruments tracked `[CITED: SATURN]` — is the reason it is not optional.

`[RUN] git ls-files --error-unmatch` on this office's four instruments
(`test_v20_r15_it30_admission.py`, `test_v20_r15_it13_phase_c_price.py`,
`phase_c_price.py`, `scripts/v20_m14_cheeger.py`): **4 of 4 UNTRACKED**, including the one
written this iteration. Consistent with SATURN.

---

## 4. LANDINGS CENSUS — **`39 → 53 of 129`. CLAUSE C. `14 of 14` LAND.**

**Clause C — §2's two cost-law cells, `Q4/W1` and `Q4/W3`, table lines `179–192`.** Every
citation in the clause, checked by opening the cited line.

    [RUN] results/v20_r15_it30_mercury_landings.txt

| table line | citation | what is there | verdict |
|---|---|---|---|
| `:179` | `V20_R15_IT8_JUPITER.md:105` | the `O(n·S²)` derivation row for `path_product` | **LANDS** |
| `:179` | `ceq/arm_smprime.py:144` | `def path_product(a: torch.Tensor) -> torch.Tensor:` | **LANDS** |
| `:179` | `scripts/v15_r1.py:137` | `S, D = 64, 24  # the shape every e3 row in results/ uses` | **LANDS** — verbatim |
| `:179` | `lean/CEQ/V16Domain.lean:129` | `theorem pathProd_eq_zero_iff (m θ : ℕ → ℝ) (i j : ℕ) :` | **LANDS** |
| `:179` | `ceq/arm_smprime.py:559` | `def zero_hop_mask(self, x: torch.Tensor) -> torch.Tensor:` | **LANDS** |
| `:180` | `ceq/sizing.py:145` | `def flops_per_token(cfg, *, arm, hops=HOPS,` | **LANDS** |
| `:180` | `scale/m3_flops.py:207` | `CELLS = ("softmax","glance","settled","twin","argmax")` | **LANDS — weak, see below** |
| `:180` | `ceq/mz_kernel.py:170` | `def attention_flops(plan: ZoomPlan, head_dim, batch_heads) -> int:` | **LANDS** |
| `:181` | `V20_R15_IT13_MERCURY.md:146` | the `RETIRE` row, `206 – 537 (point 309.0)` | **LANDS** — verbatim |
| `:181` | `scripts/v15_r1.py:249` | `t0 = time.time()` | **LANDS** — the timer open |
| `:181` | `scripts/v15_r1.py:267` | `secs = time.time() - t0` | **LANDS** — the timer close |
| `:187` | `ceq/arm_pl.py:304` | `def brute_force_path_sums(a: torch.Tensor) -> torch.Tensor:` | **LANDS** |
| `:188` | `ceq/arm_pl.py:113` | `w = (q @ k.transpose(-2,-1)) / math.sqrt(q.shape[-1])` | **LANDS** — the `S×S` block |
| `:188` | `ceq/arm_pl.py:93` | `def key_bias(g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:` | **LANDS** |

**`0` FAIL. `1` weak, named.** `14` is added; **`39 → 53`. No rate and no extrapolation
to `129` is offered.**

**THE WEAK ONE, and it argues for the table rather than against it.**
`scale/m3_flops.py:207` is cited as one of *"the three shipped cost models"*; the line is a
**cell-name tuple, not a model**. The pointer lands and the label is loose. It is also the
**strongest evidence in the clause**, because the tuple `("softmax","glance","settled",
"twin","argmax")` contains **neither `arm_smprime` nor `arm_pl`** — the table's `0 of 2
wings lie in the domain` is legible on the cited line itself.

**A SECOND READING FALLS OUT OF `:249,:267` FOR FREE.** Both lines are `time.time()`, not
`torch.cuda.Event` and not preceded by a synchronize. The table's *"the timer is
un-synchronised"* (`:179`) is confirmed at the cited lines, not merely asserted from them.

**DISJOINTNESS.** it.25's `21` (`71, 95, 104, 163–170, 171, 203, 304`), it.27's `+3`
(`339, 379, 382`), it.28's `+10` (`211–228`) and it.29's `+5` (`289, 335, 346, 347, 368`)
contain **no line in `179–192`** — it.25's `163–170` is the nearest and does not reach it.
On citation identity: **none of the fourteen targets above appears in any published
landings table this round.**

---

## 5. NOT REACHED

- The four pre-R15 git-backed instruments in §3 are **listed, not assessed**. Whether any
  of them sweeps for a file this round wrote is unmeasured.
- **No git writes. Nothing touched Kaggle.** No document outside this file was edited;
  the `:306` and `:354` replacements above are **proposed text**, not applied — `:306` is
  JUPITER's cell and `:354` is his ledger row.
