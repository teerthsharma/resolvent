# V15 LEDGER — R11 iteration state

The loop's position pointer. Every iteration reads this file first, does the
node named under NEXT, appends a row to the log, and rewrites NEXT.

Contract: `CEQ_V15_CONTRACT.md`. Architecture-and-state: `workdonenew.md`
(`ARCH.md` does not exist and must not be created).

---

## MOUNT RECORD (D-3)

- **Deactivation commit read:** `ecbedf7` — `git show --stat` reports one changed
  file, `results/r10_it8_waveB.log`, binary, 40068 -> 40194 bytes, `1 file
  changed, 0 insertions(+), 0 deletions(-)`. The commit body credits a change to
  `.claude/ralph-loop.local.md` that is not in its own diff.
- **`MISTAKES.md` read:** 54 entries across four mechanism classes — V (vacuous
  controls, 22), P (provenance, 9), M (measurement, 16), D (design, 6).
- **The 15 standing loop failures read:** `results/r10_loop_suite4.txt:21-35`,
  recorded 2026-08-31 00:02:24. Count stable at 15/15/14/15 across four
  snapshots with **three different membership sets**; the set is not fixed.
- **Cause of death, one sentence:** the R10 loop ran out of affordable work, not
  of iterations, and was killed by hand because `max_iterations: 0` disabled the
  stop hook's only ceiling (`stop-hook.sh:61`) and the `completion_promise`
  named two sequential 40-iteration ranges whose second could not begin until
  the first reached it.35, so no exit condition could ever fire.
- **Iteration count source:** the v15 DAG critical path, approximately 24 nodes
  (`CEQ_V15_CONTRACT.md`, PART V), scripted to 30 iterations by the author.
  Hard cap **30**, passed as the literal `--max-iterations 30` flag and read
  back out of `.claude/ralph-loop.local.md`.

---

## STANDING RULINGS (decided, not asked — loop autonomy)

| # | ruling | cost if wrong |
|---|---|---|
| RUL-1 | D-4 ("Round 11 it.1-22 precede every line of v-main.7") is **superseded** by the v15 script, which schedules BED-1/v-main.7 work at it.18-21 inside this same round. v15 states it supersedes everything after `attic/workdonenew.pre-v13.md`; `CONTRACT.md` §0 D-4 postdates that file, but v15 is the later authored contract and names its own order. | If wrong, BED-1 work at it.18-21 is premature and its cells are re-run behind a registration that has not happened. Recoverable: the cells are journalled, not published. |
| RUL-2 | Dispatched agents get **no git writes** (D-1: the shared state that collides is git, not the file set). The coordinator commits serially after review. | None known; this is D-1 read literally. |
| RUL-3 | `n1`-`n4` write **disjoint fresh paths** and therefore need no worktree isolation. Worktrees are reserved for agents editing files another agent also edits. | If two agents collide on a path, one write is lost. Mitigated by naming each agent's output paths explicitly in its prompt. |
| RUL-4 | The four it.0 nodes are dispatched as a **mixed fleet**: Opus where the node adjudicates or derives, Sonnet where it produces code or a mechanical census. | A Sonnet node returning a weak artifact costs one re-dispatch. |
| RUL-5 | `CEQ_V15_CONTRACT.md` is **verbatim-of-record and is never edited**. Corrections to it live in `V15_CONTRACT_ARITHMETIC_AUDIT.md` and in `MISTAKES.md`, so the contract as authored stays diffable against what was later found wrong. Same for `attic/**`, which is archive. | If wrong, the contract carries known-bad numbers. Mitigated: every finding names the contract line it corrects, so the pair is readable together. |
| RUL-6 | The contract's it.1-4 clause *"the sizing model replaced by the calibrated one everywhere it was cited"* is **already satisfied and requires no edit**. `ceq/sizing.py` is calibrated (`:48`, `:68`); the five sites a census flagged STALE are all the *record of the correction*, not uses of it. Filed as `MISTAKES.md` M-17. | If wrong, a genuinely stale use survives. Checked directly: the only executable site is `ceq/sizing.py` and it is calibrated. |
| RUL-7 | The scoreboard baseline is **`0 of 39`** (`README.md:55`, `workdonenew.md:44`), not the contract's *"the current 22%"*, which has no producer anywhere in the tree including the round archives. The five scoreboard moves (it.10, 17, 21, 23, 30) are deltas from `0 of 39`. | If `22%` names a real quantity the repo does not compute, the ceiling sentence is measuring something else and the author must name it. Cost of the wrong choice is one renumbered scoreboard, not a lost measurement. |

---

## NEPTUNE — SYSTEMS LINE (measured this session, not inherited)

| fact | value | how read |
|---|---|---|
| torch | `2.5.1+cu121` | `python -c "import torch; print(torch.__version__)"` |
| CUDA available | **True** | `torch.cuda.is_available()` |
| numpy | `1.26.4` | same probe |
| Lean toolchain | `leanprover/lean4:v4.7.0` (pinned) | `lean/lean-toolchain` |
| mathlib | vendored and BUILT | `lean/.lake/packages/mathlib/.lake/build/lib/Mathlib.olean` present |
| `lean/.lake` on disk | 4.2 GB | `du -sh lean/.lake` |
| prior project build | exists | `lean/.lake/build/lib/CEQ.olean` present |

CUDA being live is the fact that decides whether R2 (`t* = 8`, `n = 32768`) is
affordable at all. The R10 loop died at `n = 8,192` with its remaining step
budget journalled as "unaffordable, see priced DAG"
(`results/r10_r10_it21_t8_n8192_capacity_softmax_t8.jsonl:8`). Mercury prices R2
against this line at it.8, not before, and re-measures rather than inheriting
the 45.8 s/turn figure (L-TIME).

---

## LOG

| it | node | model | verdict | artifact |
|---|---|---|---|---|
| 0 | mount + D-3 read + contract to disk | opus | DONE | `CEQ_V15_CONTRACT.md`, this file |
| 0 | loop mounted, cap read back `max_iterations: 30` | opus | DONE | `.claude/ralph-loop.local.md` |
| 1 | `n1` prior art at equation level | opus | DISPATCHED | `V15_N1_PRIOR_ART.md` |
| 1 | `n2` loop-suite re-date + L-EQ into MISTAKES.md | sonnet | DISPATCHED | `V15_N2_SATURN.md` |
| 1 | `n2b` calibrated-sizing citation census (read-only) | haiku | DISPATCHED | `V15_N2B_SIZING_CITATIONS.md` |
| 1 | `n3` Lean #1,#2,#3,#5,#6,#7 | opus | DISPATCHED | `V15_N3_LEAN.md` |
| 1 | `n4` BED-K + interventional channel, TDD | sonnet | DISPATCHED | `V15_N4_BEDK.md` |
| 1 | `n5` Mars files the four standing attacks at it.0 | sonnet | DISPATCHED | `V15_MARS_ATTACKS.md` |
| 2 | `n2b` returned: 5 STALE sites, all prose; `ceq/sizing.py` ALREADY calibrated | haiku | **DONE** | `V15_N2B_SIZING_CITATIONS.md` |
| 2 | coordinator: audit of the contract's own arithmetic | opus | **DONE — 4 findings** | `V15_CONTRACT_ARITHMETIC_AUDIT.md` |

| 3 | `n2` returned: L-EQ's evidence base measured at **4/40 = 10%**, not two-thirds | sonnet | **DONE** | `V15_N2_SATURN.md`, `MISTAKES.md` P-10 |
| 3 | `n6` Venus files competing predictions for R1/R2/R3/R6 early | opus | DISPATCHED | `V15_VENUS_PREDICTIONS.md` |
| 3 | coordinator: branch `feat/ceq-v15-r11`, two commits | opus | **DONE** | `deee6c4..52d7052` |

| 4 | coordinator: M-17 filed, RUL-5/6/7 | opus | **DONE** | `6aef853` |
| 4 | `n7` Neptune systems gate (R2 affordability, the cell that killed R10) | opus | DISPATCHED | `V15_NEPTUNE_SYSTEMS.md` |
| 5 | `n3` returned: **train-gate green as algebra, PARITY CLAUSE REFUTED** | opus | **DONE** | `V15_N3_LEAN.md`, `lean/CEQ/V15.lean` |
| 5 | `n8` Jupiter-2: settle whether (L) and (P) can share one operator | opus | DISPATCHED | `V15_JUPITER2_FORK.md` |

| 6 | coordinator: C-PAR row rewritten, both routes closed | opus | **DONE** | `089e789` |
| 6 | `n9` normalization-boundary sweep (does the same defect recur?) | sonnet | DISPATCHED | `V15_NORMALIZATION_SWEEP.md` |
| 7 | `n1` returned: **10/10 [V-eq]**; the bind lands on LINEAR attention | opus | **DONE** | `V15_N1_PRIOR_ART.md` |
| 7 | `n6` Venus returned: **R1's kill-diagnostic cannot discriminate** | opus | **DONE** | `V15_VENUS_PREDICTIONS.md` |
| 7 | coordinator: M-18, M-19 filed; three Venus claims verified in source | opus | **DONE** | `MISTAKES.md` |
| 7 | hazards relayed to `n4` mid-flight (positional channel, GL NaN, α box) | opus | **SENT** | — |

| 8 | `n4` returned: BED-K built, **6/6 tests pass both halves** | sonnet | **DONE** (commit held) | `V15_N4_BEDK.md`, `ceq/beds/`, `tests/beds/` |
| 8 | `n10` Saturn-2: is the squared-feature fix a repair or an ORACLE LEAK? | sonnet | DISPATCHED | `V15_SATURN2_LEAK_RULING.md` |

| 9 | `n5` Mars returned: **21 passed, 4 skipped, 0 failed**, every skip must-fired | sonnet | **DONE** | `V15_MARS_ATTACKS.md`, `tests/mars_v15/` |
| 9 | `n7` Neptune returned: **R2 FITS, measured 4.000 GiB** | opus | **DONE** | `V15_NEPTUNE_SYSTEMS.md` |
| 9 | `n11` Mercury: `--device` flag + the three conditions Neptune attached | sonnet | DISPATCHED | `V15_MERCURY_DEVICE.md` |

---

## it.9 — THE LAST LOOP DIED BESIDE AN IDLE GPU

The R10 loop stopped when its `n = 8,192` wave journalled its remaining budget
as *"unaffordable, see priced DAG"*. v15 then scheduled R2 at `n = 32,768`, four
times that cell. Neptune priced it **by allocating it**.

| quantity | value | how |
|---|---|---|
| R2 peak at `n = 32768` | **4.000 GiB** | `max_memory_allocated()`, **run not projected** |
| free VRAM | 6.939 GiB of 7.996 total | `mem_get_info()` |
| headroom | **2.61 GiB, 38% of free** | measured |
| largest affordable `n` at 20% margin | **42,489** | `n = 65536` needs 8.26 GiB, exceeding *total* VRAM — 32768 is the last power of two this device runs |
| it.9-it.23, 184 cells | **637 CPU-hours or 47.0 GPU-hours** | fitted, ratio **13.6x** |

**The calibrated sizing model holds, and holds conservatively.** measured/predicted
`= 0.950, 0.950, 0.950, 0.949` at `n = 512 / 1024 / 2048 / 32768` — a constant
ratio across a **64x span**, which is what licenses the extrapolation. The module
over-predicts by 5% and never under-predicts. An `s`-sweep back-solves
`C_RESIDUAL = 17.92` against the module's 18.00 and `C_OPERATOR = 3.50` against
3.90. **No new correction factor needed.**

**Three contract/brief premises refuted by measurement:**

1. *"the dominant term is the `[n,s,s]` operator"* — **FALSE at BED-M's shape.**
   At `s = 64, d_model = 16`, fp32, the operator is **43.7%** of measured peak;
   the residual/MLP chain is 56.3%. The terms cross at `s = 73.8` (fp32) /
   `s = 47.8` (bf16). This premise was in the coordinator's own brief and came
   back false.
2. *`secs ~ n^1.338`* `[INHERITED]` — **does not reproduce.** Re-fits here as
   CPU `n^1.1920` (R² 0.999642) and CUDA `n^0.9734` (R² 0.999316), stable under
   4-point refit. The inherited figure's `1.634` tail at `n = 16384` is
   consistent with **page-file pressure at a ~2 GiB working set**, which is
   exactly `n = 16384` at this shape — its own tasklist entry warned of it.
3. `activation_bytes(arm="softmax")` is **the wrong call** for this comparison:
   in `sizing.py` "softmax" means fused SDPA with no `[S,S]` tensor, while
   `m3_capability.Arm`'s softmax branch materialises `[n,s,s]`. On the memory
   axis it is a *signed* arm.

**The recommendation is not a cut.** Move the cells to the CUDA device that sat
idle while the last loop died of unaffordability. That needs a `--device` flag in
`scale/r10_capacity_sweep.py` — dispatched as `n11` with Neptune's three
conditions attached, the first of which is load-bearing: `MISTAKES.md` **M-10**
records that *thread count alone* moves NRMSE by **0.464 of `Δ_eq`**, so pooling
CUDA and CPU cells would void the reading.

**Open, and stated rather than assumed:** ARM PL's own `C_OPERATOR` is **NOT
MEASURED** (a composed S-M + S-K may retain ~2x, pricing R2 at 5.74 GiB — still
fits, 1.0 GiB margin). And **if BED-K's power-law bed needs `s = 256`, R4's
matched-`n` requirement caps both beds at `n ≈ 5,273` and R2's `n = 32768` cannot
enter the two-sided table at all** — which would put the `+8` R4 item out of
reach for a reason unrelated to any architecture.

### BED-K, final: an honest negative inside a green suite

`n4` replaced its naive `(i−j)^(H−1.5)` kernel with exact FARIMA/GL weights and
reports that this **did not meaningfully shrink the DFA bias — 0.812 → 0.818 at
`H_true = 0.75`** — rather than presenting the correction as a fix. The must-fires
are clean (white noise `H ≈ 0.478`, AR(0.5) `H ≈ 0.486`, nowhere near R/S's
biased 0.75) but the estimator reads **+0.06 high on the bed it will actually
score**. R3 asks for `Ĥ = α̂ + ½` within CI of the calibrated estimator; that CI
must accommodate a known `+0.06` bias, or the bias must be corrected, and either
way it is registered now rather than discovered at R3.

`hard_delay_attention` was also narrowed to read only `b` and `pos`, never `K` —
closing a leak before it existed.

---

## it.8 — BED-K IS BUILT, AND R1 AS SPECIFIED IS NOT RUNNABLE

**`n4`: six tests, both halves each, all passing.** The discipline is what makes
them worth having:

| test | result |
|---|---|
| delay bed scan-blind | `R² = −0.000170` using **the true previous label** as recurrence state — the most generous state any scan could get. Population `R²` is exactly 0 (`b` iid, so `z_{i−1}` and `z_i` draw different noise positions for any `d ≥ 1`); 30-seed sweep measured `[−0.0027, 0.0028]` |
| the control that validates it | the **same fitting code** recovers a true AR(1) at `α = 0.8000, β = 1.0000, R² = 1.000000` — exact. So the near-zero is the bed, not a broken fitter |
| delay bed attention-reachable | `8.67e-19` against a `1e-12` bar, via **one-hot positional query/key** at offset `d = 7` |
| Jacobian oracle | delay `2.88e-11`, power-law `4.88e-10`, both `≤ 1e-9`, against an *independent* `rebuild` call |
| bumps move the label | nonzero at **every** kernel-nonzero position, **bitwise zero** at every kernel-zero position, nonzero-mask checked non-degenerate first |
| seeded reproducibility | bitwise equal same seed, differing across seeds, both beds |

**Hurst: DFA order-1, not R/S.** White-noise must-fire passes; on AR(0.5) it
lands within `~0.02–0.03` of 0.5 — nowhere near the author's documented biased
R/S reading of `0.75`. Cross-checked through autocovariance decay independently
of DFA (`H = 1 + slope/2 = 0.735` at `H_true = 0.7`), confirming the residual
bias is a property of the **generator**, not a DFA window artifact.

**Three vacuous tests refused and named by class:** a "K is lower-triangular"
check (V-3, an algebraic identity of its own construction), a self-written R/S
calibrated to reproduce the contract's own 0.75 (tests nothing about this
corpus), and a bump test asserting only `np.any(diff != 0)` (V-9 exactly).

### The it.8 verdict that matters: R1 cannot run as written

Three independent findings each block it, and together they mean the deciding
measurement of the round is measuring something other than what it says.

1. **The arm cannot represent its own gate.** `−softplus(Wx)` is monotone; the
   target `log|a|` is even. (M-18)
2. **The kill-diagnostic cannot discriminate.** `R²` on `log|a|` has
   `SST = 0.000e+00`; it returns the same value whatever the arm does. (M-18)
3. **The parity bind is false**, twice over and by two methods. (`n3`, `n1`)

And a fourth is now under adjudication by `n10`: if the drive channel of `x`
carries `a` itself, then the proposed squared-feature repair reads the oracle,
"learning the gate" is a change of variables on a visible input, and **the
existing nine-cell non-crossing census means something different from what it
has been taken to mean.**

**Registration owed before R3, from `n4`'s own numbers.** Its attention head
reaches the delay bed through **one-hot positional coding**. That establishes the
bed is attention-reachable *in principle*. It does not establish that a trained
arm can reach it, because `scale/m3_capability.py` gives its arms no positional
feature. Whether BED-K's `x` carries position must be registered **before** the
cell runs — afterwards, "attention lacks a memory kernel" and "attention was
given no position" are inseparable explanations for the same number.

---

## it.7 — TWO INDEPENDENT ROUTES REACH THE SAME REFUTATION

**`n1` confirms `n3` from prior art, not from Lean.** Dao & Gu's dual form is
`(L ∘ QK^T)V` with **no softmax** (§2.4), and §S-M specifies an *unnormalized*
hop — so `g ≡ 0` binds to causal **LINEAR** attention. Probes: `a_t = 1 →
tril(ones)`; `alpha = 1 →` ungated linear attention to `8.9e-16`. The Lean node
reached the same conclusion from row sums (`i+1 ≠ 1`). Two methods, two nodes,
one verdict, and they agree on *what the `g ≡ 0` object actually is*.

**Prior art: 10/10 at `[V-eq]`,** each with equation, hypotheses as the source
states them, and an executed numeric instance — every probe carrying an **O(1)
mis-transcription control** so a `1e-16` residual is evidence, not decoration.
Four sub-statements stall at `[V]` and are flagged inadmissible, including
Dayan 1993's SR resolvent, which PART I leans on.

**Venus: R1's registered kill-diagnostic cannot fire or fail to fire.**
Verified in source by the coordinator, not taken on report:
`scale/negation_scope.py:428` is `a = (randint(0,2)*2 − 1)` then
`a[:, :head+1] = 0.0` — Rademacher, so `|a| ∈ {0,1}` and **`log|a|` is
identically 0 on the live band.** `R²` on a constant target: `SST = 0.000e+00`.
Pooled/clamped it reads `3.2466e-04` against a `1.22e-04` null. Filed **M-18**.

**The finding underneath it is larger than the diagnostic.** `g = −softplus(Wx)`
is **monotone** in the drive channel; the true `g = log|a|` is a band mask and
therefore **even**; `scale/m3_capability.py` gives its arms no positional feature
to route around it (grep for `pos|position`: zero hits). One squared feature
recovers the band at `R² = 1.000000` exactly. **The obstruction is evenness, not
information** — no amount of data fixes a parametrization that cannot represent
its target.

**Two scoreboard clauses are one event.** `NRMSE < floor₁ ⟺ ĥ > 1` by algebra
(`ĥ = t*(1 − NRMSE²)`, `floor₁ = √((t*−1)/t*)`), verified identical at
`t* = 2, 8, 32` across four NRMSE values straddling each floor to ten decimals.
The scoreboard pays `+12` for the crossing and `+4` for `ĥ > 1` separately.

**M-19 filed.** A tent-map orbit in float64 loses one bit per step and carries
nothing from `x₀` after ~52 iterations; the Pesin probe first returned
`h_sym = 0` **at the generating partition** — the right answer for the wrong
reason. Rebuilt in exact rational arithmetic it reads `0.0003` generating against
`0.139/0.208/0.223` at four wrong guards, corroborating the contract. Bollt et
al. 2001 additionally prove the deficit is **non-monotone** in misplacement, so
ranking guards by deficit is unsound; only the `≈ 0` test is admissible.

**Recorded as read-not-verified.** `arXiv:2605.08966` (VORT) would occupy the
S-K fractional head entirely and publishes a sharper form of Lean #12. It was
read by local PDF extraction after the summarizer returned a *wrong paraphrase
that was discarded*. The independent citation registry was unavailable this
session, so **no component is retired on it** until it is confirmed.

---

## it.5 — THE LEAN TRAIN-GATE VERDICT (the contract's own milestone)

The contract's it.5 line: *"#1, #2, #5, #6, #7 green, or the failing statement
named — a failing [M] item means the arm is WRONG, and that is the point."*

**Verdict: five of five green as algebra. The contract's PARITY CLAUSE is FALSE
and the refutation is a theorem.**

| # | statement | status |
|---|---|---|
| 1 | `chain_path_product` | GREEN |
| 2 | `prefix_logit_mask` | GREEN |
| 3 | `parity_sign` | GREEN |
| 5 | `gate_zero_is_attention` | **GREEN on the mask half; the contract's parity clause FALSE** |
| 6 | `bounded_gates_stable` | GREEN, and stronger than asked (`a ∈ (0,1)` strict) |
| 7 | `scan_assoc` | GREEN, trivial version explicitly refused |

**The falsehood.** The contract reads *"PARITY WITH SELF-ATTENTION is by IDENTITY
BIND, not TOST. `g == 0` gives bitwise standard attention (Lean #5)."* §S-M
specifies "ONE **unnormalized** causal hop" — the contract's own word. At `g ≡ 0`
every `C_i = 0`, every masked entry is 1, and row `i` sums to `i + 1`:

```
gate_zero_row_sum        : ∑ j in range (i+1), Wc g i j = ↑i + 1
gate_zero_not_stochastic : 1 ≤ i → ∑ j in range (i+1), Wc g i j ≠ 1
```

Every softmax row sums to 1, for every query, key and weight matrix. Smallest
witness `i = 1`, row `(1,1)`, sum 2. Normalizing does not rescue it: the
normalized row is uniform `1/(i+1)`, which is attention only for a constant-logit
head, and §S-M's hop carries no QK term to recover. **Mathematics, not proof
engineering** — the refuting statement is three lines and green.

**The repair, also proved.** `gate_zero_logit_identity : q i j + (scan g i −
scan g j) = q i j` — under the ADDITIVE-logit reading the bind is genuine and
needs no TOST.

**The fork, which is what this costs.** Label reproduction
(`prefix_logit_computes_chain`) holds for the **multiplicative** hop; parity
(`gate_zero_logit_identity`) holds for the **additive** hop. The contract claims
both from one construction. Settling whether any single operator can carry both
— with a bind that carries information — is dispatched as `n8`.

**Provenance.** `lake build` exit 0 at `[1525/1526]`, toolchain
`leanprover/lean4:v4.7.0`, mathlib vendored at matching tag. Exit 0 alone was
treated as insufficient: all 16 theorems run through `#print axioms`, every one
depending only on `[propext, Classical.choice, Quot.sound]`, `sorryAx` nowhere.

**Two trivial versions refused, and the refusals recorded.** #7 read literally is
`add_assoc` — one token, licenses nothing. Stated instead for the affine monoid
`(a,b) ↦ (x ↦ a·x + b)` with `affApply_affComp` proving `affComp` is genuinely
map composition and `chain_step_eq_affApply` proving by `rfl` that those maps are
the recurrence's own steps. #1 stated in absolute indices so it composes with #2
into `prefix_logit_computes_chain` — the oracle-gate bind at **exact arithmetic**
rather than at the contract's inherited `4.0e-15`.

### it.3 verdicts

**`n2`, Task B — the round's sharpest finding so far.** The contract introduces
L-EQ and justifies it with *"two-thirds of the pre-v13 §5 strikes were
[V]-as-theorem"*. Counted against the section it names: `attic/workdonenew.pre-v13.md`
`:160-211` holds 44 rows, 4 of which record a claim that HELD, leaving 40
genuine strikes. **Four are [V]-as-theorem. 4/40 = 10%, not 66.7%** — the
contract overstates its own law's evidence base by 6.7x. The four, each a real
source cited and its own hypotheses not honoured:

| row | strike |
|---|---|
| `:177` | Siegmund's ARL₀ closed form applied to a non-i.i.d. null; at measured `φ̂ = 0.709` the real ARL₀ misses nominal by **24.1x** |
| `:196` | Kantz-Grassberger fed box-counting `D₀` where every fetched statement uses information dimension `D₁` |
| `:200` | `ρ_P = √2` (the random-**matrix** ensemble average) applied to an exactly antisymmetric `J`, whose value is `2.000000` at four sizes |
| `:201` | Poincaré-Hopf invoked where the field is tangent to the boundary at `μ = 0` (`dz₀ = −0.000000e+00`), so transversality fails |

Five near-misses excluded **with reasons per row**, not silently: X27b/X27c cite
no source at all (fabrication, not misreading); X27d owes a citation it never
made; X32's Fisher relation is a correct theorem on the wrong quantity; X28c
reads its source correctly and fails on tautology.

**The synthesis the coordinator adds:** two of those four are *already repaired
by the contract that cites them*. v15 carries `ρ_P` at **2** not `√2`, and
requires `μ > 0` for Poincaré-Hopf. So L-EQ's evidence base is smaller than
claimed **and** its repairs are further along than claimed. This also closes the
arithmetic audit's open item — the audit listed `ρ_P = 2` as unchecked
`[V]`-grade; it is in fact the L-EQ repair of strike `:200`, measured at four
sizes.

**`n2`, Task A — the standing set stopped moving.** `results/v15_loop_suite5.txt`:
**15 failed, 501 passed**, node ids byte-for-byte the 15 filed in
`V13_D3_LOOP_FORENSICS.md`. Zero resolved, zero new, zero turnover; the two
extra passes fall outside the standing set. The prior round measured membership
turning over by more than half across three snapshots. Across this interval it
did not. One interval is evidence it has stopped, not proof.

**`n2`, Task C — six author-owned strikes, count matches exactly.** Located in
`STRUCK.md` alone, not in `attic/workdonenew.pre-v13.md` or `V13_CLAIM_AUDIT.md`
where the round's reading order assumed. 12 struck constants collapse to 7
claims; 1 is instrument-caused (M2 decay exponent, a `floor=1e-6` artifact); 6
are author-owned. Agreement with the contract is on the **total only** — the
contract names no six, so the number is the only check available.

### it.2 verdicts

**`n2b`.** The sizing repair is smaller than the contract assumed. `ceq/sizing.py`
already holds the calibrated constants — `C_OPERATOR = 3.9` at `:48`,
`DTYPE_MODES['bf16_autocast'] = (2.2, 3.4)` at `:68`. The `6.63x` factor is
recovered as `3.9 * (3.4 / 2.0) = 6.63`. **Zero load-bearing STALE sites.** The
five STALE hits are all `.md` prose across `workdonenew.md`,
`V13_CLAIM_AUDIT.md`, `CEQ_V15_CONTRACT.md` and `attic/workdonenew.pre-v13.md`.
The contract's it.1-4 line "the sizing model replaced by the calibrated one
everywhere it was cited" is therefore a documentation edit, not a code change —
and two of the five hits are *historical statements about the error* rather than
uses of the wrong model, so a blind replace would corrupt the record. Deferred
to a node that reads each site in context.

**Contract arithmetic audit.** Six claims re-derived and CONFIRMED (`floor_1`
0.707107 / 0.935414; CI first fits at `N=23` exactly, `N=22` misses by
`0.0071 sigma`; power 0.80 first at `N=70`; `T* = ddE / ln m` crossover; ceiling
exactly 38; both GL binds). Four FINDINGS, all in clauses this contract
introduces:

| id | finding | class |
|---|---|---|
| A-1 | "TOST retires to `N >= 23`" licenses a verdict whose achieved power at `N=23` is **0.0669** — 93.3% NO VERDICT on two bit-identical arms | M-5 / M-9 |
| A-2 | `w_k = (-1)^k C(-alpha,k)` handed to `scipy.special.binom` returns **NaN at `alpha=1`**, the contract's own cumulative-sum bind. Route: ratio recurrence `w_k = w_{k-1}(alpha+k-1)/k`, verified to match scipy elsewhere and to give `[1,1,1,1,1,1]` at `alpha=1` | V-10 hazard |
| A-3 | `H = alpha + 1/2` carried in without its hypothesis `\|d\| < 0.5`; contract bounds `H > 0.5` below and not above, so `alpha >= 0.5` generates a non-stationary bed whose "Hurst" has no population value | **L-EQ, in the document that introduces L-EQ** |
| A-4 | "the current 22%" has no live producer; both live scoreboards read `0 of 39` (`README.md:55`, `workdonenew.md:44`) | P-1 / P-3 |

The audit also records a correction against **itself**: a 60k-draw Monte Carlo
read power 0.80 first at `N=69` and called the contract off by one; at 4M draws
`N=69 = 0.7985`, `N=70 = 0.8059`. The contract was right and the audit's first
pass was an `M-4` (a single-sample interval read as if it settled a boundary).
Left in the document rather than deleted.

---

## NEXT

**it.6 — the arm is BLOCKED on `n8`, and that block is correct.**

The contract's it.6-7 line is *"ARM PL + three binds"*. One of the three binds —
`g ≡ 0` gives bitwise standard attention — **is refuted**. Building ARM PL now
means building an arm around a bind known to be false, which is what L-LEAN
exists to prevent: *"binds become theorems; tests confirm theorems."* A bind that
is not a theorem cannot become one by being coded.

So it.6 does not build the arm. It waits on `n8`'s ruling and then does one of
three things, decided by that ruling and not before:

| `n8` returns | it.6-7 builds |
|---|---|
| a single operator carrying both (L) and (P) with an **informative** bind | ARM PL as the contract intends, with the new operator and the new identity setting |
| the incompatibility is a **theorem** | ARM PL with the parity claim RETIRED, and the theorem becomes the round's headline: softmax's normalizer is the obstruction to path products. That is a stronger result than the parity claim it replaces |
| only the **two-branch** form works, and it is judged vacuous | ARM PL with parity by resolution statement only, and the vacuous-bind ruling filed to `MISTAKES.md` as a new mechanism |

**Do not pre-build for any of the three.** Two of the three make the third's code
dead, and speculative construction against an unsettled algebra is what produced
four failed hop-2 terms in the prior campaign.

**Still in flight, do not duplicate:** `n1` (prior art, opus), `n4` (BED-K, sonnet),
`n5` (Mars, sonnet — test files landing), `n6` (Venus, opus), `n7` (Neptune, opus
— probe script landed), `n8` (Jupiter-2 fork, opus).

**On collection, still owed from earlier iterations:**

1. Apply the audit's **A-2** route into whatever `n4` built — if its GL code uses
   `scipy.special.binom(-alpha, k)` it cannot evaluate its own `alpha -> 1` bind
   (NaN). Replace with the ratio recurrence, assert `equal_nan=False`.
2. Register BED-K's box before any cell runs (**A-3**): `alpha in (0, 0.5)`,
   `H in (0.5, 1.0)`; the generator REFUSES `alpha >= 0.5` rather than silently
   emitting a non-stationary bed.
3. Wire **A-1** into Mars's attack #2 — the achieved-power column is the
   instrument that makes the `N >= 23` / 6.7%-power finding visible.

**it.8 is unchanged and is now partly pre-paid:** Venus files R1/R2 predictions
(dispatched early at it.3), Mars files attacks (dispatched at it.1), Mercury
prices — and Neptune's `n7` verdict decides whether R2 at `n = 32768` is a cell
that can be run at all.

In flight: `n1` (prior art, opus), `n2` (loop-suite re-date + L-EQ, sonnet),
`n3` (Lean train-gate, opus), `n4` (BED-K + interventional channel, sonnet),
`n5` (Mars's four attacks, sonnet). Do not re-dispatch and do not duplicate
their files.

On collection, in this order:

1. **Read `V15_N3_LEAN.md` first.** It gates everything. L-LEAN forbids training
   before the identity theorems are green, and the contract states that a
   failing `[M]` item means the arm is WRONG. If #1, #2, #5, #6 or #7 comes back
   SORRY or FALSE, that verdict outranks every other node's result and it.5 is
   the report of it, not a repair attempt.
2. **Apply the audit's A-2 route into whatever `n4` built.** If `n4`'s
   fractional/GL code uses `scipy.special.binom(-alpha, k)`, it cannot evaluate
   its own `alpha -> 1` bind. Replace with the ratio recurrence and assert with
   `equal_nan=False`.
3. **Register BED-K's box before any cell runs (A-3).** `alpha in (0, 0.5)`,
   `H in (0.5, 1.0)`. The generator must REFUSE `alpha >= 0.5` rather than
   silently emit a non-stationary bed. This is a pre-registration, so it lands
   before R3 and is timestamped.
4. **Carry A-1 into Mars's attack #2.** The achieved-power column Mars is
   building is the instrument that makes A-1 visible; wire the two together
   rather than filing them separately.
5. **Resolve A-4 before it.10.** The scoreboard's first move is at it.10 and a
   delta from a baseline with no producer is uncheckable. Either the baseline is
   `0 of 39` or the `22%` names a quantity that needs a producer.
6. Only then the sizing-prose edit (5 sites, read each in context — two are
   historical statements about the error, not uses of it).

**it.5 — Lean train-gate verdict.** `#1, #2, #5, #6, #7` green, or the failing
statement named. Nothing trains before this verdict is written down.

- `n1` JUPITER — prior art at **equation level** for every named component
  (SSD/Mamba-2, GLA, RetNet decay mask; negative-eigenvalue SSMs; MWU/Hedge;
  Mori-Zwanzig-in-ML; ARFIMA/fractional nets; generating-partition estimators;
  Hamiltonian/flow-conserving nets). Output `V15_N1_PRIOR_ART.md`. Every entry
  carries `[V-eq]` (statement with hypotheses + one numeric instance run) or it
  is marked `[V]` and declared inadmissible under L-EQ.
- `n2` SATURN — re-run `tests/loop`, re-date the standing failure set against
  the live tree, add L-EQ to `MISTAKES.md` with the pre-v13 section-5 census as
  evidence, and replace the sizing model with the calibrated
  `C_OPERATOR = 3.9 x 3.4 B/elt` everywhere it was cited. Output
  `V15_N2_SATURN.md` + edits to `MISTAKES.md`.
- `n3` JUPITER-LEAN — Lean #1, #2, #3, #5, #6, #7 in `lean/CEQ/V15/`. Every
  theorem compiles under `lake build` or its failing statement is named. Output
  `V15_N3_LEAN.md`.
- `n4` CAMERON/SATURN — BED-K generators (pure-delay `d`, power-law fBm-type
  `H > 0.5`) and the interventional-channel `do()`-bit with per-position bumps
  and Jacobian oracle responses. Fresh files under `ceq/beds/`. Tests first
  (TDA-TDD register). Output `V15_N4_BEDK.md`.

**Nothing trains before it.5.**
