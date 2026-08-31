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

**it.3-4 — collect the five nodes still in flight, then the two follow-ons the
audit opened.**

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
