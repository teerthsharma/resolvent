# FAILS — everything retracted, broken, or unresolved

A result nobody can check is worth nothing. The README carries what works; this
file carries the rest, in the same clear words, complete. It is not an apology
and not a confession — it is the evidence trail that makes the rest of the
repository trustworthy. Every item below was found by the project's own
instruments: its tests, its docstrings, its audits, or a command run directly
against this tree while writing this file.

Paths under `ceqjepa/` name files from the author's local experiment tree. Six
of them are published: `__init__.py`, `operator.py`, `dr1.py`,
`removal_echo.py`, `d1_norman.py` and `d2_nba.py`. An entry citing any other
`ceqjepa/` path records what that file measured; the file itself is not in this
repository.

---

## 1. Retracted claims

Assertions this repository made and then withdrew, with what killed each one.

- **"Theorem 4 gives a depth separation between a resolvent read and a
  fixed-depth stack." RETIRED.** Theorem 4's residual `Q^{L+1}(I-Q)^{-1}R`
  fixes every polynomial coefficient at 1 (the Neumann truncation). With
  FITTED coefficients a degree-8 polynomial reaches `1.448e-02%` of the
  committor's range where the truncation reads `112.6157%`, at `kappa=94.08` —
  a factor of `7,780`. Cayley-Hamilton makes degree `n-1` exact outright
  (measured `5.809e-07` at degree 63, `n=64`), so the separation is capped at
  `n`, and `ceqjepa/v2.py` trains `n=32`. The hypothesis "composes L
  data-dependent linear maps, each degree-1 in Q" also describes a linear
  message-passing GNN, not a JEPA predictor (which reads `s`, not `Q`) and not
  a transformer. Theorem 4 remains TRUE as a bound on linear message-passing
  readers; it is not a separation from JEPAs. See `ceqjepa/ood_kappa.py` and
  `ceqjepa/rank1_field.py`, both of which carry the same correction in their
  module docstrings.

- **"kappa of the bed exceeds `1/TELEPORT = 80`, so the architecture cannot
  represent it." RETIRED, a category error.** `q` is harmonic, so a model
  reproduces the committor and not the chain. An operator built by
  `operator.build_operator` at `kappa_model = 1.000000` fits a target field
  whose true chain reads `kappa = 300.8`, to `1.110e-16` (the float64 floor).
  See `ceqjepa/beds/gridworld.py:119-140`.

- **"kappa = 164.25 on the demo's 14×14 grid." WITHDRAWN AS FABRICATED** — a
  hardcoded string literal in a `print()` and a docstring; nothing ever
  computed it. Struck in `ceqjepa/beds/gridworld.py`'s own module docstring
  alongside the `1/TELEPORT` retraction above.

- **"Every cell's counterfactual is a rank-1 edit sharing one solve", asserted
  for `do(BLOCK a cell)`. WITHDRAWN.** Blocking rewrites the cell's own row and
  rescales its ≤4 neighbours', so rank ≤5, and the grid operator is not lower
  triangular — `ceqjepa/intervene.py:54-66` refuses it outright with "the
  operator would stop being lower triangular". The claim IS true for
  `do(CLAMP one cell's outgoing row)`, which is rank exactly 1
  (`sigma_2/sigma_1` measured, `RANK1_TOL`-bounded) — see
  `ceqjepa/rank1_field.py`, which is its producer and exists specifically
  because the original `BLOCK` claim's supporting measurement had no producer
  anywhere in the tree.

- **"Sum q = 1 prevents encoder collapse." Struck earlier.** A collapsed
  encoder satisfies it BETTER (`7.772e-16` vs `2.290e-13`). Recorded in
  `docs/ARCHITECTURE_OPTIONS.md:288-289`, which is untracked.

- **"`exp` is never zero, so the attention family cannot represent a closed
  class; an exact-zero gate can." RETRACTED 2026-09-20, by construction.**
  Under a causal mask any row-stochastic matrix is lower-triangular, so its
  eigenvalues are exactly its diagonal and "closed class" collapses to
  `P_ii == 1.0`. A real sparsemax projection (Martins–Astudillo, sort-based,
  not the identity shortcut) on logits `[10.0]`, `[0.5, 0.5]`, `[-5,-5,5]`,
  `[-5,-5,0.5,0.5]` returns `diag(P) = [1.0, 0.5, 1.0, 0.5]` bitwise, and
  `eigvals` gives `(1+0j)` with multiplicity 2. All 14 nonempty proper subsets
  were checked with `det(I-Q)` computed twice — exact `Fraction` arithmetic and
  float64 — agreeing bitwise on every one: 11 complements closed, 3 genuinely
  transient. That 11-vs-3 split is what shows the test discriminates rather
  than returning zero everywhere. A second construction gives six simultaneous
  closed classes at `n=6`. The Lean theorems `softmax_unique_absorbing` and
  `gate_zero_second_absorbing` remain true and both still typecheck at 0
  `sorry`, but they bind `exp`-based scores only; sparsemax, entmax and top-k
  routing are not built from `exp` and `Real.exp_pos` does not reach them.

- **"A mask closes by position, a learned gate closes by content, and that
  separates the two families." RETRACTED the same day.** The sparse-attention
  family also produces exact, learned, content-dependent zeros at positions
  unknown before the input arrives. Measured against a rival given every
  advantage — hidden width 64 against the gate's 8, a feature handing it
  segment identity almost directly, and a learning-rate sweep the gate did not
  get — sparsemax **won** per-row boundary accuracy, `0.9919` against `0.9845`,
  ahead on 3 of 5 seeds.

- **"Cross-row disagreement is invisible at `k=1` and becomes a corridor under
  composition." RETIRED 2026-09-20.** The leak does grow under powers, from
  `0.000895` at `k=1` to `0.732453` at `k=8`, crossing its `0.0043` bar. But
  the confinement the hypothesis requires is absent: sequences whose rows
  *agree* leak `0.676408` at `k=8`, within `1.08×` of the disagreeing ones. That
  is generic `A^k` mixing of a stochastic matrix, not a corridor at the
  boundary. Separately, a **learned** gate is not bitwise zero under powers —
  224 of 12,000 `(seed, k, row)` leak values are nonzero, in 4 of 5 seeds; only
  the seed whose learned `m_b` happens to be exact survives all eight.

  What survives, narrowed: a path product forces cross-row agreement
  *structurally*, 1500 of 1500 held-out sequences exactly consistent against
  sparsemax's 1352 of 1500, with a tail to spread 20 of 24 positions. It pays an
  exactly enumerated price — `i` reachable zero patterns against sparsemax's
  `2^i − 1`, every gate pattern a strict subset, `31.875×` at `i=8`. At `i=5`
  sparsemax reaches support `{0, 2, 4}`, holes at 1 and 3, which no gate
  configuration can produce, because one `m_k` multiplies every pair spanning
  it. That measurement uses a **hand-set** boundary at a single application, and
  is not evidence that a model can learn where to place the zero.

---

## 2. Currently broken

Reproduced directly against this tree while writing this file. Commands,
exact exit codes, and causes below; nothing here is copied from an older run.

- **`python -m pytest tests/gate0 -q -p no:randomly`** → exit `2`, zero tests
  run: `Interrupted: 6 errors during collection`. All six collection errors
  are the same `ModuleNotFoundError: Could not import module 'PreTrainedModel'`
  from `from transformers import PreTrainedModel` at
  `ceq/hf/modeling_ceq.py:93`, which `ceq/hf/train.py:41` imports. The cause is
  the environment, not the code: `requirements.txt` pins `torch==2.5.1`, the
  installed torch has drifted to `2.14.0`, and the installed
  `torchvision 0.20.1+cu121` was built for 2.5.1, so `import torchvision` raises
  `RuntimeError: operator torchvision::nms does not exist`, which
  `transformers 5.3.0` reports as the missing `PreTrainedModel`.
  `tests/loop/test_no_struck_constant_ships.py::test_no_struck_value_is_reachable_in_the_shipped_costs_dict`
  fails on the same import.

- **`python -m pytest tests/loop -q -p no:randomly`** → exit `1`: `15 failed,
  466 passed, 8 skipped, 3 warnings in 46.33s` (2026-09-11, with the strike
  registry restored; the reading before the restore was `16 failed, 451 passed,
  8 skipped`). Two of the 16 failures are the
  repo's own README-consistency guards —
  `test_corpus_is_recoverable_and_verifiable.py::test_some_countable_unit_of_the_corpus_equals_the_readme_figure`
  and `::test_the_readme_derivation_matches_the_corpus_it_describes` — which is
  the alarm working, not a new defect. The remaining failures span conftest
  import ordering, the struck-constant costs dict (the same torchvision import
  as above), boundary-node propagation, a weight record's identity field, and
  out-of-sample scoring; see the full names in the run output.

- **`python -m pytest tests/w11/test_w11_claims_resolve.py`** → 4 of its 8 README
  guards fail by construction after the 2026-09-11 README rewrite. They assert the
  pre-rewrite document's shape: a `## Limits, first` heading positioned before a
  `## The signed influence property, and who already had it` section, an R1-R6
  requirements table marked `DELETED`/`ALIVE`, and the failed intervention's
  `2.6151` / `4.2107` inside that limits section. The current README has none of
  those sections, by the owner's decision that the failure record lives in this
  file instead. Two of the six were repaired rather than retired: the README now
  cites tests by full node id and carries the exact reproduce commands the guard
  names, so `test_every_test_name_in_the_readme_resolves` and
  `test_the_reproduce_commands_are_real` pass (RUN 2026-09-11, `2 passed`). The
  fourth failure, `test_the_collector_actually_finds_known_tests`, needs
  `test_signed_operator_reaches_negative_influence`, deleted at `c71527a`.

- **`python -m pytest tests/mars_v20/test_p12_absence_proof_falsified_by_recording_it.py`**
  → `2 failed, 4 passed` (2026-09-11; `3 failed, 3 passed` at `6aef750`). The one
  repaired is `test_the_repaired_invocation_is_the_one_struck_md_publishes`:
  STRUCK.md's `0.743864` row now publishes the path-excluded search and the control
  symbol the same search must still find. The two that remain assert what no text
  edit can restore — that the bare `git log -S "def absorbing_boundary_kernel"`
  across all refs returns zero (it returns 5 commits, each one a document quoting
  the search), and that nothing recording the strike sits inside the search's own
  reach (STRUCK.md, the guard itself and two test files do). They keep the P-12
  mechanism visible on purpose.

- **`python scripts/k_cost.py`** → exit `1`, same root cause as gate0: the
  import chain `scripts/k_cost.py:109` → `ceq/hf/train.py:41` →
  `ceq/hf/modeling_ceq.py:93` → `ModuleNotFoundError` on `PreTrainedModel`.

- **`python scripts/k_cert.py`** → exit `124`, times out on CPU, no
  certificate produced. Not re-run here (expensive; recorded).

- **`python inspector.py`** → exit `1`,
  `FileNotFoundError: [Errno 2] No such file or directory: 'STATE.md'` at
  `inspector.py:562`. `STATE.md` was deleted at `c71527a` ("Remove the round
  reports from the tree, and commit the paper's sources beside it") and never
  restored; `git show --stat c71527a` confirms the delete.

- **The whole suite has never completed.** `python -m pytest tests/ -q` runs
  at roughly 9% in 15 minutes on CPU, estimated >3h, so no total pass/fail
  count exists — `MODEL_CARD.md:732-734` says so explicitly and instructs readers
  not to quote one. What is verified instead: 955 tests collect in 9.6s
  (measured 2026-08-25, and stated in `MODEL_CARD.md` to grow as tests are
  added — re-measure rather than trust it).

- **Of 19 `ceqjepa` modules run as `python -m`, 11 reach a terminal pass
  line; 6 do not terminate within 300-480s**: `chess_policy`,
  `bed_headroom`, `curriculum`, `positive_control`, `run_causal_test`,
  `move_ablation`. Not re-run here — six modules at up to 480s each is
  expensive; recorded from this session's own measurement.

---

## 3. Measured negative results

Experiments that ran correctly, under a pre-registered kill condition, and
returned "no."

- **The OOD-kappa experiment**, `ceqjepa/ood_kappa.py` (its docstring carries
  the full table). Claim on trial: under an operator intervention, a resolvent
  read's committor error is kappa-invariant while a learned direct read's
  grows. **DEAD on the pre-registered kill condition, 5/5 seeds, wrong sign.**
  MAE: constant `0.0774`/`0.0774`, operator(solve) `0.0620` id / `0.0741` ood,
  direct(MLP) `0.0416` id / `0.0718` ood, operator(trunc L=8) `0.0618`/`0.0735`.
  The direct read wins both in- and out-of-distribution. The truncation
  control ties the exact solve to `0.0006` at both ends — the solve
  contributes nothing — because the trained Q-head emits `kappa 1.34` against
  a true `93.51` at OOD inputs, off by 70×, having never seen a self-loop in
  training. Shuffled-label controls both land at the constant predictor
  (`0.0777`/`0.0778`); both live arms beat the constant in-distribution. Extra
  training does not rescue it: at 3× the steps the operator arm's
  in-distribution error improves `0.0620 → 0.0548` while OOD stays pinned at
  `0.0740`. The surviving statement: an exact solve downstream of a *learned*
  operator head does not inherit the solve's exactness, because the head is
  the distribution-bounded component. Also worth keeping: kappa-invariance is
  FREE on a time-change bed — the constant predictor is exactly invariant
  while predicting nothing — so invariance alone is never evidence.

- **The operator did not separate from a 2,556-parameter MLP** on the
  synthetic bed: `+0.0264`, sd `0.0358`, 4/5 seeds, against a frozen bar of
  `>= +0.020 AND 5/5`. The docstring explanation that "the bed's kappa was too
  small" is itself a misdiagnosis: that MLP reads `s`, not `Q`, so it was
  never a polynomial in `Q` at any kappa.

- **The T4 causal-arm run** (kernel `ceq-jepa-dcm-1-causal-arm-t4`, 4h43m, 3
  seeds × 12,000 steps): held-out perplexity ROSE in 15 of 15 intervals while
  training loss fell to `0.03-0.14` nats, 257 parameters per training label.
  Best checkpoint at step 2,000; the run saved step 12,000 — `wasted/useful =
  10000/2000 = 5.00`. The sharpness margin was `I_q - J(q) - KL = 0.1196 -
  (1.3967 + 0.0025) = -1.2796`, three times more negative than a deliberately
  temperature-sabotaged control.

- **The teleport that keeps Sherman-Morrison's denominator safe attenuates
  the interventional signal the do-arm trains on by 43.1%** (1,859
  blocked-cell solves, `G=9 gap=3`: mean `|dq|` `0.021501` true vs `0.012590`
  teleported; 7 sign flips among the 724 solves clearing `0.01`). At `c=0` the
  same path is exact to `1.887e-15`. Invisible to every check that runs,
  because the teleport is row-stochastic so committor rows still sum to 1.
  See `ceqjepa/beds/gridworld.py:119-140`.

- **The logit-Nash stance, `ceq/nash.py` (GitHub issue #2).** Claim on trial: a
  stance solved as a quantal-response equilibrium composes two sign flips seen
  only separately, where the learned stance of the `signed` arm does not.
  **DEAD on both pre-registered kill conditions, 5/5 seeds, in every
  configuration.** OOD NRMSE on W7's held-out composition, corpus seed 0,
  training seeds 0-4, 400 steps, from `python scripts/nash_repairs.py`:

  | arm | mean ± sd | range | beats `signed` | below 1.0 |
  |---|---|---|---|---|
  | `signed`, the learned stance | 2.7333 ± 1.0235 | 1.6122 – 3.9058 | — | 0/5 |
  | `nash` as shipped | 5.2888 ± 0.6041 | 4.6051 – 6.2479 | 0/5 | 0/5 |
  | + learned game bias | 5.0808 ± 0.3683 | 4.6539 – 5.6255 | 0/5 | 0/5 |
  | + per-example `tau` | 4.8480 ± 0.7695 | 3.9224 – 5.4408 | 0/5 | 0/5 |
  | + both | 4.6142 ± 0.7440 | 3.6929 – 5.2668 | 0/5 | 0/5 |

  The two repairs are the arm's two defects known before the run, each with its
  fix already stated: the learned game bias `ceq/arms.py` built and never read
  (MISTAKES.md P-4, second instance; now deleted), and the batch-shared
  temperature that `tests/deimos/test_deimos_r9_iteration1.py` measured at 2.2×
  the median per-example value. Together they close 26% of the gap to `signed`
  and win no seed. The issue's single run put the gap at 38%; over five seeds the shipped
  arm is 93% worse, and its best seed is worse than the learned stance's worst.
  The equilibrium machinery itself is sound (the seven structural tests pass: a
  genuine fixed point, non-affine at every probe radius, signed, L1-bounded by
  `rho`, nilpotent), so the module stays as an instrument and the two kill tests
  are `xfail(strict=True)`, which fails the run the day either bar is earned.

---

## 4. Known defects not yet fixed

- **The counterfactual-field demo page's `computeField()` reads the wrong
  row.** `buildChain(b)` does not remove `b` from `tList`, so `nT` is
  unchanged and `b`'s own row survives, but `computeField` recomputes the
  cursor's index as its position in `tList` *with `b` skipped*. Every blocked
  cell preceding the cursor in raster order reads one row early. Measured on
  the demo's frame-0 grid, cursor at `r7c7`: 92 of 168 blocked cells report a
  wrong delta-q, mean absolute error `0.0313`, worst `0.0908` — against a mean
  true `|delta q|` of `0.0087` and a largest true `|delta q|` of `0.0552`.
  Fix is one line: `qb[tIdx[target]*2]`; delete the `ti` loop. Documented at
  `ceqjepa/beds/gridworld.py:39-49`; the page itself lives outside the
  tracked tree at `scratchpad/counterfactual-field.html`. A second, unrelated
  defect on the same page: it claims a Sherman-Morrison rank-1 update verified
  against a dense re-solve, but the shipped `computeField()` performs a full
  `buildChain` + `solveCommittor` per candidate — there is no rank-1 update in
  the file. Neither bug is inherited by `ceqjepa/beds/gridworld.py`, which
  blocks a cell by writing a wall into a copy of the grid.

- **`ceqjepa/intervene.py` refuses non-triangular operators**, which is the
  exact object `ceqjepa/rank1_field.py`'s claim lives on (see §1). A Woodbury
  path was measured exact to `1.07e-14` with a matching NaN mask but was never
  landed.

- **`ceqjepa/v2.py`'s variance hinge is at its maximum (`32.0 = d`) at exactly
  the collapsed state while its gradient is exactly zero there** — a
  stationary point of the same shape as the zero-init LoRA death this project
  already ate once. Escape force returns at any asymmetry (`1.115e+00` at
  `1e-6`, measured against an assertion of `> 0.1`). Recorded and asserted in
  the module itself (`ceqjepa/v2.py`, the `THE MUST-FIRE` block); the standing
  mitigation is never zero-init the encoder output.

- **One cursor cell in thirteen has a counterfactual field entirely below
  `1e-9`** (measured by `ceqjepa/rank1_field.py`, `DEGENERATE_FLOOR = 1e-9`)
  — the clamp changes nothing measurable there, the rank statistic reads
  noise over noise (`~1e-1`), and the demo must render something at those
  pixels regardless.

---

## 5. Never verified

Claims with no producer located, stated as such rather than silently dropped.

- **"This repository's instruments measure its models rather than its beds."
  UNTESTED as of 2026-09-20, and the bed built to settle it cannot.** Five
  instruments failed this on the same day, each scoring a property of the draw:
  effective rank prefers a frozen-random encoder by `22.08×` at `D=64`; a
  hitting-time R² window is 89% draw geometry, with arm error differing `1.06×`
  between draws where counting won and broke while the label's own spread
  differs `5.16×`; an unscoreable gate sits flat at `0.89` across four decades
  of budget because it counts the sealed share of the draw; a refusal head is
  beaten `7.3×` on recall by a one-line non-causal shortcut reading no
  structure; and BED-H's magnitudes are draw properties with only their
  ordering assertable.

  The proposed cure — score every instrument as `f(model, draw) − f(null, draw)`
  on the same seed, so additive draw terms cancel — regressed at slope
  `−1.4513`, se `0.8439`, `t(3) = −1.7198`, `corr = −0.7046`, `n = 5`. The
  follow-up hypothesis was that this is what *additive* pairing of a
  *multiplicative* effect looks like, and that a log-ratio would cancel where a
  subtraction did not. **That could not be evaluated.** At the refusal bed the
  null fires on zero instances at all five seeds, so its precision is undefined
  and its recall is exactly `0.0`; `0 of 5` seeds pass the strict-positivity a
  ratio requires. The model side is well defined throughout — precision
  `1.0000, 0.8333, 1.0000, 0.6667, 0.6667` — so the failure is the twin, not the
  transform. The episodic-restart fix already in the tree guarantees eventual
  sink access, which is what makes the shortcut silent.

  Recorded rather than resolved: the additive slope's 95% interval at `n = 5`
  spans roughly `[−3.7, +0.8]`, containing zero, no effect and a doubling, so it
  is a statement about power and not a measurement. A bed with a null that
  scores something is a precondition for answering this at all.

- **The pre-rewrite README's corner floats `4.472918` / `1.144938` / `5.335671`.** A
  tree-wide search of `.py`, `.lean`, `.json`, `.jsonl`, `.txt` (re-run while
  writing this file) returns **zero hits**. They existed only in prose and left `README.md` in the 2026-09-11 rewrite
  (the earlier text: `git show 6aef750:README.md`).

- **The pre-rewrite README's §3.1 identity-bind numbers, §3.2's R1 table
  (`README.md:185-205` at `6aef750`, the `crossed` / `NO READING` populations and the
  `0.634-0.662` / `1.113-1.152` NRMSE bands), and every §5 device-round number
  (`README.md:302-392` at `6aef750`).** No producer was located for these figures, and the
  pinned stack (`torch 2.5.1+cu121`) is not installed on the author's box, so
  none of them can be regenerated here to check. They left `README.md` in the
  same rewrite.

- **`#print axioms` is absent from 10 of the 13 project Lean files.** Of the
  13 files under `lean/` (12 in `lean/CEQ/` plus `lean/CEQ.lean`; the
  dependency tree under `lean/.lake/packages/` is not part of this count),
  only `V15Phase.lean`, `V15Source.lean`, and `V16Domain.lean` carry
  `#print axioms`, though the pre-rewrite README asserted every theorem runs through it. Zero
  `sorry` across the same 13 files IS verified — a tree-wide search finds none
  outside prose comments asserting their absence.

- **The pre-rewrite README's `lake build` job count `[1530/1531]`.** The build is cached and emits
  no count when run. Exit `0` on the cached build is verified.

- **`ceqjepa/dr1.py`'s docstring figure of 7,966 clamp interventions** (100%
  agreement between the refusal rule and graph reachability) comes from the grid
  bed, whose producer is not in this repository. The published self-check,
  `python -m ceqjepa.dr1`, prints its own 400-case table (138 undefined, 262
  defined) instead, and that table is the one the README cites.

---

## 6. The other ledgers

FAILS.md is the front door, not the whole house. Detail lives in dedicated,
machine-checked ledgers:

- **[MISTAKES.md](https://github.com/teerthsharma/resolvent/blob/master/MISTAKES.md)** — the failure-taxonomy proper: 74 failure
  mechanisms across four classes (V — vacuous controls, 30; P — provenance
  failures, 16; M — measurement failures, 21; D — design-level failures, 7),
  each with an instance, a rule, and a check.
- **[STRUCK.md](https://github.com/teerthsharma/resolvent/blob/master/STRUCK.md)** — every constant this project has withdrawn:
  12 entries, rendered from a module-level registry in
  `tests/loop/test_no_struck_constant_ships.py`, never hand-edited.
- **[docs/canon/CORRECTIONS.md](canon/CORRECTIONS.md)** — the canon's own
  correction log; the only door through which the books 00–09 change.
- **[V17K_RULINGS.md](https://github.com/teerthsharma/resolvent/blob/master/V17K_RULINGS.md)** — 3 open rulings out of 7 total
  (rows 4-7 are CLOSED): determinism regime, the corner criterion, and matched
  parameter counts, each blocked on a specific named measurement or file, not
  on absence of effort.
