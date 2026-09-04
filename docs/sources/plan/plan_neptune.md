# PLAN — NEPTUNE (LINUS): the systems roadmap, as milestone cards

*NEPTUNE, 2026-09-03, HEAD `207e7b9`, branch `v17k-gate0`. The judged sections (`judge/sec_shape.md` §4.5, `judge/sec_apparatus.md` §A.9–A.11, `judge/bind_ledger.md`) outrank every design and note where they differ; this file is the systems slice of the programme (`THESIS_CORRECTIONS_2.md` §0): the author develops it alone, slowly, on the certified RTX 4060 Laptop (`7.996 GiB`, torch `2.5.1+cu121`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, READ `COSTS.md:53-63`), in any order the DAG allows. Evidence classes: `RUN` (a number executed on this box in the cost session, carried from `sections/sec_cost.md` with its section), `RUN[this]` (arithmetic re-executed this session with a Python one-liner), `READ path:line`, `CITED [V]` by canonical `references.bib` key, `DERIVED` with steps, `[FITTED]`/`[ASSUMED]` per `COSTS.md:16-20`. No card launches anything on Kaggle; the author's explicit yes is a node (`READ kaggle/README.md:9-11`; memory rule *Kaggle launch requires yes*). No card writes code here; every "what to build" is a specification a developer implements, and the one code edit the ledger names as owed (C-1) is card N-01.*

## 0. The shape of this slice

The object priced is `O = (1 − γ) P (I − γP)^{-1} V` on a causal row-stochastic softmax `P` with identity rows on the declared sets, solved by one forward substitution (`judge/sec_shape.md` Prop. 5). Four facts fix the whole roadmap, each with its provenance:

1. **The exact path is cheaper than one Neumann hop at the shipped geometry.** `solve + Pz` fwd+bwd `2.514 ms` vs `PV` `1.473 ms` vs one hop `3.001 ms` at `n = 2048, s = 64, d = 16`, float32 (RUN, `sec_cost.md` §4.x.3, median of 14 after 2 warm-ups, synchronize-bracketed). Truncation never wins in MACs (`K·s²d/2 < s²d/2` needs `K < 1`, DERIVED §4.x.3). The Neumann route survives as a certificate instrument, not as a cost path.
2. **No finite-state dual exists for softmax `P`** (`hu-2025-ssdtheory` [V], rank `T` for rank-1 logits), so the chunked-WY inter-chunk state of `yang-2024-deltanet` [V] Eq. 10 is closed; the chunked path re-shapes the same `s²d/2` MACs into `s/C` GEMMs (depth `s/C`, memory `O(sC)`), it does not reduce them (DERIVED §4.x.2). The subquadratic path is occupied: `zhao-2026-structuredsparse` [V], `Õ(n^{4/3} d)`, exact on tiles, approximate across blocks.
3. **The shape inherits the `O(S²)` memory class** unless the backward recomputes tiles (`READ ceq/sizing.py:9-22`: `3.9` tensors of `[B,H,S,S]` per layer; ratio `1.44×` at `seq 128` to `8.06×` at `seq 2048`). `solve_triangular` has no bf16 CUDA path in torch 2.5.1 (`[U]`, card N-19 verifies), so the solve's `[S,S]` operand stays at `4.0` B/elem where the softmax's autocast term reads `3.341` (`READ COSTS.md:91`).
4. **`solve_triangular` ran bitwise forward and backward under strict mode with no torch-documented guarantee** (RUN §4.x.6), which makes the softmax-corner shape the first gated wing whose *training* step can run under `use_deterministic_algorithms(True)`; `cumprod`'s backward raises (`READ COSTS.md:149-154`). That line is not yet in any device certificate (P-1) — card N-01.

Phases: **Ph0** certificate lines and must-fire batteries (0 GPU-s to seconds); **Ph1** the exact path as the shipped route, priced end to end; **Ph2** the cost law in `s` (the C9 harness debt); **Ph3** chunked, blockwise, CSR and Mapper; **Ph4** memory and sizing; **Ph5** envelope, HF package, Kaggle gate. Every card names the `MISTAKES.md` mechanism it is built against. The five cheapest decisive cards are marked **★**.

## 1. The DAG (D-1: parallel only on nodes with no shared repository state)

| card | needs | runs beside | evenings | price |
|---|---|---|---|---|
| N-01 ★ | — | everything | 1 | ≈1 GPU-s |
| N-02 | N-01 | N-03, N-04, N-14, N-19 | 1 | ≈1 GPU-min |
| N-03 ★ | — | everything | 1 | 0 GPU-s |
| N-04 | — | everything | 1 | 0 GPU-s |
| N-05 ★ | N-03, N-04 | N-14, N-18, N-19 | 1 | ≈17 s |
| N-06 | N-05 | N-18 | 1 | ≈98 s |
| N-07 | N-01, N-05 | N-06 | 1 | ≈45 s |
| N-08 | N-06 | N-18, N-20 | 2 | 206–537 GPU-s band |
| N-09 | N-08 | N-10 | 1 | inside N-08 |
| N-10 | N-08 | N-09 | 1 | inside N-08 |
| N-11 | N-09 | N-13, N-18 | 3 | NOT MEASURED |
| N-12 | N-11, N-18 | N-13 | 2 | NOT MEASURED |
| N-13 | N-09 | N-11 | 2 | NOT MEASURED |
| N-14 ★ | — | everything | 1 | 0 GPU-s |
| N-15 | N-03, N-10 | N-11, N-13 | 2 | ≈2.6 s at `s = 64` |
| N-16 | N-15, N-14 | N-17 | 2 | NOT MEASURED |
| N-17 | N-15 | N-16 | 3 | NOT MEASURED |
| N-18 | N-05 | N-08 | 2 | ≈9 GPU-min |
| N-19 ★ | — | everything | 1 | seconds |
| N-20 | N-18 | N-08 | 1 | ≈3 GPU-min |
| N-21 | N-18 | N-20 | 1 | ≈5 GPU-min |
| N-22 | N-04 | N-05..N-21 | 1 | seconds |
| N-23 | N-05, N-04 | N-08..N-21 | 2 | 0 GPU-s |
| N-24 | N-01, N-02, N-18, N-22, N-23 | — | 1 (local) | 0 GPU-s local; Kaggle NOT LAUNCHED |

Serial depth of the longest chain (N-03 → N-05 → N-06 → N-08 → N-09 → N-11 → N-12 → N-15 → N-17 → N-16): `1+1+1+2+1+3+2+2+3+2 = 18` evenings (DERIVED). The memory chain (N-18 → N-20/N-21) and the envelope/HF chain (N-22, N-23) run beside it. Nothing in Ph5 shortens on Kaggle; the certificate lines (N-01, N-02, N-18) are prerequisites of the gate, not products of it (the G0.10 circularity, `READ V17K_RULINGS.md:10-16`).

## 2. Phase 0 — certificate lines and batteries

**N-01 ★ — Add `solve_triangular` as the fourth determinism quantity to the device certificate.**
- phase: Ph0. prerequisites: none. independent-of: all.
- what to build: a specification for one more entry beside `hop / forward / gradient` in `scripts/k_cert.py::determinism_at_64` (`READ scripts/k_cert.py:579-640`, the per-quantity `repeatability` closure with its `executable=False` branch): `M = tril(rand(64,64)) + I` batched `[2048,64,64]`, `B = randn(2048,64,16)` float32, forward `solve_triangular(M, B, upper=False)` and its backward through `mse`, 8 repeats, both flag regimes, single stream, workspace pinned before process start (`READ V17_R4_RETAKE_PRICE.md:178-181` via `sec_cost.md`: set in-process it does not take). Journal fields: `solve_fwd.{bitwise,max_abs,executable}`, `solve_bwd.{…}`, `documented_guarantee`, `trsm_dispatch`. The certificate line the author will read in `COSTS.md` §1.6 is fixed here: *`solve_triangular` fwd: bitwise, max|Δ| = 0.0, 8 repeats | bwd: bitwise, 0.0 | flag ON and OFF | torch-documented guarantee: NONE (absent from both lists of the `use_deterministic_algorithms` docstring, READ torch 2.5.1) | cuBLAS `trsm` dispatch: [U] | one box, one process, one stream.*
- what to prove: nothing; a repeatability reading, class RUN, never a guarantee (V-16 forbids reading silence as a pass).
- what to measure: `python scripts/k_cert.py --out results/k_cert_local.json` on the 4060; statistic `max|Δ|` over 8 repeats per quantity, 3 outcomes distinguished (bitwise / drifting / not executable).
- PASS: `0.0` forward and backward under both regimes, reproducing RUN §4.x.6. KILL: any `max|Δ| > 0` under the flag → the shape's training cells are journalled `deterministic_regime = warn_only` like the scan arms, and Ruling 1's bitwise bar is not claimed for training (`READ V17K_RULINGS.md:39-45` via `sec_state.md` S.4); a `RuntimeError` on the backward → the same fallback, and the "first gated wing under strict mode" sentence is deleted.
- price: ≈1 GPU-s [RUN class, the record's determinism block ran inside a 523.9 s certificate run, `READ COSTS.md:61`].
- mechanism: P-1 (a number with no live producer), V-16, V-23 (the softmax control was never tested under strict mode — its row is added beside).
- deliverable: `results/k_cert_local.json` (new keys) and one line in `COSTS.md` §1.6.
- evenings: 1.

**N-02 — Make the solve microbenchmark a producer.**
- phase: Ph0. prerequisites: N-01. independent-of: N-03, N-04, N-14, N-19.
- what to build: a `solve_increment` block in `scripts/k_cert.py` timing `PV`, `solve + Pz`, and Neumann `K ∈ {1,2,4,8,16}` fwd+bwd at `n ∈ {2048,4096,8192}`, `s = 64`, `d = 16`, `γ = 0.5`, float32, `torch.cuda.synchronize` bracketed, 2 warm-ups, median of 14; journal fields `pv_ms, solve_ms, neumann_ms[K], increment_ms, n, s, d, gamma, cublas_workspace, producer_cmd`. Also peak bytes above a reset baseline per route.
- what to prove: none.
- what to measure: the increment `solve_ms − pv_ms` per `n`; the reference values RUN §4.x.3 are `1.041 / 2.312 / 4.542 ms` (RUN[this]: `2.514−1.473`, `4.659−2.347`, `9.916−5.374`).
- PASS: each within the record's `±12 %` clock spread (`READ V17_R4_RETAKE_PRICE.md:143-150` via `sec_cost.md` Limits) of the RUN values; hop `K = 1` slower than the solve at every `n`. KILL: increment at `n = 2048` above `2.2×` the RUN value → the per-op floor is re-labelled P-8 and every price card below carries the measured factor; solve slower than one hop → the Neumann route re-enters as a cost path and N-10 is promoted.
- price: ≈1 GPU-min [DERIVED: 3 `n` × 7 routes × 16 calls at ≤ 115 ms].
- mechanism: P-1, P-8, M-3 (the increment measured at the shipped `s`, never scaled from a smaller one), M-8.
- deliverable: `results/k_cert_local.json` (`solve_increment` block); `COSTS.md` §1 new subsection.
- evenings: 1.

**N-03 ★ — The kernel-path must-fire battery, as a test specification.**
- phase: Ph0. prerequisites: none. independent-of: all.
- what to build: one test module specification, CPU float64, each check with its honest half and its planted negative and the O(1) failure the plant must produce (`judge/sec_shape.md` §4.5, `sec_cost.md` §4.x.9): (1) parity `torch.equal(O(0), P V)` against the lane's own `softmaxAttn`, rejection `γ = 0.5 ⇒ 2.3002850040264393` (RUN[coord]); (2) `solve_triangular` vs dense inverse `≤ 1.8e-15`, plant: a dense `M` with `upper=False` must disagree; (3) certificate in vector units `max|z_solve − z_K| ≤ γ^{K+1}/(1−γ)·‖V‖_∞` with `‖V‖_∞` printed, plant: rows scaled to `1.5` at `γ = 0.6, K = 2`, `7.29` vs `0.54` (the *convergent* plant, `judge/sec_shape.md` Prop. 4; the `γ = 0.7` plant is labelled divergent); (4) segmentation: a `−∞` logit column at `c` gives `torch.equal` zero blocks across `c`, plant: a `−30` logit leaves the block non-zero (`e^{−30} = 9.36e-14`); (5) CSR fill-in: a schedule dropping tile `(k,l)` but keeping `(k,m),(m,l)` makes the exact solve's `(k,l)` block non-zero and the exact route is refused; (6) the three-outcome determinism table from N-01 is read, never re-run here; (7) a missing key in any identity script raises (the `basin_A` default that read `0.858` is the filed V-16, `sec_refuted.md` C12).
- what to prove: none; the theorems behind (1), (2), (4) are `judge/sec_shape.md` §4.6 rows 1, 5, 6 and are cited only once they build (L-LEAN).
- what to measure: `python -m pytest tests/shape/test_kernel_path.py -q` (path is the spec's; the file is the developer's); the invariant is the failure set, not the pass count (`sec_measured.md` M.9).
- PASS: every honest half passes and every plant fails at O(1), counts printed. KILL: any plant that passes (an empty rejection region) strikes that check from the battery and the paper's corresponding must-fire sentence.
- price: 0 GPU-s.
- mechanism: V-24, V-3 (attained bounds declared definitional), V-2, V-16, V-17 (units).
- deliverable: `docs/CEQ_KERNEL_PATH_BATTERY.md` (the spec) and the test file's journal row under `results/`.
- evenings: 1.

**N-04 — The `route` slice of the identity manifest, with its drift plant.**
- phase: Ph0. prerequisites: none. independent-of: all (the systems slice of the ledger's P0.2).
- what to build: manifest fields `route ∈ {solve_triangular, neumann_K, segmented, csr}` with `K`, chunk `C`, tile `B`; `diag_convention ∈ {kept, removed}`; `delta_vec, delta_bare, one_over_1mg, V_inf`; `deterministic_regime, cublas_workspace, threads, dtype`; `producer_cmd`; a missing declared field is a **refusal**, not an `absent` entry (`READ scale/identity_manifest.py:141-151` via `design_instrument.md` §3 reports rather than raises; the shape lane has no stored manifests to protect).
- what to prove: none.
- what to measure: the one-line-drift pattern (`READ V20_R15_WING_MANIFEST.md:148-152` via `sec_state.md`): flipping `route`, `K`, `C`, `diag_convention` or `deterministic_regime` by one unit must move `manifest_hash`; a cell is FOUND iff `results/` holds a record with the arm's `kind`.
- PASS: every flip moves the hash; a missing field raises. KILL: any flip that leaves the hash fixed → the field is not identity and the certificate printed under it is unattributable (L-2's recurrence).
- price: 0 GPU-s.
- mechanism: L-2 (`0 of 24` attributable, `READ V20_R15_LEAP_LEDGER.md:23`), P-1, V-16, M-10, M-16, L-CERT.
- deliverable: `docs/CEQ_MANIFEST.md` §route; the test's journal row.
- evenings: 1.

## 3. Phase 1 — the exact path as the shipped route

**N-05 ★ — Run the ChaCAL-diag smoke test at `γ = 0.9` on BED-M through the solve path.**
- phase: Ph1. prerequisites: N-03, N-04. independent-of: N-14, N-18, N-19.
- what to build: the arm `ChaCAL-diag` (the shape with `𝒜 = ∅`, diagonal kept, `(1−γ)` factor carried, `route = solve_triangular`) at `4,769` parameters recounted per arm (Ruling 3), `n = 2048, s = 64, d_model = 16`, 150 steps, 8 seeds, on BED-M `e3_t2` as a *containment* cell (D-2: BED-M is contained, never won; `shape − corner-3` is VOID by registration). The cell journals the N-04 fields and a `kind` so it is FOUND.
- what to prove: none.
- what to measure: seconds per cell with `synchronize()`; `eval_nrmse` per seed is recorded but decides nothing.
- PASS: cell time within the law-vs-measured band `−1.8 % … +14.4 %` of `1.680 s` ([FITTED] `0.010162 s/step` + RUN `0.001041` increment, RUN[this] `150 × 0.011203 = 1.68045 s`; band `READ V17_R4_RETAKE_PRICE.md:228-229` via `sec_cost.md` §4.x.8); 8 seeds ≈ `13.4 s` plus `4.0 s` fixed ≈ `17.4 s`; the parity plant of N-03 (1) runs inside this cell at `γ = 0`. KILL: above `2.2×` the band (the dispatch gap's lower end, `READ scale/m3_flops.py:106-108`) → the microbenchmark increment is a P-8 floor and every card's price is re-stated with the measured factor; a `nan` or a raised error → the solve path is not shippable and N-11 becomes a prerequisite of Ph1.
- price: ≈17 s [FITTED + RUN].
- mechanism: P-8, M-8 (each corner priced at its own law), M-3, D-2, V-14 (the front door: `build → forward → journal → verdict`).
- deliverable: `results/shape_smoke_bedm.jsonl`; one row in `COSTS.md` §1.1.
- evenings: 1.

**N-06 — Price the seven-arm arena at the shape's geometry, end to end.**
- phase: Ph1. prerequisites: N-05. independent-of: N-18.
- what to build: the timing harness for the arena of `judge/sec_apparatus.md` §A.4 (depth-1 softmax, ChaCAL-diag, ChaCAL-published, ChaCAL-with-sink, InfSA-style Neumann `K = 16`, cached-mixture, the shape) with synchronised wall clock per cell and randomised cell order; every arm at its own law (M-8).
- what to prove: none.
- what to measure: `7 × 8 × 1.680 + 4.0 ≈ 98 s` (DERIVED, RUN[this] `98.08 s`; the instrument's `≈ 4 GPU-min` was a `2.4×` over-book, `judge/sec_apparatus.md` §A.11 item 5); the InfSA-style cell `≈ 5.1 s` (RUN[this] `150 × (0.010162 − 0.001473 + 0.025409) = 5.11 s`, a DERIVED floor from the `25.409 ms` hop microbenchmark).
- PASS: measured arena within `2.2×` of `98 s`; run order not the strongest correlate of `secs` (the C17 defect, `READ V20_R15_JOURNAL.md:53` via `sec_state.md`). KILL: any cost ratio quoted from an un-synchronised timer; the InfSA cell faster than the solve cell → N-10 promoted.
- price: ≈98 s [DERIVED from FITTED + RUN].
- mechanism: M-8, C17 (run-order confound), P-8, D-3.
- deliverable: `results/shape_arena_price.jsonl`; `COSTS.md` §1.1 table row "arena at the shape's geometry".
- evenings: 1.

**N-07 — The training step under strict mode: the identical-seed floor for the shape.**
- phase: Ph1. prerequisites: N-01, N-05. independent-of: N-06.
- what to build: the floor run of `results/v17k_r4_floor.jsonl`'s pattern (6 identical-seed pairs, seeds 0/1 × three arms, `44.7 s`, `READ V17_R4_RETAKE.md:259-286` via `sec_measured.md` M.2.2) for the shape and the softmax control under `use_deterministic_algorithms(True)` with `warn_only=False`, journalling `deterministic_regime = strict`.
- what to prove: none.
- what to measure: `δ_nrmse` on 6 of 6 pairs; whether the backward executes.
- PASS: `0.0` bitwise on 6/6 for the shape and for softmax. KILL: the shape's backward raises → the "first gated wing under strict mode" sentence is deleted and the lane runs `warn_only=True` (Ruling 1); softmax raises and the shape does not → the sentence stands with the control's row printed beside it (V-23).
- price: ≈45 s [READ `44.7 s` for the record's six pairs].
- mechanism: Ruling 1, M-10 (thread floor `2.345e-3` is the flag-OFF comparison), V-23.
- deliverable: `results/shape_floor_strict.jsonl`; `COSTS.md` §1.6 second line.
- evenings: 1.

## 4. Phase 2 — the cost law in `s`

**N-08 — Retire the `s = 64` harness constant: the `--seq-len` sweep with synchronised, order-randomised timing.**
- phase: Ph2. prerequisites: N-06. independent-of: N-18, N-20.
- what to build: the four-edit repair the record priced and never took (`READ V20_R15_THEORY_TABLE.md:104-109` via `sec_state.md` S.5): `S, D = 64, 24` is a module constant at `scripts/v15_r1.py:137` with 9 argparse flags and no `seq_len` (RUN[this] `grep -c "seq_len|seq-len" = 0`; `grep synchronize = 0` hits) — add `--seq-len`, substitute, two `synchronize()` calls, and the fifth edit no office priced: randomised or blocked cell order. `s ∈ {64, 256, 1024, 4096}` with `n` declared per `s` (the explicit `[n,s,s]` operator at `n = 2048, s = 4096` is `2048·4096²·4 B = 137.4 GB` in float32, RUN[this], against `7.996 GiB`, so `n` shrinks with `s`).
- what to prove: none.
- what to measure: `s/step` for the shape and softmax at each `s`, `N = 8`, fitted exponent in `s` with its CI; the run-order Spearman `ρ` against `secs`.
- PASS: exponent CI for the shape contains `2` (the `s²d` law) and excludes `3`; `ρ(order, secs)` below the gate correlation `+0.5197` (the C17 bar). KILL: CI excludes `2` toward `3` → the serial depth-`s` chain is the bottleneck and N-11 is promoted to the critical path; any price quoted before this run (K-9).
- price: `206–537 GPU-s`, band only, no point (`READ V20_R15_THEORY_TABLE.md:104-109`).
- mechanism: D-3 (a dial that does not vary: `s = 64` on 40 of 40 cells), C9, C17, P-8, M-3.
- deliverable: `scripts/v15_r1.py` flag (developer's edit), `results/shape_s_sweep.jsonl`, `COSTS.md` §1.1 law in `s`.
- evenings: 2.

**N-09 — Where depth `s` becomes visible: the solve/`PV` ratio across `s`.**
- phase: Ph2. prerequisites: N-08. independent-of: N-10.
- what to build: from N-08's journal, the ratio `solve_ms / pv_ms` per `s`; the MAC prediction is `+50 %` causal (`3s²d/2` vs `s²d`, DERIVED `sec_cost.md` §4.x.1), depth `s` vs depth 1.
- what to prove: none.
- what to measure: the ratio at `s ∈ {64, 256, 1024, 4096}`.
- PASS: ratio `≤ 1.5` at `s ≤ 256` (arithmetic-bound). KILL: ratio `> 3` at any `s ≤ 1024` → the `trsm` latency chain dominates and the chunked solve (N-11) is required before any long-context row; the paper's "+50 %" sentence is restricted to the `s` where it was read.
- price: inside N-08.
- mechanism: M-3 (a MAC ratio never stands in for a clock), P-8.
- deliverable: `COSTS.md` §1.1 "depth visibility" row.
- evenings: 1.

**N-10 — The Neumann crossover, if any, across `s`.**
- phase: Ph2. prerequisites: N-08. independent-of: N-09.
- what to build: hop timings `K ∈ {1,2,4,8,16}` per `s` from N-08's harness; hops needed for `δ = 10^{-6}`: `K ≥ log(δ(1−γ))/log γ − 1`, `19.9` at `γ = 0.5` (RUN[this]; `sec_cost.md` §4.x.3 agrees), and at `γ = 0.9` RUN[this] reads `152` where `sec_cost.md` printed `129` — the discrepancy is recorded, the formula is the deliverable.
- what to prove: none; the certificate is Prop. 4 (`judge/sec_shape.md`), attained on the class, declared definitional.
- what to measure: the smallest `K` at which `K` hops are faster than the solve, per `s`.
- PASS (for the paper's sentence): no such `K` at any `s` — "truncation never wins" holds in wall-clock as in MACs. KILL (of the sentence): a crossover at some `s` → the Neumann route is a cost path there, its `δ·‖V‖_∞` printed with `1/(1−γ̂)` beside it, and the F1 mask certificate (N-15) wakes on that `s`.
- price: inside N-08.
- mechanism: L-CERT, V-17 (vector units), V-3 (the equality is definitional), P-8.
- deliverable: `COSTS.md` §1.1 "Neumann crossover" row.
- evenings: 1.

## 5. Phase 3 — chunked, blockwise, CSR, Mapper

**N-11 — Specify and time the chunked block-triangular solve.**
- phase: Ph3. prerequisites: N-09. independent-of: N-13, N-18.
- what to build: the (C2) recursion `M_kk z_k = v_k − γ Σ_{l<k} P_kl z_l` with `s/C` diagonal `[C,C]` triangular solves and one `[C, kC]·[kC, d]` GEMM per chunk (DERIVED `sec_cost.md` §4.x.2: `≈ s²d/2 + sCd/2 + sC²/3` MACs, depth `s/C`, retained diagonal blocks `sC` elements). Inputs: `P` tiles (or `Q, K` to rebuild them), `V`, `γ`, `C`; outputs `z`; invariant: bitwise-or-`1e-6` parity against `solve_triangular` in float32; journal `route = chunked, C`. The intra-chunk inverse is DeltaNet's UT-transform pattern (`yang-2024-deltanet` [V] Eq. 10; shipped kernels `yang-2024-fla` [V], tiled by `beck-2025-tfla` [V]) and is owned; the delta stated narrowly: no `d×d` inter-chunk state exists for softmax `P` (`hu-2025-ssdtheory` [V]), so the inter-chunk term is a genuine GEMM and the cost is quadratic *because* the shape keeps softmax bitwise at `γ = 0`.
- what to prove: none new; `resolvent_fromBlocks` [M] (`judge/sec_shape.md` §4.6 row 6) is the block identity.
- what to measure: parity `max|z_chunked − z_solve|`; wall-clock at `s ∈ {1024, 4096}` vs the serial solve, `C ∈ {32, 64, 128}`.
- PASS: parity `≤ 1e-6` float32 at every `C`; at `s = 4096` wall-clock `≤` the serial solve. KILL: slower at every `C` → the serial `trsm` is the shipped route and the chunked card is closed as NOT NEEDED at this device; parity failure → the kernel is refused (L-CERT: an uncertified route ships no `δ`).
- price: NOT MEASURED — needs a chunked kernel (`sec_cost.md` §4.x.9); first timing ≈ minutes once built [ASSUMED].
- mechanism: M-8, M-3, P-4 (no scaffolding claimed before it exists), P-7 (corner 3 stays in scalar vocabulary; "state tracking" is DeltaProduct's).
- deliverable: `docs/CEQ_CHUNKED_SOLVE.md`; `results/shape_chunked.jsonl`; `COSTS.md` §1.1 row.
- evenings: 3.

**N-12 — The flash-style recompute backward: leaving the `O(S²)` memory class.**
- phase: Ph3/Ph4. prerequisites: N-11, N-18. independent-of: N-13.
- what to build: the backward of N-11 that rebuilds off-diagonal tiles from `Q, K` instead of retaining `P` (two more `QKᵀ` tile products per tile, DERIVED `sec_cost.md` §4.x.7); the adjoint of the solve is one upper-triangular solve plus the outer product `M̄ = −V̄ zᵀ` (DERIVED §4.x.1); retained memory `O(sC)` plus `z`.
- what to prove: none.
- what to measure: peak bytes vs `seq` at `B=4 d=256 L=4 H=4` (the `sizing.py` sweep, `READ ceq/sizing.py:15-22`); gradient parity vs autograd through the retained-`P` route, `1e-5` float32.
- PASS: peak bytes grow as `O(S)` to within the sizing law's `R²` (N-18's constant), gradient parity holds. KILL: bytes still `O(S²)` → the recompute is not reached and the shape stays in the operator class; then the resident `n` per `s` from N-20 is the hard limit of every long-context card.
- price: NOT MEASURED — needs the kernel; the memory sweep ≈ the sizing run's cost [ASSUMED minutes].
- mechanism: P-4, V-22 (a constant fitted on another arm is not carried), P-8.
- deliverable: `docs/CEQ_CHUNKED_SOLVE.md` §backward; `results/shape_memory_recompute.jsonl`.
- evenings: 2.

**N-13 — Zhao et al.'s blockwise `Õ(n^{4/3} d)` as the occupied subquadratic control.**
- phase: Ph3. prerequisites: N-09. independent-of: N-11.
- what to build: the control arm `S_γ(A) = (1−γ)A(I−γA)^{-1}` evaluated as `zhao-2026-structuredsparse` [V] does — exact triangular solves on `m×m` diagonal tiles, cross-block interaction through a reduced `k×k` system via a row down-sampler — as a *fellow* at `s ∈ {1024, 4096}`, `n` per `s`; its `δ` is unprinted in the source and is therefore measured here as `‖O_dense − O_Zhao‖_∞/‖V‖_∞` on `1,024` draws where the dense control fits (`s = 1024`), and NOT MEASURED where it does not (`s = 4096`, `137 GB`).
- what to prove: none; the shape's *exact* block structure is Prop. 6 (F0, `δ = 0`) and holds only on gated corners or `−∞` masks.
- what to measure: visited-tile fraction, wall-clock (synchronised, randomised order), `φ`-NRMSE and harmonic residual on BED-S long-context, the measured `δ`.
- PASS (for the paper's occupancy sentence): the blockwise read within `MDE₈` of the dense read at `≥ 12 %` lower latency, reproducing the source's band in kind. KILL: the shape's F0-segmented solve (N-14) is not better than Zhao's blockwise by `MDE₈` at matched visited tiles → the subquadratic path is theirs, cited, and the shape claims only exactness (`δ = 0` on F0) beside it — never "beats".
- price: NOT MEASURED — needs the blockwise control implemented; dense control at `s = 1024, n ≤ 256` fits (`256·1024²·4 B = 1.07 GB`, DERIVED).
- mechanism: R-SKY, D-1 (racing a published solver of the same operator), V-9 (the do-nothing candidate always entered), P-8, L-CERT.
- deliverable: `results/shape_blockwise_control.jsonl`; prior-art row in `docs/PLAN.md` §7.
- evenings: 2.

**N-14 ★ — The F0 segmentation dividend census, and where it does not transfer.**
- phase: Ph3. prerequisites: none. independent-of: all.
- what to build: a census over the corpus's exact-zero gates: segment lengths `L_m`, the pair-count dividend `D = s(s+1)/Σ_m L_m(L_m+1)` (`judge/sec_shape.md` Prop. 6(iii); `(C5)`'s `s²/Σ L_m²` would read `58.5` on BED-M and is V-17), printed with the corpus named. On BED-M the live-pair fraction is `0.0322` (`READ V16_ARM_SMPRIME.md:518-522` via `sec_cost.md`; positives `257,664 / 266,240`), dividend `1/0.0322 = 31.06×` (RUN[this] `31.0559`), live pairs `0.0322·2080 ≈ 67` per block (RUN[this] `66.98`), `L̄ ≈ 1.1` — a statement that BED-M is almost entirely dead gates, not that text segments are short. Three non-transfer clauses, each printed: (i) on the softmax corner `exp > 0` has no exact zero, so the dividend is `1×` unless a `−∞` mask is declared (F0 by mask, not by gate; Prop. 6(i)); (ii) a cut at `c` makes `P_cc = 1`, a new undeclared absorbing state with `ρ(Q) = 1.0`, and severs every boundary set before `c` from every `i ≥ c` (`q = 0.0`, Prop. 6(ii)) — segmentation and the reach-avoid read coexist only if every segment carries its own sink, goal and constraint sets, the cut a registered dial; (iii) the dividend applies to the exact, Neumann and CSR routes alike (a block-diagonal `M` makes all three block-diagonal, DERIVED §4.x.4) and carries to no other corpus (V-22).
- what to prove: `segmentation_blockdiag` [M] on `StrictlyLower`, `resolvent_fromBlocks` [M], `cut_makes_segment_head_absorbing` [M] (`judge/sec_shape.md` §4.6 rows 6–7); `masked_softmax_blockdiag` [S] for regime S.
- what to measure: the census on BED-M (`n = 128, s = 64`), on BED-S at the softmax corner (expected `0 of N` exact zeros, the F0 theorems silent, `bind_ledger.md` X-14), and on any text corpus the LM cell will use (`NOT MEASURED — needs the corpus's zero-gate census`).
- PASS: BED-M reproduces `31.06×` from the printed fraction; BED-S softmax corner prints `1×` and "F0 silent". KILL: any dividend quoted for a corpus without its census, or carried across corpora (V-22); a committor cell run across an undeclared cut (Prop. 6(ii)).
- price: 0 GPU-s (CPU census).
- mechanism: V-22, V-17, V-25, D-3, L-CERT, P-class (the source line's pair count `133,120` vs `266,240` disagrees by `2×`; the fraction is what is used).
- deliverable: `docs/CEQ_SEGMENTATION_DIVIDEND.md`; a census row per corpus under `results/`.
- evenings: 1.

**N-15 — The CSR two-stage path with the union certificate; the F1 mask bind wakes.**
- phase: Ph3. prerequisites: N-03, N-10. independent-of: N-11, N-13.
- what to build: stage 1, the schedule as data — a causal CSR list of `[B,B]` tiles per query block (`READ THEORY.md:112-119`: `sharma-2026-kernels-22` [V] consumes sink blocks, local-window blocks and a 0D-persistence salience; the do-nothing schedule is always entered, `READ THEORY.md:162-165`); stage 2, Neumann on the sparse `P` (`(γP)^k` keeps `k`-hop block sparsity) with the union certificate in vector units `(ε/(1−γ) + δ_Π)·‖V‖_∞` — the resolvent amplifies dropped row mass by `1/(1−γ)` (`judge/sec_shape.md` Prop. 4(ii): RUN[J] `s = 3, γ = 0.9, ε = 0.1` reads `0.5263` against the naive `0.1`), and the *exact* solve on an F1 mask is refused because the inverse fills in along reachability (DERIVED §4.x.5). The useful-FLOP count is the tree's `4·units·B²·d·BH` (`READ ceq/mz_kernel.py:170-179`) plus `1·units·B²·d·BH` for the resolvent stage, `+25 %` over visited tiles. The two structural guards stay: an empty schedule row must not return `acc/0` (`READ ceq/mz_kernel.py:13-16, :116-121`), a negative block index must be clamped (`:17-21, :80-83`).
- what to prove: `mask_amplification` [S], `f1_cantelli_union` [D] (`judge/sec_shape.md` §4.6 rows 12, 17).
- what to measure: on the shipped mask, `‖O_full − O_mask‖_∞` over `1,024` drawn cells against the printed bound with `‖V‖_∞` printed (the `≈ 7.6e-05` of `sec_cost.md` §4.x.3 carried `‖V‖_∞ ≈ 5 [ASSUMED]`; here it is read); the fill-in guard fires on a planted `(k,l)/(k,m),(m,l)` schedule.
- PASS: no exceedance on any of `1,024` draws; `δ·‖V‖_∞ < sd(label)`; the guard fires. KILL: one exceedance → the mask is refused (L-CERT) and the exact solve, cheaper than one hop at `s = 64`, is the path; `δ·‖V‖_∞ ≥ sd(label)` → the certificate is uninformative and the row is dormant again (D-4).
- price: `1,024` forward passes at `n = 2048, s = 64` ≈ `2.6 s` [DERIVED from the `2.514 ms` RUN]; at `s = 4096` NOT MEASURED.
- mechanism: L-CERT, V-10 (a random `s = 32` draw did not violate the naive bound — a bind on random draws would pass the wrong certificate), V-17, V-24, V-9.
- deliverable: `docs/CEQ_CSR_PATH.md`; `results/shape_mask_certificate.jsonl`; `bind_ledger.md` B-I moves from dormant to live.
- evenings: 2.

**N-16 — The Mapper cover to tile quantiser (the ledger's T-2).**
- phase: Ph3. prerequisites: N-15, N-14. independent-of: N-17.
- what to build: a cover `{U_a}` of key positions from a lens (the committor or an influence score) with parameters fixed by the Reeb-estimator rule on a held-out relation (`carriere-2018-mapper-statistics` [V]; Mapper itself `singh-2007-mapper`, records [V], landing [U]), never on the bed (M-2); nerve edges quantised to contiguous `[B,B]` tiles before they are a schedule — a gather-realised candidate set pays `2.58 ms` of `index_select` against `0.82 ms` of attention at 65,536 positions on this card (`READ ceq/multizoom.py:12-16`); the far field carries the only printed bound in the tree, the mean-pool coarsening bound `‖A − Ã‖_∞ ≤ (D_∞/2)(e^{δ_max} − 1) + max_G[min(2, e^{R_G} − 1)·r_G]` (`READ ceq/multizoom.py:38-49`); a Mapper cover with a comparable `δ` is NOT FOUND (`sweep_topology.md` §5) and enters only through N-15's union bound. Cluster covers as candidate builders are occupied (`cho-2022-sbm-attention`, `roy-2021-routing-transformer`, `kitaev-2020-reformer`, `yuan-2025-nsa` [V]); the delta is the nerve of a lens cover feeding a certified resolvent, stated as such.
- what to prove: none.
- what to measure: visited-tile fraction and `φ`-NRMSE vs the do-nothing 0D-salience schedule and vs N-13's blockwise control at matched visited tiles, `s ∈ {1024, 4096}`, `n` per `s`.
- PASS: better than both controls by `MDE₈` at matched tiles with the union `δ·‖V‖_∞ < sd(label)`. KILL (K-M): not better than do-nothing (or than Zhao 2026) by `MDE₈` → the candidate builder is the merged 0D-salience schedule and Mapper adds nothing measurable; the corpus's segment-length distribution (N-14) is the only dividend left.
- price: NOT MEASURED — needs the quantiser and N-15's stage 2 at long `s`.
- mechanism: V-9, M-2, L-CERT, V-22, P-4.
- deliverable: `docs/CEQ_MAPPER_SCHEDULE.md`; `results/shape_mapper_schedule.jsonl`.
- evenings: 2.

**N-17 — The backward through `kernels#22`: the forward-only limit.**
- phase: Ph3. prerequisites: N-15. independent-of: N-16.
- what to build: the adjoint of the CSR-scheduled attention — the same schedule read transposed for `dK, dV`, the online-softmax statistics saved per query block, and for the resolvent stage the upper-triangular adjoint solve of N-12 — with the two structural guards preserved; parity of gradients against dense autograd on a dense-CSR schedule. Until it exists the CSR path is inference-side (`READ THEORY.md:223-224`, risk 6; the PR's `3.48×` is the PR page's own number, not re-measured, `sweep_topology.md` §2.H).
- what to prove: none.
- what to measure: gradient parity `1e-5` float32 on dense-CSR; wall-clock fwd+bwd vs dense SDPA at `s = 4096`.
- PASS: parity holds; the guards fire on planted empty rows and negative indices. KILL: parity fails → the CSR path stays inference-only and every training row at `s > 64` runs the chunked solve (N-11) or the dense route with `n` per `s`.
- price: NOT MEASURED — needs the kernel.
- mechanism: P-4, V-16 (a `NaN` returned silently is the recorded escape), L-CERT.
- deliverable: `docs/CEQ_CSR_PATH.md` §backward; a `results/` parity row.
- evenings: 3.

## 6. Phase 4 — memory and sizing

**N-18 — Re-solve the operator constant for the shape: the `[S,S]` class measured, not carried.**
- phase: Ph4. prerequisites: N-05. independent-of: N-08.
- what to build: the `sizing.py` sweep (`B=4 d=256 L=4 H=4`, `seq ∈ {128, 256, 512, 1024, 2048}`, forward + backward, peak allocator bytes, `READ ceq/sizing.py:13-22`) run on the shape at the softmax corner with the retained-`P` route; two-parameter least squares separating `C_RESIDUAL` and `C_OPERATOR` (`READ COSTS.md:81-90`: the record's constants `17.874` and `3.823` at `R² 0.996373` were fitted on the *signed* arm; carrying `3.823` to the shape is M-8, and `sec_cost.md` §4.x.7 estimates `C_OPERATOR_shape ≈ C_OPERATOR + 1…2` [ASSUMED]).
- what to prove: none.
- what to measure: `C_OPERATOR_shape`, `C_RESIDUAL_shape`, `R²`; the ratio shape/softmax bytes per `seq`.
- PASS: `R² ≥ 0.99`; the ratio grows with `S` (the class is `O(S²)`, as `1.44× → 8.06×` did for the signed arm). KILL: `measured/predicted` outside `[0.9, 1.1]` for the module's formula → the module is declared wrong for this arm as it was for the complex arm (N-21), and the shape's memory is quoted only from measurement.
- price: ≈9 GPU-min [DERIVED: the record's memory law ran inside the 523.9 s certificate; five `seq` points × two arms].
- mechanism: M-8, V-22, P-8 (the peak-activation formula `4·n·s·d·heads` was `64×` low, struck, `READ workdonenew.md:386` via `sec_measured.md` M.4.2).
- deliverable: `results/k_cert_local.json` (`memory/law_shape`); `COSTS.md` §1.2 row.
- evenings: 2.

**N-19 ★ — The bf16 gap: does `solve_triangular` have a bf16 CUDA path, and what the residual constant under-predicts.**
- phase: Ph4. prerequisites: none. independent-of: all.
- what to build: a one-liner attempt `torch.linalg.solve_triangular` on bf16 CUDA operands under autocast, recorded as executable or raising (the claim "no bf16 path in torch 2.5.1" is `[U]` in `sec_cost.md` §4.x.7 and is settled here); then the bytes-per-element table for the shape under autocast beside the record's: operator `3.341` B/elem (`R² 0.999830`, `READ COSTS.md:91`), residual `2.383` vs the module's `2.2` — wrong by `+8.3 %` in the optimistic direction, `READ COSTS.md:92-99`, bounded at the Q3 chunk shape (`2.715` vs `2.738 GiB`, `max_batch = 45` either way).
- what to prove: none.
- what to measure: executability; the operator term's B/elem for the shape under autocast.
- PASS (the expected reading): the solve raises or upcasts, the shape's `[S,S]` operand stays at `4.0` B/elem under autocast, printed in the sizing table as its own row. KILL (of the `[U]` sentence): a bf16 path exists → the row is re-fitted and the `4.0` sentence deleted.
- price: seconds.
- mechanism: P-3 (an `[U]` claim retired to RUN), V-22, M-8.
- deliverable: `COSTS.md` §1.2 "autocast, shape" row; `ceq/sizing.py` docstring note (the developer's edit; the module is not edited by a doc node, `READ COSTS.md:93`).
- evenings: 1.

**N-20 — The `n` that does not fit: residency of the shape at `n ∈ {8192, 16384, 32768}`.**
- phase: Ph4. prerequisites: N-18. independent-of: N-08.
- what to build: the residency table of `COSTS.md:109-116` for the shape: the workhorse arm reserves `10.578 GiB` at `n = 16384` under a training loop against `7.996 GiB` and pages over PCIe, which is why Ruling 8 dropped Q2 (`READ V17K_RULINGS.md:330-339`) and why the throughput law was fitted on three points, not five (`READ COSTS.md:76-79`: "a throughput law fitted through swap is not a throughput law"); the shape on the softmax corner inherits the softmax rows (resident to `32768`, `4.908 GiB` reserved) plus `z` at `8 MiB` per `2048` (RUN[this] `2048·64·16·4 B = 8.0 MiB`) plus the solve's transients.
- what to prove: none.
- what to measure: allocated and reserved GiB per `n`, under a training loop, with `synchronize()`.
- PASS: resident at `n = 16384`. KILL: reserved `> 7.996 GiB` at `16384` → R2's `t* = 8, n = 16384` reading stays dropped for the shape as for the arm (Ruling 8's consequence "dropping the reproduction does not supply the reading" is carried, not answered); any law point through swap is refused from the fit (P-8).
- price: ≈3 GPU-min [DERIVED: three `n` × two arms × a 150-step cell at the law].
- mechanism: P-8, Ruling 8, V-22 (no Kaggle number stands in for a local one; the T4's `59 %` is not this box's).
- deliverable: `COSTS.md` §1.3 residency row for the shape.
- evenings: 1.

**N-21 — The sizing model's under-prediction for complex arms, and what the corner-3 base inherits.**
- phase: Ph4. prerequisites: N-18. independent-of: N-20.
- what to build: the complex-arm audit as a card: `arm_phase` measured/predicted `1.842`, `C_OPERATOR = 7.50` at `8` B/elem (`4.29×` softmax's, not the `2×` a complex-is-two-floats argument gives), `C_RESIDUAL` NOT IDENTIFIED at `s = 64` (`READ V16_DEVICE_CERT.md:281, 318-334` via `sec_measured.md` M.4.2). The shape on the corner-3 base (`arm_smprime`, complex64 path product, `READ ceq/arm_smprime.py:144-172`, whose own note names a segmented associative scan as the move past `s = 64`) inherits this and, by `judge/sec_apparatus.md` §A.11 item 5, carries no certificate (`β̂ ≠ 1` rows sum `1.31 … 10.29`). The card either re-solves the complex constants on the corner-3 base with a `seq` sweep or declares that base uncertified for sizing.
- what to prove: none.
- what to measure: `measured/predicted` for the corner-3 base across `seq`; its `C_OPERATOR` at 8 B/elem.
- PASS: constants re-solved with `R² ≥ 0.99` and the ratio stated per `seq`. KILL: `R² < 0.99` or `measured/predicted` outside `[0.9, 1.1]` → the corner-3 base is journalled "uncertified for sizing" and no long-context card runs on it.
- price: ≈5 GPU-min [DERIVED: the corner-3 cell is `≈ 9.6×` softmax's, `READ V17_R4_RETAKE.md:299-301` via `sec_measured.md` M.4.1].
- mechanism: M-8, V-22, P-8, V-25 (no certificate off the row-stochastic class).
- deliverable: `COSTS.md` §1.2 complex-arm row; `MODEL_CARD.md` limits line.
- evenings: 1.

## 7. Phase 5 — the envelope, the HF package, the Kaggle gate

**N-22 — The flight envelope as the run's guard, written for the shape lane locally.**
- phase: Ph5. prerequisites: N-04. independent-of: N-05..N-21.
- what to build: the envelope of Ruling 6 (6a–6f: stale = two polls; session time is the device's; tier precedence; per-root lineage; OOM keyed per shape; the deciding-cell list frozen at launch — `READ V17K_RULINGS.md:71-81` via `sec_state.md` S.4) as fields every shape cell carries, plus the three run-guards the record already owns: the 0-step RED gate (`GATE_TOL = 1e-3`, 30 of 30 shapes pass with worst margin `1.957e-3`, `READ COSTS.md:137-139`), the bar re-certification HALT line (`δ/tol < 50 %`; the record's worst `8.58 %` with headroom `5.83×`, `READ COSTS.md:127-133` via `sec_measured.md` M.4.4), and the thread lane read from the journal never from the machine (M-16). An OOM on any shape aborts that shape and journals it; a bar `δ/tol ≥ 50 %` halts the run; a stale poll is two polls, not one.
- what to prove: none.
- what to measure: two planted faults — an OOM-sized shape and a bar drifted past `50 %` — must abort and halt respectively, with journal rows.
- PASS: both plants fire; an honest run passes both. KILL: a plant that does not fire → the envelope is V-15 (a condemning rule with no planted negative) and no launch request is made until it does.
- price: seconds (the plants), 0 GPU-s to specify.
- mechanism: Ruling 6, V-15, V-16, M-16, D-4.
- deliverable: `docs/CEQ_ENVELOPE.md`; the envelope fields in `docs/CEQ_MANIFEST.md`.
- evenings: 1.

**N-23 — The HF package: replace the dead operator with the shape, match the count, keep the card honest.**
- phase: Ph5. prerequisites: N-05, N-04. independent-of: N-08..N-21.
- what to build: `ceq/hf/` holds `configuration_ceq.py`, `modeling_ceq.py`, `train.py`, `smoke.py` (RUN[this] `ls`); no checkpoint ships (`READ MODEL_CARD.md:73-75`, ⟨SLOT `Q3_CHECKPOINT`⟩ NOT MEASURED); the counts are unequal, arm `25,736,232` vs control `25,728,000`, `+8,232 = +0.032 %`, the arm carrying more (`READ MODEL_CARD.md:76-79`); the card's header says the shipped signed operator does not work (sign-flip slope `−1.298` vs bar `−0.3`). The spec: the attention module becomes ChaCAL-diag with declared boundary rows and the N-04 `route`/`diag_convention` fields; `smoke.py` asserts `torch.equal` parity at `γ = 0` against the package's own softmax attention (the record's `1.110223e-16` on `19/64` entries against a fused kernel is named, not rounded); the parameter count is recounted per arm with the `γ` scalar and any `[m, K+2]` head included (Ruling 3: exact counts in every table header, do not re-architect to close `0.032 %`); `MODEL_CARD.md` is rewritten limits-first with the median sentence of `design_falsify.md` §6.2 (with "six" corrected to five machine-checked identities) as its capability paragraph until a cell exists.
- what to prove: none.
- what to measure: `python ceq/hf/smoke.py` on CPU: parity, count, a forward at `s = 64`.
- PASS: parity `True`, counts printed and within `0.032 %`, no checkpoint claimed. KILL: parity fails → the package ships no shape; a card sentence with a bracketed number filled by anything but a FOUND cell.
- price: 0 GPU-s.
- mechanism: P-3 (the card's stale operator retired), Ruling 3, V-3 (parity declared against the lane's own routine), P-1.
- deliverable: `ceq/hf/` (the developer's edits), `MODEL_CARD.md`.
- evenings: 2.

**N-24 — The Kaggle gate: pins, manifest, certificate lines, and the author's yes. Nothing launches from this plan.**
- phase: Ph5. prerequisites: N-01, N-02, N-18, N-22, N-23. independent-of: none (it is the sink of the DAG).
- what to build: the local half of Gate 0 only. Pins as they stand (`READ kaggle/README.md:86-97`): BED-M `ceq.corpus.build(n_train=384, n_test=128, seed=0)` `2f282a5d…7d24d`, BED-K `ceq.beds.bed_k.build_delay(n=500, d=4, seed=7)` `15de94b4…c56e4`, BED-1 `ceq.beds.bed_1.build(T=0.25, jitter=0.05, seed=11)` `f73ca0e6…881d2`, all `kind: generator, status: PINNED`, regenerated in-notebook and asserted (Ruling 7); `enwik8` = the first `100,000,000` bytes of `jamesmcguigan/hutter-prize`'s `enwik9`, `2b49720e…24a8`, split at `90,042,869 / 95,000,818` (`READ kaggle/README.md:31-43`); three sources `UNPINNED_AWAITING_KAGGLE` whose hash cell prints and then **raises** `kdata.MissingPin` (`READ kaggle/README.md:99-105`); the code pin `PINNED_SHA` in the notebook is the snapshot the run executes and may differ from the branch tip (`sec_state.md` S.6). The Kaggle certificate slot is empty and stays empty until the author's run (`READ COSTS.md:158-168`); a threshold carried across the device boundary is V-22 (`READ COSTS.md:190-193`). The gate's own circularity (G0.10: rows 1–2 blocked on numbers only the Kaggle run produces, `READ V17K_RULINGS.md:10-16`) is the author's ruling and is not taken here. Launch order unchanged: `K-CERT first on Kaggle (δ/tol < 50 % or HALT) → Q1 → Q3 → Q4 → cross-device table → HF package on explicit say-so` (`READ V17K_RULINGS.md:91-94`, Ruling 8's queue `:341-342`). The one consumer of the gate in this programme is the LM `γ` cell (the (j) row, `bind_ledger.md` P5.8): `γ̂` MOVED on `≥ 6/8` enwik8 seeds vs ChaCAL at fixed `γ = 0.9`, price NOT MEASURED.
- what to prove: none.
- what to measure: locally, that N-01, N-02 and N-18 have filled their `COSTS.md` lines, N-22's plants fire, N-23's smoke passes, and the manifest's pinned digests re-assert on regeneration (`bed_signature` twice, `READ kaggle/README.md:94-96`).
- PASS: every local line filled; open rulings listed by name (`⟨CLAUSE_1_TAIL⟩`, `⟨L_GRADE_RUBRIC⟩`, `⟨KAGGLE_ATTACH⟩`, `⟨F4_GATE⟩`, `⟨TERMINAL_VS_NOT_PUT⟩`, `sec_state.md` S.5) beside the request; the request is a notification to the author, never a launch. KILL: any Kaggle number pooled with a local one (V-22); any launch, upload or training start without the author's explicit yes (memory rule); a `MissingPin` bypassed.
- price: 0 GPU-s locally; Kaggle NOT LAUNCHED.
- mechanism: V-22, P-1, Ruling 7, G0.10, the *Kaggle launch requires yes* rule.
- deliverable: `docs/PLAN.md` §Kaggle gate (the checklist with its open rulings); no `kaggle/` file changes.
- evenings: 1 (local).

## 8. What dies if what (systems side only)

If N-05 reads above `2.2×` the band, every price in this file is a P-8 floor and the arena of `judge/sec_apparatus.md` §A.11 is re-priced before Bets A–E are scored. If N-08's exponent CI excludes `2` toward `3`, the serial solve is not the shipped route past `s = 256` and N-11 moves onto the critical path ahead of every long-context bed. If N-11 never beats the serial `trsm` on this card, the chunked card closes and long context lives on `n` per `s` (N-20) — `137 GB` at `n = 2048, s = 4096` is the number that says why. If N-13's blockwise control matches the exact F0 solve at fewer tiles, the subquadratic path is Zhao et al.'s and the paper keeps only `δ = 0` on gated corners. If N-15's mask exceeds its bound once, the mask is refused and the exact solve — cheaper than one hop at `s = 64` — remains the path, which is the median outcome the calibration record (7 of 8 optimistic, `READ V16_CALIBRATION.md:96-100`) predicts. If N-01's line reads anything but `0.0`, the shape's training runs `warn_only` like the scan arms and the determinism sentence is deleted. None of these deaths spends more than the `206–537 GPU-s` band of N-08; the Kaggle card spends nothing and decides nothing without the author.

## Limits (once)

Every RUN number is carried from `sections/sec_cost.md` at its geometry (`n ∈ {2048, 4096, 8192}`, `s = 64`, `d = 16`, float32, one box, one process, `±12 %` clock spread) or from the READ files named; the only executions this session were arithmetic one-liners (RUN[this]) and `grep`/`sed` reads. Prices for N-11, N-12, N-13, N-16, N-17 are NOT MEASURED because the kernels do not exist; their evenings are estimates of the author's time, not of GPU time. The `γ = 0.9` hop count differs between this file's formula (`152`) and `sec_cost.md`'s printed `129`; the formula is the deliverable and the discrepancy is recorded. The bf16 sentence is `[U]` until N-19 runs. `C_OPERATOR` for the shape is `[ASSUMED]` until N-18. The chunked solve's parity tolerance `1e-6` float32 and the gradient tolerance `1e-5` are pre-registration candidates frozen here before any kernel exists (M-2). The critical-path count of 18 evenings is a serial depth of this slice alone and does not include the bed, bind, Lean or arena cards of the other planets, which N-05 and N-06 feed but do not wait on.
