# PLAN — MARS (MORIARTY): the standing-attacks roadmap

*2026-09-03, HEAD `207e7b9`. Every card below is a killer: the cheapest run, with its threshold frozen here, that ends a claim sentence the judged sections permit (`judge/sec_shape.md`, `sec_obstructions.md`, `proposition_ledger.md`, `sec_apparatus.md`, `bind_ledger.md`; they outrank the designs and notes). Evidence classes: `RUN[M]` — this planet's numpy float64 one-liner (seed 0, `s = 32`, causal softmax `P` with the diagonal, sink `{0}`, goal `{5,6}`, constraints `{9,10}`, `{15}`, query position 20; the judge's draw of `sec_shape.md`); `RUN[x]` — another planet's run as its file records it; `READ path:line`; `CITED [V]` by canonical `references.bib` key; `DERIVED` with steps. Prices are on the certified RTX 4060 Laptop, tagged `[FITTED]`/`[RUN]`/`[ASSUMED]`/`DERIVED` from `sec_cost.md` §4.x.8 and `sec_apparatus.md` §A.11. Every card names the `MISTAKES.md` mechanism it is designed against (66 `###` headings at HEAD, `RUN grep`). Nothing launches on Kaggle from this plan: `kaggle/README.md:9-11` (READ via `sec_state.md` S.6) makes the author's explicit yes a node, and this plan places no work behind it. The author works alone, in evenings, in any order the DAG in §3 allows. No card produces code from this file; each names the file the author produces.*

## 0. Reading rules for the cards

1. **Fields, in order:** id; title; phase; prerequisites; independent-of (D-1 dependency law, `READ CONTRACT.md:27-33`); what to build; what to prove; what to measure; PASS; KILL and what dies; price; mechanism; deliverable; evenings.
2. **A threshold written here is frozen** (M-2, `READ MISTAKES.md:451`). A card whose threshold is a formula on a not-yet-measured sd says so and is filled by MK-18 before any verdict; the formula does not move afterwards.
3. **Phases:** P0 free killers (0 GPU-s, CPU identities, Lean, specs); P1 bed admission (0 GPU-s); P2 binds on the real draw (seconds); P3 the first pair and the arena (`≈ 34 s` per pair, `≈ 98 s` for seven arms, `sec_apparatus.md` §A.11 item 5); P4 verdicts (MDE, LR, TOST); P5 deferred instruments (NOT MEASURED, each names the missing instrument).
4. **The five cheapest decisive items** carry `★` in their title: MK-01, MK-04, MK-07, MK-12, MK-13. Each settles something alone in one sitting (`THESIS_CORRECTIONS_2.md` §0's "first evening").
5. **What would change this planet's mind** is stated per card as the PASS number: a killer that reads its PASS side is the evidence that the attacked sentence stands; the calibration rule D-CALIB-1 (`READ V16_CALIBRATION.md:142-203` via `sec_measured.md` M.5: 7 of 8 signed rows optimistic, `p = 0.0352`) makes the KILL side the point estimate until a PASS is read.

## 1. The record's transferring attacks, one card each

**MK-01 ★ — Build the ten `[M]` Lean identity targets and their refusals; a `sorry` demotes the row.**
phase P0 · prerequisites none · independent-of every other card
· build: `lean/CEQ/Shape.lean` carrying `sec_shape.md` §4.6 rows 1–10 (`gamma_zero_is_softmax` with witness `gamma_half_is_not_softmax`; `bos_row_is_absorbing` at `β = 1` with the `β = 0` refusal, row 0 reading `e^{qk_00} = 2.0138`, `RUN[M]` per `sec_apparatus.md` §A.8.2; `later_boundary_unreachable`; `softmax_corner_not_nilpotent` with `pow_card_eq_zero` as the regime-N contrast; `lower_triangular_isUnit`, `diag_one_sub_smul_pos` on `[1-γ, 1)`; `resolvent_fromBlocks`, `segmentation_blockdiag` on `StrictlyLower` with `tiny_gate_does_not_cut` at a `−30` logit; `cut_makes_segment_head_absorbing`; `displacement_identity` with `const_value_zero_displacement`; `lowerTriangular_ne_symmSupport`; `committor_is_resolvent_read` (a) and (b-causal)). Journal fields: declaration name, `#print axioms` output, `lake build` exit code.
· prove: the ten statements as written in `sec_apparatus.md` §A.8.2, grade `[M]`; any `[M]` resting on a `[U]` Mathlib name (`Matrix.det_of_lowerTriangular`, `inv_fromBlocks_zero₂₁_of_isUnit_iff`) is `[S]` until green.
· measure: `lake build` in `lean/`; `grep -c '\bsorry\b'` on the new file; axiom set per declaration.
· PASS: exit 0, `0` `sorry`, axioms `[propext, Classical.choice, Quot.sound]` only on all ten.
· KILL: any target not building by this milestone ⇒ that row is `[S]`, never a claim; the paper's "machine-checked" sentence for component (k) is limited to the rows that build. What dies: the sentence, not the identity (its float instance stands as RUN).
· price: `0 GPU-s` · mechanism: P-11 (`READ MISTAKES.md:1597`, a contract citing its own `[M]` tag as settled — the R15 `#18/#19/#22 [M]` had no declaration, `sec_proved.md` §2.0); L-LEAN · deliverable: `lean/CEQ/Shape.lean`, `docs/plan/MK-01_lean_ledger.md` · evenings: 1–2.

**MK-02 — Extend the identity manifest with the shape fields and its one-unit drift plant.**
phase P0 · prerequisites none · independent-of all
· build: the field list of `sec_apparatus.md` §A.10 (`gamma` init/final/`Λ`/verdict, `diag_convention`, `committor_route`, `route`+`K`+`C`, `boundary_sets`/`goal_set`/`sink_set` with sha256, `K`/`m`/`t_star`, `producer_cmd`, `delta_vec`/`delta_bare`/`one_over_1mg`/`V_inf`, `census`, floors, skyline settings, `void_contrasts`, `supersedes`); a missing declared field is a refusal, not `absent` (`READ scale/identity_manifest.py:141-151` reports rather than raises — the shape lane has no stored manifests to protect).
· prove: none (an instrument).
· measure: flip each of `beta / qk / g / gamma / boundary_sets` by one unit; count manifest-hash moves (the pattern of `READ V20_R15_WING_MANIFEST.md:148-152`).
· PASS: `5/5` flips move the hash; a deliberately omitted declared field raises.
· KILL: any flip leaving the hash fixed ⇒ the field is not identity and no cell may be attributed to it (L-2: `0 of 24` attributable, `READ V20_R15_LEAP_LEDGER.md:23`).
· price: `0 GPU-s` · mechanism: V-16 (`:828`), P-1 (`:289`), V-3 · deliverable: `docs/plan/MK-02_manifest_spec.md`, `results/shape_manifest_plant.jsonl` · evenings: 1.

**MK-03 — Print the vacuity inequality beside every obstruction citation before any obstruction sentence is typeset.**
phase P0 · prerequisites none · independent-of all
· build: a table with, per theorem, the model class, the hypothesis as a predicate on `(s, d, h, p)`, and its value at the record's geometry `s = 64, d = 16, h = 1, p = 32`: `h·m·p = 512 ≥ 64` (`sanford-2024-inductionheads` Thm 1), `n log₂ n = 384 < 544` (`peng-2024-transformer-limitations` Thm 1), `64^{1/16} = 1.2968` against `H·d·p = 512` (`chen-2024-multilayer` Thm 1.1), the one-vs-two-cycle conjecture named beside `sanford-2024-logdepth` Cor. 4.3 (all `[V]`, `sweep_expressivity.md` §3).
· prove: none.
· measure: the fraction of drawn BED-S cells whose geometry satisfies each hypothesis.
· PASS: every obstruction sentence in the paper carries its line; `0 %` admitted ⇒ the sentence is stated as asymptotic lineage only.
· KILL: an obstruction stated without its line (K-10 of `bind_ledger.md`) ⇒ the sentence is struck at assembly.
· price: `0 GPU-s` · mechanism: V-25 (`:1954`), P-10 (`:1382`), P-3 · deliverable: `docs/plan/MK-03_vacuity_table.md` · evenings: 1.

**MK-04 ★ — Write the BED-S generator specification and run its domain census on 512 draws (BOS declared; constraints and goal before the query).**
phase P1 · prerequisites none (MK-01 may run beside; L-LEAN forbids *training*, not construction) · independent-of MK-01..03, MK-11, MK-14, MK-15, MK-17
· build: the DAG-in-token-order environment of `sec_apparatus.md` §A.1 — edges to earlier positions, self-loops only on declared absorbing positions, `𝒜_sink = {0}` with value `0` on every indicator channel, goal `𝒜_0` and constraints `𝒜_1..𝒜_K` all before the query, the query in `T`, adjacency multi-hot node features, membership flags, `m` query-side row clamps `P_env[v_a,:] ← e_{u_a}`, oracle = the Dirichlet solve of `READ ceq/beds/bed_1.py:188-198`'s form on `P_env` at `γ = 1` on the transient block. Journal fields: the fifteen census lines of §A.7 per batch.
· prove: F1 and F2 as the census's two structural lines (`RUN[I]` R6/R10; `RUN[J]` `det(I − Q) = 0.0` undeclared; `q = 0.0` for a set after the query).
· measure: on 512 draws at `t* = 8, m = 8, K = 2`: `0 ∈ 𝒜_sink` fraction; sets-before-query fraction; `ρ(Q̂) = max_T P_ii < 1` fraction; per-coordinate label sd over the admitted query region; sink share `q^{(sink)}` separately from `q^{(0)}`; class frequency of `a*` over `m`; rule-disagreement fraction; discard count.
· PASS: lines 1–4 at `100 %`; every sd `> 0.05`; every class frequency in `(0.05, 0.95)`; disagreement `> 0`.
· KILL: any sd `= 0`, any class outside `(0.05, 0.95)`, disagreement `0`, or `< 100 %` on lines 1–4 ⇒ the bed is not admitted (D-4) and no capability card below runs; the author reroutes to `bed_1.build(jitter)` with `K = 2` (`design_falsify.md` §5 rank 1) and reprices. What dies: every BED-S sentence, at `0 GPU-s`.
· price: `0 GPU-s` (oracle `< 1 s` per draw in float64 at `|T| = 1200`, DERIVED `sec_beds.md` §6.D.5) · mechanism: V-25, V-8 (`:136`), V-12 (`:189`), D-3 (`:727`), D-4 (`:740`) — the record's dominant death (a bed dying at construction) becomes a free decision · deliverable: `docs/plan/MK-04_bed_s_spec.md`, `results/bed_s_census_512.jsonl` · evenings: 2.

**MK-05 — Run the corpus-alone feature-leak probe with its planted leak (the leak detector must fire both ways).**
phase P1 · prerequisites MK-04 · independent-of MK-06..08
· build: a linear probe from the token features `x` to (b) the membership indicators and (c) `q` at order 0, on the admitted draws; the planted positive: a move token carrying its own `q`. Clause (a) (rows of `P_env`) is dropped for BED-S because the graph is the input (`sec_apparatus.md` §A.1); clause (c) is the kill. Journal fields: `R²` per clause, honest and planted.
· prove: none.
· measure: `R²` on (c) honest; `R²` on the planted leak.
· PASS: honest `R² < 0.5` and planted `R² ≥ 0.99`.
· KILL: honest `R² ≥ 0.5` ⇒ the label is reachable without a solve and the one-read claim is void on BED-S; planted `R² < 0.99` ⇒ the detector is blind (V-24) and no leak verdict is admissible.
· price: `0 GPU-s` · mechanism: M-21 (`:2101`, the record's gate written into `x[:,:,CH_DRIVE]` read `R² = 1.000000`, `READ MISTAKES.md:2133-2143`), V-24 (`:1658`), V-7 · deliverable: `results/bed_s_leak_probe.jsonl` · evenings: 1.

**MK-06 — Register the zero-hop, one-hop and hand-rule controls at construction, before any arm trains (the static-task kill).**
phase P1 · prerequisites MK-04 · independent-of MK-05, MK-07, MK-08
· build: the argmin controls of `sec_apparatus.md` §A.4 (0-hop per-position MLP, 1-hop softmax, majority, random `1/m`, predict-the-mean) plus the closed-form rule of MK-24; the exact zero-information floor `1 − max_a π̂(a*)` from the realised class distribution (`0.875` at uniform `m = 8`, `RUN[J]`); `I(X_{≤0}; a*)` by plug-in over the zero-hop view including move tokens and flags.
· prove: none.
· measure: untrained arm at `1/m ± CP`; committor NRMSE `≥ 1 − GATE_TOL` at step 0 (`GATE_TOL = 1e-3`, `READ COSTS.md:137`).
· PASS: untrained reads chance; `I(X_{≤0}; a*)` printed with the tight Fano only where `> 0`.
· KILL (frozen, scored at MK-19): 0-hop MLP within the Clopper–Pearson half-width at `N = 8` of the shape, or McNemar `p > 0.05` against 1-hop softmax ⇒ BED-S is a third static task (`sec_refuted.md` C8) and is struck before any number is quoted.
· price: `≤ 1.524 s` per control cell `[FITTED]` when trained; `0 GPU-s` at registration · mechanism: V-10 (`:168`), D-5 (`:755`), C8 · deliverable: `docs/plan/MK-06_controls.md` · evenings: 1.

**MK-09 — Run the identity binds through the front door on BED-S's real draw, each plant as a FOUND cell.**
phase P2 · prerequisites MK-04 admitted, MK-02 · independent-of MK-11..17
· build: B-J (parity: `torch.equal(O(0), PV)` against the lane's own `softmaxAttn`; plants `γ = 0.5 ⇒ 2.3002850040264393`, `β = 0 ⇒ > 0.5`, ChaCAL-published's `A_s ⇒ O(1)` at any `γ > 0`), B-E1 (`𝒜 = ∅ ⇒ torch.equal` with ChaCAL-diag; `𝒜 ≠ ∅ ⇒ ‖Π_shape − Π_ChaCAL-diag‖_∞ = O(1)` on downstream rows, a *weight* gap — rows changed `= [3]`, `RUN[M]` of the design round), B-E2 (conservation to `1e-15`; BOS dropped ⇒ the solve raises), B-G1 (`V ≡ 𝟙 ⇒ max|Δz| ≤ 1e-15`; Gaussian `V ⇒ 0.1096/0.363/1.127`; non-causal `P ⇒ Δz[:i] ≠ 0` at `0.0761/0.0868`; Sherman–Morrison vs re-solve `≤ 1e-12`), B-G2 (metric plants: permuted oracle must not score `0`; `+0.1σ` preferred; `−Δz` distinguished), B-H1 (`q` vs `bed_1.committor` on real sets `0.0`; wrong-set plant moves `q` by O(1); the script raises on a missing key — the `basin_A` default that read `0.858` is the filed V-16, `sec_refuted.md` C12), B-P5 (dense `M` with `upper=False` must disagree). Every plant enters as `bed_s.build → arm.forward → journal row → verdict()` with its own `kind` and `manifest_hash`.
· prove: none beyond MK-01.
· measure: honest residual and each plant's residual with counts, the `0.9749 / 0.9165 / 1.000 / 0.4845` pattern (`READ workdonenewseal.md:114-122`).
· PASS: every honest half passes and every plant fails at O(1).
· KILL: any plant passing ⇒ that bind has an empty rejection region and is struck from the paper; any honest half failing ⇒ the identity is wrong on the real draw and the proposition it carries is demoted to OPEN.
· price: seconds, `0 GPU-min` to two decimals · mechanism: V-24, V-3 (`:72`), V-14 (`:238`, a plant that never enters the front door validates the matcher and not the reach), V-16 · deliverable: `results/bed_s_binds.jsonl`, `docs/plan/MK-09_bind_ledger.md` · evenings: 1.

**MK-10 — Measure the certificate delta on the shipped object, in vector units, with `‖V‖_∞` printed (dormant until a mask ships).**
phase P5 · prerequisites a mask on any route (MK-25 or later) · independent-of everything until then
· build: on the shipped mask, `‖O_full − O_mask‖_∞` over `1,024` drawn cells against `(ε/(1−γ̂) + δ_Π)·‖V‖_∞` with `ε` the dropped row mass, `δ_Π = γ̂^{K+1}` under the `(1−γ)` factor, `1/(1−γ̂)` printed beside every `δ` (`sec_shape.md` Prop. 4(ii): the resolvent amplifies dropped mass, `RUN[J]` `0.5263` vs naive `0.1` at `s = 3`). The row-sum equality `err = γ^{K+1}/(1−γ)` on the class is declared *definitional* (V-3) and never presented as evidence about `P`; the convergent plant (rows `1.5`, `γ = 0.6`, `K = 2`: `7.29` vs `0.54`, `RUN[M]` of the design round) is the rejection region; the coordinator's `119.37` vs `1.143` at `γ = 0.7` is a divergent partial sum (`ρ(γP) = 1.05`) and is labelled so.
· prove: `mask_amplification` `[S]` (MK-01's row 12).
· measure: the worst excess over `1,024` draws; `δ·‖V‖_∞` against `sd(label)`.
· PASS: no exceedance on `1,024` draws and `δ·‖V‖_∞ < sd(label)`.
· KILL: one exceedance ⇒ the mask is refused (L-CERT) and the exact solve — cheaper than one Neumann hop here, `2.514` vs `3.001` ms at `n = 2048` `[RUN NEPTUNE]` — is the only route; `δ·‖V‖_∞ ≥ sd(label)` ⇒ the certificate is uninformative and no "certified" sentence stands.
· price: `≈ 2.6 s` DERIVED (`1,024` forward passes at `n = 2048`) · mechanism: V-3, V-10, V-17 (`:855`, a threshold out of its units — the `≈ 7.6e-05` of `sec_cost.md` §4.x.3 carried `‖V‖_∞ ≈ 5 [ASSUMED]`), P-8 (`:387`) · deliverable: `results/certificate_shipped_mask.jsonl` · evenings: 1 (when a mask exists).

**MK-11 — Specify the `γ`-pinning likelihood-ratio instrument with the boundary null, held-out set and seed rule declared.**
phase P0 · prerequisites none · independent-of all; consumed by MK-21
· build: `Λ = 2[LL(γ̂) − LL(γ ≡ 0)]` on a declared held-out set (Ruling 10′ restated for `γ`, `READ V17K_RULINGS.md:389-436`); `γ` trained on `[0, 1)` so the null sits on a boundary and its `95 %` point is `½χ²₀ + ½χ²₁ = 2.7055`, not `3.8415` (`RUN[J]`; the citation for the boundary null is owed and not in `references.bib`); MOVED at `Λ > ln n_eval` (`ln 4096 = 8.318`); the interval verdict between; the `|γ̂| < 0.05` rule deleted (two verdicts for one cell, M-20); the ablation `(I − γ̂P̂)^{-1} → I` at trained weights; the mirror line `1/(1−γ̂)` beside every `δ`.
· prove: none.
· measure (at MK-21): `Λ` per seed; ablation NRMSE move in seed-sd units; `γ̂` support.
· PASS: MOVED on `≥ 6/8` seeds and ablation move `≥ 1` seed sd.
· KILL: PINNED (`Λ ≤ 2.7055`) on `≥ 6/8` ⇒ the arm is softmax wearing a name on that bed and every `γ`-dependent sentence (the horizon dial, Bet C) is withdrawn; ablation `< 1` seed sd ⇒ the same; `γ̂ > 0.99` on `≥ 6/8` ⇒ the `z`/`Δz` certificate is vacuous (`1/(1−â_max)` was undefined at all eight R1 seeds, `READ V15_R1.md:56`). The committor head is untouched (it reads at `γ = 1` on `Q̂`).
· price: `0 GPU-s` for the spec · mechanism: V-9 (`:154`, a repair that changes nothing — the hop-2 sweep where no `γ` beat `0`, `READ workdonenew.md:276`), M-20 (`:1723`), V-17 · deliverable: `docs/plan/MK-11_gamma_lr_spec.md` · evenings: 1.

**MK-12 ★ — Run the ChaCAL-diag smoke test at `γ = 0.9` on BED-M as the solve-path check.**
phase P2 · prerequisites MK-01 rows 1 and 5 green (L-LEAN) · independent-of MK-04..09
· build: one cell of the shape with `𝒜 = ∅`, diagonal kept, `γ` fixed at ChaCAL's own `0.9` (`fagnou-2024-chacal` [V] App. C), on BED-M `e3_t2`, `n = 2048, s = 64`, 150 steps, under `use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before process start; the B-J plant battery run on the same cell.
· prove: none.
· measure: cell wall-clock with `torch.cuda.synchronize()` bracketing; bitwise repeat over 2 runs; the parity plant readings.
· PASS: the cell runs forward and backward under strict mode, repeats bitwise, and its price reads within the record's law-vs-measured band (`−1.8 %` to `+14.4 %`, `READ V17_R4_RETAKE_PRICE.md:228-229`) of `1.680 s` `[FITTED + RUN]`.
· KILL: strict mode raises ⇒ the determinism sentence is withdrawn (the shape would no longer be the first gated wing whose training step runs strict); price above `2.2×` the law ⇒ every `[FITTED + RUN]` price in this plan is re-tagged as the P-8 floor it is declared to be and re-measured end to end before any GPU-minute figure is quoted. No capability number is read from BED-M (contained by Prop. 2, D-2).
· price: `≈ 1.7 s` `[FITTED + RUN]` · mechanism: M-3 (`:464`), P-8, V-16 (three-outcome determinism: executable-bitwise / drifting / raises), D-2 · deliverable: `results/chacal_smoke_bedm.jsonl` · evenings: 1.

**MK-13 ★ — Take the `≈ 6 GPU-s` capped run at seeds 2, 3, 7 on BED-M that the record priced four times and never ran.**
phase P2 · prerequisites none · independent-of all
· build: `arm_pl` at the three NO READING seeds (`1.113403 / 1.139404 / 1.152430`, `â_max 20.31 / 49.66 / 285.07`, `READ V15_R1.md:177-188` via `sec_measured.md` M.2.1) with the magnitude capped, the settled Q3/W3 question (`READ V20_R15_THEORY_TABLE.md:176`; priced at it.7, it.8, it.9, it.35, `READ V20_R15_IT35_JUPITER.md:157-159`).
· prove: none.
· measure: `eval_nrmse` per seed against `floor₁ = 0.7071067811865476`, which is a one-hop capability threshold and not a floor (C15); `â_max` after the cap.
· PASS (for the record, not the shape): the three capped seeds read below `0.7071` ⇒ the bimodal split was a magnitude-divergence artefact and the paper's negatives section says so with the number.
· KILL: the three stay above `1.0` ⇒ the split is not the cap's and the paper's account of R1 keeps "3 of 8 NO READING, mechanism unresolved". Either way the shape's own sentences are unaffected; what dies is a stale question, cheaply.
· price: `≈ 6 GPU-s` `[FITTED]` (three cells at `1.884 s` capped) · mechanism: D-6 (`:1050`, the repair that was written and never started), P-3 (`:316`), M-6 (`:518`) · deliverable: `results/bedm_capped_237.jsonl`, one paragraph in `docs/plan/MK-13_capped.md` · evenings: 1.

**MK-14 — Specify the synchronised, order-randomised timer, and price the `s`-sweep as a band until it runs (the run-order confound).**
phase P0 spec / P5 run · prerequisites none · independent-of all
· build: the harness edits the record priced and never made — a `--seq-len` flag (`S, D = 64, 24` is a module constant at `scripts/v15_r1.py:137`, `RUN[MERCURY] sed`), two `torch.cuda.synchronize()` calls around the timed region (`grep` finds none, `RUN[MERCURY]`), randomised or blocked execution order (the fifth edit, never priced by any office). Journal fields: `secs_sync`, `run_index`, `order_seed`.
· prove: none.
· measure (at the run): Spearman `ρ(secs, run_index)` on the new timer; the fitted exponent of `s/step` in `s` with its CI at `s ∈ {64, 256, 1024, 4096}`, `N = 8`.
· PASS: `|ρ(secs, run_index)| < 0.2` and the exponent CI covers `2`.
· KILL: `ρ` comparable to the record's `+0.7029` (`p = 0.0024`, `READ V20_R15_JOURNAL.md:53` C17) ⇒ every GPU-second in the lane is void as C17 voided the record's; exponent CI excluding `2` toward `3` ⇒ the cost law is restated at the measured exponent; any price quoted before the run ⇒ struck (K-9).
· price: spec `0 GPU-s`; run in the record's band `206–537 GPU-s` = `3.4–9.0 GPU-min`, **band only, no point** (`READ V20_R15_THEORY_TABLE.md:104-109`) · mechanism: D-3 (`s = 64` on `40 of 40`), P-8, M-8 (`:544`), M-10 · deliverable: `docs/plan/MK-14_timer_spec.md`; later `results/s_sweep.jsonl` · evenings: 1 + 1.

**MK-15 — Recount parameters per arm once the `[m, K+2]` head or vector readout is attached; fix width for the wide skyline (param-count wins).**
phase P0 · prerequisites none · independent-of all; consumed by MK-19
· build: a per-arm parameter count printed in every table header (Ruling 3, `READ V17K_RULINGS.md:56-59`); the `4,769` of `READ CEQ_V20_R15_CONTRACT.md:119` is recounted, never assumed; the wide constant-depth skyline's width fixed at `n_nodes` (`yehudai-2025-depthwidth` [V]) and the sentence that at `d_model = 16 < s = 64` the matched-parameter instance does not exist; the CoT decoder's step count fixed at `t*` (`merrill-2024-cot` [V]).
· prove: none.
· measure: the count per arm; the residual mismatch.
· PASS: every matched arm within `0.032 %` of the reference count (the record's residual, `READ MODEL_CARD.md:76-79`).
· KILL: a "matched" contrast with a count mismatch above that ⇒ the contrast is unmatched and is moved to the skyline column, never credited; any arm re-architected to close the gap is refused (Ruling 3).
· price: `0 GPU-s` · mechanism: M-8, D-1 (`:677`), R-SKY (`READ CEQ_V16_CONTRACT.md:209`) · deliverable: `docs/plan/MK-15_param_table.md` · evenings: 1.

**MK-16 — Register the VOID list and the identification reading `‖P̂ − P_env‖_∞` before any cell (the oracle-leak attack).**
phase P1 · prerequisites MK-04 · independent-of MK-05..09
· build: the pre-registration that `shape − softmax` and `shape − skyline` on the committor head are reproduction-versus-non-reproduction contrasts and VOID as capability numbers, because the arm's class contains the oracle's operator (`sec_apparatus.md` §A.1); the creditable contrasts named (`shape − ChaCAL-diag`, `shape − ChaCAL-published`, `shape − ChaCAL-with-sink`, `shape − InfSA-Neumann-16`, `shape − 0-hop`, `shape − 1-hop`); `‖P̂ − P_env‖_∞` journalled per seed as a *learnability* reading.
· prove: `lowerTriangular_ne_symmSupport` (MK-01 row 9) — the D-2 separation that fails on an undirected substrate and holds on the DAG.
· measure (at MK-19): `‖P̂ − P_env‖_∞` per seed.
· PASS: the identification reading is printed beside every capability number and the VOID list is in the manifest field `void_contrasts` before the first cell.
· KILL: `< 1e-3` on `≥ 6/8` seeds ⇒ the arm has copied the environment; every contrast against a non-reproducing arm is a copy-vs-no-copy statement and is filed as such (K-D2, now with a non-empty rejection region on the DAG); a capability number quoted against softmax on this bed ⇒ struck at assembly.
· price: `0 GPU-s` · mechanism: D-2 (`:710`, `settled − softmax` VOID on `e3_t*`), M-7 (`:529`), V-10 · deliverable: `docs/plan/MK-16_void_list.md` · evenings: 1.

**MK-17 — Freeze every BED-1 / BED-S guard partition before any itinerary is read (guards chosen after itineraries).**
phase P0 · prerequisites none · independent-of all
· build: the `q = ½` isocommittor guards of `READ ceq/beds/bed_1.py:5-8, :382` and any BED-S basin partition fixed from the oracle's committor *before* `guard_itinerary` runs, with the partition's sha256 in the manifest; the Pesin-deficit reading kept as the *diagnostic* it is (the record's contract guards had deficit `0.009654 = 11.9 %` of `h`, not near zero; the state partition `0.000233`, `READ V15_BED1.md:248-272` via `sec_measured.md` M.7.2) and never used to select guards by argmin.
· prove: none.
· measure: the deficit of the frozen partition; the guard-crossing balance (the record's `21` net on `65,481` events).
· PASS: the partition hash precedes the first itinerary record in the journal's order.
· KILL: a partition whose hash postdates an itinerary it scores ⇒ every itinerary statistic on that partition is struck (a threshold refitted to the data it judges).
· price: `0 GPU-s` · mechanism: M-2 (`:451`), M-18 (`:1483`), M-19 (`:1564`) · deliverable: `docs/plan/MK-17_guard_freeze.md` · evenings: 1.

**MK-18 — Run the first `N = 8` pair on BED-S; the realised paired sd fills the MDE row and every kill formula.**
phase P3 · prerequisites MK-04 admitted, MK-09 green, MK-02, MK-15, MK-16 · independent-of MK-10, MK-13, MK-14
· build: shape on the softmax corner with the committor head (exact solve at `γ = 1` on `Q̂`), the argmin columns, the `z`/`Δz` channel at learnable `γ` (init `0`), against depth-1 softmax with the same head, `n = 2048, s = 64, d_model = 16`, 150 steps, 8 seeds deduplicated (`REQUIRED_SEEDS = 8`, `READ scale/it11_verdict.py:59`), one thread lane, the flag regime journalled.
· prove: none.
· measure: paired sd of every scored statistic (argmin accuracy, `φ`-NRMSE, cosine, residual); the MDE at `n = 8` by the record's noncentral-`t` bisection (`0.039827` at sd `0.034451` is the *record's* row-2 instance, `RUN[VENUS]`, `RUN[MARS]` to `0.0398266`, not BED-S's).
· PASS: realised sd `≤ 0.034451` (then every `N = 8` kill in this plan is falsifiable as written).
· KILL: realised sd `≥ 2.18×` the pilot (M-3's precedent, `0.050146 → 0.109199`, MDE `0.126238`) ⇒ nothing below is falsifiable at `N = 8`; the plan reprices `N` from the realised sd before any prediction is scored and the paper files the bed with no capability number (K-P).
· price: `≈ 34 s = 0.57 GPU-min` `[FITTED + RUN]` (`17.4 + 16.2` s, `sec_cost.md` §4.x.8) · mechanism: M-3, M-9 (`:580`, finest `p = 0.0625` at `N = 5`), V-5, L-FLOOR · deliverable: `results/bed_s_pair_n8.jsonl`, the MDE row in `docs/plan/MK-18_mde.md` · evenings: 1.

**MK-19 — Run the seven-arm arena including ChaCAL-diag, ChaCAL-published, ChaCAL-with-sink-token, InfSA-Neumann-16 and the argmin controls (the ChaCAL-with-sink kill).**
phase P3 · prerequisites MK-18 (sd known), MK-06, MK-11 · independent-of MK-22 until scored
· build: the control arms of `sec_apparatus.md` §A.6 on byte-identical draws; ChaCAL-with-sink-token uses the shape's trained `γ̂` (the pair is sequential; the order is stated).
· prove: none.
· measure: per arm, argmin accuracy with Clopper–Pearson (`clopper-1934-binomial` [V]), McNemar paired (`mcnemar-1947-correlated` [V]), `φ`-NRMSE, harmonic residual with `σ_min(I − γP_env)` and `‖I − γP_env‖_∞` printed, `Δz` cosine; the identification reading of MK-16.
· PASS (Bet B, `bind_ledger.md` §6): `CP_upper(err_shape) < floor_exact(k = 1)` and ChaCAL-with-sink's committor `φ`-NRMSE exceeds the shape's by `> MDE₈` on `≥ 6/8` seeds.
· KILL (K-E1): ChaCAL-with-sink within `MDE₈` of the realised sd on `≥ 6/8` seeds *and* residual within `2×` ⇒ component (e) is a parameterisation of ChaCAL's sink (`gu-2024-sinkemerges` [V]: sinks are key biases storing non-informative mass; `zhu-2003-harmonic` [V]: clamped nodes are the same rows, 2003); escalate to `N = 16`, never "within TOST" at `N = 8`. What dies: the boundary-row mechanism as a capability; the paper becomes a property paper about a published operator (`design_falsify.md` §6.2) and is written to remain honest there. K-H2 (MK-06) and K-D2 (MK-16) are scored on the same cells.
· price: `7 × 8 × 1.680 + 4.0 ≈ 98 s ≈ 1.63 GPU-min` DERIVED; InfSA-16 cell `≈ 5.1 s` DERIVED floor · mechanism: V-24 (ChaCAL + same `γ` is the plant), M-13 (`:1210`), D-7 (`:2037`), R-SKY · deliverable: `results/bed_s_arena.jsonl`, `docs/plan/MK-19_arena.md` · evenings: 1.

**MK-20 — File the one-step rule as a safety filter, not a policy, with the multichain hazard tested on the oracle (Bellman failure).**
phase P1 · prerequisites MK-04 · independent-of MK-05..09
· build: on 512 admitted draws, the oracle-only comparison of the one-step rule `a* = argmax_a q^{(0)}(do a)` against the two-step optimum (apply `a`, re-solve, take the best second clamp) on the same `P_env`; the vocabulary fixed as `hsu-2023-safetyfilter` [V] / `borquez-2023-lrf` [V] (a one-step, most-restrictive filter whose value is the goal committor), the aggregations as `vanmoffaert-2013-chebyshev` [V] and `yang-2026-lexisafe` [V] columns, and `misra-2023-safety-constrained-mdp` [V] (Bellman's principle can fail for safety-constrained multichain MDPs) carried in Limits beside the rule.
· prove: DERIVED — on a DAG with declared absorbing sets every transient state is absorbed with probability one (`ρ(Q) = max_T P_ii < 1`, Prop. 10), so the *chain* is absorbing; the hazard is in the policy-level object, which the paper does not claim.
· measure: the fraction of draws on which the one-step and two-step optima disagree.
· PASS: the paper never writes "optimal move" or "policy"; the disagreement fraction is printed.
· KILL: disagreement `> 0.5` on 512 draws ⇒ the phrase "safest move" is replaced by "one-step safety filter" in every sentence, and the sequential problem is filed as future work with the Misra citation.
· price: `0 GPU-s` (oracle solves) · mechanism: P-7 (`:375`, vocabulary with no referent), V-17, P-10 · deliverable: `docs/plan/MK-20_safety_filter.md`, `results/bed_s_onestep_vs_twostep.jsonl` · evenings: 1.

**MK-21 — Score the `γ` LR verdict, the ablation and the mirror line on the MK-18 pair.**
phase P4 · prerequisites MK-11, MK-18 · independent-of MK-22, MK-23
· build/measure/PASS/KILL: as MK-11, on the eight seeds; SPLIT band `3/8–5/8` reported as SPLIT (Bet C).
· price: `0 GPU-s` beyond MK-18 (a re-read of the journal) · mechanism: V-9, M-20, D-7 · deliverable: `docs/plan/MK-21_gamma_verdict.md` · evenings: 1.

**MK-22 — Decide the `𝒜 = ∅` parity half by identity bind at `N = 8` and by TOST only at the priced `N` (parity by underpowering).**
phase P4 · prerequisites MK-18, MK-19 not killed at K-E1 · independent-of MK-21, MK-23
· build: the only contrast TOST decides is shape-at-`γ̂` vs ChaCAL-diag-at-`γ̂` with `𝒜 = ∅`; the margin `0.5σ` fixed here; paired TOST at `N = 36` if `sd_d = σ` (Monte Carlo `200,000` draws, `RUN[MARS]` of the design round), two-sample at `N = 70` (`0.7975` at `69`, `0.8073` at `70`, `RUN[VENUS]`); at `N = 8` the 90 % CI half-width is `0.8807σ` against a `0.5σ` margin, so two bit-identical arms return NO VERDICT (`READ workdonenew.md:400`).
· prove: none.
· measure: the realised `sd_d`; the TOST verdict at the `N` it licenses (`schuirmann-1987-tost` [V]).
· PASS: equivalence within `0.5σ` at the licensed `N`.
· KILL: any "parity" or "equivalent" sentence resting on `N = 8` ⇒ struck; a detected difference at the licensed `N` ⇒ the identity half of B-E1 is wrong at trained `γ̂` and the "ChaCAL-diag is the shape with `𝒜 = ∅`" sentence is limited to the bitwise bind at construction.
· price: paired `36 × 3.204 + 4 ≈ 119 s ≈ 1.98 GPU-min`; two-sample `70 × 3.204 + 4 ≈ 228 s ≈ 3.80 GPU-min` DERIVED · mechanism: M-13, M-9, V-5 (`:98`) · deliverable: `results/bed_s_tost.jsonl` · evenings: 1.

**MK-23 — Run the depth-`⌊log₂ t*⌋ + 2` softmax skyline and score Bet D with its sign logged.**
phase P3 · prerequisites MK-18 · independent-of MK-21, MK-22
· build: depth `5` at `t* = 8`, unmatched parameters, chosen *by analogy* with `sanford-2024-logdepth` Thm 4.2 (its `hop_k` is not the committor; the reduction is NOT FOUND, `sweep_expressivity.md` §3.5); `wang-2024-incontext-td` / `xie-2026-softmax-rl` [V] are the reason the stack computes the same resolvent by iteration.
· measure: `acc_shape − acc_sky`.
· PASS: within `MDE₈` (the shape's separate claims are exactness, one-read consistency as a cost statement, and the boundary mechanism).
· KILL: the skyline short by `> MDE₈` ⇒ the paper must state a capability gap it did not predict, with the sign logged; the skyline *ahead* by `> MDE₈` ⇒ "one operator" carries no accuracy sentence at all. "Beats softmax" is banned either way (R-SKY).
· price: `≈ 7.6 s` per cell `[ASSUMED linear in depth]` · mechanism: D-1, P-10, D-7 · deliverable: `results/bed_s_skyline_d5.jsonl` · evenings: 1.

**MK-25 — Run the cached-mixture arm on `ΔP` and `ΔV` plants (the re-solve is needed, or it is not).**
phase P3 · prerequisites MK-18 · independent-of MK-21..23
· build: `O_cached = P̂_base (I − γP̂_base)^{-1} V_int` with `P̂` frozen from the un-intervened context (the operational cached successor representation, `momennejad-2017-sr` / `russek-2017-predictive` [V], mechanism `[U]`); `ΔP` plants (row clamps) and `ΔV` plants.
· measure: cosine on `Δz` per plant class against the shape.
· PASS: fails on `ΔP` plants by `> MDE₈(cos)` and matches on `ΔV` plants.
· KILL (K-G1): within one seed sd of the shape on `ΔP` plants on `≥ 6/8` ⇒ the interventional channel is a per-row control wearing a name and component (g)'s re-solve sentence dies.
· price: `1.680 s` per cell `[FITTED + RUN]` · mechanism: P-7, D-2, V-9 · deliverable: `results/bed_s_cached_mixture.jsonl` · evenings: 1.

## 2. Three attacks nobody has filed (each `RUN[M]` this session on the judge's draw)

**MK-07 ★ — Census the sink-escape degeneracy of the Chebyshev rule and the exact-zero tie it creates.**
phase P1 · prerequisites MK-04 · independent-of MK-05, MK-06, MK-08
· the attack: with `𝒜_sink = {0}` declared (F1) and moves realised as clamps `P_env[v,:] ← e_u`, a clamp into any absorbing position gives `max_{k≥1} q^{(k)} = 0` exactly; the Chebyshev column `a† = argmin_a max_{k≥1} q^{(k)}` is therefore minimised *equally* by every move that sends the walk to the sink or to the goal. `RUN[M]`: at query 20, over the 20 clamp targets `u < 20`, the Chebyshev minimum is `0.000e+00`, attained by `9 of 20` targets, of which `7` carry sink share `> 0.5` and `2` goal share `> 0.5`; the `1e-9` tie rule of census line 6 would discard the draw. The safest move under Chebyshev is "fall off the prompt", and the tie-discard rule then empties the bed — nobody filed either half. Mechanism: V-12 in a new guise (a boundary that makes the label degenerate), V-8, V-10 (a rule satisfied by construction), V-5 (slice-to-nothing via discards).
· build: two census lines added to MK-04: (i) the fraction of admitted draws on which `a†` has `q^{(sink)}(a†) > q^{(0)}(a†)`; (ii) the discard fraction attributable to exact-zero Chebyshev ties. Construction fix registered *now*, not after the number: clamp targets restricted to transient positions (`u_a ∈ T`) so no move is a direct absorption, and the Chebyshev column conditioned on non-absorption (`argmin_a max_{k≥1} q^{(k)}(a)/(1 − q^{(sink)}(a))`) printed as a third column.
· prove: DERIVED — for `u ∈ 𝒜_j` the clamp gives `q^{(j)}(v) = 1` and `q^{(k)}(v) = 0` for `k ≠ j` exactly; the Chebyshev value of every absorbing-target move is `0` unless `j ≥ 1`.
· measure: lines (i) and (ii) on 512 draws, with and without the `u_a ∈ T` restriction.
· PASS: with the restriction, line (i) `< 0.05` and line (ii) `< 0.05`.
· KILL: line (i) `≥ 0.5` on the unrestricted bed ⇒ the Chebyshev column as written is struck and only the goal rule and the conditioned form are named; line (ii) `≥ 0.5` ⇒ the discard rule slices the bed to nothing and admission is refused until the restriction is in the spec. What would change this planet's mind: a restricted bed on which line (i) reads below `0.05` *and* the disagreement fraction of census line 7 stays `> 0` — then two rules survive and the attack is a census line.
· price: `0 GPU-s` · deliverable: `docs/plan/MK-07_sink_escape.md`, two columns in `results/bed_s_census_512.jsonl` · evenings: 1.

**MK-08 — Count no-op moves and correct the chance floor to the effective `m`.**
phase P1 · prerequisites MK-04 · independent-of MK-05..07
· the attack: on a causal `P_env` a clamp at a row `v > s₀` cannot change any committor at the query — `RUN[M]`: clamps at rows `21, 25, 31` move the query's four committors by `5.551e-17`, i.e. `0` to rounding — and a clamp at a row unreachable from `s₀` is a no-op for the same reason (F2 applied to moves, which nobody filed: F2 covers boundary sets, not candidate moves). A bed with `m = 8` of which `k` are no-ops has effective `m − k` distinct candidates; the chance floor `1 − 1/m = 0.875` and the tight Fano `(\ln m − \ln 2)/\ln(m − 1) = 0.7124` at `m = 8` (`RUN[J]`) are then too *low*, and a shape reading above the printed floor may be at chance on the effective set. Mechanism: L-FLOOR, V-10, V-17 (a floor in the wrong units — `m` nominal vs effective), D-3.
· build: a census line — for every move, `max_k |q^{(k)}(do a) − q^{(k)}|` at the query; a move reading `< 1e-12` is a no-op; the effective `m_eff` per draw; both floors recomputed at `m_eff`; the spec restricted to `v_a ≤ s₀` and `v_a` reachable from `s₀` (reachability read off the DAG), and the fraction of draws touched by the restriction printed.
· prove: `later_boundary_unreachable` (MK-01 row 3) extended to a clamp row — DERIVED from triangularity: `(M^{-1} e_j)_i = 0` for `j > i`.
· measure: `m_eff` distribution on 512 draws; the floor at `m_eff`.
· PASS: `m_eff = m` on `≥ 95 %` of admitted draws after the restriction.
· KILL: `m_eff < m` on `> 5 %` of draws in any cell that quotes a floor ⇒ every argmin accuracy in that cell is re-floored at `m_eff` and the census line becomes an admission line (D-4). What would change this planet's mind: a spec whose clamp rows are drawn from the query's ancestor set by construction, with the census reading `100 %`.
· price: `0 GPU-s` · deliverable: `docs/plan/MK-08_effective_m.md`, one column in `results/bed_s_census_512.jsonl` · evenings: 1.

**MK-24 — Register the clamp-target membership rule as a closed-form zero-hop control (the flag decides the argmin).**
phase P1 · prerequisites MK-04 · independent-of MK-05..08; consumed by MK-06 and MK-19
· the attack: the goal rule `a* = argmax_a q^{(0)}(do a)` is decided at zero hops whenever some clamp target lies in the goal set: `q^{(0)}(v) = 1.0000` exactly for `u = 5 ∈ 𝒜_0` (`RUN[M]`, the argmax on the judge's draw), and the goal flag of `u_a` is a token feature. The registered 0-hop control is a *trained* MLP (C-0H); a closed-form rule R0 — "pick the move whose clamp target carries the goal flag; else the move whose target carries no constraint flag and sits earliest" — costs nothing, needs no seed, and reads its accuracy at construction. The apparatus computes `I(X_{≤0}; a*)` (Prop. 8) but freezes no kill on a hand rule; the record's static beds died exactly this way (`READ D1.md:286-291` via `sec_refuted.md` C8). Mechanism: V-10, D-5, M-21 (a diagnostic prescribed by its statistic — a trained MLP can under-fit a rule a lookup solves), M-18.
· build: R0 and its two ablations (goal-flag only; earliest-transient only) as oracle-free columns computed at construction on every draw batch; the fraction of draws on which R0 is *defined* (some `u_a ∈ 𝒜_0`).
· prove: DERIVED — a clamp into `𝒜_0` yields `q^{(0)} = 1` and is the unique argmax unless another move also clamps into `𝒜_0`.
· measure: R0 accuracy against the oracle argmin, with Clopper–Pearson, on 512 draws; the same after the MK-07 restriction `u_a ∈ T`.
· PASS: R0 accuracy within `CP` of `1/m` after the restriction.
· KILL (frozen): R0 accuracy `≥ acc_shape − MDE₈` on `≥ 6/8` seeds of MK-19, or R0 above the exact zero-information floor by more than `CP` at construction on the unrestricted bed ⇒ the argmin label is a flag lookup; BED-S's argmin head is struck and only the committor-vector head remains. What would change this planet's mind: R0 at chance on the restricted bed *and* the trained 0-hop MLP at chance at MK-19.
· price: `0 GPU-s` · deliverable: `docs/plan/MK-24_flag_rule.md`, three columns in `results/bed_s_census_512.jsonl` · evenings: 1.

*Withdrawn candidate, recorded so it is not re-found (V-7 discipline).* The conditioning of the committor solve on a self-loop-saturated `Q̂`: `RUN[M]` with the diagonal logits raised by `0/4/8/12`, `max_T P_ii = 0.6927 / 0.9919 / 0.99985 / 0.999997`, `κ_∞(I − Q) = 7.9 / 214 / 771 / 810`, and the float32-vs-float64 solve gap `2.8e-08 / 2.8e-08 / 3.3e-08 / 1.9e-08` — it does not bite at `s = 32`. It stays as one printed line (`κ_∞(I − Q̂)` beside the committor head) and is not a card.

## 3. The DAG, the critical path, and the phase gates

```
P0 (any order, 0 GPU-s):  MK-01  MK-02  MK-03  MK-11  MK-14(spec)  MK-15  MK-17
P1 (0 GPU-s):             MK-04 ──► {MK-05, MK-06, MK-07, MK-08, MK-16, MK-20, MK-24}   (parallel)
P2 (seconds):             MK-09 (needs MK-04 admitted + MK-02) ;  MK-12 (needs MK-01 rows 1,5) ;  MK-13 (free-standing)
P3 (≈ 0.6–1.7 GPU-min):   MK-18 (needs MK-09, MK-15, MK-16) ──► MK-19 ; MK-23 ; MK-25   (parallel after MK-18)
P4 (≈ 2–4 GPU-min):       MK-21 (needs MK-11, MK-18) ; MK-22 (needs MK-19 alive at K-E1)
P5 (NOT MEASURED):        MK-10 (needs a mask) ; MK-14(run) ; T-1 digraph β₀, T-2 Mapper-to-tile, the LM γ cell behind the author's yes — no card here launches them
```

Critical path in evenings (this planet's own cards): MK-04 (2) → MK-07 (1) → MK-09 (1) → MK-18 (1) → MK-19 (1) → MK-22 (1) = **7 evenings**; MK-01 runs beside it and gates MK-12 and MK-18 (L-LEAN). Total GPU time at the design point if nothing dies early: `1.7 + 6 + 34 + 98 + 7.6 + 1.7 + 228 + 2.6 s ≈ 380 s ≈ 6.3 GPU-min` DERIVED from the card prices, all `[FITTED + RUN]` floors with the record's `2.0×–6.6×` dispatch gap beside them (`READ scale/m3_flops.py:101-121`); the `s`-sweep band (`3.4–9.0 GPU-min`) and the `≈ 36 min` ladder are spent only on a survivor.

Phase gates the author takes alone: after P1, admit or reroute (D-4) from the printed census; after P2, every plant failed at O(1) or the bind is out; after MK-18, the realised sd either licenses the `N = 8` kills or reprices `N`; after MK-19, K-E1 either fires (property paper) or escalates to `N = 16`; after P4, the two honest sentences of `design_falsify.md` §6 with their brackets filled. The record's own dominant failure — a bed dying at construction — is now a `0 GPU-s` decision (MK-04, MK-07, MK-08, MK-24); the expensive death (K-E1) is a `≈ 98 s` reading.

## 4. Limits (collected once)

Every `RUN[M]` number here is one numpy float64 draw at `s = 32` on this CPU — an identity or counterexample check, never a statistic; the three unfiled attacks are shown to have O(1) instances, not rates, and their kill thresholds (`0.05`, `0.5`, `5 %`, `≥ 6/8`) are pre-registration candidates frozen here without a measured null. BED-S has no cell, no realised sd, no measured `t*`; every MDE in this plan is the record's row-2 instance until MK-18 fills it. Prices are per-op floors on a laptop clock that is not stationary (`±12 %`); the depth-5 skyline price is `[ASSUMED]`; the `s`-sweep has a band and no point. The boundary-null critical value `2.7055` has no `references.bib` citation and is marked owed. The ChaCAL diagonal convention is `[V-fetched]` from one HTML read and is re-read against the PDF before MK-19's arm table is typeset. Theorem numbers inside cited sources inherit the sweeps' `[U]` marks. The Misra multichain hazard is carried, not resolved (MK-20 tests the one-step/two-step disagreement on the oracle only). No code file, no git write, no Kaggle contact, no external fetch was made by this planet.
