# The instrument roadmap — beds, binds, controls, certificates, guards, and the reading protocol

*SATURN (WATSON), 2026-09-03, HEAD `207e7b9`. This is the apparatus half of the programme: every bed, bind, control arm, certificate and guard of `judge/sec_apparatus.md` as a milestone card, followed by the reading protocol as milestone cards. The judged sections (`judge/sec_shape.md`, `sec_obstructions.md`, `proposition_ledger.md`, `sec_apparatus.md`, `bind_ledger.md`) outrank the designs where they differ; the ledger's `P0.*–P5.*` ids are cross-referenced as `[ledger P…]` so the coordinator can join the two. Evidence classes: `RUN` — executed this session by this planet (numpy float64, seed 0, `s = 32`, CPU; the two one-liners are listed in §7); `RUN[x]` — another planet's run, carried with its class; `READ path:line` at HEAD; `CITED [V]` by canonical `references.bib` key, the `[V]` being the owning sweep's abs/DOI fetch, `[U]` where the sweep says so; `DERIVED` with steps; `[FITTED]`/`[ASSUMED]` on prices per `sec_cost.md` §4.x.8. Every card names the `MISTAKES.md` mechanism it is built against by symbol (`READ MISTAKES.md`, 66 `###` headings including `V-14a`, `RUN grep`).*

*Format (`THESIS_CORRECTIONS_2.md` §0 is binding): the author develops alone, slowly, on the certified RTX 4060 (`READ COSTS.md:53-60`), in any order the DAG allows. Each card: **id · title · phase · prerequisites · independent-of (D-1) · build · prove · measure · PASS · KILL · price · mechanism · deliverable · evenings.** "build" is a specification a developer can implement — inputs, outputs, invariants, journal fields — never code. Nothing launches on Kaggle from this plan; the author's explicit yes is a node (`READ kaggle/README.md:9-11`). The five cheapest decisive items are marked ★. Prices: one 150-step shape cell on the softmax corner `1.680 s` `[FITTED + RUN]`, the softmax control `1.524 s` `[FITTED]`, one bed-cell pair at `N = 8` plus `4.0 s` fixed `≈ 34 s`, the seven-arm arena `≈ 98 s ≈ 1.6 GPU-min` (DERIVED), all from `sec_cost.md` §4.x.8 and `sec_apparatus.md` §A.11; the laptop clock is not stationary (`±12 %`, `READ V17_R4_RETAKE_PRICE.md:143-150`), so GPU-minutes are quoted to two decimals and never to the second (M-8, P-8).*

## 0. The DAG in one table

| phase | cards | gate the author takes alone at the end |
|---|---|---|
| 0 manifests and harness rules | S-01 ★, S-02, S-03 | the manifest refuses a missing field and its drift plant fires; identity scripts raise on a missing key (V-16, L-2) |
| 1 BED-S specification, census, guards | S-10, S-11, S-12 ★, S-13, S-14, S-15, S-16, S-17 | D-4 admission: every census line of §A.7 prints and none reads `0 %`; the VOID list is registered |
| 2 binds through the front door | S-20 ★, S-21, S-22, S-23, S-24, S-25, S-26, S-27, S-28 | every plant fails at O(1) as a FOUND cell with its own `kind` (V-14, V-24) |
| 3 control arms and skylines | S-30, S-31, S-32, S-33, S-34, S-35 | every arm has a `kind` and one journalled cell (FOUND, not NAMED) |
| 4 certificates and guards | S-40, S-41, S-42, S-43, S-44 | every printed `δ` is in vector units with `‖V‖_∞` and `1/(1 − γ̂)` beside it (L-CERT, V-17) |
| 5 the contained beds | S-50, S-51, S-52 ★, S-53 | BED-M, BED-K, BED-1 each carry one instrument reading and no capability number |
| 6 the reading protocol | S-60, S-61 ★, S-62, S-63, S-64, S-65, S-66 | predictions hashed before the first cell; `N = 8` in one lane; adjudicator written before data; verdict appended, prediction untouched |
| 7 deferred instruments | S-70, S-71, S-72, S-73 | `NOT MEASURED` until an instrument exists; none on the critical path |

Critical path (serial evenings): S-01 → S-11 → S-12 → S-14 → S-20 → S-21 → S-60 → S-61 → S-62 → S-63 → S-64 = **11 evenings** (S-65 adds 1 only if S-63 leaves the parity half undecided). Everything else runs beside it. Cards marked "beside" may be picked up in any order (D-1: no shared repository state; `results/*.jsonl` are append-only per lane, L-G2).

## 1. Phase 0 — manifests and harness rules (0 GPU-s)

### S-01 ★ — Extend the identity manifest so a cell cannot be journalled without its boundary condition `[ledger P0.2]`
phase 0 · prereq — · beside everything
**build:** a manifest specification for the shape lane extending `CONFIG_FIELDS` (`READ scale/identity_manifest.py:67-71`, which carries `beta` but not `qk`, `g`, `gamma`, and reports a missing declared field as `absent` outside the hash, `:141-151`). Fields, each with its support: `beta, qk, g` (corner switches; `beta` learnable per Ruling 2a); `gamma` (init, final `γ̂`, `Λ`, verdict token); `diag_convention ∈ {kept, removed}`; `committor_route ∈ {dirichlet_gamma1, discounted_gamma}`; `route ∈ {solve_triangular, neumann_K, segmented, csr}` + `K` + chunk `C`; `boundary_sets, goal_set, sink_set` as sorted position lists with sha256 each; `K, m, t_star`; `S, D, d_model, n_train, n_eval, steps`; `seed, rng_plan`; `device, threads, dtype, torch_version, cublas_workspace, deterministic_regime`; `instrument_hash, manifest_hash`; `producer_cmd`; `delta_vec, delta_bare, one_over_1mg, V_inf`; `census` block; `floor_exact, floor_zeroinfo, floor_fano_k1, ceiling_hop_k`; `sky_depth, sky_width, sky_cot, chacal_gamma, chacal_diag, params_per_arm`; `void_contrasts`; `supersedes` (`sec_apparatus.md` §A.10). Invariant: **a missing declared field is a refusal, not an `absent` entry** — the record's reason for reporting instead of raising (stored manifests would be invalidated, `READ :146-148`) does not apply to a lane with no stored manifests.
**prove:** none (an instrument).
**measure:** the drift plant — flip any one of `beta / qk / g / gamma / boundary_sets` by one unit and read `manifest_hash`; the pattern is `test_moving_one_citation_by_one_line_fires_both_binds` (`READ V20_R15_WING_MANIFEST.md:148-152`).
**PASS:** `5/5` single-field flips move the hash **and** an omitted `sink_set` raises. **KILL:** any flip leaving the hash unchanged, or an omission that returns `absent` — the lane has no identity and no cell may be journalled (L-2: `0 of 24` R15 cells attributable, `READ V20_R15_LEAP_LEDGER.md:23`).
**price:** 0 GPU-min · **mechanism:** L-2, P-1, V-16, M-10, M-16 · **deliverable:** `docs/apparatus/MANIFEST_SHAPE_LANE.md` (field table + the plant's expected outputs) · **evenings:** 1

### S-02 — Fix the identity-script rule: raise on a missing key, never default
phase 0 · prereq — · beside everything
**build:** a written rule for every identity check in the lane: bed keys are read by name and a missing key raises; no `.get(key, default)`. The filed instance: `$SCRATCH/shape_identities.py` asked `bed_1.build` for `basin_A/basin_B`, which it does not return (its keys are `A`, `B`, `READ ceq/beds/bed_1.py:167`), defaulted silently, and printed `0.858` for an identity that reads `0.0` on the real sets (`sec_refuted.md` C12; `RUN[coord]` corrected).
**measure:** the must-fire — call the check with a misspelled key and require an exception; the must-not-fire — the real keys return `max|q − (I − Q)^{-1}R𝟙| = 0.0` (`RUN[coord]`, `A = [0], B = [1], |T| = 9`).
**PASS:** exception on the misspelt key, `0.0` on the real one. **KILL:** a number returned on a misspelt key — every identity number in the lane is suspect until the rule is enforced.
**price:** 0 · **mechanism:** V-16, V-3 · **deliverable:** `docs/apparatus/IDENTITY_SCRIPT_RULES.md` · **evenings:** 0.5

### S-03 — Print the obstruction vacuity inequalities beside every obstruction citation `[ledger P0.3]`
phase 0 · prereq — · beside everything
**build:** a one-page census of the three unconditional lower bounds as predicates on the record's geometry (`s = 64, d = 16, h = 1, p = 32`): `h·m·p = 512 ≥ 64` (`sanford-2024-inductionheads` Thm 1), `n log₂ n = 384 < 544 = H(d+1)p` (`peng-2024-transformer-limitations` Thm 1), `64^{1/16} = 1.2968` against `H·d·p = 512` (`chen-2024-multilayer` Thm 1.1) — all vacuous here (DERIVED, `sec_obstructions.md` §5.2–5.4); the conditional bounds (`sanford-2024-logdepth` Cor 4.3) named conditional in the same sentence.
**PASS:** every obstruction sentence in the paper carries its inequality and its `[V]`/`[U]` theorem-number mark. **KILL:** any obstruction stated without its line, or `0 %` of a bed's draws admitted by the theorem's hypothesis — the obstruction is not stated for that bed (K-10).
**price:** 0 · **mechanism:** V-25, P-10, P-3 · **deliverable:** `docs/apparatus/OBSTRUCTION_CENSUS.md` · **evenings:** 0.5

## 2. Phase 1 — BED-S: specification, census, guards (0 GPU-s)

### S-10 — Register the placement convention: BOS is a value-zero sink set, boundary sets precede the query
phase 1 · prereq — · beside S-01..S-03
**build:** the two construction facts as a written convention. **F1:** row 0 of any causal softmax at `β = 1` is `e_0`, so position 0 is absorbing whether or not declared; undeclared, the transient block is singular. **F2:** walks descend, so a boundary set after the query is unreachable. BED-S therefore declares `𝒜_sink = {0}` with value `0` on every indicator channel (not a goal member — with `0 ∈ 𝒜_0` "reach the goal" collapses to "descend to 0", `refute_falsify_math` §3.11 via `sec_apparatus.md` §A.1), and places `𝒜_0, 𝒜_1..𝒜_K` and the sink before the query, the query in `T`.
**measure:** `RUN` this session: constraint/goal sets `{5,6},{9,10},{15}` with BOS undeclared give `ρ(Q) = 1.000000`, `det(I − Q) = 0.000e+00` (must fire: the solve raises); with `𝒜_sink = {0}` declared, `ρ(Q) = 0.692660 = max_{i∈T} P_ii`, sum-to-one on `T` to `4.4e-16`, sink share in `[0.362, 1.000]` (must not fire); a set `{20}` read from query `12` gives `q = 0.0` exactly (must fire: discard), from query `25` gives `q = 0.1033` (must not fire).
**PASS:** the four readings reproduce on the bed's own generator at `s = 64`. **KILL:** a declared-sink draw with `ρ(Q) = 1` — the generator has a second undeclared absorbing row (a zero gate at `c`, Proposition 6(ii)) and the census line 14 must be re-read.
**price:** 0 · **mechanism:** V-25, V-12, V-8, D-3 · **deliverable:** `docs/beds/BED_S_PLACEMENT.md` · **evenings:** 0.5

### S-11 — Write the BED-S generator specification (DAG substrate, sink/goal/constraints, query-side clamps) `[ledger P1.1, S-1]`
phase 1 · prereq S-10 · beside S-01..S-03, S-50..S-53
**build:** the environment is a random directed graph on the `s` token positions whose edges point to earlier positions (a DAG in token order, so a walk from the query descends as the arm's causal `P̂` does), self-loops only on declared absorbing positions; `P_env` row-stochastic including absorbing rows. Node tokens carry the out-adjacency as a multi-hot feature, the membership flags of the absorbing sets, and nothing derived from any solve; the transition matrix is a deterministic function of the edge tokens (so `sec_refuted.md` C2's leak clause (a) is dropped for BED-S and clause (c) kept). Candidate moves: `m` query-side row clamps `P_env[v_a, :] ← e_{u_a}`, each rank-one with `uᵀ𝟙 = 0` (graph-surgery precedent `READ scale/negation_scope.py:718-732`); a token rewrite is rank `s − i` and is a *different* object priced as a suffix re-solve (Proposition 7(iii)). Label per move: the reach-avoid tensor `(q^{(sink)}, q^{(0)}, q^{(1..K)})` at `γ = 1` on the transient block (`sec_apparatus.md` §A.2; `kemeny-1976-finite`, `grinstead-1997-probability` Thm 11.6 `[U]`), the full vector on `T` retained for the residual; `a* = argmax_a q^{(0)}`, the Chebyshev `argmin_a max_{k≥1} q^{(k)}` (`vanmoffaert-2013-chebyshev`) and the lexicographic form (`yang-2026-lexisafe`) as columns; `z*`, `Δz(a)` by the C6 identity on `P_env` at a bed constant `γ_env`. Dials and their registered supports: `t* ∈ {2, 8, 32}` (tolerance dial read off the hop ladder, `E2_DIAL_TOL`, `READ scale/negation_scope.py:264`), `K ∈ {2, 3, 4}`, `m ∈ {4, 8, 16}` default `8`, `|𝒜_•| ∈ {1, 2}`, `s ∈ {64, 256, 1024, 4096}`; the curriculum cited as a necessity (`wang-2025-easytohard`). Oracle: `bed_1.committor`'s Dirichlet form (`READ ceq/beds/bed_1.py:188-198`) on `P_env`; Kirchhoff second route for `K = 2` single-node sets only (`READ scale/kirchhoff.py:1-90`); multi-node extension `NOT MEASURED — needs the grounded Laplacian extended` (P-4).
**prove:** none here; the identities are S-2x.
**PASS:** the spec names every field the manifest of S-01 consumes and every census line of S-12; the oracle is stated to run on `P_env`, never on `P̂`. **KILL:** any sentence handing the arm `P_env` as an input channel other than the edge tokens (D-2), or a spec in which `K = 1` is legal (V-12: committor constant to `1.11e-14`, `READ MISTAKES.md:189-199`).
**price:** 0 · **mechanism:** D-2, V-25, V-12, V-8, D-3, M-8 · **deliverable:** `docs/beds/BED_S_SPEC.md` · **evenings:** 2

### S-12 ★ — Run the BED-S domain census on 512 draws and decide admission `[ledger P1.2, S-2]`
phase 1 · prereq S-11 · beside S-01..S-03, S-50..S-53
**build:** the fifteen census lines of `sec_apparatus.md` §A.7 printed per draw batch, before any arm is trained, as one JSON block per batch: (1) `|rowsum − 1| ≤ 1e-12` on every row incl. absorbing; (2) `0 ∈ 𝒜_sink` on `100 %`; (3) `ρ(Q̂) = max diag < 1` (a V-10 gate — printed, not counted); (4) every set before the query, query in `T`; (5) per-coordinate label sd of `q` **over the admitted query region** (the prefix before the first constraint reads `q^{(0)} ≈ 1` and inflates the sd), sink share printed apart from `q^{(0)}`; (6) every class frequency of `a*` in `(0.05, 0.95)`, ties within `1e-9` discarded and counted; (7) rule-disagreement fraction `> 0`; (8) corpus-alone linear probe to `q` at order 0 (S-15); (9) `I(s₀; a*) = 0` plug-in (S-14); (10) `Var(Δz) > 0` over coordinates `≥ i_min`, zero-coordinate fraction printed; (11) move census (`ΔP` vs `ΔV` moves; `uᵀ𝟙 = 0` and non-negative edited row on `100 %`; the Sherman–Morrison denominator `(1 − γp'_ii)/(1 − γP_ii) > 0` printed as a check); (12)–(13) filled on trained cells (S-42, S-43); (14) exact-zero-gate count (`0 of N` on the softmax corner ⇒ F0 theorems silent); (15) the S-03 lines. `nrmse` returns `nan` on a constant label and `nan >= 1.0` is `False` (`READ MISTAKES.md:149-153`), so the sd print is not optional.
**measure:** 512 draws at the design point `t* = 8, m = 8, K = 2, s = 64`.
**PASS:** every `sd(q^{(k)}) > 0.05` over the admitted region; every class frequency in `(0.05, 0.95)`; `0 ∈ 𝒜_sink` and every set before the query on `100 %`; disagreement fraction `> 0`; discard fraction printed. **KILL (K-E2, K-P):** any `sd = 0`; any class frequency outside the band; disagreement `0` (only one rule may be named, C4); or `t*` unplaceable in `{2, 8, 32}` on the DAG depth — the bed is not admitted, no reading exists, and the reroute is the jittered `bed_1` landscape with `K = 2` and `B` as goal (`design_falsify.md` §5 rank 1), repriced from its own realised sd.
**price:** 0 GPU-min (the oracle is `m` factorisations with `K + 2` right-hand sides, under a second per draw in float64, DERIVED `sec_apparatus.md` §A.11) · **mechanism:** V-8, V-12, V-25, V-10, D-3, D-4, M-3 · **deliverable:** `results/bed_s_census.jsonl` + `docs/beds/BED_S_CENSUS.md` (the admission decision, signed by the author) · **evenings:** 1

### S-13 — Register the VOID-contrast list and the identification metric before any cell `[ledger S-3]`
phase 1 · prereq S-11 · beside S-12
**build:** because the arm's operator class contains the oracle's chain, `shape − softmax` and `shape − skyline` on the committor head are reproduction-versus-non-reproduction contrasts and are VOID as capability numbers (Proposition 8). Creditable: `shape − ChaCAL-diag`, `shape − ChaCAL-published`, `shape − ChaCAL-with-sink-token`, `shape − InfSA-style Neumann-K`, `shape − 0-hop MLP` and `shape − 1-hop softmax` on the argmin, and `‖P̂ − P_env‖_∞` per seed as a **learnability** reading. The model is E1's registration comment (`READ scale/negation_scope.py:1033-1041`).
**PASS:** the list is in the manifest field `void_contrasts` on every cell (S-01). **KILL (K-D2):** `‖P̂ − P_env‖_∞ < 1e-3` on `≥ 6/8` seeds — the arm copied the environment; the reading is filed as identification, never as capability.
**price:** 0 · **mechanism:** D-2, M-7, V-10 · **deliverable:** `docs/beds/BED_S_VOID_LIST.md` · **evenings:** 0.5

### S-14 — The zero-hop guard: query carries `s₀` only; the untrained arm reads chance `[ledger P1.3, X-9]`
phase 1 · prereq S-11 · beside S-12, S-13
**build:** `s₀` drawn uniformly from `T` independently of `a*`; the plug-in `I(s₀; a*)` printed with its sampling error; the 0-step RED gate of the harness applied unchanged — untrained arm within `1/m ± CP` on the argmin and committor NRMSE `≥ 1 − GATE_TOL` (`GATE_TOL = 1e-3`, `READ COSTS.md:137`; `READ scripts/k_cert.py:222`). Precedent: BED-M's first builder leaked `1/(t*+1)` at zero hops and aborted three of five rungs; `b[s−1] = 0` restored it (`READ scale/negation_scope.py:399-415`).
**measure:** must fire — a query token carrying `q^{(0)}(a*)` must push the untrained read below `1 − GATE_TOL`; must not fire — the honest query reads at chance within CP.
**PASS:** both halves. **KILL:** the honest query below the gate — the label is legible at zero hops and the bed is a copy task.
**price:** `≤ 1.524 s` (one 0-step forward, `[FITTED]`) · **mechanism:** V-10, D-5, V-24 · **deliverable:** `results/bed_s_zerohop.jsonl` · **evenings:** 0.5

### S-15 — The leak guard, firing both ways `[ledger P1.4, X-8]`
phase 1 · prereq S-11 · beside S-12..S-14
**build:** corpus-alone linear probe from `x` to `q` at order 0; planted positive — a move token carrying its own `q`; planted negative — the strictly-local features E4′ uses (degree, ball sizes to radius 5, absorbing-endpoint-in-ball flags, `READ scale/e4_harmonic.py:220-244`), whose bars `PASS_BAR = 0.5 / FAIL_BAR = 0.9` (`READ scale/rips_gate.py:60-61`) are E4′'s and are **re-derived for BED-S** before use (V-22).
**PASS:** honest `R² < 0.5`; planted leak `R² ≥ 0.99` (the detector fires when the leak is planted, V-7 calibrated both ways). **KILL:** honest `R² ≥ 0.5` at order 0 — the one-read claim is void on the bed (C2(c)).
**price:** 0 · **mechanism:** V-24, V-7, V-22, D-2 · **deliverable:** `results/bed_s_leak.jsonl` · **evenings:** 0.5

### S-16 — Fix the metrics and floors before any arm exists (declared, not chosen after the curve)
phase 1 · prereq S-11 · beside S-12..S-15
**build:** committor head scored in the Fisher–Rao coordinate `φ(p) = 2 arcsin √p` per entry, position-matched NRMSE on `φ` (mean, max), plus the harmonic residual `r(q̂)` with the environment's `Q_env, R_env` — a **score, never a loss** (D-2) — printed beside `σ_min(I − γP_env)` and `‖I − γP_env‖_∞ ≤ 1 + γ` (its units); argmin accuracy with Clopper–Pearson (`clopper-1934-binomial`) and McNemar paired on identical draws (`mcnemar-1947-correlated`); `z*`/`Δz` by the position-matched per-coordinate NRMSE vector, the field cosine and magnitude ratio over coordinates `≥ i_min` (an arm emitting `Δẑ ≡ 0` is refused, V-16), the sign column kept because softmax owns it (`0.807843`, `READ MATHEMATICS.md:396-406`), McNemar on the sign column only, paired `t`/Wilcoxon on the cosine. Floors (L-FLOOR): exact oracle at `0.0` with the hop-ladder ceilings `NRMSE(z_k, z*)` printed per batch (`READ scale/e4_harmonic.py:194-199`); argmin zero-information floor `1 − max_a π̂(a*)` = `0.875` at uniform `m = 8` (`RUN[J]`); tight Fano `(ln m − I − ln 2)/ln(m − 1)` only where `I(X_{≤k}; a*) > 0`; the weak `1 − ln 2/ln m = 0.6667` retired (it sits below chance); `floor₁` nowhere. Refused metrics: pooled `W1` (permuted oracle scores `0.0` at NRMSE `1.421901`, `READ V20_R15_THEORY_TABLE.md:221`), Procrustes, position-free OT.
**PASS:** the metric file is hashed into `PREDICTIONS` (S-60) before S-62. **KILL:** any metric added or removed after S-62's first cell (M-2).
**price:** 0 · **mechanism:** M-2, V-26, V-17, L-FLOOR, V-16 · **deliverable:** `docs/apparatus/METRICS_AND_FLOORS.md` · **evenings:** 1

### S-17 — The `Δz` channel's own census and the cached-mixture plant list
phase 1 · prereq S-11 · beside S-12..S-16
**build:** for the `z`/`Δz` lane (its own journal — a vector label voids `PUBLISHED_SOFTMAX_8192`, `READ MISTAKES.md:701-708`): the intervened position drawn uniformly; `Var(Δz) > 0` per coordinate `≥ i_min`; per-draw flags `move_changes_P` vs `move_changes_V_only` so that S-33's cached-mixture control can be scored on `ΔP` plants and `ΔV` plants separately; the no-change predictor's error printed beside every `Δz` number (`vakalis-2026-interventiongap` owns that floor).
**PASS:** both plant classes non-empty (`≥ 64` draws each). **KILL:** a corpus with no `ΔP` move — the interventional channel has no rejection region against the cached mixture.
**price:** 0 · **mechanism:** V-8, D-5, V-24, M-1 · **deliverable:** `results/bed_s_dz_census.jsonl` · **evenings:** 0.5

## 3. Phase 2 — the binds, each through the front door (seconds)

Rule for every card here (`sec_apparatus.md` §A.8): a bind is admissible only if the honest construction passes **and** every plant fails at O(1) with counts printed — the `0.9749 / 0.9165 / 1.000 / 0.4845` pattern (`READ workdonenewseal.md:114-122`). Every plant enters at the **front door** — `bed_s.build → arm.forward → journal row → verdict()` — as a planted-negative cell FOUND under `results/` with its own `kind` and `manifest_hash` (V-14; FOUND iff `results/` holds a record with that `kind`, `READ V20_R15_WING_MANIFEST.md:59-63`), never as an identity check on random logits.

### S-20 ★ — B-J parity at `γ = 0` and the ChaCAL-diag smoke test at `γ = 0.9` on BED-M `[ledger P2.1, B-J, B-P5]`
phase 2 · prereq S-01, S-02 · beside S-1x
**build:** `torch.equal(O(0), P V)` against the lane's own `softmaxAttn` (the record's corner is `1.110223e-16` off `ceq/lm.py` on `19/64` entries, `READ V16_ARM_SMPRIME.md:266-293` — named, not rounded); then one 150-step ChaCAL-diag cell at fixed `γ = 0.9` on BED-M `e3_t2` as the solve-path smoke test (`fagnou-2024-chacal` App. C's own setting). Plants: `γ = 0.5` (`RUN` this session `max|O − PV| = 1.6143` at `s = 32`; `RUN[coord]` `2.3003` at `s = 64`); `β = 0` at the softmax corner (`> 0.5`, `READ V16_ARM_SMPRIME.md:336-339`); ChaCAL-published's `A_s` (diagonal removed inside the inverse) at any `γ > 0`; B-P5 — a dense non-triangular `M` passed with `upper=False` must disagree with the dense inverse. Struck plant: "a non-causal `W`" passes bitwise at `γ = 0` because `(I − 0·W) = I` regardless of support (`RUN[MARS]`).
**measure:** must not fire — `torch.equal True` at `γ = 0` (`RUN` this session `True`; `RUN[coord]` `True`); `solve_triangular` vs dense inverse `1.8e-15` (`RUN[coord]`). Must fire — the four plants at O(1). The smoke cell reads within the law-vs-measured band (`−1.8 %` to `+14.4 %`, `READ V17_R4_RETAKE_PRICE.md:228-229`) of `1.680 s`.
**PASS:** identity `True`, `4/4` plants at O(1), cell price inside the band. **KILL:** any plant passing (V-24), or a cell above `2.2×` the law — the microbenchmark increment is the P-8 floor it is declared to be.
**price:** `≈ 0.03 GPU-min` (one cell) · **mechanism:** V-24, V-3, M-8, P-8 · **deliverable:** `results/binds/b_j_parity.jsonl` (kinds `shape_g0`, `shape_g05_plant`, `shape_b0_plant`, `chacal_pub_plant`, `dense_M_plant`) · **evenings:** 1

### S-21 — B-E1 boundary rows against ChaCAL-diag `[ledger P2.2]`
phase 2 · prereq S-12 (admitted), S-20 · beside S-22..S-28
**build:** identity half — with `𝒜 = ∅` the shape is `torch.equal` to **ChaCAL-diag**, the lane's own re-implementation with `diag_convention = kept` declared in the manifest (a declared V-3). Rejection half — with `𝒜 ≠ ∅`, `‖Π_shape − Π_ChaCAL-diag‖_∞` on rows downstream of the boundary positions exceeds a printed O(1) gap; this is a **weight** statement, not a support statement (`RUN[MARS]` rows changed `= [3]`; on a dense causal softmax the support of `Π_γ` equals the support of `P` with or without boundary rows, `RUN[J]`). Struck plant: "a non-identity absorbing row breaks `Σ_k q = 1`" — empty (`RUN[MARS]` `0.0`; the sum uses transient rows only); replaced by the full-`P` read at `γ < 1`, where it reads `0.1491`. The InfSA base `Â = ReLU(QKᵀ)/‖·‖_F` (`roffo-2026-infsa`) must break the identity half — `NOT MEASURED — needs the base wired`.
**PASS:** `torch.equal True` at `𝒜 = ∅`; downstream gap `≥ 0.1` at `𝒜 ≠ ∅` on `100 %` of admitted draws. **KILL:** identity half fails (the lane's ChaCAL-diag is not the shape at `𝒜 = ∅` — a P-7 object) or the gap reads `< 1e-3` on any draw (boundary rows change nothing on that draw; census line 4).
**price:** seconds · **mechanism:** V-24, V-3, V-14, P-7 · **deliverable:** `results/binds/b_e1_boundary.jsonl` (kinds `shape_A0`, `chacal_diag`, `shape_A2`) · **evenings:** 1

### S-22 — B-E2 conservation with the sink set `[ledger P2.3]`
phase 2 · prereq S-12 · beside S-21, S-23..S-28
**build:** `q^{(sink)} + q^{(0)} + Σ_k q^{(k)} = 𝟙` on `T` printed with its residual on every batch including when it fails (V-23). Plants: drop BOS from every set ⇒ the solve must **raise** (`RUN` `ρ(Q) = 1.000000`, `det = 0.000e+00`), never return a number (V-16); a set after the query ⇒ `q = 0.0` exactly and the draw is discarded by census line 4 (`RUN`). Struck plant: "drop `𝒜_0 ⇒ max_k q^{(k)} ≥ 1/K`" — with the sink present the constraint sum is `< 1` and the pigeonhole is a V-3 of `Σ = 1`; the degeneracy lemma (no goal and no sink ⇒ `max_k ≥ 1/K`) is carried as a **proposition**, its plant being "BOS declared inside a constraint set".
**PASS:** residual `≤ 1e-12` on `100 %` (`RUN` `4.4e-16`; `RUN[I]` `[0.9999999999999993, 1.0]`); the BOS-drop raises; the after-query set discards. **KILL:** a number returned on the BOS-drop (the solver silently regularised — V-16), or residual `> 1e-8` on any draw with declared sets (a second undeclared absorbing row).
**price:** 0 · **mechanism:** V-12, V-23, V-16, V-3 · **deliverable:** `results/binds/b_e2_conservation.jsonl` · **evenings:** 0.5

### S-23 — B-G1 the interventional re-solve, with the triangularity plant `[ledger P2.4, B-C6]`
phase 2 · prereq S-17 · beside S-21, S-22, S-24..S-28
**build:** (a) `V ≡ 𝟙 ⇒ max|Δz| ≤ 1e-15` (`RUN[SATURN]` `0.0`, `RUN[MARS]` `1.1e-16`; the exact zero is a code-path accident, so the bar is `≤ 1e-15`, not "bitwise"); (b) `V ~ N(0,1) ⇒ max|Δz| = O(1)` (`0.1096 / 0.363 / 1.127` on three draws); (c) Sherman–Morrison closed form vs re-solve `≤ 1e-12` (`1.2e-15 / 8.9e-16 / 4.4e-16`; `sherman-1950-inverse-adjustment`, `hager-1989-updating`); (d′) **triangularity plant** — a non-causal `P` makes `Δz[:i] ≠ 0` (`0.0761 / 0.0868`), the route named (`solve_triangular` gives `Δz[:i] = 0.0`, `equal True`; LU gives `8.9e-16`, `False` — D-5); a two-row edit breaks the rank-one formula by O(1) and is repaired by Woodbury; (e) the C6 identity `Δz = (I − γP')^{-1}(ΔV + γΔP z)` residual `≤ 1e-12` (three independent RUNs `1.03e-15 / 1.2e-15 / 1.36e-15`). The EMC "feedback plant" is **deleted**: for `γ < 1` the fixed point is unique on every `P` (`RUN[MARS]` `1.33e-15`, `4.4e-16` on dense cyclic `P`); `dash-2005-emc` survives as a remark.
**PASS:** (a),(c),(e) within bars; (b),(d′) at O(1). **KILL:** (b) reading `< 1e-6` on Gaussian `V` (empty rejection region), or (d′) reading `0` on a non-causal `P` (the solver is masking, not solving).
**price:** 0 · **mechanism:** V-24, D-5, D-7, V-3 · **deliverable:** `results/binds/b_g1_resolve.jsonl` · **evenings:** 0.5

### S-24 — B-G2 the vector-metric plants `[ledger P2.5]`
phase 2 · prereq S-16 · beside S-21..S-23, S-25..S-28
**build:** the journal carries the vector, a histogram and quantiles (the Q6 census read `0 of 40`, `READ V20_R15_THEORY_TABLE.md:213`). Three plants on the metric itself: (i) the permuted oracle must **not** score `0`; (ii) `oracle + 0.1σ` must be preferred to the permutation; (iii) `−Δz` must be distinguished from `Δz`.
**PASS:** `3/3`. **KILL:** any plant passing — the metric is struck (marginal `W1` fails (i) and (ii) by `14.465×`, `READ :221`).
**price:** 0 · **mechanism:** V-26, L-14, V-16 · **deliverable:** `results/binds/b_g2_metric.jsonl` · **evenings:** 0.5

### S-25 — B-H1 the committor identity on the environment chain and on the arm's `Q̂` `[ledger P2.6]`
phase 2 · prereq S-02, S-12 · beside S-21..S-24, S-26..S-28
**build:** on the environment chain (BED-1 as instrument, `READ ceq/beds/bed_1.py:188-198`): `q = (I − Q)^{-1}R𝟙` vs `bed["q"]` on the real sets `0.0` (`RUN[coord]`), `harmonic_residual = 1.04e-17`, Kirchhoff second route `< 1e-10` at `K = 2` (`READ MATHEMATICS.md:542-600`). On the arm's causal `Q̂`: invertible iff `0 ∈ 𝒜` (`lower_triangular_isUnit`, Proposition 10) — a diagonal read `ρ(Q̂) = max_T P̂_ii` (`RUN` `0.692660` both ways). Plants: the must-fire perturbation drives the residual from `1.04e-17` to `1e-6` (ratio `9.6e10`, `READ workdonenewseal.md:214-215`); declaring `𝒜_k` on the wrong set moves `q` by O(1).
**PASS:** `0.0` on the real sets, both plants at O(1). **KILL:** the residual not moving under the perturbation (the residual reads a cached `q`, V-3), or the arm-side solve returning a number with BOS undeclared (V-16).
**price:** 0 · **mechanism:** D-2, V-3, V-16 · **deliverable:** `results/binds/b_h1_committor.jsonl` · **evenings:** 0.5

### S-26 — B-H2 the argmin identity and the static-task kill `[ledger P2.7, K-H2]`
phase 2 · prereq S-14, S-15, S-16 · beside S-21..S-25, S-27, S-28
**build:** the argmin from the oracle tensor equals the label's argmin on `100 %` of admitted draws (an identity of the builder, declared V-3); the rejection region is the control pair of S-34: a 0-hop per-position MLP on the candidate's context row and a 1-hop softmax, matched at `4,769` parameters (`READ CEQ_V20_R15_CONTRACT.md:119`; the count re-printed per arm once a `[m, K+2]` head is attached — Ruling 3), McNemar on identical draws.
**PASS:** identity `100 %`; the 0-hop MLP outside the CP half-width of the shape at `N = 8` **and** McNemar `p ≤ 0.05` against the 1-hop softmax. **KILL (K-H2):** the 0-hop MLP within the CP half-width, or `p > 0.05` — BED-S is a third static task and is struck before any number is quoted (C8; the record's static beds gave an iterating arm nothing to compute toward, `READ D1.md:286-291`).
**price:** `≈ 0.5 GPU-min` (two control arms × 8 seeds ≤ `1.524 s` each, `[FITTED]`) · **mechanism:** V-24, V-3, C8, V-10 · **deliverable:** `results/binds/b_h2_argmin.jsonl` (kinds `mlp_0hop`, `softmax_1hop`) · **evenings:** 1

### S-27 — B-K containment and segmentation, float instances `[ledger P0.1 float half, B-K]`
phase 2 · prereq — (Lean grades are JUPITER's plan) · beside everything
**build:** the float instances the `[M]` targets must match: corner 3 vs `(I − A)^{-1}` `0.0` entrywise and last row vs `equilibrium_oracle` `6.2e-15` (`RUN[coord]`, `READ scale/negation_scope.py:286-304`); segmentation zeros by `torch.equal`, never `allclose`, on a `−∞`-masked `P`. Plants: keeping the diagonal breaks `A^s = 0` (`Nilpotent.one_not_nilpotent`, `READ lean/CEQ/Nilpotent.lean:105`); a `−30` logit (F1, not F0: `e^{−30} = 9.36e-14`) leaves the block **non**-zero. Struck plant: `1e-300` — float64 underflow after `≥ 80` gates at `0.5` reads F0 (`RUN[MARS]`). Census: on BED-S at the softmax corner the exact-zero-gate count is `0 of N` and the F0 theorems are silent; on BED-M `3 of 3` values (`READ lean/CEQ/V16Domain.lean:304`).
**PASS:** `0.0`, `6.2e-15`, `torch.equal` zeros; both plants at O(1). **KILL (K-K):** any `[M]` target not building by the Lean milestone ⇒ `[S]`, and the paper cites only declarations that build (P-11: the record's `#18/#19/#22 [M]` have no declarations, `sec_proved.md` §2.0).
**price:** 0 · **mechanism:** P-11, V-25, V-2, L-LEAN · **deliverable:** `results/binds/b_k_containment.jsonl` · **evenings:** 0.5

### S-28 — The front-door plant register: every plant is a FOUND cell with its own `kind`
phase 2 · prereq S-01, S-20..S-27 · beside —
**build:** one table listing, per bind, the plant's `kind`, its `manifest_hash`, the journal path, the O(1) number it produced, and the verdict-line that consumed it; the census of `kind`s over `results/**/*.jsonl` calibrated both ways as the wing manifest did (`arm_phase` must read `0`, `softmax` `> 0`; `READ V20_R15_WING_MANIFEST.md:73-83`).
**PASS:** every plant of S-20..S-27 FOUND; the two calibration reads. **KILL:** any plant that exists only in a script's print (NAMED) — the bind is not admissible (V-14: the control that validates the matcher and never the reach).
**price:** 0 · **mechanism:** V-14, V-7, P-1 · **deliverable:** `docs/apparatus/PLANT_REGISTER.md` · **evenings:** 0.5

## 4. Phase 3 — control arms and skylines (each must be FOUND)

### S-30 — The depth-1 softmax control with the same head, parameter count re-printed per arm `[ledger C-SM1]`
phase 3 · prereq S-12 · beside S-31..S-35
**build:** the matched control at `4,769` parameters (`0.032 %` residual is matched, Ruling 3, `READ V17K_RULINGS.md:56-59`); the count is re-printed once the `[m, K+2]` head or the vector readout is attached. It is the per-row control (Obstruction 1, definitional) and the owner of the single-location ground (`duranthon-2026-softmax-advantage` Prop 4.2, whose label model admits `0 of 3` record beds).
**PASS:** one journalled cell per bed with `params_per_arm` in the manifest. **KILL:** a count differing from the shape's by more than `0.1 %` after the head is attached — the pair is not matched and "at matched parameters" may not be written.
**price:** `1.524 s`/cell `[FITTED]` · **mechanism:** D-1, Ruling 3, R-SKY · **deliverable:** `results/arena/softmax_d1.jsonl` · **evenings:** 0.5

### S-31 — Two ChaCAL arms and the sink-token arm `[ledger C-CHD, C-CHP, C-CHS]`
phase 3 · prereq S-20, S-21 · beside S-30, S-32..S-35
**build:** **ChaCAL-diag** — the shape with `𝒜 = ∅`, diagonal kept (the identity half of B-E1 is against this arm only); **ChaCAL-published** — `(1 − γ)A(I − γA_s)^{-1}V` with the diagonal removed inside the inverse (`fagnou-2024-chacal` Eq. 5, `[V-fetched]` by one HTML read; re-read against the PDF before typesetting), regime N inside a regime-S read, sub-stochastic (`RUN[J]` row sums `0.400 … 0.765`), the planted negative of the bitwise half; **ChaCAL-with-sink-token** — a column sink standing in for the boundary rows, the C2 kill's control. The "same `γ`" is the shape's trained `γ̂`, so the pair is sequential and the order is stated in the manifest (M-2 in potential form). Struck: "ChaCAL must emit chance" — a control that must fail by construction (V-2).
**PASS:** three FOUND arms, `diag_convention` declared on each. **KILL (K-E1, the expensive death):** ChaCAL-with-sink within `MDE₈` of the realised paired sd on `≥ 6/8` seeds **and** residual within `2×` ⇒ component (e) is a parameterisation of a column sink; escalate to `N = 16` (S-64), never "within TOST" at `N = 8` (M-13); what survives is a property paper about a published operator (`design_falsify.md` §6.2).
**price:** `1.680 s`/cell each `[FITTED + RUN]` · **mechanism:** V-3, P-10, V-24, M-13, R-SKY · **deliverable:** `results/arena/chacal_{diag,pub,sink}.jsonl` · **evenings:** 1.5

### S-32 — The InfSA-style Neumann read at `K = 16`, no boundaries `[ledger C-INF]`
phase 3 · prereq S-12, S-41 · beside S-30, S-31, S-33..S-35
**build:** the truncated read `Σ_{k≤16}(γP)^k V` with its printed `δ_Π = γ^{17}` and `δ·‖V‖_∞` (`roffo-2026-infsa` owns the reading; `[V]`); the exactness contrast `shape − Neumann-16` is creditable (S-13).
**PASS:** one FOUND cell with `route = neumann_K`, `K = 16`, `delta_vec` filled. **KILL:** the exact solve not within `δ·‖V‖_∞` of the `K = 16` read on any of the cell's draws — either the arm is not row-stochastic on that cell (S-42) or the certificate is in the wrong units (V-17).
**price:** `≈ 5.1 s`/cell (DERIVED from the `25.409 ms` hop microbenchmark, `sec_cost.md` §4.x.3) · **mechanism:** D-1, L-CERT, V-17 · **deliverable:** `results/arena/infsa_k16.jsonl` · **evenings:** 0.5

### S-33 — The cached-mixture arm on `ΔP` and `ΔV` plants `[ledger C-CACHE, P3.3, K-G1]`
phase 3 · prereq S-17, S-23 · beside S-30..S-32, S-34, S-35
**build:** `O_cached = P̂_base(I − γP̂_base)^{-1}V_int` — `P̂` frozen from the un-intervened context, values from the intervened one — the operational form of a cached successor representation (`momennejad-2017-sr`, `russek-2017-predictive`, mechanism `[U]`). It must fail on `ΔP` plants and pass on `ΔV` plants, or the interventional channel is not needed.
**PASS:** on `ΔV` plants within one seed sd of the shape; on `ΔP` plants worse by more than one seed sd on `≥ 6/8`. **KILL (K-G1):** within one seed sd on `ΔP` plants on `≥ 6/8` ⇒ the re-solve is a per-row control wearing a name (the `pivot_signed`-was-`pivot_unsigned` precedent, `READ workdonenew.md:379`).
**price:** `1.680 s`/cell · **mechanism:** D-2, C5-type, P-7 · **deliverable:** `results/arena/cached_mixture.jsonl` · **evenings:** 1

### S-34 — The argmin controls: 0-hop MLP, 1-hop softmax, majority, random, predict-the-mean `[ledger C-0H..C-MEAN]`
phase 3 · prereq S-26 · beside S-30..S-33, S-35
**build:** five arms with `kind`s, McNemar paired on identical draws; `random` reads `1/m` and `majority` reads `max_a π̂(a*)` by construction (declared V-3, printed as the floor row).
**PASS:** five FOUND cells; the two by-construction arms read their formula within CP. **KILL:** as S-26.
**price:** `≤ 1.524 s`/cell · **mechanism:** C8, V-10, V-3 · **deliverable:** `results/arena/argmin_controls.jsonl` · **evenings:** 0.5

### S-35 — The three skylines at unmatched depth / width / decode, settings fixed before the arena `[ledger C-SKY-D, K-1]`
phase 3 · prereq S-12 · beside S-30..S-34
**build:** the `⌊log₂ t*⌋ + 2` softmax stack (`3 / 5 / 7` at `t* = 2 / 8 / 32`) — chosen **by analogy** with `sanford-2024-logdepth` Thm 4.2, whose `hop_k` task is not the committor (the `hop_k → committor` reduction is NOT FOUND, `sweep_expressivity.md` §3.5); the wide constant-depth stack (`yehudai-2025-depthwidth`) at width `= n_nodes`, which at `d_model = 16 < s = 64` has no matched-parameter instance and the table says so; the chain-of-thought decoder at `t*` steps (`merrill-2024-cot`). `wang-2024-incontext-td` and `xie-2026-softmax-rl` show the deeper stack computes the same resolvent by iteration; "beats softmax" is banned (R-SKY, `READ CEQ_V16_CONTRACT.md:209`).
**PASS:** the depth-5 arm FOUND at `t* = 8`; the other two carried as `NOT MEASURED` with their fixed settings in the manifest. **KILL:** a depth chosen after the arena, or a skyline sentence of the form "cannot" outside the skyline table (`sec_obstructions.md` §5.7).
**price:** `≈ 7.6 s`/cell at depth 5 `[ASSUMED linear in depth]`; the others `NOT MEASURED` · **mechanism:** D-1, R-SKY, P-10, V-25, D-7 · **deliverable:** `results/arena/skyline_depth5.jsonl` + `docs/apparatus/SKYLINE_SETTINGS.md` · **evenings:** 1

## 5. Phase 4 — certificates and guards

### S-40 — Z-EX the exact route: `δ = 0`, the only route the committor head ships on `[ledger Z-EX, Z-COM]`
phase 4 · prereq S-20 · beside S-41..S-44
**build:** `solve_triangular` for `z`, `O` at any `γ < 1` and for the committor at `γ = 1` on the arm's triangular `Q̂`; determinism as **three outcomes** — `solve_triangular` forward and backward bitwise over 8 repeats under `use_deterministic_algorithms(True)` on the certified card (`RUN[NEPTUNE]` `0.0`, no torch-documented guarantee); `cumsum` must **raise**; a run reporting "pass" on an operator that raised is refused (V-16). The committor head carries **no** truncation certificate: the Neumann `δ` is `∞` at `γ = 1` and the `‖Q‖_∞` substitute reads `19.2` at `K = 4` against a true error `2.5e-03` (`RUN[MARS]`); the Perron-weighted certificate needs a weight `w` that is `NOT MEASURED` (S-72). The H2 threshold form `max_k q̂^{(k)} + δ‖V‖_∞ ≤ δ_thr` certifies the solve, never the model (`P̂ ≠ P_env` has no certificate; no move is "admitted" by it, V-17).
**PASS:** `1.8e-15` vs dense inverse; three-outcome determinism table filled; `route = solve_triangular` on every committor cell. **KILL:** a committor cell journalled with `route = neumann_K` (no certificate exists for it).
**price:** 0 (the microbenchmark is NEPTUNE's; the one code edit owed — `determinism_at_64`'s fourth quantity in `scripts/k_cert.py:579` — is the author's `[ledger C-1]`) · **mechanism:** M-8, V-16, L-CERT, V-17 · **deliverable:** `docs/apparatus/CERT_EXACT_ROUTE.md` · **evenings:** 0.5

### S-41 — Z-NEU the Neumann certificate in vector units, with the convergent plant `[ledger Z-NEU, B-I, P2.8]`
phase 4 · prereq S-01 · beside S-40, S-42..S-44
**build:** for `P ≥ 0` row-stochastic incl. absorbing rows and `γ < 1`: the matrix residual **equals** `γ^{K+1}/(1 − γ)` bare and `γ^{K+1}` under `Π_γ` (`RUN` this session `err = 0.540000 = bound` at `γ = 0.6, K = 2` with the sink/goal/constraint rows; `RUN[MARS]` `err − bound ∈ [4.4e-16, 6.7e-16]`) — attained, hence **definitional** on the class (declared V-3; `meyer-2000-matrix` owns the tail). The bind is carried by the plant and the units: rows scaled to `1.5` at `γ = 0.6, K = 2` read `7.290` vs `0.540` with `ρ(γP) = 0.90` — a **convergent** series (`RUN` this session; the coordinator's `γ = 0.7` plant `119.37` vs `1.143` is a divergent partial sum, `ρ = 1.05`, and is labelled so); the vector statement `max|z_solve − z_K| ≤ δ·‖V‖_∞` with `‖V‖_∞` **printed** (the `≈ 7.6e-05` of `sec_cost.md` §4.x.3 carried `‖V‖_∞ ≈ 5 [ASSUMED]`) and `1/(1 − γ̂)` beside every `δ` (the record's `1/(1 − â_max)` was undefined at all eight R1 seeds, `READ V15_R1.md:56`).
**PASS:** equality to `1e-15` at `K ∈ {1, 2, 4, 8, 16}`; the plant fails by `≥ 10×`; `V_inf` and `one_over_1mg` present on every Neumann cell. **KILL (K-I):** the vector bound exceeded on any of `1,024` drawn cells (`≈ 2.6 s` at `n = 2048`), or `δ·‖V‖_∞ ≥ sd(label)` (uninformative), or the row-sum identity presented as evidence about `P`.
**price:** `≈ 0.04 GPU-min` for the 1,024 passes · **mechanism:** V-3, V-10, V-17, V-24, L-CERT · **deliverable:** `results/certs/z_neu.jsonl` · **evenings:** 0.5

### S-42 — Z-BETA the row-sum guard on trained cells: `‖P̂‖_∞ > 1 ⇒` no certificate printed `[ledger X-12, Z-BETA]`
phase 4 · prereq S-62 (first trained cells) · beside S-40, S-41, S-43, S-44
**build:** census lines 12–13 of §A.7 on every trained cell: the measured support of `β̂` and of `rowsum(P̂)`; at `β̂ ≠ 1` rows sum `1.31 … 10.29` (`READ V16_ARM_SMPRIME.md:28-32`) and no row-stochastic theorem applies; the certificate field is then written `refused`, not a number.
**PASS:** every cell with `max rowsum > 1 + 1e-9` has `delta_vec = refused`. **KILL:** a `δ` printed on such a cell (V-25: a theorem whose hypothesis the cell does not satisfy).
**price:** 0 · **mechanism:** V-25, V-3 · **deliverable:** `docs/apparatus/CERT_ROWSUM_GUARD.md` · **evenings:** 0.5

### S-43 — Z-F1 the mask certificate carries the `1/(1 − γ)` amplification; dormant until a mask ships `[ledger Z-F1, K-I, P5.2]`
phase 4 · prereq S-41 · beside S-40, S-42, S-44
**build:** for a sparsified `P_m` with dropped row mass `ε`, `‖O_full − O_mask‖_∞ ≤ ε‖V‖_∞/(1 − γ) + δ_Π‖V‖_∞` — the resolvent amplifies dropped mass; `RUN` this session on the `s = 3` closed-form instance (`γ = 0.9, ε = 0.1`): `0.5263` against the naive `0.1`, `5.26×` violated, while `ε/(1 − γ) = 1.000` holds. A random `s = 32` draw did **not** violate the naive bound (`RUN[J]`), which is why the plant is the closed-form instance and not a random draw (V-10). The exact solve on an F1 mask is refused (fill-in: a dropped tile `(k,l)` with `(k,m), (m,l)` kept makes `M^{-1}`'s `(k,l)` block non-zero).
**PASS:** the `s = 3` instance reproduces; the F1 line in every mask cell carries `1/(1 − γ)`. **KILL:** a naive union bound printed anywhere; or, once a mask ships, exceedance on any of `1,024` draws.
**price:** 0 now; dormant (no mask exists; at `s = 64` the exact solve is cheaper than one hop, `2.514 < 3.001 ms`, `RUN[NEPTUNE]`) · **mechanism:** V-10, L-CERT, V-17 · **deliverable:** `docs/apparatus/CERT_F1_MASK.md` · **evenings:** 0.5

### S-44 — Z-F0 segmentation dividend in pair-count units, a corpus property `[ledger Z-F0]`
phase 4 · prereq S-27 · beside S-40..S-43
**build:** `D = s(s+1)/Σ_m L_m(L_m+1)` — BED-M reads `31.04×` from the live-pair fraction `0.0322` (`READ V16_ARM_SMPRIME.md:518-522`; the source line's pair counts disagree by `2×`, flagged, the fraction is consistent with the printed positives); `s²/ΣL_m²` would read `58.5` (V-17). `1×` on the softmax corner; a dividend carried from one bed to another is V-22.
**PASS:** the dividend is printed with the corpus named and the segment-length distribution beside it. **KILL:** a dividend quoted for BED-S without its census line 14 (`0 of N` zero gates on the softmax corner ⇒ `1×`).
**price:** 0 · **mechanism:** V-17, V-22, L-CERT · **deliverable:** `docs/apparatus/CERT_F0_SEGMENTATION.md` · **evenings:** 0.5

## 6. Phase 5 — the contained beds (instrument readings, never capability numbers)

### S-50 — BED-M as containment: the parity bind and the truncation-law must-fire
phase 5 · prereq S-20 · beside S-51..S-53
**build:** BED-M scores a scalar `z*_{s−1}` at one position (`READ scale/negation_scope.py:286-304`) — the single-location class (D-1) — and its label is the last row of corner 3's own resolvent (I2 `0.0`, D-2), so `shape − corner-3` is VOID by registration; BED-M stays as the host of S-20's parity bind and of the truncation-law must-fire `√((t*−k)/t*)` at `k = 0, 1, 2` (`RUN[VENUS]` `1.000058 / 0.712039 / 0.0` at `t* = 2`). `floor₁ = 0.7071067811865476` is a one-hop capability threshold violated by `13 of 40` banked cells (`RUN[WATSON]`; `READ V20_R15_THEORY_TABLE.md:69-91`) and is never printed as a floor.
**PASS:** the two readings journalled with `void_contrasts = [shape − corner3, shape − softmax]`. **KILL:** any BED-M cell carrying a capability number for the shape.
**price:** `≈ 34 s` for one pair if the truncation ladder is re-run · **mechanism:** D-1, D-2, C15 · **deliverable:** `results/bedm/containment.jsonl` · **evenings:** 0.5

### S-51 — BED-K: the first cell of the arm's shape ever, against `hard_delay_attention` `[ledger P3.4, C-BEDK]`
phase 5 · prereq S-01 · beside S-50, S-52, S-53
**build:** `bed_k.build_delay` (`READ ceq/beds/bed_k.py:194-234`), whose only two callers under `scripts/` consume the kernel and never the bed's data (`READ V20_R15_THEORY_TABLE.md:304-306`; `0` cells of BED-K's shape have ever run, `READ V20_R15_LEAP_LEDGER.md:323`); the native control `hard_delay_attention` reproduces the delay to `(n−1)e^{−45}` (`READ ceq/beds/bed_k.py:311-336`); the Hankel ceiling `err₁ = 0.9746794345 = √(1 − 1/20)` at `d = 20` (DERIVED). The shape with `γ > 0` has no delay advantage by theorem (`V15Kernel.first_order_cannot_delay`); the reading decides only that the shape is not worse than one head at a fixed offset (R-SKY).
**PASS:** shape and `hard_delay_attention` both within `MDE₈` of each other on the delay bed at `d = 4`. **KILL:** the shape worse than the native control by more than `MDE₈` on `≥ 6/8` — a fixed-offset head the resolvent cannot represent at matched parameters, filed as a negative.
**price:** `≈ 34 s` · **mechanism:** V-3, R-SKY, V-25 · **deliverable:** `results/bedk/first_cell.jsonl` · **evenings:** 1

### S-52 ★ — The capped run at seeds 2, 3, 7 on BED-M that the record priced three times and never took
phase 5 · prereq — · beside everything
**build:** the `≈ 6 GPU-s` capped run at the three NO-READING seeds (`1.113403 / 1.139404 / 1.152430`, `â_max 20.31 / 49.66 / 285.07`, `READ V15_R1.md:177-188`) with the magnitude capped at `1.0`, to settle Q3/W3's causal direction (priced at it.7, it.8, it.9 and it.35, taken zero times, `READ V20_R15_IT35_JUPITER.md:157-159`). Prediction and counter filed first (S-60): *prediction* — capped, all three seeds cross `floor₁` (the divergence was the cause); *counter* — none crosses (the non-learning is upstream of the magnitude); SPLIT `1–2 of 3`.
**PASS/KILL:** the reading itself is the deliverable; the sign of the miss goes to the calibration column (S-63).
**price:** `≈ 0.10 GPU-min` (three `1.884 s` capped cells, `READ V20_R15_THEORY_TABLE.md:176`) · **mechanism:** M-6, D-7, P-8 · **deliverable:** `results/bedm/capped_seeds_2_3_7.jsonl` · **evenings:** 0.5

### S-53 — BED-1 as the oracle cross-check instrument, never the shape's substrate
phase 5 · prereq S-25 · beside S-50..S-52
**build:** BED-1's chain is undirected (`SymmSupport`, `ρ(Q) = 0.9408612510154677`, `Q^{11} ≠ 0`, `RUN[coord]`), so no causal `P̂` equals it (`oracle_ne_resolvent` for `StrictlyLower`; `lowerTriangular_ne_symmSupport` is the regime-S target); it serves S-25's identity, the Kirchhoff second route at `K = 2`, and the `q = ½` isocommittor guard vocabulary (`READ ceq/beds/bed_1.py:5-8, :382`); it is also the reroute target of S-12's kill.
**PASS:** the three instrument readings journalled; `cells = 0` for any shape arm on it unless S-12 rerouted. **KILL:** a shape cell on BED-1 filed as a capability reading.
**price:** 0 · **mechanism:** D-2, V-3 · **deliverable:** `docs/beds/BED_1_ROLE.md` · **evenings:** 0.5

## 7. Phase 6 — the reading protocol (predictions hashed before the first cell; `N = 8` in one lane; adjudicator before data; verdict appended, prediction untouched)

### S-60 — File the predictions and counters, hash the file, and record the hash before the first arena cell
phase 6 · prereq S-12, S-13, S-16, S-35 · beside S-61
**build:** one file with the five bets of `bind_ledger.md` §6, each with prediction, counter of equal specificity, SPLIT band and deciding statistic: **A** `cos_shape − cos_softmax > MDE₈(cos)` / `≤` / none / paired `t` on the cosine; **B** `CP_upper(err_shape) < floor_exact(k = 1)` / `CP_lower ≥` / the straddle / the interval vs the floor; **C** `γ̂` MOVED on `≥ 6/8` under the boundary null / PINNED on `≥ 6/8` / `3/8–5/8` / `Λ` per seed; **D** depth-5 skyline within `MDE₈` of the shape's argmin accuracy / short by `> MDE₈`, sign logged / none / `acc_shape − acc_sky`; **E** `r_softmax / r_shape ≥ 10` at marginal NRMSE within `MDE₈` / `≤ 2` / `(2, 10)` / the ratio paired by seed and draw. Plus S-52's bet, and the (j) LM `γ` row left blank until an LM cell runs behind the author's Kaggle yes. Under the record's calibration (`7 of 8` optimistic, `p = 0.0352`, `READ V16_CALIBRATION.md:96-100`) the counter is the point estimate (D-CALIB-1); no shrink factor (D-CALIB-3). The file's sha256 is written into `results/arena/PREDICTIONS.sha256` **before** any S-62 cell exists, and every arena cell's manifest carries `predictions_hash`.
**PASS:** hash recorded; every bet has a counter; `MDE₈` cells reference the realised sd column to be filled by S-62. **KILL:** a bet without a counter (D-7), or a hash recorded after the first cell's timestamp (M-2).
**price:** 0 · **mechanism:** D-7, L-SIGN, M-2, M-7 · **deliverable:** `docs/apparatus/PREDICTIONS_R16.md` + `results/arena/PREDICTIONS.sha256` · **evenings:** 1

### S-61 ★ — Write the adjudicator specification before data (extends `verdict()`; includes the `γ`-pinning LR instrument) `[ledger P4.3, J-1]`
phase 6 · prereq S-01, S-16 · beside S-60
**build:** a specification for the verdict instrument that reads **only** the journals: refuses below `REQUIRED_SEEDS = 8` distinct seeds (`READ scale/it11_verdict.py:59, :192`) and deduplicates by seed (bit-identical duplicate rows exist, `:27-38`); reads the thread lane from the journal never from the machine (`threads` field; cross-thread drift `2.345e-3`, `:135-136`; a margin below `≈ 4.7e-3` is indefensible, `READ MISTAKES.md:1093-1100`); computes the realised paired sd from the first eight seeds and fills the `MDE₈` column by the same guarded bisection that reproduces `0.039827` at sd `0.034451` (`RUN[VENUS]`, `RUN[MARS]` `0.0398266`); scores each bet of S-60 as PASS / KILL / SPLIT and refuses any contrast smaller than the `n = 8` cell of the realised sd (M-3); checks `predictions_hash` on every cell against `PREDICTIONS.sha256` and **refuses to run on a mismatch**; never writes to the prediction file. The `γ`-pinning instrument inside it: `Λ = 2[LL(γ̂) − LL(γ ≡ 0)]` on the declared held-out set; with `γ ∈ [0, 1)` the null is on a boundary and the `95 %` point is `½χ²₀ + ½χ²₁ = 2.7055` (`RUN` this session `chi2.ppf(0.90, 1) = 2.7055`), not `3.8415`; MOVED at `Λ > ln n_eval = 8.318` (`RUN`); the interval verdict between; the `|γ̂| < 0.05` rule deleted (two verdicts for one cell, M-20); the mirror kill `γ̂ > 0.99` on `≥ 6/8` prints `1/(1 − γ̂)` beside every `δ`; the ablation `(I − γ̂P̂)^{-1} → I` at trained weights must move NRMSE by more than one seed sd. The boundary-null citation is **owed** (not in `references.bib`; the paper says so).
**PASS:** the spec's own plants — a journal with 7 seeds returns INSUFFICIENT; a duplicated seed is counted once; a tampered prediction file halts the run. **KILL:** any branch that computes a verdict on a standard nobody registered.
**price:** 0 · **mechanism:** M-9, V-5, M-10, M-16, M-3, M-2, M-20, V-17, V-9 · **deliverable:** `docs/apparatus/ADJUDICATOR_SPEC.md` · **evenings:** 1

### S-62 — The eight-seed arena at `t* = 8, m = 8, K = 2`, seven arms, one thread lane `[ledger P3.1, S-4]`
phase 6 · prereq S-12 (admitted), S-21..S-28, S-30..S-35, S-60, S-61 · beside —
**build:** one invocation, `device = cuda`, `threads = 8`, `steps = 150`, `n_train = 2048`, `n_eval = 4096`, `s = 64`, `d_model = 16`, the flag regime journalled on the header (`deterministic_algorithms` and `warn_only` as `V17_R4_RETAKE.md:14-16` did); seeds `0–7`, deduplicated; arms: shape, softmax-d1, ChaCAL-diag, ChaCAL-published, ChaCAL-sink, InfSA-16, cached-mixture (S-33 on the `Δz` lane) plus the depth-5 skyline (S-35); every cell carries the S-01 manifest, `predictions_hash`, `void_contrasts`, and the census block. The realised paired sd of the first eight seeds fixes `MDE₈` (M-3); the identical-seed floor is re-read (`δ_nrmse = 0.0` on `6/6` pairs, `RUN results/v17k_r4_floor.jsonl` by WATSON).
**PASS:** `8/8` seeds FOUND per arm; header flag regime present; `MDE₈` filled. **KILL (K-P):** realised sd `> 2.18×` the pilot's `0.050146` (the M-3 precedent, `0.109199 → MDE 0.126238`) — reprice `N` before any bet is scored; nothing in S-60 is falsifiable at `N = 8` and the paper files the bed and no capability number.
**price:** `7 × 8 × 1.680 + 4.0 ≈ 98 s ≈ 1.6 GPU-min` `[FITTED + RUN]` + `≈ 61 s` for the skyline `[ASSUMED]` + `≈ 41 s` for InfSA-16 (DERIVED) ≈ **3.3 GPU-min** · **mechanism:** M-10, M-16, V-26, M-3, Ruling 1 · **deliverable:** `results/arena/r16_t8_m8_k2.jsonl` · **evenings:** 1

### S-63 — Record the verdict append-only; append the calibration row; do not edit the prediction
phase 6 · prereq S-62 · beside —
**build:** the adjudicator's output appended to `results/arena/VERDICTS.jsonl` with `predictions_hash`, per bet PASS / KILL / SPLIT and the deciding number with its interval; a superseded cell gets a `supersede` marker, never a deletion (L-G2, `READ V17K_RULINGS.md:62-64`); the calibration row (checked / wrong / sign) appended to the cross-round column whether or not it flatters (D-CALIB-5, `READ V16_CALIBRATION.md:142-203`); the sign of every miss logged (`7 of 8` optimistic was the record's own count).
**PASS:** `VERDICTS.jsonl` line present; `PREDICTIONS_R16.md` byte-identical to its hash. **KILL:** any edit to the prediction file after S-60's hash (M-2), or a verdict filed without its bet id.
**price:** 0 · **mechanism:** M-2, D-7, P-3, L-G2 · **deliverable:** `results/arena/VERDICTS.jsonl` + the calibration column · **evenings:** 0.5

### S-64 — The E1 escalation to `N = 16`, only under the rule `[ledger P4.1, K-E1]`
phase 6 · prereq S-63 · beside S-65
**build:** if ChaCAL-with-sink is within `MDE₈` of the shape on `≥ 6/8` **and** the residual ratio is within `2×`, the pair is escalated to `N = 16` (seeds `8–15`, same lane, same hash); `MDE₁₆` from the `n = 16` column (`0.025820` at sd `0.034451`, `RUN[VENUS]`). Never "within TOST" at `N = 8` — at `N = 8` two bit-identical arms return NO VERDICT (M-13, `READ MISTAKES.md:1210-1247`).
**PASS:** separated at `N = 16` by more than `MDE₁₆`. **KILL:** not separated at `N = 16` — component (e) is a parameterisation of a column sink; the paper's sentence is `design_falsify.md` §6.2's.
**price:** `≈ 34 s` × 2 arms (8 new seeds each) ≈ `1.1 GPU-min` · **mechanism:** M-13, M-7, D-7 · **deliverable:** appended to `results/arena/r16_t8_m8_k2.jsonl` · **evenings:** 0.5

### S-65 — The TOST parity half on `𝒜 = ∅` only: paired `N = 36` or two-sample `N = 70` `[ledger P4.4]`
phase 6 · prereq S-63 · beside S-64
**build:** the only contrast TOST decides is shape at `γ̂` vs ChaCAL-diag at the same `γ̂` with `𝒜 = ∅` (`schuirmann-1987-tost`); margin `0.5σ` fixed here; two-sample power `0.80` first at `N = 70` (`0.7975` at `69`, `0.8073` at `70`, `RUN[VENUS]`, `RUN[MARS]`); paired power `0.80` at `N = 36` if `sd_d = σ` (Monte Carlo `200,000` draws, `RUN[MARS]`); the realised `sd_d` chooses. Sequential (anytime-valid) TOST is NOT FOUND at `[V]` (`sweep_methods.md` S8), so fixed-`N` Schuirmann stands.
**PASS:** the 90 % CI inside `±0.5σ`. **KILL:** not inside — parity at `γ̂ > 0` is not claimed; only the `γ = 0` identity bind (S-20) licenses "contains softmax".
**price:** `36 × 3.204 + 4 ≈ 119 s` / `70 × 3.204 + 4 ≈ 228 s` (DERIVED) ≈ `2.0 / 3.8 GPU-min` · **mechanism:** M-13, M-9, M-2 · **deliverable:** `results/arena/tost_parity.jsonl` · **evenings:** 1

### S-66 — The `s`-sweep of the cost law with `synchronize()` and randomised order `[ledger P5.1, K-9]`
phase 6 · prereq S-62 · beside S-64, S-65
**build:** `S ∈ {64, 256, 1024, 4096}` with `n` declared per `s` (at `n = 2048, s = 4096` the explicit `[n, s, s]` operator is `≈ 137 GB` in float32 against `7.996 GiB`, DERIVED), two `torch.cuda.synchronize()` calls bracketing the timed region, randomised or blocked execution order (the harness's `secs` correlates with run order at `ρ = +0.7029`, `READ V20_R15_JOURNAL.md:53`), `N = 8`; the record's repair band `206–537 GPU-s` (`READ V20_R15_THEORY_TABLE.md:104-109`) is band only.
**PASS:** fitted exponent in `s` with a CI. **KILL (K-9):** the CI excludes `2` toward `3`; the triangular solve not beating `QKᵀ` at the `s` where the paper claims it; **any price quoted before this run**.
**price:** `NOT MEASURED — needs the --seq-len flag and the two synchronize() calls` (band `3.4–9.0 GPU-min`) · **mechanism:** D-3, P-8, M-3 · **deliverable:** `results/cost/s_sweep.jsonl` · **evenings:** 2

## 8. Phase 7 — deferred instruments (`NOT MEASURED`; none on the critical path)

### S-70 — The CSR two-stage path and the fill-in guard `[ledger B-M, P5.3, K-M]`
phase 7 · prereq S-43 · beside S-71..S-73
**build:** the do-nothing 0D-salience schedule always entered (`sharma-2026-kernels-22`; `READ THEORY.md:162-165`); on a schedule that drops tile `(k,l)` but keeps `(k,m), (m,l)` the exact solve's `(k,l)` block is **reported** non-zero and the exact route refused unless a `δ` is printed; the empty-row `NaN` and negative-index loads of `READ ceq/mz_kernel.py:13-21` stay structural; second do-nothing control `zhao-2026-structuredsparse`'s blockwise resolvent; a Mapper cover quantised to tiles with parameters fixed by the Reeb-estimator rule on a held-out relation (`carriere-2018-mapper-statistics`), never on the bed.
**KILL (K-M):** the Mapper schedule not better than both do-nothing controls by `MDE₈` at matched visited tiles in `φ`-NRMSE. **price:** `NOT MEASURED` — backward through `kernels#22` does not exist (`READ THEORY.md:223-224`) · **mechanism:** V-9, L-CERT, M-2 · **deliverable:** `docs/apparatus/CSR_TWO_STAGE.md` · **evenings:** 2

### S-71 — A digraph `β₀` instrument before any barcode is read `[ledger P5.4, T-1]`
phase 7 · prereq — · beside S-70, S-72, S-73
**build:** the row-L barcode is deleted on BED-S (`β₀(ε = 0) ∈ {1, s}` on `100 %` of softmax draws, `RUN[F]`; `beta0_interleaving` consumes point clouds, `READ ceq/certs/topological.py:476-505`). Reversal needs: a connectivity notion for a digraph (weak / path homology, `chowdhury-2017-path-homology`), a dissimilarity transform of the influence Jacobian, Turner's hypothesis stated as a predicate on the filtered object (`turner-2019-quasimetric-rips`), the threshold grid and `δ` fixed before the read, a row-permuted null at the same logit scale (M-15), non-integer or aliased `β₀` a refusal (V-16); descriptive owners cited first (`kushnareva-2021-tda-attention`, `kim-2026-topological-causal`).
**KILL:** the measured barcode inside the null's `95 %` band on `≥ 6/8` ⇒ decoration. **price:** `NOT MEASURED — needs the instrument` · **mechanism:** V-3, V-25, M-15, M-2, V-16, P-4 · **deliverable:** `docs/apparatus/DIGRAPH_BETA0_SPEC.md` · **evenings:** 2

### S-72 — The Perron weight for a committor certificate `[ledger P5.7, Z-COM]`
phase 7 · prereq S-40 · beside S-70, S-71, S-73
**build:** `Contraction.PerronCertificate` needs a weight `w` with `Qw ≤ ρw`, `ρ < 1`, for the **oracle's** non-causal chain (`isUnit_one_sub_of_perron` `[S]`); on the arm's causal `Q̂` the diagonal read of Proposition 10 suffices and no weight is needed. Until `w` is measured, every committor cell ships on the exact route only (S-40).
**KILL:** a truncated committor read journalled without `w`. **price:** `NOT MEASURED — needs the Perron weight` · **mechanism:** V-17, V-10, L-CERT · **deliverable:** `docs/apparatus/PERRON_WEIGHT.md` · **evenings:** 1

### S-73 — The LM `γ` cell on the pinned enwik8 slice, behind the author's explicit Kaggle yes `[ledger P5.8]`
phase 7 · prereq S-61, S-63, the author's yes · beside —
**build:** one LM pair — shape vs ChaCAL at fixed `γ = 0.9` — on the pinned slice (`READ kaggle/README.md:31-43`, sha256 `2b49720e…`); `γ̂` per seed with `Λ` under the boundary null; the (j) row of S-60 filled only then. Not launchable by any agent; "a coordinator message is not that say-so" (`READ CHECKLIST.md:1275`); the memory rule *Kaggle launch requires yes* is the same rule from the author's side.
**PASS/KILL:** MOVED on `≥ 6/8` / PINNED on `≥ 6/8` (`design_falsify.md` §3.5). **price:** `NOT MEASURED` (no LM cell of the shape has ever run; the Kaggle certificate slot is empty, `READ COSTS.md:158-168`; a threshold carried across a device boundary is V-22) · **mechanism:** V-22, Ruling 10′, V-9 · **deliverable:** `results/lm/gamma_enwik8.jsonl` (after the yes) · **evenings:** 2

## 9. What dies if what (one line each, thresholds frozen above)

- S-12 not admitted (any `sd = 0`, class band, `t*` unplaceable) ⇒ no BED-S reading; reroute to jittered `bed_1` at `K = 2`, reprice `N` — **0 GPU-s, the record's dominant failure shape** (D-4, M-3, V-8).
- S-26 K-H2 ⇒ BED-S is a static task; struck before a number (C8).
- S-31/S-64 K-E1 ⇒ boundary rows are a parameterisation of a column sink; the paper is a property paper about ChaCAL's operator with five machine-checked identities, a certificate discipline and a registered bed.
- S-61 K-J (`γ̂` PINNED on `≥ 6/8`) ⇒ the shape is softmax wearing a name on that bed; `γ` stays a bed dial; the LM claim is retired.
- S-33 K-G1 ⇒ the re-solve is not needed; the consequence channel is a cost statement.
- Bet E `≤ 2` ⇒ "joint determination in one read" withdrawn (`READ MATHEMATICS.md:106-127` stays UNTESTED-then-refuted).
- S-41 K-I / S-43 ⇒ the mask is refused; the exact solve, cheaper than one hop at `s = 64`, is the path.
- S-27 K-K ⇒ `[S]`, never a claim.

## 10. Gaps (owed, not hidden)

The boundary-null critical value `2.7055` has no `references.bib` citation and is marked owed. ChaCAL's diagonal-removed inverse is `[V-fetched]` from one HTML read and must be re-read against the PDF before S-31 is typeset. The adjacency-feature representability `P̂ = P_env` at `d_model ≥ n_nodes` is unmeasured. The Kirchhoff second route covers `K = 2` single-node sets only (P-4). The one code edit owed — `determinism_at_64`'s fourth quantity and the solve microbenchmark in `scripts/k_cert.py` — is the author's (`[ledger C-1]`). The depth-5 skyline price is `[ASSUMED]`; the `s`-sweep, the chunked kernel, the CSR resolvent stage, the digraph `β₀` and the Perron weight are `NOT MEASURED`. BED-S has no cell, no realised sd, no measured `t*`, no `I(X_{≤k}; a*)`; every BED-S number above is a floor formula or a design constant. Every `RUN` in this file is one numpy float64 draw at `s = 32`, seed 0, CPU: (i) `L = rng.standard_normal((32,32)); L[triu(1)] = −inf; P = softmax(L)`; sets `{0}` sink, `{5,6}` goal, `{9,10}`, `{15}` (and `{20}` for the after-query check) set to `e_i`; `Q = P[T,T]`; `N = (I − Q)^{-1}`; `q = N P[T,S] 𝟙` per set; the Neumann tail at `γ = 0.6, K = 2` on that `P` and on `1.5·P`; (ii) the `s = 3` mask instance `P = [[1,0,0],[ε,1−ε,0],[0,1,0]]`, `γ = 0.9`, `ε = 0.1`, `V = e_0`; plus `scipy.stats.chi2.ppf`. None is a statistic; none carries an interval. No code file, no git write, no Kaggle contact was made by this planet.
