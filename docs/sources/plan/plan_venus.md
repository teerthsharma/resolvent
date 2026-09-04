# PLAN — VENUS (IRENE): the predictions roadmap

*Written 2026-09-03 against HEAD `207e7b9`. The judged sections (`judge/sec_shape.md`, `sec_obstructions.md`, `proposition_ledger.md`, `sec_apparatus.md`, `bind_ledger.md`) outrank `design_falsify.md` wherever they differ; where this file's numbers differ from the falsification design, the judge's ruling is the one carried (exact zero-information floor, committor head at `γ = 1`, boundary null `2.706`, DAG substrate, BOS in its own sink set, `≈ 98 s` arena). Evidence classes: `RUN[x]` a named planet's run as its file records it (`coord`, `J`, `M`, `I`, `F`, `NEPTUNE`, `WATSON`); `READ path:line`; `CITED` by canonical `references.bib` key; `DERIVED` with steps; `ASSUMED` labelled. No number without one. Every card names its `MISTAKES.md` mechanism. Nothing here launches on Kaggle; the author's explicit yes is a node (`READ kaggle/README.md:9-11`).*

## 0. The rules every card is written under

**0.1 L-SIGN.** Every prediction below is paired with a counter-prediction of equal specificity, the two are signed (`+` = the half that flatters the shape, `−` = the half that does not), and one number decides between them. A prediction without its counter is D-7 (`READ MISTAKES.md:2037`) and is not filed.

**0.2 The calibration discount (the R11 column).** The record's only scored prediction census reads **9 checked, 9 adverse, 7 of 8 signed rows optimistic**, one-sided sign test `p = 0.0352`; Wilson 95 % on the optimism fraction `[0.5291, 0.9776]` excludes `0.5` and fixes nothing else (`READ V16_CALIBRATION.md:17-19, 96-100, 171-184` via `sec_measured.md` M.5).
The rules are the record's own (`READ V16_CALIBRATION.md:142-203`): **D-CALIB-1** the counter is the point estimate — every "KILL" line below is the median outcome, not a tail; **D-CALIB-2** a bare prediction is blocked, not discounted; **D-CALIB-3** sign, never size — no numeric shrink factor appears anywhere in this file;
**D-CALIB-4** the cheapest refutation of the optimistic half runs first — §3 fixes that order; **D-CALIB-5** the row is appended whether or not it flatters — §4 fixes the scoring rule. Mechanism: D-7, M-2 (no threshold is refitted after data: every PASS/KILL number is frozen here).

**0.3 What is not bet on.** The resolvent read (`fagnou-2024-chacal` [V] owns `(1−γ)A(I−γA)^{-1}V`, Eq. 5, Eq. 7), the absorbing-chain reading (`roffo-2026-infsa` [V]), the exact resolvent layer on a fixed graph (`gasteiger-2019-appnp` [V]), policy evaluation / the successor representation (`bellman-1957-markovian`, `dayan-1993-successor`), the fundamental matrix (`kemeny-1976-finite` [V], `grinstead-1997-probability` Thm 11.6 [U]), the rank-one re-solve (`sherman-1950-inverse-adjustment`, `hager-1989-updating`), the reach-avoid rule (`summers-2010-reach-avoid`, `vanmoffaert-2013-chebyshev`, `hsu-2023-safetyfilter`).
No bet against the three skylines (`sanford-2024-logdepth` Thm 4.2 by analogy; `yehudai-2025-depthwidth`; `merrill-2024-cot`): `wang-2024-incontext-td` / `xie-2026-softmax-rl` [V] show a softmax stack computes the same resolvent by iteration, so "beats softmax" and "beats native" are banned sentences (R-SKY, `READ CEQ_V16_CONTRACT.md:209`).
No bet on any BED-M cell as a capability: BED-M is contained (P2, `RUN[coord]` corner 3 vs `(I−A)^{-1}` max-abs `0.0`) and `shape − softmax` / `shape − skyline` on BED-S's committor head are pre-registered VOID (`judge/sec_apparatus.md` §A.1).

**0.4 Units and floors that every card inherits.** MDE at `N = 8`, paired, two-sided `α = 0.05`, power `0.80`: `0.039827` at paired sd `0.034451`, `0.126238` at the realised-sd precedent `0.109199` (`2.18×` the pilot, M-3), `0.011677` at `0.010101` (`RUN[F]` `sec_beds.md` §6.D.2; re-verified `RUN[M]` `0.0398266`). **BED-S's paired sd is NOT MEASURED**;
`MDE₈` in every card below means the `n = 8` cell of the *realised* sd once the first eight seeds fix it, and the `0.039827` figure is the record's row-2 instance, not BED-S's.
Argmin floor: the exact zero-information floor is chance, `1 − 1/m = 0.875` at `m = 8` (`RUN[J]`); the tight Fano `(ln m − ln 2)/ln(m−1) = 0.7124` at `m = 8` (`RUN[J]`, `cover-2006-elements` §2.10, theorem number [U]) only where `I(X_{≤k}; a*) > 0` is computed; the weak Fano `0.6667` is struck as a floor (it sits `0.208` below chance).
Regression floor: the exact oracle at `0.0`; `floor₁ = 0.7071067811865476` is a one-hop capability threshold, printed as such and never as a floor (C15, `READ V20_R15_JOURNAL.md:51`).
Prices are `[FITTED + RUN]` from `sec_cost.md` §4.x.8 on the certified RTX 4060 Laptop: shape cell `1.680 s`, softmax cell `1.524 s`, one bed-cell pair (8 seeds each + `4.0 s`) `≈ 34 s`, the seven-arm arena `≈ 98 s ≈ 1.6 GPU-min` (DERIVED, `judge/sec_apparatus.md` §A.11); the solve increment is a per-op floor under a `2.0×–6.6×` dispatch gap (`READ scale/m3_flops.py:101-121`, P-8); the laptop clock is not stationary (`±12 %`).

## 1. Milestone cards (fields in the required order; `★` = one of the five cheapest decisive items)

Field key on every card: **id — title** · phase · prerequisites · independent-of (D-1) · **build** · **prove** · **measure** (prediction `+`, counter `−`, the deciding number) · **PASS** / **KILL** (the KILL line is the point estimate, D-CALIB-1) · price · mechanism · deliverable · evenings.

### V-0 — Open the calibration ledger and carry the R11 column into it
- phase 0 · prerequisites none · independent-of everything
- **build** a markdown ledger with one row per bet: `bet, source ∈ {author, leap}, prediction (+), counter (−), deciding number, frozen PASS, frozen KILL, realised, verdict ∈ {HOLDS, COUNTER, SPLIT, VOID}, sign of miss, date, producer_cmd`; the R11 nine rows carried in verbatim as the first block (`READ V16_CALIBRATION.md:40-48`), each with its sign
- **prove** nothing (an instrument)
- **measure** the column's running one-sided sign test after every scored row; the column starts at `7/8`, `p = 0.0352`
- **PASS** the ledger exists with the nine rows and the five rules printed at its head
- **KILL** a row filed without its counter is refused at the file, not discounted (D-CALIB-2)
- price `0 GPU-s`
- mechanism D-7, M-2, P-1 (every row names its producer)
- deliverable `docs/CALIBRATION.md` · 1 evening.

### V-1 — File every bet of this plan, signed, in one commit before any BED-S cell exists
- phase 0 · prerequisites V-0 · independent-of the Lean evening (L-1), the bed spec (S-1)
- **build** the filing: cards V-2 to V-23 transcribed as ledger rows with `realised = —`; the SPLIT band named where one exists; the VOID condition named (a bed not admitted at V-3 voids V-5 and V-9 to V-18, D-4)
- **prove** nothing
- **measure** the count of rows with both halves signed against the count of rows: must be `24/24`
- **PASS** `24/24`, commit hash recorded in the ledger head
- **KILL** any row appended after its cell ran is struck and the cell's number is not quoted (L-FIRST, `READ CEQ_V16_CONTRACT.md:49`)
- price `0 GPU-s`
- mechanism L-SIGN, M-7 (no pre-registration hole: the VOID branch is a row, not an omission)
- deliverable `docs/PREDICTIONS.md` · 1 evening.

### ★ V-2 — Bet K: the ten `[M]` Lean targets build with zero `sorry`
- phase 0 · prerequisites none · independent-of V-3, V-6, V-7, V-8
- **build** `lake build` on the ten `[M]` targets of `judge/sec_shape.md` §4.6 rows 1–10 (`gamma_zero_is_softmax`, `bos_row_is_absorbing` at `β = 1`, `later_boundary_unreachable`, `softmax_corner_not_nilpotent`, `lower_triangular_isUnit` + `diag_one_sub_smul_pos`, `resolvent_fromBlocks` + `segmentation_blockdiag`, `cut_makes_segment_head_absorbing`, `displacement_identity`, `lowerTriangular_ne_symmSupport`, `committor_is_resolvent_read` (a)+(b-causal)), each with its shipped refusal, `#print axioms` on every declaration
- **prove** the ten statements as written there
- **measure** prediction `+`: `10/10` elaborate, axioms `[propext, Classical.choice, Quot.sound]` only; counter `−`: at least one of the two rows resting on a `[U]` Mathlib name (`Matrix.det_of_lowerTriangular`; `inv_fromBlocks_zero₂₁_of_isUnit_iff`, `judge/sec_apparatus.md` §A.8.2) does not elaborate and is demoted to `[S]`; deciding number: green count `/10`
- **PASS** `10/10`
- **KILL** `≤ 9/10` ⇒ the demoted rows are cited `[S]` and never as proved (P-11, `sec_proved.md` §2.0: the record's `#18/#19/#22 [M]` had no declarations); a `sorryAx` anywhere ⇒ the row is deleted from the paper's §2
- price `0 GPU-s`
- mechanism P-11, L-LEAN, V-25
- deliverable `lean/CEQ/Shape.lean` (author-written) + the build log under `results/lean_shape_build.txt` · 1 evening.

### ★ V-3 — Bet P: BED-S admits at the design point on the 512-draw census
- phase 1 · prerequisites the BED-S generator spec (S-1 of the proposition ledger; DAG in token order, `𝒜_sink = {0}`, goal, `K = 2` constraints all before the query, query-side row clamps, adjacency multi-hot features) · independent-of V-2, V-6, V-7, V-8
- **build** the census of `judge/sec_apparatus.md` §A.7 lines 1–11 on 512 draws at `t* = 8, m = 8, K = 2`, printed per batch: row-stochasticity incl. absorbing rows, `0 ∈ 𝒜_sink` on `100 %`, `ρ(Q̂) < 1` (a diagonal read, printed not counted), sets-before-query on `100 %`, per-coordinate label sd over the admitted query region, sink share separately, class frequencies of `a*`, discard count, reduction-disagreement fraction, corpus-alone probe `R²` on `q`, `I(s₀; a*)`
- **prove** nothing; the degeneracy lemma (no goal and no sink ⇒ `max_k q^{(k)} ≥ 1/K`) is carried as a proposition, not a plant
- **measure** prediction `+`: every label sd `> 0.05`, every class frequency in `(0.05, 0.95)`, disagreement `> 0`, probe `R² < 0.5`, sink share median `< 0.5`; counter `−`: the sink share dominates (the one draw run read `[0.362, 1.000]`, `RUN[I]`), the admitted-region sd of `q^{(0)}` reads `≤ 0.05`, or a class frequency leaves the band — the record's dominant failure shape (BED-M's first builder leaked `1/(t*+1)` at zero hops and three of five rungs aborted, `READ scale/negation_scope.py:399-415`; E4′'s label collapsed from sd `0.499989` to `0.038445` under a kill rate, `READ scale/e4_harmonic.py:425-434`); deciding number: the minimum admitted-region sd and the sink-share median
- **PASS** all eleven lines pass
- **KILL** any line at `0 %` or any sd `= 0` blocks the bed (D-4) and voids V-5, V-9 to V-18; the replacement route is §5 tree C
- price `0 GPU-s` (the oracle is `m` factorisations of `|T|³/3 ≈ 5.76e8` MACs per draw, under a second in float64, DERIVED)
- mechanism V-8, V-12, V-25, D-3, D-4, V-10 (line 3 is a gate satisfied by construction and is labelled so)
- deliverable `results/bed_s_census.jsonl` + `docs/BED_S.md` census block · 1 evening.

### V-4 — The bind battery on BED-S's real draw fails every plant at O(1)
- phase 2 · prerequisites V-3 admitted · independent-of V-5, V-6, V-7
- **build** the binds B-J, B-E1, B-E2, B-G1, B-G2, B-H1, B-H2, B-K, B-P5 of `judge/bind_ledger.md` §1, each plant entering through the front door (`bed_s.build → arm.forward → journal row → verdict()`) as a cell FOUND under `results/` with its own `kind` (V-14)
- **prove** nothing new; the identity halves are declared V-3 in the manifest
- **measure** prediction `+`: every identity half passes and every plant fails at O(1) with the counts printed, in the pattern `0.9749 / 0.9165 / 1.000 / 0.4845` (`READ workdonenewseal.md:114-122`); counter `−`: at least one plant passes the honest bar — the record's precedent is the two-branch escape that passed parity bitwise for the label itself (V-24, `READ workdonenewseal.md:124-126`); deciding number: the count of plants that fail `/` the count planted
- **PASS** every plant fails at O(1) (`≥ 0.1` in the bind's own units)
- **KILL** one passing plant strikes that bind; the bind's proposition drops to a remark until a plant with a rejection region exists
- price seconds (CPU float64; the front-door cells `≈ 1.7 s` each)
- mechanism V-24, V-14, V-3 (declared), V-16 (the identity script raises on a missing key — the `0.858` reading of `sec_refuted.md` C12)
- deliverable `results/bed_s_binds.jsonl` · 1 evening.

### V-5 — The C8 static-task kill: BED-S's argmin is not readable at zero hops
- phase 2 · prerequisites V-3 admitted · independent-of V-4, V-6, V-7
- **build** the 0-hop per-position MLP on the candidate's context row and the 1-hop softmax at matched parameters (Ruling 3 count re-printed per arm), both on the argmin label, `N = 8`, McNemar paired on identical draws (`mcnemar-1947-correlated` [V]), accuracy with Clopper–Pearson (`clopper-1934-binomial` [V])
- **prove** nothing
- **measure** prediction `+`: the 0-hop arm reads within the CP half-width of chance (`0.875` error) and the shape at `N = 8` sits outside its CP half-width by more than that width; counter `−`: the 0-hop arm's error sits within the CP half-width of the shape's, or McNemar `p > 0.05` against the 1-hop softmax — BED-S is a third static task (`READ D1.md:286-291`); deciding number: `err_0hop − err_shape` against the CP half-width at `N = 8`, and the McNemar `p`
- **PASS** `err_0hop − err_shape >` half-width and `p ≤ 0.05`
- **KILL** either fails ⇒ BED-S is struck before any capability number is quoted; route: §5 tree C
- price `2 × 8 × 1.524 + 4.0 ≈ 28 s` (FITTED)
- mechanism C8, V-10, D-5, L-FLOOR
- deliverable `results/bed_s_static_kill.jsonl` · 1 evening.

### ★ V-6 — The `≈ 6 GPU-s` capped run at seeds 2, 3, 7 the record priced three times and never took
- phase 1 · prerequisites none (BED-M, `arm_pl`, the existing harness) · independent-of every BED-S card
- **build** three `arm_pl` cells at `e3_t2`, seeds 2, 3, 7, with the magnitude cap engaged (a capped cell reads `1.884 s`, `READ V20_R15_THEORY_TABLE.md:176`), same thread lane and flag regime as `results/v17k_r4_retake.jsonl`
- **prove** nothing
- **measure** prediction `+`: the three NO-READING seeds (`1.113403 / 1.139404 / 1.152430`, `â_max 20.31 / 49.66 / 285.07`, `READ V15_R1.md:177-188`) cross the capability threshold `floor₁ = 0.7071` on `3/3` under the cap — divergence was the cause; counter `−`: `≥ 2/3` stay above `1.0` under the cap — divergence was a symptom and the cause is initialisation (MARS's Q3/W3 killer, `READ V20_R15_IT35_MARS.md:27-34`); deciding number: crossing count `/3` on the capped cells
- **PASS** `3/3` cross
- **KILL** `≤ 1/3` cross ⇒ the Q3/W3 leap-ledger cell is TERMINAL as MARS graded it and the paper's §3 says so; `2/3` is SPLIT and reported
- price `≈ 6 GPU-s` (READ `V20_R15_IT35_JUPITER.md:157-159`)
- mechanism M-21 (the diagnostic is the trained-vs-capped discrimination on the same seeds), M-6 (three cells are read as three cells, never as a verdict on the wing), P-3 (the "priced, never taken" item is closed)
- deliverable `results/v21_capped_seeds_2_3_7.jsonl` · 1 evening.

### ★ V-7 — ChaCAL-diag at `γ = 0.9` on BED-M as the solve-path smoke test, with the ChaCAL-published plant
- phase 1 · prerequisites none · independent-of every BED-S card
- **build** one ChaCAL-diag cell (the shape with `𝒜 = ∅`, diagonal kept) at fixed `γ = 0.9` on BED-M `e3_t2`, seed 0, `n = 2048, s = 64`, under `use_deterministic_algorithms(True)`, plus the B-J battery: `γ = 0` bitwise against the lane's `softmaxAttn`; `γ = 0.5` region; `β = 0` at the softmax corner; ChaCAL-published's `A_s` (diagonal removed inside the inverse, `[V-fetched]` from one HTML read, re-read against the PDF before it is typeset)
- **prove** nothing
- **measure** prediction `+`: the cell runs to 150 steps under strict mode with the forward and backward bitwise over 8 repeats (`solve_triangular`: `0.0` on both, `RUN[NEPTUNE]`), its `eval_nrmse` lands inside the softmax band `[0.941881, 0.961652]` (`RUN[WATSON]`, `results/v15_r1.jsonl`) — contained, no capability — and every plant fails at O(1) (`γ = 0.5 ⇒ 2.3002850040264393`, `RUN[coord]`; `A_s ⇒` O(1) at every `γ > 0`); counter `−`: the strict-mode backward raises (the `cumsum` precedent, `READ COSTS.md:151-154`), or the cell price exceeds `2.2 × 1.680 s` (the dispatch gap's lower end), or the `A_s` plant reads under `0.1` (ChaCAL-published is not distinguishable from ChaCAL-diag on this bed); deciding numbers: the bitwise repeat count `/8`, the cell wall-clock, the `A_s` gap
- **PASS** `8/8` bitwise, price `≤ 3.7 s`, every plant `≥ 0.1`
- **KILL** a raise ⇒ the shape is not the first strict-mode gated wing and the sentence is deleted; price `> 3.7 s` ⇒ the microbenchmark increment is the P-8 floor it was declared to be and every arena price is re-quoted as a band; `A_s` gap `< 0.1` ⇒ the ChaCAL-published arm is dropped from the skyline table as redundant
- price `≈ 1.7 s`
- mechanism V-24, V-16 (three-outcome determinism), P-8, M-8
- deliverable `results/v21_chacal_smoke.jsonl` · 1 evening.

### ★ V-8 — The `γ`-pinning LR instrument, specified with the boundary null and tested on a planted `γ`
- phase 1 · prerequisites none · independent-of every other card
- **build** the instrument spec: `Λ = 2[LL(γ̂) − LL(γ ≡ 0)]` on a declared held-out set of `n_eval = 4096`, `γ` trained on `[0, 1)`, null `½χ²₀ + ½χ²₁` with 95 % point `2.706` (`RUN[J]` `2.7055`; the boundary-null citation is owed and is not in `references.bib` — the spec says so), MOVED at `Λ > ln n_eval = 8.318`, interval verdict between; the minimum detectable departure printed; `1/(1 − γ̂)` printed beside every `δ`; the `|γ̂| < 0.05` rule deleted (two verdicts for one cell, M-20). Planted checks on synthetic data: a generator at `γ_env = 0` must read PINNED and one at `γ_env = 0.6` must read MOVED
- **prove** nothing
- **measure** prediction `+`: the two plants read PINNED and MOVED respectively on `8/8` synthetic seeds; counter `−`: the `γ_env = 0` plant reads MOVED on `≥ 1/8` (the boundary null is wrong for this parameterisation) or the `0.6` plant reads PINNED on `≥ 1/8` (the instrument cannot see a moved dial); deciding number: the two plant counts `/8`
- **PASS** `8/8` and `8/8`
- **KILL** either plant fails ⇒ Bet C (V-12) cannot be scored and is VOID until the instrument is repaired
- price `0 GPU-s` (spec) + seconds (plants on CPU)
- mechanism Ruling 10′ (`READ V17K_RULINGS.md:389-436`), V-17, V-9, M-20, V-24 (the instrument has a rejection region before it judges a cell)
- deliverable `docs/INSTRUMENT_GAMMA_LR.md` · 1 evening.

### V-9 — The identification reading `‖P̂ − P_env‖_∞` (a learnability gate, never a capability)
- phase 2 · prerequisites V-3 admitted, V-4 · independent-of V-5, V-6, V-7
- **build** the per-seed read of `‖P̂ − P_env‖_∞` at trained weights on the eight-seed pair, with the adjacency-feature representability at `d_model ≥ n_nodes` stated as `NOT MEASURED` until this reads
- **prove** nothing
- **measure** prediction `+`: `‖P̂ − P_env‖_∞ < 1e-2` on `≥ 6/8` seeds at 150 steps (the shape learns the chain the bed hands it) and `≥ 1e-3` on `≥ 6/8` (it does not copy it byte-for-byte); counter `−`: `≥ 1e-1` on `≥ 6/8` — the committor head reads the wrong chain, and Bets B and E are VOID as capability on this cell; the mirror: `< 1e-3` on `≥ 6/8` ⇒ a copy (K-D2, `judge/bind_ledger.md` §5), no capability number creditable; deciding number: the per-seed sup-norm, sorted
- **PASS** `[1e-3, 1e-2)` on `≥ 6/8`
- **KILL** `≥ 1e-1` on `≥ 6/8` ⇒ reprice (more steps, the `≈ 36 min` ladder is NOT spent yet — a second 150-step pair at `n = 4096` first, `≈ 68 s`); `< 1e-3` on `≥ 6/8` ⇒ the D-2 kill fires and the cell is a reproduction check
- price inside the `≈ 34 s` pair
- mechanism D-2, V-10 (the substrate makes the rejection region non-empty), M-3
- deliverable a column of `results/v21_bed_s_pair.jsonl` · 0 extra evenings (read off V-10's pair).

### V-10 — Bet A: the consequence field `Δz` is read jointly, and the sign column is not
- phase 2 · prerequisites V-3, V-4 · independent-of V-6, V-7, V-14
- **build** the eight-seed pair with the `Δz` head (vector readout `[n, s, d]`, `READ scale/m3_quintuple.py:368,483`, separate lane and journal because a vector label voids `PUBLISHED_SOFTMAX_8192`, `READ MISTAKES.md:701-708`) against depth-1 softmax with the same readout on byte-identical draws; metric: position-matched per-coordinate NRMSE (mean, max), field cosine over coordinates `≥ i_min` with the masked fraction printed, magnitude ratio, harmonic residual with `σ_min(I − γP_env)` and `‖I − γP_env‖_∞` printed; the sign column kept (softmax reads `0.807843` CP `[0.754044, 0.854329]`, `READ MATHEMATICS.md:396-406`); the no-change predictor's error printed beside the exact floor (`vakalis-2026-interventiongap` [V] owns that floor)
- **prove** `displacement_identity` [M] (V-2)
- **measure** prediction `+`: `cos_shape − cos_softmax > MDE₈(cos)` with the cosine's *own* realised sd, on a paired `t`/Wilcoxon; the sign column shows no separation (shape inside softmax's CP); counter `−`: `≤ MDE₈(cos)` — a per-row mixture learns the displacement coordinate-by-coordinate as well as the joint read (D-1 in vector form; `zhang-2023-cina` [V]: single-effect estimation is softmax's ground); deciding number: the paired cosine difference against its `MDE₈`; no SPLIT (two-sided on the sign of the miss)
- **PASS** `> MDE₈(cos)`
- **KILL** `≤ MDE₈(cos)` on the paired test ⇒ "consequence as a field" is a label class, not a capability; the channel keeps C6 and the `V ≡ 𝟙` plant (`≤ 1e-15`, region `0.1096 / 0.363 / 1.127`) as properties; an arm emitting `Δẑ ≡ 0` is refused, not scored (V-16)
- price `≈ 34 s`
- mechanism D-1, V-26, V-17 (cosine sd, not NRMSE sd), D-5, V-24
- deliverable `results/v21_bed_s_pair.jsonl` (the `Δz` lane) · 2 evenings (the first pair fixes the realised sd for every later card).

### V-11 — Bet B: the argmin sits below the exact restricted-view floor at `k = 1`
- phase 2 · prerequisites V-3, V-5 · independent-of V-6, V-7, V-14
- **build** the argmin head on the same pair; the floor computed at construction: `floor_exact(k = 1)` = the tight Fano at the plug-in `I(X_{≤1}; a*)` where that `I > 0`, else chance `0.875`; the two rules (`argmax q^{(0)}`, Chebyshev `argmin max_{k≥1}`) and the lexicographic form as columns with the disagreement fraction (`vanmoffaert-2013-chebyshev`, `yang-2026-lexisafe` [V])
- **prove** nothing
- **measure** prediction `+`: `CP_upper(err_shape) < floor_exact(k = 1)` at `N = 8`; counter `−`: `CP_lower(err_shape) ≥ floor_exact(k = 1)` — the joint read is no better than a one-hop window licenses; SPLIT: `CP_lower < floor ≤ CP_upper` (the straddle, the R1 precedent `[0.617075, 1.041227]` around `0.7071`, `READ V15_R1.md:51-56`); deciding number: the CP interval against the floor
- **PASS** `CP_upper < floor`
- **KILL** `CP_lower ≥ floor` ⇒ the label class stands as a registered bed with printed floors and no capability sentence; SPLIT ⇒ reprice to `N = 16` (`≈ 68 s`) before any sentence
- price inside the `≈ 34 s` pair
- mechanism L-FLOOR, V-10 (chance is the floor; weak Fano struck), M-2 (rule and coordinate fixed here), V-1 (rules as columns)
- deliverable the argmin lane of `results/v21_bed_s_pair.jsonl` · 0 extra evenings.

### V-12 — Bet C: `γ̂` is MOVED under the boundary null on the `z`/`Δz` channel
- phase 2 · prerequisites V-8 PASS, V-10 · independent-of V-6, V-7, V-14
- **build** the per-seed `γ̂` with `Λ` from the V-8 instrument, on the `z`/`Δz` channel only (the committor head is the exact solve at `γ = 1` and does not carry `γ̂`, `judge/sec_shape.md` Definition 7); the ablation `(I − γ̂P̂)^{-1} → I` at trained weights; `1/(1 − γ̂)` printed
- **prove** nothing
- **measure** prediction `+`: MOVED (`Λ > 8.318`) on `≥ 6/8` seeds and the ablation moves NRMSE by `> 1` seed sd; counter `−`: PINNED (`Λ ≤ 2.706`) on `≥ 6/8` — softmax wearing a name (the hop-2 gain sweep where no `γ` beat `0`, `READ workdonenew.md:276`; `pivot_signed` was `pivot_unsigned` wearing a name, `READ workdonenew.md:379`); SPLIT: `3/8–5/8` or interval verdicts; mirror kill: `γ̂ > 0.99` on `≥ 6/8` ⇒ the certificate factor `1/(1 − γ̂) > 100` is printed and the `z`-channel certificate is vacuous (the record's `1/(1 − â_max)` was undefined at all eight R1 seeds, `READ V15_R1.md:56`); deciding number: the MOVED count `/8` and the ablation shift in seed-sd units
- **PASS** MOVED `≥ 6/8` and ablation `> 1` sd
- **KILL** PINNED `≥ 6/8` ⇒ the horizon-dial sentence is retired to a definition; the shape on BED-S is ChaCAL-diag at `γ = 0` = softmax, and V-15 becomes unrunnable (ChaCAL at `γ̂ = 0` *is* softmax)
- price inside the pair
- mechanism Ruling 10′, V-9, M-20, V-17, P-8
- deliverable the `gamma` fields of `results/v21_bed_s_pair.jsonl` · 0 extra evenings.

### V-13 — Bet E: the harmonic residual separates the arms where the marginal error does not
- phase 2 · prerequisites V-3, V-10 · independent-of V-6, V-7, V-14
- **build** `r(q̂) = ‖(I − Q_env)q̂ − R_env𝟙‖_∞` on the full committor vector over `T` (not the `[m, K+2]` summary) and `r(ẑ)` on the `z` channel, computed with the environment's operator as a **score never a loss** (D-2), paired by seed and draw, with the unit factors printed
- **prove** nothing
- **measure** prediction `+`: `r_softmax / r_shape ≥ 10` while the per-coordinate mean `φ`-NRMSE differs by less than `MDE₈`; counter `−`: `≤ 2` — joint consistency is learnable by a per-row mixture and the sentence at `READ MATHEMATICS.md:106-127` is withdrawn; SPLIT `(2, 10)`; deciding number: the paired ratio and the marginal difference against `MDE₈`
- **PASS** ratio `≥ 10` and marginal `< MDE₈`
- **KILL** ratio `≤ 2` ⇒ "one read" is repriced as a cost statement (one solve, depth `s`), the residual stays as a diagnostic; a `≥ 10` ratio at unequal marginal error is a statement about which mode the control mislearns, not about joint consistency (`judge/sec_obstructions.md` §5.1) and is reported as such
- price inside the pair
- mechanism V-26, V-17, D-2, M-2 (the three-way branch fixed now)
- deliverable the residual columns of `results/v21_bed_s_pair.jsonl` · 0 extra evenings.

### V-14 — Bet D: the depth-5 softmax skyline reaches the shape's argmin accuracy
- phase 3 · prerequisites V-11 scored · independent-of V-15, V-16, V-17
- **build** the `⌊log₂ 8⌋ + 2 = 5`-layer softmax stack at unmatched depth (depth chosen by analogy with `sanford-2024-logdepth` Thm 4.2 [V], whose `hop_k` task is not the committor — the `hop_k → committor` reduction is NOT FOUND, `sweep_expressivity.md` §3.5), eight seeds on the same draws, width fixed (`yehudai-2025-depthwidth` [V])
- **prove** nothing
- **measure** prediction `+`: `|acc_shape − acc_sky| ≤ MDE₈` — the honest control holds and the shape's separate claims are exactness, one-read cost, and the boundary mechanism; counter `−`: the skyline short by `> MDE₈` (sign logged: a shortfall says the bed needs something the hop construction does not supply; an excess says the shape is below a fellow approximator at unmatched depth and every capability sentence carries the `Δ_sky` column); deciding number: `acc_shape − acc_sky` against `MDE₈`, no SPLIT
- **PASS** inside `MDE₈`
- **KILL** outside in either direction, sign logged; a shortfall does not license "beats softmax" (R-SKY)
- price `≈ 8 × 7.6 + 4 ≈ 65 s` (`7.6 s` per cell ASSUMED linear in depth; the first cell's wall-clock replaces the assumption)
- mechanism D-1, R-SKY, D-7, P-10 (Thm 4.2 by analogy, said so), P-8
- deliverable `results/v21_skyline_depth5.jsonl` · 1 evening.

### V-15 — K-E1: ChaCAL-with-sink-token does not reproduce the committor read (the boundary-row mechanism)
- phase 3 · prerequisites V-12 not KILLED (ChaCAL at `γ̂ = 0` is softmax), V-11 · independent-of V-14, V-16, V-17
- **build** the three ChaCAL arms at the shape's trained `γ̂` (so the pair is sequential and the order is stated, M-2 in potential form): ChaCAL-diag, ChaCAL-published, ChaCAL-with-sink-token (a column sink standing in for the row condition; `xiao-2023-attentionsinks`, `gu-2024-sinkemerges` [V]); eight seeds each on the committor head
- **prove** nothing
- **measure** prediction `+`: the shape's `φ`-NRMSE sits below ChaCAL-with-sink's by `> MDE₈` on `≥ 6/8` seeds and its residual is `> 2×` smaller — the boundary rows are the mechanism; counter `−`: within `MDE₈` on `≥ 6/8` **and** residual within `2×` — component (e) is a parameterisation of ChaCAL's sink (`zhu-2003-harmonic` [V] clamped nodes are the same rows, 2003; the two-branch escape precedent, V-24); deciding number: the paired `φ`-NRMSE difference against `MDE₈` and the residual ratio
- **PASS** separated on `≥ 6/8`
- **KILL** the counter on `≥ 6/8` ⇒ escalate once to `N = 16` (`≈ 68 s`); if it holds at `N = 16` the paper's delta collapses to "ChaCAL + certificate + Lean containment" and §5 tree B's sentence is the paper; "not separated at `N = 8`" is never reported as parity (M-13)
- price `3 × 8 × 1.680 + 4 ≈ 44 s`, escalation `≈ 68 s`
- mechanism V-24, M-13, M-7, D-7, V-3 (ChaCAL-diag identity half declared)
- deliverable `results/v21_chacal_arms.jsonl` · 1 evening.

### V-16 — K-G1: the cached-mixture control fails on `ΔP` plants and passes on `ΔV` plants
- phase 3 · prerequisites V-10 · independent-of V-14, V-15, V-17
- **build** `O_cached = P̂_base(I − γP̂_base)^{-1}V_int` with `P̂` frozen from the un-intervened context and values from the intervened one (the operational form of a cached successor representation; `momennejad-2017-sr`, `russek-2017-predictive`, mechanism [U]), on the `Δz` head; the move census prints the `ΔP`-vs-`ΔV` fraction (§A.7 line 11)
- **prove** nothing
- **measure** prediction `+`: on `ΔP` plants the cached arm's cosine sits `> 1` seed sd below the shape on `≥ 6/8` and on `ΔV` plants within `1` sd; counter `−`: within `1` seed sd on `ΔP` plants on `≥ 6/8` — the re-solve is not needed and the interventional channel is a per-row control wearing a name; deciding number: the two per-seed cosine gaps in seed-sd units
- **PASS** `ΔP` gap `> 1` sd on `≥ 6/8` and `ΔV` gap `≤ 1` sd
- **KILL** `ΔP` gap `≤ 1` sd on `≥ 6/8` ⇒ component (g) is retired; the C6 identity stays as algebra
- price `≈ 8 × 1.680 + 4 ≈ 17 s`
- mechanism D-2, C5-type, V-24
- deliverable `results/v21_cached_mixture.jsonl` · 1 evening.

### V-17 — Bet G: a candidate move costs one column, not one solve
- phase 3 · prerequisites V-7 (the solve path runs) · independent-of every other phase-3 card
- **build** the microbenchmark: after one solve at `n = 2048, s = 64, d = 16`, price `m = 8` query-side row clamps by (i) the Sherman–Morrison column route (one forward-substitution column, `s²/2` MACs, plus an `s·d` inner product) and (ii) a fresh re-solve per candidate; `torch.cuda.synchronize()` bracketed, median of 14 after 2 warm-ups, the regime of `sec_cost.md` §4.x.3
- **prove** `sherman_morrison_row` [S] (row clamp only; a token rewrite is rank `s − i` and is priced as a suffix re-solve)
- **measure** prediction `+`: route (i) costs `≤ 0.5×` one solve for `m = 8` (`m/d = 0.5` at `(8, 16)`, `RUN[M]` DERIVED); counter `−`: `≥ 8×` (a re-solve per candidate is what ships, the FLOP model's `2×–3×` understatement precedent, `READ MISTAKES.md:469-476`); SPLIT `(0.5×, 8×)`; deciding number: the measured ratio
- **PASS** `≤ 0.5×`
- **KILL** `≥ 8×` ⇒ the consequence channel is repriced as `m` solves in every table; SPLIT ⇒ the band is printed, no point
- price seconds
- mechanism M-8, P-8, M-3 (measured at the shipped geometry)
- deliverable `results/v21_move_price.json` · 1 evening.

### V-18 — The parity half by TOST: shape at `γ̂` vs ChaCAL-diag at the same `γ̂`, `𝒜 = ∅`
- phase 4 · prerequisites V-15 scored · independent-of V-14, V-16, V-17
- **build** Schuirmann's TOST (`schuirmann-1987-tost` [V]) on the only contrast it decides (`judge/sec_apparatus.md` §A.11 item 3): paired at `N = 36` if the realised `sd_d ≤ σ` (Monte Carlo `200,000` draws, `RUN[M]`: power `0.80` at `N = 36` when `sd_d = σ`), two-sample at `N = 70` otherwise (`0.7975` at `69`, `0.8073` at `70`, half-width `0.2799σ`, `RUN[F]`, `RUN[M]`); margin `0.5σ` fixed here; the thread floor `2.345e-3` checked against the margin (`READ scale/it11_verdict.py:133-137`)
- **prove** nothing
- **measure** prediction `+`: `sd_d < σ` and equivalence is declared at `N = 36`; counter `−`: pairing buys nothing (`sd_d ≈ σ`) and `N = 70` is needed; a third outcome — a *difference* detected — is reported as such and is not a kill (the identity half is bitwise by V-4); deciding number: the realised `sd_d/σ` and the 90 % CI against `±0.5σ`
- **PASS** equivalence at `N = 36`
- **KILL** none — TOST here decides a price (`≈ 119 s` vs `≈ 228 s`), never a capability; "within TOST" is never claimed at `N = 8` (M-13)
- price `36 × 3.204 + 4 ≈ 119 s` or `70 × 3.204 + 4 ≈ 228 s` (DERIVED)
- mechanism M-13, M-9, M-7 (no branch consumes TOST as a capability), M-10
- deliverable `results/v21_tost_parity.jsonl` · 1 evening.

### V-19 — Score the ledger: append every row, sign every miss, run the sign test, separate author from leap
- phase 4 · prerequisites V-2 to V-18 as scored · independent-of the deferred cards
- **build** the scoring pass of §4
- **prove** nothing
- **measure** the realised column: rows scored, rows wrong, signs; the one-sided sign test on the pooled column and on the author-only and leap-only sub-columns
- **PASS** the pass exists and every scored card has a row
- **KILL** a scored card without a row, or a row edited in place (rows are appended and superseded, never edited — L-G2)
- price `0 GPU-s`
- mechanism D-CALIB-5, D-7, P-3, L-G2
- deliverable `docs/CALIBRATION.md` (appended) · 1 evening.

### V-20 — Bet I (dormant): the certificate holds in vector units on the shipped mask
- phase 5 · prerequisites a mask on a Neumann route ships (P5.2 of the bind ledger; none exists — at `s = 64` the exact solve is cheaper than one hop, `2.514` vs `3.001 ms`, `RUN[NEPTUNE]`) · independent-of every phase-2/3 card
- **build** 1,024 forward passes at `n = 2048`: `‖O_full − O_mask‖_∞ ≤ (ε/(1 − γ̂) + δ_Π)·‖V‖_∞` with `‖V‖_∞` printed (the `≈ 7.6e-05` of `sec_cost.md` §4.x.3 carried `‖V‖_∞ ≈ 5 [ASSUMED]`), the convergent plant (rows `1.5`, `γ = 0.6`, `K = 2`: `7.29` vs `0.54`, `RUN[M]`), and the `s = 3` amplification witness (`0.5263` vs naive `0.1`, `RUN[J]`)
- **prove** `neumann_truncation_bound`, `mask_amplification` [S]
- **measure** prediction `+`: `0/1024` exceedances and `δ·‖V‖_∞ < sd(label)`; counter `−`: exceeded on `≥ 1` draw, or `δ·‖V‖_∞ ≥ sd(label)` (uninformative), or the row-sum identity presented as evidence about `P` (V-3); deciding number: the exceedance count and `δ·‖V‖_∞ / sd(label)`
- **PASS** `0/1024` and ratio `< 1`
- **KILL** one exceedance ⇒ the mask is refused (L-CERT) and the exact solve is the path
- price `≈ 2.6 s`
- mechanism V-3 (declared definitional on the class), V-10, V-17, L-CERT, D-4
- deliverable `results/v21_mask_certificate.jsonl` · 1 evening, when a mask exists.

### V-21 — K-9: the cost law in `s` has exponent `2`, measured, not asserted
- phase 5 · prerequisites the author adds the `--seq-len` flag and two `synchronize()` calls to `scripts/v15_r1.py` (the record's own priced repair, `206–537 GPU-s` band only, `READ V20_R15_THEORY_TABLE.md:104-109`; the fifth edit, randomised execution order, has never been priced) · independent-of every BED-S card
- **build** the sweep `s ∈ {64, 256, 1024, 4096}`, `N = 8`, randomised order, `n` declared per `s` (at `n = 2048, s = 4096` the explicit `[n, s, s]` operator is `≈ 137 GB` in float32, DERIVED)
- **prove** nothing
- **measure** prediction `+`: the fitted exponent's CI contains `2` and excludes `3`, and the solve increment stays inside `+25 %…+50 %` of the causal MACs; counter `−`: the CI excludes `2` toward `3` (depth `s` becomes visible past `s = 64`, `NOT MEASURED` there), or the run-order correlate survives (`ρ = +0.7029` was the record's, C17) — then no price is quoted at any `s > 64`; deciding number: the exponent CI
- **PASS** CI `∋ 2`, `∌ 3`
- **KILL** CI excludes `2` toward `3` ⇒ every long-context price is a band with direction `≤` and the chunked kernel (`yang-2024-deltanet` pattern, closed to the softmax corner by `hu-2025-ssdtheory`) is the owed route
- price `206–537 GPU-s` band
- mechanism D-3, P-8, C9, M-3
- deliverable `results/v21_s_sweep.jsonl` · 2 evenings.

### V-22 — Bet J: a learnable `γ` on a causal LM read is MOVED (behind the author's yes)
- phase 5 · prerequisites the author's explicit Kaggle yes; the enwik8 pin (`READ kaggle/README.md:31-43`) · independent-of everything
- **build** one LM pair, shape vs ChaCAL at fixed `γ = 0.9` (its own setting, `fagnou-2024-chacal` App. C [U]), eight seeds, the V-8 instrument
- **prove** nothing
- **measure** prediction `+`: MOVED on `≥ 6/8` and the perplexity delta against ChaCAL inside `MDE₈`; counter `−`: PINNED at `0` on `≥ 6/8`, consistent with ChaCAL's own LM result being worse than its baseline (`21.46` vs `20.15`, digits [U]); deciding number: the MOVED count and the perplexity delta
- **PASS** MOVED `≥ 6/8`
- **KILL** PINNED `≥ 6/8` ⇒ the LM `γ` claim is retired; `γ` stays a bed-side dial
- price `NOT MEASURED` (no LM cell of the shape has run; the Kaggle certificate slot is empty, `READ COSTS.md:158-168`; V-22 across the device boundary)
- mechanism Ruling 10′, V-9, V-22, the Kaggle-yes rule
- deliverable `results/kaggle_lm_gamma.jsonl` · 2 evenings after the yes.

### V-23 — Bet M: a Mapper cover buys a dividend over the do-nothing schedule at `s = 4096`
- phase 5 · prerequisites a Mapper-to-tile quantiser (T-2) and the resolvent stage in `kernels#22` (forward-only, `READ THEORY.md:223-224`); V-21 · independent-of everything else
- **build** the cover with parameters fixed by the Reeb-estimator rule on a held-out relation (`carriere-2018-mapper-statistics` [V]), quantised to `[B, B]` tiles, against the do-nothing 0D-salience schedule (`sharma-2026-kernels-22` [V]) and `zhao-2026-structuredsparse` [V] at matched visited tiles
- **prove** nothing
- **measure** prediction `+`: `φ`-NRMSE better than both controls by `> MDE₈` at matched visited tiles with the union certificate `(ε/(1 − γ) + δ_Π)·‖V‖_∞ < sd(label)`; counter `−`: not better than do-nothing by `MDE₈` (V-9: a repair that changes nothing is a first-class outcome), or the union `δ·‖V‖_∞ ≥ sd(label)`; deciding number: the paired `φ`-NRMSE difference and the certificate ratio
- **PASS** both
- **KILL** either ⇒ the candidate builder is the merged 0D-salience schedule and the exact dividend is segmentation's (a corpus property, V-22)
- price `NOT MEASURED — needs a chunked kernel and the resolvent stage`
- mechanism V-9, L-CERT, M-2, P-4
- deliverable `results/v21_mapper_schedule.jsonl` · 3 evenings after T-2.

## 2. The DAG in one table

| card | needs | runs beside | price | evenings |
|---|---|---|---|---|
| V-0, V-1 | — / V-0 | everything | 0 | 1 + 1 |
| V-2 ★, V-6 ★, V-7 ★, V-8 ★ | — | each other and V-3 | 0 / 6 s / 1.7 s / 0 | 1 each |
| V-3 ★ | S-1 spec | V-2, V-6, V-7, V-8 | 0 | 1 |
| V-4, V-5 | V-3 | each other, V-6, V-7 | seconds / 28 s | 1 + 1 |
| V-9 to V-13 | V-3, V-4 (V-12 also V-8; V-11 also V-5) | V-6, V-7, V-14 | one `≈ 34 s` pair | 2 (V-10 owns the pair) |
| V-14, V-15, V-16, V-17 | V-11 / V-12 + V-11 / V-10 / V-7 | each other | 65 s / 44–112 s / 17 s / seconds | 1 each |
| V-18 | V-15 | V-14, V-16, V-17 | 119–228 s | 1 |
| V-19 | all scored | deferred cards | 0 | 1 |
| V-20, V-21, V-22, V-23 | a mask / the flag / the yes / T-2 | each other | 2.6 s / band / NM / NM | 1 / 2 / 2 / 3 |

Critical path (serial): V-0 → V-1 → V-3 → V-5 → V-10 (the pair; V-9, V-11–V-13 read off it) → V-15 → V-18 → V-19 = **9 evenings**. The whole falsification programme at the design point is `17 + 98 + 119 + 2.6 s ≈ 4.0 GPU-min` (DERIVED, `judge/sec_apparatus.md` §A.11); the `≈ 36 min` ladder is spent only on a survivor of V-15.

## 3. The death order (D-CALIB-4): optimistic halves refuted cheapest-first

1. V-2 Lean (0 s) — kills the `[M]` sentence, not the shape. 2. V-3 admission (0 s) — the record's dominant failure shape (D-4, M-3, V-8); rank 1 in `design_falsify.md` §5, adopted. 3. V-8 instrument plants (0 s) — without it Bet C is VOID. 4. V-7 smoke (1.7 s) — the solve path and the ChaCAL-published plant. 5. V-6 capped run (6 s) — the record's own open item. 6. V-4 binds (seconds). 7. V-5 static-task kill (28 s) — struck before any number. 8. One pair (34 s) reads V-9 identification, V-12 `γ̂` (rank 2: `γ` pinned makes V-15 unrunnable), V-13 residual (rank 4), V-10 cosine (rank 5), V-11 argmin at once. 9. V-15 ChaCAL-with-sink (44 s, then 68 s) — rank 3, the one death that removes a *mechanism* rather than a discipline. 10. V-16 cached mixture (17 s). 11. V-17 move price (seconds). 12. V-14 skyline (65 s). 13. V-18 TOST (119–228 s). 14. V-20 certificate (dormant). 15. V-21 exponent (band). 16. V-22, V-23 (NOT MEASURED). Ranks 1–3 are not independent: V-12 KILL makes V-15 unrunnable; V-3 KILL voids 8–13.

## 4. The scoring rule (D-CALIB-5), for the author and for the leap

- **One ledger, two sources.** Every row carries `source ∈ {author, leap}`. An author row is a card above. A leap row is any prediction inside a leap output: it is filed `[LEAP-UNTESTED]` with its counter and its cheapest killer, its instance is RUN before the row is scored, and a leap output acted on before its instance runs strikes the row (the R15 kills, `READ CEQ_V20_R15_CONTRACT.md:265-273`). The author's supersession of 2026-09-03 lifts "no new primitive that is not a repair" and nothing else (`sec_state.md` S.7).
- **Verdict tokens.** `HOLDS` (the `+` half's PASS number reached), `COUNTER` (the `−` half's KILL number reached), `SPLIT` (inside a named band), `VOID` (the bed not admitted, or a prerequisite instrument failed its plants). A VOID row is unscored, not a miss (D-4); a SPLIT row is scored `wrong` with sign `+` if the prediction was the optimistic half (it was, on every card).
- **The sign column.** `+` when the realised number fell short of the prediction in the direction that would have flattered the shape; `−` when it overshot. The one-sided sign test is run on the pooled column after `≥ 8` scored rows and on the author-only and leap-only sub-columns separately; `7/8` optimistic is `p = 0.0352`, `5/6` is `p = 0.1094` (the record's own two figures, `READ V16_CALIBRATION.md:98-105`). Direction is the only output; no shrink factor is ever fitted (D-CALIB-3); a column that stays at `≥ 0.75` optimistic after `≥ 8` rows makes the counter the paper's *reported* estimate for every unscored card, not merely the point estimate for planning.
- **Append-only.** A row is never edited; a corrected reading is a new row with `supersedes` pointing at the old one (L-G2). The row is appended whether or not it flatters; a card scored `HOLDS` gets the same row shape as one scored `COUNTER`.

## 5. The outcome trees

**A. Every prediction holds (the `+` column, the half the R11 column says is less likely).** V-2 `10/10` → §2 cites ten declarations. V-3 admits → V-5 strikes nothing → the pair reads V-9 identified, V-12 MOVED `≥ 6/8`, V-13 ratio `≥ 10`, V-10 cosine `> MDE₈`, V-11 `CP_upper < floor` → V-15 separates `≥ 6/8` → V-14 inside `MDE₈` → V-17 `≤ 0.5×` → V-18 equivalence at `N = 36`.
*Licensed sentence:* "Consequence–Equilibrium Attention is ChaCAL's causal resolvent read (`fagnou-2024-chacal`) on the record's Lean-checked three-corner base with a sink, a goal and `K` constraint sets as boundary rows; on BED-S (`t* = 8, m = 8, K = 2`, `N = 8`, params re-counted per arm) it reads the reach-avoid vector with argmin error `[e]` (CP `[l, u]`) below the exact restricted-view floor `[f]`, harmonic residual ratio `[r]` `≥ 10` at equal marginal error, displacement cosine `[c]` above depth-1 softmax's by `[Δ] > MDE₈`;
ChaCAL with the same `γ̂ = [g]` (`Λ = [Λ]`, MOVED) and a sink token reads above the floor; the depth-5 skyline reaches the same argmin accuracy within `MDE₈`, so the separate advantages are exactness (`δ = 0` on the exact route), one-read cost (one solve, depth `s`, `+50 %` causal MACs) and the boundary-row mechanism." Every bracket is a number BED-S has not produced.
*Route owed:* none; the `≈ 36 min` ladder is spent (reprice upward, one pair).

**B. The median outcome (every counter is the point estimate, D-CALIB-1).** V-2 `9/10` (one `[U]` Mathlib name demotes a row to `[S]`). V-3 admits after one repair of the sink share (the fix's own risk, `RUN[I]` `[0.362, 1.000]`). V-5 does not strike.
The pair reads V-9 identified at `[1e-3, 1e-2)`, V-12 PINNED `≥ 6/8`, V-13 SPLIT `(2, 10)`, V-10 `≤ MDE₈`, V-11 SPLIT (the straddle). V-15 is unrunnable at `γ̂ = 0` and is scored VOID; V-14 inside `MDE₈`; V-17 SPLIT; V-18 `N = 70`.
*Licensed sentence:* "Consequence–Equilibrium Attention is a re-parameterisation of ChaCAL's causal resolvent read with absorbing boundary rows on the record's three-corner base. Nine of its identities are machine-checked and the tenth is `[S]` with its numeric instance; the committor head is the exact solve at `γ = 1` (identity `0.0` on BED-1's real sets);
BED-S is registered with an exact oracle at `0.0`, the exact zero-information floor `0.875`, and a printed paired sd `[sd]`. On BED-S at `N = 8` the trained `γ̂` is not distinguishable from `0` under the boundary null, depth-1 softmax matches the marginal NRMSE and the displacement cosine within `MDE₈`, and the residual ratio is SPLIT.
No capability sentence is licensed; the contributions are the identities, the certificate discipline, the bed, the controls and the negatives." *Routes owed:* **retire** the horizon-dial sentence to a definition (V-12) and the LM `γ` claim (V-22); **reprice** "one read" as a cost statement (V-13) and V-11 to `N = 16` (`≈ 68 s`) before any argmin sentence;
**retire** "consequence as a field" to a label class (V-10); **keep** `γ = 1` regime N (corner 3, exact without a dial) and the `V ≡ 𝟙` plant. This is the sentence the author reads first.

**C. The shape dies at its weakest component (V-3, before a GPU-second).** BED-S does not admit at `t* = 8`: the sink share dominates or the admitted-region sd of `q^{(0)}` reads `≤ 0.05`, or a class frequency leaves `(0.05, 0.95)`. Everything from V-5 to V-18 is VOID (unscored).
*Licensed sentence:* "BED-S at the design point does not admit; the paper files the generator specification, the census with its printed failure line, the floors and the pricing rule, and no capability number." *Routes owed, in order:* **reroute** the dial — move `t*` to `2` or the graph depth up (the DAG substrate bounds `t*` by the causal window, `judge/sec_apparatus.md` §A.2.1) and re-run the `0 GPU-s` census;
**reroute** the substrate — `bed_1.build(jitter)` with `K = 2` and `B` as goal as an *oracle cross-check only* (undirected `SymmSupport`, `ρ(Q) = 0.9409`, never the shape lane's substrate: a causal `P̂` cannot equal it, D-2), its 13 tests and 5 killed mutants in the tree (`sec_measured.md` M.7.2);
**reprice** `N` from whatever realised sd the first admitted batch shows (the `2.18×` precedent gives `MDE₈ = 0.126238`, at which nothing in phase 2 is falsifiable at `N = 8` and `N = 16` is the first honest count).
If the second death follows (V-15 counter at `N = 16`), the paper is tree B's sentence with "boundary rows are a parameterisation of ChaCAL's sink; the shape is ChaCAL with a certificate" appended, and it is written so that it is still honest in that state.

## 6. Limits (once)

Every price is `[FITTED + RUN]` at `s = 64` only (the exponent in `s` is unidentified on `40/40` banked cells until V-21), a per-op solve increment that is a floor, and a laptop clock at `±12 %`; the depth-5 price is ASSUMED linear in depth; the TOST `N = 36` is a Monte-Carlo figure at `sd_d = σ`. BED-S has no cell, no realised sd, no measured `t*`, no `I(X_{≤k}; a*)`; every `MDE₈` above is a placeholder for the realised-sd cell and every BED-S number is a floor formula or a design constant. The boundary-null critical value `2.706` carries no `references.bib` citation and is marked owed. The `hop_k → committor` reduction is NOT FOUND, so V-14's depth is by analogy. ChaCAL's diagonal convention is `[V-fetched]` by one HTML read. V-20, V-22, V-23 have no runnable killer in the tree and say so. The R11 column is the record's; nothing in it was re-scored here. The ranking in §3 is ordinal; no probability word in this file is a number. No code file, no git write, no Kaggle contact, no hardware run was made by this planet.
