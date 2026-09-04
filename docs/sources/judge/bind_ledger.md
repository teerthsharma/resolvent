# Bind ledger — one row per bind, control, census, certificate and kill

*SATURN (WATSON), 2026-09-03, HEAD `207e7b9`. Companion to `sec_apparatus.md`. Status tokens: **KEEP** (adopted as designed), **REPAIR** (adopted with the refuter's correction applied; the correction is in the "what changed" column), **KILL→applied** (the refuter's reversal was one sentence and is applied), **KILL→deleted** (reversal needs an instrument or bed that does not exist; the row is out of the apparatus and appears only here). Every row names the `MISTAKES.md` mechanism by symbol and the PLAN item that builds it. PLAN item ids are proposed for `docs/PLAN.md`: `P0.*` Lean and manifests (0 GPU-s), `P1.*` bed specification and census (0 GPU-s), `P2.*` identity binds on BED-S's real draw (seconds), `P3.*` the eight-seed arena (`≈ 1.6 GPU-min`), `P4.*` MDE / TOST / LR verdicts, `P5.*` deferred instruments (`NOT MEASURED`). Items marked `first evening` are the five the author can settle alone in one sitting (`THESIS_CORRECTIONS_2.md` §0).*

## 1. Binds

| id | object | status | what changed (refuter, evidence) | planted negatives and required O(1) failure | mechanism | PLAN item |
|---|---|---|---|---|---|---|
| B-J | parity at `γ = 0`, `torch.equal(O(0), PV)` against the lane's `softmaxAttn` | REPAIR | the "non-causal `W`" plant passes bitwise (`RUN[MARS]` `True`, `refute_theory_occvac` §2.1) — moved to B-P5; ChaCAL-published's `A_s` added as a plant (`refute_falsify_math` §0.4) | `γ = 0.5 ⇒ 2.3003`; `β = 0 ⇒ > 0.5`; `A_s ⇒` O(1) at any `γ > 0` | V-24, V-3 | P2.1 (first evening: the ChaCAL smoke test at `γ = 0.9` on BED-M runs this bind's plant) |
| B-E1 | boundary rows vs ChaCAL-diag, identity half at `𝒜 = ∅`, rejection half on downstream rows | REPAIR | identity half is against the lane's own re-implementation with `diag_convention` declared (V-3 declared); rejection half is a **weight** gap, not a support gap (`RUN[MARS]` rows changed `= [3]`); plant (iii) "non-identity absorbing row breaks `Σ_k q = 1`" struck — empty (`RUN[MARS]` `0.0`, `refute_instrument_math` row 12) and replaced by the full-`P` read at `γ < 1` (`0.1491`) | `𝒜 ≠ ∅ ⇒ ‖Π_shape − Π_ChaCAL-diag‖_∞ = O(1)` downstream; InfSA base breaks the identity half (`NOT MEASURED — needs the base wired`) | V-24, V-3, V-14 | P2.2 |
| B-E2 | conservation `q^{(0)} + Σ_k q^{(k)} + q^{(sink)} = 𝟙` on `T` | REPAIR | sink set `𝒜_sink = {0}` added (`RUN` sum `1e-15`, sink share `[0.362, 1.000]`); "drop `𝒜_0 ⇒ max_k ≥ 1/K`" plant struck (V-3 of `Σ = 1`, `refute_instrument_math` row 3) and carried as the degeneracy **proposition** (no goal, no sink ⇒ `max_k ≥ 1/K`) | drop BOS from every set ⇒ the solve **raises** (`ρ(Q) = 1.000000`, `det = 0`, V-16); a set after the query ⇒ `q = 0.0` exactly ⇒ discard | V-12, V-23, V-16 | P1.2, P2.3 |
| B-G1 | interventional re-solve: `V ≡ 𝟙 ⇒ Δz ≡ 0`; Sherman–Morrison vs re-solve; C6 identity; forward-only | REPAIR | "bitwise `0.0`" → `≤ 1e-15` (`refute_falsify_math` §3.7); the solve route is named (LU gives `8.9e-16`, `equal False`; `solve_triangular` `0.0`, `True` — `refute_instrument_math` row 10); the EMC feedback plant struck and replaced by the **triangularity** plant | Gaussian `V ⇒ max|Δz| = O(1)` (`0.1096 / 0.363 / 1.127`); non-causal `P ⇒ Δz[:i] ≠ 0` (`0.0761 / 0.0868`); two-row edit breaks rank-one by O(1), repaired by Woodbury | V-24, D-5, D-7 | P2.4 |
| B-G2 | vector-metric plants (permuted oracle, `+0.1σ`, `−Δz`) | KEEP | — | permuted oracle scores `0` ⇒ metric struck; `+0.1σ` not preferred ⇒ struck; `−Δz` indistinguishable ⇒ struck | V-26, L-14 | P2.5 |
| B-H1 | committor identity `q = (I − Q)^{-1}R𝟙` vs `bed_1.committor` on real sets (`0.0`) | REPAIR | the identity is checked on the **environment** chain (BED-1 as instrument) and separately on the arm's triangular `Q̂` (invertible by `lower_triangular_isUnit`); the script raises on a missing key (`sec_refuted.md` C12); a wrong-set plant added (`refute_theory_occvac` §2.1) | must-fire perturbation `1.04e-17 → 1e-6` (ratio `9.6e10`); wrong `𝒜_k` ⇒ `q` moves O(1); Kirchhoff `< 1e-10` at `K = 2` | D-2, V-3, V-16 | P2.6 |
| B-H2 | argmin from the oracle tensor equals the label's argmin (V-3 declared); leak plants | KEEP | — | planted leak token ⇒ probe `R² ≥ 0.99`; E4′ local-feature decoder above `FAIL_BAR` (re-derived for BED-S, V-22) | V-24, V-3 | P1.4, P2.7 |
| B-I | Neumann tail equality (definitional) + planted non-stochastic `P` + shipped-mask measurement in vector units | REPAIR, **dormant** | the `γ = 0.7` plant (`119.37 / 1.143`) is a divergent partial sum, labelled so; the `γ = 0.6` plant (`7.29 / 0.54`, convergent) is quoted; `‖V‖_∞` must be printed (the `≈ 7.6e-05` carried `≈ 5 [ASSUMED]`); dormant until row M ships a mask (`refute_instrument_occvac` §2.6, D-4) | rows `1.5` fail the bound; on the shipped mask exceedance on any of `1,024` draws; `δ·‖V‖_∞ ≥ sd(label)` ⇒ uninformative | V-3, V-10, V-17, L-CERT | P5.2 |
| B-K | `[M]` Lean targets build; corner 3 vs `(I − A)^{-1}` `0.0`; segmentation zeros by `torch.equal` | REPAIR | `[M]` grades resting on `[U]` Mathlib names are `[S]` until `lake build` is green (P-11); the `1e-300` plant struck (underflow, `refute_instrument_math` row 14) and replaced by `−30` logit; on BED-S at the softmax corner the F0 theorems are silent (`0 of N` draws) | keeping the diagonal breaks `A^s = 0`; `−30` logit leaves the block non-zero | P-11, V-25, V-2 | P0.1 (first evening) |
| B-P5 | `solve_triangular` vs dense inverse `1.8e-15` | KEEP (plant received from B-J) | — | a dense `M` with `upper=False` must disagree | M-8 | P2.1 |
| B-M | CSR fill-in guard; do-nothing schedule always entered; empty-row `NaN` and negative-index loads structural | KEEP, unpriced | — | a dropped `(k,l)` with `(k,m),(m,l)` kept ⇒ exact solve refused unless `δ` printed | V-9, L-CERT | P5.3 |
| B-C6 | the displacement identity `Δz = (I − γP')^{-1}(ΔV + γΔP z)` | KEEP | three independent RUNs `1.03e-15 / 1.2e-15 / 1.36e-15` | carried by B-G1's plants | V-24 | P2.4 |
| B-EMC | "settle then `do` = `do` then settle" with a feedback plant | **KILL→deleted** | unique fixed point for `γ < 1` on every `P` (`RUN[MARS]` `1.33e-15`, `4.4e-16` on dense cyclic `P`); the "violation" was a suffix-only solve on a non-triangular `P`; one remark survives | — | V-24, D-7, V-3 | — |
| B-L | influence-Jacobian `β₀` barcode, `ε = 0` endpoint = segmentation count | **KILL→deleted** | `β₀(ε = 0) ∈ {1, s}` on `100 %` of softmax draws; `beta0_interleaving` consumes point clouds; Turner hypothesis unchecked | — | V-3, V-25, P-4 | P5.4 (`NOT MEASURED — needs a digraph β₀ instrument`) |
| B-manifest | flipping any of `beta/qk/g/gamma/boundary_sets` by one unit moves `manifest_hash`; a missing declared field is a refusal | REPAIR | fields `diag_convention`, `committor_route`, `sink_set`, `producer_cmd`, `V_inf`, `void_contrasts` added | the one-line-drift test pattern (`READ V20_R15_WING_MANIFEST.md:148-152`) | L-2, P-1, V-16 | P0.2 (first evening) |

## 2. Controls and skylines

| id | arm | status | what changed | mechanism | PLAN item |
|---|---|---|---|---|---|
| C-SM1 | depth-1 softmax, same head, matched params (count re-printed per arm) | KEEP | Ruling 3 count per arm (`refute_instrument_occvac` §2.5) | D-1, Ruling 3 | P3.1 |
| C-CHD | ChaCAL-diag (the shape with `𝒜 = ∅`, diagonal kept) | REPAIR | split from "ChaCAL" | V-3, R-SKY | P3.1 |
| C-CHP | ChaCAL-published (`A_s` diagonal removed inside the inverse, `[V-fetched]`) | REPAIR | new arm; sub-stochastic; planted negative for B-J's bitwise half | P-10, V-24 | P3.1 |
| C-CHS | ChaCAL-with-sink-token | KEEP | the "same `γ`" is the shape's trained `γ̂`, so the pair is sequential and the order is stated (M-2 in potential form) | V-24 | P3.1 |
| C-INF | InfSA-style Neumann `K = 16`, no boundaries | KEEP | — | D-1 | P3.1 |
| C-SKY-D | `⌊log₂ t*⌋ + 2` softmax stack (`3/5/7`) | REPAIR | depth chosen by analogy with Thm 4.2; the `hop_k → committor` reduction is NOT FOUND; carried by Bet D | P-10, V-25, D-7 | P3.2 |
| C-SKY-W | wide constant-depth stack, width `= n_nodes` | REPAIR | width fixed; at `d_model = 16 < s = 64` the matched instance does not exist and the table says so | P-4 | P5.5 (`NOT MEASURED`) |
| C-SKY-C | chain-of-thought decoder, `t*` steps | REPAIR | step count fixed | P-4 | P5.5 (`NOT MEASURED`) |
| C-NAT | chunked-WY (DeltaNet / SSD), linear corner | KEEP | closed to the softmax corner by `hu-2025-ssdtheory` | R-SKY, M-8 | P5.6 (`NOT MEASURED — needs a chunk-size sweep`) |
| C-MZ | MuZero-style value head for the consequence channel | KEEP | — | D-1 | P5.5 |
| C-0H, C-1H, C-MAJ, C-RND, C-MEAN | argmin controls | KEEP | McNemar on identical draws; the C8 kill | C8, V-10 | P3.1 |
| C-CACHE | cached-mixture `O_cached = P̂_base(I − γP̂_base)^{-1}V_int` | REPAIR | given an operational definition (`refute_instrument_occvac` §2.4); Momennejad/Russek mechanism `[U]` | P-7, D-2 | P3.3 |
| C-DT | return-conditioned DT-style arm on deterministic beds | KEEP | — | D-7 | P5.5 |
| C-CH-EMIT | "ChaCAL must emit chance" | **KILL→deleted** | a control that must fail by construction (V-2) | V-2, V-10 | — |
| C-DENSE | dense resolvent at `s ∈ {256, 1024, 4096}` | REPAIR | `n` declared per `s` (`137 GB` at `n = 2048, s = 4096`) | P-8 | P5.3 |
| C-BEDK | `hard_delay_attention` on BED-K | KEEP | — | V-3 | P3.4 |

## 3. Census lines (L-DOM)

| id | line | status | what changed | mechanism | PLAN item |
|---|---|---|---|---|---|
| X-1 | row-stochastic `P_env` incl. absorbing rows | KEEP | — | V-25 | P1.2 (first evening: the BED-S census on 512 draws) |
| X-2 | `0 ∈ 𝒜_sink` | REPAIR | sink set, not goal member | V-25, V-12 | P1.2 |
| X-3 | `ρ(Q̂) = max diag < 1` | REPAIR | a diagonal read, V-10 gate, printed not counted (`RUN` `0.692660`) | V-10 | P1.2 |
| X-4 | boundary sets before the query; query in `T` | REPAIR | the query-in-`T` clause added (`refute_theory_math` §1.3) | V-8 | P1.2 |
| X-5 | label sd over the admitted query region; sink share printed | REPAIR | region restriction (`refute_instrument_occvac` §2.7); sink share separate from goal | V-8 | P1.2 |
| X-6 | argmin class frequencies in `(0.05, 0.95)`; discards | REPAIR | the inverted "argmin-unique fraction" gate replaced | V-10, D-3 | P1.2 |
| X-7 | reduction-disagreement fraction `> 0` | KEEP | — | C4, V-1 | P1.2 |
| X-8 | corpus-alone probe to `q` at order 0 `< 0.5`; planted leak `≥ 0.99` | REPAIR | clause (a) (rows of `P_env`) dropped for BED-S — the graph is the input | V-24, V-7 | P1.4 |
| X-9 | `I(s₀; a*) = 0`; untrained arm at chance | KEEP | — | V-10, D-5 | P1.3 |
| X-10 | `Var(Δz) > 0` over coordinates `≥ i_min` | REPAIR | coordinate 0 and the prefix are zero on every draw | V-8 | P1.2 |
| X-11 | move census (`ΔP` vs `ΔV`; `uᵀ𝟙 = 0`; non-negative row) | KEEP | denominator positivity is a theorem, printed as a check | V-25 | P1.2 |
| X-12 | `β̂` and `rowsum(P̂)` on trained cells | REPAIR (new) | `‖P̂‖_∞ > 1 ⇒` no certificate printed | V-25 | P3.1 |
| X-13 | `γ̂` support with LR verdict | KEEP | boundary null `2.706` | V-17 | P4.3 |
| X-14 | exact zero gate present (segmentation rows) | KEEP | `0 of N` on BED-S softmax corner ⇒ F0 theorems silent | V-25 | P1.2 |
| X-15 | obstruction vacuity inequalities printed | KEEP | `512 ≥ 64`, `384 < 544`, `64^{1/16} = 1.2968` | V-25, P-10 | P0.3 |

## 4. Certificates

| id | object | status | what changed | mechanism | PLAN item |
|---|---|---|---|---|---|
| Z-EX | exact solve, `δ = 0` | KEEP | — | M-8 | P2.1 |
| Z-NEU | Neumann `K` hops, `δ = γ^{K+1}/(1−γ)` bare / `γ^{K+1}` for `Π_γ`, vector units, `‖V‖_∞` printed, `1/(1−γ̂)` beside | REPAIR | factor under `Π_γ`; `‖V‖_∞` printed; definitional label | V-17, V-3, L-CERT | P2.8 |
| Z-F1 | F1 mask: `ε/(1−γ) + δ_Π`, vector units | **KILL→applied** | the resolvent amplifies dropped mass (`RUN` `0.5263` vs naive `0.1`, `5.26×`) | V-10, L-CERT | P5.2 (dormant until a mask ships) |
| Z-F0 | segmentation, `δ = 0`, dividend as a pair-count ratio, corpus property | REPAIR | `31.06×` is `s(s+1)/Σ L_m(L_m+1)`, not `s²/Σ L_m²`; `1×` on the softmax corner | V-17, V-22 | P0.1 |
| Z-COM | committor at `γ = 1`: no Neumann certificate; exact route only; Perron weight `NOT MEASURED` | **KILL→applied** | H1/H2 inherit no `δ`; the H2 threshold form certifies the solve, never the model | V-17, V-10, L-CERT | P5.7 (`NOT MEASURED — needs the Perron weight w`) |
| Z-COARSE | multizoom mean-pool bound for the far field | KEEP | — | L-CERT | P5.3 |
| Z-BETA | `‖P̂‖_∞ > 1 ⇒` no certificate | REPAIR (new) | — | V-25 | P3.1 |

## 5. Kills (pre-registered, thresholds frozen here)

| id | kill | status | threshold | what changed | mechanism | PLAN item |
|---|---|---|---|---|---|---|
| K-E1 | ChaCAL-with-sink matches the committor read | REPAIR | within `MDE₈` of the realised paired sd on `≥ 6/8` seeds **and** residual within `2×` ⇒ (e) is a parameterisation; escalate to `N = 16`, never "within TOST" at `N = 8` | TOST at `N = 8` has no passing branch (M-13); TOST decides only the `𝒜 = ∅` parity half | M-13, M-7, D-7 | P4.1 |
| K-E2 | label degeneracy | REPAIR | any `sd(q^{(k)}) = 0` over the admitted region; any class frequency outside `(0.05, 0.95)`; disagreement `0` | inverted gate fixed | V-8, V-12 | P1.2 |
| K-G1 | cached-mixture within one seed sd on `ΔP` plants on `≥ 6/8` | KEEP | — | — | C5-type, D-2 | P3.3 |
| K-G2 | field cosine within the cosine's own `MDE₈` of matched-depth softmax on `≥ 6/8` | REPAIR | cosine sd, not NRMSE sd (V-17) | V-17, D-1 | P4.2 |
| K-H1 | residual ratio `≤ 2` ⇒ "joint determination in one read" withdrawn; `(2, 10)` SPLIT | REPAIR | `σ_min`, `‖I − γP_env‖_∞` printed beside `r` | V-17, V-26 | P4.2 |
| K-H2 | 0-hop MLP within the CP half-width at `N = 8`, or McNemar `p > 0.05` vs 1-hop ⇒ BED-S is a static task, struck before any number | REPAIR | "the Fano floor's resolution" replaced by the CP half-width | C8, V-10 | P3.1 |
| K-D2 | `‖P̂ − P_env‖_∞ < 1e-3` on `≥ 6/8` ⇒ a copy; registered as a learnability reading | **KILL→applied** | had an empty rejection region on the undirected substrate; non-empty on the DAG substrate | D-2, V-10 | P3.1 |
| K-J | `γ̂` PINNED (`Λ ≤ 2.706`) on `≥ 6/8` ⇒ softmax wearing a name; ablation `(I − γ̂P̂)^{-1} → I` moving NRMSE by `< 1` seed sd; mirror `γ̂ > 0.99` on `≥ 6/8` ⇒ certificate vacuous on the `z`/`Δz` channel | REPAIR | `|γ̂| < 0.05` deleted; boundary null; committor head unaffected by `γ̂` | M-20, V-17, V-9 | P4.3 |
| K-I | exceedance on any of `1,024` draws; `δ·‖V‖_∞ ≥ sd(label)` | REPAIR, dormant | — | V-10, V-17 | P5.2 |
| K-K | any `[M]` target not building by the Lean milestone ⇒ `[S]` | KEEP | — | P-11, L-LEAN | P0.1 |
| K-M | Mapper schedule not better than do-nothing (and Zhao 2026 blockwise) by `MDE₈` at matched visited tiles, in `φ`-NRMSE | REPAIR | metric named; second do-nothing control added | V-9, V-17 | P5.3 |
| K-P | BED-S admission: realised sd `2.18×` the pilot ⇒ reprice `N`; `t*` unplaceable ⇒ reroute | REPAIR | F1/F2 added to the cause list at `0 GPU-s`; BED-M's `0.034451` and E4′'s `1372.50` marked V-22, not BED-S bars | D-4, M-3, V-22 | P1.2, P3.1 |
| K-9 | `s`-sweep exponent CI excludes `2` toward `3`; any price quoted before the run | KEEP | — | D-3, P-8 | P5.1 |
| K-10 | any obstruction stated without its vacuity line; `0 %` admitted on the bed | KEEP | — | V-25, P-10 | P0.3 |
| K-11 | any sentence calling the resolvent new | KEEP | — | P-7 | assembly |
| K-EMC | — | **KILL→deleted** | — | — | — |
| K-L | — | **KILL→deleted** | — | — | — |

## 6. Predictions with counters (L-SIGN, D-7) — filed before P3

| bet | prediction | counter | SPLIT band | deciding number | PLAN item |
|---|---|---|---|---|---|
| A | `cos_shape − cos_softmax > MDE₈(cos)` | `≤ MDE₈(cos)` | none (two-sided on the sign of the miss) | paired `t` on the cosine, `N = 8` | P3.1 / P4.2 |
| B | `CP_upper(err_shape) < floor_exact(k = 1)` | `CP_lower ≥ floor_exact(k = 1)` | `CP_lower < floor ≤ CP_upper` | the interval vs the floor | P3.1 / P4.2 |
| C | `γ̂` MOVED on `≥ 6/8` seeds | PINNED on `≥ 6/8` | `3/8–5/8` | `Λ` per seed | P4.3 |
| D | depth-5 skyline within `MDE₈` of the shape's argmin accuracy | short by `> MDE₈` (sign logged) | none | `acc_shape − acc_sky` | P3.2 |
| E | `r_softmax / r_shape ≥ 10` at marginal NRMSE within `MDE₈` | `≤ 2` | `(2, 10)` | the ratio, paired by seed and draw | P4.2 |
| (j) LM `γ` | no row until an LM cell runs | — | — | — | P5.8 (Kaggle: the author's explicit yes is a node) |

## 7. Tally

| status | count |
|---|---|
| KEEP | 27 |
| REPAIR | 40 |
| KILL→applied | 4 (F1 mask certificate, committor certificate, D-2 kill's substrate, the support sentence folded into B-E1) |
| KILL→deleted | 6 (EMC bind, EMC kill, row L bind, row L kill, "ChaCAL must emit chance", the regime-N survival route — the last is recorded in `sec_apparatus.md` §A.12 without a ledger row) |

The five first-evening items, in the coordinator's order: **P0.1** the `[M]` Lean targets and their refusals (`gamma_zero_is_softmax`, `lower_triangular_isUnit`, `resolvent_fromBlocks`, `segmentation_blockdiag` on `StrictlyLower`, `displacement_identity`, `bos_row_is_absorbing` with `β = 1` stated, `softmax_corner_not_nilpotent` with `0 < γ` and lower-triangularity stated, `later_boundary_unreachable`), 0 GPU-s; **P0.2** the extended manifest with its drift plant, 0 GPU-s; **P1.2** the BED-S generator spec with the DAG substrate, sink/goal/constraint placement and the census on 512 draws printing label sd, class frequencies, sink share, discard count — PASS if every sd `> 0.05`, every class frequency in `(0.05, 0.95)`, `0 ∈ 𝒜_sink` and every set before the query on `100 %`; **P2.1** the ChaCAL-diag smoke test at `γ = 0.9` on BED-M as the solve-path check (`≈ 1.7 s`); **P4.3** the `γ`-pinning LR-test instrument spec with the boundary null `2.706` and the held-out set declared. The `~6 GPU-s` capped run at seeds 2, 3, 7 that the record priced three times and never took (`READ V20_R15_IT35_JUPITER.md:157-159`) is the sixth, on BED-M, and is independent of every row above.
